"""
Run one TONDFU checkpoint on the TONDFU test set and dump everything needed
for error analysis (per document, per node, per link), not just the flat
prediction lists that src/training/tondfu.py writes.

Outputs (in outputs/tondfu_error_analysis/<weight-stem>/):
    metrics.json  overall + per-document scores (same metrics as the paper)
    nodes.json    every entity: doc, id, text, box, true/pred class, class probs
    links.json    every edge that is a true link OR a predicted link, with TP/FP/FN tag
    run.log       stdout of this run

configs/preprocessing.yaml is overwritten by the feature flags (like main.py does)
and restored afterwards.

Usage (from repo root):
    python analysis/tondfu_eval_dump.py \
        --weights tondfu-with-pretrained-combined/e2e_char_embed-20250508-0136-6d89b3.pt
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import CHECKPOINTS, CONFIGS, CFGM, OUTPUTS, TONDFU_TEST, TEST_SAMPLES
from src.utils import get_config, set_preprocessing
from src.models.graphs import SetModel
from src.training.utils import get_device, load_checkpoint_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='path relative to src/models/checkpoints')
    p.add_argument('--model', default='e2e_char_embed')
    p.add_argument('--gpu', type=int, default=-1)
    p.add_argument('--out', default=None, help='output dir (default: outputs/tondfu_error_analysis/<weight-stem>)')
    # feature flags of "Ours + mBERT" (paper Table: pretrained, TONDFU)
    p.add_argument('--add-geom', action='store_true', default=True)
    p.add_argument('--add-visual', action='store_true', default=True)
    p.add_argument('--add-eweights', action='store_true', default=True)
    p.add_argument('--add-mbert', action='store_true', default=True)
    p.add_argument('--add-embs', action='store_true', default=False)
    p.add_argument('--add-hist', action='store_true', default=False)
    args = p.parse_args()
    # fields expected by set_preprocessing
    args.src_data = 'TONDFU'
    args.data_type = 'img'
    args.edge_type = 'fully'
    args.node_granularity = 'gt'
    args.num_polar_bins = 8
    return args


def pair_f1(labels, preds):
    """Positive-class ('pair') F1, as reported in the paper."""
    return precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1], zero_division=0)[2][1]


def main():
    args = parse_args()
    out_dir = Path(args.out) if args.out else OUTPUTS / 'tondfu_error_analysis' / Path(args.weights).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_cfg = CONFIGS / 'preprocessing.yaml'
    prep_backup = out_dir / 'preprocessing.yaml.bak'
    shutil.copy(prep_cfg, prep_backup)
    try:
        set_preprocessing(args)
        run(args, out_dir)
    finally:
        shutil.copy(prep_backup, prep_cfg)
        prep_backup.unlink()


def run(args, out_dir):
    # imported here: Document2Graph reads preprocessing.yaml at construction
    from src.data.dataloader import Document2Graph

    device = get_device(args.gpu)
    data = Document2Graph(name='TONDFU TEST', src_path=TONDFU_TEST, device=device, output_dir=TEST_SAMPLES)
    node_classes = [str(c) for c in data.node_unique_labels]
    edge_classes = [str(c) for c in data.edge_unique_labels]
    assert edge_classes == ['none', 'pair'], edge_classes

    model_cfg = get_config(CFGM / args.model)
    load_checkpoint_config(args.weights, model_cfg)
    sm = SetModel(name=args.model, device=device)
    model = sm.get_model(data.node_num_classes, data.edge_num_classes, data.get_chunks())
    model.load_state_dict(torch.load(CHECKPOINTS / args.weights, map_location=device))
    model.eval()

    nodes_out, links_out, per_doc = [], [], {}
    all_n_lab, all_n_pred, all_e_lab, all_e_pred = [], [], [], []

    with torch.no_grad():
        for g, texts, boxes, path in zip(data.graphs, data.texts, data.boxes, data.paths):
            doc = Path(path).stem
            g = g.to(device)
            n_scores, e_scores = model(g, g.ndata['feat'].to(device), list(texts))

            n_prob = F.softmax(n_scores, dim=1).cpu()
            n_pred = n_prob.argmax(dim=1)
            n_lab = g.ndata['label'].cpu()
            e_prob = F.softmax(e_scores, dim=1).cpu()[:, 1]
            e_pred = e_scores.argmax(dim=1).cpu()
            e_lab = g.edata['label'].cpu()
            src, dst = (t.cpu() for t in g.edges())

            all_n_lab.append(n_lab); all_n_pred.append(n_pred)
            all_e_lab.append(e_lab); all_e_pred.append(e_pred)

            per_doc[doc] = {
                'num_entities': len(texts),
                'num_true_links': int(e_lab.sum()),
                'num_pred_links': int(e_pred.sum()),
                'labeling_micro_f1': f1_score(n_lab, n_pred, average='micro'),
                'linking_pair_f1': pair_f1(e_lab, e_pred),
                'TP': int(((e_pred == 1) & (e_lab == 1)).sum()),
                'FP': int(((e_pred == 1) & (e_lab == 0)).sum()),
                'FN': int(((e_pred == 0) & (e_lab == 1)).sum()),
            }

            for i in range(len(texts)):
                nodes_out.append({
                    'doc': doc, 'node': i, 'text': texts[i], 'box': boxes[i],
                    'true': node_classes[n_lab[i]], 'pred': node_classes[n_pred[i]],
                    'probs': {c: round(float(p), 4) for c, p in zip(node_classes, n_prob[i])},
                })

            keep = ((e_lab == 1) | (e_pred == 1)).nonzero().flatten().tolist()
            for k in keep:
                s, d = int(src[k]), int(dst[k])
                tp, lab = int(e_pred[k]), int(e_lab[k])
                links_out.append({
                    'doc': doc, 'src': s, 'dst': d,
                    'status': 'TP' if tp and lab else ('FP' if tp else 'FN'),
                    'pair_prob': round(float(e_prob[k]), 4),
                    'src_text': texts[s], 'dst_text': texts[d],
                    'src_box': boxes[s], 'dst_box': boxes[d],
                    'src_true': node_classes[n_lab[s]], 'dst_true': node_classes[n_lab[d]],
                    'src_pred': node_classes[n_pred[s]], 'dst_pred': node_classes[n_pred[d]],
                })

    n_lab, n_pred = torch.cat(all_n_lab), torch.cat(all_n_pred)
    e_lab, e_pred = torch.cat(all_e_lab), torch.cat(all_e_pred)
    metrics = {
        'weights': args.weights,
        'node_classes': node_classes,
        'overall': {
            'labeling_micro_f1': f1_score(n_lab, n_pred, average='micro'),
            'labeling_macro_f1': f1_score(n_lab, n_pred, average='macro'),
            'labeling_per_class_f1': dict(zip(node_classes, f1_score(n_lab, n_pred, average=None).tolist())),
            'linking_pair_f1': pair_f1(e_lab, e_pred),
            'TP': int(((e_pred == 1) & (e_lab == 1)).sum()),
            'FP': int(((e_pred == 1) & (e_lab == 0)).sum()),
            'FN': int(((e_pred == 0) & (e_lab == 1)).sum()),
        },
        'per_doc': per_doc,
    }

    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / 'nodes.json').write_text(json.dumps(nodes_out, indent=1, ensure_ascii=False))
    (out_dir / 'links.json').write_text(json.dumps(links_out, indent=1, ensure_ascii=False))
    print(json.dumps(metrics['overall'], indent=2))
    print(f'-> saved to {out_dir}')


if __name__ == '__main__':
    main()
