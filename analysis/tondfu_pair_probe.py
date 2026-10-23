"""
Experiment A: where does the entity-linking signal live?

For every candidate pair (graph edge) of the TONDFU test set, capture the pair representation at
each stage of the model and probe how linearly separable "link" vs "no link" is at that stage:

    polar      raw edge feature only (polar bins + distance), no node content
    raw-<mod>  raw node features before projection, per modality (char / geom / visual / mbert)
    raw-all    all raw node modalities
    proj       after InputProjector, BEFORE message passing
    gnn        after the GraphSAGE message passing layer
    edge-in    exactly what MLPPredictor_E2E sees: [h_u, cls_u, polar, h_v, cls_v]

Pair features are [f_src, f_dst, |f_src - f_dst|] (plus the polar feature where stated).
Probe: logistic regression, leave-one-document-out over the 7 test docs, scored by average
precision. Negatives are subsampled (--neg-ratio per positive) with a fixed seed.

Reported overall, and split by link orientation (H/V) and distance band, so we can see whether
the stages lose the long-range horizontal links specifically.

Usage (from repo root):
    python analysis/tondfu_pair_probe.py \
        --weights tondfu-with-pretrained-combined/e2e_char_embed-20250508-0136-6d89b3.pt
"""
import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import CHECKPOINTS, CONFIGS, CFGM, OUTPUTS, TONDFU_TEST, TEST_SAMPLES
from src.utils import get_config, set_preprocessing
from src.models.graphs import SetModel
from src.training.utils import get_device, load_checkpoint_config

DOC_TYPE = {'IMG_2701': 'water', 'IMG_2703': 'water', 'IMG_2642': 'elec', 'IMG_2643': 'elec',
            'IMG_8153': 'veh', 'IMG_9159': 'veh', 'S__25436307': 'veh'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--model', default='e2e_char_embed')
    p.add_argument('--gpu', type=int, default=-1)
    p.add_argument('--neg-ratio', type=int, default=20, help='sampled negatives per positive')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    args.src_data, args.data_type, args.edge_type = 'TONDFU', 'img', 'fully'
    args.node_granularity, args.num_polar_bins = 'gt', 8
    args.add_geom = args.add_visual = args.add_eweights = args.add_mbert = True
    args.add_embs = args.add_hist = False
    return args


def geometry(src_box, dst_box):
    """orientation, centre distance and off-axis misalignment, in source box heights."""
    cx = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    (x1, y1), (x2, y2) = cx(src_box), cx(dst_box)
    dx, dy = x2 - x1, y2 - y1
    h = max(src_box[3] - src_box[1], 1)
    ori = 'H' if abs(dx) > abs(dy) else 'V'
    return ori, math.hypot(dx, dy) / h, (abs(dy) if ori == 'H' else abs(dx)) / h


def collect(args):
    """Run the model once over the test set, capturing per-stage node reps and pair metadata."""
    from src.data.dataloader import Document2Graph

    device = get_device(args.gpu)
    data = Document2Graph(name='TONDFU TEST', src_path=TONDFU_TEST, device=device, output_dir=TEST_SAMPLES)

    model_cfg = get_config(CFGM / args.model)
    load_checkpoint_config(args.weights, model_cfg)
    sm = SetModel(name=args.model, device=device, )
    model = sm.get_model(data.node_num_classes, data.edge_num_classes, data.get_chunks(), verbatim=False)
    model.load_state_dict(torch.load(CHECKPOINTS / args.weights, map_location=device))
    model.eval()

    cap = {}
    model.projector.register_forward_hook(lambda m, i, o: cap.update(raw=i[0].detach(), proj=o.detach()))
    model.message_passing.register_forward_hook(lambda m, i, o: cap.update(gnn=o.detach()))

    # modality chunk sizes, in the order the projector concatenates them
    chunks = list(model.projector.chunks)
    if chunks and chunks[0] == 0:
        chunks = chunks[1:]
    # chunk order: [char] + whatever FeatureBuilder added (geom, visual, mbert)
    known = {model.char_embedding_module.output_dim: 'char', 768: 'mbert'}
    names = [known.get(c, 'geom' if c <= 16 else 'visual') for c in chunks]
    names = [f'{n}{i}' if names.count(n) > 1 else n for i, n in enumerate(names)]
    print('-> modality chunks:', dict(zip(names, chunks)))
    spans, start = {}, 0
    for name, size in zip(names, chunks):
        spans[name] = (start, start + size)
        start += size

    rows = []
    with torch.no_grad():
        for g, texts, boxes, path in zip(data.graphs, data.texts, data.boxes, data.paths):
            doc = Path(path).stem
            g = g.to(device)
            n_scores, e_scores = model(g, g.ndata['feat'].to(device), list(texts))
            cls = torch.softmax(n_scores.detach(), dim=1).cpu().numpy()
            src, dst = (t.cpu().numpy() for t in g.edges())
            stage_feats = {k: v.cpu().numpy() for k, v in cap.items()}
            rows.append({
                'doc': doc, 'src': src, 'dst': dst,
                'label': g.edata['label'].cpu().numpy(),
                'polar': g.edata['feat'].cpu().numpy(),
                'pred': e_scores.argmax(dim=1).cpu().numpy(),
                'cls': cls, 'stages': stage_feats, 'boxes': boxes,
            })
    return rows, spans


def build_matrix(doc, idx, node_feats, polar=None, cls=None):
    """pair features: [f_src, f_dst, |diff|] (+ polar, + class probs)."""
    s, d = doc['src'][idx], doc['dst'][idx]
    fs, fd = node_feats[s], node_feats[d]
    parts = [fs, fd, np.abs(fs - fd)]
    if cls is not None:
        parts += [doc['cls'][s], doc['cls'][d]]
    if polar is not None:
        parts.append(polar[idx])
    return np.hstack(parts).astype(np.float32)


def main():
    args = parse_args()
    out_dir = Path(args.out) if args.out else OUTPUTS / 'tondfu_error_analysis' / Path(args.weights).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_cfg, backup = CONFIGS / 'preprocessing.yaml', out_dir / 'prep.bak'
    shutil.copy(prep_cfg, backup)
    try:
        set_preprocessing(args)
        rows, spans = collect(args)
    finally:
        shutil.copy(backup, prep_cfg)
        backup.unlink()

    rng = np.random.default_rng(args.seed)
    stage_defs = {'polar': None, 'raw-all': 'raw', 'proj': 'proj', 'gnn': 'gnn', 'edge-in': 'gnn'}
    for mod in spans:
        stage_defs[f'raw-{mod}'] = 'raw'

    X = defaultdict(list)
    y, groups, meta = [], [], []
    for doc in rows:
        pos = np.flatnonzero(doc['label'] == 1)
        neg_pool = np.flatnonzero(doc['label'] == 0)
        neg = rng.choice(neg_pool, size=min(len(neg_pool), len(pos) * args.neg_ratio), replace=False)
        idx = np.concatenate([pos, neg])
        for stage, source in stage_defs.items():
            if stage == 'polar':
                feats = doc['polar'][idx]
            else:
                nf = doc['stages'][source]
                if stage.startswith('raw-') and stage != 'raw-all':
                    a, b = spans[stage.split('-', 1)[1]]
                    nf = nf[:, a:b]
                feats = np.stack([build_matrix(doc, i, nf,
                                               polar=doc['polar'] if stage == 'edge-in' else None,
                                               cls=doc['cls'] if stage == 'edge-in' else None)
                                  for i in idx])
            X[stage].append(feats)
        y.append(doc['label'][idx])
        groups += [doc['doc']] * len(idx)
        for i in idx:
            ori, dist, mis = geometry(doc['boxes'][doc['src'][i]], doc['boxes'][doc['dst'][i]])
            meta.append({'doc': doc['doc'], 'type': DOC_TYPE.get(doc['doc'], '?'),
                         'ori': ori, 'dist': dist, 'mis': mis, 'model_pred': int(doc['pred'][i])})

    X = {k: np.vstack(v) for k, v in X.items()}
    y = np.concatenate(y)
    groups = np.array(groups)
    meta = {k: np.array([m[k] for m in meta]) for k in meta[0]}
    band = np.where(meta['dist'] < 3, 'near(<3h)', np.where(meta['dist'] < 8, 'mid(3-8h)', 'far(>8h)'))

    print(f'\npairs: {len(y)} ({y.sum()} links), stages: {list(X)}')
    results = {}
    for stage, mat in X.items():
        scores = np.zeros(len(y))
        for doc in np.unique(groups):  # leave-one-document-out
            tr, te = groups != doc, groups == doc
            sc = StandardScaler().fit(mat[tr])
            clf = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf.fit(sc.transform(mat[tr]), y[tr])
            scores[te] = clf.predict_proba(sc.transform(mat[te]))[:, 1]
        r = {'dim': int(mat.shape[1]), 'AP_all': float(average_precision_score(y, scores))}
        for key, sel in [('H', meta['ori'] == 'H'), ('V', meta['ori'] == 'V')]:
            r[f'AP_{key}'] = float(average_precision_score(y[sel], scores[sel]))
        for b in ['near(<3h)', 'mid(3-8h)', 'far(>8h)']:
            sel = band == b
            if y[sel].sum() > 2:
                r[f'AP_{b}'] = float(average_precision_score(y[sel], scores[sel]))
        for t in ['water', 'elec', 'veh']:
            sel = meta['type'] == t
            r[f'AP_{t}'] = float(average_precision_score(y[sel], scores[sel]))
        results[stage] = r
        print(f"{stage:12s} d={r['dim']:5d} AP={r['AP_all']:.3f} | H {r['AP_H']:.3f} V {r['AP_V']:.3f} | "
              + ' '.join(f"{b.split('(')[0]} {r.get('AP_'+b, float('nan')):.3f}"
                         for b in ['near(<3h)', 'mid(3-8h)', 'far(>8h)'])
              + f" | water {r['AP_water']:.3f} elec {r['AP_elec']:.3f} veh {r['AP_veh']:.3f}")

    # the model itself, on the same sampled pairs, as the reference point
    model_ap = float(average_precision_score(y, meta['model_pred']))
    results['_model_argmax_on_sampled_pairs'] = {'AP_all': model_ap}
    print(f'\nmodel (argmax) on the same pairs: AP={model_ap:.3f}')

    out = out_dir / 'pair_probe.json'
    out.write_text(json.dumps({'config': {'neg_ratio': args.neg_ratio, 'seed': args.seed,
                                          'weights': args.weights}, 'results': results}, indent=2))
    print(f'-> saved {out}')


if __name__ == '__main__':
    main()
