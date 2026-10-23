"""
Pattern Subset Analysis
========================
Zooms into specific recurring TEXTUAL PATTERNS that share identical (or near
identical) histogram vectors but are correctly disambiguated by the sequential
BiLSTM encoder. Used to build concrete, explainable qualitative examples for
the paper (e.g. "name-initials pattern", "checkbox/mark symbols", "pure numeric
strings of varying length").
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import argparse
import numpy as np
import torch

from src.paths import CHECKPOINTS, OUTPUTS
from src.data.utils import get_histogram
from advanced_separability_analysis import load_dataset_entities, compute_metrics
from histogram_collision_analysis import build_trained_char_module

PATTERNS = {
    'name_initials': re.compile(r'^[A-Z]\.\s*[A-Z]\.\s+[A-Za-z]'),           # "K. A. Sparrow"
    'pure_digits': re.compile(r'^\d+$'),                                     # "3", "82092117"
    'single_symbol': re.compile(r'^[^\w\s]$'),                               # ":" "?" "☑"
    'date_like': re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'),
    'colon_terminated_word': re.compile(r'^[A-Za-z][A-Za-z .]*:$'),          # "BRAND:" "SUBJECT:"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FUNSD', choices=['FUNSD', 'TONDFU'])
    parser.add_argument('--checkpoint', type=str,
                         default='combined-emb-bilstm-all-1-bug-fix/e2e_char_embed-20250207-1519-3f4a86.pt')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    ckpt_path = CHECKPOINTS / args.checkpoint
    trained_char_module = build_trained_char_module(ckpt_path, device)

    entities = load_dataset_entities(args.dataset)
    texts = [e['text'] for e in entities]
    labels = np.array([e['label'] for e in entities])
    valid_idx = [i for i, t in enumerate(texts) if t.strip() != '']
    texts = [texts[i] for i in valid_idx]
    labels = labels[valid_idx]

    hist_feats = np.array(get_histogram(texts), dtype=np.float32)
    with torch.no_grad():
        pre_tr, len_tr = trained_char_module.preprocess(texts)
        bilstm_feats = trained_char_module(pre_tr, len_tr).cpu().numpy()

    hist_metrics = compute_metrics(hist_feats, labels)
    bilstm_metrics = compute_metrics(bilstm_feats, labels)
    hist_pred = np.array(hist_metrics['y_pred'])
    bilstm_pred = np.array(bilstm_metrics['y_pred'])

    print(f"\n================ Pattern Subset Analysis ({args.dataset}) ================\n")

    report = {}
    for pat_name, pat in PATTERNS.items():
        idxs = [i for i, t in enumerate(texts) if pat.match(t.strip())]
        if len(idxs) == 0:
            continue
        idxs = np.array(idxs)
        sub_labels = labels[idxs]
        label_dist = {l: int(c) for l, c in zip(*np.unique(sub_labels, return_counts=True))}
        h_acc = float(np.mean(hist_pred[idxs] == labels[idxs]))
        b_acc = float(np.mean(bilstm_pred[idxs] == labels[idxs]))

        print(f"--- Pattern: {pat_name}  (n={len(idxs)}) ---")
        print(f"  True label distribution : {label_dist}")
        print(f"  Histogram accuracy       : {h_acc*100:.2f}%")
        print(f"  BiLSTM accuracy          : {b_acc*100:.2f}%")
        print(f"  Gain                     : {(b_acc-h_acc)*100:+.2f} pts")

        examples = []
        for i in idxs[:12]:
            examples.append({
                'text': texts[i],
                'true_label': labels[i],
                'hist_pred': hist_pred[i],
                'bilstm_pred': bilstm_pred[i]
            })
            flag_h = "OK" if hist_pred[i] == labels[i] else "XX"
            flag_b = "OK" if bilstm_pred[i] == labels[i] else "XX"
            print(f"    {texts[i]!r:35s} true={labels[i]:10s} hist={hist_pred[i]:10s}[{flag_h}]  bilstm={bilstm_pred[i]:10s}[{flag_b}]")
        print()

        report[pat_name] = {
            'n': len(idxs),
            'label_distribution': label_dist,
            'hist_accuracy': h_acc,
            'bilstm_accuracy': b_acc,
            'gain': b_acc - h_acc,
            'examples': examples
        }

    out_dir = OUTPUTS / 'feature_analysis'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'pattern_subset_analysis_{args.dataset.lower()}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved pattern subset report to {out_path}")


if __name__ == '__main__':
    main()
