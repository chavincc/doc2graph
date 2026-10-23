"""
Compares the ORIGINAL (buggy) histogram vs the FIXED histogram (Thai
combining marks folded into alpha) against the untouched, already-trained
BiLSTM, on both datasets. No retraining -- BiLSTM features are extracted
using the model exactly as checkpointed.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import numpy as np
import torch
from collections import defaultdict

from src.paths import CHECKPOINTS, OUTPUTS
from src.data.utils import get_histogram
from advanced_separability_analysis import load_dataset_entities, compute_metrics
from histogram_collision_analysis import build_trained_char_module
from fixed_histogram import get_histogram_fixed


def local_ceiling(hist_vecs, labels, round_decimals=2, min_group_size=3):
    hist_rounded = np.round(hist_vecs, round_decimals)
    groups = defaultdict(list)
    for i, h in enumerate(hist_rounded):
        groups[tuple(h.tolist())].append(i)

    coll_idx = []
    ceiling_correct = 0
    n_collision_groups = 0
    for key, idxs in groups.items():
        gl = labels[idxs]
        vals, counts = np.unique(gl, return_counts=True)
        if len(vals) > 1 and len(idxs) >= min_group_size:
            coll_idx.extend(idxs)
            ceiling_correct += counts.max()
            n_collision_groups += 1
    coll_idx = np.array(sorted(set(coll_idx)))
    if len(coll_idx) == 0:
        return 0, 0.0, 0
    return len(coll_idx), ceiling_correct / len(coll_idx) * 100, n_collision_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                         default='combined-emb-bilstm-all-1-bug-fix/e2e_char_embed-20250207-1519-3f4a86.pt')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    ckpt_path = CHECKPOINTS / args.checkpoint
    trained_char_module = build_trained_char_module(ckpt_path, device)

    all_results = {}

    for dataset_name in ['FUNSD', 'TONDFU']:
        entities = load_dataset_entities(dataset_name)
        texts = [e['text'] for e in entities]
        labels = np.array([e['label'] for e in entities])
        valid_idx = [i for i, t in enumerate(texts) if t.strip() != '']
        texts = [texts[i] for i in valid_idx]
        labels = labels[valid_idx]

        hist_old = np.array(get_histogram(texts), dtype=np.float32)
        hist_new = np.array(get_histogram_fixed(texts), dtype=np.float32)

        # Sanity: how many entities actually changed?
        n_changed = int(np.sum(np.any(np.round(hist_old, 4) != np.round(hist_new, 4), axis=1)))

        with torch.no_grad():
            pre_tr, len_tr = trained_char_module.preprocess(texts)
            bilstm_feats = trained_char_module(pre_tr, len_tr).cpu().numpy()

        m_old = compute_metrics(hist_old, labels)
        m_new = compute_metrics(hist_new, labels)
        m_bilstm = compute_metrics(bilstm_feats, labels)

        n_coll_old, ceil_old, ngroups_old = local_ceiling(hist_old, labels)
        n_coll_new, ceil_new, ngroups_new = local_ceiling(hist_new, labels)

        print(f"\n================ {dataset_name} ================")
        print(f"Entities with changed histogram vector due to fix: {n_changed} / {len(texts)}")
        print(f"\n{'Metric':<28}{'Hist (OLD)':<15}{'Hist (FIXED)':<15}{'BiLSTM (as-trained)':<20}")
        print(f"{'Silhouette':<28}{m_old['silhouette']:<15.4f}{m_new['silhouette']:<15.4f}{m_bilstm['silhouette']:<20.4f}")
        print(f"{'Davies-Bouldin':<28}{m_old['davies_bouldin']:<15.4f}{m_new['davies_bouldin']:<15.4f}{m_bilstm['davies_bouldin']:<20.4f}")
        print(f"{'1-NN Accuracy':<28}{m_old['knn_1_accuracy']*100:<15.2f}{m_new['knn_1_accuracy']*100:<15.2f}{m_bilstm['knn_1_accuracy']*100:<20.2f}")
        print(f"{'Linear Probe Accuracy':<28}{m_old['linear_probe_accuracy']*100:<15.2f}{m_new['linear_probe_accuracy']*100:<15.2f}{m_bilstm['linear_probe_accuracy']*100:<20.2f}")

        print(f"\nCollision-bucket analysis (identical rounded histogram, mixed true labels):")
        print(f"  OLD histogram: {n_coll_old} entities in collision buckets, local majority-vote ceiling = {ceil_old:.2f}%")
        print(f"  NEW histogram: {n_coll_new} entities in collision buckets, local majority-vote ceiling = {ceil_new:.2f}%")
        print(f"  BiLSTM (as-trained) linear probe accuracy overall: {m_bilstm['linear_probe_accuracy']*100:.2f}%")

        all_results[dataset_name] = {
            'n_entities': len(texts),
            'n_changed_by_fix': n_changed,
            'hist_old': {'silhouette': m_old['silhouette'], 'davies_bouldin': m_old['davies_bouldin'],
                         'knn_1_accuracy': m_old['knn_1_accuracy'], 'linear_probe_accuracy': m_old['linear_probe_accuracy']},
            'hist_fixed': {'silhouette': m_new['silhouette'], 'davies_bouldin': m_new['davies_bouldin'],
                           'knn_1_accuracy': m_new['knn_1_accuracy'], 'linear_probe_accuracy': m_new['linear_probe_accuracy']},
            'bilstm_as_trained': {'silhouette': m_bilstm['silhouette'], 'davies_bouldin': m_bilstm['davies_bouldin'],
                                  'knn_1_accuracy': m_bilstm['knn_1_accuracy'], 'linear_probe_accuracy': m_bilstm['linear_probe_accuracy']},
            'collision_old': {'n_in_collision': n_coll_old, 'local_ceiling_pct': ceil_old, 'n_groups': ngroups_old},
            'collision_fixed': {'n_in_collision': n_coll_new, 'local_ceiling_pct': ceil_new, 'n_groups': ngroups_new},
        }

    out_path = OUTPUTS / 'feature_analysis' / 'histogram_bugfix_comparison.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved comparison JSON to {out_path}")


if __name__ == '__main__':
    main()
