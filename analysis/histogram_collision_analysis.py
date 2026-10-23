"""
Histogram Collision Analysis
=============================
Finds entities that are PROVABLY indistinguishable in histogram feature space
(identical rounded histogram vectors) but carry different ground-truth labels,
then checks whether the sequential (BiLSTM) character encoding correctly
separates them.

This directly probes the paper's hypothesis: a frequency histogram discards
character ORDER, so any two texts with the same character-type composition
collapse to the same point in histogram space regardless of label. A sequential
encoder can, in principle, tell them apart.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from collections import defaultdict

import numpy as np
import torch

from src.paths import CHECKPOINTS, OUTPUTS
from src.data.utils import get_histogram
from src.models.char_embedding.model import CharEmbeddingModule, AggregatorMethod

from advanced_separability_analysis import load_dataset_entities, compute_metrics


def build_trained_char_module(ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    trained_char_module = CharEmbeddingModule(
        use_embedding=True,
        aggregation_method=AggregatorMethod.BILSTM,
        char_embedding_dim=32,
        lstm_hidden_dim=32,
        num_lstm_layer=3,
        max_seq_length=34,
        device=device
    )
    char_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('char_embedding_module.'):
            char_state_dict[k.replace('char_embedding_module.', '')] = v
    trained_char_module.load_state_dict(char_state_dict, strict=True)
    trained_char_module.to(device)
    trained_char_module.eval()
    return trained_char_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FUNSD', choices=['FUNSD', 'TONDFU'])
    parser.add_argument('--checkpoint', type=str,
                         default='combined-emb-bilstm-all-1-bug-fix/e2e_char_embed-20250207-1519-3f4a86.pt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--round_decimals', type=int, default=2,
                         help="Round histogram values to N decimals before grouping collisions")
    parser.add_argument('--min_group_size', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    ckpt_path = CHECKPOINTS / args.checkpoint
    trained_char_module = build_trained_char_module(ckpt_path, device)

    entities = load_dataset_entities(args.dataset)
    texts = [e['text'] for e in entities]
    labels = np.array([e['label'] for e in entities])

    # Only consider non-trivial text (skip empty)
    valid_idx = [i for i, t in enumerate(texts) if t.strip() != '']
    texts = [texts[i] for i in valid_idx]
    labels = labels[valid_idx]

    hist_feats = np.array(get_histogram(texts), dtype=np.float32)
    hist_rounded = np.round(hist_feats, args.round_decimals)

    with torch.no_grad():
        pre_tr, len_tr = trained_char_module.preprocess(texts)
        bilstm_feats = trained_char_module(pre_tr, len_tr).cpu().numpy()

    # 5-fold CV predictions in each feature space (so predictions are held-out, not memorized)
    hist_metrics = compute_metrics(hist_feats, labels)
    bilstm_metrics = compute_metrics(bilstm_feats, labels)
    hist_pred = np.array(hist_metrics['y_pred'])
    bilstm_pred = np.array(bilstm_metrics['y_pred'])

    # Group entities by identical (rounded) histogram vector
    groups = defaultdict(list)
    for i, h in enumerate(hist_rounded):
        key = tuple(h.tolist())
        groups[key].append(i)

    # Keep only collision groups: >1 distinct TRUE label sharing the same histogram vector
    collision_groups = []
    for key, idxs in groups.items():
        group_labels = set(labels[i] for i in idxs)
        if len(group_labels) > 1 and len(idxs) >= args.min_group_size:
            collision_groups.append((key, idxs))

    # Sort by group size (largest, most "confusable" groups first)
    collision_groups.sort(key=lambda kv: -len(kv[1]))

    print(f"\n================ Histogram Collision Analysis ({args.dataset}) ================")
    print(f"Total valid entities: {len(texts)}")
    print(f"Total distinct rounded-histogram buckets: {len(groups)}")
    print(f"Buckets with >1 true label AND >= {args.min_group_size} members (collision groups): {len(collision_groups)}")

    total_entities_in_collisions = sum(len(idxs) for _, idxs in collision_groups)
    print(f"Total entities living inside a multi-label collision bucket: {total_entities_in_collisions} "
          f"({total_entities_in_collisions/len(texts)*100:.1f}% of dataset)")

    # For entities inside collision groups: how well does each feature space do?
    coll_idx_all = [i for _, idxs in collision_groups for i in idxs]
    coll_idx_all = np.array(sorted(set(coll_idx_all)))

    hist_acc_in_collisions = float(np.mean(hist_pred[coll_idx_all] == labels[coll_idx_all]))
    bilstm_acc_in_collisions = float(np.mean(bilstm_pred[coll_idx_all] == labels[coll_idx_all]))

    print(f"\n--- Accuracy restricted to entities inside collision buckets (n={len(coll_idx_all)}) ---")
    print(f"  Histogram accuracy : {hist_acc_in_collisions*100:.2f}%  (theoretical ceiling = majority-vote-per-bucket)")
    print(f"  BiLSTM accuracy    : {bilstm_acc_in_collisions*100:.2f}%")
    print(f"  Gain               : {(bilstm_acc_in_collisions-hist_acc_in_collisions)*100:+.2f} pts")

    # Compute theoretical ceiling for histogram: best possible accuracy if it always predicts
    # the majority true label within its own bucket (upper bound for ANY histogram-only classifier)
    ceiling_correct = 0
    for key, idxs in groups.items():
        group_true = labels[idxs]
        vals, counts = np.unique(group_true, return_counts=True)
        ceiling_correct += counts.max()
    hist_theoretical_ceiling = ceiling_correct / len(texts)
    print(f"\nTheoretical ceiling for ANY histogram-based classifier (majority vote per identical-histogram bucket): "
          f"{hist_theoretical_ceiling*100:.2f}%")

    # Build report of top collision groups with example texts + predictions
    report_groups = []
    for key, idxs in collision_groups[:40]:
        members = []
        for i in idxs:
            members.append({
                'text': texts[i],
                'true_label': labels[i],
                'hist_pred': hist_pred[i],
                'bilstm_pred': bilstm_pred[i],
                'hist_correct': bool(hist_pred[i] == labels[i]),
                'bilstm_correct': bool(bilstm_pred[i] == labels[i])
            })
        n_bilstm_correct = sum(m['bilstm_correct'] for m in members)
        n_hist_correct = sum(m['hist_correct'] for m in members)
        report_groups.append({
            'histogram_vector': list(key),
            'group_size': len(idxs),
            'distinct_true_labels': sorted(set(labels[i] for i in idxs)),
            'hist_correct_count': n_hist_correct,
            'bilstm_correct_count': n_bilstm_correct,
            'bilstm_advantage': n_bilstm_correct - n_hist_correct,
            'members': members
        })

    # Sort the report: prioritize groups where BiLSTM clearly wins (largest advantage), then by size
    report_groups.sort(key=lambda g: (-g['bilstm_advantage'], -g['group_size']))

    out_dir = OUTPUTS / 'feature_analysis'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'histogram_collision_analysis_{args.dataset.lower()}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': args.dataset,
            'total_entities': len(texts),
            'round_decimals': args.round_decimals,
            'min_group_size': args.min_group_size,
            'num_collision_groups': len(collision_groups),
            'entities_in_collision_groups': int(len(coll_idx_all)),
            'hist_accuracy_in_collisions': hist_acc_in_collisions,
            'bilstm_accuracy_in_collisions': bilstm_acc_in_collisions,
            'hist_theoretical_ceiling_dataset_wide': hist_theoretical_ceiling,
            'top_collision_groups': report_groups
        }, f, indent=2, default=str)
    print(f"\nSaved detailed collision report to {out_path}")

    # Print the most compelling examples to console
    print("\n================ Top Collision Groups (BiLSTM correctly separates, Histogram cannot) ================")
    shown = 0
    for g in report_groups:
        if g['bilstm_advantage'] <= 0:
            continue
        print(f"\nHistogram vector {tuple(round(v,2) for v in g['histogram_vector'])}  "
              f"| group size={g['group_size']}  | true labels={g['distinct_true_labels']}  "
              f"| Hist correct={g['hist_correct_count']}/{g['group_size']}  BiLSTM correct={g['bilstm_correct_count']}/{g['group_size']}")
        for m in g['members'][:8]:
            flag_h = "OK" if m['hist_correct'] else "XX"
            flag_b = "OK" if m['bilstm_correct'] else "XX"
            print(f"    text={m['text']!r:35s} true={m['true_label']:10s} "
                  f"hist_pred={m['hist_pred']:10s}[{flag_h}]  bilstm_pred={m['bilstm_pred']:10s}[{flag_b}]")
        shown += 1
        if shown >= 15:
            break


if __name__ == '__main__':
    main()
