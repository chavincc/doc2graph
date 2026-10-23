import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import glob
import argparse
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.paths import ROOT, DATA, CHECKPOINTS, OUTPUTS, FUNSD_TEST
from src.data.utils import get_histogram
from src.models.char_embedding.model import CharEmbeddingModule, AggregatorMethod
from src.data.dataloader import Document2Graph
from src.models.graphs import SetModel


def load_dataset_entities(dataset_name: str, data_split: str = 'testing_data'):
    """Load text and node label for each entity in the dataset directly from annotation files."""
    anno_dir = DATA / dataset_name / data_split / 'adjusted_annotations'
    json_files = sorted(glob.glob(os.path.join(anno_dir, '*.json')))
    
    entities = []
    for filepath in json_files:
        doc_id = os.path.basename(filepath).replace('.json', '')
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for elem in data.get('form', []):
            text = elem.get('text', '').strip()
            label = elem.get('label', 'other')
            entities.append({
                'doc_id': doc_id,
                'id': elem.get('id'),
                'text': text,
                'label': label
            })
    return entities


def extract_pre_gnn_features(entities: list, trained_char_module: CharEmbeddingModule, device: torch.device):
    """Extract Histogram, One-Hot+AVG, Untrained BiLSTM, Trained BiLSTM, and Concat features."""
    texts = [e['text'] for e in entities]
    labels = [e['label'] for e in entities]
    
    # 1. Histogram (4D: %alpha, %digit, %symbol, %unknown)
    hist_feats = np.array(get_histogram(texts), dtype=np.float32)
    
    # 2. One-Hot + AVG (4D: %alpha, %digit, %symbol, %space)
    avg_module = CharEmbeddingModule(
        use_embedding=False,
        aggregation_method=AggregatorMethod.AVG,
        device=device
    )
    avg_module.eval()
    with torch.no_grad():
        pre_avg, len_avg = avg_module.preprocess(texts)
        avg_feats = avg_module(pre_avg, len_avg).cpu().numpy()
        
    # 3. Untrained BiLSTM (64D)
    random_bilstm = CharEmbeddingModule(
        use_embedding=True,
        aggregation_method=AggregatorMethod.BILSTM,
        char_embedding_dim=32,
        lstm_hidden_dim=32,
        num_lstm_layer=3,
        max_seq_length=34,
        device=device
    )
    random_bilstm.eval()
    with torch.no_grad():
        pre_rand, len_rand = random_bilstm.preprocess(texts)
        untrained_bilstm_feats = random_bilstm(pre_rand, len_rand).cpu().numpy()
        
    # 4. Trained BiLSTM (64D)
    trained_char_module.eval()
    with torch.no_grad():
        pre_tr, len_tr = trained_char_module.preprocess(texts)
        trained_bilstm_feats = trained_char_module(pre_tr, len_tr).cpu().numpy()
        
    # 5. Histogram + Trained BiLSTM (68D)
    concat_feats = np.hstack([hist_feats, trained_bilstm_feats])
    
    return {
        'texts': texts,
        'labels': labels,
        'Histogram (4D)': hist_feats,
        'One-Hot + AVG (4D)': avg_feats,
        'Untrained BiLSTM (64D)': untrained_bilstm_feats,
        'Trained BiLSTM (64D)': trained_bilstm_feats,
        'Histogram + Trained BiLSTM (68D)': concat_feats
    }


def compute_metrics(X: np.ndarray, y: np.ndarray):
    """Compute clustering separability and classification metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sil = float(silhouette_score(X_scaled, y))
    db = float(davies_bouldin_score(X_scaled, y))
    ch = float(calinski_harabasz_score(X_scaled, y))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn_scores = cross_val_score(knn, X_scaled, y, cv=cv)
    knn_acc = float(np.mean(knn_scores))
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(clf, X_scaled, y, cv=cv)
    lr_acc = float(np.mean(lr_scores))
    
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    classes = sorted(list(set(y)))
    cm = confusion_matrix(y, y_pred, labels=classes)
    
    return {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
        'knn_1_accuracy': knn_acc,
        'linear_probe_accuracy': lr_acc,
        'confusion_matrix': cm.tolist(),
        'classes': classes,
        'y_pred': y_pred.tolist()
    }


def run_post_gnn_experiment(ckpt_path: str, device: torch.device):
    """Extract representations before and after GNN message passing on FUNSD test set."""
    print("\n================ Running Post-GNN Analysis on FUNSD ================")
    print("Loading test dataset graph structures (with multimodal features)...")
    dataset = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device='cpu', output_dir=None)
    
    sm = SetModel(name='e2e_char_embed', device='cpu')
    model = sm.get_model(dataset.node_num_classes, dataset.edge_num_classes, dataset.get_chunks(), verbatim=False)
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    node_labels_map = dataset.node_unique_labels
    
    all_proj_feats = []
    all_gnn_feats = []
    all_logits_feats = []
    all_true_labels = []
    all_node_texts = []
    
    with torch.no_grad():
        for i in range(len(dataset.graphs)):
            g = dataset.graphs[i]
            texts = dataset.texts[i]
            h = g.ndata['feat']
            labels_idx = g.ndata['label'].numpy()
            true_labels = [node_labels_map[idx] for idx in labels_idx]
            
            # Forward pass up to projector & GNN
            if not model.use_baseline_only:
                pre_tensor, seq_lengths = model.char_embedding_module.preprocess(texts)
                text_features = model.char_embedding_module(pre_tensor, seq_lengths)
                if h is not None and h.shape[1] > 0:
                    h_cat = torch.cat((text_features, h), dim=1)
                else:
                    h_cat = text_features
            else:
                h_cat = h
                
            h_proj = model.projector(h_cat)
            h_gnn = model.message_passing(g, h_proj)
            logits = model.node_pred(h_gnn)
            
            all_proj_feats.append(h_proj.cpu().numpy())
            all_gnn_feats.append(h_gnn.cpu().numpy())
            all_logits_feats.append(logits.cpu().numpy())
            all_true_labels.extend(true_labels)
            all_node_texts.extend(texts)
            
    X_proj = np.vstack(all_proj_feats)
    X_gnn = np.vstack(all_gnn_feats)
    X_logits = np.vstack(all_logits_feats)
    y_true = np.array(all_true_labels)
    
    print(f"Total graph nodes evaluated: {len(y_true)}")
    
    metrics_proj = compute_metrics(X_proj, y_true)
    metrics_gnn = compute_metrics(X_gnn, y_true)
    metrics_logits = compute_metrics(X_logits, y_true)
    
    # Direct GNN node classification accuracy (argmax of logits)
    preds_gnn_direct = np.argmax(X_logits, axis=1)
    direct_classes = sorted(list(set(y_true)))
    direct_acc = float(np.mean([direct_classes[preds_gnn_direct[idx]] == y_true[idx] for idx in range(len(y_true))]))
    
    print(f"--- Post-Projector (Pre-GNN, Multimodal) ---")
    print(f"  Linear Probe Acc: {metrics_proj['linear_probe_accuracy']*100:.2f}% | 1-NN Acc: {metrics_proj['knn_1_accuracy']*100:.2f}% | Silhouette: {metrics_proj['silhouette']:.4f}")
    
    print(f"--- Post-GNN Message Passing ---")
    print(f"  Linear Probe Acc: {metrics_gnn['linear_probe_accuracy']*100:.2f}% | 1-NN Acc: {metrics_gnn['knn_1_accuracy']*100:.2f}% | Silhouette: {metrics_gnn['silhouette']:.4f}")
    
    print(f"--- Final Node Classifier (End-to-End Direct Acc) ---")
    print(f"  Direct Model Accuracy: {direct_acc*100:.2f}%")
    
    return {
        'Post-Projector (Pre-GNN)': metrics_proj,
        'Post-GNN Message Passing': metrics_gnn,
        'Final Logits': metrics_logits,
        'direct_model_accuracy': direct_acc,
        'y_true': y_true.tolist(),
        'y_pred_direct': [direct_classes[p] for p in preds_gnn_direct],
        'node_texts': all_node_texts
    }


def perform_qualitative_error_analysis(feats_dict: dict, metrics_dict: dict, post_gnn_res: dict = None):
    """Categorize entities into error groups to reveal what BiLSTM learned vs Histogram and how GNN helps."""
    texts = feats_dict['texts']
    labels = feats_dict['labels']
    
    y_true = np.array(labels)
    hist_preds = np.array(metrics_dict['Histogram (4D)']['y_pred'])
    avg_preds = np.array(metrics_dict['One-Hot + AVG (4D)']['y_pred'])
    bilstm_preds = np.array(metrics_dict['Trained BiLSTM (64D)']['y_pred'])
    
    bilstm_win = []   # BiLSTM correct, Histogram wrong
    hist_win = []     # Histogram correct, BiLSTM wrong
    both_wrong = []   # Both text models wrong
    both_correct = [] # Both correct
    
    for i in range(len(labels)):
        t_label = y_true[i]
        h_correct = (hist_preds[i] == t_label)
        b_correct = (bilstm_preds[i] == t_label)
        
        item = {
            'text': texts[i],
            'true_label': t_label,
            'hist_pred': hist_preds[i],
            'avg_pred': avg_preds[i],
            'bilstm_pred': bilstm_preds[i]
        }
        
        if b_correct and not h_correct:
            bilstm_win.append(item)
        elif h_correct and not b_correct:
            hist_win.append(item)
        elif not h_correct and not b_correct:
            both_wrong.append(item)
        else:
            both_correct.append(item)
            
    print(f"\n================ Qualitative Error Analysis (FUNSD) ================")
    print(f"Total entities: {len(labels)}")
    print(f"  Both Correct: {len(both_correct)} ({len(both_correct)/len(labels)*100:.1f}%)")
    print(f"  BiLSTM Wins (BiLSTM correct, Hist wrong): {len(bilstm_win)} ({len(bilstm_win)/len(labels)*100:.1f}%)")
    print(f"  Histogram Wins (Hist correct, BiLSTM wrong): {len(hist_win)} ({len(hist_win)/len(labels)*100:.1f}%)")
    print(f"  Both Text Models Wrong: {len(both_wrong)} ({len(both_wrong)/len(labels)*100:.1f}%)")
    
    # Check how many of 'both_wrong' in text space are FIXED by post-GNN spatial message passing
    gnn_fixed_count = 0
    if post_gnn_res is not None:
        gnn_preds = np.array(post_gnn_res['y_pred_direct'])
        gnn_y_true = np.array(post_gnn_res['y_true'])
        
        for i in range(min(len(labels), len(gnn_y_true))):
            t_label = gnn_y_true[i]
            h_correct = (hist_preds[i] == t_label)
            b_correct = (bilstm_preds[i] == t_label)
            g_correct = (gnn_preds[i] == t_label)
            
            if not h_correct and not b_correct and g_correct:
                gnn_fixed_count += 1
                
        print(f"  -> Entities WRONG in text space BUT FIXED by GNN spatial message passing: {gnn_fixed_count}")
        
    return {
        'bilstm_win_sample': bilstm_win[:15],
        'hist_win_sample': hist_win[:15],
        'both_wrong_sample': both_wrong[:15],
        'counts': {
            'both_correct': len(both_correct),
            'bilstm_win': len(bilstm_win),
            'hist_win': len(hist_win),
            'both_wrong': len(both_wrong),
            'gnn_fixed': gnn_fixed_count
        }
    }


def plot_overall_comparison(pre_metrics: dict, post_metrics: dict, out_dir: str, dataset_name: str):
    """Plot bar chart comparing Linear Probe and 1-NN accuracy across all representations."""
    model_names = list(pre_metrics.keys())
    linear_accs = [pre_metrics[m]['linear_probe_accuracy']*100 for m in model_names]
    knn_accs = [pre_metrics[m]['knn_1_accuracy']*100 for m in model_names]
    
    if post_metrics is not None:
        model_names.extend(['Post-Projector (Pre-GNN)', 'Post-GNN Message Passing'])
        linear_accs.extend([post_metrics['Post-Projector (Pre-GNN)']['linear_probe_accuracy']*100,
                            post_metrics['Post-GNN Message Passing']['linear_probe_accuracy']*100])
        knn_accs.extend([post_metrics['Post-Projector (Pre-GNN)']['knn_1_accuracy']*100,
                         post_metrics['Post-GNN Message Passing']['knn_1_accuracy']*100])
        
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 5))
    rects1 = ax.bar(x - width/2, linear_accs, width, label='Linear Probe Acc (%)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, knn_accs, width, label='1-NN Acc (%)', color='#ff7f0e')
    
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Representation Separability Comparison ({dataset_name})', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left')
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'full_representation_comparison_{dataset_name.lower()}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved representation comparison bar chart to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced Character & GNN Separability Analysis")
    parser.add_argument('--dataset', type=str, default='FUNSD', choices=['FUNSD', 'TONDFU', 'BOTH'],
                        help="Dataset to analyze")
    parser.add_argument('--checkpoint', type=str,
                        default='combined-emb-bilstm-all-1-bug-fix/e2e_char_embed-20250207-1519-3f4a86.pt',
                        help="Relative path to model checkpoint")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID (-1 for CPU)")
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device for feature extraction: {device}")
    
    ckpt_path = CHECKPOINTS / args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
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
    
    out_dir = OUTPUTS / 'feature_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    datasets_to_run = ['FUNSD', 'TONDFU'] if args.dataset == 'BOTH' else [args.dataset]
    all_summary = {}
    
    for dataset_name in datasets_to_run:
        print(f"\n================ Running Pre-GNN Analysis on {dataset_name} ================")
        entities = load_dataset_entities(dataset_name)
        print(f"Total entities extracted: {len(entities)}")
        
        feats = extract_pre_gnn_features(entities, trained_char_module, device)
        
        feature_keys = [
            'Histogram (4D)',
            'One-Hot + AVG (4D)',
            'Untrained BiLSTM (64D)',
            'Trained BiLSTM (64D)',
            'Histogram + Trained BiLSTM (68D)'
        ]
        
        pre_metrics = {}
        for fk in feature_keys:
            res = compute_metrics(feats[fk], feats['labels'])
            # remove y_pred array from summary metrics before print to keep output concise
            y_pred = res.pop('y_pred')
            pre_metrics[fk] = res
            res['y_pred'] = y_pred
            
            print(f"\n--- {fk} ---")
            print(f"  Silhouette Score       : {res['silhouette']:.4f}")
            print(f"  Davies-Bouldin Index   : {res['davies_bouldin']:.4f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {res['calinski_harabasz']:.4f} (higher is better)")
            print(f"  1-NN Accuracy (5-Fold) : {res['knn_1_accuracy']*100:.2f}%")
            print(f"  Linear Probe Acc (5-Fold): {res['linear_probe_accuracy']*100:.2f}%")
            
        # Run Post-GNN experiment on FUNSD if evaluating FUNSD
        post_gnn_res = None
        if dataset_name == 'FUNSD':
            post_gnn_res = run_post_gnn_experiment(ckpt_path, torch.device('cpu'))
            qualitative_res = perform_qualitative_error_analysis(feats, pre_metrics, post_gnn_res)
        else:
            qualitative_res = perform_qualitative_error_analysis(feats, pre_metrics, None)
            
        plot_overall_comparison(pre_metrics, post_gnn_res, out_dir, dataset_name)
        
        # Prepare clean JSON summary
        clean_pre_metrics = {}
        for fk in feature_keys:
            m_copy = dict(pre_metrics[fk])
            m_copy.pop('y_pred', None)
            clean_pre_metrics[fk] = m_copy
            
        clean_post_metrics = {}
        if post_gnn_res is not None:
            for pk in ['Post-Projector (Pre-GNN)', 'Post-GNN Message Passing', 'Final Logits']:
                m_copy = dict(post_gnn_res[pk])
                m_copy.pop('y_pred', None)
                clean_post_metrics[pk] = m_copy
            clean_post_metrics['direct_model_accuracy'] = post_gnn_res['direct_model_accuracy']
            
        all_summary[dataset_name] = {
            'entity_count': len(entities),
            'pre_gnn_metrics': clean_pre_metrics,
            'post_gnn_metrics': clean_post_metrics,
            'qualitative_analysis': qualitative_res
        }
        
    json_path = os.path.join(out_dir, 'advanced_separability_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64, np.int64)) else str(o))
    print(f"\nSaved advanced evaluation results JSON to {json_path}")
    print("\nAdvanced Separability Analysis complete!")


if __name__ == '__main__':
    main()
