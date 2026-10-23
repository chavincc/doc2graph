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

from src.paths import ROOT, DATA, CHECKPOINTS, OUTPUTS
from src.data.utils import get_histogram
from src.models.char_embedding.model import CharEmbeddingModule, AggregatorMethod


def load_dataset_entities(dataset_name: str, data_split: str = 'testing_data'):
    """Load text and node label for each entity in the dataset."""
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


def extract_features(entities: list, char_module: CharEmbeddingModule, device: torch.device):
    """Extract histogram features, BiLSTM char embeddings, and combined features."""
    texts = [e['text'] for e in entities]
    labels = [e['label'] for e in entities]
    
    # 1. Histogram features (4D)
    hist_feats = np.array(get_histogram(texts), dtype=np.float32)
    
    # 2. BiLSTM Char Embeddings (64D)
    char_module.eval()
    with torch.no_grad():
        pre_tensor, seq_lengths = char_module.preprocess(texts)
        bilstm_tensor = char_module(pre_tensor, seq_lengths)
        bilstm_feats = bilstm_tensor.cpu().numpy()
        
    # 3. Combined features (68D)
    concat_feats = np.hstack([hist_feats, bilstm_feats])
    
    return {
        'texts': texts,
        'labels': labels,
        'Histogram': hist_feats,
        'BiLSTM Char Embedding': bilstm_feats,
        'Histogram + BiLSTM': concat_feats
    }


def compute_metrics(X: np.ndarray, y: np.ndarray):
    """Compute clustering separability and classification metrics."""
    # Scale features for metric stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Clustering metrics
    sil = float(silhouette_score(X_scaled, y))
    db = float(davies_bouldin_score(X_scaled, y))
    ch = float(calinski_harabasz_score(X_scaled, y))
    
    # 2. 1-NN Accuracy (Stratified 5-Fold CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn_scores = cross_val_score(knn, X_scaled, y, cv=cv)
    knn_acc = float(np.mean(knn_scores))
    
    # 3. Linear Probe Accuracy (Logistic Regression Stratified 5-Fold CV)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(clf, X_scaled, y, cv=cv)
    lr_acc = float(np.mean(lr_scores))
    
    # Confusion matrix for linear probe
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
        'classes': classes
    }


def compute_pairwise_metrics(X: np.ndarray, y: np.ndarray, classes: list):
    """Compute pairwise class centroid distances and pairwise 1-NN accuracy."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_cls = len(classes)
    centroids = {c: X_scaled[np.array(y) == c].mean(axis=0) for c in classes}
    
    centroid_dist = np.zeros((n_cls, n_cls))
    pairwise_knn = np.zeros((n_cls, n_cls))
    
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i == j:
                centroid_dist[i, j] = 0.0
                pairwise_knn[i, j] = 1.0
            else:
                dist = np.linalg.norm(centroids[c1] - centroids[c2])
                centroid_dist[i, j] = dist
                
                # Pairwise 1-NN
                mask = (np.array(y) == c1) | (np.array(y) == c2)
                X_pair = X_scaled[mask]
                y_pair = np.array(y)[mask]
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                knn = KNeighborsClassifier(n_neighbors=1)
                scores = cross_val_score(knn, X_pair, y_pair, cv=cv)
                pairwise_knn[i, j] = float(np.mean(scores))
                
    return {
        'classes': classes,
        'centroid_distances': centroid_dist.tolist(),
        'pairwise_1nn_accuracy': pairwise_knn.tolist()
    }


def plot_tsne(feats_dict: dict, labels: list, out_dir: str, dataset_name: str):
    """Plot t-SNE 2D projections for each feature representation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    label_colors = {
        'question': '#1f77b4',
        'answer': '#2ca02c',
        'header': '#ff7f0e',
        'other': '#7f7f7f'
    }
    
    unique_labels = sorted(list(set(labels)))
    
    for ax, (feat_name, X) in zip(axes, feats_dict.items()):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X_scaled)
        
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            color = label_colors.get(lbl, '#333333')
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=lbl, alpha=0.6, edgecolors='none', color=color, s=25)
            
        ax.set_title(f"{feat_name} ({dataset_name})", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'tsne_comparison_{dataset_name.lower()}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {plot_path}")


def plot_histogram_distribution(hist_feats: np.ndarray, labels: list, out_dir: str, dataset_name: str):
    """Plot per-dimension histogram distributions grouped by entity label."""
    df = pd.DataFrame(hist_feats, columns=['% Alphabet', '% Digit', '% Symbol', '% Unknown'])
    df['Label'] = labels
    
    labels_order = ['question', 'answer', 'header', 'other']
    existing_labels = [l for l in labels_order if l in df['Label'].unique()]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cols = ['% Alphabet', '% Digit', '% Symbol', '% Unknown']
    
    for ax, col in zip(axes, cols):
        data_to_plot = [df[df['Label'] == lbl][col] for lbl in existing_labels]
        ax.boxplot(data_to_plot, labels=existing_labels, patch_artist=True,
                   boxprops=dict(facecolor='#a6cee3', color='#1f78b4'),
                   medianprops=dict(color='red', linewidth=1.5))
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_ylabel("Percentage")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
    plt.suptitle(f"Histogram Feature Component Distributions across Labels ({dataset_name})", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'histogram_boxplots_{dataset_name.lower()}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved histogram box plots to {plot_path}")


def plot_bilstm_heatmap(bilstm_feats: np.ndarray, labels: list, out_dir: str, dataset_name: str):
    """Plot heatmap of mean BiLSTM feature activations per entity label class."""
    df = pd.DataFrame(bilstm_feats)
    df['Label'] = labels
    mean_activations = df.groupby('Label').mean()
    
    plt.figure(figsize=(14, 4))
    im = plt.imshow(mean_activations.values, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Mean Activation')
    
    plt.yticks(ticks=range(len(mean_activations.index)), labels=mean_activations.index, fontsize=11)
    plt.xlabel("BiLSTM Output Dimension (0..63)", fontsize=11)
    plt.ylabel("Entity Label", fontsize=11)
    plt.title(f"Mean BiLSTM Feature Activations per Entity Label ({dataset_name})", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'bilstm_activation_heatmap_{dataset_name.lower()}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved BiLSTM activation heatmap to {plot_path}")


def plot_confusion_matrices(results: dict, out_dir: str, dataset_name: str):
    """Plot confusion matrices for Linear Probe across feature types."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for ax, (feat_name, res) in zip(axes, results.items()):
        cm = np.array(res['confusion_matrix'])
        classes = res['classes']
        
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
                
        ax.set_title(f"{feat_name}\n(Acc: {res['linear_probe_accuracy']*100:.1f}%)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    plt.suptitle(f"Linear Probe Confusion Matrices ({dataset_name})", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'confusion_matrices_{dataset_name.lower()}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Encoded Character Feature Separability Analysis")
    parser.add_argument('--dataset', type=str, default='FUNSD', choices=['FUNSD', 'TONDFU', 'BOTH'],
                        help="Dataset to analyze")
    parser.add_argument('--checkpoint', type=str,
                        default='combined-emb-bilstm-all-1-bug-fix/e2e_char_embed-20250207-1519-3f4a86.pt',
                        help="Relative path to model checkpoint")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID (-1 for CPU)")
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_path = CHECKPOINTS / args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    # Load CharEmbeddingModule
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    char_module = CharEmbeddingModule(
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
            
    char_module.load_state_dict(char_state_dict, strict=True)
    char_module.to(device)
    
    out_dir = OUTPUTS / 'feature_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    datasets_to_run = ['FUNSD', 'TONDFU'] if args.dataset == 'BOTH' else [args.dataset]
    
    all_summary = {}
    
    for dataset_name in datasets_to_run:
        print(f"\n================ Running Analysis on {dataset_name} ================")
        entities = load_dataset_entities(dataset_name)
        print(f"Total entities extracted from {dataset_name} test split: {len(entities)}")
        
        # Label distribution count
        label_counts = pd.Series([e['label'] for e in entities]).value_counts().to_dict()
        print(f"Label distribution: {label_counts}")
        
        # Feature extraction
        feats = extract_features(entities, char_module, device)
        
        feature_types = ['Histogram', 'BiLSTM Char Embedding', 'Histogram + BiLSTM']
        metrics_summary = {}
        pairwise_summary = {}
        
        for ft in feature_types:
            X = feats[ft]
            y = feats['labels']
            res = compute_metrics(X, y)
            metrics_summary[ft] = res
            
            p_res = compute_pairwise_metrics(X, y, res['classes'])
            pairwise_summary[ft] = p_res
            
            print(f"\n--- {ft} ---")
            print(f"  Silhouette Score       : {res['silhouette']:.4f}")
            print(f"  Davies-Bouldin Index   : {res['davies_bouldin']:.4f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {res['calinski_harabasz']:.4f} (higher is better)")
            print(f"  1-NN Accuracy (5-Fold) : {res['knn_1_accuracy']*100:.2f}%")
            print(f"  Linear Probe Acc (5-Fold): {res['linear_probe_accuracy']*100:.2f}%")
            
        # Visualizations
        feats_dict = {ft: feats[ft] for ft in feature_types}
        plot_tsne(feats_dict, feats['labels'], out_dir, dataset_name)
        plot_histogram_distribution(feats['Histogram'], feats['labels'], out_dir, dataset_name)
        plot_bilstm_heatmap(feats['BiLSTM Char Embedding'], feats['labels'], out_dir, dataset_name)
        plot_confusion_matrices(metrics_summary, out_dir, dataset_name)
        
        # Compute mean histogram percentages per label
        df_hist = pd.DataFrame(feats['Histogram'], columns=['% Alphabet', '% Digit', '% Symbol', '% Unknown'])
        df_hist['Label'] = feats['labels']
        hist_stats = {}
        for col in ['% Alphabet', '% Digit', '% Symbol', '% Unknown']:
            hist_stats[col] = {
                label: {
                    'mean': float(df_hist[df_hist['Label'] == label][col].mean()),
                    'std': float(df_hist[df_hist['Label'] == label][col].std())
                } for label in df_hist['Label'].unique()
            }
        
        all_summary[dataset_name] = {
            'entity_count': len(entities),
            'label_counts': label_counts,
            'metrics': metrics_summary,
            'pairwise': pairwise_summary,
            'histogram_stats': hist_stats
        }
        
    # Save results summary JSON
    json_path = os.path.join(out_dir, 'feature_separability_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(o))
    print(f"\nSaved overall evaluation results JSON to {json_path}")
    
    print("\nFeature Separability Analysis complete!")


if __name__ == '__main__':
    main()
