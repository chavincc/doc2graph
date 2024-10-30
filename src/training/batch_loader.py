from torch.utils.data import Dataset
from typing import List, Tuple
import dgl

class GraphAndRawTextDataset(Dataset):
    def __init__(
        self,
        graphs: List[dgl.DGLGraph],
        texts: List[List[str]] # node_text[node_idx][graph_idx]
    ):
        self.graphs = graphs
        self.texts = texts

    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[dgl.DGLGraph, List[str]]:
        return self.graphs[idx], self.texts[idx]


def graph_raw_text_collate_fn(
    batch: List[
        Tuple[dgl.DGLGraph, List[str]]
    ]
) -> Tuple[dgl.DGLGraph, List[str]]:
    graphs, texts = zip(*batch)
    batched_graph: dgl.DGLGraph = dgl.batch(graphs)
    batched_texts: List[str] = [node_text for text in texts for node_text in text]
    return batched_graph, batched_texts
