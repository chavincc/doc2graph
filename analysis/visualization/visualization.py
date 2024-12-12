from PIL import Image
import torch
from torch.nn import functional as F

from src.paths import FUNSD_TEST, CHECKPOINTS
from src.models.graphs import SetModel
from src.data.dataloader import Document2Graph
from analysis.visualization.draw import draw_results


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # set up
    funsd_test_images = FUNSD_TEST / 'images'
    image_index = 0
    device = 'cuda:2'
    # device = 'cpu'

    # create input
    data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device=device)
    # graphs, node_labels, edge_labels, features, texts, feature_chunks, num_mods
    graphs = data.graphs
    texts = data.texts
    boxes = data.boxes
    paths = data.paths

    # load image
    image_path = paths[image_index]
    image = Image.open(image_path).convert('RGB')

    # create model
    sm = SetModel(name='e2e_char_embed', device=device)
    model = sm.get_model(4, 2, data.feature_chunks, False) # 4 and 2 refers to nodes and edge classes, check paper for details!
    model.load_state_dict(torch.load(CHECKPOINTS / 'combined-one-hot-lstm-no-vis-1' / 'e2e_char_embed-20241202-1625.pt')) # load pretrained model
    model.eval() # set the model for inference only

    # inference
    with torch.no_grad():
        graph = data.graphs[image_index]
        graph_texts = texts[image_index]
        graph_boxes = boxes[image_index]
        
        n, e = model(graph.to(device), graph.ndata['feat'].to(device), graph_texts)
        _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
        _, npreds = torch.max(F.softmax(n, dim=1), dim=1)

        # save results
        links = (epreds == 1).nonzero(as_tuple=True)[0].tolist()
        u, v = graph.edges()

        draw_results(image, graph_boxes, links, u, v)

        image.save('vis.jpg')
