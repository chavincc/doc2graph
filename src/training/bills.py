from datetime import datetime
from sklearn.model_selection import KFold, ShuffleSplit
import torch
from torch.nn import functional as F
from random import shuffle, seed
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import dgl 
from dgl.dataloading import GraphDataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
from statistics import mean
import numpy as np
from PIL import Image
import gc
import uuid

from src.data.dataloader import Document2Graph
from src.paths import *
from src.models.graphs import SetModel
from src.utils import get_config
from src.training.utils import *
from src.data.graph_builder import GraphBuilder
from src.training.batch_loader import GraphAndRawTextDataset, graph_raw_text_collate_fn

def e2e_char_embed(args):
    # configs
    start_training = time.time()
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    # for test result logging. loaded first to prevent file change when training multiple instances.
    model_cfg = get_config(CFGM / args.model)

    if not args.test:
        data = Document2Graph(name='BILLS TRAIN', src_path=BILLS_TRAIN, device=device, output_dir=TRAIN_SAMPLES)
        data.get_info()

        # ensure texts and graphs matched before splitting
        assert len(data.graphs) == len(data.texts)
        assert data.graphs[0].number_of_nodes() == len(data.texts[0])
        assert data.graphs[-1].number_of_nodes() == len(data.texts[-1])

        # TODO: change fold number !!
        kf = KFold(n_splits=2, shuffle=True, random_state=cfg_train.seed)
        kf_split_fold = kf.split(data.graphs)
        
        models = []

        for fold in kf_split_fold:
            train_index, val_index = fold

            # TRAIN
            train_graphs = [data.graphs[idx] for idx in train_index]
            train_texts = [data.texts[i] for i in train_index]
            train_dataset = GraphAndRawTextDataset(train_graphs, train_texts)
            train_dataloader = GraphDataLoader(
                train_dataset,
                batch_size=cfg_train.batch_size,
                shuffle=True,
                collate_fn=graph_raw_text_collate_fn
            )
        
            val_graphs = [data.graphs[i] for i in val_index]
            val_texts = [data.texts[i] for i in val_index]
            val_dataset = GraphAndRawTextDataset(val_graphs, val_texts)
            val_dataloader = GraphDataLoader(
                val_dataset,
                batch_size=cfg_train.batch_size,
                shuffle=True,
                collate_fn=graph_raw_text_collate_fn
            )
            
            ################* STEP 1: CREATE MODEL ################
            model = sm.get_model(data.node_num_classes, data.edge_num_classes, data.get_chunks())
            # load pretrained weights if given
            if args.pretrained != None:
                model.load_state_dict(torch.load(CHECKPOINTS / args.pretrained))
                print(f'>> loaded pretrained model from {CHECKPOINTS / args.pretrained}')
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
            e = datetime.now()

            # use uuid to simply prevent train_name collision
            full_uuid = uuid.uuid4()
            truncated_uuid = str(full_uuid).replace("-", "")[:6]
            train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}-{truncated_uuid}'

            models.append(train_name+'.pt')
            stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=cfg_train.stopper_patience)

            writer = SummaryWriter(log_dir=RUNS)
        
            ################* STEP 2: TRAINING ################
            print("\n### TRAINING ###")
            print(f"-> Training samples: {len(train_dataset)}")
            print(f"-> Validation samples: {len(val_dataset)}\n")

            for epoch in range(cfg_train.epochs):
                # TRAINING
                model.train()
                total_loss: float = 0.0
                total_macro: float = 0.0
                total_auc: float = 0.0
                batches: int = 0

                for batched_graph, batched_text in train_dataloader:
                    batched_graph = batched_graph.to(device)
                    batched_graph.ndata['feat'] = batched_graph.ndata['feat'].to(device)
                    batched_graph.ndata['label'] = batched_graph.ndata['label'].to(device)
                    batched_graph.edata['label'] = batched_graph.edata['label'].to(device)

                    n_scores: torch.Tensor; e_scores: torch.Tensor
                    n_scores, e_scores = model(batched_graph, batched_graph.ndata['feat'], batched_text)

                    n_loss: torch.Tensor = compute_crossentropy_loss(n_scores, batched_graph.ndata['label'])
                    e_loss: torch.Tensor = compute_crossentropy_loss(e_scores, batched_graph.edata['label'])
                    tot_loss: torch.Tensor = n_loss + e_loss

                    macro, micro = get_f1(n_scores, batched_graph.ndata['label'])
                    auc: float = compute_auc_mc(e_scores, batched_graph.edata['label'])

                    optimizer.zero_grad()
                    tot_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()

                    total_loss += tot_loss.item()
                    total_macro += macro
                    total_auc += auc
                    batches += 1
                
                avg_loss: float = total_loss / batches
                avg_macro: float = total_macro / batches
                avg_auc: float = total_auc / batches

                #* VALIDATION
                model.eval()
                val_tot_loss: float = 0.0
                val_macro_total: float = 0.0
                val_auc_total: float = 0.0
                val_batches: int = 0
                with torch.no_grad():
                    for batched_graph, batched_texts in val_dataloader:
                        batched_graph = batched_graph.to(device)
                        batched_graph.ndata['feat'] = batched_graph.ndata['feat'].to(device)
                        batched_graph.ndata['label'] = batched_graph.ndata['label'].to(device)
                        batched_graph.edata['label'] = batched_graph.edata['label'].to(device)

                        val_n_scores, val_e_scores = model(batched_graph, batched_graph.ndata['feat'], batched_texts)

                        val_n_loss: torch.Tensor = compute_crossentropy_loss(val_n_scores, batched_graph.ndata['label'])
                        val_e_loss: torch.Tensor = compute_crossentropy_loss(val_e_scores, batched_graph.edata['label'])
                        val_tot_loss += val_n_loss.item() + val_e_loss.item()

                        val_macro, _ = get_f1(val_n_scores, batched_graph.ndata['label'])
                        val_auc: float = compute_auc_mc(val_e_scores, batched_graph.edata['label'])
                        val_macro_total += val_macro
                        val_auc_total += val_auc
                        val_batches += 1

                val_tot_loss /= val_batches
                val_macro_avg: float = val_macro_total / val_batches
                val_auc_avg: float = val_auc_total / val_batches

                #* PRINTING IMAGEs AND RESULTS
                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} | ValAUC-PR {:.4f} |"
                .format(epoch, avg_loss, avg_macro, avg_auc, val_tot_loss, val_macro_avg, val_auc_avg))
                
                if cfg_train.stopper_metric == 'loss':
                    step_value = val_tot_loss
                elif cfg_train.stopper_metric == 'acc':
                    step_value = val_auc_avg
                
                ss = stopper.step(step_value)
                if ss == 'stop':
                    break

                writer.add_scalars('AUC-PR', {'train': avg_auc, 'val': val_auc_avg}, epoch)
                writer.add_scalars('LOSS', {'train': avg_loss, 'val': val_tot_loss}, epoch)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # clear gpu memory (experimental)
            del train_graphs, train_texts, val_graphs, val_texts, model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

            # close writer
            writer.close()

            # temporary 1-fold training for param tuning
            break

    return


def train_bills(args):
    if args.model == 'e2e_char_embed':
        e2e_char_embed(args)
    else:
        raise Exception("Model selected does not exists. Choose 'e2e_char_embed'.")
    return
