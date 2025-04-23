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

def e2e(args):

    # configs
    start_training = time.time()
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device, output_dir=TRAIN_SAMPLES)
        data.get_info()

        ss = KFold(n_splits=10, shuffle=True, random_state=cfg_train.seed)
        cv_indices = ss.split(data.graphs)
        
        models = []
        train_index, val_index = next(ss.split(data.graphs))

        for cvs in cv_indices:

            train_index, val_index = cvs

            # TRAIN
            train_graphs = [data.graphs[i] for i in train_index]
            tg = dgl.batch(train_graphs)
            tg = tg.int().to(device)
        
            val_graphs = [data.graphs[i] for i in val_index]
            vg = dgl.batch(val_graphs)
            vg = vg.int().to(device)
            
            ################* STEP 1: CREATE MODEL ################
            model = sm.get_model(data.node_num_classes, data.edge_num_classes, data.get_chunks())
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
            # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=400, min_lr=1e-3, verbose=True, factor=0.01)
            # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            e = datetime.now()
            train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
            models.append(train_name+'.pt')
            stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=cfg_train.stopper_patience)
            # writer = SummaryWriter(log_dir=RUNS)
            # convert_imgs = transforms.ToTensor()
        
            ################* STEP 2: TRAINING ################
            print("\n### TRAINING ###")
            print(f"-> Training samples: {tg.batch_size}")
            print(f"-> Validation samples: {vg.batch_size}\n")

            # im_step = 0
            for epoch in range(cfg_train.epochs):

                #* TRAINING
                model.train()
                
                n_scores, e_scores = model(tg, tg.ndata['feat'].to(device))
                n_loss = compute_crossentropy_loss(n_scores.to(device), tg.ndata['label'].to(device))
                e_loss = compute_crossentropy_loss(e_scores.to(device), tg.edata['label'].to(device))
                tot_loss = n_loss + e_loss
                macro, micro = get_f1(n_scores, tg.ndata['label'].to(device))
                auc = compute_auc_mc(e_scores.to(device), tg.edata['label'].to(device))


                optimizer.zero_grad()
                tot_loss.backward()
                n = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                #* VALIDATION
                model.eval()
                with torch.no_grad():
                    val_n_scores, val_e_scores = model(vg, vg.ndata['feat'].to(device))
                    val_n_loss = compute_crossentropy_loss(val_n_scores.to(device), vg.ndata['label'].to(device))
                    val_e_loss = compute_crossentropy_loss(val_e_scores.to(device), vg.edata['label'].to(device))
                    val_tot_loss = val_n_loss + val_e_loss
                    val_macro, _ = get_f1(val_n_scores, vg.ndata['label'].to(device))
                    val_auc = compute_auc_mc(val_e_scores.to(device), vg.edata['label'].to(device))
                
                # scheduler.step(val_auc)
                # scheduler.step()

                #* PRINTING IMAGEs AND RESULTS

                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} | ValAUC-PR {:.4f} |"
                .format(epoch, tot_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))
                
                if cfg_train.stopper_metric == 'loss':
                    step_value = val_tot_loss.item()
                elif cfg_train.stopper_metric == 'acc':
                    step_value = val_auc
                
                # if val_auc > best_val_auc:
                #     best_val_auc = val_auc
                #     best_model = train_name
                
                ss = stopper.step(step_value)

                # if  ss == 'improved':
                #     im_step = epoch
                #     train_imgs = []
                #     for r in rand_tid:
                #         start, end = 0, 0
                #         for tid in train_index:
                #             start = end
                #             end += data.graphs[tid].num_edges()
                #             if tid == r: break

                #         _, targets = torch.max(F.log_softmax(e_scores[start:end], dim=1), dim=1)
                #         kvp_ids = targets.nonzero().flatten().tolist()
                #         train_imgs.append(convert_imgs(data.print_graph(num=r, labels_ids=kvp_ids, name=f'train_{r}'))[:, :, :700])
                #         # data.print_graph(num=r, name=f'train_labels_{r}')

                #     val_imgs = []
                #     for r in rand_vid:
                #         v_start, v_end = 0, 0
                #         for vid in val_index:
                #             v_start = v_end
                #             v_end += data.graphs[vid].num_edges()
                #             if vid == r: break

                #         _, val_targets = torch.max(F.log_softmax(val_e_scores[v_start:v_end], dim=1), dim=1)
                #         val_kvp_ids = val_targets.nonzero().flatten().tolist()
                #         val_imgs.append(convert_imgs(data.print_graph(num=r, labels_ids=val_kvp_ids, name=f'val_{r}'))[:, :, :700])
                        # data.print_graph(num=r, name=f'val_labels_{r}')

                if ss == 'stop':
                    break

                # writer.add_scalars('AUC-PR', {'train': auc, 'val': val_auc}, epoch)
                # writer.add_scalars('LOSS', {'train': tot_loss.item(), 'val': val_tot_loss.item()}, epoch)
                # writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

                # train_grid = torchvision.utils.make_grid(train_imgs)
                # writer.add_image('train_images', train_grid, im_step)
                # val_grid = torchvision.utils.make_grid(val_imgs)
                # writer.add_image('val_images', val_grid, im_step)

                # important!!! without this line the latter fold will cause CUDA out of memory !!!
                torch.cuda.empty_cache()

            # temporary 1-fold training for param tuning
            # break
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        models = args.weights
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = device, output_dir=TEST_SAMPLES)
    test_data.get_info()
    
    model = sm.get_model(test_data.node_num_classes, test_data.edge_num_classes, test_data.get_chunks())
    best_model = ''
    nodes_micro = []
    edges_f1 = []
    test_graph = dgl.batch(test_data.graphs).to(device)

    for m in models:
        model.load_state_dict(torch.load(CHECKPOINTS / m))
        model.eval()
        with torch.no_grad():

            n, e = model(test_graph, test_graph.ndata['feat'].to(device))
            auc = compute_auc_mc(e.to(device), test_graph.edata['label'].to(device))
            _, preds = torch.max(F.softmax(e, dim=1), dim=1)

            accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
            _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)
            edges_f1.append(classes_f1[1])

            macro, micro = get_f1(n, test_graph.ndata['label'].to(device))
            nodes_micro.append(micro)
            if classes_f1[1] >= max(edges_f1):
                best_model = m

            test_graph.edata['preds'] = preds

        ################* STEP 4: RESULTS ################
        print("\n### RESULTS {} ###".format(m))
        print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    print(f"\n -> Loading best model {best_model}")
    model.load_state_dict(torch.load(CHECKPOINTS / best_model))
    model.eval()
    with torch.no_grad():

        n, e = model(test_graph, test_graph.ndata['feat'].to(device))
        auc = compute_auc_mc(e.to(device), test_graph.edata['label'].to(device))
        
        _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
        _, npreds = torch.max(F.softmax(n, dim=1), dim=1)
        test_graph.edata['preds'] = epreds
        test_graph.ndata['preds'] = npreds
        test_graph.ndata['net'] = n

        accuracy, f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'], per_class=True)
        macro, micro = get_f1(n, test_graph.ndata['label'].to(device))

    # ################* STEP 4: RESULTS ################
    print("\n### BEST RESULTS ###")
    print("AUC {:.4f}".format(auc))
    print("Accuracy {:.4f}".format(accuracy))
    print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
    print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    print("\n### AVG RESULTS ###")
    print("Semantic Entity Labeling: MEAN ", mean(nodes_micro), " STD: ", np.std(nodes_micro))
    print("Entity Linking: MEAN ", mean(edges_f1),"STD", np.std(edges_f1))

    if not args.test:
        feat_n, feat_e = get_features(args)
        #? if skipping training, no need to save anything
        model = get_config(CFGM / args.model)
        results = {'MODEL': {
            'name': sm.get_name(),
            'weights': best_model,
            'net-params': sm.get_total_params(), 
            'num-layers': model.num_layers,
            'projector-output': model.out_chunks,
            'dropout': model.dropout,
            'lastFC': model.hidden_dim
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'RESULTS': {
                'val-loss': stopper.best_score, 
                'f1-scores': f1,
		        'f1-classes': classes_f1,
                'nodes-f1': [macro, micro],
                'std-pairs': np.std(edges_f1),
                'mean-pairs': mean(edges_f1)
            }}
        save_test_results(train_name, results)
    
        print("END TRAINING:", time.time() - start_training)
    return {'LINKS [MAX, MEAN, STD]': [classes_f1[1], mean(edges_f1), np.std(edges_f1)], 'NODES [MAX, MEAN, STD]': [micro, mean(nodes_micro), np.std(nodes_micro)]}

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
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device, output_dir=TRAIN_SAMPLES)
        data.get_info()

        # ensure texts and graphs matched before splitting
        assert len(data.graphs) == len(data.texts)
        assert data.graphs[0].number_of_nodes() == len(data.texts[0])
        assert data.graphs[-1].number_of_nodes() == len(data.texts[-1])

        kf = KFold(n_splits=10, shuffle=True, random_state=cfg_train.seed)
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
            # break
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        models = args.weights
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = device, output_dir=TEST_SAMPLES)
    test_data.get_info()
    
    model = sm.get_model(test_data.node_num_classes, test_data.edge_num_classes, test_data.get_chunks())
    best_model = ''
    best_result = {}
    nodes_micro = []
    edges_f1 = []

    test_graphs = test_data.graphs
    test_texts = test_data.texts
    test_dataset = GraphAndRawTextDataset(test_graphs, test_texts)
    test_dataloader = GraphDataLoader(
        test_dataset,
        batch_size=cfg_train.batch_size,
        collate_fn=graph_raw_text_collate_fn
    )

    # assert compatibility of flatten node texts
    assert len(test_graphs) == len(test_texts)
    assert test_graphs[0].number_of_nodes() == len(test_texts[0])

    for m in models:
        model.load_state_dict(torch.load(CHECKPOINTS / m))
        model.eval()

        all_edge_scores = [] # for computing AUC
        all_edge_preds = []
        all_edge_labels = []
        all_edge_src = [] # for mapping predictions to raw value
        all_edge_dst = [] # for mapping predictions to raw value
        all_node_scores = []
        all_node_preds = []
        all_node_labels = []
        all_node_texts = [] # for mapping predictions to raw value
        with torch.no_grad():
            for batched_graph, batched_texts in test_dataloader:
                batched_graph = batched_graph.to(device)
                batched_graph.ndata['feat'] = batched_graph.ndata['feat'].to(device)
                batched_graph.ndata['label'] = batched_graph.ndata['label'].to(device)
                batched_graph.edata['label'] = batched_graph.edata['label'].to(device)

                n_scores, e_scores = model(batched_graph, batched_graph.ndata['feat'], batched_texts)

                e_scores_softmax = F.softmax(e_scores, dim=1)
                edge_preds = e_scores_softmax.argmax(dim=1)
                node_preds = n_scores.argmax(dim=1)

                all_edge_scores.append(e_scores.cpu())
                all_edge_preds.append(edge_preds.cpu())
                all_edge_labels.append(batched_graph.edata['label'].cpu())
                all_edge_src.append(batched_graph.edges()[0])
                all_edge_dst.append(batched_graph.edges()[1])
                all_node_scores.append(n_scores.cpu())
                all_node_preds.append(node_preds.cpu())
                all_node_labels.append(batched_graph.ndata['label'].cpu())
                all_node_texts.append(batched_texts)

        all_edge_scores = torch.cat(all_edge_scores)
        all_edge_preds = torch.cat(all_edge_preds)
        all_edge_labels = torch.cat(all_edge_labels)
        all_edge_src = torch.cat(all_edge_src)
        all_edge_dst = torch.cat(all_edge_dst)
        all_node_scores = torch.cat(all_node_scores)
        all_node_preds = torch.cat(all_node_preds)
        all_node_labels = torch.cat(all_node_labels)
        all_node_texts = [[text for text_list in all_node_texts for text in text_list]] # flatten :( very unreadable

        with open('all_node_preds.json', 'w') as f:
            json.dump(all_node_preds.tolist(), f, indent=4)
        with open('all_node_labels.json', 'w') as f:
            json.dump(all_node_labels.tolist(), f, indent=4)
        with open('all_node_texts.json', 'w') as f:
            json.dump(all_node_texts, f, indent=4)

        with open('all_edge_preds.json', 'w') as f:
            json.dump(all_edge_preds.tolist(), f, indent=4)
        with open('all_edge_labels.json', 'w') as f:
            json.dump(all_edge_labels.tolist(), f, indent=4)

        with open('all_edge_pair.txt', 'w') as f:
            for src, dst in zip(all_edge_src.tolist(), all_edge_dst.tolist()):
                f.write(f"{src} {dst}\n")

        auc = compute_auc_mc(all_edge_scores, all_edge_labels)

        edge_accuracy, edge_f1 = get_binary_accuracy_and_f1(all_edge_preds, all_edge_labels)
        _, edge_classes_f1 = get_binary_accuracy_and_f1(all_edge_preds, all_edge_labels, per_class=True)
        current_edge_f1 = edge_classes_f1[1]  # Positive class F1 score
        edges_f1.append(current_edge_f1)

        node_macro_f1, node_micro_f1 = get_f1(all_node_scores, all_node_labels)
        nodes_micro.append(node_micro_f1)

        if current_edge_f1 >= max(edges_f1):
            best_model = m
            best_result = {
                'auc': auc,
                'edge_accuracy': edge_accuracy,
                'edge_f1': edge_f1,
                'edge_classes_f1': edge_classes_f1,
                'node_macro_f1': node_macro_f1,
                'node_micro_f1': node_micro_f1
            }

        ################* STEP 4: RESULTS ################
        print("\n### RESULTS {} ###".format(m))
        print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(edge_classes_f1[0], edge_classes_f1[1]))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(node_macro_f1, node_micro_f1))

    print(f"\n -> Loading best model {best_model}")
    # ################* STEP 4: RESULTS ################
    print("\n### BEST RESULTS ###")
    print("AUC {:.4f}".format(best_result['auc']))
    print("Accuracy {:.4f}".format(best_result['edge_accuracy']))
    print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(best_result['edge_f1'][0], best_result['edge_f1'][1]))
    print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(best_result['edge_classes_f1'][0], best_result['edge_classes_f1'][1]))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(best_result['node_macro_f1'], best_result['node_micro_f1']))

    print("\n### AVG RESULTS ###")
    print("Semantic Entity Labeling: MEAN ", mean(nodes_micro), " STD: ", np.std(nodes_micro))
    print("Entity Linking: MEAN ", mean(edges_f1),"STD", np.std(edges_f1))

    if not args.test:
        feat_n, feat_e = get_features(args)
        results = {
            'MODEL': {
                'name': sm.get_name(),
                'weights': best_model,
                'net-params': sm.get_total_params(), 
                'num-layers': model_cfg.num_layers,
                'projector-output': model_cfg.out_chunks,
                'dropout': model_cfg.dropout,
                'lastFC': model_cfg.hidden_dim,
                'use_baseline_only': model_cfg.use_baseline_only,
                'use_embedding': model_cfg.use_embedding,
                'aggregation_method': model_cfg.aggregation_method,
                'char_embedding_dim': model_cfg.char_embedding_dim,
                'lstm_hidden_dim': model_cfg.lstm_hidden_dim,
                'num_lstm_layer': model_cfg.num_lstm_layer,
                'max_seq_length': model_cfg.max_seq_length
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'RESULTS': {
                'val-loss': stopper.best_score, 
                'f1-scores': best_result['edge_f1'],
		        'f1-classes': best_result['edge_classes_f1'],
                'nodes-f1': [best_result['node_macro_f1'], best_result['node_micro_f1']],
                'std-pairs': np.std(edges_f1),
                'mean-pairs': mean(edges_f1)
            }}
        save_test_results(train_name, results)
    
        print("END TRAINING:", time.time() - start_training)
    return {'LINKS [MAX, MEAN, STD]': [best_result['edge_classes_f1'][1], mean(edges_f1), np.std(edges_f1)], 'NODES [MAX, MEAN, STD]': [best_result['node_micro_f1'], mean(nodes_micro), np.std(nodes_micro)]}


def entity_linking(args):

    # configs
    start_training = time.time()
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device, output_dir=TRAIN_SAMPLES)
        data.get_info()

        ss = KFold(n_splits=10, shuffle=True, random_state=cfg_train.seed)
        cv_indices = ss.split(data.graphs)
        
        models = []
        train_index, val_index = next(ss.split(data.graphs))

        for cvs in cv_indices:

            train_index, val_index = cvs

            # TRAIN
            train_graphs = [data.graphs[i] for i in train_index]
            tg = dgl.batch(train_graphs)
            tg = tg.int().to(device)
    
            val_graphs = [data.graphs[i] for i in val_index]
            vg = dgl.batch(val_graphs)
            vg = vg.int().to(device)
        
            ################* STEP 1: CREATE MODEL ################
            model = sm.get_model(None, 2, data.get_chunks())
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
            # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=100, min_lr=1e-3, verbose=True, factor=0.01)
            # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            e = datetime.now()
            train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
            models.append(train_name+'.pt')
            stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=1000)
            # writer = SummaryWriter(log_dir=RUNS)
            # convert_imgs = transforms.ToTensor()
        
            ################* STEP 2: TRAINING ################
            print("\n### TRAINING ###")
            print(f"-> Training samples: {tg.batch_size}")
            print(f"-> Validation samples: {vg.batch_size}\n")

            # im_step = 0
            for epoch in range(cfg_train.epochs):

                #* TRAINING
                model.train()
                
                scores = model(tg, tg.ndata['feat'].to(device))
                loss = compute_crossentropy_loss(scores.to(device), tg.edata['label'].to(device))
                auc = compute_auc_mc(scores.to(device), tg.edata['label'].to(device))

                optimizer.zero_grad()
                loss.backward()
                n = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                #* VALIDATION
                model.eval()
                with torch.no_grad():
                    val_scores = model(vg, vg.ndata['feat'].to(device))
                    val_loss = compute_crossentropy_loss(val_scores.to(device), vg.edata['label'].to(device))
                    val_auc = compute_auc_mc(val_scores.to(device), vg.edata['label'].to(device))
                
                # scheduler.step(val_auc)
                # scheduler.step()

                #* PRINTING IMAGEs AND RESULTS

                print("Epoch {:05d} | TrainLoss {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValAUC-PR {:.4f} |"
                .format(epoch, loss.item(), auc, val_loss.item(), val_auc))
                
                if cfg_train.stopper_metric == 'loss':
                    step_value = val_loss.item()
                elif cfg_train.stopper_metric == 'acc':
                    step_value = val_auc
                
                ss = stopper.step(step_value)

                # if  ss == 'improved':
                #     im_step = epoch
                #     train_imgs = []
                #     for r in rand_tid:
                #         start, end = 0, 0
                #         for tid in train_index:
                #             start = end
                #             end += data.graphs[tid].num_edges()
                #             if tid == r: break

                #         _, targets = torch.max(F.log_softmax(scores[start:end], dim=1), dim=1)
                #         kvp_ids = targets.nonzero().flatten().tolist()
                #         train_imgs.append(convert_imgs(data.print_graph(num=r, labels_ids=kvp_ids, name=f'train_{r}'))[:, :, :700])
                #         # data.print_graph(num=r, name=f'train_labels_{r}')

                #     val_imgs = []
                #     for r in rand_vid:
                #         v_start, v_end = 0, 0
                #         for vid in val_index:
                #             v_start = v_end
                #             v_end += data.graphs[vid].num_edges()
                #             if vid == r: break

                #         _, val_targets = torch.max(F.log_softmax(val_scores[v_start:v_end], dim=1), dim=1)
                #         val_kvp_ids = val_targets.nonzero().flatten().tolist()
                #         val_imgs.append(convert_imgs(data.print_graph(num=r, labels_ids=val_kvp_ids, name=f'val_{r}'))[:, :, :700])
                        # data.print_graph(num=r, name=f'val_labels_{r}')

                if ss == 'stop':
                    break

                # writer.add_scalars('AUC-PR', {'train': auc, 'val': val_auc}, epoch)
                # writer.add_scalars('LOSS', {'train': loss.item(), 'val': val_loss.item()}, epoch)
                # writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

                # train_grid = torchvision.utils.make_grid(train_imgs)
                # writer.add_image('train_images', train_grid, im_step)
                # val_grid = torchvision.utils.make_grid(val_imgs)
                # writer.add_image('val_images', val_grid, im_step)
        
        # print("LOADING: ", train_name+'.pt')
        # model.load_state_dict(torch.load(WEIGHTS / (train_name+'.pt')))
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        models = args.weights
        
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = device, output_dir=TEST_SAMPLES)
    test_data.get_info()
    model = sm.get_model(None, 2, test_data.get_chunks())
    best_model = ''
    pair_scores = []
    test_graph = dgl.batch(test_data.graphs).to(device)


    for m in models:
        model.load_state_dict(torch.load(CHECKPOINTS / m))
        model.eval()
        with torch.no_grad():

            scores = model(test_graph, test_graph.ndata['feat'].to(device))
            auc = compute_auc_mc(scores.to(device), test_graph.edata['label'].to(device))
            
            _, preds = torch.max(F.softmax(scores, dim=1), dim=1)

            accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
            _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)

            pair_scores.append(classes_f1[1])
            if classes_f1[1] >= max(pair_scores):
                best_model = m

        ################* STEP 4: RESULTS ################
        print(f"\n### RESULTS {m} ###")
        print("F1 Score: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
        print("F1 Classes: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))

    print(f"\nLoading best model {best_model}")
    model.load_state_dict(torch.load(CHECKPOINTS / best_model))
    model.eval()
    with torch.no_grad():

        scores = model(test_graph, test_graph.ndata['feat'].to(device))
        auc = compute_auc_mc(scores.to(device), test_graph.edata['label'].to(device))
        
        _, preds = torch.max(F.softmax(scores, dim=1), dim=1)
        test_graph.edata['preds'] = preds

        accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)

    if not args.test:
        feat_n, feat_e = get_features(args)
        #? if skipping training, no need to save anything
        model = get_config(CFGM / args.model)
        results = {'MODEL': {
            'name': sm.get_name(),
            'weights': best_model,
            'net-params': sm.get_total_params(), 
            'num-layers': model.num_layers,
            'projector-output': model.out_chunks,
            'dropout': model.dropout,
            'lastFC': model.hidden_dim
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'RESULTS': {
                'val-loss': stopper.best_score, 
                'f1-scores': f1,
		        'f1-classes': classes_f1, 
                'AUC-PR': auc,
                'ACCURACY': accuracy,
                'std-pairs': np.std(pair_scores),
                'mean-pairs': mean(pair_scores)
            }}
        save_test_results(train_name, results)
        print("END TRAINING:", time.time() - start_training)

    return {'best_model': best_model, 'Pairs-F1': {'max': max(pair_scores), 'mean': mean(pair_scores), 'std': np.std(pair_scores)}}

def train_funsd(args):
    if args.model == 'e2e':
        e2e(args)
    elif args.model == 'e2e_char_embed':
        e2e_char_embed(args)
    elif args.model == 'edge':
        entity_linking(args)
    else:
        raise Exception("Model selected does not exists. Choose 'e2e' or 'edge'.")
    return

