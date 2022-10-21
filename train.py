from faulthandler import is_enabled
import os
import gc
from this import d
import zipfile
import pickle
import pathlib
import random
from turtle import pen
import torch
import torch.nn as nn
import yaml
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from time import time
import logging
from param_parser import parse_args
from dataloader import load_data
from dataloader import load_graph_adj_mtx, load_graph_node_features
from MCN4Rec import Recommender
from sklearn.metrics import roc_auc_score, f1_score
from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from MCN4Rec import UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel, NodeAttnMap
import torch.optim as optim
from utils import calculate_laplacian_matrix, top_k_acc_last_timestep, zipdir, mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss



n_users = 0
n_pois = 0
n_categories = 0
n_nodes = 0
n_relations = 0

def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.float32)
    train_entity_pairs = torch.FloatTensor(train_entity_pairs)
    train_entity_pairs.cpu()
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pois'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.FloatTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.float32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pois'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def _get_topk_feed_data(user, pois):
    res = list()
    for poi in pois:
        res.append([user, poi, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        poi = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(poi)
    return user_history_dict

def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:

        batch = get_feed_dict(data.cpu(), start, start + args.batch_size)
        labels = data[start:start + args.batch_size, 2]
        _, scores, _, _ = model(batch)
        scores = scores.detach().cpu().numpy() 
        auc = roc_auc_score(y_true=labels.cpu(), y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels.cpu(), y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1

def topk_eval(model, train_data, data):
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    poi_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_poi_list = list(poi_set-set(train_record[user]))
        poi_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_poi_list):
            pois = test_poi_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, pois)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for poi, score in zip(pois, scores):
                poi_score_map[poi] = score
            start += args.batch_size
        if start < len(test_poi_list):
            res_pois = test_poi_list[start:] + [test_poi_list[-1]] * (args.batch_size - len(test_poi_list) + start)
            input_data = _get_topk_feed_data(user, res_pois)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for poi, score in zip(res_pois, scores):
                poi_score_map[poi] = score
        poi_score_pair_sorted = sorted(poi_score_map.items(), key=lambda x: x[1], reverse=True)
        poi_sorted = [i[0] for i in poi_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(poi_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    return recall

def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)
    print('Loading POI graph...')
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = cat_id2idx_dict[row[args.feature2]]

    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    traj_list = list(set(train_df['trajectory_id'].tolist()))

    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]
                if user_id not in user_id2idx_dict.keys():
                    continue
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        continue

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_1, shuffle=True, drop_last=False, pin_memory=False, num_workers=args.workers, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_1, shuffle=False, drop_last=False, pin_memory=False, num_workers=args.workers, collate_fn=lambda x: x)

    p_u_embed_model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    device_ids=[0, 1, 2, 3]
    p_u_embed_model = nn.DataParallel(p_u_embed_model, device_ids=[0, 1, 2, 3])
    p_u_embed_model.to(device)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    num_users = len(user_id2idx_dict)
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)
    time_embed_model = nn.DataParallel(time_embed_model, device_ids=[0, 1, 2, 3])
    time_embed_model.to(device)

    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    cat_embed_model = nn.DataParallel(cat_embed_model, device_ids=[0, 1, 2, 3])
    cat_embed_model.to(device)
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model1 = nn.DataParallel(embed_fuse_model1, device_ids=[0, 1, 2, 3])
    embed_fuse_model1.to(device)

    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim_1, args.cat_embed_dim_1)
    embed_fuse_model2 = nn.DataParallel(embed_fuse_model2, device_ids=[0, 1, 2, 3])
    embed_fuse_model2.to(device)
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model = TransformerModel(num_pois, num_cats, args.seq_input_embed, args.transformer_nhead, args.transformer_nhid, args.transformer_nlayers,dropout=args.transformer_dropout)
    seq_model = nn.DataParallel(seq_model, device_ids=[0, 1, 2, 3])
    seq_model.to(device)
    optimizer = optim.Adam(params=list(p_u_embed_model.parameters()) + list(time_embed_model.parameters()) + list(cat_embed_model.parameters()) + list(embed_fuse_model1.parameters()) + list(embed_fuse_model2.parameters()) + list(seq_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1) 
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  
    criterion_time = maksed_mse_loss

    def input_traj_to_embeddings(sample, poi_embeddings, user_embeddings):

        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

        poi_embedding = poi_embeddings
        user_embedding = user_embeddings

        poi_embedding = torch.squeeze(poi_embedding, 0).to(device=args.device)
        poi_embedding = poi_embedding.to(device=args.device)

        user_embedding = torch.squeeze(user_embedding, 0).to(device=args.device)
        user_embedding = user_embedding.to(device=args.device)
        input_seq_embed = []
        for idx in range(len(input_seq)):
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)
            time_embedding = torch.unsqueeze(time_embedding, 0).to(device=args.device)
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    # ====================== Train ======================
    p_u_embed_model = p_u_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    max_val_score = -np.inf

    for epoch in range(args.epoch):
        p_u_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()
        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch_size_1).to(args.device)
        for b_idx, batch_1 in enumerate(train_loader):
            if len(batch_1) != args.batch_size_1:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch_1)).to(args.device)
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            s = 0
            while s + args.batch_size_view <= len(train_cf):
                batch = get_feed_dict(train_cf, s, s + args.batch_size_view)
                p_u_embed_model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
                poi_embeddings,user_embeddings, batch_loss = p_u_embed_model(batch)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.module
                s += args.batch_size_view
            for sample in batch_1:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings,user_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            batch_padded = batch_padded[:, :, :1, :]
            batch_padded = batch_padded.squeeze()
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            loss = loss_poi + loss_time + loss_cat
            optimizer.zero_grad()
            loss.backward()
            optimizer.module
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len] 
                pred_pois = pred_pois[:seq_len, :]  
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

        p_u_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch_size_1).to(args.device)
        for vb_idx, batch_1 in enumerate(val_loader):
            if len(batch_1) != args.batch_size_1:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch_1)).to(args.device)
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            
            while s + args.batch_size_view <= len(train_cf):
                batch = get_feed_dict(train_cf, s, s + args.batch_size_view)
                p_u_embed_model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
                poi_embeddings,user_embeddings, batch_loss = p_u_embed_model(batch)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.module
                s += args.batch_size_view
            for sample in batch_1:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings,user_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            batch_padded = batch_padded[:, :, :1, :]
            batch_padded = batch_padded.squeeze()
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi[:512, :1].to(device=args.device, dtype=torch.long)
            y_time = label_padded_time[:512, :1].to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat[:512, :1].to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss = loss_poi + loss_time  + loss_cat
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)




if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()

    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list
    n_users = n_params['n_users']
    n_pois = n_params['n_pois']
    n_categories = n_params['n_categories']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']


    """cf data (Collaborative Filter)"""
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))
    test_cf_pairs = test_cf_pairs.to(device)
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)

