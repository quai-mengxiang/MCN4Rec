import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import os
import pandas as pd
import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


n_user = 0
n_pois = 0
n_categories = 0
n_relations = 0
n_nodes = 0
# returns a dictionary object that receives list as a parameter, and instantiates a value as the default value when the accessed key does not exist
train_user_set = defaultdict(list)  
test_user_set = defaultdict(list)

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip() 
        inters = [int(i) for i in tmps.split(" ")] 

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)



def read_cf_new():
    rating_file = args.data_path + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy') 
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64) 
        np.save(rating_file + '.npy', rating_np)     
    test_ratio = 0.2
    n_ratings = rating_np.shape[0] 
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    train_indices = list(left) 
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]

    train_rating = rating_np[train_indices]
    ui_adj = generate_ui_adj(rating_np, train_rating) 
    return train_data, eval_data, ui_adj

def generate_ui_adj(rating, train_rating):
    n_user, n_poi = len(set(rating[:, 0])), len(set(rating[:, 1]))
    ui_adj_orign = sp.coo_matrix(
        (train_rating[:, 2], (train_rating[:, 0], train_rating[:, 1]))).todok()
    ui_adj = sp.bmat([[None, ui_adj_orign],
                    [ui_adj_orign.T, None]], dtype=np.float32)

    ui_adj = ui_adj.todok()
    print('already create user-poi adjacency matrix', ui_adj.shape)
    return ui_adj

def remap_poi(train_data, eval_data):
    global n_users, n_pois
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0])) + 1
    n_pois = max(max(train_data[:, 1]), max(eval_data[:, 1])) + 1
    eval_data_label = eval_data.take([2], axis=1)
    indix_click = np.where(eval_data_label == 1)
    eval_data = eval_data.take(indix_click[0], axis=0)
    eval_data = eval_data.take([0, 1], axis=1)
    train_data = train_data.take([0, 1], axis=1)
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id)) 
    for u_id, i_id in eval_data:
        test_user_set[int(u_id)].append(int(i_id))


#   
def read_triplets(file_name):
    global n_categories, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
    triplets = can_triplets_np.copy()
    
    n_categories = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including pois + users
    n_nodes = n_categories + n_users
    n_relations = max(triplets[:, 1]) + 1
    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])
    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd

def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users 
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    return adj_mat_list, norm_mat_list, mean_mat_list

def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-poi set ...')
    train_cf, eval_cf, ui_adj = read_cf_new()
    remap_poi(train_cf, eval_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_pois': int(n_pois),
        'n_categories': int(n_categories),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }
    return train_cf, eval_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list]



