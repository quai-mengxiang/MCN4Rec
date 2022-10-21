from symbol import parameters
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from param_parser import parse_args
import multiprocessing
import heapq
from pathlib import Path
import re
import glob
import os
from time import time
from scipy.sparse.linalg import eigsh

cores = multiprocessing.cpu_count()
args = parse_args()

Ks = eval(args.Ks)

# device = torch.device("cuda:" + str(args.gpu_id))
device = torch.device("cuda")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag

def get_auc(poi_score, user_pos_test):
    poi_score = sorted(poi_score.items(), key=lambda kv: kv[1])
    poi_score.reverse()
    poi_sort = [x[0] for x in poi_score]
    posterior = [x[1] for x in poi_score]

    r = []
    for i in poi_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_heapq(user_pos_test, test_pois, rating, Ks):
    poi_score = {}
    for i in test_pois:
        poi_score[i] = rating[i]

    K_max = max(Ks)
    K_max_poi_score = heapq.nlargest(K_max, poi_score, key=poi_score.get)

    r = []
    for i in K_max_poi_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(poi_score, user_pos_test)
    return r, auc

def ranklist_by_sorted(user_pos_test, test_pois, rating, Ks):
    poi_score = {}
    for i in test_pois:
        poi_score[i] = rating[i]

    K_max = max(Ks)
    K_max_poi_score = heapq.nlargest(K_max, poi_score, key=poi_score.get)

    r = []
    for i in K_max_poi_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(poi_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test_one_user(x):
    rating = x[0]
    u = x[1]
    try:
        training_pois = train_user_set[u]
    except Exception:
        training_pois = []
    user_pos_test = test_user_set[u]

    all_pois = set(range(0, n_pois))

    test_pois = list(all_pois - set(training_pois))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_pois, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_pois, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_pois
    n_pois = n_params['n_pois']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)
    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0
    category_gcn_emb, user_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_poi_batchs = n_pois // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_pois))

            i_count = 0
            for i_batch_id in range(n_poi_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_pois)

                poi_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                i_g_embeddings = category_gcn_emb[poi_batch]

                i_rate_batch = torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().to(device)

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_pois
        else:
            poi_batch = torch.LongTensor(np.array(range(0, n_pois))).view(n_pois, -1).to(device)
            i_g_embddings = category_gcn_emb[poi_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().to(device)

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
    assert count == n_test_users
    pool.close()
    return result

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant). 
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] 
    return np.mean(r)

def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # 早停策略:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[(~mask).long()] - target[(~mask).long()]) ** 2
    loss = out.mean()
    return loss

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0

def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    #MRR
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)  
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2 
        return f"{path}{sep}{n}"  

def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)