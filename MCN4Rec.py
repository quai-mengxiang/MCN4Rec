import math
from os import device_encoding
import random
from this import d
from turtle import forward, position
from unicodedata import category
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

device = torch.device('cuda')

"""-------------------------Multi-level comparison interaction part----------------------------------"""

"""The resulting representations need to be aggregated between different levels of views to implement an aggregation class."""

class Aggregator(nn.Module):
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.n_users = n_users
    
    """Representation of users and categories Index and type of edges in Y knowledge graph of interaction matrix between user and poi"""
    def forward(self, category_emb, user_emb, edge_index, edge_type, interact_mat, weight):
        
        n_categories = category_emb.shape[0]

        """Aggregation of knowledge graph Head relation tail makes up the triple of KG"""
        head, tail = edge_index.to(device, dtype=torch.int64)
        edge_relation_emb = weight[edge_type.to(dtype=torch.int64) - 1] #排除交互 进行一个新的映射[1, n_relations)->[0, n_relations-1)
        edge_relation_emb = edge_relation_emb.to(device)
        neigh_relation_emb = category_emb[tail] * edge_relation_emb
        neigh_relation_emb = neigh_relation_emb.to(device)
        """Calculate the attention weight"""
        neigh_relation_emb_weight = self.calculate_sim_hrt(category_emb[head], category_emb[tail], weight[edge_type.to(dtype=torch.int64) - 1]).to(device)
        neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0], neigh_relation_emb.shape[1])
        # neigh_relation_emb_weight = neigh_relation_emb_weight.to(dtype=torch.int32)
        # neigh_relation_emb = neigh_relation_emb.to(dtype=torch.int32)
        neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, neigh_relation_emb.to(dtype=torch.int64)).to(device)
        neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb).to(device) #两个tensor对应位相乘
        category_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_categories, dim=0).to(device)

        user_agg = torch.sparse.mm(interact_mat, category_emb).to(device) #稀疏矩阵乘法 第一个为稀疏矩阵
        # user_agg = user_agg + user_emb * user_agg
        score = torch.mm(user_emb, weight.t()) #user_emb与weight的转置做矩阵乘法
        score = torch.softmax(score, dim=-1)
        score = score.to(device)
        user_agg  = user_agg + (torch.mm(score, weight)) * user_agg
        user_agg = user_agg.to(device)

        return category_agg, user_agg



    def calculate_sim_hrt(self, category_emb_head, category_emb_tail, relation_emb):
        
        tail_relation_emb = category_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True) #对输入的Tensor求范数
        head_relation_emb = category_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

"""Implement GCN"""
class GraphConv(nn.Module):
    def __init__(self, channel, n_hops, n_users, n_relations, interact_mat, ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):

        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.topk = 2
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        # self.device = torch.device("cuda:" + str(0))      
        self.device = torch.device("cuda")       

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight) 

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)

        
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float() 
        return torch.sparse.FloatTensor(i, v, coo.shape)   

    def forward(self, user_emb, category_emb, edge_index, edge_type,interact_mat, mess_dropout=True, node_dropout=False):
        edge_index, edge_type = edge_index.to(dtype=torch.float32), edge_type.to(dtype=torch.float32)
        """node dropout"""     
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        edge_index.to(device)
        edge_type.to(device)
        interact_mat.to(device)
        """Establish a poi-poi diagram"""
        origin_poi_adj = self.build_adj(category_emb, self.topk)
        origin_poi_adj = origin_poi_adj.to(device)

        category_res_emb = category_emb  # [n_category, channel]
        user_res_emb = user_emb  # [n_users, channel]

        for i in range(len(self.convs)):
            user_emb = user_emb.to(dtype=torch.float32)
            category_emb, user_emb = self.convs[i](category_emb, user_emb,edge_index, edge_type, interact_mat, self.weight)

            """message dropout"""
            if mess_dropout:
                category_emb = self.dropout(category_emb)
                user_emb = self.dropout(user_emb)
            category_emb = F.normalize(category_emb) #进行归一化处理
            user_emb = F.normalize(user_emb)   

            """result emb"""
            category_res_emb = torch.add(category_res_emb, category_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            category_res_emb.to(device)
            user_res_emb.to(device)

            """Update poi-poi diagrams"""
        poi_adj = (1 - self.lambda_coeff) * self.build_adj(category_res_emb,
                   self.topk) + self.lambda_coeff * origin_poi_adj
        poi_adj.to(device)

        return category_res_emb, user_res_emb, poi_adj

    def build_adj(self, context, topk):
        """Constructing similarity adjacency matrix"""
        n_category = context.shape[0]
        context = context.to(dtype=torch.float32)
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).to(self.device) 
        context = context.to(dtype=torch.int32)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0)) 
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
        knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)

        y = knn_ind.reshape(-1) 
        x = torch.arange(0, n_category).unsqueeze(dim=-1).to(self.device) 
        x = x.expand(n_category, topk).reshape(-1) 
        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_category, n_category])).to(self.device) #按照索引和值 形成固定size的稀疏矩阵

        # normalized laplacian adj
        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5) 
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_category).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_category)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_category, n_category]))
        L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt)
        return L_norm



class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        #--------------------------------
        self.n_pois = data_config['n_pois']       
        self.n_relations = data_config['n_relations']
        #--------------------------------
        self.n_categories = data_config['n_categories']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        # self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda else torch.device("cpu")        
        self.device = torch.device("cuda")
        # self.device = self.n_pois.device
    
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = self._init_model()
        self.lightgcn_layer = 2
        self.n_poi_layer = 1
        self.alpha = 0.2
        #Define network-related operation modules using Sequential
        self.fc1 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )

    # @property
    # def device(self):
    #     return self.n_pois.device

    def forward(self, batch = None):
        device = self.device
        self.all_embed.to(device)
        user = batch['users'].to(device)
        poi = batch['pois'].to(device)
        labels = batch['labels'].to(device)
        user_emb = self.all_embed[:self.n_users, :].to(dtype=torch.float32).to(device)
        user_emb = user_emb.to(device)
        poi_emb = self.all_embed[self.n_users:, :].to(dtype=torch.float32).to(device)
        poi_emb = poi_emb.to(device)
        category_gcn_emb, user_gcn_emb, poi_adj = self.gcn(user_emb, poi_emb, self.edge_index.to(device), 
            self.edge_type.to(device), self.interact_mat.to(device), mess_dropout = self.mess_dropout, node_dropout = self.node_dropout)
        category_gcn_emb, user_gcn_emb, poi_adj = category_gcn_emb.to(device), user_gcn_emb.to(device), poi_adj.to(device)
        u_e = user_gcn_emb[user.to(dtype=torch.int64)]
        i_e = category_gcn_emb[poi.to(dtype=torch.int64)]
        i_h = poi_emb
        for i in range(self.n_poi_layer):
            i_h = torch.sparse.mm(poi_adj, i_h)
        i_h = F.normalize(i_h, p=2, dim=1)
        i_e_1 = i_h[poi.to(dtype=torch.int64)]


        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()
        x = indice_old[0, :]
        y = indice_old[1, :]
        x_A = x
        y_A = y + self.n_users
        x_A_T = y + self.n_users
        y_A_T = x
        x_new = torch.cat((x_A, x_A_T), dim=-1)
        y_new = torch.cat((y_A, y_A_T), dim=-1)
        indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        value_new = torch.cat((value_old, value_old), dim=-1)
        interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size([self.n_users + self.n_categories, self.n_users + self.n_categories]))
        user_lightgcn_emb, poi_lightgcn_emb = self.gcn_(user_emb.to(device), poi_emb.to(device), interact_graph.to(device))
        u_e_2 = user_lightgcn_emb[user.to(dtype=torch.int64)]
        i_e_2 = poi_lightgcn_emb[poi.to(dtype=torch.int64)]


        poi_1 = poi_emb[poi.to(dtype=torch.int64)]
        user_1 = user_emb[user.to(dtype=torch.int64)]
        loss_contrast = self.calculate_loss(i_e_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_1(poi_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_2(user_1, u_e_2)

        u_e = torch.cat((u_e, u_e_2, u_e_2), dim=-1)
        i_e = torch.cat((i_e, i_e_1, i_e_2), dim=-1)
        return i_e, u_e, self.create_bpr_loss(u_e, i_e, labels, loss_contrast)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).long().to(self.device)

    def _init_model(self):
        return GraphConv(channel = self.emb_size, n_hops = self.context_hops, n_users = self.n_users,
                        n_relations = self.n_relations, interact_mat = self.interact_mat, ind = self.ind,
                        node_dropout_rate = self.node_dropout_rate, mess_dropout_rate = self.mess_dropout_rate)


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo() 
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape).long()

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges)) 
        index = graph_tensor[:, :-1]  
        type = graph_tensor[:, -1]  
        return index.t().long().to(self.device), type.long().to(self.device)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding):
        tau = 0.6    
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())) 

        ret = loss_1
        ret = ret.mean() 
        return ret


    def calculate_loss_1(self, A_embedding, B_embedding):
        tau = 0.6   
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))
        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret


    def calculate_loss_2(self, A_embedding, B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret  


    def gcn_(self, user_embedding, poi_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, poi_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj.to(dtype=torch.float32), ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1) 
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_categories], dim=0) 
        return u_g_embeddings, i_g_embeddings     


    def create_bpr_loss(self, users, pois, labels, loss_contrast):
        batch_size = users.shape[0]
        scores = (pois * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pois) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        return bce_loss + emb_loss + 0.001*loss_contrast


"""-----------------------Time-aware category context embedding part--------------------------"""

class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings = num_users,
            embedding_dim = embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed

class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings = num_cats,
            embedding_dim = embedding_dim,
        )
    
    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed

class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x
    
def t2v(tau, f, out_features, w, b, w0, b0, arg = None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1)) #rand是生成数符合均匀分布[0,1) randn符合标准正态分布
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin
    
    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        else:
            self.l1 = CosineActivation(1, out_dim)
        
    def forward(self, x):
        x = self.l1(x)
        return x

class TimeEmbeddings(nn.Module):
    def __init__(self, num_time_slots, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.time_embedding = nn.Embedding(
            time_embeddings = num_time_slots,
            embedding_dim = embedding_dim,
        )

    def forward(self, time_idx):
        embed = self.time_embedding(time_idx)
        return embed
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x[:, :1, :] + self.pe[:x.size(0), :, :x.size(2)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(384, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(384, num_poi)
        self.decoder_time = nn.Linear(384, 1)
        self.decoder_cat = nn.Linear(384, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat
















