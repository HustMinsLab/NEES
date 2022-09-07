from typing import Union, Optional, Callable
from collections import namedtuple
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_sparse import coalesce
from torch_geometric.nn import LEConv,GCNConv
from torch_geometric.utils import softmax, to_dense_adj, add_remaining_self_loops

from collections import defaultdict
import math
from torch.autograd import Variable


import numpy as np

from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing


class simiConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(simiConv, self).__init__(aggr='add') #'mean'

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Linear(in_channels, in_channels, bias=False)
        self.lin2 = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, edge_index):
        """"""
        a = self.lin1(x)
        b = self.lin2(x)
        out = self.propagate(edge_index, x=a, x_cluster=b)
        # return out + b
        return out#.sigmoid()


    def message(self, x_i, x_j, x_cluster_i):

        out = torch.cosine_similarity(x_i, x_j).reshape(-1,1)
        print(x_i.shape, out.shape)
        return x_cluster_i * out
        # return  out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class NEWPooling6(torch.nn.Module):
    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster"])

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.01,
                 GNN: Optional[Callable] = GCNConv, dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = False,
                 **kwargs):
        super(NEWPooling6, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.heads = 6
        self.weight = Parameter(
            torch.Tensor(in_channels, self.heads * in_channels))
        self.attention = Parameter(torch.Tensor(1, self.heads, 2 * in_channels))
        # self.attention1 = Parameter(torch.Tensor(1, self.heads, in_channels))
        self.use_attention = True

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, in_channels)
        self.gnn_score = simiConv(self.in_channels, 1)#LEConv(self.in_channels, 1)
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.CosineLoss = nn.CosineEmbeddingLoss(margin=0.2)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def glorot(self, tensor):  # inits.py中
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.glorot(self.weight)
        self.glorot(self.attention)
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def gen_subs(self, edge_index, N):
        edgelists = defaultdict(list)
        match = defaultdict(list)
        for i in range(edge_index.size()[1]):
            s = int(edge_index[0][i])
            t = int(edge_index[1][i])
            if s != t:
                edgelists[s].append(t)
        # print(len(list(edgelists.keys())))
        start = []
        end = []
        top = []
        for i in range(N):
            start.append(i)
            end.append(i)
            match[i].append(i)
            if i in edgelists.keys():
                for j in edgelists[i]:
                    set1 = set(edgelists[i])
                    set2 = set(edgelists[j])
                    d = len(set1.intersection(set2))
                    # print(set1, set2)
                    if d > 0:
                        start.append(j)
                        end.append(i)
                        match[i].append(j)
            '''motif+neighbor'''
            if len(match[i]) == 1:
                match[i].extend(edgelists[i])
                start.extend(edgelists[i])
                end.extend([i] * len(edgelists[i]))
            else:
                top.append(i)

        source_nodes = torch.Tensor(start).reshape((1, -1))
        target_nodes = torch.Tensor(end).reshape((1, -1))
        subindex = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)

        return subindex,match, edgelists, top

    def choose(self, x, x_pool,edge_index, subindex, batch, score, match, top):
        nodes_remaining = set(range(x.size(0)))
        # print(x.size(0))
        cluster = torch.empty_like(batch, device=torch.device('cpu'))


        node_argsort = torch.argsort(score, descending=True)

        i = 0
        transfer = {}
        new_node_indices = []
        loss = 0
        tar_in = []
        tar_tar = []
        for node_idx in node_argsort.tolist():#sort_node
            source = match[node_idx]

            d = [True for c in source if c not in nodes_remaining]
            if d:
                # print(1)
                continue

            transfer[i] = node_idx

            new_node_indices.append(node_idx)
            for j in source:
                cluster[j] = i
                if j != node_idx:
                    tar_in.append(j)
                    tar_tar.append(i)

            nodes_remaining = [j for j in nodes_remaining if j not in source]

            i += 1




        for node_idx in nodes_remaining:

            cluster[node_idx] = i
            transfer[i] = node_idx
            i += 1

        cluster = cluster.to(x.device)

        index = new_node_indices + nodes_remaining
        new_x_pool = x_pool[index, :]

        new_x = torch.cat([x[new_node_indices,:],x_pool[nodes_remaining,:]])


        new_score = score[new_node_indices]

        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_node_indices),))
            new_score = torch.cat([new_score, remaining_score])
        new_x = new_x * new_score.view(-1, 1)
        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        edge_score = []
        for i in range(len(new_edge_index[0])):
            s = float(score[transfer[int(new_edge_index[0][i])]]) + float(score[transfer[int(new_edge_index[1][i])]])

            edge_score.append(s)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster)

        inputs = x_pool[tar_in]
        targets = torch.Tensor(tar_tar)



        pos = []
        anchor_pos = []
        neg = []
        anchor_neg = []
        sig = {}
        for idx in range(x.size(0)):
            sig[idx] = []
            if cluster[idx].item() in range(len(new_node_indices)):
                pos.append(idx)
                anchor_pos.append(cluster[idx].item())
                for j in match[idx]:
                    if j != idx and cluster[j] != cluster[idx] and cluster[j].item() not in sig[idx]:
                        neg.append(idx)
                        anchor_neg.append(cluster[j].item())
                        sig[idx].append(cluster[j].item())

        pos_pos = x_pool[pos]
        pos_anchor = new_x[anchor_pos]
        neg_neg = x_pool[neg]
        neg_anchor = new_x[anchor_neg]

        return new_x,new_x_pool, new_edge_index, edge_score, unpool_info,cluster, pos_pos,pos_anchor, neg_neg,neg_anchor,inputs,targets#,center, summary,

    def del_tensor_ele(self,arr, index):
        arr1 = arr[0:index]
        arr2 = arr[index + 1:]
        return torch.cat((arr1, arr2), dim=0)

    def similarity(self, x, edgelist):
        score = torch.Tensor(size=([x.size(0)]))
        for k,v in edgelist.items():
            c = x[v]
            score[k] = torch.mean(torch.std(c, dim=0).view(1,-1))

        return score

    def forward(self, x, edge_index, batch=None):
        N = x.size(0)
        if N == 1:
            unpool_info = self.unpool_description(edge_index=edge_index,
                                                  cluster=torch.tensor([0]))
            return x, edge_index, unpool_info, torch.tensor(0.0,requires_grad=True), 0.0

        edge_index, _ = add_remaining_self_loops(edge_index, fill_value=1, num_nodes=N)

        if batch is None:
            batch = torch.LongTensor(size=([N]))

        subindex, match, edgelist, top = self.gen_subs(edge_index, N)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x_pool = x
        if self.GNN is not None:
            print("GNN")
            x_pool_j = self.gnn_intra_cluster(x=x, edge_index=edge_index)
        print("x_pool",x_pool.size())



        if self.use_attention:
            x_pool_j = torch.matmul(x_pool_j, self.weight)

            x_pool_j = x_pool_j.view(-1, self.heads, self.in_channels)

            x_i = x_pool_j[subindex[0]]


            x_j = scatter(x_i, subindex[1], dim=0, reduce='max')

            alpha = (torch.cat([x_i, x_j[subindex[1]]], dim=-1) * self.attention).sum(dim=-1)

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, subindex[1], num_nodes=x_pool_j.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            v_j = x_pool_j[subindex[0]] * alpha.view(-1, self.heads, 1)

            x = scatter(v_j, subindex[1], dim=0, reduce='add')

            x = x.mean(dim=1)

        fitness = self.gnn_score(x, subindex).sigmoid().view(-1)

        x, new_x_pool, edge_index, s, unpool_info,cluster, pos_pos,pos_anchor, neg_neg,neg_anchor, inputs,targets = self.choose(x, x_pool, edge_index,subindex, batch, fitness, match, top)


        loss = self.BCEloss(pos_pos, pos_anchor, neg_neg, neg_anchor)


        return x, edge_index, unpool_info, loss, s


    def BCEloss(self,pos_anchor, pos, neg_anchor,neg):
        n1, h1 = pos_anchor.size()
        n2, h2 = neg_anchor.size()

        TotalLoss = 0.0
        pos = torch.bmm(pos_anchor.view(n1, 1, h1), pos.view(n1, h1, 1))
        loss1 = self.BCEWloss(pos, torch.ones_like(pos))
        if neg_anchor.size()[0] != 0:
            neg = torch.bmm(neg_anchor.view(n2, 1, h2), neg.view(n2, h2, 1))
            loss2 = self.BCEWloss(neg, torch.zeros_like(neg))
        else:
            loss2 = 0

        TotalLoss += loss2 + loss1
        return TotalLoss
