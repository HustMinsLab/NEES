from __future__ import division
from __future__ import print_function
import time

import torch.nn.functional as F
import torch.optim as optim
from utils import *
import random
from new_pooling import *
import networkx as nx

import copy

from torch_geometric.nn import DenseSAGEConv, GCNConv, DenseGraphConv, dense_mincut_pool, dense_diff_pool, EdgePooling
from torch_geometric.utils import to_dense_batch, to_dense_adj

random.seed(18)
class Net(torch.nn.Module):
    def __init__(self, nodes, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)


        self.pool1 = NEWPooling6(out_channels)
        self.pool2 = NEWPooling6(out_channels)
        self.pool3 = NEWPooling6(out_channels)

    def edgeIndex(self, G):
        source_nodes = []
        target_nodes = []
        for e in list(G.edges()):
            n1 = int(e[0])
            n2 = int(e[1])
            source_nodes.append(n1)
            source_nodes.append(n2)
            target_nodes.append(n2)
            target_nodes.append(n1)
        source_nodes = torch.Tensor(source_nodes).reshape((1, -1))
        target_nodes = torch.Tensor(target_nodes).reshape((1, -1))
        edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)

        return edge_index

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, G, x, adj):
        x, mask = to_dense_batch(x)

        edge_index = self.edgeIndex(G)

        x = F.relu(self.conv1(x=x, edge_index=edge_index))
        x = F.relu(self.conv2(x=x, edge_index=edge_index))
        x = F.relu(self.conv3(x=x, edge_index=edge_index))

        x1, edge_index1, unpool_info1,loss1, score1  = self.pool1(x=x.squeeze(0), edge_index=edge_index)

        x2, edge_index2, unpool_info2,loss2, score2 = self.pool2(x=x1, edge_index=edge_index1)

        x3, edge_index3, unpool_info3, loss3, score3 = self.pool3(x=x2, edge_index=edge_index2)

        return (edge_index1,score1,unpool_info1), (edge_index2,score2, unpool_info2), (edge_index3,score3, unpool_info3),loss1+loss2+loss3#, self.dc(z), mu, logvar


def LGC(G):
    largest_components = 0
    GCC = 0
    for c in nx.connected_components(G):
        g = G.subgraph(c)
        m = len(list(g.nodes()))
        if m > largest_components:
            largest_components = m
            GCC = g
    return GCC,largest_components


def createGraph(file_dir,filename, N):
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for line in open(file_dir+filename):
        strlist = line.split()
        n1 = int(strlist[0])-1
        n2 = int(strlist[1])-1
        G.add_edges_from([(n1, n2)])
    return G

def gene_features(G,feats):

    N = list(G.nodes())
    i = len(N) % feats
    j = int((len(N) - i) / feats)

    if j == 0:
        feature = torch.eye(feats)[:i, :]
    else:
        feature = torch.eye(feats)

        for k in range(j - 1):
            feature = torch.cat((feature, torch.eye(feats)), 0)
        feature = torch.cat((feature, torch.eye(feats)[:i, :]), 0)

    return feature

def Trans(G, s, adj):
    s = torch.softmax(s, dim=-1)
    #print(s)
    _, X = torch.max(s, 2)
    X = X.numpy().tolist()[0]
    transfer0 = {}
    transfer0_back = {}
    for i in range(len(X)):
        if X[i] not in transfer0.keys():
            transfer0[X[i]] = []
        transfer0[X[i]].append(i)
        transfer0_back[i] = X[i]


    G0 = nx.from_numpy_matrix(adj.squeeze(0).detach().numpy())
    real_nodes1 = [i for i in list(range(s.size()[-1])) if i not in list(transfer0.keys())]
    G1 = copy.deepcopy(G0)
    G1.remove_nodes_from(real_nodes1)

    real_edges1 = []
    for (u, v) in G.edges():
        if transfer0_back[u] != transfer0_back[v]:
            real_edges1.append((transfer0_back[u], transfer0_back[v]))

    now = G1.edges()

    for (u, v) in now:
        if (u, v) and (v, u) not in real_edges1:
            G1.remove_edge(u, v)

    return G0, G1, transfer0, transfer0_back



def Trans_back(G, edge_index, score, set, cluster,essence):
    back = {}
    for i in range(len(cluster)):
        if int(cluster[i]) in back.keys():
            back[int(cluster[i])].append(i)
        else:
            back[int(cluster[i])] = [i]
    print(cluster, back)
    for id in range(edge_index.size()[1]):
        s = int(edge_index[0][id])
        t = int(edge_index[1][id])
        if s in list(set.keys()) and t in list(set.keys()) and s != t:
            flags = 0
            flagt = 0
            if len(back[s]) > 1:
                flags = 1
            if len(back[t]) > 1:
                flagt = 1
            if flags or flagt:
                for i in set[s]:
                    for j in list(G.neighbors(i)):
                        if j in set[t]:
                            G[i][j]["weight"] += score[id]
                            if flags:
                                essence.append(i)
                            if flagt:
                                essence.append(j)

    return G


def Test(G, feature, adj, model):
    (edge_index1, score1, unpool_info1), (edge_index2, score2, unpool_info2), (
    edge_index3, score3, unpool_info3), loss = model(G, feature, adj)

    set1 = {}
    set2 = {}
    set3 = {}

    for i in G.nodes():
        i1 = int(unpool_info1.cluster[i])
        i2 = int(unpool_info2.cluster[i1])
        i3 = int(unpool_info3.cluster[i2])

        if i1 not in set1.keys():
            set1[i1] = []
        if i2 not in set2.keys():
            set2[i2] = []
        if i3 not in set3.keys():
            set3[i3] = []

        set1[i1].append(i)
        set2[i2].append(i)
        set3[i3].append(i)

    for (u, v) in G.edges():
        G[u][v]["weight"] = float(1.0)

    essence = []
    G = Trans_back(G, edge_index1, score1, set1,unpool_info1.cluster,essence)

    G = Trans_back(G, edge_index2, score2, set2,unpool_info2.cluster,essence)

    G = Trans_back(G, edge_index3, score3, set3,unpool_info3 .cluster,essence)
    essence = set(essence)

    N = list(G.nodes())
    G0 = copy.deepcopy(G)
    totalnum = nx.number_of_nodes(G0)
    GCC, largest_components = LGC(G0)
    d = {}
    for i in N:
        d[i] = 0

    for (u, v, w) in G0.edges(data=True):

        d[u] += 0.5 * w['weight']
        d[v] += 0.5 * w['weight']

    rank = sorted(d.items(), key=lambda item: item[1],
                  reverse=True)

    rank_ = [i[0] for i in rank]

    D = []

    dict = {}
    for (u, v, d) in G0.edges(data=True):
        dict[(u, v)] = d['weight']

    sort_list = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    while largest_components > 0.01 * len(N):

        subset = [sort_list[0][0][0], sort_list[0][0][1]]
        for i in range(1, len(sort_list)):
            if sort_list[i][1] == sort_list[0][1]:
                subset.extend([sort_list[i][0][0], sort_list[i][0][1]])
        subset = set(subset)
        sub = list(set(subset) & essence)
        minIndex = len(N)

        if sub == []:
            for s in subset:
                a = rank_.index(s)
                if a < minIndex:
                    minIndex = a
        else:
            for s in sub:
                a = rank_.index(s)
                if a < minIndex:
                    minIndex = a


        G0.remove_node(rank_[minIndex])
        D.append(rank_[minIndex])

        GCC, largest_components = LGC(G0)

        dict = {}
        for (u, v, d) in G0.edges(data=True):
            dict[(u, v)] = d['weight']

        sort_list = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    return D, loss

def main():
    '''
    Adjecency matrix and modifications
    '''
    file_dir = "netdata/"
    FILE_NET = ["football.txt"]
    NODE_NUM = [115]


    ep = 210

    x = {}
    y = {}
    z = {}

    for f in range(len(FILE_NET)):


        model = Net(NODE_NUM[f], NODE_NUM[f], 64, 32)

        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        G = createGraph(file_dir, FILE_NET[f], NODE_NUM[f])
        A = nx.adjacency_matrix(G).todense()

        norm_adj = symnormalise(A)  # Normalization using D^(-1/2) A D^(-1/2)
        adj = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(norm_adj)).to_dense()

        feature = gene_features(G, NODE_NUM[f])
        n_nodes = list(G.nodes())

        x[f] = []
        y[f] = []
        z[f] = []
        min = len(n_nodes)

        for epoch in range(ep):
            t = time.time()
            model.train()
            G0 = copy.deepcopy(G)
            '''test'''
            D,  loss = Test(G0, feature, adj, model)

            x[f].append(epoch)
            y[f].append(len(D))

            z[f].append(loss)


            if len(D) <= min and epoch > 50:
                min = len(D)
                print("save")
                torch.save(model.state_dict(), "./result-trial-"+ FILE_NET[f][:-4] + ".pt")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", '%04d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(loss), "time=", "{:.5f}".format(time.time() - t))

        model.load_state_dict(torch.load("./result-trial-"+FILE_NET[f][:-4]+".pt"))#
        D, _ = Test(G, feature, adj, model)
        print(len(D))
        with open("./result-" + FILE_NET[f], "w") as h:
            for i in range(len(D)):
                h.write(str(D[i]) + "\n")
        h.close()


if __name__ == '__main__':
    main()