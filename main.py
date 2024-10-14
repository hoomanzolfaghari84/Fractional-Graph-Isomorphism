import numpy as np
import torch

from classic_methods_experiments import run_knn_experiment, run_knn_experiment_multithread, run_svm_experiment
from datasets import get_m_shot_loaders, load_dataset
from distance_functions import isomorphism_distance_adjmatrix, isomorphism_distance_adjmatrix_constrained, subgraph_isomorphism_distance
from fewshot_experiments import run_fewshot_without_training
import cvxpy as cp
from torch_geometric.utils import to_networkx, to_dense_adj


import torch
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import statistics

# Function to visualize a graph
def visualize_graph(data, title):
    G = to_networkx(data, to_undirected=True)  # Convert to NetworkX graph
    plt.figure(figsize=(5, 5))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title(title)
    plt.show()

def run():

    # # Graph 1 (Top graph: Two squares sharing an edge)
    # edge_index_1 = torch.tensor([[0, 1, 2, 3, 2, 3, 4, 5],
    #                             [1, 2, 3, 0, 3, 4, 5, 2]], dtype=torch.long)

    # data_1 = Data(edge_index=edge_index_1, x=torch.ones((6,1)))

    # adj1 = to_dense_adj(data_1.edge_index, max_num_nodes=data_1.x.size(0)).squeeze(0)

    # # Graph 2 (Bottom graph: Two triangles connected by an edge)
    # edge_index_2 = torch.tensor([[0, 1, 2, 2, 3, 4, 5, 3],
    #                             [1, 2, 0, 3, 4, 5, 3, 2]], dtype=torch.long)

    # data_2 = Data(edge_index=edge_index_2, x=torch.ones((6,1)))

    # adj2 = to_dense_adj(data_2.edge_index, max_num_nodes=data_2.x.size(0)).squeeze(0)

    # # Visualize Graph 1 (Top)
    # visualize_graph(data_1, "Graph 1 (Top)")

    # # Visualize Graph 2 (Bottom)
    # visualize_graph(data_2, "Graph 2 (Bottom)")

    # subgraph_isomorphism_dist,X1,_ = subgraph_isomorphism_distance(data_1.x, adj1, data_2.x, adj2, 2, mapping='integral')


    # subgraph_isomorphism_dist_frac,X2,_ = subgraph_isomorphism_distance(data_1.x, adj1, data_2.x, adj2, 2, mapping='fractional')

    # print(f'int dist:{subgraph_isomorphism_dist}')
    # print(X1)
    # print(f'frac dist:{subgraph_isomorphism_dist_frac}')
    # print(X2)

    # torch.manual_seed(42)  # For reproducibility
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset('MUTAG')
    # data = dataset[0]
    # print(data)
    # print(data.x)
    # print(data.__dict__)
    # print(data.__dict__.keys())
    # i = 0
    # set = []
    # for data in dataset:
    #     if data.y == 0:
    #         i += 1
    #         set.append(data)
    #     if i == 50:
    #         break
    
    # dists = []
    # distsf= []
    # for i , data1 in enumerate(set):
    #     adj1 = to_dense_adj(data1.edge_index, max_num_nodes=data1.x.size(0)).squeeze(0)

    #     for j, data2 in enumerate(set):
    #         if j < i :continue

    #         adj2 = to_dense_adj(data2.edge_index, max_num_nodes=data2.x.size(0)).squeeze(0)
            
    #         d, _, _ = subgraph_isomorphism_distance(data1.x, adj1, data2.x, adj2, 2, mapping='integral')
    #         df, _,_ =subgraph_isomorphism_distance(data1.x, adj1, data2.x, adj2, 2, mapping='fractional')
    #         dists.append(d)
    #         distsf.append(df)

    # print(f'int mean {statistics.mean(dists)}')
    # print(f'int var {statistics.variance(dists)}')
    # print(f'frac mean {statistics.mean(distsf)}')
    # print(f'frac var {statistics.variance(distsf)}')

    # train_loader, val_loader, test_loader = get_m_shot_loaders(dataset, 5, val_split=0.03)

    run_knn_experiment(dataset, filename='MUTAG attrib')
    # dataset = dataset.shuffle()
    # n = 5

    # dists_int = np.ones((n,n))
    # dists_frac = np.ones((n,n))

    # sub = []
    # y = 5
    # num = 0
    # # for data in dataset:

    # for i , g1 in enumerate(dataset):
    #     if i >= n: break

    #     adj1 = to_dense_adj(g1.edge_index, max_num_nodes=g1.x.size(0)).squeeze(0)

    #     print(f'graph {i}:\n {g1}')
    #     # print(f'X:\n {g1.x}')
    #     print(f'y:\n {g1.y}')
    #     # print(f'adj:\n {adj1}')


    #     for j , g2 in enumerate(dataset):
    #         if j >= n: break
            
    #         adj2 = to_dense_adj(g2.edge_index, max_num_nodes=g2.x.size(0)).squeeze(0)

    #         dists_int[i,j], X, C = subgraph_isomorphism_distance_kernelized(g1.x, adj1, g2.x, adj2,  2 , mapping='integral')
    #         dists_frac[i,j], X, C = subgraph_isomorphism_distance_kernelized(g1.x, adj1, g2.x, adj2,  2 , mapping='fractional')
        
    #     indices = np.argpartition(dists_int[i], n-1)[:n]
    #     print(indices)
    #     print(f'nearest: {dataset[indices[1]].y}')


    # print(dists_int)
    # print(dists_frac)

    # # Two nodes and one edge between them
    # edge_index1 = torch.tensor([[0, 1, 1 , 2 , 0, 2],
    #                         [1, 0, 2 , 1, 2 , 0]], dtype=torch.long)
    # x1 = torch.ones((3, 1), dtype=torch.float)

    # # Define the graph
    # path_one = Data(edge_index=edge_index1, x=x1)

    # # Three nodes and two edges between them
    # edge_index2 = torch.tensor([[0,1, 1,2, 2, 3 , 3, 0],
    #                         [1,0, 2,1, 3, 2, 0, 3]], dtype=torch.long)
    # x2 = torch.ones((4, 1), dtype=torch.float)
    # # Define the graph
    # path_two = Data(edge_index=edge_index2, x=x2)

    # # compute Fractional Isomorphism Distance with edge constrain (S in paper)
    # path_one_adj = to_dense_adj(path_one.edge_index, max_num_nodes=path_one.x.size(0)).squeeze(0)
    # path_two_adj = to_dense_adj(path_two.edge_index, max_num_nodes=path_two.x.size(0)).squeeze(0)

    # # compute Fractional Isomorphism Distance with edge contstrain (S in paper)
    # fractional_constrained_dist,_,_ = subgraph_isomorphism_distance_kernelized(path_one.x, path_one_adj,path_two.x, path_two_adj, penalty_weight=2 ,pow=2 , mapping='fractional')
    # print('################################')
    # print('distance:')
    # print(fractional_constrained_dist)
    # # print('X:')
    # # print(X)
    # # print(f'x_v_empty:\n {x_v_empty}')
    # # print(f'x_empty_i:\n {x_empty_i}')
    # # print(f'C:\n {C}')
    # print('################################')

if __name__ == '__main__':
    # print(cp.installed_solvers())
    run()
