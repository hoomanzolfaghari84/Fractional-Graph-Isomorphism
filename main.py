import torch

from classic_methods_experiments import run_knn_experiment, run_knn_experiment_multithread, run_svm_experiment
from datasets import load_dataset
from distance_functions import isomorphism_distance_adjmatrix, isomorphism_distance_adjmatrix_constrained, subgraph_isomorphism_distance, subgraph_isomorphism_distance_no_loops
from fewshot_experiments import run_fewshot_without_training
import cvxpy as cp
from torch_geometric.utils import to_networkx, to_dense_adj


import torch
from torch_geometric.data import Data

# def dist(phi_1, G1, phi_2, G2, lam):
#     # Number of vertices
#     n1 = G1.shape[0]
#     n2 = G2.shape[0]

#     # Compute the cost matrix C
#     C = torch.cdist(phi_1, phi_2, p=2).detach().cpu().numpy()

#     # Convert adjacency matrices to numpy
#     G1 = G1.detach().cpu().numpy()  # (num_v_G, num_v_G)
#     G2 = G2.detach().cpu().numpy()  # (num_v_H, num_v_H)

#     # Binary variables for selecting vertices
#     x = cp.Variable(n1, nonneg=True)  # Vertices from G1
#     y = cp.Variable(n2, nonneg=True)  # Vertices from G2

#     # Binary variables for matching vertices
#     z = cp.Variable((n1, n2), nonneg=True)  # Matching variables

#     # Objective function: Minimize the specified expression
#     objective = cp.Minimize(cp.sum(cp.multiply(C, z)) + cp.sum(cp.multiply(lam, (1 - z))))

#     # Constraints
#     constraints = []

#     # Induced subgraph constraints
#     for i in range(n1):
#         for k in range(n1):
#             if G1[i, k] == 1:  # If there's an edge in G1
#                 for j in range(n2):
#                     constraints.append(z[i, j] <= x[i])
#                     constraints.append(z[i, j] <= y[j])

#     for j in range(n2):
#         for m in range(n2):
#             if G2[j, m] == 1:  # If there's an edge in G2
#                 for i in range(n1):
#                     constraints.append(z[i, j] <= x[i])
#                     constraints.append(z[i, j] <= y[j])

#     # Matching constraints
#     for i in range(n1):
#         constraints.append(cp.sum(z[i, :]) <= x[i])  # Each vertex in G1 can be matched at most once

#     for j in range(n2):
#         constraints.append(cp.sum(z[:, j]) <= y[j])  # Each vertex in G2 can be matched at most once

#     constraints.append(x <=1)
#     constraints.append( y <=1)
#     constraints.append(z <=1)
#     # Formulate the problem
#     problem = cp.Problem(objective, constraints)

#     # Solve the problem
#     problem.solve(solver=cp.SCS)

#     # # Output the results
#     # print("Optimal value of the objective function:", problem.value)
#     # print("Selected vertices from G1 (x):", x.value)
#     # print("Selected vertices from G2 (y):", y.value)
#     # print("Matching (z):")
#     # print(z.value)

#     return problem.value, x.value, y.value, z.value, C


def run():

    # torch.manual_seed(42)  # For reproducibility
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    dataset = load_dataset('IMDB-MULTI')
    
    run_knn_experiment(dataset)

    # # Two nodes and one edge between them
    # edge_index1 = torch.tensor([[0, 1],
    #                         [1, 0]], dtype=torch.long)
    # x1 = torch.ones((2, 1), dtype=torch.float)
    # # Define the graph
    # path_one = Data(edge_index=edge_index1, x=x1)

    # # Three nodes and two edges between them
    # edge_index2 = torch.tensor([[0,1, 1,2],
    #                         [1,0, 2,1]], dtype=torch.long)
    # x2 = torch.ones((3, 1), dtype=torch.float)
    # # Define the graph
    # path_two = Data(edge_index=edge_index2, x=x2)

    # # compute Fractional Isomorphism Distance with edge constrain (S in paper)
    # path_one_adj = to_dense_adj(path_one.edge_index, max_num_nodes=path_one.x.size(0)).squeeze(0)
    # path_two_adj = to_dense_adj(path_two.edge_index, max_num_nodes=path_two.x.size(0)).squeeze(0)

    # # compute Fractional Isomorphism Distance with edge contstrain (S in paper)
    # fractional_constrained_dist,  X, C = subgraph_isomorphism_distance(path_two.x, path_two_adj, path_one.x, path_one_adj, 2, mapping='fractional')
    # print('################################')
    # print('distance:')
    # print(fractional_constrained_dist)
    # print('X:')
    # print(X)
    # # print(f'x_v_empty:\n {x_v_empty}')
    # # print(f'x_empty_i:\n {x_empty_i}')
    # print(f'C:\n {C}')
    # print('################################')

if __name__ == '__main__':
    # print(cp.installed_solvers())
    run()
