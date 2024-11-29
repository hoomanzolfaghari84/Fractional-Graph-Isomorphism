# import networkx as nx
# import numpy as np
# from scipy.linalg import eigh
# import ot
# import networkx as nx
# import numpy as np
# import ot
# from torch_geometric.utils import to_networkx, to_dense_adj
# from sklearn.metrics.pairwise import cosine_distances
# from scipy.spatial.distance import mahalanobis
# from scipy.stats import wasserstein_distance
# import torch
# import cvxpy as cp

######################################################################################################
##################################
################################## Not Used Anymore
##################################
# ######################################################################################################


# class DistanceFuntion :

#     def __init__(self, name = 'subgraph_isomorphism_distance', fractional = False, euclidean_distance = 'L2', lam = 2, dense = True):
#         self.name = name
#         self.fractional = fractional
#         self.euclidean_distance = euclidean_distance
#         self.lam = lam
#         self.dense = dense

#     def __call__(self, data1, data2, adj1=None, adj2=None):
       
#        if self.name == 'subgraph_isomorphism_distance':
           
#            if self.dense:
#             return subgraph_isomorphism_distance(data1, adj1, data2, adj2, self.lam, self.fractional)
           
#            data1_x, data1_adj = data1.x, to_dense_adj(data1.edge_index, max_num_nodes=data1.x.size(0)).squeeze(0)
#            data2_x, data2_adj = data2.x, to_dense_adj(data2.edge_index, max_num_nodes=data2.x.size(0)).squeeze(0)
#            d, _, _  = subgraph_isomorphism_distance(data1_x, data1_adj, data2_x, data2_adj, self.lam, self.fractional)
#            return d
       
#        elif self.name == 'graph_convex_isomorphism':
#            data1_x, data1_adj = data1.x, to_dense_adj(data1.edge_index, max_num_nodes=data1.x.size(0)).squeeze(0)
#            data2_x, data2_adj = data2.x, to_dense_adj(data2.edge_index, max_num_nodes=data2.x.size(0)).squeeze(0)

#            d , _, _, _, _ = graph_convex_isomorphism(data1_x, data1_adj, data2_x, data2_adj, self.lam, self.euclidean_distance)
#            return d
       
#        elif self.name == 'wasserstein_spectral_distance':
#            return wasserstein_spectral_distance(data1, data2)     
    
#     def get_name(self):
#         return self.name if not self.fractional else self.name + ' fractional'

# def graph_convex_isomorphism(phi_G, Adj_G, phi_H, Adj_H, lam, euclidean_distance = 'L2'):
#     num_v_G = phi_G.size(0)  # Number of vertices in graph G
#     num_v_H = phi_H.size(0)  # Number of vertices in graph H

#     # Compute the Euclidean distances between node features
#     if euclidean_distance == 'L2':
#         C = torch.cdist(phi_G, phi_H, p=2).numpy()
#     else:
#         phi_G = phi_G.numpy()
#         phi_H = phi_H.numpy()

#         if euclidean_distance == 'cosine':
#             C = cosine_distances(phi_G, phi_H)
#         elif euclidean_distance == 'mahalanobis':
#             C = compute_mahalanobis_distance_matrix(phi_G, phi_H)
#         elif euclidean_distance == 'wasserstein':
#             C = compute_wasserstein_distance_matrix(phi_G, phi_H)
#         else:
#             raise Exception('wrong euclid metric')

#     # Get adjacency matrices and print their shapes
#     A_G = Adj_G.numpy()  # (num_v_G, num_v_G)
#     A_H = Adj_H.numpy()  # (num_v_H, num_v_H)

    
#     X = cp.Variable((num_v_G, num_v_H), nonneg=True)  # Fractional mapping (num_v_G, num_v_H)
#     x_v_empty = cp.Variable(num_v_G, nonneg=True)  # Unmapped vertices in G (num_v_G,)
#     x_empty_i = cp.Variable(num_v_H, nonneg=True)  # Unmapped vertices in H (num_v_H,)
#     # Objective function: minimize the total cost
#     cost = cp.sum(cp.multiply(C, X)) + lam * cp.sum(x_v_empty) + lam * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")

#     # Correct structural constraint
#     constraints = [
#         X <= 1,  # Ensure the fractional mapping is between 0 and 1
#         x_v_empty <= 1,  # Ensure unmapped vertices in G are between 0 and 1
#         x_empty_i <= 1,  # Ensure unmapped vertices in H are between 0 and 1
#         x_v_empty + cp.sum(X, axis=1) == 1,  # For all v in V(G)
#         x_empty_i + cp.sum(X, axis=0) == 1,  # For all i in V(H)
        
#     ]
    
#     try:
#         # Solve the LP problem
#         problem = cp.Problem(cp.Minimize(cost), constraints)

#         problem.solve(solver = cp.SCS)

#         # print(f"LP solved successfully for validation graph {val_idx} and training graph {train_idx} with cost: {problem.value}")
#         return problem.value , X.value, x_v_empty.value, x_empty_i.value, C # Return the minimum cost

#     except Exception as e:
#         print(f"Error occurred while solving LP: {e}")
#         raise e




# def subgraph_isomorphism_distance(phi_G, Adj_G, phi_H, Adj_H, lam,  mapping='fractional', C = None):
#     num_v_G = phi_G.size(0)  # Number of vertices in graph G
#     num_v_H = phi_H.size(0)  # Number of vertices in graph H

#     # Cost matrix initialization with lambda for the added vertex
#     C = torch.cdist(phi_G, phi_H, p=2).numpy() if C is None  else C
    
#     C = np.pad(C, ((0, 1), (0, 1)), mode='constant', constant_values=lam)
#     C[num_v_G, num_v_H] = 0  # C(0,0) = 0

#     # Get adjacency matrices and print their shapes
#     A_G = Adj_G.numpy()  # (num_v_G, num_v_G)
#     A_H = Adj_H.numpy()  # (num_v_H, num_v_H)

#     A_G_aug = np.pad(A_G, ((0, 1), (0, 1)), mode='constant', constant_values=1)
#     A_H_aug = np.pad(A_H, ((0, 1), (0, 1)), mode='constant', constant_values=1)

#     if mapping == 'integral':
#         X = cp.Variable((num_v_G + 1, num_v_H + 1), boolean=True)
#     else:
#         X = cp.Variable((num_v_G + 1, num_v_H + 1), nonneg=True)

#     objective = cp.sum(cp.multiply(C, X)) 
    

#     constraints = [
#         X <= 1,
#         X[num_v_G, num_v_H] == 1 # x_{0,0} = 1
#     ]

#     # Row constraints: sum of mappings for each vertex in G
#     constraints += [
#         cp.sum(X[v, :]) == 1 for v in range(num_v_G)  # For all rows except the last
#     ]

#     # Column constraints: sum of mappings for each vertex in H
#     constraints += [
#         cp.sum(X[:, i]) == 1 for i in range(num_v_H)  # For all columns except the last
#     ]

#     # For each edge uv in G and ij not in E(H^+), sum(x_{u,i} + x_{v,j}) <= 1
#     for u in range(num_v_G):
#         for v in range(num_v_G):
#             if A_G[u, v] >= 0.5:  # If uv is an edge in G
#                 for i in range(num_v_H):
#                     for j in range(num_v_H):
#                         if A_H[i, j] < 0.5:  # If ij is NOT an edge in H^+
#                             constraints.append(X[u, i] + X[v, j] <= 1)

#     # For uv not in G^+ and ij in E(H), sum(x_{u,i} + x_{v,j}) <= 1
#     for u in range(num_v_G):
#         for v in range(num_v_G):
#             if A_G[u, v] < 0.5:  # If uv is NOT an edge in G
#                 for i in range(num_v_H):
#                     for j in range(num_v_H):
#                         if A_H[i, j] >= 0.5:  # If ij is an edge in H^+
#                             constraints.append(X[u, i] + X[v, j] <= 1)



#     try:
#         # Solve the LP problem
#         problem = cp.Problem(cp.Minimize(objective), constraints)

#         if mapping == 'integral':
            
#             problem.solve(solver=cp.GLPK_MI)#, max_iter = 1000)#, verbose=True)
#         else:
#             problem.solve(solver=cp.SCS)
    
#         # print(f'val frac:{np.sum(C * X.value)}')
#         # Return the minimum cost and mapping
#         return problem.value, X.value, C

#     except Exception as e:
#         print(f"Error occurred while solving LP: {e}")
#         raise e

# def build_mcs_constraints(X, num_v_G, num_v_H, A_G, A_H):
#     constraints = [
#         X <= 1,
#         X[num_v_G, num_v_H] == 1 # x_{0,0} = 1
#     ]

#     # Row constraints: sum of mappings for each vertex in G
#     constraints += [
#         cp.sum(X[v, :]) == 1 for v in range(num_v_G)  # For all rows except the last
#     ]

#     # Column constraints: sum of mappings for each vertex in H
#     constraints += [
#         cp.sum(X[:, i]) == 1 for i in range(num_v_H)  # For all columns except the last
#     ]

#     # For each edge uv in G and ij not in E(H^+), sum(x_{u,i} + x_{v,j}) <= 1
#     for u in range(num_v_G):
#         for v in range(num_v_G):
#             if A_G[u, v] == 1:  # If uv is an edge in G
#                 for i in range(num_v_H):
#                     for j in range(num_v_H):
#                         if A_H[i, j] == 0:  # If ij is NOT an edge in H^+
#                             constraints.append(X[u, i] + X[v, j] <= 1)

#     # For uv not in G^+ and ij in E(H), sum(x_{u,i} + x_{v,j}) <= 1
#     for u in range(num_v_G):
#         for v in range(num_v_G):
#             if A_G[u, v] == 0:  # If uv is NOT an edge in G
#                 for i in range(num_v_H):
#                     for j in range(num_v_H):
#                         if A_H[i, j] == 1:  # If ij is an edge in H^+
#                             constraints.append(X[u, i] + X[v, j] <= 1)




# # ----------------------------------------------------------------
# # ----------------------------------------------------------------
# # Wasserstien Spectral Distance from paper "FEW-SHOT LEARNING ON GRAPHS VIA SUPER-CLASSES BASED ON GRAPH SPECTRAL MEASURES"
# #

# # Function to compute the normalized Laplacian eigenvalues for a graph
# def compute_spectral_measure(G):
#     # Compute the normalized Laplacian of the graph
#     L = nx.normalized_laplacian_matrix(G).todense()
#     # Compute the eigenvalues of the normalized Laplacian
#     eigenvalues = np.sort(eigh(L, eigvals_only=True))
#     return eigenvalues

# # Function to compute the p-th Wasserstein distance between two graphs' spectral measures
# def wasserstein_spectral_distance(data1, data2, p=2):

#     G1 = to_networkx(data1, to_undirected=True)
#     G2 = to_networkx(data2, to_undirected=True)

#     # Compute the spectral measures (eigenvalues) for both graphs
#     eigs_g1 = compute_spectral_measure(G1)
#     eigs_g2 = compute_spectral_measure(G2)

#     # Dirac mass at each eigenvalue: assign uniform weights (since Dirac mass means weight 1 at each eigenvalue)
#     weights_g1 = np.ones(len(eigs_g1)) / len(eigs_g1)  # Uniform distribution over eigenvalues
#     weights_g2 = np.ones(len(eigs_g2)) / len(eigs_g2)  # Uniform distribution over eigenvalues

#     # Define the cost matrix: the difference between eigenvalues raised to the power p
#     M = np.abs(np.subtract.outer(eigs_g1, eigs_g2)) ** p  # Cost matrix (absolute differences raised to p)

#     # Compute the p-th Wasserstein distance using the Optimal Transport library
#     W_p = ot.emd2(weights_g1, weights_g2, M)

#     return W_p ** (1/p)  # Return the p-th root of the cost



# # ----------------------------------------------------------------
# # ----------------------------------------------------------------
# # Euclidean Feature Distances
# #

# def compute_mahalanobis_distance_matrix(X1, X2, epsilon=1e-5):
#   # Compute the covariance matrix of the combined set
#   cov_matrix = np.cov(np.vstack((X1, X2)).T)  # Covariance matrix
#   # Regularize the covariance matrix by adding a small value to the diagonal
#   cov_matrix += epsilon * np.eye(cov_matrix.shape[0])
#   inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse covariance matrix

#   # Compute the Mahalanobis distance between each pair of vectors
#   mahalanobis_distance_matrix = np.zeros((X1.shape[0], X2.shape[0]))
#   for i, x in enumerate(X1):
#       for j, y in enumerate(X2):
#           diff = x - y
#           mahalanobis_distance_matrix[i, j] = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))

#   return mahalanobis_distance_matrix


# def compute_wasserstein_distance_matrix(X1, X2):
#     # Compute the Wasserstein distance between each pair of vectors
#     wasserstein_distance_matrix = np.zeros((X1.shape[0], X2.shape[0]))
#     for i, x in enumerate(X1):
#         for j, y in enumerate(X2):
#             wasserstein_distance_matrix[i, j] = wasserstein_distance(x, y)

#     return wasserstein_distance_matrix


# def compute_cosine_distance_matrix(X1, X2):
#     return cosine_distances(X1, X2)