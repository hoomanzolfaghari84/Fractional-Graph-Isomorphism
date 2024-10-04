import time
import networkx as nx
import numpy as np
from scipy.linalg import eigh
import ot
import networkx as nx
from grakel import GraphKernel
import grakel
import numpy as np
import ot
from grakel import datasets
from torch_geometric.utils import to_networkx, to_dense_adj
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
import torch
import cvxpy as cp

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# subgraph isomorphism distance
#

def subgraph_isomorphism_distance_no_loops(phi_G, Adj_G, phi_H, Adj_H, lam, mapping='fractional'):
    num_v_G = phi_G.size(0)  # Number of vertices in graph G
    num_v_H = phi_H.size(0)  # Number of vertices in graph H

    C = torch.cdist(phi_G, phi_H, p=2).detach().cpu().numpy()

    C = np.pad(C, ((0, 0), (0, 1)), mode='constant', constant_values=lam)

    # Get adjacency matrices and print their shapes
    A_G = Adj_G.detach().cpu().numpy()  # (num_v_G, num_v_G)
    A_H = Adj_H.detach().cpu().numpy()  # (num_v_H, num_v_H)

    A_H_aug = np.pad(A_H, ((0, 1), (0, 1)), mode='constant', constant_values=1)

    if mapping == 'integral':
        X = cp.Variable((num_v_G, num_v_H + 1), boolean=True)
    else:
        X = cp.Variable((num_v_G, num_v_H + 1), nonneg=True)

    objective = cp.sum(cp.multiply(C, X)) 

    constraints = [
        X <= 1,
        cp.sum(X, axis=1) == 1,
    ]

    constraints += [
        cp.sum(X[:, i]) <= 1 for i in range(num_v_H)  # For all columns except the last
    ]

    # Constraint 2: For each edge uv in G and ij not in E(H^+), sum(x_{u,i} + x_{v,j}) <= 1
    for u in range(num_v_G):
        for v in range(num_v_G):
            if A_G[u, v] == 1:  # If uv is an edge in G
                for i in range(num_v_H + 1):
                    for j in range(num_v_H + 1):
                        if A_H_aug[i, j] == 0:  # If ij is NOT an edge in H^+
                            constraints.append(X[u, i] + X[v, j] <= 1)

    # Constraint 3: For uv not in G and ij in E(H), sum(x_{u,i} + x_{v,j}) <= 1
    for u in range(num_v_G):
        for v in range(num_v_G):
            if A_G[u, v] == 0:  # If uv is NOT an edge in G
                for i in range(num_v_H):
                    for j in range(num_v_H):
                        if A_H_aug[i, j] == 1:  # If ij is an edge in H^+
                            constraints.append(X[u, i] + X[v, j] <= 1)



    try:
        # Solve the LP problem
        problem = cp.Problem(cp.Minimize(objective), constraints)

        if mapping == 'integral':
            problem.solve(solver=cp.GLPK_MI)#, verbose=True)
        else:
            problem.solve(solver=cp.SCS)

        # Return the minimum cost and mapping
        return problem.value, X.value, C

    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        raise e

def subgraph_isomorphism_distance(phi_G, Adj_G, phi_H, Adj_H, lam, mapping='fractional'):
    num_v_G = phi_G.size(0)  # Number of vertices in graph G
    num_v_H = phi_H.size(0)  # Number of vertices in graph H

    C = torch.cdist(phi_G, phi_H, p=2).detach().cpu().numpy()

    C = np.pad(C, ((0, 1), (0, 1)), mode='constant', constant_values=lam)  # Add a row and column of constants

    C[num_v_G, num_v_H] = 0

    # Get adjacency matrices and print their shapes
    A_G = Adj_G.detach().cpu().numpy()  # (num_v_G, num_v_G)
    A_H = Adj_H.detach().cpu().numpy()  # (num_v_H, num_v_H)

    A_G_aug = np.pad(A_G, ((0, 1), (0, 1)), mode='constant', constant_values=1)
    A_H_aug = np.pad(A_H, ((0, 1), (0, 1)), mode='constant', constant_values=1)

    if mapping == 'integral':
        X = cp.Variable((num_v_G + 1, num_v_H + 1), boolean=True)
    else:
        X = cp.Variable((num_v_G + 1, num_v_H + 1), nonneg=True)

    objective = cp.sum(cp.multiply(C, X)) 

    constraints = [
        X <= 1,
        X[num_v_G, num_v_H] == 1
    ]

    constraints += [
        cp.sum(X[v, :]) == 1 for v in range(num_v_G)  # For all rows except the last
    ]

    constraints += [
        cp.sum(X[:, i]) == 1 for i in range(num_v_H)  # For all columns except the last
    ]

    # Constraint 2: For each edge uv in G and ij not in E(H^+), sum(x_{u,i} + x_{v,j}) <= 1
    for u in range(num_v_G):
        for v in range(num_v_G):
            if A_G_aug[u, v] == 1:  # If uv is an edge in G
                for i in range(num_v_H + 1):
                    for j in range(num_v_H + 1):
                        if A_H_aug[i, j] == 0:  # If ij is NOT an edge in H^+
                            constraints.append(X[u, i] + X[v, j] <= 1)

    # Constraint 3: For uv not in G and ij in E(H), sum(x_{u,i} + x_{v,j}) <= 1
    for u in range(num_v_G):
        for v in range(num_v_G):
            if A_G_aug[u, v] == 0:  # If uv is NOT an edge in G
                for i in range(num_v_H):
                    for j in range(num_v_H):
                        if A_H_aug[i, j] == 1:  # If ij is an edge in H^+
                            constraints.append(X[u, i] + X[v, j] <= 1)



    try:
        # Solve the LP problem
        problem = cp.Problem(cp.Minimize(objective), constraints)

        if mapping == 'integral':
            problem.solve(solver=cp.GLPK_MI)#, verbose=True)
        else:
            problem.solve(solver=cp.SCS)

        # Return the minimum cost and mapping
        return problem.value, X.value, C

    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        raise e

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Isomorphism Distance
#

euclidean_distances = ['L2','mahalanobis','wasserstein','cosine']
mappings = ['integral', 'fractional']
lp_solvers = ['cvxpy']
graph_formats = ['adj_matrix', 'coo' , 'py_data' ]

class IsomorphismDistance:
    
    def __init__(self, euclidean_distance, structural_distance, mapping, lp_solver, graph_format):
        self.euclidean_distance =euclidean_distance
        self.structural_distance = structural_distance
        self.mapping = mapping
        self.lp_solver = lp_solver
        self.graph_format = graph_format

    def __call__(self):
        pass

def add_row_column_of_ones(matrix):
    # Get the shape of the original matrix
    original_shape = matrix.shape
    
    # Create a row of ones with the same number of columns as the original matrix
    ones_row = np.ones((1, original_shape[1]))
    
    # Create a column of ones with the same number of rows as the original matrix
    ones_col = np.ones((original_shape[0] + 1, 1))
    
    # Add the row of ones to the bottom of the original matrix
    matrix_with_ones_row = np.vstack([matrix, ones_row])
    
    # Add the column of ones to the right of the new matrix
    matrix_with_ones = np.hstack([matrix_with_ones_row, ones_col])
    
    return matrix_with_ones

def isomorphism_distance_adjmatrix_constrained(phi_G, Adj_G, phi_H, Adj_H, lam, euclidean_distance='L2', mapping='fractional', max_runtime=60):
    num_v_G = phi_G.size(0)  # Number of vertices in graph G
    num_v_H = phi_H.size(0)  # Number of vertices in graph H

    # Compute the Euclidean distances between node features
    if euclidean_distance == 'L2':
        C = torch.cdist(phi_G, phi_H, p=2).detach().cpu().numpy()
    else:
        phi_G = phi_G.detach().cpu().numpy()
        phi_H = phi_H.detach().cpu().numpy()

        if euclidean_distance == 'cosine':
            C = cosine_distances(phi_G, phi_H)
        elif euclidean_distance == 'mahalanobis':
            C = compute_mahalanobis_distance_matrix(phi_G, phi_H)
        elif euclidean_distance == 'wasserstein':
            C = compute_wasserstein_distance_matrix(phi_G, phi_H)
        else:
            raise Exception('Invalid Euclidean metric')

    # Get adjacency matrices and print their shapes
    A_G = Adj_G.detach().cpu().numpy()  # (num_v_G, num_v_G)
    A_H = Adj_H.detach().cpu().numpy()  # (num_v_H, num_v_H)

    # Set up cvxpy vars
    if mapping == 'integral':
        X = cp.Variable((num_v_G, num_v_H), boolean=True)  # Integral mapping (num_v_G, num_v_H)
        x_v_empty = cp.Variable((num_v_G, 1), boolean=True)  # Unmapped vertices in G (num_v_G, 1)
        x_empty_i = cp.Variable((1, num_v_H), boolean=True)  # Unmapped vertices in H (1, num_v_H)
    else:
        X = cp.Variable((num_v_G, num_v_H), nonneg=True)  # Fractional mapping (num_v_G, num_v_H)
        x_v_empty = cp.Variable((num_v_G, 1), nonneg=True)  # Unmapped vertices in G (num_v_G, 1)
        x_empty_i = cp.Variable((1, num_v_H), nonneg=True)  # Unmapped vertices in H (1, num_v_H)

    # Objective function: minimize the total cost
    cost = cp.sum(cp.multiply(C, X)) + lam * cp.sum(x_v_empty) + lam * cp.sum(x_empty_i)


    # # Extend X to include empty variables
    # X_augmented_1 = cp.hstack([X, cp.reshape(x_v_empty, (num_v_G, 1))])  # Add x_{v,empty} as a column to X
    # X_augmented_2 = cp.vstack([X, cp.reshape(x_empty_i, (1, num_v_H))])  # Add x_{empty,i} as a row to X
     
    # Extend X to include empty variables
    X_augmented = cp.vstack([
        cp.hstack([X, cp.reshape(x_v_empty, (num_v_G, 1))]),  # Add x_{v,empty} as a column to X
        cp.hstack([x_empty_i, cp.Constant([[0]])])  # Add x_{empty,i} as a row and the final value as one
    ])

    print(f'X_augmented:\n{X_augmented}')

    A_H_aug = add_row_column_of_ones(A_H)
    A_G_aug = add_row_column_of_ones(A_G)

    print(f'A_H_aug:\n{A_H_aug}')
    print(f'A_G_aug:\n{A_G_aug}')

    # Constraints
    constraints = [
        X <= 1,  # Ensure the fractional mapping is between 0 and 1
        x_v_empty <= 1,  # Ensure unmapped vertices in G are between 0 and 1
        x_empty_i <= 1,  # Ensure unmapped vertices in H are between 0 and 1
        x_v_empty[:,0] + cp.sum(X, axis=1) == 1,  # For all v in V(G)
        x_empty_i[0] + cp.sum(X, axis=0) == 1,  # For all i in V(H)
        A_G_aug @ X_augmented == X_augmented @ A_H_aug,
        # A_G_aug @ X_augmented_2 == X_augmented_2 @ A_H
    ]

    try:
        # Solve the LP problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        if mapping == 'integral':
            problem.solve(solver=cp.GLPK_MI, verbose=True)
        else:
            problem.solve(solver=cp.SCS)

        # Return the minimum cost and mapping
        return problem.value, X.value, x_v_empty.value, x_empty_i.value, C

    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        raise e
    
def isomorphism_distance_adjmatrix_only_structure(phi_G, Adj_G, phi_H, Adj_H, lam, mapping = 'fractional', max_runtime=60):
    num_v_G = phi_G.size(0)  # Number of vertices in graph G
    num_v_H = phi_H.size(0)  # Number of vertices in graph H


    # Get adjacency matrices and print their shapes
    A_G = Adj_G.detach().cpu().numpy()  # (num_v_G, num_v_G)
    A_H = Adj_H.detach().cpu().numpy()  # (num_v_H, num_v_H)

    # Set up cvxpy problem
    X = cp.Variable((num_v_G, num_v_H), nonneg=True)  # Fractional mapping (num_v_G, num_v_H)
    x_v_empty = cp.Variable(num_v_G, nonneg=True)  # Unmapped vertices in G (num_v_G,)
    x_empty_i = cp.Variable(num_v_H, nonneg=True)  # Unmapped vertices in H (num_v_H,)

    # Objective function: minimize the total cost
    cost = lam * cp.sum(x_v_empty) + lam * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")
    # cost = lambda_param * cp.sum(x_v_empty) + lambda_param * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")

    constraints = [
        X <= 1,  # Ensure the fractional mapping is between 0 and 1
        x_v_empty <= 1,  # Ensure unmapped vertices in G are between 0 and 1
        x_empty_i <= 1,  # Ensure unmapped vertices in H are between 0 and 1
        x_v_empty + cp.sum(X, axis=1) == 1,  # For all v in V(G)
        x_empty_i + cp.sum(X, axis=0) == 1,  # For all i in V(H)
    ]

    try:
        # Solve the LP problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve(solver=cp.SCS)


        # print(f"LP solved successfully for validation graph {val_idx} and training graph {train_idx} with cost: {problem.value}")
        return problem.value , X.value, x_v_empty.value, x_empty_i.value # Return the minimum cost

    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        raise e



def isomorphism_distance_adjmatrix(phi_G, Adj_G, phi_H, Adj_H, lam, euclidean_distance = 'L2', mapping = 'fractional', max_runtime=60):
    num_v_G = phi_G.size(0)  # Number of vertices in graph G
    num_v_H = phi_H.size(0)  # Number of vertices in graph H

    # Compute the Euclidean distances between node features
    if euclidean_distance == 'L2':
        C = torch.cdist(phi_G, phi_H, p=2).detach().cpu().numpy()
    else:
        phi_G = phi_G.detach().cpu().numpy()
        phi_H = phi_H.detach().cpu().numpy()

        if euclidean_distance == 'cosine':
            C = cosine_distances(phi_G, phi_H)
        elif euclidean_distance == 'mahalanobis':
            C = compute_mahalanobis_distance_matrix(phi_G, phi_H)
        elif euclidean_distance == 'wasserstein':
            C = compute_wasserstein_distance_matrix(phi_G, phi_H)
        else:
            raise Exception('wrong euclid metric')

    # Get adjacency matrices and print their shapes
    A_G = Adj_G.detach().cpu().numpy()  # (num_v_G, num_v_G)
    A_H = Adj_H.detach().cpu().numpy()  # (num_v_H, num_v_H)

    # Set up cvxpy problem
    if mapping == 'integral':
        X = cp.Variable((num_v_G, num_v_H), boolean=True)  # Fractional mapping (num_v_G, num_v_H)
        x_v_empty = cp.Variable(num_v_G, boolean=True)  # Unmapped vertices in G (num_v_G,)
        x_empty_i = cp.Variable(num_v_H, boolean=True)  # Unmapped vertices in H (num_v_H,)
    else:
        X = cp.Variable((num_v_G, num_v_H), nonneg=True)  # Fractional mapping (num_v_G, num_v_H)
        x_v_empty = cp.Variable(num_v_G, nonneg=True)  # Unmapped vertices in G (num_v_G,)
        x_empty_i = cp.Variable(num_v_H, nonneg=True)  # Unmapped vertices in H (num_v_H,)
    # Objective function: minimize the total cost
    cost = cp.sum(cp.multiply(C, X)) + lam * cp.sum(x_v_empty) + lam * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")

    # Correct structural constraint
    constraints = [
        X <= 1,  # Ensure the fractional mapping is between 0 and 1
        x_v_empty <= 1,  # Ensure unmapped vertices in G are between 0 and 1
        x_empty_i <= 1,  # Ensure unmapped vertices in H are between 0 and 1
        x_v_empty + cp.sum(X, axis=1) == 1,  # For all v in V(G)
        x_empty_i + cp.sum(X, axis=0) == 1,  # For all i in V(H)
        # A_G @ X == X @ A_H  # Edge preservation
    ]
    
    try:
        # Solve the LP problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        if mapping == 'integral':
            problem.solve(solver=cp.ECOS_BB)
        else:
            problem.solve(solver = cp.SCS)


        # print(f"LP solved successfully for validation graph {val_idx} and training graph {train_idx} with cost: {problem.value}")
        return problem.value , X.value, x_v_empty.value, x_empty_i.value, C # Return the minimum cost

    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        raise e


def isomorphism_distance_pyg(data_G, data_H, lam, euclide_metric = 'L2', mapping = 'fractional'):
    pass

def isomorphism_distance_coo(phi_G, edge_index_G, phi_H, edge_index_H, lam, euclide_metric = 'L2', mapping = 'fractional'):
    pass

def homomorphism_distance_adjmatrix(phi_G, Adj_G, phi_H, Adj_H, lam, euclidean_distance = 'L2', mapping = 'fractional'):
    num_v_G = phi_G.size(0)  # Number of vertices in graph G
    num_v_H = phi_H.size(0)  # Number of vertices in graph H

    # Compute the Euclidean distances between node features
    if euclidean_distance == 'L2':
        C = torch.cdist(phi_G, phi_H, p=2).detach().cpu().numpy()
    else:
        phi_G = phi_G.detach().cpu().numpy()
        phi_H = phi_H.detach().cpu().numpy()

        if euclidean_distance == 'cosine':
            C = cosine_distances(phi_G, phi_H)
        elif euclidean_distance == 'mahalanobis':
            C = compute_mahalanobis_distance_matrix(phi_G, phi_H)
        elif euclidean_distance == 'wasserstein':
            C = compute_wasserstein_distance_matrix(phi_G, phi_H)
        else:
            raise Exception('wrong euclid metric')

    # Get adjacency matrices and print their shapes
    A_G = Adj_G.detach().cpu().numpy()  # (num_v_G, num_v_G)
    A_H = Adj_H.detach().cpu().numpy()  # (num_v_H, num_v_H)

    # Set up cvxpy problem
    X = cp.Variable((num_v_G, num_v_H), nonneg=True)  # Fractional mapping (num_v_G, num_v_H)
    x_v_empty = cp.Variable(num_v_G, nonneg=True)  # Unmapped vertices in G (num_v_G,)
    

    # Objective function: minimize the total cost
    cost = cp.sum(cp.multiply(C, X)) + lam * cp.sum(x_v_empty) + cp.norm(X @ A_H - A_G @ X, "fro")
    # cost = lambda_param * cp.sum(x_v_empty) + lambda_param * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")

    # Correct structural constraint
    constraints = [
        X <= 1,  # Ensure the fractional mapping is between 0 and 1
        x_v_empty <= 1,  # Ensure unmapped vertices in G are between 0 and 1
        x_v_empty + cp.sum(X, axis=1) == 1,  # For all v in V(G)
        A_G @ X <= X @ A_H
    ]



    try:
        # Solve the LP problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver = cp.SCS)

        # print(f"LP solved successfully for validation graph {val_idx} and training graph {train_idx} with cost: {problem.value}")
        return problem.value , X.value, x_v_empty.value, C
    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        return None


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Wasserstien Spectral Distance from paper "FEW-SHOT LEARNING ON GRAPHS VIA SUPER-CLASSES BASED ON GRAPH SPECTRAL MEASURES"
#

# Function to compute the normalized Laplacian eigenvalues for a graph
def compute_spectral_measure(G):
    # Compute the normalized Laplacian of the graph
    L = nx.normalized_laplacian_matrix(G).todense()
    # Compute the eigenvalues of the normalized Laplacian
    eigenvalues = np.sort(eigh(L, eigvals_only=True))
    return eigenvalues

# Function to compute the p-th Wasserstein distance between two graphs' spectral measures
def wasserstein_spectral_distance(data1, data2, p=2):

    G1 = to_networkx(data1, to_undirected=True) # idk why on my local nx works with undirected only. a version difference maybe
    G2 = to_networkx(data2, to_undirected=True)

    # Compute the spectral measures (eigenvalues) for both graphs
    eigs_g1 = compute_spectral_measure(G1)
    eigs_g2 = compute_spectral_measure(G2)

    # Dirac mass at each eigenvalue: assign uniform weights (since Dirac mass means weight 1 at each eigenvalue)
    weights_g1 = np.ones(len(eigs_g1)) / len(eigs_g1)  # Uniform distribution over eigenvalues
    weights_g2 = np.ones(len(eigs_g2)) / len(eigs_g2)  # Uniform distribution over eigenvalues

    # Define the cost matrix: the difference between eigenvalues raised to the power p
    M = np.abs(np.subtract.outer(eigs_g1, eigs_g2)) ** p  # Cost matrix (absolute differences raised to p)

    # Compute the p-th Wasserstein distance using the Optimal Transport library
    W_p = ot.emd2(weights_g1, weights_g2, M)

    return W_p ** (1/p)  # Return the p-th root of the cost


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Measures for deviation of a semi-doubly stochastic matrix form being a permutation matrix
#

# Shannon Entropy
def compute_shannon_entropy(X):
    # To avoid log(0), we clip the values of X to a small positive number
    X_clipped = np.clip(X, 1e-10, 1.0)
    entropy = - np.sum(X * np.log(X_clipped))
    return entropy

# Binary Deviation
def compute_binary_deviation(X):
    binary_deviation = np.sum(np.minimum(X, 1 - X))
    return binary_deviation

# Row Sum Deviation
def compute_row_sum_deviation(X):
    row_sums = np.sum(X, axis=1)  # Sum along rows
    row_sum_deviation = np.sum(np.abs(row_sums - 1))
    return row_sum_deviation

# Column Sum Deviation
def compute_column_sum_deviation(X):
    col_sums = np.sum(X, axis=0)  # Sum along columns
    col_sum_deviation = np.sum(np.abs(col_sums - 1))
    return col_sum_deviation

# Orthogonality Deviation (for square matrices)
def compute_orthogonality_deviation(X):
    if X.shape[0] > X.shape[1]:
      dim = X.shape[0]
      T = X @ X.T
    else:
      dim = X.shape[1]
      T = X.T @ X
        # raise ValueError("Orthogonality deviation can only be computed for square matrices")
    orthogonality_deviation = np.linalg.norm(T - np.eye(dim), ord='fro')
    return orthogonality_deviation



# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Euclidean Feature Distances
#

def compute_mahalanobis_distance_matrix(X1, X2, epsilon=1e-5):
  # Compute the covariance matrix of the combined set
  cov_matrix = np.cov(np.vstack((X1, X2)).T)  # Covariance matrix
  # Regularize the covariance matrix by adding a small value to the diagonal
  cov_matrix += epsilon * np.eye(cov_matrix.shape[0])
  inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse covariance matrix

  # Compute the Mahalanobis distance between each pair of vectors
  mahalanobis_distance_matrix = np.zeros((X1.shape[0], X2.shape[0]))
  for i, x in enumerate(X1):
      for j, y in enumerate(X2):
          diff = x - y
          mahalanobis_distance_matrix[i, j] = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))

  return mahalanobis_distance_matrix


def compute_wasserstein_distance_matrix(X1, X2):
    # Compute the Wasserstein distance between each pair of vectors
    wasserstein_distance_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x in enumerate(X1):
        for j, y in enumerate(X2):
            wasserstein_distance_matrix[i, j] = wasserstein_distance(x, y)

    return wasserstein_distance_matrix


def compute_cosine_distance_matrix(X1, X2):
    return cosine_distances(X1, X2)