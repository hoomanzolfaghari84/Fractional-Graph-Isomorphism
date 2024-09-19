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


def isomorphism_distance_adjmatrix(phi_G, Adj_G, phi_H, Adj_H, lam, euclidean_distance = 'L2', mapping = 'fractional'):
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
    x_empty_i = cp.Variable(num_v_H, nonneg=True)  # Unmapped vertices in H (num_v_H,)

    # Objective function: minimize the total cost
    cost = cp.sum(cp.multiply(C, X)) + lam * cp.sum(x_v_empty) + lam * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")
    # cost = lambda_param * cp.sum(x_v_empty) + lambda_param * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")

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
        problem.solve()

        # print(f"LP solved successfully for validation graph {val_idx} and training graph {train_idx} with cost: {problem.value}")
        return problem.value , X.value, x_v_empty.value, x_empty_i.value, C # Return the minimum cost

    except Exception as e:
        print(f"Error occurred while solving LP: {e}")
        return None


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
    x_empty_i = cp.Variable(num_v_H, nonneg=True)  # Unmapped vertices in H (num_v_H,)

    # Objective function: minimize the total cost
    cost = cp.sum(cp.multiply(C, X)) + lam * cp.sum(x_v_empty) + lam * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")
    # cost = lambda_param * cp.sum(x_v_empty) + lambda_param * cp.sum(x_empty_i) + cp.norm(A_G @ X - X @ A_H, "fro")

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
        problem.solve()

        # print(f"LP solved successfully for validation graph {val_idx} and training graph {train_idx} with cost: {problem.value}")
        return problem.value , X.value, x_v_empty.value, x_empty_i.value, C # Return the minimum cost

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