import networkx as nx
from grakel import GraphKernel
import grakel
import numpy as np
import ot
from grakel import datasets
from torch_geometric.utils import to_networkx, to_dense_adj
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
from sklearn.svm import SVC
from grakel.kernels import WeisfeilerLehman, VertexHistogram


from datasets import get_train_val_test_loaders
from distance_functions import isomorphism_distance_adjmatrix, wasserstein_spectral_distance

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# k-NN experiment
#

def find_k_nearest_label(dists, k=1):
    # Get the indices of the k smallest distances
    indices = np.argpartition(dists[:, 0], k)[:k]

    # Extract the labels of the k-nearest neighbors and cast to integers
    knn_labels = dists[indices, 1].astype(int)  # Ensure labels are integers

    # Return the label with the highest frequency
    return np.bincount(knn_labels).argmax()

def run_knn_experiment(dataset):
    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)

    spectral_matrix = np.zeros((len(val_loader), len(train_loader),2))
    fractional_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))

    lam = 3
    k = 7

    spectral_preds = []
    frational_preds = []
    true_labels = []

    for val_idx, val_data in enumerate(val_loader):

        val_adj = to_dense_adj(val_data.edge_index, max_num_nodes=val_data.x.size(0)).squeeze(0)
        val_nx = to_networkx(val_data)  # Convert to networkx graph

        for train_idx, train_data in enumerate(train_loader):

            train_adj = to_dense_adj(train_data.edge_index, max_num_nodes=train_data.x.size(0)).squeeze(0)
            train_nx = to_networkx(train_data)
            
            # compute wasserstein spectral distance
            spectral_matrix[val_idx][train_idx][0] = wasserstein_spectral_distance(val_nx,train_nx)
            spectral_matrix[val_idx][train_idx][1] = train_data.y

            # compute Fractional Isomorphism Distance
            fractional_dist, X, x_v_empty, x_empty_i, C = isomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam)

            fractional_dist_matrix[val_idx][train_idx][0] = fractional_dist
            fractional_dist_matrix[val_idx][train_idx][1] = train_data.y


        spectral_predict = find_k_nearest_label(spectral_matrix[val_idx],k)
        frational_predict = find_k_nearest_label(fractional_dist_matrix[val_idx],k)
        
        spectral_preds.append(spectral_predict)
        frational_preds.append(frational_predict)
        
        true_labels.append(val_data.y)

        # clear_output()
    print(f'spectral acc: {accuracy_score(spectral_preds, true_labels)}')
    print(f'fractional acc: {accuracy_score(frational_preds, true_labels)}')
        

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# SVM experiment
#

def fractional_rbf_kernel(phi_G, phi_H, A_G, A_H, lambda_param, sigma = 1):

    d,_,_,_,_ = solve_fractional_lp_cost(phi_G, phi_H, A_G, A_H, lambda_param)

    return np.exp(-np.linalg.norm(d) ** 2 / (2 * (sigma ** 2)))

def pyg_data_to_grakel_graph(pyg_data):

    adj = to_dense_adj(pyg_data.edge_index, max_num_nodes=pyg_data.x.size(0)).squeeze(0).numpy()

    num_nodes = pyg_data.num_nodes

    # Convert node features or indices to node labels
    labels = {i: 1 for i in range(num_nodes)}  # Assuming x contains node features or indices

    return grakel.Graph(adj, labels)

def run_svm_experiment(dataset):
    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)

    ### compute fractional rbf kernel matrices
    ### 
    fractional_rbf_matrix_train = np.zeros((len(train_loader),len(train_loader)))
    fractional_rbf_matrix_val = np.zeros((len(val_loader),len(train_loader)))
    y_train_frac = []
    y_val_frac = []

    lam = 2
    sigma = 3


    for idx_1, G_1 in enumerate(train_loader):
        adj_1 = to_dense_adj(G_1.edge_index, max_num_nodes=G_1.x.size(0)).squeeze(0)
        y_train_frac.append(G_1.y)
        for idx_2, G_2 in enumerate(train_loader):

            if idx_2 <= idx_1 : continue

            adj_2 = to_dense_adj(G_2.edge_index, max_num_nodes=G_2.x.size(0)).squeeze(0)

            fractional_rbf_matrix_train[idx_1][idx_2] = fractional_rbf_kernel(G_1.x, G_2.x, adj_1, adj_2, lam, sigma)


    for idx_1, G_1 in enumerate(val_loader):

        adj_1 = to_dense_adj(G_1.edge_index, max_num_nodes=G_1.x.size(0)).squeeze(0)
        y_val_frac.append(G_1.y)
        for idx_2, G_2 in enumerate(train_loader):

            adj_2 = to_dense_adj(G_2.edge_index, max_num_nodes=G_2.x.size(0)).squeeze(0)

            fractional_rbf_matrix_val[idx_1][idx_2] = fractional_rbf_kernel(G_1.x, G_2.x, adj_1, adj_2, lam, sigma)
        

    ### prepare grakel data
    ### 

    grakel_loader_train = []
    grakel_loader_val = []
    y_train_grakel = []
    y_val_grakel = []

    for graph in train_loader:

        grakel_loader_train.append(pyg_data_to_grakel_graph(graph))
        y_train_grakel.append(graph.y)



    for graph in val_loader:
        # adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.size(0)).squeeze(0)
        grakel_loader_val.append(pyg_data_to_grakel_graph(graph))
        y_val_grakel.append(graph.y)


    ### perform SVM
    ###

    svm = SVC(kernel="precomputed")
    svm.fit(fractional_rbf_matrix_train, y_train_frac)
    # Make predictions
    y_pred = svm.predict(fractional_rbf_matrix_val)
    # Evaluate accuracy
    accuracy = accuracy_score(y_val_frac, y_pred)
    print(f"SVM Classification Accuracy using fractional rbf Kernel: {accuracy}")


    # Initialize Weisfeiler-Lehman Kernel
    wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)
    wl_train = wl_kernel.fit_transform(grakel_loader_train)
    wl_val = wl_kernel.transform(grakel_loader_val)

    svm = SVC(kernel="precomputed")
    svm.fit(wl_train, y_train_grakel)
    # Make predictions
    y_pred = svm.predict(wl_val)
    # Evaluate accuracy
    accuracy = accuracy_score(y_val_grakel, y_pred)
    print(f"SVM Classification Accuracy using weisfeiler_lehman Kernel: {accuracy}")



# ----------------------------------------------------------------
# ----------------------------------------------------------------
# k means clustering experiment
#

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# # Apply KMeans clustering based on distance matrix
# n_clusters = 3  # Set number of clusters
# kmeans = KMeans(n_clusters=n_clusters)
# cluster_labels = kmeans.fit_predict(ged_matrix)

# # Evaluate clustering quality
# silhouette_avg = silhouette_score(ged_matrix, cluster_labels, metric='precomputed')
# print(f"Silhouette Score using GED: {silhouette_avg}")