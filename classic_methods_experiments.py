import networkx as nx
from grakel import GraphKernel
import grakel
import numpy as np
import ot
from grakel import datasets
from torch_geometric.utils import to_networkx, to_dense_adj
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from sklearn.svm import SVC
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import concurrent.futures



from datasets import get_train_val_test_loaders
from distance_functions import homomorphism_distance_adjmatrix, isomorphism_distance_adjmatrix, isomorphism_distance_adjmatrix_constrained, isomorphism_distance_adjmatrix_only_structure, wasserstein_spectral_distance

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
    return np.bincount(knn_labels).argmax(), dists[indices,0].mean()

def run_knn_experiment(dataset):
    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)

    spectral_matrix = np.zeros((len(val_loader), len(train_loader),2))

    fractional_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))
    fractional_constrained_matrix = np.zeros((len(val_loader), len(train_loader),2))

    integral_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))
    integral_constrained_matrix = np.zeros((len(val_loader), len(train_loader),2))

    # fractional_structure_only_matrix = np.zeros((len(val_loader), len(train_loader),2))
    # homomorphism_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))

    lam = 3
    ks = [1, 5, 10, 20]

    spectral_preds = np.zeros((len(val_loader), len(ks)))

    fractional_preds = np.zeros((len(val_loader), len(ks)))
    fractional_constrained_preds = np.zeros((len(val_loader), len(ks)))

    integral_preds = np.zeros((len(val_loader), len(ks)))
    integral_constrained_preds = np.zeros((len(val_loader), len(ks)))


    # fractional_structure_only_preds = np.zeros((len(val_loader), len(ks)))
    # homomorphism_dist_preds = np.zeros((len(val_loader), len(ks)))

    true_labels = np.zeros(len(val_loader))

    spectral_kdists = []
    fractional_kdists = []
    
    # file_name = f'k-NN {dataset}.txt'
    report = open("k-NN MUTAG INT.txt", "w")
    i = 0

    for val_idx, val_data in enumerate(val_loader):
        
        val_adj = to_dense_adj(val_data.edge_index, max_num_nodes=val_data.x.size(0)).squeeze(0)
        
        print(f'passed val {val_idx}')
        for train_idx, train_data in enumerate(train_loader):
            train_adj = to_dense_adj(train_data.edge_index, max_num_nodes=train_data.x.size(0)).squeeze(0)
            
            # compute wasserstein spectral distance (from few-shot paper)
            spectral_matrix[val_idx][train_idx][0] = wasserstein_spectral_distance(val_data,train_data)
            spectral_matrix[val_idx][train_idx][1] = train_data.y

            # compute Fractional Isomorphism Distance (S* in paper)
            fractional_dist,  _, _, _, _ = isomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam, max_runtime=300)

            fractional_dist_matrix[val_idx][train_idx][0] = fractional_dist
            fractional_dist_matrix[val_idx][train_idx][1] = train_data.y

            # compute Fractional Isomorphism Distance with edge contstrain (S in paper)
            fractional_constrained_dist,  _, _, _, _ = isomorphism_distance_adjmatrix_constrained(train_data.x, train_adj, val_data.x, val_adj, lam, max_runtime=300)

            fractional_constrained_matrix[val_idx][train_idx][0] = fractional_constrained_dist
            fractional_constrained_matrix[val_idx][train_idx][1] = train_data.y

            # compute Integral Isomorphism Distance (S* in paper)
            integral_dist,  _, _, _, _ = isomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam, mapping = 'integral')

            integral_dist_matrix[val_idx][train_idx][0] = integral_dist
            integral_dist_matrix[val_idx][train_idx][1] = train_data.y

            # compute Integral Isomorphism Distance with edge contstrain (S in paper)
            integral_constrained_dist,  _, _, _, _ = isomorphism_distance_adjmatrix_constrained(train_data.x, train_adj, val_data.x, val_adj, lam, mapping = 'integral')

            integral_constrained_matrix[val_idx][train_idx][0] = integral_constrained_dist
            integral_constrained_matrix[val_idx][train_idx][1] = train_data.y

            # compute Fractional Isomorphism Distance with only structure optimization
            # fractional_structure_only_dist, _, _, _ = isomorphism_distance_adjmatrix_only_structure(train_data.x, train_adj, val_data.x, val_adj, lam, max_runtime=300)

            # fractional_structure_only_matrix[val_idx][train_idx][0] = fractional_structure_only_dist
            # fractional_structure_only_matrix[val_idx][train_idx][1] = train_data.y
        
            # compute Homomorphism Distance
            # lcost, _, _, _ = homomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam) 
            # rcost, _, _, _ = homomorphism_distance_adjmatrix(val_data.x, val_adj, train_data.x, train_adj, lam)

            # homomorphism_dist_matrix[val_idx][train_idx][0] = lcost + rcost
            # homomorphism_dist_matrix[val_idx][train_idx][1] = train_data.y

        for ik, k in enumerate(ks):
            spectral_predict, spectral_kdist = find_k_nearest_label(spectral_matrix[val_idx],k)
            frational_predict, fractional_kdist = find_k_nearest_label(fractional_dist_matrix[val_idx],k)
            integral_predict, integral_kdist = find_k_nearest_label(integral_dist_matrix[val_idx],k)

            spectral_preds[val_idx,ik] = spectral_predict

            fractional_preds[val_idx,ik] = frational_predict
            fractional_constrained_preds[val_idx,ik] , _ = find_k_nearest_label(fractional_constrained_matrix[val_idx],k)

            integral_preds[val_idx,ik] = integral_predict
            integral_constrained_preds[val_idx,ik] , _ = find_k_nearest_label(integral_constrained_matrix[val_idx],k)
            
            # fractional_structure_only_preds[val_idx,ik] , _ = find_k_nearest_label(fractional_structure_only_matrix[val_idx],k)
            # homomorphism_dist_preds[val_idx,ik] , _ = find_k_nearest_label(homomorphism_dist_matrix[val_idx],k)


        true_labels[val_idx] = val_data.y

        

        # clear_output()
    
    # lines = []

    print('################################')
    print('################################')
    print(f'k-NN classification result for :{dataset}\n')
    print(f'used lamda={lam}. Train-Val-Test split:{len(train_loader)}-{len(val_loader)}-{len(test_loader)}')
    for ik, k in enumerate(ks):
        print('================================================')
        print(f'with k={k}')
        print(f'spectral report:\n {classification_report(true_labels, spectral_preds[:,ik])}')
        print('================================================================\n')
        print(f'fractional report:\n {classification_report(true_labels, fractional_preds[:,ik])}')
        print('================================================================\n')
        print(f'fractional_constrained report:\n {classification_report(true_labels, fractional_constrained_preds[:,ik])}')
        print('================================================================\n')
        print(f'integral report:\n {classification_report(true_labels, integral_preds[:,ik])}')
        print('===============================================================\n')
        print(f'integral_constrained report:\n {classification_report(true_labels, integral_constrained_preds[:,ik])}')
        # print('================================================================')
        # print(f'homomorphism report: {classification_report(true_labels, homomorphism_dist_preds[:,ik])}')
    print('################################')
    print('################################')
    
    report.write('################################\n')
    report.write('################################\n')
    report.write(f'k-NN classification result for :{dataset}\n')
    report.write(f'used lamda={lam}. Train-Val-Test split:{len(train_loader)}-{len(val_loader)}-{len(test_loader)}\n')
    for k in ks:
        report.write('================================================\n')
        report.write(f'with k={k}\n')
        report.write(f'spectral report:\n {classification_report(true_labels, spectral_preds[:,ik])}')
        report.write('================================================================\n')
        report.write(f'fractional report:\n {classification_report(true_labels, fractional_preds[:,ik])}')
        report.write('================================================================\n')
        report.write(f'fractional_constrained report:\n {classification_report(true_labels, fractional_constrained_preds[:,ik])}')
        report.write('================================================================\n')
        report.write(f'integral report:\n {classification_report(true_labels, integral_preds[:,ik])}')
        report.write('================================================================\n')
        report.write(f'integral_constrained report:\n {classification_report(true_labels, integral_constrained_preds[:,ik])}')
        # report.write('================================================================\n')
        # report.write(f'fractional_structure_only report:\n {classification_report(true_labels, fractional_structure_only_preds[:,ik])}')
        # report.write('================================================================')
        # report.write(f'homomorphism report: {classification_report(true_labels, homomorphism_dist_preds[:,ik])}')
    report.write('################################\n')
    report.write('################################')

    # report.write(lines)
    report.close()


def calculate_distances(val_data, train_data, val_adj, train_adj, lam):
    distances = {}
    
    # compute wasserstein spectral distance
    spectral_dist = wasserstein_spectral_distance(val_data, train_data)
    
    # compute Fractional Isomorphism Distance (S* in paper)
    fractional_dist, _, _, _, _ = isomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam)
    
    # compute Fractional Isomorphism Distance with edge constrain (S in paper)
    fractional_constrained_dist, _, _, _, _ = isomorphism_distance_adjmatrix_constrained(train_data.x, train_adj, val_data.x, val_adj, lam)
    
    # compute Fractional Isomorphism Distance with only structure optimization
    fractional_structure_only_dist, _, _, _ = isomorphism_distance_adjmatrix_only_structure(train_data.x, train_adj, val_data.x, val_adj, lam)
    
    distances['spectral'] = (spectral_dist, train_data.y)
    distances['fractional'] = (fractional_dist, train_data.y)
    distances['fractional_constrained'] = (fractional_constrained_dist, train_data.y)
    distances['fractional_structure_only'] = (fractional_structure_only_dist, train_data.y)

    return distances

def run_knn_experiment_multithread(dataset):
    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)

    spectral_matrix = np.zeros((len(val_loader), len(train_loader), 2))
    fractional_dist_matrix = np.zeros((len(val_loader), len(train_loader), 2))
    fractional_constrained_matrix = np.zeros((len(val_loader), len(train_loader), 2))
    fractional_structure_only_matrix = np.zeros((len(val_loader), len(train_loader), 2))

    lam = 3
    ks = [1, 5, 10, 20]

    spectral_preds = np.zeros((len(val_loader), len(ks)))
    fractional_preds = np.zeros((len(val_loader), len(ks)))
    fractional_constrained_preds = np.zeros((len(val_loader), len(ks)))
    fractional_structure_only_preds = np.zeros((len(val_loader), len(ks)))
    
    true_labels = np.zeros(len(val_loader))

    def process_val_data(val_idx, val_data):
        val_adj = to_dense_adj(val_data.edge_index, max_num_nodes=val_data.x.size(0)).squeeze(0)
        
        def process_train_data(train_idx, train_data):
            train_adj = to_dense_adj(train_data.edge_index, max_num_nodes=train_data.x.size(0)).squeeze(0)
            distances = calculate_distances(val_data, train_data, val_adj, train_adj, lam)
            return train_idx, distances

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_train = {executor.submit(process_train_data, train_idx, train_data): train_idx for train_idx, train_data in enumerate(train_loader)}
            
            for future in concurrent.futures.as_completed(future_to_train):
                train_idx, distances = future.result()
                
                # Store results in the matrices
                spectral_matrix[val_idx][train_idx] = distances['spectral']
                fractional_dist_matrix[val_idx][train_idx] = distances['fractional']
                fractional_constrained_matrix[val_idx][train_idx] = distances['fractional_constrained']
                fractional_structure_only_matrix[val_idx][train_idx] = distances['fractional_structure_only']

    # Process each validation graph
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda val_idx_data: process_val_data(*val_idx_data), enumerate(val_loader))

    # Now do the k-NN predictions
    for val_idx, val_data in enumerate(val_loader):
        for ik, k in enumerate(ks):
            spectral_predict, spectral_kdist = find_k_nearest_label(spectral_matrix[val_idx], k)
            frational_predict, fractional_kdist = find_k_nearest_label(fractional_dist_matrix[val_idx], k)
            
            spectral_preds[val_idx, ik] = spectral_predict
            fractional_preds[val_idx, ik] = frational_predict

            fractional_constrained_preds[val_idx, ik], _ = find_k_nearest_label(fractional_constrained_matrix[val_idx], k)
            fractional_structure_only_preds[val_idx, ik], _ = find_k_nearest_label(fractional_structure_only_matrix[val_idx], k)

        true_labels[val_idx] = val_data.y

    
    print('################################')
    print('################################\n')
    print(f'k-NN classification result for :{dataset}\n')
    print(f'used lamda={lam}. Train-Val-Test split:{len(train_loader)}-{len(val_loader)}-{len(test_loader)}')
    for ik, k in enumerate(ks):
        print('================================================\n')
        print(f'with k={k}\n')
        print(f'spectral report:\n {classification_report(true_labels, spectral_preds[:,ik])}')
        print('================================================================')
        print(f'fractional report:\n {classification_report(true_labels, fractional_preds[:,ik])}')
        print('================================================================')
        print(f'fractional_constrained report:\n {classification_report(true_labels, fractional_constrained_preds[:,ik])}')
        print('================================================================')
        print(f'fractional_structure_only report:\n {classification_report(true_labels, fractional_structure_only_preds[:,ik])}')
        # print('================================================================')
        # print(f'homomorphism report: {classification_report(true_labels, homomorphism_dist_preds[:,ik])}')
    print('################################')
    print('################################')
    
    report = open(f"k-NN PROTEINS.txt", "w")

    report.write('################################\n')
    report.write('################################\n')
    report.write(f'k-NN classification result for :{dataset}\n')
    report.write(f'used lamda={lam}. Train-Val-Test split:{len(train_loader)}-{len(val_loader)}-{len(test_loader)}\n')
    for k in ks:
        report.write('================================================\n')
        report.write(f'with k={k}\n')
        report.write(f'spectral report:\n {classification_report(true_labels, spectral_preds[:,ik])}')
        report.write('================================================================')
        report.write(f'fractional report:\n {classification_report(true_labels, fractional_preds[:,ik])}')
        report.write('================================================================')
        report.write(f'fractional_constrained report:\n {classification_report(true_labels, fractional_constrained_preds[:,ik])}')
        report.write('================================================================')
        report.write(f'fractional_structure_only report:\n {classification_report(true_labels, fractional_structure_only_preds[:,ik])}')
        # report.write('================================================================')
        # report.write(f'homomorphism report: {classification_report(true_labels, homomorphism_dist_preds[:,ik])}')
    report.write('################################\n')
    report.write('################################')

    # report.write(lines)
    report.close()


# def run_knn_experiment(dataset):
#     train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)

#     spectral_matrix = np.zeros((len(val_loader), len(train_loader),2))
#     fractional_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))

#     fractional_constrained_matrix = np.zeros((len(val_loader), len(train_loader),2))
#     fractional_structure_only_matrix = np.zeros((len(val_loader), len(train_loader),2))
#     homomorphism_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))

#     lam = 3
#     ks = [1, 5, 10, 20]

#     spectral_preds = np.zeros((len(val_loader), len(ks)))
#     fractional_preds = np.zeros((len(val_loader), len(ks)))

#     fractional_constrained_preds = np.zeros((len(val_loader), len(ks)))
#     fractional_structure_only_preds = np.zeros((len(val_loader), len(ks)))
#     homomorphism_dist_preds = np.zeros((len(val_loader), len(ks)))

#     true_labels = np.zeros(len(val_loader))

#     spectral_kdists = []
#     fractional_kdists = []

#     report = open(f"k-NN MUTAG.txt", "w")

#     for val_idx, val_data in enumerate(val_loader):
        
#         val_adj = to_dense_adj(val_data.edge_index, max_num_nodes=val_data.x.size(0)).squeeze(0)
         
        
#         for train_idx, train_data in enumerate(train_loader):

#             train_adj = to_dense_adj(train_data.edge_index, max_num_nodes=train_data.x.size(0)).squeeze(0)
            
#             # compute wasserstein spectral distance (from few-shot paper)
#             spectral_matrix[val_idx][train_idx][0] = wasserstein_spectral_distance(val_data,train_data)
#             spectral_matrix[val_idx][train_idx][1] = train_data.y

#             # compute Fractional Isomorphism Distance (S* in paper)
#             fractional_dist,  _, _, _, _ = isomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam)

#             fractional_dist_matrix[val_idx][train_idx][0] = fractional_dist
#             fractional_dist_matrix[val_idx][train_idx][1] = train_data.y

#             # compute Fractional Isomorphism Distance with edge contstrain (S in paper)
#             fractional_constrained_dist,  _, _, _, _ = isomorphism_distance_adjmatrix_constrained(train_data.x, train_adj, val_data.x, val_adj, lam)

#             fractional_constrained_matrix[val_idx][train_idx][0] = fractional_dist
#             fractional_constrained_matrix[val_idx][train_idx][1] = train_data.y

#             # compute Fractional Isomorphism Distance with only structure optimization
#             fractional_structure_only_dist, _, _, _ = isomorphism_distance_adjmatrix_only_structure(train_data.x, train_adj, val_data.x, val_adj, lam)

#             fractional_structure_only_matrix[val_idx][train_idx][0] = fractional_dist
#             fractional_structure_only_matrix[val_idx][train_idx][1] = train_data.y

#             # compute Homomorphism Distance
#             lcost, _, _, _ = homomorphism_distance_adjmatrix(train_data.x, train_adj, val_data.x, val_adj, lam) 
#             rcost, _, _, _ = homomorphism_distance_adjmatrix(val_data.x, val_adj, train_data.x, train_adj, lam)

#             homomorphism_dist_matrix[val_idx][train_idx][0] = lcost + rcost
#             homomorphism_dist_matrix[val_idx][train_idx][1] = train_data.y

#         for ik, k in enumerate(ks):
#             spectral_predict, spectral_kdist = find_k_nearest_label(spectral_matrix[val_idx],k)
#             frational_predict, fractional_kdist = find_k_nearest_label(fractional_dist_matrix[val_idx],k)
            
#             spectral_preds[val_idx,ik] = spectral_predict
#             fractional_preds[val_idx,ik] = frational_predict

#             fractional_constrained_preds[val_idx,ik] , _ = find_k_nearest_label(fractional_constrained_matrix[val_idx],k)
#             fractional_structure_only_preds[val_idx,ik] , _ = find_k_nearest_label(fractional_structure_only_matrix[val_idx],k)
#             homomorphism_dist_preds[val_idx,ik] , _ = find_k_nearest_label(homomorphism_dist_matrix[val_idx],k)


#         true_labels[val_idx] = val_data.y

#         # clear_output()
    
#     # lines = []

#     print('################################')
#     print('################################')
#     print(f'k-NN classification result for :{dataset}\n')
#     print(f'used lamda={lam}. Train-Val-Test split:{len(train_loader)}-{len(val_loader)}-{len(test_loader)}')
#     for ik, k in enumerate(ks):
#         print('================================================')
#         print(f'with k={k}')
#         print(f'spectral report:\n {classification_report(true_labels, spectral_preds[:,ik])}')
#         print('================================================================')
#         print(f'fractional report:\n {classification_report(true_labels, fractional_preds[:,ik])}')
#         print('================================================================')
#         print(f'fractional_constrained report:\n {classification_report(true_labels, fractional_constrained_preds[:,ik])}')
#         print('================================================================')
#         print(f'fractional_structure_only report:\n {classification_report(true_labels, fractional_structure_only_preds[:,ik])}')
#         print('================================================================')
#         print(f'homomorphism report: {classification_report(true_labels, homomorphism_dist_preds[:,ik])}')
#     print('################################')
#     print('################################')
    
#     report.write('################################')
#     report.write('################################')
#     report.write(f'k-NN classification result for :{dataset}')
#     report.write(f'used lamda={lam}. Train-Val-Test split:{len(train_loader)}-{len(val_loader)}-{len(test_loader)}')
#     for k in ks:
#         report.write('================================================')
#         report.write(f'with k={k}')
#         report.write(f'spectral report: {classification_report(true_labels, spectral_preds[:,ik])}')
#         report.write('================================================================')
#         report.write(f'fractional report: {classification_report(true_labels, fractional_preds[:,ik])}')
#         report.write('================================================================')
#         report.write(f'fractional_constrained report: {classification_report(true_labels, fractional_constrained_preds[:,ik])}')
#         report.write('================================================================')
#         report.write(f'fractional_structure_only report: {classification_report(true_labels, fractional_structure_only_preds[:,ik])}')
#         report.write('================================================================')
#         report.write(f'homomorphism report: {classification_report(true_labels, homomorphism_dist_preds[:,ik])}')
#     report.write('################################')
#     report.write('################################')

#     # report.write(lines)
#     report.close()

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# SVM experiment
#

def fractional_rbf_kernel(phi_G, phi_H, A_G, A_H, lambda_param, sigma = 1):

    d,_,_,_,_ = isomorphism_distance_adjmatrix(phi_G, A_G, phi_H, A_H, lambda_param)

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
    print(f"SVM Classification Report using fractional rbf Kernel: \n{classification_report(y_val_frac, y_pred)}")
    

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
    print(f"SVM Classification Accuracy using weisfeiler lehman Kernel: {accuracy}")
    print(f"SVM Classification Report using weisfeiler lehman Kernel: {classification_report(y_val_grakel, y_pred)}")



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




# ----------------------------------------------------------------
# ----------------------------------------------------------------
# number of train data influence experiment
#

def run_numof_train_data_experiment(dataset, device = 'cpu'):
    pass