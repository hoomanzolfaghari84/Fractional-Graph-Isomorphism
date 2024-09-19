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
from sklearn.svm import SVC
from grakel.kernels import WeisfeilerLehman, VertexHistogram


from classic_methods_experiments import find_k_nearest_label
from datasets import get_train_val_test_loaders, get_triplet_loader
from distance_functions import isomorphism_distance_adjmatrix, wasserstein_spectral_distance

## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
## true fewshot with zero training
##

from datasets import get_m_shot_loaders
from metric_learning_experiments import GINGraphModel, train_triplet_model, validate_triplet_model


def run_fewshot_without_training(dataset, device = 'cpu'):
    train_loader, val_loader, test_loader = get_m_shot_loaders(dataset, 5)

    spectral_matrix = np.zeros((len(val_loader), len(train_loader),2))
    fractional_dist_matrix = np.zeros((len(val_loader), len(train_loader),2))

    lam = 1
    k = 5

    spectral_preds = []
    frational_preds = []
    true_labels = []

    for val_idx, val_data in enumerate(val_loader):

        val_adj = to_dense_adj(val_data.edge_index, max_num_nodes=val_data.x.size(0)).squeeze(0)
        
        for train_idx, train_data in enumerate(train_loader):

            train_adj = to_dense_adj(train_data.edge_index, max_num_nodes=train_data.x.size(0)).squeeze(0)
            
            
            # compute wasserstein spectral distance
            spectral_matrix[val_idx][train_idx][0] = wasserstein_spectral_distance(val_data,train_data)
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


def run_fewshot_with_transfered_training(dataset, device='cpu'):
    # # split base class data

    # # train base data
    # base_training(base_data)

    # # fine tune on novel data
    pass



def  base_training(base_data):
    # Setup
    train_loader, val_loader, test_loader = get_train_val_test_loaders(base_data)
    triplet_loader = get_triplet_loader(train_loader.dateset)

    input_dim = base_data.num_node_features  # Number of input features per node
    hidden_dim = 20  # Hidden dimension size
    output_dim = 3  # Output dimension size
    lambda_param = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
    margin = 8
    save_path = '/models/best_model_triplet_GIN.pth'
    # Initialize the GIN model and optimizer
    model = GINGraphModel(input_dim, hidden_dim, output_dim).to(device)
    # model = GATGraphModel(input_dim=dataset.num_node_features, hidden_dim=20, output_dim=4, heads=8, dropout=0.6)


    optimizer = optim.Adam([
        {'params': model.parameters()},  # Model parameters
        {'params': [lambda_param]}        # Extra parameter
    ], lr=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

    # Training the model
    best_acc = 0
    best_k = None
    patience = 4
    for epoch in range(5):  # Example with 5 epochs
        print(f"Epoch {epoch + 1}/5")
        train_triplet_model(model, triplet_loader, optimizer, lambda_param, margin)

        acc, k = validate_triplet_model(model,train_loader,val_loader,lambda_param)

        scheduler.step(acc)

        if acc <= best_acc:
            patience -= 1
            if patience ==0 : break
        else:
            best_acc = acc
            best_k = k
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with validation accuracy: {best_acc:.4f} and k:{best_k}')

    return model