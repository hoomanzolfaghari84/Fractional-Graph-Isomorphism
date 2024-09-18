from datasets import get_train_val_test_loaders, get_triplet_loader
from distance_functions import isomorphism_distance_adjmatrix

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GATConv
import torch.optim as optim
from torch_geometric.data import DataLoader
import random

from sklearn.metrics import accuracy_score
from collections import Counter
import concurrent.futures

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

from torch_geometric.utils import to_dense_adj
import numpy as np

class GINGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINGraphModel, self).__init__()


        self.conv1 = GINConv(nn.Sequential(nn.Linear(input_dim, int(hidden_dim/2)), nn.ReLU(), nn.Linear(int(hidden_dim/2), hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, int(hidden_dim/2))))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(int(hidden_dim/2), int(hidden_dim/2)), nn.ReLU(), nn.Linear(int(hidden_dim/2), output_dim)))
        # self.conv1 = GINConv(nn.Sequential(nn.Linear(input_dim, int(hidden_dim/2)), nn.ReLU(), nn.Linear(int(hidden_dim/2), hidden_dim)))
        # self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim/2)), nn.ReLU(), nn.Linear(int(hidden_dim/2), output_dim)))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)

        return x  # Returning node embeddings without pooling
    

class GATGraphModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.6):
        super(GATGraphModel, self).__init__()
        self.dropout = dropout

        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=2, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False, dropout=dropout)


    def forward(self, x, edge_index):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GAT layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return x
    

def fractional_torch_compute(X, x_v_empty, x_empty_i, C, lam, A_G, A_H):
    X = torch.tensor(X,dtype=torch.float32)

    x_v_empty = torch.tensor(x_v_empty,dtype=torch.float32)
    x_empty_i = torch.tensor(x_empty_i,dtype=torch.float32)
    return (C * X).sum() + lam * (x_v_empty.sum() + x_empty_i.sum()) + torch.norm(A_G @ X - X @ A_H, p='fro')

# Training loop
def train_triplet_model(model, triplet_loader, optimizer, lambda_param, margin):
    model.train()
    total_loss = 0

    for batch_idx, (anchor_data, positive_data, negative_data) in enumerate(triplet_loader):
        optimizer.zero_grad()

        # Forward pass through the model (anchor, positive, and negative)
        anchor_x = model(anchor_data.x, anchor_data.edge_index)
        positive_x = model(positive_data.x, positive_data.edge_index)
        negative_x = model(negative_data.x, negative_data.edge_index)

        lam = lambda_param.detach().cpu().item()
        # Convert adjacency matrices to dense form
        A_G = to_dense_adj(anchor_data.edge_index, max_num_nodes=anchor_x.size(0)).squeeze(0)#.detach()
        A_pos = to_dense_adj(positive_data.edge_index, max_num_nodes=positive_x.size(0)).squeeze(0)#.detach()
        A_neg = to_dense_adj(negative_data.edge_index, max_num_nodes=negative_x.size(0)).squeeze(0)#.detach()

        pos_distance, X_pos, x_v_empty_pos, x_empty_i_pos, _ = isomorphism_distance_adjmatrix(anchor_x, positive_x, A_G, A_pos, lam)
        neg_distance, X_neg, x_v_empty_neg, x_empty_i_neg, _ = isomorphism_distance_adjmatrix(anchor_x, negative_x, A_G, A_neg, lam)

        pos_torch_dist = fractional_torch_compute(X_pos, x_v_empty_pos, x_empty_i_pos, torch.cdist(anchor_x, positive_x, p=2), lambda_param,A_G,A_pos)
        neg_torch_dist = fractional_torch_compute(X_neg, x_v_empty_neg, x_empty_i_neg, torch.cdist(anchor_x, negative_x, p=2), lambda_param,A_G,A_neg)

        # Compute the triplet loss using ReLU to ensure non-negative loss
        loss = torch.relu(pos_torch_dist - neg_torch_dist + margin)

        # Backpropagate and update the GIN layers
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(triplet_loader)
    print(f"Average Triplet Loss: {avg_loss:.4f}")


def validate_triplet_model(model,train_loader,val_loader,lambda_param):

    model.eval()
    lambda_param_eval = lambda_param.detach()
    k_values = [1,3,5,7]
    dist_matrix = np.zeros((len(val_loader), len(train_loader), 2))

    y_true = np.zeros(len(val_loader))
    y_pred = np.zeros((len(k_values), len(val_loader)))

    best_k = 0
    best_acc = 0

    for val_idx, val_data in enumerate(val_loader):

        y_true[val_idx] = val_data.y.item()  # Ground truth class of the validation graph

        phi_val = model(val_data.x,val_data.edge_index)
        A_val = to_dense_adj(val_data.edge_index, max_num_nodes=phi_val.size(0)).squeeze(0)#.detach()

        for train_idx, train_data in enumerate(train_loader):

            phi_train = model(train_data.x, train_data.edge_index)
            A_train = to_dense_adj(train_data.edge_index, max_num_nodes=phi_train.size(0)).squeeze(0)#.detach()

            cost, _ ,_ ,_, _ = solve_fractional_lp_cost(phi_val, phi_train, A_val, A_train, lambda_param_eval)

            if cost is not None:
                  dist_matrix[val_idx][train_idx][0] = cost
                  dist_matrix[val_idx][train_idx][1] = train_data.y.item()

            else:
                  print(f"Failed to solve LP for validation graph {val_idx} and training graph {train_idx}")
                  dist_matrix[val_idx][train_idx][0] = None # np.inf
                  dist_matrix[val_idx][train_idx][1] = train_data.y.item()

        best_acc = 0
        for i_k, k in enumerate(k_values):
            prediction = find_k_nearest_label(dist_matrix[val_idx], k)
            y_pred[i_k][val_idx] = prediction
            acc = accuracy_score(y_pred[i_k][:val_idx+1], y_true[:val_idx+1])
            # print(f'accuracy with k={k} is: {acc}')
            if  acc > best_acc:
                best_acc = acc
                best_k = k

    print(f'val best k: {k} and best acc: {best_acc}')
    return best_acc, best_k

def run_triplet_experiment(dataset, device='cpu'):

    # Setup
    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)
    triplet_loader = get_triplet_loader(train_loader.dateset)

    input_dim = dataset.num_node_features  # Number of input features per node
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
            print(f'Model saved with validation accuracy: {best_acc:.4f} and k:{k}')