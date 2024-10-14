import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import grakel

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

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to visualize a graph
def visualize_graph(data, title):
    G = to_networkx(data, to_undirected=True)  # Convert to NetworkX graph
    plt.figure(figsize=(5, 5))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title(title)
    plt.show()

# Define the GIN model
class GIN(torch.nn.Module):
    def __init__(self, inpute_dim, hidden_channels, output_dim):
        super(GIN, self).__init__()
        # Layers for the GIN
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(inpute_dim, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
            )
        )
        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
            )
        )
        self.lin = torch.nn.Linear(hidden_channels, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Pooling (summarize graph information)
        x = global_add_pool(x, batch)  # Sum pooling over the nodes in a graph
        
        # Classifier
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
    



# # Load the dataset
# dataset = TUDataset(root='data/TUDataset', name='MUTAG')

# # # Convert the PyG dataset to Grakel format
# # def pyg_to_grakel(pyg_data):
# #     gnx_graph = to_networkx(pyg_data, to_undirected=True)
# #     labels = {i: data for i, data in enumerate(pyg_data.x.cpu().numpy())} if pyg_data.x is not None else {}
# #     edges = list(gnx_graph.edges())
# #     return Graph(edges, node_labels=labels)

# # Prepare the dataset for Grakel (Graph Kernel library)
# grakel_graphs = [pyg_to_grakel(data) for data in dataset]
# graph_labels = [data.y.item() for data in dataset]

# # Split into train and test sets
# train_graphs, test_graphs, train_labels, test_labels = train_test_split(grakel_graphs, graph_labels, test_size=0.2, random_state=42)

# # Define the Weisfeiler-Lehman subtree kernel
# wl_kernel = WeisfeilerLehman(n_iter=4, base_kernel=VertexHistogram)

# # Compute the kernel matrices for training and testing
# K_train = wl_kernel.fit_transform(train_graphs)
# K_test = wl_kernel.transform(test_graphs)

# # Train a Support Vector Machine (SVM) on the kernel matrix
# svm = SVC(kernel='precomputed')
# svm.fit(K_train, train_labels)

# # Test the SVM model
# predictions = svm.predict(K_test)
# accuracy = accuracy_score(test_labels, predictions)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")


def pyg_data_to_grakel_graph(pyg_data):

    adj = to_dense_adj(pyg_data.edge_index, max_num_nodes=pyg_data.x.size(0)).squeeze(0).numpy()

    num_nodes = pyg_data.num_nodes

    # Convert node features or indices to node labels
    labels = {i: pyg_data.x[i] for i in range(num_nodes)}  # Assuming x contains node features or indices

    return grakel.Graph(adj, node_labels=labels)

# Graph 1 (Top graph: Two squares sharing an edge)
edge_index_1 = torch.tensor([[0, 1, 2, 3, 2, 3, 4, 5],
                            [1, 2, 3, 0, 3, 4, 5, 2]], dtype=torch.long)

data_1 = Data(edge_index=edge_index_1, x=torch.ones((6,1)))

adj1 = to_dense_adj(data_1.edge_index, max_num_nodes=data_1.x.size(0)).squeeze(0)

# Graph 2 (Bottom graph: Two triangles connected by an edge)
edge_index_2 = torch.tensor([[0, 1, 2, 2, 3, 4, 5, 3],
                            [1, 2, 0, 3, 4, 5, 3, 2]], dtype=torch.long)

data_2 = Data(edge_index=edge_index_2, x=torch.ones((6,1)))

adj2 = to_dense_adj(data_2.edge_index, max_num_nodes=data_2.x.size(0)).squeeze(0)

# Visualize Graph 1 (Top)
visualize_graph(data_1, "Graph 1 (Top)")

# Visualize Graph 2 (Bottom)
visualize_graph(data_2, "Graph 2 (Bottom)")

subgraph_isomorphism_dist,X1,_ = subgraph_isomorphism_distance(data_1.x, adj1, data_2.x, adj2, 2, mapping='integral')


subgraph_isomorphism_dist_frac,X2,_ = subgraph_isomorphism_distance(data_1.x, adj1, data_2.x, adj2, 2, mapping='fractional')

print(f'int dist:{subgraph_isomorphism_dist}')
print(X1)
print(f'frac dist:{subgraph_isomorphism_dist_frac}')
print(X2)

G_data_1 = pyg_data_to_grakel_graph(data_1)
G_data_2 = pyg_data_to_grakel_graph(data_2)

train_graphs = [G_data_1, G_data_2]

# Define the Weisfeiler-Lehman subtree kernel
wl_kernel = WeisfeilerLehman(n_iter=4, normalize=True)

# Compute the kernel matrices for training and testing
K_train = wl_kernel.fit_transform(train_graphs)

print(f"WL\n {K_train}")