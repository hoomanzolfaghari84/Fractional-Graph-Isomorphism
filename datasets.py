from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.transforms import NormalizeFeatures, Constant
from collections import defaultdict
import torch
from torch_geometric.datasets import TUDataset
from collections import Counter
import torch
from random import shuffle
from torch_geometric.data import DataLoader
import random

dataset = TUDataset(root='datasets/MUTAG', name='MUTAG')
dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES')
dataset = TUDataset(root='datasets/COX2', name='COX2')
dataset = TUDataset(root='datasets/Letter-high', name='Letter-high', transform=Constant(1))
dataset = TUDataset(root='datasets/Letter-low', name='Letter-low', transform=Constant(1))
dataset = TUDataset(root='datasets/TRIANGLES', name='TRIANGLES')
dataset = TUDataset(root='datasets/PROTEINS', name='IMDB-MULTI', transform=Constant(1))


def load_dataset(name, verbose=True):

    if name == 'MUTAG':
        dataset = TUDataset(root='datasets/MUTAG', name='MUTAG')
    elif name == 'ENZYMES':
        dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES')
    elif name == 'COX2':
        dataset = TUDataset(root='datasets/COX2', name='COX2')
    elif name == 'Letter-high':
        dataset = TUDataset(root='datasets/Letter-high', name='Letter-high', transform=Constant(1))
    elif name == 'Letter-low':
        dataset = TUDataset(root='datasets/Letter-low', name='Letter-low', transform=Constant(1))
    elif name == 'TRIANGLES':
        dataset = TUDataset(root='datasets/TRIANGLES', name='TRIANGLES')
    elif name == 'IMDB-MULTI':
        dataset = TUDataset(root='datasets/PROTEINS', name='IMDB-MULTI', transform=Constant(1))                 
    else: 
        raise ValueError("invading dataset name")
    
    if verbose:
        # information about the dataset
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        # information about the first graph in the dataset
        data = dataset[0]
        print()
        print(data)
        print('=============================================================')

        # some statistics about the dataset
        num_nodes_list = [data.num_nodes for data in dataset]
        num_edges_list = [data.num_edges for data in dataset]
        mean_features_list =[data.x.mean(dim = 0) for data in dataset]
        classes = {}
        for data in dataset:
            if data.y.item() not in classes:
                classes[data.y.item()] = 1
            else:
                classes[data.y.item()] += 1

        print(f'Average number of nodes: {sum(num_nodes_list) / len(num_nodes_list)}')
        print(f'Average number of edges: {sum(num_edges_list) / len(num_edges_list)}')
        print(f'Average node features: {sum(mean_features_list) / len(mean_features_list)}')
        print(f'Class frequency:{classes}')
    
    return dataset


def dataset_class_distribution(dataset):

    # Create a dictionary to hold features for each class
    class_features = defaultdict(list)

    # Extract node features for each graph and group them by class
    for graph in dataset:
        features = graph.x  # Node features (tensor of shape [num_nodes, num_features])
        graph_class = graph.y.item()  # Class label
        class_features[graph_class].append(features)

    # Combine the node features for each class
    for class_label in class_features:
        class_features[class_label] = torch.cat(class_features[class_label], dim=0)

    # Now class_features[class_label] contains all node features for graphs of that class

    class_stats = {}

    for class_label, features in class_features.items():
        mean = torch.mean(features, dim=0)  # Mean of features across all nodes in this class
        std = torch.std(features, dim=0)    # Standard deviation of features in this class
        class_stats[class_label] = {'mean': mean, 'std': std}

        print(f"Class {class_label}:")
        print(f"Mean of features: {mean}")
        print(f"Standard deviation of features: {std}")
        print("-" * 40)


def get_train_val_test_loaders(dataset):
    # Shuffle the dataset and define the train, validation, and test split ratios
    dataset = dataset.shuffle()

    # Define train, validation, and test split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoader objects for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # make true later
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Check the dataset
    print(f"Number of graphs in the training set: {len(train_dataset)}")
    print(f"Number of graphs in the validation set: {len(val_dataset)}")
    print(f"Number of graphs in the test set: {len(test_dataset)}")

    return train_loader, val_loader, test_loader



def get_m_shot_loaders(dataset, m):
    
    samples_per_class = {}
    remaining_data = []

    # Collect samples for each class
    for data in dataset:
        label = data.y.item()
        if label not in samples_per_class:
            samples_per_class[label] = []
        if len(samples_per_class[label]) < m:
            samples_per_class[label].append(data)
        else:
            # Keep remaining data for test/validation
            remaining_data.append(data)

    # Shuffle the training data within each class
    final_train_dataset = []
    for label in samples_per_class:
        shuffle(samples_per_class[label])
        final_train_dataset += samples_per_class[label][:m]

    # Convert to PyTorch dataset
    final_train_dataset = torch.utils.data.Subset(dataset, [dataset.index(data) for data in final_train_dataset])

    # Shuffle remaining data
    shuffle(remaining_data)

    # Split remaining data into test (50%) and validation (50%)
    split_ratio = 0.8
    split_idx = int(split_ratio * len(remaining_data))
    test_dataset = remaining_data[:split_idx]
    val_dataset = remaining_data[split_idx:]

    # Convert test and validation to PyTorch subsets
    final_test_dataset = torch.utils.data.Subset(dataset, [dataset.index(data) for data in test_dataset])
    final_val_dataset = torch.utils.data.Subset(dataset, [dataset.index(data) for data in val_dataset])

    train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(final_val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader



class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_graph = self.dataset[idx]
        anchor_class = anchor_graph.y.item()

        # Sample positive example (same class)
        positive_idx = random.choice([i for i, graph in enumerate(self.dataset) if graph.y.item() == anchor_class])
        positive_graph = self.dataset[positive_idx]

        # Sample negative example (different class)
        negative_idx = random.choice([i for i, graph in enumerate(self.dataset) if graph.y.item() != anchor_class])
        negative_graph = self.dataset[negative_idx]

        return anchor_graph, positive_graph, negative_graph



def get_triplet_loader(dataset):
    triplet_dataset = TripletDataset(dataset)
    triplet_loader = DataLoader(triplet_dataset, batch_size=1, shuffle=False)
    return triplet_loader