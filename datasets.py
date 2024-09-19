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
import random
import torch
from torch_geometric.data import DataLoader
from random import shuffle

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
        print(f'is directed: {data.is_directed()}')
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


def get_m_shot_loaders(dataset, m, batch_size=1, val_split=0.5):
    """
    Args:
    - dataset: PyTorch dataset where each sample has a 'y' attribute for the label.
    - m: Number of samples per class for training (m-shot learning).
    - batch_size: Batch size for the data loaders.
    - val_split: Fraction of the remaining data used for validation. The rest is used for testing.

    Returns:
    - train_loader: DataLoader for training (m samples per class).
    - val_loader: DataLoader for validation.
    - test_loader: DataLoader for testing.
    """

    # Dictionary to store samples per class
    samples_per_class = defaultdict(list)
    remaining_indices = []

    # Shuffle dataset indices
    dataset_indices = list(range(len(dataset)))
    shuffle(dataset_indices)

    # Collect samples and split m-shot for each class
    for idx in dataset_indices:
        data = dataset[idx]
        label = data.y.item()  # Assuming that each data has a 'y' attribute for the label
        if len(samples_per_class[label]) < m:
            samples_per_class[label].append(idx)
        else:
            remaining_indices.append(idx)

    # Create train dataset with m-shot samples per class
    train_indices = [idx for indices in samples_per_class.values() for idx in indices]
    train_subset = dataset.index_select(train_indices)
    train_loader = DataLoader(dataset[train_indices], batch_size=batch_size, shuffle=False) # later set shuffle true

    # Shuffle remaining data
    shuffle(remaining_indices)

    # Split remaining data into validation and test sets
    if val_split == 0:
        test_indices = remaining_indices
        test_loader = DataLoader(dataset[test_indices], batch_size=batch_size, shuffle=False)
    elif val_split == 1:
        val_indices = remaining_indices
        val_loader = DataLoader(dataset[val_indices], batch_size=batch_size, shuffle=False)
    else:
        val_size = int(val_split * len(remaining_indices))
        val_indices = remaining_indices[:val_size]
        test_indices = remaining_indices[val_size:]
        val_loader = DataLoader(dataset[val_indices], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_indices], batch_size=batch_size, shuffle=False)

    # # Create subsets for validation and test sets
    # val_subset = dataset.index_select(val_indices)
    # test_subset = dataset.index_select( test_indices)

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