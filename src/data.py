
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


def load_dataset(name, verbose=True):

    if name == 'MUTAG':
        dataset = TUDataset(root='datasets/MUTAG', name='MUTAG')
    elif name == 'ENZYMES':
        dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES')
    elif name == 'PROTEINS':
        dataset = TUDataset(root='datasets/PROTEINS', name='PROTEINS')
    elif name == 'COX2':
        dataset = TUDataset(root='datasets/COX2', name='COX2')
    elif name == 'Letter-high':
        dataset = TUDataset(root='datasets/Letter-high', name='Letter-high', use_node_attr=True)
    elif name == 'Letter-low':
        dataset = TUDataset(root='datasets/Letter-low', name='Letter-low', use_node_attr=True)
    elif name == 'TRIANGLES':
        dataset = TUDataset(root='datasets/TRIANGLES', name='TRIANGLES', transform=Constant(1))
    elif name == 'IMDB-MULTI':
        dataset = TUDataset(root='datasets/IMDB-MULTI', name='IMDB-MULTI', transform=Constant(1))                 
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

def get_dataloaders(dataset, train_num, val_num, test_num=0):
    # Ensure the total number of requested samples does not exceed the dataset size
    total_requested = train_num + val_num + test_num
    if total_requested > len(dataset):
        raise ValueError("Requested more samples than available in the dataset")

    # Shuffle and split the dataset
    shuffled_dataset = dataset.shuffle()
    train_dataset = shuffled_dataset[:train_num]
    val_dataset = shuffled_dataset[train_num:train_num + val_num]
    test_dataset = shuffled_dataset[train_num + val_num:train_num + val_num + test_num]

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) if train_num > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_num > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) if test_num > 0 else None

    return train_loader, val_loader, test_loader


def get_datasubsets(dataset, train_num, val_num, test_num=0):
    # Ensure the total number of requested samples does not exceed the dataset size
    total_requested = train_num + val_num + test_num
    if total_requested > len(dataset):
        raise ValueError("Requested more samples than available in the dataset")

    # Shuffle and split the dataset
    shuffled_dataset = dataset.shuffle()
    train_dataset = shuffled_dataset[:train_num]
    val_dataset = shuffled_dataset[train_num:train_num + val_num]
    test_dataset = shuffled_dataset[train_num + val_num:train_num + val_num + test_num]

    return train_dataset, val_dataset, test_dataset