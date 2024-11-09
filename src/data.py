
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


def balanced_subset(dataset, m):
    # Create a dictionary to store m graphs per class for the training set
    label_dict = {}
    train_data = []
    train_indices = set()

    # Create the balanced training set
    for idx, data in enumerate(dataset):
        label = data.y.item()
        if label not in label_dict:
            label_dict[label] = []
        
        if len(label_dict[label]) < m:
            label_dict[label].append(data)
            train_data.append(data)
            train_indices.add(idx)
        
        # Stop if we have m samples for each label
        if all(len(samples) >= m for samples in label_dict.values()):
            break

    return train_data, train_indices

def get_datasubsets(dataset, train_num, val_num, test_num=0, each_class_train=None):

    # Ensure the total number of requested samples does not exceed the dataset size
    if each_class_train is not None:
        total_requested = each_class_train * dataset.num_classes  + val_num + test_num
    else:
        total_requested = train_num + val_num + test_num

    if total_requested > len(dataset):
        raise ValueError("Requested more samples than available in the dataset")
    
    # Shuffle and split the dataset
    shuffled_dataset = dataset.shuffle()
    if each_class_train is not None:
        train_dataset, train_indices = balanced_subset(dataset, each_class_train)
        val_dataset, val_indices = [], set()
        # Create validation set excluding training indices
        for idx, data in enumerate(shuffled_dataset):
            if idx not in train_indices:
                val_dataset.append(data)
                val_indices.add(idx)
            if len(val_dataset) >= val_num: break
        
        test_dataset = []
        if test_num != 0:
            idx = 0
            while len(test_dataset) < test_num:
                if idx not in train_indices and idx not in val_indices:
                    test_dataset.append(shuffled_dataset[idx])
                idx = idx + 1
                
    else:
        train_dataset = shuffled_dataset[:train_num]
        val_dataset = shuffled_dataset[train_num:train_num + val_num]
        test_dataset = shuffled_dataset[train_num + val_num:train_num + val_num + test_num]

    return train_dataset, val_dataset, test_dataset