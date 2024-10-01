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


def run_GP_experiment(dataset):
    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset)
    