import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from cvxpylayers.torch import CvxpyLayer

from data import data_dir, dataset_reports
from distance_functions import DistanceFuntion, build_mcs_constraints, subgraph_isomorphism_distance



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class PoolNet(torch.nn.Module):
    def __init__(self, max_nodes, in_channels, hidden_channels, out_channels, two_stage=True):
        super().__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(in_channels, hidden_channels, num_nodes)
        self.gnn1_embed = GNN(in_channels, hidden_channels, hidden_channels, lin=False)
        
        self.two_stage = two_stage
        if two_stage:
            num_nodes = ceil(0.25 * num_nodes)
            self.gnn2_pool = GNN(3 * hidden_channels, hidden_channels, num_nodes)
            self.gnn2_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embed = GNN(3 * hidden_channels, hidden_channels, out_channels, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)

        if self.two_stage:
            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)

            x, adj, l2, e2 = dense_diff_pool(x, adj, s)
            l = l+l2
            e = e+e2

        x = self.gnn3_embed(x, adj)
        
        return x, adj, l, e
    
    
class ReadoutNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(3 * in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, adj, l=0, e=0, mask=None):
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        distances = torch.cdist(x,x)
        
        return distances, l, e
        # return F.log_softmax(x, dim=-1), adj, l, e

class MetricNet(torch.nn.Module):
    def __init__(self, max_nodes, in_channels, out_channels, lam_init : float = 2, metric=subgraph_isomorphism_distance, mapping= 'integral'):
        self.max_nodes = max_nodes
        self.lam = torch.nn.Parameter(torch.tensor(lam_init, dtype=float))
        self.metric = metric
        self.mapping = mapping

        self.lin1 = torch.nn.Linear(3 * in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)

    
    def __differentiable_dist_compute__(self,data1, adj1, data2, adj2):
        
        C = torch.cdist(data1, data2)

        d, X, _ = self.metric(data1.detach().clone().cpu(), adj1.detach().clone().cpu(),
                                                 data2.detach().clone().cpu(), adj2.detach().clone().cpu(),
                                                   self.lam.detach().clone().cpu(), mapping=self.mapping, C = C.detach().clone().cpu())
        
        C = torch.nn.functional.pad(C, (0, 1, 0, 1), value=self.lam)
        C[-1,-1] = 0

        X = torch.tensor(X, device=C.device)

        return (C*X).sum()

    def compute_distance(self, x1, adj1, x2, adj2):
        x1 = self.lin1(x1).relu()
        x1 = self.lin2(x1)

        x2 = self.lin1(x2).relu()
        x2 = self.lin2(x2)

        C = torch.cdist(x1, x2)

        d, X, _ = self.metric(x1.detach().clone().cpu(), adj1.detach().clone().cpu(),
                                                 x2.detach().clone().cpu(), adj2.detach().clone().cpu(),
                                                   self.lam.detach().clone().cpu(), mapping=self.mapping, C = C.detach().clone().cpu())
        
        return d


    def forward(self, x, adj, l=0, e=0, mask=None):
        batch_size = x.size(0)

        for i in range(batch_size):
            x[i] = self.lin1(x[i]).relu()
            x[i] = self.lin2(x[i])

        distances = torch.zeros(batch_size, batch_size)

        for idx1 in range(batch_size):
            data1 = x[idx1]
            adj1 = adj[idx1]
            for idx2 in range(idx1+1,batch_size):
                data2 = x[idx2]
                adj2 = adj[idx2]

                dist = self.__differentiable_dist_compute__(data1, adj1, data2, adj2)
                distances[idx1,idx2] = dist
                distances[idx2,idx1] = dist
        
        return distances, l, e



def compute_loss(distances, l, e, labels):

    # Calculate pool_loss
    pool_loss = l + e  # Assuming l and e are tensors with the same shape as the batch size
    # Add pool_loss to the distances using broadcasting
    # distances += pool_loss.unsqueeze(0) + pool_loss.unsqueeze(1)
    # Create a label comparison matrix
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    # Convert to float for masking
    label_matrix = label_matrix.float()
    # Adjust distances: positive for same labels, negative for different labels
    loss = distances * label_matrix - distances * (1 - label_matrix)

    # return loss.sum()
    return loss.sum() + l.sum() + e.sum()
    

def run_pooling_experiment(dataset_name ,device = 'cpu', run_name=None):
    root_dir = 'pooling_experiment_out'
    
    max_nodes = 160

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
    #                 'PROTEINS_dense')
    dataset = TUDataset(
        root=data_dir+f'/{dataset_name}',
        name=f'{dataset_name}',
        transform=T.ToDense(max_nodes),
        pre_filter=lambda data: data.num_nodes <= max_nodes,
    )

    dataset = dataset.shuffle()
    n = (len(dataset) + 9) // 10
    test_dataset = dataset[:n]
    val_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    test_loader = DenseDataLoader(test_dataset, batch_size=20)
    val_loader = DenseDataLoader(val_dataset, batch_size=20)
    train_loader = DenseDataLoader(train_dataset, batch_size=20, shuffle=True)

    in_channels = dataset.num_features
    hidden_channels = in_channels*4
    out_channels = 8

    
    model_metric = torch.nn.Sequential(PoolNet(max_nodes=max_nodes, in_channels=in_channels,
                                                 hidden_channels=hidden_channels,
                                                 out_channels=hidden_channels,two_stage=True),
                                        MetricNet(max_nodes=max_nodes, in_channels=hidden_channels, out_channels=out_channels))


    model_readout = torch.nn.Sequential(PoolNet(max_nodes=max_nodes, in_channels=in_channels,
                                                 hidden_channels=hidden_channels,
                                                 out_channels=hidden_channels,two_stage=True),
                                        ReadoutNet(in_channels=hidden_channels, out_channels=out_channels))

    model_metric.to(device)
    model_readout.to(device)

    optimizer_metric = torch.optim.Adam(model_metric.parameters(), lr=0.01)
    optimizer_readout = torch.optim.Adam(model_readout.parameters(), lr=0.01)

    models = {'Metric': model_metric, 'Readout': model_readout}
    optimizers = {'Metric': optimizer_metric, 'Readout': optimizer_readout}
    epoch_losses = {'Metric': [], 'Readout': []}
    best_val_acc = {'Metric': 0, 'Readout': 0}
    test_acc = {'Metric': 0, 'Readout': 0}
    times = {'Metric': [], 'Readout': []}

    val_accuracies = {'Metric': [], 'Readout': []}
    
    num_epochs = 5

    for epoch in range(1, num_epochs+1):
        for name, model in models:
            
            print(f"Training {name} Model:")
            start = time.time()

            train_loss = train(epoch,model=model, optimizer=optimizers[name],train_loader=train_loader,device=device)
            epoch_losses[name].append(train_loss)

            val_acc = test(val_loader)
            val_accuracies[name].append(val_acc)

            if val_acc > best_val_acc[name]:
                test_acc[name] = test(test_loader)
                best_val_acc[name] = val_acc

            print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f} Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            times[name].append(time.time() - start)
            print(f"Median time per epoch: {torch.tensor(times[name]).median():.4f}s")
            print("=====================================================================")



def train(epoch, model, optimizer, train_loader, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        distances, l, e = model(data.x, data.adj, data.mask)
        loss = compute_loss(distances, l, e, data.y)
        # loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        # loss_all += data.y.size(0) * float(loss)
        optimizer.step()
        loss_all += loss
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader, train_loader, model, device):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)

def find_knn()


# class SubGrapIsomorphismLayer(torch.nn.Module):
#     def __init__(self, max_nodes, lam, feature_dim, mapping = 'integral'):
#         self.lam = lam
        
        
#         if mapping == 'integral':
#             X = cp.Variable((max_nodes+1, max_nodes+1), boolean=True)
#         else:
#             X = cp.Variable((max_nodes+1, max_nodes+1), nonneg=True)

#         C = cp.Parameter((max_nodes+1, max_nodes+1))
#         A_G = cp.Parameter((max_nodes+1, max_nodes+1))
#         A_H = cp.Parameter((max_nodes+1, max_nodes+1))
#         E = cp.Parameter((max_nodes+1, max_nodes+1), boolean = True) # indicates the actual nodes of graph
#         # C[max_nodes, max_nodes] = 0  # C(0,0) = 0
#         # X = Extras.T @ BaseX @ Extras
#         objective = cp.sum(cp.multiply(C, X))
#         constraints = build_mcs_constraints(X,max_nodes,max_nodes,A_G,A_H)
        
#         constraints.append(X @ E == 0)
#         constraints.append(E.T @ X == 0)
        
#         if mapping == 'integral':
            
#             problem.solve(solver=cp.GLPK_MI)#, max_iter = 1000)#, verbose=True)
#         else:
#             problem.solve(solver=cp.SCS)

#         problem = cp.Problem(cp.Minimize(objective), constraints)
#         # assert problem.is_dpp()

#         self.cvx_layer = CvxpyLayer(problem,[C, A_G, A_H, E],[X]) 

#     def build_params():
#         pass

#     def forward(self, x, adj, l=0, e=0, mask=None):

#         pass