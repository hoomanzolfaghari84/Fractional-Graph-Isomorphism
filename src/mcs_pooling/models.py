import logging
import time
from math import ceil
import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from concurrent.futures import ThreadPoolExecutor, TimeoutError
# import rustworkx as rx

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

class BaseMetricNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_distances(self, x1, adj1, x2, adj2, mask1=None, mask2=None):
        raise Exception('Base Class Method Called.')

    def forward(self, x, adj, l=0, e=0, mask=None):
        raise Exception('Base Class Method Called.')


class ReadoutNet(BaseMetricNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(3 * in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def compute_distances(self, x1, adj1, x2, adj2, mask1 =None, mask2 =None):

        if self.training: raise Exception('The compute_distance() function is not for training mode')
        
        x1 = self.get_final_embeddings(x1)
        x2 = self.get_final_embeddings(x2)

        return torch.cdist(x1,x2)
    
    def get_final_embeddings(self, x):
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x

    def forward(self, x, adj, l=0, e=0, mask=None):
        x = self.get_final_embeddings(x)

        distances = torch.cdist(x,x)
        
        return distances, l, e
        # return F.log_softmax(x, dim=-1), adj, l, e

def get_actual_graph(x, adj, mask):
    """
    Extract the actual adjacency matrix and node features for a single graph.

    Args:
        x (torch.Tensor): Node feature matrix of shape [max_num_nodes, feature_dim].
        adj (torch.Tensor): Dense adjacency matrix of shape [max_num_nodes, max_num_nodes].
        mask (torch.Tensor): Node mask of shape [max_num_nodes].

    Returns:
        torch.Tensor: Filtered node feature matrix of shape [num_nodes, feature_dim].
        torch.Tensor: Filtered adjacency matrix of shape [num_nodes, num_nodes].
    """
    # Get indices of valid nodes
    valid_indices = mask.nonzero(as_tuple=True)[0]

    # Use the indices to slice node features and adjacency matrix
    filtered_x = x[valid_indices]          # Shape: [num_nodes, feature_dim]
    filtered_adj = adj[valid_indices][:, valid_indices]  # Shape: [num_nodes, num_nodes]

    return filtered_x, filtered_adj

class AssembledModel(torch.nn.Module):
    def __init__(self, pool_net : PoolNet, metric_net: BaseMetricNet):
        super().__init__()
        self.pool_net = pool_net
        self.metric_net = metric_net

    def pool_batch(self, x, adj, mask=None):
        if self.training : raise Exception('pooling alone is not for training') 
        x, adj, l, e = self.pool_net(x, adj, mask)  # Pass through PoolNet
        return x, adj, l, e

    def compute_distances(self, x1, adj1, x2, adj2):
        return self.metric_net.compute_distances(x1, adj1, x2, adj2)
    
    
        
    def forward(self, x, adj, mask=None):
        x, adj, l, e = self.pool_net(x, adj, mask)  # Pass through PoolNet
        distances, l, e = self.metric_net(x, adj, l, e, mask)  # Pass through MetricNet
        return distances, l, e



# def safe_compute_distance(func, *args, timeout=180, **kwargs):
#     """
#     Execute a function with a timeout and fallback.
#
#     Args:
#     - func: The function to execute.
#     - args: Positional arguments for the function.
#     - timeout: Timeout in seconds.
#     - kwargs: Additional keyword arguments for the function.
#
#     Returns:
#     - result or None if timeout occurred.
#     """
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(func, *args, **kwargs)
#         try:
#             result = future.result(timeout=timeout)
#             return result
#         except TimeoutError:
#             logging.error(f"Timeout ({timeout}s) in distance computation for inputs: {args} {kwargs}")
#             return None


class MetricNet(BaseMetricNet):
    def __init__(self, max_nodes, in_channels, out_channels, lam_init : float = 2, mapping= 'integral',timeout=180):
        super().__init__()  # Ensure Module initialization happens first

        self.max_nodes = max_nodes
        self.__lam__ = torch.nn.Parameter(torch.tensor(lam_init, dtype=torch.float32))
        self.mapping = mapping
        self.timeout = timeout

        self.lin1 = torch.nn.Linear(3 * in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def get_lam(self): # constraint lambda parameter to satisfy theoretical constraints, while being differentiable in R.
        return F.softplus(self.__lam__)

    # @torch.no_grad()
    def __get_mapping__(self, x1, adj1, x2, adj2):
        """
        needs unmasked graph features and adj 
        """
        import networkx as nx
        G1 = nx.Graph()
        G2 = nx.Graph()
        for i in range(len(adj1)):
            for j in range(len(adj1)):
                if adj1[i,j] >= 0.5: G1.add_edge(i,j)
        for i in range(len(adj2)):
            for j in range(len(adj2)):
                if adj2[i,j] >= 0.5: G2.add_edge(i,j)

        matcher = nx.algorithms.isomorphism.ISMAGS(G1, G2)
        mcs_mappings_iter = matcher.largest_common_subgraph()

        return mcs_mappings_iter
     
    def __compute_dist__(self,data1, adj1, data2, adj2):
        """
        needs unmasked graph features and adj 
        """
        num_v_1 = len(data1)
        num_v_2 = len(data2)
        dists = []
        # d, X, _ = self.__compute_nondif_metric(data1, adj1, data2, adj2, C)
        mappings_iter = self.__get_mapping__(data1, adj1, data2, adj2)
        for mapping in mappings_iter:
            X_adj = torch.zeros((len(data1), len(data2)), dtype=torch.float32, device=data1.device)
            X = torch.zeros((len(data1)+1, len(data2)+1), dtype=torch.float32, device=data1.device)

            C = torch.cdist(data1, data2)

            for u, i in mapping.items():
                X[u, i] = 1.0

            for i in range(len(data1)):
                if X[i].sum() == 0: X[i, num_v_2] = 1
                if X[:,i].sum() == 0: X[num_v_1, i] = 1

            lam = self.get_lam()
            # Manual padding for `C`
            pad_row = lam.expand(1, C.size(1))  # Create a row of `lam` with correct dimensions
            pad_col = lam.expand(C.size(0) + 1, 1)  # Create a column of `lam` with correct dimensions

            C = torch.cat([C, pad_row], dim=0)  # Add padding row
            C = torch.cat([C, pad_col], dim=1)  # Add padding column
            C[-1, -1] = 0  # Set bottom-right corner to 0

            dist = (C*X).sum() + torch.norm(adj1 - torch.matmul(X_adj, adj2 @ X_adj.T), p='fro') + torch.norm(adj2-torch.matmul(X_adj.T, adj1 @ X_adj), p='fro')
            dists.append(dist)

        if len(dists) == 0: return num_v_1*num_v_2*self.get_lam()

        return torch.as_tensor(dists).min()

    def compute_distances(self, x1, adj1, x2, adj2, mask1 =None, mask2 =None):

        if self.training: raise Exception('The compute_distance() function is not for training mode')
        
        distances = torch.zeros((len(x1), len(x2)), device=x1.device)

        x1 = self.lin1(x1).relu()
        x1 = self.lin2(x1)

        x2 = self.lin1(x2).relu()
        x2 = self.lin2(x2)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in range(len(x1)):
                futures = []
                for j in range(len(x2)):
                    
                    futures.append(executor.submit(self.__compute_dist__, x1[i], adj1[i], x2[j], adj2[j]))
                    
                for j in range(len(x2)):
                    distances[i,j] = futures[j].result()

        return distances
    
    
    def forward(self, x, adj, l=0, e=0, mask=None):
        
        batch_size = x.size(0)
        
        x = self.lin1(x).relu()
        x = self.lin2(x)

        # debug_tensor('lin final',x)

        distances = torch.zeros(batch_size, batch_size, device=x.device)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            
            for idx1 in range(batch_size):
                data1 = x[idx1]
                adj1 = adj[idx1]
                futures = []
                for idx2 in range(idx1+1,batch_size):
                    data2 = x[idx2]
                    adj2 = adj[idx2]
                    # for row in adj1:
                    #     if all(row == 0) : logging.warning(f"Training Graph at idx {idx1} has singular node")

                    futures.append(executor.submit(self.__compute_dist__, data1, adj1, data2, adj2))
                
                for idx2 in range(idx1+1,batch_size):
                    dist = futures[idx2-idx1-1].result()
                    distances[idx1,idx2] = dist
                    distances[idx2,idx1] = dist
        
        return distances, l, e
    
""" This module contains an implementation of a Vantage Point-tree (VP-tree)."""
import bisect
import collections
import math
import statistics as stats
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Process, Queue

# def compute_distance(args):
#     data1, data2, dist_fn = args
#     return dist_fn(data1[0].unsqueeze(0),data1[1].unsqueeze(0), data2[0].unsqueeze(0), data2[1].unsqueeze(0)).squeeze(0)

def compute_distance(data1, data2, dist_fn, debug=False):
    if debug: print(f"Shapes: data1[0]={data1[0].shape}, data1[1]={data1[1].shape}, data2[0]={data2[0].shape}, data2[1]={data2[1].shape}")

    return dist_fn(data1[0].unsqueeze(0),data1[1].unsqueeze(0), data2[0].unsqueeze(0), data2[1].unsqueeze(0)).squeeze(0)

class KNNSearcher:
    """
        Parameters
        ----------
        points : Iterable
            Construction points.
        dist_fn : Callable
            Function taking two point instances as arguments and returning
            the distance between them.


    """
    def __init__(self, points, dist_fn):
        self.points = points  # Store all points
        self.dist_fn = dist_fn  # Distance function

    def get_n_nearest_neighbors(self, query, n_neighbors):
        """
            Get `n_neighbors` nearest neighbors to the query point.

            Parameters
            ----------
            query : tuple
                Query point `(x, adj)`.
            n_neighbors : int
                Number of neighbors to fetch.

            Returns
            -------
            list
                List of tuples `(distance, point)` for the nearest neighbors.
        """

class BruteForceKNN(KNNSearcher):
    """
    Brute Force k-NN implementation for a dataset of graphs.

    Parameters
    ----------
    points : list
        List of points, where each point is a tuple (x, adj, label).
    dist_fn : Callable
        Function to compute distances between two points.
    """
    def __init__(self, points, dist_fn):
        super().__init__(points, dist_fn)

    def get_n_nearest_neighbors(self, query, n_neighbors):
        """
        Get `n_neighbors` nearest neighbors to the query point.

        Parameters
        ----------
        query : tuple
            Query point `(x, adj)`.
        n_neighbors : int
            Number of neighbors to fetch.

        Returns
        -------
        list
            List of tuples `(distance, point)` for the nearest neighbors.
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a strictly positive integer")

        # Use the CustomSortedNeighbors to maintain top k neighbors
        neighbors = CustomSortedNeighbors(max_size=n_neighbors)

        for point in self.points:
            distance = compute_distance(query, point, self.dist_fn)  # Compute distance
            neighbors.add(distance, point)

        return neighbors.get_neighbors()




class VPTree(KNNSearcher): #TODO: This class is used in Validation and Testing, thus not needing Gradients. We might be able to multi process.

    """ VP-Tree data structure for efficient nearest neighbor search.

    The VP-tree is a data structure for efficient nearest neighbor
    searching and finds the nearest neighbor in O(log n)
    complexity given a tree constructed of n data points. Construction
    complexity is O(n log n).

    Parameters
    ----------
    points : Iterable
        Construction points.
    dist_fn : Callable
        Function taking two point instances as arguments and returning
        the distance between them.

    max_workers: Number of parallel workers.(IGNORED)

    leaf_size : int
        Minimum number of points in leaves (IGNORED).
    """

    def __init__(self, points, dist_fn, max_workers=4, depth=0):#, max_parallel_depth=3, depth=0, master_process = True):
        super().__init__(points, dist_fn)
        self.left = None
        self.right = None
        self.left_min = math.inf
        self.left_max = 0
        self.right_min = math.inf
        self.right_max = 0
        self.dist_fn = dist_fn
        self.max_workers = max_workers

        

        if not len(points):
            raise ValueError('Points can not be empty.')

        # Vantage point is point furthest from parent vp.
        self.vp = points[0]
        points = points[1:]

        if len(points) == 0:
            return

        # # Choose division boundary at median of distances.
        dist_start = time.time()

        distances = [compute_distance(self.vp, p, self.dist_fn) for p in points]

        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # the task is not I.O bound and Multiprocess difficulties with PyTorch gradient management
        #     distances = list(executor.map(lambda p: compute_distance(self.vp, p, self.dist_fn), points))
        if len(points)>200:
            logging.info(f"Distance computation took {time.time() - dist_start:.2f}s for {len(points)} points")

        median = stats.median(distances)


        left_points = []
        right_points = []
        for point, distance in zip(points, distances):
            if distance >= median:
                self.right_min = min(distance, self.right_min)
                if distance > self.right_max:
                    self.right_max = distance
                    right_points.insert(0, point) # put furthest first
                else:
                    right_points.append(point)
            else:
                self.left_min = min(distance, self.left_min)
                if distance > self.left_max:
                    self.left_max = distance
                    left_points.insert(0, point) # put furthest first
                else:
                    left_points.append(point)
        
        left_points = []
        right_points = []
        for point, distance in zip(points, distances):
            if distance >= median:
                self.right_min = min(distance, self.right_min)
                if distance > self.right_max:
                    self.right_max = distance
                    right_points.insert(0, point) # put furthest first
                else:
                    right_points.append(point)
            else:
                self.left_min = min(distance, self.left_min)
                if distance > self.left_max:
                    self.left_max = distance
                    left_points.insert(0, point) # put furthest first
                else:
                    left_points.append(point)

        # ratio = len(left_points)/len(points)

        # if ratio<0.2 or ratio> 0.8:
        #     logging.warning(f"Unbalanced Tree split. Partition Sizes -> Left: {len(left_points)}, Right: {len(right_points)}")


        if len(left_points) > 0:
            self.left = VPTree(points=left_points, dist_fn=self.dist_fn, depth=depth+1)
       
        if len(right_points) > 0:
            self.right = VPTree(points=right_points, dist_fn=self.dist_fn, depth=depth+1)


    # @staticmethod
    # def _build_subtree_parallel(points, max_parallel_depth, depth, dist_fn, max_workers):
    #     """
    #     Helper method for parallel subtree construction.
    #     """
    #     if not points:
    #         return None
    #     return VPTree(points, dist_fn, max_workers, max_parallel_depth, depth, master_process=False)

    # def _build_subtree(self, points, max_parallel_depth, depth, master_process):
    #     """
    #     Helper method for sequential subtree construction.
    #     """
    #     if not points:
    #         return None
    #     return VPTree(points, self.dist_fn, self.max_workers, max_parallel_depth, depth, master_process)

        # if len(left_points) > 0:
        #     self.left = VPTree(points=left_points, dist_fn=self.dist_fn)

        # if len(right_points) > 0:
        #     self.right = VPTree(points=right_points, dist_fn=self.dist_fn)

    def _is_leaf(self):
        return (self.left is None) and (self.right is None)

    def get_nearest_neighbor(self, query):
        """ Get single nearest neighbor.
        
        Parameters
        ----------
        query : Any
            Query point.

        Returns
        -------
        Any
            Single nearest neighbor.
        """
        return self.get_n_nearest_neighbors(query, n_neighbors=1)[0]

    def get_n_nearest_neighbors(self, query, n_neighbors):
        """
        Get `n_neighbors` nearest neighbors to the query.

        Parameters
        ----------
        query : tuple
            Query point `(x, adj)`.
        n_neighbors : int
            Number of neighbors to fetch.

        Returns
        -------
        list
            List of tuples `(distance, point)` for the nearest neighbors.
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be a strictly positive integer')

        neighbors = CustomSortedNeighbors(max_size=n_neighbors)
        queue = collections.deque([self])  # BFS queue for tree traversal

        while queue:
            node = queue.popleft()
            if node is None:
                continue

            # Compute distance between query and current vantage point
            d = compute_distance(query, node.vp, self.dist_fn)

            # Add current vantage point to the neighbors heap
            # logging.info(f'dist : {d}')
            # logging.info(f'dist : {d.item()}')
            # input()
            neighbors.add(d.item(), node.vp)
            # print('passed')
            # Check if the subtrees need to be traversed based on triangle inequality
            if node.left is not None and d < node.left_max + abs(neighbors.heap[0][0]):
                queue.append(node.left)
            if node.right is not None and d >= node.right_min - abs(neighbors.heap[0][0]):
                queue.append(node.right)

        return neighbors.get_neighbors()


    def get_all_in_range(self, query, max_distance):
        """ Find all neighbours within `max_distance`.

        Parameters
        ----------
        query : Any
            Query point.
        max_distance : float
            Threshold distance for query.

        Returns
        -------
        neighbors : list
            List of points within `max_distance`.

        Notes
        -----
        Returned neighbors are not sorted according to distance.
        """
        neighbors = list()
        nodes_to_visit = [(self, 0)]

        while len(nodes_to_visit) > 0:
            node, d0 = nodes_to_visit.pop(0)
            if node is None or d0 > max_distance:
                continue

            d = compute_distance(query, self.vp, self.dist_fn)
            if d < max_distance:
                neighbors.append((d, node.vp))

            if node._is_leaf():
                continue

            if node.left_min <= d <= node.left_max:
                nodes_to_visit.insert(0, (node.left, 0))
            elif node.left_min - max_distance <= d <= node.left_max + max_distance:
                nodes_to_visit.append((node.left,
                                       node.left_min - d if d < node.left_min
                                       else d - node.left_max))

            if node.right_min <= d <= node.right_max:
                nodes_to_visit.insert(0, (node.right, 0))
            elif node.right_min - max_distance <= d <= node.right_max + max_distance:
                nodes_to_visit.append((node.right,
                                       node.right_min - d if d < node.right_min
                                       else d - node.right_max))

        return neighbors


from heapq import heappush, heappop

from heapq import heappush, heappop

class CustomSortedNeighbors:
    """Custom heap-like data structure for managing nearest neighbors."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []
        self.counter = 0  # To ensure unique ordering for equal distances

    def add(self, distance, point):
        """Add a new neighbor with distance and point."""
        heappush(self.heap, (-distance, self.counter, point))  # Include counter for tie-breaking
        self.counter += 1

        # Maintain the heap size
        if len(self.heap) > self.max_size:
            heappop(self.heap)

    def get_neighbors(self):
        """Return sorted list of neighbors by distance (ascending)."""
        return [(p, -d) for d, _, p in sorted(self.heap, key=lambda x: -x[0])]



