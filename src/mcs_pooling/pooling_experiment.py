import os
import logging
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DenseDataLoader
import matplotlib.pyplot as plt
import numpy as np

from data import dataset_reports
from mcs_pooling.models import (
    BruteForceKNN, AssembledModel, MetricNet, PoolNet,
    ReadoutNet, VPTree, KNNSearcher
)
from mcs_pooling.utils import load_dataset, load_model, save_model, save_results_report, save_plot

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def contrastive_loss(distances, l, e, labels, margin=5.0, alpha=0.1, beta=0.1):
    """
    Compute contrastive loss for graph distance pairs.

    Args:
        distances (torch.Tensor): Pairwise distances (batch_size, batch_size).
        labels (torch.Tensor): Pairwise labels (1 for similar, 0 for dissimilar).
        l (float): Pooling link loss.
        e (float): Pooling entropy loss.
        margin (float): Margin for dissimilar pairs.
        alpha (float): Pooling link loss impact factor.
        beta (float): Pooling entropy loss impact factor.

    Returns:
        torch.Tensor: Contrastive loss value.
    """

    positive_loss = labels * distances.pow(2)
    negative_loss = (1 - labels) * F.relu(margin - distances).pow(2)
    loss = positive_loss + negative_loss
    return loss.mean() + alpha * l + beta * e


def run_pooling_experiment(dataset_name, train_num, val_num, test_num, device='cpu', run_name=None):
    """
        Run the pooling experiment and train the models.

        Args:
            dataset_name (str): Name of the dataset.
            train_num (int): Number of training samples.
            val_num (int): Number of validation samples.
            test_num (int): Number of test samples.
            device (str): Device to use ('cpu' or 'cuda').
            run_name (str): Optional experiment name for saving results.

        Returns:
            None
    """

    root_dir = './../outputs/pooling_experiment_out'
    
    # Define experiment directory
    if run_name:
        root_dir = os.path.join(root_dir, run_name)
    os.makedirs(root_dir, exist_ok=True) # Set this to False immediately after running so we don't mistakenly erase last results

    # Model paths
    model_metric_path = os.path.join(root_dir, 'model_metric')
    model_readout_path = os.path.join(root_dir, 'model_readout')
    results_path = os.path.join(root_dir, 'results.txt')

    ## Dataset and loader setup
    max_nodes = 160
    dataset = load_dataset(dataset_name, max_nodes)

    dataset_reports(dataset,path=os.path.join(root_dir, 'dataset_reports.txt'))

    train_dataset = dataset[:train_num]
    val_dataset = dataset[train_num:train_num + val_num]
    test_dataset = dataset[train_num + val_num:train_num + val_num + test_num]

    logging.info(f"Train num: {len(train_dataset)}")
    logging.info(f"Val num: {len(val_dataset)}")
    logging.info(f"Test num: {len(test_dataset)}")

    batch_size = 20
    train_loader = DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DenseDataLoader(val_dataset, batch_size=batch_size)
    test_loader = DenseDataLoader(test_dataset, batch_size=batch_size)

    ## model setups
    in_channels = dataset.num_features
    hidden_channels = in_channels * 4
    out_channels = 6

    pool_net1 = PoolNet(max_nodes=max_nodes, in_channels=in_channels,
                        hidden_channels=hidden_channels, out_channels=hidden_channels, two_stage=True)
    pool_net2 = PoolNet(max_nodes=max_nodes, in_channels=in_channels,
                        hidden_channels=hidden_channels, out_channels=hidden_channels, two_stage=True)

    metric_net = MetricNet(max_nodes=max_nodes, in_channels=hidden_channels, out_channels=out_channels)

    readout_net = ReadoutNet(in_channels=hidden_channels, out_channels=out_channels)

    model_metric = AssembledModel(pool_net1, metric_net)
    model_readout = AssembledModel(pool_net2, readout_net)

    model_metric.to(device)
    model_readout.to(device)


    # Optimizer and scheduler
    lr = 0.001
    optimizer_metric = torch.optim.AdamW(model_metric.parameters(), lr=lr, weight_decay=1e-5)
    scheduler_metric = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_metric, 'max', factor=0.1, patience=5)

    optimizer_readout = torch.optim.AdamW(model_readout.parameters(), lr=lr, weight_decay=1e-5)
    scheduler_readout = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_metric, 'max', factor=0.1, patience=5)

    # Load models and optimizers from checkpoint
    start_epoch_metric = load_model(model_metric, optimizer_metric, model_metric_path, device)
    start_epoch_readout = load_model(model_readout, optimizer_readout, model_readout_path, device)

    models = {'Metric': model_metric, 'Readout': model_readout}
    optimizers = {'Metric': optimizer_metric, 'Readout': optimizer_readout}
    schedulers = {'Metric': scheduler_metric, 'Readout': scheduler_readout}
    epoch_losses = {'Metric': [], 'Readout': []}
    best_val_acc = {'Metric': 0, 'Readout': 0}
    test_acc = {'Metric': 0, 'Readout': 0}
    times = {'Metric': [], 'Readout': []}
    val_accuracies = {'Metric': [], 'Readout': []}

    num_epochs = 1
    epoch_results = {}
    test_results = None

    for epoch in range(max(start_epoch_metric, start_epoch_readout), num_epochs + 1):
        for name, model in models.items():
            logging.info(f"Training {name} Model for Epoch {epoch}:")

            start = time.time()

            train_loss = train(epoch, model=model, optimizer=optimizers[name],
                               train_loader=train_loader, device=device)

            if name == "Metric":
                with torch.no_grad():
                    logging.info(f"Metric Net Lambda : {model.metric_net.get_lam()}")

                # distance_health_check(epoch, val_loader, model, device)

            epoch_losses[name].append(train_loss)
            with torch.no_grad():
                logging.info(f"Creating VP Tree")
                vp_tree = build_knn_searcher(train_loader, model, device, name != "Metric")
                logging.info(f"Validating")
                val_acc, _ = test(val_loader, vp_tree, model, device)
                val_accuracies[name].append(val_acc)

                schedulers[name].step(val_acc)
                if val_acc > best_val_acc[name]:
                    logging.info("Testing")
                    test_acc[name], test_results = test(test_loader, vp_tree, model, device)
                    best_val_acc[name] = val_acc

                save_model(model, optimizers[name], epoch,
                           model_metric_path if name == 'Metric' else model_readout_path)
                # logging.info(
                #     f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                #     f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc[name]:.4f}'
                # )
                logging.info(
                    f'Epoch: {epoch:03d}, Model: {name}, Train loss: {train_loss:.4f} Val Acc: {val_acc:.4f}, Test Acc: {test_acc[name]:.4f}')

                times[name].append(time.time() - start)
                logging.info(f"Median Time per Epoch for {name}: {torch.tensor(times[name]).median():.4f}s")
                epoch_results[
                    f'Epoch: {epoch:03d}, Model: {name}'] = f'Train loss: {train_loss:.4f} Val Acc: {val_acc:.4f}, Test Acc: {test_acc[name]:.4f} Median time: {torch.tensor(times[name]).median():.4f}s'
                print(f"Median time: {torch.tensor(times[name]).median():.4f}s")
                print("=====================================================================")

    # Save results and plots
    results = {
        'Dataset': dataset_name,
        'Best Validation Accuracy': best_val_acc,
        'Test Accuracy': test_acc,
        'Median Time per Epoch': {name: torch.tensor(times[name]).median().item() for name in times}
    }
    try:
        results += epoch_results
    except Exception as e:
        print(f"Error {e}")

    save_results_report(results, results_path)
    save_plot(epoch_losses['Metric'], "Metric Model Loss", "Epoch", "Loss", os.path.join(root_dir, "metric_loss.png"))
    save_plot(epoch_losses['Readout'], "Readout Model Loss", "Epoch", "Loss",
              os.path.join(root_dir, "readout_loss.png"))
    save_plot(val_accuracies['Metric'], "Metric Model Validation Accuracy", "Epoch", "Accuracy",
              os.path.join(root_dir, "metric_val_acc.png"))
    save_plot(val_accuracies['Readout'], "Readout Model Validation Accuracy", "Epoch", "Accuracy",
              os.path.join(root_dir, "readout_val_acc.png"))


@torch.no_grad()
def metric_health_check(epoch, data_loader, model: AssembledModel, device, path):
    """
    Performs health checks on the distance metric for a given dataset.

    Args:
        data_loader: DataLoader containing the dataset.
        model: Model containing the metric computation.
        device: Device (CPU or GPU) for computation.

    Returns:
        None
    """
    model.eval()
    logging.info("Starting distance health checks...")
    all_distances = []
    within_class_distances = []
    between_class_distances = []
    triangle_violations = 0
    total_triplets = 0

    data_points = []
    data_adjs = []
    labels = []

    # Collect embeddings and labels
    # i = 5
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            embeddings, adj, _, _ = model.pool_batch(batch.x, batch.adj, batch.mask)
            data_points.append(embeddings)
            data_adjs.append(adj)
            labels.append(batch.y)
            # i-=1
            # if i == 0:break

    data_points = torch.cat(data_points, dim=0)
    data_adjs = torch.cat(data_adjs, dim=0)
    labels = torch.cat(labels, dim=0)

    # Compute pairwise distances
    logging.info("Computing pairwise distances...")
    distances = model.compute_distances(data_points, data_adjs, data_points, data_adjs)

    # Flatten distances for global analysis
    all_distances = distances.flatten().cpu().numpy()

    # Class-wise distance computation
    num_points = len(data_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if labels[i] == labels[j]:
                within_class_distances.append(distances[i, j].item())
            else:
                between_class_distances.append(distances[i, j].item())

            # Check triangle inequality
            for k in range(num_points):
                if k != i and k != j:
                    total_triplets += 1
                    if distances[i, k] > distances[i, j] + distances[j, k]:
                        triangle_violations += 1

    # Plot distributions
    logging.info("Plotting distance distributions...")
    plt.figure(figsize=(12, 6))
    plt.hist(all_distances, bins=50, alpha=0.7, label="All Distances")
    plt.hist(within_class_distances, bins=50, alpha=0.7, label="Within-Class Distances")
    plt.hist(between_class_distances, bins=50, alpha=0.7, label="Between-Class Distances")
    plt.title("Distance Distributions")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.savefig(f"{path}/health-checks/Epoch-{epoch}-metric_healthcheck.png")
    # plt.close()

    # Triangle inequality stats
    logging.info(f"Triangle Inequality Violations: {triangle_violations} / {total_triplets}")
    logging.info(f"Percentage of Violations: {100 * triangle_violations / max(1, total_triplets):.2f}%")

    # Print summary statistics
    logging.info(
        f"All Distances: Min={min(all_distances):.4f}, Max={max(all_distances):.4f}, Mean={torch.tensor(all_distances).mean():.4f}")
    logging.info(
        f"Within-Class Distances: Min={min(within_class_distances):.4f}, Max={max(within_class_distances):.4f}, Mean={torch.tensor(within_class_distances).mean():.4f}")
    logging.info(
        f"Between-Class Distances: Min={min(between_class_distances):.4f}, Max={max(between_class_distances):.4f}, Mean={torch.tensor(between_class_distances).mean():.4f}")

    # Optional: Check for NaNs or infinities
    if torch.isnan(distances).any():
        logging.warning("Detected NaN values in distances!")
    if torch.isinf(distances).any():
        logging.warning("Detected infinite values in distances!")


def train(epoch, model: AssembledModel, optimizer, train_loader, device, scheduler=None):
    model.train()
    loss_all = 0
    inters_num = 0
    losses = []
    for batch_idx, data in enumerate(train_loader):
        # xs, adjs, ys = data.x, data.adj, data.y
        data.to(device)

        data = data.to(device)
        optimizer.zero_grad()

        distances, l, e = model(data.x, data.adj, data.mask)
        labels = (data.y.view(-1, 1) == data.y.view(1, -1)).float()
        loss = contrastive_loss(distances, l, e, labels)

        # loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()

        # loss_all += data.y.size(0) * float(loss)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss)
        loss_all += loss.item()

        inters_num += 1

        with torch.no_grad():
            logging.info(f"Epoch {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}")
            if inters_num % 10 == 0:
                logging.info(f"====> Loss Mean: {np.mean(losses)}, Loss Var {np.var(losses)}")

    if scheduler is not None:
        # Step the scheduler after the epoch
        scheduler.step()

    logging.info(f"Epoch:{epoch}, processed {inters_num} batches. epoch train done")

    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def build_knn_searcher(train_loader, model: AssembledModel, device, brute_force=False) -> KNNSearcher:
    """

    :param train_loader
    :param model
    :param device
    :param brute_force: bool
        determine which searcher to use
    :return: object
    :rtype: KNNSearcher
    """
    model.eval()
    points = []
    i = 0
    for batch in train_loader:
        batch = batch.to(device)
        x, adj, _, _ = model.pool_batch(batch.x, batch.adj, batch.mask)  # Pool graph embeddings
        if i == 0:
            print(f"labels shape {batch.y.shape}")

        labels = batch.y.squeeze()
        if i == 0:
            print(f"labels shape after squeeze: {labels}")
            i += 1
        points.extend(list(zip(x, adj, labels)))  # Combine features and adj as graph tuple
        # labels.append(batch.y.squeeze())#.cpu().numpy())  # Convert labels to numpy


    if brute_force: return BruteForceKNN(points, model.compute_distances)

    logging.info("Building VP Tree...")
    vp_tree = VPTree(points=points, dist_fn=model.compute_distances)

    return vp_tree


@torch.no_grad()
def test(loader, vp_tree: VPTree, model, device, k=10):
    """
    Evaluate the model using VP-Tree for k-NN search.

    Args:
    - loader: DataLoader for the test/validation dataset.
    - vp_tree: Pre-built VP-Tree for training embeddings.
    - model: CombinedModel with pooling capabilities.
    - device: Device for computation.
    - k: Number of nearest neighbors to consider.

    Returns:
    - accuracy: Classification accuracy based on k-NN.
    """
    model.eval()
    correct = 0
    results = torch.zeros((2, len(loader.dataset)), dtype=loader.dataset[0].y.dtype)
    for bidx, batch in enumerate(loader):
        batch = batch.to(device)
        x, adj, l, e = model.pool_batch(batch.x, batch.adj, batch.mask)  # Get embeddings for the query

        for i in range(len(x)):
            knn_results = vp_tree.get_n_nearest_neighbors((x[i], adj[i]), k)
            # Extract labels from nearest neighbors
            neighbor_labels = [point[2] for point, _ in knn_results]  # point[2] is the label
            # Majority vote from k-NN
            # Perform majority voting
            from collections import Counter
            predicted_label = Counter(neighbor_labels).most_common(1)[0][0]
            if predicted_label == batch.y[i].item():
                correct += 1

            results[0, i] = predicted_label
            results[1, i] = batch.y[i].item()

        logging.info(f"Validated batch {bidx}")

    accuracy = correct / len(loader.dataset)
    return accuracy, results
