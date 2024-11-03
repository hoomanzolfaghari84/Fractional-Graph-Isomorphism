from concurrent.futures import ThreadPoolExecutor


from data import get_dataloaders, get_datasubsets
import numpy as np
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report
import pandas as pd
from torch_geometric.utils import to_networkx, to_dense_adj
from sklearn.metrics import accuracy_score, f1_score, classification_report

def run_knn(dataset, k_values, distance_funcs, num_train, num_val, num_test=0, n_jobs=1):
    """
    Perform k-Nearest Neighbors on PyG graphs using different custom distance functions and k values.
    
    Args:
        dataset (torch_geometric.data.Dataset): PyG dataset with graph data.
        k_values (list): List of different k values to test.
        distance_funcs (list): List of custom distance functions to use for k-NN.
        n_jobs (int): Number of threads to use for parallel processing.
    
    Returns:
        dict: A dictionary with results for each distance function and k value.
    """
    
    train_dataset, val_dataset, _ = get_datasubsets(dataset, num_train, num_val)


    # Extract embeddings/features and labels from each graph in the dataset
    labels = np.array([graph.y.item() for graph in train_dataset])  # Assuming labels are in graph.y
    true_labels = np.array([graph.y.item() for graph in val_dataset])

    results = {}

    for dist_func in distance_funcs:
        dist_func_name = dist_func.get_name()
        results[dist_func_name] = {}

        def compute_knn_for_one(data):
            distances = []
            for train_idx, train_data in enumerate(train_dataset):
                dist = dist_func(data, train_data)
                distances.append((train_idx, dist))
            
            # Sort by distance and get the k nearest neighbors
            distances.sort(key=lambda x: x[1])
            return distances

        # Compute k-NN in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            knn_results = list(executor.map(compute_knn_for_one, val_dataset))

        for k in k_values:
            # Aggregate predictions using majority voting
            y_pred = []
            for distances in knn_results:
                # Get the labels of the k nearest neighbors
                neighbors = distances[:k]
                neighbor_labels = [labels[idx] for idx, _ in neighbors]
                # Predict the most common label
                most_common_label = np.bincount(neighbor_labels).argmax()
                y_pred.append(most_common_label)
            
            # Store predictions
            results[dist_func_name][f'k={k}'] = y_pred

    return results, true_labels

def evaluate_and_save_reports(true_labels, results, filename):
    """
    Evaluate k-NN results and save classification reports to a file.
    
    Args:
        dataset (torch_geometric.data.Dataset): PyG dataset with graph data.
        k_values (list): List of different k values tested.
        distance_funcs (list): List of custom distance functions used.
        results (dict): The results from the k-NN function.
        filename (str): The file path to save the classification reports.
    """
    
    report_data = []

    for dist_func_name, k_results in results.items():
        for k, predictions in k_results.items():
            report = classification_report(true_labels, predictions, output_dict=True, zero_division=np.nan)
            report_data.append({
                'Distance Function': dist_func_name,
                'k': k,
                'wa precision': report['weighted avg']['precision'],
                'wa recall': report['weighted avg']['recall'],
                'wa f1-score': report['weighted avg']['f1-score'],
                'wa support': report['weighted avg']['support'],
                'accuracy score' : accuracy_score(true_labels, predictions)
            })

    # Convert to DataFrame and save to CSV
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(filename, index=False)

