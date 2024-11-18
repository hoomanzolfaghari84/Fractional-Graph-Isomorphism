from concurrent.futures import ProcessPoolExecutor
import logging
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

from data import get_datasubsets

logger = logging.getLogger(__name__)

root_exp_dir = '/root'

# Ensure compute_knn_for_one is defined at the top level for pickling compatibility
def compute_knn_for_one(data_tuple, train_dataset, dist_func):
    try:
        data_idx, data = data_tuple
        distances = []

        for train_idx, train_data in enumerate(train_dataset):
            dist = dist_func(data, train_data)
            distances.append((train_idx, dist))
        
        # Sort by distance and get the k nearest neighbors
        distances.sort(key=lambda x: x[1])
        print(f"Distances computed for Val data idx: {data_idx} using:{dist_func.get_name()}")
        
        return distances
    except Exception as e :print(f"Exception for Val data idx: {data_idx} using: {dist_func.get_name()}. Exception: {e}")
    
    return [np.inf] * len(train_dataset)

def run_knn(dataset, k_values, distance_funcs, num_train, num_val, num_test=0, n_jobs=1, each_class_train=None, each_class_val=None):
    train_dataset, val_dataset, _ = get_datasubsets(dataset, num_train, num_val, each_class_train=each_class_train, each_class_val=each_class_val)

    labels = np.array([graph.y.item() for graph in train_dataset])  # Assuming labels are in graph.y
    true_labels = np.array([graph.y.item() for graph in val_dataset])

    results = {}

    for dist_func in distance_funcs:
        dist_func_name = dist_func.get_name()
        results[dist_func_name] = {}

        # Use ProcessPoolExecutor for parallel processing with multiple processes
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Passing train_dataset and dist_func as additional arguments to avoid closure issues
            knn_results = list(executor.map(compute_knn_for_one, enumerate(val_dataset), [train_dataset]*len(val_dataset), [dist_func]*len(val_dataset)))

        for k in k_values:
            y_pred = []
            for distances in knn_results:
                neighbors = distances[:k]
                neighbor_labels = [labels[idx] for idx, _ in neighbors]
                neighbor_labels = [int(x) for x in neighbor_labels]
                most_common_label = np.bincount(neighbor_labels).argmax()
                y_pred.append(most_common_label)
            
            results[dist_func_name][f'k={k}'] = y_pred

    return results, true_labels

def evaluate_and_save_reports(true_labels, results, filename):
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
                'accuracy score': accuracy_score(true_labels, predictions)
            })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(filename, index=False)
