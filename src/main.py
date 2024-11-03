import argparse


from data import load_dataset
from distance_functions import DistanceFuntion
from knn import evaluate_and_save_reports, run_knn
from utils import get_unique_filename



def run(dataset_name, train_num, val_num):

    dataset = load_dataset(dataset_name)

    k_values = [1, 3, 7, 15]
    distance_funcs = [DistanceFuntion('subgraph_isomorphism_distance', False, lam=2),
                       DistanceFuntion('subgraph_isomorphism_distance', True, lam=2),
                         DistanceFuntion('graph_convex_isomorphism', lam=2),
                             DistanceFuntion('wasserstein_spectral_distance')]

    # Run k-NN and get results
    knn_results, true_labels = run_knn(dataset, k_values, distance_funcs, train_num, val_num, n_jobs=4)
    
    # Evaluate and save classification reports
    evaluate_and_save_reports(true_labels, knn_results, get_unique_filename(f"results/knn_{dataset_name}"))


if __name__ == '__main__':

    a = [1.0, 2.0, 3.0, 4.0]
    print([int(x) for x in a])

    parser = argparse.ArgumentParser(
        description="Script to run the experiments"
    )
    parser.add_argument("--dataset", required=True, type=str, default='TRAINGLES')
    parser.add_argument("--train_num", required=True, type=int, default=300)
    parser.add_argument("--val_num", required=True, type=int, default=100)
    args = parser.parse_args()

    dataset = args.dataset
    train_num = args.train_num
    val_num = args.val_num
    run(dataset, train_num, val_num)


