import argparse


from data import load_dataset
from distance_functions import DistanceFuntion
from knn import evaluate_and_save_reports, run_knn
from utils import get_unique_filename



def run(dataset_name, train_num, val_num, workers_num, each_class_train):

    dataset = load_dataset(dataset_name)

    k_values = [1, 3, 7, 15]
    distance_funcs = [DistanceFuntion('subgraph_isomorphism_distance', False, lam=2),
                       DistanceFuntion('subgraph_isomorphism_distance', True, lam=2),
                         DistanceFuntion('graph_convex_isomorphism', lam=2),
                             DistanceFuntion('wasserstein_spectral_distance')]

    # Run k-NN and get results
    knn_results, true_labels = run_knn(dataset, k_values, distance_funcs, train_num, val_num, n_jobs=workers_num, each_class_train=each_class_train)
    
    # Evaluate and save classification reports
    evaluate_and_save_reports(true_labels, knn_results, get_unique_filename(f"results/knn_{dataset_name}"))


if __name__ == '__main__':
    
    # func = DistanceFuntion('subgraph_isomorphism_distance', False, lam=2)
    # dataset = load_dataset("TRIANGLES",verbose=False)

    # print(func(dataset[0], dataset[1000]))
    # print(dataset[0].y)
    # print(dataset[1000].y)
    parser = argparse.ArgumentParser(
        description="experiment settings"
    )
    parser.add_argument("--dataset", required=True, type=str, default='TRIANGLES')
    parser.add_argument("--train_num", required=True, type=int, default=300)
    parser.add_argument("--val_num", required=True, type=int, default=100)
    parser.add_argument("--workers_num", required=True, type=int, default=4)
    parser.add_argument("--each_class_train", required=True, type=int, default=10)
    
    args = parser.parse_args()

    dataset = args.dataset
    train_num = args.train_num
    val_num = args.val_num
    workers_num = args.workers_num
    each_class_train = args.each_class_train
    run(dataset, train_num, val_num, workers_num, each_class_train)


