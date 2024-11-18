import argparse
import logging

import torch

from data import load_dataset
from distance_functions import DistanceFuntion
from knn import evaluate_and_save_reports, run_knn
from utils import get_unique_filename

logger = logging.getLogger(__name__)


def run(args):
    
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else: device = args.device
    
    logging.basicConfig(filename=get_unique_filename(f"logs/log_{args.dataset_name}", extension='.log'), level=logging.INFO)
    logger.info('Started')

    dataset = load_dataset(args.dataset_name)

    k_values = [1]
    distance_funcs = [DistanceFuntion('subgraph_isomorphism_distance', fractional=False, lam=2),
                       DistanceFuntion('subgraph_isomorphism_distance', fractional=True, lam=2),
                         DistanceFuntion('graph_convex_isomorphism', lam=2),
                             DistanceFuntion('wasserstein_spectral_distance')]

    # Run k-NN and get results
    knn_results, true_labels = run_knn(dataset, k_values, distance_funcs, args.train_num, args.val_num, n_jobs=args.workers_num, each_class_train=args.each_class_train, each_class_val=args.each_class_val)
    
    # Evaluate and save classification reports
    evaluate_and_save_reports(true_labels, knn_results, get_unique_filename(f"results/knn_{args.dataset_name}"))

    logger.info("finished")

if __name__ == '__main__':
    
    # func = DistanceFuntion('subgraph_isomorphism_distance', False, lam=2)
    # dataset = load_dataset("TRIANGLES",verbose=False)

    # print(func(dataset[0], dataset[1000]))
    # print(dataset[0].y)
    # print(dataset[1000].y)
    parser = argparse.ArgumentParser(
        description="This program runs our graph experiments"
    )
    parser.add_argument("--dataset_name", required=True, type=str, default='TRIANGLES')
    parser.add_argument("--train_num", required=True, type=int, default=300)
    parser.add_argument("--val_num", required=True, type=int, default=100)
    parser.add_argument("--workers_num", required=False, type=int, default=4)
    parser.add_argument("--each_class_train", required=False, type=int, default=10)
    parser.add_argument("--each_class_val", required=False, type=int, default=10)
    parser.add_argument("--device", required=False, type=str, default=None)
    parser.add_argument("--exp_name", required=True, type=str, default='knn')
    # options: knn, pooling
    
    args = parser.parse_args()


    run(args)


