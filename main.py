import torch

from classic_methods_experiments import run_knn_experiment, run_knn_experiment_multithread, run_svm_experiment
from datasets import load_dataset
from fewshot_experiments import run_fewshot_without_training

def run():

    # torch.manual_seed(42)  # For reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset('TRIANGLES')
    
    run_knn_experiment(dataset)


if __name__ == '__main__':
    run()
