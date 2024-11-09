# [Creating Graph Distances Based on Fractional Isomorphism]

**notice**: this repo is for our **ongoing research**, we are open to collaborations and ideas but please contact us before using the ideas in the code or cite this repo.

**this README incomplete**

[![DOI](https://zenodo.org/badge/DOI/YOUR-DOI-HERE.svg)](https://doi.org/YOUR-DOI-HERE)  
[Optional: Add status badges for build, tests, etc.]

## 📄 Abstract

In the light if GNNs being bounded by WL-test we tried to find other ways of measuring graph similarities without embedding them into Euclidean spaces and losing information. We created some distance functions as one way of measuring similarity, and proved many properties a distance function could have theoretically. Then we experimented on some classical Machine Learning tasks that easily incorporate a distance function and saw how they perform compared to when using other distances and kernels. For Better ML task results we encourage incorporating these distances in more sophisticated ML and DL methods.

## 📚 Paper

The full paper is available at:

- [Link to your paper (arXiv, journal, etc.)](#)
- Citation:

```bibtex
@article{your_paper_reference,
  author = {Your Name and Co-author},
  title = {Your Paper Title},
  journal = {Journal Name},
  year = {20XX},
  volume = {X},
  number = {Y},
  pages = {Z-ZZ},
  doi = {XX.XXXX/XXXXXX},
}
```

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### Requirements

The following packages are required:

- Python >= 3.8
- [Additional libraries such as TensorFlow, PyTorch, NumPy, CVXPY, etc.]

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install <library1> <library2> ...
```

## 🚀 Usage

### Running the Experiments

To reproduce the results presented in the paper, run the following commands:

```bash
# Example command to run the main script
python main.py --config configs/config_file.yaml
```

### Example Notebooks

We also provide Jupyter notebooks that showcase how to use the code for different experiments:

- [notebooks/Example_1.ipynb](notebooks/Example_1.ipynb)
- [notebooks/Example_2.ipynb](notebooks/Example_2.ipynb)

### Dataset

[Provide details about any datasets used in the paper.]

- **Dataset name**: [Link to dataset or instructions to download]
- **Preprocessing**: [Details on any preprocessing steps]

## 🧪 Results

[Include a summary of the main results from your paper, potentially with figures or tables.]

```bash
# Example command to run evaluation
python evaluate.py --model model_name --dataset dataset_name
```

You can also find the full results in the `results/` folder.

### Key Metrics

- Accuracy: X%
- Precision: Y%
- Recall: Z%

## 🗂 Repository Structure

```
your-repo-name/
├── data/               # Dataset files
├── docs/               # Documentation files
├── notebooks/          # Jupyter notebooks
├── src/                # Source code for the project
│   └── ...             # Other source code files
├── results/            # Results from the experiments
├── tests/              # Unit tests
├── LICENSE             # License file
├── README.md           # Readme file
└── requirements.txt    # Dependencies file
```

## 💡 Key Features

- **Feature 1**: [Explain]
- **Feature 2**: [Explain]
- **Feature 3**: [Explain]

## 💻 Contribution

We welcome contributions! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

## 📜 License

This repository is licensed under the [MIT License](LICENSE).

## 📧 Contact

For questions or further information, please contact:

- [Your Name](mailto:your.email@example.com)
- [Co-author Name](mailto:coauthor.email@example.com)

## Acknowledgements

[Optional: Acknowledge any funding, institutions, or individuals that contributed to the research.]

---

### Notes:

- **Abstract**: A quick overview of the paper's goals, methods, and results.
- **Installation**: Clear and concise steps to set up the environment.
- **Usage**: Instructions to run experiments, including examples and key commands.
- **Results**: Briefly highlight the key outcomes of the research.
- **Repository Structure**: Helps others navigate your codebase.
- **Contribution**: Guidelines for those who want to contribute to your project.

You can add or remove sections depending on your specific needs and audience.
