
# Embedding-Oriented PGNNIV

This repository contains code and experiments for the paper "Enhancing material behavior discovery using embedding-oriented Physically-Guided Neural Networks with Internal Variables".

## Abstract

Physically Guided Neural Networks with Internal Variables are Scientific Machine Learning tools that use only observable data for training and have an unravelling capacity as added feature. They incorporate physical knowledge both by prescribing the model architecture and using loss regularization, thus endowing certain specific neurons with a physical meaning as internal state variables. Despite their potential, these models face challenges in scalability when applied to high-dimensional data such as fine-grid spatial fields or time-evolving systems.

In this work, we propose some enhancements to the PGNNIV framework that address these scalability limitations through reduced-order modeling techniques. Specifically, we introduce alternatives to the original decoder structure using spectral decomposition (e.g., Fourier basis), Proper Orthogonal Decomposition (POD), and pretrained autoencoder-based mappings. These surrogate decoders offer varying trade-offs between computational efficiency, accuracy, noise tolerance, and generalization, while improving drastically the scalability. Additionally, we integrate model reuse via transfer learning and fine-tuning strategies to exploit previously acquired knowledge, supporting efficient adaptation to novel materials or configurations, and significantly reducing training time while maintaining or improving model performance. To illustrate these various techniques, we use a representative case governed by the nonlinear diffusion equation, using only observable data.

The results demonstrate that the enhanced PGNNIV framework successfully identifies the underlying constitutive state equations while maintaining high predictive accuracy. It also improves robustness to noise, mitigates overfitting, and reduces computational demands. The proposed techniques can be tailored to various scenarios depending on data availability, computational resources, and specific modeling objectives, overcoming scalability challenges in all the scenarios.

## Project Structure


- `utilities/` — Core scientific machine learning utilities, including algebraic operations, kernel and operator definitions, and general-purpose helper functions.
- `architectures/` — Neural network architectures for each embedding strategy.
- `datasets_creation/` — Scripts for generating and processing synthetic datasets. 
- `embeddings/` — Main experiments and visualizers for each embedding method:
  - `autoencoder/`
  - `baseline/`
  - `fourier/`
  - `POD/`

  Each folder contains a `main.ipynb` file for running one experiment isolatedly (for one dataset), and a `main_iterative_[embedding].py` file for running all the experiments for all the datasets for that embedding technique.
- `knowledge_transfer/` — Contains all the knowledge transfer scripts in order to run the comparative between baseline method, fine-tuning and transfer learning.
- `paper/` — Jupyter notebooks for analysis and figures used in the publication.

## Main Features

- Implements PGNNIV with different embedding strategies.
- Supports training, validation, and visualization of results.
- Modular code for easy extension and experimentation.
- Includes scripts for dataset creation.

## Getting Started

1. **Environment Setup**
   - Use the provided `environment.yml` to create a conda environment:
     ```bash
     conda env create -f environment.yml
     conda activate embedding_pgnniv_env
     ```

2. **Data Preparation**
   - Generate or place datasets in al directory called `data/`. Use scripts in `datasets_creation/` if needed.

3. **Training**
   - Run the main scripts in each embedding folder (e.g., `main_iterative_ae.py`, `main_iterative_baseline.py`, etc.) to train models.
   - Run the `knowledge_transfer/` notebooks in order to compare different knowledge transfer techniques.

4. **Visualization**
   - Use the provided Jupyter notebooks in each embedding folder to visualize results and analyze model performance.

<!-- ## Citation

If you use this code or ideas from this project, please cite the corresponding paper. -->

## License

This project is licensed under the MIT License.
