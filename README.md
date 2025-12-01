# Sow Productivity Analysis Pipeline

A machine learning pipeline for analyzing and predicting sow productivity using reproductive performance data. This project supports both CDPQ and Hypor datasets and includes tools for data synthesis, preprocessing, model training, and evaluation.

Publication: [link](https://link)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [1. Generate Synthetic Dataset](#1-generate-synthetic-dataset)
  - [2. Run Analysis Pipeline](#2-run-analysis-pipeline)
- [Command-Line Arguments](#command-line-arguments)
- [Output](#output)
- [Models](#models)
- [Dependencies](#dependencies)

## Overview

This project provides a complete machine learning pipeline for predicting sow productivity based on reproductive performance metrics. The pipeline includes:

- **Data Synthesis**: Generate synthetic datasets mimicking real CDPQ or Hypor data structures
- **Preprocessing**: Data cleaning, feature engineering, and normalization
- **Model Training**: Multiple classification algorithms with hyperparameter optimization
- **Ensemble Methods**: Voting and stacking classifiers
- **Evaluation**: Cross-validation, confusion matrices, feature importance, and performance metrics

## Project Structure

```
productivity/
├── code/
│   ├── synthesize_dataset.py    # Generate synthetic datasets
│   ├── main.py                  # Main analysis pipeline
│   ├── model.py                 # Model wrapper and training logic
│   ├── utils.py                 # Utility functions
│   ├── graphing.py              # Visualization functions
│   └── main.R                   # R analysis script
├── raw_data/                    # Raw dataset storage
├── processed_data/              # Processed datasets
├── figures/                     # Generated plots and visualizations
├── outputs/                     # Model evaluation results (CSV files)
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

If you don't have existing datasets, start by generating synthetic data:

```bash
# Generate CDPQ synthetic dataset
python3 code/synthesize_dataset.py cdpq --n_sows 1000 --max_parities 7 --random_state 42

# Generate Hypor synthetic dataset
python3 code/synthesize_dataset.py hypor --n_sows 5000 --max_parities 10 --n_farms 5 --random_state 42
```

Then run the analysis pipeline:

```bash
# Run analysis on CDPQ dataset
python3 code/main.py cdpq

# Run analysis on Hypor dataset
python3 code/main.py hypor
```

## Usage

### 1. Generate Synthetic Dataset

Use `synthesize_dataset.py` to create dummy datasets for testing or when real data is unavailable.

#### Basic Usage

```bash
python3 code/synthesize_dataset.py <dataset_type>
```

#### Examples

```bash
# Generate CDPQ dataset with default parameters (100 sows, max 7 parities)
python3 code/synthesize_dataset.py cdpq

# Generate CDPQ dataset with custom parameters
python3 code/synthesize_dataset.py cdpq --n_sows 1000 --max_parities 7 --random_state 42

# Generate Hypor dataset with 5000 sows across 5 farms
python3 code/synthesize_dataset.py hypor --n_sows 5000 --max_parities 10 --n_farms 5 --random_state 42

# Generate small test dataset for quick testing
python3 code/synthesize_dataset.py cdpq --n_sows 50 --max_parities 5 --random_state 123
```

#### Arguments for `synthesize_dataset.py`

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `dataset_type` | str | Yes | - | Dataset type: `cdpq` or `hypor` |
| `--n_sows` | int | No | 100 | Number of unique sows to generate |
| `--max_parities` | int | No | 7 | Maximum consecutive parities per sow |
| `--n_farms` | int | No | 5 | Number of farms (Hypor only) |
| `--random_state` | int | No | None | Random seed for reproducibility |

#### Output

Generated datasets are saved to:
- `raw_data/cdpq_raw_dataset.xlsx`
- `raw_data/hypor_raw_dataset.xlsx`

### 2. Run Analysis Pipeline

Use `main.py` to run the complete machine learning analysis pipeline.

#### Basic Usage

```bash
python3 code/main.py <dataset_name>
```

#### Examples

```bash
# Run analysis on CDPQ dataset with default settings
python3 code/main.py cdpq

# Run analysis on CDPQ subset (fewer features)
python3 code/main.py cdpq --subset

# Run analysis on Hypor dataset with custom parameters
python3 code/main.py hypor --seed 123 --n_jobs 4 --max_iter 200000

# Run with multiple parallel jobs for faster training
python3 code/main.py cdpq --n_jobs 4

# Use different random seed
python3 code/main.py hypor --seed 999
```

#### Arguments for `main.py`

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `dataset_name` | str | Yes | - | Dataset to use: `cdpq` or `hypor` |
| `--subset` | flag | No | False | Use CDPQ subset (drops some columns) |
| `--seed` | int | No | 42 | Random seed for reproducibility |
| `--n_jobs` | int | No | 1 | Number of parallel jobs for training |
| `--max_iter` | int | No | 100000 | Maximum iterations for iterative algorithms |

#### Help

View all available options:

```bash
python3 code/synthesize_dataset.py -h
python3 code/main.py -h
```

## Output

The pipeline generates several outputs:

### Processed Data
- `processed_data/<dataset>_processed_dataset.csv` - Cleaned and feature-engineered data

### Figures
Generated in `figures/<dataset>/`:
- Distribution plots (KDE plots for each feature)
- Learning curves for each model
- Confusion matrices (combined plot)
- Feature importance plots (combined plot)

### Evaluation Results
Generated in `outputs/<dataset>/`:
- `normality_test_results.csv` - Shapiro-Wilk test results
- `model_evaluation_scores.csv` - Repeated hold-out evaluation metrics
- `model_cv_scores.csv` - Cross-validation scores

## Models

The pipeline trains and evaluates the following classifiers:

1. **Decision Tree (DT)** - Simple tree-based classifier
2. **K-Nearest Neighbors (KNN)** - Instance-based learning
3. **Logistic Regression (LR)** - Linear probabilistic classifier
4. **Multi-Layer Perceptron (MLP)** - Neural network
5. **Random Forest (RF)** - Ensemble of decision trees
6. **Stochastic Gradient Descent (SGD)** - Linear classifier with SGD training
7. **Support Vector Machine (SVM)** - Kernel-based classifier
8. **Voting Classifier** - Ensemble using majority voting
9. **Stacking Classifier** - Ensemble using meta-learning

### Evaluation Metrics

- **Balanced Accuracy** - Accounts for class imbalance
- **Precision (weighted)** - Weighted average precision
- **Recall (weighted)** - Weighted average recall
- **F1-Score (weighted)** - Harmonic mean of precision and recall
- **F1-Score (per class)** - Individual F1 scores for Low/Medium/High classes

### Cross-Validation Strategy

- **CDPQ**: Random stratified K-fold (due to single farm)
- **Hypor**: Leave-One-Group-Out (LOGO) by farm

## Dependencies

### Python Packages

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `openpyxl` - Excel file reading/writing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scipy` - Scientific computing (for statistical tests)

### Optional

- `joblib` - Parallel processing (included with scikit-learn)

## Workflow Example

Complete workflow from data generation to analysis:

```bash
# Step 1: Generate synthetic CDPQ dataset
python3 code/synthesize_dataset.py cdpq --n_sows 1000 --max_parities 7 --random_state 42

# Step 2: Run analysis pipeline
python3 code/main.py cdpq --seed 42 --n_jobs 4

# Step 3: Check outputs
ls outputs/cdpq/
ls figures/cdpq/
```

For Hypor dataset:

```bash
# Step 1: Generate synthetic Hypor dataset
python3 code/synthesize_dataset.py hypor --n_sows 5000 --max_parities 10 --n_farms 5 --random_state 42

# Step 2: Run analysis pipeline
python3 code/main.py hypor --seed 42 --n_jobs 4

# Step 3: Check outputs
ls outputs/hypor/
ls figures/hypor/
```

## Notes

- **Memory Usage**: If you encounter memory issues with large datasets, reduce `--n_jobs` to 1 or 2
- **Reproducibility**: Always use `--random_state` when generating synthetic data and `--seed` when running analysis for reproducible results
- **CDPQ Subset**: The `--subset` flag removes body weight and backfat measurements at breeding and farrowing, keeping only weaning measurements
- **Processing Time**: Full pipeline execution can take several minutes to hours depending on dataset size and number of parallel jobs

## Troubleshooting

### File Not Found Error
If you see "Data file not found", ensure you've generated the dataset first using `synthesize_dataset.py`.

### Memory Errors
Reduce the number of parallel jobs:
```bash
python3 code/main.py cdpq --n_jobs 1
```

### Convergence Warnings
Increase maximum iterations:
```bash
python3 code/main.py cdpq --max_iter 200000
```

## Corresponding Authors

Ji Yang, jyang49@uoguelph.ca
Dan Tulpan, dtulpan@uoguelph.ca

Department of Animal Biosciences, 
Centre for Genetic Improvement of Livestock, 
University of Guelph, Guelph, ON, N1G 2W1, Canada