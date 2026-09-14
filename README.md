# Ames House Price Prediction with PyTorch

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A compact deep-learning baseline for Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

The project combines feature engineering, train/test feature alignment, five-fold cross-validation, and a PyTorch multilayer perceptron. The submitted model achieved a public Log-RMSE of **0.16117** (reported Top 20%).

## Highlights

- MLP architecture: `256 → 128 → 1`, with ReLU and dropout
- Five-fold validation with a mean Log-RMSE of `0.1442`
- Consistent log-target evaluation between local validation and Kaggle
- Reproducible preprocessing for numerical and categorical features

## Repository layout

```text
.
├── House-Price-Prediction/
│   ├── House-Price-Prediction.ipynb
│   │                    # Preprocessing, training, validation, and inference
│   ├── requirements.txt
│   ├── data/             # Competition data snapshot
│   └── results/          # Submission output
└── LICENSE
```

For the experiment details, hyperparameters, and reproduction instructions, see the [full project README](House-Price-Prediction/README.md).

## Scope

This is a learning project and a transparent baseline rather than a state-of-the-art tabular modeling claim. Its main value is the complete PyTorch training and debugging workflow.
