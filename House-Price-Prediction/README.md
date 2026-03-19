# Ames House Price Prediction (Deep Learning Approach)

!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
!\[Kaggle](https://img.shields.io/badge/Kaggle-Top20%25-00aff0.svg)

This repository provides a robust deep learning pipeline for the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). By implementing a Multi-Layer Perceptron (MLP) with optimized data pre-processing, this project achieved a **Top 20%** standing.

## 📈 Project Performance

* **Kaggle Public Score (Log-RMSE)**: `0.16117`
* **5-Fold Cross-Validation Mean Error**: `0.1442`
* **Status**: Successfully converged within 300 epochs.

## 🛠️ Key Engineering Highlights

As a Software Engineering student, I focused not only on model accuracy but also on **robustness and debugging logic**:

1. **Refactored K-Fold Logic**: Fixed a critical memory assignment bug where `torch.cat` results were not re-assigned, ensuring the model trains on the full 80% training slice per fold.
2. **Scale Alignment**: Rectified a "Log of Log" evaluation error, ensuring local validation metrics strictly align with Kaggle's Log-RMSE standards.
3. **Feature Pipeline**:

   * Engineered physical features (`TotalSF`, `HouseAge`) before normalization.
   * Strict feature alignment: Concatenated train/test sets before `get\\\_dummies` to prevent column mismatch.
   * Boolean-to-Float casting to ensure PyTorch tensor compatibility.

## 🏗️ Project Structure

```text
.
├── main.ipynb          # Core logic (Preprocessing, Training, Prediction)
├── submission.csv      # Final Kaggle-ready output
├── requirements.txt    # Environment dependencies
└── README.md           # Documentation

```
## 🚀 Getting Started

### 1. Installation

Ensure Python 3.8+ is installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Usage

Open `main.ipynb` in Jupyter Lab/Notebook and run all cells. The script will:
* **Data Management**: Download or load the dataset.
* **Feature Engineering**: Perform automated feature engineering and normalization.
* **Validation**: Execute 5-Fold Cross-Validation.
* **Deployment**: Train the final model and export `submission.csv`.

---

### ⚙️ Hyperparameters

* **Optimizer**: Adam (`lr=0.01`, `weight_decay=0.15`)
* **Scheduler**: StepLR (`gamma=0.5`, `step_size=50`)
* **Architecture**: 256 → 128 → 1 (with ReLU & 0.2 Dropout)
