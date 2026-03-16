# Gene Expression Classifier for CRISPR Editing Outcomes

This repository contains a Python pipeline for training and comparing multiple machine-learning models that predict **CRISPR editing outcome classes from single-cell gene expression profiles** stored in an **AnnData (.h5ad)** object.

The pipeline uses normalized log-transformed expression values (`adata.layers["lognorm"]`) and compares three model types:

- Feedforward neural network (TensorFlow / Keras)
- Multinomial logistic regression (interpretable linear model)
- XGBoost multi-class classifier (tree-based ensemble)

The goal is to evaluate predictive performance and extract interpretable relationships between **gene expression patterns and editing outcome classes**.

---

# Overview

Given an AnnData object containing:

- gene expression matrix
- cell metadata with editing outcome labels

the pipeline will:

1. Load the dataset  
2. Filter unwanted outcome classes  
3. Split the dataset into training and test sets  
4. Train three classification models  
5. Evaluate model performance  
6. Export predictions and interpretability outputs  

The models are trained using gene expression features directly from:

```python
adata.layers["lognorm"]
```
No additional scaling is applied.

## Requirements

Python ≥ 3.9

### Core dependencies

- anndata
- numpy
- pandas
- scikit-learn
- tensorflow
- xgboost
- matplotlib
- scipy

### Install via pip

```bash
pip install anndata numpy pandas scikit-learn tensorflow xgboost matplotlib scipy
