# Gene Expression Classifier for CRISPR Editing Outcomes (or other classes)

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
```

# Input Data

The script expects an **AnnData (.h5ad)** file with:

## Expression matrix
`adata.layers["lognorm"]`

This layer should contain **normalized + log-transformed gene expression values**.

## Metadata
`adata.obs[target_column]`

Default:

`Combined_RepairClass`

Each cell must contain a **single editing outcome class label**.

Example:

| cell_id | Combined_RepairClass |
|--------|----------------------|
| cell1  | HDR_HDR |
| cell2  | HDR_NHEJ |
| cell3  | NHEJ_NHEJ |

---

# Running the Pipeline

Example command:

```bash
python train_classifiers.py \
    --input hHSPC_48hpn_for_ML.h5ad \
    --output-dir results \
    --target-column Combined_RepairClass \
    --layer lognorm \
    --epochs 25 \
    --batch-size 32 \
    --xgb-n-estimators 500 \
    --xgb-learning-rate 0.03 \
    --xgb-max-depth 4 \
    --xgb-subsample 0.9 \
    --xgb-colsample-bytree 0.5 \
    --xgb-early-stopping-rounds 50
```


