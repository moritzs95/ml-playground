# Gene Expression Classifier for CRISPR Editing Outcomes

This repository contains a Python pipeline for training and comparing multiple machine learning models that predict **CRISPR editing outcome classes from single-cell gene expression profiles** stored in an **AnnData (`.h5ad`)** object.

The pipeline uses normalized, log-transformed expression values from `adata.layers["lognorm"]` and compares three model types:

- Feedforward neural network (TensorFlow/Keras)
- Multinomial logistic regression (interpretable linear model)
- XGBoost multi-class classifier (tree-based ensemble)

The goal is to evaluate predictive performance and extract interpretable relationships between **gene expression patterns and editing outcome classes**.

## Overview

Given an AnnData object containing:

- a gene expression matrix
- cell metadata with editing outcome labels

the pipeline will:

1. Load the dataset.
2. Filter unwanted outcome classes.
3. Split the dataset into training and test sets.
4. Train three classification models.
5. Evaluate model performance.
6. Export predictions and interpretability outputs.

The models are trained directly on gene expression features from:

```python
adata.layers["lognorm"]
```

No additional scaling is applied.

## Requirements

- Python >=3.10,<3.11

All Python dependencies are listed in [`requirements.txt`](./requirements.txt).

### Install

```bash
pip install -r requirements.txt

## Input Data

I added a (downsampled, cleaned) example .h5ad file. In general, the script expects an **AnnData (`.h5ad`)** file with the following:

### Expression Matrix

`adata.layers["lognorm"]`

This layer should contain **normalized, log-transformed gene expression values**.

### Metadata

`adata.obs[target_column]`

Default target column:

`Combined_RepairClass`

Each cell must contain a **single editing outcome class label**.

Example:

| `cell_id` | `Combined_RepairClass` |
| --- | --- |
| `cell1` | `HDR_HDR` |
| `cell2` | `HDR_NHEJ` |
| `cell3` | `NHEJ_NHEJ` |

## Running the Pipeline

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

## Main Parameters

### Dataset Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--input` | Input AnnData file | required |
| `--output-dir` | Directory for results | `outputs` |
| `--target-column` | Label column in `adata.obs` | `Combined_RepairClass` |
| `--layer` | Feature matrix layer | `lognorm` |
| `--test-size` | Fraction of cells for the test set | `0.3` |
| `--min-class-size` | Minimum number of cells per class | `50` |

### Label Filtering

| Parameter | Description |
| --- | --- |
| `--exclude-label` | Labels to remove (can be repeated) |
| `--exclude-comma-labels` | Remove multi-label classes |

Example:

```bash
--exclude-label WT_WT
```

### Neural Network Parameters

| Parameter | Description |
| --- | --- |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--hidden-dim-1` | First hidden layer size |
| `--hidden-dim-2` | Second hidden layer size |
| `--dropout` | Dropout rate |

Architecture:

`Input -> Dense -> Dropout -> Dense -> Dropout -> Softmax`

Loss:

`sparse_categorical_crossentropy`

### XGBoost Parameters

| Parameter | Description |
| --- | --- |
| `--xgb-n-estimators` | Number of boosting rounds |
| `--xgb-learning-rate` | Learning rate |
| `--xgb-max-depth` | Maximum tree depth |
| `--xgb-subsample` | Row sampling fraction |
| `--xgb-colsample-bytree` | Feature sampling fraction |
| `--xgb-early-stopping-rounds` | Early stopping rounds |

## Outputs

All results are written to the output directory.

### Model Performance

`model_comparison.csv`

Example:

| model | accuracy |
| --- | --- |
| XGBoost | 0.83 |
| Neural Network | 0.80 |
| Logistic Regression | 0.74 |

### Classification Reports

- `nn_classification_report.txt`
- `logreg_classification_report.txt`
- `xgb_classification_report.txt`

### Confusion Matrices

- `nn_confusion_matrix.png`
- `logreg_confusion_matrix.png`
- `xgb_confusion_matrix.png`

### Predictions

Per-cell predictions on the held-out test set:

- `nn_predictions.csv`
- `logreg_predictions.csv`
- `xgb_predictions.csv`

Each file contains:

| Column | Description |
| --- | --- |
| `obs_name` | Cell identifier |
| `true_label` | True editing class |
| `predicted_label` | Predicted class |
| `prob_*` | Probability for each class |

### Logistic Regression Interpretability

- `logreg_coefficients.csv`
- `logreg_coefficients_long.csv`
- `logreg_intercepts.csv`

These outputs help identify genes associated with specific editing outcomes.

### XGBoost Feature Importance

`xgb_feature_importance.csv`

Contains global importance scores for each gene.

### Neural Network Training History

`nn_training_history.csv`

Tracks:

- training loss
- validation loss
- training accuracy
- validation accuracy

### Metadata

- `run_metadata.json`
- `class_counts.csv`

These files contain dataset statistics and run parameters for reproducibility.

## Interpretation

The three models serve complementary purposes:

| Model | Strength |
| --- | --- |
| Neural network | Captures complex nonlinear relationships |
| Logistic regression | Interpretable gene-outcome associations |
| XGBoost | Strong predictive performance |

Typical workflow:

- Use XGBoost or the neural network for predictive performance.
- Use logistic regression coefficients to identify biologically interpretable gene signatures.

## Example Workflow

```text
AnnData (.h5ad)
        |
        v
Filter outcome classes
        |
        v
Train/test split
        |
        v
Train models
  - Neural Network
  - Logistic Regression
  - XGBoost
        |
        v
Model comparison + interpretation
```

## Potential Improvements for the FNN

Several changes could improve feedforward neural network performance (which is quite poor for this dataset):

- **Class balancing:** Use class weights, oversampling, or undersampling to reduce bias toward the most frequent editing outcome classes.
- **Count scaling:** Test alternative input scaling strategies, such as per-gene standardization or other count normalization approaches, to improve optimization behavior.
- **Simplified labeling strategy:** Reduce the number of target classes by merging rare or biologically similar labels, which may make the classification problem more robust.
- **Highly variable genes only:** Restrict the input feature space to highly variable genes to reduce noise and focus the model on the most informative expression patterns.

These changes can be evaluated individually or in combination to determine which preprocessing and labeling strategy gives the best predictive performance.
