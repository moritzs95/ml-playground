#!/usr/bin/env python3
"""
Train and compare multi-class classifiers on AnnData input using normalized and
log1p transformed counts (adata.layers["lognorm"]).

Final goal
----------
Compare the performance of:
- a feedforward neural network,
- a multinomial logistic regression model,
- an XGBoost multi-class classifier,

and extract interpretable information about how gene-expression patterns relate
to editing outcome classes (at least from the MLR and XGBoost models).

Models
------
- Feedforward neural network (TensorFlow / Keras)
- Multinomial logistic regression
- XGBoost classifier

Input
-----
- AnnData .h5ad file
- Features taken directly from adata.layers["lognorm"]
- Labels taken from adata.obs[target-column]

Outputs
-------
- Model comparison table
- Confusion matrices
- Classification reports
- Logistic regression coefficients per class
- Neural network training history
- Per-cell predictions
- XGBoost feature importances

Example command 
-------
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
"""

from __future__ import annotations

# -----------------------------
# Standard library imports
# -----------------------------
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

# -----------------------------
# Scientific / ML imports
# -----------------------------
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Improve TensorFlow GPU memory allocation behaviour where supported
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


# ============================================================
# Argument parsing
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments controlling data loading, filtering,
    neural-network architecture, and XGBoost hyperparameters.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and compare multi-class classifiers on AnnData input."
    )

    # Path to the input AnnData object
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .h5ad file.",
    )

    # Directory where all outputs will be written
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for saved outputs.",
    )

    # Column in adata.obs containing the class labels
    parser.add_argument(
        "--target-column",
        type=str,
        default="Combined_RepairClass",
        help="Column in adata.obs containing class labels.",
    )

    # Layer in AnnData used as the feature matrix
    parser.add_argument(
        "--layer",
        type=str,
        default="lognorm",
        help='AnnData layer to use as feature matrix, e.g. "lognorm".',
    )

    # Fraction of cells reserved for the held-out test set
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data reserved for test set.",
    )

    # Neural-network training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs for neural network.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for neural network.",
    )

    # Random seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Labels to exclude explicitly
    # Using action='append' allows repeated usage of --exclude-label
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["WT_WT"],
        help="Label to exclude. Can be passed multiple times.",
    )

    # Optional filtering of labels containing commas
    # These often represent ambiguous or compound outcome classes
    parser.add_argument(
        "--exclude-comma-labels",
        action="store_true",
        help="Exclude rows where the target label contains a comma.",
    )

    # Minimum class size required to retain a class
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=50,
        help="Minimum number of cells required per class.",
    )

    # Neural-network architecture settings
    parser.add_argument(
        "--hidden-dim-1",
        type=int,
        default=128,
        help="First hidden layer size for neural network.",
    )
    parser.add_argument(
        "--hidden-dim-2",
        type=int,
        default=64,
        help="Second hidden layer size for neural network.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for neural network.",
    )

    # XGBoost hyperparameters
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds for XGBoost.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for XGBoost.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=6,
        help="Maximum tree depth for XGBoost.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="Row subsampling fraction for XGBoost.",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsampling fraction per tree for XGBoost.",
    )
    parser.add_argument(
        "--xgb-early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping rounds for XGBoost. Set to 0 to disable.",
    )

    return parser.parse_args()


# ============================================================
# Logging / reproducibility
# ============================================================
def setup_logging() -> None:
    """
    Configure global logging format and log level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    """
    Set random seeds across Python, NumPy, and TensorFlow.

    This improves reproducibility, although some GPU operations may still
    retain non-deterministic behaviour depending on backend and hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def log_hardware_info() -> None:
    """
    Log TensorFlow version and available GPU devices.
    """
    logging.info("TensorFlow version: %s", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logging.info("Detected %d GPU(s).", len(gpus))
        for gpu in gpus:
            logging.info("GPU available: %s", gpu)
    else:
        logging.info("No GPU detected. Running on CPU.")


def gpu_sanity_check() -> None:
    """
    Run a quick TensorFlow matrix multiplication on GPU if available.

    This does not benchmark performance rigorously; it is simply a quick test
    that TensorFlow can place and execute an operation on the GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal([2000, 2000])
            b = tf.random.normal([2000, 2000])

            start = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()  # force evaluation
            elapsed = time.time() - start

        logging.info(
            "GPU sanity check passed: matmul output shape=%s, time=%.3fs",
            c.shape,
            elapsed,
        )
    except Exception as exc:
        logging.warning("GPU sanity check failed: %s", exc)


# ============================================================
# Data loading / preprocessing helpers
# ============================================================
def ensure_dense_array(x: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """
    Convert sparse matrices to dense NumPy arrays.

    Many sklearn / TensorFlow operations are easiest to handle with dense
    arrays in this pipeline.

    Parameters
    ----------
    x
        Input matrix, dense or sparse.

    Returns
    -------
    np.ndarray
        Dense array representation.
    """
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def load_anndata(input_path: Path) -> ad.AnnData:
    """
    Load an AnnData object from disk and perform basic integrity checks.

    Parameters
    ----------
    input_path
        Path to .h5ad file.

    Returns
    -------
    ad.AnnData
        Loaded AnnData object.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    adata = ad.read_h5ad(input_path)

    if adata.n_obs == 0:
        raise ValueError("AnnData object contains zero observations.")
    if adata.n_vars == 0:
        raise ValueError("AnnData object contains zero variables.")

    return adata


# ============================================================
# Dataset preparation
# ============================================================
def prepare_dataset(
    adata: ad.AnnData,
    layer_name: str,
    target_column: str,
    exclude_labels: list[str],
    exclude_comma_labels: bool,
    min_class_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Extract the feature matrix X and target vector y from AnnData, while
    applying label filtering.

    Processing steps
    ----------------
    1. Confirm requested layer exists
    2. Confirm target column exists
    3. Remove missing labels
    4. Optionally remove comma-containing labels
    5. Remove explicitly excluded labels
    6. Remove classes smaller than min_class_size
    7. Return filtered X, y, retained row indices, and feature names

    Returns
    -------
    X : np.ndarray
        Feature matrix from adata.layers[layer_name].
    y : np.ndarray
        String class labels.
    row_indices : np.ndarray
        Row indices retained after filtering.
    feature_names : list[str]
        Names of features / genes from adata.var_names.
    """
    # Make sure the requested feature layer exists
    if layer_name not in adata.layers:
        raise ValueError(
            f'Layer "{layer_name}" not found in AnnData. '
            f"Available layers: {list(adata.layers.keys())}"
        )

    # Make sure the label column exists in adata.obs
    if target_column not in adata.obs.columns:
        raise ValueError(
            f'Target column "{target_column}" not found in adata.obs. '
            f"Available columns include: {list(adata.obs.columns)}"
        )

    obs = adata.obs.copy()
    labels = obs[target_column].astype("string")

    # Start by keeping only cells with non-missing labels
    mask = labels.notna()

    # Optionally discard labels that contain commas
    # These can represent mixed or ambiguous labels
    if exclude_comma_labels:
        mask &= ~labels.str.contains(",", na=False)

    # Exclude any user-specified labels
    if exclude_labels:
        mask &= ~labels.isin(exclude_labels)

    # Count class sizes after first-pass filtering
    labels_filtered = labels[mask].astype(str)
    class_counts = labels_filtered.value_counts()

    # Keep only sufficiently large classes
    valid_classes = class_counts[class_counts >= min_class_size].index.tolist()

    if len(valid_classes) < 2:
        raise ValueError(
            "Fewer than two classes remain after filtering. "
            "Lower --min-class-size or adjust exclusions."
        )

    # Final mask also enforces membership in valid_classes
    mask &= labels.isin(valid_classes)

    # Convert boolean mask to row indices for slicing AnnData consistently
    row_indices = np.where(mask.to_numpy())[0]

    # Extract filtered labels
    y = adata.obs.iloc[row_indices][target_column].astype(str).to_numpy()

    # Extract filtered feature matrix from the requested layer
    X = adata.layers[layer_name][row_indices, :]
    X = ensure_dense_array(X)

    # Gene / feature names
    feature_names = adata.var_names.astype(str).tolist()

    # Sanity checks
    if X.shape[0] != len(y):
        raise ValueError("Feature matrix and label vector have inconsistent lengths.")

    if X.shape[1] != len(feature_names):
        raise ValueError("Number of features does not match number of var_names.")

    return X, y, row_indices, feature_names


# ============================================================
# Neural network definition
# ============================================================
def build_nn_model(
    input_dim: int,
    num_classes: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    dropout: float,
) -> tf.keras.Model:
    """
    Build a simple feedforward neural network for multi-class classification.

    Architecture
    ------------
    Input -> Dense(ReLU) -> Dropout -> Dense(ReLU) -> Dropout -> Softmax

    Parameters
    ----------
    input_dim
        Number of input features.
    num_classes
        Number of output classes.
    hidden_dim_1, hidden_dim_2
        Hidden layer sizes.
    dropout
        Dropout rate applied after each hidden layer.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim_1, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim_2, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# Output-saving helpers
# ============================================================
def save_text(text: str, path: Path) -> None:
    """
    Save plain text to disk.
    """
    path.write_text(text, encoding="utf-8")


def save_metrics_json(metrics: dict, path: Path) -> None:
    """
    Save a dictionary as formatted JSON.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    """
    Compute and save a confusion matrix figure.

    The figure size scales with the number of class labels so that
    large multi-class problems remain reasonably legible.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Scale figure size based on number of classes
    fig_width = max(10, min(24, len(labels) * 0.6))
    fig_height = max(8, min(24, len(labels) * 0.6))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_prediction_table(
    obs_names_test: np.ndarray,
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    class_probabilities: np.ndarray | None,
    class_labels: list[str],
    output_path: Path,
) -> None:
    """
    Save per-cell prediction results for the held-out test set.

    Includes:
    - observation / cell barcode
    - true label
    - predicted label
    - class probabilities (if provided)
    """
    pred_df = pd.DataFrame(
        {
            "obs_name": obs_names_test,
            "true_label": y_true_labels,
            "predicted_label": y_pred_labels,
        }
    )

    # Append one probability column per class if available
    if class_probabilities is not None:
        prob_df = pd.DataFrame(
            class_probabilities,
            columns=[f"prob_{label}" for label in class_labels],
        )
        pred_df = pd.concat([pred_df, prob_df], axis=1)

    pred_df.to_csv(output_path, index=False)


# ============================================================
# Model training: neural network
# ============================================================
def train_neural_network(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    obs_names_test: np.ndarray,
    class_labels: list[str],
    label_encoder: LabelEncoder,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    dropout: float,
) -> dict:
    """
    Train a feedforward neural network and save all relevant outputs.

    Outputs saved
    -------------
    - Trained Keras model
    - Training history CSV
    - Classification report
    - Confusion matrix image
    - Per-cell test predictions
    - Basic metrics JSON

    Returns
    -------
    dict
        Summary dictionary used later for model comparison.
    """
    logging.info("Training neural network...")

    # Build the model using the requested architecture
    model = build_nn_model(
        input_dim=X_train.shape[1],
        num_classes=len(class_labels),
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    )

    # Early stopping prevents unnecessary overtraining and restores the
    # best validation-loss weights seen during training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    # Compute balanced class weights to help compensate for class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = dict(zip(classes, weights))

    logging.info("Class weights: %s", class_weight)

    # Train model using an internal validation split taken from training data
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight,
    )

    # Predict probabilities on held-out test data
    y_pred_probs = model.predict(X_test, verbose=0)

    # Convert probability matrix to hard class calls
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert encoded integers back to original class names
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Compute evaluation metrics in label space
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels, digits=4)

    # Save model and all outputs
    model.save(output_dir / "nn_model.keras")

    pd.DataFrame(history.history).to_csv(
        output_dir / "nn_training_history.csv",
        index=False,
    )

    save_text(report, output_dir / "nn_classification_report.txt")

    save_confusion_matrix(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        labels=class_labels,
        output_path=output_dir / "nn_confusion_matrix.png",
        title="Neural Network Confusion Matrix",
    )

    save_prediction_table(
        obs_names_test=obs_names_test,
        y_true_labels=y_test_labels,
        y_pred_labels=y_pred_labels,
        class_probabilities=y_pred_probs,
        class_labels=class_labels,
        output_path=output_dir / "nn_predictions.csv",
    )

    save_metrics_json(
        {"model": "neural_network", "accuracy": accuracy},
        output_dir / "nn_metrics.json",
    )

    logging.info("Neural network accuracy: %.4f", accuracy)

    return {
        "name": "Neural Network",
        "accuracy": accuracy,
        "report": report,
    }


# ============================================================
# Model training: multinomial logistic regression
# ============================================================
def train_multinomial_logistic_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    obs_names_test: np.ndarray,
    class_labels: list[str],
    feature_names: list[str],
    label_encoder: LabelEncoder,
    output_dir: Path,
    seed: int,
) -> dict:
    """
    Train a multinomial logistic regression classifier and save outputs.

    Notes
    -----
    - This is a single multi-class model, not one-vs-rest binary models.
    - Inputs are used directly from adata.layers["lognorm"], without any extra
      scaling step.
    - Coefficients are saved both wide and long format for interpretation.

    Returns
    -------
    dict
        Summary dictionary used later for model comparison.
    """
    logging.info("Training multinomial logistic regression...")

    # Fit a true multinomial model across all classes simultaneously
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
        n_jobs=None,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Predict class labels and class probabilities on test data
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    # Decode integer labels back to original class names
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Compute performance metrics
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels, digits=4)

    # Save standard outputs
    save_text(report, output_dir / "logreg_classification_report.txt")

    save_confusion_matrix(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        labels=class_labels,
        output_path=output_dir / "logreg_confusion_matrix.png",
        title="Multinomial Logistic Regression Confusion Matrix",
    )

    save_prediction_table(
        obs_names_test=obs_names_test,
        y_true_labels=y_test_labels,
        y_pred_labels=y_pred_labels,
        class_probabilities=y_pred_probs,
        class_labels=class_labels,
        output_path=output_dir / "logreg_predictions.csv",
    )

    # Save coefficient matrix:
    # rows = classes, columns = genes/features
    coef_df = pd.DataFrame(
        model.coef_,
        index=class_labels,
        columns=feature_names,
    )
    coef_df.index.name = "Combined_category"
    coef_df.to_csv(output_dir / "logreg_coefficients.csv")

    # Save class-specific intercepts separately
    intercept_df = pd.DataFrame(
        {
            "Combined_category": class_labels,
            "intercept": model.intercept_,
        }
    )
    intercept_df.to_csv(output_dir / "logreg_intercepts.csv", index=False)

    # Also save a long-format version, sorted by coefficient magnitude
    # This is convenient for filtering top genes per class downstream
    coef_long = (
        coef_df.reset_index()
        .melt(id_vars="Combined_category", var_name="feature", value_name="coefficient")
    )
    coef_long["abs_coefficient"] = coef_long["coefficient"].abs()
    coef_long = coef_long.sort_values(
        ["Combined_category", "abs_coefficient"],
        ascending=[True, False],
    )
    coef_long.to_csv(output_dir / "logreg_coefficients_long.csv", index=False)

    save_metrics_json(
        {"model": "multinomial_logistic_regression", "accuracy": accuracy},
        output_dir / "logreg_metrics.json",
    )

    logging.info("Multinomial logistic regression accuracy: %.4f", accuracy)

    return {
        "name": "Multinomial Logistic Regression",
        "accuracy": accuracy,
        "report": report,
    }


# ============================================================
# Model training: XGBoost
# ============================================================
def train_xgboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    obs_names_test: np.ndarray,
    class_labels: list[str],
    feature_names: list[str],
    label_encoder: LabelEncoder,
    output_dir: Path,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    early_stopping_rounds: int,
) -> dict:
    """
    Train an XGBoost multi-class classifier and save outputs.

    Notes
    -----
    - Uses objective='multi:softprob' to obtain per-class probabilities.
    - Uses sample weights derived from balanced class weights.
    - Can optionally use an internal validation split for early stopping.

    Returns
    -------
    dict
        Summary dictionary used later for model comparison.
    """
    logging.info("Training XGBoost classifier...")

    # Compute class-balanced sample weights for training
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_map = dict(zip(classes, weights))
    sample_weight = np.array([class_weight_map[c] for c in y_train], dtype=np.float32)

    logging.info("XGBoost class weights: %s", class_weight_map)

    # Early stopping requires a validation set
    use_early_stopping = early_stopping_rounds > 0

    if use_early_stopping:
        (
            X_tr,
            X_val,
            y_tr,
            y_val,
            sample_weight_tr,
            _sample_weight_val,
        ) = train_test_split(
            X_train,
            y_train,
            sample_weight,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
        logging.info(
            "Using XGBoost early stopping with validation split: "
            "train=%s, val=%s",
            X_tr.shape,
            X_val.shape,
        )
    else:
        # If early stopping is disabled, train on the full training set
        X_tr, y_tr, sample_weight_tr = X_train, y_train, sample_weight
        X_val, y_val = None, None

    # Build the XGBoost model
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_labels),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        random_state=seed,
        eval_metric="mlogloss",
        n_jobs=120,
        early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
    )

    # Base fit arguments
    fit_kwargs = {
        "X": X_tr,
        "y": y_tr,
        "sample_weight": sample_weight_tr,
        "verbose": False,
    }

    # Add validation set only when early stopping is active
    if use_early_stopping and X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]

    model.fit(**fit_kwargs)

    # Predict test labels and class probabilities
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    # Decode labels back to original class names
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))

    # Compute performance metrics
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels, digits=4)

    # Save standard outputs
    save_text(report, output_dir / "xgb_classification_report.txt")

    save_confusion_matrix(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        labels=class_labels,
        output_path=output_dir / "xgb_confusion_matrix.png",
        title="XGBoost Confusion Matrix",
    )

    save_prediction_table(
        obs_names_test=obs_names_test,
        y_true_labels=y_test_labels,
        y_pred_labels=y_pred_labels,
        class_probabilities=y_pred_probs,
        class_labels=class_labels,
        output_path=output_dir / "xgb_predictions.csv",
    )

    # Save global feature importances reported by XGBoost
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "xgb_feature_importance.csv", index=False)

    # Save model in XGBoost JSON format
    model.save_model(output_dir / "xgb_model.json")

    save_metrics_json(
        {"model": "xgboost", "accuracy": accuracy},
        output_dir / "xgb_metrics.json",
    )

    logging.info("XGBoost accuracy: %.4f", accuracy)

    return {
        "name": "XGBoost",
        "accuracy": accuracy,
        "report": report,
    }


# ============================================================
# Summary output helpers
# ============================================================
def save_model_comparison(results: list[dict], output_path: Path) -> None:
    """
    Save a simple comparison table of model names and accuracies.
    """
    comparison_df = pd.DataFrame(
        [{"model": result["name"], "accuracy": result["accuracy"]} for result in results]
    ).sort_values("accuracy", ascending=False)

    comparison_df.to_csv(output_path, index=False)


def save_dataset_summary(
    class_labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save class counts for the filtered dataset.
    """
    summary_df = (
        pd.Series(class_labels)
        .value_counts()
        .rename_axis("Combined_category")
        .reset_index(name="n_cells")
    )
    summary_df.to_csv(output_path, index=False)


# ============================================================
# Main pipeline
# ============================================================
def main() -> None:
    """
    Main execution function.

    Pipeline overview
    -----------------
    1. Parse arguments
    2. Set up logging and reproducibility
    3. Load AnnData
    4. Prepare filtered feature matrix and label vector
    5. Encode labels as integers
    6. Perform stratified train/test split
    7. Train all three models
    8. Save model comparison and run metadata
    9. Print summary to console
    """
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Log hardware / TensorFlow environment
    log_hardware_info()
    gpu_sanity_check()

    # Load input AnnData object
    logging.info("Loading AnnData from %s", args.input)
    adata = load_anndata(args.input)

    logging.info(
        "AnnData loaded: %d observations, %d variables",
        adata.n_obs,
        adata.n_vars,
    )

    # Build filtered dataset from requested layer and label column
    X, y, row_indices, feature_names = prepare_dataset(
        adata=adata,
        layer_name=args.layer,
        target_column=args.target_column,
        exclude_labels=args.exclude_label,
        exclude_comma_labels=args.exclude_comma_labels,
        min_class_size=args.min_class_size,
    )

    # Keep observation names so predictions can be mapped back to cells
    obs_names = adata.obs_names[row_indices].astype(str).to_numpy()

    logging.info("Filtered dataset shape: %s", X.shape)
    logging.info("Number of classes after filtering: %d", len(pd.unique(y)))

    # Save class distribution after filtering
    save_dataset_summary(y, args.output_dir / "class_counts.csv")

    # Convert string labels to integer class IDs for model training
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_labels = label_encoder.classes_.tolist()

    # Split into train and test sets while preserving class proportions
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _obs_names_train,
        obs_names_test,
    ) = train_test_split(
        X,
        y_encoded,
        obs_names,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_encoded,
    )

    results = []

    # ------------------------------------------------------------
    # Train neural network
    # ------------------------------------------------------------
    nn_result = train_neural_network(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        obs_names_test=obs_names_test,
        class_labels=class_labels,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        dropout=args.dropout,
    )
    results.append(nn_result)

    # ------------------------------------------------------------
    # Train multinomial logistic regression
    # ------------------------------------------------------------
    logreg_result = train_multinomial_logistic_regression(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        obs_names_test=obs_names_test,
        class_labels=class_labels,
        feature_names=feature_names,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    results.append(logreg_result)

    # ------------------------------------------------------------
    # Train XGBoost classifier
    # ------------------------------------------------------------
    xgb_result = train_xgboost(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        obs_names_test=obs_names_test,
        class_labels=class_labels,
        feature_names=feature_names,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
        seed=args.seed,
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        early_stopping_rounds=args.xgb_early_stopping_rounds,
    )
    results.append(xgb_result)

    # Save single comparison table across all trained models
    save_model_comparison(results, args.output_dir / "model_comparison.csv")

    # Save run settings / metadata for reproducibility and bookkeeping
    run_metadata = {
        "input": str(args.input),
        "layer": args.layer,
        "target_column": args.target_column,
        "n_cells_after_filtering": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(class_labels)),
        "excluded_labels": args.exclude_label,
        "exclude_comma_labels": bool(args.exclude_comma_labels),
        "min_class_size": int(args.min_class_size),
        "test_size": float(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "xgb_n_estimators": int(args.xgb_n_estimators),
        "xgb_learning_rate": float(args.xgb_learning_rate),
        "xgb_max_depth": int(args.xgb_max_depth),
        "xgb_subsample": float(args.xgb_subsample),
        "xgb_colsample_bytree": float(args.xgb_colsample_bytree),
        "xgb_early_stopping_rounds": int(args.xgb_early_stopping_rounds),
    }
    save_metrics_json(run_metadata, args.output_dir / "run_metadata.json")

    # Print a concise summary to stdout
    print("\nModel comparison:")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{result['name']}: {result['accuracy']:.4f}")

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
