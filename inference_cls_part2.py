#!/usr/bin/env python3
"""
Aggregate nnUNet CLS validation predictions from per-fold results.csv files.

Layout assumed:

  ROOT_RESULTS = /hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/\
                 Dataset033_AVM_T1+C/\
                 DenseNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres

Inside ROOT_RESULTS:

  fold_0/
    cls_results_checkpoint_bestauc/results.csv
    cls_results_checkpoint_every20_best_balacc/results.csv
    ...
  fold_1/
    cls_results_checkpoint_bestauc/results.csv
    ...

Each per-fold results.csv has columns:
  - identifier
  - probs      (e.g. "[0.74, 0.26]")
  - pred_class (0/1)

Ground truth labels are taken from cls_data.csv in nnUNet_preprocessed, which
we locate automatically by:

  1. Taking dataset name from ROOT_RESULTS.parent.name (e.g. Dataset033_AVM_T1+C)
  2. Replacing "nnUNet_results" with "nnUNet_preprocessed"
  3. Recursively searching for a cls_data.csv

The script:
  - Aggregates predictions across folds for each chosen checkpoint type.
  - Joins with cls_data.csv on `identifier` to get `label` as y_true.
  - Computes:
      * Aggregated metrics
      * Per-fold metrics
      * Bootstrap 95% CIs
      * Confusion matrix PNG
  - Writes outputs to ROOT_RESULTS/<checkpoint_type>/, e.g.:

      ROOT_RESULTS/
        cls_results_checkpoint_every20_best_balacc/
          aggregated_metrics.csv
          per_fold_metrics.csv
          bootstrap_ci_95.csv
          confusion_matrix_aggregated.png
          all_predictions.csv

You can control which checkpoint result folders to aggregate by editing
CHOOSE_CHECKPOINT_RESULTS below.
"""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchmetrics import F1Score, AveragePrecision, AUROC

# -------------------- Optional custom BalancedAccuracy --------------------
_BALACC_IS_TORCHMETRIC = False
try:
    # Same as in your SSL3D codebase – adjust if needed.
    from metrics.balanced_accuracy import BalancedAccuracy  # type: ignore

    _BALACC_IS_TORCHMETRIC = True
except Exception:
    from sklearn.metrics import balanced_accuracy_score as _sk_balanced_accuracy_score  # type: ignore

    BalancedAccuracy = None  # type: ignore



# -------------------- Scan-level extra metrics config --------------------
FOCAL_ALPHA_POS = 0.25   # alpha for positive class (pos)
FOCAL_GAMMA = 2.0        # focusing parameter gamma
def add_error_types(
    df: pd.DataFrame,
    gt_col: str = "label",
    pred_col: str = "pred_class",
    out_col: str = "Error_Type",
) -> pd.DataFrame:
    """Add TP/TN/FP/FN labels for binary tasks."""
    gt = df[gt_col].astype(int)
    pr = df[pred_col].astype(int)
    out = np.where((gt == 1) & (pr == 1), "TP",
          np.where((gt == 0) & (pr == 0), "TN",
          np.where((gt == 0) & (pr == 1), "FP", "FN")))
    df[out_col] = out
    return df




# -------------------- USER CONFIG --------------------

# Hard-coded root for this experiment
ROOT_RESULTS = Path(
    "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/DenseNetTrainer_ep300_NoMirroring_x2__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/SEResNetTrainer_ep300_NoMirroring_x2__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/SwinViTTrainer_ep300_NoMirroring_x2__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/ViTTrainer_ep300_NoMirroring_x2__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset044_AVM_TOFMRA/DenseNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset044_AVM_TOFMRA/SEResNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset044_AVM_TOFMRA/SwinViTTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset044_AVM_TOFMRA/ViTTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset003_AVM_T1+C/DenseNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset003_AVM_T1+C/SEResNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset003_AVM_T1+C/SwinViTTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset003_AVM_T1+C/ViTTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    ).expanduser().resolve()

# Which checkpoint result subfolders to aggregate.
# Edit this list depending on what you want:
#   ['cls_results_checkpoint_bestauc']
#   ['cls_results_checkpoint_bestauc', 'cls_results_checkpoint_every20_best_balacc']
#   ['cls_results_checkpoint_bestauc', 'cls_results_checkpoint_bestacc', 'cls_results_checkpoint_every20_best_balacc']
CHOOSE_CHECKPOINT_RESULTS = [
    "cls_results_checkpoint_every20_best_balacc",
    "cls_results_checkpoint_every20_best_auroc",
    # "cls_results_checkpoint_bestauc",
    # "cls_results_checkpoint_bestacc",
    # "checkpoint_final",
]

# Number of bootstrap resamples & seed
N_BOOTSTRAPS_DEFAULT = 1000
BOOTSTRAP_SEED_DEFAULT = 42


# -------------------- Helpers --------------------

def softmax_np(x: Iterable[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def parse_probs_column(obj) -> np.ndarray:
    """
    Parse the 'probs' column from results.csv into a numpy array.

    Accepts formats like:
      "[0.74, 0.26]"
      (0.74, 0.26)
      list, tuple, ndarray
    """
    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj, dtype=float).reshape(-1)
        return arr

    s = str(obj).strip()
    # Try literal_eval first
    try:
        cand = ast.literal_eval(s)
        if isinstance(cand, (list, tuple)):
            arr = np.asarray(cand, dtype=float).reshape(-1)
            return arr
    except Exception:
        pass

    # Fallback: strip [] and split by comma
    s2 = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [p for p in s2.split(",") if p.strip() != ""]
    arr = np.asarray([float(p) for p in parts], dtype=float).reshape(-1)
    return arr


@dataclass
class MetricResults:
    sensitivity: float | float("nan")
    specificity: float | float("nan")
    auroc: float
    accuracy: float
    average_precision: float
    balanced_accuracy: float
    f1_macro: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Recall(Sensitivity)": float(self.sensitivity) if not np.isnan(self.sensitivity) else np.nan,
            "Specificity": float(self.specificity) if not np.isnan(self.specificity) else np.nan,
            "AUROC": float(self.auroc),
            "Accuracy": float(self.accuracy),
            "Average Precision": float(self.average_precision),
            "Balanced Accuracy": float(self.balanced_accuracy),
            "F1 Score": float(self.f1_macro),
        }


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, n_classes: int
) -> MetricResults:
    """Compute metrics for binary or multiclass."""
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_pred_t = torch.tensor(y_pred, dtype=torch.long)

    # Balanced Accuracy
    if _BALACC_IS_TORCHMETRIC and BalancedAccuracy is not None:
        balacc_metric = BalancedAccuracy(num_classes=n_classes, task="multiclass")
        balanced_acc = float(balacc_metric(y_pred_t, y_true_t).item())
    else:
        from sklearn.metrics import balanced_accuracy_score

        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Torchmetrics configs
    probs_t = torch.tensor(probs, dtype=torch.float)

    if n_classes == 2:
        # Ensure probs_t is (N,2)
        if probs.ndim == 1:
            probs_t = torch.stack([1 - probs_t, probs_t], dim=1)
        elif probs.shape[1] == 1:
            probs_t = torch.stack([1 - probs_t[:, 0], probs_t[:, 0]], dim=1)

        auroc_metric = AUROC(task="multiclass", num_classes=2)
        ap_metric = AveragePrecision(task="multiclass", num_classes=2, average="macro")
        f1_torch = F1Score(task="multiclass", num_classes=2, average="macro")

        auroc = float(auroc_metric(probs_t, y_true_t).item())
        ap = float(ap_metric(probs_t, y_true_t).item())
        f1_val = float(f1_torch(y_pred_t, y_true_t).item())

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = 0.0
            specificity = 0.0
    else:
        auroc_metric = AUROC(task="multiclass", num_classes=n_classes)
        ap_metric = AveragePrecision(task="multiclass", num_classes=n_classes, average="macro")
        f1_torch = F1Score(task="multiclass", num_classes=n_classes, average="macro")
        auroc = float(auroc_metric(probs_t, y_true_t).item())
        ap = float(ap_metric(probs_t, y_true_t).item())
        f1_val = float(f1_torch(y_pred_t, y_true_t).item())
        sensitivity = float("nan")
        specificity = float("nan")

    from sklearn.metrics import accuracy_score

    acc = float(accuracy_score(y_true, y_pred))

    return MetricResults(
        sensitivity=sensitivity,
        specificity=specificity,
        auroc=auroc,
        accuracy=acc,
        average_precision=ap,
        balanced_accuracy=balanced_acc,
        f1_macro=f1_val,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str = "Aggregated OOF",
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(1.2 * len(class_names), 1.0 * len(class_names)))
    im = ax.imshow(np.nan_to_num(cm_norm), interpolation="nearest", cmap="PRGn", vmin=0.0, vmax=1.0)

    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = 0.75
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if row_sums[i] != 0:
                val = cm_norm[i, j]
            else:
                val = 0.0
            color = "white" if val >= thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({val:.2f})", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=9)

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def find_cls_data_csv(root_results: Path) -> Path:
    """
    Given ROOT_RESULTS under nnUNet_results/<DatasetName>/<Trainer>..., locate
    the matching cls_data.csv in nnUNet_preprocessed/<DatasetName>/... by
    searching recursively.
    """
    # e.g. ROOT_RESULTS.parent.name = "Dataset033_AVM_T1+C"
    dataset_name = root_results.parent.name

    # e.g. /hpf/projects/jquon/sumin/nnUNet_data
    data_root = root_results.parents[2]
    preproc_base = data_root / "nnUNet_preprocessed" / dataset_name

    if not preproc_base.exists():
        raise FileNotFoundError(f"Could not find preprocessed base: {preproc_base}")

    candidates = list(preproc_base.rglob("cls_data.csv"))
    if not candidates:
        raise FileNotFoundError(f"No cls_data.csv found under: {preproc_base}")
    if len(candidates) > 1:
        # If multiple, you can either pick first or raise. For now, be explicit.
        print("[WARN] Multiple cls_data.csv found. Using the first:")
        for c in candidates:
            print(f"   - {c}")
        return candidates[0]

    return candidates[0]


def load_labels_df(cls_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cls_csv_path)
    required = {"identifier", "label"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"cls_data.csv at {cls_csv_path} is missing required columns {required}. "
            f"Found: {sorted(df.columns)}"
        )
    return df[["identifier", "label"]].copy()


def collect_results_for_checkpoint_type(
    root_results: Path,
    checkpoint_type: str,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a given checkpoint_type (e.g. 'cls_results_checkpoint_every20_best_balacc'),
    collect all folds' results.csv, parse probabilities, join with labels, and return
    a single DataFrame with columns:

      identifier, probs_vec, pred_class, Fold, label
    """
    all_rows = []

    # Find fold_* directories
    for fold_dir in sorted(root_results.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        fold_name = fold_dir.name  # e.g. "fold_0"
        try:
            fold_idx = int(fold_name.split("_")[1])
        except Exception:
            print(f"[WARN] Skipping {fold_dir} (cannot parse fold index).")
            continue

        ckpt_dir = fold_dir / checkpoint_type
        results_path = ckpt_dir / "results.csv"
        if not results_path.exists():
            print(f"[INFO]   No results.csv for {checkpoint_type} in {fold_dir} – skipping.")
            continue

        print(f"[INFO]   Reading {results_path}")
        df = pd.read_csv(results_path)
        required = {"identifier", "probs", "pred_class"}
        if not required.issubset(df.columns):
            print(
                f"[WARN]   Missing required columns in {results_path}. "
                f"Have: {sorted(df.columns)}; need: {sorted(required)}"
            )
            continue

        df = df.copy()
        df["probs_vec"] = df["probs"].apply(parse_probs_column)
        df["Fold"] = fold_idx

        # join with labels (on identifier)
        df = df.merge(labels_df, on="identifier", how="left")
        missing = df["label"].isna().sum()
        if missing > 0:
            print(
                f"[WARN]   {missing} rows in {results_path} could not be matched to labels in cls_data.csv. "
                f"These will be dropped for metrics."
            )
            df = df.dropna(subset=["label"])

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            f"No valid prediction rows found for checkpoint_type={checkpoint_type}. "
            f"Did you run inference to create results.csv?"
        )

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


def aggregate_and_save_metrics(
    df_all: pd.DataFrame,
    out_dir: Path,
    n_bootstraps: int = N_BOOTSTRAPS_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> None:
    """
    Given the merged df_all (identifier, probs_vec, pred_class, Fold, label),
    compute aggregated metrics, per-fold metrics, bootstrap CIs, and confusion matrix,
    and save them to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build arrays
    y_true = df_all["label"].astype(int).to_numpy()
    y_pred = df_all["pred_class"].astype(int).to_numpy()
    probs = np.vstack(
        [np.asarray(v, dtype=float).reshape(-1) for v in df_all["probs_vec"].tolist()]
    )

    # Determine number of classes
    n_classes = probs.shape[1] if probs.ndim == 2 else 2
    print(f"[INFO] Using num_classes = {n_classes}")

    # ------------------- Scan-level per-scan metrics (row-wise) -------------------
    eps = 1e-7
    df_all["IsCorrect"] = (df_all["label"].astype(int) == df_all["pred_class"].astype(int)).astype(int)

    if n_classes == 2:
        # Binary case: mirror SSL script behaviour
        add_error_types(df_all, gt_col="label", pred_col="pred_class", out_col="Error_Type")
        # Positive class prob
        prob_pos = probs[:, 1]
        df_all["Prob1"] = prob_pos
        prob_pos_clipped = np.clip(prob_pos, eps, 1.0 - eps)
        y_true_int = y_true.astype(int)
        # Brier score
        df_all["BrierScore"] = (prob_pos - y_true_int) ** 2
        # Cross-entropy
        df_all["CrossEntropy"] = -(
            y_true_int * np.log(prob_pos_clipped)
            + (1.0 - y_true_int) * np.log(1.0 - prob_pos_clipped)
        )
        # Focal loss
        alpha_pos = FOCAL_ALPHA_POS
        alpha_neg = 1.0 - FOCAL_ALPHA_POS
        p_t = np.where(y_true_int == 1, prob_pos_clipped, 1.0 - prob_pos_clipped)
        alpha_t = np.where(y_true_int == 1, alpha_pos, alpha_neg)
        df_all["FocalLoss"] = -alpha_t * ((1.0 - p_t) ** FOCAL_GAMMA) * np.log(p_t)
        # Margin (distance from 0.5)
        df_all["Margin"] = prob_pos - 0.5
        # Entropy
        df_all["Entropy"] = -(
            prob_pos_clipped * np.log(prob_pos_clipped)
            + (1.0 - prob_pos_clipped) * np.log(1.0 - prob_pos_clipped)
        )
    else:
        # Multiclass generalization
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        y_true_int = y_true.astype(int)
        # One-hot y
        y_onehot = np.eye(n_classes)[y_true_int]
        # Brier score: sum_c (p_c - y_c)^2
        df_all["BrierScore"] = np.sum((probs_clipped - y_onehot) ** 2, axis=1)
        # Probability of true class
        p_t = probs_clipped[np.arange(len(y_true_int)), y_true_int]
        # Cross-entropy: -log p_true
        df_all["CrossEntropy"] = -np.log(p_t)
        # Focal loss (uniform alpha)
        df_all["FocalLoss"] = -((1.0 - p_t) ** FOCAL_GAMMA) * np.log(p_t)
        # Margin: p_true - max(other probs)
        probs_masked = probs_clipped.copy()
        probs_masked[np.arange(len(y_true_int)), y_true_int] = -np.inf
        second_best = np.max(probs_masked, axis=1)
        df_all["Margin"] = p_t - second_best
        # Entropy: -sum_c p_c log p_c
        df_all["Entropy"] = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)

    # --------------------------------------------------------------------







    # ------------------- Aggregated metrics -------------------
    agg = compute_metrics(y_true, y_pred, probs, n_classes)
    agg_dict = agg.to_dict()

    print("\n=== Aggregated (all folds) ===")
    for k, v in agg_dict.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    pd.DataFrame([agg_dict]).to_csv(out_dir / "aggregated_metrics.csv", index=False)

    # ------------------- Per-fold metrics -------------------
    print("\n=== Per-fold ===")
    fold_metrics: Dict[str, Dict[str, Any]] = {}
    for fold_idx, df_fold in df_all.groupby("Fold"):
        y_true_f = df_fold["label"].astype(int).to_numpy()
        y_pred_f = df_fold["pred_class"].astype(int).to_numpy()
        probs_f = np.vstack(
            [np.asarray(v, dtype=float).reshape(-1) for v in df_fold["probs_vec"].tolist()]
        )
        fmetrics = compute_metrics(y_true_f, y_pred_f, probs_f, n_classes).to_dict()
        fold_metrics[f"Fold{fold_idx}"] = fmetrics

    df_metrics_folds = pd.DataFrame(fold_metrics)
    desired_order = [
        "AUROC",
        "Accuracy",
        "Average Precision",
        "Balanced Accuracy",
        "F1 Score",
        "Recall(Sensitivity)",
        "Specificity",
    ]
    df_metrics_folds = df_metrics_folds.reindex(
        [m for m in desired_order if m in df_metrics_folds.index]
    ).round(4)
    # NEW: append aggregated column (pooled across all folds)
    df_metrics_folds["Aggregated"] = pd.Series(agg_dict).reindex(df_metrics_folds.index).round(4)

    print(df_metrics_folds)
    df_metrics_folds.to_csv(out_dir / "per_fold_metrics.csv")

    # ------------------- Bootstrap CIs (aggregated) -------------------
    print("\n[INFO] Bootstrapping ...")
    rng = np.random.default_rng(int(seed))
    keys = list(agg_dict.keys())
    boot_results: Dict[str, List[float]] = {k: [] for k in keys}

    N = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.choice(N, N, replace=True)
        res = compute_metrics(y_true[idx], y_pred[idx], probs[idx], n_classes).to_dict()
        for k in keys:
            boot_results[k].append(res[k])

    rows = []
    print("\n=== 95% CI (Aggregated) ===")
    for k in keys:
        vals = np.array(boot_results[k], dtype=float)
        est = float(agg_dict[k]) if not np.isnan(agg_dict[k]) else np.nan
        if np.isnan(est):
            low = high = np.nan
            print(f"{k}: n/a")
        else:
            low, high = np.percentile(vals, [2.5, 97.5])
            print(f"{k}: {est:.3f} (95% CI: {low:.3f} – {high:.3f})")
        rows.append({"metric": k, "estimate": est, "low_95": low, "high_95": high})

    pd.DataFrame(rows).to_csv(out_dir / "bootstrap_ci_95.csv", index=False)

    # ------------------- Confusion matrix plot -------------------
    class_names = [str(i) for i in range(probs.shape[1])]
    cm_out = out_dir / "confusion_matrix_aggregated.png"
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        out_path=cm_out,
        title="Aggregated OOF Predictions",
    )
    print(f"\n[INFO] Saved confusion matrix: {cm_out}")

    # ------------------- Save combined predictions -------------------
    df_save = df_all.copy()
    # make probs_vec into list-of-floats for readability
    df_save["probs"] = df_save["probs_vec"].apply(lambda v: list(np.asarray(v, float)))
    df_save.drop(columns=["probs_vec"], inplace=True)
    df_save.to_csv(out_dir / "all_predictions.csv", index=False)
    print(f"[INFO] Saved combined predictions: {out_dir / 'all_predictions.csv'}")

    # Also save scan-level metrics with all columns
    scan_metrics_path = out_dir / "scan_level_per_scan_metrics_ALLCOLS.csv"
    df_save.to_csv(scan_metrics_path, index=False)
    print(f"[INFO] Saved scan-level per-scan metrics to: {scan_metrics_path}")

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate nnUNet CLS VAL predictions across folds."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT_RESULTS),
        help="Root results folder (default is hard-coded nnUNet_results experiment).",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=N_BOOTSTRAPS_DEFAULT,
        help="Number of bootstrap resamples for 95% CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED_DEFAULT,
        help="Random seed for bootstrap.",
    )
    args = parser.parse_args()

    root_results = Path(args.root).expanduser().resolve()
    if not root_results.exists():
        raise FileNotFoundError(f"Root results folder not found: {root_results}")

    print(f"[INFO] Root results: {root_results}")

    # Locate cls_data.csv and load labels
    cls_csv_path = find_cls_data_csv(root_results)
    print(f"[INFO] Using cls_data.csv at: {cls_csv_path}")
    labels_df = load_labels_df(cls_csv_path)

    # Loop over chosen checkpoint result types
    for ckpt_type in CHOOSE_CHECKPOINT_RESULTS:
        print("\n" + "=" * 80)
        print(f"[INFO] Processing checkpoint type: {ckpt_type}")
        print("=" * 80)

        df_all = collect_results_for_checkpoint_type(root_results, ckpt_type, labels_df)

        # Aggregated outputs go into ROOT_RESULTS/<checkpoint_type>/
        out_dir = root_results / ckpt_type
        aggregate_and_save_metrics(
            df_all,
            out_dir,
            n_bootstraps=int(args.n_bootstraps),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
