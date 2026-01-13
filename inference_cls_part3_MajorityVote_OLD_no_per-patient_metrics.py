#!/usr/bin/env python3
"""
Aggregate nnUNet CLS validation predictions with study-level majority voting
and pre-1st-recurrence majority voting (like SSL3D inference_val_part3_MajorityVote).

This version is adapted for the nnUNet classification baseline, using per-fold
results.csv files produced under:

  ROOT_RESULTS = /.../nnUNet_results/DatasetXXX_.../<TrainerName>...

where each fold directory looks like:

  fold_0/
    cls_results_checkpoint_bestauc/results.csv
    cls_results_checkpoint_every20_best_balacc/results.csv
    ...

Each per-fold results.csv has columns:
  - identifier
  - probs      (e.g. "[0.74, 0.26]")
  - pred_class (0/1)

Ground-truth labels and scan-level metadata (study_id, class_label, date, fold, ...)
are loaded from the same CSVs used in the SSL3D pipeline, chosen by dataset id
embedded in ROOT_RESULTS, for example:

  Dataset033 -> T1+C_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv
  Dataset044 -> TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv

The script, for each chosen checkpoint type:
  - Collects scan-level predictions across folds.
  - Merges with scan metadata on identifier/new_name.
  - Computes study-level majority vote (counts first, tie -> mean Prob1 >= 0.5).
  - Computes aggregated + per-fold metrics (study-level).
  - Computes bootstrap 95% CIs for study-level metrics.
  - Plots a study-level confusion matrix.
  - Recomputes majority vote restricted to "pre-1st-recurrence" slices using RULES_3
    and repeats the same metrics/CI/confusion-matrix procedure.

Outputs are written under:

  ROOT_RESULTS/<checkpoint_type>/

e.g.:

  ROOT_RESULTS/
    cls_results_checkpoint_every20_best_balacc/
      mv_metrics_per_fold.csv
      bootstrap_ci_mv_95.csv
      confusion_matrix_mv.png
      bootstrap_ci_mv_pre1st_95.csv
      df_pre1strec_metrics_per_fold.csv
      confusion_matrix_mv_df_pre1strec.png
"""

from __future__ import annotations

import argparse
import ast
import re
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
    # Same helper as in SSL3D codebase; adjust import path if needed.
    from metrics.balanced_accuracy import BalancedAccuracy  # type: ignore
    _BALACC_IS_TORCHMETRIC = True
except Exception:
    from sklearn.metrics import balanced_accuracy_score as _sk_balanced_accuracy_score  # type: ignore
    BalancedAccuracy = None  # type: ignore

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

# -------------------- USER CONFIG --------------------

# Hard-coded default root for this experiment (can be overridden via --root)
ROOT_RESULTS = Path(
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/DenseNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/SEResNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/SwinViTTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    # "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset033_AVM_T1+C/ViTTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
    ).expanduser().resolve()
# ROOT_RESULTS = Path(
#     "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/"
#     "Dataset033_AVM_T1+C/"
#     "DenseNetTrainer_ep300_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres"
# ).expanduser().resolve()

# Which checkpoint result subfolders to process.
# Edit this list depending on what you want:
#   ['cls_results_checkpoint_bestauc']
#   ['cls_results_checkpoint_bestauc', 'cls_results_checkpoint_every20_best_balacc']
#   ['cls_results_checkpoint_bestauc', 'cls_results_checkpoint_bestacc',
#    'cls_results_checkpoint_every20_best_balacc']
CHOOSE_CHECKPOINT_RESULTS = [
    "cls_results_checkpoint_every20_best_balacc",
    # "cls_results_checkpoint_bestauc",
    # "cls_results_checkpoint_bestacc",
]

N_BOOTSTRAPS_DEFAULT = 2000
BOOTSTRAP_SEED_DEFAULT = 42

# --------------------------- Majority-vote RULES --------------------------- #
RULES_3 = {
    449: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20010616, "label": 'pre-1st-recurrence'},
          {"start": 20010616, "end": 20011120, "label": '1_2'},
          {"start": 20011120, "end": 99999999, "label": '0_2'},],
    455: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20001202, "label": 'pre-1st-recurrence'},
          {"start": 20001202, "end": 20010323, "label": '1_2'},
          {"start": 20010323, "end": 99999999, "label": '0_2'},],
    456: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20030621, "label": 'pre-1st-recurrence'},
          {"start": 20030621, "end": 20030726, "label": '1_2'},
          {"start": 20030726, "end": 99999999, "label": '0_2'},],
    462: [{"start": 0, "end": 20000129, "label": 'pre-1st-recurrence'},
          {"start": 20000129, "end": 20040819, "label": 'pre-1st-recurrence'},
          {"start": 20040819, "end": 20050707, "label": '1_2'},
          {"start": 20050707, "end": 99999999, "label": '0_2'},],
    540: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20050109, "label": 'pre-1st-recurrence'},
          {"start": 20050109, "end": 20051022, "label": '1_2'},
          {"start": 20051022, "end": 99999999, "label": '0_2'},],
    554: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20100604, "label": 'pre-1st-recurrence'},
          {"start": 20100604, "end": 20100707, "label": '1_2'},
          {"start": 20100707, "end": 99999999, "label": '0_2'},],
    569: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20001118, "label": 'pre-1st-recurrence'},
          {"start": 20001118, "end": 20010120, "label": '1_2'},
          {"start": 20010120, "end": 99999999, "label": '0_2'},],
    585: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20010415, "label": 'pre-1st-recurrence'},
          {"start": 20010415, "end": 20010608, "label": '1_2'},
          {"start": 20010608, "end": 99999999, "label": '0_2'},],
    593: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20001022, "label": 'pre-1st-recurrence'},
          {"start": 20001022, "end": 20001027, "label": '1_2'},
          {"start": 20001027, "end": 20010427, "label": '0_2'},
          {"start": 20010427, "end": 20020601, "label": '1_3'},
          {"start": 20020601, "end": 20021101, "label": '0_3'},
          {"start": 20021101, "end": 20030428, "label": '1_4'},
          {"start": 20030428, "end": 99999999, "label": '0_4'},],
    597: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20010313, "label": 'pre-1st-recurrence'},
          {"start": 20010313, "end": 20010930, "label": '1_2'},
          {"start": 20010930, "end": 99999999, "label": '0_2'},],
    611: [{"start": 0, "end": 20000101, "label": 'pre-1st-recurrence'},
          {"start": 20000101, "end": 20050605, "label": 'pre-1st-recurrence'},
          {"start": 20050605, "end": 99999999, "label": '1_2'},
          {"start": 99999999, "end": 99999999, "label": '0_2'},],
}

# --------------------------- Helpers --------------------------- #
def softmax_np(x: Iterable[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def parse_probs_column(obj) -> np.ndarray:
    """
    Parse the 'probs' column from nnUNet results.csv into a numpy array.

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


def add_error_types(df: pd.DataFrame, pred_col: str, out_col: str) -> pd.DataFrame:
    gt = df["GroundTruth"].astype(int)
    pr = df[pred_col].astype(int)
    out = np.where((gt == 1) & (pr == 1), "TP",
          np.where((gt == 0) & (pr == 0), "TN",
          np.where((gt == 0) & (pr == 1), "FP", "FN")))
    df[out_col] = out
    return df


def majority_vote_with_prob_tiebreak(group: pd.DataFrame) -> int:
    """True majority vote; tie -> mean Prob1 >= 0.5."""
    preds = group["Prediction"].astype(int).tolist()
    zeros = preds.count(0)
    ones  = preds.count(1)
    if zeros > ones:
        return 0
    if ones > zeros:
        return 1
    return int(group["Prob1"].mean() >= 0.5)


@dataclass
class MetricResults:
    sensitivity: float
    specificity: float
    auroc: float
    accuracy: float
    average_precision: float
    balanced_accuracy: float
    f1_macro: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Recall(Sensitivity)": float(self.sensitivity) if not np.isnan(self.sensitivity) else np.nan,
            "Specificity": float(self.specificity) if not np.isnan(self.specificity) else np.nan,
            "AUROC": float(self.auroc),
            "Accuracy": float(self.accuracy),
            "Average Precision": float(self.average_precision),
            "Balanced Accuracy": float(self.balanced_accuracy),
            "F1 Score": float(self.f1_macro),
            "TN": int(self.tn),
            "FP": int(self.fp),
            "FN": int(self.fn),
            "TP": int(self.tp),
        }


def compute_metrics(y_true, y_pred, probs) -> Dict[str, Any]:
    """TorchMetrics for F1/AP/AUROC + repo BalancedAccuracy; sklearn for the rest."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probs  = np.asarray(probs, dtype=float)

    unique_classes = np.unique(y_true)
    has_both_classes = (len(unique_classes) == 2)

    # Confusion matrix + base rates
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Balanced Accuracy (prefer repo metric if present)
    if _BALACC_IS_TORCHMETRIC and BalancedAccuracy is not None:
        balacc = BalancedAccuracy(num_classes=2, task="multiclass")(  # type: ignore
            torch.tensor(y_pred, dtype=torch.long),
            torch.tensor(y_true, dtype=torch.long),
        ).item()
    else:
        from sklearn.metrics import balanced_accuracy_score
        balacc = balanced_accuracy_score(y_true, y_pred)

    # Torch metrics
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_pred_t = torch.tensor(y_pred, dtype=torch.long)
    probs_t  = torch.tensor(probs, dtype=torch.float)
    probs_2d = torch.stack([1 - probs_t, probs_t], dim=1)

    # F1 (macro) is still defined even if only one class appears
    f1_val = F1Score(task='multiclass', num_classes=2, average='macro')(
        y_pred_t, y_true_t
    ).item()

    # AUROC & AP: ONLY compute if we actually have both classes
    if has_both_classes:
        auroc_val = AUROC(task='multiclass', num_classes=2)(probs_2d, y_true_t).item()
        ap_val = AveragePrecision(task='multiclass', num_classes=2)(probs_2d, y_true_t).item()
    else:
        auroc_val = np.nan
        ap_val = np.nan

    res = MetricResults(
        sensitivity=sensitivity,
        specificity=specificity,
        auroc=auroc_val,
        accuracy=acc,
        average_precision=ap_val,
        balanced_accuracy=balacc,
        f1_macro=f1_val,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
    return res.to_dict()


def save_confusion_matrix(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    im = ax.imshow(np.nan_to_num(cm_norm), interpolation="nearest", cmap="PRGn", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=8)
    ax.set_yticklabels(["0", "1"], fontsize=8)
    thresh = 0.75
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j] if row_sums[i] != 0 else 0.0
            color = "white" if val >= thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({val:.2f})", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0]).ax.tick_params(labelsize=6)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def assign_scan_label(row, rules: dict) -> str:
    """
    Returns one of: 'pre-1st-recurrence', '1_2', '0_2', ... or 'FALLBACK'
    Logic:
      - If class_label == 0 => 'pre-1st-recurrence'
      - Else if study_id in rules, apply date window
      - Else 'FALLBACK'
    """
    if int(row["class_label"]) == 0:
        return "pre-1st-recurrence"
    sid = int(row["study_id"])
    date = int(row["date"])
    for rule in rules.get(sid, []):
        if rule["start"] <= date < rule["end"]:
            return rule["label"]
    return "FALLBACK"


# --------------------------- Dataset → metadata path --------------------------- #
def pick_scan_metadata_path(exp_dir: Path) -> Tuple[str, bool]:
    """
    Return (path, needs_concat) based on dataset in exp_dir (ROOT_RESULTS path).
    """
    s = str(exp_dir)

    # helper: match DatasetXXX exactly (allowing zero padding)
    def dataset_match(s: str, num: int) -> bool:
        return re.search(fr"Dataset0*{num}(?!\d)", s, flags=re.IGNORECASE) is not None

    m = re.search(r"Dataset0*([0-9]+)", s, flags=re.IGNORECASE)
    ds = m.group(1) if m else ""

    # explicit Dataset055 check first
    if dataset_match(s, 55):
        return (
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "combined_T1+C_TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv",
            False,
        )
    # Dataset033 (T1+C only, LOO)
    if dataset_match(s, 33):
        return (
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "T1+C_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv",
            False,
        )
    # Dataset044 (TOFMRA only, LOO)
    if dataset_match(s, 44):
        return (
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv",
            False,
        )

    if ds in {"4", "004", "7", "007"}:
        return (
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "TOFMRA_5folds_scanlvl_RRRRR_NEWSPLITS.csv",
            False,
        )
    if ds in {"3", "003", "6", "006"}:
        return (
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "T1+C_5folds_scanlvl_RRRRR_NEWSPLITS.csv",
            False,
        )
    if ds in {"5", "005"}:
        # Concatenate T1+C and TOFMRA
        return "CONCAT_3_6_4_7", True

    # Fallback: no match -> raise so user notices
    raise RuntimeError(
        "Could not infer dataset from ROOT_RESULTS path. Please include 'DatasetXXX' in the path "
        "or add a case in pick_scan_metadata_path()."
    )



def load_scan_metadata(exp_dir: Path) -> pd.DataFrame:
    """
    Load scan-level metadata and create a PatientID_key that matches nnUNet
    results 'PatientID' (which lacks the '.nii.gz' suffix).
    """
    meta_path, needs_concat = pick_scan_metadata_path(exp_dir)

    if not needs_concat:
        df = pd.read_csv(meta_path)
    else:
        df1 = pd.read_csv(
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "T1+C_5folds_scanlvl_RRRRR_NEWSPLITS.csv"
        )
        df2 = pd.read_csv(
            "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/"
            "TOFMRA_5folds_scanlvl_RRRRR_NEWSPLITS.csv"
        )
        df = pd.concat([df1, df2], ignore_index=True)

    required = {"new_name", "class_label", "study_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Scan metadata CSV {meta_path} is missing columns: {missing}. "
            f"Found: {sorted(df.columns)}"
        )

    # new_name like "406_MRI_19991106_T1+C.nii.gz" -> "406_MRI_19991106_T1+C"
    df["PatientID_key"] = df["new_name"].str.replace(".nii.gz", "", regex=False)

    return df



# --------------------------- Bootstrap CI helper --------------------------- #
METRIC_KEYS = [
    "Recall(Sensitivity)",
    "Specificity",
    "AUROC",
    "Accuracy",
    "Average Precision",
    "Balanced Accuracy",
    "F1 Score",
]


def bootstrap_ci(
    y_true,
    y_pred,
    probs,
    base_metrics: Dict[str, Any],
    label: str = "Aggregated",
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Nonparametric bootstrap CIs over study-level metrics.

    y_true, y_pred, probs should all be study-level vectors.
    base_metrics is the dict returned by compute_metrics on the full data.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probs  = np.asarray(probs, dtype=float)

    N = len(y_true)
    if N == 0:
        raise ValueError("bootstrap_ci called with empty arrays.")

    rng = np.random.default_rng(seed)
    boot_results = {k: [] for k in METRIC_KEYS}

    for _ in range(n_bootstraps):
        idx = rng.choice(N, N, replace=True)
        res = compute_metrics(y_true[idx], y_pred[idx], probs[idx])
        for k in METRIC_KEYS:
            boot_results[k].append(res.get(k, np.nan))

    rows = []
    print(f"\n=== 95% CI ({label}) ===")
    for k in METRIC_KEYS:
        vals = np.array(boot_results[k], dtype=float)
        est = float(base_metrics.get(k, np.nan))

        # Drop NaNs from the bootstrap distribution
        vals = vals[~np.isnan(vals)]
        if np.isnan(est) or vals.size == 0:
            low = high = np.nan
            print(f"{k}: {est} (95% CI: n/a)")
        else:
            low, high = np.percentile(vals, [2.5, 97.5])
            print(f"{k}: {est:.3f} (95% CI: {low:.3f} – {high:.3f})")

        rows.append({"metric": k, "estimate": est, "low_95": low, "high_95": high})

    return pd.DataFrame(rows)


# --------------------------- nnUNet results collector --------------------------- #
def collect_scan_predictions_nnUNet(
    root_results: Path,
    checkpoint_type: str,
) -> pd.DataFrame:
    """
    For a given checkpoint_type (e.g. 'cls_results_checkpoint_every20_best_balacc'),
    collect scan-level predictions across all folds into a single DataFrame with
    at least the columns:

      PatientID, Prediction, Prob_vec, Prob1, Fold
    """
    all_rows = []

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
        df["PatientID"] = df["identifier"].astype(str)
        df["Prediction"] = df["pred_class"].astype(int)
        df["Prob_vec"] = df["probs"].apply(parse_probs_column)
        df["Prob1"] = df["Prob_vec"].apply(
            lambda v: float(np.asarray(v, dtype=float).reshape(-1)[1])
            if len(np.asarray(v).reshape(-1)) >= 2
            else float(np.asarray(v, dtype=float).reshape(-1)[0])
        )
        df["Fold"] = fold_idx
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            f"No valid prediction rows found for checkpoint_type={checkpoint_type}. "
            f"Did you run nnUNet classification inference to create results.csv?"
        )

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


# ------------------------------ MAIN per-checkpoint flow ------------------------------ #
def run_for_checkpoint_type(
    root_results: Path,
    checkpoint_type: str,
    meta: pd.DataFrame,
    n_bootstraps: int = N_BOOTSTRAPS_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> None:
    print("\n" + "=" * 80)
    print(f"[INFO] Processing checkpoint type: {checkpoint_type}")
    print("=" * 80)

    out_dir = root_results / checkpoint_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect scan-level predictions across folds
    df_all = collect_scan_predictions_nnUNet(root_results, checkpoint_type)



    # Merge scan-level metadata (dataset-aware)
    # PatientID looks like "406_MRI_19991106_T1+C"
    # meta["PatientID_key"] is "406_MRI_19991106_T1+C" (no .nii.gz)
    df_all = df_all.merge(
        meta,
        left_on="PatientID",
        right_on="PatientID_key",
        how="left",
    )

    if "class_label" not in df_all.columns:
        raise RuntimeError(
            "Metadata is missing 'class_label' column after merge. "
            "Please check that the scan-level metadata CSV has the expected structure."
        )

    # Handle unmatched IDs (no class_label after merge)
    missing = df_all["class_label"].isna().sum()
    if missing > 0:
        print(
            f"[WARN] {missing} rows could not be matched to scan-level metadata "
            f"(class_label is NaN). These rows will be dropped before metrics."
        )
        unmatched_df = (
            df_all[df_all["class_label"].isna()][["PatientID"]]
            .drop_duplicates()
        )
        print("[WARN] Example unmatched PatientID values:")
        print(unmatched_df.head(20).to_string(index=False))

        unmatched_path = root_results / f"unmatched_ids_{checkpoint_type}.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"[WARN] Saved unmatched PatientIDs to: {unmatched_path}")

        df_all = df_all.dropna(subset=["class_label"])

    # Ensure GroundTruth column exists (match SSL3D script expectations)
    df_all["GroundTruth"] = df_all["class_label"].astype(int)




    # Per-scan error types (raw scan-level predictions)
    add_error_types(df_all, pred_col="Prediction", out_col="Error_Type")

    # Study-level majority vote (+ mean prob tiebreak)
    s = df_all.groupby("study_id").apply(majority_vote_with_prob_tiebreak)
    study_pred = s.reset_index()
    study_pred.columns = ["study_id", "StudyLevel_Prediction"]

    study_prob = (
        df_all.groupby("study_id")["Prob1"].mean().reset_index(name="StudyLevel_ProbMean")
    )
    df_all = df_all.merge(study_pred, on="study_id", how="left")
    df_all = df_all.merge(study_prob, on="study_id", how="left")

    # Debug: check for any NaNs in StudyLevel_Prediction
    mask_bad = df_all["StudyLevel_Prediction"].isna() | ~np.isfinite(
        df_all["StudyLevel_Prediction"].astype(float)
    )
    if mask_bad.any():
        print(
            "\n[DEBUG] Found rows with non-finite StudyLevel_Prediction "
            "(these would cause casting issues):"
        )
        dbg = (
            df_all.loc[mask_bad, ["study_id", "PatientID", "GroundTruth", "StudyLevel_Prediction"]]
            .drop_duplicates(subset=["study_id", "PatientID"])
        )
        print(dbg.head(50))
        print(
            "\n[DEBUG] Offending study_ids:",
            sorted(dbg["study_id"].dropna().unique().tolist()),
        )

    add_error_types(df_all, pred_col="StudyLevel_Prediction", out_col="StudyLevel_Error_Type")

    # One row per study (carry a single Fold value if present)
    mv = df_all.drop_duplicates(subset="study_id", keep="first")

    # Aggregated metrics (study level)
    y_true = mv["GroundTruth"].astype(int).values
    y_pred = mv["StudyLevel_Prediction"].astype(int).values
    probs  = mv["StudyLevel_ProbMean"].astype(float).values
    metrics_agg = compute_metrics(y_true, y_pred, probs)

    print("\n--- Aggregated Study-Level Metrics ---")
    for k in [
        "Recall(Sensitivity)", "Specificity", "AUROC", "Accuracy", "Average Precision",
        "Balanced Accuracy", "F1 Score", "TN", "FP", "FN", "TP",
    ]:
        v = metrics_agg.get(k, np.nan)
        if isinstance(v, float) and not np.isnan(v):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ---------------- Bootstrap CIs for study-level MV ---------------- #
    df_ci_mv = bootstrap_ci(
        y_true,
        y_pred,
        probs,
        base_metrics=metrics_agg,
        label=f"Aggregated Study-Level Majority Vote ({checkpoint_type})",
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    ci_path_mv = out_dir / "bootstrap_ci_mv_95.csv"
    df_ci_mv.to_csv(ci_path_mv, index=False)
    print(f"\n[INFO] Saved MV bootstrap CIs to: {ci_path_mv}")
    # ------------------------------------------------------------------ #

    # Per-fold metrics (study level)
    fold_col = "Fold" if "Fold" in mv.columns else ("fold" if "fold" in mv.columns else None)
    if fold_col is not None:
        fold_metrics: Dict[str, Dict[str, Any]] = {}

        for fold, dff in mv.groupby(fold_col):
            fm = compute_metrics(
                dff["GroundTruth"].astype(int).values,
                dff["StudyLevel_Prediction"].astype(int).values,
                dff["StudyLevel_ProbMean"].astype(float).values,
            )
            # Force column name pattern: Fold0, Fold1, ...
            try:
                fold_int = int(fold)
                col_name = f"Fold{fold_int}"
            except Exception:
                # fallback if fold is something weird
                col_name = f"Fold{fold}"
            fold_metrics[col_name] = fm

        keep = ["AUROC", "Accuracy", "Average Precision", "Balanced Accuracy", "F1 Score"]
        df_metrics = pd.DataFrame(fold_metrics).reindex(keep)

        # enforce numeric fold order: Fold0, Fold1, Fold2, ...
        fold_cols = [c for c in df_metrics.columns if c.startswith("Fold")]
        fold_cols_sorted = sorted(
            fold_cols,
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        df_metrics = df_metrics.round(4)
        if fold_cols_sorted:
            avg_vals = df_metrics[fold_cols_sorted].mean(axis=1).round(4)
            df_metrics = df_metrics[fold_cols_sorted]
            df_metrics["AvgFolds"] = avg_vals

        out_csv = out_dir / "mv_metrics_per_fold.csv"
        df_metrics.to_csv(out_csv, index_label="Metric")
        print("\n=== Per-fold (study-level) ===")
        print(df_metrics)

    # Confusion matrix (overall)
    cm_path = out_dir / "confusion_matrix_mv.png"
    save_confusion_matrix(y_true, y_pred, cm_path, title="Aggregated OOF Majority Vote")

    # ---------------- PRE-1ST-RECURRENCE SUBSET (recompute MV on subset) ---------------- #
    dfmore = df_all.copy()
    dfmore["pre1st_label"] = dfmore.apply(lambda r: assign_scan_label(r, RULES_3), axis=1)

    df_pre_slices = dfmore[dfmore["pre1st_label"] == "pre-1st-recurrence"].copy()

    print('\n[DEBUG] pre-1st slices shape:', df_pre_slices.shape)
    print('[DEBUG] pre1st_label unique:', dfmore["pre1st_label"].unique())

    if len(df_pre_slices) > 0:
        # === Recompute MV **using only pre-1st slices** ===
        s_pre = df_pre_slices.groupby("study_id").apply(majority_vote_with_prob_tiebreak)
        study_pred_pre = s_pre.reset_index()
        study_pred_pre.columns = ["study_id", "StudyLevel_Prediction_pre"]


        # === Recompute per-study mean prob **using only pre-1st slices** ===
        study_prob_pre = (
            df_pre_slices.groupby("study_id")["Prob1"]
            .mean()
            .reset_index(name="StudyLevel_ProbMean_pre")
        )

        # Determine which fold column to keep (Fold vs fold, if present)
        fold_col_pre = "Fold" if "Fold" in df_pre_slices.columns else (
            "fold" if "fold" in df_pre_slices.columns else None
        )

        # Collapse to one row per study (carry GroundTruth and optional fold)
        keep_cols = ["study_id", "GroundTruth"]
        if fold_col_pre is not None:
            keep_cols.append(fold_col_pre)

        per_study_pre = (
            df_pre_slices.drop_duplicates(subset="study_id", keep="first")[keep_cols]
            .merge(study_pred_pre, on="study_id", how="left")
            .merge(study_prob_pre, on="study_id", how="left")
        )

        # Metrics (study level, pre-1st)
        y_true_p = per_study_pre["GroundTruth"].astype(int).values
        y_pred_p = per_study_pre["StudyLevel_Prediction_pre"].astype(int).values
        probs_p  = per_study_pre["StudyLevel_ProbMean_pre"].astype(float).values

        metrics_pre = compute_metrics(y_true_p, y_pred_p, probs_p)

        print("\n--- Aggregated Metrics — pre-1st-recurrence (study level; MV from pre-1st slices) ---")
        for k in [
            "Recall(Sensitivity)", "Specificity", "AUROC", "Accuracy",
            "Average Precision", "Balanced Accuracy", "F1 Score",
            "TN", "FP", "FN", "TP",
        ]:
            v = metrics_pre.get(k, np.nan)
            if isinstance(v, float) and not np.isnan(v):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        # ------------- Bootstrap CIs for pre-1st-recurrence MV ------------- #
        df_ci_pre = bootstrap_ci(
            y_true_p,
            y_pred_p,
            probs_p,
            base_metrics=metrics_pre,
            label=f"pre-1st-recurrence (study-level MV from pre-1st slices; {checkpoint_type})",
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        ci_path_pre = out_dir / "bootstrap_ci_mv_pre1st_95.csv"
        df_ci_pre.to_csv(ci_path_pre, index=False)
        print(f"\n[INFO] Saved pre-1st-recurrence MV bootstrap CIs to: {ci_path_pre}")
        # --------------------------------------------------------------------- #

        # per-fold metrics for pre-1st subset (if we have a fold column)
        if fold_col_pre is not None and fold_col_pre in per_study_pre.columns:
            fold_metrics_pre: Dict[str, Dict[str, Any]] = {}

            for fold, dff in per_study_pre.groupby(fold_col_pre):
                fm = compute_metrics(
                    dff["GroundTruth"].astype(int).values,
                    dff["StudyLevel_Prediction_pre"].astype(int).values,
                    dff["StudyLevel_ProbMean_pre"].astype(float).values,
                )
                # Force column name pattern: Fold0, Fold1, ...
                try:
                    fold_int = int(fold)
                    col_name = f"Fold{fold_int}"
                except Exception:
                    col_name = f"Fold{fold}"
                fold_metrics_pre[col_name] = fm

            keep = ["AUROC", "Accuracy", "Average Precision", "Balanced Accuracy", "F1 Score"]
            df_metrics_pre = pd.DataFrame(fold_metrics_pre).reindex(keep)

            # enforce numeric fold order: Fold0, Fold1, Fold2, ...
            fold_cols_pre = [c for c in df_metrics_pre.columns if c.startswith("Fold")]
            fold_cols_sorted_pre = sorted(
                fold_cols_pre,
                key=lambda x: int(re.findall(r"\d+", x)[0])
            )

            df_metrics_pre = df_metrics_pre.round(4)
            if fold_cols_sorted_pre:
                avg_vals_pre = df_metrics_pre[fold_cols_sorted_pre].mean(axis=1).round(4)
                df_metrics_pre = df_metrics_pre[fold_cols_sorted_pre]
                df_metrics_pre["AvgFolds"] = avg_vals_pre

            out_csv_pre = out_dir / "df_pre1strec_metrics_per_fold.csv"
            df_metrics_pre.to_csv(out_csv_pre, index_label="Metric")
            print("\n--- Per-fold (pre-1st-recurrence; MV from pre-1st slices) ---")
            print(df_metrics_pre)

        # confusion matrix for subset
        cm_path_pre = out_dir / "confusion_matrix_mv_df_pre1strec.png"
        save_confusion_matrix(
            y_true_p, y_pred_p, cm_path_pre,
            title="Aggregated pre-1st-recurrence (mv from pre-1st slices)",
        )
    else:
        print("\n(No studies matched 'pre-1st-recurrence' subset.)")


# ------------------------------ MAIN ------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Study-level majority vote + pre-1st-recurrence MV for nnUNet CLS results."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT_RESULTS),
        help="Root nnUNet results folder (contains fold_0, fold_1, ...).",
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

    # Load scan metadata once (dataset-aware via ROOT_RESULTS path)
    meta = load_scan_metadata(root_results)
    required_meta_cols = {"new_name", "study_id", "class_label", "date"}
    if not required_meta_cols.issubset(meta.columns):
        raise RuntimeError(
            f"Scan metadata is missing required columns {required_meta_cols}. "
            f"Found: {sorted(meta.columns)}"
        )

    # Loop over chosen checkpoint result types
    for ckpt_type in CHOOSE_CHECKPOINT_RESULTS:
        run_for_checkpoint_type(
            root_results,
            ckpt_type,
            meta=meta,
            n_bootstraps=int(args.n_bootstraps),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
