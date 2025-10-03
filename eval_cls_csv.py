#!/usr/bin/env python3
import argparse
import ast
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def parse_probs(x):
    """
    Parse the 'probs' field from the predictions CSV.
    Accepts strings like "[0.1, 0.9]" or already-parsed lists/arrays.
    Returns a 1D numpy array of floats.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float).ravel()
    else:
        # string -> python list
        arr = np.asarray(ast.literal_eval(str(x)), dtype=float).ravel()
    return arr

def read_predictions(path):
    df = pd.read_csv(path)
    if 'identifier' not in df.columns:
        raise ValueError(f"'identifier' column not found in predictions CSV: {path}")
    if 'probs' not in df.columns:
        raise ValueError(f"'probs' column not found in predictions CSV: {path}")
    # Parse probs to arrays
    df['probs'] = df['probs'].apply(parse_probs)
    # If pred_class missing, compute from probs
    if 'pred_class' not in df.columns:
        df['pred_class'] = df['probs'].apply(lambda p: int(np.argmax(p)))
    return df[['identifier', 'probs', 'pred_class']]

def read_ground_truth(path):
    """
    Robustly read ground truth as two columns: identifier, label.
    Works if the CSV has no header or has header names.
    """
    # Try no-header first
    try:
        df = pd.read_csv(path, header=None, names=['identifier', 'label'])
        # ensure label numeric
        df['label'] = pd.to_numeric(df['label'])
        return df[['identifier', 'label']]
    except Exception:
        pass
    # Fallback: try with header
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    # Try to find identifier/label columns
    id_col = None
    lab_col = None
    for i, c in enumerate(cols):
        if c in ('identifier', 'id', 'case', 'subject'):
            id_col = df.columns[i]
        if c in ('label', 'gt', 'target', 'y'):
            lab_col = df.columns[i]
    if id_col is None:
        id_col = df.columns[0]
    if lab_col is None:
        lab_col = df.columns[1]
    df = df.rename(columns={id_col: 'identifier', lab_col: 'label'})
    df['label'] = pd.to_numeric(df['label'])
    return df[['identifier', 'label']]

def compute_metrics(pred_df, gt_df):
    # Drop duplicate identifiers if any, keep first occurrence
    pred_df = pred_df.drop_duplicates(subset='identifier', keep='first')
    gt_df = gt_df.drop_duplicates(subset='identifier', keep='first')

    merged = pred_df.merge(gt_df, on='identifier', how='inner')
    n_pred = len(pred_df)
    n_gt = len(gt_df)
    n_inter = len(merged)
    if n_inter == 0:
        raise ValueError("No overlapping identifiers between predictions and ground truth.")

    # Stack probabilities (shape: N x K)
    probs_list = merged['probs'].tolist()
    # Ensure all have the same length
    num_classes = max(len(p) for p in probs_list)
    probs = np.vstack([np.pad(p, (0, num_classes - len(p)), constant_values=np.nan) for p in probs_list])
    # Replace any NaNs (shouldn’t happen if probs are consistent)
    if np.isnan(probs).any():
        raise ValueError("Inconsistent probability vector lengths across rows.")

    # If the probabilities don’t sum to 1 (e.g., raw logits snuck in), normalize row-wise
    row_sums = probs.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero: if a row sums to 0, leave it as-is
    need_norm = ~np.isclose(row_sums, 1.0)
    if need_norm.any():
        with np.errstate(divide='ignore', invalid='ignore'):
            probs = np.where(row_sums > 0, probs / row_sums, probs)

    # Predictions (use provided pred_class if present; recompute otherwise)
    if 'pred_class' in merged.columns:
        y_pred = merged['pred_class'].astype(int).to_numpy()
    else:
        y_pred = np.argmax(probs, axis=1)

    y_true = merged['label'].astype(int).to_numpy()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    if num_classes == 2:
        # Binary: assume class 1 is the positive class
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
        # AUROC requires both classes present
        try:
            auroc = roc_auc_score(y_true, probs[:, 1])
        except ValueError:
            auroc = float('nan')  # not defined if only one class present
    else:
        # Multi-class
        f1 = f1_score(y_true, y_pred, average='macro')
        try:
            auroc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        except ValueError:
            auroc = float('nan')

    info = {
        "n_predictions": int(n_pred),
        "n_ground_truth": int(n_gt),
        "n_intersection": int(n_inter),
        "missing_in_pred": int(n_gt - n_inter),
        "missing_in_gt": int(n_pred - n_inter),
        "num_classes": int(num_classes),
    }

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "auroc": float(auroc) if auroc == auroc else None,  # None if NaN
    }

    return metrics, info, merged

def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions CSV against ground-truth CSV.")
    ap.add_argument("--pred", required=True, help="Path to predicted CSV (identifier, probs, pred_class).")
    ap.add_argument("--gt", required=True, help="Path to ground-truth CSV (identifier,label).")
    ap.add_argument("--out", default=None, help="Optional path to write metrics JSON.")
    ap.add_argument("--merged_csv", default=None, help="Optional path to write merged CSV.")
    args = ap.parse_args()

    pred_df = read_predictions(args.pred)
    gt_df = read_ground_truth(args.gt)

    metrics, info, merged = compute_metrics(pred_df, gt_df)

    print("\n== Evaluation Results ==")
    print(json.dumps({"metrics": metrics, "info": info}, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metrics": metrics, "info": info}, f, indent=2)

    if args.merged_csv:
        # For convenience, expand probs vector columns as p_0, p_1, ...
        maxk = max(len(p) for p in merged['probs'])
        prob_cols = [f"p_{k}" for k in range(maxk)]
        probs_mat = np.vstack(merged['probs'].tolist())
        probs_df = pd.DataFrame(probs_mat, columns=prob_cols, index=merged.index)
        out_df = pd.concat([merged.drop(columns=['probs']), probs_df], axis=1)
        out_df.to_csv(args.merged_csv, index=False)

if __name__ == "__main__":
    main()