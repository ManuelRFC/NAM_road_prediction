"""
src/evaluation/metrics.py
Authoritative metric computation module for road upgrade classification.
Calculates Accuracy, Precision, Recall, F1, Macro-F1, PR-AUC, ROC-AUC, and Confusion Matrices.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes standard performance metrics for road upgrade detection:
    - Positive Class (1): Post_Construction (Paved / Upgraded road)
    - Negative Class (0): Pre_Construction (Unpaved / Control earth track)
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "true_negative_rate": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }

    if y_prob is not None:
        try:
            # PR-AUC (Average Precision / Area under Precision-Recall Curve)
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
            metrics["pr_auc"] = float(auc(recalls, precisions))
        except Exception:
            metrics["pr_auc"] = 0.0

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics

def generate_text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generates a text summary using scikit-learn's classification_report."""
    target_names = ["Pre_Construction", "Post_Construction"]
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)
