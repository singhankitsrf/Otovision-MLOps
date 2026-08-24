from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def expected_calibration_error(y_true, probabilities, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any(): continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier_score(y_true, probabilities, num_classes: int) -> float:
    y_true = np.asarray(y_true); probabilities = np.asarray(probabilities)
    one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def per_class_specificity(y_true, y_pred, num_classes: int):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); values=[]
    for c in range(num_classes):
        tn=np.sum((y_true != c) & (y_pred != c)); fp=np.sum((y_true != c) & (y_pred == c))
        values.append(float(tn/(tn+fp)) if (tn+fp) else float("nan"))
    return values


def summary_metrics(y_true, probabilities):
    y_true=np.asarray(y_true); probabilities=np.asarray(probabilities); y_pred=probabilities.argmax(axis=1); num_classes=probabilities.shape[1]
    metrics={"accuracy":float(accuracy_score(y_true,y_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),"macro_precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),"macro_recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),"macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),"ece":expected_calibration_error(y_true,probabilities),"brier":multiclass_brier_score(y_true,probabilities,num_classes)}
    try: metrics["roc_auc_macro_ovr"]=float(roc_auc_score(y_true,probabilities,multi_class="ovr",average="macro"))
    except ValueError: metrics["roc_auc_macro_ovr"]=float("nan")
    return metrics
