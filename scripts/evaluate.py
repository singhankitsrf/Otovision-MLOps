from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from otovision.data import OtoscopyDataset, build_transforms
from otovision.inference import load_checkpoint
from otovision.metrics import per_class_specificity, summary_metrics
from otovision.triage import normalized_entropy
from otovision.utils import ensure_dir, load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    model, class_names, image_size, device = load_checkpoint(args.checkpoint)
    manifest = pd.read_csv(cfg["manifest"])
    _, eval_tf = build_transforms(image_size)
    test_df = manifest[(manifest["is_canonical"] == True) & (manifest["split"] == "test")]
    ds = OtoscopyDataset(test_df, eval_tf)
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    labels, probs = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(y.numpy().tolist())
    probabilities = np.concatenate(probs, axis=0)
    y_true = np.asarray(labels)
    y_pred = probabilities.argmax(axis=1)

    out_dir = ensure_dir(Path(cfg["output_dir"]) / "evaluation")
    metrics = summary_metrics(y_true, probabilities)
    metrics["per_class_specificity"] = dict(zip(class_names, per_class_specificity(y_true, y_pred, len(class_names))))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "classification_report.txt").write_text(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    confidence = probabilities.max(axis=1)
    correctness = (y_pred == y_true).astype(int)
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correctness[mask].mean()))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    plt.plot(xs, ys, marker="o", label="model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Observed accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_diagram.png", dpi=200)
    plt.close()

    entropies = np.asarray([normalized_entropy(p) for p in probabilities])
    rows = []
    for threshold in np.linspace(0.0, 1.0, 101):
        accept = entropies <= threshold
        if accept.any():
            rows.append({"entropy_threshold": threshold, "coverage": float(accept.mean()), "accuracy": float(correctness[accept].mean())})
    pd.DataFrame(rows).to_csv(out_dir / "selective_coverage.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
