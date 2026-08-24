from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from otovision.data import OtoscopyDataset, build_transforms
from otovision.engine import evaluate_epoch, train_epoch
from otovision.model import build_model
from otovision.utils import ensure_dir, load_yaml, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))

    manifest = pd.read_csv(cfg["manifest"])
    manifest = manifest[manifest["is_canonical"] == True].copy()
    class_names = sorted(manifest["label"].unique().tolist())
    label_to_index = {name: i for i, name in enumerate(class_names)}

    train_tf, eval_tf = build_transforms(int(cfg["image_size"]))
    train_ds = OtoscopyDataset(manifest[manifest["split"] == "train"], train_tf, label_to_index)
    val_ds = OtoscopyDataset(manifest[manifest["split"] == "val"], eval_tf, label_to_index)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), bool(cfg["pretrained"]), float(cfg["dropout"])).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]) and device.type == "cuda")

    model_dir = ensure_dir(Path(cfg["output_dir"]) / "models")
    history = []
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, bool(cfg["amp"]))
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row))

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "image_size": int(cfg["image_size"]),
                "dropout": float(cfg["dropout"]),
                "best_val_macro_f1": best_f1,
            }, model_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(cfg["patience"]):
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(Path(cfg["output_dir"]) / "training_history.csv", index=False)


if __name__ == "__main__":
    main()
