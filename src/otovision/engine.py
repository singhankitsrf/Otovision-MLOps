from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool):
    model.train()
    running_loss = 0.0
    labels, predictions = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": running_loss / max(len(loader.dataset), 1),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


@torch.inference_mode()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    labels, predictions = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": running_loss / max(len(loader.dataset), 1),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }
