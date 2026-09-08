from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .model import build_model
from .preprocessing import crop_dark_border
from .triage import triage_decision


def load_checkpoint(path: str | Path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, class_names, int(checkpoint.get("image_size", 224)), device


@torch.inference_mode()
def predict_image(
    model,
    image: Image.Image,
    class_names,
    image_size: int,
    device,
    min_confidence: float = 0.75,
    max_normalized_entropy: float = 0.65,
):
    _, eval_tf = build_transforms(image_size)
    image = crop_dark_border(image.convert("RGB"))
    x = eval_tf(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    triage = triage_decision(
        probs, min_confidence=min_confidence, max_normalized_entropy=max_normalized_entropy
    )
    return {
        "predicted_class": class_names[idx],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
        **triage,
    }
