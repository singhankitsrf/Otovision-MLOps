from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from otovision.data import build_transforms
from otovision.explainability import GradCAM, overlay_heatmap
from otovision.inference import load_checkpoint
from otovision.preprocessing import crop_dark_border


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model, class_names, image_size, device = load_checkpoint(args.checkpoint)
    image = crop_dark_border(Image.open(args.image).convert("RGB"))
    _, tf = build_transforms(image_size)
    x = tf(image).unsqueeze(0).to(device)

    cam = GradCAM(model, model.features[-1])
    heatmap, pred = cam(x)
    cam.close()

    rgb = np.asarray(image.resize((image_size, image_size))).astype(float) / 255.0
    overlay = overlay_heatmap(rgb, heatmap)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.output, overlay)
    print(f"Predicted: {class_names[pred]}")


if __name__ == "__main__":
    main()
