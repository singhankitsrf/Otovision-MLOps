from __future__ import annotations

import argparse
from pathlib import Path

import torch

from otovision.inference import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/models/model.ts")
    args = parser.parse_args()

    model, _, image_size, device = load_checkpoint(args.checkpoint)
    example = torch.randn(1, 3, image_size, image_size, device=device)
    traced = torch.jit.trace(model, example)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
