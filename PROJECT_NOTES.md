# Project Notes

## Current status

The repository scaffold, data-audit metadata, training/evaluation code,
explainability, API, Docker, Kubernetes manifests, tests, CI, and documentation
are complete.

The trained checkpoint and final numerical evaluation are intentionally absent
until the full extracted dataset is available to the runtime and an actual
training run is completed.

## Before presenting model performance publicly

Run:

```bash
python scripts/prepare_dataset.py --data-root data/raw/Otoscopic_Data
python scripts/train.py --config configs/train.yaml
python scripts/evaluate.py --config configs/train.yaml
```

Then review the generated outputs for data leakage, implausible metrics, and
class-specific failure modes before committing selected figures.
