# Data policy

The repository intentionally does **not** include the raw otoscopic images or bulky generated audit tables.

When you have legally obtained and extracted the data, place it below:

```text
data/raw/Otoscopic_Data/
```

Run the data-preparation workflow to regenerate the leakage-aware inventory, SHA-256 duplicate audit, and train/validation/test manifest locally under `artifacts/manifests/`.

This keeps the public repository lean while preserving the exact engineering workflow used to reproduce the metadata.
