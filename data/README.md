# Data policy

The repository intentionally does **not** include the raw otoscopic images.

Use `data/dataset_inventory.csv` and `data/archive_duplicate_groups.csv` to
inspect the supplied archive-level audit. When you have legally obtained and
extracted the data, place it below:

```text
data/raw/Otoscopic_Data/
```

The data-preparation script generates a leakage-aware manifest under
`artifacts/manifests/`.
