# Contributing

Use small, reviewable changes.

```bash
git checkout -b feature/short-description
pytest -q
ruff check .
git add .
git commit -m "feat: describe change"
```

For ML changes, report the experiment configuration and do not replace real
metrics with estimated or manually edited values.
