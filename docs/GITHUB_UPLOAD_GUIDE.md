# GitHub Upload Guide

## 1. Create the repository

Recommended repository name:

`otovision-mlops`

Description:

> Production-grade Healthcare AI for otoscopic image classification: leakage-aware data QA, PyTorch, uncertainty-aware triage, Grad-CAM, FastAPI, Docker, Kubernetes and CI.

Recommended topics:

`healthcare-ai`, `medical-imaging`, `pytorch`, `computer-vision`, `mlops`,
`fastapi`, `docker`, `kubernetes`, `gradcam`, `responsible-ai`, `deep-learning`

## 2. Initialize Git

From this project directory:

```bash
git init
git branch -M main
git add .
git status
```

Verify that the raw dataset, `.rar`, and model files are **not** staged.

Then:

```bash
git commit -m "feat: launch OtoVision MLOps portfolio project"
```

Create an empty repository on GitHub, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/otovision-mlops.git
git push -u origin main
```

## 3. Pin it on the profile

GitHub profile -> Customize your pins -> select this repository.

## 4. Add genuine result artifacts later

After training, consider committing small result files:

- `metrics.json`
- confusion matrix PNG
- reliability diagram PNG
- selective-coverage plot
- one or two **de-identified and redistributable** Grad-CAM examples

Do not commit a large checkpoint merely for visibility. Link to a model registry
or release asset if appropriate.

## 5. Good recruiter screenshot

The repository landing page should immediately show:

1. architecture;
2. dataset leakage audit;
3. install/test commands;
4. Docker/Kubernetes section;
5. generated, not invented, evaluation metrics;
6. responsible-AI limitations.

## 6. Resume bullet

> Built a production-oriented Healthcare AI pipeline for otoscopic image classification with leakage-aware SHA-256 data QA, PyTorch transfer learning, calibration diagnostics, uncertainty-based human-review routing, Grad-CAM, FastAPI, Docker, Kubernetes/HPA, pytest and GitHub Actions CI.
