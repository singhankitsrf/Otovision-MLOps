---
title: OtoVision MLOps
emoji: 🔬
colorFrom: blue
colorTo: teal
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: Medical imaging MLOps evidence and browser QA workbench
tags:
- computer-vision
- mlops
- healthcare-ai
- pytorch
- model-evaluation
- docker
- kubernetes
- responsible-ai
- browser-inference
---

# OtoVision MLOps — End-to-End Medical Imaging Deployment

**Author:** Ankit Kumar Singh  
**Positioning:** Computer Vision • Medical AI • MLOps • Docker • Kubernetes • Responsible AI • Deployment Engineering

OtoVision is an end-to-end otoscopic medical-imaging MLOps portfolio project. The public Hugging Face Space is a free client-side workbench that performs image-quality inspection directly in the browser and documents the model-readiness, evaluation, API, container and orchestration path implemented in GitHub.

> **Evidence boundary:** A genuine five-class trained checkpoint is not bundled in the public Space. The live demo therefore does not fabricate disease predictions or publish unsupported clinical metrics.

## End-to-end flow

```text
Permitted otoscopic image
→ browser-side image QA
→ deterministic preprocessing contract
→ model-readiness / checkpoint provenance
→ genuine checkpoint inference when available
→ saved predictions + evaluation
→ FastAPI
→ Docker
→ Kubernetes
→ GitHub Actions release validation
→ Hugging Face portfolio Space
```

## Current evidence

| Capability | Status |
|---|---|
| Browser image QA | Live |
| No server-side image upload in public demo | Yes |
| FastAPI service implementation | In GitHub |
| Docker runtime | CI validated |
| Kubernetes manifests | In GitHub |
| Checkpoint provenance contract | Implemented |
| Five-class trained checkpoint | Not configured |
| Disease-performance metrics | Not claimed |
| Clinical validation | Not established |

The repository is designed to export balanced accuracy, macro/weighted precision-recall-F1, multiclass ROC-AUC, sensitivity/specificity, confusion matrix, ECE, Brier score and selective-coverage outputs from a genuine run.

## Responsible use

This is a research/engineering portfolio demonstration, not an autonomous diagnostic system. Clinical use would require approved datasets, patient-level leakage controls, external validation, clinical oversight, security/governance review, monitoring and applicable regulatory approval.

- GitHub: https://github.com/singhankitsrf/Otovision-MLOps
- Space: https://huggingface.co/spaces/singhankit491/otovision-mlops
