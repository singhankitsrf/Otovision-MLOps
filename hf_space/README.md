---
title: OtoVision MLOps
emoji: 🔬
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Otoscopic MLOps workbench with safe checkpoint inference
tags:
- computer-vision
- mlops
- healthcare-ai
- pytorch
- model-evaluation
- docker
- kubernetes
- responsible-ai
- gradio
---

# OtoVision MLOps — End-to-End Medical Imaging Deployment

**Author:** Ankit Kumar Singh  
**Positioning:** Computer Vision • Medical AI • MLOps • Docker • Kubernetes • Responsible AI • Deployment Engineering  
**Live runtime:** Hugging Face Docker Space  
**Source repository:** https://github.com/singhankitsrf/Otovision-MLOps

OtoVision is a recruiter-facing **end-to-end medical-imaging MLOps project** for an otoscopic image classification workflow. It demonstrates how image ingestion, quality inspection, model loading, checkpoint provenance, inference readiness, evaluation artifacts, containerization, Kubernetes deployment assets, and CI/CD can be connected into one auditable engineering system.

> **Evidence boundary:** The five-class training dataset and a genuine trained checkpoint are not bundled in this repository/Space. The live Space therefore performs image-quality inspection and exposes checkpoint-backed inference only when a reviewed model is configured. It does **not** generate random predictions, reuse manuscript numbers, or claim clinical validation.

## What this project demonstrates

- otoscopic image ingestion with file-size and image-quality controls
- deterministic preprocessing shared between training and inference
- EfficientNet-family transfer-learning pipeline in the source repository
- explicit model readiness instead of silent fallback predictions
- optional immutable Hugging Face checkpoint loading
- checkpoint/source provenance returned with inference output
- saved prediction/evaluation artifact interfaces
- calibration-aware evaluation design, including ECE and Brier score
- uncertainty-aware referral/triage engineering interfaces
- FastAPI service implementation in the GitHub project
- Docker packaging and health/readiness behavior
- Kubernetes deployment manifests
- GitHub Actions validation before Hugging Face publication
- responsible-use and de-identified-input boundaries

## End-to-end system flow

```text
Permitted otoscopic image
        ↓
Input validation + image-quality inspection
        ↓
Deterministic preprocessing
        ↓
Model readiness check
   ├─ no reviewed checkpoint → QA-only evidence mode
   └─ reviewed checkpoint → classifier inference
        ↓
Class probabilities + prediction
        ↓
Checkpoint/source provenance
        ↓
Saved evaluation outputs from genuine runs
        ↓
FastAPI / Docker runtime
        ↓
Kubernetes deployment assets
        ↓
GitHub Actions validation
        ↓
Hugging Face Docker Space
```

## Current live evidence status

| Evidence | Status |
|---|---|
| Image-quality workbench | Live |
| Docker Space runtime | Implemented |
| Checkpoint provenance | Implemented |
| Random/fabricated fallback predictions | Disabled |
| Five-class checkpoint in this Space | Not configured |
| Published disease-classification accuracy | Not claimed |
| Clinical validation | Not established |
| GitHub Docker/Kubernetes deployment assets | Available |

The live application reads the repository's `evaluation/status.json`, which intentionally marks model evaluation as pending until the original five-class dataset and a genuine trained checkpoint are available.

## Reproducibility and provenance design

A genuine model release is expected to carry:

- deterministic split provenance
- preprocessing configuration
- checkpoint SHA/hash or immutable Hub revision
- saved prediction arrays
- label mapping
- evaluation metrics generated from those predictions
- runtime/library versions
- source Git revision

This prevents a polished demo from being mistaken for evidence that was never executed.

## Evaluation contract

The repository is structured to report more than accuracy alone. A completed five-class run can export:

- accuracy and balanced accuracy
- macro/weighted precision, recall, and F1
- multiclass ROC-AUC
- sensitivity and specificity
- confusion matrix
- calibration/reliability outputs
- expected calibration error (ECE)
- Brier score
- selective-coverage / confidence behavior

No numeric disease-performance values are shown here until they are backed by a real checkpoint and saved predictions.

## Deployment architecture

The public Space is packaged from the same tracked deployment source that CI validates. The Docker image starts the Gradio research workbench on port 7860 and fails safely when model requirements are not met.

The GitHub repository additionally contains conventional service/deployment assets for **FastAPI, Docker, and Kubernetes**, allowing the project to demonstrate both an accessible portfolio runtime and a production-style service architecture.

### Runtime safety boundary

- use only de-identified, permitted images
- the application is a research/portfolio demonstration
- no autonomous diagnosis or treatment recommendation
- uploaded files may be temporarily cached by Gradio and are scheduled for cleanup
- no patient database is maintained by the application
- a private checkpoint should use a read-scoped secret and immutable model revision

## CI/CD release controls

```text
Pull request / release change
        ↓
Stage only tracked Space files
        ↓
Build exact Docker image
        ↓
Start container
        ↓
HTTP smoke test
        ↓
Publish validated bundle to Hugging Face
```

## Engineering decisions

**Why fail when the model is unavailable?**  
A production-oriented inference service should expose model readiness truthfully. Returning plausible-looking random probabilities would create false evidence.

**Why separate QA mode from model mode?**  
It keeps the live demonstration useful while clearly distinguishing executable image-processing engineering from disease-classification performance that has not yet been reproduced in this environment.

**Why checkpoint provenance?**  
The same UI can otherwise hide model drift. Returning the configured model source/revision makes the runtime auditable.

## Responsible-use statement

OtoVision is a **medical-imaging software engineering portfolio demonstration**. Clinical use would require approved datasets, patient-level split controls, external validation, clinical oversight, security/governance review, monitoring, and applicable regulatory/organizational approval.

## Explore the implementation

- **GitHub source:** https://github.com/singhankitsrf/Otovision-MLOps
- **Evaluation status:** https://github.com/singhankitsrf/Otovision-MLOps/blob/main/evaluation/status.json
- **Hugging Face Space:** https://huggingface.co/spaces/singhankit491/otovision-mlops

### Portfolio signal

This project demonstrates **AI Lead / Senior ML Engineering** thinking across computer vision, model-readiness controls, reproducibility, deployment architecture, CI/CD, container orchestration, provenance, responsible AI, and evidence-backed technical communication.