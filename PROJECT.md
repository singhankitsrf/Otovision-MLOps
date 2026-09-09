# OtoVision MLOps — Project Charter

**Owner:** Ankit Kumar Singh  
**Portfolio role:** AI Lead / Senior ML Engineer / Healthcare AI Engineer  
**Project type:** Medical-computer-vision MLOps reference implementation

## Product objective

Demonstrate a leakage-aware, reproducible otoscopic-image ML lifecycle from data integrity and training through evaluation, explainability, API serving, containerization, Kubernetes deployment patterns, and responsible human-review routing.

## Current delivered scope

- 3,000-image / five-class dataset specification and integrity workflow
- SHA-256 exact-duplicate control before model splitting
- deterministic train/validation/test workflow
- EfficientNet-B0 training pipeline
- multiclass evaluation and calibration diagnostics
- confidence/entropy-based review routing
- Grad-CAM explainability
- FastAPI inference contract
- Docker + Kubernetes/HPA deployment assets
- pytest/Ruff/GitHub Actions CI
- Hugging Face portfolio workbench with explicit checkpoint/evidence boundary

## Delivery roadmap

### Phase 1 — MLOps foundation — COMPLETE
- [x] data-integrity and leakage controls
- [x] reusable training/evaluation package
- [x] calibration and selective-review design
- [x] explainability path
- [x] API + Docker + Kubernetes assets
- [x] CI and responsible-use documentation

### Phase 2 — Executed model evidence — NEXT
- [ ] Train with the genuine five-class image corpus
- [ ] Commit reproducible evaluation metadata without committing restricted images
- [ ] Publish accuracy, balanced accuracy, macro-F1, ROC-AUC, ECE and Brier only after execution
- [ ] Generate confusion matrix, calibration and selective-coverage artifacts
- [ ] Register checkpoint hash and dataset/split provenance

### Phase 3 — Deployment validation
- [ ] Run containerized inference with the genuine checkpoint
- [ ] Exercise Kubernetes deployment and autoscaling in a controlled environment
- [ ] Measure latency, throughput, memory and failure behavior
- [ ] Add external validation only when an approved dataset is available

## Success criteria

1. No exact duplicate crosses evaluation boundaries.
2. Every published metric is tied to dataset/split/checkpoint provenance.
3. Uncertain predictions route to human review.
4. No clinical-validation or medical-device claim is made without evidence.

## Risks and controls

| Risk | Control |
|---|---|
| Data leakage | SHA-256 deduplication before split |
| Overconfident model output | calibration + entropy/confidence routing |
| Unsupported clinical claim | portfolio/research boundary in docs and UI |
| Reproducibility drift | deterministic config + CI + artifact provenance |

## Recruiter signal

Healthcare AI · Computer Vision · PyTorch · MLOps · FastAPI · Docker · Kubernetes · Explainability · Calibration · Responsible AI
