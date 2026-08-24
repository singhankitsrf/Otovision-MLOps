# Architecture

```mermaid
flowchart TD
    A[Raw otoscopic archive] --> B[Archive audit]
    B --> C[Extracted image QA]
    C --> D[SHA-256 exact duplicate removal]
    D --> E[Stratified train/val/test manifest]
    E --> F[Augmentation + preprocessing]
    F --> G[EfficientNet-B0 transfer learning]
    G --> H[Validation macro-F1 selection]
    H --> I[Test evaluation]
    I --> J[Calibration diagnostics]
    I --> K[Grad-CAM]
    J --> L[Confidence + entropy review gate]
    L --> M[FastAPI]
    M --> N[Docker]
    N --> O[Kubernetes]
```

## Design decisions

### Exact-duplicate handling

The archive has repeated content. Splitting before duplicate removal creates a
high risk of train/test leakage, so SHA-256 grouping precedes the split.

### Backbone

EfficientNet-B0 is used as a deployable baseline. It is not presented as a
novel architecture; the project value is the full lifecycle around it.

### Referral logic

Maximum softmax confidence alone is not treated as a guarantee of correctness.
The serving layer also exposes normalized entropy and can route uncertain
outputs for human review.

### Deployment

The API is stateless except for a read-only model checkpoint, so it can scale
horizontally behind a service. The Kubernetes example uses readiness/liveness
probes and an HPA.
