# Model Card — OtoVision-MLOps

## Model details

- Task: five-class otoscopic image classification
- Candidate labels: Acute Otitis Media, Cerumen Impaction, Chronic Otitis Media,
  Myringosclerosis, Normal
- Default backbone: EfficientNet-B0 transfer learning
- Framework: PyTorch / torchvision

## Intended use

Research and engineering demonstration of medical-image classification and
MLOps practices.

## Out-of-scope use

- autonomous diagnosis
- treatment recommendations
- replacing ENT examination
- emergency or acute-care decisions

## Data

The source archive reports 3,000 images, but archive-level content inspection
identified repeated content. The pipeline performs training-time SHA-256
exact-duplicate removal before splitting.

No patient-level grouping is possible unless reliable patient or encounter IDs
are supplied.

## Performance

Populate this section only from generated evaluation outputs after training.
Do not manually insert an accuracy value.

Suggested table:

| Metric | Value |
|---|---:|
| Accuracy | generated |
| Balanced accuracy | generated |
| Macro F1 | generated |
| Macro ROC-AUC | generated |
| ECE | generated |
| Brier score | generated |

## Limitations

- exact duplicate control does not guarantee patient independence;
- class labels may be affected by source/acquisition bias;
- external device/site validation is absent unless separately performed;
- softmax confidence can be miscalibrated;
- Grad-CAM is an explanatory aid, not proof of causal reasoning.

## Human oversight

Low-confidence or high-entropy predictions are designed to be routed to human
review. Thresholds require validation for the intended setting.
