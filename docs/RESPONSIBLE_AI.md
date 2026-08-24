# Responsible AI and Clinical Boundary

This repository deliberately separates a **software demonstration** from a
validated medical product.

## Required safeguards for any real study

- ethics and institutional approvals where applicable;
- explicit data provenance and permitted use;
- de-identification and access controls;
- patient/exam-level leakage prevention;
- external validation across devices and sites;
- subgroup and failure-mode analysis;
- calibration monitoring;
- clinician oversight and an abstention pathway;
- security, audit logging and incident response.

## What Grad-CAM does not prove

A visually plausible heatmap does not prove that the network is using the same
causal reasoning as a clinician. Explainability artifacts should be reviewed as
part of error analysis, not presented as a guarantee.

## Referral thresholds

The confidence and entropy defaults are engineering placeholders. They must be
selected using validation data and clinical workflow consequences before any
real-world use.
