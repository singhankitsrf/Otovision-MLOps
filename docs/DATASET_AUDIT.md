# Dataset Audit

## Archive-level findings

The provided archive contained **3,000 JPEG images** across five classes, 600
per class.

The RAR table-of-contents exposed CRC values that allowed a fast first-pass
content audit. That audit found:

- 2,922 unique CRC values
- 78 duplicate-content groups
- 156 files involved in repeated-content pairs
- all repeated groups were within the Acute Otitis Media directory

This means the archive is balanced by folder count but not by unique image
content.

## Training-time policy

Archive CRC values are treated only as a warning/audit signal. After extraction,
`scripts/prepare_dataset.py` computes **SHA-256** for every readable image.
SHA-256 duplicate groups determine which items are canonical for modeling.

The manifest retains duplicate metadata for auditability while only canonical
items are assigned to train/validation/test splits.

## Leakage limitations

Exact duplicate control does not prove patient-level independence. The archive
filenames do not supply trustworthy patient/exam identifiers. Therefore:

- exact duplicates are prevented from crossing splits;
- future work should use patient/exam grouping when identifiers are available;
- external-site validation remains necessary before clinical claims.

## Licensing and governance

Before using this dataset outside a private research setting, verify source,
license, redistribution rights, consent/de-identification status, and permitted
use. Do not infer redistribution rights from the presence of an archive alone.
