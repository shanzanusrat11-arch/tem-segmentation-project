# Model & Report Part I — EMPS supervised segmentation

Run `just` from the repository root with Podman and Just installed. This builds the container, runs the existing ETL pipeline, exports EMPS train/eval tensors, checks their integrity, trains a compact U-Net for three epochs, compares against an all-background dummy and Otsu, and generates `artifacts/results.json`, `artifacts/loss.png`, `artifacts/predictions.png`, and `report.pdf` automatically.

## Scope

The full ETL retains EMPS, HRTEM demo, and Co3O4 database records, but only EMPS is used for **supervised segmentation** because its image/mask pairs are verified. The existing 465 EMPS records are split between train and held-out evaluation. Training uses fold 0 for validation and the other four training folds for fitting. The held-out evaluation is used only after training. The three-epoch model is a reproducibility demonstration, not a tuned model.

## Requirements

- `podman` (running machine) and `just`
- Internet access for the dataset downloads and first container build
- Enough free disk space for the original ETL datasets, including Co3O4

The container uses CPU PyTorch for portability, so GPU hardware is not required. The same Python training script detects Apple MPS or CUDA when run in a compatible host environment. First container build may take a while due to PyTorch. Generated results are ignored by Git; the grader recreates them with `just`.
