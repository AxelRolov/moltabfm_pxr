# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tutorial and evaluation toolkit for the **OpenADMET PXR Blind Challenge** — a community benchmark for predicting PXR (Pregnane X Receptor) induction. Two submission tracks:

- **Activity Track**: Predict pEC50 values for 513 molecules (CSV with SMILES, Molecule Name, pEC50)
- **Structure Track**: Predict protein-ligand complex structures for 78 compounds (ZIP of PDB files with ligand residue named "LIG")

Training/test data: [HuggingFace `openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)

## Environment Setup

```bash
uv sync                  # install all dependencies
uv run jupyter lab       # launch JupyterLab
```

Legacy conda setup is also available via `environment.yaml`.

Key dependencies: rdkit, lightgbm, scikit-learn, MDAnalysis, biotite, useful_rdkit_utils (from GitHub), gradio-client, tabicl.

## Running Tests

Notebooks are the test suite — executed via pytest-nbmake:

```bash
uv run --group test pytest -n=auto --nbmake --nbmake-timeout=1200 --disable-warnings notebooks/
```

There are no traditional unit tests. CI runs this on macOS and Ubuntu with Python 3.12.

## Architecture

### `evaluation/` — Scoring pipeline

- `config.py`: Metric definitions (activity: MAE, RAE, R2, Spearman R, Kendall's Tau; structure: LDDT-PLI, BiSyRMSD, LDDT-LP), constants like `BOOTSTRAP_SAMPLES=1000` and `BISYRMSD_NAN_PENALTY=20.0`
- `utils.py`: `bootstrap_sampling()` with fixed seed (0) for reproducibility, `clip_and_log_transform()`
- `evaluate_predictions.py`: `score_activity_predictions()` merges predictions with ground truth and runs bootstrap evaluation; `score_structure_predictions()` and `score_single_structure()` use OST (OpenStructure) library for structural scoring — OST is **not** in `environment.yaml` (separate install)

### `validation/` — Submission validators

- `activity_validation.py`: `validate_activity_submission()` — checks CSV format, required columns, 513-row count, duplicates, numeric values
- `structure_validation.py`: `validate_structure_submission()` — checks ZIP of 78 PDB files, verifies exactly one "LIG" residue per file, max 2 chains, validates ligand connectivity against expected SMILES using RDKit bond-order template matching

Both validators return `(bool, list[str])` — success flag and list of error messages.

### `notebooks/` — Tutorial notebooks (also serve as integration tests)

- `activity_prediction.ipynb`: End-to-end activity workflow (load from HF, EDA, descriptor calculation, LightGBM training, submission formatting)
- `structure_prediction.ipynb`: Structure track workflow

### `inputs/` — Reference protein data

PXR protein sequence (FASTA) and structure (CIF/YAML) for x01378-1.

### `outputs/` — Example submissions

`example_structure_submission/` contains 78 PDB files; `example_submission/` has a single PDB; `my_activity_submission.csv` is an example activity CSV.

## Key Domain Constraints

- Bootstrap sampling uses a **fixed seed** (`BOOTSTRAP_SEED=0`) so all submissions are evaluated on identical resamples
- Structure scoring depends on OST (`ost.mol.alg.ligand_scoring`) which is not included in the tutorial conda env
- Failed structure scores receive worst-case penalties (LDDT-PLI/LDDT-LP → 0.0, BiSyRMSD → 20.0 Å) rather than being excluded
- Ligand residue in PDB files **must** be named "LIG" — this is enforced in both validation and scoring
- Activity dataset: 513 molecules; Structure dataset: 78 complexes — these counts are hardcoded in validation
