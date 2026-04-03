# PXR Challenge Model Lab

> A compact experiment repo for testing **tabular foundation models** on **PXR pEC50 prediction**.

[![Python](https://img.shields.io/badge/Python-3.11%2B-0b3d91)](./pyproject.toml)
[![uv](https://img.shields.io/badge/env-uv-5c7cfa)](https://github.com/astral-sh/uv)
[![Task](https://img.shields.io/badge/task-tabular%20bioactivity-0f766e)](./notebooks/tabfm_activity_prediction.ipynb)
[![Baseline](https://img.shields.io/badge/baseline-LightGBM-365314)](./notebooks/activity_prediction.ipynb)

## What This Repo Is

This repository is not meant to be a general overview of the OpenADMET PXR challenge. It is a focused workspace for answering one practical question:

**How far can tabular foundation models go on the PXR activity task, and what descriptor/ensemble setup works best?**

It is based on the OpenADMET tutorial repository: [`OpenADMET/PXR-Challenge-Tutorial`](https://github.com/OpenADMET/PXR-Challenge-Tutorial).

The current emphasis is on fast, reproducible experiments around:

- `TabICL` as the main tabular foundation model
- classical descriptor baselines such as `LightGBM`
- multi-descriptor training across RDKit, Morgan, and Mordred feature spaces
- PCA vs raw feature views
- simple, robust ensemble rules for final submission generation

## Model Stack

| Component | Role |
| --- | --- |
| `TabICLRegressor` | Main tabular foundation model used for activity prediction |
| `LightGBM` | Strong non-foundation baseline for comparison |
| `RDKit 2D` | Compact physicochemical descriptor block |
| `Morgan r=2 / r=3` | Count-fingerprint views for local SAR signal |
| `Mordred 2D` | Wide descriptor space for higher-capacity tabular models |
| PCA variants | Dimensionality reduction experiments per descriptor family |
| Outlier-aware ensemble | Median-based aggregation across selected model variants |

## Where To Look

| Path | Purpose |
| --- | --- |
| [`notebooks/tabfm_activity_prediction.ipynb`](./notebooks/tabfm_activity_prediction.ipynb) | Main notebook for TabICL, descriptor sweeps, PCA variants, and ensemble selection |
| [`notebooks/activity_prediction.ipynb`](./notebooks/activity_prediction.ipynb) | Simpler RDKit + LightGBM baseline workflow |
| [`ACTIVITY_MODEL_GUIDE.md`](./ACTIVITY_MODEL_GUIDE.md) | Modeling principles and validation mindset for future iterations |
| [`validation/activity_validation.py`](./validation/activity_validation.py) | Submission-format checks for the 513-compound CSV |
| [`evaluation/`](./evaluation) | Metric code used to score activity predictions |
| [`outputs/`](./outputs) | Saved submissions and example prediction files |

## Quick Start

```bash
uv sync --group test
uv run jupyter lab
uv run pytest -n=auto --nbmake --nbmake-timeout=1200 --maxfail=0 --disable-warnings notebooks/
```

## Current Direction

The interesting part here is not challenge boilerplate. It is the model behavior:

- which descriptor family gives TabICL the cleanest signal
- when PCA helps and when it destroys useful structure
- whether a foundation model actually beats a tuned gradient-boosted baseline
- how stable ensemble predictions are on close analog series
- how much prediction spread remains across descriptor-specific models

## Data

Training and test tables come from the Hugging Face dataset:
[`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)

If you want challenge logistics, dates, or broader background, use the official OpenADMET materials. This repo is for the model experiments.
