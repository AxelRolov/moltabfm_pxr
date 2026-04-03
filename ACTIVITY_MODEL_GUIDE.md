# Activity Model Development Guide

## Scope
This guide is for the **activity track only**. Treat it as the working brief for future model development in this repository.

The task is to predict **pEC50** for the blinded **513-compound analog set** from the OpenADMET PXR Blind Challenge. In this repo, the baseline workflow lives in [`notebooks/activity_prediction.ipynb`](./notebooks/activity_prediction.ipynb), scoring logic lives in [`evaluation/`](./evaluation), and submission checks live in [`validation/activity_validation.py`](./validation/activity_validation.py).

## What Matters in This Challenge
- **Primary metric:** `RAE` (Relative Absolute Error).
- **Secondary metrics:** `MAE`, `R2`, `Spearman R`, `Kendall's Tau`.
- **Submission format:** exactly `SMILES`, `Molecule Name`, `pEC50` for **513 molecules**.

This is not a generic QSAR benchmark. The test set was built as an **analog expansion** around selective actives, so the hard part is not broad chemical coverage. The hard part is modeling **local SAR**, **selectivity-driven signal**, and **activity cliffs**.

## Dataset Facts That Should Shape Modeling
- The broader program screened **11,362 compounds** at single concentration, then moved **4,325** into 8-point dose-response.
- The training set used here contains about **4,140 valid pEC50 values**.
- The blinded 513-compound test set was built from **63 selective hits** using **ECFP4 Tanimoto > 0.4** analog expansion.
- Hit expansion favored compounds with **EC50 <= 1 uM** (`pEC50 >= 6`) and at least **1.5 log units** separation from the PXR-null counter-screen.

Implications:
- Expect the leaderboard to reward **fine-grained ranking within analog series**, not just global trend capture.
- Random CV alone will overestimate performance if close analogs leak across folds.
- Counter-screen and efficacy data are not noise; they encode **on-target specificity** and should be treated as potentially useful auxiliary signal.
- `pEC50` is already on a log scale, so do **not** apply another target transform.

## Recommended Modeling Principles
1. Start from strong, reproducible tabular baselines before chasing novelty. In this repo that means the LightGBM notebook first, then more advanced descriptor or tabular FM variants.
2. Optimize for **RAE and MAE**, not only `R2`. A model with flashy correlation but poor calibration can still rank badly.
3. Use **analog-aware validation**. Prefer scaffold, cluster, or nearest-neighbor grouped splits over pure random splits.
4. Track **local error**. For each validation fold, inspect nearest-neighbor distance, chemotype coverage, and residuals around `pEC50 ~ 6-8`, where selective actives are likely concentrated.
5. Use **ensemble thinking**. The test set is chemically local; consensus across descriptor families or model classes is likely safer than a single brittle winner.
6. Preserve experimental uncertainty. Features such as `Emax`, null-line readouts, confidence intervals, or standard errors may help as inputs, sample weights, or multi-task targets.

## Practical Workflow in This Repository
```bash
uv sync --group test
uv run jupyter lab
uv run pytest -n=auto --nbmake --nbmake-timeout=1200 --maxfail=0 --disable-warnings notebooks/
```

Working loop:
1. Prototype in [`notebooks/activity_prediction.ipynb`](./notebooks/activity_prediction.ipynb) or a new notebook beside it.
2. Save candidate submissions under [`outputs/`](./outputs).
3. Validate the CSV before comparing models.

Example validation:
```python
from pathlib import Path
from validation.activity_validation import validate_activity_submission

ok, errors = validate_activity_submission(Path("outputs/my_activity_submission.csv"))
print(ok, errors)
```

## Minimum Standard Before Calling a Model "Better"
- Report fold-level `RAE`, `MAE`, `R2`, `Spearman`, and `Kendall`.
- Show that gains survive **analog-aware CV**, not only random splits.
- Compare prediction distributions against the training-set `pEC50` range.
- Check for unstable behavior on close analogs and obvious activity cliffs.
- Record the exact feature set, split strategy, seed, and output file used.

## Submission Mindset
Prefer a model that is **stable, calibrated, and chemically local** over one that wins a noisy random split. The challenge design mirrors lead optimization, so future work in this repo should prioritize **SAR sensitivity**, **selectivity awareness**, and **defensible validation**.
