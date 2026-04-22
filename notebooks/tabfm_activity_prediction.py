import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activity Track: Multi-Descriptor TabICL Ensemble with MACCS Keys

    This marimo notebook ports `tabfm_activity_prediction.ipynb` to marimo and extends
    the descriptor ensemble with MACCS keys.

    Descriptor blocks:
    - RDKit 2D descriptors
    - MACCS keys
    - Morgan count fingerprints, radius 2
    - Morgan count fingerprints, radius 3
    - Mordred 2D descriptors

    The workflow computes descriptors, compares raw and PCA variants with cross-validation,
    selects final ensemble members, writes a submission CSV, and validates the export.
    """)
    return


@app.cell
def _():
    import sys
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import useful_rdkit_utils as uru
    from lightgbm import LGBMRegressor
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from tabicl import TabICLRegressor
    from tqdm.auto import tqdm

    sns.set_style("whitegrid")
    sns.set_context("notebook")
    return (
        Chem,
        DataStructs,
        LGBMRegressor,
        MACCSkeys,
        MordredCalculator,
        PCA,
        Path,
        SimpleImputer,
        TabICLRegressor,
        mean_absolute_error,
        mordred_descriptors,
        np,
        pd,
        plt,
        r2_score,
        rdFingerprintGenerator,
        sns,
        spearmanr,
        sys,
        tqdm,
        train_test_split,
        uru,
        warnings,
    )


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = NOTEBOOK_DIR.parent

    TRAIN_DATA_URL = (
        "hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv"
    )
    TEST_DATA_URL = (
        "hf://datasets/openadmet/pxr-challenge-train-test/"
        "pxr-challenge_TEST_BLINDED.csv"
    )
    OUTPUT_FILE = PROJECT_ROOT / "outputs" / "my_tabfm_ensemble_submission.csv"

    DESCRIPTOR_TYPES = ("rdkit2d", "maccs", "morgan2", "morgan3", "mordred")
    MACCS_NUM_BITS = 167
    PCA_N_COMPONENTS = 0.95
    N_CV_SPLITS = 3
    OUTLIER_THRESHOLDS = (0.5, 0.75, 1.0, 1.5, 2.0)
    ENABLE_HF_SUBMISSION = False
    return (
        DESCRIPTOR_TYPES,
        ENABLE_HF_SUBMISSION,
        MACCS_NUM_BITS,
        N_CV_SPLITS,
        OUTLIER_THRESHOLDS,
        OUTPUT_FILE,
        PCA_N_COMPONENTS,
        PROJECT_ROOT,
        TEST_DATA_URL,
        TRAIN_DATA_URL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Loading

    Load the training and blinded test sets from Hugging Face and prepare the
    activity-track modeling tables.
    """)
    return


@app.cell
def _(TEST_DATA_URL, TRAIN_DATA_URL, pd):
    train_df = pd.read_csv(TRAIN_DATA_URL)
    test_df = pd.read_csv(TEST_DATA_URL)

    print(f"Training compounds: {len(train_df)}")
    print(f"Test compounds:     {len(test_df)}")
    return test_df, train_df


@app.cell
def _(test_df, train_df):
    keep_cols = [
        "Molecule Name",
        "SMILES",
        "pEC50",
        "pEC50_std.error (-log10(molarity))",
        "pEC50_ci.lower (-log10(molarity))",
        "pEC50_ci.upper (-log10(molarity))",
        "Emax_estimate (log2FC vs. baseline)",
        "Emax.vs.pos.ctrl_estimate (dimensionless)",
        "Split",
    ]

    model_df = train_df[keep_cols].rename(
        columns={
            "pEC50_std.error (-log10(molarity))": "pEC50_std_error",
            "pEC50_ci.lower (-log10(molarity))": "pEC50_ci_lower",
            "pEC50_ci.upper (-log10(molarity))": "pEC50_ci_upper",
            "Emax_estimate (log2FC vs. baseline)": "Emax",
            "Emax.vs.pos.ctrl_estimate (dimensionless)": "Emax_vs_ctrl",
        }
    )
    train_pec50 = model_df.dropna(subset=["pEC50"]).reset_index(drop=True)
    test_frame = test_df.rename(columns={"CXSMILES (CDD Compatible)": "SMILES"})

    print(f"Compounds with valid pEC50: {len(train_pec50)}")
    model_df.head()
    return test_frame, train_pec50


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Molecular Descriptors

    Compute five descriptor families for all compounds:

    1. RDKit 2D descriptors
    2. MACCS keys
    3. Morgan count fingerprints, radius 2
    4. Morgan count fingerprints, radius 3
    5. Mordred 2D descriptors
    """)
    return


@app.cell
def _(
    Chem,
    DataStructs,
    MACCS_NUM_BITS,
    MACCSkeys,
    MordredCalculator,
    mordred_descriptors,
    np,
    pd,
    rdFingerprintGenerator,
    tqdm,
    uru,
):
    def compute_rdkit2d(smiles_list):
        rdkit_desc = uru.RDKitDescriptors()
        return np.stack(
            [rdkit_desc.calc_smiles(smiles) for smiles in tqdm(smiles_list, desc="RDKit 2D")]
        )


    def compute_maccs_keys(smiles_list):
        fps = []
        for smiles in tqdm(smiles_list, desc="MACCS"):
            mol = Chem.MolFromSmiles(smiles)
            arr = np.zeros(MACCS_NUM_BITS, dtype=np.int8)
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        return np.stack(fps).astype(np.float64)


    def compute_morgan_counts(smiles_list, radius, n_bits=2048):
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )
        fps = []
        for smiles in tqdm(smiles_list, desc=f"Morgan r={radius}"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fps.append(np.zeros(n_bits, dtype=np.uint32))
            else:
                fps.append(generator.GetCountFingerprintAsNumPy(mol))
        return np.stack(fps).astype(np.float64)


    def compute_mordred(smiles_list):
        calc = MordredCalculator(mordred_descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list, desc="Mordred (mols)")]
        df = calc.pandas(mols, quiet=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        min_non_null = int(len(df) * 0.5)
        return df.dropna(axis=1, thresh=min_non_null)


    def compute_descriptor_block(smiles_list, mordred_columns=None):
        descriptor_map = {
            "rdkit2d": compute_rdkit2d(smiles_list),
            "maccs": compute_maccs_keys(smiles_list),
            "morgan2": compute_morgan_counts(smiles_list, radius=2),
            "morgan3": compute_morgan_counts(smiles_list, radius=3),
        }

        mordred_df = compute_mordred(smiles_list)
        if mordred_columns is None:
            mordred_columns = mordred_df.columns.tolist()
        else:
            mordred_df = mordred_df.reindex(columns=mordred_columns)

        descriptor_map["mordred"] = mordred_df.to_numpy(dtype=np.float64)
        return descriptor_map, mordred_columns

    return (compute_descriptor_block,)


@app.cell
def _(compute_descriptor_block, pd, test_frame, train_pec50):
    train_smiles = train_pec50["SMILES"].tolist()
    test_smiles = test_frame["SMILES"].tolist()

    raw_train_descriptors, mordred_columns = compute_descriptor_block(train_smiles)
    raw_test_descriptors, _ = compute_descriptor_block(test_smiles, mordred_columns=mordred_columns)

    descriptor_shape_rows = []
    for dataset_name, descriptor_map in (
        ("train", raw_train_descriptors),
        ("test", raw_test_descriptors),
    ):
        for descriptor_name, values in descriptor_map.items():
            descriptor_shape_rows.append(
                {
                    "dataset": dataset_name,
                    "descriptor": descriptor_name,
                    "rows": values.shape[0],
                    "features": values.shape[1],
                }
            )

    descriptor_shape_df = pd.DataFrame(descriptor_shape_rows)
    descriptor_shape_df
    return raw_test_descriptors, raw_train_descriptors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preprocessing

    Impute missing values with the training median and remove constant columns
    independently for each descriptor block.
    """)
    return


@app.cell
def _(
    DESCRIPTOR_TYPES,
    SimpleImputer,
    np,
    pd,
    raw_test_descriptors,
    raw_train_descriptors,
):
    test_descriptors = {}
    train_descriptors = {}
    preprocessing_rows = []

    print("Preprocessing: impute missing values and remove constant columns")
    for descriptor_name in DESCRIPTOR_TYPES:
        x_train_raw = raw_train_descriptors[descriptor_name]
        x_test_raw = raw_test_descriptors[descriptor_name]

        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x_train_raw)
        x_test = imputer.transform(x_test_raw)

        keep_mask = np.var(x_train, axis=0) > 0
        train_descriptors[descriptor_name] = x_train[:, keep_mask]
        test_descriptors[descriptor_name] = x_test[:, keep_mask]

        preprocessing_rows.append(
            {
                "descriptor": descriptor_name,
                "features_before": x_train_raw.shape[1],
                "features_after": int(keep_mask.sum()),
                "constant_removed": int(x_train_raw.shape[1] - keep_mask.sum()),
            }
        )
        print(
            f"  {descriptor_name}: {x_train_raw.shape[1]} -> {int(keep_mask.sum())} "
            f"features ({int(x_train_raw.shape[1] - keep_mask.sum())} constant removed)"
        )

    preprocessing_df = pd.DataFrame(preprocessing_rows)
    preprocessing_df
    return test_descriptors, train_descriptors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model Setup

    Train TabICL on each descriptor family in raw and PCA space, then compare
    individual models and ensemble families with cross-validation.
    """)
    return


@app.cell
def _(np):
    def outlier_aware_ensemble(predictions_array, threshold=1.0, method="median"):
        agg_fn = np.median if method == "median" else np.mean
        result = np.zeros(predictions_array.shape[1])

        for sample_idx in range(predictions_array.shape[1]):
            sample_predictions = predictions_array[:, sample_idx]
            median_prediction = np.median(sample_predictions)
            keep_mask = np.abs(sample_predictions - median_prediction) <= threshold
            if keep_mask.sum() == 0:
                result[sample_idx] = median_prediction
            else:
                result[sample_idx] = agg_fn(sample_predictions[keep_mask])

        return result

    return (outlier_aware_ensemble,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Cross-Validation Comparison

    For each descriptor block, compare raw and PCA-transformed TabICL models,
    plus a LightGBM baseline over RDKit 2D descriptors.
    """)
    return


@app.cell
def _(
    DESCRIPTOR_TYPES,
    LGBMRegressor,
    N_CV_SPLITS,
    PCA,
    PCA_N_COMPONENTS,
    TabICLRegressor,
    mean_absolute_error,
    np,
    outlier_aware_ensemble,
    pd,
    r2_score,
    spearmanr,
    tqdm,
    train_descriptors,
    train_pec50,
    train_test_split,
    warnings,
):
    cv_results = []
    cv_prediction_store = []
    per_variant_r2 = {descriptor_name: [] for descriptor_name in DESCRIPTOR_TYPES}
    per_variant_r2.update(
        {f"{descriptor_name}_pca": [] for descriptor_name in DESCRIPTOR_TYPES}
    )

    for split_idx in tqdm(range(N_CV_SPLITS), desc="CV splits"):
        train_split, test_split = train_test_split(train_pec50, random_state=split_idx)
        train_idx = train_split.index.to_numpy()
        test_idx = test_split.index.to_numpy()
        y_train = train_split["pEC50"].to_numpy()
        y_test = test_split["pEC50"].to_numpy()

        tabicl_predictions = {}

        for descriptor_name in DESCRIPTOR_TYPES:
            x_train = train_descriptors[descriptor_name][train_idx]
            x_test = train_descriptors[descriptor_name][test_idx]

            raw_model = TabICLRegressor()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_model.fit(x_train, y_train)
                raw_predictions = raw_model.predict(x_test)

            tabicl_predictions[descriptor_name] = raw_predictions
            raw_r2 = r2_score(y_test, raw_predictions)
            per_variant_r2[descriptor_name].append(raw_r2)
            cv_results.append(
                {
                    "split": split_idx,
                    "model": f"TabICL-{descriptor_name}",
                    "variant": descriptor_name,
                    "R2": raw_r2,
                    "MAE": mean_absolute_error(y_test, raw_predictions),
                    "Spearman_R": spearmanr(y_test, raw_predictions).statistic,
                }
            )

            pca = PCA(n_components=PCA_N_COMPONENTS)
            x_train_pca = pca.fit_transform(x_train)
            x_test_pca = pca.transform(x_test)

            pca_model = TabICLRegressor()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pca_model.fit(x_train_pca, y_train)
                pca_predictions = pca_model.predict(x_test_pca)

            pca_key = f"{descriptor_name}_pca"
            tabicl_predictions[pca_key] = pca_predictions
            pca_r2 = r2_score(y_test, pca_predictions)
            per_variant_r2[pca_key].append(pca_r2)
            cv_results.append(
                {
                    "split": split_idx,
                    "model": f"TabICL-{pca_key}",
                    "variant": pca_key,
                    "R2": pca_r2,
                    "MAE": mean_absolute_error(y_test, pca_predictions),
                    "Spearman_R": spearmanr(y_test, pca_predictions).statistic,
                }
            )

        baseline_model = LGBMRegressor(verbose=-1)
        baseline_model.fit(train_descriptors["rdkit2d"][train_idx], y_train)
        baseline_predictions = baseline_model.predict(train_descriptors["rdkit2d"][test_idx])
        cv_results.append(
            {
                "split": split_idx,
                "model": "LightGBM-rdkit2d",
                "variant": "rdkit2d_baseline",
                "R2": r2_score(y_test, baseline_predictions),
                "MAE": mean_absolute_error(y_test, baseline_predictions),
                "Spearman_R": spearmanr(y_test, baseline_predictions).statistic,
            }
        )

        raw_predictions = np.stack(
            [tabicl_predictions[descriptor_name] for descriptor_name in DESCRIPTOR_TYPES]
        )
        pca_predictions = np.stack(
            [tabicl_predictions[f"{descriptor_name}_pca"] for descriptor_name in DESCRIPTOR_TYPES]
        )

        mixed_keys = []
        for descriptor_name in DESCRIPTOR_TYPES:
            mean_raw = np.mean(per_variant_r2[descriptor_name])
            mean_pca = np.mean(per_variant_r2[f"{descriptor_name}_pca"])
            mixed_keys.append(
                f"{descriptor_name}_pca" if mean_pca > mean_raw else descriptor_name
            )

        mixed_predictions = np.stack([tabicl_predictions[key] for key in mixed_keys])
        cv_prediction_store.append(
            {
                "split": split_idx,
                "y_test": y_test,
                "raw_preds": raw_predictions,
                "pca_preds": pca_predictions,
                "mixed_preds": mixed_predictions,
                "mixed_keys": mixed_keys,
            }
        )

        for ensemble_name, prediction_array in (
            ("Ens-raw", raw_predictions),
            ("Ens-pca", pca_predictions),
            ("Ens-mixed", mixed_predictions),
        ):
            for ensemble_method in ("mean", "median"):
                ensemble_predictions = outlier_aware_ensemble(
                    prediction_array,
                    threshold=1.0,
                    method=ensemble_method,
                )
                cv_results.append(
                    {
                        "split": split_idx,
                        "model": f"{ensemble_name}-{ensemble_method}",
                        "variant": f"{ensemble_name}-{ensemble_method}",
                        "R2": r2_score(y_test, ensemble_predictions),
                        "MAE": mean_absolute_error(y_test, ensemble_predictions),
                        "Spearman_R": spearmanr(y_test, ensemble_predictions).statistic,
                    }
                )

    cv_df = pd.DataFrame(cv_results)
    cv_summary = (
        cv_df.groupby("model")[["R2", "MAE", "Spearman_R"]]
        .agg(["mean", "std"])
        .sort_values(("R2", "mean"), ascending=False)
        .round(3)
    )

    cv_summary
    return cv_df, cv_prediction_store, ensemble_method, per_variant_r2


@app.cell
def _(cv_df, plt, sns):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for axis, metric_name in zip(axes, ("R2", "MAE", "Spearman_R")):
        sns.boxplot(x="model", y=metric_name, data=cv_df, ax=axis)
        axis.set_title(f"Cross-validation {metric_name}")
        axis.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Outlier Threshold Sensitivity

    Sweep thresholds and aggregation methods over the stored CV predictions.
    The sweep tunes the outlier threshold and aggregation rule; final member
    selection still happens in the next section.
    """)
    return


@app.cell
def _(
    OUTLIER_THRESHOLDS,
    cv_prediction_store,
    mean_absolute_error,
    outlier_aware_ensemble,
    pd,
    r2_score,
    spearmanr,
):
    sweep_rows = []

    for entry in cv_prediction_store:
        for threshold in OUTLIER_THRESHOLDS:
            for method_name in ("mean", "median"):
                for ensemble_family, prediction_array in (
                    ("raw", entry["raw_preds"]),
                    ("pca", entry["pca_preds"]),
                    ("mixed", entry["mixed_preds"]),
                ):
                    ensemble_predictions = outlier_aware_ensemble(
                        prediction_array,
                        threshold=threshold,
                        method=method_name,
                    )
                    sweep_rows.append(
                        {
                            "split": entry["split"],
                            "ensemble": ensemble_family,
                            "threshold": threshold,
                            "method": method_name,
                            "R2": r2_score(entry["y_test"], ensemble_predictions),
                            "MAE": mean_absolute_error(entry["y_test"], ensemble_predictions),
                            "Spearman_R": spearmanr(
                                entry["y_test"],
                                ensemble_predictions,
                            ).statistic,
                        }
                    )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_summary = (
        sweep_df.groupby(["ensemble", "threshold", "method"])[["R2", "MAE", "Spearman_R"]]
        .mean()
        .round(4)
    )

    best_sweep_ensemble_type, outlier_threshold, ensemble_method = sweep_summary["R2"].idxmax()
    print(
        "Best threshold sweep result: "
        f"ensemble={best_sweep_ensemble_type}, "
        f"threshold={outlier_threshold}, "
        f"method={ensemble_method}"
    )
    sweep_summary
    return ensemble_method, outlier_threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Smart Model Selection

    Keep RDKit 2D, MACCS, and Mordred blocks, choose the stronger Morgan radius,
    then decide raw vs PCA per surviving descriptor family using mean CV R2.
    """)
    return


@app.cell
def _(np, pd, per_variant_r2):
    variant_summary_rows = []
    for variant_name, scores in per_variant_r2.items():
        variant_summary_rows.append(
            {
                "variant": variant_name,
                "mean_R2": float(np.mean(scores)),
                "std_R2": float(np.std(scores)),
            }
        )

    variant_summary_df = pd.DataFrame(variant_summary_rows).sort_values(
        "mean_R2",
        ascending=False,
    )
    variant_summary_df

    morgan2_best = max(
        np.mean(per_variant_r2["morgan2"]),
        np.mean(per_variant_r2["morgan2_pca"]),
    )
    morgan3_best = max(
        np.mean(per_variant_r2["morgan3"]),
        np.mean(per_variant_r2["morgan3_pca"]),
    )

    surviving_types = ["rdkit2d", "maccs", "mordred"]
    if morgan2_best >= morgan3_best:
        surviving_types.append("morgan2")
        print(
            f"Morgan selection: keep r=2 (R2={morgan2_best:.4f}), "
            f"drop r=3 (R2={morgan3_best:.4f})"
        )
    else:
        surviving_types.append("morgan3")
        print(
            f"Morgan selection: keep r=3 (R2={morgan3_best:.4f}), "
            f"drop r=2 (R2={morgan2_best:.4f})"
        )

    final_ensemble_variants = []
    selection_rows = []
    for descriptor_name in surviving_types:
        mean_raw = float(np.mean(per_variant_r2[descriptor_name]))
        mean_pca = float(np.mean(per_variant_r2[f"{descriptor_name}_pca"]))
        selected_variant = (
            f"{descriptor_name}_pca" if mean_pca > mean_raw else descriptor_name
        )
        selection_rows.append(
            {
                "descriptor": descriptor_name,
                "raw_R2": mean_raw,
                "pca_R2": mean_pca,
                "selected_variant": selected_variant,
            }
        )
        final_ensemble_variants.append(selected_variant)

    selection_df = pd.DataFrame(selection_rows)
    print(f"Final ensemble variants: {final_ensemble_variants}")
    selection_df
    return (final_ensemble_variants,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train-Test Similarity

    Use Morgan fingerprints to estimate how similar each blinded test molecule
    is to the training set.
    """)
    return


@app.cell
def _(DataStructs, pd, test_frame, tqdm, train_pec50, uru):
    smi2fp = uru.Smi2Fp()
    train_fp_list = [smi2fp.get_fp(smiles) for smiles in tqdm(train_pec50["SMILES"], desc="Train similarity fingerprints")]
    test_fp_list = [smi2fp.get_fp(smiles) for smiles in tqdm(test_frame["SMILES"], desc="Test similarity fingerprints")]

    max_similarity = [
        max(DataStructs.BulkTanimotoSimilarity(test_fp, train_fp_list))
        for test_fp in test_fp_list
    ]
    similarity_df = pd.DataFrame({"max_tanimoto_similarity": max_similarity})
    similarity_df.describe()
    return (max_similarity,)


@app.cell
def _(max_similarity, plt, sns):
    fig, axis = plt.subplots(figsize=(8, 4))
    sns.histplot(max_similarity, bins=30, kde=True, ax=axis)
    axis.set_xlabel("Max Tanimoto similarity to training set")
    axis.set_ylabel("Count")
    axis.set_title("Test set similarity to training set")
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Final Predictions

    Train the selected TabICL variants on all labeled compounds and combine
    them with the tuned outlier-aware ensemble rule.
    """)
    return


@app.cell
def _(
    LGBMRegressor,
    PCA,
    PCA_N_COMPONENTS,
    TabICLRegressor,
    ensemble_method,
    final_ensemble_variants,
    np,
    outlier_aware_ensemble,
    outlier_threshold,
    pd,
    test_descriptors,
    train_descriptors,
    train_pec50,
    warnings,
):
    y_train_all = train_pec50["pEC50"].to_numpy()
    final_predictions = {}

    for variant_name in final_ensemble_variants:
        if variant_name.endswith("_pca"):
            base_name = variant_name.removesuffix("_pca")
            pca = PCA(n_components=PCA_N_COMPONENTS)
            x_train = pca.fit_transform(train_descriptors[base_name])
            x_test = pca.transform(test_descriptors[base_name])
            print(
                f"{variant_name}: {train_descriptors[base_name].shape[1]} -> "
                f"{x_train.shape[1]} PCA components"
            )
        else:
            x_train = train_descriptors[variant_name]
            x_test = test_descriptors[variant_name]

        model = TabICLRegressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train_all)
            final_predictions[f"TabICL-{variant_name}"] = model.predict(x_test)

        prediction_range = final_predictions[f"TabICL-{variant_name}"]
        print(
            f"TabICL-{variant_name}: "
            f"[{prediction_range.min():.2f}, {prediction_range.max():.2f}]"
        )

    baseline_model = LGBMRegressor(verbose=-1)
    baseline_model.fit(train_descriptors["rdkit2d"], y_train_all)
    final_predictions["LightGBM-rdkit2d"] = baseline_model.predict(
        test_descriptors["rdkit2d"]
    )

    all_tabicl_final = np.stack(
        [final_predictions[f"TabICL-{variant_name}"] for variant_name in final_ensemble_variants]
    )
    final_predictions["TabICL-Ensemble"] = outlier_aware_ensemble(
        all_tabicl_final,
        threshold=outlier_threshold,
        method=ensemble_method,
    )

    prediction_ranges_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "min_pEC50": float(predictions.min()),
                "max_pEC50": float(predictions.max()),
            }
            for model_name, predictions in final_predictions.items()
        ]
    ).sort_values("model")
    prediction_ranges_df
    return all_tabicl_final, final_predictions


@app.cell
def _(final_predictions, test_frame):
    submission_model = "TabICL-Ensemble"
    submission_df = test_frame[["SMILES", "Molecule Name"]].copy()
    submission_df["pEC50"] = final_predictions[submission_model]

    print(f"Submitting predictions from: {submission_model}")
    submission_df.head()
    return (submission_df,)


@app.cell
def _(final_predictions, pd, plt, sns, train_pec50):
    plot_frames = [
        pd.DataFrame({"pEC50": train_pec50["pEC50"], "source": "Train (actual)"})
    ]
    for model_name, predictions in final_predictions.items():
        plot_frames.append(
            pd.DataFrame(
                {"pEC50": predictions, "source": f"Test ({model_name})"}
            )
        )

    distribution_df = pd.concat(plot_frames, ignore_index=True)
    fig, axis = plt.subplots(figsize=(14, 5))
    sns.boxplot(x="source", y="pEC50", data=distribution_df, ax=axis)
    axis.set_ylabel(r"pEC$_{50}$ ($-\log_{10}$ M)")
    axis.set_title("Training vs predicted test distributions")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ensemble Diagnostics

    Inspect the per-molecule prediction spread and outlier counts across the
    selected final ensemble members.
    """)
    return


@app.cell
def _(all_tabicl_final, final_ensemble_variants, np, outlier_threshold, plt):
    n_ensemble_models = len(final_ensemble_variants)
    prediction_spread = np.std(all_tabicl_final, axis=0)
    median_predictions = np.median(all_tabicl_final, axis=0)
    outlier_counts = np.sum(
        np.abs(all_tabicl_final - median_predictions) > outlier_threshold,
        axis=0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(prediction_spread, bins=30)
    axes[0].set_xlabel(f"Std dev across {n_ensemble_models} models (pEC50)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction spread per molecule")

    counts, values = np.unique(outlier_counts, return_counts=True)
    axes[1].bar(counts, values)
    axes[1].set_xlabel("Number of outlier predictions")
    axes[1].set_ylabel("Molecules")
    axes[1].set_title(f"Outlier counts (threshold={outlier_threshold})")
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Format and Validate Submission

    Export the final ensemble predictions and validate the resulting activity file.
    """)
    return


@app.cell
def _(OUTPUT_FILE, submission_df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Submission rows: {len(submission_df)}")
    print(f"Wrote: {OUTPUT_FILE}")
    return


@app.cell
def _(OUTPUT_FILE, PROJECT_ROOT, sys, test_frame):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from validation.activity_validation import validate_activity_submission

    expected_activity_ids = set(test_frame["Molecule Name"])
    is_valid, validation_errors = validate_activity_submission(
        OUTPUT_FILE,
        expected_ids=expected_activity_ids,
    )

    if is_valid:
        print("Activity submission file is valid.")
    else:
        print("Activity submission file is invalid:")
        for message in validation_errors:
            print(f" - {message}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional Challenge Submission

    By default this marimo notebook does not submit to the Hugging Face challenge app.
    Set `ENABLE_HF_SUBMISSION = True` in the config cell and fill the metadata in the
    submission cell before enabling real submissions.
    """)
    return


@app.cell
def _(ENABLE_HF_SUBMISSION, OUTPUT_FILE):
    if not ENABLE_HF_SUBMISSION:
        submission_result = None
        print(
            "Hugging Face submission is disabled. "
            f"Submission file ready at: {OUTPUT_FILE}"
        )
    else:
        from gradio_client import Client, handle_file
        from huggingface_hub import get_token

        submission_metadata = {
            "username": "",
            "user_alias": "",
            "anon_checkbox": False,
            "participant_name": "",
            "discord_username": "",
            "email": "",
            "affiliation": "",
            "model_tag": "",
            "paper_checkbox": False,
            "track_select": "Activity Prediction",
        }

        missing_fields = [
            key
            for key, value in submission_metadata.items()
            if isinstance(value, str) and key != "user_alias" and not value
        ]
        if missing_fields:
            raise ValueError(
                "Fill the submission metadata before enabling submission: "
                + ", ".join(sorted(missing_fields))
            )

        hf_token = get_token()
        client = Client("openadmet/pxr-challenge", token=hf_token)
        submission_result = client.predict(
            **submission_metadata,
            file_input=handle_file(OUTPUT_FILE),
            api_name="/submit_predictions",
        )
        submission_result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - Add atom-pair fingerprints or additional graph descriptors
    - Swap random CV for scaffold-aware splits
    - Incorporate assay covariates such as Emax and counter-screen data
    - Explore weighted or uncertainty-aware ensembling
    """)
    return


if __name__ == "__main__":
    app.run()
