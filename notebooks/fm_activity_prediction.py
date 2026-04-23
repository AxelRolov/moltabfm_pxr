import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activity Track: Foundation-Model Embeddings for PXR pEC50 Prediction

    This notebook is a copy of the TabICL activity workflow, reworked to replace **Morgan count fingerprints as training descriptors** with
    **pretrained chemical foundation-model embeddings**.

    **Feature blocks used for training:**
    - RDKit 2D descriptors as a compact classical baseline
    - MACCS keys
    - Mordred 2D descriptors
    - CheMeleon descriptor-based foundation fingerprints
    - fine-tuned CheMeleon task-adapted embeddings
    - ChemBERTa SMILES embeddings
    - MoLFormer SMILES embeddings

    **What changed versus the original notebook:**
    - Morgan fingerprints are used only for diagnostics and analog-aware validation, not as model inputs
    - cross-validation is scaffold-grouped instead of purely random
    - fold reporting tracks the challenge metric stack: `RAE`, `MAE`, `R2`, `Spearman`, and `Kendall`
    - final model selection is driven by **low RAE / MAE**, not by `R2` alone
    - the ensemble threshold is fixed so no model is excluded from the consensus

    This is still a ligand-only workflow. Even with multiple descriptor families,
    this is not a true ligand-protein multimodal model.
    A real multimodal follow-up would need a learned protein or pocket representation, not a constant PXR sequence vector.
    """)
    return


@app.cell
def _():
    import hashlib
    import pickle
    import sys
    import warnings
    from pathlib import Path

    cwd = Path.cwd().resolve()
    PROJECT_ROOT = next((path for path in [cwd, *cwd.parents] if (path / "pyproject.toml").exists()), None)
    if PROJECT_ROOT is None:
        raise RuntimeError(f"Could not locate repo root from working directory: {cwd}")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import useful_rdkit_utils as uru
    from chemeleon_fingerprint import CheMeleonFingerprint, FineTunedCheMeleonEmbeddingModel
    from lightgbm import LGBMRegressor
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from scipy.stats import kendalltau, spearmanr
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from tabicl import TabICLRegressor
    from tqdm.auto import tqdm

    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This notebook needs the transformer stack from the repo environment. Run `uv sync` and launch Jupyter with `uv run jupyter lab`."
        ) from exc

    if int(transformers.__version__.split(".", 1)[0]) >= 5:
        raise RuntimeError(
            "MoLFormer remote code in this notebook expects transformers 4.x. "
            f"Current version: {transformers.__version__}. "
            "Run `uv sync` after the repo pin to install the compatible version."
        )

    if ".venv" not in Path(sys.executable).parts:
        raise RuntimeError(
            "This notebook is intended to run inside the repo's uv environment. "
            "Run `uv sync` and then start Jupyter with `uv run jupyter lab`, or use the `.venv` kernel explicitly. "
            f"Current interpreter: {sys.executable}"
        )

    sns.set_style("whitegrid")
    sns.set_context("notebook")
    pd.set_option("display.max_colwidth", 120)
    torch.set_grad_enabled(False)
    print(f"Using interpreter: {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    return (
        AutoModel,
        AutoTokenizer,
        CheMeleonFingerprint,
        Chem,
        DataStructs,
        FineTunedCheMeleonEmbeddingModel,
        GroupKFold,
        LGBMRegressor,
        MACCSkeys,
        MordredCalculator,
        MurckoScaffold,
        PCA,
        PROJECT_ROOT,
        SimpleImputer,
        StandardScaler,
        TabICLRegressor,
        hashlib,
        kendalltau,
        mean_absolute_error,
        mordred_descriptors,
        np,
        pd,
        pickle,
        plt,
        r2_score,
        rdFingerprintGenerator,
        sns,
        spearmanr,
        sys,
        torch,
        tqdm,
        uru,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Loading

    Load the PXR challenge train/test tables from Hugging Face and keep the columns that can be used for modeling or diagnostics.
    """)
    return


@app.cell
def _(pd):
    train_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
    test_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")

    print(f"Training compounds: {len(train_df)}")
    print(f"Test compounds:     {len(test_df)}")
    return test_df, train_df


@app.cell
def _(test_df, train_df):
    keep_cols = ['Molecule Name', 'SMILES', 'pEC50', 'pEC50_std.error (-log10(molarity))', 'pEC50_ci.lower (-log10(molarity))', 'pEC50_ci.upper (-log10(molarity))', 'Emax_estimate (log2FC vs. baseline)', 'Emax.vs.pos.ctrl_estimate (dimensionless)', 'Split']
    model_df = train_df[keep_cols].copy().rename(columns={'pEC50_std.error (-log10(molarity))': 'pEC50_std_error', 'pEC50_ci.lower (-log10(molarity))': 'pEC50_ci_lower', 'pEC50_ci.upper (-log10(molarity))': 'pEC50_ci_upper', 'Emax_estimate (log2FC vs. baseline)': 'Emax', 'Emax.vs.pos.ctrl_estimate (dimensionless)': 'Emax_vs_ctrl'})
    test_df_1 = test_df.rename(columns={'CXSMILES (CDD Compatible)': 'SMILES'})
    train_pec50 = model_df.dropna(subset=['pEC50']).reset_index(drop=True)
    print(f'Compounds with valid pEC50: {len(train_pec50)}')
    train_pec50.head()
    return test_df_1, train_pec50


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Foundation-Model Feature Blocks

    The core change in this notebook is descriptor generation.

    We keep classical descriptor blocks for comparison:
    - RDKit 2D descriptors
    - MACCS keys
    - Mordred 2D descriptors

    Then we add four foundation-model feature families:
    - `JacksonBurns/chemeleon` descriptor-based foundation fingerprints via the upstream `CheMeleonFingerprint` helper
    - `chemeleon_tuned`, a task-tuned CheMeleon encoder fine-tuned on the current training split and used as a separate embedding model
    - `DeepChem/ChemBERTa-77M-MTR`
    - `ibm-research/MoLFormer-XL-both-10pct`

    All feature blocks are cached under `outputs/fm_embedding_cache/` so repeated notebook runs do not recompute them.
    CheMeleon itself will also download its published checkpoint into `~/.chemprop/` on first use.
    """)
    return


@app.cell
def _(PROJECT_ROOT, torch):
    CACHE_DIR = PROJECT_ROOT / "outputs" / "fm_embedding_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    TRANSFORMER_MODEL_SPECS = {
        "chemberta": {
            "model_id": "DeepChem/ChemBERTa-77M-MTR",
            "pooling": "mean",
            "trust_remote_code": False,
        },
        "molformer": {
            "model_id": "ibm-research/MoLFormer-XL-both-10pct",
            "pooling": "pooler",
            "trust_remote_code": True,
            "revision": "7b12d946c181a37f6012b9dc3b002275de070314",
        },
    }

    USE_PCA = True
    PCA_N_COMPONENTS = 0.95
    EMBEDDING_BATCH_SIZE = 64
    CHEMELEON_BATCH_SIZE = 256
    CHEMELEON_TUNED_BATCH_SIZE = 64
    CHEMELEON_TUNED_EMBED_BATCH_SIZE = 256
    CHEMELEON_TUNED_EPOCHS = 5
    CHEMELEON_TUNED_HEAD_HIDDEN_DIM = 512
    CHEMELEON_TUNED_DROPOUT = 0.1
    CHEMELEON_TUNED_LR = 1e-4
    CHEMELEON_TUNED_WEIGHT_DECAY = 1e-5
    CHEMELEON_TUNED_FREEZE_ENCODER = False
    MAX_LENGTH = 256

    def pick_device():
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    DEVICE = pick_device()

    STATIC_FEATURE_BLOCKS = {
        "rdkit2d": ["rdkit2d"],
        "maccs": ["maccs"],
        "mordred": ["mordred"],
        "chemeleon": ["chemeleon"],
        "chemberta": ["chemberta"],
        "molformer": ["molformer"],
    }
    DYNAMIC_FEATURE_NAMES = ["chemeleon_tuned"]

    print(f"Embedding device: {DEVICE}")
    print(f"Feature cache dir: {CACHE_DIR}")
    return (
        CACHE_DIR,
        CHEMELEON_BATCH_SIZE,
        CHEMELEON_TUNED_BATCH_SIZE,
        CHEMELEON_TUNED_DROPOUT,
        CHEMELEON_TUNED_EMBED_BATCH_SIZE,
        CHEMELEON_TUNED_EPOCHS,
        CHEMELEON_TUNED_FREEZE_ENCODER,
        CHEMELEON_TUNED_HEAD_HIDDEN_DIM,
        CHEMELEON_TUNED_LR,
        CHEMELEON_TUNED_WEIGHT_DECAY,
        DEVICE,
        DYNAMIC_FEATURE_NAMES,
        EMBEDDING_BATCH_SIZE,
        STATIC_FEATURE_BLOCKS,
        MAX_LENGTH,
        PCA_N_COMPONENTS,
        TRANSFORMER_MODEL_SPECS,
        USE_PCA,
    )


@app.cell
def _(
    AutoModel,
    AutoTokenizer,
    CACHE_DIR,
    CheMeleonFingerprint,
    Chem,
    DataStructs,
    FineTunedCheMeleonEmbeddingModel,
    MACCSkeys,
    MordredCalculator,
    MurckoScaffold,
    PCA,
    SimpleImputer,
    StandardScaler,
    hashlib,
    kendalltau,
    mean_absolute_error,
    mordred_descriptors,
    np,
    pd,
    pickle,
    r2_score,
    rdFingerprintGenerator,
    spearmanr,
    torch,
    tqdm,
    uru,
):
    def canonicalize_smiles(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)


    def smiles_digest(smiles_list) -> str:
        joined = "\n".join(smiles_list)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


    def array_digest(values) -> str:
        array = np.asarray(values, dtype=np.float32)
        return hashlib.sha256(array.tobytes()).hexdigest()[:12]


    def compute_rdkit2d(smiles_list):
        rdkit_desc = uru.RDKitDescriptors()
        rows = [rdkit_desc.calc_smiles(smi) for smi in tqdm(smiles_list, desc="RDKit 2D")]
        return np.asarray(rows, dtype=np.float32)


    def compute_maccs_keys(smiles_list):
        rows = []
        for smiles in tqdm(smiles_list, desc="MACCS"):
            mol = Chem.MolFromSmiles(smiles)
            arr = np.zeros(167, dtype=np.int8)
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr)
        return np.stack(rows).astype(np.float32)


    def compute_mordred(smiles_list):
        calc = MordredCalculator(mordred_descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list, desc="Mordred (mols)")]
        df = calc.pandas(mols, quiet=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        min_non_null = int(len(df) * 0.5)
        return df.dropna(axis=1, thresh=min_non_null)


    def compute_chemeleon_embeddings(smiles_list, batch_size=256, device="cpu"):
        chemeleon = CheMeleonFingerprint(device=device)
        chunks = []
        iterator = range(0, len(smiles_list), batch_size)
        for start in tqdm(iterator, desc="CheMeleon"):
            batch = smiles_list[start : start + batch_size]
            chunks.append(chemeleon(batch).astype(np.float32))

        matrix = np.concatenate(chunks, axis=0)
        if device == "cuda":
            torch.cuda.empty_cache()
        return matrix


    def compute_tuned_chemeleon_matrices(
        train_smiles,
        train_targets,
        eval_smiles,
        *,
        device="cpu",
        epochs=5,
        train_batch_size=64,
        embed_batch_size=256,
        hidden_dim=512,
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-5,
        freeze_encoder=False,
        progress_desc="CheMeleon tuned",
    ):
        tuned_model = FineTunedCheMeleonEmbeddingModel(
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
        )
        tuned_model.fit(
            train_smiles,
            train_targets,
            epochs=epochs,
            batch_size=train_batch_size,
            lr=lr,
            weight_decay=weight_decay,
            progress_desc=progress_desc,
        )
        train_matrix = tuned_model.embed(train_smiles, batch_size=embed_batch_size)
        eval_matrix = tuned_model.embed(eval_smiles, batch_size=embed_batch_size)
        if device == "cuda":
            torch.cuda.empty_cache()
        return train_matrix, eval_matrix


    def mean_pool(last_hidden_state, attention_mask):
        weights = attention_mask.unsqueeze(-1)
        masked = last_hidden_state * weights
        denom = weights.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / denom


    def compute_transformer_embeddings(
        smiles_list,
        model_id,
        pooling="mean",
        batch_size=64,
        max_length=256,
        trust_remote_code=False,
        revision=None,
        device="cpu",
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token or tokenizer.unk_token

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if trust_remote_code:
            model_kwargs["deterministic_eval"] = True

        model = AutoModel.from_pretrained(model_id, revision=revision, **model_kwargs)
        model.to(device)
        model.eval()

        chunks = []
        iterator = range(0, len(smiles_list), batch_size)
        desc = f"Embeddings: {model_id.split('/')[-1]}"
        for start in tqdm(iterator, desc=desc):
            batch = smiles_list[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)

            pooler_output = getattr(outputs, "pooler_output", None)
            if pooling == "pooler" and pooler_output is not None:
                pooled = pooler_output
            else:
                pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))

        matrix = np.concatenate(chunks, axis=0)
        if device == "cuda":
            torch.cuda.empty_cache()
        return matrix


    def cache_or_compute_matrix(smiles_list, cache_key, build_fn):
        digest = smiles_digest(smiles_list)
        cache_file = CACHE_DIR / f"{cache_key}_{digest}.npy"
        if cache_file.exists():
            print(f"Loading cache: {cache_file.name}")
            return np.load(cache_file, allow_pickle=False)

        matrix = build_fn(smiles_list)
        np.save(cache_file, matrix)
        print(f"Saved cache: {cache_file.name}")
        return matrix


    def cache_or_compute_fingerprints(smiles_list, cache_key, build_fn):
        digest = smiles_digest(smiles_list)
        cache_file = CACHE_DIR / f"{cache_key}_{digest}.pkl"
        if cache_file.exists():
            print(f"Loading cache: {cache_file.name}")
            with cache_file.open("rb") as handle:
                return pickle.load(handle)

        fps = build_fn(smiles_list)
        with cache_file.open("wb") as handle:
            pickle.dump(fps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved cache: {cache_file.name}")
        return fps


    def cache_or_compute_tuned_chemeleon_matrices(
        train_smiles,
        train_targets,
        eval_smiles,
        *,
        cache_key_prefix,
        config_key,
        build_fn,
    ):
        train_digest = smiles_digest(train_smiles)
        target_digest = array_digest(train_targets)
        eval_digest = smiles_digest(eval_smiles)
        train_cache_file = CACHE_DIR / f"{cache_key_prefix}_{config_key}_{train_digest}_{target_digest}_train.npy"
        eval_cache_file = CACHE_DIR / f"{cache_key_prefix}_{config_key}_{train_digest}_{target_digest}_{eval_digest}_eval.npy"

        if train_cache_file.exists() and eval_cache_file.exists():
            print(f"Loading cache: {train_cache_file.name}")
            print(f"Loading cache: {eval_cache_file.name}")
            return (
                np.load(train_cache_file, allow_pickle=False),
                np.load(eval_cache_file, allow_pickle=False),
            )

        train_matrix, eval_matrix = build_fn(train_smiles, train_targets, eval_smiles)
        np.save(train_cache_file, train_matrix)
        np.save(eval_cache_file, eval_matrix)
        print(f"Saved cache: {train_cache_file.name}")
        print(f"Saved cache: {eval_cache_file.name}")
        return train_matrix, eval_matrix


    def cache_or_compute_mordred_matrices(
        train_smiles,
        test_smiles,
        train_cache_key,
        test_cache_key,
    ):
        train_digest = smiles_digest(train_smiles)
        test_digest = smiles_digest(test_smiles)
        train_cache_file = CACHE_DIR / f"{train_cache_key}_{train_digest}.npy"
        test_cache_file = CACHE_DIR / f"{test_cache_key}_{test_digest}.npy"
        columns_cache_file = CACHE_DIR / f"{train_cache_key}_{train_digest}_columns.txt"

        if train_cache_file.exists():
            print(f"Loading cache: {train_cache_file.name}")
            train_matrix = np.load(train_cache_file, allow_pickle=False)
            if columns_cache_file.exists():
                mordred_columns = columns_cache_file.read_text().splitlines()
            else:
                train_df = compute_mordred(train_smiles)
                mordred_columns = train_df.columns.tolist()
                columns_cache_file.write_text("\n".join(mordred_columns))
        else:
            train_df = compute_mordred(train_smiles)
            mordred_columns = train_df.columns.tolist()
            train_matrix = train_df.to_numpy(dtype=np.float32)
            np.save(train_cache_file, train_matrix)
            columns_cache_file.write_text("\n".join(mordred_columns))
            print(f"Saved cache: {train_cache_file.name}")

        if test_cache_file.exists():
            print(f"Loading cache: {test_cache_file.name}")
            test_matrix = np.load(test_cache_file, allow_pickle=False)
        else:
            test_df = compute_mordred(test_smiles).reindex(columns=mordred_columns)
            test_matrix = test_df.to_numpy(dtype=np.float32)
            np.save(test_cache_file, test_matrix)
            print(f"Saved cache: {test_cache_file.name}")

        return train_matrix, test_matrix, mordred_columns


    def stack_feature_blocks(block_dict, block_names):
        arrays = [block_dict[name] for name in block_names]
        return np.concatenate(arrays, axis=1)


    def sanitize_feature_matrix(matrix, clip_value=1e6):
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = matrix.copy()
        matrix[~np.isfinite(matrix)] = np.nan
        matrix = np.clip(matrix, -clip_value, clip_value)
        return matrix


    def fit_clean_feature_block(X_train, X_test, clip_value=1e6):
        X_train = sanitize_feature_matrix(X_train, clip_value=clip_value)
        X_test = sanitize_feature_matrix(X_test, clip_value=clip_value)

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=clip_value, neginf=-clip_value)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=clip_value, neginf=-clip_value)

        variance_mask = np.var(X_train, axis=0) > 0
        X_train = X_train[:, variance_mask]
        X_test = X_test[:, variance_mask]
        return X_train, X_test, variance_mask


    def fit_pca_projection(X_train, X_test, n_components, scaled_clip_value=25.0):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.asarray(X_train, dtype=np.float64))
        X_test_scaled = scaler.transform(np.asarray(X_test, dtype=np.float64))

        X_train_scaled = np.nan_to_num(
            np.clip(X_train_scaled, -scaled_clip_value, scaled_clip_value),
            nan=0.0,
            posinf=scaled_clip_value,
            neginf=-scaled_clip_value,
        )
        X_test_scaled = np.nan_to_num(
            np.clip(X_test_scaled, -scaled_clip_value, scaled_clip_value),
            nan=0.0,
            posinf=scaled_clip_value,
            neginf=-scaled_clip_value,
        )

        pca = PCA(n_components=n_components, svd_solver="full")
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        X_train_pca = np.nan_to_num(X_train_pca, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_pca = np.nan_to_num(X_test_pca, nan=0.0, posinf=0.0, neginf=0.0)
        return X_train_pca, X_test_pca, pca, scaler


    def rae(y_true, y_pred):
        denom = np.sum(np.abs(y_true - np.mean(y_true)))
        if denom == 0:
            return np.nan
        return np.sum(np.abs(y_true - y_pred)) / denom


    def evaluate_regression_metrics(y_true, y_pred):
        spearman = spearmanr(y_true, y_pred).statistic
        kendall = kendalltau(y_true, y_pred).statistic
        return {
            "RAE": float(rae(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "Spearman_R": float(np.nan_to_num(spearman, nan=0.0)),
            "Kendall_Tau": float(np.nan_to_num(kendall, nan=0.0)),
        }


    def murcko_group(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"invalid::{smiles}"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold:
            return scaffold
        return Chem.MolToSmiles(mol, canonical=True)


    def compute_similarity_fingerprints(smiles_list, radius=2, n_bits=2048):
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list, desc="Morgan similarity mols")]
        valid_idx = [idx for idx, mol in enumerate(mols) if mol is not None]
        valid_mols = [mols[idx] for idx in valid_idx]
        fps = [None] * len(smiles_list)
        if valid_mols:
            batch_fps = generator.GetFingerprints(valid_mols, numThreads=0)
            for idx, fp in zip(valid_idx, batch_fps):
                fps[idx] = fp
        return fps


    def max_train_similarity(query_fps, ref_fps):
        valid_ref = [fp for fp in ref_fps if fp is not None]
        result = []
        for fp in query_fps:
            if fp is None or not valid_ref:
                result.append(np.nan)
            else:
                result.append(max(DataStructs.BulkTanimotoSimilarity(fp, valid_ref)))
        return np.asarray(result, dtype=float)


    def outlier_aware_ensemble(predictions_array, threshold=1.0, method="median"):
        agg_fn = np.median if method == "median" else np.mean
        output = np.zeros(predictions_array.shape[1], dtype=float)
        for idx in range(predictions_array.shape[1]):
            preds = predictions_array[:, idx]
            center = np.median(preds)
            keep_mask = np.abs(preds - center) <= threshold
            if keep_mask.sum() == 0:
                output[idx] = center
            else:
                output[idx] = agg_fn(preds[keep_mask])
        return output

    return (
        cache_or_compute_fingerprints,
        cache_or_compute_matrix,
        cache_or_compute_mordred_matrices,
        cache_or_compute_tuned_chemeleon_matrices,
        canonicalize_smiles,
        compute_chemeleon_embeddings,
        compute_maccs_keys,
        compute_rdkit2d,
        compute_similarity_fingerprints,
        compute_tuned_chemeleon_matrices,
        compute_transformer_embeddings,
        evaluate_regression_metrics,
        fit_clean_feature_block,
        fit_pca_projection,
        max_train_similarity,
        murcko_group,
        outlier_aware_ensemble,
        stack_feature_blocks,
    )


@app.cell
def _(
    CHEMELEON_BATCH_SIZE,
    DEVICE,
    EMBEDDING_BATCH_SIZE,
    MAX_LENGTH,
    TRANSFORMER_MODEL_SPECS,
    cache_or_compute_matrix,
    cache_or_compute_mordred_matrices,
    canonicalize_smiles,
    compute_chemeleon_embeddings,
    compute_maccs_keys,
    compute_rdkit2d,
    compute_transformer_embeddings,
    test_df_1,
    train_pec50,
):
    train_smiles = [canonicalize_smiles(smi) for smi in train_pec50['SMILES']]
    test_smiles = [canonicalize_smiles(smi) for smi in test_df_1['SMILES']]
    train_blocks = {
        'rdkit2d': cache_or_compute_matrix(train_smiles, 'train_rdkit2d', compute_rdkit2d),
        'maccs': cache_or_compute_matrix(train_smiles, 'train_maccs', compute_maccs_keys),
        'chemeleon': cache_or_compute_matrix(
            train_smiles,
            'train_chemeleon',
            lambda smiles: compute_chemeleon_embeddings(
                smiles,
                batch_size=CHEMELEON_BATCH_SIZE,
                device=DEVICE,
            ),
        ),
    }
    test_blocks = {
        'rdkit2d': cache_or_compute_matrix(test_smiles, 'test_rdkit2d', compute_rdkit2d),
        'maccs': cache_or_compute_matrix(test_smiles, 'test_maccs', compute_maccs_keys),
        'chemeleon': cache_or_compute_matrix(
            test_smiles,
            'test_chemeleon',
            lambda smiles: compute_chemeleon_embeddings(
                smiles,
                batch_size=CHEMELEON_BATCH_SIZE,
                device=DEVICE,
            ),
        ),
    }
    train_mordred, test_mordred, _mordred_columns = cache_or_compute_mordred_matrices(
        train_smiles,
        test_smiles,
        train_cache_key='train_mordred',
        test_cache_key='test_mordred',
    )
    train_blocks['mordred'] = train_mordred
    test_blocks['mordred'] = test_mordred
    for _name, spec in TRANSFORMER_MODEL_SPECS.items():
        builder = lambda smiles, spec=spec: compute_transformer_embeddings(smiles, model_id=spec['model_id'], pooling=spec['pooling'], batch_size=EMBEDDING_BATCH_SIZE, max_length=MAX_LENGTH, trust_remote_code=spec['trust_remote_code'], revision=spec.get('revision'), device=DEVICE)
        train_blocks[_name] = cache_or_compute_matrix(train_smiles, f'train_{_name}', builder)
        test_blocks[_name] = cache_or_compute_matrix(test_smiles, f'test_{_name}', builder)
    print('Training block shapes:')
    for _name, matrix in train_blocks.items():
        print(f'  {_name:16s} {matrix.shape}')
    print('\nTest block shapes:')
    for _name, matrix in test_blocks.items():
        print(f'  {_name:16s} {matrix.shape}')
    return test_blocks, test_smiles, train_blocks, train_smiles


@app.cell
def _(
    DYNAMIC_FEATURE_NAMES,
    fit_clean_feature_block,
    np,
    pd,
    stack_feature_blocks,
    STATIC_FEATURE_BLOCKS,
    test_blocks,
    train_blocks,
):
    train_features = {}
    test_features = {}
    feature_cleanup_summary = []
    for _feature_name, block_names in STATIC_FEATURE_BLOCKS.items():
        X_train_raw = stack_feature_blocks(train_blocks, block_names)
        X_test_raw = stack_feature_blocks(test_blocks, block_names)
        n_before = X_train_raw.shape[1]
        n_non_finite_train = int((~np.isfinite(np.asarray(X_train_raw, dtype=np.float64))).sum())
        n_non_finite_test = int((~np.isfinite(np.asarray(X_test_raw, dtype=np.float64))).sum())
        _X_train, _X_test, variance_mask = fit_clean_feature_block(X_train_raw, X_test_raw)
        train_features[_feature_name] = _X_train
        test_features[_feature_name] = _X_test
        feature_cleanup_summary.append({'feature_name': _feature_name, 'n_blocks': len(block_names), 'features_before': n_before, 'features_after': int(_X_train.shape[1]), 'train_non_finite_replaced': n_non_finite_train, 'test_non_finite_replaced': n_non_finite_test})
        print(f'{_feature_name:10s} {len(block_names)} block(s): {n_before} -> {_X_train.shape[1]} features after cleanup; replaced non-finite train/test values: {n_non_finite_train}/{n_non_finite_test}')
    feature_cleanup_df = pd.DataFrame(feature_cleanup_summary)
    FEATURE_NAMES = list(STATIC_FEATURE_BLOCKS) + list(DYNAMIC_FEATURE_NAMES)
    feature_cleanup_df
    return FEATURE_NAMES, test_features, train_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Analog-Aware Cross-Validation

    The activity guide recommends validation that keeps close analogs together.
    Here we use **Bemis-Murcko scaffold groups** for fold assignment and keep Morgan/Tanimoto only for diagnostics.
    The Morgan diagnostic fingerprints are cached under `outputs/fm_embedding_cache/` so reruns do not rebuild them.
    """)
    return


@app.cell
def _(
    cache_or_compute_fingerprints,
    compute_similarity_fingerprints,
    murcko_group,
    np,
    pd,
    train_smiles,
):
    scaffold_groups = np.asarray([murcko_group(smi) for smi in train_smiles])
    unique_scaffolds = pd.Series(scaffold_groups).value_counts()
    n_splits = min(5, unique_scaffolds.shape[0])
    if n_splits < 2:
        raise ValueError("Need at least two unique scaffold groups for grouped cross-validation.")

    train_similarity_fps = cache_or_compute_fingerprints(
        train_smiles,
        "train_similarity_morgan_r2_2048",
        compute_similarity_fingerprints,
    )

    print(f"Unique scaffold groups: {unique_scaffolds.shape[0]}")
    print(f"Using GroupKFold with {n_splits} folds")
    unique_scaffolds.head(10)
    return n_splits, scaffold_groups, train_similarity_fps


@app.cell
def _(
    CHEMELEON_TUNED_BATCH_SIZE,
    CHEMELEON_TUNED_DROPOUT,
    CHEMELEON_TUNED_EMBED_BATCH_SIZE,
    CHEMELEON_TUNED_EPOCHS,
    CHEMELEON_TUNED_FREEZE_ENCODER,
    CHEMELEON_TUNED_HEAD_HIDDEN_DIM,
    CHEMELEON_TUNED_LR,
    CHEMELEON_TUNED_WEIGHT_DECAY,
    DEVICE,
    FEATURE_NAMES,
    GroupKFold,
    LGBMRegressor,
    PCA_N_COMPONENTS,
    TabICLRegressor,
    USE_PCA,
    cache_or_compute_tuned_chemeleon_matrices,
    compute_tuned_chemeleon_matrices,
    evaluate_regression_metrics,
    fit_clean_feature_block,
    fit_pca_projection,
    max_train_similarity,
    n_splits,
    np,
    pd,
    scaffold_groups,
    tqdm,
    train_features,
    train_pec50,
    train_similarity_fps,
    train_smiles,
    warnings,
):
    cv_rows = []
    fold_prediction_store = []
    splitter = GroupKFold(n_splits=n_splits)
    fits_per_fold = len(FEATURE_NAMES) * (2 if USE_PCA else 1) + 1
    total_fit_steps = n_splits * fits_per_fold
    tuned_config_key = (
        f"ep{CHEMELEON_TUNED_EPOCHS}_tb{CHEMELEON_TUNED_BATCH_SIZE}_"
        f"eb{CHEMELEON_TUNED_EMBED_BATCH_SIZE}_hd{CHEMELEON_TUNED_HEAD_HIDDEN_DIM}_"
        f"dr{str(CHEMELEON_TUNED_DROPOUT).replace('.', 'p')}_"
        f"lr{str(CHEMELEON_TUNED_LR).replace('.', 'p')}_"
        f"wd{str(CHEMELEON_TUNED_WEIGHT_DECAY).replace('.', 'p')}_"
        f"freeze{int(CHEMELEON_TUNED_FREEZE_ENCODER)}"
    )
    with tqdm(total=total_fit_steps, desc="Grouped CV fits") as cv_progress:
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(train_pec50, groups=scaffold_groups), start=1):
            y_train = train_pec50.loc[train_idx, 'pEC50'].to_numpy()
            y_test = train_pec50.loc[test_idx, 'pEC50'].to_numpy()
            fold_preds = {}
            fold_similarity = max_train_similarity(
                [train_similarity_fps[i] for i in test_idx],
                [train_similarity_fps[i] for i in train_idx],
            )
            similarity_summary = {'mean_test_max_tanimoto': float(np.nanmean(fold_similarity)), 'median_test_max_tanimoto': float(np.nanmedian(fold_similarity))}
            for _feature_name in FEATURE_NAMES:
                if _feature_name == "chemeleon_tuned":
                    fold_train_smiles = [train_smiles[i] for i in train_idx]
                    fold_test_smiles = [train_smiles[i] for i in test_idx]
                    X_tr_raw, X_te_raw = cache_or_compute_tuned_chemeleon_matrices(
                        fold_train_smiles,
                        y_train,
                        fold_test_smiles,
                        cache_key_prefix=f"cv_fold_{fold_idx}_chemeleon_tuned",
                        config_key=tuned_config_key,
                        build_fn=lambda split_train_smiles, split_train_targets, split_eval_smiles, fold_idx=fold_idx: compute_tuned_chemeleon_matrices(
                            split_train_smiles,
                            split_train_targets,
                            split_eval_smiles,
                            device=DEVICE,
                            epochs=CHEMELEON_TUNED_EPOCHS,
                            train_batch_size=CHEMELEON_TUNED_BATCH_SIZE,
                            embed_batch_size=CHEMELEON_TUNED_EMBED_BATCH_SIZE,
                            hidden_dim=CHEMELEON_TUNED_HEAD_HIDDEN_DIM,
                            dropout=CHEMELEON_TUNED_DROPOUT,
                            lr=CHEMELEON_TUNED_LR,
                            weight_decay=CHEMELEON_TUNED_WEIGHT_DECAY,
                            freeze_encoder=CHEMELEON_TUNED_FREEZE_ENCODER,
                            progress_desc=f"CheMeleon tuned fold {fold_idx}",
                        ),
                    )
                    X_tr, X_te, _variance_mask = fit_clean_feature_block(X_tr_raw, X_te_raw)
                else:
                    X_tr = train_features[_feature_name][train_idx]
                    X_te = train_features[_feature_name][test_idx]
                cv_progress.set_postfix(fold=fold_idx, model=f"TabICL-{_feature_name}")
                raw_model = TabICLRegressor()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    raw_model.fit(X_tr, y_train)
                    raw_pred = raw_model.predict(X_te)
                raw_key = _feature_name
                fold_preds[raw_key] = raw_pred
                raw_metrics = evaluate_regression_metrics(y_test, raw_pred)
                cv_rows.append({'split': fold_idx, 'family': 'TabICL', 'variant': raw_key, 'model': f'TabICL-{raw_key}', **raw_metrics, **similarity_summary})
                cv_progress.update(1)
                if USE_PCA:
                    X_tr_pca, X_te_pca, _pca, _scaler = fit_pca_projection(X_tr, X_te, n_components=PCA_N_COMPONENTS)
                    cv_progress.set_postfix(fold=fold_idx, model=f"TabICL-{_feature_name}_pca")
                    pca_model = TabICLRegressor()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        pca_model.fit(X_tr_pca, y_train)
                        pca_pred = pca_model.predict(X_te_pca)
                    pca_key = f'{_feature_name}_pca'
                    fold_preds[pca_key] = pca_pred
                    pca_metrics = evaluate_regression_metrics(y_test, pca_pred)
                    cv_rows.append({'split': fold_idx, 'family': 'TabICL', 'variant': pca_key, 'model': f'TabICL-{pca_key}', **pca_metrics, **similarity_summary})
                    cv_progress.update(1)
            cv_progress.set_postfix(fold=fold_idx, model="LightGBM-rdkit2d")
            _lgbm = LGBMRegressor(verbose=-1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _lgbm.fit(train_features['rdkit2d'][train_idx], y_train)
                lgbm_pred = _lgbm.predict(train_features['rdkit2d'][test_idx])
            lgbm_metrics = evaluate_regression_metrics(y_test, lgbm_pred)
            cv_rows.append({'split': fold_idx, 'family': 'LightGBM', 'variant': 'rdkit2d', 'model': 'LightGBM-rdkit2d', **lgbm_metrics, **similarity_summary})
            cv_progress.update(1)
            fold_prediction_store.append({'split': fold_idx, 'y_test': y_test, 'preds': fold_preds})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.head()
    return cv_df, fold_prediction_store


@app.cell
def _(cv_df):
    metric_cols = ["RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau"]
    cv_summary = (
        cv_df.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .sort_values(("RAE", "mean"))
        .round(4)
    )
    cv_summary
    return (metric_cols,)


@app.cell
def _(cv_df, plt, sns):
    _fig, _axes = plt.subplots(2, 2, figsize=(18, 10))
    plot_metrics = ['RAE', 'MAE', 'R2', 'Spearman_R']
    for _ax, metric in zip(_axes.ravel(), plot_metrics):
        order = cv_df.groupby('model')[metric].mean().sort_values(ascending=metric != 'R2').index
        sns.boxplot(data=cv_df, x='model', y=metric, order=order, ax=_ax)
        _ax.tick_params(axis='x', rotation=45)
        _ax.set_title(f'Grouped CV {metric}')
    plt.tight_layout()
    return (_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Variant Selection And Ensemble Tuning

    Choose the best raw/PCA form only for the manually selected descriptor families,
    then tune the aggregation rule while keeping the threshold effectively infinite
    so every manual ensemble member participates in the consensus.
    """)
    return


@app.cell
def _(FEATURE_NAMES):
    AVAILABLE_FEATURE_NAMES = list(FEATURE_NAMES)
    FEATURE_NAMES_MANUAL = ['rdkit2d', 'mordred', 'chemeleon', 'chemeleon_tuned']
    invalid_feature_names = sorted(set(FEATURE_NAMES_MANUAL) - set(AVAILABLE_FEATURE_NAMES))
    if invalid_feature_names:
        raise ValueError(
            "FEATURE_NAMES_MANUAL contains names that are not available in this notebook: "
            + ", ".join(invalid_feature_names)
        )
    print(f"Available feature families: {AVAILABLE_FEATURE_NAMES}")
    print(f"Manual ensemble feature families: {FEATURE_NAMES_MANUAL}")
    return (FEATURE_NAMES_MANUAL,)


@app.cell
def _(FEATURE_NAMES_MANUAL, USE_PCA, cv_df, metric_cols, pd):
    tabicl_summary = (
        cv_df[cv_df["family"] == "TabICL"]
        .groupby("variant")[metric_cols]
        .agg(["mean", "std"])
        .sort_values(("RAE", "mean"))
    )
    FINAL_ENSEMBLE_VARIANTS_MANUAL = []
    variant_rows = []
    for feature_name in FEATURE_NAMES_MANUAL:
        candidates = [feature_name]
        if USE_PCA:
            candidates.append(f"{feature_name}_pca")
        ranked_candidates = sorted(
            candidates,
            key=lambda name: (
                tabicl_summary.loc[name, ("RAE", "mean")],
                tabicl_summary.loc[name, ("MAE", "mean")],
                -tabicl_summary.loc[name, ("R2", "mean")],
            ),
        )
        best_variant = ranked_candidates[0]
        FINAL_ENSEMBLE_VARIANTS_MANUAL.append(best_variant)
        variant_rows.append(
            {
                "feature_family": feature_name,
                "selected_variant": best_variant,
                "mean_RAE": tabicl_summary.loc[best_variant, ("RAE", "mean")],
                "mean_MAE": tabicl_summary.loc[best_variant, ("MAE", "mean")],
                "mean_R2": tabicl_summary.loc[best_variant, ("R2", "mean")],
            }
        )
    manual_variant_selection_df = pd.DataFrame(variant_rows).sort_values(["mean_RAE", "mean_MAE"])
    print(f"Manual final ensemble variants: {FINAL_ENSEMBLE_VARIANTS_MANUAL}")
    manual_variant_selection_df.round(4)
    return (FINAL_ENSEMBLE_VARIANTS_MANUAL,)


@app.cell
def _(
    FEATURE_NAMES_MANUAL,
    FINAL_ENSEMBLE_VARIANTS_MANUAL,
    USE_PCA,
    evaluate_regression_metrics,
    fold_prediction_store,
    metric_cols,
    np,
    outlier_aware_ensemble,
    pd,
    tqdm,
):
    thresholds = [float("inf")]
    sweep_rows = []
    ensemble_key_sets = {'raw': FEATURE_NAMES_MANUAL, 'mixed': FINAL_ENSEMBLE_VARIANTS_MANUAL}
    if USE_PCA:
        ensemble_key_sets['pca'] = [f'{_name}_pca' for _name in FEATURE_NAMES_MANUAL]
    total_sweep_steps = len(fold_prediction_store) * len(ensemble_key_sets) * len(thresholds) * 2
    with tqdm(total=total_sweep_steps, desc="Ensemble sweep", leave=False) as sweep_progress:
        for fold_data in fold_prediction_store:
            for ensemble_name, keys in ensemble_key_sets.items():
                pred_matrix = np.stack([fold_data['preds'][key] for key in keys])
                for threshold in thresholds:
                    for method in ('median', 'mean'):
                        sweep_progress.set_postfix(split=fold_data['split'], ensemble=ensemble_name, method=method)
                        ensemble_pred = outlier_aware_ensemble(pred_matrix, threshold=threshold, method=method)
                        metrics = evaluate_regression_metrics(fold_data['y_test'], ensemble_pred)
                        sweep_rows.append({'split': fold_data['split'], 'ensemble': ensemble_name, 'threshold': threshold, 'method': method, **metrics})
                        sweep_progress.update(1)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_summary = sweep_df.groupby(['ensemble', 'threshold', 'method'])[metric_cols].mean().sort_values(['RAE', 'MAE', 'R2'], ascending=[True, True, False])
    sweep_summary.head(12)
    return (sweep_summary,)


@app.cell
def _(sweep_summary):
    best_setting = sweep_summary.reset_index().iloc[0]
    BEST_ENSEMBLE_TYPE = best_setting["ensemble"]
    OUTLIER_THRESHOLD = float(best_setting["threshold"])
    ENSEMBLE_METHOD = best_setting["method"]

    print(f"Best ensemble type: {BEST_ENSEMBLE_TYPE}")
    print(f"Outlier threshold:  {OUTLIER_THRESHOLD}")
    print(f"Aggregation:        {ENSEMBLE_METHOD}")
    return BEST_ENSEMBLE_TYPE, ENSEMBLE_METHOD, OUTLIER_THRESHOLD


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train-Test Similarity Diagnostics

    Morgan fingerprints are still useful for answering one question the activity guide cares about:
    *how local is the blinded test set relative to the training chemistry?*

    They are used below only for similarity diagnostics, not as training descriptors.
    """)
    return


@app.cell
def _(
    cache_or_compute_fingerprints,
    compute_similarity_fingerprints,
    max_train_similarity,
    pd,
    test_smiles,
    train_similarity_fps,
):
    test_similarity_fps = cache_or_compute_fingerprints(
        test_smiles,
        "test_similarity_morgan_r2_2048",
        compute_similarity_fingerprints,
    )
    test_max_similarity = max_train_similarity(test_similarity_fps, train_similarity_fps)

    similarity_df = pd.DataFrame({"max_tanimoto_to_train": test_max_similarity})
    similarity_df.describe().round(3)
    return (similarity_df,)


@app.cell
def _(plt, similarity_df, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(similarity_df['max_tanimoto_to_train'], bins=30, kde=True, ax=_ax)
    _ax.set_xlabel('Max Tanimoto similarity to training set')
    _ax.set_ylabel('Count')
    _ax.set_title('Blinded Test Set Similarity to Training Chemistry')
    plt.tight_layout()
    return (_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Final Predictions

    Refit the selected variants on the full training set, then build the final
    no-exclusion ensemble on the blinded test set.
    """)
    return


@app.cell
def _(
    BEST_ENSEMBLE_TYPE,
    CHEMELEON_TUNED_BATCH_SIZE,
    CHEMELEON_TUNED_DROPOUT,
    CHEMELEON_TUNED_EMBED_BATCH_SIZE,
    CHEMELEON_TUNED_EPOCHS,
    CHEMELEON_TUNED_FREEZE_ENCODER,
    CHEMELEON_TUNED_HEAD_HIDDEN_DIM,
    CHEMELEON_TUNED_LR,
    CHEMELEON_TUNED_WEIGHT_DECAY,
    DEVICE,
    ENSEMBLE_METHOD,
    FEATURE_NAMES_MANUAL,
    FINAL_ENSEMBLE_VARIANTS_MANUAL,
    LGBMRegressor,
    OUTLIER_THRESHOLD,
    PCA_N_COMPONENTS,
    TabICLRegressor,
    USE_PCA,
    cache_or_compute_tuned_chemeleon_matrices,
    compute_tuned_chemeleon_matrices,
    fit_clean_feature_block,
    fit_pca_projection,
    np,
    outlier_aware_ensemble,
    test_smiles,
    test_features,
    tqdm,
    train_features,
    train_pec50,
    train_smiles,
    warnings,
):
    y_train_all = train_pec50["pEC50"].to_numpy()
    final_predictions = {}
    raw_and_pca_variants = list(FEATURE_NAMES_MANUAL)
    if USE_PCA:
        raw_and_pca_variants = raw_and_pca_variants + [f"{name}_pca" for name in FEATURE_NAMES_MANUAL]
    tuned_config_key = (
        f"ep{CHEMELEON_TUNED_EPOCHS}_tb{CHEMELEON_TUNED_BATCH_SIZE}_"
        f"eb{CHEMELEON_TUNED_EMBED_BATCH_SIZE}_hd{CHEMELEON_TUNED_HEAD_HIDDEN_DIM}_"
        f"dr{str(CHEMELEON_TUNED_DROPOUT).replace('.', 'p')}_"
        f"lr{str(CHEMELEON_TUNED_LR).replace('.', 'p')}_"
        f"wd{str(CHEMELEON_TUNED_WEIGHT_DECAY).replace('.', 'p')}_"
        f"freeze{int(CHEMELEON_TUNED_FREEZE_ENCODER)}"
    )
    total_final_fit_steps = len(raw_and_pca_variants) + 1
    with tqdm(total=total_final_fit_steps, desc="Final model refits") as final_fit_progress:
        for variant in raw_and_pca_variants:
            base_name = variant.replace("_pca", "")
            if base_name == "chemeleon_tuned":
                X_train_raw, X_test_raw = cache_or_compute_tuned_chemeleon_matrices(
                    train_smiles,
                    y_train_all,
                    test_smiles,
                    cache_key_prefix="final_chemeleon_tuned",
                    config_key=tuned_config_key,
                    build_fn=lambda split_train_smiles, split_train_targets, split_eval_smiles: compute_tuned_chemeleon_matrices(
                        split_train_smiles,
                        split_train_targets,
                        split_eval_smiles,
                        device=DEVICE,
                        epochs=CHEMELEON_TUNED_EPOCHS,
                        train_batch_size=CHEMELEON_TUNED_BATCH_SIZE,
                        embed_batch_size=CHEMELEON_TUNED_EMBED_BATCH_SIZE,
                        hidden_dim=CHEMELEON_TUNED_HEAD_HIDDEN_DIM,
                        dropout=CHEMELEON_TUNED_DROPOUT,
                        lr=CHEMELEON_TUNED_LR,
                        weight_decay=CHEMELEON_TUNED_WEIGHT_DECAY,
                        freeze_encoder=CHEMELEON_TUNED_FREEZE_ENCODER,
                        progress_desc="CheMeleon tuned final",
                    ),
                )
                X_train, X_test, _variance_mask = fit_clean_feature_block(X_train_raw, X_test_raw)
            else:
                X_train = train_features[base_name]
                X_test = test_features[base_name]
            original_dim = X_train.shape[1]
            if variant.endswith('_pca'):
                X_train, X_test, _pca, _scaler = fit_pca_projection(
                    X_train,
                    X_test,
                    n_components=PCA_N_COMPONENTS,
                )
                print(f"{variant:18s} PCA dims: {original_dim} -> {X_train.shape[1]}")
            final_fit_progress.set_postfix(model=f"TabICL-{variant}")
            model = TabICLRegressor()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train_all)
                final_predictions[f"TabICL-{variant}"] = model.predict(X_test)
            preds = final_predictions[f"TabICL-{variant}"]
            print(f"TabICL-{variant:18s} range: [{preds.min():.2f}, {preds.max():.2f}]")
            final_fit_progress.update(1)
        final_fit_progress.set_postfix(model="LightGBM-rdkit2d")
        lgbm = LGBMRegressor(verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgbm.fit(train_features["rdkit2d"], y_train_all)
            final_predictions["LightGBM-rdkit2d"] = lgbm.predict(test_features["rdkit2d"])
        final_fit_progress.update(1)
    if BEST_ENSEMBLE_TYPE == "raw":
        ensemble_variant_names = [f"TabICL-{name}" for name in FEATURE_NAMES_MANUAL]
    elif BEST_ENSEMBLE_TYPE == "pca":
        ensemble_variant_names = [f"TabICL-{name}_pca" for name in FEATURE_NAMES_MANUAL]
    else:
        ensemble_variant_names = [f"TabICL-{name}" for name in FINAL_ENSEMBLE_VARIANTS_MANUAL]
    ensemble_matrix = np.stack([final_predictions[name] for name in ensemble_variant_names])
    final_predictions["TabICL-Ensemble"] = outlier_aware_ensemble(
        ensemble_matrix,
        threshold=OUTLIER_THRESHOLD,
        method=ENSEMBLE_METHOD,
    )
    SUBMISSION_MODEL = "TabICL-Ensemble"
    prediction_range = final_predictions[SUBMISSION_MODEL]
    print(f"\nSubmitting predictions from: {SUBMISSION_MODEL}")
    print(f"Ensemble members: {ensemble_variant_names}")
    print(f"Prediction range: [{prediction_range.min():.2f}, {prediction_range.max():.2f}]")
    return SUBMISSION_MODEL, ensemble_variant_names, final_predictions


@app.cell
def _(SUBMISSION_MODEL, final_predictions, test_df_1):
    submission_df = test_df_1[["SMILES", "Molecule Name"]].copy()
    submission_df["pEC50"] = final_predictions[SUBMISSION_MODEL]
    submission_df.head()
    return (submission_df,)


@app.cell
def _(final_predictions, test_df_1):
    all_model_prediction_df = test_df_1[["SMILES", "Molecule Name"]].copy()
    for model_name, pred_i in final_predictions.items():
        all_model_prediction_df[model_name] = pred_i
    all_model_prediction_df.head()
    return (all_model_prediction_df,)


@app.cell
def _(final_predictions, pd, plt, sns, train_pec50):
    plot_frames = [pd.DataFrame({'pEC50': train_pec50['pEC50'], 'source': 'Train (observed)'})]
    for _name, _preds in final_predictions.items():
        plot_frames.append(pd.DataFrame({'pEC50': _preds, 'source': f'Test ({_name})'}))
    combo_df = pd.concat(plot_frames, ignore_index=True)
    _fig, _ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=combo_df, x='source', y='pEC50', ax=_ax)
    _ax.set_ylabel('pEC$_{50}$ ($-\\log_{10}$ M)')
    _ax.set_title('Training vs Predicted Test Distributions')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    return (_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ensemble Diagnostics

    Prediction spread across the selected ensemble is a useful proxy for uncertainty on close analog series and activity-cliff candidates.
    """)
    return


@app.cell
def _(OUTLIER_THRESHOLD, ensemble_variant_names, final_predictions, np, plt):
    selected_matrix = np.stack([final_predictions[_name] for _name in ensemble_variant_names])
    pred_spread = np.std(selected_matrix, axis=0)
    pred_median = np.median(selected_matrix, axis=0)
    outlier_counts = np.sum(np.abs(selected_matrix - pred_median) > OUTLIER_THRESHOLD, axis=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    _axes[0].hist(pred_spread, bins=30)
    _axes[0].set_xlabel('Std dev across ensemble members (pEC50)')
    _axes[0].set_ylabel('Count')
    _axes[0].set_title('Prediction Spread per Molecule')
    counts, freqs = np.unique(outlier_counts, return_counts=True)
    _axes[1].bar(counts, freqs)
    _axes[1].set_xlabel('Outlier predictions removed')
    _axes[1].set_ylabel('Molecules')
    _axes[1].set_title(f'Outlier Counts (threshold={OUTLIER_THRESHOLD})')
    plt.tight_layout()
    return (_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Format And Validate Submission

    Keep the output interface exactly the same as the challenge requires.
    """)
    return


@app.cell
def _(PROJECT_ROOT, all_model_prediction_df, submission_df):
    all_predictions_file = PROJECT_ROOT / "outputs" / "my_fm_activity_all_model_predictions.csv"
    all_predictions_file.parent.mkdir(parents=True, exist_ok=True)
    all_model_prediction_df.to_csv(all_predictions_file, index=False)

    submission_file = PROJECT_ROOT / "outputs" / "my_fm_activity_submission.csv"
    submission_file.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_file, index=False)
    print(f"All-model prediction rows: {len(all_model_prediction_df)}")
    print(f"Saved full prediction table: {all_predictions_file}")
    print(f"Submission rows: {len(submission_df)}")
    submission_df.head()
    return (submission_file,)


@app.cell
def _(PROJECT_ROOT, submission_file, sys, test_df_1):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from validation.activity_validation import validate_activity_submission

    expected_activity_ids = set(test_df_1["Molecule Name"])
    is_valid, validation_errors = validate_activity_submission(
        submission_file,
        expected_ids=expected_activity_ids,
    )
    if is_valid:
        print("Activity submission file is valid.")
    else:
        print("Activity submission file is invalid:")
        for msg in validation_errors:
            print(f" - {msg}")
    return


@app.cell
def _(submission_file):
    from gradio_client import Client, handle_file
    from huggingface_hub import get_token

    ENABLE_API_SUBMISSION = True
    if ENABLE_API_SUBMISSION:
        submission_metadata = {
            "username": "axelrolov",
            "user_alias": "",
            "anon_checkbox": False,
            "participant_name": "",
            "discord_username": "",
            "email": "",
            "affiliation": "",
            "model_tag": "https://github.com/AxelRolov/moltabfm_pxr",
            "paper_checkbox": False,
            "track_select": "Activity Prediction",
        }
        missing_fields = [
            key
            for key, value in submission_metadata.items()
            if isinstance(value, str) and key != "user_alias" and not value
        ]
        #if missing_fields:
        #    raise ValueError(
        #        "Fill the submission metadata before enabling submission: "
        #        + ", ".join(sorted(missing_fields))
        #    )
        hf_token = get_token()
        client = Client("openadmet/pxr-challenge", token=hf_token)
        api_submission_result = client.predict(
            **submission_metadata,
            file_input=handle_file(submission_file),
            api_name="/submit_predictions",
        )
        api_submission_result
    else:
        api_submission_result = None
        print(
            "API submission is disabled. Set ENABLE_API_SUBMISSION = True and "
            "fill in your metadata before submitting."
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - swap in additional molecular foundation models and compare their grouped-CV `RAE`
    - evaluate whether the frozen feature families should stay fixed or move to downstream fine-tuning
    - add activity-cliff diagnostics by linking fold errors to nearest-neighbor similarity
    - if you want a true multimodal model, move beyond this tabular fusion setup and build a ligand-protein architecture with a real pocket representation
    """)
    return


if __name__ == "__main__":
    app.run()
