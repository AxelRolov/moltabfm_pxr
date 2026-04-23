"""Local CheMeleon fingerprint helper.

Adapted from the upstream ``chemeleon_fingerprint.py`` in
https://github.com/JacksonBurns/chemeleon (MIT-licensed), which the upstream
README explicitly suggests copying for downstream use.

The helper downloads the published CheMeleon message-passing checkpoint on
first use and exposes a small callable wrapper that returns learned molecular
fingerprints for SMILES strings or RDKit ``Mol`` objects.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles
from tqdm.auto import trange

CHEMELEON_CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_CHECKPOINT_NAME = "chemeleon_mp.pt"


def _build_mpnn(
    message_passing: nn.BondMessagePassing,
    device: str | torch.device | None = None,
) -> MPNN:
    model = MPNN(
        message_passing=message_passing,
        agg=nn.MeanAggregation(),
        predictor=RegressionFFN(input_dim=message_passing.output_dim),
    )
    model.eval()
    if device is not None:
        model.to(device=device)
    return model


def _load_chemeleon_model(
    device: str | torch.device | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[featurizers.SimpleMoleculeMolGraphFeaturizer, MPNN]:
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    ckpt_dir = checkpoint_dir or Path.home() / ".chemprop"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ckpt_dir / CHEMELEON_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        urlretrieve(CHEMELEON_CHECKPOINT_URL, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    message_passing = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
    message_passing.load_state_dict(checkpoint["state_dict"])

    model = _build_mpnn(message_passing=message_passing, device=device)
    return featurizer, model


def _build_chemprop_model(
    device: str | torch.device | None = None,
    **message_passing_kwargs,
) -> tuple[featurizers.SimpleMoleculeMolGraphFeaturizer, MPNN]:
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    message_passing = nn.BondMessagePassing(**message_passing_kwargs)
    model = _build_mpnn(message_passing=message_passing, device=device)

    return featurizer, model


def _to_mols(molecules: list[str | Mol]) -> list[Mol]:
    mols = [MolFromSmiles(m) if isinstance(m, str) else m for m in molecules]
    invalid_indices = [idx for idx, mol in enumerate(mols) if mol is None]
    if invalid_indices:
        raise ValueError(
            "CheMeleon received invalid molecules at indices: "
            + ", ".join(map(str, invalid_indices[:10]))
        )
    return mols


class CheMeleonFingerprint:
    """Generate CheMeleon learned fingerprints for molecules."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.featurizer, self.model = _load_chemeleon_model(
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

    @property
    def device(self) -> torch.device:
        """Return the current torch device for the underlying model."""

        return next(self.model.parameters()).device

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        """Return CheMeleon fingerprints for molecules."""

        mols = _to_mols(molecules)
        batch = BatchMolGraph([self.featurizer(mol) for mol in mols])
        batch.to(device=self.device)

        with torch.no_grad():
            fingerprints = self.model.fingerprint(batch)

        return fingerprints.detach().cpu().numpy()


class FineTunedCheMeleonEmbeddingModel:
    """Task-tuned CheMeleon encoder that emits adapted molecular embeddings."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        checkpoint_dir: Path | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        self.featurizer, self.model = _load_chemeleon_model(
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.model.message_passing.output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _iter_batches(self, mols: list[Mol], batch_size: int):
        for start in range(0, len(mols), batch_size):
            yield mols[start : start + batch_size]

    def _fingerprint_batch(self, mols: list[Mol]) -> torch.Tensor:
        batch = BatchMolGraph([self.featurizer(mol) for mol in mols])
        batch.to(device=self.device)
        return self.model.fingerprint(batch)

    def fit(
        self,
        molecules: list[str | Mol],
        targets: np.ndarray | list[float],
        *,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        show_progress: bool = True,
        progress_desc: str = "CheMeleon fine-tune",
    ) -> list[float]:
        mols = _to_mols(molecules)
        y = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self.device)
        if len(mols) != y.shape[0]:
            raise ValueError("Number of molecules and targets must match for CheMeleon fine-tuning.")

        parameters = list(self.head.parameters())
        if not self.freeze_encoder:
            parameters.extend(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

        history: list[float] = []
        self.model.train()
        self.head.train()
        epoch_iterator = trange(
            epochs,
            desc=progress_desc,
            leave=False,
            disable=not show_progress,
        )
        with torch.enable_grad():
            for _epoch in epoch_iterator:
                permutation = torch.randperm(len(mols))
                running_loss = 0.0
                total_seen = 0
                for start in range(0, len(mols), batch_size):
                    batch_indices = permutation[start : start + batch_size]
                    batch_mols = [mols[int(idx)] for idx in batch_indices]
                    batch_targets = y[batch_indices]

                    embeddings = self._fingerprint_batch(batch_mols)
                    preds = self.head(embeddings).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(preds, batch_targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_size_actual = batch_targets.shape[0]
                    running_loss += float(loss.detach()) * batch_size_actual
                    total_seen += batch_size_actual

                mean_loss = running_loss / max(total_seen, 1)
                history.append(mean_loss)
                if show_progress:
                    epoch_iterator.set_postfix(loss=f"{mean_loss:.4f}")

        self.model.eval()
        self.head.eval()
        return history

    def embed(
        self,
        molecules: list[str | Mol],
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        mols = _to_mols(molecules)
        self.model.eval()
        chunks = []
        with torch.inference_mode():
            for batch_mols in self._iter_batches(mols, batch_size=batch_size):
                chunks.append(self._fingerprint_batch(batch_mols).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.model.message_passing.output_dim), dtype=np.float32)

    def predict(
        self,
        molecules: list[str | Mol],
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        mols = _to_mols(molecules)
        self.model.eval()
        self.head.eval()
        chunks = []
        with torch.inference_mode():
            for batch_mols in self._iter_batches(mols, batch_size=batch_size):
                embeddings = self._fingerprint_batch(batch_mols)
                chunks.append(self.head(embeddings).squeeze(-1).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)


class FineTunedChempropEmbeddingModel(FineTunedCheMeleonEmbeddingModel):
    """Task-trained Chemprop encoder that emits learned molecular embeddings."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        **message_passing_kwargs,
    ) -> None:
        self.featurizer, self.model = _build_chemprop_model(
            device=device,
            **message_passing_kwargs,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.model.message_passing.output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
