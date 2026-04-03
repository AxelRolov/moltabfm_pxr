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

CHEMELEON_CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_CHECKPOINT_NAME = "chemeleon_mp.pt"


class CheMeleonFingerprint:
    """Generate CheMeleon learned fingerprints for molecules."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()

        ckpt_dir = checkpoint_dir or Path.home() / ".chemprop"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = ckpt_dir / CHEMELEON_CHECKPOINT_NAME
        if not checkpoint_path.exists():
            urlretrieve(CHEMELEON_CHECKPOINT_URL, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        message_passing = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
        message_passing.load_state_dict(checkpoint["state_dict"])

        self.model = MPNN(
            message_passing=message_passing,
            agg=agg,
            predictor=RegressionFFN(input_dim=message_passing.output_dim),
        )
        self.model.eval()

        if device is not None:
            self.model.to(device=device)

    @property
    def device(self) -> torch.device:
        """Return the current torch device for the underlying model."""

        return next(self.model.parameters()).device

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        """Return CheMeleon fingerprints for molecules."""

        mols = [MolFromSmiles(m) if isinstance(m, str) else m for m in molecules]
        batch = BatchMolGraph([self.featurizer(mol) for mol in mols])
        batch.to(device=self.device)

        with torch.no_grad():
            fingerprints = self.model.fingerprint(batch)

        return fingerprints.detach().cpu().numpy()
