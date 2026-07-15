"""Stage 1 pretraining config (implementation-plan.md Sec 7, "Stage 1 -
Pre-training"), mirroring dataset_processing/dataloading/config.py's
dataclass + YAML-loader pattern."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML


@dataclasses.dataclass
class PretrainConfig:
    seed: int
    # TokenFace's own fine-tuning LR (1e-4), not SMIRK's from-scratch 5e-4 - we're
    # fine-tuning a FaRL-pretrained transformer (SViT), not training a ResNet
    # encoder from scratch, so a lower LR to avoid wrecking the pretrained
    # features is the better anchor to borrow here.
    learning_rate: float
    # Epoch-based rather than SMIRK's own "60k iterations" reference schedule:
    # that iteration count was calibrated against SMIRK's own dataset composition
    # and batch sizes, which don't transfer numerically to ours. TokenFace's own
    # reference (10 epochs, batch_size=8) doesn't translate directly either - our
    # dataloader.yaml uses much larger batch sizes (128 for 2d_image/3d_image),
    # so the same epoch count means a very different total gradient-update budget.
    # Starting guess, not a derived/calibrated value - revisit empirically.
    num_epochs: int
    log_interval_steps: int
    checkpoint_interval_epochs: int
    device: str
    checkpoint_dir: str
    dataloader_config_path: str
    datasets_yaml_path: str = str(DEFAULT_DATASETS_YAML)


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return PretrainConfig(**raw)
