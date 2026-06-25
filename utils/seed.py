"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
