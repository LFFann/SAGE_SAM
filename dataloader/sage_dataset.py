"""SAGE-SAM dataset facade around the copied KnowSAM dataset implementation."""

from dataloader.dataset import build_Dataset

SAGEDataset = build_Dataset

__all__ = ["SAGEDataset", "build_Dataset"]
