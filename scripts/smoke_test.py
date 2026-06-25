"""End-to-end CPU smoke test for SAGE-SAM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Model.sage_model import DualSegmentor
from Model.sam_structure_encoder import SAMStructureEncoder
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.evaluator import inference_summary, predict_logits
from engine.sage_trainer import SAGETrainer
from sage_ssl.structure_cache import load_structure_cache, save_structure_cache, tensor_hash
from tools.make_synthetic_dataset import make_dataset
from utils.seed import seed_everything


def run(with_real_sam: bool = False, sam_checkpoint: str | None = None) -> None:
    seed_everything(2026)
    root = Path("outputs/smoke")
    make_dataset(root / "synthetic_dataset")
    device = "cpu"
    model = DualSegmentor(in_channels=3, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SAGETrainer(model, optimizer, num_classes=3)

    labeled_image = torch.rand(2, 3, 64, 64)
    labeled_mask = torch.randint(0, 3, (2, 64, 64))
    encoder = SAMStructureEncoder(checkpoint=sam_checkpoint if with_real_sam else None)
    structure = encoder(labeled_image, output_size=(16, 16))
    cache_path = save_structure_cache(root / "structure_cache", "batch0", structure, tensor_hash(labeled_image))
    structure = load_structure_cache(root / "structure_cache", "batch0", expected_shape=tuple(structure.shape))

    for _ in range(2):
        trainer.step(labeled=(labeled_image, labeled_mask))
    for _ in range(2):
        trainer.step(unlabeled=(labeled_image, structure), q=torch.tensor([0.5, 0.5, 0.5]))

    ckpt = save_checkpoint(root / "checkpoints" / "smoke.pt", model, optimizer, step=4, calibration={"q": [0.5, 0.5, 0.5]})
    load_checkpoint(ckpt, model, optimizer, strict=False)
    logits = predict_logits(model, labeled_image[:1])
    summary = inference_summary()
    assert logits.shape == (1, 3, 64, 64)
    assert summary["inference_uses_sam"] is False
    assert cache_path.exists()
    print("SAGE-SAM standalone smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-real-sam", action="store_true")
    parser.add_argument("--sam-checkpoint")
    args = parser.parse_args()
    run(with_real_sam=args.with_real_sam, sam_checkpoint=args.sam_checkpoint)


if __name__ == "__main__":
    main()
