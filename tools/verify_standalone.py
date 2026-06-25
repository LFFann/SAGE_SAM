"""Verify that SAGE-SAM has no runtime dependency on the source a3-sam checkout."""

from __future__ import annotations

from pathlib import Path

from Model.sage_model import DualSegmentor
from engine.sage_trainer import SAGETrainer
from utils.project_checks import find_symlinks, scan_for_forbidden_runtime_links


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    findings = scan_for_forbidden_runtime_links(root)
    symlinks = find_symlinks(root)
    if findings:
        raise SystemExit(f"Forbidden runtime links found: {findings}")
    if symlinks:
        raise SystemExit(f"Symlinks found: {symlinks}")
    model = DualSegmentor(in_channels=3, num_classes=3)
    names = [name for name, _ in model.named_modules()]
    if any("Discriminator" in name or "mask_decoder" in name or "prompt_encoder" in name for name in names):
        raise SystemExit("Forbidden modules are instantiated in DualSegmentor")
    sources = model.trainable_parameter_report()
    if sources["total"] != sources["UNet"] + sources["VNet"]:
        raise SystemExit("Unexpected trainable parameter source")
    print("SAGE-SAM standalone verification passed.")


if __name__ == "__main__":
    main()
