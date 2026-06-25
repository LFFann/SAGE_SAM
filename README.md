# SAGE-SAM

Standalone SAGE-SAM project physically copied from the local KnowSAM/a3-sam
checkout and refactored into a SAM-free inference path.

New active entrypoints:

- Training: `python train_sage_sam.py`
- Inference: `python prediction_sage_sam.py`
- CPU smoke: `python scripts/smoke_test.py`
- Independence check: `python tools/verify_standalone.py`

The copied legacy files are retained for reference and utility reuse. The active
SAGE-SAM path uses:

- `Model/sage_model.py` for `DualSegmentor`
- `Model/sam_structure_encoder.py` for optional frozen SAM image embeddings
- `sage_ssl/` for structure calibration, set supervision, graph propagation,
  relation consistency, and hardness-aware augmentation
- `engine/sage_trainer.py` for training steps
- `prediction_sage_sam.py` for inference without SAM

Quick verification:

```bash
python -m compileall Model dataloader sage_ssl engine utils tools tests train_sage_sam.py prediction_sage_sam.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
python tools/verify_standalone.py
```

V100-oriented command skeletons are in `scripts/train_v100.sh`,
`scripts/test_v100.sh`, `scripts/precompute_structure_v100.sh`, and
`scripts/run_ablations.sh`.
