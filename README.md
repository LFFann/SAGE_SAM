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

## Real Training

Expected dataset layout:

```text
SampleData/260513_data_multiclass/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
  test/image
  test/mask
```

Masks must contain class indices `0..num_classes-1`; mask resizing uses nearest
neighbor. Calibration is split only from `labeled/image` and `labeled/mask`;
`val` and `test` labels are never used for calibration.

Precompute structure cache:

```bash
python tools/precompute_sam_structure.py \
  --data_path ./SampleData \
  --dataset 260513_data_multiclass \
  --output_cache ./structure_cache/260513_data_multiclass \
  --sam_checkpoint ./sam_vit_b_01ec64.pth \
  --device cuda
```

Dry-run:

```bash
python train_sage_sam.py \
  --data_path ./SampleData \
  --dataset 260513_data_multiclass \
  --structure_cache ./structure_cache/260513_data_multiclass \
  --output_dir ./outputs \
  --experiment_name SAGE_DryRun \
  --device cpu \
  --dry_run
```

Formal training:

```bash
bash scripts/train_v100.sh
```

Resume:

```bash
python train_sage_sam.py \
  --data_path ./SampleData \
  --dataset 260513_data_multiclass \
  --structure_cache ./structure_cache/260513_data_multiclass \
  --output_dir ./outputs \
  --experiment_name SAGE_SAM_3Class \
  --resume ./outputs/SAGE_SAM_3Class/checkpoints/latest.pth \
  --device cuda \
  --amp
```

Initialize from a compatible UNet/VNet checkpoint without restoring progress:

```bash
python train_sage_sam.py \
  --data_path ./SampleData \
  --dataset 260513_data_multiclass \
  --structure_cache ./structure_cache/260513_data_multiclass \
  --output_dir ./outputs \
  --experiment_name SAGE_SAM_Init \
  --init_checkpoint ./checkpoints/knowsam_unet_vnet.pth \
  --device cuda \
  --amp
```

Checkpoint files:

- `checkpoints/latest.pth`: periodic training state.
- `checkpoints/best_fused_dice.pth`: best validation `fused_avg_dice`.
- `checkpoints/final.pth`: final training state.

Smoke:

```bash
python train_sage_sam.py --smoke --device cpu
```

Windows PowerShell dry-run:

```powershell
python train_sage_sam.py `
  --data_path "F:/postgraduate/KnowSAM/SAGE_SAM/SampleData" `
  --dataset "260513_data_multiclass" `
  --structure_cache "F:/postgraduate/KnowSAM/SAGE_SAM/structure_cache/260513_data_multiclass" `
  --output_dir "F:/postgraduate/KnowSAM/SAGE_SAM/outputs" `
  --experiment_name "SAGE_DryRun" `
  --device cpu `
  --dry_run
```

Windows PowerShell training:

```powershell
python train_sage_sam.py `
  --data_path "F:/postgraduate/KnowSAM/SAGE_SAM/SampleData" `
  --dataset "260513_data_multiclass" `
  --structure_cache "F:/postgraduate/KnowSAM/SAGE_SAM/structure_cache/260513_data_multiclass" `
  --structure_cache_mode required `
  --output_dir "F:/postgraduate/KnowSAM/SAGE_SAM/outputs" `
  --experiment_name "SAGE_SAM_3Class" `
  --device cuda `
  --num_classes 3 `
  --max_iterations 10000 `
  --warmup_iterations 1000 `
  --amp
```
