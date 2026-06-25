# SAGE-SAM Notice

SAGE-SAM was created as a physical-copy derivative of the local KnowSAM/a3-sam
checkout whose remote is `https://github.com/LFFann/a3-sam.git`.

The copied baseline includes code derived from:

- a3-sam / KnowSAM: UNet, VNet, dataset utilities, metrics, checkpoint and visualization helpers.
- SSL4MIS-style semi-supervised segmentation utilities in the copied baseline.
- Segment Anything: local `Model/sam` and image-encoder code retained only for optional structure precomputation.

SAGE-SAM's active training and inference path is implemented in:

- `Model/sage_model.py`
- `Model/sam_structure_encoder.py`
- `sage_ssl/`
- `engine/sage_trainer.py`
- `train_sage_sam.py`
- `prediction_sage_sam.py`

The active SAGE-SAM path does not instantiate HAM/Discriminator, Super Prompt,
SAM prompt encoder, SAM mask decoder, SAM semantic logits, or SAM distillation.
SAM code is retained for license-compatible optional structure precomputation
only; inference uses UNet and VNet.
