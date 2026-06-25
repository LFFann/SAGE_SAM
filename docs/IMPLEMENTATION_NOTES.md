# Implementation Notes

- `prediction_sage_sam.py` imports only `DualSegmentor` and evaluator helpers.
- `SAMStructureEncoder` dynamically imports the SAM registry only when a real
  checkpoint is supplied for precomputation.
- Checkpoint conversion drops inactive HAM/SAM decoder keys but preserves
  `UNet.*` and `VNet.*` naming.
- The structure graph stores local right/down edges instead of a dense
  `G^2 x G^2` affinity matrix.
