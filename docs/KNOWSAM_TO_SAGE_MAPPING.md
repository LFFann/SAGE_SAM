# KnowSAM To SAGE-SAM Mapping

| KnowSAM component | SAGE-SAM treatment |
| --- | --- |
| UNet | Copied and retained as branch A. |
| VNet | Copied and retained as branch B. |
| Discriminator / HAM | Not instantiated by the new main path. |
| Super_Prompt | Not used by the new main path. |
| SAM Prompt Encoder | Not called by SAGE-SAM. |
| SAM Mask Decoder | Not called by SAGE-SAM. |
| SAM Adapter | Not trained. |
| SAM semantic prediction | Removed from active supervision. |
| SAM KD | Removed. |
| mutual BCE | Replaced by set-valued partial supervision. |
| global entropy minimization | Removed from the active path. |
| fixed-grid UGDA | Replaced by hardness-controlled structure-safe perturbation. |
| fusion_map | Replaced by non-parametric branch averaging and set reliability. |
| SAM optimizer | Removed. |
| labeled/unlabeled equal-size constraint | The trainer accepts separate labeled and unlabeled batches. |
