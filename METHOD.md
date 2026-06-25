# SAGE-SAM Method

SAGE-SAM changes the copied KnowSAM/a3-sam path in three ways.

1. SAM is a calibrated, class-agnostic structure reference. SAGE-SAM uses only
   the frozen SAM image encoder for optional structure precomputation. It does
   not call the prompt encoder or mask decoder, does not train adapters, and
   does not distill semantic SAM logits.
2. UNet/VNet disagreement becomes set-valued supervision. Each branch produces
   a calibrated candidate set. The union receives partial-label supervision,
   the intersection is treated as reliable core evidence, and classes outside
   the union receive negative supervision.
3. Structure consistency is local and non-parametric. SAM embeddings define
   same/boundary/unknown local edges after target-domain calibration. SAGE-SAM
   propagates only over reliable same edges and uses hardness to reduce
   perturbation strength for difficult ambiguous images.

Inference loads only `DualSegmentor`, whose trainable parameters are `UNet` and
`VNet`.
