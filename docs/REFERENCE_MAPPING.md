# Reference Mapping

The implementation reuses copied local baseline files for UNet, VNet, metrics,
data loading, visualization, and checkpoint-adjacent utilities. SAGE-SAM's new
research mechanisms live in `sage_ssl/`, `Model/sage_model.py`,
`Model/sam_structure_encoder.py`, and `engine/sage_trainer.py`.

External papers listed in the design brief are treated as conceptual references.
This repository does not copy ESL prototype banks, U2PL queues, MLLC GNNs,
GraphCL GCNs, BoundMatch boundary heads, language decoders, probabilistic
projection heads, cross-attention teachers, or EMA teachers.
