"""SAGE-SAM dual-branch segmentor.

This file reuses the copied KnowSAM UNet and VNet implementations from this
repository, but removes the active HAM/Discriminator and SAM decoder path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import torch
from torch import nn

from Model.model import UNet, VNet


class DualSegmentor(nn.Module):
    """UNet/VNet-only SAGE-SAM segmentor.

    The member names `UNet` and `VNet` are intentionally preserved so copied
    KnowSAM checkpoints with `UNet.*` and `VNet.*` keys can be remapped without
    renaming convolution parameters.
    """

    def __init__(self, args: object | None = None, in_channels: int = 3, num_classes: int = 3, bilinear: bool = False):
        super().__init__()
        if args is not None:
            in_channels = int(getattr(args, "in_channels", in_channels))
            num_classes = int(getattr(args, "num_classes", num_classes))
            bilinear = bool(getattr(args, "bilinear", bilinear))
        self.args = SimpleNamespace(in_channels=in_channels, num_classes=num_classes, bilinear=bilinear)
        self.UNet = UNet(in_chns=in_channels, class_num=num_classes, bilinear=bilinear)
        self.VNet = VNet(n_channels=in_channels, n_classes=num_classes)

    def _forward_unet(self, x: torch.Tensor, return_features: bool) -> dict[str, torch.Tensor | None]:
        x0 = self.UNet.in_conv(x)
        x1 = self.UNet.down1(x0)
        x2 = self.UNet.down2(x1)
        x3 = self.UNet.down3(x2)
        x4 = self.UNet.down4(x3)
        dec = self.UNet.up1(x4, x3)
        dec = self.UNet.up2(dec, x2)
        dec = self.UNet.up3(dec, x1)
        dec = self.UNet.up4(dec, x0)
        logits = self.UNet.out_conv(dec)
        return {
            "logits_a": logits,
            "prob_a": torch.softmax(logits, dim=1),
            "feature_a": x4 if return_features else None,
            "decoder_feature_a": dec if return_features else None,
        }

    def _forward_vnet(self, x: torch.Tensor, return_features: bool) -> dict[str, torch.Tensor | None]:
        x0 = self.VNet.block_one(x)
        x1 = self.VNet.block_one_dw(x0)
        x2 = self.VNet.block_two_dw(self.VNet.block_two(x1))
        x3 = self.VNet.block_three_dw(self.VNet.block_three(x2))
        x4 = self.VNet.block_four_dw(self.VNet.block_four(x3))

        dec = self.VNet.block_five_up(self.VNet.block_five(x4))
        dec = dec + x3
        dec = self.VNet.block_six_up(self.VNet.block_six(dec))
        dec = dec + x2
        dec = self.VNet.block_seven_up(self.VNet.block_seven(dec))
        dec = dec + x1
        dec = self.VNet.block_eight_up(self.VNet.block_eight(dec))
        dec = dec + x0
        dec = self.VNet.block_nine(dec)
        if self.VNet.has_dropout:
            dec = self.VNet.dropout(dec)
        logits = self.VNet.out_conv(dec)
        return {
            "logits_b": logits,
            "prob_b": torch.softmax(logits, dim=1),
            "feature_b": x4 if return_features else None,
            "decoder_feature_b": dec if return_features else None,
        }

    def forward(
        self,
        x: torch.Tensor,
        branches: Iterable[str] = ("A", "B"),
        return_features: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        selected = {branch.upper() for branch in branches}
        invalid = selected.difference({"A", "B"})
        if invalid:
            raise ValueError(f"Unknown branches: {sorted(invalid)}")

        output: dict[str, torch.Tensor | None] = {
            "logits_a": None,
            "prob_a": None,
            "feature_a": None,
            "decoder_feature_a": None,
            "logits_b": None,
            "prob_b": None,
            "feature_b": None,
            "decoder_feature_b": None,
        }
        if "A" in selected:
            output.update(self._forward_unet(x, return_features))
        if "B" in selected:
            output.update(self._forward_vnet(x, return_features))
        return output

    def trainable_parameter_report(self) -> dict[str, int]:
        unet = sum(p.numel() for p in self.UNet.parameters() if p.requires_grad)
        vnet = sum(p.numel() for p in self.VNet.parameters() if p.requires_grad)
        return {"UNet": unet, "VNet": vnet, "total": unet + vnet}
