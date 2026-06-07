"""BDINR-style restoration model for pre-synthesized coded-flash inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from srcs.model.bd_model import BDNeRV_RC


DEFAULT_CODE = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]


class CodedFlashBDINR(nn.Module):
    """Recover a sharp frame sequence from one coded-flash image."""

    def __init__(
        self,
        ce_code_n: int = 16,
        frame_n: int = 16,
        ce_code_init: list[int] | None = None,
        bd_net: str = "BDNeRV_RC",
    ) -> None:
        super().__init__()
        self.ce_code_n = int(ce_code_n)
        self.frame_n = int(frame_n)
        if self.ce_code_n != self.frame_n:
            raise ValueError("CodedFlashBDINR currently requires ce_code_n == frame_n")

        code = DEFAULT_CODE if ce_code_init is None else ce_code_init
        code_tensor = torch.tensor(code, dtype=torch.float32).view(self.ce_code_n, 1)
        if code_tensor.numel() != self.ce_code_n:
            raise ValueError("ce_code_init length must equal ce_code_n")
        if not torch.all((code_tensor == 0) | (code_tensor == 1)):
            raise ValueError("ce_code_init must be binary")

        self.register_buffer("time_idx", torch.linspace(0, 1, self.ce_code_n).view(self.ce_code_n, 1))
        self.register_buffer("ce_code", code_tensor)

        if bd_net == "BDNeRV_RC":
            self.DeBlurNet = BDNeRV_RC()
        else:
            raise NotImplementedError(f"No restoration model named {bd_net}")

    def forward(self, coded_image: torch.Tensor) -> torch.Tensor:
        return self.DeBlurNet(
            ce_blur=coded_image,
            time_idx=self.time_idx.to(coded_image.device),
            ce_code=self.ce_code.to(coded_image.device),
        )


def coded_flash_reblur(
    sequence: torch.Tensor,
    off_frames: torch.Tensor,
    code: torch.Tensor,
) -> torch.Tensor:
    """Recompose a coded-flash observation from a predicted/on sequence."""
    if sequence.ndim != 5 or off_frames.ndim != 5:
        raise ValueError("sequence and off_frames must have shape [B,T,C,H,W]")
    if sequence.shape != off_frames.shape:
        raise ValueError("sequence and off_frames must have the same shape")

    batch, frame_n = sequence.shape[:2]
    code = code.to(device=sequence.device, dtype=sequence.dtype)
    if code.ndim == 1:
        code = code.view(1, frame_n, 1, 1, 1)
    elif code.ndim == 2:
        if code.shape[0] != batch or code.shape[1] != frame_n:
            raise ValueError("batched code must have shape [B,T]")
        code = code.view(batch, frame_n, 1, 1, 1)
    else:
        raise ValueError("code must have shape [T] or [B,T]")
    return torch.mean(off_frames + code * (sequence - off_frames), dim=1)
