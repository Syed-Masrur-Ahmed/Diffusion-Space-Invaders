"""PSNR, SSIM, and LPIPS metrics (ROADMAP 4.11). Owned by CP3-A.

Every metric takes `pred` and `target` as float tensors in [0, 1] with shape (N, C, H, W),
C = 3 or 1, on any device, and returns one value per image, shape (N,).
"""

import functools

import torch
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure

PSNR_CAP = 100.0  # dB, returned for identical images


def _check(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"pred and target must both be (N,C,H,W), got {tuple(pred.shape)} "
                         f"and {tuple(target.shape)}")


def psnr(pred: Tensor, target: Tensor) -> Tensor:
    """Per-image PSNR in dB with data_range 1, capped at 100 dB."""
    _check(pred, target)
    mse = (pred.float() - target.float()).square().flatten(1).mean(1)
    return (-10 * torch.log10(mse)).clamp(max=PSNR_CAP)  # mse = 0 gives inf, then the cap


def ssim(pred: Tensor, target: Tensor) -> Tensor:
    """Per-image SSIM (torchmetrics defaults: Gaussian 11x11 window, sigma 1.5), data_range 1."""
    _check(pred, target)
    return structural_similarity_index_measure(pred.float(), target.float(), data_range=1.0,
                                               reduction="none")


@functools.cache
def _lpips_model(device: torch.device) -> torch.nn.Module:
    """LPIPS(alex), built once per device (the first call downloads AlexNet weights)."""
    import lpips as lpips_lib  # lazy: importing it loads torchvision

    return lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)


def lpips(pred: Tensor, target: Tensor) -> Tensor:
    """Per-image LPIPS(alex) at native resolution; greyscale is repeated to 3 channels."""
    _check(pred, target)
    if pred.shape[1] == 1:
        pred, target = pred.expand(-1, 3, -1, -1), target.expand(-1, 3, -1, -1)
    with torch.inference_mode():
        out = _lpips_model(pred.device)(pred.float() * 2 - 1, target.float() * 2 - 1)
    return out.flatten()


METRICS = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
