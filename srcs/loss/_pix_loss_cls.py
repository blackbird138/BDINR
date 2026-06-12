import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM  # pip install pytorch-msssim

# ===========================
# global loss info extract
# ===========================
LOSSES = {}

def add2loss(cls):
    if cls.__name__ in LOSSES:
        raise ValueError(f'{cls.__name__} is already in the LOSSES list')
    else:
        LOSSES[cls.__name__] = cls
    return cls

# ===========================
# weighted_loss
# ===========================


class WeightedLoss(nn.Module):
    """
    weighted multi-loss
    loss_conf_dict: {loss_type1: weight|[weight,{kwargs_dict_for_init}], ...}
        eg: loss_conf_dict = {'CharbonnierLoss':0.5, 'EdgeLoss':0.5}
        eg: loss_conf_dict = {'CharbonnierLoss':[0.5, {'eps':1e-3}], 'EdgeLoss':0.5}
    """

    def __init__(self, loss_conf_dict):
        super(WeightedLoss, self).__init__()
        self.loss_conf_dict = loss_conf_dict

        # instantiate classes
        self.losses = []
        for k, v in loss_conf_dict.items():
            if isinstance(v, (float, int)):
                assert v >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES[k](), 'weight': v})
            elif isinstance(v, list) and len(v) == 2:
                assert v[0] >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES[k](**v[1]), 'weight': v[0]})
            else:
                raise ValueError(
                    f"the Key({k})'s Value {v} in Dict(loss_conf_dict) should be scalar(weight) | list[weight, args] ")

    def forward(self, output, target):
        loss_v = 0
        for loss in self.losses:
            loss_v += loss['cls'](output, target)*loss['weight']

        return loss_v

# ===========================
# basic_loss
# ===========================


@add2loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        diff = output.to('cuda:0') - target.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


@add2loss
class L1Loss(nn.Module):
    """Mean Square Error Loss (L2)"""

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, target):
        return F.l1_loss(output, target)


@add2loss
class MSELoss(nn.Module):
    """Mean Square Error Loss (L2)"""

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)


@add2loss
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@add2loss
class SSIMLoss(SSIM):
    """Structural Similarity Index Measure Loss
    Directly use the SSIM class provided by pytorch_msssim
    """
    pass


@add2loss
class OneMinusSSIMLoss(SSIM):
    """SSIM distance for inputs normalized to [0, 1]."""

    def __init__(self, data_range=1.0, **kwargs):
        super(OneMinusSSIMLoss, self).__init__(data_range=data_range, **kwargs)

    def forward(self, output, target):
        return 1.0 - super().forward(output, target)


@add2loss
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, output, target):
        loss = self.loss(self.laplacian_kernel(output.to('cuda:0')),
                         self.laplacian_kernel(target.to('cuda:0')))
        return loss


class CodedFlashPairLoss(nn.Module):
    """Composite loss for paired on/off coded-flash sequence restoration."""

    def __init__(
        self,
        gamma=2.2,
        foreground_weight=4.0,
        linear_seq_weight=0.2,
        srgb_seq_weight=1.0,
        ssim_weight=0.1,
        edge_weight=0.05,
        reblur_weight=0.2,
        eps=1e-3,
        ssim_win_size=7,
    ):
        super(CodedFlashPairLoss, self).__init__()
        self.gamma = float(gamma)
        self.foreground_weight = float(foreground_weight)
        self.linear_seq_weight = float(linear_seq_weight)
        self.srgb_seq_weight = float(srgb_seq_weight)
        self.ssim_weight = float(ssim_weight)
        self.edge_weight = float(edge_weight)
        self.reblur_weight = float(reblur_weight)
        self.eps = float(eps)
        self.ssim_win_size = int(ssim_win_size)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3, win_size=self.ssim_win_size)

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.register_buffer("edge_kernel", torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, pred_sequence, batch):
        target = batch["target"]
        off_frames = batch["off_frames"]
        coded_image = batch["coded_image"]
        code = batch["code"]
        if "mask" not in batch:
            raise KeyError("CodedFlashPairLoss requires batch['mask']; set data_loader.use_mask=true")
        mask = batch["mask"]

        self._validate_shapes(pred_sequence, target, off_frames, coded_image, mask)

        weight = self._mask_weight(mask, pred_sequence.dtype)
        seq_weight = weight[:, None, :, :, :]

        pred_srgb = self._linear_to_srgb(pred_sequence)
        target_srgb = self._linear_to_srgb(target)

        loss_linear_seq = self._weighted_charbonnier(pred_sequence, target, seq_weight)
        loss_srgb_seq = self._weighted_charbonnier(pred_srgb, target_srgb, seq_weight)
        loss_ssim = self._masked_ssim_distance(pred_srgb, target_srgb, mask)
        loss_edge = self._masked_edge_loss(pred_srgb, target_srgb, seq_weight)

        reblurred = self._coded_flash_reblur(pred_sequence, off_frames, code)
        loss_reblur = self._charbonnier(reblurred, coded_image)

        total = (
            self.linear_seq_weight * loss_linear_seq
            + self.srgb_seq_weight * loss_srgb_seq
            + self.ssim_weight * loss_ssim
            + self.edge_weight * loss_edge
            + self.reblur_weight * loss_reblur
        )
        parts = {
            "loss_linear_seq": loss_linear_seq.detach(),
            "loss_srgb_seq": loss_srgb_seq.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_reblur": loss_reblur.detach(),
        }
        return {"loss": total, "parts": parts, "reblurred": reblurred}

    def _validate_shapes(self, pred_sequence, target, off_frames, coded_image, mask):
        if pred_sequence.ndim != 5 or target.ndim != 5 or off_frames.ndim != 5:
            raise ValueError("pred_sequence, target, and off_frames must have shape [B,T,C,H,W]")
        if pred_sequence.shape != target.shape or pred_sequence.shape != off_frames.shape:
            raise ValueError("pred_sequence, target, and off_frames must have the same shape")
        if coded_image.shape != pred_sequence[:, 0].shape:
            raise ValueError("coded_image must have shape [B,C,H,W]")
        if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[0] != pred_sequence.shape[0]:
            raise ValueError("mask must have shape [B,1,H,W]")
        if mask.shape[-2:] != pred_sequence.shape[-2:]:
            raise ValueError("mask spatial shape must match pred_sequence")

    def _mask_weight(self, mask, dtype):
        weight = 1.0 + self.foreground_weight * mask.to(dtype=dtype)
        denom = weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return weight / denom

    def _linear_to_srgb(self, tensor):
        return torch.clamp(tensor, 0.0, 1.0).pow(1.0 / self.gamma)

    def _charbonnier(self, output, target):
        diff = output - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

    def _weighted_charbonnier(self, output, target, weight):
        diff = output - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss * weight.to(device=output.device, dtype=output.dtype))

    def _masked_ssim_distance(self, pred_srgb, target_srgb, mask):
        if min(pred_srgb.shape[-2:]) < self.ssim_win_size:
            return pred_srgb.new_zeros(())
        mask_seq = mask[:, None, :, :, :].to(device=pred_srgb.device, dtype=pred_srgb.dtype)
        mask_seq = mask_seq.expand(-1, pred_srgb.shape[1], -1, -1, -1)
        pred_flat = torch.flatten(pred_srgb * mask_seq + target_srgb * (1.0 - mask_seq), end_dim=1)
        target_flat = torch.flatten(target_srgb, end_dim=1)
        self.ssim.to(device=pred_flat.device)
        return 1.0 - self.ssim(pred_flat, target_flat)

    def _masked_edge_loss(self, pred_srgb, target_srgb, seq_weight):
        pred_flat = torch.flatten(pred_srgb, end_dim=1)
        target_flat = torch.flatten(target_srgb, end_dim=1)
        weight_flat = seq_weight.expand(-1, pred_srgb.shape[1], -1, -1, -1)
        weight_flat = torch.flatten(weight_flat, end_dim=1)
        pred_edge = self._laplacian_kernel(pred_flat)
        target_edge = self._laplacian_kernel(target_flat)
        return self._weighted_charbonnier(pred_edge, target_edge, weight_flat)

    def _laplacian_kernel(self, current):
        filtered = self._conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self._conv_gauss(new_filter)
        return current - filtered

    def _conv_gauss(self, image):
        kernel = self.edge_kernel.to(device=image.device, dtype=image.dtype)
        _, _, kw, kh = kernel.shape
        image = F.pad(image, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(image, kernel, groups=image.shape[1])

    def _coded_flash_reblur(self, sequence, off_frames, code):
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


@add2loss
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, output, target):
        diff = torch.fft.fft2(output.to('cuda:0')) - \
            torch.fft.fft2(target.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


@add2loss
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self, output, *args):
        batch_size = output.size()[0]
        h_x = output.size()[2]
        w_x = output.size()[3]
        count_h = self._tensor_size(output[:, :, 1:, :])
        count_w = self._tensor_size(output[:, :, :, 1:])
        h_tv = torch.pow((output[:, :, 1:, :]-output[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((output[:, :, :, 1:]-output[:, :, :, :w_x-1]), 2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size


if __name__ == "__main__":
    import PerceptualLoss
    output = torch.randn(4, 3, 10, 10)
    target = torch.randn(4, 3, 10, 10)
    loss_conf_dict = {'CharbonnierLoss': 0.5, 'fftLoss': 0.5}

    Weighted_Loss = WeightedLoss(loss_conf_dict)
    loss_v = Weighted_Loss(output, target)
    print('loss: ', loss_v)
