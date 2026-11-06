"""Depth noise adapted from InstinctMJ SuperMimic's pinned depth.py.

Source project-instinct/InstinctMJ revision 03563ee, CC BY-NC 4.0.
The stateless source implementation preserves deterministic stages and random
marginals. Its artifact centers use replacement and close-block values are
independent per image, as declared by the source implementation.
"""
from __future__ import annotations
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
PATCH = 25
CONTOUR = ((.5,(-1,-1)),(.1,(-1,0)),(.5,(-1,1)),(1.2,(0,-1)),(1.2,(0,1)),(.5,(1,-1)),(.1,(1,0)),(.5,(1,1)))
def binomial_cdf(n: int, p: float, tail: float=1e-07) -> torch.Tensor:
    """``cdf[j] = P(Binomial(n, p) <= j)`` for ``j < K``, K the smallest count with ``P(count > K) < tail``."""
    pmf = lambda j: math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p) + (n - j) * math.log1p(-p))
    cdf, total = ([], 0.0)
    while True:
        total += pmf(len(cdf))
        cdf.append(total)
        if 1 - total < tail:
            return torch.tensor(cdf)

def box_profile(c: torch.Tensor, size: torch.Tensor, length: int) -> torch.Tensor:
    """1-D factor (..., length) of the ``grid_sample`` of the box patch stretched over ``[c - size/2, c + size/2]`` (both ends
    clipped to ``[0, length]``) along one image axis: the same grid, ``align_corners=False``, bilinear, zero padding."""
    lo, hi = ((c - size / 2).clamp(0.0, float(length)), (c + size / 2).clamp(0.0, float(length)))
    lin = torch.linspace(-1, 1, length, device=c.device)
    extent = hi - lo
    ok = extent > 0
    g = (lin * length + length - hi.unsqueeze(-1) - lo.unsqueeze(-1)) / torch.where(ok, extent, 1.0).unsqueeze(-1)
    ix = ((g + 1) * PATCH - 1) / 2
    i0 = ix.floor()
    inside = lambda i: ((i >= 1) & (i <= PATCH - 2)).float()
    w1 = ix - i0
    return torch.where(ok.unsqueeze(-1), (1 - w1) * inside(i0) + w1 * inside(i0 + 1), 0.0)

class SourceDepthNoise:

    def __init__(self, device):
        self.device, self.H, self.W = (device, 27, 48)
        self.contour_cfg = SimpleNamespace(contour_threshold=1.8)
        self.artifact_cfg = SimpleNamespace(artifacts_prob=0.0001, artifacts_height_mean_std=(2, 0.5), artifacts_width_mean_std=(2, 0.5), noise_value=0.0)
        self.stereo_cfg = SimpleNamespace(min_value=0.12, max_value=1.2, noise_std=0.02)
        self.close_cfg = SimpleNamespace(close_threshold=0.12, full_block_artifacts_prob=0.008, full_block_height_mean_std=(62, 1.5), full_block_width_mean_std=(3, 0.01), full_block_values=[0.0, 0.25, 0.5, 1.0, 3.0], half_block_spark_prob=0.02, half_block_value=30.0)
        self.normalize_cfg = SimpleNamespace(depth_range=(0.0, 2.0), output_range=(0.0, 1.0))
        self.crop_cfg = SimpleNamespace(crop_region=(2, 2, 2, 2), resize_shape=(18, 32))
        self.artifact_cdf = binomial_cdf(27 * 48, 0.0001).to(device)
        self.close_cdf = binomial_cdf(27 * 48, 0.008).to(device)
        kernel = torch.exp(-0.5 * (torch.arange(-1.0, 2.0) / 0.5) ** 2)
        self.blur_w = (kernel / kernel.sum()).tolist()
        self.close_values = torch.tensor(self.close_cfg.full_block_values, device=device)

    def contour_mask(self, d: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> bool mask of ``depth_contour_noise``: max over the 8 kernels of |c (center - neighbor)| > threshold
        (zero padding: a neighbor outside the image is 0)."""
        p = F.pad(d, (1, 1, 1, 1))
        H, W = d.shape[-2:]
        m = torch.zeros_like(d)
        for coef, (dy, dx) in CONTOUR:
            m = torch.maximum(m, (coef * (d - p[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W])).abs())
        return m > self.contour_cfg.contour_threshold

    def blur(self, d: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> ``gaussian_blur_noise`` (3x3, reflect padding, separable weights of torchvision's ``GaussianBlur``)."""
        w0, w1, w2 = self.blur_w
        p = F.pad(d.unsqueeze(1), (1, 1, 1, 1), mode='reflect')[:, 0]
        r = w0 * p[:, :, :-2] + w1 * p[:, :, 1:-1] + w2 * p[:, :, 2:]
        return w0 * r[:, :-2] + w1 * r[:, 1:-1] + w2 * r[:, 2:]

    def artifact_mask(self, cdf: torch.Tensor, prob_valid: torch.Tensor | None, hm_std, wm_std, n: int) -> torch.Tensor:
        """(n, H, W) union (sum clamped to [0, 1]) of the random rectangle artifacts of ``_add_depth_artifacts``;
        ``prob_valid`` (n, H, W) bool = the pixels allowed as centers (``data > 0``), None = all."""
        H, W, dev = (self.H, self.W, self.device)
        K = len(cdf)
        count = (torch.rand(n, 1, device=dev) > cdf).sum(1, keepdim=True)
        active = torch.arange(K, device=dev) < count
        cy, cx = (torch.rand(2, n, K, device=dev) * torch.tensor([[[H]], [[W]]], device=dev)).floor()
        if prob_valid is not None:
            active = active & prob_valid.flatten(1).gather(1, (cy * W + cx).long())
        sh = (hm_std[0] + torch.randn(n, K, device=dev) * hm_std[1]).clamp(0.0, float(H))
        sw = (wm_std[0] + torch.randn(n, K, device=dev) * wm_std[1]).clamp(0.0, float(W))
        fy, fx = (box_profile(c, z, L) * active.unsqueeze(-1) for c, z, L in ((cy, sh, H), (cx, sw, W)))
        return torch.einsum('nkh,nkw->nhw', fy, fx).clamp(0.0, 1.0)

    def noised(self, meters: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 1) raw meters -> the policy's observation of the image (B, 1, 18, 32): the sensor's whole noise pipeline
        (new random samples on every call; differentiable w.r.t. ``meters``; stateless, see the module docstring)."""
        d = meters[..., 0]
        n = d.shape[0]
        a, s, c = (self.artifact_cfg, self.stereo_cfg, self.close_cfg)
        with torch.no_grad():
            mask = self.contour_mask(d)
        d = torch.where(mask, torch.zeros_like(d), d)
        m = self.artifact_mask(self.artifact_cdf, d.detach() > 0, a.artifacts_height_mean_std, a.artifacts_width_mean_std, n)
        d = d * (1 - m) + m * a.noise_value
        apply = (d.detach() >= s.min_value) & (d.detach() <= s.max_value)
        d = d + torch.randn_like(d) * s.noise_std * apply
        d = self.blur(d)
        close = d.detach() < c.close_threshold
        vertical = close.sum(1, keepdim=True) > self.H * 0.6
        full, half = (vertical & close, ~vertical & close)
        value = self.close_values[torch.randint(len(self.close_values), (n, 1, 1), device=self.device)]
        art = self.artifact_mask(self.close_cdf, None, c.full_block_height_mean_std, c.full_block_width_mean_std, n)
        spark = torch.rand_like(d) < c.half_block_spark_prob
        d = torch.where(full, art * value, d)
        d = torch.where(half, spark.float() * c.half_block_value, d)
        nz, cr = (self.normalize_cfg, self.crop_cfg)
        d = d.clip(nz.depth_range[0], nz.depth_range[1])
        d = (d - nz.depth_range[0]) / (nz.depth_range[1] - nz.depth_range[0])
        d = d * (nz.output_range[1] - nz.output_range[0]) + nz.output_range[0]
        up, down, left, right = cr.crop_region
        d = d[:, up:self.H - down, left:self.W - right]
        return F.interpolate(d.unsqueeze(1), size=cr.resize_shape, mode='bilinear', align_corners=False)
