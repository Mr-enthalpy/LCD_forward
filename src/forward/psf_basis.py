import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MaskEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.fc(feat)


class PSFBasisForward(nn.Module):
    """
    Low-rank PSF-space baseline.

    Input:
        masks: [B, T, 1, Hm, Wm]

    Output dict:
        {
            "psfs":  [B, T, L, Hp, Wp],
            "coeff": [B, T, L, K],
        }
    """

    def __init__(
        self,
        n_lambda: int,
        psf_size: int = 128,
        rank: int = 16,
        embed_dim: int = 128,
        normalize_psf: bool = True,
    ):
        super().__init__()
        self.n_lambda = n_lambda
        self.psf_size = psf_size
        self.rank = rank
        self.normalize_psf = normalize_psf

        self.encoder = MaskEncoder(in_ch=1, embed_dim=embed_dim)
        self.coeff_head = nn.Linear(embed_dim, n_lambda * rank)

        self.psf_mean = nn.Parameter(
            torch.randn(n_lambda, psf_size, psf_size) * 0.01
        )
        self.psf_basis = nn.Parameter(
            torch.randn(n_lambda, rank, psf_size, psf_size) * 0.01
        )

    def _normalize_psf(self, psfs: torch.Tensor) -> torch.Tensor:
        return psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    def forward_single_mask(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        mask: [B, 1, Hm, Wm]
        """
        bsz = mask.shape[0]
        z = self.encoder(mask)
        coeff = self.coeff_head(z).reshape(bsz, self.n_lambda, self.rank)  # [B, L, K]

        psfs = self.psf_mean.unsqueeze(0) + torch.einsum(
            "blk,lkhw->blhw", coeff, self.psf_basis
        )
        psfs = F.softplus(psfs)

        if self.normalize_psf:
            psfs = self._normalize_psf(psfs)

        return {
            "psfs": psfs,    # [B, L, Hp, Wp]
            "coeff": coeff,  # [B, L, K]
        }

    def forward(self, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        masks: [B, T, 1, Hm, Wm]
        """
        bsz, num_frames = masks.shape[:2]
        masks_flat = masks.reshape(bsz * num_frames, 1, masks.shape[-2], masks.shape[-1])

        out = self.forward_single_mask(masks_flat)

        psfs = out["psfs"].reshape(
            bsz, num_frames, self.n_lambda, self.psf_size, self.psf_size
        )
        coeff = out["coeff"].reshape(
            bsz, num_frames, self.n_lambda, self.rank
        )

        return {
            "psfs": psfs,
            "coeff": coeff,
        }