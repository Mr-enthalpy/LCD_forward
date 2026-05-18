import torch
import torch.nn as nn
from typing import Dict, Optional


class MaskEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   # 64 -> 32
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 32 -> 16
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.fc(feat)


class ComplexFieldBasisForward(nn.Module):
    """
    Low-rank complex-field forward model.

    Input:
        masks: [B, T, 1, Hm, Wm]

    Output dict:
        {
            "psfs":       [B, T, L, Hp, Wp],
            "coeff_real": [B, T, L, K],
            "coeff_imag": [B, T, L, K],
            "field_real": [B, T, L, Hp, Wp],
            "field_imag": [B, T, L, Hp, Wp],
        }
    """

    def __init__(
        self,
        n_lambda: int,
        psf_size: int = 128,
        rank: int = 16,
        embed_dim: int = 128,
        normalize_psf: bool = True,
        coeff_scale: Optional[float] = None,
    ):
        super().__init__()
        self.n_lambda = n_lambda
        self.psf_size = psf_size
        self.rank = rank
        self.normalize_psf = normalize_psf
        self.coeff_scale = coeff_scale

        self.encoder = MaskEncoder(in_ch=1, embed_dim=embed_dim)

        # [B, embed_dim] -> [B, L, K, 2]
        self.coeff_head = nn.Linear(embed_dim, n_lambda * rank * 2)

        # learnable complex basis
        self.field_basis_real = nn.Parameter(
            torch.randn(n_lambda, rank, psf_size, psf_size) * 0.01
        )
        self.field_basis_imag = nn.Parameter(
            torch.randn(n_lambda, rank, psf_size, psf_size) * 0.01
        )

        # learnable complex mean field
        self.field_mean_real = nn.Parameter(
            torch.randn(n_lambda, psf_size, psf_size) * 0.01
        )
        self.field_mean_imag = nn.Parameter(
            torch.randn(n_lambda, psf_size, psf_size) * 0.01
        )

    def _normalize_psf(self, psfs: torch.Tensor) -> torch.Tensor:
        return psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    def _predict_coeff(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        mask: [B, 1, Hm, Wm]
        returns:
            coeff_real: [B, L, K]
            coeff_imag: [B, L, K]
        """
        bsz = mask.shape[0]
        z = self.encoder(mask)
        coeff = self.coeff_head(z).reshape(bsz, self.n_lambda, self.rank, 2)

        coeff_real = coeff[..., 0]
        coeff_imag = coeff[..., 1]

        if self.coeff_scale is not None:
            coeff_real = torch.tanh(coeff_real) * self.coeff_scale
            coeff_imag = torch.tanh(coeff_imag) * self.coeff_scale

        return {
            "coeff_real": coeff_real,
            "coeff_imag": coeff_imag,
        }

    def forward_single_mask(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        mask: [B, 1, Hm, Wm]
        """
        coeff_dict = self._predict_coeff(mask)
        coeff_real = coeff_dict["coeff_real"]  # [B, L, K]
        coeff_imag = coeff_dict["coeff_imag"]  # [B, L, K]

        field_real = (
            self.field_mean_real.unsqueeze(0)
            + torch.einsum("blk,lkhw->blhw", coeff_real, self.field_basis_real)
            - torch.einsum("blk,lkhw->blhw", coeff_imag, self.field_basis_imag)
        )

        field_imag = (
            self.field_mean_imag.unsqueeze(0)
            + torch.einsum("blk,lkhw->blhw", coeff_real, self.field_basis_imag)
            + torch.einsum("blk,lkhw->blhw", coeff_imag, self.field_basis_real)
        )

        psfs = field_real.square() + field_imag.square()

        if self.normalize_psf:
            psfs = self._normalize_psf(psfs)

        return {
            "psfs": psfs,                    # [B, L, Hp, Wp]
            "coeff_real": coeff_real,        # [B, L, K]
            "coeff_imag": coeff_imag,        # [B, L, K]
            "field_real": field_real,        # [B, L, Hp, Wp]
            "field_imag": field_imag,        # [B, L, Hp, Wp]
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
        coeff_real = out["coeff_real"].reshape(
            bsz, num_frames, self.n_lambda, self.rank
        )
        coeff_imag = out["coeff_imag"].reshape(
            bsz, num_frames, self.n_lambda, self.rank
        )
        field_real = out["field_real"].reshape(
            bsz, num_frames, self.n_lambda, self.psf_size, self.psf_size
        )
        field_imag = out["field_imag"].reshape(
            bsz, num_frames, self.n_lambda, self.psf_size, self.psf_size
        )

        return {
            "psfs": psfs,
            "coeff_real": coeff_real,
            "coeff_imag": coeff_imag,
            "field_real": field_real,
            "field_imag": field_imag,
        }