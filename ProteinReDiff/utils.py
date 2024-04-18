"""
Adapted from Nakata, S., Mori, Y. & Tanaka, S. 
End-to-end protein–ligand complex structure generation with diffusion-based generative models.
BMC Bioinformatics 24, 233 (2023).
https://doi.org/10.1186/s12859-023-05354-5

Repository: https://github.com/shuyana/DiffusionProteinLigand

ProteinReDiff includes significant innovations including:
- Stochastically masking & featurization of protein sequences
- Adaptations of Single Representation Attention and Outer Product Mean from AF2
- Parameterization of \beta_T diffusion (instead of using variational lower bound in DPL)
- Denoising through both sequence and structure spaces
- Flexible generation output (sequences only, sequence-structures)

"""

from typing import Tuple

import einops
import torch


def angstrom_to_nanometre(pos: torch.Tensor) -> torch.Tensor:
    return 0.1 * pos


def nanometre_to_angstrom(x: torch.Tensor) -> torch.Tensor:
    return 10.0 * x


def remove_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = einops.repeat(mask, "b i -> b i xyz", xyz=x.size(-1))
    x_sum = einops.reduce(mask * x, "b i xyz -> b 1 xyz", "sum")
    norm = einops.reduce(mask, "b i xyz -> b 1 xyz", "sum")
    return x - mask * x_sum / norm


def nearest_bin(
    x: torch.Tensor, bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    factory_kwargs = torch.nn.factory_kwargs({"dtype": x.dtype, "device": x.device})
    w = 0.5 * (max_val - min_val) / bins
    v = torch.linspace(min_val + w, max_val - w, bins, **factory_kwargs)
    return torch.argmin(torch.abs(x.unsqueeze(-1) - v), dim=-1)


def pseudo_beta(
    residue_atom_pos: torch.Tensor, residue_atom_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_pos = residue_atom_pos[:, :, 0]
    n_mask = residue_atom_mask[:, :, 0]
    ca_pos = residue_atom_pos[:, :, 1]
    ca_mask = residue_atom_mask[:, :, 1]
    c_pos = residue_atom_pos[:, :, 2]
    c_mask = residue_atom_mask[:, :, 2]
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = torch.cross(b, c, dim=-1)
    # The magic numbers below are from:
    # https://github.com/RosettaCommons/trRosetta2/blob/dba6078ebda9f2429264ace3deaffe50d9899def/trRosetta/coords6d.py#L49
    pseudo_cb_pos = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_pos
    pseudo_cb_mask = n_mask * ca_mask * c_mask
    return pseudo_cb_pos, pseudo_cb_mask
