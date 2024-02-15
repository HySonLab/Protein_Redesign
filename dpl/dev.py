
import math
import numpy as np
from typing import List, Tuple, Union, Optional, Callable, Dict
import random
import einops
import torch
from einops.layers.torch import Rearrange #it's a layer
from torch import nn
from einops import rearrange  # noqa
from torch import Tensor
from .features import ALLOWABLE_ATOM_FEATURES, ALLOWABLE_BOND_FEATURES

get_max_val = lambda x: torch.finfo(x.dtype).max  # noqa

def safe_norm(x: Tensor, dim: Union[int, Tuple[int, ...]], keepdim: bool = False, eps: float = 1e-12):
    """Safe norm of a vector"""
    return torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim) + eps)

def default(x, y):
    """Returns x if x exists, otherwise y."""
    return x if x is not None else y

def exists(x):
    """Returns true iff x is not None."""
    return x is not None

def get_mask_len(
        n_res: int,
        min_n_max_p: Tuple[float, float],
        min_n_max_res: Optional[Tuple[int, int]] = None,
) -> int:
    """gets number of residues to mask given
    (1) number of residues in sequence
    (2) minimum/maximum percentage of sequence to mask (min_n_max_p)
    (3) [Optional] upper and lower bounds on number of residues to mask (min_n_max_res)
    """
    min_frac, max_frac = min_n_max_p
    min_res, max_res = default(min_n_max_res, [0, n_res])
    min_res = int(max(min_res, n_res * min_frac, 0))
    max_res = int(max(min_res, min(max_res, n_res * max_frac, n_res)))
    mask_len = random.randint(min_res, max_res)
    return min(n_res, mask_len)

def k_spatial_nearest_neighbors(
    points: Tensor, idx: int, top_k: int, max_dist: Optional[float] = None, include_self: bool = False
) -> Tensor:
    """Get k nearest neighbors for point at index idx"""
    assert points.ndim == 2, (
        f"this function expects coordinate of shape (n,3) - " f"does not work for batched coordinates"
    )
    diffs = points - rearrange(points[idx], "c -> () c")
    dists = safe_norm(diffs, dim=-1)
    dists[idx] = 0 if include_self else get_max_val(dists)
    top_k = min(points.shape[0] - int(~include_self), top_k)
    nbr_dists, nbr_indices = dists.topk(k=int(top_k), dim=-1, largest=False)
    return nbr_indices[nbr_dists < default(max_dist, get_max_val(dists))]

def bool_tensor(n, fill=True, posns: Optional[Tensor] = None):
    """Bool tensor of length n, initialized to 'fill' in all given positions
    and ~fill in other positions.
    """
    # create bool tensor initialized to ~fill
    mask = torch.zeros(n).bool() if fill else torch.ones(n).bool()
    mask[posns if exists(posns) else torch.arange(n)] = fill
    return mask

def spatial_mask(
        num_residue: int,
        coords: Tensor,
        min_n_max_p: Tuple[float, float],
        top_k: Union[List[int], int] = 30,
        max_radius=12,
        mask_self: bool = False,
        atom_pos: int = 1,
        **kwargs,  # noqa

) -> Tensor:
    """Masks positions in a sequence based on spatial proximity to a random query residue
    
    atom_pos: index of protein RESIDUE_ATOMS, 1 -> Ca
    """
    coords = coords.squeeze(0) if coords.ndim == 4 else coords
    top_k = get_mask_len(
        n_res=num_residue,
        min_n_max_p=min_n_max_p,
        min_n_max_res=(1, top_k)
    )
    mask_posns = k_spatial_nearest_neighbors(
        points=coords[:, atom_pos],
        idx=np.random.choice(num_residue),
        top_k=min(num_residue, top_k - int(not mask_self)),
        max_dist=max_radius,
        include_self=not mask_self
    )
    return bool_tensor(num_residue, posns=mask_posns, fill=True)



class SpatialMaskingModule(nn.Module):
    def __init__(self,
                max_p: float = 1.0,
                inf = 1e10
                ):
        super().__init__()
        self.max_p = self.max_p
        self.inf = inf
    
    
    def forward(self, residue_ca_pos, 
                      residue_mask,
                      atom_pos,
                      atom_mask):
        """
        residue_ca_pos : Ca coords; [N_seq, 3]
        
        """
        n_res = residue_mask.sum(-1).max()
        n_mean_res = residue_mask.sum(-1).median()
        
        atom_centroids = atom_pos.sum(-2) / atom_mask.sum(-1, keepdim = True)
        atom_centroids = atom_centroids.unsqueeze(-2).expand(-1, atom_pos.shape[1], -1)
        
        dists = (safe_norm(atom_centroids - residue_ca_pos, dim = -1) +  
                        (1-residue_mask) * self.inf) #with padding
        
        topk = np.round(
                np.random.choice(
                                np.linspace(0,self.max_p,1000)
                                )
                *n_mean_res
                )
        nbr_dists, nbr_indices = dists.topk(k=topk, dim=-1, largest=False)
        residue_spatial_mask = residue_mask.scatter_(dim=-11, index=nbr_indices, value=0)
        
        return residue_spatial_mask
        
        
    

def random_mask(num_residue: int, coords: Tensor, min_p: float, max_p: float, **kwargs) -> Tensor:  # noqa
    """Randomly masks each sequence position w.p. in range (min_p, max_p)"""
    mask_prob = np.random.uniform(min_p, max_p)
    mask_posns = torch.arange(num_residue)[torch.rand(num_residue) < mask_prob]
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


