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

def safe_norm(x: Tensor, dim: Union[int, Tuple[int, ...]], keepdim: bool = False, eps: float = 1e-12):
    """Safe norm of a vector"""
    return torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim) + eps)

class SpatialMaskingModule(nn.Module):
    def __init__(self,
                # max_p: float = 0.5,
                inf = 1e10
                ):
        super().__init__()
        # self.max_p = max_p
        self.inf = inf
    
    
    def forward(self, residue_ca_pos, 
                      residue_mask,
                      atom_pos,
                      atom_mask, 
                      max_p,
                      inverse_mask = False):
        """
        residue_ca_pos : Ca coords; [N_seq, 3]
        
        """
        n_res = residue_mask.sum(-1).max()
        n_mean_res = residue_mask.sum(-1).median()
        
        atom_centroids = atom_pos.sum(-2) / atom_mask.sum(-1, keepdim = True)
        atom_centroids = atom_centroids.unsqueeze(-2).expand(-1, atom_pos.shape[1], -1)
        
        dists = (safe_norm(atom_centroids - residue_ca_pos, dim = -1) +  
                        (1-residue_mask) * self.inf) #with padding
        
        top_k = (
                np.random.choice(
                                np.linspace(0,max_p,1000)
                                )
                *n_mean_res
                ).int().item()
        
        self.residue_spatial_mask = residue_mask.detach().clone()
        self.residue_spatial_mask_esm = 1 - residue_mask.detach().clone()
        nbr_dists, nbr_indices = dists.topk(k=top_k, dim=-1, largest=False)
        self.residue_spatial_mask.scatter_(dim=-1, index=nbr_indices, value=0)
        self.residue_spatial_mask_esm.scatter_(dim=-1, index=nbr_indices, value=32)
        
        if inverse_mask:
            self.residue_inv_spatial_mask = torch.zeros_like(residue_mask)
            self.residue_inv_spatial_mask.scatter_(dim=-1, index=nbr_indices, value=1)
            return self.residue_spatial_mask, self.residue_inv_spatial_mask, self.residue_spatial_mask_esm
        else:
            return self.residue_spatial_mask, self.residue_spatial_mask_esm
    

    def mask_residue_esm(self, residue_esm_tokens):
        residue_esm_tokens *= self.residue_spatial_mask.int()
        residue_esm_tokens += self.residue_spatial_mask_esm.int()
    
        return residue_esm_tokens


class RandomMaskingModule(nn.Module):
    def __init__(self, 
                inf = 1e10):
        super().__init__()
        # self.max_p = max_p
    def forward(self, residue_mask, max_p, inverse_mask = False, stochastic =True):
        
        if stochastic:
            max_p = np.random.rand()*max_p
            # max_p = np.random.beta(8, 2, 1)[0]*max_p #pure experimental
        ones_mask = residue_mask == 1
        num_ones = ones_mask.sum().item()
        num_to_convert = int(num_ones * max_p)

        # Get the indices of the 1s and shuffle them
        one_indices = torch.where(ones_mask)
        shuffled_indices = torch.randperm(num_ones)[:num_to_convert]

        # Apply the conversion only to the selected 1s
        self.residue_rand_mask = residue_mask.clone()
        self.residue_rand_mask[one_indices[0][shuffled_indices], one_indices[1][shuffled_indices]] = 0

        self.residue_rand_mask_esm = 1 - residue_mask.detach().clone()
        self.residue_rand_mask_esm[one_indices[0][shuffled_indices], one_indices[1][shuffled_indices]] = 32

        if inverse_mask:
            self.residue_inv_rand_mask = torch.zeros_like(residue_mask)
            self.residue_inv_rand_mask[one_indices[0][shuffled_indices], one_indices[1][shuffled_indices]] = 1
            return self.residue_rand_mask, self.residue_inv_rand_mask, self.residue_rand_mask_esm
        else:
            return self.residue_rand_mask, self.residue_rand_mask_esm

    def mask_residue_esm(self, residue_esm_tokens):
        residue_esm_tokens *= self.residue_rand_mask.int()
        residue_esm_tokens += self.residue_rand_mask_esm.int()
    
        return residue_esm_tokens
    



