import math

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import numpy as np
# from .attention_core import attention_core
from typing import Optional, Callable, List, Tuple, Sequence

DEFAULT_LMA_Q_CHUNK_SIZE=1024
DEFAULT_LMA_KV_CHUNK_SIZE=4096

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))

def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")
                    
class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()
        
        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x): 
        d = x.dtype
        
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out
    
def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")



class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor, 
        kv_x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor, 
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if(self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))
        
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE,
        use_flash: bool = False,
        flash_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch 
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if(use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None)):
            raise ValueError(
                "If use_lma is specified, lma_q_chunk_size and "
                "lma_kv_chunk_size must be provided"
            )

        if(use_flash and biases is not None):
            raise ValueError(
                "use_flash is incompatible with the bias option. For masking, "
                "use flash_mask instead"
            )

        attn_options = [use_memory_efficient_kernel, use_lma, use_flash]
        if(sum(attn_options) > 1):
            raise ValueError(
                "Choose at most one alternative attention algorithm"
            )

        if(biases is None):
            biases = []
        
        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, Q, H, C_hidden]
        
        biases = [
            b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],)) 
            for b in biases
        ]
        
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o

class SPAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(SPAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)
        
        # self.layer_norm_z = None
        # self.linear_z = None
        if self.pair_bias:
            self.linear_z = nn.Sequential( ## (NN)
                LayerNorm(self.c_z),
                Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
            )
            # self.linear_z_rev = nn.Sequential(
            #     LayerNorm(self.no_heads),
            #     Linear(
            #     self.no_heads, self.c_z, bias=False, init="normal"
            # ))
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )
        
    def forward(self, 
                    m: torch.Tensor, 
                    z: Optional[torch.Tensor] = None, 
                    mask: Optional[torch.Tensor] = None):
            
        """
        Args:
            m:
                 Single representation embedding (MSA row)
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] Single representation mask
        """
        
        
        n_seq, n_res = m.shape[-3:-1]

        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )
            
        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        
        if (self.pair_bias and 
            z is not None and                       # For the 
            # self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            chunks = []

            # for i in range(0, z.shape[-3], 128): ## (NN)
            #     z_chunk = z[..., i: i + 128, :, :]

            #     # [*, N_res, N_res, C_z]
            #     z_chunk = self.layer_norm_z(z_chunk)
            
            #     # [*, N_res, N_res, no_heads]
            #     z_chunk = self.linear_z(z_chunk)

            #     chunks.append(z_chunk)

            # z = torch.cat(chunks, dim=-3)#.mean(0)

            # z = z + self.layer_norm_z(z)
            z = self.linear_z(z)#.mean(0)
            
            # b_ij^h [*, 1, no_heads, N_res, N_res]
            # z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

            # mask_bias [*, N_seq, 1, 1, N_res] 
            # b_ij^h [*, N_seq, no_heads, N_res, N_res]
            z = permute_final_dims(z, (0, 3, 1, 2))

        biases = [] #[mask_bias] ## (NN)
        if(z is not None):
            biases.append(z)
                
        m = self.layer_norm_m(m) ## (NN)
        m = m + self.mha( ## (NN)
            q_x=m, 
            kv_x=m, 
            biases=biases
        )

        #reverse
        # z = permute_final_dims(z, (0, 2, 3, 1))
        # z = self.linear_z_rev(z)
        return m
            
            
class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")
        
    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] Single representation embedding
            mask:
                [*, N_seq, N_res] Single representation mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) 
        a = a * mask
        
        b = self.linear_2(ln) 
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        outer = outer / norm

        return outer
    
class OuterProductUpdate(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden, c_z, init="final")
        
    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] Single representation embedding
            mask:
                [*, N_seq, N_res] Single representation mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) 
        a = a * mask
        
        b = self.linear_2(ln) 
        b = b * mask

        del ln

        # a = a.transpose(-2, -3)
        # b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        # outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # outer = torch.einsum("...abc,...ade->...abdce", a, b)
        outer = torch.einsum("...abc, ...adc->...abdc", a,b)

        # [*, N_res, N_res, C * C]
        # outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # [*, N_res, N_res, 1]
        # norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = torch.einsum("...abc,...adc->...abdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        outer = outer / norm

        return outer

class OuterProductMeanModified(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")
        
    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] Single representation embedding
            mask:
                [*, N_seq, N_res] Single representation mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) 
        a = a * mask
        
        b = self.linear_2(ln) 
        b = b * mask

        del ln

        # a = a.transpose(-2, -3)
        # b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        # outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = torch.einsum("...abc,...ade->...abdce", a, b)
        # outer = torch.einsum("...abc, ...adc->...abdc", a,b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # [*, N_res, N_res, 1]
        # norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = torch.einsum("...abc,...adc->...abdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        outer = outer / norm

        return outer
        
        
def _lma(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    biases: List[torch.Tensor], 
    q_chunk_size: int, 
    kv_chunk_size: int,
):
    ## Local Memory Attention
    
    no_q, no_kv = q.shape[-2], k.shape[-2]

    # [*, H, Q, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s: q_s + q_chunk_size, :]
        large_bias_chunks = [
            b[..., q_s: q_s + q_chunk_size, :] for b in biases
        ]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s: kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s: kv_s + kv_chunk_size, :]
            small_bias_chunks = [
                b[..., kv_s: kv_s + kv_chunk_size] for b in large_bias_chunks
            ]

            a = torch.einsum(
                "...hqd,...hkd->...hqk", q_chunk, k_chunk,
            )
       
            for b in small_bias_chunks:
                a += b
        
            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...hvf,...hqv->...hqf", v_chunk, exp_a)
 
            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s: q_s + q_chunk_size, :] = q_chunk_out

    return o

def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = torch.nn.functional.softmax(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a