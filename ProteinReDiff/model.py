from argparse import ArgumentParser, Namespace
from typing import Mapping, Union

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_ema import ExponentialMovingAverage

from .difffusion import get_betas

from .modules import (
    Linear,
    Denoiser,
    RadialBasisProjection,
    BondEmbedding,
    AtomEmbedding,
    SinusoidalProjection

)
from .protein import RESIDUE_TYPES
from .utils import (
    angstrom_to_nanometre,
    nanometre_to_angstrom,
    nearest_bin,
    pseudo_beta,
    remove_mean,
)

from .mask_utils import SpatialMaskingModule, RandomMaskingModule



    
class ProteinReDiffModel(pl.LightningModule):
    def __init__(self, args: Union[Namespace, Mapping]):
        super().__init__()
        if isinstance(args, Mapping):
            args = Namespace(**args)
        # self.no_cb_distogram = args.no_cb_distogram
        self.pair_dim = args.pair_dim
        self.single_dim = args.single_dim
        self.dist_dim = args.dist_dim
        self.time_dim = args.time_dim
        self.max_bond_distance = args.max_bond_distance
        self.max_relpos = args.max_relpos
        self.esm_dim = args.esm_dim
        
        self.setup_schedule = False
        self.setup_esm = False
        self.mask_prob = args.mask_prob
        # self.gamma_0 = args.gamma_0
        # self.gamma_1 = args.gamma_1
        self.num_steps = args.num_steps
        self.diffusion_schedule = args.diffusion_schedule
        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.ema_decay = args.ema_decay
        self.n_recycles = args.n_recycles
        self.training_mode = args.training_mode
            
            
        self.SpatialMaskingBlock = SpatialMaskingModule()
        self.RandomMaskingBlock = RandomMaskingModule()
        self.Denoiser = Denoiser(args)

        self.embed_atom_feats = AtomEmbedding(self.single_dim)
        self.embed_beta = nn.Sequential(
            SinusoidalProjection(self.time_dim),
            Linear(self.time_dim, self.pair_dim, bias=False, init="normal"),
        )
        # self.embed_residue_type = nn.Embedding(len(RESIDUE_TYPES)+1, self.single_dim) ## (NN) +1 to ignore gaps 0
        self.embed_residue_type = nn.Sequential(
            nn.LayerNorm(len(RESIDUE_TYPES)+1, elementwise_affine=False),
            Linear(len(RESIDUE_TYPES)+1, self.single_dim, bias = False, init = 'normal'),
            nn.ReLU()
        )
        self.embed_bond_feats = BondEmbedding(self.pair_dim)
        self.embed_bond_distance = nn.Embedding(
            self.max_bond_distance + 1, self.pair_dim
        )
        
        self.embed_residue_esm = nn.Sequential(
            nn.LayerNorm(self.esm_dim, elementwise_affine=False),
            Linear(self.esm_dim, self.single_dim, bias=False, init="normal"),
        )
        self.embed_relpos = nn.Embedding(self.max_relpos * 2 + 1, self.pair_dim)
        # self.embed_cb_distogram = nn.Embedding(39, self.pair_dim)
        # self.embed_ca_distogram = nn.Embedding(39, self.pair_dim)
        self.embed_dist = nn.Sequential(
            RadialBasisProjection(self.dist_dim),
            Linear(self.dist_dim, self.pair_dim, bias=False, init="normal"),
        )

        self.weight_radial = nn.Sequential(
            nn.LayerNorm(self.pair_dim, elementwise_affine=False),
            Linear(self.pair_dim, self.pair_dim, init="relu"),
            nn.ReLU(),
            Linear(self.pair_dim, 1, bias=False, init="final"),
        )

        
        self.seq_mlp = nn.Sequential(
            nn.LayerNorm(self.single_dim, elementwise_affine=False),
            Linear(self.single_dim, self.single_dim, init="relu"),
            nn.ReLU(),
            Linear(self.single_dim, len(RESIDUE_TYPES)+1, bias=False, init="final"),
        )
        

        # if self.training_mode:
        #     with torch.no_grad():
        #         self.esm_model, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        #         for param in self.esm_model.parameters():
        #             param.requires_grad = False
        #         self.esm_model.to(self.device).eval() #to(self.device)
            ## ignore esm params:
            # ema_parameters = [param for name, param in self.named_parameters() if not name.startswith("esm_model")]
        #     ema_parameters = list(self.parameters())[:-538]
            
        #     self.ema = ExponentialMovingAverage(ema_parameters, decay=self.ema_decay) #self.parameters()
        # else:
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay) #
        
        self.save_hyperparameters(args) ## TODO: Uncomment before training

        

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = ProteinReDiffModel.add_diffusion_args(parent_parser)
        parent_parser = ProteinReDiffModel.add_iterative_denoiser_args(parent_parser)
        return parent_parser
    
    @staticmethod
    def add_diffusion_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DiffusionModel")
        # parser.add_argument("--no_cb_distogram", action="store_true")
        parser.add_argument("--training_mode", action="store_true")
        parser.add_argument("--mask_prob", type=float, default=1.0)
        parser.add_argument("--esm_dim", type=int, default=1280)
        parser.add_argument("--time_dim", type=int, default=256)
        parser.add_argument("--dist_dim", type=int, default=256)
        parser.add_argument("--single_dim", type=int, default=512)
        parser.add_argument("--pair_dim", type=int, default=64)
        parser.add_argument("--head_dim", type=int, default=16)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--transition_factor", type=int, default=4)
        parser.add_argument("--num_blocks", type=int, default=12)
        parser.add_argument("--max_bond_distance", type=int, default=7)
        parser.add_argument("--max_relpos", type=int, default=32)
        parser.add_argument("--num_steps", type=int, default=64)
        parser.add_argument("--diffusion_schedule", type=str, default="linear")
        parser.add_argument("--learning_rate", type=float, default=4e-4)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        
        return parent_parser

    @staticmethod
    def add_iterative_denoiser_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("IterativeDenoiser")
        parser.add_argument("--n_recycles", type = int, default = 4)
        parser.add_argument("--top_k_neighbors", type = int, default = 30)
        parser.add_argument("--dropout", type = float, default = 0.3)
        parser.add_argument("--num_gvp_encoder_layers", type = int, default = 3)
        parser.add_argument("--num_positional_embeddings", type = int, default = 16)
        # parser.add_argument("--gvp_node_hidden_dim_scalar", type = int, default = 128)
        # parser.add_argument("--gvp_node_hidden_dim_vector", type = int, default = 64)
        parser.add_argument("--gvp_edge_hidden_dim_scalar", type = int, default = 32)
        parser.add_argument("--gvp_edge_hidden_dim_vector", type = int, default = 32)
        return parent_parser

    def run_setup_schedule(self):
        self.betas = get_betas(self.num_steps, self.diffusion_schedule).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]).to(self.device), self.alphas_cumprod[:-1]])
        self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1. - self.alphas_cumprod_prev

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * self.sqrt_alphas_cumprod_prev / self.one_minus_alphas_cumprod
        self.posterior_mean_coef2 = self.one_minus_alphas_cumprod_prev * self.sqrt_alphas / self.one_minus_alphas_cumprod
        self.posterior_variance = self.betas * self.one_minus_alphas_cumprod_prev / self.one_minus_alphas_cumprod

    def to(self, *args, **kwargs):
        out = torch._C._nn._parse_to(*args, **kwargs)
        self.ema.to(device=out[0], dtype=out[1])
        return super().to(*args, **kwargs)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler_config = {
            # "scheduler": torch.optim.lr_scheduler.LinearLR(
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / self.warmup_steps,
                total_iters=self.warmup_steps - 1,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # if self.training_mode:
        #     ema_parameters = list(self.parameters())[:-538]
        #     self.ema.update(ema_parameters)
        # else:
        self.ema.update(self.parameters())

    def load_esm(self):
        with torch.no_grad():
            self.esm_model, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
            for param in self.esm_model.parameters():
                param.requires_grad = False
            self.esm_model.to(self.device).eval() #to(self.device)
            
    def validation_step(self, batch, batch_idx):
        ## (NN)
        if not self.setup_schedule:
            self.run_setup_schedule()
            self.setup_schedule =True

        # if not self.setup_esm:
        #     if self.training_mode:
        #         self.load_esm()
        #     self.setup_esm = True

        batch = self.prepare_batch(batch, batch_idx)
        x = batch["x"]
        mask = batch["residue_and_atom_mask"]
        batch_size = x.size(0)
        num_nodes = einops.reduce(mask > 0.5, "b i -> b", "sum")
        # t_struc = torch.rand(batch_size, device=self.device)
        # t_seq = torch.rand(batch_size, device=self.device) ## (NN)
        t = torch.randint(0, self.num_steps, size=(batch_size,)).to(self.device)
        with self.ema.average_parameters(): ## (NN) for valudation only
            diff_loss= self.diffusion_loss(batch, x, mask, t) ## (NN)
        loss = torch.mean(diff_loss / num_nodes) ## (NN)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def predict_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            x, seq_t = self.sample(batch)
        return x, seq_t

    def forward(self, batch, z, seq_t, mask, t):
        atom_feats = batch["atom_feats"]
        atom_mask = batch["atom_mask"]
        bond_feats = batch["bond_feats"]
        bond_mask = batch["bond_mask"]
        bond_distance = batch["bond_distance"]
        residue_type = batch["residue_type"]
        residue_mask = batch["residue_mask"]
        # if self.training:
        residue_extra_mask = batch["residue_extra_mask"]
        residue_esm = batch["residue_esm"]
        residue_chain_index = batch["residue_chain_index"]
        residue_index = batch["residue_index"]
        residue_one_hot = batch["residue_one_hot"]
        residue_ca_pos = batch["residue_atom_pos"][:, :, 1]
        residue_type_masked = batch["residue_type_masked"]
        residue_inv_extra_mask = batch["residue_inv_extra_mask"]
        residue_atom_pos = batch["residue_atom_pos"]
        residue_atom_mask = batch["residue_atom_mask"]

        atom_mask_2d = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        residue_mask_2d = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)
        relpos = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        chain_mask = (
            residue_chain_index.unsqueeze(-1) == residue_chain_index.unsqueeze(-2)
        ).float()
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        zi_zj = z.unsqueeze(-2) - z.unsqueeze(-3)
        noise_dist = torch.linalg.norm(zi_zj, dim=-1)

        

        scaled_t = (t)/self.num_steps

        single = atom_mask.unsqueeze(-1) * self.embed_atom_feats(atom_feats)
        single += residue_mask.unsqueeze(-1) * (
                    self.embed_residue_type(seq_t) + ## (NN)
                    self.embed_residue_esm(residue_esm) ## (NN)
                )

        pair = atom_mask_2d.unsqueeze(-1) * (
            bond_mask.unsqueeze(-1) * self.embed_bond_feats(bond_feats)
            + self.embed_bond_distance(bond_distance.clamp(max=self.max_bond_distance))
        )
        
        pair += residue_mask_2d.unsqueeze(-1) * (
            chain_mask.unsqueeze(-1)
            * self.embed_relpos(
                self.max_relpos
                + relpos.clamp(min=-self.max_relpos, max=self.max_relpos)
            )

        ) 
        
        pair += mask_2d.unsqueeze(-1) * (
            self.embed_dist(noise_dist) + self.embed_beta(scaled_t[:, None, None]) 
        )
        


        cache = None
        single, pair, cache = self.Denoiser(batch, residue_ca_pos, t, single, pair, cache)
        
        
        
        w = self.weight_radial(pair)
        r = zi_zj * torch.rsqrt(
            torch.sum(torch.square(zi_zj), -1, keepdim=True) + 1e-4
        )
        noise_pred = einops.reduce(
            mask_2d.unsqueeze(-1) * w * r,
            "b i j xyz -> b i xyz",
            "sum",
        )
        noise_pred = remove_mean(noise_pred, mask)

        seq_pred = self.seq_mlp(single)

        return noise_pred, seq_pred
    
    def sample_step(self, batch, z, seq_t, mask, t):
        atom_feats = batch["atom_feats"]
        atom_mask = batch["atom_mask"]
        bond_feats = batch["bond_feats"]
        bond_mask = batch["bond_mask"]
        bond_distance = batch["bond_distance"]
        residue_type = batch["residue_type"]
        residue_mask = batch["residue_mask"]
        # if self.training:
        residue_extra_mask = batch["residue_extra_mask"]
        residue_esm = batch["residue_esm"]
        residue_chain_index = batch["residue_chain_index"]
        residue_index = batch["residue_index"]
        residue_one_hot = batch["residue_one_hot"]
        residue_ca_pos = batch["residue_atom_pos"][:, :, 1]


        atom_mask_2d = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        residue_mask_2d = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)
        relpos = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        chain_mask = (
            residue_chain_index.unsqueeze(-1) == residue_chain_index.unsqueeze(-2)
        ).float()
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        zi_zj = z.unsqueeze(-2) - z.unsqueeze(-3)
        noise_dist = torch.linalg.norm(zi_zj, dim=-1)

        

        scaled_t = (t)/self.num_steps

        single = atom_mask.unsqueeze(-1) * self.embed_atom_feats(atom_feats)
        single += residue_mask.unsqueeze(-1) * (
                    self.embed_residue_type(seq_t) + ## (NN)
                    self.embed_residue_esm(residue_esm) ## (NN)
                )

        pair = atom_mask_2d.unsqueeze(-1) * (
            bond_mask.unsqueeze(-1) * self.embed_bond_feats(bond_feats)
            + self.embed_bond_distance(bond_distance.clamp(max=self.max_bond_distance))
        )
        
        pair += residue_mask_2d.unsqueeze(-1) * (
            chain_mask.unsqueeze(-1)
            * self.embed_relpos(
                self.max_relpos
                + relpos.clamp(min=-self.max_relpos, max=self.max_relpos)
            )

        ) 
        
        pair += mask_2d.unsqueeze(-1) * (
            self.embed_dist(noise_dist) + self.embed_beta(scaled_t[:, None, None]) 
        )
        
        cache = None
        single, pair, cache = self.Denoiser(batch, residue_ca_pos, t, single, pair, cache)
        
        w = self.weight_radial(pair)
        r = zi_zj * torch.rsqrt(
            torch.sum(torch.square(zi_zj), -1, keepdim=True) + 1e-4
        )
        noise_pred = einops.reduce(
            mask_2d.unsqueeze(-1) * w * r,
            "b i j xyz -> b i xyz",
            "sum",
        )
        noise_pred = remove_mean(noise_pred, mask)

        seq_pred = self.seq_mlp(single)

        return noise_pred, seq_pred

    @torch.inference_mode()
    def sample(self, batch):
        if not self.setup_schedule: ##(NN)
            self.run_setup_schedule()
            self.setup_schedule = True

        if not self.setup_esm:
            if self.training_mode:
                self.load_esm()
            self.setup_esm = True
                        
        batch = self.prepare_batch(batch)
        x = batch["x"]
        mask = batch["residue_and_atom_mask"]
        residue_mask = batch["residue_mask"]
        seq = batch["residue_one_hot"]
        residue_extra_mask = batch["residue_extra_mask"]
        residue_inv_extra_mask = batch["residue_inv_extra_mask"]
        batch_size = x.size(0)
        time_steps = torch.linspace(
            self.num_steps-1, 0, steps=self.num_steps, device=self.device
        ).long()

        z_struc_t = remove_mean(torch.randn_like(x), mask)
        seq_t = remove_mean(torch.randn_like(seq), residue_mask)
        seq_t = residue_extra_mask.unsqueeze(-1) * seq + residue_inv_extra_mask.unsqueeze(-1) * seq_t

        for i in range(self.num_steps):

            t = torch.broadcast_to(time_steps[i], (batch_size,))
            
            # [b, 1, 1]
            w_noise = ((1. - self.alphas[t].to(self.device)) / self.sqrt_one_minus_alphas_cumprod[t].to(self.device))
            
            noise_pred, seq_pred = self.sample_step(batch, z_struc_t, seq_t, mask, t) 
            mean = (1. / self.sqrt_alphas[t])[:, None, None].to(self.device) * (
                 z_struc_t - w_noise[:, None, None] * noise_pred
             )
            
            seq_t = torch.softmax(seq_pred, dim=-1) * 2 - 1
            if (t == 0).all():
                z_struc_t = mean
            else:
                noise = remove_mean(torch.randn_like(x), mask)
                std = self.sqrt_betas[t][:,None, None].to(self.device)
                z_struc_t = mean + std * noise

        pos = nanometre_to_angstrom(z_struc_t)
        
        return pos, residue_mask.unsqueeze(-1) * seq_pred

    def prepare_batch(self, batch, id=None):

        atom_pos = batch["atom_pos"]
        atom_mask = batch["atom_mask"]
        residue_ca_pos = batch["residue_atom_pos"][:, :, 1]
        residue_mask = batch["residue_mask"]
        residue_type = batch["residue_type"]
        

        batch["residue_one_hot"] = F.one_hot(residue_type, num_classes = len(RESIDUE_TYPES)+1) * 2. - 1.
        
        pos = (
            atom_mask.unsqueeze(-1) * atom_pos
            + residue_mask.unsqueeze(-1) * residue_ca_pos
        )
        x = angstrom_to_nanometre(pos)
        mask = atom_mask + residue_mask #both mask

        # if self.training_mode:
        #     residue_spatial_mask, residue_spatial_mask_esm = self.SpatialMaskingBlock(residue_ca_pos, residue_mask,
        #                                                     atom_pos, atom_mask)
            # esm_tokens = self.SpatialMaskingBlock.mask_residue_esm(residue_esm_tokens)
            
            
            # with torch.no_grad():
            #     results = self.esm_model(esm_tokens.to(self.device), repr_layers=[self.esm_model.num_layers])
            #     token_representations = results["representations"][self.esm_model.num_layers] 
            #     batch["residue_esm"] = token_representations * residue_mask.unsqueeze(-1)
            

            # batch["residue_esm"] = batch["residue_esm"] * residue_spatial_mask.unsqueeze(-1)
            # spatial_mask = atom_mask + residue_spatial_mask
            # batch["residue_spatial_mask"] = residue_spatial_mask
            # batch["residue_and_atom_spatial_mask"] = spatial_mask
            # batch["residue_type_masked"] = (residue_type * residue_spatial_mask).long()
            # batch["residue_atom_pos"] = residue_spatial_mask[:,:,None,None]* batch["residue_atom_pos"]
            # batch["residue_one_hot"] = batch["residue_one_hot"] * residue_spatial_mask[:,:,None]
        # else:
        #     batch["residue_type_masked"] = batch["residue_type"]

        if self.training_mode:
            residue_esm_tokens = batch["residue_esm_tokens"]
            # torch.manual_seed(id)
            rt = torch.rand(1)
            mask_prob = np.random.uniform(0.1,self.mask_prob)
            if rt< 0.3:
                residue_extra_mask, residue_inv_extra_mask, residue_extra_mask_esm = self.RandomMaskingBlock(residue_mask, mask_prob, inverse_mask = True)
                esm_tokens = self.RandomMaskingBlock.mask_residue_esm(residue_esm_tokens)
            elif (rt>=0.3) & (rt<0.5):
                residue_extra_mask, residue_inv_extra_mask, residue_extra_mask_esm = self.SpatialMaskingBlock(residue_ca_pos, residue_mask,
                                                            atom_pos, atom_mask, mask_prob, inverse_mask = True)
                esm_tokens = self.SpatialMaskingBlock.mask_residue_esm(residue_esm_tokens)
                
            else:
                # residue_extra_mask = residue_mask.clone()
                residue_extra_mask, residue_inv_extra_mask, residue_extra_mask_esm = self.RandomMaskingBlock(residue_mask, 0., inverse_mask = True)

            # if rt<0.5:

                # with torch.set_grad_enabled(False):
                #     results = self.esm_model(esm_tokens.to(self.device), repr_layers=[self.esm_model.num_layers])  #
                #     token_representations = results["representations"][self.esm_model.num_layers] 
                # batch["residue_esm"] = (token_representations * residue_extra_mask.unsqueeze(-1)).requires_grad_(True)
            batch["residue_esm"] = batch["residue_esm"] * residue_extra_mask.unsqueeze(-1)
            # spatial_mask = atom_mask + residue_spatial_mask
            # batch["residue_spatial_mask"] = residue_spatial_mask
            # batch["residue_and_atom_spatial_mask"] = spatial_mask
            batch["residue_type_masked"] = (residue_type * residue_extra_mask).long()
            batch["residue_one_hot"] = batch["residue_one_hot"] * residue_extra_mask.unsqueeze(-1)
        else:
            residue_extra_mask, residue_inv_extra_mask, residue_extra_mask_esm = self.RandomMaskingBlock(residue_mask, self.mask_prob, inverse_mask = True, stochastic = False)
            batch["residue_esm"] = batch["residue_esm"] * residue_extra_mask.unsqueeze(-1)
            batch["residue_type_masked"] = (residue_type * residue_extra_mask).long()
            batch["residue_one_hot"] = batch["residue_one_hot"] * residue_extra_mask.unsqueeze(-1)


        batch["residue_extra_mask"] = residue_extra_mask
        batch["residue_inv_extra_mask"] = residue_inv_extra_mask
        batch["x"] = x
        batch["residue_and_atom_mask"] = mask
        
        
        
        return batch
    
    ## TODO: Consider to do random stochastic for beta_t
    def gamma(self, t):
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * t

    ### (NN) Add noise to structure, sequence
    def add_noise(self, x, mask, t): 
        with torch.enable_grad():
            t = t.clone().detach().requires_grad_(True)
            gamma_t = self.gamma(t)
            if x.size(-1)==3: ## (NN) only require grad for structure: TODO check diffusion grad
                grad_gamma_t = torch.autograd.grad(gamma_t.sum(), t, create_graph=True)[0]
            else:
                grad_gamma_t = None
        gamma_t = gamma_t.detach()
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        noise = remove_mean(torch.randn_like(x), mask)
        z_t = alpha_t.view([-1] + [1]*x.size(0)) * x \
                + sigma_t.view([-1] + [1]*x.size(0)) * noise
        return z_t, gamma_t, noise, grad_gamma_t
    ###

    def q(self, x, seq, t, noise_z, noise_seq, batch):
        """
        (NN) (TODONE:q process in Genie)
        Forward noising step on structure x
        """
        
        #noise x
        residue_inv_extra_mask = batch["residue_inv_extra_mask"]
        residue_extra_mask = batch["residue_extra_mask"]

        
        z_t = self.sqrt_alphas_cumprod[t][:,None, None].to(self.device) * x + \
			self.sqrt_one_minus_alphas_cumprod[t][:,None, None].to(self.device) * noise_z

        
        seq_t = self.sqrt_alphas_cumprod[t][:,None, None].to(self.device) * seq + \
			self.sqrt_one_minus_alphas_cumprod[t][:,None, None].to(self.device) * noise_seq ## (NN) used to be 1 None
        seq_t = residue_extra_mask.unsqueeze(-1) * seq + residue_inv_extra_mask.unsqueeze(-1) * seq_t

        t1 = (t - 1).clamp(min = 0)
        seq_t1 = self.sqrt_alphas_cumprod[t1][:,None, None].to(self.device) * seq + \
			self.sqrt_one_minus_alphas_cumprod[t1][:,None, None].to(self.device) * noise_seq ## (NN)


        return z_t, seq_t, seq_t1, t1

    def diffusion_loss(self, batch, x, mask, t):
        
        seq = batch["residue_one_hot"]
        residue_mask = batch["residue_mask"]
        residue_inv_extra_mask = batch["residue_inv_extra_mask"]
        noise_z = remove_mean(torch.randn_like(x), mask)
        noise_seq = remove_mean(torch.randn_like(seq), residue_mask)
        z_t, seq_t, seq_t1, t1 = self.q(x, seq, t, noise_z, noise_seq, batch) 
        noise_pred, seq_pred = self(batch, z_t, seq_t, mask, t) # (NN): not noising sequence

        # noise_seq_t1 = remove_mean(torch.randn_like(seq), residue_mask)
        seq_pred_t1 = self.sqrt_alphas_cumprod[t1][:,None, None].to(self.device) * seq_pred + \
			self.sqrt_one_minus_alphas_cumprod[t1][:,None, None].to(self.device) * noise_seq #noise_seq_t1

        

        diff_loss = (
            1
            # 0.5
            # * grad_gamma_t 
            * einops.reduce(
                mask.unsqueeze(-1) * torch.square(noise_pred - noise_z),
                "b i xyz -> b",
                "sum",
            )
        )
        # torch.log_softmax(output, dim=1), torch.softmax(target, dim=1)
        diff_loss += ( 
                        F.kl_div(
                                torch.log_softmax(seq_pred_t1, dim = -1) * residue_mask.unsqueeze(-1),
                                torch.softmax(seq_t1, dim = -1) * residue_mask.unsqueeze(-1),
                                reduction = "none"
                            ) #* mask.unsqueeze(-1)
                            ).sum()
        
        seq_pred = (seq_pred + 1)/2
        diff_loss += ( 
                        F.cross_entropy(seq_pred.view(-1, len(RESIDUE_TYPES)+1), batch["residue_type"].view(-1),  
                                     reduction="none", ignore_index= 0 
                                     )* mask.view(-1)
                                     ).sum() ## (NN); index of 0 
        return diff_loss 

    def training_step(self, batch, batch_idx):
        if not self.setup_schedule:
            self.run_setup_schedule()
            self.setup_schedule =True

        # if not self.setup_esm:
        #     self.run_setup_schedule()
        #     self.setup_schedule =True
        
        batch = self.prepare_batch(batch, batch_idx)
        x = batch["x"]
        mask = batch["residue_and_atom_mask"]
        batch_size = x.size(0)
        num_nodes = einops.reduce(mask > 0.5, "b i -> b", "sum")
        t = torch.randint(0, self.num_steps, size=(batch_size,)).to(self.device)
        diff_loss= self.diffusion_loss(batch, x, mask, t) ##(NN)
        loss = torch.mean(diff_loss / num_nodes)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss