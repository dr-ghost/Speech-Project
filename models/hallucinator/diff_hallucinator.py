import torch
from torch import nn, einsum
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from inspect import isfunction
from functools import partial

from ._transformers import *
from .data import hallu_collate_fn

from sklearn.model_selection import train_test_split

from chamferdist import ChamferDistance
from geomloss import SamplesLoss

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Upsample(in_dim, out_dim=None, scale_factor=None) -> nn.Module:
    assert not(out_dim is None and scale_factor is None)
    
    if scale_factor is None:
        scale_factor = out_dim // in_dim
    
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode="nearest"),
        nn.Linear(scale_factor * in_dim, default(out_dim, scale_factor * in_dim))
    )


def Downsample(in_dim, out_dim=None, scale_factor=None) -> nn.Module:
    assert not(out_dim is None and scale_factor is None)
    
    if scale_factor is None:
        scale_factor = in_dim // out_dim
        
    return nn.Sequential(
        Reduce('b ls (n1 embed_n) -> b ls embed_n', 'mean',n1=scale_factor),
        nn.Linear(in_dim // scale_factor, default(out_dim, in_dim // scale_factor))
    )
    
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    
class UnetT(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_mults: tuple = (1, 2, 4, 8),
        n_mid_blocks: int = 2,
        self_condition=False
    ) -> None:
        super().__init__()
        
        self.self_condition = self_condition
        
        self.init_block = SelfAttentionBlock(dim, dim)
        
        dims = [dim, *map(lambda m: dim // m, dim_mults)]
        
        in_outs = list(zip(dims[:-1], dims[1:]))
        
        # print(in_outs)
        
        time_dim = dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mids = nn.ModuleList([])
        
        
        for idx, (in_dim, out_dim) in enumerate(in_outs):
            self.downs.append(
                nn.ModuleList([
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(in_dim, dim, dim, 4*dim, nheads=4, dropout=0.0)), # cross-attention with set embeddings
                    Downsample(in_dim, out_dim)
                ])
            )
        
        mid_dim = dims[-1]
        
        for idx in range(n_mid_blocks):
            self.mids.append(
                nn.ModuleList([
                    TransformerBlock(mid_dim, mid_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(mid_dim, dim, dim, 4*dim, nheads=4, dropout=0.0)), # cross-attention with set embeddings
                ])
            )
        
        
        for idx, (out_dim, in_dim) in enumerate(reversed(in_outs)):
            self.ups.append(
                nn.ModuleList([
                    Residual(MultiHeadAttention(in_dim, out_dim, out_dim, 4*out_dim, nheads=4, dropout=0.0)), # cross-attention in place of skip connections
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(in_dim, out_dim, out_dim, 4*out_dim, nheads=4, dropout=0.0)), # cross-attention in place of skip connections
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(in_dim, dim, dim, 4*dim, nheads=4, dropout=0.0)), # cross-attention with set embeddings
                    Upsample(in_dim, out_dim)
                ])
            )
            
        self.out_dim = dim
        self.out_skip = Residual(MultiHeadAttention(list(reversed(in_outs))[-1][0], dim, dim, 4*dim, nheads=4, dropout=0.0)) # cross-attention in place of skip connections
        self.out_block = TransformerBlock(list(reversed(in_outs))[-1][0], self.out_dim, time_embed_dim=time_dim)
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, cross_set: torch.Tensor = None, cross_mask: torch.Tensor = None, x_self_cond=None, return_nested=True) -> torch.Tensor:
        """
        x: nested tensor
        time: int
        cross_set: nested tensor
        cross_mask: nested tensor
        """
        
        device = self.parameters().__next__().device
        
        x_ = x.to_padded_tensor(0.0) # shape (batch, max_seq_len, embed_dim)
        lens = torch.tensor([sample.shape[0] for sample in x.unbind(0)], device=device)  # (batch,)
            
        seq_range = torch.arange(x_.shape[1], device=device).unsqueeze(0)  # shape (1, max_seq_len)
        mask_seq = seq_range < lens.unsqueeze(1) # shape (batch, max_seq_len)
            
        mask_seq = repeat(mask_seq, 'b max_seql -> b max_seql max_dim', max_dim=max(x_.shape[-1], x_.shape[-2]))
            
        x = x_
        assert not((cross_set is None) ^ (cross_mask is None))
            
        mask_set = None
        if cross_set is not None:
            cross_set = cross_set.to_padded_tensor(0.0)
            cross_mask = cross_mask.to_padded_tensor(False)
                
            mask_set = repeat(cross_mask, 'b max_setl -> b max_seql max_setl', max_seql=x.shape[1])
            
        if self.self_condition:
            pass
            
        x = self.init_block(x, mask=mask_seq)
        r = x.clone()
            
        t = self.time_mlp(time)
            
        h = []
            
        for _block1, _block2, _cattn, _downsample in self.downs:
            x = _block1(x, t, mask_seq)
                
            h.append(x)
                
            x = _block2(x, t, mask_seq)
                
            # print("#")
            if cross_set is not None:
                x = _cattn(x, cross_set, cross_set, attn_mask=mask_set)
            
            h.append(x)
            
            x = _downsample(x)
            # print("!")
            
        # print("!!")
        for _block1, _cattn in self.mids:
            x = _block1(x, t, mask_seq)
                
            if cross_set is not None:
                x = _cattn(x, cross_set, cross_set, attn_mask=mask_set)
                
        # print("!!")
        
        for  _skipattn1, _block1, _skipattn2, _block2, _cattn1, _upsample in self.ups:
            
            skip_x = h.pop()
            x = _skipattn1(x, skip_x, skip_x, attn_mask=mask_seq[:, :, :mask_seq.shape[1]].transpose(1, 2))
            
            x = _block1(x, t, mask_seq)
            
            skip_x = h.pop()
            x = _skipattn2(x, skip_x, skip_x, attn_mask=mask_seq[:, :, :mask_seq.shape[1]].transpose(1, 2))
            
            x = _block2(x, t, mask_seq)
            
            if cross_set is not None:
                x = _cattn1(x, cross_set, cross_set, attn_mask=mask_set)
                      
            x = _upsample(x)
        
        x = self.out_skip(x, r, r, attn_mask=mask_seq[:, :, :mask_seq.shape[1]].transpose(1, 2))
        x = self.out_block(x, t, mask_seq)
                
        
        if return_nested:
            nested_x = torch.nested.as_nested_tensor(
                [x[i, :lens[i]] for i in range(len(lens))],
                dtype=x.dtype,
                device=x.device,
                layout=torch.jagged
            )
            
            return nested_x
        else:
            return x
    

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cum_prod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cum_prod = alphas_cum_prod / alphas_cum_prod[0]
    betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class SetDDPM:
    def __init__(self, T_timesteps: int, schedule: callable, unet: nn.Module=None, device: torch.DeviceObjType=None) -> None:
        self.T = T_timesteps
        self.schedule = schedule
        
        self.betas = self.schedule(timesteps=self.T)
        
        self.alphas = 1.0 - self.betas
        
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cum_prod_prev = F.pad(self.alphas_cum_prod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1.0 - self.alphas_cum_prod)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cum_prod_prev) / (1.0 - self.alphas_cum_prod)
        
        if unet is None:
            assert device is not None
            self.device = device
            self.unet = UnetT(dim=1024, dim_mults=(1, 2, 4, 8), n_mid_blocks=2).to(self.device)
        else:
            self.device = device if device is not None else unet.device
            self.unet = unet.to(self.device)
        
        self.dim = self.unet.out_dim
        
    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        a = repeat(a, 'd -> b d', b=batch_size)
        
        out = a[torch.arange(a.size(0)), t.cpu() - 1]
        
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cum_prod_t = self.extract(self.sqrt_alphas_cum_prod, t, x_0.shape)
        sqrt_one_minus_alphas_cum_prod_t = self.extract(self.sqrt_one_minus_alphas_cum_prod, t, x_0.shape)
        
        return sqrt_alphas_cum_prod_t * x_0 + sqrt_one_minus_alphas_cum_prod_t * noise
        
    def p_loss(self, x_0: torch.Tensor, cross_set: torch.Tensor, cross_mask: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None, loss_type='l1') -> torch.Tensor:
        x_padd = x_0.to_padded_tensor(0.0)
        
        lens = torch.tensor([sample.shape[0] for sample in x_0.unbind(0)], device=self.device)  # (batch,)
            
        seq_range = torch.arange(x_padd.shape[1], device=self.device).unsqueeze(0)  # shape (1, max_seq_len)
        
        mask_ = seq_range >= lens.unsqueeze(1) # shape (batch, max_seq_len)
        mask_.unsqueeze_(-1)
        
        
        if noise is None:
            noise = torch.randn_like(x_padd)
            
        x_t = self.q_sample(x_padd, t, noise)
        
        #x_t to nested tensor
        x_t = torch.nested.as_nested_tensor(
            [x_t[i, :lens[i]] for i in range(len(lens))],
            dtype=x_0.dtype,
            device=x_0.device,
            layout=torch.jagged
        )
        
        noise_pred = self.unet(x_t, t, cross_set=cross_set, cross_mask=cross_mask, return_nested=False)
        noise.masked_fill_(mask_, 0.0)
        
        if loss_type == 'l1':
            return F.l1_loss(noise, noise_pred)
        elif loss_type == 'l2':
            return F.mse_loss(noise, noise_pred)
        elif loss_type == "huber":
            return F.smooth_l1_loss(noise, noise_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, cross_set: torch.Tensor, cross_mask: torch.Tensor, t: torch.Tensor, t_index):
        x_padd = x_t.to_padded_tensor(0.0)
        
        lens = torch.tensor([sample.shape[0] for sample in x_t.unbind(0)], device=self.device)  # (batch,)
            
        # seq_range = torch.arange(x_padd.shape[1], device=self.device).unsqueeze(0)  # shape (1, max_seq_len)
        # mask_ = seq_range >= lens.unsqueeze(1) # shape (batch, max_seq_len)
        # mask_.unsqueeze_(-1)
        
        betas_t = self.extract(self.betas, t, x_padd.shape)
        sqrt_one_minus_alphas_cum_prod_t = self.extract(self.sqrt_one_minus_alphas_cum_prod, t, x_padd.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x_padd.shape)
        
        

        model_mean = sqrt_recip_alphas_t * (
            x_padd - betas_t * self.unet(x_t, t, cross_set=cross_set, cross_mask=cross_mask, return_nested=False) / sqrt_one_minus_alphas_cum_prod_t
        )

        if t_index == 0:
            sample = torch.nested.as_nested_tensor(
                [model_mean[i, :lens[i]] for i in range(len(lens))],
                dtype=model_mean.dtype,
                device=model_mean.device,
                layout=torch.jagged
            )
            
            return sample
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x_padd.shape)
            noise = torch.randn_like(x_padd)
            # Algorithm 2 line 4:
            sample = model_mean + torch.sqrt(posterior_variance_t) * noise
            
            sample = torch.nested.as_nested_tensor(
                [sample[i, :lens[i]] for i in range(len(lens))],
                dtype=sample.dtype,
                device=sample.device,
                layout=torch.jagged
            )
            
            return sample
    
        
    @torch.no_grad()
    def sample_batch(self, n_batch: int, n_samples: list, cross_set: torch.Tensor, cross_mask: torch.Tensor) -> torch.Tensor:
        batch_tensors = [
            torch.randn(n, self.dim, device=self.device) 
            for n in n_samples
        ]
    
        x_T = torch.nested.nested_tensor(batch_tensors, device=self.device, dtype=batch_tensors[0].dtype, layout=torch.jagged)
        
        x_t = x_T
        x_t1 = None
        
        cross_set = cross_set.to(self.device)
        cross_mask = cross_mask.to(self.device)
        
        for it in tqdm(range(self.T, 0, -1), desc='sampling loop time step', total=self.T):
            x_t1 = self.p_sample(x_t, cross_set=cross_set, cross_mask=cross_mask, t=torch.full((n_batch,), it, device=self.device, dtype=torch.long), t_index=it)
            
            # Visualiztions and shit
            
            x_t = x_t1
        return x_t
    
    def train(self, n_epochs: int, dataset: Dataset, batch_size: int=64, model_save_path: str=None, lr: float = 1e-3) -> list:
        train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.01, random_state=42)
        train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=hallu_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=hallu_collate_fn)
        
        opt = optim.AdamW(self.unet.parameters(), lr=lr)
        
        train_loss = []
        val_cad = []
        val_eml = []
        
        self.chamfer_dist = ChamferDistance()
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
        
        for epoch in range(n_epochs):
            t_l = 0.0
            n_t = 0
            
            print(f"epoch: {epoch}")
            for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                opt.zero_grad()
                self.unet.zero_grad()
                
                x_0 = batch['masked_out_set'].to(self.device)
                cross_set = batch['total_speech_set'].to(self.device)
                cross_mask = batch['mask'].to(self.device)
                
                t = torch.randint(0, self.T, (x_0.shape[0],), device=self.device)
                
                loss = self.p_loss(x_0, cross_set=cross_set, cross_mask=cross_mask, t=t)
                
                loss.backward()
                opt.step()
                
                if idx % 10_000 == 0:
                    pass
                
                t_l += loss.detach().cpu().item()
                n_t += x_0.shape[0]
                
            train_loss.append(t_l / n_t)
            
            if epoch % 5 == 0:
                with torch.no_grad():
                    n_val = 0
                    t_eml = 0.0
                    t_cad = 0.0
                    for idx, batch in enumerate(val_dataloader):
                        
                        x_0 = batch['masked_out_set'].to(self.device)
                        cross_set = batch['total_speech_set'].to(self.device)
                        cross_mask = batch['mask'].to(self.device)
                        
                        x_0_pred = self.sample_batch(x_0.shape[0], [x_0[i, ...].shape[0] for i in range(x_0.shape[0])], cross_set=cross_set, cross_mask=cross_mask) 

                        
                        for i_ in range(x_0.shape[0]):
                            t_eml += self.sinkhorn_loss(x_0[i_, ...], x_0_pred[i_, ...]).detach().cpu().item()
                            t_cad += self.chamfer_dist(x_0[i_, ...].unsqueeze(0), x_0_pred[i_, ...].unsqueeze(0)).detach().cpu().item()
                        n_val += x_0.shape[0]
                        
                    val_eml.append(t_eml / n_val)
                    val_cad.append(t_cad / n_val)
        
        
        if model_save_path is not None:
            torch.save(self.unet.state_dict(), model_save_path)
        return train_loss, val_eml, val_cad
                    
        
        
        
        