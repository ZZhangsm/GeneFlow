"""GeneFlow model definitions.

Portions of this implementation are adapted from Stem (MIT) and DiT
(CC BY-NC 4.0). See NOTICE in the repository root for attribution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar flow timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        return self.mlp(
            self.timestep_embedding(t, self.frequency_embedding_size)
        )


class GeneJointEmbedding(nn.Module):
    """Encode gene counts into a learned sequence of gene tokens."""

    def __init__(self, input_size, token_dim, hidden_dim):
        super().__init__()
        self.gene_name_ebd = nn.Parameter(
            torch.empty((token_dim, hidden_dim)), requires_grad=True
        )
        torch.nn.init.kaiming_uniform_(self.gene_name_ebd, a=math.sqrt(5))

        self.gene_reduction = nn.Sequential(
            nn.Linear(input_size, token_dim * hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        torch.nn.init.xavier_uniform_(self.gene_reduction[0].weight)

        self.hidden_dim = hidden_dim
        self.token_dim = token_dim

    def forward(self, x):
        batch_size = x.size(0)
        gene_count_ebd = self.gene_reduction(x)
        gene_count_ebd = gene_count_ebd.view(
            batch_size, self.token_dim, self.hidden_dim
        )
        return torch.add(gene_count_ebd, self.gene_name_ebd)


class MLP(nn.Module):
    """MLP block with squared ReLU activation."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.c_fc = nn.Linear(hidden_size, hidden_features, bias=False)
        self.c_proj = nn.Linear(hidden_features, hidden_size, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer normalization."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.RMSNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        modulation = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa = modulation[:3]
        shift_mlp, scale_mlp, gate_mlp = modulation[3:]
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Map token features back to the requested gene expression vector."""

    def __init__(self, output_size, token_dim, hidden_size):
        super().__init__()
        self.norm_final = nn.RMSNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size * token_dim, output_size, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x.reshape(x.size(0), -1))


class GeneFlowModel(nn.Module):
    """Histology-conditioned flow model for spatial gene expression."""

    def __init__(
        self,
        input_size=1000,
        token_dim=200,
        hidden_size=512,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        label_size=1152,
        **block_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.token_dim = token_dim

        self.gene_joint_embed = GeneJointEmbedding(
            input_size, token_dim, hidden_size
        )
        self.time_embed = TimestepEmbedder(hidden_size)
        self.label_embed = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.SiLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    **block_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(input_size, token_dim, hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.label_embed[0].weight, std=0.02)
        nn.init.normal_(self.label_embed[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, exp, t, img_features):
        x = self.gene_joint_embed(exp)
        timestep_embedding = self.time_embed(t)
        image_embedding = self.label_embed(img_features)
        condition = timestep_embedding + image_embedding
        for block in self.blocks:
            x = block(x, condition)
        return self.final_layer(x, condition)


def GeneFlow(**kwargs):
    return GeneFlowModel(**kwargs)


GeneFlow_models = {"GeneFlow": GeneFlow}
