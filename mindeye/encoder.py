import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class BidirectionalMamba(nn.Module):

    def __init__(self, features, state_dim=16, expand=2, **kwargs):
        super().__init__()

        mamba_kwargs = dict(d_state=state_dim, expand=(expand / 2), **kwargs)
        self.mamba_fwd = Mamba(features, **mamba_kwargs)
        self.mamba_bwd = Mamba(features, **mamba_kwargs)

    def forward(self, input):
        out_fwd = self.mamba_fwd(input)
        out_bwd = self.mamba_bwd(input.flip(-2)).flip(-2)
        return out_fwd + out_bwd


class CrossAttentionPooling(nn.Module):

    def __init__(self, features, L, num_heads):
        assert features % num_heads == 0
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = features // num_heads

        self.q = nn.Parameter(torch.randn(self.num_heads, L, self.head_dim))
        self.proj_kv = nn.Linear(features, features * 2, bias=False)
        self.proj_out = nn.Linear(features, features)

    def forward(self, input):
        kv = self.proj_kv(input)
        kv = einops.rearrange(kv, "... l (h d) -> ... h l d", h=self.num_heads)
        k, v = kv.chunk(2, dim=-1)
        q = self.q

        # TODO: flash?
        out = F.scaled_dot_product_attention(q, k, v)
        out = einops.rearrange(out, "... h l d -> ... l (h d)")

        return self.proj_out(out.to(input))


class MambaEncoder(nn.Module):

    def __init__(
        self,
        input_features=64,
        hidden_features=1024,
        out_features=768,
        out_length=77,
        depth=16,
        pool_heads=16,
        bias=False,
    ):
        super().__init__()

        features = hidden_features

        self.stem = nn.Linear(input_features, features)

        # Mamba
        self.norms = nn.ModuleList([
            nn.LayerNorm(features, bias=bias)
            for _ in range(depth + 1)
        ])
        self.mambas = nn.ModuleList([
            BidirectionalMamba(features, bias=bias)
            for _ in range(depth)
        ])
        self.mambas.append(nn.Identity())

        # Cross-attention
        self.pool = CrossAttentionPooling(features, L=out_length, num_heads=pool_heads)

        # Head
        self.use_projector = True  # hack!
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(features, 2 * out_features),
            nn.SiLU(),
            nn.Linear(2 * out_features, 2 * out_features),
        )

    def forward(self, input):  # (B L C)
        h = self.stem(input)

        # Mamba
        res = None
        for norm, mix in zip(self.norms, self.mambas):
            res = (h + res) if (res is not None) else h
            h = mix(norm(res))
            res = res.float()
        h = h.to(input)

        # Pooling with cross-attention
        h = self.pool(h)

        # Done!
        return self.head(h).chunk(2, dim=-1)
