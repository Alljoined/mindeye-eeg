import torch.nn as nn
from functools import partial
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, out_dim=768, in_dim=512, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=False):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        # Calculate the size after pooling
        self.pool_kernel_size = 4
        num_channels = 128
        pooled_size = (in_dim // self.pool_kernel_size) * num_channels

        # Linear and MLP layer
        self.lin0 = nn.Sequential(
            nn.Linear(pooled_size, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        '''
            bs, 128, 512 -> bs, 128, 128
            bs, 16384 -> bs, 32h
            b2, 32h -> bs, 768
        '''
        import pdb
        pdb.set_trace()
        # Average pooling over time steps
        x = F.avg_pool1d(x, self.pool_kernel_size)
        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Pass through MLPs
        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x