import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FourierProjection(nn.Module):
    def __init__(self, projection_dim, num_embed=64):
        super(FourierProjection, self).__init__()
        self.projection_dim = projection_dim
        self.num_embed = num_embed

        self.dense1 = nn.Linear(num_embed, 2 * projection_dim, bias=False)
        self.dense2 = nn.Linear(2 * projection_dim, projection_dim, bias=False)

    def forward(self, x):
        half_dim = self.num_embed // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = emb.float().to(x.device)
        freq = torch.exp(-emb * torch.arange(0, half_dim, dtype=torch.float32, device=x.device)).to(x.dtype).to(
            x.device)

        angle = x * freq * 1000.0

        sin_part = torch.sin(angle)
        cos_part = torch.cos(angle)

        embedding = torch.cat([sin_part, cos_part], dim=-1) * x
        embedding = F.silu(self.dense1(embedding))  # Use swish active function
        embedding = F.silu(self.dense2(embedding))

        return embedding
