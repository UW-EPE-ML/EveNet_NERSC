from torch import Tensor, nn
import torch
from evenet.control.config import DotDict
from evenet.network.layers.linear_block.masking import create_masking
from evenet.dataset.types import InputType


class LocalEmbedBlock(nn.Module):
  def __init__(self, options: DotDict, input_dim: int):
    super(LocalEmbedBlock, self).__init__()
    self.options = options
    self.local_Krank = self.options.Network.local_Krank

    self.input_dim      = input_dim
    self.projection_dim = self.options.Network.hidden_dim

    self.dense1 = nn.Linear(2 * self.input_dim, 2 * self.projection_dim)
    self.dense2 = nn.Linear(2 * self.projection_dim, self.projection_dim)
    self.gelu = nn.GELU()

  def forward(self, points: Tensor, features: Tensor) -> Tensor:

    # -------------------------
    # points represent the inital tensor to be calculated for local edge distance
    # points: [T, B, P]
    # features: [T, B, D]
    # -------------------------
    drij = self.pairwise_distance(points).transpose(0,1).contiguous() # [B, T, T]
    _, indices = torch.topk(-drij, k = self.local_Krank + 1, dim = -1) # [B, T, K+1]
    indices = indices[:, :, 1:] # [B, T, K] -> Get rid of itself
    knn_fts = self.knn(indices, features) # [B, T, K, D]
    knn_fts_center = features.transpose(0,1).unsqueeze(2).expand_as(knn_fts) # [B, T, K, D]
    local = torch.cat([knn_fts - knn_fts_center, knn_fts_center], dim = -1) # [B, T, K, 2D]
    local = self.gelu(self.dense1(local)) # [B, T, K, 2D]
    local = self.gelu(self.dense2(local)) # [B, T, K ,H]
    local = torch.mean(local, dim = 2) # [B, T, H]
    local = local.transpose(0,1) # [T, B, H]
    return local


  def pairwise_distance(self, x: Tensor) -> Tensor:
    # -------------------------
    # Calculate pairwise distance
    # x: [T, B, D]
    # -------------------------
    
    point_cloud = x.transpose(0,1).contiguous() # [B, T, D]

    r = torch.sum(point_cloud * point_cloud, dim = 2, keepdim = True) # [B, T, 1]
    m = torch.bmm(point_cloud, point_cloud.transpose(1, 2)) # [B, T, T]
    D = r - 2 * m + r.transpose(1, 2) + 1e-5

    # ---------------------
    # Return pairwise distrance
    # D: [T, B, T]
    # ---------------------

    return D.transpose(0,1).contiguous()
  
  def knn(self, topk_indices: Tensor, features: Tensor) -> Tensor:

    # ------------------
    # topk_indices: [B, T, K]
    # features: [T, B, D]
    # ------------------

    num_points = topk_indices.size(1)
    k = topk_indices.size(-1)
    batch_size = features.size(1)
    batch_indices = torch.arange(batch_size, device = features.device).view(-1, 1, 1) # [B, 1, 1]
    batch_indices = batch_indices.repeat(1, num_points, k) # [B, T, K]
    indices = torch.stack([batch_indices, topk_indices], dim = -1) # [B, T, K, 2]
    features_transpose = features.transpose(0,1).contiguous() # [B, T, D]
    knn_features = features_transpose[indices[..., 0], indices[..., 1]] # [B, T, K, D] 
    return knn_features


class LocalEmbedding(nn.Module):
  def __init__(self, options: DotDict, input_dim: int):
    super(LocalEmbedding, self).__init__()
    self.options           = options
    self.num_local_layer   = options.Network.num_local_layer
    self.local_point_index = options.Network.local_point_index

    self.input_dim         = input_dim

    self.masking           = create_masking(options.Network.masking)
    self.local_embed_layer = nn.ModuleList([(LocalEmbedBlock(options, self.input_dim) if i == 0 else LocalEmbedBlock(options, options.Network.hidden_dim)) for i in range(self.num_local_layer)])

  def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
    coord_shift = self.masking(Tensor([999.]).to(x.device), padding_mask).to(x.dtype).transpose(0,1) 
    coord_shift = coord_shift.unsqueeze(-1) # [ T, B, 1]
    points      = x[..., self.local_point_index]
    local_features = x.contiguous()
    for idx, local_embed in enumerate(self.local_embed_layer):
      local_features = local_embed(coord_shift + points, local_features) # [T, B, D]
      points = local_features

    return self.masking(local_features, sequence_mask)

