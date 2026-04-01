import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from torch import Tensor
from typing import Optional


class EnchantedViaDFGLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dropout,
        residual=True,
        norm_type="batchnorm",
        act="relu",
        k=3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.k = k

        self.structure_conv = GCNConv(dim_in, dim_out)
        self.feature_conv = GCNConv(dim_in, dim_out)
        self.gate_linear = nn.Linear(dim_out * 2, dim_out)

        act_dict = {"relu": F.relu, "leaky_relu": F.leaky_relu, "gelu": F.gelu}
        self.act_fn = act_dict.get(act, F.relu)

        if norm_type == "batchnorm":
            self.norm_feat = nn.BatchNorm1d(dim_out)
            self.norm_stru = nn.BatchNorm1d(dim_out)
            self.norm_fusion = nn.BatchNorm1d(dim_out)
        elif norm_type == "layernorm":
            self.norm_feat = nn.LayerNorm(dim_out)
            self.norm_stru = nn.LayerNorm(dim_out)
            self.norm_fusion = nn.LayerNorm(dim_out)
        else:
            self.norm_feat = None
            self.norm_stru = None
            self.norm_fusion = None

        if self.residual and dim_in != dim_out:
            self.residual_proj = nn.Linear(dim_in, dim_out)
        else:
            self.residual_proj = None

    def forward(self, data):
        ori_x = data.x
        batch = data.batch

        feature_graph_edge_index = k_farthest_graph(
            data.x, k=self.k, batch=batch, loop=True, cosine=True, direction=True
        )
        feat_x = self.feature_conv(data.x, feature_graph_edge_index)
        feat_x = self.act_fn(feat_x)
        if self.norm_feat is not None:
            feat_x = self.norm_feat(feat_x)
        feat_x = F.dropout(feat_x, self.dropout, training=self.training)

        stru_x = self.structure_conv(data.x, data.edge_index)
        stru_x = self.act_fn(stru_x)
        if self.norm_stru is not None:
            stru_x = self.norm_stru(stru_x)
        stru_x = F.dropout(stru_x, self.dropout, training=self.training)

        combined = torch.cat([feat_x, stru_x], dim=-1)
        gated = self.gate_linear(combined)
        gated = F.sigmoid(gated)
        fusion_x = gated * feat_x + (1 - gated) * stru_x

        if self.norm_fusion is not None:
            fusion_x = self.norm_fusion(fusion_x)

        if self.residual:
            if self.residual_proj is not None:
                fusion_x = fusion_x + self.residual_proj(ori_x)
            else:
                fusion_x = fusion_x + ori_x

        data.x = fusion_x

        return data


def k_farthest_graph(
    x: Tensor,
    k: int,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    cosine: bool = True,
    direction: bool = True,
) -> Tensor:
    """
    计算基于特征空间中 K-最远邻居的图的边索引 (edge_index)，支持批处理。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 [num_nodes, num_features]。
        k (int): 要连接的最远邻居的数量。
        batch (torch.Tensor, optional): 批次向量，将节点映射到对应的图。
                                        如果为 None，则假设所有节点属于单个图。
        loop (bool, optional): 如果为 True，则图中包含自环 (self-loops)。
        cosine (bool, optional): 如果为 True，则使用余弦距离；否则使用欧几里得距离。

    返回:
        torch.Tensor: 图的边索引 (edge_index)，形状为 [2, num_edges]。
    """
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    dense_x, mask = to_dense_batch(x, batch)
    B, N_max, _ = dense_x.shape

    if N_max == 0:
        return x.new_empty((2, 0), dtype=torch.long)

    # Compute pairwise distances for all graphs in parallel
    if cosine:
        x_norm = dense_x / dense_x.norm(dim=2, keepdim=True).clamp(min=1e-8)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        dist = 1.0 - sim
    else:
        x_sq = torch.sum(dense_x**2, dim=2, keepdim=True)
        dist = (
            x_sq
            + x_sq.transpose(1, 2)
            - 2 * torch.bmm(dense_x, dense_x.transpose(1, 2))
        )
        dist = dist.clamp(min=0.0)

    # Mask out padding: set distance to -inf for pairs involving padding nodes
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    dist = dist.masked_fill(~valid_pair, float("-inf"))

    # Select k+1 farthest neighbors (original uses k_adjusted = min(k+1, n_i))
    k_adj = min(k + 1, N_max)
    _, indices = torch.topk(dist, k=k_adj, dim=2, largest=True)

    # Build source and target local indices
    src_local = (
        torch.arange(N_max, device=x.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .expand(B, -1, k_adj)
    )
    tgt_local = indices

    # Convert to global indices using batch offsets
    num_per_graph = mask.sum(dim=1).long()
    offsets = num_per_graph.cumsum(0) - num_per_graph
    src_global = src_local + offsets.view(B, 1, 1)
    tgt_global = tgt_local + offsets.view(B, 1, 1)

    # Valid edge mask: both source and target must be real nodes
    src_valid = mask.unsqueeze(2).expand_as(src_local)
    mask_3d = mask.unsqueeze(1).expand(-1, N_max, -1)
    tgt_valid = mask_3d.gather(2, tgt_local)
    valid_edge = src_valid & tgt_valid

    if not loop:
        valid_edge = valid_edge & (src_local != tgt_local)

    src_global = src_global[valid_edge]
    tgt_global = tgt_global[valid_edge]

    edge_index = torch.stack([src_global, tgt_global], dim=0)

    if not direction:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return edge_index
