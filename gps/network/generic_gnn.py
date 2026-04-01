import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from gps.layer.gatedgcn_layer import GatedGCNLayer
from gps.layer.gine_conv_layer import GINEConvLayer
from gps.layer.egdfg_layer import EnchantedViaDFGLayer


@register_network("generic_gnn")
class GenericGNN(torch.nn.Module):
    """
    Generic GNN model with configurable normalization and residual connections.

    Supports stacking any registered conv layer with per-layer norm, activation,
    and residual control via the `gnn` config section.

    Config keys used:
        gnn.layer_type      - name of the conv layer (e.g. 'egdfg', 'gatedgcnconv')
        gnn.layers_mp       - number of message-passing layers
        gnn.dim_inner       - hidden dimension
        gnn.dropout         - dropout rate
        gnn.residual        - whether to use residual connections
        gnn.norm_type       - 'batchnorm', 'layernorm', or 'none'
        gnn.act             - activation: 'relu', 'leaky_relu', 'gelu'
        gnn.layers_pre_mp   - number of pre-MP linear layers
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(
                conv_model(
                    dim_in,
                    dim_in,
                    dropout=cfg.gnn.dropout,
                    residual=cfg.gnn.residual,
                    norm_type=getattr(cfg.gnn, "norm_type", "none"),
                    act=getattr(cfg.gnn, "act", "relu"),
                )
            )
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == "gatedgcnconv":
            return GatedGCNLayer
        elif model_type == "gineconv":
            return GINEConvLayer
        elif model_type == "egdfg":
            return EnchantedViaDFGLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
