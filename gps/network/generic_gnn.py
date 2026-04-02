import inspect

import torch
from torch import nn
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

    For layers that expose `norm_type` in their constructor (e.g. EGDFG), the
    normalization config is passed directly to that layer. Otherwise GenericGNN
    applies an external normalization on `batch.x` after each message-passing
    layer.

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

        norm_type = getattr(cfg.gnn, "norm_type", None)
        if norm_type is None:
            norm_type = "batchnorm" if getattr(cfg.gnn, "batchnorm", False) else "none"
        self.norm_type = norm_type.lower()

        act = getattr(cfg.gnn, "act", "relu")

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        conv_params = inspect.signature(conv_model.__init__).parameters
        supports_norm_type = "norm_type" in conv_params
        supports_act = "act" in conv_params

        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            conv_kwargs = {
                "dropout": cfg.gnn.dropout,
                "residual": cfg.gnn.residual,
            }
            if supports_norm_type:
                conv_kwargs["norm_type"] = self.norm_type
            if supports_act:
                conv_kwargs["act"] = act

            self.gnn_layers.append(conv_model(dim_in, dim_in, **conv_kwargs))

            if supports_norm_type:
                self.norm_layers.append(nn.Identity())
            else:
                self.norm_layers.append(self._build_norm_layer(dim_in, self.norm_type))

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    @staticmethod
    def _build_norm_layer(dim_hidden, norm_type):
        if norm_type == "batchnorm":
            return nn.BatchNorm1d(dim_hidden)
        if norm_type == "layernorm":
            return nn.LayerNorm(dim_hidden)
        if norm_type == "none":
            return nn.Identity()
        raise ValueError(
            f"Unsupported gnn.norm_type '{norm_type}', choose from "
            "['batchnorm', 'layernorm', 'none']."
        )

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
        batch = self.encoder(batch)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)

        for gnn_layer, norm_layer in zip(self.gnn_layers, self.norm_layers):
            batch = gnn_layer(batch)
            batch.x = norm_layer(batch.x)

        batch = self.post_mp(batch)
        return batch
