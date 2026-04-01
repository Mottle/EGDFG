from torch_geometric.graphgym.register import register_config


@register_config("custom_gnn")
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False


@register_config("generic_gnn")
def generic_gnn_cfg(cfg):
    """Config for the GenericGNN network with normalization support."""

    # Normalization type: 'batchnorm', 'layernorm', or 'none'
    cfg.gnn.norm_type = "batchnorm"

    # Activation function: 'relu', 'leaky_relu', 'gelu'
    cfg.gnn.act = "relu"

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = True
