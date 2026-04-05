# Agent Guidelines for GPS Repository

GPS (Graph Positional Encoding with Self-Attention) is a GNN framework built on PyTorch Geometric.

## Language Preference
- **默认使用中文交流** (Use Chinese by default)

## Environment & Commands

```bash
# Install and activate environment
pixi install
pixi shell

# Train with config
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml wandb.use=False

# Run with multiple seeds
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 5 wandb.use=False
```

## Testing

```bash
# Run all tests
python -m pytest unittests/

# Run single test file
python -m unittest unittests.test_egdfg_layer

# Run specific test
python -m unittest unittests.test_egdfg_layer.TestKFarthestGraph.test_single_graph_loop_true
```

## Code Style
- Use **Black** (>= 26.1.0), max line length 88
- Import order: stdlib → third-party → local (`gps.*`)
- Use `Optional[X]` instead of `X | None` (Python 3.13)
- Module name is `gps/` (not `graphgps/`)

## Architecture

### Config System
Uses YACS. Key sections: `dataset`, `model`, `train`, `optim`, `gnn`.
Configs in `configs/` organized by model type (GPS, EGDFG, GatedGCN, GINE, Graphormer, SAN).

### Network Registration
`gps/network/custom_gnn.py` registers `custom_gnn` network type. Custom layers are added via `build_conv_model()` and referenced as `model.type: custom_gnn` + `gnn.layer_type: <name>` in configs.

### EGDFG Layer
`gps/layer/egdfg_layer.py` — dual-branch GNN with feature-space (k-farthest neighbors) and structure-space (original edges) convolutions, fused via learned gate.
- `k_farthest_graph()` uses batched tensor parallelism via `to_dense_batch`
- Config: `configs/EGDFG/zinc-EGDFG.yaml`

### Model Architecture
`GPSModel` = node/edge encoders → `GPSLayer` (local GNN + global attention) → graph head.
Supported local GNNs: GCN, GIN, GINE, GAT, PNA, GatedGCN.
Supported global attention: Transformer, Performer, BigBird.

### Project Structure
```
gps/
├── config/      # Configuration definitions
├── encoder/     # Node/edge encoders
├── layer/       # GNN layers (GPSLayer, EGDFG, etc.)
├── head/        # Graph pooling heads
├── loader/      # Data loaders
├── logger.py    # Logging utilities
├── network/     # Model architectures
├── pool/        # Graph pooling
├── stage/       # Model stages
├── train/       # Training loops
└── transform/   # Graph transforms
```

### Positional Encodings
RWSE, LapPE, EquivStableLapPE, SignNet — implemented in `gps/transform/posenc_stats.py`.
