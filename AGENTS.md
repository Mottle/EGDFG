# AGENTS Guide (EGDFG/GPS)

## 语言与沟通
- 默认使用中文。

## 环境与运行（以可执行配置为准）
- 使用 `pixi`：`pixi install`，运行命令优先 `pixi run ...`。
- `pixi.toml` 固定 Python `3.13.1`，PyTorch `2.8` + CUDA `12.9` 轮子（PyG 相关轮子也已固定 URL）。
- 训练入口是 `main.py`，常用：
  - `pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml wandb.use=False`
  - `pixi run python main.py --cfg <cfg> --repeat 5 wandb.use=False`
- `run/run_experiments.sh` 和 `run/wrapper.sb` 是 SLURM+conda(`graphgps`) 工作流，不是本地默认开发路径。

## 关键执行逻辑（避免改错位置）
- `main.py` 里 `import gps` 会触发各子模块注册；新增 layer/network/config/train 时必须确保在 `gps/*` 下可被导入。
- 默认训练模式被 `gps/config/defaults_config.py` 改成 `train.mode=custom`，实际走的是 `gps/train/custom_train.py`。
- 多次运行规则（`main.py`）：
  - `--repeat N`：按 seed 递增重复。
  - 若设置 `cfg.run_multiple_splits`，则必须 `--repeat 1`，否则会抛 `NotImplementedError`。

## 配置与模型约定
- 本仓库包名是 `gps/`（README 里仍有 `graphgps/` 旧路径示例，按代码实际路径操作）。
- `generic_gnn` 的归一化：
  - 配置键：`gnn.norm_type` in `{batchnorm, layernorm, none}`。
  - 兼容旧键：未设 `gnn.norm_type` 时回退到 `gnn.batchnorm`。
  - 对不支持 `norm_type` 的层（如 `gineconv/gatedgcnconv`），`GenericGNN` 会在层后对 `batch.x` 做外部 norm。
- EGDFG 任务配置目录：`configs/EGDFG/`，当前包含
  `zinc-EGDFG.yaml`、`proteins-EGDFG.yaml`、`nci1-EGDFG.yaml`、`nci109-EGDFG.yaml`、`frankenstein-EGDFG.yaml`、`dd-EGDFG.yaml`。

## 数据集与切分坑点
- 自定义数据入口在 `gps/loader/master_loader.py`（`custom_master_loader`）。
- `PyG-TUDataset` 仅支持硬编码名称；不在白名单会直接报错：
  - `DD`, `NCI1`, `NCI109`, `ENZYMES`, `PROTEINS`, `FRANKENSTEIN`, `TRIANGLES`, `IMDB-*`, `COLLAB`。
- split 默认是 `standard`（见 `gps/config/split_config.py`）。
  - 对无官方固定 split 的任务（如多数 TU 图分类）需在 yaml 显式设 `split_mode: random` 和 `split: [train, val, test]`。

## 测试与快速验证
- 全量单测：`python -m pytest unittests/`
- EGDFG 定向：
  - `python -m unittest unittests.test_egdfg_layer`
  - `python -m unittest unittests.test_egdfg_layer.TestKFarthestGraph.test_single_graph_loop_true`
- 做小改动可先用 `python -m py_compile <file.py>` 做语法冒烟检查。
