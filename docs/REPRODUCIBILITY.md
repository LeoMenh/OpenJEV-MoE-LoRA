# 运行与复现

此发布只包含当前 `batch_nll_mlp_openjev_1233_1024_20261027` 实验。代码来自正在运行的冻结源码快照，发布和校验均不修改原实验，不自动开始训练。

## 1. 安装与代码检查

使用 Python 3.10；[requirements.txt](../requirements.txt) 记录导出时可用的包版本。正式训练需要 NVIDIA CUDA / bitsandbytes；当前总控按八张不同 GPU 调度。

```bash
python -m pip install -r requirements.txt
python tools/verify_release.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

发布校验检查所有导出文件 hash、95 个冻结 Python 源文件及文档链接。测试使用小型随机初始化模型，核对当前软混合、梯度传递、病例独立性、前缀缓存、padding 和尾批等行为；不下载或运行 14B 正式训练。

## 2. 本地输入资产

训练所需大文件与病例数据没有放进 Git。对应哈希和体积见 [model_artifacts.json](../provenance/model_artifacts.json)，初始化模型与数据 revision 见 [assets.json](../initialization/records/assets.json)。

需要本地已准备的以下资产：

- Qwen3-14B 权重、tokenizer 与 738 类 DRG train/validation 固定切分；
- 冻结专家及分类头 `expert_and_head.pt`；
- Qwen3-1.7B 原始模型目录及已核验 O2 checkpoint；
- 当前 train16 oracle `train.pt`，其中保存固定成员顺序的18路损失与批目标。

原有本地资产树根目录为 `/ai/data/moe-lora`，下面包含 `Jev/` 与 `drg-llama/`。可以整体移到其他根目录；准备工具会映射配置中的原路径。没有这些资产时，仅克隆仓库不能复现数值成绩。病例级数据、标签关联表、逐病例预测和 hidden caches 由原工作区管理。

## 3. 创建独立运行目录

先只读检查，不写入：

```bash
python tools/prepare_local_run.py \
  --assets-root /ai/data/moe-lora \
  --run-dir "$PWD/local_runs/batch_nll_reproduction" \
  --dry-run
```

准备一个新的独立目录，复用本地专家与当前 train oracle，并重建路径变更后的运行合约：

```bash
python tools/prepare_local_run.py \
  --assets-root /ai/data/moe-lora \
  --run-dir "$PWD/local_runs/batch_nll_reproduction" \
  --check-weight-hashes
```

工具拒绝覆盖已存在的运行目录，复制固定源码，给权重与 oracle 建只读使用的符号链接。它不复制旧 router 权重、特征缓存或完成标记，也不会启动训练。使用同一专家资产时仍保持其只读，训练仅写新目录。

原始公开记录删去了继承的旧验证成绩字段，见 `provenance/source_manifest.json` 的 `publication_redactions`。`records/config.json` 和 `records/experiment_contract.json` 是审阅记录，不能直接拿原 digest 当作公开删减配置的有效运行合约。准备工具为本地路径和当前方法文档重新建立合约，冻结 Python 源文件内容不变。

明确启动训练：

```bash
python local_runs/batch_nll_reproduction/source/launch_batch_nll_routers.py \
  --run-dir "$PWD/local_runs/batch_nll_reproduction" \
  --gpus 0,1,2,3,4,5,6,7 --detach
```

总控先检查源码、配置、专家、O2 和训练 oracle 的 hash，再执行 preflight → 最优前缀特征 → 两方法阶段一训练与独立验证 oracle → 阶段一评价 → 顺序软前缀特征/训练 → 阶段二评价 → 审计与选模。此命令需要完整八卡资源；不会在轻量仓库检查时执行。

恢复用相同命令。配置、源码或资产变化时应新建运行目录，不混用缓存。`STOP` 文件或 SIGTERM 可停止该独立实验。查看 `progress.json`、`logs/` 及各方法的 `progress.json`。

## 4. 从头构建所需资产

`expert_training/` 保存了固定切分、探针信号、四块分组、共享/私有专家及联合软混合训练的实现与源配置。`initialization/` 保存 O1/O2 包装和固定来源；第三方作者代码在 `vendor/openjev-rlcd/`。这两组是原布局的来源代码，原始配置保留了绝对路径；从空工作区重建时应先按自己的目录复制并调整配置，不把归档入口直接当成最新实验总控。

初始化数据准备脚本原本查询当时的远端 dataset revision。复现实验必须使用 `initialization/records/assets.json` 固定的 revision，不能让未来的远端 HEAD 替换原输入。O1/O2 参数见对应 `*_arguments.json`；本地 `mmlu_splits.json` 的预期 hash 也在 assets 清单中。

`prepare_batch_nll.py` 的输入是经过专家身份验证的每病例原始 NLL 分块，不是旧路由器预测；重新计算这些分块时，应使用当前相同专家、1024 tokens、计算微批8和全部18条硬路线。之后按固定 train16 求均值并生成 oracle。该脚本保留训练病例数212553的断言，用不同数据集做新实验需要新建协议与合约。

## 5. 验证、结果和资源边界

公开 `records/progress.json` 是导出快照，实际训练可能继续前进。最终评价完成后，才能发布该轮 `evaluation/*/metrics.json`、选模记录及配对比较；现阶段的训练 CE、动作命中率和 preflight 都不等于最终开发集 DRG 准确率。

本轮仅 train/validation，不访问 test。历史数据使用范围和数值精度限制见 [Methods](METHOD_OPENJEV_LATEST_ZH.md)。此发布已做的轻量验证以仓库校验记录为准；不会把一次 CPU 测试称作完整八卡重训成功。
