# OpenJEV–MoE–LoRA：DRG 分类与可学习专家路由

以 Qwen3-14B 为病例编码与分类骨干，使用共享/私有 LoRA 专家池，并研究本地 OpenJEV 能否从 block 入口隐藏状态预测专家混合权重。

**当前实验是 2026-10-27 的 batch NLL 监督对照实验，发布快照时仍在训练，尚无最终验证成绩。本仓库仅发布这一轮实验，不收录旧实验结果。**

- [完整、通俗的最新 Methods](docs/METHOD_OPENJEV_LATEST_ZH.md)：模型、专家来源、初始化、监督、训练、推理及评价。
- [当前实验进展](docs/RESULTS.md)：记录本轮真实进度及尚待生成的最终指标。
- [运行与复现](docs/REPRODUCIBILITY.md)：环境、输入资产、校验、独立运行目录及恢复命令。
- [模型及大文件产物清单](provenance/model_artifacts.json)：权重大小、原始本地位置及 SHA256。

## 当前模型

| 部分 | 实际实现 |
|---|---|
| 任务 | 738 类 DRG 分类 |
| 病例骨干 | Qwen3-14B，40 层，最大 1024 tokens，NF4 / BF16 |
| 分块 | 0–9、10–19、20–29、30–39，每块 10 层 |
| 专家数 | **[1,2,3,3]**；18 条完整硬路线 |
| 专家池 | 复用验证选中的第 25285 步专家与分类头，最新实验全部冻结 |
| OpenJEV | Qwen3-1.7B 经 O1/O2 RLCD 初始化；冻结 decoder 原权重，各块独立 bridge、q/v LoRA 与输出头 |
| 当前监督 | 固定 train16 批的 18 路平均 DRG NLL 最优路线，复制给批中各病例 |
| 当前执行 | **逐病例软混合**；一个批内可有不同路由权重 |
| 匹配对照 | 新初始化 MLP，隐藏宽度 32，学习率 1e-5 |
| 数据 | train 212553；validation 23617；本轮不访问 test |

运行目录名中的 `1233` 对应 `[1,2,3,3]`。旧方案 `[1,3,3,3]` 有 27 条路线；请以当前配置为准。

## 仓库内容

```text
experiments/batch_nll_20261027/      最新实验的实际源码快照和运行记录
expert_training/                   专家池构建、分组、专家训练和联合训练代码
initialization/                    OpenJEV O1/O2 初始化包装与来源记录
vendor/openjev-rlcd/                固定作者版本及原 MIT 许可证
tools/                             发布校验、资产检查及独立运行准备
tests/                             数值等价检查与历史回归测试
provenance/                        文件来源、环境与大文件校验清单
```

源码来自实际运行目录，保留源文件 hash；公开配置移除了继承的旧成绩字段，详见来源清单。运行状态是导出时的静态快照，不会随服务器训练自动更新。病例文本、病例 ID 表、逐病例输出、隐藏缓存和模型大权重保留在本地；Git 中提供汇总成果及产物校验清单。

## 快速检查

```bash
python tools/verify_release.py
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

完整实验需要已准备的模型、数据切分、专家和训练 oracle；准备方法见[复现说明](docs/REPRODUCIBILITY.md)。发布操作不会重启或修改服务器正在运行的实验。

## 来源

OpenJEV-RLCD 作者实现：[ZimmyGao/openjev-rlcd](https://github.com/ZimmyGao/openjev-rlcd)，固定提交 `22db40ad3333f5cf0bc5139097da70e0f8c05018`。本项目使用本地模型训练，不调用官方 Jev API。第三方代码的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
