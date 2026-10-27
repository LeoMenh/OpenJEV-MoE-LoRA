# 当前实验状态与成果

本仓库仅记录 `batch_nll_mlp_openjev_1233_1024_20261027`，不收录旧实验成绩。发布快照时仍在训练，尚无最终验证结果；状态不会随服务器自动更新。

固定 train16 的 18 路平均最终 DRG NLL 最优路线，复制标签到病例；部署执行逐病例软混合。MLP 和 OpenJEV 均为 3 轮最优前缀 CE + 2 轮实际软前缀 CE。train/validation=212553/23617，1024 tokens，K=[1,2,3,3]，seed=20260718。

已完成监督完整性核验、最优前缀隐藏特征提取和真实模型 preflight；路由器仍在训练。首批 16 例重算全部 18 路与来源代价表误差为 0。专家和分类头保持冻结。固定计算形状下改变同伴病例的路由影响为 0。

| 本轮方法 | 开发集 Top1 | 开发集 NLL | 状态 |
|---|---|---|---|
| 新 MLP，batch NLL 监督 | 待生成 | 待生成 | 训练中 |
| 新 OpenJEV，batch NLL 监督 | 待生成 | 待生成 | 训练中 |

原始记录：[进度快照](../experiments/batch_nll_20261027/records/progress.json)、[监督核验](../experiments/batch_nll_20261027/records/oracle/verification.json)、[preflight](../experiments/batch_nll_20261027/records/preflight.json)、[MLP 进度](../experiments/batch_nll_20261027/records/mlp/progress.json)、[OpenJEV 进度](../experiments/batch_nll_20261027/records/openjev/progress.json)。

最终将按 validation NLL 选择完整策略，报告 DRG Top1/5/10、NLL、macro-F1、batch NLL oracle 动作命中率、使用率与熵，以及两种方法的配对 bootstrap 差值。本轮不访问测试集。
