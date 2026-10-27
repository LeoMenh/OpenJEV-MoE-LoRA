# 固定 batch NLL oracle：MLP / OpenJev 逐病例软路由重新训练

沿用 Qwen3-14B、1024 tokens、K=[1,2,3,3]、源 best.pt 第25285步的共享/私有 LoRA 与分类头；专家及分类头冻结，不重新训练。所有原始实验文件保持原状。

## 新监督

固定 train16 成员（212553例、13285批、尾批9例），对每批18条完整硬路线的真实 DRG NLL 求病例均值，取字典序打破并列的全局 argmin。将该路线三个分量复制给本批各病例。两个模型仍然逐病例输出 softmax 权重，执行 soft mixture；没有整个逻辑 batch 共用一个预测权重的机制。batch 成员固定，只打乱 batch 顺序。

训练 NLL 从已完成且同专家/输入/微批的原始逐例损失表重新聚合；3322个分块均核对身份、标签、来源合约和 SHA256。首批16例实际模型重算全部18路，误差精确为0。旧的逐病例最优标签、soft targets、旧隐藏缓存及旧路由器权重均不用于新训练。兼容性分组仅是源专家的历史，不是本次 router 标签。

## 两种模型的匹配训练

MLP 从 LayerNorm(5120) -> Linear(5120,32) -> SiLU -> Linear(32,K) 新初始化，末层零初始化。OpenJev 复用通用 O2，重新初始化各 block 的 bridge/qv-LoRA/head；不载入旧 DRG 路由器。

1. 最优硬前缀下重新提取病例入口特征；两种模型各 block 训练3轮硬标签 CE，对应所给方法文档式(33)。
2. 按 block1、2、3 顺序，在冻结已完成上游 router 的实际逐病例软混合下重新提特征，各继续2轮相同 batch 标签 CE。此阶段**不是**文档的硬前缀条件重标注 R-B，标签保持用户明确要求的 batch 全局最优路线。block1只有固定上游，因此可精确复用其特征。

两个模型都使用16例病例优化批。按用户最新要求，MLP隐藏宽度32、lr=1e-5（总参数522600）；OpenJev桥接宽度128、bridge/head lr=1e-4、q/v-LoRA lr=1e-5，AdamW、weight decay=0、clip=1。尾批按实际人数，以式(35)缩放；无负载均衡、class weights、soft targets、label smoothing，也不另接最终 DRG CE 微调。特征/eval可用8卡，两个 router 的缓存特征训练分别用一张卡。两种模型训练曝光匹配，算力不匹配。

## 验证与路由准确率

固定 validation8，共23617例、2953批、尾批1例；真实标签损失 oracle 在独立目录生成，不进入模型输入或训练。推理先输出全部病例分类分数和路由概率，再离线比较 argmax 与本批全局最低NLL路线的对应分量。主路由准确率按病例计数，block0不计入。它不是兼容性命中率，也不是独立优化单层专家的标签。soft mixture 不受枚举硬路线的 NLL 下界约束，不声称 soft oracle gap 非负。

在 optimal / soft-prefix 两个完整阶段候选中按验证 NLL 最小选模（文档准则），同时报告 Top1/5/10、NLL、macro-F1、每块 oracle 命中率、usage/entropy、完整路线命中率。比较使用 batch8 为重采样单位的配对 bootstrap 2000次。仅1个 seed=20260718，无测试集访问。

## 运行与恢复

progress.json、logs/、各方法 progress.json 记录真实状态。launcher 自动串联前置检查、特征提取、两种 router 训练、独立 validation oracle、两阶段验证与审计。源代码/配置/专家/O2/训练oracle均有合约hash。故障停止并记录 failed，不静默重试。STOP 文件或 SIGTERM 仅停止本独立实验。重启 source/launch_batch_nll_routers.py --run-dir 本目录 --detach，可跳过已完成分块及恢复训练。训练每200步保存 optimizer 与随机状态。
