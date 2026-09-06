# QM9 Scaling Law Research

研究分支：`research/scaling-law-v1`。目标是发现、解释与验证分子性质预测的 scaling laws；保留 Morgan＋XGBoost、D-MPNN、冻结 MoLFormer＋MLP，以及 μ、HOMO、LUMO、G。

## 从这里开始

| 文件 | 用途 |
|---|---|
| [冻结实验协议](docs/SCALING_LAW_PROTOCOL.md) | 研究问题、完整矩阵、超参数、外推、对照、统计与投稿证据 |
| [执行手册](docs/RUNBOOK.md) | 环境验收、调参、冻结、训练、评价与分析的确切命令 |
| [修改报告](reports/SCALING_LAW_CHANGE_REPORT.md) | 本分支改了什么、测试了什么、尚未实际验证什么 |
| [机器配置](config/scaling_law_v1.json) | 可执行的规模、种子、搜索空间与预定判据 |
| [原始 README 归档](docs/BASELINE_README.md) | 参考 SMILES v2 的来源、修正和历史试运行说明，原文不变 |
| [原有特征说明](FEATURES.md) | 已保存三种输入的结构与来源 |

```bash
python -m scaling.cli plan
```

主设计：15 个训练规模、10 次配对重复；先用 N≤20k 的曲线冻结对 30k/50k/75k/100k 的预测。增加 G 残差、计数指纹、读出头、容量、训练/验证预算和划分对照。名义完整矩阵 6,439 次训练（含 HPO）；实际骨架池截断以 prepare 生成的清单为准。

正式入口为 `python -m scaling.cli`。先按执行手册安装固定环境、拉取 Git LFS 并通过 preflight，不能直接把旧 pilot 命令当成新协议执行。

## 数据与当前状态

继续使用参考 SMILES v2 的 130,744 个分子、原主划分和已有特征。原始 `data/`、`splits/`、`features/`、`runs/` 和 `audit_previous/` 均不因本分支被重算或覆盖；新的资产与结果写入 `studies/scaling-law-v1/`。

本分支新增设计和执行代码，**不是已经完成的规模律研究结果**。原仓库仍只有历史 XGBoost 试运行的真实训练指标。本次本地验证为合成数据软件测试；完整 QM9、固定版本环境、GPU及正式 D-MPNN 训练状态见修改报告，启动前由程序强制验收。
