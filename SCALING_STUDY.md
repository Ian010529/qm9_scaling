# Scaling law 研究入口

`research/scaling-law-v1` 保留三条原始路线、四个性质及参考 SMILES 数据版本，提供可执行的前瞻性规模律研究协议。

| 文件 | 用途 |
|---|---|
| [冻结实验协议](docs/SCALING_LAW_PROTOCOL.md) | 当前研究设计、实验矩阵、统计和投稿证据 |
| [执行手册](docs/RUNBOOK.md) | 从环境验收到正式实验和结果分析的命令 |
| [修改与验证报告](reports/SCALING_LAW_CHANGE_REPORT.md) | 已实现内容、实际测试及尚未执行的正式验证 |
| [机器配置](config/scaling_law_v1.json) | 当前可执行的参数与判据 |

```bash
python -m scaling.cli plan
```

当前完整清单名义为 6,439 次训练，包含调参、主曲线、G 残差及预定对照；骨架划分截断后的实际清单由 prepare 保存。所有原始训练规模都包含在新网格中。

此前的 `docs/SCALING_LAW_EXPERIMENT_PLAN.md`、`SCALING_LAW_PLAN_CHANGE_REPORT.md` 保留作设计过程的历史资料；其预算和“尚无训练器”等状态不作为当前执行依据。当前协议和机器配置以上表文件为准，不混用不同方案。

本分支新增代码和设计，不包含虚构的规模律发现或正式实验结果。正式机器必须先通过固定依赖、真实 LFS 资产及官方 Chemprop 前后向验收。
