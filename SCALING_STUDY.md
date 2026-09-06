# Scaling law 研究入口

`research/scaling-law-v1` 保留三条原始路线、四个性质及参考SMILES数据版本，增加可执行的前瞻性规模律研究协议。

1. 完整设计：[docs/SCALING_LAW_PROTOCOL.md](docs/SCALING_LAW_PROTOCOL.md)
2. 执行命令：[docs/RUNBOOK.md](docs/RUNBOOK.md)
3. 修改与验证报告：[reports/SCALING_LAW_CHANGE_REPORT.md](reports/SCALING_LAW_CHANGE_REPORT.md)
4. 机器配置：[config/scaling_law_v1.json](config/scaling_law_v1.json)

```bash
python -m scaling.cli plan
```

名义完整清单6,439次训练，含调参、原始主曲线、G残差、容量与机制对照；骨架划分的实际池截断以prepare后的matrix为准。原README中的pilot仍保留作历史记录，不是新协议的正式实验入口。

本分支新增代码和设计，不包含虚构的规模律发现或已完成的正式研究结果。正式机器必须先通过固定依赖、真实LFS资产及官方Chemprop前后向验收。
