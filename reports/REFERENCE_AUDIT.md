# QM9 原始结构与特征核对：已完成修正

用户上传的 `qm9.csv` 已完整读取，共 133,885 个分子，所有编号及偶极矩、HOMO、LUMO、g298
与此前下载镜像完全一致，四个目标的最大绝对差均为 0。

## 发现的问题

| 对照项 | 数量 |
|---|---:|
| 旧版已保留分子 | 128,780 |
| 旧版规范化 SMILES 与参考字符串不同 | 28,990 |
| 其中 Morgan 指纹不同 | 27,910 |
| 旧版解析失败、此次成功恢复 | 1,819 / 1,819 |
| 旧版重复项中额外恢复 | 145 |
| 新增分子总数 | 1,964 |
| 原有保留 ID 被删除 | 0 |

SMILES 字符串不同不一定表示化学结构不同，例如显式氢表示可能不同；但部分差异确实涉及
键级、形式电荷等，且有 27,910 条实际指纹不同。此前只验证旧特征对旧数据的自洽性，
不能证明结构来源正确，旧版不得直接作为正式基准数据。

两个直观例子：

| 原始编号 | 旧版 SDF 导出 SMILES | CSV 参考规范化 SMILES |
|---|---|---|
| gdb_23 | `[CH]C#C[CH]` | `C#CC#C` |
| gdb_24 | `[CH]C#C[NH3+]` | `C#CC#N` |

## 已完成修正

- 使用上传 CSV 的 SMILES 生成所有模型结构；SDF 不再参与结构生成，仅保留其标签表用于交叉核对。
- 删除官方名单 3,054 条后，130,831 条均可解析；保持原定无立体标记 SMILES 去重规则，再去重 87 条。
- 最终保留 **130,744 个分子**。这属于本协议的去重版本，不是未经去重的 130,831 条版本。
- 全量重新生成 **130,744 × 2,048 位 Morgan 指纹**。
- 重新冻结训练池 104,595、验证集 13,075、测试集 13,074，以及五条嵌套训练序列。
- 全量核对全部保留结构、全部指纹和四个标签，无差异；固定划分与嵌套性检查通过。
- 旧版指标不沿用；在新数据上重新跑了一次 1k/HOMO XGBoost。测试集尚未用于评估。

## 本版试运行

验证集 MAE：0.01014170 Hartree。
验证集 RMSE：0.01389654 Hartree。
这只是流程试运行，未完成模型调参或三模型比较。由于数据和划分更新，不能将与旧版误差的差异
全部归因于结构修正。

## 追溯文件

更新的 `qm9_scaling_starter.zip` 内包含原样参考 CSV、逐条核对表、新数据、新指纹、
新划分及脚本。`audit_previous/` 保存旧数据和指纹，仅用于复现对照；训练使用顶层新版目录。

`reports/reference_comparison.csv.gz` 是全量对照表；`reports/affected_molecules.csv.gz` 是受影响条目。

参考 CSV SHA-256：`3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4`。

## 机器核对摘要

```json
{
  "reference_rows": 133885,
  "reference_sha256": "3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4",
  "reference_ids_equal_raw_ids": true,
  "target_checks": {
    "mu": {
      "changed": 0,
      "max_absolute_difference": 0.0
    },
    "homo": {
      "changed": 0,
      "max_absolute_difference": 0.0
    },
    "lumo": {
      "changed": 0,
      "max_absolute_difference": 0.0
    },
    "g298": {
      "changed": 0,
      "max_absolute_difference": 0.0
    }
  },
  "previous_retained": 128780,
  "previous_invalid": 1819,
  "previous_invalid_parseable_now": 1819,
  "previous_invalid_retained_now": 1819,
  "previous_retained_same_structure": 99790,
  "previous_retained_changed_structure": 28990,
  "previous_retained_changed_fingerprint": 27910,
  "new_nonexcluded_parse_failures": 0,
  "new_duplicates": 87,
  "new_retained": 130744,
  "added_ids": 1964,
  "removed_ids": 0,
  "changed_structure_ids_retained_now": 28990,
  "comparison_scope": "Canonical non-isomeric SMILES and Morgan(radius=2, bits=2048); stereochemical differences are outside this protocol"
}
```
