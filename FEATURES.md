# 三种模型输入的使用方式

所有输入按 `data/processed/qm9.csv.gz` 的分子顺序存储，另附分子 ID；抽样以 ID 对齐。
最终数据为 130,744 个分子，四个性质共用这些输入。

| 模型 | 保存的输入 | 含义 |
|---|---|---|
| Morgan + XGBoost | `features/morgan.npz` | 每个分子 2,048 位二进制指纹，按位打包 |
| D-MPNN | `features/dmpnn/` | 原子特征、键特征和有向图连接关系；长度随分子而变 |
| MoLFormer + MLP | `features/molformer.npy` | 每个分子 768 维冻结编码器向量，float32 |

分子图缓存是 D-MPNN 的输入。D-MPNN 训练中学到的表示还会随参数改变。
MoLFormer 向量来自固定的预训练模型，不使用 QM9 性质标签，可供所有训练规模和四个预测头复用。
项目尚未训练 D-MPNN 和 MLP，因此没有这两个模型的预测指标。

## 直接读取

在项目根目录执行：

```bash
python -m pip install -r requirements-base.txt
# CPU 环境；使用 GPU 时先按 PyTorch 官方说明安装对应版本。
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-deep.txt
python scripts/load_inputs.py
```

Python 示例：

```python
import sys
sys.path.insert(0, "scripts")
from load_inputs import load_run

run = load_run(size=1000, subset_seed=11, target="homo")
x_morgan = run["morgan"]["train"]      # (1000, 2048)
x_molformer = run["molformer"]["train"]  # (1000, 768)
y = run["targets"]["train"]            # 原单位标签
graphs = [run["graphs"][str(mid)] for mid in run["ids"]["train"]]
```

`load_run` 检查三种输入的 ID 顺序和数据来源哈希，使用同一组固定划分。
它不拟合标准化器，不训练模型，也不计算测试指标。
对标签和 MoLFormer 输入所做的任何均值/方差标准化，都只能在当前训练子集上拟合。

## 分子图格式

- `V.npy`：拼接的原子特征，每个原子 72 维。
- `E.npy`：拼接的有向边特征，每条有向边 14 维。
- `atom_ptr.npy` / `edge_ptr.npy`：各分子在拼接数组中的起止位置。
- `edge_index.npy`：边的起点和终点，编号为分子内局部原子编号。
- `rev_edge_index.npy`：每条边所对应的反向边，编号同样为分子内局部编号。
- `mol_id.npy`：分子顺序。

`scripts/graph_cache.py` 的 `GraphCache` 按行号或 ID 返回 Chemprop 的 `MolGraph`，
可交给官方 `BatchMolGraph`。缓存使用 Chemprop 2.2.1 默认 v2 原子特征。
`exports/` 中的 Chemprop CLI 示例仍从 SMILES 建图；CLI 不会自动读取这个缓存。

在没有图缓存的重建目录中执行 `python scripts/graph_cache.py` 即可生成并全量核对。
已有缓存时可用 `python scripts/graph_cache.py --verify-only` 核对。

## MoLFormer 提取设置

- 模型：`ibm-research/MoLFormer-XL-both-10pct`。
- 固定提交：`7b12d946c181a37f6012b9dc3b002275de070314`，来自官方 `compat-v4` 分支。
- 输入：统一的规范化无立体标记 SMILES。
- 输出：`pooler_output`；编码器冻结，eval 模式，`deterministic_eval=True`，float32。
- 本次设备：CPU；批大小 64；线程数 8；编码器随机种子 0。
- 不允许未知 token 或超长字符串被静默截断。

结果元数据在 `features/molformer.json`，包括权重和特征哈希、版本、重复前向与跨批次抽查结果。
权重文件不包含在项目压缩包中；使用已生成向量训练 MLP 不需要下载权重。
重新提取需要联网获取上述固定版本，或用 `--model-path` 指向完整本地快照：

```bash
python scripts/extract_molformer.py --device cpu --batch-size 64 --threads 8
# 中断后，在相同版本、批大小及参数下继续：
python scripts/extract_molformer.py --device cpu --batch-size 64 --threads 8 --resume
```

提取脚本拒绝覆盖已完成的 `molformer.npy`，从零复现请用新的项目目录。
验证细节见 `features/dmpnn/verification.json` 和 `features/molformer.json`。
