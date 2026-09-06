# QM9 数据规模实验：参考 SMILES 核对版

本版已用用户上传的 `qm9.csv` 全量核对并替换旧 SDF 推导的结构。
正式使用本版 `data/processed`、`splits` 和 `features`，不要与旧版数据或索引混用。
目前已完成三种模型的输入准备，以及一次 XGBoost 试运行；不是完整三模型基准结果。
三种输入的格式和读取方式见 `FEATURES.md`。

## 核对发现

- 参考 CSV 包含 133,885 个分子，编号和四个性质与下载镜像完全一致。
- 旧版 128,780 条中，28,990 条规范化 SMILES 字符串不同，其中 27,910 条 Morgan 指纹不同。
- 字符串不一致数并不等于化学结构差异数：其中包含显式氢表示差异。
  但确有键级、形式电荷等实质差异，例如 gdb_23、gdb_24；详细旧/新字符串见核对表。
- 旧版解析失败的 1,819 条全部可由参考 SMILES 正确解析，并全部恢复。
- 旧版误归入重复项的条目中另有 145 条恢复。新增合计 1,964 条，旧版保留 ID 无删除。
- 之前的“指纹重算一致”只说明特征与旧数据自洽，不能证明旧数据与参考结构一致。
  旧版试运行模型和结果不作为本版正式结果。

## 最终数据口径

| 步骤 | 分子数 |
|---|---:|
| 原始参考 CSV | 133,885 |
| 删除官方异常名单 | 3,054 |
| 官方异常名单删除后 | 130,831 |
| 额外 SMILES 解析失败 | 0 |
| 缺失或非有限目标 | 0 |
| 按规范化无立体标记 SMILES 去重 | 87 |
| 本协议保留 | 130,744 |
| 固定训练池 | 104,595 |
| 固定验证集 | 13,075 |
| 固定测试集 | 13,074 |

87 条去重按原始编号保留最小编号，不依据目标值选样，也不平均标签。
因此本项目是 130,744 分子的去重版本，不应报告为未经去重的 130,831 条。
删除记录和保留的代表 ID 均可追溯。

## 实验参数

- 四个性质单独训练：`mu`、`homo`、`lumo`、`G`。
- 偶极矩单位 Debye；三个能量目标保留 Hartree。
- `G` 对应 `g298`：298.15 K 下分子总 Gibbs 自由能，不是溶剂化自由能或原子化自由能。
- 所有模型使用同一个规范化无立体标记 SMILES，带立体信息的 SMILES 仅留作追踪。
- 模型输入不包含三维坐标。SDF 文件只用于核对原始标签，不再用于生成模型结构。
- 固定划分种子 20260905；抽样种子 11、22、33、44、55。
- 对应训练种子 101、102、103、104、105。
- 每个抽样种子只打乱一次训练池；前 N 个 ID 是规模 N 的训练集。
- 规模 500、1k、2k、5k、10k、20k、50k、100k。
- 本版数据修正后重新生成划分；从本版开始保持固定。
- Morgan 半径 2、2048 位二进制指纹、无手性位，所有四个性质共用。

## 文件

| 路径 | 内容 |
|---|---|
| `data/raw/qm9_smiles.csv` | 上传的参考 CSV 原样副本 |
| `data/raw/qm9.zip` | 原始镜像；内含 SDF 和标签表；SDF 不用作模型输入 |
| `data/raw/uncharacterized.txt` | 官方异常分子名单 |
| `data/raw/manifest.json` | 来源说明与 SHA-256 |
| `data/processed/qm9.csv.gz` | 已核对的分子表和四个目标 |
| `data/processed/exclusions.csv` | 官方排除和 87 条去重记录 |
| `splits/fixed.npz`、`fixed_assignment.csv` | 新版固定训练池、验证集、测试集 |
| `splits/sequence_*.npy`、`*.csv.gz` | 五条嵌套训练序列，CSV 中 rank 不超过 N 即 D_N |
| `features/morgan.npz` | 130,744 条分子编号及打包存储的 2048 位指纹 |
| `features/morgan.json` | 参数、对应数据哈希、特征哈希和相同指纹统计 |
| `features/dmpnn/` | 全量 Chemprop 分子图输入及核对报告 |
| `features/molformer.npy`、`molformer_ids.npy` | 全量 768 维冻结向量和分子 ID |
| `features/molformer.json` | 固定模型提交、哈希、提取设置及验证结果 |
| `FEATURES.md` | 三种输入的读取与复现说明 |
| `reports/reference_comparison.csv.gz` | 原始 133,885 条逐编号对照，旧状态、旧/新 SMILES、指纹变化 |
| `reports/affected_molecules.csv.gz` | 旧 SMILES 不同或旧解析失败的受影响条目 |
| `reports/reference_audit.json` | 核对统计 |
| `reports/verification.json` | 全量结构、特征及划分检查 |
| `runs/` | 在新版数据上重新执行的 1k/HOMO XGBoost 试运行 |
| `exports/` | 新版 D-MPNN 1k/HOMO 的显式训练/验证输入 |
| `audit_previous/` | 旧版数据和特征，仅供复现对照，禁止作为新版训练输入 |
| `scripts/` | 数据处理、抽样、指纹、训练、导出与核对脚本 |
| `config/experiment.json` | 统一配置 |

## 运行方式

Python 3.12；基础依赖已在本次环境实际验证。

```bash
python -m pip install -r requirements-base.txt
```

GitHub 仓库包含上表列出的原始与处理数据、固定划分、三种特征、试运行模型、导出文件及
`audit_previous/` 对照归档，不必重新运行下载和生成。大型特征数组通过 Git LFS 保存；
请先安装 Git LFS，再克隆并拉取完整文件：

```bash
git lfs install
git clone https://github.com/Ian010529/qm9_scaling.git
cd qm9_scaling
git lfs pull
```

已有克隆可在仓库目录执行 `git pull` 和 `git lfs pull`。不要将 LFS 指针文件当作数组使用。
`.gitignore` 仅排除环境、缓存、凭据和临时文件。历史核验报告描述的是本次本地实验，
不代表新环境已经运行验证。
若需从零重建，请在没有旧输出的新目录中运行：

```bash
python scripts/download.py
python scripts/prepare.py
python scripts/make_splits.py
python scripts/morgan.py
python scripts/verify.py
python scripts/train_xgb.py --target homo --size 1000 --subset-seed 11 --training-seed 101
python scripts/export_chemprop.py --target homo --size 1000 --subset-seed 11
```

下载脚本会首先验证现有原始文件，校验通过即直接复用，不重新下载。
`prepare.py` 默认读取 `data/raw/qm9_smiles.csv`；也可通过 `--smiles-csv` 指定其他同内容文件。
它要求原始 ID 集完全一致，四个目标逐编号与镜像核对通过后才生成新数据。
输出生成脚本拒绝覆盖已存在的固定数据或模型，避免无意更改基准。

复现本次与旧版的对照：

```bash
python scripts/audit_reference.py --previous-project audit_previous
```

## 验证与试运行

所有 130,744 条保留结构都重新从参考 CSV 规范化核对；所有 130,744 条指纹也逐条重新计算，
均无差异。四个目标、排除名单、ID 顺序、划分互斥和嵌套性检查全部通过。

XGBoost 试运行只训练 1,000 个分子的 HOMO。目标标准化只拟合这 1,000 个分子；
最终预测和误差已还原到 Hartree。早停依据验证集 MAE，尚未计算测试集指标。
`learning_history.json` 的训练历史使用标准化目标尺度，与最终原单位误差区分使用。
本次试运行参数尚未经过超参数搜索，不是正式调参结论。

## 深度模型输入与后续工作

D-MPNN 的全部分子图已缓存，并逐条与 Chemprop 2.2.1 官方特征生成器核对通过。
MoLFormer 的全部 130,744 × 768 冻结向量已提取，模型版本固定到具体提交，
已检查全部向量有限、重复前向一致和跨批次数值一致。
详细结果见 `reports/feature_inputs_verification.json` 和两个特征目录的验证元数据。

深度依赖安装与读取示例见 `FEATURES.md`。Chemprop CLI 示例包含训练种子 101 和早停耐心值 20；
其训练尚未执行。命令中的训练轮数、早停和网络参数仍需纳入正式实验的统一调参协议。

尚未完成 D-MPNN / MLP 训练、MLP 训练器、480 次批量实验调度及骨架划分补充验证。
冻结特征可全量计算；之后任何需要拟合的标准化仅使用当前训练子集。

## 来源

- 原始 QM9：https://doi.org/10.6084/m9.figshare.c.978904
- 参考 CSV 公开地址：https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
- 本次参考 CSV 由用户上传；四个标签与镜像全量一致，并保存内容哈希。
- 镜像：https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
- 排除名单：https://ndownloader.figshare.com/files/3195404
- MoLFormer：https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct
- Chemprop：https://chemprop.readthedocs.io/en/main/tutorial/cli/train.html
