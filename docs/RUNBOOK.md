# Scaling law v1：执行手册

所有命令在仓库根目录执行。原 README 中的 XGBoost pilot 和 Chemprop CLI 导出仍是历史流程；本研究使用 `scaling.cli`，不要混用两套运行目录或把旧 pilot 当正式重复。

## 1. 环境与输入

Linux/POSIX、Python 3.12 为参考环境；任务互斥锁使用 `fcntl`。先拉取新研究分支和 Git LFS 真实文件，再安装仓库固定依赖：

```bash
git switch research/scaling-law-v1
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU 参考安装；GPU 使用相同 torch 2.8.0 的对应官方 CUDA wheel。
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-scaling.txt
python -m pytest tests/test_scaling.py -q
python scripts/verify.py
python -m scaling.cli plan
```

基础版本继承现有 requirements-base/deep；正式 prepare 要求 RDKit 与原 Morgan 元数据一致（2026.3.6）。preflight 要求 Chemprop2.2.1、Torch2.8.0、原 XGBoost版本3.4.1，并冻结其余运行版本。CPU/CUDA torch 的本地版本后缀可以不同，完整版本和设备仍记录。

新派生 count 数组采用uint8，2048/8192维分别约268MB/1.07GB十进制文件体积，另有checkpoint和预测文件。训练档案会明显大于派生输入；先检查存储容量，不将 `--limit` 的少量运行误认为完整研究。没有统一的预估训练时间；以真实 `fit_seconds` 和设备记录规划资源。

## 2. 准备独立研究资产并做正式环境验收

```bash
python -m scaling.cli prepare
python -m scaling.cli preflight --device cpu
# 在实际 GPU 训练节点另做对应设备检查，例如：
# python -m scaling.cli preflight --device cuda:0
```

输出为 `studies/scaling-law-v1/`，原 data/splits/features 不变。准备过程检查原始特征/划分内容哈希，生成额外嵌套序列、validation角色、两套新随机划分、骨架划分、组成及count特征，保存实际matrix。

preflight 不训练正式模型；它检查版本并对官方Chemprop执行小批图前向/反向。每台实际环境/设备需要匹配的验收记录。没有通过时，训练命令会停止；不要删除版本检查绕过它。

prepare 不覆盖既有研究目录。中断时保留 `.partial`，先确认无准备进程仍在写入，再将该未完成目录移到单独失败归档并原命令重试。不得把 `.partial` 改名冒充完成。

## 3. 调参、冻结超参数与小规模校准

```bash
python -m scaling.cli run --stage tune --device cpu --threads 4
python -m scaling.cli lock
python -m scaling.cli run --stage calibration --device cpu --threads 4
python -m scaling.cli forecast
python -m scaling.cli seal
```

GPU将设备改为已验收的 `cuda:0`。默认XGBoost/Ridge仍使用CPU；神经路线使用指定设备。

`lock` 要求全部1,008次HPO记录完整；`forecast` 要求全部1,650次小规模运行完整。后者用V_law拟合竞争公式并生成四个大规模数值预测及bootstrap；`seal` 核对来源，冻结代码、环境、超参数和预测。不能先跑大规模，再补写一份声称“提前预测”的文件。

## 4. 大规模和所有预定对照

封存后执行：

```bash
python -m scaling.cli run --stage confirmation --device cpu --threads 4
python -m scaling.cli run --stage controls --device cpu --threads 4
python -m scaling.cli run --stage capacity --device cpu --threads 4
python -m scaling.cli run --stage budget --device cpu --threads 4
python -m scaling.cli run --stage validation_budget --device cpu --threads 4
python -m scaling.cli run --stage aggregation --device cpu --threads 4
python -m scaling.cli run --stage seed_variance --device cpu --threads 4
python -m scaling.cli run --stage robustness --device cpu --threads 4
```

这些组可以在依赖条件满足后分片并行。每个stage的分片索引独立：

```bash
# 四个独立任务分别使用0/4、1/4、2/4、3/4；下例是其中一个。
python -m scaling.cli run --stage calibration --shard 0/4 --device cuda:0 --threads 4
python -m scaling.cli status
```

`--limit 1` 只用于分批执行/硬件检查，会限制本次新增完成数，不会改变matrix；之后用同配置无limit继续。已完成运行核验后跳过。中断或失败的目录不会被自动覆盖：检查 `failure.json`，确认属于同配置技术重试后，加 `--retry-failed`，旧目录会移入 `failed/`。进程崩溃后的fcntl锁会由系统释放；不要手动删除别的活跃进程数据。

代码、配置、协议和手册在 prepare 时进入哈希；改变任何这些文件都会使正式Workspace拒绝继续。真正需要改动时建立带理由的新版本，而不是原目录上改参数。

## 5. 最终评价与自动分析

```bash
python -m scaling.cli evaluate --device cpu
python -m scaling.cli analyze
```

evaluate 会先检查全部计划训练（包含HPO和所有对照）已完成且参数一致。仅加载模型和已保存变换，不重新训练。可用 `--shard i/k` 划分评价任务；同一评价分片不要同时启动两个进程。训练与评价完成后再运行analyze。

`status` 显示完成标记数量，是进度显示，不是完整内容审计。缺文件、错哈希、错环境和缺失运行会被严格步骤阻止。模型测试结果来自各自固定测试集；新划分不是复用主测试ID。

## 6. 输出及论文交付

每个run保存参数、版本、训练历史、变换、模型和逐分子预测；`complete.json`最后写入，代表该训练产物完整。评价另外保存 `test.npz`、`evaluation.json`，记录封存文件哈希。

`analysis/` 中读取：

- 主学习曲线、原单位逐run指标、PNG/SVG图；主外推在V_law和T的确认及全范围描述性拟合。
- 15个预定配对比较及Holm校正、模型交叉区间、联合N/P留出诊断。
- 表示、容量、预算、验证预算、划分、种子、聚合对照；配对容量/预算表、优化敏感性标记和方差分量。
- 去重标签差异、同指纹有限测试集下界诊断、完整性audit。

将协议、base提交、研究提交、matrix、manifest、selected/hyper_lock、forecast/inputs/seal、环境、全部结果与失败记录一并归档。`publication_audit.json`检查实验完整性，不判断文章是否一定接收。

## 7. 当前交付的验证边界

本次开发环境没有完整QM9/LFS数组，且Chemprop不可安装，因此没有执行正式prepare/preflight或真实D-MPNN训练。开发环境使用的Torch/RDKit/XGBoost也不是正式固定版本。已执行的是合成数据单元与集成测试，详见 `reports/SCALING_LAW_CHANGE_REPORT.md`。

进入正式机器后，第1–2节是必须通过的启动验收，不是临时调整研究问题、删改规模或观察测试分数后的重新调方案。
