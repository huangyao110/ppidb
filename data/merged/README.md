# P2PSigLip merged PPI database

`data/merged` 是当前项目的整合版蛋白质相互作用数据库。它把多个 PPI 来源统一到同一套 protein id 和 pair id 体系中，并在 pair 层面保留来源、证据类型、证据标签和粗粒度质量分层。

这个目录描述的是数据库本身，不是某一次训练的 train/val/test split。

## 数据规模

| 项目 | 数量 |
|---|---:|
| 去重蛋白质序列 | 356,373 |
| 去重无序 PPI pairs | 8,717,404 |
| 正样本 pairs (`label=1`) | 8,182,023 |
| 合成负样本 pairs (`label=0`) | 535,381 |
| source tags | 21 |
| 多来源支持 pairs (`n_sources >= 2`) | 511,413 |
| 高多来源支持 pairs (`n_sources >= 3`) | 166,469 |
| 序列长度 median / p10 / p90 / max | 262 / 85 / 689 / 35,213 aa |

## 主要文件

| 文件 | 行数 | 内容 |
|---|---:|---|
| `proteins.csv` | 356,373 | 蛋白质主表。一行对应一个唯一氨基酸序列。 |
| `sequences.csv` | 356,373 | 简化序列表，包含 `id,sequence`。 |
| `interactions.csv` | 8,717,404 | PPI 主表。一行对应一个去重后的无序 pair，包含来源、证据和标签。 |
| `pairs.csv` | 8,717,404 | 简化 pair 表，包含 `fpid_1,fpid_2,label`。 |
| `archive/` | - | 去重前或中间版本，用于追溯，不建议作为默认数据库入口。 |
| `reports/` | - | 合并、去重、证据清洗和质量分层的统计报告。 |
| `figures/` | - | 数据库统计图。Nature 风格汇总图位于 `figures/nature/`。 |

## 硬性数据库契约

`data/merged` 是下游 API 的输入边界，不允许贡献者或 agent 随意漂移。契约写死在 `p2psiglip_db/data/merged_contract.py`，任何重新构建数据库的变更都必须通过：

```bash
python ppidb.py validate-merged --merged-root data/merged
```

固定规则如下：

- 文件名固定为 `proteins.csv`、`sequences.csv`、`interactions.csv`、`pairs.csv`。
- 表头和列顺序固定，不能新增、删除、重命名或重排字段。
- `fpid` 是唯一公开 protein id，格式固定为 `FP` + 7 位数字，例如 `FP0000001`。
- `fpid` 必须从 `FP0000001` 连续递增，同一个 `fpid` 永远不能指向另一条序列。
- `protein_md5` 必须等于规范化氨基酸序列的 MD5。
- `sequences.csv` 必须严格等于 `proteins.csv` 投影出的 `id,sequence`。
- `interactions.csv` 一行只能表示一个无序 pair，且必须满足 `FPid_1 < FPid_2`，禁止 self-pair 和重复 pair。
- `pairs.csv` 必须严格等于 `interactions.csv` 投影出的 `fpid_1,fpid_2,label`。
- `Evidence_Type`、`Evidence_Tags`、`PPI_Tier`、`PPI_Source` 只能使用 contract 中列出的枚举。
- `label=0` 只能表示 `negative_synthetic`，正负样本冲突时必须采用 positive wins。
- 当前发布快照的行数和 SHA256 已写入 `EXPECTED_SNAPSHOT`。任何改变当前 CSV 字节、行顺序或 ID 映射的更新都必须同步修改 contract，并说明 API 兼容性影响。

## Protein 表

`proteins.csv` 是蛋白质目录表，核心字段如下。

| 字段 | 含义 |
|---|---|
| `protein_md5` | 序列的 MD5 hash。相同序列共享同一个 hash。 |
| `fpid` | 项目内部 protein id，例如 `FP0000001`。 |
| `sequence` | 氨基酸序列。 |
| `length` | 序列长度。 |
| `hydrophobicity` | 序列疏水性统计值。 |
| `is_canonical` | 是否标记为 canonical sequence。 |
| `original_ids` | 上游数据库中的原始 id，多个 id 用 `;` 连接。 |

`sequences.csv` 是 `proteins.csv` 的轻量视图，只保留 `id,sequence`。

## Interaction 表

`interactions.csv` 是数据库的核心 pair 表。

| 字段 | 含义 |
|---|---|
| `FPid_1`, `FPid_2` | pair 两端的内部 protein id。pair 已按无序关系去重。 |
| `original_id1`, `original_id2` | 对应上游来源中的原始 protein id。 |
| `PPI_Source` | 支持该 pair 的来源标签，多个来源用 `;` 连接。 |
| `Seq_Source` | 序列来源。 |
| `label` | `1` 表示相互作用正样本，`0` 表示合成负样本。 |
| `Experimental_Method` | 上游提供或归一化后的实验/来源方法描述。 |
| `Evidence_Type` | 清洗后的单一证据类别，用于筛选和统计。 |
| `Evidence_Tags` | 证据 provenance 标签合集，保留多来源证据并集。 |
| `PPI_Tier` | 英文质量层级：`diamond/gold/silver/bronze/negative_synthetic`。 |
| `PPI_Tier_ZH` | 中文质量层级：`钻石/黄金/白银/青铜/负样本`。 |
| `n_sources` | 支持该 pair 的不同 `PPI_Source` 数量。 |

`pairs.csv` 是 `interactions.csv` 的轻量视图，只保留 `fpid_1,fpid_2,label`。

## 证据类型

`Evidence_Type` 是清洗后的单一类别。更完整的多来源证据并集保存在 `Evidence_Tags` 中。

| Evidence_Type | 数量 | 说明 |
|---|---:|---|
| `no_exp` | 5,126,798 | PPIDB 中没有明确 throughput 或实验类型支持的 pair。 |
| `LTP` | 1,231,427 | 低通量实验或文献级证据。 |
| `HTP` | 708,421 | 高通量实验筛选证据。 |
| `negative_synthetic` | 535,381 | 随机采样或构造得到的合成负样本。 |
| `mixed` | 341,230 | 来源层面可信，但无法干净拆分为 HTP/LTP/structural 的证据。 |
| `HTP_LTP` | 280,385 | 同时具有 HTP 和 LTP 支持，或 PPIDB `both` 标签。 |
| `structural` | 263,381 | 来自结构相关来源的 pair。 |
| `complex_curation` | 230,381 | 复合物整理证据，不等同于 PDB interface 结构证据。 |

注意：`no_exp` 不代表负样本，只表示该正样本缺少明确实验类型标注。`complex_curation` 也不应直接当作结构界面证据使用。

## PPI 质量层级

`PPI_Tier` 是为了便于筛选而增加的粗粒度证据强度分层，不是绝对真值等级。

| PPI_Tier | 中文 | 数量 | 大致含义 |
|---|---|---:|---|
| `diamond` | 钻石 | 663,938 | 结构证据，或多来源支持的强实验类别。 |
| `gold` | 黄金 | 1,142,390 | LTP/HTP_LTP，或多来源 HTP，或高度多来源 mixed。 |
| `silver` | 白银 | 714,310 | HTP、多个来源 mixed，或多个来源 complex curation。 |
| `bronze` | 青铜 | 5,661,385 | `no_exp`、单来源 mixed、单来源 complex curation 或未知正样本证据。 |
| `negative_synthetic` | 负样本 | 535,381 | 合成负样本。 |

如果需要高置信度子集，优先考虑 `diamond/gold` 或 `n_sources >= 2` 的 pair；如果需要尽量大的覆盖范围，可以纳入 `bronze`，但需要意识到其中大量来自 `no_exp`。

## 来源构成

`PPI_Source` 是来源标签。一个去重 pair 可能有多个来源，因此下面是 source-tag 计数，不是互斥 pair 计数。

| Source | Source-tag count |
|---|---:|
| `PPIDB` | 7,639,122 |
| `DSCRIPT_human_train` | 421,342 |
| `HINT` | 230,473 |
| `PPI3D` | 210,229 |
| `PLM_interact` | 162,081 |
| `BERNETT_pos` | 137,250 |
| `BERNETT_neg` | 137,250 |
| `BIOGRID` | 61,931 |
| `DSCRIPT_mouse` | 54,963 |
| `DSCRIPT_worm` | 54,956 |
| `DSCRIPT_yeast` | 54,940 |
| `DSCRIPT_fly` | 54,912 |
| `PINDER` | 54,410 |
| `DSCRIPT_human_test` | 52,721 |
| `MINT` | 44,403 |
| `DSCRIPT_ecoli` | 18,225 |
| `PLMDA_PPI` | 4,679 |
| `PPIREF_10A_clust03` | 4,365 |
| `PepBDB` | 4,186 |
| `SKEMPI2` | 959 |
| `FoldBench` | 264 |

## 清洗规则摘要

- 蛋白质按完整氨基酸序列做 MD5 去重，同一序列只保留一个 `fpid`。
- Pair 按无序 `(protein A, protein B)` 去重。
- 如果同一个 pair 同时出现在正样本和合成负样本中，采用 positive wins。
- 已移除同序列 self-pair。
- PPIDB 的 `both` 归一化为 `HTP_LTP`，`no_exp` 单独保留。
- `Evidence_Type` 是单一清洗类别，`Evidence_Tags` 保留多来源证据并集。

## 相关统计文件

| 文件 | 内容 |
|---|---|
| `reports/merge_report.json` | 各来源合并前后的蛋白质和 pair 统计。 |
| `reports/dedup_report.json` | pair 去重、label conflict 和 `n_sources` 统计。 |
| `reports/dedup_no_selfseq_report.json` | self-sequence pair 移除统计。 |
| `reports/evidence_normalize_report.json` | `Evidence_Type` 和 `PPI_Tier` 清洗统计。 |
| `figures/nature/source_data/*.csv` | 当前数据库统计图对应的 source data。 |
