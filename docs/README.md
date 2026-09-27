# 文档与笔记索引

## 论文 ↔ 算法 ↔ 代码 映射总表

| 论文 | 论文笔记 | 算法实现 | 数据集 | 状态 |
|---|---|---|---|---|
| Isolation Forest (ICDM 2008) | [isolation_forest_icdm2008](papers/isolation_forest_icdm2008.md) | [anomaly_detection/isolation_forest](../algorithms/anomaly_detection/isolation_forest/) | synthetic-point-anomaly | 已复现 |
| Attention Is All You Need (NeurIPS 2017) | [transformer_neurips2017](papers/transformer_neurips2017.md) | [representation/transformer](../algorithms/representation/transformer/) | synthetic-copy-task / wikitext-2-raw | 已复现(合成复制任务 + 真实文本续写) |

> 新增算法时,务必在本表登记一行(流程见 [CONTRIBUTING.md](../CONTRIBUTING.md))。

## 目录说明

- `papers/` — 论文精读笔记,文件名用 `<算法名>_<会议年份>.md`(小写 snake_case)
- `benchmarks.md` — 同一数据集上多算法的对比结果表

## 论文笔记模板

复制以下骨架新建 `papers/<paper_key>.md`:

```markdown
# <论文标题>

| 项目 | 内容 |
|---|---|
| 标题 | |
| 作者 | |
| 发表 | <会议/期刊 + 年份> |
| 链接 | <arXiv 或 DOI> |
| 算法实现 | [../../algorithms/<category>/<algo>/](../../algorithms/<category>/<algo>/) |

## 动机
## 方法
## 实验设置
## 复现要点
```
