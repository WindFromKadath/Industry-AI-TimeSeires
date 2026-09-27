# {{ALGO_NAME}}

| 项目 | 内容 |
|---|---|
| 论文笔记 | [../../../docs/papers/&lt;paper_key&gt;.md](../../../docs/papers/<paper_key>.md) |
| 论文出处 | <标题 / 作者 / 年份 / 会议或期刊 / 链接> |
| 复现状态 | 未开始 |
| 任务类别 | <anomaly_detection / forecasting / classification / representation> |
| 数据集 | <data/README.md 中的登记名;合成数据请注明> |
| 结果摘要 | <关键指标,与论文或基准的对照> |

## 原理简介

<用 3-5 句话概括核心思想,注明关键公式编号>

## 数据

<数据来源、字段、预处理方式;真实数据一律从 `data/` 读取>

## 运行方式

```bash
cd algorithms/<category>/{{ALGO_NAME}}
uv run python train.py --dataset <数据集登记名>
uv run python train.py --dataset <数据集登记名> --seed 0   # 改参数复跑
```

## 结果

每次运行写入 `outputs/{{ALGO_NAME}}/<dataset>/metrics.json`;运行 `uv run python scripts/collect_results.py` 后对比表自动刷新到 `docs/benchmarks.md`。

## 参考文献

<论文、代码仓库等链接>

## TODO

- [ ] <待办>
