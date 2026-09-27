# isolation_forest

| 项目 | 内容 |
|---|---|
| 论文笔记 | [../../../docs/papers/isolation_forest_icdm2008.md](../../../docs/papers/isolation_forest_icdm2008.md) |
| 论文出处 | Isolation Forest / Liu, Ting, Zhou / ICDM 2008 / [doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) |
| 复现状态 | 已复现(基于 sklearn 成熟实现) |
| 任务类别 | anomaly_detection |
| 数据集 | synthetic-point-anomaly(内存合成,见 `data.py`) |
| 结果摘要 | 合成点异常检测 AP ≈ 0.99 / ROC-AUC ≈ 1.0,见 `outputs/isolation_forest/` |

## 原理简介

孤立森林通过"随机选特征、随机选切分点"递归隔离样本:异常点稀少且取值偏离,平均只需更少的切分次数即可被孤立(平均路径长度更短)。以平均路径长度相对期望值构造异常分数,见论文 Section 3,Eq. (1)-(3)。

## 数据

默认数据集 `synthetic-point-anomaly` 为内存生成(趋势 + 周期 + 高斯噪声 + 尖峰点异常),无需文件;真实数据集按仓库标准格式(`data/processed/<name>/{train.csv,test.csv,meta.json}`)经 `load_dataset()` 统一加载,换数据集只需改 `--dataset` 参数。特征为居中滑动窗口的残差与标准差(去趋势/季节化,见 `data.py`)。

## 运行方式

```bash
cd algorithms/anomaly_detection/isolation_forest
uv run python train.py                                  # 默认合成数据
uv run python train.py --dataset <登记名> --seed 0       # 换数据集/种子
```

## 结果

每次运行写入 `outputs/isolation_forest/<dataset>/`:`metrics.json`(指标与参数快照)+ `scores.png`(测试段序列/真实异常标注/异常分数双联图)。`uv run python scripts/collect_results.py` 将对比表自动刷新到 `docs/benchmarks.md`。

## 参考文献

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. ICDM 2008.
- scikit-learn: [`sklearn.ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

## TODO

- [ ] 接入真实工业数据集(如 MSL/SMAP/SMD),验证标准格式加载路径并登记 `docs/benchmarks.md`
