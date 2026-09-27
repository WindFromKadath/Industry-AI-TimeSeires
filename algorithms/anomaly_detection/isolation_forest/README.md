# isolation_forest

| 项目 | 内容 |
|---|---|
| 论文笔记 | [../../../docs/papers/isolation_forest_icdm2008.md](../../../docs/papers/isolation_forest_icdm2008.md) |
| 论文出处 | Isolation Forest / Liu, Ting, Zhou / ICDM 2008 / [doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) |
| 复现状态 | 已复现(基于 sklearn 成熟实现) |
| 任务类别 | anomaly_detection |
| 数据集 | synthetic-point-anomaly(内存合成,见 `data.py`) |
| 结果摘要 | 合成点异常检测 AP / ROC-AUC 见 `outputs/isolation_forest/metrics.json` |

## 原理简介

孤立森林通过"随机选特征、随机选切分点"递归隔离样本:异常点稀少且取值偏离,平均只需更少的切分次数即可被孤立(平均路径长度更短)。以平均路径长度相对期望值构造异常分数,见论文 Section 3,Eq. (1)-(3)。

## 数据

内存生成的合成单变量时序(趋势 + 周期 + 高斯噪声),随机注入大幅尖峰作为点异常;以**居中滑动窗口**去趋势/季节化,取残差与窗口标准差构造特征(见 `data.py`)。

## 运行方式

```bash
cd algorithms/anomaly_detection/isolation_forest
uv run python train.py
```

## 结果

训练后指标写入 `outputs/isolation_forest/metrics.json`,终端同步打印。

## 参考文献

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. ICDM 2008.
- scikit-learn: [`sklearn.ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

## TODO

- [ ] 接入 `data/` 中的真实工业数据集后,在 `docs/benchmarks.md` 登记对比结果
