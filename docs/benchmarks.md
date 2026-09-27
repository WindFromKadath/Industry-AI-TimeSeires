# 基准对比

同一数据集上多个算法的结果对比。指标取自各算法 `outputs/<algo>/metrics.json`,人工汇总;登记时注明数据预处理版本与划分方式(见 `data/README.md`)。

## synthetic-point-anomaly(内存合成,仅冒烟验证)

| 算法 | AP | ROC-AUC | 备注 |
|---|---|---|---|
| isolation_forest | 见 outputs | 见 outputs | sklearn 实现,非论文数值复现 |

<!-- 新增真实数据集时,按上方格式为每个数据集建一节 -->
