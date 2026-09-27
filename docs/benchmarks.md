# 基准对比

<!-- 本文件由 scripts/collect_results.py 自动生成,请勿手改 -->

同一数据集上多个算法的结果对比,指标取自 `outputs/<algo>/<dataset>/metrics.json`。

## synthetic-point-anomaly

| 算法 | average_precision | roc_auc | 参数 | 运行日期 |
|---|---|---|---|---|
| isolation_forest | 1.0 | 1.0 | `{"seed": 7, "n_estimators": 200, "contamination": 0.03, "window": 24}` | 2026-09-27 |
