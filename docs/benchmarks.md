# 基准对比

<!-- 本文件由 scripts/collect_results.py 自动生成,请勿手改 -->

同一数据集上多个算法的结果对比,指标取自 `outputs/<algo>/<dataset>/metrics.json`。

## synthetic-copy-task

| 算法 | final_train_loss | test_perplexity | test_sequence_accuracy | test_token_accuracy | 参数 | 运行日期 |
|---|---|---|---|---|---|---|
| transformer | 0.5746 | 1.1 | 1.0 | 1.0 | `{"seed": 42, "epochs": 30, "batch_size": 64, "d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 512, "dropout": 0.1, "warmup_steps": 400, "label_smoothing": 0.1}` | 2026-09-27 |

## synthetic-point-anomaly

| 算法 | average_precision | roc_auc | 参数 | 运行日期 |
|---|---|---|---|---|
| isolation_forest | 0.9932 | 0.9998 | `{"seed": 42, "n_estimators": 200, "contamination": 0.03, "window": 24}` | 2026-09-27 |

## wikitext-2-raw

| 算法 | final_train_loss | test_perplexity | test_token_accuracy | 参数 | 运行日期 |
|---|---|---|---|---|---|
| transformer | 5.3769 | 150.2 | 0.2125 | `{"seed": 42, "epochs": 8, "batch_size": 128, "d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 512, "dropout": 0.1, "warmup_steps": 400, "label_smoothing": 0.1}` | 2026-09-27 |
