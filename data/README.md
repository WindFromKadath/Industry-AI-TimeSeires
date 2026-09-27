# 数据集登记表

所有数据集统一存放于本目录。**数据文件不入 Git**(见 `.gitignore`),本表是唯一需要提交的内容。

## 目录约定

- `raw/<dataset>/` — 原始数据,只增不改,保持来源原貌
- `processed/<dataset>/` — 预处理产物,须符合下方**标准格式**;预处理脚本放 `scripts/preprocess_<dataset>.py`
- 算法只**读** `data/`,运行结果写入 `outputs/<algo>/<dataset>/`,算法目录内不放数据

## 预处理产物标准格式

算法的 `load_dataset(name)` 只依赖"登记名 + 本格式",换数据集不改代码:

```
data/processed/<dataset>/
├── train.csv     # 训练段(异常检测任务可为纯正常段)
├── test.csv      # 测试段(含标签列)
└── meta.json     # 见下
```

`meta.json` 字段:

| 字段 | 含义 |
|---|---|
| `task` | anomaly_detection / forecasting / classification / representation |
| `value_column` | 取值列名(多变量时为列名列表) |
| `label_column` | 标签列名(无标签任务可省略) |
| `preprocessing` | 预处理说明(缩放、去趋势等)与对应脚本路径 |
| `split` | 划分方式说明 |
| `source` | 原始数据来源 |

## 数据集

| 登记名 | 来源/链接 | 许可 | 格式与字段 | 适用任务 | 预处理与划分 |
|---|---|---|---|---|---|
| synthetic-point-anomaly | 内存合成,无文件(见 `algorithms/anomaly_detection/isolation_forest/data.py`) | — | 单变量时序 + 0/1 点异常标签 | 异常检测 | 居中滑窗(24)特征;前 70% 训练 / 后 30% 测试 |

新增数据集时在此表登记一行;下载链接失效时应及时更新。
