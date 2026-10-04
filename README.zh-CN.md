# Industry-LLM-Timeseires

中文 | [English](README.md)

工业时序 AI 学习仓库:以经典论文复现为主线,覆盖异常检测、预测、分类与时序表征(含 LLM 方向)等任务。

> AI Agent / 维护者入口:[AGENTS.md](AGENTS.md)(架构铁律、操作流程、验收清单)。

## 设计原则

1. **单一环境,目录级解耦**:全仓库共用一个 uv 环境(根目录 `.venv`),所有依赖集中在根 `pyproject.toml`;每个算法是一个自包含源码目录,不打包、不互相 import,删除任一算法目录不影响其他。
2. **三次原则**:同一段逻辑重复出现 3 次之前不抽共享代码,宁可少量复制,保持算法间零耦合。
3. **论文 ↔ 算法 ↔ 代码三段映射**:每篇论文有精读笔记(`docs/papers/`),每个算法 README 回链论文笔记,代码 docstring 标注论文章节/公式编号,总表见 `docs/README.md`。
4. **数据与代码分离,格式标准化**:数据集统一放 `data/` 并登记(`data/README.md`),数据文件不入 Git;预处理产物遵循统一格式(`train.csv`/`test.csv`/`meta.json`),算法经 `load_dataset(name)` 加载,**换数据集只改参数不改代码**;结果按 `outputs/<algo>/<dataset>/` 存放,除 `metrics.json`(含参数快照)外,迭代式算法必须产出 `history.csv`+`curves.png`,非迭代算法必须产出至少一张结果可视化图——数字之外对人直观;`scripts/collect_results.py` 自动汇总算法×数据集对比表到 `docs/benchmarks.md`。

## 目录结构

```
├── algorithms/             # 所有算法,按任务分类
│   └── anomaly_detection/
│       └── isolation_forest/   # 示例算法(兼作模板活实例)
├── templates/algorithm-template/  # 新算法模板
├── scripts/                # new_algorithm.py(脚手架)、test_all.py(全量测试)、
│                           # collect_results.py(结果汇总)、preprocess_<dataset>.py(数据预处理)
├── data/                   # 共享数据集(raw/ processed/,内容不入 Git)
│   └── README.md           # 数据集登记表
├── docs/
│   ├── README.md           # 论文↔算法映射总表
│   ├── papers/             # 论文精读笔记
│   └── benchmarks.md       # 同数据集多算法对比表
├── outputs/                # 运行产物(不入 Git)
└── pyproject.toml          # 全部依赖 + ruff 配置
```

预留任务分类(用到再建):`forecasting`(预测)、`classification`(分类)、`representation`(表征/LLM 方向)。

## 快速开始

```bash
uv sync                                          # 安装全部依赖(唯一环境)

# 运行示例算法(默认合成数据集;--dataset 切换已登记数据集)
cd algorithms/anomaly_detection/isolation_forest
uv run python train.py                           # 结果写入 outputs/isolation_forest/<dataset>/
uv run pytest -q                                 # 单算法测试

# 回到仓库根目录
cd ../../..
uv run python scripts/test_all.py                # 全部算法测试(逐目录子进程)
uv run python scripts/collect_results.py         # 汇总结果 -> docs/benchmarks.md
uv run ruff check .                              # 代码检查
```

## 新增一个算法

```bash
uv run scripts/new_algorithm.py <类别> <算法名>
# 例:uv run scripts/new_algorithm.py forecasting dlinear
```

然后(详见 [CONTRIBUTING.md](CONTRIBUTING.zh-CN.md)):

1. 填写新目录下 `README.md` 元信息(论文笔记链接、数据集、状态)
2. 在 `docs/papers/` 建论文笔记,并在 `docs/README.md` 总表登记一行
3. 实现 `data.py` / `model.py` / `train.py`,`cd` 进目录运行验证
4. 若需新依赖:根目录 `uv add <包名>`

## 维护规则速查

- 命名:目录、文件一律 `snake_case`;论文笔记 `<算法名>_<会议年份>.md`
- 测试:算法目录内 `uv run pytest`;全量用 `scripts/test_all.py`
- 数据:先入 `data/raw/` 并在 `data/README.md` 登记,预处理脚本放 `scripts/preprocess_<dataset>.py`
- 结果:运行后执行 `uv run python scripts/collect_results.py` 刷新对比表(勿手改 `docs/benchmarks.md`)
- 依赖:只准 `uv add`(禁止 `uv pip install`,会被 `uv sync` 清除);CUDA 版 PyTorch 已固定在 `pyproject.toml`
- 推送 GitHub 前:过一遍 [CONTRIBUTING.md](CONTRIBUTING.zh-CN.md) 的隐私检查清单

## License

[MIT](LICENSE)
