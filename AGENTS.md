# AGENTS.md — 维护 Agent 入口

本文件是 AI Agent(及新维护者)接手本仓库的**唯一入口**:读完本文件即可开始工作;更细的规范见 [CONTRIBUTING.md](CONTRIBUTING.md),项目全貌见 [README.md](README.md)。

## 1. 项目定位

工业时序 AI 个人学习仓库,以**经典论文复现**为主线。任务方向:异常检测 `anomaly_detection`、预测 `forecasting`、分类 `classification`、时序表征/LLM `representation`。GitHub 公开仓库,作者署名 `WindFromKadath`。

## 2. 架构铁律(改动前必读)

1. **单一 uv 环境**:所有依赖集中在根 `pyproject.toml`,全仓库共用一个 `.venv`。新增依赖只能在根目录 `uv add <pkg>`。**禁止**为单个算法创建独立环境或子项目(算法目录内不得出现 `pyproject.toml`)。
2. **目录级解耦**:每个算法是 `algorithms/<类别>/<算法名>/` 下的自包含纯源码目录(无 `__init__.py`、不打包、不发布)。算法之间**禁止互相 import**;小量工具代码宁可复制。
3. **三次原则**:同一段逻辑重复满 3 次才允许抽共享代码;抽取前先在 `docs/` 记录三处调用点。
4. **数据与代码分离**:数据只放 `data/`(文件不入 Git),算法只读 `data/`、写 `outputs/<算法名>/`。
5. **三段映射同步**:论文笔记(`docs/papers/`)↔ 算法 README 元信息 ↔ 代码 docstring 公式标注,三者必须一致;新增算法必须在 `docs/README.md` 总表登记。

## 3. 目录地图

```
algorithms/<类别>/<算法名>/   # 算法:model.py / data.py / train.py / paths.py / config.py / README.md / tests/
templates/algorithm-template/ # 新算法模板(脚手架的复制源)
scripts/new_algorithm.py      # 新建算法脚手架
scripts/test_all.py           # 全量测试(逐算法目录子进程)
data/README.md                # 数据集登记表(raw/ processed/ 内容不入 Git)
docs/README.md                # 论文↔算法映射总表 + 论文笔记模板
docs/papers/                  # 论文精读笔记
docs/benchmarks.md            # 同数据集多算法对比表
outputs/                      # 运行产物(不入 Git)
```

## 4. 常用命令

```bash
uv sync                                             # 同步环境(首次/依赖变更后)
uv run scripts/new_algorithm.py <类别> <算法名>      # 新建算法骨架
cd algorithms/<类别>/<算法名> && uv run python train.py   # 运行算法
cd algorithms/<类别>/<算法名> && uv run pytest -q         # 单算法测试
uv run python scripts/test_all.py                   # 全量测试
uv run ruff check .                                 # 检查;加 --fix 自动修
```

**必须在算法目录内跑 pytest**:各算法目录存在同名 `model.py`/`data.py`,同进程 pytest 会模块名冲突;`test_all.py` 用子进程隔离,不要在根目录直接 `pytest`。

## 5. 新增算法操作顺序

1. `uv run scripts/new_algorithm.py <类别> <算法名>`(`snake_case`;新类别会自动建目录)
2. 涉及论文:在 `docs/papers/` 新建 `<算法名>_<会议年份>.md`(模板在 `docs/README.md`)
3. 填算法 README 元信息表(论文笔记相对链接、出处、复现状态、数据集、结果摘要)
4. 在 `docs/README.md` 总表登记一行
5. 实现 `data.py` / `model.py` / `train.py`;关键实现的 docstring 标注论文章节/公式编号
6. 用真实数据:放入 `data/raw/<dataset>/` 并在 `data/README.md` 登记
7. 验证(见第 8 节)后更新 README 的复现状态与结果摘要;有对比结果登记 `docs/benchmarks.md`

## 6. 隐私红线(提交 Git 前)

`.gitignore` 已覆盖 `.vscode/`、数据、权重、`outputs/`、PDF、`.env`,但仍须复核 `git status`:

- 禁止出现:本机绝对路径(如 `C:\Users\<用户名>`)、真实姓名、私人邮箱、API Key/Token
- `pyproject.toml` 作者只保留 `WindFromKadath`,不得加回邮箱
- 论文只放链接,PDF 不入库

## 7. 代码风格

- ruff:line-length 88,规则集 E4/E7/E9/F/I;标识符英文,注释与文档中文
- 命名:目录/文件 `snake_case`;论文笔记 `<算法名>_<会议年份>.md`
- 不写复述代码的注释;改动行为后同步更新受影响的 README、笔记与总表

## 8. 交付前验收清单

- [ ] `uv sync` 成功
- [ ] 新增/修改的算法 `train.py` 实跑成功,指标写入 `outputs/`
- [ ] `uv run python scripts/test_all.py` 全绿
- [ ] `uv run ruff check .` 零告警
- [ ] 三段映射已同步(论文笔记 / 算法 README / `docs/README.md` 总表)
- [ ] `git status` 中无隐私内容、无数据/权重/产物文件
