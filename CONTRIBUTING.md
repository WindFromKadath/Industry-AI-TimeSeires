# CONTRIBUTING

本仓库为个人学习仓库,以下约定用于保证长期的**可维护性、可复现性与隐私安全**。

## 一、新增算法流程

```bash
uv run scripts/new_algorithm.py <category> <algo_name>
```

脚手架会从 `templates/algorithm-template/` 复制骨架并替换占位符。之后按顺序完成:

1. **算法 README**:填写元信息表——论文笔记相对链接、论文出处、复现状态(`未开始/进行中/已复现`)、数据集登记名、结果摘要。
2. **论文笔记**:在 `docs/papers/` 新建 `<算法名>_<会议年份>.md`(模板见 `docs/README.md`),填写动机/方法/实验设置/复现要点。
3. **登记总表**:在 `docs/README.md` 的映射总表加一行:论文 | 论文笔记 | 算法实现 | 数据集 | 状态。
4. **实现代码**:`data.py`(加载,数据只从 `data/` 读)、`model.py`、`train.py`(指标写 `outputs/<algo>/metrics.json`)。
5. **代码↔论文标注**:关键类/函数的 docstring 注明对应论文的章节或公式编号,如 `Implements Eq. (3) of <paper>`。
6. **验证**:算法目录内 `uv run pytest -q` 通过;根目录 `uv run ruff check .` 零告警。

### 新增依赖

根目录执行 `uv add <包名>`(全仓库共享单一环境)。不要为单个算法创建独立环境。

**禁止使用 `uv pip install`**:它写入环境但不写入 `pyproject.toml`,`uv sync` 会把环境严格对齐到声明依赖并将其**清除**。所有依赖必须落在 `pyproject.toml` 里。

**CUDA 版 PyTorch 已固定**:通过 `[tool.uv.sources]` + `[[tool.uv.index]]` 指向官方 `cu130` 索引(适配 RTX 50 系),升级时改版本号即可,不要改用 `uv pip install` 覆盖。

**依赖冲突兜底**(预期极少):若某算法确实需要与其他算法冲突的版本,仅在该算法目录内 `uv init` 独立环境,并在其 README 中醒目标注;仓库其余部分保持不变。

## 二、数据与对比约定

- 数据统一放 `data/raw/<dataset>/`,预处理产物放 `data/processed/<dataset>/`;**数据文件不提交 Git**。
- 每个数据集必须在 `data/README.md` 登记:来源、许可、格式字段、预处理与划分方式——这是多算法结果可比的前提。
- 多算法对比结果人工汇总进 `docs/benchmarks.md`,指标取自各算法的 `outputs/<algo>/metrics.json`。

## 三、共享代码纪律(三次原则)

同一段逻辑在不同算法中重复满 **3 次**之前,不抽取共享模块。确需抽取时,先在 Issue/笔记中记录三处调用点,再考虑建 `libs/`(届时更新本文件与 README)。

## 四、提交前隐私检查清单

推送 GitHub 前逐项确认:

- [ ] 不含本机绝对路径(如 `C:\Users\<用户名>`、用户主目录)
- [ ] 不含真实姓名、私人邮箱、电话等个人信息(作者署名除外)
- [ ] 不含 API Key / Token / 密码(检查 `sk-`、`AKIA`、`ghp_` 等前缀;`.env` 已被 gitignore)
- [ ] 不含数据文件与模型权重(`data/raw/`、`data/processed/`、`outputs/`、`*.pt` 等已 gitignore)
- [ ] 不含论文 PDF(`*.pdf` 已 gitignore,笔记中只放链接)
- [ ] `git status` 待提交清单逐项过目

## 五、代码风格

- 由 ruff 统一检查(`uv run ruff check .`),行宽 88,import 排序自动修复:`uv run ruff check --fix .`
- 注释与文档用中文,代码标识符用英文
