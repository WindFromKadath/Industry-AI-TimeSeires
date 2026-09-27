# 提示词工程手册(维护用)

本文件固化维护本仓库的**标准提示词模板**。用法:新开 AI 会话后,先确认 Agent 已读取根目录 `AGENTS.md`(Kimi Code 等会自动读取),再从下方复制对应任务的模板,替换 `<占位符>` 后发出。

## 通用原则(所有模板的隐含前提)

- Agent 必须先读 `AGENTS.md`,遵守其中 6 条架构铁律
- 完成标准 = 通过 `AGENTS.md` 第 8 节验收清单
- 提交 Git 前必须过隐私红线检查,且先列变更清单等人工确认
- 明令禁止:`uv pip install`、手改 `docs/benchmarks.md`、算法间互相 import、未满三次抽共享代码、提交数据/权重/产物文件

## 模板 1:复现一篇论文(新增算法)

```text
请在本仓库复现论文 <论文标题>(<论文链接>),任务类别 <类别>,算法名 <算法名>。
要求:
1. 先阅读 AGENTS.md 并严格遵守其中铁律
2. 用 uv run scripts/new_algorithm.py <类别> <算法名> 建骨架
3. 在 docs/papers/ 写论文笔记(模板见 docs/README.md);算法 README 填元信息并与论文笔记互链;docs/README.md 总表登记一行
4. 实现 data.py / model.py / train.py(argparse 入口,支持 --dataset/--seed),关键实现 docstring 标注论文章节/公式编号
5. 迭代训练必须写 history.csv 并用本算法 plot.py 渲染 curves.png;非迭代算法输出至少一张结果可视化图
6. 验证:算法目录内 uv run pytest -q;根目录 uv run ruff check .、uv run python scripts/test_all.py、uv run python scripts/collect_results.py 全部通过
7. 实跑训练确认收敛,把关键结果写入算法 README 结果摘要
8. 列出变更清单,先不要 git commit,等我确认
```

## 模板 2:接入新数据集

```text
请为仓库接入数据集 <名称>(<来源/链接>),适用任务 <任务类别>:
1. 先阅读 AGENTS.md 与 data/README.md 的标准格式约定
2. 原始数据放到 data/raw/<名称>/ 并在 data/README.md 登记表登记(数据文件不提交 Git)
3. 新建 scripts/preprocess_<名称>.py,产出标准格式 data/processed/<名称>/{train.csv,test.csv,meta.json}
4. 用算法 <算法名> 实跑验证:cd algorithms/<类别>/<算法名> && uv run python train.py --dataset <名称>,确认产物齐全(metrics.json + 可视化)
5. uv run python scripts/collect_results.py 刷新对比表
6. 列出变更清单,先不要 git commit,等我确认
```

## 模板 3:多算法 × 多数据集对比实验

```text
请执行对比实验:算法 <算法A、算法B、...> × 数据集 <数据集X、数据集Y、...>:
1. 确认每个数据集已在 data/README.md 登记且符合标准格式
2. 逐组合运行各算法的 train.py --dataset <名称>(必要时加 --seed 做多 seed),确保每次运行产物齐全
3. 运行 uv run python scripts/collect_results.py 生成对比表
4. 基于 docs/benchmarks.md 的结果,向我总结:每个数据集上哪个算法更好、差距多大、可能原因;不要手改 benchmarks.md
```

## 模板 4:审查算法合规性

```text
请审查 algorithms/<类别>/<算法名> 是否符合仓库规范:
1. 对照 AGENTS.md 第 2 节铁律与第 8 节验收清单逐项检查
2. 检查三段映射是否同步:docs/papers/ 笔记、算法 README 元信息、docs/README.md 总表、代码 docstring 公式标注
3. 检查输出产物是否齐全:metrics.json、history.csv、curves.png(或等效可视化)
4. 运行验证:算法目录内 uv run pytest -q;根目录 uv run ruff check .
5. 输出合规项/违规项清单与修复建议,先不要改代码,等我确认
```

## 模板 5:修改框架本身(模板 / scripts / 约定)

```text
请修改仓库框架级内容:<要改的内容与原因>:
1. 先阅读 AGENTS.md,分析该改动影响哪些既有约定(铁律/流程/文档),先把影响面告诉我
2. 改动后同步更新所有受影响文档(AGENTS.md、CONTRIBUTING.md、README.md、templates/algorithm-template/)
3. 用现有算法(isolation_forest、transformer)回归验证:uv run python scripts/test_all.py、uv run ruff check .、uv run python scripts/collect_results.py
4. 列出变更清单,先不要 git commit,等我确认
```

## 占位符速查

| 占位符 | 取值 |
|---|---|
| `<类别>` | anomaly_detection / forecasting / classification / representation |
| `<算法名>` | 小写 snake_case,与目录名一致 |
| `<论文链接>` | arXiv 或 DOI 链接(PDF 不入库) |
| `<名称>`(数据集) | 小写 snake_case,与 `data/README.md` 登记名一致 |
