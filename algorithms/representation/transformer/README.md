# transformer

| 项目 | 内容 |
|---|---|
| 论文笔记 | [../../../docs/papers/transformer_neurips2017.md](../../../docs/papers/transformer_neurips2017.md) |
| 论文出处 | Attention Is All You Need / Vaswani et al. / NeurIPS 2017 / [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 复现状态 | 已复现(架构级:合成复制任务 + 真实文本续写) |
| 任务类别 | representation |
| 数据集 | synthetic-copy-task(内存合成)、wikitext-2-raw(真实英文文本,见 `data/README.md`) |
| 结果摘要 | 复制任务 test token/序列准确率 1.0;WikiText-2 续写 test token 准确率 0.2125、困惑度 150.2(默认小模型 8 epoch),详见 `outputs/transformer/<dataset>/metrics.json` |

## 原理简介

Transformer 完全抛弃循环与卷积,仅依靠注意力机制做序列转导:多头缩放点积注意力(Section 3.2,Eq. (1))以 O(1) 最大路径长度并行建模任意距离依赖;位置逐点 FFN(Section 3.3,Eq. (2))提供逐位置非线性变换;正弦位置编码(Section 3.5)注入顺序信息。编码器-解码器各堆叠 N 个相同层,子层统一为 `LayerNorm(x + Sublayer(x))` 残差结构(Section 3.1);解码器自注意力加后续掩码保持自回归(Section 3.2.3)。训练用 Adam + Noam 学习率(Section 5.3,Eq. (3))与 label smoothing(Section 5.4)。

## 数据

- `synthetic-copy-task`(默认,内存合成):长度 10 的随机符号序列(0=PAD、1=BOS、2..15 为符号,词表 16),目标是原样复制源序列(objective=copy);9600 条训练 / 1024 条测试。这是 encoder-decoder 架构的经典冒烟任务(The Annotated Transformer)。
- `wikitext-2-raw`(真实文本):WikiText-2 raw 单词级英文文本,经 `scripts/preprocess_wikitext_2_raw.py` 切为定长 64 token 窗口(objective=continuation):前 32 个 token 作为编码器上下文,后 32 个为解码器续写目标;词表 16000。注意续写目标下编码器只看上下文、不看答案,无信息泄漏。

切换数据集只改 `--dataset` 参数,无需改代码;新数据集按 `data/README.md` 标准格式登记即可。

## 运行方式

```bash
cd algorithms/representation/transformer
uv run python train.py                              # 默认合成数据集
uv run python train.py --dataset wikitext-2-raw --epochs 8 --batch-size 128
```

默认配置为适合合成任务的小模型(N=2、d_model=128、h=4、d_ff=512);论文 base 配置(N=6、d_model=512、h=8、d_ff=2048、warmup=4000)可用 CLI 参数复现,如 `--n-layers 6 --d-model 512 --n-heads 8 --d-ff 2048 --warmup-steps 4000`。

## 结果

每次运行写入 `outputs/transformer/<dataset>/`:`metrics.json`(指标与参数快照)+ `history.csv`(每轮训练损失/学习率)+ `curves.png`(训练曲线图)。运行 `uv run python scripts/collect_results.py` 后对比表自动刷新到 `docs/benchmarks.md`。

- `synthetic-copy-task`:默认配置 30 个 epoch 收敛到 test token 准确率 1.0 / 序列准确率 1.0。
- `wikitext-2-raw`:默认小模型 8 个 epoch,test token 准确率 0.2125、困惑度 150.2——远高于随机基线(词表 16000,随机准确率约 6e-5),且训练损失持续下降,说明真实文本上的续写链路工作正常;数值不可与论文翻译任务直接比较。

## 参考文献

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Harvard NLP:[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)(复制任务出处)
- 原始实现:[tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## TODO

- [ ] 增大模型或训练步数压低 WikiText-2 困惑度;接入时序预测数据集后复用本架构做 forecasting 对比
