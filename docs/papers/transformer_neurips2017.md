# Attention Is All You Need

| 项目 | 内容 |
|---|---|
| 标题 | Attention Is All You Need |
| 作者 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin |
| 发表 | NeurIPS (NIPS) 2017 |
| 链接 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 算法实现 | [../../algorithms/representation/transformer/](../../algorithms/representation/transformer/) |

## 动机

RNN/LSTM 序列转导模型沿符号位置逐步计算,训练无法并行,长序列时尤甚;卷积方案(ByteNet、ConvS2S)虽可并行,但关联两个任意位置的操作数随距离增长(线性或对数),难以学习远程依赖。注意力机制建模依赖与距离无关,本文提出完全抛弃循环与卷积、仅基于注意力的 Transformer:任意两位置间最大路径长度 O(1),并行度最高。

## 方法

- **整体结构(Figure 1)**:encoder-decoder。编码器/解码器各 N=6 个相同层;每个子层外包裹残差连接 + 层归一化,即 `LayerNorm(x + Sublayer(x))`(Section 3.1),所有子层与 embedding 输出维度 d_model=512。
- **Scaled Dot-Product Attention(Section 3.2.1, Eq. (1))**:`Attention(Q,K,V) = softmax(QK^T / √d_k) V`;缩放因子 1/√d_k 防止 d_k 较大时点积量级过大把 softmax 压入梯度极小区域。
- **Multi-Head Attention(Section 3.2.2)**:将 Q/K/V 分别线性投影 h=8 次(d_k=d_v=d_model/h=64),并行做注意力后拼接再投影 W^O;不同头可在不同表示子空间、不同位置共同注意,单头的平均化会抑制这一点。
- **三种注意力用法(Section 3.2.3)**:编码器自注意力;解码器自注意力(对 softmax 输入中非法连接置 -inf 实现后续掩码,配合输出右移一位保证自回归);编码器-解码器交叉注意力(Q 来自解码器,K/V 来自编码器输出)。
- **Position-wise FFN(Section 3.3, Eq. (2))**:`FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`,对每个位置独立同参数地施加;d_ff=2048。
- **Embedding 与 Softmax(Section 3.4)**:输入/输出 embedding 与 pre-softmax 线性层共享同一权重矩阵;embedding 层权重乘以 √d_model。
- **位置编码(Section 3.5)**:正弦/余弦 `PE(pos,2i)=sin(pos/10000^(2i/d_model))`、`PE(pos,2i+1)=cos(...)`;波长从 2π 到 10000·2π 几何递进,可外推到比训练更长的序列(与可学习位置 embedding 效果几乎相同,Table 3 行 E)。

## 实验设置

- **数据**:WMT 2014 英德(约 4.5M 句对,BPE 共享词表约 37K)、WMT 2014 英法(36M 句对,32K word-piece);按近似序列长度组 batch,每 batch 约 2.5 万源 + 2.5 万目标 token。
- **硬件与时长**:单机 8×P100;base 模型 10 万步(12 小时,0.4s/步),big 模型 30 万步(3.5 天,1.0s/步)。
- **优化器(Section 5.3)**:Adam(β1=0.9, β2=0.98, ε=1e-9),学习率 Eq. (3):`lr = d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)`,warmup=4000(线性预热后按步数平方根倒数衰减)。
- **正则(Section 5.4)**:残差 dropout P_drop=0.1(子层输出、embedding+位置编码之和上;big EN-FR 用 0.3);label smoothing ε_ls=0.1(损害 PPL 但提升准确率与 BLEU)。
- **结果(Table 2)**:EN-DE base 27.3 / big 28.4 BLEU(超过此前最佳含集成 2+ BLEU);EN-FR big 41.8 BLEU,单模型 SOTA,训练成本仅为零头。消融(Table 3):单头注意力差 0.9 BLEU,d_k 过小伤性能,dropout 至关重要。迁移到英语成分句法分析(Table 4):仅 WSJ 小数据 91.3 F1,半监督 92.7 F1。

## 复现要点

- 本仓库复现聚焦**架构正确性**而非 WMT 数值:在合成复制任务(随机符号序列 → 原样输出)上完整实现 Figure 1 全部组件(Eq. (1)-(3)、残差+LayerNorm、正弦位置编码、embedding 权重共享 ×√d_model),测试集 token/序列准确率达 1.0 即说明编码器-解码器、注意力与掩码机制实现无误。
- 正确性关键在解码器的 subsequent mask:位置 i 的预测只能依赖 < i 的已知输出;本实现有专门测试锁定该性质(`tests/test_smoke.py::test_decoder_is_autoregressive`)。
- 训练配方(Eq. (3) 学习率 + label smoothing + 残差 dropout)对小数据任务并非必需,但按论文实现以保持配方可迁移;复现翻译数值时需严格沿用 Section 5 全部设置。
- 原始实现为 tensor2tensor(TensorFlow,论文结论末附仓库链接);2017 年发表、无独立官方 PyTorch 代码,数值结论不可逐字对齐,关注机制与趋势一致性。
- 除合成复制任务外,已在真实英文文本(WikiText-2 raw)上以**续写**目标训练:定长 64 token 窗口的前 32 个 token 作编码器上下文、后 32 个作解码器目标(编码器不含答案,无泄漏);默认小模型 8 epoch 得 test token 准确率 0.2125、困惑度 150.2(见 `docs/benchmarks.md`),验证真实数据训练链路。
