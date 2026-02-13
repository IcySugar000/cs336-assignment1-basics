# 2. BPE

## unicode1

### a

`\x00`，空字符，表示字符串结束

### b

- repr: `'\x00'`
- printed: 不显示

### c

不显示该字符，其他字符正常显示

## unicode2

### a

utf-16与utf-32编码的字符串太长，计算负担更大

### b

`"你好"`

有的字符会跨多个byte，不能单个单个解析

### c

`0b11000001, 0b11111111`

二字节utf-8要求形如`0b110xxxxx, 0b10xxxxxx`，同时首字节不能为`0xc0`或`0xc1`

## train_bpe_tinystories

### a

- 在pretokenize时用大约8G内存，merge时大约用1.5G
- 最长的token是`"Ġaccomplishment"`，有意义

### b

- pretokenize过程中的正则表达式匹配

## train_bpe_expts_owt

### a

`Ġ-----------------------`, no

### b

- 两者词表都主要由英文词构成，许多词有前导空格
- owt有显著更多的nonsense词汇

## tokenizer_experiments

### a

- TinyStories: 4.16
- OpenWebText: 4.56

### b

压缩率下降到3.60

### c

- 9231250 bytes / second
- 约1.1天

### d

多数词表的长度都在uint16范围内，且uint16是满足条件的足够小的类型，节省空间

# 3. Transformer Language Model Architecture

## transformer_accounting

### a

| 模块 | 参数量 |
| --- | --- |
| Embedding | 50257 * 1600 = 80411200 |
| MHA | 1600 * 1600 * 4 = 10240000 |
| FFN | 1600 * 6400 * 3 = 30720000 |
| Norm | 1600 |
| Transformer Block | 10240000 + 30720000 + 1600 * 2 = 40963200 |
| Total | 80411200 + 40963200 * 48 + 1600 + 1600 * 50257 = 2127057600 |

使用FP32，占用内存约为7.9G

### b

| 模块 | 矩阵乘法 | 尺寸 | FLOPs | 
| --- | --- | --- | --- |
| MHA | W_Q*x | (1600, 1600), (1600, 1024) | 5242880000 |
|     | W_K*x | (1600, 1600), (1600, 1024) | 5242880000 |
|     | W_V*x | (1600, 1600), (1600, 1024) | 5242880000 |
|     | Q*K_T | (1024, 1600), (1600, 1024) | 3355443200 |
|     | softmax(QK_T)*V | (1024, 1024), (1024, 1600) | 3355443200 |
|     | W_O*Heads | (1600, 1600), (1600, 1024) | 5242880000 |
| FFN | W1 * x | (6400, 1600), (1600, 1024) | 20971520000 |
|     | W3 * x | (6400, 1600), (1600, 1024) | 20971520000 |
|     | W2 * (...) | (1600, 6400), (6400, 1024) | 20971520000 |

单层共`90596966400` FLOPs，48层共约`4.34`TFLOPs

### c

FFN

### d

| 模型 | 模块 | 比例 |
| --- | --- | --- |
| Small | MHA | 21% |
|       | FFN | 79% |
| Medium | MHA | 24% |
|        | FFN | 76% |
| Large | MHA | 27% |
|       | FFN | 73% |

随着模型尺寸变大，MHA占用的FLOPs比例逐渐增大，而FFN逐渐减小

### e

| Context Length | 总FLOPs | 模块 | 模块FLOPs | 模块FLOPs比例 |
| - | - | - | - | - |
| 1600 | 4.34T | MHA | 27.7G | 31% |
|      |       | FFN | 62.9G | 69% |
| 16384 | 139.78T | MHA | 2267.7G | 78% |
|       |         | FFN | 644.2G | 22% |

MHA对FLOPs的贡献大幅上升，而FFN大幅缩小

# 4. Training a Transformer LM

## learning_rate_tuning

- 在lr=1, 10, 100时，loss逐渐下降，且lr越大loss下降越快
- 在lr=1000时，loss一开始下降到很低，然后逐渐上升

## adamwAccounting

### a

- V = vocab_size
- L = context_length
- d = d_model
- N = num_layers
- H = num_heads

Parameters:
- MHA: $4Nd^2$
- FFN: $4Nd\times\text{d\_ff} = 12Nd^2$
- RMSNorm: $(2N+1)d$
- Embedding: $2Vd$
- Total: $16Nd^2 + (2N+2V+1)d$
- Total(mem): $64Nd^2 + (8N+8V+4)d$

Activations:
- RMSNorm: $(2N+1)BLd$
- QKV: $3NBLd$
- Q(T)K: $NBHL^2$
- softmax: $NBHL^2$
- Attn-V: $NBLd$
- output-proj: $NBLd$
- FFN: $4NBLd + 4NBLd + 4NBLd + 4NBLd + NBLd = 17NBLd$
- Embedding: $BLd$
- logits: $BLV$
- Total: $(24N+2)BLd + 2NBHL^2 + BLV$
- Total(mem): $(96N+8)BLd + 8NBHL^2 + 4BLV$

Gradients:
- Total: $16Nd^2 + (2N+2V+1)d$
- Total(mem): $64Nd^2 + (8N+8V+4)d$

Optimizer State:
- Total: $32Nd^2 + (4N+4V+2)d$
- Total(mem): $128Nd^2 + (16N+16V+8)d$

Total:
- Params: $256Nd^2 + (32N+32V+16)d + (96N+8)BLd + 8NBHL^2 + 4BLV$

### b

$17835036672B + 34032921600$

2.98 -> 2

### c

$14(16Nd^2 + (2N+2V+1)d)$

AdamW每个参数有14个FLOP，乘以总参数量即可

### d

- Forward: 4.34TFLOPs
- Backward: 8.68TFLOPs
- AdamW(per step): 0.027TFLOPs(忽略)

$400000\times1024\times13.02\div(19.5\times0.5)\div(3600\times24) \approx 6331$ 天