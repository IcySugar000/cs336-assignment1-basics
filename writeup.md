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