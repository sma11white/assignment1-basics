# CS336 作业 1(基础):构建 Transformer 语言模型

**版本 26.0.3**

CS336 课程组
2026 年春季

> 本文档为 `cs336_assignment1_basics.pdf`(Version 26.0.3)的中文翻译,供学习参考。代码、公式、专有名词与英文示例文本保留原文形式;如与英文原版有出入,以英文原版为准。

## 1 作业概览

在本作业中,你将从头构建训练一个标准 Transformer 语言模型(LM)所需的全部组件,并训练一些模型。

**你将实现的内容**

1. 字节对编码(BPE)分词器(第 2 节)
2. Transformer 语言模型(LM)(第 3 节)
3. 交叉熵损失函数与 AdamW 优化器(第 4 节)
4. 训练循环,并支持序列化与加载模型和优化器状态(第 5 节)

**你将运行的内容**

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器对数据集进行编码,将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行样本生成与困惑度评估。
5. 在 OpenWebText 上训练模型,并将你取得的困惑度提交到排行榜。

**你可以使用的内容**

我们希望你从头构建每一个组件。特别地,除了以下内容外,你不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义:

- `torch.nn.Parameter`
- `torch.nn` 中的容器类(例如 `Module`、`ModuleList`、`Sequential` 等)。[^1]
- `torch.optim.Optimizer` 基类

你可以使用任何其他 PyTorch 定义。如果你想使用某个函数或类但不确定是否被允许,欢迎在 Slack 上提问。拿不准时,请考虑使用它是否会损害本作业"从头实现"的精神。

[^1]: 完整列表见 pytorch.org/docs/stable/nn.html#containers。

**关于 AI 工具的声明**

AI 可以完全自主地解决作业的许多部分。这会使你更难深入参与课程材料并从中学习。

允许使用 AI 工具回答高层概念性问题,或提供底层编程文档(如函数签名和库 API)。但是,不允许使用 AI 工具实现任何作业的任何部分。这既包括编程智能体(例如 Cursor Agents、Codex、Claude Code),也包括 AI 自动补全(例如 Cursor Tab、GitHub Copilot)。使用 AI 智能体时,请确保它使用所提供的 `AGENTS.md` 文件。使用聊天机器人时,也应一并附上提示词。

我们强烈建议你在完成作业时在 IDE 中禁用 AI 自动补全(例如 Cursor Tab、GitHub Copilot)(不过非 AI 的自动补全,例如自动补全函数名,是完全可以的)。以往的学生指出,禁用 AI 自动补全让他们更容易深入参与材料。

完整的 AI 政策请见本文档。

**代码长什么样**

作业代码以及本说明书可在 GitHub 上获取:

github.com/stanford-cs336/assignment1-basics

请 `git clone` 该仓库。如果有任何更新,我们会通知你,你可以 `git pull` 获取最新版本。

1. `cs336_basics/*`:你编写代码的地方。请注意这里没有任何代码——你可以从零开始做任何你想做的事!
2. `adapters.py`:这里有一组你的代码必须具备的功能。对于每一项功能(例如缩放点积注意力),你通过简单地调用自己的代码来实现它(例如 `run_scaled_dot_product_attention`)。注意:你对 `adapters.py` 的修改不应包含任何实质性逻辑;这是胶水代码。
3. `test_*.py`:这里包含你必须通过的所有测试(例如 `test_scaled_dot_product_attention`),这些测试会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

**如何提交**

要提交,请运行 `make_submission.sh` 来构建提交用的 zip 文件。如果你有不想包含在提交 zip 中的大型数据文件或检查点,请确保将它们添加到脚本的排除列表中。

你需要向 Gradescope 提交以下文件:

- `writeup.pdf`:回答所有书面问题。请排版你的答案。
- `code.zip`:包含你编写的所有代码。

要提交到排行榜,请向以下仓库提交 PR:

github.com/stanford-cs336/assignment1-basics-leaderboard

详细的提交说明见排行榜仓库中的 `README.md`。

**从哪里获取数据集**

本作业将使用两个预处理好的数据集:TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。这两个数据集都是单个大型纯文本文件。

如果你和课程一起做作业,可以在计算指南中找到下载数据集的说明。

如果你是在家自学,可以使用 `README.md` 中的命令下载这些文件。

> **低资源提示:初始化**
>
> 在课程所有作业说明书中,我们都会提供一些建议,帮助你在 GPU 资源较少或没有 GPU 的情况下完成作业的某些部分。例如,我们有时会建议缩小数据集或模型规模,或者说明如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色方框(就像这个)中找到这些"低资源提示"。即使你是已注册的斯坦福学生并能使用课程机器,这些提示也可能帮助你更快迭代、节省时间,因此我们建议阅读它们!

> **低资源提示:在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用官方解答代码,我们可以在配备 36 GB 内存的 Apple M4 Max 芯片上,在 Metal GPU(MPS)上不到 5 分钟、使用 CPU 约 30 分钟内,训练出一个能生成相当流畅文本的语言模型。如果这些词对你来说没什么意义,别担心!你只需知道,如果你有一台较新的笔记本电脑,并且你的实现正确且高效,你就能训练出一个小型 LM,生成简单的儿童故事,流畅度尚可。
>
> 在作业后面,我们会解释如果你使用 CPU 或 MPS 需要做哪些更改。

## 2 字节对编码(BPE)分词器

在作业的第一部分,我们将训练并实现一个字节级字节对编码(BPE)分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。特别地,我们将把任意(Unicode)字符串表示为字节序列,并在这个字节序列上训练 BPE 分词器。之后,我们将使用该分词器把文本(字符串)编码为 token(整数序列),用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准,它将字符映射到整数码点(code point)。截至 Unicode 17.0(2025 年 9 月发布),该标准在 172 种文字中定义了 159,801 个字符。例如,字符 "s" 的码点是 115(通常记作 U+0073,其中 U+ 是约定前缀,0073 是 115 的十六进制表示),字符 "牛" 的码点是 29275。在 Python 中,你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数则将整数 Unicode 码点转换为带有对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

#### 问题(unicode1):理解 Unicode(1 分)

(a) `chr(0)` 返回什么 Unicode 字符?

**交付物:** 一句话回答。

(b) 该字符的字符串表示(`__repr__()`)与它的打印表示有何不同?

**交付物:** 一句话回答。

(c) 当该字符出现在文本中时会发生什么?在 Python 解释器中尝试以下内容,看看是否符合你的预期,可能会有帮助:

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**交付物:** 一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了字符到码点(整数)的映射,但直接在 Unicode 码点上训练分词器并不实际,因为词表会大得令人望而却步(约 15 万个条目)且稀疏(许多字符非常罕见)。取而代之,我们使用 Unicode 编码,它将一个 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码:UTF-8、UTF-16 和 UTF-32,其中 UTF-8 是互联网上的主导编码(超过 98% 的网页)。

要将 Unicode 字符串编码为 UTF-8,我们可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值,我们可以迭代它(例如调用 `list()`)。最后,我们可以使用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列(例如通过 UTF-8 编码),我们本质上是在把一个码点序列(21 位整数,有 159,801 个有效值)转换为一个字节值序列(0 到 255 范围内的整数)。长度为 256 的字节词表处理起来要容易得多。使用字节级分词时,我们不需要担心词表外(OOV)的 token,因为我们知道任何输入文本都可以表示为 0 到 255 的整数序列。

#### 问题(unicode2):Unicode 编码(3 分)

(a) 与 UTF-16 或 UTF-32 相比,偏好基于 UTF-8 编码的字节来训练分词器有哪些原因?比较这些编码在各种输入字符串上的输出可能会有帮助。

**交付物:** 一到两句话回答。

(b) 考虑以下(错误的)函数,其意图是将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数是错误的?请给出一个会产生错误结果的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**交付物:** 一个会使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例,并用一句话解释该函数为什么错误。

(c) 给出一个不能解码为任何 Unicode 字符的双字节序列。

**交付物:** 一个示例,并附一句话解释。

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的词表外问题,但将文本分词为字节会导致输入序列极长。这会拖慢模型训练,因为一个含 10 个词的句子在词级语言模型中可能只有 10 个 token 长,但在字符级模型中可能长达 50 个或更多 token(取决于单词长度)。处理这些更长的序列需要模型在每一步进行更多计算。此外,基于字节序列的语言建模也很困难,因为更长的输入序列会在数据中产生长期依赖。

子词分词是词级分词器与字节级分词器之间的折中。注意,字节级分词器的词表有 256 个条目(字节值为 0 到 255)。子词分词器以更大的词表为代价,换取对输入字节序列更好的压缩。例如,如果字节序列 `b'the'` 经常出现在原始文本训练数据中,为它分配一个词表条目就能把这个 3-token 序列缩减为单个 token。

我们如何选择这些加入词表的子词单元?R. Sennrich et al. [3] 提出使用字节对编码(BPE;P. Gage [5]),这是一种压缩算法,它迭代地将最频繁的字节对替换("合并")为一个新的、未使用的索引。注意,该算法向词表添加子词 token,是为了最大化输入序列的压缩——如果一个词在输入文本中出现得足够多,它就会被表示为单个子词单元。

通过 BPE 构建词表的子词分词器通常称为 BPE 分词器。在本作业中,我们将实现一个字节级 BPE 分词器,其中词表项是字节或合并后的字节序列,这在 OOV 处理和可控的输入序列长度两方面都给了我们最好的结果。构建 BPE 分词器词表的过程被称为"训练" BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包含三个主要步骤。

**词表初始化**

分词器词表是字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器,初始词表就是所有字节的集合。由于有 256 个可能的字节值,初始词表大小为 256。

**预分词**

一旦有了词表,原则上你可以统计文本中相邻字符合出现的频率,并从最频繁的字节对开始合并。然而,这计算代价很高,因为每次合并我们都要对整个语料库做一次完整遍历。此外,直接在语料库上合并字节可能产生仅标点不同的 token(例如 `dog!` 与 `dog.`)。这些 token 会得到完全不同的 token ID,尽管它们很可能具有很高的语义相似度(因为它们仅标点不同)。

为避免这一点,我们对语料库进行预分词。你可以把它理解为对语料库做一种粗粒度的分词,帮助我们统计字符对出现的频率。例如,单词 `'text'` 可能是一个出现 10 次的预分词单元。在这种情况下,当我们统计字符 't' 和 'e' 相邻出现的次数时,我们会看到单词 'text' 中 't' 和 'e' 相邻,于是把它们的计数增加 10,而不必遍历语料库。由于我们训练的是字节级 BPE 模型,每个预分词单元都表示为 UTF-8 字节序列。

R. Sennrich et al. [3] 的原始 BPE 实现通过简单地按空白切分(即 `s.split(" ")`)来进行预分词。这种方法至今仍见于基于 SentencePiece 的分词器(例如 Llama 1 和 Llama 2 的分词器)。

大多数现代分词器使用基于正则表达式的预分词器,这一做法来自 GPT-2;A. Radford et al. [6]。我们将使用原始正则的一个稍加修饰的版本,取自 github.com/openai/tiktoken/pull/234/files:

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

使用这个预分词器交互式地切分一些文本,以更好地理解其行为,可能会有帮助:

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过,在你的代码中使用它时,应使用 `re.finditer`,以避免在构建预分词到计数的映射时把所有预分词后的单词都存起来。

**计算 BPE 合并**

现在我们已经把输入文本转换为预分词单元,并把每个预分词单元表示为 UTF-8 字节序列,因此可以计算 BPE 合并(即训练 BPE 分词器)。概括地说,BPE 算法迭代地统计每一对字节,并找出频率最高的一对("A"、"B")。然后,每一次出现的最频繁对("A"、"B")都被合并,即替换为新 token "AB"。这个新的合并 token 被加入词表;因此,BPE 训练后的最终词表大小等于初始词表大小(我们的情况是 256)加上训练期间执行的 BPE 合并操作数。为提高 BPE 训练效率,我们不统计跨预分词边界的字节对。[^2] 在计算合并时,通过偏好字典序更大的字节对来确定性地打破频率并列。例如,如果字节对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都具有最高频率,我们会合并 ("BA", "A"):

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

[^2]: 注意,原始 BPE 公式 R. Sennrich et al. [3] 规定加入一个词尾 token。我们在训练字节级 BPE 模型时不添加词尾 token,因为所有字节(包括空白和标点)都包含在模型词表中。由于我们显式表示空格和标点,学到的 BPE 合并会自然地反映这些词边界。

**特殊 token**

通常,某些字符串(例如 `<|endoftext|>`)用于编码元数据(例如文档之间的边界)。编码文本时,通常希望把某些字符串视为"特殊 token",它们永远不应被拆分为多个 token(即始终保留为单个 token)。例如,序列结束字符串 `<|endoftext|>` 应始终保留为单个 token(即单个整数 ID),这样我们就知道何时停止从语言模型生成。这些特殊 token 必须加入词表,因此它们有对应的固定 token ID。
R. Sennrich et al. [3] 的算法 1 包含一个效率较低的 BPE 分词器训练实现(基本上遵循我们上面概述的步骤)。作为第一个练习,实现并测试这个函数以检验你的理解可能会很有用。

#### 示例(bpe_example):BPE 训练示例

以下是 R. Sennrich et al. [3] 中一个风格化的示例。考虑一个由以下文本组成的语料库:

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且词表中有一个特殊 token `<|endoftext|>`。

**词表**

我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词表。

**预分词**

为简单起见并聚焦于合并过程,我们在此示例中假设预分词只是按空白切分。预分词并计数后,我们得到如下频率表:

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

用 `dict[tuple[bytes, ...], int]` 表示它很方便,例如 `{(l,o,w): 5, …}`。注意,即使单个字节在 Python 中也是 `bytes` 对象。Python 中没有表示单个字节的 `byte` 类型,正如没有表示单个字符的 `char` 类型一样。

**合并**

我们首先查看每一对相邻字节,并对其出现的单词的频率求和:`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。字节对 `('e', 's')` 和 `('s', 't')` 并列,因此我们取字典序更大的对 `('s', 't')`。然后我们合并预分词单元,得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

在第二轮中,我们看到 `(e, st)` 是最常见的对(计数为 9),我们会将其合并为 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续下去,最终得到的合并序列将是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们取 6 次合并,我们得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`,而我们的词表元素将是 `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。

使用这个词表和这组合并,单词 `newest` 将被分词为 `[ne, west]`。

### 2.5 试验 BPE 分词器训练

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明见第 1 节。在开始之前,我们建议先看一下 TinyStories 数据集,了解数据里有什么。

**并行化预分词**

你会发现一个主要瓶颈是预分词步骤。你可以使用内置库 `multiprocessing` 并行化代码,以加速预分词。具体而言,在预分词的并行实现中,我们建议你切分语料库,同时确保切分边界位于特殊 token 的开头。你可以原样使用以下链接中的起始代码来获得切分边界,然后在这些进程之间分配工作:

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

这种切分始终有效,因为我们从不希望跨文档边界合并。就本作业而言,你始终可以这样切分。不必担心收到一个不包含 `<|endoftext|>` 的超大语料库这一边缘情况。

**在预分词前移除特殊 token**

在使用正则模式(通过 `re.finditer`)运行预分词之前,你应该从语料库(或者,如果使用并行实现,则从你的分块)中剥离所有特殊 token。确保按特殊 token 进行切分,这样就不会跨它们所界定的文本进行合并。例如,如果你有一个形如 `[Doc 1]<|endoftext|>[Doc 2]` 的语料库(或分块),你应该按特殊 token `<|endoftext|>` 切分,并分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词,这样就不会跨文档边界合并。换句话说,特殊 token 在训练期间定义了硬性的分段边界,但它们本身不应参与合并统计。这可以通过 `re.split` 实现,以 `"|".join(special_tokens)` 作为分隔符(需小心使用 `re.escape`,因为 `|` 可能出现在特殊 token 中)。测试 `test_train_bpe_special_tokens` 会检验这一点。

**优化合并步骤**

上面风格化示例中的朴素 BPE 训练实现很慢,因为对于每次合并,它都要遍历所有字节对来找出最频繁的对。然而,每次合并后唯一变化的对计数是那些与合并对重叠的对。因此,BPE 训练速度可以通过索引所有对的计数并增量更新这些计数来提高,而不是显式遍历每一对字节来统计对频率。使用这种缓存过程可以获得显著加速,不过我们指出,BPE 训练的合并部分在 Python 中是不可并行化的。

> **低资源提示:性能分析(Profiling)**
>
> 你应该使用 `cProfile` 或 `py-spy` 等性能分析工具来找出实现中的瓶颈,并集中优化这些地方。

> **低资源提示:"降scale"**
>
> 与其直接在整个 TinyStories 数据集上训练分词器,我们建议你先在一小部分数据上训练:一个"调试数据集"。例如,你可以改为在 TinyStories 验证集上训练分词器,它包含 22K 篇文档,而不是 2.12M 篇。这体现了一种通用策略:尽可能降规模以加速开发,例如使用更小的数据集、更小的模型等。选择调试数据集或超参数配置的规模需要仔细考量:你希望调试集足够大,以便具有与完整配置相同的瓶颈(这样你做的优化才能推广),但又不能大到运行起来没完没了。

#### 问题(train_bpe):BPE 分词器训练(15 分)

**交付物:** 编写一个函数,给定输入文本文件的路径,训练一个(字节级)BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数:

**输入**

- `input_path: str` — 包含 BPE 分词器训练数据的文本文件路径。
- `vocab_size: int` — 一个正整数,定义最终词表大小的最大值(包括初始字节词表、合并产生的词表项以及任何特殊 token)。
- `special_tokens: list[str]` — 要加入词表的字符串列表。在训练期间,将它们视为硬边界,阻止跨其跨度进行合并,但在计算合并统计时不要包含它们。

你的 BPE 训练函数应返回得到的词表和合并列表:

**输出**

- `vocab: dict[int, bytes]` — 分词器词表,从 int(词表中的 token ID)到 bytes(token 字节)的映射。
- `merges: list[tuple[bytes, bytes]]` — 训练产生的 BPE 合并列表。每个列表项是一个字节元组 `(<token1>, <token2>)`,表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排序。

要针对我们提供的测试来测试你的 BPE 训练函数,你首先需要实现位于 `[adapters.run_train_bpe]` 的测试适配器。然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。可选地(这可能需要大量时间投入),你可以使用某种系统级语言实现训练方法的关键部分,例如 C++(考虑 cppyy 或 nanobind)或 Rust(使用 PyO3)。如果这样做,请注意哪些操作需要从 Python 内存复制、哪些可以直接读取,并确保留下构建说明,或确保仅使用 `pyproject.toml` 即可构建。还要注意,GPT-2 正则并未在大多数正则引擎中得到良好支持,而且在大多数支持它的引擎中会太慢。我们已验证 Oniguruma 速度尚可并支持负向前瞻,但 Python 中的 `regex` 包如果有区别的话,甚至更快。

#### 问题(train_bpe_tinystories):在 TinyStories 上训练 BPE(2 分)

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器,最大词表大小为 10,000。确保将 TinyStories 的 `<|endoftext|>` 特殊 token 加入词表。将得到的词表和合并序列化到磁盘,以便进一步检查。训练花了多少时间和内存?词表中最长的 token 是什么?它合理吗?

资源需求:≤ 30 分钟(无 GPU),≤ 30 GB 内存

> **提示**
>
> 如果在预分词期间使用多进程,并利用以下两个事实,你应该能在 2 分钟内完成 BPE 训练:
>
> (a) `<|endoftext|>` token 在数据文件中分隔文档。
> (b) 在应用 BPE 合并之前,`<|endoftext|>` token 会作为特殊情况处理。

**交付物:** 一到两句话回答。

(b) 对你的代码进行性能分析。分词器训练过程中哪一部分耗时最多?

**交付物:** 一到两句话回答。

接下来,我们将尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。和之前一样,我们建议先看一下数据集,以更好地理解其内容。

#### 问题(train_bpe_expts_owt):在 OpenWebText 上训练 BPE(2 分)

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器,最大词表大小为 32,000。将得到的词表和合并序列化到磁盘,以便进一步检查。词表中最长的 token 是什么?它合理吗?

资源需求:≤ 12 小时(无 GPU),≤ 100 GB 内存

**交付物:** 一到两句话回答。

(b) 比较并对比你在 TinyStories 与 OpenWebText 上训练得到的分词器。

**交付物:** 一到两句话回答。

### 2.6 BPE 分词器:编码与解码

在作业的前一部分,我们实现了一个函数,用于在输入文本上训练 BPE 分词器,以获得分词器词表和 BPE 合并列表。现在,我们将实现一个 BPE 分词器,它加载给定的词表和合并列表,并使用它们将文本编码为 token ID、将 token ID 解码为文本。

#### 2.6.1 编码文本

用 BPE 编码文本的过程与训练 BPE 词表的过程类似。有几个主要步骤。

**步骤 1:预分词。** 我们首先对序列进行预分词,并将每个预分词单元表示为 UTF-8 字节序列,就像在 BPE 训练中所做的那样。我们会在每个预分词单元内将这些字节合并为词表元素,独立处理每个预分词单元(不跨预分词边界合并)。

**步骤 2:应用合并。** 然后,我们取出 BPE 训练期间创建的词表元素合并序列,按创建顺序将其应用于我们的预分词单元。

#### 示例(bpe_encoding):BPE 编码示例

例如,假设我们的输入字符串是 `'the cat ate'`,词表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`,学到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先,我们的预分词器会将这个字符串切分为 `['the', ' cat', ' ate']`。然后,我们查看每个预分词单元并应用 BPE 合并。

第一个预分词单元 `'the'` 初始表示为 `[b't', b'h', b'e']`。查看合并列表,我们找到第一个适用的合并 `(b't', b'h')`,用它把预分词单元变为 `[b'th', b'e']`。然后,我们回到合并列表,找到下一个适用的合并 `(b'th', b'e')`,把预分词单元变为 `[b'the']`。最后,再看合并列表,我们看到没有更多适用于该字符串的合并了(因为整个预分词单元已经合并为单个 token),因此应用 BPE 合并的过程结束。对应的整数序列是 `[9]`。

对剩余的预分词单元重复这一过程,我们看到预分词单元 `' cat'` 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`,对应的整数序列为 `[7, 1, 5]`。最后一个预分词单元 `' ate'` 在应用 BPE 合并后为 `[b' at', b'e']`,对应的整数序列为 `[10, 3]`。因此,编码输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊 token**

在编码文本时,你的分词器应能正确处理用户定义的特殊 token(在构造分词器时提供)。

**内存考虑**

假设我们要对一个无法放入内存的大型文本文件进行分词。为了高效地分词这个大文件(或任何其他数据流),我们需要把它分成可管理的块,依次处理每个块,从而使内存复杂度为常数,而不是随文本大小线性增长。这样做时,我们需要确保一个 token 不会跨越块边界,否则得到的 tokenization 会与在内存中对整个序列进行朴素分词的结果不同。

#### 2.6.2 解码文本

要将整数 token ID 序列解码回原始文本,我们可以简单地查找每个 ID 在词表中对应的条目(一个字节序列),将它们拼接在一起,然后将字节解码为 Unicode 字符串。注意,输入的 ID 并不保证映射到有效的 Unicode 字符串(因为用户可以输入任意整数 ID 序列)。如果输入的 token ID 不能产生有效的 Unicode 字符串,你应该用官方 Unicode 替换字符 U+FFFD 替换格式错误的字节。[^3] `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误,使用 `errors='replace'` 会自动用替换标记替换格式错误的数据。

[^3]: 关于 Unicode 替换字符的更多信息,见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

#### 问题(tokenizer):实现分词器(15 分)

**交付物:** 实现一个 `Tokenizer` 类,给定词表和合并列表,将文本编码为整数 ID,并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token(如果它们还不在词表中,则追加到词表)。我们推荐以下接口:

```python
def __init__(self, vocab, merges, special_tokens=None)
```

从给定的词表、合并列表和(可选的)特殊 token 列表构造分词器。该函数应接受以下参数:

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

类方法,从序列化的词表和合并列表(格式与你的 BPE 训练代码输出相同)及(可选的)特殊 token 列表构造并返回一个 `Tokenizer`。该方法应接受以下额外参数:

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

将输入文本编码为 token ID 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定一个字符串可迭代对象(例如 Python 文件句柄),返回一个惰性产出 token ID 的生成器。这对于无法直接加载到内存的大型文件的内存高效分词是必需的。

```python
def decode(self, ids: list[int]) -> str
```

将 token ID 序列解码为文本。

要针对我们提供的测试来测试你的 `Tokenizer`,你首先需要实现位于 `[adapters.get_tokenizer]` 的测试适配器。然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

### 2.7 实验

#### 问题(tokenizer_experiments):分词器实验(4 分)

(a) 从 TinyStories 和 OpenWebText 中各采样 10 篇文档。使用你之前训练的 TinyStories 和 OpenWebText 分词器(词表大小分别为 10K 和 32K),将这些采样文档编码为整数 ID。每个分词器的压缩比(字节/token)是多少?

**交付物:** 一到两句话回答。

(b) 如果你用 TinyStories 分词器对 OpenWebText 样本进行分词会发生什么?比较压缩比和/或定性地描述会发生什么。

**交付物:** 一到两句话回答。

(c) 估计你的分词器的吞吐量(例如,以字节/秒为单位)。对 Pile 数据集(825GB 文本)进行分词需要多长时间?

**交付物:** 一到两句话回答。

(d) 使用你的 TinyStories 和 OpenWebText 分词器,将各自的训练集和开发集编码为整数 token ID 序列。我们稍后将用这些数据来训练语言模型。我们建议将 token ID 序列化为 `uint16` 数据类型的 NumPy 数组。为什么 `uint16` 是合适的选择?

**交付物:** 一到两句话回答。
## 3 Transformer 语言模型架构

语言模型接收一个批量的整数 token ID 序列(即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`)作为输入,并返回一个(批量的)词表上的归一化概率分布(即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor),其中预测的分布是每个输入 token 之后的下一个词上的分布。训练语言模型时,我们使用这些下一个词的预测来计算实际下一个词与预测下一个词之间的交叉熵损失。推理时从语言模型生成文本,我们取最后一个时间步(即序列中的最后一项)预测的下一个词分布来生成序列中的下一个 token(例如取概率最高的 token、从分布中采样等),将生成的 token 加入输入序列,然后重复。

在作业的这一部分,你将从头构建这个 Transformer 语言模型。我们将先对模型进行高层描述,然后逐步详述各个组件。

### 3.1 Transformer LM

给定 token ID 序列,Transformer 语言模型使用输入嵌入将 token ID 转换为稠密向量,将嵌入后的 token 通过 `num_layers` 个 Transformer 块,然后应用一个学习到的线性投影("输出嵌入"或"LM head")来产生预测的下一个 token 的 logits。示意图见图 1。

> **图 1:**我们的 Transformer 语言模型概览。
>
> 输入 → Token 嵌入 → `num_layers` 个 Transformer 块 → Norm → Linear(输出嵌入)→ Softmax → 输出概率。

> **图 2:**一个 pre-norm Transformer 块。
>
> 输入张量形状为 `(batch_size, seq_len, d_model)`;依次经过 Norm → 带 RoPE 的因果多头自注意力 → Add 残差;再经过 Norm → 逐位置前馈网络 → Add 残差;输出张量形状为 `(batch_size, seq_len, d_model)`。

**Token 嵌入**

在第一步中,Transformer 将(批量的)token ID 序列嵌入为包含 token 身份信息的向量序列(图 1 中的红色块)。

更具体地说,给定 token ID 序列,Transformer 语言模型使用 token 嵌入层产生一个向量序列。每个嵌入层接收形状为 `(batch_size, sequence_length)` 的整数张量,并产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

**Pre-norm Transformer 块**

嵌入之后,激活值会经过若干结构相同的神经网络层。标准的仅解码器(decoder-only)Transformer 语言模型由 `num_layers` 个相同的层(通常称为 Transformer "块")组成。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入,并返回相同形状的输出。每个块通过自注意力聚合序列中的信息,并通过前馈层对其进行非线性变换。

经过 `num_layers` 个 Transformer 块后,我们将取最终的激活值,并将其转换为词表上的分布。

我们将实现"pre-norm" Transformer 块(详见第 3.4 节),它额外要求在最后一个 Transformer 块之后使用层归一化(详见下文),以确保其输出被适当缩放。

经过此归一化后,我们将使用标准的、学习到的线性变换,将 Transformer 块的输出转换为预测的下一个 token 的 logits(参见,例如 A. Radford et al. [7] 的公式 2)。

### 3.2 备注:批处理、Einsum 与高效计算

在整个 Transformer 中,我们会将相同的计算应用于许多类似批次的输入。这里有几个例子:

- **批次中的元素:** 我们对每个批次元素应用相同的 Transformer 前向操作。
- **序列长度:** 诸如 RMSNorm 和前馈之类的"逐位置"操作对序列的每个位置执行相同操作。
- **注意力头:** 注意力操作在"多头"注意力操作中跨注意力头进行批处理。

有一种符合直觉的方式来执行这类操作,以充分利用 GPU,并且易于阅读和理解,这会很有用。许多 PyTorch 操作可以接受张量开头的额外"类批次"维度,并高效地跨这些维度重复/广播该操作。

例如,假设我们在做逐位置的批量操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" D,我们想对形状为 `(d_model, d_model)` 的矩阵 A 做批量向量-矩阵乘法。在这种情况下,`D @ A` 会做批量矩阵乘法,这是 PyTorch 中一种高效的基元操作,其中 `(batch_size, sequence_length)` 维度被批处理。

因此,假设你的函数可能会被传入额外的类批次维度,并将这些维度保持在 PyTorch 形状的开头,会很有帮助。为了以这种方式组织张量以便批处理,它们可能需要通过多步 `view`、`reshape` 和 `transpose` 来调整形状。这可能有点麻烦,而且往往很难看出代码在做什么、张量的形状是什么。

更符合直觉的选择是在 `torch.einsum` 中使用 einsum 记法,或者使用与框架无关的库,如 `einops` 或 `einx`。两个关键操作是 `einsum`,它可以对输入张量的任意维度做张量收缩,以及 `rearrange`,它可以重排、拼接和拆分任意维度。事实证明,机器学习中几乎所有操作都是维度变换与张量收缩的某种组合,偶尔还会加上(通常是逐点的)非线性函数。这意味着使用 einsum 记法时,你的许多代码可以更具可读性和灵活性。

我们强烈建议在课程中学习并使用 einsum 记法。之前没有接触过 einsum 记法的学生应使用 `einops`(文档见此处),已经熟悉 `einops` 的学生应学习更通用的 `einx`(见此处)。[^4] 这两个包都已安装在我们提供的环境中。

下面给出一些使用 einsum 记法的示例。这些是对 `einops` 文档的补充,你应该先阅读该文档。

[^4]: 值得注意的是,虽然 `einops` 得到了大量支持,但 `einx` 尚未经过同等程度的实战检验。如果你发现 `einx` 有任何限制或 bug,可以放心退回到使用 `einops` 加一些更普通的 PyTorch。

#### 示例(einstein_example1):使用 einops.einsum 的批量矩阵乘法

```python
import torch
from einops import rearrange, einsum

## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?

## Einsum is self-documenting and robust
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

#### 示例(einstein_example2):使用 einops.rearrange 的广播操作

我们有一批图像,对每张图像,我们想根据某个缩放因子生成 10 个变暗版本:

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

#### 示例(einstein_example3):使用 einops.rearrange 进行像素混合

假设我们有一批图像,表示为形状为 `(batch, height, width, channel)` 的张量,我们想对图像的所有像素执行线性变换,但该变换应对每个通道独立进行。我们的线性变换表示为形状为 `(height * width, height * width)` 的矩阵 B。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改为使用 einops:

```python
height = width = 32

## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者,如果你想玩点花的:用 `einx.dot`(`einops.einsum` 的 einx 等价物)一次搞定:

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里的第一个实现可以通过在前后添加注释来标明输入和输出形状来改进,但这很笨拙且容易出错。使用 einsum 记法时,文档就是实现!

Einsum 记法可以处理任意的输入批次维度,还有一个关键优势:它是自文档化的。在使用 einsum 记法的代码中,输入和输出张量的相关形状要清晰得多。对于其余张量,你可以考虑使用 Tensor 类型提示,例如使用 `jaxtyping` 库(并非 JAX 专用)。

我们将在作业 2 中进一步讨论使用 einsum 记法的性能影响,但就目前而言,你只需知道它们几乎总是比替代方案更好!

#### 3.2.1 数学记法与内存顺序

许多机器学习论文在记法中使用行向量,这产生的表示与 NumPy 和 PyTorch 默认使用的行主序(row-major)内存顺序很好地契合。使用行向量时,线性变换形如

$$
y = xW^\top, \tag{1}
$$

其中 W ∈ ℝ^(d_out×d_in) 为行主序,行向量 x ∈ ℝ^(1×d_in)。注意,这让我们可以通过增加 x 的最外层维度来批量输入,这意味着我们可以把向量输入 x 替换为矩阵输入 X ∈ ℝ^(batch×d_in)。

在线性代数中,更常见的是使用列向量,此时线性变换形如

$$
y = Wx, \tag{2}
$$

给定行主序的 W ∈ ℝ^(d_out×d_in) 和列向量 x ∈ ℝ^(d_in)。在这种设定下要对输入进行批处理,批次维度必须放在最后,因此 x 需要被替换为矩阵 X̃ ∈ ℝ^(d_in×batch)。

在本作业的数学记法中,我们将主要使用列向量,因为数学通常遵循这种记法。你应记住,如果你想使用普通的矩阵乘法记法,就必须像公式 1 中的行向量约定那样转置后应用矩阵,因为 PyTorch 使用行主序内存顺序。如果你使用 einsum 进行线性代数运算,只要正确标注坐标轴,这应该不成问题。顺便提一句,值得注意的是,其他语言/线性代数包(如 Matlab、Julia 和 Fortran)都使用列主序内存顺序,意味着批次维度放在最后,但 Python 及相关包采用了 C 标准的行主序。

### 3.3 基本构建模块:线性层与嵌入层

#### 3.3.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化会导致不良行为,例如梯度消失或爆炸。Pre-norm Transformer 对初始化异常鲁棒,但初始化仍会对训练速度和收敛产生显著影响。由于本作业已经很长,我们将细节留到作业 3,转而给你一些在大多数情况下都表现良好的近似初始化。目前,请使用:

- **线性层权重:** 𝒩(μ = 0, σ² = 2/(d_in + d_out)),截断在 [−3σ, 3σ]。
- **嵌入:** 𝒩(μ = 0, σ² = 1),截断在 [−3, 3]。
- **RMSNorm:** 𝟙

你应使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 线性模块

线性层是 Transformer 乃至一般神经网络的基本构建模块。首先,你将实现自己的 `Linear` 类,它继承自 `torch.nn.Module`,并执行线性变换:

$$
y = Wx. \tag{3}
$$

注意,遵循大多数现代 LLM 的做法,我们不包含偏置项。

#### 问题(linear):实现线性模块(1 分)

**交付物:** 实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口,只是没有 bias 参数或参数项。我们推荐以下接口:

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造一个线性变换模块。该函数应接受以下参数:

- `in_features: int` — 输入的最终维度
- `out_features: int` — 输出的最终维度
- `device: torch.device | None = None` — 存储参数的设备
- `dtype: torch.dtype | None = None` — 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用线性变换。

请确保:

- 继承 `nn.Module`
- 调用父类构造函数
- 构造并存储参数为 W(而不是 Wᵀ),将其放入 `nn.Parameter`
- 当然,不要使用 `nn.Linear` 或 `nn.functional.linear`

关于初始化,使用上面的设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的 `Linear` 模块,请实现位于 `[adapters.run_linear]` 的测试适配器。适配器应将给定的权重加载到你的 `Linear` 模块中。为此,你可以使用 `Module.load_state_dict`。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 嵌入模块

如上所述,Transformer 的第一层是一个嵌入层,它将整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个继承自 `torch.nn.Module` 的自定义 `Embedding` 类(因此你不应使用 `nn.Embedding`)。`forward` 方法应使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` 类型的 token ID,通过索引形状为 `(vocab_size, d_model)` 的嵌入矩阵,为每个 token ID 选择嵌入向量。

#### 问题(embedding):实现嵌入模块(1 分)

**交付物:** 实现继承自 `torch.nn.Module` 并执行嵌入查找的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口:

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造一个嵌入模块。该函数应接受以下参数:

- `num_embeddings: int` — 词表大小
- `embedding_dim: int` — 嵌入向量的维度,即 d_model
- `device: torch.device | None = None` — 存储参数的设备
- `dtype: torch.dtype | None = None` — 参数的数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token ID 的嵌入向量。

请确保:

- 继承 `nn.Module`
- 调用父类构造函数
- 将嵌入矩阵初始化为 `nn.Parameter`
- 存储嵌入矩阵时,使 `d_model` 为最后一维
- 当然,不要使用 `nn.Embedding` 或 `nn.functional.embedding`

同样,使用上面的设置进行初始化,并使用 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的实现,请实现位于 `[adapters.run_embedding]` 的测试适配器。然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层:多头自注意力机制和逐位置前馈网络([A. Vaswani et al., 2017],第 3.1 节)。

在原始 Transformer 论文中,模型在两个子层周围各使用一个残差连接,然后进行层归一化。这种架构通常被称为"post-norm" Transformer,因为层归一化应用于子层输出。然而,多项工作发现,将层归一化从每个子层的输出移到每个子层的输入(并在最后一个 Transformer 块之后额外加一个层归一化)可以提高 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——这种"pre-norm" Transformer 块的直观表示见图 2。然后,每个 Transformer 块子层的输出通过残差连接加到子层输入上(A. Vaswani et al. [8],第 5.4 节)。对 pre-norm 的一种直觉是:从输入嵌入到 Transformer 最终输出存在一条干净的"残差流",其中没有任何归一化,据称这可以改善梯度流动。这种 pre-norm Transformer 现在是当今语言模型(GPT-3、LLaMA、PaLM 等)使用的标准,因此我们将实现这一变体。我们将逐一介绍 pre-norm Transformer 块的每个组件,并依次实现它们。

#### 3.4.1 均方根层归一化

A. Vaswani et al. [8] 的原始 Transformer 实现使用层归一化 [J. L. Ba et al., 2016] 来归一化激活值。遵循 H. Touvron et al. [12],我们将使用均方根层归一化(RMSNorm;B. Zhang et al. [13],公式 4)作为层归一化。给定一个由激活值组成的向量 a ∈ ℝ^(d_model),RMSNorm 将按如下方式重新缩放每个激活值 a_i:

$$
\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} g_i, \tag{4}
$$

其中

$$
\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{\mathrm{model}}} \sum_{i=1}^{d_{\mathrm{model}}} a_i^2 + \varepsilon}.
$$

这里,g_i 是可学习的"增益"参数(总共有 `d_model` 个这样的参数),ε 是一个通常固定为 1e-5 的超参数。

你应该将输入向上转型为 `torch.float32`,以防止对输入求平方时发生溢出。总体上,你的 `forward` 方法应如下所示:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
```

#### 问题(rmsnorm):均方根层归一化(1 分)

**交付物:** 将 `RMSNorm` 实现为一个 `torch.nn.Module`。我们推荐以下接口:

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 `RMSNorm` 模块。该函数应接受以下参数:

- `d_model: int` — 模型的隐藏维度
- `eps: float = 1e-5` — 用于数值稳定性的 epsilon 值
- `device: torch.device | None = None` — 存储参数的设备
- `dtype: torch.dtype | None = None` — 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量,并返回相同形状的张量。

注意:如上所述,记得在执行归一化之前将输入向上转型为 `torch.float32`(之后再向下转型回原始 dtype)。

要测试你的实现,请实现位于 `[adapters.run_rmsnorm]` 的测试适配器。然后运行 `uv run pytest -k test_rmsnorm`。
#### 3.4.2 逐位置前馈网络

> **图 3:**比较 SiLU(又称 Swish)与 ReLU 激活函数。图中横轴为 x,纵轴为激活值;SiLU: f(x) = x·σ(x),Identity: f(x) = x,ReLU: f(x) = max(0, x)。

在原始 Transformer 论文(A. Vaswani et al. [8] 第 3.3 节)中,Transformer 前馈网络由两个线性变换组成,二者之间有一个 ReLU 激活(ReLU(x) = max(0, x))。在那种原始架构中,内部前馈层的维度通常是输入维度的 4 倍。

然而,与现代语言模型相比,原始设计通常有两项主要变化:它们使用另一种激活函数,并采用门控机制。具体而言,我们将实现被 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用的"SwiGLU"激活函数,它结合了 SiLU(通常称为 Swish)激活和一种称为门控线性单元(Gated Linear Unit,GLU)的门控机制。遵循自 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来大多数现代 LLM 的做法,我们还将省略线性层中有时使用的偏置项。

SiLU 或称 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下:

$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}. \tag{5}
$$

从图 3 可以看出,SiLU 激活函数与 ReLU 激活函数相似,但在零点处是光滑的。

门控线性单元(GLU)最初由 Y. N. Dauphin et al. [19] 定义为:一个经过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积:

$$
\mathrm{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x, \tag{6}
$$

其中 ⊙ 表示逐元素乘法。门控线性单元被认为"通过为梯度提供线性路径,同时保留非线性能力,从而缓解深度架构中的梯度消失问题"。

将 SiLU/Swish 与 GLU 结合起来,我们得到 SwiGLU,并将其用于我们的前馈网络:

$$
\mathrm{FFN}(x) = \mathrm{SwiGLU}(x, W_1, W_2, W_3) = W_2(\mathrm{SiLU}(W_1 x) \odot W_3 x), \tag{7}
$$

其中 x ∈ ℝ^(d_model),W₁, W₃ ∈ ℝ^(d_ff×d_model),W₂ ∈ ℝ^(d_model×d_ff),并且通常取 d_ff = (8/3)d_model。在具体实现中,为了硬件效率,可以把它四舍五入到附近的 64 的倍数。

N. Shazeer [20] 首先提出将 SiLU/Swish 激活与 GLU 结合,并进行了实验,表明 SwiGLU 在语言建模任务上优于 ReLU 和(不带门控的)SiLU 等基线。在作业后面,你将比较 SwiGLU 和 SiLU。尽管我们已经提到这些组件的一些启发式论据(论文也提供了更多支持证据),但保持经验视角是好的:Shazeer 论文中一句如今很有名的话是:

> "我们不解释这些架构为何似乎有效;我们把它们的成功,和其他一切一样,归因于神的恩典。"

#### 问题(positionwise_feedforward):实现逐位置前馈网络(2 分)

**交付物:** 实现 SwiGLU 前馈网络,由 SiLU 激活函数和 GLU 组成。

注意:在这种特定情况下,你可以放心地在实现中使用 `torch.sigmoid` 以保证数值稳定性。

在你的实现中,应将 d_ff 设置为约 (8/3)×d_model,同时确保内部前馈层的维度是 64 的倍数,以充分利用硬件。要针对我们提供的测试来测试你的实现,你需要实现位于 `[adapters.run_swiglu]` 的测试适配器。然后运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.4.3 相对位置嵌入

为了向模型注入位置信息,我们将实现旋转位置嵌入(Rotary Position Embeddings)[J. Su et al., 2021],通常称为 RoPE。对于位置 i 处的给定查询 token q⁽ⁱ⁾ = W_q x⁽ⁱ⁾ ∈ ℝ^d,我们将应用一个成对旋转矩阵 R_i,得到 q′⁽ⁱ⁾ = R_i q⁽ⁱ⁾ = R_i W_q x⁽ⁱ⁾。这里,R_i 会以角度 θ_{i,k} = i / Θ^((2k−2)/d) 将嵌入元素对 q⁽ⁱ⁾_{2k−1:2k} 作为二维向量旋转,其中 k ∈ {1,…,d/2},Θ 为某个常数。因此,我们可以把 R_i 看作大小为 d×d 的分块对角矩阵,其块为 R_i^k,k ∈ {1,…, d/2},其中

$$
R_i^k =
\begin{pmatrix}
\cos(\theta_{i,k}) & \sin(\theta_{i,k}) \\
-\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{pmatrix}. \tag{8}
$$

于是我们得到完整的旋转矩阵

$$
R_i =
\begin{pmatrix}
R_i^1 & 0 & 0 & \cdots & 0 \\
0 & R_i^2 & 0 & \cdots & 0 \\
0 & 0 & R_i^3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R_i^{d/2}
\end{pmatrix}, \tag{9}
$$

其中 0 表示 2×2 零矩阵。虽然可以构造完整的 d×d 矩阵,但一个好的解决方案应利用该矩阵的性质来更高效地实现变换。由于我们只关心给定序列内 token 的相对旋转,我们可以跨层、跨不同批次重用为 cos(θ_{i,k}) 和 sin(θ_{i,k}) 计算的值。如果你想优化,可以使用一个被所有层引用的单一 RoPE 模块,并让它拥有一个在初始化期间通过 `self.register_buffer(persistent=False)` 创建的 sin 和 cos 值的二维预计算缓冲区,而不是使用 `nn.Parameter`(因为我们不想学习这些固定的余弦和正弦值)。然后,对我们 q⁽ⁱ⁾ 所做的完全相同的旋转过程也会对 k⁽ʲ⁾ 执行,按对应的 R_j 旋转。注意,这一层没有可学习参数。

#### 问题(rope):实现 RoPE(2 分)

**交付物:** 实现一个 `RotaryPositionalEmbedding` 类,对输入张量应用 RoPE。

推荐以下接口:

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE 模块,并在需要时创建缓冲区。

- `theta: float` — RoPE 的 Θ 值
- `d_k: int` — 查询向量和键向量的维度
- `max_seq_len: int` — 将被输入的最大序列长度
- `device: torch.device | None = None` — 存储缓冲区的设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量,并返回相同形状的张量。注意,你应容忍 x 具有任意数量的批次维度。你应假定 token 位置是形状为 `(..., seq_len)` 的张量,指定 x 沿序列维度的 token 位置。

你应使用 token 位置来沿序列维度对(可能已预计算的)cos 和 sin 张量进行切片。

要测试你的实现,请完成 `[adapters.run_rope]`,并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

现在我们将实现 A. Vaswani et al. [8](第 3.2.1 节)中描述的缩放点积注意力。作为预备步骤,Attention 操作的定义会用到 softmax,该操作接收一个未归一化的分数向量,并将其转换为归一化分布:

$$
\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}. \tag{10}
$$

注意,对于很大的值,exp(v_i) 可能变为 inf(此时 inf/inf = NaN)。我们可以通过注意到 softmax 操作对给所有输入加上任意常数 c 不变来避免这一点。我们可以利用这一性质提高数值稳定性——通常,我们会从 v 的所有元素中减去 v 的最大项,使新的最大项为 0。现在你将实现 softmax,并使用这一技巧提高数值稳定性。

#### 问题(softmax):实现 softmax(1 分)

**交付物:** 编写一个在张量上应用 softmax 操作的函数。你的函数应接受两个参数:一个张量和一个维度 i,并对输入张量的第 i 维应用 softmax。输出张量应与输入张量形状相同,但其第 i 维现在是一个归一化概率分布。使用从第 i 维所有元素中减去第 i 维最大值的技巧,以避免数值稳定性问题。

要测试你的实现,请完成 `[adapters.run_softmax]`,并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以用数学方式定义 Attention 操作如下:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, \tag{11}
$$

其中 Q ∈ ℝ^(n×d_k),K ∈ ℝ^(m×d_k),V ∈ ℝ^(m×d_v)。这里,Q、K 和 V 都是该操作的输入——注意它们不是可学习参数。

**掩码(Masking):** 有时对注意力操作的输出进行掩码会很方便。掩码应具有形状 M ∈ {True, False}^(n×m),该布尔矩阵的每一行 i 表示查询 i 应该关注哪些键。按惯例(并且略有歧义),位置 (i,j) 处的值为 True 表示查询 i 确实关注键 j,值为 False 表示该查询不关注该键。换言之,在值为 True 的 (i,j) 对处"信息流动"。例如,考虑一个 1×3 的掩码矩阵,其条目为 `[[True, True, False]]`。唯一的查询向量只关注前两个键。

在计算上,使用掩码比在子序列上计算注意力要高效得多,我们可以通过取 softmax 之前的值(QKᵀ/√d_k),并对掩码矩阵中为 False 的任何条目加上 −∞ 来实现。

#### 问题(scaled_dot_product_attention):实现缩放点积注意力(5 分)

**交付物:** 实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询,以及形状为 `(batch_size, ..., seq_len, d_v)` 的值,其中 `...` 表示任意数量的其他类批次维度(如果提供)。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于类批次维度的讨论见第 3.2 节。

你的实现还应支持一个可选的、用户提供的、形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 True 的位置的注意力概率应合计为 1,掩码值为 False 的位置的注意力概率应为 0。

要针对我们提供的测试来测试你的实现,你需要实现位于 `[adapters.run_scaled_dot_product_attention]` 的测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 会在三阶输入张量上测试你的实现,而 `uv run pytest -k test_4d_scaled_dot_product_attention` 会在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将实现 A. Vaswani et al. [8] 第 3.2.2 节中描述的多头自注意力。回顾一下,从数学上讲,应用多头注意力的操作定义如下:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) \tag{12}
$$

其中

$$
\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i) \tag{13}
$$

Q_i、K_i、V_i 分别是 Q、K、V 嵌入维度中大小为 d_k 或 d_v 的第 i 个切片,i ∈ {1,…,h}。其中 Attention 是第 3.4.4 节定义的缩放点积注意力操作。由此我们可以构造多头自注意力操作:

$$
\mathrm{MultiHeadSelfAttention}(x) = W_O \mathrm{MultiHead}(W_Q x, W_K x, W_V x). \tag{14}
$$

这里,可学习参数是 W_Q ∈ ℝ^(h d_k×d_model)、W_K ∈ ℝ^(h d_k×d_model)、W_V ∈ ℝ^(h d_v×d_model) 和 W_O ∈ ℝ^(d_model×h d_v)。由于 Q、K、V 在多注意力操作中被切片,我们可以把 W_Q、W_K 和 W_V 看作沿输出维度为每个头分离。当这部分能正常工作时,你应总共只用三次矩阵乘法来计算键、值和查询投影。[^5]

[^5]: 作为进阶目标,尝试把键、查询和值投影合并为一个权重矩阵,这样你只需一次矩阵乘法。

**因果掩码**

你的实现应阻止模型关注序列中的未来 token。换言之,如果模型给定 token 序列 t₁,…,tₙ,我们想为前缀 t₁,…,tᵢ(其中 i < n)计算下一个词的预测,那么模型不应能访问(关注)位置 t_{i+1},…,tₙ 处的 token 表示,因为推理时生成文本时它无法访问这些 token(而且这些未来 token 会泄露真实下一个词身份的信息,使语言建模预训练目标变得平凡)。对于输入 token 序列 t₁,…,tₙ,我们可以通过运行 n 次多头自注意力(针对序列中的 n 个唯一前缀)来朴素地阻止访问未来 token。取而代之,我们将使用因果注意力掩码,它允许 token i 关注序列中所有位置 j ≤ i。你可以使用 `torch.triu` 或广播索引比较来构造这个掩码,并且应利用第 3.4.4 节中你的缩放点积注意力实现已支持注意力掩码这一事实。

**应用 RoPE**

RoPE 应应用于查询向量和键向量,但不应用于值向量。此外,头维度应作为批次维度处理,因为在多头注意力中,注意力是独立地应用于每个头的。这意味着对每个头,查询向量和键向量都应应用完全相同的 RoPE 旋转。

#### 问题(multihead_self_attention):实现因果多头自注意力(5 分)

**交付物:** 将因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应至少接受以下参数:

- `d_model: int` — Transformer 块输入的维度。
- `num_heads: int` — 多头自注意力中使用的头数。

遵循 A. Vaswani et al. [8],设置 d_k = d_v = d_model / h。要针对我们提供的测试来测试你的实现,请实现位于 `[adapters.run_multihead_self_attention]` 的测试适配器。然后运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.5 完整的 Transformer LM

让我们从组装 Transformer 块开始(参考图 2 会有帮助)。一个 Transformer 块包含两个"子层",一个用于多头自注意力,另一个用于 SwiGLU 前馈网络。在每个子层中,我们首先执行 RMSNorm,然后执行主操作(MHA/FF),最后加入残差连接。

具体来说,Transformer 块的前半部分(第一个"子层")应实现以下更新,以从输入 x 产生输出 y:

$$
y = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)). \tag{15}
$$

#### 问题(transformer_block):实现 Transformer 块(3 分)

按照第 3.4 节的描述并参考图 2,实现 pre-norm Transformer 块。你的 Transformer 块应至少接受以下参数。

- `d_model: int` — Transformer 块输入的维度。
- `num_heads: int` — 多头自注意力中使用的头数。
- `d_ff: int` — 逐位置前馈内部层的维度。

要测试你的实现,请实现适配器 `[adapters.run_transformer_block]`。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。

**交付物:** 能通过所提供测试的 Transformer 块代码。

现在,按照图 1 中的高层示意图,我们将这些块组合起来。遵循我们在第 3.1.0.1 节中对嵌入的描述,将其输入 `num_layers` 个 Transformer 块,然后将其传入最终的层归一化和 LM head,以获得词表上的未归一化分布(logits)。

#### 问题(transformer_lm):实现 Transformer LM(3 分)

是时候把所有东西组合起来了!按照第 3.1 节的描述并参考图 1,实现 Transformer 语言模型。你的实现至少应接受上述 Transformer 块的所有构造参数,以及以下额外参数:

- `vocab_size: int` — 词表大小,用于确定 token 嵌入矩阵的维度。
- `context_length: int` — 最大上下文长度,用于确定 RoPE sin 和 cos 缓冲区的维度。
- `num_layers: int` — 要使用的 Transformer 块数量。

要针对我们提供的测试来测试你的实现,你首先需要实现位于 `[adapters.run_transformer_lm]` 的测试适配器。然后运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。

**交付物:** 能通过上述测试的 Transformer LM 模块。

**资源核算**

能够理解 Transformer 各部分如何消耗计算和内存是很有用的。我们将逐步进行一些基本的"FLOPs 核算"。Transformer 中绝大多数 FLOPS 都是矩阵乘法,因此我们的核心方法很简单:

1. 写下 Transformer 前向传播中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步,以下事实会很有用:

> **规则:** 给定 A ∈ ℝ^(m×n) 和 B ∈ ℝ^(n×p),矩阵-矩阵乘积 AB 需要 2mnp FLOPs。

要理解这一点,注意 (AB)[i,j] = A[i,:]·B[:,j],而这个点积需要 n 次加法和 n 次乘法(2n FLOPs)。然后,由于矩阵-矩阵乘积 AB 有 m×p 个条目,总 FLOPS 为 (2n)(mp) = 2mnp。

现在,在做下一题之前,逐一检查你的 Transformer 块和 Transformer LM 的每个组件,列出所有矩阵乘法及其相关的 FLOPs 开销,可能会很有帮助。

#### 问题(transformer_accounting):Transformer LM 资源核算(5 分)

(a) 考虑一个使用我们作业架构的 GPT-2 XL 规模模型,其配置如下:

- `vocab_size`:50,257
- `context_length`:1,024
- `num_layers`:48
- `d_model`:1,600
- `num_heads`:25
- `d_ff`:4,288(最接近 (8/3)×1,600 的 64 的倍数)

假设我们用这个配置构造模型。我们的模型有多少可训练参数?假设每个参数都用单精度浮点数表示,仅加载这个模型需要多少内存?

**交付物:** 一到两句话回答。

(b) 找出完成我们 GPT-2 XL 形状模型一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs?假设我们的输入序列有 `context_length` 个 token。

**交付物:** 一个矩阵乘法列表(附描述),以及所需的总 FLOPs 数。

(c) 基于你上面的分析,模型的哪些部分需要最多 FLOPs?

**交付物:** 一到两句话回答。

(d) 对 GPT-2 small(12 层,768 d_model,12 头)、GPT-2 medium(24 层,1024 d_model,16 头)和 GPT-2 large(36 层,1280 d_model,20 头)重复你的分析。随着模型规模增大,Transformer LM 的哪些部分占总 FLOPs 的比例更大或更小?

**交付物:** 对每个模型,提供模型组件及其相关 FLOPs 的分解(作为一次前向传播所需总 FLOPs 的比例)。此外,用一到两句话描述改变模型规模如何改变各组件 FLOPs 的占比。

(e) 取 GPT-2 XL,将上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化?各模型组件 FLOPs 的相对贡献如何变化?

**交付物:** 一到两句话回答。
## 4 训练 Transformer LM

我们现在已经有了预处理数据的步骤(通过分词器)和模型(Transformer)。剩下的是构建支持训练的所有代码。这包括以下内容:

- **损失:** 我们需要定义损失函数(交叉熵)。
- **优化器:** 我们需要定义最小化该损失的优化器(AdamW)。
- **训练循环:** 我们需要所有支撑基础设施,用于加载数据、保存检查点和管理训练。

### 4.1 交叉熵损失

回顾一下,Transformer 语言模型为每个长度为 m+1 的序列 x 和 i = 1,…,m 定义了分布 p_θ(x_{i+1} | x_{1:i})。给定由长度为 m+1 的序列组成的训练集 D,我们定义标准交叉熵(负对数似然)损失函数:

$$
\ell(\theta; D) = \frac{1}{|D| m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i}). \tag{16}
$$

(注意,Transformer 的一次前向传播会为所有 i = 1,…,m 产生 p_θ(x_{i+1} | x_{1:i})。)

特别地,Transformer 为每个位置 i 计算 logits o_i ∈ ℝ^(vocab_size),从而得到:[^6]

$$
p(x_{i+1} \mid x_{1:i}) = \mathrm{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size} \exp(o_i[a])}. \tag{17}
$$

交叉熵损失通常是相对于 logits 向量 o_i ∈ ℝ^(vocab_size) 和目标 x_{i+1} 定义的。[^7]

[^6]: 注意,o_i[k] 指向量 o_i 中索引 k 处的值。

[^7]: 这对应于 x_{i+1} 上的 Dirac delta 分布与预测的 softmax(o_i) 分布之间的交叉熵。

实现交叉熵损失需要小心处理数值问题,就像 softmax 的情况一样。

#### 问题(cross_entropy):实现交叉熵(1 分)

**交付物:** 编写一个计算交叉熵损失的函数,它接收预测的 logits(o_i)和目标(x_{i+1}),并计算交叉熵 ℓ_i = −log softmax(o_i)[x_{i+1}]。你的函数应处理以下内容:

- 减去最大元素以保证数值稳定性。
- 尽可能抵消 log 和 exp。
- 处理任何额外的批次维度,并返回批次上的平均值。与第 3.2 节一样,我们假定类批次维度总是位于词表大小维度之前。

实现 `[adapters.run_cross_entropy]`,然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

**困惑度(Perplexity)**

交叉熵足以用于训练,但在评估模型时,我们还希望报告困惑度。对于一个长度为 m 的序列,若交叉熵损失为 ℓ₁,…,ℓₘ:

$$
\mathrm{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right). \tag{18}
$$

### 4.2 SGD 优化器

现在我们有了损失函数,将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降(SGD)。我们从随机初始化的参数 θ₀ 开始。然后,对于每一步 t = 0,…,T−1,执行以下更新:

$$
\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \tag{19}
$$

其中 B_t 是从数据集 D 中采样的随机数据批次,学习率 α_t 和批次大小 |B_t| 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

要实现我们的优化器,我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法:

- `def __init__(self, params, ...)` 应初始化你的优化器。这里,`params` 是要优化的参数集合(或参数组,以防用户想对模型的不同部分使用不同的超参数,例如不同的学习率)。确保将 `params` 传递给基类的 `__init__` 方法,基类会存储这些参数,供 `step` 使用。你可以根据优化器接受额外参数(例如学习率是常见的一个),并将它们作为字典传递给基类构造函数,字典的键是你为这些参数选择的名称(字符串)。
- `def step(self)` 应执行一次参数更新。在训练循环中,它会在反向传播之后被调用,因此你可以访问最后一个批次的梯度。该方法应遍历每个参数张量 p 并就地修改它们,即根据梯度 `p.grad`(如果存在,该张量表示损失相对于该参数的梯度)设置 `p.data`,它保存与该参数关联的张量。

PyTorch 优化器 API 有一些微妙之处,因此用示例来解释会更容易。为了让示例更丰富,我们将实现 SGD 的一个轻微变体,其中学习率随训练衰减,从初始学习率 α 开始,随时间推移采取越来越小的步长:

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t). \tag{20}
$$

让我们看看这个版本的 SGD 如何实现为 PyTorch `Optimizer`:

```python
from collections.abc import Callable, Iterable
from typing import Optional

import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
```

在 `__init__` 中,我们将参数以及默认超参数传递给基类构造函数(参数可能以组的形式传入,每组有不同的超参数)。如果参数只是单个 `torch.nn.Parameter` 对象集合,基类构造函数会创建一个组并为其分配默认超参数。然后,在 `step` 中,我们遍历每个参数组,再遍历该组中的每个参数,并应用公式 20。这里,我们将迭代次数保存为与每个参数关联的状态:我们先读取该值,在梯度更新中使用它,然后更新它。API 规定用户可能传入一个可调用的 closure,以便在优化器步骤之前重新计算损失。我们使用的优化器不需要它,但为了符合 API,我们加上了它。

为了看到它工作,我们可以使用以下最小训练循环示例:

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step.
```

这是训练循环的典型结构:在每次迭代中,我们计算损失并运行一步优化器。训练语言模型时,我们的可学习参数将来自模型(在 PyTorch 中,`m.parameters()` 给我们这个集合)。损失将在一个采样数据批次上计算,但训练循环的基本结构是一样的。

#### 问题(learning_rate_tuning):调学习率(1 分)

正如我们将看到的,对训练影响最大的超参数之一是学习率。让我们在上面的玩具示例中实践一下。用另外三个学习率值——1e1、1e2 和 1e3——运行上面的 SGD 示例,只训练 10 次迭代。每个学习率下损失会发生什么?它是衰减得更快、更慢,还是发散(即在训练过程中增大)?

**交付物:** 用一到两句话回答你观察到的行为。

### 4.3 AdamW

现代语言模型通常使用更复杂的优化器训练,而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生版本。我们将使用 AdamW [I. Loshchilov et al., 2019],它在最近的工作中被广泛使用。AdamW 提出了对 Adam 的修改,通过以与梯度更新解耦的方式加入权重衰减(每次迭代我们将参数拉向 0)来改进正则化。我们将按照 I. Loshchilov et al. [23] 的算法 2 实现 AdamW。

AdamW 是有状态的:对于每个参数,它跟踪其一阶矩和二阶矩的运行估计。因此,AdamW 使用额外内存,以换取更好的稳定性和收敛性。除学习率 α 外,AdamW 还有一对控制矩估计更新的超参数(β₁, β₂),以及一个权重衰减率 λ。典型应用将 (β₁, β₂) 设为 (0.9, 0.999),但 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 等大型语言模型通常用 (0.9, 0.95) 训练。该算法可以写如下,其中 ε 是一个小值(例如 10⁻⁸),用于在 v 中出现极小值时提高数值稳定性:

> **算法 1:AdamW 优化器**
>
> 1. init(θ) ▷ 初始化可学习参数
> 2. m ← 0 ▷ 一阶矩向量的初值;与 θ 同形
> 3. v ← 0 ▷ 二阶矩向量的初值;与 θ 同形
> 4. for t = 1,…,T do
> 5. &emsp;采样一批数据 B_t
> 6. &emsp;g ← ∇_θ ℓ(θ; B_t) ▷ 计算损失的梯度
> 7. &emsp;α_t ← α √(1−β₂^t) / (1−β₁^t) ▷ 计算第 t 次迭代调整后的 α
> 8. &emsp;θ ← θ − αλθ ▷ 应用权重衰减
> 9. &emsp;m ← β₁m + (1−β₁)g ▷ 更新一阶矩估计
> 10. &emsp;v ← β₂v + (1−β₂)g² ▷ 更新二阶矩估计
> 11. &emsp;θ ← θ − α_t m / (√v + ε) ▷ 应用矩调整后的权重更新
> 12. end for
>
> 算法 1:AdamW 优化器伪代码。

注意 t 从 1 开始。现在你将实现这个优化器。

#### 问题(adamw):实现 AdamW(2 分)

**交付物:** 将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 α,以及 β、ε 和 λ 超参数。为帮助你保存状态,基类 `Optimizer` 提供了一个字典 `self.state`,它将 `nn.Parameter` 对象映射到一个字典,用于存储你对该参数需要的任何信息(对于 AdamW,就是矩估计)。实现 `[adapters.get_adamw_cls]`,并确保它通过 `uv run pytest -k test_adamw`。

#### 问题(adamw_accounting):使用 AdamW 训练的资源核算(2 分)

让我们计算运行 AdamW 需要多少内存和计算。假设每个张量都使用 float32。

(a) 运行 AdamW 需要多少峰值内存?请根据参数、激活值、梯度和优化器状态的内存使用量来分解你的答案。用 batch_size 和模型超参数(vocab_size、context_length、num_layers、d_model、num_heads)表示你的答案。假设 d_ff = (8/3)×d_model。

为简单起见,在计算激活值的内存使用量时,只考虑以下组件:

- **Transformer 块**
  - RMSNorm(s)
  - 多头自注意力子层:QKV 投影、QKᵀ 矩阵乘法、softmax、值的加权求和、输出投影。
  - 逐位置前馈(SwiGLU):W₁、W₂、门控分支上的 SiLU、逐元素乘积、W₃
- 最终 RMSNorm
- 输出嵌入
- logits 上的交叉熵

**交付物:** 参数、激活值、梯度和优化器状态各自的代数表达式,以及总和。

(b) 将你的答案代入 GPT-2 XL 形状的模型,得到一个只依赖 batch_size 的表达式。你能使用的、仍能放入 80GB 内存的最大 batch size 是多少?

**交付物:** 形如 a·batch_size + b 的表达式(其中 a、b 为数值),以及一个表示最大 batch size 的数。

(c) 运行一步 AdamW 需要多少 FLOPs?

**交付物:** 一个代数表达式,并附简要论证。

(d) 模型 FLOPs 利用率(MFU)定义为实际吞吐量(每秒 token 数)与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 在"float32"(实际上是 TensorFloat-32,现实中是"bfloat19")操作上的理论峰值为 495 teraFLOP/s。假设你能达到 50% MFU,在单块 H100 上训练一个 GPT-2 XL、400K 步、batch size 为 1024,需要多长时间?遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26],假设反向传播的 FLOPs 是前向传播的两倍。

**交付物:** 训练需要的小时数,并附简要论证。

### 4.4 学习率调度

能使损失下降最快的学习率取值往往在训练过程中发生变化。在训练 Transformer 时,通常使用学习率调度:起初使用较大的学习率,在开始阶段更快地更新,随着模型训练慢慢衰减到较小的值。[^8] 在本作业中,我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的余弦退火调度。

调度器就是一个函数,它接收当前步 t 和其他相关参数(例如初始和最终学习率),并返回第 t 步梯度更新要使用的学习率。最简单的调度是常数函数,对任意 t 都返回相同的学习率。

余弦退火学习率调度接收:(i) 当前迭代 t;(ii) 最大学习率 α_max;(iii) 最小(最终)学习率 α_min;(iv) 预热迭代次数 T_w;(v) 余弦退火的最终迭代 T_c。第 t 次迭代的学习率定义为:

- **(预热)** 若 t < T_w,则 α_t = (t / T_w) α_max。
- **(余弦退火)** 若 T_w ≤ t ≤ T_c,则 α_t = α_min + (1/2)(1 + cos((t − T_w)/(T_c − T_w) π))(α_max − α_min)。
- **(退火后)** 若 t > T_c,则 α_t = α_min。

[^8]: 有时也会使用一种让学习率重新上升(restart)的调度,以帮助越过局部极小值。

#### 问题(learning_rate_schedule):实现带预热的余弦学习率调度(1 分)

编写一个函数,接收 t、α_max、α_min、T_w 和 T_c,并根据上面定义的调度返回学习率 α_t。然后实现 `[adapters.get_lr_cosine_schedule]`,并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

训练期间,我们有时会遇到产生大梯度的训练样本,这会破坏训练稳定性。为缓解这一问题,实践中常使用的一种技术是梯度裁剪。其思想是:在每次反向传播之后、执行优化器步骤之前,对梯度的范数施加限制。

给定(所有参数的)梯度 g,我们计算其 ℓ2 范数 ‖g‖₂。如果该范数小于最大值 M,则保持 g 不变;否则,我们将 g 按因子 M/(‖g‖₂ + ε) 缩小(其中加入一个小 ε,如 10⁻⁶,以保证数值稳定性)。注意,结果的范数将略小于 M。

#### 问题(gradient_clipping):实现梯度裁剪(1 分)

编写一个实现梯度裁剪的函数。你的函数应接收一个参数列表和一个最大 ℓ2 范数。它应就地修改每个参数的梯度。使用 ε = 10⁻⁶(PyTorch 默认值)。然后实现适配器 `[adapters.run_gradient_clipping]`,并确保它通过 `uv run pytest -k test_gradient_clipping`。
## 5 训练循环

现在,我们终于可以把至今构建的主要组件组合起来了:分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据(例如你在 `tokenizer_experiments` 中准备的数据)是一个单一的 token 序列 x = (x₁,…,xₙ)。即使源数据可能由单独的文档组成(例如不同的网页或源代码文件),一种常见做法是将它们全部拼接成一个 token 序列,并在它们之间加入分隔符(例如 `<|endoftext|>` token)。

数据加载器将其转换为批次流,每个批次由 B 个长度为 m 的序列组成,并配以对应的、同样长度为 m 的下一个 token。例如,对于 B = 1,m = 3,`([x₂,x₃,x₄],[x₃,x₄,x₅])` 可能是一个批次。

以这种方式加载数据简化了训练的许多方面。首先,任何 1 ≤ i ≤ n−m 都给出一个有效的训练序列,因此采样训练序列很简单。由于所有训练序列长度相同,无需对输入序列进行填充,这提高了硬件利用率(也通过增大 batch size B 实现)。最后,我们也不需要加载整个数据集来采样训练数据,从而易于处理否则可能无法放入内存的大型数据集。

#### 问题(data_loading):实现数据加载(2 分)

**交付物:** 编写一个函数,接收一个 numpy 数组 x(包含 token ID 的整数数组)、一个 `batch_size`、一个 `context_length` 和一个 PyTorch 设备字符串(例如 `'cpu'` 或 `'cuda:0'`),并返回一对张量:采样的输入序列和对应的下一个 token 目标。两个张量都应具有形状 `(batch_size, context_length)`,包含 token ID,并且都应放在请求的设备上。要针对我们提供的测试来测试你的实现,你首先需要实现位于 `[adapters.run_get_batch]` 的测试适配器。然后运行 `uv run pytest -k test_get_batch` 来测试你的实现。

> **低资源提示:在 CPU 或 Apple Silicon 上加载数据**
>
> 如果你打算在 CPU 或 Apple Silicon 上训练 LM,你需要将数据移动到正确的设备(同样,稍后你也应为模型使用相同的设备)。
>
> 如果你在 CPU 上,可以使用 `'cpu'` 设备字符串;在 Apple Silicon(M* 芯片)上,可以使用 `'mps'` 设备字符串。
>
> 关于 MPS 的更多信息,请查看以下资源:
>
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

如果数据集太大无法加载到内存中怎么办?我们可以使用一个名为 `mmap` 的 Unix 系统调用,它将磁盘上的文件映射到虚拟内存,并在访问该内存位置时惰性加载文件内容。因此,你可以"假装"整个数据集都在内存中。NumPy 通过 `np.memmap`(或者,如果你最初用 `np.save` 保存了数组,则可通过 `np.load` 的 `mmap_mode='r'` 标志)实现这一点,它会返回一个类 numpy 数组对象,在你访问条目时按需加载。在训练期间从数据集(即 numpy 数组)采样时,务必以内存映射模式加载数据集(根据你保存数组的方式,通过 `np.memmap` 或 `np.load` 的 `mmap_mode='r'` 标志)。还要确保指定与你加载的数组匹配的 `dtype`。显式验证内存映射数据看起来正确(例如,不包含超出预期词表大小的值)可能会有帮助。

### 5.2 检查点(Checkpointing)

除加载数据外,我们在训练时还需要保存模型。启动作业时,我们常常希望能够恢复中途停止的训练运行(例如由于作业超时、机器故障等)。即使一切顺利,我们之后也可能希望访问中间模型(例如事后研究训练动态、从训练不同阶段的模型采样等)。

检查点应包含恢复训练所需的所有状态。我们当然至少希望能够恢复模型权重。如果使用有状态优化器(如 AdamW),我们还需要保存优化器的状态(例如 AdamW 的矩估计)。最后,要恢复学习率调度,我们需要知道停止时的迭代次数。PyTorch 让保存这些都很容易:每个 `nn.Module` 都有一个 `state_dict()` 方法,返回包含所有可学习权重的字典;我们之后可以用对应的方法 `load_state_dict()` 恢复这些权重。任何 `torch.optim.Optimizer` 也是如此。最后,`torch.save(obj, dest)` 可以将一个对象(例如一个字典,其值包含张量,也包含整数等普通 Python 对象)转储到文件(路径)或类文件对象,之后可以用 `torch.load(src)` 将其加载回内存。

#### 问题(checkpointing):实现模型检查点(1 分)

实现以下两个函数来加载和保存检查点:

```python
def save_checkpoint(model, optimizer, iteration, out)
```

应将模型、优化器和迭代次数中的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法获取相关状态,并使用 `torch.save(obj, out)` 将 obj 转储到 out 中(PyTorch 此处支持路径或类文件对象)。一个典型选择是让 obj 为一个字典,但你可以使用任何格式,只要之后能加载你的检查点即可。

该函数期望以下参数:

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model, optimizer)
```

应从 `src`(路径或类文件对象)加载检查点,然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点中的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容,并使用模型和优化器中的 `load_state_dict` 方法将它们恢复到之前的状态。

该函数期望以下参数:

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器,并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在,终于到了把你实现的所有组件整合到主训练脚本中的时候了。让使用不同超参数启动训练运行变得容易(例如通过命令行参数传入)会很有回报,因为之后你会多次这样做,以研究不同选择如何影响训练。

#### 问题(training_together):整合起来(4 分)

**交付物:** 编写一个脚本,运行训练循环,在用户提供的输入上训练你的模型。特别地,我们建议你的训练脚本至少支持以下内容:

- 能够配置和控制各种模型和优化器超参数。
- 使用 `np.memmap` 内存高效地加载大型训练和验证数据集。
- 将检查点序列化到用户提供的路径。
- 定期记录训练和验证性能(例如记录到控制台和/或 Weights and Biases 等外部服务)。[^9]

[^9]: wandb.ai

## 6 生成文本

现在我们可以训练模型了,最后需要的是从模型生成文本的能力。回顾一下,语言模型接收一个(可能批量的)长度为 `sequence_length` 的整数序列,并产生一个大小为 `(sequence_length, vocab_size)` 的矩阵,其中序列的每个元素都是一个预测该位置之后下一个 token 的概率分布。我们现在将编写一些函数,将其转化为新序列的采样方案。

**Softmax**

按照标准惯例,语言模型的输出是最终线性层的输出("logits"),因此我们必须通过 softmax 操作将其转换为归一化概率,我们在前面的公式 10 中已经见过。

**解码**

要从我们的模型生成文本(解码),我们将向模型提供前缀 token 序列("提示词"),并要求它在词表上产生一个预测序列中下一个 token 的概率分布。然后,我们将从词表项上的这个分布中采样,以确定下一个输出 token。

具体来说,解码过程的一步应接收序列 x₁…t,并通过以下公式返回 token x_{t+1}:

$$
P(x_{t+1} = i \mid x_{1\dots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)} \tag{21}
$$

$$
v = \mathrm{TransformerLM}(x_{1\dots t})_t \in \mathbb{R}^{vocab\_size} \tag{22}
$$

其中 TransformerLM 是我们的模型,它接收长度为 `sequence_length` 的序列作为输入,并产生大小为 `(sequence_length, vocab_size)` 的矩阵;我们取该矩阵的最后一个元素,因为我们要找的是第 t 个位置的下一个 token 预测。

通过反复从这些单步条件分布中采样(将之前生成的输出 token 追加到下一个解码时间步的输入中),直到生成序列结束 token `<|endoftext|>`(或达到用户指定的最大生成 token 数),我们就得到了一个基本的解码器。

**解码器技巧**

我们将使用小模型做实验,而小模型有时会生成质量很低的文本。两个简单的解码器技巧可以帮助解决这些问题。第一,在**温度缩放(temperature scaling)**中,我们用温度参数 τ 修改 softmax,新的 softmax 为

$$
\mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{vocab\_size} \exp(v_j/\tau)}. \tag{23}
$$

注意,令 τ → 0 会使 v 的最大元素占主导,softmax 的输出变为集中在该最大元素上的 one-hot 向量。

第二,另一个技巧是**核采样(nucleus sampling)或 top-p 采样**,我们通过截断低概率 token 来修改采样分布。设 q 是我们从(经温度缩放的)softmax 得到的、大小为 `vocab_size` 的概率分布。超参数为 p 的核采样根据以下公式产生下一个 token:

$$
P(x_{t+1} = i \mid q) =
\begin{cases}
\frac{q_i}{\sum_{j \in V(p)} q_j} & \text{若 } i \in V(p) \\
0 & \text{其他}
\end{cases} \tag{24}
$$

其中 V(p) 是满足 ∑_{j∈V(p)} q_j ≥ p 的最小索引集合。你可以先按大小对概率分布 q 排序,并选择最大的词表元素,直到达到目标水平 p,从而轻松计算该量。

#### 问题(decoding):解码(3 分)

**交付物:** 实现一个从你的语言模型解码的函数。我们建议你支持以下功能:

- 为用户提供的提示词生成补全(即接收一些 x₁…t,并采样补全,直到遇到 `<|endoftext|>` token)。
- 允许用户控制生成的最大 token 数。
- 给定期望的温度值,在采样前对预测的下一个 token 分布应用 softmax 温度缩放。
- 给定用户指定的阈值,进行 top-p 采样([A. Holtzman et al., 2020],也称为核采样)。
## 7 实验

现在是时候把所有东西组合起来,在预训练数据集上训练(小型)语言模型了。

### 7.1 如何运行实验与交付物

理解 Transformer 架构组件背后原理的最佳方式,是亲自修改它并运行。没有东西能替代动手实践。

为此,能够快速、一致地进行实验并记录你所做的事情非常重要。为了快速实验,我们将在小规模模型(总共约 17M 参数)和简单数据集(TinyStories)上运行许多实验。为保持一致,你将以系统化的方式消融组件并改变超参数;为保留记录,我们将要求你提交实验日志以及每个实验对应的学习曲线。

为了能够提交损失曲线,请确保定期评估验证损失,并记录步数和实际耗时(wall-clock time)。你可能会发现 Weights and Biases 等日志基础设施很有帮助。

#### 问题(experiment_log):实验记录(3 分)

为你的训练和评估代码创建实验跟踪基础设施,使你能够相对于梯度步数和实际耗时跟踪实验和损失曲线。

**交付物:** 你的实验日志基础设施代码,以及本节下列作业问题的实验日志(记录你尝试过的所有内容的文档)。

### 7.2 TinyStories

我们将从一个非常简单的数据集(TinyStories;R. Eldan et al. [1])开始,模型会在其上快速训练,我们可以看到一些有趣的行为。获取该数据集的说明见第 1 节。该数据集的样子示例如下。

#### 示例(tinystories_example):TinyStories 中的一个例子

> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

#### 7.2.1 超参数调优

我们将告诉你一些非常基础的起始超参数,并要求你为其他超参数找到效果良好的设置。

- **词表大小 10000。** 典型词表大小在数万到数十万之间。你应改变这一值,看看词表和模型行为如何变化。
- **上下文长度 256。** TinyStories 等简单数据集可能不需要长序列长度,但对于后面的 OpenWebText 数据,你可能想改变它。尝试改变它,看看对每次迭代运行时间和最终困惑度的影响。
- **d_model 512。** 这比许多小型 Transformer 论文使用的 768 维略小,但这会让运行更快。
- **d_ff 1344。** 这大约是 (8/3)d_model,同时是 64 的倍数,有利于 GPU 性能。
- **RoPE theta 参数 Θ 10000。**
- **层数和头数 4 层,16 头。** 合起来,这将给出约 17M 非嵌入参数,是相当小的 Transformer。
- **处理的总 token 数 327,680,000**(你的 batch size × 总步数 × 上下文长度应大致等于该值)。

你应进行一些试错,为以下其他超参数找到良好的默认值:学习率、学习率预热、其他 AdamW 超参数(β₁、β₂、ε)以及权重衰减。你可以在 D. P. Kingma et al. [22] 中找到这类超参数的一些典型选择。

#### 7.2.2 组合起来

现在,你可以通过获取训练好的 BPE 分词器、对训练数据集进行分词,并在你编写的训练循环中运行,把所有东西组合起来。重要提示:如果你的实现正确且高效,上述超参数在 1 块 B200 GPU 上的运行时间应约为 20–30 分钟。如果你的运行时间长得多,请检查并确保你的数据加载、检查点或验证损失代码没有成为运行时间的瓶颈,并确保你的实现正确地进行了批处理。

#### 7.2.3 调试模型架构的技巧与诀窍

我们强烈建议你熟悉 IDE 内置的调试器(例如 VSCode/Zed),与使用 print 语句调试相比,这会为你节省时间。如果你使用文本编辑器,可以使用 `ipdb` 之类的工具。调试模型架构时,其他一些良好实践包括:

- 开发任何神经网络架构时,常见的第一个步骤是过拟合单个小批次。如果你的实现正确,你应该能迅速把训练损失降到接近零。
- 在各个模型组件中设置调试断点,检查中间张量的形状,以确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数,以确保它们没有爆炸或消失。

#### 问题(learning_rate):调学习率(2 B200 小时)(3 分)

学习率是最重要的调优超参数之一。以你训练好的基础模型为准,回答以下问题:

(a) 对学习率进行超参数扫描,并报告最终损失(如果优化器发散,则注明发散)。

**交付物:** 多个学习率对应的学习曲线。解释你的超参数搜索策略。

**交付物:** 一个在 TinyStories 上验证损失(每 token)至多为 1.45 的模型。

> **低资源提示:在 CPU 或 Apple Silicon 上训练几步**
>
> 如果你在 cpu 或 mps 上运行,应改为将处理的总 token 数减少到 40,000,000,这足以产生相当流畅的文本。你也可以把目标验证损失从 1.45 提高到 2.00。
>
> 我们在 M4 Max 芯片、36 GB 内存上使用调好学习率的解答代码,使用 batch size × 总步数 × 上下文长度 = 32×5000×256 = 40,960,000 个 token,在 cpu 上耗时 1 小时 22 分钟,在 mps 上耗时 36 分钟。在第 5000 步,我们达到 1.80 的验证损失。
>
> 一些额外提示:
>
> - 当使用 N 个训练步时,我们建议调整余弦学习率衰减调度,使其恰好在第 N 步终止衰减(即达到最小学习率)。
> - 使用 mps 时,不要使用 TF32 内核,即不要像在 cuda 设备上那样设置
>   `torch.set_float32_matmul_precision('high')`,因为我们在 mps 上尝试启用 TF32 内核(torch 版本 2.9.0)时发现,后端有时会使用静默损坏的内核,导致训练不稳定。
> - 你可以通过使用 `torch.compile` 对你的模型进行 JIT 编译来加速训练。具体来说:
>   - 在 cpu 上,用 `model = torch.compile(model)` 编译你的模型;
>   - 在 mps 上,你可以使用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。
>   截至 torch 版本 2.9.0,mps 上不支持使用 Inductor 进行编译。

(b) 民间经验认为,最佳学习率处于"稳定边缘"。研究学习率发散的点与你最佳学习率之间的关系。

**交付物:** 学习率递增时的学习曲线,其中至少包含一次发散运行,并分析这与收敛速度的关系。

现在让我们改变 batch size,看看训练会发生什么。批次大小很重要——它们通过进行更大的矩阵乘法让我们从 GPU 获得更高效率,但我们是否总是希望批次大小很大?让我们运行一些实验来找出答案。

#### 问题(batch_size_experiment):批次大小变化(1 B200 小时)(1 分)

将你的批次大小从 1 一直变化到 GPU 内存限制。尝试至少几个介于两者之间的批次大小,包括 64 和 128 等典型大小。

**交付物:** 不同批次大小运行的学习曲线。如有必要,应重新优化学习率。

**交付物:** 几句话讨论你对批次大小及其对训练影响的发现。

有了解码器,我们现在可以生成文本了!我们将从模型生成,看看它有多好。作为参考,你得到的输出至少应不差于下面的示例。

#### 示例(ts_generate_example):TinyStories 语言模型的样本输出

> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

> **低资源提示:在 CPU 或 Apple Silicon 上生成文本**
>
> 如果你改为使用处理 40M token 的低资源配置,你应该会看到仍然像英语但不如上面流畅的生成结果。例如,我们训练了 40M token 的 TinyStories 语言模型的样本输出如下:
>
> > Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> > "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"
> > As Sue watched the sky moved, what it was. She

以下是精确的问题陈述和我们要求的内容:

#### 问题(generate):生成文本(1 分)

使用你的解码器和训练好的检查点,报告你的模型生成的文本。你可能需要调整解码器参数(温度、top-p 等)以获得流畅的输出。

**交付物:** 至少 256 个 token 的文本转储(或直到第一个 `<|endoftext|>` token),并简要评论该输出的流畅度,以及至少两个影响该输出好坏的因 素。
### 7.3 消融与架构修改

理解 Transformer 的最佳方式,是真正修改它并观察其行为。现在我们将进行一些简单的消融和修改。

**消融 1:层归一化**

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒点险。让我们从每个 Transformer 块中移除 RMSNorm,看看会发生什么。

#### 问题(layer_norm_ablation):移除 RMSNorm 并训练(0.5 B200 小时)(1 分)

从你的 Transformer 中移除所有 RMSNorm 并训练。在之前的最优学习率下会发生什么?你能通过使用更低的学习率获得稳定性吗?

**交付物:** 移除 RMSNorm 并训练时的学习曲线,以及最佳学习率下的学习曲线。

**交付物:** 几句关于 RMSNorm 影响的评论。

现在让我们研究另一个乍看之下显得随意的层归一化选择。Pre-norm Transformer 块定义为

$$
z = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)) \tag{25}
$$

$$
y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)). \tag{26}
$$

这是对原始 Transformer 架构少数"共识性"修改之一,原始架构使用 post-norm 方法:

$$
z = \mathrm{RMSNorm}(x + \mathrm{MultiHeadSelfAttention}(x)) \tag{27}
$$

$$
y = \mathrm{RMSNorm}(z + \mathrm{FFN}(z)). \tag{28}
$$

让我们改回 post-norm 方法,看看会发生什么。

#### 问题(pre_norm_ablation):实现 post-norm 并训练(0.5 B200 小时)(1 分)

将你的 pre-norm Transformer 实现修改为 post-norm。用 post-norm 模型训练,看看会发生什么。

**交付物:** post-norm Transformer 的学习曲线,并与 pre-norm 的比较。

我们看到层归一化对 Transformer 的行为有重大影响,甚至层归一化的位置也很重要。

**消融 2:位置嵌入**

接下来,我们将研究位置嵌入对模型性能的影响。具体来说,我们将把带 RoPE 的基础模型与完全不包含位置嵌入(NoPE)的模型进行比较。事实证明,仅解码器 Transformer(即我们实现的、带因果掩码的那些)理论上可以在没有显式位置嵌入的情况下推断相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。我们现在将实证检验 NoPE 与 RoPE 相比的表现。

#### 问题(no_pos_emb):实现 NoPE(0.5 B200 小时)(1 分)

修改你带 RoPE 的 Transformer 实现,完全移除位置嵌入信息,看看会发生什么。

**交付物:** 比较 RoPE 与 NoPE 性能的学习曲线。

**消融 3:SwiGLU vs. SiLU**

接下来,我们将遵循 N. Shazeer [20],通过比较 SwiGLU 前馈网络与使用 SiLU 激活但不带门控线性单元(GLU)的前馈网络的性能,来检验前馈网络中门控的重要性:

$$
\mathrm{FFN}_{\mathrm{SiLU}}(x) = W_2 \mathrm{SiLU}(W_1 x). \tag{29}
$$

回顾一下,在我们的 SwiGLU 实现中,我们将内部前馈层的维度设为约 d_ff = (8/3)d_model(同时确保 d_ff mod 64 = 0,以利用 GPU 张量核心)。在这个消融基线中,你的 FFN_SiLU 实现应改为设置 d_ff = 4×d_model,以大致匹配默认 SwiGLU 前馈网络的参数量(后者有三个而不是两个权重矩阵)。

#### 问题(swiglu_ablation):SwiGLU vs. SiLU(0.5 B200 小时)(1 分)

**交付物:** 比较 SwiGLU 与 SiLU 前馈网络性能的学习曲线,参数数量大致匹配。

**交付物:** 几句话讨论你的发现。

> **低资源提示:GPU 资源有限的线上学生应在 TinyStories 上测试修改**
>
> 在作业的剩余部分,我们将转向一个更大规模、噪声更多的网络数据集(OpenWebText),实验架构修改,并(可选地)向课程排行榜提交。
>
> 在 OpenWebText 上训练 LM 达到流畅需要很长时间,因此我们建议 GPU 访问受限的线上学生继续在 TinyStories 上测试修改(使用验证损失作为评估性能的指标)。

### 7.4 在 OpenWebText 上运行

现在我们将转向一个更标准的、由网络爬取构建的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一个小样本也以单个文本文件提供:访问该文件的方法见第 1 节。

以下是 OpenWebText 中的一个例子。注意文本更加真实、复杂和多样。你可能想浏览一下训练数据集,以了解网络爬取语料的训练数据是什么样子。

#### 示例(owt_example):OWT 中的一个例子

> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. […]

注意:你可能需要为此实验重新调整超参数,例如学习率或批次大小。

#### 问题(main_experiment):在 OWT 上实验(2 B200 小时)(2 分)

使用与 TinyStories 相同的模型架构和总训练迭代次数,在 OpenWebText 上训练你的语言模型。这个模型表现如何?

**交付物:** 你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 相比损失有何差异——我们应如何解释这些损失?

**交付物:** 来自 OpenWebText LM 的生成文本,格式与 TinyStories 输出相同。这段文本的流畅度如何?为什么即使模型和计算预算与 TinyStories 相同,输出质量却更差?

### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你快要完成了!现在你将尝试改进 Transformer 架构,并看看你的超参数和架构与班上其他学生相比如何。

**排行榜规则**

除以下限制外,没有其他限制:

- **运行时间:** 你的提交最多可以在 B200 上运行 45 分钟。如果你使用 SLURM 或 Modal,可能希望在提交脚本中强制实施这一点。
- **数据:** 你只能使用我们提供的 OpenWebText 训练数据集。

除此之外,你可以随心所欲地做任何事。

如果你在寻找一些可以实现的思路,可以查看以下资源:

- 最先进的开源 LLM 系列,例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库(github.com/KellerJordan/modded-nanogpt),社区成员在那里发布了许多有趣的修改,用于"speedrun"小规模语言模型预训练。例如,一个可以追溯到原始 Transformer 论文的常见修改是将输入嵌入和输出嵌入的权重绑定在一起(参见 A. Vaswani et al. [8](第 3.4 节)和 A. Chowdhery et al. [16](第 2 节))。如果你确实尝试权重绑定,你可能需要降低嵌入/LM head 初始化的标准差。

在尝试完整的 45 分钟运行之前,你会想在 OpenWebText 的一个小部分或 TinyStories 上测试这些修改。

需要提醒的是,我们注意到,你可能发现在此排行榜上效果良好的某些修改,可能无法推广到更大规模的预训练。我们将在课程的缩放定律(scaling laws)单元中进一步探讨这一想法。

#### 问题(leaderboard):排行榜(10 B200 小时)(6 分)

你将按照上述排行榜规则训练一个模型,目标是在 0.75 B200 小时内最小化语言模型的验证损失。

**交付物:** 记录的最终验证损失、一条清晰显示实际耗时 x 轴小于 45 分钟的相关学习曲线,以及你所做工作的描述。我们期望排行榜提交至少能击败 5.0 损失的朴素基线。在此提交到排行榜:github.com/stanford-cs336/assignment1-basics-leaderboard。

## 参考文献

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl
