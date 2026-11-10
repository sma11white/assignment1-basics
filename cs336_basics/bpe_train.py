import regex as re
import os
from collections import Counter
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

def bytes_to_unicode() -> dict[int, str]:
    """字节 (0-255) -> Unicode 替身字符 的双射映射表"""
    # 1. 收集所有"安全"的可打印字节 (恒等映射区间)
    bs = (
        list(range(ord("!"), ord("~") + 1))      # 0x21 ~ 0x7E  (94 个)
        + list(range(ord("¡"), ord("¬") + 1))    # 0xA1 ~ 0xAC  (12 个)
        + list(range(ord("®"), ord("ÿ") + 1))    # 0xAE ~ 0xFF  (82 个)
    )                                            # 共 188 个

    cs = bs[:]                                   

    # 2. 剩下 68 个"不安全"字节, 依次分配到码点 256, 257, 258...
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    # 3. 字节值 -> 替身字符
    return dict(zip(bs, [chr(c) for c in cs]))

def _processs_chunk(args) -> Counter[tuple[str,...]]:
    """
    完整的预分词流程:原始语料 -> (去除<|endoftext|>,这里作为长文本的分隔符) -> 正则分词 -> 对得到的str列表编码为utf8,再使用bytes_to_unicode修改为unicode码 -> 统计各个unicode的str列表的个数
    """
    # 读取原始的语料
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8")
    # 根据特殊token进行拆分(去除特殊token)
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        s = re.split(special_pattern,chunk)
    else:
        s = [chunk]
    # 正则分词
    counter:Counter[tuple[str,...]] = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = bytes_to_unicode()
    for seg in s:
        if not seg:
            continue
        for token in re.findall(PAT,seg):
            # 编码统计合并 unicode
            counter[tuple(vocab[i] for i in token.encode("utf-8"))] += 1
    return counter

def pre_token(input_path: str | os.PathLike, special_tokens: list[str],split_special_token: bytes) -> dict[tuple[str,...],int]:
    """
    预分词阶段
    这里先分块,然后开进程并行的处理各个块
    """
    # 1. 安全的分块
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f,desired_num_chunks=160,split_special_token=split_special_token)
    # 2. 并行处理每个块
    tasks = [
        (input_path,start,end,special_tokens) for start,end in zip(boundaries[:-1],boundaries[1:])
    ]

    with Pool(processes=16) as pool:
        total: Counter[tuple[str,...]] = Counter()
        for counter in pool.map(_processs_chunk, tasks):
            total += counter

    return dict(total)

def _merge_word(word: tuple[str,...], a: str, b: str, merged_str: str) -> tuple[str,...]:
    """
    合并word中的a和b为merged_str
    """
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(merged_str)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    这里先使用的是pair,最后全部转为int
    return:
        vocab: 词表,将bytes映射到int
        merge: 合并表,记录合并的pair
    """
    # 1. 预分词阶段:原始语料 -> (去除特殊token) -> 构建初始词表 256 个字节的映射 -> 正则分词 -> 将str转bytes,再使用词表进行映射 
    word_freq:dict[tuple[str,...],int] = pre_token(input_path=input_path,special_tokens=special_tokens,split_special_token=special_tokens[0].encode("utf-8"))
    # 2. 初始化词表vocab,合并表merge,unicode和utf8的转换表
    b2u: dict[int, str] = bytes_to_unicode()
    token_bytes: dict[str, bytes] = {u: bytes([byte]) for byte, u in b2u.items()}
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        if len(vocab) < vocab_size:
            vocab[len(vocab)] = tok.encode("utf-8")
    merge: list[tuple[bytes, bytes]] = []
    # 3. 重复以下步骤直至达到词表大小:
    while len(vocab) < vocab_size:
    # a. 生成相邻表
        pair_counts:Counter[tuple[str,str]] = Counter()
        for k,v in word_freq.items():
            for j in range(len(k)-1):
                pair_counts[(k[j],k[j+1])] += v
    # b. 选最高频pair
        best_pair, _ = max(pair_counts.items(), key=lambda kv: (kv[1], (token_bytes[kv[0][0]], token_bytes[kv[0][1]])))
        a, b = best_pair
    # c. 记录合并pair,加入词表中
        merged_str = a + b
        merged_bytes = token_bytes[a] + token_bytes[b]
        merge.append((token_bytes[a], token_bytes[b]))
        token_bytes[merged_str] = merged_bytes
        vocab[len(vocab)] = merged_bytes  
    # d. 更新word_freq
        new_word_freq: dict[tuple[str,...],int] = {}
        for k,v in word_freq.items():
            w = _merge_word(k, a, b, merged_str) if (a in k and b in k) else k
            new_word_freq[w] = new_word_freq.get(w, 0) + v
        word_freq = new_word_freq
    return vocab, merge


if __name__ == "__main__":
    import pathlib
    f = pathlib.Path(__file__).parent.parent/"data"/"temp.txt"
    special = ["<|endoftext|>"]
    vocab_size = 10000
    import tracemalloc
    import time
    start_time = time.time()
    tracemalloc.start()
    temp = pre_token(input_path=f,special_tokens=special,split_special_token=special[0].encode("utf-8"))
    print(temp) 
    current, peak = tracemalloc.get_traced_memory()
    print(f"当前内存: {current / 1024 / 1024:.1f} MB, 峰值: {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()
    end_time = time.time()
    print(f"耗时: {end_time - start_time:.1f} s")
