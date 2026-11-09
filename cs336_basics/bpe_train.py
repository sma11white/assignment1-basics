# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]

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

    cs = bs[:]                                   # 替身初始值 = 自身

    # 2. 剩下 68 个"不安全"字节, 依次分配到码点 256, 257, 258...
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    # 3. 字节值 -> 替身字符
    return dict(zip(bs, [chr(c) for c in cs]))

def _processs_chunk(args) -> Counter[str]:
    """
    完整的预分词流程:原始语料 -> (去除<|endoftext|>,这里作为长文本的分隔符) -> 正则分词 -> 对得到的str列表编码为utf8,再使用bytes_to_unicode修改utf8编码 -> 统计各个bytes的个数
    """
    # 读取原始的语料
    input_path, start, end, special_tokens = args
    with open(input_path, "r", encoding="utf-8") as f:
        f.seek(start)
        chunk = f.read(end-start)
    # 根据特殊token进行拆分(去除特殊token)
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        s = re.split(special_pattern,chunk)
    else:
        s = [chunk]
    # 正则分词
    counter:Counter[str] = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = bytes_to_unicode()
    for seg in s:
        if not seg:
            continue
        for token in re.findall(PAT,seg):
            # 编码统计合并
            temp = [vocab[i] for i in list(token.encode("utf-8"))]
            counter["".join(temp)] = 1
    return counter

def pre_token(input_path: str | os.PathLike, special_tokens: list[str],split_special_token: bytes) -> dict[str,int]:
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
        total: Counter[str] = Counter()
        for counter in pool.map(_processs_chunk, tasks):
            total += counter

    return dict(total)

def bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 预分词阶段:原始语料 -> (去除特殊token) -> 构建初始词表 256 个字节的映射 -> 正则分词 -> 将str转bytes,再使用词表进行映射 
    return ({},[])


if __name__ == "__main__":
    import pathlib
    f = pathlib.Path(__file__).parent.parent/"data"/"TinyStoriesV2-GPT4-train.txt"
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
