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

def _processs_chunk(args) -> Counter:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    input_path, start, end, special_tokens = args
    # 读取对应段
    with open(input_path, "r", encoding="utf-8") as f:
        f.seek(start)
        chunk = f.read(end-start)
    # 根据特殊token进行拆分
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        s = re.split(special_pattern,chunk)
    else:
        s = [chunk]
    # 对每个段进行分词统计
    counter:Counter[str] = Counter()
    for seg in s:
        if not seg:
            continue
        for token in re.findall(PAT,seg):
            counter[token] += 1
    return counter

def pre_token(input_path: str | os.PathLike, special_tokens: list[str],split_special_token: bytes) -> dict[str,int]:
    """
    预分词
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
    # 1. 初始化词表
    # 2. 预分词
    # 3. 统计相邻对
    # 4. 选择频率最高的相邻对
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
