# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]

import regex as re
import os
from collections import Counter
from heapq import heapify, heappush, heappop

def pre_token(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    """预分词: 先按特殊 token 切开, 特殊 token 内部的文字不参与训练"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "r", encoding="utf-8") as f:
        s = f.read()

    # 1. 按特殊 token 分段 (re.escape 防止 <| 等字符被当正则语法)
    if special_tokens:
        split_pat = "|".join(re.escape(st) for st in special_tokens)
        segments = re.split(split_pat, s)
    else:
        segments = [s]

    # 2. 只对普通片段做预分词
    res = []
    for seg in segments:
        res.extend(re.findall(PAT, seg))
    return res

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    


if __name__ == "__main__":
    import pathlib
    f = pathlib.Path(__file__).parent.parent/"data"/"TinyStoriesV2-GPT4-train.txt"
    special = ["<|endoftext|>"]
    vocab_size = 10000
    import tracemalloc
    tracemalloc.start()
    vocab, merges = bpe(input_path=f, special_tokens=special, vocab_size=vocab_size)
    current, peak = tracemalloc.get_traced_memory()
    print(f"当前内存: {current / 1024 / 1024:.1f} MB, 峰值: {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()
