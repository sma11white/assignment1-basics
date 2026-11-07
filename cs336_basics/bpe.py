# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]

import regex as re
import os
import numpy as np
from math import inf

def pre_token(input_path: str | os.PathLike)->list[str]:
    """
    预分词,不用清洗标点符号 
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "r", encoding="utf-8") as f:
        s = f.read()
    res = re.findall(PAT,s)

    return res

def bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 预分词, 确定除特殊词外的词表大小, 建立词表, 建立组合表
    vocab_rev: dict[bytes, int] = dict()          # token(bytes) -> 出现次数
    merge: list[tuple[bytes, bytes]] = list()
    # 预分词
    s = pre_token(input_path)
    # 每个词拆成单字节 token 列表（保持 bytes，不要转成 int/str 混用）
    arr: list[list[bytes]] = [list(w.encode("utf-8")) for w in s]
    # 确定除特殊词外的大小
    voclen = vocab_size - len(special_tokens)
    # 建词表：统计每个单字节 token 的出现次数
    for word in arr:
        for j in word:
            vocab_rev[j] = vocab_rev.get(j, 0) + 1  
    # 建立组合表 + 选出当前频次最高的相邻 pair（抽成函数，因为循环里要重复用）
    def find_max_pair(arr):
        a: dict[tuple[bytes, bytes], int] = {}
        maxa_num: int = -inf
        maxa_pair = None
        for word in arr:
            if len(word) <= 1:                       
                continue
            for j in range(len(word) - 1):           
                pair = (word[j], word[j + 1])
                a[pair] = a.get(pair, 0) + 1         
                if a[pair] > maxa_num:               
                    maxa_num = a[pair]
                    maxa_pair = pair
        return maxa_num, maxa_pair

    maxa_num, maxa_pair = find_max_pair(arr)
    maxa_sa, maxa_sb = maxa_pair

    now_voclen: int = len(vocab_rev)

    # 2. 更新词表（循环直到词表达到目标大小）
    while now_voclen < voclen and maxa_pair is not None:   
        merge.append(maxa_pair)                            
        maxa_str = maxa_sa + maxa_sb
        vocab_rev[maxa_str] = maxa_num
        vocab_rev[maxa_sa] -= maxa_num
        vocab_rev[maxa_sb] -= maxa_num
        # 3. 更新原始表（在词内用新 token 替换旧的 pair),重建组合表
        # 更新原始表
        for i in range(len(arr)):
            word = arr[i]
            if len(word) <= 1:
                continue
            temp_list = []
            j = 0
            while j < len(word):                           
                if j < len(word) - 1 and word[j] == maxa_sa and word[j + 1] == maxa_sb:
                    temp_list.append(word[j] + word[j + 1])   
                    j += 2
                else:
                    temp_list.append(word[j])
                    j += 1
            arr[i] = temp_list
        # 重建组合表
        maxa_num, maxa_pair = find_max_pair(arr)
        if maxa_pair is not None:
            maxa_sa, maxa_sb = maxa_pair
        now_voclen = len(vocab_rev)
    # 4. 将 vocab_rev 翻转成 id -> bytes，再加入特殊 token，返回结果
    vocab: dict[int, bytes] = {idx: tok for idx, tok in enumerate(vocab_rev)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    return vocab, merge


if __name__ == "__main__":
    s = "TODO"
    special_tokens:list[str] = ["|end|","|start|"]
    bpe(s,special_tokens=special_tokens)