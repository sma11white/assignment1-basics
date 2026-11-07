# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]

import regex as re
import os
import numpy as np


def pre_token(input_path: str | os.PathLike)->list[str]:
    """
    预分词,不用清洗标点符号 
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # TODO:逻辑之后再改
    s = "hello!who are you?I'm fine thank you"

    res = re.findall(PAT,s)

    return res

def bpe(input_path:str|os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs
        )->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 分词后词表的最大长度
    num = len(special_tokens)
    voclen = vocab_size - num
    # 预先分词后的结果
    arr = pre_token(input_path)

    a = [list(i.encode("utf-8")) for i in arr]

    count = dict()
    for i in arr:
        temp_num = len(i)
        if temp_num==1:
            continue
        for j in range(temp_num-1):
            if i[j:j+2] not in count:
                count[i[j:j+2]] = 1
            else:
                count[i[j:j+2]] += 1

    if len(count)<voclen:
        raise ValueError("vocab size too large")
    elif len(count)==voclen:
        return tuple(count,list(tuple()))
    while len(count) > voclen:
        temp = list(2)
        

    pass

if __name__ == "__main__":
    s = "TODO"
    special_tokens:list[str] = ["|end|","|start|"]
    bpe(s,special_tokens=special_tokens)