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

class _PairItem:
    """堆元素: 频次最高优先; 频次相同时字典序最大的 pair 优先"""
    __slots__ = ("neg_count", "pair")

    def __init__(self, neg_count: int, pair: tuple[bytes, bytes]):
        self.neg_count = neg_count
        self.pair = pair

    def __lt__(self, other: "_PairItem") -> bool:
        if self.neg_count != other.neg_count:
            return self.neg_count < other.neg_count
        return self.pair > other.pair  # 同频次: 字典序大者视为"更小", 先出堆


def _pairs_of(word: tuple[bytes, ...]) -> Counter:
    """统计一个词内所有相邻 pair"""
    return Counter(zip(word, word[1:]))


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """在单个词内把 pair 替换成合并后的 token"""
    a, b = pair
    ab = a + b
    out: list[bytes] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(ab)
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
    # 1. 预分词
    s = pre_token(input_path,special_tokens)

    # 增量式更新
    word_freq = Counter(s)
    arr: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in w.encode("utf-8")): c
        for w, c in word_freq.items()
    }

    # 除特殊词外的目标词表大小
    voclen = vocab_size - len(special_tokens)

    # 词表: token(bytes) -> 出现次数
    # GPT 初始化: 初始词表固定包含全部 256 个单字节 token
    vocab_rev: dict[bytes, int] = {bytes([i]): 0 for i in range(256)}
    for word, c in arr.items():
        for tok in word:
            vocab_rev[tok] += c

    # 全局 pair 计数表 (增量维护)
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for word, c in arr.items():
        for p, cnt in _pairs_of(word).items():
            pair_counts[p] = pair_counts.get(p, 0) + cnt * c

    # 最大堆 (惰性失效)
    heap: list[_PairItem] = [_PairItem(-cnt, p) for p, cnt in pair_counts.items()]
    heapify(heap)

    merge: list[tuple[bytes, bytes]] = []
    now_voclen = len(vocab_rev)

    # 2. 循环直到词表达到目标大小
    while now_voclen < voclen:
        # 取出当前频次最高(并列时字典序最大)且计数仍有效的 pair
        maxa_pair = None
        while heap:
            item = heappop(heap)
            cur = pair_counts.get(item.pair, 0)
            if cur > 0 and item.neg_count == -cur:  # 校验: 条目未过期
                maxa_pair = item.pair
                break
        if maxa_pair is None:
            break

        merge.append(maxa_pair)
        a, b = maxa_pair
        ab = a + b

        # 3. 增量更新: 只改写包含该 pair 的词, 只修正受影响的 pair 计数
        for word, c in list(arr.items()):
            if a not in word or b not in word:  # 快速剪枝
                continue

            old_pairs = _pairs_of(word)
            n_merged = old_pairs[maxa_pair]
            new_word = _merge_word(word, maxa_pair)
            new_pairs = _pairs_of(new_word)

            for p in old_pairs.keys() | new_pairs.keys():
                delta = (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * c
                if delta == 0:
                    continue
                nc = pair_counts.get(p, 0) + delta
                if nc > 0:
                    pair_counts[p] = nc
                    # 计数无论增减都要重新入堆, 防止"饥饿"
                    # (旧的偏高/偏低条目靠弹出时校验自然淘汰)
                    heappush(heap, _PairItem(-nc, p))
                else:
                    pair_counts.pop(p, None)

            vocab_rev[ab] = vocab_rev.get(ab, 0) + n_merged * c
            vocab_rev[a] = vocab_rev.get(a, 0) - n_merged * c
            vocab_rev[b] = vocab_rev.get(b, 0) - n_merged * c

            del arr[word]
            arr[new_word] = arr.get(new_word, 0) + c

        now_voclen = len(vocab_rev)

    # 4. 翻转成 id -> bytes, 再加入特殊 token, 返回结果
    vocab: dict[int, bytes] = {idx: tok for idx, tok in enumerate(vocab_rev)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    return vocab, merge


if __name__ == "__main__":
    s = "TODO"
    special_tokens:list[str] = ["|end|","|start|"]
    bpe(s,special_tokens=special_tokens)