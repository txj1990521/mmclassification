# zhanlan/ivfflat/zhanlan_retrieval/rank_fusion.py
import os
from collections import Counter


def clean_rank(rank_list):
    seen = set()
    out = []
    for x in rank_list:
        if x is None:
            continue
        x = int(x)
        if x < 0 or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def rank_to_rrf_score(rank_list, k=60):
    s = {}
    for r, i in enumerate(rank_list, 1):
        s[i] = 1.0 / (k + r)
    return s


def rrf_fuse(rankA, rankB, k=60):
    score = {}
    for r, i in enumerate(rankA, 1):
        score[i] = score.get(i, 0) + 1 / (k + r)
    for r, i in enumerate(rankB, 1):
        score[i] = score.get(i, 0) + 1 / (k + r)
    return [i for i, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


def global_confidence(global_rank, img_paths, topn=20):
    names = [os.path.basename(str(img_paths[i])) for i in global_rank[:topn]]
    prefix = [n.split('_')[0] if '_' in n else n[:4] for n in names]
    c = Counter(prefix).most_common(1)[0][1]
    return c / max(1, len(prefix))
