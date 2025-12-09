import numpy as np


def recall_at_k(ranked_items, gt_item, k=20):
    topk = ranked_items[:k]
    return 1.0 if gt_item in topk else 0.0


def hit_rate_at_k(ranked_items, gt_item, k=20):
    return recall_at_k(ranked_items, gt_item, k)


def ndcg_at_k(ranked_items, gt_item, k=20):
    topk = ranked_items[:k]
    if gt_item in topk:
        idx = int(np.where(topk == gt_item)[0][0])
        return 1.0 / np.log2(idx + 2)  # DCG for single relevant item
    return 0.0

