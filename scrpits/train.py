#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import yaml
import numpy as np
import pandas as pd

# ======= CPU 多核设置：必须在 import torch 之前 =======
# 如果外部已经设置了这些环境变量，这里不会覆盖（用 setdefault）
os.environ.setdefault("OMP_NUM_THREADS", "64")
os.environ.setdefault("MKL_NUM_THREADS", "64")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")

import torch
from tqdm import tqdm
from scipy.sparse import load_npz

from model.lightgcn import LightGCN
from scripts.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k

# PyTorch 线程数（和上面环境变量一致）
try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "64")))
    torch.set_num_interop_threads(int(os.environ.get("OMP_NUM_THREADS", "64")))
except Exception:
    # 某些环境下可能不支持 set_num_interop_threads，忽略即可
    pass


def build_sparse_adj(adj_path, device):
    adj = load_npz(adj_path)
    adj = adj.tocoo()
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(adj.data)
    shape = adj.shape
    adj_t = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce().to(device)
    return adj_t


# ========= 负采样：轻量 vectorized 版本 =========
def sample_negative(user_pos_set, n_items, users):
    """
    users: 一批用户 ID（list 或 ndarray）
    返回: 每个用户一个负 item，ndarray[int64]
    实现思路：
      1) 先一次性随机采一个 neg_items 数组
      2) 对于落在正例集合里的位置，再局部 while 重采样
    比原版“每个用户 while”要快不少
    """
    users = np.asarray(users, dtype=np.int64)
    neg_items = np.random.randint(0, n_items, size=len(users), dtype=np.int64)

    for idx, u in enumerate(users):
        pos = user_pos_set[u]
        # 一般 item 空间很大，这里冲突概率低，while 循环很快
        while neg_items[idx] in pos:
            neg_items[idx] = np.random.randint(0, n_items)

    return neg_items


def evaluate(
    model: LightGCN,
    test_df: pd.DataFrame,
    user_pos_set: dict,
    topk: int,
    filter_seen: bool,
    valid_df: pd.DataFrame,
    device: str,
    eval_batch_size: int,
):
    model.eval()
    n_items = model.n_items

    # 构造过滤集合（训练正例 + 验证集）
    seen = None
    if filter_seen:
        seen = {}
        hist = pd.concat(
            [
                pd.DataFrame(list(user_pos_set.items()))
                .explode(1)
                .rename(columns={0: "user_id", 1: "item_id"}),
                valid_df[["user_id", "item_id"]],
            ]
        )
        for u, g in hist.groupby("user_id"):
            seen[u] = set(g["item_id"].astype(int).tolist())

    users = test_df["user_id"].astype(int).values
    gt_items = test_df["item_id"].astype(int).values

    recalls, ndcgs, hits = [], [], []

    with torch.no_grad():
        for start in range(0, len(users), eval_batch_size):
            end = min(start + eval_batch_size, len(users))
            batch_users = torch.LongTensor(users[start:end]).to(device)

            scores = model.getUsersRating(batch_users).cpu().numpy()  # (B, n_items)

            for i, u in enumerate(users[start:end]):
                # 过滤已交互
                if filter_seen and seen and u in seen:
                    scores[i, list(seen[u])] = -1e9

                top_items = np.argpartition(scores[i], -topk)[-topk:]
                top_items = top_items[np.argsort(scores[i][top_items])[::-1]].tolist()

                gt = int(gt_items[start + i])
                recalls.append(recall_at_k(top_items, gt, topk))
                ndcgs.append(ndcg_at_k(top_items, gt, topk))
                hits.append(hit_rate_at_k(top_items, gt, topk))

    return float(np.mean(recalls)), float(np.mean(ndcgs)), float(np.mean(hits))


def main():
    parser = argparse.ArgumentParser(description="Train LightGCN baseline (BPR)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--proc_dir", type=str, default="data/processed")
    parser.add_argument("--graph_dir", type=str, default="data/graph")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if (args.device != "cpu" and cfg.get("use_gpu", False)) else "cpu"
    )

    split_dir = os.path.join(args.proc_dir, "split")
    map_dir = os.path.join(args.proc_dir, "mapping")

    with open(os.path.join(map_dir, "user2id.json")) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, "item2id.json")) as f:
        item2id = json.load(f)

    n_users = len(user2id)
    n_items = len(item2id)

    # ========= Data loading（保持 pandas，不动输出结构） =========
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(split_dir, "valid.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))

    with open(os.path.join(split_dir, "user_pos.json")) as f:
        user_pos_set = {int(k): set(v) for k, v in json.load(f).items()}

    # ========= Graph & model =========
    adj_t = build_sparse_adj(os.path.join(args.graph_dir, "adj_norm.npz"), device)
    model = LightGCN(
        n_users,
        n_items,
        cfg["embed_dim"],
        cfg["n_layers"],
        adj_t,
        device=device,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # 构造 user -> 正例 item 列表
    user_to_pos = {}
    for u, it in train_df[["user_id", "item_id"]].itertuples(index=False):
        user_to_pos.setdefault(int(u), []).append(int(it))
    user_list = list(user_to_pos.keys())

    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    reg = float(cfg.get("reg", 1e-4))

    # ========= Training loop（加入 tqdm 进度条） =========
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = np.random.permutation(user_list)
        losses = []

        num_batches = (len(perm) + batch_size - 1) // batch_size

        for start in tqdm(
            range(0, len(perm), batch_size),
            desc=f"Epoch {epoch}",
            total=num_batches,
            ncols=100,
        ):
            end = min(start + batch_size, len(perm))
            batch_users = perm[start:end]

            # 随机正例采样（保持原逻辑）
            pos_sample = [
                np.random.choice(user_to_pos[u]) for u in batch_users
            ]
            # 向量化负采样
            neg_sample = sample_negative(user_pos_set, n_items, batch_users)

            bu = torch.LongTensor(batch_users).to(device)
            bi_pos = torch.LongTensor(pos_sample).to(device)
            bi_neg = torch.LongTensor(neg_sample).to(device)

            loss = model.bpr_loss(bu, bi_pos, bi_neg, reg=reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch % cfg.get("log_interval", 1) == 0:
            rec, nd, hit = evaluate(
                model,
                test_df,
                user_pos_set,
                args.topk,
                cfg.get("filter_seen", True),
                valid_df,
                device,
                eval_batch_size,
            )
            print(
                f"Epoch {epoch} | Loss {np.mean(losses):.4f} | "
                f"Recall@{args.topk} {rec:.4f} | "
                f"NDCG@{args.topk} {nd:.4f} | "
                f"HitRate@{args.topk} {hit:.4f}"
            )

    # ========= 保存模型（保持原输出结构） =========
    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "emb_dim": cfg["embed_dim"],
            "n_layers": cfg["n_layers"],
        },
        cfg["save_path"],
    )
    print(f"[Saved] {cfg['save_path']}")


if __name__ == "__main__":
    main()
