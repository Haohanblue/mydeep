import argparse
import os
import json
import yaml

import numpy as np
import polars as pl
import pandas as pd

import torch
from tqdm import tqdm
from scipy.sparse import load_npz
import torch
import os

# ✅ 让 PyTorch 吃满 CPU
torch.set_num_threads(64)
torch.set_num_interop_threads(64)

# ✅ 建议同时约束 BLAS / OMP
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
from model.lightgcn import LightGCN
from scripts.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k


# ==================================================
# Sparse adjacency loader
# ==================================================

def build_sparse_adj(adj_path, device):
    adj = load_npz(adj_path).tocoo()
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(adj.data)
    shape = adj.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce().to(device)


# ==================================================
# Vectorized negative sampling (much faster)
# ==================================================

def sample_negative_fast(user_pos_set, n_items, users):
    """
    Vectorized negative sampling:
      - First random draw
      - Only re-draw conflicts
    """
    neg_items = np.random.randint(0, n_items, size=len(users))
    for i, u in enumerate(users):
        pos = user_pos_set[u]
        while neg_items[i] in pos:
            neg_items[i] = np.random.randint(0, n_items)
    return neg_items.astype(np.int64)


# ==================================================
# Evaluation (unchanged logic, only formatting)
# ==================================================

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
    topk = min(topk, n_items)

    seen = None
    if filter_seen:
        seen = {}
        hist = pd.concat([
            pd.DataFrame(list(user_pos_set.items())).explode(1)
                .rename(columns={0: "user_id", 1: "item_id"}),
            valid_df[["user_id", "item_id"]],
        ])
        for u, g in hist.groupby("user_id"):
            seen[u] = set(g["item_id"].astype(int).tolist())

    users = test_df["user_id"].astype(int).values
    gt_items = test_df["item_id"].astype(int).values

    recalls, ndcgs, hits = [], [], []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(users), eval_batch_size),
            desc="Eval",
            total=(len(users) + eval_batch_size - 1) // eval_batch_size,
            leave=True,
        ):
            end = min(start + eval_batch_size, len(users))
            batch_users = torch.LongTensor(users[start:end]).to(device)

            scores = model.getUsersRating(batch_users).cpu().numpy()
            for i, u in enumerate(users[start:end]):
                gt = int(gt_items[start + i])
                if filter_seen and seen and u in seen:
                    hide_idx = [idx for idx in seen[u] if 0 <= idx < n_items and idx != gt]
                    if hide_idx:
                        scores[i, hide_idx] = -1e9

                top_items = np.argpartition(-scores[i], topk - 1)[:topk]
                top_items = top_items[np.argsort(scores[i][top_items])[::-1]]

                recalls.append(recall_at_k(top_items, gt, topk))
                ndcgs.append(ndcg_at_k(top_items, gt, topk))
                hits.append(hit_rate_at_k(top_items, gt, topk))

    return float(np.mean(recalls)), float(np.mean(ndcgs)), float(np.mean(hits))


def evaluate_fast(model, test_df, user_pos_set, topk, filter_seen, valid_df, device, eval_batch=1024):
    model.eval()
    n_items = model.n_items
    user_emb, item_emb = model.computer()
    user_emb = user_emb.to(device).half()
    item_emb = item_emb.to(device).half()
    topk = min(topk, n_items)

    # 构造 seen mask
    if filter_seen:
        seen = {}
        hist = pd.concat([
            pd.DataFrame(list(user_pos_set.items())).explode(1)
                .rename(columns={0: "user_id", 1: "item_id"}),
            valid_df[["user_id", "item_id"]],
        ])
        for u, g in hist.groupby("user_id"):
            seen[u] = set(g["item_id"].astype(int).tolist())
    else:
        seen = None

    users = test_df["user_id"].astype(int).values
    gt_items = test_df["item_id"].astype(int).values

    all_recall, all_ndcg, all_hit = [], [], []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(users), eval_batch),
            desc="Eval",
            total=(len(users) + eval_batch - 1) // eval_batch,
            leave=True,
        ):
            end = min(start + eval_batch, len(users))
            batch_u = torch.as_tensor(users[start:end], dtype=torch.long, device=device)
            scores = (user_emb[batch_u] @ item_emb.t())

            if filter_seen and seen:
                for i, u in enumerate(users[start:end]):
                    if u in seen:
                        gt_i = int(gt_items[start + i])
                        hide_idx = [idx for idx in seen[u] if 0 <= idx < n_items and idx != gt_i]
                        if hide_idx:
                            idx_t = torch.as_tensor(hide_idx, dtype=torch.long, device=device)
                            scores[i].index_fill_(0, idx_t, float("-inf"))

            top_scores, top_idx = torch.topk(scores, topk, dim=1, largest=True, sorted=True)
            gt_vec = torch.as_tensor(gt_items[start:end], dtype=torch.long, device=device).unsqueeze(1)
            match = top_idx.eq(gt_vec)
            hit = match.any(dim=1).float()
            pos = match.float().argmax(dim=1)
            ndcg = torch.where(hit > 0, 1.0 / torch.log2(pos + 2), torch.zeros_like(hit))

            all_hit.extend(hit.cpu().numpy().tolist())
            all_recall.extend(hit.cpu().numpy().tolist())
            all_ndcg.extend(ndcg.cpu().numpy().tolist())

    return float(np.mean(all_recall)), float(np.mean(all_ndcg)), float(np.mean(all_hit))


def evaluate_sampled(model, test_df, user_pos_set, topk, filter_seen, valid_df, device, candidates=2000, max_users=None, seed=None):
    model.eval()
    n_items = model.n_items
    topk = min(topk, n_items)
    users = test_df["user_id"].astype(int).values
    gt_items = test_df["item_id"].astype(int).values
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if max_users is not None and max_users > 0 and max_users < len(users):
        sel = rng.choice(len(users), size=max_users, replace=False)
        users = users[sel]
        gt_items = gt_items[sel]
    user_emb, item_emb = model.computer()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    recalls, ndcgs, hits = [], [], []
    with torch.no_grad():
        from tqdm import tqdm
        for i, u in tqdm(enumerate(users), total=len(users), desc="Eval(sampled)", leave=True):
            gt = int(gt_items[i])
            cand = [gt]
            if filter_seen:
                seen = user_pos_set.get(u, set())
            else:
                seen = set()
            while len(cand) < candidates:
                x = int(rng.integers(0, n_items))
                if x == gt:
                    continue
                if x in seen:
                    continue
                cand.append(x)
            ue = user_emb[u].unsqueeze(0)
            ie = item_emb[torch.as_tensor(cand, dtype=torch.long, device=device)]
            scores = (ue @ ie.t()).squeeze(0)
            _, idx = torch.topk(scores, topk, largest=True, sorted=True)
            top_items = torch.as_tensor(cand, dtype=torch.long, device=device)[idx].cpu().numpy()
            recalls.append(recall_at_k(top_items, gt, topk))
            ndcgs.append(ndcg_at_k(top_items, gt, topk))
            hits.append(hit_rate_at_k(top_items, gt, topk))
    return float(np.mean(recalls)), float(np.mean(ndcgs)), float(np.mean(hits))


# ==================================================
# Polars CSV loader
# ==================================================

def load_split_polars(path):
    """
    Fast multi-core CSV loading via Polars
    """
    return (
        pl.scan_csv(path)
        .select(
            [
                pl.col("user_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.Int32),
            ]
        )
        .collect()
        .to_pandas()
    )


# ==================================================
# Main training
# ==================================================

def main():
    parser = argparse.ArgumentParser(description="Train LightGCN baseline (BPR)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--proc_dir", type=str, default="data/processed")
    parser.add_argument("--graph_dir", type=str, default="data/graph")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    # ----------------------
    # Config & device
    # ----------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if (args.device != "cpu" and cfg.get("use_gpu", False)) else "cpu"
    )
    seed_train = int(cfg.get("seed", {}).get("train", 42))
    seed_eval = int(cfg.get("seed", {}).get("eval", 42))
    np.random.seed(seed_train)
    torch.manual_seed(seed_train)

    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 1))
    if epochs < 1:
        epochs = 1

    # ----------------------
    # Load metadata
    # ----------------------
    split_dir = os.path.join(args.proc_dir, "split")
    map_dir = os.path.join(args.proc_dir, "mapping")

    with open(os.path.join(map_dir, "user2id.json")) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, "item2id.json")) as f:
        item2id = json.load(f)

    n_users = len(user2id)
    n_items = len(item2id)

    # ----------------------
    # Load splits (Polars)
    # ----------------------
    print("[Load] Reading train via Polars", flush=True)
    train_df = load_split_polars(os.path.join(split_dir, "train.csv"))
    print("[Load] train loaded", flush=True)
    print("[Load] Reading valid via Polars", flush=True)
    valid_df = load_split_polars(os.path.join(split_dir, "valid.csv"))
    print("[Load] valid loaded", flush=True)
    print("[Load] Reading test via Polars", flush=True)
    test_df = load_split_polars(os.path.join(split_dir, "test.csv"))
    print("[Load] test loaded", flush=True)

    with open(os.path.join(split_dir, "user_pos.json")) as f:
        user_pos_set = {int(k): set(v) for k, v in json.load(f).items()}

    # ----------------------
    # Graph
    # ----------------------
    print("[Graph] building adj", flush=True)
    try:
        adj_t = build_sparse_adj(
            os.path.join(args.graph_dir, "adj_norm.npz"),
            device,
        )
        print("[Graph] adj ready", flush=True)
    except Exception as e:
        print(f"[GraphError] {e}", flush=True)
        raise

    model = LightGCN(
        n_users,
        n_items,
        cfg["embed_dim"],
        cfg["n_layers"],
        adj_t,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ----------------------
    # Build user -> pos items
    # ----------------------
    user_to_pos = {}
    for u, it in train_df[["user_id", "item_id"]].itertuples(index=False):
        user_to_pos.setdefault(int(u), []).append(int(it))
    # 仅保留至少2个正例的用户，避免BPR无法取正例或负例导致batch极小
    user_list = np.array([u for u, items in user_to_pos.items() if len(items) >= 2], dtype=np.int64)

    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    reg = float(cfg.get("reg", 1e-4))
    print("[Train] data ready", flush=True)

    # ----------------------
    # Training loop
    # ----------------------
    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}] start", flush=True)
        model.train()
        perm = np.random.permutation(user_list)
        losses = []

        for start in tqdm(
            range(0, len(perm), batch_size),
            desc=f"Epoch {epoch}",
            total=(len(perm) + batch_size - 1) // batch_size,
            leave=True,
        ):
            end = min(start + batch_size, len(perm))
            batch_users = perm[start:end]
            if len(batch_users) == 0:
                continue

            pos_items = [np.random.choice(user_to_pos[u]) for u in batch_users]
            neg_items = sample_negative_fast(
                user_pos_set,
                n_items,
                batch_users,
            )

            bu = torch.LongTensor(batch_users).to(device)
            bi_pos = torch.LongTensor(pos_items).to(device)
            bi_neg = torch.LongTensor(neg_items).to(device)

            loss = model.bpr_loss(bu, bi_pos, bi_neg, reg=reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"[Epoch {epoch}] train done", flush=True)
        if epoch % cfg.get("log_interval", 1) == 0:
            print("[Eval] start evaluation ...", flush=True)
            try:
                model.final_user_emb = None
                model.final_item_emb = None
                np.random.seed(seed_eval)
                torch.manual_seed(seed_eval)
                if cfg.get("eval_sample", False):
                    rec, nd, hit = evaluate_sampled(
                        model,
                        test_df,
                        user_pos_set,
                        args.topk,
                        cfg.get("filter_seen", True),
                        valid_df,
                        device,
                        int(cfg.get("eval_candidates", 2000)),
                        int(cfg.get("eval_users_limit", 10000)),
                        int(cfg.get("eval_seed", 42)),
                    )
                else:
                    rec, nd, hit = evaluate_fast(
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
                    f"Epoch {epoch} | "
                    f"Loss {np.mean(losses):.4f} | "
                    f"Recall@{args.topk} {rec:.4f} | "
                    f"NDCG@{args.topk} {nd:.4f} | "
                    f"HitRate@{args.topk} {hit:.4f}",
                    flush=True,
                )
            except Exception as e:
                print(f"[EvalError] Epoch {epoch} {e}", flush=True)
                import traceback
                traceback.print_exc()

    # ----------------------
    # Save model
    # ----------------------
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
    print(f"[Saved] {cfg['save_path']}", flush=True)


if __name__ == "__main__":
    main()
