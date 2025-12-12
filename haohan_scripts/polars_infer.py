import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import load_npz
from model.lightgcn import LightGCN

def build_sparse_adj(path, device):
    adj = load_npz(path).tocoo()
    i = torch.LongTensor(np.vstack((adj.row, adj.col)))
    v = torch.FloatTensor(adj.data)
    shape = adj.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce().to(device)

def load_split_polars(path):
    import polars as pl
    return (
        pl.scan_csv(path)
        .select([pl.col("user_id").cast(pl.Int32), pl.col("item_id").cast(pl.Int32)])
        .collect()
        .to_pandas()
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lightgcn.pt")
    parser.add_argument("--proc_dir", type=str, default="data/processed")
    parser.add_argument("--graph_dir", type=str, default="data/graph")
    parser.add_argument("--output_dir", type=str, default="data/predictions")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--users_source", type=str, choices=["train","valid","test","all"], default="test")
    parser.add_argument("--limit_users", type=int, default=10000)
    parser.add_argument("--filter_seen", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    split_dir = os.path.join(args.proc_dir, "split")
    map_dir = os.path.join(args.proc_dir, "mapping")

    with open(os.path.join(map_dir, "user2id.json")) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, "item2id.json")) as f:
        item2id = json.load(f)
    inv_user = {int(v): k for k, v in user2id.items()}
    inv_item = {int(v): k for k, v in item2id.items()}

    n_users = len(user2id)
    n_items = len(item2id)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "recommendations.json")

    train_df = load_split_polars(os.path.join(split_dir, "train.csv"))
    valid_df = load_split_polars(os.path.join(split_dir, "valid.csv"))
    test_df = load_split_polars(os.path.join(split_dir, "test.csv"))

    if args.users_source == "train":
        users = train_df["user_id"].unique()
    elif args.users_source == "valid":
        users = valid_df["user_id"].unique()
    elif args.users_source == "test":
        users = test_df["user_id"].unique()
    else:
        users = pd.concat([train_df["user_id"], valid_df["user_id"], test_df["user_id"]]).unique()

    if args.limit_users and args.limit_users > 0 and args.limit_users < len(users):
        sel = np.random.choice(len(users), size=args.limit_users, replace=False)
        users = users[sel]

    seen = None
    if args.filter_seen:
        seen = {}
        for u, it in train_df.itertuples(index=False):
            seen.setdefault(int(u), set()).add(int(it))
        for u, it in valid_df.itertuples(index=False):
            seen.setdefault(int(u), set()).add(int(it))

    adj_t = build_sparse_adj(os.path.join(args.graph_dir, "adj_norm.npz"), device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LightGCN(
        ckpt["n_users"],
        ckpt["n_items"],
        ckpt["emb_dim"],
        ckpt["n_layers"],
        adj_t,
        device=device,
    ).to(device)
    sd = ckpt["state_dict"]
    for k in ["final_user_emb", "final_item_emb"]:
        if k in sd:
            sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.eval()

    res = {}
    batch = 2048
    with torch.no_grad():
        for start in tqdm(range(0, len(users), batch), total=(len(users)+batch-1)//batch, desc="Infer"):
            end = min(start + batch, len(users))
            batch_users = torch.as_tensor(users[start:end], dtype=torch.long, device=device)
            scores = model.getUsersRating(batch_users)
            if args.filter_seen and seen:
                for i, u in enumerate(users[start:end]):
                    if u in seen:
                        hide = [idx for idx in seen[u] if 0 <= idx < n_items]
                        if hide:
                            idx_t = torch.as_tensor(hide, dtype=torch.long, device=device)
                            scores[i].index_fill_(0, idx_t, float("-inf"))
            top_scores, top_idx = torch.topk(scores, args.topk, dim=1, largest=True, sorted=True)
            for i, u in enumerate(users[start:end]):
                items = top_idx[i].cpu().numpy().tolist()
                res[inv_user[int(u)]] = [inv_item[int(x)] for x in items]

    with open(out_path, "w") as f:
        json.dump(res, f)
    print(out_path)

if __name__ == "__main__":
    main()
