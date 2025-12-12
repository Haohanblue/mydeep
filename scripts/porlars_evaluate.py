import argparse
import os
import json
import yaml
import numpy as np
import polars as pl
from tqdm import tqdm
import torch
from scipy.sparse import load_npz

from model.lightgcn import LightGCN
from scripts.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k


def build_sparse_adj(adj_path, device):
    adj = load_npz(adj_path)
    adj = adj.tocoo()
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(adj.data)
    shape = adj.shape
    adj_t = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce().to(device)
    return adj_t


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved LightGCN model')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--proc_dir', type=str, default='data/processed')
    parser.add_argument('--graph_dir', type=str, default='data/graph')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    import torch as _torch
    device = _torch.device(args.device if (args.device != 'cpu' and cfg.get('use_gpu', False)) else 'cpu')

    split_dir = os.path.join(args.proc_dir, 'split')
    train_df = (
        pl.scan_csv(os.path.join(split_dir, 'train.csv'))
        .select([pl.col('user_id').cast(pl.Int32), pl.col('item_id').cast(pl.Int32)])
        .collect()
    )
    valid_df = (
        pl.scan_csv(os.path.join(split_dir, 'valid.csv'))
        .select([pl.col('user_id').cast(pl.Int32), pl.col('item_id').cast(pl.Int32)])
        .collect()
    )
    test_df = (
        pl.scan_csv(os.path.join(split_dir, 'test.csv'))
        .select([pl.col('user_id').cast(pl.Int32), pl.col('item_id').cast(pl.Int32)])
        .collect()
    )

    if not os.path.exists(cfg['save_path']):
        raise FileNotFoundError(f"checkpoint not found: {cfg['save_path']}")
    ckpt = _torch.load(cfg['save_path'], map_location=device)
    n_users = ckpt['n_users']
    n_items = ckpt['n_items']
    adj_t = build_sparse_adj(os.path.join(args.graph_dir, 'adj_norm.npz'), device)
    model = LightGCN(n_users, n_items, ckpt['emb_dim'], ckpt['n_layers'], adj_t, device=device)
    sd = ckpt['state_dict']
    for k in ['final_user_emb', 'final_item_emb']:
        if k in sd:
            sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    if adj_t.shape[0] != (n_users + n_items):
        # 图形维度与模型不匹配，降级到零层传播
        model.n_layers = 0

    train_df = train_df.filter(
        (pl.col('user_id') >= 0) & (pl.col('user_id') < n_users) & (pl.col('item_id') >= 0) & (pl.col('item_id') < n_items)
    )
    valid_df = valid_df.filter(
        (pl.col('user_id') >= 0) & (pl.col('user_id') < n_users) & (pl.col('item_id') >= 0) & (pl.col('item_id') < n_items)
    )
    test_df = test_df.filter(
        (pl.col('user_id') >= 0) & (pl.col('user_id') < n_users) & (pl.col('item_id') >= 0) & (pl.col('item_id') < n_items)
    )

    with open(os.path.join(split_dir, 'user_pos.json')) as f:
        raw_pos = json.load(f)
        user_pos_set = {}
        for k, v in raw_pos.items():
            u = int(k)
            if 0 <= u < n_users:
                user_pos_set[u] = set(int(it) for it in v if 0 <= int(it) < n_items)

    eval_batch_size = cfg.get('eval_batch_size', cfg.get('batch_size', 2048))

    # 评估
    users = test_df['user_id'].to_numpy()
    gt_items = test_df['item_id'].to_numpy()

    seen = {}
    if cfg.get('filter_seen', True):
        for u, items in user_pos_set.items():
            seen[u] = set(items)
        v_users = valid_df['user_id'].to_numpy()
        v_items = valid_df['item_id'].to_numpy()
        for u, it in zip(v_users, v_items):
            if 0 <= int(it) < n_items:
                seen.setdefault(int(u), set()).add(int(it))

    recalls, ndcgs, hits = [], [], []
    if cfg.get('eval_sample', False):
        candidates = int(cfg.get('eval_candidates', 2000))
        max_users = int(cfg.get('eval_users_limit', 10000))
        seed = int(cfg.get('eval_seed', 42))
        rng = np.random.default_rng(seed)
        import torch
        user_emb, item_emb = model.get_all_embeddings()
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)
        if max_users > 0 and max_users < len(users):
            sel = rng.choice(len(users), size=max_users, replace=False)
            users = users[sel]
            gt_items = gt_items[sel]
        with torch.no_grad():
            for i, u in tqdm(enumerate(users), total=len(users), desc='Eval', leave=True):
                gt = int(gt_items[i])
                cand = [gt]
                s = seen.get(int(u), set()) if cfg.get('filter_seen', True) else set()
                while len(cand) < candidates:
                    x = int(rng.integers(0, n_items))
                    if x == gt or x in s:
                        continue
                    cand.append(x)
                ue = user_emb[int(u)].unsqueeze(0)
                ie = item_emb[torch.as_tensor(cand, dtype=torch.long, device=device)]
                scores = (ue @ ie.t()).squeeze(0)
                _, idx = torch.topk(scores, args.topk, largest=True, sorted=True)
                top_items = torch.as_tensor(cand, dtype=torch.long, device=device)[idx].cpu().numpy()
                recalls.append(recall_at_k(top_items, gt, args.topk))
                ndcgs.append(ndcg_at_k(top_items, gt, args.topk))
                hits.append(hit_rate_at_k(top_items, gt, args.topk))
    else:
        import torch
        torch.set_num_threads(os.cpu_count())
        user_emb, item_emb = model.get_all_embeddings()
        user_emb = user_emb.half()
        item_emb = item_emb.half()
        item_emb_dev = item_emb.to(device)
        block = 2048
        with torch.no_grad():
            for start in tqdm(
                range(0, len(users), eval_batch_size),
                total=(len(users) + eval_batch_size - 1) // eval_batch_size,
                desc='Eval',
                leave=True,
            ):
                end = min(start + eval_batch_size, len(users))
                batch_users = users[start:end]
                scores = torch.empty(len(batch_users), n_items, dtype=torch.float16)
                for i in range(0, len(batch_users), block):
                    j = min(i+block, len(batch_users))
                    u = user_emb[batch_users[i:j]].to(device)
                    scores[i:j] = u @ item_emb_dev.T
                scores = scores.cpu().numpy().astype(np.float32)
                for i, u in enumerate(users[start:end]):
                    if cfg.get('filter_seen', True) and u in seen:
                        gt = int(gt_items[start + i])
                        hide_idx = [idx for idx in seen[u] if 0 <= idx < n_items and idx != gt]
                        if hide_idx:
                            scores[i, hide_idx] = -1e9
                    top_items = np.argpartition(-scores[i], args.topk-1)[:args.topk]
                    top_items = top_items[np.argsort(scores[i][top_items])[::-1]]
                    gt = int(gt_items[start + i])
                    recalls.append(recall_at_k(top_items, gt, args.topk))
                    ndcgs.append(ndcg_at_k(top_items, gt, args.topk))
                    hits.append(hit_rate_at_k(top_items, gt, args.topk))

    print(f"Recall@{args.topk}: {float(np.mean(recalls)):.4f}")
    print(f"NDCG@{args.topk}: {float(np.mean(ndcgs)):.4f}")
    print(f"HitRate@{args.topk}: {float(np.mean(hits)):.4f}")


if __name__ == '__main__':
    main()
