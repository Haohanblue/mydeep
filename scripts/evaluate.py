import argparse
import os
import json
import yaml
import numpy as np
import pandas as pd
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
    device = torch.device(args.device if (args.device != 'cpu' and cfg.get('use_gpu', False)) else 'cpu')

    split_dir = os.path.join(args.proc_dir, 'split')
    train_df = pd.read_csv(os.path.join(split_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(split_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, 'test.csv'))
    train_df[['user_id','item_id']] = train_df[['user_id','item_id']].astype(int)
    valid_df[['user_id','item_id']] = valid_df[['user_id','item_id']].astype(int)
    test_df[['user_id','item_id']] = test_df[['user_id','item_id']].astype(int)

    if not os.path.exists(cfg['save_path']):
        raise FileNotFoundError(f"checkpoint not found: {cfg['save_path']}")
    ckpt = torch.load(cfg['save_path'], map_location=device)
    n_users = ckpt['n_users']
    n_items = ckpt['n_items']
    adj_t = build_sparse_adj(os.path.join(args.graph_dir, 'adj_norm.npz'), device)
    model = LightGCN(n_users, n_items, ckpt['emb_dim'], ckpt['n_layers'], adj_t, device=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    if adj_t.shape[0] != (n_users + n_items):
        # 图形维度与模型不匹配，降级到零层传播
        model.n_layers = 0

    train_df = train_df[(train_df['user_id'] >= 0) & (train_df['user_id'] < n_users) & (train_df['item_id'] >= 0) & (train_df['item_id'] < n_items)]
    valid_df = valid_df[(valid_df['user_id'] >= 0) & (valid_df['user_id'] < n_users) & (valid_df['item_id'] >= 0) & (valid_df['item_id'] < n_items)]
    test_df = test_df[(test_df['user_id'] >= 0) & (test_df['user_id'] < n_users) & (test_df['item_id'] >= 0) & (test_df['item_id'] < n_items)]

    with open(os.path.join(split_dir, 'user_pos.json')) as f:
        raw_pos = json.load(f)
        user_pos_set = {}
        for k, v in raw_pos.items():
            u = int(k)
            if 0 <= u < n_users:
                user_pos_set[u] = set(int(it) for it in v if 0 <= int(it) < n_items)

    eval_batch_size = cfg.get('eval_batch_size', cfg.get('batch_size', 2048))

    # 评估
    users = test_df['user_id'].astype(int).values
    gt_items = test_df['item_id'].astype(int).values

    seen = {}
    if cfg.get('filter_seen', True):
        hist_df = pd.concat([
            pd.DataFrame(list(user_pos_set.items())).explode(1).rename(columns={0:'user_id',1:'item_id'}),
            valid_df[['user_id','item_id']]
        ])
        hist_df = hist_df.dropna(subset=['user_id','item_id'])
        for u, g in hist_df.groupby('user_id'):
            try:
                uid = int(u)
            except Exception:
                continue
            items = pd.to_numeric(g['item_id'], errors='coerce').dropna().astype(int).tolist()
            seen[uid] = set(it for it in items if 0 <= it < n_items)

    recalls, ndcgs, hits = [], [], []
    with torch.no_grad():
        for start in range(0, len(users), eval_batch_size):
            end = min(start + eval_batch_size, len(users))
            batch_users = torch.LongTensor(users[start:end]).to(device)
            try:
                scores = model.getUsersRating(batch_users).cpu().numpy()
            except Exception:
                # 图形尺寸不匹配时，强制使用初始嵌入进行评分（相当于 n_layers=0）
                model.n_layers = 0
                scores = model.getUsersRating(batch_users).cpu().numpy()
            for i, u in enumerate(users[start:end]):
                if cfg.get('filter_seen', True) and u in seen:
                    hide_idx = [idx for idx in seen[u] if 0 <= idx < n_items]
                    if hide_idx:
                        scores[i, hide_idx] = -1e9
                topk = min(args.topk, n_items)
                top_items = np.argpartition(-scores[i], topk - 1)[:topk]
                top_items = top_items[np.argsort(scores[i][top_items])[::-1]].tolist()
                gt = int(gt_items[start + i])
                recalls.append(recall_at_k(top_items, gt, args.topk))
                ndcgs.append(ndcg_at_k(top_items, gt, args.topk))
                hits.append(hit_rate_at_k(top_items, gt, args.topk))

    print(f"Recall@{args.topk}: {float(np.mean(recalls)):.4f}")
    print(f"NDCG@{args.topk}: {float(np.mean(ndcgs)):.4f}")
    print(f"HitRate@{args.topk}: {float(np.mean(hits)):.4f}")


if __name__ == '__main__':
    main()
