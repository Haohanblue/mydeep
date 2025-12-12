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

    with open(os.path.join(split_dir, 'user_pos.json')) as f:
        user_pos_set = {int(k): set(v) for k, v in json.load(f).items()}

    ckpt = torch.load(cfg['save_path'], map_location=device)
    n_users = ckpt['n_users']
    n_items = ckpt['n_items']
    adj_t = build_sparse_adj(os.path.join(args.graph_dir, 'adj_norm.npz'), device)

    model = LightGCN(n_users, n_items, ckpt['emb_dim'], ckpt['n_layers'], adj_t, device=device)
    sd = ckpt['state_dict'].copy()
    sd.pop('final_user_emb', None)
    sd.pop('final_item_emb', None)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    eval_batch_size = cfg.get('eval_batch_size', cfg.get('batch_size', 2048))

    # 评估
    users = test_df['user_id'].astype(int).values
    gt_items = test_df['item_id'].astype(int).values

    seen = {}
    if cfg.get('filter_seen', True):
        for u, g in pd.concat([pd.DataFrame(list(user_pos_set.items())).explode(1).rename(columns={0:'user_id',1:'item_id'}), valid_df[['user_id','item_id']]]).groupby('user_id'):
            seen[u] = set(g['item_id'].astype(int).tolist())

    recalls, ndcgs, hits = [], [], []
    with torch.no_grad():
        for start in range(0, len(users), eval_batch_size):
            end = min(start + eval_batch_size, len(users))
            batch_users = torch.LongTensor(users[start:end]).to(device)
            scores = model.getUsersRating(batch_users).cpu().numpy()
            for i, u in enumerate(users[start:end]):
                gt = int(gt_items[start + i])
                if cfg.get('filter_seen', True) and u in seen:
                    hide = [idx for idx in seen[u] if 0 <= idx < n_items and idx != gt]
                    if hide:
                        scores[i, hide] = -1e9
                top_items = np.argpartition(scores[i], -args.topk)[-args.topk:]
                top_items = top_items[np.argsort(scores[i][top_items])[::-1]].tolist()
                recalls.append(recall_at_k(top_items, gt, args.topk))
                ndcgs.append(ndcg_at_k(top_items, gt, args.topk))
                hits.append(hit_rate_at_k(top_items, gt, args.topk))

    print(f"Recall@{args.topk}: {float(np.mean(recalls)):.4f}")
    print(f"NDCG@{args.topk}: {float(np.mean(ndcgs)):.4f}")
    print(f"HitRate@{args.topk}: {float(np.mean(hits)):.4f}")


if __name__ == '__main__':
    main()
