import argparse
import os
import json
import yaml
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from scipy.sparse import load_npz

from model.lightgcn import LightGCN
from scripts.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k

torch.set_num_threads(64)
torch.set_num_interop_threads(64)
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"


def build_sparse_adj(adj_path, device):
    adj = load_npz(adj_path).tocoo()
    i = torch.LongTensor(np.vstack((adj.row, adj.col)))
    v = torch.FloatTensor(adj.data)
    shape = adj.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce().to(device)


def augment_adj(adj_coo: torch.Tensor, aug_type: str, ratio: float):
    i = adj_coo.indices()
    v = adj_coo.values()
    nnz = v.shape[0]
    keep = max(1, int(nnz * (1.0 - ratio)))

    def edge_dropout():
        idx = torch.randperm(nnz)[:keep]
        return torch.sparse_coo_tensor(i[:, idx], v[idx], adj_coo.size()).coalesce()

    def node_dropout():
        n = adj_coo.size(0)
        drop = max(1, int(n * ratio))
        drop_nodes = set(torch.randperm(n)[:drop].cpu().numpy().tolist())
        mask = [(int(i[0, k].item()) not in drop_nodes) and (int(i[1, k].item()) not in drop_nodes) for k in range(nnz)]
        idx = torch.tensor([k for k, m in enumerate(mask) if m], dtype=torch.long)
        if idx.numel() == 0:
            return adj_coo
        return torch.sparse_coo_tensor(i[:, idx], v[idx], adj_coo.size()).coalesce()

    if aug_type == 'node':
        return node_dropout(), node_dropout()
    else:
        return edge_dropout(), edge_dropout()


def load_split_polars(path):
    return (
        pl.scan_csv(path)
        .select([pl.col('user_id').cast(pl.Int32), pl.col('item_id').cast(pl.Int32)])
        .collect()
        .to_pandas()
    )


def sample_negative_fast(user_pos_set, n_items, users):
    neg_items = np.random.randint(0, n_items, size=len(users))
    for i, u in enumerate(users):
        pos = user_pos_set[u]
        while int(neg_items[i]) in pos:
            neg_items[i] = np.random.randint(0, n_items)
    return neg_items.astype(np.int64)


def evaluate_fast(model, test_df, user_pos_set, topk, filter_seen, valid_df, device, eval_batch=4096):
    model.eval()
    n_items = model.n_items
    topk = min(topk, n_items)

    if filter_seen:
        seen = {}
        import pandas as pd
        hist = pd.concat([
            pd.DataFrame(list(user_pos_set.items())).explode(1).rename(columns={0: 'user_id', 1: 'item_id'}),
            valid_df[['user_id', 'item_id']],
        ])
        for u, g in hist.groupby('user_id'):
            seen[u] = set(g['item_id'].astype(int).tolist())
    else:
        seen = None

    users = test_df['user_id'].astype(int).values
    gt_items = test_df['item_id'].astype(int).values

    all_recall, all_ndcg, all_hit = [], [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(users), eval_batch), total=(len(users)+eval_batch-1)//eval_batch, desc='Eval', leave=True):
            end = min(start + eval_batch, len(users))
            batch_u = torch.as_tensor(users[start:end], dtype=torch.long, device=device)
            scores = model.getUsersRating(batch_u)  # [B, n_items]
            if filter_seen and seen:
                for i, u in enumerate(users[start:end]):
                    if u in seen:
                        gt_i = int(gt_items[start + i])
                        hide_idx = [idx for idx in seen[u] if 0 <= idx < n_items and idx != gt_i]
                        if hide_idx:
                            idx_t = torch.as_tensor(hide_idx, dtype=torch.long, device=device)
                            scores[i].index_fill_(0, idx_t, float('-inf'))
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
    users = test_df['user_id'].astype(int).values
    gt_items = test_df['item_id'].astype(int).values
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
        for i, u in tqdm(enumerate(users), total=len(users), desc='Eval(sampled)', leave=True):
            gt = int(gt_items[i])
            cand = [gt]
            seen = user_pos_set.get(u, set()) if filter_seen else set()
            while len(cand) < candidates:
                x = int(rng.integers(0, n_items))
                if x == gt or x in seen:
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


def main():
    parser = argparse.ArgumentParser(description='Train LightGCN + SGL (Polars)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--proc_dir', type=str, default='data/processed_sgl')
    parser.add_argument('--graph_dir', type=str, default='data/graph_sgl')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--ssl_ratio', type=float, default=None)
    parser.add_argument('--aug', type=str, default=None, choices=['edge','node','rw'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ssl_cfg = cfg.get('ssl', {})
    ratio = args.ssl_ratio if args.ssl_ratio is not None else ssl_cfg.get('ratio', 0.1)
    aug_type = args.aug if args.aug is not None else ssl_cfg.get('aug', 'edge')
    temperature = ssl_cfg.get('temperature', 0.2)

    device = torch.device(args.device if (args.device != 'cpu' and cfg.get('use_gpu', False)) else 'cpu')
    seed_train = int(cfg.get('seed', {}).get('train', 42))
    seed_eval = int(cfg.get('seed', {}).get('eval', 42))
    np.random.seed(seed_train)
    torch.manual_seed(seed_train)
    split_dir = os.path.join(args.proc_dir, 'split')
    map_dir = os.path.join(args.proc_dir, 'mapping')

    with open(os.path.join(map_dir, 'user2id.json')) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, 'item2id.json')) as f:
        item2id = json.load(f)
    n_users = len(user2id)
    n_items = len(item2id)

    # Polars 加载
    train_df = load_split_polars(os.path.join(split_dir, 'train.csv'))
    valid_df = load_split_polars(os.path.join(split_dir, 'valid.csv'))
    test_df = load_split_polars(os.path.join(split_dir, 'test.csv'))

    with open(os.path.join(split_dir, 'user_pos.json')) as f:
        user_pos_set = {int(k): set(v) for k, v in json.load(f).items()}

    adj_t = build_sparse_adj(os.path.join(args.graph_dir, 'adj_norm.npz'), device)
    model = LightGCN(n_users, n_items, cfg['embed_dim'], cfg['n_layers'], adj_t, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    batch_size = cfg['batch_size']
    eval_batch_size = cfg.get('eval_batch_size', batch_size)
    reg = float(cfg.get('reg', 1e-4))

    # 用户-正例
    user_to_pos = {}
    for u, it in train_df[['user_id','item_id']].itertuples(index=False):
        user_to_pos.setdefault(int(u), []).append(int(it))
    user_list = np.array(list(user_to_pos.keys()), dtype=np.int64)

    epochs = args.epochs if args.epochs is not None else int(cfg.get('epochs', 1))
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(user_list)
        losses = []
        view1, view2 = augment_adj(model.adj, aug_type=aug_type, ratio=ratio)
        mv1 = LightGCN(n_users, n_items, cfg['embed_dim'], cfg['n_layers'], view1.to(device), device=device)
        mv2 = LightGCN(n_users, n_items, cfg['embed_dim'], cfg['n_layers'], view2.to(device), device=device)
        mv1.user_emb.weight.data = model.user_emb.weight.data.clone()
        mv1.item_emb.weight.data = model.item_emb.weight.data.clone()
        mv2.user_emb.weight.data = model.user_emb.weight.data.clone()
        mv2.item_emb.weight.data = model.item_emb.weight.data.clone()
        z1_u, _ = mv1.computer()
        z2_u, _ = mv2.computer()
        z1_u = torch.nn.functional.normalize(z1_u, dim=1).detach()
        z2_u = torch.nn.functional.normalize(z2_u, dim=1).detach()
        for start in tqdm(range(0, len(perm), batch_size), total=(len(perm)+batch_size-1)//batch_size, desc=f'Epoch {epoch}', leave=True):
            end = min(start + batch_size, len(perm))
            batch_users = perm[start:end]
            pos_items = [np.random.choice(user_to_pos[u]) for u in batch_users]
            neg_items = sample_negative_fast(user_pos_set, n_items, batch_users)

            bu = torch.as_tensor(batch_users, dtype=torch.long, device=device)
            bi_pos = torch.as_tensor(pos_items, dtype=torch.long, device=device)
            bi_neg = torch.as_tensor(neg_items, dtype=torch.long, device=device)
            loss_bpr = model.bpr_loss(bu, bi_pos, bi_neg, reg=reg)
            idx = torch.as_tensor(batch_users, dtype=torch.long, device=device)
            z1 = z1_u[idx]
            z2 = z2_u[idx]
            sim = torch.matmul(z1.detach(), z2.detach().t()) / temperature
            labels = torch.arange(z1.size(0), device=device)
            loss_ssl = torch.nn.functional.cross_entropy(sim, labels)

            loss = loss_bpr + loss_ssl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.final_user_emb = None
        model.final_item_emb = None
        np.random.seed(seed_eval)
        torch.manual_seed(seed_eval)
        if cfg.get('eval_sample', False):
            rec, nd, hit = evaluate_sampled(
                model,
                test_df,
                user_pos_set,
                args.topk,
                cfg.get('filter_seen', True),
                valid_df,
                device,
                int(cfg.get('eval_candidates', 2000)),
                int(cfg.get('eval_users_limit', 10000)),
                int(cfg.get('eval_seed', 42)),
            )
        else:
            rec, nd, hit = evaluate_fast(
                model,
                test_df,
                user_pos_set,
                args.topk,
                cfg.get('filter_seen', True),
                valid_df,
                device,
                eval_batch_size,
            )
        print(f"Epoch {epoch} | Loss {np.mean(losses):.4f} | Recall@{args.topk} {rec:.4f} | NDCG@{args.topk} {nd:.4f} | HitRate@{args.topk} {hit:.4f}")

    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'n_users': n_users,
        'n_items': n_items,
        'emb_dim': cfg['embed_dim'],
        'n_layers': cfg['n_layers'],
    }, cfg['save_path'])
    print(f"[Saved] {cfg['save_path']}")


if __name__ == '__main__':
    main()
