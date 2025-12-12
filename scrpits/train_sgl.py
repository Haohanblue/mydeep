import argparse
import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import load_npz

from model.lightgcn import LightGCN
from scripts.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k


def build_sparse_adj(adj_path, device):
    from scipy.sparse import load_npz
    adj = load_npz(adj_path)
    adj = adj.tocoo()
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(adj.data)
    shape = adj.shape
    adj_t = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce().to(device)
    return adj_t


def augment_adj(adj_coo, aug_type: str, ratio: float):
    """
    简化版 SGL 视图增强：
    - edge: 随机丢弃一定比例的边
    - node: 随机丢弃一定比例的节点（对应行列清零）
    - rw: 简化为边丢弃（占位）
    返回两个视图的稀疏邻接（torch.sparse）
    """
    import numpy as np
    import torch
    i = adj_coo.indices()
    v = adj_coo.values()
    nnz = v.shape[0]
    keep = int(nnz * (1.0 - ratio))

    def edge_dropout():
        perm = torch.randperm(nnz)
        idx = perm[:keep]
        return torch.sparse_coo_tensor(i[:, idx], v[idx], adj_coo.size()).coalesce()

    def node_dropout():
        n = adj_coo.size(0)
        drop = int(n * ratio)
        drop_nodes = set(np.random.choice(n, drop, replace=False).tolist())
        mask = [(int(i[0, k].item()) not in drop_nodes) and (int(i[1, k].item()) not in drop_nodes) for k in range(nnz)]
        idx = torch.tensor([k for k, m in enumerate(mask) if m], dtype=torch.long)
        return torch.sparse_coo_tensor(i[:, idx], v[idx], adj_coo.size()).coalesce()

    if aug_type == 'edge':
        view1 = edge_dropout()
        view2 = edge_dropout()
    elif aug_type == 'node':
        view1 = node_dropout()
        view2 = node_dropout()
    else:  # rw 简化为 edge dropout
        view1 = edge_dropout()
        view2 = edge_dropout()
    return view1, view2


def contrastive_loss(z1_u, z2_u, temperature=0.2):
    """InfoNCE 风格：同一用户在两视图的表示为正样本，不同用户为负样本"""
    z1 = torch.nn.functional.normalize(z1_u, dim=1)
    z2 = torch.nn.functional.normalize(z2_u, dim=1)
    sim = torch.matmul(z1, z2.t()) / temperature  # (U, U)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss


def evaluate(model: LightGCN, test_df: pd.DataFrame, user_pos_set: dict, topk: int, filter_seen: bool, valid_df: pd.DataFrame, device: str, eval_batch_size: int):
    model.eval()
    n_items = model.n_items
    seen = None
    if filter_seen:
        seen = {}
        for u, g in pd.concat([pd.DataFrame(list(user_pos_set.items())).explode(1).rename(columns={0:'user_id',1:'item_id'}), valid_df[['user_id','item_id']]]).groupby('user_id'):
            seen[u] = set(g['item_id'].astype(int).tolist())
    users = test_df['user_id'].astype(int).values
    gt_items = test_df['item_id'].astype(int).values
    recalls, ndcgs, hits = [], [], []
    with torch.no_grad():
        for start in range(0, len(users), eval_batch_size):
            end = min(start + eval_batch_size, len(users))
            batch_users = torch.LongTensor(users[start:end]).to(device)
            scores = model.getUsersRating(batch_users).cpu().numpy()
            for i, u in enumerate(users[start:end]):
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
    parser = argparse.ArgumentParser(description='Train LightGCN + SGL (self-supervised)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--proc_dir', type=str, default='data/processed')
    parser.add_argument('--graph_dir', type=str, default='data/graph')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--ssl_ratio', type=float, default=None)
    parser.add_argument('--aug', type=str, default=None, choices=['edge','node','rw'])
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ssl_cfg = cfg.get('ssl', {})
    ratio = args.ssl_ratio if args.ssl_ratio is not None else ssl_cfg.get('ratio', 0.1)
    aug_type = args.aug if args.aug is not None else ssl_cfg.get('aug', 'edge')
    temperature = ssl_cfg.get('temperature', 0.2)

    device = torch.device(args.device if (args.device != 'cpu' and cfg.get('use_gpu', False)) else 'cpu')
    split_dir = os.path.join(args.proc_dir, 'split')
    map_dir = os.path.join(args.proc_dir, 'mapping')
    with open(os.path.join(map_dir, 'user2id.json')) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, 'item2id.json')) as f:
        item2id = json.load(f)
    n_users = len(user2id)
    n_items = len(item2id)

    train_df = pd.read_csv(os.path.join(split_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(split_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, 'test.csv'))
    with open(os.path.join(split_dir, 'user_pos.json')) as f:
        user_pos_set = {int(k): set(v) for k, v in json.load(f).items()}

    adj_t = build_sparse_adj(os.path.join(args.graph_dir, 'adj_norm.npz'), device)
    model = LightGCN(n_users, n_items, cfg['embed_dim'], cfg['n_layers'], adj_t, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    batch_size = cfg['batch_size']
    eval_batch_size = cfg.get('eval_batch_size', batch_size)
    reg = float(cfg.get('reg', 1e-4))

    # 构造用户-正例索引
    user_to_pos = {}
    for u, it in train_df[['user_id','item_id']].itertuples(index=False):
        user_to_pos.setdefault(int(u), []).append(int(it))
    user_list = list(user_to_pos.keys())

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = np.random.permutation(user_list)
        losses = []
        for start in range(0, len(perm), batch_size):
            end = min(start + batch_size, len(perm))
            batch_users = perm[start:end]
            pos_sample = [np.random.choice(user_to_pos[u]) for u in batch_users]
            # 负采样
            def sample_negative(users):
                neg_items = []
                for u in users:
                    pos = user_pos_set[u]
                    while True:
                        ni = np.random.randint(0, n_items)
                        if ni not in pos:
                            neg_items.append(ni)
                            break
                return np.array(neg_items, dtype=np.int64)
            neg_sample = sample_negative(batch_users)

            bu = torch.LongTensor(batch_users).to(device)
            bi_pos = torch.LongTensor(pos_sample).to(device)
            bi_neg = torch.LongTensor(neg_sample).to(device)
            loss_bpr = model.bpr_loss(bu, bi_pos, bi_neg, reg=reg)

            # SGL 对比损失：基于两视图 user 表示
            view1, view2 = augment_adj(model.adj, aug_type=aug_type, ratio=ratio)
            model_view1 = LightGCN(n_users, n_items, cfg['embed_dim'], cfg['n_layers'], view1.to(device), device=device)
            model_view2 = LightGCN(n_users, n_items, cfg['embed_dim'], cfg['n_layers'], view2.to(device), device=device)
            # 共享底层嵌入初始化（简化）：拷贝当前模型嵌入权重
            model_view1.user_emb.weight.data = model.user_emb.weight.data.clone()
            model_view1.item_emb.weight.data = model.item_emb.weight.data.clone()
            model_view2.user_emb.weight.data = model.user_emb.weight.data.clone()
            model_view2.item_emb.weight.data = model.item_emb.weight.data.clone()
            z1_u, _ = model_view1.computer()
            z2_u, _ = model_view2.computer()
            loss_ssl = contrastive_loss(z1_u, z2_u, temperature=temperature)

            loss = loss_bpr + loss_ssl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        rec, nd, hit = evaluate(model, test_df, user_pos_set, args.topk, cfg.get('filter_seen', True), valid_df, device, eval_batch_size)
        print(f"Epoch {epoch} | Loss {np.mean(losses):.4f} | Recall@{args.topk} {rec:.4f} | NDCG@{args.topk} {nd:.4f} | HitRate@{args.topk} {hit:.4f}")

    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'n_users': n_users, 'n_items': n_items, 'emb_dim': cfg['embed_dim'], 'n_layers': cfg['n_layers']}, cfg['save_path'])
    print(f"[Saved] {cfg['save_path']}")


if __name__ == '__main__':
    main()
