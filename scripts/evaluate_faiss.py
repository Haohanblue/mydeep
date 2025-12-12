import argparse, os, json, yaml, numpy as np, pandas as pd, torch
from scipy.sparse import load_npz
from model.lightgcn import LightGCN
from scripts.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k
import faiss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--proc_dir', type=str, default='data/processed')
    parser.add_argument('--graph_dir', type=str, default='data/graph')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)

    split_dir = os.path.join(args.proc_dir, 'split')
    test_df = pd.read_csv(os.path.join(split_dir, 'test.csv'))
    test_df[['user_id','item_id']] = test_df[['user_id','item_id']].astype(int)

    ckpt = torch.load(cfg['save_path'], map_location=device)
    n_users, n_items = ckpt['n_users'], ckpt['n_items']
    adj = load_npz(os.path.join(args.graph_dir, 'adj_norm.npz'))
    model = LightGCN(n_users, n_items, ckpt['emb_dim'], ckpt['n_layers'], torch.sparse_coo_tensor(*map(torch.from_numpy, (adj.tocoo().row, adj.tocoo().col, adj.data)), torch.Size(adj.shape)).coalesce().to(device), device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    user_emb, item_emb = model.get_all_embeddings()
    user_emb, item_emb = user_emb.numpy(), item_emb.numpy()

    # Faiss IVF1024
    quantizer = faiss.IndexFlatIP(item_emb.shape[1])
    index = faiss.IndexIVFFlat(quantizer, item_emb.shape[1], 1024)
    index.train(item_emb)
    index.add(item_emb)
    index.nprobe = 16

    users = test_df['user_id'].values
    gt_items = test_df['item_id'].values
    recalls, ndcgs, hits = [], [], []
    batch = 10000
    for start in range(0, len(users), batch):
        end = min(start+batch, len(users))
        u = user_emb[users[start:end]]
        scores, I = index.search(u, args.topk)  # I: topk item id
        for i, (topk_idx, gt) in enumerate(zip(I, gt_items[start:end])):
            topk_idx = topk_idx.tolist()
            recalls.append(recall_at_k(topk_idx, gt, args.topk))
            ndcgs.append(ndcg_at_k(topk_idx, gt, args.topk))
            hits.append(hit_rate_at_k(topk_idx, gt, args.topk))
    print(f"Recall@{args.topk}: {np.mean(recalls):.4f}")
    print(f"NDCG@{args.topk}: {np.mean(ndcgs):.4f}")
    print(f"HitRate@{args.topk}: {np.mean(hits):.4f}")

if __name__ == '__main__':
    main()