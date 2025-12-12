import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def build_adj(train_csv: str, n_users: int, n_items: int) -> csr_matrix:
    df = pd.read_csv(train_csv)
    rows = df['user_id'].astype(int).values
    cols = df['item_id'].astype(int).values
    weights = df['edge_weight'].astype(float).values if 'edge_weight' in df.columns else np.ones_like(rows, dtype=np.float32)
    # CSR for bipartite user-item matrix
    mat = csr_matrix((weights, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    return mat


def sym_normalize_bipartite(mat_ui: csr_matrix) -> csr_matrix:
    # 构建对称归一化的用户-物品二部图邻接：A = [[0, UI],[UI^T, 0]]
    n_users, n_items = mat_ui.shape
    # degree for users
    du = np.array(mat_ui.sum(axis=1)).flatten()
    di = np.array(mat_ui.sum(axis=0)).flatten()
    du_inv_sqrt = np.power(du + 1e-12, -0.5)
    di_inv_sqrt = np.power(di + 1e-12, -0.5)
    D_u_inv = csr_matrix((du_inv_sqrt, (np.arange(n_users), np.arange(n_users))), shape=(n_users, n_users))
    D_i_inv = csr_matrix((di_inv_sqrt, (np.arange(n_items), np.arange(n_items))), shape=(n_items, n_items))
    mat_norm = D_u_inv @ mat_ui @ D_i_inv
    # 组装成稀疏邻接（用户+物品）
    zero_u = csr_matrix((n_users, n_users), dtype=np.float32)
    zero_i = csr_matrix((n_items, n_items), dtype=np.float32)
    top = csr_matrix(np.hstack([zero_u.toarray(), mat_norm.toarray()]))
    bottom = csr_matrix(np.hstack([mat_norm.T.toarray(), zero_i.toarray()]))
    adj = csr_matrix(np.vstack([top.toarray(), bottom.toarray()]))
    return adj


def main():
    parser = argparse.ArgumentParser(description='Build bipartite graph adjacency and symmetric normalization (CSR)')
    parser.add_argument('--proc_dir', type=str, default='data/processed')
    parser.add_argument('--out_dir', type=str, default='data/graph')
    parser.add_argument('--use_sparse', type=bool, default=True)
    args = parser.parse_args()

    split_dir = os.path.join(args.proc_dir, 'split')
    map_dir = os.path.join(args.proc_dir, 'mapping')
    with open(os.path.join(map_dir, 'user2id.json')) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, 'item2id.json')) as f:
        item2id = json.load(f)
    n_users = len(user2id)
    n_items = len(item2id)

    os.makedirs(args.out_dir, exist_ok=True)
    mat_ui = build_adj(os.path.join(split_dir, 'train.csv'), n_users, n_items)
    save_npz(os.path.join(args.out_dir, 'ui_train.npz'), mat_ui)
    adj = sym_normalize_bipartite(mat_ui)
    save_npz(os.path.join(args.out_dir, 'adj_norm.npz'), adj)
    print(f"[Graph] Saved UI and normalized adj to {args.out_dir}")


if __name__ == '__main__':
    main()
