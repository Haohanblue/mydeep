import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.lightgcn import LightGCN

def main():
    n_users, n_items, emb_dim, n_layers = 5, 7, 8, 2
    # Build tiny bipartite adjacency (users first, items next)
    indices = []
    values = []
    for u in range(n_users):
        i = u % n_items
        indices.append([u, n_users + i])
        indices.append([n_users + i, u])
        values.extend([1.0, 1.0])
    idx = torch.tensor(indices, dtype=torch.long).t()
    val = torch.tensor(values, dtype=torch.float)
    adj = torch.sparse_coo_tensor(idx, val, (n_users + n_items, n_users + n_items)).coalesce()
    model = LightGCN(n_users, n_items, emb_dim, n_layers, adj, device='cpu')
    ue, ie = model.get_all_embeddings()
    print(ue.shape, ie.shape)

if __name__ == '__main__':
    main()
