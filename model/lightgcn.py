import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers, adj_sparse, device='cpu'):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.device = device

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # 邻接矩阵（用户+物品）大小 (n_users + n_items, n_users + n_items)
        self.adj = adj_sparse  # torch.sparse_coo_tensor

    def computer(self):
        """K 层传播并聚合：最终表示为各层（含初始）加权和，这里采用平均或求和（简化）"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # (U+I, D)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            try:
                x = torch.sparse.mm(self.adj, x)
            except Exception:
                # 图形不匹配时降级为零层传播（使用初始嵌入）
                break
            embs.append(x)
        # LightGCN：通常对各层嵌入求和（或加权）；这里直接平均
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        user_out, item_out = torch.split(out, [self.n_users, self.n_items], dim=0)
        return user_out, item_out

    def get_all_embeddings(self):
        user_out, item_out = self.computer()
        return user_out.detach(), item_out.detach()

    def getUsersRating(self, users):
        user_out, item_out = self.computer()
        users_emb = user_out[users]  # (B, D)
        scores = torch.matmul(users_emb, item_out.t())  # (B, n_items)
        return scores

    def bpr_loss(self, users, pos_items, neg_items, reg=1e-4):
        user_out, item_out = self.computer()
        u = user_out[users]
        pos = item_out[pos_items]
        neg = item_out[neg_items]
        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        try:
            reg_term = (u.pow(2).sum() + pos.pow(2).sum() + neg.pow(2).sum())
            batch_size = float(users.size(0))
            reg_loss = reg * reg_term / batch_size
            return loss + reg_loss
        except Exception as e:
            print('DEBUG bpr_loss error:', e)
            print('types:', type(users), type(pos), type(neg), type(reg))
            print('sizes:', getattr(users,'size',lambda:None)(), getattr(pos,'size',lambda:None)(), getattr(neg,'size',lambda:None)())
            raise
