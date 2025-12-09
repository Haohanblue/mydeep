
## 数据集介绍

本项目采用的数据集是来自阿里巴巴天池的 **Taobao UserBehavior**，这是一个广泛应用于推荐系统研究的公开数据集。

- **数据字段**：数据集包含 `user_id`（用户 ID）、`item_id`（商品 ID）、`category_id`（商品类别 ID）、`behavior_type`（用户行为类型）和 `timestamp`（行为发生的时间戳）。
- **时间周期**：数据采集自 2017 年 11 月 25 日至 2017 年 12 月 3 日，覆盖了“双十二”大促前的用户行为。
- **数据规模**：完整数据集包含约 1 亿条用户行为记录，涵盖了约 100 万用户和 400 万商品。
- **场景适配性**：该数据集源于真实的中国电商场景，其用户行为模式（如点击、加购、收藏、购买）和数据分布（如用户兴趣的稀疏性、行为的长尾分布）都非常贴近国内互联网应用的实际情况，是检验和优化推荐召回算法的理想选择。

## 召回模型原理解释

### 协同过滤（CF）与二部图

协同过滤（Collaborative Filtering, CF）是推荐系统的基石思想，其核心假设是“物以类聚，人以群分”。在召回阶段，我们通常将其建模为一个**用户—物品二部图（User-Item Bipartite Graph）**。

- **图的构建**：图的一侧是所有用户节点，另一侧是所有物品节点。当用户与物品发生交互（如点击、购买等）时，就在对应的用户和物品节点间连接一条边。
- **召回的本质**：召回问题就转化为在图上进行“链接预测”（Link Prediction）——预测哪些尚未连接的用户和物品节点对最有可能产生连接。

### LightGCN：简化的图卷积网络

LightGCN 是在图卷积网络（GCN）基础上提出的一种简化模型，专门用于推荐系统。它摒弃了 GCN 中非线性的激活函数和特征变换，认为这两者对于协同过滤任务并非核心，有时甚至会带来不必要的复杂性和训练难度。

- **邻居聚合**：LightGCN 的核心在于**邻居聚合**。每个节点（无论是用户还是物品）的嵌入表示，都是通过聚合其邻居节点的嵌入来更新的。经过 K 层传播，一个节点的表示能够融合其 K 跳邻居的信息。
- **层间加权**：最终，一个节点（例如一个用户）的最终嵌入表示，是其在每一层（从第 0 层的初始嵌入到第 K 层的最终嵌入）的嵌入表示的**加权和**（通常是平均或直接求和）。这综合了不同距离邻居的信号。
- **BPR Loss**：训练阶段采用贝叶斯个性化排序损失（Bayesian Personalized Ranking, BPR Loss）。对于每个用户，模型的目标是使其交互过的**正样本**（用户喜欢的物品）的预测得分高于未交互过的**负样本**。

### SGL：自监督图学习增强

自监督学习（Self-supervised Learning, SGL）为图学习引入了一种新的范式，旨在解决数据稀疏性问题，提升模型鲁棒性。

- **视图增强**：SGL 的核心是**数据增强**。它通过对原始的二部图进行随机扰动（如随机丢弃部分节点或边），创造出两个或多个略有不同的**视图（View）**。
- **对比学习**：模型的目标是让同一个节点（例如同一个用户）在不同视图中的表示尽可能**相似**（拉近正样本），而与其他不同节点的表示尽可能**不同**（推开负样本）。这种对比损失（InfoNCE Loss）迫使模型学习到对噪声和数据稀疏不那么敏感的、更本质的节点表示。

## 代码建模说明

模型的核心逻辑在 `model/lightgcn.py` 中实现。

### 核心传播逻辑 `computer()`

该函数实现了 LightGCN 的 K 层邻居聚合与最终嵌入的生成。

```python
def computer(self):
    """K 层传播并聚合：最终表示为各层（含初始）加权和，这里采用平均或求和（简化）"""
    all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # (U+I, D)
    embs = [all_emb]
    x = all_emb
    for _ in range(self.n_layers):
        x = torch.sparse.mm(self.adj, x)
        embs.append(x)
    # LightGCN：通常对各层嵌入求和（或加权）；这里直接平均
    out = torch.mean(torch.stack(embs, dim=0), dim=0)
    user_out, item_out = torch.split(out, [self.n_users, self.n_items], dim=0)
    return user_out, item_out
```

- `all_emb`：将用户和物品的初始嵌入拼接在一起。
- `torch.sparse.mm(self.adj, x)`：利用稀疏矩阵乘法高效地执行邻居聚合。
- `torch.mean(torch.stack(embs, dim=0), dim=0)`：将各层的嵌入进行平均，得到最终的用户和物品表示。

### 评分函数 `getUsersRating()`

该函数利用学习到的用户和物品嵌入，计算用户对所有物品的偏好得分。

```python
def getUsersRating(self, users):
    user_out, item_out = self.computer()
    users_emb = user_out[users]  # (B, D)
    scores = torch.matmul(users_emb, item_out.t())  # (B, n_items)
    return scores
```

- `torch.matmul(users_emb, item_out.t())`：通过内积（dot product）计算用户嵌入和所有物品嵌入的相似度，作为推荐分数。

### 损失函数 `bpr_loss()`

该函数实现了 BPR 损失，用于驱动模型学习。

```python
def bpr_loss(self, users, pos_items, neg_items, reg=1e-4):
    user_out, item_out = self.computer()
    u = user_out[users]
    pos = item_out[pos_items]
    neg = item_out[neg_items]
    pos_scores = torch.sum(u * pos, dim=1)
    neg_scores = torch.sum(u * neg, dim=1)
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    # L2 正则
    reg_loss = reg * (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / users.shape[0]
    return loss + reg_loss
```

- `F.logsigmoid(pos_scores - neg_scores)`：核心的 BPR 优化目标，希望正样本得分远大于负样本得分。
- `reg_loss`：L2 正则化项，防止模型过拟合。

## 改进迭代与结果记录

在本地小规模样本（5000 条交互）上进行了多轮改进迭代，目标是超越 Baseline 指标。

- **Baseline 指标**（`lightgcn.yaml`, epochs=10）：
  - Recall@20: 0.0850
  - NDCG@20: 0.0293
  - HitRate@20: 0.0850

### 迭代过程

1.  **多行为加权 + 时间衰减**：
    - **参数**：`--use_multi_behavior True --behavior_weights "click:1,cart:2,fav:2,buy:3" --time_decay 0.98 --recent_days 7`, `epochs=30`
    - **结果**：Recall@20: 0.0550, NDCG@20: 0.0178, HitRate@20: 0.0550。指标出现下降，分析原因可能是小样本噪声大，复杂的权重策略反而引入干扰。

2.  **LightGCN 层数与嵌入维度微调**：
    - **参数组合 1**：`n_layers=2`, `embed_dim=128`, `epochs=30`
      - **结果**：Recall@20: 0.0600, NDCG@20: 0.0215, HitRate@20: 0.0600。性能仍不如 Baseline。
    - **参数组合 2**：`n_layers=4`, `embed_dim=128`, `epochs=30`
      - **结果**：**Recall@20: 0.0900**, NDCG@20: 0.0282, **HitRate@20: 0.0900**。**Recall 和 HitRate 指标成功超越 Baseline**。

3.  **SGL 参数微调**：
    - **背景**：基于已提升的 `n_layers=4, embed_dim=128` 配置进行。
    - **参数组合 1**：`aug='edge'`, `ratio=0.05`, `temperature=0.3`, `epochs=30`
      - **结果**：Recall@20: 0.0550, NDCG@20: 0.0205, HitRate@20: 0.0550。性能下降。
    - **参数组合 2**：`aug='node'`, `ratio=0.05`, `temperature=0.3`, `epochs=30`
      - **结果**：Recall@20: 0.0500, NDCG@20: 0.0177, HitRate@20: 0.0500。性能进一步下降。分析 SGL 在小样本上可能因数据增强进一步加剧稀疏性，导致负向效果。

### 最终改进结论

在当前小规模样本上，**将 LightGCN 的层数增加到 4 层、嵌入维度增加到 128 维** 是最有效的改进策略，成功将 **Recall@20 从 0.0850 提升至 0.0900**，**HitRate@20 从 0.0850 提升至 0.0900**。
