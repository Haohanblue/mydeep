## 方法与评估解答（FAQ）

### 训练/验证/测试集如何划分
- **按用户分组的留一法**：对每个用户的交互按 `timestamp` 升序排序，最后一条作为 **Test**、倒数第二条作为 **Valid**、其余作为 **Train**。该策略确保评估阶段只对“未来一次交互”进行预测，符合真实线上时序。
- **min_interactions 的筛选与对齐**：
  - 训练集按用户互动数阈值 `min_interactions`（如 2/5）进行筛选，仅保留互动数≥阈值的用户；
  - 为保持评估集合可用性，Valid/Test 仅保留训练后仍存在的用户（与 Train 对齐）。脚本实现见 `scripts/prepare_taobao.py`：对 Train 过滤后用 `keep_users` 对 Valid/Test 进行对齐。
- **轻量采样 vs 全量模式**：
  - 轻量采样（`--mode light`）用于快速验证流程：按时间取前 N 条（如 100 万行或本地示例 5000 行），显著降低内存占用与构图/训练耗时；
  - 全量模式（`--mode full`）保留全部符合日期范围的交互，适合得到稳定结论与更高上限，但需要更强算力与更长训练。

### 评估的标准是什么
- **评估协议（leave-one-out）**：每位用户的 Test 集仅 1 个正例，模型对该用户的所有物品打分并排序，取 Top-K 列表，与该正例进行比对；评估时默认**过滤已看交互**（Train/Valid）以避免将已交互物品纳入候选。
- **过滤已看（train/valid）交互**：评估阶段对每个用户构建已交互集合（Train + Valid），在打分矩阵上将这些物品的分数置为极小值以实现过滤（参见 `train.py/evaluate.py` 中 `filter_seen: true`）。
- **Top-K 的选择与理由**：常用 K=20/50。K 过小不利于覆盖、K 过大评估成本与用户真实点击转化相关性下降。K=20 兼顾覆盖率与计算开销，是电商推荐召回的常用取值。
- **指标定义（文字表达）**：
  - **Recall@K**：若 Test 正例出现在 Top-K 中，记 1，否则记 0；对所有用户求平均。反映 Top-K 命中能力。
  - **HitRate@K**：与 Recall@K 等价（单一正例场景下）。
  - **NDCG@K**：若正例位于 Top-K 的第 `rank` 位，则 `NDCG = 1 / log2(rank + 1)`；未命中则为 0。该指标更强调“命中的排序位置”。

### 我们算法的目标与问题
- **召回阶段目标**：为后续排序模型提供**高质量候选集**，兼顾**覆盖**（多样性与长尾覆盖）与**精准**（Top-K 命中率与前列位置）。
- **问题性质**：这是典型的“**用户—物品链接预测 / Top-K 推荐**”，目标是最大化 Top-K 命中与排序质量（Recall/NDCG/HitRate）。
- **挑战**：
  - **数据稀疏**：大量用户只产生极少互动，导致嵌入学习不充分；
  - **长尾场景**：海量低曝光/低互动的长尾物品难以学习到可靠表示，需要更强的正则与自监督增强来提升鲁棒性与泛化。

### 用到的图神经网络与结构（LightGCN 与 BPR）
- **结构组成**：
  - 用户/物品嵌入（`nn.Embedding`），无非线性与 MLP 参数；
  - **对称归一化二部图邻接**：将用户—物品矩阵 UI 组装成 `[0, UI; UI^T, 0]` 并做度归一化；
  - **K 层传播 + 层间聚合**：通过稀疏邻接进行 K 次线性传播，将各层（含初始）嵌入聚合（常用平均/加权和）。
- **目标函数（BPR）**：使用对偶比较的**BPR 排序损失**：对每个用户采样一个已互动的正例 `i+` 与一个未互动的负例 `i-`，最大化 `score(u, i+) - score(u, i-)` 的间隔，并做 L2 正则化。
- **核心代码片段（引自 `model/lightgcn.py`）**：
```python
# computer：K 层传播并聚合
def computer(self):
    all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
    embs = [all_emb]
    x = all_emb
    for _ in range(self.n_layers):
        x = torch.sparse.mm(self.adj, x)
        embs.append(x)
    out = torch.mean(torch.stack(embs, dim=0), dim=0)
    user_out, item_out = torch.split(out, [self.n_users, self.n_items], dim=0)
    return user_out, item_out
```
```python
# getUsersRating：用户对所有物品的评分
def getUsersRating(self, users):
    user_out, item_out = self.computer()
    users_emb = user_out[users]
    scores = torch.matmul(users_emb, item_out.t())
    return scores
```
```python
# bpr_loss：BPR 排序目标 + L2 正则
def bpr_loss(self, users, pos_items, neg_items, reg=1e-4):
    user_out, item_out = self.computer()
    u = user_out[users]
    pos = item_out[pos_items]
    neg = item_out[neg_items]
    pos_scores = torch.sum(u * pos, dim=1)
    neg_scores = torch.sum(u * neg, dim=1)
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    reg_loss = reg * (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / users.shape[0]
    return loss + reg_loss
```

### 改进模型说明
- **SGL（Self-supervised Graph Learning）**：
  - **视图增强**：对邻接进行 `edge`/`node`/`rw`（简化为边丢弃）增强，得到两个扰动视图；
  - **对比损失（InfoNCE）**：同一用户在两视图中的表示为正样本、不同用户为负样本，通过温度参数约束相似度学习，提升表示稳健性与长尾泛化；
  - **预期作用**：在稀疏与噪声场景下提升鲁棒性与泛化，缓解过拟合。
- **多行为加权 / 时间衰减（可选）**：
  - 多行为加权：为边赋予权重（如 click=1, cart=2, fav=2, buy=3），强调强信号；
  - 时间衰减：近期交互权重更高（如 `edge_weight *= gamma ** days_ago`，`recent_days` 上限裁剪），体现新近性；
  - 两者共同作用于图的边权，影响度归一化与传播过程的强弱。

### 数据集介绍
- **Taobao UserBehavior**：包含字段 `user_id, item_id, category_id, behavior_type(pv/click/cart/fav/buy), timestamp`，时间范围 **2017-11-25 至 2017-12-03**。
- **中国互联网场景适配性**：电商行为天然包含多类型与强新近性，适合用二部图 + LightGCN 进行召回建模；与主流公开数据（Gowalla/Yelp2018）相比，电商长尾更明显、行为语义更丰富。
- **本作业数据使用说明**：为便于流程核验，采用小规模样本进行运行与评估；建议在**全量数据**上进行**更长时训练**以得到稳定结论与更高指标上限。
