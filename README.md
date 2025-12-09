# LightGCN 基于 Taobao UserBehavior 的二部图召回（中国互联网场景）

本项目提供在阿里巴巴天池 UserBehavior（2017-11-25 至 2017-12-03）数据集上的 LightGCN 召回实现与可复现实验，支持：
- 轻量采样与原始全量两种预处理模式
- 按时间“留一法”切分（last 为 test、前一条为 valid、其余为 train），支持按用户互动阈值筛选
- 构建用户—物品二部图邻接（CSR/torch.sparse），对称度归一化，K 层传播可配
- 可选多行为加权（click=1, cart=2, fav=2, buy=3）与时间衰减（近 7 天权重↑）
- 训练：BPR loss + mini-batch 负采样，GPU 可选；评测 Recall@K / NDCG@K / HitRate@K
- SGL（Self-supervised Graph Learning）可选：节点/边 dropout、随机游走视图增强 + 对比损失
- OOM 规避：稀疏邻接、批量评估、可选邻居采样、可配 batch_size

目录结构
- lightgcn_taobao/
  - README.md
  - requirements.txt
  - configs/
    - lightgcn.yaml
    - lightgcn_sgl.yaml
    - ablation.yaml
  - data/
    - README.md
  - scripts/
    - prepare_taobao.py
    - build_graph.py
    - metrics.py
    - train.py
    - train_sgl.py
    - evaluate.py
  - model/
    - lightgcn.py
  - run.sh

快速开始
1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 准备数据（支持本地已有 CSV）
- 从天池下载 UserBehavior.csv（含字段：user_id,item_id,category_id,behavior_type,timestamp），放置到 `data/UserBehavior.csv`
- 或在 data/README.md 中按说明下载

3) 预处理（ID 映射、留一法、采样、边权）
```bash
python scripts/prepare_taobao.py \
  --data_path data/UserBehavior.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode light \
  --light_samples 1000000 \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 --recent_days 7
```

4) 构图与归一化
```bash
python scripts/build_graph.py \
  --proc_dir data/processed \
  --out_dir data/graph \
  --use_sparse True
```

5) 训练与评测（Baseline LightGCN）
```bash
python scripts/train.py \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cuda:0
```

6) SGL 改进
```bash
python scripts/train_sgl.py \
  --config configs/lightgcn_sgl.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --ssl_ratio 0.1 \
  --aug edge \
  --device cuda:0
```

评测说明（leave-one-out）
- 每个用户 test 集仅 1 个正例，评估时对所有物品打分，过滤 train/valid 交互后取 Top-K
- 指标：Recall@K、NDCG@K、HitRate@K，默认 K=20，可在命令行 `--topk` 指定

常见坑与规避
- OOM：
  - 使用 `--use_sparse True` 构建 torch.sparse_coo_tensor 邻接
  - 降低 `batch_size` 与 `eval_batch_size`（configs 内可配）
  - 选择 `--neighbor_sampling True --num_neighbors 20`（可选）
- 负采样偏差：从用户未交互集合中均匀采样，且每轮重采，避免固定负样本
- 冷用户与长尾 item：提高 `--min_interactions` 阈值；或在 SGL 下提升长尾鲁棒性
- TopK 设置：过大 K 会导致评估耗时与内存飙升，建议 20/50

与公开实现参数对齐建议（便于对照）
- Gowalla：embed_dim=64, n_layers=3, lr=0.001, batch_size=2048, epochs=1000
- Yelp2018：embed_dim=64, n_layers=3, lr=0.001, batch_size=2048, epochs=1000
（参考 LightGCN 官方与社区复现仓库的常用设置）

一键运行示例
```bash
bash run.sh
```

更多说明见 data/README.md 与各脚本帮助（`-h`）。
