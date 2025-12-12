#!/usr/bin/env bash
set -e

# 1) 预处理（轻量采样 + 多行为权重 + 时间衰减）
python scripts/prepare_taobao.py \
  --data_path data/UserBehavior_sample.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode light \
  --light_samples 4000 \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 --recent_days 7

# 2) 构图
python scripts/build_graph.py \
  --proc_dir data/processed \
  --out_dir data/graph \
  --use_sparse True

# 3) Baseline 训练
python scripts/train.py \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu

# 4) SGL 训练（可选）
python scripts/train_sgl.py \
  --config configs/lightgcn_sgl.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --ssl_ratio 0.1 \
  --aug edge \
  --device cpu

# 5) 统一评测
python scripts/evaluate.py \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu
