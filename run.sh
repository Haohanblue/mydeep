#!/usr/bin/env bash
set -e

# 1) 预处理（轻量采样 + 多行为权重 + 时间衰减）
uv run scripts/prepare_taobao.py \
  --data_path data/UserBehavior.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode light \
  --light_samples 1000000 \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 --recent_days 7

# 全量不抽样（3GB 数据）
uv run scripts/polars_prepare_taobao.py \
  --data_path data/UserBehavior.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode full \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 \
  --recent_days 7
  
  uv run scripts/polars_prepare_taobao.py \
  --data_path data/UserBehavior_sample.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode full \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 \
  --recent_days 7

# 全量不抽样（3GB 数据）
uv run scripts/polars_prepare_taobao.py \
  --data_path data/UserBehavior_sample.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode light \
  --light_samples 2000 \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 \
  --recent_days 7
  

uv run scripts/big_prepare_taobao.py \
  --data_path /path/to/UserBehavior.csv \
  --output_dir data/processed \
  --mode full \
  --min_interactions 5 \
  --use_multi_behavior \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 --recent_days 7 \
  --num_output_partitions 256
# 2) 构图
uv run scripts/build_graph.py \
  --proc_dir data/processed \
  --out_dir data/graph \
  --use_sparse True
n
  # 2) 构图
uv run scripts/polars_build_gsraph.py \
  --proc_dir data/processed \
  --out_dir data/graph

# 3) Baseline 训练
uv run -m scripts.train \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu

# 让 NumPy 用多核 + 关闭 Python GIL 小对象缓冲
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMBA_NUM_THREADS=$(nproc)
export PYTHONOPTIMIZE=1

nohup uv run -m scripts.polars_train \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu \
  > logs/lightgcn_train.out 2>&1 &
  
uv run -m scripts.polars_train \
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
