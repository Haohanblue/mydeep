## 新版流程
# 先过滤原始数据，指定交互次数范围。

uv run scripts/filter_min_interactions.py --data_path UserBehavior.csv --output_path data/UserBehavior_filtered_50_200.csv --min_interactions 50 --max_interactions 200 --has_header False --print_stats True
uv run scripts/sample_users.py --data_path data/UserBehavior_filtered_50_200.csv --output_path data/UserBehavior_user_sampled.csv --num_users 30000 --has_header False --seed 2025 --print_stats True


# ## 1.从全量数据集中，抽样数据，以下为50w条，seed=2025，输出到data/UserBehavior_fixed_sample.csv
# uv run scripts/make_fixed_sample.py \
#   --data_path UserBehavior.csv \
#   --output_path data/UserBehavior_fixed_sample.csv \
#   --light_samples 1000000 \
#   --seed 2025 \
#   --has_header False

# uv run scripts/filter_min_interactions.py --data_path data/UserBehavior_fixed_sample.csv --output_path data/UserBehavior_filtered.csv --min_interactions 5 --has_header False --print_stats True
# uv run scripts/make_fixed_sample.py --data_path UserBehavior.csv --output_path data/UserBehavior_fixed_sample.csv --light_samples 100000 --seed 2025 --has_header False --min_interactions 5
## 2.从50w条采样数据中，数据预处理，拆分成train、valid、test三部分,指定输出到路径，和最小交互次数的标准
## 这个是lightgcn模型的预处理
uv run scripts/polars_prepare_taobao.py \
  --data_path data/UserBehavior_user_sampled.csv \
  --output_dir data/processed \
  --min_interactions 2 \
  --mode full \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 \
  --recent_days 7
## 这个是lighgcn-gcl的预处理
uv run scripts/polars_prepare_taobao.py \
  --data_path data/UserBehavior_user_sampled.csv \
  --output_dir data/processed_sgl \
  --min_interactions 3 \
  --mode full \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 \
  --recent_days 7

## 3.构建图
## lightgcn模型的图构建
uv run scripts/polars_build_graph.py \
  --proc_dir data/processed \
  --out_dir data/graph
## lighgcn-gcl模型的图构建  
uv run scripts/polars_build_graph.py \
  --proc_dir data/processed_sgl \
  --out_dir data/graph_sgl

## 4.训练模型
## lightgcn模型的训练
uv run -m scripts.polars_train \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu
## lighgcn-gcl模型的训练
uv run -m scripts.polars_train_sgl \
  --config configs/lightgcn_sgl.yaml \
  --proc_dir data/processed_sgl \
  --graph_dir data/graph_sgl \
  --topk 20 \
  --device cpu

## 5.模型评估
uv run -m scripts.porlars_evaluate \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu \
  --eval_sample False

## lighgcn-gcl模型的评估
uv run -m scripts.porlars_evaluate \
  --config configs/lightgcn_sgl.yaml \
  --proc_dir data/processed_sgl \
  --graph_dir data/graph_sgl \
  --topk 20 \
  --device cpu\
  --eval_sample False



## 最原始的脚本
#### lightgcn模型
uv run scripts/prepare_taobao.py \
  --data_path data/UserBehavior_sample.csv \
  --output_dir data/processed \
  --min_interactions 5 \
  --mode light \
  --light_samples 100000 \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 --recent_days 7

  uv run scripts/build_graph.py \
  --proc_dir data/processed \
  --out_dir data/graph \
  --use_sparse True

 uv run -m scripts.train \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu 

 uv run -m scripts.evaluate \
  --config configs/lightgcn.yaml \
  --proc_dir data/processed \
  --graph_dir data/graph \
  --topk 20 \
  --device cpu

### lighgcn-gcl模型
uv run scripts/prepare_taobao.py \
  --data_path data/UserBehavior_sample.csv \
  --output_dir data/processed_sgl \
  --min_interactions 5 \
  --mode light \
  --light_samples 100000 \
  --use_multi_behavior True \
  --behavior_weights "click:1,cart:2,fav:2,buy:3" \
  --time_decay 0.98 --recent_days 7

  uv run scripts/build_graph.py \
  --proc_dir data/processed_sgl \
  --out_dir data/graph_sgl \
  --use_sparse True

 uv run -m scripts.train \
  --config configs/lightgcn_sgl.yaml \
  --proc_dir data/processed_sgl \
  --graph_dir data/graph_sgl \
  --topk 20 \
  --device cpu 

 uv run -m scripts.evaluate \
  --config configs/lightgcn_sgl.yaml \
  --proc_dir data/processed_sgl \
  --graph_dir data/graph_sgl \
  --topk 20 \
  --device cpu