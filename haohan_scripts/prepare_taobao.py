import argparse
import os
import json
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

BEHAVIOR_DEFAULT_WEIGHTS = {
    'pv': 1, 'click': 1, 'cart': 2, 'fav': 2, 'buy': 3
}


def parse_behavior_weights(arg: str) -> Dict[str, float]:
    if not arg:
        return BEHAVIOR_DEFAULT_WEIGHTS
    m = {}
    for kv in arg.split(','):
        k, v = kv.split(':')
        m[k.strip()] = float(v)
    return m


def leave_one_out_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按用户分组按时间排序：最后一条为 test，倒数第二条为 valid，其余为 train"""
    train_rows = []
    valid_rows = []
    test_rows = []
    for uid, g in df.groupby('user_id'):
        g = g.sort_values('timestamp')
        if len(g) == 1:
            test_rows.append(g.iloc[0])
            continue
        test_rows.append(g.iloc[-1])
        valid_rows.append(g.iloc[-2])
        if len(g) > 2:
            train_rows.extend(g.iloc[:-2].to_dict('records'))
    train = pd.DataFrame(train_rows) if len(train_rows) > 0 else pd.DataFrame(columns=df.columns)
    valid = pd.DataFrame(valid_rows) if len(valid_rows) > 0 else pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(test_rows) if len(test_rows) > 0 else pd.DataFrame(columns=df.columns)
    return train, valid, test


def apply_multi_behavior_weights(df: pd.DataFrame, weights: Dict[str, float]):
    df['edge_weight'] = df['behavior_type'].map(lambda x: weights.get(x, 1.0))


def apply_time_decay(df: pd.DataFrame, gamma: float, recent_days: int):
    # 最近 N 天权重较高：采用 w = w_behavior * (gamma ** days_ago)，days_ago 超过 recent_days 限制到 recent_days 以增强近期交互
    max_ts = df['timestamp'].max()
    one_day = 24 * 3600
    days_ago = ((max_ts - df['timestamp']) / one_day).clip(lower=0)
    days_ago = np.minimum(days_ago, recent_days)
    df['edge_weight'] = df.get('edge_weight', 1.0) * (gamma ** days_ago)


def build_id_mapping(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    users = pd.concat([train['user_id'], valid['user_id'], test['user_id']]).unique()
    items = pd.concat([train['item_id'], valid['item_id'], test['item_id']]).unique()
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}
    return user2id, item2id


def remap_ids(df: pd.DataFrame, user2id: Dict, item2id: Dict):
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)
    return df


def filter_by_min_interactions(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    if min_interactions <= 1:
        return df
    counts = df.groupby('user_id')['item_id'].count()
    keep_users = counts[counts >= min_interactions].index
    return df[df['user_id'].isin(keep_users)]


def main():
    parser = argparse.ArgumentParser(description='Prepare Taobao UserBehavior: ID mapping, leave-one-out, sampling, edge weights')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--min_interactions', type=int, default=5)
    parser.add_argument('--mode', type=str, choices=['light', 'full'], default='light')
    parser.add_argument('--light_samples', type=int, default=1000000, help='轻量模式下采样的交互条数')
    parser.add_argument('--use_multi_behavior', type=bool, default=False)
    parser.add_argument('--behavior_weights', type=str, default='')
    parser.add_argument('--time_decay', type=float, default=0.0)
    parser.add_argument('--recent_days', type=int, default=7)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Prepare] Load CSV: {args.data_path}")
    # 尝试无表头读取；若失败则带列名读取
    try:
        df = pd.read_csv(args.data_path, header=None, names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])
    except Exception:
        df = pd.read_csv(args.data_path)
        if 'user_id' not in df.columns:
            df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

    # 过滤日期范围（2017-11-25 至 2017-12-03）
    start_ts = int(time.mktime(time.strptime("2017-11-25 00:00:00", "%Y-%m-%d %H:%M:%S")))
    end_ts = int(time.mktime(time.strptime("2017-12-03 23:59:59", "%Y-%m-%d %H:%M:%S")))
    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]

    # 轻量采样
    if args.mode == 'light':
        if len(df) > args.light_samples:
            df = df.sort_values('timestamp').tail(args.light_samples)
        print(f"[Prepare] Light sampling: {len(df)} rows")
    else:
        print(f"[Prepare] Full mode: {len(df)} rows")

    # 多行为权重
    if args.use_multi_behavior:
        wmap = parse_behavior_weights(args.behavior_weights)
        apply_multi_behavior_weights(df, wmap)

    # 时间衰减权重
    if args.time_decay and args.time_decay > 0:
        apply_time_decay(df, args.time_decay, args.recent_days)
    else:
        if 'edge_weight' not in df.columns:
            df['edge_weight'] = 1.0

    df = filter_by_min_interactions(df, args.min_interactions)
    train, valid, test = leave_one_out_split(df)
    print(f"[Split] Train={len(train)}, Valid={len(valid)}, Test={len(test)}")

    # 构建 ID 映射并重映射
    user2id, item2id = build_id_mapping(train, valid, test)
    train = remap_ids(train, user2id, item2id)
    valid = remap_ids(valid, user2id, item2id)
    test = remap_ids(test, user2id, item2id)


    # 保存映射与切分
    mapping_dir = os.path.join(args.output_dir, 'mapping')
    os.makedirs(mapping_dir, exist_ok=True)
    serializable_user2id = {str(k): int(v) for k, v in user2id.items()}
    serializable_item2id = {str(k): int(v) for k, v in item2id.items()}
    with open(os.path.join(mapping_dir, 'user2id.json'), 'w') as f:
        json.dump(serializable_user2id, f)
    with open(os.path.join(mapping_dir, 'item2id.json'), 'w') as f:
        json.dump(serializable_item2id, f)

    split_dir = os.path.join(args.output_dir, 'split')
    os.makedirs(split_dir, exist_ok=True)
    train.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(split_dir, 'valid.csv'), index=False)
    test.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

    # 保存用户-正例集合（训练集）便于负采样
    user_pos = defaultdict(set)
    for r in train[['user_id', 'item_id']].itertuples(index=False):
        user_pos[r.user_id].add(int(r.item_id))
    with open(os.path.join(split_dir, 'user_pos.json'), 'w') as f:
        json.dump({str(k): list(v) for k, v in user_pos.items()}, f)

    print(f"[Done] Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
