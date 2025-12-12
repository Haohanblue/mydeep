#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工程版：支持 1 亿行级别 Taobao UserBehavior 预处理（Spark）
功能基本等价于你原来的 pandas 版：
- 日期过滤
- 轻量 / 全量模式
- 多行为权重
- 时间衰减
- leave-one-out 切分 (train/valid/test)
- user/item ID 重映射为连续索引
- min_interactions 过滤
- 生成 user_pos（训练集正例集合）

输出目录结构：
  output_dir/
    mapping/
      user2id.json
      item2id.json
    split/
      train.parquet
      valid.parquet
      test.parquet
      user_pos.json

注意：本脚本依赖 PySpark。
"""

import argparse
import os
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window


BEHAVIOR_DEFAULT_WEIGHTS = {
    'pv': 1.0, 'click': 1.0, 'cart': 2.0, 'fav': 2.0, 'buy': 3.0
}


def parse_behavior_weights(arg: str):
    if not arg:
        return BEHAVIOR_DEFAULT_WEIGHTS
    m = {}
    for kv in arg.split(','):
        k, v = kv.split(':')
        m[k.strip()] = float(v)
    return m


def build_behavior_weight_expr(col, weights, default=1.0):
    """
    在 Spark 中构建一个 map 表达式：
        edge_weight = weights[behavior_type]，不存在则 default
    """
    pairs = []
    for k, v in weights.items():
        pairs.extend([F.lit(k), F.lit(float(v))])
    mapping = F.create_map(*pairs) if pairs else F.create_map()
    return F.coalesce(mapping[col], F.lit(float(default)))


def write_mapping_json(df, key_col, val_col, out_path):
    """
    流式把两列写成一个 JSON 对象：
      { "<key>": <val>, ... }
    避免 collect()/toPandas() 导致内存爆。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('{')
        first = True
        for row in df.select(key_col, val_col).toLocalIterator():
            if not first:
                f.write(',')
            first = False
            key = str(row[key_col])
            val = int(row[val_col])
            f.write(json.dumps(key))  # key 必须 json 转义
            f.write(':')
            f.write(str(val))         # val 可以直接写 int
        f.write('}')


def write_user_pos_json(df, user_col, item_col, out_path):
    """
    user_pos: { "user_id": [item1, item2, ...], ... }
    也是流式写出。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('{')
        first_user = True
        # collect_set 会在 executor 侧聚合，driver 只接受一行一行结果
        agg_df = df.groupBy(user_col).agg(
            F.collect_set(item_col).alias('pos_items')
        )
        for row in agg_df.toLocalIterator():
            if not first_user:
                f.write(',')
            first_user = False
            uid = int(row[user_col])
            items = [int(x) for x in row['pos_items']]
            f.write(json.dumps(str(uid)))
            f.write(':')
            f.write(json.dumps(items))
        f.write('}')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Taobao UserBehavior with Spark: '
                    'ID mapping, leave-one-out, sampling, edge weights'
    )

    parser.add_argument('--data_path', type=str, required=True,
                        help='原始 Taobao UserBehavior CSV 路径')
    parser.add_argument('--output_dir', type=str, default='data/processed_spark')
    parser.add_argument('--min_interactions', type=int, default=5)
    parser.add_argument('--mode', type=str, choices=['light', 'full'], default='light')
    parser.add_argument('--light_samples', type=int, default=1_000_000,
                        help='轻量模式下采样的交互条数（近似）')
    parser.add_argument('--use_multi_behavior', action='store_true')
    parser.add_argument('--behavior_weights', type=str, default='')
    parser.add_argument('--time_decay', type=float, default=0.0)
    parser.add_argument('--recent_days', type=int, default=7)
    parser.add_argument('--num_output_partitions', type=int, default=128,
                        help='输出 parquet 分区数')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName('PrepareTaobaoUserBehavior')
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"[Prepare] Load CSV via Spark: {args.data_path}")

    # 尝试无表头读取；若失败则再尝试有表头
    # schema 大致固定：user_id,item_id,category_id,behavior_type,timestamp
    df = spark.read.csv(
        args.data_path,
        header=False,
        inferSchema=True
    )

    if len(df.columns) != 5:
        # 再尝试 header=True
        df = spark.read.csv(
            args.data_path,
            header=True,
            inferSchema=True
        )
        if 'user_id' not in df.columns:
            df = df.toDF('user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp')
    else:
        df = df.toDF('user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp')

    # 确保 timestamp 是 long
    df = df.withColumn('timestamp', F.col('timestamp').cast('long'))

    # 日期过滤（2017-11-25 至 2017-12-03）
    start_ts = int(time.mktime(time.strptime("2017-11-25 00:00:00", "%Y-%m-%d %H:%M:%S")))
    end_ts = int(time.mktime(time.strptime("2017-12-03 23:59:59", "%Y-%m-%d %H:%M:%S")))
    df = df.filter((F.col('timestamp') >= start_ts) & (F.col('timestamp') <= end_ts))

    print(f"[Filter] after date range: ~{df.count()} rows")

    # 轻量采样模式
    if args.mode == 'light':
        total = df.count()
        if total > args.light_samples:
            frac = float(args.light_samples) / float(total)
            frac = min(1.0, max(frac, 0.000001))
            print(f"[Prepare] Light mode: target ~{args.light_samples} rows, "
                  f"total={total}, fraction={frac:.6f}")
            df = df.sample(withReplacement=False, fraction=frac, seed=42)
        print(f"[Prepare] Light sampling done: ~{df.count()} rows")
    else:
        print(f"[Prepare] Full mode: ~{df.count()} rows")

    # 多行为权重
    if args.use_multi_behavior:
        wmap = parse_behavior_weights(args.behavior_weights)
        edge_expr = build_behavior_weight_expr(F.col('behavior_type'), wmap, default=1.0)
        df = df.withColumn('edge_weight', edge_expr)
    else:
        df = df.withColumn('edge_weight', F.lit(1.0))

    # 时间衰减权重： w = w_behavior * (gamma ** days_ago_clipped)
    if args.time_decay and args.time_decay > 0:
        max_ts_row = df.agg(F.max('timestamp').alias('max_ts')).first()
        max_ts = max_ts_row['max_ts']
        print(f"[TimeDecay] max timestamp={max_ts}")

        one_day = 24 * 3600.0
        days_ago = (F.lit(max_ts) - F.col('timestamp')) / F.lit(one_day)
        # clip [0, recent_days]
        days_ago = F.when(days_ago < 0, F.lit(0.0)).otherwise(days_ago)
        days_ago = F.when(days_ago > float(args.recent_days),
                          F.lit(float(args.recent_days))).otherwise(days_ago)

        df = df.withColumn(
            'edge_weight',
            F.col('edge_weight') * F.pow(F.lit(float(args.time_decay)), days_ago)
        )

    # leave-one-out 切分
    print("[Split] Computing leave-one-out split with window functions...")
    w = Window.partitionBy('user_id').orderBy('timestamp')
    w_cnt = Window.partitionBy('user_id')

    df = df.withColumn('rn', F.row_number().over(w)) \
           .withColumn('cnt', F.count(F.lit(1)).over(w_cnt))

    test = df.filter(F.col('rn') == F.col('cnt'))
    valid = df.filter((F.col('cnt') >= 2) & (F.col('rn') == F.col('cnt') - 1))
    train = df.filter(F.col('rn') <= F.col('cnt') - 2)

    print(f"[Split] "
          f"Train ~{train.count()}, Valid ~{valid.count()}, Test ~{test.count()}")

    # min_interactions 过滤：只对 train 限制，valid/test 对齐保留用户
    if args.min_interactions > 1:
        print(f"[Filter] Applying min_interactions={args.min_interactions} on train...")
        user_cnt = train.groupBy('user_id').agg(F.count('item_id').alias('cnt'))
        keep_users = user_cnt.filter(F.col('cnt') >= args.min_interactions).select('user_id')
        # inner join / semi join
        train = train.join(keep_users, on='user_id', how='inner')
        valid = valid.join(keep_users, on='user_id', how='left_semi')
        test = test.join(keep_users, on='user_id', how='left_semi')
        print(f"[Filter] After min_interactions: "
              f"Train ~{train.count()}, Valid ~{valid.count()}, Test ~{test.count()}")

    # 构建 ID 映射：基于筛选后的 train+valid+test（确保所有 target 都在映射里）
    print("[Mapping] Building user/item ID mappings...")
    all_users = (train.select('user_id')
                 .union(valid.select('user_id'))
                 .union(test.select('user_id'))
                 .distinct())

    all_items = (train.select('item_id')
                 .union(valid.select('item_id'))
                 .union(test.select('item_id'))
                 .distinct())

    # 为了得到连续 ID，用 row_number over orderBy
    w_user_map = Window.orderBy('user_id')
    w_item_map = Window.orderBy('item_id')

    user_map = (all_users
                .withColumn('user_idx', F.row_number().over(w_user_map) - 1)
                .persist())

    item_map = (all_items
                .withColumn('item_idx', F.row_number().over(w_item_map) - 1)
                .persist())

    print(f"[Mapping] #users={user_map.count()}, #items={item_map.count()}")

    # 把原来的 user_id/item_id 映射为 index
    train = (train
             .join(user_map, on='user_id', how='inner')
             .join(item_map, on='item_id', how='inner'))

    valid = (valid
             .join(user_map, on='user_id', how='inner')
             .join(item_map, on='item_id', how='inner'))

    test = (test
            .join(user_map, on='user_id', how='inner')
            .join(item_map, on='item_id', how='inner'))

    # 只保留必要列，并将 user_id/item_id 列名对齐为原来含义（索引）
    out_cols = ['user_idx', 'item_idx', 'category_id',
                'behavior_type', 'timestamp', 'edge_weight']

    train = train.select(*out_cols)
    valid = valid.select(*out_cols)
    test = test.select(*out_cols)

    # 重命名为 user_id/item_id 供后续 LightGCN 等代码使用
    for df_name, df_ref in [('train', train), ('valid', valid), ('test', test)]:
        pass  # 只是占位，下面真正 rename

    train = train.withColumnRenamed('user_idx', 'user_id') \
                 .withColumnRenamed('item_idx', 'item_id')

    valid = valid.withColumnRenamed('user_idx', 'user_id') \
                 .withColumnRenamed('item_idx', 'item_id')

    test = test.withColumnRenamed('user_idx', 'user_id') \
               .withColumnRenamed('item_idx', 'item_id')

    # 输出映射和切分结果
    mapping_dir = os.path.join(args.output_dir, 'mapping')
    split_dir = os.path.join(args.output_dir, 'split')
    os.makedirs(mapping_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # 1) 映射：流式写 JSON（不会 OOM）
    user2id_json = os.path.join(mapping_dir, 'user2id.json')
    item2id_json = os.path.join(mapping_dir, 'item2id.json')
    print(f"[Save] Writing user2id -> {user2id_json}")
    write_mapping_json(user_map, 'user_id', 'user_idx', user2id_json)

    print(f"[Save] Writing item2id -> {item2id_json}")
    write_mapping_json(item_map, 'item_id', 'item_idx', item2id_json)

    # 2) train/valid/test 保存为 parquet（工程上更合理）
    # 若你强制需要 CSV，可以在 parquet 基础上再转一次。
    print("[Save] Writing train/valid/test parquet files...")
    train.repartition(args.num_output_partitions) \
         .write.mode('overwrite').parquet(os.path.join(split_dir, 'train.parquet'))

    valid.repartition(max(1, args.num_output_partitions // 4)) \
         .write.mode('overwrite').parquet(os.path.join(split_dir, 'valid.parquet'))

    test.repartition(max(1, args.num_output_partitions // 4)) \
        .write.mode('overwrite').parquet(os.path.join(split_dir, 'test.parquet'))

    # 3) user_pos.json：训练集 user_id -> 正例 item_id 列表
    print("[Save] Writing user_pos.json (from train)...")
    user_pos_json = os.path.join(split_dir, 'user_pos.json')
    write_user_pos_json(train, 'user_id', 'item_id', user_pos_json)

    print(f"[Done] All artifacts saved under: {args.output_dir}")

    spark.stop()


if __name__ == '__main__':
    main()
