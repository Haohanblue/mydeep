#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产级版本：单机支持 1e8 行 Taobao UserBehavior 预处理（Polars）
- 日期过滤
- 轻量 / 全量模式
- 多行为权重（join 小表）
- 时间衰减（纯 Expr）
- leave-one-out 切分 (train/valid/test)
- user/item ID 重映射为连续索引
- min_interactions 过滤
- 生成 user_pos（训练集正例集合）

输出目录结构（与你原来一致）：
  output_dir/
    mapping/
      user2id.json
      item2id.json
    split/
      train.csv
      valid.csv
      test.csv
      user_pos.json
"""

import argparse
import os
import json
import time
from typing import Dict, Tuple

import polars as pl


BEHAVIOR_DEFAULT_WEIGHTS = {
    "pv": 1.0,
    "click": 1.0,
    "cart": 2.0,
    "fav": 2.0,
    "buy": 3.0,
}


# ------------------ 配置解析 ------------------ #

def parse_behavior_weights(arg: str) -> Dict[str, float]:
    if not arg:
        return BEHAVIOR_DEFAULT_WEIGHTS
    m: Dict[str, float] = {}
    for kv in arg.split(","):
        k, v = kv.split(":")
        m[k.strip()] = float(v)
    return m


# ------------------ 权重相关 ------------------ #

def apply_multi_behavior_weights(df: pl.DataFrame, weights: Dict[str, float]) -> pl.DataFrame:
    """
    把行为权重 dict 变成小 DataFrame，然后 join：
    完全避免 Python lambda，向量化 + 并行，适合 1e8 行。
    """
    wdf = pl.DataFrame(
        {
            "behavior_type": list(weights.keys()),
            "edge_weight": [float(v) for v in weights.values()],
        }
    )
    df = df.join(wdf, on="behavior_type", how="left")
    df = df.with_columns(pl.col("edge_weight").fill_null(1.0))
    return df


def apply_time_decay(df: pl.DataFrame, gamma: float, recent_days: int) -> pl.DataFrame:
    """
    w = w_behavior * (gamma ** days_ago_clipped)
    全部用 when/otherwise 组合，不用 clip(lower=...) 这种有版本差异的写法。
    """
    # 在 DataFrame 上拿一个标量 max_ts
    max_ts = df.select(pl.max("timestamp")).item()
    one_day = 24 * 3600

    days_ago_expr = (pl.lit(max_ts) - pl.col("timestamp")) / one_day

    days_ago_expr = (
        pl.when(days_ago_expr < 0)
        .then(0)
        .when(days_ago_expr > recent_days)
        .then(recent_days)
        .otherwise(days_ago_expr)
    )

    return df.with_columns(
        (pl.col("edge_weight") * (gamma ** days_ago_expr)).alias("edge_weight")
    )


# ------------------ 切分 & 映射 ------------------ #

def leave_one_out_split(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    使用窗口函数 + 排序实现 leave-one-out：
    - 先按 user_id, timestamp 排序
    - rn = 0,1,2,...（在每个 user 分组内）
    - cnt = len(user)
    - test:  rn == cnt-1
    - valid: cnt >= 2 & rn == cnt-2
    - train: rn <= cnt-3
    """
    df = df.sort(["user_id", "timestamp"])

    df = df.with_columns(
        [
            # 在每个 user 分组内，从 0 开始生成行号
            pl.int_range(pl.len()).over("user_id").alias("rn"),
            pl.len().over("user_id").alias("cnt"),  # pl.count 已经 deprecated
        ]
    )

    test = df.filter(pl.col("rn") == pl.col("cnt") - 1)
    valid = df.filter((pl.col("cnt") >= 2) & (pl.col("rn") == pl.col("cnt") - 2))
    train = df.filter(pl.col("rn") <= pl.col("cnt") - 3)

    train = train.drop(["rn", "cnt"])
    valid = valid.drop(["rn", "cnt"])
    test = test.drop(["rn", "cnt"])

    return train, valid, test


def build_id_mapping(
    train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    用 DataFrame + with_row_index 构建连续 ID 映射：
    - user_map: [user_id, user_idx]
    - item_map: [item_id, item_idx]
    再导出 dict 写 JSON（用户数 << 交互数，Python 开销 OK）。
    """
    users = pl.concat([train["user_id"], valid["user_id"], test["user_id"]]).unique()
    items = pl.concat([train["item_id"], valid["item_id"], test["item_id"]]).unique()

    user_map = pl.DataFrame({"user_id": users}).with_row_index("user_idx")
    item_map = pl.DataFrame({"item_id": items}).with_row_index("item_idx")

    user2id = dict(zip(user_map["user_id"].to_list(), user_map["user_idx"].to_list()))
    item2id = dict(zip(item_map["item_id"].to_list(), item_map["item_idx"].to_list()))

    return user_map, item_map, user2id, item2id


def remap_ids(
    df: pl.DataFrame,
    user_map: pl.DataFrame,
    item_map: pl.DataFrame,
) -> pl.DataFrame:
    """
    用 join 完成 ID 重映射，完全向量化。
    """
    df = df.join(user_map, on="user_id", how="left")
    df = df.join(item_map, on="item_id", how="left")

    df = df.with_columns(
        [
            pl.col("user_idx").alias("user_id"),
            pl.col("item_idx").alias("item_id"),
        ]
    ).drop(["user_idx", "item_idx"])

    return df


def filter_by_min_interactions(df: pl.DataFrame, min_interactions: int) -> pl.DataFrame:
    """
    只在 train 上用：统计每个 user 的交互次数，过滤掉过于稀疏的用户。
    """
    if min_interactions <= 1:
        return df

    cnt_df = (
        df.group_by("user_id")
        .len()  # -> ["user_id", "len"]
        .filter(pl.col("len") >= min_interactions)
        .select("user_id")
    )
    df = df.join(cnt_df, on="user_id", how="inner")
    return df


# ------------------ 主流程 ------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Taobao UserBehavior with Polars: "
            "ID mapping, leave-one-out, sampling, edge weights"
        )
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--mode", type=str, choices=["light", "full"], default="light")
    parser.add_argument(
        "--light_samples",
        type=int,
        default=1_000_000,
        help="轻量模式下采样的交互条数",
    )
    # 保持你原来的 CLI：--use_multi_behavior True / False
    parser.add_argument("--use_multi_behavior", type=bool, default=False)
    parser.add_argument("--behavior_weights", type=str, default="")
    parser.add_argument("--time_decay", type=float, default=0.0)
    parser.add_argument("--recent_days", type=int, default=7)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Prepare] Load CSV: {args.data_path}")

    # ---------- 1) Lazy 读取 + 选择列 + 日期过滤 ---------- #
    lf = pl.scan_csv(
        args.data_path,
        has_header=False,
        infer_schema_length=1000,
    )
    schema_names = lf.collect_schema().names()

    if len(schema_names) == 5:
        lf = lf.rename(
            {
                schema_names[0]: "user_id",
                schema_names[1]: "item_id",
                schema_names[2]: "category_id",
                schema_names[3]: "behavior_type",
                schema_names[4]: "timestamp",
            }
        )
    else:
        # 再尝试有表头
        lf = pl.scan_csv(
            args.data_path,
            has_header=True,
            infer_schema_length=1000,
        )
        schema_names = lf.collect_schema().names()
        if "user_id" not in schema_names:
            lf = lf.rename(
                {
                    schema_names[0]: "user_id",
                    schema_names[1]: "item_id",
                    schema_names[2]: "category_id",
                    schema_names[3]: "behavior_type",
                    schema_names[4]: "timestamp",
                }
            )

    lf = lf.select(["user_id", "item_id", "category_id", "behavior_type", "timestamp"])

    start_ts = int(time.mktime(time.strptime("2017-11-25 00:00:00", "%Y-%m-%d %H:%M:%S")))
    end_ts = int(time.mktime(time.strptime("2017-12-03 23:59:59", "%Y-%m-%d %H:%M:%S")))

    lf = (
        lf.with_columns(
            [
                pl.col("user_id").cast(pl.Utf8),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("category_id").cast(pl.Utf8),
                pl.col("behavior_type").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Int64),
            ]
        )
        .filter(
            (pl.col("timestamp") >= start_ts)
            & (pl.col("timestamp") <= end_ts)
        )
    )

    if args.mode == "light":
        lf = lf.sort("timestamp").tail(args.light_samples)
        df = lf.collect()  # 不再传 streaming 参数
        print(f"[Prepare] Light sampling: {df.height} rows")
    else:
        df = lf.collect()
        print(f"[Prepare] Full mode: {df.height} rows")

    # ---------- 2) 多行为权重 ---------- #
    if args.use_multi_behavior:
        wmap = parse_behavior_weights(args.behavior_weights)
        df = apply_multi_behavior_weights(df, wmap)
    else:
        df = df.with_columns(pl.lit(1.0).alias("edge_weight"))

    # ---------- 3) 时间衰减权重 ---------- #
    if args.time_decay and args.time_decay > 0:
        df = apply_time_decay(df, args.time_decay, args.recent_days)

    # ---------- 4) leave-one-out 切分 ---------- #
    train, valid, test = leave_one_out_split(df)
    print(f"[Split] Train={train.height}, Valid={valid.height}, Test={test.height}")

    # ---------- 5) 构建 ID 映射并重映射 ---------- #
    user_map, item_map, user2id, item2id = build_id_mapping(train, valid, test)
    train = remap_ids(train, user_map, item_map)
    valid = remap_ids(valid, user_map, item_map)
    test = remap_ids(test, user_map, item_map)

    # ---------- 6) min_interactions 过滤 + 对齐 valid/test ---------- #
    train = filter_by_min_interactions(train, args.min_interactions)
    keep_users = train["user_id"].unique()
    keep_users_df = pl.DataFrame({"user_id": keep_users})
    valid = valid.join(keep_users_df, on="user_id", how="inner")
    test = test.join(keep_users_df, on="user_id", how="inner")

    # ---------- 7) 保存映射 JSON ---------- #
    mapping_dir = os.path.join(args.output_dir, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)
    with open(os.path.join(mapping_dir, "user2id.json"), "w") as f:
        json.dump(user2id, f)
    with open(os.path.join(mapping_dir, "item2id.json"), "w") as f:
        json.dump(item2id, f)

    # ---------- 8) 保存切分 CSV ---------- #
    split_dir = os.path.join(args.output_dir, "split")
    os.makedirs(split_dir, exist_ok=True)
    train.write_csv(os.path.join(split_dir, "train.csv"))
    valid.write_csv(os.path.join(split_dir, "valid.csv"))
    test.write_csv(os.path.join(split_dir, "test.csv"))

    # ---------- 9) 保存 user_pos.json ---------- #
    user_pos_df = (
        train.select(["user_id", "item_id"])
        .group_by("user_id")
        .agg(pl.col("item_id").unique().alias("items"))
    )

    user_pos = {
        str(row["user_id"]): [int(x) for x in row["items"]]
        for row in user_pos_df.iter_rows(named=True)
    }

    with open(os.path.join(split_dir, "user_pos.json"), "w") as f:
        json.dump(user_pos, f)

    print(f"[Done] Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
