#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json

import polars as pl
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, bmat, save_npz

from tqdm import tqdm

# ---------------------------
# Step 1: 构建 User-Item CSR
# ---------------------------

def build_ui_matrix_polars(train_csv: str, n_users: int, n_items: int) -> csr_matrix:
    print(f"[Graph] Loading train.csv via Polars: {train_csv}")

    with tqdm(total=3, desc="Build UI matrix", unit="stage") as pbar:
        # Stage 1: 读取 CSV
        lf = pl.scan_csv(train_csv, infer_schema_length=1000)
        df = lf.select(
            [
                pl.col("user_id").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
                (
                    pl.col("edge_weight").cast(pl.Float32)
                    if "edge_weight" in lf.collect_schema().names()
                    else pl.lit(1.0, dtype=pl.Float32)
                ).alias("edge_weight"),
            ]
        ).collect()
        pbar.update(1)

        # Stage 2: COO
        rows = df["user_id"].to_numpy()
        cols = df["item_id"].to_numpy()
        data = df["edge_weight"].to_numpy()
        pbar.update(1)

        # Stage 3: CSR
        mat_ui = coo_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        ).tocsr()
        pbar.update(1)

    return mat_ui


# ------------------------------------
# Step 2: 稀疏对称归一化（LightGCN）
# ------------------------------------

def sym_normalize_bipartite_sparse(mat_ui: csr_matrix) -> csr_matrix:
    n_users, n_items = mat_ui.shape

    with tqdm(total=4, desc="Normalize graph", unit="step") as pbar:
        # Step 1: degree
        du = np.asarray(mat_ui.sum(axis=1)).ravel()
        di = np.asarray(mat_ui.sum(axis=0)).ravel()
        pbar.update(1)

        # Step 2: D^-1/2
        du_inv_sqrt = np.power(du + 1e-12, -0.5)
        di_inv_sqrt = np.power(di + 1e-12, -0.5)
        D_u = diags(du_inv_sqrt, 0, shape=(n_users, n_users), dtype=np.float32)
        D_i = diags(di_inv_sqrt, 0, shape=(n_items, n_items), dtype=np.float32)
        pbar.update(1)

        # Step 3: normalize
        mat_norm = D_u @ mat_ui @ D_i
        pbar.update(1)

        # Step 4: block adj
        adj = bmat(
            [
                [csr_matrix((n_users, n_users), dtype=np.float32), mat_norm],
                [mat_norm.T, csr_matrix((n_items, n_items), dtype=np.float32)],
            ],
            format="csr",
        )
        pbar.update(1)

    return adj


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build LightGCN bipartite graph (Polars + CSR)"
    )
    parser.add_argument("--proc_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="data/graph")
    args = parser.parse_args()

    split_dir = os.path.join(args.proc_dir, "split")
    map_dir = os.path.join(args.proc_dir, "mapping")
    train_csv = os.path.join(split_dir, "train.csv")

    # 加载映射，确定规模
    with open(os.path.join(map_dir, "user2id.json")) as f:
        user2id = json.load(f)
    with open(os.path.join(map_dir, "item2id.json")) as f:
        item2id = json.load(f)

    n_users = len(user2id)
    n_items = len(item2id)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1️⃣ UI 矩阵
    mat_ui = build_ui_matrix_polars(train_csv, n_users, n_items)
    save_npz(os.path.join(args.out_dir, "ui_train.npz"), mat_ui)

    # 2️⃣ 对称归一化邻接
    adj = sym_normalize_bipartite_sparse(mat_ui)
    save_npz(os.path.join(args.out_dir, "adj_norm.npz"), adj)

    print(f"[Graph] Saved graph files to {args.out_dir}")
    print(" - ui_train.npz")
    print(" - adj_norm.npz")


if __name__ == "__main__":
    main()
