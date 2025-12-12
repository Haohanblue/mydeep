import argparse
import sys
import os
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_users", type=int, required=True)
    parser.add_argument("--has_header", type=str, default="False")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--print_stats", type=str, default="False")
    args = parser.parse_args()
    try:
        import polars as pl
    except Exception:
        print("polars not available", file=sys.stderr)
        sys.exit(1)
    lf = pl.scan_csv(args.data_path, has_header=(args.has_header.lower() == "true"), infer_schema_length=1000)
    names = lf.collect_schema().names()
    if len(names) >= 5:
        lf = lf.select(names[:5])
    if "user_id" not in lf.collect_schema().names():
        names = lf.collect_schema().names()
        lf = lf.rename({
            names[0]: "user_id",
            names[1]: "item_id",
            names[2]: "category_id",
            names[3]: "behavior_type",
            names[4]: "timestamp",
        })
    lf = lf.select(["user_id", "item_id", "category_id", "behavior_type", "timestamp"]).with_columns([
        pl.col("user_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_id").cast(pl.Utf8),
        pl.col("behavior_type").cast(pl.Utf8),
        pl.col("timestamp").cast(pl.Int64),
    ])
    users_df = lf.select("user_id").unique().collect()
    users = users_df["user_id"].to_list()
    if not users:
        out = pl.DataFrame({"user_id": [], "item_id": [], "category_id": [], "behavior_type": [], "timestamp": []})
    else:
        rng = random.Random(args.seed)
        n = args.num_users if args.num_users < len(users) else len(users)
        sel = rng.sample(users, n)
        sel_df = pl.DataFrame({"user_id": sel}).lazy()
        out_lf = lf.join(sel_df, on="user_id", how="inner")
        out = out_lf.collect()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out.write_csv(args.output_path, include_header=(args.has_header.lower() == "true"))
    print(out.height)
    if args.print_stats.lower() == "true":
        import polars as pl
        uc = out.select(pl.n_unique("user_id")).item()
        ic = out.select(pl.n_unique("item_id")).item()
        print(f"users={uc}")
        print(f"items={ic}")

if __name__ == "__main__":
    main()
