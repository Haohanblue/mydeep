import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--min_interactions", type=int, required=True)
    parser.add_argument("--has_header", type=str, default="False")
    parser.add_argument("--print_stats", type=str, default="False")
    parser.add_argument("--max_interactions", type=int, default=0)
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
    if args.min_interactions <= 1 and (not args.max_interactions or args.max_interactions <= 0):
        df = lf.collect()
    else:
        cnt_lf = lf.group_by("user_id").len()
        cond = pl.col("len") >= args.min_interactions
        if args.max_interactions and args.max_interactions > 0:
            cond = cond & (pl.col("len") <= args.max_interactions)
        cnt = cnt_lf.filter(cond).select("user_id")
        lf = lf.join(cnt, on="user_id", how="inner")
        df = lf.collect()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    df.write_csv(args.output_path, include_header=(args.has_header.lower() == "true"))
    print(df.height)
    if args.print_stats.lower() == "true":
        import polars as pl
        uc = df.select(pl.n_unique("user_id")).item()
        ic = df.select(pl.n_unique("item_id")).item()
        print(f"users={uc}")
        print(f"items={ic}")

if __name__ == "__main__":
    main()
