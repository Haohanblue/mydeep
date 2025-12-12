import argparse
import sys

def run(data_path: str, output_path: str, light_samples: int, seed: int, has_header: bool):
    try:
        import polars as pl
    except Exception:
        print("polars not available", file=sys.stderr)
        sys.exit(1)
    try:
        from tqdm import tqdm
        bar = tqdm(total=4)
        bar.set_description("sampling")
    except Exception:
        bar = None
    if bar:
        bar.update(1)
    df = pl.read_csv(data_path, has_header=has_header)
    if bar:
        bar.update(1)
    # 标准化列名以兼容下游脚本
    names = df.columns
    if len(names) >= 5:
        df = df.select(names[:5])
        df = df.rename(
            {
                names[0]: "user_id",
                names[1]: "item_id",
                names[2]: "category_id",
                names[3]: "behavior_type",
                names[4]: "timestamp",
            }
        )
    n = df.height
    k = light_samples if light_samples < n else n
    out = df.sample(n=k, shuffle=True, seed=seed)
    out = out.select(["user_id", "item_id", "category_id", "behavior_type", "timestamp"])
    # 强制时间列为整数字符串，避免后续 lazy cast 报错
    out = out.with_columns(
        [
            pl.col("timestamp").cast(pl.Int64).cast(pl.Utf8),
            pl.col("user_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("category_id").cast(pl.Utf8),
            pl.col("behavior_type").cast(pl.Utf8),
        ]
    )
    out.write_csv(output_path, include_header=has_header)
    if bar:
        bar.update(2)
        bar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--light_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--has_header", type=str, default="True")
    args = parser.parse_args()
    run(
        data_path=args.data_path,
        output_path=args.output_path,
        light_samples=args.light_samples,
        seed=args.seed,
        has_header=(args.has_header.lower() == "true"),
    )

if __name__ == "__main__":
    main()
