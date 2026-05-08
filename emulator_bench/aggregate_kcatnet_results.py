"""
Aggregate KcatNet baseline results across seeds.

Walks retrain_from_optuna/kcatnet_optuna under the KcatNet data root, reads
final_results_{val,test}.csv for every seed, and outputs
mean + variance per metric per TVT split.

Directory layout:
  <root>/retrain_from_optuna/kcatnet_optuna/<split_group>/threshold_X/seed_N/train/
  <root>/retrain_from_optuna/kcatnet_optuna/<split_group>/<other>/seed_N/train/

Split groups ending in _old are excluded (superseded by updated splits).

Output: aggregate_kcatnet_results.csv saved to the KcatNet data directory.
"""

from pathlib import Path

import pandas as pd

DATA_ROOT = Path("~/github/EMULaToR/data/processed/baselines/KcatNet").expanduser()
RETRAIN_ROOT = DATA_ROOT / "retrain_from_optuna" / "kcatnet_optuna"
SPLITS = ("val", "test")
METRICS = ("rmse", "pearson", "spearman", "r2_score", "mae", "mse")


def parse_path(seed_dir: Path) -> dict:
    """Extract split_group / threshold from a seed_* path under RETRAIN_ROOT."""
    # seed_dir: <RETRAIN_ROOT>/<split_group>/<threshold_X|other>/seed_N
    parts = seed_dir.relative_to(RETRAIN_ROOT).parts
    split_group = parts[0]
    middle = parts[1]
    threshold = middle if middle.startswith("threshold_") else None
    return dict(split_group=split_group, threshold=threshold, seed=seed_dir.name)


def load_seed_results(seed_dir: Path) -> dict[str, pd.Series] | None:
    """Return {split: metrics_series} for one seed dir, or None if incomplete."""
    results_dir = seed_dir / "train"
    results = {}
    for split in SPLITS:
        fpath = results_dir / f"final_results_{split}.csv"
        if not fpath.exists():
            return None
        df = pd.read_csv(fpath)
        results[split] = df.iloc[0]
    return results


def main():
    rows = []

    for seed_dir in sorted(RETRAIN_ROOT.rglob("seed_*")):
        if not seed_dir.is_dir():
            continue
        # skip superseded _old splits
        if any(p.endswith("_old") for p in seed_dir.parts):
            continue

        meta = parse_path(seed_dir)
        split_results = load_seed_results(seed_dir)
        if split_results is None:
            print(f"  [skip] incomplete: {seed_dir.relative_to(DATA_ROOT)}")
            continue

        for split, series in split_results.items():
            row = {**meta, "tvt_split": split}
            for metric in METRICS:
                if metric in series.index:
                    row[metric] = series[metric]
            rows.append(row)

    if not rows:
        print("No complete results found.")
        return

    df = pd.DataFrame(rows)

    group_keys = ["split_group", "threshold", "tvt_split"]
    agg = (
        df.groupby(group_keys, dropna=False)[list(METRICS)]
        .agg(["mean", "var"])
    )
    # Flatten MultiIndex columns: (metric, stat) -> metric_mean / metric_var
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = df.groupby(group_keys, dropna=False).size()
    agg = agg.reset_index()

    out_path = DATA_ROOT / "aggregate_kcatnet_results.csv"
    agg.to_csv(out_path, index=False)
    print(f"Saved {len(agg)} rows to {out_path}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
