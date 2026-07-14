import argparse
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

"""
Repairs a known bug in `lerobot.datasets.v30.convert_dataset_v21_to_v30`: after
converting a v2.1 dataset to v3.0, the `index` column in each episode's data file is
carried over verbatim from the v2.1 source instead of being renumbered to match the
freshly-computed `dataset_from_index`/`dataset_to_index` row-position boundaries in
`meta/episodes/*.parquet`.

`LeRobotDataset._get_query_indices` uses `item["index"]` directly as a row position when
no `episodes=` subset is requested (see `_absolute_to_relative_idx`, only built when
episodes is not None), so this mismatch makes every delta_timestamps chunk query for every
non-first episode fall outside its own `[dataset_from_index, dataset_to_index)` range --
including the delta=0 (current-frame) query -- so the whole action chunk gets marked
`actions_is_pad=True`. A batch made up entirely of such samples has `mask.sum() == 0`,
which turns `_masked_flow_loss`'s division into 0/0 = NaN.

Confirmed against KaiyueChen/black_smash_03 (500/500 episodes affected except episode 0)
after a fresh `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30`, so this is a
conversion-tool bug, not something specific to a hand-built local subset.

Fix: each episode's `index` column is already internally contiguous (verified below), just
offset from where the v3.0 meta expects it to start -- so shift it by a constant per
episode to align with `dataset_from_index`.
"""


def fix_dataset(root: str):
    eps = pd.read_parquet(
        f"{root}/meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "dataset_from_index", "dataset_to_index"],
    ).set_index("episode_index")

    files = sorted(glob.glob(f"{root}/data/chunk-000/*.parquet"))
    print(f"{len(files)} files to check/fix")

    fixed = already_ok = 0
    for f in files:
        table = pq.read_table(f)
        df = table.to_pandas()
        ep = int(df["episode_index"].iloc[0])
        assert (df["episode_index"] == ep).all(), f"{f} has multiple episodes"

        expected_start = int(eps.loc[ep, "dataset_from_index"])
        expected_end = int(eps.loc[ep, "dataset_to_index"])
        actual_start = int(df["index"].min())
        actual_end = int(df["index"].max()) + 1
        n = len(df)
        assert expected_end - expected_start == n, (
            f"{f} ep{ep} length mismatch: meta={expected_end - expected_start} actual_rows={n}"
        )
        assert (df["index"].values == (df["index"].values[0] + pd.RangeIndex(n).values)).all(), (
            f"{f} ep{ep} index not contiguous internally -- offset-shift fix does not apply here"
        )

        if actual_start == expected_start and actual_end == expected_end:
            already_ok += 1
            continue

        df["index"] = df["index"] + (expected_start - actual_start)
        pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), f)
        fixed += 1
        if fixed % 50 == 0:
            print(f"  fixed {fixed} so far...")

    print(f"\nDone. already_ok={already_ok} fixed={fixed} total={len(files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True,
        help="Local v3.0 dataset root produced by lerobot.datasets.v30.convert_dataset_v21_to_v30 "
             "(e.g. ~/.cache/huggingface/lerobot/<repo_id> if converted in place with --push-to-hub=false)",
    )
    args = parser.parse_args()
    fix_dataset(args.root)
