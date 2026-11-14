"""Download the dataset, build listening sessions and check for a mood signal

Run:
    pip install pandas pyarrow huggingface_hub
    python prepare_yambda.py

Output:
    and you obtain data/sessions.csv with one row per track: session_id, uid, position, context,
    item_id, skip

The dataset has no sessions and no skip label, so this script builds both
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct",
           "track_length_seconds"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-file", help="use this parquet file, no download")
    parser.add_argument("--output", default="data/sessions.csv")
    parser.add_argument("--n-users", type=int, default=3000)
    parser.add_argument("--skip-threshold", type=float, default=50,
                        help="a play at or below this percentage is a skip")
    parser.add_argument("--max-idle", type=float, default=60,
                        help="seconds of silence that end a session")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def get_parquet_path(args):
    if args.local_file:
        return args.local_file
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="yandex/yambda", repo_type="dataset",
                           filename="flat/50m/listens.parquet")


def load_users(path, n_users, rng):
    """Read the listens of a random group of users"""
    all_uids = pc.unique(pq.read_table(path, columns=["uid"])["uid"]).to_numpy()
    chosen = rng.choice(all_uids, size=min(n_users, len(all_uids)), replace=False)
    table = pq.read_table(path, columns=COLUMNS,
                          filters=[("uid", "in", [int(u) for u in chosen])])
    frame = table.to_pandas()
    return frame.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)


def add_seconds(frame):
    """Convert timestamps to seconds

    The dataset card speaks of 5 second bins. If every value is a multiple
    of 5 the column is already in seconds. If not, it counts bins
    """
    all_multiples = (frame["timestamp"] % 5 == 0).all()
    unit = 1 if all_multiples else 5
    frame["seconds"] = frame["timestamp"].astype("int64") * unit
    frame["played_seconds"] = (frame["track_length_seconds"]
                               * frame["played_ratio_pct"] / 100)
    print(f"Timestamp unit: {unit} second(s) per step")
    return frame


def compute_idle(frame):
    """Compute the silence before each track

    The card does not say if a timestamp marks the start or the end of a play
    We test both readings and keep the one where the time between two events
    best matches the time played
    """
    same_user = frame["uid"] == frame["uid"].shift(1)
    time_step = frame["seconds"].diff().where(same_user)
    idle_if_start = time_step - frame["played_seconds"].shift(1)
    idle_if_end = time_step - frame["played_seconds"]

    share_start = (idle_if_start.abs() <= 10).sum() / time_step.notna().sum()
    share_end = (idle_if_end.abs() <= 10).sum() / time_step.notna().sum()
    print(f"Time steps that match the time played: "
          f"{share_start:.1%} if timestamps mark the start, "
          f"{share_end:.1%} if they mark the end")

    if share_start >= share_end:
        return idle_if_start
    return idle_if_end


def build_sessions(frame, args):
    """Cut sessions at long silences and when the listening context changes"""
    frame["skip"] = (frame["played_ratio_pct"] <= args.skip_threshold).astype(int)
    frame["context"] = np.where(frame["is_organic"] == 1, "organic", "recommendation")

    idle = compute_idle(frame)
    new_user = frame["uid"] != frame["uid"].shift(1)
    new_context = frame["context"] != frame["context"].shift(1)
    long_silence = idle > args.max_idle
    frame["session_id"] = (new_user | new_context | long_silence).cumsum()

    lengths = frame.groupby("session_id")["skip"].transform("size")
    print(f"Sessions before the length filter: {frame['session_id'].nunique()}")
    kept = frame[lengths >= args.min_length].copy()
    kept["position"] = kept.groupby("session_id").cumcount()
    return kept[kept["position"] < args.max_length].copy()


def keep_one_session_per_user(sessions, rng):
    """Keep one session per user and context, so sessions are more independent"""
    table = sessions[["uid", "context", "session_id"]].drop_duplicates()
    table = table.assign(order=rng.random(len(table))).sort_values("order")
    chosen = table.drop_duplicates(["uid", "context"])["session_id"]
    return sessions[sessions["session_id"].isin(chosen)].copy()


def lag_gap(sessions):
    """Return P(skip | last track skipped) - P(skip | last track played)"""
    previous = sessions.groupby("session_id")["skip"].shift(1)
    return (sessions.loc[previous == 1, "skip"].mean()
            - sessions.loc[previous == 0, "skip"].mean())


def shuffled_gap(sessions, rng, n_repeats=20):
    """Return the same gap after shuffling tracks inside each session

    Shuffling keeps sessions that skip a lot, but it destroys the track order
    So a real gap above this value means the order matters inside a session
    """
    gaps = []
    for _ in range(n_repeats):
        shuffled = sessions.assign(order=rng.random(len(sessions)))
        shuffled = shuffled.sort_values(["session_id", "order"])
        gaps.append(lag_gap(shuffled))
    return np.mean(gaps), np.std(gaps)


def report(sessions, rng):
    print(f"\nFinal data: {sessions['session_id'].nunique()} sessions, "
          f"{len(sessions)} tracks, {sessions['uid'].nunique()} users")
    print(f"Mean session length: "
          f"{sessions.groupby('session_id').size().mean():.1f} tracks")
    for context, part in sessions.groupby("context"):
        rate = part["skip"].mean()
        spread = part.groupby("session_id")["skip"].mean().std()
        gap = lag_gap(part)
        mean_shuffled, sd_shuffled = shuffled_gap(part, rng)
        print(f"\n[{context}] {part['session_id'].nunique()} sessions")
        print(f"  skip rate: {rate:.3f}")
        print(f"  spread of skip rates between sessions: {spread:.3f}")
        print(f"  gap after a skip: {gap:.3f}")
        print(f"  gap after shuffling: {mean_shuffled:.3f} (sd {sd_shuffled:.3f})")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    path = get_parquet_path(args)
    frame = add_seconds(load_users(path, args.n_users, rng))
    sessions = keep_one_session_per_user(build_sessions(frame, args), rng)
    report(sessions, rng)
    columns = ["session_id", "uid", "position", "context", "item_id", "skip"]
    folder = os.path.dirname(args.output)
    if folder:
        os.makedirs(folder, exist_ok=True)
    sessions[columns].to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()