"""Tools shared for the three layers, so we compare them in the same way"""

import numpy as np
import pandas as pd


def load_sessions(path="data/sessions.csv"):
    """Read the sessions built by prepare_yambda.py"""
    try:
        sessions = pd.read_csv(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{path} is missing. Run prepare_yambda.py first"
        ) from error
    return sessions.sort_values(["session_id", "position"]).reset_index(drop=True)


def split_users(sessions, seed=2026, test_share=0.2):
    """Put each user in the training group or the test group

    We split users and not tracks, so the test sessions are really new
    All layers use the same seed, so they share the same split
    """
    users = np.sort(sessions["uid"].unique())
    shuffled = np.random.default_rng(seed).permutation(users)
    n_test = int(round(test_share * len(users)))
    test_users = set(shuffled[:n_test].tolist())
    is_test = sessions["uid"].isin(test_users)
    return sessions[~is_test].copy(), sessions[is_test].copy()


def to_matrix(sessions):
    """Turn sessions into a table with one row per session

    Sessions have different lengths, so we also return a mask
    The mask is True where a track exists
    """
    ids, rows = np.unique(sessions["session_id"], return_inverse=True)
    n_columns = sessions["position"].max() + 1
    skips = np.zeros((len(ids), n_columns), dtype=int)
    mask = np.zeros((len(ids), n_columns), dtype=bool)
    columns = sessions["position"].to_numpy()
    skips[rows, columns] = sessions["skip"].to_numpy()
    mask[rows, columns] = True
    return skips, mask


def spread_of_skip_rates(skips, mask):
    """Return the standard deviation of the skip rate between sessions"""
    rates = (skips * mask).sum(axis=1) / mask.sum(axis=1)
    return rates.std(ddof=1)


def gap_after_skip(skips, mask):
    """Return P(skip | last track skipped) - P(skip | last track played)"""
    previous = skips[:, :-1]
    current = skips[:, 1:]
    both_exist = mask[:, 1:]
    after_skip = current[both_exist & (previous == 1)]
    after_play = current[both_exist & (previous == 0)]
    return after_skip.mean() - after_play.mean()


def mean_longest_run(skips, mask):
    """Return the mean length of the longest run of skips in a session"""
    run = np.zeros(len(skips), dtype=int)
    longest = np.zeros(len(skips), dtype=int)
    for column in range(skips.shape[1]):
        is_skip = (skips[:, column] == 1) & mask[:, column]
        run = (run + 1) * is_skip
        longest = np.maximum(longest, run)
    return longest.mean()


def session_checks(skips, mask):
    """Return the three numbers that a good model must reproduce"""
    return {
        "spread of skip rates": spread_of_skip_rates(skips, mask),
        "gap after a skip": gap_after_skip(skips, mask),
        "longest run of skips": mean_longest_run(skips, mask),
    }


def average_log_score(skip_probabilities, skips, mask):
    """Return the mean log probability given to each real test track
    """
    probability_of_truth = np.where(skips == 1, skip_probabilities,
                                    1 - skip_probabilities)
    return np.log(probability_of_truth[mask]).mean()

# The real data and every simulated data set use the same number of shuffles, so the observed and the simulated statistics are computed in the same way
N_SHUFFLES = 20

def shuffle_within_sessions(skips, mask, rng):
    """Shuffle the tracks within each session"""
    # Give each track a random key and sort the keys of each row
    # Empty cells get the key +inf, so they stay at the end of the row (pour le mask)
    keys = rng.random(skips.shape)
    keys = np.where(mask, keys, np.inf)   # where a track exists, keep its random key while in empty cells, put +infinity
    order = np.argsort(keys, axis=1)    

    return np.take_along_axis(skips, order, axis=1)


def shuffled_gap(skips, mask, rng, n_shuffles=N_SHUFFLES):
    """Return the gap after a skip, as a mean over several shuffles"""
    # Shuffling destroys the order inside sessions but keeps the differences between sessions, so this is the part of the gap due to session differences -> to know if it is about the session of the order or tracks wihtin a sesh
    gaps = []
    for i in range(n_shuffles):
        # mix the tracks inside each session (one new random order each time)
        shuffled = shuffle_within_sessions(skips, mask, rng)
        # measure the gap after a skip on the shuffled table
        gaps.append(gap_after_skip(shuffled, mask))
    return np.mean(gaps)


def within_session_gap(skips, mask, rng, n_shuffles=N_SHUFFLES):
    """Return the gap after a skip minus the gap after shuffling"""
    # To check if the gap after skip comes from within the session of from the session itself
    return gap_after_skip(skips, mask) - shuffled_gap(skips, mask, rng, n_shuffles)


def mean_switches(skips, mask):
    """Return the mean number of skip/play changes per session"""
    changes = (skips[:, 1:] != skips[:, :-1]) & mask[:, 1:]
    return changes.sum(axis=1).mean()


def all_checks(skips, mask, rng, n_shuffles=N_SHUFFLES):
    """Return the three session checks, the shuffled gap and two within-session statistics"""
    gap = gap_after_skip(skips, mask)
    shuffled = shuffled_gap(skips, mask, rng, n_shuffles)
    return {
        **session_checks(skips, mask),
        "gap after shuffling": shuffled,
        "within-session gap": gap - shuffled,
        "skip/play changes": mean_switches(skips, mask),
    }


def real_data_checks(skips, mask, seed=2026, n_shuffles=N_SHUFFLES):
    """Return all_checks on the real data with a fixed seed, the same in every layer"""
    return all_checks(skips, mask, np.random.default_rng(seed), n_shuffles)
