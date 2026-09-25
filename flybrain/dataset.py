"""Train, validation and hidden test splits, decided by each input's own content."""
import hashlib
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_TRAIN, N_VAL, N_TEST = 8000, 1000, 1000


def split_of(problem, x):
    """Hash the input itself, so an input lives in exactly one of train, val or test.

    The hidden test cases are therefore inputs the fly never saw during training.
    """
    h = int(hashlib.sha256(f"{problem.number}:{problem.show(x)}".encode()).hexdigest(), 16) % 10
    return "train" if h < 7 else "val" if h == 7 else "test"


def make_data(problem, seed):
    rng = np.random.default_rng(seed)
    want = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
    data = {k: [] for k in want}
    seen_test = set()
    for _ in range(400_000):
        x = problem.sample(rng)
        s = split_of(problem, x)
        if len(data[s]) >= want[s]:
            if all(len(data[k]) >= want[k] for k in want):
                break
            continue
        if s == "test":  # LeetCode test cases are distinct
            key = problem.show(x)
            if key in seen_test:
                continue
            seen_test.add(key)
        data[s].append(x)
    return data
