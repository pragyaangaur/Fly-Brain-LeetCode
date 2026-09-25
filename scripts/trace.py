"""Record what the brain does while it reads two parentheses strings, for the write-up figure.

Writes results/trace.json: per substep, mean activity of each super class, plus the
trajectories of 60 descending neurons for a valid and an invalid string.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import BrainConfig, FlyBrain  # noqa: E402
from flybrain.dataset import RESULTS as OUT  # noqa: E402
from flybrain.problems import BY_NUMBER  # noqa: E402



def main():
    p = BY_NUMBER[20]
    brain = FlyBrain.build(BrainConfig(gain=4.0))
    lab = brain.labels
    classes = ["sensory", "central", "optic", "visual_projection", "ascending", "descending", "motor"]
    groups = {c: lab["index"][lab.super_class == c].to_numpy() for c in classes}
    rng = np.random.default_rng(0)
    desc = np.sort(rng.choice(groups["descending"], 60, replace=False))
    record = np.concatenate([groups[c] for c in classes])
    bounds = np.cumsum([0] + [len(groups[c]) for c in classes])
    strings = ["([])()", "([)]()"]
    feats, traj = brain.run([list(s) for s in strings], brain.symbol_map(p.vocab, seed=p.number), record=record)
    a = np.abs(traj)  # (T, neurons, 2)
    out = {"strings": strings, "steps_per_symbol": brain.cfg.on_steps + brain.cfg.off_steps,
           "settle": brain.cfg.settle_steps, "classes": classes, "sizes": [len(groups[c]) for c in classes],
           "mean_activity": {c: a[:, bounds[i]:bounds[i + 1]].mean(1).T.round(4).tolist() for i, c in enumerate(classes)}}
    pos = {n: i for i, n in enumerate(record)}
    out["descending_traces"] = traj[:, [pos[n] for n in desc]].transpose(2, 1, 0).round(3).tolist()
    (OUT / "trace.json").write_text(json.dumps(out))
    print("wrote results/trace.json")


if __name__ == "__main__":
    main()
