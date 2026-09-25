"""Is the rewired brain beating the real one a fluke of one seed and one gain?

For two problems that need memory of order, run three real-fly seeds (different sensory
neurons per symbol) and three different rewirings, each at three gains. Each brain gets the
gain that does best on its own validation set, and only then is it scored on the hidden tests.

Uses the v1 label-only readout. Writes results/v1/robust.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain import connectome as cx  # noqa: E402
from flybrain.brain import BrainConfig, FlyBrain, fit_readout  # noqa: E402
from flybrain.dataset import RESULTS as OUT, make_data  # noqa: E402
from flybrain.problems import BY_NUMBER, label_of  # noqa: E402

PROBLEMS = [20, 121]
GAINS = [2.0, 4.0, 8.0]
SEEDS = [0, 1, 2]


def main():
    labels, edges = cx.load_labels(), cx.load_edges()
    path = OUT / "v1" / "robust.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    data = {n: make_data(BY_NUMBER[n], seed=BY_NUMBER[n].number) for n in PROBLEMS}
    for kind in ["fly", "rewired"]:
        for seed in SEEDS:
            brain = FlyBrain.build(BrainConfig(rewired=kind == "rewired", seed=seed), labels, edges)
            for n in PROBLEMS:
                key = f"{n}/{kind}/{seed}"
                if key in res:
                    continue
                p, d, t0 = BY_NUMBER[n], data[n], time.time()
                y = {k: np.array([label_of(p, p.solve(x)) for x in v]) for k, v in d.items()}
                symbols = brain.symbol_map(p.vocab, seed=1000 * seed + n)
                by_gain = []
                for g in GAINS:
                    brain.cfg.gain = g
                    X = {k: brain.run([p.tokens(x) for x in v], symbols)["descending"] for k, v in d.items()}
                    val, lam, m = fit_readout(X["train"], y["train"], X["val"], y["val"], len(p.classes))
                    by_gain.append({"gain": g, "val": float(val), "test": float((m.predict(X["test"]) == y["test"]).mean())})
                best = max(by_gain, key=lambda r: r["val"])
                res[key] = {"best": best, "by_gain": by_gain}
                path.write_text(json.dumps(res, indent=2))
                print(f"{key:18s} gain {best['gain']}  test {best['test']:.3f}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
