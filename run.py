"""Train a readout on the frozen fly brain for every problem and judge it the way LeetCode does.

    python run.py                 # every problem
    python run.py --only 20 191   # a subset

Writes results/results.json and results/readouts.npz.
"""
import argparse
import json
import time

import numpy as np

from flybrain import connectome as cx
from flybrain.brain import BrainConfig, FlyBrain, Readout, fit_readout
from flybrain.dataset import RESULTS as OUT, make_data
from flybrain.problems import PROBLEMS, bag_of_tokens, label_of

def verdict(passed, total):
    return "Accepted" if passed == total else "Wrong Answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--gain", type=float, default=4.0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    labels, edges = cx.load_labels(), cx.load_edges()
    brains = {
        "fly": FlyBrain.build(BrainConfig(gain=args.gain), labels, edges),
        "rewired": FlyBrain.build(BrainConfig(gain=args.gain, rewired=True), labels, edges),
    }
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    readouts = dict(np.load(OUT / "readouts.npz")) if (OUT / "readouts.npz").exists() else {}

    for p in PROBLEMS:
        if args.only and p.number not in args.only:
            continue
        t0 = time.time()
        data = make_data(p, seed=p.number)
        seqs = {k: [p.tokens(x) for x in v] for k, v in data.items()}
        y = {k: np.array([label_of(p, p.solve(x)) for x in v]) for k, v in data.items()}
        C = len(p.classes)
        symbols = brains["fly"].symbol_map(p.vocab, seed=p.number)  # same sensory neurons for both brains

        entry = {"number": p.number, "title": p.title, "slug": p.slug, "difficulty": p.difficulty,
                 "n_train": len(y["train"]), "n_test": len(y["test"]),
                 "n_train_distinct": len({p.show(x) for x in data["train"]}),
                 "chance": float((y["test"] == np.bincount(y["train"], minlength=C).argmax()).mean()),
                 "scores": {}}

        # No brain at all: a linear readout on how many times each symbol appeared.
        bag = {k: np.array([bag_of_tokens(p, s) for s in v], dtype=float) for k, v in seqs.items()}
        acc, lam, m = fit_readout(bag["train"], y["train"], bag["val"], y["val"], C)
        entry["scores"]["bag_of_tokens"] = int((m.predict(bag["test"]) == y["test"]).sum())

        for bname, brain in brains.items():
            feats = {k: brain.run(v, symbols) for k, v in seqs.items()}
            for rname in brain.readouts:
                if bname == "rewired" and rname != "descending":
                    continue
                X = {k: feats[k][rname] for k in feats}
                acc, lam, m = fit_readout(X["train"], y["train"], X["val"], y["val"], C)
                pred = m.predict(X["test"])
                key = f"{bname}_{rname}"
                entry["scores"][key] = int((pred == y["test"]).sum())
                if key == "fly_descending":
                    entry["lambda"] = lam
                    # Refit on train and val together for the saved model the demo uses.
                    full = Readout().fit(np.vstack([X["train"], X["val"]]),
                                         np.concatenate([y["train"], y["val"]]), C, lam)
                    readouts[f"{p.number}_B"], readouts[f"{p.number}_mu"], readouts[f"{p.number}_sd"] = full.B, full.mu, full.sd
                    wrong = np.flatnonzero(pred != y["test"])
                    right = np.flatnonzero(pred == y["test"])
                    entry["examples"] = {
                        "wrong": [{"input": p.show(data["test"][i]), "expected": p.classes[y["test"][i]],
                                   "fly": p.classes[pred[i]]} for i in wrong[:8]],
                        "right": [{"input": p.show(data["test"][i]), "expected": p.classes[y["test"][i]]}
                                  for i in right[:8]],
                    }
        entry["verdict"] = verdict(entry["scores"]["fly_descending"], entry["n_test"])
        entry["seconds"] = round(time.time() - t0)
        results[str(p.number)] = entry
        path.write_text(json.dumps(results, indent=2, default=str))
        np.savez(OUT / "readouts.npz", **readouts)
        s, n = entry["scores"], entry["n_test"]
        print(f"{p.name:40s} fly {s['fly_descending']}/{n}  rewired {s['rewired_descending']}/{n}  "
              f"central {s['fly_random_central']}/{n}  bag {s['bag_of_tokens']}/{n}  chance {entry['chance']:.2f}  "
              f"[{entry['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()
