"""Train a readout on the frozen fly brain for every problem and judge it the way LeetCode does.

This is v2. For each problem it chooses, on the validation split only, the gain (2 or 4), the
readout head (a label, or a number that is rounded) and the ridge penalty. It then scores the
hidden test cases once. The shuffled brain goes through exactly the same search.

    python run.py                 # every problem
    python run.py --only 20 191   # a subset

Writes results/results.json and results/readouts.npz.
"""
import argparse
import json
import time

import numpy as np

from flybrain import connectome as cx
from flybrain.baselines import order_blind
from flybrain.brain import BrainConfig, FlyBrain
from flybrain.dataset import RESULTS as OUT, make_data
from flybrain.problems import PROBLEMS, label_of
from flybrain.readout import LAMBDAS, RidgePath, decode, heads_for, targets

GAINS = (2.0, 4.0)


def verdict(passed, total):
    return "Accepted" if passed == total else "Wrong Answer"


def search(brain, p, seqs, y):
    """Try every gain, head and penalty. Return the best on validation and a log of all gains."""
    best, log = None, []
    for gain in GAINS:
        brain.cfg.gain = gain
        X = {k: brain.run(v, brain_symbols(brain, p))["descending"] for k, v in seqs.items()}
        path = RidgePath(X["train"])
        for head in heads_for(p):
            T = targets(head, y["train"], p.classes)
            for lam in LAMBDAS:
                W, ym = path.weights(T, lam)
                val = (decode(head, path.scores(X["val"], W, ym), p.classes) == y["val"]).mean()
                if best is None or val > best["val"]:
                    pred = decode(head, path.scores(X["test"], W, ym), p.classes)
                    best = {"val": float(val), "gain": gain, "head": head, "lam": lam, "pred": pred, "X": X}
        log.append({"gain": gain, "best_val_so_far": best["val"]})
    return best, log


def brain_symbols(brain, p):
    return brain.symbol_map(p.vocab, seed=p.number)  # the same sensory neurons in both brains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    labels, edges = cx.load_labels(), cx.load_edges()
    brains = {
        "fly": FlyBrain.build(BrainConfig(), labels, edges),
        "rewired": FlyBrain.build(BrainConfig(rewired=True), labels, edges),
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
        entry = {"number": p.number, "title": p.title, "slug": p.slug, "difficulty": p.difficulty,
                 "n_train": len(y["train"]), "n_test": len(y["test"]),
                 "n_train_distinct": len({p.show(x) for x in data["train"]}),
                 "chance": float((y["test"] == np.bincount(y["train"], minlength=C).argmax()).mean()),
                 "scores": {"order_blind": order_blind(p, data)}, "chosen": {}}

        for bname, brain in brains.items():
            best, _ = search(brain, p, seqs, y)
            pred = best["pred"]
            entry["scores"][f"{bname}_descending"] = int((pred == y["test"]).sum())
            entry["chosen"][bname] = {k: best[k] for k in ("gain", "head", "lam", "val")}
            if bname != "fly":
                continue
            # Refit the chosen readout on train and val together for the saved model the demo uses.
            X, head = best["X"], best["head"]
            full = RidgePath(np.vstack([X["train"], X["val"]]))
            W, ym = full.weights(targets(head, np.concatenate([y["train"], y["val"]]), p.classes), best["lam"])
            n = p.number
            readouts.update({f"{n}_W": W, f"{n}_ym": ym, f"{n}_mu": full.mu, f"{n}_sd": full.sd,
                             f"{n}_gain": np.array(best["gain"]), f"{n}_head": np.array(head)})
            wrong, right = np.flatnonzero(pred != y["test"]), np.flatnonzero(pred == y["test"])
            entry["examples"] = {
                "wrong": [{"input": p.show(data["test"][i]), "expected": p.classes[y["test"][i]],
                           "fly": p.classes[pred[i]]} for i in wrong[:8]],
                "right": [{"input": p.show(data["test"][i]), "expected": p.classes[y["test"][i]]} for i in right[:8]],
            }
        entry["verdict"] = verdict(entry["scores"]["fly_descending"], entry["n_test"])
        entry["seconds"] = round(time.time() - t0)
        results[str(p.number)] = entry
        path.write_text(json.dumps(results, indent=2, default=str))
        np.savez(OUT / "readouts.npz", **readouts)
        s, n, c = entry["scores"], entry["n_test"], entry["chosen"]["fly"]
        print(f"{p.name:40s} fly {s['fly_descending']}/{n} ({c['head']}, gain {c['gain']})  "
              f"rewired {s['rewired_descending']}/{n}  order-blind {s['order_blind']}/{n}  "
              f"{entry['verdict']}  [{entry['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()
