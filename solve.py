"""Hand the fly one input and see what it answers.

    python solve.py 20 "([)]"
    python solve.py 136 "[4,1,2,1,2]"
    python solve.py 191 11

Needs results/readouts.npz from run.py.
"""
import ast
import sys
from pathlib import Path

import numpy as np

from flybrain.brain import BrainConfig, FlyBrain
from flybrain.problems import BY_NUMBER

ROOT = Path(__file__).resolve().parent


def parse(problem, raw):
    if problem.number in (20, 58):
        return raw
    x = ast.literal_eval(raw)
    return (x[0], x[1]) if problem.number == 1 else x


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    p = BY_NUMBER[int(sys.argv[1])]
    x = parse(p, sys.argv[2])
    toks = p.tokens(x)
    bad = [t for t in toks if t not in p.vocab]
    if bad:
        sys.exit(f"the fly was only trained on the symbols {p.vocab}, not {bad}")
    r = np.load(ROOT / "results" / "readouts.npz")
    B, mu, sd = r[f"{p.number}_B"], r[f"{p.number}_mu"], r[f"{p.number}_sd"]
    brain = FlyBrain.build(BrainConfig(gain=4.0))
    feats = brain.run([toks], brain.symbol_map(p.vocab, seed=p.number))["descending"]
    scores = np.hstack([(feats - mu) / sd, np.ones((1, 1))]) @ B
    answer = p.classes[int(scores.argmax())]
    truth = p.solve(x)
    print(f"{p.name}\n  input     {p.show(x)}\n  symbols   {' '.join(toks)}")
    print(f"  fly says  {answer}\n  expected  {truth}\n  {'correct' if answer == truth else 'wrong'}")


if __name__ == "__main__":
    main()
