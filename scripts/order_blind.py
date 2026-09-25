"""The best any order-blind method can do on each problem's hidden tests.

For every test input, look up the training inputs with exactly the same symbol counts and
answer with their most common label. If the counts never appeared in training, guess the most
common label overall. Problems this scores near 100% on do not need memory of order at all.

Adds "order_blind" to each entry in results/results.json.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.dataset import RESULTS as OUT, make_data  # noqa: E402
from flybrain.problems import BY_NUMBER, bag_of_tokens, label_of  # noqa: E402


def main():
    path = OUT / "results.json"
    results = json.loads(path.read_text())
    for key, entry in results.items():
        p = BY_NUMBER[int(key)]
        d = make_data(p, seed=p.number)
        table = defaultdict(Counter)
        for x in d["train"] + d["val"]:
            table[tuple(bag_of_tokens(p, p.tokens(x)))][label_of(p, p.solve(x))] += 1
        overall = Counter(label_of(p, p.solve(x)) for x in d["train"]).most_common(1)[0][0]
        hits = 0
        for x in d["test"]:
            c = table.get(tuple(bag_of_tokens(p, p.tokens(x))))
            guess = c.most_common(1)[0][0] if c else overall
            hits += guess == label_of(p, p.solve(x))
        entry["scores"]["order_blind"] = hits
        print(f"{p.name:40s} order-blind {hits}/{entry['n_test']}   fly {entry['scores']['fly_descending']}")
    path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
