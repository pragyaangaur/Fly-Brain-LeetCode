"""Add the order-blind ceiling to an existing results file, for runs made before it was built in.

    python scripts/order_blind.py results/v1/results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.baselines import order_blind  # noqa: E402
from flybrain.dataset import make_data  # noqa: E402
from flybrain.problems import BY_NUMBER  # noqa: E402


def main():
    path = Path(sys.argv[1])
    results = json.loads(path.read_text())
    for key, entry in results.items():
        p = BY_NUMBER[int(key)]
        entry["scores"]["order_blind"] = order_blind(p, make_data(p, seed=p.number))
        print(f"{p.name:40s} order-blind {entry['scores']['order_blind']}/{entry['n_test']}")
    path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
