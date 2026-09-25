"""The best any order-blind method can do on a problem's hidden tests.

For every test input, look up the training inputs with exactly the same symbol counts and answer
with their most common label. If the counts never appeared in training, guess the most common
label overall. Problems this scores near 100% on do not need memory of order at all.
"""
from collections import Counter, defaultdict

from .problems import bag_of_tokens, label_of


def order_blind(problem, data):
    table = defaultdict(Counter)
    for x in data["train"] + data["val"]:
        table[tuple(bag_of_tokens(problem, problem.tokens(x)))][label_of(problem, problem.solve(x))] += 1
    overall = Counter(label_of(problem, problem.solve(x)) for x in data["train"]).most_common(1)[0][0]
    hits = 0
    for x in data["test"]:
        c = table.get(tuple(bag_of_tokens(problem, problem.tokens(x))))
        guess = c.most_common(1)[0][0] if c else overall
        hits += guess == label_of(problem, problem.solve(x))
    return int(hits)
