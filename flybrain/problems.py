"""LeetCode problems, scaled down to inputs a fly can be shown one symbol at a time.

Each problem has a generator for inputs, a tokenizer that turns an input into the stream of
symbols the fly smells or feels, and the ordinary reference solution that decides the right
answer. The reference solutions are written the way you would write them on LeetCode.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

DIGITS = [str(d) for d in range(10)]


@dataclass
class Problem:
    number: int
    slug: str
    title: str
    difficulty: str
    vocab: list          # every symbol that can appear in the token stream
    classes: list        # every possible answer
    sample: Callable     # rng -> input
    tokens: Callable     # input -> list of symbols
    solve: Callable      # input -> answer (the reference solution)
    show: Callable = repr  # input -> how LeetCode would print it

    @property
    def name(self):
        return f"{self.number}. {self.title}"


def coin(rng):
    return rng.random() < 0.5


# 1. Two Sum -------------------------------------------------------------------------------

def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i


def sample_two_sum(rng):
    while True:
        nums = [int(x) for x in rng.integers(0, 10, 4)]
        i, j = sorted(rng.choice(4, 2, replace=False))
        target = nums[i] + nums[j]
        if sum(nums[a] + nums[b] == target for a, b in combinations(range(4), 2)) == 1:
            return nums, target


# 9. Palindrome Number ---------------------------------------------------------------------

def is_palindrome_number(x):
    return str(x) == str(x)[::-1]


def sample_palindrome(rng):
    n = int(rng.integers(3, 7))
    if coin(rng):
        half = [str(rng.integers(1, 10))] + [str(d) for d in rng.integers(0, 10, (n + 1) // 2 - 1)]
        digits = half + half[: n // 2][::-1]
    else:
        while True:
            digits = [str(rng.integers(1, 10))] + [str(d) for d in rng.integers(0, 10, n - 1)]
            if digits != digits[::-1]:
                break
    return int("".join(digits))


# 20. Valid Parentheses --------------------------------------------------------------------

def is_valid(s):
    stack, pair = [], {")": "(", "]": "["}
    for ch in s:
        if ch in pair:
            if not stack or stack.pop() != pair[ch]:
                return False
        else:
            stack.append(ch)
    return not stack


def random_valid(rng, n):
    if n == 0:
        return ""
    k = 2 * int(rng.integers(0, n // 2))
    o, c = ("(", ")") if coin(rng) else ("[", "]")
    return o + random_valid(rng, k) + c + random_valid(rng, n - 2 - k)


def sample_parens(rng):
    n = 2 * int(rng.integers(1, 6))
    s = random_valid(rng, n)
    if coin(rng):
        return s
    while True:  # a near miss: one symbol changed or two swapped, so the fly cannot win on length alone
        t = list(s)
        if coin(rng):
            t[rng.integers(n)] = "([)]"[rng.integers(4)]
        else:
            a, b = rng.choice(n, 2, replace=False)
            t[a], t[b] = t[b], t[a]
        t = "".join(t)
        if not is_valid(t):
            return t


# 121. Best Time to Buy and Sell Stock -----------------------------------------------------

def max_profit(prices):
    best, low = 0, prices[0]
    for p in prices:
        low = min(low, p)
        best = max(best, p - low)
    return best


# 136. Single Number -----------------------------------------------------------------------

def single_number(nums):
    x = 0
    for n in nums:
        x ^= n
    return x


def sample_single(rng):
    vals = [int(v) for v in rng.choice(6, 3, replace=False)]
    nums = [vals[0], vals[1], vals[1], vals[2], vals[2]]
    return [nums[i] for i in rng.permutation(5)]


# 169. Majority Element --------------------------------------------------------------------

def majority_element(nums):
    cand, count = None, 0
    for n in nums:
        if count == 0:
            cand = n
        count += 1 if n == cand else -1
    return cand


def sample_majority(rng):
    m = int(rng.integers(3))
    k = int(rng.integers(4, 8))
    others = [int(v) for v in rng.choice([v for v in range(3) if v != m], 7 - k)]
    nums = [m] * k + others
    return [nums[i] for i in rng.permutation(7)]


# 191. Number of 1 Bits --------------------------------------------------------------------

def hamming_weight(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


def sample_bits(rng):
    k = int(rng.integers(0, 9))  # uniform over the answer, not over n
    bits = [1] * k + [0] * (8 - k)
    return int("".join(str(bits[i]) for i in rng.permutation(8)), 2)


# 217. Contains Duplicate ------------------------------------------------------------------

def contains_duplicate(nums):
    return len(set(nums)) < len(nums)


def sample_dup(rng):
    if coin(rng):
        return [int(v) for v in rng.choice(10, 5, replace=False)]
    nums = [int(v) for v in rng.choice(10, 4, replace=False)]
    nums.append(nums[rng.integers(4)])
    return [nums[i] for i in rng.permutation(5)]


# 268. Missing Number ----------------------------------------------------------------------

def missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)


def sample_missing(rng):
    miss = int(rng.integers(7))
    nums = [v for v in range(7) if v != miss]
    return [nums[i] for i in rng.permutation(6)]


# 485. Max Consecutive Ones ----------------------------------------------------------------

def find_max_consecutive_ones(nums):
    best = run = 0
    for n in nums:
        run = run + 1 if n else 0
        best = max(best, run)
    return best


def sample_ones(rng):
    target = int(rng.integers(0, 9))
    while True:
        nums = [int(v) for v in rng.random(10) < rng.uniform(0.2, 0.9)]
        if find_max_consecutive_ones(nums) == target or rng.random() < 0.02:
            return nums


# 1822. Sign of the Product of an Array ----------------------------------------------------

def array_sign(nums):
    sign = 1
    for n in nums:
        if n == 0:
            return 0
        if n < 0:
            sign = -sign
    return sign


def sample_sign(rng):
    r = rng.random()
    if r < 1 / 3:
        nums = [int(v) for v in rng.choice([-2, -1, 0, 1, 2], 6)]
        nums[rng.integers(6)] = 0
    else:
        nums = [int(v) for v in rng.choice([-2, -1, 1, 2], 6)]
        if array_sign(nums) != (1 if r < 2 / 3 else -1):
            nums[0] = -nums[0]
    return nums


# 58. Length of Last Word ------------------------------------------------------------------

def length_of_last_word(s):
    return len(s.split()[-1])


def sample_last_word(rng):
    target = int(rng.integers(1, 7))
    tail = " " * int(rng.integers(0, 3))
    head_len = 10 - target - len(tail) - 1
    head = "".join("a" if rng.random() < 0.6 else " " for _ in range(head_len))
    return head + " " + "a" * target + tail


def num(nums):
    return [str(x) for x in nums]


PROBLEMS = [
    Problem(1, "two-sum", "Two Sum", "Easy", DIGITS + [f"t{t}" for t in range(19)],
            [[a, b] for a, b in combinations(range(4), 2)], sample_two_sum,
            lambda x: num(x[0]) + [f"t{x[1]}"], lambda x: two_sum(*x),
            lambda x: f"nums = {x[0]}, target = {x[1]}"),
    Problem(9, "palindrome-number", "Palindrome Number", "Easy", DIGITS, [True, False], sample_palindrome,
            lambda x: list(str(x)), is_palindrome_number, lambda x: f"x = {x}"),
    Problem(20, "valid-parentheses", "Valid Parentheses", "Easy", list("()[]"), [True, False], sample_parens,
            list, is_valid, lambda s: f's = "{s}"'),
    Problem(58, "length-of-last-word", "Length of Last Word", "Easy", ["a", " "], list(range(1, 7)),
            sample_last_word, list, length_of_last_word, lambda s: f's = "{s}"'),
    Problem(121, "best-time-to-buy-and-sell-stock", "Best Time to Buy and Sell Stock", "Easy", DIGITS[:7],
            list(range(7)), lambda rng: [int(v) for v in rng.integers(0, 7, 6)], num, max_profit,
            lambda x: f"prices = {x}"),
    Problem(136, "single-number", "Single Number", "Easy", DIGITS[:6], list(range(6)), sample_single, num,
            single_number, lambda x: f"nums = {x}"),
    Problem(169, "majority-element", "Majority Element", "Easy", DIGITS[:3], list(range(3)), sample_majority, num,
            majority_element, lambda x: f"nums = {x}"),
    Problem(191, "number-of-1-bits", "Number of 1 Bits", "Easy", ["0", "1"], list(range(9)), sample_bits,
            lambda n: list(format(n, "08b")), hamming_weight, lambda n: f"n = {n}"),
    Problem(217, "contains-duplicate", "Contains Duplicate", "Easy", DIGITS, [True, False], sample_dup, num,
            contains_duplicate, lambda x: f"nums = {x}"),
    Problem(268, "missing-number", "Missing Number", "Easy", DIGITS[:7], list(range(7)), sample_missing, num,
            missing_number, lambda x: f"nums = {x}"),
    Problem(485, "max-consecutive-ones", "Max Consecutive Ones", "Easy", ["0", "1"], list(range(11)), sample_ones,
            num, find_max_consecutive_ones, lambda x: f"nums = {x}"),
    Problem(1822, "sign-of-the-product-of-an-array", "Sign of the Product of an Array", "Easy",
            ["-2", "-1", "0", "1", "2"], [-1, 0, 1], sample_sign, num, array_sign, lambda x: f"nums = {x}"),
]

BY_NUMBER = {p.number: p for p in PROBLEMS}


def label_of(problem, answer):
    return problem.classes.index(answer)


def bag_of_tokens(problem, toks):
    c = Counter(toks)
    return [c[v] for v in problem.vocab]
