# Fly Brain LeetCode

Context for anyone, human or AI, picking this up cold. Written 25 September 2026.

## What it is

The FlyWire v783 connectome of an adult fruit fly, run as a frozen recurrent network and asked to answer twelve scaled down LeetCode problems. Symbols go in through non-visual sensory neurons and answers are read from the 1,303 descending neurons with a ridge regression. The readout is the only trained part. It is a follow up to leetclaude and is meant as a portfolio piece with a LinkedIn post.

## Status

The full run, the order-blind ceiling and the robustness run are done and their outputs are in `results/`. The write-up page is `site/template.html`, built by `scripts/build_site.py`. The video is rendered by `scripts/render_video.py` into `media/`, which git ignores apart from `poster.png`. The repository has not been pushed. The LinkedIn draft is `linkedin_post.md`, which git also ignores.

## The results that matter

- The fly was Accepted on one problem, Missing Number, which a symbol counter also solves.
- A degree-preserving shuffle of the wiring beat the real fly on 11 of 12 problems and tied on the twelfth. `scripts/robust.py` confirmed it on Valid Parentheses and Best Time to Buy and Sell Stock: with three seeds each and a per-brain gain, every shuffled brain beat every real one.
- On Two Sum, Length of Last Word, Best Time to Buy and Sell Stock and Max Consecutive Ones the fly beats the best order-blind method, so its dynamics do carry order.
- A pilot with overlapping train and test inputs showed 80% on Number of 1 Bits. With inputs split by content it is 3 of 59. Keep this in the write-up.

## Traps

- Always split data with `flybrain/dataset.py`. Several problems have only a few hundred distinct inputs, and a random split lets the readout memorise them.
- `data/Completeness_783.csv` comes from the Eon Systems repo and `data/Connectivity_783.parquet` from the Shiu et al. repo. Their neuron order was checked to match on 25 September 2026.
- After moving the folder, `.venv/bin/pip` still points at the old path. Use `.venv/bin/python -m pip`.
- The gain was fixed at 4 from a pilot on separate data before the full run. Do not retune it on test results.
- The heatmap caption on the page says the two traces split within a few steps of the third symbol. That was checked against `results/trace.json`.
