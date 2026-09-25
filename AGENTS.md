# Fly Brain LeetCode

Context for anyone, human or AI, picking this up cold. Written 25 September 2026.

## What it is

The FlyWire v783 connectome of an adult fruit fly, run as a frozen recurrent network and asked to answer twelve scaled down LeetCode problems. Symbols go in through non-visual sensory neurons and answers are read from the 1,303 descending neurons with a ridge regression. The readout is the only trained part. It is a follow up to leetclaude and is meant as a portfolio piece with a LinkedIn post.

## Status

v2 is the current run, in `results/`. The first run is kept in `results/v1/` with `scripts/run_v1.py` and `scripts/robust.py`. The write-up page is `site/template.html`, built by `scripts/build_site.py`. The LinkedIn video is `scripts/render_screen.py`, a plain screen recording with the NeuroMechFly body, and `scripts/render_video.py` is an older whole-brain animation. Git ignores the videos in `media/` apart from `poster.png`. The LinkedIn draft is `linkedin_post.md`, which git also ignores. The repository is on GitHub as pragyaangaur/Fly-Brain-LeetCode.

## The results that matter

- v2 is Accepted on 3 of 12: Number of 1 Bits, Majority Element and Missing Number. v1 was Accepted on 1.
- v1 read every answer as a label, which cannot decode a count. v2 can read integer answers as a rounded number, and that alone took Number of 1 Bits from 3 of 59 to 59 of 59 with the same frozen brain.
- The nine failures need a stack, a pair comparison or a parity, and a linear readout of a frozen network does not expose those. Getting all twelve would need a nonlinear model on top or trained synapses, and neither would be the fly doing it. Do not add either and call it the fly.
- The shuffled brain ties the real fly on the three solved problems and beats it on the other nine. In v1, `scripts/robust.py` confirmed the shuffle result with three seeds per brain.
- On Two Sum, Length of Last Word, Best Time to Buy and Sell Stock and Max Consecutive Ones the fly beats the best order-blind method, so its dynamics do carry order.
- v2 chooses gain and head on validation, and on some problems that did worse on test than v1. Length of Last Word fell from 128 to 82 of 156. Keep that in the write-up.

## Traps

- Always split data with `flybrain/dataset.py`. Several problems have only a few hundred distinct inputs, and a random split lets the readout memorise them.
- `data/Completeness_783.csv` comes from the Eon Systems repo and `data/Connectivity_783.parquet` from the Shiu et al. repo. Their neuron order was checked to match on 25 September 2026.
- After moving the folder, `.venv/bin/pip` still points at the old path. Use `.venv/bin/python -m pip`.
- v1 fixed the gain at 4 from a pilot on separate data. v2 picks the gain on validation. Never tune anything on test results.
- The heatmap caption on the page says the two traces split within a few steps of the third symbol. That was checked against `results/trace.json`.
- The NeuroMechFly body in the video is posed by hand and its legs are animated. Only the neuron activity, answers and verdicts are real, and the README says so.
- `flygym` and `torch` share the venv. MuJoCo writes `MUJOCO_LOG.TXT` on a depth warning, and git ignores it.
