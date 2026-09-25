# Fly Brain LeetCode

<p align="center"><img src="media/poster.png" width="480"></p>

I took the complete wiring diagram of an adult fruit fly brain and made it answer LeetCode problems. The brain is FlyWire's v783 connectome, with 138,639 neurons and 2.7 million connections, and it is the same data behind the Eon Systems whole-brain emulation. Its wiring is never changed. Problems go in through its taste, touch and smell neurons one symbol at a time, and answers are read off the 1,303 descending neurons that carry commands from the brain to the body. The only thing trained is a linear readout on those neurons.

It was then judged the way LeetCode judges, on hidden test cases it had never seen. It got Accepted on one problem out of twelve, and that problem can be solved by counting symbols. The more interesting result came from the control. When I shuffled which neuron each connection lands on, the scrambled brain did better than the real one on 11 of the 12 problems.

## Results

Each cell is hidden test cases passed. "Shuffled" keeps every neuron, weight and sign but rewires where connections land. "Order-blind" is the best any method can score while ignoring the order of the symbols, computed with a lookup table on symbol counts. Chance is always guessing the most common answer.

| Problem | Fly | Shuffled | Order-blind | Chance | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1. Two Sum | 294/1000 | 344/1000 | 110/1000 | 146/1000 | Wrong Answer |
| 9. Palindrome Number | 818/1000 | 912/1000 | 864/1000 | 686/1000 | Wrong Answer |
| 20. Valid Parentheses | 637/1000 | 794/1000 | 675/1000 | 217/1000 | Wrong Answer |
| 58. Length of Last Word | 128/156 | 133/156 | 51/156 | 7/156 | Wrong Answer |
| 121. Best Time to Buy and Sell Stock | 742/1000 | 878/1000 | 595/1000 | 279/1000 | Wrong Answer |
| 136. Single Number | 254/330 | 330/330 | 330/330 | 52/330 | Wrong Answer |
| 169. Majority Element | 204/208 | 207/208 | 208/208 | 72/208 | Wrong Answer |
| 191. Number of 1 Bits | 3/59 | 12/59 | 59/59 | 0/59 | Wrong Answer |
| 217. Contains Duplicate | 814/1000 | 998/1000 | 1000/1000 | 495/1000 | Wrong Answer |
| 268. Missing Number | 1000/1000 | 1000/1000 | 1000/1000 | 145/1000 | Accepted |
| 485. Max Consecutive Ones | 151/211 | 181/211 | 102/211 | 76/211 | Wrong Answer |
| 1822. Sign of the Product of an Array | 646/1000 | 685/1000 | 999/1000 | 300/1000 | Wrong Answer |

## What the results say

**The fly's own wiring is not what helps.** The shuffled brain scored at least as well as the real one on every problem and better on 11. To check that this was not luck, `scripts/robust.py` reran Valid Parentheses and Best Time to Buy and Sell Stock with three real-fly seeds and three different shufflings, and let every brain pick its own best gain on validation data. Every shuffled brain beat every real one on both problems. The real fly averaged 65.8% and 71.2%, and the shuffled brains averaged 75.7% and 84.9%. The shuffled brain was also Accepted twice, on Missing Number and on Single Number, where the real fly passed 254 of 330. Whatever the fly's wiring is tuned for, it is not holding a stack of brackets.

**It does carry order.** On Two Sum, Length of Last Word, Best Time to Buy and Sell Stock and Max Consecutive Ones, the fly beats the best any order-blind method can do. Something in its dynamics remembers what came first.

**The first good number was memorisation.** A pilot run scored 80% on Number of 1 Bits. That pilot let the same inputs appear in training and testing, and the problem only has 256 possible inputs. Once inputs were split by content so that test cases were never seen in training, the fly passed 3 of 59. `run.py` hashes each input to decide its split for this reason, and it keeps LeetCode's convention that test cases are distinct.

**The one Accepted is the cheap one.** Missing Number is perfect for the fly, for the shuffled brain and for a symbol counter alike. It needs no memory of order.

## How it works

The brain is a rate model. At every substep each neuron updates as `r ← (1 − leak)·r + leak·tanh(gain·W·r + u)`. `W` holds synapse counts signed by predicted neurotransmitter, with each neuron's inputs normalised by its total absolute input so one global gain sets the regime. Connections with fewer than five synapses are dropped, which is the usual FlyWire threshold.

Each symbol in a problem's alphabet gets a fixed random group of 40 neurons drawn from the 5,577 non-visual sensory neurons. A symbol is shown for three substeps and followed by one silent substep, so `"11"` differs from `"1"`. After the last symbol the network settles for six substeps. A ridge regression then maps the descending neurons' activity to the answer, with its penalty chosen on a validation split.

The gain was fixed at 4 from a pilot run on separate data before the full run looked at any test case. The robustness run in `robust.py` later lets each brain pick its own gain from 2, 4 and 8 on validation data.

## Limits

It is a rate model, which is a simplification of the spiking model in Shiu et al. and at Eon Systems. At gain 4 most of the brain is active at once, which helps computation and is not how a fly brain behaves. The problems are scaled down to a few symbols each, so Two Sum uses four numbers from 0 to 9. The readout is trained per problem, so no single fly knows all twelve. The fly produces answers rather than code, so nothing here can be submitted to LeetCode itself.

## Running it

The trained readouts and all result files are in `results/`, so `solve.py` works straight after fetching the data. Everything else regenerates them.

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
./fetch_data.sh
.venv/bin/python solve.py 20 "([)]"          # ask the fly one question
.venv/bin/python run.py                      # train and judge all twelve, about 30 minutes on an M4
.venv/bin/python scripts/order_blind.py      # the order-blind ceiling
.venv/bin/python scripts/robust.py           # seeds and gains, about an hour
.venv/bin/python scripts/trace.py            # neuron traces for the page
.venv/bin/python scripts/build_site.py       # site/index.html from the result files
.venv/bin/python scripts/render_video.py     # media/fly_brain_leetcode.mp4
.venv/bin/python scripts/render_hook.py      # media/fly_brain_leetcode_hook.mp4
```

The simulation uses Apple's MPS backend when it is there and falls back to the CPU otherwise.

## Layout

```
flybrain/connectome.py      load FlyWire, sign and normalise weights, the shuffled control
flybrain/brain.py           the rate model and the ridge readout
flybrain/problems.py        twelve problems, their generators and reference solutions
flybrain/dataset.py         train, validation and hidden test splits by input content
run.py                      train and judge every problem, write results/results.json
solve.py                    ask the fly one question
scripts/order_blind.py      the lookup-table ceiling for any order-blind method
scripts/robust.py           seeds and gains for the shuffled-versus-real comparison
scripts/trace.py            neuron traces for the page figure
scripts/build_site.py       fill site/template.html with the result files
scripts/render_video.py     the long video of the brain reading a problem
scripts/render_hook.py      the short video with the NeuroMechFly body
results/                    result files and the saved readouts
```

## Credits

Connectome from the FlyWire Consortium (Dorkenwald et al. 2024, Schlegel et al. 2024, Nature). The packaged connectivity files and the Drosophila brain model are from Shiu et al. 2024, Nature, and the Eon Systems fly-brain repository. This is a follow up to [leetclaude](https://github.com/pragyaangaur/LeetClaude).
