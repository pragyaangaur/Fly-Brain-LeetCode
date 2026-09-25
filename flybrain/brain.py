"""The frozen fly brain as a recurrent network, and the only thing that is ever trained: a linear readout.

State update, one substep at a time:

    r <- (1 - leak) * r + leak * tanh(gain * W @ r + u)

W is the FlyWire wiring with synapse counts as weights and neurotransmitter signs. u is current
injected into the sensory neurons assigned to the current symbol. This is a rate model, a
deliberate simplification of the spiking model Shiu et al. and Eon Systems use.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from . import connectome as cx


@dataclass
class BrainConfig:
    gain: float = 1.0
    leak: float = 0.5
    amp: float = 2.0
    neurons_per_symbol: int = 40
    on_steps: int = 3     # substeps each symbol is presented
    off_steps: int = 1    # silent substeps after each symbol, so "11" differs from "1"
    settle_steps: int = 6  # substeps between the last symbol and reading the answer
    rewired: bool = False
    seed: int = 0


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class FlyBrain:
    cfg: BrainConfig
    labels: object = None
    W: object = None
    sensory: np.ndarray = None
    readouts: dict = field(default_factory=dict)

    @classmethod
    def build(cls, cfg, labels=None, edges=None):
        labels = labels if labels is not None else cx.load_labels()
        pre, post, w = edges if edges is not None else cx.load_edges()
        if cfg.rewired:
            pre, post, w = cx.rewire(pre, post, w, seed=cfg.seed + 1000)
        b = cls(cfg, labels)
        b.W = cx.weight_matrix(pre, post, w, device())
        b.sensory = labels.index[labels.cell_class.isin(cx.INPUT_CLASSES)].map(labels["index"]).to_numpy()
        rng = np.random.default_rng(cfg.seed)
        desc = labels["index"][labels.super_class == "descending"].to_numpy()
        # Control readout: as many neurons as the descending set, drawn from the central brain.
        central = labels["index"][labels.super_class == "central"].to_numpy()
        b.readouts = {"descending": desc, "random_central": np.sort(rng.choice(central, len(desc), replace=False))}
        return b

    def symbol_map(self, vocab, seed):
        """Each symbol stimulates its own fixed random group of sensory neurons."""
        rng = np.random.default_rng(seed)
        return {s: rng.choice(self.sensory, self.cfg.neurons_per_symbol, replace=False) for s in vocab}

    @torch.no_grad()
    def run(self, seqs, symbols, batch=1024, record=None):
        """Present each token sequence and return readout features {name: (n, k) array}.

        Shorter sequences start later so every sequence ends at the same moment. Starting from
        silence is exact here because the zero state is a fixed point when there is no input.
        If `record` is a list of neuron indices, also return their full trajectories.
        """
        c, dev = self.cfg, device()
        per = c.on_steps + c.off_steps
        T = max(len(s) for s in seqs)
        out = {k: [] for k in self.readouts}
        traj = []
        idx = {k: torch.tensor(v, device=dev) for k, v in self.readouts.items()}
        for b0 in range(0, len(seqs), batch):
            chunk = seqs[b0:b0 + batch]
            B = len(chunk)
            # Build the input schedule: drive[t] lists (neuron, batch) pairs to stimulate.
            drive = [[] for _ in range(T)]
            for j, s in enumerate(chunk):
                off = T - len(s)
                for t, sym in enumerate(s):
                    drive[off + t].append((symbols[sym], j))
            r = torch.zeros(cx.N_NEURONS, B, device=dev)
            rec = []
            for t in range(T):
                u = torch.zeros(cx.N_NEURONS, B, device=dev)
                if drive[t]:
                    rows = np.concatenate([n for n, _ in drive[t]])
                    cols = np.concatenate([np.full(len(n), j) for n, j in drive[t]])
                    u[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = c.amp
                for k in range(per):
                    inp = u if k < c.on_steps else 0.0
                    r = (1 - c.leak) * r + c.leak * torch.tanh(c.gain * torch.sparse.mm(self.W, r) + inp)
                    if record is not None:
                        rec.append(r[record].cpu().numpy())
            for _ in range(c.settle_steps):
                r = (1 - c.leak) * r + c.leak * torch.tanh(c.gain * torch.sparse.mm(self.W, r))
                if record is not None:
                    rec.append(r[record].cpu().numpy())
            for k in out:
                out[k].append(r[idx[k]].T.cpu().numpy())
            if record is not None:
                traj.append(np.stack(rec))
        feats = {k: np.concatenate(v) for k, v in out.items()}
        if record is not None:
            return feats, np.concatenate(traj, axis=2)
        return feats


class Readout:
    """Ridge regression from neuron activity onto one-hot answers. The only learned weights."""

    def fit(self, X, y, n_classes, lam):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        Y = np.eye(n_classes)[y]
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        self.B = np.linalg.solve(A, Z.T @ Y)
        return self

    def predict(self, X):
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return (Z @ self.B).argmax(1)


def fit_readout(Xtr, ytr, Xva, yva, n_classes, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    best = None
    for lam in lams:
        m = Readout().fit(Xtr, ytr, n_classes, lam)
        acc = (m.predict(Xva) == yva).mean()
        if best is None or acc > best[0]:
            best = (acc, lam, m)
    return best
