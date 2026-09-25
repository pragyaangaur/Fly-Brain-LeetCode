"""Load the FlyWire v783 connectome and turn it into a signed, frozen recurrent weight matrix."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).resolve().parent.parent / "data"
N_NEURONS = 138_639
MIN_SYNAPSES = 5  # the usual FlyWire threshold for a real connection
# Non-visual senses. Photoreceptors are left out because they only reach the brain through the huge optic lobe.
INPUT_CLASSES = ["gustatory", "mechanosensory", "olfactory", "hygrosensory", "thermosensory", "unknown_sensory"]


def load_labels():
    """Annotations aligned to the simulation index (row i of Completeness_783.csv is neuron i)."""
    ids = pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index
    ann = pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_type", "side"])
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    ann["index"] = np.arange(len(ids))
    return ann


def load_edges(min_synapses=MIN_SYNAPSES):
    c = pd.read_parquet(DATA / "Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory x Connectivity"])
    c = c[c.Connectivity >= min_synapses]
    return (c.Presynaptic_Index.to_numpy(), c.Postsynaptic_Index.to_numpy(),
            c["Excitatory x Connectivity"].to_numpy().astype(np.float32))


def rewire(pre, post, w, seed=0):
    """Configuration-model control: shuffle which neuron each synapse lands on.

    Every neuron keeps its exact in-degree, out-degree, outgoing weights and sign.
    Only the specific wiring diagram is destroyed.
    """
    rng = np.random.default_rng(seed)
    return pre, rng.permutation(post), w


def weight_matrix(pre, post, w, device):
    """W[post, pre], with each row divided by that neuron's total absolute input.

    Normalising by input keeps every neuron's drive on the same scale regardless of how many
    synapses it receives, so one global gain sets the dynamical regime.
    """
    row_abs = np.bincount(post, weights=np.abs(w), minlength=N_NEURONS)
    wn = w / np.maximum(row_abs[post], 1e-9)
    idx = torch.tensor(np.stack([post, pre]), dtype=torch.int64)
    W = torch.sparse_coo_tensor(idx, torch.tensor(wn, dtype=torch.float32), (N_NEURONS, N_NEURONS)).coalesce()
    return W.to(device)


def spectral_radius(W, iters=200, device="cpu"):
    """Rough power-iteration estimate of the dominant eigenvalue magnitude."""
    x = torch.randn(N_NEURONS, 1, device=device)
    rho = 0.0
    for _ in range(iters):
        y = torch.sparse.mm(W, x)
        rho = y.norm().item() / x.norm().item()
        x = y / y.norm()
    return rho
