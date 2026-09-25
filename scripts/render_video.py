"""Render the LinkedIn video: all 138,639 neurons at their real positions, reading one problem.

Every neuron's activity in the video is the simulation's own output for the input shown, and
the fly's answer is what the saved readout says. Nothing is animated by hand except the camera
and the captions.

    python scripts/render_video.py            # media/fly_brain_leetcode.mp4 and media/poster.png
    python scripts/render_video.py --preview  # a few stills in media/preview/
"""
import argparse
import json
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain import connectome as cx  # noqa: E402
from flybrain.brain import BrainConfig, FlyBrain  # noqa: E402
from flybrain.dataset import RESULTS  # noqa: E402
from flybrain.problems import BY_NUMBER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "media"
W, H, FPS = 1080, 1350, 30
INPUT = "(())))((()"

BG = np.array([9, 12, 11], float)
BASE = np.array([120, 150, 140], float) / 255   # resting neurons
POS = np.array([240, 176, 70], float) / 255     # positive activity, amber
NEG = np.array([90, 150, 225], float) / 255     # negative activity, blue
HOT = np.array([255, 245, 225], float) / 255    # sensory neurons being stimulated
DESC = np.array([235, 90, 150], float) / 255    # descending neurons at readout
INK, MUTED = (232, 238, 234), (150, 165, 158)
RED, GREEN, AMBER, SLATE = (242, 100, 94), (60, 203, 108), (227, 169, 69), (134, 166, 198)


def font(name, size, index=0):
    for path in [f"/System/Library/Fonts/{name}", f"/System/Library/Fonts/Supplemental/{name}"]:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


F_BIG = font("Avenir Next.ttc", 64, 2)
F_MED = font("Avenir Next.ttc", 40, 2)
F_TXT = font("Avenir Next.ttc", 32, 0)
F_SMALL = font("Avenir Next.ttc", 26, 0)
F_MONO = font("Menlo.ttc", 44)
F_MONO_S = font("Menlo.ttc", 28)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def positions():
    """A point on each neuron, in the simulation's neuron order."""
    import pandas as pd
    ids = pd.read_csv(cx.DATA / "Completeness_783.csv", index_col=0).index
    ann = pd.read_csv(cx.DATA / "neuron_annotations.tsv", sep="\t", usecols=["root_id", "pos_x", "pos_y", "pos_z"])
    return ann.drop_duplicates("root_id").set_index("root_id").reindex(ids).to_numpy(float)


def simulate():
    """Run the real brain on INPUT, recording every neuron, and read the fly's answer."""
    p = BY_NUMBER[20]
    brain = FlyBrain.build(BrainConfig(gain=4.0))
    symbols = brain.symbol_map(p.vocab, seed=p.number)
    feats, traj = brain.run([list(INPUT)], symbols, record=np.arange(cx.N_NEURONS))
    r = np.load(RESULTS / "readouts.npz")
    z = np.hstack([(feats["descending"] - r["20_mu"]) / r["20_sd"], np.ones((1, 1))]) @ r["20_B"]
    answer = p.classes[int(z.argmax())]
    return {
        "traj": traj[:, :, 0].astype(np.float32),                 # (steps, neurons)
        "symbols": {s: v for s, v in symbols.items()},
        "desc": brain.readouts["descending"],
        "answer": answer, "expected": p.solve(INPUT),
        "pos": positions(),
        "per": brain.cfg.on_steps + brain.cfg.off_steps, "on": brain.cfg.on_steps,
    }


class Camera:
    def __init__(self, pos):
        # FlyWire coordinates are 4 nm voxels in x and y and 40 nm sections in z.
        p = pos * np.array([4, 4, 40]) / 1000.0
        p -= np.nanmedian(p, 0)
        p[:, 1] *= -1
        self.p = np.nan_to_num(p)
        self.scale = 0.92 * W / (np.percentile(self.p[:, 0], 99.8) - np.percentile(self.p[:, 0], 0.2))
        self.cx, self.cy = W / 2, 640

    def project(self, yaw, pitch):
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        x, y, z = self.p.T
        x1, z1 = cy * x + sy * z, -sy * x + cy * z
        y2, z2 = cp * y - sp * z1, sp * y + cp * z1
        persp = 1 + z2 / 2500.0
        u = (self.cx + x1 * self.scale * persp).astype(int)
        v = (self.cy - y2 * self.scale * persp).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return u, v, ok, z2


def splat(u, v, ok, weight, color):
    idx = v[ok] * W + u[ok]
    img = np.empty((H, W, 3))
    for c in range(3):
        img[:, :, c] = np.bincount(idx, weights=weight[ok] * color[c], minlength=W * H).reshape(H, W)
    return img


def render_brain(cam, t, act, base_level, hot=None, desc=None, desc_level=0.0):
    yaw = 0.55 * np.sin(2 * np.pi * t / 26.0)
    pitch = 0.18 + 0.08 * np.sin(2 * np.pi * t / 17.0)
    u, v, ok, depth = cam.project(yaw, pitch)
    fade = 0.75 + 0.25 * np.clip(-depth / 300, -1, 1)
    img = splat(u, v, ok, base_level * 0.11 * fade, BASE)
    img += gaussian_filter(img, (3, 3, 0)) * 2.5
    if act is not None:
        a = np.clip(act, -1, 1)
        glow = splat(u, v, ok, np.maximum(a, 0) * 0.30 * fade, POS) + splat(u, v, ok, np.maximum(-a, 0) * 0.30 * fade, NEG)
        img += glow * 2 + gaussian_filter(glow, (3, 3, 0)) * 5 + gaussian_filter(glow, (14, 14, 0)) * 14
    if hot is not None and len(hot):
        w = np.zeros(len(u))
        w[hot] = 3.0
        h = splat(u, v, ok, w, HOT)
        img += h + gaussian_filter(h, (4, 4, 0)) * 25 + gaussian_filter(h, (18, 18, 0)) * 120
    if desc is not None and desc_level > 0:
        w = np.zeros(len(u))
        w[desc] = desc_level * 2.5
        d = splat(u, v, ok, w, DESC)
        img += d + gaussian_filter(d, (4, 4, 0)) * 20 + gaussian_filter(d, (14, 14, 0)) * 60
    out = BG / 255 + (1 - np.exp(-img * 1.1))
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def text(d, xy, s, f, fill, anchor="la", alpha=1.0):
    if alpha <= 0:
        return
    col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(fill, BG))
    d.text(xy, s, font=f, fill=col, anchor=anchor)


def input_line(d, y, active, alpha, done=False):
    """Draw s = "..." with the current character lit."""
    prefix = 's = "'
    widths = [d.textlength(ch, font=F_MONO) for ch in INPUT]
    total = d.textlength(prefix, font=F_MONO) + sum(widths) + d.textlength('"', font=F_MONO)
    x = (W - total) / 2
    text(d, (x, y), prefix, F_MONO, MUTED, alpha=alpha)
    x += d.textlength(prefix, font=F_MONO)
    for i, ch in enumerate(INPUT):
        col = HOT_TXT if i == active else (INK if (done or i < (active if active is not None else 0)) else MUTED)
        text(d, (x, y), ch, F_MONO, col, alpha=alpha)
        x += widths[i]
    text(d, (x, y), '"', F_MONO, MUTED, alpha=alpha)


HOT_TXT = (255, 214, 120)


def scoreboard(d, results, alpha, y0=980):
    probs = sorted(results.values(), key=lambda p: p["number"])
    x0, x1 = 120, W - 120
    text(d, (x0, y0 - 56), "hidden tests passed, per problem", F_SMALL, MUTED, alpha=alpha)
    text(d, (x1 - 150, y0 - 56), "real", F_SMALL, AMBER, "ra", alpha=alpha)
    text(d, (x1, y0 - 56), "shuffled", F_SMALL, SLATE, "ra", alpha=alpha)
    for i, p in enumerate(probs):
        y = y0 + i * 22
        f = p["scores"]["fly_descending"] / p["n_test"]
        r = p["scores"]["rewired_descending"] / p["n_test"]
        c = lambda col: tuple(int(v * alpha + b * (1 - alpha)) for v, b in zip(col, BG))
        d.line([(x0, y + 8), (x1, y + 8)], fill=c((40, 50, 46)), width=2)
        for val, col in ((r, SLATE), (f, AMBER)):
            x = x0 + val * (x1 - x0)
            d.ellipse([x - 7, y + 1, x + 7, y + 15], fill=c(col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    MEDIA.mkdir(exist_ok=True)
    sim = simulate()
    print("fly answer", sim["answer"], "expected", sim["expected"])
    results = json.loads((RESULTS / "results.json").read_text())
    p20 = results["20"]
    wins = sum(p["scores"]["rewired_descending"] > p["scores"]["fly_descending"] for p in results.values())
    cam = Camera(sim["pos"])
    traj, per, on = sim["traj"], sim["per"], sim["on"]
    steps = len(traj)
    sym_steps = len(INPUT) * per

    # Timeline in seconds.
    T_IN, T_Q, T_SIM0 = 3.5, 6.5, 7.0
    T_SIM1 = T_SIM0 + 11.0
    T_READ, T_VERDICT, T_TWIST, T_END = T_SIM1 + 0.3, T_SIM1 + 3.2, T_SIM1 + 7.6, T_SIM1 + 12.6
    total = T_END + 3.5

    def activity_at(t):
        s = (t - T_SIM0) / (T_SIM1 - T_SIM0) * (steps - 1)
        if s <= 0:
            return None, None, s
        s = min(s, steps - 1)
        i = int(np.floor(s))
        j = min(i + 1, steps - 1)
        a = traj[i] * (1 - (s - i)) + traj[j] * (s - i)
        return a, i, s

    def frame(t):
        base = ease(t / 2.5)
        act, step, s = activity_at(t)
        hot, active_char = None, None
        if step is not None and step < sym_steps:
            k, within = divmod(step, per)
            active_char = k
            if within < on:
                hot = sim["symbols"][INPUT[k]]
        desc_level = ease((t - T_READ) / 0.8) * (1 - ease((t - T_TWIST) / 1.0))
        dim = 1 - 0.55 * desc_level
        if act is not None:
            act = act * dim * (1 - 0.6 * ease((t - T_TWIST) / 1.5))
        img = Image.fromarray(render_brain(cam, t, act, base * dim, hot, sim["desc"], desc_level))
        d = ImageDraw.Draw(img)

        # Opening caption.
        a = ease(t / 1.2) * (1 - ease((t - T_Q + 0.5) / 0.6))
        text(d, (W / 2, 120), "138,639 neurons", F_BIG, INK, "ma", alpha=a)
        text(d, (W / 2, 200), "the complete wiring of a fruit fly brain", F_TXT, MUTED, "ma", alpha=a * ease((t - 0.8) / 1))
        a2 = ease((t - 3.0) / 1) * (1 - ease((t - T_Q + 0.5) / 0.6))
        text(d, (W / 2, 1180), "I never changed a single connection.", F_TXT, INK, "ma", alpha=a2)

        # The problem and the input.
        a = ease((t - T_Q) / 0.6) * (1 - ease((t - T_TWIST) / 0.6))
        text(d, (W / 2, 96), "LeetCode 20. Valid Parentheses", F_MED, INK, "ma", alpha=a)
        input_line(d, 162, active_char, a, done=step is not None and step >= sym_steps)
        if T_SIM0 <= t < T_READ:
            if step is not None and step < sym_steps:
                cap = f'symbol {active_char + 1} of {len(INPUT)}: "{INPUT[active_char]}" goes in through 40 sensory neurons'
            else:
                cap = "the brain settles"
            text(d, (W / 2, 1180), cap, F_SMALL, MUTED, "ma", alpha=ease((t - T_SIM0) / 0.5))
            text(d, (W / 2, 1222), f"simulation step {min(int(max(s, 0)) + 1, steps)} of {steps}", F_MONO_S, MUTED, "ma",
                 alpha=ease((t - T_SIM0) / 0.5))

        # Readout.
        if T_READ <= t < T_TWIST:
            a = ease((t - T_READ) / 0.6) * (1 - ease((t - T_TWIST + 0.6) / 0.6))
            text(d, (W / 2, 1150), "reading 1,303 descending neurons", F_TXT, (240, 140, 185), "ma", alpha=a)
            a3 = ease((t - T_READ - 1.0) / 0.5) * (1 - ease((t - T_TWIST + 0.6) / 0.6))
            text(d, (W / 2, 1200), f"the fly says: {str(sim['answer']).lower()}", F_MONO, INK, "ma", alpha=a3)

        # Verdict card, drawn over the brain.
        if T_VERDICT <= t < T_TWIST:
            a = ease((t - T_VERDICT) / 0.4) * (1 - ease((t - T_TWIST + 0.6) / 0.6))
            box = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            bd = ImageDraw.Draw(box)
            bd.rounded_rectangle([140, 470, W - 140, 890], 22, fill=(16, 22, 19, int(235 * a)), outline=(52, 64, 58, int(255 * a)), width=2)
            img.paste(Image.alpha_composite(img.convert("RGBA"), box).convert("RGB"))
            d = ImageDraw.Draw(img)
            wrong = sim["answer"] != sim["expected"]
            text(d, (190, 510), "Wrong Answer" if wrong else "Accepted", F_BIG, RED if wrong else GREEN, alpha=a)
            n, k = p20["n_test"], p20["scores"]["fly_descending"]
            text(d, (190, 600), f"{k} / {n} testcases passed", F_MONO_S, INK, alpha=a)
            d.rounded_rectangle([190, 648, W - 190, 658], 5, fill=tuple(int(c * a + b * (1 - a)) for c, b in zip((80, 36, 34), BG)))
            d.rounded_rectangle([190, 648, 190 + (W - 380) * k / n, 658], 5, fill=tuple(int(c * a + b * (1 - a)) for c, b in zip(GREEN, BG)))
            text(d, (190, 700), "Output", F_SMALL, MUTED, alpha=a)
            text(d, (360, 694), str(sim["answer"]).lower(), F_MONO_S, RED, alpha=a)
            text(d, (190, 760), "Expected", F_SMALL, MUTED, alpha=a)
            text(d, (360, 754), str(sim["expected"]).lower(), F_MONO_S, INK, alpha=a)
            text(d, (190, 820), "Submitted by Drosophila melanogaster", F_SMALL, MUTED, alpha=a)

        # The twist.
        if T_TWIST <= t < T_END:
            a = ease((t - T_TWIST) / 0.7) * (1 - ease((t - T_END + 0.6) / 0.6))
            text(d, (W / 2, 110), "Then I shuffled its wiring.", F_BIG, INK, "ma", alpha=a)
            a2 = ease((t - T_TWIST - 1.2) / 0.7) * (1 - ease((t - T_END + 0.6) / 0.6))
            text(d, (W / 2, 200), "Same neurons, same synapse weights,", F_TXT, MUTED, "ma", alpha=a2)
            text(d, (W / 2, 244), "connections landing in random places.", F_TXT, MUTED, "ma", alpha=a2)
            a3 = ease((t - T_TWIST - 2.4) / 0.7) * (1 - ease((t - T_END + 0.6) / 0.6))
            text(d, (W / 2, 830), f"The shuffled brain did better on {wins} of 12.", F_MED, AMBER, "ma", alpha=a3)
            scoreboard(d, results, a3)

        # End card.
        if t >= T_END:
            a = ease((t - T_END) / 0.8)
            text(d, (W / 2, 120), "Fly Brain LeetCode", F_BIG, INK, "ma", alpha=a)
            text(d, (W / 2, 200), "a real connectome, judged on hidden test cases", F_TXT, MUTED, "ma", alpha=a)
            text(d, (W / 2, 1180), "FlyWire v783 · 138,639 neurons · 2.7M connections", F_SMALL, MUTED, "ma", alpha=a)
        return np.asarray(img)

    if args.preview:
        out = MEDIA / "preview"
        out.mkdir(exist_ok=True)
        for t in [2.0, 5.5, T_SIM0 + 1.0, T_SIM0 + 6.0, T_READ + 1.8, T_VERDICT + 1.5, T_TWIST + 3.5, T_END + 2]:
            Image.fromarray(frame(t)).save(out / f"t{t:05.1f}.png")
        print("wrote", out)
        return

    n = int(total * FPS)
    writer = imageio_ffmpeg.write_frames(str(MEDIA / "fly_brain_leetcode.mp4"), (W, H), fps=FPS, codec="libx264",
                                         quality=None, bitrate=None,
                                         output_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow", "-movflags", "+faststart"])
    writer.send(None)
    for i in range(n):
        f = frame(i / FPS)
        writer.send(np.ascontiguousarray(f))
        if i == int((T_VERDICT + 1.5) * FPS):
            Image.fromarray(f).save(MEDIA / "poster.png")
        if i % 60 == 0:
            print(f"frame {i}/{n}", flush=True)
    writer.close()
    print("wrote", MEDIA / "fly_brain_leetcode.mp4")


if __name__ == "__main__":
    main()
