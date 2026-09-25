"""Render the LinkedIn video as a plain screen recording: the fly on the left, a judge on the right.

Each problem shows one hidden test case, the answer the fly gave for it during the run, the real
verdict and pass count from results/results.json, and the activity of 96 of the fly's descending
neurons while it reads that input, simulated at the gain its readout was chosen for. The fly body
is the NeuroMechFly model posed by hand. Its tapping legs are animation and are not driven by the
brain.

    python scripts/render_screen.py            # media/fly_brain_leetcode.mp4
    python scripts/render_screen.py --preview  # a few stills in media/preview/
"""
import argparse
import json
import sys
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import BrainConfig, FlyBrain  # noqa: E402
from flybrain.dataset import RESULTS  # noqa: E402
from flybrain.problems import BY_NUMBER  # noqa: E402

MEDIA = Path(__file__).resolve().parent.parent / "media"
W, H, FPS = 1920, 1080, 30
SIM_W = 960
INTRO, PER, READ, OUTRO = 1.4, 1.55, 0.95, 3.0

BG, PANEL, LINE = (26, 26, 26), (40, 40, 40), (58, 58, 58)
TXT, DIM, FAINT = (239, 241, 246), (160, 164, 170), (110, 114, 120)
GREEN, RED, EASY = (44, 187, 93), (239, 71, 67), (0, 184, 163)


def font(name, size):
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default(size)


SANS = {s: font("SFNS.ttf", s) for s in (18, 20, 22, 26, 30)}
MONO = {s: font("SFNSMono.ttf", s) for s in (17, 20, 24)}

# Standing pose in radians per leg: coxa roll, coxa pitch, coxa yaw, femur roll, femur pitch, tibia pitch, tarsus pitch.
POSE = {
    "f": (0.4, 0.6, 0.0, 0.3, -2.1, 1.9, -0.5),
    "m": (1.3, 0.0, 0.0, 0.6, -1.6, 1.7, -0.5),
    "h": (1.1, -0.5, 0.0, 0.6, -1.25, 1.6, -0.4),
}
PARTS = ["c_thorax-{s}_coxa-roll", "c_thorax-{s}_coxa-pitch", "c_thorax-{s}_coxa-yaw",
         "{s}_coxa-{s}_trochanterfemur-roll", "{s}_coxa-{s}_trochanterfemur-pitch",
         "{s}_trochanterfemur-{s}_tibia-pitch", "{s}_tibia-{s}_tarsus1-pitch"]


class Fly:
    """NeuroMechFly standing on MuJoCo's default checker floor, posed kinematically."""

    def __init__(self):
        from flygym.anatomy import AxisOrder, ContactBodiesPreset, JointPreset, Skeleton
        from flygym.compose import FlatGroundWorld, KinematicPosePreset, NeuroMechFly
        from flygym.utils.math import Rotation3D
        fly = NeuroMechFly()
        fly.add_joints(Skeleton(axis_order=AxisOrder.ROLL_PITCH_YAW, joint_preset=JointPreset.ALL_BIOLOGICAL),
                       neutral_pose=KinematicPosePreset.NEUTRAL)
        fly.colorize()
        world = FlatGroundWorld()
        world.add_fly(fly, [0, 0, 0.7], Rotation3D(format="quat", values=[1, 0, 0, 0]),
                      bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
        self.m, self.d = world.compile()
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = SIM_W, H
        self.r = mujoco.Renderer(self.m, H, SIM_W)
        self.addr = {}
        for side in "lr":
            for leg in POSE:
                for part in PARTS:
                    name = "nmf/" + part.format(s=side + leg)
                    j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if j >= 0:
                        self.addr[name] = (self.m.jnt_qposadr[j], leg, part)
        self.tips = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"nmf/{s}{leg}_tarsus5")
                     for s in "lr" for leg in "mh"]
        self.pose(0.0, 0.0)
        self.center = self.d.xpos[1:].mean(0).copy()

    def pose(self, t, tap):
        self.d.qpos[:] = self.m.qpos0
        for name, (adr, leg, part) in self.addr.items():
            side = name[4]
            v = POSE[leg][PARTS.index(part)]
            if leg == "f" and ("femur-pitch" in part or "tibia-pitch" in part):
                phase = 0 if side == "l" else np.pi
                v += tap * 0.3 * max(0.0, np.sin(2 * np.pi * 4.0 * t + phase)) * (1 if "femur" in part else -1)
            self.d.qpos[adr] = v
        mujoco.mj_forward(self.m, self.d)
        self.d.qpos[2] -= self.d.xpos[self.tips][:, 2].min() - 0.02  # stand on the floor
        self.d.qpos[2] += 0.015 * np.sin(2 * np.pi * 0.7 * t)       # a little breathing sway
        mujoco.mj_forward(self.m, self.d)

    def render(self, t, tap):
        self.pose(t, tap)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.center
        cam.distance, cam.azimuth, cam.elevation = 6.2, 128 + 1.6 * t, -17
        self.r.update_scene(self.d, cam)
        return self.r.render()


def show(v):
    if isinstance(v, bool):
        return str(v).lower()
    return json.dumps(v) if isinstance(v, list) else str(v)


def pick_case(entry):
    """The test case LeetCode would show: the first failing one, or a passing one if all passed."""
    ex = entry["examples"]
    if ex["wrong"]:
        e = ex["wrong"][0]
        return e["input"], e["fly"], e["expected"]
    e = ex["right"][0]
    return e["input"], e["expected"], e["expected"]


def parse_input(p, shown):
    """Rebuild the problem input from its printed form, to feed the same case to the brain."""
    import ast
    body = shown.split(" = ", 1)[1]
    if p.number == 1:
        nums, target = shown.split(", target = ")
        return ast.literal_eval(nums.split(" = ", 1)[1]), int(target)
    if p.number in (20, 58):
        return ast.literal_eval(body)
    return ast.literal_eval(body)


def traces(results):
    """Descending-neuron activity for each problem's shown case, at the gain its readout used."""
    brain = FlyBrain.build(BrainConfig())
    rng = np.random.default_rng(0)
    desc = np.sort(rng.choice(brain.readouts["descending"], 96, replace=False))
    out = {}
    for key, entry in results.items():
        p = BY_NUMBER[int(key)]
        brain.cfg.gain = entry.get("chosen", {}).get("fly", {}).get("gain", 4.0)
        x = parse_input(p, pick_case(entry)[0])
        _, tr = brain.run([p.tokens(x)], brain.symbol_map(p.vocab, seed=p.number), record=desc)
        out[key] = (tr[:, :, 0], len(p.tokens(x)), brain.cfg.on_steps + brain.cfg.off_steps)
    return out


def raster(d, box, tr, frac):
    """Draw neurons as rows and simulation steps as columns, revealed up to `frac`."""
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(18, 18, 18))
    steps, n = tr.shape
    shown = int(np.ceil(frac * steps))
    tr = tr / (np.abs(tr).max(0, keepdims=True) + 1e-9)  # each neuron on its own colour scale, for display
    cw, rh = (x1 - x0) / steps, (y1 - y0) / n
    for s in range(shown):
        for i in range(n):
            v = float(np.clip(tr[s, i], -1, 1))
            if abs(v) < 0.02:
                continue
            c = (255, 176, 60) if v > 0 else (86, 156, 255)
            a = abs(v) ** 0.7
            col = tuple(int(18 + (ch - 18) * a) for ch in c)
            d.rectangle([x0 + s * cw, y0 + i * rh, x0 + (s + 1) * cw - 1, y0 + (i + 1) * rh], fill=col)


def judge_panel(img, entry, case, tr, t_local, counts):
    d = ImageDraw.Draw(img)
    X0 = SIM_W
    d.rectangle([X0, 0, W, H], fill=BG)
    d.line([(X0, 0), (X0, H)], fill=LINE, width=2)
    x = X0 + 48
    d.text((x, 34), "Problems", font=SANS[20], fill=DIM)
    d.text((x + 98, 34), "›", font=SANS[20], fill=FAINT)
    d.text((x + 120, 34), f"{entry['number']}. {entry['title']}", font=SANS[20], fill=TXT)
    ac, wa = counts
    d.text((W - 48, 34), f"✓ {ac}   ✗ {wa}", font=SANS[20], fill=DIM, anchor="ra")
    d.text((x, 96), f"{entry['number']}. {entry['title']}", font=SANS[30], fill=TXT)
    d.rounded_rectangle([x, 148, x + 58, 176], 14, fill=(30, 58, 54))
    d.text((x + 29, 162), "Easy", font=SANS[18], fill=EASY, anchor="mm")

    inp, out, exp = case
    y = 222
    d.rounded_rectangle([x, y, W - 48, y + 420], 10, fill=PANEL)
    d.text((x + 24, y + 20), "Testcase", font=SANS[20], fill=DIM)
    d.text((x + 24, y + 62), "Input", font=SANS[18], fill=FAINT)
    d.rounded_rectangle([x + 24, y + 90, W - 72, y + 144], 8, fill=(52, 52, 52))
    d.text((x + 42, y + 104), inp, font=MONO[24], fill=TXT)
    done = t_local >= READ
    if done:
        ok = out == exp
        d.text((x + 24, y + 168), "Output", font=SANS[18], fill=FAINT)
        d.rounded_rectangle([x + 24, y + 196, W - 72, y + 250], 8, fill=(52, 52, 52))
        d.text((x + 42, y + 210), show(out), font=MONO[24], fill=TXT if ok else RED)
        d.text((x + 24, y + 274), "Expected", font=SANS[18], fill=FAINT)
        d.rounded_rectangle([x + 24, y + 302, W - 72, y + 356], 8, fill=(52, 52, 52))
        d.text((x + 42, y + 316), show(exp), font=MONO[24], fill=TXT)
    else:
        dots = "." * (1 + int(t_local * 6) % 3)
        d.text((x + 24, y + 176), f"Running{dots}", font=SANS[22], fill=DIM)

    y = 670
    if done:
        passed, n = entry["scores"]["fly_descending"], entry["n_test"]
        acc = passed == n
        d.text((x, y), "Accepted" if acc else "Wrong Answer", font=SANS[30], fill=GREEN if acc else RED)
        d.text((x, y + 46), f"{passed} / {n} testcases passed", font=SANS[20], fill=DIM)
    y = 780
    d.text((x, y), "descending neurons, live", font=SANS[18], fill=FAINT)
    reading = min(t_local / READ, 1.0)
    raster(d, (x, y + 30, W - 48, H - 48), tr, reading)


def problem_list(img, results, statuses):
    d = ImageDraw.Draw(img)
    X0 = SIM_W
    d.rectangle([X0, 0, W, H], fill=BG)
    d.line([(X0, 0), (X0, H)], fill=LINE, width=2)
    x = X0 + 48
    d.text((x, 34), "Problems", font=SANS[20], fill=TXT)
    ac = sum(s == "ac" for s in statuses.values())
    d.text((W - 48, 34), f"Solved {ac} / {len(results)}", font=SANS[20], fill=DIM, anchor="ra")
    d.text((x + 60, 96), "Title", font=SANS[18], fill=FAINT)
    d.text((W - 230, 96), "Acceptance", font=SANS[18], fill=FAINT, anchor="ra")
    d.text((W - 60, 96), "Difficulty", font=SANS[18], fill=FAINT, anchor="ra")
    for i, (key, e) in enumerate(sorted(results.items(), key=lambda kv: int(kv[0]))):
        y = 132 + i * 62
        if i % 2 == 0:
            d.rectangle([x - 12, y, W - 36, y + 62], fill=(33, 33, 33))
        st = statuses.get(key)
        if st == "ac":
            d.text((x + 8, y + 31), "✓", font=SANS[26], fill=GREEN, anchor="mm")
        elif st == "wa":
            d.text((x + 8, y + 31), "○", font=SANS[22], fill=RED, anchor="mm")
        d.text((x + 60, y + 31), f"{e['number']}. {e['title']}", font=SANS[22], fill=TXT, anchor="lm")
        pct = 100 * e["scores"]["fly_descending"] / e["n_test"]
        if st:
            d.text((W - 230, y + 31), f"{pct:.1f}%", font=SANS[20], fill=DIM, anchor="rm")
        d.text((W - 60, y + 31), "Easy", font=SANS[20], fill=EASY, anchor="rm")


def hud(img, text_lines):
    d = ImageDraw.Draw(img)
    for i, s in enumerate(text_lines):
        d.text((22, 20 + i * 26), s, font=MONO[17], fill=(20, 20, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    MEDIA.mkdir(exist_ok=True)
    results = json.loads((RESULTS / "results.json").read_text())
    keys = sorted(results, key=int)
    cases = {k: pick_case(results[k]) for k in keys}
    tr = traces(results)
    fly = Fly()
    total = INTRO + PER * len(keys) + OUTRO

    def frame(t):
        k_idx = int((t - INTRO) // PER) if t >= INTRO else -1
        in_problems = 0 <= k_idx < len(keys)
        t_local = (t - INTRO) - k_idx * PER if in_problems else 0.0
        tap = 1.0 if in_problems and t_local < READ else 0.0
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(fly.render(t, tap)), (0, 0))
        statuses = {}
        for i, k in enumerate(keys):
            if i < k_idx or (i == k_idx and t_local >= READ) or k_idx >= len(keys):
                e = results[k]
                statuses[k] = "ac" if e["scores"]["fly_descending"] == e["n_test"] else "wa"
        if in_problems:
            key = keys[k_idx]
            p = BY_NUMBER[int(key)]
            trace, L, per = tr[key]
            step = min(int(t_local / READ * len(trace)), len(trace) - 1)
            sym = min(step // per, L - 1)
            toks = p.tokens(parse_input(p, cases[key][0]))
            counts = (sum(s == "ac" for s in statuses.values()), sum(s == "wa" for s in statuses.values()))
            judge_panel(img, results[key], cases[key], trace, t_local, counts)
            status = f"reading symbol {sym + 1}/{L}  '{toks[sym]}'" if t_local < READ else "readout: 1,303 descending neurons"
            hud(img, ["FlyWire v783  138,639 neurons  2.7M connections", f"step {step + 1}/{len(trace)}  {status}"])
        else:
            problem_list(img, results, statuses)
            hud(img, ["FlyWire v783  138,639 neurons  2.7M connections", "idle"])
        return np.asarray(img)

    if args.preview:
        out = MEDIA / "preview"
        out.mkdir(exist_ok=True)
        for t in [0.5, INTRO + 0.5, INTRO + 1.2, INTRO + PER * 7 + 1.2, total - 1.0]:
            Image.fromarray(frame(t)).save(out / f"screen_{t:05.2f}.png")
        print("wrote", out)
        return

    path = MEDIA / "fly_brain_leetcode.mp4"
    writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", quality=None, bitrate=None,
                                         output_params=["-crf", "20", "-pix_fmt", "yuv420p", "-preset", "slow",
                                                        "-movflags", "+faststart"])
    writer.send(None)
    n = int(total * FPS)
    poster = int((INTRO + PER * keys.index("191") + READ + 0.3) * FPS)  # Number of 1 Bits, Accepted
    for i in range(n):
        f = frame(i / FPS)
        writer.send(np.ascontiguousarray(f))
        if i == poster:
            Image.fromarray(f).resize((W // 2, H // 2), Image.LANCZOS).save(MEDIA / "poster.png")
        if i % 90 == 0:
            print(f"frame {i}/{n}", flush=True)
    writer.close()
    print("wrote", path)


if __name__ == "__main__":
    main()
