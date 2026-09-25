"""Render the short hook video: the fly, a dive into its brain, and twelve verdicts.

The brain activity is the simulation's real output for the Valid Parentheses input used in
render_video.py, and every verdict is read from results/results.json. The fly body is the
NeuroMechFly model posed by hand. Its leg tapping is animation and is not driven by the brain.

    python scripts/render_hook.py            # media/fly_brain_leetcode_hook.mp4
    python scripts/render_hook.py --preview  # a few stills in media/preview/
"""
import argparse
import json
import sys
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_video import (AMBER, BG, F_BIG, F_MED, F_SMALL, F_TXT, GREEN, H, INK, MEDIA, MUTED,  # noqa: E402
                          RED, RESULTS, W, Camera, ease, font, render_brain, simulate, text)

FPS = 30
F_HUGE = font("Avenir Next.ttc", 76, 2)
F_ROW = font("Avenir Next.ttc", 25, 0)
F_PILL = font("Avenir Next.ttc", 22, 2)

# Standing pose in radians per leg: coxa roll, coxa pitch, coxa yaw, femur roll, femur pitch, tibia pitch, tarsus pitch.
POSE = {
    "f": (0.4, 0.6, 0.0, 0.3, -2.1, 1.9, -0.5),
    "m": (1.3, 0.0, 0.0, 0.6, -1.6, 1.7, -0.5),
    "h": (1.1, -0.5, 0.0, 0.6, -1.5, 1.8, -0.5),
}
PARTS = ["c_thorax-{s}_coxa-roll", "c_thorax-{s}_coxa-pitch", "c_thorax-{s}_coxa-yaw",
         "{s}_coxa-{s}_trochanterfemur-roll", "{s}_coxa-{s}_trochanterfemur-pitch",
         "{s}_trochanterfemur-{s}_tibia-pitch", "{s}_tibia-{s}_tarsus1-pitch"]

# Timeline in seconds.
T_DIVE, T_BRAIN, T_LIST, T_BACK, T_END = 3.0, 4.3, 5.0, 10.4, 13.6


class Fly:
    def __init__(self):
        from flygym.anatomy import AxisOrder, JointPreset, Skeleton
        from flygym.compose import KinematicPosePreset, NeuroMechFly, TetheredWorld
        from flygym.utils.math import Rotation3D
        fly = NeuroMechFly()
        fly.add_joints(Skeleton(axis_order=AxisOrder.ROLL_PITCH_YAW, joint_preset=JointPreset.ALL_BIOLOGICAL),
                       neutral_pose=KinematicPosePreset.NEUTRAL)
        fly.colorize()
        world = TetheredWorld()
        world.add_fly(fly, [0, 0, 0], Rotation3D(format="quat", values=[1, 0, 0, 0]))
        self.m, self.d = world.compile()
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = W, H
        self.m.vis.headlight.ambient[:] = 0.22
        self.m.vis.headlight.diffuse[:] = 0.75
        self.m.vis.headlight.specular[:] = 0.3
        self.r = mujoco.Renderer(self.m, H, W)
        self.addr = {}
        for side in "lr":
            for leg in POSE:
                for part in PARTS:
                    name = "nmf/" + part.format(s=side + leg)
                    j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if j >= 0:
                        self.addr[name] = self.m.jnt_qposadr[j]
        self.pose(0)
        self.center = self.d.xpos[1:].mean(0).copy()
        self.head = self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "nmf/c_head")].copy()
        yy, xx = np.mgrid[0:H, 0:W]
        self.vignette = 1 - 0.55 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2.2

    def pose(self, t):
        self.d.qpos[:] = 0
        for side in "lr":
            for leg, vals in POSE.items():
                for part, v in zip(PARTS, vals):
                    name = "nmf/" + part.format(s=side + leg)
                    if name not in self.addr:
                        continue
                    if leg == "f" and ("femur-pitch" in part or "tibia-pitch" in part):
                        # The front legs tap like someone typing, left and right out of phase.
                        phase = 0 if side == "l" else np.pi
                        v += 0.28 * max(0.0, np.sin(2 * np.pi * 3.2 * t + phase)) * (1 if "femur" in part else -1)
                    self.d.qpos[self.addr[name]] = v
        mujoco.mj_forward(self.m, self.d)

    def render(self, t, lookat, distance, azimuth, elevation):
        self.pose(t)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
        self.r.update_scene(self.d, cam)
        rgb = self.r.render().astype(float)
        self.r.enable_depth_rendering()
        self.r.update_scene(self.d, cam)
        dep = self.r.render()
        self.r.disable_depth_rendering()
        mask = (dep < dep.max() * 0.999)[..., None]
        out = rgb * mask + BG * (1 - mask)
        return np.clip(out * self.vignette[..., None], 0, 255).astype(np.uint8)


def pill(d, x, y, label, col, alpha):
    w = d.textlength(label, font=F_PILL) + 22
    bg = tuple(int(c * 0.22 * alpha + b * (1 - 0.22 * alpha)) for c, b in zip(col, BG))
    d.rounded_rectangle([x - w, y - 17, x, y + 17], 17, fill=bg)
    text(d, (x - w / 2, y), label, F_PILL, col, "mm", alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    MEDIA.mkdir(exist_ok=True)

    results = json.loads((RESULTS / "results.json").read_text())
    probs = sorted(results.values(), key=lambda p: p["number"])
    # Reveal order: the one Accepted lands near the end, so the tally has a small moment of hope.
    probs = [p for p in probs if p["verdict"] != "Accepted"][:9] + [p for p in probs if p["verdict"] == "Accepted"] + \
            [p for p in probs if p["verdict"] != "Accepted"][9:]
    n_wa = sum(p["verdict"] != "Accepted" for p in probs)
    n_ac = len(probs) - n_wa

    sim = simulate()
    cam = Camera(sim["pos"])
    traj = sim["traj"]
    fly = Fly()
    step_every = (T_BACK - 0.6 - T_LIST) / len(probs)

    def brain_frame(t):
        s = np.clip((t - T_BRAIN + 0.4) / 5.0, 0, 1) * (len(traj) - 1)
        i = int(s)
        j = min(i + 1, len(traj) - 1)
        act = traj[i] * (1 - (s - i)) + traj[j] * (s - i)
        return render_brain(cam, t + 6.0, act, 1.0)

    def fly_frame(t, dive, ending=False):
        if ending:  # a slow three-quarter view for the tally
            return fly.render(t, fly.center, 5.8, 125 + 6 * (t - T_BACK), -12)
        az = 150 + 22 * t
        lookat = fly.center * (1 - dive) + fly.head * dive
        dist = 5.2 * (1 - dive) + 0.9 * dive
        return fly.render(t, lookat, dist, az, -14 + 6 * dive)

    def frame(t):
        if t < T_BRAIN:
            dive = ease((t - T_DIVE) / (T_BRAIN - T_DIVE)) ** 1.6
            img = fly_frame(t, dive).astype(float)
            if t > T_BRAIN - 0.5:  # crossfade into the brain through a brief glow
                k = ease((t - (T_BRAIN - 0.5)) / 0.5)
                img = img * (1 - k) + brain_frame(t).astype(float) * k + 60 * np.sin(np.pi * k)
        elif t < T_BACK:
            img = brain_frame(t).astype(float)
        else:
            k = ease((t - T_BACK) / 0.5)
            img = brain_frame(t).astype(float) * (1 - k) + fly_frame(t, 0.0, ending=True).astype(float) * k
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)

        # Hook line.
        a = ease(t / 0.5) * (1 - ease((t - T_DIVE) / 0.5))
        text(d, (W / 2, 110), "I gave a fruit fly's brain", F_BIG, INK, "ma", alpha=a)
        text(d, (W / 2, 190), "LeetCode.", F_BIG, AMBER, "ma", alpha=ease((t - 0.5) / 0.4) * (1 - ease((t - T_DIVE) / 0.5)))
        text(d, (W / 2, 1230), "real FlyWire connectome · 138,639 neurons", F_SMALL, MUTED, "ma",
             alpha=ease((t - 1.0) / 0.6) * (1 - ease((t - T_DIVE) / 0.5)))

        # Verdicts, one problem at a time, with a running tally.
        if T_LIST - 0.3 <= t < T_BACK + 0.3:
            fade = ease((t - T_LIST + 0.3) / 0.4) * (1 - ease((t - T_BACK) / 0.4))
            shown = int(np.clip((t - T_LIST) / step_every + 1, 0, len(probs)))
            wa = sum(p["verdict"] != "Accepted" for p in probs[:shown])
            ac = shown - wa
            text(d, (W / 2 - 24, 90), f"Wrong Answer ×{wa}", F_MED, RED, "ra", alpha=fade)
            text(d, (W / 2 + 24, 90), f"Accepted ×{ac}", F_MED, GREEN, "la", alpha=fade)
            for i, p in enumerate(probs[:shown]):
                col, row = divmod(i, 6)
                x0 = 60 + col * 500
                y = 930 + row * 54
                appear = ease((t - T_LIST - i * step_every) / 0.18) * fade
                name = f"{p['number']}. {p['title']}"
                while d.textlength(name, font=F_ROW) > 270:
                    name = name[:-2].rstrip() + "…"
                text(d, (x0, y), name, F_ROW, INK, "lm", alpha=appear)
                ok = p["verdict"] == "Accepted"
                pill(d, x0 + 460, y, "Accepted" if ok else "Wrong Answer", GREEN if ok else RED, appear)

        # Ending.
        if t >= T_BACK + 0.2:
            a = ease((t - T_BACK - 0.2) / 0.5)
            text(d, (W / 2, 100), f"{n_wa} Wrong Answers.", F_HUGE, RED, "ma", alpha=a)
            text(d, (W / 2, 196), f"{n_ac} Accepted.", F_MED, GREEN, "ma", alpha=ease((t - T_BACK - 0.8) / 0.5))
            text(d, (W / 2, 1230), "hidden test cases it had never seen", F_SMALL, MUTED, "ma",
                 alpha=ease((t - T_BACK - 1.2) / 0.5))
        return np.asarray(img)

    if args.preview:
        out = MEDIA / "preview"
        out.mkdir(exist_ok=True)
        for t in [1.5, 3.6, 4.1, 5.5, 7.5, 10.0, 12.5]:
            Image.fromarray(frame(t)).save(out / f"hook_{t:04.1f}.png")
        print("wrote", out)
        return

    path = MEDIA / "fly_brain_leetcode_hook.mp4"
    writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", quality=None, bitrate=None,
                                         output_params=["-crf", "20", "-pix_fmt", "yuv420p", "-preset", "slow",
                                                        "-movflags", "+faststart"])
    writer.send(None)
    n = int(T_END * FPS)
    for i in range(n):
        writer.send(np.ascontiguousarray(frame(i / FPS)))
        if i % 60 == 0:
            print(f"frame {i}/{n}", flush=True)
    writer.close()
    print("wrote", path)


if __name__ == "__main__":
    main()
