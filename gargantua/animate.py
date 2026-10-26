"""Render Gargantua's disk turning, seen from a fixed camera, as an MP4.

The camera does not move, so the ray map is traced once. Each frame then
rotates the disk's texture: gas at radius r turns at its own orbital
angular velocity Omega(r), and each ray shows the gas as it was when the
light left, a coordinate time |t_hit| before it reached the camera. So the
different images of the disk (the near side, the arch over the top, the
ring below) show the same gas at different moments.

    ../.venv/bin/python animate.py                 # both versions
    ../.venv/bin/python animate.py --shift none    # as in the film
    ../.venv/bin/python animate.py --shift full    # with Doppler shifts
"""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blackhole import disk as D, kerr, render as R, sky as SK          # noqa: E402
from blackhole.colour import bloom, tonemap, xyz_to_rgb               # noqa: E402

A, R_CAM, TH_CAM, R_OUT = 0.6, 74.1, 86.56, 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", choices=["none", "full", "both"], default="both")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--fov", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--span", type=float, default=90.0,
                    help="coordinate time covered by the clip, in units of M")
    ap.add_argument("--exposure", type=float, default=0.35)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "sci-fi" / "gargantua"))
    ap.add_argument("--audio", default=None,
                    help="optional soundtrack to mux in, faded in and out")
    args = ap.parse_args()
    audio = Path(args.audio) if args.audio else None

    t0 = time.time()
    tr = R.trace_image(R_CAM, TH_CAM, A, args.width, args.height, args.fov, ss=args.ss,
                       disk=True, r_out=R_OUT)
    print(f"ray map: {tr['outcome'].size:,} rays, traced in {float(tr['seconds']):.1f} s "
          f"(this run {time.time() - t0:.1f} s)", flush=True)

    # the background (sky and stars) does not change
    milky = SK.Sky(SK.milky_way(2048))
    stars = SK.star_catalogue()
    bg = R.downsample(R.sky_radiance(tr, milky), args.ss) + 0.6 * R.stars_image(tr, stars, args.ss)

    hit = tr["outcome"] == kerr.DISK
    r = tr["r_hit"][hit]
    phi = tr["phi_hit"][hit]
    t_hit = tr["t_hit"][hit]
    Om = D.omega(r, A)
    S = D.streak_basis(r)
    # extra clumps with more azimuthal structure, so that the rotation shows
    C = D.streak_basis(r, seed=7, n_modes=60, m_range=(6, 24), k_range=(3, 25))
    fade = np.clip((R_OUT - r) / (0.25 * R_OUT), 0, 1) ** 1.5
    print(f"light travel time from the disk: {-t_hit.max():.0f} to {-t_hit.min():.0f} M; "
          f"inner-edge orbital period {2 * math.pi / D.omega(D.isco(A), A):.0f} M", flush=True)

    shifts = ["none", "full"] if args.shift == "both" else [args.shift]
    for shift in shifts:
        # per-ray colour, normalised as in render.disk_radiance
        tmp = dict(tr)
        tmp["phi_hit"] = tr["phi_hit"]
        base = R.disk_radiance(tmp, A, model="film", T=4500, shift=shift, texture=False, r_out=None)[hit]
        base *= fade[:, None]
        name = Path(args.out) / f"gargantua_{'film' if shift == 'none' else 'doppler'}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-"]
        if audio is not None:
            fade = max(args.seconds - 2.5, 0)
            cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k",
                    "-af", f"afade=t=in:d=0.5,afade=t=out:st={fade}:d=2.5", "-t", str(args.seconds)]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(name)]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(round(args.seconds * args.fps))
        img = np.zeros(tr["outcome"].shape + (3,))
        t1 = time.time()
        for f in range(n_frames):
            t_obs = args.span * f / n_frames
            # the gas seen along each ray left at t_obs + t_hit; a feature now at
            # phi was at phi - Omega t then
            ph = phi - Om * (t_obs + t_hit)
            tex = D.streaks_rotated(S, ph) * D.streaks_rotated(C, ph, contrast=0.3)
            img[hit] = base * tex[:, None]
            lin = R.downsample(img, args.ss) + bg
            frame = tonemap(bloom(lin), args.exposure)
            ff.stdin.write((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
            if f % 24 == 0:
                el = time.time() - t1
                print(f"{shift}: frame {f}/{n_frames}, {el / max(f, 1):.2f} s per frame", flush=True)
        ff.stdin.close()
        ff.wait()
        print(f"wrote {name} in {time.time() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main()
