"""A soundtrack for the Gargantua clips, computed from the disk itself.

Each note is a ring of the disk. A ring at radius r orbits with angular
velocity Omega(r) = 1 / (r^1.5 + a); for a 10^8-solar-mass hole that is a few
millionths of a hertz. Raising every orbital frequency by the same whole
number of octaves keeps their ratios, so the chord is exactly the chord the
rings make. The rings are chosen so that their orbits are in resonance with
the outer edge (1 : 3/2 : 2 : 3 : 4 : 6 : 8), which gives the open fifths and
octaves of an organ chord. Each note also swells at the rate its ring turns
in the clip, so the sound pulses with the picture.

    ../.venv/bin/python soundtrack.py      # writes out/soundtrack.wav and muxes it into both clips
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, signal
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blackhole import disk as D                                         # noqa: E402

A, R_OUT = 0.6, 20.0
RATIOS = (1, 1.5, 2, 3, 4, 6, 8)
G, C, M_SUN = 6.674e-11, 2.998e8, 1.989e30
T_M = G * 1e8 * M_SUN / C ** 3          # GM/c^3 in seconds for a 10^8-solar-mass hole
OCTAVES = 24                            # transposition into the audible range
SECONDS, SPAN = 15.0, 90.0              # clip length, and the coordinate time it covers (M)
RATE = 48000
POST = Path(__file__).resolve().parents[1] / "sci-fi" / "gargantua"


def notes(a=A, r_out=R_OUT, ratios=RATIOS):
    """Radii of the rings whose orbital frequencies are ratios * Omega(r_out),
    their true frequencies (Hz), the audible pitches (Hz), and how fast each
    ring turns in the clip (revolutions per second)."""
    om0 = D.omega(r_out, a)
    r = np.array([optimize.brentq(lambda x: D.omega(x, a) - n * om0, D.isco(a), r_out + 1e-9)
                  if n != 1 else r_out for n in ratios])
    om = D.omega(r, a)
    true = om / (2 * math.pi * T_M)
    return r, true, true * 2 ** OCTAVES, om * SPAN / (2 * math.pi * SECONDS)


def organ(f, t, rng):
    """An organ-like tone: 8', 4', 2 2/3', 2' ... ranks, three slightly
    detuned pipes per note."""
    reg = [1.0, 0.6, 0.35, 0.3, 0.12, 0.1, 0.05, 0.06]
    out = np.zeros_like(t)
    for cents in (-1.5, 0.0, 1.5):
        ff = f * 2 ** (cents / 1200)
        for k, amp in enumerate(reg, 1):
            if k * ff < 8000:
                out += amp * np.sin(2 * math.pi * k * ff * t + rng.uniform(0, 2 * math.pi))
    return out / 3


def reverb(x, rng, seconds=3.0, tau=0.6, wet=0.5):
    """Convolve with decaying noise (RT60 about 7 tau) for a stone-hall tail."""
    n = int(seconds * RATE)
    env = np.exp(-np.arange(n) / (tau * RATE))
    out = []
    for _ in range(2):
        ir = rng.standard_normal(n) * env
        ir /= np.sqrt(np.sum(ir ** 2))
        out.append((1 - wet) * x + wet * signal.fftconvolve(x, ir)[:x.size])
    return np.stack(out, axis=1)


def render(seconds=SECONDS, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    r, true, pitch, turn = notes()
    mix = np.zeros_like(t)
    for i, (f, rate) in enumerate(zip(pitch, turn)):
        start = 1.5 * i                                     # outer ring first, inner rings join
        swell = np.clip((t - start) / 2.0, 0, 1) ** 2
        pulse = 1 + 0.25 * np.sin(2 * math.pi * rate * t + rng.uniform(0, 2 * math.pi))
        tone = organ(f, t, rng)
        if i == 0:
            tone += 0.5 * organ(f / 2, t, rng)              # a 16' rank under the bass
        mix += swell * pulse * tone / math.sqrt(1 + i)
    stereo = reverb(mix, rng)
    fade = np.clip((seconds - t) / 2.5, 0, 1)[:, None] * np.clip(t / 0.3, 0, 1)[:, None]
    stereo *= fade
    return stereo / np.abs(stereo).max() * 10 ** (-4 / 20)  # peak at -4 dBFS


def mux(wav, video):
    tmp = video.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-t", str(SECONDS), "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(video)


def main():
    r, true, pitch, turn = notes()
    for n, rr, f0, f, w in zip(RATIOS, r, true, pitch, turn):
        print(f"{n:>4g} x outer: ring at r = {rr:5.2f} M, {f0:.3e} Hz -> {f:6.1f} Hz, "
              f"turns {w:.3f} times per second in the clip")
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    wav = out / "soundtrack.wav"
    wavfile.write(wav, RATE, (render() * 32767).astype(np.int16))
    print(f"wrote {wav}")
    for name in ("gargantua_film.mp4", "gargantua_doppler.mp4"):
        for folder in (POST, POST.parents[1] / "docs" / "sci-fi" / "gargantua"):
            if (folder / name).exists():
                mux(wav, folder / name)
                print(f"muxed into {folder / name}")


if __name__ == "__main__":
    main()
