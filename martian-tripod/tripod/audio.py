"""Sound for the tripod, synthesised from the simulation itself.

Nothing here is a recording. While the simulation runs, `SoundRecorder` logs
what makes noise, against simulation time:

  * every leg motor's torque and speed (its mechanical power, |torque x speed|),
  * each foot's ground force, from which footfalls and their weight are found,
  * whether the Heat-Ray is firing and where it lands, which targets catch
    fire, and how many are burning,
  * where the listener (the camera) is and which way it faces.

`synthesise` then builds a 48 kHz stereo track from that log:

  * motors: a geared whine per leg whose pitch follows joint speed and whose
    loudness follows power, over a hydraulic hiss;
  * footfalls: a deep falling thud plus a short metallic ring from the brass
    foot, scaled by how hard the foot came down;
  * Heat-Ray: a crackle as the beam strikes, a low electric drone while it
    is on, a "whoomph" each time something catches, and fire crackle that
    builds with the number of things alight. (Wells says nothing about the
    ray's sound; this part is invented.)

Every source sits somewhere in the world. Its sound reaches the listener
distance / 343 m/s later, quieter with distance, panned left or right by
where it is relative to the camera. From the village camera you see a foot
come down about half a second before you hear it.

Sync is exact by construction: video frame n shows simulation time
t0 + n * speed / fps, and audio sample k plays simulation time
t0 + k * speed / rate, so the two cannot drift. With --speed > 1 the
events come faster but pitches stay natural, because the sound is
synthesised on the playback timeline rather than sped up.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import mujoco

C_SOUND = 343.0          # m/s
REF_DIST = 40.0          # m: closer than this, no further gain


class SoundRecorder:
    def __init__(self, model: mujoco.MjModel, martian, camera: int, rate: float = 250.0):
        self.m, self.martian, self.cam = model, martian, camera
        self.every = max(1, round(1.0 / (rate * model.opt.timestep)))
        self.n = 0
        self.legs = [[model.joint(f"leg{i}_{j}").dofadr[0] for j in ("yaw", "hip", "knee")] for i in range(3)]
        self.knees = [model.body(f"leg{i}_shin").id for i in range(3)]
        self.feet = [model.site(f"foot{i}").id for i in range(3)]
        self.touch = [model.sensor(f"foot{i}_touch").adr[0] for i in range(3)]
        self.rows: dict[str, list] = {k: [] for k in
                                      ("t", "power", "speed", "knee_pos", "foot_force", "foot_pos", "firing",
                                       "muzzle", "hit", "burning", "fire_pos", "cam_pos", "cam_right")}
        self.ignitions: list[tuple[float, np.ndarray]] = []
        self._alight: set[int] = set()

    def __call__(self, data: mujoco.MjData) -> None:
        self.n += 1
        ray = self.martian.ray
        for g, h in ray.heat.items():                    # ignitions are checked every step, so none are missed
            if h >= ray.ignite and g not in self._alight:
                self._alight.add(g)
                self.ignitions.append((data.time, data.geom_xpos[g].copy()))
        if self.n % self.every:
            return
        r = self.rows
        r["t"].append(data.time)
        tau = data.actuator_force[:9]
        qd = np.array([data.qvel[d] for leg in self.legs for d in leg])
        r["power"].append(np.abs(tau * qd).reshape(3, 3).sum(axis=1))
        r["speed"].append(np.abs(qd).reshape(3, 3).sum(axis=1))
        r["knee_pos"].append(np.array([data.xpos[b] for b in self.knees]))
        r["foot_force"].append(np.array([data.sensordata[a] for a in self.touch]))
        r["foot_pos"].append(np.array([data.site_xpos[s] for s in self.feet]))
        r["firing"].append(float(ray.firing))
        r["muzzle"].append(data.site_xpos[ray.muzzle].copy())
        r["hit"].append(ray.hit_point.copy() if ray.hit_point is not None else data.site_xpos[ray.muzzle].copy())
        burning = ray.burning()
        r["burning"].append(len(burning))
        r["fire_pos"].append(np.mean([data.geom_xpos[g] for g in burning], axis=0) if burning else np.zeros(3))
        r["cam_pos"].append(data.cam_xpos[self.cam].copy())
        r["cam_right"].append(data.cam_xmat[self.cam].reshape(3, 3)[:, 0].copy())

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self.rows.items()}


# ---------------------------------------------------------------- synthesis helpers
def one_pole(x: np.ndarray, cutoff: float, sr: int, high: bool = False) -> np.ndarray:
    """First-order low-pass (or high-pass) filter, vectorised with scipy if available."""
    a = math.exp(-2 * math.pi * cutoff / sr)
    try:
        from scipy.signal import lfilter
        y = lfilter([1 - a], [1, -a], x)
    except ImportError:                                  # pragma: no cover
        y = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = (1 - a) * v + a * acc
            y[i] = acc
    return x - y if high else y


def band(x, lo, hi, sr):
    return one_pole(one_pole(x, hi, sr), lo, sr, high=True)


class Placer:
    """Distance delay, attenuation and panning of a moving source for a moving listener."""

    def __init__(self, rec: dict, t0: float, speed: float, sr: int, n: int):
        self.rec, self.t0, self.speed, self.sr, self.n = rec, t0, speed, sr, n
        self.tau = np.arange(n) / sr                     # playback time
        self.t_play = t0 + self.tau * speed              # the simulation time each sample shows

    def emitted_at(self, src: np.ndarray):
        """For a source track src (one xyz per control sample), the gain, pan and the
        simulation time of emission heard at each audio sample."""
        r = self.rec
        d = np.linalg.norm(src - r["cam_pos"], axis=1)
        # a source that jumps (the beam moving to a new target) can make arrival times run
        # backwards for an instant; keep them monotonic so the mapping stays one-to-one
        t_arrive = np.maximum.accumulate(r["t"] + d / C_SOUND)
        t_emit = np.interp(self.t_play, t_arrive, r["t"])
        dist = np.interp(t_emit, r["t"], d)
        gain = REF_DIST / np.maximum(dist, REF_DIST)
        rel = src - r["cam_pos"]
        side = np.einsum("ij,ij->i", rel, r["cam_right"]) / np.maximum(d, 1e-6)
        pan = np.interp(t_emit, r["t"], side)
        return t_emit, gain, pan, dist

    def stereo(self, mono: np.ndarray, gain: np.ndarray, pan: np.ndarray) -> np.ndarray:
        th = (np.clip(pan, -1, 1) + 1) * math.pi / 4      # equal-power pan
        return np.stack([mono * gain * np.cos(th), mono * gain * np.sin(th)], axis=1)


def air(x: np.ndarray, dist: float, sr: int) -> np.ndarray:
    """Distant sounds lose their top end."""
    return one_pole(x, max(1500.0, 16000.0 * 60.0 / max(dist, 60.0)), sr)


# ---------------------------------------------------------------- the sources
def motors(P: Placer, rec, rng) -> np.ndarray:
    out = np.zeros((P.n, 2))
    sr = P.sr
    p_ref = 4e6                                          # W: a hard-working leg
    for i in range(3):
        t_emit, gain, pan, _ = P.emitted_at(rec["knee_pos"][:, i])
        power = np.interp(t_emit, rec["t"], rec["power"][:, i])
        speed = np.interp(t_emit, rec["t"], rec["speed"][:, i])
        level = np.sqrt(np.clip(power / p_ref, 0, 4))    # loudness follows power
        level = one_pole(level, 8.0, sr)
        f0 = 38.0 + 260.0 * np.clip(speed, 0, 1.2) + 4.0 * i   # gearbox whine rises with joint speed
        phase = 2 * math.pi * np.cumsum(f0) / sr
        whine = (np.sin(phase) + 0.5 * np.sin(2 * phase + 0.3) + 0.25 * np.sin(3 * phase)
                 + 0.12 * np.sin(5.01 * phase))
        grind = band(rng.standard_normal(P.n), 60, 400, sr) * 0.8
        hiss = band(rng.standard_normal(P.n), 1800, 6000, sr) * 0.35
        mono = (0.5 * whine + grind) * level + hiss * level ** 1.5
        out += P.stereo(mono * 0.35, gain, pan)
    return out


def footfalls(P: Placer, rec, rng) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Find each foot's landings (rising edges of ground force) and place a thud at each."""
    out = np.zeros((P.n, 2))
    sr = P.sr
    t = rec["t"]
    weight = 2.5e6
    events = []
    dt = t[1] - t[0]
    for i in range(3):
        F = rec["foot_force"][:, i]
        z = rec["foot_pos"][:, i, 2]
        # a landing: the foot was clear of the ground (by height, which unlike contact
        # force does not chatter) for at least a third of a second, and is now loaded
        airborne_since = None
        for e in range(len(t)):
            if z[e] > 1.6:                                # the foot sphere's centre sits at 1.1 m when planted
                if airborne_since is None:
                    airborne_since = t[e]
            elif airborne_since is not None and F[e] > 0.05 * weight / 3:
                if t[e] - airborne_since > 0.3:
                    peak = F[e:e + int(0.3 / dt) + 1].max()
                    events.append((t[e], rec["foot_pos"][e, i].copy(), peak))
                airborne_since = None
    marks = []
    dur = 1.6
    k = np.arange(int(dur * sr)) / sr
    for te, pos, peak in events:
        j = int(np.searchsorted(t, te).clip(0, len(t) - 1))
        d = float(np.linalg.norm(pos - rec["cam_pos"][j]))
        arrive = te + d / C_SOUND
        start = int(round((arrive - P.t0) / P.speed * sr))
        if start >= P.n or start < -len(k):
            continue
        strength = min(2.0, (peak / (0.4 * weight)) ** 0.6)
        f = 16 + 30 * np.exp(-k / 0.12)                   # a falling boom
        boom = np.sin(2 * math.pi * np.cumsum(f) / sr) * np.exp(-k / 0.45)
        thump = one_pole(rng.standard_normal(len(k)), 180, sr) * np.exp(-k / 0.06) * 3.0
        ring = sum(a * np.sin(2 * math.pi * fr * k + ph) * np.exp(-k / dc)
                   for a, fr, ph, dc in ((0.25, 183, 0.1, 0.6), (0.18, 297, 1.3, 0.45), (0.12, 412, 2.0, 0.3),
                                         (0.07, 655, 0.7, 0.2)))
        mono = air(strength * (1.4 * boom + thump + 0.5 * ring), d, sr)
        gain = REF_DIST / max(d, REF_DIST)
        pan = float((pos - rec["cam_pos"][j]) @ rec["cam_right"][j] / max(d, 1e-6))
        a, b = max(0, start), min(P.n, start + len(k))
        seg = mono[a - start:b - start]
        out[a:b] += P.stereo(seg * 0.9, np.full(len(seg), gain), np.full(len(seg), pan))
        marks.append((te, arrive))
    return out, marks


def heatray(P: Placer, rec, ignitions, rng) -> np.ndarray:
    out = np.zeros((P.n, 2))
    sr = P.sr
    t = rec["t"]
    # the drone and crackle come from where the beam lands; the projector's own hum from the muzzle
    t_emit, gain, pan, dist = P.emitted_at(rec["hit"])
    on = one_pole(np.interp(t_emit, t, rec["firing"]), 25.0, sr)
    hum_phase = 2 * math.pi * np.cumsum(np.full(P.n, 55.0)) / sr
    hum = (np.sin(hum_phase) + 0.6 * np.sin(2 * hum_phase) + 0.3 * np.sin(3 * hum_phase + 1)) * (
        1 + 0.3 * np.sin(2 * math.pi * 7.0 * P.tau))
    sizzle = band(rng.standard_normal(P.n), 2500, 9000, sr) * (rng.random(P.n) < 0.02) * 6
    sizzle = one_pole(sizzle, 6000, sr)
    out += P.stereo((0.25 * hum + 0.3 * sizzle) * on, gain, pan)
    # a crack each time the beam comes on
    rise = np.flatnonzero(np.diff(rec["firing"]) > 0.5) + 1
    k = np.arange(int(0.35 * sr)) / sr
    for e in rise:
        d = np.linalg.norm(rec["hit"][e] - rec["cam_pos"][e])
        start = int(round((t[e] + d / C_SOUND - P.t0) / P.speed * sr))
        if not 0 <= start < P.n:
            continue
        crack = band(rng.standard_normal(len(k)), 400, 7000, sr) * np.exp(-k / 0.05)
        crack = air(crack, d, sr) * 1.5
        b = min(P.n, start + len(k))
        pn = float((rec["hit"][e] - rec["cam_pos"][e]) @ rec["cam_right"][e] / max(d, 1e-6))
        out[start:b] += P.stereo(crack[:b - start], np.full(b - start, REF_DIST / max(d, REF_DIST)),
                                 np.full(b - start, pn))
    # whoomph as each thing catches
    k = np.arange(int(1.4 * sr)) / sr
    env = (1 - np.exp(-k / 0.08)) * np.exp(-k / 0.45)
    for te, pos in ignitions:
        j = np.searchsorted(t, te).clip(0, len(t) - 1)
        d = np.linalg.norm(pos - rec["cam_pos"][j])
        start = int(round((te + d / C_SOUND - P.t0) / P.speed * sr))
        if not 0 <= start < P.n:
            continue
        whoomph = one_pole(rng.standard_normal(len(k)), 350, sr) * env * 4
        whoomph = air(whoomph, d, sr)
        b = min(P.n, start + len(k))
        pn = float((pos - rec["cam_pos"][j]) @ rec["cam_right"][j] / max(d, 1e-6))
        out[start:b] += P.stereo(whoomph[:b - start], np.full(b - start, REF_DIST / max(d, REF_DIST)),
                                 np.full(b - start, pn))
    # fire crackle, more of it the more is burning
    fire_src = np.where(rec["burning"][:, None] > 0, rec["fire_pos"], rec["hit"])
    t_emit, gain, pan, dist = P.emitted_at(fire_src)
    n_burn = np.interp(t_emit, t, rec["burning"])
    clicks = (rng.random(P.n) < np.clip(n_burn * 15 / sr, 0, 0.5)) * rng.standard_normal(P.n) * 3
    crackle = band(clicks, 700, 5000, sr) + band(rng.standard_normal(P.n), 80, 400, sr) * 0.25 * np.sqrt(n_burn)
    out += P.stereo(crackle * 0.3, np.minimum(gain * 1.5, 1.0), pan)
    return out


def ambience(P: Placer, rng) -> np.ndarray:
    wind = band(rng.standard_normal(P.n), 40, 500, P.sr)
    wind *= 0.6 + 0.4 * np.sin(2 * math.pi * 0.07 * P.tau + rng.random() * 6)
    return np.stack([wind, np.roll(wind, P.sr // 50)], axis=1) * 0.03


def synthesise(recorder: SoundRecorder, t0: float, duration: float, speed: float = 1.0,
               sr: int = 48000, seed: int = 1898) -> tuple[np.ndarray, list]:
    """Build the stereo track for playback time [0, duration), showing simulation time t0 onward."""
    rng = np.random.default_rng(seed)
    rec = recorder.arrays()
    n = int(round(duration * sr))
    P = Placer(rec, t0, speed, sr, n)
    feet, marks = footfalls(P, rec, rng)
    mix = 1.3 * motors(P, rec, rng) + 1.0 * feet + 0.6 * heatray(P, rec, recorder.ignitions, rng) + ambience(P, rng)
    peak = np.percentile(np.abs(mix), 99.99) + 1e-9
    mix = np.tanh(0.9 * mix / peak) * 0.95                # level, keeping the footfalls' punch, and soft-limit
    return mix.astype(np.float32), marks


def write_wav(path: Path, audio: np.ndarray, sr: int = 48000) -> None:
    from scipy.io import wavfile
    wavfile.write(str(path), sr, (audio * 32767).astype(np.int16))


def mux(video: Path, wav: Path, out: Path) -> Path:
    """Attach the soundtrack to the video with ffmpeg (the one bundled with imageio-ffmpeg)."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:                                  # pragma: no cover
        ffmpeg = "ffmpeg"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    return out
