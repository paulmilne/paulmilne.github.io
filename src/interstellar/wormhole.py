"""Render Interstellar's wormhole (the Double Negative 'Dneg' wormhole, James et al. 2015,
Am. J. Phys. 83, 486) with the film's parameters: half-length a = 0.005 rho, lensing width
W = 0.05 rho. Gravity is neglected, as in the paper, so the metric is
ds^2 = -dt^2 + dl^2 + r(l)^2 dOmega^2, and every ray stays in a plane through the axis.
Rays are traced backwards from the camera; each ends on one of two skies, l -> +inf
(the Saturn side) or l -> -inf (the Gargantua side)."""
import numpy as np
from scipy.integrate import solve_ivp
from PIL import Image

RHO, A, W = 1.0, 0.005, 0.05
MM = W / 1.42953                                # the paper's M parameter
def r_of(l):
    x = 2 * (np.abs(l) - A) / (np.pi * MM)
    return np.where(np.abs(l) <= A, RHO, RHO + MM * (x * np.arctan(x) - 0.5 * np.log1p(x * x)))
def dr_of(l):
    x = 2 * (np.abs(l) - A) / (np.pi * MM)
    return np.where(np.abs(l) <= A, 0.0, np.sign(l) * (2 / np.pi) * np.arctan(x))

def trace_angles(alphas, l0, L=400.0):
    """Trace all rays together (vectorised RK4, step growing with distance). A ray leaves the
    camera at angle alpha from the inward axis. Returns, per ray, the sky it ends on (+1 or -1)
    and its final direction Phi, measured from the axis that points back at the camera's side."""
    B = r_of(l0) * np.sin(alphas)
    y = np.stack([np.full_like(alphas, l0), -np.cos(alphas), np.zeros_like(alphas)])
    live = np.ones(alphas.size, bool)
    def f(y, B):
        r = r_of(y[0]); return np.stack([y[1], B * B * dr_of(y[0]) / r**3, B / r**2])
    for _ in range(200000):
        if not live.any(): break
        yy, Bb = y[:, live], B[live]
        h = 0.0005 + 0.01 * np.abs(yy[0])
        k1 = f(yy, Bb); k2 = f(yy + h/2*k1, Bb); k3 = f(yy + h/2*k2, Bb); k4 = f(yy + h*k3, Bb)
        yy = yy + h/6*(k1 + 2*k2 + 2*k3 + k4)
        y[:, live] = yy; live[live] = np.abs(yy[0]) < L
    side = np.where(y[0] > 0, 1, -1)
    Phi = y[2] + np.arcsin(np.clip(B / r_of(y[0]), -1, 1))
    return side, Phi

def sky_texture(seed, kind, H=2048):
    """Equirectangular sky. kind 'saturn': blue-white stars and the Milky Way; 'gargantua':
    warm stars, a dense galactic band and a bright galaxy."""
    rng = np.random.default_rng(seed); Wd = 2 * H
    img = np.zeros((H, Wd, 3))
    lat = (0.5 - (np.arange(H) + 0.5) / H)[:, None] * np.pi; lon = ((np.arange(Wd) + 0.5) / Wd)[None, :] * 2 * np.pi
    band_lat = 0.35 * np.sin(lon + (0.8 if kind == 'saturn' else 2.4))
    band = np.exp(-((lat - band_lat) / (0.10 if kind == 'saturn' else 0.16))**2)
    if kind == 'saturn': img += band[..., None] * np.array([0.10, 0.10, 0.12]) + np.array([0.005, 0.006, 0.012])
    else:                img += band[..., None] * np.array([0.22, 0.13, 0.07]) + np.array([0.012, 0.006, 0.004])
    n = 30000 if kind == 'saturn' else 50000
    y = rng.integers(0, H, n); x = rng.integers(0, Wd, n)
    w = np.cos((0.5 - (y + 0.5) / H) * np.pi); keep = rng.random(n) < w; y, x = y[keep], x[keep]
    br = rng.pareto(2.2, y.size) * 0.25 + 0.15
    col = np.array([0.80, 0.88, 1.0]) if kind == 'saturn' else np.array([1.0, 0.82, 0.62])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k = 1.0 if dx == dy == 0 else 0.25
            img[(y + dy) % H, (x + dx) % Wd] += (br * k)[:, None] * col
    if kind == 'gargantua':                       # a bright spiral-ish galaxy
        cy, cx = int(H * 0.42), int(Wd * 0.47)
        yy, xx = np.mgrid[-120:121, -220:221]; e = np.exp(-(xx**2 / 8800 + yy**2 / 2000)); img[cy-120:cy+121, cx-220:cx+221] += e[..., None] * np.array([1.1, 0.9, 0.7])
    else:                                         # Saturn, small, beside the line of sight
        cy, cx = int(H * 0.47), int(Wd * 0.53)
        yy, xx = np.mgrid[-80:81, -140:141]
        disc = (xx**2 + (yy * 1.0)**2) < 22**2
        ring = (np.abs(np.hypot(xx, yy * 2.6) - 42) < 9) & ~((yy < 0) & disc)
        img[cy-80:cy+81, cx-140:cx+141][disc] = [0.95, 0.80, 0.55]; img[cy-80:cy+81, cx-140:cx+141][ring] = [0.85, 0.78, 0.62]
    return np.clip(img, 0, 1)

def render(l0, nx=900, ny=450, fov=70.0, nalpha=3000):
    skyA, skyB = sky_texture(1, 'saturn'), sky_texture(2, 'gargantua')
    alphas = np.linspace(0, np.pi, nalpha)
    side_t, Phi_t = trace_angles(alphas, l0)
    f = 0.5 * nx / np.tan(np.radians(fov) / 2)
    j, i = np.mgrid[0:ny, 0:nx]
    dx = (i + 0.5 - nx / 2) / f; dy = -(j + 0.5 - ny / 2) / f
    d = np.stack([np.ones_like(dx), dx, dy], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    # camera at +l0 on the x axis, looking towards the hole (-x). Local ray direction:
    v = np.stack([-d[..., 0], d[..., 1], d[..., 2]], -1)          # world: x outward axis
    alpha = np.arccos(np.clip(-v[..., 0], -1, 1))                 # angle from inward axis
    k = np.clip(np.rint(alpha / np.pi * (nalpha - 1)).astype(int), 0, nalpha - 1)
    side = side_t[k]; Phi = Phi_t[k]
    perp = v.copy(); perp[..., 0] = 0; nrm = np.linalg.norm(perp, axis=-1, keepdims=True); perp = perp / np.where(nrm == 0, 1, nrm)
    # final direction: angle Phi from the axis pointing back at the camera side (+x), in the ray's plane
    # On side +1 the outward axis is +x; the ray left the camera heading -x with phi=0 at +x axis.
    out = np.cos(Phi)[..., None] * np.array([1.0, 0, 0]) + np.sin(Phi)[..., None] * perp
    # on the far side, the outward axis is -x of that universe; mirror so its sky is seen ahead
    out[side < 0, 0] *= 1
    lonA = np.arctan2(out[..., 1], out[..., 0]) % (2 * np.pi); latA = np.arcsin(np.clip(out[..., 2], -1, 1))
    H = skyA.shape[0]; Wd = skyA.shape[1]
    yy = np.clip(((0.5 - latA / np.pi) * H).astype(int), 0, H - 1); xx = (lonA / (2 * np.pi) * Wd).astype(int) % Wd
    img = np.where((side > 0)[..., None], skyA[yy, xx], skyB[yy, xx])
    return img, side

if __name__ == "__main__":
    import sys
    for name, l0 in [("wormhole_far", 6.0), ("wormhole_near", 2.2)]:
        img, side = render(l0, nx=1800, ny=900)
        im = Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.4) * 255).astype(np.uint8)).resize((900, 450), Image.LANCZOS)
        im.save(f"{name}.png")
        print(name, "fraction of view through the wormhole:", (side < 0).mean())
