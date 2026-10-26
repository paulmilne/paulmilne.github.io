"""Celestial spheres: equirectangular textures and filtered lookup.

A texture is an (H, W, 3) array of linear radiance indexed by polar angle
theta in [0, pi] (rows) and azimuth phi in [0, 2 pi) (columns), in the
Boyer-Lindquist frame of the hole, so the spin axis is the texture's pole.

Gravitational lensing conserves surface brightness (Liouville's theorem), so
a pixel that sees a patch of sky should show that patch's average radiance.
The lookup therefore filters the texture over each pixel's footprint on the
sky using a mip-map pyramid, a cheap version of the ray bundles of the
DNGR renderer.
"""
import math

import numpy as np
from scipy.ndimage import map_coordinates

from .colour import chromaticity_rgb


def _unit(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], -1)


def grid(H):
    th = (np.arange(H) + 0.5) / H * math.pi
    ph = (np.arange(2 * H) + 0.5) / (2 * H) * 2 * math.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return TH, PH


def checkerboard(H=1024, n_lat=18, n_lon=36, line=0.0):
    """The paper's test sphere: coloured quadrants overlaid with a
    checkerboard, so every part of the lensed image can be traced home."""
    TH, PH = grid(H)
    quad = (PH >= math.pi).astype(int) * 2 + (TH >= math.pi / 2).astype(int)
    cols = np.array([[0.85, 0.25, 0.20], [0.20, 0.55, 0.85],
                     [0.95, 0.80, 0.20], [0.30, 0.75, 0.35]])
    tex = cols[quad]
    chk = ((np.floor(TH / math.pi * n_lat) + np.floor(PH / (2 * math.pi) * n_lon)) % 2).astype(float)
    tex = tex * (0.55 + 0.45 * chk[..., None])
    return tex.astype(np.float32)


def _noise3(v, seed, octaves=6, base=2.0):
    """Smooth, seamless noise on the sphere: random plane waves in 3-D."""
    rng = np.random.default_rng(seed)
    out = np.zeros(v.shape[:-1])
    for o in range(octaves):
        k = base * 2 ** o
        for _ in range(8):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            out += 0.6 ** o * np.cos(k * (v @ d) + rng.uniform(0, 2 * math.pi))
    return out / np.std(out)


def galactic_pole(tilt=60.0, node=90.0):
    t, n = math.radians(tilt), math.radians(node)
    return np.array([math.sin(t) * math.cos(n), math.sin(t) * math.sin(n), math.cos(t)])


def milky_way(H=2048, seed=11, pole=None, brightness=0.035, width=9.0):
    """Diffuse light of a dusty galactic band along the great circle
    perpendicular to `pole`, as an (H, 2H, 3) radiance texture."""
    pole = galactic_pole() if pole is None else pole
    TH, PH = grid(H)
    V = _unit(TH, PH)
    lat = np.arcsin(np.clip(V @ pole, -1, 1))
    w = math.radians(width)
    band = np.exp(-0.5 * (lat / w) ** 2)
    nz = _noise3(V, seed + 1)
    dust = _noise3(V, seed + 2, octaves=7, base=3.0)
    glow = band * np.clip(1 + 0.45 * nz, 0, None) * np.clip(1 - 0.55 * np.clip(dust, 0, None) * band, 0.05, 1)
    glow += 0.25 * np.exp(-0.5 * (lat / (3 * w)) ** 2)
    return (brightness * glow[..., None] * np.array([1.0, 0.93, 0.82])).astype(np.float32)


def star_catalogue(n_stars=30000, seed=11, pole=None, width=9.0, f_min=0.6, f_max=400.0):
    """Point stars: unit vectors v, fluxes F with the Euclidean count law
    N(>F) ~ F^(-3/2), and linear-RGB colours from blackbody temperatures
    (log-normal about 5800 K). Stars crowd towards the galactic plane.
    F is in units of radiance x texel solid angle of a 2048-row texture."""
    rng = np.random.default_rng(seed)
    pole = galactic_pole() if pole is None else pole
    w = math.radians(width)
    u = rng.uniform(-1, 1, size=3 * n_stars)
    p = rng.uniform(0, 2 * math.pi, size=3 * n_stars)
    vs = _unit(np.arccos(u), p)
    keep_p = 0.35 + 0.65 * np.exp(-0.5 * (np.arcsin(np.clip(vs @ pole, -1, 1)) / (2.5 * w)) ** 2)
    vs = vs[rng.uniform(size=len(u)) < keep_p][:n_stars]
    F = np.minimum(f_min * rng.uniform(size=len(vs)) ** (-2 / 3), f_max)
    T = np.exp(rng.normal(np.log(5800.0), 0.35, size=len(vs)))
    col = chromaticity_rgb(np.clip(T, 2500, 30000))
    return dict(v=vs, flux=F, rgb=col, T=T)


def splat_stars(tex, stars, sigma=0.6):
    """Paint a star catalogue into an equirectangular texture as small
    Gaussians (the 'environment map' approach to a star field)."""
    tex = np.array(tex, dtype=np.float64, copy=True)
    H, W = tex.shape[:2]
    v = stars["v"]
    th_s = np.arccos(np.clip(v[:, 2], -1, 1))
    p = np.arctan2(v[:, 1], v[:, 0]) % (2 * math.pi)
    F = stars["flux"] * (2048.0 / H) ** 2      # same total flux at any resolution
    col = stars["rgb"]
    r = 3
    row = th_s / math.pi * H - 0.5
    colm = p / (2 * math.pi) * W - 0.5
    st = np.maximum(np.sin(th_s), 0.05)
    i0 = np.floor(row).astype(int)
    j0 = np.floor(colm).astype(int)
    wsum = np.zeros(len(v))
    taps = []
    for di in range(-r, r + 2):
        for dj in range(-r, r + 2):
            ii, jj = i0 + di, j0 + dj
            dy = ii - row
            dx = (jj - colm) * st
            wgt = np.exp(-0.5 * (dx * dx + dy * dy) / sigma ** 2)
            taps.append((ii, jj, wgt))
            wsum += wgt
    for ii, jj, wgt in taps:
        ok = (ii >= 0) & (ii < H)
        np.add.at(tex, (ii[ok], jj[ok] % W), (F * wgt / wsum)[ok, None] * col[ok])
    return tex.astype(np.float32)


def starfield(H=2048, n_stars=30000, seed=11):
    """Milky Way plus splatted stars in one texture."""
    return splat_stars(milky_way(H, seed), star_catalogue(n_stars, seed))


class Sky:
    """Mip-mapped equirectangular texture with footprint-filtered lookup."""

    def __init__(self, tex):
        self.levels = [np.asarray(tex, dtype=np.float32)]
        while self.levels[-1].shape[0] > 4:
            t = self.levels[-1]
            t = 0.25 * (t[0::2, 0::2] + t[1::2, 0::2] + t[0::2, 1::2] + t[1::2, 1::2])
            self.levels.append(t)
        self.H = self.levels[0].shape[0]

    def _sample_level(self, L, theta, phi):
        t = self.levels[L]
        H, W = t.shape[:2]
        row = theta / math.pi * H - 0.5
        col = (phi % (2 * math.pi)) / (2 * math.pi) * W - 0.5
        # pad one column either side for wrap-around in phi
        tp = np.concatenate([t[:, -1:], t, t[:, :1]], 1)
        out = np.empty(theta.shape + (3,), dtype=np.float32)
        for c in range(3):
            out[..., c] = map_coordinates(tp[..., c], [row, col + 1], order=1, mode="nearest")
        return out

    def sample(self, v, footprint=None):
        """Radiance in unit directions v (..., 3). footprint is the angular
        size (radians) of each pixel on the sky; None means no filtering."""
        theta = np.arccos(np.clip(v[..., 2], -1, 1))
        phi = np.arctan2(v[..., 1], v[..., 0]) % (2 * math.pi)
        if footprint is None:
            return self._sample_level(0, theta, phi)
        texel = math.pi / self.H
        lev = np.clip(np.log2(np.maximum(footprint, 1e-12) / texel), 0, len(self.levels) - 1)
        out = np.zeros(theta.shape + (3,), dtype=np.float32)
        lo = np.floor(lev).astype(int)
        fr = (lev - lo)[..., None]
        for L in np.unique(lo):
            m = lo == L
            a = self._sample_level(L, theta[m], phi[m])
            if L + 1 < len(self.levels):
                b = self._sample_level(L + 1, theta[m], phi[m])
                out[m] = (1 - fr[m]) * a + fr[m] * b
            else:
                out[m] = a
        return out


def footprint(v, valid):
    """Angular size on the sky of each pixel of a (ny, nx, 3) direction
    image: the largest angle to a valid neighbour."""
    fp = np.zeros(v.shape[:2])
    for ax in (0, 1):
        for sh in (1, -1):
            w = np.roll(v, sh, axis=ax)
            ok = np.roll(valid, sh, axis=ax) & valid
            ang = np.arccos(np.clip((v * w).sum(-1), -1, 1))
            fp = np.where(ok, np.maximum(fp, ang), fp)
    return fp


# ----------------------------------------------------------------------------
# point stars through an arbitrary lensing map
# ----------------------------------------------------------------------------
def solid_angle(v0, v1, v2):
    """Signed solid angle of spherical triangles with unit-vector corners
    (Van Oosterom and Strackee 1983)."""
    num = np.einsum("...i,...i", v0, np.cross(v1, v2))
    den = 1 + np.einsum("...i,...i", v0, v1) + np.einsum("...i,...i", v1, v2) + np.einsum("...i,...i", v2, v0)
    return 2 * np.arctan2(num, den)


def lensed_stars(cam_dirs, sky_dirs, valid, stars, max_chord=0.5):
    """Find every image of every point star.

    The camera's sky is cut into triangles, two per cell of the ray grid;
    each maps to a spherical triangle on the celestial sphere. A star inside
    a sky triangle has an image inside the matching camera triangle, at the
    same barycentric position, and its magnification is the ratio of the
    two triangles' solid angles, negative where the image is mirror-reversed.

    cam_dirs and sky_dirs are (ny, nx, 3) unit vectors, valid is (ny, nx).
    cam_dirs must be on a right-handed triad matching the sky's (see
    Camera.right_handed), so that an image with the sky's orientation has
    positive parity. Returns star index, pixel x and y (in grid units),
    signed magnification and the camera-side triangle's solid angle."""
    from scipy.spatial import cKDTree
    ny, nx = valid.shape
    J, I = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    I, J = I.ravel(), J.ravel()
    tris = [((I, J), (I, J + 1), (I + 1, J)), ((I + 1, J + 1), (I + 1, J), (I, J + 1))]
    out = [[], [], [], [], []]
    tree = cKDTree(stars["v"])
    for tri in tris:
        ok = valid[tri[0]] & valid[tri[1]] & valid[tri[2]]
        S = [sky_dirs[t][ok] for t in tri]
        C = [cam_dirs[t][ok] for t in tri]
        P = [np.stack([t[1][ok], t[0][ok]], -1).astype(float) for t in tri]   # (x, y)
        chord = np.max([np.linalg.norm(S[0] - S[1], axis=1), np.linalg.norm(S[1] - S[2], axis=1),
                        np.linalg.norm(S[2] - S[0], axis=1)], axis=0)
        keep = chord < max_chord
        S = [s[keep] for s in S]
        C = [c[keep] for c in C]
        P = [p[keep] for p in P]
        cen = S[0] + S[1] + S[2]
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        rad = np.max([np.linalg.norm(s - cen, axis=1) for s in S], axis=0) * 1.0001
        lists = tree.query_ball_point(cen, rad, workers=-1)
        lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
        if lens.sum() == 0:
            continue
        ti = np.repeat(np.arange(len(lists)), lens)
        si = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)])
        sv = stars["v"][si]
        a, b, c = S[0][ti], S[1][ti], S[2][ti]
        o = np.sign(np.einsum("ij,ij->i", a, np.cross(b, c)))
        s1 = np.einsum("ij,ij->i", sv, np.cross(a, b)) * o
        s2 = np.einsum("ij,ij->i", sv, np.cross(b, c)) * o
        s3 = np.einsum("ij,ij->i", sv, np.cross(c, a)) * o
        inside = (s1 >= 0) & (s2 >= 0) & (s3 >= 0) & (o != 0)
        ti, si, s1, s2, s3 = ti[inside], si[inside], s1[inside], s2[inside], s3[inside]
        # barycentric weights: the triple product opposite each corner
        tot = s1 + s2 + s3
        wa, wb, wc = s2 / tot, s3 / tot, s1 / tot
        px = wa[:, None] * P[0][ti] + wb[:, None] * P[1][ti] + wc[:, None] * P[2][ti]
        om_sky = solid_angle(S[0][ti], S[1][ti], S[2][ti])
        om_cam = solid_angle(C[0][ti], C[1][ti], C[2][ti])
        out[0].append(si)
        out[1].append(px[:, 0])
        out[2].append(px[:, 1])
        out[3].append(om_cam / om_sky)
        out[4].append(om_cam)
    return tuple(np.concatenate(o) if o else np.zeros(0) for o in out)


def paint_stars(shape, stars, idx, x, y, mu, pixel_solid_angle, sigma=0.7, gain=1.0):
    """Render star images as small Gaussian spots. Each image carries flux
    |mu| F; dividing by the pixel solid angle turns it into radiance so it
    adds consistently to a radiance image. x, y are in output pixels."""
    ny, nx = shape
    img = np.zeros((ny, nx, 3))
    # F is defined per texel of a 2048-row equirectangular map
    texel = (math.pi / 2048) ** 2
    amp = gain * np.abs(mu) * stars["flux"][idx] * texel / pixel_solid_angle
    col = stars["rgb"][idx] * amp[:, None]
    r = int(math.ceil(3 * sigma))
    i0 = np.round(y).astype(int)
    j0 = np.round(x).astype(int)
    taps = []
    wsum = np.zeros(len(x))
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            ii, jj = i0 + di, j0 + dj
            w = np.exp(-0.5 * ((ii - y) ** 2 + (jj - x) ** 2) / sigma ** 2)
            taps.append((ii, jj, w))
            wsum += w
    for ii, jj, w in taps:
        ok = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        np.add.at(img, (ii[ok], jj[ok]), (w / wsum)[ok, None] * col[ok])
    return img
