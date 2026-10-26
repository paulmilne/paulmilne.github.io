"""Render a Kerr black hole, its thin disk and the sky behind it.

Rendering is split in two. trace_image() follows every ray back from the
camera once (the expensive part) and caches what each ray hit. shade()
turns that into an image for a given sky and disk model, so the same
geometry can be shown with and without Doppler shifts.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import disk as D
from . import kerr
from .colour import BlackbodyTable, tonemap, xyz_to_rgb, bloom
from .sky import footprint

CACHE = Path(__file__).resolve().parents[1] / "cache"


def trace_image(r, theta_deg, a, nx, ny, fov, ss=2, beta=0.0, disk=True,
                r_out=None, centre=(0.0, 0.0), roll=0.0, tol=1e-9, cache=True,
                projection="pinhole", r_in=None):
    """Trace an (ny*ss, nx*ss) grid of rays. Returns a dict of per-ray maps:
    outcome, sky direction (escaped), disk hit radius and azimuth, the
    redshift g of disk hits, plus the camera and run statistics."""
    params = dict(r=r, theta=theta_deg, a=a, nx=nx, ny=ny, fov=fov, ss=ss, beta=beta,
                  disk=disk, r_out=r_out, centre=list(centre), roll=roll, tol=tol,
                  projection=projection, r_in=r_in, v=6)
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"trace_{key}.npz"
    cam = kerr.Camera(r, math.radians(theta_deg), a, beta=beta, fov=fov, roll=roll)
    if cache and path.exists():
        z = np.load(path)
        out = {k: z[k] for k in z.files}
        out["camera"] = cam
        return out

    NX, NY = nx * ss, ny * ss
    if projection == "panorama":
        N = cam.panorama(NX, NY)
    else:
        N = cam.directions(NX, NY, centre=centre)
    Y0, B, E_cam = cam.initial_conditions(N)
    r_in = (D.isco(a) if r_in is None else r_in) if disk else 0.0
    r_o = (r_out if r_out is not None else 0.0) if disk else 0.0
    import time
    t0 = time.time()
    Y, oc, ns = kerr.trace(Y0, B, a, r_in=r_in, r_out=r_o, r_esc=max(1000.0, 2.0 * r), tol=tol)
    secs = time.time() - t0
    v = np.zeros((len(Y), 3))
    esc = oc == kerr.ESCAPED
    v[esc] = kerr.sky_direction(Y[esc], B[esc], a)
    hit = oc == kerr.DISK
    g = np.zeros(len(Y))
    g[hit] = D.redshift(Y[hit, 0], B[hit], E_cam[hit], a)
    out = dict(outcome=oc.reshape(NY, NX), sky=v.reshape(NY, NX, 3), cam=N.reshape(NY, NX, 3),
               r_hit=np.where(hit, Y[:, 0], 0.0).reshape(NY, NX),
               phi_hit=np.where(hit, Y[:, 2], 0.0).reshape(NY, NX),
               t_hit=np.where(hit, Y[:, 5], 0.0).reshape(NY, NX),
               g=g.reshape(NY, NX), b=B.reshape(NY, NX), steps=ns.reshape(NY, NX),
               seconds=np.array(secs), beta=np.array(cam.beta))
    if cache:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(path, **out)
    out["camera"] = cam
    return out


_BB = None


def _bb():
    global _BB
    if _BB is None:
        _BB = BlackbodyTable()
    return _BB


def _luma(rgb):
    # the weighting DNGR used to remove brightness changes but keep colours
    return rgb @ np.array([0.30, 0.59, 0.11])


def disk_radiance(tr, a, model="film", T=4500.0, shift="none", texture=True, r_out=None):
    """Linear sRGB radiance of the rays that hit the disk.

    model "film": a blackbody at one temperature T everywhere, the
    "anemic" artist's disk of Interstellar (4500 K in the DNGR paper).
    model "novikov-thorne": T(r) from the Page-Thorne flux, peaking at T.

    shift "none": no frequency shifts, as in the film.
    shift "colour": colours shifted by g, brightness held at the unshifted
    value (DNGR's Figure 15b).
    shift "full": the camera sees a blackbody at g T. This is both the
    colour change and the brightness change I_nu / nu^3 = invariant.
    The result is normalised so that an unshifted blackbody at T has unit
    luminance."""
    bb = _bb()
    hit = tr["outcome"] == kerr.DISK
    rr = tr["r_hit"][hit]
    if model == "film":
        Te = np.full_like(rr, T)
    else:
        Te = D.temperature(rr, a, T)
    g = tr["g"][hit] if shift != "none" else np.ones_like(rr)
    rgb = xyz_to_rgb(bb(g * Te))
    if shift == "colour":
        rgb0 = xyz_to_rgb(bb(Te))
        rgb = rgb * (_luma(rgb0) / np.maximum(_luma(rgb), 1e-300))[:, None]
    rgb = rgb / bb(np.array([T]))[0, 1]
    w = np.ones_like(rr)
    if texture:
        w *= D.streaks(rr, tr["phi_hit"][hit])
    if r_out is not None:
        # soft outer edge
        w *= np.clip((r_out - rr) / (0.25 * r_out), 0, 1) ** 1.5
    out = np.zeros(tr["outcome"].shape + (3,))
    out[hit] = rgb * w[:, None]
    return out


def sky_radiance(tr, sky):
    """Radiance of the diffuse sky texture, footprint-filtered."""
    oc = tr["outcome"]
    esc = oc == kerr.ESCAPED
    v = tr["sky"]
    fp = footprint(v, esc)
    img = np.zeros(oc.shape + (3,))
    img[esc] = sky.sample(v[esc], fp[esc])
    return img


def compose(tr, a, ss, sky=None, stars=None, disk=None, disk_gain=1.0,
            star_gain=1.0, exposure=1.0, glare=True):
    """Full image: diffuse sky and disk at the supersampled resolution,
    averaged down, plus point stars at output resolution, then optional
    glare and tone mapping. Returns (display sRGB, linear)."""
    img = np.zeros(tr["outcome"].shape + (3,))
    if sky is not None:
        img += sky_radiance(tr, sky)
    if disk is not None:
        img += disk_gain * disk_radiance(tr, a, **disk)
    lin = downsample(img, ss)
    if stars is not None:
        lin += stars_image(tr, stars, ss, gain=star_gain)
    out = bloom(lin) if glare else lin
    return tonemap(out, exposure), lin


def stars_image(tr, stars, ss, sigma=0.7, gain=1.0):
    """Point-star layer at output resolution (see sky.lensed_stars)."""
    from .sky import lensed_stars, paint_stars
    from .kerr import Camera
    cam_rh = Camera.right_handed(tr["cam"])
    idx, x, y, mu, om = lensed_stars(cam_rh, tr["sky"], tr["outcome"] == kerr.ESCAPED, stars)
    ny, nx = tr["outcome"].shape[0] // ss, tr["outcome"].shape[1] // ss
    # supersample-grid coordinates -> output pixels; a pixel is 2 ss^2 triangles
    xo = (x + 0.5) / ss - 0.5
    yo = (y + 0.5) / ss - 0.5
    return paint_stars((ny, nx), stars, idx, xo, yo, mu, 2 * np.abs(om) * ss * ss, sigma, gain)


def downsample(img, ss):
    ny, nx = img.shape[0] // ss, img.shape[1] // ss
    return img[: ny * ss, : nx * ss].reshape(ny, ss, nx, ss, -1).mean((1, 3))


