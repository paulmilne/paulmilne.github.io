"""Null geodesics of a Kerr black hole (M = 1) in Boyer-Lindquist coordinates,
integrated as a Hamiltonian system with fixed-step RK4 and a step size that
shrinks near the hole. The same arithmetic runs in any floating-point format:
pass dtype=np.float64, np.float32 or np.float16, or rnd=bf16 to round every
intermediate result to bfloat16. The browser renderer (gargantua_gl.js in the post folder)
uses identical equations in 32-bit floats.

H = [ D pr^2 + pth^2 - P^2/D + (L/s - a s)^2 ] / (2 S),
    S = r^2 + a^2 cos^2 th,  D = r^2 - 2r + a^2,  P = r^2 + a^2 - a L,  s = sin th,
with the photon's energy at infinity scaled to E = 1 and L its angular momentum."""
import numpy as np

def bf16(x):
    """Round float32 values to bfloat16 (8-bit significand), round-to-nearest-even."""
    x = np.asarray(x, np.float32); b = x.view(np.uint32)
    b = ((b + np.uint32(0x7FFF) + ((b >> 16) & 1)) & np.uint32(0xFFFF0000)).astype(np.uint32)
    return b.view(np.float32)

def deriv(r, th, pr, pth, L, a, R):
    """Right-hand side of Hamilton's equations. R rounds each intermediate."""
    c = R(np.cos(th)); s = R(np.sin(th)); s = R(np.where(np.abs(s) < 1e-3, np.copysign(1e-3, s + 1e-30), s))
    r2 = R(r * r); a2 = a * a
    S = R(r2 + R(a2 * R(c * c))); D = R(R(r2 - R(2 * r)) + a2); P = R(R(r2 + a2) - R(a * L))
    q = R(R(L / s) - R(a * s))                              # L/s - a s
    A = R(R(R(D * R(pr * pr)) + R(pth * pth)) - R(R(P * P) / D) + R(q * q))
    inv2S = R(1 / R(2 * S))
    dr = R(R(D * pr) / S); dth = R(pth / S)
    dph = R(R(R(R(a * P) / D) + R(R(L / s - a * s) / s)) / S)
    Dp = R(2 * r - 2)
    dA_dr = R(R(Dp * R(pr * pr)) - R(R(R(4 * r) * P) / D) + R(R(R(P * P) * Dp) / R(D * D)))
    dA_dth = R(R(R(2 * q) * R(R(-L / R(s * s)) - a)) * c)
    dpr = R(-R(R(dA_dr * inv2S) - R(R(A * inv2S) * R(R(2 * r) / S))))
    dpth = R(-R(R(dA_dth * inv2S) - R(R(A * inv2S) * R(R(R(-2 * a2) * c) * s) / S)))
    return dr, dth, dph, dpr, dpth

def camera_rays(nx, ny, fov_deg, r0, th0, a, yaw=0.0):
    """Initial conditions for a pinhole camera carried by a zero-angular-momentum observer
    (ZAMO) at (r0, th0), looking at the hole. Returns float64 arrays."""
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    f = 0.5 * nx / np.tan(np.radians(fov_deg) / 2)
    x = (i + 0.5 - nx / 2) / f; y = -(j + 0.5 - ny / 2) / f
    n = np.stack([-np.ones_like(x), y, x], 0); n /= np.linalg.norm(n, axis=0)   # (radial, theta, phi) in the ZAMO frame
    nr, nth, nph = n[0], -n[1], n[2]                      # n[0] < 0: looking inwards; +theta is downwards on screen
    nr, nph = nr * np.cos(yaw) - nph * np.sin(yaw), nr * np.sin(yaw) + nph * np.cos(yaw)
    S = r0**2 + a**2 * np.cos(th0)**2; D = r0**2 - 2 * r0 + a**2
    Aq = (r0**2 + a**2)**2 - a**2 * D * np.sin(th0)**2
    alpha = np.sqrt(S * D / Aq); om = 2 * a * r0 / Aq; varpi = np.sqrt(Aq / S) * np.sin(th0)
    E = alpha + om * varpi * nph                            # energy at infinity per unit local energy
    pr = np.sqrt(S / D) * nr / E; pth = np.sqrt(S) * nth / E; L = varpi * nph / E
    return pr, pth, L, E

def trace(pr, pth, L, r0, th0, a, dtype=np.float64, rnd=None, r_far=60.0, disk=(None, None), max_steps=4000, h0=0.05):
    """Integrate every ray until it crosses the horizon, escapes beyond r_far, or hits the
    equatorial disk between disk[0] and disk[1]. Returns fate (0 escaped, 1 horizon, 2 disk),
    final theta and phi, the disk radius hit, and the number of steps."""
    R = (lambda x: x) if rnd is None else rnd
    T = (lambda x: np.array(x, dtype, copy=True)) if rnd is None else (lambda x: rnd(np.array(x, np.float32, copy=True)))
    sh = pr.shape; pr = T(pr.ravel()); pth = T(pth.ravel()); L = T(L.ravel()); a_ = T(a)
    r = T(np.full(pr.shape, r0)); th = T(np.full(pr.shape, th0)); ph = T(np.zeros(pr.shape))
    rh = 1 + np.sqrt(1 - a * a); fate = np.full(pr.shape, -1, np.int8); rdisk = np.full(pr.shape, np.nan); steps = np.zeros(pr.shape, np.int32)
    live = np.ones(pr.shape, bool)
    for k in range(max_steps):
        idx = np.nonzero(live)[0]
        if idx.size == 0: break
        y = [v[idx] for v in (r, th, ph, pr, pth)]; Ll = L[idx]
        h = T(h0 * np.clip(np.asarray(y[0], np.float64) - rh, 0.05, None) ** 1.0 * np.minimum(1, 20 / np.asarray(y[0], np.float64)) + 0.002)
        h = np.minimum(np.asarray(h, np.float64), 1.5) * np.clip(5 * np.abs(np.sin(np.asarray(y[1], np.float64))), 0.02, 1)   # shorter steps near the poles
        h = T(h)
        def F(st):
            d = deriv(st[0], st[1], st[3], st[4], Ll, a_, R); return [d[0], d[1], d[2], d[3], d[4]]
        k1 = F(y); y2 = [R(v + R(h / 2 * d)) for v, d in zip(y, k1)]
        k2 = F(y2); y3 = [R(v + R(h / 2 * d)) for v, d in zip(y, k2)]
        k3 = F(y3); y4 = [R(v + R(h * d)) for v, d in zip(y, k3)]
        k4 = F(y4)
        new = [R(v + R(R(h / 6) * R(R(d1 + R(2 * d2)) + R(R(2 * d3) + d4)))) for v, d1, d2, d3, d4 in zip(y, k1, k2, k3, k4)]
        th_old = np.asarray(y[1], np.float64); th_new = np.asarray(new[1], np.float64); r_new = np.asarray(new[0], np.float64)
        steps[idx] += 1
        done = np.zeros(idx.size, bool)
        if disk[0] is not None:
            cross = (th_old - np.pi / 2) * (th_new - np.pi / 2) < 0
            frac = (np.pi / 2 - th_old) / np.where(cross, th_new - th_old, 1)
            rc = np.asarray(y[0], np.float64) + frac * (r_new - np.asarray(y[0], np.float64))
            hit = cross & (rc >= disk[0]) & (rc <= disk[1])
            fate[idx[hit]] = 2; rdisk[idx[hit]] = rc[hit]; done |= hit
        fell = ~done & ((r_new < rh * 1.01) | ~np.isfinite(r_new)); fate[idx[fell]] = 1; done |= fell
        esc = ~done & (r_new > r_far); fate[idx[esc]] = 0; done |= esc
        for v, nv in zip((r, th, ph, pr, pth), new): v[idx] = nv
        live[idx[done]] = False
    fate[fate < 0] = 3                                     # still integrating at max_steps
    return dict(fate=fate.reshape(sh), th=np.asarray(th, np.float64).reshape(sh), ph=np.asarray(ph, np.float64).reshape(sh),
                rdisk=rdisk.reshape(sh), steps=steps.reshape(sh))
