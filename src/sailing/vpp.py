"""A small velocity prediction program (VPP) for a sailing keelboat.

Three balances are solved at each true wind speed and angle:
  drive      sail force along the course   = hull and appendage resistance
  side force sail force across the course  = keel, rudder and hull lift (sets the leeway)
  heel       sail heeling moment           = righting moment (hull, ballast and crew)
The sail trim is then chosen to give the highest speed: "flat" scales the sails' lift
(flattening, easing, twisting off) and "reef" scales the sail area (reefing).

Structure and coefficient tables follow the ORC VPP documentation (Offshore Racing
Congress): sail lift and drag coefficients against apparent wind angle (main, jib,
asymmetric spinnaker), effective rig height, and the ORC residuary-resistance surfaces
for the canoe body. The tables in dat/ are as digitised in Python-VPP
(github.com/marinlauber/Python-VPP, MIT licence). The code here is independent and
simplified: see the notes on each function.

Units: SI throughout (m, kg, s, N); angles in degrees at the interface.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DAT = Path(__file__).parent / "dat"
G = 9.81
RHO_W, NU_W = 1025.0, 1.19e-6          # sea water
RHO_A = 1.225                          # air
KN = 0.514444                          # m/s per knot
Z_REF, Z0 = 10.0, 0.005                # true wind given at 10 m; log-profile roughness (ORC)


# ---------------------------------------------------------------------------
# Tables

def _sail_table(name):
    """ORC sail table: kp (parasitic quadratic factor), AWA, CD, CL. Linear interpolation
    on a fine cubic resampling of the published points."""
    rows = [r for r in (DAT / f"{name}.dat").read_text().splitlines() if not r.startswith("#")]
    vals = [[float(x) for x in r.split(",") if x.strip()] for r in rows]
    kp, awa, cd, cl = vals[0][0], np.array(vals[1]), np.array(vals[2]), np.array(vals[3])
    return kp, awa, cd, cl


def _pchip(x, y, xq):
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    k = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = (xq - x[k]) / h[k]
    return ((2 * t**3 - 3 * t**2 + 1) * y[k] + (t**3 - 2 * t**2 + t) * h[k] * m[k]
            + (-2 * t**3 + 3 * t**2) * y[k + 1] + (t**3 - t**2) * h[k] * m[k + 1])


class Coeff:
    """Sail coefficient curve sampled every 0.5 degrees (monotone cubic through the table)."""
    GRID = np.arange(0.0, 180.01, 0.5)

    def __init__(self, name):
        self.name = name
        self.kp, self.awa, cd, cl = _sail_table(name)
        self.cd = _pchip(self.awa, cd, self.GRID)
        self.cl = _pchip(self.awa, cl, self.GRID)
        self.cd_pts, self.cl_pts = cd, cl

    def __call__(self, awa):
        a = min(max(awa, 0.0), 180.0)
        return np.interp(a, self.GRID, self.cl), np.interp(a, self.GRID, self.cd)


def _table2(name):
    rows = (DAT / f"{name}.dat").read_text().splitlines()
    vals = [[float(x) for x in r.split(",") if x.strip()] for r in rows if not r.startswith("#")]
    return np.array(vals[0]), np.array(vals[1])


KHEFF = _table2("kheff")          # effective-height factor against AWA (ORC)
FCDMULT = _table2("fcdmult")      # drag multiplier against flat (ORC)


def _load_rr():
    """ORC residuary-resistance surfaces: Rr / (displacement weight) x 1000 against
    Froude number, length-volume ratio LVR (rows, 3 to 9) and beam-draft ratio BTR
    (columns, 2.5 to 9). Rr rises steeply as LVR falls and hardly changes with BTR."""
    s = np.genfromtxt(DAT / "ORCi_Drag_Surfaces.csv", delimiter=",", skip_header=1).reshape((24, 43, 42))
    fn = np.r_[0.0, [s[k, 0, 3] for k in range(24)]]
    lvr, btr = s[0, 2:, 0], s[0, 1, 1:]
    data = np.zeros((25, len(lvr), len(btr)))
    data[1:] = s[:, 2:, 1:]
    return fn, lvr, btr, data


RR_FN, RR_LVR, RR_BTR, RR = _load_rr()


def rr_coeff(fn, lvr, btr):
    """Trilinear interpolation in the ORC surfaces (Fn clipped to the table)."""
    fn = min(max(fn, 0.0), RR_FN[-1])
    def idx(ax, v):
        v = min(max(v, ax[0]), ax[-1])
        i = int(np.clip(np.searchsorted(ax, v) - 1, 0, len(ax) - 2))
        return i, (v - ax[i]) / (ax[i + 1] - ax[i])
    i, a = idx(RR_FN, fn)
    j, b = idx(RR_LVR, lvr)
    k, c = idx(RR_BTR, btr)
    out = 0.0
    for di, wi in ((0, 1 - a), (1, a)):
        for dj, wj in ((0, 1 - b), (1, b)):
            for dk, wk in ((0, 1 - c), (1, c)):
                out += wi * wj * wk * RR[i + di, j + dj, k + dk]
    return out


# ---------------------------------------------------------------------------
# The boat

@dataclass
class Foil:
    """Keel or rudder: trapezoid of root chord, tip chord and span (m)."""
    root: float
    tip: float
    span: float
    tc: float                     # thickness-chord ratio

    @property
    def area(self):
        return 0.5 * (self.root + self.tip) * self.span

    @property
    def cla(self):
        """Lift per radian of leeway x area. Aspect ratio doubled for the hull's
        reflection (lifting-line slope 2 pi Ae / (Ae + 2))."""
        ae = 2 * self.span**2 / self.area
        return 2 * np.pi * ae / (ae + 2) * self.area

    @property
    def teff(self):
        return 1.8 * self.span     # effective span for induced drag (ORC)


@dataclass
class Boat:
    name: str
    lwl: float                    # waterline length
    bwl: float                    # waterline beam
    tc: float                     # canoe-body draft
    beam: float
    tmax: float                   # draft
    mass: float                   # sailing mass incl. crew, kg
    wss: float                    # canoe-body wetted surface
    crew: float                   # crew mass, kg
    crew_arm: float               # crew's mean distance from the centreline when hiking, m
    gm: float                     # metacentric height: GZ = GM sin(heel), m
    keel: Foil
    rudder: Foil
    I: float
    J: float
    P: float
    E: float
    hbi: float                    # height of the sheerline at the mast above the waterline
    bad: float                    # boom above deck
    main_area: float
    jib_area: float
    kite_area: float
    freeboard: float = 1.1
    lsm: float = 0.0              # sailing length (fitted); 0 means use the waterline length
    sails: dict = field(default_factory=dict)

    @property
    def vol(self):
        return self.mass / RHO_W

    @property
    def ls(self):
        return self.lsm if self.lsm > 0 else self.lwl

    @property
    def lvr(self):
        return self.ls / self.vol ** (1 / 3)

    @property
    def btr(self):
        return self.bwl / self.tc


def j109(gm=1.0, **kw):
    """The J/109. Sources: ORC certificates 2024 (median of 43): displacement 5,089 kg,
    crew 620 kg, wetted surface 26.23 m2, draft 2.14 m, main 37.83 m2, jib 31.65 m2,
    asymmetric spinnaker 108.3 m2. J/109 class association: LOA 10.75 m, LWL 9.30 m,
    beam 3.51 m, ballast 1,769 kg, I 14.17, J 4.05, P 13.18, E 4.72 m.
    Assumed (not published): waterline beam, canoe-body draft, keel and rudder planforms,
    heights of deck and boom, crew arm. GM is fitted to the ORC polars."""
    b = dict(
        name="J/109", lwl=9.30, bwl=3.00, tc=0.50, beam=3.51, tmax=2.14,
        mass=5089.0 + 620.0, wss=26.23, crew=620.0, crew_arm=1.45, gm=gm,
        keel=Foil(1.00, 0.62, 2.14 - 0.50, 0.12), rudder=Foil(0.46, 0.30, 1.45, 0.11),
        I=14.17, J=4.05, P=13.18, E=4.72, hbi=1.35, bad=0.95,
        main_area=37.83, jib_area=31.65, kite_area=108.3)
    b.update(kw)
    return Boat(**b)


MAIN, JIB, KITE = Coeff("main"), Coeff("jib"), Coeff("asym_cl_kite")


# ---------------------------------------------------------------------------
# Aerodynamics

def wind_gradient(z):
    return np.log(max(z, 10 * Z0) / Z0) / np.log(Z_REF / Z0)


def apparent(tws, twa, vb, heel=0.0, z=Z_REF):
    """Apparent wind angle (deg) and speed at height z. The true wind is scaled to z by the
    log profile; its cross-track part is projected into the heeled mast plane (ORC)."""
    t = np.radians(twa)
    vt = tws * wind_gradient(z)
    vy = vt * np.sin(t) * np.cos(np.radians(heel))
    vx = vt * np.cos(t) + vb
    return np.degrees(np.arctan2(vy, vx)), np.hypot(vx, vy)


def sail_plan(boat, downwind):
    """(sail, area, height of its centre of effort above the waterline) for each sail set."""
    if downwind:
        return [(MAIN, boat.main_area, boat.hbi + boat.bad + 0.39 * boat.P),
                (KITE, boat.kite_area, boat.hbi + 0.565 * boat.I)]
    return [(MAIN, boat.main_area, boat.hbi + boat.bad + 0.39 * boat.P),
            (JIB, boat.jib_area, boat.hbi + 0.39 * boat.I)]


def aero(boat, vb, heel, tws, twa, flat, reef, downwind):
    """Sail forces in the course frame: drive (along the course), side force (across it,
    in the heeled mast plane), heeling moment about the waterline, and the pieces for
    the page. flat scales lift (and, through the ORC multiplier, drag); reef scales the
    area by reef^2 and the centre of effort by reef."""
    plan = sail_plan(boat, downwind)
    area0 = sum(a for _, a, _ in plan)
    zce0 = sum(a * z for _, a, z in plan) / area0
    awa, aws = apparent(tws, twa, vb, heel, zce0 * reef)
    cl = cd = kpp = 0.0
    zsum = 0.0
    for s, a, z in plan:
        l_, d_ = s(awa)
        cl += l_ * a
        cd += d_ * a
        kpp += l_**2 * a * s.kp
        zsum += a * z * np.hypot(l_, d_)
    cl, cd = cl / area0, cd / area0
    # effective rig height for induced drag (ORC: height of the rig times kheff(AWA))
    b_rig = max(boat.P + boat.bad, boat.I) + boat.hbi
    heff = b_rig * 1.086 * np.interp(min(awa, 90.0), *KHEFF)
    if downwind:
        heff = b_rig * 0.95
    ce = (kpp / (area0 * cl**2) if cl else 0.0) + area0 / (np.pi * heff**2)
    fm = np.interp(flat, *FCDMULT)
    cd_tot = cd * fm + ce * (cl * flat) ** 2 * fm
    cl_tot = cl * flat
    area = area0 * reef**2
    q = 0.5 * RHO_A * aws**2
    lift, drag = q * area * cl_tot, q * area * cd_tot
    # hull windage (ORC: frontal area of the topsides, drag coefficient 0.816)
    _, aws_h = apparent(tws, twa, vb, heel, 0.5 * boat.freeboard)
    s2 = 0.5 * (1 - np.cos(2 * np.radians(awa)))
    windage = 0.5 * RHO_A * aws_h**2 * boat.freeboard * ((1 - s2) * boat.beam + s2 * 10.75) * 0.816
    # mast and standing rigging: a 0.18 m section over the mast height plus shrouds and stays,
    # in cross-flow (drag coefficient about 1), at the wind speed half-way up the rig
    _, aws_r = apparent(tws, twa, vb, heel, boat.hbi + 0.5 * boat.I)
    windage += 0.5 * RHO_A * aws_r**2 * (0.18 * (boat.I + 0.5) + 0.6) * 1.0
    a = np.radians(awa)
    drive = lift * np.sin(a) - (drag + windage) * np.cos(a)
    side = lift * np.cos(a) + (drag + windage) * np.sin(a)
    zce = zsum / np.hypot(cl, cd) / area0 if np.hypot(cl, cd) > 0 else zce0
    zce = boat.hbi + (zce - boat.hbi) * reef * (1 - 0.25 * (1 - flat))   # flattening twists the head off
    return dict(awa=awa, aws=aws, lift=lift, drag=drag + windage, drive=drive, side=side,
                hm=side * zce, zce=zce, cl=cl_tot, cd=cd_tot, area=area)


# ---------------------------------------------------------------------------
# Hydrodynamics

def cf_hughes(re):
    return 0.066 / (np.log10(max(re, 1e4)) - 2.03) ** 2


def cf_ittc(re):
    return 0.075 / (np.log10(max(re, 1e4)) - 2.0) ** 2


def hydro(boat, vb, heel, side):
    """Resistance components (N) and leeway (deg) for a given side force to be carried."""
    q = 0.5 * RHO_W * vb**2
    fn = vb / np.sqrt(G * boat.ls)
    rr = rr_coeff(fn, boat.lvr, boat.btr) * boat.mass * G * 1e-3
    if heel > 30:                                       # immersed rail (ORC)
        rr *= 1 + 0.0004 * (heel - 30) ** 2
    rv = q * boat.wss * cf_hughes(vb * 0.85 * boat.ls / NU_W) * 1.05
    app = 0.0
    for f in (boat.keel, boat.rudder):
        c = 0.5 * (f.root + f.tip)
        app += q * 2 * f.area * cf_ittc(vb * c / NU_W) * (1 + 1.75 * f.tc + 30 * f.tc**4)
    # side force shared by keel, rudder (more of it as the boat heels) and canoe body
    hull_area = boat.lwl * boat.tc * 0.666
    cla = np.array([boat.keel.cla,
                    boat.rudder.cla * (1 - np.cos(min(heel, 30) / 30 * np.pi)) if heel < 30 else 2 * boat.rudder.cla,
                    hull_area * 2 * np.pi / (1 + 0.5 * hull_area / boat.tc)])
    teff = np.array([boat.keel.teff, boat.rudder.teff, 2.07 * boat.tc])
    if q <= 0:
        return dict(rr=rr, rv=rv, app=app, ri=0.0, total=rr + rv + app, leeway=0.0, fn=fn)
    lam = side / (q * cla.sum())                        # leeway, radians
    share = q * cla * lam
    ri = np.sum(share**2 / (q * np.pi * teff**2))
    ri /= max(np.cos(np.radians(heel)), 0.3) ** 2        # heeled foils have less vertical span
    return dict(rr=rr, rv=rv, app=app, ri=ri, total=rr + rv + app + ri, leeway=np.degrees(lam), fn=fn)


def righting(boat, heel, side_force):
    """Righting moment (N m): hull and ballast GM sin(heel), the crew on the rail (faded in
    from 2.5 to 7.5 deg of heel, as in the ORC VPP), less the moment of the keel's side
    force acting 0.43 x draft below the waterline (ORC)."""
    h = np.radians(heel)
    rm_hull = boat.mass * G * boat.gm * np.sin(h)
    ramp = 0.0 if heel <= 2.5 else 1.0 if heel >= 7.5 else 0.5 * (1 - np.cos((heel - 2.5) / 5 * np.pi))
    rm_crew = boat.crew * G * boat.crew_arm * np.cos(h) * ramp
    return rm_hull, rm_crew, side_force * 0.43 * boat.tmax


# ---------------------------------------------------------------------------
# Equilibrium

def _brent(f, a, b, fa=None, fb=None, tol=1e-6, it=80):
    fa = f(a) if fa is None else fa
    fb = f(b) if fb is None else fb
    if fa * fb > 0:
        return None
    c, fc = a, fa
    d = e = b - a
    for _ in range(it):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2e-12 * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol1 or fb == 0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2 * m * s, 1 - s
            else:
                q_, r = fa / fc, fb / fc
                p = s * (2 * m * q_ * (q_ - r) - (b - a) * (r - 1))
                q = (q_ - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            if 2 * p < min(3 * m * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if m > 0 else -tol1)
        fb = f(b)
    return b


def heel_for(boat, vb, tws, twa, flat, reef, downwind):
    """Heel at which the sails' heeling moment equals the righting moment."""
    def g(phi):
        a = aero(boat, vb, phi, tws, twa, flat, reef, downwind)
        rh, rc, rk = righting(boat, phi, a["side"])
        return a["hm"] - (rh + rc - rk)
    g0 = g(0.0)
    if g0 <= 0:
        return 0.0
    phi = _brent(g, 0.0, 70.0, fa=g0)
    return 70.0 if phi is None else phi


def state(boat, vb, tws, twa, flat, reef, downwind):
    phi = heel_for(boat, vb, tws, twa, flat, reef, downwind)
    a = aero(boat, vb, phi, tws, twa, flat, reef, downwind)
    h = hydro(boat, vb, phi, a["side"])
    return phi, a, h


def speed_for(boat, tws, twa, flat, reef, downwind):
    """Boat speed (m/s) where drive equals resistance, at the given trim."""
    def f(v):
        _, a, h = state(boat, v, tws, twa, flat, reef, downwind)
        return a["drive"] - h["total"]
    # At very low speed the keel cannot carry the side force without huge leeway and induced
    # drag, so f < 0 there as well as at high speed: take the upper root.
    lo, flo = None, None
    v = 0.4
    while v < 16:
        fv = f(v)
        if fv > 0:
            lo, flo = v, fv
        elif lo is not None:
            hi, fhi = v, fv
            break
        v += 0.4
    else:
        return 0.0 if lo is None else lo
    if lo is None:
        return 0.0
    v = _brent(f, lo, hi, fa=flo, fb=fhi, tol=1e-5)
    return 0.0 if v is None else v


def golden(f, a, b, tol=0.01):
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2


MAX_HEEL = 30.0     # crews depower or reef rather than sail a boat like this beyond about 30 deg


def best_trim(boat, tws, twa, downwind, max_heel=MAX_HEEL):
    """Flat and reef that give the highest speed with the heel no more than max_heel.
    A coarse scan of flat (and, upwind, of reef when the boat is over-pressed), then a
    golden-section refinement of flat around the best. Returns (v, flat, reef)."""
    def score(fl, rf):
        v = speed_for(boat, tws, twa, fl, rf, downwind)
        if v <= 0:
            return -1.0, 0.0
        phi = heel_for(boat, v, tws, twa, fl, rf, downwind)
        return v - 0.2 * max(0.0, phi - max_heel), v

    def best_at(rf):
        flats = np.arange(0.4, 1.001, 0.1)
        sc = [score(f, rf)[0] for f in flats]
        k = int(np.argmax(sc))
        if sc[k] <= 0:
            return -1.0, 1.0
        lo, hi = max(0.4, flats[k] - 0.1), min(1.0, flats[k] + 0.1)
        fl = golden(lambda x: score(x, rf)[0], lo, hi, tol=0.01)
        s_ = score(fl, rf)[0]
        return (s_, fl) if s_ >= sc[k] else (sc[k], flats[k])

    s_best, f_best = best_at(1.0)
    r_best = 1.0
    if not downwind:
        _, v0 = score(f_best, 1.0)
        phi0 = heel_for(boat, v0, tws, twa, f_best, 1.0, downwind) if v0 > 0 else 90.0
        if s_best <= 0 or f_best < 0.62 or phi0 > max_heel - 0.5:
            for rf in (0.9, 0.8, 0.7, 0.6):
                s_, fl = best_at(rf)
                if s_ > s_best:
                    s_best, f_best, r_best = s_, fl, rf
                elif s_best > 0:
                    break
            if r_best < 1.0:       # refine the reef between its neighbours
                def at_r(rf):
                    return best_at(rf)[0]
                r = golden(at_r, max(0.55, r_best - 0.1), min(1.0, r_best + 0.1), tol=0.02)
                s_, fl = best_at(r)
                if s_ > s_best:
                    s_best, f_best, r_best = s_, fl, r
    v = speed_for(boat, tws, twa, f_best, r_best, downwind)
    return v, f_best, r_best


def solve_point(boat, tws_kn, twa):
    """Best of the upwind (main and jib) and downwind (main and spinnaker) sail sets.
    The spinnaker is only tried from 70 deg TWA, as on the ORC certificates' polars."""
    tws = tws_kn * KN
    best = None
    for dw in ((False, True) if twa >= 70 else (False,)):
        v, fl, rf = best_trim(boat, tws, twa, dw)
        if best is None or v > best[0]:
            best = (v, fl, rf, dw)
    v, fl, rf, dw = best
    phi, a, h = state(boat, v, tws, twa, fl, rf, dw)
    rh, rc, rk = righting(boat, phi, a["side"])
    return dict(tws=tws_kn, twa=twa, v=v / KN, heel=phi, leeway=h["leeway"], flat=fl, reef=rf,
                kite=dw, awa=a["awa"], aws=a["aws"] / KN, lift=a["lift"], drag=a["drag"],
                drive=a["drive"], side=a["side"], hm=a["hm"], zce=a["zce"], cl=a["cl"], cd=a["cd"],
                area=a["area"], rr=h["rr"], rv=h["rv"], rapp=h["app"], ri=h["ri"], rtot=h["total"],
                fn=h["fn"], rm_hull=rh, rm_crew=rc, rm_keel=rk)
