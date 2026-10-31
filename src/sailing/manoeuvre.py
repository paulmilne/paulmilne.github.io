"""Tacks and gybes for the J/109: a time-domain extension of the steady model in vpp.py.

The helmsman's turn is prescribed (heading against time); everything else follows from
the physics, second by second:
  surge  (m + m_x) du/dt = drive - resistance
  roll   (I + I_a) d2(phi)/dt2 = heeling moment - righting moment - damping
  sway   quasi-steady: the keel takes whatever leeway it needs to carry the sails' side
         force plus the centripetal force of the turn (m u r), capped at the keel's stall.
Sail forces come from vpp.aero with the ORC coefficients, scaled by a "fill" factor: the
sails luff (lose their lift and flog) when the apparent wind is within about 20 degrees of
the bow, and need a few seconds to be sheeted in on the new side after a tack; in a gybe
the spinnaker partly collapses while it is gybed. The crew move across in a few seconds.

Assumed, not published (flagged in the post): roll radius of gyration, added mass and
roll inertia, roll damping, the luffing drag of flogging sails, the timing of the crew's
sheet and weight changes, and the turn itself.
"""
import json
import sys
from pathlib import Path

import numpy as np

import vpp as V

KN, G = V.KN, V.G
LOA = 10.75

# Assumed dynamic properties (see module docstring).
ADDED_SURGE = 0.05          # surge added mass as a fraction of the displacement
K_ROLL = 1.45               # roll radius of gyration, m (mast and keel make it large)
ADDED_ROLL = 0.3            # roll added inertia as a fraction of the dry inertia
ZETA_ROLL = 0.15            # roll damping, fraction of critical
CD_LUFF = 0.15              # drag coefficient of flogging sails (on the full sail area)
STALL_DEG = 12.0            # the keel's leeway cannot exceed this
TRIM_S = 4.0                # time to sheet in fully on the new tack after the bow crosses the wind
CREW_S = 3.0                # time for the crew to move to the new windward rail
KITE_COLLAPSE_S = 3.0       # duration of the spinnaker's partial collapse in a gybe
KITE_COLLAPSE = 0.35        # fraction of its drive left at the worst moment


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Run:
    def __init__(self, boat, tws_kn, kind, turn_s, entry, flat, reef, kite, t_before=10.0, t_after=70.0, lee0=0.0):
        self.b, self.kind, self.T = boat, kind, turn_s
        self.tws = tws_kn * KN
        # entry is the track angle of the steady state; the helmsman steers the heading, which is
        # the track less the leeway, and comes out at the mirror-image heading
        self.entry, self.flat, self.reef, self.kite = entry - lee0, flat, reef, kite
        self.lee0 = lee0
        self.t0, self.t1 = t_before, t_before + turn_s
        self.t_end = t_before + turn_s + t_after
        m = boat.mass
        self.mx = m * (1 + ADDED_SURGE)
        self.I = m * K_ROLL**2 * (1 + ADDED_ROLL)
        omega = np.sqrt(m * G * boat.gm / self.I)
        self.c_roll = 2 * ZETA_ROLL * self.I * omega
        self.cross = None                  # time the wind crosses the bow (tack) or stern (gybe)

    # heading: signed true wind angle of the bow, + with the wind over starboard
    def twa(self, t):
        s = smooth((t - self.t0) / self.T) if self.T > 0 else float(t >= self.t0)
        if self.kind == "tack":
            return self.entry * (1 - 2 * s)
        a = self.entry + (360 - 2 * self.entry) * s            # through 180
        return a if a <= 180 else a - 360

    def turn_rate(self, t, h=0.01):
        a, b = self.twa(t - h), self.twa(t + h)
        d = (b - a + 540) % 360 - 180
        return np.radians(d) / (2 * h)

    def fill(self, t, awa_abs):
        """Fraction of the sails' lift available."""
        if self.kind == "tack":
            f = smooth((awa_abs - 10.0) / 12.0)                  # luffing near head to wind
            if self.cross is not None and t >= self.cross:
                f *= 0.45 + 0.55 * smooth((t - self.cross) / TRIM_S)   # sheeting in on the new side
            return f
        f = 1.0
        if self.cross is not None and self.kite:
            x = abs(t - self.cross) / (KITE_COLLAPSE_S / 2)
            f = 1 - (1 - KITE_COLLAPSE) * (1 - smooth(x))
        return f

    def crew_side(self, t, side0):
        if self.kind != "tack" or self.cross is None or t < self.cross:
            return side0
        return side0 * (1 - 2 * smooth((t - self.cross) / CREW_S))

    def forces(self, t, u, phi):
        b = self.b
        twa = self.twa(t)
        side0 = 1.0 if self.entry > 0 else -1.0
        sgn = 1.0 if twa >= 0 else -1.0                         # + : wind over starboard, sails to port
        # crossing time
        if self.cross is None and t > self.t0 and np.sign(twa) != np.sign(self.entry):
            self.cross = t
        u_ = max(u, 0.05)
        # leeway from the previous step (quasi-steady), applied to the track direction
        lee = getattr(self, "_lee", 0.0)
        twa_track = min(abs(twa) + lee, 180.0)
        a0 = V.aero(b, u_, abs(phi), self.tws, twa_track, self.flat, self.reef, self.kite)
        f = self.fill(t, abs(a0["awa"]))
        a = V.aero(b, u_, abs(phi), self.tws, twa_track, max(1e-3, self.flat * f), self.reef, self.kite) if f < 0.999 else a0
        drive, side = a["drive"], a["side"]
        q = 0.5 * V.RHO_A * a["aws"] ** 2
        luff = q * a["area"] * CD_LUFF * (1 - f) if self.kind == "tack" else 0.0
        ar = np.radians(a["awa"])
        drive -= luff * np.cos(ar)
        side += luff * np.sin(ar)
        # lateral balance in the boat's frame: + to starboard
        A_y = -sgn * side                                         # sails push to leeward (balance as in vpp.py)
        r = self.turn_rate(t)
        C_y = -self.b.mass * u_ * r                               # centripetal force needed (r > 0 turning to port)
        K_y = C_y - A_y                                           # keel and rudder supply the rest
        h = V.hydro(b, u_, abs(phi), abs(K_y))
        lee_deg = min(h["leeway"], STALL_DEG)
        if h["leeway"] > STALL_DEG:                               # stalled: extra drag
            h["total"] += 0.5 * abs(K_y) * np.sin(np.radians(h["leeway"] - STALL_DEG))
        self._lee = lee_deg
        # roll: + heel to port. Sails heel the boat to leeward; the keel's side force acts below the water.
        hm = sgn * side * a["zce"]                              # sails heel the boat to leeward
        hull = b.mass * G * b.gm * np.sin(np.radians(phi))
        # the crew's righting moment as in vpp.py (on the rail once the boat heels 2.5-7.5 deg), on the
        # windward side, moving across after a tack
        crew = V.righting(b, abs(phi), 0.0)[1] * self.crew_side(t, side0)
        keel_m = K_y * 0.43 * b.tmax                              # keel force to starboard, below the water, heels to port
        roll_moment = hm + keel_m - hull - crew
        return dict(drive=drive, rtot=h["total"], roll=roll_moment, twa=twa, awa=sgn * a["awa"], aws=a["aws"],
                    lift=a["lift"], drag=a["drag"], side=side, keel=K_y, leeway=lee_deg, fill=f, r=r,
                    zce=a["zce"], crew=crew)

    def simulate(self, u0, phi0, dt=0.02, out_dt=0.25):
        t, u, phi, p = 0.0, u0, phi0, 0.0
        x = y = 0.0
        rows = []
        nxt = 0.0
        self._lee = self.lee0
        while t <= self.t_end + 1e-9:
            f1 = self.forces(t, u, phi)
            # Heun step
            du1, dp1 = (f1["drive"] - f1["rtot"]) / self.mx, (f1["roll"] - self.c_roll * p) / self.I
            u2, phi2, p2 = u + dt * du1, phi + dt * np.degrees(p), p + dt * dp1
            f2 = self.forces(t + dt, u2, phi2)
            du2, dp2 = (f2["drive"] - f2["rtot"]) / self.mx, (f2["roll"] - self.c_roll * p2) / self.I
            # track: heading rotated to leeward by the leeway; wind from +y (upwind is +y)
            sgn = 1.0 if f1["twa"] >= 0 else -1.0
            trk = np.radians(f1["twa"] + sgn * f1["leeway"])
            if t >= nxt - 1e-9:
                rows.append([round(t, 3), round(f1["twa"], 3), round(u / KN, 4), round(phi, 3), round(sgn * f1["leeway"], 3),
                             round(f1["awa"], 3), round(f1["aws"] / KN, 3), round(f1["drive"], 1), round(f1["rtot"], 1),
                             round(f1["lift"], 1), round(f1["drag"], 1), round(f1["side"], 1), round(f1["keel"], 1),
                             round(x, 3), round(y, 3), round(f1["fill"], 3), round(f1["crew"], 1), round(np.degrees(f1["r"]), 3)])
                nxt += out_dt
            x += dt * u * (-np.sin(trk))
            y += dt * u * np.cos(trk)
            u = max(0.05, u + dt * 0.5 * (du1 + du2))
            phi = phi + dt * np.degrees(0.5 * (p + p2))
            p = p + dt * 0.5 * (dp1 + dp2)
            t += dt
        return rows


COLUMNS = ["t", "twa", "v", "heel", "leeway", "awa", "aws", "drive", "rtot", "lift", "drag", "side", "keel",
           "x", "y", "fill", "crew", "rate"]


def steady(grid, cols, tws, twa):
    C = {c: i for i, c in enumerate(cols)}
    r = next(g for g in grid if g[0] == tws and g[1] == twa)
    return {c: r[i] for c, i in C.items()}


def run_case(boat, data, tws, kind, turn_s):
    vm = next(v for v in data["vmg"] if v[0] == tws)
    entry = vm[1] if kind == "tack" else vm[3]
    s = steady(data["grid"], data["meta"]["columns"], tws, entry)
    R = Run(boat, tws, kind, turn_s, entry, s["flat"], s["reef"], bool(s["kite"]), lee0=s["leeway"])
    rows = R.simulate(s["v"] * KN, s["heel"])
    t = np.array([r[0] for r in rows]); y = np.array([r[14] for r in rows]); v = np.array([r[2] for r in rows])
    vmg = s["v"] * KN * np.cos(np.radians(entry))                 # m/s towards (tack) or away from (gybe) the wind
    y_ref = y[0] + vmg * (t - t[0])
    loss_m = (y_ref[-1] - y[-1]) if kind == "tack" else (y[-1] - y_ref[-1])
    i_min = int(np.argmin(v))
    after = np.where((t > R.t1) & (v >= 0.95 * s["v"]))[0]
    t95 = float(t[after[0]] - R.t0) if len(after) else None
    # where the loss is incurred: during the turn vs while accelerating
    k1 = int(np.searchsorted(t, R.t1))
    loss_turn = ((y_ref[k1] - y[k1]) if kind == "tack" else (y[k1] - y_ref[k1]))
    return dict(kind=kind, tws=tws, turn_s=turn_s, entry=entry, v0=s["v"], heel0=s["heel"], flat=s["flat"], reef=s["reef"],
                kite=bool(s["kite"]), t0=R.t0, t1=R.t1, cross=R.cross, vmin=float(v[i_min]), t_vmin=float(t[i_min] - R.t0),
                t95=t95, loss_m=float(loss_m), loss_bl=float(loss_m / LOA), loss_turn_bl=float(loss_turn / LOA),
                vmg=vmg / KN, rows=rows)


def main(data_path, out_path):
    data = json.loads(Path(data_path).read_text())
    boat = V.j109(gm=data["meta"]["gm"], lsm=data["meta"]["lsm"])
    cases = []
    for kind in ("tack", "gybe"):
        for tws in (6, 8, 10, 12, 14, 16, 18, 20):
            for turn in (5.0, 8.0, 12.0):
                c = run_case(boat, data, tws, kind, turn)
                print("%s %2d kn turn %4.1f s: entry %5.1f v0 %.2f vmin %.2f at %.1f s, 95%% at %s s, loss %.2f BL (turn %.2f)"
                      % (kind, tws, turn, c["entry"], c["v0"], c["vmin"], c["t_vmin"], c["t95"], c["loss_bl"], c["loss_turn_bl"]), flush=True)
                cases.append(c)
    Path(out_path).write_text(json.dumps(dict(columns=COLUMNS, loa=LOA, cases=cases), separators=(",", ":")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
