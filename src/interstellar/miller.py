"""Time dilation on Miller's planet: a circular equatorial prograde orbit around a Kerr
black hole (G = c = M = 1). dt/dtau for an orbit at Boyer-Lindquist radius r is
  u^t = (r^1.5 + a) / ( r^0.75 * sqrt(r^1.5 - 3 r^0.5 + 2a) )   (Bardeen, Press & Teukolsky 1972).
Near-extremal spins need ~40 significant digits, so mpmath throughout."""
import json, mpmath as mp
mp.mp.dps = 60
YEAR_H = mp.mpf('8766')                      # hours in a Julian year
TARGET = 7 * YEAR_H                          # 1 hour on the planet = 7 years on Earth

def ut(r, a): return (r**1.5 + a) / (r**0.75 * mp.sqrt(r**1.5 - 3*r**0.5 + 2*a))
def rhor(a): return 1 + mp.sqrt(1 - a*a)
def risco(a):
    z1 = 1 + mp.cbrt(1 - a*a) * (mp.cbrt(1 + a) + mp.cbrt(1 - a)); z2 = mp.sqrt(3*a*a + z1*z1)
    return 3 + z2 - mp.sqrt((3 - z1) * (3 + z1 + 2*z2))
def rphoton(a): return 2 * (1 + mp.cos(mp.mpf(2)/3 * mp.acos(-a)))

if __name__ == '__main__':
    out = {"target_dilation": float(TARGET), "rows": []}
    for k in [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        eps = mp.mpf(10)**(-k); a = 1 - eps
        ri = risco(a); u_isco = ut(ri, a)
        row = dict(log10_eps=-k, r_horizon=float(rhor(a)), r_isco=float(ri), ut_isco=float(u_isco))
        if u_isco > TARGET:      # radius of the orbit with exactly the film's dilation (outside the ISCO)
            r7 = mp.findroot(lambda r: ut(r, a) - TARGET, (ri * (1 + mp.mpf('1e-30')), 3), solver='anderson')
            row["r_miller"] = float(r7)
        out["rows"].append(row)
        print(row)

    # The book's case: a = 1 - 1e-14, M = 1e8 Msun
    a = 1 - mp.mpf('1e-14'); M_sun_s = mp.mpf('4.925490947e-6'); M = mp.mpf('1e8')
    ri = risco(a); r7 = mp.findroot(lambda r: ut(r, a) - TARGET, (ri*(1+mp.mpf('1e-30')), 3), solver='anderson')
    # orbital period seen from far away: T = 2 pi (r^1.5 + a) M
    T_far = 2*mp.pi*(r7**1.5 + a) * M * M_sun_s
    out["book"] = dict(a="1 - 1e-14", r_isco=float(ri), ut_isco=float(ut(ri, a)), r_miller=float(r7),
                       r_horizon=float(rhor(a)), period_far_hours=float(T_far/3600),
                       period_planet_seconds=float(T_far/ut(r7, a)),
                       gravitational_radius_km=float(M * mp.mpf('1.476625'))*1.0)
    print(out["book"])
    # smallest spin that allows the film's dilation at all (at the ISCO)
    f = lambda le: ut(risco(1 - mp.power(10, le)), 1 - mp.power(10, le)) - TARGET
    le = mp.findroot(f, -12); out["min_eps_for_film"] = float(mp.power(10, le)); print("min 1-a:", out["min_eps_for_film"])
    json.dump(out, open("miller.json", "w"), indent=1)

