"""Orientation of the Moon, for selenographic coordinates and lunar gravity.

Uses the IAU WGCCRE (Archinal et al. 2011) model for the Moon's pole and
prime meridian, which approximates the Mean Earth/polar axis frame used for
lunar maps and the LRO landing-site coordinates to a few hundred metres.
The IAU model is referred to the ICRF (J2000 equator); it is rotated here
into the working frame of frames.py, the mean equator and equinox of the
launch date, with the IAU 1976 precession angles. numpy only.
"""
import numpy as np
import frames as F

R_MOON_REF = 1738.09    # km, the reference radius the Apollo tables measure lunar altitude from
D2R = np.pi / 180.0


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def precession_j2000_to_date(jd_tt):
    """IAU 1976 precession matrix: J2000 mean equator -> mean equator of date."""
    T = (jd_tt - 2451545.0) / 36525.0
    asec = D2R / 3600.0
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * asec
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * asec
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * asec
    return _rz(-z) @ _ry(theta) @ _rz(-zeta)


P0 = precession_j2000_to_date(F.jd_tt(0.0))    # fixed for the mission (precession over 8 days ~ 0.002 deg)


def iau_angles(get_s):
    """Pole right ascension, declination and prime meridian angle W (radians)."""
    d = F.jd_tt(get_s) - 2451545.0
    T = d / 36525.0
    E = np.deg2rad(np.array([
        125.045 - 0.0529921 * d, 250.089 - 0.1059842 * d, 260.008 + 13.0120009 * d,
        176.625 + 13.3407154 * d, 357.529 + 0.9856003 * d, 311.589 + 26.4057084 * d,
        134.963 + 13.0649930 * d, 276.617 + 0.3287146 * d, 34.226 + 1.7484877 * d,
        15.134 - 0.1589763 * d, 119.743 + 0.0036096 * d, 239.961 + 0.1643573 * d,
        25.053 + 12.9590088 * d]))
    s, c = np.sin(E), np.cos(E)
    a0 = (269.9949 + 0.0031 * T - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2] - 0.0172 * s[3]
          + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12])
    d0 = (66.5392 + 0.0130 * T + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2] + 0.0068 * c[3]
          - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9] - 0.0009 * c[12])
    W = (38.3213 + 13.17635815 * d - 1.4e-12 * d * d + 3.5610 * s[0] + 0.1208 * s[1] - 0.0642 * s[2]
         + 0.0158 * s[3] + 0.0252 * s[4] - 0.0066 * s[5] - 0.0047 * s[6] - 0.0046 * s[7]
         + 0.0028 * s[8] + 0.0052 * s[9] + 0.0040 * s[10] + 0.0019 * s[11] - 0.0044 * s[12])
    return np.deg2rad(a0), np.deg2rad(d0), np.deg2rad(W)


def body_to_work(get_s):
    """Rotation matrix taking Moon body-fixed (selenographic) vectors into the working frame."""
    a0, d0, W = iau_angles(get_s)
    # ICRF -> body: Rz(W) Rx(pi/2 - d0) Rz(pi/2 + a0); invert, then precess to date.
    M_icrf_to_body = _rz(W) @ _rx(np.pi / 2 - d0) @ _rz(np.pi / 2 + a0)
    return P0 @ M_icrf_to_body.T


def to_selenographic(get_s, r_rel_moon):
    """Working-frame vector from the Moon's centre -> (lat deg, east lon deg, radius km)."""
    b = body_to_work(get_s).T @ r_rel_moon
    rad = np.linalg.norm(b)
    return np.rad2deg(np.arcsin(b[2] / rad)), np.rad2deg(np.arctan2(b[1], b[0])), rad


def from_selenographic(get_s, lat_deg, lon_deg, radius_km):
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    b = radius_km * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return body_to_work(get_s) @ b
