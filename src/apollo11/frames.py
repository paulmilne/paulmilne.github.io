"""Time and reference frames for the Apollo 11 reconstruction.

Working frame: the mean equator and equinox of the launch epoch ("MOD-1969").
It is inertial for our purposes (precession over the 8-day mission is about
0.002 deg), and it is the frame in which Apollo-era tables quote right
ascension via Greenwich Mean Sidereal Time.

Times are carried as GET, ground elapsed time in seconds from range zero,
1969-07-16 13:32:00 UTC.
"""
import numpy as np

# Range zero, 16 July 1969 13:32:00 UTC, as a Julian date (UTC).
JD_RANGE_ZERO_UTC = 2440419.06388889
# TT - UTC on that date (TAI-UTC = 7.5622 s in July 1969, plus 32.184 s).
TT_MINUS_UTC = 39.746
# UT1 - UTC on that date, from IERS EOP C04.
UT1_MINUS_UTC = 0.0115

# Fischer 1960 ("Mercury") ellipsoid, used for Apollo altitudes.
RE_EQ = 6378.166          # km
FLAT = 1.0 / 298.3
RE_POL = RE_EQ * (1 - FLAT)

OMEGA_EARTH = 7.2921158553e-5   # rad/s, sidereal rotation rate

FT = 0.3048e-3            # km per foot
NMI = 1.852               # km per nautical mile


def jd_ut1(get_s):
    return JD_RANGE_ZERO_UTC + (get_s + UT1_MINUS_UTC) / 86400.0


def jd_tt(get_s):
    return JD_RANGE_ZERO_UTC + (get_s + TT_MINUS_UTC) / 86400.0


def gmst_rad(get_s):
    """Greenwich mean sidereal time (IAU 1982 expression), radians."""
    jd = jd_ut1(np.asarray(get_s, dtype=float))
    T = (jd - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600 * 3600 + 8640184.812866) * T
           + 0.093104 * T**2 - 6.2e-6 * T**3)
    return np.mod(np.deg2rad(sec / 240.0), 2 * np.pi)


def ellipsoid_radius(geoc_lat_rad):
    """Distance from Earth's centre to the Fischer ellipsoid at a geocentric latitude."""
    c, s = np.cos(geoc_lat_rad), np.sin(geoc_lat_rad)
    return 1.0 / np.sqrt((c / RE_EQ) ** 2 + (s / RE_POL) ** 2)


def local_axes(r):
    """Unit up, east, north vectors at inertial position r (geocentric)."""
    up = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return up, east, north


def state_from_table(get_s, geoc_lat_deg, lon_deg, alt_km, v_kms, fpa_deg, head_deg,
                     radius_km=None):
    """Inertial state from the quantities printed in Apollo trajectory tables.

    Latitude is geocentric, longitude east-positive, altitude above the Fischer
    ellipsoid (unless radius_km gives the geocentric distance directly).
    Velocity is space-fixed: speed, flight-path angle above local horizontal,
    heading east of north.
    """
    lat = np.deg2rad(geoc_lat_deg)
    ra = np.deg2rad(lon_deg) + gmst_rad(get_s)
    r_mag = radius_km if radius_km is not None else ellipsoid_radius(lat) + alt_km
    r = r_mag * np.array([np.cos(lat) * np.cos(ra), np.cos(lat) * np.sin(ra), np.sin(lat)])
    up, east, north = local_axes(r)
    g, h = np.deg2rad(fpa_deg), np.deg2rad(head_deg)
    v = v_kms * (np.sin(g) * up + np.cos(g) * (np.cos(h) * north + np.sin(h) * east))
    return r, v


def table_from_state(get_s, r, v):
    """Inverse of state_from_table: the quantities a 1969 table would print."""
    r_mag = np.linalg.norm(r)
    lat = np.arcsin(r[2] / r_mag)
    ra = np.arctan2(r[1], r[0])
    lon = np.rad2deg(np.mod(ra - gmst_rad(get_s) + np.pi, 2 * np.pi) - np.pi)
    up, east, north = local_axes(r)
    v_mag = np.linalg.norm(v)
    fpa = np.rad2deg(np.arcsin(np.dot(v, up) / v_mag))
    head = np.rad2deg(np.mod(np.arctan2(np.dot(v, east), np.dot(v, north)), 2 * np.pi))
    alt = r_mag - ellipsoid_radius(lat)
    # Earth-relative speed: subtract the velocity of the rotating ground.
    v_rel = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
    return dict(lat=np.rad2deg(lat), lon=lon, alt=alt, r=r_mag, v=v_mag, fpa=fpa,
                head=head, v_rel=np.linalg.norm(v_rel))
