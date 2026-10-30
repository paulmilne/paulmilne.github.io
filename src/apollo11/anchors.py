"""Documented Apollo 11 values used as anchors for the reconstruction.

Every number here is copied from a 1969-70 NASA source and carries its
reference. Nothing in this file is computed. Units are as printed; the
helpers convert them.

Sources
-------
ABTN  Orloff & Garber, *Apollo by the Numbers: A Statistical Reference*,
      NASA SP-2000-4029 (2024 revision, Apollo Flight Journal edition),
      Apollo 11 ascent, earth-orbit, translunar and lunar-orbit tables.
BOEING  Boeing D5-15560-6, *Apollo/Saturn V Postflight Trajectory - AS-506*
      (Oct 1969), table B-VII, TLI state, as transcribed by Grant Hutchison
      (oikofuge.com, "Converting Apollo State Vectors to Orbits").
NSSDC  LM landing site, LRO-derived (Wagner et al. 2017, Icarus 283).
"""

# Range zero: 16 July 1969, 13:32:00 UTC (ABTN; Boeing section 1).

# Ascent-phase table (ABTN). Columns:
# event, GET s, altitude n mi, range n mi, earth-fixed vel ft/s,
# space-fixed vel ft/s, geocentric lat deg N, longitude deg E,
# space-fixed flight-path angle deg, space-fixed heading deg E of N.
ASCENT = [
    ("Liftoff",                     0.63,    0.032,    0.000,     1.5,  1340.7, 28.4470, -80.6041,  0.06,  90.00),
    ("Mach 1",                     66.3,     4.236,    1.044,  1054.1,  2023.9, 28.4523, -80.5853, 27.88,  85.32),
    ("Maximum dynamic pressure",   83.0,     7.326,    3.012,  1653.4,  2671.9, 28.4624, -80.5499, 29.23,  82.41),
    # Longitude printed as -81.1517 in the scanned table; with 25 n mi of
    # downrange travel to the east it must be -80.1517 (scan digit error).
    ("S-IC centre engine cutoff", 135.2,    23.761,   25.067,  5320.8,  6492.8, 28.5739, -80.1517, 22.957, 76.315),
    ("S-IC outboard engine cutoff", 161.63, 35.701,   50.529,  7851.9,  9068.6, 28.7007, -79.6908, 19.114, 75.439),
    ("S-IC/S-II separation",      162.30,   36.029,   51.323,  7882.9,  9100.6, 28.7046, -79.6764, 19.020, 75.436),
    ("S-II centre engine cutoff", 460.62,   97.280,  601.678, 17404.8, 18725.5, 30.9513, -69.4309,  0.897, 79.646),
    ("S-II outboard engine cutoff", 548.22, 101.142, 873.886, 21368.2, 22690.8, 31.7089, -64.1983,  0.619, 82.396),
    ("S-II/S-IVB separation",     549.00,  101.175,  876.550, 21377.0, 22699.6, 31.7152, -64.1467,  0.611, 82.426),
    ("S-IVB first cutoff",        699.33,  103.202, 1421.959, 24237.6, 25561.6, 32.4865, -53.4588,  0.011, 88.414),
    ("Earth orbit insertion",     709.33,  103.176, 1460.697, 24243.9, 25567.9, 32.5027, -52.6491,  0.012, 88.848),
]

# Earth parking orbit (ABTN): apogee and perigee in n mi above a spherical
# Earth of equatorial radius; period minutes; inclination deg.
PARKING_ORBIT = dict(apogee_nmi=100.4, perigee_nmi=98.9, period_min=88.18, incl_deg=32.521)

# S-IVB second burn (TLI) timing (ABTN).
TLI_IGNITION_GET = 2 * 3600 + 44 * 60 + 16.20      # 002:44:16.20
TLI_IGNITION_SFV_FTS = 25560.2                      # space-fixed velocity at ignition
TLI_CUTOFF_GET = 2 * 3600 + 50 * 60 + 3.03          # 002:50:03.03
TLI_BURN_S = 346.83

# Translunar injection, defined as cutoff + 10 s (ABTN, BOEING).
TLI = dict(
    get=2 * 3600 + 50 * 60 + 13.03,     # 002:50:13.03 (range time 10213.03 s)
    radius_km=6711.964,                 # BOEING geocentric distance
    lat=9.9204, lon=-164.8373,          # BOEING geocentric lat, longitude E
    alt_nmi=180.581,                    # ABTN
    v_fts=35545.6, v_kms=10.8343,       # ABTN, BOEING
    fpa=7.367, head=60.073,             # ABTN
    incl=31.386,                        # ABTN
)

# Translunar coast events (ABTN).
CSM_SEPARATION_GET = 3 * 3600 + 15 * 60 + 23.0     # 003:15:23.0
DOCKING_GET = 3 * 3600 + 24 * 60 + 3.7             # 003:24:03.7
MCC2_GET = 26 * 3600 + 44 * 60 + 58.64             # 026:44:58.64, 3.13 s SPS burn
LOI1_IGNITION_GET = 75 * 3600 + 49 * 60 + 50.37    # 075:49:50.37
LOI1_IGNITION_ALT_NMI = 86.7                        # altitude above the Moon
LOI1_IGNITION_V_FTS = 8250.0

# Lunar landing site (NSSDC, LRO): planetocentric lat, lon (deg).
LM_SITE = (0.67416, 23.47314)


# ---------------------------------------------------------------------------
# Part 2: translunar coast and lunar orbit.
# ABTN translunar and lunar-orbit tables; maneuver PADs and PAO commentary
# from the Apollo 11 Flight Journal (AFJ); Mission Director's post-launch
# report (MDR, NASA M-932-69-11).
# Lunar altitudes are above a 1738.09 km reference sphere, as in the tables.
# LVLH components are the PAD "Noun 81" values in ft/s (X fwd, Y -h, Z down).

def _get(h, m, s):
    return h * 3600 + m * 60 + s

# Documented coast states (ABTN): (event, GET, altitude n mi, space-fixed vel ft/s, fpa, heading).
# The docking row's flight-path angle prints as 44.94; every neighbouring quantity
# fits 49.94, so the printed value is treated as a typesetting slip.
COAST_STATES = [
    ("CSM separated from S-IVB", _get(3, 15, 23.0), 3815.190, 24962.5, 45.148, 93.758),
    ("CSM docked with LM/S-IVB", _get(3, 24, 3.7), 5317.6, 22662.5, 49.94, 99.57),
    ("Evasive maneuver ignition", _get(4, 40, 1.72), 16620.8, 14680.0, 64.30, 113.73),
    ("Evasive maneuver cutoff", _get(4, 40, 4.65), 16627.3, 14663.0, 64.25, 113.74),
    ("Midcourse correction 2 ignition", _get(26, 44, 58.64), 109475.3, 5025.0, 77.05, 120.88),
    ("Midcourse correction 2 cutoff", _get(26, 45, 1.77), 109477.2, 5010.0, 76.88, 120.87),
]

# Burns. PAD vector (planned), actual ignition, duration and delta-v (ABTN).
EVASIVE = dict(tig=_get(4, 40, 1.72), dur=2.93, dv_fps=19.7, pad=(5.1, 0.0, 19.0), body="earth")
MCC2 = dict(tig=_get(26, 44, 58.64), dur=3.13, dv_fps=20.9, pad=(11.8, -0.3, 17.7), pad_dv=21.3, body="earth")
LOI1 = dict(tig=_get(75, 49, 50.37), tco=_get(75, 55, 47.90), dur=357.53, dv_fps=2917.5,
            pad=(-2889.7, -394.4, -68.6), pad_dv=2917.3,
            alt_ign=86.7, v_ign=8250.0, alt_co=60.1, v_co=5479.0, apo=169.7, peri=60.0,
            mass_lb=96012.0, body="moon")
LOI2 = dict(tig=_get(80, 11, 36.75), tco=_get(80, 11, 53.63), dur=16.88, dv_fps=158.8,
            pad=(-140.8, 0.0, -74.3), pad_dv=159.2,
            alt_ign=61.8, v_ign=5477.3, alt_co=61.6, v_co=5338.3, apo=66.1, peri=54.5, body="moon")
UNDOCK = dict(get=_get(100, 12, 0), alt=62.9, v=5333.8)
SEP_ORBIT = dict(get=_get(100, 40, 1.9), apo=63.7, peri=56.0)   # after the 2.7 ft/s separation burn

# MDR: entry into the lunar sphere of influence.
SOI = dict(get=_get(61, 39, 55), earth_dist_nmi=186436, moon_dist_nmi=33823,
           v_earth_fps=2990, v_moon_fps=3775)

# AFJ: radio loss/acquisition behind the Moon (PADs and PAO).
LOS_AOS = [
    ("LOS before LOI-1", "LOS", _get(75, 41, 23)),
    ("AOS after LOI-1 (predicted with burn)", "AOS", _get(76, 15, 29)),
    ("LOS, end of rev 1 ('a little over 2 minutes' at 77:38:16)", "LOS", _get(77, 40, 30)),
    ("AOS, rev 2 (predicted)", "AOS", _get(78, 23, 31)),
    ("LOS, end of rev 2 (two minutes after 79:46:23)", "LOS", _get(79, 48, 23)),
    ("LOS, end of rev 3 (PAO)", "LOS", _get(81, 45, 0)),
    ("LOS, end of rev 4 (PAO, 'less than a minute' at 83:43)", "LOS", _get(83, 44, 0)),
    ("LOS, end of rev 5 (PAO)", "LOS", _get(85, 42, 0)),
    ("AOS, rev 8 (PAO prediction)", "AOS", _get(90, 25, 0)),
    ("AOS, rev 10 (PAO)", "AOS", _get(94, 21, 0)),
    ("AOS, rev 12 (PAO prediction)", "AOS", _get(98, 18, 0)),
]
AOS_NO_BURN = _get(76, 5, 30)        # LOI-1 PAD: AOS if the burn had not happened
LOI1_PERIOD = _get(2, 8, 37)          # AFJ 77:35:54, orbital period after LOI-1

# Orbits read out by PAO in lunar orbit: (GET, apolune n mi, perilune n mi).
PAO_ORBITS = [(_get(78, 24, 11), 170.2, 61.3), (_get(79, 9, 0), 170.2, 61.2),
              (_get(81, 40, 0), 65.4, 53.9), (_get(83, 34, 0), 65.1, 54.2)]

# Landing site (NSSDC/LRO) and time, for the plane of the lunar orbit.
LANDING_GET = _get(102, 45, 40)

# Television from the spacecraft (MDR; AFJ).
TV = [(_get(10, 32, 0), 16 * 60, "Unscheduled TV, recorded at Goldstone"),
      (_get(30, 28, 0), 50 * 60, "Unscheduled TV"),
      (_get(33, 59, 0), 36 * 60, "Scheduled TV"),
      (_get(55, 8, 0), 96 * 60, "TV inside Eagle"),
      (_get(78, 24, 11), 35 * 60, "TV from lunar orbit")]


# ---------------------------------------------------------------------------
# Part 3: descent and landing.
# ABTN lunar-orbit table; TRW Note 70-FMT-819 (Apollo 11 trajectory
# reconstruction, 1970); Apollo 11 Mission Report MSC-00171 (MR);
# F. V. Bennett, "Apollo Lunar Descent and Ascent Trajectories", NASA TM X-58040
# (1970) (BENNETT); A. R. Klumpp, "Apollo Lunar-Descent Guidance", MIT/Draper
# R-695 (1971) (KLUMPP); Luminary 99 pad-loaded erasables (Apollo 11 LM
# software) (PADLOAD).

LM_UNDOCK = dict(get=_get(100, 12, 0), alt=62.9, v=5333.8)                 # ABTN (as UNDOCK)
CSM_SEP = dict(tig=_get(100, 39, 52.9), tco=_get(100, 40, 1.9), dur=9.0, dv_fps=2.7,
               alt_ign=62.7, v_ign=5332.7, alt_co=62.5, v_co=5332.2, apo=63.7, peri=56.0,
               direction="radially down")                                  # ABTN; MR: radial
DOI = dict(tig=_get(101, 36, 14.0), tco=_get(101, 36, 44.0), dur=30.0, dv_fps=76.4,
           alt_ign=56.4, v_ign=5364.9, alt_co=57.8, v_co=5284.9,           # ABTN
           orbit_onboard=(57.2, 9.1), orbit_predicted=(57.2, 8.5))         # MR 4.10.1 (apolune, perilune)
PDI = dict(tig=_get(102, 33, 5.01), alt=6.4, v=5564.9,                     # ABTN
           tco=_get(102, 45, 41.40), dur=756.39, dv_fps=6775.6,            # ABTN; TRW gives 6775.8
           alt_est_ft=52000)                                               # MR: from landmark sightings
TOUCHDOWN_GET = _get(102, 45, 39.9)                                        # TRW, MR (contact light)
THROTTLE_HOLD_S = 26.0                                                     # MR, BENNETT: minimum thrust for 26 s
THROTTLE_DOWN_S = 386.0                                                    # MR: throttle recovery 386 s after ignition (planned 384)
P64 = dict(get=_get(102, 41, 32), alt_ft=7129.0, hdot_fps=-125.0)          # MR 5.3; time AFJ/ALSJ
P66 = dict(get=_get(102, 43, 22), alt_ft=410.0, fwd_fps=50.0)              # BENNETT (alt, speed); AFJ (time)
LONG_FT = 1100.0        # MR, BENNETT: landing downrange of the point the guidance was flying to
CROSS_FT = 400.0        # BENNETT: crossrange from that point
PDI_STATE_ERROR_FT = 21000.0   # MR 5.2: onboard position estimate uprange of the true position at PDI
SITE_ALT_FT = -8557.0   # TRW: landing site altitude relative to the 1738.09 km reference sphere

# Descent engine and guidance constants.
DPS_RATED_N = 46706.0          # KLUMPP: rated thrust (10,500 lbf)
DPS_FTP = 0.925                # BENNETT: fixed throttle point 92.5% of rated
DPS_MIN = 0.10                 # 10% (trim) throttle
DPS_THROTTLE_MAX = 0.65        # KLUMPP: permitted throttle region 11-65%
DPS_THROTTLE_MIN = 0.11
THROTTLE_RECOVERY = 0.57       # KLUMPP, BENNETT: maximum thrust while guidance demands > ~57%
DPS_VE = 2.97                  # km/s; Isp 303 s at full thrust (assumed)
LM_MASS_DOI_KG = 15269.7       # PADLOAD LEMMASS, 33,664 lb at DOI ignition
DPS_USABLE_LB = 17934.6        # BENNETT Table II: DPS propellant available for delta-v
LEADTIME = 2.2                 # PADLOAD/KLUMPP, s
# Approach-phase (P64) targets, guidance frame (x up, y cross, z downrange), feet (PADLOAD RAPFG etc.).
P64_TARGETS = dict(r=(111.085, 0.0, -26.794), v=(-4.993, 0.0, 0.248), a=(-0.2624, 0.0, -0.5120), jz=0.0018077200)
# Braking-phase (P63) targets (PADLOAD RBRFG etc.); the downrange velocity target is
# illegible in the scan and is solved for.
P63_TARGETS = dict(r=(171.835, 0.0, -10678.596), vx=-105.876, a=(0.6241, 0.0, -9.1044), jz=-0.01882677)

# Program alarms (air-to-ground transcript / AFJ).
ALARMS = [(_get(102, 38, 22), "1202"), (_get(102, 39, 2), "1202"), (_get(102, 42, 18), "1201"),
          (_get(102, 42, 43), "1202"), (_get(102, 42, 58), "1202")]
BINGO_CALLS = [(_get(102, 45, 2), "60 seconds"), (_get(102, 45, 31), "30 seconds")]
YAW = (_get(102, 36, 57), _get(102, 37, 59))    # windows-up yaw manoeuvre (AFJ)

# Aldrin's (and Armstrong's) callouts, from the air-to-ground and onboard transcripts:
# (GET s, altitude ft or None, vertical rate ft/s (negative down) or None, forward ft/s or None).
CALLOUTS = [
    (_get(102, 42, 17), 3000, None, None), (_get(102, 42, 29), 2000, None, None),
    (_get(102, 43, 1), 750, -23, None), (_get(102, 43, 7), 700, -21, None),
    (_get(102, 43, 11), 600, -19, None), (_get(102, 43, 15), 540, -15, None),
    (_get(102, 43, 26), 400, -9, 58), (_get(102, 43, 32), 350, -4, None),
    (_get(102, 43, 35), 330, -6.5, None), (_get(102, 43, 46), 300, -3.5, 47),
    (_get(102, 43, 52), None, -1.5, None), (_get(102, 43, 57), 270, None, None),
    (_get(102, 44, 7), 250, -2.5, 19), (_get(102, 44, 16), 220, -3.5, 13),
    (_get(102, 44, 23), None, None, 11), (_get(102, 44, 24), 200, -4.5, None),
    (_get(102, 44, 26), None, -5.5, None), (_get(102, 44, 31), 160, -6.5, None),
    (_get(102, 44, 33), None, -5.5, 9), (_get(102, 44, 40), 120, None, None),
    (_get(102, 44, 45), 100, -3.5, 9), (_get(102, 44, 54), 75, -0.5, 6),
    (_get(102, 45, 0), 60, -2.5, 2), (_get(102, 45, 17), 40, -2.5, None),
    (_get(102, 45, 21), 30, -2.5, None), (_get(102, 45, 25), None, None, 4),
    (_get(102, 45, 26), 20, -0.5, None),
]
# Landing point designator angles read out by Aldrin (air-to-ground transcript): (GET, degrees).
LPD_CALLS = [(_get(102, 42, 31), 47.0), (_get(102, 42, 58), 35.0), (_get(102, 43, 7), 33.0)]
ATT_HOLD_GET = _get(102, 41, 53)   # Armstrong to attitude hold in P64 (AFJ/ALSJ)
# Public Affairs Officer commentary during the braking phase, from NASA's Day 5 audio reel
# (times from the reel's alignment with the transcript, +/- 2 s): (GET, quantity, value).
PAO_DESCENT = [(369550, "altitude_ft", 27000), (369597, "altitude_ft", 21000), (369606, "velocity_fps", 1200)]



# ---------------------------------------------------------------------------
# Part 4: lunar ascent, rendezvous, transearth injection, coast and entry.
# ABTN lunar-orbit and transearth tables; MR; BENNETT; the crew's PAD readbacks
# and callouts from the air-to-ground and onboard transcripts (TEC); the
# Apollo 11 entry postflight analysis, MSC Internal Note (Feb 1970) (ENTRY).
# The ascent and rendezvous rows of ABTN give heights above the landing-site
# radius, not the 1738.09 km sphere: their speeds and orbits agree to a fraction
# of a foot per second only on that reading (build_part4.py tests it).

LIFTOFF_GET = _get(124, 22, 0.79)                                           # ABTN
INSERTION = dict(get=_get(124, 29, 15.67), alt=10.0, v=5537.9, dur=434.88,  # ABTN
                 dv_fps=6070.1, apo=48.0, peri=9.4,                         # ABTN
                 dsky_v=5537.3, dsky_hdot=32.8, dsky_alt_ft=60666.0,        # TEC 124:29:21 (read as "53373")
                 onboard_orbit=(47.3, 9.5), ags_orbit=(46.6, 9.5),          # TEC 124:31:00, MR 4.15
                 ydot_fps=-1.0)                                             # TEC 124:58:36: CSM's estimate of Eagle's out-of-plane rate
APS_THRUST_N = 15569.0          # BENNETT: ascent engine 3,500 lbf
APS_VERTICAL_S = 10.0           # BENNETT: vertical rise before pitchover
APS_PROPELLANT_LB = 4934.0      # BENNETT: ascent propellant loaded (approximate)
# Aldrin's ascent callouts (TEC): (GET, horizontal ft/s, vertical ft/s, altitude ft).
ASCENT_CALLS = [(_get(124, 22, 6), None, 36.0, None),
                (_get(124, 23, 36), 700.0, 150.0, 9000.0),
                (_get(124, 24, 5), 1000.0, 170.0, 14000.0),
                (_get(124, 24, 32), 1500.0, 185.0, None)]
# Velocity-to-go callouts near the end of the ascent burn (TEC): (GET, ft/s).
ASCENT_VGO = [(_get(124, 28, 33), 800.0), (_get(124, 28, 40), 700.0), (_get(124, 28, 56), 350.0)]
DELTA_H_AGS = 15.5             # TEC 124:39:24: AGS height difference for the coelliptic sequence, n mi

# Rendezvous burns (ABTN), with the crew's Noun 81 readbacks (TEC) where given:
# components ft/s in the LM's LVLH frame (X forward, Y -h, Z down).
CSI = dict(tig=_get(125, 19, 35.0), tco=_get(125, 20, 22.0), dur=47.0, dv_fps=51.5,
           alt_ign=47.4, v_ign=5328.1, alt_co=48.4, v_co=5376.6, apo=49.3, peri=45.7,
           pad_tig=_get(125, 19, 34.70))                                    # TEC 125:59:24
CDH = dict(tig=_get(126, 17, 49.6), tco=_get(126, 18, 7.4), dur=17.8, dv_fps=19.9,
           apo=47.4, peri=42.1, noun81=(-8.1, -1.8, -18.2))                 # TEC 126:14:05
TPI = dict(tig=_get(127, 3, 51.8), tco=_get(127, 4, 14.5), dur=22.7, dv_fps=25.3,
           alt_ign=44.1, v_ign=5391.5, alt_co=44.0, v_co=5413.2, apo=61.7, peri=43.7,
           pad_tig=_get(127, 3, 30.82), noun81=(22.7, 1.7, -10.6))          # TEC 126:58:23
MCC_R = [dict(tig=_get(127, 18, 30.8), dv_fps=1.0, noun81=(0.0, 0.4, 0.9)),  # ABTN; TEC 127:16:37
         dict(tig=_get(127, 33, 30.8), dv_fps=1.5, noun81=(0.1, 1.2, 0.5))]  # ABTN; TEC 127:34:54
BRAKING_GET = _get(127, 36, 57.3)                                           # ABTN
TPF = dict(tig=_get(127, 46, 9.8), tco=_get(127, 46, 38.2), dur=28.4, dv_fps=31.4,
           apo=63.0, peri=56.5)                                             # ABTN
STATIONKEEP_GET = _get(127, 52, 5.3)                                        # ABTN
DOCK = dict(get=_get(128, 3, 0.0), alt=60.6, v=5341.5)                     # ABTN
# Range and range-rate callouts during braking and docking (TEC): (GET, range ft, closing ft/s).
RANGE_CALLS = [(_get(127, 41, 21), 1.5 * 6076.1, 32.0), (_get(127, 43, 43), 0.7 * 6076.1, 31.0),
               (_get(127, 46, 13), None, 11.0), (_get(127, 58, 42), 15.0, None), (_get(127, 59, 14), 7.5, None)]
JETTISON = dict(get=_get(130, 9, 31.2), alt=61.6, v=5335.9)                # ABTN
FINAL_SEP = dict(tig=_get(130, 30, 1.0), tco=_get(130, 30, 8.1), dur=7.2, dv_fps=2.2,
                 alt_ign=62.7, v_ign=5330.1, alt_co=62.7, v_co=5326.9, apo=62.7, peri=54.0,
                 pad=(2.0, 0.0, 0.0))                                       # ABTN; PAD (AFJ)

# Transearth injection (ABTN; PAD from AFJ; crew report TEC).
TEI = dict(tig=_get(135, 23, 42.28), tco=_get(135, 26, 13.69), dur=151.41, dv_fps=3279.0,
           alt_ign=52.4, v_ign=5376.0, fpa_ign=-0.03,
           alt_co=58.1, v_co=8589.0, fpa_co=5.13,
           pad=(3201.1, 681.8, -265.0))
# MCC-5 (ABTN; PAD read up at 148:52, TEC): the only transearth correction flown.
MCC5 = dict(tig=_get(150, 29, 57.4), dur=10.0, dv_fps=4.8, pad=(-4.8, 0.0, 0.1),
            pad_tig=_get(150, 29, 54.53),
            alt_ign=169087.2, v_ign=4075.0, fpa_ign=-80.34,
            alt_co=169080.6, v_co=4074.0, fpa_co=-80.41)
CM_SM_SEP = dict(get=_get(194, 49, 12.7), alt=1778.3, v=29615.5, fpa=-35.26)   # ABTN

# Entry (ENTRY, best estimate of trajectory; longitude from the BET and the onboard
# initialisation, which agree; the best-estimate column prints 171.1960).
EI = dict(get=_get(195, 3, 5.66), v=36194.368, fpa=-6.5230, az=50.1761,
          lon=171.9603, geod_lat=-3.1933, alt_ft=400000.72)
ENTRY_G = dict(g1=6.73, v1=31810.0, t1=82.0, gmin=0.48, vmin=20500.0, g2=6.00, v2=12390.0, t2=374.0,
               hmin_ft=182109.0, t_hmin=88.0, t_005g=30.0, h_005g=289926.0,
               t_final=200.0, v_final=21301.0)                              # ENTRY
CM_LD = 0.282                   # ENTRY: hypersonic lift-to-drag ratio (reconstruction)
CM_MASS_LB = 12148.0            # ENTRY: CM weight used by the onboard computer
ENTRY_TARGET = (13.32, -169.17)                                             # ENTRY, entry PAD
SPLASH = dict(get=_get(195, 18, 35), lat=13.30, lon=-169.15,               # MR (time); ENTRY (onboard)
              mr_lat=13 + 19 / 60, mr_lon=-(169 + 9 / 60))                 # MR: 13 deg 19'N, 169 deg 09'W
DROGUE_GET = _get(195, 12, 6.9)                                             # ABTN; ENTRY gives E1 + 9:13
MAIN_DELAY_S = 50.0                                                         # ENTRY: mains about 50 s after drogues
# Altitude calls by Armstrong under the main parachutes (TEC times, which run ~17 s
# behind the splashdown time in MR): (GET, ft).
CHUTE_CALLS = [(_get(195, 15, 52), 3750.0), (_get(195, 16, 48), 2500.0),
               (_get(195, 17, 24), 1500.0), (_get(195, 17, 54), 300.0)]
# Public Affairs Officer, Day 6 reel: "about 6 and 1/2 minutes to ignition on the terminal phase
# maneuver ... distance now between the two spacecraft some 38.6 nautical miles closing at a rate
# of 110 feet per second."
PAO_TPI = dict(get=TPI["tig"] - 375.0, range_nmi=38.6, rdot_fps=-110.0)
CHUTE_CALL_SHIFT_S = 17.0       # TEC splashdown 195:18:18 against MR 195:18:35
