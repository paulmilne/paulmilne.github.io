"""Write Moon and Sun geocentric states for the mission to ephem_1969.csv.

Needs skyfield and the JPL DE421 ephemeris (pip install skyfield skyfield-data).
The CSV it writes is committed, so the rest of the pipeline needs only numpy.

Positions are rotated from ICRF into the mean equator and equinox of the
launch epoch, the working frame of frames.py.
"""
from pathlib import Path
import numpy as np
from skyfield.api import load
from skyfield.framelib import ICRS_to_J2000
from skyfield_data import get_skyfield_data_path

import frames

HERE = Path(__file__).parent


def main(step_h=1.0, get_start_h=-2.0, get_end_h=200.0):
    ts = load.timescale(delta_t=frames.TT_MINUS_UTC - frames.UT1_MINUS_UTC)
    eph = load(str(Path(get_skyfield_data_path()) / "de421.bsp"))
    earth, moon, sun = eph["earth"], eph["moon"], eph["sun"]
    t0 = ts.tt_jd(frames.jd_tt(0.0))
    rot = t0.P @ ICRS_to_J2000            # ICRF -> mean of launch date

    gets = np.arange(get_start_h, get_end_h + 1e-9, step_h) * 3600.0
    t = ts.tt_jd(frames.jd_tt(gets))
    rows = []
    for body in (moon, sun):
        st = (body - earth).at(t)
        rows.append((rot @ st.position.km, rot @ st.velocity.km_per_s))
    with open(HERE / "ephem_1969.csv", "w") as f:
        f.write("get_s,moon_x,moon_y,moon_z,moon_vx,moon_vy,moon_vz,"
                "sun_x,sun_y,sun_z,sun_vx,sun_vy,sun_vz\n")
        for i, g in enumerate(gets):
            vals = [g]
            for p, v in rows:
                vals += list(p[:, i]) + list(v[:, i])
            f.write(",".join(f"{x:.6f}" if j == 0 or j % 6 in (1, 2, 3) else f"{x:.9f}"
                             for j, x in enumerate(vals)) + "\n")
    print("wrote", len(gets), "rows")


if __name__ == "__main__":
    main()
