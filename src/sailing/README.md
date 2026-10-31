# Sailing: a small velocity prediction program

Code behind the sailing posts in `mechanics/`. It solves the balance of forces on a
sailing keelboat (drive = resistance, sail side force = keel side force, heeling moment =
righting moment) at each true wind speed and angle, and picks the sail trim (flatten,
reef) that gives the most speed with at most 30 degrees of heel.

| File | Purpose |
|---|---|
| `vpp.py` | The model: apparent wind, sail forces (ORC coefficients), resistance (ORC residuary surfaces, friction), keel and rudder lift, righting moment, equilibrium solver, trim optimiser. numpy only. |
| `build_sailing.py` | Fits the two unknown numbers (GM and sailing length) to the J/109's ORC certificates by least squares, solves the full polar and writes `data/sailing.json`. Needs scipy for the fit. |
| `manoeuvre.py` | Part 2: tacks and gybes in the time domain (surge and roll dynamics, prescribed turn, luffing sails, centripetal keel load); writes `data/manoeuvres.json` from Part 1's `sailing.json`. |
| `assemble.py` | Fills the numbers in the post's text from `sailing.json` (`post_template.qmd` -> `index.qmd`). |
| `post_template.qmd` | The post with `%%NAME%%` placeholders. |
| `dat/` | Published coefficient tables (see below). |

## Data

- `dat/main.dat`, `jib.dat`, `asym_cl_kite.dat`, `asym_pole_kite.dat`, `kheff.dat`,
  `fcdmult.dat`, `rrk.dat`, `ORCi_Drag_Surfaces.csv`: ORC VPP documentation tables (sail lift
  and drag coefficients, effective rig height, drag multiplier, residuary resistance surfaces),
  as digitised in Python-VPP (github.com/marinlauber/Python-VPP, MIT licence).
  In `ORCi_Drag_Surfaces.csv` the rows are the length-volume ratio and the columns the
  beam-draft ratio; `vpp.py` reads it that way.
- The ORC certificates come from github.com/jieter/orc-data (MIT licence), file `ALL2024.json`.

## Rebuild

```bash
git clone --depth 1 https://github.com/jieter/orc-data /tmp/orc-data
cd src/sailing
python3 build_sailing.py /tmp/orc-data/ALL2024.json ../../mechanics/sailing-part-1-two-wings/data/sailing.json
python3 assemble.py post_template.qmd ../../mechanics/sailing-part-1-two-wings/data/sailing.json \
  ../../mechanics/sailing-part-1-two-wings/index.qmd
```

The build takes about 20 minutes on two cores (the fit solves the polar 20 or so times).
The fitted values printed should be GM about 1.05 m and sailing length about 10.09 m,
with an RMS difference from the certificates' median of about 0.15 knots.

`data/terms.json` (sailing terms and origins) was compiled by hand from the Online
Etymology Dictionary, Merriam-Webster and the other sources listed in each entry.

Part 2's manoeuvres (about 2 minutes):

```bash
python3 manoeuvre.py ../../mechanics/sailing-part-1-two-wings/data/sailing.json manoeuvres_full.json
```

The page uses a thinned copy (every 0.25 s around the turn, every second afterwards) at
`mechanics/sailing-part-2-tacking-gybing-tide/data/manoeuvres.json`.
