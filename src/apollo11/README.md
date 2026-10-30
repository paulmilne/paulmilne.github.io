# Apollo 11 reconstruction

Code behind the Apollo 11 series in `space/`. It rebuilds the flight from
documented 1969-70 state vectors (`anchors.py`), filling the gaps by
interpolation, numerical propagation and fitted burns.

| File | Purpose |
|---|---|
| `anchors.py` | Documented values, each with its source. Nothing computed. |
| `frames.py` | Time scales, sidereal time, Fischer ellipsoid, table <-> state conversion. |
| `dynamics.py` | Earth (J2) + Moon + Sun force model, the Moon's J2 and C22 within 20,000 km of the Moon, and a Dormand-Prince 5(4) integrator. numpy only. |
| `moon_frame.py` | IAU 2009 lunar orientation, precession to the 1969 frame, selenographic coordinates. |
| `burns.py` | Impulsive and finite (constant-thrust) burns given in the Apollo LVLH frame (Noun 81). |
| `lunar.py` | Moon-relative elements, LOS/AOS by occultation, landing-site plane test, Levenberg-Marquardt solver. |
| `pao_callouts.py` | Distances and speeds announced by the Public Affairs Officer, read from the transcript (checks only). |
| `tli_burn.py` | Constant-thrust burn model and Gauss-Newton fit to the TLI anchor. |
| `ephem_1969.csv` | Moon and Sun geocentric states, hourly, from JPL DE421 (committed). |
| `make_ephemeris.py` | Regenerates `ephem_1969.csv`; needs `skyfield` and `skyfield-data`. |
| `build_part1.py` | Builds `space/apollo-11-part-1-launch-to-tli/data/part1.json`. Uses `build_part2.coast_to_mcc` for the free-return path. |
| `descent.py` | The LM's descent guidance (Klumpp's quadratic law, Luminary 99 targets) and engine/throttle model. |
| `build_part3.py` | Builds `space/apollo-11-part-3-descent-and-landing/data/part3.json`: PDI fit, P63/P64 simulation, P66 from the callouts, DOI and undocking traced back. Needs Part 2's `part2.json`. |
| `part4_model.py` | Part 4 models: lunar ascent (linear-tangent steering), RCS rendezvous burns, relative motion, transearth injection, entry through the US Standard Atmosphere 1976 with a bank-angle profile, parachutes. |
| `build_part4.py` | Builds `space/apollo-11-part-4-rendezvous-return-and-entry/data/part4.json`: Columbia's orbit fitted to signal times, ascent, CSI/CDH/TPI, braking and docking, TEI fitted to entry interface, MCC-5, entry and splashdown. Needs Part 3's `part3.json`. |
| `build_part2.py` | Builds `space/apollo-11-part-2-translunar-coast/data/part2.json`: coast, MCC-2 fit, LOI-1 and LOI-2 fits, orbit to 100 h. |
| `transcripts.py` | Extracts the NASA air-to-ground and onboard voice transcripts (via Spacelog) into `data/transcript.json`. |

Rebuild Part 1 (about 30 seconds, needs only numpy):

```bash
cd src/apollo11
python3 build_part1.py ../../space/apollo-11-part-1-launch-to-tli/data/part1.json
```

Rebuild Part 2 (about 3 minutes):

```bash
python3 build_part2.py ../../space/apollo-11-part-2-translunar-coast/data/part2.json
```

Rebuild Part 3 (about a minute):

```bash
python3 build_part3.py ../../space/apollo-11-part-3-descent-and-landing/data/part3.json \
  ../../space/apollo-11-part-2-translunar-coast/data/part2.json
```

Rebuild Part 4 (about 6 minutes):

```bash
python3 build_part4.py ../../space/apollo-11-part-4-rendezvous-return-and-entry/data/part4.json \
  ../../space/apollo-11-part-3-descent-and-landing/data/part3.json
```

Each script prints a table of documented values against the reconstruction.

Rebuild the transcript (needs a clone of github.com/Spacelog/spacelog):

```bash
python3 transcripts.py /path/to/spacelog/missions/a11/transcripts \
  ../../space/apollo-11-part-1-launch-to-tli/data/transcript.json -3600 21600
python3 transcripts.py /path/to/spacelog/missions/a11/transcripts \
  ../../space/apollo-11-part-2-translunar-coast/data/transcript.json 10213 360000
python3 transcripts.py /path/to/spacelog/missions/a11/transcripts \
  ../../space/apollo-11-part-3-descent-and-landing/data/transcript.json 356400 374400
python3 transcripts.py /path/to/spacelog/missions/a11/transcripts \
  ../../space/apollo-11-part-4-rendezvous-return-and-entry/data/transcript.json 444000 706000
```

`data/media.json` maps NASA's Day 1 audio highlights reel and a NASA launch film clip
(both on archive.org) to mission time. The audio segments were found by matching the reel's
speech-recognition captions against the air-to-ground transcript; the film is pinned by the
first visible engine flame to the S-IC ignition sequence (T-8.6 s). The page streams both
from archive.org; nothing is copied into the site.

Part 2's `data/media.json` does the same for NASA's Day 2, 3 and 4 audio reels. Each segment
needs at least two caption lines matched to the transcript with a consistent offset (within 8 s);
segments are padded by 2 s before and 7 s after. No synced video is used in Part 2.

Part 3's `data/media.json` maps NASA's Day 5 audio reel the same way, and the NASA JSC transfer of the
16 mm onboard landing film. The film was shot at 6 frames per second and transferred at 30, so it is
mapped with `rate: 5`; it is pinned at touchdown (the LM's shadow stops moving at clip 190.6 s).

Part 4's `data/media.json` maps NASA's Day 6, 7 and 8 audio reels the same way (Day 6 has hand-made captions with coarser
timing; the Day 9 reel has no captions and is not used), plus two 16 mm onboard films: the rendezvous seen from Columbia
and the docking and entry. The films are placed approximately: the docking clip so its closing ends at the reconstruction's
capture, the entry clip so the glow begins at 0.05 g.
