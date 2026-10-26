# Gargantua

Ray-traced images of a spinning (Kerr) black hole, its thin accretion disk and
the star field behind it, following the equations of James, von Tunzelmann,
Franklin and Thorne (2015), *Gravitational lensing by spinning black holes in
astrophysics, and in the movie Interstellar*, Class. Quantum Grav. 32, 065001.

Behind the blog post `sci-fi/gargantua/` and four explainers:
`explainers/light-bending-schwarzschild`, `explainers/black-hole-ray-tracing`,
`explainers/kerr-black-hole` and `explainers/accretion-disk-light`.

## Setup

Uses the blog's virtual environment (numpy, scipy, numba, matplotlib):

```bash
cd ~/Dev/Quarto/math-blog
.venv/bin/pip install -r gargantua/requirements.txt
```

## Video

```bash
../.venv/bin/python animate.py        # 15 s clips of the disk turning, with and without Doppler shifts
```

The camera is fixed, so the ray map is traced once; each frame rotates the
disk's gas at its own orbital speed, delayed by each ray's light-travel time.
Writes `sci-fi/gargantua/gargantua_film.mp4` and `gargantua_doppler.mp4` (needs ffmpeg). With `--audio FILE`, the opening of that file is muxed in as a soundtrack, faded in and out.

## Files

| module | contents |
|---|---|
| `blackhole/schwarzschild.py` | orbit equation, exact (elliptic-integral) and approximate deflection, the lookup table for a camera at finite radius |
| `blackhole/kerr.py` | metric functions, Hamiltonian ray equations, a numba Dormand-Prince tracer, the camera (ZAMO frame plus aberration), Bardeen's shadow curve |
| `blackhole/disk.py` | circular orbits, ISCO, Page-Thorne flux, redshift factor g, a procedural streak texture |
| `blackhole/sky.py` | celestial-sphere textures, mip-mapped lookup, point stars lensed through any ray map |
| `blackhole/colour.py` | blackbody colours (CIE 1931), tone mapping, glare |
| `blackhole/render.py` | tracing an image (cached in `cache/`) and shading it |

Units are geometrised, G = c = M = 1: lengths in units of the hole's mass.
Traced ray maps are cached by parameter hash in `cache/`; delete it to retrace.
