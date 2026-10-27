"""Meccano parts as triangle meshes, to the real system's dimensions.

Everything in Meccano is built on a 1/2-inch grid: holes are 5/32 inch across
and sit 1/2 inch apart, strips and girder flanges are 1/2 inch wide. Getting
that grid right is most of what makes a model read as Meccano, so each part is
cut from a 2-D outline with real holes (shapely), then extruded (trimesh).

Each function returns a trimesh.Trimesh in metres, in the part's own frame:
  strip(n)            holes along +x from x=0, lying in the xy plane (normal +z)
  angle_girder(n)     corner along +x; round-hole flange in +y, slotted flange in +z
  flat_plate(a, b)    a x b hole grid in the xy plane, hole (0,0) at the origin
  flanged_plate(a, b) the same with flanges turned down (-z) on all four sides
  gear(teeth)         in the xy plane, centred, with boss along +z
  pulley(d)           a spoked/holed disc with boss, in the xy plane
  sprocket(teeth)     chain wheel
  boiler(d, h)        perforated cylinder along +z, from z=0
  nut()               hex nut, axis +z
"""
from __future__ import annotations

import math

import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import trimesh
from shapely import affinity

INCH = 0.0254
P = 0.5 * INCH             # hole pitch and strip width
HOLE_R = 5 / 64 * INCH     # 5/32-inch holes
T = 0.0009                 # sheet thickness (real strips are ~0.8 mm)
SEG = 14                   # segments in a small hole


def _circle(x, y, r, seg=SEG):
    return sg.Point(x, y).buffer(r, quad_segs=max(2, seg // 4))


def _extrude(poly, h, z0=0.0):
    """Extrude a shapely polygon (or multipolygon) h along +z, from z0."""
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    ms = [trimesh.creation.extrude_polygon(p, h) for p in polys if p.area > 0]
    m = trimesh.util.concatenate(ms)
    m.apply_translation([0, 0, z0])
    return m


def _holes_row(n, y=0.0, x0=0.0, r=HOLE_R):
    return so.unary_union([_circle(x0 + k * P, y, r) for k in range(n)])


def _slots_row(n, y=0.0, x0=0.0, r=HOLE_R, extra=0.0035):
    return so.unary_union([sg.LineString([(x0 + k * P - extra / 2, y), (x0 + k * P + extra / 2, y)]).buffer(r, quad_segs=3)
                           for k in range(n)])


# ------------------------------------------------------------------ strips and girders
def strip(n: int) -> trimesh.Trimesh:
    """Perforated strip with n holes and rounded ends."""
    L = (n - 1) * P
    outline = sg.LineString([(0, 0), (max(L, 1e-6), 0)]).buffer(P / 2, quad_segs=6)
    return _extrude(outline.difference(_holes_row(n)), T, -T / 2)


def angle_girder(n: int) -> trimesh.Trimesh:
    """Angle girder: two 1/2-inch flanges at right angles, one with round holes, one slotted."""
    L = (n - 1) * P
    rect = sg.box(-P / 2, 0, L + P / 2, P)
    a = _extrude(rect.difference(_holes_row(n, y=P / 2)), T)                      # flange in the xy plane
    b = _extrude(rect.difference(_slots_row(n, y=P / 2)), T)                      # flange to be turned up
    b.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))  # xy -> xz
    b.apply_translation([0, T, 0])
    return trimesh.util.concatenate([a, b])


def flat_girder(n: int, rows: int = 2) -> trimesh.Trimesh:
    """Flat girder: a wide strip, a row of round holes and a row of slots."""
    L = (n - 1) * P
    rect = sg.box(-P / 2, -P / 2, L + P / 2, (rows - 0.5) * P)
    holes = so.unary_union([_holes_row(n, 0.0), _slots_row(n, (rows - 1) * P)])
    return _extrude(rect.difference(holes), T, -T / 2)


def flat_plate(a: int, b: int, every: int = 1) -> trimesh.Trimesh:
    """Flat plate a x b holes (Meccano plates have holes round the edge and some inside)."""
    rect = sg.box(-P / 2, -P / 2, (a - 1) * P + P / 2, (b - 1) * P + P / 2)
    hs = []
    for i in range(a):
        for j in range(b):
            edge = i in (0, a - 1) or j in (0, b - 1)
            if edge or (i % every == 0 and j % every == 0):
                hs.append(_circle(i * P, j * P, HOLE_R))
    return _extrude(rect.difference(so.unary_union(hs)), T, -T / 2)


def flanged_plate(a: int, b: int) -> trimesh.Trimesh:
    """Flanged plate: a x b plate with 1/2-inch flanges turned down on all sides."""
    top = flat_plate(a, b, every=2)
    parts = [top]
    L, W = (a - 1) * P + P, (b - 1) * P + P
    x0, y0 = -P / 2, -P / 2
    for (n, length, place) in (
        (a, L, lambda m: m.apply_translation([0, y0, 0])),
        (a, L, lambda m: m.apply_translation([0, y0 + W - T, 0])),
    ):
        f = sg.box(x0 + 0.002, -P, x0 + length - 0.002, 0).difference(_holes_row(n, -P / 2))
        m = _extrude(f, T)
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        m.apply_translation([0, T, 0])
        place(m)
        parts.append(m)
    for (n, xpos) in ((b, x0), (b, x0 + L - T)):
        f = sg.box(y0 + 0.002, -P, y0 + W - 0.002, 0).difference(_holes_row(n, -P / 2))
        m = _extrude(f, T)
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1]))
        m.apply_translation([xpos + T, 0, 0])
        parts.append(m)
    return trimesh.util.concatenate(parts)


def bracket(width_holes: int = 1) -> trimesh.Trimesh:
    """Small right-angle bracket (two 1-hole tabs)."""
    a = _extrude(sg.box(-P / 2, 0, P / 2, P).difference(_circle(0, P / 2, HOLE_R)), T)
    b = a.copy()
    b.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    b.apply_translation([0, T, 0])
    return trimesh.util.concatenate([a, b])


# ------------------------------------------------------------------ round things
def _boss(r_out, length, r_in=HOLE_R * 0.8):
    ring = sg.Point(0, 0).buffer(r_out, quad_segs=8).difference(_circle(0, 0, r_in))
    return _extrude(ring, length)


def gear(teeth: int, face: float = 0.0032, boss: bool = True, holes: int = 0) -> trimesh.Trimesh:
    """Spur gear on Meccano's 38 diametral pitch: pitch diameter = teeth/38 inch."""
    pd = teeth / 38 * INCH
    m = pd / teeth                      # module
    ra, rr = pd / 2 + m, pd / 2 - 1.25 * m
    pts = []
    for k in range(teeth):
        a0 = 2 * math.pi * k / teeth
        da = 2 * math.pi / teeth
        for frac, r in ((0.00, rr), (0.18, rr), (0.30, ra), (0.55, ra), (0.67, rr), (1.0, rr)):
            a = a0 + frac * da
            pts.append((r * math.cos(a), r * math.sin(a)))
    g = sg.Polygon(pts).buffer(0)
    cut = [_circle(0, 0, HOLE_R)]
    if holes:
        rh = (rr + 0.006) / 2 if rr > 0.012 else 0
        if rh:
            cut += [_circle(rh * math.cos(2 * math.pi * k / holes), rh * math.sin(2 * math.pi * k / holes), HOLE_R)
                    for k in range(holes)]
    body = _extrude(g.difference(so.unary_union(cut)), face, -face / 2)
    if boss:
        return trimesh.util.concatenate([body, _boss(0.0045, 0.0075, HOLE_R)])
    return body


def sprocket(teeth: int, face: float = 0.0022) -> trimesh.Trimesh:
    """Chain wheel for Meccano sprocket chain (1/2-inch... pitch about 3/8 inch)."""
    pitch = 0.0095
    r = pitch / (2 * math.sin(math.pi / teeth))
    pts = []
    for k in range(teeth):
        a0 = 2 * math.pi * k / teeth
        da = 2 * math.pi / teeth
        for frac, rr in ((0.0, r - 0.0018), (0.25, r + 0.0022), (0.5, r - 0.0018)):
            a = a0 + frac * da
            pts.append((rr * math.cos(a), rr * math.sin(a)))
        a = a0 + 0.75 * da
        pts.append(((r - 0.0028) * math.cos(a), (r - 0.0028) * math.sin(a)))
    g = sg.Polygon(pts).buffer(0)
    ring_holes = [_circle(0.55 * r * math.cos(2 * math.pi * k / 6), 0.55 * r * math.sin(2 * math.pi * k / 6), HOLE_R * 1.6)
                  for k in range(6)] if r > 0.015 else []
    body = _extrude(g.difference(so.unary_union([_circle(0, 0, HOLE_R)] + ring_holes)), face, -face / 2)
    return trimesh.util.concatenate([body, _boss(0.0045, 0.0075, HOLE_R)])


def pulley(d: float, holes: int = 0, rim: bool = True, boss: bool = True, face: float = 0.003) -> trimesh.Trimesh:
    """Pulley wheel of diameter d, grooved rim, optional ring of holes."""
    r = d / 2
    disc = sg.Point(0, 0).buffer(r - 0.0012, quad_segs=24)
    cut = [_circle(0, 0, HOLE_R)]
    if holes:
        rh = 0.6 * r
        cut += [_circle(rh * math.cos(2 * math.pi * k / holes), rh * math.sin(2 * math.pi * k / holes), HOLE_R)
                for k in range(holes)]
    parts = [_extrude(disc.difference(so.unary_union(cut)), face * 0.6, -face * 0.3)]
    if rim:   # two flanges with a groove between them
        ring = sg.Point(0, 0).buffer(r, quad_segs=24).difference(sg.Point(0, 0).buffer(r - 0.003, quad_segs=24))
        parts += [_extrude(ring, face * 0.3, face * 0.2), _extrude(ring, face * 0.3, -face * 0.5)]
    if boss:
        parts.append(_boss(0.0045, 0.0075, HOLE_R))
    return trimesh.util.concatenate(parts)


def disc(d: float, face: float = 0.002, holes: int = 0) -> trimesh.Trimesh:
    cut = [_circle(0, 0, HOLE_R)]
    if holes:
        rh = 0.62 * d / 2
        cut += [_circle(rh * math.cos(2 * math.pi * k / holes), rh * math.sin(2 * math.pi * k / holes), HOLE_R)
                for k in range(holes)]
    return _extrude(sg.Point(0, 0).buffer(d / 2, quad_segs=24).difference(so.unary_union(cut)), face, -face / 2)


def tube(r_out: float, r_in: float, h: float, seg: int = 48) -> trimesh.Trimesh:
    ring = sg.Point(0, 0).buffer(r_out, quad_segs=seg // 4).difference(sg.Point(0, 0).buffer(r_in, quad_segs=seg // 4))
    return _extrude(ring, h)


def boiler(d: float, h: float, rows: int = 5, per_row: int = 12) -> trimesh.Trimesh:
    """Meccano boiler: a rolled sheet with rows of perforations, closed by end caps."""
    import manifold3d as mf
    r = d / 2
    shell = mf.Manifold.cylinder(h, r, r, 64).__sub__(mf.Manifold.cylinder(h + 0.002, r - T * 1.2, r - T * 1.2, 64)
                                                       .translate([0, 0, -0.001]))
    hole = mf.Manifold.cylinder(0.01, HOLE_R, HOLE_R, 12).rotate([0, 90, 0]).translate([r - 0.005, 0, 0])
    zs = np.linspace(0.18, 0.82, rows) * h
    for i, z in enumerate(zs):
        for k in range(per_row):
            ang = 360 * (k + 0.5 * (i % 2)) / per_row
            shell = shell - hole.rotate([0, 0, ang]).translate([0, 0, z])
    mesh = shell.to_mesh()
    m = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)
    caps = []
    for z in (0.0, h - 0.002):
        caps.append(_extrude(sg.Point(0, 0).buffer(r + 0.0015, quad_segs=16).difference(_circle(0, 0, HOLE_R)), 0.002, z))
    return trimesh.util.concatenate([m] + caps)


def nut() -> trimesh.Trimesh:
    """5/32 Whitworth hex nut (about 7/32 inch across flats)."""
    af = 7 / 32 * INCH
    hexa = sg.Polygon([(af / math.sqrt(3) * math.cos(math.pi / 3 * k), af / math.sqrt(3) * math.sin(math.pi / 3 * k))
                       for k in range(6)])
    return _extrude(hexa.difference(_circle(0, 0, HOLE_R * 0.8, 8)), 0.0022)


def bolt_head() -> trimesh.Trimesh:
    """Round-headed bolt, slotted."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.0035)
    m.vertices[:, 2] = np.clip(m.vertices[:, 2], 0, None) * 0.55
    return m


def chain_link() -> trimesh.Trimesh:
    """One link of sprocket chain: a pair of side plates and a roller."""
    side = sg.LineString([(0, 0), (0.0095, 0)]).buffer(0.0022, quad_segs=4)
    a = _extrude(side, 0.0006, 0.0018)
    b = _extrude(side, 0.0006, -0.0024)
    roll = trimesh.creation.cylinder(radius=0.0016, height=0.0042, sections=8)
    return trimesh.util.concatenate([a, b, roll])


def save(mesh: trimesh.Trimesh, path) -> None:
    mesh.export(path)
