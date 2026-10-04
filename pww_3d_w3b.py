# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d_w3b.py — vpython scenes: the rhombic dodecahedron, an octahedron in
a cube, the Menger sponge and the stepped pyramid (I20, I22, I26, I27).

Each scene_<ID>() is found by name by pww_3d.py / the app. Every
construction shown here is rebuilt from its rule and checked before
anything is drawn: tilings, landings, coplanar faces, free paths and
volumes are verified numerically, so a construction that does not close
fails at start-up instead of animating a wrong picture.
"""

from pww_3d_kit import *          # noqa: F401,F403  — must come first

import itertools

FPS = 40


# ================================================================ helpers

def _check(cond, what):
    """Refuse to start a scene whose construction is wrong."""
    if not cond:
        raise AssertionError(f"construction check failed: {what}")


def _smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _frames(seconds):
    """Yield t in [0, 1] once per frame for `seconds`."""
    n = max(1, int(round(seconds * FPS)))
    for i in range(n + 1):
        rate(FPS)
        yield i / n


def _wait(seconds):
    for _ in _frames(seconds):
        pass


def _ready(sc):
    """Start the clock once the browser has received the whole scene
    (vpython already blocks at the first canvas until a page connects)."""
    sc.waitfor("draw_complete")


def _pre(t):
    return f"<pre style='margin:0'>{t}</pre>"


def _rows(sc, n):
    """n readout lines under the canvas."""
    rows = []
    for _ in range(n):
        sc.append_to_caption("\n")
        rows.append(wtext(text=_pre(" ")))
    return rows


def _lab(pos, text, col=C_HL, height=16, visible=True, back=False):
    """A label; with back=True it gets a dark backing, so that it can sit
    on a coloured face and stay legible."""
    if back:
        return label(pos=pos, text=text, height=height, box=False, color=col,
                     background=V(0.03, 0.03, 0.05), opacity=0.62,
                     visible=visible)
    return label(pos=pos, text=text, height=height, box=False, color=col,
                 opacity=0, visible=visible)


def _show(objs, flag):
    for ob in objs:
        ob.visible = flag


def _sub(n):
    """An integer in subscript digits."""
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _sup(n):
    """An integer in superscript digits."""
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


class _AxisTurn:
    """The rotation by s·angle about a fixed axis through the origin
    (s = 1 is the whole turn): R(s)·v by Rodrigues' formula."""

    def __init__(self, axis, angle):
        self.axis = axis.norm()
        self.theta = angle

    def apply(self, s, v):
        a = s * self.theta
        if abs(a) < 1e-15:
            return V(v.x, v.y, v.z)
        k = self.axis
        c, sn = math.cos(a), math.sin(a)
        return v * c + k.cross(v) * sn + k * (k.dot(v) * (1 - c))


_IDENT = _AxisTurn(V(0, 1, 0), 0.0)


class _Rigid:
    """A rigid body made of several vpython objects (compounds), posed as a
    whole: world = P + R·offset for every part, all parts turned by R."""

    def __init__(self):
        self.parts = []          # (obj, offset from body centre, |axis|)

    def add(self, obj, offset):
        self.parts.append((obj, offset, obj.axis.mag))

    def pose(self, turn, s, P):
        for obj, off, L0 in self.parts:
            obj.pos = P + turn.apply(s, off)
            obj.axis = turn.apply(s, V(1, 0, 0)) * L0
            obj.up = turn.apply(s, V(0, 1, 0))


def _vt(v):
    return (v.x, v.y, v.z)


class _Dim:
    """A dimension line beside the edge p–q: a thin line set off by `off`,
    with a tick at each end and its label beyond the middle."""

    def __init__(self, p, q, off, text, col=C_HL, visible=True, lab_gap=2.4):
        self.off, self.lab_gap = off, lab_gap
        self.line = curve(pos=[p, q], radius=0.008, color=col, visible=visible)
        self.ticks = [curve(pos=[p, q], radius=0.008, color=col, visible=visible)
                      for _ in range(2)]
        self.lab = _lab(p, text, col, visible=visible)
        self.set(p, q)

    def set(self, p, q, trim=0.025):
        d = (q - p).norm()
        a, b = p + d * trim, q - d * trim
        o = self.off
        self.line.modify(0, pos=a + o)
        self.line.modify(1, pos=b + o)
        for tk, e in zip(self.ticks, (a, b)):
            tk.modify(0, pos=e + o * 0.45)
            tk.modify(1, pos=e + o * 1.55)
        self.lab.pos = (p + q) / 2 + o * self.lab_gap

    def show(self, flag):
        for c in [self.line, self.lab] + self.ticks:
            c.visible = flag


def _screen_basis(forward):
    """Unit vectors to the right and up on the screen, for the viewing
    direction `forward` the canvas was made with. (Not read back from the
    canvas: once the page is live, canvas.forward on the Python side need
    not be the direction the view really has.)"""
    f = forward.norm()
    right = f.cross(V(0, 1, 0)).norm()
    return right, right.cross(f).norm()


def _rot_about_line(p, a, k, ang, lift=0.0):
    """The point p turned by `ang` about the line through a (raised by
    `lift` along y) with unit direction k (Rodrigues)."""
    a = a + V(0, lift, 0)
    v = p - a
    c, s = math.cos(ang), math.sin(ang)
    return a + v * c + k.cross(v) * s + k * (k.dot(v) * (1 - c))


def _convex_apart(pa, ea, fa, pb, eb, fb, eps=1e-9):
    """Are two convex polyhedra disjoint (their interiors)? Given as vertex
    lists, edge direction lists and face normal lists: they are apart if
    and only if some face normal or cross product of edges separates them
    (the separating axis theorem)."""
    axes = list(fa) + list(fb)
    for d1 in ea:
        for d2 in eb:
            c = d1.cross(d2)
            if c.mag > 1e-9:
                axes.append(c.norm())
    for a in axes:
        a1 = [p.dot(a) for p in pa]
        b1 = [p.dot(a) for p in pb]
        if max(a1) <= min(b1) + eps or max(b1) <= min(a1) + eps:
            return True
    return False


def _poly_parts(faces):
    """Vertices, edge directions and face normals of a convex polyhedron
    given by its faces, for _convex_apart (directions kept once each, up
    to sign: a parallel copy adds no new separating axis)."""
    pts, edges, nrms = [], [], []

    def add(lst, d):
        if all(d.cross(e).mag > 1e-9 for e in lst):
            lst.append(d)

    for f in faces:
        for p in f:
            if all((p - q).mag > 1e-9 for q in pts):
                pts.append(p)
        for a, b in zip(f, f[1:] + f[:1]):
            add(edges, (b - a).norm())
        add(nrms, (f[1] - f[0]).cross(f[2] - f[0]).norm())
    return pts, edges, nrms


def _mesh_tris(faces, col, opacity=1.0):
    """The triangles of a closed polyhedron given by planar faces (lists of
    points, either orientation), each face lit by its outward normal.
    Returns (triangles, centroid of the corners, volume); the volume comes
    from the divergence theorem on the outward-oriented faces, so it
    checks that the faces really close up into the intended solid."""
    pts = [p for f in faces for p in f]
    g = sum(pts, V(0, 0, 0)) / len(pts)
    tris, vol = [], 0.0
    for f in faces:
        fc = sum(f, V(0, 0, 0)) / len(f)
        nrm = (f[1] - f[0]).cross(f[2] - f[0]).norm()
        if nrm.dot(fc - g) < 0:
            f = list(reversed(f))
            nrm = -nrm
        for i in range(1, len(f) - 1):
            p, q, r_ = f[0], f[i], f[i + 1]
            vol += (p - g).dot((q - g).cross(r_ - g)) / 6.0
            tris.append(triangle(vs=[vertex(pos=p, color=col, normal=nrm, opacity=opacity),
                                     vertex(pos=q, color=col, normal=nrm, opacity=opacity),
                                     vertex(pos=r_, color=col, normal=nrm, opacity=opacity)]))
    return tris, g, vol


def _mesh(faces, col, opacity=1.0):
    """A closed polyhedron as one compound whose reference point is the
    centroid of its corners: (compound, centroid, volume)."""
    tris, g, vol = _mesh_tris(faces, col, opacity)
    return compound(tris, origin=g), g, vol


def _hull_faces(verts):
    """Faces of a convex polyhedron (index lists in cyclic order) from its
    vertices: the planes through three vertices with all others behind."""
    found = {}
    for i, j, l in itertools.combinations(range(len(verts)), 3):
        nr = (verts[j] - verts[i]).cross(verts[l] - verts[i])
        if nr.mag < 1e-9:
            continue
        nr = nr.norm()
        d = nr.dot(verts[i])
        side = [nr.dot(v) - d for v in verts]
        if max(side) > 1e-9:
            nr, d = -nr, -d
            side = [-s_ for s_ in side]
            if max(side) > 1e-9:
                continue
        on = tuple(q for q, sv in enumerate(side) if sv > -1e-9)
        found[on] = nr
    faces = []
    for idx, nr in found.items():
        c = sum((verts[i] for i in idx), V(0, 0, 0)) / len(idx)
        e1 = (verts[idx[0]] - c).norm()
        e2 = nr.cross(e1)
        faces.append(sorted(idx, key=lambda i: math.atan2((verts[i] - c).dot(e2),
                                                          (verts[i] - c).dot(e1))))
    return faces


def _dedup(pts, eps=1e-9):
    out = []
    for p in pts:
        if all((p - q).mag > eps for q in out):
            out.append(p)
    return out


def _clip_y(faces, upper):
    """The part of a convex polyhedron (given by its faces) on one side of
    the plane y = 0, as faces (point lists), from its corners on that side
    and the points where its edges cross the plane."""
    sg = 1 if upper else -1
    pts = []
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            if sg * a.y >= -1e-12:
                pts.append(a)
            if (a.y > 1e-12 and b.y < -1e-12) or (a.y < -1e-12 and b.y > 1e-12):
                t = a.y / (a.y - b.y)
                pts.append(a + (b - a) * t)
    pts = _dedup(pts)
    return [[pts[i] for i in f] for f in _hull_faces(pts)]


def _faces_volume(faces):
    pts = [p for f in faces for p in f]
    g = sum(pts, V(0, 0, 0)) / len(pts)
    vol = 0.0
    for f in faces:
        fc = sum(f, V(0, 0, 0)) / len(f)
        nrm = (f[1] - f[0]).cross(f[2] - f[0])
        if nrm.dot(fc - g) < 0:
            f = list(reversed(f))
        for i in range(1, len(f) - 1):
            vol += (f[0] - g).dot((f[i] - g).cross(f[i + 1] - g)) / 6.0
    return vol


def _face_edges(faces):
    """The edges of a polyhedron given by its faces, each once."""
    segs = []
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            if not any(((a - p).mag < 1e-9 and (b - q).mag < 1e-9) or
                       ((a - q).mag < 1e-9 and (b - p).mag < 1e-9) for p, q in segs):
                segs.append((a, b))
    return segs


def _arc(c, u, w, r, n=24):
    """Points of the circular arc of radius r about c from the unit
    direction u to the unit direction w (the non-reflex way)."""
    ang = math.acos(max(-1.0, min(1.0, u.dot(w))))
    k = u.cross(w)
    if k.mag < 1e-12:
        return [c + u * r]
    k = k.norm()
    return [c + _AxisTurn(k, ang).apply(i / n, u) * r for i in range(n + 1)]


# ============================================================== I20

def scene_I20():
    """V = a³ + 6·a³/6 = 2a³: the planes through the centre of a cube and
    its edges cut it into six congruent pyramids of height a/2, one on
    each face (each a³/6). Part them, bring a second cube of the same edge
    into the middle, turn every pyramid over so that its base faces the
    cube, and close them onto its faces: the result is the rhombic
    dodecahedron, cube + six pyramids = 2a³. Its faces are rhombi because
    each slanted face rises 45° from its square (height a/2 over the
    half-width a/2), and two of them meet over an edge of the cube where
    the cube's faces make 90°: 45° + 90° + 45° = 180°, the two triangles
    lie in one plane."""
    # the view: wide while the pyramids are apart, close on the solid, and
    # a little higher while its top half is lifted off
    CAM_WIDE = (2.75, V(0.25, 0.1, 0))
    CAM_SOLID = (1.62, V(0.0, 0.0, 0))
    CAM_CUT = (1.95, V(0.0, 0.42, 0))
    sc = new_canvas("rhombic dodecahedron:   V  =  a³ + 6 · a³/6  =  2a³",
                    "the six pyramids from the centre of one cube, turned over onto the faces "
                    "of a second, equal cube",
                    rng=CAM_WIDE[0], forward=V(-0.52, -0.50, -0.69), centre=CAM_WIDE[1])
    sc.fov = 0.45

    def cam(c0, c1, t):
        """The view part way (t) from c0 to c1."""
        sc.range = c0[0] + (c1[0] - c0[0]) * t
        sc.center = c0[1] + (c1[1] - c0[1]) * t

    h = 0.62
    a = 2 * h
    O = V(0, 0, 0)
    AX = [V(1, 0, 0), V(-1, 0, 0), V(0, 1, 0), V(0, -1, 0), V(0, 0, 1), V(0, 0, -1)]
    D = 2.2 * h                                   # how far the pyramids part
    # each pyramid turns over about the line through its base centre along
    # E[k], by SG[k]·180° (the side ones tip their apex upward)
    E = [V(0, 0, 1), V(0, 0, 1), V(0, 0, 1), V(0, 0, 1), V(1, 0, 0), V(1, 0, 0)]
    SG = [-1, 1, 1, 1, 1, -1]
    TURN = [_AxisTurn(e, sg * math.pi) for e, sg in zip(E, SG)]
    B0 = V(1, 0, -1) * (2.6 * h)                  # where the second cube waits

    def face(nv):
        """The face of the cube (centred at O) with outward normal nv."""
        a_ = V(nv.y, nv.z, nv.x)
        b_ = nv.cross(a_)
        c = nv * h
        return [c + (a_ * sa + b_ * sb) * h for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

    pyr = []                                      # faces of the pyramids, in the first cube
    for nv in AX:
        base = face(nv)
        pyr.append([base] + [[O, base[i], base[(i + 1) % 4]] for i in range(4)])
    cube_faces = [face(nv) for nv in AX]

    def T(k, d, s, p):
        """Where the point p of pyramid k is when the pyramids have parted
        by d and pyramid k has turned by the fraction s."""
        nv = AX[k]
        return nv * (h + d) + TURN[k].apply(s, p - nv * h)

    def pyr_at(k, d, s):
        return [[T(k, d, s, p) for p in f] for f in pyr[k]]

    def cube_at(P):
        return [[p + P for p in f] for f in cube_faces]

    # ---- checks: the six pyramids are a³/6 each and fill the first cube
    for fs in pyr:
        _check(abs(_faces_volume(fs) - a ** 3 / 6) < 1e-12 and
               abs(a ** 3 / 6 - (a * a) * h / 3) < 1e-12, "each pyramid is a³/6 = ⅓·a²·(a/2)")
    n_g = 10
    for i in range(n_g):
        for j in range(n_g):
            for l_ in range(n_g):
                p = V(-h + a * (i + 0.5) / n_g, -h + a * (j + 0.37) / n_g,
                      -h + a * (l_ + 0.61) / n_g)
                inside = 0
                for nv in AX:
                    t = p.dot(nv)
                    others = [e for e in AX if abs(e.dot(nv)) < 0.5]
                    inside += all(t > p.dot(e) + 1e-12 for e in others)
                _check(inside == 1, "the six pyramids fill the cube, each point once")
    # ---- the turn: the base stays on its face, the apex goes to 2h·n
    for k, nv in enumerate(AX):
        _check((T(k, 0, 1, O) - nv * a).mag < 1e-12, "the apex ends a/2 beyond the face")
        moved = {tuple(round(c, 9) for c in _vt(T(k, 0, 1, p))) for p in pyr[k][0]}
        _check(moved == {tuple(round(c, 9) for c in _vt(p)) for p in pyr[k][0]},
               "turned over, the base lies on the same face")

    # ---- every stage of the motion: no two pieces overlap. The timeline
    # (u in [0, 1] per stage): part, the cube comes in, turn over, close.
    def state(stage, u):
        if stage == 0:
            return D * u, 0.0, B0
        if stage == 1:
            return D, 0.0, B0 * (1 - u)
        if stage == 2:
            return D, u, O
        return D * (1 - u), 1.0, O

    for stage in range(4):
        for q in range(0, 31):
            d, s, P = state(stage, _smooth(q / 30))
            bodies_ = [_poly_parts(pyr_at(k, d, s)) for k in range(6)]
            bodies_.append(_poly_parts(cube_at(P)))
            for b1, b2 in itertools.combinations(bodies_, 2):
                _check(_convex_apart(*b1, *b2), "the pieces never run into each other")
    # before parting, the second cube is clear of the first
    _check(_convex_apart(*_poly_parts(cube_at(B0)), *_poly_parts(cube_faces)),
           "the second cube waits beside the first")

    # ---- the solid: the cube and the six pyramids on it, apexes 2h·n
    apex = [nv * a for nv in AX]
    corners = [V(x, y, z) * h for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    rhombi = []                                   # (apex j, corner, apex k, corner)
    for j, k in itertools.combinations(range(6), 2):
        if abs(AX[j].dot(AX[k])) > 0.5:
            continue                              # opposite faces share no edge
        m = AX[j].cross(AX[k])
        e1 = (AX[j] + AX[k]) * h + m * h
        e2 = (AX[j] + AX[k]) * h - m * h
        quad_ = [apex[j], e1, apex[k], e2]
        nrm = (AX[j] + AX[k]).norm()
        # the four corners lie in one plane: n_j·p + n_k·p = a ...
        off = max(abs(nrm.dot(p) - nrm.dot(apex[j])) for p in quad_)
        det = (apex[k] - apex[j]).dot((e1 - apex[j]).cross(e2 - apex[j]))
        _check(off < 1e-12 and abs(det) < 1e-12 and
               all(abs((AX[j] + AX[k]).dot(p) - a) < 1e-12 for p in quad_),
               "the two triangles over an edge of the cube are coplanar")
        # ... a supporting plane of the whole solid, so it is a face ...
        _check(all(nrm.dot(p) <= nrm.dot(apex[j]) + 1e-12 for p in apex + corners),
               "that plane bounds the solid: the rhombus is a face")
        # ... and a rhombus: four equal sides, diagonals a√2 and a
        sides = [(quad_[i] - quad_[(i + 1) % 4]).mag for i in range(4)]
        _check(max(sides) - min(sides) < 1e-12 and abs((apex[j] - apex[k]).mag - a * math.sqrt(2))
               < 1e-12 and abs((e1 - e2).mag - a) < 1e-12, "a rhombus, diagonals a√2 and a")
        rhombi.append(quad_)
    _check(len(rhombi) == 12, "twelve rhombi")
    _check(abs(_faces_volume(rhombi) - 2 * a ** 3) < 1e-12, "the solid is 2a³")
    _check(abs(a ** 3 + 6 * a ** 3 / 6 - 2 * a ** 3) < 1e-12, "a³ + 6·a³/6 = 2a³")

    # ---- the cut at mid-height: the section is a square through the four
    # side apexes; each of its sides runs straight through a corner of the
    # cube's section, where 45° + 90° + 45° = 180°
    Pc = V(h, 0, h)                               # the front corner of the section
    ax_, az_ = apex[0], apex[4]                   # the apexes beside it
    _check(((ax_ - Pc).cross(az_ - Pc)).mag < 1e-12 and (ax_ - Pc).dot(az_ - Pc) < 0,
           "the corner lies on the side between the two apexes")
    u_x, u_sq1, u_sq2, u_z = ((ax_ - Pc).norm(), V(0, 0, -1), V(-1, 0, 0), (az_ - Pc).norm())

    def deg(u, w):
        return math.degrees(math.acos(max(-1.0, min(1.0, u.dot(w)))))

    _check(abs(deg(u_x, u_sq1) - 45) < 1e-9 and abs(deg(u_sq1, u_sq2) - 90) < 1e-9 and
           abs(deg(u_sq2, u_z) - 45) < 1e-9 and abs(deg(u_x, u_z) - 180) < 1e-9,
           "45° + 90° + 45° = 180°")

    # ---- drawing. The pyramids are coloured by axis (opposite ones alike);
    # the second cube is purple. Each pyramid's edges are lines that move
    # with it: the four slanted ones bright, the base square fainter.
    COLS = [C_A, C_A, C_C, C_C, C_B, C_B]
    C_CUBE = C_F
    bodies = []
    for k, (fs, col) in enumerate(zip(pyr, COLS)):
        cp, g, vol = _mesh(fs, col)
        _check(abs(vol - a ** 3 / 6) < 1e-9, "the drawn pyramid closes up: a³/6")
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        base = fs[0]
        segs = [(O, b_) for b_ in base] + [(base[i], base[(i + 1) % 4]) for i in range(4)]
        edges = [curve(pos=[p, q], radius=0.011, color=C_HL * 0.9) for p, q in segs]
        bodies.append(dict(body=body, cp=cp, g=g, segs=segs, edges=edges))
    cubeB = box(pos=B0, size=V(a, a, a), color=C_CUBE)
    cube_segs = [(p, q) for p, q in itertools.combinations(corners, 2)
                 if abs((p - q).mag - a) < 1e-9]
    cubeB_edges = [curve(pos=[p + B0, q + B0], radius=0.011, color=C_HL * 0.9)
                   for p, q in cube_segs]
    ghost = [curve(pos=[p, q], radius=0.008, color=C_HL * 0.5) for p, q in cube_segs]
    dotO = sphere(pos=O, radius=0.045, color=C_HL)

    def pose(k, d, s):
        b = bodies[k]
        b["body"].pose(TURN[k], s, T(k, d, s, b["g"]))
        for cv, (p, q) in zip(b["edges"], b["segs"]):
            cv.modify(0, pos=T(k, d, s, p))
            cv.modify(1, pos=T(k, d, s, q))

    def place_B(P):
        cubeB.pos = P
        for cv, (p, q) in zip(cubeB_edges, cube_segs):
            cv.modify(0, pos=p + P)
            cv.modify(1, pos=q + P)

    def look(op, base_bright=0.9):
        for b in bodies:
            b["cp"].opacity = op
            for i, cv in enumerate(b["edges"]):
                cv.color = C_HL * (0.9 if i < 4 else base_bright)

    # the edge of each cube named a (the front edge at the bottom)
    dimA = _Dim(V(-h, -h, h), V(h, -h, h), V(0, -0.06, 0.13), "a", lab_gap=2.7)
    dimB = _Dim(V(-h, -h, h) + B0, V(h, -h, h) + B0, V(0, -0.06, 0.13), "a", lab_gap=2.7)

    # ---- the solid cut at mid-height: the lower half stays, the upper half
    # lifts. Both are built from the pieces in their final places, so the
    # section shows the cube's square and the four triangles around it.
    fin = [pyr_at(k, 0, 1) for k in range(6)]
    up_parts, low_parts = [], []
    vol_up = vol_low = 0.0
    for k, fs in enumerate(fin + [cube_faces]):
        col = COLS[k] if k < 6 else C_CUBE
        for upper, parts in ((True, up_parts), (False, low_parts)):
            ys = [p.y for f in fs for p in f]
            if (upper and min(ys) >= -1e-12) or (not upper and max(ys) <= 1e-12):
                half = fs
            elif (upper and max(ys) <= 1e-12) or (not upper and min(ys) >= -1e-12):
                continue
            else:
                half = _clip_y(fs, upper)
            tris, _, vh = _mesh_tris(half, col)
            if upper:
                vol_up += vh
            else:
                vol_low += vh
            parts += tris
            for p, q in _face_edges(half):
                on_cut = abs(p.y) < 1e-12 and abs(q.y) < 1e-12
                parts.append(cylinder(pos=p, axis=q - p, radius=0.011 if on_cut else 0.009,
                                      color=C_HL * (1.0 if on_cut else 0.75)))
    _check(abs(vol_up - a ** 3) < 1e-9 and abs(vol_low - a ** 3) < 1e-9,
           "the cut halves the solid: a³ above, a³ below")
    upper_half = compound(up_parts, origin=O, visible=False)
    lower_half = compound(low_parts, origin=O, visible=False)
    LIFT = 1.45 * h
    # the side of the section through the front corner, and its angles
    side_line = curve(pos=[ax_ + V(0, 0.004, 0), az_ + V(0, 0.004, 0)], radius=0.016,
                      color=C_HL, visible=False)
    rA = 0.62 * h
    yy = V(0, 0.006, 0)
    arcs = [curve(pos=[p + yy for p in _arc(Pc, u_x, u_sq1, rA)], radius=0.012, color=C_HL,
                  visible=False),
            curve(pos=[p + yy for p in [Pc + u_sq1 * (rA * 0.5),
                                        Pc + (u_sq1 + u_sq2) * (rA * 0.5),
                                        Pc + u_sq2 * (rA * 0.5)]],
                  radius=0.012, color=C_HL, visible=False),
            curve(pos=[p + yy for p in _arc(Pc, u_sq2, u_z, rA)], radius=0.012, color=C_HL,
                  visible=False)]
    arc_labs = [_lab(Pc + (u_x + u_sq1).norm() * (rA * 1.6), "45°", back=True, visible=False),
                _lab(Pc + (u_sq1 + u_sq2).norm() * (rA * 1.25), "90°", back=True, visible=False),
                _lab(Pc + (u_sq2 + u_z).norm() * (rA * 1.6), "45°", back=True, visible=False)]
    # one rhombus picked out: the one over the front vertical edge, outlined
    # and washed over in one colour (it is flat)
    rh_front = [apex[0], V(h, h, h), apex[4], V(h, -h, h)]
    rh_n = (AX[0] + AX[4]).norm()
    rh_line = curve(pos=[p + rh_n * 0.012 for p in rh_front + rh_front[:1]], radius=0.018,
                    color=C_HL, visible=False)
    rh_fill = quad(vs=[vertex(pos=p + rh_n * 0.012, normal=rh_n, color=C_D)
                       for p in rh_front], visible=False)

    def cut_marks(flag):
        _show([side_line] + arcs + arc_labs, flag)

    row1, row2, row3 = _rows(sc, 3)
    _ready(sc)
    while True:
        # ---- two equal cubes; the first is six pyramids from its centre
        cam(CAM_WIDE, CAM_WIDE, 0)
        for k in range(6):
            pose(k, 0, 0)
        place_B(B0)
        look(0.45)
        _show([dotO], True)
        dimA.show(True)
        dimB.show(True)
        row1.text = _pre("two equal cubes of edge a;  the first is cut from its centre "
                         "through its edges")
        row2.text = row3.text = _pre(" ")
        _wait(2.3)
        # ---- six pyramids part
        dimA.show(False)
        dimB.show(False)
        _show([dotO], False)
        row1.text = _pre("six congruent pyramids, one on each face:  base a², height a/2,  "
                         "each  ⅓ · a² · a/2  =  a³/6")
        for t in _frames(1.7):
            d = D * _smooth(t)
            for k in range(6):
                pose(k, d, 0)
            look(0.45 + 0.5 * t)
        _wait(0.5)
        # ---- the second cube moves into the middle
        row1.text = _pre("the second cube moves into the middle")
        for t in _frames(1.5):
            place_B(B0 * (1 - _smooth(t)))
        _wait(0.2)
        # ---- each pyramid turns over: its base now faces the cube
        row1.text = _pre("each pyramid turns over:  its base faces the cube, its apex points out")
        for t in _frames(1.9):
            for k in range(6):
                pose(k, D, _smooth(t))
        _wait(0.2)
        # ---- and closes onto its face
        row1.text = _pre("each base lands on a face of the second cube")
        for t in _frames(1.6):
            for k in range(6):
                pose(k, D * (1 - _smooth(t)), 1)
            cam(CAM_WIDE, CAM_SOLID, _smooth(t))
        look(1.0, base_bright=0.35)
        row2.text = _pre("V  =  a³ + 6 · a³/6  =  2a³")
        _wait(0.7)
        # ---- the faces: two triangles over each edge make one rhombus
        _show([rh_line, rh_fill], True)
        row1.text = _pre("over every edge of the cube two triangles lie in one plane:  "
                         "24 triangles make 12 rhombi")
        row3.text = _pre("this one (cube centred at 0):  its corners (a, 0, 0), (a/2, ±a/2, a/2), "
                         "(0, 0, a)  all have  x + z = a")
        _wait(3.4)
        # ---- why: cut at mid-height and lift the top
        _show([rh_line, rh_fill], False)
        for b in bodies:
            b["cp"].visible = False
            _show(b["edges"], False)
        cubeB.visible = False
        _show(cubeB_edges + ghost, False)
        upper_half.visible = lower_half.visible = True
        row1.text = _pre("cut at mid-height:  each slanted face rises at 45°  (height a/2 over "
                         "half-width a/2)")
        row3.text = _pre(" ")
        for t in _frames(1.4):
            upper_half.pos = V(0, LIFT * _smooth(t), 0)
            cam(CAM_SOLID, CAM_CUT, _smooth(t))
        cut_marks(True)
        row3.text = _pre("over each edge of the cube:  45° + 90° + 45° = 180°,  "
                         "so the two triangles make one flat rhombus")
        _wait(4.6)
        cut_marks(False)
        for t in _frames(1.2):
            upper_half.pos = V(0, LIFT * (1 - _smooth(t)), 0)
            cam(CAM_CUT, CAM_SOLID, _smooth(t))
        upper_half.visible = lower_half.visible = False
        for b in bodies:
            b["cp"].visible = True
            _show(b["edges"], True)
        cubeB.visible = True
        _show(cubeB_edges + ghost, True)
        row1.text = _pre("the rhombic dodecahedron:  the second cube and the six pyramids "
                         "of the first")
        row3.text = _pre("twelve rhombic faces, two triangles over each of the cube's "
                         "twelve edges")
        _wait(3.6)
        # ---- and back
        row1.text = row2.text = row3.text = _pre(" ")
        look(0.95)
        for t in _frames(1.0):
            for k in range(6):
                pose(k, D * _smooth(t), 1)
            cam(CAM_SOLID, CAM_WIDE, _smooth(t))
        for t in _frames(0.9):
            for k in range(6):
                pose(k, D, 1 - _smooth(t))
        for t in _frames(0.9):
            place_B(B0 * _smooth(t))
        for t in _frames(1.0):
            d = D * (1 - _smooth(t))
            for k in range(6):
                pose(k, d, 0)
            look(0.95 - 0.5 * t)
        _wait(0.4)


# ============================================================== I22

def scene_I22():
    """V = a³/6: the centres of the six faces of a cube of edge a are the
    corners of an octahedron. Its middle square cuts it into two square
    pyramids, apex at the centres of the top and bottom faces, height
    a/2. The square joins the midpoints of the sides of the cube's
    mid-section (an a × a square); the four corners of that section fold
    over its sides onto it and cover it exactly, so it is half the face,
    ½·a². Each pyramid is ⅓·½a²·a/2 = a³/12, the two together a³/6."""
    FWD = V(-0.52, -0.56, -0.65).norm()
    sc = new_canvas("octahedron in a cube:   V  =  2 · ⅓ · ½a² · a/2  =  a³/6",
                    "the centres of the faces span two square pyramids of height a/2 "
                    "on a base of half a face",
                    rng=2.15, forward=FWD, centre=V(0.1, 0.42, 0))
    sc.fov = 0.45
    s_right, s_up = _screen_basis(FWD)
    h = 1.0
    a = 2 * h
    O = V(0, 0, 0)
    X, Y, Z = V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)
    top, bot = Y * h, -Y * h
    ring = [X * h, Z * h, -X * h, -Z * h]          # the middle square, in order
    centres = [X * h, -X * h, top, bot, Z * h, -Z * h]
    up_faces = [ring[:]] + [[top, ring[i], ring[(i + 1) % 4]] for i in range(4)]
    lo_faces = [ring[:]] + [[bot, ring[i], ring[(i + 1) % 4]] for i in range(4)]
    LIFT = 1.6 * h        # how far the top pyramid rises: clear of the whole section

    # ---- checks: the corners are the face centres; neighbouring centres
    # are joined (12 edges a/√2); two pyramids of ⅓·½a²·a/2 = a³/12
    corners = [V(x, y, z) * h for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    cube_segs = [(p, q) for p, q in itertools.combinations(corners, 2)
                 if abs((p - q).mag - a) < 1e-9]
    for c in centres:
        fc = [p for p in corners if abs(p.dot(c.norm()) - h) < 1e-12]
        _check(len(fc) == 4 and (sum(fc, V(0, 0, 0)) / 4 - c).mag < 1e-12,
               "each corner of the octahedron is the centre of a face")
    oct_edges = [(p, q) for p, q in itertools.combinations(centres, 2)
                 if abs(p.dot(q)) < 1e-12]
    _check(len(oct_edges) == 12 and
           all(abs((p - q).mag - a / math.sqrt(2)) < 1e-12 for p, q in oct_edges),
           "twelve edges a/√2 join the centres of neighbouring faces")
    v_up, v_lo = _faces_volume(up_faces), _faces_volume(lo_faces)
    B_area = sum(((ring[i] - O).cross(ring[(i + 1) % 4] - O)).mag / 2 for i in range(4))
    _check(abs(B_area - a * a / 2) < 1e-12, "the middle square is ½a²")
    _check(abs(v_up - B_area * h / 3) < 1e-12 and abs(v_lo - v_up) < 1e-12 and
           abs(v_up - a ** 3 / 12) < 1e-12, "each pyramid is ⅓·½a²·a/2 = a³/12")
    _check(abs(v_up + v_lo - a ** 3 / 6) < 1e-12, "the octahedron is a³/6")
    # the octahedron is |x| + |y| + |z| ≤ a/2, inside the cube: a sample of
    # the cube lands in it a sixth of the time, as the volume says
    n_g, hits = 24, 0
    for i in range(n_g):
        for j in range(n_g):
            for k in range(n_g):
                p = V(-h + a * (i + 0.5) / n_g, -h + a * (j + 0.5) / n_g,
                      -h + a * (k + 0.5) / n_g)
                hits += abs(p.x) + abs(p.y) + abs(p.z) < h
    _check(abs(hits / n_g ** 3 - 1 / 6) < 0.02, "the octahedron fills a sixth of the cube")

    # ---- the cube's mid-section: the middle square and four corner
    # triangles, each hinged on a side of the square; turned over the hinge
    # by 180° a corner lands on a quarter of the square
    sec = [V(sx, 0, sz) * h for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    flaps = []                     # (hinge p, hinge q, corner, hinge direction, turn sign)
    for c in sec:
        p, q = V(c.x, 0, 0), V(0, 0, c.z)
        k = (q - p).norm()
        sg = 1 if k.cross(c - p).y > 0 else -1     # the way that lifts the corner first
        flaps.append((p, q, c, k, sg))

    def fold(fl, x, u, lift=0.0):
        """x turned with flap fl by u·180° about its hinge."""
        p, q, c, k, sg = fl
        return _rot_about_line(x, p, k, sg * math.pi * u, lift)

    for fl in flaps:
        p, q, c, k, sg = fl
        _check(abs(fold(fl, c, 0.5).y - (c - p).cross(k).mag) < 1e-12 and
               fold(fl, c, 0.5).y > 0, "each corner turns upward")
        _check(fold(fl, c, 1).mag < 1e-12 and (fold(fl, p, 1) - p).mag < 1e-12 and
               (fold(fl, q, 1) - q).mag < 1e-12, "turned over its hinge, the corner lands "
                                                   "on the centre")
        _check(abs(((q - p).cross(c - p)).mag / 2 - h * h / 2) < 1e-12, "each corner is ¼ of ½a²")
        # the whole turn stays below the lifted pyramid
        for u in [i / 30 for i in range(31)]:
            _check(fold(fl, c, u).y < LIFT - 0.15 * h, "the corners turn under the top pyramid")
    for i in range(n_g):
        for j in range(n_g):
            x_, z_ = -h + a * (i + 0.5) / n_g, -h + a * (j + 0.31) / n_g
            if abs(x_) + abs(z_) >= h - 1e-9:
                continue
            n_in = 0
            for fl in flaps:
                tri = [fl[0], fl[1], fold(fl, fl[2], 1)]
                s_ = [((tri[(m + 1) % 3] - tri[m]).cross(V(x_, 0, z_) - tri[m])).y for m in range(3)]
                n_in += all(v > 1e-12 for v in s_) or all(v < -1e-12 for v in s_)
            _check(n_in == 1, "the four corners cover the middle square exactly once")
    _check(abs(4 * h * h / 2 - B_area) < 1e-12 and abs(B_area - a * a / 2) < 1e-12,
           "the four corners add up to the middle square: it is half of a × a")

    # ---- drawing
    box(pos=O, size=V(a, a, a), color=V(0.55, 0.6, 0.75), opacity=0.08)
    for p, q in cube_segs:
        curve(pos=[p, q], radius=0.009, color=C_HL * 0.55)
    diags = []
    for c in centres:
        fc = [p for p in corners if abs(p.dot(c.norm()) - h) < 1e-12]
        for p in fc:
            for q in fc:
                if (p + q - c * 2).mag < 1e-9 and _vt(p) < _vt(q):
                    diags.append(curve(pos=[p, q], radius=0.006, color=C_HL * 0.4))
    dots = [sphere(pos=c, radius=0.045, color=C_HL) for c in centres]
    edge_cv = [curve(pos=[p, p], radius=0.012, color=C_HL) for p, _ in oct_edges]
    cp_up, g_up, _ = _mesh(up_faces, C_C)
    cp_lo, g_lo, _ = _mesh(lo_faces, C_A)
    for cp in (cp_up, cp_lo):
        cp.opacity = 0
        cp.visible = False
    up_segs = [(p, q) for p, q in oct_edges if p.y > -1e-12 and q.y > -1e-12]
    # the lines that leave with the top pyramid (they stay drawn on it)
    top_cv = [cv for cv, (p, q) in zip(edge_cv, oct_edges)
              if (p - top).mag < 1e-9 or (q - top).mag < 1e-9] + [dots[2]]
    up_cv = [curve(pos=[p, q], radius=0.012, color=C_HL, visible=False) for p, q in up_segs]
    # the flaps: one triangle each, its corners moved along the fold itself
    # (an exact turn about the hinge), lit on the side that faces us. The
    # hinge is set a hair above the section, so that a folded flap lies just
    # above the square, not in it. (Which side faces us is judged from the
    # view direction the canvas was made with: canvas.forward read back on
    # the Python side is not reliable.)
    FL = 0.004
    C_FLAP = V(0.97, 0.84, 0.28)
    sec_line = curve(pos=[x + V(0, 0.003, 0) for x in sec + sec[:1]], radius=0.010,
                     color=C_HL * 0.8, visible=False)
    flap_tri, flap_cv = [], []
    for fl in flaps:
        vs = [vertex(pos=x, normal=V(0, 1, 0), color=C_FLAP) for x in fl[:3]]
        flap_tri.append((triangle(vs=vs, visible=False), vs))
        flap_cv.append(curve(pos=list(fl[:3]) + [fl[0]], radius=0.009, color=C_HL * 0.9,
                             visible=False))
    flaps_on = [False]
    flap_at = [0.0]

    def set_flaps(u):
        for (tri, vs), cv, fl in zip(flap_tri, flap_cv, flaps):
            pts = [fold(fl, x, u, FL) for x in fl[:3]]
            nr = (pts[1] - pts[0]).cross(pts[2] - pts[0]).norm()
            if nr.dot(FWD) > 0:                    # lit on the side we see
                nr = -nr
            for vx_, x in zip(vs, pts):
                vx_.pos = x
                vx_.normal = nr
            for i, x in enumerate(pts + pts[:1]):
                cv.modify(i, pos=x)
            tri.visible = cv.visible = flaps_on[0]
        flap_at[0] = u

    def show_flaps(flag):
        flaps_on[0] = flag
        set_flaps(flap_at[0] if flag else 0.0)
        sec_line.visible = flag

    def lift_top(y):
        cp_up.pos = g_up + V(0, y, 0)
        for cv, (p, q) in zip(up_cv, up_segs):
            cv.modify(0, pos=p + V(0, y, 0))
            cv.modify(1, pos=q + V(0, y, 0))

    # labels: the edge a; the height a/2 of the lifted pyramid; ½a² on the base
    dim_a = _Dim(V(-h, -h, h), V(h, -h, h), V(0, -0.07, 0.15), "a", lab_gap=2.6)
    xh = h + 0.32
    dim_h = _Dim(V(xh, LIFT, 0), V(xh, LIFT + h, 0), s_right * 0.1, "a/2",
                 visible=False, lab_gap=3.0)
    lab_B = _lab(V(0, 0.02, 0), "½a²", back=True, visible=False)

    row1, row2, row3 = _rows(sc, 3)
    _ready(sc)
    while True:
        # ---- a cube and the centres of its faces
        for cv, (p, q) in zip(edge_cv, oct_edges):
            cv.modify(1, pos=p)
            cv.visible = True
        cp_up.visible = cp_lo.visible = False
        _show(up_cv, False)
        show_flaps(False)
        _show(diags + dots, True)
        dim_a.show(True)
        dim_h.show(False)
        lab_B.visible = False
        row1.text = _pre("a cube of edge a, and the centres of its six faces")
        row2.text = row3.text = _pre(" ")
        _wait(2.2)
        # ---- join the centres of neighbouring faces: an octahedron
        row1.text = _pre("join the centres of neighbouring faces:  an octahedron")
        for t in _frames(1.6):
            for cv, (p, q) in zip(edge_cv, oct_edges):
                cv.modify(1, pos=p + (q - p) * _smooth(t))
        _show(diags, False)
        cp_up.pos, cp_lo.pos = g_up, g_lo
        cp_up.visible = cp_lo.visible = True
        for t in _frames(1.0):
            cp_up.opacity = cp_lo.opacity = 0.9 * t
        _wait(0.6)
        # ---- two square pyramids: lift the top one
        row1.text = _pre("its middle square cuts it into two square pyramids, apex at the "
                         "centres of the top and bottom faces")
        _show(up_cv, True)
        _show(top_cv, False)
        for t in _frames(1.6):
            lift_top(LIFT * _smooth(t))
        # ---- the base: the four corners of the mid-section fold onto it
        set_flaps(0)
        show_flaps(True)
        row1.text = _pre("the base joins the midpoints of the cube's mid-section, an a × a "
                         "square:  fold its four corners over")
        _wait(1.2)
        for t in _frames(2.2):
            set_flaps(_smooth(t))
        lab_B.visible = True
        row1.text = _pre("the four corners cover the base exactly:  it is half of a × a")
        row2.text = _pre("base  B = ½·a²")
        _wait(1.6)
        # ---- the height
        dim_h.show(True)
        row2.text = _pre("base  B = ½·a²,   height  a/2:   each pyramid  ⅓ · ½a² · a/2  =  a³/12")
        _wait(2.6)
        # ---- back together
        dim_h.show(False)
        lab_B.visible = False
        for t in _frames(1.4):
            set_flaps(1 - _smooth(t))
        show_flaps(False)
        for t in _frames(1.2):
            lift_top(LIFT * (1 - _smooth(t)))
        _show(top_cv, True)
        _show(up_cv, False)
        row1.text = _pre("the octahedron:  two such pyramids")
        row3.text = _pre(f"V  =  2 · a³/12  =  a³/6        (a = 1:  {1 / 6:.4f})")
        _wait(4.4)
        for t in _frames(0.8):
            cp_up.opacity = cp_lo.opacity = 0.9 * (1 - t)
        _wait(0.3)


# ============================================================== I26

def _menger(n):
    """The cells (i, j, k), 0 ≤ i, j, k < 3ⁿ, kept after n steps: in no
    base-3 digit place are two of i, j, k equal to 1."""
    N = 3 ** n
    out = set()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                good = True
                for d in range(n):
                    p = 3 ** d
                    if ((i // p) % 3 == 1) + ((j // p) % 3 == 1) + ((k // p) % 3 == 1) >= 2:
                        good = False
                        break
                if good:
                    out.add((i, j, k))
    return out


def _merge_boxes(cells):
    """The cells merged greedily into boxes (i0, i1, j0, j1, k0, k1): runs
    along x, widened along z, then stacked along y. Checked: the boxes
    cover every cell exactly once."""
    left = set(cells)
    out = []
    for c in sorted(cells, key=lambda c: (c[1], c[2], c[0])):
        if c not in left:
            continue
        i, j, k = c
        i1 = i
        while (i1 + 1, j, k) in left:
            i1 += 1
        k1 = k
        while all((x, j, k1 + 1) in left for x in range(i, i1 + 1)):
            k1 += 1
        j1 = j
        while all((x, j1 + 1, z) in left for x in range(i, i1 + 1) for z in range(k, k1 + 1)):
            j1 += 1
        for x in range(i, i1 + 1):
            for y in range(j, j1 + 1):
                for z in range(k, k1 + 1):
                    left.remove((x, y, z))
        out.append((i, i1, j, j1, k, k1))
    _check(not left and sum((b[1] - b[0] + 1) * (b[3] - b[2] + 1) * (b[5] - b[4] + 1)
                            for b in out) == len(cells), "the boxes cover the cells exactly")
    return out


def scene_I26():
    """V → 0: cut a cube into 27 equal cubes and take out the centre one
    and the six in the middles of the faces: 20 of the 27 remain, 20/27 of
    the volume. Do the same to each of the 20, and again to each of those:
    every step keeps 20/27 of what is there, so after n steps 20ⁿ of the
    27ⁿ small cubes are left, (20/27)ⁿ of the volume, which goes to 0. The
    first three steps are drawn exactly (20, 400 and 8000 cubes)."""
    sc = new_canvas("the Menger sponge has no volume:   V  =  (20/27)ⁿ  →  0",
                    "every step cuts each cube into 27 and keeps 20:  the centre and the six "
                    "face centres go",
                    rng=2.2, forward=V(-0.56, -0.46, -0.69), centre=V(0, 0, 0))
    sc.fov = 0.45
    S = 2.4                                       # the edge of the cube
    NST = 3                                       # steps drawn
    lo = V(-S / 2, -S / 2, -S / 2)
    MID = (1, 1, 1)
    FACES = [(2, 1, 1), (0, 1, 1), (1, 2, 1), (1, 0, 1), (1, 1, 2), (1, 1, 0)]
    NRM = [V(1, 0, 0), V(-1, 0, 0), V(0, 1, 0), V(0, -1, 0), V(0, 0, 1), V(0, 0, -1)]

    # ---- the steps, checked: each kept cube of step n−1 has exactly 27
    # parts at step n, of which the 20 kept are all but the centre and the
    # six face centres; so 20ⁿ cubes of (1/27)ⁿ each are left
    kept = [_menger(n) for n in range(NST + 1)]
    gone = [None]
    for n in range(1, NST + 1):
        _check(len(kept[n]) == 20 ** n, "20ⁿ cubes after n steps")
        by_type = {MID: [], **{f: [] for f in FACES}}
        for (i, j, k) in kept[n - 1]:
            for loc in itertools.product(range(3), repeat=3):
                c = (3 * i + loc[0], 3 * j + loc[1], 3 * k + loc[2])
                removed = sum(1 for x in loc if x == 1) >= 2
                _check((c in kept[n]) == (not removed), "the 20 kept are the 27 less 7")
                if removed:
                    by_type[loc].append(c)
        _check(len(by_type) == 7 and all(len(v) == 20 ** (n - 1) for v in by_type.values()),
               "seven taken out of every cube: the centre and the six face centres")
        for c in kept[n]:
            _check((c[0] // 3, c[1] // 3, c[2] // 3) in kept[n - 1], "each kept cube lies in "
                                                                     "a cube of the step before")
        _check(abs(len(kept[n]) / 27 ** n - (20 / 27) ** n) < 1e-15, "V = (20/27)ⁿ")
        gone.append(by_type)
    # the volumes left go down by 20/27 each step and drop below any bound
    vols = [(20 / 27) ** n for n in range(80)]
    _check(all(vols[n + 1] < vols[n] * 0.75 for n in range(79)) and vols[-1] < 1e-10,
           "(20/27)ⁿ → 0")

    # ---- drawing: a step's cubes as one compound of boxes (cells merged
    # into larger boxes); the cubes taken out, one compound per kind (each
    # kind leaves the same way). At the last step only the red cubes with a
    # face open to the outside are drawn: every other one lies between two
    # cubes that go as well, sealed inside the solid, so none of it shows.
    C_KEEP = V(0.30, 0.62, 0.78)
    C_GONE = C_E

    def blocks(cells, n, col):
        s = S / 3 ** n
        out = []
        for (i0, i1, j0, j1, k0, k1) in _merge_boxes(cells):
            a_ = V(i0, j0, k0) * s
            b_ = V(i1 + 1, j1 + 1, k1 + 1) * s
            out.append(box(pos=lo + (a_ + b_) / 2, size=b_ - a_, color=col))
        return out

    def group(cells, n, col):
        """[(object, home position)] drawing the cells of step n."""
        return [(compound(blocks(cells, n, col), origin=V(0, 0, 0), visible=False),
                 V(0, 0, 0))]

    kept_g = [group(kept[n], n, C_KEEP) for n in range(NST + 1)]
    gone_g = [None]
    for n in range(1, NST + 1):
        d = {}
        for loc in [MID] + FACES:
            cells = set(gone[n][loc])
            if n == NST:
                if loc == MID:
                    continue                      # sealed inside until its neighbours go
                # the parent's neighbour beyond this face: if it is kept, the
                # red cube is sealed between two red cubes
                off = (loc[0] - 1, loc[1] - 1, loc[2] - 1)
                open_ = {c for c in cells
                         if (c[0] // 3 + off[0], c[1] // 3 + off[1], c[2] // 3 + off[2])
                         not in kept[n - 1]}
                for c in cells - open_:
                    q = (c[0] + off[0], c[1] + off[1], c[2] + off[2])    # one cell over
                    _check(q in gone[n][(2 - loc[0], 2 - loc[1], 2 - loc[2])],
                           "a hidden red cube faces a red cube of the next cube")
                cells = open_
            d[loc] = group(cells, n, C_GONE) if cells else []
        gone_g.append(d)
    _check(sum(len(set(gone[NST][f])) for f in FACES) == 6 * 20 ** (NST - 1),
           "every cube of the step before loses six face centres")

    def put(grp, shift=V(0, 0, 0), op=None, vis=None):
        for ob, home in grp:
            ob.pos = home + shift
            if op is not None:
                ob.opacity = op
            if vis is not None:
                ob.visible = vis

    row1, row2, row3 = _rows(sc, 3)
    bar = wtext(text="")

    def show_bar(v):
        w = int(round(420 * v))
        bar.text = ("<div style='margin:4px 0 0 0;height:12px;width:420px;background:#2a2a33'>"
                    f"<div style='height:12px;width:{w}px;background:#4d9ec7'></div></div>")

    def vol_text(n):
        return (f"V{_sub(n)}  =  (20/27){_sup(n)}  =  {20 ** n}/{27 ** n}  =  "
                f"{(20 / 27) ** n:.4f}")

    TXT = {1: "cut it into 27 equal cubes;  take out the centre and the six face centres:  "
              "20 of 27 remain",
           2: "do the same to each of the 20:  20 · 20 = 400 of 27 · 27 = 729 remain",
           3: "and again to each of the 400:  8000 of 19683 remain"}
    _ready(sc)
    while True:
        for n in range(NST + 1):
            put(kept_g[n], vis=False)
        put(kept_g[0], vis=True)
        row1.text = _pre("a cube of volume 1")
        row2.text = _pre(vol_text(0))
        row3.text = _pre(" ")
        show_bar(1.0)
        _wait(1.8)
        for n in range(1, NST + 1):
            # ---- mark: the cubes of this step, the ones to go in red
            put(kept_g[n - 1], vis=False)
            put(kept_g[n], vis=True)
            for grp in gone_g[n].values():
                put(grp, op=1, vis=True)
            row1.text = _pre(TXT[n])
            _wait(1.3)
            # ---- the face centres come out, then the centre goes
            s = S / 3 ** n
            for t in _frames(1.4):
                u = _smooth(t)
                for loc, nv in zip(FACES, NRM):
                    put(gone_g[n][loc], nv * (s * 1.2 * u), max(1 - 1.15 * t, 0.0))
            for loc in FACES:
                put(gone_g[n][loc], vis=False)
            if MID in gone_g[n]:
                for t in _frames(0.7):
                    put(gone_g[n][MID], op=max(1 - t, 0.0))
                put(gone_g[n][MID], vis=False)
            row2.text = _pre(vol_text(n))
            show_bar((20 / 27) ** n)
            _wait(1.1)
        # ---- and so on: every step keeps 20/27 of what is left
        row1.text = _pre("every further step keeps 20/27 of what is left")
        for n in range(NST + 1, 31):
            row2.text = _pre(f"V{_sub(n)}  =  (20/27){_sup(n)}  =  {(20 / 27) ** n:.6f}")
            show_bar((20 / 27) ** n)
            _wait(0.12)
        row3.text = _pre("20/27 < 1,  so  (20/27)ⁿ → 0:  the sponge that is left in the limit "
                         "has no volume")
        _wait(4.0)


# ============================================================== I27

def _overlap(lo1, hi1, lo2, hi2, eps=1e-9):
    """Do two axis-aligned boxes (as vectors) share interior points?"""
    return (lo1.x < hi2.x - eps and lo2.x < hi1.x - eps and lo1.y < hi2.y - eps and
            lo2.y < hi1.y - eps and lo1.z < hi2.z - eps and lo2.z < hi1.z - eps)


def _trim(n, k, W, lo):
    """What slab k of the n-staircase (box W) loses when the slabs halve:
    the L of width u/2 round its upper half, as two boxes."""
    u = W / n
    y0, y1 = (n - k) * u + u / 2, (n - k + 1) * u
    m = (2 * k - 1) * u / 2
    return [(lo + V(m, y0, 0), lo + V(k * u, y1, k * u)),
            (lo + V(0, y0, m), lo + V(m, y1, k * u))]


def _trim_vol(n, k, W):
    u = W / n
    return (k * k - ((2 * k - 1) / 2) ** 2) * u * u * (u / 2)


def scene_I27():
    """(1² + 2² + … + n²)/n³ → ⅓: in a box n × n × n, the pyramid on the
    floor with its apex above a corner is a third of the box (three such
    pyramids with that apex fill it). A staircase of n square slabs of
    sides 1, 2, …, n, set into that corner, holds the pyramid, and with
    the outer row and column of every slab taken off (an L of 2k − 1
    cubes) what is left lies inside it. The L's rise straight up and fit
    together into one n × n layer: so the staircase exceeds the pyramid by
    at most 1/n of the box, ⅓ ≤ (1² + … + n²)/n³ ≤ ⅓ + 1/n. Thinner slabs
    squeeze it onto ⅓."""
    sc = new_canvas("the stepped pyramid:   (1² + 2² + … + n²) / n³  →  ⅓",
                    "a staircase of n square slabs holds the pyramid that is a third of "
                    "the box, and exceeds it by less than one layer",
                    rng=2.3, forward=V(-0.58, -0.55, -0.62), centre=V(0.05, 0.3, 0))
    sc.fov = 0.45
    W = 2.2                                       # the box
    N0 = 4
    NS = [4, 8, 16, 32]
    lo = V(-W / 2, -W / 2, -W / 2)
    A = lo + V(0, W, 0)                           # the apex, above the back corner

    def rel(p):
        return p - lo

    # ---- three pyramids with apex A on the three faces away from it fill
    # the box; each is a third
    sq = {"floor": [V(0, 0, 0), V(W, 0, 0), V(W, 0, W), V(0, 0, W)],
          "right": [V(W, 0, 0), V(W, W, 0), V(W, W, W), V(W, 0, W)],
          "front": [V(0, 0, W), V(W, 0, W), V(W, W, W), V(0, W, W)]}
    pyr3 = {}
    for key, base in sq.items():
        bw = [lo + p for p in base]
        pyr3[key] = [bw] + [[A, bw[i], bw[(i + 1) % 4]] for i in range(4)]
        _check(abs(_faces_volume(pyr3[key]) - W ** 3 / 3) < 1e-12, "each pyramid is a third")

    def in_floor_pyr(p):                          # the one on the floor
        q = rel(p)
        return 0 <= q.y <= W and 0 <= q.x <= W - q.y and 0 <= q.z <= W - q.y

    n_g = 18
    for i in range(n_g):
        for j in range(n_g):
            for k in range(n_g):
                q = V(W * (i + 0.5) / n_g, W * (j + 0.37) / n_g, W * (k + 0.61) / n_g)
                # floor: x, z ≤ W − y;  right: y, z ≤ x;  front: x, y ≤ z (apex at x = z = 0, y = W)
                yy = W - q.y
                cnt = ((q.x < yy and q.z < yy) + (yy < q.x and q.z < q.x) + (yy < q.z and q.x < q.z))
                _check(cnt == 1, "three pyramids fill the box")

    # ---- the staircase for n: slab k (k = 1 at the top) is k × k × 1 at
    # height n − k, in the back corner; its rim is the L of its last row
    # and column, 2k − 1 cubes
    def slab(n, k):
        u = W / n
        return lo + V(0, (n - k) * u, 0), lo + V(k * u, (n - k + 1) * u, k * u)

    def inner(n, k):
        u = W / n
        return lo + V(0, (n - k) * u, 0), lo + V((k - 1) * u, (n - k + 1) * u, (k - 1) * u)

    def rim(n, k):
        u = W / n
        y0, y1 = (n - k) * u, (n - k + 1) * u
        out = [(lo + V((k - 1) * u, y0, 0), lo + V(k * u, y1, k * u))]
        if k > 1:
            out.append((lo + V(0, y0, (k - 1) * u), lo + V((k - 1) * u, y1, k * u)))
        return out

    def vol(b):
        d = b[1] - b[0]
        return d.x * d.y * d.z

    def inside(p, b):
        return (b[0].x <= p.x <= b[1].x and b[0].y <= p.y <= b[1].y and
                b[0].z <= p.z <= b[1].z)

    n = N0
    u0 = W / n
    _check(abs(sum(vol(slab(n, k)) for k in range(1, n + 1)) -
               sum(k * k for k in range(1, n + 1)) * u0 ** 3) < 1e-12, "the staircase is Σk²")
    for k in range(1, n + 1):
        _check(abs(vol(slab(n, k)) - vol(inner(n, k)) - sum(vol(b) for b in rim(n, k)))
               < 1e-12 and abs(sum(vol(b) for b in rim(n, k)) - (2 * k - 1) * u0 ** 3) < 1e-12,
               "a slab is its inner part and an L of 2k − 1 cubes")
    # inner steps ⊂ pyramid ⊂ staircase (by sampling), for every n drawn;
    # a sample point lies at the height of exactly one slab, k = n − ⌊y/u⌋
    for nn in NS:
        un = W / nn
        for i in range(13):
            for j in range(13):
                for k in range(13):
                    p = lo + V(W * (i + 0.5) / 13, W * (j + 0.29) / 13, W * (k + 0.71) / 13)
                    kk = nn - int((p.y - lo.y) / un)
                    in_st = inside(p, slab(nn, kk))
                    in_in = kk > 1 and inside(p, inner(nn, kk))
                    _check(not in_floor_pyr(p) or in_st, "the staircase holds the pyramid")
                    _check(not in_in or in_floor_pyr(p), "the inner steps lie in the pyramid")
    # the rims rise straight up by k slabs each, side by side: they never
    # meet anything, and at the top they make one n × n layer on the box
    rims = [(k, b) for k in range(1, n + 1) for b in rim(n, k)]
    still = [inner(n, k) for k in range(2, n + 1)]
    for q in range(0, 41):
        f = q / 40
        moved = [(b[0] + V(0, k * u0 * f, 0), b[1] + V(0, k * u0 * f, 0)) for k, b in rims]
        for m1, m2 in itertools.combinations(moved, 2):
            _check(not _overlap(*m1, *m2), "the rims rise side by side")
        for m1 in moved:
            for s_ in still:
                _check(not _overlap(*m1, *s_), "the rims rise clear of the steps left")
    top = [(b[0] + V(0, k * u0, 0), b[1] + V(0, k * u0, 0)) for k, b in rims]
    _check(all(abs(b[0].y - (lo.y + W)) < 1e-12 and abs(b[1].y - (lo.y + W + u0)) < 1e-12
               for b in top) and abs(sum(vol(b) for b in top) - W * W * u0) < 1e-12,
           "on top of the box the rims make one layer, 1/n of the box")
    for i in range(12):
        for j in range(12):
            p = lo + V(W * (i + 0.5) / 12, W + u0 / 2, W * (j + 0.5) / 12)
            _check(sum(inside(p, b) for b in top) == 1, "the rims tile the layer exactly")
    # thinning n → 2n: every slab's upper half loses an L of width u/2
    for a_, b_ in zip(NS, NS[1:]):
        _check(b_ == 2 * a_, "the slabs halve")
        for k in range(1, a_ + 1):
            lo_h, hi_h = slab(b_, 2 * k), slab(b_, 2 * k - 1)
            old = slab(a_, k)
            _check(abs(vol(old) - vol(lo_h) - vol(hi_h) - _trim_vol(a_, k, W)) < 1e-12 and
                   inside(lo_h[0], old) and inside(lo_h[1], old) and inside(hi_h[0], old) and
                   inside(hi_h[1], old), "the thinner slabs lie in the thicker ones")

    # the slabs come down one at a time, the largest first, from DROP above
    # their places: none meets a slab already down
    DROP = 1.4
    down = []
    for k in range(n, 0, -1):
        for q in range(0, 21):
            b = slab(n, k)
            m1 = (b[0] + V(0, DROP * q / 20, 0), b[1] + V(0, DROP * q / 20, 0))
            for b2 in down:
                _check(not _overlap(*m1, *b2), "each slab comes down freely")
        down.append(slab(n, k))

    def frac(nn):
        s2 = sum(k * k for k in range(1, nn + 1))
        _check(s2 * 6 == nn * (nn + 1) * (2 * nn + 1) and
               1 / 3 <= s2 / nn ** 3 <= 1 / 3 + 1 / nn, "⅓ ≤ Σk²/n³ ≤ ⅓ + 1/n")
        return s2, s2 / nn ** 3

    for nn in NS:
        frac(nn)

    # ---- drawing
    for p_, q_ in itertools.combinations([lo + V(x, y, z) for x in (0, W) for y in (0, W)
                                          for z in (0, W)], 2):
        if abs((p_ - q_).mag - W) < 1e-9:
            curve(pos=[p_, q_], radius=0.009, color=C_HL * 0.55)
    # the floor one appears in place; the other two come in from outside,
    # each straight along the normal of its base: they never meet
    COLP = {"floor": C_D, "right": C_C, "front": C_F}
    OUT = {"floor": V(0, -1, 0), "right": V(1, 0, 0), "front": V(0, 0, 1)}
    DIN = 0.6
    placed = []
    for key in ("floor", "right", "front"):
        for q in range(1, 21):
            me = _poly_parts([[p_ + OUT[key] * (DIN * q / 20) for p_ in f] for f in pyr3[key]])
            for ot in placed:
                _check(_convex_apart(*me, *ot), "each pyramid comes in freely")
        placed.append(_poly_parts(pyr3[key]))
    pyr_cp, pyr_g, pyr_cv = {}, {}, {}
    for key, fs in pyr3.items():
        cp, g, _ = _mesh(fs, COLP[key])
        cp.visible = False
        pyr_cp[key], pyr_g[key] = cp, g
        pyr_cv[key] = [(curve(pos=[p_, q_], radius=0.011, color=C_HL * 0.9, visible=False),
                        p_, q_) for p_, q_ in _face_edges(fs)]

    def put_pyr(key, d):
        pyr_cp[key].pos = pyr_g[key] + OUT[key] * d
        for cv, p_, q_ in pyr_cv[key]:
            cv.modify(0, pos=p_ + OUT[key] * d)
            cv.modify(1, pos=q_ + OUT[key] * d)

    def show_pyr(key, flag):
        pyr_cp[key].visible = flag
        for cv, _, _ in pyr_cv[key]:
            cv.visible = flag

    def mkbox(b, col, vis=False):
        return box(pos=(b[0] + b[1]) / 2, size=b[1] - b[0], color=col, visible=vis)

    C_ST = [C_B, C_B * 0.78]                      # alternate slabs, so the steps show
    C_RIM = [C_C, V(0.97, 0.70, 0.25)]          # alternate rims, so the L's show
    inner_bx = {k: mkbox(inner(n, k), C_ST[k % 2]) for k in range(2, n + 1)}
    rim_bx = [(k, b, mkbox(b, C_ST[k % 2])) for k, b in rims]
    slabs_n = {nn: [mkbox(slab(nn, k), C_ST[k % 2]) for k in range(1, nn + 1)] for nn in NS[1:]}
    trims = {}
    for a_, b_ in zip(NS, NS[1:]):
        trims[b_] = [mkbox(t, C_E) for k in range(1, a_ + 1) for t in _trim(a_, k, W, lo)]
    # labels: the L's on the layer, 1, 3, 5, 7, each on the end of its arm
    # along the back edge
    lay_labs = [_lab(lo + V((k - 0.5) * u0, W + u0, 0.5 * u0), str(2 * k - 1),
                     back=True, visible=False) for k in range(1, n + 1)]

    def set_rims(f, col=None):
        for k, b, bx in rim_bx:
            bx.pos = (b[0] + b[1]) / 2 + V(0, k * u0 * f, 0)
            if col is not None:
                bx.color = C_RIM[k % 2] if col == "rim" else C_ST[k % 2]

    def staircase_n4(flag, op=1.0):
        for bx in list(inner_bx.values()) + [r[2] for r in rim_bx]:
            bx.visible = flag
            bx.opacity = op

    row1, row2, row3 = _rows(sc, 3)

    def sums(nn):
        s2, fr = frac(nn)
        terms = " + ".join(f"{k}²" for k in range(1, nn + 1)) if nn <= 4 else \
            f"1² + 2² + … + {nn}²"
        return (f"n = {nn}:   ({terms}) / {nn}³  =  {s2} / {nn ** 3}  =  {fr:.4f}")

    _ready(sc)
    while True:
        # ---- the box; three pyramids with a common apex fill it
        for key in pyr_cp:
            show_pyr(key, False)
            pyr_cp[key].opacity = 0.9
        staircase_n4(False)
        set_rims(0, "slab")
        for lst in list(slabs_n.values()) + list(trims.values()):
            _show(lst, False)
        _show(lay_labs, False)
        row1.text = _pre("three pyramids with their apex at one corner of the box fill it:  "
                         "each is ⅓ of the box")
        row2.text = row3.text = _pre(" ")
        put_pyr("floor", 0)
        show_pyr("floor", True)
        for t in _frames(0.6):
            pyr_cp["floor"].opacity = 0.9 * t
        for key in ("right", "front"):
            put_pyr(key, DIN)
            show_pyr(key, True)
            for t in _frames(0.75):
                put_pyr(key, DIN * (1 - _smooth(t)))
        _wait(1.0)
        for t in _frames(0.8):
            pyr_cp["right"].opacity = pyr_cp["front"].opacity = 0.9 * (1 - t)
            pyr_cp["floor"].opacity = 0.9 - 0.55 * t
        show_pyr("right", False)
        show_pyr("front", False)
        row1.text = _pre("the pyramid on the floor:  ⅓ of the box")
        _wait(0.6)
        # ---- a staircase of n slabs, sides 1 … n, in the same corner
        row1.text = _pre(f"a staircase of n square slabs, sides 1, 2, …, n,  in the same corner  "
                         f"(n = {n})")
        for k in range(n, 0, -1):
            parts = ([inner_bx[k]] if k > 1 else []) + [r[2] for r in rim_bx if r[0] == k]
            homes = [V(bx.pos.x, bx.pos.y, bx.pos.z) for bx in parts]   # copies
            for bx in parts:
                bx.visible = True
                bx.opacity = 0.6
            for t in _frames(0.55):
                for bx, hm in zip(parts, homes):
                    bx.pos = hm + V(0, DROP * (1 - _smooth(t)), 0)
        row2.text = _pre(sums(n))
        row3.text = _pre("it holds the pyramid")
        _wait(1.6)
        # ---- the rims: the last row and column of every slab
        staircase_n4(True, 1.0)
        set_rims(0, "rim")
        row1.text = _pre("take off the rim of every slab, its last row and column:  "
                         "an L of 2k − 1 cubes")
        row3.text = _pre(" ")
        _wait(0.6)
        for t in _frames(1.8):
            set_rims(_smooth(t))
        row3.text = _pre("what is left of the steps lies inside the pyramid")
        _show(lay_labs, True)
        row1.text = _pre("the rims rise straight up and fit together into one n × n layer:  "
                         "1 + 3 + 5 + 7 = 4²,  1/n of the box")
        row3.text = _pre("so   steps left  ≤  pyramid  ≤  staircase  =  steps left + 1/n:   "
                         "⅓  ≤  (1² + … + n²)/n³  ≤  ⅓ + 1/n")
        _wait(3.6)
        _show(lay_labs, False)
        for t in _frames(1.2):
            set_rims(1 - _smooth(t))
        set_rims(0, "slab")
        _wait(0.3)
        # ---- thinner slabs: each slab's upper half loses an L of half the width
        prev = None
        for nn in NS[1:]:
            row1.text = _pre(f"thinner slabs:  n = {nn}")
            if prev is None:
                staircase_n4(False)
            else:
                _show(slabs_n[prev], False)
            _show(slabs_n[nn] + trims[nn], True)
            for bx in trims[nn]:
                bx.opacity = 1
            _wait(0.7)
            for t in _frames(0.9):
                for bx in trims[nn]:
                    bx.opacity = 1 - t
            _show(trims[nn], False)
            row2.text = _pre(sums(nn))
            s2, fr = frac(nn)
            row3.text = _pre(f"⅓ = 0.3333  ≤  {fr:.4f}  ≤  ⅓ + 1/{nn} = {1 / 3 + 1 / nn:.4f}")
            _wait(1.3)
            prev = nn
        row1.text = _pre("as the slabs thin out, the staircase closes in on the pyramid:   "
                         "(1² + 2² + … + n²)/n³  →  ⅓")
        _wait(4.0)
        _show(slabs_n[prev], False)
