# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2g.py — proofs without words, 2D (manim), triangle centres
# and triangle areas:
#     F21 perpendicular bisectors meet      F22 angle bisectors meet
#     F23 the Euler line                    F24 the nine-point circle
#     F31 the Fermat point                  F56 Heron's formula
#     F18 medians cut six equal areas       F38 the midpoint triangle
#     F32 the one-seventh triangle
#
# Every move is rigid (shift / Rotate / a fold, which is a half-turn in
# space about a line of the plane) unless it is an announced enlargement
# (a homothety, drawn as a turn-and-scale about its centre) or a shear
# (apex sliding along a parallel to its fixed base). Every landing, area,
# angle and tiling claim is checked numerically with check(...) before or
# right after it is drawn, and every label is checked against the lines it
# must clear, so a wrong construction fails the render instead of drawing
# a wrong picture.


# ------------------------------------------------------------ private helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _angle(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _dir(p, q):
    """Direction angle of the ray p -> q."""
    d = to3(q) - to3(p)
    return float(np.arctan2(d[1], d[0]))


def _polar(a):
    return np.array([np.cos(a), np.sin(a), 0.0])


def _foot(p, a, b):
    """Foot of the perpendicular from p onto the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = b - a
    return a + np.dot(p - a, d) / np.dot(d, d) * d


def _meet(p1, p2, q1, q2):
    """Intersection of the lines p1p2 and q1q2."""
    p1, p2, q1, q2 = to3(p1), to3(p2), to3(q1), to3(q2)
    m = np.array([(p2 - p1)[:2], (q1 - q2)[:2]]).T
    t, _ = np.linalg.solve(m, (q1 - p1)[:2])
    return p1 + t * (p2 - p1)


def _rot2(p, c, th):
    """Point p turned by th about c (in the plane)."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


def _mirror(p, a, b):
    """Mirror image of p in the line ab."""
    f = _foot(p, a, b)
    return 2 * f - to3(p)


def _on_line(p, a, b, tol=1e-9):
    return abs(_cross(to3(b) - to3(a), to3(p) - to3(a))) < tol * max(
        1.0, float(np.linalg.norm(to3(b) - to3(a))))


def _circumcentre(A, B, C):
    return _meet((A + B) / 2, (A + B) / 2 + _perp(B - A),
                 (B + C) / 2, (B + C) / 2 + _perp(C - B))


def _perp(v):
    v = to3(v)
    return np.array([-v[1], v[0], 0.0])


def _orthocentre(A, B, C):
    return _meet(A, A + _perp(C - B), B, B + _perp(A - C))


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ticks(p, q, n=1, color=WHITE, size=0.13, width=2.5, at=0.5):
    """n short tick marks across segment pq (equal-length marks)."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    g = VGroup()
    for k in range(n):
        o = c + d * 0.09 * (k - (n - 1) / 2)
        g.add(Line(o - nrm * size, o + nrm * size, color=color,
                   stroke_width=width))
    return g


def _chevron(p, q, color=YELLOW_B, size=0.17, width=3, n=1):
    """Parallel-line mark: n small '>' at the middle of segment pq."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = (p + q) / 2
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7,
                                          tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


def _arcs(v, p, q, r, n=1, color=YELLOW_B, width=4, gap=0.08):
    """n concentric arcs for the non-reflex angle p-v-q (equal-angle marks)."""
    return VGroup(*[angle_arc(v, p, q, r + k * gap, color, width)
                    for k in range(n)])


def _wedge(vertex, p, q, r, color, op=FILL, stroke=1.5, stroke_color=WHITE):
    """Filled sector for the NON-REFLEX angle p-vertex-q (cf. angle_arc)."""
    v, p, q = to3(vertex), to3(p), to3(q)
    a1 = float(np.arctan2(*(p - v)[1::-1]))
    a2 = float(np.arctan2(*(q - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    s = Sector(radius=r, start_angle=a1, angle=span, arc_center=v,
               color=color, fill_opacity=op)
    s.set_stroke(stroke_color, stroke)
    return s


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


def _homothety(mob, centre, factor, turn=0.0, **kw):
    """Enlargement about `centre`, drawn as a turn by `turn` together with
    a scaling that reaches `factor` (geometrically in time). With turn = π
    and factor = ½ the end state is the homothety of ratio −½."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=c)
                 .scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _glide(mob, pivot, target, angle, **kw):
    """Rigid on every frame: the piece turns by `angle` about its pivot
    while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _pip(p, poly):
    """Point strictly inside polygon (ray casting)."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def _tiles_exactly(polys, region, n=110):
    """True if the polygons tile `region` with no gap and no overlap: on a
    jittered grid over the region's bounding box every point inside the
    region lies in exactly one polygon and every point outside in none."""
    xs = [float(p[0]) for p in region]
    ys = [float(p[1]) for p in region]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for i in range(n):
        for j in range(n):
            x = x0 + (x1 - x0) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = y0 + (y1 - y0) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            k = sum(_pip((x, y), P) for P in polys)
            if k != (1 if _pip((x, y), region) else 0):
                return False
    return True


def _disjoint(polys, n=160):
    """True if no sample point lies in two of the polygons."""
    pts = np.array([to3(p) for P in polys for p in P])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for i in range(n):
        for j in range(n):
            x = lo[0] + (hi[0] - lo[0]) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = lo[1] + (hi[1] - lo[1]) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            if sum(_pip((x, y), P) for P in polys) > 1:
                return False
    return True


def _same_poly(P, Q, tol=1e-9):
    """Same vertices in the same cyclic order (any start, either sense)."""
    P = [to3(p)[:2] for p in P]
    Q = [to3(q)[:2] for q in Q]
    if len(P) != len(Q):
        return False
    n = len(P)
    for d in (1, -1):
        for s in range(n):
            if all(close(P[i], Q[(s + d * i) % n], tol) for i in range(n)):
                return True
    return False


def _landed(mob, pts, tol=1e-6):
    """The polygon mob's vertices are pts (same order)."""
    v = mob.get_vertices()
    return len(v) == len(pts) and all(close(a, to3(b), tol)
                                      for a, b in zip(v, pts))


def _icon(poly, k, color, op=FILL):
    """A small copy of a piece, scaled by k (all icons share k, so their
    sizes compare as the pieces' areas do)."""
    return mk([to3(p) for p in poly], color, op, stroke_width=1.5).scale(k)


def _free_dir(X, towards):
    """Unit vector at X along the middle of the widest angular gap between
    the directions to the points `towards` (to park a label)."""
    X = to3(X)
    angs = sorted(_dir(X, q) % TAU for q in towards)
    best, mid = -1.0, 0.0
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        if a1 - a0 > best:
            best, mid = a1 - a0, (a0 + a1) / 2
    return _polar(mid)


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on
    the side the vector `side` points to, with its box clear of the line
    by `gap` whatever the slope."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


_BASE_CHARS = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789+=")


def _baseline(t):
    """Baseline of a Text: the median bottom of its non-descending glyphs
    (Text drops spaces from both .text and its glyphs, so they align)."""
    ys = [g.get_bottom()[1] for ch, g in zip(t.text, t.submobjects)
          if ch in _BASE_CHARS]
    if not ys:
        ys = [g.get_bottom()[1] for g in t.submobjects]
    return float(np.median(ys))


def _subline(parts, size=26, color=WHITE, sub_scale=0.66, drop=0.3):
    """A line of text with true subscripts, without TeX and without relying
    on subscript-letter glyphs (absent from some fonts). parts: list of
    (string, is_subscript); a subscript sits `drop` x-heights below the
    common baseline, smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for s_, sub in parts:
        lead = len(s_) - len(s_.lstrip(" "))
        trail = len(s_) - len(s_.rstrip(" "))
        x += lead * sp
        t = Text(s_.strip(" "), font_size=size * (sub_scale if sub else 1.0),
                 color=color)
        y0 = -drop * xh if sub else 0.0
        t.shift(np.array([x - t.get_left()[0], y0 - _baseline(t), 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sub else 0.02)
        g.add(t)
    return g


# ---- label hygiene: every label is checked against the lines it must clear

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _segs(*pts, closed=False):
    """Segments of the polyline through pts (closed: back to the start)."""
    pts = [to3(p) for p in pts]
    out = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if closed:
        out.append((pts[-1], pts[0]))
    return out


def _circle_segs(c, r, n=120):
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(0.0, TAU, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by
    pad), or None."""
    x0, x1, y0, y1 = _box(m, pad)
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.02) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _overlap(a, b, gap=0.05):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _in_frame(m, tol=0.02):
    return (m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
            and m.get_top()[1] <= SAFE_TOP + tol
            and m.get_bottom()[1] >= SAFE_BOTTOM - tol)


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every line in segs and of
    every other label."""
    for m in labels:
        check(_in_frame(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{_name(m)}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _in_regions(m, regions, pad=0.04):
    """True if a corner, an edge midpoint or the centre of m's box (grown
    by pad) lies inside one of the polygons."""
    x0, x1, y0, y1 = _box(m, pad)
    probes = [(x, y) for x in (x0, (x0 + x1) / 2, x1) for y in (y0, (y0 + y1) / 2, y1)]
    return any(_pip(q, poly) for poly in regions for q in probes)


def _park(m, X, towards, segs, d0=0.3, d1=1.2, pad=0.07, avoid=(), sep=0.05,
          regions=()):
    """Move label m next to the point X: along the middles of the angular
    gaps between the directions to `towards` (widest gap first), at the
    smallest distance where m clears every segment in segs, stays `sep`
    away from every mobject in `avoid` and outside every polygon in
    `regions`. Fails the render if no such spot exists."""
    X = to3(X)
    angs = sorted(_dir(X, q) % TAU for q in towards)
    gaps = []
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        gaps.append((a1 - a0, (a0 + a1) / 2))
    gaps.sort(reverse=True)
    found = []
    for w, mid in gaps:
        for d in np.arange(d0, d1 + 1e-9, 0.03):
            m.move_to(X + d * _polar(mid))
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)
                    and not _in_regions(m, regions)):
                found.append((d, -w, mid))
                break
    check(bool(found), f"a free spot for the label '{_name(m)}'")
    d, _, mid = min(found)                       # the nearest free spot
    return m.move_to(X + d * _polar(mid))


def _park_beside(m, choices, segs, pad=0.06, avoid=(), prefer=None, only=False,
                 ats=(0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82)):
    """Park label m beside one of the segments in `choices` (tried in
    order, several positions along it; the side `prefer` points to first,
    or only that side if `only`) where it clears every segment in segs and
    every mobject in `avoid`."""
    for p, q in choices:
        p, q = to3(p), to3(q)
        nrm = _perp(_unit(q - p))
        if prefer is not None and np.dot(nrm, to3(prefer)) < 0:
            nrm = -nrm
        for at in ats:
            for sd in ((nrm,) if only else (nrm, -nrm)):
                for gap in (0.08, 0.13, 0.18, 0.24):
                    _beside(m, p, q, sd, gap, at)
                    if (_hit(m, segs, pad) is None and _in_frame(m)
                            and not any(_overlap(m, o, 0.05) for o in avoid)):
                        return m
    check(False, f"a free spot for the label '{_name(m)}'")


def _park_dirs(m, X, dirs_deg, segs, d0=0.25, d1=0.9, pad=0.06, avoid=(), sep=0.08):
    """Put label m next to X along the first of the given directions
    (degrees) where, at the least distance in [d0, d1], it clears every
    segment in segs and stays `sep` away from the mobjects in `avoid`."""
    X = to3(X)
    for a in dirs_deg:
        for d in np.arange(d0, d1 + 1e-9, 0.02):
            m.move_to(X + d * _polar(a * DEGREES))
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"a free spot for the label '{_name(m)}'")


def _angle_label(m, V, P, Q, r0, pad=0.06, r1=3.0, segs=(),
                 fracs=(0.5,), avoid=(), sep=0.05):
    """Put label m inside the non-reflex angle P-V-Q -- on its bisector, or
    at the other fractions of the angle given in `fracs`, tried in order --
    at the least distance >= r0 from V where it clears both arms, segs and
    the mobjects in `avoid`."""
    V = to3(V)
    a1 = float(np.arctan2(*(to3(P) - V)[1::-1]))
    a2 = float(np.arctan2(*(to3(Q) - V)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    arms = [_seg(V, P), _seg(V, Q)] + list(segs)
    for f in fracs:
        u = _polar(a1 + f * span)
        for d in np.arange(r0, r1, 0.03):
            m.move_to(V + d * u)
            if (_hit(m, arms, pad) is None
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"room for the angle label '{_name(m)}'")


def _all_in_frame(mobs, what):
    for i, m in enumerate(mobs):
        check(_in_frame(m), f"{what}: item {i} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")


def _scalene_acute(A, B, C):
    angs = sorted((_angle(A, B, C), _angle(B, C, A), _angle(C, A, B)))
    return angs[2] < PI / 2 - 0.05 and angs[1] - angs[0] > 0.05 and angs[2] - angs[1] > 0.05


def _tri_from_angles(Bdeg, Cdeg, a=1.0):
    """Math triangle with B = (0, 0), C = (a, 0) and the given angles at B
    and C (A above BC)."""
    Bn, Cn = Bdeg * DEGREES, Cdeg * DEGREES
    c = a * np.sin(Cn) / np.sin(PI - Bn - Cn)
    return (np.array([c * np.cos(Bn), c * np.sin(Bn)]), np.array([0.0, 0.0]),
            np.array([a, 0.0]))


# ============================================ F21 PERPENDICULAR BISECTORS MEET

class F21_PerpendicularBisectors(Board):
    """Fold the plane over the perpendicular bisector of AB: the fold fixes
    every point of the bisector and carries A onto B, so every point of it
    is as far from A as from B. Likewise for BC. Where the two bisectors
    cross, O is as far from A as from B and from B as from C, so OA = OC.
    Then the triangles O M A and O M C (M the midpoint of CA) have three
    equal sides: folded over OM one lands on the other, so the two angles
    at M are equal, hence right, and OM is the third bisector. The circle
    about O through A passes through B and C."""

    def construct(self):
        # angles 78°, 42°, 60°: vertex Z lies R·|sin(X − Y)| from the
        # bisector of XY, so no bisector passes near a vertex
        Am, Bm, Cm = _tri_from_angles(78, 42)
        Om = _circumcentre(to3(Am), to3(Bm), to3(Cm))
        Rm = float(np.linalg.norm(to3(Am) - Om))
        m = 0.13
        F = Frame(Om[0] - Rm - m, Om[0] + Rm + m, Om[1] - Rm - m, Om[1] + Rm + m)
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        O = F.P(Om)
        R = Rm * F.k
        n = np.linalg.norm
        Mc, Ma, Mb = (A + B) / 2, (B + C) / 2, (C + A) / 2

        def diam(M):
            u = _unit(M - O)
            return O + R * u, O - R * u          # the bisector is a diameter

        Lc1, Lc2 = diam(Mc)
        La1, La2 = diam(Ma)
        Lb1, Lb2 = diam(Mb)

        check(_scalene_acute(A, B, C), "scalene acute triangle (O inside)")
        for Z, X, Y in ((C, A, B), (A, B, C), (B, C, A)):
            Mz = (X + Y) / 2
            check(abs(np.dot(Z - Mz, _unit(Y - X))) > 0.25 * R,
                  "no vertex close to the opposite bisector")
        check(abs(np.dot(Lc2 - Lc1, B - A)) < 1e-9 and _on_line(Mc, Lc1, Lc2),
              "the first line is the perpendicular bisector of AB")
        check(abs(np.dot(La2 - La1, C - B)) < 1e-9 and _on_line(Ma, La1, La2),
              "the second line is the perpendicular bisector of BC")
        check(close(_mirror(A, Lc1, Lc2), B) and close(_mirror(B, La1, La2), C),
              "the folds carry A onto B and B onto C")
        check(close(_meet(Lc1, Lc2, La1, La2), O), "the two bisectors cross at O")
        check(abs(n(O - A) - n(O - B)) < 1e-9 and abs(n(O - B) - n(O - C)) < 1e-9,
              "OA = OB = OC")
        check(close(_mirror(A, O, Mb), C), "folding OMA over OM lands on OMC")
        check(abs(np.dot(Mb - O, C - A)) < 1e-9, "OM is perpendicular to CA")

        cC, cA, cB = BLUE_B, TEAL_B, ORANGE
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)

        def vlab(s, P):
            return tag(s, 30).move_to(P + 0.4 * _unit(P - O))

        lA, lB, lC = vlab("A", A), vlab("B", B), vlab("C", C)
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        def bisector(M, L1, L2, P, Q, col, nt):
            """Midpoint ticks, the perpendicular line, and the fold over it
            that carries P onto Q."""
            dot = Dot(M, radius=0.06, color=col)
            tk = VGroup(_ticks(P, M, nt), _ticks(M, Q, nt))
            self.play(FadeIn(dot), FadeIn(tk), run_time=0.5)
            ln = Line(L1, L2, color=col, stroke_width=3.5)
            ra = _ra(M, L1 if np.dot(L1 - M, O - M) < 0 else L2, Q, 0.18, col)
            self.play(Create(ln), Create(ra), run_time=0.8)
            half = mk([L1, L2, P], col, op=0.35, stroke_color=col, stroke_width=2)
            self.play(FadeIn(half), run_time=0.3)
            self.play(_fold(half, L1, L2), run_time=1.4)
            check(_landed(half, [L1, L2, Q]), "the fold landed on the other side")
            self.play(FadeOut(half), run_time=0.35)
            return VGroup(dot, tk, ln, ra), ln

        _, lnc = bisector(Mc, Lc1, Lc2, A, B, cC, 1)
        _, lna = bisector(Ma, La1, La2, B, C, cA, 2)

        # O: on both bisectors -- OA = OB and OB = OC
        dO = Dot(O, radius=0.08, color=YELLOW_B)
        rays = [A, B, C, Lc1, Lc2, La1, La2, Lb1, Lb2]
        segs = (_segs(A, B, C, closed=True) + _circle_segs(O, R)
                + _segs(Lc1, Lc2) + _segs(La1, La2) + _segs(Lb1, Lb2)
                + _segs(O, A) + _segs(O, B) + _segs(O, C))
        lO = _park(tag("O", 30, YELLOW_B), O, rays, segs)
        sA = Line(O, A, color=YELLOW_B, stroke_width=3)
        sB = Line(O, B, color=YELLOW_B, stroke_width=3)
        sC = Line(O, C, color=YELLOW_B, stroke_width=3)
        tA, tB, tC = (_ticks(O, X, 1, YELLOW_B, at=0.62) for X in (A, B, C))
        self.play(FadeIn(dO), FadeIn(lO), run_time=0.4)
        self.play(Create(sA), Create(sB), lnc.animate.set_stroke(width=6),
                  run_time=0.8)
        self.play(FadeIn(tA), FadeIn(tB), lnc.animate.set_stroke(width=3.5),
                  run_time=0.5)
        self.play(Create(sC), lna.animate.set_stroke(width=6), run_time=0.8)
        self.play(FadeIn(tC), lna.animate.set_stroke(width=3.5), run_time=0.5)
        self.hold(0.3)

        # the third side: OMA and OMC have three equal sides -- fold
        dMb = Dot(Mb, radius=0.06, color=cB)
        tkb = VGroup(_ticks(C, Mb, 3), _ticks(Mb, A, 3))
        om = Line(O, Mb, color=cB, stroke_width=3.5)
        self.play(FadeIn(dMb), FadeIn(tkb), Create(om), run_time=0.8)
        half = mk([O, Mb, A], cB, op=0.4, stroke_color=cB, stroke_width=2)
        self.play(FadeIn(half), run_time=0.3)
        self.play(_fold(half, O, Mb), run_time=1.4)
        check(_landed(half, [O, Mb, C]), "OMA folded lands on OMC")
        rab = _ra(Mb, O, C, 0.18, cB)
        lnb = Line(Lb1, Lb2, color=cB, stroke_width=3.5)
        self.play(FadeOut(half), Create(rab), run_time=0.5)
        self.play(Create(lnb), FadeOut(om), run_time=0.8)

        # the circle about O through A, B and C
        circ = Circle(radius=R, color=YELLOW_B, stroke_width=3).move_to(O)
        self.play(Create(circ), run_time=1.2)

        labels = [lA, lB, lC, lO]
        _labels_ok(labels, segs, "F21")
        _all_in_frame([tri, circ], "F21")
        self.play(Write(caption("the perpendicular bisectors meet in one point O:"
                                "   OA = OB = OC", 30)))
        self.hold(2.2)


# ============================================ F22 ANGLE BISECTORS MEET

class F22_AngleBisectors(Board):
    """Fold the angle at A over its bisector: AB falls along AC, so the
    perpendicular from a point of the bisector to AB lands on the one to
    AC -- every point of the bisector is as far from AB as from AC.
    Likewise at B. Where the two bisectors cross, I is as far from AB as
    from AC and from BC: three equal distances r. The right triangles CDI
    and CEI (D, E the feet on BC, CA) share the hypotenuse CI and have
    equal legs r: folded over CI one lands on the other, so CI halves the
    angle at C. The circle about I of radius r touches all three sides."""

    def construct(self):
        # the bisector from a vertex and the perpendicular from I to the
        # opposite side make half the difference of the other two angles:
        # angles 78°, 42°, 60° keep every such pair at least 9° apart
        Am, Bm, Cm = _tri_from_angles(78, 42)
        pts = np.array([Am, Bm, Cm])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        F = Frame(lo[0] - 0.07, hi[0] + 0.07, lo[1] - 0.1, hi[1] + 0.1)
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        n = np.linalg.norm
        a, b, c = n(B - C), n(C - A), n(A - B)
        La = B + (C - B) * c / (b + c)          # the bisectors' feet
        Lb = C + (A - C) * a / (a + c)
        Lc = A + (B - A) * b / (a + b)
        I = _meet(A, La, B, Lb)
        D, E, Fo = _foot(I, B, C), _foot(I, C, A), _foot(I, A, B)
        r = n(I - D)
        Pa_c, Pa_b = _foot(La, A, B), _foot(La, A, C)    # feet of La
        Pb_a, Pb_c = _foot(Lb, B, A), _foot(Lb, B, C)    # feet of Lb

        check(_scalene_acute(A, B, C), "scalene acute triangle")
        check(abs(_angle(A, B, La) - _angle(A, La, C)) < 1e-9
              and abs(_angle(B, A, Lb) - _angle(B, Lb, C)) < 1e-9
              and abs(_angle(C, A, Lc) - _angle(C, Lc, B)) < 1e-9,
              "AL, BL, CL halve the angles")
        check(close(_mirror(Pa_c, A, La), Pa_b) and close(_mirror(Pb_a, B, Lb), Pb_c),
              "the folds carry the perpendicular to one side onto the other")
        check(abs(n(I - Fo) - r) < 1e-9 and abs(n(I - E) - r) < 1e-9,
              "ID = IE = IF")
        check(close(_mirror(D, C, I), E), "folding CDI over CI lands on CEI")
        check(_on_line(I, C, Lc), "CI is the bisector at C")
        for X, P1, P2 in ((D, B, C), (E, C, A), (Fo, A, B), (Pa_c, A, B),
                          (Pa_b, A, C), (Pb_a, B, A), (Pb_c, B, C)):
            check(0 < np.dot(X - P1, P2 - P1) < n(P2 - P1) ** 2,
                  "the feet lie on the sides")

        cA, cB, cC = BLUE_B, TEAL_B, ORANGE
        G = (A + B + C) / 3
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)

        def vlab(s, P):
            return tag(s, 30).move_to(P + 0.4 * _unit(P - G))

        lA, lB, lC = vlab("A", A), vlab("B", B), vlab("C", C)
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        def bisect(V, L, P1, P2, Q1, Q2, col, nm, r_arc):
            """The bisector VL with its two equal half-angles, then the fold
            that carries L's perpendicular to VP1 onto the one to VP2."""
            ln = Line(V, L, color=col, stroke_width=3.5)
            a1 = angle_arc(V, P1, L, r_arc, col, 4)
            a2 = angle_arc(V, L, P2, r_arc + 0.08, col, 4)
            g1 = _angle_label(tag(nm, 24, col), V, P1, L, r_arc + 0.3)
            g2 = _angle_label(tag(nm, 24, col), V, L, P2, r_arc + 0.3)
            self.play(Create(ln), Create(a1), Create(a2), FadeIn(g1), FadeIn(g2),
                      run_time=0.9)
            p1 = Line(L, Q1, color=WHITE, stroke_width=2.5)
            p2 = Line(L, Q2, color=WHITE, stroke_width=2.5)
            m1, m2 = _ra(Q1, L, V, 0.16), _ra(Q2, L, V, 0.16)
            self.play(Create(p1), Create(m1), run_time=0.5)
            half = mk([V, L, Q1], col, op=0.4, stroke_color=col, stroke_width=2)
            self.play(FadeIn(half), run_time=0.3)
            self.play(_fold(half, V, L), run_time=1.3)
            check(_landed(half, [V, L, Q2]), "the fold landed on the other side")
            tk = VGroup(_ticks(L, Q1, 1, col), _ticks(L, Q2, 1, col))
            self.play(FadeIn(p2), FadeIn(m2), FadeOut(half), FadeIn(tk),
                      run_time=0.5)
            return VGroup(ln, a1, a2), VGroup(g1, g2), VGroup(p1, p2, m1, m2, tk)

        bA, gA, xA = bisect(A, La, B, C, Pa_c, Pa_b, cA, "α", 0.62)
        self.hold(0.2)
        self.play(FadeOut(xA), run_time=0.35)
        bB, gB, xB = bisect(B, Lb, C, A, Pb_c, Pb_a, cB, "β", 0.55)
        self.hold(0.2)
        self.play(FadeOut(xB), run_time=0.35)

        # I: on both bisectors -- three equal distances
        # I: on both bisectors (from here on each is drawn from its vertex
        # to I only) -- three equal distances
        dI = Dot(I, radius=0.08, color=YELLOW_B)
        rays = [A, B, C, D, E, Fo]
        segs = (_segs(A, B, C, closed=True) + _circle_segs(I, r)
                + _segs(A, I) + _segs(B, I) + _segs(C, I)
                + _segs(I, D) + _segs(I, E) + _segs(I, Fo))
        lI = _park(tag("I", 30, YELLOW_B), I, rays, segs, d0=0.25)
        sD = Line(I, D, color=YELLOW_B, stroke_width=3)
        sE = Line(I, E, color=YELLOW_B, stroke_width=3)
        sF = Line(I, Fo, color=YELLOW_B, stroke_width=3)
        mD, mE, mF = _ra(D, I, C, 0.16), _ra(E, I, A, 0.16), _ra(Fo, I, B, 0.16)
        tD, tE, tF = (_ticks(I, X, 1, YELLOW_B) for X in (D, E, Fo))
        self.play(FadeIn(dI), FadeIn(lI), run_time=0.4)
        self.play(bA[0].animate.put_start_and_end_on(A, I),
                  bB[0].animate.put_start_and_end_on(B, I), run_time=0.6)
        self.play(Create(sF), Create(sE), Create(mF), Create(mE),
                  bA[0].animate.set_stroke(width=6), run_time=0.8)
        self.play(FadeIn(tF), FadeIn(tE), bA[0].animate.set_stroke(width=3.5),
                  run_time=0.5)
        self.play(Create(sD), Create(mD), bB[0].animate.set_stroke(width=6),
                  run_time=0.8)
        self.play(FadeIn(tD), bB[0].animate.set_stroke(width=3.5), run_time=0.5)
        self.hold(0.3)

        # the third corner: CDI and CEI share CI and have equal legs -- fold
        ci = Line(C, I, color=cC, stroke_width=3.5)
        self.play(Create(ci), run_time=0.6)
        half = mk([C, D, I], cC, op=0.4, stroke_color=cC, stroke_width=2)
        self.play(FadeIn(half), run_time=0.3)
        self.play(_fold(half, C, I), run_time=1.3)
        check(_landed(half, [C, E, I]), "CDI folded lands on CEI")
        r_c = 0.75
        c1 = angle_arc(C, B, I, r_c, cC, 4)
        c2 = angle_arc(C, I, A, r_c + 0.08, cC, 4)
        g1 = _angle_label(tag("γ", 24, cC), C, B, I, r_c + 0.3, segs=segs)
        g2 = _angle_label(tag("γ", 24, cC), C, I, A, r_c + 0.3, segs=segs)
        self.play(FadeOut(half), Create(c1), Create(c2), FadeIn(g1), FadeIn(g2),
                  run_time=0.6)
        self.play(ci.animate.set_stroke(width=5), run_time=0.4)
        self.play(ci.animate.set_stroke(width=3.5), run_time=0.4)

        # the circle about I of radius r touches the three sides
        circ = Circle(radius=r, color=YELLOW_B, stroke_width=3).move_to(I)
        lr = _park_beside(tag("r", 26, YELLOW_B), [(I, Fo), (I, E), (I, D)], segs,
                          pad=0.1, avoid=[lA, lB, lC, lI, *gA, *gB, g1, g2,
                                          tD, tE, tF, mD, mE, mF])
        self.play(Create(circ), FadeIn(lr), run_time=1.1)

        labels = [lA, lB, lC, lI, lr, *gA, *gB, g1, g2]
        _labels_ok(labels, segs, "F22")
        self.play(Write(caption("the angle bisectors meet in one point I:"
                                "   d(I, AB) = d(I, BC) = d(I, CA)", 30)))
        self.hold(2.2)


# ============================================ F23 THE EULER LINE

class F23_EulerLine(Board):
    """G divides each median 2 : 1, so the half-turn about G combined with
    halving (the homothety with centre G and ratio −½) carries A, B, C to
    the midpoints A', B', C' of the opposite sides. It keeps directions of
    lines (only reverses them), so the altitude from A -- through A,
    perpendicular to BC -- goes to the line through A' perpendicular to
    B'C' ∥ BC: the perpendicular bisector of BC. So the three altitudes go
    to the three perpendicular bisectors, and their common point H goes to
    the common point O of those. A point, its image and the centre are on
    one line, and the image is half as far from G: H, G, O are collinear
    and HG = 2·GO."""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(77, 37)
        pts = np.array([Am, Bm, Cm])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        F = Frame(lo[0] - 0.07, hi[0] + 0.07, lo[1] - 0.1, hi[1] + 0.08)
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        n = np.linalg.norm
        G = (A + B + C) / 3
        H, O = _orthocentre(A, B, C), _circumcentre(A, B, C)
        Ap, Bp, Cp = (B + C) / 2, (C + A) / 2, (A + B) / 2
        D, E, Fo = _foot(A, B, C), _foot(B, C, A), _foot(C, A, B)

        def hom(X):
            return G - 0.5 * (to3(X) - G)

        Ds, Es, Fs = hom(D), hom(E), hom(Fo)
        X = (H + G) / 2                              # HG = HX + XG

        check(_scalene_acute(A, B, C), "scalene acute triangle (H, O inside)")
        check(close(hom(A), Ap) and close(hom(B), Bp) and close(hom(C), Cp),
              "the map takes A, B, C to the midpoints A', B', C'")
        check(_on_line(H, A, D) and _on_line(H, B, E) and _on_line(H, C, Fo),
              "the three altitudes meet at H")
        check(close(hom(H), O), "the map takes H to O")
        for P1, Q1, M, S in ((B, C, Ap, Ds), (C, A, Bp, Es), (A, B, Cp, Fs)):
            check(abs(np.dot(S - M, Q1 - P1)) < 1e-9 and _on_line(O, M, S),
                  "each image altitude is a perpendicular bisector through O")
        check(abs(n(O - A) - n(O - B)) < 1e-9 and abs(n(O - A) - n(O - C)) < 1e-9,
              "O is the circumcentre")
        check(_on_line(Ds, Bp, Cp) and _on_line(Es, Cp, Ap) and _on_line(Fs, Ap, Bp),
              "the feet go to points of the midpoint triangle's sides")
        check(_on_line(G, H, O) and abs(n(H - G) - 2 * n(G - O)) < 1e-9,
              "H, G, O collinear with HG = 2 GO")

        cAlt, cImg, cMid, cEu = YELLOW_B, ORANGE, BLUE_B, GREEN_B
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)

        def vlab(s, P):
            return tag(s, 30).move_to(P + 0.42 * _unit(P - G))

        lA, lB, lC = vlab("A", A), vlab("B", B), vlab("C", C)
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        # midpoints and medians: G
        mids = VGroup(*[Dot(M, radius=0.06, color=cMid) for M in (Ap, Bp, Cp)])
        stk = VGroup(_ticks(B, Ap, 1), _ticks(Ap, C, 1), _ticks(C, Bp, 2),
                     _ticks(Bp, A, 2), _ticks(A, Cp, 3), _ticks(Cp, B, 3))
        meds = VGroup(*[DashedLine(V, M, color=GREY_B, stroke_width=2,
                                   dash_length=0.08)
                        for V, M in ((A, Ap), (B, Bp), (C, Cp))])
        dG = Dot(G, radius=0.08, color=WHITE)

        # every line the labels must clear: the final figure (segs_eu) and,
        # while they are shown, the medians (segs_m)
        segs = (_segs(A, B, C, closed=True) + _segs(Ap, Bp, Cp, closed=True)
                + _segs(A, D) + _segs(B, E) + _segs(C, Fo)
                + _segs(Ap, Ds) + _segs(Bp, Es) + _segs(Cp, Fs))
        eu0, eu1 = H + 0.35 * (H - O), O + 0.6 * (O - H)
        segs_m = segs + _segs(eu0, eu1) + _segs(A, Ap) + _segs(B, Bp) + _segs(C, Cp)
        segs_eu = segs + _segs(eu0, eu1)
        lAp = vlab("A'", Ap)
        lBp = _park(tag("B'", 28), Bp, [A, C, B, Ap, Cp, Es], segs_m, d0=0.3)
        lCp = _park(tag("C'", 28), Cp, [A, B, C, Ap, Bp, Fs], segs_m, d0=0.3)
        lG = _park(tag("G", 28), G, [A, B, C, Ap, Bp, Cp, eu0, eu1], segs_m,
                   d0=0.25)
        self.play(FadeIn(mids), FadeIn(stk), FadeIn(lAp), FadeIn(lBp),
                  FadeIn(lCp), run_time=0.7)
        self.play(Create(meds), FadeIn(dG), FadeIn(lG), run_time=1.0)

        # the altitudes: H
        alts = VGroup(*[Line(V, Ft, color=cAlt, stroke_width=3.5)
                        for V, Ft in ((A, D), (B, E), (C, Fo))])
        ras = VGroup(_ra(D, A, C, 0.16, cAlt), _ra(E, B, A, 0.16, cAlt),
                     _ra(Fo, C, B, 0.16, cAlt))
        dH = Dot(H, radius=0.08, color=cAlt)
        lH = _park(tag("H", 28, cAlt), H, [A, B, C, D, E, Fo, G, eu0], segs_m,
                   d0=0.25, avoid=[lG], sep=0.3)
        self.play(Create(alts), Create(ras), run_time=1.0)
        self.play(FadeIn(dH), FadeIn(lH), run_time=0.4)

        # the map: half a turn about G with halving, applied to a copy of
        # the triangle, its altitudes, H and the segment HG
        cp = VGroup(Polygon(A, B, C, stroke_color=cMid, stroke_width=4),
                    *[Line(V, Ft, color=cImg, stroke_width=4)
                      for V, Ft in ((A, D), (B, E), (C, Fo))],
                    Line(H, G, color=cEu, stroke_width=5),
                    Dot(H, radius=0.08, color=cImg))
        # the map's name, in the empty corner above C
        mtag = VGroup(tag("G", 30), tag("×(−½)", 30, cMid)).arrange(RIGHT, buff=0.15)
        mtag.move_to([C[0] + 0.2, A[1] - 0.6, 0.0])
        check(_hit(mtag, segs_m, 0.15) is None and _in_frame(mtag),
              "the map's tag sits in free space")
        _labels_ok([lA, lB, lC, lAp, lBp, lCp, lG, lH, mtag], segs_m,
                   "F23 with medians")
        self.play(FadeIn(cp), FadeIn(mtag), run_time=0.5)
        self.play(_homothety(cp, G, 0.5, turn=PI), run_time=3.2)
        check(_landed(cp[0], [Ap, Bp, Cp]), "the copy of ABC lands on A'B'C'")
        for ln, (S, M) in zip(cp[1:4], ((Ds, Ap), (Es, Bp), (Fs, Cp))):
            check(close(ln.get_start(), M, 1e-6) and close(ln.get_end(), S, 1e-6),
                  "each altitude lands on a perpendicular bisector")
        check(close(cp[4].get_start(), O, 1e-6) and close(cp[4].get_end(), G, 1e-6),
              "the segment HG lands on OG")
        check(close(cp[5].get_center(), O, 1e-6), "H lands on O")

        # the images are the perpendicular bisectors of the sides: O
        pbm = VGroup(_ra(Ap, Ds, C, 0.16, cImg), _ra(Bp, Es, A, 0.16, cImg),
                     _ra(Cp, Fs, B, 0.16, cImg))
        # O's label: in the pocket below-right of O (between the bisector
        # through A' and the side A'B' of the midpoint triangle), away from G
        lO = _park_dirs(tag("O", 28, cImg), O, [-50, -40, -60], segs_eu, d0=0.25,
                        avoid=[lG, lH], sep=0.4)
        self.play(Create(pbm), FadeIn(lO), FadeOut(meds), run_time=0.8)
        self.hold(0.5)

        # the Euler line: H, G, O in a row, HG = 2·GO
        eu = DashedLine(eu0, eu1, color=cEu, stroke_width=2.5, dash_length=0.1)
        tks = VGroup(_ticks(H, X, 1, cEu), _ticks(X, G, 1, cEu),
                     _ticks(G, O, 1, cEu))
        hg = Line(H, G, color=cEu, stroke_width=5)
        dX = Dot(X, radius=0.055, color=cEu)
        self.play(Create(eu), Create(hg), run_time=0.8)
        self.play(FadeIn(dX), FadeIn(tks), run_time=0.5)
        self.bring_to_front(dG, dH)

        labels = [lA, lB, lC, lAp, lBp, lCp, lG, lH, lO, mtag]
        _labels_ok(labels, segs_eu, "F23")
        _all_in_frame([tri, eu], "F23")
        self.play(Write(caption("H, G, O lie on one line, and HG = 2·GO", 34)))
        self.hold(2.2)


# ============================================ F24 THE NINE-POINT CIRCLE

class F24_NinePointCircle(Board):
    """Two halvings of the circumcircle (centre O, radius R). (1) The
    homothety with centre G and ratio −½ (half a turn about G with
    halving) takes A, B, C to the side midpoints A', B', C'. (2) The
    homothety with centre H and ratio ½ takes A, B, C to the midpoints
    A'', B'', C'' of HA, HB, HC. Both take O to the same point N: since
    HG = 2·GO (the Euler line, F23), the midpoint of HO is also the point
    half of GO beyond G. So both images are one circle, centre N, radius
    R/2, through six of the points. The two images of the radius OC point
    in opposite directions from N: C'C'' is a diameter. The foot F of the
    altitude from C sees it at a right angle (FC' lies on AB, FC'' on the
    altitude), so F is on the circle too; likewise D (diameter A'A'') and
    E (diameter B'B''). Only the C-family is labelled, to keep the busy
    middle of the figure legible."""

    def construct(self):
        # angles 59°, 75°, 46°: OH ≈ R/2, H at least R/4 from every side,
        # the nine points at least R/5 apart (a numerical search)
        Am, Bm, Cm = _tri_from_angles(75, 46)
        Om = _circumcentre(to3(Am), to3(Bm), to3(Cm))
        Rm = float(np.linalg.norm(to3(Am) - Om))
        m = 0.02                     # the circle's top and bottom are the extremes
        F = Frame(Om[0] - Rm - m, Om[0] + Rm + m, Om[1] - Rm - m, Om[1] + Rm + m,
                  centre=(0.9, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        O = F.P(Om)
        R = Rm * F.k
        n = np.linalg.norm
        G = (A + B + C) / 3
        H = _orthocentre(A, B, C)
        N = (O + H) / 2
        Ap, Bp, Cp = (B + C) / 2, (C + A) / 2, (A + B) / 2
        App, Bpp, Cpp = (H + A) / 2, (H + B) / 2, (H + C) / 2
        D, E, Fo = _foot(A, B, C), _foot(B, C, A), _foot(C, A, B)

        def h1(P):
            return G - 0.5 * (to3(P) - G)

        def h2(P):
            return H + 0.5 * (to3(P) - H)

        check(_scalene_acute(A, B, C), "scalene acute triangle")
        check(_on_line(G, H, O) and abs(n(H - G) - 2 * n(G - O)) < 1e-9,
              "the Euler line: HG = 2 GO")
        check(close(h1(O), N) and close(h2(O), N), "both maps take O to N")
        check(all(close(h1(P), Q) for P, Q in ((A, Ap), (B, Bp), (C, Cp))),
              "map 1 takes A, B, C to the side midpoints")
        check(all(close(h2(P), Q) for P, Q in ((A, App), (B, Bpp), (C, Cpp))),
              "map 2 takes A, B, C to the midpoints of HA, HB, HC")
        for P, Q, S in ((Ap, App, D), (Bp, Bpp, E), (Cp, Cpp, Fo)):
            check(close((P + Q) / 2, N), "A'A'' (etc.) is a diameter through N")
            check(abs(np.dot(P - S, Q - S)) < 1e-9, "the foot sees it at 90°")
        for P in (Ap, Bp, Cp, App, Bpp, Cpp, D, E, Fo):
            check(abs(n(P - N) - R / 2) < 1e-9, "nine points at distance R/2 from N")
        for V, P1, P2, S in ((A, B, C, D), (B, C, A, E), (C, A, B, Fo)):
            check(0 < np.dot(S - P1, P2 - P1) < n(P2 - P1) ** 2,
                  "the feet lie on the sides")
            check(_on_line(H, V, S), "H lies on the altitude")

        cG, cH, c9, cAlt, cEu = BLUE_B, TEAL_B, GREEN_B, YELLOW_B, GREY_A
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)

        def vlab(s, P):
            return tag(s, 30).move_to(P + 0.38 * _unit(P - O))

        lA, lB, lC = vlab("A", A), vlab("B", B), vlab("C", C)
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        # lines the labels must clear in the final figure
        eu0, eu1 = H + 0.3 * (H - O), O + 0.3 * (O - H)
        segs = (_segs(A, B, C, closed=True) + _circle_segs(O, R)
                + _circle_segs(N, R / 2) + _segs(A, D) + _segs(B, E) + _segs(C, Fo)
                + _segs(O, C) + _segs(eu0, eu1) + _segs(Cp, Cpp))
        segs_m = segs + _segs(A, Ap) + _segs(B, Bp) + _segs(C, Cp)

        # the circumcircle, its centre O and radius R
        dO = Dot(O, radius=0.07, color=WHITE)
        rOC = Line(O, C, color=GREY_B, stroke_width=2.5)
        lR = _park_beside(tag("R", 26, GREY_A), [(O, C)], segs_m)
        self.play(Create(circ), FadeIn(dO), Create(rOC), FadeIn(lR), run_time=1.2)

        # H (altitudes) and G (medians); the Euler line: HG = 2·GO
        alts = VGroup(*[Line(V, S, color=cAlt, stroke_width=2.5)
                        for V, S in ((A, D), (B, E), (C, Fo))])
        ras = VGroup(_ra(D, A, C, 0.14, cAlt), _ra(E, B, A, 0.14, cAlt),
                     _ra(Fo, C, B, 0.14, cAlt))
        dH = Dot(H, radius=0.07, color=cAlt)
        meds = VGroup(*[DashedLine(V, M, color=GREY_B, stroke_width=1.8,
                                   dash_length=0.08)
                        for V, M in ((A, Ap), (B, Bp), (C, Cp))])
        dG = Dot(G, radius=0.07, color=WHITE)
        # the H-G-N-O cluster is crowded with lines: each label goes in the
        # free sector found from the figure's directions (checked below)
        eud = np.degrees(_dir(H, O))
        lH = _park_dirs(tag("H", 26, cAlt), H, [eud + 110, eud + 90, eud + 130],
                        segs_m, d0=0.25)
        # G's label comes in as the medians go, so it must clear only the
        # lines of the final figure
        lG = _park_dirs(tag("G", 26), G, [eud + 90, eud + 100, eud + 80],
                        segs, d0=0.25, avoid=[lH], sep=0.12)
        lO = _park_dirs(tag("O", 26), O, [eud - 100, eud - 115, eud - 85],
                        segs, d0=0.25, avoid=[lH, lG, lR], sep=0.15)
        self.play(Create(alts), Create(ras), run_time=1.0)
        self.play(FadeIn(dH), FadeIn(lH), Create(meds), FadeIn(dG), run_time=0.9)
        eu = DashedLine(eu0, eu1, color=cEu, stroke_width=2.2, dash_length=0.08)
        t0 = tag("HG = 2·GO", 26, cEu)
        t0.move_to([O[0] - R - 1.35, O[1] + R - 0.45, 0.0])
        self.play(Create(eu), FadeIn(t0), FadeOut(meds), FadeIn(lG), FadeIn(lO),
                  run_time=0.9)
        self.hold(0.3)

        def payload(col):
            """The circumcircle with its centre, the radius OC and A, B, C."""
            return VGroup(Circle(radius=R, color=col, stroke_width=3.5).move_to(O),
                          Line(O, C, color=col, stroke_width=3),
                          Dot(O, radius=0.07, color=col),
                          *[Dot(V, radius=0.07, color=col) for V in (A, B, C)])

        def landed(cp, centre, ends):
            ok = (close(cp[0].get_center(), centre, 1e-6)
                  and abs(cp[0].width / 2 - R / 2) < 1e-6
                  and close(cp[1].get_start(), centre, 1e-6)
                  and close(cp[1].get_end(), ends[2], 1e-6)
                  and close(cp[2].get_center(), centre, 1e-6))
            for dt, P in zip(cp[3:6], ends):
                ok = ok and close(dt.get_center(), P, 1e-6)
            return ok

        # map 1: half a turn about G with halving -> the side midpoints
        t1 = VGroup(tag("G", 28), tag("×(−½)", 28, cG)).arrange(RIGHT, buff=0.12)
        t1.next_to(t0, DOWN, buff=0.4).align_to(t0, LEFT)
        cp1 = payload(cG)
        self.play(FadeIn(cp1), FadeIn(t1), run_time=0.5)
        self.play(_homothety(cp1, G, 0.5, turn=PI), run_time=3.0)
        check(landed(cp1, N, (Ap, Bp, Cp)), "map 1 lands on the circle (N, R/2)")
        placed = [lA, lB, lC, lH, lG, lO, lR]
        lCp = _park(tag("C'", 26, cG), Cp, [A, B, N, Fo, Cpp], segs, d0=0.28,
                    avoid=placed, sep=0.12)
        placed.append(lCp)
        self.play(FadeIn(lCp), run_time=0.4)

        # map 2: halving towards H -> the midpoints of HA, HB, HC
        t2 = VGroup(tag("H", 28, cAlt), tag("×½", 28, cH)).arrange(RIGHT, buff=0.12)
        t2.next_to(t1, DOWN, buff=0.35).align_to(t1, LEFT)
        cp2 = payload(cH)
        self.play(FadeIn(cp2), FadeIn(t2), run_time=0.5)
        self.play(_homothety(cp2, H, 0.5), run_time=2.6)
        check(landed(cp2, N, (App, Bpp, Cpp)), "map 2 lands on the same circle")
        dN = Dot(N, radius=0.07, color=c9)       # the common centre
        cd = np.degrees(_dir(N, Cpp))          # outwards from N through C''
        lCpp = _park_dirs(tag("C''", 26, cH), Cpp, [cd - 20, cd - 30, cd - 10, cd],
                          segs, d0=0.28, avoid=placed, sep=0.12)
        placed.append(lCpp)
        self.play(FadeIn(dN), FadeIn(lCpp), run_time=0.6)
        self.hold(0.4)

        # the two images of OC point opposite ways from N: C'C'' is a
        # diameter; the foot F sees it at a right angle (Thales)
        dia = Line(Cp, Cpp, color=c9, stroke_width=6)
        self.play(Create(dia), run_time=0.7)
        legs = VGroup(Line(Fo, Cp, color=cAlt, stroke_width=6),
                      Line(Fo, Cpp, color=cAlt, stroke_width=6))
        raF = _ra(Fo, Cp, Cpp, 0.2, WHITE, 3)
        dF = Dot(Fo, radius=0.08, color=cAlt)
        lF = _park(tag("F", 26, cAlt), Fo, [A, B, C, Cp, Cpp], segs, d0=0.28,
                   avoid=placed, sep=0.12)
        self.play(Create(legs), Create(raF), run_time=0.8)
        self.play(FadeIn(dF), FadeIn(lF), run_time=0.5)
        self.play(FadeOut(legs), dia.animate.set_stroke(width=3), run_time=0.4)

        # likewise for A and B (A'A'', B'B'' are diameters by the same two
        # maps): the feet D and E see them at right angles
        legsDE = VGroup(Line(D, Ap, color=cAlt, stroke_width=6),
                        Line(D, App, color=cAlt, stroke_width=6),
                        Line(E, Bp, color=cAlt, stroke_width=6),
                        Line(E, Bpp, color=cAlt, stroke_width=6))
        raDE = VGroup(_ra(D, Ap, App, 0.16, WHITE, 2.5), _ra(E, Bp, Bpp, 0.16, WHITE, 2.5))
        dDE = VGroup(Dot(D, radius=0.08, color=cAlt), Dot(E, radius=0.08, color=cAlt))
        self.play(Create(legsDE), Create(raDE), run_time=0.8)
        self.play(FadeIn(dDE), run_time=0.5)
        self.play(FadeOut(legsDE), run_time=0.4)

        # the nine-point circle
        nine = Circle(radius=R / 2, color=c9, stroke_width=5).move_to(N)
        dots = VGroup(*[Dot(P, radius=0.08, color=cG) for P in (Ap, Bp, Cp)],
                      *[Dot(P, radius=0.08, color=cH) for P in (App, Bpp, Cpp)])
        self.play(FadeOut(cp1), FadeOut(cp2), FadeIn(nine), FadeIn(dots),
                  FadeOut(VGroup(dia, raF, raDE)), run_time=1.0)
        self.bring_to_front(dots, dF, dDE, dN)

        labels = [lA, lB, lC, lO, lH, lG, lCp, lCpp, lF, lR, t0, t1, t2]
        _labels_ok(labels, segs, "F24")
        _all_in_frame([circ, t0, t1, t2], "F24")
        self.play(Write(caption("side midpoints, feet of the altitudes, midpoints of "
                                "HA, HB, HC: one circle of radius ½R", 28)))
        self.hold(2.2)


# ============================================ F31 THE FERMAT POINT

class F31_FermatPoint(Board):
    """Turn the triangle APB by 60° about A, away from C: P goes to P', B to
    B'. AP = AP' with a 60° angle between them, so APP' is equilateral and
    PP' = PA; and P'B' = PB. So PA + PB + PC is the length of the broken
    path C → P → P' → B', whose ends C and B' do not depend on P: it is
    least when the path is straight. Then the angles of the equilateral
    triangle at P and P' (60°) leave 120° for ∠APC (C, P, P' in a row) and
    for ∠AP'B' = ∠APB (P, P', B' in a row); the third angle, ∠BPC, is
    360° − 240° = 120°. (All angles of ABC are under 120°.)"""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(72, 46)
        th = -60 * DEGREES                     # the turn about A (away from C)
        Bpm = _rot2(Bm, Am, th)[:2]
        pts = np.array([Am, Bm, Cm, Bpm])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        F = Frame(lo[0] - 0.07, hi[0] + 0.07, lo[1] - 0.12, hi[1] + 0.1,
                  max_w=9.4, centre=(-1.55, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, Bp = F.P(Am), F.P(Bm), F.P(Cm), F.P(Bpm)
        n = np.linalg.norm

        def rot(X):
            return _rot2(X, A, th)

        # the Fermat point: on CB' (and on the line from A to the apex of
        # the equilateral triangle on BC, outside)
        Astar = _rot2(C, B, th)               # outer apex on BC (same sense)
        Fp = _meet(C, Bp, A, Astar)
        P0 = B + 0.45 * (C - B) + 0.16 * (A - B)    # a start inside, whose
        # image P' stays clear of B: |P' - B| = |P - (B turned back about A)|

        check(_scalene_acute(A, B, C), "scalene acute triangle (angles < 120°)")
        check(_cross(B - A, C - A) * _cross(B - A, Bp - A) < 0,
              "B' and C lie on opposite sides of AB")
        check(abs(n(Bp - A) - n(B - A)) < 1e-9 and abs(_angle(A, B, Bp) - PI / 3) < 1e-9,
              "ABB' is equilateral")
        for X, Y, Z in ((Fp, A, B), (Fp, B, C), (Fp, C, A)):
            check(abs(_angle(X, Y, Z) - 2 * PI / 3) < 1e-9, "the Fermat point sees 120°")
        Fq = rot(Fp)
        check(_on_line(Fp, C, Bp) and _on_line(Fq, C, Bp)
              and 0 < np.dot(Fp - C, Bp - C) < np.dot(Fq - C, Bp - C) < n(Bp - C) ** 2,
              "C, F, F', B' lie on one line in this order")
        check(_pip(P0, [A, B, C]) and n(P0 - Fp) > 0.8, "the start P is inside, away from F")
        check(n(rot(P0) - B) > 1.2, "the start P' is well clear of B")

        def total(P):
            return n(P - A) + n(P - B) + n(P - C)

        check(abs(total(Fp) - n(Bp - C)) < 1e-9, "at F the sum is CB'")
        check(total(P0) > n(Bp - C) + 0.3, "elsewhere it is longer")

        cA, cB, cC, cY = BLUE_B, TEAL_B, ORANGE, YELLOW_B
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        G = (A + B + C) / 3
        lA = tag("A", 30).move_to(A + 0.4 * UP)
        lB = tag("B", 30).move_to(B + 0.42 * _unit(B - G))
        lC = tag("C", 30).move_to(C + 0.42 * _unit(C - G))
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        t = ValueTracker(0.0)

        def Pt():
            a = t.get_value()
            return P0 + (Fp - P0) * a

        def Qt():
            return rot(Pt())

        sPA = always_redraw(lambda: Line(Pt(), A, color=cA, stroke_width=5))
        sPB = always_redraw(lambda: Line(Pt(), B, color=cB, stroke_width=5))
        sPC = always_redraw(lambda: Line(Pt(), C, color=cC, stroke_width=5))
        dP = always_redraw(lambda: Dot(Pt(), radius=0.08, color=WHITE))
        lP = always_redraw(lambda: tag("P", 28).move_to(
            Pt() + 0.36 * _free_dir(Pt(), [A, B, C, Qt()])))
        self.play(Create(sPA), Create(sPB), Create(sPC), FadeIn(dP), FadeIn(lP),
                  run_time=1.0)

        # the readout: PA + PB + PC stacked, beside CB'
        kb = 4.6 / n(Bp - C)                    # bar scale
        bx, by = 4.35, -2.2

        def stack():
            P = Pt()
            ls = [n(P - A), n(P - B), n(P - C)]
            g, y = VGroup(), by
            for L, col in zip(ls, (cA, cB, cC)):
                g.add(Line([bx, y, 0], [bx, y + L * kb, 0], color=col, stroke_width=16))
                y += L * kb
            return g

        bars = always_redraw(stack)
        ref = Line([bx + 1.55, by, 0], [bx + 1.55, by + n(Bp - C) * kb, 0],
                   color=cY, stroke_width=16)
        lsum = tag("PA+PB+PC", 22).move_to([bx, by - 0.32, 0])
        lref = tag("CB'", 22, cY).move_to([bx + 1.55, by - 0.32, 0])
        check(by + total(P0) * kb <= SAFE_TOP - 0.05, "the starting stack fits")
        self.play(FadeIn(bars), FadeIn(lsum), run_time=0.7)

        # turn APB by 60° about A (rigid)
        wedge = VGroup(mk([A, P0, B], cB, op=0.25, stroke_width=0),
                       Line(P0, A, color=cA, stroke_width=5),
                       Line(P0, B, color=cB, stroke_width=5))
        self.add(wedge)
        self.play(Rotate(wedge, angle=th, about_point=A), run_time=2.0)
        Q0 = rot(P0)
        check(close(wedge[1].get_start(), Q0, 1e-6) and close(wedge[2].get_end(), Bp, 1e-6),
              "the turn takes P to P' and B to B'")
        self.remove(wedge)
        tri2 = always_redraw(lambda: mk([A, Qt(), Bp], cB, op=0.25, stroke_width=0))
        sQA = always_redraw(lambda: Line(Qt(), A, color=cA, stroke_width=5))
        sQB = always_redraw(lambda: Line(Qt(), Bp, color=cB, stroke_width=5))
        dQ = always_redraw(lambda: Dot(Qt(), radius=0.08, color=WHITE))
        lQ = always_redraw(lambda: tag("P'", 28).move_to(
            Qt() + 0.38 * _free_dir(Qt(), [A, Bp, Pt(), C])))
        abp = DashedLine(A, Bp, color=GREY_B, stroke_width=2, dash_length=0.08)
        bbp = DashedLine(B, Bp, color=GREY_B, stroke_width=2, dash_length=0.08)
        lBp = tag("B'", 30).move_to(Bp + 0.42 * _unit(Bp - (A + B) / 2))
        self.add(tri2, sQA, sQB, dQ)
        self.bring_to_front(dP)
        self.play(FadeIn(lQ), FadeIn(lBp), Create(abp), Create(bbp), run_time=0.6)

        # APP' is equilateral: PP' = PA
        sPQ = always_redraw(lambda: Line(Pt(), Qt(), color=cA, stroke_width=5))
        arc60 = always_redraw(lambda: angle_arc(A, Pt(), Qt(), 0.55, cY, 4))
        l60 = always_redraw(lambda: tag("60°", 22, cY).move_to(
            A + 0.92 * angle_mid_dir(A, Pt(), Qt())))
        tks = always_redraw(lambda: VGroup(_ticks(A, Pt(), 1, cA), _ticks(A, Qt(), 1, cA),
                                           _ticks(Pt(), Qt(), 1, cA)))
        self.play(Create(arc60), FadeIn(l60), run_time=0.6)
        self.play(Create(sPQ), FadeIn(tks), run_time=0.8)
        self.hold(0.3)

        # the broken path C -> P -> P' -> B' and the straight segment CB'
        cb = DashedLine(C, Bp, color=cY, stroke_width=3, dash_length=0.1)
        self.play(Create(cb), FadeIn(ref), FadeIn(lref), run_time=0.8)
        self.bring_to_front(sPC, sPQ, sQB, dP, dQ)
        self.play(t.animate.set_value(1.0), run_time=3.2, rate_func=smooth)
        check(close(Pt(), Fp) and close(Qt(), Fq), "P has reached the Fermat point")
        check(abs(sum(b.get_length() for b in stack()) - ref.get_length()) < 1e-6,
              "the stack equals CB'")

        # the angles at F
        for m in (sPA, sPB, sPC, dP, lP, tri2, sQA, sQB, dQ, lQ, sPQ, arc60, l60, tks, bars):
            m.clear_updaters()
        segs = (_segs(A, B, C, closed=True) + _segs(C, Bp) + _segs(A, Bp) + _segs(B, Bp)
                + _segs(Fp, A) + _segs(Fp, B) + _segs(Fq, A))
        # P (now the Fermat point) keeps its label in the 60° gap between
        # PA and PP'; three equal sectors of 120° around it
        lF = _angle_label(tag("P", 28), Fp, A, Fq, 0.3, segs=segs)
        wedges = VGroup(*[_wedge(Fp, X, Y, 0.3, cY, op=0.55, stroke=0)
                          for X, Y in ((A, B), (B, C), (C, A))])
        l120 = VGroup()
        for X, Y in ((A, B), (B, C), (C, A)):
            # (the line C P P' B' halves the angle APB: go off-centre there)
            l120.add(_angle_label(tag("120°", 20, cY), Fp, X, Y, 0.5, r1=1.6,
                                  segs=segs, fracs=(0.5, 0.75, 0.25, 0.8, 0.2),
                                  avoid=list(l120) + [lA, lB, lC, lF, lQ], sep=0.08))
        self.remove(lP)
        self.add(lF)
        self.add(wedges)
        self.bring_to_back(wedges)
        self.bring_to_back(tri2)
        self.play(FadeIn(wedges), FadeIn(l120), FadeOut(l60), FadeOut(arc60),
                  FadeOut(tks), run_time=1.0)
        self.bring_to_front(sPA, sPB, sPC, sPQ, dP)

        _labels_ok([lA, lB, lC, lBp, lF, lQ, *l120], segs, "F31 figure")
        _labels_ok([lsum, lref], [], "F31 readout")
        _all_in_frame([tri, abp, bbp, ref, bars], "F31")
        self.play(Write(caption("PA + PB + PC is least where "
                                "∠APB = ∠BPC = ∠CPA = 120°", 32)))
        self.hold(2.2)


# ============================================ F56 HERON'S FORMULA

class F56_Heron(Board):
    """A = rs: the incentre I cuts ABC into three triangles of height r on
    the sides a, b, c. A = rₐ(s − a): the excentre Iₐ (opposite A) sees the
    sides at height rₐ; the triangles IₐAB and IₐCA make up ABC together
    with IₐBC, so A = ½rₐ(c + b − a). At B the internal and external
    bisectors BI, BIₐ are perpendicular, so the right triangles IZB (legs
    r = IZ, s − b = ZB, Z the incircle's contact on AB) and BTIₐ (legs
    s − c = BT, rₐ = TIₐ, T the excircle's contact on AB produced) have the
    same angle (half of B) at B and at Iₐ: turned a quarter turn and
    enlarged, the first lands exactly on the second, so
    r : (s − b) = (s − c) : rₐ. Multiplying A = rs by A = rₐ(s − a) gives
    A² = s(s − a)·r·rₐ = s(s − a)(s − b)(s − c). (The contact lengths
    s − b, s − c are the equal-tangent lengths.)"""

    def construct(self):
        aA, aB = 50 * DEGREES, 72 * DEGREES
        aC = PI - aA - aB
        am, bm, cm = 1.0, np.sin(aB) / np.sin(aA), np.sin(aC) / np.sin(aA)
        Am, Bm = np.array([0.0, 0.0]), np.array([cm, 0.0])
        Cm = bm * np.array([np.cos(aA), np.sin(aA)])
        sm = (am + bm + cm) / 2
        Iam = (-am * Am + bm * Bm + cm * Cm) / (-am + bm + cm)
        ram = float(np.sqrt(sm * (sm - am) * (sm - bm) * (sm - cm)) / (sm - am))
        F = Frame(-0.17, Iam[0] + ram + 0.03, -0.3, Iam[1] + ram + 0.03,
                  max_w=8.7, centre=(-2.15, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        n = np.linalg.norm
        a, b, c = n(C - B), n(A - C), n(B - A)
        s = (a + b + c) / 2
        Ar = abs(area([A, B, C]))
        I = (a * A + b * B + c * C) / (a + b + c)
        Ia = (-a * A + b * B + c * C) / (-a + b + c)
        r, ra = Ar / s, Ar / (s - a)
        Z, X, Y = _foot(I, A, B), _foot(I, B, C), _foot(I, C, A)
        T, Xa, Tb = _foot(Ia, A, B), _foot(Ia, B, C), _foot(Ia, C, A)
        P1 = _meet(A, Ia, B, C)                    # the bisector's foot

        check(_scalene_acute(A, B, C), "scalene triangle")
        for Q in (Z, X, Y):
            check(abs(n(I - Q) - r) < 1e-9, "the incircle touches the sides")
        for Q in (T, Xa, Tb):
            check(abs(n(Ia - Q) - ra) < 1e-9, "the excircle touches the side lines")
        check(abs(Ar - r * s) < 1e-9 and abs(Ar - ra * (s - a)) < 1e-9,
              "A = rs = rₐ(s − a)")
        check(abs(n(B - Z) - (s - b)) < 1e-9 and abs(n(B - T) - (s - c)) < 1e-9
              and abs(n(A - T) - s) < 1e-9, "the tangent lengths s − b, s − c, s")
        check(np.dot(T - B, B - A) > 0 and np.dot(Tb - C, C - A) > 0
              and 0 < np.dot(Xa - B, C - B) < a * a and 0 < np.dot(Z - A, B - A) < c * c,
              "T beyond B, T' beyond C, X' on BC, Z on AB")
        check(abs(np.dot(I - B, Ia - B)) < 1e-9, "BI ⊥ BIₐ")
        check(abs(r * ra - (s - b) * (s - c)) < 1e-9, "r·rₐ = (s − b)(s − c)")
        check(0 < np.dot(P1 - B, C - B) < a * a and _on_line(I, A, Ia),
              "A, I, Iₐ in a row; AIₐ crosses BC")
        check(abs(Ar ** 2 - s * (s - a) * (s - b) * (s - c)) < 1e-9, "Heron")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        # no vertex names: A is the area here, as in the formula; the
        # sides carry a, b, c and Iₐ is the centre of the circle on side a
        incirc = Circle(radius=r, color=GREY_A, stroke_width=3).move_to(I)
        excirc = Circle(radius=ra, color=GREY_A, stroke_width=3).move_to(Ia)
        ext = VGroup(DashedLine(B, T, color=GREY_B, stroke_width=2.5, dash_length=0.08),
                     DashedLine(C, Tb, color=GREY_B, stroke_width=2.5, dash_length=0.08))
        segs = (_segs(A, B, C, closed=True) + _segs(B, T) + _segs(C, Tb)
                + _circle_segs(I, r) + _circle_segs(Ia, ra)
                + _segs(I, Z) + _segs(I, X) + _segs(I, Y)
                + _segs(Ia, T) + _segs(Ia, Xa) + _segs(Ia, Tb))
        segs1 = segs + _segs(I, A) + _segs(I, B) + _segs(I, C)
        segs2 = segs + _segs(Ia, A) + _segs(Ia, B) + _segs(Ia, C)
        # side names outside the triangle where there is room (BC has the
        # excircle against it, so a goes inside, next to BC)
        la = _park_beside(tag("a", 28), [(B, C)], segs1 + segs2, avoid=[],
                          prefer=A - B, only=True)
        lb = _park_beside(tag("b", 28), [(C, A)], segs1 + segs2, avoid=[],
                          prefer=_foot(B, C, A) - B, only=True)
        # c above AB towards A: below AB, Z..B..T carry s − b and s − c
        lc = _park_beside(tag("c", 28), [(A, B)], segs1 + segs2, avoid=[],
                          prefer=UP, only=True, ats=(0.3, 0.26, 0.34, 0.22, 0.38))
        self.play(Create(tri), *[FadeIn(m) for m in (la, lb, lc)],
                  run_time=1.2)

        # the ledger
        lx = 2.55
        rows = [tag("A = ½r(a + b + c) = rs", 22),
                _subline([("A = ½r", False), ("a", True), ("(b + c − a)", False)], 22),
                _subline([("= r", False), ("a", True), ("(s − a)", False)], 22),
                _subline([("r : (s − b) = (s − c) : r", False), ("a", True)], 22),
                _subline([("A² = rs · r", False), ("a", True), ("(s − a)", False)], 22),
                tag("= s(s − a)(s − b)(s − c)", 22)]
        y = 3.2
        for i, m in enumerate(rows):
            m.move_to([0, y, 0]).align_to(np.array([lx, 0, 0]), LEFT)
            if i in (2, 5):
                m.shift(RIGHT * 0.42)
            y -= 0.62 if i not in (1, 4) else 0.5
        check(all(_in_frame(m) for m in rows) and rows[0].get_left()[0] > F.P((Iam[0] + ram, 0))[0] + 0.1,
              "the ledger sits right of the figure")

        # ---- A = rs: three triangles of height r on a, b, c
        dI = Dot(I, radius=0.06, color=WHITE)
        rads = VGroup(*[Line(I, Q, color=YELLOW_B, stroke_width=3) for Q in (Z, X, Y)])
        ras = VGroup(_ra(Z, I, B, 0.14), _ra(X, I, C, 0.14), _ra(Y, I, A, 0.14))
        lr = _park_beside(tag("r", 26, YELLOW_B), [(I, Z)], segs1 + _segs(B, Ia),
                          avoid=[la, lb, lc], prefer=A - B, only=True)
        lI = _park(tag("I", 26), I, [A, B, C, Z, X, Y], segs1 + _segs(B, Ia), d0=0.22,
                   avoid=[lr, la, lb, lc], sep=0.08)
        self.play(Create(incirc), FadeIn(dI), FadeIn(lI), run_time=0.8)
        self.play(Create(rads), Create(ras), FadeIn(lr), run_time=0.7)
        f1 = VGroup(mk([I, B, C], BLUE_D, 0.45, stroke_width=1.5),
                    mk([I, C, A], TEAL_D, 0.45, stroke_width=1.5),
                    mk([I, A, B], ORANGE, 0.45, stroke_width=1.5))
        self.add(f1)
        self.bring_to_back(f1)
        self.play(FadeIn(f1), run_time=0.7)
        self.play(FadeIn(rows[0], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.4)
        self.play(FadeOut(f1), run_time=0.4)

        # ---- A = rₐ(s − a): IₐAB + IₐCA − IₐBC
        dIa = Dot(Ia, radius=0.06, color=WHITE)
        lIa = _park(_subline([("I", False), ("a", True)], 26), Ia, [B, C, T, Xa, Tb, A],
                    segs2, d0=0.25)
        radsa = VGroup(*[Line(Ia, Q, color=BLUE_B, stroke_width=3) for Q in (T, Xa, Tb)])
        rasa = VGroup(_ra(T, Ia, A, 0.16), _ra(Xa, Ia, B, 0.16), _ra(Tb, Ia, A, 0.16))
        lra = _park_beside(_subline([("r", False), ("a", True)], 26, BLUE_B), [(Ia, T)], segs2,
                           avoid=[lIa, la, lb, lc], prefer=T - B, only=True)
        self.play(Create(ext), Create(excirc), FadeIn(dIa), FadeIn(lIa), run_time=1.0)
        self.play(Create(radsa), Create(rasa), FadeIn(lra), run_time=0.7)
        inB, outB = mk([A, B, P1], BLUE_D, 0.45, stroke_width=0), \
            mk([B, Ia, P1], BLUE_D, 0.45, stroke_width=0)
        inC, outC = mk([A, P1, C], TEAL_D, 0.45, stroke_width=0), \
            mk([P1, Ia, C], TEAL_D, 0.45, stroke_width=0)
        check(_tiles_exactly([[A, B, P1], [B, Ia, P1]], [A, B, Ia])
              and _tiles_exactly([[A, P1, C], [P1, Ia, C]], [A, Ia, C])
              and _tiles_exactly([[A, B, P1], [A, P1, C]], [A, B, C])
              and _tiles_exactly([[B, Ia, P1], [P1, Ia, C]], [B, Ia, C]),
              "IₐAB + IₐCA = ABC + IₐBC")
        edges = VGroup(Polygon(A, B, Ia, stroke_color=BLUE_B, stroke_width=3),
                       Polygon(A, Ia, C, stroke_color=TEAL_B, stroke_width=3))
        self.add(inB, outB, inC, outC)
        self.bring_to_back(inB, outB, inC, outC)
        self.play(FadeIn(VGroup(inB, outB, inC, outC)), Create(edges), run_time=0.9)
        self.play(FadeIn(rows[1], shift=LEFT * 0.2), run_time=0.6)
        minus = mk([B, Ia, C], RED_D, 0.6, stroke_color=RED_B, stroke_width=4)
        self.play(FadeIn(minus), run_time=0.5)
        self.play(FadeOut(minus), FadeOut(outB), FadeOut(outC), FadeOut(edges),
                  run_time=0.8)
        self.play(FadeIn(rows[2], shift=LEFT * 0.2), run_time=0.6)
        self.hold(0.3)
        self.play(FadeOut(inB), FadeOut(inC), run_time=0.4)

        # ---- at B: IZB, turned a quarter turn and enlarged, is BTIₐ
        cR, cS = YELLOW_B, BLUE_B
        small = VGroup(mk([I, Z, B], GREY_B, 0.35, stroke_width=0),
                       Line(I, Z, color=cR, stroke_width=6),
                       Line(Z, B, color=cS, stroke_width=6),
                       Line(B, I, color=WHITE, stroke_width=3))
        big = VGroup(Line(B, T, color=cR, stroke_width=6),
                     Line(T, Ia, color=cS, stroke_width=6),
                     Line(Ia, B, color=WHITE, stroke_width=3))
        # the leg lengths inside their triangles, just above the legs
        lsb = _park_beside(tag("s − b", 22, cS), [(Z, B)], segs + _segs(B, I),
                           avoid=[la, lb, lc, lI, lr, lIa, lra],
                           prefer=DOWN, only=True)
        lsc = _park_beside(tag("s − c", 22, cR), [(B, T)], segs + _segs(B, Ia),
                           avoid=[la, lb, lc, lI, lr, lIa, lra, lsb],
                           prefer=DOWN, only=True)
        self.play(FadeIn(small), Create(big), FadeIn(lsb), FadeIn(lsc),
                  lr.animate.set_color(cR), lra.animate.set_color(cS), run_time=0.9)
        cp = small.copy()
        self.add(cp)
        self.play(_glide(cp, B, Ia, PI / 2), run_time=2.0)
        Zs, Is = Ia + (_rot2(Z, B, PI / 2) - B), Ia + (_rot2(I, B, PI / 2) - B)
        check(_landed(cp[0], [Is, Zs, Ia]), "the turned copy sits in the corner at Iₐ")
        check(_on_line(Zs, Ia, T) and _on_line(Is, Ia, B), "its legs lie along IₐT, IₐB")
        self.play(_homothety(cp, Ia, ra / (s - b)), run_time=1.6)
        check(_landed(cp[0], [B, T, Ia]), "enlarged, it lands exactly on BTIₐ")
        self.bring_to_front(lsc, lsb, lra, lr)
        self.play(FadeIn(rows[3], shift=LEFT * 0.2), run_time=0.6)
        self.hold(0.3)

        # ---- multiply
        self.play(FadeIn(rows[4], shift=LEFT * 0.2), run_time=0.6)
        self.play(FadeIn(rows[5], shift=LEFT * 0.2), run_time=0.6)

        labels = [la, lb, lc, lI, lr, lIa, lra, lsb, lsc]
        _labels_ok(labels, segs, "F56")
        _labels_ok(rows, [], "F56 ledger")
        _all_in_frame([tri, excirc, *rows], "F56")
        self.play(Write(caption("A = √(s(s − a)(s − b)(s − c)),   s = ½(a + b + c)",
                                34)))
        self.hold(2.2)


# ============================================ F18 MEDIANS: SIX EQUAL AREAS

def _guide_in_frame(p, d, color=YELLOW_B):
    """Dashed line through p along d, clipped to the content area."""
    p, d = to3(p), _unit(d)
    t0, t1 = -1e9, 1e9
    for k, lo, hi in ((0, -SAFE_X, SAFE_X), (1, SAFE_BOTTOM, SAFE_TOP)):
        if abs(d[k]) < 1e-12:
            continue
        ta, tb = (lo - p[k]) / d[k], (hi - p[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return DashedLine(p + t0 * d, p + t1 * d, color=color, stroke_width=2,
                      dash_length=0.09)


def _colored_row(parts, size=30):
    """A line of text pieces [(string, colour), ...] side by side."""
    return VGroup(*[tag(t, size, c) for t, c in parts]).arrange(RIGHT, buff=0.12)


class F18_SixEqualAreas(Board):
    """D, E, F are the midpoints of BC, CA, AB; the medians meet at G. Two
    triangles with equal bases on one line and a common apex have equal
    areas: slide one along the line onto the other's base (rigid), then
    slide its apex parallel to the base back to the common apex (a shear,
    which keeps the area). So the six small triangles are equal in pairs:
    x, x on BC, y, y on CA, z, z on AB. The median AD halves ABC the same
    way: z + z + x = x + y + y, so z = y; the median CF gives
    z + y + y = z + x + x, so y = x."""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(74, 44)
        pts = np.array([Am, Bm, Cm])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Fr = Frame(lo[0] - 0.06, hi[0] + 0.06, lo[1] - 0.1, hi[1] + 0.09,
                   max_w=7.9, centre=(-2.45, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        D, E, F = (B + C) / 2, (C + A) / 2, (A + B) / 2
        G = (A + B + C) / 3

        P = {"BD": [G, B, D], "DC": [G, D, C], "CE": [G, C, E],
             "EA": [G, E, A], "AF": [G, A, F], "FB": [G, F, B]}
        T = abs(area([A, B, C]))
        check(_scalene_acute(A, B, C), "scalene triangle")
        check(_on_line(G, A, D) and _on_line(G, B, E) and _on_line(G, C, F),
              "the medians meet at G")
        check(_tiles_exactly(list(P.values()), [A, B, C]), "six pieces tile ABC")
        for k, Q in P.items():
            check(abs(abs(area(Q)) - T / 6) < 1e-9, f"piece {k} is 1/6 of ABC")

        cx, cy, cz = BLUE_D, TEAL_D, ORANGE
        tx, ty, tz = BLUE_B, TEAL_B, ORANGE
        col = {"BD": cx, "DC": cx, "CE": cy, "EA": cy, "AF": cz, "FB": cz}
        let = {"BD": ("x", tx), "DC": ("x", tx), "CE": ("y", ty), "EA": ("y", ty),
               "AF": ("z", tz), "FB": ("z", tz)}
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        lA = tag("A", 30).move_to(A + 0.4 * _unit(A - G))
        lB = tag("B", 30).move_to(B + 0.42 * _unit(B - G))
        lC = tag("C", 30).move_to(C + 0.42 * _unit(C - G))
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        tks = VGroup(_ticks(B, D, 1), _ticks(D, C, 1), _ticks(C, E, 2),
                     _ticks(E, A, 2), _ticks(A, F, 3), _ticks(F, B, 3))
        meds = VGroup(*[Line(V, M, color=WHITE, stroke_width=2.5)
                        for V, M in ((A, D), (B, E), (C, F))])
        dG = Dot(G, radius=0.07, color=YELLOW_B)
        segs = (_segs(A, B, C, closed=True) + _segs(A, D) + _segs(B, E) + _segs(C, F))
        lG = _park(tag("G", 28, YELLOW_B), G, [A, B, C, D, E, F], segs, d0=0.25)
        self.play(FadeIn(tks), Create(meds), run_time=1.0)
        self.play(FadeIn(dG), FadeIn(lG), run_time=0.4)
        pieces = {k: mk(Q, col[k], 0.55, stroke_width=0) for k, Q in P.items()}
        self.add(*pieces.values())
        self.bring_to_back(*pieces.values())
        letters = {}
        for k, Q in P.items():
            m = tag(let[k][0], 30, WHITE)
            m.move_to(sum(to3(q) for q in Q) / 3)
            letters[k] = m
        self.play(*[FadeIn(m) for m in pieces.values()], run_time=0.8)

        def slide_shear(src, dst):
            """src = [apex, p, q], dst = [apex, p', q'] with p, q, p', q' on
            one line and p' - p = q' - q: a copy of src slides by that
            vector (rigid), then its apex slides back along the parallel to
            the base line (a shear) onto dst."""
            v = dst[1] - src[1]
            check(close(dst[2] - src[2], v) and _on_line(dst[1], src[1], src[2])
                  and _on_line(dst[2], src[1], src[2]), "equal bases on one line")
            par = _guide_in_frame(src[0], v)
            cp = mk(src, YELLOW_B, 0.35, stroke_color=YELLOW_B, stroke_width=3)
            moved = [to3(q) + v for q in src]
            check(close(moved[1], dst[1]) and close(moved[2], dst[2]), "the bases match")
            check(all(_in_frame(Dot(q)) for q in moved), "the slid copy stays in view")
            return par, cp, v

        def move(pairs, run=(0.8, 0.8)):
            items = [(*slide_shear(s1, s2), s2) for s1, s2 in pairs]
            self.play(*[Create(it[0]) for it in items],
                      *[FadeIn(it[1]) for it in items], run_time=0.45)
            self.play(*[it[1].animate.shift(it[2]) for it in items], run_time=run[0])
            self.play(*[Transform(it[1], mk(it[3], YELLOW_B, 0.35, stroke_color=YELLOW_B,
                                            stroke_width=3)) for it in items],
                      run_time=run[1])
            for it in items:
                check(_landed(it[1], it[3]), "the sheared copy lands on its twin")
            self.play(*[FadeOut(it[0]) for it in items],
                      *[FadeOut(it[1]) for it in items], run_time=0.35)

        # the pairs: equal bases, common apex G -- equal areas x, y, z
        move([(P["BD"], P["DC"])])
        self.play(FadeIn(letters["BD"]), FadeIn(letters["DC"]), run_time=0.4)
        move([(P["CE"], P["EA"])])
        self.play(FadeIn(letters["CE"]), FadeIn(letters["EA"]), run_time=0.35)
        move([(P["FB"], P["AF"])])             # (this way round it stays in view)
        self.play(FadeIn(letters["AF"]), FadeIn(letters["FB"]), run_time=0.35)

        # the bookkeeping, in words of x, y, z
        px = 4.35
        W = WHITE
        r1 = _colored_row([("z", tz), ("+", W), ("z", tz), ("+", W), ("x", tx), ("=", W),
                           ("x", tx), ("+", W), ("y", ty), ("+", W), ("y", ty)])
        r2 = _colored_row([("z", tz), ("=", W), ("y", ty)])
        r3 = _colored_row([("z", tz), ("+", W), ("y", ty), ("+", W), ("y", ty), ("=", W),
                           ("z", tz), ("+", W), ("x", tx), ("+", W), ("x", tx)])
        r4 = _colored_row([("y", ty), ("=", W), ("x", tx)])
        r5 = _colored_row([("x", tx), ("=", W), ("y", ty), ("=", W), ("z", tz)], 34)
        for m, y in ((r1, 2.9), (r2, 2.15), (r3, 1.0), (r4, 0.25), (r5, -1.0)):
            m.move_to([px, y, 0.0])

        def strike(m):
            return Line(m.get_corner(DL) + LEFT * 0.04, m.get_corner(UR) + RIGHT * 0.04,
                        color=RED_B, stroke_width=4)

        # the median AD halves ABC (bases BD = DC on BC, apex A)
        half = Polygon(A, B, D, stroke_color=YELLOW_B, stroke_width=5)
        self.play(Create(half), run_time=0.5)
        move([([A, B, D], [A, D, C])])
        self.play(FadeOut(half), FadeIn(r1, shift=LEFT * 0.2), run_time=0.6)
        self.play(Create(strike(r1[4])), Create(strike(r1[6])), run_time=0.5)
        self.play(FadeIn(r2, shift=DOWN * 0.15), run_time=0.6)

        # the median CF halves ABC (bases FB = AF on AB, apex C)
        half = Polygon(C, F, B, stroke_color=YELLOW_B, stroke_width=5)
        self.play(Create(half), run_time=0.5)
        move([([C, F, B], [C, A, F])])
        self.play(FadeOut(half), FadeIn(r3, shift=LEFT * 0.2), run_time=0.6)
        self.play(Create(strike(r3[0])), Create(strike(r3[6])), run_time=0.5)
        self.play(FadeIn(r4, shift=DOWN * 0.15), run_time=0.6)

        # so all six are equal
        self.play(*[m.animate.set_fill(YELLOW_E, opacity=0.6) for m in pieces.values()],
                  FadeIn(r5, shift=DOWN * 0.15), run_time=0.9)
        labels = [lA, lB, lC, lG, *letters.values()]
        _labels_ok(labels, segs, "F18")
        _all_in_frame([tri, r1, r2, r3, r4, r5], "F18")
        check(min(m.get_left()[0] for m in (r1, r3)) > tri.get_right()[0] + 0.2,
              "the rows sit right of the figure")
        self.play(Write(caption("the three medians cut ABC into six triangles of "
                                "equal area, each 1/6 of ABC", 30)))
        self.hold(2.2)


# ============================================ F38 THE MIDPOINT TRIANGLE

class F38_MidpointTriangle(Board):
    """D, E, F are the midpoints of BC, CA, AB. Halving ABC towards A gives
    the triangle AFE, towards B the triangle FBD, towards C the triangle
    EDC: three copies at half scale in the corners. Half a turn about the
    midpoint of FE swaps F and E and takes A to F + E − A = D: the corner
    AFE lands exactly on the middle triangle DEF. Four congruent copies
    tile ABC, so each is a quarter of it."""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(70, 46)
        pts = np.array([Am, Bm, Cm])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Fr = Frame(lo[0] - 0.08, hi[0] + 0.08, lo[1] - 0.1, hi[1] + 0.09)
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        D, E, F = (B + C) / 2, (C + A) / 2, (A + B) / 2
        M = (F + E) / 2
        T = abs(area([A, B, C]))

        def half(X, V):
            return V + 0.5 * (to3(X) - V)

        corners = {"A": ([A, F, E], A, BLUE_D), "B": ([F, B, D], B, TEAL_D),
                   "C": ([E, D, C], C, ORANGE)}
        check(_scalene_acute(A, B, C), "scalene triangle")
        for key, (Q, V, _) in corners.items():
            check(_same_poly([half(X, V) for X in (A, B, C)], Q),
                  f"halving ABC towards {key} gives that corner")
        check(close(_rot2(A, M, PI), D) and close(_rot2(F, M, PI), E)
              and close(_rot2(E, M, PI), F), "the half-turn about M takes AFE to DEF")
        check(_tiles_exactly([[A, F, E], [F, B, D], [E, D, C], [D, E, F]], [A, B, C]),
              "the four triangles tile ABC")
        check(abs(abs(area([D, E, F])) - T / 4) < 1e-9, "[DEF] = [ABC]/4")

        G = (A + B + C) / 3
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        lA = tag("A", 30).move_to(A + 0.4 * _unit(A - G))
        lB = tag("B", 30).move_to(B + 0.42 * _unit(B - G))
        lC = tag("C", 30).move_to(C + 0.42 * _unit(C - G))
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        tks = VGroup(_ticks(B, D, 1), _ticks(D, C, 1), _ticks(C, E, 2),
                     _ticks(E, A, 2), _ticks(A, F, 3), _ticks(F, B, 3))
        dots = VGroup(*[Dot(X, radius=0.06) for X in (D, E, F)])
        lD = tag("D", 28).move_to(D + 0.4 * _unit(D - G))
        lE = tag("E", 28).move_to(E + 0.4 * _unit(E - G))
        lF = tag("F", 28).move_to(F + 0.4 * _unit(F - G))
        self.play(FadeIn(tks), FadeIn(dots), FadeIn(lD), FadeIn(lE), FadeIn(lF),
                  run_time=0.8)

        # three copies of ABC, each halved towards a vertex
        for keys in (("A",), ("B", "C")):
            cps = []
            for key in keys:
                Q, V, colr = corners[key]
                cp = mk([A, B, C], colr, 0.55, stroke_color=WHITE, stroke_width=3)
                cps.append((cp, Q, V))
            self.play(*[FadeIn(cp) for cp, _, _ in cps], run_time=0.4)
            self.play(*[cp.animate.scale(0.5, about_point=V) for cp, _, V in cps],
                      run_time=1.4)
            for cp, Q, V in cps:
                check(_same_poly(cp.get_vertices(), Q, 1e-6), "the halved copy lands")
            self.hold(0.2)

        # the middle: AFE turned half a turn about the midpoint of FE
        dM = Dot(M, radius=0.07, color=YELLOW_B)
        cp = mk([A, F, E], YELLOW_E, 0.7, stroke_color=WHITE, stroke_width=3)
        self.play(FadeIn(dM), FadeIn(cp), run_time=0.5)
        self.play(Rotate(cp, angle=PI, about_point=M), run_time=1.6)
        check(_landed(cp, [D, E, F]), "the turned corner lands on DEF")
        self.play(FadeOut(dM), run_time=0.3)
        self.bring_to_front(tri, dots)

        # four congruent copies at half scale: each is a quarter
        quarters = [tag("¼", 36, WHITE).move_to(sum(Q) / 3)
                    for Q in ([A, F, E], [F, B, D], [E, D, C], [D, E, F])]
        labels = [lA, lB, lC, lD, lE, lF, *quarters]
        segs = _segs(A, B, C, closed=True) + _segs(D, E, F, closed=True)
        _labels_ok(labels, segs, "F38")
        self.play(*[FadeIn(q) for q in quarters], run_time=0.8)
        self.play(Write(caption("[DEF] = ¼ [ABC]:  four congruent copies of ABC "
                                "at half scale", 32)))
        self.hold(2.2)


# ============================================ F32 THE ONE-SEVENTH TRIANGLE

class F32_OneSeventh(Board):
    """The cevians AA₁, BB₁, CC₁ go to the points one third along the sides
    (BA₁ = BC/3, CB₁ = CA/3, AC₁ = AB/3) and cut out the triangle DEF; each
    cevian is cut 3 : 3 : 1. Three more cuts, each through a vertex of DEF
    parallel to a cevian, run from the midpoint K of one side to the other
    trisection point of the next (F: from the midpoint of AB to the point
    one third along CA from A, parallel to BB₁; D and E likewise). That
    makes thirteen pieces: DEF, three small triangles on the sides, and in
    each of the three corner regions a corner piece, a piece at a side's
    midpoint and a remainder. Half a turn about each side's midpoint K
    moves the two pieces at that side out across it. Then the pieces form
    seven triangles: DEF, the half-turns of DEF about its three vertices
    (each a small side triangle plus a moved corner piece) and about the
    midpoints of its three sides (each a remainder plus a moved piece).
    Seven congruent triangles from the pieces of ABC: [DEF] = [ABC]/7. Only
    half-turns and parallels are used, so the picture works for every
    triangle."""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(66, 50)
        A0, B0, C0 = to3(Am), to3(Bm), to3(Cm)

        def build(A, B, C):
            A1, B1, C1 = B + (C - B) / 3, C + (A - C) / 3, A + (B - A) / 3
            A2, B2, C2 = B + 2 * (C - B) / 3, C + 2 * (A - C) / 3, A + 2 * (B - A) / 3
            Ka, Kb, Kc = (B + C) / 2, (C + A) / 2, (A + B) / 2
            D, E, F = _meet(A, A1, B, B1), _meet(B, B1, C, C1), _meet(C, C1, A, A1)
            sides = [  # (K, small, corner, mid, rem, vertex-copy, edge-copy)
                (Kb, [C, B1, E], [A, B2, F], [Kb, B1, E], [B2, Kb, E, F],
                 [E, 2 * E - F, 2 * E - D], [E + F - D, E, F]),
                (Kc, [A, C1, F], [B, C2, D], [Kc, C1, F], [C2, Kc, F, D],
                 [F, 2 * F - D, 2 * F - E], [F + D - E, F, D]),
                (Ka, [B, A1, D], [C, A2, E], [Ka, A1, D], [A2, Ka, D, E],
                 [D, 2 * D - E, 2 * D - F], [D + E - F, D, E])]
            return dict(A1=A1, B1=B1, C1=C1, A2=A2, B2=B2, C2=C2, D=D, E=E, F=F,
                        sides=sides)

        g0 = build(A0, B0, C0)
        allp = [A0, B0, C0]
        for side in g0["sides"]:
            allp += side[5] + side[6]
        pts = np.array([p[:2] for p in allp])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Fr = Frame(lo[0] - 0.07, hi[0] + 0.07, lo[1] - 0.07, hi[1] + 0.12)
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        g = build(A, B, C)
        D, E, F = g["D"], g["E"], g["F"]
        T, t = abs(area([A, B, C])), abs(area([D, E, F]))

        # ---- the mathematics, checked before anything is drawn
        check(abs(T / t - 7) < 1e-9, "[ABC] = 7 [DEF]")
        for X, Y, Z, W in ((A, F, D, g["A1"]), (B, D, E, g["B1"]), (C, E, F, g["C1"])):
            check(close(Y - X, Z - Y) and close(W - Z, (Z - Y) / 3),
                  "each cevian is cut 3 : 3 : 1")
        pieces = [[D, E, F]]
        for K, small, corner, mid, rem, vcp, ecp in g["sides"]:
            pieces += [small, corner, mid, rem]
            turned_c = [_rot2(X, K, PI) for X in corner]
            turned_m = [_rot2(X, K, PI) for X in mid]
            check(_tiles_exactly([small, turned_c], vcp), "small + turned corner = vertex copy")
            check(_tiles_exactly([rem, turned_m], ecp), "rest + turned piece = edge copy")
            check(abs(abs(area(vcp)) - t) < 1e-9 and abs(abs(area(ecp)) - t) < 1e-9,
                  "both copies have the area of DEF")
        check(len(pieces) == 13 and _tiles_exactly(pieces, [A, B, C]),
              "the thirteen pieces tile ABC")
        sev = [[D, E, F]] + [x for sd in g["sides"] for x in (sd[5], sd[6])]
        check(_disjoint(sev), "the seven triangles do not overlap")
        for (K, small, corner, mid, rem, vcp, ecp), V, (P1, P2) in zip(
                g["sides"], (E, F, D), ((E, F), (F, D), (D, E))):
            check(_same_poly([_rot2(X, V, PI) for X in (D, E, F)], vcp),
                  "a vertex copy is DEF turned half a turn about a vertex")
            M = (P1 + P2) / 2
            check(_same_poly([_rot2(X, M, PI) for X in (D, E, F)], ecp),
                  "an edge copy is DEF turned half a turn about a side's midpoint")
        # the cuts are parallel to cevians
        check(abs(_cross(g["sides"][0][2][1] - F, g["B1"] - B)) < 1e-9
              and abs(_cross(g["sides"][0][0] - E, g["A1"] - A)) < 1e-9,
              "the cuts on CA are parallel to BB₁ and AA₁")

        # ---- the picture
        cDEF = YELLOW_E
        cols = [(ORANGE, BLUE_D), (RED_D, TEAL_D), (PURPLE_B, GREEN_D)]
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        cev = VGroup(*[Line(X, Y, color=WHITE, stroke_width=3)
                       for X, Y in ((A, g["A1"]), (B, g["B1"]), (C, g["C1"]))])
        thirds = VGroup()
        for (P1, P2), nt in (((B, C), 1), ((C, A), 2), ((A, B), 3)):
            for k in range(3):
                thirds.add(_ticks(P1 + (P2 - P1) * k / 3, P1 + (P2 - P1) * (k + 1) / 3, nt))
        feet = VGroup(*[Dot(X, radius=0.06) for X in (g["A1"], g["B1"], g["C1"])])

        # every line of the final picture, for the label checks
        segs = _segs(A, B, C, closed=True)
        for poly in sev:
            segs += _segs(*poly, closed=True)
        for K, small, corner, mid, rem, vcp, ecp in g["sides"]:
            for poly in (small, corner, mid, rem):
                segs += _segs(*poly, closed=True)
        def nbrs(X):
            """The far ends of every drawn segment that starts at X."""
            out = [q for p_, q in segs if close(p_, X, 1e-7) and not close(q, X, 1e-7)]
            out += [p_ for p_, q in segs if close(q, X, 1e-7) and not close(p_, X, 1e-7)]
            return out

        # vertex names outside ABC and outside every one of the seven copies
        reg = [[A, B, C]] + sev
        lA = _park(tag("A", 30), A, nbrs(A), segs, d0=0.3, d1=1.6, regions=reg)
        lB = _park(tag("B", 30), B, nbrs(B), segs, d0=0.3, d1=1.6, avoid=[lA],
                   regions=reg)
        lC = _park(tag("C", 30), C, nbrs(C), segs, d0=0.3, d1=1.6, avoid=[lA, lB],
                   regions=reg)
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)
        self.play(FadeIn(thirds), FadeIn(feet), run_time=0.7)
        self.play(Create(cev), run_time=1.0)

        mDEF = mk([D, E, F], cDEF, 0.8, stroke_color=WHITE, stroke_width=2)
        cen = (D + E + F) / 3
        lD, lE, lF = (tag(s, 26, BLACK).move_to(X + 0.42 * _unit(cen - X))
                      for s, X in (("D", D), ("E", E), ("F", F)))
        self.play(FadeIn(mDEF), FadeIn(lD), FadeIn(lE), FadeIn(lF), run_time=0.7)

        # the cuts and the side midpoints
        cuts = VGroup()
        for K, small, corner, mid, rem, vcp, ecp in g["sides"]:
            cuts.add(Line(corner[2], corner[1], color=WHITE, stroke_width=2.5),
                     Line(mid[2], K, color=WHITE, stroke_width=2.5))
        kdots = VGroup(*[Dot(sd[0], radius=0.07, color=YELLOW_B) for sd in g["sides"]])
        self.play(Create(cuts), FadeIn(kdots), run_time=1.0)

        # colour every piece by the triangle it will help to make
        mobs = []
        for (K, small, corner, mid, rem, vcp, ecp), (cv, ce) in zip(g["sides"], cols):
            ms = mk(small, cv, 0.75, stroke_width=1.5)
            mc = mk(corner, cv, 0.75, stroke_width=1.5)
            mm = mk(mid, ce, 0.75, stroke_width=1.5)
            mr = mk(rem, ce, 0.75, stroke_width=1.5)
            mobs.append((K, ms, mc, mm, mr))
        allm = [m for _, *ms in mobs for m in ms]
        self.add(*allm)
        self.bring_to_back(*allm)
        self.play(*[FadeIn(m) for m in allm], FadeOut(cev), FadeOut(cuts),
                  run_time=0.9)
        self.bring_to_front(kdots)

        # half a turn about each side's midpoint, for two pieces
        for (K, ms, mc, mm, mr), sd in zip(mobs, g["sides"]):
            self.bring_to_front(mc, mm)
            self.play(Rotate(VGroup(mc, mm), angle=PI, about_point=K), run_time=1.3)
            check(_same_poly(mc.get_vertices(), [_rot2(X, K, PI) for X in sd[2]], 1e-6)
                  and _same_poly(mm.get_vertices(), [_rot2(X, K, PI) for X in sd[3]],
                                 1e-6), "the turned pieces landed")
            self.hold(0.15)
        self.play(FadeOut(kdots), FadeOut(thirds), FadeOut(feet), run_time=0.4)

        # seven triangles: outline them; each is DEF turned half a turn
        outl = VGroup(*[Polygon(*poly, stroke_color=WHITE, stroke_width=4)
                        for poly in sev])
        self.play(Create(outl), run_time=0.8)
        ghosts, turns = [], []
        for V, (P1, P2) in ((E, (E, F)), (F, (F, D)), (D, (D, E))):   # as in sev
            for centre in (V, (P1 + P2) / 2):
                gh = Polygon(D, E, F, stroke_color=YELLOW_B, stroke_width=5)
                ghosts.append(gh)
                turns.append(Rotate(gh, angle=PI, about_point=centre))
        self.add(*ghosts)
        self.play(*turns, run_time=1.6)
        for gh, poly in zip(ghosts, sev[1:]):
            check(_same_poly(gh.get_vertices(), poly, 1e-6), "a turned DEF covers a copy")
        self.play(*[FadeOut(gh) for gh in ghosts], run_time=0.5)
        sevens = [tag(str(k + 1), 26, WHITE).move_to(sum(to3(q) for q in poly) / 3)
                  for k, poly in enumerate(sev)]
        sevens[0].set_color(BLACK)
        labels = [lA, lB, lC]
        _labels_ok(labels, segs, "F32")
        _labels_ok(sevens, segs, "F32 numbers", pad=0.03)
        _all_in_frame([tri, outl], "F32")
        trace = DashedVMobject(Polygon(A, B, C, stroke_color=GREY_A, stroke_width=2.5),
                               num_dashes=60)
        self.play(FadeOut(lD), FadeOut(lE), FadeOut(lF), FadeOut(tri), FadeIn(trace),
                  *[FadeIn(m) for m in sevens], run_time=0.7)
        self.play(Write(caption("seven copies of the middle triangle DEF:   "
                                "[DEF] = 1/7 [ABC]", 32)))
        self.hold(2.2)
