# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3d.py — proofs without words, 2D (manim), circles and conics:
#     E14 the angle between two secants      E40 equal angles on a segment
#     E19 the hyperbola's reflection         E36 the Reuleaux triangle's area
#     E26 the Simson line                    E37 the circle of Apollonius
#     E22 the salinon                        E31 the tangent to y = x²
#     E34 three circles on a line            E38 the common chord
#
# Every move is rigid (a shift, manim's Rotate, or a fold — a half-turn in
# space about a line of the plane) and every landing, angle, length and
# tiling claim is checked numerically with check(...) before or right after
# it is drawn. Angles are drawn with angle_arc (never the reflex angle) and
# equal angles carry matching arcs. Every label is checked against the
# lines, arcs, marks and dots it must clear, so a wrong construction fails
# the render instead of drawing a wrong picture.


# ------------------------------------------------------------ private helpers

def _dir(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _perp(v):
    v = to3(v)
    return np.array([-v[1], v[0], 0.0])


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _heading(p, q):
    """Direction angle of the ray p -> q."""
    d = to3(q) - to3(p)
    return float(np.arctan2(d[1], d[0]))


def _foot(p, a, b):
    """Foot of the perpendicular from p onto the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = b - a
    return a + np.dot(p - a, d) / np.dot(d, d) * d


def _refl(p, c, u):
    """Mirror image of screen point p in the line through c along u."""
    p, c, u = to3(p), to3(c), _unit(u)
    d = p - c
    return c + 2 * np.dot(d, u) * u - d


def _rot(p, c, t):
    """Screen point p turned by t about c."""
    d = to3(p) - to3(c)
    return to3(c) + np.array([d[0] * np.cos(t) - d[1] * np.sin(t),
                              d[0] * np.sin(t) + d[1] * np.cos(t), 0.0])


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _wedge_span(vertex, p, q):
    """(start angle, span) of the non-reflex angle p-vertex-q, as _wedge
    and angle_arc draw it (counter-clockwise from start)."""
    v = to3(vertex)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return a1, span


def _wedge(vertex, p, q, radius, color, op=0.6):
    """Filled sector for the non-reflex angle p-vertex-q (screen points)."""
    a1, span = _wedge_span(vertex, p, q)
    return Sector(radius=radius, angle=span, start_angle=a1,
                  arc_center=to3(vertex), fill_color=color, fill_opacity=op,
                  stroke_width=0)


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame. In the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


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


def _tick_segs(p, q, n=1, size=0.13, at=0.5):
    """The strokes of _ticks(p, q, n, ...), as segments."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    return [(c + d * 0.09 * (k - (n - 1) / 2) - nrm * size,
             c + d * 0.09 * (k - (n - 1) / 2) + nrm * size) for k in range(n)]


def _chevron(p, q, color=YELLOW_B, size=0.17, width=3, n=1, at=0.5):
    """Parallel-line mark: n small '>' on segment pq (at fraction `at`),
    pointing from p towards q."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = p + (q - p) * at
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7, tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


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


def _dot(p, color=WHITE, r=0.06):
    return Dot(to3(p), radius=r, color=color)


def _free_dir(X, towards=(), dirs=()):
    """Unit vector at X along the middle of the widest angular gap between
    the directions to the points `towards` and the extra angles `dirs`
    (radians) — where a point's label sits clear of every line at it."""
    X = to3(X)
    angs = [_heading(X, q) for q in towards] + [float(d) for d in dirs]
    angs = sorted(a % TAU for a in angs)
    best, mid = -1.0, PI / 2
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        if a1 - a0 > best:
            best, mid = a1 - a0, (a0 + a1) / 2
    return _dir(mid)


def _arc_pts(c, r, a0, a1, n=120):
    """n + 1 screen points along the circle (c, r) from angle a0 to a1."""
    c = to3(c)
    return [c + r * _dir(t) for t in np.linspace(a0, a1, n + 1)]


def _curve(pts, color=YELLOW_B, width=4):
    """Open polyline through screen points (dense sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


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
    jittered grid over the region's box every point inside the region lies
    in exactly one polygon and every point outside in none."""
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


# ---- label hygiene: every label is checked against the lines it must clear

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _arc_segs(c, r, a0, a1, n=48):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    pts = _arc_pts(c, r, a0, a1, n)
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _poly_segs(pts, closed=True):
    """The outline of a polygon (or open polyline), as segments."""
    pts = [to3(p) for p in pts]
    out = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if closed:
        out.append((pts[-1], pts[0]))
    return out


def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


def _span_segs(v, p, q, r, n=24):
    """The arc angle_arc(v, p, q, r) draws, as segments."""
    a0, sp = _wedge_span(v, p, q)
    return _arc_segs(v, r, a0, a0 + sp, n)


def _dot_segs(p, r=0.07):
    """A dot as a small cross of segments (so labels keep off the dots)."""
    p = to3(p)
    return [(p - r * RIGHT, p + r * RIGHT), (p - r * UP, p + r * UP)]


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by
    pad), or None."""
    x0, x1, y0, y1 = _box(m, pad)
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        if (max(p[0], q[0]) < x0 or min(p[0], q[0]) > x1
                or max(p[1], q[1]) < y0 or min(p[1], q[1]) > y1):
            continue
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.015) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _overlap(a, b, gap=0.05):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _in_frame(*mobs, tol=0.0):
    return all(m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
               and m.get_top()[1] <= SAFE_TOP + tol
               and m.get_bottom()[1] >= SAFE_BOTTOM - tol for m in mobs)


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every line in segs (a
    list of screen segments) and of every other label."""
    for m in labels:
        bx = ", ".join(f"{v:.2f}" for v in _box(m))
        check(_in_frame(m), f"{what}: '{_name(m)}' [{bx}] inside the safe area")
        i = _hit(m, segs, pad)
        sg = "" if i is None else (f" {np.round(segs[i][0][:2], 2)}-"
                                   f"{np.round(segs[i][1][:2], 2)}")
        check(i is None, f"{what}: '{_name(m)}' [{bx}] clear of the lines (hits #{i}{sg})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _park(m, X, segs, base, dist=0.4, avoid=(), pad=0.08, step=15):
    """Put label m beside point X: try directions turning away from the
    unit vector `base` in `step`° steps (and a little farther out) until
    its box is clear of the segments and of the labels in `avoid`."""
    a0 = float(np.arctan2(base[1], base[0]))
    for dd in (dist, dist + 0.12, dist + 0.24):
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9,
                  10, -10, 11, -11, 12):
            m.move_to(to3(X) + dd * _dir(a0 + k * step * DEGREES))
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, 0.05) for o in avoid)):
                return m
    check(False, f"a free spot for the label '{_name(m)}'")


def _park_beside(m, p, q, side, segs, ats=(0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74),
                 gaps=(0.08, 0.13, 0.18, 0.24), avoid=(), pad=0.08):
    """Park label m beside the segment pq on the side `side` points to, at
    the first fraction in `ats` (and gap) where it clears segs and avoid."""
    for at in ats:
        for gap in gaps:
            _beside(m, p, q, side, gap, at)
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, 0.05) for o in avoid)):
                return m
    check(False, f"a free spot beside the segment for '{_name(m)}'")


def _angle_label(m, V, P, Q, r0, segs=(), pad=0.06, r1=3.0, fracs=(0.5,),
                 avoid=(), sep=0.05):
    """Put label m inside the non-reflex angle P-V-Q — on its bisector, or
    at the other fractions of the angle given in `fracs`, tried in order —
    at the least distance >= r0 from V where it clears both arms, segs and
    the mobjects in `avoid`."""
    V = to3(V)
    a1, span = _wedge_span(V, P, Q)
    arms = [_seg(V, P), _seg(V, Q)] + list(segs)
    for f in fracs:
        u = _dir(a1 + f * span)
        for d in np.arange(r0, r1, 0.03):
            m.move_to(V + d * u)
            if (_hit(m, arms, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"room for the angle label '{_name(m)}'")


def _final_ok(scene, cap):
    """Closing frame: the caption in its band, everything else inside the
    safe area, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_top()[1] < SAFE_BOTTOM,
          "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if m.has_points() or m.submobjects:
            check(_in_frame(m, tol=0.03), f"{_name(m)} inside the safe area "
                  f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
                  f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(not _overlap(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# ============================================ E14  — the angle between two secants

class E14_SecantAngle(Board):
    """Two secants from the outside point P: the first meets the circle at
    A then B, the second at C then D; θ = ∠BPD. The far arc BD is α, the
    near arc AC is β. Draw the chord CE through C parallel to PB. Fold the
    circle over its diameter square to the two parallel chords: the circle
    and both chords go to themselves, A lands on B and C on E, so the arc
    BE is β too and the rest of the far arc, ED, is α − β. The angle θ slid
    along the secant from P to C fills ∠ECD (its sides stay parallel), and
    ∠ECD is an inscribed angle on the arc ED: θ = (α − β)/2."""

    def construct(self):
        R = 2.7
        O = np.array([-1.85, 0.22, 0.0])
        P = np.array([4.85, -0.98, 0.0])
        n = np.linalg.norm
        e = _unit(O - P)

        def ends(u):
            b = float(np.dot(u, O - P))
            disc = float(np.sqrt(b * b - (np.dot(O - P, O - P) - R * R)))
            return P + (b - disc) * u, P + (b + disc) * u

        u1, u2 = _rot(e, ORIGIN, -16 * DEGREES), _rot(e, ORIGIN, 17 * DEGREES)
        A, B = ends(u1)
        C, D = ends(u2)
        E = C + 2 * np.dot(u1, O - C) * u1                 # chord through C ∥ PB
        w = _perp(u1)                                       # the fold: diameter ⊥ chords
        hd = {k: _heading(O, X) for k, X in zip("ABCDE", (A, B, C, D, E))}
        al = (hd["D"] - hd["B"]) % TAU                      # far arc B → D
        be = (hd["A"] - hd["C"]) % TAU                      # near arc C → A
        th = _ang(P, B, D)
        for X in (A, B, C, D, E):
            check(abs(n(X - O) - R) < 1e-9, "A, B, C, D, E on the circle")
        check(n(A - P) < n(B - P) and n(C - P) < n(D - P), "A, C the near points")
        check(close(_unit(E - C), u1), "CE ∥ PB, same sense")
        check(close(_refl(A, O, w), B) and close(_refl(C, O, w), E),
              "the fold over the diameter ⊥ the chords swaps A, B and C, E")
        check(0 < (hd["E"] - hd["B"]) % TAU < al and abs((hd["E"] - hd["B"]) % TAU - be) < 1e-9,
              "E on the far arc, and the arc BE is β")
        check(be < PI and al < PI and al > be, "both arcs minor, α > β")
        check(abs(_ang(C, E, D) - th) < 1e-9, "∠ECD = θ")
        check(abs(th - (al - be) / 2) < 1e-9, "θ = (α − β)/2")
        check(close(_unit(B - P), _unit(E - C)) and close(_unit(D - P), _unit(D - C)),
              "slid from P to C, θ's sides lie along CE and CD")
        rw = 1.25                                           # θ's arcs
        rb = R + 0.72                                       # the bracket over the far arc
        K1, K2 = O + 1.18 * R * w, O - 1.18 * R * w         # the fold's axis
        M1, M2 = _foot(O, A, B), _foot(O, C, E)
        segs = (_arc_segs(O, R, 0, TAU, 160)
                + [_seg(P, B), _seg(P, D), _seg(C, E)]
                + _arc_segs(O, rb, hd["B"], hd["B"] + al, 40)
                + [_seg(O + (rb - 0.14) * _dir(hd[k]), O + (rb + 0.14) * _dir(hd[k])) for k in "BD"]
                + _span_segs(P, B, D, rw) + _span_segs(C, E, D, rw)
                + [s for X in (A, B, C, D, E, P, O) for s in _dot_segs(X)])
        segs_axis = (segs + [_seg(K1, K2)] + _ra_segs(M1, A, O, 0.17)
                     + _ra_segs(M2, E, O, 0.17))

        # ---- the circle, P and the two secants
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        dO = _dot(O, GREY_A, 0.05)
        dP = _dot(P, YELLOW_B, 0.07)
        lP = tag("P", 30, YELLOW_B).move_to(P + 0.42 * RIGHT)
        self.play(Create(circ), FadeIn(dO), FadeIn(dP), FadeIn(lP), run_time=1.0)
        s1 = Line(P, B, color=WHITE, stroke_width=3)
        s2 = Line(P, D, color=WHITE, stroke_width=3)
        self.play(Create(s1), Create(s2), run_time=1.0)

        def tdirs(X):                                       # the circle's tangent at X
            a = _heading(O, X)
            return [a + PI / 2, a - PI / 2]

        dots = VGroup(*[_dot(X) for X in (A, B, C, D)])
        labs = {}
        for k, X, others in (("A", A, [P, B]), ("B", B, [P]), ("C", C, [P, D]), ("D", D, [P]),
                             ("E", E, [C])):
            labs[k] = _park(tag(k, 28), X, segs_axis, _free_dir(X, others + [O], tdirs(X)),
                            0.36, avoid=list(labs.values()), step=12)
        self.play(FadeIn(dots), *[FadeIn(labs[k]) for k in "ABCD"], run_time=0.6)

        # ---- θ at P, and the two arcs between the secants
        aP = angle_arc(P, B, D, rw, YELLOW_B, 4)
        wP = _wedge(P, B, D, rw, YELLOW_E, 0.55)
        lthP = tag("θ", 30, YELLOW_B)
        _angle_label(lthP, P, B, D, rw + 0.2)
        self.add(wP)
        self.bring_to_back(wP)
        wP.set_fill(opacity=0)
        self.play(Create(aP), wP.animate.set_fill(opacity=0.55), FadeIn(lthP), run_time=0.8)
        arcBD = Arc(radius=R, start_angle=hd["B"], angle=al, arc_center=O,
                    color=BLUE_B, stroke_width=9)
        arcCA = Arc(radius=R, start_angle=hd["C"], angle=be, arc_center=O,
                    color=ORANGE, stroke_width=9)
        mal = hd["B"] + al / 2
        lal = _park(tag("α", 32, BLUE_B), O + (R + 0.42) * _dir(mal), segs_axis, _dir(mal),
                    0.0, avoid=list(labs.values()), step=10)
        lbe = tag("β", 32, ORANGE).move_to(O + (R + 0.42) * _dir(hd["C"] + be / 2))
        self.play(Create(arcBD), FadeIn(lal), run_time=0.8)
        self.play(Create(arcCA), FadeIn(lbe), run_time=0.7)
        self.bring_to_front(dots, dP)
        self.hold(0.5)

        # ---- the chord through C parallel to PB
        ce = Line(C, E, color=TEAL_B, stroke_width=4)
        dE = _dot(E)
        lE = labs.pop("E")
        chev = VGroup(_chevron(P, A, YELLOW_B, at=0.45), _chevron(C, E, YELLOW_B, at=0.62))
        self.play(Create(ce), FadeIn(dE), FadeIn(lE), run_time=0.9)
        self.play(FadeIn(chev), run_time=0.5)

        # ---- fold over the diameter square to both chords: arc AC lands on arc BE
        axis = DashedLine(K2, K1, color=GREY_A, stroke_width=2.5, dash_length=0.1)
        ra1 = _ra(M1, A, O, 0.17, GREY_A, 2)
        ra2 = _ra(M2, E, O, 0.17, GREY_A, 2)
        self.play(Create(axis), Create(ra1), Create(ra2), run_time=0.8)
        flap = arcCA.copy().set_stroke(width=11)
        self.add(flap)
        self.bring_to_front(dots, dE, dO)
        self.play(_fold(flap, O, O + w), run_time=1.8)
        fl_ends = [flap.get_start(), flap.get_end()]
        check(any(close(p_, B, 1e-6) for p_ in fl_ends) and any(close(p_, E, 1e-6) for p_ in fl_ends),
              "folded, the arc AC lies on the arc BE")
        arcBE = Arc(radius=R, start_angle=hd["B"], angle=be, arc_center=O,
                    color=ORANGE, stroke_width=9)
        arcED = Arc(radius=R, start_angle=hd["E"], angle=al - be, arc_center=O,
                    color=GREEN_B, stroke_width=9)
        # the labels of the two parts of the far arc sit inside the circle
        segs_in = segs_axis
        mBE, mED = hd["B"] + be / 2, hd["E"] + (al - be) / 2
        lbe2 = _park(tag("β", 32, ORANGE), O + (R - 0.42) * _dir(mBE), segs_in,
                     _dir(mBE + PI), 0.0, step=10)
        lamb = _park(tag("α − β", 28, GREEN_B), O + (R - 0.75) * _dir(mED), segs_in,
                     _dir(mED + PI), 0.0, avoid=(lbe2,), step=10)
        # α now names the union BE + ED: an outer bracket spans the whole far arc
        brk = VGroup(Arc(radius=rb, start_angle=hd["B"], angle=al, arc_center=O,
                         color=BLUE_B, stroke_width=3),
                     Line(O + (rb - 0.14) * _dir(hd["B"]), O + (rb + 0.14) * _dir(hd["B"]),
                          color=BLUE_B, stroke_width=3),
                     Line(O + (rb - 0.14) * _dir(hd["D"]), O + (rb + 0.14) * _dir(hd["D"]),
                          color=BLUE_B, stroke_width=3))
        lal2 = tag("α", 32, BLUE_B).move_to(O + (rb + 0.36) * _dir(hd["B"] + al / 2))
        self.remove(flap)
        self.add(arcBE)
        self.bring_to_front(dots, dE)
        self.play(FadeIn(lbe2), Create(brk), lal.animate.move_to(lal2.get_center()),
                  run_time=0.8)
        self.play(Create(arcED), FadeIn(lamb), run_time=0.8)
        self.bring_to_front(dots, dE)
        self.play(FadeOut(axis), FadeOut(ra1), FadeOut(ra2), run_time=0.5)

        # ---- θ slid along the secant from P to C fills ∠ECD
        wm = wP.copy()                                      # same fill: the same angle
        self.add(wm)
        self.bring_to_front(s2, ce, dots, dP)
        self.play(wm.animate.shift(C - P), run_time=1.8)
        check(close(wm.get_arc_center(), C, 1e-6), "the slid corner lands on C")
        aC = angle_arc(C, E, D, rw, YELLOW_B, 4)
        lthC = tag("θ", 30, YELLOW_B)
        segs_c = [_seg(P, B), _seg(P, D), _seg(C, E)] + _arc_segs(O, R, 0, TAU, 120)
        _angle_label(lthC, C, E, D, rw + 0.2, segs_c)
        self.play(Create(aC), FadeIn(lthC), run_time=0.6)
        self.bring_to_front(dots)

        _labels_ok(list(labs.values()) + [lP, lE, lthP, lal, lbe, lbe2, lamb], segs_axis,
                   "E14 (with the fold's axis)")
        texts = list(labs.values()) + [lP, lE, lthP, lthC, lal, lbe, lbe2, lamb]
        _labels_ok(texts, segs, "E14")
        cap = caption("θ  =  ½ (α − β)", 38)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E40  — equal angles on a segment

def _second_hit(A, d, O, R):
    """The other point where the line through A (on the circle (O, R))
    along d meets the circle."""
    d = _unit(d)
    return to3(A) - 2 * np.dot(d, to3(A) - to3(O)) * d


class E40_EqualAnglesConcyclic(Board):
    """The circle through A, B and P: every point of the arc on P's side
    sees AB at the same angle φ (the inscribed angle theorem — P travels
    along the arc and its angle does not change). A point Q inside the
    circle: the ray AQ meets the arc at X, which sees AB at φ; slide that
    angle along the line from X to Q — its sides stay parallel — and it
    fits strictly inside ∠AQB (in the triangle QXB the exterior angle at Q
    is φ plus the angle at B), so ∠AQB > φ. A point Q outside: the segment
    AQ meets the arc at X; φ slid from X to Q overshoots QB, so ∠AQB < φ.
    Only the points of the arc see AB at φ: if ∠AQB = ∠APB, with P and Q on
    the same side of AB, then Q is on the circle through A, B, P."""

    def construct(self):
        phi = 55 * DEGREES
        n = np.linalg.norm
        A = np.array([-2.2, -1.45, 0.0])
        B = np.array([2.2, -1.45, 0.0])
        R = n(B - A) / (2 * np.sin(phi))
        O = (A + B) / 2 + R * np.cos(phi) * UP
        tB, tA = _heading(O, B), _heading(O, A) % TAU        # the arc above AB: tB → tA
        check(tB < 0 < PI < tA, "the arc above AB runs counter-clockwise from B to A")

        def arc_pt(t):
            return O + R * _dir(t)

        for t in np.linspace(tB + 0.05, tA - 0.05, 41):
            check(abs(_ang(arc_pt(t), A, B) - phi) < 1e-9,
                  "every point of the arc above AB sees AB at φ")
        Q1 = np.array([1.2, 0.5, 0.0])
        X1 = _second_hit(A, Q1 - A, O, R)
        Q2 = np.array([3.4, 2.3, 0.0])
        X2 = _second_hit(A, Q2 - A, O, R)
        check(n(Q1 - O) < R and Q1[1] > A[1], "Q₁ inside the circle, above AB")
        check(n(Q2 - O) > R and Q2[1] > A[1], "Q₂ outside the circle, above AB")
        check(abs(n(X1 - O) - R) < 1e-9 and np.dot(Q1 - A, X1 - Q1) > 0,
              "X₁ on the circle, beyond Q₁ on the ray AQ₁")
        check(abs(n(X2 - O) - R) < 1e-9 and np.dot(X2 - A, Q2 - X2) > 0,
              "X₂ on the circle, between A and Q₂")
        for X in (X1, X2):
            check(X[1] > A[1] and abs(_ang(X, A, B) - phi) < 1e-9, "X on the arc: it sees AB at φ")
        g1, g2 = _ang(Q1, A, B), _ang(Q2, A, B)
        check(g1 > phi + 10 * DEGREES and g2 < phi - 8 * DEGREES,
              "inside: ∠AQB > φ; outside: ∠AQB < φ (plainly)")
        # slid from X to Q along XA, φ keeps the side towards A and a side ∥ XB
        for Q, X, inside in ((Q1, X1, True), (Q2, X2, False)):
            check(close(_unit(A - X), _unit(A - Q)), "the slide keeps the side towards A")
            par = _ang(Q, A, Q + (B - X))                   # the slid φ at Q
            check(abs(par - phi) < 1e-9, "the slid angle is φ")
            gap = _ang(Q, Q + (B - X), B)
            if inside:
                check(abs(_ang(Q, A, B) - (phi + gap)) < 1e-9 and gap > 0.05,
                      "inside: ∠AQB = φ + the angle at B")
            else:
                check(abs(_ang(Q, A, B) - (phi - gap)) < 1e-9 and gap > 0.05,
                      "outside: ∠AQB = φ − the angle at B")

        rp = 0.78                                            # φ's arcs
        P0 = arc_pt(128 * DEGREES)
        tP = ValueTracker(128 * DEGREES)

        def segs_base():
            return ([_seg(A, B)] + _arc_segs(O, R, 0, TAU, 160)
                    + [s for X in (A, B) for s in _dot_segs(X)])

        # ---- A, B and a point P that sees AB at φ
        ab = Line(A, B, color=WHITE, stroke_width=4)
        dA, dB = _dot(A), _dot(B)
        lA = tag("A", 28).move_to(A + 0.4 * LEFT)
        lB = tag("B", 28).move_to(B + 0.4 * RIGHT)
        self.play(Create(ab), FadeIn(dA), FadeIn(dB), FadeIn(lA), FadeIn(lB), run_time=0.8)

        def pparts():
            P = arc_pt(tP.get_value())
            g = VGroup(Line(P, A, color=BLUE_B, stroke_width=3),
                       Line(P, B, color=BLUE_B, stroke_width=3),
                       _wedge(P, A, B, rp, BLUE_D, 0.6),
                       angle_arc(P, A, B, rp, BLUE_B, 4),
                       _dot(P, BLUE_B, 0.07))
            return g

        def plabels():
            P = arc_pt(tP.get_value())
            lp = tag("P", 28, BLUE_B).move_to(P + 0.4 * _unit(P - O))
            lf = tag("φ", 28, BLUE_B).move_to(P + (rp + 0.3) * angle_mid_dir(P, A, B))
            return VGroup(lp, lf)

        for t in np.linspace(62, 138, 39) * DEGREES:        # P's labels stay clear as it moves
            tP.set_value(t)
            P = arc_pt(t)
            L = plabels()
            _labels_ok(list(L) + [lA, lB], segs_base() + [_seg(P, A), _seg(P, B)]
                       + _span_segs(P, A, B, rp) + _dot_segs(P), f"E40 (P at {np.degrees(t):.0f}°)")
        tP.set_value(128 * DEGREES)
        pp, pl = pparts(), plabels()
        self.play(Create(pp[0]), Create(pp[1]), FadeIn(pp[4]), FadeIn(pl[0]), run_time=0.8)
        self.play(FadeIn(pp[2]), Create(pp[3]), FadeIn(pl[1]), run_time=0.6)
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        self.play(Create(circ), run_time=1.1)
        self.bring_to_front(ab, pp, dA, dB)

        # ---- P travels along the arc: the angle stays φ
        self.remove(*pp.submobjects, *pl.submobjects)     # they entered the scene one by one
        pp, pl = always_redraw(pparts), always_redraw(plabels)
        self.add(pp, pl)
        self.play(tP.animate.set_value(62 * DEGREES), run_time=1.6)
        self.play(tP.animate.set_value(138 * DEGREES), run_time=1.8)
        self.play(tP.animate.set_value(128 * DEGREES), run_time=0.6)
        pp.clear_updaters()
        pl.clear_updaters()
        self.hold(0.3)

        def demo(Q, X, inside):
            """Q, the point X of the arc on the line AQ, φ at X slid to Q."""
            col = ORANGE
            g_lines = VGroup(Line(Q, A, color=col, stroke_width=3),
                             Line(Q, B, color=col, stroke_width=3))
            ext = (DashedLine(Q, X, color=GREY_A, stroke_width=2.5, dash_length=0.08) if inside
                   else VGroup())
            xb = Line(X, B, color=BLUE_B, stroke_width=3)
            dQ, dX = _dot(Q, col, 0.07), _dot(X, BLUE_B, 0.07)
            wX = _wedge(X, A, B, rp, BLUE_D, 0.6)
            aX = angle_arc(X, A, B, rp, BLUE_B, 4)
            segs = (segs_base() + [_seg(Q, A), _seg(Q, B), _seg(X, B), _seg(Q, X)]
                    + _span_segs(X, A, B, rp) + _span_segs(Q, A, B, rp + 0.16)
                    + _span_segs(Q, A, Q + (B - X), rp)
                    + _dot_segs(Q) + _dot_segs(X) + [_seg(Q, Q + rp * _unit(B - X))]
                    + list(pp_segs))
            lQ = _park(tag("Q", 28, col), Q, segs, _free_dir(Q, [A, B, X]), 0.38,
                       avoid=fixed_labels)
            lX = _park(tag("X", 28, BLUE_B), X, segs, _unit(X - O), 0.38,
                       avoid=fixed_labels + [lQ])
            lfX = tag("φ", 28, BLUE_B)
            _angle_label(lfX, X, A, B, rp + 0.12, segs, avoid=fixed_labels + [lQ, lX])
            self.play(Create(g_lines), FadeIn(dQ), FadeIn(lQ), run_time=0.8)
            self.play(Create(ext), Create(xb), FadeIn(dX), FadeIn(lX), run_time=0.8)
            self.play(FadeIn(wX), Create(aX), FadeIn(lfX), run_time=0.6)
            self.bring_to_front(dX, dQ)
            wm = wX.copy()
            self.add(wm)
            self.bring_to_front(g_lines, dX, dQ)
            self.play(wm.animate.shift(Q - X), run_time=1.5)
            check(close(wm.get_arc_center(), Q, 1e-6), "the slid corner lands on Q")
            aQ = angle_arc(Q, A, B, rp + 0.16, col, 4)
            lg = tag("> φ" if inside else "< φ", 28, col)
            if inside:                                      # inside the wide angle, by its arc
                _angle_label(lg, Q, A, B, rp + 0.3, segs, fracs=(0.5, 0.35, 0.65),
                             r1=1.5, avoid=fixed_labels + [lQ, lX, lfX])
            else:                                           # the narrow angle: just beyond Q
                _park(lg, Q, segs, _unit(Q - (A + B) / 2), 0.62,
                      avoid=fixed_labels + [lQ, lX, lfX], step=10)
                check(n(lg.get_center() - Q) < 1.0, "'< φ' sits by the arc at Q")
            self.play(Create(aQ), FadeIn(lg), run_time=0.7)
            self.bring_to_front(dQ)
            _labels_ok(fixed_labels + [lQ, lX, lfX, lg], segs, f"E40 ({'inside' if inside else 'outside'})")
            self.hold(0.6)
            return VGroup(g_lines, ext, xb, dQ, dX, wX, aX, wm, aQ, lQ, lX, lfX, lg)

        P = arc_pt(128 * DEGREES)
        pp_segs = [_seg(P, A), _seg(P, B)] + _span_segs(P, A, B, rp) + _dot_segs(P)
        fixed_labels = [lA, lB, pl[0], pl[1]]
        pp_dim = pparts()                                   # P's figure, dimmed meanwhile
        for m in pp_dim:
            m.set_stroke(opacity=0.35)
        pp_dim[2].set_fill(opacity=0.25)
        pp_dim[4].set_fill(opacity=0.5)
        self.play(FadeOut(pp), FadeIn(pp_dim), run_time=0.3)
        g_in = demo(Q1, X1, True)
        self.play(FadeOut(g_in), run_time=0.5)
        g_out = demo(Q2, X2, False)
        pp = pparts()                                       # back at full strength
        self.play(FadeOut(g_out), FadeOut(pp_dim), FadeIn(pp), run_time=0.5)

        # ---- so a point seeing AB at φ is on the arc: A, B, P, Q concyclic
        tQ = 55 * DEGREES
        Q = arc_pt(tQ)
        check(abs(_ang(Q, A, B) - phi) < 1e-9, "Q on the arc sees AB at φ")
        qg = VGroup(Line(Q, A, color=ORANGE, stroke_width=3), Line(Q, B, color=ORANGE, stroke_width=3),
                    _wedge(Q, A, B, rp, BLUE_D, 0.6), angle_arc(Q, A, B, rp, BLUE_B, 4),
                    _dot(Q, ORANGE, 0.07))
        lQ = tag("Q", 28, ORANGE).move_to(Q + 0.4 * _unit(Q - O))
        lfQ = tag("φ", 28, BLUE_B).move_to(Q + (rp + 0.3) * angle_mid_dir(Q, A, B))
        self.play(Create(qg[0]), Create(qg[1]), FadeIn(qg[4]), FadeIn(lQ), run_time=0.7)
        self.play(FadeIn(qg[2]), Create(qg[3]), FadeIn(lfQ), run_time=0.6)
        self.bring_to_front(dA, dB)
        segs = (segs_base() + pp_segs + [_seg(Q, A), _seg(Q, B)] + _span_segs(Q, A, B, rp)
                + _dot_segs(Q))
        _labels_ok([lA, lB, pl[0], pl[1], lQ, lfQ], segs, "E40 final")
        cap = caption("∠APB = ∠AQB   ⟹   A, B, P, Q concyclic", 32)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E19  — the hyperbola's reflection property

class E19_HyperbolaReflection(Board):
    """The branch of the hyperbola round F₂ is where PF₁ − PF₂ = 2a;
    between the branches the difference is smaller. At a point P of it,
    take the line ℓ that halves ∠F₁PF₂ and fold PF₂ over ℓ: it lies along
    PF₁, F₂ landing at F₂′, so F₂′F₁ = PF₁ − PF₂ = 2a. For any other point Q
    of ℓ, QF₂ = QF₂′ (the fold) and in the triangle F₁QF₂′ the side QF₁ is
    shorter than the path QF₂′ + F₂′F₁: QF₁ − QF₂ < 2a. So every other
    point of ℓ lies between the branches: ℓ touches the branch at P without
    crossing it — ℓ is the tangent, and it halves the angle between the
    focal radii. Its vertical angle shows the optics: light from F₂ leaves
    the mirror at P along F₁P produced, as if it came from F₁."""

    def construct(self):
        a, b = 1.65, 1.55
        c = float(np.sqrt(a * a + b * b))
        Z = np.array([-0.3, -0.35, 0.0])
        F1, F2 = Z + c * LEFT, Z + c * RIGHT
        n = np.linalg.norm

        def H(t, s=1):                                    # s = +1: the branch round F₂
            return Z + np.array([s * a * np.cosh(t), b * np.sinh(t), 0.0])

        def diff(X):
            return n(X - F1) - n(X - F2)

        tP = 0.85
        P = H(tP)
        vin = _unit(_unit(F1 - P) + _unit(F2 - P))        # ℓ, into the angle F₁PF₂
        u = vin if vin[1] > 0 else -vin                   # ℓ's direction, pointing up
        F2p = _refl(F2, P, vin)
        Hf = _foot(F2, P, P + vin)
        ext = P + 1.7 * _unit(P - F1)                     # F₁P produced: the reflected light
        tang = _unit(np.array([a * np.sinh(tP), b * np.cosh(tP), 0.0]))
        check(abs(diff(P) - 2 * a) < 1e-9, "PF₁ − PF₂ = 2a")
        check(abs(_ang(P, F1, P - u) - _ang(P, P - u, F2)) < 1e-9, "ℓ halves ∠F₁PF₂")
        check(abs(_ang(P, P + u, ext) - _ang(P, F1, P - u)) < 1e-9, "the vertical angle is θ too")
        check(abs(_cross(F2p - P, F1 - P)) < 1e-9 and np.dot(F2p - P, F1 - P) > 0
              and abs(n(F2p - P) - n(F2 - P)) < 1e-9, "the fold lays PF₂ along PF₁")
        check(abs(n(F1 - F2p) - 2 * a) < 1e-9, "F₂′F₁ = 2a")
        check(close(tang, u), "ℓ is the tangent of the branch at P")
        sQ = (-3.0, 1.5)                                  # Q's range along ℓ
        for s_ in np.linspace(sQ[0], sQ[1], 91):
            if abs(s_) > 1e-6:
                Q = P + s_ * u
                check(abs(n(Q - F2p) - n(Q - F2)) < 1e-9, "QF₂′ = QF₂")
                check(n(Q - F1) < n(Q - F2p) + n(F2p - F1) - 1e-9, "QF₁ < QF₂′ + F₂′F₁")
                check(abs(diff(Q)) < 2 * a - 1e-9, "every other point of ℓ is between the branches")
        din, dout = _unit(P - F2), _unit(ext - P)
        check(close(2 * np.dot(din, u) * u - din, dout), "light from F₂ reflects along F₁P produced")

        # ---- the two branches, the foci, P and its focal radii
        tm = 1.29
        ts = np.linspace(-tm, tm, 161)
        brR = _curve([H(t) for t in ts], WHITE, 3)
        brL = _curve([H(t, -1) for t in ts], GREY_B, 3)
        hyp_segs = (_poly_segs([H(t) for t in ts], closed=False)
                    + _poly_segs([H(t, -1) for t in ts], closed=False))
        rr = 0.55                                         # θ's arcs
        L_lo, L_hi = P + sQ[0] * u, P + sQ[1] * u         # ℓ as drawn
        segs = (hyp_segs + [_seg(F1, P), _seg(P, F2), _seg(L_lo, L_hi), _seg(F2, F2p), _seg(P, ext)]
                + _ra_segs(Hf, F2p, P, 0.15) + _tick_segs(F2, Hf) + _tick_segs(Hf, F2p)
                + _span_segs(P, F1, P - u, rr) + _span_segs(P, P - u, F2, rr)
                + _span_segs(P, P + u, ext, rr)
                + [s for X in (F1, F2, P, F2p) for s in _dot_segs(X)])
        dF1, dF2 = _dot(F1, YELLOW_B, 0.07), _dot(F2, YELLOW_B, 0.07)
        dP = _dot(P)
        r1 = Line(F1, P, color=BLUE_B, stroke_width=4)
        r2 = Line(P, F2, color=TEAL_B, stroke_width=4)
        lF1 = _park(tag("F₁", 28, YELLOW_B), F1, segs, DOWN, 0.42, step=12)
        lF2 = _park(tag("F₂", 28, YELLOW_B), F2, segs, RIGHT, 0.42, avoid=(lF1,), step=12)
        lP = _park(tag("P", 28), P, segs, _dir(125 * DEGREES), 0.4, avoid=(lF1, lF2), step=12)
        self.play(Create(brR), Create(brL), run_time=1.3)
        self.play(FadeIn(dF1), FadeIn(dF2), FadeIn(lF1), FadeIn(lF2), FadeIn(dP), FadeIn(lP),
                  run_time=0.5)
        self.play(Create(r1), Create(r2), run_time=0.8)
        self.bring_to_front(dF1, dF2, dP)

        # ---- the readout: QF₁ against QF₂ + 2a (equal at P)
        x0, y1, y2, sc = -4.3, 3.35, 2.85, 0.42
        s = ValueTracker(0.0)

        def Qpt():
            return P + s.get_value() * u

        def bars():
            Q = Qpt()
            d1, d2 = n(Q - F1), n(Q - F2)
            return VGroup(Line([x0, y1, 0], [x0 + sc * d1, y1, 0], color=BLUE_B, stroke_width=8),
                          Line([x0, y2, 0], [x0 + sc * d2, y2, 0], color=TEAL_B, stroke_width=8),
                          Line([x0 + sc * d2, y2, 0], [x0 + sc * (d2 + 2 * a), y2, 0],
                               color=WHITE, stroke_width=8))

        bar = bars()
        l1P, l2P = tag("PF₁", 24, BLUE_B), tag("PF₂ + 2a", 24)
        l1Q, l2Q = tag("QF₁", 24, ORANGE), tag("QF₂ + 2a", 24, ORANGE)
        for m, y in ((l1P, y1), (l2P, y2), (l1Q, y1), (l2Q, y2)):
            m.move_to([x0 - 0.22 - m.width / 2, y, 0])
        xe = x0 + sc * n(P - F1)
        endm = DashedLine([xe, y1 + 0.2, 0], [xe, y2 - 0.2, 0], color=GREY_B, stroke_width=2,
                          dash_length=0.06)
        self.play(FadeIn(bar), FadeIn(l1P), FadeIn(l2P), Create(endm), run_time=0.9)
        self.hold(0.4)

        # ---- ℓ halves ∠F₁PF₂; folded over it, PF₂ lies along PF₁
        ell = Line(L_lo, L_hi, color=YELLOW_B, stroke_width=4)
        lell = _park(tag("ℓ", 32, YELLOW_B), L_hi, segs, _perp(u), 0.3, avoid=(lP,), step=12)
        aw = VGroup(angle_arc(P, F1, P - u, rr, GREEN_B, 4), angle_arc(P, P - u, F2, rr, GREEN_B, 4))
        la = [tag("θ", 26, GREEN_B), tag("θ", 26, GREEN_B)]
        _angle_label(la[0], P, F1, P - u, rr + 0.1, segs, avoid=(lP, lell))
        _angle_label(la[1], P, P - u, F2, rr + 0.1, segs, r1=1.1,       # the branch runs
                     fracs=(0.5, 0.62, 0.72, 0.8), avoid=(lP, lell, la[0]))  # near ℓ here
        for m in la:
            check(n(m.get_center() - P) < rr + 0.55, "θ sits by its arc")
        self.play(Create(ell), FadeIn(lell), run_time=0.8)
        self.play(Create(aw), *[FadeIn(x) for x in la], run_time=0.7)
        self.bring_to_front(dP)
        mir = DashedLine(F2, F2p, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        raH = _ra(Hf, F2p, P, 0.15, GREY_A, 2)
        tk = VGroup(_ticks(F2, Hf, 1, GREY_A), _ticks(Hf, F2p, 1, GREY_A))
        self.play(Create(mir), Create(raH), FadeIn(tk), run_time=0.7)
        r2c = r2.copy()
        self.add(r2c)
        self.play(_fold(r2c, P, P + u), run_time=1.4)
        check(close(r2c.get_end(), F2p, 1e-6) or close(r2c.get_start(), F2p, 1e-6),
              "the fold carries F₂ onto F₂′")
        self.remove(r2c)
        r2p = Line(P, F2p, color=TEAL_B, stroke_width=6)
        dF2p = _dot(F2p, TEAL_B, 0.07)
        side = _perp(F1 - P)
        if side[1] < 0:
            side = -side
        lF2p = _park(tag("F₂′", 28, TEAL_B), F2p, segs, side, 0.42,
                     avoid=(lF1, lF2, lP, lell) + tuple(la), step=12)
        self.add(r2p)
        self.play(FadeIn(dF2p), FadeIn(lF2p), run_time=0.4)
        seg2a = Line(F2p, F1, color=WHITE, stroke_width=7)
        l2a = _beside(tag("2a", 28), F2p, F1, side, 0.14, 0.5)
        self.play(Create(seg2a), FadeIn(l2a), run_time=0.8)
        self.bring_to_front(dF1, dF2p, dP)
        self.hold(0.4)

        # ---- any other Q on ℓ: QF₁ < QF₂′ + F₂′F₁ = QF₂ + 2a, so Q is off the branch
        fixed = [lF1, lF2, lP, lell, lF2p, l2a]

        def qsegs(Q):
            return (hyp_segs + [_seg(F1, Q), _seg(Q, F2), _seg(Q, F2p), _seg(L_lo, L_hi),
                                _seg(F1, P), _seg(P, F2), _seg(F2, F2p)]
                    + _dot_segs(F1) + _dot_segs(F2) + _dot_segs(F2p) + _dot_segs(P)
                    + _dot_segs(Q, 0.09))

        def qop(sv):
            """Q's label fades out near P and where Q passes close to the
            branch and the fold's mirror line (just below P)."""
            if sv >= 0:
                return float(np.clip((sv - 0.4) / 0.35, 0.0, 1.0))
            return float(np.clip((-sv - 1.45) / 0.3, 0.0, 1.0))

        def qparts():
            Q = Qpt()
            g = VGroup(Line(F1, Q, color=BLUE_B, stroke_width=3),
                       Line(Q, F2, color=TEAL_B, stroke_width=3),
                       DashedLine(Q, F2p, color=TEAL_B, stroke_width=3, dash_length=0.08))
            if abs(s.get_value()) > 0.3:
                g.add(_ticks(Q, F2, 2, TEAL_B, at=0.55), _ticks(Q, F2p, 2, TEAL_B, at=0.55))
            g.add(_dot(Q, ORANGE, 0.08))
            return g

        def qlabel():
            Q = Qpt()
            op = qop(s.get_value())
            m = tag("Q", 28, ORANGE)
            base = _free_dir(Q, [F1, F2, F2p, Q + u, Q - u])
            if op > 0:
                _park(m, Q, qsegs(Q), base, 0.4, avoid=fixed, step=10)
            else:
                m.move_to(Q + 0.42 * base)
            return m.set_opacity(op)

        for sv in np.linspace(sQ[0], sQ[1], 91):
            s.set_value(sv)
            Q = Qpt()
            if qop(sv) > 0:
                ql = qlabel()                               # _park fails the render if no spot
                check(_hit(ql, qsegs(Q), 0.04) is None and _in_frame(ql),
                      f"Q's label clear of its lines (s = {sv:.2f})")
                check(not any(_overlap(ql, m, 0.03) for m in fixed),
                      f"Q's label clear of the other labels (s = {sv:.2f})")
            check(Q[1] < 2.6 and (Q[0] > -1.0 or Q[1] < 2.4), "Q's lines keep below the readout")
        s.set_value(0.0)
        qg, ql = always_redraw(qparts), always_redraw(qlabel)
        lbar = always_redraw(bars)
        self.remove(bar)
        self.add(lbar)
        self.play(r1.animate.set_stroke(opacity=0.3), r2.animate.set_stroke(opacity=0.3),
                  r2p.animate.set_stroke(opacity=0.3), aw.animate.set_stroke(opacity=0.25),
                  *[m.animate.set_opacity(0.25) for m in la], FadeIn(qg), FadeIn(ql),
                  ReplacementTransform(l1P, l1Q), ReplacementTransform(l2P, l2Q), run_time=0.6)
        self.play(s.animate.set_value(sQ[1]), run_time=1.5)
        self.play(s.animate.set_value(sQ[0]), run_time=2.8)
        self.play(s.animate.set_value(0.0), run_time=1.6)
        for m in (qg, ql, lbar):
            m.clear_updaters()
        self.play(FadeOut(qg), FadeOut(ql), FadeOut(lbar), FadeOut(l1Q), FadeOut(l2Q),
                  FadeOut(endm), r1.animate.set_stroke(opacity=1.0),
                  r2.animate.set_stroke(opacity=1.0), r2p.animate.set_stroke(opacity=1.0),
                  aw.animate.set_stroke(opacity=1.0), *[m.animate.set_opacity(1.0) for m in la],
                  run_time=0.6)
        self.bring_to_front(seg2a, dF1, dF2p, dP)

        # ---- so ℓ is the tangent; light from F₂ leaves along F₁P produced
        self.play(ShowPassingFlash(ell.copy().set_stroke(WHITE, 8), time_width=0.6), run_time=0.9)
        tip_in = F2 + 0.3 * (P - F2)
        lin = Arrow(F2, tip_in, buff=0.0, color=YELLOW_A, stroke_width=4,
                    max_tip_length_to_length_ratio=0.3)
        lout = Arrow(P, ext, buff=0.0, color=YELLOW_A, stroke_width=4,
                     max_tip_length_to_length_ratio=0.16)
        a3 = angle_arc(P, P + u, ext, rr, GREEN_B, 4)
        l3 = tag("θ", 26, GREEN_B)
        _angle_label(l3, P, P + u, ext, rr + 0.1, segs, avoid=fixed + la)
        check(n(l3.get_center() - P) < rr + 0.55, "θ sits by its arc")
        self.play(GrowArrow(lin), run_time=0.7)
        self.play(GrowArrow(lout), Create(a3), FadeIn(l3), run_time=0.9)
        self.bring_to_front(dP)

        heads = [_seg(X - 0.12 * RIGHT, X + 0.12 * RIGHT) for X in (tip_in, ext)]
        heads += [_seg(X - 0.12 * UP, X + 0.12 * UP) for X in (tip_in, ext)]
        _labels_ok(fixed + la + [l3], segs + heads, "E19")
        cap = caption("|PF₁ − PF₂| = 2a   ⟹   the tangent at P halves ∠F₁PF₂", 30)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E36  — the area of the Reuleaux triangle

def _pts_match(mob, pts, tol=1e-6):
    """A polygon mobject's corners are exactly the points pts (any order)."""
    vs = [to3(v) for v in mob.get_vertices()]
    ps = [to3(q) for q in pts]
    return (len(vs) == len(ps)
            and all(min(np.linalg.norm(v - q) for q in ps) < tol for v in vs)
            and all(min(np.linalg.norm(v - q) for v in vs) < tol for q in ps))


class E36_ReuleauxArea(Board):
    """The Reuleaux triangle on the equilateral triangle ABC of side s is
    the triangle plus three caps; the cap on each side is cut off by an arc
    of radius s about the opposite corner, so the triangle and one cap make
    a 60° sector of radius s. Take a copy and add two more triangles: the
    sector at B slid by BA and the sector at C turned half a turn about the
    midpoint of AC join the sector at A round the centre A — three 60°
    sectors, a half-disc of radius s. So A + 2·(√3/4)s² = ½πs², and
    A = ½(π − √3)s²."""

    def construct(self):
        s = 3.8
        h = s * np.sqrt(3) / 2
        n = np.linalg.norm
        A0 = np.array([-6.25, -2.2, 0.0])                   # the Reuleaux triangle
        A1 = A0 + (2 * s + 0.9) * RIGHT                      # the working copy (centre of the half-disc)

        def tri(A):
            return [A, A + s * RIGHT, A + s / 2 * RIGHT + h * UP]

        def caps(A):
            """Caps on BC, CA, AB: arcs about A, B, C (as closed polygons)."""
            A_, B_, C_ = tri(A)
            return [_arc_pts(A_, s, 0, PI / 3, 48), _arc_pts(B_, s, 2 * PI / 3, PI, 48),
                    _arc_pts(C_, s, 4 * PI / 3, 5 * PI / 3, 48)]

        A, B, C = tri(A0)
        cA, cB, cC = caps(A0)
        Rpts = cA[:-1] + cB[:-1] + cC[:-1]                  # the Reuleaux outline
        check(abs(abs(area(Rpts)) - 0.5 * (PI - np.sqrt(3)) * s * s) < 2e-3,
              "the Reuleaux triangle has area ½(π − √3)s²")
        check(_tiles_exactly([tri(A0), cA, cB, cC], Rpts, 90), "△ + three caps tile it")
        # the working copy: B1-sector slides by B1A1, C1-sector half-turns about mid A1C1
        A1_, B1_, C1_ = tri(A1)
        M1 = (A1_ + C1_) / 2
        k1, k2, k3 = caps(A1)
        shift = A1_ - B1_
        tB_dst = [p + shift for p in tri(A1)]
        kB_dst = [p + shift for p in k2]
        tC_dst = [_rot(p, M1, PI) for p in tri(A1)]
        kC_dst = [_rot(p, M1, PI) for p in k3]
        half = _arc_pts(A1_, s, 0, PI, 144)                 # the half-disc (chord closes it)
        secA = [A1_] + k1                                   # △ + cap on B1C1 = the 60° sector at A1
        secC = [A1_] + _arc_pts(A1_, s, PI / 3, 2 * PI / 3, 48)
        secB = [A1_] + _arc_pts(A1_, s, 2 * PI / 3, PI, 48)
        check(_tiles_exactly([tri(A1), k1], secA) and _tiles_exactly([tC_dst, kC_dst], secC)
              and _tiles_exactly([tB_dst, kB_dst], secB), "each moved pair is a 60° sector about A₁")
        check(_tiles_exactly([secA, secC, secB], half), "three 60° sectors tile the half-disc")
        check(all(abs(n(p - A1_) - s) < 1e-9 for p in kC_dst + kB_dst),
              "the moved caps' arcs lie on the circle of radius s about A₁")
        check(abs(abs(area(half)) - (abs(area(Rpts)) + 2 * np.sqrt(3) / 4 * s * s)) < 2e-3,
              "half-disc = Reuleaux + 2 triangles")

        # ---- the Reuleaux triangle: the triangle and three caps
        col_t, col_c, col_x = YELLOW_E, BLUE_D, ORANGE
        T0 = mk(tri(A0), col_t, 0.85)
        arcs = VGroup(*[_curve(cp, WHITE, 3) for cp in (cA, cB, cC)])
        caps0 = VGroup(*[mk(cp, col_c, 0.8, stroke_width=0) for cp in (cA, cB, cC)])
        ls = tag("s", 30, BLACK).move_to(A + s / 2 * RIGHT + 0.42 * UP)
        self.play(Create(T0), run_time=0.8)
        self.play(*[Create(x) for x in arcs], run_time=1.3)
        self.add(caps0)
        self.bring_to_back(caps0)
        caps0.set_fill(opacity=0)
        self.play(caps0.animate.set_fill(opacity=0.8), FadeIn(ls), run_time=0.7)
        lA = tag("A", 34).move_to(A + s / 2 * RIGHT + h / 2.6 * UP)
        self.play(FadeIn(lA), run_time=0.4)
        self.hold(0.4)

        # ---- a copy of it, to be cut up
        T1 = mk(tri(A1), col_t, 0.85)
        P1 = [mk(k, col_c, 0.8, stroke_color=WHITE, stroke_width=2) for k in (k1, k2, k3)]
        start = VGroup(T0.copy(), *[mk(k, col_c, 0.8, stroke_color=WHITE, stroke_width=2)
                                    for k in (cA, cB, cC)])
        self.add(start)
        self.play(start.animate.shift(A1 - A0), run_time=1.2)
        check(_pts_match(start[0], tri(A1)), "the copy lands on the working triangle")
        self.remove(start)
        self.add(T1, *P1)

        # ---- add a triangle: with the cap on C₁A₁ it is the 60° sector about B₁; slide it by B₁A₁
        X1 = mk(tri(A1), col_x, 0.85)
        self.play(FadeIn(X1), run_time=0.6)
        gB = VGroup(X1, P1[1])
        self.play(gB.animate.shift(shift), run_time=1.5)
        check(_pts_match(X1, tB_dst) and _pts_match(P1[1], kB_dst),
              "slid by B₁A₁, the sector about B₁ is the sector about A₁ from 120° to 180°")

        # ---- another triangle: with the cap on A₁B₁ it is the sector about C₁; half a turn about
        # the midpoint of A₁C₁
        X2 = mk(tri(A1), col_x, 0.85)
        piv = _dot(M1, WHITE, 0.05)
        self.play(FadeIn(X2), FadeIn(piv), run_time=0.6)
        gC = VGroup(X2, P1[2])
        self.play(Rotate(gC, angle=PI, about_point=M1), run_time=1.8)
        check(_pts_match(X2, tC_dst, 1e-6) and _pts_match(P1[2], kC_dst, 1e-6),
              "turned, the sector about C₁ is the sector about A₁ from 60° to 120°")
        self.play(FadeOut(piv), run_time=0.3)

        # ---- three 60° sectors round A₁: a half-disc of radius s
        rim = VGroup(_curve(half, WHITE, 4), Line(A1_ - s * RIGHT, A1_ + s * RIGHT,
                                                   color=WHITE, stroke_width=4))
        lhalf = tag("½πs²", 34).move_to(A1_ + (s + 0.45) * UP)
        lr = tag("s", 30, BLACK).move_to(A1_ + s / 2 * RIGHT + 0.42 * UP)
        self.play(Create(rim), FadeIn(lhalf), FadeIn(lr), run_time=1.0)
        lt1 = tag("(√3/4)s²", 24, BLACK).move_to(sum(tB_dst) / 3)
        lt2 = tag("(√3/4)s²", 24, BLACK).move_to(sum(tC_dst) / 3)
        self.play(FadeIn(lt1), FadeIn(lt2), run_time=0.6)
        for lab, T in ((lt1, tB_dst), (lt2, tC_dst), (lr, tri(A1)), (ls, tri(A0))):
            box = _box(lab)
            probes = [(x, y) for x in (box[0], box[1]) for y in (box[2], box[3])]
            check(all(_pip(p, T) for p in probes), f"'{_name(lab)}' inside its triangle")
        check(_in_frame(lhalf) and _hit(lhalf, _poly_segs(half, closed=False), 0.08) is None,
              "½πs² above the half-disc, clear of its rim")
        check(not any(_overlap(lhalf, m) for m in (lA, ls)), "labels apart")
        check(_pip(lA.get_center(), tri(A0)) and all(_pip(p, Rpts) for p in
              [(_box(lA)[i], _box(lA)[j]) for i in (0, 1) for j in (2, 3)]), "A inside the shape")
        cap = caption("A + 2 · (√3/4)s²  =  ½πs²     ⟹     A  =  ½(π − √3)s²", 32)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E26  — the Simson line

class E26_SimsonLine(Board):
    """P on the circumcircle of ABC, on the arc BC; its feet on the side
    lines are D (on BC), E (on CA) and F (on AB produced beyond B). Let γ be
    the angle PCA and δ = 180° − γ. ABPC is a cyclic quadrilateral, so its
    angle ABP opposite γ is δ, and the angle PBF beside it on the line ABF
    is γ. The right angles at D and F put them on the circle with diameter
    PB; sliding the vertex along that circle from B to D keeps the angle on
    the chord PF, so ∠PDF = γ. The right angles at D and E put them on the
    circle with diameter PC: PDCE is cyclic, so ∠PDE, opposite γ, is δ. At
    the middle foot D the two angles γ and δ add to 180°: DF and DE make a
    straight line, and D, E, F are collinear."""

    def construct(self):
        R = 2.85
        O = np.array([-0.2, 0.55, 0.0])
        n = np.linalg.norm
        A, B, C, P = (O + R * _dir(t * DEGREES) for t in (135, 205, 330, 263))
        D, E, F = _foot(P, B, C), _foot(P, C, A), _foot(P, A, B)
        tD = np.dot(D - B, C - B) / np.dot(C - B, C - B)
        tE = np.dot(E - C, A - C) / np.dot(A - C, A - C)
        tF = np.dot(F - A, B - A) / np.dot(B - A, B - A)
        check(0 < tD < 1 and 0 < tE < 1 and tF > 1, "D on BC, E on CA, F on AB beyond B")
        ga = _ang(C, P, A)
        de = PI - ga
        check(abs(_ang(B, A, P) - de) < 1e-9, "ABPC cyclic: ∠ABP = 180° − γ = δ")
        check(abs(_ang(B, P, F) - ga) < 1e-9, "∠PBF = γ (the straight line ABF)")
        c1, r1 = (P + B) / 2, n(P - B) / 2
        c2, r2 = (P + C) / 2, n(P - C) / 2
        check(abs(n(D - c1) - r1) < 1e-9 and abs(n(F - c1) - r1) < 1e-9,
              "D and F on the circle with diameter PB")
        check(abs(n(D - c2) - r2) < 1e-9 and abs(n(E - c2) - r2) < 1e-9,
              "D and E on the circle with diameter PC")
        hB, hD, hF, hP = (_heading(c1, X) for X in (B, D, F, P))
        arc1 = (hP - hD) % TAU                               # the circle on PB, drawn D→B→F→P
        check((hB - hD) % TAU < arc1 and (hF - hD) % TAU < arc1, "B and F on the drawn arc")
        # the vertex slides from B to D along the arc that does not hold P or F
        sweep = -((hB - hD) % TAU)
        for k in np.linspace(0, 1, 41):
            V = c1 + r1 * _dir(hB + k * sweep)
            check(abs(_ang(V, P, F) - ga) < 1e-9, "on that arc the angle on PF stays γ")
        check(abs(_ang(D, P, F) - ga) < 1e-9 and abs(_ang(D, P, E) - de) < 1e-9,
              "at D: ∠PDF = γ and ∠PDE = δ")
        check(abs(_cross(E - F, D - F)) < 1e-9 and 0 < np.dot(D - F, E - F) < np.dot(E - F, E - F),
              "D, E, F collinear, D between")
        hC2, hP2 = _heading(c2, C), _heading(c2, P)
        semi0 = hC2                                          # the semicircle on PC holding D and E
        for X in (D, E):
            check(0 < (_heading(c2, X) - semi0) % TAU < PI, "D, E on the drawn half of the circle PC")

        ra_s = 0.15
        rg = 0.38                                            # γ, δ arcs
        L0, L1 = F + 0.45 * _unit(F - E), E + 0.6 * _unit(E - F)   # the Simson line as drawn
        semi_pts = _arc_pts(c2, r2, semi0, semi0 + PI, 120)
        segs = (_arc_segs(O, R, 0, TAU, 160)
                + [_seg(A, B), _seg(B, C), _seg(C, A), _seg(B, F), _seg(P, B), _seg(P, C),
                   _seg(P, D), _seg(P, E), _seg(P, F), _seg(L0, L1)]
                + _ra_segs(D, P, C, ra_s) + _ra_segs(E, P, A, ra_s) + _ra_segs(F, P, A, ra_s)
                + _arc_segs(c1, r1, hD, hD + arc1, 72) + _poly_segs(semi_pts, closed=False)
                + _span_segs(C, P, A, rg) + _span_segs(B, A, P, rg) + _span_segs(B, P, F, rg)
                + _span_segs(D, P, F, rg) + _span_segs(D, P, E, rg)
                + [s for X in (A, B, C, P, D, E, F) for s in _dot_segs(X)])

        # ---- the triangle, its circumcircle and P on the arc BC
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3)
        dots = {k: _dot(X) for k, X in zip("ABC", (A, B, C))}
        labs = {}
        for k, X in zip("ABC", (A, B, C)):
            labs[k] = _park(tag(k, 28), X, segs, _unit(X - O), 0.4, avoid=list(labs.values()))
        self.play(Create(circ), Create(tri), *[FadeIn(m) for m in dots.values()],
                  *[FadeIn(m) for m in labs.values()], run_time=1.3)
        dP = _dot(P, YELLOW_B, 0.07)
        labs["P"] = _park(tag("P", 28, YELLOW_B), P, segs, _unit(P - O), 0.4,
                          avoid=list(labs.values()))
        self.play(FadeIn(dP), FadeIn(labs["P"]), run_time=0.5)

        # ---- the perpendiculars to the three side lines
        extF = DashedLine(B, F + 0.3 * _unit(F - B), color=GREY_A, stroke_width=2.5,
                          dash_length=0.08)
        perps = VGroup(*[Line(P, X, color=YELLOW_B, stroke_width=3) for X in (D, E, F)])
        ras = VGroup(_ra(D, P, C, ra_s, YELLOW_B, 2), _ra(E, P, A, ra_s, YELLOW_B, 2),
                     _ra(F, P, A, ra_s, YELLOW_B, 2))
        fdots = VGroup(*[_dot(X, YELLOW_B) for X in (D, E, F)])
        for k, X, base in (("D", D, UP), ("E", E, _unit(E - c2)), ("F", F, _unit(F - c1))):
            labs[k] = _park(tag(k, 28, YELLOW_B), X, segs, base, 0.4, avoid=list(labs.values()))
        self.play(Create(extF), run_time=0.5)
        self.play(Create(perps), run_time=1.0)
        self.play(Create(ras), FadeIn(fdots), *[FadeIn(labs[k]) for k in "DEF"], run_time=0.6)
        self.bring_to_front(dP)

        # ---- ABPC is cyclic: δ at B opposite γ at C, and γ beside δ on the line ABF
        pb = Line(P, B, color=BLUE_B, stroke_width=3)
        pc = Line(P, C, color=TEAL_B, stroke_width=3)
        self.play(Create(pb), Create(pc), run_time=0.7)
        quad = mk([A, B, P, C], GREY_D, 0.5, stroke_width=0)
        self.add(quad)
        self.bring_to_back(quad)
        quad.set_fill(opacity=0)
        self.play(quad.animate.set_fill(opacity=0.5), run_time=0.5)
        aC = angle_arc(C, P, A, rg, GREEN_B, 4)
        aBd = angle_arc(B, A, P, rg, ORANGE, 4)
        lgC = tag("γ", 28, GREEN_B)
        _angle_label(lgC, C, P, A, rg + 0.1, segs, fracs=(0.27, 0.22, 0.33),   # CB runs
                     avoid=list(labs.values()))                                 # through γ
        check(n(lgC.get_center() - C) < rg + 0.6, "γ sits by its arc at C")
        ldB = tag("δ", 28, ORANGE)
        _angle_label(ldB, B, A, P, rg + 0.1, segs, fracs=(0.82, 0.86, 0.78),   # BC runs
                     avoid=list(labs.values()) + [lgC])                         # through δ
        rule = tag("γ + δ = 180°", 30).move_to([-4.6, 3.2, 0.0])
        self.play(Create(aC), FadeIn(lgC), run_time=0.6)
        self.play(Create(aBd), FadeIn(ldB), FadeIn(rule), run_time=0.8)
        aBg = angle_arc(B, P, F, rg, GREEN_B, 4)
        lgB = tag("γ", 28, GREEN_B)
        _angle_label(lgB, B, P, F, rg + 0.1, segs, fracs=(0.3, 0.25, 0.35),    # the arc BP
                     avoid=list(labs.values()) + [lgC, ldB, rule])              # runs through γ
        check(n(ldB.get_center() - B) < rg + 0.6 and n(lgB.get_center() - B) < rg + 0.6,
              "δ and γ sit by their arcs at B")
        self.play(Create(aBg), FadeIn(lgB), quad.animate.set_fill(opacity=0), run_time=0.8)
        self.remove(quad)
        self.hold(0.3)

        # ---- the circle on PB holds D and F (right angles); the angle γ slides from B to D
        circ1 = DashedVMobject(Arc(radius=r1, start_angle=hD, angle=arc1, arc_center=c1,
                                   color=BLUE_B, stroke_width=2.5), num_dashes=40)
        self.play(Create(circ1), run_time=0.9)
        kv = ValueTracker(0.0)

        def vparts():
            V = c1 + r1 * _dir(hB + kv.get_value() * sweep)
            return VGroup(Line(V, P, color=GREEN_B, stroke_width=3),
                          Line(V, F, color=GREEN_B, stroke_width=3),
                          angle_arc(V, P, F, rg, GREEN_B, 4), _dot(V, GREEN_B, 0.07))

        vg = always_redraw(vparts)
        self.add(vg)
        self.play(kv.animate.set_value(1.0), run_time=2.2)
        vg.clear_updaters()
        aDg = angle_arc(D, P, F, rg, GREEN_B, 4)
        lgD = tag("γ", 28, GREEN_B)
        _angle_label(lgD, D, P, F, rg + 0.1, segs,
                     avoid=list(labs.values()) + [lgC, ldB, lgB, rule])
        df = Line(D, F, color=GREEN_B, stroke_width=3)
        self.play(FadeOut(vg), FadeIn(aDg), FadeIn(df), FadeIn(lgD), run_time=0.6)
        self.bring_to_front(fdots, dP)

        # ---- the circle on PC holds D and E: PDCE is cyclic, δ at D opposite γ at C
        semi = DashedVMobject(_curve(semi_pts, TEAL_B, 2.5), num_dashes=50)
        self.play(Create(semi), run_time=0.9)
        quad2 = mk([P, D, E, C], TEAL_E, 0.45, stroke_width=0)
        self.add(quad2)
        self.bring_to_back(quad2)
        quad2.set_fill(opacity=0)
        de_ = Line(D, E, color=ORANGE, stroke_width=3)
        aDd = angle_arc(D, P, E, rg, ORANGE, 4)
        ldD = tag("δ", 28, ORANGE)
        _angle_label(ldD, D, P, E, rg + 0.1, segs, fracs=(0.3, 0.25, 0.35),   # DC runs
                     avoid=list(labs.values()) + [lgC, ldB, lgB, lgD, rule])   # through δ
        check(n(ldD.get_center() - D) < rg + 0.6 and n(lgD.get_center() - D) < rg + 0.6,
              "γ and δ sit by their arcs at D")
        self.play(quad2.animate.set_fill(opacity=0.45), Create(de_), run_time=0.7)
        self.play(Create(aDd), FadeIn(ldD), run_time=0.6)
        self.play(quad2.animate.set_fill(opacity=0), run_time=0.5)
        self.remove(quad2)
        self.bring_to_front(fdots, dP)

        # ---- γ + δ = 180° at D: F, D, E on one line
        sline = Line(L0, L1, color=YELLOW_B, stroke_width=5)
        self.play(Create(sline), run_time=1.0)
        self.bring_to_front(fdots, dP)
        texts = list(labs.values()) + [lgC, ldB, lgB, lgD, ldD, rule]
        _labels_ok(texts, segs, "E26")
        cap = caption("γ + δ = 180°   ⟹   D, E, F collinear", 34)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E37  — the circle of Apollonius

class E37_ApolloniusCircle(Board):
    """P with PA : PB = k (k = 2 drawn). The bisector of the angle APB
    meets AB at C, and the bisector of the outer angle at P (between PB and
    AP produced) meets AB produced at D. By the angle bisector theorem C
    divides AB internally and D externally in the ratio PA : PB = k (C is
    as far from PA as from PB, so the triangles APC and CPB have areas as
    PA : PB and also as AC : CB) — so C and D do not depend on P. At P the
    angles α, α, β, β fill the straight angle on the line AP, so
    α + β = 90°: P sees CD at a right angle, and P lies on the circle with
    diameter CD. As P moves, the two bisectors swing about the fixed C and
    D, always at right angles."""

    def construct(self):
        k, u = 2.0, 1.45
        n = np.linalg.norm
        A = np.array([-6.1, 0.0, 0.0])
        B = A + 3 * u * RIGHT
        C = A + 2 * u * RIGHT                                 # AC : CB = 2 : 1
        D = A + 6 * u * RIGHT                                 # AD : DB = 2 : 1
        M, rad = (C + D) / 2, 2 * u

        def Ppt(phi):
            return M + rad * _dir(phi)

        def parts_at(P):
            bi = _unit(_unit(A - P) + _unit(B - P))            # inner bisector
            bo = _unit(_unit(B - P) + _unit(P - A))            # outer bisector
            return bi, bo

        for phi in np.linspace(25, 160, 55) * DEGREES:
            P = Ppt(phi)
            bi, bo = parts_at(P)
            check(abs(n(P - A) / n(P - B) - k) < 1e-9, "on the circle PA : PB = k")
            check(abs(_cross(C - P, bi)) < 1e-9 and abs(_cross(D - P, bo)) < 1e-9,
                  "the bisectors meet AB at the fixed points C and D")
            check(abs(np.dot(C - P, D - P)) < 1e-9, "∠CPD = 90°")
        check(abs(n(C - A) / n(B - C) - k) < 1e-12 and abs(n(D - A) / n(D - B) - k) < 1e-12,
              "AC : CB = AD : DB = k")
        phi0 = 112 * DEGREES
        P = Ppt(phi0)
        X = P + 1.45 * _unit(P - A)                          # AP produced
        al = _ang(P, A, C)
        be = _ang(P, B, D)
        check(abs(_ang(P, C, B) - al) < 1e-9 and abs(_ang(P, D, X) - be) < 1e-9,
              "the two bisectors halve the angle and the outer angle")
        check(abs(2 * al + 2 * be - PI) < 1e-9, "α + α + β + β = 180°")
        HA, HB = _foot(C, A, P), _foot(C, B, P)
        check(abs(n(HA - C) - n(HB - C)) < 1e-9, "C is as far from PA as from PB")
        check(abs(area([A, P, C]) / area([C, P, B]) - k) < 1e-9, "areas APC : CPB = k")

        ra = 0.9                                              # α arcs (α is narrow)
        rb = 0.62                                             # β arcs
        lineAD = Line(A + 0.3 * LEFT, D + 0.45 * RIGHT, color=GREY_B, stroke_width=2.5)
        segs = ([_seg(A + 0.3 * LEFT, D + 0.45 * RIGHT), _seg(P, A), _seg(P, B), _seg(P, C),
                 _seg(P, D), _seg(P, X), _seg(C, HA), _seg(C, HB)]
                + _ra_segs(HA, C, P, 0.14) + _ra_segs(HB, C, P, 0.14)
                + _tick_segs(C, HA) + _tick_segs(C, HB)
                + _span_segs(P, A, C, ra) + _span_segs(P, C, B, ra)
                + _span_segs(P, B, D, rb) + _span_segs(P, D, X, rb) + _ra_segs(P, C, D, 0.2)
                + [s for Y in (A, B, C, D, P) for s in _dot_segs(Y)])  # (the circle comes later)

        segs_pts = segs + _arc_segs(M, rad, 0, TAU, 160)     # point labels clear the circle too

        # ---- A, B and a point P with PA : PB = k
        dA, dB = _dot(A), _dot(B)
        lA = _park(tag("A", 28), A, segs_pts, DOWN, 0.4)
        lB = _park(tag("B", 28), B, segs_pts, DOWN, 0.4)
        self.play(Create(lineAD), FadeIn(dA), FadeIn(dB), FadeIn(lA), FadeIn(lB), run_time=0.9)
        dP = _dot(P, YELLOW_B, 0.07)
        lP = _park(tag("P", 28, YELLOW_B), P, segs, UP, 0.4, avoid=(lA, lB))
        pa = Line(P, A, color=BLUE_B, stroke_width=3.5)
        pb = Line(P, B, color=TEAL_B, stroke_width=3.5)
        rows = VGroup(tag("PA : PB = k", 28), tag("AC : CB = k", 28, GREEN_B),
                      tag("AD : DB = k", 28, ORANGE)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        rows.move_to([-6.45 + rows.width / 2, 3.25 - rows.height / 2 + 0.15, 0.0])
        self.play(Create(pa), Create(pb), FadeIn(dP), FadeIn(lP), FadeIn(rows[0]), run_time=0.9)
        self.bring_to_front(dA, dB, dP)

        # ---- the inner bisector meets AB at C: C is as far from PA as from PB
        pc = Line(P, C, color=GREEN_B, stroke_width=3.5)
        aA1, aA2 = angle_arc(P, A, C, ra, GREEN_B, 4), angle_arc(P, C, B, ra, GREEN_B, 4)
        la1, la2 = tag("α", 26, GREEN_B), tag("α", 26, GREEN_B)
        _angle_label(la1, P, A, C, ra + 0.1, segs, avoid=(lP,))
        _angle_label(la2, P, C, B, ra + 0.1, segs, avoid=(lP, la1))
        for m in (la1, la2):
            check(n(m.get_center() - P) < ra + 0.45, "α sits by its arc")
        dC = _dot(C, GREEN_B, 0.07)
        lC = _park(tag("C", 28, GREEN_B), C, segs_pts, _dir(-125 * DEGREES), 0.42, avoid=(lA, lB))
        self.play(Create(aA1), Create(aA2), FadeIn(la1), FadeIn(la2), run_time=0.7)
        self.play(Create(pc), FadeIn(dC), FadeIn(lC), run_time=0.8)
        perpC = VGroup(DashedLine(C, HA, color=GREEN_B, stroke_width=2.5, dash_length=0.07),
                       DashedLine(C, HB, color=GREEN_B, stroke_width=2.5, dash_length=0.07),
                       _ra(HA, C, P, 0.14, GREEN_B, 2), _ra(HB, C, P, 0.14, GREEN_B, 2),
                       _ticks(C, HA, 1, GREEN_B), _ticks(C, HB, 1, GREEN_B))
        tAPC = mk([A, P, C], BLUE_D, 0.35, stroke_width=0)
        tCPB = mk([C, P, B], TEAL_D, 0.35, stroke_width=0)
        self.add(tAPC, tCPB)
        self.bring_to_back(tAPC, tCPB)
        tAPC.set_fill(opacity=0)
        tCPB.set_fill(opacity=0)
        self.play(Create(perpC), tAPC.animate.set_fill(opacity=0.35),
                  tCPB.animate.set_fill(opacity=0.35), run_time=1.0)
        self.play(FadeIn(rows[1]), run_time=0.6)
        self.play(FadeOut(perpC), tAPC.animate.set_fill(opacity=0), tCPB.animate.set_fill(opacity=0),
                  run_time=0.6)
        self.remove(tAPC, tCPB)
        self.bring_to_front(dC, dP)

        # ---- the outer bisector meets AB produced at D
        ext = DashedLine(P, X, color=BLUE_B, stroke_width=2.5, dash_length=0.08)
        pd = Line(P, D, color=ORANGE, stroke_width=3.5)
        aB1, aB2 = angle_arc(P, B, D, rb, ORANGE, 4), angle_arc(P, D, X, rb, ORANGE, 4)
        lb1, lb2 = tag("β", 26, ORANGE), tag("β", 26, ORANGE)
        _angle_label(lb1, P, B, D, rb + 0.1, segs, avoid=(lP, la1, la2))
        _angle_label(lb2, P, D, X, rb + 0.1, segs, avoid=(lP, la1, la2, lb1))
        for m in (lb1, lb2):
            check(n(m.get_center() - P) < rb + 0.45, "β sits by its arc")
        dD = _dot(D, ORANGE, 0.07)
        lD = _park(tag("D", 28, ORANGE), D, segs_pts, _dir(-55 * DEGREES), 0.42,
                   avoid=(lA, lB, lC))
        self.play(Create(ext), run_time=0.5)
        self.play(Create(aB1), Create(aB2), FadeIn(lb1), FadeIn(lb2), run_time=0.7)
        self.play(Create(pd), FadeIn(dD), FadeIn(lD), run_time=0.8)
        self.play(FadeIn(rows[2]), run_time=0.6)
        self.bring_to_front(dD, dP)

        # ---- α + α + β + β = 180°: the bisectors are at right angles
        flash = VGroup(aA1.copy(), aA2.copy(), aB1.copy(), aB2.copy())
        self.play(*[ShowPassingFlash(x.set_stroke(WHITE, 8), time_width=0.8) for x in flash],
                  run_time=0.9)
        raP = _ra(P, C, D, 0.2, WHITE, 2.5)
        self.play(Create(raP), run_time=0.5)
        texts = [lA, lB, lC, lD, lP, la1, la2, lb1, lb2] + list(rows)
        _labels_ok(texts, segs, "E37")
        self.hold(0.4)

        # ---- C and D depend only on k: P moves, the bisectors swing about C and D
        ph = ValueTracker(phi0)

        def live():
            Q = Ppt(ph.get_value())
            return VGroup(Line(Q, A, color=BLUE_B, stroke_width=3.5),
                          Line(Q, B, color=TEAL_B, stroke_width=3.5),
                          Line(Q, C, color=GREEN_B, stroke_width=3.5),
                          Line(Q, D, color=ORANGE, stroke_width=3.5),
                          _ra(Q, C, D, 0.2, WHITE, 2.5), _dot(Q, YELLOW_B, 0.07))

        def live_lab():
            Q = Ppt(ph.get_value())
            return tag("P", 28, YELLOW_B).move_to(Q + 0.42 * _unit(Q - M))

        for phv in np.linspace(40, 150, 23) * DEGREES:       # P's label stays clear as it moves
            Q = Ppt(phv)
            ll = live_lab()
            check(_hit(ll, [_seg(Q, A), _seg(Q, B), _seg(Q, C), _seg(Q, D)]
                       + _arc_segs(M, rad, 0, TAU, 120), 0.05) is None and _in_frame(ll),
                  f"P's label clear while P moves ({np.degrees(phv):.0f}°)")
            check(not any(_overlap(ll, m) for m in rows), "P's label clear of the rows")
        trace = TracedPath(lambda: Ppt(ph.get_value()), stroke_color=YELLOW_B, stroke_width=3)
        lv, lvl = always_redraw(live), always_redraw(live_lab)
        self.play(FadeOut(VGroup(pa, pb, pc, pd, ext, raP, aA1, aA2, aB1, aB2, la1, la2, lb1,
                                 lb2, dP, lP)), FadeIn(lv), FadeIn(lvl), run_time=0.6)
        self.add(trace)
        self.bring_to_front(dC, dD, dA, dB)
        self.play(ph.animate.set_value(40 * DEGREES), run_time=1.8)
        self.play(ph.animate.set_value(150 * DEGREES), run_time=2.8)
        self.play(ph.animate.set_value(phi0), run_time=1.2)
        lv.clear_updaters()
        lvl.clear_updaters()

        # ---- P sees CD at a right angle: the circle with diameter CD
        circ = Circle(radius=rad, color=YELLOW_B, stroke_width=4).move_to(M)
        self.play(Create(circ), FadeOut(trace), run_time=1.2)
        self.bring_to_front(lv, dC, dD, dA, dB)
        _labels_ok([lA, lB, lC, lD, lvl] + list(rows),
                   [_seg(A + 0.3 * LEFT, D + 0.45 * RIGHT), _seg(P, A), _seg(P, B), _seg(P, C),
                    _seg(P, D)] + _arc_segs(M, rad, 0, TAU, 160) + _ra_segs(P, C, D, 0.2)
                   + [s for Y in (A, B, C, D, P) for s in _dot_segs(Y)], "E37 final")
        cap = caption("PA : PB = k   ⟹   ∠CPD = 90°:  P on the circle with diameter CD", 30)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E22  — the salinon

class E22_Salinon(Board):
    """Archimedes' salinon: on AB the big semicircle, less the semicircles
    on AC and DB (AC = DB), plus the semicircle on CD below. A semicircle
    on d has area π/8·d², so the salinon is π/8·(AB² + CD² − AC² − DB²).
    Its axis EF, turned a quarter turn about the centre O, lies on AD
    (OE = OA, OF = OD): EF = AD. In the square on AB = AD + DB put two
    squares on AD in opposite corners: they overlap in the square on
    AD − DB = CD and leave the squares on DB and AC = DB in the other two
    corners, so AB² + CD² = 2·AD² + AC² + DB². Hence the salinon is
    π/8·2·EF² = π(EF/2)², the circle on its axis."""

    def construct(self):
        x_, y_ = 1.2, 1.6                                   # AC = DB, CD (math units)
        k = 1.45
        n = np.linalg.norm
        AB, AD, EF = 2 * x_ + y_, x_ + y_, x_ + y_
        A = np.array([-6.45, 0.35, 0.0])
        C, D, B = A + k * x_ * RIGHT, A + k * (x_ + y_) * RIGHT, A + k * AB * RIGHT
        O = (A + B) / 2
        Rb, rs, rc = k * AB / 2, k * x_ / 2, k * y_ / 2       # big, small, lower radii
        O1, O2 = (A + C) / 2, (D + B) / 2
        E, F = O + Rb * UP, O + rc * DOWN
        check(close((C + D) / 2, O), "the lower semicircle is centred at O too")
        check(close(_rot(E, O, PI / 2), A) and close(_rot(F, O, PI / 2), D),
              "the quarter turn about O takes E to A and F to D: EF = AD")
        sal = (_arc_pts(O, Rb, PI, 0, 240) + _arc_pts(O2, rs, 0, PI, 60)[1:]
               + _arc_pts(O, rc, 0, -PI, 120)[1:] + _arc_pts(O1, rs, 0, PI, 60)[1:-1])
        sal_area = PI / 8 * (AB ** 2 + y_ ** 2 - 2 * x_ ** 2) * k * k
        check(abs(abs(area(sal)) - sal_area) < 2e-3, "salinon = π/8·(AB² + CD² − AC² − DB²)")
        check(abs(sal_area - PI * (k * EF / 2) ** 2) < 1e-9, "… = π(EF/2)²")

        # the square on AB, the two squares on AD in opposite corners
        S, a_, b_, c_ = k * AB, k * AD, k * x_, k * y_
        x0, y0 = 0.7, 2.95                                   # top-left corner

        def P2(x, y):
            return np.array([x, y, 0.0])

        def rect(xa, ya, xb, yb):
            return [P2(xa, ya), P2(xb, ya), P2(xb, yb), P2(xa, yb)]

        sq = rect(x0, y0, x0 + S, y0 - S)
        q1 = rect(x0, y0, x0 + a_, y0 - a_)
        q2 = rect(x0 + S - a_, y0 - S + a_, x0 + S, y0 - S)
        ov = rect(x0 + S - a_, y0 - S + a_, x0 + a_, y0 - a_)
        cTR = rect(x0 + a_, y0, x0 + S, y0 - S + a_)
        cBL = rect(x0, y0 - a_, x0 + S - a_, y0 - S)
        L2 = [P2(x0 + a_, y0 - S + a_), P2(x0 + S, y0 - S + a_), P2(x0 + S, y0 - S),
              P2(x0 + S - a_, y0 - S), P2(x0 + S - a_, y0 - a_), P2(x0 + a_, y0 - a_)]
        check(abs(abs(area(ov)) - c_ ** 2) < 1e-9 and abs(n(ov[1] - ov[0]) - c_) < 1e-9
              and abs(n(ov[2] - ov[1]) - c_) < 1e-9, "the overlap is the square on CD")
        check(abs(abs(area(cTR)) - b_ ** 2) < 1e-9 and abs(abs(area(cBL)) - b_ ** 2) < 1e-9,
              "the free corners are the squares on DB and AC")
        check(_tiles_exactly([q1, L2, cTR, cBL], sq, 90),
              "AD² + (AD² less CD²) + DB² + AC² tile the square on AB")
        check(_tiles_exactly([ov, L2], q2, 60), "the second AD² is the overlap CD² plus the L")
        check(close(q2[0], q1[0] + (S - a_) * RIGHT + (S - a_) * DOWN),
              "the second AD² is the first slid by DB along the diagonal")

        # ---- the salinon
        base = Line(A, B, color=WHITE, stroke_width=3)
        dots = VGroup(*[_dot(X) for X in (A, C, D, B)])
        self.play(Create(base), FadeIn(dots), run_time=0.8)
        big = _curve(_arc_pts(O, Rb, PI, 0, 160), WHITE, 3)
        smA = _curve(_arc_pts(O1, rs, PI, 0, 60), WHITE, 3)
        smB = _curve(_arc_pts(O2, rs, PI, 0, 60), WHITE, 3)
        low = _curve(_arc_pts(O, rc, PI, 2 * PI, 90), WHITE, 3)
        self.play(Create(big), run_time=0.9)
        self.play(Create(smA), Create(smB), Create(low), run_time=0.9)
        salm = mk(sal, YELLOW_E, 0.7, stroke_width=0)
        self.add(salm)
        self.bring_to_back(salm)
        salm.set_fill(opacity=0)
        self.play(salm.animate.set_fill(opacity=0.7), run_time=0.7)

        # ---- its axis EF, turned a quarter turn about O, lies along AD
        axis = DashedLine(F, E, color=ORANGE, stroke_width=3.5, dash_length=0.09)
        dE, dF, dO = _dot(E), _dot(F), _dot(O, WHITE, 0.05)
        self.play(Create(axis), FadeIn(dE), FadeIn(dF), run_time=0.7)
        rod = Line(F, E, color=ORANGE, stroke_width=7)
        self.play(FadeIn(rod), FadeIn(dO), run_time=0.3)
        self.play(Rotate(rod, angle=PI / 2, about_point=O), run_time=1.4)
        check((close(rod.get_start(), D, 1e-6) and close(rod.get_end(), A, 1e-6))
              or (close(rod.get_start(), A, 1e-6) and close(rod.get_end(), D, 1e-6)),
              "turned, EF lies exactly on AD")
        self.bring_to_front(dots, dO)                        # EF now lies on AD, in orange
        self.hold(0.3)

        cc = (E + F) / 2                                     # the circle on the axis (shown last)
        segs_s = (_poly_segs(sal) + [_seg(A, B), _seg(F, E)]
                  + [s for X in (A, C, D, B, E, F, O) for s in _dot_segs(X)]
                  + _arc_segs(cc, k * EF / 2, 0, TAU, 120))
        labs = {}
        for name, X, base_ in (("A", A, DOWN), ("C", C, DOWN + 0.6 * LEFT), ("D", D, DOWN + 0.6 * RIGHT),
                               ("B", B, DOWN), ("E", E, UP), ("F", F, DOWN)):
            labs[name] = _park(tag(name, 28), X, segs_s, _unit(base_), 0.36,
                               avoid=list(labs.values()), step=12)
        self.play(*[FadeIn(m) for m in labs.values()], run_time=0.5)

        # ---- a semicircle on d is π/8·d²; so the salinon is π/8·(AB² + CD² − AC² − DB²)
        ic = np.array([-6.05, -1.75, 0.0])
        icon = VGroup(mk(_arc_pts(ic, 0.36, 0, PI, 40), GREY_B, 0.6, stroke_width=2),
                      tag("d", 22).move_to(ic + 0.22 * DOWN))
        ieq = tag("=  π/8 · d²", 26).next_to(icon, RIGHT, buff=0.25).align_to(icon[0], DOWN)
        row1 = tag("salinon = π/8 · (AB² + CD² − AC² − DB²)", 23)
        row1.move_to([A[0] + row1.width / 2, -2.55, 0.0])
        self.play(FadeIn(icon), FadeIn(ieq), run_time=0.7)
        self.play(FadeIn(row1), run_time=0.7)

        # ---- the square on AB with two squares on AD in opposite corners
        frame = Polygon(*sq, stroke_color=WHITE, stroke_width=3)
        eAD = tag("AD", 24, ORANGE).move_to([x0 + a_ / 2, y0 + 0.25, 0])
        eDB = tag("DB", 24).move_to([x0 + a_ + (S - a_) / 2, y0 + 0.25, 0])
        tick = Line(P2(x0 + a_, y0 + 0.12), P2(x0 + a_, y0 - 0.12), color=WHITE, stroke_width=3)
        self.play(Create(frame), FadeIn(eAD), FadeIn(eDB), Create(tick), run_time=0.9)
        m1 = mk(q1, BLUE_D, 0.5, stroke_color=BLUE_B, stroke_width=4)
        m2 = mk(q1, BLUE_D, 0.5, stroke_color=BLUE_B, stroke_width=4)
        self.play(FadeIn(m1), run_time=0.6)
        self.add(m2)
        self.play(m2.animate.shift(q2[0] - q1[0]), run_time=1.4)
        check(_pts_match(m2, q2), "slid, the copy is the square on AD in the far corner")
        lq1 = tag("AD²", 30).move_to((q1[0] + P2(x0 + S - a_, y0 - S + a_)) / 2)
        lq2 = tag("AD²", 30).move_to((q2[2] + P2(x0 + a_, y0 - a_)) / 2)
        lov = tag("CD²", 30, YELLOW_B).move_to(sum(ov) / 4)
        mov = Polygon(*ov, stroke_color=YELLOW_B, stroke_width=4)
        self.play(FadeIn(lq1), FadeIn(lq2), run_time=0.5)
        self.play(Create(mov), FadeIn(lov), run_time=0.7)
        mTR, mBL = mk(cTR, TEAL_D, 0.75), mk(cBL, TEAL_D, 0.75)
        lTR = tag("DB²", 28).move_to(sum(cTR) / 4)
        lBL = tag("AC²", 28).move_to(sum(cBL) / 4)
        self.play(FadeIn(mTR), FadeIn(mBL), FadeIn(lTR), FadeIn(lBL), run_time=0.7)
        self.bring_to_front(frame, mov, lov)
        self.hold(0.5)

        # ---- the salinon is π/8 · 2·EF²: the circle on its axis
        disc = Circle(radius=k * EF / 2, color=ORANGE, stroke_width=4).move_to(cc)
        disc.set_fill(ORANGE, opacity=0.25)
        self.play(Create(disc), run_time=1.2)
        self.bring_to_front(axis, dE, dF, dO, *labs.values())

        for lab, box in ((lq1, [q1[0], P2(x0 + S - a_, y0), P2(x0 + S - a_, y0 - S + a_), q1[3]]),
                         (lq2, [P2(x0 + a_, y0 - a_), P2(x0 + S, y0 - a_), q2[2],
                                P2(x0 + a_, y0 - S)]),
                         (lov, ov), (lTR, cTR), (lBL, cBL)):
            bx = _box(lab)
            probes = [(xx, yy) for xx in (bx[0], bx[1]) for yy in (bx[2], bx[3])]
            check(all(_pip(p_, box) for p_ in probes), f"'{_name(lab)}' inside its piece")
        segs = (segs_s + _poly_segs(sq)
                + _poly_segs(ov) + _poly_segs(q1) + _poly_segs(q2) + _poly_segs(cTR) + _poly_segs(cBL)
                + _poly_segs(_arc_pts(ic, 0.36, 0, PI, 40)) + [_seg(tick.get_start(), tick.get_end())])
        _labels_ok(list(labs.values()) + [eAD, eDB, ieq, icon[1], row1], segs, "E22")
        check(not any(_overlap(row1, m) for m in (lq1, lq2, lov, lTR, lBL)), "rows apart")
        cap = caption("salinon  =  π/8 · (AB² + CD² − AC² − DB²)  =  π/8 · 2 EF²  =  π (EF/2)²", 28)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E31  — the tangent to y = x²

class E31_ParabolaTangent(Board):
    """Every point of y = x² is as far from F = (0, ¼) as from the line d:
    y = −¼ (x² + (x² − ¼)² = (x² + ¼)²). For P = (a, a²) with foot D on d,
    PF = PD, so the perpendicular bisector ℓ of FD passes through P; any
    other point Q of ℓ has QF = QD, longer than its distance QQ′ to d, so Q
    lies below the curve: ℓ is the tangent at P. F is ¼ above the x-axis
    and D is ¼ below it, so ℓ crosses the axis at the midpoint M of FD:
    M = (a/2, 0) — the subtangent is half the abscissa. A half-turn about M
    keeps ℓ and carries the triangle P(a, a²), N(a, 0), M onto T, the
    origin V, M: T = (0, −a²) is on ℓ. So ℓ rises a² over a/2: y = 2ax − a²."""

    def construct(self):
        a = 0.7
        n = np.linalg.norm
        Fr = Frame(-1.25, 1.5, -0.86, 1.32)
        S = Fr.P

        def f(x):
            return x * x

        Fm, Pm, Dm = np.array([0.0, 0.25]), np.array([a, a * a]), np.array([a, -0.25])
        Mm, Nm, Vm, Tm = np.array([a / 2, 0.0]), np.array([a, 0.0]), np.array([0.0, 0.0]), \
            np.array([0.0, -a * a])
        t = _unit(np.array([1.0, 2 * a, 0.0]))[:2]          # slope f′(a) = 2a
        check(all(abs(n(np.array([x, f(x)]) - Fm) - (f(x) + 0.25)) < 1e-12
                  for x in np.linspace(-1.2, 1.2, 49)), "y = x²: XF = distance to d")
        check(abs(n(Pm - Fm) - n(Pm - Dm)) < 1e-12, "PF = PD")
        check(close((Fm + Dm) / 2, Mm) and abs(Mm[1]) < 1e-12, "the midpoint of FD is (a/2, 0)")
        check(abs(np.dot(Dm - Fm, t)) < 1e-12 and abs(_cross(np.r_[Pm - Mm, 0], np.r_[t, 0])) < 1e-12,
              "ℓ ⊥ FD through M, and P on it")
        for s_ in np.linspace(-1.2, 0.9, 43):
            if abs(s_) > 1e-6:
                Q = Pm + s_ * t
                check(n(Q - Fm) > Q[1] + 0.25 + 1e-12 and Q[1] < f(Q[0]),
                      "every other point of ℓ is below the curve")
        check(close(2 * Mm - Pm, Tm) and close(2 * Mm - Nm, Vm) and close(2 * Mm - Fm, Dm),
              "the half-turn about M: P→T, N→V, F→D")
        check(abs(_cross(np.r_[Tm - Mm, 0], np.r_[t, 0])) < 1e-12, "T on ℓ")
        check(abs((Pm[1] - Mm[1]) / (Pm[0] - Mm[0]) - 2 * a) < 1e-12 and abs(Tm[1] + a * a) < 1e-12,
              "slope a² ÷ (a/2) = 2a, intercept −a²")

        F, P, D, M, N, V, T = (S(p) for p in (Fm, Pm, Dm, Mm, Nm, Vm, Tm))
        xs = np.linspace(-1.12, 1.12, 161)
        curve = _curve([S((x, f(x))) for x in xs], WHITE, 4)
        xax = Arrow(S((-1.22, 0)), S((1.48, 0)), buff=0, color=GREY_B, stroke_width=2.5,
                    max_tip_length_to_length_ratio=0.03)
        yax = Arrow(S((0, -0.84)), S((0, 1.3)), buff=0, color=GREY_B, stroke_width=2.5,
                    max_tip_length_to_length_ratio=0.04)
        dline = Line(S((-1.22, -0.25)), S((1.48, -0.25)), color=GREY_B, stroke_width=2.5)
        L0, L1 = S(Pm - 1.25 * t), S(Pm + 0.95 * t)
        segs = ([_seg(S((-1.22, 0)), S((1.48, 0))), _seg(S((0, -0.84)), S((0, 1.3))),
                 _seg(S((-1.22, -0.25)), S((1.48, -0.25))), _seg(L0, L1),
                 _seg(P, F), _seg(P, D), _seg(F, D), _seg(P, N), _seg(P, S((0, a * a)))]
                + _poly_segs([S((x, f(x))) for x in xs], closed=False)
                + _ra_segs(M, F, L1, 0.15) + _ra_segs(D, P, S((a + 1, -0.25)), 0.15)
                + _tick_segs(P, F, 2) + _tick_segs(P, D, 2) + _tick_segs(F, M) + _tick_segs(M, D)
                + [s for X in (F, P, D, M, N, V, T) for s in _dot_segs(X)])

        # ---- y = x² and P = (a, a²)
        self.play(Create(xax), Create(yax), run_time=0.8)
        self.play(Create(curve), run_time=1.2)
        dP = _dot(P)
        lP = _park(tag("P", 28), P, segs, RIGHT, 0.36, step=12)
        dropx = DashedLine(P, N, color=GREY_A, stroke_width=2, dash_length=0.07)
        dropy = DashedLine(P, S((0, a * a)), color=GREY_A, stroke_width=2, dash_length=0.07)
        la = _park(tag("a", 26), N, segs, UP + 0.4 * RIGHT, 0.3, avoid=(lP,), step=10)
        la2 = _park(tag("a²", 26), S((0, a * a)), segs, LEFT, 0.36, avoid=(lP,), step=10)
        self.play(FadeIn(dP), FadeIn(lP), Create(dropx), Create(dropy), FadeIn(la), FadeIn(la2),
                  run_time=0.9)

        # ---- the focus F = (0, ¼) and the line d: y = −¼; every point is as far from both
        dF = _dot(F, YELLOW_B, 0.07)
        x_rest = (-1.05, 0.9)                                # where the sample point X rests
        segs_x = segs + [s_ for xr in x_rest for s_ in
                         (_seg(S((xr, f(xr))), F), _seg(S((xr, f(xr))), S((xr, -0.25))))]
        lF = _park(tag("F", 28, YELLOW_B), F, segs_x, LEFT, 0.36, avoid=(lP, la, la2), step=8)
        ld = tag("d", 28, GREY_B).move_to(S((-1.1, -0.25)) + 0.3 * DOWN)
        lq1 = _park(tag("¼", 24, GREY_A), S((0, 0.125)), segs, LEFT, 0.26, avoid=(lF, la2), step=10)
        lq2 = _park(tag("¼", 24, GREY_A), S((0, -0.125)), segs, LEFT, 0.26, avoid=(lF, la2, lq1),
                    step=10)
        self.play(FadeIn(dF), FadeIn(lF), Create(dline), FadeIn(ld), FadeIn(lq1), FadeIn(lq2),
                  run_time=0.9)
        xv = ValueTracker(x_rest[0])

        def xparts():
            X = np.array([xv.get_value(), f(xv.get_value())])
            Xd = np.array([X[0], -0.25])
            return VGroup(Line(S(X), F, color=ORANGE, stroke_width=3),
                          Line(S(X), S(Xd), color=ORANGE, stroke_width=3),
                          _ticks(S(X), F, 1, ORANGE), _ticks(S(X), S(Xd), 1, ORANGE),
                          _dot(S(X), ORANGE, 0.07))

        for xr in x_rest:                                    # where X rests: its lines clear the labels
            X = np.array([xr, f(xr)])
            sg = [_seg(S(X), F), _seg(S(X), S((xr, -0.25)))]
            for m in (lP, la, la2, lF, ld, lq1, lq2):
                check(_hit(m, sg, 0.05) is None, f"X's lines clear of '{_name(m)}' (x = {xr})")
        xg = always_redraw(xparts)
        self.play(FadeIn(xg), run_time=0.3)
        self.play(xv.animate.set_value(x_rest[1]), run_time=2.6)
        xg.clear_updaters()
        self.play(FadeOut(xg), run_time=0.3)
        pf = Line(P, F, color=BLUE_B, stroke_width=3.5)
        pd = Line(P, D, color=BLUE_B, stroke_width=3.5)
        dD = _dot(D)
        lD = _park(tag("D", 28), D, segs, DOWN + 0.5 * RIGHT, 0.36, avoid=(lP, la, ld), step=12)
        raD = _ra(D, P, S((a + 1, -0.25)), 0.15, GREY_A, 2)
        tkP = VGroup(_ticks(P, F, 2, YELLOW_B), _ticks(P, D, 2, YELLOW_B))
        self.play(Create(pf), Create(pd), FadeIn(dD), FadeIn(lD), Create(raD), run_time=0.8)
        self.play(FadeIn(tkP), run_time=0.4)
        self.bring_to_front(dP, dF, dD)

        # ---- ℓ: the perpendicular bisector of FD, through P; it meets the x-axis at FD's midpoint
        fd = DashedLine(F, D, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        tkM = VGroup(_ticks(F, M, 1, GREY_A), _ticks(M, D, 1, GREY_A))
        dM = _dot(M, YELLOW_B, 0.07)
        raM = _ra(M, F, L1, 0.15, YELLOW_B, 2)
        ell = Line(L0, L1, color=YELLOW_B, stroke_width=4)
        lell = _park(tag("ℓ", 32, YELLOW_B), L1, segs, _perp(L1 - L0), 0.32, avoid=(lP,), step=12)
        self.play(Create(fd), FadeIn(tkM), FadeIn(dM), run_time=0.8)
        self.play(Create(ell), Create(raM), FadeIn(lell), run_time=0.9)
        lM = _park(tag("a/2", 26, YELLOW_B), S((a / 4, 0.0)), segs, DOWN, 0.3,   # names VM
                   avoid=(lP, la, la2, lF, ld, lq1, lq2, lD, lell), step=10)
        check(V[0] < lM.get_center()[0] < M[0] and lM.get_center()[1] < V[1], "a/2 sits under VM")
        self.play(FadeIn(lM), run_time=0.5)
        self.bring_to_front(dM, dP)

        # ---- ℓ touches only at P: any other Q on it has QF = QD > QQ′
        sQ = ValueTracker(-1.0)

        def qparts():
            Q = Pm + sQ.get_value() * t
            Qp = np.array([Q[0], -0.25])
            g = VGroup(Line(S(Q), F, color=ORANGE, stroke_width=2.5),
                       Line(S(Q), D, color=ORANGE, stroke_width=2.5),
                       DashedLine(S(Q), S(Qp), color=ORANGE, stroke_width=2.5, dash_length=0.07),
                       _ticks(S(Q), F, 1, ORANGE), _ticks(S(Q), D, 1, ORANGE),
                       _dot(S(Q), ORANGE, 0.08))
            return g

        qg = always_redraw(qparts)
        self.play(FadeIn(qg), run_time=0.3)
        self.play(sQ.animate.set_value(0.85), run_time=2.2)
        self.play(sQ.animate.set_value(-0.6), run_time=1.4)
        qg.clear_updaters()
        self.play(FadeOut(qg), run_time=0.3)

        # ---- a half-turn about M keeps ℓ and carries P to T = (0, −a²)
        triP = mk([P, N, M], BLUE_D, 0.55, stroke_color=BLUE_B, stroke_width=2.5)
        ghost = mk([P, N, M], BLUE_D, 0.2, stroke_color=BLUE_B, stroke_width=1.5)
        self.add(ghost)
        self.bring_to_back(ghost)
        ghost.set_fill(opacity=0)
        self.play(FadeIn(triP), ghost.animate.set_fill(opacity=0.2), run_time=0.4)
        self.play(Rotate(triP, angle=PI, about_point=M), run_time=1.6)
        check(_pts_match(triP, [T, V, M]), "turned half a turn about M, PNM lies on TVM")
        dT = _dot(T, YELLOW_B, 0.07)
        lT = _park(tag("−a²", 26, YELLOW_B), T, segs, LEFT, 0.42,
                   avoid=(lF, ld, lq1, lq2, la2, lM), step=10)
        self.play(FadeIn(dT), FadeIn(lT), run_time=0.5)
        self.bring_to_front(dM, dT, lM)
        texts = [lP, la, la2, lF, ld, lq1, lq2, lD, lell, lM, lT]
        _labels_ok(texts, segs + _poly_segs([T, V, M]), "E31")
        cap = caption("tangent at (a, a²):   through (a/2, 0) and (0, −a²)   ⟹   y = 2ax − a²", 30)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E34  — three circles on a line

class E34_ThreeCircles(Board):
    """Two circles touching each other and a line ℓ, at T₁ and T₂. Join the
    centres and drop the radii to ℓ: a right triangle with hypotenuse
    r₁ + r₂ (through the point of contact) and leg r₁ − r₂, whose other leg
    has the length T₁T₂. In the square on r₁ + r₂, four r₁ × r₂ rectangles
    surround the square on r₁ − r₂: (r₁ + r₂)² − (r₁ − r₂)² = 4r₁r₂, so
    T₁T₂ = 2√(r₁r₂). The small circle (radius r) touching both and ℓ at T
    gives T₁T = 2√(r₁r) and TT₂ = 2√(rr₂) the same way, and the two pieces
    make up T₁T₂: 2√(r₁r) + 2√(rr₂) = 2√(r₁r₂). Divide by 2√(r₁r₂r):
    1/√r = 1/√r₁ + 1/√r₂."""

    def construct(self):
        r1, r2 = 2.0, 1.25
        r = 1.0 / (1.0 / np.sqrt(r1) + 1.0 / np.sqrt(r2)) ** 2
        k = 1.25
        n = np.linalg.norm
        yl = -1.72
        x1 = -6.35 + k * r1
        t12, t1, t2 = 2 * np.sqrt(r1 * r2), 2 * np.sqrt(r1 * r), 2 * np.sqrt(r * r2)
        T1 = np.array([x1, yl, 0.0])
        T2 = T1 + k * t12 * RIGHT
        T = T1 + k * t1 * RIGHT
        O1, O2, O = T1 + k * r1 * UP, T2 + k * r2 * UP, T + k * r * UP
        H = np.array([O1[0], O2[1], 0.0])
        check(abs(n(O1 - O2) - k * (r1 + r2)) < 1e-9, "the big circles touch: O₁O₂ = r₁ + r₂")
        check(abs(n(O - O1) - k * (r1 + r)) < 1e-9 and abs(n(O - O2) - k * (r2 + r)) < 1e-9,
              "the small circle touches both")
        check(abs(n(H - O1) - k * (r1 - r2)) < 1e-9 and abs(n(O2 - H) - k * t12) < 1e-9
              and abs(np.dot(O1 - H, O2 - H)) < 1e-9, "the right triangle: legs r₁ − r₂, T₁T₂")
        check(abs((r1 + r2) ** 2 - (r1 - r2) ** 2 - 4 * r1 * r2) < 1e-12, "(r₁+r₂)² − (r₁−r₂)² = 4r₁r₂")
        check(abs(t1 + t2 - t12) < 1e-12, "2√(r₁r) + 2√(rr₂) = 2√(r₁r₂)")
        check(abs(1 / np.sqrt(r) - 1 / np.sqrt(r1) - 1 / np.sqrt(r2)) < 1e-12, "1/√r = 1/√r₁ + 1/√r₂")
        Tc = (O1 * r2 + O2 * r1) / (r1 + r2)                  # where the big circles touch
        check(abs(n(Tc - O1) - k * r1) < 1e-9 and abs(n(Tc - O2) - k * r2) < 1e-9,
              "the contact point lies on O₁O₂")
        H1 = np.array([O1[0], O[1], 0.0])                     # the small circle's two triangles
        H2 = np.array([O2[0], O[1], 0.0])
        check(abs(n(H1 - O1) - k * (r1 - r)) < 1e-9 and abs(n(O - H1) - k * t1) < 1e-9
              and abs(n(H2 - O2) - k * (r2 - r)) < 1e-9 and abs(n(H2 - O) - k * t2) < 1e-9,
              "the same right triangles for the small circle")

        # ---- the line and the two big circles
        lline = Line(np.array([-6.5, yl, 0]), np.array([2.3, yl, 0]), color=WHITE, stroke_width=3)
        c1 = Circle(radius=k * r1, color=BLUE_B, stroke_width=3.5).move_to(O1)
        c2 = Circle(radius=k * r2, color=TEAL_B, stroke_width=3.5).move_to(O2)
        dT1, dT2 = _dot(T1), _dot(T2)
        self.play(Create(lline), run_time=0.6)
        self.play(Create(c1), Create(c2), run_time=1.2)
        R1e = O1 + k * r1 * _dir(140 * DEGREES)              # r₁ on a radius of its own
        rad1 = Line(O1, R1e, color=BLUE_B, stroke_width=3)
        rad2 = Line(O2, T2, color=TEAL_B, stroke_width=3)
        dO1, dO2 = _dot(O1, BLUE_B), _dot(O2, TEAL_B)
        lr1 = _beside(tag("r₁", 26, BLUE_B), O1, R1e, UP + RIGHT, 0.1, 0.5)
        lr2 = _beside(tag("r₂", 26, TEAL_B), O2, T2, RIGHT, 0.1, 0.5)
        self.play(Create(rad1), Create(rad2), FadeIn(dO1), FadeIn(dO2), FadeIn(dT1), FadeIn(dT2),
                  FadeIn(lr1), FadeIn(lr2), run_time=0.9)

        # ---- the right triangle: hypotenuse r₁ + r₂, leg r₁ − r₂
        hyp = Line(O1, O2, color=WHITE, stroke_width=3.5)
        leg = Line(O1, H, color=ORANGE, stroke_width=5)
        base_ = Line(H, O2, color=YELLOW_B, stroke_width=4)
        raH = _ra(H, O1, O2, 0.18, WHITE, 2)
        segs0 = ([_seg(O1, R1e), _seg(H, T1), _seg(O2, T2), _seg(O1, O2), _seg(O1, H), _seg(H, O2)]
                 + _arc_segs(O1, k * r1, 0, TAU, 160) + _arc_segs(O2, k * r2, 0, TAU, 120)
                 + _arc_segs(O, k * r, 0, TAU, 60) + _ra_segs(H, O1, O2, 0.18)
                 + [_seg(O1, O), _seg(H1, O), _seg(O2, O), _seg(O, H2)])
        lhyp = _park_beside(tag("r₁ + r₂", 24), O1, O2, UP + RIGHT, segs0,
                            ats=(0.32, 0.38, 0.26, 0.44, 0.2), avoid=(lr1, lr2))
        lleg = _park_beside(tag("r₁ − r₂", 22, ORANGE), O1, H, LEFT, segs0,
                            ats=(0.5, 0.4, 0.6), avoid=(lr1, lr2, lhyp))
        self.play(Create(hyp), run_time=0.7)
        self.play(Create(leg), Create(base_), Create(raH), FadeIn(lhyp), FadeIn(lleg), run_time=0.9)
        drops = VGroup(DashedLine(H, T1, color=YELLOW_B, stroke_width=2, dash_length=0.07),
                       DashedLine(O2, T2, color=YELLOW_B, stroke_width=2, dash_length=0.07))
        self.play(Create(drops), run_time=0.5)

        # ---- the square on r₁ + r₂: four r₁ × r₂ round the square on r₁ − r₂
        ki = 0.7
        Q0 = np.array([3.35, 0.95, 0.0])                     # inset, bottom-left corner
        a1, a2 = ki * r1, ki * r2

        def R(xa, ya, xb, yb):
            return [Q0 + np.array([xa, ya, 0]), Q0 + np.array([xb, ya, 0]),
                    Q0 + np.array([xb, yb, 0]), Q0 + np.array([xa, yb, 0])]

        sq = R(0, 0, a1 + a2, a1 + a2)
        rects = [R(0, 0, a1, a2), R(a1, 0, a1 + a2, a1), R(a2, a1, a1 + a2, a1 + a2), R(0, a2, a2, a1 + a2)]
        mid = R(a2, a2, a1, a1)
        check(_tiles_exactly(rects + [mid], sq, 80), "four r₁ × r₂ and the square on r₁ − r₂ tile it")
        frame = Polygon(*sq, stroke_color=WHITE, stroke_width=3)
        rm = VGroup(*[mk(p, BLUE_D if i % 2 == 0 else TEAL_D, 0.7) for i, p in enumerate(rects)])
        mm = mk(mid, ORANGE, 0.85)
        lrr = VGroup(*[tag("r₁r₂", 20).move_to(sum(p) / 4) for p in rects])
        top1 = tag("r₁", 22).move_to(Q0 + np.array([a1 / 2, a1 + a2 + 0.25, 0]))
        top2 = tag("r₂", 22).move_to(Q0 + np.array([a1 + a2 / 2, a1 + a2 + 0.25, 0]))
        tick = Line(Q0 + np.array([a1, a1 + a2 + 0.1, 0]), Q0 + np.array([a1, a1 + a2 - 0.1, 0]),
                    color=WHITE, stroke_width=3)
        ident = tag("(r₁+r₂)² − (r₁−r₂)²  =  4r₁r₂", 22)
        ident.move_to(Q0 + np.array([(a1 + a2) / 2, -0.38, 0]))
        self.play(Create(frame), FadeIn(top1), FadeIn(top2), Create(tick), run_time=0.7)
        self.play(FadeIn(rm), FadeIn(lrr), run_time=0.8)
        self.play(FadeIn(mm), Indicate(leg, color=ORANGE, scale_factor=1.0), run_time=0.8)
        self.play(FadeIn(ident), run_time=0.6)

        # ---- so T₁T₂ = 2√(r₁r₂)
        yB = -2.55
        brk = VGroup(Line([T1[0], yB, 0], [T2[0], yB, 0], color=YELLOW_B, stroke_width=3),
                     Line([T1[0], yB + 0.1, 0], [T1[0], yB - 0.1, 0], color=YELLOW_B, stroke_width=3),
                     Line([T2[0], yB + 0.1, 0], [T2[0], yB - 0.1, 0], color=YELLOW_B, stroke_width=3))
        l12 = tag("2√(r₁r₂)", 24, YELLOW_B).move_to([(T1[0] + T2[0]) / 2, yB - 0.26, 0])
        self.play(Create(brk), FadeIn(l12), run_time=0.8)
        self.hold(0.3)

        # ---- the small circle between: the same two right triangles
        c0 = Circle(radius=k * r, color=ORANGE, stroke_width=3.5).move_to(O)
        dT = _dot(T)
        lr = tag("r", 24, ORANGE)                            # inside, between the lines to O₁, O₂
        _angle_label(lr, O, O1, O2, 0.18, r1=0.34)
        check(n(lr.get_center() - O) + 0.17 < k * r, "'r' inside the small circle")
        self.play(Create(c0), FadeIn(dT), FadeIn(lr), run_time=0.8)
        tri1 = VGroup(DashedLine(O1, O, color=ORANGE, stroke_width=2, dash_length=0.06),
                      DashedLine(H1, O, color=ORANGE, stroke_width=2, dash_length=0.06))
        tri2 = VGroup(DashedLine(O2, O, color=ORANGE, stroke_width=2, dash_length=0.06),
                      DashedLine(O, H2, color=ORANGE, stroke_width=2, dash_length=0.06))
        self.play(Create(tri1), Create(tri2), run_time=0.9)
        yA = -1.95
        b1 = VGroup(Line([T1[0], yA, 0], [T[0], yA, 0], color=BLUE_B, stroke_width=3),
                    Line([T1[0], yA + 0.08, 0], [T1[0], yA - 0.08, 0], color=BLUE_B, stroke_width=3),
                    Line([T[0], yA + 0.08, 0], [T[0], yA - 0.08, 0], color=BLUE_B, stroke_width=3))
        b2 = VGroup(Line([T[0], yA, 0], [T2[0], yA, 0], color=TEAL_B, stroke_width=3),
                    Line([T2[0], yA + 0.08, 0], [T2[0], yA - 0.08, 0], color=TEAL_B, stroke_width=3))
        l1 = tag("2√(r₁r)", 22, BLUE_B).move_to([(T1[0] + T[0]) / 2, yA - 0.27, 0])
        l2 = tag("2√(rr₂)", 22, TEAL_B).move_to([(T[0] + T2[0]) / 2, yA - 0.27, 0])
        self.play(Create(b1), Create(b2), FadeIn(l1), FadeIn(l2), run_time=0.9)

        # ---- the pieces make up T₁T₂; divide by 2√(r₁r₂r)
        eq1 = tag("2√(r₁r) + 2√(rr₂)  =  2√(r₁r₂)", 24)
        eq2 = tag("÷ 2√(r₁r₂r)", 24, GREY_A)
        eq3 = tag("1/√r₂ + 1/√r₁  =  1/√r", 24)
        eq1.move_to([6.5 - eq1.width / 2, -0.5, 0])
        eq2.move_to([6.5 - eq2.width / 2, -1.05, 0])
        eq3.move_to([6.5 - eq3.width / 2, -1.6, 0])
        self.play(FadeIn(eq1), run_time=0.7)
        self.play(FadeIn(eq2), run_time=0.5)
        self.play(FadeIn(eq3), run_time=0.7)

        segs = ([_seg([-6.5, yl, 0], [2.3, yl, 0]), _seg(O1, R1e), _seg(O2, T2),
                 _seg(O1, O2),
                 _seg(O1, H), _seg(H, O2), _seg(H, T1), _seg(O1, O), _seg(H1, O), _seg(O2, O),
                 _seg(O, H2)]
                + _arc_segs(O1, k * r1, 0, TAU, 160) + _arc_segs(O2, k * r2, 0, TAU, 120)
                + _arc_segs(O, k * r, 0, TAU, 60) + _ra_segs(H, O1, O2, 0.18)
                + [s for X in (T1, T2, T, O1, O2) for s in _dot_segs(X)]
                + [_seg(m.get_start(), m.get_end()) for m in list(brk) + list(b1) + list(b2)]
                + _poly_segs(sq) + [_seg(tick.get_start(), tick.get_end())]
                + [s for p in rects for s in _poly_segs(p)])
        texts = [lr1, lr2, lr, lhyp, lleg, l12, l1, l2, ident, eq1, eq2, eq3, top1, top2]
        _labels_ok(texts, segs, "E34")
        for lab, p in zip(lrr, rects):
            bx = _box(lab)
            check(all(_pip((xx, yy), p) for xx in (bx[0], bx[1]) for yy in (bx[2], bx[3])),
                  "r₁r₂ inside its rectangle")
        cap = caption("1/√r  =  1/√r₁  +  1/√r₂", 38)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E38  — the common chord

class E38_CommonChord(Board):
    """Two circles about O₁ and O₂ cross at A and B. Fold the figure over
    the line of centres: each circle has its centre on the fold, so each
    goes to itself, and the half above the line lands on the half below.
    A lies on both circles, so its image does too; it is not on the line
    (else the circles would only touch), so its image is the other common
    point, B. A and B are mirror images in O₁O₂: the chord AB is square to
    the line of centres and is halved by it at M. The same holds however
    the circles are placed, as long as they cross."""

    def construct(self):
        r1, r2 = 2.6, 2.05
        n = np.linalg.norm
        O1 = np.array([-1.75, 0.45, 0.0])
        th = -12 * DEGREES
        u = _dir(th)
        w = _perp(u)                                           # up, square to the line of centres

        def config(d):
            O2 = O1 + d * u
            m = (d * d + r1 * r1 - r2 * r2) / (2 * d)
            h = float(np.sqrt(r1 * r1 - m * m))
            M = O1 + m * u
            return O2, M, M + h * w, M - h * w

        for d in np.linspace(2.3, 4.3, 41):
            O2, M, A, B = config(d)
            check(abs(n(A - O1) - r1) < 1e-9 and abs(n(A - O2) - r2) < 1e-9
                  and abs(n(B - O1) - r1) < 1e-9 and abs(n(B - O2) - r2) < 1e-9,
                  "A and B on both circles")
            check(close(_refl(A, O1, u), B), "the fold over O₁O₂ carries A onto B")
            check(abs(np.dot(B - A, u)) < 1e-9 and close((A + B) / 2, M)
                  and abs(_cross(M - O1, u)) < 1e-9, "AB ⊥ O₁O₂, halved at M on it")
        d0 = 3.4
        O2, M, A, B = config(d0)
        L0, L1 = O1 - 3.1 * u, O1 + (d0 + 2.6) * u               # the line of centres as drawn

        # ---- the two circles, their centres and the line of centres
        c1 = Circle(radius=r1, color=BLUE_B, stroke_width=3.5).move_to(O1)
        c2 = Circle(radius=r2, color=TEAL_B, stroke_width=3.5).move_to(O2)
        dO1, dO2 = _dot(O1, BLUE_B), _dot(O2, TEAL_B)
        self.play(Create(c1), Create(c2), FadeIn(dO1), FadeIn(dO2), run_time=1.3)
        axis = DashedLine(L0, L1, color=GREY_A, stroke_width=2.5, dash_length=0.1)
        self.play(Create(axis), run_time=0.7)
        dA, dB = _dot(A, YELLOW_B, 0.07), _dot(B, YELLOW_B, 0.07)
        ab = Line(A, B, color=YELLOW_B, stroke_width=4)
        self.play(FadeIn(dA), FadeIn(dB), Create(ab), run_time=0.8)

        def segs_for(O2_, M_, A_, B_):
            return ([_seg(L0, O1 + (n(O2_ - O1) + 2.6) * u), _seg(A_, B_)]
                    + _arc_segs(O1, r1, 0, TAU, 140) + _arc_segs(O2_, r2, 0, TAU, 120)
                    + _ra_segs(M_, A_, O2_, 0.2) + _tick_segs(A_, M_, 2) + _tick_segs(M_, B_, 2)
                    + [s for X in (O1, O2_, M_, A_, B_) for s in _dot_segs(X)])

        def cusp(m, X, O2_, sg):
            """Label m outside both circles at X, on the bisector of the cusp
            between them, at the least distance where it is clear."""
            b = _unit(_unit(X - O1) + _unit(X - O2_))
            for dd in np.arange(0.36, 1.0, 0.02):
                m.move_to(X + dd * b)
                if _hit(m, sg, 0.07) is None:
                    return m
            check(False, f"room for '{_name(m)}' in the cusp")

        def moving_labels(d):
            """O₂, A, B while the circle moves (M's label waits)."""
            O2_, M_, A_, B_ = config(d)
            sg = segs_for(O2_, M_, A_, B_)
            return VGroup(tag("O₂", 28, TEAL_B).move_to(O2_ + 0.42 * _unit(-w + 0.6 * u)),
                          cusp(tag("A", 28, YELLOW_B), A_, O2_, sg),
                          cusp(tag("B", 28, YELLOW_B), B_, O2_, sg))

        def labels_for(O2_, M_, A_, B_, segs):
            ls = {}
            ls["O₁"] = _park(tag("O₁", 28, BLUE_B), O1, segs, -w, 0.42, step=12)
            ls["O₂"], ls["A"], ls["B"] = moving_labels(n(O2_ - O1))
            ls["M"] = _park(tag("M", 26, YELLOW_B), M_, segs, _unit(-u - w), 0.42,
                            avoid=list(ls.values()), step=12)
            return ls

        segs0 = segs_for(O2, M, A, B)
        labs = labels_for(O2, M, A, B, segs0)
        self.play(*[FadeIn(labs[k_]) for k_ in ("O₁", "O₂", "A", "B")], run_time=0.5)
        self.hold(0.3)

        # ---- fold the upper halves over the line of centres
        top1 = Arc(radius=r1, start_angle=th, angle=PI, arc_center=O1, color=BLUE_B, stroke_width=9)
        top2 = Arc(radius=r2, start_angle=th, angle=PI, arc_center=O2, color=TEAL_B, stroke_width=9)
        halfA = Line(A, M, color=YELLOW_B, stroke_width=8)
        pA = _dot(A, YELLOW_B, 0.1)
        flap = VGroup(top1, top2, halfA, pA)
        self.add(flap)
        self.bring_to_front(*[labs[k_] for k_ in ("O₁", "O₂", "A", "B")])
        self.play(FadeIn(flap), run_time=0.5)
        self.play(_fold(flap, O1, O1 + u), run_time=2.0)
        check(close(pA.get_center(), B, 1e-6), "folded, A lands on B")
        check(all(abs(n(top1.point_from_proportion(t_) - O1) - r1) < 1e-6
                  and np.dot(top1.point_from_proportion(t_) - O1, w) < 1e-6 for t_ in np.linspace(0, 1, 21)),
              "the upper half of circle 1 lies on its lower half")
        check(all(abs(n(top2.point_from_proportion(t_) - O2) - r2) < 1e-6
                  and np.dot(top2.point_from_proportion(t_) - O2, w) < 1e-6 for t_ in np.linspace(0, 1, 21)),
              "the upper half of circle 2 lies on its lower half")
        check((close(halfA.get_start(), B, 1e-6) and close(halfA.get_end(), M, 1e-6)),
              "AM lies on BM")
        dM = _dot(M, YELLOW_B, 0.06)
        raM = _ra(M, A, O2, 0.2, YELLOW_B, 2.5)
        tk = VGroup(_ticks(A, M, 2, YELLOW_B), _ticks(M, B, 2, YELLOW_B))
        self.play(FadeOut(flap), FadeIn(dM), Create(raM), FadeIn(tk), FadeIn(labs["M"]), run_time=0.8)
        self.bring_to_front(dA, dB, dM)
        _labels_ok(list(labs.values()), segs0, "E38")
        self.hold(0.4)

        # ---- wherever the circles are, as long as they cross
        dv = ValueTracker(d0)

        def live():
            O2_, M_, A_, B_ = config(dv.get_value())
            return VGroup(Circle(radius=r2, color=TEAL_B, stroke_width=3.5).move_to(O2_),
                          Line(A_, B_, color=YELLOW_B, stroke_width=4),
                          _ra(M_, A_, O2_, 0.2, YELLOW_B, 2.5),
                          _ticks(A_, M_, 2, YELLOW_B), _ticks(M_, B_, 2, YELLOW_B),
                          _dot(O2_, TEAL_B), _dot(A_, YELLOW_B, 0.07), _dot(B_, YELLOW_B, 0.07),
                          _dot(M_, YELLOW_B, 0.06))

        def live_axis():
            return DashedLine(L0, O1 + (dv.get_value() + 2.6) * u, color=GREY_A, stroke_width=2.5,
                              dash_length=0.1)

        def live_labels():
            return moving_labels(dv.get_value())

        for d in np.linspace(2.7, 3.6, 25):                    # the labels stay clear as it moves
            O2_, M_, A_, B_ = config(d)
            sg = segs_for(O2_, M_, A_, B_)
            _labels_ok(list(moving_labels(d)) + [labs["O₁"]], sg, f"E38 (d = {d:.2f})")
        lv, la_, lx = always_redraw(live), always_redraw(live_labels), always_redraw(live_axis)
        self.remove(c2, ab, raM, tk, dO2, dA, dB, dM, axis, *labs.values())
        self.add(lx, lv, la_, labs["O₁"])
        self.bring_to_front(dO1)
        self.play(dv.animate.set_value(2.7), run_time=1.2)
        self.play(dv.animate.set_value(3.6), run_time=1.8)
        self.play(dv.animate.set_value(d0), run_time=0.8)
        for m in (lv, la_, lx):
            m.clear_updaters()
        self.remove(la_)                                       # back to the parked labels
        self.add(*[labs[k_] for k_ in ("O₂", "A", "B")])
        self.play(FadeIn(labs["M"]), run_time=0.4)
        cap = caption("O₁O₂ ⊥ AB,    AM  =  MB", 38)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)
