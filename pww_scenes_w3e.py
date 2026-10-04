# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3e.py — proofs without words, 2D (manim):
#     E35 common tangent of touching circles   E27 Miquel's theorem
#     E32 tangent to a hyperbola               E33 Archimedes' broken chord
#     G26 arcsin + arccos                      G18 the projection formula
#     G23 15° and 75°                          G24 regular polygon from its radius
#     G20 Jordan's inequality                  G28 quadrilateral from its diagonals
#
# Every move of a piece is rigid — a slide (shift), a turn about a named
# point (Rotate), or a turn about a moving pivot that stays rigid on every
# frame — except the announced squeeze of E32, (x, y) -> (kx, y/k), a linear
# map that keeps the hyperbola, lines, midpoints and areas. Every landing,
# tiling, angle and length claim is checked with check(...) before or right
# after it is drawn, and every label is checked against the lines, marks and
# other labels it must clear, so a wrong construction fails the render
# instead of drawing a wrong picture. No LaTeX: every label is Text (tag /
# caption) with Unicode.


# ------------------------------------------------------------ private helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _polar(a):
    return np.array([np.cos(a), np.sin(a), 0.0])


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


def _perp(v):
    v = to3(v)
    return np.array([-v[1], v[0], 0.0])


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


def _on_line(p, a, b, tol=1e-9):
    return abs(_cross(to3(b) - to3(a), to3(p) - to3(a))) < tol * max(
        1.0, float(np.linalg.norm(to3(b) - to3(a))))


def _between(p, a, b, tol=1e-9):
    """p lies on the segment ab (strictly inside)."""
    p, a, b = to3(p), to3(a), to3(b)
    d = b - a
    s = np.dot(p - a, d) / np.dot(d, d)
    return _on_line(p, a, b, tol) and tol < s < 1 - tol


def _circumcentre(A, B, C):
    A, B, C = to3(A), to3(B), to3(C)
    return _meet((A + B) / 2, (A + B) / 2 + _perp(B - A),
                 (B + C) / 2, (B + C) / 2 + _perp(C - B))


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments (for label checks)."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


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
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    return [(c + d * 0.09 * (k - (n - 1) / 2) - nrm * size,
             c + d * 0.09 * (k - (n - 1) / 2) + nrm * size) for k in range(n)]


def _wedge(vertex, p, q, r, color, op=FILL, stroke=0.0, stroke_color=WHITE):
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


def _span(vertex, p, q):
    """(start angle, span) of the non-reflex angle p-vertex-q."""
    v = to3(vertex)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return a1, span


def _glide(mob, pivot, target, angle, **kw):
    """Rigid on every frame: the piece turns by `angle` about its pivot
    while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centre while that
    centre slides straight — carrying the screen points src onto dst (same
    order). Fails the render unless one rotation + translation lands every
    point. `turn` (radians) picks the direction for ±180°."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = _dir(src[0], src[1])
        a1 = _dir(dst[0], dst[1])
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    check(all(close(m1 + _rot2(p, m0, turn) - m0, q, 1e-6)
              for p, q in zip(src, dst)), "a rigid motion lands every point")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
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


def _same_poly(P, Q, tol=1e-9):
    """Same vertices in the same cyclic order (any start, either sense)."""
    P = [to3(p) for p in P]
    Q = [to3(q) for q in Q]
    if len(P) != len(Q):
        return False
    n = len(P)
    for d in (1, -1):
        for s in range(n):
            if all(close(P[i], Q[(s + d * i) % n], tol) for i in range(n)):
                return True
    return False


def _landed(mob, pts, what, tol=1e-6):
    """After a move: the polygon's vertices are exactly pts, in order."""
    got = [to3(v) for v in mob.get_vertices()]
    want = [to3(p) for p in pts]
    check(len(got) == len(want)
          and all(close(g, w, tol) for g, w in zip(got, want)), what)


def _curve(pts, color=WHITE, width=3):
    """Open polyline through screen points (dense sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _bracket(p, q, nrm, text, gap=0.28, color=YELLOW_B, size=26, tick=0.11,
             buff=0.12, width=3):
    """Dimension bar beside segment pq, `gap` away along the unit normal
    nrm, with end ticks; the label sits beyond the bar. It spans the whole
    segment, so it names the union of whatever lies along it."""
    p, q, n = to3(p), to3(q), _unit(nrm)
    a, b = p + gap * n, q + gap * n
    bar = VGroup(Line(a, b, color=color, stroke_width=width),
                 Line(a - tick * n, a + tick * n, color=color, stroke_width=width),
                 Line(b - tick * n, b + tick * n, color=color, stroke_width=width))
    lab = tag(text, size, color)
    half = abs(n[0]) * lab.width / 2 + abs(n[1]) * lab.height / 2
    lab.move_to((a + b) / 2 + n * (buff + half + 0.02))
    return bar, lab


def _bracket_segs(p, q, nrm, gap=0.28, tick=0.11):
    p, q, n = to3(p), to3(q), _unit(nrm)
    a, b = p + gap * n, q + gap * n
    return [(a, b), (a - tick * n, a + tick * n), (b - tick * n, b + tick * n)]


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


def _arc_segs(c, r, a0, a1, n=24):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _angle_segs(v, p, q, r, n=12):
    """The arc angle_arc(v, p, q, r) draws, as segments."""
    a1, span = _span(v, p, q)
    return _arc_segs(v, r, a1, a1 + span, n)


def _dot_segs(c, r=0.08):
    """A dot as a small cross of segments (for label checks)."""
    c = to3(c)
    return [(c - r * RIGHT, c + r * RIGHT), (c - r * UP, c + r * UP),
            (c - 0.7 * r * (RIGHT + UP), c + 0.7 * r * (RIGHT + UP)),
            (c - 0.7 * r * (RIGHT - UP), c + 0.7 * r * (RIGHT - UP))]


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


def _park(m, X, towards, segs, d0=0.3, d1=1.2, pad=0.07, avoid=(), sep=0.05):
    """Move label m next to the point X: along the middles of the angular
    gaps between the directions to `towards` (widest gap first), at the
    smallest distance where m clears every segment in segs and stays `sep`
    away from every mobject in `avoid`. Fails the render if no such spot."""
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
                    and not any(_overlap(m, o, sep) for o in avoid)):
                found.append((d, -w, mid))
                break
    check(bool(found), f"a free spot for the label '{_name(m)}'")
    d, _, mid = min(found)                       # the nearest free spot
    return m.move_to(X + d * _polar(mid))


def _park_dirs(m, X, dirs_deg, segs, d0=0.25, d1=1.2, pad=0.06, avoid=(), sep=0.08):
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


def _angle_label(m, V, P, Q, r0, pad=0.06, r1=3.0, segs=(),
                 fracs=(0.5,), avoid=(), sep=0.05):
    """Put label m inside the non-reflex angle P-V-Q -- on its bisector, or
    at the other fractions of the angle given in `fracs`, tried in order --
    at the least distance >= r0 from V where it clears both arms, segs and
    the mobjects in `avoid`."""
    V = to3(V)
    a1, span = _span(V, P, Q)
    arms = [_seg(V, P), _seg(V, Q)] + list(segs)
    for f in fracs:
        u = _polar(a1 + f * span)
        for d in np.arange(r0, r1, 0.03):
            m.move_to(V + d * u)
            if (_hit(m, arms, pad) is None
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"room for the angle label '{_name(m)}'")


def _all_in_frame(mobs, what, tol=0.02):
    for i, m in enumerate(mobs):
        check(_in_frame(m, tol), f"{what}: item {i} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption in its band, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip
             and len(m.get_all_points()) > 0]
    for m in shown:
        check(_in_frame(m, 0.03), f"{type(m).__name__} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(not _overlap(texts[i], texts[j], 0.02),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# ============================================ E35 COMMON TANGENT OF TOUCHING CIRCLES

class E35_TouchingCirclesTangent(Board):
    """Circles of radii r₁ > r₂ touch at P and touch a line at T₁, T₂. The
    radii O₁T₁, O₂T₂ are perpendicular to the line, so they are parallel.
    Slide the tangent segment T₁T₂ = t up by r₂, to HO₂ (H on O₁T₁): with
    T₁T₂ it bounds a rectangle, so O₁H = r₁ − r₂, and O₁HO₂ is a right
    triangle with hypotenuse O₁O₂ = r₁ + r₂ (through P). So
    t² = (r₁ + r₂)² − (r₁ − r₂)². The hypotenuse, carried over rigidly with
    its point P, is the side of a square cut into four r₁ × r₂ rectangles
    round a hole of side r₁ − r₂ (the leg O₁H, carried over): the
    difference of the squares is 4r₁r₂, so t = 2√(r₁r₂)."""

    def construct(self):
        r1m, r2m = 1.9, 0.78
        tm = 2 * np.sqrt(r1m * r2m)
        Lm = r1m + r2m
        # math coordinates: the tangent line is y = 0, T₁ at the origin
        sq0 = np.array([tm + r2m + 1.05, 0.55])          # square's lower-left
        F = Frame(-r1m - 0.32, sq0[0] + Lm + 0.12, -0.55, 2 * r1m + 0.05)
        k = F.k
        O1, O2 = F.P((0.0, r1m)), F.P((tm, r2m))
        T1, T2 = F.P((0.0, 0.0)), F.P((tm, 0.0))
        H = F.P((0.0, r2m))
        r1, r2, t, L = r1m * k, r2m * k, tm * k, Lm * k
        n = np.linalg.norm
        u12 = _unit(O2 - O1)
        P = O1 + r1 * u12
        check(abs(n(O2 - O1) - (r1 + r2)) < 1e-9, "the circles touch: O₁O₂ = r₁ + r₂")
        check(close(P, O2 - r2 * u12), "P is on both circles")
        check(abs(np.dot(O1 - T1, T2 - T1)) < 1e-9 and abs(np.dot(O2 - T2, T2 - T1)) < 1e-9,
              "both radii to the line are perpendicular to it")
        check(close(H + (T2 - T1), O2), "T₁T₂ slid up by r₂ is HO₂")
        check(abs(n(O1 - H) - (r1 - r2)) < 1e-9, "O₁H = r₁ − r₂")
        check(abs(np.dot(O1 - H, O2 - H)) < 1e-9, "the angle at H is right")
        check(abs(t ** 2 + (r1 - r2) ** 2 - (r1 + r2) ** 2) < 1e-9, "Pythagoras in O₁HO₂")
        check(abs(t - 2 * np.sqrt(r1 * r2)) < 1e-9, "t = 2√(r₁r₂)")

        # the square of side r₁ + r₂: four r₁ × r₂ rectangles round a hole
        S = F.P(sq0)

        def Q(x, y):
            return S + np.array([x, y, 0.0])
        rects = [[Q(0, 0), Q(r1, 0), Q(r1, r2), Q(0, r2)],
                 [Q(r1, 0), Q(L, 0), Q(L, r1), Q(r1, r1)],
                 [Q(r2, r1), Q(L, r1), Q(L, L), Q(r2, L)],
                 [Q(0, r2), Q(r2, r2), Q(r2, L), Q(0, L)]]
        hole = [Q(r2, r2), Q(r1, r2), Q(r1, r1), Q(r2, r1)]
        square = [Q(0, 0), Q(L, 0), Q(L, L), Q(0, L)]
        check(_tiles_exactly(rects + [hole], square),
              "four r₁ × r₂ rectangles and the hole tile the square")
        for R in rects:
            check(abs(abs(area(R)) - r1 * r2) < 1e-9, "each rectangle is r₁ × r₂")
        check(abs(abs(area(hole)) - (r1 - r2) ** 2) < 1e-9, "the hole is (r₁ − r₂)²")
        check(abs(L ** 2 - (r1 - r2) ** 2 - 4 * r1 * r2) < 1e-9,
              "(r₁ + r₂)² − (r₁ − r₂)² = 4r₁r₂")

        c1, c2, cT = BLUE_B, ORANGE, YELLOW_B
        # ---- the line, the two circles, touching at P
        line = Line(F.P((-r1m - 0.3, 0.0)), F.P((tm + r2m + 0.45, 0.0)),
                    color=GREY_B, stroke_width=3)
        circ1 = Circle(radius=r1, color=c1, stroke_width=3).move_to(O1)
        circ2 = Circle(radius=r2, color=c2, stroke_width=3).move_to(O2)
        dO1, dO2 = Dot(O1, radius=0.06), Dot(O2, radius=0.06)
        dP = Dot(P, radius=0.07, color=YELLOW_B)
        self.play(Create(line), Create(circ1), Create(circ2), run_time=1.3)
        self.play(FadeIn(dO1), FadeIn(dO2), FadeIn(dP), run_time=0.5)

        # ---- the radii to the line, perpendicular to it
        rad1 = Line(O1, T1, color=c1, stroke_width=4)
        rad2 = Line(O2, T2, color=c2, stroke_width=4)
        ra1, ra2 = _ra(T1, O1, T2, 0.2), _ra(T2, T1, O2, 0.2)
        tseg = Line(T1, T2, color=cT, stroke_width=6)

        base_segs = (_circle_segs(O1, r1, 160) + _circle_segs(O2, r2, 100)
                     + [_seg(line.get_start(), line.get_end())]
                     + _dot_segs(O1) + _dot_segs(O2) + _dot_segs(P))
        segs1 = (base_segs + [_seg(O1, T1), _seg(O2, T2)]
                 + _ra_segs(T1, O1, T2, 0.2) + _ra_segs(T2, T1, O2, 0.2))
        lO1 = _park_dirs(tag("O₁", 28), O1, [135, 160, 110, 45], segs1, d0=0.3)
        lO2 = _park_dirs(tag("O₂", 28), O2, [40, 70, 20, 110], segs1, d0=0.28,
                         avoid=[lO1])
        lr1 = _park_beside(tag("r₁", 28, c1), [(T1, O1)], segs1, prefer=LEFT, only=True,
                           avoid=[lO1, lO2])
        lr2 = _park_beside(tag("r₂", 28, c2), [(T2, O2)], segs1, prefer=RIGHT, only=True,
                           avoid=[lO1, lO2, lr1])
        lt = tag("t", 30, cT).next_to(tseg, DOWN, buff=0.16)
        self.play(Create(rad1), Create(rad2), FadeIn(lO1), FadeIn(lO2), run_time=0.9)
        self.play(Create(ra1), Create(ra2), FadeIn(lr1), FadeIn(lr2), run_time=0.7)
        self.play(Create(tseg), FadeIn(lt), run_time=0.7)

        # ---- the line of centres passes through P: r₁ + r₂
        cl1 = Line(O1, P, color=c1, stroke_width=4)
        cl2 = Line(P, O2, color=c2, stroke_width=4)
        segs2 = segs1 + [_seg(O1, O2), _seg(T1, T2)]
        lc1 = _park_beside(tag("r₁", 28, c1), [(O1, P)], segs2, prefer=UP + RIGHT,
                           only=True, avoid=[lO1, lO2, lr1, lr2, lt])
        lc2 = _park_beside(tag("r₂", 28, c2), [(P, O2)], segs2, prefer=UP + RIGHT,
                           only=True, avoid=[lO1, lO2, lr1, lr2, lt, lc1])
        self.play(Create(cl1), Create(cl2), run_time=0.8)
        self.bring_to_front(dO1, dO2, dP)
        self.play(FadeIn(lc1), FadeIn(lc2), run_time=0.5)
        self.hold(0.3)

        # ---- slide t up by r₂: the rectangle T₁T₂O₂H, so O₁H = r₁ − r₂
        tcopy = tseg.copy()
        self.add(tcopy)
        tseg.set_stroke(opacity=0.35)
        self.play(tcopy.animate.shift(H - T1), run_time=1.3)
        check(close(tcopy.get_start(), H, 1e-6) and close(tcopy.get_end(), O2, 1e-6),
              "the slid segment is HO₂")
        rect = mk([T1, T2, O2, H], cT, 0.12, stroke_width=0)
        hseg = Line(H, T1, color=c2, stroke_width=4)
        leg = Line(O1, H, color=WHITE, stroke_width=5)
        raH = _ra(H, O1, O2, 0.2)
        segs3 = (base_segs + [_seg(O1, T1), _seg(O2, T2), _seg(O1, O2), _seg(H, O2),
                              _seg(T1, T2)]
                 + _ra_segs(T1, O1, T2, 0.2) + _ra_segs(T2, T1, O2, 0.2)
                 + _ra_segs(H, O1, O2, 0.2))
        lt2 = _park_beside(tag("t", 30, cT), [(H, O2)], segs3, prefer=UP, only=True,
                           avoid=[lO1, lO2, lc1, lc2, lr2, lt],
                           ats=(0.4, 0.33, 0.47, 0.26, 0.55))
        lh = _park_beside(tag("r₂", 28, c2), [(T1, H)], segs3, prefer=LEFT, only=True,
                          avoid=[lO1, lO2, lc1, lc2, lr2, lt2, lt])
        ld = _park_beside(tag("r₁ − r₂", 28), [(H, O1)], segs3, prefer=LEFT, only=True,
                          avoid=[lO1, lO2, lc1, lc2, lr2, lt2, lh, lt])
        self.add(rect)
        self.bring_to_back(rect)
        self.play(FadeIn(rect), FadeIn(lt2), FadeOut(lr1), Create(hseg), FadeIn(lh),
                  run_time=0.8)
        self.play(Create(leg), FadeIn(ld), Create(raH), run_time=0.8)

        # ---- the right triangle O₁HO₂
        tri = mk([O1, H, O2], GREY_B, 0.35, stroke_width=0)
        self.add(tri)
        self.bring_to_back(tri, rect)
        self.play(FadeIn(tri), tseg.animate.set_stroke(opacity=1.0), FadeOut(rect),
                  run_time=0.7)
        self.bring_to_front(dO1, dO2, dP)
        self.hold(0.5)

        # ---- the hypotenuse, with P on it, carried over to the square's side
        hyp = VGroup(Line(O1, P, color=c1, stroke_width=6), Line(P, O2, color=c2, stroke_width=6),
                     Dot(P, radius=0.07, color=YELLOW_B))
        self.add(hyp)
        A0, A1 = Q(0, 0), Q(L, 0)
        self.play(_rigid(hyp, [O1, O2], [A0, A1], run_time=1.6))
        check(close(hyp[0].get_start(), A0, 1e-6) and close(hyp[1].get_end(), A1, 1e-6)
              and close(hyp[0].get_end(), Q(r1, 0), 1e-6),
              "the hypotenuse lands on the square's side, P at r₁ from its end")
        sq = Polygon(*square, stroke_color=WHITE, stroke_width=3)
        rmobs = [mk(R, TEAL_D, 0.7, stroke_width=2) for R in rects]
        hole_m = mk(hole, GREY_E, 0.9, stroke_width=2)
        lb1 = tag("r₁", 28, c1).next_to(Line(A0, Q(r1, 0)), DOWN, buff=0.14)
        lb2 = tag("r₂", 28, c2).next_to(Line(Q(r1, 0), A1), DOWN, buff=0.14)
        self.play(Create(sq), FadeIn(lb1), FadeIn(lb2), run_time=0.8)
        self.add(*rmobs)
        self.bring_to_front(hyp)
        self.play(LaggedStart(*[FadeIn(m) for m in rmobs], lag_ratio=0.25), run_time=1.3)
        self.bring_to_front(hyp)          # the group re-added the rectangles on top
        lrr = [tag("r₁r₂", 26).move_to(sum(R) / 4) for R in rects]
        self.play(*[FadeIn(m) for m in lrr], run_time=0.6)

        # ---- the leg O₁H, carried over, is the side of the hole
        legc = leg.copy()
        self.add(hole_m, legc)
        hl0, hl1 = Q(r2, r1), Q(r2, r2)
        self.play(FadeIn(hole_m), _rigid(legc, [O1, H], [hl0, hl1], run_time=1.4))
        check(close(legc.get_start(), hl0, 1e-6) and close(legc.get_end(), hl1, 1e-6),
              "the leg r₁ − r₂ is the hole's side")
        lhole = tag("(r₁−r₂)²", 24).move_to(sum(hole) / 4)
        self.play(FadeIn(lhole), run_time=0.5)

        figure_labels = [lO1, lO2, lr2, lt, lt2, lh, ld, lc1, lc2]
        segs_f = segs3
        _labels_ok(figure_labels, segs_f, "E35 figure")
        sq_segs = (_segs(*square, closed=True) + [s for R in rects for s in _segs(*R, closed=True)]
                   + _segs(*hole, closed=True))
        for m, R in zip(lrr, rects):
            check(all(_pip(c, R) for c in (m.get_corner(UL), m.get_corner(DR))),
                  "each r₁r₂ sits inside its rectangle")
        check(all(_pip(c, hole) for c in (lhole.get_corner(UL), lhole.get_corner(DR))),
              "the hole's label sits inside the hole")
        _labels_ok(lrr + [lhole], [s for s in sq_segs], "E35 square", pad=0.04)
        _labels_ok([lb1, lb2] + figure_labels, segs_f + sq_segs, "E35 all")
        _all_in_frame([circ1, circ2, line, sq], "E35")
        cap = caption("t²  =  (r₁ + r₂)² − (r₁ − r₂)²  =  4r₁r₂     ⟹     t  =  2√(r₁r₂)", 32)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ E27 MIQUEL'S THEOREM

class E27_Miquel(Board):
    """D, E, F on the sides BC, CA, AB. The circles AEF and BFD meet again
    at M. AFME is cyclic, so its exterior angle at M (between MF and the
    extension ME' of EM) equals its interior angle at A: the wedge α turned
    and slid from A fills it exactly. Likewise BDMF: the wedge β fills the
    angle between MD' (DM extended) and MF. On the straight line D'MD the
    angle left over, between ME' and MD, is 180° − α − β = γ: the wedge γ
    fills it. So the exterior angle of CDME at M equals its interior angle
    at C, and CDME is cyclic too: the circle CDE passes through M.
    (Drawn with M inside the triangle.)"""

    def construct(self):
        Am, Bm, Cm = np.array([1.8, 3.6]), np.array([0.0, 0.0]), np.array([7.0, 0.0])
        Fm = Am + 0.45 * (Bm - Am)
        Dm = Bm + 0.45 * (Cm - Bm)
        Em = Cm + 0.65 * (Am - Cm)
        cen = {}
        for key, (X, Y, Z) in {"1": (Am, Em, Fm), "2": (Bm, Fm, Dm), "3": (Cm, Dm, Em)}.items():
            O = _circumcentre(X, Y, Z)[:2]
            cen[key] = (O, float(np.linalg.norm(O - X)))
        xs = min(cen[k][0][0] - cen[k][1] for k in cen)
        xe = max(cen[k][0][0] + cen[k][1] for k in cen)
        ys = min(cen[k][0][1] - cen[k][1] for k in cen)
        ye = max(cen[k][0][1] + cen[k][1] for k in cen)
        F_ = Frame(xs - 0.3, xe + 0.3, ys - 0.05, ye + 0.4)    # room above A
        k = F_.k
        A, B, C, D, E, F = (F_.P(p) for p in (Am, Bm, Cm, Dm, Em, Fm))
        O1, O2, O3 = (F_.P(cen[c][0]) for c in "123")
        R1, R2, R3 = (cen[c][1] * k for c in "123")
        # M: the second meeting point of circles 1 and 2 (F mirrored in O₁O₂)
        u = _unit(O2 - O1)
        M = 2 * (O1 + np.dot(F - O1, u) * u) - F
        n = np.linalg.norm
        check(abs(n(M - O1) - R1) < 1e-9 and abs(n(M - O2) - R2) < 1e-9
              and n(M - F) > 1.0, "M is the other common point of circles AEF, BFD")
        check(_pip(M, [A, B, C]), "M lies inside the triangle")
        Ep, Dp = 2 * M - E, 2 * M - D                 # E', D': EM, DM extended
        al, be, ga = _angle(A, B, C), _angle(B, C, A), _angle(C, A, B)
        check(abs(al + be + ga - PI) < 1e-9, "α + β + γ = 180°")
        slotA, slotB, slotC = (F, Ep), (Dp, F), (Ep, D)
        check(abs(_angle(M, *slotA) - al) < 1e-9, "cyclic AFME: ∠FME' = α")
        check(abs(_angle(M, *slotB) - be) < 1e-9, "cyclic BDMF: ∠D'MF = β")
        check(abs(_angle(M, *slotC) - ga) < 1e-9, "on the line D'D the rest is γ")
        # the three slots lie side by side: D' -> F -> E' -> D, a straight angle
        a0, s0 = _span(M, *slotB)
        a1, s1 = _span(M, *slotA)
        a2, s2 = _span(M, *slotC)
        check(close(np.cos(a0 + s0), np.cos(a1)) and close(np.sin(a0 + s0), np.sin(a1))
              and close(np.cos(a1 + s1), np.cos(a2)) and close(np.sin(a1 + s1), np.sin(a2))
              and abs(s0 + s1 + s2 - PI) < 1e-9, "the slots β, α, γ fill the half-turn at M")
        check(abs(n(M - O3) - R3) < 1e-9, "Miquel: the circle CDE passes through M")

        cA, cB, cC = BLUE_D, GREEN_D, ORANGE
        kA, kB, kC = BLUE_B, GREEN_B, GOLD_D
        rw = 0.72

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3)
        circ1 = Circle(radius=R1, color=kA, stroke_width=3).move_to(O1)
        circ2 = Circle(radius=R2, color=kB, stroke_width=3).move_to(O2)
        circ3 = Circle(radius=R3, color=kC, stroke_width=3).move_to(O3)
        rays = VGroup(Line(M, D, color=WHITE, stroke_width=2.5),
                      Line(M, E, color=WHITE, stroke_width=2.5),
                      Line(M, F, color=WHITE, stroke_width=2.5))
        ext = VGroup(DashedLine(M, M + 1.25 * _unit(Ep - M), color=GREY_A, stroke_width=2.5,
                                dash_length=0.08),
                     DashedLine(M, M + 1.25 * _unit(Dp - M), color=GREY_A, stroke_width=2.5,
                                dash_length=0.08))
        Ex, Dx = M + 1.25 * _unit(Ep - M), M + 1.25 * _unit(Dp - M)

        # every line, circle and mark of the final picture, for the label checks
        segs = (_segs(A, B, C, closed=True) + _circle_segs(O1, R1, 160)
                + _circle_segs(O2, R2, 160) + _circle_segs(O3, R3, 160)
                + _segs(M, D) + _segs(M, E) + _segs(M, F) + _segs(M, Ex) + _segs(M, Dx)
                + _arc_segs(A, rw, *[_span(A, B, C)[0], sum(_span(A, B, C))])
                + _arc_segs(B, rw, *[_span(B, C, A)[0], sum(_span(B, C, A))])
                + _arc_segs(C, rw, *[_span(C, A, B)[0], sum(_span(C, A, B))])
                + _circle_segs(M, rw, 48)
                + sum((_dot_segs(X) for X in (D, E, F, M)), []))
        # each point's label goes in the free sector between the sides and the
        # circles through it, outside the triangle where there is room
        def free(X, cs):
            """Directions (degrees) of the sides' and circles' tangents at X."""
            out = []
            for O in cs:
                tg = _perp(X - O)
                out += [X + tg, X - tg]
            return out
        lA = _park_dirs(tag("A", 30), A, [100, 120, 80, 140, 160], segs, d0=0.3)
        lB = _park_dirs(tag("B", 30), B, [200, 215, 185, 230], segs, d0=0.3, avoid=[lA])
        lC = _park_dirs(tag("C", 30), C, [340, 325, 355, 310], segs, d0=0.3, avoid=[lA, lB])
        # free sectors (outside both circles through the point, outside the
        # triangle): D below BC, E above CA, F left of AB; M on the side of
        # the line D'D away from the wedges
        def mid_dir(X, P, Q):
            return float(np.degrees(_span(X, P, Q)[0] + _span(X, P, Q)[1] / 2))

        def fan(c):
            return [c, c - 7, c + 7, c - 14, c + 14, c - 21, c + 21]
        tD, tE, tF = free(D, [O2, O3]), free(E, [O1, O3]), free(F, [O1, O2])
        # the two tangent rays bounding each free sector (picked by side)
        sD = [p for p in tD if np.dot(p - D, A - D) < 0]
        sE = [p for p in tE if np.dot(p - E, B - E) < 0]
        sF = [p for p in tF if np.dot(p - F, C - F) < 0]
        check(len(sD) == 2 and len(sE) == 2 and len(sF) == 2, "two tangents bound each sector")
        lD = _park_dirs(tag("D", 28), D, fan(mid_dir(D, *sD)), segs, d0=0.25,
                        avoid=[lA, lB, lC])
        lE = _park_dirs(tag("E", 28), E, fan(mid_dir(E, *sE)), segs, d0=0.25,
                        avoid=[lA, lB, lC, lD])
        lF = _park_dirs(tag("F", 28), F, fan(mid_dir(F, *sF)), segs, d0=0.25,
                        avoid=[lA, lB, lC, lD, lE])
        lM = _park_dirs(tag("M", 28), M, fan(np.degrees(_dir(M, (Ep + F) / 2)) + 180), segs,
                        d0=0.28, avoid=[lA, lB, lC, lD, lE, lF])
        # (the nearest free spot must be in the expected sector: outside the
        # triangle for D, E, F, and not on the wedge side of M)
        for lab, X, Y in ((lD, D, A), (lE, E, B), (lF, F, C)):
            check(np.dot(lab.get_center() - X, Y - X) < 0, f"'{lab.text}' outside the triangle")
        check(np.dot(lM.get_center() - M, (Ep + F) / 2 - M) < 0, "'M' off the wedge side")
        vlabels = [lA, lB, lC, lD, lE, lF, lM]

        wA = _wedge(A, B, C, rw, cA, 0.8)
        wB = _wedge(B, C, A, rw, cB, 0.8)
        wC = _wedge(C, A, B, rw, cC, 0.8)
        lal = _angle_label(tag("α", 28, BLUE_B), A, B, C, rw + 0.22, segs=segs,
                           avoid=vlabels)
        lbe = _angle_label(tag("β", 28, GREEN_B), B, C, A, rw + 0.22, segs=segs,
                           avoid=vlabels + [lal])
        lga = _angle_label(tag("γ", 28, kC), C, A, B, rw + 0.22, segs=segs,
                           avoid=vlabels + [lal, lbe])

        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)
        dots = VGroup(*[Dot(X, radius=0.065) for X in (D, E, F)])
        self.play(FadeIn(dots), FadeIn(lD), FadeIn(lE), FadeIn(lF), run_time=0.6)
        self.add(wA, wB, wC)
        self.bring_to_back(wA, wB, wC)
        self.play(FadeIn(wA), FadeIn(wB), FadeIn(wC), FadeIn(lal), FadeIn(lbe), FadeIn(lga),
                  run_time=0.8)

        # ---- the circles AEF and BFD meet again at M
        dM = Dot(M, radius=0.075, color=YELLOW_B)
        self.play(Create(circ1), run_time=1.0)
        self.play(Create(circ2), run_time=1.0)
        self.play(FadeIn(dM), FadeIn(lM), run_time=0.5)
        self.play(Create(rays), Create(ext), run_time=0.9)
        self.bring_to_front(dots, dM)

        def move_wedge(w, V, P, Q, slot, quad, col):
            a_v, s_v = _span(V, P, Q)
            a_m, s_m = _span(M, *slot)
            check(abs(s_v - s_m) < 1e-9, "the wedge and its slot are equal angles")
            turn = (a_m + s_m / 2) - (a_v + s_v / 2)
            turn = (turn + PI) % TAU - PI
            # where the wedge's arms land: exactly the slot's arms
            arms = sorted([(a_v + turn) % TAU, (a_v + s_v + turn) % TAU])
            want = sorted([a_m % TAU, (a_m + s_m) % TAU])
            check(all(min(abs(x - y), TAU - abs(x - y)) < 1e-9 for x, y in zip(arms, want)),
                  "the turned wedge's arms are the slot's arms")
            cp = w.copy()
            hl = mk(quad, col, 0.22, stroke_width=0)
            self.add(hl)
            self.bring_to_back(hl)
            self.play(FadeIn(hl), run_time=0.4)
            self.add(cp)
            self.play(_glide(cp, V, M, turn, run_time=1.6))
            got = Sector(radius=rw, start_angle=a_m, angle=s_m, arc_center=M)
            check(np.allclose(cp.get_center(), got.get_center(), atol=1e-6)
                  and abs(cp.width - got.width) < 1e-6 and abs(cp.height - got.height) < 1e-6,
                  "the wedge lands in its slot")
            self.play(FadeOut(hl), run_time=0.4)
            return cp

        move_wedge(wA, A, B, C, slotA, [A, F, M, E], cA)
        self.bring_to_front(circ1, circ2, rays, ext, dots, dM)
        move_wedge(wB, B, C, A, slotB, [B, D, M, F], cB)
        self.bring_to_front(circ1, circ2, rays, ext, dots, dM)
        # what is left of the straight angle D'MD is 180° − α − β = γ
        straight = Line(Dx, D, color=YELLOW_B, stroke_width=5)
        self.play(Create(straight), run_time=0.7)
        self.play(FadeOut(straight), run_time=0.5)
        move_wedge(wC, C, A, B, slotC, [C, D, M, E], cC)
        self.bring_to_front(circ1, circ2, rays, ext, dots, dM)
        ml = []

        # ---- ∠E'MD = γ = ∠C: CDME is cyclic, the third circle passes through M
        q3 = mk([C, D, M, E], cC, 0.22, stroke_width=0)
        self.add(q3)
        self.bring_to_back(q3)
        self.play(FadeIn(q3), run_time=0.4)
        self.play(Create(circ3), run_time=1.2)
        self.play(FadeOut(q3), run_time=0.4)
        self.bring_to_front(dots, dM)

        _labels_ok(vlabels + [lal, lbe, lga] + ml, segs, "E27 final")
        _all_in_frame([circ1, circ2, circ3, tri], "E27")
        cap = caption("the circles AEF, BFD and CDE pass through one point M", 32)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ E32 TANGENT TO A HYPERBOLA

class E32_HyperbolaTangent(Board):
    """xy = 1 at P = (1, 1): the mirror y = x maps the curve onto itself and
    fixes P, so the tangent at P is perpendicular to it: x + y = 2, which
    meets the axes at (2, 0) and (0, 2) — P is its midpoint, and the
    triangle is the unit square under P plus two half squares: area 2. The
    squeeze (x, y) -> (tx, y/t) slides the hyperbola along itself, keeps
    the axes, straight lines, tangency, midpoints and areas: it carries P to
    (t, 1/t) and the triangle to the one cut off by the tangent there, with
    intercepts 2t and 2/t and the same area, ½ · 2t · 2/t = 2."""

    def construct(self):
        F_ = Frame(-0.42, 3.72, -0.42, 3.72, max_w=6.9,
                   centre=(-3.15, (SAFE_TOP + SAFE_BOTTOM) / 2))
        k = F_.k
        Om = F_.P((0.0, 0.0))
        tt = ValueTracker(1.0)
        n = np.linalg.norm

        def pts(t):
            """O, X, Y, P and the feet of P, for the point P = (t, 1/t)."""
            return dict(O=F_.P((0, 0)), X=F_.P((2 * t, 0)), Y=F_.P((0, 2 / t)),
                        P=F_.P((t, 1 / t)), Px=F_.P((t, 0)), Py=F_.P((0, 1 / t)))

        # ---- the claims, at the start and all along the squeeze
        ts = np.concatenate([np.linspace(1.0, 1.6, 13), np.linspace(1.6, 0.65, 19),
                             np.linspace(0.65, 1.45, 17)])
        for t in ts:
            q = pts(t)
            check(close((q["X"] + q["Y"]) / 2, q["P"]), "P is the midpoint of XY")
            check(abs(abs(area([q["O"], q["X"], q["Y"]])) - 2 * k * k) < 1e-9,
                  "the triangle has area 2")
            check(abs(abs(area([q["O"], q["Px"], q["P"], q["Py"]])) - k * k) < 1e-9,
                  "the rectangle under P has area 1")
            # the line XY touches xy = 1 at P only: xy < 1 elsewhere on it
            for s in np.linspace(0.0, 1.0, 41):
                x, y = 2 * t * (1 - s), 2 / t * s
                if abs(s - 0.5) > 1e-9:
                    check(x * y < 1 - 1e-12, "the line stays below the curve")
            check(_tiles_exactly([[q["O"], q["Px"], q["P"], q["Py"]],
                                  [q["Px"], q["X"], q["P"]], [q["Py"], q["P"], q["Y"]]],
                                 [q["O"], q["X"], q["Y"]], n=40),
                  "rectangle + two corner triangles tile the triangle")
        # the squeeze keeps xy = 1 and areas: det = t · 1/t = 1
        for t in (0.65, 1.45, 1.6):
            for x in (0.3, 1.0, 2.5):
                check(abs((t * x) * ((1 / x) / t) - 1) < 1e-12, "the squeeze keeps xy = 1")

        cP, cR, cC = YELLOW_B, TEAL_D, BLUE_D
        xs = np.linspace(1 / 3.6, 3.6, 400)
        hyp = _curve([F_.P((x, 1 / x)) for x in xs], WHITE, 3.5)
        ax = Line(F_.P((-0.25, 0)), F_.P((3.7, 0)), color=GREY_B, stroke_width=2.5)
        ay = Line(F_.P((0, -0.25)), F_.P((0, 3.7)), color=GREY_B, stroke_width=2.5)
        lx = tag("x", 26, GREY_B).next_to(F_.P((3.62, 0)), DOWN, buff=0.12)
        ly = tag("y", 26, GREY_B).next_to(F_.P((0, 3.58)), LEFT, buff=0.14)
        lcurve = tag("xy = 1", 28).move_to(F_.P((3.15, 0.62)))
        self.play(Create(ax), Create(ay), FadeIn(lx), FadeIn(ly), run_time=0.8)
        self.play(Create(hyp), FadeIn(lcurve), run_time=1.2)

        # ---- the mirror y = x maps the curve onto itself and fixes P
        q1 = pts(1.0)
        mirror = DashedLine(Om, F_.P((3.3, 3.3)), color=GREY_A, stroke_width=2.5,
                            dash_length=0.09)
        dP0 = Dot(q1["P"], radius=0.08, color=cP)
        self.play(Create(mirror), FadeIn(dP0), run_time=0.8)
        ghost = hyp.copy().set_stroke(YELLOW_B, 5, opacity=0.8)
        self.add(ghost)
        self.play(Rotate(ghost, angle=PI, axis=_unit(np.array([1.0, 1.0, 0.0])),
                         about_point=Om), run_time=1.4)
        check(all(abs((p[0] - Om[0]) * (p[1] - Om[1]) - k * k) < 1e-6 * k * k
                  for p in ghost.get_anchors()), "the folded curve is xy = 1 again")
        self.play(FadeOut(ghost), run_time=0.4)

        # ---- so the tangent at P is perpendicular to the mirror: x + y = 2
        def tangent(q):
            return Line(q["X"], q["Y"], color=cP, stroke_width=5)
        tan0 = tangent(q1)
        ra = _ra(q1["P"], q1["X"], F_.P((2.0, 2.0)), 0.2)
        check(abs(np.dot(q1["X"] - q1["Y"], F_.P((1, 1)) - Om)) < 1e-9, "XY ⊥ the mirror")
        self.play(Create(tan0), Create(ra), run_time=1.0)
        tk0 = VGroup(_ticks(q1["X"], q1["P"], 1, cP), _ticks(q1["P"], q1["Y"], 1, cP))
        self.play(FadeIn(tk0), run_time=0.5)
        self.hold(0.4)
        self.play(FadeOut(mirror), FadeOut(ra), run_time=0.5)

        # ---- the triangle: the unit square under P and two half squares
        def pieces(q):
            return VGroup(mk([q["O"], q["Px"], q["P"], q["Py"]], cR, 0.75, stroke_width=1.5),
                          mk([q["Px"], q["X"], q["P"]], cC, 0.6, stroke_width=1.5),
                          mk([q["Py"], q["P"], q["Y"]], cC, 0.6, stroke_width=1.5))

        def incentre(A_, B_, C_):
            a_, b_, c_ = n(B_ - C_), n(C_ - A_), n(A_ - B_)
            return (a_ * A_ + b_ * B_ + c_ * C_) / (a_ + b_ + c_)

        def area_labels(q):
            c1 = (q["O"] + q["P"]) / 2
            c2 = incentre(q["Px"], q["X"], q["P"])
            c3 = incentre(q["Py"], q["P"], q["Y"])
            return VGroup(tag("1", 28).move_to(c1), tag("½", 28).move_to(c2),
                          tag("½", 28).move_to(c3))
        pc0 = pieces(q1)
        al0 = area_labels(q1)
        self.add(pc0)
        self.bring_to_back(pc0)
        self.play(FadeIn(pc0), FadeIn(al0), run_time=0.9)
        self.hold(0.6)

        # ---- the squeeze (x, y) -> (tx, y/t): the curve slides along itself
        rule = tag("(x, y)  →  (tx, y/t)", 30).move_to(np.array([3.6, 2.6, 0.0]))
        keep = tag("(tx) · (y/t)  =  xy", 30).next_to(rule, DOWN, buff=0.4)
        self.play(FadeIn(rule), run_time=0.6)
        self.play(FadeIn(keep), run_time=0.6)

        def fig():
            q = pts(tt.get_value())
            return VGroup(pieces(q), tangent(q),
                          _ticks(q["X"], q["P"], 1, cP), _ticks(q["P"], q["Y"], 1, cP),
                          area_labels(q))
        rx = (0.5, 2.2)                  # two more points of the curve ride along
        riders = [F_.P((x, 1 / x)) for x in rx]
        for t in ts:
            check(all(xs[0] <= t * x <= xs[-1] for x in rx), "the riders stay on the drawn curve")

        def dots():
            t = tt.get_value()
            g = VGroup(Dot(pts(t)["P"], radius=0.08, color=cP))
            for xr in rx:
                g.add(Dot(F_.P((t * xr, 1 / (t * xr))), radius=0.06, color=GREY_A))
            return g

        def axis_labels():
            q = pts(tt.get_value())
            return VGroup(tag("t", 26).next_to(q["Px"], DOWN, buff=0.16),
                          tag("2t", 26, cP).next_to(q["X"], DOWN, buff=0.16),
                          tag("1/t", 26).next_to(q["Py"], LEFT, buff=0.16),
                          tag("2/t", 26, cP).next_to(q["Y"], LEFT, buff=0.16))

        def guides():
            q = pts(tt.get_value())
            return VGroup(DashedLine(q["P"], q["Px"], color=GREY_A, stroke_width=2,
                                     dash_length=0.07),
                          DashedLine(q["P"], q["Py"], color=GREY_A, stroke_width=2,
                                     dash_length=0.07))
        figm = always_redraw(fig)
        dts = always_redraw(dots)
        self.remove(pc0, al0, tan0, tk0, dP0)
        self.add(figm, hyp, dts)
        r0 = VGroup(*[Dot(p, radius=0.06, color=GREY_A) for p in riders])
        self.play(FadeIn(r0), run_time=0.3)
        self.remove(r0)
        self.play(tt.animate.set_value(1.6), run_time=1.8)
        self.play(tt.animate.set_value(0.65), run_time=2.4)
        self.play(tt.animate.set_value(1.45), run_time=1.8)
        figm.clear_updaters()
        dts.clear_updaters()

        q = pts(1.45)
        axl = axis_labels()
        gd = guides()
        lP = tag("P", 28, cP)
        segs = (_segs(*[F_.P((x, 1 / x)) for x in xs[::4]]) + [_seg(ax.get_start(), ax.get_end()),
                                                               _seg(ay.get_start(), ay.get_end())]
                + [_seg(q["X"], q["Y"]), _seg(q["P"], q["Px"]), _seg(q["P"], q["Py"]),
                   _seg(q["O"], q["Px"])]
                + _tick_segs(q["X"], q["P"]) + _tick_segs(q["P"], q["Y"])
                + sum((_dot_segs(F_.P((1.45 * xr, 1 / (1.45 * xr))), 0.07) for xr in rx),
                      []) + _dot_segs(q["P"], 0.09))
        nrm = _unit(np.array([1 / 1.45, 1.45, 0.0]))
        _park_dirs(lP, q["P"], [np.degrees(np.arctan2(nrm[1], nrm[0])) + d
                                for d in (0, 12, -12, 24, -24)], segs, d0=0.3)
        self.add(gd)
        self.bring_to_front(figm, hyp, dts)
        self.play(FadeIn(gd), FadeIn(axl), FadeIn(lP), run_time=0.8)

        # every label clear, at every sampled moment of the squeeze
        for t in ts:
            qq = pts(t)
            al = area_labels(qq)
            for lab, poly in zip(al, ([qq["O"], qq["Px"], qq["P"], qq["Py"]],
                                      [qq["Px"], qq["X"], qq["P"]],
                                      [qq["Py"], qq["P"], qq["Y"]])):
                check(all(_pip(c, poly) for c in (lab.get_corner(UL), lab.get_corner(UR),
                                                   lab.get_corner(DL), lab.get_corner(DR))),
                      f"area label inside its piece at t = {t:.2f}")
            _labels_ok(list(al) + [lcurve, rule, keep, lx, ly],
                       _segs(qq["O"], qq["X"], qq["Y"], closed=True)
                       + _segs(qq["P"], qq["Px"]) + _segs(qq["P"], qq["Py"])
                       + _tick_segs(qq["X"], qq["P"]) + _tick_segs(qq["P"], qq["Y"]),
                       f"E32 at t = {t:.2f}")
            _labels_ok([lcurve, rule, keep, lx, ly],
                       _segs(*[F_.P((x, 1 / x)) for x in xs[::4]])
                       + sum((_dot_segs(F_.P((t * xr, 1 / (t * xr))), 0.07) for xr in rx), []),
                       "E32 curve labels")
        _labels_ok([lP, lcurve, rule, keep, lx, ly] + list(axl), segs, "E32 final")
        _all_in_frame([hyp, ax, ay], "E32")
        cap = caption("tangent at P = (t, 1/t):   ½ · 2t · 2/t  =  2", 32)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ E33 ARCHIMEDES' BROKEN CHORD

class E33_BrokenChord(Board):
    """M is the midpoint of the arc ABC (B nearer C). MA = MC (equal arcs),
    so the turn about M that takes C to A carries the triangle MCB onto
    MAG: G lands on the longer chord AB, because the angles MCB and MAB
    stand on the same arc MB. So AG = BC and MG = MB: MGB is isosceles, and
    the foot F of the perpendicular from M halves GB. Hence
    AF = AG + GF = BC + FB."""

    def construct(self):
        R = 3.0
        O = np.array([-2.2, 0.2, 0.0])
        aA, aC, aB = 195 * DEGREES, -13 * DEGREES, 50 * DEGREES
        aM = (aA + aC) / 2

        def S(a):
            return O + R * _polar(a)
        A, B, C, M = S(aA), S(aB), S(aC), S(aM)
        n = np.linalg.norm
        check(aC < aB < aM < aA, "B lies on the arc ABC between M and C")
        check(n(B - A) > n(C - B), "AB is the longer chord")
        check(abs(n(M - A) - n(M - C)) < 1e-9, "MA = MC")
        turn = _dir(M, A) - _dir(M, C)
        turn = (turn + PI) % TAU - PI
        G = _rot2(B, M, turn)
        F = _foot(M, A, B)
        check(close(_rot2(C, M, turn), A), "the turn about M takes C to A")
        check(_between(G, A, B), "B lands at G on the chord AB")
        check(abs(_angle(C, M, B) - _angle(A, M, G)) < 1e-9 and
              abs(_angle(C, M, B) - _angle(A, M, B)) < 1e-9,
              "∠MCB = ∠MAB (same arc MB)")
        check(abs(n(G - A) - n(C - B)) < 1e-9 and abs(n(G - M) - n(B - M)) < 1e-9,
              "AG = BC and MG = MB")
        check(_between(F, G, B) and abs(n(F - G) - n(B - F)) < 1e-9, "F halves GB")
        check(abs(n(F - A) - (n(B - F) + n(C - B))) < 1e-9, "AF = FB + BC")
        for kk in np.linspace(0, 1, 40):
            for X in (C, B):
                Y = _rot2(X, M, kk * turn)
                check(abs(Y[0]) <= SAFE_X and SAFE_BOTTOM <= Y[1] <= SAFE_TOP,
                      "the turning triangle stays on screen")

        cO, cT = ORANGE, TEAL_B
        circ = Circle(radius=R, color=GREY_B, stroke_width=2.5).move_to(O)
        dots = VGroup(*[Dot(X, radius=0.07) for X in (A, B, C)])
        ab = Line(A, B, color=WHITE, stroke_width=4)
        bc = Line(B, C, color=WHITE, stroke_width=4)
        arcABC = Arc(radius=R, start_angle=aC, angle=aA - aC, arc_center=O,
                     color=YELLOW_B, stroke_width=5)
        tkA = _ticks(S((aA + aM) / 2 - 0.02) , S((aA + aM) / 2 + 0.02), 1, YELLOW_B, 0.16)
        tkC = _ticks(S((aC + aM) / 2 - 0.02), S((aC + aM) / 2 + 0.02), 1, YELLOW_B, 0.16)
        dM = Dot(M, radius=0.08, color=YELLOW_B)
        ma = Line(M, A, color=GREY_A, stroke_width=3)
        mc = Line(M, C, color=GREY_A, stroke_width=3)
        mb = Line(M, B, color=GREY_A, stroke_width=3)
        mf = DashedLine(M, F, color=WHITE, stroke_width=3, dash_length=0.09)
        raF = _ra(F, M, B, 0.2)

        arc_ticks = (_tick_segs(S((aA + aM) / 2 - 0.02), S((aA + aM) / 2 + 0.02), 1, 0.16)
                     + _tick_segs(S((aC + aM) / 2 - 0.02), S((aC + aM) / 2 + 0.02), 1, 0.16))
        # before the turn: the chord MC is there; after it, MC (now MA) is gone
        segs_early = (_circle_segs(O, R, 180) + _segs(A, B) + _segs(B, C) + _segs(M, A)
                      + _segs(M, C) + _segs(M, B) + arc_ticks + _tick_segs(M, A)
                      + _tick_segs(M, C) + sum((_dot_segs(X, 0.09) for X in (A, B, C, M)), []))
        segs = (_circle_segs(O, R, 180) + _segs(A, B) + _segs(B, C) + _segs(M, A)
                + _segs(M, B) + _segs(M, F) + _segs(M, G) + _ra_segs(F, M, B, 0.2)
                + arc_ticks + _tick_segs(M, B, 2)
                + _tick_segs(M, G, 2) + _tick_segs(G, F, 3) + _tick_segs(F, B, 3)
                + sum((_dot_segs(X, 0.09) for X in (A, B, C, M, F, G)), []))
        lA = _park_dirs(tag("A", 30), A, [np.degrees(_dir(O, A)) + d for d in (0, -15, 15, -30)],
                        segs_early + segs, d0=0.3)
        lB = _park_dirs(tag("B", 30), B, [np.degrees(_dir(O, B)) + d for d in (0, 15, -15, 30)],
                        segs_early + segs, d0=0.3, avoid=[lA])
        lC = _park_dirs(tag("C", 30), C, [np.degrees(_dir(O, C)) + d for d in (0, -15, 15, -30)],
                        segs_early + segs, d0=0.3, avoid=[lA, lB])
        lM = _park_dirs(tag("M", 30, YELLOW_B), M, [90, 75, 105, 60, 120], segs_early + segs,
                        d0=0.3, avoid=[lA, lB, lC])
        perpAB = _perp(_unit(B - A))
        if np.dot(perpAB, M - F) > 0:
            perpAB = -perpAB                    # the side of AB away from M
        dd = np.degrees(np.arctan2(perpAB[1], perpAB[0]))
        lF = _park_dirs(tag("F", 30, cT), F, [dd, dd + 15, dd - 15, dd + 30], segs, d0=0.3,
                        avoid=[lA, lB, lC, lM])
        lG = _park_dirs(tag("G", 30, cO), G, [dd, dd - 15, dd + 15, dd - 30], segs, d0=0.3,
                        avoid=[lA, lB, lC, lM, lF])

        self.play(Create(circ), run_time=1.0)
        self.play(FadeIn(dots), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=0.6)
        self.play(Create(ab), Create(bc), run_time=0.9)

        # ---- M halves the arc ABC: equal arcs, equal chords MA = MC
        self.play(Create(arcABC), run_time=1.0)
        self.play(FadeIn(tkA), FadeIn(tkC), FadeIn(dM), FadeIn(lM), run_time=0.6)
        tma, tmc = _ticks(M, A), _ticks(M, C)
        self.play(Create(ma), Create(mc), FadeIn(tma), FadeIn(tmc),
                  arcABC.animate.set_stroke(opacity=0.0), run_time=0.9)
        self.remove(arcABC)
        self.bring_to_front(dots, dM)
        self.hold(0.3)

        # ---- turn MCB about M until C reaches A: B lands at G on AB
        tri = VGroup(mk([M, C, B], cO, 0.45, stroke_width=0),
                     Line(C, B, color=cO, stroke_width=6),
                     Line(M, B, color=GREY_A, stroke_width=3),
                     Line(M, C, color=GREY_A, stroke_width=3))
        bc_hi = Line(B, C, color=cO, stroke_width=6)
        self.play(Create(mb), FadeIn(tri[0]), Create(bc_hi), run_time=0.8)
        self.add(tri)
        piv = Dot(M, radius=0.08, color=YELLOW_B)
        self.add(piv)
        self.play(Rotate(tri, angle=turn, about_point=M), run_time=2.2)
        check(close(tri[1].get_start(), A, 1e-6) and close(tri[1].get_end(), G, 1e-6),
              "the side CB lands on AG")
        check(close(tri[3].get_end(), A, 1e-6), "the side MC lands on MA")
        dG = Dot(G, radius=0.07, color=cO)
        # MC has done its work (it is now MA): it leaves
        self.play(FadeIn(dG), FadeIn(lG), FadeOut(mc), FadeOut(tmc), FadeOut(tma),
                  run_time=0.6)
        tmb, tmg = _ticks(M, B, 2), _ticks(M, G, 2)
        self.play(FadeIn(tmb), FadeIn(tmg), tri[0].animate.set_fill(opacity=0.25), run_time=0.7)

        # ---- MGB is isosceles: the perpendicular from M halves GB at F
        iso = mk([M, G, B], TEAL_D, 0.35, stroke_width=0)
        self.add(iso)
        self.bring_to_back(iso)
        self.play(FadeIn(iso), run_time=0.6)
        self.play(Create(mf), Create(raF), FadeIn(lF), run_time=0.8)
        gf = Line(G, F, color=cT, stroke_width=6)
        fb = Line(F, B, color=cT, stroke_width=6)
        tgf, tfb = _ticks(G, F, 3, cT), _ticks(F, B, 3, cT)
        self.play(Create(gf), Create(fb), FadeIn(tgf), FadeIn(tfb), run_time=0.8)
        self.play(FadeOut(iso), FadeOut(tri[0]), run_time=0.5)
        self.bring_to_front(tri[1], gf, fb, tgf, tfb, dots, dG, dM, piv)

        # ---- read off: AF = AG + GF = BC + FB
        r1 = VGroup(tag("AG", 32, cO), tag("=", 32), tag("BC", 32, cO)).arrange(RIGHT, buff=0.22)
        r2 = VGroup(tag("GF", 32, cT), tag("=", 32), tag("FB", 32, cT)).arrange(RIGHT, buff=0.22)
        r1.move_to(np.array([4.6, 1.3, 0.0]))
        r2.next_to(r1, DOWN, buff=0.5).align_to(r1, LEFT)
        self.play(FadeIn(r1), run_time=0.6)
        self.play(FadeIn(r2), run_time=0.6)

        labels = [lA, lB, lC, lM, lF, lG]
        _labels_ok([lA, lB, lC, lM], segs_early, "E33 before the turn")
        _labels_ok(labels, segs, "E33 figure")
        _labels_ok(labels + [r1, r2], [], "E33 labels apart")
        for r in (r1, r2):
            check(_hit(r, _circle_segs(O, R, 180)) is None, "readouts clear of the circle")
        cap = caption("AF  =  AG + GF  =  BC + FB", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ G26 ARCSIN + ARCCOS

class G26_ArcsinArccos(Board):
    """The right triangle ACB with hypotenuse AB = 1 and leg BC = x: the
    angle α at A has sin α = x, so α = arcsin x; the angle β at B has
    cos β = x, so β = arccos x. Half a turn about the midpoint of AB
    carries the triangle onto BDA, completing the rectangle ACBD: at its
    corner A the copy's angle β sits beside α, and together they fill the
    right angle. As x changes the rectangle changes, the corner does not.
    (Drawn for 0 < x < 1; for negative x, arcsin(−x) = −arcsin x and
    arccos(−x) = π − arccos x give the same sum.)"""

    def construct(self):
        L = 5.6
        A = np.array([-5.3, -1.82, 0.0])
        xv = ValueTracker(0.6)
        cA, cB = BLUE_B, ORANGE
        n = np.linalg.norm

        def geo(x):
            al = float(np.arcsin(x))
            B = A + L * _polar(al)
            C = np.array([B[0], A[1], 0.0])
            D = np.array([A[0], B[1], 0.0])
            return A, B, C, D, al

        # ---- the claims, at every x the sweep visits
        xs = np.linspace(0.4, 0.8, 21)
        for x in list(xs) + [0.6]:
            A_, B, C, D, al = geo(x)
            be = _angle(B, C, A_)
            check(abs(n(B - A_) - L) < 1e-9 and abs(n(B - C) - x * L) < 1e-9,
                  "hypotenuse 1, leg x")
            check(abs(np.dot(B - C, A_ - C)) < 1e-9, "the angle at C is right")
            check(abs(np.sin(_angle(A_, C, B)) - x) < 1e-9, "sin α = x")
            check(abs(np.cos(be) - x) < 1e-9, "cos β = x")
            mid = (A_ + B) / 2
            check(close(2 * mid - C, D) and close(2 * mid - B, A_),
                  "the half-turn about the middle of AB takes C to D, B to A")
            check(abs(_angle(A_, C, B) + _angle(A_, B, D) - PI / 2) < 1e-9,
                  "at A: α + β = 90°")

        def in_angle(s_, col, V, P, Q):
            """Label inside the angle P-V-Q on its bisector, beyond the arc
            and far enough out that its box clears both arms."""
            m = tag(s_, 30, col)
            half = _angle(V, P, Q) / 2
            d = max(1.08, (0.5 * np.hypot(m.width, m.height) + 0.1) / np.sin(half))
            return m.move_to(V + d * angle_mid_dir(V, P, Q))

        def arcs_and_labels(x):
            A_, B, C, D, al = geo(x)
            g = VGroup(angle_arc(A_, C, B, 0.72, cA, 5), angle_arc(B, A_, C, 0.72, cB, 5))
            return g, VGroup(in_angle("α", cA, A_, C, B), in_angle("β", cB, B, A_, C))

        def copy_marks(x):
            A_, B, C, D, al = geo(x)
            g = VGroup(angle_arc(A_, B, D, 0.72, cB, 5), angle_arc(B, A_, D, 0.72, cA, 5))
            return g, VGroup(in_angle("β", cB, A_, B, D), in_angle("α", cA, B, A_, D))

        def side_labels(x):
            A_, B, C, D, al = geo(x)
            l1 = _beside(tag("1", 30), A_, B, C - (A_ + B) / 2, 0.16, at=0.55)
            lx = _beside(tag("x", 30), C, B, RIGHT, 0.16)
            lw = _beside(tag("√(1 − x²)", 28), A_, C, DOWN, 0.16)
            return VGroup(l1, lx, lw)

        A_, B, C, D, al = geo(0.6)
        tri = mk([A_, C, B], BLUE_E, 0.45, stroke_width=0)
        sides = VGroup(Line(A_, B, color=WHITE, stroke_width=4),
                       Line(B, C, color=WHITE, stroke_width=4),
                       Line(C, A_, color=WHITE, stroke_width=4))
        raC = _ra(C, A_, B, 0.24)
        arcs, alabels = arcs_and_labels(0.6)
        sl = side_labels(0.6)
        self.play(FadeIn(tri), Create(sides), Create(raC), run_time=1.1)
        self.play(FadeIn(sl), run_time=0.6)

        # ---- the two acute angles, read as arcsin x and arccos x
        row1 = Text("sin α = x   ⟹   α = arcsin x", font_size=28,
                    t2c={"α": cA, "arcsin x": cA})
        row1.move_to(np.array([0.75 + row1.width / 2, 2.55, 0.0]))
        row2 = Text("cos β = x   ⟹   β = arccos x", font_size=28,
                    t2c={"β": cB, "arccos x": cB})
        row2.next_to(row1, DOWN, buff=0.5).align_to(row1, LEFT)
        check(row1.get_right()[0] <= SAFE_X and row2.get_right()[0] <= SAFE_X,
              "the readout rows fit")
        self.play(Create(arcs[0]), FadeIn(alabels[0]), run_time=0.7)
        self.play(FadeIn(row1), run_time=0.8)
        self.play(Create(arcs[1]), FadeIn(alabels[1]), run_time=0.7)
        self.play(FadeIn(row2), run_time=0.8)
        self.hold(0.4)

        # ---- half a turn about the midpoint of AB: the rectangle ACBD
        cm, cl = copy_marks(0.6)
        piece = VGroup(mk([A_, C, B], TEAL_E, 0.45, stroke_width=0),
                       Line(A_, C, color=WHITE, stroke_width=4),
                       Line(C, B, color=WHITE, stroke_width=4),
                       angle_arc(A_, C, B, 0.72, cA, 5), angle_arc(B, A_, C, 0.72, cB, 5))
        mid = (A_ + B) / 2
        for kk in np.linspace(0.0, 1.0, 61):
            for X in (A_, B, C):
                Y = _rot2(X, mid, kk * PI)
                check(abs(Y[0]) <= SAFE_X and SAFE_BOTTOM <= Y[1] <= SAFE_TOP,
                      "the turning copy stays in the safe area")
        dmid = Dot(mid, radius=0.06, color=YELLOW_B)
        self.add(piece, dmid)
        self.play(Rotate(piece, angle=PI, about_point=mid), run_time=1.8)
        check(close(piece[1].get_start(), B, 1e-6) and close(piece[1].get_end(), D, 1e-6)
              and close(piece[2].get_end(), A_, 1e-6), "the copy lands on BDA")
        self.remove(piece[3], piece[4])
        self.add(cm)
        self.play(FadeIn(cl), FadeOut(dmid), run_time=0.6)
        raA = _ra(A_, C, D, 0.3, YELLOW_B, 3)
        row3 = Text("α + β  =  90°", font_size=32, t2c={"α": cA, "β": cB})
        row3.next_to(row2, DOWN, buff=0.5).align_to(row1, LEFT)
        self.play(Create(raA), FadeIn(row3), run_time=0.9)
        self.hold(0.5)

        # ---- any x: the rectangle changes, its corner stays a right angle
        def whole():
            x = xv.get_value()
            A_, B, C, D, al = geo(x)
            g1, l1 = arcs_and_labels(x)
            g2, l2 = copy_marks(x)
            return VGroup(mk([A_, C, B], BLUE_E, 0.45, stroke_width=0),
                          mk([B, D, A_], TEAL_E, 0.45, stroke_width=0),
                          Line(A_, B, color=WHITE, stroke_width=4),
                          Line(B, C, color=WHITE, stroke_width=4),
                          Line(C, A_, color=WHITE, stroke_width=4),
                          Line(B, D, color=WHITE, stroke_width=4),
                          Line(D, A_, color=WHITE, stroke_width=4),
                          _ra(C, A_, B, 0.24), _ra(A_, C, D, 0.3, YELLOW_B, 3),
                          g1, g2, l1, l2, side_labels(x))
        live = always_redraw(whole)
        # swap the static figure (every piece of it) for the live one
        for m in list(self.mobjects):
            if m not in (row1, row2, row3):
                self.remove(m)
        self.add(live)
        self.play(xv.animate.set_value(0.8), run_time=1.4)
        self.play(xv.animate.set_value(0.4), run_time=1.8)
        self.play(xv.animate.set_value(0.6), run_time=1.2)
        live.clear_updaters()

        # every label clear of the lines, marks and other labels at every x
        for x in xs:
            A_, B, C, D, al = geo(x)
            g1, l1 = arcs_and_labels(x)
            g2, l2 = copy_marks(x)
            segs = (_segs(A_, C, B, D, closed=True) + _segs(A_, B)
                    + _ra_segs(C, A_, B, 0.24) + _ra_segs(A_, C, D, 0.3)
                    + _angle_segs(A_, C, B, 0.72) + _angle_segs(B, A_, C, 0.72)
                    + _angle_segs(A_, B, D, 0.72) + _angle_segs(B, A_, D, 0.72))
            _labels_ok(list(l1) + list(l2) + list(side_labels(x)) + [row1, row2, row3],
                       segs, f"G26 at x = {x:.2f}")
        cap = caption("arcsin x  +  arccos x  =  π/2", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ G18 THE PROJECTION FORMULA

class G18_ProjectionFormula(Board):
    """a = b cos C + c cos B. The altitude from A meets the line BC at H.
    Each side, projected straight down onto BC (every point dropping
    perpendicular to BC), casts a shadow: the right triangle ABH gives the
    shadow of c, c cos B; the right triangle AHC gives the shadow of b,
    b cos C. With B and C acute the two shadows meet at H and tile BC. With
    B obtuse, H falls outside, beyond B: the shadow of b, HC, overshoots BC
    by the shadow of c, HB = c cos(180° − B) = −c cos B, so again
    a = b cos C − (−c cos B) = b cos C + c cos B. (If C is the obtuse one,
    swap the roles of B and C.)"""

    def construct(self):
        cb_, cc_ = TEAL_B, ORANGE
        yb = -1.15                       # the base line BC in both panels
        y1, y2 = yb - 0.32, yb - 0.64    # shadow levels
        panels = [
            dict(A=np.array([-4.15, 2.35, 0.0]), B=np.array([-5.95, yb, 0.0]),
                 C=np.array([-0.95, yb, 0.0])),
            dict(A=np.array([0.95, 2.35, 0.0]), B=np.array([2.55, yb, 0.0]),
                 C=np.array([6.05, yb, 0.0])),
        ]
        n = np.linalg.norm
        for k, p in enumerate(panels):
            A, B, C = p["A"], p["B"], p["C"]
            H = _foot(A, B, C)
            p["H"] = H
            a, b, c = n(C - B), n(A - C), n(B - A)
            Bang, Cang = _angle(B, A, C), _angle(C, A, B)
            check(abs(np.dot(A - H, C - B)) < 1e-9, "AH ⊥ BC")
            check(abs(a - (b * np.cos(Cang) + c * np.cos(Bang))) < 1e-9, "a = b cos C + c cos B")
            if k == 0:
                check(Bang < PI / 2 and Cang < PI / 2 and _between(H, B, C),
                      "acute B, C: the foot H lies on BC")
                check(abs(n(H - B) - c * np.cos(Bang)) < 1e-9 and abs(n(C - H) - b * np.cos(Cang)) < 1e-9,
                      "BH = c cos B, HC = b cos C")
            else:
                check(Bang > PI / 2 and _between(B, H, C), "obtuse B: H falls beyond B")
                check(abs(n(B - H) + c * np.cos(Bang)) < 1e-9 and abs(n(C - H) - b * np.cos(Cang)) < 1e-9,
                      "HB = −c cos B, HC = b cos C")
                check(abs(n(C - H) - n(B - H) - a) < 1e-9, "a = HC − HB")

        def drop(seg_from, seg_to):
            """The vertical projection: each point of the segment moves
            straight down (linear interpolation of a segment onto a
            parallel-shadow keeps every point's x)."""
            (p0, p1), (q0, q1) = seg_from, seg_to
            check(abs(p0[0] - q0[0]) < 1e-9 and abs(p1[0] - q1[0]) < 1e-9,
                  "the shadow lies straight below the side")

        for k, p in enumerate(panels):
            A, B, C, H = p["A"], p["B"], p["C"], p["H"]
            sa = Line(B, C, color=WHITE, stroke_width=4)
            sb = Line(C, A, color=cb_, stroke_width=4)
            sc = Line(A, B, color=cc_, stroke_width=4)
            lA = tag("A", 30).next_to(A, UP, buff=0.14)
            ylow = y1 if k == 0 else y2          # the altitude runs down to the shadows
            segs = (_segs(A, B, C, closed=True) + _segs(A, H)
                    + [_seg(np.array([H[0], ylow, 0]), H)]
                    + _ra_segs(H, A, C, 0.22)
                    + [_seg(np.array([B[0] if k == 0 else H[0], y1, 0]),
                            np.array([C[0], y1, 0]))])
            if k == 1:
                segs += _segs(H, B) + [_seg(np.array([H[0], y2, 0]), np.array([B[0], y2, 0]))]
            lB = _park_dirs(tag("B", 30), B, ([200, 215, 185, 160] if k == 0
                                              else [60, 50, 70, 80, 40]),
                            segs, d0=0.3, d1=1.0)
            lC = _park_dirs(tag("C", 30), C, [340, 325, 355, 20], segs, d0=0.3, avoid=[lB])
            lc = _park_beside(tag("c", 30, cc_), [(A, B)], segs, prefer=(B - C) if k == 0 else C - B,
                              only=True, avoid=[lA, lB, lC], ats=(0.55, 0.45, 0.62, 0.38))
            lb = _park_beside(tag("b", 30, cb_), [(A, C)], segs, prefer=C - B, only=True,
                              avoid=[lA, lB, lC, lc], ats=(0.45, 0.55, 0.38, 0.62))
            p["base"] = VGroup(sa, sb, sc)
            p["labels"] = [lA, lB, lC, lc, lb]
            p["segs"] = segs
            self.play(Create(sa), Create(sb), Create(sc), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                      FadeIn(lc), FadeIn(lb), run_time=1.1)

            # ---- the altitude from A, and its foot H (beyond B when B is obtuse)
            alt = DashedLine(A, np.array([H[0], ylow, 0]), color=GREY_A, stroke_width=2.5,
                             dash_length=0.09)
            raH = _ra(H, A, C, 0.22)
            lH = _park_dirs(tag("H", 28, GREY_A), H, [160, 145, 175, 130], segs, d0=0.28,
                            avoid=[lA, lB, lC, lc, lb])
            extra = []
            if k == 1:
                ext = DashedLine(B, H, color=GREY_A, stroke_width=2.5, dash_length=0.09)
                extra = [Create(ext)]
            self.play(Create(alt), Create(raH), FadeIn(lH), *extra, run_time=0.8)
            p["labels"].append(lH)

            # ---- the shadows: each side drops straight down onto its level
            lev_c = y1 if k == 0 else y2
            hl_c = mk([A, B, H], cc_, 0.18, stroke_width=0)
            self.add(hl_c)
            self.bring_to_back(hl_c)
            shc = Line(A, B, color=cc_, stroke_width=9)
            tgt_c = (np.array([H[0], lev_c, 0]), np.array([B[0], lev_c, 0]))
            drop((A, B), tgt_c)
            self.play(FadeIn(hl_c), run_time=0.4)
            self.add(shc)
            self.play(Transform(shc, Line(*tgt_c, color=cc_, stroke_width=9)), run_time=1.2)
            txt_c = "c cos B" if k == 0 else "−c cos B"
            lsc = tag(txt_c, 24, cc_)
            lsc.next_to(shc, DOWN, buff=0.14)
            self.play(FadeIn(lsc), FadeOut(hl_c), run_time=0.5)

            hl_b = mk([A, H, C], cb_, 0.18, stroke_width=0)
            self.add(hl_b)
            self.bring_to_back(hl_b)
            shb = Line(A, C, color=cb_, stroke_width=9)
            tgt_b = (np.array([H[0], y1, 0]), np.array([C[0], y1, 0]))
            drop((A, C), tgt_b)
            self.play(FadeIn(hl_b), run_time=0.4)
            self.add(shb)
            self.play(Transform(shb, Line(*tgt_b, color=cb_, stroke_width=9)), run_time=1.2)
            lsb = tag("b cos C", 24, cb_).next_to(shb, DOWN, buff=0.14)   # under all of HC
            check(lsb.get_left()[0] > B[0] + 0.1 or k == 0,
                  "the label of HC clears the shadow HB below it")
            self.play(FadeIn(lsb), FadeOut(hl_b), run_time=0.5)

            # ---- the side a, against the shadows
            ya_ = y2 - 0.52
            bar, la = _bracket(np.array([B[0], ya_, 0]), np.array([C[0], ya_, 0]),
                               DOWN, "a", gap=0.0, color=WHITE, size=30, tick=0.1, buff=0.08)
            guides = VGroup(*[DashedLine(np.array([X[0], yb, 0]), np.array([X[0], ya_, 0]),
                                         color=GREY_B, stroke_width=1.5, dash_length=0.06)
                              for X in ((B, C) if k == 0 else (C,))])
            if k == 0:
                row = Text("a  =  c cos B  +  b cos C", font_size=28,
                           t2c={"c cos B": cc_, "b cos C": cb_})
            else:
                row = Text("a  =  b cos C  −  (−c cos B)", font_size=28,
                           t2c={"b cos C": cb_, "(−c cos B)": cc_})
            row.move_to(np.array([(B[0] + C[0]) / 2 if k == 0 else (A[0] + C[0]) / 2, 3.35, 0]))
            self.play(Create(bar), FadeIn(la), Create(guides), run_time=0.8)
            self.play(FadeIn(row), run_time=0.7)
            p["labels"] += [lsc, lsb, la, row]
            p["segs"] += (_segs(*[bar[0].get_start(), bar[0].get_end()])
                          + [_seg(bar[1].get_start(), bar[1].get_end()),
                             _seg(bar[2].get_start(), bar[2].get_end())]
                          + [_seg(g.get_start(), g.get_end()) for g in guides]
                          + [_seg(*tgt_c), _seg(*tgt_b)])
            self.hold(0.4)

        for k, p in enumerate(panels):
            _labels_ok(p["labels"], p["segs"], f"G18 panel {k + 1}")
        _labels_ok(panels[0]["labels"] + panels[1]["labels"], [], "G18 both panels")
        cap = caption("a  =  b cos C  +  c cos B", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ G23 15° AND 75°

class G23_FifteenSeventyFive(Board):
    """The 30°–60°–90° triangle ACB: BC = 1, CA = √3, AB = 2. Swing AB about
    A down onto the line, past the 30° corner: AD = 2, so ABD is isosceles
    and its two base angles, slid and half-turned to A, fill the exterior
    angle 30° there: each is 15°. In the right triangle BCD, with BC = 1,
    CD = 2 + √3 and ∠CBD = 75°: tan 75° = 2 + √3. Swing AB the other way, onto
    D' beyond C: D'C = 2 − √3. The two swings trace the semicircle on D'D,
    so ∠D'BD = 90°; the 15° wedge at D, turned by 90°, fits ∠D'BC (its arms
    turn onto BD' ⊥ BD and BC ⊥ DC): tan 15° = (2 − √3)/1."""

    def construct(self):
        k = 2.68
        x0, y0 = -4.82, -1.72

        def P(x, y):
            return np.array([x0 + k * x, y0 + k * y, 0.0])
        r3 = np.sqrt(3.0)
        C, B, A = P(0, 0), P(0, 1), P(r3, 0)
        D, Dp = P(r3 + 2, 0), P(r3 - 2, 0)
        n = np.linalg.norm
        check(abs(n(B - A) - 2 * k) < 1e-9 and abs(n(A - C) - r3 * k) < 1e-9,
              "the 30-60-90 triangle: 1, √3, 2")
        check(abs(_angle(A, B, C) - PI / 6) < 1e-9, "30° at A")
        check(abs(n(D - A) - 2 * k) < 1e-9 and abs(n(Dp - A) - 2 * k) < 1e-9, "AD = AD' = AB = 2")
        check(abs(_angle(B, A, D) - _angle(D, A, B)) < 1e-9 and
              abs(_angle(D, A, B) - PI / 12) < 1e-9 and abs(_angle(B, A, D) - PI / 12) < 1e-9,
              "base angles of ABD: 15° each")
        check(abs(_angle(B, C, D) - 5 * PI / 12) < 1e-9, "∠CBD = 75°")
        check(abs(n(D - C) / n(B - C) - (2 + r3)) < 1e-9, "tan 75° = 2 + √3")
        check(abs(np.dot(Dp - B, D - B)) < 1e-9, "∠D'BD = 90° (Thales)")
        check(abs(_angle(B, Dp, C) - PI / 12) < 1e-9 and
              abs(n(C - Dp) / n(B - C) - (2 - r3)) < 1e-9, "tan 15° = 2 − √3")
        check(abs(np.tan(PI / 12) - (2 - r3)) < 1e-12 and abs(np.tan(5 * PI / 12) - (2 + r3)) < 1e-12,
              "the values")

        cY = YELLOW_B
        rw = 1.6                          # both base-angle wedges (and the copy at B)
        rD = rw
        lev1, lev2 = y0 - 0.3, y0 - 0.72

        def dim(p, q, text, color):
            """Length bar below the line with ticks pointing up (toward the
            figure) and its label hanging below: it spans the whole segment."""
            bar = VGroup(Line(p, q, color=color, stroke_width=3),
                         Line(p, p + 0.12 * UP, color=color, stroke_width=3),
                         Line(q, q + 0.12 * UP, color=color, stroke_width=3))
            lab = tag(text, 28, color).next_to(bar[0], DOWN, buff=0.12)
            return bar, lab, [(p, q), (p, p + 0.12 * UP), (q, q + 0.12 * UP)]
        br1, lb1, sb1 = dim(np.array([C[0], lev2, 0]), np.array([D[0], lev2, 0]), "2 + √3", ORANGE)
        br2, lb2, sb2 = dim(np.array([Dp[0], lev2, 0]), np.array([C[0], lev2, 0]), "2 − √3", TEAL_B)

        # every line and mark of the final picture, for the label checks
        segs = (_segs(Dp, D) + _segs(B, C) + _segs(A, B) + _segs(B, D) + _segs(B, Dp)
                + _arc_segs(A, 2 * k, 0.0, PI, 90) + _ra_segs(C, A, B, 0.24)
                + _ra_segs(B, Dp, D, 0.26) + _angle_segs(A, B, C, 0.85)
                + _angle_segs(B, C, D, 0.6)
                + _arc_segs(D, rD, PI - PI / 12, PI, 6) + _segs(D, D + rD * _polar(PI - PI / 12))
                + _arc_segs(B, rD, 3 * PI / 2 - PI / 12, 3 * PI / 2, 6)
                + _segs(B, B + rD * _polar(3 * PI / 2 - PI / 12))
                + _tick_segs(A, B) + _tick_segs(A, D) + sb1 + sb2
                + sum((_dot_segs(X, 0.08) for X in (D, Dp)), []))
        segs_early = segs + _arc_segs(B, rw, 5 * PI / 3, 5 * PI / 3 + PI / 12, 6)

        base = Line(C, A, color=WHITE, stroke_width=4)
        bc = Line(B, C, color=WHITE, stroke_width=4)
        ab = Line(A, B, color=WHITE, stroke_width=4)
        raC = _ra(C, A, B, 0.24)
        a30 = angle_arc(A, B, C, 0.85, BLUE_B, 4)
        lA = tag("A", 30).move_to(np.array([A[0], lev1, 0.0]))
        lB = tag("B", 30).next_to(B, LEFT, buff=0.2)
        lC = tag("C", 30).move_to(np.array([C[0], lev1, 0.0]))
        lr3 = tag("√3", 30).move_to(np.array([(C[0] + A[0]) / 2, lev1, 0.0]))
        l1 = _park_beside(tag("1", 30), [(C, B)], segs, prefer=RIGHT, only=True,
                          avoid=[lA, lB, lC, lr3])
        l2 = _park_beside(tag("2", 30), [(A, B)], segs, prefer=C - A, only=True,
                          avoid=[lA, lB, lC, lr3, l1], ats=(0.42, 0.36, 0.48, 0.3))
        l30 = _angle_label(tag("30°", 26, BLUE_B), A, B, C, 1.0, segs=segs,
                           avoid=[lA, lB, lC, lr3, l1, l2])
        lD = tag("D", 30).next_to(D, RIGHT, buff=0.18)
        lDp = tag("D'", 30).next_to(Dp, LEFT, buff=0.18)
        l2b = tag("2", 30).move_to(np.array([A[0] + 0.7 * (D[0] - A[0]), lev1, 0.0]))
        l15 = _angle_label(tag("15°", 26, cY), D, B, A, rD + 0.15, segs=segs,
                           avoid=[lD, l2b, lA])
        l75 = _angle_label(tag("75°", 26, ORANGE), B, C, D, 0.75, segs=segs,
                           avoid=[lB, l1, l2, l30], fracs=(0.5, 0.4, 0.6, 0.3))
        labels = [l1, lr3, l2, lA, lB, lC, l30, lD, l2b, l15, l75, lb1, lDp, lb2]
        _labels_ok(labels, segs, "G23 final")
        _labels_ok([l1, lr3, l2, lA, lB, lC, l30], segs_early, "G23 with the wedge at B")

        self.play(Create(base), Create(bc), Create(ab), Create(raC), FadeIn(lA), FadeIn(lB),
                  FadeIn(lC), run_time=1.2)
        self.play(FadeIn(l1), FadeIn(lr3), FadeIn(l2), Create(a30), FadeIn(l30), run_time=0.8)

        # ---- swing AB about A onto the line, past the 30° corner: AD = 2
        swing = Line(A, B, color=cY, stroke_width=5)
        trace = DashedVMobject(Arc(radius=2 * k, start_angle=5 * PI / 6, angle=-5 * PI / 6,
                                   arc_center=A, color=GREY_B, stroke_width=2.5), num_dashes=60)
        ext = Line(A, D, color=WHITE, stroke_width=4)
        self.add(swing)
        self.play(Rotate(swing, angle=-5 * PI / 6, about_point=A), Create(trace), run_time=2.0)
        check(close(swing.get_end(), D, 1e-6), "the swung side lands at D")
        dD = Dot(D, radius=0.07)
        self.add(ext)
        self.play(FadeOut(swing), FadeIn(dD), FadeIn(lD), FadeIn(l2b), run_time=0.6)
        bd = Line(B, D, color=WHITE, stroke_width=4)
        tkAB, tkAD = _ticks(A, B, 1, cY), _ticks(A, D, 1, cY)
        self.play(Create(bd), FadeIn(tkAB), FadeIn(tkAD), run_time=0.8)

        # ---- the equal base angles of ABD, moved to A, fill its exterior 30°
        wD = _wedge(D, B, A, rD, cY, 0.7)
        wB = _wedge(B, A, D, rw, cY, 0.7)
        self.play(FadeIn(wD), FadeIn(wB), run_time=0.5)
        cD, cB_ = wD.copy(), wB.copy()
        mAB = (A + B) / 2
        self.add(cD, cB_)
        self.play(cD.animate.shift(A - D), Rotate(cB_, angle=PI, about_point=mAB), run_time=1.8)
        a1, s1 = _span(A, B, C)
        got = Sector(radius=rD, start_angle=PI - PI / 12, angle=PI / 12, arc_center=A)
        check(np.allclose(cD.get_center(), got.get_center(), atol=1e-6),
              "the wedge from D lands at A along AC")
        got2 = Sector(radius=rw, start_angle=5 * PI / 6, angle=PI / 12, arc_center=A)
        check(np.allclose(cB_.get_center(), got2.get_center(), atol=1e-6),
              "the wedge from B lands at A along AB")
        check(abs(a1 - 5 * PI / 6) < 1e-9 and abs(s1 - PI / 6) < 1e-9,
              "together they fill ∠BAC = 30°: [150°, 165°] and [165°, 180°]")
        self.play(FadeIn(l15), run_time=0.5)
        self.play(FadeOut(cD), FadeOut(cB_), FadeOut(wB), run_time=0.6)

        # ---- the right triangle BCD: ∠CBD = 75°, tan 75° = (2 + √3)/1
        a75 = angle_arc(B, C, D, 0.6, ORANGE, 4)
        self.play(Create(a75), FadeIn(l75), run_time=0.7)
        self.play(Create(br1), FadeIn(lb1), run_time=0.8)
        self.hold(0.4)

        # ---- swing AB the other way, onto D' beyond C: D'C = 2 − √3
        swing2 = Line(A, B, color=cY, stroke_width=5)
        trace2 = DashedVMobject(Arc(radius=2 * k, start_angle=5 * PI / 6, angle=PI / 6,
                                    arc_center=A, color=GREY_B, stroke_width=2.5), num_dashes=12)
        self.add(swing2)
        self.play(Rotate(swing2, angle=PI / 6, about_point=A), Create(trace2), run_time=1.3)
        check(close(swing2.get_end(), Dp, 1e-6), "the swung side lands at D'")
        dDp = Dot(Dp, radius=0.07)
        ext2 = Line(C, Dp, color=WHITE, stroke_width=4)
        self.add(ext2)
        self.play(FadeOut(swing2), FadeIn(dDp), FadeIn(lDp), Create(br2), FadeIn(lb2),
                  run_time=0.8)

        # ---- B is on the semicircle on D'D: ∠D'BD = 90°
        bdp = Line(B, Dp, color=WHITE, stroke_width=4)
        raB = _ra(B, Dp, D, 0.26, WHITE, 2.5)
        self.play(Create(bdp), Create(raB), run_time=0.8)

        # ---- the 15° wedge at D, turned by 90°, fits ∠D'BC
        cw = wD.copy()
        self.add(cw)
        self.play(_glide(cw, D, B, PI / 2, run_time=1.8))
        got3 = Sector(radius=rD, start_angle=3 * PI / 2 - PI / 12, angle=PI / 12, arc_center=B)
        check(np.allclose(cw.get_center(), got3.get_center(), atol=1e-6)
              and abs(cw.width - got3.width) < 1e-6 and abs(cw.height - got3.height) < 1e-6,
              "the turned wedge fills ∠D'BC")
        check(abs(_angle(B, Dp, C) - _angle(D, B, C)) < 1e-9, "∠D'BC = ∠BDC")
        self.bring_to_front(bc, bdp, raB)
        _all_in_frame([trace, trace2], "G23 arcs")
        cap = caption("tan 15°  =  2 − √3          tan 75°  =  2 + √3", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ G24 REGULAR POLYGON FROM ITS RADIUS

class G24_PolygonFromRadius(Board):
    """The regular n-gon (here n = 7) in a circle of radius r. The triangle
    OV₀V₁ has two sides r and the angle 2π/n between them; its height on
    the side OV₀ is the leg opposite that angle in a right triangle with
    hypotenuse r: r·sin(2π/n). So it has area ½·r·r sin(2π/n). Turned about
    O by 2π/n again and again, it lands on every one of the n triangles,
    which tile the polygon: A = (n/2) r² sin(2π/n)."""

    def construct(self):
        nn = 7
        R = 3.0
        O = np.array([-3.35, 0.3, 0.0])
        th = TAU / nn
        V = [O + R * _polar(k * th) for k in range(nn)]
        H1 = np.array([V[1][0], O[1], 0.0])
        n_ = np.linalg.norm
        h = R * np.sin(th)
        check(all(abs(n_(V[k] - O) - R) < 1e-9 for k in range(nn)), "all vertices on the circle")
        check(all(abs(n_(V[(k + 1) % nn] - V[k]) - n_(V[1] - V[0])) < 1e-9 for k in range(nn)),
              "a regular polygon: equal sides")
        check(abs(_angle(O, V[0], V[1]) - th) < 1e-9, "apex angle 2π/n")
        check(abs(n_(V[1] - H1) - h) < 1e-9 and abs(np.dot(V[1] - H1, V[0] - O)) < 1e-9,
              "the height on OV₀ is r sin(2π/n)")
        check(_between(H1, O, V[0]), "the foot lies on OV₀ (2π/n < 90°)")
        T0 = [O, V[0], V[1]]
        check(abs(abs(area(T0)) - 0.5 * R * h) < 1e-9, "area ½ · r · r sin(2π/n)")
        tris = [[O, V[k], V[(k + 1) % nn]] for k in range(nn)]
        for k in range(nn):
            check(all(close(_rot2(p, O, k * th), q) for p, q in zip(T0, tris[k])),
                  "the k-th turn of T₀ is the k-th triangle")
        check(_tiles_exactly(tris, V), "the n triangles tile the polygon")
        check(abs(abs(area(V)) - nn / 2 * R * R * np.sin(th)) < 1e-9, "A = (n/2) r² sin(2π/n)")

        cT = [BLUE_D, TEAL_D]
        circ = Circle(radius=R, color=GREY_B, stroke_width=2).move_to(O)
        poly = Polygon(*V, stroke_color=WHITE, stroke_width=3)
        dO = Dot(O, radius=0.06)
        self.play(Create(circ), run_time=0.9)
        self.play(Create(poly), FadeIn(dO), run_time=1.0)

        # ---- one triangle: sides r, r and the angle 2π/n between them
        t0 = mk(T0, cT[0], 0.8, stroke_width=2)
        tk = VGroup(_ticks(O, V[0]), _ticks(O, V[1]))
        aO = angle_arc(O, V[0], V[1], 0.6, YELLOW_B, 4)
        segs = (_segs(*V, closed=True) + _circle_segs(O, R, 140) + _segs(O, V[0]) + _segs(O, V[1])
                + _segs(V[1], H1) + _ra_segs(H1, V[1], V[0], 0.2) + _angle_segs(O, V[0], V[1], 0.6)
                + _tick_segs(O, V[0]) + _tick_segs(O, V[1]) + _dot_segs(O, 0.08))
        lr0 = _park_beside(tag("r", 30), [(O, V[0])], segs, prefer=DOWN, only=True,
                           ats=(0.3, 0.25, 0.36, 0.2))
        lr1 = _park_beside(tag("r", 30), [(O, V[1])], segs, prefer=_perp(V[1] - O), only=True,
                           avoid=[lr0], ats=(0.55, 0.62, 0.48))
        lang = _angle_label(tag("2π/n", 26, YELLOW_B), O, V[0], V[1], 0.75, segs=segs,
                            avoid=[lr0, lr1])
        self.add(t0)
        self.bring_to_back(t0)
        self.play(FadeIn(t0), FadeIn(tk), FadeIn(lr0), FadeIn(lr1), run_time=0.8)
        self.play(Create(aO), FadeIn(lang), run_time=0.7)

        # ---- its height on OV₀: r sin(2π/n)
        hl = DashedLine(V[1], H1, color=WHITE, stroke_width=3, dash_length=0.09)
        raH = _ra(H1, V[1], V[0], 0.2)
        xb = O[0] + R + 0.42
        gd = DashedLine(V[1], np.array([xb + 0.1, V[1][1], 0]), color=GREY_B, stroke_width=1.5,
                        dash_length=0.07)
        gd0 = DashedLine(V[0], np.array([xb + 0.1, O[1], 0]), color=GREY_B, stroke_width=1.5,
                         dash_length=0.07)
        bar = VGroup(Line(np.array([xb, O[1], 0]), np.array([xb, V[1][1], 0]), color=YELLOW_B,
                          stroke_width=3),
                     Line(np.array([xb - 0.1, O[1], 0]), np.array([xb + 0.1, O[1], 0]),
                          color=YELLOW_B, stroke_width=3),
                     Line(np.array([xb - 0.1, V[1][1], 0]), np.array([xb + 0.1, V[1][1], 0]),
                          color=YELLOW_B, stroke_width=3))
        lh = tag("r sin(2π/n)", 28, YELLOW_B).next_to(bar[0], RIGHT, buff=0.16)
        check(all(n_(p - O) > R for p in (gd.get_end(), gd0.get_end()))
              and _hit(lh, _circle_segs(O, R, 140)) is None, "the dimension stays outside the circle")
        self.play(Create(hl), Create(raH), run_time=0.8)
        self.play(Create(gd), Create(gd0), Create(bar), FadeIn(lh), run_time=0.9)

        # ---- its area, read off as ½ · base · height
        def icon(pts, col, s=0.32):
            c = sum(pts) / len(pts)
            return mk([c + s * (p - c) for p in pts], col, 0.85, stroke_width=1.5)
        ic1 = icon(T0, cT[0])
        row1 = VGroup(ic1, tag("=", 32), tag("½ · r · r sin(2π/n)", 30)).arrange(RIGHT, buff=0.22)
        row1.move_to(np.array([1.0 + row1.width / 2, 2.6, 0.0]))
        self.play(FadeIn(row1), run_time=0.8)
        self.hold(0.5)

        # ---- turned about O by 2π/n, again and again: it tiles the polygon
        copies = [mk(T0, cT[k % 2], 0.8, stroke_width=2) for k in range(1, nn)]
        for c in copies:
            self.add(c)
        self.bring_to_back(*copies)
        self.bring_to_back(circ)
        self.play(LaggedStart(*[Rotate(c, angle=(k + 1) * th, about_point=O)
                                for k, c in enumerate(copies)], lag_ratio=0.18), run_time=3.2)
        for k, c in enumerate(copies):
            _landed(c, tris[k + 1], f"copy {k + 1} lands on its triangle")
        # (an animation group re-adds its pieces on top: lines and labels back up)
        self.bring_to_front(poly, dO, hl, raH, tk, aO, lr0, lr1, lang)
        nl = tag("n = 7", 30)
        nl.move_to(np.array([1.0 + nl.width / 2, 0.45, 0.0]))
        ic2 = mk([sum(V) / nn + 0.18 * (p - sum(V) / nn) for p in V], cT[1], 0.85, stroke_width=1.5)
        row2 = VGroup(ic2, tag("=", 32), tag("n · ½ r² sin(2π/n)", 30)).arrange(RIGHT, buff=0.22)
        row2.move_to(np.array([0.0, -0.75, 0.0])).align_to(row1, LEFT)
        self.play(FadeIn(nl), run_time=0.5)
        self.play(FadeIn(row2), run_time=0.8)

        labels = [lr0, lr1, lang, lh, row1, row2, nl]
        _labels_ok([lr0, lr1, lang], segs, "G24 triangle")
        _labels_ok(labels, [_seg(gd.get_start(), gd.get_end()), _seg(gd0.get_start(), gd0.get_end())]
                   + _circle_segs(O, R, 140) + [_seg(bar[i].get_start(), bar[i].get_end())
                                                 for i in range(3)], "G24 all")
        _all_in_frame([circ, row1, row2], "G24")
        cap = caption("A  =  (n/2) · r² · sin(2π/n)", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ G20 JORDAN'S INEQUALITY

class G20_Jordan(Board):
    """sin x ≥ 2x/π for 0 ≤ x ≤ π/2. On the unit circle the arc PQ of angle
    2x has length 2x; its chord PQ is 2 sin x. The semicircle on PQ (centre
    N, radius sin x), bulging the same way, has length π sin x, and the arc
    PQ lies inside that half-disc: a point at angle t (|t| ≤ x) is at
    distance² sin²x − 2 cos x (cos t − cos x) ≤ sin²x from N. A convex arc
    inside another with the same ends is the shorter: 2x ≤ π sin x, with
    equality at x = π/2, where the two arcs coincide. On the graph: the sine
    arch lies above the chord y = 2x/π."""

    def construct(self):
        kc = 2.6
        O = np.array([-5.05, 0.45, 0.0])
        G0 = np.array([1.0, -1.6, 0.0])
        kg = 3.15
        x0 = 50 * DEGREES
        xv = ValueTracker(x0)
        cY, cT = YELLOW_B, TEAL_B
        n = np.linalg.norm

        def C(x, y):
            return O + kc * np.array([x, y, 0.0])

        def Gp(x, y):
            return G0 + kg * np.array([x, y, 0.0])

        # ---- the claims, along every x the sweep visits
        sweep = np.concatenate([np.linspace(x0, PI / 2, 15), np.linspace(PI / 2, 20 * DEGREES, 25),
                                np.linspace(20 * DEGREES, x0, 10)])
        for x in sweep:
            Nn, s_, c_ = np.array([np.cos(x), 0.0]), np.sin(x), np.cos(x)
            for t in np.linspace(-x, x, 41):
                q = np.array([np.cos(t), np.sin(t)])
                check(n(q - Nn) <= s_ + 1e-12 and q[0] >= c_ - 1e-12,
                      "the arc PQ lies in the half-disc on PQ")
            check(2 * x <= PI * s_ + 1e-12, "2x ≤ π sin x")
            check(abs(2 * x - PI * s_) > 1e-6 or abs(x - PI / 2) < 1e-9, "equality only at π/2")
            check(np.sin(x) >= 2 * x / PI - 1e-12, "sin x ≥ 2x/π")

        # ---- the left panel: unit arc PQ inside the semicircle on PQ
        half = Arc(radius=kc, start_angle=-PI / 2, angle=PI, arc_center=O, color=GREY_B,
                   stroke_width=2.5)
        dO = Dot(O, radius=0.06)
        lO = tag("O", 28).move_to(O + 0.27 * DOWN + 0.32 * LEFT)
        for x in sweep:                   # every chord PQ of the sweep misses it
            cx = O[0] + kc * np.cos(x)
            check(lO.get_right()[0] + 0.06 < cx, "the chord clears the label O")

        def left(x):
            P, Q, N = C(np.cos(x), np.sin(x)), C(np.cos(x), -np.sin(x)), C(np.cos(x), 0.0)
            rs = kc * np.sin(x)
            disc = Sector(radius=rs, start_angle=-PI / 2, angle=PI, arc_center=N,
                          fill_color=cT, fill_opacity=0.16, stroke_width=0)
            semi = Arc(radius=rs, start_angle=-PI / 2, angle=PI, arc_center=N, color=cT,
                       stroke_width=5)
            arc = Arc(radius=kc, start_angle=-x, angle=2 * x, arc_center=O, color=cY,
                      stroke_width=6)
            g = VGroup(disc, Line(O, P, color=GREY_A, stroke_width=2.5),
                       Line(O, Q, color=GREY_A, stroke_width=2.5),
                       DashedLine(O, N, color=GREY_A, stroke_width=2, dash_length=0.08),
                       Line(P, Q, color=WHITE, stroke_width=3), semi, arc,
                       angle_arc(O, N if x < PI / 2 - 1e-6 else C(1, 0), P, 0.55, WHITE, 3),
                       _ra(N, O, P, 0.17) if x < PI / 2 - 1e-6 else VGroup(),
                       Dot(P, radius=0.06), Dot(Q, radius=0.06))
            return g

        xb = O[0] - 0.42                 # the bar that measures NP = sin x

        def sin_bar(x):
            P = C(np.cos(x), np.sin(x))
            bar = VGroup(Line(np.array([xb, O[1], 0]), np.array([xb, P[1], 0]), color=WHITE,
                              stroke_width=3),
                         Line(np.array([xb - 0.09, O[1], 0]), np.array([xb + 0.09, O[1], 0]),
                              color=WHITE, stroke_width=3),
                         Line(np.array([xb - 0.09, P[1], 0]), np.array([xb + 0.09, P[1], 0]),
                              color=WHITE, stroke_width=3))
            gds = VGroup(DashedLine(P, np.array([xb - 0.09, P[1], 0]), color=GREY_B,
                                    stroke_width=1.5, dash_length=0.07),
                         DashedLine(O, np.array([xb - 0.09, O[1], 0]), color=GREY_B,
                                    stroke_width=1.5, dash_length=0.07))
            return bar, gds

        def left_labels(x):
            P, Q, N = C(np.cos(x), np.sin(x)), C(np.cos(x), -np.sin(x)), C(np.cos(x), 0.0)
            l1 = _beside(tag("1", 28), O, P, _perp(P - O), 0.14, at=0.6)
            lx = tag("x", 28).move_to(O + 0.85 * angle_mid_dir(O, N, P))
            bar, gds = sin_bar(x)
            ls = tag("sin x", 28).next_to(bar[0], LEFT, buff=0.14)
            l2x = tag("2x", 30, cY).move_to((N + C(1, 0)) / 2 + 0.33 * UP)
            lps = tag("π sin x", 30, cT).next_to(C(np.cos(x) + np.sin(x), 0), RIGHT, buff=0.18)
            lP = tag("P", 28).move_to(P + 0.44 * _unit(P - O))
            lQ = tag("Q", 28).move_to(Q + 0.44 * _unit(Q - O))
            lN = tag("N", 26).move_to(N + 0.3 * DOWN + 0.28 * LEFT)
            return VGroup(l1, lx, ls, l2x, lps, lP, lQ, lN, bar, gds)

        self.play(Create(half), FadeIn(dO), FadeIn(lO), run_time=1.0)
        L0 = left(x0)
        LL = left_labels(x0)
        self.play(Create(L0[1]), Create(L0[2]), Create(L0[3]), Create(L0[7]), FadeIn(L0[9]),
                  FadeIn(L0[10]), FadeIn(LL[0]), FadeIn(LL[1]), FadeIn(LL[5]), FadeIn(LL[6]),
                  run_time=1.0)
        # the arc of angle 2x: length 2x
        self.play(Create(L0[6]), FadeIn(LL[3]), run_time=0.9)
        # its chord: 2 sin x
        self.play(Create(L0[4]), Create(L0[8]), FadeIn(LL[7]), run_time=0.7)
        self.play(Create(LL[9]), Create(LL[8]), FadeIn(LL[2]), run_time=0.8)
        # the semicircle on the chord: π sin x, and the arc lies inside it
        self.add(L0[0])
        self.bring_to_back(L0[0])
        self.play(Create(L0[5]), FadeIn(L0[0]), FadeIn(LL[4]), run_time=1.1)
        ineq = Text("2x  ≤  π sin x", font_size=32, t2c={"2x": cY, "π sin x": cT})
        ineq.move_to(np.array([O[0] + 1.75, -2.55, 0.0]))
        self.play(FadeIn(ineq), run_time=0.7)
        self.hold(0.4)

        # ---- the right panel: the sine arch above its chord y = 2x/π
        axx = Line(Gp(0, 0), Gp(PI / 2 + 0.13, 0), color=GREY_B, stroke_width=2.5)
        axy = Line(Gp(0, 0), Gp(0, 1.12), color=GREY_B, stroke_width=2.5)
        tx = Line(Gp(PI / 2, 0) + 0.08 * DOWN, Gp(PI / 2, 0) + 0.08 * UP, color=GREY_B,
                  stroke_width=2.5)
        ty = Line(Gp(0, 1) + 0.08 * LEFT, Gp(0, 1) + 0.08 * RIGHT, color=GREY_B, stroke_width=2.5)
        lpi2 = tag("π/2", 26).next_to(Gp(PI / 2, 0), DOWN, buff=0.16)
        lone = tag("1", 26).next_to(Gp(0, 1), LEFT, buff=0.16)
        lzero = tag("0", 26).next_to(Gp(0, 0), DOWN + LEFT, buff=0.08)
        xs = np.linspace(0, PI / 2, 120)
        curve = _curve([Gp(t, np.sin(t)) for t in xs], cT, 4)
        chord = Line(Gp(0, 0), Gp(PI / 2, 1), color=cY, stroke_width=4)
        lsin = tag("sin x", 28, cT).move_to(Gp(0.62, 0.84))
        lch = tag("2x/π", 28, cY).move_to(Gp(1.08, 0.5))
        self.play(Create(axx), Create(axy), Create(tx), Create(ty), FadeIn(lpi2), FadeIn(lone),
                  FadeIn(lzero), run_time=0.8)
        self.play(Create(curve), Create(chord), FadeIn(lsin), FadeIn(lch), run_time=1.2)

        def marker(x):
            a, b = Gp(x, 2 * x / PI), Gp(x, np.sin(x))
            return VGroup(DashedLine(Gp(x, 0), a, color=GREY_A, stroke_width=2, dash_length=0.07),
                          Line(a, b, color=WHITE, stroke_width=5),
                          Dot(a, radius=0.07, color=cY), Dot(b, radius=0.07, color=cT))
        mk0 = marker(x0)
        lxm = tag("x", 26).next_to(Gp(x0, 0), DOWN, buff=0.16)
        self.play(FadeIn(mk0), FadeIn(lxm), run_time=0.6)

        # ---- every x: the arcs and the graph move together
        self.play(FadeOut(LL), FadeOut(lxm), run_time=0.4)
        for m in list(self.mobjects):
            if m in L0 or m is L0 or m is mk0:
                self.remove(m)
        self.remove(*L0, mk0)
        live = always_redraw(lambda: VGroup(left(xv.get_value()), marker(xv.get_value())))
        self.add(live)
        self.bring_to_front(dO)
        self.play(xv.animate.set_value(PI / 2), run_time=1.6)
        self.hold(0.5)
        self.play(xv.animate.set_value(20 * DEGREES), run_time=2.2)
        self.play(xv.animate.set_value(x0), run_time=1.2)
        live.clear_updaters()
        LL2 = left_labels(x0)
        lxm2 = tag("x", 26).next_to(Gp(x0, 0), DOWN, buff=0.16)
        self.play(FadeIn(LL2), FadeIn(lxm2), run_time=0.6)

        P, Q, N = C(np.cos(x0), np.sin(x0)), C(np.cos(x0), -np.sin(x0)), C(np.cos(x0), 0.0)
        rs = kc * np.sin(x0)
        bar, gds = sin_bar(x0)
        segs = (_arc_segs(O, kc, -PI / 2, PI / 2, 60) + _arc_segs(N, rs, -PI / 2, PI / 2, 50)
                + _arc_segs(O, kc, -x0, x0, 30)
                + _segs(O, P) + _segs(O, Q) + _segs(O, N) + _segs(P, Q)
                + _angle_segs(O, N, P, 0.55) + _ra_segs(N, O, P, 0.17)
                + [_seg(b.get_start(), b.get_end()) for b in bar]
                + [_seg(g.get_start(), g.get_end()) for g in gds]
                + sum((_dot_segs(X, 0.07) for X in (O, P, Q)), []))
        texts = [m for m in LL2 if isinstance(m, Text)]
        _labels_ok(texts + [lO, ineq], segs, "G20 circle")
        gsegs = (_segs(*[Gp(t, np.sin(t)) for t in xs[::3]]) + _segs(Gp(0, 0), Gp(PI / 2, 1))
                 + _segs(Gp(0, 0), Gp(PI / 2 + 0.13, 0)) + _segs(Gp(0, 0), Gp(0, 1.12))
                 + _segs(Gp(x0, 0), Gp(x0, np.sin(x0)))
                 + _dot_segs(Gp(x0, 2 * x0 / PI), 0.08) + _dot_segs(Gp(x0, np.sin(x0)), 0.08)
                 + [_seg(tx.get_start(), tx.get_end()), _seg(ty.get_start(), ty.get_end())])
        _labels_ok([lpi2, lone, lzero, lsin, lch, lxm2], gsegs, "G20 graph")
        _labels_ok(texts + [lO, ineq, lpi2, lone, lzero, lsin, lch, lxm2], [], "G20 apart")
        cap = caption("2x ≤ π sin x   ⟹   sin x ≥ 2x/π   (0 ≤ x ≤ π/2)", 32)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ G28 QUADRILATERAL FROM ITS DIAGONALS

class G28_QuadDiagonals(Board):
    """The diagonals AC = d₁ and BD = d₂ of a convex quadrilateral cross at
    P at the angle θ. Lines through A and C parallel to BD, and through B
    and D parallel to AC, make a parallelogram with sides d₁, d₂ and angle
    θ. The diagonals cut it into four cells, each a parallelogram with P as
    a corner; each side of the quadrilateral is a diagonal of its cell and
    halves it — a half-turn about the side's midpoint carries the inner
    triangle onto the outer one. So the quadrilateral is half the
    parallelogram, whose area is base d₁ times height d₂ sin θ:
    A = ½ d₁ d₂ sin θ. (Drawn for a convex quadrilateral.)"""

    def construct(self):
        th = 64 * DEGREES
        pa, pc, pb, pd = 2.6, 2.2, 1.5, 2.0
        um, vm = np.array([1.0, 0.0]), np.array([np.cos(th), np.sin(th)])
        Am, Cm, Bm, Dm = -pa * um, pc * um, -pb * vm, pd * vm
        corners_m = [Am + Bm, Bm + Cm, Cm + Dm, Dm + Am]
        Fm = np.array([(Cm + Dm)[0], (Am + Bm)[1]])
        # the half-turns sweep the circles on the quadrilateral's sides: the
        # frame must hold them as well as the parallelogram and its labels
        sweep_y = [m_[1] + sg * np.linalg.norm(X - Y) / 2
                   for X, Y in ((Am, Bm), (Bm, Cm), (Cm, Dm), (Dm, Am))
                   for m_ in [(X + Y) / 2] for sg in (-1, 1)]
        F_ = Frame(-3.75, Fm[0] + 1.75, min(min(sweep_y), (Am + Bm)[1] - 0.8) - 0.05,
                   max(max(sweep_y), (Cm + Dm)[1] + 0.55) + 0.05)
        A, B, C, D = F_.P(Am), F_.P(Bm), F_.P(Cm), F_.P(Dm)
        P = F_.P((0.0, 0.0))
        E1, E2, E3, E4 = (F_.P(c) for c in corners_m)        # A+B, B+C, C+D, D+A
        Fh = F_.P(Fm)
        n = np.linalg.norm
        d1, d2 = n(C - A), n(D - B)
        quad = [A, B, C, D]
        para = [E1, E2, E3, E4]
        check(_pip(P, quad) and _between(P, A, C) and _between(P, B, D),
              "a convex quadrilateral: the diagonals cross inside")
        check(abs(_angle(P, C, D) - th) < 1e-9, "the diagonals meet at θ")
        for (X, Y), E in zip(((A, B), (B, C), (C, D), (D, A)), para):
            check(_on_line(E, X, X + (D - B)) or _on_line(E, X, X + (C - A)), "corner on the lines")
            check(close(X + Y - P, E), "the cell P, X, E, Y is a parallelogram")
        check(abs(n(E2 - E1) - d1) < 1e-9 and abs(n(E3 - E2) - d2) < 1e-9,
              "the parallelogram's sides are d₁ and d₂")
        check(abs(_angle(E2, Fh, E3) - th) < 1e-9, "its angle is θ (at the corner, outside)")
        check(abs(n(E3 - Fh) - d2 * np.sin(th)) < 1e-9 and abs(np.dot(E3 - Fh, E2 - E1)) < 1e-9,
              "its height is d₂ sin θ")
        tris = [[P, A, B], [P, B, C], [P, C, D], [P, D, A]]
        mids = [(A + B) / 2, (B + C) / 2, (C + D) / 2, (D + A) / 2]
        outer = [[E1, B, A], [E2, C, B], [E3, D, C], [E4, A, D]]
        for T, m_, O_ in zip(tris, mids, outer):
            check(_same_poly([2 * m_ - p for p in T], O_), "the half-turn lands on the outer half")
        check(_tiles_exactly(tris, quad), "the four triangles tile the quadrilateral")
        check(_tiles_exactly(tris + outer, para), "with their copies they tile the parallelogram")
        check(abs(abs(area(quad)) - 0.5 * d1 * d2 * np.sin(th)) < 1e-9, "A = ½ d₁ d₂ sin θ")

        cols = [BLUE_D, TEAL_D, GREEN_D, PURPLE_D]
        c1, c2 = BLUE_B, ORANGE
        qd = Polygon(*quad, stroke_color=WHITE, stroke_width=3)
        ac = Line(A, C, color=c1, stroke_width=5)
        bd = Line(B, D, color=c2, stroke_width=5)
        aP = angle_arc(P, C, D, 0.5, YELLOW_B, 4)

        segs = (_segs(*quad, closed=True) + _segs(*para, closed=True) + _segs(A, C) + _segs(B, D)
                + _angle_segs(P, C, D, 0.5) + _segs(E2, Fh) + _segs(E3, Fh)
                + _ra_segs(Fh, E3, E1, 0.22) + _angle_segs(E2, Fh, E3, 0.55))
        lA = _park_dirs(tag("A", 30), A, [180, 165, 195], segs, d0=0.3)
        lB = _park_dirs(tag("B", 30), B, [270, 255, 285], segs, d0=0.3)
        lC = _park_dirs(tag("C", 30), C, [0, 345, 15, 330], segs, d0=0.3)
        lD = _park_dirs(tag("D", 30), D, [90, 75, 105], segs, d0=0.3)
        lth = _angle_label(tag("θ", 30, YELLOW_B), P, C, D, 0.62, segs=segs)
        ld1 = _park_beside(tag("d₁", 30, c1), [(A, P)], segs, prefer=UP, only=True,
                           avoid=[lA, lB, lC, lD, lth], ats=(0.5, 0.4, 0.6))
        ld2 = _park_beside(tag("d₂", 30, c2), [(P, D)], segs, prefer=LEFT, only=True,
                           avoid=[lA, lB, lC, lD, lth, ld1], ats=(0.6, 0.5, 0.7))
        self.play(Create(qd), FadeIn(lA), FadeIn(lB), FadeIn(lC), FadeIn(lD), run_time=1.0)
        self.play(Create(ac), Create(bd), run_time=0.9)
        self.play(Create(aP), FadeIn(lth), FadeIn(ld1), FadeIn(ld2), run_time=0.6)

        # ---- the four triangles the diagonals cut
        tm = [mk(T, c, 0.75, stroke_width=0) for T, c in zip(tris, cols)]
        self.add(*tm)
        self.bring_to_back(*tm)
        self.play(*[FadeIn(t) for t in tm], run_time=0.8)

        # ---- lines through the vertices parallel to the diagonals
        def through(X, d):
            return DashedLine(X - 0.62 * d, X + 0.62 * d, color=GREY_B, stroke_width=2,
                              dash_length=0.08)
        u, v = (C - A) / d1, (D - B) / d2
        par = VGroup(through(A, v * d2), through(C, v * d2), through(B, u * d1), through(D, u * d1))
        self.play(Create(par), run_time=1.0)
        pg = Polygon(*para, stroke_color=WHITE, stroke_width=3)
        self.play(Create(pg), FadeOut(par), run_time=0.8)

        # ---- each triangle, half a turn about the middle of its side, fills its cell
        cps = [mk(T, c, 0.45, stroke_width=0) for T, c in zip(tris, cols)]
        dots = VGroup(*[Dot(m_, radius=0.05, color=WHITE) for m_ in mids])
        self.play(FadeIn(dots), run_time=0.3)
        for T, m_ in zip(tris, mids):
            for kk in np.linspace(0.0, 1.0, 61):
                for X in T:
                    Y = _rot2(X, m_, kk * PI)
                    check(abs(Y[0]) <= SAFE_X and SAFE_BOTTOM <= Y[1] <= SAFE_TOP,
                          "the turning copies stay in the safe area")
        for c_ in cps:
            self.add(c_)
        self.play(*[Rotate(c_, angle=PI, about_point=m_) for c_, m_ in zip(cps, mids)],
                  run_time=2.0)
        for c_, O_ in zip(cps, outer):
            check(_same_poly(c_.get_vertices(), O_, 1e-6), "each copy lands in its cell's outer half")
        self.play(FadeOut(dots), run_time=0.3)
        self.bring_to_front(qd, ac, bd, aP, pg)

        # ---- the parallelogram's sides are the diagonals, its height d₂ sin θ
        ac2, bd2 = ac.copy(), bd.copy()
        self.add(ac2, bd2)
        self.play(ac2.animate.shift(B - P), bd2.animate.shift(C - P), run_time=1.3)
        check(close(ac2.get_start(), E1, 1e-6) and close(ac2.get_end(), E2, 1e-6)
              and close(bd2.get_start(), E2, 1e-6) and close(bd2.get_end(), E3, 1e-6),
              "the slid diagonals are the sides E₁E₂ and E₂E₃")
        ld1b = tag("d₁", 30, c1).next_to(Line(E1, E2), DOWN, buff=0.16)
        ld1b.set_x(E1[0] + 0.33 * (E2[0] - E1[0]))
        ld2b = _park_beside(tag("d₂", 30, c2), [(E2, E3)], segs, prefer=Fh - E2, only=True,
                            avoid=[lA, lB, lC, lD], ats=(0.62, 0.7, 0.55, 0.78))
        self.play(FadeIn(ld1b), FadeIn(ld2b), run_time=0.6)
        ext = DashedLine(E2, Fh, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        hgt = DashedLine(E3, Fh, color=WHITE, stroke_width=3, dash_length=0.09)
        raF = _ra(Fh, E3, E1, 0.22)
        aE = angle_arc(E2, Fh, E3, 0.55, YELLOW_B, 4)
        lthE = _angle_label(tag("θ", 30, YELLOW_B), E2, Fh, E3, 0.68, segs=segs,
                            avoid=[lC, ld2b])
        lh = tag("d₂ sin θ", 28).next_to(hgt, RIGHT, buff=0.18)
        self.play(Create(ext), Create(aE), FadeIn(lthE), run_time=0.8)
        self.play(Create(hgt), Create(raF), FadeIn(lh), run_time=0.8)

        labels = [lA, lB, lC, lD, lth, ld1, ld2, ld1b, ld2b, lthE, lh]
        _labels_ok(labels, segs + _segs(E1, E2) + _segs(E2, E3), "G28 final")
        cap = caption("A  =  ½ · d₁ · d₂ sin θ", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)
