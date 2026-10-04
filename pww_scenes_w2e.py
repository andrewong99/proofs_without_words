# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes_w2e.py — proofs without words: E12 the tangent–chord angle,
E18 the reflection property of the ellipse, E29 the radius square to a
chord, E25 two secants from a point, E21 the arbelos, E13 the angle between
two chords, E30 the tangent square to the radius, E39 the length of a chord,
G12 the half-angle formulas.

Built on pww_kit (Frame, mk, tag, caption, guide, angle_arc, check, ...).
Every scene states its geometric invariants with check(...) — lengths,
angles, exact landings of moved pieces, and that labels stay clear of the
lines and of each other — so a wrong construction fails the render instead
of drawing a wrong picture. Angles are drawn with angle_arc (never the
reflex angle); equal angles carry matching arcs.
"""

from pww_kit import *


# ------------------------------------------------------------ private helpers

def _dir(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


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


def _wedge_pts(vertex, p, q, radius):
    """Apex and the two rim ends of _wedge(vertex, p, q, radius), in the
    order start ray, end ray (for _rigid)."""
    a1, span = _wedge_span(vertex, p, q)
    v = to3(vertex)
    return [v, v + radius * _dir(a1), v + radius * _dir(a1 + span)]


def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centroid while the
    centroid slides — carrying the screen points src onto dst (same order).
    Fails the render unless one rotation + translation lands every point.
    `turn` (radians) picks the direction when the angle is ±180°."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = np.arctan2(*(src[1] - src[0])[1::-1])
        a1 = np.arctan2(*(dst[1] - dst[0])[1::-1])
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    cs, sn = np.cos(turn), np.sin(turn)
    Rm = np.array([[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]])
    check(all(close(m1 + Rm @ (p - m0), q, 1e-6) for p, q in zip(src, dst)),
          "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


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


def _chevron(p, q, color=YELLOW_B, size=0.17, width=3, n=1):
    """Parallel-line mark: n small '>' at the middle of segment pq,
    pointing from p towards q."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = (p + q) / 2
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


def _same_vertices_ok(poly, pts, tol=1e-6):
    """True if a manim Polygon has exactly the given corner points (any order)."""
    vs = [to3(v) for v in poly.get_vertices()]
    ps = [to3(q) for q in pts]
    return (len(vs) == len(ps)
            and all(any(np.linalg.norm(v - q) < tol for q in ps) for v in vs)
            and all(any(np.linalg.norm(v - q) < tol for v in vs) for q in ps))


def _rot90(p, c):
    """The screen point p turned a quarter turn counter-clockwise about c."""
    d = to3(p) - to3(c)
    return to3(c) + np.array([-d[1], d[0], 0.0])


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


# ---- label hygiene: every label is checked against the lines it must clear

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _arc_segs(c, r, a0, a1, n=48):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    pts = [c + r * _dir(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _poly_segs(pts):
    """The closed outline of a polygon, as segments."""
    pts = [to3(p) for p in pts]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


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
        check(_in_frame(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{_name(m)}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _park(m, X, segs, base, dist=0.4, avoid=(), pad=0.05):
    """Put label m beside point X: try directions turning away from the
    unit vector `base` in 20° steps until its box is clear of the segments
    and of the labels in `avoid`; fall back to `base`."""
    a0 = float(np.arctan2(base[1], base[0]))
    for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9):
        m.move_to(to3(X) + dist * _dir(a0 + k * 20 * DEGREES))
        if (_hit(m, segs, pad) is None and _in_frame(m)
                and not any(_overlap(m, o, 0.03) for o in avoid)):
            return m
    return m.move_to(to3(X) + dist * _dir(a0))


def _final_ok(scene, cap):
    """Closing frame: the caption in its band, everything else inside the
    safe area, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_top()[1] < SAFE_BOTTOM,
          "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if m.has_points() or m.submobjects:
            check(_in_frame(m, tol=0.03), f"{_name(m)} inside the safe area")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(not _overlap(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# ============================================ E12  — the tangent–chord angle

class E12_TangentChord(Board):
    """The angle θ between the tangent t at A and the chord AB. The radius
    OA is square to t, and OM — the axis of the isosceles triangle AOB,
    through the midpoint M of the chord — is square to AB. Both sides of θ
    turned through a right angle are the sides of ∠AOM, so θ, turned a
    quarter turn and carried to O, is ∠AOM. Folding over the axis OM gives
    ∠MOB = ∠AOM: θ is half the central angle AOB. The inscribed angle ACB
    on the other side of the chord is half the same central angle (two
    copies of it fill ∠AOB), so it is θ too."""

    def construct(self):
        th = 50 * DEGREES
        R = 2.65
        O = np.array([-0.6, 0.75, 0.0])
        A = O + R * _dir(-PI / 2)
        B = O + R * _dir(-PI / 2 + 2 * th)
        M = (A + B) / 2
        C = O + R * _dir(155 * DEGREES)
        tL, tR = A + 5.0 * LEFT, A + 5.0 * RIGHT
        rw = 0.85
        check(abs(np.dot(A - O, tR - A)) < 1e-9, "OA ⊥ t")
        check(abs(_ang(A, tR, B) - th) < 1e-9, "the tangent–chord angle is θ")
        check(abs(np.dot(M - O, B - A)) < 1e-9, "OM ⊥ AB at the midpoint M")
        check(abs(_ang(O, A, M) - th) < 1e-9 and abs(_ang(O, M, B) - th) < 1e-9,
              "∠AOM = ∠MOB = θ")
        check(close(_refl(A, O, M - O), B), "the fold over OM carries A onto B")
        check(abs(_ang(C, A, B) - th) < 1e-9, "the inscribed angle ACB is θ")
        # θ turned −90°: the tangent ray → OA, the chord → OM
        check(close(_dir(0.0 - PI / 2), _unit(A - O))
              and close(_dir(th - PI / 2), _unit(M - O)),
              "both sides of θ, turned a quarter turn, are OA and OM")
        sA = [A, A + rw * _dir(0.0), A + rw * _dir(th)]
        dOM = [O, O + rw * _dir(-PI / 2), O + rw * _dir(-PI / 2 + th)]
        dMB = [O, O + rw * _dir(-PI / 2 + th), O + rw * _dir(-PI / 2 + 2 * th)]
        check(all(close(_refl(p, O, M - O), q) for p, q in
                  zip(dOM, [O, dMB[2], dMB[1]])), "the fold carries ∠AOM onto ∠MOB")

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        dO = _dot(O)
        lO = tag("O", 26, GREY_A).move_to(O + 0.42 * _dir(128 * DEGREES))
        tline = Line(tL, tR, color=WHITE, stroke_width=3)
        lt = tag("t", 30).move_to(tR + np.array([-0.2, 0.34, 0.0]))
        dA = _dot(A)
        lA = tag("A", 28).move_to(A + 0.42 * DOWN)
        rOA = Line(O, A, color=GREY_A, stroke_width=2.5)
        raA = _ra(A, O, tL, 0.22, GREY_A, 2)
        self.play(Create(circ), FadeIn(dO), FadeIn(lO), run_time=1.0)
        self.play(Create(tline), FadeIn(dA), FadeIn(lA), FadeIn(lt), run_time=1.0)
        self.play(Create(rOA), Create(raA), run_time=0.7)

        chord = Line(A, B, color=WHITE, stroke_width=3)
        dB = _dot(B)
        lB = tag("B", 28).move_to(B + 0.4 * _unit(B - O))
        self.play(Create(chord), FadeIn(dB), FadeIn(lB), run_time=0.8)
        wA = _wedge(A, tR, B, rw, BLUE_D, 0.75)
        aA = angle_arc(A, tR, B, rw, BLUE_B, 4)
        lthA = tag("θ", 30, BLUE_B).move_to(A + 1.2 * _dir(0.64 * th))   # clear of the arc
        self.play(FadeIn(wA), Create(aA), FadeIn(lthA), run_time=0.8)
        self.bring_to_front(dA)
        self.hold(0.5)

        # the axis of the isosceles triangle AOB: square to the chord
        rOB = Line(O, B, color=GREY_A, stroke_width=2.5)
        lOM = DashedLine(O, M, color=GREY_A, stroke_width=2.5, dash_length=0.09)
        raM = _ra(M, O, B, 0.2, GREY_A, 2)
        self.play(Create(rOB), Create(lOM), Create(raM), run_time=0.9)

        # θ, turned through a right angle, is ∠AOM
        w1 = wA.copy()
        self.add(w1)
        self.play(_rigid(w1, sA, dOM, turn=-PI / 2), run_time=1.8)
        lth1 = tag("θ", 26, BLUE_B).move_to(O + (rw + 0.3) * _dir(-PI / 2 + th / 2))
        aO1 = angle_arc(O, A, M, rw, BLUE_B, 4)
        self.bring_to_front(dO)
        self.play(FadeIn(lth1), Create(aO1), run_time=0.4)

        # folded over the axis OM it is ∠MOB: θ is half of ∠AOB
        w2 = w1.copy()
        self.add(w2)
        self.play(_fold(w2, O, M), run_time=1.4)
        lth2 = tag("θ", 26, BLUE_B).move_to(O + (rw + 0.3) * _dir(-PI / 2 + 1.5 * th))
        aO2 = angle_arc(O, M, B, rw, BLUE_B, 4)
        self.bring_to_front(dO)
        self.play(FadeIn(lth2), Create(aO2), run_time=0.4)
        self.hold(0.4)

        # the inscribed angle on the other side of the chord
        dC = _dot(C)
        lC = tag("C", 28).move_to(C + 0.4 * _unit(C - O))
        cCA = Line(C, A, color=WHITE, stroke_width=2.5)
        cCB = Line(C, B, color=WHITE, stroke_width=2.5)
        self.play(FadeIn(dC), FadeIn(lC), Create(cCA), Create(cCB), run_time=1.0)
        wC = _wedge(C, A, B, rw, BLUE_D, 0.75)
        aC = angle_arc(C, A, B, rw, BLUE_B, 4)
        self.play(FadeIn(wC), Create(aC), run_time=0.6)
        self.bring_to_front(dC)

        # two copies of it fill the central angle AOB
        self.play(w1.animate.set_fill(opacity=0.15), w2.animate.set_fill(opacity=0.15),
                  run_time=0.4)
        c1, c2 = wC.copy(), wC.copy()
        self.add(c1, c2)
        self.bring_to_front(lO, lB, lC)                # labels stay on top of the moving copies
        sC = _wedge_pts(C, A, B, rw)
        self.play(LaggedStart(_rigid(c1, sC, dOM), _rigid(c2, sC, dMB),
                              lag_ratio=0.4), run_time=2.2)
        self.bring_to_front(aO1, aO2, dO)
        lthC = tag("θ", 30, BLUE_B).move_to(C + (rw + 0.38) * angle_mid_dir(C, A, B))
        self.play(FadeIn(lthC), run_time=0.5)
        self.remove(w1, w2)

        segs = (_arc_segs(O, R, 0.0, TAU, 96)
                + [_seg(tL, tR), _seg(A, B), _seg(O, A), _seg(O, B), _seg(O, M),
                   _seg(C, A), _seg(C, B)]
                + _ra_segs(A, O, tL, 0.22) + _ra_segs(M, O, B, 0.2)
                + _arc_segs(A, rw, 0.0, th, 12) + _arc_segs(O, rw, -PI / 2, -PI / 2 + 2 * th, 16)
                + _arc_segs(C, rw, *(lambda a, s: (a, a + s))(*_wedge_span(C, A, B)), 12))
        _labels_ok([lO, lt, lA, lB, lC, lthA, lth1, lth2, lthC], segs, "E12")
        cap = caption("θ  =  ½ ∠AOB  =  ∠ACB", 36)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E18  — the ellipse's reflection property

class E18_EllipseReflection(Board):
    """The ellipse is where PF₁ + PF₂ = 2a; outside it the sum is larger.
    The tangent t at P meets the ellipse only at P, so every other point Q
    of t is outside: QF₁ + QF₂ > 2a, and along t the sum is least at P.
    Reflect F₂ in t to F₂′: QF₂′ = QF₂ for every Q on t, so the path
    F₁ → Q → F₂′ is shortest at Q = P. The shortest path from F₁ to F₂′ is
    the straight segment, so F₁, P, F₂′ lie on one line. The angle between
    PF₁ and t then equals its vertical angle between t and PF₂′, which the
    mirror makes equal to the angle between t and PF₂: the two focal radii
    make equal angles with the tangent, and a ray from F₁ reflects to F₂."""

    def construct(self):
        a, b = 3.9, 2.25
        c = float(np.sqrt(a * a - b * b))
        Z = np.array([-0.4, -0.6, 0.0])
        F1, F2 = Z + c * LEFT, Z + c * RIGHT
        n = np.linalg.norm

        def E(t):
            return Z + np.array([a * np.cos(t), b * np.sin(t), 0.0])

        def tdir(t):                       # unit tangent, clockwise sense
            return _unit(np.array([a * np.sin(t), -b * np.cos(t), 0.0]))

        def outside(X):
            return ((X - Z)[0] / a) ** 2 + ((X - Z)[1] / b) ** 2 > 1 + 1e-12

        tP = 62 * DEGREES
        P, u = E(tP), tdir(tP)
        nout = np.array([-u[1], u[0], 0.0])           # outward normal at P
        if np.dot(nout, P - Z) < 0:
            nout = -nout
        F2p = _refl(F2, P, u)
        H = _foot(F2, P, P + u)
        check(abs(n(P - F1) + n(P - F2) - 2 * a) < 1e-9, "PF₁ + PF₂ = 2a")
        check(abs(np.dot(u, np.array([(P - Z)[0] / a ** 2, (P - Z)[1] / b ** 2, 0.0])))
              < 1e-12, "t is the tangent at P")
        for s in np.linspace(-3.0, 3.0, 121):
            if abs(s) > 1e-6:
                Q = P + s * u
                check(outside(Q) and n(Q - F1) + n(Q - F2) > 2 * a + 1e-9,
                      "every other point of t is outside: QF₁ + QF₂ > 2a")
                check(abs(n(Q - F2p) - n(Q - F2)) < 1e-9, "QF₂′ = QF₂")
        check(abs(_cross(P - F1, F2p - F1)) < 1e-9 and np.dot(P - F1, F2p - P) > 0,
              "F₁, P, F₂′ in a line, P between")
        check(abs(n(F2p - F1) - 2 * a) < 1e-9, "F₁F₂′ = 2a")
        al = _ang(P, F1, P - u)
        check(abs(_ang(P, F2p, P + u) - al) < 1e-9 and abs(_ang(P, F2, P + u) - al) < 1e-9,
              "vertical angles, then the mirror: the three angles at P are equal")
        ray_t = [105, 140, 172, 262, 290, 318]
        for tk in ray_t:
            X = E(tk * DEGREES)
            uX = tdir(tk * DEGREES)
            check(abs(_ang(X, F1, X + uX) + _ang(X, F2, X + uX) - PI) < 1e-9,
                  "at every point of the ellipse the focal radii make equal angles")

        # ---- the ellipse, its foci, a point P and its focal radii
        ell = Ellipse(width=2 * a, height=2 * b, color=WHITE, stroke_width=3).move_to(Z)
        dF1, dF2 = _dot(F1, YELLOW_B, 0.07), _dot(F2, YELLOW_B, 0.07)
        rayX = [E(tk * DEGREES) for tk in ray_t]
        # the foci's labels sit in the widest gap among every line that will meet them
        lF1 = tag("F₁", 28, YELLOW_B).move_to(
            F1 + 0.42 * _free_dir(F1, [P, 2 * F1 - F2p] + rayX))
        lF2 = tag("F₂", 28, YELLOW_B).move_to(F2 + 0.42 * _free_dir(F2, [P, F2p] + rayX))
        self.play(Create(ell), run_time=1.2)
        self.play(FadeIn(dF1), FadeIn(dF2), FadeIn(lF1), FadeIn(lF2), run_time=0.5)
        dP = _dot(P)
        lP = tag("P", 28).move_to(P + 0.42 * UP)
        r1 = Line(F1, P, color=BLUE_B, stroke_width=4)
        r2 = Line(P, F2, color=TEAL_B, stroke_width=4)
        self.play(FadeIn(dP), FadeIn(lP), run_time=0.4)
        self.play(Create(r1), Create(r2), run_time=0.8)
        self.bring_to_front(dF1, dF2, dP)

        # ---- the readout: the string F₁PF₂ laid straight is 2a long
        x0, y1, y2, sc = -4.6, 3.4, 3.0, 0.75
        bar2a = Line([x0, y1, 0], [x0 + sc * 2 * a, y1, 0], color=WHITE, stroke_width=7)
        l2a = tag("2a", 26)
        l2a.move_to([x0 - 0.2 - l2a.width / 2, y1, 0])
        end = DashedLine([x0 + sc * 2 * a, y1 + 0.18, 0], [x0 + sc * 2 * a, y2 - 0.18, 0],
                         color=GREY_B, stroke_width=2, dash_length=0.06)
        s = ValueTracker(0.0)

        def Qpt():
            return P + s.get_value() * u

        def lowbar():
            Q = Qpt()
            d1, d2 = n(Q - F1), n(Q - F2)
            return VGroup(Line([x0, y2, 0], [x0 + sc * d1, y2, 0], color=BLUE_B, stroke_width=7),
                          Line([x0 + sc * d1, y2, 0], [x0 + sc * (d1 + d2), y2, 0],
                               color=TEAL_B, stroke_width=7))

        bar = lowbar()
        lsP = tag("PF₁ + PF₂", 24)
        lsP.move_to([x0 - 0.2 - lsP.width / 2, y2, 0])
        lsQ = tag("QF₁ + QF₂", 24, ORANGE)
        lsQ.move_to([x0 - 0.2 - lsQ.width / 2, y2, 0])
        self.play(FadeIn(bar2a), FadeIn(l2a), FadeIn(bar), FadeIn(lsP), Create(end),
                  run_time=1.0)
        self.hold(0.4)

        # ---- the tangent at P
        tl = Line(P - 3.8 * u, P + 3.8 * u, color=YELLOW_B, stroke_width=4)
        lt = tag("t", 30, YELLOW_B).move_to(P + 3.8 * u + 0.36 * nout)
        self.play(Create(tl), FadeIn(lt), run_time=0.8)

        # ---- F₂ mirrored in t
        mir = DashedLine(F2, F2p, color=GREY_A, stroke_width=2.5, dash_length=0.09)
        raH = _ra(H, F2p, P, 0.18, GREY_A, 2)
        tk = VGroup(_ticks(F2, H, 1, GREY_A), _ticks(H, F2p, 1, GREY_A))
        self.play(Create(mir), Create(raH), FadeIn(tk), run_time=0.8)
        r2c = r2.copy()
        self.add(r2c)
        self.play(_fold(r2c, P, P + u), run_time=1.3)
        check(close(r2c.get_end(), F2p, 1e-6) or close(r2c.get_start(), F2p, 1e-6),
              "the fold carries F₂ onto F₂′")
        r2p = Line(P, F2p, color=TEAL_B, stroke_width=4)
        self.remove(r2c)
        self.add(r2p)
        dF2p = _dot(F2p, TEAL_B, 0.07)
        lF2p = tag("F₂′", 28, TEAL_B).move_to(F2p + 0.45 * _free_dir(F2p, [F2, P]))
        self.play(FadeIn(dF2p), FadeIn(lF2p), run_time=0.4)
        self.bring_to_front(dP)

        # ---- Q on t: F₁Q + QF₂ = F₁Q + QF₂′ > 2a, least at P
        def qparts():
            Q = Qpt()
            g = VGroup(Line(F1, Q, color=BLUE_B, stroke_width=3),
                       Line(Q, F2, color=TEAL_B, stroke_width=3),
                       DashedLine(Q, F2p, color=TEAL_B, stroke_width=3, dash_length=0.08))
            if abs(s.get_value()) > 0.25:
                g.add(_ticks(Q, F2, 2, TEAL_B, at=0.62), _ticks(Q, F2p, 2, TEAL_B, at=0.62))
            g.add(_dot(Q, ORANGE, 0.08))
            return g

        def qlabel():
            Q = Qpt()
            # fades out as Q nears P, so it never sits on P's label
            op = float(np.clip((abs(s.get_value()) - 0.35) / 0.4, 0.0, 1.0))
            return tag("Q", 28, ORANGE).move_to(
                Q + 0.42 * _free_dir(Q, [F1, F2, F2p, Q + u, Q - u])).set_opacity(op)

        for sv in np.linspace(-2.3, 2.3, 47):          # Q's label stays clear
            s.set_value(sv)
            Q = Qpt()
            ql = qlabel()
            segs_q = [_seg(F1, Q), _seg(Q, F2), _seg(Q, F2p), _seg(P - 3.8 * u, P + 3.8 * u),
                      _seg(F2, F2p)]
            check(_hit(ql, segs_q, 0.04) is None, f"Q's label clear of its lines (s={sv:.2f})")
            check(_in_frame(ql), "Q's label inside the frame")
            if abs(sv) > 0.35:                         # visible: clear of the other labels
                check(not any(_overlap(ql, m, 0.03) for m in (lP, lt, lF1, lF2, lF2p)),
                      f"Q's label clear of the point labels (s={sv:.2f})")
        s.set_value(0.0)
        qg = always_redraw(qparts)
        ql = always_redraw(qlabel)
        lbar = always_redraw(lowbar)
        self.remove(bar)
        self.add(lbar)
        self.play(r1.animate.set_stroke(opacity=0.3), r2.animate.set_stroke(opacity=0.3),
                  r2p.animate.set_stroke(opacity=0.3), FadeIn(qg), FadeIn(ql),
                  ReplacementTransform(lsP, lsQ), run_time=0.6)
        self.play(s.animate.set_value(2.3), run_time=1.6)
        self.play(s.animate.set_value(-2.3), run_time=2.4)
        self.play(s.animate.set_value(0.0), run_time=1.4)
        for m in (qg, ql, lbar):
            m.clear_updaters()
        self.play(FadeOut(qg), FadeOut(ql), FadeOut(lbar), FadeOut(bar2a), FadeOut(l2a),
                  FadeOut(lsQ), FadeOut(end),
                  r1.animate.set_stroke(opacity=1.0), r2.animate.set_stroke(opacity=1.0),
                  r2p.animate.set_stroke(opacity=1.0), run_time=0.6)

        # ---- the shortest path F₁ → F₂′ is straight: F₁, P, F₂′ in a line
        self.play(ShowPassingFlash(Line(F1, F2p, color=WHITE, stroke_width=9),
                                   time_width=0.5), run_time=1.3)
        self.bring_to_front(dF1, dP, dF2p)

        # ---- equal angles at P
        rr = 0.72
        aw = [angle_arc(P, F1, P - u, rr, GREEN_B, 4),
              angle_arc(P, P + u, F2p, rr, GREEN_B, 4),
              angle_arc(P, F2, P + u, rr, GREEN_B, 4)]
        la = [tag("α", 26, GREEN_B).move_to(P + (rr + 0.34) * angle_mid_dir(P, F1, P - u)),
              tag("α", 26, GREEN_B).move_to(P + (rr + 0.34) * angle_mid_dir(P, P + u, F2p)),
              tag("α", 26, GREEN_B).move_to(P + (rr + 0.34) * angle_mid_dir(P, F2, P + u))]
        self.play(*[Create(x) for x in aw], *[FadeIn(x) for x in la], run_time=1.0)

        # ---- every ray from F₁ reflects to F₂
        rays = VGroup(*[VMobject(stroke_color=YELLOW_A, stroke_width=1.8, stroke_opacity=0.75)
                        .set_points_as_corners([F1, E(tk * DEGREES), F2]) for tk in ray_t])
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.2), run_time=1.6)
        self.bring_to_front(dF1, dF2, dP)

        segs = ([_seg(ell.point_from_proportion(i / 160), ell.point_from_proportion((i + 1) / 160))
                 for i in range(160)]
                + [_seg(F1, P), _seg(P, F2), _seg(P, F2p), _seg(P - 3.8 * u, P + 3.8 * u),
                   _seg(F2, F2p)]
                + _ra_segs(H, F2p, P, 0.18)
                + [_seg(F1, E(tk * DEGREES)) for tk in ray_t]
                + [_seg(E(tk * DEGREES), F2) for tk in ray_t]
                + _arc_segs(P, rr, 0, TAU, 72))
        _labels_ok([lF1, lF2, lP, lt, lF2p] + la, segs, "E18")
        cap = caption("PF₁ + PF₂ = 2a   ⟹   equal angles at P:  a ray from F₁ "
                      "reflects to F₂", 30)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E29  — a radius square to a chord bisects it

class E29_RadiusBisectsChord(Board):
    """The radius ON meets the chord AB at M at right angles. Fold the
    picture over the line ON (a diameter): the circle goes to itself, and
    so does the line AB, which is square to the fold. A and B are the only
    points where that line meets the circle, so the fold swaps them, and
    the half AM lands on MB: AM = MB. Any chord square to the radius is cut
    the same way."""

    def construct(self):
        R = 3.0
        O = np.array([0.0, 0.6, 0.0])
        v = _dir(15 * DEGREES)                 # along the chord
        w = _dir(-75 * DEGREES)                # along the radius ON
        N = O + R * w
        n = np.linalg.norm

        def chord(d):
            M = O + d * w
            h = float(np.sqrt(R * R - d * d))
            return M, M - h * v, M + h * v

        for d in np.linspace(0.3, 2.7, 25):
            M, A, B = chord(d)
            check(abs(n(A - O) - R) < 1e-9 and abs(n(B - O) - R) < 1e-9,
                  "A and B on the circle")
            check(abs(np.dot(M - O, B - A)) < 1e-9, "OM ⊥ AB")
            check(close(_refl(A, O, w), B), "the fold over ON carries A onto B")
            check(abs(n(A - M) - n(B - M)) < 1e-9, "AM = MB")
        d0 = 1.7
        M, A, B = chord(d0)
        check(np.dot(A - O, v) < 0 < np.dot(B - O, v), "A and B on either side of ON")

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        dO = _dot(O)
        lO = tag("O", 28).move_to(O + 0.4 * _dir(195 * DEGREES))
        self.play(Create(circ), FadeIn(dO), FadeIn(lO), run_time=1.0)
        ab = Line(A, B, color=WHITE, stroke_width=4)
        dA, dB = _dot(A), _dot(B)
        lA = tag("A", 28).move_to(A + 0.42 * _unit(A - O))
        lB = tag("B", 28).move_to(B + 0.42 * _unit(B - O))
        self.play(Create(ab), FadeIn(dA), FadeIn(dB), FadeIn(lA), FadeIn(lB), run_time=0.9)

        # the radius ON, square to the chord at M; the diameter along it
        on = Line(O, N, color=YELLOW_B, stroke_width=4)
        axis = DashedLine(O - R * w, O, color=GREY_A, stroke_width=2.5, dash_length=0.1)
        dN, dM = _dot(N), _dot(M, YELLOW_B)
        lN = tag("N", 28).move_to(N + 0.42 * w)
        lM = tag("M", 28, YELLOW_B).move_to(M + 0.42 * _dir(-30 * DEGREES))
        raM = _ra(M, O, B, 0.22, YELLOW_B, 2.5)
        self.play(Create(on), Create(axis), FadeIn(dN), FadeIn(lN), run_time=0.9)
        self.play(FadeIn(dM), FadeIn(lM), Create(raM), run_time=0.6)
        self.hold(0.4)

        # fold the half on A's side over the diameter through N
        half = Sector(radius=R, start_angle=(-75 + 180) * DEGREES, angle=PI, arc_center=O,
                      fill_color=BLUE_D, fill_opacity=0.45, stroke_color=BLUE_B, stroke_width=2)
        check(all(np.dot(_dir(t), v) < 1e-9 for t in np.linspace(105, 285, 37) * DEGREES),
              "the sector is the half on A's side")
        half_ab = Line(A, M, color=BLUE_B, stroke_width=8)
        flap = VGroup(half, half_ab)
        self.add(flap)                                 # in the scene before the fold plays,
        self.bring_to_front(dO, dM, lO, lM)            # so the labels stay on top of it
        self.play(FadeIn(half), Create(half_ab), run_time=0.7)
        self.play(_fold(flap, O, N), run_time=2.0)
        check(close(half_ab.get_start(), B, 1e-6) and close(half_ab.get_end(), M, 1e-6),
              "folded, AM lies on BM with A on B")
        self.bring_to_front(on, dO, dM, dN, dA, dB)
        tk = VGroup(_ticks(A, M, 2, YELLOW_B), _ticks(M, B, 2, YELLOW_B))
        self.play(FadeOut(flap), FadeIn(tk), run_time=0.7)
        self.hold(0.3)

        # ... and for every chord square to the radius
        dd = ValueTracker(d0)

        def parts():
            M_, A_, B_ = chord(dd.get_value())
            return VGroup(Line(A_, B_, color=WHITE, stroke_width=4),
                          _ra(M_, O, B_, 0.22, YELLOW_B, 2.5),
                          _ticks(A_, M_, 2, YELLOW_B), _ticks(M_, B_, 2, YELLOW_B),
                          _dot(A_), _dot(B_), _dot(M_, YELLOW_B))

        def labels():
            M_, A_, B_ = chord(dd.get_value())
            return VGroup(tag("A", 28).move_to(A_ + 0.42 * _unit(A_ - O)),
                          tag("B", 28).move_to(B_ + 0.42 * _unit(B_ - O)),
                          tag("M", 28, YELLOW_B).move_to(M_ + 0.42 * _dir(-30 * DEGREES)))

        for d in np.linspace(0.75, 2.2, 16):            # labels stay clear while it moves
            dd.set_value(d)
            M_, A_, B_ = chord(d)
            L = labels()
            segs = (_arc_segs(O, R, 0, TAU, 96) + [_seg(A_, B_), _seg(O - R * w, N)]
                    + _ra_segs(M_, O, B_, 0.22))
            _labels_ok(list(L) + [lO, lN], segs, f"E29 (d = {d:.2f})")
        dd.set_value(d0)
        live = always_redraw(parts)
        live_l = always_redraw(labels)
        self.remove(ab, raM, tk, dA, dB, dM, lA, lB, lM)
        self.add(live, live_l)
        self.bring_to_front(on, dO, dN)
        self.play(dd.animate.set_value(0.75), run_time=1.3)
        self.play(dd.animate.set_value(2.2), run_time=1.6)
        self.play(dd.animate.set_value(d0), run_time=1.0)
        live.clear_updaters()
        live_l.clear_updaters()
        self.bring_to_front(live_l)
        cap = caption("OM ⊥ AB   ⟹   AM  =  MB", 36)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E25  — two secants from a point

class E25_TwoSecants(Board):
    """Two secants from P: the first meets the circle at A then B, the
    second at C then D. Triangles PAD and PCB share the angle at P, and
    their angles at D and B are inscribed angles on the same arc AC, so
    they are equal. Flipped over the bisector of the angle at P, PAD lies
    along the other two sides (A on PC, D on PB); enlarged from P by
    PC : PA it lands exactly on PCB. Similar triangles:
    PA : PC = PD : PB, i.e. PA·PB = PC·PD."""

    def construct(self):
        R = 2.75
        O = np.array([-2.6, 0.5, 0.0])
        P = np.array([2.9, -0.8, 0.0])
        n = np.linalg.norm
        e = _unit(O - P)

        def ends(u):
            b = float(np.dot(u, O - P))
            disc = float(np.sqrt(b * b - (np.dot(O - P, O - P) - R * R)))
            return P + (b - disc) * u, P + (b + disc) * u

        def turn(v, t):
            return np.array([v[0] * np.cos(t) - v[1] * np.sin(t),
                             v[0] * np.sin(t) + v[1] * np.cos(t), 0.0])

        u1, u2 = turn(e, -8 * DEGREES), turn(e, 26 * DEGREES)
        A, B = ends(u1)
        C, D = ends(u2)
        w = _unit(_unit(A - P) + _unit(C - P))          # bisector of the angle at P
        k = n(C - P) / n(A - P)
        As, Ds = _refl(A, P, w), _refl(D, P, w)
        for X in (A, B, C, D):
            check(abs(n(X - O) - R) < 1e-9, "A, B, C, D on the circle")
        check(n(A - P) < n(B - P) and n(C - P) < n(D - P), "A, C the near points")
        check(abs(_ang(D, P, A) - _ang(B, P, C)) < 1e-9, "∠PDA = ∠PBC (same arc AC)")
        check(close(As, P + n(A - P) * _unit(C - P)) and close(Ds, P + n(D - P) * _unit(B - P)),
              "the flip puts A on PC and D on PB")
        check(abs(_cross(Ds - As, B - C)) < 1e-9, "flipped, AD is parallel to CB")
        check(close(P + k * (As - P), C) and close(P + k * (Ds - P), B),
              "enlarged from P by PC : PA it lands on PCB")
        check(abs(n(A - P) * n(B - P) - n(C - P) * n(D - P)) < 1e-9, "PA·PB = PC·PD")
        check(k > 1.25, "the enlargement is plain to see")

        def tdirs(X):                                   # the circle's tangent at X
            a = _heading(O, X)
            return [a + PI / 2, a - PI / 2]

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        dP = _dot(P, YELLOW_B, 0.07)
        lP = tag("P", 30, YELLOW_B).move_to(P + 0.42 * RIGHT)
        self.play(Create(circ), FadeIn(dP), FadeIn(lP), run_time=1.0)
        s1 = Line(P, B, color=WHITE, stroke_width=3)
        s2 = Line(P, D, color=WHITE, stroke_width=3)
        dots = VGroup(*[_dot(X) for X in (A, B, C, D)])
        labs = {}
        for name, X, others in (("A", A, [P, B, D]), ("B", B, [P, C]),
                                ("C", C, [P, D, B]), ("D", D, [P, A])):
            labs[name] = tag(name, 28).move_to(X + 0.42 * _free_dir(X, others, tdirs(X)))
        self.play(Create(s1), Create(s2), run_time=1.1)
        self.play(FadeIn(dots), *[FadeIn(m) for m in labs.values()], run_time=0.5)

        # the two triangles: light fills under the secants, coloured third sides
        tPAD = mk([P, A, D], BLUE_D, 0.34, stroke_width=0)
        tPCB = mk([P, C, B], ORANGE, 0.28, stroke_width=0)
        cAD = Line(A, D, color=BLUE_B, stroke_width=4)
        cCB = Line(C, B, color=ORANGE, stroke_width=4)
        self.add(tPAD, tPCB)
        self.bring_to_back(tPAD, tPCB)
        tPCB.set_fill(opacity=0)
        tPAD.set_fill(opacity=0)
        self.play(tPAD.animate.set_fill(opacity=0.34), Create(cAD), run_time=0.8)
        self.play(tPCB.animate.set_fill(opacity=0.28), Create(cCB), run_time=0.8)
        self.bring_to_front(s1, s2, cAD, cCB, dots, dP)

        # the shared angle at P; the angles at D and B stand on the arc AC
        aP = angle_arc(P, A, C, 1.15, YELLOW_B, 4)
        lphi = tag("φ", 28, YELLOW_B).move_to(P + 1.55 * angle_mid_dir(P, A, C))
        self.play(Create(aP), FadeIn(lphi), run_time=0.6)
        tA, tC = _heading(O, A), _heading(O, C)
        arcAC = Arc(radius=R, start_angle=tC, angle=(tA - tC) % TAU, arc_center=O,
                    color=GREEN_B, stroke_width=8)
        check((tA - tC) % TAU < PI, "the arc AC faces P")
        rr = 0.95
        aD = angle_arc(D, P, A, rr, GREEN_B, 4)
        aB = angle_arc(B, P, C, rr, GREEN_B, 4)
        lbD = tag("β", 26, GREEN_B).move_to(D + (rr + 0.65) * angle_mid_dir(D, P, A))
        lbB = tag("β", 26, GREEN_B).move_to(B + (rr + 0.65) * angle_mid_dir(B, P, C))
        self.play(Create(arcAC), run_time=0.7)
        self.play(Create(aD), Create(aB), FadeIn(lbD), FadeIn(lbB), run_time=0.8)
        self.bring_to_front(dots, dP)
        self.hold(0.6)

        # flip PAD over the bisector at P, then enlarge it from P
        axis = DashedLine(P, P + 6.2 * w, color=GREY_A, stroke_width=2, dash_length=0.1)
        cp = VGroup(mk([P, A, D], BLUE_D, 0.6, stroke_color=BLUE_B, stroke_width=4),
                    angle_arc(D, P, A, rr, GREEN_B, 5))
        self.add(cp)
        self.bring_to_front(dots, dP, *labs.values())
        self.play(Create(axis), run_time=0.6)
        self.play(_fold(cp, P, P + w), run_time=1.8)
        check(_same_vertices_ok(cp[0], [P, As, Ds]), "the flip lands on P, A*, D*")
        chev = VGroup(_chevron(As, Ds, YELLOW_B), _chevron(C, B, YELLOW_B))
        self.play(FadeOut(axis), FadeIn(chev), run_time=0.6)
        tagk = tag("× PC : PA", 26, BLUE_B).move_to([4.6, 0.55, 0.0])
        self.play(FadeIn(tagk), run_time=0.4)
        self.play(cp.animate.scale(k, about_point=P), FadeOut(chev), run_time=1.6)
        check(_same_vertices_ok(cp[0], [P, C, B]), "the enlarged copy is PCB")
        self.bring_to_front(dots, dP)
        self.hold(0.4)

        row = VGroup(tag("PA", 30, BLUE_B), tag(":", 30), tag("PC", 30, ORANGE),
                     tag("=", 30), tag("PD", 30, BLUE_B), tag(":", 30),
                     tag("PB", 30, ORANGE)).arrange(RIGHT, buff=0.16).move_to([3.9, 2.7, 0.0])
        self.play(FadeOut(cp), FadeOut(tagk), FadeIn(row), run_time=0.8)

        segs = (_arc_segs(O, R, 0, TAU, 120) + [_seg(P, B), _seg(P, D), _seg(A, D), _seg(C, B)]
                + _arc_segs(P, 1.15, 0, TAU, 60))
        texts = list(labs.values()) + [lP, lphi, lbD, lbB, row]
        _labels_ok(texts, segs + _arc_segs(D, rr, 0, TAU, 40) + _arc_segs(B, rr, 0, TAU, 40),
                   "E25", pad=0.05)
        _labels_ok([tagk], segs, "E25 scale tag")
        cap = caption("PA · PB  =  PC · PD", 36)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E21  — the arbelos

def _arc_pts(c, r, a0, a1, n=120):
    """n + 1 screen points along the circle (c, r) from angle a0 to a1."""
    c = to3(c)
    return [c + r * _dir(t) for t in np.linspace(a0, a1, n + 1)]


def _tiles_exactly(polys, region, n=110):
    """True if the polygons tile `region` with no gap and no overlap: on a
    jittered grid over the region's box every point inside the region lies
    in exactly one polygon and every point outside in none."""
    def pip(pt, poly):
        x, y = float(pt[0]), float(pt[1])
        inside = False
        for i in range(len(poly)):
            x1, y1 = float(poly[i][0]), float(poly[i][1])
            x2, y2 = float(poly[(i + 1) % len(poly)][0]), float(poly[(i + 1) % len(poly)][1])
            if (y1 > y) != (y2 > y) and x1 + (y - y1) * (x2 - x1) / (y2 - y1) > x:
                inside = not inside
        return inside
    xs = [float(p[0]) for p in region]
    ys = [float(p[1]) for p in region]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for i in range(n):
        for j in range(n):
            x = x0 + (x1 - x0) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = y0 + (y1 - y0) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            if sum(pip((x, y), P) for P in polys) != (1 if pip((x, y), region) else 0):
                return False
    return True


class E21_Arbelos(Board):
    """Archimedes' arbelos: the big semicircle on AB less the semicircles on
    AC and CB. A semicircle on a diameter d has area π/8·d², so the arbelos
    is π/8·(AB² − AC² − CB²); in the square on AB = AC + CB the squares on
    AC and CB leave two rectangles AC·CB. Each of these is the square on
    the half-chord CD: ADB is right-angled at D, so turning the triangle CDB
    a quarter turn about C puts its hypotenuse parallel to AD; then the
    triangle with legs AC and CB (half the rectangle) shears — base kept,
    apex sliding along AD — into the triangle with legs CD and CD (half the
    square). So the arbelos is π/8·2·CD², the circle on CD."""

    def construct(self):
        p, q = 3.8, 2.2                                  # AC, CB
        y0 = -1.2
        A = np.array([-6.3, y0, 0.0])
        C, B = A + p * RIGHT, A + (p + q) * RIGHT
        O, R = (A + B) / 2, (p + q) / 2
        O1, r1 = (A + C) / 2, p / 2
        O2, r2 = (C + B) / 2, q / 2
        h = float(np.sqrt(p * q))
        D = C + h * UP
        Dq, Bq = C + h * LEFT, C + q * UP                # D, B turned a quarter turn about C
        n = np.linalg.norm

        check(abs(n(D - O) - R) < 1e-9, "D is on the big semicircle")
        check(abs(np.dot(A - D, B - D)) < 1e-9, "ADB is right-angled at D")
        check(close(_rot90(D, C), Dq) and close(_rot90(B, C), Bq),
              "a quarter turn about C takes D to D′ and B to B′")
        check(0 < n(Dq - C) < n(A - C) and 0 < n(Bq - C) < n(D - C),
              "D′ lies on CA and B′ on CD")
        check(abs(_cross(Bq - Dq, D - A)) < 1e-9, "D′B′ ∥ AD")
        check(abs(abs(area([C, A, Bq])) - p * q / 2) < 1e-9
              and abs(abs(area([C, Dq, D])) - h * h / 2) < 1e-9, "the two right triangles")
        check(_tiles_exactly([[C, Dq, Bq], [Dq, A, Bq]], [C, A, Bq])
              and _tiles_exactly([[C, Dq, Bq], [Dq, D, Bq]], [C, Dq, D]),
              "CAB′ = CD′B′ + D′AB′ and CD′D = CD′B′ + D′DB′")
        check(abs(abs(area([Dq, A, Bq])) - abs(area([Dq, D, Bq]))) < 1e-9,
              "the shear keeps the area")
        arb = (_arc_pts(O, R, PI, 0, 600) + _arc_pts(O2, r2, 0, PI, 600)[1:]
               + _arc_pts(O1, r1, 0, PI, 600)[1:-1])
        check(abs(abs(area(arb)) - PI / 8 * ((p + q) ** 2 - p * p - q * q)) < 1e-3
              and abs(PI / 8 * ((p + q) ** 2 - p * p - q * q) - PI * (h / 2) ** 2) < 1e-9,
              "arbelos = π/8·(AB² − AC² − CB²) = π·(CD/2)²")

        # ---- the arbelos
        base = Line(A, B, color=WHITE, stroke_width=3)
        big = Arc(radius=R, start_angle=0, angle=PI, arc_center=O, color=WHITE, stroke_width=3)
        lA = tag("A", 28).move_to(A + 0.4 * DOWN)
        lC = tag("C", 28).move_to(C + 0.4 * DOWN)
        lB = tag("B", 28).move_to(B + 0.4 * DOWN)
        self.play(Create(base), Create(big), FadeIn(lA), FadeIn(lC), FadeIn(lB), run_time=1.3)
        s1 = mk(_arc_pts(O1, r1, 0, PI, 120), BLUE_D, 0.75)
        s2 = mk(_arc_pts(O2, r2, 0, PI, 120), TEAL_D, 0.75)
        arbelos = mk(_arc_pts(O, R, PI, 0, 160) + _arc_pts(O2, r2, 0, PI, 90)[1:]
                     + _arc_pts(O1, r1, 0, PI, 120)[1:-1], YELLOW_E, 0.75, stroke_width=0)
        self.play(FadeIn(s1), FadeIn(s2), run_time=0.9)
        self.add(arbelos)
        self.bring_to_back(arbelos)
        arbelos.set_fill(opacity=0)
        self.play(arbelos.animate.set_fill(opacity=0.75), run_time=0.8)

        # ---- a semicircle on d is π/8 · d²
        ic = np.array([-5.75, 2.75, 0.0])
        icon = VGroup(mk(_arc_pts(ic, 0.36, 0, PI, 40), GREY_B, 0.6, stroke_width=2),
                      tag("d", 24).move_to(ic + 0.24 * DOWN))
        ieq = tag("=  π/8 · d²", 28).next_to(icon, RIGHT, buff=0.25).align_to(icon[0], DOWN)
        self.play(FadeIn(icon), FadeIn(ieq), run_time=0.8)

        # ---- the square on AB: AC², CB² and two rectangles AC·CB
        x0, yT = 0.35, 3.15
        S = p + q
        sqAB = [np.array([x0, yT, 0]), np.array([x0 + S, yT, 0]),
                np.array([x0 + S, yT - S, 0]), np.array([x0, yT - S, 0])]

        def rect(xa, ya, xb, yb):
            return [np.array([xa, ya, 0]), np.array([xb, ya, 0]),
                    np.array([xb, yb, 0]), np.array([xa, yb, 0])]

        qAC = rect(x0, yT, x0 + p, yT - p)
        qCB = rect(x0 + p, yT - p, x0 + S, yT - S)
        rc1 = rect(x0 + p, yT, x0 + S, yT - p)
        rc2 = rect(x0, yT - p, x0 + p, yT - S)
        check(_tiles_exactly([qAC, qCB, rc1, rc2], sqAB),
              "AC², CB² and two rectangles AC·CB tile the square on AB")
        check(abs(abs(area(rc1)) - p * q) < 1e-9 and abs(abs(area(rc2)) - p * q) < 1e-9,
              "each rectangle is AC·CB")
        frame = Polygon(*sqAB, stroke_color=WHITE, stroke_width=3)
        eAC = tag("AC", 24).move_to([x0 + p / 2, yT + 0.3, 0])
        eCB = tag("CB", 24).move_to([x0 + p + q / 2, yT + 0.3, 0])
        tick = Line([x0 + p, yT + 0.12, 0], [x0 + p, yT - 0.12, 0], color=WHITE, stroke_width=3)
        self.play(Create(frame), FadeIn(eAC), FadeIn(eCB), Create(tick),
                  ShowPassingFlash(big.copy().set_stroke(YELLOW_B, 8), time_width=0.6),
                  run_time=0.9)
        pAC, pCB = mk(qAC, BLUE_D, 0.75), mk(qCB, TEAL_D, 0.75)
        lAC = tag("AC²", 30).move_to(np.mean(qAC, axis=0))
        lCB = tag("CB²", 28).move_to(np.mean(qCB, axis=0))
        self.play(FadeIn(pAC), FadeIn(lAC), Indicate(s1, color=WHITE, scale_factor=1.0),
                  run_time=0.7)
        self.play(FadeIn(pCB), FadeIn(lCB), Indicate(s2, color=WHITE, scale_factor=1.0),
                  run_time=0.7)
        pr1, pr2 = mk(rc1, YELLOW_E, 0.75), mk(rc2, YELLOW_E, 0.75)
        lr1 = tag("AC·CB", 26, BLACK).move_to(np.mean(rc1, axis=0))
        lr2 = tag("AC·CB", 26, BLACK).move_to(np.mean(rc2, axis=0))
        self.play(FadeIn(pr1), FadeIn(pr2), FadeIn(lr1), FadeIn(lr2), run_time=0.8)
        self.bring_to_front(frame)
        self.hold(0.5)

        # ---- the half-chord CD; ADB is right-angled at D
        self.play(arbelos.animate.set_fill(opacity=0.2), s1.animate.set_fill(opacity=0.15),
                  s2.animate.set_fill(opacity=0.15), run_time=0.5)
        cd = Line(C, D, color=WHITE, stroke_width=4)
        ad = Line(A, D, color=GREY_A, stroke_width=2.5)
        db = Line(D, B, color=GREY_A, stroke_width=2.5)
        dD = _dot(D)
        lD = tag("D", 28).move_to(D + 0.38 * UP)
        raD = _ra(D, A, B, 0.22, GREY_A, 2)
        raC = _ra(C, B, D, 0.2, GREY_A, 2)
        self.play(Create(cd), FadeIn(dD), FadeIn(lD), Create(raC), run_time=0.7)
        self.play(Create(ad), Create(db), Create(raD), run_time=0.7)

        # turn CDB a quarter turn about C: its hypotenuse turns square to DB, so ∥ AD
        tri = mk([C, D, B], TEAL_D, 0.65, stroke_color=TEAL_B, stroke_width=2.5)
        self.add(tri)
        self.bring_to_front(lC, lD)
        self.play(FadeIn(tri), run_time=0.4)
        self.play(Rotate(tri, angle=PI / 2, about_point=C), run_time=1.6)
        check(_same_vertices_ok(tri, [C, Dq, Bq]), "the turned triangle is CD′B′")
        tks = VGroup(_ticks(C, D, 1, YELLOW_B), _ticks(C, Dq, 1, YELLOW_B),
                     _ticks(C, B, 2, YELLOW_B, at=0.5), _ticks(C, Bq, 2, YELLOW_B, at=0.5))
        chev = VGroup(_chevron(A, D, YELLOW_B), _chevron(Dq, Bq, YELLOW_B))
        self.play(FadeIn(tks), FadeIn(chev), run_time=0.6)

        # the triangle with legs AC and CB (half of AC·CB) shears into legs CD, CD
        keep = mk([C, Dq, Bq], YELLOW_E, 0.8, stroke_width=1.5)
        slide = mk([Dq, A, Bq], YELLOW_E, 0.8, stroke_width=1.5)
        half_r = DashedVMobject(Polygon(C, A, A + q * UP, Bq, stroke_color=YELLOW_B,
                                        stroke_width=2), num_dashes=40)
        self.add(keep, slide)
        self.bring_to_front(lC, lA)
        self.play(FadeOut(tri), FadeIn(keep), FadeIn(slide), Create(half_r), run_time=0.8)
        gpar = guide(A, D, YELLOW_B)                    # the line the apex slides along
        bD = Line(Dq, Bq, color=YELLOW_B, stroke_width=7)
        self.play(Create(gpar), Create(bD), FadeOut(half_r), run_time=0.6)
        self.play(Transform(slide, mk([Dq, D, Bq], YELLOW_E, 0.8, stroke_width=1.5)),
                  run_time=1.8)
        check(_same_vertices_ok(slide, [Dq, D, Bq]), "the apex slid from A to D")
        half_s = DashedVMobject(Polygon(C, Dq, Dq + h * UP, D, stroke_color=YELLOW_B,
                                        stroke_width=2), num_dashes=40)
        self.play(FadeOut(gpar), FadeOut(bD), FadeOut(chev), Create(half_s), run_time=0.7)
        self.hold(0.3)
        nr1 = tag("CD²", 30, BLACK).move_to(lr1.get_center())
        nr2 = tag("CD²", 30, BLACK).move_to(lr2.get_center())
        self.play(Transform(lr1, nr1), Transform(lr2, nr2), run_time=0.8)
        self.hold(0.4)

        # ---- so the arbelos is π/8 · 2 CD²: the two halves of the circle on CD
        self.play(FadeOut(keep), FadeOut(slide), FadeOut(half_s), FadeOut(tks),
                  FadeOut(ad), FadeOut(db), FadeOut(raD),
                  arbelos.animate.set_fill(opacity=0.75), s1.animate.set_fill(opacity=0.75),
                  s2.animate.set_fill(opacity=0.75), run_time=0.7)
        self.bring_to_front(cd, raC, dD)
        cc = C + h / 2 * UP
        disc = Circle(radius=h / 2, color=ORANGE, stroke_width=4).move_to(cc)
        disc.set_fill(ORANGE, opacity=0.35)
        self.play(Create(disc), run_time=1.0)
        self.bring_to_front(cd, dD, lC, lD)

        segs = ([_seg(A, B), _seg(C, D)] + _arc_segs(O, R, 0, PI, 120)
                + _arc_segs(O1, r1, 0, PI, 80) + _arc_segs(O2, r2, 0, PI, 60)
                + _arc_segs(cc, h / 2, 0, TAU, 80) + _poly_segs(sqAB)
                + [_seg(qAC[1], qAC[2]), _seg(qAC[2], qAC[3]), _seg(qCB[0], qCB[3]),
                   _seg(qCB[0], qCB[1])])
        _labels_ok([lA, lC, lB, lD, eAC, eCB, ieq, icon[1]], segs, "E21")
        for lab, box in ((lAC, qAC), (lCB, qCB), (lr1, rc1), (lr2, rc2)):
            check(_in_frame(lab) and lab.get_left()[0] > box[0][0] + 0.1
                  and lab.get_right()[0] < box[1][0] - 0.1
                  and lab.get_top()[1] < box[0][1] - 0.1 and lab.get_bottom()[1] > box[2][1] + 0.1,
                  f"'{_name(lab)}' inside its piece")
        cap = caption("arbelos  =  π/8 · (AB² − AC² − CB²)  =  π/8 · 2 CD²  =  π (CD/2)²", 30)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E13  — the angle between two chords

class E13_ChordAngle(Board):
    """Chords AB and CD cross at P. The angle θ = ∠APC is the exterior
    angle of the triangle APD at P, so it is the sum of the two remote
    interior angles: slide the angle at D along DC to P and turn the angle
    at A half a turn about the midpoint of AP, and together they fill θ.
    Those two are inscribed angles: ∠ADC stands on the arc AC (α), ∠DAB on
    the arc DB (β), so they are α/2 and β/2, and θ = (α + β)/2. The same
    θ sits in the vertical angle BPD."""

    def construct(self):
        R = 2.95
        O = np.array([-0.4, 0.55, 0.0])
        deg = {"A": 205.0, "B": 45.0, "C": 125.0, "D": 265.0}
        A, B, C, D = (O + R * _dir(deg[k] * DEGREES) for k in "ABCD")
        al, be = (deg["A"] - deg["C"]) * DEGREES, (deg["B"] + 360 - deg["D"]) * DEGREES
        Mx = np.array([(B - A)[:2], (C - D)[:2]]).T
        s_, t_ = np.linalg.solve(Mx, (C - A)[:2])
        P = A + s_ * (B - A)
        K = (A + P) / 2
        th = _ang(P, A, C)
        check(0 < s_ < 1 and 0 < t_ < 1 and abs(_cross(P - C, D - C)) < 1e-9,
              "the chords cross at P inside the circle")
        check(abs(_ang(D, A, C) - al / 2) < 1e-9 and abs(_ang(A, D, B) - be / 2) < 1e-9,
              "the inscribed angles are half their arcs")
        check(abs(th - (al + be) / 2) < 1e-9 and abs(th - _ang(D, A, P) - _ang(A, D, P)) < 1e-9,
              "θ = ∠D + ∠A = (α + β)/2")
        check(abs(_ang(P, B, D) - th) < 1e-9, "the vertical angle BPD is θ too")
        rw = 1.0
        # the moved wedges: their rays at P, as unit vectors
        dD = [_unit(A - D), _unit(C - D)]                        # slid along DC
        dA = [-_unit(D - A), -_unit(B - A)]                      # half a turn
        check(close(dD[1], _unit(C - P)) and close(dA[1], _unit(A - P)) and close(dA[0], dD[0]),
              "slid and turned, the two angles share a side and fill ∠APC")
        check(abs(_ang(P, P + dD[0], C) + _ang(P, A, P + dD[0]) - th) < 1e-9,
              "… exactly, with no gap or overlap")
        check(close(2 * K - A, P), "the half-turn about the midpoint of AP takes A to P")

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        self.play(Create(circ), run_time=1.0)
        ab = Line(A, B, color=WHITE, stroke_width=3)
        cd = Line(C, D, color=WHITE, stroke_width=3)
        dots = VGroup(*[_dot(X) for X in (A, B, C, D)], _dot(P, YELLOW_B, 0.07))
        lab = {k: tag(k, 28).move_to(O + (R + 0.38) * _dir(deg[k] * DEGREES)) for k in "ABCD"}
        lP = tag("P", 28, YELLOW_B).move_to(P + 0.6 * _dir(70 * DEGREES))   # in ∠BPC
        self.play(Create(ab), Create(cd), run_time=1.0)
        self.play(FadeIn(dots), *[FadeIn(m) for m in lab.values()], FadeIn(lP), run_time=0.5)

        # θ and its vertical angle
        aT1 = angle_arc(P, A, C, 0.5, YELLOW_B, 4)
        aT2 = angle_arc(P, B, D, 0.5, YELLOW_B, 4)
        lT1 = tag("θ", 28, YELLOW_B).move_to(P + 0.85 * angle_mid_dir(P, A, C))
        lT2 = tag("θ", 28, YELLOW_B).move_to(P + 0.85 * angle_mid_dir(P, B, D))
        self.play(Create(aT1), Create(aT2), FadeIn(lT1), FadeIn(lT2), run_time=0.7)

        # the two arcs the angle and its vertical angle cut off
        arcAC = Arc(radius=R, start_angle=deg["C"] * DEGREES, angle=al, arc_center=O,
                    color=BLUE_B, stroke_width=8)
        arcDB = Arc(radius=R, start_angle=deg["D"] * DEGREES, angle=be, arc_center=O,
                    color=ORANGE, stroke_width=8)
        lal = tag("α", 30, BLUE_B).move_to(O + (R + 0.45) * _dir(deg["C"] * DEGREES + al / 2))
        lbe = tag("β", 30, ORANGE).move_to(O + (R + 0.45) * _dir(deg["D"] * DEGREES + be / 2))
        self.play(Create(arcAC), Create(arcDB), FadeIn(lal), FadeIn(lbe), run_time=0.9)
        self.bring_to_front(dots)

        # the triangle APD; its angles at D and A are inscribed angles
        ad = Line(A, D, color=GREY_A, stroke_width=3)
        tri = mk([A, P, D], GREY_D, 0.45, stroke_width=0)
        self.add(tri)
        self.bring_to_back(tri)
        self.play(Create(ad), FadeIn(tri), run_time=0.7)
        wD = _wedge(D, A, C, rw, BLUE_D, 0.8)
        wA = _wedge(A, D, B, rw, ORANGE, 0.75)
        lD2 = tag("α/2", 26, BLUE_B).move_to(D + (rw + 0.48) * angle_mid_dir(D, A, C))
        lA2 = tag("β/2", 26, ORANGE).move_to(A + (rw + 0.45) * angle_mid_dir(A, D, B))
        self.play(FadeIn(wD), FadeIn(wA), FadeIn(lD2), FadeIn(lA2), run_time=0.8)
        self.bring_to_front(ab, cd, ad, dots)
        self.hold(0.5)

        # slide one, turn the other: they fill θ — the exterior angle
        mD, mA = wD.copy(), wA.copy()
        piv = _dot(K, WHITE, 0.05)
        self.add(mD, mA)
        self.play(FadeIn(piv), FadeOut(lT1), run_time=0.3)
        self.play(mD.animate.shift(P - D), Rotate(mA, angle=PI, about_point=K), run_time=1.9)
        check(close(mD.get_arc_center(), P, 1e-6) and close(mA.get_arc_center(), P, 1e-6),
              "both corners land on P")
        self.bring_to_front(ab, cd, aT1, dots)
        lP1 = tag("α/2", 24, BLUE_B).move_to(P + (rw + 0.42) * _dir(np.arctan2(
            *(dD[0] + dD[1])[1::-1])))
        lP2 = tag("β/2", 24, ORANGE).move_to(P + (rw + 0.38) * _dir(np.arctan2(
            *(dA[0] + dA[1])[1::-1])))
        self.play(FadeOut(piv), FadeIn(lP1), FadeIn(lP2), run_time=0.6)

        def span_segs(v, p_, q_, r):
            a0, sp = _wedge_span(v, p_, q_)
            return _arc_segs(v, r, a0, a0 + sp, 24)

        segs = (_arc_segs(O, R, 0, TAU, 120) + [_seg(A, B), _seg(C, D), _seg(A, D)]
                + span_segs(P, A, C, 0.5) + span_segs(P, B, D, 0.5) + span_segs(P, A, C, rw)
                + span_segs(D, A, C, rw) + span_segs(A, D, B, rw)
                + [_seg(P, P + rw * dD[0])])
        _labels_ok(list(lab.values()) + [lP, lT2, lal, lbe, lD2, lA2, lP1, lP2], segs, "E13")
        cap = caption("θ  =  α/2 + β/2  =  (α + β)/2", 36)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E30  — the tangent is square to the radius

class E30_TangentRadius(Board):
    """The tangent t touches the circle only at T: every other point X of t
    lies outside, so OX > OT — the radius OT is the shortest way from O to
    t. Turn the line about T away from t: the foot F of the perpendicular
    from O is a point of the turned line, and the right triangle OFT has
    hypotenuse OT, so OF < OT: F is inside the circle and the line cuts it
    again at S, the mirror image of T in F. Only when F is T itself does
    the line keep outside, i.e. only the line square to OT is a tangent."""

    def construct(self):
        R = 2.7
        O = np.array([-0.5, 0.2, 0.0])
        aT = 20 * DEGREES
        T = O + R * _dir(aT)
        u = _dir(aT + PI / 2)                          # along t
        n = np.linalg.norm
        check(abs(np.dot(T - O, u)) < 1e-12, "t is square to OT")
        for s in np.linspace(-3.4, 2.4, 59):
            if abs(s) > 1e-9:
                check(n(T + s * u - O) > R + 1e-12, "every other point of t is outside: OX > OT")

        def tilt(phi):
            """The line through T turned by phi from t: its direction, the
            foot F from O and the second crossing S."""
            v = _dir(aT + PI / 2 + phi)
            F = _foot(O, T, T + v)
            return v, F, 2 * F - T

        for phi in np.linspace(-26, 32, 59) * DEGREES:
            v, F, S = tilt(phi)
            check(abs(np.dot(F - O, v)) < 1e-9 and abs(n(S - O) - R) < 1e-9,
                  "OF ⊥ the turned line; S on the circle")
            if abs(phi) > 1e-9:
                check(n(F - O) < R - 1e-12 and abs(n(F - O) - R * np.cos(phi)) < 1e-9,
                      "OF < OT: the turned line enters the circle")

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        dO = _dot(O)
        lO = tag("O", 28).move_to(O + 0.4 * _dir(aT + PI))
        self.play(Create(circ), FadeIn(dO), FadeIn(lO), run_time=1.0)
        dT = _dot(T, YELLOW_B, 0.07)
        lT = tag("T", 28, YELLOW_B).move_to(T + 0.42 * _dir(aT - 0.45))
        rOT = Line(O, T, color=YELLOW_B, stroke_width=4)
        self.play(Create(rOT), FadeIn(dT), FadeIn(lT), run_time=0.7)
        t0, t1 = T - 4.2 * u, T + 2.6 * u
        tl = Line(t0, t1, color=WHITE, stroke_width=3)
        lt = tag("t", 30).move_to(t0 + 0.36 * _dir(aT) + 0.22 * UP)      # at the lower end
        self.play(Create(tl), FadeIn(lt), run_time=0.8)
        self.bring_to_front(dT)

        # ---- every other point X of t is farther from O than T
        sX = ValueTracker(0.0)

        def xparts():
            X = T + sX.get_value() * u
            Y = O + R * _unit(X - O)
            g = VGroup(DashedLine(O, X, color=GREY_A, stroke_width=2.5, dash_length=0.08))
            if n(X - Y) > 0.02:
                g.add(Line(Y, X, color=ORANGE, stroke_width=7))
            g.add(_dot(X, ORANGE, 0.08))
            return g

        def xlabel():
            X = T + sX.get_value() * u
            op = float(np.clip((abs(sX.get_value()) - 0.6) / 0.3, 0.0, 1.0))
            return tag("X", 28, ORANGE).move_to(
                X + 0.42 * _free_dir(X, [O, X + u, X - u])).set_opacity(op)

        for sv in np.linspace(-3.2, 2.2, 28):
            sX.set_value(sv)
            X = T + sv * u
            if abs(sv) > 0.6:                          # where X's label is visible
                xl = xlabel()
                check(_hit(xl, [_seg(O, X), _seg(t0, t1)] + _arc_segs(O, R, 0, TAU, 96), 0.04)
                      is None and _in_frame(xl), f"X's label clear (s = {sv:.2f})")
                check(not any(_overlap(xl, m, 0.03) for m in (lT, lt, lO)), "X's label apart")
        sX.set_value(0.0)
        xg, xl = always_redraw(xparts), always_redraw(xlabel)
        self.play(FadeIn(xg), FadeIn(xl), run_time=0.4)
        self.play(sX.animate.set_value(2.2), run_time=1.3)
        self.play(sX.animate.set_value(-3.2), run_time=2.2)
        self.play(sX.animate.set_value(0.0), run_time=1.3)
        xg.clear_updaters()
        xl.clear_updaters()
        self.play(FadeOut(xg), FadeOut(xl), run_time=0.4)
        self.bring_to_front(rOT, dT)

        # ---- turn the line about T: the foot F comes inside, the line cuts again at S
        ph = ValueTracker(0.0)

        def ends(v):
            """The turned line through T, cut to the content area."""
            lo, hi = -4.0, 3.4
            for k in (0, 1):
                bnd = ((-SAFE_X + 0.1, SAFE_X - 0.1), (SAFE_BOTTOM + 0.1, SAFE_TOP - 0.1))[k]
                if abs(v[k]) > 1e-9:
                    ta, tb = sorted(((bnd[0] - T[k]) / v[k], (bnd[1] - T[k]) / v[k]))
                    lo, hi = max(lo, ta), min(hi, tb)
            return T + lo * v, T + hi * v

        def sparts():
            v, F, S = tilt(ph.get_value())
            e0, e1 = ends(v)
            g = VGroup(Line(e0, e1, color=ORANGE, stroke_width=3.5))
            if n(F - T) > 0.05:
                g.add(Line(T, S, color=ORANGE, stroke_width=8),
                      DashedLine(O, F, color=WHITE, stroke_width=2.5, dash_length=0.08))
                if n(F - T) > 0.3:
                    g.add(_ra(F, O, T, 0.18, WHITE, 2))
                g.add(_dot(S, ORANGE, 0.07), _dot(F, WHITE, 0.06))
            return g

        def slabels():
            v, F, S = tilt(ph.get_value())
            op = float(np.clip((n(F - T) - 0.35) / 0.3, 0.0, 1.0))
            segs = ([_seg(*ends(v)), _seg(O, F), _seg(O, T)] + _arc_segs(O, R, 0, TAU, 96)
                    + _ra_segs(F, O, T, 0.18))
            aS = _heading(O, S)
            lS = _park(tag("S", 26, ORANGE), S, segs,
                       _free_dir(S, [O, S + v, S - v], [aS + PI / 2, aS - PI / 2]), 0.42,
                       avoid=(lT, lO, lt))
            # F's label: inside the circle, between FS and FO (the mark is between FO, FT)
            fs = _unit(S - F) if n(S - F) > 1e-6 else v
            lF = _park(tag("F", 26), F, segs, _unit(fs + _unit(O - F)), 0.45,
                       avoid=(lT, lO, lt, lS))
            return VGroup(lF, lS).set_opacity(op)

        for phv in np.linspace(-24, 30, 28) * DEGREES:
            ph.set_value(phv)
            v, F, S = tilt(phv)
            if n(F - T) > 0.35:
                L = slabels()
                segs = ([_seg(*ends(v)), _seg(O, F), _seg(O, T)]
                        + _arc_segs(O, R, 0, TAU, 96))
                for m in L:
                    check(_hit(m, segs, 0.04) is None and _in_frame(m),
                          f"F, S labels clear (φ = {np.degrees(phv):.0f}°)")
                    check(not any(_overlap(m, k, 0.03) for k in (lT, lO, lt)),
                          "F, S labels apart from the others")
                check(not _overlap(L[0], L[1], 0.03), "F and S labels apart")
        ph.set_value(0.0)
        sg, sl = always_redraw(sparts), always_redraw(slabels)
        self.play(tl.animate.set_stroke(opacity=0.3), lt.animate.set_opacity(0.3),
                  FadeIn(sg), FadeIn(sl), run_time=0.5)
        self.bring_to_front(rOT, dO, dT)
        self.play(ph.animate.set_value(30 * DEGREES), run_time=1.6)
        self.hold(0.4)
        self.play(ph.animate.set_value(-24 * DEGREES), run_time=2.4)
        self.play(ph.animate.set_value(0.0), run_time=1.5)
        sg.clear_updaters()
        sl.clear_updaters()
        self.play(FadeOut(sg), FadeOut(sl), tl.animate.set_stroke(opacity=1.0),
                  lt.animate.set_opacity(1.0), run_time=0.5)

        ra = _ra(T, O, t1, 0.26, YELLOW_B, 3)
        self.play(Create(ra), run_time=0.6)
        self.bring_to_front(dT)
        segs = [_seg(t0, t1), _seg(O, T)] + _arc_segs(O, R, 0, TAU, 96) + _ra_segs(T, O, t1, 0.26)
        _labels_ok([lO, lT, lt], segs, "E30")
        cap = caption("OT ⊥ t", 38)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ E39  — the length of a chord

class E39_ChordLength(Board):
    """The chord AB of a circle of radius r subtends the central angle θ.
    The radius ON bisecting θ is the axis of the isosceles triangle OAB:
    folding OMA over it lands it on OMB, so the chord is halved at M, the
    two angles at M are equal and together straight — right angles — and
    each half of θ is θ/2. In the right triangle OMA the hypotenuse is r
    and the angle at O is θ/2, so the half-chord AM is r·sin(θ/2), and
    AB = 2r·sin(θ/2)."""

    def construct(self):
        R = 3.0
        O = np.array([0.0, 0.75, 0.0])
        th = 116 * DEGREES
        A = O + R * _dir(-PI / 2 - th / 2)
        B = O + R * _dir(-PI / 2 + th / 2)
        M = (A + B) / 2
        N = O + R * DOWN
        n = np.linalg.norm
        check(abs(_ang(O, A, B) - th) < 1e-9, "the central angle is θ")
        check(close(_refl(A, O, DOWN), B), "the fold over ON carries A onto B")
        check(abs(np.dot(M - O, B - A)) < 1e-9 and abs(_cross(M - O, N - O)) < 1e-9,
              "M is on ON, and OM ⊥ AB")
        check(abs(_ang(O, A, M) - th / 2) < 1e-9 and abs(n(A - M) - R * np.sin(th / 2)) < 1e-9,
              "angle θ/2 at O; AM = r sin(θ/2)")
        check(abs(n(B - A) - 2 * R * np.sin(th / 2)) < 1e-9, "AB = 2r sin(θ/2)")

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        dO = _dot(O)
        lO = tag("O", 28).move_to(O + 0.4 * UP)
        self.play(Create(circ), FadeIn(dO), FadeIn(lO), run_time=1.0)
        oa = Line(O, A, color=WHITE, stroke_width=3)
        ob = Line(O, B, color=WHITE, stroke_width=3)
        ab = Line(A, B, color=YELLOW_B, stroke_width=4)
        dA, dB = _dot(A), _dot(B)
        lA = tag("A", 28).move_to(A + 0.4 * _unit(A - O))
        lB = tag("B", 28).move_to(B + 0.4 * _unit(B - O))
        lr1 = _beside(tag("r", 28), O, A, np.array([-np.cos(th / 2), np.sin(th / 2), 0.0]), 0.14)
        lr2 = _beside(tag("r", 28), O, B, np.array([np.cos(th / 2), np.sin(th / 2), 0.0]), 0.14)
        self.play(Create(oa), Create(ob), FadeIn(lr1), FadeIn(lr2), run_time=0.8)
        self.play(Create(ab), FadeIn(dA), FadeIn(dB), FadeIn(lA), FadeIn(lB), run_time=0.8)
        aT = angle_arc(O, A, B, 0.62, GREEN_B, 4)
        lT = tag("θ", 30, GREEN_B).move_to(O + 1.0 * DOWN)
        self.play(Create(aT), FadeIn(lT), run_time=0.6)

        # the radius bisecting θ
        on = DashedLine(O, N, color=GREY_A, stroke_width=2.5, dash_length=0.09)
        dM = _dot(M, YELLOW_B)
        self.play(FadeOut(lT), run_time=0.3)           # θ's label sits on the bisector
        self.play(Create(on), FadeIn(dM), run_time=0.7)

        # fold OMA over it: it lands on OMB
        flap = mk([O, M, A], BLUE_D, 0.55, stroke_color=BLUE_B, stroke_width=3)
        self.add(flap)
        self.bring_to_front(dO, dM, lO)
        self.play(FadeIn(flap), run_time=0.5)
        self.play(_fold(flap, O, N), run_time=1.7)
        check(_same_vertices_ok(flap, [O, M, B]), "folded, OMA lies on OMB")
        a1 = angle_arc(O, A, M, 0.66, GREEN_B, 4)          # matching arcs: equal angles
        a2 = angle_arc(O, M, B, 0.66, GREEN_B, 4)
        l1 = tag("θ/2", 26, GREEN_B).move_to(O + 1.12 * _dir(-PI / 2 - th / 4))
        l2 = tag("θ/2", 26, GREEN_B).move_to(O + 1.12 * _dir(-PI / 2 + th / 4))
        raM = _ra(M, O, B, 0.22, YELLOW_B, 2.5)
        tk = VGroup(_ticks(A, M, 2, YELLOW_B), _ticks(M, B, 2, YELLOW_B))
        self.play(FadeOut(flap), FadeOut(aT), Create(a1), Create(a2), FadeIn(l1), FadeIn(l2),
                  Create(raM), FadeIn(tk), run_time=0.9)
        self.bring_to_front(dM)
        self.hold(0.4)

        # hypotenuse r, angle θ/2: each half-chord is r·sin(θ/2)
        h1 = Line(A, M, color=ORANGE, stroke_width=7)
        h2 = Line(M, B, color=ORANGE, stroke_width=7)
        ls1 = tag("r sin(θ/2)", 24, ORANGE).move_to((A + M) / 2 + 0.36 * DOWN)
        ls2 = tag("r sin(θ/2)", 24, ORANGE).move_to((M + B) / 2 + 0.36 * DOWN)
        self.play(Create(h1), oa.animate.set_stroke(width=5), FadeIn(ls1), run_time=0.8)
        self.play(Create(h2), ob.animate.set_stroke(width=5), FadeIn(ls2), run_time=0.8)
        self.bring_to_front(tk, dA, dB, dM)

        segs = (_arc_segs(O, R, 0, TAU, 120) + [_seg(O, A), _seg(O, B), _seg(A, B), _seg(O, N)]
                + _arc_segs(O, 0.66, -PI / 2 - th / 2, -PI / 2 + th / 2, 30)
                + _ra_segs(M, O, B, 0.22))
        _labels_ok([lO, lA, lB, lr1, lr2, l1, l2, ls1, ls2], segs, "E39")
        cap = caption("AB  =  2r sin(θ/2)", 38)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)


# ============================================ G12  — the half-angle formulas

class G12_HalfAngle(Board):
    """On the circle of radius 1, P at angle θ. The triangle VOP has two
    radii for sides, so its base angles at V and P are equal; slid along
    the diameter and turned about the midpoint of OP they fill θ, so each
    is θ/2. VPE is right-angled at P (angle in a semicircle), with
    hypotenuse 2: VP = 2 cos(θ/2), PE = 2 sin(θ/2). The foot M of P cuts the
    diameter into VM = 1 + cos θ and ME = 1 − cos θ. The angle θ/2 at V,
    turned a quarter turn, is the angle MPE (its sides PM, PE are square to
    VE, VP). Read in the right triangles VMP and PME: VM = VP·cos(θ/2) =
    2cos²(θ/2) and ME = PE·sin(θ/2) = 2sin²(θ/2)."""

    def construct(self):
        R = 3.3
        O = np.array([0.0, -0.75, 0.0])
        th = 70 * DEGREES
        hw = th / 2
        V, E = O + R * LEFT, O + R * RIGHT
        P = O + R * _dir(th)
        M = np.array([P[0], O[1], 0.0])
        K = (O + P) / 2
        n = np.linalg.norm
        check(abs(n(P - O) - R) < 1e-9, "OP = OV: VOP is isosceles")
        check(abs(_ang(V, E, P) - hw) < 1e-9 and abs(_ang(P, V, O) - hw) < 1e-9,
              "its base angles are θ/2 each")
        check(close(_dir(PI + hw), _unit(V - P)) and close(_dir(PI + th), _unit(O - P))
              and close(2 * K - P, O), "turned about the midpoint of OP, ∠VPO sits between "
              "the directions θ/2 and θ at O")
        check(abs(np.dot(V - P, E - P)) < 1e-9, "∠VPE = 90°")
        check(abs(n(P - V) - 2 * R * np.cos(hw)) < 1e-9 and abs(n(E - P) - 2 * R * np.sin(hw)) < 1e-9,
              "VP = 2cos(θ/2), PE = 2sin(θ/2)")
        check(close(_dir(-PI / 2), _unit(M - P)) and close(_dir(hw - PI / 2), _unit(E - P)),
              "θ/2 at V turned −90° is ∠MPE")
        check(abs(n(M - V) - n(P - V) * np.cos(hw)) < 1e-9
              and abs(n(M - V) - R * (1 + np.cos(th))) < 1e-9
              and abs(n(E - M) - n(E - P) * np.sin(hw)) < 1e-9
              and abs(n(E - M) - R * (1 - np.cos(th))) < 1e-9,
              "VM = VP·cos(θ/2) = 1 + cos θ, ME = PE·sin(θ/2) = 1 − cos θ")
        rw = 0.9

        # ---- the circle of radius 1 and P at angle θ
        arc = Arc(radius=R, start_angle=0, angle=PI, arc_center=O, color=GREY_B, stroke_width=3)
        diam = Line(V, E, color=GREY_A, stroke_width=3)
        dO = _dot(O)
        lV = tag("V", 28).move_to(V + 0.38 * LEFT)
        lE = tag("E", 28).move_to(E + 0.38 * RIGHT)
        lO = tag("O", 26).move_to(O + 0.4 * _dir(125 * DEGREES))
        self.play(Create(arc), Create(diam), FadeIn(dO), FadeIn(lV), FadeIn(lE), FadeIn(lO),
                  run_time=1.1)
        op = Line(O, P, color=WHITE, stroke_width=3)
        dP = _dot(P)
        lP = tag("P", 28).move_to(P + 0.4 * _dir(th))
        aT = angle_arc(O, E, P, 0.55, YELLOW_B, 4)
        lT = tag("θ", 28, YELLOW_B).move_to(O + 0.88 * _dir(0.74 * th))   # off the seam at θ/2
        self.play(Create(op), FadeIn(dP), FadeIn(lP), Create(aT), FadeIn(lT), run_time=0.9)

        # ---- the isosceles triangle VOP: its two base angles fill θ
        vp = Line(V, P, color=BLUE_B, stroke_width=4)
        tk = VGroup(_ticks(V, O, 1, WHITE, at=0.5), _ticks(O, P, 1, WHITE, at=0.5))
        aV = angle_arc(V, E, P, rw, GREEN_B, 4)
        aP = angle_arc(P, V, O, rw, GREEN_B, 4)
        self.play(Create(vp), FadeIn(tk), run_time=0.7)
        self.play(Create(aV), Create(aP), run_time=0.6)
        wV, wP = _wedge(V, E, P, rw, GREEN_D, 0.8), _wedge(P, V, O, rw, GREEN_D, 0.8)
        for w_ in (wV, wP):
            w_.set_stroke(WHITE, 1.5)                  # the seam between them stays visible
        piv = _dot(K, WHITE, 0.05)
        self.add(wV, wP)
        self.bring_to_front(lO, lT)
        self.play(FadeIn(wV), FadeIn(wP), FadeIn(piv), run_time=0.4)
        self.play(wV.animate.shift(O - V), Rotate(wP, angle=PI, about_point=K), run_time=1.8)
        check(close(wV.get_arc_center(), O, 1e-6) and close(wP.get_arc_center(), O, 1e-6),
              "both base angles land at O")
        self.bring_to_front(op, aT, lT, dO)
        self.hold(0.4)
        lV2 = tag("θ/2", 24, GREEN_B).move_to(V + (rw + 0.42) * _dir(hw / 2))
        self.play(FadeOut(wV), FadeOut(wP), FadeOut(piv), FadeOut(aP), FadeIn(lV2), run_time=0.6)

        # ---- VPE: a right angle at P, hypotenuse 2
        pe = Line(P, E, color=ORANGE, stroke_width=4)
        raP = _ra(P, V, E, 0.24, WHITE, 2.5)
        self.play(Create(pe), Create(raP), run_time=0.7)
        lc = tag("2 cos(θ/2)", 28, BLUE_B).move_to([-4.65, 2.45, 0.0])
        ls = tag("2 sin(θ/2)", 28, ORANGE).move_to([4.65, 2.45, 0.0])
        ldc = DashedLine(lc.get_bottom() + 0.14 * DOWN + 0.3 * RIGHT, (V + P) / 2,
                         color=BLUE_B, stroke_width=2, dash_length=0.07)
        lds = DashedLine(ls.get_bottom() + 0.14 * DOWN + 0.3 * LEFT, (P + E) / 2,
                         color=ORANGE, stroke_width=2, dash_length=0.07)
        self.play(FadeIn(lc), Create(ldc), FadeIn(ls), Create(lds), run_time=0.9)

        # ---- the foot M of P: 1 + cos θ and 1 − cos θ along the diameter
        pm = DashedLine(P, M, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        raM = _ra(M, P, E, 0.2, GREY_A, 2)
        dM = _dot(M)
        lM = tag("M", 26).move_to(M + np.array([0.5, 0.36, 0.0]))      # clear of the marks
        self.play(Create(pm), Create(raM), FadeIn(dM), FadeIn(lM), run_time=0.7)
        l1 = tag("1", 24).move_to([(V[0] + O[0]) / 2, O[1] - 0.32, 0])
        lcos = tag("cos θ", 24).move_to([(O[0] + M[0]) / 2, O[1] - 0.32, 0])
        yb = O[1] - 0.72
        bVM = Line([V[0], yb, 0], [M[0], yb, 0], color=BLUE_D, stroke_width=9)
        bME = Line([M[0], yb, 0], [E[0], yb, 0], color=ORANGE, stroke_width=9)
        gd = VGroup(*[DashedLine(X + 0.08 * DOWN, [X[0], yb - 0.1, 0], color=GREY_B,
                                 stroke_width=1.5, dash_length=0.06) for X in (V, M, E)])
        r1a = tag("1 + cos θ", 26, BLUE_B).move_to([(V[0] + M[0]) / 2, yb - 0.38, 0])
        r1b = tag("1 − cos θ", 26, ORANGE).move_to([(M[0] + E[0]) / 2, yb - 0.38, 0])
        self.play(FadeIn(l1), FadeIn(lcos), run_time=0.5)
        self.play(Create(gd), Create(bVM), Create(bME), FadeIn(r1a), FadeIn(r1b), run_time=0.9)

        # ---- θ/2 at V, turned a quarter turn, is the angle MPE
        w2 = _wedge(V, E, P, rw, GREEN_D, 0.8)
        self.add(w2)
        self.play(FadeIn(w2), run_time=0.3)
        self.play(_rigid(w2, _wedge_pts(V, E, P, rw),
                         [P, P + rw * _dir(-PI / 2), P + rw * _dir(hw - PI / 2)], turn=-PI / 2),
                  run_time=1.7)
        aP2 = angle_arc(P, M, E, rw, GREEN_B, 4)
        lP2 = tag("θ/2", 24, GREEN_B).move_to(P + (rw + 0.42) * _dir(hw / 2 - PI / 2))
        self.play(FadeOut(w2), Create(aP2), FadeIn(lP2), run_time=0.6)

        # ---- read the two right triangles
        tV = mk([V, M, P], BLUE_E, 0.35, stroke_width=0)
        self.add(tV)
        self.bring_to_back(tV)
        tV.set_fill(opacity=0)
        r2a = tag("= 2 cos(θ/2) · cos(θ/2)", 26, BLUE_B)
        r2a.move_to([V[0] + r2a.width / 2, yb - 0.86, 0])        # left-aligned with its bar
        self.play(tV.animate.set_fill(opacity=0.35), run_time=0.5)
        self.play(FadeIn(r2a), run_time=0.7)
        self.hold(0.3)
        tE = mk([P, M, E], ORANGE, 0.3, stroke_width=0)
        self.add(tE)
        self.bring_to_back(tE)
        tE.set_fill(opacity=0)
        r2b = tag("= 2 sin(θ/2) · sin(θ/2)", 26, ORANGE)
        r2b.move_to([M[0] + r2b.width / 2, yb - 0.86, 0])
        self.play(tE.animate.set_fill(opacity=0.3), run_time=0.5)
        self.play(FadeIn(r2b), run_time=0.7)

        segs = (_arc_segs(O, R, 0, PI, 100)
                + [_seg(V, E), _seg(O, P), _seg(V, P), _seg(P, E), _seg(P, M),
                   _seg(ldc.get_start(), ldc.get_end()), _seg(lds.get_start(), lds.get_end()),
                   _seg(bVM.get_start(), bME.get_end())]
                + [_seg(g.get_start(), g.get_end()) for g in gd]
                + _arc_segs(O, 0.55, 0, th, 20) + _arc_segs(V, rw, 0, hw, 12)
                + _arc_segs(P, rw, -PI / 2, hw - PI / 2, 12)
                + _ra_segs(P, V, E, 0.24) + _ra_segs(M, P, E, 0.2))
        labels = [lV, lE, lO, lP, lT, lV2, lc, ls, lM, l1, lcos, r1a, r1b, lP2, r2a, r2b]
        _labels_ok(labels, segs, "G12")
        cap = caption("cos²(θ/2)  =  (1 + cos θ)/2          sin²(θ/2)  =  (1 − cos θ)/2", 32)
        self.play(Write(cap))
        _final_ok(self, cap)
        self.hold(2.2)
