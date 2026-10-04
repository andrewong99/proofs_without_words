# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3b.py — proofs without words, 2D (manim):
#     C21 n² is one more than (n−1)(n+1)     C17 the Brahmagupta–Fibonacci identity
#     C22 ½ + ⅓ + ⅙ = 1                      C18 Sophie Germain's identity
#     C23 the difference of fourth powers    A16 the reciprocal Pythagorean theorem
#     A22 Pythagoras from the incircle       A24 two half-squares
#     A27 Pythagoras from Ptolemy            A26 the 120° law of cosines
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# stacked fractions are built from Text pieces and a bar.  Every move is
# rigid (a slide, a Rotate about a point, or a fold, which is a half-turn in
# space about a line of the plane) unless it is an announced shear (an apex
# sliding along a parallel to its fixed base) or an announced enlargement
# (a homothety about a marked point).  Every landing, tiling, length and
# angle the argument rests on is checked with check(...) before or right
# after it is drawn, and the closing frame is checked for labels that
# overlap or leave the safe area, so a wrong picture fails the render.


# ------------------------------------------------------------ geometry helpers

def _u(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _dir(t):
    return np.array([np.cos(t), np.sin(t), 0.0])


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _rot2(p, c, th):
    """The point p turned by th about c (in the plane), as a 3D point."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


def _L(p, q):
    return float(np.linalg.norm(to3(p) - to3(q)))


def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


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


def _landed(mob, pts, what, tol=1e-6):
    """Fail the render unless the polygon mob has exactly the vertices pts
    (screen points, same order): the move landed where the proof says."""
    v = mob.get_vertices()
    check(len(v) == len(pts) and all(close(a, to3(b), tol) for a, b in zip(v, pts)),
          what)


def _sweep_ok(pts, pivot, angle, what, keep=None, n=72):
    """Every corner of a piece turning by `angle` about `pivot` stays in the
    safe area on the way (and satisfies keep(point), if given)."""
    for i in range(n + 1):
        for p in pts:
            q = _rot2(p, pivot, angle * i / n)
            check(abs(q[0]) <= SAFE_X and SAFE_BOTTOM <= q[1] <= SAFE_TOP
                  and (keep is None or keep(q)), what)


# ------------------------------------------------------------ marks

def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _u(p - v), _u(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ra_segs(v, p, q, s):
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _u(p - v), _u(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


def _ticks(p, q, n=1, color=WHITE, size=0.12, width=2.5, at=0.5):
    """n short tick marks across the segment pq (equal-length marks)."""
    p, q = to3(p), to3(q)
    d = _u(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    g = VGroup()
    for k in range(n):
        o = c + d * 0.09 * (k - (n - 1) / 2)
        g.add(Line(o - nrm * size, o + nrm * size, color=color, stroke_width=width))
    return g


def _tick_segs(p, q, n=1, size=0.12, at=0.5):
    p, q = to3(p), to3(q)
    d = _u(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    return [(c + d * 0.09 * (k - (n - 1) / 2) - nrm * size,
             c + d * 0.09 * (k - (n - 1) / 2) + nrm * size) for k in range(n)]


def _dim(p, q, side, label, color=GREY_A, size=26, off=0.3, tick=0.09, gap=0.12,
         width=2.5):
    """Dimension bar beside the screen segment pq, `off` away towards `side`,
    with end ticks; the label (a string or a ready mobject) sits beyond it.
    It spans the whole segment, so it names everything along it."""
    p, q = to3(p), to3(q)
    n = _u(side)
    a, b = p + off * n, q + off * n
    bar = VGroup(Line(a, b, color=color, stroke_width=width),
                 Line(a - tick * n, a + tick * n, color=color, stroke_width=width),
                 Line(b - tick * n, b + tick * n, color=color, stroke_width=width))
    t = label if isinstance(label, Mobject) else tag(label, size, color)
    half = abs(n[0]) * t.width / 2 + abs(n[1]) * t.height / 2
    t.move_to((a + b) / 2 + n * (gap + half))
    return VGroup(bar, t)


def _dim_segs(d):
    """The strokes of a _dim bar, as screen segments (for label checks)."""
    return [(ln.get_start(), ln.get_end()) for ln in d[0]]


def _poly_segs(pts):
    pts = [to3(p) for p in pts]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _arc_segs(c, r, a0, a1, n=24):
    c = to3(c)
    pts = [c + r * _dir(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _angle_segs(v, p, q, r, n=12):
    """The arc that angle_arc(v, p, q, r) draws, as segments."""
    v = to3(v)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return _arc_segs(v, r, a1, a1 + span, n)


def _circle_segs(c, r, n=120):
    return _arc_segs(c, r, 0.0, TAU, n)


# ------------------------------------------------------------ text helpers

def _lbl(s, size=26, color=WHITE, bg=0.0):
    """tag() with an optional dark backing box for text sitting on fills."""
    t = tag(s, size, color)
    if bg > 0:
        box = BackgroundRectangle(t, color=BLACK, fill_opacity=bg, buff=0.06)
        return VGroup(box, t)
    return t


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on
    the side the vector `side` points to, with its box clear of the line
    by `gap` whatever the slope."""
    p, q = to3(p), to3(q)
    d = _u(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


def _on_base(s, size, color):
    """Text s with its baseline at y = 0, as Pango sets it (a reference 'M'
    is laid out in front of it, measured and removed)."""
    t = Text("M" + s, font_size=size, color=color)
    m = t.submobjects[0]
    base = m.get_bottom()[1]
    t.remove(m)
    t.shift(np.array([0.0, -base, 0.0]))
    t.text = s.strip()
    return t


def _frac(num, den, size=30, color=WHITE, gap=0.07, width=2.5):
    """A stacked fraction num/den built from Text pieces and a bar."""
    a = Text(num, font_size=size, color=color)
    b = Text(den, font_size=size, color=color)
    w = max(a.width, b.width) + 0.12 * size / 30
    bar = Line(LEFT * w / 2, RIGHT * w / 2, color=color, stroke_width=width)
    a.next_to(bar, UP, buff=gap)
    b.next_to(bar, DOWN, buff=gap)
    return VGroup(a, bar, b)


def _row(*parts, buff=0.16):
    """Parts set side by side, their centres on one horizontal line."""
    return VGroup(*parts).arrange(RIGHT, buff=buff)


def _cap(group, buff=0.3):
    """Place a composite formula where caption() puts its Text: centred,
    along the bottom edge."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.set_x(0.0)
    group.to_edge(DOWN, buff=buff)
    return group


# ------------------------------------------------------------ label hygiene

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _hit(m, segs, pad=0.05):
    """Index of the first screen segment passing through m's box (grown by
    pad), or None."""
    x0, x1, y0, y1 = _box(m, pad)
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.015) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _overlap(a, b, gap=0.04):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _inside(m, tol=0.02):
    return (m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
            and m.get_bottom()[1] >= SAFE_BOTTOM - tol
            and m.get_top()[1] <= SAFE_TOP + tol)


def _name(m):
    t = getattr(m, "text", None)
    if t:
        return t
    txt = [s.text for s in m.get_family() if isinstance(s, Text)]
    return "/".join(txt) if txt else type(m).__name__


def _labels_ok(labels, segs, what, pad=0.05, gap=0.04):
    """Every label inside the safe area, clear of every screen segment in
    segs (lines, ticks, arcs, marks) and of every other label."""
    for m in labels:
        bx = ", ".join(f"{v:.2f}" for v in _box(m))
        check(_inside(m), f"{what}: '{_name(m)}' [{bx}] inside the safe area")
        i = _hit(m, segs, pad)
        sg = "" if i is None else (f" {np.round(to3(segs[i][0])[:2], 2)}-"
                                   f"{np.round(to3(segs[i][1])[:2], 2)}")
        check(i is None, f"{what}: '{_name(m)}' [{bx}] clear of the lines "
              f"(hits #{i}{sg})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption inside its band, and no two pieces of text overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if not m.has_points() and not m.submobjects:
            continue
        bx = ", ".join(f"{v:.2f}" for v in _box(m))
        check(_inside(m), f"{_name(m)} [{bx}] inside the safe area")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(not _overlap(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# ------------------------------------------------------------ motions

def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centroid while the
    centroid slides straight — carrying the screen points src onto dst
    (same order). Fails the render unless one rotation + translation lands
    every point. `turn` (radians) picks the direction for ±180°."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = float(np.arctan2(*(src[1] - src[0])[1::-1]))
        a1 = float(np.arctan2(*(dst[1] - dst[0])[1::-1]))
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    check(all(close(m1 + _rot2(p, m0, turn) - m0, q, 1e-6) for p, q in zip(src, dst)),
          "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _homothety(mob, centre, factor, **kw):
    """Enlargement (or reduction) by `factor` about the screen point centre,
    geometric in time, exact at the end."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_u(q - p), about_point=p, **kw)


def _turn_arrow(pivot, a0, a1, r=0.6, color=YELLOW_B, width=4):
    """A curved arrow about pivot from direction a0 to a1 (radians)."""
    arc = Arc(radius=r, start_angle=a0, angle=a1 - a0, arc_center=to3(pivot),
              color=color, stroke_width=width)
    arc.add_tip(tip_length=0.16, tip_width=0.16)
    return arc


# =================================================================== C21

class C21_OneMoreThanRectangle(Board):
    """The n-square of cells (drawn for n = 5). Its last column turns a
    quarter-turn about the top right corner of the rest and slides along the
    top: n − 1 of its cells complete an (n − 1) × (n + 1) rectangle, one
    cell is left over:  n² = (n − 1)(n + 1) + 1."""

    def construct(self):
        n = 5
        piv = np.array([n - 1.0, float(n)])          # top right corner of the rest
        block = [(i, j) for i in range(n - 1) for j in range(n)]
        column = [(n - 1, j) for j in range(n)]
        slide = np.array([-(n - 1.0), 0.0])

        def land(ij):
            """Where the column cell ij ends: turned about piv, then slid."""
            c = _rot2((ij[0] + 0.5, ij[1] + 0.5), piv, PI / 2)[:2] + slide
            return (int(round(c[0] - 0.5)), int(round(c[1] - 0.5)))

        landed = [land(ij) for ij in column]
        check(sorted(landed) == [(i, n) for i in range(n)],
              "the column lands as a row along the top")
        rect = {(i, j) for i in range(n - 1) for j in range(n + 1)}
        pieces = block + [c for c in landed if c in rect]
        check(len(pieces) == len(set(pieces)) and set(pieces) == rect,
              "the block and n − 1 moved cells tile the (n−1) × (n+1) rectangle")
        extra = [c for c in landed if c not in rect]
        check(extra == [(n - 1, n)] and land(column[0]) == (n - 1, n),
              "one cell (the bottom one of the column) is left over")
        check(n * n == (n - 1) * (n + 1) + 1, "n² = (n−1)(n+1) + 1")

        # the box is centred on the finished figure (x from −1.4 to n) and
        # still holds the swing of the column (out to x = 2n − 1 + 0.1)
        F = Frame(n - 1.4 - (2 * n - 1 + 0.35), 2 * n - 1 + 0.35, -1.05, n + 1 + 0.3)
        P, k = F.P, F.k
        O = P((0, 0))
        corners = [(n - 1, 0), (n, 0), (n, n), (n - 1, n)]
        # the quarter-turn sweeps only the region right of the rest
        _sweep_ok([P(c) for c in corners], P(piv), PI / 2,
                  "the turning column stays on screen and right of the rest",
                  keep=lambda q: q[0] >= P(piv)[0] - 1e-9)

        blk = VGroup(*[cell(i, j, k, O, BLUE_D) for (i, j) in block])
        col = VGroup(*[cell(i, j, k, O, BLUE_D) for (i, j) in column])
        d_bot = _dim(P((0, 0)), P((n, 0)), DOWN, "n", size=30)
        d_lft = _dim(P((0, 0)), P((0, n)), LEFT, "n", size=30)
        self.play(LaggedStart(*[FadeIn(c) for c in [*blk, *col]], lag_ratio=0.03),
                  run_time=1.4)
        self.play(FadeIn(d_bot), FadeIn(d_lft), run_time=0.6)
        self.hold(0.5)

        # ---- the last column, turned a quarter-turn about the corner
        self.play(col.animate.set_fill(ORANGE), run_time=0.6)
        dot = Dot(P(piv), radius=0.07, color=YELLOW_B)
        # the path of the column's far end: a quarter circle about the corner
        arr = _turn_arrow(P(piv), -PI / 2 + 0.3, -0.06, (n - 0.5) * k, YELLOW_B, 3)
        self.play(FadeIn(dot), Create(arr), FadeOut(d_bot), run_time=0.6)
        self.bring_to_front(col, dot)
        self.play(Rotate(col, angle=PI / 2, about_point=P(piv)), run_time=1.8)
        for c, ij in zip(col, column):
            m = _rot2((ij[0] + 0.5, ij[1] + 0.5), piv, PI / 2)
            check(close(c.get_center(), P(m), 1e-6), "the turn lays the column flat")
        self.play(FadeOut(dot), FadeOut(arr), run_time=0.3)

        # ---- slid along the top of the rest
        self.play(col.animate.shift(k * to3(slide)), run_time=1.4)
        for c, ij in zip(col, landed):
            check(close(c.get_center(), P((ij[0] + 0.5, ij[1] + 0.5)), 1e-6),
                  "each cell lands on the top row")
        self.hold(0.3)

        # ---- an (n−1) × (n+1) rectangle and one cell over
        frame = Polygon(*[P(p) for p in _rect(0, 0, n - 1, n + 1)],
                        stroke_color=YELLOW_B, stroke_width=5)
        one = col[0]
        l1 = tag("1", 32).move_to(one.get_center())
        D_bot = _dim(P((0, 0)), P((n - 1, 0)), DOWN, "n − 1", YELLOW_B, 30)
        D_lft = _dim(P((0, 0)), P((0, n + 1)), LEFT, "n + 1", YELLOW_B, 30)
        self.play(FadeOut(d_lft), run_time=0.3)
        self.play(Create(frame), one.animate.set_fill(RED_D, opacity=0.9),
                  run_time=0.9)
        self.play(FadeIn(D_bot), FadeIn(D_lft), FadeIn(l1), run_time=0.7)

        segs = ([s for c in [*blk, *col] for s in _poly_segs(c.get_vertices())]
                + _dim_segs(D_bot) + _dim_segs(D_lft))
        _labels_ok([D_bot[1], D_lft[1]], segs, "C21 brackets")
        check(_hit(l1, [s for c in [*blk, *col[1:]] for s in _poly_segs(c.get_vertices())]
                   + _poly_segs(one.get_vertices()), 0.02) is None,
              "C21: the '1' sits inside its cell")
        cap = caption("n²  =  (n − 1)(n + 1)  +  1", 36)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# =================================================================== C17

class C17_BrahmaguptaFibonacci(Board):
    """The right triangle with legs a, b and hypotenuse r (r² = a² + b²).
    A copy enlarged by c and a copy enlarged by d and turned a quarter-turn
    stand on one vertical line, the second on top of the first. At their
    common vertex U the angles β (of the c-copy) and α (of the d-copy) lie
    along that straight line, and α + β = 90°, so their hypotenuses cr, dr
    meet at a right angle: OW² = (cr)² + (dr)² = (c² + d²)(a² + b²).  The
    perpendicular from W to the base cuts off the right triangle OZW with
    legs OZ = ac − bd (ZXYW is a rectangle, so ZX = YW = bd) and
    ZW = XY = bc + ad:  OW² = (ac − bd)² + (ad + bc)²."""

    def construct(self):
        a, b, c, d = 2.1, 1.2, 1.45, 0.85
        r = float(np.hypot(a, b))
        k = 1.35
        O = np.array([-5.6, -2.0, 0.0])

        def S(p):
            return O + k * to3(p)

        X, U = S((a * c, 0)), S((a * c, b * c))
        Y, W = S((a * c, b * c + a * d)), S((a * c - b * d, b * c + a * d))
        Z = S((a * c - b * d, 0))
        O0 = np.array([1.55, 1.8, 0.0])
        X0, U0 = O0 + k * to3((a, 0)), O0 + k * to3((a, b))
        al, be = float(np.arctan2(b, a)), float(np.arctan2(a, b))

        # ---- the claims
        for (P1, P2, P3), f, what in (((O, X, U), c, "the c-copy"),
                                      ((U, Y, W), d, "the d-copy")):
            check(abs(_L(P1, P2) - f * a * k) < 1e-9 and abs(_L(P2, P3) - f * b * k) < 1e-9
                  and abs(_L(P1, P3) - f * r * k) < 1e-9
                  and abs(np.dot(P1 - P2, P3 - P2)) < 1e-9,
                  f"{what}: legs {f}a, {f}b, hypotenuse {f}r, right angle")
        check(abs(_cross(Y - X, U - X)) < 1e-9 and 0 < np.dot(U - X, Y - X) < _L(X, Y) ** 2,
              "X, U, Y on one line, U between")
        check(abs(_ang(U, X, O) - be) < 1e-9 and abs(_ang(U, Y, W) - al) < 1e-9
              and abs(al + be - PI / 2) < 1e-12, "β and α at U; α + β = 90°")
        check(abs(np.dot(O - U, W - U)) < 1e-9, "the hypotenuses meet at a right angle")
        check(abs(np.dot(W - Z, X - O)) < 1e-9 and abs(_L(O, Z) - (a * c - b * d) * k) < 1e-9
              and abs(_L(Z, W) - (a * d + b * c) * k) < 1e-9 and a * c > b * d,
              "OZW: legs ac − bd and ad + bc")
        check(close(X - Z, Y - W) and abs(_L(Z, X) - b * d * k) < 1e-9
              and abs(np.dot(X - Z, W - Z)) < 1e-9, "ZXYW is a rectangle")
        ow2 = (_L(O, W) / k) ** 2
        check(abs(ow2 - (c * c + d * d) * (a * a + b * b)) < 1e-9
              and abs(ow2 - ((a * c - b * d) ** 2 + (a * d + b * c) ** 2)) < 1e-9,
              "OW² both ways")

        # ---- the triangle a, b, r
        T0 = mk([O0, X0, U0], GREY_D, 0.6)
        ra0 = _ra(X0, O0, U0, 0.18)
        la = tag("a", 28).next_to((O0 + X0) / 2, DOWN, buff=0.16)
        lb = tag("b", 28).next_to((X0 + U0) / 2, RIGHT, buff=0.16)
        lr = _beside(tag("r", 28), O0, U0, UP + LEFT, 0.1)
        aA = angle_arc(O0, X0, U0, 0.55, ORANGE, 4)
        aB = angle_arc(U0, O0, X0, 0.42, PINK, 4)
        lA = tag("α", 24, ORANGE).move_to(O0 + 0.85 * angle_mid_dir(O0, X0, U0))
        lB = tag("β", 24, PINK).move_to(U0 + 0.72 * angle_mid_dir(U0, O0, X0))
        lr2 = tag("r²  =  a² + b²", 28).move_to([3.0, 0.95, 0])
        self.play(FadeIn(T0), Create(ra0), FadeIn(la), FadeIn(lb), FadeIn(lr), run_time=1.0)
        self.play(Create(aA), Create(aB), FadeIn(lA), FadeIn(lB), FadeIn(lr2), run_time=0.8)
        self.hold(0.4)

        # ---- a copy enlarged by c
        tc = mk([O0, X0, U0], BLUE_D)
        self.add(tc)
        self.play(_rigid(tc, [O0, X0, U0], [O, S((a, 0)), S((a, b))], turn=0.0),
                  run_time=1.4)
        fc = tag("× c", 28, BLUE_B).move_to(S((a * c * 0.5, b * c * 0.85)) + 0.3 * LEFT)
        dotO = Dot(O, radius=0.06, color=BLUE_B)
        self.play(FadeIn(fc), FadeIn(dotO), run_time=0.4)
        self.play(_homothety(tc, O, c), run_time=1.3)
        _landed(tc, [O, X, U], "the c-copy lands on O, X, U")
        d_ac = _dim(O, X, DOWN, "ac", WHITE, 28, off=0.28)
        lbc = tag("bc", 28).next_to((X + U) / 2, RIGHT, buff=0.16)
        lcr = _beside(tag("cr", 28), O, U, DOWN + RIGHT, 0.12, at=0.36)
        ra_X = _ra(X, O, U, 0.18)
        self.play(FadeOut(fc), FadeOut(dotO), FadeIn(d_ac), FadeIn(lbc), FadeIn(lcr),
                  Create(ra_X), run_time=0.7)

        # ---- a copy turned a quarter-turn and enlarged by d
        td = mk([O0, X0, U0], TEAL_D)
        self.add(td)
        self.play(_rigid(td, [O0, X0, U0], [U, U + k * to3((0, a)), U + k * to3((-b, a))],
                         turn=PI / 2), run_time=1.6)
        fd = tag("× d", 28, TEAL_B).move_to(U + k * to3((-b * 0.95, a * 0.3)))
        dotU = Dot(U, radius=0.06, color=TEAL_B)
        self.play(FadeIn(fd), FadeIn(dotU), run_time=0.4)
        self.play(_homothety(td, U, d), run_time=1.2)
        _landed(td, [U, Y, W], "the d-copy lands on U, Y, W")
        lad = tag("ad", 28).next_to((U + Y) / 2, RIGHT, buff=0.16)
        lbd = tag("bd", 28).next_to((Y + W) / 2, UP, buff=0.16)
        ldr = _beside(tag("dr", 28), U, W, RIGHT + UP, 0.1, at=0.6)
        ra_Y = _ra(Y, U, W, 0.18)
        self.play(FadeOut(fd), FadeOut(dotU), FadeIn(lad), FadeIn(lbd), FadeIn(ldr),
                  Create(ra_Y), run_time=0.7)
        self.hold(0.3)

        # ---- at U: β + (?) + α along a straight line, and α + β = 90°
        cB = angle_arc(U, X, O, 0.42, PINK, 4)
        cA = angle_arc(U, Y, W, 0.55, ORANGE, 4)
        mB = tag("β", 24, PINK).move_to(U + 0.72 * angle_mid_dir(U, X, O))
        mA = tag("α", 24, ORANGE).move_to(U + 0.86 * angle_mid_dir(U, Y, W))
        self.play(TransformFromCopy(aB, cB), TransformFromCopy(aA, cA),
                  FadeIn(mB), FadeIn(mA), run_time=1.2)
        raU = _ra(U, O, W, 0.24, YELLOW_B, 3)
        self.play(Create(raU), run_time=0.6)
        self.hold(0.5)

        # ---- OW: hypotenuse of OUW (legs cr, dr)
        OUW = mk([O, U, W], GOLD_D, 0.62, stroke_width=0)
        ow = Line(O, W, color=YELLOW_B, stroke_width=5)
        lO = tag("O", 26).move_to(O + 0.33 * LEFT)
        lW = tag("W", 26).move_to(W + 0.3 * LEFT + 0.12 * UP)
        self.add(OUW)
        self.bring_to_back(OUW)
        self.play(FadeOut(cB), FadeOut(cA), FadeOut(mB), FadeOut(mA), FadeIn(OUW),
                  Create(ow), FadeIn(lO), FadeIn(lW), run_time=1.0)
        x_l = 1.15                                  # left edge of the readouts
        R1 = tag("OW² = (cr)² + (dr)²", 26)
        R1.move_to([x_l + R1.width / 2, -0.3, 0])
        R2 = tag("= (c² + d²)(a² + b²)", 26, YELLOW_B)
        R2.next_to(R1, DOWN, buff=0.24)
        R2.shift((R1[3].get_left()[0] - R2[0].get_left()[0]) * RIGHT)   # '=' under '='
        self.play(FadeIn(R1), run_time=0.7)
        self.play(FadeIn(R2), run_time=0.7)
        self.hold(0.4)

        # ---- the perpendicular WZ: OZW has legs ac − bd and ad + bc
        wz = DashedLine(W, Z, color=WHITE, stroke_width=3, dash_length=0.1)
        raZ = _ra(Z, O, W, 0.2, WHITE, 2.5)
        OZW = Polygon(O, Z, W, stroke_color=RED_B, stroke_width=5)
        self.play(Create(wz), Create(raZ), run_time=0.8)
        d_oz = _dim(O, Z + 0.03 * LEFT, DOWN, "ac − bd", RED_B, 28, off=0.28)
        d_zx = _dim(Z + 0.03 * RIGHT, X, DOWN, "bd", WHITE, 28, off=0.28)
        d_xy = _dim(X, Y, RIGHT, "ad + bc", RED_B, 28, off=0.82)
        self.play(FadeOut(d_ac), FadeIn(d_oz), FadeIn(d_zx), FadeIn(d_xy), run_time=0.8)
        self.play(Create(OZW), run_time=1.0)
        R3 = tag("OW² = (ac − bd)² + (ad + bc)²", 26, RED_B)
        R3.move_to([x_l + R3.width / 2, -1.75, 0])
        self.play(FadeIn(R3), run_time=0.7)

        segs = (_poly_segs([O, X, Y, W]) + [(O, U), (U, W), (W, Z), (O0, X0), (X0, U0),
                                            (U0, O0)]
                + _ra_segs(X, O, U, 0.18) + _ra_segs(Y, U, W, 0.18)
                + _ra_segs(U, O, W, 0.24) + _ra_segs(Z, O, W, 0.2)
                + _ra_segs(X0, O0, U0, 0.18) + _angle_segs(O0, X0, U0, 0.55)
                + _angle_segs(U0, O0, X0, 0.42)
                + _dim_segs(d_oz) + _dim_segs(d_zx) + _dim_segs(d_xy))
        labels = [la, lb, lr, lA, lB, lr2, lbc, lcr, lad, lbd, ldr, lO, lW,
                  d_oz[1], d_zx[1], d_xy[1], R1, R2, R3]
        _labels_ok(labels, segs, "C17")
        cap = caption("(a² + b²)(c² + d²)  =  (ac − bd)² + (ad + bc)²", 34)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# =================================================================== C22

class C22_EgyptianFractions(Board):
    """A rectangle (the whole, 1). Its midline halves it: the top half,
    slid down, lands exactly on the bottom half, so it is ½. Two cuts at
    the thirds of the width make three equal columns, each ⅓. The bottom
    half is cut into two thirds and one third
    of its width: the larger piece, turned a quarter-turn, is exactly one
    column (⅓ of the whole); the small piece and a copy slid up fill one
    column, so it is half of ⅓, i.e. ⅙.  The three pieces fill the whole:
    ½ + ⅓ + ⅙ = 1."""

    def construct(self):
        W, H = 3.0, 2.0
        whole = _rect(0, 0, W, H)
        top = _rect(0, 1, 3, 1)
        bot = _rect(0, 0, 3, 1)
        dom = _rect(0, 0, 2, 1)               # two thirds of the bottom half
        sml = _rect(2, 0, 1, 1)               # one third of the bottom half
        col3 = _rect(2, 0, 1, 2)              # the last of three equal columns
        ctr = (1.5, 1.0)
        F = Frame(-0.6, W + 0.6, -0.5, H + 0.5)
        P, k = F.P, F.k

        def quarter(p):                       # clockwise about (2, 0)
            return (2 + p[1], 2 - p[0])

        check(_same_poly([(x, y - 1) for (x, y) in top], bot),
              "the top half slid down is the bottom half")
        check(_tiles_exactly([top, bot], whole), "two halves tile the whole")
        check(_tiles_exactly([_rect(i, 0, 1, 2) for i in range(3)], whole),
              "three equal columns tile the whole")
        check(_same_poly([quarter(p) for p in dom], col3),
              "the two-thirds piece, turned a quarter-turn, is one column")
        check(_tiles_exactly([sml, _rect(2, 1, 1, 1)], col3)
              and _same_poly([(x, y + 1) for (x, y) in sml], _rect(2, 1, 1, 1)),
              "the small piece and a copy slid up fill one column")
        check(_tiles_exactly([top, dom, sml], whole), "the three pieces tile the whole")
        check(abs(area(top) - area(whole) / 2) < 1e-12
              and abs(area(dom) - area(whole) / 3) < 1e-12
              and abs(area(sml) - area(whole) / 6) < 1e-12, "½, ⅓, ⅙ of the whole")

        # ---- the whole
        frame = Polygon(*[P(p) for p in whole], stroke_color=WHITE, stroke_width=5)
        base = F.poly(whole, GREY_D, 0.55, stroke_width=0)
        l1 = tag("1", 48).move_to(P(ctr))
        self.play(FadeIn(base), Create(frame), FadeIn(l1), run_time=1.0)
        self.hold(0.4)

        # ---- halve it: the halves are congruent (one slides onto the other)
        mid = Line(P((0, 1)), P((W, 1)), color=WHITE, stroke_width=4)
        pTop = F.poly(top, BLUE_D)
        self.play(Create(mid), FadeOut(l1), run_time=0.6)
        self.add(pTop)
        self.bring_to_front(mid, frame)
        self.play(FadeIn(pTop), run_time=0.5)
        ghost = F.poly(top, BLUE_B, 0.3, stroke_color=BLUE_A, stroke_width=3)
        self.add(ghost)
        self.play(ghost.animate.shift(k * DOWN), run_time=1.2)
        _landed(ghost, [P((x, y - 1)) for (x, y) in top],
                "the top half lands on the bottom half")
        f2 = _frac("1", "2", 40).move_to(P((1.5, 1.5)))
        self.play(FadeOut(ghost), FadeIn(f2), run_time=0.6)
        self.hold(0.3)

        # ---- thirds: three equal columns
        thirds = VGroup(*[DashedLine(P((x, -0.02)), P((x, H + 0.02)), color=GREY_A,
                                     stroke_width=3, dash_length=0.1) for x in (1, 2)])
        tks = VGroup(*[_ticks(P((x, 0)), P((x + 1, 0)), 1, GREY_A, 0.13) for x in range(3)],
                     *[_ticks(P((x, H)), P((x + 1, H)), 1, GREY_A, 0.13) for x in range(3)])
        self.play(Create(thirds), FadeIn(tks), run_time=0.9)

        # ---- the bottom half: two thirds and one third of it
        cut = Line(P((2, 0)), P((2, 1)), color=WHITE, stroke_width=4)
        pDom, pSml = F.poly(dom, TEAL_D), F.poly(sml, ORANGE)
        self.add(pDom, pSml)
        self.bring_to_front(thirds, cut, mid, frame, f2)
        self.play(FadeIn(pDom), FadeIn(pSml), Create(cut), run_time=0.8)
        self.hold(0.3)

        # ---- the two-thirds piece is one column: ⅓
        g2 = F.poly(dom, TEAL_B, 0.3, stroke_color=TEAL_A, stroke_width=3)
        piv = Dot(P((2, 0)), radius=0.07, color=YELLOW_B)
        hl = Polygon(*[P(p) for p in col3], stroke_color=YELLOW_B, stroke_width=5)
        self.add(g2)
        self.play(FadeIn(piv), run_time=0.3)
        self.play(Rotate(g2, angle=-PI / 2, about_point=P((2, 0))), run_time=1.5)
        _landed(g2, [P(quarter(p)) for p in dom], "the turned piece covers one column")
        self.play(Create(hl), run_time=0.5)
        f3 = _frac("1", "3", 40).move_to(P((1.0, 0.5)))
        self.play(FadeOut(g2), FadeOut(piv), FadeOut(thirds), run_time=0.5)
        self.play(FadeIn(f3), run_time=0.5)

        # ---- the one-third piece is half a column: ⅙
        g3 = F.poly(sml, ORANGE, 0.45, stroke_color=YELLOW_A, stroke_width=3)
        self.add(g3)
        self.play(g3.animate.shift(k * UP), run_time=1.1)
        _landed(g3, [P((x, y + 1)) for (x, y) in sml], "the slid copy fills the column")
        f6 = _frac("1", "6", 40).move_to(P((2.5, 0.5)))
        self.play(FadeIn(f6), run_time=0.5)
        self.hold(0.4)
        self.play(FadeOut(g3), FadeOut(hl), FadeOut(tks), run_time=0.7)

        # labels sit on their own pieces, clear of the edges
        segs = (_poly_segs([P(p) for p in whole]) + [(P((0, 1)), P((W, 1))),
                                                       (P((2, 0)), P((2, 1)))])
        _labels_ok([f2, f3, f6], segs, "C22", pad=0.1)
        for f, piece in ((f2, top), (f3, dom), (f6, sml)):
            check(_pip(F.P(np.mean(np.array(piece, float), axis=0))[:2],
                       [P(p)[:2] for p in piece])
                  and all(_pip(q, [P(p)[:2] for p in piece])
                          for q in (f.get_corner(UL)[:2], f.get_corner(DR)[:2])),
                  "each fraction sits inside its piece")
        plus = lambda s: tag(s, 34, YELLOW_B)
        cap = _cap(_row(_frac("1", "2", 30, YELLOW_B), plus("+"),
                        _frac("1", "3", 30, YELLOW_B), plus("+"),
                        _frac("1", "6", 30, YELLOW_B), plus("="), tag("1", 34, YELLOW_B),
                        buff=0.24), buff=0.2)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# =================================================================== C18

class C18_SophieGermain(Board):
    """Left: a⁴ (a square of side a²) and 4b⁴ (four b⁴-squares, one square of
    side 2b²) in opposite corners of the square of side S = a² + 2b²; two
    a² × 2b² rectangles, 4a²b² in all, complete it.  Right: the same
    square with a square of side 2ab — area (2ab)² = 4a²b², the same as the
    two rectangles — in a corner instead.  So what is left on the right, an
    L-shaped gnomon, has area a⁴ + 4b⁴.  The gnomon is a difference of two
    squares: its upper arm slides across and turns a quarter-turn down
    beside the lower one, making the rectangle (S + 2ab) × (S − 2ab):
        a⁴ + 4b⁴ = (a² + 2b² + 2ab)(a² + 2b² − 2ab).
    (The equality 2·a²·2b² = (2ab)² is the only step read from the area
    labels; the rest is cut and moved.)"""

    def construct(self):
        a, b = 1.5, 0.8
        A2, B2 = a * a, b * b
        S, t = A2 + 2 * B2, 2 * a * b
        k = 0.95
        OL = np.array([-6.5, -2.35, 0.0])
        OR = np.array([OL[0] + S * k + 0.6, OL[1], 0.0])

        def PL(p):
            return OL + k * to3(p)

        def PR(p):
            return OR + k * to3(p)

        sqA = _rect(0, 0, A2, A2)
        sqB = _rect(A2, A2, 2 * B2, 2 * B2)
        smalls = [_rect(A2 + i * B2, A2 + j * B2, B2, B2) for i in range(2) for j in range(2)]
        r1, r2 = _rect(A2, 0, 2 * B2, A2), _rect(0, A2, A2, 2 * B2)
        big = _rect(0, 0, S, S)
        corner = _rect(S - t, S - t, t, t)
        Bst = _rect(0, 0, S, S - t)                      # lower arm of the gnomon
        Lst = _rect(0, S - t, S - t, t)                  # upper arm
        Lsl = [(x + S, y) for (x, y) in Lst]             # slid across
        piv = (S, S - t)

        def turn(p):                                     # clockwise about piv
            return (piv[0] + (p[1] - piv[1]), piv[1] - (p[0] - piv[0]))

        Lt = [turn(p) for p in Lsl]
        final = _rect(0, 0, S + t, S - t)
        check(_tiles_exactly(smalls, sqB) and _tiles_exactly([sqA, sqB, r1, r2], big),
              "a⁴, 4b⁴ and two a² × 2b² rectangles tile the square on a² + 2b²")
        check(abs(2 * (A2 * 2 * B2) - t * t) < 1e-12, "2·a²·2b² = (2ab)²")
        check(t < S, "the 2ab-square fits in a corner")
        check(_tiles_exactly([corner, Bst, Lst], big), "corner square + gnomon = the square")
        check(_same_poly(Lt, _rect(S, 0, t, S - t)), "the upper arm lands beside the lower")
        check(_tiles_exactly([Bst, Lt], final),
              "the two arms tile the (S+2ab) × (S−2ab) rectangle")
        check(abs((S + t) * (S - t) - (a ** 4 + 4 * b ** 4)) < 1e-12,
              "a⁴ + 4b⁴ = (S + 2ab)(S − 2ab)")
        _sweep_ok([PR(p) for p in Lsl], PR(piv), -PI / 2,
                  "the turning arm stays on screen, right of the lower arm",
                  keep=lambda q: q[0] >= PR(piv)[0] - 1e-9)

        # ---- left: a⁴ and 4b⁴ in opposite corners of the (a² + 2b²)-square
        ghostL = DashedVMobject(Polygon(*[PL(p) for p in big], stroke_color=GREY_B,
                                        stroke_width=2.5), num_dashes=60)
        mA = mk([PL(p) for p in sqA], BLUE_D)
        lA4 = tag("a⁴", 34).move_to(PL((A2 / 2, A2 / 2)))
        dA = _dim(PL((0, 0)), PL((A2 - 0.02, 0)), DOWN, "a²", GREY_A, 26, off=0.25)
        self.play(Create(ghostL), FadeIn(mA), FadeIn(lA4), FadeIn(dA), run_time=1.2)
        mSm = VGroup(*[mk([PL(p) for p in q], TEAL_D) for q in smalls])
        lSm = VGroup(*[tag("b⁴", 18).move_to(PL((q[0][0] + B2 / 2, q[0][1] + B2 / 2)))
                       for q in smalls])
        dB = _dim(PL((A2 + 0.02, 0)), PL((S, 0)), DOWN, "2b²", GREY_A, 26, off=0.25)
        self.play(LaggedStart(*[FadeIn(m) for m in mSm], lag_ratio=0.2), FadeIn(lSm),
                  FadeIn(dB), run_time=1.2)
        mB = mk([PL(p) for p in sqB], TEAL_D)
        l4B = tag("4b⁴", 26).move_to(PL((A2 + B2, A2 + B2)))
        self.play(FadeIn(mB), FadeOut(mSm), FadeOut(lSm), FadeIn(l4B), run_time=0.8)

        # ---- two a² × 2b² rectangles complete the square: 4a²b² more
        m1, m2 = mk([PL(p) for p in r1], ORANGE), mk([PL(p) for p in r2], ORANGE)
        l1 = tag("2a²b²", 22).move_to(PL((A2 + B2, A2 / 2)))
        l2 = tag("2a²b²", 22).move_to(PL((A2 / 2, A2 + B2)))
        frameL = Polygon(*[PL(p) for p in big], stroke_color=YELLOW_B, stroke_width=4)
        dS_L = _dim(PL((0, S)), PL((S, S)), UP, "a² + 2b²", YELLOW_B, 26, off=0.25)
        self.play(FadeIn(m1), FadeIn(m2), FadeIn(l1), FadeIn(l2), run_time=0.9)
        self.remove(ghostL)
        self.play(Create(frameL), FadeIn(dS_L), run_time=0.8)
        self.hold(0.5)

        # ---- right: the same square, with a (2ab)-square in a corner
        frameR = Polygon(*[PR(p) for p in big], stroke_color=YELLOW_B, stroke_width=4)
        dS_R = _dim(PR((0, S)), PR((S, S)), UP, "a² + 2b²", YELLOW_B, 26, off=0.25)
        self.play(TransformFromCopy(frameL, frameR), TransformFromCopy(dS_L, dS_R),
                  run_time=1.1)
        mC = mk([PR(p) for p in corner], ORANGE)
        lC = tag("(2ab)²", 26).move_to(PR((S - t / 2, S - t / 2)))
        d2ab = _dim(PR((S, S - t)), PR((S, S)), RIGHT, "2ab", GREY_A, 26, off=0.25)
        self.add(mC)
        self.bring_to_front(frameR)
        self.play(FadeIn(mC), FadeIn(lC), FadeIn(d2ab), run_time=0.9)
        eq = tag("2a²b² + 2a²b²  =  4a²b²  =  (2ab)²", 28, ORANGE)
        eq.move_to([(OL[0] + OR[0] + S * k) / 2 + 0.6, 2.75, 0])
        # the two rectangles and the corner square: the same area, 4a²b²
        rims = VGroup(*[Polygon(*[P_(p) for p in q], stroke_color=YELLOW_B, stroke_width=6)
                        for P_, q in ((PL, r1), (PL, r2), (PR, corner))])
        self.play(FadeIn(eq), Create(rims), run_time=1.2)
        self.play(FadeOut(rims), run_time=0.5)

        # ---- the rest of the right square: a⁴ + 4b⁴
        mBst = mk([PR(p) for p in Bst], GREEN_D)
        mLst = mk([PR(p) for p in Lst], GREEN_D)
        lG = tag("a⁴ + 4b⁴", 26).move_to(PR((S / 2, (S - t) / 2)))
        self.add(mBst, mLst)
        self.bring_to_front(frameR)
        self.play(FadeIn(mBst), FadeIn(mLst), FadeIn(lG), run_time=0.9)
        self.hold(0.5)

        # ---- take the (2ab)-square out; the gnomon becomes a rectangle
        ghostC = DashedVMobject(Polygon(*[PR(p) for p in corner], stroke_color=ORANGE,
                                        stroke_width=2.5), num_dashes=32)
        self.add(ghostC)
        self.play(FadeOut(mC), FadeOut(lC), FadeOut(d2ab), FadeOut(frameR), FadeOut(dS_R),
                  run_time=0.8)
        self.bring_to_front(mLst)
        self.play(mLst.animate.shift(k * S * RIGHT), run_time=1.3)
        _landed(mLst, [PR(p) for p in Lsl], "the upper arm slides across")
        dot = Dot(PR(piv), radius=0.07, color=YELLOW_B)
        self.play(FadeIn(dot), FadeOut(ghostC), run_time=0.4)
        self.play(Rotate(mLst, angle=-PI / 2, about_point=PR(piv)), run_time=1.5)
        _landed(mLst, [PR(p) for p in Lt], "the arm turned down beside the lower arm")
        frameF = Polygon(*[PR(p) for p in final], stroke_color=YELLOW_B, stroke_width=4)
        dP = _dim(PR((0, 0)), PR((S + t, 0)), DOWN, "a² + 2b² + 2ab", YELLOW_B, 26, off=0.25)
        dM = _dim(PR((S + t, 0)), PR((S + t, S - t)), RIGHT, "a² + 2b² − 2ab", YELLOW_B, 26,
                  off=0.25)
        # the two arms are now one rectangle: one region, one label
        mF = mk([PR(p) for p in final], GREEN_D)
        self.add(mF)
        self.bring_to_back(mF)
        self.play(FadeOut(dot), FadeOut(mBst), FadeOut(mLst), Create(frameF),
                  lG.animate.move_to(PR(((S + t) / 2, (S - t) / 2))), run_time=0.9)
        self.play(FadeIn(dP), FadeIn(dM), run_time=0.7)

        segs = (_poly_segs([PL(p) for p in big]) + _poly_segs([PL(p) for p in sqA])
                + _poly_segs([PL(p) for p in sqB]) + _poly_segs([PR(p) for p in final])
                + _dim_segs(dA) + _dim_segs(dB) + _dim_segs(dS_L) + _dim_segs(dP)
                + _dim_segs(dM))
        _labels_ok([lA4, l4B, l1, l2, lG, dA[1], dB[1], dS_L[1], dP[1], dM[1], eq],
                   segs, "C18")
        cap = caption("a⁴ + 4b⁴  =  (a² + 2b² + 2ab)(a² + 2b² − 2ab)", 34)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# =================================================================== C23

class C23_DifferenceFourthPowers(Board):
    """The difference-of-squares move, twice.  A square of side a² with a
    square of side b² taken out of its corner leaves a⁴ − b⁴, an L-shaped
    gnomon; its upper arm slides across and turns a quarter-turn down beside
    the lower one: an (a² + b²) × (a² − b²) rectangle.  The same move on
    squares of side a and b turns a² − b² into an (a + b) × (a − b)
    rectangle, so the short side a² − b² of the first rectangle is
    (a + b)(a − b):  a⁴ − b⁴ = (a² + b²)(a + b)(a − b)."""

    def construct(self):
        a, b = 1.8, 1.1

        def stage(O, k, A, B, col, nm, size, rt):
            """Square A minus corner square B -> rectangle (A+B) x (A-B)."""
            def P(p):
                return O + k * to3(p)
            sq, cor = _rect(0, 0, A, A), _rect(A - B, A - B, B, B)
            low, up = _rect(0, 0, A, A - B), _rect(0, A - B, A - B, B)
            piv = (A, A - B)
            up_s = [(x + A, y) for (x, y) in up]
            up_t = [(piv[0] + (y - piv[1]), piv[1] - (x - piv[0])) for (x, y) in up_s]
            fin = _rect(0, 0, A + B, A - B)
            check(_tiles_exactly([cor, low, up], sq), "corner + two arms tile the square")
            check(_same_poly(up_t, _rect(A, 0, B, A - B)), "the arm lands beside the lower arm")
            check(_tiles_exactly([low, up_t], fin), "the arms tile the (A+B) × (A−B) rectangle")
            check(abs((A + B) * (A - B) - (A * A - B * B)) < 1e-12, "A² − B² = (A+B)(A−B)")
            _sweep_ok([P(p) for p in up_s], P(piv), -PI / 2, "the turning arm stays on screen",
                      keep=lambda q: q[0] >= P(piv)[0] - 1e-9)

            mSq = mk([P(p) for p in sq], col)
            lSq = tag(nm["sqA"], size + 6).move_to(P((A * 0.4, A * 0.4)))
            dA = _dim(P((0, 0)), P((A, 0)), DOWN, nm["sideA"], GREY_A, size, off=0.25)
            self.play(FadeIn(mSq), FadeIn(lSq), FadeIn(dA), run_time=0.9 * rt)
            mCor = mk([P(p) for p in cor], RED_D, 0.95)
            lCor = tag(nm["sqB"], size).move_to(P((A - B / 2, A - B / 2)))
            dB = _dim(P((A, A - B)), P((A, A)), RIGHT, nm["sideB"], GREY_A, size, off=0.25)
            self.play(FadeIn(mCor), FadeIn(lCor), FadeIn(dB), run_time=0.8 * rt)
            self.hold(0.3 * rt)

            # take the corner square away: the gnomon A² − B²
            mLow, mUp = mk([P(p) for p in low], col), mk([P(p) for p in up], col)
            ghost = DashedVMobject(Polygon(*[P(p) for p in cor], stroke_color=RED_B,
                                           stroke_width=2.5), num_dashes=28)
            lG = tag(nm["gn"], size).move_to(P((A / 2, (A - B) / 2)))
            self.add(mLow, mUp)
            self.remove(mSq)
            self.add(ghost)
            self.play(FadeOut(mCor), FadeOut(lCor), FadeOut(dB), FadeOut(lSq), FadeIn(lG),
                      run_time=0.8 * rt)
            self.bring_to_front(mUp)
            self.play(mUp.animate.shift(k * A * RIGHT), run_time=1.1 * rt)
            _landed(mUp, [P(p) for p in up_s], "the upper arm slides across")
            dot = Dot(P(piv), radius=0.06, color=YELLOW_B)
            self.play(FadeIn(dot), FadeOut(ghost), FadeOut(dA), run_time=0.3 * rt)
            self.play(Rotate(mUp, angle=-PI / 2, about_point=P(piv)), run_time=1.3 * rt)
            _landed(mUp, [P(p) for p in up_t], "the arm turned down beside the lower one")
            mF = mk([P(p) for p in fin], col)
            frame = Polygon(*[P(p) for p in fin], stroke_color=YELLOW_B, stroke_width=4)
            dP = _dim(P((0, 0)), P((A + B, 0)), DOWN, nm["plus"], YELLOW_B, size, off=0.25)
            dM = _dim(P((A + B, 0)), P((A + B, A - B)), RIGHT, nm["minus"], YELLOW_B, size,
                      off=0.25)
            self.add(mF)
            self.bring_to_back(mF)
            self.play(FadeOut(dot), FadeOut(mLow), FadeOut(mUp), Create(frame),
                      lG.animate.move_to(P(((A + B) / 2, (A - B) / 2))), run_time=0.8 * rt)
            self.play(FadeIn(dP), FadeIn(dM), run_time=0.6 * rt)
            return mF, frame, lG, dP, dM, P

        O1 = np.array([-6.35, -1.95, 0.0])
        O2 = np.array([0.95, -1.95, 0.0])
        big = stage(O1, 0.9, a * a, b * b, BLUE_D,
                    dict(sqA="a⁴", sqB="b⁴", gn="a⁴ − b⁴", sideA="a²", sideB="b²",
                         plus="a² + b²", minus="a² − b²"), 26, 1.0)
        r1 = tag("a⁴ − b⁴  =  (a² + b²)(a² − b²)", 28).move_to([-3.2, 2.7, 0])
        self.play(FadeIn(r1), run_time=0.7)
        self.hold(0.4)
        small = stage(O2, 1.4, a, b, TEAL_D,
                      dict(sqA="a²", sqB="b²", gn="a² − b²", sideA="a", sideB="b",
                           plus="a + b", minus="a − b"), 26, 0.8)
        r2 = tag("a² − b²  =  (a + b)(a − b)", 28).move_to([3.6, 2.7, 0])
        self.play(FadeIn(r2), run_time=0.7)
        self.hold(0.3)

        # ---- the short side of the first rectangle is (a + b)(a − b)
        dM1 = big[4]
        new = tag("(a + b)(a − b)", 26, YELLOW_B)
        new.move_to(dM1[1].get_center() + (new.width - dM1[1].width) / 2 * RIGHT)
        rim = small[1].copy().set_stroke(YELLOW_B, 8)
        self.play(Create(rim), Indicate(dM1[1], color=YELLOW_B, scale_factor=1.15),
                  run_time=0.9)
        self.play(FadeOut(dM1[1]), FadeOut(rim), FadeIn(new), run_time=0.9)

        segs = []
        for st in (big, small):
            mF, frame, lG, dP, dM, P = st
            segs += _poly_segs(frame.get_vertices()) + _dim_segs(dP) + _dim_segs(dM)
        labels = [big[2], big[3][1], new, small[2], small[3][1], small[4][1], r1, r2]
        _labels_ok(labels, segs, "C23")
        cap = caption("a⁴ − b⁴  =  (a² + b²)(a + b)(a − b)", 36)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# =================================================================== A16

class A16_ReciprocalPythagorean(Board):
    """The right triangle with legs a, b, hypotenuse c and altitude h on c.
    Twice its area two ways: the triangle and a copy turned half a turn
    about the midpoint of c make an a × b rectangle; the triangle and its
    two halves (cut along h), each turned half a turn about the midpoint of
    its leg, make a c × h rectangle.  So ab = ch.  Scaled by 1/(ab) the
    triangle has legs a/(ab) = 1/b and b/(ab) = 1/a and hypotenuse
    c/(ab) = c/(ch) = 1/h, and it is still right-angled:
        (1/a)² + (1/b)² = (1/h)²."""

    def construct(self):
        a, b = 1.15, 1.65                     # CB = a, CA = b
        c = float(np.hypot(a, b))
        h = a * b / c
        k = 2.6
        O = np.array([-6.05, 0.4, 0.0])       # screen position of A

        def S(p):
            return O + k * to3(p)

        Am, Bm, Cm, Hm = (0.0, 0.0), (c, 0.0), (b * b / c, h), (b * b / c, 0.0)
        A, B, C, H = S(Am), S(Bm), S(Cm), S(Hm)
        M = (A + B) / 2
        Cq = A + B - C                         # C turned half a turn about M
        A1, B1 = S((0, h)), S((c, h))          # top corners of the c × h rectangle
        mAC, mBC = (A + C) / 2, (B + C) / 2
        check(abs(_L(C, A) - b * k) < 1e-9 and abs(_L(C, B) - a * k) < 1e-9
              and abs(np.dot(A - C, B - C)) < 1e-9, "legs a, b and a right angle at C")
        check(abs(np.dot(C - H, B - A)) < 1e-9 and abs(_L(C, H) - h * k) < 1e-9,
              "CH is the altitude h on c")
        check(abs(np.dot(A - C, B - C)) < 1e-9 and close(Cq - A, B - C)
              and abs(_L(A, Cq) - a * k) < 1e-9, "A C B C' is an a × b rectangle")
        check(close(2 * mAC - H, A1) and close(2 * mBC - H, B1),
              "the half-turns take H to the top corners")
        check(_tiles_exactly([[A, B, C], [C, A, A1], [C, B, B1]], [A, B, B1, A1]),
              "the triangle and its turned halves tile the c × h rectangle")
        check(_tiles_exactly([[A, B, C], [B, A, Cq]], [A, C, B, Cq]),
              "the triangle and its turned copy tile the a × b rectangle")
        check(abs(a * b - c * h) < 1e-12, "ab = ch")
        f = 1 / (a * b)
        check(abs(a * f - 1 / b) < 1e-12 and abs(b * f - 1 / a) < 1e-12
              and abs(c * f - 1 / h) < 1e-12, "scaled by 1/(ab): 1/b, 1/a, 1/h")
        check(abs((1 / a) ** 2 + (1 / b) ** 2 - (1 / h) ** 2) < 1e-12,
              "(1/a)² + (1/b)² = (1/h)²")

        # ---- the triangle, its altitude
        tri = mk([A, B, C], GREY_D, 0.7, stroke_width=3)
        raC = _ra(C, A, B, 0.2)
        la = _beside(tag("a", 30), C, B, A - B, 0.1, at=0.45)
        lb = _beside(tag("b", 30), C, A, B - A, 0.12, at=0.5)
        lc = _beside(tag("c", 30), A, B, C - H, 0.1, at=0.3)
        self.play(FadeIn(tri), Create(raC), FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=1.1)
        alt = DashedLine(C, H, color=WHITE, stroke_width=3, dash_length=0.09)
        raH = _ra(H, C, A, 0.17)
        lh = _beside(tag("h", 30), C, H, A - B, 0.1, at=0.45)
        self.play(Create(alt), Create(raH), FadeIn(lh), run_time=0.8)
        self.hold(0.3)

        # ---- twice the triangle: an a × b rectangle
        cp = mk([A, B, C], BLUE_D, 0.55, stroke_width=0)
        dotM = Dot(M, radius=0.06, color=BLUE_B)
        self.add(cp)
        self.play(FadeIn(dotM), run_time=0.3)
        self.play(Rotate(cp, angle=PI, about_point=M), run_time=1.3)
        _landed(cp, [B, A, Cq], "the half-turn lays the copy under c")
        rAB = Polygon(A, C, B, Cq, stroke_color=BLUE_B, stroke_width=5)
        lab = _beside(tag("ab", 32, BLUE_B), A, Cq, (A + Cq) / 2 - M, 0.14)   # outside
        self.play(Create(rAB), FadeIn(lab), FadeOut(dotM), run_time=0.8)

        # ---- twice the triangle again: a c × h rectangle
        hL = mk([A, C, H], ORANGE, 0.55, stroke_width=0)
        hR = mk([B, C, H], ORANGE, 0.55, stroke_width=0)
        dots = VGroup(Dot(mAC, radius=0.06, color=ORANGE), Dot(mBC, radius=0.06, color=ORANGE))
        self.play(FadeIn(hL), FadeIn(hR), FadeIn(dots), run_time=0.5)
        self.play(Rotate(hL, angle=PI, about_point=mAC), Rotate(hR, angle=PI, about_point=mBC),
                  run_time=1.4)
        _landed(hL, [C, A, A1], "the left half fills the left corner")
        _landed(hR, [C, B, B1], "the right half fills the right corner")
        rCH = Polygon(A, B, B1, A1, stroke_color=ORANGE, stroke_width=5)
        lch = tag("ch", 32, ORANGE).next_to((A1 + C) / 2, UP, buff=0.16)
        self.play(Create(rCH), FadeIn(lch), FadeOut(dots), run_time=0.8)
        eq = _row(tag("ab", 32, BLUE_B), tag("=", 32), tag("ch", 32, ORANGE), buff=0.2)
        eq.move_to([3.7, 3.05, 0])
        self.play(FadeIn(eq), run_time=0.7)
        self.hold(0.4)

        # ---- the triangle scaled by 1/(ab)
        sh = np.array([7.05, -1.85, 0.0])                  # where the copy goes
        cp2 = mk([A, B, C], TEAL_D, 0.8)
        self.add(cp2)
        self.play(cp2.animate.shift(sh), run_time=1.3)
        A2, B2, C2 = A + sh, B + sh, C + sh
        _landed(cp2, [A2, B2, C2], "the copy slides across")
        fac = tag("× 1/(ab)", 28, TEAL_B).move_to(C2 + np.array([-2.15, 0.25, 0]))
        dotC = Dot(C2, radius=0.06, color=TEAL_B)
        self.play(FadeIn(fac), FadeIn(dotC), run_time=0.5)
        self.play(_homothety(cp2, C2, f), run_time=1.4)
        As, Bs = C2 + f * (A2 - C2), C2 + f * (B2 - C2)
        _landed(cp2, [As, Bs, C2], "the copy is scaled by 1/(ab) about its right angle")
        check(abs(_L(C2, Bs) - k / b) < 1e-9 and abs(_L(C2, As) - k / a) < 1e-9
              and abs(_L(As, Bs) - k / h) < 1e-9, "its sides are 1/b, 1/a, 1/h")
        raS = _ra(C2, As, Bs, 0.17)
        l1b = _beside(tag("1/b", 28), C2, Bs, Bs - As, 0.1)
        l1a = _beside(tag("1/a", 28), C2, As, As - Bs, 0.1)
        l1h = _beside(tag("1/h", 28), As, Bs, (As + Bs) / 2 - C2, 0.12)
        self.play(FadeOut(dotC), Create(raS), FadeIn(l1b), FadeIn(l1a), FadeIn(l1h),
                  run_time=0.8)
        rh = tag("c/(ab)  =  c/(ch)  =  1/h", 26).move_to([3.7, -1.55, 0])
        self.play(FadeIn(rh), run_time=0.8)
        rp = tag("(1/a)² + (1/b)²  =  (1/h)²", 28, YELLOW_B).move_to([3.7, -2.35, 0])
        self.play(FadeIn(rp), run_time=0.8)

        segs = (_poly_segs([A, C, B, Cq]) + _poly_segs([A, B, B1, A1]) + [(A, B), (C, H)]
                + _ra_segs(C, A, B, 0.2) + _ra_segs(H, C, A, 0.17)
                + _poly_segs([As, Bs, C2]) + _ra_segs(C2, As, Bs, 0.17))
        labels = [la, lb, lc, lh, lab, lch, eq, fac, l1b, l1a, l1h, rh, rp]
        _labels_ok(labels, segs, "A16")
        cap = caption("1/a² + 1/b²  =  1/h²", 38)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# =================================================================== A22

class A22_IncirclePythagoras(Board):
    """The incircle of the right triangle (legs a, b, hypotenuse c) touches
    the sides; the two tangents from each vertex are equal: r, r from the
    right angle (the radii make a square there), x, x from A and y, y from
    B.  So a + b − c = 2r and a + b + c = 2(r + x + y).  Cut the
    triangle into that square and the two kites at A and B, each halved
    along its axis (the line to the incentre).  Folding one half of each
    kite over the perpendicular bisector of its axis completes an x × r
    and an r × y rectangle; the second turns a quarter-turn beside the
    square: all of the triangle makes an r × (r + x + y) strip.  So
        ab/2 = r(r + x + y) = (a + b − c)/2 · (a + b + c)/2,
    i.e. 2ab = (a + b)² − c², and with (a + b)² = a² + 2ab + b²: c² = a² + b²."""

    def construct(self):
        a, b = 2.3, 3.4
        c = float(np.hypot(a, b))
        r = (a + b - c) / 2
        x, y = b - r, a - r
        s = r + x + y
        D = r + 0.95                                   # how far the strip drops
        Cm, Am, Bm, Im = (0.0, 0.0), (b, 0.0), (0.0, a), (r, r)
        Tbm, Tam = (r, 0.0), (0.0, r)
        Tcm = tuple(np.array(Am) + x * (np.array(Bm) - np.array(Am)) / c)
        EAm, EBm = (b, r), (r, a)
        F = Frame(-y - 0.75, b + 0.45, -D - 1.25, a + 0.35, max_w=7.2,
                  centre=(-2.95, (SAFE_TOP + SAFE_BOTTOM) / 2))
        P, k = F.P, F.k
        C, A, B, I = P(Cm), P(Am), P(Bm), P(Im)
        Tb, Ta, Tc, EA, EB = P(Tbm), P(Tam), P(Tcm), P(EAm), P(EBm)
        R = r * k

        # ---- the claims
        check(abs(_L(I, Ta) - R) < 1e-9 and abs(_L(I, Tb) - R) < 1e-9
              and abs(_L(I, Tc) - R) < 1e-9 and abs(np.dot(I - Tc, B - A)) < 1e-9,
              "the circle (I, r) touches all three sides")
        check(abs(_L(A, Tb) - _L(A, Tc)) < 1e-9 and abs(_L(B, Ta) - _L(B, Tc)) < 1e-9
              and abs(_L(C, Ta) - R) < 1e-9 and abs(_L(C, Tb) - R) < 1e-9,
              "equal tangents r, r / x, x / y, y")
        check(abs((a + b - c) - 2 * r) < 1e-12 and abs((a + b + c) - 2 * s) < 1e-12,
              "a + b − c = 2r, a + b + c = 2(r + x + y)")
        sq = [C, Tb, I, Ta]
        KA1, KA2 = [A, I, Tb], [A, Tc, I]
        KB1, KB2 = [B, Ta, I], [B, Tc, I]
        check(_tiles_exactly([sq, KA1, KA2, KB1, KB2], [C, A, B]),
              "square + four kite halves tile the triangle")
        mAI, mBI = (A + I) / 2, (B + I) / 2
        nAI = np.array([-(I - A)[1], (I - A)[0], 0.0])
        nBI = np.array([-(I - B)[1], (I - B)[0], 0.0])

        def mirror(p, m, n):
            d = _u(n)
            f = m + np.dot(p - m, d) * d
            return 2 * f - p

        KA2f = [mirror(p, mAI, nAI) for p in KA2]
        KB2f = [mirror(p, mBI, nBI) for p in KB2]
        check(_same_poly(KA2f, [I, EA, A]) and _same_poly(KB2f, [I, EB, B]),
              "each fold carries the half kite onto the other half of a rectangle")
        RA, RB = [Tb, A, EA, I], [Ta, I, EB, B]
        check(_tiles_exactly([KA1, KA2f], RA) and _tiles_exactly([KB1, KB2f], RB),
              "x × r and r × y rectangles")

        def turn(p):                                    # quarter-turn about Ta
            return _rot2(p, Ta, PI / 2)

        drop1 = -R * UP                                  # slide down by r
        RBt = [turn(p) + drop1 for p in RB]
        strip = [P((-y, 0)), A, P((b, r)), P((-y, r))]
        check(_tiles_exactly([sq, RA, RBt], strip),
              "the three pieces tile the r × (r + x + y) strip")
        check(abs(abs(area(strip)) - r * s * k * k) < 1e-9
              and abs(r * s - a * b / 2) < 1e-12, "ab/2 = r(r + x + y)")
        _sweep_ok(RB, Ta, PI / 2, "the turning arm stays left of x = r, above y = r",
                  keep=lambda q: q[0] <= I[0] + 1e-9 and q[1] >= Ta[1] - 1e-9)

        # ---- the triangle and its incircle
        tri = Polygon(C, A, B, stroke_color=WHITE, stroke_width=3)
        raC = _ra(C, A, B, 0.18)
        circ = Circle(radius=R, color=GREY_A, stroke_width=3).move_to(I)
        dotI = Dot(I, radius=0.05)
        radii = VGroup(*[Line(I, T, color=GREY_A, stroke_width=2) for T in (Ta, Tb, Tc)])
        la = tag("a", 30).next_to((C + B) / 2, LEFT, buff=0.5)
        lb = tag("b", 30).next_to((C + A) / 2, DOWN, buff=0.5)
        lc = _beside(tag("c", 30), A, B, (A + B) / 2 - I, 0.5)
        self.play(Create(tri), Create(raC), FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=1.1)
        self.play(Create(circ), FadeIn(dotI), Create(radii), run_time=1.0)

        # ---- equal tangents
        cR, cX, cY = YELLOW_E, BLUE_B, RED_B
        segs_t = [(C, Ta, cR), (C, Tb, cR), (A, Tb, cX), (A, Tc, cX), (B, Ta, cY), (B, Tc, cY)]
        bars = VGroup(*[Line(p, q, color=col, stroke_width=8) for p, q, col in segs_t])
        out = lambda p, q: _u((p + q) / 2 - I)
        tl = VGroup(*[_beside(tag(nm, 26, col), p, q, out(p, q), 0.1)
                      for (p, q, col), nm in zip(segs_t, "rrxxyy")])
        self.play(FadeOut(la), FadeOut(lb), FadeOut(lc), Create(bars), FadeIn(tl), run_time=1.0)
        x_l = 1.35
        R1 = tag("a + b − c  =  2r", 28)
        R2 = tag("a + b + c  =  2(r + x + y)", 28)
        R1.move_to([x_l + R1.width / 2, 2.85, 0])
        R2.move_to([x_l + R2.width / 2, 2.15, 0])
        self.play(FadeIn(R1), run_time=0.6)
        self.play(FadeIn(R2), run_time=0.6)
        self.hold(0.4)

        # ---- cut: the square at C and the kites at A and B, halved
        m_sq = mk(sq, YELLOW_E, 0.85)
        m_a1, m_a2 = mk(KA1, BLUE_D), mk(KA2, BLUE_D)
        m_b1, m_b2 = mk(KB1, RED_D), mk(KB2, RED_D)
        pieces = [m_sq, m_a1, m_a2, m_b1, m_b2]
        cuts = VGroup(Line(A, I, color=WHITE, stroke_width=2.5),
                      Line(B, I, color=WHITE, stroke_width=2.5))
        self.add(*pieces)
        for m in pieces:
            m.set_opacity(0)
        self.bring_to_front(bars, radii, dotI)
        self.play(*[m.animate.set_fill(opacity=FILL if m is not m_sq else 0.85)
                    .set_stroke(opacity=1) for m in pieces], Create(cuts),
                  FadeOut(circ), run_time=1.0)
        self.remove(cuts, radii)
        ghost = DashedVMobject(Polygon(C, A, B, stroke_color=GREY_B, stroke_width=2.5),
                               num_dashes=70)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.remove(tri, raC)

        # ---- fold one half of each kite over the perpendicular bisector of its axis
        axA = DashedLine(mAI - 0.65 * _u(nAI), mAI + 0.65 * _u(nAI), color=WHITE,
                         stroke_width=3, dash_length=0.08)
        axB = DashedLine(mBI - 0.65 * _u(nBI), mBI + 0.65 * _u(nBI), color=WHITE,
                         stroke_width=3, dash_length=0.08)
        self.play(Create(axA), Create(axB), FadeOut(bars), FadeOut(tl), run_time=0.6)
        self.bring_to_front(m_a2, m_b2)
        self.play(_fold(m_a2, mAI, mAI + nAI), _fold(m_b2, mBI, mBI + nBI), run_time=1.6)
        _landed(m_a2, KA2f, "the half kite at A lands on the rectangle x × r")
        _landed(m_b2, KB2f, "the half kite at B lands on the rectangle r × y")
        self.play(FadeOut(axA), FadeOut(axB), run_time=0.4)

        # ---- the r × y rectangle turns down beside the square
        armB = VGroup(m_b1, m_b2)
        dot = Dot(Ta, radius=0.06, color=YELLOW_B)
        self.play(FadeIn(dot), run_time=0.3)
        self.play(Rotate(armB, angle=PI / 2, about_point=Ta), run_time=1.4)
        self.play(FadeOut(dot), armB.animate.shift(drop1), run_time=0.9)
        _landed(m_b1, [turn(p) + drop1 for p in KB1], "the arm lies beside the square")
        _landed(m_b2, [turn(p) + drop1 for p in KB2f], "the arm lies beside the square")

        # ---- the whole strip drops below the triangle
        allp = VGroup(*pieces)
        drop2 = -k * D * UP
        self.play(allp.animate.shift(drop2), run_time=1.1)
        st = [p + drop2 for p in strip]
        for m, src in ((m_sq, sq), (m_a1, KA1), (m_a2, KA2f),
                       (m_b1, [turn(p) + drop1 for p in KB1]),
                       (m_b2, [turn(p) + drop1 for p in KB2f])):
            _landed(m, [p + drop2 for p in src], "the strip moves whole")
        frame = Polygon(*st, stroke_color=YELLOW_B, stroke_width=4)
        X0, X1, X2, X3 = st[0], st[0] + y * k * RIGHT, st[0] + (y + r) * k * RIGHT, st[1]
        g = 0.025 * RIGHT
        dy = _dim(X0, X1 - g, DOWN, "y", cY, 26, off=0.22)
        dr = _dim(X1 + g, X2 - g, DOWN, "r", cR, 26, off=0.22)
        dx = _dim(X2 + g, X3, DOWN, "x", cX, 26, off=0.22)
        ds = _dim(X0, X3, DOWN, "r + x + y", YELLOW_B, 28, off=0.72)
        dh = _dim(st[0], st[3], LEFT, "r", cR, 26, off=0.22)
        circ2 = Circle(radius=R, color=GREY_A, stroke_width=2.5).move_to(I)
        lab2 = tag("ab/2", 28).move_to(P((2.2, 0.36)))
        la2 = tag("a", 28).next_to((C + B) / 2, LEFT, buff=0.18)
        lb2 = tag("b", 28).next_to((C + A) / 2, DOWN, buff=0.16)
        lc2 = _beside(tag("c", 28), A, B, (A + B) / 2 - I, 0.14)
        self.play(Create(frame), FadeIn(dy), FadeIn(dr), FadeIn(dx), FadeIn(dh),
                  FadeIn(circ2), FadeIn(lab2), FadeIn(la2), FadeIn(lb2), FadeIn(lc2),
                  run_time=0.9)
        self.play(FadeIn(ds), run_time=0.5)
        R3 = tag("ab/2  =  r(r + x + y)", 28, YELLOW_B)
        R4 = tag("2ab  =  (a + b)² − c²", 28)
        R3.move_to([x_l + R3.width / 2, 0.55, 0])
        R4.move_to([x_l + R4.width / 2, -0.2, 0])
        self.play(FadeIn(R3), run_time=0.7)
        self.play(FadeIn(R4), run_time=0.7)

        segs = (_poly_segs([C, A, B]) + _circle_segs(I, R) + _poly_segs(st)
                + [(st[0] + (y * k) * RIGHT, st[3] + (y * k) * RIGHT),
                   (st[0] + ((y + r) * k) * RIGHT, st[3] + ((y + r) * k) * RIGHT)]
                + _dim_segs(dy) + _dim_segs(dr) + _dim_segs(dx) + _dim_segs(ds) + _dim_segs(dh))
        labels = [lab2, la2, lb2, lc2, dy[1], dr[1], dx[1], ds[1], dh[1], R1, R2, R3, R4]
        _labels_ok(labels, segs, "A22")
        cap = caption("ab/2  =  (a + b − c)/2 · (a + b + c)/2    ⟹    c²  =  a² + b²", 32)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# =================================================================== A24

class A24_TwoHalfSquares(Board):
    """Half-squares on the legs of the right triangle (legs a, b), both with
    their right angle at C, make a quadrilateral AA'BB' (a dart: concave at
    B).  Its diagonals are AB = c and A'B' — the triangle turned a
    quarter-turn about C — so they are equal and perpendicular.  Cut along
    AB: two triangles on the base c whose apexes A', B' lie on the line
    A'B' ⊥ AB, on opposite sides.  Shear each (apex sliding parallel to AB)
    onto the perpendicular to AB at B: one triangle with base c and height
    c.  Shear it once more (A sliding parallel to that base): a right
    triangle with legs c and c, half the square on c.  Shears keep areas:
        a²/2 + b²/2 = c·c/2."""

    def construct(self):
        a, b = 1.6, 2.6
        c = float(np.hypot(a, b))
        Cm, Am, Bm = np.array([0.0, 0.0]), np.array([b, 0.0]), np.array([0.0, a])
        Apm, Bpm = np.array([0.0, b]), np.array([-a, 0.0])
        u, n = (Bm - Am) / c, np.array([a, b]) / c
        beta = float((Apm - Bm) @ n)
        A2m = Bm + beta * n
        B2m = A2m - c * n
        Asm = Am + beta * n
        Qsm = Asm - c * n

        def dist(p):                                  # signed distance from AB
            return float((np.asarray(p, float) - Am) @ n)

        lam1 = float((A2m - Apm) @ u) / dist(Apm)     # shear of the A' side
        lam2 = float((B2m - Bpm) @ u) / dist(Bpm)     # shear of the B' side

        def S1a(p):
            return np.asarray(p, float) + lam1 * dist(p) * u

        def S1b(p):
            return np.asarray(p, float) + lam2 * dist(p) * u

        def S2(p):                                     # fixes line A''B'', slides along n
            p = np.asarray(p, float)
            return p - (beta / c) * float((p - A2m) @ u) * n

        # the pieces: the upper part of the b half-square, its lower part (the
        # triangle CAB) and the a half-square
        P1 = [Am, Apm, Bm]
        P2t = [Cm, Am, Bm]
        P2o = [Cm, Bm, Bpm]
        Ta, Tb = [Cm, Bm, Bpm], [Cm, Am, Apm]
        dart = [Am, Apm, Bm, Bpm]
        check(abs(np.linalg.norm(Bpm - Cm) - a) < 1e-12 and abs((Bpm - Cm) @ (Bm - Cm)) < 1e-12
              and abs(np.linalg.norm(Apm - Cm) - b) < 1e-12 and abs((Apm - Cm) @ (Am - Cm)) < 1e-12,
              "half-squares on the legs, right angles at C")
        check(_tiles_exactly([Ta, Tb], dart), "the two half-squares make the dart AA'BB'")
        check(close(_rot2(Am, Cm, PI / 2)[:2], Apm) and close(_rot2(Bm, Cm, PI / 2)[:2], Bpm),
              "the quarter-turn about C takes AB to A'B'")
        check(abs(np.linalg.norm(Apm - Bpm) - c) < 1e-12 and abs((Apm - Bpm) @ (Bm - Am)) < 1e-12,
              "the diagonals are equal and perpendicular")
        check(dist(Apm) * dist(Bpm) < 0, "A' and B' on opposite sides of AB")
        check(_tiles_exactly([P1, P2t, P2o], dart), "cut along AB")
        check(close(S1a(Apm), A2m) and close(S1b(Bpm), B2m) and close(S1a(Am), Am)
              and close(S1b(Bm), Bm), "shear 1: the apexes reach the perpendicular at B")
        check(abs(np.linalg.norm(A2m - B2m) - c) < 1e-12 and abs((A2m - Bm) @ u) < 1e-12
              and abs((B2m - Bm) @ u) < 1e-12, "A''B'' = c on the perpendicular at B")
        T1 = [Am, A2m, B2m]
        sh1 = [[S1a(p) for p in P1], [S1b(p) for p in P2t], [S1b(p) for p in P2o]]
        check(_tiles_exactly(sh1, T1), "after shear 1: one triangle, base c, height c")
        check(close(S2(Am), Asm) and close(S2(A2m), A2m) and close(S2(B2m), B2m),
              "shear 2 keeps the base A''B'' and takes A to A*")
        sh2 = [[S2(p) for p in P_] for P_ in sh1]
        T2 = [Asm, A2m, B2m]
        check(_tiles_exactly(sh2, T2), "after shear 2: the same pieces")
        check(abs(np.linalg.norm(Asm - A2m) - c) < 1e-12 and abs((Asm - A2m) @ (B2m - A2m)) < 1e-12,
              "a right triangle with legs c, c: half the square on c")
        check(abs(abs(area(T2)) - (a * a + b * b) / 2) < 1e-12
              and abs(abs(area(sh2[2])) - a * a / 2) < 1e-12, "areas kept by the shears")

        F = Frame(-2.15, 3.55, -2.25, 3.0, max_w=9.0, centre=(-1.2, (SAFE_TOP + SAFE_BOTTOM) / 2))
        P = F.P
        C, A, B, Ap, Bp = P(Cm), P(Am), P(Bm), P(Apm), P(Bpm)
        A2, B2, As, Qs = P(A2m), P(B2m), P(Asm), P(Qsm)

        # ---- the triangle
        tri = Polygon(C, A, B, stroke_color=WHITE, stroke_width=3)
        raC = _ra(C, A, B, 0.18)
        la = tag("a", 28).move_to(P((0.25, a * 0.55)))
        lb = tag("b", 28).move_to(P((b * 0.55, 0.22)))
        lc = _beside(tag("c", 28), A, B, Apm - Cm, 0.1, at=0.55)
        self.play(Create(tri), Create(raC), FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=1.0)

        # ---- the squares on the legs, halved by a diagonal
        sqb = DashedVMobject(Polygon(C, A, P((b, b)), Ap, stroke_color=TEAL_B, stroke_width=2.5),
                             num_dashes=48)
        sqa = DashedVMobject(Polygon(C, B, P((-a, a)), Bp, stroke_color=ORANGE, stroke_width=2.5),
                             num_dashes=32)
        self.play(Create(sqb), Create(sqa), run_time=0.9)
        mTb, mTa = mk([C, A, Ap], TEAL_D), mk([C, B, Bp], ORANGE)
        lB2 = tag("b²/2", 28).move_to(P((0.62, 1.98)))
        lA2 = tag("a²/2", 26).move_to(P((-0.5, 0.48)))
        self.add(mTb, mTa)
        self.bring_to_back(mTb, mTa)
        for m in (mTb, mTa):
            m.set_opacity(0)
        self.play(mTb.animate.set_fill(opacity=FILL).set_stroke(opacity=1),
                  mTa.animate.set_fill(opacity=FILL).set_stroke(opacity=1),
                  FadeIn(lB2), FadeIn(lA2), run_time=1.0)
        self.play(FadeOut(sqb), FadeOut(sqa), run_time=0.5)
        self.hold(0.4)

        # ---- the diagonals of the dart: AB and its quarter-turn A'B'
        ghost = Polygon(C, A, B, stroke_color=YELLOW_B, stroke_width=3)
        dotC = Dot(C, radius=0.06, color=YELLOW_B)
        self.add(ghost)
        self.play(FadeIn(dotC), run_time=0.3)
        self.play(Rotate(ghost, angle=PI / 2, about_point=C), run_time=1.4)
        check(close(ghost.get_vertices()[1], Ap, 1e-6) and close(ghost.get_vertices()[2], Bp, 1e-6),
              "the turned copy has its hypotenuse on A'B'")
        dAB = Line(A, B, color=YELLOW_B, stroke_width=5)
        dApBp = Line(Ap, Bp, color=YELLOW_B, stroke_width=5)
        Xm = Bm + float((Apm - Bm) @ u) * u                     # where the lines cross
        ext = DashedLine(B, P(Xm), color=YELLOW_B, stroke_width=2.5, dash_length=0.08)
        raX = _ra(P(Xm), A, Ap, 0.17, YELLOW_B)
        self.play(Create(dAB), Create(dApBp), FadeOut(ghost), FadeOut(dotC), run_time=0.8)
        self.play(Create(ext), Create(raX), run_time=0.6)
        lcc = _beside(tag("c", 28, YELLOW_B), Ap, Bp, (Ap + Bp) / 2 - C, 0.1)
        self.play(FadeIn(lcc), run_time=0.4)
        self.hold(0.5)

        # ---- cut along AB; shear the two triangles onto the perpendicular at B
        mP1, mP2t, mP2o = mk([A, Ap, B], TEAL_D), mk([C, A, B], TEAL_D), mk([C, B, Bp], ORANGE)
        self.add(mP1, mP2t, mP2o)
        self.remove(mTb, mTa, tri, raC)
        self.bring_to_front(dAB)
        perp = DashedLine(P(Bm + 1.25 * n), P(Bm - 2.25 * n), color=WHITE, stroke_width=2.5,
                          dash_length=0.09)
        # the tracks of the two apexes, parallel to AB
        g1 = DashedLine(P(Apm + 0.75 * u), P(A2m - 0.75 * u), color=TEAL_A, stroke_width=3.5,
                        dash_length=0.1)
        g2 = DashedLine(P(Bpm + 0.75 * u), P(B2m - 0.75 * u), color=ORANGE, stroke_width=3.5,
                        dash_length=0.1)
        self.play(FadeOut(la), FadeOut(lb), FadeOut(lc), FadeOut(lA2), FadeOut(lB2),
                  FadeOut(lcc), FadeOut(ext), FadeOut(raX), FadeOut(dApBp),
                  Create(perp), Create(g1), Create(g2), run_time=0.8)
        self.play(Transform(mP1, mk([P(p) for p in sh1[0]], TEAL_D)),
                  Transform(mP2t, mk([P(p) for p in sh1[1]], TEAL_D)),
                  Transform(mP2o, mk([P(p) for p in sh1[2]], ORANGE)), run_time=1.8)
        for m, pts in zip((mP1, mP2t, mP2o), sh1):
            _landed(m, [P(p) for p in pts], "shear 1 lands")
        self.play(FadeOut(g1), FadeOut(g2), FadeOut(dAB), run_time=0.4)

        # ---- shear once more: A slides parallel to the new base A''B''
        base = Line(A2, B2, color=WHITE, stroke_width=6)
        g3 = DashedLine(P(Am - 0.35 * n), P(Asm + 0.35 * n), color=YELLOW_B, stroke_width=2.5,
                        dash_length=0.08)
        self.play(Create(base), Create(g3), FadeOut(perp), run_time=0.6)
        self.play(Transform(mP1, mk([P(p) for p in sh2[0]], TEAL_D)),
                  Transform(mP2t, mk([P(p) for p in sh2[1]], TEAL_D)),
                  Transform(mP2o, mk([P(p) for p in sh2[2]], ORANGE)), run_time=1.8)
        for m, pts in zip((mP1, mP2t, mP2o), sh2):
            _landed(m, [P(p) for p in pts], "shear 2 lands")
        self.play(FadeOut(g3), FadeOut(base), run_time=0.4)

        # ---- half the square on c
        half = DashedVMobject(Polygon(As, A2, B2, Qs, stroke_color=YELLOW_B, stroke_width=3),
                              num_dashes=72)
        raA2 = _ra(A2, As, B2, 0.2, YELLOW_B)
        lc1 = _beside(tag("c", 30, YELLOW_B), A2, As, A2 - Qs, 0.12)
        lc2 = _beside(tag("c", 30, YELLOW_B), A2, B2, A2 - Qs, 0.12)
        cen_o = P(np.mean(np.array(sh2[2]), axis=0))
        lA2f = tag("a²/2", 24).move_to(cen_o)
        # the two teal pieces are the b half-square, sheared: one region
        tealq = [sh2[1][0], Asm, A2m, Bm]
        check(_tiles_exactly([sh2[0], sh2[1]], tealq), "the teal pieces make one quadrilateral")
        mTeal = mk([P(p) for p in tealq], TEAL_D)
        lB2f = tag("b²/2", 28).move_to(P(np.mean(np.array(tealq), axis=0)))
        self.add(mTeal)
        self.bring_to_back(mTeal)
        self.add(half)
        self.bring_to_back(half)
        self.play(FadeOut(mP1), FadeOut(mP2t), Create(half), Create(raA2), FadeIn(lc1),
                  FadeIn(lc2), FadeIn(lA2f), FadeIn(lB2f), run_time=1.1)

        tri_o = [P(p) for p in sh2[2]]
        tri_t = [P(p) for p in tealq]
        check(all(_pip(q, [t[:2] for t in tri_o]) for q in
                  (lA2f.get_corner(UL)[:2], lA2f.get_corner(UR)[:2], lA2f.get_corner(DL)[:2],
                   lA2f.get_corner(DR)[:2])), "a²/2 sits inside the orange part")
        check(all(_pip(q, [t[:2] for t in tri_t]) for q in
                  (lB2f.get_corner(UL)[:2], lB2f.get_corner(UR)[:2], lB2f.get_corner(DL)[:2],
                   lB2f.get_corner(DR)[:2])), "b²/2 sits inside the teal part")
        segs = (_poly_segs([As, A2, B2, Qs]) + _poly_segs(tri_o) + _poly_segs(tri_t)
                + _ra_segs(A2, As, B2, 0.2))
        _labels_ok([lc1, lc2, lA2f, lB2f], segs, "A24")
        cap = caption("a²/2  +  b²/2  =  c · c/2", 36)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# =================================================================== A27

def _trow(parts, size=30, buff=0.14):
    """A row of coloured Text pieces [(s, colour), ...] on one baseline (y = 0)."""
    g = VGroup()
    x = 0.0
    for s, col in parts:
        t = _on_base(s, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + buff
        g.add(t)
    return g


class A27_PtolemyPythagoras(Board):
    """The right triangle ABC (right angle at B, legs a = AB, b = BC,
    hypotenuse c = AC) and a copy turned half a turn about the midpoint O of
    AC make the rectangle ABCD.  Its diagonals are equal (each is the
    hypotenuse of a copy of the triangle) and bisect each other at O, so
    OA = OB = OC = OD: the circle about O through A passes through all four
    corners, and the diagonals are diameters.  Ptolemy's theorem for this
    cyclic quadrilateral (proved in E7, cited here), AC·BD = AB·CD + BC·DA,
    reads c·c = a·a + b·b."""

    def construct(self):
        a, b = 2.4, 1.45
        c = float(np.hypot(a, b))
        k = 1.95
        O = np.array([-3.25, 0.35, 0.0])

        def S(p):
            return O + k * to3(p)

        A, B, C, D = S((-a / 2, -b / 2)), S((a / 2, -b / 2)), S((a / 2, b / 2)), S((-a / 2, b / 2))
        Rk = c / 2 * k
        check(abs(np.dot(A - B, C - B)) < 1e-9 and abs(_L(A, B) - a * k) < 1e-9
              and abs(_L(B, C) - b * k) < 1e-9, "legs a, b, right angle at B")
        check(close(2 * O - B, D) and close(2 * O - A, C), "the half-turn about O makes ABCD")
        check(all(abs(_L(O, X) - Rk) < 1e-9 for X in (A, B, C, D)),
              "OA = OB = OC = OD: one circle through the four corners")
        check(abs(_L(A, C) - c * k) < 1e-9 and abs(_L(B, D) - c * k) < 1e-9,
              "both diagonals are c (diameters)")
        check(abs(_L(A, C) * _L(B, D) - (_L(A, B) * _L(C, D) + _L(B, C) * _L(D, A))) < 1e-9,
              "Ptolemy for the rectangle")
        check(abs(c * c - (a * a + b * b)) < 1e-12, "c·c = a·a + b·b")

        # ---- the triangle, completed to a rectangle
        cA, cB, cD = BLUE_B, ORANGE, YELLOW_B
        tri = mk([A, B, C], GREY_D, 0.75, stroke_width=3)
        raB = _ra(B, A, C, 0.2)
        la1 = tag("a", 28).next_to((A + B) / 2, DOWN, buff=0.16)
        lb1 = tag("b", 28).next_to((B + C) / 2, LEFT, buff=0.16)      # inside
        lc1 = _beside(tag("c", 28), A, C, B - D, 0.1, at=0.3)
        self.play(FadeIn(tri), Create(raB), FadeIn(la1), FadeIn(lb1), FadeIn(lc1), run_time=0.9)
        cp = mk([A, B, C], GREY_D, 0.75, stroke_width=3)
        dotO = Dot(O, radius=0.06, color=YELLOW_B)
        self.add(cp)
        self.play(FadeIn(dotO), run_time=0.3)
        self.play(Rotate(cp, angle=PI, about_point=O), run_time=1.3)
        _landed(cp, [C, D, A], "the copy completes the rectangle")
        raD = _ra(D, C, A, 0.2)
        self.play(Create(raD), run_time=0.3)

        # ---- equal diagonals, bisecting each other: a circle through all four
        dAC = Line(A, C, color=YELLOW_B, stroke_width=4)
        dBD = Line(B, D, color=YELLOW_B, stroke_width=4)
        self.play(Create(dAC), Create(dBD), run_time=0.8)
        self.bring_to_front(dotO)
        tks = VGroup(*[_ticks(O, X, 2, YELLOW_B, 0.11, at=0.62) for X in (A, B, C, D)])
        self.play(FadeIn(tks), run_time=0.5)
        circ = Circle(radius=Rk, color=GREY_A, stroke_width=3).move_to(O)
        self.play(Create(circ), run_time=1.2)

        # ---- the three pairs: diagonals, opposite sides a, opposite sides b
        sAB = Line(A, B, color=cA, stroke_width=7)
        sCD = Line(C, D, color=cA, stroke_width=7)
        sBC = Line(B, C, color=cB, stroke_width=7)
        sDA = Line(D, A, color=cB, stroke_width=7)
        self.play(Create(sAB), Create(sCD), Create(sBC), Create(sDA), run_time=0.8)
        self.bring_to_front(raB, raD)
        out = lambda X: O + (Rk + 0.32) * _u(X - O)
        vl = VGroup(*[tag(s, 28).move_to(out(X)) for s, X in zip("ABCD", (A, B, C, D))])
        la2 = tag("a", 28, cA).next_to((C + D) / 2, UP, buff=0.16)
        lb2 = tag("b", 28, cB).next_to((D + A) / 2, RIGHT, buff=0.16)
        lc2 = _beside(tag("c", 28, cD), B, D, C - A, 0.1, at=0.3)
        self.play(FadeIn(vl), la1.animate.set_color(cA), lb1.animate.set_color(cB),
                  lc1.animate.set_color(cD), FadeIn(la2), FadeIn(lb2), FadeIn(lc2), run_time=0.8)
        self.hold(0.4)

        # ---- Ptolemy's relation for ABCD, and what it says here
        W = WHITE
        r1 = _trow([("AC", cD), ("·", W), ("BD", cD), ("=", W), ("AB", cA), ("·", W), ("CD", cA),
                    ("+", W), ("BC", cB), ("·", W), ("DA", cB)], 28, 0.16)
        r1.shift(np.array([3.6 - r1.get_center()[0], 1.0, 0.0]))
        base1 = 1.0
        r2 = VGroup()
        for t1, (s2, col) in zip(r1, [("c", cD), ("·", W), ("c", cD), ("=", W), ("a", cA),
                                       ("·", W), ("a", cA), ("+", W), ("b", cB), ("·", W),
                                       ("b", cB)]):
            t2 = _on_base(s2, 28, col)
            t2.shift(np.array([t1.get_center()[0] - t2.get_center()[0], base1 - 0.85, 0.0]))
            r2.add(t2)
        self.play(FadeIn(r1), run_time=1.0)
        self.hold(0.6)
        self.play(FadeIn(r2), run_time=1.0)

        segs = (_poly_segs([A, B, C, D]) + [(A, C), (B, D)] + _circle_segs(O, Rk)
                + _ra_segs(B, A, C, 0.2) + _ra_segs(D, C, A, 0.2)
                + [s for X in (A, B, C, D) for s in _tick_segs(O, X, 2, 0.11, 0.62)])
        labels = [*vl, la1, la2, lb1, lb2, lc1, lc2, r1, r2]
        _labels_ok(labels, segs, "A27")
        cap = caption("c²  =  a² + b²", 38)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# =================================================================== A26

def _clip_convex(p, d, poly):
    """The part of the line p + t·d inside the convex polygon poly (2D, CCW),
    as two 2D points, or None (Cyrus–Beck)."""
    t0, t1 = -1e9, 1e9
    n = len(poly)
    for i in range(n):
        q0, q1 = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % n], float)
        e = q1 - q0
        nrm = np.array([e[1], -e[0]])                  # outward for a CCW polygon
        den = float(nrm @ d)
        num = float(nrm @ (q0 - p))
        if abs(den) < 1e-12:
            if num < 0:
                return None
            continue
        t = num / den
        if den > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
    if t1 - t0 < 1e-6:
        return None
    return p + t0 * d, p + t1 * d


class A26_LawOfCosines120(Board):
    """The triangle T with sides a, b enclosing 120° (drawn for the
    Eisenstein triple a, b, c = 3, 5, 7, on the triangular grid).  On the
    side c of the equilateral triangle on c, and turned by thirds of a turn
    about its centre, three copies of T fit inside it: at each corner the
    angles α and β of two neighbouring copies fill the 60° corner, so the
    long side b of one copy runs along the short side a of the next, and
    the copies leave an equilateral triangle of side b − a in the middle
    (its angles are 180° − 120°).  Measured in unit triangles, the
    equilateral triangle on s has area s², and T is half of the 60°
    parallelogram on a and b, which holds 2ab of them:
        c² = 3ab + (b − a)² = a² + ab + b²."""

    def construct(self):
        a, b, c = 3, 5, 7
        r3 = float(np.sqrt(3.0))
        e1, e2 = np.array([1.0, 0.0]), np.array([0.5, r3 / 2])

        def rot(p, o, t):
            p, o = np.asarray(p, float), np.asarray(o, float)
            d = p - o
            return o + np.array([d[0] * np.cos(t) - d[1] * np.sin(t),
                                 d[0] * np.sin(t) + d[1] * np.cos(t)])

        Q0 = np.zeros(2)
        P0, P1 = Q0 + b * e1, Q0 + a * (e2 - e1)
        P2 = P0 + rot(P1 - P0, (0, 0), PI / 3)
        Om = (P0 + P1 + P2) / 3
        Q1, Q2 = rot(Q0, Om, 2 * PI / 3), rot(Q0, Om, 4 * PI / 3)
        M2 = (P2 + P0) / 2
        Q2o = 2 * M2 - Q2                                 # copy 2 turned half round

        def is_lattice(p):
            j = p[1] / (r3 / 2)
            i = p[0] - j / 2
            return abs(i - round(i)) < 1e-9 and abs(j - round(j)) < 1e-9

        T0, T1, T2 = [P0, P1, Q0], [P1, P2, Q1], [P2, P0, Q2]
        hole = [Q0, Q1, Q2]
        big = [P0, P1, P2]
        L = lambda p, q: float(np.linalg.norm(np.asarray(p) - np.asarray(q)))
        check(abs(L(P0, Q0) - b) < 1e-9 and abs(L(Q0, P1) - a) < 1e-9
              and abs(L(P0, P1) - c) < 1e-9
              and abs(_ang(Q0, P0, P1) - 2 * PI / 3) < 1e-9, "T: a, b, 120°, c")
        check(all(abs(L(big[i], big[(i + 1) % 3]) - c) < 1e-9 for i in range(3))
              and area(big) > 0, "the equilateral triangle on c")
        check(all(abs(L(hole[i], hole[(i + 1) % 3]) - (b - a)) < 1e-9 for i in range(3)),
              "the hole is equilateral with side b − a")
        check(abs(_cross(Q1 - P1, Q0 - P1)) < 1e-9 and abs(L(P1, Q0) - a) < 1e-9
              and abs(L(P1, Q1) - b) < 1e-9, "at P1: Q0 on P1Q1, a + (b − a) = b")
        check(abs(_ang(P1, P0, Q0) + _ang(P1, Q1, P2) - PI / 3) < 1e-9, "β + α = 60°")
        check(_tiles_exactly([T0, T1, T2, hole], big), "three copies and the hole tile it")
        check(all(is_lattice(p) for p in (P0, P1, P2, Q0, Q1, Q2, Q2o)),
              "all corners on the triangular grid")
        unit = r3 / 4
        check(abs(abs(area(T0)) / unit - a * b) < 1e-9
              and abs(abs(area(hole)) / unit - (b - a) ** 2) < 1e-9
              and abs(abs(area(big)) / unit - c * c) < 1e-9,
              "areas ab, (b−a)², c² in unit triangles")
        check(c * c == 3 * a * b + (b - a) ** 2 == a * a + a * b + b * b,
              "c² = 3ab + (b−a)² = a² + ab + b²")
        par = [P0, Q2o, P2, Q2]
        check(_tiles_exactly([T2, [P0, P2, Q2o]], par)
              and abs(abs(area(par)) / unit - 2 * a * b) < 1e-9,
              "T and its half-turn copy: the 60° parallelogram on a, b (2ab unit triangles)")

        # ---- screen: base P0P1 horizontal at the bottom
        th0 = -float(np.arctan2(*(P1 - P0)[::-1]))
        # a third of a turn about the centre sweeps the whole circumcircle
        # (radius c/√3): it must fit in the safe area, so the centre sits at
        # the middle of the safe area's height
        k = 0.99 * (SAFE_TOP - SAFE_BOTTOM) / 2 / (c / r3)
        G = np.array([-1.05, (SAFE_TOP + SAFE_BOTTOM) / 2, 0.0])

        def S(p):
            q = rot(p, Om, th0) - Om
            return G + k * np.array([q[0], q[1], 0.0])

        sP0, sP1, sP2, sQ0, sQ1, sQ2, sQ2o, sO = (S(p) for p in (P0, P1, P2, Q0, Q1, Q2, Q2o,
                                                                  Om))
        check(abs(sP0[1] - sP1[1]) < 1e-9 and sP2[1] > sP0[1] and sP0[0] < sP1[0],
              "base horizontal, apex up")
        _sweep_ok([S(p) for p in T0], sO, 4 * PI / 3, "the turning copies stay on screen")
        # the half-turn about M2 is done as two folds (over the side P2P0, then
        # over its perpendicular bisector): every point moves in a straight
        # line on screen, so the end positions being on screen is enough
        def mirror(p, u_, v_):
            u_, v_ = np.asarray(u_, float), np.asarray(v_, float)
            d = (v_ - u_) / np.linalg.norm(v_ - u_)
            f = u_ + np.dot(np.asarray(p, float) - u_, d) * d
            return 2 * f - np.asarray(p, float)

        perp2 = M2 + np.array([-(P0 - P2)[1], (P0 - P2)[0]])
        Q2m = mirror(Q2, P2, P0)
        check(close(mirror(Q2m, M2, perp2), Q2o) and close(mirror(P2, M2, perp2), P0),
              "two folds over perpendicular lines through M2 make the half-turn")
        for q in (Q2m, Q2o):
            check(_inside(Dot(S(q), radius=0.01)), "the folded copy stays on screen")

        def grid(poly):
            segs = []
            for m in range(-14, 15):
                for p, d in ((m * e2, e1), (m * e1, e2), (m * e1, e2 - e1)):
                    s_ = _clip_convex(p, d, poly)
                    if s_ is not None:
                        segs.append(Line(S(s_[0]), S(s_[1]), stroke_width=1.3,
                                         color=GREY_A, stroke_opacity=0.5))
            return VGroup(*segs)

        # ---- T on the base, then the equilateral triangle on c
        m0 = mk([S(p) for p in T0], BLUE_D)
        arc0 = angle_arc(sQ0, sP0, sP1, 0.32, YELLOW_B, 4)
        l120 = tag("120°", 22, YELLOW_B).move_to(sQ0 + 0.62 * angle_mid_dir(sQ0, sP0, sP1))
        la = _beside(tag("a", 28), sQ0, sP1, sQ0 - (sP0 + sP1) / 2, 0.1)
        lb = _beside(tag("b", 28), sP0, sQ0, sQ0 - (sP0 + sP1) / 2, 0.1)
        lc = tag("c", 30).next_to(sP0 + 0.33 * (sP1 - sP0), DOWN, buff=0.14)
        self.play(FadeIn(m0), Create(arc0), FadeIn(l120), FadeIn(la), FadeIn(lb), FadeIn(lc),
                  run_time=1.2)
        tri = Polygon(sP0, sP1, sP2, stroke_color=WHITE, stroke_width=4)
        tks = VGroup(*[_ticks(p, q, 1, WHITE, 0.13) for p, q in ((sP0, sP1), (sP1, sP2),
                                                                 (sP2, sP0))])
        self.play(Create(tri), FadeIn(tks), run_time=1.0)
        self.hold(0.3)

        # ---- copies turned by thirds of a turn about the centre
        dotO = Dot(sO, radius=0.06, color=YELLOW_B)
        arr = _turn_arrow(sO, -PI / 2 + 0.3, -PI / 2 + 2 * PI / 3 - 0.3, 0.36, YELLOW_B, 3)
        self.play(FadeOut(la), FadeOut(lb), FadeOut(l120), FadeOut(arc0), FadeIn(dotO),
                  Create(arr), run_time=0.6)
        m1 = mk([S(p) for p in T0], TEAL_D)
        self.add(m1)
        self.bring_to_front(tri, dotO)
        self.play(Rotate(m1, angle=2 * PI / 3, about_point=sO), run_time=1.5)
        _landed(m1, [S(p) for p in T1], "a third of a turn: copy on the next side")
        m2 = mk([S(p) for p in T0], GREEN_D)
        self.add(m2)
        self.bring_to_front(tri, dotO)
        self.play(Rotate(m2, angle=4 * PI / 3, about_point=sO), run_time=1.8)
        _landed(m2, [S(p) for p in T2], "two thirds of a turn: copy on the third side")
        self.play(FadeOut(arr), FadeOut(dotO), run_time=0.3)

        # ---- at each corner α + β = 60°; the hole has side b − a
        aB = angle_arc(sP1, sP0, sQ0, 0.95, PINK, 4)
        aA = angle_arc(sP1, sQ1, sP2, 1.15, ORANGE, 4)
        self.play(Create(aB), Create(aA), run_time=0.7)
        mh = mk([S(p) for p in hole], YELLOW_E, 0.92)
        self.add(mh)
        self.bring_to_front(tri)
        self.play(FadeIn(mh), run_time=0.6)
        # copy 0's long side b: the short side a of copy 2, then a side of the hole
        n0 = _u(np.array([-(sQ0 - sP0)[1], (sQ0 - sP0)[0], 0.0]))   # into copy 0
        if np.dot(n0, (sP0 + sP1 + sQ0) / 3 - sP0) < 0:
            n0 = -n0
        d1 = _dim(sP0, sQ2 - 0.03 * _u(sQ2 - sP0), -n0, "a", WHITE, 24, off=0.22)   # in copy 2
        d2 = _dim(sQ2 + 0.03 * _u(sQ2 - sP0), sQ0, n0, "b − a", YELLOW_B, 24, off=0.22)
        self.play(FadeIn(d1), FadeIn(d2), run_time=0.7)
        self.hold(0.6)

        # ---- measure in unit triangles: T is half the parallelogram on a, b
        self.play(FadeOut(aB), FadeOut(aA), run_time=0.4)
        gb = grid(big)
        out = DashedVMobject(Polygon(sP0, sQ2o, sP2, stroke_color=GREEN_B, stroke_width=3),
                             num_dashes=40)
        go = grid([P0, Q2o, P2] if area([P0, Q2o, P2]) > 0 else [P0, P2, Q2o])
        self.play(Create(gb), run_time=1.0)
        cp2 = mk([S(p) for p in T2], GREEN_D, 0.35, stroke_width=0)
        self.add(cp2)
        self.play(_fold(cp2, sP2, sP0), run_time=1.1)
        _landed(cp2, [S(p) for p in (P2, P0, Q2m)], "copy 2 turned over its side")
        u2 = _u(S(perp2) - S(M2))
        ax2 = DashedLine(S(M2) + 0.45 * u2, S(M2) - 0.45 * u2, color=WHITE, stroke_width=2.5,
                         dash_length=0.07)
        self.play(Create(ax2), run_time=0.3)
        self.play(_fold(cp2, S(M2), S(perp2)), run_time=1.1)
        _landed(cp2, [S(p) for p in (P0, P2, Q2o)],
                "copy 2 turned half round about the midpoint")
        self.play(FadeOut(ax2), run_time=0.3)
        self.play(Create(out), Create(go), run_time=0.8)
        oa = _beside(tag("a", 28), sQ2o, sP2, sQ2o - S(M2), 0.1)
        ob = _beside(tag("b", 28), sP0, sQ2o, sQ2o - S(M2), 0.1)
        arcO = angle_arc(sQ2o, sP0, sP2, 0.32, YELLOW_B, 4)
        lO = _lbl("120°", 20, YELLOW_B, 0.7)
        lO.move_to(sQ2o + 0.84 * angle_mid_dir(sQ2o, sP0, sP2))
        l2ab = _lbl("2ab", 26, WHITE, 0.7).move_to(S(M2))
        self.play(FadeIn(oa), FadeIn(ob), Create(arcO), FadeIn(lO), FadeIn(l2ab), run_time=0.8)
        self.hold(0.4)

        # ---- areas: ab, ab, ab and (b − a)²; the whole is c²
        def incentre(T):
            A_, B_, C_ = T
            wa, wb, wc = L(B_, C_), L(C_, A_), L(A_, B_)
            return ((wa * np.asarray(A_) + wb * np.asarray(B_) + wc * np.asarray(C_))
                    / (wa + wb + wc))

        # one 'ab' in each copy, clear of its edges, of the bars a | b − a and
        # of the other labels: the free spot nearest the copy's incentre
        edges = [(S(p), S(q)) for T in (T0, T1, T2) for p, q in zip(T, T[1:] + T[:1])]
        bars = _dim_segs(d1) + _dim_segs(d2)
        placed = [d1[1], d2[1], l2ab, lO]
        lab = VGroup()
        for T in (T0, T1, T2):
            m = _lbl("ab", 26, WHITE, 0.65)
            best = None
            for w0 in np.arange(0.05, 0.91, 0.05):
                for w1 in np.arange(0.05, 0.96 - w0, 0.05):
                    X = w0 * T[0] + w1 * T[1] + (1 - w0 - w1) * T[2]
                    m.move_to(S(X))
                    if (_hit(m, edges + bars, 0.05) is None
                            and not any(_overlap(m, o, 0.32) for o in placed)
                            and all(_pip(q[:2], [S(p)[:2] for p in T]) for q in
                                    (m.get_corner(UL), m.get_corner(DR)))):
                        dist = float(np.linalg.norm(X - incentre(T)))
                        if best is None or dist < best[0]:
                            best = (dist, X)
            check(best is not None, "a free spot for 'ab' in each copy")
            m.move_to(S(best[1]))
            placed.append(m)
            lab.add(m)
        th_ = tag("(b−a)²", 18)
        lh = VGroup(BackgroundRectangle(th_, color=BLACK, fill_opacity=0.7, buff=0.035),
                    th_).move_to(S(Om))
        unitT = [S(p) for p in (np.zeros(2), e1, e2)]
        leg_o = np.array([4.55, 3.05, 0.0]) - (unitT[0] + unitT[1] + unitT[2]) / 3
        leg = mk([p + leg_o for p in unitT], GREY_D, 0.9, stroke_color=GREY_A, stroke_width=2)
        leg_l = tag("=  1", 26).next_to(leg, RIGHT, buff=0.2)
        self.play(FadeIn(lab), FadeIn(lh), FadeIn(leg), FadeIn(leg_l), run_time=0.9)
        R1 = tag("c²  =  3ab + (b − a)²", 28)
        R2 = tag("=  a² + ab + b²", 28, YELLOW_B)
        R1.move_to([4.15, 1.45, 0])
        R2.next_to(R1, DOWN, buff=0.3)
        R2.shift((R1[2].get_left()[0] - R2[0].get_left()[0]) * RIGHT)
        self.play(FadeIn(R1), run_time=0.7)
        self.play(FadeIn(R2), run_time=0.7)

        segs = (_poly_segs([sP0, sP1, sP2]) + _poly_segs([S(p) for p in hole])
                + [(S(P0), S(Q0)), (S(Q0), S(P1)), (S(P1), S(Q1)), (S(Q1), S(P2)),
                   (S(P2), S(Q2)), (S(Q2), S(P0)), (sP0, sQ2o), (sQ2o, sP2)]
                + _dim_segs(d1) + _dim_segs(d2) + _angle_segs(sQ2o, sP0, sP2, 0.32)
                + [s for p, q in ((sP0, sP1), (sP1, sP2), (sP2, sP0))
                   for s in _tick_segs(p, q, 1, 0.13)])
        labels = [lc, oa, ob, lO, d1[1], d2[1], R1, R2, leg_l]
        _labels_ok(labels, segs, "A26")
        _labels_ok([*lab], edges + bars, "A26 area labels", pad=0.03)
        hole_s = [S(p)[:2] for p in hole]
        check(all(_pip(q, hole_s) for q in (lh.get_corner(UL)[:2], lh.get_corner(UR)[:2],
                                             lh.get_corner(DL)[:2], lh.get_corner(DR)[:2])),
              "(b−a)² sits inside the hole")
        check(all(_pip(m.get_center()[:2], [S(p)[:2] for p in T])
                  for m, T in zip(lab, (T0, T1, T2))), "each 'ab' sits in its copy")
        cap = caption("c²  =  3ab + (b − a)²  =  a² + ab + b²", 34)
        _final_check(self, cap)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)
