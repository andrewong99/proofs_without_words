# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2h.py — proofs without words, 2D (manim):
#     F28 area of a regular polygon        F30 area of a regular hexagon
#     F43 area of a regular dodecagon      F39 a triangle in a parallelogram
#     F40 diagonals of a parallelogram     F49 a point inside a parallelogram
#     F53 the British flag theorem         F64 every triangle and quadrilateral tiles
#     G16 three arctangents make π         G17 two arctangents make π/4
#
# Every move of a piece is rigid — a slide (shift), a half-turn or other
# turn about a named point (Rotate), or a turn about a moving centre that
# stays rigid on every frame — except the shear of F39, which keeps its base
# and its pair of parallels. Every landing, tiling, area and angle claim is
# checked with check(...) before it is drawn, so a wrong construction fails
# the render instead of drawing a wrong picture. No LaTeX: every label is
# Text (tag / caption) with Unicode.


# ------------------------------------------------------------ private helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _polar(t):
    return np.array([np.cos(t), np.sin(t), 0.0])


def _ang_of(v):
    """Direction angle of a 2D/3D vector."""
    return float(np.arctan2(v[1], v[0]))


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _angle(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _turn(p, c, th):
    """Point p turned by th about c, in the plane."""
    d = to3(p) - to3(c)
    cs, sn = np.cos(th), np.sin(th)
    return to3(c) + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1], 0.0])


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


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ticks(p, q, n=1, color=WHITE, size=0.12, width=2.5, at=0.5):
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


def _chevron(p, q, color=YELLOW_B, size=0.16, width=3, n=1, at=0.5):
    """Parallel-line mark: n small '>' on segment pq, pointing p -> q."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = p + (q - p) * at
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7,
                                          tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


def _darrow(p, q, color=YELLOW_B, width=3):
    """Double arrow p <-> q (a height between two parallels)."""
    return DoubleArrow(to3(p), to3(q), buff=0, stroke_width=width,
                       color=color, tip_length=0.16,
                       max_tip_length_to_length_ratio=0.5)


def _bracket(p, q, nrm, text, gap=0.28, color=YELLOW_B, size=28, tick=0.11,
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


def _dashed(pts, color=GREY_B, width=2, dashes=40):
    return DashedVMobject(Polygon(*[to3(p) for p in pts], stroke_color=color,
                                  stroke_width=width), num_dashes=dashes)


def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centre while that
    centre slides straight — carrying the screen points src onto dst (same
    order). Fails the render unless one rotation + translation lands every
    point. `turn` (radians) picks the direction for ±180°."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = _ang_of(src[1] - src[0])
        a1 = _ang_of(dst[1] - dst[0])
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    check(all(close(m1 + _turn(p, m0, turn) - m0, q, 1e-6)
              for p, q in zip(src, dst)), "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _sweep_ok(pts, pivot, angle, what, n=72):
    """The convex piece with corners pts stays in the safe area all the way
    through a turn by `angle` about pivot."""
    for k in range(n + 1):
        for p in pts:
            q = _turn(p, pivot, angle * k / n)
            check(abs(q[0]) <= SAFE_X and SAFE_BOTTOM <= q[1] <= SAFE_TOP, what)


def _inside(m, tol=1e-6):
    return (m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
            and m.get_bottom()[1] >= SAFE_BOTTOM - tol
            and m.get_top()[1] <= SAFE_TOP + tol)


def _apart(a, b, pad=0.04):
    a0, a1 = a.get_corner(DL), a.get_corner(UR)
    b0, b1 = b.get_corner(DL), b.get_corner(UR)
    return (a1[0] + pad < b0[0] or b1[0] + pad < a0[0]
            or a1[1] + pad < b0[1] or b1[1] + pad < a0[1])


def _clear_of(m, p, q, gap=0.05):
    """True if the segment pq misses m's bounding box (grown by gap)."""
    lo, hi = m.get_corner(DL), m.get_corner(UR)
    p, q = to3(p), to3(q)
    L = float(np.linalg.norm(q - p))
    for f in np.linspace(0.0, 1.0, max(2, int(L / 0.01) + 2)):
        x = p + (q - p) * f
        if lo[0] - gap < x[0] < hi[0] + gap and lo[1] - gap < x[1] < hi[1] + gap:
            return False
    return True


def _labels_clear(labels, segs, what, gap=0.05):
    """Every label clear of every segment in segs (screen point pairs)."""
    for lab in labels:
        for i, (p, q) in enumerate(segs):
            check(_clear_of(lab, p, q, gap),
                  f"{what}: '{getattr(lab, 'text', '?')}' clear of line #{i}")


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption in its band, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        check(_inside(m, 0.02), f"{type(m).__name__} inside the safe area")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_apart(texts[i], texts[j], 0.02),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# =========================================================== F28 POLYGON

class F28_RegularPolygonArea(Board):
    """Spokes from the centre cut the regular n-gon (here n = 7) into n
    congruent triangles, each with a side s as base and the apothem a as
    height. Laid in a row their bases make the perimeter p = ns, their apexes
    sit at height a. A copy of each, turned half a turn about the midpoint
    of its right side, drops into the gap beside it (as two triangles make a
    parallelogram); the last copy overhangs by half a triangle, which slides
    back by p into the half gap at the left end. The row and its copies now
    fill the p × a rectangle exactly: 2A = p·a."""

    def construct(self):
        n = 7
        Rs = 1.8
        s = 2 * Rs * np.sin(PI / n)
        a = Rs * np.cos(PI / n)
        p = n * s
        O = np.array([0.0, SAFE_TOP - 0.1 - Rs, 0.0])
        V = [O + Rs * _polar(-PI / 2 - PI / n + TAU * k / n) for k in range(n)]
        side = [(V[j], V[(j + 1) % n]) for j in range(n)]
        yb, x0 = -1.95, -p / 2
        Rect = [np.array([x0, yb, 0]), np.array([x0 + p, yb, 0]),
                np.array([x0 + p, yb + a, 0]), np.array([x0, yb + a, 0])]

        def slot(i):                      # row triangle i: base-left, base-right, apex
            return [np.array([x0 + i * s, yb, 0.0]),
                    np.array([x0 + (i + 1) * s, yb, 0.0]),
                    np.array([x0 + (i + 0.5) * s, yb + a, 0.0])]

        order = [(i - n // 2) % n for i in range(n)]   # side j at row place i
        tris = {j: [side[j][0], side[j][1], O] for j in range(n)}

        # ---- checks: a regular polygon, n congruent triangles, the tiling
        for j in range(n):
            check(abs(np.linalg.norm(side[j][1] - side[j][0]) - s) < 1e-9,
                  "every side is s")
            foot = (side[j][0] + side[j][1]) / 2
            check(abs(np.linalg.norm(foot - O) - a) < 1e-9
                  and abs(np.dot(foot - O, side[j][1] - side[j][0])) < 1e-9,
                  "every apothem is a, perpendicular to its side")
        check(_tiles_exactly([tris[j] for j in range(n)], V),
              "the n triangles tile the polygon")
        mids = [(slot(i)[1] + slot(i)[2]) / 2 for i in range(n)]
        copies = [[2 * mids[i] - q for q in slot(i)] for i in range(n)]
        for i in range(n - 1):
            gap = [slot(i)[2], slot(i + 1)[2], slot(i)[1]]
            check(_same_poly(copies[i], gap), "each copy fills the gap beside it")
        last = copies[n - 1]
        X1 = np.array([x0 + p, yb, 0.0])
        X2 = np.array([x0 + p, yb + a, 0.0])
        inner = [X1, X2, last[1]]
        outer = [X1, last[0], X2]
        check(close(last[2], X1), "the last copy's foot is the row's end")
        check(abs(area(inner) + area(outer) - area(last)) < 1e-9
              and close((last[0] + last[1]) / 2, X2), "the edge x = p halves it")
        moved = [q - np.array([p, 0, 0]) for q in outer]
        check(_same_poly(moved, [Rect[0], slot(0)[2], Rect[3]]),
              "the overhang, slid back by p, fills the left half gap")
        fill = ([slot(i) for i in range(n)] + copies[:n - 1] + [inner, moved])
        check(_tiles_exactly(fill, Rect), "row + copies tile the p × a rectangle")
        check(abs(area(V) - 0.5 * p * a) < 1e-9, "A = ½ p a")
        for i in range(n):
            _sweep_ok(slot(i), mids[i], PI, "a turning copy stays on screen")

        # ---- the polygon, its apothem and side, the spokes
        cols = [BLUE_D if i % 2 == 0 else TEAL_D for i in range(n)]
        colour_of = {order[i]: cols[i] for i in range(n)}
        whole = mk(V, BLUE_D, stroke_width=0)
        tri_m = {j: mk(tris[j], BLUE_D, stroke_width=2) for j in range(n)}
        bases = {j: Line(side[j][0], side[j][1], color=YELLOW_B, stroke_width=5)
                 for j in range(n)}
        dotO = Dot(O, radius=0.06)
        foot0 = (V[0] + V[1]) / 2
        apo = Line(O, foot0, color=WHITE, stroke_width=3)
        ra0 = _ra(foot0, O, V[1], 0.18)
        la = tag("a", 30).move_to(foot0 + 0.36 * a * UP + 0.24 * LEFT)
        m1 = (V[1] + V[2]) / 2
        ls = tag("s", 30, YELLOW_B).move_to(m1 + 0.34 * _unit(m1 - O))
        spokes = VGroup(*[Line(O, V[k], color=WHITE, stroke_width=2)
                          for k in range(n)])

        self.play(FadeIn(whole), *[Create(b) for b in bases.values()],
                  FadeIn(dotO), run_time=1.1)
        self.play(FadeIn(ls), Create(apo), Create(ra0), FadeIn(la), run_time=0.9)
        self.play(Create(spokes), run_time=0.9)
        self.remove(whole, spokes)
        self.add(*tri_m.values(), *bases.values(), apo, ra0, dotO, la, ls)
        self.play(*[tri_m[j].animate.set_fill(colour_of[j]) for j in range(n)],
                  run_time=0.5)
        self.hold(0.4)

        # ---- lay the n triangles in a row: bases end to end (the perimeter)
        ghost = _dashed(V, GREY_B, 2, 56)
        g_apo = DashedLine(O, foot0, color=GREY_B, stroke_width=2,
                           dash_length=0.08)
        self.add(ghost, g_apo)
        self.bring_to_back(ghost)
        self.remove(apo)
        groups = {j: VGroup(tri_m[j], bases[j]) for j in range(n)}
        moves = []
        for i, j in enumerate(order):
            src = [side[j][0], side[j][1], O]
            moves.append(_rigid(groups[j], src, slot(i), run_time=2.2))
        self.play(FadeOut(dotO), run_time=0.2)
        self.play(LaggedStart(*moves, lag_ratio=0.12))
        for i, j in enumerate(order):
            _landed(tri_m[j], slot(i), "each triangle lands in its place")
        lA = tag("A", 34).move_to(O + 0.62 * UP)
        self.play(FadeIn(lA), run_time=0.5)

        # ---- the p × a box around the row
        box = DashedVMobject(Polygon(*Rect, stroke_color=YELLOW_B,
                                     stroke_width=2.5), num_dashes=70)
        harr = _darrow(Rect[0] + 0.36 * LEFT, Rect[3] + 0.36 * LEFT)
        lha = tag("a", 30, YELLOW_B).next_to(harr, LEFT, buff=0.12)
        pbar, lp = _bracket(Rect[0], Rect[1], DOWN, "p", gap=0.3, size=30)
        self.play(Create(box), GrowFromCenter(harr), FadeIn(lha),
                  FadeIn(pbar), FadeIn(lp), run_time=1.1)
        self.hold(0.4)

        # ---- copies interleave: half a turn about each right side's midpoint
        dots = VGroup(*[Dot(m, radius=0.05, color=WHITE) for m in mids])
        cps = [mk(slot(i), ORANGE, stroke_width=2) for i in range(n)]
        self.play(FadeIn(dots), run_time=0.4)
        self.add(*cps)
        self.bring_to_front(dots)
        self.play(LaggedStart(*[Rotate(c, angle=-PI, about_point=m)
                                for c, m in zip(cps, mids)], lag_ratio=0.1),
                  run_time=2.6)
        for i in range(n):
            _landed(cps[i], copies[i], "each copy lands in its gap")
        self.play(FadeOut(dots), run_time=0.3)

        # ---- the overhang at the right end slides back by p
        cut = DashedLine(X1, X2 + 0.25 * UP, color=WHITE, stroke_width=2.5,
                         dash_length=0.07)
        pin = mk(inner, ORANGE, stroke_width=2)
        pout = mk(outer, ORANGE, stroke_width=2)
        self.play(Create(cut), run_time=0.5)
        self.remove(cps[-1])
        self.add(pin, pout)
        self.play(pout.animate.set_fill(GOLD_D), run_time=0.3)
        self.play(pout.animate.shift(-p * RIGHT), run_time=1.6)
        _landed(pout, moved, "the overhang fills the left end")
        self.play(FadeOut(cut), pout.animate.set_fill(ORANGE), run_time=0.4)

        frame = Polygon(*Rect, stroke_color=YELLOW_B, stroke_width=5)
        self.play(FadeOut(box), Create(frame), run_time=0.8)
        self.bring_to_front(harr, lha, pbar, lp)

        for lab in (la, ls, lha, lp, lA):
            check(_inside(lab), "labels inside the safe area")
        _labels_clear([la], [(O, foot0), (O, V[0]), (V[0], V[1])], "F28 a")
        _labels_clear([ls], [(V[1], V[2]), (V[2], V[3])], "F28 s")
        cap = caption("A  =  ½ · p · a", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


def _path_ok(pts, m0, m1, turn, what, n=60):
    """Every point stays in the safe area through the rigid motion of
    _rigid: a turn about the centre m0 while m0 slides to m1."""
    m0, m1 = to3(m0), to3(m1)
    for k in range(n + 1):
        al = k / n
        for p in pts:
            q = _turn(p, m0, al * turn) + al * (m1 - m0)
            check(abs(q[0]) <= SAFE_X and SAFE_BOTTOM <= q[1] <= SAFE_TOP, what)


# =========================================================== F30 HEXAGON

class F30_RegularHexagonArea(Board):
    """The six spokes make six triangles with 60° at the centre and two
    equal sides, so all six are equilateral with side s and height h, where
    h² + (s/2)² = s². The long diagonal halves the hexagon into two strips
    of three triangles. The lower strip slides right along the diagonal's
    line and then up along the edge it will share, so it lies beside the
    upper strip: a parallelogram of base 3s and height h. Half a triangle,
    cut off one end along its altitude and slid by 3s, fills the other end:
    a 3s × h rectangle, A = 3s · (√3/2)s."""

    def construct(self):
        s = 2.3
        h = s * np.sqrt(3) / 2
        O = np.array([-s, -0.45, 0.0])
        V = [O + s * _polar(PI / 3 * k) for k in range(6)]
        R_ = RIGHT * s
        t1 = 2 * s * RIGHT                           # slide along the diagonal
        t2 = V[1] - V[0]                             # then up along the edge
        tt = t1 + t2
        K = (O + V[3]) / 2                           # foot of V2's altitude
        T = (V[1] + V[2]) / 2                        # top of the h in O V1 V2
        tris = [[O, V[k], V[(k + 1) % 6]] for k in range(6)]
        upper, lower = tris[:3], tris[3:]
        endL = [V[3], K, V[2]]                       # cut off the left end
        endR = [q + 3 * R_ for q in endL]
        Rect = [K, K + 3 * R_, K + 3 * R_ + h * UP, V[2]]
        Para = [V[3], V[3] + 3 * R_, V[2] + 3 * R_, V[2]]

        for k in range(6):
            P_ = tris[k]
            for i in range(3):
                check(abs(np.linalg.norm(P_[i] - P_[(i + 1) % 3]) - s) < 1e-9,
                      "six equilateral triangles of side s")
        check(abs(h * h + (s / 2) ** 2 - s * s) < 1e-9, "h² + (s/2)² = s²")
        check(abs(np.linalg.norm(T - O) - h) < 1e-9
              and abs(np.dot(T - O, V[2] - V[1])) < 1e-9, "OT is the height h")
        check(close(tt, np.array([1.5 * s, h, 0.0])), "the strip moves by (3s/2, h)")
        check(abs(_cross(t2, V[0] - V[1])) < 1e-9,
              "the second slide runs along the shared edge")
        check(close(V[3] + tt, V[1]) and close(V[4] + tt, V[0]),
              "the strip's slanted end lands on the upper strip's end")
        # after the first slide the strip's left edge lies on the line V0V1
        check(abs(_cross(V[4] + t1 - V[0], V[1] - V[0])) < 1e-9
              and close(V[3] + t1, V[0]), "first slide: left edge on line V0V1")
        moved = [[q + tt for q in P_] for P_ in lower]
        check(_tiles_exactly(upper + moved, Para), "two strips: a parallelogram")
        check(_tiles_exactly([[O, V[0], V[1]], [O, V[1], V[2]], [K, O, V[2]]]
                             + moved + [endR], Rect),
              "after the end slides: a 3s × h rectangle")
        check(abs(abs(area(V)) - 3 * s * h) < 1e-9
              and abs(3 * s * h - 1.5 * np.sqrt(3) * s * s) < 1e-9,
              "A = 3s · (√3/2)s = (3√3/2)s²")
        for P_ in lower:
            for q in P_:
                for al in np.linspace(0, 1, 21):
                    for x in (q + al * t1, q + t1 + al * t2):
                        check(abs(x[0]) <= SAFE_X and x[1] >= SAFE_BOTTOM,
                              "the sliding strip stays on screen")

        # ---- the hexagon, its six equilateral triangles
        whole = mk(V, BLUE_D, stroke_width=0)
        rim = Polygon(*V, stroke_color=WHITE, stroke_width=2.5)
        dotO = Dot(O, radius=0.06)
        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D, BLUE_D, TEAL_D]
        tri_m = [mk(P_, BLUE_D, stroke_width=2) for P_ in tris]
        spokes = VGroup(*[Line(O, V[k], color=WHITE, stroke_width=2)
                          for k in range(6)])
        a60 = angle_arc(O, V[0], V[1], radius=0.5, color=YELLOW_B, width=4)
        l60 = tag("60°", 22, YELLOW_B).move_to(O + 0.86 * _polar(PI / 6))
        tks = VGroup(*[_ticks(O, V[k], 1, at=0.62) for k in range(6)])
        ls = tag("s", 30).move_to((V[0] + V[5]) / 2 + 0.3 * _unit((V[0] + V[5]) / 2 - O))

        self.play(FadeIn(whole), Create(rim), FadeIn(dotO), run_time=1.0)
        self.play(Create(spokes), run_time=0.8)
        self.remove(whole, spokes)
        self.add(*tri_m, rim, dotO)
        self.play(*[m.animate.set_fill(c) for m, c in zip(tri_m, cols)],
                  Create(a60), FadeIn(l60), FadeIn(tks), FadeIn(ls), run_time=1.0)
        self.hold(0.5)

        # ---- the height h of one triangle, by Pythagoras
        alt = DashedLine(O, T, color=WHITE, stroke_width=2.5, dash_length=0.08)
        raT = _ra(T, O, V[1], 0.16)
        lh = tag("h", 30).move_to(O + 0.5 * (T - O) + 0.21 * LEFT)
        lhalf = tag("s/2", 22).move_to((T + V[1]) / 2 + 0.26 * UP)
        r1 = tag("h² + (s/2)²  =  s²", 28)
        r2 = tag("h  =  (√3/2) s", 28, YELLOW_B)
        r1.move_to(np.array([3.15, 3.3, 0.0]))
        r2.next_to(r1, DOWN, buff=0.3)
        self.play(Create(alt), Create(raT), FadeIn(lh), FadeIn(lhalf),
                  FadeOut(a60), FadeOut(l60), run_time=0.9)
        self.play(FadeIn(r1), run_time=0.6)
        self.play(FadeIn(r2), run_time=0.6)
        self.hold(0.4)

        # ---- the lower strip slides along the diagonal, then up the edge
        diag = Line(V[3], V[0], color=YELLOW_B, stroke_width=4)
        self.play(Create(diag), FadeOut(tks), FadeOut(ls), run_time=0.6)
        strip = VGroup(*tri_m[3:])
        ghost = _dashed(V, GREY_B, 2, 48)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.remove(rim)
        self.bring_to_front(*tri_m[:3], alt, raT, lh, lhalf, dotO)
        self.play(strip.animate.shift(t1), FadeOut(diag), run_time=1.4)
        edge = DashedLine(V[0] + 0.25 * (V[0] - V[1]), V[1] + 0.25 * (V[1] - V[0]),
                          color=YELLOW_B, stroke_width=2.5, dash_length=0.08)
        self.play(Create(edge), run_time=0.4)
        self.play(strip.animate.shift(t2), run_time=1.3)
        for m, P_ in zip(tri_m[3:], moved):
            _landed(m, P_, "the lower strip lands beside the upper one")
        self.play(FadeOut(edge), FadeOut(dotO), FadeOut(ghost), run_time=0.5)

        # ---- half a triangle off the left end, slid by 3s to the right end
        cutK = DashedLine(V[2], K, color=WHITE, stroke_width=2.5, dash_length=0.08)
        self.play(Create(cutK), run_time=0.5)
        keep = mk([K, O, V[2]], cols[2], stroke_width=2)
        endm = mk(endL, cols[2], stroke_width=2)
        self.remove(tri_m[2])
        self.add(keep, endm)
        self.bring_to_front(alt, raT, lh, lhalf)
        self.play(endm.animate.set_fill(GOLD_D), FadeOut(cutK), run_time=0.4)
        self.play(endm.animate.shift(3 * R_), run_time=1.5)
        _landed(endm, endR, "the half triangle fills the right end")
        self.play(endm.animate.set_fill(cols[2]), run_time=0.3)

        frame = Polygon(*Rect, stroke_color=YELLOW_B, stroke_width=5)
        bar, l3s = _bracket(Rect[0], Rect[1], DOWN, "3s", gap=0.3, size=30)
        harr = _darrow(Rect[1] + 0.36 * RIGHT, Rect[2] + 0.36 * RIGHT)
        lr = tag("(√3/2) s", 28, YELLOW_B).next_to(harr, RIGHT, buff=0.14)
        self.play(Create(frame), FadeIn(bar), FadeIn(l3s), GrowFromCenter(harr),
                  FadeIn(lr), run_time=1.0)
        _labels_clear([lh], [(O, T), (O, V[1]), (O, V[2]), (V[2], K)], "F30 h")
        _labels_clear([lhalf], [(V[1], V[2]), (O, V[1])], "F30 s/2")
        cap = caption("A  =  3s · (√3/2) s  =  (3√3/2) s²", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F43 DODECAGON

class F43_RegularDodecagonArea(Board):
    """Three spokes cut the dodecagon of circumradius r into three
    congruent thirds. Every second vertex makes the inscribed regular
    hexagon of side r (a chord of 60° equals the radius), so each third is
    a 60° rhombus of side r — two equilateral triangles — with a thin cap
    on each of its two outer sides. One altitude of an equilateral triangle
    cuts the rhombus; the cut-off half slides by r to the other side: an
    r × (√3/2)r rectangle. One cap stays on it; the other, cut along its
    axis, fills the two corners of the strip of height r − (√3/2)r above
    it. Each third becomes an r × r square, so A = 3r²."""

    def construct(self):
        R = 1.95
        C = np.array([-4.4, 0.35, 0.0])
        V = [C + R * _polar(PI / 6 + PI / 6 * k) for k in range(12)]
        th = [-PI / 6, -5 * PI / 6, PI / 2]          # turn of each third
        S = [np.array([2.6, 1.65 - k * (R + 0.3), 0.0]) for k in range(3)]

        def to_slot(k, X):
            return S[k] + _turn(X, C, th[k]) - C

        def pieces(k):
            v = [V[(4 * k + j) % 12] for j in range(5)]
            M, N = (C + v[0]) / 2, (v[0] + v[2]) / 2
            return v, M, N, [[C, M, v[2], v[4]], [M, v[0], v[2]],
                             [v[2], v[3], v[4]], [v[0], N, v[1]], [N, v[2], v[1]]]

        # ---- checks: the dissection of each third and the square it makes
        for k in range(3):
            v, M, N, P5 = pieces(k)
            third = [C] + v
            for q in (v[0], v[2], v[4]):
                check(abs(np.linalg.norm(q - C) - R) < 1e-9, "spokes are r")
            check(abs(np.linalg.norm(v[2] - v[0]) - R) < 1e-9
                  and abs(np.linalg.norm(v[4] - v[2]) - R) < 1e-9,
                  "a chord of 60° is r: the rhombus has side r")
            check(abs(np.dot(v[2] - M, v[0] - C)) < 1e-9, "the cut is an altitude")
            check(abs(np.dot(v[1] - N, v[2] - v[0])) < 1e-9
                  and abs(_cross(v[1] - C, N - C)) < 1e-9,
                  "the cap's axis lies on the radius through its apex")
            check(_tiles_exactly(P5, third), "five pieces tile the third")
            Sq = [S[k] + R * np.array(d) for d in
                  ((-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0))]
            check(close(to_slot(k, C), S[k]) and close(to_slot(k, v[0]), S[k] + R * RIGHT),
                  "each third arrives with its first spoke along the x-axis")
            Pl = [[to_slot(k, q) for q in P_] for P_ in P5]
            P2t = [q - R * RIGHT for q in Pl[1]]
            CR, CL = Sq[2], Sq[3]
            V3s, V2s, V4s = Pl[2][1], Pl[2][0], Pl[2][2]
            P4t, P5t = [V3s, CR, V2s], [CL, V3s, V4s]
            check(_same_poly(P2t, [Sq[0], S[k], V4s]),
                  "the cut-off half slides by r to the other side")
            check(_tiles_exactly([Pl[0], P2t, Pl[2], P4t, P5t], Sq),
                  "the five pieces tile the r × r square")
            check(abs(abs(area(third)) - R * R) < 1e-9, "a third has area r²")
        check(abs(area(V) - 3 * R * R) < 1e-9, "A = 3r²")

        # ---- the dodecagon, its radius r
        cols = [BLUE_D, TEAL_D, ORANGE]
        whole = mk(V, BLUE_D, stroke_width=0)
        rim = Polygon(*V, stroke_color=WHITE, stroke_width=2.5)
        dotC = Dot(C, radius=0.06)
        rad = Line(C, V[8], color=YELLOW_B, stroke_width=4)
        lr = tag("r", 30, YELLOW_B).move_to((C + V[8]) / 2 + 0.26 * LEFT)
        self.play(FadeIn(whole), Create(rim), FadeIn(dotC), run_time=1.0)
        self.play(Create(rad), FadeIn(lr), run_time=0.6)

        # ---- three thirds
        thirds = [mk([C] + pieces(k)[0], BLUE_D, stroke_width=0) for k in range(3)]
        spokes = VGroup(*[Line(C, V[4 * k], color=WHITE, stroke_width=2.5)
                          for k in range(3)])
        self.play(Create(spokes), run_time=0.8)
        self.remove(whole)
        self.add(*thirds)
        self.bring_to_back(*thirds)
        self.play(*[t.animate.set_fill(c) for t, c in zip(thirds, cols)], run_time=0.6)

        # ---- the inscribed hexagon of side r: a rhombus and two caps per third
        chords = VGroup(*[Line(V[2 * j], V[(2 * j + 2) % 12], color=WHITE,
                               stroke_width=2.5) for j in range(6)])
        v0, M0, N0, _ = pieces(0)
        tk = dict(size=0.15, width=3.5)
        tks = VGroup(_ticks(C, v0[0], 1, YELLOW_B, at=0.7, **tk),
                     _ticks(v0[0], v0[2], 1, YELLOW_B, **tk),
                     _ticks(v0[2], v0[4], 1, YELLOW_B, **tk),
                     _ticks(v0[4], C, 1, YELLOW_B, at=0.3, **tk))
        self.play(Create(chords), run_time=1.0)
        self.play(FadeIn(tks), run_time=0.5)
        self.hold(0.3)

        # ---- the cuts: an altitude of the rhombus, the axis of one cap
        cuts = VGroup()
        for k in range(3):
            v, M, N, _ = pieces(k)
            cuts.add(Line(v[2], M, color=WHITE, stroke_width=2.5))
            cuts.add(Line(N, v[1], color=WHITE, stroke_width=2.5))
        raM = _ra(M0, v0[2], v0[0], 0.17, YELLOW_B)
        self.play(Create(cuts), Create(raM), run_time=1.0)
        self.hold(0.4)

        # ---- the pieces (their edges are exactly the lines drawn)
        pm = []
        for k in range(3):
            pm.append([mk(P_, cols[k], stroke_width=2,
                          joint_type=LineJointType.BEVEL)
                       for P_ in pieces(k)[3]])
        self.remove(*thirds, chords, cuts, spokes)
        for k in range(3):
            self.add(*pm[k])
        self.bring_to_front(rim, rad, lr, dotC, tks, raM)
        ghost = _dashed(V, GREY_B, 2, 72)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.play(FadeOut(tks), FadeOut(raM), FadeOut(rim), run_time=0.5)
        self.bring_to_front(rad, lr, dotC)

        # ---- carry each third, turned, to its own place on the right
        for k in (0, 2, 1):
            v, M, N, P5 = pieces(k)
            src = [C] + v
            dst = [to_slot(k, q) for q in src]
            m0 = sum(src) / len(src)
            m1 = sum(dst) / len(dst)
            _path_ok([q for P_ in P5 for q in P_], m0, m1, th[k],
                     "a carried third stays on screen")
            self.play(_rigid(VGroup(*pm[k]), src, dst, turn=th[k]),
                      run_time=1.7 if k != 1 else 1.9)
            for m, P_ in zip(pm[k], P5):
                _landed(m, [to_slot(k, q) for q in P_], "the third arrives intact")
        lA = tag("A", 34).move_to(C + 0.75 * UP)
        self.play(FadeIn(lA), run_time=0.4)

        # ---- in each place: the half slides by r, the cap halves turn up
        slides, turns, targets = [], [], []
        for k in range(3):
            v, M, N, P5 = pieces(k)
            Pl = [[to_slot(k, q) for q in P_] for P_ in P5]
            Sq2, Sq3 = S[k] + R * np.array([0.5, 1, 0]), S[k] + R * np.array([-0.5, 1, 0])
            V2s, V3s, V4s = Pl[2]
            slides.append(pm[k][1].animate.shift(-R * RIGHT))
            turns.append(_rigid(pm[k][3], Pl[3], [V3s, Sq2, V2s]))
            turns.append(_rigid(pm[k][4], Pl[4], [Sq3, V3s, V4s]))
            targets.append(([q - R * RIGHT for q in Pl[1]], [V3s, Sq2, V2s],
                            [Sq3, V3s, V4s]))
        self.play(*slides, run_time=1.3)
        self.play(*turns, run_time=1.6)
        for k in range(3):
            _landed(pm[k][1], targets[k][0], "the half lands on the other side")
            _landed(pm[k][3], targets[k][1], "a cap half fills a corner")
            _landed(pm[k][4], targets[k][2], "the other half fills the other corner")

        # ---- three r × r squares
        frames, labs = VGroup(), VGroup()
        for k in range(3):
            Sq = [S[k] + R * np.array(d) for d in
                  ((-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0))]
            frames.add(Polygon(*Sq, stroke_color=YELLOW_B, stroke_width=4))
            labs.add(tag("r²", 32).move_to(S[k] + R * np.array([0.08, 0.42, 0])))
            labs.add(tag("r", 28, YELLOW_B).move_to(
                (Sq[1] + Sq[2]) / 2 + 0.28 * RIGHT))
        self.play(Create(frames), FadeIn(labs), run_time=0.9)
        for k in range(3):
            v, M, N, P5 = pieces(k)
            P1s = [to_slot(k, q) for q in P5[0]]
            check(_pip(labs[2 * k].get_center(), P1s), "r² sits inside its square")
            _labels_clear([labs[2 * k]], [(P1s[0], P1s[3]), (P1s[1], P1s[2])],
                          "F43 r²")
        v1, M1, N1, _ = pieces(1)
        _labels_clear([lr], [(C, V[8]), (v1[2], M1), (v1[2], v1[4]),
                             (V[7], V[8]), (N1, v1[1])], "F43 r")
        cap = caption("A  =  3r²", 38)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


def _free_dir(X, towards):
    """Unit vector at X along the middle of the widest angular gap between
    the directions to the points `towards` (where a label can sit)."""
    X = to3(X)
    angs = sorted(_ang_of(to3(q) - X) % TAU for q in towards)
    best, mid = -1.0, 0.0
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        if a1 - a0 > best:
            best, mid = a1 - a0, (a0 + a1) / 2
    return _polar(mid)


def _vlabel(text, V, centre, d=0.36, size=30, color=WHITE):
    """Vertex label pushed out from the figure's centre."""
    return tag(text, size, color).move_to(to3(V) + d * _unit(to3(V) - to3(centre)))


# =========================================================== F39 TRIANGLE IN A PARALLELOGRAM

class F39_TriangleInParallelogram(Board):
    """Triangle ABE stands on the side AB of the parallelogram ABCD with its
    apex E on the opposite side DC. Slide E along DC to the corner D: the
    base AB and the height h between the two parallels stay, so the area
    stays (a shear). The diagonal BD then halves the parallelogram: a copy
    of ABD turned half a turn about the centre O, the midpoint of BD, lands
    exactly on CDB. So [ABE] = [ABD] = ½[ABCD]."""

    def construct(self):
        Am, Bm, Cm, Dm = (0.0, 0.0), (4.6, 0.0), (6.1, 3.2), (1.5, 3.2)
        Em = (4.0, 3.2)
        k = 1.15                          # the half-turn of ABD must fit
        Om = (np.array(Bm) + np.array(Dm)) / 2

        def S(q):
            return np.array([0.0, 1.08, 0.0]) + k * to3(np.array(q) - Om)
        A, B, C, D, E = [S(q) for q in (Am, Bm, Cm, Dm, Em)]
        O = (B + D) / 2
        G = (A + B + C + D) / 4
        cp = [2 * O - q for q in (A, B, D)]

        check(abs(_cross(B - A, C - D)) < 1e-9 and close(C - B, D - A),
              "ABCD is a parallelogram")
        check(abs(_cross(E - D, C - D)) < 1e-9
              and 0 < np.dot(E - D, C - D) < np.dot(C - D, C - D), "E lies on DC")
        check(abs(abs(area([A, B, E])) - abs(area([A, B, D]))) < 1e-9,
              "same base AB, same height: [ABE] = [ABD]")
        check(_same_poly(cp, [C, D, B]), "the half-turn about O carries ABD onto CDB")
        check(_tiles_exactly([[A, B, D], cp], [A, B, C, D]),
              "ABD and its turned copy tile the parallelogram")
        check(abs(2 * abs(area([A, B, D])) - abs(area([A, B, C, D]))) < 1e-9,
              "[ABD] = ½[ABCD]")
        _sweep_ok([A, B, D], O, PI, "the turning copy stays on screen")

        x0, x1 = S((-1.2, 0))[0], S((7.3, 0))[0]
        g1 = DashedLine([x0, A[1], 0], [x1, A[1], 0], color=GREY_B, stroke_width=2,
                        dash_length=0.09)
        g2 = DashedLine([x0, D[1], 0], [x1, D[1], 0], color=GREY_B, stroke_width=2,
                        dash_length=0.09)
        par = mk([A, B, C, D], BLUE_D, 0.3, stroke_width=3)
        labs = VGroup(*[_vlabel(t, q, G) for t, q in
                        (("A", A), ("B", B), ("C", C), ("D", D))])
        self.play(Create(g1), Create(g2), FadeIn(par), FadeIn(labs), run_time=1.2)

        # ---- the triangle ABE: base AB, apex on DC, height h
        tri = mk([A, B, E], ORANGE)
        dotE = Dot(E, radius=0.07, color=WHITE)
        lE = tag("E", 30).move_to(E + 0.36 * UP)
        base = Line(A, B, color=YELLOW_B, stroke_width=7)
        hx = S((-0.7, 0))[0]
        harr = _darrow([hx, A[1], 0], [hx, D[1], 0])
        lh = tag("h", 30, YELLOW_B).next_to(harr, LEFT, buff=0.12)
        self.play(FadeIn(tri), FadeIn(dotE), FadeIn(lE), run_time=0.9)
        self.play(Create(base), GrowFromCenter(harr), FadeIn(lh), run_time=0.9)
        self.hold(0.4)

        # ---- shear: E slides along DC to D (base and height kept)
        ghost = _dashed([A, B, E], WHITE, 2, 40)
        self.add(ghost)
        self.bring_to_front(base, dotE)
        g2b = DashedLine([x0, D[1], 0], [x1, D[1], 0], color=YELLOW_B,
                         stroke_width=3, dash_length=0.09)
        self.play(Create(g2b), run_time=0.5)
        rider = Dot(E, radius=0.07, color=YELLOW_B)
        self.add(rider)
        self.play(Transform(tri, mk([A, B, D], ORANGE)), rider.animate.move_to(D),
                  run_time=2.2)
        _landed(tri, [A, B, D], "the apex lands on D")
        self.play(FadeOut(rider), FadeOut(g2b), run_time=0.4)
        self.hold(0.3)

        # ---- the diagonal BD halves ABCD: a half-turn about its midpoint O
        bd = Line(B, D, color=WHITE, stroke_width=3)
        dO = Dot(O, radius=0.07, color=WHITE)
        nO = _unit(np.array([-(D - B)[1], (D - B)[0], 0.0]))
        if np.dot(nO, A - O) < 0:
            nO = -nO
        lO = tag("O", 26).move_to(O + 0.36 * nO)
        tk = VGroup(_ticks(B, O, 1), _ticks(O, D, 1))
        self.play(Create(bd), FadeIn(dO), FadeIn(lO), FadeIn(tk), run_time=0.8)
        copy = mk([A, B, D], TEAL_D, 0.85)
        self.add(copy)
        self.bring_to_front(bd, base, ghost, tk, dO, lO)
        self.play(Rotate(copy, angle=PI, about_point=O), run_time=2.2)
        _landed(copy, cp, "the copy lands on CDB")
        self.bring_to_front(bd, dO, lO, tk, ghost, dotE, lE)
        frame = Polygon(A, B, C, D, stroke_color=YELLOW_B, stroke_width=4)
        self.play(Create(frame), run_time=0.7)

        _labels_clear([lO], [(B, D), (A, E), (B, E), (A, B)], "F39 O")
        _labels_clear([lE], [(D, C), (A, E), (B, E)], "F39 E")
        cap = caption("[ABE]  =  [ABD]  =  ½ [ABCD]", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F40 DIAGONALS BISECT

class F40_ParallelogramDiagonals(Board):
    """O is the midpoint of the diagonal AC. A copy of the parallelogram,
    turned half a turn about O, swaps A and C. A half-turn sends every line
    to a parallel line, so the copy's side AB lands on the line through C
    parallel to AB — the side CD — and its side CB on the line through A
    parallel to CB — the side AD. Their crossing B therefore lands on D (and
    D on B): the copy covers the parallelogram exactly. The half-turn about O
    swaps B and D, so O is also the midpoint of BD: the diagonals bisect
    each other."""

    def construct(self):
        Am, Bm, Cm, Dm = (0.0, 0.0), (4.0, 0.0), (4.9, 3.4), (0.9, 3.4)
        k = 1.1                           # the half-turn of ABCD must fit
        Om = (np.array(Am) + np.array(Cm)) / 2

        def S(q):
            return np.array([0.0, (SAFE_TOP + SAFE_BOTTOM) / 2, 0.0]) \
                + k * to3(np.array(q) - Om)
        A, B, C, D = [S(q) for q in (Am, Bm, Cm, Dm)]
        O = (A + C) / 2
        img = [2 * O - q for q in (A, B, C, D)]

        check(close(C - B, D - A), "ABCD is a parallelogram")
        check(close(img[0], C) and close(img[2], A), "the half-turn swaps A and C")
        check(abs(_cross(img[1] - img[0], B - A)) < 1e-9
              and abs(_cross(img[1] - C, D - C)) < 1e-9,
              "the copy's AB lies along CD (parallel, through C)")
        check(close(img[1], D) and close(img[3], B), "so B lands on D and D on B")
        check(close((B + D) / 2, O), "O is the midpoint of BD")
        _sweep_ok([A, B, C, D], O, PI, "the turning copy stays on screen")

        par = mk([A, B, C, D], BLUE_D, 0.55, stroke_width=3)
        labs = VGroup(*[_vlabel(t, q, O) for t, q in
                        (("A", A), ("B", B), ("C", C), ("D", D))])
        chev = VGroup(_chevron(A, B), _chevron(D, C), _chevron(A, D, n=2),
                      _chevron(B, C, n=2))
        self.play(FadeIn(par), FadeIn(labs), run_time=1.0)
        self.play(FadeIn(chev), run_time=0.6)

        # ---- the diagonal AC and its midpoint O
        ac = Line(A, C, color=YELLOW_B, stroke_width=4)
        dO = Dot(O, radius=0.07, color=YELLOW_B)
        lO = tag("O", 28, YELLOW_B).move_to(O + 0.44 * _free_dir(O, [A, B, C, D]))
        tk = dict(size=0.16, width=3.5)
        tk1 = VGroup(_ticks(A, O, 1, YELLOW_B, **tk), _ticks(O, C, 1, YELLOW_B, **tk))
        self.play(Create(ac), run_time=0.7)
        self.play(FadeIn(dO), FadeIn(lO), FadeIn(tk1), run_time=0.6)
        self.hold(0.3)

        # ---- a copy turned half a turn about O lands on the parallelogram
        copy = Polygon(A, B, C, D, stroke_color=ORANGE, stroke_width=6,
                       fill_color=ORANGE, fill_opacity=0.25)
        dA = Dot(A, radius=0.08, color=ORANGE)
        dB = Dot(B, radius=0.08, color=ORANGE)
        rA, rB = np.linalg.norm(A - O), np.linalg.norm(B - O)
        arcA = DashedVMobject(Arc(radius=rA, start_angle=_ang_of(A - O), angle=PI,
                                  arc_center=O, color=ORANGE, stroke_width=2.5),
                              num_dashes=36)
        arcB = DashedVMobject(Arc(radius=rB, start_angle=_ang_of(B - O), angle=PI,
                                  arc_center=O, color=ORANGE, stroke_width=2.5),
                              num_dashes=26)
        self.play(FadeIn(copy), FadeIn(dA), FadeIn(dB), run_time=0.6)
        self.bring_to_front(dO)
        self.play(Rotate(VGroup(copy, dA, dB), angle=PI, about_point=O),
                  Create(arcA), Create(arcB), run_time=2.6)
        _landed(copy, img, "the copy covers the parallelogram, B on D")
        check(close(dA.get_center(), C) and close(dB.get_center(), D),
              "A arrives at C, B arrives at D")
        self.hold(0.9)

        # ---- so O is the midpoint of BD too
        bd = Line(B, D, color=GREEN_B, stroke_width=4)
        tk2 = VGroup(_ticks(B, O, 2, GREEN_B, **tk), _ticks(O, D, 2, GREEN_B, **tk))
        self.play(FadeOut(copy), FadeOut(arcA), FadeOut(arcB), FadeOut(dA),
                  FadeOut(dB), run_time=0.6)
        self.play(Create(bd), run_time=0.8)
        self.bring_to_front(dO)
        self.play(FadeIn(tk2), run_time=0.5)

        _labels_clear([lO], [(A, C), (B, D)], "F40 O")
        cap = caption("AO  =  OC,    BO  =  OD", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F49 POINT IN A PARALLELOGRAM

class F49_PointInParallelogram(Board):
    """Join an interior point P to the corners of the parallelogram ABCD and
    draw through P the parallel XY to AB. In the lower parallelogram ABYX
    the triangle PAB has its apex on the far side; the two triangles left
    over, AXP and PYB, turned half a turn about the midpoints of PA and PB,
    fill PAB exactly (they meet where XP + PY = XY = AB ends). Likewise in
    XYCD the leftovers XDP and PYC, turned about the midpoints of PD and PC,
    fill PCD. So [PAB] + [PCD] = [PBC] + [PDA]: each pair is half."""

    def construct(self):
        Am, Bm, Cm, Dm = (0.0, 0.0), (4.6, 0.0), (5.8, 3.6), (1.2, 3.6)
        Pm = (2.5, 1.35)
        k = 1.25                          # the turning pieces must fit
        mid = np.array([2.99, 1.86])      # middle of everything they sweep

        def S(q):
            return np.array([0.0, (SAFE_TOP + SAFE_BOTTOM) / 2, 0.0]) \
                + k * to3(np.array(q) - mid)
        A, B, C, D, P = [S(q) for q in (Am, Bm, Cm, Dm, Pm)]
        t = (P[1] - A[1]) / (D[1] - A[1])
        X, Y = A + t * (D - A), B + t * (C - B)
        G = (A + B + C + D) / 4
        outer = {"AXP": [A, X, P], "PYB": [P, Y, B],
                 "XDP": [X, D, P], "PYC": [P, Y, C]}
        piv = {"AXP": (A + P) / 2, "PYB": (P + B) / 2,
               "XDP": (D + P) / 2, "PYC": (P + C) / 2}
        land = {n: [2 * piv[n] - q for q in outer[n]] for n in outer}

        check(close(C - B, D - A), "ABCD is a parallelogram")
        check(_pip(P, [A, B, C, D]), "P is inside")
        check(abs(_cross(Y - X, B - A)) < 1e-9, "XY is parallel to AB")
        check(_tiles_exactly([[P, A, B], outer["AXP"], outer["PYB"]], [A, B, Y, X])
              and _tiles_exactly([[P, C, D], outer["XDP"], outer["PYC"]], [X, Y, C, D]),
              "XY cuts ABCD into two parallelograms, each a triangle + two leftovers")
        check(_tiles_exactly([land["AXP"], land["PYB"]], [P, A, B]),
              "the two lower leftovers, turned, tile PAB")
        check(_tiles_exactly([land["XDP"], land["PYC"]], [P, C, D]),
              "the two upper leftovers, turned, tile PCD")
        check(close(land["AXP"][1], land["PYB"][1]),
              "they meet on AB, since XP + PY = AB")
        tot = abs(area([A, B, C, D]))
        check(abs(abs(area([P, A, B])) + abs(area([P, C, D])) - tot / 2) < 1e-9,
              "[PAB] + [PCD] = ½[ABCD]")
        for n in outer:
            _sweep_ok(outer[n], piv[n], PI, "a turning piece stays on screen")

        rim = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=3)
        labs = VGroup(*[_vlabel(s, q, G) for s, q in
                        (("A", A), ("B", B), ("C", C), ("D", D))])
        dP = Dot(P, radius=0.07)
        lP = tag("P", 30).move_to(P + 0.4 * _polar(1.33))
        self.play(Create(rim), FadeIn(labs), FadeIn(dP), FadeIn(lP), run_time=1.1)

        # ---- join P to the four corners
        tAB, tCD = mk([P, A, B], ORANGE), mk([P, C, D], ORANGE)
        tBC, tDA = mk([P, B, C], BLUE_D), mk([P, D, A], BLUE_D)
        self.add(tAB, tBC, tCD, tDA)
        self.bring_to_back(tAB, tBC, tCD, tDA)
        for m in (tAB, tBC, tCD, tDA):
            m.set_opacity(0)
        self.play(*[m.animate.set_fill(opacity=FILL).set_stroke(opacity=1)
                    for m in (tAB, tBC, tCD, tDA)], run_time=1.0)
        self.hold(0.3)

        # ---- the parallel to AB through P cuts PDA and PBC in two
        xy = Line(X, Y, color=WHITE, stroke_width=3)
        chev = VGroup(_chevron(A, B, at=0.3), _chevron(X, Y, at=0.82),
                      _chevron(D, C, at=0.3))
        self.play(Create(xy), FadeIn(chev), run_time=0.9)
        # opaque, in the colour a FILL-opacity piece shows on black, so a
        # copy lying on the orange looks exactly like its original
        cols = {n: interpolate_color(BLACK, c, FILL) for n, c in
                (("AXP", BLUE_D), ("PYB", GREEN_D), ("XDP", PURPLE_B),
                 ("PYC", TEAL_D))}
        pcs = {n: mk(outer[n], interpolate_color(BLACK, BLUE_D, FILL), 1.0)
               for n in outer}
        self.remove(tBC, tDA)
        self.add(*pcs.values())
        self.bring_to_back(*pcs.values())
        self.bring_to_front(xy, dP, lP)
        self.play(*[pcs[n].animate.set_fill(cols[n]) for n in outer], run_time=0.7)
        self.hold(0.3)

        # ---- the lower leftovers turn about the midpoints of PA and PB
        def turn_pair(names):
            dots = VGroup(*[Dot(piv[n], radius=0.06, color=WHITE) for n in names])
            cps = [mk(outer[n], cols[n], 1.0) for n in names]
            self.play(FadeIn(dots), run_time=0.4)
            self.add(*cps)
            self.bring_to_front(xy, chev, dots, dP, lP)
            self.play(*[Rotate(c, angle=PI, about_point=piv[n])
                        for c, n in zip(cps, names)], run_time=2.0)
            for c, n in zip(cps, names):
                _landed(c, land[n], "a turned leftover lands inside the triangle")
            self.play(FadeOut(dots), run_time=0.3)
            return cps

        low = turn_pair(["AXP", "PYB"])
        self.bring_to_front(xy, dP, lP)
        up = turn_pair(["XDP", "PYC"])
        self.bring_to_front(xy, dP, lP)

        o1 = Polygon(P, A, B, stroke_color=ORANGE, stroke_width=6)
        o2 = Polygon(P, C, D, stroke_color=ORANGE, stroke_width=6)
        self.play(Create(o1), Create(o2), run_time=0.9)
        self.bring_to_front(chev, dP, lP)

        _labels_clear([lP], [(P, A), (P, B), (P, C), (P, D), (X, Y),
                             (land["XDP"][1], P)], "F49 P")
        cap = caption("[PAB] + [PCD]  =  ½ [ABCD]", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F53 BRITISH FLAG

class F53_BritishFlag(Board):
    """P inside the rectangle ABCD. The perpendiculars from P to the sides
    have lengths x₁, x₂ (to AD, BC) and y₁, y₂ (to AB, DC), and the lines
    through P cut the rectangle into four corner boxes: x₁ × y₁ at A,
    x₂ × y₁ at B, x₂ × y₂ at C, x₁ × y₂ at D. Each segment from P to a
    corner is a box's diagonal, so by Pythagoras PA² = x₁² + y₁²,
    PC² = x₂² + y₂², PB² = x₂² + y₁², PD² = x₁² + y₂² (the squares are
    drawn to one scale). Both sides of the identity use the same four
    squares, once each — only the pairing differs: swap the two x-squares
    and the second block becomes the first."""

    def construct(self):
        W, H = 5.0, 3.5
        x1, y1 = 2.0, 1.4
        x2, y2 = W - x1, H - y1
        k, kt = 1.0, 0.46
        org = np.array([-6.05, (SAFE_TOP + SAFE_BOTTOM) / 2 - k * H / 2, 0.0])

        def S(q):
            return org + k * to3(q)
        A, B, C, D, P = [S(q) for q in ((0, 0), (W, 0), (W, H), (0, H), (x1, y1))]
        FL, FR, FB, FT = S((0, y1)), S((W, y1)), S((x1, 0)), S((x1, H))
        corner = {"A": A, "B": B, "C": C, "D": D}
        boxes = {"A": [A, FB, P, FL], "B": [FB, B, FR, P],
                 "C": [P, FR, C, FT], "D": [FL, P, FT, D]}
        sides = {"A": ("x1", "y1"), "B": ("x2", "y1"),
                 "C": ("x2", "y2"), "D": ("x1", "y2")}
        L = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

        for c in "ABCD":
            a, b = sides[c]
            bx = boxes[c]
            check(abs(abs(area(bx)) - k * k * L[a] * L[b]) < 1e-9,
                  f"the box at {c} is {a} × {b}")
            check(any(close(q, P) for q in bx)
                  and any(close(q, corner[c]) for q in bx),
                  f"P{c} is a diagonal of the box at {c}")
            check(abs(np.linalg.norm(corner[c] - P) ** 2
                      - k * k * (L[a] ** 2 + L[b] ** 2)) < 1e-9,
                  f"P{c}² = {a}² + {b}²")
        check(_tiles_exactly(list(boxes.values()), [A, B, C, D]),
              "the four boxes tile the rectangle")
        n = np.linalg.norm
        check(abs(n(P - A) ** 2 + n(P - C) ** 2 - n(P - B) ** 2 - n(P - D) ** 2)
              < 1e-9, "PA² + PC² = PB² + PD²")

        # ---- the rectangle, P, the four perpendiculars
        cseg = {"x1": BLUE_B, "x2": ORANGE, "y1": GREEN_B, "y2": PURPLE_A}
        ctile = {"x1": BLUE_D, "x2": ORANGE, "y1": GREEN_D, "y2": PURPLE_B}
        names = {"x1": "x₁", "x2": "x₂", "y1": "y₁", "y2": "y₂"}
        rect = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=3.5)
        G = (A + C) / 2
        labs = VGroup(*[_vlabel(t, q, G, 0.34) for t, q in
                        (("A", A), ("B", B), ("C", C), ("D", D))])
        dP = Dot(P, radius=0.07)
        lP = tag("P", 28).move_to(P + 0.5 * _free_dir(P, [A, B, C, D, FL, FR, FB, FT]))
        self.play(Create(rect), FadeIn(labs), FadeIn(dP), FadeIn(lP), run_time=1.1)

        segs = {"x1": Line(P, FL, color=cseg["x1"], stroke_width=6),
                "x2": Line(P, FR, color=cseg["x2"], stroke_width=6),
                "y1": Line(P, FB, color=cseg["y1"], stroke_width=6),
                "y2": Line(P, FT, color=cseg["y2"], stroke_width=6)}
        ras = VGroup(_ra(FL, P, D, 0.16), _ra(FR, P, C, 0.16),
                     _ra(FB, P, B, 0.16), _ra(FT, P, C, 0.16))
        sl = {"x1": tag(names["x1"], 28, cseg["x1"]).next_to(segs["x1"], UP, buff=0.1),
              "x2": tag(names["x2"], 28, cseg["x2"]).next_to(segs["x2"], UP, buff=0.1),
              "y1": tag(names["y1"], 28, cseg["y1"]).next_to(segs["y1"], LEFT, buff=0.1),
              "y2": tag(names["y2"], 28, cseg["y2"]).next_to(segs["y2"], RIGHT, buff=0.1)}
        sl["x1"].shift(0.3 * LEFT)
        sl["x2"].shift(0.55 * RIGHT)
        sl["y1"].shift(0.1 * DOWN)
        sl["y2"].shift(0.4 * UP)
        self.play(*[Create(s) for s in segs.values()], Create(ras),
                  *[FadeIn(t) for t in sl.values()], run_time=1.3)
        self.bring_to_front(dP)
        self.hold(0.3)

        # ---- the panel: each row is sized by its squares (one scale for all)
        gap_r, gap_b, top = 0.24, 0.5, 3.66
        cx1, cxp, cx2 = 4.0, 5.05, 5.9
        lx = cx1 - kt * max(x2, y2) / 2 - 0.22

        def hrow(a, b):
            return kt * max(L[a], L[b])

        def block_rows(y0, pairs):
            ys, y = [], y0
            for a, b in pairs:
                h = hrow(a, b)
                ys.append(y - h / 2)
                y -= h + gap_r
            return ys, y + gap_r

        ys1, bot1 = block_rows(top, [sides["A"], sides["C"]])
        top2 = bot1 - gap_b
        ys2, bot2 = block_rows(top2, [sides["B"], sides["D"]])
        ys2n, bot2n = block_rows(top2, [sides["A"], sides["C"]])   # after the swap
        rowy = {"A": ys1[0], "C": ys1[1], "B": ys2[0], "D": ys2[1]}

        def tile(name):
            s = kt * L[name]
            sq = Square(side_length=s, fill_color=ctile[name], fill_opacity=FILL,
                        stroke_color=WHITE, stroke_width=1.5)
            t = tag(names[name] + "²", 24)
            if t.width > s - 0.1:
                t.scale_to_fit_width(s - 0.12)
            return VGroup(sq, t.move_to(sq))

        tiles, rlabs, plus = {}, {}, {}
        for c in "ACBD":
            a, b = sides[c]
            y = rowy[c]
            tiles[c] = (tile(a).move_to([cx1, y, 0]), tile(b).move_to([cx2, y, 0]))
            lab = tag(f"P{c}²  =", 30)
            rlabs[c] = lab.move_to([lx - lab.width / 2, y, 0])
            plus[c] = tag("+", 30).move_to([cxp, y, 0])

        for c in "ACBD":
            bx = mk(boxes[c], WHITE, 0.2, stroke_width=0)
            dg = Line(P, corner[c], color=WHITE, stroke_width=4.5)
            a, b = sides[c]
            self.add(bx)
            self.bring_to_back(bx)
            bx.set_fill(opacity=0)
            self.play(bx.animate.set_fill(opacity=0.2), Create(dg),
                      segs[a].animate.set_stroke(width=10),
                      segs[b].animate.set_stroke(width=10), run_time=0.6)
            ta, tb = tiles[c]
            self.play(FadeIn(rlabs[c]), FadeIn(ta), FadeIn(plus[c]), FadeIn(tb),
                      run_time=0.7)
            self.play(bx.animate.set_fill(opacity=0), dg.animate.set_stroke(width=2.5),
                      segs[a].animate.set_stroke(width=6),
                      segs[b].animate.set_stroke(width=6), run_time=0.35)
            self.remove(bx)
            self.bring_to_front(*segs.values(), dP)

        # ---- each block: one label for its two rows
        bx_l = cx1 - kt * max(L.values()) / 2 - 0.28

        def block_label(text, ytop, ybot):
            bar = VGroup(Line([bx_l, ytop, 0], [bx_l, ybot, 0], color=YELLOW_B,
                              stroke_width=3),
                         Line([bx_l, ytop, 0], [bx_l + 0.12, ytop, 0],
                              color=YELLOW_B, stroke_width=3),
                         Line([bx_l, ybot, 0], [bx_l + 0.12, ybot, 0],
                              color=YELLOW_B, stroke_width=3))
            lab = tag(text, 30, YELLOW_B)
            lab.move_to([bx_l - 0.2 - lab.width / 2, (ytop + ybot) / 2, 0])
            return bar, lab

        bar1, bl1 = block_label("PA² + PC²", top, bot1)
        bar2, bl2 = block_label("PB² + PD²", top2, bot2)
        self.play(FadeOut(VGroup(*rlabs.values())), FadeIn(bar1), FadeIn(bl1),
                  FadeIn(bar2), FadeIn(bl2), run_time=0.9)
        self.hold(0.4)

        # ---- the same four squares, paired the other way: swap the x-squares
        bar2n, bl2n = block_label("PB² + PD²", top2, bot2n)
        xB, yB = tiles["B"]
        xD, yD = tiles["D"]
        self.play(xB.animate.move_to([cx1, ys2n[1], 0]),
                  xD.animate.move_to([cx1, ys2n[0], 0]),
                  yB.animate.move_to([cx2, ys2n[0], 0]),
                  yD.animate.move_to([cx2, ys2n[1], 0]),
                  plus["B"].animate.move_to([cxp, ys2n[0], 0]),
                  plus["D"].animate.move_to([cxp, ys2n[1], 0]),
                  Transform(bar2, bar2n), Transform(bl2, bl2n),
                  path_arc=-PI / 3, run_time=1.5)
        dy = top2 - top
        for u, v in ((tiles["A"][0], xD), (tiles["A"][1], yB),
                     (tiles["C"][0], xB), (tiles["C"][1], yD)):
            check(abs(u[0].width - v[0].width) < 1e-9
                  and close(v.get_center() - u.get_center(), np.array([0, dy, 0]), 1e-6),
                  "each square sits under its twin")

        _labels_clear(list(sl.values()) + [lP],
                      [(A, B), (B, C), (C, D), (D, A), (P, FL), (P, FR), (P, FB),
                       (P, FT), (P, A), (P, B), (P, C), (P, D)], "F53 labels")
        for t in list(sl.values()) + [lP]:
            check(all(_apart(t, v, 0.04) for v in labs),
                  "segment labels clear of vertex labels")
        check(bl1.get_left()[0] > max(v.get_right()[0] for v in labs) + 0.6,
              "the block labels stand well clear of the figure")
        cap = caption("PA² + PC²  =  PB² + PD²", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F64 TILING

def _wedges(P, cols, r):
    """Corner wedges of the polygon P (one colour per corner)."""
    n = len(P)
    return VGroup(*[Sector(radius=r, start_angle=_ang_of(P[(i + 1) % n] - P[i]),
                           angle=float(np.arctan2(
                               _cross(P[(i + 1) % n] - P[i], P[i - 1] - P[i]),
                               np.dot(P[(i + 1) % n] - P[i], P[i - 1] - P[i]))),
                           arc_center=to3(P[i]), fill_color=cols[i],
                           fill_opacity=1.0, stroke_width=0)
                    for i in range(n)])


class F64_QuadrilateralTiles(Board):
    """Any quadrilateral ABCD (here convex, no two sides parallel). A copy
    turned half a turn about the midpoint of a side shares that whole side
    with it, its ends swapped. Turn copies about the midpoints of AB and AD,
    and a copy of the first about the midpoint of its side ending at A: at A
    the corners α, β, γ, δ now sit side by side, and since
    α + β + γ + δ = 360° they close up exactly. The same happens at every
    corner, and the pattern repeats by translations (along the diagonals AC
    and BD), so copies of the quadrilateral tile the plane. A triangle and
    its copy turned half a turn about the midpoint of a side make a
    parallelogram — a quadrilateral — so every triangle tiles too."""

    def construct(self):
        k, tilt = 0.72, 8 * DEGREES
        base = [(0.0, 0.0), (3.2, 0.0), (2.2, 2.0), (0.4, 1.6)]
        Aorg = np.array([-1.9, 0.0, 0.0])
        Q = [Aorg + k * _turn(to3(p), ORIGIN, tilt) for p in base]
        A, B, C, D = Q
        u, v = C - A, D - B
        X0, X1, Y0, Y1 = -6.4, 2.7, SAFE_BOTTOM + 0.1, SAFE_TOP - 0.1

        def tile_pts(m, n, s):
            t = m * u + n * v
            return [q + t for q in Q] if s == 1 else [A + B - q + t for q in Q]

        patch = []
        for m in range(-6, 7):
            for n in range(-6, 7):
                for s in (1, -1):
                    P = tile_pts(m, n, s)
                    if all(X0 <= q[0] <= X1 and Y0 <= q[1] <= Y1 for q in P):
                        patch.append((m, n, s))
        core = [(0, 0, 1), (0, 0, -1), (0, 1, -1), (-1, 0, 1)]

        # ---- checks
        angs = [_angle(Q[i], Q[i - 1], Q[(i + 1) % 4]) for i in range(4)]
        check(all(_cross(Q[(i + 1) % 4] - Q[i], Q[(i + 2) % 4] - Q[(i + 1) % 4]) > 0
                  for i in range(4)), "ABCD is convex, counter-clockwise")
        check(abs(_cross(B - A, C - D)) > 0.1 and abs(_cross(C - B, D - A)) > 0.1,
              "no two sides are parallel")
        check(abs(sum(angs) - TAU) < 1e-9, "α + β + γ + δ = 360°")
        check(all(c in patch for c in core), "the four tiles around A are in the patch")
        MAB, MAD = (A + B) / 2, (A + D) / 2
        c1 = [2 * MAB - q for q in Q]
        c2 = [2 * MAD - q for q in Q]
        M3 = (c1[1] + c1[2]) / 2              # copy 1's side from A (its B) on
        c3 = [2 * M3 - q for q in c1]
        check(_same_poly(c1, tile_pts(0, 0, -1)) and _same_poly(c2, tile_pts(0, 1, -1))
              and _same_poly(c3, tile_pts(-1, 0, 1)), "the three half-turns give the core")
        check(close(c1[1], A) and close(c2[3], A) and close(c3[2], A),
              "β, δ and γ of the copies arrive at A")
        check(close(c3[0], A - u), "the third copy is the first one moved by −AC")
        _sweep_ok(Q, MAB, PI, "the first turning copy stays on screen")
        _sweep_ok(Q, MAD, -PI, "the second turning copy stays on screen")
        _sweep_ok(c1, M3, PI, "the third turning copy stays on screen")
        # no overlaps anywhere in the patch; full turn at every inner corner
        polys = [tile_pts(*c) for c in patch]
        for i in range(len(polys)):
            ci = np.mean(polys[i], axis=0)
            for j in range(len(polys)):
                if i != j:
                    check(not _pip(ci, polys[j]), "patch tiles do not overlap")
        rr = 0.12
        corners = {}
        for c, P in zip(patch, polys):
            for i, q in enumerate(P):
                key = tuple(np.round(q[:2], 6))
                corners.setdefault(key, []).append((P, i))
        inner = 0
        for key, lst in corners.items():
            if len(lst) == 4:
                inner += 1
                tot = sum(_angle(P[i], P[i - 1], P[(i + 1) % 4]) for P, i in lst)
                check(abs(tot - TAU) < 1e-9, "four corners close up at every vertex")
                X = to3(np.array(key))
                for th in np.linspace(0, TAU, 37)[:-1] + 0.0123:
                    q = X + rr * _polar(th)
                    check(sum(_pip(q, P) for P, _ in lst) == 1,
                          "around a vertex the four tiles cover each direction once")
                check(sorted(i for _, i in lst) == [0, 1, 2, 3],
                      "each inner vertex gathers α, β, γ and δ")
        check(inner >= 5, "the patch has several inner vertices")

        # ---- the quadrilateral and its four angles
        wc = [YELLOW_B, RED_B, GREEN_B, PURPLE_A]
        rw = 0.27

        def tile(P, s):
            return VGroup(mk(P, BLUE_D if s == 1 else TEAL_E, 0.85, stroke_width=2),
                          _wedges(P, wc, rw))

        q0 = tile(Q, 1)
        gl = [tag(t, 26, wc[i]).move_to(
            Q[i] + 0.56 * angle_mid_dir(Q[i], Q[i - 1], Q[(i + 1) % 4]))
            for i, t in enumerate(("α", "β", "γ", "δ"))]
        self.play(FadeIn(q0[0]), run_time=0.8)
        self.play(FadeIn(q0[1]), *[FadeIn(g) for g in gl], run_time=0.9)
        self.hold(0.4)

        # ---- half-turns about the midpoints of AB and AD
        def half_turn(src_group, M, colour, angle=PI):
            dot = Dot(M, radius=0.06, color=WHITE)
            cp = src_group.copy()
            cp[0].set_fill(colour)
            self.play(FadeIn(dot), run_time=0.3)
            self.add(cp)
            self.bring_to_front(dot)
            self.play(Rotate(cp, angle=angle, about_point=M), run_time=1.6)
            self.play(FadeOut(dot), run_time=0.2)
            return cp

        g1 = half_turn(q0, MAB, TEAL_E)
        _landed(g1[0], c1, "copy 1 shares AB")
        g2 = half_turn(q0, MAD, TEAL_E, -PI)
        _landed(g2[0], c2, "copy 2 shares AD")
        g3 = half_turn(g1, M3, BLUE_D)
        _landed(g3[0], c3, "copy 3 brings γ to A")
        self.bring_to_front(*gl)

        ring = Circle(radius=rw + 0.04, color=WHITE, stroke_width=3).move_to(A)
        l360 = tag("360°", 26)
        l360.move_to(A + 0.62 * _free_dir(A, [B, D, c1[2], c2[2]]))
        bg = BackgroundRectangle(l360, color=BLACK, fill_opacity=0.6, buff=0.05)
        self.play(Create(ring), FadeIn(bg), FadeIn(l360), run_time=0.8)
        self.hold(0.6)

        # ---- the pattern repeats: the plane is tiled
        rest = [c for c in patch if c not in core]
        rest.sort(key=lambda c: np.linalg.norm(np.mean(tile_pts(*c), axis=0) - A))
        others = [tile(tile_pts(*c), c[2]) for c in rest]
        self.play(FadeOut(ring), FadeOut(bg), FadeOut(l360), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(g) for g in others], lag_ratio=0.12),
                  run_time=2.6)
        self.bring_to_front(*gl)
        self.hold(0.5)

        # ---- a triangle: two copies make a parallelogram, which tiles
        t0 = np.array([3.35, -0.85, 0.0])
        a_, b_ = np.array([1.2, -0.18, 0.0]), np.array([0.42, 1.0, 0.0])
        T = [t0, t0 + a_, t0 + b_]
        Mt = (T[1] + T[2]) / 2
        Tc = [2 * Mt - q for q in T]
        check(_same_poly(Tc, [t0 + a_ + b_, T[2], T[1]]), "the copy completes a parallelogram")
        check(_tiles_exactly([T, Tc], [t0, t0 + a_, t0 + a_ + b_, t0 + b_]),
              "triangle + copy = parallelogram")
        _sweep_ok(T, Mt, PI, "the turning triangle stays on screen")
        tri = mk(T, GREEN_D, 0.85, stroke_width=2)
        tcp = mk(T, GOLD_D, 0.85, stroke_width=2)
        self.play(FadeIn(tri), run_time=0.6)
        self.add(tcp)
        self.play(Rotate(tcp, angle=PI, about_point=Mt), run_time=1.5)
        _landed(tcp, Tc, "the triangle's copy lands")
        par = Polygon(t0, t0 + a_, t0 + a_ + b_, t0 + b_, stroke_color=YELLOW_B,
                      stroke_width=4)
        self.play(Create(par), run_time=0.6)
        more = VGroup()
        for (i, j) in ((1, 0), (0, 1), (1, 1)):
            sh = i * a_ + j * b_
            more.add(mk([q + sh for q in T], GREEN_D, 0.85, stroke_width=2),
                     mk([q + sh for q in Tc], GOLD_D, 0.85, stroke_width=2))
        self.play(LaggedStart(*[FadeIn(m) for m in more], lag_ratio=0.1), run_time=1.4)
        self.bring_to_front(par)
        for m in more:
            check(all(q[0] > X1 + 0.4 for q in m.get_vertices()) and _inside(m),
                  "the triangle patch stays in its own corner, on screen")

        cap = caption("α + β + γ + δ  =  360°", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== G16 / G17 ARCTANGENTS

def _grid(L, xs, ys, color=GREY_D, width=1.5):
    """Grid lines x = xs[i], y = ys[j] (lattice units) through the map L."""
    g = VGroup()
    for x in xs:
        g.add(Line(L((x, ys[0])), L((x, ys[-1])), color=color, stroke_width=width))
    for y in ys:
        g.add(Line(L((xs[0], y)), L((xs[-1], y)), color=color, stroke_width=width))
    return g


def _tan_row(greek, value, color, size=32):
    """'tan θ = v' with the angle letter coloured like its triangle."""
    g = VGroup(tag("tan", size), tag(greek, size, color), tag("=", size),
               tag(value, size))
    g.arrange(RIGHT, buff=0.16)
    g[1].shift(0.02 * DOWN)
    return g


class G16_ThreeArctangents(Board):
    """At the lattice point O on a grid line, three right triangles sit side
    by side. Legs 1 and 3: its angle at O is arctan 3. Legs along grid
    diagonals, one diagonal and two (the lattice point at its middle shows
    the long leg is twice the short one), at right angles: arctan 2. Legs 1
    and 1: arctan 1. Their angles at O fill the straight angle: arctan 1 +
    arctan 2 + arctan 3 = π."""

    def construct(self):
        u = 1.6
        O0 = np.array([-2.55, -2.45, 0.0])

        def L(p):
            return O0 + u * np.array([p[0], p[1], 0.0])
        O = L((0, 0))
        T3 = [L((0, 0)), L((1, 0)), L((1, 3))]
        T2 = [L((0, 0)), L((1, 3)), L((-1, 1))]
        T1 = [L((0, 0)), L((-1, 1)), L((-1, 0))]

        check(abs(_angle(T3[1], O, T3[2]) - PI / 2) < 1e-9
              and abs(np.tan(_angle(O, T3[1], T3[2])) - 3) < 1e-9, "T3: right angle, tan 3")
        check(abs(_angle(T2[2], O, T2[1]) - PI / 2) < 1e-9
              and abs(np.linalg.norm(T2[1] - T2[2]) - 2 * np.linalg.norm(T2[2] - O)) < 1e-9
              and abs(np.tan(_angle(O, T2[1], T2[2])) - 2) < 1e-9,
              "T2: right angle at (−1, 1), long leg twice the short: tan 2")
        check(close((T2[1] + T2[2]) / 2, L((0, 2))), "the long leg's midpoint is a lattice point")
        check(abs(_angle(T1[2], O, T1[1]) - PI / 2) < 1e-9
              and abs(np.tan(_angle(O, T1[1], T1[2])) - 1) < 1e-9, "T1: tan 1")
        a1, a2, a3 = (_angle(O, T1[1], T1[2]), _angle(O, T2[1], T2[2]),
                      _angle(O, T3[1], T3[2]))
        check(abs(a1 + a2 + a3 - PI) < 1e-12, "arctan 1 + arctan 2 + arctan 3 = π")
        check(abs(_cross(T3[2] - O, T2[1] - O)) < 1e-9
              and abs(_cross(T2[2] - O, T1[1] - O)) < 1e-9,
              "neighbours share a side: no gap, no overlap at O")

        grid = _grid(L, range(-2, 3), range(0, 4))
        base = Line(L((-2.4, 0)), L((2.4, 0)), color=WHITE, stroke_width=3)
        dO = Dot(O, radius=0.07)
        self.play(Create(grid), run_time=0.9)
        self.play(Create(base), FadeIn(dO), run_time=0.7)

        cols = [ORANGE, TEAL_D, BLUE_D]           # T1, T2, T3
        lcol = [ORANGE, TEAL_B, BLUE_B]
        rp = 0.62
        panel_x, panel_y = 3.6, [2.6, 1.6, 0.6]
        rows = []

        def show(T, col, lcolr, ra, extra, greek, value, row_y, lab_r):
            m = mk(T, col, 0.8, stroke_width=2.5)
            arc = angle_arc(O, T[1], T[2], radius=rp, color=lcolr, width=5)
            g = tag(greek, 30, lcolr).move_to(O + lab_r * angle_mid_dir(O, T[1], T[2]))
            row = _tan_row(greek, value, lcolr).move_to([panel_x, row_y, 0])
            self.play(FadeIn(m), Create(ra), *[FadeIn(e) for e in extra], run_time=1.0)
            self.bring_to_front(dO)
            self.play(Create(arc), FadeIn(g), FadeIn(row), run_time=0.8)
            self.hold(0.3)
            rows.append(row)
            return m, arc, g

        # T3: legs 1 and 3
        ra3 = _ra(T3[1], O, T3[2], 0.18)
        l1a = tag("1", 26).move_to((O + T3[1]) / 2 + 0.3 * DOWN)
        l3 = tag("3", 26).move_to((T3[1] + T3[2]) / 2 + 0.3 * RIGHT)
        m3, arc3, g3 = show(T3, cols[2], lcol[2], ra3, [l1a, l3], "γ", "3",
                            panel_y[0], 1.05)
        # T2: legs one diagonal and two diagonals
        ra2 = _ra(T2[2], O, T2[1], 0.18)
        mid = L((0, 2))
        tk = VGroup(_ticks(O, T2[2], 1, size=0.14, width=3),
                    _ticks(T2[2], mid, 1, size=0.14, width=3),
                    _ticks(mid, T2[1], 1, size=0.14, width=3),
                    Dot(mid, radius=0.06, color=WHITE))
        m2, arc2, g2 = show(T2, cols[1], lcol[1], ra2, [tk], "β", "2",
                            panel_y[1], 1.05)
        # T1: legs 1 and 1
        ra1 = _ra(T1[2], O, T1[1], 0.18)
        l1b = tag("1", 26).move_to((O + T1[2]) / 2 + 0.3 * DOWN)
        l1c = tag("1", 26).move_to((T1[1] + T1[2]) / 2 + 0.3 * LEFT)
        m1, arc1, g1 = show(T1, cols[0], lcol[0], ra1, [l1b, l1c], "α", "1",
                            panel_y[2], 1.1)

        # ---- side by side at O they fill the straight angle
        half = Arc(radius=rp + 0.16, start_angle=0, angle=PI, arc_center=O,
                   color=YELLOW_B, stroke_width=5)
        base2 = Line(L((-2.4, 0)), L((2.4, 0)), color=YELLOW_B, stroke_width=5)
        sumrow = VGroup(tag("α", 32, lcol[0]), tag("+", 32), tag("β", 32, lcol[1]),
                        tag("+", 32), tag("γ", 32, lcol[2]), tag("=", 32),
                        tag("π", 32, YELLOW_B)).arrange(RIGHT, buff=0.16)
        sumrow.move_to([panel_x, -0.75, 0])
        self.play(Create(base2), Create(half), run_time=1.0)
        self.bring_to_front(dO)
        self.play(FadeIn(sumrow), run_time=0.7)

        labs = [g1, g2, g3, l1a, l3, l1b, l1c]
        segs = ([(T3[0], T3[1]), (T3[1], T3[2]), (T3[2], T3[0]), (T2[1], T2[2]),
                 (T2[2], T2[0]), (T1[1], T1[2]), (T1[2], T1[0]),
                 (L((-2.4, 0)), L((2.4, 0)))]
                + [(g.get_start(), g.get_end()) for g in grid])
        _labels_clear(labs, segs, "G16 labels")
        for r in rows + [sumrow]:
            check(r.get_left()[0] > L((2, 0))[0] + 0.4, "the panel clears the grid")
        cap = caption("arctan 1 + arctan 2 + arctan 3  =  π", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


class G17_TwoArctangents(Board):
    """On a grid: the right triangle with legs 3 and 1 has angle arctan ⅓
    at O. On its hypotenuse OZ stands the right triangle OWZ with legs along
    grid diagonals, two diagonals and one, so its angle at O is arctan ½.
    With the half unit square at the end they make the triangle OBW, whose
    legs OW and WB are each two diagonals at right angles: a right isosceles
    triangle, so its angle at O — the two angles together — is π/4."""

    def construct(self):
        u = 1.72
        O0 = np.array([-5.9, (SAFE_TOP + SAFE_BOTTOM) / 2 - 1.72, 0.0])

        def L(p):
            return O0 + u * np.array([p[0], p[1], 0.0])
        O, W, B, Z, Zp = L((0, 0)), L((2, 2)), L((4, 0)), L((3, 1)), L((3, 0))
        Tt = [O, Zp, Z]                    # legs 3, 1
        Th = [O, Z, W]                     # right angle at W
        Ts = [Zp, B, Z]                    # the half unit square
        big = [O, B, W]

        check(abs(_angle(Zp, O, Z) - PI / 2) < 1e-9
              and abs(np.tan(_angle(O, Zp, Z)) - 1 / 3) < 1e-9, "tan β = ⅓")
        check(abs(_angle(W, O, Z) - PI / 2) < 1e-9
              and abs(np.linalg.norm(W - O) - 2 * np.linalg.norm(Z - W)) < 1e-9
              and abs(np.tan(_angle(O, Z, W)) - 0.5) < 1e-9,
              "OWZ: right angle at W, OW = 2·WZ: tan α = ½")
        check(close((O + W) / 2, L((1, 1))), "OW runs through a lattice point: two diagonals")
        check(abs(np.linalg.norm(W - O) - np.linalg.norm(B - W)) < 1e-9
              and abs(_angle(W, O, B) - PI / 2) < 1e-9, "OBW is right isosceles")
        check(_tiles_exactly([Tt, Th, Ts], big), "the three pieces make OBW")
        check(abs(_angle(O, Zp, Z) + _angle(O, Z, W) - PI / 4) < 1e-12,
              "arctan ⅓ + arctan ½ = π/4")

        grid = _grid(L, range(0, 5), range(0, 3))
        dO = Dot(O, radius=0.07)
        lO = tag("O", 28).move_to(O + 0.36 * LEFT + 0.18 * DOWN)
        self.play(Create(grid), FadeIn(dO), FadeIn(lO), run_time=1.0)

        panel_x = 4.0
        # ---- legs 3 and 1: tan β = ⅓
        mt = mk(Tt, BLUE_D, 0.8, stroke_width=2.5)
        rat = _ra(Zp, O, Z, 0.17)
        l3 = tag("3", 26).move_to((O + Zp) / 2 + 0.3 * DOWN)
        l1 = tag("1", 24).move_to((Zp + Z) / 2 + 0.24 * LEFT + 0.06 * DOWN)
        arcb = angle_arc(O, Zp, Z, radius=2.25, color=BLUE_B, width=5)
        gb = tag("β", 28, BLUE_B).move_to(O + 2.62 * angle_mid_dir(O, Zp, Z))
        rb = _tan_row("β", "⅓", BLUE_B).move_to([panel_x, 2.4, 0])
        self.play(FadeIn(mt), Create(rat), FadeIn(l3), FadeIn(l1), run_time=1.0)
        self.bring_to_front(dO)
        self.play(Create(arcb), FadeIn(gb), FadeIn(rb), run_time=0.8)
        self.hold(0.3)

        # ---- on its hypotenuse: legs two diagonals and one, tan α = ½
        mh = mk(Th, TEAL_D, 0.8, stroke_width=2.5)
        rah = _ra(W, O, Z, 0.17)
        m11 = L((1, 1))
        tk = VGroup(_ticks(O, m11, 1, size=0.14, width=3),
                    _ticks(m11, W, 1, size=0.14, width=3),
                    _ticks(W, Z, 1, size=0.14, width=3),
                    Dot(m11, radius=0.06, color=WHITE))
        arca = angle_arc(O, Z, W, radius=1.9, color=TEAL_B, width=5)
        bis = angle_mid_dir(O, Z, W)              # between grid lines x = 1, 2
        ga = tag("α", 28, TEAL_B).move_to(O + (1.22 * u / bis[0]) * bis)
        ra_ = _tan_row("α", "½", TEAL_B).move_to([panel_x, 1.4, 0])
        self.play(FadeIn(mh), Create(rah), FadeIn(tk), run_time=1.0)
        self.bring_to_front(dO)
        self.play(Create(arca), FadeIn(ga), FadeIn(ra_), run_time=0.8)
        self.hold(0.3)

        # ---- with the half square: a right isosceles triangle, π/4 at O
        ms = mk(Ts, GREY_D, 0.8, stroke_width=2.5)
        ras = _ra(Zp, B, Z, 0.17)
        outline = Polygon(*big, stroke_color=YELLOW_B, stroke_width=5)
        tk2 = VGroup(_ticks(Z, B, 1, size=0.14, width=3))
        rq = 1.5
        arcq = angle_arc(O, B, W, radius=rq, color=YELLOW_B, width=5)
        # at the arc's end, just outside OW (above the tick on OW's lower half)
        gq = tag("π/4", 26, YELLOW_B)
        nq = _polar(3 * PI / 4)
        gq.move_to(O + rq * _polar(PI / 4)
                   + (0.707 * (gq.width + gq.height) / 2 + 0.12) * nq)
        self.play(FadeIn(ms), Create(ras), FadeIn(tk2), run_time=0.8)
        self.play(Create(outline), run_time=0.8)
        self.bring_to_front(dO, tk, tk2)
        self.play(Create(arcq), FadeIn(gq), run_time=0.7)
        sumrow = VGroup(tag("α", 32, TEAL_B), tag("+", 32), tag("β", 32, BLUE_B),
                        tag("=", 32), tag("π/4", 32, YELLOW_B)).arrange(RIGHT, buff=0.16)
        sumrow.move_to([panel_x, 0.2, 0])
        self.play(FadeIn(sumrow), run_time=0.6)

        labs = [l3, l1, gb, ga, gq, lO]
        segs = ([(O, B), (B, W), (W, O), (O, Z), (Zp, Z), (Z, W), (Zp, B)]
                + [(g.get_start(), g.get_end()) for g in grid]
                + [(t.get_start(), t.get_end()) for grp in (tk, tk2)
                   for t in grp if isinstance(t, Line)])
        _labels_clear(labs, segs, "G17 labels", gap=0.08)
        for r in (rb, ra_, sumrow):
            check(r.get_left()[0] > B[0] + 0.5, "the panel clears the figure")
        cap = caption("arctan ½ + arctan ⅓  =  π/4", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)
