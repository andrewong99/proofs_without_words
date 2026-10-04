# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w1b.py — area and angle classics (category F), 2D (manim):
#     F12 area of a parallelogram        F13 area of a trapezoid
#     F14 triangle area by a half-turn   F15 area of a rhombus or kite
#     F16 area from the inradius         F17 midline theorem
#     F26 isosceles base angles          F29 area of an equilateral triangle
#     F44 angle sum by a parallel line   F65 two triangles, a parallelogram
#
# Every move of a piece is rigid: a slide (shift), a half-turn (Rotate by π
# about the named midpoint) or a flip (a half-turn in space about the mirror
# line, which the flat camera shows as a reflection). The only other moves
# are the shears of F16, which keep each base and its pair of parallels.
# Every landing is checked with check(...) before it is drawn, so a wrong
# construction fails the render instead of drawing a wrong picture.
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.


# ------------------------------------------------------------ helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _dir(p, q):
    """Direction angle of the ray p -> q."""
    d = to3(q) - to3(p)
    return float(np.arctan2(d[1], d[0]))


def _polar(t):
    return np.array([np.cos(t), np.sin(t), 0.0])


def _angle(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _turn(p, c, th):
    """Point p turned by th about c, in the plane."""
    d = to3(p) - to3(c)
    cs, sn = np.cos(th), np.sin(th)
    return to3(c) + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1],
                              0.0])


def _mirror(p, a, w):
    """Mirror image of p in the line through a along w."""
    p, a, w = to3(p), to3(a), _unit(w)
    d = p - a
    return a + 2 * np.dot(d, w) * w - d


def _out(p, q, inner):
    """Unit normal of segment pq pointing away from the point `inner`."""
    d = _unit(to3(q) - to3(p))
    n = np.array([-d[1], d[0], 0.0])
    return n if np.dot(n, to3(inner) - to3(p)) < 0 else -n


def _span(vertex, p, q):
    """(start direction, span) of the NON-REFLEX angle p-vertex-q."""
    a1, a2 = _dir(vertex, p), _dir(vertex, q)
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return a1, span


def _wedge(vertex, p, q, r, color, op=FILL, stroke=1.5, stroke_color=WHITE):
    """Filled sector for the NON-REFLEX angle p-vertex-q."""
    a1, span = _span(vertex, p, q)
    s = Sector(radius=r, start_angle=a1, angle=span, arc_center=to3(vertex),
               color=color, fill_opacity=op)
    s.set_stroke(stroke_color, stroke)
    return s


def _ring(vertex, p, q, r0, r1, color, op=0.95):
    """Annular sector r0..r1 for the NON-REFLEX angle p-vertex-q."""
    a1, span = _span(vertex, p, q)
    return AnnularSector(inner_radius=r0, outer_radius=r1, angle=span,
                         start_angle=a1, arc_center=to3(vertex), color=color,
                         fill_opacity=op, stroke_width=0)


def _arc_tick(vertex, ang, r, color=YELLOW_B, size=0.13, width=3):
    """A short radial stroke across an angle arc (equal-angle mark)."""
    d = _polar(ang)
    return Line(to3(vertex) + (r - size) * d, to3(vertex) + (r + size) * d,
                color=color, stroke_width=width)


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


def _chevron(p, q, color=YELLOW_B, size=0.17, width=3, n=1, at=0.5):
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


def _right_mark(foot, d1, d2, size=0.2, color=GREY_A, width=2):
    """Right-angle square at `foot` between directions d1 and d2."""
    f, a, b = to3(foot), _unit(d1) * size, _unit(d2) * size
    return VMobject(stroke_color=color, stroke_width=width) \
        .set_points_as_corners([f + a, f + a + b, f + b])


def _dim(p, q, text, nrm, gap=0.4, color=YELLOW_B, size=28, tick=0.11,
         buff=0.14, width=3):
    """Dimension bar beside segment pq (screen points), `gap` away along the
    unit normal nrm, with end ticks; the label sits beyond the bar."""
    p, q, n = to3(p), to3(q), _unit(nrm)
    a, b = p + gap * n, q + gap * n
    bar = VGroup(Line(a, b, color=color, stroke_width=width),
                 Line(a - tick * n, a + tick * n, color=color,
                      stroke_width=width),
                 Line(b - tick * n, b + tick * n, color=color,
                      stroke_width=width))
    lab = tag(text, size, color)
    half = abs(n[0]) * lab.width / 2 + abs(n[1]) * lab.height / 2
    lab.move_to((a + b) / 2 + n * (buff + half))
    return bar, lab


def _darrow(p, q, color=YELLOW_B, width=3):
    """Double arrow p <-> q: a height between two parallels."""
    return DoubleArrow(to3(p), to3(q), buff=0, stroke_width=width,
                       color=color, tip_length=0.18,
                       max_tip_length_to_length_ratio=0.5)


def _dashed(pts, color=GREY_B, width=2, dashes=40):
    return DashedVMobject(Polygon(*[to3(p) for p in pts], stroke_color=color,
                                  stroke_width=width), num_dashes=dashes)


def _landed(mob, pts, what, tol=1e-6):
    """After a move: the polygon's vertices are exactly pts, in order."""
    got = np.array([np.asarray(v, float)[:2] for v in mob.get_vertices()])
    want = np.array([to3(p)[:2] for p in pts])
    check(got.shape == want.shape and np.allclose(got, want, atol=tol), what)


def _on_screen(what, *mobs):
    """Every mobject inside the safe area (the caption band excluded)."""
    for m in mobs:
        check(m.get_left()[0] >= -SAFE_X - 1e-6
              and m.get_right()[0] <= SAFE_X + 1e-6
              and m.get_bottom()[1] >= SAFE_BOTTOM - 1e-6
              and m.get_top()[1] <= SAFE_TOP + 1e-6, what)


def _sweep_ok(pts, pivot, angle, what, n=72):
    """The convex piece with corners pts stays in the safe area all the way
    through a turn by `angle` about pivot."""
    for k in range(n + 1):
        for p in pts:
            q = _turn(p, pivot, angle * k / n)
            check(abs(q[0]) <= SAFE_X and SAFE_BOTTOM <= q[1] <= SAFE_TOP,
                  what)


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


def _tiles(polys, region, n=110):
    """True if the polygons tile `region` with no gap and no overlap: on a
    jittered grid over its bounding box, every point inside the region lies
    in exactly one polygon and every point outside it in none."""
    xs = [float(p[0]) for p in region]
    ys = [float(p[1]) for p in region]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for i in range(n):
        for j in range(n):
            # irrational offsets keep the samples off every edge
            x = x0 + (x1 - x0) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = y0 + (y1 - y0) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            k = sum(_pip((x, y), P) for P in polys)
            if k != (1 if _pip((x, y), region) else 0):
                return False
    return True


# ===================================================== F12 PARALLELOGRAM

class F12_ParallelogramArea(Board):
    """The altitude from D cuts a right triangle off the left end of the
    parallelogram. Slid along the base by b (a translation), its slanted
    side lands on the opposite side BC, which is parallel and equal to it,
    so it fits the right end exactly: the parallelogram has become a b × h
    rectangle between the same two parallels. (One cut is enough when the
    top overhangs the base by less than b; a more slanted parallelogram
    needs several such cuts, or the shear of F5.)"""

    def construct(self):
        b, h, s = 5.0, 2.8, 1.7                 # base, height, overhang
        Am, Bm, Cm, Dm = (0.0, 0.0), (b, 0.0), (b + s, h), (s, h)
        Em, Gm = (s, 0.0), (b + s, 0.0)         # foot of the cut, new corner
        F = Frame(-1.75, b + s + 0.45, -1.05, h + 0.45)
        A, B, C, D, E, G = [F.P(p) for p in (Am, Bm, Cm, Dm, Em, Gm)]
        v = B - A                               # the slide: along AB, by b

        check(0 < s < b, "the altitude from D falls inside the base")
        check(abs(np.dot(D - E, B - A)) < 1e-9, "the cut is perpendicular")
        check(close(C - B, D - A), "BC is AD moved by b")
        check(close([p + v for p in (A, E, D)], [B, G, C]),
              "the cut-off triangle, slid by b, fits the right end")
        check(_tiles([[E, B, C, D], [B, G, C]], [E, G, C, D]),
              "trapezoid + slid triangle tile the rectangle EGCD")
        check(abs(area([Am, Bm, Cm, Dm]) - b * h) < 1e-9
              and abs(area([Em, Gm, Cm, Dm]) - b * h) < 1e-9,
              "parallelogram = rectangle = bh")

        par = mk([A, B, C, D], BLUE_D)
        g_base = guide(F.P((-1.3, 0)), F.P((b + s + 0.3, 0)), GREY_B)
        g_top = guide(F.P((-1.3, h)), F.P((b + s + 0.3, h)), GREY_B)
        base = Line(A, B, color=YELLOW_B, stroke_width=7)
        b_lab = tag("b", 32, YELLOW_B).next_to(base, DOWN, buff=0.2)
        harrow = _darrow(F.P((-0.85, 0)), F.P((-0.85, h)))
        h_lab = tag("h", 32, YELLOW_B).next_to(harrow, LEFT, buff=0.15)
        _on_screen("F12 labels on screen", h_lab, b_lab, par)

        self.play(Create(g_base), Create(g_top), FadeIn(par), run_time=1.2)
        self.play(Create(base), FadeIn(b_lab), GrowFromCenter(harrow),
                  FadeIn(h_lab), run_time=1.0)
        self.hold(0.5)

        # the cut: the altitude from D
        cut = Line(D, E, color=WHITE, stroke_width=3)
        ra = _right_mark(E, UP, RIGHT, 0.22, WHITE)
        self.play(Create(cut), run_time=0.8)
        tri = mk([A, E, D], BLUE_D)
        rest = mk([E, B, C, D], BLUE_D)
        self.remove(par)
        self.add(rest, tri, cut, base)
        self.play(Create(ra), tri.animate.set_fill(ORANGE), run_time=0.6)
        self.hold(0.3)

        # slide it along the base by b
        ghost = mk([A, E, D], ORANGE, op=0.13, stroke_width=0)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.bring_to_back(g_base, g_top)
        self.play(FadeOut(base), run_time=0.3)
        self.play(tri.animate.shift(v), run_time=2.0)
        _landed(tri, [B, G, C], "the triangle fits the right end")

        rect = Polygon(E, G, C, D, stroke_color=YELLOW_B, stroke_width=5)
        b_new = b_lab.copy().next_to(Line(E, G), DOWN, buff=0.2)
        self.play(Create(rect), b_lab.animate.move_to(b_new), run_time=1.0)
        self.bring_to_front(ra)
        self.play(Write(caption("A  =  b · h", 36)))
        self.hold(2.2)


# ===================================================== F13 TRAPEZOID

class F13_TrapezoidArea(Board):
    """A copy of the trapezoid, turned half a turn about the midpoint M of
    the leg BC, lands with that leg on itself (B and C exchanged). Its two
    bases then continue the original ones along the same two parallels, so
    the pair is a parallelogram of base a + b and height h: twice the
    trapezoid."""

    def construct(self):
        a, b, h, p = 4.0, 2.0, 3.0, 0.8
        Am, Bm, Cm, Dm = (0.0, 0.0), (a, 0.0), (p + b, h), (p, h)
        F = Frame(-1.45, a + b + p + 0.45, -2.35, h + 0.55)
        A, B, C, D = [F.P(q) for q in (Am, Bm, Cm, Dm)]
        M = (B + C) / 2
        A2, B2, C2, D2 = [2 * M - q for q in (A, B, C, D)]    # the half-turn

        check(close(B2, C) and close(C2, B), "the leg BC lands on itself")
        check(abs(D2[1] - A[1]) < 1e-9 and abs(A2[1] - D[1]) < 1e-9,
              "the copy's bases lie on the same two parallels")
        check(close(D2 - B, (b * F.k) * RIGHT)
              and close(A2 - C, (a * F.k) * RIGHT),
              "bottom: a then b; top: b then a")
        check(close(D2 - A, A2 - D), "A D2 A2 D is a parallelogram")
        check(_tiles([[A, B, C, D], [A2, B2, C2, D2]], [A, D2, A2, D]),
              "trapezoid + copy tile the parallelogram")
        check(abs(2 * area([Am, Bm, Cm, Dm]) - (a + b) * h) < 1e-9,
              "2A = (a + b)h")
        _sweep_ok([A, B, C, D], M, PI, "the turning copy stays on screen")

        trap = mk([A, B, C, D], BLUE_D)
        g_base = guide(F.P((-1.05, 0)), F.P((a + b + p + 0.35, 0)), GREY_B)
        g_top = guide(F.P((-1.05, h)), F.P((a + b + p + 0.35, h)), GREY_B)
        a_lab = tag("a", 32, BLUE_B).move_to(F.P((a / 2, 0)) + DOWN * 0.36)
        b_lab = tag("b", 32, BLUE_B).move_to(F.P((p + b / 2, h)) + UP * 0.36)
        harrow = _darrow(F.P((-0.62, 0)), F.P((-0.62, h)))
        h_lab = tag("h", 32, YELLOW_B).next_to(harrow, LEFT, buff=0.15)

        self.play(Create(g_base), Create(g_top), FadeIn(trap), run_time=1.2)
        self.play(FadeIn(a_lab), FadeIn(b_lab), GrowFromCenter(harrow),
                  FadeIn(h_lab), run_time=0.9)
        self.hold(0.5)

        # the midpoint of the leg BC
        dM = Dot(M, radius=0.07, color=ORANGE)
        tks = VGroup(_ticks(B, M, 1), _ticks(M, C, 1))
        self.play(FadeIn(dM), FadeIn(tks), run_time=0.6)

        # a copy turns half a turn about M
        copy = mk([A, B, C, D], ORANGE)
        self.play(FadeIn(copy), run_time=0.5)
        self.bring_to_front(a_lab, b_lab, dM)
        self.play(Rotate(copy, angle=PI, about_point=M), run_time=2.4)
        _landed(copy, [A2, B2, C2, D2], "the copy completes the parallelogram")
        self.bring_to_front(tks, dM)

        b2_lab = tag("b", 32, ORANGE).move_to(
            F.P((a + b / 2, 0)) + DOWN * 0.36)
        a2_lab = tag("a", 32, ORANGE).move_to(
            F.P((p + b + a / 2, h)) + UP * 0.36)
        self.play(FadeIn(b2_lab), FadeIn(a2_lab), FadeOut(dM), FadeOut(tks),
                  run_time=0.7)

        outline = Polygon(A, D2, A2, D, stroke_color=YELLOW_B, stroke_width=5)
        chev = VGroup(_chevron(A, D), _chevron(D2, A2))
        bar, ab_lab = _dim(A, D2, "a + b", DOWN, gap=0.8)
        _on_screen("F13 labels on screen", ab_lab, a2_lab, h_lab, bar)
        self.play(Create(outline), FadeIn(chev), run_time=0.9)
        self.play(FadeIn(bar), FadeIn(ab_lab), run_time=0.7)
        self.play(Write(caption("A  =  ½ · (a + b) · h", 36)))
        self.hold(2.2)


# ===================================================== F14 HALF-TURN

class F14_TriangleHalfTurn(Board):
    """M and N halve the sides CA and CB, so the midline MN runs at half the
    height. Cut the top triangle off along it, and cut that along its
    altitude. Each top piece turns half a turn about M or N: the apex C
    goes to A or to B (MC = MA, NC = NB), and the altitude stands upright
    at the two ends of the base. The triangle has become a b × h/2
    rectangle on the same base. (The altitude must fall inside the base:
    take the longest side as the base.)"""

    def construct(self):
        b, cx, h = 6.0, 2.3, 3.8
        Am, Bm, Cm = (0.0, 0.0), (b, 0.0), (cx, h)
        F = Frame(-2.35, b + 1.55, -0.95, h + 0.95)
        A, B, C = [F.P(q) for q in (Am, Bm, Cm)]
        M, N = (A + C) / 2, (B + C) / 2
        K, H = F.P((cx, h / 2)), F.P((cx, 0.0))
        TL, TR = F.P((0.0, h / 2)), F.P((b, h / 2))
        L2 = [2 * M - q for q in (M, K, C)]       # left piece, turned
        R2 = [2 * N - q for q in (K, N, C)]       # right piece, turned

        check(0 < cx < b, "the altitude falls inside the base")
        check(abs(M[1] - K[1]) < 1e-9 and abs(N[1] - K[1]) < 1e-9
              and abs(K[0] - C[0]) < 1e-9, "K is where the altitude meets MN")
        check(close(M[1] - A[1], (C[1] - A[1]) / 2), "MN at half the height")
        check(close(L2, [M, TL, A]), "left piece: C -> A, K -> top left")
        check(close(R2, [TR, N, B]), "right piece: C -> B, K -> top right")
        check(_tiles([[A, B, N, M], L2, R2], [A, B, TR, TL]),
              "the three pieces tile the b × h/2 rectangle")
        check(abs(area([Am, Bm, Cm]) - b * (h / 2)) < 1e-9,
              "½bh = b · h/2")
        _sweep_ok([M, K, C], M, PI, "left piece stays on screen")
        _sweep_ok([K, N, C], N, -PI, "right piece stays on screen")

        tri = mk([A, B, C], BLUE_D)
        g_base = guide(F.P((-2.0, 0)), F.P((b + 0.45, 0)), GREY_B)
        g_top = guide(F.P((-2.0, h)), F.P((cx + 1.2, h)), GREY_B)
        base = Line(A, B, color=YELLOW_B, stroke_width=7)
        b_lab = tag("b", 32, YELLOW_B).next_to(base, DOWN, buff=0.2)
        alt = DashedLine(C, H, color=WHITE, stroke_width=2.5,
                         dash_length=0.09)
        ra = _right_mark(H, UP, RIGHT, 0.2, WHITE)
        harrow = _darrow(F.P((-1.55, 0)), F.P((-1.55, h)))
        h_lab = tag("h", 32, YELLOW_B).next_to(harrow, LEFT, buff=0.15)
        _on_screen("F14 left labels on screen", h_lab, b_lab)

        self.play(Create(g_base), FadeIn(tri), run_time=1.1)
        self.play(Create(base), FadeIn(b_lab), run_time=0.6)
        self.play(Create(g_top), Create(alt), Create(ra),
                  GrowFromCenter(harrow), FadeIn(h_lab), run_time=1.0)
        self.hold(0.4)

        # the midpoints and the midline
        dots = VGroup(Dot(M, radius=0.07), Dot(N, radius=0.07))
        t_am, t_mc = _ticks(A, M, 1), _ticks(M, C, 1)
        t_bn, t_nc = _ticks(B, N, 2), _ticks(N, C, 2)
        self.play(FadeIn(dots), FadeIn(VGroup(t_am, t_mc, t_bn, t_nc)),
                  run_time=0.7)
        midline = Line(M, N, color=WHITE, stroke_width=3)
        top_cut = Line(C, K, color=WHITE, stroke_width=3)
        self.play(Create(midline), run_time=0.6)
        self.play(Create(top_cut), FadeOut(alt), FadeOut(ra), run_time=0.6)

        trap = mk([A, B, N, M], BLUE_D)
        Lp, Rp = mk([M, K, C], BLUE_D), mk([K, N, C], BLUE_D)
        self.remove(tri, midline, top_cut)
        self.add(trap, Lp, Rp, base, dots, t_am, t_mc, t_bn, t_nc)
        self.play(Lp.animate.set_fill(TEAL_D), Rp.animate.set_fill(ORANGE),
                  FadeOut(base), run_time=0.6)
        self.hold(0.3)

        # half a turn about M, then about N; each carries its tick along
        ghost = _dashed([M, C, N], GREY_B, 2, 24)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.bring_to_back(g_base, g_top)
        self.play(Rotate(VGroup(Lp, t_mc), angle=PI, about_point=M),
                  run_time=1.9)
        _landed(Lp, L2, "left piece lands at the left end")
        self.play(Rotate(VGroup(Rp, t_nc), angle=-PI, about_point=N),
                  run_time=1.9)
        _landed(Rp, R2, "right piece lands at the right end")
        self.bring_to_front(dots, t_am, t_bn, t_mc, t_nc)

        rect = Polygon(A, B, TR, TL, stroke_color=YELLOW_B, stroke_width=5)
        h2 = _darrow(F.P((b + 0.6, 0)), F.P((b + 0.6, h / 2)))
        h2_lab = tag("h/2", 30, YELLOW_B).next_to(h2, RIGHT, buff=0.14)
        _on_screen("F14 h/2 label on screen", h2_lab)
        self.play(Create(rect), GrowFromCenter(h2), FadeIn(h2_lab),
                  run_time=1.0)
        self.play(Write(caption("A  =  b · h/2  =  ½ · b · h", 36)))
        self.hold(2.2)


# ===================================================== F65 TWO TRIANGLES

class F65_TwoTriangles(Board):
    """A copy of the triangle turns half a turn about the midpoint M of BC.
    BC comes back onto itself, the copy's base runs along the parallel
    through C, and its other two sides are parallel to AB and AC: the two
    triangles make a parallelogram with the same base b and height h.
    Each triangle is half of it."""

    def construct(self):
        b, cx, h = 4.6, 1.5, 3.0
        Am, Bm, Cm = (0.0, 0.0), (b, 0.0), (cx, h)
        F = Frame(-1.95, b + cx + 0.55, -0.95, h + 2.05)
        A, B, C = [F.P(q) for q in (Am, Bm, Cm)]
        M = (B + C) / 2
        D = 2 * M - A
        A2, B2, C2 = [2 * M - q for q in (A, B, C)]

        check(close([A2, B2, C2], [D, C, B]), "BC lands on itself")
        check(close(D - C, B - A) and close(D - B, C - A),
              "CD equal and parallel to AB, BD to AC: a parallelogram")
        check(abs(D[1] - C[1]) < 1e-9, "the copy's base on the parallel")
        check(_tiles([[A, B, C], [A2, B2, C2]], [A, B, D, C]),
              "the two triangles tile the parallelogram")
        check(abs(2 * area([Am, Bm, Cm]) - b * h) < 1e-9, "2A = bh")
        _sweep_ok([A, B, C], M, -PI, "the turning copy stays on screen")

        tri = mk([A, B, C], BLUE_D)
        g_base = guide(F.P((-1.35, 0)), F.P((b + cx + 0.35, 0)), GREY_B)
        g_top = guide(F.P((-1.35, h)), F.P((b + cx + 0.35, h)), GREY_B)
        base = Line(A, B, color=YELLOW_B, stroke_width=7)
        b_lab = tag("b", 32, YELLOW_B).next_to(base, DOWN, buff=0.2)
        harrow = _darrow(F.P((-0.95, 0)), F.P((-0.95, h)))
        h_lab = tag("h", 32, YELLOW_B).next_to(harrow, LEFT, buff=0.15)
        _on_screen("F65 labels on screen", h_lab, b_lab)

        self.play(Create(g_base), Create(g_top), FadeIn(tri), run_time=1.2)
        self.play(Create(base), FadeIn(b_lab), GrowFromCenter(harrow),
                  FadeIn(h_lab), run_time=0.9)
        self.hold(0.4)

        dM = Dot(M, radius=0.07, color=ORANGE)
        tks = VGroup(_ticks(B, M, 1), _ticks(M, C, 1))
        self.play(FadeIn(dM), FadeIn(tks), run_time=0.6)
        copy = mk([A, B, C], ORANGE)
        self.play(FadeIn(copy), run_time=0.5)
        self.bring_to_front(base, dM)
        self.play(Rotate(copy, angle=-PI, about_point=M), run_time=2.4)
        _landed(copy, [A2, B2, C2], "the copy completes the parallelogram")
        self.bring_to_front(tks, dM)
        self.play(FadeOut(dM), FadeOut(tks), run_time=0.4)

        outline = Polygon(A, B, D, C, stroke_color=YELLOW_B, stroke_width=5)
        chev = VGroup(_chevron(A, B, at=0.62), _chevron(C, D, at=0.62),
                      _chevron(A, C, n=2), _chevron(B, D, n=2))
        self.play(Create(outline), FadeIn(chev), run_time=0.9)
        h1 = tag("½bh", 32).move_to((A + B + C) / 3)
        h2 = tag("½bh", 32).move_to((D + B + C) / 3)
        self.play(FadeIn(h1), FadeIn(h2), run_time=0.7)
        self.play(Write(caption("A  =  ½ · b · h", 36)))
        self.hold(2.2)


# ===================================================== F44 PARALLEL LINE

class F44_ParallelLine(Board):
    """Through C draw the parallel to AB. Along each transversal, CA and CB,
    the two parallels make a Z; the half-turn about the transversal's
    midpoint swaps its ends and maps the Z onto itself, so it carries the
    angle at A onto the alternate angle at C, and the angle at B likewise.
    At C the three angles now sit side by side along one straight line."""

    def construct(self):
        Am, Bm, Cm = (0.0, 0.0), (6.0, 0.0), (2.2, 3.4)
        F = Frame(-1.3, 7.3, -0.8, 4.05)
        A, B, C = [F.P(q) for q in (Am, Bm, Cm)]
        M, N = (A + C) / 2, (B + C) / 2
        u = _unit(B - A)
        R = 0.85
        al, be, ga = _angle(A, B, C), _angle(B, C, A), _angle(C, A, B)
        Lz = 2.1                                       # arm of each Z

        check(close(2 * M - A, C) and close(2 * N - B, C),
              "the half-turns send A and B to C")
        check(close(2 * M - (A + u), C - u) and close(2 * N - (B - u), C + u),
              "they send the base line onto the parallel through C")
        check(close(2 * M - (C - Lz * u), A + Lz * u)
              and close(2 * N - (C + Lz * u), B - Lz * u),
              "each Z is mapped onto itself")
        check(abs(_dir(C, A) % TAU - (PI + al)) < 1e-9
              and abs(_dir(C, B) % TAU - (TAU - be)) < 1e-9,
              "at C: alpha, then gamma, then beta, below the parallel")
        check(abs(al + be + ga - PI) < 1e-9, "alpha + beta + gamma = 180°")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        wA = _wedge(A, B, C, R, BLUE_D)
        wB = _wedge(B, C, A, R, TEAL_D)
        wC = _wedge(C, A, B, R, ORANGE)

        def lab(s, v, p, q, col, k=R + 0.34):
            return tag(s, 32, col).move_to(v + k * angle_mid_dir(v, p, q))

        lA = lab("α", A, B, C, BLUE_B)
        lB = lab("β", B, C, A, TEAL_B)
        lC = lab("γ", C, A, B, ORANGE, R + 0.52)
        self.play(Create(tri), run_time=1.0)
        self.play(FadeIn(wA), FadeIn(wB), FadeIn(wC), FadeIn(lA), FadeIn(lB),
                  FadeIn(lC), run_time=0.9)
        self.bring_to_front(tri)
        self.hold(0.4)

        # the parallel through C
        p0, p1 = F.P((-1.0, Cm[1])), F.P((7.0, Cm[1]))
        ell = Line(p0, p1, color=YELLOW_B, stroke_width=3)
        chev = VGroup(_chevron(A, B, at=0.7), _chevron(p0, p1, at=0.78))
        self.play(Create(ell), FadeIn(chev), run_time=1.0)
        self.hold(0.3)

        def carry(wedge, mid, end, nt, ang, s, col):
            """Half-turn about `mid` (nt ticks on its halves) carrying the
            wedge at `end` to C; ang = -1 / +1: the Z's arm at C points
            left / right."""
            z = VMobject(stroke_color=YELLOW_B, stroke_width=7)
            z.set_points_as_corners([C + ang * Lz * u, C, end,
                                     end - ang * Lz * u])
            dot = Dot(mid, radius=0.07, color=col)
            tk = VGroup(_ticks(end, mid, nt), _ticks(mid, C, nt))
            cp = wedge.copy()
            self.add(cp)
            self.play(Create(z), FadeIn(dot), FadeIn(tk), run_time=0.8)
            self.play(Rotate(cp, angle=ang * PI, about_point=mid),
                      run_time=1.8)
            new = lab(s, C, C + ang * u, end, col, R + 0.52)
            self.play(FadeIn(new), FadeOut(z), FadeOut(dot), FadeOut(tk),
                      run_time=0.6)
            return cp, new

        # alpha: the Z through C, A; beta: the Z through C, B
        cA, lA2 = carry(wA, M, A, 1, -1, "α", BLUE_B)
        cB, lB2 = carry(wB, N, B, 2, 1, "β", TEAL_B)
        self.bring_to_front(tri, ell)
        _on_screen("F44 labels on screen", lA2, lB2, lA, lB)

        straight = Arc(radius=R + 0.12, start_angle=PI, angle=PI,
                       arc_center=C, color=YELLOW_B, stroke_width=5)
        self.play(Create(straight), run_time=0.9)
        self.play(Write(caption("α + β + γ  =  180°", 36)))
        self.hold(2.2)


# ===================================================== F26 ISOSCELES

class F26_IsoscelesBaseAngles(Board):
    """AB = AC. Turn the triangle over about the bisector of the angle at
    A: the ray AB falls on the ray AC (the two halves of the angle are
    equal), and B falls on C because AB = AC; likewise C falls on B. The
    triangle lands on itself with B and C exchanged, so the angle at B
    fits the angle at C exactly."""

    def construct(self):
        L, half, tilt = 4.3, 34 * DEGREES, 8 * DEGREES
        ax = -PI / 2 + tilt                       # direction of the bisector
        Am = np.zeros(2)
        Bm = L * np.array([np.cos(ax - half), np.sin(ax - half)])
        Cm = L * np.array([np.cos(ax + half), np.sin(ax + half)])
        F = Frame(-2.6, 3.6, -4.6, 0.55)
        A, B, C = [F.P(q) for q in (Am, Bm, Cm)]
        w = _polar(ax)
        X = A + 5.6 * w                           # the axis, past the base

        check(abs(np.linalg.norm(B - A) - np.linalg.norm(C - A)) < 1e-9,
              "AB = AC")
        check(abs(_angle(A, B, X) - _angle(A, X, C)) < 1e-9,
              "the axis halves the angle at A")
        check(close(_mirror(B, A, w), C) and close(_mirror(C, A, w), B),
              "turned over the bisector, B and C trade places")
        check(abs(_angle(B, C, A) - _angle(C, A, B)) < 1e-9,
              "the base angles agree")
        check(np.dot(X - A, w) > np.dot((B + C) / 2 - A, w),
              "the axis reaches past the base")

        tri = mk([A, B, C], BLUE_D, op=0.5)
        G = (A + B + C) / 3
        names = VGroup(*[tag(s, 32).move_to(P + 0.42 * _unit(P - G))
                         for s, P in (("A", A), ("B", B), ("C", C))])
        sides = VGroup(_ticks(A, B, 1), _ticks(A, C, 1))
        rw = 0.78
        wB = _wedge(B, C, A, rw, TEAL_D, op=0.95)
        wC = _wedge(C, A, B, rw, ORANGE, op=0.95)

        self.play(FadeIn(tri), FadeIn(names), run_time=1.0)
        self.play(FadeIn(sides), run_time=0.6)
        self.play(FadeIn(wB), FadeIn(wC), run_time=0.7)
        self.hold(0.4)

        # the bisector of the angle at A: two equal halves
        ra = 1.05
        hb1 = angle_arc(A, B, X, ra, YELLOW_B, 4)
        hb2 = angle_arc(A, X, C, ra, YELLOW_B, 4)
        m1 = _arc_tick(A, ax - half / 2, ra)
        m2 = _arc_tick(A, ax + half / 2, ra)
        axis = DashedLine(A, X, color=GREY_A, stroke_width=2.5,
                          dash_length=0.1)
        self.play(Create(axis), Create(hb1), Create(hb2), FadeIn(m1),
                  FadeIn(m2), run_time=1.1)
        self.hold(0.4)

        # turn the whole figure over about that line
        r0, r1 = rw + 0.08, rw + 0.42
        flip = VGroup(Polygon(A, B, C, stroke_color=YELLOW_B, stroke_width=5),
                      _ring(B, C, A, r0, r1, TEAL_B),
                      _ring(C, A, B, r0, r1, ORANGE),
                      _ticks(A, B, 1, YELLOW_B), _ticks(A, C, 1, YELLOW_B))
        self.play(FadeIn(flip), run_time=0.6)
        self.play(Rotate(flip, angle=PI, axis=w, about_point=A), run_time=2.6)
        _landed(flip[0], [A, C, B], "the turned triangle lands on ABC")
        self.bring_to_front(names)
        self.hold(0.8)
        self.play(Write(caption("AB = AC   ⟹   ∠B = ∠C", 36)))
        self.hold(2.2)


# ===================================================== F17 MIDLINE

class F17_MidlineTheorem(Board):
    """M, N are the midpoints of AB and AC. Turn the top triangle AMN half a
    turn about N: A goes to C, M to M′ on the line MN with NM′ = MN, and
    CM′ is AM turned, so it is equal and parallel to MB. MBCM′ is a
    parallelogram: BC slides along BM exactly onto MM′. So MM′ = 2·MN is
    parallel and equal to BC."""

    def construct(self):
        Am, Bm, Cm = (1.6, 3.6), (0.0, 0.0), (6.0, 0.0)
        F = Frame(-0.95, 7.65, -0.8, 5.0)
        A, B, C = [F.P(q) for q in (Am, Bm, Cm)]
        M, N = (A + B) / 2, (A + C) / 2
        M2 = 2 * N - M
        v = M - B

        check(close(2 * N - A, C), "the half-turn about N sends A to C")
        check(abs(_cross(M2 - M, N - M)) < 1e-9
              and close(M2 - M, 2 * (N - M)), "M, N, M′ in line, NM′ = MN")
        check(close(C - M2, B - M), "M′C equal and parallel to MB")
        check(close(B + v, M) and close(C + v, M2), "BC slid by BM is MM′")
        check(abs(np.linalg.norm(N - M) - np.linalg.norm(C - B) / 2) < 1e-9,
              "MN = ½ BC")
        _sweep_ok([A, M, N], N, -PI, "the turning copy stays on screen")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        G = (A + B + C) / 3
        names = VGroup(*[tag(s, 32).move_to(P + 0.4 * _unit(P - G))
                         for s, P in (("A", A), ("B", B), ("C", C))])
        dM, dN = Dot(M, radius=0.07), Dot(N, radius=0.07)
        lM = tag("M", 30).move_to(M + 0.42 * _out(A, B, C))
        lN = tag("N", 30).move_to(N + 0.42 * _polar(62 * DEGREES))
        t1 = VGroup(_ticks(A, M, 1), _ticks(M, B, 1))
        t2 = VGroup(_ticks(A, N, 2), _ticks(N, C, 2))
        self.play(Create(tri), FadeIn(names), run_time=1.1)
        self.play(FadeIn(dM), FadeIn(dN), FadeIn(lM), FadeIn(lN), FadeIn(t1),
                  FadeIn(t2), run_time=0.9)
        mid = Line(M, N, color=YELLOW_B, stroke_width=6)
        top = mk([A, M, N], TEAL_D, op=0.7, stroke_width=0)
        self.add(top)
        self.bring_to_back(top)
        self.play(Create(mid), FadeIn(top), run_time=0.8)
        self.hold(0.4)

        # a copy of AMN turns half a turn about N, its marks with it
        copy = VGroup(mk([A, M, N], ORANGE, op=0.85),
                      Line(M, N, color=YELLOW_B, stroke_width=6),
                      _ticks(A, M, 1), _ticks(A, N, 2))
        self.play(FadeIn(copy), run_time=0.5)
        self.bring_to_front(dN, lN)
        self.play(Rotate(copy, angle=-PI, about_point=N), run_time=2.4)
        _landed(copy[0], [C, M2, N], "the copy lands on C M′ N")
        dM2 = Dot(M2, radius=0.07)
        lM2 = tag("M′", 30).move_to(M2 + 0.48 * RIGHT)
        _on_screen("F17 labels on screen", lM2, lM, names)
        self.play(FadeIn(dM2), FadeIn(lM2), run_time=0.5)
        self.hold(0.3)

        # MB and M′C: equal and parallel, so MBCM′ is a parallelogram
        par = mk([M, B, C, M2], YELLOW_E, op=0.2, stroke_width=0)
        chev = VGroup(_chevron(M, B, at=0.24), _chevron(M2, C, at=0.24))
        side = Line(C, M2, color=WHITE, stroke_width=4)
        self.add(par)
        self.bring_to_back(par)
        self.play(FadeIn(par), FadeIn(chev), Create(side), run_time=0.9)

        # BC slides along BM onto MM′
        bc = Line(B, C, color=GREEN_B, stroke_width=9)
        self.play(Create(bc), run_time=0.6)
        self.play(bc.animate.shift(v), run_time=1.8)
        check(close(bc.get_start(), M) and close(bc.get_end(), M2),
              "BC landed on MM′")
        self.bring_to_front(dM, dN, dM2)
        halves = VGroup(_ticks(M, N, 3, YELLOW_B, size=0.16),
                        _ticks(N, M2, 3, YELLOW_B, size=0.16))
        self.play(FadeIn(halves), run_time=0.6)
        self.play(Write(caption("MN ∥ BC,    MN  =  ½ BC", 36)))
        self.hold(2.2)


# ===================================================== F15 KITE

class F15_KiteArea(Board):
    """The diagonals cross at right angles and cut the kite into four right
    triangles; the rectangle around it, with sides along the diagonals,
    is cut by the kite's sides into four pairs. In each pair the outer
    triangle is the inner one turned half a turn about the midpoint of
    their common side, so the kite is exactly half of the d₁ × d₂
    rectangle. (Nothing uses the kite's symmetry: a rhombus, or any
    convex quadrilateral with perpendicular diagonals, works the same.)"""

    def construct(self):
        t, u, w = 2.8, 1.6, 2.0
        F = Frame(-w - 1.3, w + 1.3, -u - 0.75, t + 0.6)
        T, R, Bo, L, O = [F.P(q) for q in
                          ((0, t), (w, 0), (0, -u), (-w, 0), (0, 0))]
        corners = [F.P(q) for q in ((w, t), (-w, t), (-w, -u), (w, -u))]
        inner = [[O, R, T], [O, T, L], [O, L, Bo], [O, Bo, R]]
        mids = [(R + T) / 2, (T + L) / 2, (L + Bo) / 2, (Bo + R) / 2]
        outer = [[2 * c - q for q in tri] for c, tri in zip(mids, inner)]
        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D]
        rect = [corners[2], corners[3], corners[0], corners[1]]

        check(abs(np.dot(R - L, T - Bo)) < 1e-9, "diagonals perpendicular")
        for tri, out, cor in zip(inner, outer, corners):
            check(close(out[0], cor), "the turned triangle fills a corner")
        check(_tiles(inner + outer, rect),
              "four triangles and their turned copies tile the rectangle")
        check(abs(area([T, L, Bo, R]) - area(rect) / 2) < 1e-9,
              "kite = half the rectangle")
        check(abs(area([(0, t), (-w, 0), (0, -u), (w, 0)])
                  - 0.5 * (2 * w) * (t + u)) < 1e-9, "A = ½ d₁ d₂")
        for tri, c in zip(inner, mids):
            _sweep_ok(tri, c, -PI, "turning triangle stays on screen")

        kite = mk([T, L, Bo, R], BLUE_D)
        self.play(FadeIn(kite), run_time=1.0)
        d1 = Line(L, R, color=YELLOW_B, stroke_width=4)
        d2 = Line(Bo, T, color=GREEN_B, stroke_width=4)
        ram = _right_mark(O, RIGHT, UP, 0.22, WHITE)
        l1 = tag("d₁", 34, YELLOW_A).move_to(F.P((-w / 2, 0)) + DOWN * 0.31)
        l2 = tag("d₂", 34, GREEN_A).move_to(F.P((0, t / 2)) + LEFT * 0.4)
        self.play(Create(d1), Create(d2), Create(ram), FadeIn(l1), FadeIn(l2),
                  run_time=1.1)
        self.hold(0.4)

        pieces = [mk(tri, c) for tri, c in zip(inner, cols)]
        self.remove(kite)
        self.add(*pieces)
        self.bring_to_front(d1, d2, ram, l1, l2)
        frame = _dashed(rect, GREY_B, 2.5, 56)
        self.play(*[FadeIn(p) for p in pieces], Create(frame), run_time=0.9)
        self.hold(0.3)

        # each copy turns half a turn about the midpoint of its kite side
        copies = [mk(tri, c, op=0.42) for tri, c in zip(inner, cols)]
        dots = VGroup(*[Dot(c, radius=0.06, color=WHITE) for c in mids])
        self.add(*copies)
        self.bring_to_front(d1, d2, ram, l1, l2)
        self.bring_to_front(dots)
        self.play(FadeIn(dots), run_time=0.4)
        for pair in ((0, 2), (1, 3)):           # opposite corners together
            self.play(*[Rotate(copies[i], angle=-PI, about_point=mids[i])
                        for i in pair], run_time=1.9)
        for cp, out in zip(copies, outer):
            _landed(cp, out, "each copy fills its corner")
        self.play(FadeOut(dots), run_time=0.3)

        box = Polygon(*rect, stroke_color=YELLOW_B, stroke_width=5)
        bar1, lab1 = _dim(rect[0], rect[1], "d₁", DOWN, gap=0.32,
                          color=YELLOW_B)
        bar2, lab2 = _dim(rect[1], rect[2], "d₂", RIGHT, gap=0.32,
                          color=GREEN_B)
        _on_screen("F15 labels on screen", lab1, lab2, bar1, bar2)
        self.remove(frame)
        self.play(Create(box), FadeIn(bar1), FadeIn(lab1), FadeIn(bar2),
                  FadeIn(lab2), run_time=1.0)
        self.play(Write(caption("A  =  ½ · d₁ · d₂", 36)))
        self.hold(2.2)


# ===================================================== F29 EQUILATERAL

class F29_EquilateralArea(Board):
    """The altitude halves the equilateral triangle into two right
    triangles with legs s/2 and h, where h² + (s/2)² = s², h = (√3/2)s.
    The halves are mirror images: turned over about the altitude, the left
    one covers the right one exactly. Turned half a turn about the
    midpoint of the hypotenuse, it completes an s/2 × (√3/2)s rectangle."""

    def construct(self):
        s = 5.0
        h = s * np.sqrt(3) / 2
        F = Frame(-s / 2 - 1.15, s / 2 + 4.55, -0.95, h + 0.45)
        A, B, C, O = [F.P(q) for q in ((-s / 2, 0), (s / 2, 0), (0, h),
                                       (0, 0))]
        Mb = (B + C) / 2
        X = 2 * Mb - O
        flipped = [_mirror(q, O, UP) for q in (A, O, C)]
        turned = [2 * Mb - q for q in flipped]

        check(close(flipped, [B, O, C]),
              "turned over the altitude, the left half covers the right")
        check(close(turned, [C, X, B]), "the half-turn completes the box")
        check(_tiles([[O, B, C], turned], [O, B, X, C]),
              "the two halves tile the rectangle")
        check(abs(h ** 2 + (s / 2) ** 2 - s ** 2) < 1e-9,
              "h² + (s/2)² = s²")
        check(abs(area([(-s / 2, 0), (s / 2, 0), (0, h)])
                  - (s / 2) * h) < 1e-9
              and abs((s / 2) * h - np.sqrt(3) / 4 * s * s) < 1e-9,
              "A = s/2 · (√3/2)s = (√3/4)s²")
        _sweep_ok([O, B, C], Mb, PI, "the turning half stays on screen")

        tri = mk([A, B, C], BLUE_D)
        G = (A + B + C) / 3
        sL = tag("s", 32).move_to((A + C) / 2 + 0.36 * _out(A, C, G))
        sR = tag("s", 32).move_to((B + C) / 2 + 0.36 * _out(B, C, G))
        sB = tag("s", 32).move_to(O + DOWN * 0.38)
        self.play(FadeIn(tri), FadeIn(sL), FadeIn(sR), FadeIn(sB),
                  run_time=1.2)
        self.hold(0.4)

        # the altitude halves the base
        alt = DashedLine(C, O, color=WHITE, stroke_width=2.5,
                         dash_length=0.09)
        ra = _right_mark(O, UP, RIGHT, 0.22, WHITE)
        hL = tag("s/2", 30).move_to((A + O) / 2 + DOWN * 0.38)
        hR = tag("s/2", 30).move_to((O + B) / 2 + DOWN * 0.38)
        self.play(Create(alt), Create(ra), FadeOut(sB), FadeIn(hL),
                  FadeIn(hR), run_time=1.0)
        h_lab = tag("h", 32).move_to(F.P((0, h * 0.45)) + LEFT * 0.32)
        r1 = tag("h² + (s/2)²  =  s²", 30)
        r2 = tag("h  =  (√3/2) s", 30, YELLOW_B)
        r1.move_to(F.P((s / 2 + 1.55, h * 0.62)), aligned_edge=LEFT)
        r2.next_to(r1, DOWN, buff=0.38, aligned_edge=LEFT)
        _on_screen("F29 readout on screen", r1, r2)
        self.play(FadeIn(h_lab), FadeIn(r1), run_time=0.8)
        self.play(FadeIn(r2), run_time=0.6)
        self.hold(0.4)

        # cut along it: two halves
        left, right = mk([A, O, C], BLUE_D), mk([O, B, C], BLUE_D)
        cut = Line(C, O, color=WHITE, stroke_width=3)
        self.remove(tri)
        self.add(right, left, cut, ra, h_lab)
        self.play(FadeOut(alt), left.animate.set_fill(ORANGE),
                  right.animate.set_fill(TEAL_D), FadeOut(sL), run_time=0.7)

        # turned over about the altitude: it covers the other half
        ghost = _dashed([A, O, C], GREY_B, 2, 30)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.bring_to_front(h_lab)
        self.play(Rotate(left, angle=PI, axis=UP, about_point=O),
                  run_time=2.2)
        _landed(left, [B, O, C], "the left half covers the right half")
        self.hold(0.3)

        # then half a turn about the midpoint of the hypotenuse
        dMb = Dot(Mb, radius=0.07, color=WHITE)
        tks = VGroup(_ticks(B, Mb, 1), _ticks(Mb, C, 1))
        self.play(FadeOut(sR), FadeIn(dMb), FadeIn(tks), run_time=0.5)
        self.bring_to_front(hR, hL)
        self.play(Rotate(left, angle=PI, about_point=Mb), run_time=2.2)
        _landed(left, turned, "the half lands at the top right")
        self.bring_to_front(cut, ra)

        box = Polygon(O, B, X, C, stroke_color=YELLOW_B, stroke_width=5)
        side = tag("(√3/2) s", 30, YELLOW_B)
        side.next_to(cut, LEFT, buff=0.16)
        _on_screen("F29 side label on screen", side)
        self.play(Create(box), FadeOut(dMb), FadeOut(tks), FadeOut(hL),
                  FadeOut(ghost), ReplacementTransform(h_lab, side),
                  run_time=1.0)
        self.play(Write(caption("A  =  s/2 · (√3/2) s  =  (√3/4) s²",
                                34)))
        self.hold(2.2)


# ===================================================== F16 INRADIUS

class F16_InradiusArea(Board):
    """Join the incentre I to the vertices: three triangles on the sides
    a, b, c, each of height r (the radius to the point of contact is
    perpendicular to the side). Roll the triangle along the line of BC,
    leaving each piece where its side touches down: the pieces stand side
    by side on a base a + b + c, their apexes on the parallel at height r.
    Slide the outer apexes along that parallel to the middle one (a shear:
    base and height kept): one triangle of base a + b + c = 2s, height r."""

    def construct(self):
        Bm, Cm, Am = np.array([0.0, 0.0]), np.array([5.0, 0.0]), \
            np.array([1.4, 3.0])
        a = float(np.linalg.norm(Cm - Bm))
        b = float(np.linalg.norm(Am - Cm))
        c = float(np.linalg.norm(Bm - Am))
        s = (a + b + c) / 2
        Im = (a * Am + b * Bm + c * Cm) / (2 * s)
        r = area([Am, Bm, Cm]) / s
        F = Frame(-0.55, 2 * s + 0.55, -1.62, 5.18)
        k = F.k
        A, B, C, I = [F.P(q) for q in (Am, Bm, Cm, Im)]

        def foot(P, U, V):
            d = _unit(V - U)
            return U + np.dot(P - U, d) * d

        D, E, Fp = foot(I, B, C), foot(I, C, A), foot(I, A, B)
        angC, angA = _angle(C, A, B), _angle(A, B, C)
        rot1, rot2 = -(PI - angC), -(PI - angA)
        A1 = C + b * k * RIGHT                    # A after the first roll
        I1 = _turn(I, C, rot1)
        B1 = _turn(B, C, rot1)
        B2 = A1 + c * k * RIGHT                   # B after the second roll
        I2 = _turn(I1, A1, rot2)
        Pm = I1                                   # where the apexes meet

        check(abs(np.linalg.norm(I - D) - r * k) < 1e-9
              and abs(np.linalg.norm(I - E) - r * k) < 1e-9
              and abs(np.linalg.norm(I - Fp) - r * k) < 1e-9,
              "I is r from all three sides")
        check(abs(area([Bm, Cm, Im]) + area([Cm, Am, Im])
                  + area([Am, Bm, Im]) - r * s) < 1e-9,
              "½ar + ½br + ½cr = rs = [ABC]")
        check(close(_turn(A, C, rot1), A1),
              "first roll: CA lands on the line")
        check(close(_turn(B1, A1, rot2), B2),
              "second roll: AB lands on the line")
        check(abs(I1[1] - B[1] - r * k) < 1e-9
              and abs(I2[1] - B[1] - r * k) < 1e-9, "apexes at height r")
        check(abs(np.linalg.norm(B2 - B) - 2 * s * k) < 1e-9,
              "the bases add up to a + b + c")
        check(abs(area([B, C, Pm]) - area([B, C, I])) < 1e-9
              and abs(area([A1, B2, Pm]) - area([A1, B2, I2])) < 1e-9,
              "the shears keep each area")
        check(abs(area([B, B2, Pm]) / k ** 2 - r * s) < 1e-9,
              "one triangle: ½ · 2s · r")
        _sweep_ok([C, A, I, B], C, rot1, "first roll stays on screen")
        _sweep_ok([A1, B1, I1], A1, rot2, "second roll stays on screen")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        G = (A + B + C) / 3
        la = tag("a", 32).move_to((B + C) / 2 + DOWN * 0.36)
        lb = tag("b", 32).move_to((C + A) / 2 + 0.36 * _out(C, A, G))
        lc = tag("c", 32).move_to((A + B) / 2 + 0.36 * _out(A, B, G))
        self.play(Create(tri), FadeIn(la), FadeIn(lb), FadeIn(lc),
                  run_time=1.1)

        circ = Circle(radius=r * k, color=YELLOW_B, stroke_width=3).move_to(I)
        rad = [Line(I, X, color=YELLOW_B, stroke_width=3) for X in (D, E, Fp)]
        ram = [_right_mark(X, I - X, side, 0.15, YELLOW_B)
               for X, side in ((D, C - B), (E, A - C), (Fp, B - A))]
        rlab = VGroup(*[tag("r", 26, YELLOW_B).move_to(
            (I + X) / 2 + 0.2 * _unit(side)) for X, side in
            ((D, C - B), (E, A - C), (Fp, B - A))])
        dI = Dot(I, radius=0.05, color=YELLOW_B)
        self.play(Create(circ), FadeIn(dI), run_time=0.9)
        self.play(*[Create(x) for x in rad + ram], FadeIn(rlab), run_time=0.9)
        self.hold(0.4)

        # join I to the vertices: three triangles of height r
        p1, p2, p3 = (mk([B, C, I], BLUE_D, op=0.8),
                      mk([C, A, I], TEAL_D, op=0.8),
                      mk([A, B, I], ORANGE, op=0.8))
        spokes = VGroup(*[Line(I, X, color=WHITE, stroke_width=2.5)
                          for X in (A, B, C)])
        self.add(p1, p2, p3)
        self.bring_to_back(p1, p2, p3)
        self.play(Create(spokes), FadeIn(p1), FadeIn(p2), FadeIn(p3),
                  run_time=1.0)
        self.hold(0.5)

        # roll the triangle along the line of BC
        ghost = _dashed([A, B, C], GREY_B, 2, 48)
        g2 = VGroup(p2, rad[1], ram[1])
        g3 = VGroup(p3, rad[2], ram[2])
        self.remove(spokes, tri)
        self.add(ghost)
        self.bring_to_back(ghost)
        self.play(FadeOut(circ), FadeOut(dI), FadeOut(rlab), FadeOut(lb),
                  FadeOut(lc), run_time=0.5)
        self.play(Rotate(VGroup(g2, g3), angle=rot1, about_point=C),
                  run_time=2.2)
        _landed(p2, [C, A1, I1], "piece b lands on the line")
        self.play(Rotate(g3, angle=rot2, about_point=A1), run_time=1.8)
        _landed(p3, [A1, B2, I2], "piece c lands on the line")

        par = guide(F.P((-0.35, r)), F.P((2 * s + 0.35, r)), GREY_B)
        lb2 = tag("b", 32).move_to((C + A1) / 2 + DOWN * 0.36)
        lc2 = tag("c", 32).move_to((A1 + B2) / 2 + DOWN * 0.36)
        r_lab = tag("r", 28, YELLOW_B).next_to(rad[1], RIGHT, buff=0.1)
        self.play(FadeOut(ghost), Create(par), FadeIn(lb2), FadeIn(lc2),
                  FadeIn(r_lab), run_time=0.9)
        self.hold(0.4)

        # shear: the outer apexes slide along the parallel to the middle one
        self.play(FadeOut(VGroup(rad[0], ram[0], rad[2], ram[2])),
                  run_time=0.4)
        self.play(Transform(p1, mk([B, C, Pm], BLUE_D, op=0.8)),
                  Transform(p3, mk([A1, B2, Pm], ORANGE, op=0.8)),
                  run_time=2.2)
        self.bring_to_front(rad[1], ram[1])

        whole = Polygon(B, B2, Pm, stroke_color=YELLOW_B, stroke_width=5)
        bar, lab = _dim(B, B2, "a + b + c  =  2s", DOWN, gap=0.78)
        _on_screen("F16 labels on screen", lab, bar, lc2)
        self.play(Create(whole), FadeIn(bar), FadeIn(lab), run_time=1.0)
        self.play(Write(caption("A  =  ½ · (a + b + c) · r  =  r · s",
                                36)))
        self.hold(2.2)
