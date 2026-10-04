# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes_w1f.py — proofs without words: A9 Thābit's two cuts, A10 Liu
Hui's out-in dissection, A11 Leonardo's hexagons, A15 the lunes of
Alhazen, A17 similar areas, A18 the Pythagorean tiling, A23 Bhaskara's
chair, E16 equal tangents, E17 the parabola's reflection property and
E24 π between two hexagons.

Every piece moves rigidly — a translation, a Rotate about a point, or (where
the argument is a mirror symmetry) a fold about an in-plane axis, which is
a rotation in space — and every claimed landing, tiling and equality is
checked numerically with check() before anything is drawn, so a wrong
construction fails the render. The Pythagoras scenes use the legs 2.4 and
3.7: no special triangle, so no coincidence can hide an error.
"""

from pww_kit import *


LEG_A, LEG_B = 2.4, 3.7          # the right triangle of the A-scenes (a < b)
ZHU, QING = "#E0452C", "#1F9E8C"  # vermilion and blue-green


# ------------------------------------------------------------ private helpers

def _u(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _ang(v):
    """Direction angle of a 2D/3D vector."""
    return float(np.arctan2(v[1], v[0]))


def _dir(t):
    return np.array([np.cos(t), np.sin(t)])


def _rot(p, t, about=(0.0, 0.0)):
    """The 2D point p turned by the angle t about the point `about`."""
    p = np.asarray(p, float)[:2]
    o = np.asarray(about, float)[:2]
    cs, sn = np.cos(t), np.sin(t)
    d = p - o
    return o + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])


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
    P = [np.asarray(p, float)[:2] for p in P]
    Q = [np.asarray(q, float)[:2] for q in Q]
    if len(P) != len(Q):
        return False
    n = len(P)
    for d in (1, -1):
        for s in range(n):
            if all(close(P[i], Q[(s + d * i) % n], tol) for i in range(n)):
                return True
    return False


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1 = (p - v) / np.linalg.norm(p - v)
    u2 = (q - v) / np.linalg.norm(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ticks(p, q, n=1, color=WHITE, size=0.12, width=2.5, at=0.5):
    """n short tick marks across the screen segment pq (equal lengths)."""
    p, q = to3(p), to3(q)
    d = (q - p) / np.linalg.norm(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    g = VGroup()
    for k in range(n):
        o = c + d * 0.09 * (k - (n - 1) / 2)
        g.add(Line(o - nrm * size, o + nrm * size, color=color,
                   stroke_width=width))
    return g


def _lbl(s, size=26, color=WHITE, bg=0.0):
    """tag() with an optional dark backing box for text sitting on fills."""
    t = tag(s, size, color)
    if bg > 0:
        box = BackgroundRectangle(t, color=BLACK, fill_opacity=bg, buff=0.06)
        return VGroup(box, t)
    return t


def _curve(pts, color=WHITE, width=3):
    """Open polyline through screen points (fine sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _arc(o, r, t0, t1, n=90):
    """n points of the circle (o, r) from angle t0 to t1."""
    o = np.asarray(o, float)[:2]
    return [o + r * _dir(t) for t in np.linspace(t0, t1, n)]


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame. In the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=(q - p) / np.linalg.norm(q - p),
                  about_point=p, **kw)


def _clip(subj, clipp):
    """Sutherland–Hodgman: polygon subj ∩ convex polygon clipp (CCW)."""
    def inside(p, a1, a2):
        return ((a2[0] - a1[0]) * (p[1] - a1[1])
                - (a2[1] - a1[1]) * (p[0] - a1[0])) >= -1e-12

    def inter(p1, p2, a1, a2):
        d1, d2 = p2 - p1, a2 - a1
        den = d1[0] * d2[1] - d1[1] * d2[0]
        t = ((a1[0] - p1[0]) * d2[1] - (a1[1] - p1[1]) * d2[0]) / den
        return p1 + t * d1

    out = [np.asarray(p, float)[:2] for p in subj]
    n = len(clipp)
    for i in range(n):
        a1 = np.asarray(clipp[i], float)[:2]
        a2 = np.asarray(clipp[(i + 1) % n], float)[:2]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a1, a2):
                if not inside(s, a1, a2):
                    out.append(inter(s, e, a1, a2))
                out.append(e)
            elif inside(s, a1, a2):
                out.append(inter(s, e, a1, a2))
            s = e
    clean = []
    for p in out:
        if not clean or np.linalg.norm(p - clean[-1]) > 1e-9:
            clean.append(p)
    if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) < 1e-9:
        clean.pop()
    return clean


def _cross2(u, v):
    return float(u[0] * v[1] - u[1] * v[0])


def _free_dir(X, towards):
    """Unit 2D vector at X along the middle of the widest angular gap
    between the directions to the points `towards` (to park a label)."""
    X = np.asarray(X, float)[:2]
    angs = sorted(_ang(np.asarray(q, float)[:2] - X) % TAU for q in towards)
    best, mid = -1.0, 0.0
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        if a1 - a0 > best:
            best, mid = a1 - a0, (a0 + a1) / 2
    return _dir(mid)


def _in_safe(mobs, tol=0.03):
    """Every mobject lies inside the content area (the band below
    SAFE_BOTTOM belongs to the caption)."""
    for m in mobs:
        if not m.has_points() and not m.submobjects:
            continue
        if (m.get_left()[0] < -SAFE_X - tol or m.get_right()[0] > SAFE_X + tol
                or m.get_bottom()[1] < SAFE_BOTTOM - tol
                or m.get_top()[1] > SAFE_TOP + tol):
            return False
    return True


def _dim(p, q, side, color=GREY_A, off=0.32, tick=0.12, width=2.5):
    """Dimension bracket for the screen segment pq, drawn `off` away on the
    side of the unit screen vector `side`, with end ticks towards pq."""
    p, q, sd = to3(p), to3(q), to3(side)
    sd = sd / np.linalg.norm(sd)
    p1, q1 = p + sd * off, q + sd * off
    return VGroup(Line(p1, q1, color=color, stroke_width=width),
                  Line(p1, p1 - sd * tick, color=color, stroke_width=width),
                  Line(q1, q1 - sd * tick, color=color, stroke_width=width))


def _right_triangle_ok(T, a, b):
    """T is a right triangle with legs a, b (any order) and hypotenuse c."""
    T = [np.asarray(p, float)[:2] for p in T]
    s = sorted(np.linalg.norm(T[i] - T[(i + 1) % 3]) for i in range(3))
    return (abs(s[0] - min(a, b)) < 1e-9 and abs(s[1] - max(a, b)) < 1e-9
            and abs(s[2] - np.hypot(a, b)) < 1e-9)


def _mirror(p, u, v):
    """Mirror image of the 2D point p in the line uv."""
    p, u, v = (np.asarray(x, float)[:2] for x in (p, u, v))
    d = _u(v - u)
    f = u + np.dot(p - u, d) * d
    return 2 * f - p


def _inside_or_on(p, poly, tol=1e-9):
    """p inside or on the boundary of the convex polygon poly (any sense)."""
    s = np.sign(area(poly))
    n = len(poly)
    for i in range(n):
        e = np.asarray(poly[(i + 1) % n], float) - np.asarray(poly[i], float)
        if s * _cross2(e, np.asarray(p, float) - np.asarray(poly[i], float)) < -tol:
            return False
    return True


def _clip_line(p, d, x0, x1, y0, y1):
    """The part of the line p + t·d inside the rectangle, or None."""
    t0, t1 = -1e9, 1e9
    for pc, dc, lo, hi in ((p[0], d[0], x0, x1), (p[1], d[1], y0, y1)):
        if abs(dc) < 1e-12:
            if pc < lo or pc > hi:
                return None
            continue
        ta, tb = (lo - pc) / dc, (hi - pc) / dc
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    if t1 - t0 < 1e-6:
        return None
    return p + t0 * d, p + t1 * d


# ======================================================== A15 LUNES

class A15_AlhazenLunes(Board):
    """The semicircle on the hypotenuse passes through the right angle
    (Thales). It is the triangle plus the two circular segments cut off by
    the legs; each semicircle on a leg is its lune plus the same segment.
    The three semicircles are similar, so their areas are πa²/8, πb²/8 and
    πc²/8, and the two small ones together equal the big one. Take the two
    shared segments away from both sides: the lunes equal the triangle."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        R = c / 2
        O = np.zeros(2)
        A, B = np.array([-R, 0.0]), np.array([R, 0.0])
        C = A + b * np.array([b / c, a / c])               # AC = b, BC = a
        check(close(np.linalg.norm(C - O), R), "C on the circle on AB")
        check(close(np.linalg.norm(C - A), b)
              and close(np.linalg.norm(C - B), a), "legs AC = b, BC = a")
        check(abs(np.dot(A - C, B - C)) < 1e-9, "right angle at C")
        tC = _ang(C - O)

        def semi(P, Q, away, n=120):
            """Semicircle on the diameter PQ, from P to Q, bulging away
            from the point `away`."""
            M = (P + Q) / 2
            r = np.linalg.norm(Q - P) / 2
            t0 = _ang(P - M)
            sg = 1.0 if np.dot(_dir(t0 + PI / 2), M - away) > 0 else -1.0
            pts = [M + r * _dir(t0 + sg * PI * s)
                   for s in np.linspace(0.0, 1.0, n)]
            check(close(pts[-1], Q), "semicircle ends at Q")
            return pts

        big_CA = _arc(O, R, tC, PI, 100)            # minor arc C -> A
        big_BC = _arc(O, R, 0.0, tC, 70)            # minor arc B -> C
        big_all = _arc(O, R, 0.0, PI, 160)          # B -> C -> A
        sb = semi(A, C, B)                          # A -> C, outwards
        sa = semi(B, C, A)                          # B -> C, outwards
        lune_b = sb + big_CA[1:-1]                  # outer arc, inner arc
        lune_a = sa + big_BC[::-1][1:-1]
        Mb, Ma = (A + C) / 2, (B + C) / 2

        # ---- the claims
        def seg_area(phi):
            return R * R / 2 * (phi - np.sin(phi))
        Lb = PI * b * b / 8 - seg_area(PI - tC)
        La = PI * a * a / 8 - seg_area(tC)
        check(abs(PI * a * a / 8 + PI * b * b / 8 - PI * c * c / 8) < 1e-9,
              "the semicircles on the legs add up to the one on c")
        check(abs(seg_area(PI - tC) + seg_area(tC) + a * b / 2
                  - PI * c * c / 8) < 1e-9,
              "big semicircle = triangle + two segments")
        check(abs(La + Lb - a * b / 2) < 1e-9, "lunes = triangle, exactly")
        check(abs(abs(area(lune_a)) + abs(area(lune_b)) - a * b / 2)
              < 2e-3 * a * b, "lunes = triangle, as drawn")
        check(all(np.linalg.norm(p - Mb) <= b / 2 + 1e-9 for p in big_CA)
              and all(np.linalg.norm(p - Ma) <= a / 2 + 1e-9 for p in big_BC),
              "each segment lies inside the semicircle on its leg")
        check(all(np.linalg.norm(p - O) >= R - 1e-9 for p in sb + sa),
              "the leg semicircles run outside the big circle")

        # ---- screen: one figure in the middle, later two side by side
        pts = np.array(sb + sa + big_all)
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y1 = pts[:, 1].max()
        k, gap = 1.06, 1.5
        w = (x1 - x0) * k
        dx = (w + gap) / 2
        check(dx + w / 2 <= SAFE_X, "two copies fit side by side")
        y_base = -0.5

        def S(p):
            return np.array([(p[0] - (x0 + x1) / 2) * k, p[1] * k + y_base, 0.0])
        check(S((0, y1))[1] <= SAFE_TOP, "figure below the top edge")

        tri = Polygon(S(A), S(B), S(C), stroke_color=WHITE, stroke_width=3)
        ra = _ra(S(C), S(A), S(B), 0.18)
        I = (a * A + b * B + c * C) / (a + b + c)            # incentre
        la = tag("a", 26).move_to(S(Ma) + 0.3 * _u(S(I) - S(Ma)))
        lb = tag("b", 26).move_to(S(Mb) + 0.3 * _u(S(I) - S(Mb)))
        lc = tag("c", 26).move_to(S(O) + DOWN * 0.3)
        arc_c = _curve([S(p) for p in big_all], GREY_A, 3)
        arc_a = _curve([S(p) for p in sa], WHITE, 3)
        arc_b = _curve([S(p) for p in sb], WHITE, 3)
        f_la = mk([S(p) for p in lune_a], GOLD_D, 0.85, stroke_width=0)
        f_lb = mk([S(p) for p in lune_b], GOLD_D, 0.85, stroke_width=0)
        # each lune's label on its widest part, halfway across
        na, nb = _u(Ma - C), _u(Mb - C)
        na = np.array([-na[1], na[0]])
        nb = np.array([-nb[1], nb[0]])
        na = na if np.dot(na, Ma - A) > 0 else -na              # away from A
        nb = nb if np.dot(nb, Mb - B) > 0 else -nb              # away from B
        sag_a = R - np.linalg.norm(Ma - O)
        sag_b = R - np.linalg.norm(Mb - O)
        pL1 = Ma + na * (sag_a + a / 2) / 2
        pL2 = Mb + nb * (sag_b + b / 2) / 2
        L1 = tag("L₁", 26, BLACK).move_to(S(pL1))
        L2 = tag("L₂", 26, BLACK).move_to(S(pL2))

        self.play(Create(tri), Create(ra), FadeIn(la), FadeIn(lb), FadeIn(lc),
                  run_time=1.1)
        self.play(Create(arc_c), run_time=1.2)               # through C
        self.play(Create(arc_a), Create(arc_b), run_time=1.2)
        self.add(f_la, f_lb)
        self.bring_to_back(f_la, f_lb)
        self.play(FadeIn(f_la), FadeIn(f_lb), FadeIn(L1), FadeIn(L2),
                  run_time=0.9)
        self.hold(0.6)

        # ---- two copies: leg semicircles on the left, the big one right
        fig = VGroup(f_la, f_lb, tri, ra, arc_c, arc_a, arc_b, la, lb, lc,
                     L1, L2)
        self.play(fig.animate.shift(LEFT * dx), run_time=1.2)

        def SL(p):
            return S(p) + LEFT * dx

        def SR(p):
            return S(p) + RIGHT * dx

        right = VGroup(tri.copy(), arc_c.copy())
        self.play(right.animate.shift(RIGHT * 2 * dx), run_time=1.2)

        # left: each leg semicircle = its lune + a segment of the big disc
        GREY_SEG = GREY_C
        segL = [mk([SL(p) for p in big_CA], GREY_SEG, 0.9, stroke_width=0),
                mk([SL(p) for p in big_BC], GREY_SEG, 0.9, stroke_width=0)]
        semL = VGroup(
            Polygon(*[SL(p) for p in sa], stroke_color=TEAL_B, stroke_width=5),
            Polygon(*[SL(p) for p in sb], stroke_color=TEAL_B, stroke_width=5))
        # right: the big semicircle = the triangle + the same two segments
        segR = [mk([SR(p) for p in big_CA], GREY_SEG, 0.9, stroke_width=0),
                mk([SR(p) for p in big_BC], GREY_SEG, 0.9, stroke_width=0)]
        triR = mk([SR(A), SR(B), SR(C)], BLUE_D, 0.9, stroke_width=0)
        semR = Polygon(*[SR(p) for p in big_all], stroke_color=TEAL_B,
                       stroke_width=5)
        for m in segL + segR + [triR]:
            self.add(m)
            self.bring_to_back(m)
            m.set_opacity(0)
        y_ro = S((0, 0))[1] - 0.95
        roL = tag("πa²/8 + πb²/8", 28).move_to([-dx, y_ro, 0])
        roR = tag("πc²/8", 28).move_to([dx, y_ro, 0])
        eq1 = tag("=", 34).move_to([0, y_ro, 0])
        eq2 = tag("=", 40).move_to([0, S((0, R / 2))[1], 0])
        check(roL.get_left()[0] > -SAFE_X and roR.get_right()[0] < SAFE_X,
              "readouts inside the frame")
        self.play(*[m.animate.set_fill(opacity=0.9) for m in segL],
                  Create(semL), FadeOut(lc), run_time=1.0)
        self.play(FadeIn(roL), run_time=0.6)
        self.play(*[m.animate.set_fill(opacity=0.9) for m in segR + [triR]],
                  Create(semR), run_time=1.0)
        self.play(FadeIn(roR), FadeIn(eq1), FadeIn(eq2), run_time=0.6)
        self.hold(0.8)

        # the same two segments on both sides: take them away
        self.play(*[m.animate.set_fill(YELLOW_B, opacity=0.9)
                    for m in segL + segR], run_time=0.5)
        self.play(*[m.animate.set_fill(GREY_SEG, opacity=0.9)
                    for m in segL + segR], run_time=0.4)
        self.play(*[FadeOut(m) for m in segL + segR], FadeOut(semL),
                  FadeOut(semR), run_time=1.0)
        roL2 = tag("L₁ + L₂", 30).move_to(roL)
        roR2 = tag("½ab", 30).move_to(roR)
        self.play(Transform(roL, roL2), Transform(roR, roR2), run_time=0.8)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("L₁ + L₂  =  ½ab", 36)))
        self.hold(2.2)


# ======================================================== A23 BHASKARA

class A23_BhaskaraChair(Board):
    """Zhao Shuang's figure on the tilted square c²: four copies of the
    triangle around a central square of side b − a. Two of the triangles
    slide straight across (translations, no turning); the same five pieces
    then fill the chair-shaped union of b² and a² standing side by side:
    c² = (b − a)² + 2ab = a² + b²."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        Q = [(a, 0), (a + b, a), (b, a + b), (0, b)]           # c-square
        T1 = [(0, b), (b, b), (b, a + b)]                      # slides
        T2 = [(b, a + b), (b, a), (a + b, a)]                  # slides
        T3 = [(a + b, a), (a, a), (a, 0)]
        T4 = [(a, 0), (a, b), (0, b)]
        In = [(a, a), (b, a), (b, b), (a, b)]
        v1, v2 = np.array([a, -b]), np.array([-b, -a])
        T1e = [np.array(p, float) + v1 for p in T1]
        T2e = [np.array(p, float) + v2 for p in T2]
        chair = [(0, 0), (a + b, 0), (a + b, a), (b, a), (b, b), (0, b)]
        Sb = [(0, 0), (b, 0), (b, b), (0, b)]
        Sa = [(b, 0), (a + b, 0), (a + b, a), (b, a)]
        Qa = [np.array(p, float) for p in Q]
        for i in range(4):
            p, q, r = Qa[i], Qa[(i + 1) % 4], Qa[(i + 2) % 4]
            check(close(np.linalg.norm(q - p), c)
                  and abs(np.dot(q - p, r - q)) < 1e-9, "c² is a square")
        check(all(_right_triangle_ok(T, a, b) for T in (T1, T2, T3, T4)),
              "four copies of the triangle")
        check(_tiles_exactly([T1, T2, T3, T4, In], Q),
              "four triangles and (b−a)² tile c²")
        check(_same_poly(T1e, [(a, 0), (a + b, 0), (a + b, a)])
              and _same_poly(T2e, [(0, b), (0, 0), (a, 0)]),
              "the two slides land on the chair's lower corners")
        check(_tiles_exactly([T1e, T2e, T3, T4, In], chair),
              "the same five pieces tile the chair")
        check(_tiles_exactly([Sb, Sa], chair), "chair = b² and a² side by side")
        check(abs(area(In) - (b - a) ** 2) < 1e-9
              and abs(abs(area(chair)) - (a * a + b * b)) < 1e-9
              and abs(abs(area(Q)) - c * c) < 1e-9, "areas")

        F = Frame(-0.75, a + b + 0.75, -0.95, a + b + 0.2)
        P, k = F.P, F.k
        cols = {"T1": ORANGE, "T2": RED_D, "T3": BLUE_D, "T4": TEAL_D}
        m1 = F.poly(T1, cols["T1"])
        m2 = F.poly(T2, cols["T2"])
        m3 = F.poly(T3, cols["T3"])
        m4 = F.poly(T4, cols["T4"])
        mIn = F.poly(In, YELLOW_E, 0.92)
        lIn = tag("(b−a)²", 22, BLACK).move_to(P(((a + b) / 2, (a + b) / 2)))
        check(lIn.width < (b - a) * k - 0.12, "(b−a)² label fits its square")
        outline = Polygon(*[P(p) for p in Q], stroke_color=YELLOW_B,
                          stroke_width=5)
        ctr = np.array([(a + b) / 2, (a + b) / 2])

        def side_lbl(i, s, size=28, d=0.38):
            p, q = Qa[i], Qa[(i + 1) % 4]
            m = (p + q) / 2
            return tag(s, size, YELLOW_B).move_to(P(m + d * _u(m - ctr)))

        lc1, lc2 = side_lbl(3, "c"), side_lbl(0, "c")
        lc2sq = side_lbl(1, "c²", 32, 0.55)

        self.play(Create(outline), run_time=1.0)
        self.play(FadeIn(lc1), FadeIn(lc2), FadeIn(lc2sq), run_time=0.5)
        self.add(m3, m4, m1, m2)
        for m in (m3, m4, m1, m2):
            m.set_opacity(0)
        self.bring_to_front(outline)
        self.play(LaggedStart(*[m.animate.set_fill(opacity=FILL)
                                .set_stroke(opacity=1)
                                for m in (m3, m4, m1, m2)], lag_ratio=0.3),
                  run_time=1.8)
        self.play(FadeIn(mIn), FadeIn(lIn), run_time=0.6)
        self.hold(0.8)

        ghost = DashedVMobject(outline.copy().set_stroke(GREY_B, 2.5),
                               num_dashes=60)
        self.play(FadeOut(lc1), FadeOut(lc2), FadeOut(lc2sq), FadeOut(outline),
                  FadeIn(ghost), run_time=0.6)
        self.bring_to_front(m2)
        self.play(m2.animate.shift(k * to3(v2)), run_time=1.6)
        self.bring_to_front(m1)
        self.play(m1.animate.shift(k * to3(v1)), run_time=1.6)
        check(close(m1.get_vertices(), [P(p) for p in T1e], 1e-6)
              and close(m2.get_vertices(), [P(p) for p in T2e], 1e-6),
              "slid pieces landed")

        ch = Polygon(*[P(p) for p in chair], stroke_color=YELLOW_B,
                     stroke_width=5)
        div = DashedLine(P((b, 0)), P((b, a)), color=YELLOW_B, stroke_width=3,
                         dash_length=0.1)
        lb2 = _lbl("b²", 34, WHITE, 0.55).move_to(P((b / 2, b * 0.62)))
        la2 = _lbl("a²", 32, WHITE, 0.55).move_to(P((b + a / 2, a * 0.42)))
        sides = VGroup(tag("b", 28).move_to(P((b / 2, -0.42))),
                       tag("a", 28).move_to(P((b + a / 2, -0.42))),
                       tag("b", 28).move_to(P((-0.42, b / 2))),
                       tag("a", 28).move_to(P((a + b + 0.42, a / 2))))
        self.play(FadeOut(ghost), Create(ch), Create(div), run_time=1.0)
        self.play(FadeIn(lb2), FadeIn(la2), FadeIn(sides), run_time=0.7)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("c²  =  (b−a)² + 2ab  =  a² + b²", 34)))
        self.hold(2.2)


# ======================================================== A9 THABIT

class A9_ThabitTwoCuts(Board):
    """a² and b² side by side on one base. From the point at distance b
    from the left end, two cuts run to the far top corners: both are c long
    and they meet at a right angle. Each cuts off a copy of the triangle.
    The left one turns a quarter-turn up about the top corner of a², the
    right one a quarter-turn up about the top corner of b²: they close the
    square on c, and nothing else has moved."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        h = a * (b - a) / b                    # the left cut crosses x = a
        K = np.array([b, 0.0])
        UL, UR = np.array([0.0, a]), np.array([a + b, b])
        Sa = [(0, 0), (a, 0), (a, a), (0, a)]
        Sb = [(a, 0), (a + b, 0), (a + b, b), (a, b)]
        TLa = [(0, 0), (a, 0), (a, h), (0, a)]     # left triangle, a² part
        TLb = [(a, 0), (b, 0), (a, h)]             # left triangle, b² part
        TR = [(b, 0), (a + b, 0), (a + b, b)]
        Ra = [(0, a), (a, h), (a, a)]              # these two stay
        Rb = [(a, h), (b, 0), (a + b, b), (a, b)]
        Cq = [(0, a), (b, 0), (a + b, b), (a, a + b)]

        def turnL(Pp):
            return [_rot(p, PI / 2, UL) for p in Pp]

        def turnR(Pp):
            return [_rot(p, -PI / 2, UR) for p in Pp]

        check(close(np.linalg.norm(K - UL), c) and close(np.linalg.norm(UR - K), c)
              and abs(np.dot(UL - K, UR - K)) < 1e-9,
              "two cuts of length c at a right angle")
        check(abs(_cross2(K - UL, np.array([a, h]) - UL)) < 1e-9,
              "(a, h) lies on the left cut")
        check(_tiles_exactly([TLa, Ra], Sa) and _tiles_exactly([TLb, TR, Rb], Sb),
              "the cuts split a² and b²")
        check(_right_triangle_ok([(0, 0), K, UL], a, b)
              and _right_triangle_ok(TR, a, b), "both pieces are the triangle")
        check(_same_poly(turnL([(0, 0), K, UL]), [UL, (a, a), (a, a + b)]),
              "left piece turns into the notch above a²")
        check(_same_poly(turnR(TR), [UR, (a, b), (a, a + b)]),
              "right piece turns into the notch above b²")
        final = [Ra, Rb, turnL(TLa), turnL(TLb), turnR(TR)]
        check(_tiles_exactly(final, Cq), "the five parts tile the square on c")
        Cqa = [np.array(p, float) for p in Cq]
        check(all(close(np.linalg.norm(Cqa[(i + 1) % 4] - Cqa[i]), c)
                  for i in range(4)), "square on c")

        F = Frame(-0.6, a + b + 0.6, -1.05, a + b + 0.15)
        P, k = F.P, F.k
        sqA = F.poly(Sa, ORANGE)
        sqB = F.poly(Sb, TEAL_D)
        lA2 = tag("a²", 34).move_to(P((a / 2, a / 2)))
        lB2 = tag("b²", 36).move_to(P((a + b / 2, b / 2)))
        sa_ = tag("a", 28).move_to(P((-0.35, a / 2)))
        sb_ = tag("b", 28).move_to(P((a + b + 0.35, b / 2)))
        g = 0.05 / k
        dimB = _dim(P((0, 0)), P((b - g, 0)), DOWN)
        dimA = _dim(P((b + g, 0)), P((a + b, 0)), DOWN)
        lDb = tag("b", 28).next_to(dimB, DOWN, buff=0.1)
        lDa = tag("a", 28).next_to(dimA, DOWN, buff=0.1)
        dotK = Dot(P(K), radius=0.07, color=YELLOW_B)

        self.play(FadeIn(sqA), FadeIn(sqB), run_time=1.0)
        self.play(FadeIn(lA2), FadeIn(lB2), FadeIn(sa_), FadeIn(sb_),
                  run_time=0.6)
        self.play(Create(dimB), Create(dimA), FadeIn(lDb), FadeIn(lDa),
                  FadeIn(dotK), run_time=0.9)
        self.hold(0.4)

        cut1 = Line(P(K), P(UL), color=WHITE, stroke_width=4)
        cut2 = Line(P(K), P(UR), color=WHITE, stroke_width=4)
        self.play(Create(cut1), Create(cut2), FadeOut(lA2), FadeOut(lB2),
                  run_time=1.0)
        pieces = {n: F.poly(Pp, col) for n, Pp, col in
                  (("TLa", TLa, ORANGE), ("TLb", TLb, TEAL_D), ("TR", TR, TEAL_D),
                   ("Ra", Ra, ORANGE), ("Rb", Rb, TEAL_D))}
        self.remove(sqA, sqB)
        self.add(*pieces.values())
        self.bring_to_back(*pieces.values())
        mid1, mid2 = (K + UL) / 2, (K + UR) / 2
        lc1 = tag("c", 28).move_to(P(mid1 + 0.33 * _u(np.array([a, b]))))
        lc2 = tag("c", 28).move_to(P(mid2 + 0.33 * _u(np.array([-b, a]))))
        raK = _ra(P(K), P(UL), P(UR), 0.22)
        self.play(FadeIn(lc1), FadeIn(lc2), Create(raK), run_time=0.6)
        self.remove(cut1, cut2)
        self.hold(0.3)
        # the base and the outer sides are about to move: drop their labels
        self.play(FadeOut(VGroup(dimA, dimB, lDa, lDb, sa_, sb_)), run_time=0.5)

        def quarter(mob, pivot, sign, start_dir):
            t0 = _ang(start_dir)
            arc = Arc(radius=0.55, start_angle=t0, angle=sign * PI / 2,
                      arc_center=P(pivot), color=YELLOW_B, stroke_width=5)
            deg = tag("90°", 24, YELLOW_B).move_to(
                P(pivot) + 0.95 * to3(_dir(t0 + sign * PI / 4)))
            dot = Dot(P(pivot), radius=0.07, color=YELLOW_B)
            self.bring_to_front(mob)
            self.play(Create(arc), FadeIn(deg), FadeIn(dot), run_time=0.5)
            self.bring_to_front(arc, deg, dot)
            self.play(Rotate(mob, angle=sign * PI / 2, about_point=P(pivot)),
                      run_time=1.8)
            self.play(FadeOut(arc), FadeOut(deg), FadeOut(dot), run_time=0.3)

        left = VGroup(pieces["TLa"], pieces["TLb"])
        quarter(left, UL, +1, np.array([0.0, -1.0]))
        quarter(pieces["TR"], UR, -1, np.array([0.0, -1.0]))
        check(close(pieces["TLa"].get_vertices(), [P(p) for p in turnL(TLa)], 1e-6)
              and close(pieces["TR"].get_vertices(), [P(p) for p in turnR(TR)],
                        1e-6), "turned pieces landed")

        sq_c = Polygon(*[P(p) for p in Cq], stroke_color=YELLOW_B, stroke_width=5)
        lC2 = _lbl("c²", 38, WHITE, 0.5).move_to(P(((a + b) / 2, (a + b) / 2)))
        self.play(Create(sq_c), FadeOut(raK), run_time=1.0)
        self.play(FadeIn(lC2), run_time=0.5)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("a² + b²  =  c²", 36)))
        self.hold(2.2)


# ======================================================== A10 LIU HUI

class A10_LiuHuiOutIn(Board):
    """The out-in principle (chu ru xiang bu). The square on c is laid over
    the triangle; the vermilion square on the short leg is laid over the
    triangle too, the blue-green square on the long leg outside it. Three
    pieces stick out of c²: one vermilion triangle and two blue-green ones.
    Each slides, without turning, parallel to a side of c² into one of the
    three gaps inside it; everything already inside stays where it is. The
    square on c is then filled exactly by a² and b²."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        h1 = a * (b - a) / b
        O = np.zeros(2)
        A, B = np.array([0.0, a]), np.array([b, 0.0])
        Bp, Ap = np.array([b - a, -b]), np.array([-a, a - b])
        Y, H = np.array([0.0, h1 - b]), np.array([0.0, a - b])
        Csq = [A, B, Bp, Ap]
        Sa = [(0, 0), (a, 0), (a, a), (0, a)]          # over the triangle
        Sb = [(0, 0), (0, -b), (b, -b), (b, 0)]        # outside it
        Ia = [O, (a, 0), (a, h1), A]                   # stays
        Za = [A, (a, a), (a, h1)]                      # sticks out
        Ib = [O, B, Bp, Y]                             # stays
        P1 = [B, (b, -b), Bp]                          # sticks out
        P2 = [(0, -b), Bp, Y]                          # sticks out
        moves = [("P2", P2, np.array([a, b]), [(a, 0), B, (a, h1)]),
                 ("P1", P1, np.array([-b, a]), [A, Ap, H]),
                 ("Za", Za, np.array([-a, -b]), [Ap, H, Y])]
        gaps = [g for _, _, _, g in moves]

        for i in range(4):
            p, q, r = Csq[i], Csq[(i + 1) % 4], Csq[(i + 2) % 4]
            check(close(np.linalg.norm(q - p), c)
                  and abs(np.dot(q - p, r - q)) < 1e-9, "c² is a square on AB")
        check(_pip((b / 3, a / 3), Csq), "c² is laid over the triangle")
        check(_tiles_exactly([Ia, Za], Sa) and _tiles_exactly([Ib, P1, P2], Sb),
              "the sides of c² cut a² and b² into these pieces")
        for P_ in (Ia, Ib):
            check(_pip(np.mean(np.array(P_, float), axis=0), Csq),
                  "the staying pieces lie inside c²")
        for _, P_, v, G in moves:
            check(not _pip(np.mean(np.array(P_, float), axis=0), Csq),
                  "the moving pieces stick out of c²")
            check(_same_poly([np.array(p, float) + v for p in P_], G),
                  "each piece lands exactly on its gap")
            check(abs(_cross2(v, B - A)) < 1e-9 or abs(_cross2(v, Ap - A)) < 1e-9,
                  "each slide is parallel to a side of c²")
        check(_tiles_exactly([Ia, Ib] + gaps, Csq),
              "pieces that stay + the three gaps tile c² exactly")
        check(abs(a * a + b * b - abs(area(Csq))) < 1e-9, "a² + b² = c²")

        F = Frame(-a - 0.4, b + 0.5, -b - 0.45, a + 0.45)
        P, k = F.P, F.k

        tri = Polygon(P(O), P(B), P(A), stroke_color=WHITE, stroke_width=3)
        ra = _ra(P(O), P(A), P(B), 0.2)
        la = tag("a", 28).move_to(P((-0.32, a / 2)))
        lb = tag("b", 28).move_to(P((b / 2, -0.34)))
        mAB = (A + B) / 2
        lc = tag("c", 28).move_to(P(mAB + 0.34 * _u(np.array([a, b]))))
        csq = Polygon(*[P(p) for p in Csq], stroke_color=YELLOW_B,
                      stroke_width=5)
        sqA = F.poly(Sa, ZHU, 0.8)
        sqB = F.poly(Sb, QING, 0.8)
        lA2 = tag("a²", 34).move_to(P((a * 0.36, a * 0.36)))
        lB2 = tag("b²", 36).move_to(P((b * 0.4, -b * 0.42)))

        self.play(Create(tri), Create(ra), FadeIn(la), FadeIn(lb), FadeIn(lc),
                  run_time=1.1)
        self.play(Create(csq), run_time=1.2)
        self.hold(0.3)
        self.add(sqA, sqB)
        self.bring_to_back(sqA, sqB)
        sqA.set_opacity(0)
        sqB.set_opacity(0)
        self.play(sqA.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                  sqB.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                  FadeOut(la), FadeOut(lb), FadeOut(lc),
                  FadeIn(lA2), FadeIn(lB2), run_time=1.0)
        self.hold(0.6)

        # the sides of c² cut the squares: pieces (the seams lie on c²)
        mob = {n: F.poly(P_, col, 0.8) for n, P_, col in
               (("Ia", Ia, ZHU), ("Za", Za, ZHU), ("Ib", Ib, QING),
                ("P1", P1, QING), ("P2", P2, QING))}
        self.remove(sqA, sqB)
        self.add(*mob.values())
        self.bring_to_back(*mob.values())
        # out: the three pieces outside c²; in: the three gaps inside it
        outs = [mob[n] for n in ("Za", "P1", "P2")]
        self.play(*[m.animate.set_fill(opacity=1.0).set_stroke(WHITE, 4)
                    for m in outs], run_time=0.7)
        holes = [VGroup(F.poly(G, YELLOW_B, 0.2, stroke_width=0),
                        DashedVMobject(Polygon(*[P(p) for p in G],
                                               stroke_color=YELLOW_A,
                                               stroke_width=2.5),
                                       num_dashes=24))
                 for G in gaps]
        self.play(*[Create(hh) for hh in holes], FadeOut(lA2), FadeOut(lB2),
                  run_time=0.9)
        self.bring_to_front(tri, csq)
        self.hold(0.4)

        for (n, P_, v, G), hh in zip(moves, holes):
            m = mob[n]
            c0 = P(np.mean(np.array(P_, float), axis=0))
            path = DashedLine(c0, c0 + k * to3(v), color=WHITE, stroke_width=2,
                              dash_length=0.08)
            self.play(Create(path), run_time=0.4)
            self.bring_to_front(m)
            self.play(m.animate.shift(k * to3(v)), run_time=1.3)
            check(close(m.get_vertices(), [P(np.array(p, float) + v)
                                           for p in P_], 1e-6), "landed")
            self.play(FadeOut(path), FadeOut(hh),
                      m.animate.set_fill(opacity=0.8).set_stroke(WHITE, 2),
                      run_time=0.4)

        self.bring_to_front(csq)
        lC2 = _lbl("c²", 40, WHITE, 0.5).move_to(P((Csq[0] + Csq[2]) / 2))
        self.play(csq.animate.set_stroke(width=6), FadeOut(tri), FadeOut(ra),
                  FadeIn(lC2), run_time=0.8)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("a² + b²  =  c²", 36)))
        self.hold(2.2)


# ======================================================== A11 LEONARDO

class A11_LeonardoHexagons(Board):
    """The squares on the legs, the triangle, and a copy of it joining the
    squares' outer corners make a hexagon, symmetric in the line through
    those corners. The square on c, the triangle, and a copy turned half a
    turn about the square's centre make a second hexagon, symmetric under
    that half-turn. The half of the first hexagon that holds the triangle,
    turned a quarter-turn about the triangle's vertex A, lands vertex for
    vertex on half of the second — no flip needed. So the hexagons are
    equal: a² + b² + 2·½ab = c² + 2·½ab."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        C = np.zeros(2)
        A, B = np.array([b, 0.0]), np.array([0.0, a])
        E1, E2 = np.array([-a, a]), np.array([b, -b])    # outer corners
        Ga = np.array([-a, 0.0])
        Gb = np.array([0.0, -b])
        G = np.array([a + b, a + b])
        Z = G / 2                                          # centre of c²
        D1, D2 = np.array([a + b, b]), np.array([a, a + b])
        Sa = [C, B, E1, Ga]
        Sb = [C, Gb, E2, A]
        T = [C, A, B]
        T1 = [C, Ga, Gb]
        Sc = [A, D1, D2, B]
        T2 = [D1, G, D2]
        hex1 = [B, A, E2, Gb, Ga, E1]
        hex2 = [C, A, D1, G, D2, B]
        H1 = [E2, A, B, E1]                  # the half of hex1 holding T
        H1m = [E2, Gb, Ga, E1]
        H2 = [C, A, D1, G]
        H2m = [C, G, D2, B]

        def turn(p):
            return _rot(p, -PI / 2, A)

        check(_tiles_exactly([Sa, Sb, T, T1], hex1), "hex 1 = a² + b² + 2 triangles")
        check(_tiles_exactly([T, Sc, T2], hex2), "hex 2 = c² + 2 triangles")
        check(all(_right_triangle_ok(t, a, b) for t in (T, T1, T2)),
              "three copies of the triangle")
        check(all(close(np.linalg.norm(Sc[(i + 1) % 4] - Sc[i]), c)
                  for i in range(4))
              and abs(np.dot(D1 - A, B - A)) < 1e-9, "c² is a square")
        check(abs(_cross2(E2 - E1, C - E1)) < 1e-9, "the axis of hex 1 passes C")
        check(_tiles_exactly([H1, H1m], hex1) and _tiles_exactly([H2, H2m], hex2),
              "each axis halves its hexagon")
        check(_same_poly([np.array([-p[1], -p[0]]) for p in H1], H1m),
              "hex 1 is symmetric in its axis")
        check(_same_poly([2 * Z - p for p in H2], H2m),
              "hex 2 is symmetric under the half-turn about Z")
        check(all(close(turn(p), q) for p, q in zip(H1, H2)),
              "quarter-turn about A: half of hex 1 -> half of hex 2, vertex by vertex")
        check(abs(abs(area(hex1)) - (a * a + b * b + a * b)) < 1e-9
              and abs(abs(area(hex2)) - (c * c + a * b)) < 1e-9, "areas")

        # ---- screen: the figure turned to fit, the swing included
        phi = -72.5 * DEGREES
        swing = [_rot(p, -PI / 2 * t, A) for t in np.linspace(0, 1, 25)
                 for p in H1]
        pts = np.array([_rot(p, phi) for p in hex1 + hex2 + swing])
        F = Frame(pts[:, 0].min() - 0.15, pts[:, 0].max() + 0.15,
                  pts[:, 1].min() - 0.15, pts[:, 1].max() + 0.15)

        def S(p):
            return F.P(_rot(p, phi))

        def poly(Pp, col, op=FILL, **kw):
            return mk([S(p) for p in Pp], col, op, **kw)

        mT, mT1, mT2 = poly(T, BLUE_D), poly(T1, BLUE_E), poly(T2, BLUE_E)
        mSa, mSb, mSc = poly(Sa, ORANGE), poly(Sb, TEAL_D), poly(Sc, GOLD_D)
        raC = _ra(S(C), S(A), S(B), 0.18)
        dg = _u(np.array([-1.0, -1.0]))
        la2 = tag("a²", 32).move_to(S((Sa[0] + Sa[2]) / 2 + 0.27 * a * dg))
        lb2 = tag("b²", 34).move_to(S((Sb[0] + Sb[2]) / 2 + 0.27 * b * dg))
        lc2 = tag("c²", 34).move_to(S(Z + 0.27 * c * _u(np.array([1.0, -1.0]))))
        out1 = Polygon(*[S(p) for p in hex1], stroke_color=PURPLE_A,
                       stroke_width=6)
        out2 = Polygon(*[S(p) for p in hex2], stroke_color=YELLOW_B,
                       stroke_width=6)

        self.play(FadeIn(mT), Create(raC), run_time=0.8)
        self.play(FadeIn(mSa), FadeIn(mSb), FadeIn(la2), FadeIn(lb2),
                  run_time=0.9)
        join = DashedLine(S(Ga), S(Gb), color=WHITE, stroke_width=3,
                          dash_length=0.1)
        self.play(Create(join), run_time=0.6)
        self.play(FadeIn(mT1), FadeOut(join), run_time=0.6)
        self.play(Create(out1), run_time=0.9)
        self.play(FadeIn(mSc), FadeIn(lc2), run_time=0.8)
        self.play(FadeIn(mT2), run_time=0.6)
        self.play(Create(out2), run_time=0.9)
        self.hold(0.5)

        def ghost(Pp, col):
            return mk([S(p) for p in Pp], col, 0.5, stroke_color=col,
                      stroke_width=4)

        # hex 1: its two halves are mirror images in the axis
        e1 = 0.12 * _u(E1 - E2)          # the figure fills the height
        ax1 = DashedLine(S(E1 + e1), S(E2 - e1),
                         color=WHITE, stroke_width=2.5, dash_length=0.1)
        g1 = ghost(H1, PURPLE_A)
        self.play(Create(ax1), run_time=0.6)
        self.play(FadeIn(g1), run_time=0.4)
        self.play(_fold(g1, S(E1), S(E2)), run_time=1.6)
        check(_same_poly(g1.get_vertices(), [S(p) for p in H1m], 1e-6),
              "fold lands on the other half of hex 1")
        self.play(FadeOut(g1), run_time=0.4)

        # hex 2: its halves are swapped by the half-turn about Z
        e2 = 0.35 * _u(G)
        ax2 = DashedLine(S(C - e2), S(G + e2), color=WHITE,
                         stroke_width=2.5, dash_length=0.1)
        g2 = ghost(H2, YELLOW_B)
        dZ = Dot(S(Z), radius=0.06, color=WHITE)
        self.play(Create(ax2), FadeIn(dZ), run_time=0.6)
        self.play(FadeIn(g2), run_time=0.4)
        self.play(Rotate(g2, angle=PI, about_point=S(Z)), run_time=1.6)
        check(_same_poly(g2.get_vertices(), [S(p) for p in H2m], 1e-6),
              "half-turn lands on the other half of hex 2")
        self.play(FadeOut(g2), FadeOut(dZ), run_time=0.4)

        # the key move: half of hex 1, a quarter-turn about A, is half of hex 2
        g3 = ghost(H1, PURPLE_A)
        dA = Dot(S(A), radius=0.08, color=WHITE)
        t0 = _ang(S(E2) - S(A))
        arc = Arc(radius=0.75, start_angle=t0, angle=-PI / 2, arc_center=S(A),
                  color=WHITE, stroke_width=5)
        deg = tag("90°", 26).move_to(S(A) + 1.12 * to3(_dir(t0 - PI / 4)))
        self.play(FadeIn(g3), FadeIn(dA), Create(arc), FadeIn(deg), run_time=0.6)
        self.bring_to_front(dA, arc, deg)
        self.play(Rotate(g3, angle=-PI / 2, about_point=S(A)), run_time=2.4)
        check(all(close(v, S(q), 1e-6) for v, q in zip(g3.get_vertices(), H2)),
              "the turned half lands vertex for vertex")
        flash = Polygon(*[S(p) for p in H2], stroke_color=WHITE, stroke_width=7)
        self.play(Create(flash), run_time=0.6)
        self.play(FadeOut(flash), FadeOut(arc), FadeOut(deg), run_time=0.5)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("a² + b² + 2·½ab  =  c² + 2·½ab", 34)))
        self.hold(2.2)


# ======================================================== A17 SIMILAR AREAS

class A17_SimilarAreas(Board):
    """The altitude cuts the right triangle into two triangles similar to
    it, on hypotenuses a and b (the angles α and β reappear). Fold each of
    the three triangles over its hypotenuse into the square on that
    hypotenuse: being similar, each fills the same fraction k of its
    square. The two small triangles make up the large one, so
    k·a² + k·b² = k·c², and a² + b² = c²."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        A, B = np.zeros(2), np.array([c, 0.0])
        C = np.array([b * b / c, a * b / c])
        Hf = np.array([b * b / c, 0.0])                   # foot of the altitude
        na, nb = a * np.array([b, a]) / c, b * np.array([-a, b]) / c
        Sc = [A, B, B + np.array([0, -c]), A + np.array([0, -c])]
        Sb = [A, C, C + nb, A + nb]
        Sa = [C, B, B + na, C + na]
        Tb, Ta = [A, Hf, C], [Hf, B, C]
        Tbf = [A, _mirror(Hf, A, C), C]                   # folded into b²
        Taf = [_mirror(Hf, B, C), B, C]                   # folded into a²
        Cf = _mirror(C, A, B)
        Tcf_b, Tcf_a = [A, Hf, Cf], [Hf, B, Cf]           # folded into c²

        check(abs(np.dot(A - C, B - C)) < 1e-9 and close(np.linalg.norm(C - A), b)
              and close(np.linalg.norm(C - B), a), "right triangle, legs a, b")
        check(abs(np.dot(C - Hf, B - A)) < 1e-9, "CH is the altitude")
        check(_right_triangle_ok([p * c / b for p in Tb], a, b)
              and _right_triangle_ok([p * c / a for p in Ta], a, b),
              "the two parts are similar to the whole, ratios b/c and a/c")
        check(_tiles_exactly([Ta, Tb], [A, B, C]), "the two parts make the whole")
        for sq, n in ((Sa, a), (Sb, b), (Sc, c)):
            check(all(close(np.linalg.norm(sq[(i + 1) % 4] - sq[i]), n)
                      for i in range(4)), "squares on the sides")
        for tri, sq in ((Taf, Sa), (Tbf, Sb), (Tcf_a, Sc), (Tcf_b, Sc)):
            check(all(_inside_or_on(p, sq) for p in tri),
                  "each folded triangle lies in its square")
        kk = (a * b / 2) / (c * c)
        check(abs(abs(area(Taf)) / a ** 2 - kk) < 1e-12
              and abs(abs(area(Tbf)) / b ** 2 - kk) < 1e-12
              and abs((abs(area(Tcf_a)) + abs(area(Tcf_b))) / c ** 2 - kk) < 1e-12,
              "each triangle is the same fraction k of its square")

        phi = 75 * DEGREES
        allp = np.array([_rot(p, phi) for p in Sa + Sb + Sc])
        F = Frame(allp[:, 0].min() - 0.2, allp[:, 0].max() + 0.2,
                  allp[:, 1].min() - 0.2, allp[:, 1].max() + 0.2)

        def S(p):
            return F.P(_rot(p, phi))

        def poly(Pp, col, op=FILL, **kw):
            return mk([S(p) for p in Pp], col, op, **kw)

        tri = Polygon(S(A), S(B), S(C), stroke_color=WHITE, stroke_width=3)
        raC = _ra(S(C), S(A), S(B), 0.2)
        I = (a * A + b * B + c * C) / (a + b + c)
        side_l = VGroup(*[tag(s, 28).move_to(S(m) + 0.36 * _u(S(m) - S(I)))
                          for s, m in (("a", (B + C) / 2), ("b", (A + C) / 2),
                                       ("c", (A + B) / 2))])
        self.play(Create(tri), Create(raC), FadeIn(side_l), run_time=1.1)

        # the altitude: two triangles with the angles α, β of the whole
        alt = Line(S(C), S(Hf), color=WHITE, stroke_width=3)
        raH = _ra(S(Hf), S(B), S(C), 0.17)
        mTa, mTb = poly(Ta, TEAL_D), poly(Tb, BLUE_D)
        self.play(Create(alt), Create(raH), FadeOut(raC), run_time=0.8)
        self.add(mTa, mTb)
        self.bring_to_back(mTa, mTb)
        mTa.set_opacity(0)
        mTb.set_opacity(0)
        self.play(mTa.animate.set_fill(opacity=FILL).set_stroke(opacity=1),
                  mTb.animate.set_fill(opacity=FILL).set_stroke(opacity=1),
                  run_time=0.7)
        cA, cB = GREEN_B, YELLOW_B
        arcs = VGroup(angle_arc(S(A), S(B), S(C), 0.55, cA, 4),
                      angle_arc(S(C), S(B), S(Hf), 0.55, cA, 4),
                      angle_arc(S(B), S(A), S(C), 0.5, cB, 4),
                      angle_arc(S(C), S(A), S(Hf), 0.42, cB, 4))
        albl = VGroup(
            tag("α", 24, cA).move_to(S(A) + 0.82 * angle_mid_dir(S(A), S(B), S(C))),
            tag("α", 24, cA).move_to(S(C) + 0.8 * angle_mid_dir(S(C), S(B), S(Hf))),
            tag("β", 24, cB).move_to(S(B) + 0.78 * angle_mid_dir(S(B), S(A), S(C))),
            tag("β", 24, cB).move_to(S(C) + 0.68 * angle_mid_dir(S(C), S(A), S(Hf))))
        self.play(Create(arcs), FadeIn(albl), run_time=1.0)
        self.hold(0.8)

        # the squares on the three hypotenuses
        sqs = VGroup(*[mk([S(p) for p in sq], GREY_D, 0.35, stroke_color=GREY_A,
                          stroke_width=2.5) for sq in (Sa, Sb, Sc)])

        def far_lbl(sq, s, size):
            m = (sq[0] + sq[1]) / 2
            ctr = (sq[0] + sq[2]) / 2
            return tag(s, size).move_to(S(m + 1.42 * (ctr - m)))

        sq_l = VGroup(far_lbl(Sa, "a²", 30), far_lbl(Sb, "b²", 32),
                      far_lbl(Sc, "c²", 34))
        self.play(FadeOut(arcs), FadeOut(albl), FadeOut(side_l), run_time=0.5)
        self.add(sqs)
        self.bring_to_back(sqs)
        self.play(FadeIn(sqs), FadeIn(sq_l), run_time=0.9)

        # fold each triangle over its hypotenuse into its square
        fb = mTb.copy()
        self.add(fb)
        self.play(_fold(fb, S(A), S(C)), run_time=1.4)
        fa = mTa.copy()
        self.add(fa)
        self.play(_fold(fa, S(B), S(C)), run_time=1.4)
        fc = VGroup(mTb.copy(), mTa.copy(), alt.copy())
        self.add(fc)
        self.play(_fold(fc, S(A), S(B)), run_time=1.6)
        check(_same_poly(fb.get_vertices(), [S(p) for p in Tbf], 1e-6)
              and _same_poly(fa.get_vertices(), [S(p) for p in Taf], 1e-6)
              and _same_poly(fc[0].get_vertices(), [S(p) for p in Tcf_b], 1e-6)
              and _same_poly(fc[1].get_vertices(), [S(p) for p in Tcf_a], 1e-6),
              "folds land in the squares")

        # the same fraction k in each square
        def k_lbl(tri_pts, s, size=24):
            p = [np.asarray(q, float) for q in tri_pts]
            L = [np.linalg.norm(p[(i + 1) % 3] - p[(i + 2) % 3]) for i in range(3)]
            inc = sum(L[i] * p[i] for i in range(3)) / sum(L)
            r_in = 2 * abs(area(p)) / sum(L) * F.k
            t = _lbl(s, size, WHITE, 0.0).move_to(S(inc))
            return t, r_in

        ka, ra_ = k_lbl(Taf, "k·a²", 22)
        kb, rb_ = k_lbl(Tbf, "k·b²", 24)
        kc, rc_ = k_lbl([A, B, Cf], "k·c²", 26)
        for t, r in ((ka, ra_), (kb, rb_)):
            check(np.hypot(t.width, t.height) / 2 < r, "k-label fits its triangle")
        out_c = Polygon(S(A), S(B), S(Cf), stroke_color=WHITE, stroke_width=4)
        self.play(FadeIn(ka), FadeIn(kb), Create(out_c), FadeIn(kc),
                  run_time=1.0)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("k·a² + k·b²  =  k·c²   ⟹   a² + b² = c²", 32)))
        self.hold(2.2)


# ======================================================== A18 TILING

class A18_PythagoreanTiling(Board):
    """Squares of sides a and b tile the plane, and the tiling repeats under
    the two shifts (b, a) and (−a, b), each of length c. Mark the same point
    in every big square: the marks are the corners of a grid of squares of
    side c laid over the tiling. One grid square holds a piece of every kind
    exactly once; shifting each piece by a period of the tiling brings the
    pieces together into one b-square and one a-square: c² = a² + b²."""

    def construct(self):
        a, b = LEG_A, LEG_B
        c = float(np.hypot(a, b))
        u, v = np.array([b, a]), np.array([-a, b])

        def Bt(m, n):
            t = m * u + n * v
            return [t, t + (b, 0), t + (b, b), t + (0, b)]

        def St(m, n):
            t = m * u + n * v + np.array([b, 0.0])
            return [t, t + (a, 0), t + (a, a), t + (0, a)]

        o = np.array([0.4 * b, 0.4 * b])            # the marked point
        Cs = [o, o + u, o + u + v, o + v]
        mh, nh = 2, -1                               # where a² + b² gathers
        pieces = []
        for m in range(-3, 4):
            for n in range(-3, 4):
                for kind, T in (("B", Bt(m, n)), ("S", St(m, n))):
                    pc = _clip(T, Cs)
                    if len(pc) >= 3 and abs(area(pc)) > 1e-9:
                        vec = (mh - m) * u + (nh - n) * v
                        pieces.append((kind, pc, vec))
        Bp = [pc for kd, pc, _ in pieces if kd == "B"]
        Sp = [pc for kd, pc, _ in pieces if kd == "S"]
        check(close(np.linalg.norm(u), c) and abs(np.dot(u, v)) < 1e-12,
              "the periods are perpendicular, of length c")
        test = [(-1.0, -1.0), (9.0, -1.0), (9.0, 8.0), (-1.0, 8.0)]
        tl = [_clip(T, test) for m in range(-4, 5) for n in range(-4, 5)
              for T in (Bt(m, n), St(m, n))]
        check(_tiles_exactly([t for t in tl if len(t) >= 3], test),
              "a- and b-squares tile the plane")
        check(_tiles_exactly([pc for _, pc, _ in pieces], Cs),
              "the tiles cut the c-square into pieces without gap or overlap")
        check(abs(sum(abs(area(p)) for p in Bp) - b * b) < 1e-9
              and abs(sum(abs(area(p)) for p in Sp) - a * a) < 1e-9,
              "pieces of big squares total b², of small squares a²")
        check(_tiles_exactly([[q + vec for q in pc] for kd, pc, vec in pieces
                              if kd == "B"], Bt(mh, nh)),
              "the shifted big-square pieces tile one b-square")
        check(_tiles_exactly([[q + vec for q in pc] for kd, pc, vec in pieces
                              if kd == "S"], St(mh, nh)),
              "the shifted small-square pieces tile one a-square")
        check(len(Sp) == 1 and len(Bp) == 4, "one whole a-square, four b-pieces")

        k = 0.55
        ctr = (Cs[0] + Cs[2]) / 2
        shift = np.array([-3.2, 0.6]) - k * ctr

        def S(p):
            q = k * np.asarray(p, float)[:2] + shift
            return np.array([q[0], q[1], 0.0])

        X0, X1, Y0, Y1 = -SAFE_X, SAFE_X, SAFE_BOTTOM, SAFE_TOP
        win = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]
        tiles = VGroup()
        for m in range(-9, 10):
            for n in range(-9, 10):
                for col, T in ((TEAL_D, Bt(m, n)), (ORANGE, St(m, n))):
                    sc = _clip([S(p)[:2] for p in T], win)
                    if len(sc) >= 3 and abs(area(sc)) > 1e-6:
                        tiles.add(mk(sc, col, 0.75, stroke_width=1.5))
        dots, lines = VGroup(), VGroup()
        for m in range(-9, 10):
            for n in range(-9, 10):
                q = S(o + m * u + n * v)
                if X0 + 0.05 < q[0] < X1 - 0.05 and Y0 + 0.05 < q[1] < Y1 - 0.05:
                    dots.add(Dot(q, radius=0.065, color=YELLOW_B))
        for n in range(-9, 10):
            for base, d in ((o + n * v, u), (o + n * u, v)):
                seg = _clip_line(S(base)[:2], k * d, X0, X1, Y0, Y1)
                if seg is not None:
                    lines.add(Line(to3(seg[0]), to3(seg[1]), color=YELLOW_B,
                                   stroke_width=2.5))

        self.play(LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.01),
                  run_time=1.6)
        self.hold(0.4)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.03), run_time=1.0)
        self.play(Create(lines), run_time=1.6)
        self.hold(0.5)

        # one grid square: its pieces
        cs_out = Polygon(*[S(p) for p in Cs], stroke_color=YELLOW_B,
                         stroke_width=6)
        shades = [TEAL_C, TEAL_E, GREEN_D, BLUE_D]
        mobs, vecs = [], []
        ib = 0
        for kd, pc, vec in pieces:
            col = ORANGE if kd == "S" else shades[ib % 4]
            ib += kd == "B"
            mobs.append(mk([S(p) for p in pc], col, 0.95, stroke_width=2))
            vecs.append(vec)
        self.play(tiles.animate.set_fill(opacity=0.22).set_stroke(opacity=0.35),
                  lines.animate.set_stroke(opacity=0.3),
                  dots.animate.set_opacity(0.3), Create(cs_out), run_time=0.9)
        self.play(*[FadeIn(m_) for m_ in mobs], run_time=0.7)
        self.bring_to_front(cs_out)
        homeB = DashedVMobject(Polygon(*[S(p) for p in Bt(mh, nh)],
                                       stroke_color=WHITE, stroke_width=3),
                               num_dashes=40)
        homeS = DashedVMobject(Polygon(*[S(p) for p in St(mh, nh)],
                                       stroke_color=WHITE, stroke_width=3),
                               num_dashes=28)
        self.play(Create(homeB), Create(homeS), run_time=0.7)
        self.hold(0.4)

        # every piece moves by a period of the tiling
        self.play(LaggedStart(*[m_.animate.shift(k * to3(vec))
                                for m_, vec in zip(mobs, vecs)], lag_ratio=0.3),
                  run_time=3.2)
        for m_, (kd, pc, vec) in zip(mobs, pieces):
            check(close(m_.get_vertices(), [S(q + vec) for q in pc], 1e-6),
                  "piece landed")
        outB = Polygon(*[S(p) for p in Bt(mh, nh)], stroke_color=YELLOW_B,
                       stroke_width=5)
        outS = Polygon(*[S(p) for p in St(mh, nh)], stroke_color=YELLOW_B,
                       stroke_width=5)
        lB = _lbl("b²", 36, WHITE, 0.55).move_to(S((Bt(mh, nh)[0] + Bt(mh, nh)[2]) / 2))
        lS = _lbl("a²", 32, WHITE, 0.55).move_to(S((St(mh, nh)[0] + St(mh, nh)[2]) / 2))
        lC = _lbl("c²", 40, WHITE, 0.6).move_to(S(ctr))
        self.play(FadeOut(homeB), FadeOut(homeS), Create(outB), Create(outS),
                  run_time=0.7)
        self.play(FadeIn(lB), FadeIn(lS), FadeIn(lC), run_time=0.7)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("a² + b²  =  c²", 36)))
        self.hold(2.2)


# ======================================================== E16 EQUAL TANGENTS

class E16_EqualTangents(Board):
    """Reflect the figure in the line through the centre O and the outside
    point P: the circle and P are unmoved, so the tangent PA goes to a
    tangent from P — the other one, PB. The triangle OAP folds onto OBP,
    A lands on B, and PA = PB. The same holds wherever P is."""

    def construct(self):
        r = 1.85
        O = np.array([-2.7, 0.3])

        def tang(P):
            d = np.linalg.norm(P - O)
            th = float(np.arccos(r / d))
            t = _ang(P - O)
            return O + r * _dir(t + th), O + r * _dir(t - th)

        P0 = np.array([3.3, -0.35])
        path = [P0, np.array([2.1, 1.75]), np.array([4.1, -1.55]), P0]
        for P in path + [np.array([1.0, 0.9])]:
            A, B = tang(P)
            check(abs(np.dot(A - O, P - A)) < 1e-9
                  and abs(np.dot(B - O, P - B)) < 1e-9, "radius ⊥ tangent")
            check(close(_mirror(A, O, P), B), "the mirror in OP swaps A and B")
            check(abs(np.linalg.norm(P - A) - np.linalg.norm(P - B)) < 1e-9
                  and abs(np.linalg.norm(P - A) ** 2
                          - (np.linalg.norm(P - O) ** 2 - r * r)) < 1e-9,
                  "PA = PB")

        Pt = [P0.copy()]                       # the live outside point

        def parts():
            P = Pt[0]
            A, B = tang(P)
            uOP = _u(P - O)
            d = {}
            d["axis"] = DashedLine(to3(O - 1.2 * uOP), to3(P), color=GREY_A,
                                   stroke_width=2.5, dash_length=0.1)
            d["pa"] = Line(to3(P), to3(A + 0.5 * _u(A - P)), color=BLUE_B,
                           stroke_width=4)
            d["pb"] = Line(to3(P), to3(B + 0.5 * _u(B - P)), color=BLUE_B,
                           stroke_width=4)
            d["oa"] = Line(to3(O), to3(A), color=GREY_A, stroke_width=2.5)
            d["ob"] = Line(to3(O), to3(B), color=GREY_A, stroke_width=2.5)
            d["rA"] = _ra(A, O, P, 0.2, GREY_A, 2)
            d["rB"] = _ra(B, O, P, 0.2, GREY_A, 2)
            d["tk"] = VGroup(_ticks(P, A, 2, YELLOW_B), _ticks(P, B, 2, YELLOW_B))
            d["dA"] = Dot(to3(A), radius=0.06)
            d["dB"] = Dot(to3(B), radius=0.06)
            d["dP"] = Dot(to3(P), radius=0.07, color=YELLOW_B)
            d["lA"] = tag("A", 28).move_to(to3(A + 0.36 * _u(A - O)))
            d["lB"] = tag("B", 28).move_to(to3(B + 0.36 * _u(B - O)))
            d["lP"] = tag("P", 28, YELLOW_B).move_to(to3(P + 0.38 * uOP))
            d["lO"] = tag("O", 28).move_to(
                to3(O + 0.4 * _free_dir(O, [A, B, P, O - uOP])))
            return d

        circ = Circle(radius=r, color=GREY_B, stroke_width=3).move_to(to3(O))
        dO = Dot(to3(O), radius=0.06)
        d = parts()
        self.play(Create(circ), FadeIn(dO), FadeIn(d["lO"]), run_time=0.9)
        self.play(FadeIn(d["dP"]), FadeIn(d["lP"]), run_time=0.4)
        self.play(Create(d["pa"]), Create(d["pb"]), run_time=1.0)
        self.play(FadeIn(d["dA"]), FadeIn(d["dB"]), FadeIn(d["lA"]),
                  FadeIn(d["lB"]), run_time=0.5)
        self.play(Create(d["axis"]), run_time=0.7)
        self.play(Create(d["oa"]), Create(d["ob"]), Create(d["rA"]),
                  Create(d["rB"]), run_time=0.8)
        self.hold(0.4)

        # fold the upper half of the figure over the line OP
        A0, B0 = tang(P0)
        flap = VGroup(mk([to3(O), to3(A0), to3(P0)], BLUE_D, 0.55,
                         stroke_color=BLUE_B, stroke_width=3))
        self.play(FadeIn(flap), run_time=0.4)
        self.play(_fold(flap, O, P0), run_time=1.8)
        check(_same_poly(flap[0].get_vertices(), [to3(O), to3(B0), to3(P0)], 1e-6),
              "the fold carries A onto B")
        self.play(FadeIn(d["tk"]), run_time=0.5)
        self.play(FadeOut(flap), run_time=0.4)

        # ... wherever P is
        keys = ["axis", "pa", "pb", "oa", "ob", "rA", "rB", "tk", "dA", "dB",
                "dP", "lA", "lB", "lP", "lO"]
        self.remove(*[d[k_] for k_ in keys])
        live = always_redraw(lambda: VGroup(*parts().values()))
        self.add(live)
        for p_from, p_to in zip(path[:-1], path[1:]):
            tr = ValueTracker(0.0)

            def upd(m, p_from=p_from, p_to=p_to, tr=tr):
                Pt[0] = p_from + tr.get_value() * (p_to - p_from)

            live.add_updater(upd)
            self.play(tr.animate.set_value(1.0), run_time=1.3)
            live.remove_updater(upd)
        Pt[0] = P0.copy()
        live.clear_updaters()
        final = VGroup(*parts().values())
        self.remove(live)
        self.add(final)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("PA  =  PB", 38)))
        self.hold(2.2)


# ======================================================== E17 PARABOLA

class E17_ParabolaReflection(Board):
    """The parabola is the set of points as far from the focus F as from the
    directrix. For P on it, PF = PD (D the foot on the directrix), so the
    perpendicular bisector ℓ of FD passes through P and halves ∠FPD. Any
    other point Q of ℓ has QF = QD > its distance QQ′ to the directrix, so
    it lies outside: ℓ touches only at P — it is the tangent (its slope is
    the derivative x/2p). A ray coming down parallel to the axis meets ℓ at
    the angle ∠ℓPD has (vertical angles), so it leaves along PF."""

    def construct(self):
        p, x0 = 1.0, 2.2

        def f(x):
            return x * x / (4 * p)

        Fm, Pm, Dm = np.array([0.0, p]), np.array([x0, f(x0)]), np.array([x0, -p])
        Mm = (Fm + Dm) / 2
        t = _u(np.array([1.0, x0 / (2 * p)]))          # slope f'(x0)
        check(abs(np.linalg.norm(Pm - Fm) - np.linalg.norm(Pm - Dm)) < 1e-12,
              "PF = PD")
        check(abs(np.dot(Dm - Fm, t)) < 1e-12 and abs(_cross2(Pm - Mm, t)) < 1e-12,
              "the tangent (slope f′) is the perpendicular bisector of FD")
        check(abs(Mm[1]) < 1e-12, "it crosses the vertex tangent at FD's midpoint")
        for s in np.linspace(-3.0, 3.0, 61):
            if abs(s) > 1e-6:
                Q = Pm + s * t
                check(np.linalg.norm(Q - Fm) > Q[1] + p + 1e-12
                      and Q[1] < f(Q[0]), "every other point of ℓ is outside")

        def ang(u1, u2):
            return float(np.arccos(np.clip(np.dot(_u(u1), _u(u2)), -1, 1)))

        th = ang(Fm - Pm, -t)
        check(abs(ang(Dm - Pm, -t) - th) < 1e-12
              and abs(ang(np.array([0, 1.0]), t) - th) < 1e-12,
              "angle in = angle out")
        d_in = np.array([0.0, -1.0])
        d_out = 2 * np.dot(d_in, t) * t - d_in
        check(close(d_out, _u(Fm - Pm)), "the reflected ray heads for F")

        Fr = Frame(-3.75, 3.95, -1.6, 3.5)
        S = Fr.P
        xs = np.linspace(-3.55, 3.55, 141)
        curve = _curve([S((x, f(x))) for x in xs], WHITE, 4)
        direc = Line(S((-3.75, -p)), S((3.95, -p)), color=GREY_B, stroke_width=3)
        axis = DashedLine(S((0, -1.45)), S((0, 3.4)), color=GREY_D,
                          stroke_width=2, dash_length=0.1)
        ray_x = (-3.05, -2.0, -1.0, 0.55, 1.25)
        hits = [np.array([xr, f(xr)]) for xr in ray_x]
        dF = Dot(S(Fm), radius=0.07, color=YELLOW_B)
        # the label of F sits in the widest gap among everything meeting F
        lF = tag("F", 28, YELLOW_B).move_to(S(Fm) + 0.4 * to3(_free_dir(
            Fm, hits + [Pm, Fm + (0, 1), Fm - (0, 1)])))
        ld = tag("d", 28, GREY_B).move_to(S((-3.45, -p - 0.32)))
        dP = Dot(S(Pm), radius=0.07)
        nP = _u(np.array([x0 / (2 * p), -1.0]))           # outward normal
        lP = tag("P", 28).move_to(S(Pm + 0.33 * nP))
        dD = Dot(S(Dm), radius=0.06)
        lD = tag("D", 28).move_to(S(Dm) + DOWN * 0.36)
        PF = Line(S(Pm), S(Fm), color=BLUE_B, stroke_width=4)
        PD = Line(S(Pm), S(Dm), color=BLUE_B, stroke_width=4)
        rD = _ra(S(Dm), S(Pm), S((x0 + 1, -p)), 0.18, GREY_A, 2)
        tks = VGroup(_ticks(S(Pm), S(Fm), 2, YELLOW_B),
                     _ticks(S(Pm), S(Dm), 2, YELLOW_B))

        self.play(Create(direc), Create(axis), FadeIn(dF), FadeIn(lF),
                  FadeIn(ld), run_time=0.9)
        self.play(Create(curve), run_time=1.4)
        self.play(FadeIn(dP), FadeIn(lP), run_time=0.4)
        self.play(Create(PD), FadeIn(dD), FadeIn(lD), Create(rD), run_time=0.7)
        self.play(Create(PF), run_time=0.6)
        self.play(FadeIn(tks), run_time=0.4)

        FD = DashedLine(S(Fm), S(Dm), color=GREY_A, stroke_width=2.5,
                        dash_length=0.09)
        dM = Dot(S(Mm), radius=0.055, color=YELLOW_B)
        rM = _ra(S(Mm), S(Dm), S(Mm + t), 0.15, YELLOW_B, 2)
        ell = Line(S(Pm - 2.9 * t), S(Pm + 1.75 * t), color=YELLOW_B,
                   stroke_width=4)
        lell = tag("ℓ", 30, YELLOW_B).move_to(S(Pm + 1.75 * t) + 0.3 * to3(_u(
            np.array([-t[1], t[0]]))))
        self.play(Create(FD), FadeIn(dM), run_time=0.7)
        self.play(Create(ell), Create(rM), FadeIn(lell), run_time=1.0)
        a1 = angle_arc(S(Pm), S(Fm), S(Pm - t), 0.62, GREEN_B, 4)
        a2 = angle_arc(S(Pm), S(Pm - t), S(Dm), 0.62, GREEN_B, 4)
        l1 = tag("θ", 24, GREEN_B).move_to(
            S(Pm) + 0.92 * angle_mid_dir(S(Pm), S(Fm), S(Pm - t)))
        l2 = tag("θ", 24, GREEN_B).move_to(
            S(Pm) + 0.92 * angle_mid_dir(S(Pm), S(Pm - t), S(Dm)))
        self.play(Create(a1), Create(a2), FadeIn(l1), FadeIn(l2), run_time=0.8)
        self.hold(0.4)

        # ℓ touches only at P: any other Q on it has QF = QD > QQ′
        sQ = ValueTracker(-2.1)

        def qparts():
            Q = Pm + sQ.get_value() * t
            Qp = np.array([Q[0], -p])
            g = VGroup(Line(S(Q), S(Fm), color=ORANGE, stroke_width=2.5),
                       Line(S(Q), S(Dm), color=ORANGE, stroke_width=2.5),
                       DashedLine(S(Q), S(Qp), color=ORANGE, stroke_width=2.5,
                                  dash_length=0.07))
            if abs(Q[0] - Dm[0]) > 0.15:
                g.add(_ra(S(Qp), S(Q), S(Dm), 0.14, ORANGE, 2))
            g.add(_ticks(S(Q), S(Fm), 1, ORANGE), _ticks(S(Q), S(Dm), 1, ORANGE),
                  Dot(S(Q), radius=0.07, color=ORANGE))
            return g

        qg = always_redraw(qparts)
        def q_label():
            Q = Pm + sQ.get_value() * t
            fd = _free_dir(Q, [Fm, Dm, np.array([Q[0], -p]), Q + t, Q - t])
            return tag("Q", 26, ORANGE).move_to(S(Q) + 0.36 * to3(fd))

        lQ = always_redraw(q_label)
        self.play(FadeIn(qg), FadeIn(lQ), run_time=0.5)
        self.play(sQ.animate.set_value(1.35), run_time=2.6)
        self.play(sQ.animate.set_value(-1.0), run_time=1.2)
        qg.clear_updaters()
        lQ.clear_updaters()
        self.play(FadeOut(qg), FadeOut(lQ), run_time=0.4)

        # a ray down the axis direction leaves through F
        top = np.array([x0, 3.35])
        rin = Arrow(S(top), S(Pm), buff=0, color=YELLOW_B, stroke_width=5,
                    max_tip_length_to_length_ratio=0.12)
        rout = Arrow(S(Pm), S(Fm), buff=0.08, color=YELLOW_B, stroke_width=5,
                     max_tip_length_to_length_ratio=0.08)
        a3 = angle_arc(S(Pm), S(top), S(Pm + t), 0.62, GREEN_B, 4)
        l3 = tag("θ", 24, GREEN_B).move_to(
            S(Pm) + 0.92 * angle_mid_dir(S(Pm), S(top), S(Pm + t)))
        self.play(GrowArrow(rin), run_time=0.8)
        self.play(Create(a3), FadeIn(l3), run_time=0.5)
        self.play(GrowArrow(rout), run_time=0.8)
        rays = VGroup()
        for h_ in hits:
            rays.add(_curve([S((h_[0], 3.35)), S(h_), S(Fm)], YELLOW_A, 2))
        self.play(LaggedStart(*[Create(rr) for rr in rays], lag_ratio=0.2),
                  run_time=1.8)
        self.bring_to_front(dF)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("PF = PD   ⟹   every ray parallel to the axis "
                                "reflects through F", 30)))
        self.hold(2.2)


# ======================================================== E24 PI, HEXAGONS

class E24_PiHexagons(Board):
    """Six radii cut the inscribed hexagon into equilateral triangles: its
    side is r. The circumscribed hexagon touches the circle at the inner
    hexagon's corners; half its side is r·tan 30°, so its side is 2r/√3.
    Over each sixth of the circle the chord, the arc and the two tangent
    halves share their ends and are nested, the arc convex between them:
    r < πr/3 < 2r/√3. Six times over: 6r < 2πr < 4√3·r."""

    def construct(self):
        R = 2.3
        O = np.array([-3.45, 0.25])
        inner = [O + R * _dir(k * PI / 3) for k in range(6)]
        Ro = R / np.cos(PI / 6)
        outer = [O + Ro * _dir(PI / 6 + k * PI / 3) for k in range(6)]
        T1, T2, W = inner[1], inner[2], outer[1]          # the top sixth
        L1, L2, L3 = R, PI * R / 3, 2 * R / np.sqrt(3)
        check(all(close(np.linalg.norm(inner[k] - inner[(k + 1) % 6]), R)
                  for k in range(6)), "inner side = r")
        for k in range(6):
            e0, e1 = outer[k], outer[(k + 1) % 6]
            m = (e0 + e1) / 2
            check(close(np.linalg.norm(e1 - e0), L3)
                  and close(m, inner[(k + 1) % 6])
                  and abs(np.dot(e1 - e0, m - O)) < 1e-9,
                  "outer side 2r/√3, tangent at an inner corner")
        check(close(np.linalg.norm(W - T1) + np.linalg.norm(W - T2), L3),
              "the tangent path over one sixth is 2r/√3")
        arcp = _arc(O, R, PI / 3, 2 * PI / 3, 80)
        tri = [T1, W, T2]
        check(all(_inside_or_on(q, tri, 1e-9) for q in arcp),
              "the arc lies between its chord and the tangent path")
        check(L1 < L2 < L3, "r < πr/3 < 2r/√3")
        check(abs(6 * L1 - 6 * R) < 1e-12 and abs(6 * L3 - 4 * np.sqrt(3) * R) < 1e-9,
              "perimeters 6r and 4√3·r")
        check(SAFE_BOTTOM < O[1] - Ro and O[1] + Ro < SAFE_TOP
              and O[0] - Ro > -SAFE_X, "figure inside the frame")

        circ = Circle(radius=R, color=WHITE, stroke_width=3).move_to(to3(O))
        dO = Dot(to3(O), radius=0.06)
        hin = Polygon(*[to3(p) for p in inner], stroke_color=BLUE_B,
                      stroke_width=3)
        hout = Polygon(*[to3(p) for p in outer], stroke_color=RED_B,
                       stroke_width=3)
        spokes = VGroup(*[Line(to3(O), to3(p), color=GREY_B, stroke_width=1.8)
                          for p in inner])
        eq = VGroup(_ticks(O, T1, 1, BLUE_B), _ticks(O, T2, 1, BLUE_B),
                    _ticks(T1, T2, 1, BLUE_B))
        lr = tag("r", 26, BLUE_B).move_to(to3((O + inner[0]) / 2 + np.array([0, -0.28])))

        self.play(Create(circ), FadeIn(dO), run_time=1.0)
        self.play(Create(spokes), Create(hin), FadeIn(lr), run_time=1.2)
        self.play(FadeIn(eq), run_time=0.5)
        self.play(Create(hout), run_time=1.1)
        # the top sixth: radius ⊥ tangent at T1, 30° at the centre
        rad_w = DashedLine(to3(O), to3(W), color=GREY_B, stroke_width=2,
                           dash_length=0.08)
        rT1 = _ra(T1, O, W, 0.16, GREY_A, 2)
        a30 = angle_arc(to3(O), to3(T1), to3(W), 0.6, GREY_A, 3)
        l30 = tag("30°", 20, GREY_A).move_to(
            to3(O) + 1.5 * angle_mid_dir(to3(O), to3(T1), to3(W)))
        gap30 = 2 * 1.5 * np.sin(PI / 12) - l30.width
        check(gap30 > 0.2, "30° label clear of both radii")
        self.play(Create(rad_w), Create(rT1), Create(a30), FadeIn(l30),
                  run_time=0.8)
        self.hold(0.4)

        chord = Line(to3(T2), to3(T1), color=BLUE_B, stroke_width=7)
        arc = _curve(arcp[::-1], YELLOW_B, 7)                  # T2 -> T1
        roofL = Line(to3(T2), to3(W), color=RED_B, stroke_width=7)
        roofR = Line(to3(W), to3(T1), color=RED_B, stroke_width=7)
        self.play(Create(chord), Create(arc), Create(roofL), Create(roofR),
                  run_time=1.0)
        self.hold(0.4)

        # straighten each onto a ruler, all starting at x0
        x0, ys = 0.35, (2.35, 1.6, 0.85)
        c_ch, c_arc = chord.copy(), arc.copy()
        rf = VGroup(roofL.copy(), roofR.copy())
        self.add(c_ch, c_arc, rf)
        self.play(c_ch.animate.shift(np.array([x0, ys[0], 0]) - to3(T2)),
                  run_time=1.0)

        top = O + np.array([0.0, R])                           # arc midpoint
        mid2 = np.array([x0 + L2 / 2, ys[1]])
        self.play(c_arc.animate.shift(to3(mid2) - to3(top)), run_time=1.0)

        def bent(kappa):
            ss = np.linspace(-L2 / 2, L2 / 2, 80)
            if kappa < 1e-9:
                pts = [mid2 + np.array([s_, 0.0]) for s_ in ss]
            else:
                pts = [mid2 + np.array([np.sin(kappa * s_) / kappa,
                                        -(1 - np.cos(kappa * s_)) / kappa])
                       for s_ in ss]
            return _curve(pts, YELLOW_B, 7)

        check(close(bent(1 / R).get_start(), c_arc.get_start(), 1e-6)
              and close(bent(1 / R).get_end(), c_arc.get_end(), 1e-6),
              "the unbending starts from the arc itself")
        self.play(UpdateFromAlphaFunc(c_arc, lambda m_, al: m_.become(
            bent((1 - al) / R))), run_time=1.6)

        apex = np.array([x0 + L3 / 2, ys[2]])
        self.play(rf.animate.shift(to3(apex) - to3(W)), run_time=1.0)
        self.play(Rotate(rf[0], angle=-PI / 6, about_point=to3(apex)),
                  Rotate(rf[1], angle=PI / 6, about_point=to3(apex)),
                  run_time=1.0)
        check(close(rf[0].get_start(), to3((x0, ys[2])), 1e-6)
              and close(rf[1].get_end(), to3((x0 + L3, ys[2])), 1e-6),
              "the tangent path lies flat, length 2r/√3")
        check(close(c_arc.get_start(), to3((x0, ys[1])), 1e-6)
              and close(c_arc.get_end(), to3((x0 + L2, ys[1])), 1e-6),
              "the arc lies flat, length πr/3")

        # all three lengths magnified together (one factor: order kept)
        mag = 1.6
        bars = VGroup(c_ch, c_arc, rf)
        self.play(bars.animate.scale(mag, about_point=to3((x0, ys[1]))),
                  run_time=1.0)
        yb = [ys[1] + mag * (y_ - ys[1]) for y_ in ys]
        for bar_, Ln, y_ in ((c_ch, L1, yb[0]), (c_arc, L2, yb[1])):
            check(close(bar_.get_start(), to3((x0, y_)), 1e-6)
                  and close(bar_.get_end(), to3((x0 + mag * Ln, y_)), 1e-6),
                  "magnified bar in place")
        check(close(rf[1].get_end(), to3((x0 + mag * L3, yb[2])), 1e-6),
              "magnified tangent path in place")
        g = VGroup(*[DashedLine(to3((xx, yb[0] + 0.3)), to3((xx, yb[2] - 0.3)),
                                color=GREY_B, stroke_width=1.8, dash_length=0.07)
                     for xx in (x0, x0 + mag * L1, x0 + mag * L2)])
        xl = x0 + mag * L3 + 0.3
        labs = VGroup(tag("r", 28, BLUE_B).move_to(to3((xl, yb[0])), aligned_edge=LEFT),
                      tag("πr/3", 28, YELLOW_B).move_to(to3((xl, yb[1])), aligned_edge=LEFT),
                      tag("2r/√3", 28, RED_B).move_to(to3((xl, yb[2])), aligned_edge=LEFT))
        check(yb[0] + 0.3 < SAFE_TOP, "rulers below the top edge")
        check(labs.get_right()[0] < SAFE_X, "ruler labels inside the frame")
        self.bring_to_back(g)
        self.play(Create(g), FadeIn(labs), run_time=0.8)
        ineq1 = tag("r  <  πr/3  <  2r/√3", 32).move_to(to3((3.2, -0.55)))
        ineq6 = tag("6r  <  2πr  <  4√3·r", 32).move_to(to3((3.2, -1.55)))
        times6 = tag("×6", 26, GREY_A).move_to(to3((3.2, -1.05)))
        check(ineq1.get_right()[0] < SAFE_X and ineq6.get_right()[0] < SAFE_X,
              "inequalities inside the frame")
        self.play(FadeIn(ineq1), run_time=0.7)
        self.play(FadeIn(times6), FadeIn(ineq6), run_time=0.8)
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("3  <  π  <  2√3", 38)))
        self.hold(2.2)
