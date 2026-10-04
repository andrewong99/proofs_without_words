# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes_f.py — triangle & polygon proofs without words (category F):
F2 exterior angle, F3 polygon angle sum, F4 exterior angles, F5 triangle
area by shear, F6 centroid, F8 Varignon, F9 Napoleon, F10 Ceva, F11 Pick.

Every move of a piece is rigid (shift / Rotate) unless it is a shear, and
every claimed landing is checked numerically with `check` before anything
is drawn, so a wrong construction cannot render quietly.
"""

from pww_kit import *


# ------------------------------------------------------------ helpers

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


def _wedge_dir(vertex, a1, span, r, color, op=FILL, stroke=1.5,
               stroke_color=WHITE):
    """Sector at vertex from direction angle a1, counter-clockwise by span."""
    s = Sector(radius=r, start_angle=a1, angle=span, arc_center=to3(vertex),
               color=color, fill_opacity=op)
    s.set_stroke(stroke_color, stroke)
    return s


def _dir(p, q):
    """Direction angle of the ray p -> q."""
    d = to3(q) - to3(p)
    return float(np.arctan2(d[1], d[0]))


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _angle(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


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
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7,
                                          tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


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


def _right_mark(foot, d1, d2, size=0.2, color=GREY_A, width=2):
    """Right-angle square at `foot` between unit directions d1 and d2."""
    f, a, b = to3(foot), _unit(d1) * size, _unit(d2) * size
    return VMobject(stroke_color=color, stroke_width=width) \
        .set_points_as_corners([f + a, f + a + b, f + b])


def _glide(mob, pivot, target, angle, **kw):
    """A rigid motion that stays rigid on every frame: the piece turns by
    `angle` about its pivot while the pivot travels straight to `target`.
    (A Transform would interpolate vertex by vertex and squash it.)"""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _convex_ccw(pts):
    n = len(pts)
    return all(_cross(to3(pts[(i + 1) % n]) - to3(pts[i]),
                      to3(pts[(i + 2) % n]) - to3(pts[(i + 1) % n])) > 1e-9
               for i in range(n))


# ======================================================== F2 EXTERIOR ANGLE

class F2_ExteriorAngle(Board):
    """Through C draw the parallel to BA. The angle at B slides
    along the base line to C (corresponding angles: a translation); the
    angle at A turns half a turn about the midpoint of AC (alternate
    angles: a point reflection). Together they fill the exterior angle."""

    def construct(self):
        A = np.array([-3.72, 3.18, 0.0])
        B = np.array([-5.7, -2.1, 0.0])
        C = np.array([0.9, -2.1, 0.0])
        D = C + 4.7 * _unit(C - B)                 # BC produced beyond C
        u = _unit(A - B)
        E = C + 3.9 * u                            # parallel to BA through C
        M = (A + C) / 2
        r = 1.05

        al, be, ga = _angle(A, B, C), _angle(B, A, C), _angle(C, A, B)
        check(abs(al + be + ga - PI) < 1e-9, "triangle angle sum")
        check(abs(_angle(C, A, D) - (al + be)) < 1e-9,
              "exterior angle = alpha + beta")
        check(abs(_cross(E - C, A - B)) < 1e-9, "CE parallel to BA")
        check(close(2 * M - A, C), "half-turn about M sends A to C")
        check(abs(_cross(E - C, (2 * M - B) - C)) < 1e-9,
              "half-turn sends line AB onto line CE")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        ext_line = Line(C, D, color=WHITE, stroke_width=4)
        w_a = _wedge(A, B, C, r, BLUE_D)
        w_b = _wedge(B, C, A, r, TEAL_D)
        w_c = _wedge(C, A, B, 0.75, ORANGE, op=0.7)
        ext = _wedge(C, A, D, r, YELLOW_E, op=0.0, stroke=4,
                     stroke_color=YELLOW_B)

        def lab(s, v, p, q, k, col):
            return tag(s, 30, col).move_to(v + k * angle_mid_dir(v, p, q))

        l_a = lab("α", A, B, C, 1.42, BLUE_B)
        l_b = lab("β", B, A, C, 1.40, TEAL_B)
        l_c = lab("γ", C, A, B, 1.10, ORANGE)
        dots = VGroup(*[Dot(P, radius=0.05) for P in (A, B, C)])

        self.play(Create(tri), run_time=1.0)
        self.play(Create(ext_line), FadeIn(dots), run_time=0.6)
        self.play(FadeIn(w_a), FadeIn(w_b), FadeIn(w_c),
                  FadeIn(l_a), FadeIn(l_b), FadeIn(l_c), run_time=0.9)
        self.play(Create(ext), run_time=0.9)
        self.hold(0.5)

        # the parallel to BA through C
        ab_bold = Line(B, A, color=YELLOW_B, stroke_width=7)
        par = guide(C - 0.65 * u, E, YELLOW_B, width=3)   # clear of caption
        ch1 = _chevron(B, A)
        ch2 = _chevron(C, E)
        self.play(Create(ab_bold), Create(par), FadeIn(ch1), FadeIn(ch2),
                  run_time=1.0)
        self.hold(0.4)

        # beta: translate along the base line, B -> C
        rail = Line(B, D, color=TEAL_B, stroke_width=7, stroke_opacity=0.6)
        cb = w_b.copy()
        self.add(cb)
        self.play(FadeIn(rail), run_time=0.4)
        self.play(cb.animate.shift(C - B), run_time=1.6)
        self.play(FadeOut(rail), FadeIn(tag("β", 30, TEAL_B).move_to(
            C + 1.40 * angle_mid_dir(C, D, E))), run_time=0.5)

        # alpha: half-turn about the midpoint of AC
        mdot = Dot(M, radius=0.07, color=BLUE_B)
        tk = VGroup(_ticks(A, M, color=BLUE_B), _ticks(M, C, color=BLUE_B))
        ca = w_a.copy()
        self.add(ca)
        self.play(FadeIn(mdot), FadeIn(tk), run_time=0.5)
        self.play(Rotate(ca, angle=PI, about_point=M), run_time=2.0)
        self.play(FadeIn(tag("α", 30, BLUE_B).move_to(
            C + 1.42 * angle_mid_dir(C, E, A))), run_time=0.5)
        self.play(FadeOut(mdot), FadeOut(tk), run_time=0.4)

        self.bring_to_front(ext)
        self.play(Indicate(ext, color=YELLOW, scale_factor=1.0), run_time=0.8)
        self.play(Write(caption("exterior angle  =  α + β", 34)))
        self.hold(2.2)


# ===================================================== F3 POLYGON ANGLE SUM

class F3_PolygonAngleSum(Board):
    """Diagonals from one vertex fan a convex n-gon into n − 2 triangles.
    Their corners exactly fill the polygon's corners; each triangle's three
    corners close into a straight angle (F1). So S = (n − 2)·180°."""

    def construct(self):
        off = np.array([-2.55, 0.32, 0.0])
        raw = [(-3.0, -0.9), (-1.6, -2.6), (1.1, -2.75), (2.9, -1.05),
               (2.8, 1.2), (0.9, 2.85), (-2.2, 2.05)]
        V = [to3(p) + off for p in raw]
        n = len(V)
        r = 0.62
        check(_convex_ccw(V), "heptagon convex, counter-clockwise")
        interior = [_angle(V[i], V[i - 1], V[(i + 1) % n]) for i in range(n)]
        check(abs(sum(interior) - (n - 2) * PI) < 1e-9, "angle sum")
        check(min(np.linalg.norm(V[(i + 1) % n] - V[i]) for i in range(n))
              > 2 * r, "wedges at the two ends of a side never meet")

        cols = PALETTE[:n - 2]
        poly = Polygon(*V, stroke_color=WHITE, stroke_width=4)
        dots = VGroup(*[Dot(p, radius=0.05) for p in V])
        arcs = VGroup(*[angle_arc(V[i], V[i - 1], V[(i + 1) % n], radius=r,
                                  color=YELLOW_B, width=4) for i in range(n)])

        # fan triangles (V0, Vk, Vk+1), counter-clockwise like the polygon
        tris, wedges = [], []
        per_vertex = np.zeros(n)
        for k in range(1, n - 1):
            idx = (0, k, k + 1)
            P = [V[j] for j in idx]
            col = cols[k - 1]
            tris.append(mk(P, col, op=0.32, stroke_width=0))
            group = []
            for m in range(3):
                v, nxt, prv = P[m], P[(m + 1) % 3], P[(m + 2) % 3]
                a1 = _dir(v, nxt)                  # interior: ccw from v->nxt
                span = _angle(v, nxt, prv)
                check(abs(((_dir(v, prv) - a1) % TAU) - span) < 1e-9,
                      "corner runs counter-clockwise")
                group.append((_wedge_dir(v, a1, span, r, col, op=0.92),
                              v, a1, span))
                per_vertex[idx[m]] += span
            check(abs(sum(g[3] for g in group) - PI) < 1e-9,
                  "triangle corners sum to a straight angle")
            wedges.append(group)
        check(np.allclose(per_vertex, interior, atol=1e-9),
              "triangle corners fill every polygon corner exactly")

        diags = VGroup(*[Line(V[0], V[k], color=WHITE, stroke_width=2.5)
                         for k in range(2, n - 1)])

        n_tag = tag(f"n = {n}", 28).move_to(V[0] + np.array([-0.3, -2.0, 0]))
        self.play(Create(poly), FadeIn(dots), FadeIn(n_tag), run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.12),
                  run_time=1.2)
        self.hold(0.4)
        self.play(LaggedStart(*[Create(d) for d in diags], lag_ratio=0.25),
                  *[FadeIn(t) for t in tris], run_time=1.4)
        self.play(*[FadeIn(g[0]) for grp in wedges for g in grp],
                  run_time=1.0)
        self.bring_to_front(arcs)
        self.hold(0.6)

        # each triangle's three corners glide (rigidly) into one half-disc
        hx, ys = 2.45, [2.72, 1.52, 0.32, -0.88, -2.08]
        moves = []
        for k, grp in enumerate(wedges):
            H = np.array([hx, ys[k], 0.0])
            start = 0.0
            anims = []
            for w, v, a1, span in grp:
                anims.append(_glide(w, v, H, start - a1))
                start += span
            moves.append(AnimationGroup(*anims))
        self.play(LaggedStart(*moves, lag_ratio=0.3), run_time=3.6)
        self.hold(0.3)

        base = VGroup(*[Line([hx - r - 0.06, y, 0], [hx + r + 0.06, y, 0],
                             color=WHITE, stroke_width=3) for y in ys])
        degs = VGroup(*[tag("180°", 28).move_to([hx + 1.35, y + r / 2, 0])
                        for y in ys])
        self.play(Create(base), LaggedStart(*[FadeIn(d) for d in degs],
                                            lag_ratio=0.15), run_time=1.2)
        bx = hx + 2.25
        bracket = VMobject(stroke_color=YELLOW_B, stroke_width=3)
        bracket.set_points_as_corners([[bx - 0.15, ys[0] + r, 0],
                                       [bx, ys[0] + r, 0],
                                       [bx, ys[-1], 0],
                                       [bx - 0.15, ys[-1], 0]])
        self.play(Create(bracket),
                  FadeIn(tag("n − 2", 30, YELLOW_B).next_to(
                      bracket, RIGHT, buff=0.25)), run_time=0.8)
        self.play(Write(caption("S  =  (n − 2) · 180°", 34)))
        self.hold(2.2)


# ================================================= F4 EXTERIOR ANGLES 360°

class F4_ExteriorAngles(Board):
    """Shrink the polygon towards a point, keeping every side's direction
    (a homothety). Each exterior angle only translates with its vertex, so
    in the limit they all sit at one point, edge to edge: a full turn."""

    def construct(self):
        k, off = 1.08, np.array([-0.55, 0.25, 0.0])
        raw = [(-2.6, -1.2), (1.8, -1.9), (2.4, 0.2), (0.2, 2.3), (-1.6, 1.6)]
        V = [to3(p) * k + off for p in raw]
        n = len(V)
        O = sum(V) / n                              # centre of the shrink
        rho, L = 0.95, 1.35
        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D, RED_D]
        check(_convex_ccw(V), "pentagon convex, counter-clockwise")

        d = [_unit(V[(i + 1) % n] - V[i]) for i in range(n)]      # side i
        a_in = [_dir(ORIGIN, d[i - 1]) for i in range(n)]   # incoming dir
        turn = [(_dir(ORIGIN, d[i]) - a_in[i]) % TAU for i in range(n)]
        check(all(0 < x < PI for x in turn), "every exterior angle < 180°")
        check(abs(sum(turn) - TAU) < 1e-9, "exterior angles sum to 360°")
        # at the limit the wedges must sit edge to edge around O
        order = sorted(range(n), key=lambda i: a_in[i] % TAU)
        for j in range(n):
            i, nx = order[j], order[(j + 1) % n]
            check(abs((a_in[i] + turn[i] - a_in[nx]) % TAU) < 1e-9
                  or abs((a_in[i] + turn[i] - a_in[nx]) % TAU - TAU) < 1e-9,
                  "wedges meet edge to edge at the centre")

        t = ValueTracker(1.0)

        def Vt(i):
            return O + t.get_value() * (V[i] - O)

        # ghost of the full-size figure (stays, dimmed)
        ghost = VGroup(
            Polygon(*V, stroke_color=GREY_B, stroke_width=2,
                    stroke_opacity=0.55),
            *[DashedLine(V[i], V[i] + L * d[i - 1], color=GREY_B,
                         stroke_width=1.5, dash_length=0.09,
                         stroke_opacity=0.55) for i in range(n)],
            *[Arc(radius=rho, start_angle=a_in[i], angle=turn[i],
                  arc_center=V[i], color=cols[i], stroke_width=3,
                  stroke_opacity=0.6) for i in range(n)],
            *[_chevron(V[i], V[(i + 1) % n], GREY_B, size=0.16, width=2.5)
              for i in range(n)])

        poly = always_redraw(
            lambda: Polygon(*[Vt(i) for i in range(n)], stroke_color=WHITE,
                            stroke_width=4)
            if t.get_value() > 1e-3 else Dot(O, radius=0.04))
        rays = VGroup(*[always_redraw(
            lambda i=i: DashedLine(Vt(i), Vt(i) + L * d[i - 1],
                                   color=GREY_A, stroke_width=2.5,
                                   dash_length=0.09))
            for i in range(n)])
        wedges = VGroup(*[always_redraw(
            lambda i=i: _wedge_dir(Vt(i), a_in[i], turn[i], rho, cols[i],
                                   op=0.88))
            for i in range(n)])
        labels = VGroup()
        for i in range(n):
            mid = a_in[i] + turn[i] / 2
            o = (rho + 0.36) * np.array([np.cos(mid), np.sin(mid), 0.0])
            lb = tag(f"ε{'₁₂₃₄₅'[i]}", 28, cols[i]).move_to(V[i] + o)
            lb.add_updater(lambda m, i=i, o=o: m.move_to(Vt(i) + o))
            labels.add(lb)

        dots = VGroup(*[Dot(p, radius=0.05) for p in V])
        self.play(Create(poly), FadeIn(dots), run_time=1.2)
        self.add(ghost[n + 1 + n:])              # direction-of-travel marks
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.15),
                  run_time=1.2)
        self.play(LaggedStart(*[FadeIn(w) for w in wedges], lag_ratio=0.12),
                  LaggedStart(*[FadeIn(lb) for lb in labels], lag_ratio=0.12),
                  run_time=1.4)
        self.hold(0.8)

        # the shrink: every vertex slides towards O along its own ray
        self.add(ghost)
        self.bring_to_back(ghost)
        self.remove(dots)
        self.play(t.animate.set_value(0.0), run_time=4.5,
                  rate_func=rate_functions.ease_in_out_sine)
        self.hold(0.4)

        ring = Circle(radius=rho + 0.02, color=YELLOW_B, stroke_width=4)
        ring.move_to(O)
        full = tag("360°", 34, YELLOW_B).move_to(O + np.array([2.25, -0.95, 0]))
        self.play(Create(ring), FadeIn(full), run_time=1.0)
        self.play(Write(caption("ε₁ + ε₂ + … + εₙ  =  360°", 34)))
        self.hold(2.2)


# ===================================================== F5 AREA BY SHEAR

class F5_TriangleShear(Board):
    """The apex slides along the parallel to the base: a shear. Every
    horizontal slice keeps its length (Cavalieri), so the area never moves.
    With the apex over an end of the base the triangle is right-angled, and
    a half-turned copy completes the b × h rectangle."""

    def construct(self):
        y0, h = -2.3, 4.1
        A = np.array([-2.7, y0, 0.0])
        B = np.array([1.9, y0, 0.0])
        b = B[0] - A[0]
        C0 = np.array([-1.8, y0 + h, 0.0])           # well off-centre
        C_far = np.array([4.4, y0 + h, 0.0])
        C1 = np.array([B[0], y0 + h, 0.0])            # apex over B
        N = 7
        cols = [BLUE_D, BLUE_B]

        def stripes(C):
            out = []
            for k in range(N):
                f0, f1 = k / N, (k + 1) / N
                out.append([A + f0 * (C - A), B + f0 * (C - B),
                            B + f1 * (C - B), A + f1 * (C - A)])
            return out

        for C in (C0, C_far, C1):
            check(all(abs(area(s1) - area(s2)) < 1e-9 for s1, s2 in
                      zip(stripes(C0), stripes(C))),
                  "every slice keeps its area under the shear")
            check(abs(area([A, B, C]) - b * h / 2) < 1e-9, "area = bh/2")
        M = (A + C1) / 2
        D1 = A + (C1 - B)
        check(close([2 * M - p for p in (A, B, C1)], [C1, D1, A]),
              "half-turn about the hypotenuse midpoint completes the box")

        def striped(C):
            return VGroup(*[mk(s, cols[k % 2], op=0.85, stroke_width=0.8)
                            for k, s in enumerate(stripes(C))])

        tri = striped(C0)
        outline = always_redraw(lambda: Polygon(
            A, B, tri[-1].get_vertices()[2], stroke_color=WHITE,
            stroke_width=3))
        base_line = guide(A + LEFT * 0.8, B + RIGHT * 3.2, YELLOW_B)
        top_line = guide(np.array([A[0] - 0.8, y0 + h, 0]),
                         np.array([B[0] + 3.2, y0 + h, 0]), YELLOW_B)
        base = Line(A, B, color=YELLOW_B, stroke_width=8)
        harrow = DoubleArrow([A[0] - 1.05, y0, 0], [A[0] - 1.05, y0 + h, 0],
                             buff=0, stroke_width=3, color=YELLOW_B,
                             tip_length=0.2, max_tip_length_to_length_ratio=1)
        h_lab = tag("h", 32, YELLOW_B).next_to(harrow, LEFT, buff=0.15)
        b_lab = tag("b", 32, YELLOW_B).next_to(base, DOWN, buff=0.18)

        self.play(Create(base_line), FadeIn(tri), Create(outline),
                  run_time=1.4)
        self.play(Create(base), FadeIn(b_lab), run_time=0.7)
        self.play(Create(top_line), GrowFromCenter(harrow), FadeIn(h_lab),
                  run_time=1.0)
        self.hold(0.5)

        # the shear: apex along the top parallel; slices slide, keep length
        self.play(Transform(tri, striped(C_far)), run_time=2.2)
        self.hold(0.3)
        self.play(Transform(tri, striped(C1)), run_time=1.8)
        ra = _right_mark(B, UP, LEFT, size=0.24, color=WHITE)
        self.play(Create(ra), run_time=0.4)
        self.hold(0.3)

        # a half-turned copy completes the rectangle b x h
        copy = mk([A, B, C1], ORANGE, op=0.85)
        mdot = Dot(M, radius=0.07, color=ORANGE)
        self.play(FadeIn(copy), FadeIn(mdot), run_time=0.5)
        self.play(Rotate(copy, angle=PI, about_point=M), run_time=2.0)
        rect = Polygon(A, B, C1, D1, stroke_color=YELLOW_B, stroke_width=5)
        half1 = tag("½bh", 32).move_to((A + B + C1) / 3 + DOWN * 0.15)
        half2 = tag("½bh", 32).move_to((A + C1 + D1) / 3 + UP * 0.15)
        self.play(FadeOut(mdot), Create(rect), FadeIn(half1), FadeIn(half2),
                  run_time=1.0)
        self.play(Write(caption("A  =  ½ · b · h", 36)))
        self.hold(2.2)


# ================================================ F6 CENTROID DIVIDES 2:1

class F6_Centroid(Board):
    """Medians AM and BN meet at G; P, Q are the midpoints of AG, BG.
    Two midlines: NM (AB halved towards C) and PQ (AB halved towards G)
    are equal and parallel, so PQMN is a parallelogram. A half-turn about
    G maps it onto itself, so GM = GP = PA and GN = GQ = QB."""

    def construct(self):
        A = np.array([-1.2, 3.2, 0.0])
        B = np.array([-4.8, -2.4, 0.0])
        C = np.array([4.0, -2.4, 0.0])
        M = (B + C) / 2                  # midpoint of BC (the formula's M)
        N = (C + A) / 2                  # midpoint of CA
        K = (A + B) / 2                  # midpoint of AB (third median)
        G = (A + B + C) / 3
        P, Q = (A + G) / 2, (B + G) / 2

        check(abs(_cross(M - A, G - A)) < 1e-9
              and abs(_cross(N - B, G - B)) < 1e-9,
              "G lies on medians AM and BN")
        check(close(Q - P, (B - A) / 2) and close(M - N, (B - A) / 2),
              "PQ and NM: both AB halved, same direction")
        check(close(2 * G - P, M) and close(2 * G - Q, N),
              "half-turn about G swaps P<->M and Q<->N")
        check(abs(_cross(K - C, G - C)) < 1e-9, "third median through G")
        check(abs(np.linalg.norm(A - G) - 2 * np.linalg.norm(G - M)) < 1e-9,
              "AG = 2 GM")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)

        def name(s, p, d, col=WHITE):
            return tag(s, 30, col).move_to(p + d)

        names = VGroup(name("A", A, UP * 0.32), name("B", B, LEFT * 0.35),
                       name("C", C, RIGHT * 0.35))
        side_ticks = VGroup(_ticks(C, N), _ticks(N, A),
                            _ticks(B, M, 2), _ticks(M, C, 2))
        dN, dM = Dot(N, radius=0.06), Dot(M, radius=0.06)
        m_lab = name("M", M, DOWN * 0.34)

        self.play(Create(tri), FadeIn(names), run_time=1.2)
        self.play(FadeIn(dN), FadeIn(dM), FadeIn(side_ticks), FadeIn(m_lab),
                  run_time=0.8)
        med_a = Line(A, M, color=ORANGE, stroke_width=4)
        med_b = Line(B, N, color=TEAL_D, stroke_width=4)
        dG = Dot(G, radius=0.08, color=YELLOW_B)
        g_lab = name("G", G, 0.46 * np.array([np.cos(1.06), np.sin(1.06), 0]),
                     YELLOW_B)
        self.play(Create(med_a), Create(med_b), run_time=1.2)
        self.play(FadeIn(dG), FadeIn(g_lab), run_time=0.5)

        # midline 1: AB halved towards C lands on NM
        ab1 = Line(A, B, color=YELLOW_B, stroke_width=6)
        self.play(Create(ab1), run_time=0.5)
        self.play(ab1.animate.scale(0.5, about_point=C), run_time=1.5)

        # P, Q: midpoints of AG and BG
        dP, dQ = Dot(P, radius=0.06), Dot(Q, radius=0.06)
        tA = VGroup(_ticks(A, P, 1, ORANGE), _ticks(P, G, 1, ORANGE))
        tB = VGroup(_ticks(B, Q, 2, TEAL_B), _ticks(Q, G, 2, TEAL_B))
        self.play(FadeIn(dP), FadeIn(dQ), FadeIn(tA), FadeIn(tB),
                  run_time=0.8)

        # midline 2: AB halved towards G lands on PQ
        ab2 = Line(A, B, color=YELLOW_B, stroke_width=6)
        self.play(Create(ab2), run_time=0.5)
        self.play(ab2.animate.scale(0.5, about_point=G), run_time=1.5)

        # equal and parallel: a parallelogram
        par = mk([P, Q, M, N], YELLOW_E, op=0.3, stroke_width=0)
        self.add(par)
        self.bring_to_back(par)
        self.play(FadeIn(par), FadeIn(_chevron(Q, P, YELLOW_B)),
                  FadeIn(_chevron(M, N, YELLOW_B)), run_time=0.7)
        self.hold(0.4)

        # half-turn about G maps the parallelogram onto itself
        ghost = Polygon(P, Q, M, N, stroke_color=YELLOW_B, stroke_width=5)
        corner = Dot(P, radius=0.09, color=RED_B)
        self.play(FadeIn(ghost), FadeIn(corner), run_time=0.4)
        self.play(Rotate(VGroup(ghost, corner), angle=PI, about_point=G),
                  run_time=2.0)
        self.play(FadeOut(ghost), FadeOut(corner), run_time=0.4)
        self.play(FadeIn(_ticks(G, M, 1, ORANGE)),
                  FadeIn(_ticks(G, N, 2, TEAL_B)), run_time=0.7)
        self.hold(0.4)

        # the same holds for the third median
        med_c = DashedLine(C, K, color=GREY_B, stroke_width=3,
                           dash_length=0.1)
        self.play(FadeIn(Dot(K, radius=0.06)),
                  FadeIn(VGroup(_ticks(A, K, 3), _ticks(K, B, 3))),
                  Create(med_c), run_time=0.9)
        self.play(Write(caption("AG : GM  =  2 : 1", 36)))
        self.hold(2.2)


# ===================================================== F8 VARIGNON

class F8_Varignon(Board):
    """Each side of the midpoint quadrilateral is half a diagonal, moved
    parallel to itself (a half-size copy towards the opposite vertex), so
    it is a parallelogram. The diagonals cut the figure into four triangles;
    in each, the midlines make four congruent pieces, two inside the
    parallelogram and two outside. The outside pair folds exactly onto the
    inside pair (one slides along the diagonal, one turns over), so the
    parallelogram is half the quadrilateral. (Convex case: the diagonals
    must cross inside.)"""

    def construct(self):
        A = np.array([-4.6, -2.2, 0.0])
        B = np.array([2.4, -2.6, 0.0])
        C = np.array([4.6, 1.4, 0.0])
        D = np.array([-1.2, 3.25, 0.0])
        Q = [A, B, C, D]
        check(_convex_ccw(Q), "quadrilateral convex, counter-clockwise")
        Mid = [(Q[i] + Q[(i + 1) % 4]) / 2 for i in range(4)]   # M1..M4
        M1, M2, M3, M4 = Mid
        check(close(M2 - M1, (C - A) / 2) and close(M3 - M4, (C - A) / 2),
              "M1M2 and M4M3 are AC halved")
        check(close(M4 - M1, (D - B) / 2) and close(M3 - M2, (D - B) / 2),
              "M1M4 and M2M3 are BD halved")
        # diagonals meet at X inside both
        t, s = np.linalg.solve(np.column_stack([(C - A)[:2], -(D - B)[:2]]),
                               (B - A)[:2])
        check(0 < t < 1 and 0 < s < 1, "diagonals cross inside")
        X = A + t * (C - A)
        Ph = [(X + V) / 2 for V in Q]                 # half-way to X

        cols = [BLUE_D, ORANGE, GREEN_D, RED_D]       # quadrants XAB ...
        slides, flips, s_moves, f_moves = [], [], [], []
        outer_area = 0.0
        for i in range(4):
            U, W = Q[i], Q[(i + 1) % 4]
            pU, pW, m = Ph[i], Ph[(i + 1) % 4], Mid[i]
            sl = [pU, U, m]                           # slides towards X
            fl = [pW, m, W]                           # turns over pW-m
            check(close([p + (X - pU) for p in sl], [X, pU, pW]),
                  "sliding piece lands on the X-corner piece")
            c = (pW + m) / 2
            check(close([2 * c - p for p in fl], [m, pW, pU]),
                  "turned piece lands on the central piece")
            outer_area += area(sl) + area(fl)
            slides.append(mk(sl, cols[i], op=0.85, stroke_width=1.5))
            flips.append(mk(fl, cols[i], op=0.85, stroke_width=1.5))
            s_moves.append(X - pU)
            f_moves.append(c)
        check(abs(area(Mid) - area(Q) / 2) < 1e-9, "[M1M2M3M4] = ½[ABCD]")
        check(abs(outer_area - area(Mid)) < 1e-9, "outside = inside")

        quad = Polygon(*Q, stroke_color=WHITE, stroke_width=4)
        out_dirs = [_unit(V - X) for V in Q]
        names = VGroup(*[tag(nm, 30).move_to(V + 0.36 * d) for nm, V, d in
                         zip("ABCD", Q, out_dirs)])
        mids = VGroup(*[Dot(m, radius=0.06) for m in Mid])
        ticks = VGroup()
        for i in range(4):
            U, W = Q[i], Q[(i + 1) % 4]
            ticks.add(_ticks(U, Mid[i], i + 1), _ticks(Mid[i], W, i + 1))
        mlabs = VGroup()
        for i, m in enumerate(Mid):
            U, W = Q[i], Q[(i + 1) % 4]
            nrm = _unit(np.array([(W - U)[1], -(W - U)[0], 0.0]))  # outward
            mlabs.add(tag(f"M{'₁₂₃₄'[i]}", 26, YELLOW_B).move_to(m + 0.4 * nrm))
        var = Polygon(*Mid, stroke_color=YELLOW_B, stroke_width=4)

        self.play(Create(quad), FadeIn(names), run_time=1.2)
        self.play(FadeIn(mids), FadeIn(ticks), FadeIn(mlabs), run_time=0.9)
        self.play(Create(var), run_time=0.8)

        # parallelogram: each side is half a diagonal, slid parallel
        chevs = VGroup()
        for (P0, P1), (c1, c2), n_ch in (((A, C), (B, D), 1),
                                         ((B, D), (A, C), 2)):
            diag = DashedLine(P0, P1, color=GREY_B, stroke_width=2.5,
                              dash_length=0.1)
            k1 = Line(P0, P1, color=GREY_A, stroke_width=5)
            k2 = Line(P0, P1, color=GREY_A, stroke_width=5)
            self.play(Create(diag), FadeIn(k1), FadeIn(k2), run_time=0.7)
            self.play(k1.animate.scale(0.5, about_point=c1),
                      k2.animate.scale(0.5, about_point=c2), run_time=1.5)
            sides = ((M1, M2), (M4, M3)) if n_ch == 1 else ((M1, M4), (M2, M3))
            ch = VGroup(*[_chevron(p, q, YELLOW_B, n=n_ch) for p, q in sides])
            chevs.add(ch)
            self.play(FadeOut(k1), FadeOut(k2), FadeIn(ch), run_time=0.5)
        self.hold(0.4)

        # area: the diagonals and the midlines of the four triangles
        inner = Polygon(*Ph, stroke_color=WHITE, stroke_width=1.5)
        diags = VGroup(Line(A, C, color=WHITE, stroke_width=1.5),
                       Line(B, D, color=WHITE, stroke_width=1.5))
        par_fill = mk(Mid, YELLOW_E, op=0.22, stroke_width=0)
        self.add(par_fill)
        self.bring_to_back(par_fill)
        self.play(FadeIn(par_fill), Create(inner), FadeIn(diags),
                  *[FadeIn(p) for p in slides + flips], run_time=1.2)
        self.bring_to_front(var)
        self.hold(0.6)

        # where the outside pieces were: a faint fill stays behind
        traces = VGroup(*[p.copy().set_fill(opacity=0.16).set_stroke(
            width=0) for p in slides + flips])
        self.add(traces)
        self.bring_to_front(quad, *slides, *flips)
        self.play(*[p.animate.shift(v) for p, v in zip(slides, s_moves)],
                  run_time=1.6)
        self.play(*[Rotate(p, angle=PI, about_point=c)
                    for p, c in zip(flips, f_moves)], run_time=1.9)
        self.bring_to_front(var, chevs)
        self.play(Indicate(var, color=YELLOW, scale_factor=1.0), run_time=0.8)
        self.play(Write(caption("[M₁M₂M₃M₄]  =  ½ [ABCD]", 34)))
        self.hold(2.2)


# ===================================================== F9 NAPOLEON

def _rot2(p, c, th):
    """Point p turned by th about c (in the plane)."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


class F9_Napoleon(Board):
    """The centres N₁ N₂ N₃ and the vertices A B C make a hexagon whose
    angles at the centres are 120°. Turn the piece at B by 120° about N₃
    and the piece at C by 120° about N₂: both reach A, and with the piece
    at A they close up (the hexagon's angles at A, B, C sum to 360°) into
    a mirror image of N₁N₂N₃ across N₂N₃. So N₂N₃ halves the 120° at N₃
    and at N₂: two angles of 60°."""

    def construct(self):
        def R(p, th):
            return np.array([p[0] * np.cos(th) - p[1] * np.sin(th),
                             p[0] * np.sin(th) + p[1] * np.cos(th)])

        th0 = 240 * DEGREES                  # lay the base CA level
        A, B, C = [R(np.array(p), th0) for p in
                   ((0.55, 1.25), (-1.75, -0.95), (2.0, -0.95))]

        def apex(X, Y):                       # outward for counter-clockwise
            return X + R(Y - X, -PI / 3)

        Ap, Bp, Cp = apex(B, C), apex(C, A), apex(A, B)
        N1m, N2m, N3m = [(X + Y + Z) / 3 for X, Y, Z in
                         ((B, C, Ap), (C, A, Bp), (A, B, Cp))]
        N1pm = _rot2(N1m, N3m, TAU / 3)[:2]
        pts = np.array([A, B, C, Ap, Bp, Cp, N1pm])
        lo, hi = pts.min(axis=0) - 0.45, pts.max(axis=0) + 0.45
        F = Frame(lo[0], hi[0], lo[1], hi[1])
        A, B, C, Ap, Bp, Cp = [F.P(p) for p in (A, B, C, Ap, Bp, Cp)]
        N1, N2, N3 = [F.P(p) for p in (N1m, N2m, N3m)]
        N1p = _rot2(N1, N3, TAU / 3)

        check(_cross(B - A, C - A) > 0, "ABC counter-clockwise")
        check(max(_angle(A, B, C), _angle(B, A, C), _angle(C, A, B))
              < 2 * PI / 3, "all angles < 120°: the hexagon is convex")
        check(close(_rot2(B, N3, TAU / 3), A), "120° about N3 sends B to A")
        check(close(_rot2(C, N2, -TAU / 3), A), "−120° about N2 sends C to A")
        check(close(_rot2(N1, N2, -TAU / 3), N1p),
              "both turned pieces meet at the same N1'")
        u = _unit(N3 - N2)
        refl = N2 + 2 * np.dot(N1 - N2, u) * u - (N1 - N2)
        check(close(refl, N1p), "N1' is the mirror image of N1 in N2N3")
        sides = [np.linalg.norm(N1 - N2), np.linalg.norm(N2 - N3),
                 np.linalg.norm(N3 - N1)]
        check(max(sides) - min(sides) < 1e-9, "N1N2N3 equilateral")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        eqs = VGroup(mk([B, C, Ap], GREY_B, op=0.22),
                     mk([C, A, Bp], GREY_B, op=0.22),
                     mk([A, B, Cp], GREY_B, op=0.22))
        G0 = (A + B + C) / 3
        names = VGroup(*[tag(s, 30).move_to(P + 0.38 * _unit(P - G0))
                         for s, P in (("A", A), ("B", B), ("C", C))])
        Ns = [N1, N2, N3]
        ndots = VGroup(*[Dot(P, radius=0.07, color=YELLOW_B) for P in Ns])
        # label each centre away from its own triangle's base
        nlabs = VGroup(*[tag(s, 26, YELLOW_B).move_to(P + 0.42 * _unit(P - G0))
                         for s, P in (("N₁", N1), ("N₂", N2), ("N₃", N3))])
        hexv = [A, N3, B, N1, C, N2]
        spokes = VGroup(*[Line(hexv[i], hexv[(i + 1) % 6], color=WHITE,
                               stroke_width=2.5) for i in range(6)])
        arcs120 = VGroup(angle_arc(N1, B, C, 0.36, YELLOW_B, 4),
                         angle_arc(N2, C, A, 0.36, YELLOW_B, 4),
                         angle_arc(N3, A, B, 0.36, YELLOW_B, 4))
        # equal spokes: N1B = N1C, N2C = N2A, N3A = N3B
        sticks = VGroup(_ticks(N1, B, 1), _ticks(N1, C, 1),
                        _ticks(N2, C, 2), _ticks(N2, A, 2),
                        _ticks(N3, A, 3), _ticks(N3, B, 3))

        self.play(Create(tri), FadeIn(names), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(e) for e in eqs], lag_ratio=0.25),
                  run_time=1.2)
        self.bring_to_front(tri)
        self.play(FadeIn(ndots), FadeIn(nlabs), run_time=0.7)
        self.play(Create(spokes), Create(arcs120), FadeIn(sticks),
                  run_time=1.3)
        self.hold(0.5)

        # the hexagon = Napoleon's triangle + three outer pieces
        nap = mk(Ns, YELLOW_E, op=0.45, stroke_color=YELLOW_B, stroke_width=4)
        T1 = mk([N2, A, N3], BLUE_D, op=0.85)
        T2 = mk([N3, B, N1], TEAL_D, op=0.85)
        T3 = mk([N1, C, N2], ORANGE, op=0.85)
        self.play(FadeOut(arcs120),
                  eqs.animate.set_fill(opacity=0.08).set_stroke(opacity=0.35),
                  tri.animate.set_stroke(opacity=0.25),
                  FadeIn(nap), FadeIn(T1), FadeIn(T2), FadeIn(T3),
                  run_time=1.2)
        self.bring_to_front(ndots)
        self.hold(0.5)

        # 120° about N3 carries B to A; −120° about N2 carries C to A
        r_arc = 0.62
        arc3 = Arc(radius=r_arc, start_angle=_dir(N3, N1), angle=TAU / 3,
                   arc_center=N3, color=YELLOW_B, stroke_width=4)
        arc2 = Arc(radius=r_arc, start_angle=_dir(N2, N1), angle=-TAU / 3,
                   arc_center=N2, color=YELLOW_B, stroke_width=4)
        # N2N3 bisects both 120° arcs (that is the theorem), so the
        # labels go in the Napoleon-side half, where 60° will replace them
        l3 = tag("120°", 24, YELLOW_B).move_to(
            N3 + 1.0 * angle_mid_dir(N3, N1, N2))
        l2 = tag("120°", 24, YELLOW_B).move_to(
            N2 + 1.0 * angle_mid_dir(N2, N1, N3))
        self.bring_to_front(sticks)
        self.play(Rotate(T2, angle=TAU / 3, about_point=N3), Create(arc3),
                  FadeIn(l3), run_time=2.0)
        self.play(Rotate(T3, angle=-TAU / 3, about_point=N2), Create(arc2),
                  FadeIn(l2), run_time=2.0)
        self.bring_to_front(ndots, names[0], l3, l2)
        kite_half = Polygon(N2, N3, N1p, stroke_color=WHITE, stroke_width=4)
        self.play(Create(kite_half), run_time=0.6)
        self.hold(0.4)

        # the Napoleon triangle turned over N2N3 lands exactly on them
        flip = mk(Ns, YELLOW_E, op=0.0, stroke_color=YELLOW_B, stroke_width=5)
        self.play(FadeOut(sticks), run_time=0.3)
        self.play(Rotate(flip, angle=PI, axis=_unit(N3 - N2), about_point=N2),
                  run_time=1.8)
        self.hold(0.3)
        self.play(FadeOut(flip), run_time=0.4)

        # so N2N3 halves each 120°: 60° and 60°
        def sx(P, X, Y, k=0.98):
            return tag("60°", 24).move_to(P + k * angle_mid_dir(P, X, Y))

        def mirror_half(P, Nn):
            # the mirror half is cut by the edge PA: label the wider part
            w1, w2 = _angle(P, Nn, A), _angle(P, A, N1p)
            return sx(P, Nn, A, 1.0) if w1 > w2 else sx(P, A, N1p, 1.0)

        halves = VGroup(sx(N3, N1, N2), mirror_half(N3, N2),
                        sx(N2, N1, N3), mirror_half(N2, N3))
        arc1 = angle_arc(N1, N2, N3, 0.42, YELLOW_B, 4)
        self.play(FadeOut(l3), FadeOut(l2), FadeIn(halves), run_time=0.8)
        self.play(Create(arc1), FadeIn(sx(N1, N2, N3, 0.7)), run_time=0.7)
        self.play(Write(caption("N₁N₂N₃ is equilateral", 34)))
        self.hold(2.2)


# ===================================================== F10 CEVA BY AREAS

class F10_Ceva(Board):
    """Cevians AD, BE, CF meet at P; x, y, z are the areas of PBC, PCA,
    PAB. Lemma (per cevian): z = [ABP] and y = [ACP] share the base AP.
    Slide AP along its own line until P reaches D: neither area changes
    (base length and line kept, apex fixed: a shear). Now both triangles
    have the apex A* and bases BD, DC on BC, so z : y = BD : DC. Likewise
    CE : EA = x : z and AF : FB = y : x; the product telescopes to 1."""

    def construct(self):
        A = np.array([-5.3, 3.25, 0.0])
        B = np.array([-6.3, -2.45, 0.0])
        C = np.array([0.7, -2.45, 0.0])
        wx, wy, wz = 4.0, 3.0, 5.0                    # x : y : z
        P = (wx * A + wy * B + wz * C) / (wx + wy + wz)

        def meet(U, V, W1, W2):
            t, _ = np.linalg.solve(np.column_stack([(V - U)[:2],
                                                    (W1 - W2)[:2]]),
                                   (W1 - U)[:2])
            return U + t * (V - U)

        D, E, F = meet(A, P, B, C), meet(B, P, C, A), meet(C, P, A, B)
        x, y, z = area([P, B, C]), area([P, C, A]), area([P, A, B])
        n = np.linalg.norm
        check(abs(n(D - B) / n(C - D) - z / y) < 1e-9, "BD/DC = z/y")
        check(abs(n(E - C) / n(A - E) - x / z) < 1e-9, "CE/EA = x/z")
        check(abs(n(F - A) / n(B - F) - y / x) < 1e-9, "AF/FB = y/x")
        check(abs((z / y) * (x / z) * (y / x) - 1) < 1e-12, "product = 1")

        cz, cy, cx = BLUE_D, TEAL_D, ORANGE
        tz = mk([P, A, B], cz, op=0.75)
        ty = mk([P, C, A], cy, op=0.75)
        tx = mk([P, B, C], cx, op=0.75)
        areas = {"x": tx, "y": ty, "z": tz}
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        cev = VGroup(*[Line(U, V, color=WHITE, stroke_width=2.5)
                       for U, V in ((A, D), (B, E), (C, F))])
        G0 = (A + B + C) / 3
        vl = VGroup(*[tag(s, 28).move_to(Q + 0.36 * _unit(Q - G0))
                      for s, Q in (("A", A), ("B", B), ("C", C))])

        def side_lab(s, Q, U, W):
            nrm = _unit(np.array([(W - U)[1], -(W - U)[0], 0.0]))
            if np.dot(nrm, Q - G0) < 0:
                nrm = -nrm
            return tag(s, 28).move_to(Q + 0.33 * nrm)

        fl = VGroup(side_lab("D", D, B, C), side_lab("E", E, C, A),
                    side_lab("F", F, A, B))
        rays = sorted(_dir(P, Q) % TAU for Q in (A, B, C, D, E, F))
        gaps = [((rays[(i + 1) % 6] - rays[i]) % TAU, rays[i])
                for i in range(6)]
        g, a0 = max(gaps)
        p_lab = tag("P", 26).move_to(P + 0.34 * np.array(
            [np.cos(a0 + g / 2), np.sin(a0 + g / 2), 0.0]))
        letters = VGroup(*[tag(s, 34).move_to(np.mean(pts, axis=0)) for s, pts in
                           (("x", [P, B, C]), ("y", [P, C, A]),
                            ("z", [P, A, B]))])

        self.play(Create(tri), FadeIn(vl), run_time=1.0)
        self.play(Create(cev), FadeIn(fl), FadeIn(Dot(P, radius=0.06)),
                  FadeIn(p_lab), run_time=1.2)
        self.add(tx, ty, tz)
        self.bring_to_back(tx, ty, tz)
        self.play(FadeIn(tx), FadeIn(ty), FadeIn(tz), FadeIn(letters),
                  run_time=0.9)
        self.hold(0.4)

        col = {"x": ORANGE, "y": TEAL_B, "z": BLUE_B}
        rows = [2.55, 1.55, 0.55]
        tx0 = 3.9
        fills = {"x": cx, "y": cy, "z": cz}

        def lemma(U, V, Q1, Q2, s1, s2, text, row, fast=1.0):
            """U: a vertex, V: the foot of its cevian on side Q1Q2; s1, s2:
            the area letters of U-P-Q1 and U-P-Q2. Slide UP along its own
            line until P reaches V: each triangle keeps its base length on
            the same line and its apex, hence its area. Then both have the
            apex U* and bases Q1V, VQ2 on one line: areas as Q1V : VQ2."""
            v = V - P
            Us = U + v
            a1, a2 = area([U, P, Q1]), area([U, Q2, P])
            check(abs(area([Us, V, Q1]) - a1) < 1e-9
                  and abs(area([Us, Q2, V]) - a2) < 1e-9,
                  "sliding the base along its line keeps both areas")
            check(abs(a1 / a2 - n(V - Q1) / n(Q2 - V)) < 1e-9,
                  "same apex: areas as the bases")
            d = _unit(Q2 - Q1)
            H = Q1 + np.dot(Us - Q1, d) * d
            check(0 < np.dot(H - Q1, d) < n(Q2 - Q1), "height lands on side")

            t1 = mk([U, P, Q1], fills[s1], op=0.9)
            t2 = mk([U, P, Q2], fills[s2], op=0.9)
            base = Line(U, P, color=WHITE, stroke_width=8)
            self.play(*[m.animate.set_fill(opacity=0.14)
                        for m in areas.values()],
                      FadeIn(t1), FadeIn(t2), Create(base),
                      run_time=0.7 * fast)
            self.play(Transform(t1, mk([Us, V, Q1], fills[s1], op=0.9)),
                      Transform(t2, mk([Us, V, Q2], fills[s2], op=0.9)),
                      base.animate.move_to((Us + V) / 2),
                      run_time=1.5 * fast)
            hl = DashedLine(Us, H, color=WHITE, stroke_width=3,
                            dash_length=0.08)
            rm = _right_mark(H, Us - H, d, 0.16, WHITE)
            b1 = Line(Q1, V, color=col[s1], stroke_width=9)
            b2 = Line(V, Q2, color=col[s2], stroke_width=9)
            self.play(Create(hl), Create(rm), Create(b1), Create(b2),
                      FadeIn(Dot(Us, radius=0.06)), run_time=0.8 * fast)
            eq = Text(text, font_size=30, t2c=col).move_to([tx0, rows[row], 0])
            self.play(FadeIn(eq, shift=RIGHT * 0.2), run_time=0.6 * fast)
            self.hold(0.3 * fast)
            gone = [m for m in self.mobjects if m not in keep]
            self.play(*[FadeOut(m) for m in gone if m is not eq],
                      *[m.animate.set_fill(opacity=0.75)
                        for m in areas.values()], run_time=0.5 * fast)
            keep.append(eq)
            return eq

        keep = list(self.mobjects)
        lemma(A, D, B, C, "z", "y", "BD : DC  =  z : y", 0)
        lemma(B, E, C, A, "x", "z", "CE : EA  =  x : z", 1, fast=0.8)
        lemma(C, F, A, B, "y", "x", "AF : FB  =  y : x", 2, fast=0.8)

        prod = Text("z/y · x/z · y/x  =  1", font_size=32, t2c=col)
        prod.move_to([tx0, -0.85, 0])
        rule = Line([tx0 - 2.2, -0.15, 0], [tx0 + 2.2, -0.15, 0],
                    color=GREY_B, stroke_width=2)
        self.play(Create(rule), FadeIn(prod), run_time=0.8)
        # each letter meets itself once above and once below the bar
        for i, j in ((0, 6), (2, 8), (4, 10)):
            self.play(Indicate(prod[i], scale_factor=1.4),
                      Indicate(prod[j], scale_factor=1.4), run_time=0.5)
            self.play(prod[i].animate.set_opacity(0.28),
                      prod[j].animate.set_opacity(0.28), run_time=0.3)
        self.play(Indicate(prod[12], scale_factor=1.5), run_time=0.6)
        self.play(Write(caption("(BD/DC) · (CE/EA) · (AF/FB)  =  1", 32)))
        self.hold(2.2)


# ===================================================== F11 PICK

class F11_Pick(Board):
    """Cut the lattice polygon into triangles using every lattice point.
    Here each edge climbs one lattice row per lattice step, so the cut can
    run strip by strip: every triangle has a unit side on one lattice line
    and its apex on the next, area ½·1·1 = ½, so A = ½T. Count the T
    triangles by their angles: each gives 180°; gathered by lattice point
    they make a full turn at each interior point, a half turn at each
    boundary point, less the exterior angles at the corners, which add up
    to one full turn (F4). So 180°·T = 360°·I + 180°·B − 360°."""

    def construct(self):
        u = 1.25
        O = np.array([-5.2, -1.05, 0.0])         # lattice (0, 0) on screen

        def L(p):
            return O + u * np.array([p[0], p[1], 0.0])

        Vl = [(0, 0), (2, 0), (4, 1), (2, 2), (0, 1)]
        Tl = [[(0, 0), (1, 0), (0, 1)], [(0, 1), (1, 1), (1, 0)],
              [(1, 0), (2, 0), (1, 1)], [(1, 1), (2, 1), (2, 0)],
              [(2, 1), (3, 1), (2, 0)], [(3, 1), (4, 1), (2, 0)],
              [(0, 1), (1, 1), (2, 2)], [(1, 1), (2, 1), (2, 2)],
              [(2, 1), (3, 1), (2, 2)], [(3, 1), (4, 1), (2, 2)]]
        nv = len(Vl)
        grid = [(i, j) for i in range(-1, 6) for j in range(-1, 4)]

        def on_edge(p):
            for k in range(nv):
                a, b = np.array(Vl[k], float), np.array(Vl[(k + 1) % nv], float)
                q = np.array(p, float)
                if (abs(_cross(b - a, q - a)) < 1e-12
                        and np.dot(q - a, q - b) <= 1e-12):
                    return True
            return False

        def strictly_in(p, poly):
            m = len(poly)
            return all(_cross(np.subtract(poly[(k + 1) % m], poly[k]),
                              np.subtract(p, poly[k])) > 1e-12
                       for k in range(m))

        Bl = [p for p in grid if on_edge(p)]
        Il = [p for p in grid if strictly_in(p, Vl)]
        I, B, T = len(Il), len(Bl), len(Tl)
        check(_convex_ccw([L(p) for p in Vl]), "convex, counter-clockwise")
        check(all(abs(abs(area(t)) - 0.5) < 1e-12 for t in Tl),
              "every triangle has area ½")
        check(abs(sum(abs(area(t)) for t in Tl) - area(Vl)) < 1e-12,
              "the triangles fill the polygon")
        check(set(q for t in Tl for q in t) == set(Bl) | set(Il),
              "every lattice point of the polygon is used")
        for t in Tl:
            tt = t if area(t) > 0 else t[::-1]
            check(not any(q not in t and (strictly_in(q, tt) or any(
                abs(_cross(np.subtract(tt[(k + 1) % 3], tt[k]),
                           np.subtract(q, tt[k]))) < 1e-12
                and np.dot(np.subtract(q, tt[k]),
                           np.subtract(q, tt[(k + 1) % 3])) <= 0
                for k in range(3))) for q in grid),
                "triangle holds no other lattice point")
        for q in set(Bl) | set(Il):
            s = sum(_angle(L(q), *[L(w) for w in t if w != q])
                    for t in Tl if q in t)
            if q in Il:
                want = TAU
            elif q in Vl:
                k = Vl.index(q)
                want = _angle(L(q), L(Vl[k - 1]), L(Vl[(k + 1) % nv]))
            else:
                want = PI
            check(abs(s - want) < 1e-9, f"angles around {q}")
        check(T == 2 * I + B - 2, "T = 2I + B - 2")
        check(abs(area(Vl) - (I + B / 2 - 1)) < 1e-12, "Pick")

        cI, cB, cT = YELLOW_B, ORANGE, [BLUE_D, TEAL_D]
        dots = VGroup(*[Dot(L(p), radius=0.035, color=GREY_B) for p in grid])
        V = [L(p) for p in Vl]
        poly = Polygon(*V, stroke_color=WHITE, stroke_width=4)
        bdots = VGroup(*[Dot(L(p), radius=0.09, color=cB) for p in Bl])
        idots = VGroup(*[Dot(L(p), radius=0.09, color=cI) for p in Il])
        tris = VGroup(*[mk([L(w) for w in t], cT[k % 2], op=0.5,
                           stroke_width=1.5) for k, t in enumerate(Tl)])
        strips = VGroup(*[DashedLine(L((-0.6, j)), L((4.6, j)), color=GREY_B,
                                     stroke_width=1.5, dash_length=0.08)
                          for j in (0, 1, 2)])
        halves = VGroup(*[tag("½", 28).move_to(np.mean([L(w) for w in t],
                                                       axis=0))
                          for t in Tl])

        px = 4.1
        row_bi = VGroup(tag(f"B = {B}", 32, cB), tag(f"I = {I}", 32, cI)
                        ).arrange(RIGHT, buff=0.8).move_to([px, 3.3, 0])
        row_t = VGroup(tag(f"T = {T}", 32, BLUE_B), tag("A = ½ · T", 32)
                       ).arrange(RIGHT, buff=0.8).move_to([px, 2.5, 0])

        self.play(FadeIn(dots), run_time=0.7)
        self.play(Create(poly), run_time=1.0)
        self.play(FadeIn(bdots), FadeIn(row_bi[0]), run_time=0.7)
        self.play(FadeIn(idots), FadeIn(row_bi[1]), run_time=0.7)

        # strip by strip: each triangle has a unit side and height 1
        self.add(tris)
        self.bring_to_back(tris)
        self.bring_to_back(dots)
        self.play(Create(strips), LaggedStart(*[FadeIn(t) for t in tris],
                                              lag_ratio=0.12),
                  run_time=1.6)
        self.bring_to_front(poly, bdots, idots)
        self.play(FadeIn(row_t[0]), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(h) for h in halves], lag_ratio=0.08),
                  FadeIn(row_t[1]), run_time=1.1)
        self.hold(0.5)

        # angle count at the lattice points
        r = 0.3
        discs = VGroup()
        gaps = []
        for p in Il:
            discs.add(Circle(radius=r, color=cI, fill_opacity=0.95,
                             stroke_width=1.5).move_to(L(p)))
        for p in Bl:
            if p in Vl:
                k = Vl.index(p)
                prv, nxt = L(Vl[k - 1]), L(Vl[(k + 1) % nv])
                discs.add(_wedge(L(p), prv, nxt, r, cB, op=0.95))
                a_in = _dir(prv, L(p))
                turn = (_dir(L(p), nxt) - a_in) % TAU
                gaps.append((_wedge_dir(L(p), a_in, turn, r, RED_D, op=0.95),
                             L(p), a_in, turn))
            else:                                 # on an edge: half turn
                k = next(m for m in range(nv) if abs(_cross(
                    np.subtract(Vl[(m + 1) % nv], Vl[m]),
                    np.subtract(p, Vl[m]))) < 1e-12)
                a_in = _dir(L(Vl[k]), L(Vl[(k + 1) % nv]))
                discs.add(_wedge_dir(L(p), a_in, PI, r, cB, op=0.95))
        check(abs(sum(g[3] for g in gaps) - TAU) < 1e-9,
              "the exterior gaps add up to one full turn")
        edges = VGroup(*[Line(L(a), L(b), color=WHITE, stroke_width=1.5)
                         for t in Tl for a, b in ((t[0], t[1]), (t[1], t[2]),
                                                  (t[2], t[0]))])
        self.play(FadeOut(halves), FadeOut(strips), run_time=0.4)
        self.play(FadeIn(discs), *[FadeIn(g[0]) for g in gaps],
                  run_time=1.0)
        self.add(edges, poly)
        self.hold(0.5)

        # the pictorial equation: (half turn)·T = (turn)·I + (half)·B − (turn)
        def half(col):
            return Sector(radius=r, start_angle=0, angle=PI, color=col,
                          fill_opacity=0.95).set_stroke(WHITE, 1.5)

        def full(col):
            return Circle(radius=r, color=col, fill_opacity=0.95,
                          stroke_width=1.5)

        slot_T = half(BLUE_D).set_opacity(0)
        slot_R = full(RED_D).set_opacity(0)
        eq = VGroup(slot_T, tag("T", 30, BLUE_B), tag("=", 30),
                    full(cI), tag("I", 30, cI), tag("+", 30),
                    half(cB), tag("B", 30, cB), tag("−", 30), slot_R)
        eq.arrange(RIGHT, buff=0.1)
        eq.move_to([px, 1.2, 0])
        check(eq.get_left()[0] > L((5, 0))[0] + 0.3
              and eq.get_right()[0] < 6.6, "equation fits the panel")

        # one triangle's three corners close into a half turn (F1)
        demo = Tl[6]
        Hc = slot_T.get_bottom()          # centre of the half disc's arc
        P3 = [L(w) for w in demo]
        start = 0.0
        moves = []
        for m in range(3):
            v, nx, pv = P3[m], P3[(m + 1) % 3], P3[(m + 2) % 3]
            if _cross(nx - v, pv - v) < 0:
                nx, pv = pv, nx
            a1, span = _dir(v, nx), _angle(v, nx, pv)
            w = _wedge_dir(v, a1, span, r, BLUE_B, op=1.0)
            self.add(w)
            moves.append(_glide(w, v, Hc, start - a1))
            start += span
        check(abs(start - PI) < 1e-9, "demo corners close a half turn")
        self.play(*moves, run_time=1.6)
        self.play(FadeIn(eq[1]), run_time=0.4)

        # the exterior gaps gather into one full turn (F4)
        Rc = slot_R.get_center()
        self.play(*[g[0].animate.shift(Rc - g[1]) for g in gaps],
                  run_time=1.6)
        self.play(*[FadeIn(m) for m in eq[2:9]], run_time=0.9)
        self.hold(0.4)

        # rows sit between lattice rows, clear of the dots at column 5
        row_4 = tag("T = 2I + B − 2", 32).move_to([px, 0.25, 0])
        row_5 = tag("A = ½T = I + B/2 − 1", 32, YELLOW_B).move_to(
            [px, -0.62, 0])
        for row in (row_4, row_5):
            check(row.get_right()[0] < SAFE_X and
                  row.get_left()[0] > L((5, 0))[0] + 0.3,
                  "panel rows inside the frame, clear of the lattice")
        self.play(FadeIn(row_4, shift=DOWN * 0.15), run_time=0.7)
        self.play(FadeOut(discs), FadeIn(halves), run_time=0.8)
        self.play(FadeIn(row_5, shift=DOWN * 0.15), run_time=0.7)
        self.play(Write(caption("A  =  I + B/2 − 1", 36)))
        self.hold(2.2)
