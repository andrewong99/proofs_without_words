# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_e.py — circle & conic scenes: E2, E4, E5, E6, E7, E8, E9, E10.
#
# Every scene builds its figure from verified coordinates and asserts the
# proof's invariants with check(), so a wrong construction fails the render.


# ----------------------------------------------------------- private helpers

def _lbl(s, size=26, color=WHITE, bg=0.0):
    """tag() with an optional dark backing box for text that sits on fills."""
    t = tag(s, size, color)
    if bg > 0:
        box = BackgroundRectangle(t, color=BLACK, fill_opacity=bg, buff=0.06)
        return VGroup(box, t)
    return t


def _dot(p, color=WHITE, r=0.06):
    return Dot(to3(p), radius=r, color=color)


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _free_dir(X, towards=(), dirs=()):
    """Unit vector at X along the middle of the widest angular gap between
    the directions to the points `towards` and the extra angles `dirs`
    (degrees).  Used to park a point's label clear of every line at it."""
    X = to3(X)
    angs = [float(np.degrees(np.arctan2(*(to3(q) - X)[1::-1])))
            for q in towards] + [float(d) for d in dirs]
    angs = sorted(a % 360.0 for a in angs)
    best, mid = -1.0, 90.0
    for i, a in enumerate(angs):
        b = angs[(i + 1) % len(angs)] + (360.0 if i == len(angs) - 1 else 0.0)
        if b - a > best:
            best, mid = b - a, (a + b) / 2
    t = mid * DEGREES
    return np.array([np.cos(t), np.sin(t), 0.0])


def _circle_dirs(X, centre):
    """The two tangent directions (degrees) of a circle through X."""
    d = to3(X) - to3(centre)
    a = float(np.degrees(np.arctan2(d[1], d[0])))
    return (a + 90.0, a - 90.0)


# =================================================================== E2

def _unroll(rho, phi, mu, r):
    """Unrolling the disc of radius r, cut along its upward radius.

    (rho, phi): radius, and angle along that circle measured from its
    lowest point (phi in [-pi, pi]; +-pi are the two lips of the cut).
    mu = 1 is the disc, mu = 0 the triangle.  At stage mu every circle is an
    arc about the common centre (0, c), c = r(1-mu)/mu, of radius c + rho
    and length 2*pi*rho, its lowest point (0, -rho) fixed.  The map is
    area-preserving at every stage: (c + rho) * d(psi)/d(phi) = rho.
    Written with sinc so mu -> 0 is exact: (rho*phi, -rho), a trapezoid
    strip per ring, the whole disc an isosceles triangle 2*pi*r by r.
    """
    rho = np.asarray(rho, float)
    phi = np.asarray(phi, float)
    den = r * (1.0 - mu) + rho * mu
    safe = np.where(den > 0, den, 1.0)
    psi = np.where(den > 0, phi * rho * mu / safe, 0.0)
    x = phi * rho * np.sinc(psi / PI)
    y = -rho + 0.5 * phi * rho * psi * np.sinc(psi / TAU) ** 2
    return np.stack([x, y], -1)


class E2_CircleAnnuli(Board):
    """Cut the disc along a radius and unroll its rings: each circle keeps
    its length 2πρ and drops to depth ρ below the centre, so the rings
    straighten into strips stacking to a triangle of base 2πr, height r."""

    def construct(self):
        r, n = 2.0, 8
        O = np.array([0.0, 1.1, 0.0])
        rk = [r * k / n for k in range(n + 1)]
        cols = [BLUE_D, TEAL_D]
        NA, NS = 241, 25
        m = ValueTracker(1.0)

        def scr(P):
            return [O + np.array([p[0], p[1], 0.0]) for p in P]

        def ring_xy(r1, r2, mu):
            ph = np.linspace(-PI, PI, NA)
            rr = np.linspace(r2, r1, NS)
            parts = [_unroll(r2, ph, mu, r),
                     _unroll(rr, np.full(NS, PI), mu, r)[1:],
                     _unroll(r1, ph[::-1], mu, r)[1:],
                     _unroll(rr[::-1], np.full(NS, -PI), mu, r)[1:-1]]
            return np.vstack(parts)

        def circle_xy(rho, mu):
            return _unroll(rho, np.linspace(-PI, PI, NA), mu, r)

        # ---- exactness: every ring keeps its area at every stage, ends as a
        # trapezoid between its two circumferences, and they tile the triangle
        tri = [(-PI * r, -r), (PI * r, -r), (0.0, 0.0)]
        for k in range(n):
            ann = PI * (rk[k + 1] ** 2 - rk[k] ** 2)
            for mu in (0.75, 0.4, 0.1):
                check(abs(abs(area(ring_xy(rk[k], rk[k + 1], mu))) - ann)
                      < 2e-3 * ann, f"ring {k} keeps its area at mu={mu}")
            trap = [(-PI * rk[k + 1], -rk[k + 1]), (PI * rk[k + 1], -rk[k + 1]),
                    (PI * rk[k], -rk[k]), (-PI * rk[k], -rk[k])]
            check(abs(abs(area(trap)) - ann) < 1e-12, f"strip {k} = ring {k}")
            check(close(ring_xy(rk[k], rk[k + 1], 0.0)[0], trap[0]),
                  f"strip {k} lands on its trapezoid")
        check(abs(abs(area(tri)) - PI * r * r) < 1e-12, "triangle = πr²")
        for mu in np.linspace(0, 1, 41):           # stays inside the frame
            P = scr(circle_xy(r, mu))
            check(max(p[1] for p in P) < SAFE_TOP and
                  max(abs(p[0]) for p in P) < SAFE_X, "unroll stays on screen")

        def ring(k):
            return always_redraw(lambda k=k: Polygon(
                *scr(ring_xy(rk[k], rk[k + 1], m.get_value())),
                fill_color=cols[k % 2], fill_opacity=FILL,
                stroke_color=WHITE, stroke_width=1.0))

        rings = [ring(k) for k in range(n)]
        rho = rk[n // 2]

        def curve(rr, color, w):
            return always_redraw(lambda: VMobject(
                stroke_color=color, stroke_width=w).set_points_as_corners(
                    scr(circle_xy(rr, m.get_value()))))

        outer = curve(r, YELLOW_B, 6)
        inner = curve(rho, ORANGE, 6)
        cut_l = always_redraw(lambda: VMobject(
            stroke_color=RED_D, stroke_width=5).set_points_as_corners(
                scr(_unroll(np.linspace(0, r, 40), np.full(40, -PI),
                            m.get_value(), r))))
        cut_r = always_redraw(lambda: VMobject(
            stroke_color=RED_D, stroke_width=5).set_points_as_corners(
                scr(_unroll(np.linspace(0, r, 40), np.full(40, PI),
                            m.get_value(), r))))

        # the downward radius never moves: it becomes the triangle's height
        rad_r = Line(O, O + DOWN * r, color=YELLOW_B, stroke_width=6)
        rad_p = Line(O, O + DOWN * rho, color=ORANGE, stroke_width=6)
        l_r = _lbl("r", 28, YELLOW_B, 0.6).move_to(
            O + DOWN * (r * 0.72) + RIGHT * 0.3)
        l_p = _lbl("ρ", 28, ORANGE, 0.6).move_to(
            O + DOWN * (rho * 0.5) + LEFT * 0.3)
        d135 = np.array([np.cos(0.75 * PI), np.sin(0.75 * PI), 0.0])
        l_2pr = _lbl("2πr", 26, YELLOW_B, 0.7).move_to(O + d135 * r)
        l_2pp = _lbl("2πρ", 26, ORANGE, 0.7).move_to(O + d135 * rho)

        self.play(LaggedStart(*[FadeIn(g) for g in rings], lag_ratio=0.12),
                  run_time=1.8)
        self.add(*rings)
        self.play(Create(outer), Create(inner), run_time=1.2)
        self.play(FadeIn(l_2pr), FadeIn(l_2pp), run_time=0.6)
        self.play(Create(rad_r), Create(rad_p), FadeIn(_dot(O)),
                  FadeIn(l_r), FadeIn(l_p), run_time=0.8)
        self.hold(0.8)
        self.play(Create(cut_l), run_time=0.7)
        self.add(cut_r)
        self.play(FadeOut(l_2pr), FadeOut(l_2pp), run_time=0.4)

        # ---- unroll
        self.bring_to_front(rad_r, rad_p, l_r, l_p)
        self.play(m.animate.set_value(0.0), run_time=5.0,
                  rate_func=smooth)
        self.hold(0.5)

        base_y = O[1] - r
        frame = Polygon(*[O + np.array([p[0], p[1], 0.0]) for p in tri],
                        stroke_color=YELLOW_B, stroke_width=4)
        b_2pr = tag("2πr", 28, YELLOW_B).move_to([0.0, base_y - 0.38, 0.0])
        b_2pp = tag("2πρ", 26, ORANGE)
        b_2pp.move_to([-PI * (rho + 0.6 * b_2pp.height) - 0.3
                       - b_2pp.width / 2, O[1] - rho, 0.0])
        self.play(Create(frame), run_time=1.0)
        self.play(FadeIn(b_2pr), FadeIn(b_2pp), run_time=0.6)
        self.play(Write(caption("A  =  ½ · 2πr · r  =  πr²", 34)))
        self.hold(2.2)


# =================================================================== E4

def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _tick(p, q, size=0.22, color=WHITE, n=1, gap=0.08):
    """n short ticks across segment pq at its midpoint (equal-length marks)."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = (p + q) / 2
    g = VGroup()
    for i in range(n):
        c = mid + d * gap * (i - (n - 1) / 2)
        g.add(Line(c - nrm * size / 2, c + nrm * size / 2,
                   color=color, stroke_width=3))
    return g


def _right_mark(v, p, q, s=0.24, color=RED_C, width=3):
    """Square corner at v between rays v->p and v->q (must be 90°)."""
    v = to3(v)
    u1, u2 = _unit(to3(p) - v), _unit(to3(q) - v)
    return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(
        [v + u1 * s, v + (u1 + u2) * s, v + u2 * s])


class E4_Thales(Board):
    """The radius to C cuts the triangle into two isosceles triangles: the
    angles at C are α and β, the base angles again α and β.  Carried to C
    by the parallel to AB, the four make a straight angle: 2α + 2β = 180°,
    so ∠ACB = α + β = 90° wherever C is."""

    def construct(self):
        R = 3.0
        O = np.array([0.0, 0.2, 0.0])
        A, B = O + LEFT * R, O + RIGHT * R
        th = ValueTracker(115.0)
        COL_A, COL_B = BLUE_B, GREEN_B

        def Cpt(deg):
            t = deg * DEGREES
            return O + R * np.array([np.cos(t), np.sin(t), 0.0])

        for deg in (60.0, 90.0, 115.0, 122.0):          # the proof's claims
            C = Cpt(deg)
            al, be = _ang(A, B, C), _ang(B, A, C)
            check(abs(_ang(C, A, O) - al) < 1e-9, "isosceles AOC: α = α")
            check(abs(_ang(C, O, B) - be) < 1e-9, "isosceles BOC: β = β")
            check(abs(_ang(C, C + LEFT, A) - al) < 1e-9, "alternate angle α")
            check(abs(_ang(C, B, C + RIGHT) - be) < 1e-9, "alternate angle β")
            check(abs(2 * al + 2 * be - PI) < 1e-9, "2α + 2β = 180°")
            check(abs(_ang(C, A, B) - PI / 2) < 1e-9, "∠ACB = 90°")

        def polar(deg, rr):
            t = deg * DEGREES
            return rr * np.array([np.cos(t), np.sin(t), 0.0])

        def lab_pos(deg):
            """Where each label sits when C is at angle deg.  The alternate-
            angle labels hug the parallel, above the circle's dip."""
            C = Cpt(deg)
            return {"lA": A + 1.05 * angle_mid_dir(A, B, C),
                    "lC1": C + 1.0 * angle_mid_dir(C, A, O),
                    "lB": B + 1.05 * angle_mid_dir(B, A, C),
                    "lC2": C + 1.0 * angle_mid_dir(C, O, B),
                    "lpA": C + polar(191.0, 0.98),
                    "lpB": C + polar(-11.0, 0.98),
                    "tC": C + 0.33 * _unit(C - O)}

        labels = {"lA": tag("α", 28, COL_A), "lC1": tag("α", 28, COL_A),
                  "lB": tag("β", 28, COL_B), "lC2": tag("β", 28, COL_B),
                  "lpA": tag("α", 28, COL_A), "lpB": tag("β", 28, COL_B),
                  "tC": tag("C", 28)}
        for key, pos in lab_pos(th.get_value()).items():
            labels[key].move_to(pos)

        def parts(deg):
            C = Cpt(deg)
            L, Rr = C + LEFT * 2.2, C + RIGHT * 2.2
            d = {}
            d["fA"] = mk([A, O, C], BLUE_D, 0.38, stroke_width=0)
            d["fB"] = mk([B, O, C], TEAL_D, 0.38, stroke_width=0)
            d["par"] = DashedLine(L, Rr, color=YELLOW_B, stroke_width=3,
                                  dash_length=0.12)
            d["ca"] = Line(C, A, color=WHITE, stroke_width=4)
            d["cb"] = Line(C, B, color=WHITE, stroke_width=4)
            d["oc"] = Line(O, C, color=GREY_A, stroke_width=3)
            d["tk"] = VGroup(_tick(O, A), _tick(O, B), _tick(O, C))
            d["aA"] = angle_arc(A, B, C, 0.7, COL_A)
            d["aC1"] = angle_arc(C, A, O, 0.68, COL_A)
            d["aB"] = angle_arc(B, A, C, 0.7, COL_B)
            d["aC2"] = angle_arc(C, O, B, 0.68, COL_B)
            d["pA"] = angle_arc(C, L, A, 0.46, COL_A)
            d["pB"] = angle_arc(C, B, Rr, 0.46, COL_B)
            d["ra"] = _right_mark(C, A, B, 0.3)
            d["dC"] = _dot(C)
            return d

        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(O)
        diam = Line(A, B, color=WHITE, stroke_width=4)
        fixed = VGroup(_dot(A), _dot(B), _dot(O),
                       tag("A", 28).next_to(A, LEFT, buff=0.18),
                       tag("B", 28).next_to(B, RIGHT, buff=0.18),
                       tag("O", 26).next_to(O, DOWN, buff=0.16))

        P = parts(th.get_value())
        P.update(labels)
        C0 = Cpt(th.get_value())
        self.play(Create(circ), Create(diam), run_time=1.2)
        self.play(FadeIn(fixed), run_time=0.6)
        self.play(FadeIn(P["dC"]), FadeIn(P["tC"]), Create(P["ca"]),
                  Create(P["cb"]), run_time=1.0)
        self.play(Create(P["oc"]), FadeIn(P["tk"]), run_time=0.9)
        self.play(FadeIn(P["fA"]), FadeIn(P["fB"]), run_time=0.6)
        self.bring_to_front(P["ca"], P["cb"], P["oc"], P["tk"], P["dC"])
        self.play(Create(P["aA"]), Create(P["aC1"]),
                  FadeIn(P["lA"]), FadeIn(P["lC1"]), run_time=0.9)
        self.play(Create(P["aB"]), Create(P["aC2"]),
                  FadeIn(P["lB"]), FadeIn(P["lC2"]), run_time=0.9)
        self.hold(0.5)
        self.play(Create(P["par"]), run_time=0.8)
        self.play(TransformFromCopy(P["aA"], P["pA"]),
                  TransformFromCopy(P["aB"], P["pB"]), run_time=1.6)
        self.play(FadeIn(P["lpA"]), FadeIn(P["lpB"]), run_time=0.5)
        # the four angles at C fill the straight angle on the parallel
        half = Arc(radius=1.45, start_angle=PI, angle=PI, arc_center=C0,
                   color=YELLOW_B, stroke_width=4)
        self.play(Create(half), run_time=0.9)
        self.play(FadeOut(half), run_time=0.5)
        self.play(Create(P["ra"]), run_time=0.7)
        self.hold(0.6)

        # ---- the same figure for every C on the circle
        shapes = ["fA", "fB", "par", "ca", "cb", "oc", "tk", "aA", "aC1", "aB",
                  "aC2", "pA", "pB", "ra", "dC"]
        self.remove(*[P[k] for k in shapes])
        live = always_redraw(lambda: VGroup(
            *[v for k, v in parts(th.get_value()).items()]))
        self.add(live)
        for key, lab in labels.items():
            lab.add_updater(lambda m, key=key: m.move_to(
                lab_pos(th.get_value())[key]))
        self.bring_to_front(*labels.values())
        self.play(th.animate.set_value(66.0), run_time=2.2)
        self.play(th.animate.set_value(122.0), run_time=2.6)
        self.play(th.animate.set_value(115.0), run_time=0.9)
        for lab in labels.values():
            lab.clear_updaters()
        self.play(Write(caption("2α + 2β = 180°   ⟹   ∠ACB = α + β = 90°", 32)))
        self.hold(2.2)


# =================================================================== E5

def _chord_ends(P, u):
    """Unit circle: the ends of the chord through P along u — (far end in
    direction +u, near end in direction -u)."""
    P, u = np.asarray(P, float)[:2], np.asarray(u, float)[:2]
    pu = float(P @ u)
    disc = np.sqrt(pu * pu + 1.0 - float(P @ P))
    return P + (-pu + disc) * u, P + (-pu - disc) * u


def _reflect(X, P, w):
    """Mirror image of X in the line through P along w."""
    X, P, w = to3(X), to3(P), _unit(w)
    d = X - P
    return P + 2 * np.dot(d, w) * w - d


def _row(*bits, buff=0.12):
    """A formula from coloured pieces: _row(("PD", TEAL_B), (":", WHITE), ...)."""
    return VGroup(*[tag(s, size, c) for s, c, size in
                    [(b[0], b[1], b[2] if len(b) > 2 else 28) for b in bits]]
                  ).arrange(RIGHT, buff=buff)


class E5_PowerOfAPoint(Board):
    """Chords AB and CD cross at P.  ∠A = ∠D (both stand on arc BC) and the
    angles at P are vertical, so triangle PDB, flipped over the bisector of
    ∠APD, lands inside PAC with its third side parallel to AC: the two
    triangles are similar, PD : PA = PB : PC, i.e. PA·PB = PC·PD."""

    def construct(self):
        R = 3.0
        c = np.array([-0.7, 0.2, 0.0])
        Pm = np.array([0.3, -0.25])
        u1 = np.array([np.cos(160 * DEGREES), np.sin(160 * DEGREES)])
        u2 = np.array([np.cos(100 * DEGREES), np.sin(100 * DEGREES)])
        Am, Bm = _chord_ends(Pm, u1)
        Cm, Dm = _chord_ends(Pm, u2)

        def S(p):
            return c + R * to3(p)

        P, A, B, C, D = (S(x) for x in (Pm, Am, Bm, Cm, Dm))
        w = _unit(A - P) + _unit(D - P)                 # bisector of ∠APD
        D1, B1 = _reflect(D, P, w), _reflect(B, P, w)

        # ---- the claims, here and in a spread of other configurations
        rng = np.random.default_rng(5)
        for trial in range(200):
            q = 0.8 * np.sqrt(rng.random()) * np.array(
                [np.cos(t := rng.random() * TAU), np.sin(t)])
            a1 = rng.random() * TAU
            a2 = a1 + rng.uniform(0.3, PI - 0.3)
            e1 = np.array([np.cos(a1), np.sin(a1)])
            e2 = np.array([np.cos(a2), np.sin(a2)])
            a_, b_ = _chord_ends(q, e1)
            c_, d_ = _chord_ends(q, e2)
            check(abs(_ang(a_, q, c_) - _ang(d_, q, b_)) < 1e-9,
                  "inscribed angles on arc BC agree")
            ww = _unit(to3(a_) - to3(q)) + _unit(to3(d_) - to3(q))
            d1, b1 = _reflect(d_, q, ww), _reflect(b_, q, ww)
            check(abs(np.linalg.det([(b1 - d1)[:2], (to3(c_) - to3(a_))[:2]]))
                  < 1e-9, "flipped third side ∥ AC")
            check(abs(np.linalg.norm(a_ - q) * np.linalg.norm(b_ - q) -
                      np.linalg.norm(c_ - q) * np.linalg.norm(d_ - q)) < 1e-12,
                  "PA·PB = PC·PD")
        check(close(D1, P + np.linalg.norm(D - P) * _unit(A - P), 1e-9),
              "D lands on PA")
        check(close(B1, P + np.linalg.norm(B - P) * _unit(C - P), 1e-9),
              "B lands on PC")
        check(abs(np.linalg.det([(B1 - D1)[:2], (C - A)[:2]])) < 1e-9, "D′B′ ∥ AC")

        COL_V, COL_I = YELLOW_B, ORANGE
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(c)
        ch1 = Line(A, B, color=WHITE, stroke_width=3)
        ch2 = Line(C, D, color=WHITE, stroke_width=3)
        big = mk([P, A, C], BLUE_D, 0.55)
        small = mk([P, D, B], TEAL_D, 0.85)
        dots = VGroup(*[_dot(X) for X in (A, B, C, D, P)])

        def out(X, k=0.36):
            return X + k * _unit(X - c)

        lA = tag("A", 28).move_to(out(A))
        lB = tag("B", 28).move_to(out(B))
        lC = tag("C", 28).move_to(out(C, 0.33))
        lD = tag("D", 28).next_to(D, RIGHT, buff=0.16)
        lP = tag("P", 28).move_to(P + 0.5 * np.array(
            [np.cos(10 * DEGREES), np.sin(10 * DEGREES), 0.0]))

        vA = angle_arc(P, A, C, 0.55, COL_V)
        vD = angle_arc(P, D, B, 0.55, COL_V)
        iA = angle_arc(A, P, C, 0.75, COL_I)
        iD = angle_arc(D, P, B, 0.75, COL_I)
        tB = float(np.arctan2(Bm[1], Bm[0]))
        tC = float(np.arctan2(Cm[1], Cm[0]))
        arcBC = Arc(radius=R, start_angle=tB, angle=(tC - tB) % TAU,
                    arc_center=c, color=COL_I, stroke_width=7)

        self.play(Create(circ), run_time=1.0)
        self.play(Create(ch1), Create(ch2), run_time=1.2)
        self.play(FadeIn(dots), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                  FadeIn(lD), FadeIn(lP), run_time=0.6)
        self.play(FadeIn(big), FadeIn(small), run_time=0.9)
        self.bring_to_front(dots)
        self.play(Create(vA), Create(vD), run_time=0.8)
        self.play(Create(arcBC), run_time=0.8)
        self.play(Create(iA), Create(iD), run_time=0.8)
        self.hold(0.8)

        # ---- flip PDB over the bisector of ∠APD
        axis = DashedLine(P - 1.6 * _unit(w), P + 1.6 * _unit(w),
                          color=GREY_A, stroke_width=2, dash_length=0.09)
        self.play(Create(axis), FadeOut(lD), FadeOut(lB), run_time=0.7)
        flip = VGroup(small, iD, vD)
        ghost = mk([P, D, B], TEAL_D, 0.16, stroke_width=0)
        self.add(ghost)
        self.bring_to_front(ch1, ch2, flip)
        self.play(Rotate(flip, angle=PI, axis=_unit(w), about_point=P),
                  run_time=2.2)
        check(close(small.get_vertices()[1], D1, 1e-6) and
              close(small.get_vertices()[2], B1, 1e-6), "flip lands on D′, B′")

        def polar(X, deg, k):
            t = deg * DEGREES
            return X + k * np.array([np.cos(t), np.sin(t), 0.0])

        lD1 = tag("D′", 26).move_to(polar(D1, 282, 0.36))
        lB1 = tag("B′", 26).move_to(polar(B1, 345, 0.38))
        self.play(FadeOut(axis), FadeIn(lD1), FadeIn(lB1),
                  FadeIn(lD), FadeIn(lB), run_time=0.7)
        self.bring_to_front(dots, _dot(D1), _dot(B1))
        g1 = guide(A, C, GREY_A, extend=0.18)
        g2 = guide(D1, B1, GREY_A, extend=0.55)
        self.play(Create(g1), Create(g2), run_time=0.9)
        prop = _row(("PD", TEAL_B, 30), (":", WHITE, 30), ("PA", BLUE_B, 30),
                    ("=", WHITE, 30), ("PB", TEAL_B, 30), (":", WHITE, 30),
                    ("PC", BLUE_B, 30))
        prop.move_to([4.6, 0.9, 0.0])
        self.play(FadeIn(prop), run_time=0.8)
        self.play(Write(caption("PA · PB  =  PC · PD", 34)))
        self.hold(2.2)


# =================================================================== E6

class E6_SecantTangent(Board):
    """Tangent PT and secant PAB.  The tangent–chord angle at T equals the
    inscribed angle at B (both stand on arc TA) and the angle at P is shared,
    so triangle PTA, flipped over the bisector of ∠TPB, lands inside PBT
    with its third side parallel to BT: PT : PB = PA : PT, i.e. PT² = PA·PB."""

    def construct(self):
        R = 2.6
        c = np.array([-2.4, 0.3, 0.0])

        def config(Pm, delta):
            """Unit circle: tangent point on the upper side; secant turned by
            delta from the line PO."""
            d = float(np.linalg.norm(Pm))
            e = -np.asarray(Pm, float) / d
            beta = np.arccos(1.0 / d)
            phiP = float(np.arctan2(Pm[1], Pm[0]))
            Tm = np.array([np.cos(phiP + beta), np.sin(phiP + beta)])
            cs, sn = np.cos(-delta), np.sin(-delta)
            u = np.array([cs * e[0] - sn * e[1], sn * e[0] + cs * e[1]])
            pu = float(Pm @ u)
            disc = np.sqrt(pu * pu - (float(Pm @ Pm) - 1.0))
            return Tm, Pm + (-pu - disc) * u, Pm + (-pu + disc) * u

        # ---- the claims, here and in a spread of other configurations
        rng = np.random.default_rng(6)
        for trial in range(200):
            d = rng.uniform(1.2, 4.0)
            th0 = rng.uniform(0, TAU)
            q = d * np.array([np.cos(th0), np.sin(th0)])
            dl = rng.uniform(-0.95, 0.95) * np.arcsin(1.0 / d)
            t_, a_, b_ = config(q, dl)
            check(abs(_ang(t_, q, a_) - _ang(b_, q, t_)) < 1e-9,
                  "tangent–chord angle = inscribed angle")
            ww = _unit(to3(t_) - to3(q)) + _unit(to3(b_) - to3(q))
            t1, a1 = _reflect(t_, q, ww), _reflect(a_, q, ww)
            check(abs(np.linalg.det([(a1 - t1)[:2], (to3(t_) - to3(b_))[:2]])) < 1e-9,
                  "flipped third side ∥ BT")
            check(abs(np.linalg.norm(t_ - q) ** 2 - np.linalg.norm(a_ - q)
                      * np.linalg.norm(b_ - q)) < 1e-9, "PT² = PA·PB")

        Pm = np.array([2.0, 0.0])
        Tm, Am, Bm = config(Pm, -16 * DEGREES)

        def S(p):
            return c + R * to3(p)

        O, P, T, A, B = S((0, 0)), S(Pm), S(Tm), S(Am), S(Bm)
        w = _unit(T - P) + _unit(B - P)                 # bisector of ∠TPB
        T1, A1 = _reflect(T, P, w), _reflect(A, P, w)
        check(abs(np.dot(T - O, T - P)) < 1e-9, "OT ⊥ PT")
        check(close(T1, P + np.linalg.norm(T - P) * _unit(B - P), 1e-9),
              "T lands on PB")
        check(close(A1, P + np.linalg.norm(A - P) * _unit(T - P), 1e-9),
              "A lands on PT")
        check(abs(np.linalg.det([(A1 - T1)[:2], (T - B)[:2]])) < 1e-9, "T′A′ ∥ BT")

        COL_V, COL_I = YELLOW_B, ORANGE
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(c)
        tan = Line(P, T, color=WHITE, stroke_width=3)
        sec = Line(P, B, color=WHITE, stroke_width=3)
        rad = Line(O, T, color=GREY_A, stroke_width=2)
        ra = _right_mark(T, O, P, 0.24)
        big = mk([P, B, T], BLUE_D, 0.5)
        small = mk([P, T, A], TEAL_D, 0.88)
        dots = VGroup(*[_dot(X) for X in (P, T, A, B)], _dot(O, GREY_A))

        ext_T = 2 * T - B                               # guide beyond T
        ext_B = 2 * B - T                               # guide beyond B
        lT = tag("T", 28).move_to(T + 0.4 * _free_dir(
            T, (P, O, B, A, ext_T), _circle_dirs(T, c)))
        lA = tag("A", 28).move_to(A + 0.38 * _free_dir(
            A, (P, B, T), _circle_dirs(A, c)))
        lB = tag("B", 28).move_to(B + 0.38 * _free_dir(
            B, (P, T, ext_B), _circle_dirs(B, c)))
        lP = tag("P", 28).next_to(P, RIGHT, buff=0.16)
        lO = tag("O", 24, GREY_A).next_to(O, DOWN, buff=0.14)

        vP = angle_arc(P, T, B, 0.9, COL_V)
        iT = angle_arc(T, P, A, 0.62, COL_I)
        iB = angle_arc(B, P, T, 0.9, COL_I)
        tA = float(np.arctan2(Am[1], Am[0]))
        tT = float(np.arctan2(Tm[1], Tm[0]))
        arcTA = Arc(radius=R, start_angle=tA, angle=(tT - tA) % TAU,
                    arc_center=c, color=COL_I, stroke_width=7)

        self.play(Create(circ), FadeIn(dots[4]), FadeIn(lO), run_time=1.0)
        self.play(Create(tan), Create(rad), FadeIn(dots[0]), FadeIn(dots[1]),
                  FadeIn(lP), FadeIn(lT), run_time=1.1)
        self.play(Create(ra), run_time=0.5)
        self.play(Create(sec), FadeIn(dots[2]), FadeIn(dots[3]),
                  FadeIn(lA), FadeIn(lB), run_time=1.0)
        self.play(FadeIn(big), FadeIn(small), run_time=0.9)
        self.bring_to_front(tan, sec, ra, dots)
        self.play(Create(vP), run_time=0.7)
        self.play(Create(arcTA), run_time=0.8)
        self.play(Create(iT), Create(iB), run_time=0.8)
        self.hold(0.8)

        # ---- flip PTA over the bisector of ∠TPB
        axis = DashedLine(P, P + 5.2 * _unit(w), color=GREY_A, stroke_width=2,
                          dash_length=0.09)
        ghost = mk([P, T, A], TEAL_D, 0.16, stroke_width=0)
        self.play(Create(axis), FadeOut(lA), run_time=0.7)
        flip = VGroup(small, iT, vP.copy())
        self.add(ghost)
        self.bring_to_front(tan, sec, flip)
        self.play(Rotate(flip, angle=PI, axis=_unit(w), about_point=P),
                  run_time=2.2)
        check(close(small.get_vertices()[1], T1, 1e-6) and
              close(small.get_vertices()[2], A1, 1e-6), "flip lands on T′, A′")
        lT1 = tag("T′", 26).move_to(T1 + 0.4 * _free_dir(
            T1, (P, B, A1, 2 * T1 - A1), _circle_dirs(T1, c)))
        lA1 = tag("A′", 26).move_to(A1 + 0.4 * _free_dir(
            A1, (P, T, T1, 2 * A1 - T1)))
        self.play(FadeOut(axis), FadeIn(lT1), FadeIn(lA1), FadeIn(lA),
                  run_time=0.7)
        self.bring_to_front(dots, _dot(T1), _dot(A1))
        g1 = guide(B, T, GREY_A, extend=0.1)
        g2 = guide(T1, A1, GREY_A, extend=0.3)
        self.play(Create(g1), Create(g2), run_time=0.9)
        prop = _row(("PT", TEAL_B, 30), (":", WHITE, 30), ("PB", BLUE_B, 30),
                    ("=", WHITE, 30), ("PA", TEAL_B, 30), (":", WHITE, 30),
                    ("PT", BLUE_B, 30))
        prop.move_to([4.55, 2.2, 0.0])
        self.play(FadeIn(prop), run_time=0.8)
        self.play(Write(caption("PT²  =  PA · PB", 34)))
        self.hold(2.2)


# =================================================================== E7

def _cx(p):
    p = to3(p)
    return complex(p[0], p[1])


def _pt(z):
    return np.array([z.real, z.imag, 0.0])


def _similarity(src, dst):
    """The direct similarity z -> m z + k taking src[0], src[1] to dst[0],
    dst[1]; fails the render unless it takes every src point to its dst."""
    s = [_cx(p) for p in src]
    t = [_cx(p) for p in dst]
    m = (t[1] - t[0]) / (s[1] - s[0])
    k = t[0] - m * s[0]
    for si, ti in zip(s, t):
        check(abs(m * si + k - ti) < 1e-9, "a direct similarity")
    return m, k


def _similar_path(src, dst):
    """alpha -> the polygon src carried part-way by the similarity onto dst:
    scale and turn grow geometrically about the moving centroid, so every
    intermediate polygon is similar to src (no vertex interpolation)."""
    m, k = _similarity(src, dst)
    s = [_cx(p) for p in src]
    g0 = sum(s) / len(s)
    g1 = m * g0 + k
    lam, ang = abs(m), float(np.angle(m))

    def at(alpha):
        mm = lam ** alpha * np.exp(1j * ang * alpha)
        gg = g0 + (g1 - g0) * alpha
        return [_pt(gg + mm * (z - g0)) for z in s]
    return at


class E7_Ptolemy(Board):
    """Scale triangle ABD by p = AC, ACD by a = AB and ABC by d = DA.  Glued
    along their equal sides (ap, ad) they make a parallelogram: the bottom is
    straight because ∠B + ∠D = 180° in a cyclic quadrilateral, and the two
    halves either side of the diagonal have sides ap, pd around the same
    angle ∠A.  The half-turn that swaps the halves carries the top side pq
    onto the bottom side ac + bd."""

    def construct(self):
        # ---- the cyclic quadrilateral (unit circle, counter-clockwise)
        wxyz = (62.0, 46.0, 26.0, 46.0)             # half-arcs AB, BC, CD, DA
        tA = 205.0
        angs = [tA, tA + 2 * wxyz[0], tA + 2 * sum(wxyz[:2]),
                tA + 2 * sum(wxyz[:3])]
        Au, Bu, Cu, Du = (np.array([np.cos(t * DEGREES), np.sin(t * DEGREES)])
                          for t in angs)

        def L(p, q):
            return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))

        a, b, c, d = L(Au, Bu), L(Bu, Cu), L(Cu, Du), L(Du, Au)
        p, q = L(Au, Cu), L(Bu, Du)
        check(abs(p * q - (a * c + b * d)) < 1e-12, "Ptolemy (numerically)")

        Rs = 2.2
        cC = np.array([-4.15, 0.35, 0.0])

        def S(u):
            return cC + Rs * to3(u)

        A, B, C, D = (S(u) for u in (Au, Bu, Cu, Du))

        # ---- the parallelogram, in product units, then fitted on screen
        kap = 1.3
        D3m = np.zeros(3)
        Apm = np.array([p * q * kap, 0.0, 0.0])
        m3, k3 = _similarity([Du, Bu], [D3m, Apm])
        C1m = _pt(m3 * _cx(Au) + k3)                       # T3 = ABD·p
        m1, k1 = _similarity([Au, Cu], [Apm, C1m])
        Gm = _pt(m1 * _cx(Du) + k1)                        # T1 = ACD·a
        m2, k2 = _similarity([Au, Bu], [Apm, Gm])
        C2m = _pt(m2 * _cx(Cu) + k2)                       # T2 = ABC·d
        check(abs(abs(m3) - p * kap) < 1e-9 and abs(abs(m1) - a * kap) < 1e-9
              and abs(abs(m2) - d * kap) < 1e-9, "scale factors p, a, d")
        xs = [v[0] for v in (D3m, Apm, C1m, C2m)]
        ys = [v[1] for v in (D3m, Apm, C1m, C2m)]
        shift = np.array([2.95 - (min(xs) + max(xs)) / 2,
                          0.25 - (min(ys) + max(ys)) / 2, 0.0])
        D3, Ap, C1, G, C2 = (v + shift for v in (D3m, Apm, C1m, Gm, C2m))
        T3 = [C1, Ap, D3]                # images of A, B, D
        T1 = [Ap, C1, G]                 # images of A, C, D
        T2 = [Ap, G, C2]                 # images of A, B, C
        check(abs(np.linalg.det([(G - C1)[:2], (C2 - C1)[:2]])) < 1e-9 and
              np.dot(G - C1, C2 - G) > 0, "the bottom is straight at G")
        check(close(Ap + C1, D3 + C2, 1e-9), "a parallelogram")
        check(abs(area(T1) + area(T2) + area(T3) -
                  area([D3, C1, C2, Ap])) < 1e-9 and
              all(area(t) > 0 for t in (T1, T2, T3)), "three tiles, no overlap")
        M = (Ap + C1) / 2
        check(close(2 * M - D3, C2, 1e-9) and close(2 * M - Ap, C1, 1e-9),
              "the half-turn takes the top side onto the bottom side")
        check(abs(L(C1[:2], C2[:2]) - (a * c + b * d) * kap) < 1e-9 and
              abs(L(D3[:2], Ap[:2]) - p * q * kap) < 1e-9, "pq vs ac + bd")
        for v in (D3, Ap, C1, C2, A, B, C, D):
            check(abs(v[0]) < SAFE_X and SAFE_BOTTOM < v[1] < SAFE_TOP,
                  "on screen")

        COL3, COL1, COL2 = BLUE_D, ORANGE, GREEN_D
        COL_B, COL_D, COL_A = PINK, BLUE_C, TEAL_A

        # ---- circle, quadrilateral, labels
        circ = Circle(radius=Rs, color=GREY_B, stroke_width=3).move_to(cC)
        quad = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=3)
        diag_p = DashedLine(A, C, color=GREY_A, stroke_width=2, dash_length=0.08)
        diag_q = DashedLine(B, D, color=GREY_A, stroke_width=2, dash_length=0.08)

        def rim(t_deg, k=0.33, s="", size=26, col=WHITE):
            t = t_deg * DEGREES
            return tag(s, size, col).move_to(
                cC + (Rs + k) * np.array([np.cos(t), np.sin(t), 0.0]))

        vlabels = VGroup(*[rim(t, 0.32, nm, 28) for t, nm in zip(angs, "ABCD")])
        mids = [(angs[i] + angs[(i + 1) % 4] + (360 if i == 3 else 0)) / 2
                for i in range(4)]
        slabels = VGroup(*[rim(t, 0.3, nm, 26, GREY_A)
                           for t, nm in zip(mids, "abcd")])

        def on_seg(P0, P1, f, off, s, col=GREY_A, size=24):
            dd = _unit(P1 - P0)
            n = np.array([-dd[1], dd[0], 0.0])
            return tag(s, size, col).move_to(P0 + f * (P1 - P0) + off * n)

        lp = on_seg(A, C, 0.3, 0.2, "p")
        lq = on_seg(B, D, 0.3, -0.2, "q")

        self.play(Create(circ), run_time=0.9)
        self.play(Create(quad), FadeIn(vlabels), FadeIn(slabels), run_time=1.2)
        self.play(Create(diag_p), Create(diag_q), FadeIn(lp), FadeIn(lq),
                  run_time=0.8)
        self.hold(0.4)

        # ---- the three scaled triangles
        shown = []

        def carry(src, dst, col, factor, tri_lbls):
            ghost = mk(src, col, 0.55)
            self.play(FadeIn(ghost), run_time=0.5)
            path = _similar_path(src, dst)
            mob = mk(src, col, FILL)
            fac = tag(factor, 28, WHITE)
            g_src = sum(to3(v) for v in src) / 3
            g_dst = sum(to3(v) for v in dst) / 3
            fac.move_to(g_src)
            self.add(mob, fac)

            def upd(m_, al):
                m_.become(mk(path(al), col, FILL))

            self.play(UpdateFromAlphaFunc(mob, upd),
                      fac.animate.move_to(g_dst), FadeOut(ghost),
                      run_time=1.8)
            self.add(*shown)
            self.bring_to_front(*shown, fac)
            self.play(*[FadeIn(t) for t in tri_lbls], run_time=0.5)
            shown.extend(tri_lbls + [fac])

        def side_lbl(P0, P1, s, away, k=0.3, col=WHITE, size=24, bg=0.0):
            mid = (to3(P0) + to3(P1)) / 2
            dd = _unit(to3(P1) - to3(P0))
            n = np.array([-dd[1], dd[0], 0.0])
            if np.dot(n, mid - to3(away)) < 0:
                n = -n
            return _lbl(s, size, col, bg).move_to(mid + k * n)

        cen = (D3 + Ap + C1 + C2) / 4
        l_pq = side_lbl(D3, Ap, "pq", cen, 0.3, YELLOW_B, 28)
        l_pd1 = side_lbl(D3, C1, "pd", cen, 0.36, WHITE, 26)
        l_ap = side_lbl(Ap, C1, "ap", D3, 0.28, WHITE, 24, 0.6)
        carry([A, B, D], T3, COL3, "×p", [l_pq, l_pd1, l_ap])

        l_ac = side_lbl(C1, G, "ac", cen, 0.32, YELLOW_B, 28)
        l_ad = side_lbl(Ap, G, "ad", C1, 0.26, WHITE, 24, 0.6)
        carry([A, C, D], T1, COL1, "×a", [l_ac, l_ad])

        l_bd = side_lbl(G, C2, "bd", cen, 0.32, YELLOW_B, 28)
        l_pd2 = side_lbl(Ap, C2, "pd", cen, 0.36, WHITE, 26)
        carry([A, B, C], T2, COL2, "×d", [l_bd, l_pd2])
        self.hold(0.4)

        # ---- why it closes up: ∠B + ∠D = 180° at G, and ∠A at both ends
        # of the diagonal with the sides pd beside it
        r_ang = 0.48
        cB = angle_arc(B, A, C, r_ang, COL_B)
        cD = angle_arc(D, C, A, r_ang, COL_D)
        gB = angle_arc(G, Ap, C2, r_ang, COL_B)      # T2's angle at B
        gD = angle_arc(G, C1, Ap, r_ang, COL_D)      # T1's angle at D
        check(abs(_ang(G, Ap, C2) - _ang(B, A, C)) < 1e-9 and
              abs(_ang(G, C1, Ap) - _ang(D, C, A)) < 1e-9 and
              abs(_ang(G, Ap, C2) + _ang(G, C1, Ap) - PI) < 1e-9,
              "∠B + ∠D = 180° at G")
        self.play(Create(cB), Create(cD), run_time=0.7)
        self.play(TransformFromCopy(cB, gB), TransformFromCopy(cD, gD),
                  run_time=1.2)
        cA = angle_arc(A, B, D, 0.5, COL_A)
        aC1 = angle_arc(C1, Ap, D3, 0.5, COL_A)      # T3's angle at A
        aAp = angle_arc(Ap, C1, C2, 0.5, COL_A)      # T1 + T2 at A'
        check(abs(_ang(C1, Ap, D3) - _ang(A, B, D)) < 1e-9 and
              abs(_ang(Ap, C1, C2) - _ang(A, B, D)) < 1e-9,
              "∠A at both ends of the diagonal")
        self.play(Create(cA), run_time=0.5)
        self.play(TransformFromCopy(cA, aC1), TransformFromCopy(cA, aAp),
                  run_time=1.2)
        self.hold(0.5)

        # ---- the half-turn about the centre: top side onto bottom side
        top = Line(D3, Ap, color=YELLOW_B, stroke_width=8)
        turn = top.copy()
        centre = _dot(M, YELLOW_B, 0.07)
        self.play(Create(top), FadeIn(centre), run_time=0.6)
        self.add(turn)
        self.play(Rotate(turn, angle=PI, about_point=M), run_time=2.2)
        check(close(turn.get_start(), C2, 1e-6) and
              close(turn.get_end(), C1, 1e-6), "pq lands on ac + bd")
        self.play(FadeOut(centre), run_time=0.4)
        self.bring_to_front(*shown)
        self.play(Write(caption("AC · BD  =  AB · CD  +  BC · AD", 34)))
        self.hold(2.2)


# =================================================================== E8

class E8_EllipseArea(Board):
    """Squash the disc of radius a vertically by b/a.  Every vertical chord,
    hence every vertical slice, keeps its width and has its height scaled by
    b/a, so every area is scaled by b/a: the ellipse has (b/a)·πa² = πab."""

    def construct(self):
        a, b = 2.9, 1.65
        k = b / a
        O = np.array([0.0, 0.4, 0.0])
        n, NP = 16, 41
        xs = np.linspace(-a, a, n + 1)

        def slice_pts(x0, x1, kk=1.0):
            t = np.linspace(x0, x1, NP)
            top = [O + np.array([x, kk * np.sqrt(max(a * a - x * x, 0.0)), 0.0])
                   for x in t]
            bot = [O + np.array([x, -kk * np.sqrt(max(a * a - x * x, 0.0)), 0.0])
                   for x in t[::-1]]
            return top + bot

        cols = [BLUE_D, TEAL_D]
        slices = [mk(slice_pts(xs[i], xs[i + 1]), cols[i % 2], FILL,
                     stroke_width=1.2) for i in range(n)]

        # ---- the claims: each slice's area scales by exactly b/a, the image
        # of the circle is the ellipse x²/a² + y²/b² = 1
        tot0 = tot1 = 0.0
        for i in range(n):
            s0 = abs(area(slice_pts(xs[i], xs[i + 1])))
            s1 = abs(area(slice_pts(xs[i], xs[i + 1], k)))
            check(abs(s1 - k * s0) < 1e-9, f"slice {i}: area × b/a")
            tot0, tot1 = tot0 + s0, tot1 + s1
        check(abs(tot0 - PI * a * a) < 2e-3 * PI * a * a, "slices fill πa²")
        check(abs(tot1 - PI * a * b) < 2e-3 * PI * a * b, "slices fill πab")
        for th in np.linspace(0, TAU, 37):
            x, y = a * np.cos(th), k * a * np.sin(th)
            check(abs((x / a) ** 2 + (y / b) ** 2 - 1) < 1e-12, "on the ellipse")

        ghost = Circle(radius=a, color=GREY_B, stroke_width=2).move_to(O)
        ghost = DashedVMobject(ghost, num_dashes=60)
        rad_h = Line(O, O + RIGHT * a, color=YELLOW_B, stroke_width=5)
        rad_v = Line(O, O + UP * a, color=YELLOW_B, stroke_width=5)
        x0 = 0.5 * (xs[4] + xs[5])
        h0 = np.sqrt(a * a - x0 * x0)
        chord = Line(O + np.array([x0, -h0, 0.0]), O + np.array([x0, h0, 0.0]),
                     color=ORANGE, stroke_width=6)
        l_a1 = _lbl("a", 28, YELLOW_B, 0.65).move_to(O + np.array([a / 2, 0.3, 0]))
        l_a2 = _lbl("a", 28, YELLOW_B, 0.65).move_to(O + np.array([0.3, a / 2, 0]))
        l_h = _lbl("h", 28, ORANGE, 0.65).move_to(
            O + np.array([x0 - 0.3, 0.42 * h0, 0.0]))
        l_A = _lbl("πa²", 30, WHITE, 0.65).move_to(O + np.array([1.35, -1.3, 0.0]))

        self.play(LaggedStart(*[FadeIn(s) for s in slices], lag_ratio=0.06),
                  run_time=1.6)
        self.play(Create(rad_h), Create(rad_v), FadeIn(l_a1), FadeIn(l_a2),
                  FadeIn(l_A), run_time=0.9)
        self.play(Create(chord), FadeIn(l_h), run_time=0.8)
        self.hold(0.6)
        self.add(ghost)
        self.play(FadeIn(ghost), run_time=0.4)

        # ---- the squash: y -> (b/a) y, every intermediate stage affine too
        movers = [*slices, rad_v, chord]
        self.play(FadeOut(l_a2), FadeOut(l_h), FadeOut(l_A), run_time=0.4)
        self.play(*[m.animate.stretch(k, 1, about_point=O) for m in movers],
                  run_time=2.6)
        check(close(rad_v.get_end(), O + UP * b, 1e-9), "radius a -> b")
        check(abs(chord.get_length() - 2 * k * h0) < 1e-9, "chord h -> (b/a)h")
        self.bring_to_front(rad_h, l_a1)

        l_b = _lbl("b", 28, YELLOW_B, 0.65).move_to(O + np.array([0.3, b / 2, 0]))
        l_kh = _lbl("(b/a)·h", 26, ORANGE, 0.65)
        l_kh.move_to(O + np.array([x0 - 0.1 - l_kh.width / 2, 0.42 * k * h0, 0]))
        l_B = _lbl("(b/a)·πa²", 28, WHITE, 0.65).move_to(
            O + np.array([1.35, -0.78, 0.0]))
        self.play(FadeIn(l_b), FadeIn(l_kh), run_time=0.7)
        self.play(FadeIn(l_B), run_time=0.7)
        self.play(Write(caption("A  =  (b/a) · πa²  =  πab", 34)))
        self.hold(2.2)


# =================================================================== E9

class E9_AnnulusChord(Board):
    """The chord of the outer circle tangent to the inner one is bisected at
    the point of tangency, at right angles to the radius r there:
    (L/2)² + r² = R².  So the ring's area π(R² − r²) is π(L/2)², the disc on
    the chord as diameter, whatever the two radii are."""

    def construct(self):
        h = 2.15                           # L/2, fixed
        O = np.array([-3.35, 0.15, 0.0])
        rt = ValueTracker(1.25)

        def geo(r):
            R = float(np.sqrt(r * r + h * h))
            T = O + UP * r
            return R, T, T + LEFT * h, T + RIGHT * h

        for r in (0.45, 1.25, 2.0):                         # the claims
            R, T, E1, E2 = geo(r)
            check(abs(np.linalg.norm(E1 - O) - R) < 1e-12 and
                  abs(np.linalg.norm(E2 - O) - R) < 1e-12, "chord ends on R")
            check(abs(np.dot(T - O, E2 - E1)) < 1e-12, "OT ⊥ chord at T")
            check(abs((R * R - r * r) - h * h) < 1e-12, "R² − r² = (L/2)²")
            check(O[1] + R < SAFE_TOP and O[1] - R > SAFE_BOTTOM and
                  O[0] - R > -SAFE_X, "ring on screen")

        def ring(r):
            R = geo(r)[0]
            return AnnularSector(inner_radius=r, outer_radius=R, angle=TAU,
                                 start_angle=0, fill_color=TEAL_D,
                                 fill_opacity=FILL, stroke_width=0
                                 ).move_arc_center_to(O)

        def parts(r):
            R, T, E1, E2 = geo(r)
            d = {}
            d["ring"] = ring(r)
            d["outer"] = Circle(radius=R, color=WHITE, stroke_width=3).move_to(O)
            d["inner"] = Circle(radius=r, color=WHITE, stroke_width=3).move_to(O)
            d["chord"] = Line(E1, E2, color=YELLOW_B, stroke_width=6)
            d["OT"] = Line(O, T, color=WHITE, stroke_width=4)
            d["OE"] = Line(O, E2, color=WHITE, stroke_width=4)
            d["ra"] = _right_mark(T, O, E2, 0.22)
            d["dO"] = _dot(O)
            d["dT"] = _dot(T, YELLOW_B)
            d["dE"] = VGroup(_dot(E1, YELLOW_B), _dot(E2, YELLOW_B))
            return d

        labels = {"lr": _lbl("r", 28, WHITE, 0.6),
                  "lh": _lbl("L/2", 26, YELLOW_B, 0.6),
                  "lR": _lbl("R", 28, WHITE, 0.6),
                  "lA": _lbl("π(R² − r²)", 24, WHITE, 0.6)}

        def lab_pos(r):
            R, T, E1, E2 = geo(r)
            n = _unit(E2 - O)
            return {"lr": (O + T) / 2 + LEFT * (0.12 + labels["lr"].width / 2),
                    "lh": (T + E2) / 2 + UP * 0.3,
                    "lR": (O + E2) / 2 + 0.3 * np.array([n[1], -n[0], 0.0]),
                    "lA": O + DOWN * (r + R) / 2}

        def place_labels(r):
            for key, pos in lab_pos(r).items():
                labels[key].move_to(pos)

        place_labels(rt.get_value())
        P = parts(rt.get_value())
        P.update(labels)
        self.play(FadeIn(P["ring"]), Create(P["outer"]), Create(P["inner"]),
                  run_time=1.3)
        self.play(FadeIn(P["lA"]), run_time=0.5)
        self.play(Create(P["chord"]), FadeIn(P["dT"]), FadeIn(P["dE"]),
                  run_time=0.9)
        self.play(Create(P["OT"]), FadeIn(P["dO"]), FadeIn(P["lr"]),
                  run_time=0.7)
        self.play(Create(P["ra"]), run_time=0.4)
        self.play(Create(P["OE"]), FadeIn(P["lR"]), FadeIn(P["lh"]),
                  run_time=0.8)
        pyth = tag("R²  =  r²  +  (L/2)²", 30).move_to([3.55, 3.05, 0.0])
        self.play(FadeIn(pyth), run_time=0.8)
        self.hold(0.6)

        # ---- the disc on the chord as diameter, set beside the ring
        _, T0, E10, E20 = geo(rt.get_value())
        cd = DashedVMobject(Circle(radius=h, color=ORANGE, stroke_width=3)
                            .move_to(T0), num_dashes=50)
        self.play(Create(cd), run_time=0.9)
        disc = Circle(radius=h, color=WHITE, stroke_width=3, fill_color=ORANGE,
                      fill_opacity=FILL).move_to(T0)
        diam = Line(E10, E20, color=YELLOW_B, stroke_width=6)
        side = VGroup(disc, diam)
        target = np.array([3.55, 0.05, 0.0])
        self.add(side)
        self.play(side.animate.move_to(target), run_time=1.5)
        check(close(disc.get_center(), target, 1e-9), "disc parked")
        l_L = _lbl("L", 28, YELLOW_B, 0.0).next_to(diam, UP, buff=0.14)
        l_D = _lbl("π(L/2)²", 28, WHITE, 0.6).move_to(target + DOWN * 0.95)
        self.play(FadeOut(cd), FadeIn(l_L), FadeIn(l_D), run_time=0.8)
        self.hold(0.5)

        # ---- any pair of radii with this chord: the ring keeps that area
        shapes = ["ring", "outer", "inner", "chord", "OT", "OE", "ra", "dO",
                  "dT", "dE"]
        self.remove(*[P[k] for k in shapes])
        live = always_redraw(lambda: VGroup(*parts(rt.get_value()).values()))
        self.add(live)
        updater = VMobject().add_updater(
            lambda m: place_labels(rt.get_value()))
        self.add(updater)
        self.bring_to_front(*labels.values())
        self.play(rt.animate.set_value(0.45), run_time=2.2)
        self.play(rt.animate.set_value(2.0), run_time=2.6)
        self.play(rt.animate.set_value(1.25), run_time=1.4)
        self.remove(updater)
        self.play(Write(caption("A  =  π(R² − r²)  =  π(L/2)²", 34)))
        self.hold(2.2)


# =================================================================== E10

class E10_ParabolaQuadrature(Board):
    """Archimedes: inscribe in the parabolic segment the triangle T whose
    apex has its tangent parallel to the chord, then the same in each
    leftover segment, and so on.  Each leftover segment slides along the
    parabola (an area-preserving map taking the parabola to itself) onto
    the tip of the first, where its triangle is T squeezed ×½ across and
    ×¼ up: T/8.  Two per parent, so every generation is ¼ of the one
    before, and A = T(1 + ¼ + 1/16 + …) = (4/3)T."""

    def construct(self):
        h = 1.5                                   # chord y = h², |x| ≤ h
        k = 2.3
        org = np.array([-2.2, -2.75, 0.0])

        def S(p):
            return org + k * np.array([p[0], p[1], 0.0])

        def par(x):
            return (x, x * x)

        def tri(u, v):
            m = (u + v) / 2
            return [par(u), par(v), par(m)]

        # ---- the claims, exactly as stated
        gens, chords = [], [(-h, h)]
        for g in range(5):
            gens.append([tri(u, v) for (u, v) in chords])
            chords = [c for (u, v) in chords
                      for c in ((u, (u + v) / 2), ((u + v) / 2, v))]
        tot = [sum(abs(area(t)) for t in gg) for gg in gens]
        T_area = tot[0]
        for g in range(4):
            check(abs(tot[g + 1] - tot[g] / 4) < 1e-12, f"generation {g + 2} = ¼")
        for (u, v) in [(-h, h), (-h, 0.0), (0.37, 1.1)]:
            m = (u + v) / 2
            check(abs((v * v - u * u) / (v - u) - 2 * m) < 1e-12,
                  "apex tangent ∥ chord")
        xs = np.linspace(-h, h, 2001)
        seg = abs(area([par(x) for x in xs]))
        check(abs(seg - 4 * T_area / 3) < 1e-5, "segment = (4/3)·T")

        def slide(pts, s):
            """(x, y) -> (x + s·h/2, y + s·h·x + s²h²/4): keeps y = x², det 1."""
            return [(x + s * h / 2, y + s * h * x + s * s * h * h / 4)
                    for (x, y) in pts]

        sub_xy = [par(x) for x in np.linspace(-h, 0.0, 60)]
        t2_xy = tri(-h, 0.0)
        tip_xy = [(-h / 2, h * h / 4), (h / 2, h * h / 4), (0.0, 0.0)]
        landed = slide(t2_xy, 1.0)
        check(close(landed, tip_xy, 1e-12), "slid triangle = tip triangle")
        check(close([(p[0] / 2, p[1] / 4) for p in tri(-h, h)], tip_xy, 1e-12),
              "tip triangle = T squeezed by ½ and ¼")
        check(all(abs(q[1] - q[0] ** 2) < 1e-12
                  for q in slide([par(x) for x in (-1.2, -0.4, 0.0)], 0.6)),
              "the slide keeps the parabola")
        check(abs(abs(area(slide(sub_xy, 1.0))) - abs(area(sub_xy))) < 1e-12,
              "the slide keeps area")

        # ---- figure
        curve = VMobject(stroke_color=WHITE, stroke_width=4).set_points_smoothly(
            [S(par(x)) for x in np.linspace(-1.62, 1.62, 81)])
        chord = Line(S(par(-h)), S(par(h)), color=YELLOW_B, stroke_width=5)
        segfill = mk([S(par(x)) for x in np.linspace(-h, h, 121)], GREY_B, 0.14,
                     stroke_width=0)
        cols = [YELLOW_E, TEAL_D, BLUE_D, GREEN_D, PURPLE_B]
        polys = [[mk([S(p) for p in t], cols[g], 0.85,
                     stroke_width=2 if g < 2 else 1.2) for t in gens[g]]
                 for g in range(5)]
        tanT = DashedLine(S((-0.75, 0)), S((0.75, 0)), color=YELLOW_B,
                          stroke_width=2.5, dash_length=0.1)

        def tangent_at(m, half=0.42):
            d = _unit(np.array([1.0, 2 * m, 0.0]))
            c0 = S(par(m))
            return DashedLine(c0 - d * half * k, c0 + d * half * k,
                              color=TEAL_A, stroke_width=3.5, dash_length=0.09)

        lT = tag("T", 34, BLACK).move_to(S((0.0, 1.5)))
        led_x = 3.9

        def led_line(i, sign, term, col):
            y = 2.35 - 0.72 * i
            g = VGroup(tag(term, 32, col).move_to([led_x + 0.55, y, 0.0],
                                                  aligned_edge=LEFT))
            if sign:
                g.add(tag(sign, 32, col).move_to([led_x + 0.2, y, 0.0]))
            return g

        ledger = [led_line(0, "", "T", YELLOW_E),
                  led_line(1, "+", "T/4", TEAL_B),
                  led_line(2, "+", "T/16", BLUE_B),
                  led_line(3, "+", "T/64", GREEN_B),
                  led_line(4, "+", "…", PURPLE_A)]

        self.play(Create(curve), run_time=1.2)
        self.play(Create(chord), FadeIn(segfill), run_time=0.8)
        self.play(Create(tanT), run_time=0.6)
        self.play(FadeIn(polys[0][0]), FadeIn(lT), run_time=0.8)
        self.play(FadeIn(ledger[0]), FadeOut(tanT), run_time=0.6)
        tg = [tangent_at(-h / 2), tangent_at(h / 2)]
        self.play(*[FadeIn(p) for p in polys[1]], *[Create(t) for t in tg],
                  run_time=0.9)
        self.play(*[FadeOut(t) for t in tg], run_time=0.4)
        self.hold(0.4)

        # ---- one leftover segment slides along the parabola to the tip
        dim = [polys[0][0], *polys[1]]
        self.play(*[p.animate.set_fill(opacity=0.22) for p in dim],
                  lT.animate.set_opacity(0.3), run_time=0.5)
        sub = mk([S(p) for p in sub_xy], TEAL_D, 0.3, stroke_width=0)
        t2 = mk([S(p) for p in t2_xy], TEAL_D, 0.95)

        def upd_sub(m_, a):
            m_.become(mk([S(p) for p in slide(sub_xy, a)], TEAL_D, 0.3,
                         stroke_width=0))

        def upd_t2(m_, a):
            m_.become(mk([S(p) for p in slide(t2_xy, a)], TEAL_D, 0.95))

        self.add(sub, t2)
        self.play(UpdateFromAlphaFunc(sub, upd_sub),
                  UpdateFromAlphaFunc(t2, upd_t2), run_time=2.4)
        check(close(t2.get_vertices(), [S(p) for p in tip_xy], 1e-9),
              "landed on the tip")

        # ---- and T, squeezed ×½ across and ×¼ up, is exactly that triangle
        Tc = Polygon(*[S(p) for p in tri(-h, h)], stroke_color=YELLOW_B,
                     stroke_width=5)
        self.play(Create(Tc), run_time=0.6)
        f2 = tag("×½", 30, YELLOW_B).move_to([1.55, 0.2, 0.0])
        self.play(FadeIn(f2), Tc.animate.stretch(0.5, 0, about_point=S((0, 0))),
                  run_time=1.2)
        f4 = tag("×¼", 30, YELLOW_B).next_to(f2, DOWN, buff=0.25)
        self.play(FadeIn(f4), Tc.animate.stretch(0.25, 1, about_point=S((0, 0))),
                  run_time=1.2)
        check(close(Tc.get_vertices(), [S(p) for p in tip_xy], 1e-9),
              "T squeezed lands on the tip triangle")
        l8 = tag("T/8", 30, WHITE).move_to(
            S(tip_xy[1]) + RIGHT * 0.95 + DOWN * 0.05)
        self.play(FadeIn(l8), run_time=0.6)
        self.hold(0.8)
        self.play(FadeOut(VGroup(sub, t2, Tc, f2, f4, l8)),
                  *[p.animate.set_fill(opacity=0.85) for p in dim],
                  lT.animate.set_opacity(1.0), run_time=0.7)
        self.play(FadeIn(ledger[1]), run_time=0.6)

        # ---- the same at every stage: each generation a quarter of the last
        for g in (2, 3, 4):
            self.play(*[FadeIn(p) for p in polys[g]],
                      FadeIn(ledger[g]), run_time=0.8 if g < 4 else 0.6)
        total = led_line(5, "=", "(4/3)·T", YELLOW_B).shift(DOWN * 0.1)
        self.play(FadeIn(total), run_time=0.7)
        self.play(Write(caption("A  =  T(1 + ¼ + 1/16 + …)  =  (4/3)·T", 34)))
        self.hold(2.2)
