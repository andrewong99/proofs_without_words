# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes.py — the manim (2D) proofs without words.

Rendered by subprocess:
    python -m manim render -qm pww_scenes.py A2_ZhaoShuang

Deliberate constraint: NO MathTex / Tex / Brace / DecimalNumber anywhere in
this file. Those mobjects require a LaTeX installation. Everything here uses
`Text` (Pango) with Unicode superscripts, so manim needs no TeX at all.
Captions are kept symbolic — these are proofs
*without words*; the prose lives in the GUI panel.
"""

from manim import *
import numpy as np

# ----------------------------------------------------------------- helpers

FILL = 0.78
EDGE = dict(stroke_color=WHITE, stroke_width=2)


def caption(s, size=30, color=WHITE):
    t = Text(s, font_size=size, color=color)
    t.to_edge(DOWN, buff=0.3)
    return t


def tag(s, size=26, color=WHITE):
    return Text(s, font_size=size, color=color)


def mk(pts, color, op=FILL, **kw):
    """Polygon from a list of 3D points."""
    o = dict(EDGE)
    o.update(kw)
    return Polygon(*pts, fill_color=color, fill_opacity=op, **o)


class _Board(Scene):
    """Common intro/outro pacing."""

    def hold(self, t=1.4):
        self.wait(t)


# ============================================================ A. PYTHAGORAS

class A1_TwoArrangements(_Board):
    """One (a+b) square, dissected two ways: the same four triangles leave
    c² in one arrangement and a² + b² in the other. (Not Zhao Shuang's
    figure — that is A2.)"""

    def construct(self):
        a, b, k = 3.0, 2.0, 1.02
        s = a + b

        def P(x, y):
            return np.array([(x - s / 2) * k, (y - s / 2) * k - 0.4, 0.0])

        def poly(pts, color, op=FILL):
            return mk([P(*p) for p in pts], color, op)

        outer = Polygon(P(0, 0), P(s, 0), P(s, s), P(0, s),
                        stroke_color=WHITE, stroke_width=5)

        cfg1 = [[(0, 0), (a, 0), (0, b)],
                [(s, 0), (s, a), (a, 0)],
                [(s, s), (b, s), (s, a)],
                [(0, s), (0, b), (b, s)]]
        cfg2 = [[(a, 0), (s, 0), (s, a)],
                [(a, 0), (s, a), (a, a)],
                [(0, a), (a, a), (a, s)],
                [(0, a), (a, s), (0, s)]]
        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D]

        tris = [poly(t, c) for t, c in zip(cfg1, cols)]
        csq = poly([(a, 0), (s, a), (b, s), (0, b)], YELLOW_E, 0.9)
        c_lbl = tag("c²", 34).move_to(P(s / 2, s / 2))

        self.play(Create(outer), run_time=1.2)
        self.play(*[FadeIn(t) for t in tris], run_time=1.0)
        self.play(FadeIn(csq), FadeIn(c_lbl))
        self.hold()

        tgt = [poly(t, c) for t, c in zip(cfg2, cols)]
        asq = poly([(0, 0), (a, 0), (a, a), (0, a)], YELLOW_E, 0.9)
        bsq = poly([(a, a), (s, a), (s, s), (a, s)], ORANGE, 0.9)
        a_lbl = tag("a²", 34).move_to(P(a / 2, a / 2))
        b_lbl = tag("b²", 30).move_to(P((a + s) / 2, (a + s) / 2))

        self.play(FadeOut(csq), FadeOut(c_lbl),
                  *[Transform(t, g) for t, g in zip(tris, tgt)],
                  run_time=2.6)
        self.play(FadeIn(asq), FadeIn(bsq), FadeIn(a_lbl), FadeIn(b_lbl))
        self.play(Write(caption("a² + b² = c²", 36, YELLOW_B)))
        self.hold(2.0)


class A2_ZhaoShuang(_Board):
    """赵爽弦图: four right triangles (朱实, vermilion) around a central square
    of side b−a (中黄实, yellow) make the square on the hypotenuse (弦实).
    Bhāskara II drew the same figure some nine centuries later."""

    def construct(self):
        a, b, c, k = 3.0, 4.0, 5.0, 0.9
        ctr = np.array([c / 2, c / 2, 0.0])

        def P(v):
            return np.array([(v[0] - c / 2) * k, (v[1] - c / 2) * k - 0.4, 0.0])

        def rot(v, n):
            """Rotate point v about the square's centre by n*90 degrees."""
            x, y = v[0] - c / 2, v[1] - c / 2
            for _ in range(n):
                x, y = -y, x
            return (x + c / 2, y + c / 2)

        base = [(0.0, 0.0), (c, 0.0), (b * b / c, a * b / c)]
        # Zhao Shuang's own colour terms: 朱实 (vermilion) for the triangles,
        # two shades so neighbours stay distinct; 中黄实 (yellow) centre
        cols = ["#E0452C", "#B5331F", "#E0452C", "#B5331F"]
        tris, apexes = [], []
        for n in range(4):
            pts = [rot(p, n) for p in base]
            apexes.append(pts[2])
            tris.append(mk([P(p) for p in pts], cols[n], 0.95))

        square = Polygon(P((0, 0)), P((c, 0)), P((c, c)), P((0, c)),
                         stroke_color=YELLOW_B, stroke_width=5)
        hole = mk([P(p) for p in apexes], YELLOW_E, 0.95)

        self.play(Create(square), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.4) for t in tris],
                              lag_ratio=0.25), run_time=2.4)
        self.hold(0.8)
        self.play(FadeIn(hole), Flash(P(ctr[:2]), color=YELLOW, line_length=0.2))
        self.play(FadeIn(tag("(b−a)²", 26).next_to(hole, RIGHT, buff=0.45)))
        self.play(Write(caption("c² = (b−a)² + 4·½ab  =  a² + b²", 32, YELLOW_B)))
        self.hold(2.2)


class A3_EuclidWindmill(_Board):
    """Euclid I.47. Each square on a leg becomes the matching rectangle of the
    hypotenuse square by shear -> quarter-turn -> shear.

    Each shear keeps one side fixed and slides the opposite side along its own
    line, so base and height, hence area, never change (I.35). The quarter
    turn is Euclid's SAS congruence (I.4) made visible. The two rectangles are
    cut off by the altitude from the right angle: widths b²/c and a²/c.
    """

    def construct(self):
        a, b = 3.0, 4.0                         # legs: a = |BC|, b = |AC|
        c = float(np.hypot(a, b))
        A = np.array([0.0, 0.0])                # hypotenuse AB on the x-axis
        B = np.array([c, 0.0])
        C = np.array([b * b / c, a * b / c])    # the right angle
        H = np.array([b * b / c, 0.0])          # foot of the altitude from C
        D = np.array([0.0, -c])                 # down one side of the c-square
        n = np.array([-a * b / c, b * b / c])   # outward edge of square on AC
        m = np.array([a * b / c, a * a / c])    # outward edge of square on BC

        xmin, xmax = -a * b / c, c + a * b / c
        ymin, ymax = -c, a * b / c + b * b / c
        k = 6.4 / (ymax - ymin)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

        def P(v):
            return np.array([(v[0] - cx) * k, (v[1] - cy) * k + 0.38, 0.0])

        def quad(pts, color):
            return Polygon(*[P(p) for p in pts], fill_color=color,
                           fill_opacity=FILL, stroke_color=WHITE,
                           stroke_width=2)

        def guide(p, q, color, extend=0.22):
            p, q = np.asarray(p, float), np.asarray(q, float)
            d = q - p
            return DashedLine(P(p - extend * d), P(q + extend * d),
                              color=color, stroke_width=2, dash_length=0.09)

        COL_B, COL_A = TEAL_D, ORANGE
        tri = Polygon(P(A), P(B), P(C), stroke_color=WHITE, stroke_width=3)
        csq = Polygon(P(A), P(B), P(B + D), P(A + D),
                      stroke_color=YELLOW_B, stroke_width=4)
        ghost_b = Polygon(P(A), P(A + n), P(C + n), P(C),
                          stroke_color=COL_B, stroke_width=2)
        ghost_a = Polygon(P(B), P(B + m), P(C + m), P(C),
                          stroke_color=COL_A, stroke_width=2)
        # Vertex orders are chosen so that every shear below is a pure slide
        # of two vertices along the line of the side they lie on.
        sq_b = quad([A, A + n, C + n, C], COL_B)
        sq_a = quad([B, B + m, C + m, C], COL_A)
        lab_b = tag("b²", 30).move_to(P((A + C + n) / 2))
        lab_a = tag("a²", 28).move_to(P((B + C + m) / 2))
        altitude = guide(C, H + D, GREY_B, extend=0.0)

        self.play(Create(tri), run_time=0.9)
        self.play(Create(csq), Create(ghost_b), Create(ghost_a), run_time=1.2)
        self.play(FadeIn(sq_b), FadeIn(sq_a), FadeIn(lab_b), FadeIn(lab_a))
        self.play(Create(altitude), run_time=0.9)      # Euclid's line AL
        self.hold(0.8)
        self.play(FadeOut(lab_b), FadeOut(lab_a), run_time=0.4)

        def shear(mob, target_pts, color, base, opposite):
            """Fixed base drawn bold; both parallels dashed while it slides."""
            edge = Line(P(base[0]), P(base[1]), color=color, stroke_width=8)
            g1 = guide(base[0], base[1], color)
            g2 = guide(opposite[0], opposite[1], color)
            self.play(Create(edge), Create(g1), Create(g2), run_time=0.6)
            self.play(Transform(mob, quad(target_pts, color)), run_time=1.8)
            self.play(FadeOut(edge), FadeOut(g1), FadeOut(g2), run_time=0.4)

        def quarter_turn(mob, pivot, start_dir, sign, color):
            start = float(np.arctan2(start_dir[1], start_dir[0]))
            arc = Arc(radius=0.55, start_angle=start, angle=sign * PI / 2,
                      arc_center=P(pivot), color=color, stroke_width=5)
            mid = start + sign * PI / 4
            deg = tag("90°", 24, color).move_to(
                P(pivot) + 0.95 * np.array([np.cos(mid), np.sin(mid), 0.0]))
            self.play(Create(arc), FadeIn(deg), run_time=0.5)
            self.play(Rotate(mob, angle=sign * PI / 2, about_point=P(pivot)),
                      run_time=1.6)
            self.play(FadeOut(arc), FadeOut(deg), run_time=0.4)

        # ---- b²: the square on AC
        self.bring_to_front(sq_b)
        shear(sq_b, [A, A + n, B + n, B], COL_B,
              base=(A, A + n), opposite=(C + n, B))
        quarter_turn(sq_b, A, n, -1, COL_B)            # now (A, C, C+D, A+D)
        shear(sq_b, [A, H, H + D, A + D], COL_B,
              base=(A, A + D), opposite=(C, H + D))
        self.play(FadeIn(tag("b²", 30).move_to(P(((A + H) + D) / 2))))

        # ---- a²: the square on BC
        self.bring_to_front(sq_a)
        shear(sq_a, [B, B + m, A + m, A], COL_A,
              base=(B, B + m), opposite=(C + m, A))
        quarter_turn(sq_a, B, m, +1, COL_A)            # now (B, C, C+D, B+D)
        shear(sq_a, [B, H, H + D, B + D], COL_A,
              base=(B, B + D), opposite=(C, H + D))
        self.play(FadeIn(tag("a²", 28).move_to(P(((B + H) + D) / 2))))

        self.play(Write(caption("a² + b²  =  c²", 36, YELLOW_B)))
        self.hold(2.4)


class A5_Garfield(_Board):
    """Garfield's trapezoid, its area counted two ways."""

    def construct(self):
        a, b, k = 3.0, 2.0, 1.05
        s = a + b

        def P(x, y):
            return np.array([(x - s / 2) * k - 0.2, (y - s / 2) * k - 0.4, 0.0])

        trap = Polygon(P(0, 0), P(s, 0), P(s, a), P(0, b),
                       stroke_color=WHITE, stroke_width=5)
        t1 = mk([P(0, 0), P(a, 0), P(0, b)], BLUE_D)
        t2 = mk([P(a, 0), P(s, 0), P(s, a)], BLUE_D)
        t3 = mk([P(0, b), P(a, 0), P(s, a)], YELLOW_E, 0.9)
        ra = RightAngle(Line(P(a, 0), P(0, b)), Line(P(a, 0), P(s, a)),
                        length=0.28, color=RED)

        self.play(Create(trap), run_time=1.2)
        self.play(FadeIn(t1), FadeIn(t2))
        self.play(FadeIn(t3), Create(ra))
        self.play(FadeIn(tag("½ab", 24).move_to(P(a / 3, b / 3))),
                  FadeIn(tag("½ab", 24).move_to(P(a + 2 * b / 3, a / 3))),
                  FadeIn(tag("½c²", 26).move_to(P(s / 2, a * 0.72))))
        self.hold()
        self.play(Write(caption("½(a+b)²  =  ½ab + ½ab + ½c²", 32, YELLOW_B)))
        self.hold(2.2)


class A6_SimilarTriangles(_Board):
    """The altitude cuts the triangle into two copies of itself."""

    def construct(self):
        a, b = 3.0, 4.0
        c = np.hypot(a, b)
        k = 1.15
        off = np.array([-2.2, -1.9, 0.0])

        def P(x, y):
            return np.array([x * k, y * k, 0.0]) + off

        C, A, B = (0.0, 0.0), (0.0, a), (b, 0.0)
        # foot of the altitude from C onto AB
        t = a * a / (c * c)
        H = (A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]))

        tri = Polygon(P(*C), P(*A), P(*B), stroke_color=WHITE, stroke_width=4)
        alt = DashedLine(P(*C), P(*H), color=GREY_B)
        left = mk([P(*A), P(*H), P(*C)], BLUE_D)
        right = mk([P(*H), P(*B), P(*C)], TEAL_D)
        ra = RightAngle(Line(P(*C), P(*A)), Line(P(*C), P(*B)),
                        length=0.3, color=RED)

        self.play(Create(tri), Create(ra), run_time=1.2)
        self.play(Create(alt))
        self.play(FadeIn(left), FadeIn(right))
        self.hold(0.9)

        # the hypotenuse, split into a²/c and b²/c
        y0 = 2.55
        x0, w = -3.6, 7.2
        split = x0 + w * (a * a / (c * c))
        bar_l = Line([x0, y0, 0], [split, y0, 0], color=BLUE_D, stroke_width=16)
        bar_r = Line([split, y0, 0], [x0 + w, y0, 0], color=TEAL_D, stroke_width=16)
        l1 = tag("a²/c", 26, BLUE_B).next_to(bar_l, UP, buff=0.18)
        l2 = tag("b²/c", 26, TEAL_B).next_to(bar_r, UP, buff=0.18)
        l3 = tag("c", 28).next_to(VGroup(bar_l, bar_r), DOWN, buff=0.18)

        self.play(TransformFromCopy(left, bar_l), TransformFromCopy(right, bar_r),
                  run_time=1.8)
        self.play(FadeIn(l1), FadeIn(l2), FadeIn(l3))
        self.play(Write(caption("a²/c + b²/c = c   ⟹   a² + b² = c²", 32, YELLOW_B)))
        self.hold(2.2)


# ================================================== B. SUMS & FIGURATE NUMBERS

class B1_Triangular(_Board):
    """Two staircases interlock into an n by (n+1) rectangle."""

    def construct(self):
        n, u = 6, 0.52

        def cell(i, j, color):
            sq = Square(side_length=u, fill_color=color, fill_opacity=FILL,
                        stroke_color=WHITE, stroke_width=1.5)
            sq.move_to(np.array([(i + 0.5) * u, (j + 0.5) * u, 0.0]))
            return sq

        # column i of the staircase holds cells j = 0..i  (i+1 cells)
        stair = VGroup(*[cell(i, j, BLUE_D)
                         for i in range(n) for j in range(i + 1)])
        # its complement in the n x (n+1) grid is the same staircase turned
        # through 180 degrees: column i holds j = i+1..n  (n-i cells)
        comp = VGroup(*[cell(i, j, TEAL_D)
                        for i in range(n) for j in range(i + 1, n + 1)])

        grp = VGroup(stair, comp)
        grp.move_to(DOWN * 0.35)
        comp.save_state()
        comp.shift(UP * 1.15 + RIGHT * 1.05)

        self.play(LaggedStart(*[FadeIn(s) for s in stair], lag_ratio=0.04),
                  run_time=2.0)
        self.hold(0.5)
        self.play(FadeIn(comp), run_time=0.8)
        self.play(Restore(comp), run_time=1.8)
        frame = SurroundingRectangle(VGroup(stair, comp), buff=0,
                                     color=YELLOW_B, stroke_width=5)
        self.play(Create(frame))
        self.play(FadeIn(tag("n", 28, YELLOW_B).next_to(frame, DOWN, buff=0.2)),
                  FadeIn(tag("n+1", 28, YELLOW_B).next_to(frame, RIGHT, buff=0.2)))
        self.play(Write(caption("1 + 2 + … + n  =  n(n+1)/2", 32, YELLOW_B)))
        self.hold(2.2)


class B2_OddSquares(_Board):
    """Each odd number is an L-shaped shell of a growing square."""

    def construct(self):
        n, u = 6, 0.72
        cols = [BLUE_D, TEAL_D, GREEN_D, YELLOW_E, ORANGE, RED_D]
        origin = np.array([-n * u / 2, -n * u / 2 - 0.4, 0.0])

        def cell(i, j, color):
            sq = Square(side_length=u, fill_color=color, fill_opacity=FILL,
                        stroke_color=WHITE, stroke_width=1.5)
            sq.move_to(origin + np.array([(i + 0.5) * u, (j + 0.5) * u, 0.0]))
            return sq

        shells = []
        for k in range(n):
            cells = [cell(k, j, cols[k]) for j in range(k)] + \
                    [cell(i, k, cols[k]) for i in range(k + 1)]
            shells.append(VGroup(*cells))

        run = tag("", 30)
        for k, sh in enumerate(shells):
            lbl = tag(f"{2*k+1}", 26, cols[k])
            lbl.next_to(sh, RIGHT, buff=0.25) if k else lbl.next_to(sh, RIGHT, buff=0.25)
            self.play(FadeIn(sh, shift=UP * 0.2), run_time=0.55)
        self.hold(0.7)
        frame = SurroundingRectangle(VGroup(*shells), buff=0,
                                     color=YELLOW_B, stroke_width=5)
        self.play(Create(frame))
        self.play(FadeIn(tag("n", 30, YELLOW_B).next_to(frame, DOWN, buff=0.2)),
                  FadeIn(tag("n", 30, YELLOW_B).next_to(frame, RIGHT, buff=0.2)))
        self.play(Write(caption("1 + 3 + 5 + … + (2n−1)  =  n²", 32, YELLOW_B)))
        self.hold(2.2)


class B4_Nicomachus(_Board):
    """Gnomon k of the square on T(n) has area k³."""

    def construct(self):
        n = 5
        T = [0] + [k * (k + 1) // 2 for k in range(1, n + 1)]
        side = T[n]
        u = 5.4 / side
        origin = np.array([-side * u / 2, -side * u / 2 - 0.35, 0.0])
        cols = [BLUE_D, TEAL_D, GREEN_D, YELLOW_E, ORANGE]

        def rect(x, y, w, h, color):
            r = Rectangle(width=w * u, height=h * u,
                          fill_color=color, fill_opacity=FILL,
                          stroke_color=WHITE, stroke_width=1.2)
            r.move_to(origin + np.array([(x + w / 2) * u, (y + h / 2) * u, 0.0]))
            return r

        outer = Rectangle(width=side * u, height=side * u,
                          stroke_color=YELLOW_B, stroke_width=5)
        outer.move_to(origin + np.array([side * u / 2, side * u / 2, 0.0]))
        self.play(Create(outer), run_time=1.2)

        for k in range(1, n + 1):
            arm_h = rect(0, T[k - 1], T[k], k, cols[k - 1])      # horizontal arm
            arm_v = rect(T[k - 1], 0, k, T[k - 1], cols[k - 1])  # vertical arm
            g = VGroup(arm_h, arm_v)
            # subdivision into k x k blocks
            lines = VGroup()
            for m in range(1, (T[k] + k - 1) // k + 1):
                x = m * k
                if x < T[k]:
                    lines.add(Line(origin + np.array([x * u, T[k - 1] * u, 0]),
                                   origin + np.array([x * u, T[k] * u, 0]),
                                   stroke_color=WHITE, stroke_width=1.2))
                y = m * k
                if y < T[k - 1]:
                    lines.add(Line(origin + np.array([T[k - 1] * u, y * u, 0]),
                                   origin + np.array([T[k] * u, y * u, 0]),
                                   stroke_color=WHITE, stroke_width=1.2))
            self.play(FadeIn(g), Create(lines), run_time=0.7)
            self.play(FadeIn(tag(f"{k}³", 24).move_to(
                origin + np.array([(T[k] - k / 2) * u, (T[k] - k / 2) * u, 0]))),
                run_time=0.3)

        self.play(FadeIn(tag("side = 1+2+…+n", 24, YELLOW_B)
                         .next_to(outer, UP, buff=0.22)))
        self.play(Write(caption("1³ + 2³ + … + n³  =  (1+2+…+n)²", 32, YELLOW_B)))
        self.hold(2.2)


class B5_GeometricHalves(_Board):
    """Halve the square forever; nothing is left over."""

    def construct(self):
        S = 4.6
        sq = Square(side_length=S, stroke_color=YELLOW_B, stroke_width=5)
        sq.shift(DOWN * 0.35)
        self.play(Create(sq), run_time=1.0)

        cols = [BLUE_D, TEAL_D, GREEN_D, YELLOW_E, ORANGE, RED_D, PURPLE_B, MAROON_B]
        x0 = sq.get_corner(DL)[0]
        y0 = sq.get_corner(DL)[1]
        w, h = S, S
        horiz = True
        parts = VGroup()
        for i in range(9):
            if horiz:
                w2, h2 = w / 2, h
                piece = Rectangle(width=w2, height=h2, fill_color=cols[i % len(cols)],
                                  fill_opacity=FILL, stroke_color=WHITE, stroke_width=1.5)
                piece.move_to([x0 + w2 / 2, y0 + h2 / 2, 0])
                x0 += w2
                w = w2
            else:
                w2, h2 = w, h / 2
                piece = Rectangle(width=w2, height=h2, fill_color=cols[i % len(cols)],
                                  fill_opacity=FILL, stroke_color=WHITE, stroke_width=1.5)
                piece.move_to([x0 + w2 / 2, y0 + h2 / 2, 0])
                y0 += h2
                h = h2
            horiz = not horiz
            parts.add(piece)
            lab = tag(["1/2", "1/4", "1/8", "1/16"][i] if i < 4 else "", 22)
            if i < 4:
                lab.move_to(piece)
                self.play(FadeIn(piece), FadeIn(lab), run_time=0.55)
            else:
                self.play(FadeIn(piece), run_time=0.3)

        self.play(Write(caption("½ + ¼ + ⅛ + …  =  1", 34, YELLOW_B)))
        self.hold(2.2)


class B6_ThreeLs(_Board):
    """Three congruent shapes share the square equally at every stage."""

    def construct(self):
        S = 4.8
        sq = Square(side_length=S, stroke_color=YELLOW_B, stroke_width=5)
        sq.shift(DOWN * 0.3)
        self.play(Create(sq), run_time=1.0)

        cols = [BLUE_D, TEAL_D, ORANGE]
        cx, cy = sq.get_center()[0], sq.get_center()[1]
        w = S
        for depth in range(6):
            half = w / 2
            # three of the four quadrants get the three colours;
            # recursion continues into the fourth (lower-left)
            spots = [(cx + half / 2, cy + half / 2),
                     (cx - half / 2, cy + half / 2),
                     (cx + half / 2, cy - half / 2)]
            group = VGroup()
            for (px, py), col in zip(spots, cols):
                r = Square(side_length=half, fill_color=col, fill_opacity=FILL,
                           stroke_color=WHITE, stroke_width=1.2)
                r.move_to([px, py, 0])
                group.add(r)
            self.play(FadeIn(group), run_time=0.6 if depth < 3 else 0.3)
            cx, cy = cx - half / 2, cy - half / 2
            w = half

        self.play(Write(caption("¼ + 1/16 + 1/64 + …  =  ⅓", 34, YELLOW_B)))
        self.hold(2.2)


class B9_FibonacciSquares(_Board):
    """Fibonacci squares spiral into an F(n) by F(n+1) rectangle."""

    def construct(self):
        F = [1, 1, 2, 3, 5, 8]
        u = 0.52
        cols = [BLUE_D, TEAL_D, GREEN_D, YELLOW_E, ORANGE, RED_D]
        # place squares in the classic spiral: right, up, left, down, ...
        x, y = 0.0, 0.0
        rects = []
        # anchor positions computed by walking the spiral
        pos = [(0, 0), (1, 0), (0, 1), (-2, -1), (-2, -4), (5, -4)]
        # explicit, verified layout for F = 1,1,2,3,5,8
        layout = [(0, 0, 1), (1, 0, 1), (0, 1, 2), (-3, 0, 3),
                  (-3, 3, 5), (2, -5, 8)]
        # corrected spiral: build it programmatically instead
        layout = []
        x0, y0, x1, y1 = 0, 0, 0, 0  # current bounding box
        for i, f in enumerate(F):
            if i == 0:
                x0, y0, x1, y1 = 0, 0, f, f
                layout.append((0, 0, f))
                continue
            side = i % 4
            if side == 1:      # attach to the right
                layout.append((x1, y0, f)); x1 += f
            elif side == 2:    # attach on top
                layout.append((x0, y1, f)); y1 += f
            elif side == 3:    # attach on the left
                layout.append((x0 - f, y0, f)); x0 -= f
            else:              # attach below
                layout.append((x0, y0 - f, f)); y0 -= f

        grp = VGroup()
        for (lx, ly, f), col in zip(layout, cols):
            r = Rectangle(width=f * u, height=f * u, fill_color=col,
                          fill_opacity=FILL, stroke_color=WHITE, stroke_width=1.8)
            r.move_to(np.array([(lx + f / 2) * u, (ly + f / 2) * u, 0.0]))
            lab = tag(f"{f}²", 22).move_to(r)
            grp.add(VGroup(r, lab))

        grp.move_to(ORIGIN).shift(DOWN * 0.3)
        for piece in grp:
            self.play(FadeIn(piece), run_time=0.5)
        frame = SurroundingRectangle(grp, buff=0, color=YELLOW_B, stroke_width=5)
        self.play(Create(frame))
        # bottom side is F(n+1) = 13, right side is F(n) = 8
        self.play(FadeIn(tag("Fₙ₊₁", 26, YELLOW_B).next_to(frame, DOWN, buff=0.2)),
                  FadeIn(tag("Fₙ", 26, YELLOW_B).next_to(frame, RIGHT, buff=0.2)))
        self.play(Write(caption("F₁² + F₂² + … + Fₙ²  =  Fₙ · Fₙ₊₁", 32, YELLOW_B)))
        self.hold(2.2)


# ============================================== C. ALGEBRAIC IDENTITIES (2D)

class C1_SquareOfSum(_Board):
    """A square of side a+b splits into four tiles."""

    def construct(self):
        a, b, k = 3.0, 2.0, 1.05
        s = a + b

        def R(x, y, w, h, color, label):
            r = Rectangle(width=w * k, height=h * k, fill_color=color,
                          fill_opacity=FILL, stroke_color=WHITE, stroke_width=2)
            r.move_to(np.array([(x + w / 2 - s / 2) * k,
                                (y + h / 2 - s / 2) * k - 0.35, 0.0]))
            return VGroup(r, tag(label, 30).move_to(r))

        aa = R(0, 0, a, a, BLUE_D, "a²")
        ab1 = R(a, 0, b, a, TEAL_D, "ab")
        ab2 = R(0, a, a, b, TEAL_D, "ab")
        bb = R(a, a, b, b, ORANGE, "b²")
        outer = Square(side_length=s * k, stroke_color=YELLOW_B, stroke_width=5)
        outer.move_to([0, -0.35, 0])

        self.play(Create(outer), run_time=1.0)
        self.play(FadeIn(aa)); self.play(FadeIn(ab1), FadeIn(ab2)); self.play(FadeIn(bb))
        self.play(FadeIn(tag("a+b", 28, YELLOW_B).next_to(outer, DOWN, buff=0.2)),
                  FadeIn(tag("a+b", 28, YELLOW_B).next_to(outer, RIGHT, buff=0.2)))
        self.play(Write(caption("(a+b)²  =  a² + 2ab + b²", 34, YELLOW_B)))
        self.hold(2.2)


class C3_DifferenceOfSquares(_Board):
    """An L-shaped gnomon straightens into a rectangle."""

    def construct(self):
        a, b, k = 4.0, 1.6, 1.0

        def P(x, y, dx=0.0, dy=0.0):
            return np.array([(x - a / 2) * k + dx, (y - a / 2) * k + dy - 0.3, 0.0])

        big = Square(side_length=a * k, stroke_color=YELLOW_B, stroke_width=5)
        big.move_to(P(a / 2, a / 2))
        hole = Rectangle(width=b * k, height=b * k, stroke_color=RED_D,
                         stroke_width=4, fill_color=BLACK, fill_opacity=1.0)
        hole.move_to(P(a - b / 2, a - b / 2))

        # the gnomon, cut into two rectangles
        r1 = mk([P(0, 0), P(a, 0), P(a, a - b), P(0, a - b)], BLUE_D)
        r2 = mk([P(0, a - b), P(a - b, a - b), P(a - b, a), P(0, a)], TEAL_D)

        self.play(Create(big), run_time=1.0)
        self.play(FadeIn(r1), FadeIn(r2))
        b_lbl = tag("b²", 26, RED_B).move_to(hole)
        self.play(FadeIn(hole), FadeIn(b_lbl))
        self.hold(1.0)

        # slide r2 to the right of r1: rectangle (a+b) by (a-b)
        target1 = mk([P(0, 0), P(a, 0), P(a, a - b), P(0, a - b)], BLUE_D)
        target2 = mk([P(a, 0), P(a + b, 0), P(a + b, a - b), P(a, a - b)], TEAL_D)
        grp = VGroup(target1, target2)

        self.play(FadeOut(big), FadeOut(hole), FadeOut(b_lbl),
                  Transform(r1, target1), Transform(r2, target2), run_time=2.2)
        self.play(VGroup(r1, r2).animate.move_to([0, -0.5, 0]))
        joined = VGroup(r1, r2)
        frame = SurroundingRectangle(joined, buff=0, color=YELLOW_B, stroke_width=5)
        self.play(Create(frame))
        self.play(FadeIn(tag("a+b", 26, YELLOW_B).next_to(frame, DOWN, buff=0.2)),
                  FadeIn(tag("a−b", 26, YELLOW_B).next_to(frame, RIGHT, buff=0.2)))
        self.play(Write(caption("a² − b²  =  (a+b)(a−b)", 34, YELLOW_B)))
        self.hold(2.2)


# ==================================================== D. MEANS & INEQUALITIES

class D1_AMGM(_Board):
    """A radius is never shorter than the altitude standing under it."""

    def construct(self):
        L = 6.4
        y0 = -1.6
        t = ValueTracker(0.30)

        A = np.array([-L / 2, y0, 0.0])
        B = np.array([L / 2, y0, 0.0])
        O = np.array([0.0, y0, 0.0])
        R = L / 2

        arc = Arc(radius=R, start_angle=0, angle=PI, arc_center=O,
                  color=GREY_B, stroke_width=3)
        diam = Line(A, B, color=GREY_B)

        def Ppt():
            f = t.get_value()
            return np.array([A[0] + f * L, y0, 0.0])

        def Qpt():
            p = Ppt()
            a = p[0] - A[0]
            b = B[0] - p[0]
            return np.array([p[0], y0 + np.sqrt(a * b), 0.0])

        seg_a = always_redraw(lambda: Line(A, Ppt(), color=BLUE_D, stroke_width=10))
        seg_b = always_redraw(lambda: Line(Ppt(), B, color=TEAL_D, stroke_width=10))
        gm = always_redraw(lambda: Line(Ppt(), Qpt(), color=GREEN_B, stroke_width=6))
        am = always_redraw(lambda: Line(O, np.array([O[0], y0 + R, 0.0]),
                                        color=YELLOW_B, stroke_width=6))
        rad = always_redraw(lambda: DashedLine(O, Qpt(), color=GREY_A, stroke_width=3))
        dotP = always_redraw(lambda: Dot(Ppt(), color=WHITE))
        dotQ = always_redraw(lambda: Dot(Qpt(), color=GREEN_B))

        la = always_redraw(lambda: tag("a", 26, BLUE_B).move_to(
            (A + Ppt()) / 2 + DOWN * 0.35))
        lb = always_redraw(lambda: tag("b", 26, TEAL_B).move_to(
            (Ppt() + B) / 2 + DOWN * 0.35))
        lg = always_redraw(lambda: tag("√(ab)", 24, GREEN_B).next_to(
            gm, LEFT, buff=0.15))
        lm = tag("(a+b)/2", 24, YELLOW_B).move_to(
            np.array([O[0] + 1.05, y0 + R / 2, 0.0]))

        self.play(Create(arc), Create(diam), run_time=1.2)
        self.play(FadeIn(seg_a), FadeIn(seg_b), FadeIn(dotP))
        self.play(Create(gm), FadeIn(dotQ), Create(rad))
        self.play(Create(am), FadeIn(lm), FadeIn(la), FadeIn(lb), FadeIn(lg))
        self.play(Write(caption("(a+b)/2  ≥  √(ab)", 34, YELLOW_B)))
        self.play(t.animate.set_value(0.5), run_time=2.4)
        self.play(t.animate.set_value(0.08), run_time=2.4)
        self.play(t.animate.set_value(0.30), run_time=1.6)
        self.hold(1.4)


class D2_MeansChain(_Board):
    """Four means, ordered by construction, in one semicircle."""

    def construct(self):
        L = 6.6
        y0 = -2.1
        f = 0.20                       # a = f*L, b = (1-f)*L
        A = np.array([-L / 2, y0, 0.0])
        B = np.array([L / 2, y0, 0.0])
        O = np.array([0.0, y0, 0.0])
        R = L / 2
        P = np.array([A[0] + f * L, y0, 0.0])
        a, b = f * L, (1 - f) * L
        Q = np.array([P[0], y0 + np.sqrt(a * b), 0.0])
        C = np.array([0.0, y0 + R, 0.0])
        # R_ = foot of the perpendicular from P onto OQ
        u = (Q - O) / np.linalg.norm(Q - O)
        Rf = O + np.dot(P - O, u) * u

        arc = Arc(radius=R, start_angle=0, angle=PI, arc_center=O,
                  color=GREY_B, stroke_width=3)
        base = Line(A, B, color=GREY_B)
        oq = Line(O, Q, color=GREY_A, stroke_width=2)

        rms = Line(P, C, color=RED_D, stroke_width=7)
        am = Line(O, C, color=YELLOW_B, stroke_width=7)
        gm = Line(P, Q, color=GREEN_B, stroke_width=7)
        hm = Line(Q, Rf, color=BLUE_B, stroke_width=7)

        self.play(Create(arc), Create(base), run_time=1.2)
        self.play(FadeIn(Dot(P)), FadeIn(Dot(Q, color=GREEN_B)),
                  FadeIn(Dot(C, color=YELLOW_B)))
        self.play(Create(oq))
        # labels are pinned to fixed slots so they never collide
        slots = [
            (rms, "RMS = √((a²+b²)/2)", RED_B, np.array([-4.4, 1.15, 0.0])),
            (am, "AM = (a+b)/2", YELLOW_B, np.array([2.9, 1.15, 0.0])),
            (gm, "GM = √(ab)", GREEN_B, np.array([-4.4, 0.15, 0.0])),
            (hm, "HM = 2ab/(a+b)", BLUE_B, np.array([2.9, 0.15, 0.0])),
        ]
        for seg, lbl, col, at in slots:
            txt = tag(lbl, 22, col).move_to(at)
            lead = DashedLine(txt.get_center(), seg.get_center(),
                              color=col, stroke_width=1.5, dash_length=0.06)
            self.play(Create(seg), FadeIn(txt), Create(lead), run_time=0.8)
        self.play(Create(RightAngle(Line(Rf, Q), Line(Rf, P),
                                    length=0.2, color=GREY_A)))
        self.play(Write(caption("RMS  ≥  AM  ≥  GM  ≥  HM", 34, YELLOW_B)))
        self.hold(2.4)


# ============================================================== E. THE CIRCLE

class E1_CircleWedges(_Board):
    """Cut the disc into wedges, interleave them into a rectangle.

    The interleave is exact: neighbouring wedges, one apex-up and one
    apex-down, share a whole straight edge, so consecutive apexes are a
    chord 2r·sin(d/2) apart and the two rows of apexes r·cos(d/2) apart.
    Each wedge moves rigidly (a turn about its apex while the apex slides).
    As the wedges get thinner the strip closes in on the πr × r rectangle.
    """

    def construct(self):
        R = 1.65
        n = 24
        d = TAU / n                 # angle of one wedge
        half_chord = R * np.sin(d / 2)
        c = R * np.cos(d / 2)
        cols = [BLUE_D, TEAL_D]
        centre = np.array([0.0, 1.75, 0.0])

        def wedge(start_angle, apex, color):
            w = AnnularSector(inner_radius=0.0, outer_radius=R,
                              angle=d, start_angle=start_angle,
                              fill_color=color, fill_opacity=FILL,
                              stroke_color=WHITE, stroke_width=1.0)
            w.shift(apex)           # AnnularSector's apex sits at the origin
            return w

        wedges = [wedge(i * d, centre, cols[i % 2]) for i in range(n)]
        self.play(FadeIn(VGroup(*wedges)), run_time=1.2)
        self.hold(0.7)

        # Lay the wedges out alternately apex-up / apex-down along a strip.
        half = n // 2
        width = n * half_chord              # = half * 2 * half_chord
        y_bot = -2.2
        x0 = -width / 2 - half_chord / 2
        moves = []
        for i in range(n):
            k = i // 2
            if i % 2 == 0:          # apex at the top, arc along the bottom
                apex = np.array([x0 + (2 * k + 1) * half_chord, y_bot + c, 0.0])
                start = -PI / 2 - d / 2
            else:                   # apex at the bottom, arc along the top
                apex = np.array([x0 + (2 * k + 2) * half_chord, y_bot, 0.0])
                start = PI / 2 - d / 2
            turn = start - i * d
            moves.append((wedges[i], centre, apex, turn))

        # exactness: each pair of neighbours shares a whole straight edge
        def edge_ends(apex, start):
            return [apex + R * np.array([np.cos(a), np.sin(a), 0.0])
                    for a in (start, start + d)]
        ends = []
        for i, (_, _, apex, turn) in enumerate(moves):
            ends.append((apex, edge_ends(apex, i * d + turn)))
        for (a1, e1), (a2, e2) in zip(ends, ends[1:]):
            shared = [p for p in (a1, *e1) if any(
                np.allclose(p, q, atol=1e-9) for q in (a2, *e2))]
            assert len(shared) == 2, "neighbouring wedges share an edge"

        def rigid(w, src, dst, turn):
            ghost = w.copy()

            def upd(m, a):
                m.become(ghost.copy().rotate(a * turn, about_point=src)
                         .shift(a * (dst - src)))
            return UpdateFromAlphaFunc(w, upd)

        self.play(*[rigid(*m) for m in moves], run_time=3.0)
        box = Rectangle(width=PI * R, height=R, color=YELLOW_B,
                        stroke_width=5)
        box.move_to([0.0, y_bot + c / 2, 0])
        self.play(Create(box))
        self.play(FadeIn(tag("πr", 26, YELLOW_B).next_to(box, DOWN, buff=0.3)),
                  FadeIn(tag("r", 26, YELLOW_B).next_to(box, RIGHT, buff=0.35)))
        self.play(Write(caption("A  =  πr · r  =  πr²", 34, YELLOW_B)))
        self.hold(2.2)


class E3_InscribedAngle(_Board):
    """The inscribed angle is half the central angle, wherever B sits."""

    def construct(self):
        R = 2.4
        O = np.array([0.0, -0.4, 0.0])
        circ = Circle(radius=R, color=GREY_B).move_to(O)
        # chord endpoints at 200 deg and 340 deg: the arc AC below the chord
        # spans 140 deg, so the central angle is safely non-reflex.
        aA, aC = 200 / 180, 340 / 180
        A = O + R * np.array([np.cos(aA * PI), np.sin(aA * PI), 0.0])
        C = O + R * np.array([np.cos(aC * PI), np.sin(aC * PI), 0.0])
        t = ValueTracker(0.50)      # B rides the major arc, above the chord

        def Bpt():
            th = t.get_value() * PI
            return O + R * np.array([np.cos(th), np.sin(th), 0.0])

        chord = Line(A, C, color=YELLOW_B, stroke_width=4)
        oa = Line(O, A, color=GREY_A, stroke_width=2)
        oc = Line(O, C, color=GREY_A, stroke_width=2)
        ba = always_redraw(lambda: Line(Bpt(), A, color=BLUE_D, stroke_width=3))
        bc = always_redraw(lambda: Line(Bpt(), C, color=BLUE_D, stroke_width=3))
        bo = always_redraw(lambda: DashedLine(Bpt(), O, color=GREY_A, stroke_width=2))
        dotB = always_redraw(lambda: Dot(Bpt(), color=BLUE_B))
        central = Angle(Line(O, A), Line(O, C), radius=0.66, color=RED_D,
                        stroke_width=7)
        def inscribed_arc():
            """Always the non-reflex angle at B, whichever side B is on."""
            Bp = Bpt()
            a1 = np.arctan2(*(A - Bp)[1::-1])
            a2 = np.arctan2(*(C - Bp)[1::-1])
            span = (a2 - a1) % TAU
            if span > PI:               # take the explementary arc instead
                a1, span = a2, TAU - span
            return Arc(radius=0.62, start_angle=a1, angle=span, arc_center=Bp,
                       color=GREEN_B, stroke_width=7)

        inscribed = always_redraw(inscribed_arc)
        lbl2t = tag("2θ", 26, RED_B).move_to(O + 1.05 * DOWN)
        lblt = always_redraw(
            lambda: tag("θ", 26, GREEN_B).move_to(
                Bpt() + 0.95 * (O - Bpt()) / np.linalg.norm(O - Bpt())))

        self.play(Create(circ), FadeIn(Dot(O)), run_time=1.0)
        self.play(Create(chord), Create(oa), Create(oc))
        self.play(Create(central), FadeIn(lbl2t))
        self.play(FadeIn(dotB), Create(ba), Create(bc), Create(bo))
        self.play(FadeIn(inscribed), FadeIn(lblt))
        self.play(Write(caption("inscribed angle  =  ½ · central angle", 32, YELLOW_B)))
        self.play(t.animate.set_value(0.14), run_time=2.4)
        self.play(t.animate.set_value(0.86), run_time=2.8)
        self.play(t.animate.set_value(0.50), run_time=1.6)
        self.hold(1.2)


# ================================================== F. TRIANGLE & POLYGON

class F1_AngleSum(_Board):
    """The three angles rotate onto a straight line."""

    def construct(self):
        A = np.array([-2.6, -1.4, 0.0])
        B = np.array([2.9, -1.4, 0.0])
        C = np.array([0.3, 1.9, 0.0])
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        aA = Angle(Line(A, B), Line(A, C), radius=0.7, color=BLUE_D, stroke_width=8)
        aB = Angle(Line(B, C), Line(B, A), radius=0.7, color=TEAL_D, stroke_width=8)
        aC = Angle(Line(C, A), Line(C, B), radius=0.7, color=ORANGE, stroke_width=8)

        self.play(Create(tri), run_time=1.2)
        self.play(FadeIn(aA), FadeIn(aB), FadeIn(aC))
        self.hold(0.8)

        # a line through C parallel to AB; the two base angles swing up to it
        d = (B - A) / np.linalg.norm(B - A)
        par = Line(C - d * 3.2, C + d * 3.2, color=YELLOW_B, stroke_width=3)
        self.play(Create(par))
        aA2 = Angle(Line(C, C + d * 3.2).reverse_direction(), Line(C, A),
                    radius=0.7, color=BLUE_D, stroke_width=8)
        aB2 = Angle(Line(C, B), Line(C, C + d * 3.2), radius=0.7,
                    color=TEAL_D, stroke_width=8)
        self.play(TransformFromCopy(aA, aA2), TransformFromCopy(aB, aB2),
                  run_time=2.0)
        self.play(Write(caption("α + β + γ  =  180°", 34, YELLOW_B)))
        self.hold(2.2)


class F7_Viviani(_Board):
    """Three perpendiculars from any interior point stack to the altitude."""

    def construct(self):
        s = 4.6
        h = s * np.sqrt(3) / 2
        A = np.array([-s / 2, -h / 2 - 0.3, 0.0])
        B = np.array([s / 2, -h / 2 - 0.3, 0.0])
        C = np.array([0.0, h / 2 - 0.3, 0.0])
        tri = Polygon(A, B, C, stroke_color=YELLOW_B, stroke_width=5)

        tx, ty = ValueTracker(0.0), ValueTracker(-0.6)

        def Ppt():
            return np.array([tx.get_value(), ty.get_value() - 0.3, 0.0])

        def foot(P, U, V):
            d = (V - U) / np.linalg.norm(V - U)
            return U + np.dot(P - U, d) * d

        cols = [BLUE_D, TEAL_D, ORANGE]
        edges = [(A, B), (B, C), (C, A)]
        perps = [always_redraw(lambda i=i: Line(Ppt(), foot(Ppt(), *edges[i]),
                                                color=cols[i], stroke_width=7))
                 for i in range(3)]
        dotP = always_redraw(lambda: Dot(Ppt(), color=WHITE))

        # the stacked bar on the right, always exactly the altitude
        bx = 3.9

        def bar(i):
            below = sum(np.linalg.norm(Ppt() - foot(Ppt(), *edges[j]))
                        for j in range(i))
            ln = np.linalg.norm(Ppt() - foot(Ppt(), *edges[i]))
            y = A[1] + below
            return Line([bx, y, 0], [bx, y + ln, 0], color=cols[i], stroke_width=18)

        bars = [always_redraw(lambda i=i: bar(i)) for i in range(3)]
        ruler = Line([bx + 0.55, A[1], 0], [bx + 0.55, A[1] + h, 0],
                     color=YELLOW_B, stroke_width=5)

        self.play(Create(tri), run_time=1.2)
        self.play(FadeIn(dotP), *[Create(p) for p in perps])
        self.play(*[FadeIn(b) for b in bars], Create(ruler),
                  FadeIn(tag("h", 26, YELLOW_B).next_to(ruler, RIGHT, buff=0.15)))
        self.play(Write(caption("d₁ + d₂ + d₃  =  h,  for every interior point",
                                30, YELLOW_B)))
        self.play(tx.animate.set_value(-1.2), ty.animate.set_value(0.2), run_time=2.2)
        self.play(tx.animate.set_value(1.4), ty.animate.set_value(-1.4), run_time=2.6)
        self.play(tx.animate.set_value(0.0), ty.animate.set_value(-0.6), run_time=1.8)
        self.hold(1.2)


# ============================================================ G. TRIGONOMETRY

class G2_SineAddition(_Board):
    """One unit step at angle A, then a perpendicular step, lands on A+B."""

    def construct(self):
        Aang, Bang = 34 * DEGREES, 38 * DEGREES
        R = 3.5
        O = np.array([-5.0, -2.6, 0.0])

        uA = np.array([np.cos(Aang), np.sin(Aang), 0.0])
        nA = np.array([-np.sin(Aang), np.cos(Aang), 0.0])
        Q = O + R * np.cos(Bang) * uA        # a step of length cos B at angle A
        P = Q + R * np.sin(Bang) * nA        # then sin B, perpendicular to it

        circ = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O,
                   color=GREY_B, stroke_width=2)
        xax = Line(O, O + RIGHT * (R + 0.5), color=GREY_A, stroke_width=2)
        ray = Line(O, O + uA * (R + 0.25), color=GREY_A, stroke_width=2)

        seg1 = Line(O, Q, color=BLUE_D, stroke_width=8)
        seg2 = Line(Q, P, color=TEAL_D, stroke_width=8)
        hyp = Line(O, P, color=YELLOW_B, stroke_width=4)
        ra = RightAngle(Line(Q, O), Line(Q, P), length=0.26, color=RED)

        # the height of P, split at the height of Q
        bx = O[0] + R + 1.35                 # bar column, clear of the figure
        gx = O[0] + R + 2.15
        vQ = DashedLine(Q, np.array([bx, Q[1], 0.0]), color=GREY_A, stroke_width=1.5)
        vP = DashedLine(P, np.array([gx, P[1], 0.0]), color=GREY_A, stroke_width=1.5)
        base = DashedLine(np.array([O[0], O[1], 0.0]),
                          np.array([gx, O[1], 0.0]), color=GREY_A, stroke_width=1.5)

        h1 = Line([bx, O[1], 0], [bx, Q[1], 0], color=BLUE_D, stroke_width=14)
        h2 = Line([bx, Q[1], 0], [bx, P[1], 0], color=TEAL_D, stroke_width=14)
        htot = Line([gx, O[1], 0], [gx, P[1], 0], color=YELLOW_B, stroke_width=8)

        angA = Angle(xax, ray, radius=0.9, color=BLUE_B, stroke_width=5)
        angAB = Angle(xax, hyp, radius=1.5, color=YELLOW_B, stroke_width=5)

        self.play(Create(xax), Create(circ), run_time=1.0)
        self.play(Create(ray), Create(angA),
                  FadeIn(tag("A", 24, BLUE_B).move_to(
                      O + 0.72 * np.array([np.cos(Aang / 2),
                                           np.sin(Aang / 2), 0.0]))))
        self.play(Create(hyp), Create(angAB),
                  FadeIn(tag("A+B", 22, YELLOW_B).move_to(
                      O + 2.10 * np.array([np.cos((Aang + Bang) / 2),
                                           np.sin((Aang + Bang) / 2), 0.0]))))
        self.play(Create(seg1),
                  FadeIn(tag("cos B", 21, BLUE_B).move_to((O + Q) / 2 - nA * 0.50)))
        self.play(Create(seg2), Create(ra),
                  FadeIn(tag("sin B", 21, TEAL_B).move_to((Q + P) / 2 - uA * 0.55)))
        self.play(Create(base), Create(vQ), Create(vP))
        self.play(Create(h1), Create(h2), Create(htot))
        self.play(FadeIn(tag("sinA·cosB", 19, BLUE_B)
                         .next_to(h1, RIGHT, buff=0.10).shift(RIGHT * 1.05)),
                  FadeIn(tag("cosA·sinB", 19, TEAL_B)
                         .next_to(h2, RIGHT, buff=0.10).shift(RIGHT * 1.05)),
                  FadeIn(tag("sin(A+B)", 20, YELLOW_B)
                         .next_to(htot, RIGHT, buff=0.10).shift(RIGHT * 0.80)))
        self.play(Write(caption("sin(A+B)  =  sinA·cosB + cosA·sinB", 32, YELLOW_B)))
        self.hold(2.4)


# ================================================================ J. CURIOS

class J2_GoldenRectangle(_Board):
    """Remove a square; what remains is the same rectangle again."""

    def construct(self):
        phi = (1 + np.sqrt(5)) / 2
        H = 3.2
        W = H * phi
        rect = Rectangle(width=W, height=H, stroke_color=YELLOW_B, stroke_width=5)
        rect.move_to(LEFT * 0.8 + DOWN * 0.3)
        self.play(Create(rect), run_time=1.2)
        self.play(FadeIn(tag("φ", 30, YELLOW_B).next_to(rect, DOWN, buff=0.2)),
                  FadeIn(tag("1", 30, YELLOW_B).next_to(rect, LEFT, buff=0.2)))

        cols = [BLUE_D, TEAL_D, GREEN_D, ORANGE, RED_D, PURPLE_B]
        x0, y0 = rect.get_corner(DL)[0], rect.get_corner(DL)[1]
        w, h = W, H
        vertical = True
        for i in range(6):
            side = h if vertical else w
            sq = Square(side_length=side, fill_color=cols[i], fill_opacity=FILL,
                        stroke_color=WHITE, stroke_width=2)
            if vertical:
                sq.move_to([x0 + side / 2, y0 + side / 2, 0])
                x0 += side
                w -= side
            else:
                sq.move_to([x0 + side / 2, y0 + h - side / 2, 0])
                h -= side
            vertical = not vertical
            self.play(FadeIn(sq), run_time=0.8 if i < 3 else 0.4)
            if i == 0:
                self.play(FadeIn(tag("1×1", 24).move_to(sq)), run_time=0.4)

        self.play(Write(caption("φ = 1 + 1/φ    ⟹    φ² = φ + 1", 34, YELLOW_B)))
        self.hold(2.4)
