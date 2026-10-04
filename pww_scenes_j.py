# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_j.py — number-theory curios and dissections (J1 J3 J4 J7 J8 J9).
#
#     J1_RootTwoIrrational   Tennenbaum's descent: a² = 2b² has no least solution
#     J3_GoldenFraction      golden rectangle, square by square: the continued fraction
#     J4_TheodorusSpiral     chained right triangles with legs √n and 1
#     J7_PizzaTheorem        eight slices; nine congruent pairs of pieces
#     J8_MissingSquare       the 13 × 5 "triangle" is not a triangle
#     J9_BolyaiGerwien       triangle → rectangle → square, area kept at every cut
#
# Every scene checks its own geometry with check(...) before it animates.


# ---------------------------------------------------------------- helpers

def _dim(p, q, label, off, color=GREY_A, size=24, gap=0.24):
    """Dimension line beside segment pq (screen points), shifted by `off`:
    a thin line with end ticks and the label beyond it."""
    p, q, off = to3(p), to3(q), to3(off)
    n = off / np.linalg.norm(off)
    a, b = p + off, q + off
    tick = 0.09 * n
    g = VGroup(Line(a, b, color=color, stroke_width=2),
               Line(a - tick, a + tick, color=color, stroke_width=2),
               Line(b - tick, b + tick, color=color, stroke_width=2))
    t = tag(label, size, color)
    t.move_to((a + b) / 2 + n * (gap + 0.5 * (abs(n[0]) * t.width
                                             + abs(n[1]) * t.height)))
    return VGroup(g, t)


def _square(c0, s, color, op=FILL, sw=2, stroke=WHITE):
    """Axis-aligned square with bottom-left corner c0 (screen) and side s."""
    x, y = c0[0], c0[1]
    return mk([(x, y), (x + s, y), (x + s, y + s), (x, y + s)], color, op,
              stroke_width=sw, stroke_color=stroke)


def _rows(lines, x0, y0, dy):
    """Left-align a list of mobjects at x0, first baseline-ish at y0."""
    for i, m in enumerate(lines):
        m.move_to(np.array([x0 + m.width / 2, y0 - i * dy, 0.0]))
    return lines


# ======================================================================= J1

class J1_RootTwoIrrational(Board):
    """Tennenbaum. If a² = 2b² with a least, put the two b-squares in opposite
    corners of the a-square: the doubly covered centre must equal the two
    uncovered corners, (2b−a)² = 2(a−b)², a smaller solution. The picture
    is drawn at the only ratio that could work, a/b = √2."""

    def construct(self):
        r2 = np.sqrt(2.0)
        A = 4.8                                   # screen side of the a-square
        X0, Y0 = -6.0, -2.2                       # its bottom-left corner
        C = np.array([X0 + A / 2, Y0 + A / 2, 0.0])

        # levels of the descent: a_k, b_k with a_{k+1} = 2b_k − a_k,
        # b_{k+1} = a_k − b_k, and b_k = a_k/√2 at every level
        sides = [(A, A / r2)]
        for _ in range(4):
            a, b = sides[-1]
            sides.append((2 * b - a, a - b))
        for a, b in sides:
            check(close(a, r2 * b), "every level keeps a = √2·b")
            check(0 < 2 * b - a < a and a - b > 0, "0 < 2b − a < a")
            check(close((2 * b - a) ** 2, 2 * (a - b) ** 2),
                  "overlap = two uncovered corners")

        def level(k):
            """Arena, two b-squares, overlap and the two corners of level k."""
            a, b = sides[k]
            o = C - np.array([a / 2, a / 2, 0.0])        # arena corner
            return dict(
                arena=(o, a),
                b1=(o, b), b2=(o + np.array([a - b, a - b, 0.0]), b),
                ov=(o + np.array([a - b, a - b, 0.0]), 2 * b - a),
                c1=(o + np.array([b, 0.0, 0.0]), a - b),           # bottom-right
                c2=(o + np.array([0.0, b, 0.0]), a - b),           # top-left
            )

        L0 = level(0)
        big = _square(L0["arena"][0], A, GREY_E, 0.35, sw=4, stroke=YELLOW_B)
        dim_a = _dim(L0["arena"][0], L0["arena"][0] + UP * A, "a",
                     LEFT * 0.22, YELLOW_B, 28)

        prem1 = tag("√2 = a/b   ⟹   a² = 2b²", 32, YELLOW_B)
        prem2 = tag("a, b ∈ ℕ,   a least", 24, GREY_A)
        _rows([prem1, prem2], 0.3, 3.25, 0.55)

        self.play(Create(big), FadeIn(dim_a), run_time=1.2)
        self.play(FadeIn(prem1), FadeIn(prem2), run_time=0.8)

        # the two b-squares, slid into opposite corners
        BLUE_OP = 0.42
        b1 = _square(*L0["b1"], BLUE_D, BLUE_OP)
        b2 = _square(*L0["b2"], BLUE_D, BLUE_OP)
        dim_b = _dim(L0["b1"][0], L0["b1"][0] + RIGHT * L0["b1"][1], "b",
                     DOWN * 0.22, BLUE_B, 26)
        self.play(FadeIn(b1, shift=RIGHT * 0.6 + UP * 0.6),
                  FadeIn(b2, shift=LEFT * 0.6 + DOWN * 0.6), run_time=1.4)
        self.play(FadeIn(dim_b), run_time=0.5)
        self.hold(0.6)

        # doubly covered centre and the two uncovered corners
        ov = _square(*L0["ov"], ORANGE, 0.9)
        c1 = _square(*L0["c1"], GREEN_D, 0.9)
        c2 = _square(*L0["c2"], GREEN_D, 0.9)
        o = L0["arena"][0]
        a0, bb0 = sides[0]
        dim_ab = _dim(o + RIGHT * bb0, o + RIGHT * a0, "a−b", DOWN * 0.22,
                      GREEN_B, 24)
        dim_ov = _dim(o + np.array([a0 - bb0, a0, 0]), o + np.array([bb0, a0, 0]),
                      "2b−a", UP * 0.22, ORANGE, 24)
        eq1 = Text("(2b−a)² = 2(a−b)²", font_size=30,
                   t2c={"(2b−a)²": ORANGE, "2(a−b)²": GREEN_B})
        _rows([eq1], 0.3, 1.75, 0)
        self.play(FadeIn(ov), FadeIn(dim_ov), run_time=0.9)
        self.play(FadeIn(c1), FadeIn(c2), FadeIn(dim_ab), run_time=0.9)
        self.play(Write(eq1), run_time=1.0)
        self.hold(0.6)

        sub1 = tag("a′ = 2b − a,    b′ = a − b", 26)
        sub2 = Text("a′² = 2b′²,    0 < a′ < a  ↯", font_size=26,
                    t2c={"↯": RED_B})
        _rows([sub1, sub2], 0.3, 0.85, 0.8)
        # the descending chain, laid out once so tokens never shift
        tokens = ["a", ">", "a′", ">", "a″", ">", "a‴", ">", "…"]
        chain = VGroup(*[tag(t, 30, YELLOW_B) for t in tokens])
        chain.arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        chain.move_to(np.array([0.3 + chain.width / 2, -0.95, 0.0]))
        reveal = {1: chain[0:3], 2: chain[3:5], 3: chain[5:7]}

        # the descent: the two corners become the next level's b-squares
        prev_ov = ov
        prev_corners = (c1, c2)
        for k in range(1, 4):
            Lk = level(k)
            nb1 = _square(*Lk["b1"], BLUE_D, BLUE_OP)
            nb2 = _square(*Lk["b2"], BLUE_D, BLUE_OP)
            check(close(Lk["arena"][1], 2 * sides[k - 1][1] - sides[k - 1][0]),
                  "next arena = previous overlap")
            mv1, mv2 = prev_corners[0].copy(), prev_corners[1].copy()
            arena = _square(*Lk["arena"], GREY_E, 0.55, sw=2.5, stroke=ORANGE)
            self.play(Transform(mv1, nb1), Transform(mv2, nb2),
                      Transform(prev_ov, arena), run_time=1.6 if k == 1 else 1.0)
            if k == 1:
                self.play(FadeIn(sub1), FadeIn(sub2), run_time=0.8)
            nov = _square(*Lk["ov"], ORANGE, 0.9, sw=1.5)
            nc1 = _square(*Lk["c1"], GREEN_D, 0.9, sw=1.5)
            nc2 = _square(*Lk["c2"], GREEN_D, 0.9, sw=1.5)
            self.play(FadeIn(nov), FadeIn(nc1), FadeIn(nc2),
                      FadeIn(reveal[k]), run_time=0.9 if k == 1 else 0.6)
            prev_ov, prev_corners = nov, (nc1, nc2)
            if k == 1:
                self.hold(0.5)

        never = Text("an endless descent in ℕ", font_size=24, color=RED_B)
        never.next_to(chain, DOWN, buff=0.3).align_to(chain, LEFT)
        self.play(FadeIn(chain[7:]), FadeIn(never), run_time=0.8)
        self.play(Write(caption("a² = 2b² has no least solution in ℕ   ⟹   √2 ∉ ℚ",
                                30)))
        self.hold(2.4)


# ======================================================================= J3

def _rot(v, deg):
    """Rotate a screen vector by deg degrees (z untouched)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    v = to3(v)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])


def _similarity(mob, c_from, c_to, angle_deg, factor, **kw):
    """Animate a direct similarity: turn by angle about c_from, scale by
    factor about c_from, carry c_from to c_to. Every frame is a similar copy
    (rotation + uniform scale), never a vertex-by-vertex morph."""
    start = mob.copy()
    c_from, c_to = to3(c_from), to3(c_to)

    def upd(m, a):
        m.become(start.copy())
        m.rotate(np.radians(angle_deg) * a, about_point=c_from)
        m.scale(factor ** a, about_point=c_from)
        m.shift(a * (c_to - c_from))
    return UpdateFromAlphaFunc(mob, upd, **kw)


class J3_GoldenFraction(Board):
    """φ as a continued fraction. Cut the unit square off a golden rectangle:
    the remainder, turned a quarter and enlarged by φ, is the same rectangle,
    so the cut repeats forever, and each cut writes one more 1 + 1/(…)."""

    def construct(self):
        phi = (1 + np.sqrt(5)) / 2
        check(close(phi, 1 + 1 / phi), "φ = 1 + 1/φ")
        s = 5.8 / phi                               # screen units per unit
        BL = np.array([-6.0, -1.45, 0.0])

        def S(x, y):
            return BL + s * np.array([x, y, 0.0])

        def box(x0, y0, x1, y1, col, op, sw=2.0, stroke=WHITE):
            return mk([S(x0, y0), S(x1, y0), S(x1, y1), S(x0, y1)], col, op,
                      stroke_width=sw, stroke_color=stroke)

        c_s = S(phi / 2, 0.5)                       # centre of the rectangle
        c_r = S(1 + 1 / (2 * phi), 0.5)             # centre of the remainder

        def Z(p):            # remainder -> whole rectangle (quarter turn, ×φ)
            return c_s + phi * _rot(to3(p) - c_r, 90)

        def Zinv(p):         # one level outward
            return c_r + _rot(to3(p) - c_s, -90) / phi

        rect_pts = [S(0, 0), S(phi, 0), S(phi, 1), S(0, 1)]
        rem_pts = [S(1, 0), S(phi, 0), S(phi, 1), S(1, 1)]
        img = [Z(p) for p in rem_pts]
        check(all(any(close(q, r, 1e-9) for r in rect_pts) for q in img),
              "the remainder, turned and enlarged by φ, is the rectangle")

        # the squares as they sit in the original rectangle
        unit = [S(0, 0), S(1, 0), S(1, 1), S(0, 1)]
        nested, pts = [], unit
        for k in range(7):
            nested.append(pts)
            pts = [Zinv(p) for p in pts]
        rem3 = rem_pts
        for _ in range(3):
            rem3 = [Zinv(p) for p in rem3]
        tot = sum(abs(area(q)) for q in nested[:4]) + abs(area(rem3))
        check(close(tot, phi * s * s, 1e-9), "four squares + remainder = rectangle")

        cols = [(TEAL_D, TEAL_B), (GOLD_D, GOLD_B), (MAROON_B, MAROON_A),
                (BLUE_D, BLUE_B), (GREEN_D, GREEN_B), (PURPLE_B, PURPLE_A),
                (RED_D, RED_B)]

        # ---------------- the rectangle and its sides
        frame = box(0, 0, phi, 1, BLACK, 0.0, sw=4, stroke=YELLOW_B)
        d_left = _dim(S(0, 0), S(0, 1), "1", LEFT * 0.22, YELLOW_B, 28)
        d_top = _dim(S(0, 1), S(phi, 1), "φ", UP * 0.22, YELLOW_B, 30)
        d_b1 = _dim(S(0, 0), S(1, 0), "1", DOWN * 0.22, GREY_A, 26)
        d_b2 = _dim(S(1, 0), S(phi, 0), "1/φ", DOWN * 0.22, GREY_A, 24)

        # ---------------- the continued fraction, laid out once
        y0, dx, dy, x_end = 2.75, 0.92, 0.84, 6.45
        eq = tag("φ =", 36, YELLOW_B)
        eq.move_to(np.array([0.35 + eq.width / 2, y0, 0.0]))
        xe = eq.get_right()[0] + 0.22
        lv = []
        for i in range(4):
            yi = y0 - i * dy
            plus = Text("1 +", font_size=34, t2c={"1": cols[i][1]})
            plus.move_to(np.array([xe + i * dx + plus.width / 2, yi, 0.0]))
            xa = plus.get_right()[0] + 0.12
            bar = Line(np.array([xa, yi, 0]), np.array([x_end, yi, 0]),
                       stroke_width=2.5, color=WHITE)
            num = tag("1", 30).move_to(np.array([(xa + x_end) / 2, yi + 0.33, 0]))
            den = tag("φ", 32, GREY_A).move_to(
                np.array([(xa + x_end) / 2, yi - 0.36, 0]))
            lv.append((plus, bar, num, den))
        tail = Text("1 + ⋯", font_size=34, t2c={"1": cols[4][1]})
        tail.move_to(np.array([xe + 4 * dx + tail.width / 2, y0 - 4 * dy, 0]))
        check(tail.get_right()[0] < 6.6, "fraction fits the frame")

        self.play(Create(frame), FadeIn(d_left), FadeIn(d_top), run_time=1.3)

        sq = box(0, 0, 1, 1, cols[0][0], 0.9)
        rem = box(1, 0, phi, 1, GREY_B, 0.22)
        self.play(FadeIn(sq), FadeIn(rem), run_time=0.8)
        self.play(FadeIn(d_b1), FadeIn(d_b2), run_time=0.7)
        self.play(FadeIn(eq), *[FadeIn(m) for m in lv[0]], run_time=1.0)
        self.hold(0.6)

        # ---------------- zoom: the remainder becomes the rectangle again
        for k in range(1, 4):
            rt = 1.5 if k == 1 else 1.1
            self.play(FadeOut(sq),
                      _similarity(rem, c_r, c_s, 90, phi, run_time=rt),
                      run_time=rt)
            self.remove(rem)
            sq = box(0, 0, 1, 1, cols[k][0], 0.9)
            rem = box(1, 0, phi, 1, GREY_B, 0.22)
            self.add(rem, frame)
            self.play(FadeIn(sq), run_time=0.6)
            self.play(FadeOut(lv[k - 1][3], scale=0.6),
                      *[FadeIn(m, shift=UP * 0.12) for m in lv[k]], run_time=0.7)
            if k == 1:
                self.hold(0.4)

        # ---------------- zoom back out: every square in its true place
        here = VGroup(sq, rem)
        c3 = Zinv(Zinv(Zinv(c_s)))
        olds = [mk(nested[k], cols[k][0], 0.9, stroke_width=2) for k in range(3)]
        news = [mk(nested[k], cols[k][0], 0.9, stroke_width=1.2)
                for k in range(4, 7)]
        self.play(_similarity(here, c_s, c3, 90, phi ** -3, run_time=2.0),
                  *[FadeIn(m) for m in olds], run_time=2.0)
        self.add(frame)
        self.play(*[FadeIn(m) for m in news], FadeOut(lv[3][3], scale=0.6),
                  FadeIn(tail, shift=UP * 0.12), run_time=0.9)
        check(close(sq.get_center(), np.mean(nested[3], axis=0), 1e-6),
              "the fourth square lands in its nested place")
        self.play(Write(caption("φ = 1 + 1/φ   ⟹   φ = 1 + 1/(1 + 1/(1 + 1/(1 + ⋯)))",
                                30)))
        self.hold(2.4)


# ======================================================================= J4

def _inside_convex(pt, poly):
    """True if pt lies strictly inside the convex polygon (any orientation)."""
    pt = np.asarray(pt, float)[:2]
    poly = [np.asarray(p, float)[:2] for p in poly]
    sgn = 0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        c = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(c) < 1e-12:
            return False
        s = 1 if c > 0 else -1
        if sgn == 0:
            sgn = s
        elif s != sgn:
            return False
    return True


def _box(m, pad=0.04):
    """Axis-aligned screen box of a mobject: (xmin, ymin, xmax, ymax)."""
    lo, hi = m.get_corner(DL), m.get_corner(UR)
    return (lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad)


def _boxes_meet(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_hits_polys(bx, polys, n=7):
    xs = np.linspace(bx[0], bx[2], n)
    ys = np.linspace(bx[1], bx[3], n)
    return any(_inside_convex((x, y), P) for x in xs for y in ys for P in polys)


class J4_TheodorusSpiral(Board):
    """Spiral of Theodorus. Each new right triangle has the previous
    hypotenuse √n as one leg and 1 as the other, so its hypotenuse is
    √(n+1) by Pythagoras. Sixteen triangles reach √17 just short of a turn."""

    def construct(self):
        N = 16
        V = [None, np.array([1.0, 0.0])]
        for n in range(1, N + 1):
            v = V[-1]
            V.append(v + np.array([-v[1], v[0]]) / np.linalg.norm(v))
        for m in range(1, N + 2):
            check(close(V[m] @ V[m], m), f"|OV_{m}|² = {m}")
        for n in range(1, N + 1):
            check(close((V[n + 1] - V[n]) @ V[n], 0.0), "right angle at V_n")
            check(close(np.linalg.norm(V[n + 1] - V[n]), 1.0), "outer leg = 1")
        turn = sum(np.arctan(1 / np.sqrt(n)) for n in range(1, N + 1))
        check(turn < TAU < turn + np.arctan(1 / np.sqrt(N + 1)),
              "16 triangles fit in one turn, a 17th would not")

        k = 1.04
        O = np.array([0.85, 1.19, 0.0])

        def P(v):
            return O + k * np.array([v[0], v[1], 0.0])

        cols = [BLUE_D, TEAL_D]
        tris, polys, marks, unit_legs, labels = [], [], [], [], []
        for n in range(1, N + 1):
            pts = [O, P(V[n]), P(V[n + 1])]
            polys.append(pts)
            tris.append(mk(pts, cols[n % 2], FILL, stroke_width=1.5))
            u1 = (O - P(V[n])) / np.linalg.norm(O - P(V[n]))
            u2 = (P(V[n + 1]) - P(V[n])) / np.linalg.norm(P(V[n + 1]) - P(V[n]))
            r = 0.13
            mark = VMobject(stroke_color=WHITE, stroke_width=1.6)
            mark.set_points_as_corners([P(V[n]) + r * u1, P(V[n]) + r * (u1 + u2),
                                        P(V[n]) + r * u2])
            marks.append(mark)
            unit_legs.append(Line(P(V[n]), P(V[n + 1]), color=YELLOW_B,
                                  stroke_width=4))
            m = n + 1
            lab = tag(f"√{m}", 22 if m < 10 else 21, YELLOW_B if m in (4, 9, 16)
                      else WHITE)
            rad = V[m] / np.linalg.norm(V[m])
            ext = 0.5 * (abs(rad[0]) * lab.width + abs(rad[1]) * lab.height)
            lab.move_to(P(V[m]) + (0.14 + ext) * np.array([rad[0], rad[1], 0.0]))
            labels.append(lab)

        first_leg = Line(O, P(V[1]), color=YELLOW_B, stroke_width=4)
        one_b = tag("1", 22, YELLOW_B).next_to(unit_legs[0], RIGHT, buff=0.12)

        rows = [tag("1² + 1² = 2", 28), tag("(√2)² + 1² = 3", 28),
                tag("(√3)² + 1² = 4", 28), tag("⋯", 28)]
        _rows(rows, -6.3, 3.25, 0.62)

        # nothing may collide: labels with labels, labels with triangles
        boxes = [_box(t) for t in labels + [one_b] + rows]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                check(not _boxes_meet(boxes[i], boxes[j]), "labels apart")
        for t in labels + [one_b] + rows:
            check(not _box_hits_polys(_box(t), polys), "label clear of triangles")
            bx = _box(t, 0)
            check(-6.6 <= bx[0] and bx[2] <= 6.6 and -3.0 <= bx[1]
                  and bx[3] <= 3.75, "label inside the safe area")

        # ---- the first triangle, slowly
        self.play(FadeIn(Dot(O, radius=0.05)), Create(first_leg), run_time=0.8)
        self.play(Create(unit_legs[0]), Create(marks[0]), FadeIn(one_b),
                  run_time=0.8)
        spoke = Line(O, P(V[2]), color=WHITE, stroke_width=3)
        self.play(Create(spoke), run_time=0.6)
        self.add(tris[0], first_leg, unit_legs[0], marks[0])
        self.play(FadeIn(tris[0]), FadeIn(labels[0]), FadeIn(rows[0]),
                  run_time=0.7)
        self.remove(spoke)
        self.hold(0.4)

        # ---- the old hypotenuse becomes a leg: two more, slowly
        for n in (2, 3):
            i = n - 1
            leg = Line(O, P(V[n]), color=YELLOW_B, stroke_width=6)
            self.play(ShowPassingFlash(leg.copy().set_color(YELLOW), time_width=0.6),
                      run_time=0.5)
            self.play(Create(unit_legs[i]), Create(marks[i]), run_time=0.5)
            spoke = Line(O, P(V[n + 1]), color=WHITE, stroke_width=3)
            self.play(Create(spoke), run_time=0.4)
            self.add(tris[i], unit_legs[i], marks[i])
            self.bring_to_front(*unit_legs[:i + 1])
            self.play(FadeIn(tris[i]), FadeIn(labels[i]), FadeIn(rows[i]),
                      run_time=0.6)
            self.remove(spoke)

        # ---- the rest, briskly
        for i in range(3, N):
            self.add(tris[i])
            self.bring_to_front(*unit_legs[:i], first_leg)
            self.play(FadeIn(tris[i]), Create(unit_legs[i]), Create(marks[i]),
                      FadeIn(labels[i]),
                      *([FadeIn(rows[3])] if i == 3 else []), run_time=0.42)
        self.bring_to_front(*unit_legs, first_leg, *marks)
        self.play(Write(caption("(√n)² + 1² = n + 1   ⟹   hₙ = √n,   n = 1, 2, 3, …",
                                30)))
        self.hold(2.4)


# ======================================================================= J7

def _disc_pts(n=720):
    """Polygon of the unit disc, samples at multiples of 2π/n from angle 0
    (n divisible by 8, so every reflection used below permutes them)."""
    t = np.arange(n) * TAU / n
    return [np.array([np.cos(a), np.sin(a)]) for a in t]


def _clip(poly, a, b, c):
    """Part of a convex polygon where a·x + b·y + c ≥ 0."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        fp = a * P[0] + b * P[1] + c
        fq = a * Q[0] + b * Q[1] + c
        if fp >= 0:
            out.append(P)
        if (fp >= 0) != (fq >= 0):
            out.append(P + fp / (fp - fq) * (Q - P))
    return out


def _region(*halfplanes):
    poly = _disc_pts()
    for h in halfplanes:
        poly = _clip(poly, *h)
    return poly


def _inside_tol(pt, poly, tol):
    n = len(poly)
    s = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n))
    sg = 1 if s > 0 else -1
    for i in range(n):
        A, B = poly[i], poly[(i + 1) % n]
        e = B - A
        L = float(np.hypot(*e))
        if L < 1e-14:
            continue
        if sg * (e[0] * (pt[1] - A[1]) - e[1] * (pt[0] - A[0])) / L < -tol:
            return False
    return True


def _same_set(X, Y, tol=2e-4):
    """Two convex polygons coincide (up to the arc sampling)."""
    return (abs(abs(area(X)) - abs(area(Y))) < 1e-4
            and all(_inside_tol(v, Y, tol) for v in X)
            and all(_inside_tol(v, X, tol) for v in Y))


def _pole(poly, n=40):
    """A point deep inside a convex polygon (grid search for the largest
    distance to the boundary) — where a label fits best."""
    P = np.array(poly)
    lo, hi = P.min(0), P.max(0)
    edges = [(P[i], P[(i + 1) % len(P)]) for i in range(len(P))
             if np.hypot(*(P[(i + 1) % len(P)] - P[i])) > 1e-9]
    best, bd = P.mean(0), -1.0
    for x in np.linspace(lo[0], hi[0], n):
        for y in np.linspace(lo[1], hi[1], n):
            pt = np.array([x, y])
            if not _inside_tol(pt, poly, 0.0):
                continue
            d = min(abs((b - a)[0] * (pt - a)[1] - (b - a)[1] * (pt - a)[0])
                    / np.hypot(*(b - a)) for a, b in edges)
            if d > bd:
                bd, best = d, pt
    return best, bd


class J7_PizzaTheorem(Board):
    """The pizza theorem by dissection. Eight slices from P, alternately
    orange and blue. Nine moves — reflections in the vertical, horizontal and
    diagonal diameters, then two folds over the cuts through P — carry nine
    orange pieces exactly onto nine blue ones. The drawn point is P = (2q, q),
    for which the last central parallelogram splits into two copies of the
    leftover blue parallelograms."""

    def construct(self):
        p, q = 0.44, 0.22                         # P = (p, q), p = 2q
        check(close(p, 2 * q), "drawn instance has p = 2q")
        Rs = 3.3
        C = np.array([0.0, 0.375, 0.0])

        def S(v):
            return C + Rs * np.array([v[0], v[1], 0.0])

        # half-planes a·x + b·y + c ≥ 0 for the sides of the four cuts
        L0 = lambda s: (0, s, -s * q)             # +1: y ≥ q
        L90 = lambda s: (s, 0, -s * p)            # +1: x ≥ p
        L45 = lambda s: (-s, s, -s * (q - p))     # +1: y − x ≥ q − p
        L135 = lambda s: (s, s, -s * (p + q))     # +1: x + y ≥ p + q
        R = _region
        sl = [R(L0(+1), L45(-1)), R(L90(+1), L45(+1)), R(L90(-1), L135(+1)),
              R(L0(+1), L135(-1)), R(L0(-1), L45(+1)), R(L90(-1), L45(-1)),
              R(L90(+1), L135(-1)), R(L0(-1), L135(+1))]
        ORG, BLU = ORANGE, BLUE_D
        colour = [ORG if k % 2 == 0 else BLU for k in range(8)]

        # the pieces (orange ones first in each pair)
        pc = dict(
            S0=sl[0], S2=sl[2], S7=sl[7], S1=sl[1],
            r0S0=R(L0(+1), (-1, -1, q - p)),
            r0S7=R(L0(-1), (-1, 1, -(p + q))),
            r90S2=R(L90(-1), (1, -1, -(p + q))),
            r90S1=R(L90(+1), (-1, -1, p - q)),
            D2=R(L90(+1), (1, 1, -(p - q)), L135(-1)),
            L1a=R((0, 1, -p), (1, 1, -(p - q)), L135(-1)),
            L2p=R((-1, 0, q), (-1, 1, p + q), L45(-1)),
            sdL2p=R(L0(-1), (-1, 1, -(p - q)), (1, -1, p + q)),
            L1c=R(L0(+1), (1, 1, -(q - p)), (-1, -1, p - q)),
            r90L1c=R((0, -1, -q), (-1, 1, -(q - p)), (1, -1, p - q)),
            Pup=R((0, 1, 0), L0(-1), (-1, 1, -(q - p)), (1, -1, p - q)),
            Pdn=R((0, -1, 0), (0, 1, q), (-1, 1, -(q - p)), (1, -1, p - q)),
            ParA=R((0, 1, -q), (0, -1, p), (1, 1, -(p - q)), L135(-1)),
            ParB=R((1, 0, -q), L90(-1), (-1, 1, p + q), L45(-1)),
        )
        rho0 = lambda v: np.array([-v[0], v[1]])
        rho90 = lambda v: np.array([v[0], -v[1]])
        sd = lambda v: np.array([v[1], v[0]])
        fold0 = lambda v: np.array([v[0], 2 * q - v[1]])
        fold45 = lambda v: np.array([v[1] - (q - p), v[0] + (q - p)])
        half = lambda v: -np.asarray(v)
        mp = lambda g, X: [g(v) for v in X]

        # (number, orange piece, blue piece, which one moves, the map)
        pairs = [
            (1, "S0", "r0S0", "S0", rho0),
            (2, "r0S7", "S7", "S7", rho0),
            (3, "S2", "r90S2", "S2", rho90),
            (4, "r90S1", "S1", "S1", rho90),
            (5, "D2", "L1a", "D2", sd),
            (6, "sdL2p", "L2p", "L2p", sd),
            (7, "r90L1c", "L1c", "L1c", rho90),
            (8, "Pup", "ParA", "Pup", fold0),
            (9, "Pdn", "ParB", "Pdn", lambda v: fold45(half(v))),
        ]
        for n, o, b, mv, g in pairs:
            tgt = b if mv == o else o
            check(_same_set(mp(g, pc[mv]), pc[tgt]), f"move {n} lands exactly")
        orange = [pc[o] for _, o, _, _, _ in pairs]
        blue = [pc[b] for _, _, b, _, _ in pairs]
        a_or, a_bl = sum(abs(area(X)) for X in orange), sum(abs(area(X)) for X in blue)
        check(close(a_or, sum(abs(area(sl[k])) for k in (0, 2, 4, 6)), 1e-6),
              "orange pieces tile the orange slices")
        check(close(a_bl, sum(abs(area(sl[k])) for k in (1, 3, 5, 7)), 1e-6),
              "blue pieces tile the blue slices")
        check(close(a_or, a_bl, 1e-6) and close(a_or, PI / 2, 1e-4),
              "orange = blue = half the disc")

        def poly(X, col, op=FILL, sw=1.5):
            return mk([S(v) for v in X], col, op, stroke_width=sw)

        def outline(X, sw=1.6):
            return mk([S(v) for v in X], BLACK, 0.0, stroke_width=sw)

        def number(n, X, col=WHITE):
            c, _ = _pole(X)
            return tag(str(n), 24, col).move_to(S(c))

        # ---------------- the pizza
        circle = Circle(radius=Rs, color=WHITE, stroke_width=3).move_to(C)
        Ps = S((p, q))

        def chord(d, colr=WHITE, sw=2.5):
            d = np.asarray(d, float) / np.linalg.norm(d)
            P2 = np.array([p, q])
            b = P2 @ d
            t = np.sqrt(b * b - (P2 @ P2 - 1.0))
            return Line(S(P2 + (-b - t) * d), S(P2 + (-b + t) * d), color=colr,
                        stroke_width=sw)

        cuts = VGroup(*[chord(d) for d in ((1, 0), (1, 1), (0, 1), (-1, 1))])
        slices = VGroup(*[poly(sl[k], colour[k], FILL, 0) for k in range(8)])
        dotP = Dot(Ps, radius=0.06, color=WHITE)
        self.play(Create(circle), FadeIn(dotP), run_time=1.0)
        self.play(LaggedStart(*[Create(c) for c in cuts], lag_ratio=0.2),
                  run_time=1.2)
        self.play(FadeIn(slices), run_time=0.9)
        self.bring_to_front(cuts, circle, dotP)
        # eight equal angles at P
        rays = [Ps + np.array([np.cos(a), np.sin(a), 0.0]) for a in
                np.arange(8) * PI / 4]
        arcs = VGroup(*[angle_arc(Ps, rays[i], rays[(i + 1) % 8],
                                  radius=0.30 if i % 2 else 0.42,
                                  color=WHITE, width=2.5) for i in range(8)])
        bis = np.array([np.cos(7 * PI / 8), np.sin(7 * PI / 8), 0.0])
        deg = tag("45°", 22).move_to(Ps + 0.98 * bis)
        check(all(close(abs(a.angle), PI / 4) for a in arcs), "eight 45° angles")
        self.play(Create(arcs), FadeIn(deg), run_time=0.8)
        self.hold(0.6)
        self.play(FadeOut(arcs), FadeOut(deg), run_time=0.4)

        def mirror(d, through=(0.0, 0.0), ext=1.12):
            d = np.asarray(d, float) / np.linalg.norm(d)
            o = np.asarray(through, float)
            return DashedLine(S(o - ext * d), S(o + ext * d), color=YELLOW_B,
                              stroke_width=3.5, dash_length=0.12)

        lines_kept = VGroup()

        def flip(items, axis, about, mline, rt=1.4):
            """items: list of (pair number, moving piece, target piece, colour)."""
            self.play(Create(mline), run_time=0.45)
            copies = [poly(X, col, 0.92, 2) for _, X, _, col in items]
            self.add(*copies)
            self.play(*[Rotate(c, angle=PI, axis=axis, about_point=about)
                        for c in copies], run_time=rt)
            land = []
            for (n, X, Y, col), c in zip(items, copies):
                o1, o2 = outline(X), outline(Y)
                lines_kept.add(o1, o2)
                land += [FadeOut(c), FadeIn(o1), FadeIn(o2),
                         FadeIn(number(n, X)), FadeIn(number(n, Y))]
            self.play(*land, FadeOut(mline), run_time=0.7)
            self.bring_to_front(cuts, circle, dotP)

        UPv = np.array([0.0, 1.0, 0.0])
        RIGHTv = np.array([1.0, 0.0, 0.0])
        DIAG = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)

        # 1, 2: the vertical diameter
        flip([(1, pc["S0"], pc["r0S0"], ORG), (2, pc["S7"], pc["r0S7"], BLU)],
             UPv, C, mirror((0, 1), ext=1.02))     # stays inside y ≤ 3.75
        # 3, 4: the horizontal diameter
        flip([(3, pc["S2"], pc["r90S2"], ORG), (4, pc["S1"], pc["r90S1"], BLU)],
             RIGHTv, C, mirror((1, 0)))
        # 5, 6: the diagonal diameter (first cut the blue leftover at x = q)
        cut_q = outline(pc["L2p"], 1.6)
        self.play(Create(cut_q), run_time=0.5)
        lines_kept.add(cut_q)
        flip([(5, pc["D2"], pc["L1a"], ORG), (6, pc["L2p"], pc["sdL2p"], BLU)],
             DIAG, C, mirror((1, 1)))
        # 7: the horizontal diameter again
        cut_c = outline(pc["L1c"], 1.6)
        self.play(Create(cut_c), run_time=0.5)
        lines_kept.add(cut_c)
        flip([(7, pc["L1c"], pc["r90L1c"], BLU)], RIGHTv, C, mirror((1, 0)))
        # 8, 9: the central parallelogram, halved, folds onto the last two
        cut_m = outline(pc["Pup"], 1.6)
        self.play(Create(cut_m), run_time=0.5)
        lines_kept.add(cut_m)
        flip([(8, pc["Pup"], pc["ParA"], ORG)], RIGHTv, Ps,
             chord((1, 0), YELLOW, 6), rt=1.1)
        c9 = poly(pc["Pdn"], ORG, 0.92, 2)
        dotO = Dot(C, radius=0.05, color=YELLOW_B)
        self.add(c9)
        self.play(FadeIn(dotO), Rotate(c9, angle=PI, about_point=C), run_time=1.1)
        m45 = chord((1, 1), YELLOW, 6)
        self.play(Create(m45), FadeOut(dotO), run_time=0.45)
        self.play(Rotate(c9, angle=PI, axis=DIAG, about_point=Ps), run_time=1.1)
        o1, o2 = outline(pc["Pdn"]), outline(pc["ParB"])
        self.play(FadeOut(c9), FadeIn(o1), FadeIn(o2),
                  FadeIn(number(9, pc["Pdn"])), FadeIn(number(9, pc["ParB"])),
                  FadeOut(m45), run_time=0.7)
        self.bring_to_front(cuts, circle, dotP)

        cap = Text("A(■) = A(■) = ½ πr²", font_size=32, color=YELLOW_B,
                   t2c={"[2:3]": ORG, "[9:10]": BLUE_B})
        cap.to_edge(DOWN, buff=0.3)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.4)


# ======================================================================= J8

class J8_MissingSquare(Board):
    """Curry's missing square — a caution. The same four pieces fill a
    13 × 5 'triangle' twice, once with a hole. Neither figure is a triangle:
    the hypotenuses have slopes 3/8 and 2/5, so the long edge bends, inward
    once and outward once, and the sliver between the two outlines — the
    parallelogram on (5, 2) and (8, 3) — has area |5·3 − 8·2| = 1."""

    def construct(self):
        u = 0.52
        T0 = np.array([-6.3, 0.72, 0.0])          # origin of the upper figure
        B0 = np.array([-6.3, -2.62, 0.0])         # origin of the lower figure

        def Tp(x, y):
            return T0 + u * np.array([x, y, 0.0])

        def Bp(x, y):
            return B0 + u * np.array([x, y, 0.0])

        T8 = [(0, 0), (8, 0), (8, 3)]
        T5 = [(8, 3), (13, 3), (13, 5)]
        A1 = [(8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (8, 1), (9, 1)]
        B1 = [(8, 2), (9, 2), (10, 2), (11, 2), (12, 2), (10, 1), (11, 1), (12, 1)]
        shift = {"T8": (5, 2), "T5": (-8, -3), "A": (-3, 0), "B": (0, -1)}
        A2 = [(x - 3, y) for x, y in A1]
        B2 = [(x, y - 1) for x, y in B1]
        hole = (7, 1)

        # the bookkeeping, checked
        check(set(A1).isdisjoint(B1) and set(A1) | set(B1) ==
              {(x, y) for x in range(8, 13) for y in range(3)}, "A, B fill 5×3")
        strip = {(x, y) for x in range(5, 13) for y in range(2)}
        check(set(A2).isdisjoint(B2) and strip - set(A2) - set(B2) == {hole},
              "A, B fill 8×2 but one cell")
        out1 = [(0, 0), (13, 0), (13, 5), (8, 3)]
        out2 = [(0, 0), (13, 0), (13, 5), (5, 2)]
        check(close(abs(area(out1)), 32) and close(abs(area(out2)), 33),
              "outlines hold 32 and 33 cells")
        check(close(12 + 5 + 7 + 8, 32), "pieces: 12 + 5 + 7 + 8 = 32")
        sliver = [(0, 0), (8, 3), (13, 5), (5, 2)]
        check(close(abs(area(sliver)), 1.0) and abs(5 * 3 - 8 * 2) == 1,
              "sliver area 1")
        check(3 / 8 < 5 / 13 < 2 / 5, "3/8 < 5/13 < 2/5")

        COL = {"T8": RED_D, "T5": BLUE_D, "A": GOLD_D, "B": GREEN_D}
        HOLE = PINK

        def tri(pts, P, col, op=FILL):
            return mk([P(*v) for v in pts], col, op, stroke_width=2)

        def cells(cs, P, col):
            return VGroup(*[mk([P(x, y), P(x + 1, y), P(x + 1, y + 1), P(x, y + 1)],
                               col, FILL, stroke_width=1.5) for x, y in cs])

        def grid(P):
            g = VGroup()
            for x in range(14):
                g.add(Line(P(x, 0), P(x, 5), color=GREY_D, stroke_width=1))
            for y in range(6):
                g.add(Line(P(0, y), P(13, y), color=GREY_D, stroke_width=1))
            return g

        top = {"T8": tri(T8, Tp, COL["T8"]), "T5": tri(T5, Tp, COL["T5"]),
               "A": cells(A1, Tp, COL["A"]), "B": cells(B1, Tp, COL["B"])}
        gT, gB = grid(Tp), grid(Bp)
        d13 = _dim(Tp(0, 0), Tp(13, 0), "13", DOWN * 0.12, GREY_A, 22, gap=0.16)
        d5 = _dim(Tp(13, 0), Tp(13, 5), "5", RIGHT * 0.14, GREY_A, 22, gap=0.16)

        self.play(FadeIn(gT), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(top[k]) for k in ("T8", "A", "B", "T5")],
                              lag_ratio=0.3), FadeIn(d13), FadeIn(d5), run_time=1.8)
        self.hold(0.6)

        # the same pieces slide into the second arrangement below
        self.play(FadeIn(gB), run_time=0.5)
        moved = {k: top[k].copy() for k in ("T8", "T5", "A", "B")}
        self.play(*[moved[k].animate.shift(B0 - T0 + u * np.array([*shift[k], 0.0]))
                    for k in moved], run_time=2.2)
        # check the landed copies against the second arrangement
        check(close(moved["T8"].get_vertices()[0], Bp(5, 2), 1e-6), "T8 lands")
        check(close(moved["T5"].get_vertices()[0], Bp(0, 0), 1e-6), "T5 lands")
        hole_sq = mk([Bp(7, 1), Bp(8, 1), Bp(8, 2), Bp(7, 2)], HOLE, 0.9,
                     stroke_width=3, stroke_color=HOLE)
        self.play(FadeIn(hole_sq), Flash(hole_sq.get_center(), color=HOLE,
                                         line_length=0.25), run_time=0.9)
        self.hold(0.8)

        # ---- the resolution: the long edge is bent
        st_T = DashedLine(Tp(0, 0), Tp(13, 5), color=WHITE, stroke_width=2,
                          dash_length=0.08)
        st_B = DashedLine(Bp(0, 0), Bp(13, 5), color=WHITE, stroke_width=2,
                          dash_length=0.08)
        self.play(Create(st_T), Create(st_B), run_time=0.8)

        # magnified window around the joint (8, 3) of the upper figure
        wx0, wx1, wy0, wy1 = 7.2, 8.8, 2.56, 3.56
        ic = np.array([3.75, 1.55, 0.0])
        m = 6 * u                                 # magnification 6
        wc = np.array([(wx0 + wx1) / 2, (wy0 + wy1) / 2])

        def M(v):
            return ic + m * np.array([v[0] - wc[0], v[1] - wc[1], 0.0])

        def win(pts):
            poly = [np.array(v, float) for v in pts]
            for h in ((1, 0, -wx0), (-1, 0, wx1), (0, 1, -wy0), (0, -1, wy1)):
                poly = _clip(poly, *h)
            return poly

        def inset_poly(pts, col, op=FILL, sw=2):
            return mk([M(v) for v in win(pts)], col, op, stroke_width=sw)

        frame = Rectangle(width=m * (wx1 - wx0), height=m * (wy1 - wy0),
                          color=YELLOW_B, stroke_width=2.5).move_to(ic)
        lens = Rectangle(width=u * (wx1 - wx0), height=u * (wy1 - wy0),
                         color=YELLOW_B, stroke_width=2.5).move_to(Tp(*wc))
        check(close(m / u, 6.0, 1e-9), "the window is magnified exactly 6 times")
        x6 = tag("×6", 24, YELLOW_B).next_to(frame, UP, buff=0.1).align_to(frame, LEFT)
        ins = VGroup(
            inset_poly(T8, COL["T8"]), inset_poly(T5, COL["T5"]),
            inset_poly([(8, 2), (9, 2), (9, 3), (8, 3)], COL["B"], FILL, 1.5),
        )
        sl_in = inset_poly(sliver, HOLE, 0.85, 0)
        # the straight segment through the window
        xs = [wx0, wx1]
        seg = [(x, 5 * x / 13) for x in xs]
        straight = DashedLine(M(seg[0]), M(seg[1]), color=WHITE, stroke_width=2.5,
                              dash_length=0.12)
        s38 = tag("3/8", 26).move_to(M((7.55, 2.67)))
        s25 = tag("2/5", 24).move_to(M((8.6, 3.085)))
        check(_inside_convex(M((7.55, 2.67)), [M(v) for v in win(T8)]),
              "3/8 sits in the red piece")
        self.play(Create(lens), Create(frame), FadeIn(x6), run_time=0.8)
        self.play(FadeIn(ins), run_time=0.7)
        self.play(Create(straight), FadeIn(s38), FadeIn(s25), run_time=0.8)
        self.play(FadeIn(sl_in), run_time=0.7)
        sl_top = tri(sliver, Tp, HOLE, 0.95)
        self.play(FadeIn(sl_top), run_time=0.6)

        r1 = tag("3/8 ≠ 2/5", 32, YELLOW_B)
        r2 = Text("▰ = |5·3 − 8·2| = 1 = ■", font_size=30,
                  t2c={"▰": HOLE, "■": HOLE})
        r1.move_to(np.array([ic[0], -0.75, 0.0]))
        r2.move_to(np.array([ic[0], -1.55, 0.0]))
        self.play(FadeIn(r1), run_time=0.6)
        self.play(FadeIn(r2), run_time=0.8)
        self.play(Write(caption("3/8 ≠ 2/5: the long edge bends, and the sliver "
                                "between the two outlines has area 1", 28)))
        self.hold(2.4)


# ======================================================================= J9

class J9_BolyaiGerwien(Board):
    """Wallace–Bolyai–Gerwien on one triangle. Cut along the midline and turn
    the two top corners half a turn about the side midpoints: a rectangle.
    Too long (6 : 1), so halve it and stack: 3 : 2. Cut along the line from
    one corner to the top of the square of equal area and slide two
    triangles along it: the square. Area is kept at every cut, so every
    polygon (a union of triangles) can be cut into one square, and any two
    polygons of equal area into each other."""

    def construct(self):
        b, h, cx = 7.2, 2.4, 2.6                 # base, height, foot of apex
        k = 1.1
        O = np.array([-6.3, -1.45, 0.0])

        def S(v):
            return O + k * np.array([v[0], v[1], 0.0])

        A, B, Cp = np.array([0.0, 0.0]), np.array([b, 0.0]), np.array([cx, h])
        Mac, Mbc = (A + Cp) / 2, (B + Cp) / 2
        F = np.array([cx, h / 2])
        a2, b2 = b / 2, h                        # the stacked rectangle
        s = np.sqrt(a2 * b2)                     # side of the square

        trap = [A, B, Mbc, Mac]
        top_l = [Mac, F, Cp]
        top_r = [F, Mbc, Cp]
        half = lambda P, c: [2 * c - p for p in P]
        T = abs(area([A, B, Cp]))
        # every step keeps the area — checked before anything moves
        check(close(abs(area(trap)) + abs(area(top_l)) + abs(area(top_r)), T),
              "triangle = trapezoid + two corners")
        check(close(half(top_l, Mac)[2], A) and close(half(top_r, Mbc)[2], B),
              "half-turns carry the apex to the base corners")
        check(close(T, b * h / 2) and close(T, a2 * b2) and close(T, s * s),
              "½bh = b·½h = ½b·h = s²")
        check(b / (h / 2) > 4 >= a2 / b2 >= 1, "6:1 needs halving, 3:2 does not")
        check(b2 <= s <= a2 and s <= 2 * b2, "the slide applies: b ≤ s ≤ a ≤ 4b")
        T1 = [np.array(v) for v in ((a2, 0), (a2, b2), (a2 - s, b2))]
        T2 = [np.array(v) for v in ((s, 0), (a2, 0), (s, s - b2))]
        Q = [np.array(v) for v in ((0, 0), (s, 0), (s, s - b2), (a2 - s, b2), (0, b2))]
        v1 = np.array([-(a2 - s), s - b2])
        v2 = np.array([-s, b2])
        ell = np.array([-a2, s])
        check(abs(v1[0] * ell[1] - v1[1] * ell[0]) < 1e-12 and
              abs(v2[0] * ell[1] - v2[1] * ell[0]) < 1e-12, "both slides run along ℓ")
        check(close(abs(area(Q)) + abs(area(T1)) + abs(area(T2)), s * s),
              "Q + T1 + T2 = s²")
        check(close(T1[2] + v1, (0, s)) and close(T2[0] + v2, (0, b2)),
              "the slid triangles close the square")

        def poly(P, col, op=FILL, sw=2):
            return mk([S(v) for v in P], col, op, stroke_width=sw)

        # ---------------- panel
        x0 = 2.0
        rows = [Text("△ = ½ b h", font_size=30),
                Text("▭ = b · ½h", font_size=30),
                Text("▯ = ½b · h", font_size=30),
                Text("□ = s²,   s = √(½ b h)", font_size=30)]
        notes = [None, tag("6:1 > 4:1", 22, RED_B), tag("3:2 ≤ 4:1", 22, GREEN_B),
                 None]
        _rows(rows, x0, 2.95, 0.8)
        for r, n in zip(rows, notes):
            if n is not None:
                n.next_to(r, RIGHT, buff=0.35)

        # ---------------- the triangle
        tri = poly([A, B, Cp], BLUE_D)
        alt = DashedLine(S(Cp), S((cx, 0)), color=GREY_B, stroke_width=2,
                         dash_length=0.08)
        lb = tag("b", 26).next_to(Line(S(A), S(B)), DOWN, buff=0.15)
        lh = tag("h", 26).next_to(alt, RIGHT, buff=0.1)
        self.play(FadeIn(tri), run_time=0.8)
        self.play(Create(alt), FadeIn(lb), FadeIn(lh), FadeIn(rows[0]), run_time=0.8)
        self.hold(0.4)

        # ---------------- triangle → rectangle
        mid = DashedLine(S(Mac), S(Mbc), color=YELLOW_B, stroke_width=3,
                         dash_length=0.1)
        cut = Line(S(Cp), S(F), color=YELLOW_B, stroke_width=3)
        self.play(Create(mid), Create(cut), FadeOut(alt), FadeOut(lh), run_time=0.8)
        p_tr = poly(trap, BLUE_D)
        p_l = poly(top_l, TEAL_D)
        p_r = poly(top_r, GREEN_D)
        dots = VGroup(Dot(S(Mac), radius=0.06, color=YELLOW_B),
                      Dot(S(Mbc), radius=0.06, color=YELLOW_B))
        self.add(p_tr, p_l, p_r)
        self.remove(tri)
        self.play(FadeOut(mid), FadeOut(cut), FadeIn(dots), run_time=0.4)
        self.play(Rotate(p_l, angle=PI, about_point=S(Mac)),
                  Rotate(p_r, angle=-PI, about_point=S(Mbc)), run_time=1.8)
        got = [p_l.get_vertices()[2], p_r.get_vertices()[2]]
        check(close(got[0], S(A), 1e-6) and close(got[1], S(B), 1e-6),
              "the corners land on A and B")
        rect1 = [A, B, (b, h / 2), (0, h / 2)]
        lh2 = tag("½h", 24).next_to(Line(S(B), S((b, h / 2))), RIGHT, buff=0.12)
        self.play(FadeOut(dots), FadeIn(lh2), FadeIn(rows[1]), FadeIn(notes[1]),
                  run_time=0.7)
        self.hold(0.5)

        # ---------------- 6 : 1 is too long: halve and stack
        R1 = poly(rect1, BLUE_D)
        left = poly([A, (b / 2, 0), (b / 2, h / 2), (0, h / 2)], BLUE_D)
        right = poly([(b / 2, 0), B, (b, h / 2), (b / 2, h / 2)], BLUE_D)
        self.add(R1)
        self.remove(p_tr, p_l, p_r)
        cut2 = DashedLine(S((b / 2, -0.15)), S((b / 2, h / 2 + 0.15)), color=YELLOW_B,
                          stroke_width=3, dash_length=0.1)
        self.play(Create(cut2), run_time=0.5)
        self.add(left, right)
        self.remove(R1)
        self.play(FadeOut(cut2), FadeOut(lh2), FadeOut(lb), run_time=0.3)
        self.play(right.animate.shift(k * np.array([-b / 2, h / 2, 0.0])), run_time=1.3)
        check(close(right.get_vertices()[0], S((0, h / 2)), 1e-6), "the half lands on top")
        lb2 = tag("½b", 24).next_to(Line(S(A), S((a2, 0))), DOWN, buff=0.15)
        lh3 = tag("h", 26).next_to(Line(S((a2, 0)), S((a2, b2))), RIGHT, buff=0.12)
        self.play(FadeIn(lb2), FadeIn(lh3), FadeIn(rows[2]), FadeIn(notes[2]),
                  run_time=0.7)
        self.hold(0.4)

        # ---------------- rectangle → square: cut along ℓ, slide two triangles
        R2 = poly([A, (a2, 0), (a2, b2), (0, b2)], BLUE_D)
        self.add(R2)
        self.remove(left, right)
        l_line = DashedLine(S((a2, 0) - 0.06 * ell), S((0, s) + 0.06 * ell),
                            color=YELLOW_B, stroke_width=3, dash_length=0.1)
        v_cut = Line(S((s, 0)), S((s, s - b2)), color=YELLOW_B, stroke_width=3)
        sq_ghost = Square(side_length=k * s, color=GREY_B, stroke_width=2)
        sq_ghost.move_to(S((s / 2, s / 2)))
        self.play(Create(sq_ghost), FadeOut(lh3), FadeOut(lb2), run_time=0.7)
        self.play(Create(l_line), run_time=0.6)
        self.play(Create(v_cut), run_time=0.4)
        pQ = poly(Q, BLUE_D)
        p1 = poly(T1, ORANGE)
        p2 = poly(T2, GOLD_D)
        self.add(pQ, p1, p2)
        self.remove(R2)
        self.play(FadeOut(v_cut), run_time=0.3)
        self.play(p1.animate.shift(k * np.array([*v1, 0.0])),
                  p2.animate.shift(k * np.array([*v2, 0.0])), run_time=2.0)
        check(close(p1.get_vertices()[2], S((0, s)), 1e-6) and
              close(p2.get_vertices()[0], S((0, b2)), 1e-6), "the square closes")
        sq = Square(side_length=k * s, color=YELLOW_B, stroke_width=5)
        sq.move_to(S((s / 2, s / 2)))
        ls = tag("s", 28, YELLOW_B).next_to(sq, DOWN, buff=0.15)
        ls2 = tag("s", 28, YELLOW_B).next_to(sq, RIGHT, buff=0.15)
        self.play(FadeOut(l_line), FadeOut(sq_ghost), Create(sq), FadeIn(ls),
                  FadeIn(ls2), FadeIn(rows[3]), run_time=0.9)

        summ = Text("P = ∪△ → ∪□ → □ ← Q", font_size=28, color=GREY_A)
        summ.move_to(np.array([x0 + summ.width / 2, -0.75, 0.0]))
        for t in rows + [n for n in notes if n is not None] + [summ]:
            check(t.get_right()[0] <= 6.6 and t.get_left()[0] >= -6.6,
                  "panel text inside the frame")
        self.play(FadeIn(summ), run_time=0.7)
        self.play(Write(caption("area kept at every cut   ⟹   [P] = [Q]  ⇒  P ~ Q",
                                30)))
        self.hold(2.4)
