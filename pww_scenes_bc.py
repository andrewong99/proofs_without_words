# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_bc.py — proofs without words, 2D (manim):
#     B7  geometric series by similar triangles
#     B10 Fibonacci partial sums
#     B11 hockey-stick identity
#     B12 Pascal row sum
#     C2  square of a difference
#     C6  distributive law
#     C7  completing the square
#     J5  handshakes
#     J6  Vandermonde's identity
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Each scene checks its own geometry / counts with check(...), so a wrong
# construction fails the render instead of drawing a wrong picture.

from itertools import combinations


# ============================================== B. SUMS & FIGURATE NUMBERS

class B7_GeometricSimilar(Board):
    """Squares of side 1, r, r², … stand in a row on a base line. Each one
    is the one before it shrunk by r towards a point V of the base, so their
    top-left corners lie on one line through V and the row fills the base up
    to V. The top of the r-square, carried across the 1-square, cuts off a
    small triangle with legs 1 and 1 − r; magnified about its top vertex by
    1/(1 − r) it becomes the whole triangle under that line. Its base is
    therefore 1/(1 − r), and the same base is 1 + r + r² + …  (0 < r < 1)."""

    def construct(self):
        r = 0.6
        X = 1 / (1 - r)                       # where the line meets the base
        K = 11                                # squares drawn
        F = Frame(-0.42, 2.72, -0.36, 1.08)
        P = F.P
        S = [(1 - r ** k) / (1 - r) for k in range(K + 1)]   # left ends

        def sq_pts(k):
            return [(S[k], 0.0), (S[k] + r ** k, 0.0),
                    (S[k] + r ** k, r ** k), (S[k], r ** k)]

        V = np.array([X, 0.0])
        top = np.array([0.0, 1.0])

        def towards_V(p):                     # the homothety: centre V, ratio r
            return V + r * (np.asarray(p, float) - V)

        def blow_up(p):                       # centre (0,1), ratio 1/(1-r)
            return top + (np.asarray(p, float) - top) / (1 - r)

        for k in range(K - 1):
            check(close(S[k] + r ** k, S[k + 1]), "squares abut")
            check(all(close(towards_V(p), q)
                      for p, q in zip(sq_pts(k), sq_pts(k + 1))),
                  f"square {k + 1} = square {k} shrunk by r towards V")
        for k in range(K):
            x, y = sq_pts(k)[3]
            check(close(y, 1 - (1 - r) * x), f"corner {k} on the line")
        small = [(0.0, 1.0), (0.0, r), (1.0, r)]
        big = [(0.0, 1.0), (0.0, 0.0), (X, 0.0)]
        check(all(close(blow_up(p), q) for p, q in zip(small, big)),
              "small triangle x 1/(1-r) = big triangle")
        check(abs(sum(r ** k for k in range(400)) - X) < 1e-9, "series sum")

        cols = [BLUE_D, TEAL_D]
        base = Line(P((-0.05, 0)), P((X + 0.14, 0)), color=GREY_B,
                    stroke_width=2)
        squares = [F.poly(sq_pts(k), cols[k % 2], FILL,
                          stroke_width=2 if k < 5 else 1.2)
                   for k in range(K)]
        names = ["1", "r", "r²", "r³", "r⁴"]
        below = [tag(names[k], 26 if k < 3 else 22).move_to(
                     P((S[k] + r ** k / 2, 0)) + DOWN * 0.34)
                 for k in range(4)]
        dots_lbl = tag("…", 22).move_to(P((S[4] + 0.08, 0)) + DOWN * 0.34)

        self.play(Create(base), run_time=0.7)
        self.play(FadeIn(squares[0]), FadeIn(below[0]), run_time=0.8)
        self.play(TransformFromCopy(squares[0], squares[1]), run_time=1.3)
        self.play(FadeIn(below[1]), run_time=0.4)

        line = Line(P(top), P(V), color=YELLOW_B, stroke_width=4)
        vdot = Dot(P(V), color=YELLOW_B, radius=0.07)
        self.play(Create(line), run_time=1.0)
        self.play(FadeIn(vdot), run_time=0.3)

        for k in range(2, K):
            anims = [TransformFromCopy(squares[k - 1], squares[k])]
            if k < 4:
                anims.append(FadeIn(below[k]))
            if k == 4:
                anims.append(FadeIn(dots_lbl))
            self.play(*anims, run_time=max(0.25, 0.8 - 0.12 * k))
        self.bring_to_front(line, vdot)
        self.hold(0.6)

        # the top of the r-square, carried across the 1-square, cuts off a
        # small triangle: legs 1 (across) and 1 − r (down)
        cut = guide(P((0, r)), P((1, r)), color=YELLOW_B, width=3)
        tri = F.poly(small, ORANGE, 0.9, stroke_color=YELLOW_B,
                     stroke_width=3)
        leg_v = Line(P((0, r)), P((0, 1)), color=YELLOW_B, stroke_width=8)
        leg_h = Line(P((0, r)), P((1, r)), color=YELLOW_B, stroke_width=8)
        lab_v = tag("1−r", 24).move_to(P((0.105, (1 + r) / 2 - 0.01)))
        lab_h = tag("1", 24).move_to(P((0.48, r + 0.055)))
        self.play(Create(cut), run_time=0.6)
        self.play(FadeIn(tri), Create(leg_v), Create(leg_h), run_time=0.8)
        self.play(FadeIn(lab_v), FadeIn(lab_h), run_time=0.5)
        self.hold(0.8)

        # magnify it about its top vertex until its vertical leg is 1:
        # it becomes the whole triangle under the line
        ghost = tri.copy()
        legs = VGroup(leg_v.copy(), leg_h.copy())
        whole = F.poly(big, YELLOW_E, 0.22, stroke_color=YELLOW_B,
                       stroke_width=3)
        self.add(ghost, legs)
        self.bring_to_front(tri, lab_v, lab_h)
        side = Line(P((-0.075, 0)), P((-0.075, 1)), color=YELLOW_B,
                    stroke_width=3)
        side_t = VGroup(*[Line(P((-0.095, y0)), P((-0.055, y0)),
                               color=YELLOW_B, stroke_width=3)
                          for y0 in (0.0, 1.0)])
        lab_1 = tag("1", 26, YELLOW_B).next_to(side, LEFT, buff=0.14)
        dim = Line(P((0, -0.20)), P((X, -0.20)), color=YELLOW_B,
                   stroke_width=3)
        ticks = VGroup(*[Line(P((x0, -0.235)), P((x0, -0.165)),
                              color=YELLOW_B, stroke_width=3)
                         for x0 in (0.0, X)])
        lab_X = tag("1/(1−r)", 26, YELLOW_B).next_to(dim, DOWN, buff=0.12)
        self.play(Transform(ghost, whole),
                  legs.animate.scale(1 / (1 - r), about_point=P(top)),
                  FadeOut(cut), run_time=2.4)
        self.bring_to_front(tri, lab_v, lab_h, line, vdot)
        self.play(Create(side), Create(side_t), FadeIn(lab_1),
                  Create(dim), Create(ticks), FadeIn(lab_X), run_time=1.2)
        self.play(Write(caption("1 + r + r² + r³ + …  =  1/(1−r),"
                                "   0 < r < 1", 30)))
        self.hold(2.2)


class B10_FibonacciSums(Board):
    """Bars of length F₁, F₂, …, Fₙ₊₂ stacked left-aligned, one per row.
    Each row is the row above it plus the row two above it, so the step it
    sticks out past the row above (its tread) is a copy of the row two
    above. The treads, dropped straight down, tile the bottom bar except
    for its first cell:  Fₙ₊₂ = 1 + F₁ + F₂ + … + Fₙ."""

    def construct(self):
        n = 5
        F = [0, 1, 1]
        while len(F) < 30:
            F.append(F[-1] + F[-2])
        for m in range(1, 25):
            check(sum(F[1:m + 1]) == F[m + 2] - 1, f"sum identity, n = {m}")
        R = n + 2                                   # rows 1 .. n+2
        u, gap = 0.66, 0.12
        pitch = u + gap
        x0 = -(F[R] * u + 0.8) / 2
        y_top = SAFE_TOP - u / 2 - 0.08             # centre of row 1

        def yc(row):
            return y_top - (row - 1) * pitch

        cols = {1: BLUE_D, 2: TEAL_D, 3: ORANGE, 4: YELLOW_E, 5: GREEN_D,
                6: RED_D, 7: PURPLE_B}

        def bar(length, start, row, color, op=FILL):
            g = VGroup()
            for i in range(length):
                sq = Square(side_length=u, fill_color=color, fill_opacity=op,
                            stroke_color=WHITE, stroke_width=1.5)
                sq.move_to([x0 + (start + i + 0.5) * u, yc(row), 0.0])
                g.add(sq)
            return g

        subs = "₀₁₂₃₄₅₆₇₈₉"

        def fname(k):
            return "F" + "".join(subs[int(c)] for c in str(k))

        rows, labels = {}, {}
        for r in (1, 2):
            rows[r] = bar(F[r], 0, r, cols[r])
            labels[r] = tag(fname(r), 24, cols[r]).next_to(rows[r], RIGHT,
                                                            buff=0.18)
        self.play(FadeIn(rows[1]), FadeIn(rows[2]),
                  FadeIn(labels[1]), FadeIn(labels[2]), run_time=1.0)

        # row r = (row r-1, slid straight down) + (row r-2, into the step)
        for r in range(3, R + 1):
            check(F[r - 1] + F[r - 2] == F[r], "recurrence")
            a = bar(F[r - 1], 0, r, cols[r - 1])
            b = bar(F[r - 2], F[r - 1], r, cols[r - 2])
            self.play(TransformFromCopy(rows[r - 1], a),
                      TransformFromCopy(rows[r - 2], b), run_time=1.1)
            rows[r] = bar(F[r], 0, r, cols[r])
            labels[r] = tag(fname(r), 24, cols[r]).next_to(rows[r], RIGHT,
                                                            buff=0.18)
            self.play(FadeOut(a), FadeOut(b), FadeIn(rows[r]),
                      FadeIn(labels[r]), run_time=0.45)
        self.hold(0.6)

        # the step lines, down through the bottom bar
        y_bot = yc(R) - u / 2
        steps = VGroup(*[guide([x0 + F[r] * u, yc(r) - u / 2, 0],
                               [x0 + F[r] * u, y_bot, 0], color=YELLOW_B,
                               width=4)
                         for r in range(2, R)])
        self.play(Create(steps), run_time=0.9)

        # every tread is a copy of the row two above ...
        treads = {k: bar(F[k], F[k + 1], k + 2, cols[k], 0.95)
                  for k in range(1, n + 1)}
        # ... and dropped straight down, the treads tile the bottom bar
        drops = {k: bar(F[k], F[k + 1], R, cols[k], 0.95)
                 for k in range(1, n + 1)}
        covered = {0}
        for k in range(1, n + 1):
            seg = set(range(F[k + 1], F[k + 2]))
            check(not (seg & covered), "treads do not overlap")
            covered |= seg
        check(covered == set(range(F[R])), "treads + one cell = bottom bar")
        self.play(*[TransformFromCopy(rows[k], treads[k])
                    for k in range(1, n + 1)], run_time=1.4)
        self.hold(0.5)
        self.play(*[Transform(treads[k], drops[k]) for k in range(1, n + 1)],
                  run_time=1.4)

        # name the pieces under the bar; the whole bar is the last row
        y_nm = y_bot - 0.24
        names = VGroup(tag("1", 24).move_to([x0 + 0.5 * u, y_nm, 0]))
        for k in range(1, n + 1):
            names.add(tag(fname(k), 22, cols[k]).move_to(
                [x0 + (F[k + 1] + F[k] / 2) * u, y_nm, 0]))
        y_dim = y_bot - 0.56
        dim = Line([x0, y_dim, 0], [x0 + F[R] * u, y_dim, 0],
                   color=YELLOW_B, stroke_width=3)
        ticks = VGroup(*[Line([x, y_dim - 0.08, 0], [x, y_dim + 0.08, 0],
                              color=YELLOW_B, stroke_width=3)
                         for x in (x0, x0 + F[R] * u)])
        lab = tag(fname(R), 26, YELLOW_B).next_to(dim, DOWN, buff=0.1)
        first = SurroundingRectangle(rows[R][0], buff=0, color=YELLOW_B,
                                     stroke_width=5)
        self.play(FadeIn(names), Create(first), run_time=0.8)
        self.play(Create(dim), Create(ticks), FadeIn(lab), run_time=0.7)
        self.play(Write(caption("F₁ + F₂ + … + Fₙ  =  Fₙ₊₂ − 1", 32)))
        self.hold(2.2)


def _lattice_paths(W, H):
    """All east/north lattice paths from (0,0) to (W,H), as point lists,
    in lexicographic order of their step words (E before N)."""
    out = []
    for Es in combinations(range(W + H), W):
        pts = [(0, 0)]
        for i in range(W + H):
            x, y = pts[-1]
            pts.append((x + 1, y) if i in Es else (x, y + 1))
        out.append(pts)
    return out


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


def _polyline(pts, color, width):
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _grid_lines(W, H, cell, origin, color=GREY_B, width=1.5):
    o = to3(origin)
    g = VGroup()
    for i in range(W + 1):
        g.add(Line(o + [i * cell, 0, 0], o + [i * cell, H * cell, 0],
                   stroke_color=color, stroke_width=width))
    for j in range(H + 1):
        g.add(Line(o + [0, j * cell, 0], o + [W * cell, j * cell, 0],
                   stroke_color=color, stroke_width=width))
    return g


class B11_HockeyStick(Board):
    """Paths of unit steps east and north from O to the corner (r+1, n−r)
    number C(n+1, r+1). Every such path takes its LAST east step through
    exactly one gate, from (r, j) to (r+1, j), and then only climbs. The
    paths through gate j are the paths to (r, j) — C(r+j, r) of them — so
    the gates count C(r,r) + C(r+1,r) + … + C(n,r).  Shown for r = 2,
    n = 5; the numbers at the lattice points are path counts (Pascal)."""

    def construct(self):
        r, n = 2, 5
        W, H = r + 1, n - r
        for nn in range(12):
            for rr in range(nn + 1):
                check(sum(_binom(k, rr) for k in range(rr, nn + 1))
                      == _binom(nn + 1, rr + 1), "hockey-stick identity")
        paths = _lattice_paths(W, H)
        check(len(paths) == _binom(n + 1, r + 1), "all paths listed")
        groups = {j: [] for j in range(H + 1)}
        for pts in paths:
            last = max(i for i in range(W + H)
                       if pts[i + 1][0] == pts[i][0] + 1)
            j = pts[last][1]
            check(pts[last] == (r, j) and pts[last + 1] == (r + 1, j),
                  "last east step is a gate")
            check(all(p[0] == r + 1 for p in pts[last + 1:]),
                  "after the gate the path only climbs")
            groups[j].append((pts, last))
        for j in range(H + 1):
            check(len(groups[j]) == _binom(r + j, r), f"gate {j} count")

        cb = 1.55                                   # big-grid cell
        O = np.array([-6.1, -1.95, 0.0])

        def B(p):
            return O + np.array([p[0] * cb, p[1] * cb, 0.0])

        gcol = [BLUE_D, GREEN_D, ORANGE, RED_D]
        grid = _grid_lines(W, H, cb, O, GREY_B, 2)
        dO = Dot(B((0, 0)), color=WHITE, radius=0.08)
        dT = Dot(B((W, H)), color=YELLOW_B, radius=0.1)
        lO = tag("O", 24).next_to(dO, DOWN + LEFT, buff=0.08)
        self.play(Create(grid), FadeIn(dO), FadeIn(dT), FadeIn(lO),
                  run_time=1.0)

        # path counts at the lattice points: Pascal's triangle
        nums = {}
        for s in range(W + H + 1):
            for x in range(W + 1):
                y = s - x
                if 0 <= y <= H:
                    col = GREY_A
                    if x == r:
                        col = gcol[y]
                    if (x, y) == (W, H):
                        col = YELLOW_B
                    nums[(x, y)] = tag(str(_binom(x + y, x)), 22, col).move_to(
                        B((x, y)) + np.array([-0.30, 0.25, 0.0]))
        self.play(LaggedStart(*[FadeIn(nums[(x, s - x)])
                                for s in range(W + H + 1)
                                for x in range(W + 1) if 0 <= s - x <= H],
                              lag_ratio=0.08), run_time=1.6)

        # the gates: the last east step, from (r, j) to (r+1, j)
        gates = VGroup(*[Line(B((r, j)), B((r + 1, j)), color=gcol[j],
                              stroke_width=10) for j in range(H + 1)])
        climb = guide(B((W, 0)), B((W, H)), color=WHITE, width=4)
        self.play(Create(gates), Create(climb), run_time=1.0)

        # minis: every path, filed beside its gate (5 to a row at most)
        cm, mp, rows_gap = 0.28, 1.0, 0.12
        x_start = B((W, 0))[0] + 0.75

        def mini(pts, last, j, k):
            size = len(groups[j])
            nrow = 1 if size <= 6 else 2
            ncol = -(-size // nrow)
            row, col = divmod(k, ncol)
            yc0 = B((0, j))[1] + (nrow - 1) * ((H * cm + rows_gap) / 2 + 0.1)
            org = np.array([x_start + col * mp,
                            yc0 - row * (H * cm + rows_gap) - H * cm / 2, 0.0])
            g = VGroup(_grid_lines(W, H, cm, org, GREY_D, 1))
            pre = [org + np.array([p[0] * cm, p[1] * cm, 0]) for p in
                   pts[:last + 2]]
            post = [org + np.array([p[0] * cm, p[1] * cm, 0]) for p in
                    pts[last + 1:]]
            g.add(_polyline(pre, gcol[j], 3.5))
            if len(post) > 1:
                g.add(_polyline(post, WHITE, 2.5))
            return g

        for j in range(H + 1):
            pts, last = groups[j][0]
            sample = VGroup(_polyline([B(p) for p in pts[:last + 2]],
                                      gcol[j], 8))
            if last + 2 <= len(pts) - 1:
                sample.add(_polyline([B(p) for p in pts[last + 1:]],
                                     WHITE, 5))
            minis = [mini(p, l, j, k) for k, (p, l) in enumerate(groups[j])]
            cnt = tag(str(len(minis)), 30, gcol[j]).next_to(
                VGroup(*minis), RIGHT, buff=0.25)
            self.play(Create(sample), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(m) for m in minis],
                                  lag_ratio=0.12),
                      FadeIn(cnt), run_time=0.6 + 0.08 * len(minis))
            self.play(FadeOut(sample), run_time=0.3)

        # the hockey stick in Pascal's triangle
        stick = VGroup(*[SurroundingRectangle(nums[(r, j)], buff=0.05,
                                              color=gcol[j], stroke_width=3)
                         for j in range(H + 1)])
        blade = SurroundingRectangle(nums[(W, H)], buff=0.05,
                                     color=YELLOW_B, stroke_width=3)
        parts = [tag(f"C({r + j},{r})", 24, gcol[j]) for j in range(H + 1)]
        eq = VGroup(parts[0])
        for p in parts[1:]:
            eq.add(tag("+", 24), p)
        eq.add(tag("=", 24), tag(f"C({n + 1},{r + 1})", 24, YELLOW_B))
        eq.arrange(RIGHT, buff=0.16).move_to([1.6, -2.72, 0])
        self.play(Create(stick), Create(blade), run_time=0.8)
        self.play(FadeIn(eq), run_time=0.8)
        self.play(Write(caption("C(r,r) + C(r+1,r) + … + C(n,r)  =  "
                                "C(n+1, r+1)", 30)))
        self.hold(2.2)


class B12_PascalRowSum(Board):
    """Every subset of an n-element set is a row of n cells, each cell
    lit (in) or dark (out). Deciding the elements one at a time doubles the
    list each time: 2ⁿ subsets. Sorting the same subsets by how many cells
    are lit puts C(n,k) of them in column k.  Shown for n = 4."""

    def construct(self):
        n = 4
        N = 2 ** n
        for nn in range(14):
            check(sum(_binom(nn, k) for k in range(nn + 1)) == 2 ** nn,
                  f"row sum, n = {nn}")
        ecol = [BLUE_D, GREEN_D, ORANGE, RED_D]
        s = 0.40
        ws = n * s
        px, py = ws + 0.32, s + 0.26
        gx0 = -5.6 + ws / 2                     # centre x of grid column 0
        gy0 = 3.6 - s / 2                       # centre y of grid row 0

        def bits(m):
            return [(m >> i) & 1 for i in range(n)]

        def gpos(m):
            b = bits(m)
            c, r = b[0] + 2 * b[2], b[1] + 2 * b[3]
            return np.array([gx0 + c * px, gy0 - r * py, 0.0])

        def strip(m, centre):
            g = VGroup()
            for i, on in enumerate(bits(m)):
                sq = Square(side_length=s, stroke_color=WHITE if on else GREY_B,
                            stroke_width=1.5, fill_color=ecol[i] if on else BLACK,
                            fill_opacity=0.95 if on else 1.0)
                sq.move_to(centre + np.array([(i - (n - 1) / 2) * s, 0, 0]))
                g.add(sq)
            return g

        check(len({tuple(np.round(gpos(m), 6)) for m in range(N)}) == N,
              "one grid place per subset")

        grid = {0: strip(0, gpos(0))}
        prod_txt = ["1", "2", "2 × 2", "2 × 2 × 2", "2 × 2 × 2 × 2  =  2⁴"]
        tag_at = np.array([gx0 + 3 * px + ws / 2 + 0.35, gy0 - 1.5 * py, 0.0])
        count = tag(prod_txt[0], 28, YELLOW_B)
        count.move_to(tag_at, aligned_edge=LEFT)
        self.play(FadeIn(grid[0]), FadeIn(count), run_time=0.8)

        # decide element d: copy every subset so far, switch element d on
        for d in range(n):
            olds = sorted(grid)
            news = {m | (1 << d): strip(m | (1 << d), gpos(m | (1 << d)))
                    for m in olds}
            for m in olds:
                check((m | (1 << d)) not in grid, "copies are new subsets")
            new_count = tag(prod_txt[d + 1], 28, YELLOW_B)
            new_count.move_to(tag_at, aligned_edge=LEFT)
            self.play(*[TransformFromCopy(grid[m], news[m | (1 << d)])
                        for m in olds],
                      Transform(count, new_count), run_time=1.1)
            grid.update(news)
            self.hold(0.25)
        check(len(grid) == N, "2^n subsets")
        self.hold(0.5)

        # the same subsets, sorted by size
        cx = {k: (k - n / 2) * 1.9 for k in range(n + 1)}
        yb = -2.3
        heads = VGroup()
        for k in range(n + 1):
            members = [m for m in range(N) if sum(bits(m)) == k]
            check(len(members) == _binom(n, k), f"size-{k} column")
            targets = [strip(m, np.array([cx[k], yb + s / 2 + i * (s + 0.12),
                                          0.0]))
                       for i, m in enumerate(members)]
            lab = tag(f"C({n},{k})", 24).move_to([cx[k], yb - 0.32, 0])
            heads.add(lab)
            extra = []
            if k > 0:
                extra.append(FadeIn(tag("+", 26).move_to(
                    [(cx[k - 1] + cx[k]) / 2, yb - 0.32, 0])))
            self.play(*[TransformFromCopy(grid[m], t)
                        for m, t in zip(members, targets)],
                      FadeIn(lab), *extra, run_time=1.1)
        self.hold(0.4)
        self.play(Write(caption("C(n,0) + C(n,1) + … + C(n,n)  =  2ⁿ", 32)))
        self.hold(2.2)


# ================================================== C. ALGEBRAIC IDENTITIES

def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _dashed_outline(pts, color, width=3, dashes=24):
    return DashedVMobject(Polygon(*[to3(p) for p in pts], stroke_color=color,
                                  stroke_width=width), num_dashes=dashes)


class C2_SquareOfDifference(Board):
    """Lift the top strip (a × b) off the a-square: it takes the corner b²
    with it. The right strip (b × a) needs that corner too — it would be
    removed twice — so a b² is added back to complete it, and the second
    strip comes off. What is left is (a − b)²:
        a² + b² − ab − ab = (a − b)²."""

    def construct(self):
        a, b, g = 5.0, 1.7, 0.45
        d = a - b
        # room on the right for the bracketed "ab" of the moved strip
        F = Frame(-1.15, a + b + g + 1.2, -0.95, a + b + g + 0.15)
        P, k = F.P, F.k
        A1, A2 = _rect(0, 0, d, d), _rect(0, d, d, b)        # (a-b)², top-left
        A3, A4 = _rect(d, 0, b, d), _rect(d, d, b, b)        # right, corner b²
        check(abs(sum(area(p) for p in (A1, A2, A3, A4)) - a * a) < 1e-9,
              "four pieces tile a²")
        check(abs(area(A2) + area(A4) - a * b) < 1e-9, "top strip is ab")
        check(abs(area(A3) + b * b - a * b) < 1e-9,
              "right strip is ab once the corner is put back")
        check(abs(area(A1) + 2 * a * b - (a * a + b * b)) < 1e-9,
              "(a-b)² + 2ab = a² + b²")

        pieces = [F.poly(A, BLUE_D) for A in (A1, A2, A3, A4)]
        p1, p2, p3, p4 = pieces
        whole_sq = F.poly(_rect(0, 0, a, a), BLUE_D)
        lab_a2 = tag("a²", 36).move_to(P((a / 2, a / 2)))
        la_b = tag("a", 28).move_to(P((a / 2, -0.45)))
        la_l = tag("a", 28).move_to(P((-0.5, a / 2)))
        self.play(FadeIn(whole_sq), FadeIn(lab_a2), FadeIn(la_b),
                  FadeIn(la_l), run_time=1.2)
        self.hold(0.6)

        # cut at a − b, both ways
        cuts = VGroup(Line(P((d, 0)), P((d, a)), color=WHITE, stroke_width=2),
                      Line(P((0, d)), P((a, d)), color=WHITE, stroke_width=2))
        ghost = _dashed_outline([P(p) for p in _rect(0, 0, a, a)], GREY_B, 2,
                                40)
        self.play(Create(cuts), FadeOut(lab_a2), run_time=0.8)
        self.remove(cuts, whole_sq)
        self.add(ghost, *pieces)

        # the top strip, corner and all, comes off
        lab_s1 = tag("ab", 30).move_to(P((a / 2, d + b / 2)))
        lab_b1 = tag("b", 26).move_to(P((-0.42, d + b / 2)))
        self.play(p2.animate.set_fill(TEAL_D), p4.animate.set_fill(TEAL_D),
                  FadeIn(lab_s1), FadeIn(lab_b1), run_time=0.6)
        strip1 = VGroup(p2, p4, lab_s1, lab_b1)
        self.play(strip1.animate.shift(UP * (b + g) * k), run_time=1.2)

        # the right strip: its corner has already gone
        frame2 = Polygon(*[P(p) for p in _rect(d, 0, b, a)],
                         stroke_color=ORANGE, stroke_width=5)
        hole = _dashed_outline([P(p) for p in A4], RED_B, 4, 16)
        self.play(p3.animate.set_fill(ORANGE), Create(frame2), Create(hole),
                  run_time=0.8)
        self.hold(0.5)
        # ... so a b² is added back
        y = F.poly(A4, YELLOW_E, 0.95)
        lab_y = tag("b²", 28, BLACK).move_to(y)
        yg = VGroup(y, lab_y).shift(RIGHT * 2.2 * k + UP * 0.4 * k)
        plus = tag("+", 34, YELLOW_B).next_to(yg, LEFT, buff=0.15)
        self.play(FadeIn(yg), FadeIn(plus), run_time=0.6)
        self.play(yg.animate.shift(LEFT * 2.2 * k + DOWN * 0.4 * k),
                  FadeOut(plus), run_time=1.0)
        self.remove(hole)

        # the completed strip (orange + the b² put back) is b × a = ab:
        # a bracket along its whole height carries the label
        xb, tk = a + 0.26, 0.14
        bracket = VGroup(
            Line(P((xb, 0)), P((xb, a)), color=ORANGE, stroke_width=3),
            Line(P((xb - tk, 0)), P((xb, 0)), color=ORANGE, stroke_width=3),
            Line(P((xb - tk, a)), P((xb, a)), color=ORANGE, stroke_width=3))
        lab_s2 = tag("ab", 30).next_to(bracket, RIGHT, buff=0.14)
        lab_b2 = tag("b", 26).move_to(P((d + b / 2, -0.42)))
        self.play(Create(bracket), FadeIn(lab_s2), FadeIn(lab_b2),
                  run_time=0.6)
        strip2 = VGroup(p3, yg, frame2, bracket, lab_s2, lab_b2)
        self.play(strip2.animate.shift(RIGHT * (b + g) * k), run_time=1.2)
        check(close(p3.get_corner(DL), P((d + b + g, 0))), "strip 2 landed")
        check(lab_s2.get_right()[0] < SAFE_X, "strip label inside the frame")

        rest = Polygon(*[P(p) for p in A1], stroke_color=YELLOW_B,
                       stroke_width=5)
        lab_r = tag("(a−b)²", 32).move_to(P((d / 2, d / 2)))
        self.play(Create(rest), FadeIn(lab_r), run_time=0.8)
        self.play(Write(caption("(a − b)²  =  a² − 2ab + b²", 34)))
        self.hold(2.2)


class C6_Distributive(Board):
    """One rectangle of height a and width b + c, cut into an a × b and an
    a × c rectangle: the same area counted whole and in two parts."""

    def construct(self):
        a, b, c, g = 3.0, 4.2, 2.6, 1.4
        F = Frame(-g / 2 - 0.85, b + c + g / 2 + 0.35, -1.45, a + 0.35)
        P, k = F.P, F.k
        R1, R2 = _rect(0, 0, b, a), _rect(b, 0, c, a)
        check(abs(area(R1) + area(R2) - a * (b + c)) < 1e-9, "areas add")

        outer = Polygon(*[P(p) for p in _rect(0, 0, b + c, a)],
                        stroke_color=YELLOW_B, stroke_width=5)
        r1, r2 = F.poly(R1, BLUE_D), F.poly(R2, TEAL_D)
        la = tag("a", 30).move_to(P((-0.42, a / 2)))
        dim = Line(P((0, -0.8)), P((b + c, -0.8)), color=YELLOW_B,
                   stroke_width=3)
        ticks = VGroup(*[Line(P((x, -0.88)), P((x, -0.72)), color=YELLOW_B,
                              stroke_width=3) for x in (0, b + c)])
        lbc = tag("b + c", 30, YELLOW_B).next_to(dim, DOWN, buff=0.1)
        self.play(Create(outer), FadeIn(la), Create(dim), Create(ticks),
                  FadeIn(lbc), run_time=1.4)
        self.play(FadeIn(r1), FadeIn(r2), run_time=0.8)
        cut = Line(P((b, 0)), P((b, a)), color=WHITE, stroke_width=4)
        lb = tag("b", 28).move_to(P((b / 2, -0.38)))
        lc = tag("c", 28).move_to(P((b + c / 2, -0.38)))
        self.play(Create(cut), FadeIn(lb), FadeIn(lc), run_time=0.8)
        self.hold(0.5)

        # pull the two pieces apart: two rectangles, each of height a
        lab1 = tag("ab", 34).move_to(P((b / 2, a / 2)))
        lab2 = tag("ac", 34).move_to(P((b + c / 2, a / 2)))
        g1, g2 = VGroup(r1, lab1, lb), VGroup(r2, lab2, lc)
        self.play(FadeOut(cut), FadeOut(outer),
                  g1.animate.shift(LEFT * g / 2 * k),
                  g2.animate.shift(RIGHT * g / 2 * k),
                  la.animate.shift(LEFT * g / 2 * k), run_time=1.2)
        la2 = tag("a", 30).move_to(P((b + g / 2 - 0.36, a / 2)))
        check(r2.get_left()[0] > la2.get_right()[0] + 0.05 and
              r1.get_right()[0] < la2.get_left()[0] - 0.05,
              "second a sits in the gap")
        self.play(FadeIn(lab1), FadeIn(lab2), FadeIn(la2), run_time=0.6)
        self.hold(1.0)

        # and back: together they are the whole rectangle again
        self.play(g1.animate.shift(RIGHT * g / 2 * k),
                  g2.animate.shift(LEFT * g / 2 * k),
                  la.animate.shift(RIGHT * g / 2 * k), FadeOut(la2),
                  run_time=1.2)
        check(close(r1.get_corner(DR), r2.get_corner(DL)), "pieces abut")
        self.play(Create(outer), run_time=0.8)
        self.play(Write(caption("a(b + c)  =  ab + ac", 36)))
        self.hold(2.2)


class C7_CompletingSquare(Board):
    """Cut the b × x strip beside the x-square into two strips of width
    b/2. One stays; the other folds over a quarter-turn on its top corner
    and slides along the top of the square. The two strips and the square
    make (x + b/2)² short of one corner (b/2)²:
        x² + bx = (x + b/2)² − (b/2)²."""

    def construct(self):
        x, bb = 3.6, 2.4
        h = bb / 2
        F = Frame(-1.55, x + bb + 0.25, -1.45, x + h + 0.2)
        P = F.P
        SQ = _rect(0, 0, x, x)
        S1, S2 = _rect(x, 0, h, x), _rect(x + h, 0, h, x)
        TOP = _rect(0, x, x, h)
        CORNER = _rect(x, x, h, h)
        hinge = np.array([x + bb, x])          # top-right corner of the strip

        def fold(p):                          # −90° about the hinge, then slide
            v = np.asarray(p, float) - hinge
            return hinge + np.array([v[1], -v[0]]) - np.array([x - h, 0.0])

        check(sorted(tuple(np.round(fold(p), 9)) for p in S2) ==
              sorted(tuple(np.round(p, 9)) for p in TOP),
              "fold + slide lands the strip on top of the square")
        check(abs(area(SQ) + area(S1) + area(TOP) + area(CORNER)
                  - (x + h) ** 2) < 1e-9, "L plus corner = (x+b/2)²")
        check(abs(area(SQ) + area(S1) + area(S2) - (x * x + bb * x)) < 1e-9,
              "x² + bx")

        sq = F.poly(SQ, BLUE_D)
        rect = F.poly(_rect(x, 0, bb, x), TEAL_D)
        lx2 = tag("x²", 36).move_to(P((x / 2, x / 2)))
        lbx = tag("bx", 34).move_to(P((x + h, x / 2)))
        lxl = tag("x", 28).move_to(P((-0.4, x / 2)))
        lxb = tag("x", 28).move_to(P((x / 2, -0.38)))
        lb = tag("b", 28).move_to(P((x + h, -0.38)))
        self.play(FadeIn(sq), FadeIn(lx2), FadeIn(lxl), FadeIn(lxb),
                  run_time=1.0)
        self.play(FadeIn(rect), FadeIn(lbx), FadeIn(lb), run_time=0.9)
        self.hold(0.5)

        # cut the strip in half
        s1, s2 = F.poly(S1, TEAL_D), F.poly(S2, TEAL_D)
        cut = guide(P((x + h, 0)), P((x + h, x)), WHITE, width=3)
        lh1 = tag("b/2", 26).move_to(P((x + h / 2, -0.38)))
        lh2 = tag("b/2", 26).move_to(P((x + 1.5 * h, -0.38)))
        in1 = tag("½bx", 28).move_to(P((x + h / 2, x / 2)))
        in2 = tag("½bx", 28).move_to(P((x + 1.5 * h, x / 2)))
        self.play(Create(cut), FadeOut(lbx), FadeOut(lb), run_time=0.6)
        self.remove(rect)
        self.add(s1, s2, cut)
        self.play(FadeIn(lh1), FadeIn(lh2), FadeIn(in1), FadeIn(in2),
                  run_time=0.6)
        self.hold(0.5)

        # one half folds over a quarter-turn on its top corner, then slides
        # along the top of the square
        self.remove(cut)
        piv = P(hinge)
        turn = Arc(radius=0.6, start_angle=-PI / 2, angle=-PI / 2,
                   arc_center=piv, color=YELLOW_B, stroke_width=5)
        deg = tag("90°", 24, YELLOW_B).next_to(piv, RIGHT, buff=0.18)
        pdot = Dot(piv, color=YELLOW_B, radius=0.07)
        self.play(FadeOut(lh2), FadeOut(in2), FadeIn(pdot),
                  Create(turn), FadeIn(deg), run_time=0.5)
        self.play(Rotate(s2, angle=-PI / 2, about_point=piv), run_time=1.6)
        self.play(FadeOut(pdot), FadeOut(turn), FadeOut(deg), run_time=0.3)
        rail = guide(P((0, x + h)), P((x + bb, x + h)), GREY_A)
        self.play(Create(rail), run_time=0.3)
        self.play(s2.animate.shift(LEFT * (x - h) * F.k), run_time=1.2)
        self.play(FadeOut(rail), run_time=0.3)
        check(close(s2.get_corner(DL), P((0, x))) and
              close(s2.get_corner(UR), P((x, x + h))), "strip on top")
        in3 = tag("½bx", 28).move_to(P((x / 2, x + h / 2)))
        lh3 = tag("b/2", 26).move_to(P((-0.45, x + h / 2)))
        self.play(FadeIn(in3), FadeIn(lh3), run_time=0.5)
        self.hold(0.4)

        # the missing corner, and the completed square
        corner = _dashed_outline([P(p) for p in CORNER], YELLOW_B, 4, 16)
        cfill = F.poly(CORNER, YELLOW_E, 0.25, stroke_width=0)
        lc = tag("(b/2)²", 24, YELLOW_B).move_to(P((x + h / 2, x + h / 2)))
        self.play(FadeIn(cfill), Create(corner), FadeIn(lc), run_time=0.9)
        big = Polygon(*[P(p) for p in _rect(0, 0, x + h, x + h)],
                      stroke_color=YELLOW_B, stroke_width=5)
        dim = Line(P((0, -0.85)), P((x + h, -0.85)), color=YELLOW_B,
                   stroke_width=3)
        ticks = VGroup(*[Line(P((t, -0.93)), P((t, -0.77)), color=YELLOW_B,
                              stroke_width=3) for t in (0, x + h)])
        lxh = tag("x + b/2", 28, YELLOW_B).next_to(dim, DOWN, buff=0.1)
        self.play(Create(big), Create(dim), Create(ticks), FadeIn(lxh),
                  run_time=1.0)
        self.play(Write(caption("x² + bx  =  (x + b/2)² − (b/2)²", 34)))
        self.hold(2.2)


# ================================================ J. NUMBER THEORY & CURIOS

class J5_Handshakes(Board):
    """n people; the n × n grid holds every ordered pair (row, column).
    Strike the diagonal (nobody shakes their own hand): n² − n cells are
    left. Reflection in the diagonal swaps (i, j) and (j, i), so the two
    staircases left over are congruent, and each handshake {i, j} is one
    cell of the upper staircase:  C(n,2) = n(n − 1)/2."""

    def construct(self):
        n = 6
        pcol = [BLUE_D, TEAL_D, GREEN_D, GOLD_D, ORANGE, RED_D]
        ctr = np.array([-4.25, 0.25, 0.0])
        R = 1.9
        pos = [ctr + R * np.array([np.cos(PI / 2 - TAU * i / n),
                                   np.sin(PI / 2 - TAU * i / n), 0.0])
               for i in range(n)]

        def person(i, at, rad=0.27, size=22):
            return VGroup(Circle(radius=rad, fill_color=pcol[i],
                                 fill_opacity=1.0, stroke_color=WHITE,
                                 stroke_width=2).move_to(at),
                          tag(str(i + 1), size).move_to(at))

        people = [person(i, pos[i]) for i in range(n)]
        self.play(LaggedStart(*[FadeIn(p) for p in people], lag_ratio=0.15),
                  run_time=1.0)

        u = 0.7
        gx, gy = 0.95, 2.75                      # top-left corner of the grid

        def cpos(i, j):
            return np.array([gx + (j + 0.5) * u, gy - (i + 0.5) * u, 0.0])

        rows = [person(i, np.array([gx - 0.42, cpos(i, 0)[1], 0.0]), 0.22, 18)
                for i in range(n)]
        cols = [person(j, np.array([cpos(0, j)[0], gy + 0.42, 0.0]), 0.22, 18)
                for j in range(n)]
        cells = {(i, j): Square(side_length=u, fill_color=GREY_D,
                                fill_opacity=0.9, stroke_color=WHITE,
                                stroke_width=1.5).move_to(cpos(i, j))
                 for i in range(n) for j in range(n)}
        grid = VGroup(*cells.values())
        ln = tag("n", 30, YELLOW_B).next_to(grid, DOWN, buff=0.2)
        lnr = tag("n", 30, YELLOW_B).next_to(grid, RIGHT, buff=0.22)
        self.play(*[TransformFromCopy(people[i], rows[i]) for i in range(n)],
                  *[TransformFromCopy(people[j], cols[j]) for j in range(n)],
                  run_time=1.3)
        self.play(FadeIn(grid), FadeIn(ln), FadeIn(lnr), run_time=0.8)
        self.hold(0.5)

        # nobody shakes their own hand
        crosses = VGroup(*[tag("×", 34, RED_B).move_to(cpos(i, i))
                           for i in range(n)])
        self.play(*[cells[(i, i)].animate.set_fill(BLACK, 1.0)
                    for i in range(n)], FadeIn(crosses), run_time=0.8)

        upper = [(i, j) for i in range(n) for j in range(n) if i < j]
        lower = [(i, j) for i in range(n) for j in range(n) if i > j]
        check(len(upper) == len(lower) == n * (n - 1) // 2, "two halves")
        self.play(*[cells[c].animate.set_fill(TEAL_D, 0.85) for c in upper],
                  *[cells[c].animate.set_fill(ORANGE, 0.85) for c in lower],
                  run_time=0.8)
        self.hold(0.4)

        # reflect the lower staircase in the diagonal: it covers the upper
        centre = np.array([gx + n * u / 2, gy - n * u / 2, 0.0])
        axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        for i, j in lower:
            v = cpos(i, j) - centre
            check(close(centre + 2 * np.dot(v, axis) * axis - v, cpos(j, i)),
                  "mirror cell lands on its transpose")
        diag = DashedLine(cpos(0, 0) + np.array([-u / 2, u / 2, 0]),
                          cpos(n - 1, n - 1) + np.array([u / 2, -u / 2, 0]),
                          color=YELLOW_B, stroke_width=3, dash_length=0.1)
        flip = VGroup(*[cells[c].copy() for c in lower])
        self.play(Create(diag), run_time=0.5)
        self.play(Rotate(flip, angle=PI, axis=axis, about_point=centre),
                  run_time=1.8)
        self.hold(0.4)
        self.play(FadeOut(flip), FadeOut(diag), run_time=0.5)

        # one cell of the upper staircase for each handshake
        edges = []
        anims = []
        for i, j in upper:
            e = Line(pos[i], pos[j], color=TEAL_B, stroke_width=3)
            edges.append(e)
            anims.append(AnimationGroup(
                Create(e),
                cells[(i, j)].animate.set_fill(TEAL_C, 1.0).set_stroke(
                    WHITE, 3)))
        self.add(*edges)                         # beneath the people
        self.bring_to_front(*people)
        self.play(LaggedStart(*anims, lag_ratio=0.35), run_time=4.0)
        self.bring_to_front(*people)
        self.play(Write(caption("C(n,2)  =  (n² − n)/2  =  n(n − 1)/2", 32)))
        self.hold(2.2)


class J6_Vandermonde(Board):
    """Lattice paths from O to (r, m+n−r) take m + n steps, r of them east:
    C(m+n, r) paths. After exactly m steps every path is on the line
    x + y = m, at the point (k, m−k) where k is the number of east steps
    so far. The paths through that point are any of the C(m,k) ways in
    followed by any of the C(n, r−k) ways on — a full table of them. Sum
    over k.  Shown for m = 3, n = 2, r = 2."""

    def construct(self):
        m, n, r = 3, 2, 2
        W, H = r, m + n - r
        for M in range(6):
            for N in range(6):
                for RR in range(M + N + 1):
                    check(sum(_binom(M, k) * _binom(N, RR - k)
                              for k in range(RR + 1)) == _binom(M + N, RR),
                          "Vandermonde")
        paths = [tuple(p) for p in _lattice_paths(W, H)]
        check(len(paths) == _binom(m + n, r), "all paths")
        ks = sorted({p[m][0] for p in paths})
        table = {}
        for k in ks:
            g = [p for p in paths if p[m][0] == k]
            check(all(p[m] == (k, m - k) for p in g), "on the line after m")
            firsts = sorted({p[:m + 1] for p in g})
            seconds = sorted({p[m:] for p in g})
            check(len(firsts) == _binom(m, k) and
                  len(seconds) == _binom(n, r - k), "ways in, ways on")
            check({f + s[1:] for f in firsts for s in seconds} == set(g),
                  "every (in, on) pair is exactly one path")
            table[k] = (firsts, seconds)

        gcol = {0: BLUE_D, 1: GREEN_D, 2: ORANGE}
        cb = 1.35
        O = np.array([-6.0, -1.9, 0.0])

        def B(p):
            return O + np.array([p[0] * cb, p[1] * cb, 0.0])

        grid = _grid_lines(W, H, cb, O, GREY_B, 2)
        dO = Dot(B((0, 0)), radius=0.08)
        dT = Dot(B((W, H)), color=YELLOW_B, radius=0.1)
        lO = tag("O", 24).next_to(dO, DOWN + LEFT, buff=0.06)
        lT = tag("(r, m+n−r)", 22, YELLOW_B).next_to(dT, UP, buff=0.14)
        self.play(Create(grid), FadeIn(dO), FadeIn(dT), FadeIn(lO),
                  FadeIn(lT), run_time=1.0)
        par = tag("m = 3,  n = 2,  r = 2", 24, GREY_A).move_to([-4.7, 3.45, 0])

        cut = DashedLine(B((-0.3, m + 0.3)), B((m - 0.7, 0.7)),
                         color=YELLOW_B, stroke_width=3, dash_length=0.1)
        lcut = tag("x + y = m", 22, YELLOW_B).move_to(
            B((-0.3, m + 0.3)) + np.array([0.85, 0.36, 0.0]))
        cross = {k: Dot(B((k, m - k)), color=gcol[k], radius=0.11)
                 for k in ks}
        self.play(Create(cut), FadeIn(lcut), FadeIn(par),
                  *[FadeIn(d) for d in cross.values()], run_time=1.0)

        cm, pxm, pym, hg = 0.30, 0.85, 1.12, 0.18

        def mini(org, pts_in=None, pts_on=None, k=0):
            g = VGroup(_grid_lines(W, H, cm, org, GREY_D, 1))
            if pts_in:
                g.add(_polyline([org + np.array([p[0] * cm, p[1] * cm, 0])
                                 for p in pts_in], gcol[k], 3.5))
            if pts_on:
                g.add(_polyline([org + np.array([p[0] * cm, p[1] * cm, 0])
                                 for p in pts_on], WHITE, 3))
            return g

        bx = -1.9                                # left edge of block 0
        by = 3.18                                # centre of the header row
        y_lab = -1.25                            # one baseline for the labels
        labels = []
        for k in ks:
            firsts, seconds = table[k]
            nc = len(seconds)
            hx = bx + W * cm / 2                 # header column centre

            def org_at(xc, yc):
                return np.array([xc - W * cm / 2, yc - H * cm / 2, 0.0])

            tx = [hx + pxm + hg + j * pxm for j in range(nc)]
            ty = [by - pym - hg - i * pym for i in range(len(firsts))]
            heads_in = [mini(org_at(hx, ty[i]), pts_in=f, k=k)
                        for i, f in enumerate(firsts)]
            heads_on = [mini(org_at(tx[j], by), pts_on=s, k=k)
                        for j, s in enumerate(seconds)]
            cellm = [mini(org_at(tx[j], ty[i]), pts_in=f, pts_on=s, k=k)
                     for i, f in enumerate(firsts)
                     for j, s in enumerate(seconds)]
            sep = VGroup(
                Line([hx + W * cm / 2 + hg / 2 + 0.12, by + 0.5, 0],
                     [hx + W * cm / 2 + hg / 2 + 0.12, ty[-1] - 0.5, 0],
                     color=GREY_C, stroke_width=1.5),
                Line([hx - 0.4, by - H * cm / 2 - hg / 2 - 0.12, 0],
                     [tx[-1] + 0.4, by - H * cm / 2 - hg / 2 - 0.12, 0],
                     color=GREY_C, stroke_width=1.5))

            # in the big grid: every way in (colour), every way on (white)
            fan = VGroup(*[_polyline([B(p) for p in f], gcol[k], 7)
                           for f in firsts],
                         *[_polyline([B(p) for p in s], WHITE, 5)
                           for s in seconds])
            self.play(Create(fan), Flash(cross[k].get_center(),
                                         color=gcol[k], line_length=0.22),
                      run_time=1.0)
            self.play(FadeIn(sep), *[FadeIn(h) for h in heads_in],
                      *[FadeIn(h) for h in heads_on], run_time=0.7)
            self.play(LaggedStart(*[FadeIn(c) for c in cellm],
                                  lag_ratio=0.15), run_time=0.9)
            lab = tag(f"C({m},{k})·C({n},{r - k})", 22, gcol[k])
            lab.move_to([(hx - W * cm / 2 + tx[-1] + W * cm / 2) / 2,
                         y_lab, 0])
            labels.append(lab)
            self.play(FadeIn(lab), FadeOut(fan), run_time=0.6)
            bx = tx[-1] + W * cm / 2 + 0.75

        # the labels, joined into the identity
        row = VGroup()
        for i, l in enumerate(labels):
            if i:
                row.add(tag("+", 24))
            row.add(l.copy())
        row.add(tag("=", 24), tag(f"C({m + n},{r})", 24, YELLOW_B))
        row.arrange(RIGHT, buff=0.2).move_to([0.9, -2.3, 0])
        self.play(*[TransformFromCopy(l, row[2 * i])
                    for i, l in enumerate(labels)],
                  *[FadeIn(row[j]) for j in range(1, len(row), 2)],
                  FadeIn(row[-1]), run_time=1.2)
        self.play(Write(caption("∑ C(m,k)·C(n,r−k)  =  C(m+n, r)", 32)))
        self.hold(2.2)
