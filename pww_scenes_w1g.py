# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w1g.py — proofs without words, 2D (manim):
#     K1  counting lattice paths            K2  Pascal's rule by paths
#     K23 the binomial theorem              K8  stars and bars
#     K6  domino tilings are Fibonacci      K11 the mutilated chessboard
#     K13 inclusion–exclusion               F33 regular polygons that tile
#     J32 doubling the square               J15 Euclid's algorithm
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Counting scenes enumerate a small case completely and check the counts
# (and the bijection) with check(...); geometric scenes check landings,
# areas and angle sums, so a wrong construction fails the render.

from itertools import combinations
from math import comb


# ------------------------------------------------------------ helpers

def _binom(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def _lattice_paths(W, H):
    """All east/north lattice paths from (0,0) to (W,H), as point lists,
    in lexicographic order of the positions of their east steps."""
    out = []
    for Es in combinations(range(W + H), W):
        pts = [(0, 0)]
        for i in range(W + H):
            x, y = pts[-1]
            pts.append((x + 1, y) if i in Es else (x, y + 1))
        out.append(pts)
    return out


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


def _arrow(p, q, color, width=6, tip=0.2):
    """Straight arrow from p to q (screen points), tip exactly at q."""
    a = Arrow(to3(p), to3(q), buff=0, color=color, stroke_width=width,
              tip_length=tip, max_tip_length_to_length_ratio=0.5,
              max_stroke_width_to_length_ratio=100)
    return a


def _rot2(p, c, ang):
    """2D point p turned by ang about c."""
    p, c = np.asarray(p, float)[:2], np.asarray(c, float)[:2]
    ca, sa = np.cos(ang), np.sin(ang)
    v = p - c
    return c + np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]])


def _rect_pts(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _box(x0, y0, w, h, color, op=FILL, sw=1.5, stroke=WHITE):
    return mk(_rect_pts(x0, y0, w, h), color, op, stroke_width=sw,
              stroke_color=stroke)


def _inside(m):
    """True if the mobject lies inside the safe content area."""
    return (m.get_left()[0] >= -SAFE_X - 1e-6 and m.get_right()[0] <= SAFE_X + 1e-6
            and m.get_bottom()[1] >= SAFE_BOTTOM - 1e-6
            and m.get_top()[1] <= SAFE_TOP + 1e-6)


def _apart(a, b, pad=0.04):
    """True if the bounding boxes of two mobjects do not meet."""
    a0, a1 = a.get_corner(DL), a.get_corner(UR)
    b0, b1 = b.get_corner(DL), b.get_corner(UR)
    return (a1[0] + pad < b0[0] or b1[0] + pad < a0[0]
            or a1[1] + pad < b0[1] or b1[1] + pad < a0[1])


_BASE_CHARS = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789+=")


def _baseline(t):
    """Baseline of a Text: the median bottom of its non-descending glyphs."""
    ys = [g.get_bottom()[1] for ch, g in zip(t.text, t.submobjects)
          if ch in _BASE_CHARS]
    if not ys:
        ys = [g.get_bottom()[1] for g in t.submobjects]
    return float(np.median(ys))


def _supline(parts, size=30, color=YELLOW_B, sup_scale=0.62, rise=0.48):
    """A line of text with true superscripts, without TeX.

    parts: list of (string, is_superscript). Normal pieces share one
    baseline; a superscript sits `rise` x-heights above it, smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for s, sup in parts:
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        body = s.strip(" ")
        x += lead * sp
        t = Text(body, font_size=size * (sup_scale if sup else 1.0),
                 color=color)
        y0 = rise * xh if sup else 0.0
        t.shift(np.array([x - t.get_left()[0], y0 - _baseline(t), 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sup else 0.02)
        g.add(t)
    return g


def _sup_caption(parts, size=30, color=YELLOW_B):
    g = _supline(parts, size, color)
    if g.width > 13.4:
        g.scale_to_fit_width(13.4)
    g.set_x(0.0)
    g.to_edge(DOWN, buff=0.3)
    return g


def _glide(mob, pivot, target, angle, **kw):
    """A rigid motion that stays rigid on every frame: the piece turns by
    `angle` about its pivot while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _boxes(m):
    """Sorted bounding boxes of every polygon in a mobject's family."""
    return sorted(tuple(np.round(np.concatenate([q.get_corner(DL)[:2],
                                                 q.get_corner(UR)[:2]]), 6))
                  for q in m.get_family() if isinstance(q, Polygon))


def _same_vertices(poly, pts, tol=1e-6):
    """True if a manim Polygon has exactly the given corner points."""
    vs = [to3(v) for v in poly.get_vertices()]
    ps = [to3(q) for q in pts]
    return (len(vs) == len(ps)
            and all(any(np.linalg.norm(v - q) < tol for q in ps) for v in vs)
            and all(any(np.linalg.norm(v - q) < tol for v in vs) for q in ps))


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption inside the frame, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        check(_inside(m), f"{type(m).__name__} inside the safe area")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_apart(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# ===================================================================== K1

class K1_LatticePaths(Board):
    """A path of m east and n north unit steps is a word of m + n steps;
    numbering the steps 1 … m+n, the path is fixed by WHICH m of the
    numbers are east steps, and every choice of m numbers gives a path.
    Shown for m = 3, n = 2: all C(5,3) = 10 choices, each beside its path."""

    def construct(self):
        m, n = 3, 2
        N = m + n
        paths = _lattice_paths(m, n)
        choices = list(combinations(range(N), m))
        check(len(paths) == len(choices) == _binom(N, m) == 10,
              "10 paths, 10 choices")
        for pts, ch in zip(paths, choices):
            east = tuple(i for i in range(N) if pts[i + 1][0] == pts[i][0] + 1)
            check(east == ch, "path <-> its set of east steps")
            check(pts[-1] == (m, n), "every word ends at (m, n)")
        check(len({tuple(p) for p in paths}) == len(paths), "paths distinct")

        CE, CN = ORANGE, BLUE_D
        cb = 1.2
        O = np.array([-5.15, 0.55, 0.0])

        def B(p):
            return O + cb * np.array([p[0], p[1], 0.0])

        grid = _grid_lines(m, n, cb, O, GREY_B, 2)
        dO = Dot(B((0, 0)), radius=0.08)
        dT = Dot(B((m, n)), radius=0.1, color=YELLOW_B)
        lm = tag("m = 3", 26).move_to(B((m / 2, 0)) + DOWN * 0.42)
        ln = tag("n = 2", 26).move_to(B((0, n / 2)) + LEFT * 0.72)
        check(_inside(ln), "n label inside the frame")
        self.play(Create(grid), FadeIn(dO), FadeIn(dT), run_time=1.0)
        self.play(FadeIn(lm), FadeIn(ln), run_time=0.5)

        # the step strip: one slot per step, numbered
        sc = 0.62
        sx0 = B((m / 2, 0))[0] - N * sc / 2
        sy = -1.05

        def slot_c(i):
            return np.array([sx0 + (i + 0.5) * sc, sy, 0.0])

        slots = VGroup(*[Square(side_length=sc, stroke_color=GREY_B,
                                stroke_width=2).move_to(slot_c(i))
                         for i in range(N)])
        nums = VGroup(*[tag(str(i + 1), 20, GREY_A).move_to(
            slot_c(i) + DOWN * (sc / 2 + 0.22)) for i in range(N)])
        self.play(FadeIn(slots), FadeIn(nums), run_time=0.6)

        # trace one path; each step fills its slot
        demo = 3                                   # east steps 1, 3, 4
        pts, ch = paths[demo], choices[demo]
        check(ch == (0, 2, 3), "demo path is E N E E N")
        steps, marks, fills = VGroup(), VGroup(), VGroup()
        for i in range(N):
            p, q = B(pts[i]), B(pts[i + 1])
            east = i in ch
            col = CE if east else CN
            ar = _arrow(p, q, col, width=8, tip=0.24)
            off = DOWN * 0.27 if east else RIGHT * 0.27
            mk_ = tag(str(i + 1), 20, col if east else BLUE_B).move_to(
                (p + q) / 2 + off)
            c = slot_c(i)
            d = RIGHT if east else UP
            f = VGroup(Square(side_length=sc, fill_color=col, fill_opacity=0.85,
                              stroke_color=WHITE, stroke_width=2).move_to(c),
                       _arrow(c - d * 0.2, c + d * 0.2, WHITE, width=5,
                              tip=0.14))
            steps.add(ar)
            marks.add(mk_)
            fills.add(f)
            self.play(GrowArrow(ar), FadeIn(mk_), FadeIn(f), run_time=0.55)
        sel = VGroup(*[Square(side_length=sc + 0.08, stroke_color=YELLOW_B,
                              stroke_width=5).move_to(slot_c(i)) for i in ch])
        self.play(Create(sel), run_time=0.7)
        self.hold(0.6)

        # every choice of 3 slots of 5, each with its path
        cm, ss = 0.42, 0.3
        col_x = [-0.2, 3.35]
        row_y = [3.05 - r * 1.36 for r in range(5)]
        gallery = VGroup()
        items = []
        for k, (pts_k, ch_k) in enumerate(zip(paths, choices)):
            c_, r_ = divmod(k, 5)
            org = np.array([col_x[c_], row_y[r_] - n * cm / 2, 0.0])
            g = VGroup(_grid_lines(m, n, cm, org, GREY_D, 1))
            for i in range(N):
                p = org + cm * np.array([*pts_k[i], 0.0])
                q = org + cm * np.array([*pts_k[i + 1], 0.0])
                g.add(Line(p, q, color=CE if i in ch_k else BLUE_B,
                           stroke_width=4))
            strip = VGroup()
            for i in range(N):
                e = i in ch_k
                strip.add(Square(side_length=ss, fill_color=CE if e else BLUE_E,
                                 fill_opacity=0.9 if e else 0.55,
                                 stroke_color=WHITE, stroke_width=1.2).move_to(
                    [org[0] + m * cm + 0.3 + (i + 0.5) * ss, row_y[r_], 0]))
            g.add(strip)
            items.append(g)
            gallery.add(g)
        check(_inside(gallery), "gallery inside the frame")
        check(all(_apart(items[i], items[j], 0.1) for i in range(10)
                  for j in range(i + 1, 10)), "gallery items do not touch")

        src = VGroup(grid.copy(), steps.copy(), fills.copy())
        self.play(FadeTransform(src, items[demo]), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(items[k]) for k in range(10) if k != demo],
                              lag_ratio=0.18), run_time=2.6)
        frame_d = SurroundingRectangle(items[demo], buff=0.1, color=YELLOW_B,
                                       stroke_width=3)
        self.play(Create(frame_d), run_time=0.5)

        total = tag("C(5, 3)  =  10", 32, YELLOW_B).move_to(
            [B((m / 2, 0))[0], -2.35, 0])
        check(_inside(total), "total inside the frame")
        self.play(FadeIn(total, shift=UP * 0.1), run_time=0.8)
        cap = caption("m steps →,  n steps ↑ :    N  =  C(m + n, m)", 32)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K2

class K2_PascalPaths(Board):
    """Paths of unit steps east and north from O to T = (k, n−k) number
    C(n,k): n steps, k of them east. The last step into T comes either
    from the west, W = (k−1, n−k), or from the south, S = (k, n−k−1); so
    the paths to T are the paths to W with one step east added, followed
    by the paths to S with one step north added — no path in both lists,
    none missing. Shown for n = 5, k = 2: all ten paths, sorted."""

    def construct(self):
        n, k = 5, 2
        W, H = k, n - k
        T, Wp, Sp = (W, H), (W - 1, H), (W, H - 1)
        paths = _lattice_paths(W, H)
        check(len(paths) == _binom(n, k) == 10, "C(5,2) paths to T")
        from_w = [p for p in paths if p[-2] == Wp]
        from_s = [p for p in paths if p[-2] == Sp]
        check(len(from_w) + len(from_s) == len(paths), "every path is in a list")
        check(not set(map(tuple, from_w)) & set(map(tuple, from_s)),
              "no path in both lists")
        check(sorted(tuple(p[:-1]) for p in from_w)
              == sorted(tuple(p) for p in _lattice_paths(*Wp)),
              "west list = paths to W, plus a step east")
        check(sorted(tuple(p[:-1]) for p in from_s)
              == sorted(tuple(p) for p in _lattice_paths(*Sp)),
              "south list = paths to S, plus a step north")
        check(len(from_w) == _binom(n - 1, k - 1) == 4
              and len(from_s) == _binom(n - 1, k) == 6, "4 + 6")
        for nn in range(1, 15):
            for kk in range(1, nn):
                check(_binom(nn, kk) == _binom(nn - 1, kk - 1) + _binom(nn - 1, kk),
                      "Pascal's rule")

        CW, CS = ORANGE, TEAL_C
        cb = 1.3
        O = np.array([-6.1, -2.25, 0.0])

        def B(p):
            return O + cb * np.array([p[0], p[1], 0.0])

        grid = _grid_lines(W, H, cb, O, GREY_B, 2)
        dO = Dot(B((0, 0)), radius=0.08)
        dT = Dot(B(T), radius=0.11, color=YELLOW_B)
        lO = tag("O", 24).next_to(dO, DOWN + LEFT, buff=0.06)
        lT = tag("T", 26, YELLOW_B).next_to(dT, UP + RIGHT, buff=0.06)
        self.play(Create(grid), FadeIn(dO), FadeIn(dT), FadeIn(lO), FadeIn(lT),
                  run_time=1.0)

        # the last step: from the west or from the south
        aw = _arrow(B(Wp), B(T), CW, width=9, tip=0.26)
        as_ = _arrow(B(Sp), B(T), CS, width=9, tip=0.26)
        dW = Dot(B(Wp), radius=0.1, color=CW)
        dS = Dot(B(Sp), radius=0.1, color=CS)
        self.play(GrowArrow(aw), GrowArrow(as_), FadeIn(dW), FadeIn(dS),
                  run_time=0.9)
        self.hold(0.4)

        # the ten paths, filed by their last step
        cm = 0.4
        x0, pitch = -2.45, 1.15
        row_y = {CW: 2.6, CS: 0.3}

        def mini(pts, j, col):
            org = np.array([x0 + j * pitch, row_y[col] - H * cm / 2, 0.0])
            g = VGroup(_grid_lines(W, H, cm, org, GREY_D, 1))
            sp = [org + cm * np.array([*q, 0.0]) for q in pts]
            g.add(_polyline(sp[:-1], WHITE, 3))
            g.add(Line(sp[-2], sp[-1], color=col, stroke_width=6))
            return g

        rows = {}
        for col, lst in ((CW, from_w), (CS, from_s)):
            minis = [mini(p, j, col) for j, p in enumerate(lst)]
            rows[col] = VGroup(*minis)
            prev = None
            for p, mm in zip(lst, minis):
                big = VGroup(_polyline([B(q) for q in p[:-1]], WHITE, 6),
                             Line(B(p[-2]), B(p[-1]), color=col,
                                  stroke_width=10))
                anims = [Create(big), FadeIn(mm)]
                if prev is not None:
                    anims.append(FadeOut(prev))
                self.play(*anims, run_time=0.5)
                prev = big
            self.play(FadeOut(prev), run_time=0.25)
        self.bring_to_front(aw, as_, dW, dS, dT)

        boxW = SurroundingRectangle(rows[CW], buff=0.14, color=CW, stroke_width=3)
        boxS = SurroundingRectangle(rows[CS], buff=0.14, color=CS, stroke_width=3)
        cW = tag("C(4,1) = 4", 28, CW).next_to(boxW, RIGHT, buff=0.3)
        cS = tag("C(4,2) = 6", 28, CS).next_to(boxS, RIGHT, buff=0.3)
        for m_ in (boxW, boxS, cW, cS):
            check(_inside(m_), "rows and counts inside the frame")
        check(boxW.get_left()[0] > B(T)[0] + 0.6, "rows clear of the big grid")
        self.play(Create(boxW), Create(boxS), FadeIn(cW), FadeIn(cS),
                  run_time=0.9)

        eq1 = VGroup(tag("C(5,2)", 32, YELLOW_B), tag("=", 32),
                     tag("C(4,1)", 32, CW), tag("+", 32),
                     tag("C(4,2)", 32, CS)).arrange(RIGHT, buff=0.25)
        eq2 = VGroup(tag("10", 32, YELLOW_B), tag("=", 32), tag("4", 32, CW),
                     tag("+", 32), tag("6", 32, CS)).arrange(RIGHT, buff=0.25)
        eq1.move_to([1.2, -1.3, 0])
        for a_, b_ in zip(eq2, eq1):
            a_.move_to([b_.get_center()[0], -2.15, 0])
        check(_inside(eq1) and _inside(eq2), "equations inside the frame")
        self.play(FadeIn(eq1, shift=UP * 0.1), run_time=0.8)
        self.play(FadeIn(eq2, shift=UP * 0.1), run_time=0.7)

        # the same rule at every lattice point: Pascal's triangle
        labs = []
        for s in range(W + H + 1):
            wave = []
            for x in range(W + 1):
                y = s - x
                if 0 <= y <= H:
                    col = {Wp: CW, Sp: CS, T: YELLOW_B}.get((x, y), GREY_A)
                    off = (np.array([0.3, 0.24, 0.0]) if (x, y) == T
                           else np.array([-0.26, 0.22, 0.0]))
                    t = tag(str(_binom(x + y, x)), 24, col).move_to(
                        B((x, y)) + off)
                    wave.append(t)
            labs.append(wave)
        self.play(LaggedStart(*[AnimationGroup(*[FadeIn(t) for t in w])
                                for w in labs], lag_ratio=0.35),
                  FadeOut(lT), run_time=1.8)
        cap = caption("C(n, k)  =  C(n−1, k−1)  +  C(n−1, k)", 34)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K23

class K23_BinomialTheorem(Board):
    """Multiplying out (a + b)(a + b)(a + b) means picking a or b from each
    factor in turn: a tree with 2·2·2 = 8 leaves, one term per leaf. A
    term with k letters b is a choice of WHICH k factors give b, so
    C(3, k) leaves carry a^(3−k) b^k. Sorting the leaves by k gives
    (a + b)³ = a³ + 3a²b + 3ab² + b³; the same count works for every n."""

    def construct(self):
        n = 3
        words = ["".join(w) for w in
                 [[("a", "b")[(i >> (n - 1 - j)) & 1] for j in range(n)]
                  for i in range(2 ** n)]]
        check(len(set(words)) == 2 ** n, "8 different choices")
        for kk in range(n + 1):
            grp = [w for w in words if w.count("b") == kk]
            check(len(grp) == _binom(n, kk), f"C(3,{kk}) words with {kk} b's")
            check({tuple(i for i, c in enumerate(w) if c == "b") for w in grp}
                  == set(combinations(range(n), kk)),
                  "the b-places are exactly the k-subsets of the factors")
        # the expansion itself, checked numerically
        for a_, b_ in ((1.3, 0.7), (2.0, -0.5), (0.4, 3.1)):
            check(abs((a_ + b_) ** n - sum(a_ ** (n - w.count("b")) *
                                           b_ ** w.count("b") for w in words))
                  < 1e-9, "sum over the leaves = (a + b)³")

        CA, CB = BLUE_B, ORANGE
        t2c = {"a": CA, "b": CB}

        def word(s, size):
            return Text(s, font_size=size, t2c=t2c)

        xc, dx = 0.6, 1.45
        ys = [3.45, 2.55, 1.65, 0.75]

        def node_x(level, idx):
            span = 2 ** (n - level)                # leaves under this node
            return xc + ((idx + 0.5) * span - 0.5 - (2 ** n - 1) / 2) * dx

        factors = VGroup(*[Text("(a + b)", font_size=28, t2c=t2c).move_to(
            [-5.85, ys[lv], 0]) for lv in (1, 2, 3)])
        root = Dot([xc, ys[0], 0], radius=0.08, color=WHITE)
        self.play(FadeIn(factors), FadeIn(root), run_time=0.9)

        nodes = {(0, 0): root}
        leaves = []
        for lv in range(1, n + 1):
            edges, labs = [], []
            for idx in range(2 ** lv):
                s = words[idx * 2 ** (n - lv)][:lv]
                w = word(s, 30 if lv == n else 28).move_to(
                    [node_x(lv, idx), ys[lv], 0])
                par = nodes[(lv - 1, idx // 2)]
                p0 = par.get_bottom() + DOWN * 0.06 if lv > 1 else par.get_center()
                p1 = w.get_top() + UP * 0.08
                edges.append(Line(p0, p1, color=CA if s[-1] == "a" else CB,
                                  stroke_width=3))
                labs.append(w)
                nodes[(lv, idx)] = w
            self.play(Indicate(factors[lv - 1], color=YELLOW_B, scale_factor=1.12),
                      *[Create(e) for e in edges], *[FadeIn(w) for w in labs],
                      run_time=1.1)
            if lv == n:
                leaves = labs
        check(all(_inside(m) for m in leaves) and _inside(factors),
              "tree inside the frame")
        check(factors[2].get_right()[0] + 0.15 < leaves[0].get_left()[0],
              "factor labels clear of the leaves")
        self.hold(0.5)

        # sort the leaves by how many factors gave b
        colx = [-3.5, -0.7, 2.1, 4.9]
        stack_y = [-0.2, -0.62, -1.04]
        mons = ["a³", "3a²b", "3ab²", "b³"]
        clabs = []
        for kk in range(n + 1):
            idxs = [i for i, w in enumerate(words) if w.count("b") == kk]
            targets = [word(words[i], 28).move_to([colx[kk], stack_y[j], 0])
                       for j, i in enumerate(idxs)]
            cl = tag(f"C(3,{kk}) = {len(idxs)}", 22, GREY_A).move_to(
                [colx[kk], -1.6, 0])
            clabs.append(cl)
            self.play(*[TransformFromCopy(leaves[i], t)
                        for i, t in zip(idxs, targets)],
                      FadeIn(cl), run_time=1.0)

        terms = VGroup(*[Text(s, font_size=34, t2c=t2c).move_to([colx[i], -2.4, 0])
                         for i, s in enumerate(mons)])
        pluses = VGroup(*[tag("+", 34).move_to([(colx[i] + colx[i + 1]) / 2,
                                                -2.4, 0]) for i in range(3)])
        lhs = Text("(a + b)³  =", font_size=34, t2c=t2c)
        lhs.move_to([terms[0].get_left()[0] - 0.3 - lhs.width / 2, -2.4, 0])
        for t_ in terms:
            t_.align_to(lhs, DOWN)
        check(_inside(lhs) and _inside(terms), "expansion inside the frame")
        self.play(FadeIn(lhs), FadeIn(terms), FadeIn(pluses), run_time=1.0)
        cap = _sup_caption([("(a + b)ⁿ  =  ∑  C(n, k)  a", False),
                            ("n−k", True), (" b", False), ("k", True),
                            ("      k = 0, 1, …, n", False)], 32)
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K8

def _star(c, r=0.17, color=YELLOW_C):
    s = Star(n=5, outer_radius=r, inner_radius=r * 0.42, color=color,
             fill_opacity=1.0, stroke_width=0)
    return s.move_to(to3(c))


def _bar(c, h, width=6, color=WHITE):
    c = to3(c)
    return Line(c + DOWN * h / 2, c + UP * h / 2, color=color,
                stroke_width=width)


class K8_StarsAndBars(Board):
    """Put n identical stars into k boxes in a row: the k − 1 inner walls
    and the n stars, read left to right, form a row of n + k − 1 symbols,
    and the row is fixed by which k − 1 places hold walls (bars). Every
    choice of places gives a filling (empty boxes allowed), so there are
    C(n + k − 1, k − 1) fillings. Shown for n = 4, k = 3: all 15."""

    def construct(self):
        n, k = 4, 3
        L = n + k - 1
        bar_sets = list(combinations(range(L), k - 1))
        check(len(bar_sets) == _binom(L, k - 1) == 15, "C(6,2) = 15")

        def split(bars):
            sizes, prev = [], -1
            for b in tuple(bars) + (L,):
                sizes.append(b - prev - 1)
                prev = b
            return tuple(sizes)

        def bars_of(dist):
            return tuple(sum(dist[:j + 1]) + j for j in range(k - 1))

        dists = [split(b) for b in bar_sets]
        every = {(a, b, n - a - b) for a in range(n + 1)
                 for b in range(n + 1 - a)}
        check(len(set(dists)) == len(dists) and set(dists) == every,
              "each filling of 3 boxes with 4 stars appears exactly once")
        check(all(bars_of(d) == b for d, b in zip(dists, bar_sets)),
              "filling -> bar places -> the same filling")

        # ---------------- the tray: three boxes side by side
        cw, fy, wh = 1.5, 2.3, 0.9
        wx = [(-1.5 + j) * cw for j in range(k + 1)]       # wall x's
        floor = Line([wx[0], fy, 0], [wx[-1], fy, 0], color=WHITE,
                     stroke_width=5)
        walls = [Line([x, fy, 0], [x, fy + wh, 0], color=WHITE, stroke_width=5)
                 for x in wx]
        lab_n = tag("n = 4", 28, YELLOW_C).move_to([-4.9, 3.0, 0])
        lab_k = tag("k = 3", 28).move_to([-4.9, 2.4, 0])
        sc, ry = 0.62, 1.05

        def slot_x(i):
            return (i - (L - 1) / 2) * sc

        slots = VGroup(*[Square(side_length=sc, stroke_color=GREY_C,
                                stroke_width=1.5).move_to([slot_x(i), ry, 0])
                         for i in range(L)])
        nums = VGroup(*[tag(str(i + 1), 18, GREY_A).move_to(
            [slot_x(i), ry - sc / 2 - 0.2, 0]) for i in range(L)])

        def run(dist, first):
            tray = VGroup(floor.copy(), *[w.copy() for w in walls])
            if first:
                self.play(Create(tray), FadeIn(lab_n), FadeIn(lab_k),
                          run_time=1.0)
            else:
                self.play(FadeIn(tray), run_time=0.6)
            stars = [_star([(t - (n - 1) / 2) * 0.42, 3.5]) for t in range(n)]
            check(stars[0].get_top()[1] <= SAFE_TOP, "stars inside the frame")
            self.play(*[FadeIn(s) for s in stars], run_time=0.4)
            moves, t = [], 0
            for j, c in enumerate(dist):
                cx = (wx[j] + wx[j + 1]) / 2
                for i in range(c):
                    moves.append(stars[t].animate.move_to(
                        [cx + (i - (c - 1) / 2) * 0.42, fy + 0.32, 0]))
                    t += 1
            cnts = VGroup(*[tag(str(c), 26, GREY_A).move_to(
                [(wx[j] + wx[j + 1]) / 2, fy - 0.3, 0])
                for j, c in enumerate(dist)])
            self.play(*moves, run_time=0.9)
            self.play(FadeIn(cnts), run_time=0.35)

            # collapse: the inner walls are the bars
            self.play(FadeOut(cnts), run_time=0.3)
            bset = bars_of(dist)
            star_slots = [i for i in range(L) if i not in bset]
            anims = [stars[t].animate.move_to([slot_x(i), ry, 0])
                     for t, i in enumerate(star_slots)]
            bars = [_bar([slot_x(i), ry], 0.5) for i in bset]
            anims += [Transform(tray[2 + j], bars[j]) for j in range(k - 1)]
            anims += [FadeOut(tray[0]), FadeOut(tray[1]), FadeOut(tray[-1])]
            if first:
                anims += [FadeIn(slots), FadeIn(nums)]
            self.play(*anims, run_time=1.4)
            hi = VGroup(*[Square(side_length=sc + 0.06, stroke_color=YELLOW_B,
                                 stroke_width=5).move_to([slot_x(i), ry, 0])
                          for i in bset])
            lab = tag(" + ".join(map(str, dist)), 28, GREY_A).move_to(
                [3.15, ry, 0])
            self.play(Create(hi), FadeIn(lab), run_time=0.6)
            self.hold(0.5)
            return VGroup(*stars, tray[2], tray[3], hi, lab)

        row1 = run((1, 2, 1), True)
        self.play(FadeOut(row1), run_time=0.4)
        row2 = run((2, 0, 2), False)

        # ---------------- all 15 choices of the two bar places
        gs = 0.34
        gx = [-5.1, -2.55, 0.0, 2.55, 5.1]
        gy = [-0.3, -1.27, -2.24]
        items = []
        for idx, (bset, dist) in enumerate(zip(bar_sets, dists)):
            r_, c_ = divmod(idx, 5)
            x0 = gx[c_] - L * gs / 2
            g = VGroup()
            for i in range(L):
                c = [x0 + (i + 0.5) * gs, gy[r_], 0]
                g.add(Square(side_length=gs, stroke_color=GREY_D,
                             stroke_width=1).move_to(c))
                g.add(_bar(c, 0.28, 4) if i in bset else _star(c, 0.125))
            g.add(tag("+".join(map(str, dist)), 20, GREY_A).move_to(
                [gx[c_], gy[r_] - 0.37, 0]))
            items.append(g)
        allg = VGroup(*items)
        check(_inside(allg), "gallery inside the frame")
        check(allg.get_top()[1] < nums.get_bottom()[1] - 0.15,
              "gallery clear of the demo row")
        self.play(LaggedStart(*[FadeIn(g) for g in items], lag_ratio=0.15),
                  run_time=3.0)
        marks = VGroup(*[SurroundingRectangle(items[dists.index(d)], buff=0.08,
                                              color=YELLOW_B, stroke_width=3)
                         for d in ((1, 2, 1), (2, 0, 2))])
        self.play(Create(marks), run_time=0.6)
        tot = tag("C(6, 2)  =  15", 34, YELLOW_B).move_to([4.6, 2.75, 0])
        check(_inside(tot), "total inside the frame")
        self.play(FadeIn(tot, shift=UP * 0.1), run_time=0.8)
        cap = caption("n stars in k groups:   C(n + k − 1, k − 1)", 32)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K6

def _domino_tilings(n):
    """Tilings of the 2 × n strip, built by the first-piece rule: an
    upright domino then a tiling of 2 × (n−1), or two flat dominoes then a
    tiling of 2 × (n−2). Pieces: ('V', x, 0) or ('H', x, row)."""
    T = {0: [[]], 1: [[("V", 0, 0)]]}
    for m in range(2, n + 1):
        T[m] = ([[("V", 0, 0)] + [(k, x + 1, y) for k, x, y in t]
                 for t in T[m - 1]]
                + [[("H", 0, 0), ("H", 0, 1)] + [(k, x + 2, y) for k, x, y in t]
                   for t in T[m - 2]])
    return T


def _all_tilings_brute(n):
    """Every domino tiling of the 2 × n strip by exhaustive search."""
    out = []

    def go(covered, pieces):
        free = [(x, y) for x in range(n) for y in (0, 1) if (x, y) not in covered]
        if not free:
            out.append(frozenset(pieces))
            return
        x, y = free[0]
        if y == 0 and (x, 1) not in covered:
            go(covered | {(x, 0), (x, 1)}, pieces + [("V", x, 0)])
        if x + 1 < n and (x + 1, y) not in covered:
            go(covered | {(x, y), (x + 1, y)}, pieces + [("H", x, y)])

    go(frozenset(), [])
    return out


class K6_DominoFibonacci(Board):
    """The top-left cell of a 2 × n strip is covered either by an upright
    domino — the rest is a 2 × (n−1) strip — or by a flat one, and then the
    cell below can only take a flat one too — the rest is 2 × (n−2). So
    tₙ = tₙ₋₁ + tₙ₋₂ with t₁ = 1, t₂ = 2: the Fibonacci numbers. Every
    tiling of the 2 × 5 strip is listed, each built from a shorter one."""

    def construct(self):
        N = 5
        T = _domino_tilings(N)
        fib = [0, 1]
        for _ in range(N + 3):
            fib.append(fib[-1] + fib[-2])
        for m in range(1, N + 1):
            built = [frozenset(t) for t in T[m]]
            check(len(set(built)) == len(built), "no tiling listed twice")
            check(set(built) == set(_all_tilings_brute(m)),
                  f"the first-piece rule finds every tiling of 2 × {m}")
            check(len(built) == fib[m + 1], f"t{m} = F{m + 1}")
            for t in T[m]:
                cells = []
                for k, x, y in t:
                    cells += ([(x, 0), (x, 1)] if k == "V" else [(x, y), (x + 1, y)])
                check(sorted(cells) == sorted((x, y) for x in range(m)
                                              for y in (0, 1)),
                      "each tiling covers the strip exactly once")

        CV, CH = TEAL_D, ORANGE

        def piece(kind, x, y, u, org, col=None, sw=1.5):
            o = to3(org)
            if kind == "V":
                return _box(o[0] + x * u, o[1], u, 2 * u, col or CV, 0.9, sw)
            return _box(o[0] + x * u, o[1] + y * u, 2 * u, u, col or CH, 0.9, sw)

        # ---------------- why only two ways to start
        u0, n0 = 0.6, 6

        def strip_outline(org, m, u, color=GREY_B):
            o = to3(org)
            rect = Polygon(*[o + np.array([p[0], p[1], 0]) for p in
                             _rect_pts(0, 0, m * u, 2 * u)],
                           stroke_color=color, stroke_width=2.5)
            g = VGroup(rect)
            for x in range(1, m):
                g.add(Line(o + [x * u, 0, 0], o + [x * u, 2 * u, 0],
                           stroke_color=GREY_D, stroke_width=1))
            g.add(Line(o + [0, u, 0], o + [m * u, u, 0], stroke_color=GREY_D,
                       stroke_width=1))
            return g

        top_org = np.array([-n0 * u0 / 2, 2.15, 0.0])
        s0 = strip_outline(top_org, n0, u0)
        l0 = tag("tₙ", 30, YELLOW_B).next_to(s0, LEFT, buff=0.3)
        corner = Square(side_length=u0, stroke_color=YELLOW_B, stroke_width=6
                        ).move_to(top_org + np.array([u0 / 2, 1.5 * u0, 0]))
        self.play(Create(s0), FadeIn(l0), run_time=0.8)
        self.play(Create(corner), run_time=0.5)

        A_org = np.array([-6.0, 0.0, 0.0])
        B_org = np.array([0.6, 0.0, 0.0])
        sA, sB = strip_outline(A_org, n0, u0), strip_outline(B_org, n0, u0)
        self.play(TransformFromCopy(s0, sA), TransformFromCopy(s0, sB),
                  run_time=1.0)
        vA = piece("V", 0, 0, u0, A_org, sw=2.5)
        restA = DashedVMobject(Polygon(*[A_org + np.array([p[0], p[1], 0]) for p in
                                         _rect_pts(u0, 0, (n0 - 1) * u0, 2 * u0)],
                                       stroke_color=YELLOW_B, stroke_width=4),
                               num_dashes=36)
        lA = tag("tₙ₋₁", 30, YELLOW_B).move_to(A_org + np.array(
            [u0 + (n0 - 1) * u0 / 2, u0, 0]))
        self.play(FadeIn(vA), run_time=0.6)
        self.play(Create(restA), FadeIn(lA), run_time=0.7)

        hB1 = piece("H", 0, 1, u0, B_org, sw=2.5)
        self.play(FadeIn(hB1), run_time=0.6)
        below = Square(side_length=u0, stroke_color=RED_B, stroke_width=6
                       ).move_to(B_org + np.array([u0 / 2, u0 / 2, 0]))
        self.play(Create(below), run_time=0.4)
        hB2 = piece("H", 0, 0, u0, B_org, sw=2.5)
        self.play(FadeIn(hB2), FadeOut(below), run_time=0.6)
        restB = DashedVMobject(Polygon(*[B_org + np.array([p[0], p[1], 0]) for p in
                                         _rect_pts(2 * u0, 0, (n0 - 2) * u0, 2 * u0)],
                                       stroke_color=YELLOW_B, stroke_width=4),
                               num_dashes=30)
        lB = tag("tₙ₋₂", 30, YELLOW_B).move_to(B_org + np.array(
            [2 * u0 + (n0 - 2) * u0 / 2, u0, 0]))
        self.play(Create(restB), FadeIn(lB), run_time=0.7)
        eq = tag("tₙ  =  tₙ₋₁  +  tₙ₋₂", 34, YELLOW_B).move_to([0, -1.6, 0])
        plus = tag("+", 40, YELLOW_B).move_to(
            [(A_org[0] + n0 * u0 + B_org[0]) / 2, u0, 0])
        check(_inside(VGroup(sA, sB, l0)), "demo inside the frame")
        self.play(FadeIn(eq), FadeIn(plus), run_time=0.8)
        self.hold(0.8)
        demo = VGroup(s0, l0, corner, sA, sB, vA, restA, lA, hB1, hB2, restB, lB,
                      eq, plus)
        self.play(FadeOut(demo), run_time=0.6)

        # ---------------- every tiling, column by column
        u, pitch, ytop = 0.32, 0.75, 3.0
        colx = {1: -4.9, 2: -3.5, 3: -1.7, 4: 0.5, 5: 3.1}

        def org(m, i):
            return np.array([colx[m], ytop - 2 * u - i * pitch, 0.0])

        def strip(m, i, t):
            return VGroup(*[piece(k, x, y, u, org(m, i)) for k, x, y in t])

        cols = {m: [strip(m, i, t) for i, t in enumerate(T[m])]
                for m in range(1, N + 1)}
        heads = {m: tag(f"t{'₁₂₃₄₅'[m - 1]} = {len(T[m])}", 26, YELLOW_B).move_to(
            [colx[m] + m * u / 2, ytop + 0.4, 0]) for m in range(1, N + 1)}
        check(all(_inside(VGroup(*cols[m])) for m in cols), "columns inside")
        self.play(*[FadeIn(s) for s in cols[1] + cols[2]], FadeIn(heads[1]),
                  FadeIn(heads[2]), run_time=0.9)

        for m in range(3, N + 1):
            a, b = len(T[m - 1]), len(T[m - 2])
            movers, news = [], []
            for i in range(a):
                src = cols[m - 1][i]
                tgt = src.copy().shift(org(m, i) + RIGHT * u - org(m - 1, i))
                movers.append(TransformFromCopy(src, tgt))
                news.append(tgt)
            first = [piece("V", 0, 0, u, org(m, i), sw=3).set_stroke(YELLOW_B)
                     for i in range(a)]
            self.play(*movers, run_time=0.9)
            self.play(*[FadeIn(f) for f in first], run_time=0.4)
            movers2, news2 = [], []
            for i in range(b):
                src = cols[m - 2][i]
                tgt = src.copy().shift(org(m, a + i) + RIGHT * 2 * u
                                       - org(m - 2, i))
                movers2.append(TransformFromCopy(src, tgt))
                news2.append(tgt)
            first2 = [VGroup(piece("H", 0, 0, u, org(m, a + i), sw=3),
                             piece("H", 0, 1, u, org(m, a + i), sw=3)
                             ).set_stroke(YELLOW_B) for i in range(b)]
            self.play(*movers2, run_time=0.9)
            self.play(*[FadeIn(f) for f in first2], FadeIn(heads[m]),
                      run_time=0.5)
            for i in range(a):
                check(_boxes(VGroup(news[i], first[i])) == _boxes(cols[m][i]),
                      "upright domino + shorter tiling = the listed tiling")
            for i in range(b):
                check(_boxes(VGroup(news2[i], first2[i])) == _boxes(cols[m][a + i]),
                      "flat pair + shorter tiling = the listed tiling")
            self.play(*[f.animate.set_stroke(WHITE, 1.5) for f in first + first2],
                      run_time=0.3)

        # the last column: its two blocks
        a, b = len(T[N - 1]), len(T[N - 2])
        xr = colx[N] + N * u + 0.18
        y_a0, y_a1 = org(N, 0)[1] + 2 * u, org(N, a - 1)[1]
        y_b0, y_b1 = org(N, a)[1] + 2 * u, org(N, a + b - 1)[1]

        def bracket(y0, y1, col):
            return VGroup(Line([xr, y0, 0], [xr, y1, 0], color=col, stroke_width=3),
                          Line([xr - 0.12, y0, 0], [xr, y0, 0], color=col,
                               stroke_width=3),
                          Line([xr - 0.12, y1, 0], [xr, y1, 0], color=col,
                               stroke_width=3))

        brA, brB = bracket(y_a0, y_a1, CV), bracket(y_b0, y_b1, CH)
        tA = tag("t₄ = 5", 28, TEAL_B).next_to(brA, RIGHT, buff=0.2)
        tB = tag("t₃ = 3", 28, ORANGE).next_to(brB, RIGHT, buff=0.2)
        check(_inside(tA) and _inside(tB), "bracket labels inside")
        self.play(Create(brA), Create(brB), FadeIn(tA), FadeIn(tB), run_time=0.8)
        cap = caption("tₙ  =  tₙ₋₁ + tₙ₋₂ ,   t₁ = 1,  t₂ = 2   ⟹   "
                      "tₙ  =  Fₙ₊₁", 32)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K11

class K11_MutilatedBoard(Board):
    """Opposite corners of a chessboard have the same colour, so cutting
    them off leaves 30 dark and 32 light squares. Two neighbouring squares
    always differ in colour, so a domino covers one of each and any set of
    dominoes covers as many dark squares as light: line the squares up in
    two rows and every domino takes one from each row — two light squares
    are always left over. 31 dominoes cannot cover the 62 squares."""

    def construct(self):
        LIGHT, DARK, DOM = "#E6D7B5", "#8C6442", BLUE_C
        cut = [(0, 0), (7, 7)]
        cells = [(i, j) for j in range(8) for i in range(8) if (i, j) not in cut]

        def dark(p):
            return (p[0] + p[1]) % 2 == 0

        check(all(dark(p) for p in cut), "opposite corners share a colour")
        lights = [p for p in cells if not dark(p)]
        darks = [p for p in cells if dark(p)]
        check(len(lights) == 32 and len(darks) == 30, "32 light, 30 dark")
        for p in cells:
            for q in ((p[0] + 1, p[1]), (p[0], p[1] + 1)):
                if q in cells:
                    check(dark(p) != dark(q), "neighbours differ in colour")

        # thirty dominoes: everything but two light corners
        doms = ([((x, 0), (x + 1, 0)) for x in (1, 3, 5)]
                + [((x, 1), (x, 2)) for x in range(8)]
                + [((x, y), (x + 1, y)) for y in (3, 4) for x in (0, 2, 4, 6)]
                + [((x, 5), (x, 6)) for x in range(8)]
                + [((x, 7), (x + 1, 7)) for x in (1, 3, 5)])
        left = [(0, 7), (7, 0)]
        used = [q for d in doms for q in d]
        check(len(doms) == 30 and len(set(used)) == 60, "30 dominoes, no overlap")
        check(set(used) | set(left) == set(cells), "they cover all but two")
        check(all(not dark(p) for p in left), "the two left over are light")
        check(all(dark(a) != dark(b) for a, b in doms), "one of each colour")

        c = 0.5
        O = np.array([-4 * c, -0.6, 0.0])

        def Cpos(p):
            return O + np.array([(p[0] + 0.5) * c, (p[1] + 0.5) * c, 0.0])

        def sq(p, side=c):
            return Square(side_length=side, fill_color=DARK if dark(p) else LIGHT,
                          fill_opacity=1.0, stroke_color=BLACK,
                          stroke_width=0.5).move_to(Cpos(p))

        squares = {p: sq(p) for p in cells}
        frame = Square(side_length=8 * c, stroke_color=WHITE, stroke_width=2
                       ).move_to(O + np.array([4 * c, 4 * c, 0]))
        ghosts = [Square(side_length=c, fill_color=DARK, fill_opacity=1.0,
                         stroke_width=0).move_to(Cpos(p)) for p in cut]
        self.play(FadeIn(VGroup(*squares.values(), *ghosts)), Create(frame),
                  run_time=1.2)
        holes = VGroup(*[DashedVMobject(Square(side_length=c - 0.04,
                                               stroke_color=GREY_B,
                                               stroke_width=2), num_dashes=12
                                        ).move_to(Cpos(p)) for p in cut])
        self.play(*[g.animate.set_fill(RED_D) for g in ghosts], run_time=0.4)
        self.play(*[FadeOut(g, scale=0.3) for g in ghosts], FadeIn(holes),
                  run_time=0.7)

        def domino(a, b, col=DOM, sw=4.5, op=0.08):
            pa, pb = Cpos(a), Cpos(b)
            horiz = a[1] == b[1]
            w = (2 * c if horiz else c) - 0.08
            h = (c if horiz else 2 * c) - 0.08
            return RoundedRectangle(corner_radius=0.07, width=w, height=h,
                                    stroke_color=col, stroke_width=sw,
                                    fill_color=col, fill_opacity=op
                                    ).move_to((pa + pb) / 2)

        # a domino anywhere covers one light and one dark square
        legend_y = 1.55
        icon = VGroup(Square(side_length=c, fill_color=LIGHT, fill_opacity=1,
                             stroke_width=0),
                      Square(side_length=c, fill_color=DARK, fill_opacity=1,
                             stroke_width=0)).arrange(RIGHT, buff=0)
        icon.add(RoundedRectangle(corner_radius=0.07, width=2 * c - 0.08,
                                  height=c - 0.08, stroke_color=DOM,
                                  stroke_width=4.5, fill_color=DOM,
                                  fill_opacity=0.08).move_to(icon))
        legend = VGroup(icon, tag("=", 30),
                        Square(side_length=c, fill_color=LIGHT, fill_opacity=1,
                               stroke_width=0), tag("+", 30),
                        Square(side_length=c, fill_color=DARK, fill_opacity=1,
                               stroke_width=0)).arrange(RIGHT, buff=0.22)
        legend.move_to([-4.6, legend_y, 0])
        check(legend.get_right()[0] < frame.get_left()[0] - 0.4, "legend clear")
        d1 = domino((2, 4), (3, 4))
        self.play(FadeIn(d1), run_time=0.5)
        self.play(Indicate(squares[(2, 4)], scale_factor=1.0, color=YELLOW_B),
                  Indicate(squares[(3, 4)], scale_factor=1.0, color=YELLOW_B),
                  run_time=0.6)
        d2 = domino((5, 1), (5, 2))
        self.play(_glide(d1, d1.get_center(), d2.get_center(), -PI / 2),
                  run_time=0.9)
        check(close(d1.get_center(), d2.get_center(), 1e-6)
              and abs(d1.width - d2.width) < 1e-6, "the domino lands upright")
        self.play(Indicate(squares[(5, 1)], scale_factor=1.0, color=YELLOW_B),
                  Indicate(squares[(5, 2)], scale_factor=1.0, color=YELLOW_B),
                  run_time=0.6)
        self.play(FadeIn(legend), run_time=0.7)
        self.play(FadeOut(d1), run_time=0.3)

        # line the squares up: light above, dark below
        s, x0, yl, yd = 0.36, -5.76, -1.42, -1.78
        order_l = [a if not dark(a) else b for a, b in doms] + left
        order_d = [a if dark(a) else b for a, b in doms]
        check(sorted(order_l) == sorted(lights) and sorted(order_d) == sorted(darks),
              "the rows hold every square once")
        tl = {p: Square(side_length=s, fill_color=LIGHT, fill_opacity=1,
                        stroke_color=BLACK, stroke_width=1).move_to(
            [x0 + (i + 0.5) * s, yl, 0]) for i, p in enumerate(order_l)}
        td = {p: Square(side_length=s, fill_color=DARK, fill_opacity=1,
                        stroke_color=BLACK, stroke_width=1).move_to(
            [x0 + (i + 0.5) * s, yd, 0]) for i, p in enumerate(order_d)}
        n_l = tag("32", 28, LIGHT).next_to(tl[order_l[0]], LEFT, buff=0.16)
        n_d = tag("30", 28, "#C49A6C").next_to(td[order_d[0]], LEFT, buff=0.16)
        check(_inside(n_l) and _inside(n_d) and _inside(VGroup(*tl.values())),
              "rows inside")
        self.play(LaggedStart(*[TransformFromCopy(squares[p], tl[p])
                                for p in order_l], lag_ratio=0.04),
                  run_time=1.6)
        self.play(FadeIn(n_l), run_time=0.3)
        self.play(LaggedStart(*[TransformFromCopy(squares[p], td[p])
                                for p in order_d], lag_ratio=0.04),
                  run_time=1.5)
        self.play(FadeIn(n_d), run_time=0.3)

        # place the dominoes: each takes one square from each row
        on_board = [domino(a, b) for a, b in doms]
        in_rows = [RoundedRectangle(corner_radius=0.06, width=s - 0.06,
                                    height=2 * s - 0.06, stroke_color=DOM,
                                    stroke_width=3.5, fill_color=DOM,
                                    fill_opacity=0.08).move_to(
            [x0 + (i + 0.5) * s, (yl + yd) / 2, 0]) for i in range(len(doms))]
        self.play(LaggedStart(*[AnimationGroup(FadeIn(a), FadeIn(b))
                                for a, b in zip(on_board, in_rows)],
                              lag_ratio=0.25), run_time=5.0)

        # what is left: two light squares, and no dark square for them
        red = VGroup(*[Square(side_length=c - 0.03, stroke_color=RED,
                              stroke_width=5).move_to(Cpos(p)) for p in left],
                     *[Square(side_length=s - 0.02, stroke_color=RED,
                              stroke_width=4).move_to(tl[p]) for p in left])
        self.play(Create(red), run_time=0.8)
        cap = caption("a domino covers one light and one dark square:"
                      "    32  ≠  30", 30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K13

class K13_InclusionExclusion(Board):
    """Every element gets a ring each time it is counted. Adding |A|, |B|
    and |C| rings the elements of two sets twice and the common ones three
    times; taking away |A∩B|, |B∩C| and |C∩A| removes one ring per pair an
    element belongs to — one from the double-ringed, three from the
    common ones; putting back |A∩B∩C| restores those. Every element of
    A∪B∪C ends with exactly one ring. (14 elements, all 7 parts nonempty.)"""

    def construct(self):
        R = 2.0
        ctr = {"A": np.array([-4.15, 1.45, 0.0]), "B": np.array([-1.85, 1.45, 0.0]),
               "C": np.array([-3.0, -0.5, 0.0])}
        col = {"A": BLUE, "B": ORANGE, "C": GREEN}
        dots_at = {"A": [(-4.95, 2.35), (-5.3, 1.55), (-4.55, 1.75)],
                   "B": [(-1.1, 2.3), (-0.85, 1.55)],
                   "C": [(-3.65, -1.45), (-3.0, -1.8), (-2.35, -1.45)],
                   "AB": [(-3.0, 2.55), (-3.0, 1.85)],
                   "BC": [(-1.9, 0.2)],
                   "AC": [(-4.14, 0.54), (-4.42, -0.05)],
                   "ABC": [(-3.0, 0.82)]}
        pts = []
        for code, lst in dots_at.items():
            for p in lst:
                q = np.array([p[0], p[1], 0.0])
                member = {s for s in "ABC" if np.linalg.norm(q - ctr[s]) < R}
                check(member == set(code), f"dot {p} lies in region {code}")
                check(all(abs(np.linalg.norm(q - ctr[s]) - R) > 0.33 for s in "ABC"),
                      "rings clear of the circles")
                pts.append((q, member))
        check(all(np.linalg.norm(a[0] - b[0]) >= 0.62 for i, a in enumerate(pts)
                  for b in pts[i + 1:]), "dots spaced apart")
        check(all(len(v) > 0 for v in dots_at.values()) and len(dots_at) == 7,
              "all seven parts nonempty")

        def size(*sets):
            return sum(1 for _, m in pts if set(sets) <= m)

        nA, nB, nC = size("A"), size("B"), size("C")
        nAB, nBC, nCA, nABC = size("A", "B"), size("B", "C"), size("C", "A"), \
            size("A", "B", "C")
        union = len(pts)
        check(nA + nB + nC - nAB - nBC - nCA + nABC == union,
              "inclusion-exclusion on the picture")

        circ = {s: Circle(radius=R, stroke_color=col[s], stroke_width=4,
                          fill_color=col[s], fill_opacity=0.07).move_to(ctr[s])
                for s in "ABC"}
        labs = {"A": tag("A", 34, col["A"]).move_to(
                    ctr["A"] + 1.17 * R * np.array([-0.72, 0.69, 0])),
                "B": tag("B", 34, col["B"]).move_to(
                    ctr["B"] + 1.17 * R * np.array([0.72, 0.69, 0])),
                "C": tag("C", 34, col["C"]).move_to(ctr["C"] + 1.15 * R * DOWN)}
        check(all(_inside(m) for m in labs.values()), "set labels inside")
        for s_, m_ in labs.items():
            near = min(np.linalg.norm(m_.get_corner(v) - ctr[s_])
                       for v in (UL, UR, DL, DR))
            check(near > R + 0.04, "set label outside its circle")
        dots = VGroup(*[Dot(q, radius=0.07, color=WHITE) for q, _ in pts])
        self.play(*[Create(c) for c in circ.values()],
                  *[FadeIn(l) for l in labs.values()], run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.05), run_time=0.8)

        # the running count, one line per term
        rows = []

        def row(sign, term, num, color, y):
            t = VGroup(tag(sign, 30, color), tag(term, 30, color)).arrange(
                RIGHT, buff=0.18)
            t.move_to([1.0 + t.width / 2, y, 0])
            v = tag(str(num), 30, color)
            v.move_to([6.1 - v.width / 2, y, 0])
            g = VGroup(t, v)
            check(_inside(g), "panel row inside")
            rows.append(g)
            return g

        ys = [3.25, 2.6, 1.95, 1.3, 0.65, 0.0, -0.65]
        rings = [[] for _ in pts]
        radii = [0.13, 0.205, 0.28]

        def add(sets, color, sign, term, num, y):
            new = []
            for i, (q, m) in enumerate(pts):
                if set(sets) <= m:
                    r = Circle(radius=radii[len(rings[i])], stroke_color=color,
                               stroke_width=4).move_to(q)
                    rings[i].append(r)
                    new.append(r)
            check(len(new) == num, f"{term}: {num} elements")
            extra = []
            if len(sets) == 1:
                extra.append(circ[sets[0]].animate.set_fill(opacity=0.16))
            self.play(*[Create(r) for r in new], FadeIn(row(sign, term, num, color, y)),
                      *extra, run_time=1.0)

        add("A", col["A"], "+", "|A|", nA, ys[0])
        add("B", col["B"], "+", "|B|", nB, ys[1])
        add("C", col["C"], "+", "|C|", nC, ys[2])
        check(sorted(len(r) for r in rings) == sorted(len(m) for _, m in pts),
              "each element ringed once per set it is in")
        self.hold(0.5)

        def take(s1, s2, term, num, y):
            lens = Intersection(circ[s1], circ[s2], fill_color=WHITE,
                                fill_opacity=0.28, stroke_color=WHITE,
                                stroke_width=4)
            gone = []
            for i, (q, m) in enumerate(pts):
                if {s1, s2} <= m:
                    gone.append(rings[i].pop())
            check(len(gone) == num, f"{term}: {num} elements")
            self.play(FadeIn(lens), FadeIn(row("−", term, num, WHITE, y)),
                      *[r.animate.set_stroke(RED, 5) for r in gone], run_time=0.7)
            self.play(*[FadeOut(r, scale=1.4) for r in gone], FadeOut(lens),
                      run_time=0.6)

        take("A", "B", "|A∩B|", nAB, ys[3])
        take("B", "C", "|B∩C|", nBC, ys[4])
        take("C", "A", "|C∩A|", nCA, ys[5])
        i0 = next(i for i, (_, m) in enumerate(pts) if m == {"A", "B", "C"})
        check(len(rings[i0]) == 0, "the common part is now counted 0 times")
        check(all(len(rings[i]) == 1 for i in range(len(pts)) if i != i0),
              "everything else is counted once")

        core = Intersection(Intersection(circ["A"], circ["B"]), circ["C"],
                            fill_color=YELLOW, fill_opacity=0.3, stroke_color=YELLOW,
                            stroke_width=4)
        back = Circle(radius=radii[0], stroke_color=YELLOW, stroke_width=4
                      ).move_to(pts[i0][0])
        rings[i0].append(back)
        self.play(FadeIn(core), Create(back),
                  FadeIn(row("+", "|A∩B∩C|", nABC, YELLOW, ys[6])), run_time=0.9)
        self.play(FadeOut(core), run_time=0.4)
        check(all(len(r) == 1 for r in rings), "every element counted once")

        sep = Line([1.0, -1.05, 0], [6.1, -1.05, 0], color=GREY_B, stroke_width=2)
        tot = VGroup(VGroup(tag("=", 30, YELLOW_B), tag("|A∪B∪C|", 30, YELLOW_B)
                            ).arrange(RIGHT, buff=0.18),
                     tag(str(union), 30, YELLOW_B))
        tot[0].move_to([1.0 + tot[0].width / 2, -1.5, 0])
        tot[1].move_to([6.1 - tot[1].width / 2, -1.5, 0])
        self.play(Create(sep), FadeIn(tot),
                  *[Indicate(r[0], scale_factor=1.5, color=YELLOW_B) for r in rings],
                  run_time=1.2)
        cap = caption("|A∪B∪C| = |A| + |B| + |C| − |A∩B| − |B∩C| − |C∩A|"
                      " + |A∩B∩C|", 28)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== F33

def _reg_at(V, s, n, th):
    """Regular n-gon (2D points, counter-clockwise) with a vertex at V and
    its first side leaving V in direction th: its corner at V fills the
    angular sector [th, th + (n−2)π/n]."""
    pts = [np.asarray(V, float)[:2]]
    d = th
    for _ in range(n - 1):
        pts.append(pts[-1] + s * np.array([np.cos(d), np.sin(d)]))
        d += TAU / n
    return pts


def _clip_convex(subject, clip):
    """Sutherland–Hodgman: the part of convex polygon `subject` inside the
    convex counter-clockwise polygon `clip` (2D point lists)."""
    out = [np.asarray(p, float) for p in subject]
    m = len(clip)
    for i in range(m):
        a, b = np.asarray(clip[i], float), np.asarray(clip[(i + 1) % m], float)

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
    return out


class F33_RegularTilings(Board):
    """Copies of one regular polygon meeting at a point must fill the full
    turn exactly: k corners of (n−2)·180°/n make 360°. Triangles (6 × 60°),
    squares (4 × 90°) and hexagons (3 × 120°) close up; pentagons leave a
    gap with three and overlap with four; from the heptagon on the corner
    lies strictly between 120° and 180°, so two leave a gap and three
    overlap. On the scale of corner angles the only ones of the form
    360°/k are 60°, 90° and 120°. (Tilings by copies of a single regular
    polygon, meeting corner to corner.)"""

    def construct(self):
        rho = 0.9
        cols_good = [BLUE_D, TEAL_D, GREEN_D]
        cols_bad = [ORANGE, GOLD_D, MAROON_B, ORANGE]
        reach = {3: 1.0, 4: np.sqrt(2), 5: (1 + np.sqrt(5)) / 2,
                 6: 2.0, 7: np.sin(3 * PI / 7) / np.sin(PI / 7),
                 8: 1 / np.sin(PI / 8)}
        for n_, r_ in reach.items():
            P = _reg_at((0, 0), 1.0, n_, 0.0)
            check(abs(max(np.linalg.norm(p) for p in P) - r_) < 1e-9,
                  "farthest vertex from the corner")

        def alpha(n_):
            return PI - TAU / n_

        def panel(n_, centre, th0, good):
            V = np.asarray(centre, float)
            s = rho / reach[n_]
            a = alpha(n_)
            k = int(round(TAU / a)) if good else int(TAU // a)
            polys = [_reg_at(V, s, n_, th0 + j * a) for j in range(k)]
            for j in range(1, k):
                check(close(polys[j][1], polys[j - 1][-1]),
                      "neighbours share a side")
            if good:
                check(abs(k * a - TAU) < 1e-9, "the corners make a full turn")
                check(close(polys[0][1], polys[-1][-1]), "the ring closes")
            else:
                check(k * a < TAU - 1e-6 and (k + 1) * a > TAU + 1e-6,
                      "k corners fall short, k + 1 too many")
            return V, s, a, k, polys

        def mob(P, color, op=0.85):
            return mk([to3(p) for p in P], color, op, stroke_width=2)

        # ---------------- row 1: the three that close up
        xs = [-4.4, 0.0, 4.4]
        y1, y2 = 2.8, 0.42
        good = [panel(3, (xs[0], y1), 0.0, True), panel(4, (xs[1], y1), PI / 4, True),
                panel(6, (xs[2], y1), PI / 2, True)]
        firsts, fans, ring_mobs = [], [], []
        for (V, s, a, k, polys), n_ in zip(good, (3, 4, 6)):
            ms = [mob(P, cols_good[j % (3 if n_ == 6 else 2)]) for j, P in
                  enumerate(polys)]
            firsts.append(ms[0])
            for j in range(1, k):
                c = ms[0].copy().set_fill(cols_good[j % (3 if n_ == 6 else 2)])
                fans.append(Rotate(c, angle=j * a, about_point=to3(V)))
                ring_mobs.append((c, ms[j]))
        labs1 = [tag(t, 24).move_to([x, y1 - rho - 0.28, 0]) for t, x in
                 zip(("6 × 60° = 360°", "4 × 90° = 360°", "3 × 120° = 360°"), xs)]
        self.play(*[FadeIn(f) for f in firsts], run_time=0.6)
        self.play(*fans, run_time=1.8)
        for c, m_ in ring_mobs:
            check(close(c.get_center(), m_.get_center(), 1e-6),
                  "each copy turns into its place")
        self.play(*[FadeIn(l) for l in labs1],
                  *[FadeIn(Dot(to3(g[0]), radius=0.05)) for g in good], run_time=0.6)

        # ---------------- row 2: gap or overlap
        bad = [panel(5, (xs[0], y2), PI / 2 + 0.5 * (TAU - 3 * alpha(5)), False),
               panel(7, (xs[1], y2), PI / 2 + 0.5 * (TAU - 2 * alpha(7)), False),
               panel(8, (xs[2], y2), PI / 2 + 0.5 * (TAU - 2 * alpha(8)), False)]
        firsts2, fans2, gaps, extras, overl = [], [], [], [], []
        for (V, s, a, k, polys), n_ in zip(bad, (5, 7, 8)):
            m0 = mob(polys[0], cols_bad[0])
            firsts2.append(m0)
            for j in range(1, k):
                c = m0.copy().set_fill(cols_bad[j])
                fans2.append(Rotate(c, angle=j * a, about_point=to3(V)))
            th0 = float(np.arctan2(*(polys[0][1] - V)[::-1]))
            gap = Sector(radius=0.9 * s, start_angle=th0 + k * a,
                         angle=TAU - k * a, arc_center=to3(V), color=RED_E,
                         fill_opacity=0.35).set_stroke(RED_B, 3)
            gaps.append(gap)
            ext = DashedVMobject(Polygon(*[to3(p) for p in polys[0]],
                                         stroke_color=WHITE, stroke_width=3),
                                 num_dashes=6 * n_)
            extras.append(Rotate(ext, angle=k * a, about_point=to3(V)))
            extra_pts = _reg_at(V, s, n_, th0 + k * a)
            check(close(np.array(_rot2(polys[0][2], V, k * a)), extra_pts[2])
                  and close(np.array(_rot2(polys[0][-1], V, k * a)), extra_pts[-1]),
                  "the turned copy is the polygon whose overlap is shaded")
            ov = _clip_convex(extra_pts, polys[0])
            check(len(ov) >= 3 and area(ov) > 1e-4, "the extra copy overlaps")
            overl.append(mk([to3(p) for p in ov], RED_D, 0.9, stroke_color=RED_B,
                            stroke_width=2))
        labs2 = [VGroup(tag(t1, 19), tag(t2, 19)).arrange(DOWN, buff=0.08)
                 .move_to([x, y2 - rho - 0.36, 0]) for (t1, t2), x in
                 zip((("3 × 108° < 360°", "4 × 108° > 360°"),
                      ("2 × 128.6° < 360°", "3 × 128.6° > 360°"),
                      ("2 × 135° < 360°", "3 × 135° > 360°")), xs)]
        for i in range(2):
            check(_apart(labs2[i], labs2[i + 1], 0.2), "row-2 labels apart")
        for (V, s_, a_, k_, P_), l1 in zip(good, labs1):
            check(min(p[1] for P in P_ for p in P) > l1.get_top()[1] + 0.05,
                  "row-1 label below its figure")
        for (V, s_, a_, k_, P_), l2 in zip(bad, labs2):
            ext_ = _reg_at(V, s_, len(P_[0]), float(np.arctan2(
                *(P_[0][1] - V)[::-1])) + k_ * a_)
            ys_ = [p[1] for P in P_ + [ext_] for p in P]
            check(min(ys_) > l2.get_top()[1] + 0.05, "row-2 label below its figure")
            check(max(ys_) < min(l.get_bottom()[1] for l in labs1) - 0.08,
                  "row-2 figure below the row-1 labels")
        check(all(_inside(l) for l in labs1 + labs2), "labels inside the frame")
        self.play(*[FadeIn(f) for f in firsts2], run_time=0.5)
        self.play(*fans2, run_time=1.4)
        self.play(*[FadeIn(g) for g in gaps], run_time=0.6)
        self.hold(0.3)
        self.play(*extras, run_time=1.3)
        self.play(*[FadeIn(o) for o in overl], *[FadeIn(l) for l in labs2],
                  *[FadeIn(Dot(to3(b[0]), radius=0.05)) for b in bad], run_time=0.8)
        self.hold(0.5)

        # ---------------- the scale of corner angles
        ya = -1.85

        def X(deg):
            return -5.7 + (deg - 60.0) * 0.095

        axis = Line([X(60) - 0.3, ya, 0], [X(180) + 0.15, ya, 0], color=GREY_B,
                    stroke_width=3)
        ns = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 60]
        ups, nlabs = VGroup(), VGroup()
        for n_ in ns:
            d = 180 - 360 / n_
            okn = n_ in (3, 4, 6)
            ups.add(Line([X(d), ya, 0], [X(d), ya + 0.28, 0],
                         color=GREEN if okn else RED_B, stroke_width=4))
            if n_ <= 8:
                nlabs.add(tag(str(n_), 22, GREEN if okn else RED_B).move_to(
                    [X(d), ya + 0.47, 0]))
        nlabs.add(tag("…", 22, RED_B).move_to([X(150), ya + 0.47, 0]))
        nhead = tag("n", 24, GREY_A).move_to([X(60) - 0.6, ya + 0.47, 0])
        ks = [6, 5, 4, 3, 2]
        downs, klabs = VGroup(), VGroup()
        for k_ in ks:
            d = 360 / k_
            downs.add(Line([X(d), ya, 0], [X(d), ya - 0.28, 0], color=YELLOW_B,
                           stroke_width=4))
            klabs.add(tag(f"360°/{k_}", 19, YELLOW_B).move_to([X(d), ya - 0.47, 0]))
        band = Rectangle(width=X(180) - X(120) - 0.12, height=0.42,
                         fill_color=RED_D, fill_opacity=0.35, stroke_width=0
                         ).move_to([(X(120) + X(180)) / 2, ya + 0.08, 0])
        hits = VGroup(*[Circle(radius=0.17, stroke_color=GREEN, stroke_width=4
                               ).move_to([X(d), ya, 0]) for d in (60, 90, 120)])
        check(abs(X(180 - 360 / 7) - X(120)) > 0.3, "n = 7 tick clear of 120°")
        for grp in (nlabs, klabs, VGroup(nhead)):
            check(_inside(grp), "scale labels inside")
        check(nlabs.get_top()[1] < min(l.get_bottom()[1] for l in labs2) - 0.1,
              "scale clear of the panel labels")
        check(all(_apart(klabs[i], klabs[i + 1], 0.1) for i in range(len(ks) - 1)),
              "divisor labels apart")
        self.play(Create(axis), FadeIn(nhead), Create(ups), FadeIn(nlabs),
                  run_time=1.2)
        self.play(Create(downs), FadeIn(klabs), run_time=1.0)
        self.play(Create(hits), FadeIn(band), run_time=0.9)
        cap = caption("k · (n − 2)·180°/n  =  360°    ⟹    n = 3, 4, 6", 32)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== J32

class J32_DoublingSquare(Board):
    """The diagonal d cuts the square of side s into two half-squares. The
    square built on d has its centre at the far corner B of the original
    square, and its four quarters about B are the half-square turned
    through 0°, 90°, 180°, 270°. Four half-squares make two squares:
    d² = 2s². (The classical figure: a tilted square in a 2 × 2 grid.)"""

    def construct(self):
        B = np.array([1.0, 1.0])
        T0 = [np.array(p, float) for p in ((1, 0), (1, 1), (0, 1))]
        U = [np.array(p, float) for p in ((0, 0), (1, 0), (0, 1))]
        Dsq = [np.array(p, float) for p in ((1, 0), (2, 1), (1, 2), (0, 1))]
        Ts = [[_rot2(p, B, -j * PI / 2) for p in T0] for j in range(4)]
        check(abs(abs(area(T0)) - 0.5) < 1e-12 and abs(abs(area(U)) - 0.5) < 1e-12,
              "the diagonal halves the square")
        check(abs(sum(abs(area(t)) for t in Ts) - abs(area(Dsq))) < 1e-12,
              "four half-squares have the area of the d-square")
        for j in range(4):                       # each quarter of the d-square
            check(close(Ts[j][1], B), "right angle at the centre B")
            check(any(close(Ts[j][0], q) for q in Dsq)
                  and any(close(Ts[j][2], q) for q in Dsq),
                  "hypotenuse is a side of the d-square")
            check(close(Ts[j][2], Ts[(j + 1) % 4][0]), "quarters meet edge to edge")
        d = np.linalg.norm(T0[0] - T0[2])
        check(abs(d * d - 2.0) < 1e-12, "d² = 2s²")

        # where the four halves go: two s-squares on the right
        a_, b_, c_ = 3.25, 0.5, 5.05
        moves = [np.array([a_, b_]), np.array([c_, b_ - 1]),
                 np.array([a_ - 1, b_ - 1]), np.array([c_ - 1, b_])]
        Q1 = [(a_, b_), (a_ + 1, b_), (a_ + 1, b_ + 1), (a_, b_ + 1)]
        Q2 = [(c_, b_), (c_ + 1, b_), (c_ + 1, b_ + 1), (c_, b_ + 1)]
        for (i, j), Q in (((0, 2), Q1), ((1, 3), Q2)):
            vs = [p + moves[i] for p in Ts[i]] + [p + moves[j] for p in Ts[j]]
            check(all(any(close(v, q) for v in vs) for q in Q)
                  and all(any(close(v, q) for q in Q) for v in vs),
                  "two halves make a whole s-square")
            check(close(Ts[i][0] + moves[i], Ts[j][2] + moves[j])
                  or close(Ts[i][0] + moves[i], Ts[j][0] + moves[j])
                  or close(Ts[i][2] + moves[i], Ts[j][0] + moves[j]),
                  "the two halves share their hypotenuse")

        F = Frame(-0.65, 6.45, -0.62, 2.3)
        P = F.P
        sq = F.poly(((0, 0), (1, 0), (1, 1), (0, 1)), BLUE_D)
        ls1 = tag("s", 28).move_to(P((0.5, -0.2)))
        ls2 = tag("s", 28).move_to(P((-0.2, 0.5)))
        self.play(FadeIn(sq), FadeIn(ls1), FadeIn(ls2), run_time=1.0)

        diag = Line(P(T0[0]), P(T0[2]), color=YELLOW_B, stroke_width=5)
        ld = tag("d", 30, YELLOW_B).move_to(P((0.36, 0.36)))
        u = F.poly(U, BLUE_D)
        t0 = F.poly(T0, TEAL_D)
        self.play(Create(diag), FadeIn(ld), run_time=0.8)
        self.remove(sq)
        self.add(u, t0, diag, ld)
        self.play(t0.animate.set_fill(TEAL_D), run_time=0.3)

        grid = VGroup(DashedVMobject(Polygon(P((0, 0)), P((2, 0)), P((2, 2)),
                                             P((0, 2)), stroke_color=GREY_D,
                                             stroke_width=2), num_dashes=48),
                      DashedLine(P((1, 0)), P((1, 2)), color=GREY_D,
                                 stroke_width=2, dash_length=0.08),
                      DashedLine(P((0, 1)), P((2, 1)), color=GREY_D,
                                 stroke_width=2, dash_length=0.08))
        dsq = Polygon(*[P(p) for p in Dsq], stroke_color=YELLOW_B, stroke_width=5)
        self.play(FadeIn(grid), Create(dsq), run_time=1.0)
        self.bring_to_front(diag, ld)

        # the half-square, turned about B, fills the square on d
        copies = []
        prev = t0
        for j in range(1, 4):
            c = prev.copy()
            self.add(c)
            self.play(Rotate(c, angle=-PI / 2, about_point=P(B)),
                      run_time=0.9 if j == 1 else 0.7)
            check(_same_vertices(c, [P(q) for q in Ts[j]]),
                  "the turned half lands on its quarter")
            copies.append(c)
            prev = c
        orig = Polygon(P((0, 0)), P((1, 0)), P((1, 1)), P((0, 1)),
                       stroke_color=WHITE, stroke_width=3)
        self.add(orig)
        self.bring_to_front(dsq, ld)
        ldd = tag("d²", 32, YELLOW_B).move_to(P((1.78, 1.78)))

        def inv(pt):
            return ((pt[0] - F.centre[0]) / F.k + F.cx,
                    (pt[1] - F.centre[1]) / F.k + F.cy)

        check(all(inv(ldd.get_corner(v))[0] + inv(ldd.get_corner(v))[1] > 3.05
                  and max(inv(ldd.get_corner(v))) < 2.0 for v in (UL, UR, DL, DR)),
              "d² label outside the d-square, inside the grid")
        self.play(FadeIn(ldd), run_time=0.5)
        self.hold(0.5)

        # four halves = two squares
        pieces = [t0] + copies
        movers = [m.copy() for m in pieces]
        self.play(*[m.animate.shift(F.k * np.array([mv[0], mv[1], 0.0]))
                    for m, mv in zip(movers, moves)], run_time=1.8)
        for m, t, mv in zip(movers, Ts, moves):
            check(_same_vertices(m, [P(q + mv) for q in t]), "translated half lands")
        out1 = Polygon(*[P(p) for p in Q1], stroke_color=WHITE, stroke_width=4)
        out2 = Polygon(*[P(p) for p in Q2], stroke_color=WHITE, stroke_width=4)
        eq = tag("=", 40).move_to(P((2.62, 1.0)))
        plus = tag("+", 40).move_to(P((c_ - 0.4, 1.0)))
        l1 = tag("s²", 30).move_to(P((a_ + 0.5, b_ - 0.25)))
        l2 = tag("s²", 30).move_to(P((c_ + 0.5, b_ - 0.25)))
        check(_inside(VGroup(out2, l2, ldd)), "right side inside the frame")
        self.play(Create(out1), Create(out2), FadeIn(eq), FadeIn(plus),
                  FadeIn(l1), FadeIn(l2), run_time=0.9)
        cap = caption("d²  =  4 · ½s²  =  2s²", 36)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== J15

def _euclid_squares(w, h):
    """Cut squares off a w × h rectangle (corner at the origin): always the
    largest square on the shorter side. Returns [(x, y, side)] and the
    sizes (w, h) of the rectangle before each cut."""
    x0 = y0 = 0
    out, sizes = [], []
    while w > 0 and h > 0:
        sizes.append((w, h))
        if w >= h:
            out.append((x0, y0, h))
            x0 += h
            w -= h
        else:
            out.append((x0, y0, w))
            y0 += w
            h -= w
    return out, sizes


class J15_EuclidAlgorithm(Board):
    """Cut the largest possible square off the a × b rectangle, again and
    again. Each cut replaces the sides (a, b) by (a − b, b): a length that
    measures both a and b measures a − b, and one that measures b and a − b
    measures a, so the common measures — and the greatest one — never
    change. The cutting ends with a square, which measures itself; laying
    that last square back over every piece tiles the whole rectangle, so
    its side measures both a and b: it is gcd(a, b). Shown for 45 × 12."""

    def construct(self):
        from math import gcd
        a, b = 45, 12
        sq, sizes = _euclid_squares(a, b)
        g = sq[-1][2]
        check(g == gcd(a, b) == 3, "the last square is gcd(45, 12) = 3")
        check(sum(s_ * s_ for _, _, s_ in sq) == a * b,
              "the squares fill the rectangle")
        check(all(gcd(w, h) == g for w, h in sizes), "gcd unchanged by every cut")
        check(all(x % g == 0 and y % g == 0 and s_ % g == 0 for x, y, s_ in sq),
              "the last square measures every piece")
        check([s_ for _, _, s_ in sq] == [12, 12, 12, 9, 3, 3, 3],
              "cuts 12, 12, 12, 9, 3, 3, 3")

        u = 0.23
        O = np.array([-4.3, -1.6, 0.0])

        def S(x, y):
            return O + u * np.array([x, y, 0.0])

        def rect_mob(x, y, w, h, col, op=FILL, sw=2, stroke=WHITE):
            return mk([S(x, y), S(x + w, y), S(x + w, y + h), S(x, y + h)], col,
                      op, stroke_width=sw, stroke_color=stroke)

        frame = rect_mob(0, 0, a, b, BLACK, 0.0, 4, YELLOW_B)
        la = tag("45", 30, YELLOW_B).move_to(S(a / 2, 0) + DOWN * 0.36)
        lb = tag("12", 30, YELLOW_B)
        lb.move_to(S(0, b / 2) + LEFT * (0.2 + lb.width / 2))
        check(_inside(frame), "rectangle inside")
        self.play(Create(frame), FadeIn(la), FadeIn(lb), run_time=1.0)

        # the gcd chain: one more term after every cut
        def term(w, h):
            return tag(f"gcd({max(w, h)}, {min(w, h)})", 26, GREY_A)

        t = [term(w, h) for w, h in sizes]
        eqs = [tag("=", 26, GREY_A) for _ in range(len(sizes))]
        last = tag(str(g), 28, YELLOW_B)
        row1 = VGroup(t[0], eqs[0], t[1], eqs[1], t[2], eqs[2], t[3])
        row2 = VGroup(eqs[3], t[4], eqs[4], t[5], eqs[5], t[6], eqs[6], last)
        row1.arrange(RIGHT, buff=0.18).move_to([0.6, 3.3, 0])
        row2.arrange(RIGHT, buff=0.18)
        row2.move_to([row1.get_left()[0] + 0.5 + row2.width / 2, 2.6, 0])
        reveal = [(eqs[i], t[i + 1]) for i in range(len(sizes) - 1)] + \
                 [(eqs[-1], last)]
        check(_inside(row1) and _inside(row2), "chain inside the frame")
        check(row2.get_bottom()[1] > frame.get_top()[1] + 0.4, "chain clear")
        self.play(FadeIn(t[0]), run_time=0.5)

        cols = {12: BLUE_D, 9: TEAL_D, 3: ORANGE}
        fsz = {12: 32, 9: 30, 3: 22}
        pieces = []
        rem = None
        for i, (x, y, s_) in enumerate(sq):
            pc = rect_mob(x, y, s_, s_, cols[s_])
            lab = tag(str(s_), fsz[s_]).move_to(S(x + s_ / 2, y + s_ / 2))
            pieces.append(VGroup(pc, lab))
            anims = [FadeIn(pc), FadeIn(lab), *[FadeIn(m) for m in reveal[i]]]
            if i + 1 < len(sq):              # outline the rectangle left over
                x2, y2 = sq[i + 1][0], sq[i + 1][1]
                w2, h2 = sizes[i + 1]
                nr = DashedVMobject(rect_mob(x2, y2, w2, h2, BLACK, 0.0, 3, WHITE),
                                    num_dashes=40)
                anims.append(Transform(rem, nr) if rem is not None else FadeIn(nr))
                if rem is None:
                    rem = nr
            else:
                anims.append(FadeOut(rem))
            self.play(*anims, run_time=0.8 if s_ > 3 else 0.6)
        self.bring_to_front(frame)
        self.hold(0.5)

        # the last square measures everything: lay it over every piece
        unit = pieces[-1][0]
        cells = []
        order = [3, 2, 1, 0]                     # back up the chain of cuts
        for k_ in order:
            x, y, s_ = sq[k_]
            cells.append([rect_mob(x + i * g, y + j * g, g, g, BLACK, 0.0, 2.5,
                                   YELLOW_B)
                          for j in range(s_ // g) for i in range(s_ // g)])
        n_cells = sum(len(c) for c in cells) + 3
        check(n_cells == (a // g) * (b // g), "15 × 4 copies of the last square")
        self.play(Indicate(unit, color=YELLOW_B, scale_factor=1.15), run_time=0.6)
        for k_, grp in zip(order, cells):
            x, y, s_ = sq[k_]
            lab = pieces[k_][1]
            # a label stays only if it sits inside one cell of the grid
            mid = (s_ // g) % 2 == 1
            top = tag(str(s_), 26).move_to(S(x + s_ / 2, y + s_) + UP * 0.25)
            check(mid or (top.get_bottom()[1] > frame.get_top()[1] + 0.04
                          and top.get_top()[1] < row2.get_bottom()[1] - 0.1),
                  "side label between the rectangle and the chain")
            extra = [] if mid else [FadeOut(lab), FadeIn(top)]
            if mid:
                check(close(lab.get_center(), S(x + s_ / 2, y + s_ / 2))
                      and lab.width < g * u - 0.2, "label inside the middle cell")
            self.play(LaggedStart(*[TransformFromCopy(unit, c) for c in grp],
                                  lag_ratio=0.06), *extra,
                      run_time=0.9 if len(grp) < 10 else 1.1)
        ticks = VGroup(*[Line(S(i * g, 0), S(i * g, 0) + DOWN * 0.12,
                              color=YELLOW_B, stroke_width=3)
                         for i in range(a // g + 1)],
                       *[Line(S(0, j * g), S(0, j * g) + LEFT * 0.12,
                              color=YELLOW_B, stroke_width=3)
                         for j in range(b // g + 1)])
        la2 = tag("45 = 15 × 3", 30, YELLOW_B).move_to(la)
        lb2 = tag("12 = 4 × 3", 26, YELLOW_B)
        lb2.move_to(S(0, b / 2) + LEFT * (0.24 + lb2.width / 2))
        check(_inside(lb2) and _inside(la2), "measure labels inside")
        outl = VGroup(*[Polygon(S(x, y), S(x + s_, y), S(x + s_, y + s_),
                                S(x, y + s_), stroke_color=WHITE, stroke_width=4)
                        for x, y, s_ in sq])
        self.play(Create(ticks), Create(outl), Transform(la, la2),
                  Transform(lb, lb2), run_time=0.9)
        cap = caption("gcd(a, b)  =  gcd(a − b, b)     ⟹     "
                      "gcd(45, 12)  =  3", 32)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)
