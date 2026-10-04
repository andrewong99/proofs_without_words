# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2i.py — proofs without words, 2D (manim):
#     K3  symmetry of binomial coefficients   K14 squares on a grid
#     K5  Catalan numbers by reflection       K15 rectangles on a grid
#     K12 trominoes on a defective board      K10 regions of a circle (caution)
#     K4  sum of squares of a row             K16 lines cutting the plane
#     K18 six people and Ramsey               K20 the handshake lemma
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Counting scenes enumerate a small case completely and check the counts and
# the bijections with check(...). The arrangement scenes (K10, K16) compute
# every region by cutting convex faces and check the counts against the
# formula, so a wrong picture fails the render. A reflection is shown as
# what it is in space: a half-turn about the mirror line (rigid throughout).

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


def _east(p, i):
    """True if step i of the lattice path p goes east."""
    return p[i + 1][0] == p[i][0] + 1


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
    return Arrow(to3(p), to3(q), buff=0, color=color, stroke_width=width,
                 tip_length=tip, max_tip_length_to_length_ratio=0.5,
                 max_stroke_width_to_length_ratio=100)


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


def _clear_of_segment(m, p, q, gap=0.05, what="a label"):
    """Fail the render if the segment pq passes through m's bounding box."""
    lo, hi = m.get_corner(DL), m.get_corner(UR)
    for f in np.linspace(0, 1, 240):
        x = to3(p) + (to3(q) - to3(p)) * f
        if (lo[0] - gap < x[0] < hi[0] + gap
                and lo[1] - gap < x[1] < hi[1] + gap):
            check(False, f"a line runs through {what}")


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption inside the frame, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        check(_inside(m), f"{type(m).__name__} inside the safe area")
    texts = [t for m in shown for t in m.get_family()
             if isinstance(t, Text) and t.get_fill_opacity() > 0.01]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_apart(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


_DIAG = np.array([1.0, 1.0, 0.0])


def _flip(mob, p, d, target=None, **kw):
    """A reflection in the line through screen point p with direction d,
    shown as what it is in space: a half-turn about that line. Rigid on
    every frame; meanwhile the line may travel straight to `target`."""
    start = mob.copy()
    p = to3(p)
    axis = to3(d) / np.linalg.norm(to3(d))
    sh = (to3(target) - p) if target is not None else np.zeros(3)

    def upd(m, a):
        m.become(start.copy().rotate(a * PI, axis=axis, about_point=p)
                 .shift(a * sh))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _ends(m):
    """Start and end points of a Line / Arrow, flattened to the screen."""
    s, e = m.get_start(), m.get_end()
    return np.array([s[0], s[1], 0.0]), np.array([e[0], e[1], 0.0])


# ===================================================================== K3

class K3_BinomialSymmetry(Board):
    """A path of unit steps east and north from O to T = (k, n − k) has n
    steps, k of them east, so there are C(n, k) such paths. Reflect the
    picture in the diagonal through O (a half-turn about it in space):
    every east step becomes a north step and every north step an east
    step, so the path becomes a path to T′ = (n − k, k), with n − k east
    steps. Reflecting twice gives the path back, so the reflection pairs
    the two sets of paths one to one: C(n, k) = C(n, n − k).
    Shown for n = 5, k = 2: one path step by step, then all ten pairs."""

    def construct(self):
        n, k = 5, 2
        W, H = k, n - k
        paths = _lattice_paths(W, H)
        choices = list(combinations(range(n), W))
        mirror = [[(y, x) for x, y in p] for p in paths]
        check(len(paths) == _binom(n, k) == 10, "C(5, 2) = 10 paths to (2, 3)")
        check(sorted(map(tuple, mirror)) == sorted(map(tuple, _lattice_paths(H, W))),
              "the mirror images are exactly the paths to (3, 2)")
        check(len(set(map(tuple, mirror))) == len(paths), "no two share an image")
        for p, q in zip(paths, mirror):
            check([(y, x) for x, y in q] == p, "reflecting twice gives the path back")
            for i in range(n):
                d = (p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1])
                e = (q[i + 1][0] - q[i][0], q[i + 1][1] - q[i][1])
                check(e == (d[1], d[0]), "east and north steps swap")
        for nn in range(16):
            for kk in range(nn + 1):
                check(_binom(nn, kk) == _binom(nn, nn - kk), "the row is symmetric")

        CE, CN = ORANGE, BLUE_D
        cb = 0.95
        O = np.array([-5.6, -1.1, 0.0])

        def B(p):
            return O + cb * np.array([p[0], p[1], 0.0])

        gridT = _grid_lines(W, H, cb, O, GREY_B, 2)
        dO = Dot(B((0, 0)), radius=0.08)
        dT = Dot(B((W, H)), radius=0.1, color=YELLOW_B)
        lO = tag("O", 24).next_to(dO, DR, buff=0.08)
        lT = tag("(2, 3)", 24, YELLOW_B).next_to(dT, UR, buff=0.06)
        self.play(Create(gridT), FadeIn(dO), FadeIn(dT), FadeIn(lO), FadeIn(lT),
                  run_time=1.0)

        # one path, step by step, and its word of steps
        demo = choices.index((1, 4))
        pts = paths[demo]
        sc, y1, y2 = 0.48, -1.95, -2.6

        def slot_c(i, y):
            return np.array([O[0] + (i + 0.5) * sc, y, 0.0])

        def slot(i, y, east):
            c = slot_c(i, y)
            d = RIGHT if east else UP
            return VGroup(Square(side_length=sc, fill_color=CE if east else CN,
                                 fill_opacity=0.85, stroke_color=WHITE,
                                 stroke_width=2).move_to(c),
                          _arrow(c - d * 0.14, c + d * 0.14, WHITE, width=5,
                                 tip=0.13))

        steps, word1 = VGroup(), VGroup()
        for i in range(n):
            east = _east(pts, i)
            ar = _arrow(B(pts[i]), B(pts[i + 1]), CE if east else CN, width=8,
                        tip=0.24)
            sl = slot(i, y1, east)
            steps.add(ar)
            word1.add(sl)
            self.play(GrowArrow(ar), FadeIn(sl), run_time=0.45)
        check(lO.get_bottom()[1] > word1.get_top()[1] + 0.05, "O label clear of the word")

        # the mirror: the diagonal through O
        mir = DashedLine(B((-0.45, -0.45)), B((3.5, 3.5)), color=YELLOW_B,
                         stroke_width=3, dash_length=0.12)
        self.play(Create(mir), run_time=0.7)
        img = VGroup(gridT.copy(), steps.copy(), dT.copy())
        self.add(img)
        self.play(_flip(img, O, _DIAG),
                  gridT.animate.set_stroke(opacity=0.3),
                  steps.animate.set_opacity(0.3), dT.animate.set_opacity(0.3),
                  run_time=1.8)
        q = mirror[demo]
        for i in range(n):
            s, e = _ends(img[1][i])
            check(close(s, B(q[i]), 1e-6) and close(e, B(q[i + 1]), 1e-6),
                  "the flipped step lands on the mirror path")
        check(close(img[2].get_center()[:2], B((H, W))[:2], 1e-6), "T lands on T′")
        lT2 = tag("(3, 2)", 24, YELLOW_B).next_to(B((H, W)), UR, buff=0.12)
        _clear_of_segment(lO, mir.get_start(), mir.get_end(), what="O")
        _clear_of_segment(lO, B((0, 0)), B((1, 0)), gap=0.04, what="O")
        _clear_of_segment(lT, mir.get_start(), mir.get_end(), what="(2, 3)")
        _clear_of_segment(lT2, mir.get_start(), mir.get_end(), what="(3, 2)")
        # east and north swap: recolour each step by its new direction
        self.play(*[img[1][i].animate.set_color(CE if _east(q, i) else CN)
                    for i in range(n)], FadeIn(lT2), run_time=0.8)

        word2 = word1.copy()
        self.play(word2.animate.shift((y2 - y1) * UP), run_time=0.6)
        self.play(*[_flip(word2[i][1], slot_c(i, y2), _DIAG) for i in range(n)],
                  *[word2[i][0].animate.set_fill(CE if _east(q, i) else CN)
                    for i in range(n)], run_time=1.2)
        for i in range(n):
            s, e = _ends(word2[i][1])
            d = RIGHT if _east(q, i) else UP
            check(close(e - s, 0.28 * d, 1e-6), "each letter of the word turned")
        self.hold(0.5)

        # every path to (2, 3), each flipped onto its mirror image
        cm = 0.3
        xs = [-1.3 + 1.28 * j for j in range(5)]
        ysA, ysB = [2.55, 1.25], [-0.85, -2.05]

        def mini(p, org, Wm, Hm):
            g = VGroup(_grid_lines(Wm, Hm, cm, org, GREY_D, 1))
            for i in range(n):
                g.add(Line(to3(org) + cm * np.array([*p[i], 0.0]),
                           to3(org) + cm * np.array([*p[i + 1], 0.0]),
                           color=CE if _east(p, i) else CN, stroke_width=4))
            return g

        orgA = [np.array([xs[j % 5], ysA[j // 5], 0.0]) for j in range(10)]
        orgB = [np.array([xs[j % 5], ysB[j // 5], 0.0]) for j in range(10)]
        blockA = [mini(p, orgA[j], W, H) for j, p in enumerate(paths)]
        self.play(LaggedStart(*[FadeIn(m) for m in blockA], lag_ratio=0.12),
                  run_time=1.6)
        labA = VGroup(tag("C(5, 2)", 26, YELLOW_B), tag("= 10", 26, YELLOW_B)
                      ).arrange(DOWN, buff=0.12).move_to(
            [5.65, (ysA[0] + ysA[1] + H * cm) / 2, 0])
        self.play(FadeIn(labA), run_time=0.5)

        flips = [m.copy() for m in blockA]
        self.add(*flips)
        self.play(*[_flip(f, orgA[j], _DIAG, target=orgB[j])
                    for j, f in enumerate(flips)], run_time=2.0)
        for j, f in enumerate(flips):
            for i in range(n):
                s, e = _ends(f[1 + i])
                check(close(s, orgB[j] + cm * np.array([*mirror[j][i], 0.0]), 1e-6)
                      and close(e, orgB[j] + cm * np.array([*mirror[j][i + 1], 0.0]),
                                1e-6), "each mini path lands on its mirror image")
        self.play(*[f[1 + i].animate.set_color(CE if _east(mirror[j], i) else CN)
                    for j, f in enumerate(flips) for i in range(n)], run_time=0.8)
        labB = VGroup(tag("C(5, 3)", 26, YELLOW_B), tag("= 10", 26, YELLOW_B)
                      ).arrange(DOWN, buff=0.12).move_to(
            [5.65, (ysB[0] + ysB[1] + W * cm) / 2, 0])
        self.play(FadeIn(labB), run_time=0.5)
        frames = VGroup(SurroundingRectangle(blockA[demo], buff=0.08, color=YELLOW_B,
                                             stroke_width=3),
                        SurroundingRectangle(flips[demo], buff=0.08, color=YELLOW_B,
                                             stroke_width=3))
        self.play(Create(frames), run_time=0.6)
        for m_ in (labA, labB, VGroup(*blockA), VGroup(*flips)):
            check(_inside(m_), "gallery inside the frame")
        check(labA.get_left()[0] > blockA[4].get_right()[0] + 0.2
              and labB.get_left()[0] > flips[4].get_right()[0] + 0.2,
              "counts clear of the blocks")
        check(VGroup(*blockA).get_left()[0] > lT2.get_right()[0] + 0.25,
              "gallery clear of the demo")
        cap = caption("C(n, k)  =  C(n, n − k)", 36)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K14

class K14_SquaresOnGrid(Board):
    """A k × k square drawn on the lines of an n × n grid is fixed by its
    lower-left cell, and that cell can sit in any of the first n − k + 1
    columns and any of the first n − k + 1 rows. Slide the square through
    all its places and keep its corner cells: they fill an
    (n − k + 1) × (n − k + 1) square. For n = 4 the four sizes give
    1² + 2² + 3² + 4² = 30 squares (squares whose sides lie on the grid
    lines; tilted squares are not counted)."""

    def construct(self):
        n = 4
        pos = {kk: [(i, j) for j in range(n - kk + 1) for i in range(n - kk + 1)]
               for kk in range(1, n + 1)}
        brute = [(i, j, s) for s in range(1, n + 1) for i in range(n + 1)
                 for j in range(n + 1) if i + s <= n and j + s <= n]
        check(len(brute) == sum(len(v) for v in pos.values()) == 30,
              "30 squares on the 4 × 4 grid")
        for kk in range(1, n + 1):
            check(len(pos[kk]) == (n - kk + 1) ** 2, "(n − k + 1)² places")
            check(sorted((i, j) for i, j, s in brute if s == kk) == sorted(pos[kk]),
                  "the corner cells of the k × k squares fill an (n−k+1)² block")
        for nn in range(1, 30):
            check(sum((nn - kk + 1) ** 2 for kk in range(1, nn + 1))
                  == sum(j * j for j in range(1, nn + 1)), "reversed order")

        col = {1: BLUE_D, 2: TEAL_D, 3: ORANGE, 4: YELLOW_E}
        u = 0.95
        G0 = np.array([-6.3, -2.35, 0.0])

        def C(i, j):
            return G0 + u * np.array([i + 0.5, j + 0.5, 0.0])

        grid = _grid_lines(n, n, u, G0, GREY_B, 2)
        ln = tag("n = 4", 28).move_to(G0 + np.array([n * u / 2, n * u + 0.42, 0]))
        self.play(Create(grid), FadeIn(ln), run_time=1.0)

        # the panel: one block of corner cells per size, the smallest block
        # first, so each new block lands nearer the grid than those before it
        pu, gap, ybase = 0.42, 0.74, -0.45
        order = list(range(n, 0, -1))              # k = 4, 3, 2, 1: m = 1 … 4
        wid = {kk: (n - kk + 1) * pu for kk in order}
        x = 1.75 - (sum(wid.values()) + gap * (n - 1)) / 2
        bx = {}
        for kk in order:
            bx[kk] = x
            x += wid[kk] + gap
        cx = {kk: bx[kk] + wid[kk] / 2 for kk in order}
        yhead, y1, y2 = ybase + n * pu + 0.32, ybase - 0.36, ybase - 1.05

        def panel_cell(kk, i, j):
            return Square(side_length=pu - 0.05, fill_color=col[kk], fill_opacity=0.85,
                          stroke_width=0).move_to(
                [bx[kk] + (i + 0.5) * pu, ybase + (j + 0.5) * pu, 0])

        heads, blocks, sqrs = {}, {}, {}
        rt = {1: 0.12, 2: 0.24, 3: 0.4, 4: 0.6}
        for kk in range(1, n + 1):
            m = n - kk + 1
            sq = Square(side_length=kk * u, stroke_color=col[kk], stroke_width=7,
                        fill_color=col[kk], fill_opacity=0.16)
            sq.move_to(G0 + u * np.array([kk / 2, kk / 2, 0.0]))
            anchors = VGroup()
            self.play(FadeIn(sq), run_time=0.35)
            for i, j in pos[kk]:
                target = G0 + u * np.array([i + kk / 2, j + kk / 2, 0.0])
                a = Square(side_length=u - 0.14, fill_color=col[kk], fill_opacity=0.85,
                           stroke_width=0).move_to(C(i, j))
                anchors.add(a)
                self.play(sq.animate.move_to(target), FadeIn(a), run_time=rt[kk])
                check(close(sq.get_corner(DL), G0 + u * np.array([i, j, 0.0]), 1e-9),
                      "the square's corner cell is the marked cell")
                check(sq.get_corner(UR)[0] <= G0[0] + n * u + 1e-9
                      and sq.get_corner(UR)[1] <= G0[1] + n * u + 1e-9,
                      "the square stays on the grid")
            self.bring_to_front(sq)
            blocks[kk] = VGroup(*[panel_cell(kk, i, j) for i, j in pos[kk]])
            check(all(blocks[kk].get_right()[0] + 0.3 < blocks[q].get_left()[0]
                      for q in blocks if q < kk), "the new block lands left of the others")
            heads[kk] = tag(f"{kk} × {kk}", 24, col[kk]).move_to([cx[kk], yhead, 0])
            sqrs[kk] = tag(f"{m}²", 30, col[kk]).move_to([cx[kk], y1, 0])
            self.play(TransformFromCopy(anchors, blocks[kk]), FadeIn(heads[kk]),
                      run_time=0.9)
            self.play(FadeIn(sqrs[kk]), FadeOut(anchors), FadeOut(sq), run_time=0.45)

        # the sum: each number under its block, plus signs midway between
        nums = {kk: tag(str((n - kk + 1) ** 2), 30, col[kk]).move_to([cx[kk], y2, 0])
                for kk in order}

        def pluses(row, y):
            out = VGroup()
            for a_, b_ in zip(order, order[1:]):
                xm = (row[a_].get_right()[0] + row[b_].get_left()[0]) / 2
                out.add(tag("+", 30, GREY_A).move_to([xm, y, 0]))
            return out

        plus1, plus2 = pluses(sqrs, y1), pluses(nums, y2)
        res = tag("=  30", 32, YELLOW_B)
        res.move_to([bx[1] + wid[1] + 0.3 + res.width / 2, y2, 0])
        self.play(FadeIn(plus1), run_time=0.4)
        self.play(*[FadeIn(nums[kk]) for kk in order], FadeIn(plus2), run_time=0.7)
        self.play(FadeIn(res, shift=LEFT * 0.1), run_time=0.6)
        parts = ([ln] + list(heads.values()) + list(sqrs.values()) + list(nums.values())
                 + list(plus1) + list(plus2) + [res])
        for p_ in parts + list(blocks.values()):
            check(_inside(p_), "panel inside the frame")
        for i_ in range(len(parts)):
            for j_ in range(i_ + 1, len(parts)):
                check(_apart(parts[i_], parts[j_], 0.08), "panel labels apart")
        for kk in order:
            check(heads[kk].get_bottom()[1] > blocks[kk].get_top()[1] + 0.1
                  and sqrs[kk].get_top()[1] < blocks[kk].get_bottom()[1] - 0.1,
                  "labels clear of their block")
        check(min(b.get_left()[0] for b in blocks.values()) > G0[0] + n * u + 0.5,
              "blocks clear of the grid")
        cap = caption("k × k :  (n − k + 1)² places   ⟹   1² + 2² + 3² + … + n²", 30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K5

def _first_touch(p):
    """Index of the first point of the path on the line y = x + 1."""
    return next((i for i, (x, y) in enumerate(p) if y == x + 1), None)


def _reflect_tail(p):
    """André's reflection: after the first touch of y = x + 1, reflect the
    rest of the path in that line, (x, y) -> (y − 1, x + 1)."""
    i = _first_touch(p)
    return p[:i + 1] + [(y - 1, x + 1) for x, y in p[i + 1:]]


class K5_CatalanReflection(Board):
    """A path of n steps east and n steps north from O to T = (n, n) is good
    if it never rises above the diagonal. A bad path touches the line
    y = x + 1; reflect its part after the first touch in that line: it
    becomes a path to T* = (n − 1, n + 1). Every path to T* touches the
    line too (T* lies beyond it, O before it), and reflecting back undoes
    the change, so the bad paths match the C(2n, n + 1) paths to T* one to
    one: Cₙ = C(2n, n) − C(2n, n + 1). Shown for n = 3: all 20 paths."""

    def construct(self):
        n = 3
        T, Ts = (n, n), (n - 1, n + 1)
        allp = _lattice_paths(n, n)
        good = [p for p in allp if all(y <= x for x, y in p)]
        bad = [p for p in allp if any(y > x for x, y in p)]
        check(len(allp) == _binom(2 * n, n) == 20 and len(good) == 5
              and len(bad) == 15, "20 paths: 5 good, 15 bad")
        check(all(_first_touch(p) is not None for p in bad)
              and all(_first_touch(p) is None for p in good),
              "bad means touching y = x + 1")
        imgs = [_reflect_tail(p) for p in bad]
        check(sorted(map(tuple, imgs)) == sorted(map(tuple, _lattice_paths(*Ts))),
              "the reflected bad paths are exactly the paths to (2, 4)")
        check(all(_reflect_tail(q) == p for p, q in zip(bad, imgs)),
              "reflecting back gives the bad path again")
        check(len(_lattice_paths(*Ts)) == _binom(2 * n, n + 1) == 15, "C(6, 4) = 15")
        for nn in range(1, 8):
            gp = [p for p in _lattice_paths(nn, nn) if all(y <= x for x, y in p)]
            check(len(gp) == _binom(2 * nn, nn) - _binom(2 * nn, nn + 1)
                  == _binom(2 * nn, nn) // (nn + 1), "Catalan numbers")

        CG, CF, CR = GREEN_C, ORANGE, RED_C
        cb = 0.92
        O = np.array([-6.15, -2.45, 0.0])

        def B(p):
            return O + cb * np.array([p[0], p[1], 0.0])

        grid = _grid_lines(n, n, cb, O, GREY_B, 2)
        top = VGroup(*[Line(B((i, n)), B((i, n + 1)), stroke_color=GREY_D,
                            stroke_width=2) for i in range(n + 1)],
                     Line(B((0, n + 1)), B((n, n + 1)), stroke_color=GREY_D,
                          stroke_width=2))
        diag = DashedLine(B((0, 0)), B((n, n)), color=GREY_A, stroke_width=2.5,
                          dash_length=0.1)
        dO = Dot(B((0, 0)), radius=0.08)
        dT = Dot(B(T), radius=0.1, color=YELLOW_B)
        lO = tag("O", 24).next_to(dO, DL, buff=0.06)
        lT = tag("(3, 3)", 24, YELLOW_B).next_to(dT, RIGHT, buff=0.14)
        self.play(Create(grid), Create(diag), FadeIn(dO), FadeIn(dT), FadeIn(lO),
                  FadeIn(lT), run_time=1.0)

        # a good path stays on or below the diagonal
        gdemo = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (3, 3)]
        check(gdemo in good, "the green demo path is good")
        gpath = _polyline([B(q) for q in gdemo], CG, 7)
        self.play(Create(gpath), run_time=1.0)
        self.play(FadeOut(gpath), run_time=0.4)

        # a bad path rises above it: it touches the line y = x + 1
        bdemo = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
        i0 = _first_touch(bdemo)
        check(bdemo in bad and i0 == 3, "the demo path first touches at (1, 2)")
        head = _polyline([B(q) for q in bdemo[:i0 + 1]], WHITE, 7)
        tail = VGroup(_polyline([B(q) for q in bdemo[i0:]], WHITE, 7),
                      Dot(B(bdemo[-1]), radius=0.07, color=WHITE))
        self.play(Create(head), Create(tail[0]), run_time=1.2)
        red = DashedLine(B((-0.3, 0.7)), B((3.25, 4.25)), color=CR, stroke_width=3.5,
                         dash_length=0.12)
        touch = Dot(B(bdemo[i0]), radius=0.11, color=YELLOW_B)
        self.play(Create(red), run_time=0.8)
        self.play(FadeIn(touch, scale=1.6), run_time=0.5)

        # reflect the rest of the path in that line
        ghost = tail[0].copy().set_stroke(opacity=0.28)
        self.add(ghost)
        self.add(tail[1])
        self.play(_flip(tail, B(bdemo[i0]), _DIAG), FadeIn(top), run_time=1.8)
        want = _reflect_tail(bdemo)
        check(close(tail[1].get_center()[:2], B(Ts)[:2], 1e-6), "the end lands on (2, 4)")
        s_, e_ = _ends(tail[0])
        check(close(s_, B(want[i0]), 1e-6) and close(e_, B(want[-1]), 1e-6),
              "the reflected part runs from the touch point to (2, 4)")
        dTs = Dot(B(Ts), radius=0.1, color=CF)
        lTs = tag("(2, 4)", 24, CF).next_to(dTs, UP, buff=0.12)
        link = DashedLine(B(T), B(Ts), color=GREY_B, stroke_width=2, dash_length=0.07)
        self.play(tail.animate.set_color(CF), FadeIn(dTs), FadeIn(lTs), Create(link),
                  run_time=0.8)
        self.bring_to_front(touch)
        for lab, nm in ((lTs, "(2, 4)"), (lT, "(3, 3)"), (lO, "O")):
            _clear_of_segment(lab, red.get_start(), red.get_end(), what=nm)
            _clear_of_segment(lab, diag.get_start(), diag.get_end(), what=nm)
        self.hold(0.4)

        # and back: every path to (2, 4) touches the line; reflect back
        # (dim strokes only: an open polyline given a fill shows as a polygon)
        lines1, dots1 = [head, tail[0], link], [tail[1], touch, dTs]
        self.play(*[m.animate.set_stroke(opacity=0.22) for m in lines1],
                  ghost.animate.set_stroke(opacity=0.1),
                  *[d.animate.set_opacity(0.22) for d in dots1], run_time=0.4)
        idemo = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4)]
        j0 = _first_touch(idemo)
        check(j0 == 1 and _reflect_tail(idemo) in bad,
              "the path to (2, 4) reflects back to a bad path to (3, 3)")
        ihead = _polyline([B(q) for q in idemo[:j0 + 1]], TEAL_B, 7)
        itail = VGroup(_polyline([B(q) for q in idemo[j0:]], TEAL_B, 7),
                       Dot(B(idemo[-1]), radius=0.07, color=TEAL_B))
        itouch = Dot(B(idemo[j0]), radius=0.11, color=YELLOW_B)
        self.play(Create(ihead), Create(itail[0]), FadeIn(itail[1]), run_time=1.0)
        self.play(FadeIn(itouch, scale=1.6), run_time=0.4)
        self.play(_flip(itail, B(idemo[j0]), _DIAG), run_time=1.5)
        check(close(itail[1].get_center()[:2], B(T)[:2], 1e-6), "back at (3, 3)")
        self.play(Indicate(dT, color=YELLOW_B, scale_factor=1.6), run_time=0.6)
        self.play(FadeOut(VGroup(ihead, itail, itouch)),
                  *[m.animate.set_stroke(opacity=1) for m in lines1],
                  ghost.animate.set_stroke(opacity=0.28),
                  *[d.animate.set_opacity(1) for d in dots1], run_time=0.6)
        check(all(m.get_fill_opacity() == 0 for m in (head, tail[0], ghost)),
              "the paths stay unfilled")

        # all twenty paths: the good ones, and the bad ones reflected
        cm = 0.25
        xs = [-1.9 + 1.2 * j for j in range(5)]
        ys = [2.55, 1.15, -0.25, -1.65]

        def mini(p, org, kind):
            o = to3(org)

            def P(q):
                return o + cm * np.array([q[0], q[1], 0.0])

            g = VGroup(_grid_lines(n, n + 1, cm, o, GREY_D, 1),
                       Line(P((0, 0)), P((n, n)), color=GREY_B, stroke_width=1.5),
                       Line(P((0, 1)), P((n, n + 1)), color=CR, stroke_width=1.5))
            if kind == "good":
                g.add(_polyline([P(q) for q in p], CG, 3.5),
                      Dot(P(p[-1]), radius=0.045, color=CG))
                return g, None
            i = _first_touch(p)
            tl = VGroup(_polyline([P(q) for q in p[i:]], WHITE, 3.5),
                        Dot(P(p[-1]), radius=0.045, color=WHITE))
            g.add(_polyline([P(q) for q in p[:i + 1]], WHITE, 3.5), tl,
                  Dot(P(p[i]), radius=0.05, color=YELLOW_B))
            return g, (tl, P(p[i]), [P(q) for q in _reflect_tail(p)])

        minis, flips = [], []
        for idx, p in enumerate(good + bad):
            org = np.array([xs[idx % 5], ys[idx // 5], 0.0])
            g, f = mini(p, org, "good" if idx < 5 else "bad")
            minis.append(g)
            if f:
                flips.append(f)
        gal = VGroup(*minis)
        check(_inside(gal), "gallery inside the frame")
        check(gal.get_left()[0] > lT.get_right()[0] + 0.3, "gallery clear of the demo")
        self.play(LaggedStart(*[FadeIn(m) for m in minis], lag_ratio=0.06),
                  run_time=1.8)

        xb = gal.get_right()[0] + 0.25

        def bracket(y0, y1, col):
            return VGroup(Line([xb, y0, 0], [xb, y1, 0], color=col, stroke_width=3),
                          Line([xb - 0.12, y0, 0], [xb, y0, 0], color=col, stroke_width=3),
                          Line([xb - 0.12, y1, 0], [xb, y1, 0], color=col, stroke_width=3))

        def two(a, b, col):
            return VGroup(tag(a, 26, col), tag(b, 26, col)).arrange(DOWN, buff=0.1,
                                                                    aligned_edge=LEFT)

        y_top, y_bot = ys[0] + (n + 1) * cm, ys[3]
        brA = bracket(y_bot, y_top, WHITE)
        labA = two("C(6, 3)", "= 20", WHITE)
        labA.next_to(brA, RIGHT, buff=0.2)
        self.play(Create(brA), FadeIn(labA), run_time=0.7)

        self.play(*[_flip(tl, pv, _DIAG) for tl, pv, _ in flips], run_time=2.0)
        for tl, pv, wp in flips:
            check(close(tl[1].get_center()[:2], wp[-1][:2], 1e-6),
                  "each reflected end lands on (2, 4)")
            s_, e_ = _ends(tl[0])
            check(close(e_, wp[-1], 1e-6), "the reflected part ends at (2, 4)")
            anc = {tuple(np.round(a[:2], 6)) for a in tl[0].get_anchors()}
            i = next(k for k, q in enumerate(wp) if close(q, pv, 1e-9))
            check(anc == {tuple(np.round(q[:2], 6)) for q in wp[i:]},
                  "the reflected part runs through the mirror points")
        self.play(*[tl.animate.set_color(CF) for tl, _, _ in flips], run_time=0.5)

        brG = bracket(ys[0], y_top, CG)
        brF = bracket(y_bot, ys[1] + (n + 1) * cm, CF)
        labG = two("C₃", "= 5", CG).next_to(brG, RIGHT, buff=0.2)
        labF = two("C(6, 4)", "= 15", CF).next_to(brF, RIGHT, buff=0.2)
        self.play(ReplacementTransform(brA, VGroup(brG, brF)), FadeOut(labA),
                  FadeIn(labF), run_time=0.8)
        eq = tag("C₃  =  C(6, 3) − C(6, 4)  =  20 − 15  =  5", 30, YELLOW_B)
        eq.move_to([1.75, -2.42, 0])
        self.play(FadeIn(labG), FadeIn(eq, shift=UP * 0.1), run_time=0.9)
        for m_ in (labG, labF, eq, brG, brF):
            check(_inside(m_), "labels inside the frame")
        check(eq.get_top()[1] < gal.get_bottom()[1] - 0.1, "equation clear of the gallery")
        check(eq.get_left()[0] > B((n, 0))[0] + 0.25, "equation clear of the demo")
        cap = caption("Cₙ  =  C(2n, n) − C(2n, n + 1)  =  C(2n, n) / (n + 1)", 32)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K15

class K15_RectanglesOnGrid(Board):
    """A rectangle drawn on an n × n grid has its left and right sides on
    two of the n + 1 vertical lines and its bottom and top on two of the
    n + 1 horizontal lines, and any two of each bound exactly one
    rectangle. So the rectangles are the pairs (vertical pair, horizontal
    pair): a C(n + 1, 2) × C(n + 1, 2) table. Shown for n = 3: all 36."""

    def construct(self):
        n = 3
        pairs = list(combinations(range(n + 1), 2))
        brute = {(x0, x1, y0, y1) for x0 in range(n + 1) for x1 in range(x0 + 1, n + 1)
                 for y0 in range(n + 1) for y1 in range(y0 + 1, n + 1)}
        table = {(a, b, c, d) for (c, d) in pairs for (a, b) in pairs}
        check(len(pairs) == _binom(n + 1, 2) == 6, "6 pairs of lines each way")
        check(table == brute and len(brute) == 36, "every rectangle exactly once")
        for nn in range(1, 12):
            check(sum(1 for x0 in range(nn + 1) for x1 in range(x0 + 1, nn + 1)
                      for y0 in range(nn + 1) for y1 in range(y0 + 1, nn + 1))
                  == _binom(nn + 1, 2) ** 2, "C(n + 1, 2)² rectangles")

        CV, CH, CR = ORANGE, BLUE_C, YELLOW_E
        u = 1.05
        G0 = np.array([-6.25, -1.9, 0.0])

        def B(x, y):
            return G0 + u * np.array([x, y, 0.0])

        grid = _grid_lines(n, n, u, G0, GREY_B, 2)
        ext = 0.32

        def vline(i, col=CV, w=6):
            return Line(B(i, 0) + DOWN * ext, B(i, n) + UP * ext, color=col,
                        stroke_width=w)

        def hline(j, col=CH, w=6):
            return Line(B(0, j) + LEFT * ext, B(n, j) + RIGHT * ext, color=col,
                        stroke_width=w)

        def rect(a, b, c, d):
            return VGroup(mk([B(a, c), B(b, c), B(b, d), B(a, d)], CR, 0.55,
                             stroke_width=0),
                          vline(a).put_start_and_end_on(B(a, c), B(a, d)),
                          vline(b).put_start_and_end_on(B(b, c), B(b, d)),
                          hline(c).put_start_and_end_on(B(a, c), B(b, c)),
                          hline(d).put_start_and_end_on(B(a, d), B(b, d)))

        self.play(Create(grid), run_time=1.0)
        allv = VGroup(*[vline(i) for i in range(n + 1)])
        allh = VGroup(*[hline(j) for j in range(n + 1)])
        self.play(LaggedStart(*[Create(l) for l in allv], lag_ratio=0.2), run_time=0.8)
        self.play(LaggedStart(*[Create(l) for l in allh], lag_ratio=0.2),
                  allv.animate.set_stroke(opacity=0.25), run_time=0.8)
        self.play(FadeOut(allv), FadeOut(allh), run_time=0.4)

        demos = [((1, 3), (0, 2)), ((0, 1), (1, 3))]
        shown = []
        for (a, b), (c, d) in demos:
            vv = VGroup(vline(a), vline(b))
            hh = VGroup(hline(c), hline(d))
            self.play(Create(vv), run_time=0.6)
            self.play(Create(hh), run_time=0.6)
            r = rect(a, b, c, d)
            check(close(r[0].get_corner(DL), B(a, c), 1e-9)
                  and close(r[0].get_corner(UR), B(b, d), 1e-9),
                  "the four lines bound the rectangle")
            self.play(FadeIn(r[0]), FadeOut(vv), FadeOut(hh), FadeIn(r[1:]),
                      run_time=0.7)
            shown.append(r)
            self.hold(0.3)

        # the table: column = pair of vertical lines, row = pair of horizontal
        s, pitch = 0.6, 0.78
        xh, yh = 0.62, 3.28                       # header column x, header row y
        cxs = [xh + pitch * (k + 1) for k in range(6)]
        rys = [yh - pitch * (k + 1) for k in range(6)]

        def frame_at(cx, cy):
            o = np.array([cx - s / 2, cy - s / 2, 0.0])
            return VGroup(_grid_lines(n, n, s / n, o, GREY_D, 1)), o

        def m_v(cx, cy, a, b):
            g, o = frame_at(cx, cy)
            for i in (a, b):
                g.add(Line(o + [i * s / n, 0, 0], o + [i * s / n, s, 0], color=CV,
                           stroke_width=4))
            return g

        def m_h(cx, cy, c, d):
            g, o = frame_at(cx, cy)
            for j in (c, d):
                g.add(Line(o + [0, j * s / n, 0], o + [s, j * s / n, 0], color=CH,
                           stroke_width=4))
            return g

        def m_r(cx, cy, a, b, c, d):
            g, o = frame_at(cx, cy)
            q = s / n

            def P(x, y):
                return o + np.array([x * q, y * q, 0.0])

            g.add(mk([P(a, c), P(b, c), P(b, d), P(a, d)], CR, 0.75, stroke_width=0),
                  Line(P(a, c), P(a, d), color=CV, stroke_width=3),
                  Line(P(b, c), P(b, d), color=CV, stroke_width=3),
                  Line(P(a, c), P(b, c), color=CH, stroke_width=3),
                  Line(P(a, d), P(b, d), color=CH, stroke_width=3))
            return g

        colh = [m_v(cxs[k], yh, *pairs[k]) for k in range(6)]
        rowh = [m_h(xh, rys[k], *pairs[k]) for k in range(6)]
        cells = {(r, c): m_r(cxs[c], rys[r], *pairs[c], *pairs[r])
                 for r in range(6) for c in range(6)}
        labV = tag("C(4, 2) = 6", 26, CV)
        labV.move_to([xh - s / 2 - 0.25 - labV.width / 2, yh, 0])
        labH = tag("C(4, 2) = 6", 26, CH)
        labH.move_to([xh - s / 2 - 0.25 - labH.width / 2, (rys[2] + rys[3]) / 2, 0])
        for m_ in (labV, labH, VGroup(*colh), VGroup(*rowh), VGroup(*cells.values())):
            check(_inside(m_), "table inside the frame")
        check(labH.get_left()[0] > G0[0] + n * u + ext + 0.3, "labels clear of the grid")
        self.play(LaggedStart(*[FadeIn(m) for m in colh], lag_ratio=0.15),
                  FadeIn(labV), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(m) for m in rowh], lag_ratio=0.15),
                  FadeIn(labH), run_time=1.2)
        order = [cells[(r, c)] for r in range(6) for c in range(6)]
        self.play(LaggedStart(*[FadeIn(m) for m in order], lag_ratio=0.06),
                  run_time=3.0)

        # the two demonstration rectangles in the table
        marks = VGroup()
        for (a, b), (c, d) in demos:
            ci, ri = pairs.index((a, b)), pairs.index((c, d))
            cell = cells[(ri, ci)]
            q = s / n
            o = np.array([cxs[ci] - s / 2, rys[ri] - s / 2, 0.0])
            check(close(cell[1].get_corner(DL), o + q * np.array([a, c, 0.0]), 1e-9)
                  and close(cell[1].get_corner(UR), o + q * np.array([b, d, 0.0]), 1e-9),
                  "the table cell shows the rectangle of its two pairs")
            for m_ in (cell, colh[ci], rowh[ri]):
                marks.add(SurroundingRectangle(m_, buff=0.05, color=YELLOW_B,
                                               stroke_width=3))
        self.play(Create(marks), *[Indicate(r[0], color=YELLOW_B, scale_factor=1.0)
                                   for r in shown], run_time=1.0)
        tot = tag("6 × 6  =  36", 32, YELLOW_B).move_to(
            [(cxs[0] + cxs[-1]) / 2, rys[-1] - s / 2 - 0.5, 0])
        check(_inside(tot), "total inside the frame")
        self.play(FadeIn(tot, shift=UP * 0.1), run_time=0.8)
        cap = caption("2 of n + 1 vertical lines,  2 of n + 1 horizontal lines   ⟹   "
                      "C(n + 1, 2)²", 30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K12

def _tromino_split(x0, y0, size, hole, level=0, out=None):
    """The induction, run to the end: quarter the defective board and lay
    one L-tromino on the three centre cells outside the defective
    quarter; recurse into the four quarters, each now missing one cell.
    Returns (level, bx, by, missing cell of the 2 × 2 block, the 3 cells)."""
    if out is None:
        out = []
    h = size // 2
    if size == 2:
        cells = [(x0 + i, y0 + j) for j in range(2) for i in range(2)
                 if (x0 + i, y0 + j) != hole]
        out.append((level, x0, y0, hole, cells))
        return out
    quads = [(x0, y0), (x0 + h, y0), (x0, y0 + h), (x0 + h, y0 + h)]
    centre = [(x0 + h - 1, y0 + h - 1), (x0 + h, y0 + h - 1),
              (x0 + h - 1, y0 + h), (x0 + h, y0 + h)]
    qi = next(k for k, (qx, qy) in enumerate(quads)
              if qx <= hole[0] < qx + h and qy <= hole[1] < qy + h)
    out.append((level, x0 + h - 1, y0 + h - 1, centre[qi],
                [centre[k] for k in range(4) if k != qi]))
    for k in range(4):
        _tromino_split(*quads[k], h, hole if k == qi else centre[k], level + 1, out)
    return out


def _L_poly(bx, by, m):
    """The L-tromino filling the 2 × 2 block at (bx, by) less cell m,
    as a counter-clockwise hexagon in cell units."""
    mx, my = m[0] - bx, m[1] - by
    if (mx, my) == (0, 0):
        pts = [(1, 0), (2, 0), (2, 2), (0, 2), (0, 1), (1, 1)]
    elif (mx, my) == (1, 0):
        pts = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (0, 2)]
    elif (mx, my) == (1, 1):
        pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    else:
        pts = [(0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1)]
    return [(bx + x, by + y) for x, y in pts]


def _pip(p, poly):
    """Point strictly inside a simple polygon (ray casting)."""
    x, y = p
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


class K12_TrominoBoard(Board):
    """Induction on n: a 2ⁿ × 2ⁿ board with any one cell removed can be
    tiled by L-trominoes. Cut it into four 2ⁿ⁻¹ × 2ⁿ⁻¹ quarters; the
    missing cell lies in one of them. One L-tromino on the three centre
    cells of the other quarters leaves four boards, each with exactly one
    cell missing — smaller instances of the same problem. A 2 × 2 board
    less a cell is one tromino. Run completely for 8 × 8: 1 + 4 + 16 = 21
    trominoes cover the 63 cells, so 3 divides 8² − 1 = 4³ − 1."""

    def construct(self):
        N = 8
        hole = (2, 5)
        tiles = _tromino_split(0, 0, N, hole)
        lev = {k: [t for t in tiles if t[0] == k] for k in range(3)}
        check([len(lev[k]) for k in range(3)] == [1, 4, 16] and len(tiles) == 21,
              "1 + 4 + 16 = 21 trominoes")
        covered = [c for t in tiles for c in t[4]]
        check(len(covered) == len(set(covered)) == N * N - 1
              and set(covered) | {hole} == {(i, j) for i in range(N) for j in range(N)},
              "the trominoes cover every cell but the hole, once each")
        for lv, bx, by, m, cells in tiles:
            blk = {(bx + i, by + j) for i in range(2) for j in range(2)}
            check(m in blk and set(cells) == blk - {m}, "an L: a 2 × 2 block less one")
            poly = _L_poly(bx, by, m)
            check(abs(area(poly) - 3) < 1e-12, "the L has area 3")
            check(all(_pip((x + 0.5, y + 0.5), poly) == ((x, y) in cells)
                       for x, y in blk), "the L polygon covers exactly its three cells")
        # each quarter, at each stage, has exactly one cell missing
        for size, lv in ((4, 0), (2, 1)):
            done = {c for t in tiles if t[0] <= lv for c in t[4]} | {hole}
            for qx in range(0, N, size):
                for qy in range(0, N, size):
                    k = sum(1 for (x, y) in done
                            if qx <= x < qx + size and qy <= y < qy + size)
                    check(k == 1, f"each {size} × {size} quarter misses one cell")
        for m_ in range(1, 12):
            check((4 ** m_ - 1) % 3 == 0, "3 | 4ⁿ − 1")

        c = 0.58
        G0 = np.array([-6.3, -2.8, 0.0])

        def B(x, y):
            return G0 + c * np.array([x, y, 0.0])

        def Cc(x, y):
            return B(x + 0.5, y + 0.5)

        board = VGroup(*[Square(side_length=c, fill_color="#2B303B", fill_opacity=1,
                                stroke_color=GREY_C, stroke_width=1.2).move_to(Cc(x, y))
                         for x in range(N) for y in range(N) if (x, y) != hole])
        rim = Square(side_length=N * c, stroke_color=WHITE, stroke_width=3
                     ).move_to(B(N / 2, N / 2))
        hp = Cc(*hole)
        hcell = VGroup(Square(side_length=c, fill_color=BLACK, fill_opacity=1,
                              stroke_color=RED_C, stroke_width=3).move_to(hp),
                       Line(hp + 0.17 * (UL), hp + 0.17 * DR, color=RED_C, stroke_width=4),
                       Line(hp + 0.17 * UR, hp + 0.17 * DL, color=RED_C, stroke_width=4))
        lsz = tag("8 × 8", 30).next_to(rim, UP, buff=0.22)
        self.play(FadeIn(board), Create(rim), FadeIn(hcell), FadeIn(lsz), run_time=1.0)

        cols = {0: GOLD_D, 1: TEAL_D, 2: BLUE_D}

        def tromino(t, op=0.95):
            lv, bx, by, m, _ = t
            return mk([B(*q) for q in _L_poly(bx, by, m)], cols[lv], op,
                      stroke_color=WHITE, stroke_width=3)

        def cross(x0, y0, size, w):
            h = size / 2
            return VGroup(Line(B(x0 + h, y0), B(x0 + h, y0 + size), color=WHITE,
                               stroke_width=w),
                          Line(B(x0, y0 + h), B(x0 + size, y0 + h), color=WHITE,
                               stroke_width=w))

        def centre_box(bx, by):
            return DashedVMobject(Square(side_length=2 * c - 0.08, stroke_color=YELLOW_B,
                                         stroke_width=4), num_dashes=16
                                  ).move_to(B(bx + 1, by + 1))

        # the panel: one row per stage
        px = 2.6
        rows_y = [2.6, 1.55, 0.5]

        def icon(col):
            s_ = 0.22
            g = VGroup(*[Square(side_length=s_, fill_color=col, fill_opacity=0.95,
                                stroke_color=WHITE, stroke_width=1.5).move_to(
                [i * s_, j * s_, 0]) for i, j in ((0, 0), (1, 0), (0, 1))])
            return g

        def row(size_txt, k, col, y):
            g = VGroup(tag(size_txt, 30, GREY_A), icon(col), tag(f"× {k}", 30, col))
            g.arrange(RIGHT, buff=0.3)
            g.move_to([px, y, 0])
            return g

        rows = [row("8 × 8 :", 1, cols[0], rows_y[0]),
                row("4 × 4 :", 4, cols[1], rows_y[1]),
                row("2 × 2 :", 16, cols[2], rows_y[2])]
        for r_ in rows[1:]:
            r_.align_to(rows[0], LEFT)

        # stage 1: quarter the board; one tromino at the centre
        cr0 = cross(0, 0, N, 5)
        self.play(Create(cr0), run_time=0.6)
        t0 = lev[0][0]
        box0 = centre_box(t0[1], t0[2])
        self.play(Create(box0), run_time=0.6)
        tr0 = tromino(t0)
        self.play(FadeIn(tr0), FadeOut(box0), FadeIn(rows[0]), run_time=0.8)
        # four smaller boards, each missing one cell
        rings = VGroup(*[Circle(radius=0.2, stroke_color=YELLOW_B, stroke_width=4
                                ).move_to(Cc(*q)) for q in t0[4] + [hole]])
        quads = VGroup(*[Square(side_length=4 * c - 0.1, stroke_color=YELLOW_B,
                                stroke_width=4).move_to(B(qx + 2, qy + 2))
                         for qx in (0, 4) for qy in (0, 4)])
        self.play(Create(quads), Create(rings), run_time=0.8)
        self.play(FadeOut(quads), FadeOut(rings), run_time=0.5)

        # stage 2: the same in each quarter
        cr1 = VGroup(*[cross(qx, qy, 4, 3.5) for qx in (0, 4) for qy in (0, 4)])
        self.play(Create(cr1), run_time=0.6)
        boxes1 = VGroup(*[centre_box(t[1], t[2]) for t in lev[1]])
        self.play(Create(boxes1), run_time=0.6)
        tr1 = VGroup(*[tromino(t) for t in lev[1]])
        self.play(FadeIn(tr1), FadeOut(boxes1), FadeIn(rows[1]), run_time=0.8)
        rings1 = VGroup(*[Circle(radius=0.2, stroke_color=YELLOW_B, stroke_width=3.5
                                 ).move_to(Cc(*q))
                          for q in [q for t in tiles if t[0] <= 1 for q in t[4]] + [hole]])
        quads1 = VGroup(*[Square(side_length=2 * c - 0.1, stroke_color=YELLOW_B,
                                 stroke_width=3).move_to(B(qx + 1, qy + 1))
                          for qx in range(0, N, 2) for qy in range(0, N, 2)])
        check(len(rings1) == 16, "sixteen 2 × 2 boards, one cell missing in each")
        self.play(Create(quads1), Create(rings1), run_time=0.8)
        self.play(FadeOut(quads1), FadeOut(rings1), run_time=0.5)

        # stage 3: every 2 × 2 board less a cell is one tromino
        tr2 = VGroup(*[tromino(t) for t in lev[2]])
        self.play(LaggedStart(*[FadeIn(t) for t in tr2], lag_ratio=0.12),
                  FadeIn(rows[2]), run_time=2.0)
        self.bring_to_front(hcell)

        s1 = tag("1 + 4 + 16  =  21", 32, YELLOW_B)
        s2 = tag("3 · 21  =  63  =  8² − 1", 32, YELLOW_B)
        s1.move_to([px, -0.75, 0]).align_to(rows[0], LEFT)
        s2.move_to([px, -1.6, 0]).align_to(rows[0], LEFT)
        self.play(FadeIn(s1, shift=UP * 0.1), run_time=0.7)
        self.play(FadeIn(s2, shift=UP * 0.1), run_time=0.7)
        for m_ in [*rows, s1, s2, rim, lsz]:
            check(_inside(m_), "inside the frame")
        check(min(r_.get_left()[0] for r_ in rows) > rim.get_right()[0] + 0.5,
              "panel clear of the board")
        cap = caption("2ⁿ × 2ⁿ less one cell  =  (4ⁿ − 1) / 3  L-trominoes   ⟹   "
                      "3  |  4ⁿ − 1", 30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ================================================== K10 (and K16): regions

def _cut(poly, a, b, eps=1e-10):
    """Split a convex polygon (list of 2D points) by the line ab. Returns
    (left part, right part); a part is None if the line misses it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    s = [d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0]) for p in poly]
    L, R = [], []
    m = len(poly)
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        sp, sq = s[i], s[(i + 1) % m]
        if sp >= -eps:
            L.append(p)
        if sp <= eps:
            R.append(p)
        if (sp > eps and sq < -eps) or (sp < -eps and sq > eps):
            x = p + (sp / (sp - sq)) * (q - p)
            L.append(x)
            R.append(x)

    def ok(P):
        return len(P) >= 3 and abs(area(P)) > 1e-9

    return (L if ok(L) else None), (R if ok(R) else None)


def _arrange(region, cuts):
    """Faces of a convex region cut by lines (each given by two points),
    added one at a time. Every face stays convex. Returns the faces and,
    for each line, how many faces it split (= the regions it added)."""
    faces, added = [region], []
    for a, b in cuts:
        nxt, k = [], 0
        for f in faces:
            l, r = _cut(f, a, b)
            if l is not None and r is not None:
                nxt += [l, r]
                k += 1
            else:
                nxt.append(f)
        faces = nxt
        added.append(k)
    return faces, added


def _parity(face, cuts):
    """2-colouring of an arrangement: the number of lines with the face
    on their left, mod 2 (neighbours across a line always differ)."""
    c = np.mean(face, axis=0)
    k = 0
    for a, b in cuts:
        d = np.asarray(b) - np.asarray(a)
        if d[0] * (c[1] - a[1]) - d[1] * (c[0] - a[0]) > 0:
            k += 1
    return k % 2


def _xing(p1, p2, q1, q2):
    """Crossing point of segments p1p2 and q1q2 strictly inside both."""
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    w = q1 - p1
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    s = (w[0] * d1[1] - w[1] * d1[0]) / den
    if 1e-9 < t < 1 - 1e-9 and 1e-9 < s < 1 - 1e-9:
        return p1 + t * d1
    return None


def _seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    d = b - a
    t = np.clip(np.dot(p - a, d) / np.dot(d, d), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * d)))


def _disk(angles, N=720):
    """The unit disk as a polygon whose vertices include the given points."""
    ts = sorted({round(TAU * i / N, 10) for i in range(N)}
                | {round(a % TAU, 10) for a in angles})
    out = []
    for t in ts:
        if not out or abs(t - out[-1][0]) > 1e-6:
            out.append((t, np.array([np.cos(t), np.sin(t)])))
    return [p for _, p in out]


class K10_CircleRegions(Board):
    """Caution. Put n points on a circle and join every pair: the disc is
    cut into 1, 2, 4, 8, 16 regions for n = 1 … 5, and the guess 32 for
    n = 6 is wrong — there are 31 (points in general position: no three
    chords through one point). Why: draw the chords one at a time; a new
    chord is cut by the c chords it crosses into c + 1 pieces, and each
    piece splits one region in two, so it adds 1 + c regions. Hence
    R = 1 + (chords) + (crossings), and a crossing is fixed by the four
    ends of its two chords: Rₙ = 1 + C(n, 2) + C(n, 4) — not 2ⁿ⁻¹."""

    def construct(self):
        degs = {1: [90], 2: [35, 215], 3: [90, 210, 330], 4: [60, 150, 240, 325],
                5: [18, 90, 162, 234, 306], 6: [33, 87, 156, 200, 268, 328]}
        data = {}
        for n, dg in degs.items():
            ang = sorted(np.radians(dg))
            P = [np.array([np.cos(t), np.sin(t)]) for t in ang]
            ch = list(combinations(range(n), 2))
            faces, added = _arrange(_disk(ang), [(P[i], P[j]) for i, j in ch])
            X = []                                # (point, chord, chord)
            for u, (i, j) in enumerate(ch):
                xs = [(x, (k, l), (i, j)) for (k, l) in ch[:u]
                      if (x := _xing(P[i], P[j], P[k], P[l])) is not None]
                check(added[u] == 1 + len(xs), "a chord adds 1 + its crossings")
                X += xs
            check(len(X) == _binom(n, 4), "crossings = C(n, 4)")
            check(len(faces) == 1 + _binom(n, 2) + _binom(n, 4) == 1 + len(ch) + len(X),
                  f"R{n} = 1 + C(n,2) + C(n,4)")
            data[n] = (ang, P, ch, faces, X)
        check([len(data[n][3]) for n in range(1, 7)] == [1, 2, 4, 8, 16, 31],
              "1, 2, 4, 8, 16, then 31")
        # general position for n = 6: crossings apart, none on a third chord
        P6, ch6, X6 = data[6][1], data[6][2], data[6][4]
        check(min(np.linalg.norm(a[0] - b[0]) for a, b in combinations(X6, 2)) > 0.15,
              "the 15 crossings are distinct points")
        for x, c1, c2 in X6:
            check(min(_seg_dist(x, P6[k], P6[l]) for k, l in ch6 if (k, l) not in (c1, c2))
                  > 0.12, "no third chord passes near a crossing")
            check(len(set(c1) | set(c2)) == 4, "a crossing has four distinct ends")
        check(len({tuple(sorted(set(c1) | set(c2))) for _, c1, c2 in X6}) == 15,
              "different crossings, different sets of four points")
        hexa = [np.array([np.cos(t), np.sin(t)]) for t in np.radians([0, 60, 120, 180,
                                                                       240, 300])]
        f_reg, _ = _arrange(_disk(np.radians([0, 60, 120, 180, 240, 300])),
                            [(hexa[i], hexa[j]) for i, j in combinations(range(6), 2)])
        check(len(f_reg) == 30, "(a regular hexagon gives only 30: three chords meet)")

        CA, CB = BLUE_D, ORANGE                   # the two region tints

        def figure(n, ctr, r, cw, dr, op=0.6):
            _, P, ch, faces, _ = data[n]
            cuts = [(P[i], P[j]) for i, j in ch]

            def S(p):
                return to3(ctr) + r * np.array([p[0], p[1], 0.0])

            tint = VGroup(*[Polygon(*[S(p) for p in f], stroke_width=0,
                                    fill_color=(CA, CB)[_parity(f, cuts)],
                                    fill_opacity=op) for f in faces])
            lines = VGroup(*[Line(S(P[i]), S(P[j]), color=WHITE, stroke_width=cw)
                             for i, j in ch])
            circ = Circle(radius=r, color=WHITE, stroke_width=cw + 0.5).move_to(to3(ctr))
            dots = VGroup(*[Dot(S(p), radius=dr, color=WHITE) for p in P])
            return VGroup(tint, circ, lines, dots), S

        # ---------------- the pattern: 1, 2, 4, 8, 16, …?
        rs, ytop = 0.52, 2.98
        slot = [np.array([-5.5 + 2.2 * i, ytop, 0.0]) for i in range(6)]
        smalls, counts = [], []
        for n in range(1, 6):
            g, _ = figure(n, slot[n - 1], rs, 1.6, 0.045)
            cnt = tag(str(len(data[n][3])), 28).move_to(slot[n - 1] + DOWN * (rs + 0.34))
            smalls.append(g)
            counts.append(cnt)
            self.play(FadeIn(g), FadeIn(cnt), run_time=0.75)
        ring6 = DashedVMobject(Circle(radius=rs, color=GREY_B, stroke_width=2),
                               num_dashes=24).move_to(slot[5])
        guess = tag("32 ?", 28, GREY_B).move_to(slot[5] + DOWN * (rs + 0.34))
        self.play(Create(ring6), FadeIn(guess), run_time=0.7)

        # ---------------- n = 6, one chord at a time
        Rb, Cb = 2.0, np.array([-3.4, -0.8, 0.0])
        ang6 = data[6][0]

        def S(p):
            return Cb + Rb * np.array([p[0], p[1], 0.0])

        circ = Circle(radius=Rb, color=WHITE, stroke_width=3).move_to(Cb)
        pdots = VGroup(*[Dot(S(p), radius=0.075, color=WHITE) for p in P6])

        def chord_icon():
            return VGroup(Line(LEFT * 0.34, RIGHT * 0.34, color=WHITE, stroke_width=3),
                          Dot(LEFT * 0.34, radius=0.06), Dot(RIGHT * 0.34, radius=0.06))

        def cross_icon():
            return VGroup(Line([-0.3, -0.22, 0], [0.3, 0.22, 0], color=WHITE,
                               stroke_width=3),
                          Line([-0.3, 0.22, 0], [0.3, -0.22, 0], color=WHITE,
                               stroke_width=3),
                          Dot(ORIGIN, radius=0.07, color=YELLOW_B))

        px0, yk, yc, yr = 0.55, 0.75, -0.15, -1.25
        ick = chord_icon().move_to([px0, yk, 0])
        icx = cross_icon().move_to([px0, yc, 0])

        def num(v, y, col=WHITE):
            t = tag(str(v), 32, col)
            return t.move_to([px0 + 0.62 + t.width / 2, y, 0])

        kt, ct = num(0, yk), num(0, yc)
        # R = 1 + k + c on fixed slots: the constant parts stay put, the three
        # numbers are cross-faded in place (no glyph morphing)
        widest = VGroup(*[tag(s_, 32) for s_ in
                          ("R", "=", "1", "+", "15", "+", "15", "=", "31")]
                        ).arrange(RIGHT, buff=0.24)
        widest.move_to([px0 - 0.3 + widest.width / 2, yr, 0])
        rfixed = VGroup(*[tag(s_, 32).move_to(widest[q])
                          for q, s_ in ((0, "R"), (1, "="), (2, "1"), (3, "+"),
                                        (5, "+"), (7, "="))])

        def rnum(v, q, col=WHITE):
            t = tag(str(v), 32, col)
            return t.move_to([widest[q].get_center()[0], widest[q].get_center()[1], 0])

        rk, rc, rt_ = rnum(0, 4), rnum(0, 6), rnum(1, 8, YELLOW_B)
        self.play(Create(circ), FadeIn(pdots), FadeIn(ick), FadeIn(icx), FadeIn(kt),
                  FadeIn(ct), FadeIn(rfixed), FadeIn(rk), FadeIn(rc), FadeIn(rt_),
                  run_time=1.0)

        drawn, xdots, prev, kk, cc = [], VGroup(), None, 0, 0
        chords_drawn = VGroup()
        inc = None
        for (i, j) in ch6:
            a, b = P6[i], P6[j]
            xs = [x for (k, l) in drawn if (x := _xing(a, b, P6[k], P6[l])) is not None]
            xs.sort(key=lambda x: float(np.dot(x - a, b - a)))
            nodes = [a] + xs + [b]
            pieces = VGroup(*[Line(S(nodes[t]), S(nodes[t + 1]),
                                   color=(YELLOW_B, RED_C)[t % 2], stroke_width=8)
                              for t in range(len(nodes) - 1)])
            check(len(pieces) == 1 + len(xs), "c crossings cut the chord in c + 1")
            nd = VGroup(*[Dot(S(x), radius=0.06, color=WHITE) for x in xs])
            kk, cc = kk + 1, cc + len(xs)
            nk, nc = num(kk, yk), num(cc, yc)
            nrk, nrc, nrt = rnum(kk, 4), rnum(cc, 6), rnum(1 + kk + cc, 8, YELLOW_B)
            ninc = tag(f"+{len(pieces)}", 28, YELLOW_B).next_to(widest, RIGHT, buff=0.45)
            swaps = [FadeOut(kt), FadeIn(nk), FadeOut(ct), FadeIn(nc), FadeOut(rk),
                     FadeIn(nrk), FadeOut(rc), FadeIn(nrc), FadeOut(rt_), FadeIn(nrt),
                     FadeIn(ninc)]
            if inc is not None:
                swaps.append(FadeOut(inc))
            anims = [LaggedStart(*[Create(pc) for pc in pieces], lag_ratio=0.5),
                     FadeIn(nd)] + swaps
            if prev is not None:
                anims.append(prev.animate.set_stroke(color=WHITE, width=2.5))
            self.play(*anims, run_time=0.6)
            self.bring_to_front(xdots, nd, pdots)
            kt, ct, rk, rc, rt_, inc = nk, nc, nrk, nrc, nrt, ninc
            drawn.append((i, j))
            xdots.add(*nd)
            chords_drawn.add(pieces)
            prev = pieces
        rl = VGroup(rfixed, rk, rc, rt_)
        self.play(prev.animate.set_stroke(color=WHITE, width=2.5), FadeOut(inc),
                  run_time=0.3)
        check(kk == 15 and cc == 15 and len(data[6][3]) == 1 + kk + cc,
              "R = 1 + 15 + 15 = 31")

        # the 31 regions
        cuts6 = [(P6[i], P6[j]) for i, j in ch6]
        tint = VGroup(*[Polygon(*[S(p) for p in f], stroke_width=0,
                                fill_color=(CA, CB)[_parity(f, cuts6)], fill_opacity=0.6)
                        for f in data[6][3]])
        self.add(tint)
        self.bring_to_back(tint)
        tint.set_opacity(0)
        self.play(tint.animate.set_fill(opacity=0.6), Indicate(rt_, color=YELLOW_B),
                  run_time=1.0)

        # a crossing is four points: the two chords are the diagonals
        for pair in (((0, 2), (1, 4)), ((0, 4), (3, 5))):
            x, (i, j), (k, l) = next(e for e in X6 if (e[1], e[2]) == pair)
            four = sorted({i, j, k, l}, key=lambda q: ang6[q])
            check(len(four) == 4, "four distinct ends")
            quad = Polygon(*[S(P6[q]) for q in four], color=YELLOW_B, stroke_width=4)
            diag = VGroup(Line(S(P6[i]), S(P6[j]), color=YELLOW_B, stroke_width=7),
                          Line(S(P6[k]), S(P6[l]), color=YELLOW_B, stroke_width=7))
            big = VGroup(*[Dot(S(P6[q]), radius=0.12, color=YELLOW_B) for q in four],
                         Dot(S(x), radius=0.11, color=RED_C))
            self.play(Create(quad), FadeIn(diag), FadeIn(big), run_time=0.7)
            self.play(FadeOut(VGroup(quad, diag, big)), run_time=0.45)
        lk = tag("=  C(6, 2)", 32).next_to(kt, RIGHT, buff=0.35)
        lc = tag("=  C(6, 4)", 32).next_to(ct, RIGHT, buff=0.35)
        self.play(FadeIn(lk), FadeIn(lc), run_time=0.7)

        # back to the pattern: 31, not 32
        bigfig = VGroup(tint, circ, chords_drawn, xdots, pdots)
        small6 = bigfig.copy().scale(rs / Rb).move_to(slot[5])
        hit = tag("31", 30, RED_B)
        self.play(TransformFromCopy(bigfig, small6), FadeOut(ring6),
                  guess.animate.shift(LEFT * 0.42), run_time=1.1)
        strike = Line(guess.get_left() + LEFT * 0.06, guess.get_right() + RIGHT * 0.06,
                      color=RED_C, stroke_width=4)
        hit.next_to(guess, RIGHT, buff=0.22)
        self.play(Create(strike), FadeIn(hit, scale=1.3), run_time=0.7)

        panel = [ick, icx, kt, ct, rl, lk, lc]
        for m_ in panel + smalls + counts + [guess, hit, small6, bigfig]:
            check(_inside(m_), "inside the frame")
        for t_ in (kt, ct, lk, lc):
            check(_apart(t_, ick, 0.1) and _apart(t_, icx, 0.1), "numbers clear of icons")
        check(VGroup(*panel).get_left()[0] > circ.get_right()[0] + 0.5,
              "panel clear of the circle")
        check(circ.get_top()[1] < min(c_.get_bottom()[1] for c_ in counts) - 0.3,
              "big circle clear of the pattern row")
        cap = caption("Rₙ  =  1 + C(n, 2) + C(n, 4)  =  1, 2, 4, 8, 16, 31, 57, 99, …",
                      30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K4

class K4_SumOfSquaresRow(Board):
    """Every east/north path from O to T = (n, n) is on the anti-diagonal
    x + y = n after exactly n steps, at one point (k, n − k). It gets there
    in C(n, k) ways and goes on to T in C(n, n − k) = C(n, k) ways, any way
    in with any way out: C(n, k)² paths through that point. Adding over the
    n + 1 points counts every path once: ∑ C(n, k)² = C(2n, n).
    Shown for n = 3: 1 + 9 + 9 + 1 = 20, every path drawn."""

    def construct(self):
        n = 3
        allp = _lattice_paths(n, n)
        ins = {k: _lattice_paths(k, n - k) for k in range(n + 1)}
        outs = {k: [[(x + k, y + n - k) for x, y in p] for p in _lattice_paths(n - k, k)]
                for k in range(n + 1)}
        check(len(allp) == _binom(2 * n, n) == 20, "C(6, 3) = 20 paths")
        check(all(sum(p[n]) == n for p in allp), "every path is on x + y = n after n steps")
        glued = sorted(tuple(a + b[1:]) for k in range(n + 1) for a in ins[k] for b in outs[k])
        check(glued == sorted(map(tuple, allp)), "in-path + out-path: every path once")
        for k in range(n + 1):
            check(len(ins[k]) == len(outs[k]) == _binom(n, k), "C(n, k) in, C(n, k) out")
        for nn in range(1, 14):
            check(sum(_binom(nn, k) ** 2 for k in range(nn + 1)) == _binom(2 * nn, nn),
                  "∑ C(n,k)² = C(2n, n)")

        CI, CO = BLUE_C, ORANGE
        kc = [GREEN_C, YELLOW_E, PURPLE_B, TEAL_C]
        cb = 0.85
        O = np.array([-6.0, 0.62, 0.0])

        def B(p):
            return O + cb * np.array([p[0], p[1], 0.0])

        grid = _grid_lines(n, n, cb, O, GREY_B, 2)
        anti = DashedLine(B((-0.35, n + 0.35)), B((n + 0.35, -0.35)), color=GREY_A,
                          stroke_width=2.5, dash_length=0.1)
        dO = Dot(B((0, 0)), radius=0.07)
        dT = Dot(B((n, n)), radius=0.09, color=WHITE)
        lO = tag("O", 24).next_to(dO, DL, buff=0.05)
        lT = tag("T", 24).next_to(dT, UR, buff=0.05)
        pts = VGroup(*[Dot(B((k, n - k)), radius=0.1, color=kc[k]) for k in range(n + 1)])
        self.play(Create(grid), FadeIn(dO), FadeIn(dT), FadeIn(lO), FadeIn(lT),
                  run_time=1.0)
        self.play(Create(anti), FadeIn(pts), run_time=0.8)

        # one path: n steps in, then n steps out
        sp = allp[7]
        k_ = sp[n][0]
        p_in = _polyline([B(q) for q in sp[:n + 1]], CI, 7)
        p_out = _polyline([B(q) for q in sp[n:]], CO, 7)
        self.play(Create(p_in), run_time=0.7)
        self.play(Indicate(pts[k_], color=kc[k_], scale_factor=1.8), run_time=0.5)
        self.play(Create(p_out), run_time=0.7)
        self.play(FadeOut(p_in), FadeOut(p_out), run_time=0.4)

        def lab_in(k):
            return tag(str(_binom(n, k)), 26, CI).move_to(B((k - 0.36, n - k - 0.36)))

        def lab_out(k):
            return tag(str(_binom(n, k)), 26, CO).move_to(B((k + 0.36, n - k + 0.36)))

        # the gallery: one table per crossing point, rows = ways in, cols = ways out
        cm, pitch = 0.24, 0.92
        gw = {k: (len(ins[k]) - 1) * pitch + n * cm for k in range(n + 1)}
        gap = 0.95
        eq_w = tag("=  20", 34).width
        total_w = sum(gw.values()) + gap * n + 0.45 + eq_w
        gx, x = {}, -total_w / 2
        for k in range(n + 1):
            gx[k] = x
            x += gw[k] + gap
        gtop = -0.42

        def mini(path, org, k):
            o = to3(org)

            def P(q):
                return o + cm * np.array([q[0], q[1], 0.0])

            return VGroup(_grid_lines(n, n, cm, o, GREY_D, 1),
                          Line(P((0, n)), P((n, 0)), color=GREY_C, stroke_width=1.2),
                          _polyline([P(q) for q in path[:n + 1]], CI, 3.5),
                          _polyline([P(q) for q in path[n:]], CO, 3.5),
                          Dot(P(path[n]), radius=0.05, color=kc[k]))

        groups, heads = {}, {}
        rmax = len(ins[1])
        for k in range(n + 1):
            g = VGroup()
            m = len(ins[k])
            r0 = (rmax - m) / 2                   # centre the small tables
            for r, a in enumerate(ins[k]):
                for c_, b in enumerate(outs[k]):
                    org = np.array([gx[k] + c_ * pitch,
                                    gtop - n * cm - (r + r0) * pitch, 0.0])
                    g.add(mini(a + b[1:], org, k))
            groups[k] = g
            heads[k] = tag(f"{m} · {m}", 28, kc[k]).move_to(
                [gx[k] + gw[k] / 2, g.get_top()[1] + 0.3, 0])

        # the point (1, 2): three ways in, three ways out (drawn side by side,
        # shared steps as parallel strands), any way in with any way out
        def strand_pts(paths):
            m = len(paths)
            return [[B(q) + (t - (m - 1) / 2) * 0.09 * np.array([1.0, -1.0, 0.0])
                     for q in path] for t, path in enumerate(paths)]

        def strands(paths, col):
            return VGroup(*[_polyline(sp_, col, 5) for sp_ in strand_pts(paths)])

        k = 1
        lin, lout = lab_in(k), lab_out(k)
        s_in, s_out = strands(ins[k], CI), strands(outs[k], CO)
        self.play(LaggedStart(*[Create(s_) for s_ in s_in], lag_ratio=0.35),
                  run_time=1.2)
        self.play(FadeIn(lin), run_time=0.4)
        self.play(LaggedStart(*[Create(s_) for s_ in s_out], lag_ratio=0.35),
                  run_time=1.2)
        self.play(FadeIn(lout), run_time=0.4)
        for li in (lin, lout):
            for sp_ in strand_pts(ins[k]) + strand_pts(outs[k]):
                for u_ in range(len(sp_) - 1):
                    _clear_of_segment(li, sp_[u_], sp_[u_ + 1], gap=0.04,
                                      what="a count (strand)")
        self.play(LaggedStart(*[FadeIn(m) for m in groups[k]], lag_ratio=0.12),
                  FadeIn(heads[k]), run_time=1.4)
        self.play(FadeOut(s_in), FadeOut(s_out), run_time=0.4)

        # the other points
        labs = [lin, lout]
        for k in (0, 2, 3):
            li, lo = lab_in(k), lab_out(k)
            labs += [li, lo]
            self.play(FadeIn(li), FadeIn(lo), FadeIn(groups[k]), FadeIn(heads[k]),
                      run_time=0.8)

        ymid = (groups[1].get_top()[1] + groups[1].get_bottom()[1]) / 2
        plus = VGroup(*[tag("+", 34).move_to([gx[k] + gw[k] + gap / 2, ymid, 0])
                        for k in range(n)])
        eq20 = tag("=  20", 34, YELLOW_B)
        eq20.move_to([gx[n] + gw[n] + 0.45 + eq20.width / 2, ymid, 0])
        top = tag("1² + 3² + 3² + 1²  =  20  =  C(6, 3)", 32, YELLOW_B).move_to(
            [2.3, 2.55, 0])
        self.play(FadeIn(plus), FadeIn(eq20), run_time=0.7)
        self.play(FadeIn(top, shift=UP * 0.1), run_time=0.8)

        for m_ in [*groups.values(), *heads.values(), plus, eq20, top, *labs]:
            check(_inside(m_), "inside the frame")
        for li in labs:
            _clear_of_segment(li, anti.get_start(), anti.get_end(), gap=0.04,
                              what=f"count {li.text}")
            check(all(_apart(li, d, 0.02) for d in pts), "counts clear of the points")
        check(top.get_left()[0] > B((n, 0))[0] + 0.9, "equation clear of the grid")
        check(VGroup(*heads.values()).get_top()[1] < lO.get_bottom()[1] - 0.1,
              "gallery below the grid")
        check(all(_apart(groups[a], groups[b], 0.3) for a in range(n + 1)
                  for b in range(a + 1, n + 1)), "groups apart")
        cap = caption("C(n, 0)² + C(n, 1)² + … + C(n, n)²  =  C(2n, n)", 34)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K16

class K16_LinesCutPlane(Board):
    """Lines in general position (no two parallel, no three through a
    point), drawn one at a time. The k-th line meets the k − 1 lines
    before it in k − 1 points, which cut it into k pieces; each piece runs
    across one region and splits it in two, so the line adds k regions.
    Rₙ = 1 + (1 + 2 + … + n) = 1 + n + C(n, 2). The window holds every
    crossing, so it shows every region of the plane (the outer ones cut
    off by the frame). Shown for n = 5: 16 regions."""

    def construct(self):
        th = [15.6, 63.1, 106.7, 136.8, 165.2]
        base = [(-1.1, 0.2), (1.19, 0.21), (0.5, -1.16), (0.48, 0.52), (0.03, -0.47)]
        n = len(th)
        Ls = [(1.3 * (np.array(p, float) - np.array([0.17, 0.0])),
               np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]))
              for p, t in zip(base, th)]
        Wx, Wy = 3.15, 3.25                        # the window, math units
        rect = [np.array(v, float) for v in ((-Wx, -Wy), (Wx, -Wy), (Wx, Wy), (-Wx, Wy))]

        def meet(l1, l2):
            (p1, d1), (p2, d2) = l1, l2
            den = d1[0] * d2[1] - d1[1] * d2[0]
            w = p2 - p1
            return p1 + ((w[0] * d2[1] - w[1] * d2[0]) / den) * d1

        def clip(l):
            """The part of line l inside the window."""
            p, d = l
            ts = []
            for ax, lim in ((0, Wx), (1, Wy)):
                if abs(d[ax]) > 1e-12:
                    for s_ in (-lim, lim):
                        t = (s_ - p[ax]) / d[ax]
                        q = p + t * d
                        if abs(q[1 - ax]) <= (Wy if ax == 0 else Wx) + 1e-9:
                            ts.append(t)
            return p + min(ts) * d, p + max(ts) * d

        X = {(i, j): meet(Ls[i], Ls[j]) for i, j in combinations(range(n), 2)}
        check(all(abs(x[0]) < Wx - 0.5 and abs(x[1]) < Wy - 0.5 for x in X.values()),
              "every crossing inside the window")
        check(min(np.linalg.norm(a - b) for a, b in combinations(X.values(), 2)) > 0.45,
              "the crossings are distinct points")
        for (i, j), x in X.items():
            for k in range(n):
                if k not in (i, j):
                    p, d = Ls[k]
                    v = x - p
                    check(abs(v[0] * d[1] - v[1] * d[0]) > 0.4, "no three lines through a point")
        check(min(abs((a - b + 90) % 180 - 90) for a, b in combinations(th, 2)) > 20,
              "no two lines parallel")
        segs = [clip(l) for l in Ls]
        faces, added = _arrange(rect, segs)
        check(added == [1, 2, 3, 4, 5] and len(faces) == 16 == 1 + n + _binom(n, 2),
              "line k adds k regions: 16")
        for nn in range(0, 30):
            check(1 + sum(range(1, nn + 1)) == 1 + nn + _binom(nn, 2), "1 + n + C(n, 2)")

        Cs = np.array([-3.35, 0.4, 0.0])

        def S(p):
            return Cs + np.array([p[0], p[1], 0.0])

        CA, CB = BLUE_D, ORANGE
        frame = Polygon(*[S(v) for v in rect], color=GREY_B, stroke_width=2.5)

        def tint(m):
            fs, _ = _arrange(rect, segs[:m])
            check(len(fs) == 1 + m + _binom(m, 2), f"{len(fs)} regions after {m} lines")
            return VGroup(*[Polygon(*[S(p) for p in f], stroke_width=0,
                                    fill_color=(CA, CB)[_parity(f, segs[:m])],
                                    fill_opacity=0.42) for f in fs])

        # the panel: one row per line
        px, ys = 0.35, [3.2, 2.5, 1.8, 1.1, 0.4, -0.3]
        R = [1 + k + _binom(k, 2) for k in range(n + 1)]
        sub = "₀₁₂₃₄₅"

        def row(k):
            if k == 0:
                g = VGroup(tag("R₀  =  1", 28))
            else:
                g = VGroup(tag(f"R{sub[k]}  =  {R[k - 1]}", 28),
                           tag(f"+ {k}", 28, YELLOW_B), tag(f"=  {R[k]}", 28))
                g.arrange(RIGHT, buff=0.25)
            g.move_to([px + g.width / 2, ys[k], 0])
            return g

        t0, r0 = tint(0), row(0)
        self.play(FadeIn(t0), Create(frame), FadeIn(r0), run_time=1.0)
        cur = t0
        rows = [r0]
        lines, dots = VGroup(), VGroup()
        for k in range(1, n + 1):
            a, b = segs[k - 1]
            xs = sorted([X[(min(i, k - 1), max(i, k - 1))] for i in range(k - 1)],
                        key=lambda x: float(np.dot(x - a, b - a)))
            nodes = [a] + xs + [b]
            pieces = VGroup(*[Line(S(nodes[t]), S(nodes[t + 1]),
                                   color=(YELLOW_B, RED_C)[t % 2], stroke_width=8)
                              for t in range(len(nodes) - 1)])
            check(len(pieces) == k, f"line {k} is cut into {k} pieces")
            nd = VGroup(*[Dot(S(x), radius=0.07, color=WHITE) for x in xs])
            self.play(LaggedStart(*[Create(pc) for pc in pieces], lag_ratio=0.6),
                      FadeIn(nd), run_time=0.9)
            new = tint(k)
            r_ = row(k)
            self.add(new)
            self.bring_to_back(new)
            new.set_fill(opacity=0)
            self.play(cur.animate.set_fill(opacity=0), new.animate.set_fill(opacity=0.42),
                      FadeIn(r_), run_time=0.7)
            self.remove(cur)
            cur = new
            rows.append(r_)
            self.play(pieces.animate.set_stroke(color=WHITE, width=3), run_time=0.3)
            lines.add(pieces)
            dots.add(*nd)
            self.bring_to_front(dots)

        f1 = tag("R₅ = 1 + 1 + 2 + 3 + 4 + 5", 28, YELLOW_B)
        f2 = tag("= 1 + 5 + C(5, 2) = 16", 28, YELLOW_B)
        f1.move_to([px + f1.width / 2, -1.25, 0])
        f2.move_to([px + 0.48 + f2.width / 2, -1.9, 0])
        self.play(FadeIn(f1, shift=UP * 0.1), run_time=0.7)
        self.play(FadeIn(f2, shift=UP * 0.1), run_time=0.7)
        for m_ in rows + [f1, f2, frame]:
            check(_inside(m_), "inside the frame")
        check(min(r_.get_left()[0] for r_ in rows) > frame.get_right()[0] + 0.4,
              "panel clear of the window")
        cap = caption("Rₙ  =  1 + (1 + 2 + … + n)  =  1 + n + C(n, 2)", 34)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K18

class K18_RamseySix(Board):
    """Colour the 15 edges of K₆ red and blue. A vertex v has five edges;
    two colours cannot share five edges two-and-two, so three of them, say
    red, go to a, b, c. If any of ab, bc, ca is red, it closes a red
    triangle with v; if none is, abc is a blue triangle. So every colouring
    of K₆ has a one-colour triangle. Five vertices are not enough: colour
    K₅ as a red pentagon and a blue pentagram — each colour is a 5-cycle,
    which has no triangle. Hence R(3, 3) = 6."""

    def construct(self):
        E6 = list(combinations(range(6), 2))
        red = {(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 4), (3, 4), (3, 5)}
        col = {e: ("R" if e in red else "B") for e in E6}

        def mono(c, vs):
            return [t for t in combinations(vs, 3)
                    if len({c[(t[0], t[1])], c[(t[0], t[2])], c[(t[1], t[2])]}) == 1]

        # the argument works for every colouring and every vertex
        for bits in range(1 << 15):
            c = {e: ("R" if (bits >> i) & 1 else "B") for i, e in enumerate(E6)}
            for v in range(6):
                nb = [u for u in range(6) if u != v]
                for C_ in "RB":
                    same = [u for u in nb if c[tuple(sorted((v, u)))] == C_]
                    if len(same) >= 3:
                        a, b, cc = same[:3]
                        tri = [(a, b), (b, cc), (a, cc)]
                        ok = (any(c[tuple(sorted(e))] == C_ for e in tri)
                              or all(c[tuple(sorted(e))] != C_ for e in tri))
                        check(ok, "the case split is exhaustive")
                        break
                else:
                    check(False, "five edges, two colours: three share one")
            if bits % 4096 == 0:
                check(len(mono(c, range(6))) > 0, "a one-colour triangle")
        check(all(len(mono({e: ("R" if (bits >> i) & 1 else "B")
                            for i, e in enumerate(E6)}, range(6))) > 0
                  for bits in range(0, 1 << 15, 7)), "sampled: always a triangle")
        E5 = list(combinations(range(5), 2))
        c5 = {e: ("R" if (e[1] - e[0]) % 5 in (1, 4) else "B") for e in E5}
        check(mono(c5, range(5)) == [], "pentagon / pentagram: no one-colour triangle")
        def E(x, y):
            return (min(x, y), max(x, y))

        v, a, b, c_ = 0, 5, 3, 1                  # a left, b bottom, c right
        check([col[(0, u)] for u in range(1, 6)].count("R") == 3, "v: three red, two blue")
        check(all(col[E(v, u)] == "R" for u in (a, b, c_)), "v-a, v-b, v-c red")
        check(col[E(a, b)] == "R" and col[E(b, c_)] == "B" and col[E(a, c_)] == "B",
              "here ab is red: v, a, b is the red triangle")

        CR, CB = RED_C, BLUE_C
        hue = {"R": CR, "B": CB}
        K6c, r6 = np.array([-4.3, 0.4, 0.0]), 2.0
        V6 = [K6c + r6 * np.array([np.cos(np.radians(90 - 60 * i)),
                                   np.sin(np.radians(90 - 60 * i)), 0.0]) for i in range(6)]
        edges = {e: Line(V6[e[0]], V6[e[1]], color=hue[col[e]], stroke_width=4)
                 for e in E6}
        vd = VGroup(*[Dot(p, radius=0.1, color=WHITE) for p in V6])
        k6l = tag("K₆", 30).move_to([-6.1, 3.3, 0])
        self.play(LaggedStart(*[Create(edges[e]) for e in E6], lag_ratio=0.06),
                  FadeIn(vd), FadeIn(k6l), run_time=1.6)

        def vlab(s, i, ctr, r, size=26):
            d = V6[i] - ctr                       # radially outside the vertex
            return tag(s, size).move_to(ctr + (r + 0.36) * d / np.linalg.norm(d))

        lv = vlab("v", v, K6c, r6)
        others = [e for e in E6 if v not in e]
        self.play(FadeIn(lv), *[edges[e].animate.set_stroke(opacity=0.18) for e in others],
                  *[edges[(v, u)].animate.set_stroke(width=8) for u in range(1, 6)],
                  run_time=0.9)

        # five edges into two piles: one pile has at least three
        px0, yR, yB, bl = -1.1, 1.55, 0.35, 0.6
        reds = [u for u in range(1, 6) if col[(v, u)] == "R"]
        blues = [u for u in range(1, 6) if col[(v, u)] == "B"]
        bars = {}
        for k_, u in enumerate(reds):
            bars[u] = Line([px0 + 0.3 * k_, yR - bl / 2, 0], [px0 + 0.3 * k_, yR + bl / 2, 0],
                           color=CR, stroke_width=8)
        for k_, u in enumerate(blues):
            bars[u] = Line([px0 + 0.3 * k_, yB - bl / 2, 0], [px0 + 0.3 * k_, yB + bl / 2, 0],
                           color=CB, stroke_width=8)
        self.play(*[TransformFromCopy(edges[(v, u)], bars[u]) for u in range(1, 6)],
                  run_time=1.2)
        ge3 = tag("≥ 3", 30, CR).next_to(VGroup(*[bars[u] for u in reds]), RIGHT, buff=0.3)
        pig = tag("5 > 2 + 2", 28, GREY_A).move_to([px0 + 0.45, -0.55, 0])
        self.play(FadeIn(ge3), FadeIn(pig), run_time=0.7)

        # their ends a, b, c, and the three edges among them
        la, lb, lc = vlab("a", a, K6c, r6), vlab("b", b, K6c, r6), vlab("c", c_, K6c, r6)
        tri = [E(a, b), E(b, c_), E(a, c_)]
        self.play(FadeIn(la), FadeIn(lb), FadeIn(lc),
                  *[edges[(v, u)].animate.set_stroke(opacity=0.18, width=4) for u in blues],
                  *[edges[e].animate.set_stroke(opacity=1, width=7) for e in tri],
                  run_time=0.9)

        # the two cases, side by side
        def case(ctr, kind):
            P = {"v": ctr + np.array([0.0, 0.85, 0]), "a": ctr + np.array([-0.8, -0.42, 0]),
                 "b": ctr + np.array([0.0, -0.75, 0]), "c": ctr + np.array([0.8, -0.42, 0])}
            g = VGroup()
            for u in "abc":
                g.add(Line(P["v"], P[u], color=CR, stroke_width=5))
            if kind == "red":
                g.add(DashedLine(P["b"], P["c"], color=GREY_B, stroke_width=3,
                                 dash_length=0.08),
                      DashedLine(P["a"], P["c"], color=GREY_B, stroke_width=3,
                                 dash_length=0.08),
                      Line(P["a"], P["b"], color=CR, stroke_width=5))
                fill = Polygon(P["v"], P["a"], P["b"], color=CR, fill_opacity=0.35,
                               stroke_width=0)
            else:
                g.add(*[Line(P[x], P[y], color=CB, stroke_width=5)
                        for x, y in ("ab", "bc", "ac")])
                fill = Polygon(P["a"], P["b"], P["c"], color=CB, fill_opacity=0.35,
                               stroke_width=0)
            g.add(*[Dot(P[u], radius=0.08) for u in "vabc"])
            labs = VGroup(*[tag(u, 22).move_to(P[u] + d) for u, d in
                            (("v", UP * 0.3), ("a", LEFT * 0.28), ("b", DOWN * 0.3),
                             ("c", RIGHT * 0.28))])
            return g, fill, labs

        gA, fA, lA = case(np.array([2.2, 2.15, 0.0]), "red")
        gB, fB, lB = case(np.array([5.15, 2.15, 0.0]), "blue")
        orw = tag("or", 28, GREY_A).move_to([3.67, 2.0, 0])
        self.play(FadeIn(gA), FadeIn(lA), run_time=0.8)
        self.play(FadeIn(fA), run_time=0.5)
        self.play(FadeIn(orw), FadeIn(gB), FadeIn(lB), run_time=0.8)
        self.play(FadeIn(fB), run_time=0.5)

        # here: ab is red, so v, a, b is a red triangle
        found = Polygon(V6[v], V6[a], V6[b], color=CR, fill_opacity=0.3, stroke_width=0)
        self.add(found)
        self.bring_to_back(found)
        found.set_fill(opacity=0)
        self.play(found.animate.set_fill(opacity=0.3),
                  *[edges[e].animate.set_stroke(opacity=0.18, width=4)
                    for e in tri if e != E(a, b)],
                  edges[E(a, b)].animate.set_stroke(width=9),
                  edges[E(v, a)].animate.set_stroke(width=9),
                  edges[E(v, b)].animate.set_stroke(width=9),
                  edges[E(v, c_)].animate.set_stroke(opacity=0.3), run_time=1.0)

        # five vertices are not enough
        K5c, r5 = np.array([1.75, -1.6, 0.0]), 1.0
        V5 = [K5c + r5 * np.array([np.cos(np.radians(90 + 72 * i)),
                                   np.sin(np.radians(90 + 72 * i)), 0.0]) for i in range(5)]
        k5e = VGroup(*[Line(V5[e[0]], V5[e[1]], color=hue[c5[e]], stroke_width=4)
                       for e in E5])
        k5d = VGroup(*[Dot(p, radius=0.08) for p in V5])
        k5l = tag("K₅", 30).move_to(K5c + np.array([-1.45, 0.0, 0]))
        self.play(Create(k5e), FadeIn(k5d), FadeIn(k5l), run_time=1.0)

        def cls(kind, ctr, r):
            W = [ctr + r * np.array([np.cos(np.radians(90 + 72 * i)),
                                     np.sin(np.radians(90 + 72 * i)), 0.0]) for i in range(5)]
            return VGroup(*[Line(W[e[0]], W[e[1]], color=hue[kind], stroke_width=4)
                            for e in E5 if c5[e] == kind],
                          *[Dot(p, radius=0.06) for p in W])

        redc = cls("R", np.array([4.05, -1.6, 0.0]), 0.62)
        bluc = cls("B", np.array([5.75, -1.6, 0.0]), 0.62)
        self.play(TransformFromCopy(VGroup(*[k5e[i] for i, e in enumerate(E5)
                                             if c5[e] == "R"]), VGroup(*redc[:5])),
                  TransformFromCopy(VGroup(*[k5e[i] for i, e in enumerate(E5)
                                             if c5[e] == "B"]), VGroup(*bluc[:5])),
                  FadeIn(redc[5:]), FadeIn(bluc[5:]), run_time=1.2)
        check(len(redc) - 5 == 5 and len(bluc) - 5 == 5, "each colour: five edges, a 5-cycle")

        allm = [k6l, lv, la, lb, lc, ge3, pig, gA, gB, lA, lB, orw, k5e, k5l, redc, bluc]
        for m_ in allm + list(edges.values()):
            check(_inside(m_), "inside the frame")
        for t_ in (lv, la, lb, lc):
            for e in E6:
                _clear_of_segment(t_, V6[e[0]], V6[e[1]], gap=0.03, what=t_.text)
        check(_apart(VGroup(gA, lA), VGroup(gB, lB), 0.4), "the two cases apart")
        check(_apart(orw, gA, 0.1) and _apart(orw, gB, 0.1), "'or' between the cases")
        check(VGroup(gA, gB, lA, lB).get_bottom()[1] > VGroup(k5e, k5l, redc, bluc
                                                               ).get_top()[1] + 0.3,
              "cases clear of K₅")
        check(VGroup(*bars.values(), ge3, pig).get_right()[0] < VGroup(gA, lA).get_left()[0]
              - 0.4, "piles clear of the cases")
        check(VGroup(*bars.values(), pig).get_left()[0] > K6c[0] + r6 + 0.6,
              "piles clear of K₆")
        cap = caption("R(3, 3) = 6 :  every red/blue K₆ has a one-colour triangle,"
                      "  K₅ need not", 30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)


# ==================================================================== K20

class K20_Handshake(Board):
    """Cut every edge at its midpoint: each half belongs to the vertex it
    touches, so a vertex owns exactly deg v halves. Count the halves two
    ways: by edges, two each — 2E; by vertices — the sum of the degrees.
    The same halves, so ∑ deg v = 2E. In each vertex's stack the halves
    pair off, leaving one over exactly when the degree is odd; the total
    is even, so the left-over halves pair up too: an even number of
    vertices have odd degree. (A graph with 7 vertices and 10 edges.)"""

    def construct(self):
        V = [(-5.6, 2.2), (-3.7, 3.0), (-3.8, 0.9), (-1.9, 2.0), (-5.9, -0.4),
             (-3.9, -1.6), (-1.8, -0.5)]
        V = [np.array([x, y, 0.0]) for x, y in V]
        Ed = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 3), (2, 5), (2, 6), (3, 6), (4, 5),
              (5, 6)]
        nv, ne = len(V), len(Ed)
        deg = [sum(1 for e in Ed if u in e) for u in range(nv)]
        check(sum(deg) == 2 * ne == 20, "∑ deg = 2E = 20")
        odd = [u for u in range(nv) if deg[u] % 2]
        check(len(odd) % 2 == 0 and odd == [0, 3, 5, 6], "four odd vertices")
        for e1, e2 in combinations(Ed, 2):
            if not set(e1) & set(e2):
                check(_xing(V[e1[0]][:2], V[e1[1]][:2], V[e2[0]][:2], V[e2[1]][:2]) is None,
                      "the drawing has no crossing edges")
        rng = np.random.default_rng(1)
        for _ in range(200):                       # the lemma on random graphs
            m = int(rng.integers(2, 9))
            es = [e for e in combinations(range(m), 2) if rng.random() < 0.5]
            dg = [sum(1 for e in es if u in e) for u in range(m)]
            check(sum(dg) == 2 * len(es) and sum(d % 2 for d in dg) % 2 == 0,
                  "handshake lemma")

        cols = [BLUE_C, YELLOW_C, ORANGE, GREEN_C, RED_C, PINK, PURPLE_A]
        lines = VGroup(*[Line(V[a], V[b], color=GREY_B, stroke_width=5) for a, b in Ed])
        dots = VGroup(*[Dot(V[u], radius=0.13, color=cols[u]) for u in range(nv)])
        self.play(Create(lines), FadeIn(dots), run_time=1.2)

        # every edge is two halves, one for each end
        halves = {}
        for i, (a, b) in enumerate(Ed):
            mid = (V[a] + V[b]) / 2
            halves[(i, a)] = Line(V[a], mid, color=cols[a], stroke_width=7)
            halves[(i, b)] = Line(mid, V[b], color=cols[b], stroke_width=7)
        ticks = VGroup(*[Dot((V[a] + V[b]) / 2, radius=0.05, color=WHITE) for a, b in Ed])
        self.play(FadeOut(lines), *[FadeIn(h) for h in halves.values()], FadeIn(ticks),
                  run_time=1.0)
        self.bring_to_front(dots)

        def out_dir(u):
            """Direction for the degree label: the middle of the widest gap
            between the vertex's edges."""
            ang = sorted(float(np.arctan2(*(V[w] - V[u])[1::-1]))
                         for e in Ed if u in e for w in e if w != u)
            gaps = [((ang[(i + 1) % len(ang)] - ang[i]) % TAU or TAU, ang[i])
                    for i in range(len(ang))]
            g, a0 = max(gaps)
            t = a0 + g / 2
            return np.array([np.cos(t), np.sin(t), 0.0])

        dlabs = VGroup(*[tag(str(deg[u]), 28, cols[u]).move_to(V[u] + 0.5 * out_dir(u))
                         for u in range(nv)])
        for u, lb in enumerate(dlabs):
            for a, b in Ed:
                _clear_of_segment(lb, V[a], V[b], gap=0.04, what=f"degree of vertex {u}")
            check(_inside(lb), "degree label inside the frame")
        self.play(FadeIn(dlabs), run_time=0.8)

        # count the halves by edges: two per edge
        L, gapy = 0.42, 0.08
        xA = [0.35 + 0.56 * i for i in range(ne)]
        yA = 2.55
        segA = {}
        for i, (a, b) in enumerate(Ed):
            segA[(i, a)] = Line([xA[i], yA + gapy / 2, 0], [xA[i], yA + gapy / 2 + L, 0],
                                color=cols[a], stroke_width=9)
            segA[(i, b)] = Line([xA[i], yA - gapy / 2 - L, 0], [xA[i], yA - gapy / 2, 0],
                                color=cols[b], stroke_width=9)
        tokens = {k: halves[k].copy() for k in halves}
        self.play(*[Transform(tokens[k], segA[k]) for k in halves], run_time=1.6)
        labA = tag("2E  =  2 · 10  =  20", 30, YELLOW_B).move_to(
            [(xA[0] + xA[-1]) / 2, yA + gapy / 2 + L + 0.4, 0])
        self.play(FadeIn(labA), run_time=0.6)

        # the same halves, by vertices: deg v of them in v's stack
        xB = [0.3 + 0.75 * u for u in range(nv)]
        yB0 = -1.75
        segB, level = {}, [0] * nv
        for i, (a, b) in enumerate(Ed):
            for u in (a, b):
                y0 = yB0 + level[u] * (L + gapy)
                segB[(i, u)] = Line([xB[u], y0, 0], [xB[u], y0 + L, 0], color=cols[u],
                                    stroke_width=9)
                level[u] += 1
        check(level == deg, "each stack has deg v halves")
        tokB = {k: tokens[k].copy() for k in halves}
        self.play(*[Transform(tokB[k], segB[k]) for k in halves], run_time=1.6)
        nums = VGroup(*[tag(str(deg[u]), 30, cols[u]).move_to([xB[u], yB0 - 0.35, 0])
                        for u in range(nv)])
        pl = VGroup(*[tag("+", 28, GREY_A).move_to([(xB[u] + xB[u + 1]) / 2, yB0 - 0.35, 0])
                      for u in range(nv - 1)])
        eqB = tag("=  20", 30, YELLOW_B)
        eqB.move_to([xB[-1] + 0.4 + eqB.width / 2, yB0 - 0.35, 0])
        self.play(FadeIn(nums), FadeIn(pl), FadeIn(eqB), run_time=0.8)

        # pair the halves in each stack: one is left over where deg v is odd
        pairs = VGroup()
        for u in range(nv):
            for t in range(deg[u] // 2):
                y0 = yB0 + 2 * t * (L + gapy) - 0.04
                pairs.add(RoundedRectangle(corner_radius=0.06, width=0.3,
                                           height=2 * L + gapy + 0.08,
                                           stroke_color=GREY_B, stroke_width=2).move_to(
                    [xB[u], y0 + (2 * L + gapy + 0.08) / 2, 0]))
        tops = {u: segB[next(k for k in segB if k[1] == u and
                             abs(segB[k].get_start()[1] - (yB0 + (deg[u] - 1) * (L + gapy)))
                             < 1e-9)] for u in odd}
        spare = VGroup(*[SurroundingRectangle(tops[u], buff=0.06, color=YELLOW_B,
                                              stroke_width=4) for u in odd])
        self.play(Create(pairs), run_time=0.8)
        self.play(Create(spare), run_time=0.6)
        arcs = VGroup()
        for u, w in ((odd[0], odd[1]), (odd[2], odd[3])):
            p, q = tops[u].get_end() + UP * 0.1, tops[w].get_end() + UP * 0.1
            h = yB0 + max(deg[u:w + 1]) * (L + gapy) + 0.3   # above every stack between
            arcs.add(VMobject(stroke_color=YELLOW_B, stroke_width=3).set_points_smoothly(
                [p, np.array([p[0], h, 0]), np.array([q[0], h, 0]), q]))
        rings = VGroup(*[Circle(radius=0.27, stroke_color=YELLOW_B, stroke_width=4
                                ).move_to(V[u]) for u in odd])
        self.play(Create(arcs), Create(rings), run_time=1.0)

        stacks = VGroup(*[tokB[k] for k in halves])
        for a_ in arcs:                          # the arcs pass over the stacks
            curve = [a_.point_from_proportion(t) for t in np.linspace(0, 1, 400)]
            for u in range(nv):
                top_u = yB0 + deg[u] * (L + gapy)
                for pt in curve:
                    if abs(pt[0] - xB[u]) < 0.1:
                        check(pt[1] > top_u, "arcs clear of the stacks")
        for m_ in (labA, nums, pl, eqB, stacks, arcs, VGroup(*segA.values())):
            check(_inside(m_), "inside the frame")
        check(arcs.get_top()[1] < VGroup(*segA.values()).get_bottom()[1] - 0.3,
              "arcs clear of the dominoes")
        check(VGroup(*[tokens[k] for k in halves], *[tokB[k] for k in halves], pairs,
                     labA).get_left()[0] > max(v_[0] for v_ in V) + 0.6,
              "displays clear of the graph")
        check(labA.get_bottom()[1] > VGroup(*segA.values()).get_top()[1] + 0.1,
              "2E line above the dominoes, out of the halves' way")
        for u in odd:
            for dl in dlabs:                     # nearest point of each label box
                lo, hi = dl.get_corner(DL), dl.get_corner(UR)
                near = np.clip(V[u], lo, hi)
                check(np.linalg.norm((near - V[u])[:2]) > 0.27 + 0.05,
                      "rings clear of the degree labels")
        cap = caption("∑ deg v  =  2E   ⟹   an even number of vertices have odd degree", 30)
        self.play(Write(cap))
        _final_check(self, cap)
        self.hold(2.2)
