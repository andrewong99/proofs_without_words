# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w1a.py — proofs without words, 2D (manim): sums and series.
#
#     B13 Gauss's pairing              B14 sum of even numbers
#     B15 sum of squares in the plane  B16 two triangular numbers, one square
#     B17 eight staircases and a dot   B18 up and down the staircase
#     B20 powers of two                B23 arithmetic series
#     B34 the harmonic series          B49 halving a triangle
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Every tiling is checked cell by cell, and every rigid motion is checked to
# land where the picture says it lands, with check(...), so a wrong
# construction fails the render instead of drawing a wrong picture.

from fractions import Fraction


# ---------------------------------------------------------------- helpers

def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _tiles_once(pieces, W, H):
    """True if the cell sets in `pieces` cover the W x H grid exactly once."""
    seen = {}
    for cells in pieces:
        for c in cells:
            seen[c] = seen.get(c, 0) + 1
    full = {(i, j) for i in range(W) for j in range(H)}
    return set(seen) == full and all(v == 1 for v in seen.values())


def _safe(*mobs, tol=0.03):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for m in mobs:
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area: {type(m).__name__} "
              f"[{lo[0]:.2f},{hi[0]:.2f}]x[{lo[1]:.2f},{hi[1]:.2f}]")


def _apart(*mobs, gap=0.04):
    """Fail the render if any two mobjects' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            a0, a1 = mobs[i].get_critical_point(DL), mobs[i].get_critical_point(UR)
            b0, b1 = mobs[j].get_critical_point(DL), mobs[j].get_critical_point(UR)
            sep = (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
                   or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])
            check(sep, f"boxes {i} and {j} do not overlap")


def _dim(p, q, label, off, color=GREY_A, size=24, gap=0.16):
    """Dimension line beside the screen segment pq, shifted by `off`: a thin
    line with end ticks, and the label beyond it."""
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


def _rigid(mob, angle=0.0, about=None, shift=ORIGIN, **kw):
    """Rigid motion as an animation: rotate by `angle` about `about` and
    translate by `shift`, both growing with alpha, so every frame shows a
    congruent copy (never a vertex-interpolating Transform)."""
    start = mob.copy()
    about = mob.get_center() if about is None else to3(about)
    shift = to3(shift)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * angle, about_point=about)
                 .shift(alpha * shift))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _clear_of(label, group, gap=0.05):
    """Fail the render if the label's box touches any member of `group`
    (members tested one by one: a staircase's own box includes its empty
    corner, which is exactly where its name goes)."""
    for m in group:
        _apart(label, m, gap=gap)


def _same_spots(group, spots, tol=1e-6):
    """Every member of `group` sits on one of `spots` (screen points), and
    every spot is taken: the motion landed exactly."""
    got = sorted(tuple(np.round(m.get_center()[:2] / tol).astype(int))
                 for m in group)
    want = sorted(tuple(np.round(np.asarray(s, float)[:2] / tol).astype(int))
                  for s in spots)
    return got == want


def _cap(group):
    """Place a composite formula where caption() puts its Text."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.to_edge(DOWN, buff=0.3)
    group.set_x(0.0)
    return group


def _sup_line(before, sup, after, size=32, color=YELLOW_B, gap=0.3):
    """'before' + a small raised 'sup' + 'after' on one baseline, built from
    Text pieces (superscripts such as n−1 have no safe Unicode glyphs).
    Both main pieces must end on digits or symbols without descenders."""
    t1 = Text(before, font_size=size, color=color)
    t2 = Text(sup, font_size=size * 0.6, color=color)
    t3 = Text(after, font_size=size, color=color)
    k = size / 32
    t2.next_to(t1, RIGHT, buff=0.03 * k).align_to(t1, UP).shift(UP * 0.07 * k)
    t3.next_to(t2, RIGHT, buff=gap * k).align_to(t1, DOWN)
    return VGroup(t1, t2, t3)


# ====================================================================== B13

class B13_GaussPairing(Board):
    """Gauss's pairing. The numbers 1, 2, …, n stand as columns of 1, 2, …,
    n cells. The same columns turned over (a half-turn of the whole row)
    drop into the gaps above them, so along the top the row now reads
    n, n−1, …, 1. Every column holds k + (n+1−k) = n+1 cells: the n columns
    make an n × (n+1) rectangle, which is twice 1 + 2 + … + n.
    Drawn for n = 8."""

    def construct(self):
        n, u = 8, 0.56
        W, H = n, n + 1
        x0, y0 = -W * u / 2 - 0.35, -1.79
        org = np.array([x0, y0, 0.0])
        ctr = org + np.array([W * u / 2, H * u / 2, 0.0])

        def at(i, j):                       # centre of cell (i, j)
            return org + np.array([(i + 0.5) * u, (j + 0.5) * u, 0.0])

        low = [(i, j) for i in range(n) for j in range(i + 1)]
        high = [(n - 1 - i, n - j) for (i, j) in low]       # half-turn image
        check(_tiles_once([low, high], W, H),
              "the row and the row turned over tile n x (n+1)")
        for i in range(n):
            check((i + 1) + sum(1 for c in high if c[0] == i) == n + 1,
                  f"column {i} holds n+1 cells")

        cols = VGroup(*[VGroup(*[cell(i, j, u, org, BLUE_D)
                                 for j in range(i + 1)]) for i in range(n)])
        below = VGroup(*[tag(str(i + 1), 26, BLUE_B).move_to(
            at(i, 0) + DOWN * (0.5 * u + 0.3)) for i in range(n)])

        self.play(LaggedStart(*[AnimationGroup(FadeIn(c, shift=UP * 0.15),
                                               FadeIn(b))
                                for c, b in zip(cols, below)],
                              lag_ratio=0.25), run_time=2.6)
        self.hold(0.6)

        # the same row, turned over: a half-turn about the rectangle's centre
        turned = cols.copy().set_fill(TEAL_D)
        self.add(turned)
        self.play(Rotate(turned, PI, about_point=ctr), run_time=2.2)
        check(_same_spots([c for col in turned for c in col],
                          [at(i, j) for (i, j) in high]),
              "the turned row lands exactly in the gaps")
        above = VGroup(*[tag(str(n - i), 26, TEAL_B).move_to(
            at(i, n) + UP * (0.5 * u + 0.3)) for i in range(n)])
        self.play(LaggedStart(*[FadeIn(a) for a in above], lag_ratio=0.12),
                  run_time=1.0)
        self.hold(0.5)

        # every column: k below, n+1−k above, n+1 cells in all
        def col_box(i):
            return Rectangle(width=u, height=H * u, stroke_color=YELLOW_B,
                             stroke_width=6).move_to(
                org + np.array([(i + 0.5) * u, H * u / 2, 0.0]))

        box = col_box(0)
        self.play(Create(box), below[0].animate.set_color(YELLOW_B),
                  above[0].animate.set_color(YELLOW_B), run_time=0.5)
        for i in range(1, n):
            self.play(Transform(box, col_box(i)),
                      below[i - 1].animate.set_color(BLUE_B),
                      above[i - 1].animate.set_color(TEAL_B),
                      below[i].animate.set_color(YELLOW_B),
                      above[i].animate.set_color(YELLOW_B), run_time=0.32)
        self.play(FadeOut(box), below[n - 1].animate.set_color(BLUE_B),
                  above[n - 1].animate.set_color(TEAL_B), run_time=0.4)

        frame = Rectangle(width=W * u, height=H * u, stroke_color=YELLOW_B,
                          stroke_width=5).move_to(ctr)
        right = org + np.array([W * u, 0.0, 0.0])
        d_h = _dim(right, right + UP * H * u, "n+1", RIGHT * 0.3, YELLOW_B, 28)
        d_w = _dim(org + DOWN * 0.66, right + DOWN * 0.66, "n", DOWN * 0.0001,
                   YELLOW_B, 28, gap=0.12)
        _safe(below, above, d_h, d_w)
        _apart(*below, d_w[1])
        self.play(Create(frame), FadeIn(d_h), FadeIn(d_w), run_time=1.0)
        self.play(Write(caption("2 · (1 + 2 + … + n)  =  n(n+1)", 34)))
        self.hold(2.2)


# ====================================================================== B34

class B34_HarmonicDivergence(Board):
    """Oresme. The terms 1, ½, ⅓, ¼, … stand as bars. Group them ½ | ⅓ + ¼ |
    1/5 + … + 1/8 | …: group j holds the 2^(j−1) bars from 1/(2^(j−1)+1) to
    1/2^j, and every one of them reaches at least the height 1/2^j of the
    last. Cut each bar at that level: the 2^(j−1) lower pieces stack to
    exactly ½, and the cut-off tops stack above them. So every group is at
    least ½, the groups never run out, and the partial sums pass every
    bound. Drawn for the first four groups (½ to 1/16)."""

    def construct(self):
        K, G = 16, 4                           # bars shown, groups shown
        H = 4.2                                # screen height of the bar "1"
        yb = -1.85                             # common baseline
        bw, pitch, xL = 0.30, 0.37, -6.3
        cols = [GREY_B, BLUE_D, TEAL_D, ORANGE, GREEN_D]

        def grp(k):
            return 0 if k == 1 else int(np.ceil(np.log2(k)))

        members = {j: [k for k in range(1, K + 1) if grp(k) == j]
                   for j in range(G + 1)}
        for j in range(1, G + 1):
            ks = members[j]
            check(ks == list(range(2 ** (j - 1) + 1, 2 ** j + 1)),
                  f"group {j} is 1/(2^(j-1)+1) .. 1/2^j")
            check(all(Fraction(1, k) >= Fraction(1, 2 ** j) for k in ks),
                  f"every bar of group {j} reaches 1/2^j")
            check(len(ks) * Fraction(1, 2 ** j) == Fraction(1, 2),
                  f"the cut pieces of group {j} make exactly 1/2")
            check(sum(Fraction(1, k) for k in ks) >= Fraction(1, 2),
                  f"group {j} is at least 1/2")

        def box(x, y, w, h, color, op=FILL, sw=1.2):
            r = Rectangle(width=w, height=max(h, 1e-4), fill_color=color,
                          fill_opacity=op, stroke_color=WHITE,
                          stroke_width=sw)
            r.move_to([x + w / 2, y + h / 2, 0.0])
            return r

        def bx(k):                              # left edge of bar k
            return xL + (k - 1) * pitch

        # whole bars first; at the cut each becomes a lower piece up to its
        # group's level and the top above it (same outline, same colour)
        whole, lows, tops, level = {}, {}, {}, {}
        for k in range(1, K + 1):
            j = grp(k)
            lv = 1.0 if j == 0 else 1.0 / 2 ** j
            level[k] = lv
            whole[k] = box(bx(k), yb, bw, H / k, BLUE_D)
            lows[k] = box(bx(k), yb, bw, lv * H, cols[j])
            ex = (1.0 / k - lv) * H
            if ex > 1e-9:
                tops[k] = box(bx(k), yb + lv * H, bw, ex, cols[j], sw=1.0)
            check(abs(lows[k].get_bottom()[1] - whole[k].get_bottom()[1]) < 1e-9
                  and abs((tops[k].get_top()[1] if k in tops
                           else lows[k].get_top()[1])
                          - whole[k].get_top()[1]) < 1e-9,
                  f"bar {k}: the two pieces make the whole bar")
        base = Line([xL - 0.15, yb, 0], [bx(K) + bw + 0.12, yb, 0],
                    color=GREY_B, stroke_width=2)
        more = tag("…", 30).move_to([bx(K) + bw + 0.42, yb + 0.12, 0])

        names = {1: "1", 2: "½", 3: "⅓", 4: "¼"}
        ylab = yb - 0.42
        labs = VGroup(*[tag(names[k], 24).move_to(
            [bx(k) + bw / 2, ylab, 0]) for k in (1, 2, 3, 4)])
        glabs = VGroup(
            tag("1/5…1/8", 18, cols[3]).move_to(
                [(bx(5) + bx(8) + bw) / 2, ylab, 0]),
            tag("1/9 … 1/16", 18, cols[4]).move_to(
                [(bx(9) + bx(16) + bw) / 2, ylab, 0]))

        self.play(Create(base), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(whole[k], shift=UP * 0.2)
                                for k in range(1, K + 1)],
                              lag_ratio=0.12), FadeIn(labs), run_time=2.4)
        self.play(FadeIn(more), run_time=0.3)

        # group them: ½ | ⅓ + ¼ | 1/5 + … + 1/8 | 1/9 + … + 1/16 | …
        unders = VGroup()
        for j in range(G + 1):
            ks = members[j]
            unders.add(Line([bx(ks[0]) + 0.02, yb - 0.13, 0],
                            [bx(ks[-1]) + bw - 0.02, yb - 0.13, 0],
                            color=cols[j], stroke_width=5))
        self.play(*[whole[k].animate.set_fill(cols[grp(k)])
                    for k in range(1, K + 1)],
                  Create(unders), FadeIn(glabs),
                  labs[1].animate.set_color(BLUE_B),
                  labs[2].animate.set_color(TEAL_B),
                  labs[3].animate.set_color(TEAL_B), run_time=1.0)
        self.hold(0.4)

        # in every group each bar reaches the level of the group's last bar:
        # cut it there
        lv_lines = VGroup()
        for j in range(2, G + 1):
            ks = members[j]
            y = yb + H / 2 ** j
            lv_lines.add(DashedLine([bx(ks[0]) - 0.04, y, 0],
                                    [bx(ks[-1]) + bw + 0.04, y, 0],
                                    color=YELLOW_B, stroke_width=3,
                                    dash_length=0.07))
        self.play(Create(lv_lines), run_time=0.9)
        self.remove(*whole.values())
        self.add(*lows.values(), *tops.values())
        self.bring_to_front(lv_lines)
        self.play(*[tops[k].animate.set_fill(opacity=0.42) for k in tops],
                  run_time=0.7)
        self.hold(0.5)

        # each group, moved to its own column: lower pieces first, tops above
        xr = [None, 1.0, 2.3, 3.6, 4.9]
        rbase = Line([0.55, yb, 0], [5.5, yb, 0], color=GREY_B,
                     stroke_width=2)
        self.play(Create(rbase), run_time=0.5)
        half_y = yb + H / 2
        half = DashedLine([0.62, half_y, 0], [5.55, half_y, 0],
                          color=YELLOW_B, stroke_width=3, dash_length=0.09)
        half_lab = tag("½", 30, YELLOW_B).next_to(half, LEFT, buff=0.12)
        col_labs = VGroup(
            tag("½", 22, BLUE_B), tag("⅓ + ¼", 20, TEAL_B),
            tag("1/5…1/8", 18, cols[3]), tag("1/9…1/16", 18, cols[4]))
        for j in range(1, G + 1):
            col_labs[j - 1].move_to([xr[j] + bw / 2, ylab, 0])
        for j in range(1, G + 1):
            ks = members[j]
            y = yb
            moves, low_copies = [], []
            for k in ks:
                c = lows[k].copy()
                target = np.array([xr[j] + bw / 2, y + level[k] * H / 2, 0])
                moves.append(c.animate.move_to(target))
                low_copies.append(c)
                y += level[k] * H
            self.play(*moves, run_time=1.1 if j < 3 else 1.3)
            check(abs(max(c.get_top()[1] for c in low_copies) - half_y) < 1e-6,
                  f"group {j}: the cut pieces stack to exactly 1/2")
            if j == 1:
                self.play(Create(half), FadeIn(half_lab),
                          FadeIn(col_labs[0]), run_time=0.8)
            top_copies, moves = [], []
            for k in ks:
                if k in tops:
                    c = tops[k].copy()
                    h = (1.0 / k - level[k]) * H
                    moves.append(c.animate.move_to(
                        [xr[j] + bw / 2, y + h / 2, 0]))
                    top_copies.append(c)
                    y += h
            if moves:
                self.play(*moves, FadeIn(col_labs[j - 1]), run_time=1.0)
                total = sum(1.0 / k for k in ks) * H
                check(abs(max(c.get_top()[1] for c in top_copies)
                          - (yb + total)) < 1e-6,
                      f"group {j}: the column is the group's sum")
                check(yb + total > half_y, f"group {j} rises above 1/2")
        self.bring_to_front(half)
        rmore = tag("…", 34).move_to([5.9, half_y - 0.3, 0])
        self.play(FadeIn(rmore), run_time=0.4)

        readout = tag("1 + ½ + ⅓ + … + 1/2ⁿ   ≥   1 + n/2", 26)
        readout.move_to([3.15, 2.85, 0])
        _safe(readout, half_lab, more, rmore, *labs, *glabs, *col_labs)
        _apart(*col_labs)
        _apart(*labs, *glabs)
        check(readout.get_left()[0] > bx(1) + bw + 0.3, "readout clear of the bars")
        self.play(FadeIn(readout, shift=DOWN * 0.15), run_time=0.8)
        self.play(Write(caption(
            "1 + ½ + ⅓ + ¼ + …   ≥   1 + ½ + ½ + ½ + …   =   ∞", 32)))
        self.hold(2.2)


# ====================================================================== B16

class B16_TwoTriangulars(Board):
    """Cut the n × n square of dots along the staircase just above its
    diagonal. The lower piece keeps the diagonal: rows of n, n−1, …, 1
    dots, the triangular number Tₙ. The upper piece has rows of 1, 2, …,
    n−1 dots: Tₙ₋₁. Put back, the two fill the square:
    Tₙ₋₁ + Tₙ = n².  Drawn for n = 6."""

    def construct(self):
        n, s, r = 6, 0.62, 0.11
        O = np.array([-(n - 1) * s / 2, 0.35 - (n - 1) * s / 2, 0.0])

        def P(p):
            return O + s * np.array([p[0], p[1], 0.0])

        low = [(i, j) for i in range(n) for j in range(n) if j <= i]
        up = [(i, j) for i in range(n) for j in range(n) if j > i]
        check(sorted(sum(1 for (i, j) in low if j == row) for row in range(n))
              == list(range(1, n + 1)), "lower piece: rows of 1 .. n dots")
        check(sorted(sum(1 for (i, j) in up if j == row) for row in range(1, n))
              == list(range(1, n)), "upper piece: rows of 1 .. n-1 dots")
        check(len(low) + len(up) == n * n and not set(low) & set(up),
              "T(n) + T(n-1) = n²")

        dots = {p: Dot(P(p), radius=r, color=GREY_A) for p in low + up}
        lowg = VGroup(*[dots[p] for p in low])
        upg = VGroup(*[dots[p] for p in up])
        self.play(LaggedStart(*[FadeIn(dots[(i, j)], scale=0.5)
                                for j in range(n) for i in range(n)],
                              lag_ratio=0.03), run_time=1.6)
        self.hold(0.4)

        # the cut: a staircase between the diagonal and the dots above it
        path = [(-0.75, 0.5)]
        for i in range(n - 1):
            path += [(i + 0.5, i + 0.5), (i + 0.5, i + 1.5)]
        path += [(n - 1.5, n - 0.25)]
        cut = VMobject(stroke_color=YELLOW_B, stroke_width=5)
        cut.set_points_as_corners([P(p) for p in path])
        self.play(Create(cut), run_time=1.2)
        self.play(lowg.animate.set_color(BLUE_C), upg.animate.set_color(ORANGE),
                  run_time=0.8)
        self.hold(0.4)

        # pull the pieces apart, straight across the cut
        v = np.array([0.62, -0.62, 0.0])
        self.play(FadeOut(cut), lowg.animate.shift(v), upg.animate.shift(-v),
                  run_time=1.6)
        check(_same_spots(lowg, [P(p) + v for p in low]) and
              _same_spots(upg, [P(p) - v for p in up]), "pieces moved apart")

        # the lower piece: bottom row n, the upper piece: top row n−1
        yb = P((0, 0))[1] + v[1] - r
        d_low = _dim([P((0, 0))[0] + v[0] - r, yb, 0],
                     [P((n - 1, 0))[0] + v[0] + r, yb, 0], "n",
                     DOWN * 0.24, BLUE_B, 28, gap=0.1)
        yt = P((0, n - 1))[1] - v[1] + r
        d_up = _dim([P((0, n - 1))[0] - v[0] - r, yt, 0],
                    [P((n - 2, n - 1))[0] - v[0] + r, yt, 0], "n−1",
                    UP * 0.24, ORANGE, 28, gap=0.1)
        nm_low = tag("Tₙ", 36, BLUE_B).move_to(
            P((n - 1, 1.5)) + v + RIGHT * 0.75)
        nm_up = tag("Tₙ₋₁", 36, ORANGE).move_to(
            P((0, n - 2.5)) - v + LEFT * 0.95)
        _safe(d_low, d_up, nm_low, nm_up)
        _clear_of(nm_low, lowg)
        _clear_of(nm_up, upg)
        _clear_of(d_low, lowg)
        _clear_of(d_up, upg)
        self.play(FadeIn(d_low), FadeIn(nm_low), run_time=0.8)
        self.play(FadeIn(d_up), FadeIn(nm_up), run_time=0.8)
        self.hold(1.4)

        # and back together: the square
        self.play(FadeOut(d_low), FadeOut(d_up), FadeOut(nm_low),
                  FadeOut(nm_up), run_time=0.5)
        self.play(lowg.animate.shift(-v), upg.animate.shift(v), run_time=1.4)
        check(_same_spots(upg, [P(p) for p in up]) and
              _same_spots(lowg, [P(p) for p in low]),
              "the two staircases fill the n x n square")
        pad = 0.36
        frame = Polygon(P((-pad / s, -pad / s)), P((n - 1 + pad / s, -pad / s)),
                        P((n - 1 + pad / s, n - 1 + pad / s)),
                        P((-pad / s, n - 1 + pad / s)),
                        stroke_color=YELLOW_B, stroke_width=4)
        d_b = _dim(frame.get_corner(DL), frame.get_corner(DR), "n",
                   DOWN * 0.22, YELLOW_B, 28, gap=0.1)
        d_r = _dim(frame.get_corner(DR), frame.get_corner(UR), "n",
                   RIGHT * 0.22, YELLOW_B, 28, gap=0.12)
        _safe(frame, d_b, d_r)
        self.play(Create(frame), FadeIn(d_b), FadeIn(d_r), run_time=1.0)
        self.play(Write(caption("Tₙ₋₁ + Tₙ  =  n²", 36)))
        self.hold(2.2)


# ====================================================================== B15

class B15_SquaresInThePlane(Board):
    """Three copies of the squares 1², 2², …, n². Two of them stand as
    stacks at the two ends of a rectangle 2n+1 wide and 1 + 2 + … + n
    high, the n-square at the bottom. Beside the k-squares the gap between
    the stacks is k rows of width 2(n−k)+1. The third copy is cut into
    unit strips along its L-shaped shells: the k-square's shells have
    1, 3, 5, …, 2k−1 cells, and each shell, cut at its corner, swings
    straight into a row. The gap beside the k-squares takes the shell
    of width 2(n−k)+1 from each of the k squares that have one, so the
    third copy fills the gap exactly:
        3(1² + … + n²) = (2n+1)(1 + 2 + … + n).   Drawn for n = 4."""

    def construct(self):
        n, u = 4, 0.56
        W, H = 2 * n + 1, n * (n + 1) // 2

        def T(k):
            return k * (k + 1) // 2

        Y = {k: H - T(k) for k in range(1, n + 1)}       # bottom of band k
        R0 = np.array([-3.85, -2.35, 0.0])               # rectangle origin

        def RC(x, y):
            return R0 + u * np.array([x + 0.5, y + 0.5, 0.0])

        # ---- the tiling, cell by cell
        A = {k: [(x, y) for x in range(k) for y in range(Y[k], Y[k] + k)]
             for k in range(1, n + 1)}
        B = {k: [(x, y) for x in range(W - k, W) for y in range(Y[k], Y[k] + k)]
             for k in range(1, n + 1)}
        row_of = {}                    # shell m of C-square j -> its row
        for j in range(1, n + 1):
            for m in range(j):
                y = Y[n - m] + j - m - 1
                row_of[(j, m)] = [(x, y) for x in range(n - m, n + m + 1)]
        check(_tiles_once(list(A.values()) + list(B.values())
                          + list(row_of.values()), W, H),
              "A, B and the rows of C tile (2n+1) x T(n) exactly once")
        for j in range(1, n + 1):
            shells = []
            for m in range(j):
                L = [(x, m) for x in range(m + 1)] + [(m, y) for y in range(m)]
                check(len(L) == 2 * m + 1 == len(row_of[(j, m)]),
                      f"shell {m} of {j}² is one row of 2m+1 cells")
                shells.append(L)
            check(_tiles_once(shells, j, j), f"the shells of {j}² tile it")
        check(3 * sum(k * k for k in range(1, n + 1)) == W * H, "3·Σk² = W·H")

        # ---- the third copy, stacked like the first, to the right
        gap = 0.12
        S = {n: np.array([1.75, R0[1], 0.0])}
        for j in range(n - 1, 0, -1):
            S[j] = S[j + 1] + np.array([0.0, (j + 1) * u + gap, 0.0])

        def SC(j, x, y):
            return S[j] + u * np.array([x + 0.5, y + 0.5, 0.0])

        def block(cells, centre_of, color, bold=3.5):
            """Cells of one piece (faint lines) with a bold outline."""
            g = VGroup(*[Square(side_length=u, fill_color=color,
                                fill_opacity=FILL, stroke_color=WHITE,
                                stroke_width=1.0, stroke_opacity=0.5
                                ).move_to(centre_of(*c))
                         for c in cells])
            xs = [centre_of(*c)[0] for c in cells]
            ys = [centre_of(*c)[1] for c in cells]
            box = Rectangle(width=max(xs) - min(xs) + u,
                            height=max(ys) - min(ys) + u,
                            stroke_color=WHITE, stroke_width=bold)
            box.move_to([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0])
            return VGroup(g, box)

        def sq_label(k, centre, color=WHITE):
            return tag(f"{k}²", [0, 18, 24, 30, 34, 36, 36][min(k, 6)],
                       color).move_to(centre)

        outline = DashedVMobject(Rectangle(width=W * u, height=H * u,
                                           stroke_color=GREY_B,
                                           stroke_width=2),
                                 num_dashes=70).move_to(
            R0 + np.array([W * u / 2, H * u / 2, 0]))
        self.play(Create(outline), run_time=1.0)

        sqA = {k: block(A[k], RC, BLUE_D) for k in range(1, n + 1)}
        sqB = {k: block(B[k], RC, TEAL_D) for k in range(1, n + 1)}
        labA = {k: sq_label(k, sqA[k].get_center()) for k in sqA}
        labB = {k: sq_label(k, sqB[k].get_center()) for k in sqB}
        self.play(LaggedStart(*[FadeIn(VGroup(sqA[k], labA[k]),
                                       shift=RIGHT * 0.3)
                                for k in range(n, 0, -1)], lag_ratio=0.3),
                  run_time=1.6)
        self.play(LaggedStart(*[FadeIn(VGroup(sqB[k], labB[k]),
                                       shift=LEFT * 0.3)
                                for k in range(n, 0, -1)], lag_ratio=0.3),
                  run_time=1.6)

        # the third copy, whole first
        whole = {j: block([(x, y) for x in range(j) for y in range(j)],
                          lambda x, y, j=j: SC(j, x, y), ORANGE)
                 for j in range(1, n + 1)}
        labC = {j: sq_label(j, whole[j].get_center()) for j in whole}
        self.play(LaggedStart(*[FadeIn(VGroup(whole[j], labC[j]))
                                for j in range(n, 0, -1)], lag_ratio=0.25),
                  run_time=1.3)
        self.hold(0.4)

        # cut into L-shaped unit strips, each cut again at its corner
        tops, arms = {}, {}
        for j in range(1, n + 1):
            for m in range(j):
                tops[(j, m)] = block([(x, m) for x in range(m + 1)],
                                     lambda x, y, j=j: SC(j, x, y), ORANGE)
                if m:
                    arms[(j, m)] = block([(m, y) for y in range(m)],
                                         lambda x, y, j=j: SC(j, x, y), ORANGE)
        self.play(*[FadeOut(labC[j]) for j in labC], run_time=0.4)
        self.remove(*whole.values())
        self.add(*tops.values(), *arms.values())
        shell_col = {m: (ORANGE if m % 2 == 0 else YELLOW_E) for m in range(n)}
        self.play(*[p[0].animate.set_fill(shell_col[key[1]])
                    for key, p in list(tops.items()) + list(arms.items())],
                  run_time=1.0)
        self.hold(0.5)

        # every L swings straight: its arm turns a quarter about the corner
        self.play(*[Rotate(arms[key], PI / 2, about_point=SC(key[0], key[1],
                                                             key[1]))
                    for key in arms], run_time=1.6)
        for (j, m), arm in arms.items():
            check(_same_spots(arm[0], [SC(j, x, m) for x in range(m + 1, 2 * m + 1)]),
                  f"shell {m} of {j}² lies straight: 2m+1 cells in a row")
        self.hold(0.4)

        # band by band, the straightened shells drop into the gap
        for k in range(n, 0, -1):
            m = n - k
            moves = []
            for j in range(m + 1, n + 1):
                piece = VGroup(tops[(j, m)], *([arms[(j, m)]] if m else []))
                x0, y0 = row_of[(j, m)][0]
                d = RC(x0, y0) - SC(j, 0, m)
                self.bring_to_front(piece)
                moves.append(piece.animate.shift(d))
            self.play(*moves, run_time=1.1)
            for j in range(m + 1, n + 1):
                cells = list(tops[(j, m)][0]) + (list(arms[(j, m)][0]) if m else [])
                check(_same_spots(cells, [RC(x, y) for (x, y) in row_of[(j, m)]]),
                      f"shell {m} of {j}² lands on its row")
        self.hold(0.4)

        frame = Rectangle(width=W * u, height=H * u, stroke_color=YELLOW_B,
                          stroke_width=5).move_to(outline)
        d_w = _dim(R0, R0 + RIGHT * W * u, "2n+1", DOWN * 0.2, YELLOW_B, 26,
                   gap=0.08)
        d_h = _dim(R0, R0 + UP * H * u, "1 + 2 + … + n", LEFT * 0.2,
                   YELLOW_B, 24, gap=0.12)
        _safe(d_w, d_h, frame)
        self.remove(outline)
        self.play(Create(frame), FadeIn(d_w), FadeIn(d_h), run_time=1.0)
        self.play(Write(caption(
            "3 · (1² + 2² + … + n²)  =  (2n+1) · (1 + 2 + … + n)", 32)))
        self.hold(2.2)


# ====================================================================== B17

class B17_EightTriangulars(Board):
    """Two staircases Tₙ, one turned half a turn, close into an n × (n+1)
    rectangle. Four such rectangles, each a quarter-turn of the last about
    the centre of the (2n+1)-square, fill that square except for the one
    cell in the middle: the sides n and n+1 overlap by exactly one.
        8 · Tₙ + 1 = (2n+1)².   Drawn for n = 4."""

    def construct(self):
        n, u = 4, 0.6
        N = 2 * n + 1
        Q0 = np.array([-N * u / 2 - 0.4, -2.33, 0.0])
        C = Q0 + np.array([N * u / 2, N * u / 2, 0.0])     # the square's centre

        def QC(x, y):
            return Q0 + u * np.array([x + 0.5, y + 0.5, 0.0])

        def rot(c, k):            # quarter-turns (CCW) of a cell about the centre
            x, y = c
            for _ in range(k):
                x, y = N - 1 - y, x
            return (x, y)

        def rotp(p, k):           # the same for a point in cell-corner units
            x, y = p
            for _ in range(k):
                x, y = N - y, x
            return (x, y)

        S1 = [(x, r) for r in range(n) for x in range(r + 1)]
        S2 = [(n - x, n - 1 - r) for (x, r) in S1]           # half-turn in R1
        R1 = [(x, y) for x in range(n + 1) for y in range(n)]
        check(sorted(S1 + S2) == sorted(R1), "two staircases make n x (n+1)")
        pieces = [[rot(c, k) for c in S] for k in range(4) for S in (S1, S2)]
        check(_tiles_once(pieces + [[(n, n)]], N, N),
              "eight staircases and one cell tile the (2n+1)-square")
        check(8 * n * (n + 1) // 2 + 1 == N * N, "8 T(n) + 1 = (2n+1)²")

        # the outline of S1 in cell-corner units, and of its half-turn twin
        rim1 = [(0, 0)]
        for r in range(n):
            rim1 += [(r + 1, r), (r + 1, r + 1)]
        rim1 += [(0, n)]
        rim2 = [(n + 1 - x, n - y) for (x, y) in rim1]

        def stair(cells, rim, color):
            g = VGroup(*[Square(side_length=u, fill_color=color,
                                fill_opacity=FILL, stroke_color=WHITE,
                                stroke_width=1.0, stroke_opacity=0.45
                                ).move_to(QC(x, y)) for (x, y) in cells])
            out = Polygon(*[Q0 + u * np.array([x, y, 0.0]) for (x, y) in rim],
                          stroke_color=WHITE, stroke_width=4)
            return VGroup(g, out)

        ghost = DashedVMobject(Square(side_length=N * u, stroke_color=GREY_B,
                                      stroke_width=2), num_dashes=80).move_to(C)
        self.play(Create(ghost), run_time=0.8)

        s1 = stair(S1, rim1, BLUE_D)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in s1[0]],
                              lag_ratio=0.06), run_time=1.3)
        self.play(Create(s1[1]), run_time=0.5)
        # each name sits in the cell at its staircase's centroid
        cen = tuple(np.mean([(x + 0.5, y + 0.5) for (x, y) in S1], axis=0))
        check(close(cen, (1.5, 2.5)) and (1, 2) in S1, "centroid cell of S1")
        lab_pts = [cen, (n + 1 - cen[0], n - cen[1])]
        check((round(lab_pts[1][0] - 0.5), round(lab_pts[1][1] - 0.5)) in S2,
              "centroid cell of S2")

        def lab_at(p, k):
            q = rotp(p, k)
            return tag("Tₙ", 24).move_to(Q0 + u * np.array([q[0], q[1], 0.0]))

        l1 = lab_at(lab_pts[0], 0)
        self.play(FadeIn(l1), run_time=0.4)

        # its twin, turned half a turn about the centre of the n × (n+1) box
        r1c = Q0 + u * np.array([(n + 1) / 2, n / 2, 0.0])
        s2 = s1.copy()
        s2[0].set_fill(TEAL_D)
        self.add(s2)
        self.play(Rotate(s2, PI, about_point=r1c), run_time=1.6)
        check(_same_spots(s2[0], [QC(*c) for c in S2]), "the twin lands in place")
        l2 = lab_at(lab_pts[1], 0)
        box = Rectangle(width=(n + 1) * u, height=n * u, stroke_color=YELLOW_B,
                        stroke_width=5).move_to(r1c)
        d_w = _dim(Q0, Q0 + RIGHT * (n + 1) * u, "n+1", DOWN * 0.2, YELLOW_B,
                   26, gap=0.08)
        d_h = _dim(Q0, Q0 + UP * n * u, "n", LEFT * 0.2, YELLOW_B, 26, gap=0.12)
        self.play(FadeIn(l2), Create(box), FadeIn(d_w), FadeIn(d_h),
                  run_time=0.9)
        self.hold(0.8)
        self.play(FadeOut(box), FadeOut(d_w), FadeOut(d_h), run_time=0.5)

        # three quarter-turns of the pair about the centre of the big square
        pair = VGroup(s1, s2)
        labels = [l1, l2]
        for k in range(1, 4):
            nxt = pair.copy()
            self.add(nxt)
            self.play(Rotate(nxt, PI / 2, about_point=C), run_time=1.3)
            check(_same_spots(nxt[0][0], [QC(*rot(c, k)) for c in S1]) and
                  _same_spots(nxt[1][0], [QC(*rot(c, k)) for c in S2]),
                  f"quarter-turn {k} lands on its staircases")
            new_labs = [lab_at(lab_pts[0], k), lab_at(lab_pts[1], k)]
            self.play(*[FadeIn(m) for m in new_labs], run_time=0.35)
            labels += new_labs
            pair = nxt

        # one cell is left in the middle
        mid = cell(n, n, u, Q0, YELLOW_E, op=0.95, stroke=2)
        one = tag("1", 24, BLACK).move_to(mid)
        self.play(FadeIn(mid, scale=0.3), FadeIn(one), run_time=0.7)
        self.play(Flash(mid.get_center(), color=YELLOW_B, line_length=0.25,
                        flash_radius=0.45), run_time=0.6)

        frame = Square(side_length=N * u, stroke_color=YELLOW_B,
                       stroke_width=5).move_to(C)
        D_w = _dim(Q0, Q0 + RIGHT * N * u, "2n+1", DOWN * 0.2, YELLOW_B, 28,
                   gap=0.08)
        D_h = _dim(Q0 + RIGHT * N * u, Q0 + RIGHT * N * u + UP * N * u, "2n+1",
                   RIGHT * 0.2, YELLOW_B, 28, gap=0.12)
        _safe(D_w, D_h, frame, *labels)
        self.remove(ghost)
        self.play(Create(frame), FadeIn(D_w), FadeIn(D_h), run_time=1.0)
        self.play(Write(caption("8 · Tₙ + 1  =  (2n+1)²", 36)))
        self.hold(2.2)


# ====================================================================== B18

class B18_UpAndDown(Board):
    """Rows of 1, 2, …, n, …, 2, 1 dots, centred one under another, stand
    on a square lattice turned 45°: each row is a diagonal of the n × n
    square of dots. Turned back, the diamond is that square:
        1 + 2 + … + (n−1) + n + (n−1) + … + 2 + 1 = n².   Drawn for n = 5."""

    def construct(self):
        n, s, r = 5, 0.74, 0.12
        Cd = np.array([-2.35, 0.4, 0.0])           # the diamond, left
        Cs = np.array([3.05, 0.4, 0.0])            # the square, right
        c = (n - 1) / 2

        def sq_pt(i, j, centre=Cd):            # the upright n x n lattice
            return centre + s * np.array([i - c, j - c, 0.0])

        def tilt(p):                          # turned by +45° about Cd
            v = to3(p) - Cd
            a = PI / 4
            return Cd + np.array([np.cos(a) * v[0] - np.sin(a) * v[1],
                                  np.sin(a) * v[0] + np.cos(a) * v[1], 0.0])

        pts = [(i, j) for i in range(n) for j in range(n)]
        rows = {q: [p for p in pts if p[0] + p[1] == q] for q in range(2 * n - 1)}
        counts = [len(rows[q]) for q in range(2 * n - 1)]
        check(counts == list(range(1, n + 1)) + list(range(n - 1, 0, -1)),
              "the diagonals hold 1, 2, …, n, …, 2, 1 dots")
        check(sum(counts) == n * n, "1 + … + n + … + 1 = n²")
        for q in range(2 * n - 1):
            ys = {round(float(tilt(sq_pt(*p))[1]), 9) for p in rows[q]}
            check(len(ys) == 1, f"diagonal {q} is one horizontal row when turned")

        shade = {1: BLUE_C, 2: TEAL_C, 3: GREEN_C, 4: YELLOW_C, 5: ORANGE,
                 6: RED_C, 7: PURPLE_B}
        dot = {p: Dot(tilt(sq_pt(*p)), radius=r,
                      color=shade[len(rows[p[0] + p[1]])]) for p in pts}
        allg = VGroup(*[dot[p] for p in pts])

        x_lab = Cd[0] - (n - 1) * s / np.sqrt(2) - 0.85
        labs = VGroup()
        anims = []
        for q in range(2 * n - 2, -1, -1):                # top row first
            y = tilt(sq_pt(*rows[q][0]))[1]
            lab = tag(str(len(rows[q])), 28, shade[len(rows[q])]).move_to(
                [x_lab, y, 0])
            labs.add(lab)
            anims.append(AnimationGroup(
                *[FadeIn(dot[p], scale=0.5) for p in rows[q]], FadeIn(lab)))
        self.play(LaggedStart(*anims, lag_ratio=0.35), run_time=3.0)
        self.hold(0.6)

        # the tilted square around the diamond
        pad = 0.38
        corners = [(-pad / s, -pad / s), (n - 1 + pad / s, -pad / s),
                   (n - 1 + pad / s, n - 1 + pad / s), (-pad / s, n - 1 + pad / s)]
        rim = Polygon(*[tilt(sq_pt(*q)) for q in corners],
                      stroke_color=YELLOW_B, stroke_width=4)
        _safe(labs, allg, rim)
        _apart(labs, rim)
        self.play(Create(rim), run_time=1.0)
        self.hold(0.5)

        # a copy, turned back by 45°, set beside it
        turned = VGroup(allg, rim).copy()
        self.add(turned)
        self.play(_rigid(turned, angle=-PI / 4, about=Cd, shift=Cs - Cd),
                  run_time=2.4)
        check(_same_spots(turned[0], [sq_pt(*p, centre=Cs) for p in pts]),
              "turned back, the dots stand on the n x n square lattice")
        d_b = _dim(sq_pt(*corners[0], centre=Cs), sq_pt(*corners[1], centre=Cs),
                   "n", DOWN * 0.22, YELLOW_B, 30, gap=0.1)
        d_r = _dim(sq_pt(*corners[1], centre=Cs), sq_pt(*corners[2], centre=Cs),
                   "n", RIGHT * 0.22, YELLOW_B, 30, gap=0.12)
        eq = tag("=", 44, YELLOW_B).move_to((rim.get_right() + turned[1].get_left()) / 2)
        _safe(d_b, d_r, turned)
        _apart(eq, rim, gap=0.1)
        _apart(eq, turned[1], gap=0.1)
        self.play(FadeIn(d_b), FadeIn(d_r), FadeIn(eq), run_time=0.8)
        self.play(Write(caption(
            "1 + 2 + … + (n−1) + n + (n−1) + … + 2 + 1  =  n²", 32)))
        self.hold(2.2)


# ====================================================================== B20

class B20_PowersOfTwo(Board):
    """Start from one missing cell. Each new block is a copy of everything
    so far with the missing cell filled in: 1, then 2, 4, 8, … cells, each
    laid beside the figure, turn by turn to the right and on top. So all the
    blocks together are always the next power of two short of one cell:
        1 + 2 + 4 + … + 2^(n−1) = 2ⁿ − 1.   Drawn for n = 6 (an 8 × 8 board)."""

    def construct(self):
        n, u = 6, 0.6
        G0 = np.array([-5.7, -2.42, 0.0])
        cols = [BLUE_D, TEAL_D, ORANGE, YELLOW_E, GREEN_D, RED_D]

        def GC(x, y):
            return G0 + u * np.array([x + 0.5, y + 0.5, 0.0])

        # region after k blocks = missing cell + blocks 0..k-1 = 2^k cells
        region = [(0, 0)]
        blocks, shifts = [], []
        for k in range(n):
            w = max(x for x, _ in region) + 1
            h = max(y for _, y in region) + 1
            d = (w, 0) if k % 2 == 0 else (0, h)          # right, then on top
            blk = [(x + d[0], y + d[1]) for (x, y) in region]
            check(not set(blk) & set(region), f"block {k} lands beside the rest")
            check(len(blk) == 2 ** k, f"block {k} has 2^{k} cells")
            blocks.append(blk)
            shifts.append(d)
            region = region + blk
        check(_tiles_once([[(0, 0)]] + blocks, 8, 8),
              "the missing cell and the blocks tile the 8 x 8 board")
        check(sum(2 ** k for k in range(n)) == 2 ** n - 1, "1 + 2 + … = 2^n - 1")

        def piece(cells, color, op=FILL):
            g = VGroup(*[Square(side_length=u, fill_color=color,
                                fill_opacity=op, stroke_color=WHITE,
                                stroke_width=1.0, stroke_opacity=0.5
                                ).move_to(GC(x, y)) for (x, y) in cells])
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            box = Rectangle(width=(max(xs) - min(xs) + 1) * u,
                            height=(max(ys) - min(ys) + 1) * u,
                            stroke_color=WHITE, stroke_width=3.5)
            box.move_to(G0 + u * np.array([(max(xs) + min(xs) + 1) / 2,
                                           (max(ys) + min(ys) + 1) / 2, 0]))
            return VGroup(g, box)

        hole = DashedVMobject(Square(side_length=u * 0.94,
                                     stroke_color=YELLOW_B, stroke_width=5),
                              num_dashes=12).move_to(GC(0, 0))
        self.play(Create(hole), run_time=0.8)

        lines = ["1  =  2 − 1", "1 + 2  =  4 − 1", "1 + 2 + 4  =  8 − 1",
                 "1 + 2 + 4 + 8  =  16 − 1",
                 "1 + 2 + 4 + 8 + 16  =  32 − 1",
                 "1 + 2 + 4 + 8 + 16 + 32  =  64 − 1"]
        rows = VGroup(*[tag(t, 26) for t in lines])
        for i, t in enumerate(rows):
            t.move_to([0.0 + t.width / 2, 2.75 - i * 0.62, 0])
        _safe(rows)

        placed = []
        for k in range(n):
            src = [(0, 0)] + [c for b in blocks[:k] for c in b]
            ghost = piece(src, cols[k], 0.45)
            self.play(FadeIn(ghost), run_time=0.45 if k < 4 else 0.35)
            d = u * np.array([shifts[k][0], shifts[k][1], 0.0])
            self.play(ghost[0].animate.shift(d).set_fill(opacity=FILL),
                      ghost[1].animate.shift(d),
                      run_time=1.0 if k < 4 else 0.9)
            check(_same_spots(ghost[0], [GC(*c) for c in blocks[k]]),
                  f"block {k} landed")
            num = tag(str(2 ** k), [22, 24, 28, 32, 36, 40][k],
                      BLACK if cols[k] == YELLOW_E else WHITE).move_to(ghost[1])
            self.play(FadeIn(num), FadeIn(rows[k], shift=LEFT * 0.2),
                      run_time=0.5)
            placed.append(VGroup(ghost, num))
        self.bring_to_front(hole)
        self.hold(0.5)

        frame = Square(side_length=8 * u, stroke_color=YELLOW_B,
                       stroke_width=5).move_to(G0 + 4 * u * np.array([1, 1, 0]))
        top = tag("2ⁿ", 30, YELLOW_B).next_to(frame, UP, buff=0.12)
        _safe(frame, top)
        self.play(Create(frame), FadeIn(top), run_time=0.9)
        self.play(Indicate(hole, color=YELLOW_B, scale_factor=1.4), run_time=0.9)
        cap = _cap(_sup_line("1 + 2 + 4 + … + 2", "n−1", "=  2ⁿ − 1", 34))
        self.play(Write(cap))
        self.hold(2.2)


# ====================================================================== B23

class B23_ArithmeticSeries(Board):
    """Bars of heights a, a+d, …, a+(n−1)d, each of width 1, stand side by
    side. The same row of bars turned through a half-turn drops onto them:
    over the bar a+kd stands the bar a+(n−1−k)d, so every column is
    2a+(n−1)d high and the two rows make an n × (2a+(n−1)d) rectangle:
        a + (a+d) + … + (a+(n−1)d) = n(2a+(n−1)d)/2.
    Drawn with n = 6 and d about half of a."""

    def construct(self):
        n, a, d = 6, 1.2, 0.55
        Ht = 2 * a + (n - 1) * d
        # scale and place so that the half-turn's sweep stays on screen
        k, yc = 1.0, 0.3
        x0, y0 = -3.65, yc - Ht * k / 2

        class _F:
            @staticmethod
            def P(q):
                return np.array([x0 + q[0] * k, y0 + q[1] * k, 0.0])

            @staticmethod
            def poly(pts, color, op=FILL, **kw):
                return mk([_F.P(q) for q in pts], color, op, **kw)

        F, P = _F, _F.P
        h = [a + i * d for i in range(n)]
        reach = (n + h[-1]) / (2 * np.sqrt(2)) * k    # widest extent mid-turn
        check(yc + reach < 4.0 and yc - reach > -4.0,
              "the turning copy stays inside the frame")
        for i in range(n):
            check(abs(h[i] + h[n - 1 - i] - Ht) < 1e-12,
                  f"column {i}: a+kd and a+(n-1-k)d make 2a+(n-1)d")
        check(abs(2 * sum(h) - n * Ht) < 1e-12, "two rows = n(2a+(n-1)d)")

        bars = VGroup(*[F.poly(_rect(i, 0, 1, h[i]), BLUE_D) for i in range(n)])
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars],
                              lag_ratio=0.3), run_time=2.0)

        d_a = _dim(P((0, 0)), P((0, a)), "a", LEFT * 0.2, BLUE_B, 28, gap=0.12)
        d_d = _dim(P((1, a)), P((1, a + d)), "d", LEFT * 0.16, GREY_A, 26,
                   gap=0.1)
        d_l = _dim(P((n, 0)), P((n, h[-1])), "a + (n−1)d", RIGHT * 0.2,
                   BLUE_B, 28, gap=0.12)
        step = DashedLine(P((0, a)), P((1, a)), color=GREY_A, stroke_width=2,
                          dash_length=0.06)
        _safe(d_a, d_d, d_l)
        _apart(d_d[1], bars[0], gap=0.05)
        self.play(FadeIn(d_a), FadeIn(d_l), run_time=0.7)
        self.play(Create(step), FadeIn(d_d), run_time=0.7)
        self.hold(0.8)

        # the same bars, turned over, drop onto them
        ctr = P((n / 2, Ht / 2))
        turned = bars.copy().set_fill(TEAL_D)
        self.play(FadeOut(d_d), FadeOut(step), FadeOut(d_l), run_time=0.4)
        self.add(turned)
        self.play(Rotate(turned, PI, about_point=ctr), run_time=2.4)
        for i in range(n):
            j = n - 1 - i                       # the bar that lands on bar i
            check(close(turned[j].get_corner(DL), P((i, h[i]))) and
                  close(turned[j].get_corner(UR), P((i + 1, Ht))),
                  f"turned bar {j} stands exactly on bar {i}")
        self.hold(0.4)

        d_top = _dim(P((n, h[-1])), P((n, Ht)), "a", RIGHT * 0.2, TEAL_B, 28,
                     gap=0.12)
        d_n = _dim(P((0, 0)), P((n, 0)), "n", DOWN * 0.22, YELLOW_B, 30,
                   gap=0.1)
        frame = Polygon(*[P(q) for q in _rect(0, 0, n, Ht)],
                        stroke_color=YELLOW_B, stroke_width=5)
        _safe(d_top, d_n)
        _apart(d_top[1], d_l[1])
        self.play(FadeIn(d_l), FadeIn(d_top), run_time=0.7)
        self.play(Create(frame), FadeIn(d_n), run_time=0.9)
        self.play(Write(caption(
            "a + (a+d) + … + (a+(n−1)d)  =  n · (2a + (n−1)d) / 2", 32)))
        self.hold(2.2)


# ====================================================================== B14

class B14_EvenNumbers(Board):
    """Each even number 2k is a column two cells wide and k high. The
    columns 2, 4, …, 2n stand side by side as a staircase. Cut it down the
    middle and give the taller half a half-turn: it drops onto the shorter
    half, the column 2(n+1−k) onto the column 2k, and every double column is
    n+1 high. The halves close into an n × (n+1) rectangle:
        2 + 4 + … + 2n = n(n+1).
    Drawn for n = 6. (For odd n the cut runs down the middle column, which
    is split into two columns of width 1; the half-turn works the same.)"""

    def construct(self):
        n, u = 6, 0.5
        hcut = n // 2                                   # columns per half
        G0 = np.array([-n * u, 0.25 - (n + 1) * u / 2, 0.0])

        def GC(x, y):
            return G0 + u * np.array([x + 0.5, y + 0.5, 0.0])

        cols_cells = {k: [(x, y) for x in (2 * k - 2, 2 * k - 1) for y in range(k)]
                      for k in range(1, n + 1)}
        pivot = (n, (n + 1) / 2)                        # cell-corner units

        def half_turn(c):
            x, y = c
            return (round(2 * pivot[0] - x - 1), round(2 * pivot[1] - y - 1))

        moved = {k: [half_turn(c) for c in cols_cells[k]]
                 for k in range(hcut + 1, n + 1)}
        check(_tiles_once([cols_cells[k] for k in range(1, hcut + 1)]
                          + list(moved.values()), n, n + 1),
              "the two halves close into n x (n+1)")
        for k in range(hcut + 1, n + 1):
            xs = {x for x, _ in moved[k]}
            partner = n + 1 - k
            check(xs == {2 * partner - 2, 2 * partner - 1} and
                  min(y for _, y in moved[k]) == partner,
                  f"column 2·{k} lands on column 2·{partner}")
        check(sum(2 * k for k in range(1, n + 1)) == n * (n + 1), "2+4+…+2n")

        def column(k, color):
            g = VGroup(*[Square(side_length=u, fill_color=color,
                                fill_opacity=FILL, stroke_color=WHITE,
                                stroke_width=1.0, stroke_opacity=0.5
                                ).move_to(GC(*c)) for c in cols_cells[k]])
            box = Rectangle(width=2 * u, height=k * u, stroke_color=WHITE,
                            stroke_width=3.5).move_to(
                G0 + u * np.array([2 * k - 1, k / 2, 0.0]))
            return VGroup(g, box)

        cols = {k: column(k, BLUE_D) for k in range(1, n + 1)}
        ylab = G0[1] - 0.3
        labs = {k: tag(str(2 * k), 26, BLUE_B).move_to(
            [G0[0] + (2 * k - 1) * u, ylab, 0]) for k in range(1, n + 1)}
        self.play(LaggedStart(*[AnimationGroup(FadeIn(cols[k], shift=UP * 0.15),
                                               FadeIn(labs[k]))
                                for k in range(1, n + 1)], lag_ratio=0.3),
                  run_time=2.4)
        self.hold(0.6)

        # cut down the middle
        cut = DashedLine(G0 + u * np.array([n, -0.2, 0]),
                         G0 + u * np.array([n, n + 0.6, 0]), color=YELLOW_B,
                         stroke_width=4, dash_length=0.1)
        right = VGroup(*[cols[k] for k in range(hcut + 1, n + 1)])
        self.play(Create(cut), run_time=0.8)
        self.play(*[cols[k][0].animate.set_fill(TEAL_D)
                    for k in range(hcut + 1, n + 1)],
                  *[labs[k].animate.set_color(TEAL_B)
                    for k in range(hcut + 1, n + 1)], run_time=0.6)
        self.hold(0.4)

        # the taller half, turned half a turn, drops onto the shorter one
        P0 = G0 + u * np.array([pivot[0], pivot[1], 0.0])
        self.play(FadeOut(cut), *[FadeOut(labs[k])
                                  for k in range(hcut + 1, n + 1)],
                  run_time=0.4)
        self.play(Rotate(right, PI, about_point=P0), run_time=2.2)
        for k in range(hcut + 1, n + 1):
            check(_same_spots(cols[k][0], [GC(*c) for c in moved[k]]),
                  f"column 2·{k} landed")
        ytop = G0[1] + (n + 1) * u + 0.3
        tops = {k: tag(str(2 * k), 26, TEAL_B).move_to(
            [G0[0] + (2 * (n + 1 - k) - 1) * u, ytop, 0])
            for k in range(hcut + 1, n + 1)}
        self.play(*[FadeIn(t) for t in tops.values()], run_time=0.6)
        self.hold(0.4)

        # centre the rectangle, then measure it
        whole = VGroup(*cols.values(), *[labs[k] for k in range(1, hcut + 1)],
                       *tops.values())
        dx = RIGHT * (n * u / 2)
        self.play(whole.animate.shift(dx), run_time=0.9)
        R0 = G0 + dx
        frame = Rectangle(width=n * u, height=(n + 1) * u, stroke_color=YELLOW_B,
                          stroke_width=5).move_to(
            R0 + np.array([n * u / 2, (n + 1) * u / 2, 0]))
        d_w = _dim(R0 + DOWN * 0.58, R0 + RIGHT * n * u + DOWN * 0.58, "n",
                   DOWN * 0.0001, YELLOW_B, 30, gap=0.12)
        d_h = _dim(R0 + RIGHT * n * u, R0 + RIGHT * n * u + UP * (n + 1) * u,
                   "n+1", RIGHT * 0.25, YELLOW_B, 30, gap=0.12)
        _safe(d_w, d_h, *tops.values(), *labs.values())
        _apart(d_w, *[labs[k] for k in range(1, hcut + 1)])
        self.play(Create(frame), FadeIn(d_w), FadeIn(d_h), run_time=1.0)
        self.play(Write(caption("2 + 4 + … + 2n  =  n(n+1)", 36)))
        self.hold(2.2)


# ====================================================================== B49

def _incentre(tri):
    A, B, C = [np.asarray(p, float) for p in tri]
    a, b, c = (np.linalg.norm(B - C), np.linalg.norm(C - A),
               np.linalg.norm(A - B))
    return (a * A + b * B + c * C) / (a + b + c)


def _inside_tri(p, tri, margin=0.0):
    """p strictly inside the triangle, at least `margin` from every side."""
    A, B, C = [np.asarray(q, float)[:2] for q in tri]
    p = np.asarray(p, float)[:2]
    s = np.sign((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
    for U, V in ((A, B), (B, C), (C, A)):
        e = V - U
        cr = s * (e[0] * (p[1] - U[1]) - e[1] * (p[0] - U[0]))
        if cr / np.linalg.norm(e) < margin:
            return False
    return True


class B49_HalvingTriangle(Board):
    """A right isosceles triangle is halved by the altitude from its right
    angle: folded along it, one half falls exactly on the other, and each
    half is again a right isosceles triangle. Keep one half (½), halve the
    other the same way (¼), and so on. The pieces ½, ¼, ⅛, … crowd into
    the corner and leave nothing of the triangle uncovered:
        ½ + ¼ + ⅛ + … = 1."""

    def construct(self):
        F = Frame(-4.3, 4.3, -0.42, 4.35)
        P_ = F.P
        R, P, Q = np.array([0.0, 4.0]), np.array([-4.0, 0.0]), np.array([4.0, 0.0])
        total = abs(area([R, P, Q]))
        outline = Polygon(P_(R), P_(P), P_(Q), stroke_color=YELLOW_B,
                          stroke_width=5)
        self.play(Create(outline), run_time=1.0)

        cols = [BLUE_D, TEAL_D, ORANGE, YELLOW_E, GREEN_D, RED_D, PURPLE_B,
                MAROON_B, GOLD_D, BLUE_D, TEAL_D, ORANGE]
        names = ["½", "¼", "⅛", "1/16", "1/32"]
        sizes = [48, 42, 32, 24, 18]
        steps = 12
        kept = []
        for s in range(steps):
            M = (P + Q) / 2
            K, rest = [Q, M, R], [P, M, R]
            check(abs(np.dot(R - M, Q - P)) < 1e-12, "RM is the altitude")
            check(close(np.linalg.norm(M - P), np.linalg.norm(M - R)) and
                  close(np.linalg.norm(M - Q), np.linalg.norm(M - R)),
                  "both halves are right isosceles")
            check(abs(abs(area(K)) - total / 2 ** (s + 1)) < 1e-12,
                  f"piece {s} is 1/2^{s + 1} of the triangle")
            fast = s >= 5
            alt = Line(P_(R), P_(M), color=WHITE, stroke_width=3 if s < 6 else 2)
            self.play(Create(alt), run_time=0.6 if s == 0 else
                      (0.45 if not fast else 0.25))
            piece = F.poly(K, cols[s], FILL, stroke_width=2 if s < 7 else 1)
            if s == 0:
                # fold the other half over the altitude: it covers this one
                flap = F.poly(rest, GREY_B, 0.45, stroke_color=WHITE,
                              stroke_width=2)
                self.add(flap)
                axis = to3(R - M) / np.linalg.norm(R - M)
                self.play(Rotate(flap, PI, axis=axis, about_point=P_(M)),
                          run_time=1.8)
                got = sorted(tuple(np.round(v[:2], 6)) for v in flap.get_vertices())
                want = sorted(tuple(np.round(P_(v)[:2], 6)) for v in K)
                check(got == want, "folded, the halves coincide")
                self.play(FadeIn(piece), FadeOut(flap), run_time=0.7)
            else:
                self.play(FadeIn(piece), run_time=0.55 if not fast else 0.3)
            if s < len(names):
                lab = tag(names[s], sizes[s]).move_to(P_(_incentre(K)))
                corners = [lab.get_corner(c) for c in (UL, UR, DL, DR)]
                check(all(_inside_tri(c, [P_(v) for v in K], 0.03)
                          for c in corners), f"label {names[s]} inside its piece")
                self.play(FadeIn(lab), run_time=0.35)
            kept.append(K)
            R, P, Q = M, P, R
        check(abs(sum(abs(area(t)) for t in kept) + abs(area([R, P, Q]))
                  - total) < 1e-12, "the pieces and the last remnant make 1")
        check(abs(area([R, P, Q])) / total < 1e-3, "the remnant shrinks to nothing")
        self.bring_to_front(outline)
        self.play(Indicate(outline, color=YELLOW_B, scale_factor=1.02),
                  run_time=0.8)
        self.play(Write(caption("½ + ¼ + ⅛ + …  =  1", 38)))
        self.hold(2.2)
