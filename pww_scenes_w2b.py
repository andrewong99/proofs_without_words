# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2b.py — proofs without words, 2D (manim):
#
#     B44 odd numbers in a pyramid          B27 pentagonal numbers
#     B33 Galileo's ratio                   B30 adding triangular numbers
#     B28 hexagonal numbers are triangular  C9  (a+b)² + (a−b)² = 2(a² + b²)
#     C15 Babylonian multiplication         C20 a² − b² by two trapezoids
#     J31 parity by pairing dots            J10 endless Pythagorean triples
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# letter subscripts or superscripts are built from smaller Text pieces.
# Dot and cell pictures are drawn for one concrete n (or k); the caption
# states the general result. Every piece moves rigidly (a slide, a turn
# in the plane, or a turn-over in space about an in-plane line), and every
# tiling, count and landing is checked with check(...) before or right
# after it is drawn, so a wrong construction fails the render.


# ---------------------------------------------------------------- helpers

def _safe(*mobs, tol=0.03):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for i, m in enumerate(mobs):
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area (item {i}): [{lo[0]:.2f},{hi[0]:.2f}] x "
              f"[{lo[1]:.2f},{hi[1]:.2f}]")


def _boxes_apart(a, b, gap=0.04):
    a0, a1 = a.get_critical_point(DL), a.get_critical_point(UR)
    b0, b1 = b.get_critical_point(DL), b.get_critical_point(UR)
    return (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
            or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])


def _apart(*mobs, gap=0.04):
    """Fail the render if any two labels' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            check(_boxes_apart(mobs[i], mobs[j], gap),
                  f"labels {i} and {j} do not overlap")


def _clear_of(label, group, gap=0.05):
    """Fail the render if the label's box touches any member of `group`."""
    for k, m in enumerate(group):
        check(_boxes_apart(label, m, gap), f"label clear of item {k}")


def _lands_on(group, spots, tol=1e-5):
    """Every member's centre sits on one of `spots` (screen points) and
    every spot is taken exactly once: the motion landed exactly."""
    left = [to3(s) for s in spots]
    mems = list(group)
    if len(mems) != len(left):
        return False
    for m in mems:
        c = m.get_center()
        hit = [i for i, s in enumerate(left) if np.linalg.norm(c - s) < tol]
        if len(hit) != 1:
            return False
        left.pop(hit[0])
    return True


def _tiles_once(pieces, cells):
    """True if the cell sets in `pieces` cover the cell set `cells` exactly
    once (no cell missed, none covered twice, nothing outside)."""
    seen = {}
    for p in pieces:
        for c in p:
            seen[c] = seen.get(c, 0) + 1
    return set(seen) == set(cells) and all(v == 1 for v in seen.values())


def _glide(mob, pivot, target, angle, **kw):
    """A rigid motion that stays rigid on every frame: the piece turns by
    `angle` about its pivot while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _turn_over(mob, p, q, **kw):
    """Turn mob over about the screen line pq: a half-turn in space about
    that line, rigid on every frame, landing as the mirror image in pq."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=(q - p) / np.linalg.norm(q - p),
                  about_point=p, **kw)


def _dim(p, q, label, off, color=GREY_A, size=26, gap=0.12, width=2.5):
    """Dimension line beside the screen segment pq, moved by the vector
    `off`: a thin line with end ticks, and the label (a string, or a ready
    mobject) beyond it. Returns VGroup(lines, label)."""
    p, q, off = to3(p), to3(q), to3(off)
    n = off / np.linalg.norm(off)
    a, b = p + off, q + off
    tk = 0.09 * n
    g = VGroup(Line(a, b, color=color, stroke_width=width),
               Line(a - tk, a + tk, color=color, stroke_width=width),
               Line(b - tk, b + tk, color=color, stroke_width=width))
    t = tag(label, size, color) if isinstance(label, str) else label
    t.move_to((a + b) / 2 + n * (gap + 0.5 * (abs(n[0]) * t.width
                                             + abs(n[1]) * t.height)))
    return VGroup(g, t)


_BASE = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789+=")


def _baseline(t):
    """Baseline of a Text: the median bottom of its non-descending glyphs."""
    chars = [c for c in t.text if not c.isspace()]
    glyphs = list(t.submobjects)
    if len(chars) == len(glyphs):
        ys = [g.get_bottom()[1] for c, g in zip(chars, glyphs) if c in _BASE]
    else:
        ys = []
    if not ys:
        ys = [g.get_bottom()[1] for g in glyphs]
    return float(np.median(ys))


def _rich(parts, size=30, color=WHITE, small=0.64, rise=0.5, drop=0.42):
    """One line of text with true sub- and superscripts, without TeX.
    parts: list of (string, kind), kind "" (normal), "_" (subscript) or
    "^" (superscript). Normal pieces share one baseline."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for s, kind in parts:
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        body = s.strip(" ")
        x += lead * sp
        if body:
            t = Text(body, font_size=size * (small if kind else 1.0),
                     color=color)
            y0 = {"": 0.0, "^": rise * xh, "_": -drop * xh}[kind]
            t.shift(np.array([x - t.get_left()[0], y0 - _baseline(t), 0.0]))
            x = t.get_right()[0] + (0.015 if kind else 0.02)
            g.add(t)
        x += trail * sp
    return g


def _cap(group):
    """Place a composite formula where caption() puts its Text."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.to_edge(DOWN, buff=0.3)
    group.set_x(0.0)
    return group


def _cap_ok(cap):
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")


def _all_safe(scene, cap):
    """Closing frame: everything except the caption inside the safe area."""
    _cap_ok(cap)
    skip = {id(x) for x in cap.get_family()}
    _safe(*[m for m in scene.mobjects if id(m) not in skip])


# ====================================================================== B44

class B44_OddPyramid(Board):
    """Rows of 1, 3, 5, …, 2n−1 dots, centred one under another, make a
    pyramid. Cut it just right of the middle column: the left half keeps
    rows of 1, 2, …, n dots, the right half rows of 1, …, n−1 (the row of
    2k−1 dots splits as k + (k−1)). Turn the right half over — a half-turn
    in space about the horizontal line just above the apex — so it hangs
    upside down above the pyramid, then slide it down along the slanted
    side of the left half: its rows of n−1, …, 1 dots drop in beside the
    rows of 1, …, n−1 dots of the left half, and every row of the n-square
    is full:  1 + 3 + 5 + … + (2n−1) = n².  Drawn for n = 5."""

    def construct(self):
        n, s, r = 5, 0.6, 0.125
        O = np.array([0.3, -2.02, 0.0])

        def P(p):
            return O + s * np.array([p[0], p[1], 0.0])

        # lattice: row y (0 = bottom) of the pyramid holds x = −(n−1−y) … n−1−y
        pyr = [(x, y) for y in range(n) for x in range(-(n - 1 - y), n - y)]
        L = [(x, y) for (x, y) in pyr if x <= 0]          # keeps the middle
        R = [(x, y) for (x, y) in pyr if x >= 1]
        hinge = n - 0.5                                   # just above the apex
        flipped = [(x, round(2 * hinge - y)) for (x, y) in R]
        M = [(x - n, y - n) for (x, y) in flipped]        # after the slide
        square = [(x, y) for x in range(-(n - 1), 1) for y in range(n)]
        check([sum(1 for p in pyr if p[1] == y) for y in range(n - 1, -1, -1)]
              == [2 * k - 1 for k in range(1, n + 1)], "rows 1, 3, …, 2n−1")
        check([sum(1 for p in L if p[1] == y) for y in range(n)]
              == list(range(n, 0, -1)), "left half: rows n, …, 1")
        check([sum(1 for p in R if p[1] == y) for y in range(n - 1)]
              == list(range(n - 1, 0, -1)), "right half: rows n−1, …, 1")
        check(_tiles_once([L, M], square), "left half + turned half = n-square")
        # the turned half slides along y − x = const; the left half has
        # y − x ≤ n − 1 and the turned half y − x ≥ n: they never meet
        check(max(y - x for x, y in L) == n - 1
              and min(y - x for x, y in flipped) == n, "the slide is clear")
        check(min(x for x, _ in R) >= 1 and max(x for x, _ in L) <= 0,
              "the turn-over stays right of the cut")

        dots = {p: Dot(P(p), radius=r, color=GREY_A) for p in pyr}
        Lg = VGroup(*[dots[p] for p in L])
        Rg = VGroup(*[dots[p] for p in R])

        # ---- the pyramid, row by row from the apex, each row counted
        counts = VGroup()
        for y in range(n - 1, -1, -1):
            row = [dots[(x, y)] for x in range(-(n - 1 - y), n - y)]
            lab = tag(str(2 * (n - y) - 1), 26, GREY_A).next_to(
                row[0], LEFT, buff=0.34)
            counts.add(lab)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in row],
                                  lag_ratio=0.12), FadeIn(lab), run_time=0.55)
        self.hold(0.5)

        # ---- cut just right of the middle column
        cut = DashedLine(P((0.5, -0.4)), P((0.5, n - 0.35)), color=YELLOW_B,
                         stroke_width=4, dash_length=0.1)
        for lab in counts:
            check(_boxes_apart(lab, cut, 0.12), "row counts clear of the cut")
        self.play(Create(cut), run_time=0.8)
        self.play(Lg.animate.set_color(BLUE_C), Rg.animate.set_color(ORANGE),
                  run_time=0.7)
        yb = P((0, 0))[1] - r
        d_l = _dim([P((-(n - 1), 0))[0] - r, yb, 0], [P((0, 0))[0] + r, yb, 0],
                   "n", DOWN * 0.26, BLUE_B, 28, gap=0.1)
        d_r = _dim([P((1, 0))[0] - r, yb, 0], [P((n - 1, 0))[0] + r, yb, 0],
                   "n−1", DOWN * 0.26, ORANGE, 28, gap=0.1)
        _safe(d_l, d_r, counts)
        _apart(d_l[1], d_r[1], *counts)
        check(_boxes_apart(d_l, cut, 0.06) and _boxes_apart(d_r, cut, 0.06),
              "brackets clear of the cut")
        self.play(FadeIn(d_l), FadeIn(d_r), run_time=0.7)
        self.hold(1.2)

        # ---- turn the right half over, about the line just above the apex
        hl = DashedLine(P((0.35, hinge)), P((n - 0.35, hinge)), color=YELLOW_B,
                        stroke_width=4, dash_length=0.1)
        self.play(FadeOut(counts), FadeOut(d_l), FadeOut(d_r), FadeOut(cut),
                  Create(hl), run_time=0.7)
        self.play(_turn_over(Rg, P((0, hinge)), P((1, hinge))), run_time=2.0)
        check(_lands_on(Rg, [P(p) for p in flipped]), "turned over in place")
        self.play(FadeOut(hl), run_time=0.4)

        # ---- slide it down along the slanted side of the left half
        self.play(Rg.animate.shift(s * np.array([-n, -n, 0.0])), run_time=2.0)
        check(_lands_on(Rg, [P(p) for p in M]), "the turned half fills the gap")
        check(_lands_on(VGroup(*Lg, *Rg), [P(p) for p in square]),
              "the dots fill the n-square")
        self.hold(0.4)

        # centre the finished square, then measure it
        sq = VGroup(*Lg, *Rg)
        v = np.array([0.35, 0.55, 0.0]) - sq.get_center()
        self.play(sq.animate.shift(v), run_time=0.9)
        O = O + v
        check(_lands_on(sq, [P(p) for p in square]), "square moved whole")
        pad = 0.55
        c0, c1 = P((-(n - 1), 0)), P((0, n - 1))
        frame = Rectangle(width=c1[0] - c0[0] + 2 * pad * s,
                          height=c1[1] - c0[1] + 2 * pad * s,
                          stroke_color=YELLOW_B, stroke_width=4
                          ).move_to((c0 + c1) / 2)
        D_b = _dim(frame.get_corner(DL), frame.get_corner(DR), "n",
                   DOWN * 0.22, YELLOW_B, 30, gap=0.1)
        D_l = _dim(frame.get_corner(DL), frame.get_corner(UL), "n",
                   LEFT * 0.22, YELLOW_B, 30, gap=0.12)
        _safe(frame, D_b, D_l, Lg, Rg)
        _apart(D_b[1], D_l[1])
        self.play(Create(frame), FadeIn(D_b), FadeIn(D_l), run_time=1.0)
        cap = caption("1 + 3 + 5 + … + (2n−1)  =  n²", 34)
        self.play(Write(cap), run_time=1.4)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== B27

def _dirs(*degs):
    return [np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
            for a in degs]


def _glide_pts(src, dst, t):
    """Points of a rigid row of dots part-way (t in [0, 1]) along the glide
    used on screen: a turn about its centre while the centre moves
    straight from the start to the end position."""
    src, dst = np.array(src, float), np.array(dst, float)
    c0, c1 = src.mean(axis=0), dst.mean(axis=0)
    th = 0.0
    if len(src) > 1:
        u, v = src[1] - src[0], dst[1] - dst[0]
        th = (np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]) + PI) % TAU - PI
    cs, sn = np.cos(t * th), np.sin(t * th)
    R = np.array([[cs, -sn], [sn, cs]])
    return np.array([c0 + t * (c1 - c0) + R @ (p - c0) for p in src])


def _turn_of(src, dst):
    """The turn (radians) of the rigid motion carrying the row src onto
    the row dst, after checking that one exists."""
    src, dst = np.array(src, float), np.array(dst, float)
    if len(src) == 1:
        return 0.0
    u, v = src[1] - src[0], dst[1] - dst[0]
    th = (np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]) + PI) % TAU - PI
    cs, sn = np.cos(th), np.sin(th)
    R = np.array([[cs, -sn], [sn, cs]])
    check(all(close(dst[0] + R @ (p - src[0]), q) for p, q in zip(src, dst)),
          "a rigid motion carries the row onto its place")
    return float(th)


def _stages_clear(stages, start, end, min_gap, steps=48):
    """Pieces move family by family (each stage together, by _glide_pts);
    every moving dot must stay at least min_gap from every other dot."""
    cur = {key: np.array(start[key], float) for key in start}
    worst = 1e9
    for stage in stages:
        for t in np.linspace(0.0, 1.0, steps):
            pos = dict(cur)
            for key in stage:
                pos[key] = _glide_pts(start[key], end[key], t)
            for key in stage:
                for other, pts in pos.items():
                    if other != key:
                        dmin = np.min(np.linalg.norm(
                            pos[key][:, None, :] - pts[None, :, :], axis=2))
                        worst = min(worst, float(dmin))
        for key in stage:
            cur[key] = np.array(end[key], float)
    return worst >= min_gap


class B27_PentagonalHouse(Board):
    """The pentagonal numbers 1, 5, 12, 22, … are nested regular pentagons
    sharing one corner: the k-th adds three sides of k dots, k + (k−1) +
    (k−1) = 3k − 2 new dots. Of these three sides, the lower-left one
    (turned 18° upright) and the bottom one (slid, not turned) are the two
    arms of an L — the k-th shell of an n × n square (2k − 1 dots) — and
    the lower-right one (k − 1 dots, turned 72° level) is a row of a
    triangle. Moved over rigidly, side by side, the n pentagons become an
    n-square with the triangle Tₙ₋₁ on top:
        Pₙ = n² + Tₙ₋₁ = n(3n−1)/2.     Drawn for n = 5."""

    def construct(self):
        n, s, r = 5, 0.6, 0.12
        h = np.sqrt(3) / 2
        X0 = np.array([-2.3, 1.05, 0.0])               # the shared corner V0

        def P(p):
            return X0 + s * np.array([p[0], p[1], 0.0])

        dd = _dirs(216, 288, 0, 72, 144)                # sides, round from V0
        verts, A, B, C = {}, {}, {}, {}
        for k in range(1, n + 1):
            v = [np.zeros(2)]
            for i in range(4):
                v.append(v[-1] + (k - 1) * dd[i])
            verts[k] = v
            A[k] = [v[1] + j * dd[1] for j in range(k)]          # V1 .. V2
            if k > 1:
                B[k] = [v[2] + j * dd[2] for j in range(1, k)]   # .. V3
                C[k] = [v[3] + j * dd[3] for j in range(1, k)]   # .. V4
            check(close(v[4] + (k - 1) * dd[4], (0, 0)), "the pentagon closes")
            check(len(A[k]) + len(B.get(k, [])) + len(C.get(k, [])) == 3 * k - 2,
                  "gnomon k has 3k − 2 dots")
        dots_all = ([p for k in A for p in A[k]] + [p for k in B for p in B[k]]
                    + [p for k in C for p in C[k]])
        check(len(dots_all) == n * (3 * n - 1) // 2, "Pₙ = n(3n−1)/2 dots")
        check(min(np.linalg.norm(p - q) for i, p in enumerate(dots_all)
                  for q in dots_all[i + 1:]) > 1 - 1e-9, "no two dots coincide")

        # the house: n-square (top-left dot at H) and the triangle on top
        H = np.array([5.0, 0.5])
        tA = {k: [H + np.array([n - k, -j]) for j in range(k)] for k in A}
        tB = {k: [H + np.array([n - k + j, -(k - 1)]) for j in range(1, k)]
              for k in B}
        tC = {k: [H + np.array([(n - 1) / 2 - (k - 2) / 2 + (j - 1),
                                (n - k + 1) * h]) for j in range(1, k)]
              for k in C}
        sq = {(c, rr) for c in range(n) for rr in range(n)}

        def cellof(p):
            return (int(round(p[0] - H[0])), int(round(H[1] - p[1])))

        check(_tiles_once([[cellof(p) for p in tA[k] + tB.get(k, [])]
                           for k in A], sq), "the L's tile the n-square")
        for k in A:
            shell = {(c, rr) for (c, rr) in sq
                     if max(n - 1 - c, rr) == k - 1}
            check({cellof(p) for p in tA[k] + tB.get(k, [])} == shell,
                  f"gnomon {k}'s L is shell {k} of the square")
        roof = [tuple(np.round(p, 9)) for k in C for p in tC[k]]
        check(len(set(roof)) == len(roof) == (n - 1) * n // 2, "roof = Tₙ₋₁")
        for k in C:
            check(abs(tC[k][-1][0] - tC[k][0][0] - (k - 2)) < 1e-9
                  and abs(tC[k][0][1] - H[1] - (n - k + 1) * h) < 1e-9,
                  f"the roof row of {k - 1} dots")
        turns = {}
        for f, src, dst in (("A", A, tA), ("B", B, tB), ("C", C, tC)):
            for k in src:
                turns[(f, k)] = _turn_of(src[k], dst[k])
        check(all(abs(np.degrees(turns[("A", k)]) + 18) < 1e-6 for k in A if k > 1)
              and all(abs(turns[("B", k)]) < 1e-9 for k in B)
              and all(abs(np.degrees(turns[("C", k)]) + 72) < 1e-6
                      for k in C if k > 2),
              "turns: −18° (left sides), 0° (bottoms), −72° (right sides)")
        start = {**{("A", k): A[k] for k in A}, **{("B", k): B[k] for k in B},
                 **{("C", k): C[k] for k in C}}
        end = {**{("A", k): tA[k] for k in A}, **{("B", k): tB[k] for k in B},
               **{("C", k): tC[k] for k in C}}
        stages = [[("C", k) for k in C], [("B", k) for k in B],
                  [("A", k) for k in A]]
        check(_stages_clear(stages, start, end, 0.85),
              "no dot comes within 0.85 of another while the sides move")

        # ---- the pentagonal numbers, gnomon by gnomon
        piece = {key: VGroup(*[Dot(P(p), radius=r, color=GREY_A) for p in pts])
                 for key, pts in start.items()}
        outlines = VGroup(*[Polygon(*[P(v) for v in verts[k]], stroke_color=GREY_D,
                                    stroke_width=2) for k in range(2, n + 1)])
        vals = [k * (3 * k - 1) // 2 for k in range(1, n + 1)]
        ctr = VGroup(*[tag(str(v), 28, GREY_A) for v in vals]).arrange(
            RIGHT, buff=0.5).move_to(P((0, 0)) + UP * 0.95)
        for k in range(1, n + 1):
            new = [piece[(f, k)] for f in "ABC" if (f, k) in piece]
            anims = [FadeIn(VGroup(*new), scale=0.6), FadeIn(ctr[k - 1])]
            if k > 1:
                anims.append(Create(outlines[k - 2]))
            self.play(*anims, run_time=0.6)
        self.bring_to_front(*piece.values())
        self.hold(0.5)

        # ---- each gnomon: an L (blue) and one more side (orange)
        blues, oranges = [BLUE_B, BLUE_D], [ORANGE, GOLD_E]
        self.play(*[piece[("A", k)].animate.set_color(blues[k % 2]) for k in A],
                  *[piece[("B", k)].animate.set_color(blues[k % 2]) for k in B],
                  *[piece[("C", k)].animate.set_color(oranges[k % 2]) for k in C],
                  run_time=0.9)
        house_pts = [H, H + np.array([0, -(n - 1)]), H + np.array([n - 1, -(n - 1)]),
                     H + np.array([n - 1, 0]),
                     H + np.array([(n - 1) / 2, (n - 1) * h])]
        ghost = DashedVMobject(Polygon(*[P(p) for p in house_pts],
                                       stroke_color=GREY_B, stroke_width=2),
                               num_dashes=60)
        self.play(Create(ghost), run_time=0.8)
        self.hold(0.3)

        # ---- right sides to the roof, bottoms to rows, left sides to columns
        for stage in stages:
            anims = []
            for key in stage:
                self.bring_to_front(piece[key])
                tgt = np.mean(np.array(end[key], float), axis=0)
                anims.append(_glide(piece[key], piece[key].get_center(), P(tgt),
                                    turns[key]))
            self.play(*anims, run_time=1.7)
            for key in stage:
                check(_lands_on(piece[key], [P(p) for p in end[key]]),
                      f"side {key} landed")
            self.hold(0.2)

        # ---- centre the house, then name the square and the triangle
        self.play(FadeOut(outlines), FadeOut(ghost), FadeOut(ctr), run_time=0.6)
        house = VGroup(*piece.values())
        v = np.array([-0.15, 0.55, 0.0]) - house.get_center()
        self.play(house.animate.shift(v), run_time=1.0)
        X0 = X0 + v
        for key in piece:
            check(_lands_on(piece[key], [P(p) for p in end[key]]),
                  "the house moved whole")
        c_bl, c_tr = P(H + np.array([0, -(n - 1)])), P(H + np.array([n - 1, 0]))
        d_b = _dim(c_bl + LEFT * r + DOWN * r, [c_tr[0] + r, c_bl[1] - r, 0],
                   "n", DOWN * 0.24, YELLOW_B, 30, gap=0.1)
        d_l = _dim(c_bl + LEFT * r + DOWN * r, [c_bl[0] - r, c_tr[1] + r, 0],
                   "n", LEFT * 0.24, YELLOW_B, 30, gap=0.12)
        l_sq = tag("n²", 36, BLUE_B).next_to(
            np.array([c_tr[0] + r, (c_bl[1] + c_tr[1]) / 2, 0]), RIGHT, buff=0.42)
        mid_roof = P(H + np.array([n - 1.5 - 0.5 * (n // 2 - 1), (n // 2) * h]))
        l_tr = tag("Tₙ₋₁", 36, ORANGE).next_to(mid_roof, RIGHT, buff=0.45)
        labels = [d_b, d_l, l_sq, l_tr]
        _safe(*labels, *piece.values())
        _apart(d_b[1], d_l[1], l_sq, l_tr)
        for lab in (l_sq, l_tr, d_b, d_l):
            _clear_of(lab, [d for g in piece.values() for d in g], gap=0.08)
        self.play(FadeIn(d_b), FadeIn(d_l), FadeIn(l_sq), FadeIn(l_tr),
                  run_time=0.9)
        cap = caption("Pₙ  =  n² + Tₙ₋₁  =  n(3n−1)/2", 34)
        self.play(Write(cap), run_time=1.4)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== B33

def _cells_group(cells, u, org, color, stroke=1.2):
    return VGroup(*[cell(i, j, u, org, color, op=0.85, stroke=stroke)
                    for (i, j) in cells])


class B33_GalileoRatio(Board):
    """The odd numbers 1, 3, 5, … are the L-shaped shells of a growing
    square. The first n of them make the n-square; the next n make the L
    that grows it to the 2n-square. Halving the 2n-square both ways shows
    that L as three copies of the n-square, so
        (1 + 3 + … + (2n−1)) : ((2n+1) + … + (4n−1)) = 1 : 3.
    Drawn for n = 4: 16 : 48."""

    def construct(self):
        n, u = 4, 0.6
        N = 2 * n
        org = np.array([-N * u / 2 + 0.05, -2.33, 0.0])

        def C(x, y):                                  # grid corner -> screen
            return org + u * np.array([x, y, 0.0])

        shell = {k: [(i, j) for i in range(k) for j in range(k)
                     if max(i, j) == k - 1] for k in range(1, N + 1)}
        quads = {"Q0": (0, 0), "Q1": (n, 0), "Q2": (0, n), "Q3": (n, n)}
        Q = {q: [(i + a, j + b) for i in range(n) for j in range(n)]
             for q, (a, b) in quads.items()}
        for k in shell:
            check(len(shell[k]) == 2 * k - 1, f"shell {k} has {2 * k - 1} cells")
        check(_tiles_once([shell[k] for k in range(1, n + 1)], Q["Q0"]),
              "the first n shells tile the n-square")
        check(_tiles_once([shell[k] for k in range(n + 1, N + 1)],
                          Q["Q1"] + Q["Q2"] + Q["Q3"]),
              "the next n shells tile the other three quarters")
        check(sum(range(1, 2 * n, 2)) * 3 == sum(range(2 * n + 1, 4 * n, 2)),
              "1 + 3 + … + (2n−1) is a third of (2n+1) + … + (4n−1)")

        blues, oranges = [BLUE_C, BLUE_E], [ORANGE, GOLD_E]
        sh = {k: _cells_group(shell[k], u, org,
                              (blues if k <= n else oranges)[k % 2])
              for k in shell}
        nums = {k: tag(str(2 * k - 1), 22, BLUE_B if k <= n else ORANGE)
                .move_to(C(k - 0.5, 0) + DOWN * 0.3) for k in shell}
        _apart(*nums.values(), gap=0.05)

        # ---- the first n odd numbers: the n-square
        for k in range(1, n + 1):
            self.play(FadeIn(sh[k], shift=UP * 0.12), FadeIn(nums[k]),
                      run_time=0.5)
        sqA = Square(side_length=n * u, stroke_color=YELLOW_B,
                     stroke_width=5).move_to(C(n / 2, n / 2))
        d_a = _dim(C(0, 0), C(0, n), "n", LEFT * 0.24, YELLOW_B, 30, gap=0.12)
        lab_a = tag("n²", 36).move_to(C(n / 2, n / 2))
        bg_a = BackgroundRectangle(lab_a, color=BLACK, fill_opacity=0.55,
                                   buff=0.08)
        self.play(Create(sqA), FadeIn(d_a), FadeIn(bg_a), FadeIn(lab_a),
                  run_time=0.9)
        self.hold(0.5)

        # ---- the next n odd numbers: the L up to the 2n-square
        for k in range(n + 1, N + 1):
            self.bring_to_front(sqA, bg_a, lab_a)
            self.play(FadeIn(sh[k], shift=UP * 0.12), FadeIn(nums[k]),
                      run_time=0.5)
        self.bring_to_front(sqA, bg_a, lab_a)
        big = Square(side_length=N * u, stroke_color=WHITE,
                     stroke_width=4).move_to(C(n, n))
        d_N = _dim(C(N, 0), C(N, N), "2n", RIGHT * 0.24, WHITE, 30, gap=0.12)
        self.play(Create(big), FadeIn(d_N), run_time=0.8)
        self.hold(0.4)

        # ---- halve the 2n-square both ways: the L is three n-squares
        halves = VGroup(DashedLine(C(n, 0), C(n, N), color=WHITE,
                                   stroke_width=5, dash_length=0.12),
                        DashedLine(C(0, n), C(N, n), color=WHITE,
                                   stroke_width=5, dash_length=0.12))
        self.play(Create(halves), run_time=0.8)
        copies = []
        for q in ("Q1", "Q2", "Q3"):
            a, b = quads[q]
            cp = VGroup(sqA.copy(), bg_a.copy(), lab_a.copy())
            self.add(cp)
            self.play(cp.animate.shift(u * np.array([a, b, 0.0])), run_time=0.9)
            check(close(cp[0].get_center(), C(a + n / 2, b + n / 2), 1e-6),
                  f"a copy of the n-square lands exactly on {q}")
            copies.append(cp)
        labs = [lab_a] + [cp[2] for cp in copies]
        _apart(*labs, d_a[1], d_N[1], *nums.values())
        _safe(big, d_a, d_N, *nums.values(), *copies)
        cap = caption("(1 + 3 + … + (2n−1))  :  ((2n+1) + … + (4n−1))  =  1 : 3",
                      32)
        self.play(Write(cap), run_time=1.5)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== B30

def _Tsub(sub, size=34, color=WHITE):
    """The letter T with a true subscript (a, b, a+b …)."""
    return _rich([("T", ""), (sub, "_")], size, color)


class B30_AddingTriangulars(Board):
    """The staircase T(a+b) has columns of 1, 2, …, a+b cells. Its first a
    columns are the staircase T(a). Each of the other b columns is a + j
    cells high (j = 1 … b): its bottom a cells make an a × b rectangle and
    the j cells on top make the staircase T(b). Cut along the two lines and
    pull the three pieces apart, then put them back:
        T(a+b) = T(a) + T(b) + ab.     Drawn for a = 4, b = 3."""

    def construct(self):
        a, b, u = 4, 3, 0.62
        N = a + b
        org = np.array([-2.3, -2.2, 0.0])

        def C(x, y):                                   # cell corner -> screen
            return org + u * np.array([x, y, 0.0])

        T = [(i, j) for i in range(N) for j in range(i + 1)]
        Ta = [(i, j) for (i, j) in T if i < a]
        R = [(i, j) for (i, j) in T if i >= a and j < a]
        Tb = [(i, j) for (i, j) in T if i >= a and j >= a]
        check(_tiles_once([Ta, R, Tb], T), "T(a), the rectangle and T(b) tile T(a+b)")
        check([sum(1 for c in Ta if c[0] == i) for i in range(a)]
              == list(range(1, a + 1)), "T(a): columns 1 … a")
        check({(i - a, j) for (i, j) in R}
              == {(x, y) for x in range(b) for y in range(a)}, "an a × b rectangle")
        check([sum(1 for c in Tb if c[0] == i) for i in range(a, N)]
              == list(range(1, b + 1)), "T(b): columns 1 … b")
        check(len(T) == len(Ta) + len(Tb) + a * b, "T(a+b) = T(a) + T(b) + ab")

        cA, cR, cB = BLUE_D, TEAL_D, ORANGE
        gA = _cells_group(Ta, u, org, GREY_C)
        gR = _cells_group(R, u, org, GREY_C)
        gB = _cells_group(Tb, u, org, GREY_C)
        # ---- the staircase T(a+b), column by column
        cols = [VGroup(*[m for g, cells in ((gA, Ta), (gR, R), (gB, Tb))
                         for m, c in zip(g, cells) if c[0] == i])
                for i in range(N)]
        lab_T = _Tsub("a+b", 36).move_to(C(1.6, 4.6))
        d_ab = _dim(C(0, 0), C(N, 0), "a + b", DOWN * 0.26, YELLOW_B, 28, gap=0.1)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cols],
                              lag_ratio=0.25), run_time=2.0)
        _clear_of(lab_T, [*gA, *gR, *gB])
        self.play(FadeIn(lab_T), FadeIn(d_ab), run_time=0.7)
        self.hold(0.6)

        # ---- two cuts: up the side of T(a), across at height a
        cut1 = DashedLine(C(a, 0), C(a, a), color=YELLOW_C, stroke_width=8,
                          dash_length=0.13)
        cut2 = DashedLine(C(a, a), C(N, a), color=YELLOW_C, stroke_width=8,
                          dash_length=0.13)
        self.play(Create(cut1), Create(cut2), FadeOut(lab_T), FadeOut(d_ab),
                  run_time=0.9)
        self.play(gA.animate.set_fill(cA), gR.animate.set_fill(cR),
                  gB.animate.set_fill(cB), run_time=0.7)

        # ---- pull the pieces apart and name them
        vA, vB = LEFT * u, UP * u
        self.play(FadeOut(cut1), FadeOut(cut2), gA.animate.shift(vA),
                  gB.animate.shift(vB), run_time=1.2)
        check(_lands_on(gA, [C(i + 0.5, j + 0.5) + vA for (i, j) in Ta])
              and _lands_on(gB, [C(i + 0.5, j + 0.5) + vB for (i, j) in Tb]),
              "pieces pulled straight apart")

        def labels(shiftA, shiftB):
            lA = _Tsub("a", 36, BLUE_B).move_to(C(1.0, 3.05) + shiftA)
            lB = _Tsub("b", 36, ORANGE).move_to(C(3.0, 5.55) + shiftB)
            lR = tag("ab", 34).move_to(C(a + b / 2, a / 2))
            dA = _dim(C(0, 0) + shiftA, C(a, 0) + shiftA, "a", DOWN * 0.26,
                      BLUE_B, 28, gap=0.1)
            dR = _dim(C(a, 0), C(N, 0), "b", DOWN * 0.26, TEAL_B, 28, gap=0.1)
            dRh = _dim(C(N, 0), C(N, a), "a", RIGHT * 0.26, TEAL_B, 28, gap=0.12)
            dBh = _dim(C(N, a) + shiftB, C(N, N) + shiftB, "b", RIGHT * 0.26,
                       ORANGE, 28, gap=0.12)
            return [lA, lB, lR, dA, dR, dBh, dRh]

        apart = labels(vA, vB)
        lA, lB, lR, dA, dR, dBh, dRh = apart
        _clear_of(lA, [*gA, *gR, *gB])
        _clear_of(lB, [*gA, *gR, *gB])
        _safe(*apart, gA, gR, gB)
        _apart(lA, lB, lR, dA[1], dR[1], dBh[1], dRh[1])
        self.play(*[FadeIn(m) for m in apart], run_time=0.9)
        self.hold(1.4)

        # ---- and back together: the same staircase
        back = labels(ORIGIN, ORIGIN)
        self.play(gA.animate.shift(-vA), gB.animate.shift(-vB),
                  *[Transform(m, t) for m, t in zip(apart, back)], run_time=1.3)
        check(_lands_on(VGroup(*gA, *gR, *gB), [C(i + 0.5, j + 0.5) for (i, j) in T]),
              "the three pieces fill T(a+b) again")
        lA2, lB2 = back[0], back[1]
        _clear_of(lA2, [*gA, *gR, *gB])
        _clear_of(lB2, [*gA, *gR, *gB])
        _apart(back[0], back[1], back[2], *[d[1] for d in back[3:]])
        _safe(*back)
        cap = _cap(_rich([("T", ""), ("a+b", "_"), ("  =  T", ""), ("a", "_"),
                          ("  +  T", ""), ("b", "_"), ("  +  ab", "")],
                         36, YELLOW_B))
        self.play(FadeIn(cap), run_time=1.2)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== B28

class B28_HexagonalTriangular(Board):
    """The hexagonal numbers 1, 6, 15, 28, … are nested regular hexagons
    sharing one corner (the top one): the k-th adds four sides, k + 3(k−1)
    = 4k − 3 dots. Split the k-th gnomon into the two vertical sides
    (k − 1 dots each, the lower corners left out) and the bottom "V"
    (2k − 1 dots, both of its upper corners in). Turned level and moved
    over rigidly, the V is row 2k − 1 of a triangle of dots and the two
    vertical sides, end to end, are row 2k − 2: every gnomon is two
    consecutive rows, (2k−2) + (2k−1) = 4k − 3, and the n hexagons stack
    into the triangle with 2n − 1 rows:
        Hₙ = n(2n−1) = T₂ₙ₋₁.     Drawn for n = 4."""

    def construct(self):
        n, s, r = 4, 0.72, 0.13
        h = np.sqrt(3) / 2
        X0 = np.array([-2.66, 2.36, 0.0])              # the shared corner V0

        def P(p):
            return X0 + s * np.array([p[0], p[1], 0.0])

        E = _dirs(210, 270, 330, 30, 90, 150)
        verts, pcs = {}, {}
        for k in range(1, n + 1):
            v = [np.zeros(2)]
            for i in range(5):
                v.append(v[-1] + (k - 1) * E[i])
            verts[k] = v
            check(close(v[5] + (k - 1) * E[5], (0, 0)), "the hexagon closes")
            if k == 1:
                pcs[("O", 1)] = [v[0]]
                continue
            pcs[("A", k)] = [v[1] + j * E[1] for j in range(k - 1)]   # left side
            pcs[("B", k)] = [v[2] + j * E[2] for j in range(k)]       # V, left arm
            pcs[("C", k)] = [v[3] + j * E[3] for j in range(1, k)]    # V, right arm
            pcs[("D", k)] = [v[4] + j * E[4] for j in range(1, k)]    # right side
            check(sum(len(pcs[(f, k)]) for f in "ABCD") == 4 * k - 3,
                  "gnomon k has 4k − 3 dots")
        dots_all = [p for v in pcs.values() for p in v]
        check(len(dots_all) == n * (2 * n - 1), "Hₙ = n(2n−1) dots")
        check(min(np.linalg.norm(p - q) for i, p in enumerate(dots_all)
                  for q in dots_all[i + 1:]) > 1 - 1e-9, "no two dots coincide")

        # the triangle with 2n − 1 rows, apex at T0: row ρ has ρ dots
        T0 = np.array([7.0, 0.5])
        tg = {("O", 1): [T0]}
        for k in range(2, n + 1):
            yo, ye = -(2 * k - 2) * h, -(2 * k - 3) * h          # rows 2k−1, 2k−2
            tg[("B", k)] = [T0 + np.array([-(k - 1) + j, yo]) for j in range(k)]
            tg[("C", k)] = [T0 + np.array([j, yo]) for j in range(1, k)]
            tg[("A", k)] = [T0 + np.array([-(2 * k - 3) / 2 + j, ye])
                            for j in range(k - 1)]
            tg[("D", k)] = [T0 + np.array([j - 0.5, ye]) for j in range(1, k)]
        tri = {(round(-(rho - 1) / 2 + i, 6), round(-(rho - 1) * h, 6))
               for rho in range(1, 2 * n) for i in range(rho)}
        got = [(round(p[0] - T0[0], 6), round(p[1] - T0[1], 6))
               for v in tg.values() for p in v]
        check(len(got) == len(set(got)) and set(got) == tri,
              "the moved sides fill the triangle T₂ₙ₋₁ exactly once")
        turns = {key: _turn_of(pcs[key], tg[key]) for key in pcs}
        for k in range(3, n + 1):
            check(abs(np.degrees(turns[("A", k)]) - 90) < 1e-6
                  and abs(np.degrees(turns[("B", k)]) - 30) < 1e-6
                  and abs(np.degrees(turns[("C", k)]) + 30) < 1e-6
                  and abs(np.degrees(turns[("D", k)]) + 90) < 1e-6,
                  "turns +90°, +30°, −30°, −90°")
        stages = [[key for key in pcs if key[0] == f] for f in "DCBOA"]
        check(_stages_clear(stages, pcs, tg, 0.75),
              "no dot comes within 0.75 of another while the sides move")

        # ---- the hexagonal numbers, gnomon by gnomon
        cols = {1: YELLOW_D, 2: BLUE_C, 3: ORANGE, 4: GREEN_C}
        piece = {key: VGroup(*[Dot(P(p), radius=r, color=cols[key[1]])
                               for p in pts]) for key, pts in pcs.items()}
        outlines = VGroup(*[Polygon(*[P(v) for v in verts[k]], stroke_color=GREY_D,
                                    stroke_width=2) for k in range(2, n + 1)])
        vals = [k * (2 * k - 1) for k in range(1, n + 1)]
        ctr = VGroup(*[tag(str(v), 28, GREY_A) for v in vals]).arrange(
            RIGHT, buff=0.5).move_to(P((0, 0)) + UP * 0.75)
        for k in range(1, n + 1):
            new = VGroup(*[piece[key] for key in pcs if key[1] == k])
            anims = [FadeIn(new, scale=0.6), FadeIn(ctr[k - 1])]
            if k > 1:
                anims.append(Create(outlines[k - 2]))
            self.play(*anims, run_time=0.65)
        self.bring_to_front(*piece.values())
        ghost = DashedVMobject(Polygon(P(T0), P(T0 + np.array([-(n - 1), -(2 * n - 2) * h])),
                                       P(T0 + np.array([n - 1, -(2 * n - 2) * h])),
                                       stroke_color=GREY_B, stroke_width=2),
                               num_dashes=54)
        self.play(Create(ghost), run_time=0.8)
        self.hold(0.3)

        # ---- right sides, the V's two arms, the corner, left sides
        for stage in stages:
            anims = []
            for key in stage:
                self.bring_to_front(piece[key])
                tgt = np.mean(np.array(tg[key], float), axis=0)
                anims.append(_glide(piece[key], piece[key].get_center(), P(tgt),
                                    turns[key]))
            self.play(*anims, run_time=1.25 if stage[0][0] != "O" else 0.8)
            for key in stage:
                check(_lands_on(piece[key], [P(p) for p in tg[key]]),
                      f"side {key} landed")
        self.hold(0.3)

        # ---- centre the triangle; each gnomon is two rows
        self.play(FadeOut(outlines), FadeOut(ghost), FadeOut(ctr), run_time=0.6)
        allg = VGroup(*piece.values())
        v = np.array([-0.35, 0.55, 0.0]) - allg.get_center()
        self.play(allg.animate.shift(v), run_time=1.0)
        X0 = X0 + v
        for key in piece:
            check(_lands_on(piece[key], [P(p) for p in tg[key]]),
                  "the triangle moved whole")
        bands = VGroup()
        for k in range(1, n + 1):
            y_mid = T0[1] - (2 * k - 2.5) * h if k > 1 else T0[1]
            x_end = T0[0] + (k - 1) + 0.85
            bands.add(tag(str(4 * k - 3), 26, cols[k]).move_to(P((x_end, y_mid))))
        yb = P(T0 + np.array([0, -(2 * n - 2) * h]))[1] - r
        d_b = _dim([P(T0 + np.array([-(n - 1), 0]))[0] - r, yb, 0],
                   [P(T0 + np.array([n - 1, 0]))[0] + r, yb, 0], "2n − 1",
                   DOWN * 0.24, YELLOW_B, 30, gap=0.1)
        _safe(bands, d_b, allg)
        _apart(*bands, d_b[1])
        for lab in bands:
            _clear_of(lab, [d for g in piece.values() for d in g], gap=0.08)
        self.play(LaggedStart(*[FadeIn(b_) for b_ in bands], lag_ratio=0.2),
                  FadeIn(d_b), run_time=1.0)
        cap = caption("Hₙ  =  n(2n−1)  =  T₂ₙ₋₁", 36)
        self.play(Write(cap), run_time=1.3)
        _all_safe(self, cap)
        self.hold(2.2)


# ======================================================================= C9

def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _rect_tiling(rects, region, tol=1e-9):
    """True when the axis-aligned rectangles (lists of (x0,x1,y0,y1)) lie
    in the region, overlap nowhere, and their areas add up to it."""
    X0, X1, Y0, Y1 = region
    tot = 0.0
    for (x0, x1, y0, y1) in rects:
        if x0 < X0 - tol or x1 > X1 + tol or y0 < Y0 - tol or y1 > Y1 + tol:
            return False
        tot += (x1 - x0) * (y1 - y0)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a_, b_ = rects[i], rects[j]
            w = min(a_[1], b_[1]) - max(a_[0], b_[0])
            hh = min(a_[3], b_[3]) - max(a_[2], b_[2])
            if w > tol and hh > tol:
                return False
    return abs(tot - (X1 - X0) * (Y1 - Y0)) < 1e-7


class C9_SumDifferenceSquares(Board):
    """Lay two a-squares in opposite corners of the (a+b)-square. They
    overlap in the middle in a square of side a − b (from b to a both
    ways), and they leave two b-squares uncovered in the other corners.
    So the two a-squares and the two b-squares cover (a+b)² once and the
    middle square twice; lifting the second layer off leaves it beside:
        (a+b)² + (a−b)² = 2(a² + b²)."""

    def construct(self):
        a, b = 3.4, 1.4
        S, d = a + b, a - b
        g = 1.0                                        # gap to the lifted square
        F = Frame(-1.05, S + g + d + 0.45, -1.0, S + 0.35)
        P, k = F.P, F.k
        A1 = (0, a, 0, a)                                # bottom-left a-square
        A2 = (b, S, b, S)                                # top-right a-square
        O2 = (b, a, b, a)                                # their overlap
        B1, B2 = (0, b, a, S), (a, S, 0, b)              # the two corners left
        L2 = [(b, a), (a, a), (a, b), (S, b), (S, S), (b, S)]   # A2 less O2
        check(abs(abs(area(L2)) - (a * a - d * d)) < 1e-9, "A2 less the overlap")
        check(_rect_tiling([A1, B1, B2, (a, S, b, a), (b, S, a, S)], (0, S, 0, S)),
              "A1, the corners and A2 less the overlap tile (a+b)²")
        check(abs(min(A1[1], A2[1]) - max(A1[0], A2[0]) - d) < 1e-12
              and abs(min(A1[3], A2[3]) - max(A1[2], A2[2]) - d) < 1e-12,
              "the a-squares overlap in a square of side a − b")
        check(abs((S * S + d * d) - 2 * (a * a + b * b)) < 1e-9,
              "(a+b)² + (a−b)² = 2(a² + b²)")

        def box(r, color, op, **kw):
            x0, x1, y0, y1 = r
            return F.poly(_rect(x0, y0, x1 - x0, y1 - y0), color, op, **kw)

        ghost = DashedVMobject(Polygon(*[P(p) for p in _rect(0, 0, S, S)],
                                       stroke_color=GREY_B, stroke_width=2.5),
                               num_dashes=64)
        d_bot = VGroup(_dim(P((0, 0)), P((a, 0)), "a", DOWN * 0.3, GREY_A, 28),
                       _dim(P((a, 0)), P((S, 0)), "b", DOWN * 0.3, GREY_A, 28))
        d_lft = VGroup(_dim(P((0, 0)), P((0, a)), "a", LEFT * 0.3, GREY_A, 28),
                       _dim(P((0, a)), P((0, S)), "b", LEFT * 0.3, GREY_A, 28))
        self.play(Create(ghost), FadeIn(d_bot), FadeIn(d_lft), run_time=1.0)

        # ---- an a-square into one corner, another into the opposite one
        op_a = 0.42
        q1 = box(A1, BLUE_D, op_a)
        lab1 = tag("a²", 34, BLUE_A).move_to(P((a / 2, b / 2)))
        self.play(FadeIn(q1, shift=UP * 0.3 + RIGHT * 0.3), FadeIn(lab1),
                  run_time=1.0)
        l2 = F.poly(L2, TEAL_D, op_a, stroke_width=0)
        o2 = box(O2, TEAL_D, op_a, stroke_width=0)
        e2 = Polygon(*[P(p) for p in _rect(b, b, a, a)], stroke_color=TEAL_B,
                     stroke_width=3)
        q2 = VGroup(l2, o2, e2)
        lab2 = tag("a²", 34, TEAL_A).move_to(P((b + a / 2, a + b / 2)))
        self.play(FadeIn(q2, shift=DOWN * 0.3 + LEFT * 0.3), FadeIn(lab2),
                  run_time=1.0)
        self.hold(0.4)

        # ---- they overlap in the middle: a square of side a − b
        ov = Polygon(*[P(p) for p in _rect(b, b, d, d)], stroke_color=YELLOW_B,
                     stroke_width=5)
        lab_o = tag("(a−b)²", 32).move_to(P((S / 2, S / 2)))
        self.play(Create(ov), FadeIn(lab_o), run_time=0.9)
        self.hold(0.4)

        # ---- the two corners left over are b-squares
        c1, c2 = box(B1, ORANGE, 0.85), box(B2, ORANGE, 0.85)
        lb1 = tag("b²", 30).move_to(P((b / 2, a + b / 2)))
        lb2 = tag("b²", 30).move_to(P((a + b / 2, b / 2)))
        self.play(FadeIn(c1, shift=DOWN * 0.25), FadeIn(c2, shift=LEFT * 0.25),
                  FadeIn(lb1), FadeIn(lb2), run_time=1.0)
        self.bring_to_front(lab1, lab2, ov, lab_o)
        self.hold(0.6)

        # ---- the middle is covered twice: lift the second layer off
        lifted = VGroup(o2, ov, lab_o)
        self.bring_to_front(lifted)
        self.play(o2.animate.set_fill(TEAL_D, opacity=0.85), run_time=0.4)
        shift = k * np.array([S + g - b, 0.0, 0.0])
        self.play(lifted.animate.shift(shift), run_time=1.6)
        check(close(o2.get_corner(DL), P((S + g, b)), 1e-6)
              and close(o2.get_corner(UR), P((S + g + d, a)), 1e-6),
              "the extra layer lands beside the square")
        d_o = _dim(P((S + g, b)), P((S + g + d, b)), "a − b", DOWN * 0.3,
                   YELLOW_B, 28)
        frame = Polygon(*[P(p) for p in _rect(0, 0, S, S)], stroke_color=YELLOW_B,
                        stroke_width=5)
        self.remove(ghost)
        self.play(Create(frame), FadeIn(d_o), run_time=0.9)

        labels = [lab1, lab2, lb1, lb2, lab_o, d_o[1], *[m[1] for m in d_bot],
                  *[m[1] for m in d_lft]]
        _apart(*labels)
        _safe(frame, d_o, d_bot, d_lft, lifted)
        check(_boxes_apart(VGroup(*lifted), frame, 0.3), "lifted square clear")
        cap = caption("(a+b)² + (a−b)²  =  2(a² + b²)", 36)
        self.play(Write(cap), run_time=1.3)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== C15

def _same_pts(p1, p2, tol=1e-6):
    """Same set of points (any order)."""
    p1 = [np.asarray(p, float)[:2] for p in p1]
    p2 = [np.asarray(p, float)[:2] for p in p2]
    return len(p1) == len(p2) and all(
        any(np.linalg.norm(p - q) < tol for q in p2) for p in p1) and all(
        any(np.linalg.norm(p - q) < tol for q in p1) for p in p2)


class C15_BabylonianProduct(Board):
    """An a × b rectangle (a > b) is a b-square and an excess strip a − b
    wide. Cut the strip in half; swing the far half a quarter-turn about
    its top inner corner and slide it onto the top of the b-square. The
    pieces now fill a square of side b + (a−b)/2 = (a+b)/2 except for a
    corner square of side (a−b)/2:
        ab = ((a+b)/2)² − ((a−b)/2)²."""

    def construct(self):
        a, b = 5.2, 2.2
        d = (a - b) / 2
        m = b + d                                        # = (a + b)/2
        F = Frame(-1.35, 6.75, -1.15, 4.35)
        P, k = F.P, F.k
        Sq, H1, H2 = _rect(0, 0, b, b), _rect(b, 0, d, b), _rect(m, 0, d, b)
        pivot = (m, b)

        def turn(p):                                     # quarter-turn, CCW
            x, y = p[0] - pivot[0], p[1] - pivot[1]
            return (pivot[0] - y, pivot[1] + x)

        H2t = [turn(p) for p in H2]
        H2f = [(x - m, y) for (x, y) in H2t]
        check(_same_pts(H2t, _rect(m, b, b, d)), "the turn lays the half level")
        check(_same_pts(H2f, _rect(0, b, b, d)), "and it slides onto the b-square")
        check(_rect_tiling([(0, b, 0, b), (b, m, 0, b), (0, b, b, m), (b, m, b, m)],
                           (0, m, 0, m)), "pieces + corner tile the (a+b)/2-square")
        check(abs(a * b - (m * m - d * d)) < 1e-9, "ab = ((a+b)/2)² − ((a−b)/2)²")
        # the swing stays right of the cut (x ≥ m): it never crosses a piece
        for t in np.linspace(0, PI / 2, 46):
            for p in H2:
                x, y = p[0] - pivot[0], p[1] - pivot[1]
                check(pivot[0] + np.cos(t) * x - np.sin(t) * y >= m - 1e-9,
                      "the swing stays clear of the other pieces")

        rect = F.poly(_rect(0, 0, a, b), BLUE_D)
        lab_ab = tag("ab", 36).move_to(P((a / 2, b / 2)))
        d_a = _dim(P((0, 0)), P((a, 0)), "a", DOWN * 0.3, GREY_A, 30)
        d_b = _dim(P((0, 0)), P((0, b)), "b", LEFT * 0.3, GREY_A, 30)
        self.play(FadeIn(rect), FadeIn(lab_ab), FadeIn(d_a), FadeIn(d_b),
                  run_time=1.2)
        self.hold(0.5)

        # ---- cut off the excess beyond the b-square, then halve it
        cut1 = DashedLine(P((b, -0.04)), P((b, b + 0.14)), color=YELLOW_B,
                          stroke_width=5, dash_length=0.1)
        pS, pE = F.poly(Sq, BLUE_D), F.poly(_rect(b, 0, a - b, b), TEAL_D)
        p1, p2 = F.poly(H1, TEAL_D), F.poly(H2, TEAL_D)
        lab_b2 = tag("b²", 34).move_to(P((b / 2, b / 2)))
        d_sq = _dim(P((0, 0)), P((b, 0)), "b", DOWN * 0.3, GREY_A, 30)
        d_ex = _dim(P((b, 0)), P((a, 0)), "a − b", DOWN * 0.3, TEAL_B, 30)
        self.play(Create(cut1), run_time=0.6)
        self.remove(rect)
        self.add(pS, pE, cut1)
        self.play(FadeOut(lab_ab), FadeIn(lab_b2), FadeOut(d_a), FadeIn(d_sq),
                  FadeIn(d_ex), run_time=0.8)
        cut2 = DashedLine(P((m, -0.04)), P((m, b + 0.14)), color=YELLOW_B,
                          stroke_width=5, dash_length=0.1)
        d_h1 = _dim(P((b, 0)), P((m, 0)), "(a−b)/2", DOWN * 0.3, TEAL_B, 26)
        d_h2 = _dim(P((m, 0)), P((a, 0)), "(a−b)/2", DOWN * 0.3, ORANGE, 26)
        _apart(d_sq[1], d_h1[1], d_h2[1], d_b[1], gap=0.25)
        self.play(Create(cut2), FadeOut(d_ex), run_time=0.6)
        self.remove(pE)
        self.add(p1, p2, cut1, cut2)
        self.play(p2.animate.set_fill(ORANGE), FadeIn(d_h1), FadeIn(d_h2),
                  run_time=0.7)
        self.hold(0.6)

        # ---- swing the far half up about its top inner corner …
        dot = Dot(P(pivot), radius=0.07, color=YELLOW_B)
        arc = Arc(radius=0.75, start_angle=-PI / 2 + 0.2, angle=PI / 2 - 0.4,
                  arc_center=P(pivot), color=YELLOW_B, stroke_width=4)
        arc.add_tip(tip_length=0.18, tip_width=0.18)
        self.play(FadeOut(cut1), FadeOut(cut2), FadeOut(d_h2), FadeIn(dot),
                  Create(arc), run_time=0.6)
        self.bring_to_front(p2)
        self.play(Rotate(p2, angle=PI / 2, about_point=P(pivot)), run_time=1.6)
        check(_same_pts(p2.get_vertices(), [P(p) for p in H2t]), "turned level")
        self.play(FadeOut(arc), FadeOut(dot), run_time=0.3)
        # ---- … and slide it onto the b-square
        self.play(p2.animate.shift(k * np.array([-m, 0.0, 0.0])), run_time=1.3)
        check(_same_pts(p2.get_vertices(), [P(p) for p in H2f]), "landed on top")

        # ---- a square of side (a+b)/2, short of a corner ((a−b)/2)²
        corner = DashedVMobject(Polygon(*[P(p) for p in _rect(b, b, d, d)],
                                        stroke_color=RED_B, stroke_width=4),
                                num_dashes=20)
        frame = Polygon(*[P(p) for p in _rect(0, 0, m, m)], stroke_color=YELLOW_B,
                        stroke_width=5)
        D_b = _dim(P((0, 0)), P((m, 0)), "(a+b)/2", DOWN * 0.3, YELLOW_B, 28)
        D_l = _dim(P((0, 0)), P((0, m)), "(a+b)/2", LEFT * 0.3, YELLOW_B, 28)
        D_c = _dim(P((m, b)), P((m, m)), "(a−b)/2", RIGHT * 0.3, RED_B, 26)
        self.play(FadeOut(d_sq), FadeOut(d_h1), FadeOut(d_b), run_time=0.4)
        self.play(Create(frame), Create(corner), FadeIn(D_b), FadeIn(D_l),
                  FadeIn(D_c), run_time=1.1)
        _safe(frame, D_b, D_l, D_c, pS, p1, p2)
        _apart(D_b[1], D_l[1], D_c[1], lab_b2)
        cap = caption("ab  =  ((a+b)/2)²  −  ((a−b)/2)²", 36)
        self.play(Write(cap), run_time=1.3)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== C20

class C20_TrapezoidDifference(Board):
    """Take the b-square out of a corner of the a-square. The diagonal of
    the a-square cuts the L-shaped rest, a² − b², into two congruent right
    trapezoids (parallel sides a and b, height a − b), mirror images of
    each other. Turn one over — a half-turn in space about a line across
    it — and it is the other one turned half round: slid over and set
    against it, side a against side b, the two make a parallelogram with
    base a + b and height a − b:
        a² − b² = (a+b)(a−b)."""

    def construct(self):
        a, b = 4.0, 1.6
        d = a - b
        c2 = 5.2                                   # turn-over axis x + y = c2
        F = Frame(-1.25, 9.35, -1.15, 5.45)
        P, k = F.P, F.k
        T1 = [(0, 0), (a, 0), (a, d), (d, d)]       # below the diagonal
        T2 = [(0, 0), (d, d), (d, a), (0, a)]       # above it
        T2f = [(c2 - y, c2 - x) for (x, y) in T2]   # mirror image in x + y = c2
        lift = 2 * a - c2                           # slide right, then down
        T2e = [(x + lift, y - (c2 - d)) for (x, y) in T2f]
        T1h = [(a, 0), (a + b, 0), (2 * a, d), (a, d)]   # T1 turned half round
        par = [(0, 0), (a + b, 0), (2 * a, d), (d, d)]
        check(_same_pts(T2, [(y, x) for (x, y) in T1]), "T2 is T1 mirrored")
        check(_same_pts(T1h, [(2 * a - x, d - y) for (x, y) in T1]),
              "T1h is T1 turned half round")
        check(_same_pts(T2e, T1h), "T2 turned over and slid lands on T1h")
        check(abs(abs(area(T1)) + abs(area(T2)) - (a * a - b * b)) < 1e-9,
              "the trapezoids make a² − b²")
        check(abs(abs(area(T1)) + abs(area(T1h)) - abs(area(par))) < 1e-9
              and abs(abs(area(par)) - (a + b) * d) < 1e-9,
              "T1 and T1h make the parallelogram (a+b) × (a−b)")
        check(all(y - x >= -1e-12 for x, y in T2) and all(y - x <= 1e-12 for x, y in T1),
              "the turn-over (which keeps x − y) never crosses T1")
        check(min(y for _, y in T2f) > d + 0.3, "the slide right passes over T1")
        check(min(x for x, _ in T2e) >= a - 1e-12, "the drop stays right of T1")

        # ---- the a-square less a b-square in its corner
        ghost = DashedVMobject(Polygon(*[P(p) for p in _rect(d, d, b, b)],
                                       stroke_color=RED_B, stroke_width=3),
                               num_dashes=16)
        lab_b = tag("b²", 30, RED_B).move_to(P((d + b / 2, d + b / 2)))
        Lsh = F.poly([(0, 0), (a, 0), (a, d), (d, d), (d, a), (0, a)], BLUE_D)
        lab_L = tag("a² − b²", 34).move_to(P((2.85, 0.95)))
        d_a1 = _dim(P((0, 0)), P((a, 0)), "a", DOWN * 0.3, GREY_A, 30)
        d_a2 = _dim(P((0, 0)), P((0, a)), "a", LEFT * 0.3, GREY_A, 30)
        d_b = _dim(P((d, a)), P((a, a)), "b", UP * 0.3, RED_B, 28)
        self.play(FadeIn(Lsh), Create(ghost), FadeIn(lab_b), FadeIn(lab_L),
                  FadeIn(d_a1), FadeIn(d_a2), FadeIn(d_b), run_time=1.4)
        self.hold(0.6)

        # ---- cut along the diagonal: two trapezoids
        cut = DashedLine(P((0, 0)), P((d, d)), color=YELLOW_B, stroke_width=5,
                         dash_length=0.1)
        p1, p2 = F.poly(T1, BLUE_D), F.poly(T2, BLUE_D)
        self.play(Create(cut), FadeOut(lab_L), run_time=0.8)
        self.remove(Lsh)
        self.add(p1, p2, cut)
        self.play(p2.animate.set_fill(ORANGE), run_time=0.5)
        self.hold(0.4)

        # ---- turn the upper one over about a line across it
        ax0, ax1 = (0.95, c2 - 0.95), (2.6, c2 - 2.6)
        axis = DashedLine(P(ax0), P(ax1), color=YELLOW_B, stroke_width=4,
                          dash_length=0.1)
        self.play(FadeOut(cut), FadeOut(d_b), FadeOut(ghost), FadeOut(lab_b),
                  Create(axis), run_time=0.7)
        self.bring_to_front(p2)
        self.play(_turn_over(p2, P((c2, 0)), P((0, c2))), run_time=2.0)
        check(_same_pts(p2.get_vertices(), [P(p) for p in T2f]), "turned over")
        self.play(FadeOut(axis), run_time=0.3)

        # ---- slide it over and set it down beside the other
        self.play(p2.animate.shift(k * RIGHT * lift), run_time=1.1)
        self.play(p2.animate.shift(k * DOWN * (c2 - d)), run_time=1.1)
        check(_same_pts(p2.get_vertices(), [P(p) for p in T1h]), "set down")

        # ---- centre it: a parallelogram, base a + b and height a − b
        grp = VGroup(p1, p2)
        v = np.array([0.0, 0.62 - grp.get_center()[1], 0.0])
        self.play(FadeOut(d_a1), FadeOut(d_a2), grp.animate.shift(v),
                  run_time=0.9)

        def Q(p):
            return P(p) + v

        check(_same_pts(p1.get_vertices(), [Q(p) for p in T1])
              and _same_pts(p2.get_vertices(), [Q(p) for p in T1h]),
              "moved whole")
        frame = Polygon(*[Q(p) for p in par], stroke_color=YELLOW_B, stroke_width=5)
        D_base = _dim(Q((0, 0)), Q((a + b, 0)), "a + b", DOWN * 0.3, YELLOW_B, 30)
        D_h = _dim(Q((0, 0)), Q((0, d)), "a − b", LEFT * 0.3, YELLOW_B, 30)
        # the top edge is the short side b of one trapezoid and the long
        # side a of the other, as the base is a and then b
        t_b = _dim(Q((d, d)), Q((a, d)), "b", UP * 0.3, BLUE_B, 28)
        t_a = _dim(Q((a, d)), Q((2 * a, d)), "a", UP * 0.3, ORANGE, 28)
        self.play(Create(frame), FadeIn(D_base), FadeIn(D_h), FadeIn(t_b),
                  FadeIn(t_a), run_time=1.0)
        _safe(frame, D_base, D_h, t_a, t_b, p1, p2)
        _apart(D_base[1], D_h[1], t_a[1], t_b[1])
        cap = caption("a² − b²  =  (a+b)(a−b)", 36)
        self.play(Write(cap), run_time=1.2)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== J31

def _capsule(p, q, color, width=3, pad=0.22):
    """A rounded outline around the two dots at screen points p, q."""
    p, q = to3(p), to3(q)
    L = float(np.linalg.norm(q - p))
    c = RoundedRectangle(width=L + 2 * pad, height=2 * pad, corner_radius=pad,
                         stroke_color=color, stroke_width=width)
    c.rotate(float(np.arctan2(*(q - p)[1::-1])))
    return c.move_to((p + q) / 2)


class J31_ParityPairs(Board):
    """An odd number of dots is pairs with one dot over. Two odd numbers:
    turn the second half round, and its spare dot lands over the spare of
    the first — they pair up, so odd + odd is even. An odd number of rows
    of an odd number of dots: every row is pairs and one spare; the spares
    stand in one column, an odd number of them — pairs and one over. So
    odd × odd is odd. Drawn as 7 + 5 and 5 × 7."""

    def construct(self):
        s, r = 0.6, 0.11
        m1, k1 = 3, 2                                  # 7 = 2·3+1, 5 = 2·2+1
        x0, y1 = -5.4, 0.05

        def PA(c, row):
            return np.array([x0 + c * s, y1 + row * s, 0.0])

        A = [(c, 0) for c in range(m1 + 1)] + [(c, 1) for c in range(m1)]
        B = [(m1 + 2 + c, 0) for c in range(k1 + 1)] + \
            [(m1 + 2 + c, 1) for c in range(k1)]
        cB = np.array([x0 + (m1 + 2 + k1 / 2) * s, y1 + s / 2, 0.0])
        slide = -2 * s
        Bend = [(2 * (m1 + 2) + k1 - c + slide / s, 1 - row) for (c, row) in B]
        check(len(A) == 2 * m1 + 1 and len(B) == 2 * k1 + 1, "two odd numbers")
        check(sorted(A + [(round(c), row) for (c, row) in Bend])
              == sorted((c, row) for c in range(m1 + k1 + 1) for row in (0, 1)),
              "turned and slid, the two make full pairs: an even number")

        dA = VGroup(*[Dot(PA(*q), radius=r, color=GREY_A) for q in A])
        dB = VGroup(*[Dot(PA(*q), radius=r, color=GREY_A) for q in B])
        capA = VGroup(*[_capsule(PA(c, 0), PA(c, 1), BLUE_B) for c in range(m1)])
        capB = VGroup(*[_capsule(PA(m1 + 2 + c, 0), PA(m1 + 2 + c, 1), BLUE_B)
                        for c in range(k1)])
        spA = [d for d, q in zip(dA, A) if q == (m1, 0)][0]
        spB = [d for d, q in zip(dB, B) if q == (m1 + 2 + k1, 0)][0]
        lA = tag("2m+1", 28).next_to(VGroup(dA, capA), DOWN, buff=0.35)
        lB = tag("2k+1", 28).next_to(VGroup(dB, capB), DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dA], lag_ratio=0.08),
                  run_time=0.9)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dB], lag_ratio=0.08),
                  run_time=0.8)
        self.play(*[Create(c) for c in capA], *[Create(c) for c in capB],
                  spA.animate.set_color(ORANGE), spB.animate.set_color(ORANGE),
                  FadeIn(lA), FadeIn(lB), run_time=1.0)
        self.hold(0.6)

        # ---- turn the second one half round and push it against the first
        gB = VGroup(dB, capB)
        self.play(FadeOut(lB), run_time=0.3)
        self.play(Rotate(gB, PI, about_point=cB), run_time=1.4)
        self.play(gB.animate.shift(RIGHT * slide), run_time=1.0)
        check(_lands_on(dB, [PA(c, row) for (c, row) in Bend]), "landed")
        check(close(spB.get_center(), PA(m1, 1)), "the spare lands on the spare")
        cap_new = _capsule(PA(m1, 0), PA(m1, 1), YELLOW_B, width=4)
        lsum = tag("2(m+k+1)", 28, YELLOW_B).next_to(
            VGroup(dA, dB), DOWN, buff=0.35)
        self.play(Create(cap_new), FadeOut(lA), FadeIn(lsum), run_time=0.8)
        self.hold(0.6)

        # ---- an odd number of rows of an odd number of dots
        M, K = 2, 3                                    # 5 rows of 7
        R, Cn = 2 * M + 1, 2 * K + 1
        X0, Y0 = 1.5, -0.85

        def PB(c, row):
            return np.array([X0 + c * s, Y0 + row * s, 0.0])

        grid = {(c, row): Dot(PB(c, row), radius=r, color=GREY_A)
                for row in range(R) for c in range(Cn)}
        pairs_h = [((2 * j, row), (2 * j + 1, row)) for row in range(R)
                   for j in range(K)]
        spare = [(Cn - 1, row) for row in range(R)]
        pairs_v = [((Cn - 1, 2 * j), (Cn - 1, 2 * j + 1)) for j in range(M)]
        left = (Cn - 1, R - 1)
        cells = [q for pr in pairs_h + pairs_v for q in pr] + [left]
        check(sorted(cells) == sorted(grid) and len(set(cells)) == len(cells),
              "row pairs, column pairs and one dot use every dot once")
        check(R * Cn == 2 * (len(pairs_h) + len(pairs_v)) + 1, "odd × odd is odd")
        G = VGroup(*grid.values())
        top = _dim(PB(0, R - 1) + UP * r + LEFT * r, PB(Cn - 1, R - 1) + UP * r
                   + RIGHT * r, "2m+1", UP * 0.3, GREY_A, 28)
        side = _dim(PB(0, 0) + LEFT * r + DOWN * r, PB(0, R - 1) + LEFT * r + UP * r,
                    "2k+1", LEFT * 0.3, GREY_A, 28)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in G], lag_ratio=0.01),
                  FadeIn(top), FadeIn(side), run_time=1.2)
        caps_h = VGroup(*[_capsule(PB(*p), PB(*q), BLUE_B) for p, q in pairs_h])
        self.play(LaggedStart(*[Create(c) for c in caps_h], lag_ratio=0.05),
                  *[grid[q].animate.set_color(ORANGE) for q in spare], run_time=1.6)
        caps_v = VGroup(*[_capsule(PB(*p), PB(*q), ORANGE) for p, q in pairs_v])
        self.play(*[Create(c) for c in caps_v], run_time=0.8)
        ring = Circle(radius=0.24, stroke_color=YELLOW_B, stroke_width=5).move_to(
            PB(*left))
        self.play(Create(ring), grid[left].animate.set_color(YELLOW_B),
                  Flash(PB(*left), color=YELLOW_B, line_length=0.18,
                        flash_radius=0.36), run_time=0.9)

        _safe(dA, dB, capA, capB, cap_new, lsum, G, caps_h, caps_v, ring, top, side)
        _apart(lsum, top[1], side[1])
        check(_boxes_apart(VGroup(dA, dB, capA, capB, cap_new, lsum),
                           VGroup(G, side, top), 0.3), "the two pictures apart")
        cap = caption("odd + odd  =  even,        odd × odd  =  odd", 36)
        self.play(Write(cap), run_time=1.4)
        _all_safe(self, cap)
        self.hold(2.2)


# ====================================================================== J10

class J10_OddSquareTriples(Board):
    """Every odd number 2m + 1 is an L of width one that grows the m-square
    into the (m+1)-square. When the odd number is itself a square,
    (2k+1)², that L belongs to m = ((2k+1)² − 1)/2 = 2k² + 2k. So cut the
    (2k+1)-square into strips and lay them round the (2k²+2k)-square: the
    next square is complete,
        (2k+1)² + (2k²+2k)² = (2k²+2k+1)²,
    one Pythagorean triple for every k. Drawn for k = 2: 5² + 12² = 13²."""

    def construct(self):
        k = 2
        o, m = 2 * k + 1, 2 * k * k + 2 * k            # 5, 12
        u = 0.4
        X0, Y0 = -5.6, -2.2

        def CC(x, y):                                  # centre of cell (x, y)
            return np.array([X0 + (x + 0.5) * u, Y0 + (y + 0.5) * u, 0.0])

        def CP(x, y):                                  # cell corner (x, y)
            return np.array([X0 + x * u, Y0 + y * u, 0.0])

        sq = {(m + 2 + i, m - o + 1 + j) for i in range(o) for j in range(o)}
        rows = {j: [(m + 2 + i, m - o + 1 + j) for i in range(o)] for j in range(o)}
        pcs = {"R0": rows[0], "R1": rows[1], "R2a": rows[2][2:], "R2b": rows[2][:2],
               "R3": rows[3], "R4": rows[4]}
        L = {(m, y) for y in range(m + 1)} | {(x, m) for x in range(m)}
        block = {(x, y) for x in range(m) for y in range(m)}
        check(len(L) == 2 * m + 1 == o * o, "the L of width one has (2k+1)² cells")
        check(_tiles_once([block, L], {(x, y) for x in range(m + 1)
                                       for y in range(m + 1)}),
              "block + L = the (m+1)-square")
        check(_tiles_once(list(pcs.values()), sq), "the strips cut the (2k+1)-square")
        for kk in range(1, 9):
            a_, b_ = 2 * kk + 1, 2 * kk * kk + 2 * kk
            check(a_ * a_ + b_ * b_ == (b_ + 1) ** 2, "a triple for every k")

        # the moves, cell by cell: ("shift", dx, dy) or ("turn", pivot cell, angle)
        plan = [(("R4",), "shift", (-(m + 2), 0)),
                (("R3", "R2b"), "shift", (0, 1)),
                (("R3",), "shift", (-(m + 2) + 5, 0)),
                (("R2b",), "shift", (0, 1)),
                (("R2b",), "shift", (-(m + 2) + 10, 0)),
                (("R0",), "turn", ((m + 6, m - o + 1), PI / 2)),
                (("R0",), "shift", (-6, -(m - o + 1) + 4)),
                (("R1",), "turn", ((m + 6, m - o + 2), PI / 2)),
                (("R1",), "shift", (-6, 0)),
                (("R2a",), "turn", ((m + 4, m - o + 3), PI / 2)),
                (("R2a",), "shift", (-4, 0))]
        cur = {key: [np.array(c, float) for c in v] for key, v in pcs.items()}

        def moved(pts, kind, arg):
            if kind == "shift":
                return [p + np.array(arg, float) for p in pts]
            c, ang = np.array(arg[0], float), arg[1]
            cs, sn = np.cos(ang), np.sin(ang)
            return [c + np.array([cs * (p - c)[0] - sn * (p - c)[1],
                                  sn * (p - c)[0] + cs * (p - c)[1]]) for p in pts]

        expect = []
        for names, kind, arg in plan:
            for t in np.linspace(0.0, 1.0, 41)[1:]:
                part = (("shift", tuple(np.array(arg) * t)) if kind == "shift"
                        else ("turn", (arg[0], arg[1] * t)))
                stat = [q for key, v in cur.items() if key not in names for q in v] + \
                       [np.array(c, float) for c in block]
                for nm in names:
                    for p in moved(cur[nm], *part):
                        check(min(np.linalg.norm(p - q) for q in stat) > 1 - 1e-9,
                              "a moving strip never runs into another cell")
            for nm in names:
                cur[nm] = moved(cur[nm], kind, arg)
            expect.append({nm: [tuple(np.round(p, 9)) for p in cur[nm]] for nm in names})
        check({tuple(int(round(c)) for c in p) for v in cur.values() for p in v} == L,
              "the strips end exactly on the L")

        # ---- the (2k²+2k)-square and the (2k+1)-square
        blk = Rectangle(width=m * u, height=m * u, fill_color=BLUE_D,
                        fill_opacity=0.8, stroke_color=WHITE, stroke_width=2
                        ).move_to(CP(m / 2, m / 2))
        grid = VGroup(*[Line(CP(i, 0), CP(i, m), stroke_color=WHITE,
                             stroke_width=0.8, stroke_opacity=0.35) for i in range(1, m)],
                      *[Line(CP(0, j), CP(m, j), stroke_color=WHITE,
                             stroke_width=0.8, stroke_opacity=0.35) for j in range(1, m)])
        lab_blk = tag("(2k² + 2k)²", 34).move_to(CP(m / 2, m / 2))
        d_blk = _dim(CP(0, 0), CP(m, 0), "2k² + 2k", DOWN * 0.26, BLUE_B, 26, gap=0.1)
        self.play(FadeIn(blk), Create(grid), run_time=1.0)
        self.play(FadeIn(lab_blk), FadeIn(d_blk), run_time=0.6)
        shade = {"R0": ORANGE, "R1": GOLD_E, "R2a": ORANGE, "R2b": ORANGE,
                 "R3": GOLD_E, "R4": ORANGE}
        piece = {key: VGroup(*[cell(x, y, u, CP(0, 0), shade[key], op=0.9)
                               for (x, y) in v]) for key, v in pcs.items()}
        allp = VGroup(*piece.values())
        x1, y1 = m + 2, m - o + 1
        d_sq = _dim(CP(x1 + o, y1), CP(x1 + o, y1 + o), "2k+1", RIGHT * 0.26,
                    ORANGE, 26, gap=0.12)
        lab_sq = tag("(2k+1)²", 28, ORANGE).next_to(CP(x1 + o / 2, y1), DOWN,
                                                    buff=0.22)
        self.play(FadeIn(allp), FadeIn(d_sq), FadeIn(lab_sq), run_time=0.9)
        self.hold(0.6)

        # ---- cut into strips: rows, the middle one in 3 + 2
        cut = Line(CP(x1 + 2, y1 + 1.85), CP(x1 + 2, y1 + 3.15), color=YELLOW,
                   stroke_width=8)
        self.play(FadeOut(lab_sq), FadeOut(d_sq), Create(cut), run_time=0.6)
        self.hold(0.5)
        self.play(FadeOut(cut), run_time=0.3)

        # ---- lay the strips round the block
        for (names, kind, arg), exp in zip(plan, expect):
            if kind == "shift":
                anims = [piece[nm].animate.shift(u * np.array([arg[0], arg[1], 0.0]))
                         for nm in names]
            else:
                anims = [Rotate(piece[nm], arg[1], about_point=CC(*arg[0]))
                         for nm in names]
            self.play(*anims, run_time=0.75 if kind == "shift" else 0.85)
            for nm in names:
                check(_lands_on(piece[nm], [CC(*p) for p in exp[nm]]),
                      f"strip {nm} landed")

        # ---- the next square
        frame = Rectangle(width=(m + 1) * u, height=(m + 1) * u,
                          stroke_color=YELLOW_B, stroke_width=5
                          ).move_to(CP((m + 1) / 2, (m + 1) / 2))
        d_one = _dim(CP(m, 0), CP(m + 1, 0), "1", DOWN * 0.26, ORANGE, 26, gap=0.1)
        d_all = _dim(CP(m + 1, 0), CP(m + 1, m + 1), "2k² + 2k + 1", RIGHT * 0.26,
                     YELLOW_B, 26, gap=0.12)
        lab_L = tag("(2k+1)²", 28, ORANGE).next_to(CP(m / 2, m + 1), UP, buff=0.12)
        Lline = Polygon(CP(0, m), CP(m, m), CP(m, 0), CP(m + 1, 0), CP(m + 1, m + 1),
                        CP(0, m + 1), stroke_color=ORANGE, stroke_width=5)
        self.play(Create(Lline), FadeIn(lab_L), run_time=0.8)
        self.play(Create(frame), FadeIn(d_one), FadeIn(d_all), run_time=1.0)
        trip = VGroup(*[tag(f"{2 * kk + 1}²  +  {2 * kk * kk + 2 * kk}²  =  "
                            f"{2 * kk * kk + 2 * kk + 1}²", 26,
                            YELLOW_B if kk == k else GREY_A)
                        for kk in range(1, 5)], tag("…", 26, GREY_A))
        trip.arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([4.55, 0.75, 0])
        self.play(LaggedStart(*[FadeIn(t_) for t_ in trip], lag_ratio=0.2),
                  run_time=1.2)
        _safe(blk, frame, d_blk, d_one, d_all, lab_L, trip, allp, Lline)
        _apart(d_blk[1], d_one[1], d_all[1], lab_L, trip, lab_blk)
        check(_boxes_apart(d_all, trip, 0.3), "readout clear of the bracket")
        cap = caption("(2k+1)² + (2k²+2k)²  =  (2k²+2k+1)²", 36)
        self.play(Write(cap), run_time=1.4)
        _all_safe(self, cap)
        self.hold(2.2)
