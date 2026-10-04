# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w1d.py — proofs without words, 2D (manim):
#     C8   (a+b)² − (a−b)² = 4ab: four rectangles turned by quarter-turns
#     C10  square of a trinomial, nine tiles
#     C12  (a+b)(c+d), one rectangle cut twice
#     C13  factoring x² + (p+q)x + pq with tiles
#     C14  al-Khwārizmī's completion of x² + 10x = 39
#     C19  ab = ba, a dot array turned a quarter-turn
#     D7   AM–GM: four rectangles round a square hole
#     D8   a² + b² ≥ 2ab: two rectangles on the two squares
#     D11  among rectangles of one perimeter the square encloses most
#     D21  Heron's shortest path by reflection
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Lengths in the algebra pictures are unequal non-integers drawn to scale,
# so no two different lengths can be mistaken for one another. Each scene
# checks its own geometry with check(...) before it animates.


# ---------------------------------------------------------------- helpers

def _rect(x0, y0, w, h):
    """Axis-aligned rectangle, counter-clockwise from the bottom-left."""
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _box(r):
    xs = [float(p[0]) for p in r]
    ys = [float(p[1]) for p in r]
    return min(xs), max(xs), min(ys), max(ys)


def _tiling(rects, region, tol=1e-9):
    """True when the axis-aligned rectangles lie in the axis-aligned region,
    overlap nowhere, and their areas add up to the region's: an exact,
    gap-free tiling."""
    X0, X1, Y0, Y1 = _box(region)
    tot = 0.0
    for r in rects:
        x0, x1, y0, y1 = _box(r)
        if x0 < X0 - tol or x1 > X1 + tol or y0 < Y0 - tol or y1 > Y1 + tol:
            return False
        tot += (x1 - x0) * (y1 - y0)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = _box(rects[i]), _box(rects[j])
            w = min(a[1], b[1]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[2], b[2])
            if w > tol and h > tol:
                return False
    return abs(tot - (X1 - X0) * (Y1 - Y0)) < 1e-7


def _same_pts(p1, p2, tol=1e-6):
    """Same set of points (any order)."""
    p1 = [np.asarray(p, float)[:2] for p in p1]
    p2 = [np.asarray(p, float)[:2] for p in p2]
    return len(p1) == len(p2) and all(
        any(np.linalg.norm(p - q) < tol for q in p2) for p in p1)


def _turn(p, c, n):
    """Point p turned n quarter-turns counter-clockwise about c (math)."""
    x, y = p[0] - c[0], p[1] - c[1]
    for _ in range(n % 4):
        x, y = -y, x
    return (x + c[0], y + c[1])


def _inset(r, d):
    """Axis-aligned rectangle r shrunk by d on every side."""
    x0, x1, y0, y1 = _box(r)
    return _rect(x0 + d, y0 + d, x1 - x0 - 2 * d, y1 - y0 - 2 * d)


def _verts(poly):
    return [np.array(v) for v in poly.get_vertices()]


def _dashed_outline(pts, color, width=3, dashes=24):
    return DashedVMobject(Polygon(*[to3(p) for p in pts], stroke_color=color,
                                  stroke_width=width), num_dashes=dashes)


def _on_base(s, x, base_y, size=28, color=WHITE):
    """A label centred on x whose baseline sits at base_y, so a row of
    labels shares one baseline whatever their ascenders and descenders."""
    t = Text("M" + s, font_size=size, color=color)
    m = t.submobjects[0]
    base = m.get_bottom()[1]
    t.remove(m)
    t.shift(np.array([x - t.get_center()[0], base_y - base, 0.0]))
    return t


def _ctag(s, size=26, color=WHITE, t2c=None):
    """A label in one Text (one font run, one baseline) whose substrings are
    recoloured glyph by glyph afterwards."""
    t = Text(s, font_size=size, color=color)
    flat = s.replace(" ", "").replace("\n", "")
    check(len(t.submobjects) == len(flat), f"one glyph per character in {s!r}")
    for sub, col in (t2c or {}).items():
        key = sub.replace(" ", "")
        i = flat.find(key)
        while i != -1:
            for g in t.submobjects[i:i + len(key)]:
                g.set_color(col)
            i = flat.find(key, i + len(key))
    return t


def _lab(s, p, q, out, size=28, color=WHITE, gap=0.14):
    """Label s beside the screen segment pq, on the side `out`. Under or
    over a horizontal side the labels share a baseline."""
    m = (to3(p) + to3(q)) / 2
    o = to3(out)
    if abs(o[0]) < 1e-9 and o[1] < 0:            # under
        cap = Text("M", font_size=size).height
        return _on_base(s, m[0], m[1] - gap - cap, size, color)
    if abs(o[0]) < 1e-9 and o[1] > 0:            # over
        probe = Text("Mg", font_size=size)
        depth = (probe.submobjects[0].get_bottom()[1]
                 - probe.submobjects[1].get_bottom()[1])
        return _on_base(s, m[0], m[1] + gap + depth, size, color)
    return tag(s, size, color).next_to(m, o, buff=gap)


def _dim(p, q, s, out, off=0.5, size=26, color=YELLOW_B, gap=0.1,
         width=2.5):
    """Dimension line beside the screen segment pq, moved `off` towards
    `out`, with end ticks; returns (line group, label beyond it)."""
    p, q, n = to3(p), to3(q), to3(out)
    n = n / np.linalg.norm(n)
    p1, q1 = p + off * n, q + off * n
    t = 0.09 * n
    g = VGroup(Line(p1, q1, color=color, stroke_width=width),
               Line(p1 - t, p1 + t, color=color, stroke_width=width),
               Line(q1 - t, q1 + t, color=color, stroke_width=width))
    lab = tag(s, size, color).next_to((p1 + q1) / 2, n, buff=gap)
    return g, lab


def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centroid while the
    centroid slides — carrying the screen points src onto dst (same order).
    Fails the render unless one rotation + translation lands every point."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = np.arctan2(*(src[1] - src[0])[1::-1])
        a1 = np.arctan2(*(dst[1] - dst[0])[1::-1])
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    cs, sn = np.cos(turn), np.sin(turn)
    R = np.array([[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]])
    check(all(close(m1 + R @ (p - m0), q, 1e-6) for p, q in zip(src, dst)),
          "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _spin(mob, angle, about, color=None, **kw):
    """Rigid rotation about a fixed screen point; the fill may shade to
    `color` on the way (shape and size never change)."""
    start = mob.copy()
    c0 = start.get_fill_color()
    about = to3(about)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * angle, about_point=about))
        if color is not None:
            m.set_fill(interpolate_color(c0, ManimColor(color), alpha))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _safe(*mobs, tol=0.03):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for m in mobs:
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area: {type(m).__name__} "
              f"[{lo[0]:.2f},{hi[0]:.2f}]x[{lo[1]:.2f},{hi[1]:.2f}]")


def _apart(*mobs, gap=0.04):
    """Fail the render if any two labels' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            a0, a1 = mobs[i].get_critical_point(DL), mobs[i].get_critical_point(UR)
            b0, b1 = mobs[j].get_critical_point(DL), mobs[j].get_critical_point(UR)
            sep = (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
                   or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])
            check(sep, f"labels {i} and {j} do not overlap")


def _inside(m, poly_pts, pad=0.05):
    """Fail the render unless label m sits inside the screen box poly_pts."""
    x0, x1, y0, y1 = _box(poly_pts)
    lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
    check(lo[0] >= x0 + pad and hi[0] <= x1 - pad and lo[1] >= y0 + pad
          and hi[1] <= y1 - pad, "label fits inside its piece")


# ============================================== C. ALGEBRAIC IDENTITIES

class C8_FourProducts(Board):
    """Four a × b rectangles, each a quarter-turn of the one before about
    the centre of the (a + b)-square, fill that square except for a square
    hole. Each side of the square is one long side a and one short side b;
    each side of the hole is a long side a less a short side b:
        (a + b)² − (a − b)² = 4ab."""

    def construct(self):
        a, b = 3.9, 1.4
        s, d = a + b, a - b
        F = Frame(-0.65, s + 0.65, -0.65, s + 0.65)
        P = F.P
        c = (s / 2, s / 2)
        Rm = [_rect(0, 0, a, b), _rect(a, 0, b, a), _rect(b, a, a, b),
              _rect(0, b, b, a)]
        Hm = _rect(b, b, d, d)
        for n in range(4):
            check(_same_pts([_turn(p, c, n) for p in Rm[0]], Rm[n]),
                  f"rectangle {n + 1} is rectangle 1 turned {n} quarter-turns")
        check(_tiling(Rm + [Hm], _rect(0, 0, s, s)),
              "four rectangles and the hole tile the (a+b)-square")
        check(abs(area(Hm) - d * d) < 1e-12
              and abs(sum(area(r) for r in Rm) - 4 * a * b) < 1e-12,
              "hole (a−b)², rectangles 4ab")

        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D]
        # outer sides of each rectangle: (from, to, label, outward)
        sides = [[((0, 0), (a, 0), "a", DOWN), ((0, 0), (0, b), "b", LEFT)],
                 [((a, 0), (s, 0), "b", DOWN), ((s, 0), (s, a), "a", RIGHT)],
                 [((s, a), (s, s), "b", RIGHT), ((b, s), (s, s), "a", UP)],
                 [((0, s), (b, s), "b", UP), ((0, b), (0, s), "a", LEFT)]]

        def labels(n):
            return VGroup(*[_lab(t, P(p), P(q), o, 30) for p, q, t, o in sides[n]])

        def inner(n):
            x0, x1, y0, y1 = _box(Rm[n])
            return tag("ab", 32).move_to(P(((x0 + x1) / 2, (y0 + y1) / 2)))

        r = F.poly(Rm[0], cols[0])
        rects, labs, ins = [r], [labels(0)], [inner(0)]
        self.play(FadeIn(r), run_time=0.9)
        self.play(FadeIn(labs[0]), FadeIn(ins[0]), run_time=0.6)
        self.hold(0.4)

        # three quarter-turns about the centre of the square to be
        ghost = _dashed_outline([P(p) for p in _rect(0, 0, s, s)], GREY_B, 2, 48)
        cdot = Dot(P(c), radius=0.06, color=YELLOW_B)
        deg = tag("90°", 22, YELLOW_B).move_to(P(c) + DOWN * 0.3)
        self.play(Create(ghost), FadeIn(cdot), FadeIn(deg), run_time=0.7)
        for n in range(1, 4):
            arc = Arc(radius=0.62, start_angle=(n - 1) * PI / 2 - PI / 4,
                      angle=PI / 2, arc_center=P(c), color=YELLOW_B,
                      stroke_width=4)
            arc.add_tip(tip_length=0.16, tip_width=0.16)
            self.play(Create(arc), run_time=0.4)
            cp = rects[-1].copy()
            self.add(cp)
            self.play(_spin(cp, PI / 2, P(c), cols[n]), run_time=1.3)
            check(_same_pts(_verts(cp), [P(p) for p in Rm[n]]),
                  f"rectangle {n + 1} lands in its place")
            rects.append(cp)
            labs.append(labels(n))
            ins.append(inner(n))
            self.play(FadeOut(arc), FadeIn(labs[n]), FadeIn(ins[n]),
                      run_time=0.5)
        self.play(FadeOut(cdot), FadeOut(deg), FadeOut(ghost), run_time=0.4)

        # the hole: a long side less a short side, all round
        hole = F.poly(Hm, RED_D, 0.35, stroke_width=0)
        hline = _dashed_outline([P(p) for p in Hm], RED_B, 4, 28)
        hlab = tag("(a−b)²", 32).move_to(P(c))
        dline, dlab = _dim(P((b, b)), P((a, b)), "a − b", UP, off=0.32,
                           size=26, color=RED_B)
        self.play(FadeIn(hole), Create(hline), run_time=0.8)
        self.play(FadeIn(hlab), Create(dline), FadeIn(dlab), run_time=0.7)
        outer = Polygon(*[P(p) for p in _rect(0, 0, s, s)],
                        stroke_color=YELLOW_B, stroke_width=5)
        self.play(Create(outer), run_time=0.8)

        _inside(hlab, [P(p) for p in Hm])
        _apart(hlab, dlab)
        _apart(*[m for g in labs for m in g])
        for m, r_ in zip(ins, Rm):
            _inside(m, [P(p) for p in r_])
        _safe(*labs, *ins, outer)
        self.play(Write(caption("(a+b)² − (a−b)²  =  4ab", 36)))
        self.hold(2.2)


class C12_TwoSums(Board):
    """An (a + b) × (c + d) rectangle. One cut across, at height c, makes a
    c-strip and a d-strip, each a + b long: (a+b)c + (a+b)d. One cut up, at
    a, halves both strips at once:
        (a + b)(c + d) = ac + bc + ad + bd."""

    def construct(self):
        a, b, c, d = 3.6, 1.9, 2.5, 1.2
        W, H = a + b, c + d
        F = Frame(-1.45, W + 0.3, -1.1, H + 0.3)
        P = F.P
        strips = [_rect(0, 0, W, c), _rect(0, c, W, d)]
        tiles = {"ac": _rect(0, 0, a, c), "bc": _rect(a, 0, b, c),
                 "ad": _rect(0, c, a, d), "bd": _rect(a, c, b, d)}
        check(_tiling(strips, _rect(0, 0, W, H)), "two strips fill the rectangle")
        check(_tiling(list(tiles.values()), _rect(0, 0, W, H)),
              "four tiles fill the rectangle")
        check(abs(sum(area(t) for t in tiles.values()) - (a + b) * (c + d)) < 1e-12,
              "ac + bc + ad + bd = (a+b)(c+d)")

        outer = Polygon(*[P(p) for p in _rect(0, 0, W, H)],
                        stroke_color=YELLOW_B, stroke_width=5)
        dw, lw = _dim(P((0, 0)), P((W, 0)), "a + b", DOWN, off=0.8)
        dh, lh = _dim(P((0, 0)), P((0, H)), "c + d", LEFT, off=0.78)
        self.play(Create(outer), Create(dw), Create(dh), FadeIn(lw), FadeIn(lh),
                  run_time=1.3)
        self.hold(0.3)

        # the cut across: a c-strip and a d-strip
        s_c, s_d = F.poly(strips[0], BLUE_D), F.poly(strips[1], TEAL_D)
        cut1 = Line(P((0, c)), P((W, c)), color=WHITE, stroke_width=4)
        lc = _lab("c", P((0, 0)), P((0, c)), LEFT, 30)
        ld = _lab("d", P((0, c)), P((0, H)), LEFT, 30)
        in_c = tag("(a+b)c", 32).move_to(P((W / 2, c / 2)))
        in_d = tag("(a+b)d", 32).move_to(P((W / 2, c + d / 2)))
        self.play(FadeIn(s_c), FadeIn(s_d), Create(cut1), run_time=0.9)
        self.add(outer)
        self.play(FadeIn(lc), FadeIn(ld), FadeIn(in_c), FadeIn(in_d),
                  run_time=0.7)
        self.hold(0.9)

        # the cut up halves both strips at once
        cols = {"ac": BLUE_D, "bc": PURPLE_B, "ad": TEAL_D, "bd": ORANGE}
        tl = {k: F.poly(v, cols[k]) for k, v in tiles.items()}
        cut2 = Line(P((a, 0)), P((a, H)), color=WHITE, stroke_width=4)
        la = _lab("a", P((0, 0)), P((a, 0)), DOWN, 30)
        lb = _lab("b", P((a, 0)), P((W, 0)), DOWN, 30)
        self.play(Create(cut2), FadeIn(la), FadeIn(lb),
                  FadeOut(in_c), FadeOut(in_d), run_time=0.8)
        self.remove(s_c, s_d)
        self.add(*tl.values(), cut1, cut2, outer)
        names = {}
        for k, v in tiles.items():
            x0, x1, y0, y1 = _box(v)
            names[k] = tag(k, 34).move_to(P(((x0 + x1) / 2, (y0 + y1) / 2)))
        self.play(*[tl[k].animate.set_fill(cols[k]) for k in tl],
                  *[FadeIn(names[k]) for k in names], run_time=0.9)
        self.remove(cut1, cut2)
        self.hold(0.8)

        for k, v in tiles.items():
            _inside(names[k], [P(p) for p in v])
        _apart(la, lb, lw, lc, ld, lh)
        _safe(outer, dw, dh, lw, lh, la, lb, lc, ld)
        self.play(Write(caption("(a+b)(c+d)  =  ac + bc + ad + bd", 36)))
        self.hold(2.2)


class C10_TrinomialSquare(Board):
    """The (a + b + c)-square cut at a and a + b both ways: nine tiles. The
    three on the diagonal are a², b², c². The other six pair off across the
    diagonal: folding the square over its diagonal carries each tile onto a
    congruent partner, so they come as 2ab, 2bc, 2ca."""

    def construct(self):
        a, b, c = 3.1, 1.9, 1.2
        L = [a, b, c]
        nm = ["a", "b", "c"]
        s = a + b + c
        cut = [0.0, a, a + b, s]
        F = Frame(-0.75, s + 0.3, -1.4, s + 0.25)
        P = F.P

        def T(i, j):            # column i (x), row j (y)
            return _rect(cut[i], cut[j], L[i], L[j])

        allt = [T(i, j) for i in range(3) for j in range(3)]
        check(_tiling(allt, _rect(0, 0, s, s)), "nine tiles fill the square")
        for i in range(3):
            for j in range(3):
                check(_same_pts([(p[1], p[0]) for p in T(i, j)], T(j, i)),
                      "the diagonal mirror swaps tile (i, j) and tile (j, i)")
        check(abs(sum(area(t) for t in allt)
                  - (a * a + b * b + c * c + 2 * a * b + 2 * b * c + 2 * c * a)) < 1e-12,
              "a² + b² + c² + 2ab + 2bc + 2ca")

        outer = Polygon(*[P(p) for p in _rect(0, 0, s, s)],
                        stroke_color=YELLOW_B, stroke_width=5)
        sl = VGroup(*[_lab(nm[i], P((cut[i], 0)), P((cut[i + 1], 0)), DOWN, 30)
                      for i in range(3)],
                    *[_lab(nm[i], P((0, cut[i])), P((0, cut[i + 1])), LEFT, 30)
                      for i in range(3)])
        dl, dlab = _dim(P((0, 0)), P((s, 0)), "a + b + c", DOWN, off=0.8)
        grid = VGroup(*[Line(P((cut[i], 0)), P((cut[i], s)), color=WHITE,
                             stroke_width=2) for i in (1, 2)],
                      *[Line(P((0, cut[i])), P((s, cut[i])), color=WHITE,
                             stroke_width=2) for i in (1, 2)])
        self.play(Create(outer), FadeIn(sl), Create(dl), FadeIn(dlab),
                  run_time=1.2)
        self.play(Create(grid), run_time=0.8)

        def label(i, j, s_):
            x0, x1, y0, y1 = _box(T(i, j))
            return tag(s_, 30 if min(L[i], L[j]) > 1.5 else 28).move_to(
                P(((x0 + x1) / 2, (y0 + y1) / 2)))

        # the mirror: the square's diagonal
        diag = DashedLine(P((-0.25, -0.25)), P((s + 0.25, s + 0.25)),
                          color=GREY_A, stroke_width=3, dash_length=0.12)
        self.play(Create(diag), run_time=0.6)
        axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        pairs = [((1, 0), "ab", ORANGE), ((2, 1), "bc", PURPLE_B),
                 ((2, 0), "ca", RED_D)]
        placed = []
        for (i, j), name, col in pairs:
            t0 = F.poly(T(i, j), col)
            l0 = label(i, j, name)
            self.play(FadeIn(t0), FadeIn(l0), run_time=0.6)
            t1 = t0.copy()
            self.add(t1)
            self.play(Rotate(t1, angle=PI, axis=axis, about_point=P((0, 0))),
                      run_time=1.2)
            check(_same_pts(_verts(t1), [P(p) for p in T(j, i)]),
                  f"the folded {name} tile lands on its partner")
            l1 = label(j, i, name)
            self.play(FadeIn(l1), run_time=0.4)
            placed += [t0, t1, l0, l1]
            _inside(l0, [P(p) for p in T(i, j)])
            _inside(l1, [P(p) for p in T(j, i)])
        self.play(FadeOut(diag), run_time=0.5)

        # the three squares on the diagonal
        sq = [F.poly(T(i, i), col) for i, col in zip(range(3),
                                                    (BLUE_D, TEAL_D, GREEN_D))]
        sql = [label(i, i, nm[i] + "²") for i in range(3)]
        self.play(LaggedStart(*[FadeIn(q) for q in sq], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(q) for q in sql], lag_ratio=0.3),
                  run_time=1.2)
        self.add(grid, outer)
        for i in range(3):
            _inside(sql[i], [P(p) for p in T(i, i)])
        _safe(outer, sl, dl, dlab)
        self.play(Write(caption("(a+b+c)²  =  a² + b² + c² + 2ab + 2bc + 2ca", 32)))
        self.hold(2.2)


class C13_FactorTiles(Board):
    """The tiles of x² + (p + q)x + pq: an x-square, an x by (p + q) strip
    and a p × q tile. Cut the strip at p: the x × p part slides against
    the square's side, the x × q part turns a quarter and lies along its
    top, and the p × q tile fills the corner exactly — an (x + p) × (x + q)
    rectangle:  x² + (p+q)x + pq = (x + p)(x + q)."""

    def construct(self):
        x, p, q = 3.6, 1.7, 1.1
        g1, g2 = 1.4, 1.6
        F = Frame(-1.0, 11.6, -1.0, 5.0)
        P, k = F.P, F.k
        X2 = _rect(0, 0, x, x)
        s0 = x + g1
        S = _rect(s0, 0, p + q, x)
        SP, SQ = _rect(s0, 0, p, x), _rect(s0 + p, 0, q, x)
        t0 = s0 + p + q + g2
        TT = _rect(t0, x / 2 - q / 2, p, q)
        SPt, SQt, Tt = _rect(x, 0, p, x), _rect(0, x, x, q), _rect(x, x, p, q)
        check(_tiling([X2, SPt, SQt, Tt], _rect(0, 0, x + p, x + q)),
              "the four tiles fill the (x+p) × (x+q) rectangle")
        check(abs(area(X2) + area(S) + area(TT) - (x * x + (p + q) * x + p * q))
              < 1e-12, "tile areas x², (p+q)x, pq")
        check(_tiling([SP, SQ], S), "the strip cut at p")

        sq = F.poly(X2, BLUE_D)
        st = F.poly(S, TEAL_D)
        tt = F.poly(TT, ORANGE)
        lx2 = tag("x²", 36).move_to(P((x / 2, x / 2)))
        lxb = _lab("x", P((0, 0)), P((x, 0)), DOWN, 30)
        lxl = _lab("x", P((0, 0)), P((0, x)), LEFT, 30)
        lst = tag("(p+q)x", 30).move_to(P((s0 + (p + q) / 2, x / 2)))
        lpq = _lab("p + q", P((s0, 0)), P((s0 + p + q, 0)), DOWN, 30)
        ltt = tag("pq", 30).move_to(P((t0 + p / 2, x / 2)))
        ltp = _lab("p", P(TT[0]), P(TT[1]), DOWN, 30)
        ltq = _lab("q", P(TT[1]), P(TT[2]), RIGHT, 30)
        plus = [tag("+", 44, YELLOW_B).move_to(P((x + g1 / 2, x / 2))),
                tag("+", 44, YELLOW_B).move_to(P((s0 + p + q + g2 / 2, x / 2)))]
        self.play(FadeIn(sq), FadeIn(lx2), FadeIn(lxb), FadeIn(lxl), run_time=0.8)
        self.play(FadeIn(plus[0]), FadeIn(st), FadeIn(lst), FadeIn(lpq),
                  run_time=0.8)
        self.play(FadeIn(plus[1]), FadeIn(tt), FadeIn(ltt), FadeIn(ltp),
                  FadeIn(ltq), run_time=0.8)
        self.hold(0.6)

        # cut the strip at p
        sp, sq_ = F.poly(SP, TEAL_D), F.poly(SQ, GREEN_D)
        cut = Line(P((s0 + p, 0)), P((s0 + p, x)), color=WHITE, stroke_width=4)
        lp = _lab("p", P(SP[0]), P(SP[1]), DOWN, 30)
        lq = _lab("q", P(SQ[0]), P(SQ[1]), DOWN, 30)
        lsp = tag("px", 30).move_to(P((s0 + p / 2, x / 2)))
        lsq = tag("qx", 28).move_to(P((s0 + p + q / 2, x / 2)))
        self.play(Create(cut), FadeOut(lst), FadeOut(lpq), run_time=0.6)
        self.remove(st)
        self.add(sp, sq_, cut)
        self.play(sq_.animate.set_fill(GREEN_D), FadeIn(lp), FadeIn(lq),
                  FadeIn(lsp), FadeIn(lsq), run_time=0.6)
        self.remove(cut)
        self.hold(0.5)

        # x × p against the side of the square
        self.play(FadeOut(plus[0]), FadeOut(plus[1]), FadeOut(lp), run_time=0.4)
        self.play(VGroup(sp, lsp).animate.shift(LEFT * g1 * k), run_time=1.0)
        check(_same_pts(_verts(sp), [P(v) for v in SPt]), "x × p in place")

        # x × q: a quarter-turn up into the band above, then along the top
        c0 = np.array([s0 + p + q / 2, x / 2])
        c1 = np.array([c0[0], x + q / 2])

        def up_turn(v):
            w = np.asarray(v, float) - c0
            return c1 + np.array([-w[1], w[0]])

        mid = [up_turn(v) for v in SQ]
        check(_same_pts(mid, _rect(c0[0] - x / 2, x, x, q)), "turned strip in the top band")
        check(c0[0] - x / 2 > x + p + 0.05 and c0[0] + np.hypot(x, q) / 2 < t0 - 0.05,
              "the turning strip clears its neighbours")
        self.play(FadeOut(lq), FadeOut(lsq), run_time=0.4)
        self.play(_rigid(sq_, [P(v) for v in SQ], [P(v) for v in mid], turn=PI / 2),
                  run_time=1.3)
        self.play(sq_.animate.shift(LEFT * (c0[0] - x / 2) * k), run_time=1.1)
        check(_same_pts(_verts(sq_), [P(v) for v in SQt]), "x × q on top")
        lsq2 = tag("qx", 30).move_to(P((x / 2, x + q / 2)))
        self.play(FadeIn(lsq2), run_time=0.4)

        # the p × q tile fills the corner
        self.play(FadeOut(ltp), FadeOut(ltq), run_time=0.4)
        rise = x - TT[0][1]
        self.play(VGroup(tt, ltt).animate.shift(UP * rise * k), run_time=0.8)
        self.play(VGroup(tt, ltt).animate.shift(LEFT * (t0 - x) * k), run_time=1.1)
        check(_same_pts(_verts(tt), [P(v) for v in Tt]), "p × q in the corner")

        # the whole rectangle, moved to the middle and measured
        fig = VGroup(sq, sp, sq_, tt, lx2, lsp, lsq2, ltt, lxb, lxl)
        shift = np.array([0.6, 0.95, 0.0]) - P(((x + p) / 2, (x + q) / 2))
        self.play(fig.animate.shift(shift), run_time=1.0)

        def Q(v):
            return P(v) + shift

        outer = Polygon(*[Q(v) for v in _rect(0, 0, x + p, x + q)],
                        stroke_color=YELLOW_B, stroke_width=5)
        lp2 = _lab("p", Q((x, 0)), Q((x + p, 0)), DOWN, 30)
        lq2 = _lab("q", Q((0, x)), Q((0, x + q)), LEFT, 30)
        dw, lw = _dim(Q((0, 0)), Q((x + p, 0)), "x + p", DOWN, off=0.8)
        dh, lh = _dim(Q((0, 0)), Q((0, x + q)), "x + q", LEFT, off=0.78)
        self.play(Create(outer), FadeIn(lp2), FadeIn(lq2), run_time=0.8)
        self.play(Create(dw), Create(dh), FadeIn(lw), FadeIn(lh), run_time=0.8)
        for m, v in ((lx2, X2), (lsp, SPt), (lsq2, SQt), (ltt, Tt)):
            _inside(m, [Q(w) for w in v])
        _apart(lxb, lp2, lw, lxl, lq2, lh)
        _safe(outer, dw, dh, lw, lh, lxl, lq2)
        self.play(Write(caption("x² + (p+q)x + pq  =  (x+p)(x+q)", 36)))
        self.hold(2.2)


class C19_ArrayTurn(Board):
    """a rows of b dots. Turned a quarter-turn — the same dots, none added
    or lost — the rows stand up as columns, and the array is b rows of a:
        ab = ba."""

    def construct(self):
        a, b = 4, 7
        u = 0.68
        L0 = np.array([-3.35, 0.35, 0.0])
        R0 = np.array([3.45, 0.35, 0.0])

        def grid(rows, cols, centre):
            return [centre + np.array([(j - (cols - 1) / 2) * u,
                                       ((rows - 1) / 2 - i) * u, 0.0])
                    for i in range(rows) for j in range(cols)]

        def turned(pt):                   # quarter-turn about L0, moved to R0
            w = pt - L0
            return R0 + np.array([-w[1], w[0], 0.0])

        left = grid(a, b, L0)
        check(_same_pts([turned(pt) for pt in left], grid(b, a, R0)),
              "the turned a × b array is the b × a array")

        def caps(rows, cols, centre, color, horizontal=True):
            out = VGroup()
            for i in range(rows if horizontal else cols):
                if horizontal:
                    w, h = (cols - 1) * u + 0.44, 0.44
                    c = centre + np.array([0.0, ((rows - 1) / 2 - i) * u, 0.0])
                else:
                    w, h = 0.44, (rows - 1) * u + 0.44
                    c = centre + np.array([(i - (cols - 1) / 2) * u, 0.0, 0.0])
                out.add(RoundedRectangle(width=w, height=h, corner_radius=0.22,
                                         stroke_color=color, stroke_width=3)
                        .move_to(c))
            return out

        def brk(p, q, s, out, color):
            """A bracket beside the screen segment pq, ends turned in."""
            p, q, n = to3(p), to3(q), to3(out)
            g = VGroup(Line(p, q, color=color, stroke_width=3),
                       Line(p, p - 0.13 * n, color=color, stroke_width=3),
                       Line(q, q - 0.13 * n, color=color, stroke_width=3))
            return g, tag(s, 32, color).next_to((p + q) / 2, n, buff=0.14)

        dots = VGroup(*[Dot(pt, radius=0.11, color=WHITE) for pt in left])
        rows_l = caps(a, b, L0, TEAL_B)
        hw, hh = (b - 1) * u / 2 + 0.22, (a - 1) * u / 2 + 0.22
        ga, la = brk(L0 + np.array([-hw - 0.2, hh, 0]), L0 + np.array([-hw - 0.2, -hh, 0]),
                     "a", LEFT, TEAL_B)
        gb, lb = brk(L0 + np.array([-hw, hh + 0.2, 0]), L0 + np.array([hw, hh + 0.2, 0]),
                     "b", UP, YELLOW_B)
        self.play(LaggedStart(*[FadeIn(dd, scale=0.5) for dd in dots],
                              lag_ratio=0.03), run_time=1.4)
        self.play(LaggedStart(*[Create(r) for r in rows_l], lag_ratio=0.2),
                  run_time=1.0)
        self.play(Create(gb), FadeIn(lb), Create(ga), FadeIn(la), run_time=0.8)
        tl = _ctag("a × b", 34, WHITE, {"a": TEAL_B, "b": YELLOW_B}).move_to(
            np.array([L0[0], -2.45, 0.0]))
        self.play(FadeIn(tl), run_time=0.5)
        self.hold(0.6)

        # the same array, turned a quarter-turn on its way across
        moving = VGroup(dots.copy(), rows_l.copy())
        self.add(moving)
        self.play(_rigid(moving, [L0, L0 + RIGHT], [R0, R0 + UP]), run_time=2.2)
        check(_same_pts([dd.get_center() for dd in moving[0]], grid(b, a, R0)),
              "dots land on the b × a array")
        # the rows of a have become columns; read it by rows instead
        rows_r = caps(b, a, R0, YELLOW_B)
        hw2, hh2 = (a - 1) * u / 2 + 0.22, (b - 1) * u / 2 + 0.22
        gb2, lb2 = brk(R0 + np.array([-hw2 - 0.2, hh2, 0]),
                       R0 + np.array([-hw2 - 0.2, -hh2, 0]), "b", LEFT, YELLOW_B)
        ga2, la2 = brk(R0 + np.array([-hw2, hh2 + 0.2, 0]),
                       R0 + np.array([hw2, hh2 + 0.2, 0]), "a", UP, TEAL_B)
        self.play(Create(ga2), FadeIn(la2), run_time=0.6)
        self.hold(0.4)
        self.play(FadeOut(moving[1]), LaggedStart(*[Create(r) for r in rows_r],
                                                  lag_ratio=0.12), run_time=1.3)
        self.play(Create(gb2), FadeIn(lb2), run_time=0.6)
        tr = _ctag("b × a", 34, WHITE, {"a": TEAL_B, "b": YELLOW_B}).move_to(
            np.array([R0[0], -2.45, 0.0]))
        eq = tag("=", 40).move_to(np.array([(L0[0] + R0[0]) / 2, -2.45, 0.0]))
        self.play(FadeIn(tr), FadeIn(eq), run_time=0.6)
        _safe(dots, rows_l, ga, gb, la, lb, moving[0], rows_r, ga2, gb2,
              la2, lb2, tl, tr)
        _apart(la, lb, tl, la2, lb2, tr, eq)
        self.play(Write(caption("ab  =  ba", 40)))
        self.hold(2.2)


class C14_AlKhwarizmi(Board):
    """x² + 10x = 39, solved as al-Khwārizmī did. The ten x's (5x on each
    side of the x-square) are cut into four strips x by 2½. Two stay on
    the sides; the outer two each turn a quarter about a corner and slide
    onto the top and the bottom: a cross of area 39. Four corner squares
    (2½)², 25 together, complete the square of side x + 5:
        (x + 5)² = 39 + 25 = 64,  x + 5 = 8,  x = 3.
    (The picture finds the positive root only.)"""

    def construct(self):
        x, h = 3.0, 2.5
        F = Frame(-6.55, 9.55, -3.35, 5.75)
        P, k = F.P, F.k
        X2 = _rect(0, 0, x, x)
        Q = [_rect(-2 * h, 0, h, x), _rect(-h, 0, h, x), _rect(x, 0, h, x),
             _rect(x + h, 0, h, x)]
        TOP, BOT = _rect(0, x, x, h), _rect(0, -h, x, h)
        CORN = [_rect(-h, -h, h, h), _rect(x, -h, h, h), _rect(x, x, h, h),
                _rect(-h, x, h, h)]
        BIG = _rect(-h, -h, x + 2 * h, x + 2 * h)
        piv1, piv4 = (-h, x), (x + h, 0.0)       # the two corners turned about

        def cw(v, c):                            # quarter-turn clockwise about c
            return (c[0] + (v[1] - c[1]), c[1] - (v[0] - c[0]))

        up1 = [cw(v, piv1) for v in Q[0]]
        dn4 = [cw(v, piv4) for v in Q[3]]
        check(_same_pts(up1, _rect(-h - x, x, x, h)), "strip 1 turns up into the top band")
        check(_same_pts([(v[0] + x + h, v[1]) for v in up1], TOP), "... and slides onto the top")
        check(_same_pts(dn4, _rect(x + h, -h, x, h)), "strip 4 turns down into the bottom band")
        check(_same_pts([(v[0] - x - h, v[1]) for v in dn4], BOT), "... and slides onto the bottom")
        check(_tiling([X2, Q[1], Q[2], TOP, BOT] + CORN, BIG), "cross + four corners = the big square")
        check(abs(area(X2) + sum(area(q) for q in Q) - 39) < 1e-12, "x² + 10x = 39 at x = 3")
        check(abs(sum(area(c) for c in CORN) - 25) < 1e-12 and
              abs(area(BIG) - 64) < 1e-12 and abs(np.sqrt(64) - (x + 5)) < 1e-12,
              "(x+5)² = 39 + 25 = 64")

        sq = F.poly(X2, BLUE_D)
        halves = [F.poly(_rect(-2 * h, 0, 2 * h, x), TEAL_D),
                  F.poly(_rect(x, 0, 2 * h, x), TEAL_D)]
        lx2 = tag("x²", 34).move_to(P((x / 2, x / 2)))
        lxu = _lab("x", P((0, 0)), P((x, 0)), DOWN, 28)
        l5x = [tag("5x", 32).move_to(P((-h, x / 2))),
               tag("5x", 32).move_to(P((x + h, x / 2)))]
        l5 = [_lab("5", P((-2 * h, 0)), P((0, 0)), DOWN, 28),
              _lab("5", P((x, 0)), P((x + 2 * h, 0)), DOWN, 28)]
        lxl = _lab("x", P((-2 * h, 0)), P((-2 * h, x)), LEFT, 28)
        X0 = 3.45                                 # left edge of the readouts

        def row(s_, y, color=WHITE):
            t = tag(s_, 30, color)
            return t.move_to(np.array([X0 + t.width / 2, y, 0.0]))

        r1 = row("x² + 10x = 39", 3.3)
        self.play(FadeIn(sq), FadeIn(lx2), FadeIn(lxu), run_time=0.8)
        self.play(*[FadeIn(m) for m in halves + l5x + l5], FadeIn(lxl), run_time=0.9)
        self.play(FadeIn(r1), run_time=0.6)
        self.hold(0.6)

        # ten x's in four strips, x by 2½
        strips = [F.poly(qq, TEAL_D) for qq in Q]
        cuts = VGroup(*[DashedLine(P((cx, 0)), P((cx, x)), color=WHITE,
                                   stroke_width=3, dash_length=0.1)
                        for cx in (-h, x + h)])
        lh = [_lab("2½", P(qq[0]), P(qq[1]), DOWN, 28) for qq in Q]
        lin = [tag("2½x", 28).move_to(P(((qq[0][0] + qq[1][0]) / 2, x / 2)))
               for qq in Q]
        self.play(Create(cuts), FadeOut(VGroup(*l5x, *l5)), run_time=0.7)
        self.remove(*halves)
        self.add(*strips, cuts, lxl)
        self.play(*[FadeIn(m) for m in lh + lin], run_time=0.6)
        self.remove(cuts)
        self.hold(0.5)

        def quarter_turn(mob, pivot, start_dir):
            pv = P(pivot)
            dot = Dot(pv, radius=0.07, color=YELLOW_B)
            st = float(np.arctan2(start_dir[1], start_dir[0]))
            arc = Arc(radius=0.55, start_angle=st, angle=-PI / 2, arc_center=pv,
                      color=YELLOW_B, stroke_width=4)
            arc.add_tip(tip_length=0.15, tip_width=0.15)
            self.play(FadeIn(dot), Create(arc), run_time=0.4)
            self.play(_spin(mob, -PI / 2, pv), run_time=1.3)
            self.play(FadeOut(dot), FadeOut(arc), run_time=0.3)

        # strip 1: a quarter-turn up about its top corner, then onto the top
        self.play(FadeOut(lin[0]), FadeOut(lh[0]), FadeOut(lxl), run_time=0.4)
        quarter_turn(strips[0], piv1, (-1.0, -1.0))
        self.play(strips[0].animate.shift(RIGHT * (x + h) * k), run_time=1.1)
        check(_same_pts(_verts(strips[0]), [P(v) for v in TOP]), "strip 1 on top")
        # strip 4: a quarter-turn down about its bottom corner, then under
        self.play(FadeOut(lin[3]), FadeOut(lh[3]), run_time=0.4)
        quarter_turn(strips[3], piv4, (1.0, 1.0))
        self.play(strips[3].animate.shift(LEFT * (x + h) * k), run_time=1.1)
        check(_same_pts(_verts(strips[3]), [P(v) for v in BOT]), "strip 4 below")
        lt = tag("2½x", 28).move_to(P((x / 2, x + h / 2)))
        lb = tag("2½x", 28).move_to(P((x / 2, -h / 2)))
        self.play(FadeIn(lt), FadeIn(lb), FadeOut(lh[1]), FadeOut(lh[2]),
                  FadeOut(lxu), run_time=0.6)

        # the cross is x² + 10x = 39
        cross = [(0, -h), (x, -h), (x, 0), (x + h, 0), (x + h, x), (x, x),
                 (x, x + h), (0, x + h), (0, x), (-h, x), (-h, 0), (0, 0)]
        check(abs(area(cross) - 39) < 1e-12, "the cross has area 39")
        cr = Polygon(*[P(v) for v in cross], stroke_color=YELLOW_B, stroke_width=5)
        self.play(Create(cr), Indicate(r1, color=YELLOW_B, scale_factor=1.08),
                  run_time=1.0)
        self.hold(0.3)

        # four corners of (2½)² complete the square
        cdash = [_dashed_outline([P(v) for v in cc], YELLOW_B, 3, 16) for cc in CORN]
        cfill = [F.poly(cc, YELLOW_E, 0.88) for cc in CORN]
        clab = [tag("(2½)²", 26, BLACK).move_to(P(((cc[0][0] + cc[1][0]) / 2,
                                                  (cc[0][1] + cc[2][1]) / 2)))
                for cc in CORN]
        r2 = row("4 · (2½)² = 25", 2.55)
        self.play(*[Create(c_) for c_ in cdash], run_time=0.7)
        self.play(*[FadeIn(f) for f in cfill], *[FadeIn(l_) for l_ in clab],
                  FadeIn(r2), run_time=0.8)
        self.remove(*cdash)
        self.remove(cr)
        big = Polygon(*[P(v) for v in BIG], stroke_color=YELLOW_B, stroke_width=5)
        lb1 = _lab("2½", P((-h, -h)), P((0, -h)), DOWN, 28)
        lb2 = _lab("x", P((0, -h)), P((x, -h)), DOWN, 28)
        lb3 = _lab("2½", P((x, -h)), P((x + h, -h)), DOWN, 28)
        dg, dlab = _dim(P((-h, -h)), P((-h, x + h)), "x + 5", LEFT, off=0.45)
        r3 = row("(x + 5)² = 64", 1.8, YELLOW_B)
        self.play(Create(big), FadeIn(lb1), FadeIn(lb2), FadeIn(lb3),
                  Create(dg), FadeIn(dlab), run_time=1.0)
        self.play(FadeIn(r3), run_time=0.6)
        r4 = row("x + 5 = 8", 1.05)
        r5 = row("x = 3", 0.3, YELLOW_B)
        self.play(FadeIn(r4), run_time=0.6)
        self.play(FadeIn(r5), run_time=0.6)

        for m, cc in zip(clab, CORN):
            _inside(m, [P(v) for v in cc])
        _inside(lt, [P(v) for v in TOP])
        _inside(lb, [P(v) for v in BOT])
        _apart(r1, r2, r3, r4, r5, lb1, lb2, lb3, dlab)
        _safe(big, r1, r2, r3, r4, r5, lb1, lb2, lb3, dg, dlab)
        check(r1.get_left()[0] > big.get_right()[0] + 0.3, "readouts clear of the square")
        self.play(Write(caption("x² + 10x = 39   ⟹   (x + 5)² = 64,   x = 3", 34)))
        self.hold(2.2)


# ==================================================== D. MEANS & INEQUALITIES

class D7_AMGMFourRectangles(Board):
    """Four a × b rectangles laid round the inside of the (a + b)-square,
    each turned a quarter from the last, leave a square hole of side a − b:
        (a + b)² = 4ab + (a − b)² ≥ 4ab,
    equality exactly when the hole closes, a = b. Keeping a + b fixed and
    letting a and b approach each other closes it. Quarter the square and
    take roots: √(ab) ≤ (a + b)/2."""

    def construct(self):
        a0, b0 = 3.8, 1.5
        s = a0 + b0
        t0 = (a0 - b0) / 2
        F = Frame(-0.7, s + 0.2, -0.7, s + 0.2, max_w=7.4, centre=(-2.75, 0.375))
        P, k = F.P, F.k
        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D]

        def geo(t):
            a, b = s / 2 + t, s / 2 - t
            R = [_rect(0, 0, a, b), _rect(a, 0, b, a), _rect(b, a, a, b),
                 _rect(0, b, b, a)]
            return a, b, R, _rect(b, b, a - b, a - b)

        for t in np.linspace(t0, 0.0, 7):
            a, b, R, H = geo(t)
            pieces = R + ([H] if t > 1e-12 else [])
            check(_tiling(pieces, _rect(0, 0, s, s)),
                  "rectangles (and hole) tile the (a+b)-square")
            check(abs(4 * a * b + (a - b) ** 2 - s * s) < 1e-9,
                  "(a+b)² = 4ab + (a−b)²")
        a, b, R, H = geo(t0)

        outer = Polygon(*[P(v) for v in _rect(0, 0, s, s)],
                        stroke_color=YELLOW_B, stroke_width=5)
        self.play(Create(outer), run_time=0.9)

        rects = [F.poly(r, c) for r, c in zip(R, cols)]
        ins = [tag("ab", 32).move_to(P(np.mean(r, axis=0))) for r in R]
        come = [UP, LEFT, DOWN, RIGHT]           # direction each one slides in
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r, shift=d * 0.7 * k),
                                               FadeIn(l, shift=d * 0.7 * k))
                                for r, l, d in zip(rects, ins, come)],
                              lag_ratio=0.35), run_time=2.2)
        self.add(outer)

        # the sides: a long and a short side make up the square's side
        def side_labels(a, b):
            return [_lab("a", P((0, 0)), P((a, 0)), DOWN, 30),
                    _lab("b", P((a, 0)), P((s, 0)), DOWN, 30),
                    _lab("b", P((0, 0)), P((0, b)), LEFT, 30),
                    _lab("a", P((0, b)), P((0, s)), LEFT, 30)]

        sl = side_labels(a, b)
        self.play(*[FadeIn(m) for m in sl], run_time=0.6)

        def hole_mob(t):
            a, b, R, H = geo(t)
            if a - b < 1e-6:
                return VGroup()
            return VGroup(F.poly(H, RED_D, 0.45, stroke_width=0),
                          Polygon(*[P(v) for v in H], stroke_color=RED_B,
                                  stroke_width=3))

        hole = hole_mob(t0)
        hlab = tag("(a−b)²", 30).move_to(P(np.mean(H, axis=0)))
        self.play(FadeIn(hole), FadeIn(hlab), run_time=0.8)

        X0 = 1.35
        def row(s_, y, color=WHITE, t2c=None):
            t = _ctag(s_, 32, color, t2c)
            return t.move_to(np.array([X0 + t.width / 2, y, 0.0]))

        r1 = row("(a+b)²  =  4ab + (a−b)²", 2.3, WHITE, {"(a−b)²": RED_B})
        r2 = row("(a+b)²  ≥  4ab", 1.35)
        self.play(FadeIn(r1), run_time=0.7)
        self.play(FadeIn(r2), run_time=0.7)
        self.hold(0.6)

        # let a and b meet, a + b fixed: the hole closes
        tr = ValueTracker(t0)
        self.play(FadeOut(hlab), run_time=0.4)
        self.remove(*rects, hole)
        dyn = always_redraw(lambda: VGroup(
            *[F.poly(r, c) for r, c in zip(geo(tr.get_value())[2], cols)],
            hole_mob(tr.get_value())))
        self.add(dyn, outer)
        for m, r in zip(ins, range(4)):
            m.add_updater(lambda mm, r=r: mm.move_to(
                P(np.mean(geo(tr.get_value())[2][r], axis=0))))
        fix = [(0, lambda a, b: ((0, 0), (a, 0)), DOWN),
               (1, lambda a, b: ((a, 0), (s, 0)), DOWN),
               (2, lambda a, b: ((0, 0), (0, b)), LEFT),
               (3, lambda a, b: ((0, b), (0, s)), LEFT)]
        for i, seg, o in fix:
            p0, q0 = seg(a0, b0)
            off = sl[i].get_center() - (P(p0) + P(q0)) / 2

            def upd(mm, seg=seg, off=off):
                a, b = geo(tr.get_value())[:2]
                p, q = seg(a, b)
                mm.move_to((P(p) + P(q)) / 2 + off)
            sl[i].add_updater(upd)
        self.add(*ins, *sl)
        self.play(tr.animate.set_value(0.0), run_time=2.6)
        r3 = row("a = b  ⟺  (a−b)² = 0", 0.4, GREEN_B)
        self.play(FadeIn(r3), Flash(P((s / 2, s / 2)), color=GREEN_B,
                                    line_length=0.3), run_time=0.8)
        self.hold(0.6)
        self.play(tr.animate.set_value(t0), run_time=2.0)
        for m in ins + sl:
            m.clear_updaters()
        self.remove(dyn)
        rects = [F.poly(r, c) for r, c in zip(R, cols)]
        hole = hole_mob(t0)
        self.add(*rects, hole, outer, *ins, *sl)
        self.play(FadeIn(hlab), run_time=0.5)

        _inside(hlab, [P(v) for v in H])
        for m, r in zip(ins, R):
            _inside(m, [P(v) for v in r])
        _apart(*sl, r1, r2, r3)
        _safe(outer, *sl, r1, r2, r3)
        check(r1.get_left()[0] > outer.get_right()[0] + 0.4, "readouts clear of the square")
        self.play(Write(caption("4ab ≤ (a+b)²   ⟹   √(ab) ≤ (a+b)/2", 36)))
        self.hold(2.2)


class D8_TwoSquaresTwoRectangles(Board):
    """Two a × b rectangles laid along two sides of the a-square overlap in
    a b × b square: the b-square slides onto the overlap and fits it
    exactly. So the two rectangles cover the a-square once and the
    b-square once more, all but the corner square (a − b)²:
        a² + b² = 2ab + (a − b)² ≥ 2ab,
    equality exactly when the corner closes, a = b (shown with a + b
    fixed, letting a and b approach each other)."""

    def construct(self):
        a0, b0 = 3.8, 1.3
        s = a0 + b0
        t0 = (a0 - b0) / 2
        g = 0.95
        F = Frame(-1.7, a0 + g + b0 + 0.55, -1.25, a0 + 0.55, centre=(0.95, 0.375))
        P, k = F.P, F.k

        def geo(t):
            a, b = s / 2 + t, s / 2 - t
            return a, b, dict(A=_rect(0, 0, a, a), R1=_rect(0, a - b, a, b),
                              R2=_rect(a - b, 0, b, a), O=_rect(a - b, a - b, b, b),
                              C=_rect(0, 0, a - b, a - b),
                              Bq=_rect(a + g, a - b, b, b))

        for t in np.linspace(t0, 0.0, 6):
            a, b, G = geo(t)
            check(_tiling([G["R1"], _rect(a - b, 0, b, a - b)] +
                          ([G["C"]] if t > 1e-12 else []), G["A"]),
                  "the rectangles (overlap counted once) and the corner tile a²")
            check(_same_pts(G["O"], _rect(a - b, a - b, b, b)) and
                  abs(area(G["O"]) - b * b) < 1e-12, "the overlap is a b-square")
            check(abs(a * a + b * b - (2 * a * b + (a - b) ** 2)) < 1e-9,
                  "a² + b² = 2ab + (a−b)²")
        a, b, G = geo(t0)

        sqA, sqB = F.poly(G["A"], BLUE_D), F.poly(G["Bq"], ORANGE, 0.9)
        la2 = tag("a²", 36).move_to(P(np.mean(G["A"], axis=0)))
        lb2 = tag("b²", 30).move_to(P(np.mean(G["Bq"], axis=0)))

        def dims(a):
            d1, t1 = _dim(P((0, 0)), P((a, 0)), "a", DOWN, off=0.75, color=BLUE_B)
            d2, t2 = _dim(P((0, 0)), P((0, a)), "a", LEFT, off=1.35, color=BLUE_B)
            return d1, t1, d2, t2

        dA, ldA, dL, ldL = dims(a)
        lbb = _lab("b", P(G["Bq"][0]), P(G["Bq"][1]), DOWN, 30, ORANGE)
        lbr = _lab("b", P(G["Bq"][1]), P(G["Bq"][2]), RIGHT, 30, ORANGE)
        self.play(FadeIn(sqA), FadeIn(la2), Create(dA), Create(dL), FadeIn(ldA),
                  FadeIn(ldL), run_time=1.0)
        self.play(FadeIn(sqB), FadeIn(lb2), FadeIn(lbb), FadeIn(lbr), run_time=0.8)
        self.hold(0.5)

        # two a × b rectangles: along the top, and up the right side
        r1, r2 = F.poly(G["R1"], TEAL_D, 0.85), F.poly(G["R2"], GREEN_D, 0.85)

        def rims(G):
            """Each rectangle's whole outline, inset a little (by different
            amounts), so both stay visible where they overlap."""
            return VGroup(Polygon(*[P(v) for v in _inset(G["R1"], 0.05)],
                                  stroke_color=TEAL_A, stroke_width=4),
                          Polygon(*[P(v) for v in _inset(G["R2"], 0.12)],
                                  stroke_color=GREEN_A, stroke_width=4))

        rim = rims(G)
        l1 = tag("ab", 32).move_to(P(np.mean(G["R1"], axis=0)))
        l2 = tag("ab", 32).move_to(P(np.mean(G["R2"], axis=0)))
        lr1 = _lab("b", P((0, a - b)), P((0, a)), LEFT, 30, TEAL_B)
        lr2 = _lab("b", P((a - b, 0)), P((a, 0)), DOWN, 30, GREEN_B)
        self.play(FadeOut(la2), FadeIn(VGroup(r1, rim[0]), shift=DOWN * 0.8 * k),
                  FadeIn(l1), FadeIn(lr1), run_time=1.0)
        self.play(FadeIn(VGroup(r2, rim[1]), shift=LEFT * 0.8 * k), FadeIn(l2),
                  FadeIn(lr2), run_time=1.0)
        ov = _dashed_outline([P(v) for v in G["O"]], WHITE, 4, 16)
        self.play(Create(ov), run_time=0.6)
        self.hold(0.4)

        # the overlap is exactly the b-square
        self.play(FadeOut(lbb), FadeOut(lbr), run_time=0.3)
        self.play(VGroup(sqB, lb2).animate.shift(LEFT * (b + g) * k), run_time=1.3)
        check(_same_pts(_verts(sqB), [P(v) for v in G["O"]]), "b² lands on the overlap")
        self.remove(ov)
        self.add(rim, lb2)
        self.hold(0.3)

        # what is left uncovered: the corner (a − b)² of the a-square
        cfr = Polygon(*[P(v) for v in G["C"]], stroke_color=RED_B, stroke_width=5)
        lc = tag("(a−b)²", 32).move_to(P(np.mean(G["C"], axis=0)))
        lcb = _lab("a − b", P((0, 0)), P((a - b, 0)), DOWN, 28, RED_B)
        lcl = _lab("a − b", P((0, 0)), P((0, a - b)), LEFT, 28, RED_B)
        self.play(Create(cfr), FadeIn(lc), FadeIn(lcb), FadeIn(lcl), run_time=0.9)
        self.hold(0.8)

        # a and b approach each other (a + b fixed): the corner closes
        tr = ValueTracker(t0)
        self.play(FadeOut(lc), FadeOut(lcb), FadeOut(lcl), FadeOut(l1), FadeOut(l2),
                  FadeOut(lb2), run_time=0.4)
        statics = [sqA, r1, r2, sqB, rim, cfr, dA, dL, ldA, ldL]
        self.remove(*statics)

        def frame_at(t):
            a, b, G = geo(t)
            out = VGroup(F.poly(G["A"], BLUE_D), F.poly(G["R1"], TEAL_D, 0.85),
                         F.poly(G["R2"], GREEN_D, 0.85), F.poly(G["O"], ORANGE, 0.9),
                         rims(G))
            if a - b > 1e-6:
                out.add(Polygon(*[P(v) for v in G["C"]], stroke_color=RED_B,
                                stroke_width=5))
            d1, t1, d2, t2 = dims(a)
            out.add(d1, d2, t1, t2)
            return out

        dyn = always_redraw(lambda: frame_at(tr.get_value()))
        self.add(dyn)
        riders = []
        for m, seg in ((lr1, lambda a, b: ((0, a - b), (0, a))),
                       (lr2, lambda a, b: ((a - b, 0), (a, 0)))):
            p0, q0 = seg(a0, b0)
            riders.append((m, seg, m.get_center() - (P(p0) + P(q0)) / 2))
        for m, seg, off in riders:
            m.add_updater(lambda mm, seg=seg, off=off: mm.move_to(
                (P(seg(*geo(tr.get_value())[:2])[0])
                 + P(seg(*geo(tr.get_value())[:2])[1])) / 2 + off))
        self.add(lr1, lr2)
        self.play(tr.animate.set_value(0.0), run_time=2.6)
        eq = tag("a = b", 34, GREEN_B).move_to(P((s / 4, s / 2)) + UP * 0.55)
        self.play(FadeIn(eq), Flash(P((s / 4, s / 4)), color=GREEN_B,
                                    line_length=0.3), run_time=0.7)
        self.hold(0.6)
        self.play(FadeOut(eq), tr.animate.set_value(t0), run_time=2.2)
        for m in (lr1, lr2):
            m.clear_updaters()
        self.remove(dyn)
        self.add(*statics, lr1, lr2)
        self.play(FadeIn(l1), FadeIn(l2), FadeIn(lb2), FadeIn(lc), FadeIn(lcb),
                  FadeIn(lcl), run_time=0.6)

        _inside(lc, [P(v) for v in G["C"]])
        _inside(lb2, [P(v) for v in G["O"]])
        _inside(l1, [P(v) for v in _rect(0, a - b, a - b, b)])
        _inside(l2, [P(v) for v in _rect(a - b, 0, b, a - b)])
        _apart(lr1, lcl, ldL, eq)
        _apart(lr2, lcb, ldA)
        check(lcl.get_left()[0] > dL[0].get_right()[0] + 0.08, "a − b clear of the dimension line")
        _safe(sqA, dA, dL, ldA, ldL, lr1, lr2, lcb, lcl, eq)
        self.play(Write(caption("a² + b²  =  2ab + (a−b)²  ≥  2ab", 36)))
        self.hold(2.2)


class D11_SquareEnclosesMost(Board):
    """An a × b rectangle and the square of the same perimeter, side
    (a + b)/2, share a corner. The rectangle overhangs the square by
    (a − b)/2 and falls short of its top by the same (a − b)/2. Turn the
    overhang a quarter about its top corner and slide it along the top of
    the rectangle: it fills that gap except for a corner square of side
    (a − b)/2, which is always left over:
        ab = ((a+b)/2)² − ((a−b)/2)² ≤ ((a+b)/2)²,
    equality exactly when a = b (shown with a + b, the perimeter, fixed)."""

    def construct(self):
        a0, b0 = 5.2, 2.2
        m = (a0 + b0) / 2
        t0 = (a0 - b0) / 2
        F = Frame(-1.55, a0 + 0.3, -1.45, m + 0.75)
        P, k = F.P, F.k

        def geo(t):
            a, b = m + t, m - t
            return a, b, dict(
                main=_rect(0, 0, m, b), over=_rect(m, 0, a - m, b),
                turned=_rect(a - b, b, b, a - m), moved=_rect(m - b, b, b, m - b),
                corner=_rect(0, b, m - b, m - b), sq=_rect(0, 0, m, m))

        def cw(v, c):
            return (c[0] + (v[1] - c[1]), c[1] - (v[0] - c[0]))

        for t in np.linspace(t0, 0.0, 6):
            a, b, G = geo(t)
            check(abs(a + b - 2 * m) < 1e-12, "same perimeter: 2a + 2b = 4·(a+b)/2")
            if t > 1e-12:
                check(_same_pts([cw(v, (a, b)) for v in G["over"]], G["turned"]),
                      "the overhang turned about its top corner lies on the rectangle")
                check(_same_pts([(v[0] - (a - m), v[1]) for v in G["turned"]], G["moved"]),
                      "... and slides into the square")
                check(_tiling([G["main"], G["moved"], G["corner"]], G["sq"]),
                      "rectangle pieces + corner tile the square")
                check(abs((m - b) - (a - b) / 2) < 1e-12 and abs((a - m) - (a - b) / 2) < 1e-12,
                      "overhang and shortfall are both (a−b)/2")
            check(abs(a * b - (m * m - t * t)) < 1e-9, "ab = ((a+b)/2)² − ((a−b)/2)²")
        a, b, G = geo(t0)
        d = t0

        whole = F.poly(_rect(0, 0, a, b), BLUE_D)
        main = F.poly(G["main"], BLUE_D)
        over = F.poly(G["over"], BLUE_D)
        lb = _lab("b", P((0, 0)), P((0, b)), LEFT, 30)
        dA, ldA = _dim(P((0, 0)), P((a, 0)), "a", DOWN, off=0.85, color=WHITE)
        self.play(FadeIn(whole), FadeIn(lb), Create(dA), FadeIn(ldA), run_time=1.0)
        self.hold(0.4)

        # the square of the same perimeter
        sq = Polygon(*[P(v) for v in G["sq"]], stroke_color=YELLOW_B, stroke_width=5)
        lm = _lab("(a+b)/2", P((0, 0)), P((m, 0)), DOWN, 26, YELLOW_B)
        per = tag("2a + 2b  =  4 · (a+b)/2", 26, YELLOW_B)
        per.move_to(np.array([SAFE_X - per.width / 2 - 0.1, 3.42, 0.0]))
        self.play(Create(sq), FadeIn(lm), run_time=1.0)
        self.remove(whole)
        self.add(main, over, sq)
        self.play(FadeIn(per), run_time=0.6)

        # it overhangs by (a − b)/2 and falls short of the top by (a − b)/2
        gap = _dashed_outline([P(v) for v in _rect(0, b, m, m - b)], TEAL_B, 3, 30)
        lo = _lab("(a−b)/2", P((m, 0)), P((a, 0)), DOWN, 26, TEAL_B)
        lg = _lab("(a−b)/2", P((0, b)), P((0, m)), LEFT, 26, TEAL_B)
        self.play(over.animate.set_fill(TEAL_D), FadeIn(lo), run_time=0.7)
        self.play(Create(gap), FadeIn(lg), run_time=0.8)
        self.hold(0.5)

        # turn the overhang a quarter about its top corner ...
        pv = P((a, b))
        dot = Dot(pv, radius=0.07, color=YELLOW_B)
        arc = Arc(radius=0.55, start_angle=-PI / 2 - 0.35, angle=-PI / 2 + 0.2,
                  arc_center=pv, color=YELLOW_B, stroke_width=4)
        arc.add_tip(tip_length=0.15, tip_width=0.15)
        self.play(FadeOut(lo), FadeOut(dA), FadeOut(ldA), FadeIn(dot), Create(arc),
                  run_time=0.6)
        self.play(_spin(over, -PI / 2, pv), run_time=1.4)
        check(_same_pts(_verts(over), [P(v) for v in G["turned"]]), "turned onto the rectangle")
        self.play(FadeOut(dot), FadeOut(arc), run_time=0.3)
        # ... and slide it along the top of the rectangle into the square
        rail = guide(P((a - b - d - 0.2, m + 0.18)), P((a + 0.2, m + 0.18)), YELLOW_B)
        self.play(Create(rail), run_time=0.3)
        self.play(over.animate.shift(LEFT * d * k), FadeOut(gap), run_time=1.1)
        self.play(FadeOut(rail), run_time=0.3)
        check(_same_pts(_verts(over), [P(v) for v in G["moved"]]), "slid into the square")

        # the rectangle, rearranged, is the square less a corner ((a−b)/2)²
        def L_pts(t):
            a, b, _ = geo(t)
            return [(0, 0), (m, 0), (m, m), (m - b, m), (m - b, b), (0, b)]

        lsh = Polygon(*[P(v) for v in L_pts(t0)], stroke_color=WHITE, stroke_width=4)
        lab_ab = tag("ab", 34).move_to(P((m / 2, b / 2)))
        cfill = F.poly(G["corner"], RED_D, 0.4, stroke_width=0)
        cfr = Polygon(*[P(v) for v in G["corner"]], stroke_color=RED_B, stroke_width=4)
        lc = tag("((a−b)/2)²", 22).move_to(P(np.mean(G["corner"], axis=0)))
        self.play(Create(lsh), FadeIn(lab_ab), run_time=0.8)
        self.play(FadeIn(cfill), Create(cfr), FadeIn(lc), run_time=0.8)
        self.hold(0.8)

        # a and b approach each other, a + b fixed: the corner closes
        tr = ValueTracker(t0)
        self.play(FadeOut(lc), FadeOut(lg), run_time=0.4)
        statics = [main, over, cfill, sq, lsh, cfr]
        self.remove(*statics)

        def frame_at(t):
            a, b, G = geo(t)
            out = VGroup(F.poly(G["main"], BLUE_D))
            if t > 1e-6:
                out.add(F.poly(G["moved"], TEAL_D),
                        F.poly(G["corner"], RED_D, 0.4, stroke_width=0))
            out.add(Polygon(*[P(v) for v in G["sq"]], stroke_color=YELLOW_B,
                            stroke_width=5))
            if t > 1e-6:
                out.add(Polygon(*[P(v) for v in L_pts(t)], stroke_color=WHITE,
                                stroke_width=4),
                        Polygon(*[P(v) for v in G["corner"]], stroke_color=RED_B,
                                stroke_width=4))
            return out

        dyn = always_redraw(lambda: frame_at(tr.get_value()))
        self.add(dyn)
        off_b = lb.get_center() - (P((0, 0)) + P((0, b))) / 2
        lb.add_updater(lambda mm: mm.move_to(
            (P((0, 0)) + P((0, geo(tr.get_value())[1]))) / 2 + off_b))
        lab_ab.add_updater(lambda mm: mm.move_to(
            P((m / 2, geo(tr.get_value())[1] / 2))))
        self.add(lb, lab_ab)
        self.play(tr.animate.set_value(0.0), run_time=2.6)
        eq = tag("a = b", 32, GREEN_B).next_to(P((m / 2, m)), UP, buff=0.22)
        self.play(FadeIn(eq), run_time=0.5)
        self.hold(0.6)
        self.play(FadeOut(eq), tr.animate.set_value(t0), run_time=2.2)
        lb.clear_updaters()
        lab_ab.clear_updaters()
        self.remove(dyn)
        self.add(*statics, lb, lab_ab)
        self.play(FadeIn(lc), FadeIn(lg), run_time=0.5)

        _inside(lc, [P(v) for v in G["corner"]], pad=0.04)
        _inside(lab_ab, [P(v) for v in G["main"]])
        _apart(lb, lg, lm, per, eq)
        _safe(sq, lb, lg, lm, per, eq, lc)
        check(per.get_bottom()[1] > P((0, m))[1] + 0.25, "readout clear of the square")
        self.play(Write(caption("ab  =  ((a+b)/2)² − ((a−b)/2)²  ≤  ((a+b)/2)²", 32)))
        self.hold(2.2)


class D21_HeronShortestPath(Board):
    """From A to the line and on to B. Reflect B in the line to B′: for
    every point P of the line, PB = PB′ (the triangle folds over the line),
    so the path A→P→B is as long as A→P→B′, a broken line from A to B′ —
    never shorter than the straight segment AB′. AB′ crosses the line at X,
    and there the path A→X→B is exactly AB′ long: the shortest. At X the
    angle with the line on A's side equals the vertical angle on B′'s side,
    which the reflection carries to B's side: equal angles."""

    def construct(self):
        Am, Bm = np.array([-4.6, 2.05]), np.array([3.7, 2.9])
        Bpm = np.array([Bm[0], -Bm[1]])
        tX = Am[1] / (Am[1] + Bm[1])
        Xm = Am + tX * (Bpm - Am)
        Pm0 = np.array([1.3, 0.0])
        F = Frame(-6.3, 6.3, -3.4, 3.4)
        P = F.P

        def L(p, q):
            return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))

        def ang(v):                   # angle of v with the line, in [0, π/2]
            return float(np.arctan2(abs(v[1]), abs(v[0])))

        check(abs(Xm[1]) < 1e-12, "X lies on the line")
        check(abs(L(Am, Xm) + L(Xm, Bm) - L(Am, Bpm)) < 1e-9, "AX + XB = AB′")
        for px in np.linspace(-6.0, 6.0, 25):
            Pt = np.array([px, 0.0])
            check(abs(L(Pt, Bm) - L(Pt, Bpm)) < 1e-12, "PB = PB′")
            check(L(Am, Pt) + L(Pt, Bm) >= L(Am, Bpm) - 1e-12, "AP + PB ≥ AB′")
        check(abs(ang(Am - Xm) - ang(Bm - Xm)) < 1e-12, "equal angles at X")
        check(Am[0] < Xm[0] < Bm[0], "X between the feet of A and B")
        th = ang(Am - Xm)

        A, B, Bp, X = P(Am), P(Bm), P(Bpm), P(Xm)
        line = Line(P((-6.3, 0)), P((6.3, 0)), color=GREY_B, stroke_width=4)
        dA, dB = Dot(A, radius=0.08), Dot(B, radius=0.08)
        lA = tag("A", 30).next_to(A, UL, buff=0.08)
        lB = tag("B", 30).next_to(B, UP, buff=0.14)
        self.play(Create(line), FadeIn(dA), FadeIn(dB), FadeIn(lA), FadeIn(lB),
                  run_time=1.0)

        # some path A → P → B
        tr = ValueTracker(Pm0[0])

        def Pp():
            return P((tr.get_value(), 0.0))

        pathA = always_redraw(lambda: Line(A, Pp(), color=ORANGE, stroke_width=5))
        pathB = always_redraw(lambda: Line(Pp(), B, color=ORANGE, stroke_width=5))
        dP = always_redraw(lambda: Dot(Pp(), radius=0.08, color=ORANGE))
        lP = always_redraw(lambda: tag("P", 30, ORANGE).next_to(Pp(), DL, buff=0.06))
        self.play(Create(pathA), Create(pathB), FadeIn(dP), FadeIn(lP), run_time=1.0)
        self.hold(0.4)

        # reflect B in the line
        foot = P((Bm[0], 0.0))
        perp = DashedLine(B, Bp, color=GREY_A, stroke_width=2.5, dash_length=0.1)
        sq = 0.2
        ra = VMobject(stroke_color=GREY_A, stroke_width=2.5).set_points_as_corners(
            [foot + LEFT * sq, foot + LEFT * sq + UP * sq, foot + UP * sq])
        ticks = VGroup(*[Line(m + LEFT * 0.12, m + RIGHT * 0.12, color=GREY_A,
                              stroke_width=3)
                         for m in ((B + foot) / 2, (foot + Bp) / 2)])
        dBp = Dot(Bp, radius=0.08)
        lBp = tag("B′", 30).next_to(Bp, DOWN, buff=0.14)
        self.play(Create(perp), Create(ra), run_time=0.8)
        self.play(Create(ticks), FadeIn(dBp), FadeIn(lBp), run_time=0.6)

        # PB folds over the line onto PB′: equal lengths
        flip = Line(Pp(), B, color=ORANGE, stroke_width=5)
        self.add(flip)
        self.play(Rotate(flip, angle=PI, axis=RIGHT, about_point=P((0.0, 0.0))),
                  run_time=1.3)
        check(close(flip.get_start(), Pp(), 1e-6) and close(flip.get_end(), Bp, 1e-6),
              "PB folds onto PB′")
        self.remove(flip)
        pathBp = always_redraw(lambda: DashedLine(Pp(), Bp, color=ORANGE,
                                                  stroke_width=4, dash_length=0.12))
        self.add(pathBp)

        # the straight line AB′ is shorter than the broken A → P → B′
        ab = Line(A, Bp, color=YELLOW_B, stroke_width=4)
        self.play(Create(ab), run_time=1.0)
        self.bring_to_front(dA, dBp)
        self.hold(0.4)

        # slide P to where AB′ crosses: the broken path straightens; back
        self.play(tr.animate.set_value(Xm[0]), run_time=2.4)
        self.hold(0.4)
        self.play(tr.animate.set_value(Pm0[0]), run_time=1.8)
        dX = Dot(X, radius=0.08, color=GREEN_B)
        lX = tag("X", 30, GREEN_B).next_to(X, DL, buff=0.06)
        self.play(FadeIn(dX), FadeIn(lX), run_time=0.5)

        # the shortest path A → X → B, and its equal angles
        best = VGroup(Line(A, X, color=GREEN_B, stroke_width=6),
                      Line(X, B, color=GREEN_B, stroke_width=6))
        self.play(Create(best), run_time=1.0)
        self.bring_to_front(dA, dB, dX)
        left_ray, right_ray = P((Xm[0] - 1.0, 0.0)), P((Xm[0] + 1.0, 0.0))
        r = 0.62
        arcs = VGroup(angle_arc(X, left_ray, A, r, GREEN_B, 5),
                      angle_arc(X, right_ray, B, r, GREEN_B, 5),
                      angle_arc(X, right_ray, Bp, r - 0.14, YELLOW_B, 5))
        lth = VGroup(*[tag("θ", 28, c).move_to(X + rr * angle_mid_dir(X, q, w))
                       for q, w, c, rr in ((left_ray, A, GREEN_B, 1.08),
                                           (right_ray, B, GREEN_B, 1.08),
                                           (right_ray, Bp, YELLOW_B, 0.98))])
        self.play(Create(arcs[0]), FadeIn(lth[0]), run_time=0.6)
        self.play(Create(arcs[2]), FadeIn(lth[2]), run_time=0.6)
        self.play(Create(arcs[1]), FadeIn(lth[1]), run_time=0.6)

        labels = [lA, lB, lBp, lX, lP, *lth]
        _apart(*labels)
        _safe(line, dA, dB, dBp, *labels)
        for m, rr, (q, w) in zip(lth, (r, r, r - 0.14),
                                 ((left_ray, A), (right_ray, B), (right_ray, Bp))):
            check(np.linalg.norm(m.get_center() - X) > rr + 0.15,
                  "θ label outside its arc")
            for end in (q, w):                      # clear of both sides
                u = (end - X) / np.linalg.norm(end - X)
                for cx in (m.get_corner(UL), m.get_corner(UR), m.get_corner(DL),
                           m.get_corner(DR)):
                    v = cx - X
                    check(abs(u[0] * v[1] - u[1] * v[0]) > 0.03 or np.dot(u, v) < 0,
                          "θ label clear of the sides of its angle")
        check(abs(th - ang(Bpm - Xm)) < 1e-12, "vertical angle at X equals θ")
        self.play(Write(caption("AP + PB  =  AP + PB′  ≥  AB′  =  AX + XB", 32)))
        self.hold(2.2)
