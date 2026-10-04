# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3a.py — proofs without words, 2D (manim): series, figurate
# numbers and sums.
#
#     B47 Sierpiński's triangle has no area     B48 the Cantor set has no length
#     B41 Gabriel's staircase                   B32 T₂ₙ = 3Tₙ + Tₙ₋₁
#     B38 powers of three                       B39 an alternating geometric series
#     B37 Fibonacci sums of odd and even terms  B40 reciprocals of triangular numbers
#     B31 the triangular number of a product    B43 centred square numbers
#     B45 sums of consecutive numbers
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# letter sub- and superscripts are set from smaller Text pieces by _rich.
# Infinite processes are drawn exactly for a few stages and then "…"; the
# caption states the limit. Cell and dot pictures are drawn for one concrete
# size; the caption states the general result. Every tiling, landing and
# count is checked with check(...) before or right after it is drawn, so a
# wrong construction fails the render instead of rendering quietly into a
# wrong picture.

from fractions import Fraction


# ---------------------------------------------------------------- helpers

def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _box(m):
    return m.get_critical_point(DL), m.get_critical_point(UR)


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _safe(*mobs, tol=0.02):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for i, m in enumerate(mobs):
        lo, hi = _box(m)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area (item {i}, {_name(m)}): [{lo[0]:.2f},"
              f"{hi[0]:.2f}] x [{lo[1]:.2f},{hi[1]:.2f}]")


def _sep(a, b, gap=0.04):
    a0, a1 = _box(a)
    b0, b1 = _box(b)
    return (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
            or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])


def _apart(*mobs, gap=0.04):
    """Fail the render if any two mobjects' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            check(_sep(mobs[i], mobs[j], gap),
                  f"items {i} ({_name(mobs[i])}) and {j} ({_name(mobs[j])}) "
                  f"do not overlap")


def _clear_of(label, group, gap=0.04, what="label"):
    """Fail the render if the label's box touches any member of `group`."""
    for k, m in enumerate(group):
        check(_sep(label, m, gap), f"{what} '{_name(label)}' clear of item {k}")


def _clear_of_segment(m, p, q, gap=0.05, what="a label"):
    """Fail the render if the screen segment pq passes through m's box."""
    lo, hi = _box(m)
    p, q = to3(p), to3(q)
    for f in np.linspace(0.0, 1.0, 400):
        x = p + (q - p) * f
        if lo[0] - gap < x[0] < hi[0] + gap and lo[1] - gap < x[1] < hi[1] + gap:
            check(False, f"a line runs through {what}")


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption in its band, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    c0, c1 = _box(cap)
    check(cap.width <= 13.45 and c0[1] > -4.0 and c1[1] < SAFE_BOTTOM,
          "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if m.has_points() or m.submobjects:
            _safe(m)
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_sep(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


def _dim(p, q, label, off, color=GREY_A, size=24, gap=0.12, width=2.5):
    """Dimension line beside the screen segment pq, shifted by `off`: a thin
    line with end ticks, and the label (a string or a ready mobject) beyond
    it. Returns VGroup(lines, label)."""
    p, q, off = to3(p), to3(q), to3(off)
    n = off / np.linalg.norm(off)
    a, b = p + off, q + off
    tick = 0.09 * n
    g = VGroup(Line(a, b, color=color, stroke_width=width),
               Line(a - tick, a + tick, color=color, stroke_width=width),
               Line(b - tick, b + tick, color=color, stroke_width=width))
    t = tag(label, size, color) if isinstance(label, str) else label
    t.move_to((a + b) / 2 + n * (gap + 0.5 * (abs(n[0]) * t.width
                                             + abs(n[1]) * t.height)))
    return VGroup(g, t)


def _same_spots(group, spots, tol=1e-6):
    """Every member of `group` sits on one of `spots` (screen points), and
    every spot is taken: the motion landed exactly."""
    got = sorted(tuple(np.round(m.get_center()[:2] / tol).astype(int))
                 for m in group)
    want = sorted(tuple(np.round(np.asarray(s, float)[:2] / tol).astype(int))
                  for s in spots)
    return got == want


def _verts(m):
    return [np.asarray(v, float)[:2] for v in m.get_vertices()]


def _same_poly(P, Q, tol=1e-6):
    """Same vertices in the same cyclic order (any start, either sense)."""
    P = [np.asarray(p, float)[:2] for p in P]
    Q = [np.asarray(q, float)[:2] for q in Q]
    if len(P) != len(Q):
        return False
    n = len(P)
    for d in (1, -1):
        for s in range(n):
            if all(close(P[i], Q[(s + d * i) % n], tol) for i in range(n)):
                return True
    return False


def _tiles_once(pieces, cells):
    """True if the cell sets in `pieces` cover the cell set `cells` exactly
    once (no cell missed, none covered twice, nothing outside)."""
    seen = {}
    for p in pieces:
        for c in p:
            seen[c] = seen.get(c, 0) + 1
    return set(seen) == set(cells) and all(v == 1 for v in seen.values())


def _glide(mob, pivot, target, angle, **kw):
    """A rigid motion on every frame: the piece turns by `angle` about its
    pivot while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))

    return UpdateFromAlphaFunc(mob, upd, **kw)


# ------------------------------------------------------------ text helpers

def _on_base(s, size, color):
    """Text s with its baseline at y = 0, as Pango sets it: a reference 'M'
    is laid out in front of it, measured and removed (so a lone '=' or '+'
    sits at its proper height, not with its own bottom on the baseline)."""
    t = Text("M" + s, font_size=size, color=color)
    m = t.submobjects[0]
    base = m.get_bottom()[1]
    t.remove(m)
    t.shift(np.array([0.0, -base, 0.0]))
    return t


def _rich(parts, size=30, color=WHITE, small=0.62, rise=0.48, drop=0.36):
    """One line of text with true sub- and superscripts, without TeX.

    parts: list of (string, kind) or (string, kind, colour); kind is ""
    (normal), "^" (superscript) or "_" (subscript). Normal pieces share one
    baseline; a superscript sits `rise` x-heights above it, a subscript
    `drop` x-heights below it, both smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for part in parts:
        s, kind = part[0], part[1]
        col = part[2] if len(part) > 2 else color
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        body = s.strip(" ")
        x += lead * sp
        if body:
            t = _on_base(body, size * (small if kind else 1.0), col)
            y0 = {"": 0.0, "^": rise * xh, "_": -drop * xh}[kind]
            t.shift(np.array([x - t.get_left()[0], y0, 0.0]))
            x = t.get_right()[0] + (0.012 if kind else 0.02)
            g.add(t)
        x += trail * sp
    return g


def _cap(group):
    """Place a composite formula where caption() puts its Text: centred,
    along the bottom edge."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.set_x(0.0)
    group.to_edge(DOWN, buff=0.3)
    return group


# ====================================================================== B47

def _halves(T):
    """Midpoint split of the triangle T = (P0, P1, P2): its three corner
    triangles and the middle one."""
    P0, P1, P2 = T
    M01, M12, M20 = (P0 + P1) / 2, (P1 + P2) / 2, (P2 + P0) / 2
    return [(P0, M01, M20), (M01, P1, M12), (M20, M12, P2)], (M01, M12, M20)


class B47_SierpinskiNoArea(Board):
    """Sierpiński's triangle: cut a triangle of area 1 by its midlines into
    four triangles — congruent, since a half-turn about the midpoint of a
    midline carries a corner triangle onto the middle one — and remove the
    middle quarter; repeat in every triangle that is left, for ever. Each
    stage removes a quarter of what is left, so after n stages (¾)ⁿ is
    left and ¼ + 3/16 + 9/64 + … has been removed. The removed pieces are
    moved, each by the same slide, into a second copy of the triangle: they
    fill it exactly, except where the pieces still left would go. What is
    left, (¾)ⁿ, tends to 0, so
        ¼ + 3/16 + 9/64 + … = (1/4)·(1 + ¾ + (¾)² + …) = 1.
    Five stages drawn, then "…"."""

    def construct(self):
        side, yb = 5.8, -2.02
        h = side * np.sqrt(3) / 2
        A = np.array([-6.3, yb, 0.0])
        B = A + np.array([side, 0.0, 0.0])
        C = A + np.array([side / 2, h, 0.0])
        D = np.array([6.8, 0.0, 0.0])              # the slide to the copy
        K = 5
        cols = [ORANGE, TEAL_D, GREEN_D, RED_D, PURPLE_B]
        tcols = [ORANGE, TEAL_B, GREEN_B, RED_B, PURPLE_A]
        sw = [2.5, 2.0, 1.5, 1.0, 0.7, 0.45]

        # exact bookkeeping: every split is into four congruent quarters
        left, stages = [(A, B, C)], []
        for k in range(1, K + 1):
            kids, mids = [], []
            for T in left:
                ch, md = _halves(T)
                aT = abs(area(T))
                check(all(abs(abs(area(c)) - aT / 4) < 1e-9 for c in ch)
                      and abs(abs(area(md)) - aT / 4) < 1e-9,
                      f"stage {k}: four quarters of equal area")
                turned = [ch[0][1] + ch[0][2] - p for p in ch[0]]
                check(_same_poly(turned, md),
                      f"stage {k}: a corner, half-turned, is the middle")
                kids += ch
                mids.append(md)
            check(len(mids) == 3 ** (k - 1) and len(kids) == 3 ** k,
                  f"stage {k}: 3^(k-1) removed, 3^k left")
            check(abs(sum(abs(area(c)) for c in kids) + sum(abs(area(m)) for m in mids)
                      - sum(abs(area(T)) for T in left)) < 1e-9,
                  f"stage {k}: what was left = what is left + what is removed")
            stages.append((kids, mids))
            left = kids
        for n in range(1, K + 1):
            rem = sum(Fraction(3 ** (k - 1), 4 ** k) for k in range(1, n + 1))
            check(rem + Fraction(3, 4) ** n == 1,
                  f"after {n} stages: removed + (3/4)^{n} = 1")
        _safe(Polygon(A, B, C), Polygon(A + D, B + D, C + D))

        tri = mk([A, B, C], BLUE_D, FILL, stroke_width=sw[0])
        copy_outline = DashedVMobject(Polygon(A + D, B + D, C + D,
                                              stroke_color=GREY_B,
                                              stroke_width=2.5), num_dashes=75)
        one = tag("1", 44).move_to((A + B + C) / 3)
        self.play(FadeIn(tri), FadeIn(one), Create(copy_outline), run_time=1.2)

        # readouts under the two triangles
        y_ro = yb - 0.46
        x_l = (A[0] + B[0]) / 2
        x_r = x_l + D[0]
        left_txt = ["1", "¾", "(¾)²", "(¾)³", "(¾)⁴", "(¾)⁵"]
        lro = tag(left_txt[0], 28, BLUE_B).move_to([x_l, y_ro, 0])
        terms = ["¼", "+ 3/16", "+ 9/64", "+ 27/256"]
        rro = VGroup(*[tag(t, 24, c) for t, c in zip(terms, tcols)])
        rro.arrange(RIGHT, buff=0.16)
        rdots = tag("+ …", 24, WHITE).next_to(rro, RIGHT, buff=0.16)
        VGroup(rro, rdots).move_to([x_r, y_ro, 0])
        for t in rro:
            t.align_to(rro[0], DOWN)
        rdots.align_to(rro[0], DOWN)
        check(VGroup(rro, rdots).width < side, "the sum fits under its triangle")
        _safe(lro, rro, rdots)

        # ---- stage 1: the midlines; a corner, half-turned, is the middle
        kids, mids = stages[0]
        midl = Polygon(*mids[0], stroke_color=WHITE, stroke_width=sw[1])
        self.play(FadeOut(one), Create(midl), run_time=0.8)
        corner = mk(kids[0], YELLOW_E, 0.45, stroke_color=YELLOW_B, stroke_width=4)
        self.play(FadeIn(corner), run_time=0.35)
        piv = (kids[0][1] + kids[0][2]) / 2
        self.play(Rotate(corner, angle=PI, about_point=piv), run_time=1.4)
        check(_same_poly(_verts(corner), mids[0]),
              "the turned corner covers the middle triangle")
        self.play(FadeOut(corner), FadeIn(lro), run_time=0.4)

        left_m = [mk(c, BLUE_D, FILL, stroke_width=sw[1]) for c in kids]
        mid_m = mk(mids[0], BLUE_D, FILL, stroke_width=sw[1])
        self.remove(tri, midl)
        self.add(*left_m, mid_m)
        quarter = tag("¼", 40).move_to(sum(mids[0]) / 3)
        self.play(mid_m.animate.set_fill(cols[0], opacity=FILL), FadeIn(quarter),
                  run_time=0.5)
        self.play(VGroup(mid_m, quarter).animate.shift(D), run_time=1.2)
        check(_same_poly(_verts(mid_m), [p + D for p in mids[0]]),
              "the removed quarter lands in the copy")
        self.play(Transform(lro, tag(left_txt[1], 28, BLUE_B).move_to(lro)),
                  FadeIn(rro[0]), run_time=0.6)
        self.hold(0.5)

        # ---- stages 2 … 5: the middle of every triangle that is left
        removed = [mid_m]
        for k in range(2, K + 1):
            kids, mids = stages[k - 1]
            fast = k >= 4
            lines = VGroup(*[Polygon(*md, stroke_color=WHITE, stroke_width=sw[k])
                             for md in mids])
            self.play(Create(lines), run_time=0.6 if not fast else 0.45)
            new_left = [mk(c, BLUE_D, FILL, stroke_width=sw[k]) for c in kids]
            new_mid = [mk(md, BLUE_D, FILL, stroke_width=sw[k]) for md in mids]
            self.remove(*left_m, lines)
            self.add(*new_left, *new_mid)
            left_m = new_left
            self.play(*[m.animate.set_fill(cols[k - 1], opacity=FILL) for m in new_mid],
                      run_time=0.35 if not fast else 0.3)
            grp = VGroup(*new_mid)
            self.play(grp.animate.shift(D), run_time=1.0 if not fast else 0.8)
            for m, md in zip(new_mid, mids):
                check(_same_poly(_verts(m), [p + D for p in md]),
                      f"stage {k}: a removed piece lands in the copy")
            removed += new_mid
            anims = [Transform(lro, tag(left_txt[k], 28, BLUE_B).move_to(lro))]
            anims.append(FadeIn(rro[k - 1]) if k <= len(terms) else FadeIn(rdots))
            self.play(*anims, run_time=0.45)
            if k == 2:
                self.hold(0.3)
        self.hold(0.4)

        # ---- what is left fills exactly the holes of the copy
        kids = stages[-1][0]
        ghost = VGroup(*[mk(c, BLUE_D, 0.9, stroke_width=sw[K]) for c in kids])
        self.add(ghost)
        self.play(ghost.animate.shift(D), run_time=1.2)
        for m, c in zip(ghost, kids):
            check(_same_poly(_verts(m), [p + D for p in c]),
                  "a piece that is left lands in a hole of the copy")
        got = sum(abs(area(_verts(m))) for m in removed) + \
            sum(abs(area(_verts(m))) for m in ghost)
        check(abs(got - abs(area([A, B, C]))) < 1e-6,
              "removed + left = the whole triangle")
        self.hold(0.5)
        self.play(FadeOut(ghost), run_time=0.6)
        lim = tag("(¾)ⁿ → 0", 28, BLUE_B).move_to(lro)
        self.play(Transform(lro, lim), run_time=0.7)
        cap = caption("¼ + 3/16 + 9/64 + …   =   ¼ · (1 + ¾ + (¾)² + …)   =   1", 34)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B48

class B48_CantorNoLength(Board):
    """The Cantor set: from the unit interval remove the open middle third,
    from each of the two pieces left remove its middle third, and so on.
    The rows show the stages one under another; each removed piece falls
    straight down into the bottom bar, at the place it came from. Each
    stage removes a third of what is left, so after n stages (⅔)ⁿ is left
    and ⅓ + 2/9 + 4/27 + … has been removed; the pieces left, dropped as
    well, fill the bottom bar's gaps exactly. (⅔)ⁿ tends to 0, so
        ⅓ + 2/9 + 4/27 + … = (1/3)·(1 + ⅔ + (⅔)² + …) = 1.
    Five stages drawn, then "…"."""

    def construct(self):
        L, x0 = 10.6, -6.3
        hb = 0.36
        K = 5
        ys = [3.3 - 0.64 * k for k in range(K + 1)]
        y_rm = -1.62
        cols = [ORANGE, TEAL_D, GREEN_D, RED_D, PURPLE_B]
        tcols = [ORANGE, TEAL_B, GREEN_B, RED_B, PURPLE_A]
        sw = [2.0, 1.8, 1.4, 1.0, 0.7, 0.5]

        def X(t):
            return x0 + L * float(t)

        def bar(a, b, y, color, op=FILL, w=1.5):
            return mk([[X(a), y - hb / 2], [X(b), y - hb / 2],
                       [X(b), y + hb / 2], [X(a), y + hb / 2]], color, op,
                      stroke_width=w)

        # exact bookkeeping with fractions
        rows = [[(Fraction(0), Fraction(1))]]
        mids = []
        for k in range(1, K + 1):
            nxt, md = [], []
            for a, b in rows[-1]:
                t = (b - a) / 3
                nxt += [(a, a + t), (b - t, b)]
                md.append((a + t, b - t))
            check(len(md) == 2 ** (k - 1) and all(b - a == Fraction(1, 3 ** k)
                                                   for a, b in md),
                  f"stage {k}: 2^(k-1) middle thirds of 1/3^k")
            rows.append(nxt)
            mids.append(md)
            rem = sum(Fraction(2 ** (j - 1), 3 ** j) for j in range(1, k + 1))
            check(rem + Fraction(2, 3) ** k == 1, f"stage {k}: removed + (2/3)^k = 1")
        pieces = sorted(rows[-1] + [p for md in mids for p in md])
        check(pieces[0][0] == 0 and pieces[-1][1] == 1
              and all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1)),
              "the removed pieces and the pieces left tile [0, 1]")

        row_m = [VGroup(bar(0, 1, ys[0], BLUE_D, w=sw[0]))]
        rm_outline = DashedVMobject(Polygon(
            [X(0), y_rm - hb / 2, 0], [X(1), y_rm - hb / 2, 0],
            [X(1), y_rm + hb / 2, 0], [X(0), y_rm + hb / 2, 0],
            stroke_color=GREY_B, stroke_width=2), num_dashes=80)
        ticks = VGroup(*[Line([X(t), y_rm - hb / 2 - 0.1, 0],
                              [X(t), y_rm - hb / 2, 0], color=GREY_B,
                              stroke_width=2) for t in (0, 1)])
        lab0 = tag("0", 22, GREY_A).next_to(ticks[0], DOWN, buff=0.06)
        lab1 = tag("1", 22, GREY_A).next_to(ticks[1], DOWN, buff=0.06)
        self.play(FadeIn(row_m[0]), Create(rm_outline), FadeIn(ticks),
                  FadeIn(lab0), FadeIn(lab1), run_time=1.1)

        # readouts: what is left, beside each row; what is removed, below
        x_ro = X(1) + 0.32
        left_txt = ["1", "⅔", "(⅔)²", "(⅔)³", "(⅔)⁴", "(⅔)⁵"]
        lros = [tag(t, 26, BLUE_B) for t in left_txt]
        for t, y in zip(lros, ys):
            t.move_to([x_ro + t.width / 2, y, 0])
        terms = ["⅓", "+ 2/9", "+ 4/27", "+ 8/81", "+ 16/243"]
        rro = VGroup(*[tag(t, 26, c) for t, c in zip(terms, tcols)])
        rro.arrange(RIGHT, buff=0.18)
        rdots = tag("+ …", 26).next_to(rro, RIGHT, buff=0.18)
        VGroup(rro, rdots).move_to([X(0.5), y_rm - 0.62, 0])
        for t in [*rro, rdots]:
            t.align_to(rro[0], DOWN)
        _safe(*lros, rro, rdots, lab0, lab1)
        _apart(lab0, lab1, rro, rdots)
        self.play(FadeIn(lros[0]), run_time=0.4)

        removed = []
        third = None
        for k in range(1, K + 1):
            fast = k >= 4
            # the row is copied one step down ...
            down = VGroup(*[bar(a, b, ys[k - 1], BLUE_D, w=sw[k - 1])
                            for a, b in rows[k - 1]])
            self.add(down)
            self.play(down.animate.shift(DOWN * (ys[k - 1] - ys[k])),
                      run_time=0.6 if not fast else 0.45)
            # ... cut into thirds, and the middle thirds drop to the bottom bar
            outer = VGroup(*[bar(a, b, ys[k], BLUE_D, w=sw[k]) for a, b in rows[k]])
            md_m = VGroup(*[bar(a, b, ys[k], BLUE_D, w=sw[k]) for a, b in mids[k - 1]])
            self.remove(down)
            self.add(outer, md_m)
            row_m.append(outer)
            self.play(md_m.animate.set_fill(cols[k - 1], opacity=FILL),
                      run_time=0.4 if not fast else 0.3)
            extra = []
            if k == 1:
                third = tag("⅓", 24).move_to(md_m[0])
                check(third.height < hb - 0.04, "the label ⅓ fits in its piece")
                extra.append(FadeIn(third))
                self.play(*extra, run_time=0.3)
            grp = VGroup(md_m, third) if k == 1 else md_m
            self.play(grp.animate.shift(DOWN * (ys[k] - y_rm)),
                      run_time=0.9 if not fast else 0.7)
            for m, (a, b) in zip(md_m, mids[k - 1]):
                lo, hi = _box(m)
                check(close(lo[:2], [X(a), y_rm - hb / 2], 1e-6)
                      and close(hi[:2], [X(b), y_rm + hb / 2], 1e-6),
                      f"stage {k}: a middle third lands below its place")
            removed += list(md_m)
            self.play(FadeIn(lros[k]), FadeIn(rro[k - 1]), run_time=0.45)
            if k == 1:
                self.hold(0.3)
        vdots = tag("⋮", 30).move_to([X(0.5), (ys[K] + y_rm) / 2 + 0.05, 0])
        _apart(vdots, rm_outline, row_m[-1], gap=0.05)
        self.play(FadeIn(rdots), FadeIn(vdots), run_time=0.5)
        self.hold(0.4)

        # the pieces left fill exactly the gaps of the bottom bar
        ghost = VGroup(*[bar(a, b, ys[K], BLUE_D, 0.9, w=sw[K]) for a, b in rows[K]])
        self.add(ghost)
        self.play(ghost.animate.shift(DOWN * (ys[K] - y_rm)), run_time=1.0)
        for m, (a, b) in zip(ghost, rows[K]):
            lo, hi = _box(m)
            check(close(lo[:2], [X(a), y_rm - hb / 2], 1e-6)
                  and close(hi[:2], [X(b), y_rm + hb / 2], 1e-6),
                  "a piece that is left lands in a gap")
        cover = sorted([(m.get_left()[0], m.get_right()[0]) for m in removed + list(ghost)])
        check(close(cover[0][0], X(0), 1e-6) and close(cover[-1][1], X(1), 1e-6)
              and all(close(cover[i][1], cover[i + 1][0], 1e-6)
                      for i in range(len(cover) - 1)),
              "removed + left fill the bar without gaps or overlaps")
        self.hold(0.5)
        self.play(FadeOut(ghost), run_time=0.6)
        lim = tag("(⅔)ⁿ → 0", 26, BLUE_B)
        lim.move_to([x_ro + lim.width / 2, (ys[K] + y_rm) / 2 + 0.05, 0])
        _safe(lim)
        _apart(lim, lros[-1], rm_outline, vdots, gap=0.05)
        self.play(FadeIn(lim), run_time=0.6)
        cap = caption("⅓ + 2/9 + 4/27 + …   =   ⅓ · (1 + ⅔ + (⅔)² + …)   =   1", 34)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B41

def _eq_line(left, right, x_eq, y, size=26, color=WHITE, gap=None):
    """'left  =  right' on one baseline y, the '=' sign centred at x_eq (rows
    of a readout line up on their '=' signs)."""
    eq = _on_base("=", size, color)
    eq.shift(np.array([x_eq - eq.get_center()[0], y, 0.0]))
    gap = eq.width / 2 + 0.2 * size / 28 if gap is None else gap
    g = VGroup()
    if left:
        lt = _on_base(left, size, color)
        lt.shift(np.array([x_eq - gap - lt.get_right()[0], y, 0.0]))
        g.add(lt)
    g.add(eq)
    rt = _on_base(right, size, color)
    rt.shift(np.array([x_eq + gap - rt.get_left()[0], y, 0.0]))
    g.add(rt)
    return g


def _stair_outline(H, j0, j1):
    """Outline of the columns j0 … j1 of a staircase whose column j is
    H[j] high, as (x, y) points (math coordinates)."""
    pts = [(j0 - 1, 0), (j1, 0), (j1, H[j1])]
    for j in range(j1, j0, -1):
        pts += [(j - 1, H[j]), (j - 1, H[j - 1])]
    pts.append((j0 - 1, H[j0]))
    return pts


class B41_GabrielStaircase(Board):
    """Gabriel's staircase, drawn for r = 3/5 (the argument is the same for
    every 0 < r < 1). Row k is k blocks, each 1 wide and rᵏ high; the rows
    are stacked from the top down, so the area is S = r + 2r² + 3r³ + ….
    Read by columns instead: column j holds one block of every row k ≥ j,
    so column 1 is c = r + r² + r³ + … high, and every column is the one
    before it squashed to r of its height. Squash column 1 to r of its
    height: it fits exactly under its own top block, so c = r + r·c, i.e.
    c = r/(1 − r). Squash the whole staircase the same way and move it one
    column to the right: it is exactly the staircase without column 1, so
    S = c + r·S, i.e. S = c/(1 − r) = r/(1 − r)². Rows 1–8 are drawn; the
    rows below them are the thin grey band (columns past the eighth are
    "…"). Width and height use different screen scales; the squashes are
    vertical, so they scale every area by r all the same."""

    def construct(self):
        r = Fraction(3, 5)
        rf = float(r)
        N = 8
        wx, hy = 0.84, 3.95
        x0, yf = -5.85, -2.52

        def P(x, y):
            return np.array([x0 + wx * float(x), yf + hy * float(y), 0.0])

        H = {j: r ** j / (1 - r) for j in range(1, N + 3)}      # column heights
        for k in range(1, N + 1):
            check(H[k] - H[k + 1] == r ** k, f"row {k} is r^{k} high")
            check(H[k + 1] == r * H[k], f"column {k + 1} is column {k} squashed by r")
        c = H[1]
        check(c == r + r * c and c == r / (1 - r), "c = r + r·c = r/(1−r)")
        S = (sum(k * r ** k for k in range(1, N + 1)) + N * H[N + 1]
             + H[N + 1] / (1 - r))
        check(S == r / (1 - r) ** 2 and S == c + r * S,
              "rows + band + later columns = c/(1−r) = r/(1−r)²")
        o_all = _stair_outline(H, 1, N - 1)
        o_rest = _stair_outline(H, 2, N)
        check([(x + 1, r * y) for x, y in o_all] == o_rest,
              "the staircase squashed by r and moved one column is the rest")

        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D, RED_D, PURPLE_B, GOLD_D, MAROON_B]
        blocks = {}
        rows = []
        for k in range(1, N + 1):
            row = VGroup()
            for j in range(1, k + 1):
                m = mk([P(*q) for q in [(j - 1, H[k + 1]), (j, H[k + 1]),
                                         (j, H[k]), (j - 1, H[k])]],
                       cols[k - 1], FILL, stroke_width=1.6 if k < 5 else 0.9)
                blocks[(k, j)] = m
                row.add(m)
            rows.append(row)
        band = VGroup(*[mk([P(*q) for q in _rect(j - 1, 0, 1, float(H[N + 1]))],
                           GREY_C, 0.6, stroke_width=0.5) for j in range(1, N + 1)])
        floor = Line(P(0, 0) + LEFT * 0.08, P(N, 0) + RIGHT * 0.7, color=GREY_B,
                     stroke_width=2)
        more = tag("…", 28).move_to(P(N, 0) + RIGHT * 0.42 + UP * 0.16)

        rl_txt = {1: ("r", 34), 2: ("2r²", 32), 3: ("3r³", 28), 4: ("4r⁴", 24)}
        rlabs = {}
        for k, (txt, sz) in rl_txt.items():
            t = tag(txt, sz)
            t.move_to(P(k, (H[k] + H[k + 1]) / 2) + RIGHT * (0.16 + t.width / 2))
            check(t.height < float(H[k] - H[k + 1]) * hy - 0.08,
                  f"row {k}: its label fits beside it")
            rlabs[k] = t
        _safe(floor, more, *rlabs.values(), *rows, band)
        for t in rlabs.values():
            _clear_of(t, list(blocks.values()), 0.04, "row label")

        # ---- the rows, from the top down
        self.play(Create(floor), run_time=0.5)
        for k in range(1, N + 1):
            anims = [LaggedStart(*[FadeIn(m, shift=DOWN * 0.08) for m in rows[k - 1]],
                                 lag_ratio=0.12)]
            if k in rlabs:
                anims.append(FadeIn(rlabs[k]))
            if k == N:
                anims += [FadeIn(band), FadeIn(more)]
            self.play(*anims, run_time=0.8 if k <= 3 else 0.45)
        x_eq = 0.75
        ln = [_eq_line("S", "r + 2r² + 3r³ + 4r⁴ + …", x_eq, 2.95, 28, WHITE),
              _eq_line("c", "r + r² + r³ + r⁴ + …", x_eq, 2.2, 28, YELLOW_B),
              _eq_line("c", "r + r·c", x_eq, 1.45, 28, YELLOW_B),
              _eq_line("", "r/(1−r)", x_eq, 0.85, 28, YELLOW_B),
              _eq_line("S", "c + r·S", x_eq, 0.0, 28, WHITE),
              _eq_line("", "c/(1−r)  =  r/(1−r)²", x_eq, -0.6, 28, WHITE)]
        _safe(*ln)
        _apart(*ln, gap=0.08)
        for l in ln:
            _clear_of(l, [*blocks.values(), *band, more, floor, *rlabs.values()],
                      0.15, "readout line")
        whole = Polygon(*[P(*q) for q in _stair_outline(H, 1, N)],
                        stroke_color=YELLOW_B, stroke_width=4)
        self.play(FadeIn(ln[0]), Create(whole), run_time=0.9)
        self.play(FadeOut(whole), run_time=0.4)
        self.hold(0.3)

        # ---- read by columns: column 1 is r + r² + r³ + …
        outl = VGroup(*[Polygon(*[P(*q) for q in _rect(j - 1, 0, 1, float(H[j]))],
                                stroke_color=YELLOW_B, stroke_width=3)
                        for j in range(1, N + 1)])
        self.play(LaggedStart(*[Create(o) for o in outl], lag_ratio=0.15),
                  run_time=1.2)
        d_c = _dim(P(0, 0), P(0, H[1]), "c", LEFT * 0.17, YELLOW_B, 30, gap=0.1)
        _safe(d_c)
        self.play(FadeOut(VGroup(*outl[1:])), outl[0].animate.set_stroke(width=5),
                  FadeIn(d_c), FadeIn(ln[1]), run_time=0.8)
        self.hold(0.4)

        # ---- column 1 squashed to r of its height fits under its top block
        sq1 = Polygon(*[P(*q) for q in _rect(0, 0, 1, float(H[1]))],
                      stroke_color=YELLOW_B, stroke_width=4, fill_color=YELLOW_E,
                      fill_opacity=0.35)
        self.add(sq1)
        self.play(FadeIn(sq1), run_time=0.3)
        self.play(sq1.animate.stretch(rf, 1, about_point=P(0, 0)), run_time=1.5)
        check(_same_poly(_verts(sq1), [P(*q) for q in _rect(0, 0, 1, float(H[2]))]),
              "column 1 squashed by r fits exactly under its top block")
        lab_rc = tag("r·c", 26).move_to(P(0.5, (H[3] + H[4]) / 2))
        check(lab_rc.width < wx - 0.1 and lab_rc.get_top()[1] < P(0, H[3])[1] - 0.04
              and lab_rc.get_bottom()[1] > P(0, H[4])[1] + 0.04,
              "r·c inside one block of the squashed column")
        self.play(FadeIn(lab_rc), FadeIn(ln[2]), run_time=0.6)
        self.play(FadeIn(ln[3]), run_time=0.5)
        self.hold(0.5)
        self.play(FadeOut(sq1), FadeOut(lab_rc), run_time=0.4)

        # ---- the whole staircase squashed by r, moved one column: the rest
        sq2 = Polygon(*[P(*q) for q in o_all], stroke_color=YELLOW_B, stroke_width=4,
                      fill_color=YELLOW_E, fill_opacity=0.35)
        whole = Polygon(*[P(*q) for q in _stair_outline(H, 1, N)],
                        stroke_color=YELLOW_B, stroke_width=4)
        self.play(Create(whole), *[FadeOut(t) for t in rlabs.values()], run_time=0.6)
        self.add(sq2)
        self.play(FadeIn(sq2), FadeOut(whole), run_time=0.4)
        self.play(sq2.animate.stretch(rf, 1, about_point=P(0, 0)).shift(RIGHT * wx),
                  run_time=1.8)
        check(_same_poly(_verts(sq2), [P(*q) for q in o_rest]),
              "the squashed staircase, one column on, is the staircase less column 1")
        lab_rS = tag("r·S", 26).move_to(P(1.5, (H[2] + H[3]) / 2))
        check(lab_rS.width < wx - 0.1 and lab_rS.get_top()[1] < P(0, H[2])[1] - 0.04
              and lab_rS.get_bottom()[1] > P(0, H[3])[1] + 0.04,
              "r·S inside one block of column 2")
        self.play(FadeIn(lab_rS), FadeIn(ln[4]), *[FadeIn(t) for t in rlabs.values()],
                  run_time=0.6)
        self.play(FadeIn(ln[5]), run_time=0.6)
        _apart(lab_rS, *rlabs.values(), d_c[1], gap=0.05)
        cap = caption("r + 2r² + 3r³ + 4r⁴ + …   =   r/(1−r)²,     0 < r < 1", 34)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B32

def _cells(cells, u, org, color, op=FILL, stroke=1.3):
    """Unit cells (i, j) of side u on the grid whose (0, 0) corner is org."""
    return VGroup(*[cell(i, j, u, org, color, op, stroke) for (i, j) in cells])


def _outline_cells(cells, u, org, color=WHITE, width=4):
    """The boundary of a set of grid cells, as one VGroup of unit edges (the
    edges that belong to exactly one cell of the set)."""
    edges = {}
    for (i, j) in cells:
        for e in (((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)),
                  ((i, j + 1), (i + 1, j + 1)), ((i, j), (i, j + 1))):
            edges[e] = edges.get(e, 0) + 1
    o = to3(org)
    return VGroup(*[Line(o + u * np.array([a[0], a[1], 0.0]),
                         o + u * np.array([b[0], b[1], 0.0]),
                         color=color, stroke_width=width)
                    for (a, b), cnt in edges.items() if cnt == 1])


class B32_DoublingTriangular(Board):
    """The staircase T₂ₙ (columns of 1, 2, …, 2n cells) is cut by two
    straight cuts into the staircase Tₙ on the left, the staircase Tₙ on
    top and the n × n square under it; a staircase cut splits the square
    into Tₙ and Tₙ₋₁ (upside down). Three of the four pieces are slides of
    one another; the last, given a half-turn, is Tₙ₋₁:
        T₂ₙ = 3Tₙ + Tₙ₋₁.     Drawn for n = 4 (36 = 3·10 + 6)."""

    def construct(self):
        n, u = 4, 0.43
        N = 2 * n
        O = np.array([-6.4, -1.55, 0.0])

        def C(x, y):
            return O + u * np.array([float(x), float(y), 0.0])

        T = [(i, j) for i in range(N) for j in range(i + 1)]
        pA = [(i, j) for (i, j) in T if i < n]
        pC = [(i, j) for (i, j) in T if i >= n and j < n and j <= i - n]
        pB = [(i, j) for (i, j) in T if i >= n and j >= n]
        pD = [(i, j) for (i, j) in T if i >= n and j < n and j > i - n]
        Tn = {(i, j) for i in range(n) for j in range(i + 1)}
        Tm = {(i, j) for i in range(n - 1) for j in range(i + 1)}
        check(_tiles_once([pA, pB, pC, pD], T), "the four pieces tile T(2n)")
        check(set(pA) == Tn, "the left piece is T(n)")
        check({(i - n, j) for (i, j) in pC} == Tn, "the lower square part is T(n)")
        check({(i - n, j - n) for (i, j) in pB} == Tn, "the top piece is T(n)")
        check({(2 * n - 2 - i, n - 1 - j) for (i, j) in pD} == Tm,
              "the upper square part, half-turned, is T(n−1)")
        check(len(T) == 3 * len(Tn) + len(Tm) == 36, "36 = 3·10 + 6")

        # slots on the right: T(n), T(n), T(n), T(n−1), with + between
        x_eq = C(N, 0)[0] + 0.42
        sx, gap = x_eq + 0.42, 0.66
        slot = [sx, sx + n * u + gap, sx + 2 * (n * u + gap), sx + 3 * (n * u + gap)]
        y_mid = O[1] + n * u / 2

        def S(k, x, y):
            return np.array([slot[k] + u * x, O[1] + u * y, 0.0])

        # ---- the staircase T(2n), column by column
        g = {}
        for name, cells in (("A", pA), ("B", pB), ("C", pC), ("D", pD)):
            g[name] = {c: cell(c[0], c[1], u, O, GREY_C, FILL, 1.3) for c in cells}
        cols = [VGroup(*[g[nm][c] for nm in g for c in g[nm] if c[0] == i])
                for i in range(N)]
        lab_T = _rich([("T", ""), ("2n", "_")], 36).move_to(C(1.7, 5.6))
        d_2n = _dim(C(0, 0), C(N, 0), "2n", DOWN * 0.2, YELLOW_B, 28, gap=0.08)
        allcells = [m for nm in g for m in g[nm].values()]
        _clear_of(lab_T, allcells, 0.08, "T(2n)")
        _safe(lab_T, d_2n, *allcells)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cols],
                              lag_ratio=0.2), run_time=1.8)
        self.play(FadeIn(lab_T), FadeIn(d_2n), run_time=0.6)
        self.hold(0.4)

        # ---- two straight cuts and a staircase cut
        cut1 = DashedLine(C(n, 0), C(n, n), color=YELLOW_B, stroke_width=6,
                          dash_length=0.1)
        cut2 = DashedLine(C(n, n), C(N, n), color=YELLOW_B, stroke_width=6,
                          dash_length=0.1)
        sp = [C(n, 1)]
        for p in range(n - 1):
            sp += [C(n + p + 1, p + 1), C(n + p + 1, p + 2)]
        cut3 = DashedVMobject(VMobject(stroke_color=YELLOW_B, stroke_width=6)
                              .set_points_as_corners(sp), num_dashes=24)
        d_n1 = _dim(C(0, 0), C(n, 0), "n", DOWN * 0.2, YELLOW_B, 28, gap=0.08)
        d_n2 = _dim(C(n, 0), C(N, 0), "n", DOWN * 0.2, YELLOW_B, 28, gap=0.08)
        self.play(Create(cut1), Create(cut2), run_time=0.8)
        self.play(Create(cut3), ReplacementTransform(d_2n, VGroup(d_n1, d_n2)),
                  run_time=0.9)
        fill = {"A": BLUE_D, "C": GREEN_D, "B": PURPLE_B, "D": ORANGE}
        self.play(*[m.animate.set_fill(fill[nm]) for nm in g for m in g[nm].values()],
                  run_time=0.7)
        rims = {nm: _outline_cells(list(g[nm]), u, O, WHITE, 5) for nm in g}
        self.play(FadeOut(cut1), FadeOut(cut2), FadeOut(cut3),
                  *[Create(rims[nm]) for nm in g], run_time=0.7)
        self.hold(0.4)

        # ---- copies of the pieces, moved into a row: one half-turn, three
        # slides (the farthest first, so nothing passes over a placed piece)
        copies = {}
        cpD = VGroup(VGroup(*[m.copy() for m in g["D"].values()]), rims["D"].copy())
        self.add(cpD)
        piv = C(1.5 * n, 0.5 * n)
        tgt = piv + (S(3, 0, 0) - C(n + 1, 0))
        self.play(_glide(cpD, piv, tgt, PI), run_time=1.4)
        check(_same_spots(cpD[0], [S(3, i + 0.5, j + 0.5) for (i, j) in Tm]),
              "the upper square part, half-turned, is the staircase T(n−1)")
        copies["D"] = cpD
        moves = [("B", 2, C(n, n)), ("C", 1, C(n, 0)), ("A", 0, C(0, 0))]
        for nm, k, corner in moves:
            cp = VGroup(VGroup(*[m.copy() for m in g[nm].values()]), rims[nm].copy())
            self.add(cp)
            self.play(cp.animate.shift(S(k, 0, 0) - corner), run_time=0.9)
            off = {"A": (0, 0), "C": (n, 0), "B": (n, n)}[nm]
            check(_same_spots(cp[0], [S(k, i - off[0] + 0.5, j - off[1] + 0.5)
                                      for (i, j) in g[nm]]),
                  f"piece {nm} slides onto its place in the row")
            copies[nm] = cp

        labs = [_rich([("T", ""), ("n", "_")], 32, c) for c in (BLUE_B, GREEN_B, PURPLE_A)]
        labs.append(_rich([("T", ""), ("n−1", "_")], 32, ORANGE))
        for k, t in enumerate(labs):
            w = (n if k < 3 else n - 1) * u
            t.move_to([slot[k] + w / 2, O[1] - 0.42, 0.0])
        signs = [tag("=", 36).move_to([x_eq, y_mid, 0])]
        for k in range(3):
            signs.append(tag("+", 36).move_to([(slot[k] + n * u + slot[k + 1]) / 2,
                                               y_mid, 0]))
        nums = [tag(t, 28, GREY_A) for t in ("36", "=", "10", "+", "10", "+", "10",
                                              "+", "6")]
        xs = [C(n, 0)[0], x_eq]
        for k in range(4):
            w = (n if k < 3 else n - 1) * u
            xs.append(slot[k] + w / 2)
            if k < 3:
                xs.append((slot[k] + n * u + slot[k + 1]) / 2)
        y_num = C(0, N)[1] + 0.45
        for t, x in zip(nums, xs):
            t.move_to([x, y_num, 0.0])
        pieces = [m for cp in copies.values() for m in cp[0]]
        for t in labs + signs:
            _clear_of(t, pieces + allcells, 0.06, "row label")
        _apart(*labs, *signs, d_n1[1], d_n2[1], lab_T, *nums, gap=0.06)
        _safe(*labs, *signs, *nums, *pieces)
        self.play(*[FadeIn(t) for t in labs + signs], run_time=0.8)
        self.play(FadeIn(VGroup(*nums)), run_time=0.6)
        cap = _cap(_rich([("T", ""), ("2n", "_"), ("  =  3T", ""), ("n", "_"),
                          ("  +  T", ""), ("n−1", "_")], 38, YELLOW_B))
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B38

def _line(s, x_left, y, size=28, color=WHITE):
    """Text s, its left end at x_left and its baseline at y."""
    t = _on_base(s, size, color)
    t.shift(np.array([x_left - t.get_left()[0], y, 0.0]))
    return t


class B38_PowersOfThree(Board):
    """Start from one cell. Lay two copies of the figure against it on
    opposite sides — first left and right, then below and above, and so on,
    turn by turn: with each pair the figure triples, so after the pairs of
    1, 3, 9, …, 3ⁿ⁻¹ cells it holds 3ⁿ cells. The copies on one side and
    the copies on the other are swapped by a half-turn about the middle
    cell (shown as two turn-overs, about the middle row and the middle
    column), so each side holds half of 3ⁿ − 1:
        1 + 3 + 9 + … + 3ⁿ⁻¹ = (3ⁿ − 1)/2.     Drawn for n = 4 (a 9 × 9 board)."""

    def construct(self):
        u = 0.64
        G0 = np.array([-6.25, -2.62, 0.0])
        K = 4

        def GC(x, y):
            return G0 + u * np.array([x + 0.5, y + 0.5, 0.0])

        region, steps = [(4, 4)], []
        for k in range(K):
            xs = [x for x, _ in region]
            ys = [y for _, y in region]
            w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
            d = (w, 0) if k % 2 == 0 else (0, h)       # left/right, then below/above
            blue = [(x - d[0], y - d[1]) for x, y in region]
            orng = [(x + d[0], y + d[1]) for x, y in region]
            check(not (set(blue) | set(orng)) & set(region) and not set(blue) & set(orng),
                  f"step {k}: the two copies lie beside the figure, apart")
            check(len(blue) == len(orng) == 3 ** k, f"step {k}: copies of 3^{k} cells")
            steps.append((list(region), blue, orng, d))
            region = region + blue + orng
            check(len(region) == 3 ** (k + 1), f"step {k}: the figure triples")
        check(set(region) == {(x, y) for x in range(9) for y in range(9)},
              "the figure is the 9 x 9 board")
        all_b = [c for st in steps for c in st[1]]
        all_o = [c for st in steps for c in st[2]]
        check({(8 - x, 8 - y) for x, y in all_b} == set(all_o),
              "a half-turn about the middle cell swaps the two sides")
        check(1 + 2 * sum(3 ** k for k in range(K)) == 3 ** K, "1 + 2(1+3+9+27) = 81")

        def block(cells, color, op=FILL):
            sq = VGroup(*[Square(side_length=u, fill_color=color, fill_opacity=op,
                                 stroke_color=WHITE, stroke_width=1.0,
                                 stroke_opacity=0.6).move_to(GC(x, y))
                          for (x, y) in cells])
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            box = Rectangle(width=(max(xs) - min(xs) + 1) * u,
                            height=(max(ys) - min(ys) + 1) * u,
                            stroke_color=WHITE, stroke_width=3.5)
            box.move_to(G0 + u * np.array([(max(xs) + min(xs) + 1) / 2,
                                           (max(ys) + min(ys) + 1) / 2, 0.0]))
            return VGroup(sq, box)

        mid = block([(4, 4)], YELLOW_E, 0.95)
        one = tag("1", 28, BLACK).move_to(GC(4, 4))
        self.play(FadeIn(mid), FadeIn(one), run_time=0.8)

        x_l = 0.25
        rows = [_line(f"{3 ** k} + 2 · {3 ** k}  =  {3 ** (k + 1)}", x_l,
                      2.85 - 0.68 * k, 30) for k in range(K)]
        sizes = [26, 32, 40, 46]
        placed_b, placed_o = [], []
        for k, (fig, blue, orng, d) in enumerate(steps):
            gb = block(fig, BLUE_D, 0.45)
            go = block(fig, ORANGE, 0.45)
            self.add(gb, go)
            self.play(FadeIn(gb), FadeIn(go), run_time=0.4)
            v = u * np.array([d[0], d[1], 0.0])
            self.play(gb.animate.shift(-v), go.animate.shift(v),
                      gb[0].animate.shift(-v).set_fill(opacity=FILL),
                      go[0].animate.shift(v).set_fill(opacity=FILL),
                      run_time=1.1 if k < 3 else 1.3)
            check(_same_spots(gb[0], [GC(*c) for c in blue])
                  and _same_spots(go[0], [GC(*c) for c in orng]),
                  f"step {k}: both copies land beside the figure")
            nb = tag(str(3 ** k), sizes[k]).move_to(gb[1])
            no = tag(str(3 ** k), sizes[k]).move_to(go[1])
            for t, bx in ((nb, gb[1]), (no, go[1])):
                check(t.width < bx.width - 0.12 and t.height < bx.height - 0.12,
                      "a block's number fits inside it")
            self.play(FadeIn(nb), FadeIn(no), FadeIn(rows[k], shift=LEFT * 0.2),
                      run_time=0.55)
            placed_b.append(VGroup(gb, nb))
            placed_o.append(VGroup(go, no))
            if k < 2:
                self.hold(0.3)
        self.hold(0.4)

        fin1 = _line("1 + 2 · (1 + 3 + 9 + 27)  =  81", x_l, -0.2, 27, YELLOW_B)
        fin2 = _line("1 + 3 + 9 + 27  =  (81 − 1)/2", x_l, -0.95, 27, YELLOW_B)
        _safe(*rows, fin1, fin2)
        _apart(*rows, fin1, fin2, gap=0.08)
        board_right = G0[0] + 9 * u
        check(min(t.get_left()[0] for t in rows + [fin1, fin2]) > board_right + 0.4,
              "the readout clear of the board")
        self.play(FadeIn(fin1), run_time=0.7)

        # the two sides are swapped by a half-turn about the middle cell, shown
        # as two turn-overs (about the row and the column through the middle
        # cell), so the moving copy never leaves the board
        turned = VGroup(*[b[0][0].copy() for b in placed_b])
        turned.set_fill(BLUE_D, opacity=0.9).set_stroke(YELLOW_B, width=2.5, opacity=1)
        self.add(turned)
        self.play(Rotate(turned, angle=PI, axis=RIGHT, about_point=GC(4, 4)),
                  run_time=1.2)
        check(_same_spots([m for sq in turned for m in sq],
                          [GC(x, 8 - y) for (x, y) in all_b]),
              "the first turn-over mirrors the blue side in the middle row")
        self.play(Rotate(turned, angle=PI, axis=UP, about_point=GC(4, 4)), run_time=1.2)
        check(_same_spots([m for sq in turned for m in sq], [GC(*c) for c in all_o]),
              "the turned blue side covers the orange side exactly")
        self.play(FadeOut(turned), run_time=0.5)
        self.play(FadeIn(fin2), run_time=0.7)
        frame = Square(side_length=9 * u, stroke_color=YELLOW_B, stroke_width=5)
        frame.move_to(G0 + 4.5 * u * np.array([1.0, 1.0, 0.0]))
        _safe(frame)
        self.play(Create(frame), run_time=0.8)
        cap = _cap(_rich([("1 + 3 + 9 + … + 3", ""), ("n−1", "^"),
                          ("  =  (3ⁿ − 1)/2", "")], 36, YELLOW_B))
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B39

class B39_AlternatingGeometric(Board):
    """1 − ½ + ¼ − ⅛ + …, done literally on a unit square: fill the square,
    empty its left half, fill the lower-left quarter, empty its left half,
    and so on. Each pair of steps leaves the right half of a square of
    side 1/2ᵖ filled: ½, ⅛, 1/32, …. The square falls apart into nested
    L's; each L is three equal squares (a copy of its empty top-left square
    slides onto the other two), and in every L exactly two of the three are
    filled — the empty ones are ¼ + 1/16 + 1/64 + …, the filled ones twice
    that. The L's fill the square, the shrinking corner aside, so
        1 − ½ + ¼ − ⅛ + … = ½ + ⅛ + 1/32 + … = ⅔.
    Four pairs drawn, then "…"."""

    def construct(self):
        S = 5.7
        O = np.array([-6.2, -2.68, 0.0])
        K = 4

        def P(x, y):
            return O + S * np.array([float(x), float(y), 0.0])

        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D]
        tcols = [BLUE_B, TEAL_B, ORANGE, GREEN_B]
        # exact bookkeeping: pair p fills Q_p = [0, 2^-p]^2 and empties its left half
        side = [Fraction(1, 2 ** p) for p in range(K + 1)]
        for p in range(K):
            s, hh = side[p], side[p] / 2
            check(s * s - hh * s == hh * s and hh * s == 2 * hh * hh,
                  f"pair {p}: what stays filled is two squares of side 2^-(p+1)")
        alt = sum(Fraction((-1) ** i, 2 ** i) for i in range(2 * K))
        filled = sum(side[p] / 2 * side[p] for p in range(K))
        empty = sum((side[p] / 2) ** 2 for p in range(K))
        check(alt == filled and filled == 2 * empty
              and filled + empty + side[K] ** 2 == 1,
              "filled = 2 × empty, and filled + empty + corner = 1")
        check(filled == Fraction(2, 3) * (1 - Fraction(1, 4 ** K)), "partial sum = ⅔(1 − 4^-K)")

        outline = Polygon(P(0, 0), P(1, 0), P(1, 1), P(0, 1), stroke_color=WHITE,
                          stroke_width=3)
        d_1 = _dim(P(0, 0), P(0, 1), "1", LEFT * 0.17, GREY_A, 28, gap=0.08)
        _safe(outline, d_1)

        # readout: one row per pair, '=' signs lined up
        x_eq = 3.65
        terms_a = ["1", "+ ¼", "+ 1/16", "+ 1/64"]           # filled ...
        terms_b = ["− ½", "− ⅛", "− 1/32", "− 1/128"]        # ... then half emptied
        rights = ["½", "+ ⅛", "+ 1/32", "+ 1/128"]
        ys = [2.95, 2.25, 1.55, 0.85]
        rows_a, rows_b, rows_e, rows_r = [], [], [], []
        for p in range(K):
            ln = _eq_line("", rights[p], x_eq, ys[p], 28, tcols[p])
            gap = ln[0].width / 2 + 0.2
            # the natural gap between the two terms, as one Text would set it
            whole = _on_base(terms_a[p] + " " + terms_b[p], 28, tcols[p]).width
            sp_ = (whole - _on_base(terms_a[p], 28, tcols[p]).width
                   - _on_base(terms_b[p], 28, tcols[p]).width)
            tb = _on_base(terms_b[p], 28, tcols[p])
            tb.shift(np.array([x_eq - gap - tb.get_right()[0], ys[p], 0.0]))
            ta = _on_base(terms_a[p], 28, tcols[p])
            ta.shift(np.array([tb.get_left()[0] - sp_ - ta.get_right()[0], ys[p], 0.0]))
            rows_a.append(ta)
            rows_b.append(tb)
            rows_e.append(ln[0])
            rows_r.append(ln[1])
        more_l = _line("+ …", rows_a[-1].get_left()[0], 0.15, 28)
        more_r = _line("+ …", rows_r[-1].get_left()[0], 0.15, 28)
        _safe(*rows_a, *rows_b, *rows_r, more_l, more_r)
        _apart(*rows_a, *rows_b, *rows_e, *rows_r, gap=0.05)
        check(min(t.get_left()[0] for t in rows_a) > P(1, 0)[0] + 0.4,
              "readout clear of the square")

        self.play(Create(outline), FadeIn(d_1), run_time=0.9)
        pieces = []
        for p in range(K):
            s = float(side[p])
            fast = p >= 2
            q = mk([P(*v) for v in _rect(0, 0, s, s)], cols[p], FILL,
                   stroke_width=2 if p < 3 else 1.2)
            self.add(q)
            self.bring_to_front(outline)
            q.set_opacity(0)
            self.play(q.animate.set_fill(opacity=FILL).set_stroke(opacity=1),
                      FadeIn(rows_a[p]), run_time=0.8 if not fast else 0.5)
            # the left half empties
            lh = mk([P(*v) for v in _rect(0, 0, s / 2, s)], cols[p], FILL,
                    stroke_width=2 if p < 3 else 1.2)
            rh = mk([P(*v) for v in _rect(s / 2, 0, s / 2, s)], cols[p], FILL,
                    stroke_width=2 if p < 3 else 1.2)
            self.remove(q)
            self.add(lh, rh)
            self.bring_to_front(outline)
            self.play(lh.animate.set_fill(opacity=0).set_stroke(opacity=0),
                      FadeIn(rows_b[p]), run_time=0.7 if not fast else 0.45)
            self.remove(lh)
            pieces.append(rh)
            self.play(FadeIn(rows_e[p]), FadeIn(rows_r[p]),
                      run_time=0.5 if not fast else 0.35)
            if p == 0:
                self.hold(0.4)
        corner = float(side[K])
        dots = tag("…", 18).move_to(P(corner / 2, corner * 0.62))
        check(dots.width < corner * S - 0.06, "the dots fit in the corner that is left")
        self.play(FadeIn(more_l), FadeIn(more_r), FadeIn(dots), run_time=0.5)
        self.hold(0.5)

        # ---- the nested L's: three equal squares each, two of them filled
        Ls = VGroup()
        splits = VGroup()
        tls = VGroup()
        for p in range(K):
            s = float(side[p])
            h = s / 2
            Ls.add(Polygon(P(h, 0), P(s, 0), P(s, s), P(0, s), P(0, h), P(h, h),
                           stroke_color=YELLOW_B, stroke_width=5 if p < 2 else 3))
            splits.add(Line(P(h, h), P(s, h), color=WHITE,
                            stroke_width=3 if p < 2 else 1.5))
            tls.add(DashedVMobject(Polygon(*[P(*v) for v in _rect(0, h, h, h)],
                                           stroke_color=GREY_A,
                                           stroke_width=2.5 if p < 2 else 1.5),
                                   num_dashes=[28, 20, 12, 8][p]))
        self.play(Create(Ls), Create(splits), Create(tls), run_time=1.3)
        h0 = 0.5
        ghost = Polygon(*[P(*v) for v in _rect(0, h0, h0, h0)], stroke_color=YELLOW_B,
                        stroke_width=4, fill_color=YELLOW_E, fill_opacity=0.3)
        q_tl = tag("¼", 44, GREY_A).move_to(P(h0 / 2, 1.5 * h0))
        self.play(FadeIn(ghost), FadeIn(q_tl), run_time=0.4)
        self.play(ghost.animate.shift(RIGHT * S * h0), run_time=0.8)
        check(_same_poly(_verts(ghost), [P(*v) for v in _rect(h0, h0, h0, h0)]),
              "the empty square slides exactly onto the upper filled one")
        q_tr = tag("¼", 44).move_to(P(1.5 * h0, 1.5 * h0))
        self.play(FadeIn(q_tr), run_time=0.3)
        self.play(ghost.animate.shift(DOWN * S * h0), run_time=0.8)
        check(_same_poly(_verts(ghost), [P(*v) for v in _rect(h0, 0, h0, h0)]),
              "and onto the lower filled one")
        q_br = tag("¼", 44).move_to(P(1.5 * h0, 0.5 * h0))
        self.play(FadeIn(q_br), FadeOut(ghost), run_time=0.5)
        self.hold(0.4)

        tot = _eq_line("", "⅔", x_eq, -0.75, 34, YELLOW_B)
        rule = Line([rows_r[0].get_left()[0] - 0.1, -0.28, 0],
                    [rows_r[0].get_left()[0] + 1.6, -0.28, 0], color=YELLOW_B,
                    stroke_width=3)
        _safe(tot, rule)
        _apart(tot, more_l, more_r, rule, gap=0.06)
        segs = [(L_.get_vertices()[i], L_.get_vertices()[(i + 1) % 6])
                for L_ in Ls for i in range(6)]
        segs += [(sp_.get_start(), sp_.get_end()) for sp_ in splits]
        for t in (q_tl, q_tr, q_br):
            for a_, b_ in segs:
                _clear_of_segment(t, a_, b_, 0.08, f"the label {t.text}")
        self.play(Create(rule), FadeIn(tot), run_time=0.8)
        cap = caption("1 − ½ + ¼ − ⅛ + …   =   ½ + ⅛ + 1/32 + …   =   ⅔", 34)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B37

_SUBS = "₀₁₂₃₄₅₆₇₈₉"


def _fname(k):
    return "F" + "".join(_SUBS[int(c)] for c in str(k))


class B37_FibonacciOddEven(Board):
    """Bars of F₁, F₂, …, F₂ₙ₊₁ cells, one per row, left-aligned; each row
    is built from copies of the row above it and the row two above it, so
    whatever a row has beyond the row above is a copy of the row two above:
    Fₖ − Fₖ₋₁ = Fₖ₋₂. Now lay copies of the even rows F₂ₙ, F₂ₙ₋₂, … end to
    end along the bar F₂ₙ₊₁: after each one, what is still uncovered is the
    next odd Fibonacci number down (F₂ₙ₋₁, F₂ₙ₋₃, …), so the next even bar
    fits, and in the end exactly one cell (F₁) is left. In the same way the
    odd rows F₂ₙ₋₁, F₂ₙ₋₃, …, F₃ cover the bar F₂ₙ up to its last cell, F₂,
    which F₁ fills:
        F₁ + F₃ + … + F₂ₙ₋₁ = F₂ₙ,   F₂ + F₄ + … + F₂ₙ = F₂ₙ₊₁ − 1.
    Drawn for n = 4."""

    def construct(self):
        n = 4
        F = [0, 1, 1]
        while len(F) < 24:
            F.append(F[-1] + F[-2])
        R = 2 * n + 1
        u, pitch = 0.355, 0.5
        x0, ytop = -5.95, 3.42
        for m in range(1, 9):
            check(sum(F[1:2 * m:2]) == F[2 * m], f"odd sum, n = {m}")
            check(sum(F[2:2 * m + 1:2]) == F[2 * m + 1] - 1, f"even sum, n = {m}")

        def yc(r):
            return ytop - (r - 1) * pitch

        def cc(i, r):
            return np.array([x0 + (i + 0.5) * u, yc(r), 0.0])

        def colr(r):
            return BLUE_D if r % 2 else ORANGE

        def bar(length, start, r, color, op=FILL):
            return VGroup(*[Square(side_length=u, fill_color=color, fill_opacity=op,
                                   stroke_color=WHITE, stroke_width=1.0)
                            .move_to(cc(start + i, r)) for i in range(length)])

        # the landing plans, and what stays uncovered after each bar
        def plan(top):                    # bars top-1, top-3, … into the bar F[top]
            srcs = list(range(top - 1, 0, -2))
            starts, left = [], []
            pos = 0
            for s in srcs:
                starts.append(pos)
                pos += F[s]
                left.append(F[top] - pos)
            return srcs, starts, left

        ev_src, ev_start, ev_left = plan(R)          # 8, 6, 4, 2 into F9
        od_src, od_start, od_left = plan(R - 1)      # 7, 5, 3, 1 into F8
        check(ev_left == [F[s - 1] for s in ev_src] and ev_left[-1] == 1,
              "after each even bar the gap is the next odd Fibonacci; one cell is left")
        check(od_left[:-1] == [F[s - 1] for s in od_src[:-1]] and od_left[-1] == 0
              and F[2] == F[1], "after each odd bar the gap is the next even Fibonacci;"
                                " the last gap F2 takes F1")

        # ---- the ladder: each row = the row above + the row two above
        rows = {1: bar(1, 0, 1, colr(1)), 2: bar(1, 0, 2, colr(2))}
        labs = {}
        for r in range(1, R + 1):
            t = tag(_fname(r), 22, BLUE_B if r % 2 else ORANGE)
            t.move_to([x0 - 0.16 - t.width / 2, yc(r), 0.0])
            labs[r] = t
        _safe(*labs.values(), bar(F[R], 0, R, GREY))
        self.play(FadeIn(rows[1]), FadeIn(rows[2]), FadeIn(labs[1]), FadeIn(labs[2]),
                  run_time=0.8)
        for r in range(3, R + 1):
            a = rows[r - 1].copy()
            b = rows[r - 2].copy()
            self.add(a, b)
            self.play(a.animate.shift(DOWN * pitch).set_fill(colr(r)),
                      b.animate.shift(DOWN * 2 * pitch + RIGHT * F[r - 1] * u)
                      .set_fill(colr(r)),
                      FadeIn(labs[r]), run_time=0.75 if r < 6 else 0.6)
            rows[r] = VGroup(*a, *b)
            check(_same_spots(rows[r], [cc(i, r) for i in range(F[r])]),
                  f"row {r} = row {r - 1} + row {r - 2}")
        self.hold(0.5)

        def gap_box(a, b, r):
            return Rectangle(width=(b - a) * u, height=u, stroke_color=YELLOW_B,
                             stroke_width=4).move_to((cc(a, r) + cc(b - 1, r)) / 2)

        def gap_lab(s, a, b, r, size=24):
            return tag(s, size, YELLOW_B).move_to([(cc(a, r)[0] + cc(b - 1, r)[0]) / 2,
                                                   yc(r - 1), 0.0])

        def lay(srcs, starts, lefts, target, final_lab):
            box, glab = None, None
            for s, st, lf in zip(srcs, starts, lefts):
                cp = rows[s].copy()
                self.add(cp)
                v = cc(st, target) - cc(0, s)
                anims = [cp.animate.shift(v).set_fill(opacity=0.95)]
                self.play(*anims, run_time=0.85 if s >= srcs[1] else 0.65)
                check(_same_spots(cp, [cc(st + i, target) for i in range(F[s])]),
                      f"{_fname(s)} lands at cell {st} of {_fname(target)}")
                rim = Rectangle(width=F[s] * u, height=u, stroke_color=WHITE,
                                stroke_width=4.5).move_to(
                    (cc(st, target) + cc(st + F[s] - 1, target)) / 2)
                self.add(rim)
                if lf:
                    nb = gap_box(F[target] - lf, F[target], target)
                    gtxt = _fname(s - 1) if lf == F[s - 1] and s > 2 else final_lab
                    nl = gap_lab(gtxt, F[target] - lf, F[target], target)
                    others = [m for rr in range(1, R + 1) for m in rows[rr]]
                    _clear_of(nl, others, 0.03, "gap label")
                    _clear_of(nl, list(labs.values()), 0.05, "gap label")
                    if box is None:
                        box, glab = nb, nl
                        self.play(Create(box), FadeIn(glab), run_time=0.4)
                    else:
                        self.play(Transform(box, nb), Transform(glab, nl), run_time=0.4)
                elif box is not None:
                    self.play(FadeOut(box), FadeOut(glab), run_time=0.3)
                    box = glab = None
            return box, glab

        # ---- the even rows along F(2n+1): one cell is left over
        box9, lab9 = lay(ev_src, ev_start, ev_left, R, "1")
        ring = Square(side_length=u * 1.25, stroke_color=YELLOW_B, stroke_width=4)
        ring.move_to(cc(F[R] - 1, R))
        y_seg = yc(R) - u / 2 - 0.2
        seg_e = [tag(_fname(s), 20, ORANGE).move_to(
            [(cc(st, R)[0] + cc(st + F[s] - 1, R)[0]) / 2, y_seg, 0.0])
            for s, st in zip(ev_src, ev_start)]
        self.play(FadeOut(box9), Create(ring), *[FadeIn(t) for t in seg_e],
                  run_time=0.6)
        x_eq = -1.5
        row_e = _eq_line("F₂ + F₄ + F₆ + F₈", "F₉ − 1", x_eq, -1.6, 30, ORANGE)
        num_e = _line("1 + 3 + 8 + 21  =  34 − 1", 1.25, -1.6, 26, GREY_A)
        self.play(FadeIn(row_e), FadeIn(num_e), run_time=0.7)
        self.hold(0.4)

        # ---- the odd rows along F(2n): F1 fills the last gap, F2
        lay(od_src, od_start, od_left, R - 1, "F₂")
        seg_o = [tag(_fname(s), 20, BLUE_B).move_to(
            [(cc(st, R - 1)[0] + cc(st + F[s] - 1, R - 1)[0]) / 2, yc(R - 2), 0.0])
            for s, st in zip(od_src[1:], od_start[1:])]
        cells_all = [m for rr in range(1, R + 1) for m in rows[rr]]
        for t in seg_e + seg_o:
            _clear_of(t, cells_all, 0.03, "segment label")
        row_o = _eq_line("F₁ + F₃ + F₅ + F₇", "F₈", x_eq, -2.3, 30, BLUE_B)
        num_o = _line("1 + 2 + 5 + 13  =  21", 1.25, -2.3, 26, GREY_A)
        _safe(row_e, row_o, num_e, num_o, ring, lab9, *seg_e, *seg_o)
        _apart(row_e, row_o, num_e, num_o, *seg_e, *seg_o, lab9, gap=0.08)
        self.play(*[FadeIn(t) for t in seg_o], run_time=0.5)
        self.play(FadeIn(row_o), FadeIn(num_o), run_time=0.7)
        cap = caption("F₁ + F₃ + … + F₂ₙ₋₁  =  F₂ₙ        "
                      "F₂ + F₄ + … + F₂ₙ  =  F₂ₙ₊₁ − 1", 32)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B40

class B40_ReciprocalTriangulars(Board):
    """Tₖ = k(k+1)/2, so 1/Tₖ = 2 · 1/k · 1/(k+1): two strips of width
    1/k · 1/(k+1). Cut a unit square at 1/2, 1/3, 1/4, …: the strip
    between the cuts 1/(k+1) and 1/k is exactly that wide — the triangle
    under the line from the top-left corner to the cut 1/k, shrunk by
    1/(k+1) towards its right end, has its top on the diagonal, so it
    stands exactly between the two cuts, and its base is 1/k · 1/(k+1).
    A copy of every strip slides into a second unit square. The cuts run
    down to 0, so the strips fill both squares:
        1/T₁ + 1/T₂ + 1/T₃ + … = 1 + ⅓ + 1/6 + 1/10 + … = 2.
    Six strips built this way, a few more added, then "…"."""

    def construct(self):
        S = 4.2
        O = np.array([-5.98, -2.25, 0.0])

        def P(x, y):
            return O + S * np.array([float(x), float(y), 0.0])

        K, KMORE = 6, 14
        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D, RED_D, PURPLE_B, GOLD_D, MAROON_B]
        tcol = [BLUE_B, TEAL_B, ORANGE, GREEN_B, RED_B, PURPLE_A]
        for k in range(1, KMORE + 1):
            a, s = Fraction(1, k), Fraction(1, k + 1)
            top = (a + s * (0 - a), s * 1)                 # image of the corner (0, 1)
            check(top == (s, s), f"k={k}: the shrunk top lands on the diagonal")
            check(a - s == a * s, f"k={k}: the strip is 1/k · 1/(k+1) wide")
            check(2 * a * s == 1 / Fraction(k * (k + 1), 2), f"k={k}: 2 strips = 1/T_k")
        check(2 * sum(Fraction(1, k * (k + 1)) for k in range(1, KMORE + 1))
              == 2 - Fraction(2, KMORE + 1), "the strips and the rest make 2")

        sqA = Polygon(P(0, 0), P(1, 0), P(1, 1), P(0, 1), stroke_color=WHITE,
                      stroke_width=3)
        sqB = Polygon(P(1, 0), P(2, 0), P(2, 1), P(1, 1), stroke_color=WHITE,
                      stroke_width=3)
        diag = DashedLine(P(0, 0), P(1, 1), color=GREY_A, stroke_width=2.5,
                          dash_length=0.1)
        ylab = O[1] - 0.3
        blabs = {1: tag("1", 24), 2: tag("½", 22), 3: tag("⅓", 21), 4: tag("¼", 21)}
        zero = tag("0", 24).move_to([O[0], ylab, 0])
        two = tag("2", 24).move_to([P(2, 0)[0], ylab, 0])
        for k, t in blabs.items():
            t.move_to([P(Fraction(1, k), 0)[0], ylab, 0])
        _apart(zero, two, *blabs.values(), gap=0.1)
        _safe(sqA, sqB, zero, two, *blabs.values())
        self.play(Create(sqA), Create(sqB), FadeIn(zero), FadeIn(blabs[1]),
                  FadeIn(two), run_time=1.1)
        self.play(Create(diag), run_time=0.5)

        # readout: one row per pair of strips
        x_eq = 4.1
        hdr = _rich([("T", ""), ("k", "_"), ("  =  k(k+1)/2", "")], 28, GREY_A)
        hdr.move_to([4.45, 3.2, 0.0])
        names = ["1", "½", "⅓", "¼", "1/5", "1/6", "1/7"]
        rows = []
        for k in range(1, K + 1):
            lt = "1/T" + _SUBS[k]
            rows.append(_eq_line(lt, f"2 · {names[k - 1]} · {names[k]}", x_eq,
                                 2.45 - (k - 1) * 0.58, 26, tcol[k - 1]))
        _safe(hdr, *rows)
        _apart(hdr, *rows, gap=0.06)
        check(min(r_.get_left()[0] for r_ in rows) > P(2, 0)[0] + 0.3,
              "readout clear of the squares")
        self.play(FadeIn(hdr), run_time=0.5)

        for k in range(1, K + 1):
            a, s = 1.0 / k, 1.0 / (k + 1)
            fast = k >= 3
            tri = mk([P(0, 1), P(0, 0), P(a, 0)], GREY_B, 0.42,
                     stroke_color=WHITE, stroke_width=2.5)
            self.play(FadeIn(tri), run_time=0.3 if fast else 0.45)
            self.play(tri.animate.scale(s, about_point=P(a, 0)),
                      run_time=0.7 if fast else 1.1)
            check(_same_poly(_verts(tri), [P(s, s), P(s, 0), P(a, 0)]),
                  f"k={k}: the shrunk triangle stands between the cuts, top on the diagonal")
            level = DashedLine(P(0, s), P(s, s), color=GREY_A, stroke_width=2,
                               dash_length=0.07)
            self.play(Create(level), run_time=0.2 if fast else 0.35)
            strip = mk([P(*q) for q in _rect(s, 0, a - s, 1)], cols[k - 1], FILL,
                       stroke_width=1.5)
            cut = Line(P(s, 0), P(s, 1), color=WHITE, stroke_width=2)
            self.add(strip)
            self.bring_to_back(strip)
            self.bring_to_front(sqA, diag, tri)
            anims = [FadeIn(strip), Create(cut)]
            if k + 1 in blabs:
                anims.append(FadeIn(blabs[k + 1]))
            self.play(*anims, run_time=0.35 if fast else 0.5)
            twin = strip.copy()
            self.add(twin)
            self.play(twin.animate.shift(RIGHT * S), FadeOut(tri), FadeOut(level),
                      FadeIn(rows[k - 1]), run_time=0.6 if fast else 0.9)
            check(_same_poly(_verts(twin), [P(*q) for q in _rect(1 + s, 0, a - s, 1)]),
                  f"k={k}: the copy lands in the second square")
            self.bring_to_front(sqB)
            if k <= 2:
                self.hold(0.3)

        # more strips the same way, then the rest of each square is "…"
        extra = VGroup()
        for k in range(K + 1, KMORE + 1):
            a, s = 1.0 / k, 1.0 / (k + 1)
            for dx in (0, 1):
                extra.add(mk([P(*q) for q in _rect(dx + s, 0, a - s, 1)],
                             cols[(k - 1) % len(cols)], FILL, stroke_width=0.8))
        vdots = tag("⋮", 30).move_to(rows[-1].get_center() + DOWN * 0.62)
        rest = 1.0 / (KMORE + 1)
        dots = [tag("…", 20).move_to(P(dx + rest / 2, 0.62)) for dx in (0, 1)]
        for d_ in dots:
            check(d_.width < rest * S - 0.04, "the dots fit in what is left")
        _clear_of_segment(dots[0], P(0, 0), P(1, 1), what="the dots")
        self.add(extra)
        self.bring_to_back(extra)
        self.bring_to_front(sqA, sqB, diag)
        extra.set_opacity(0)
        self.play(LaggedStart(*[m.animate.set_fill(opacity=FILL).set_stroke(opacity=1)
                                for m in extra], lag_ratio=0.15),
                  FadeIn(vdots), run_time=1.3)
        self.play(*[FadeIn(d_) for d_ in dots], run_time=0.4)
        self.hold(0.3)

        frame = Polygon(P(0, 0), P(2, 0), P(2, 1), P(0, 1), stroke_color=YELLOW_B,
                        stroke_width=6)
        d_w = _dim(P(0, 1), P(2, 1), "2", UP * 0.17, YELLOW_B, 30, gap=0.1)
        d_h = _dim(P(0, 0), P(0, 1), "1", LEFT * 0.17, YELLOW_B, 30, gap=0.1)
        _safe(frame, d_w, d_h, vdots)
        _apart(d_w[1], d_h[1], hdr, *rows, vdots, gap=0.05)
        self.play(Create(frame), FadeIn(d_w), FadeIn(d_h), run_time=0.9)
        cap = _cap(_rich([("1/T", ""), ("1", "_"), (" + 1/T", ""), ("2", "_"),
                          (" + 1/T", ""), ("3", "_"),
                          (" + …   =   1 + ⅓ + 1/6 + 1/10 + …   =   2", "")],
                         34, YELLOW_B))
        self.play(FadeIn(cap), run_time=1.3)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B31

class B31_TriangularOfProduct(Board):
    """The staircase T_ab (columns of 1, 2, …, ab cells) cut into b × b
    blocks: the blocks themselves stand as a staircase of a columns, T_a
    of them — the a on the diagonal are small staircases T_b, the T_(a−1)
    below them are full squares. Every full square is a staircase T_b and
    an upside-down staircase T_(b−1). So every block holds one T_b, and
    every full block one T_(b−1) besides: pulled apart, the T_b's stand as
    T_a blocks and the T_(b−1)'s as T_(a−1) blocks:
        T_ab = T_a·T_b + T_(a−1)·T_(b−1).     Drawn for a = 4, b = 3
    (78 = 10·6 + 6·3)."""

    def construct(self):
        a, b, u = 4, 3, 0.38
        N = a * b
        O = np.array([-6.3, -2.45, 0.0])

        def C(x, y):
            return O + u * np.array([float(x), float(y), 0.0])

        def tri(m):
            return m * (m + 1) // 2

        T = [(i, j) for i in range(N) for j in range(i + 1)]
        blocks = [(I, J) for I in range(a) for J in range(I + 1)]
        Tb = {(p, q) for p in range(b) for q in range(p + 1)}
        Tb1 = {(p, q) for p in range(b - 1) for q in range(p + 1)}
        blue, orng = {}, {}
        for (I, J) in blocks:
            cs = [(i, j) for (i, j) in T if b * I <= i < b * I + b and b * J <= j < b * J + b]
            blue[(I, J)] = [(i, j) for (i, j) in cs if j - b * J <= i - b * I]
            orng[(I, J)] = [(i, j) for (i, j) in cs if j - b * J > i - b * I]
            check({(i - b * I, j - b * J) for (i, j) in blue[(I, J)]} == Tb,
                  f"block {I},{J}: its lower part is T_b")
            if J < I:
                check(len(cs) == b * b, f"block {I},{J} is a full b × b square")
                check({(b - 2 - (i - b * I), b - 1 - (j - b * J))
                       for (i, j) in orng[(I, J)]} == Tb1,
                      f"block {I},{J}: its upper part, half-turned, is T_(b−1)")
            else:
                check(not orng[(I, J)], f"diagonal block {I}: just T_b")
        check(_tiles_once(list(blue.values()) + [v for v in orng.values() if v], T),
              "the pieces tile T_ab")
        full = [(I, J) for (I, J) in blocks if J < I]
        check(len(blocks) == tri(a) and len(full) == tri(a - 1),
              "T_a blocks, T_(a−1) of them full")
        check(len(T) == tri(N) == tri(a) * tri(b) + tri(a - 1) * tri(b - 1) == 78,
              "T_ab = T_a·T_b + T_(a−1)·T_(b−1)")

        cells = {}
        for (i, j) in T:
            cells[(i, j)] = cell(i, j, u, O, GREY_C, FILL, 1.1)
        cols_ = [VGroup(*[cells[(i, j)] for j in range(i + 1)]) for i in range(N)]
        lab_T = _rich([("T", ""), ("ab", "_")], 40).move_to(C(2.4, 8.6))
        d_ab = _dim(C(0, 0), C(N, 0), "ab", DOWN * 0.18, YELLOW_B, 28, gap=0.08)
        _clear_of(lab_T, list(cells.values()), 0.1, "T_ab")
        _safe(lab_T, d_ab, *cells.values())
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cols_],
                              lag_ratio=0.15), run_time=1.8)
        self.play(FadeIn(lab_T), FadeIn(d_ab), run_time=0.6)
        self.hold(0.3)

        # ---- b × b blocks: a staircase of blocks
        rims = {}
        for (I, J) in blocks:
            if J < I:
                rims[(I, J)] = Polygon(*[C(*q) for q in _rect(b * I, b * J, b, b)],
                                       stroke_color=YELLOW_B, stroke_width=4)
            else:
                rims[(I, J)] = _outline_cells(blue[(I, J)], u, O, YELLOW_B, 4)
        d_b = VGroup(*[_dim(C(b * I, 0), C(b * I + b, 0), "b", DOWN * 0.18, YELLOW_B,
                            26, gap=0.08) for I in range(a)])
        self.play(Create(VGroup(*rims.values())), ReplacementTransform(d_ab, d_b),
                  run_time=1.2)
        self.hold(0.3)
        # ---- every block: a T_b below, and in the full ones a T_(b−1) above
        self.play(*[cells[c].animate.set_fill(BLUE_D) for v in blue.values() for c in v],
                  *[cells[c].animate.set_fill(ORANGE) for v in orng.values() for c in v],
                  run_time=0.8)
        self.hold(0.4)

        # ---- pull the blocks apart, then the upper parts away from the lower
        g = 0.3
        grp = {}
        for (I, J) in blocks:
            grp[(I, J)] = VGroup(*[cells[c] for c in blue[(I, J)] + orng[(I, J)]],
                                 rims[(I, J)])
        self.play(FadeOut(d_b), FadeOut(lab_T),
                  *[grp[k].animate.shift(g * np.array([k[0], k[1], 0.0]))
                    for k in blocks], run_time=1.2)
        for (I, J) in blocks:
            for c in blue[(I, J)] + orng[(I, J)]:
                check(close(cells[c].get_center(),
                            C(c[0] + 0.5, c[1] + 0.5) + g * np.array([I, J, 0.0]), 1e-6),
                      "blocks pulled straight apart")
        up = VGroup(*[cells[c] for k in full for c in orng[k]])
        upr = VGroup(*[_outline_cells(orng[k], u, O, WHITE, 3.5).shift(
            g * np.array([k[0], k[1], 0.0])) for k in full])
        lowr = VGroup(*[_outline_cells(blue[k], u, O, WHITE, 3.5).shift(
            g * np.array([k[0], k[1], 0.0])) for k in blocks])
        self.play(FadeOut(VGroup(*rims.values())), Create(upr), Create(lowr),
                  run_time=0.8)
        D = np.array([6.55, 0.0, 0.0])
        self.bring_to_front(up, upr)            # carried over the blue pieces
        self.play(VGroup(up, upr).animate.shift(D), run_time=1.4)
        for k in full:
            for c in orng[k]:
                check(close(cells[c].get_center(), C(c[0] + 0.5, c[1] + 0.5)
                            + g * np.array([k[0], k[1], 0.0]) + D, 1e-6),
                      "the upper parts moved together")
        # each upper part, half-turned in place, is an upright T_(b−1)
        turns = []
        for idx, k in enumerate(full):
            piece = VGroup(*[cells[c] for c in orng[k]], upr[idx])
            lo = C(b * k[0], b * k[1] + 1) + g * np.array([k[0], k[1], 0.0]) + D
            ctr = lo + u * np.array([(b - 1) / 2, (b - 1) / 2, 0.0])
            turns.append((piece, ctr, lo, k))
        self.play(*[Rotate(pc, angle=PI, about_point=ct) for pc, ct, _, _ in turns],
                  run_time=1.2)
        for pc, ct, lo, k in turns:
            got = {tuple(np.round((cells[c].get_center() - lo) / u - 0.5).astype(int)[:2])
                   for c in orng[k]}
            check(got == Tb1, "a half-turned upper part is the staircase T_(b−1)")

        # ---- name the two arrangements
        lab_b = _rich([("T", "", BLUE_B), ("a", "_", BLUE_B), (" · T", "", BLUE_B),
                       ("b", "_", BLUE_B)], 36)
        lab_b.move_to(C(2.6, 9.4))
        lab_o = _rich([("T", "", ORANGE), ("a−1", "_", ORANGE), (" · T", "", ORANGE),
                       ("b−1", "_", ORANGE)], 36)
        lab_o.move_to(C(5.0, 8.4) + D)
        nums = _line("a = 4,  b = 3:    78  =  10 · 6  +  6 · 3", 0.0, 3.25, 24, GREY_A)
        nums.set_x(3.3)
        allc = list(cells.values())
        for t in (lab_b, lab_o, nums):
            _clear_of(t, allc + list(upr) + list(lowr), 0.1, "arrangement label")
        _safe(lab_b, lab_o, nums, *allc)
        _apart(lab_b, lab_o, nums, gap=0.1)
        self.play(FadeIn(lab_b), FadeIn(lab_o), run_time=0.8)
        self.play(FadeIn(nums), run_time=0.6)
        cap = _cap(_rich([("T", ""), ("ab", "_"), ("  =  T", ""), ("a", "_"),
                          (" · T", ""), ("b", "_"), ("  +  T", ""), ("a−1", "_"),
                          (" · T", ""), ("b−1", "_")], 38, YELLOW_B))
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B43

class B43_CentredSquare(Board):
    """The centred square number Cₙ is a diamond of dots with rows of 1, 3,
    …, 2n−1, …, 3, 1. Colour the dots like a chessboard: the dots of one
    colour sit on a square grid turned 45° with n dots a side, those of the
    other colour on the same kind of grid, nested inside it, with n − 1 a
    side. Pulled apart and turned back 45°, they are the squares n² and
    (n−1)²:
        Cₙ = 1 + 3 + … + (2n−1) + … + 3 + 1 = n² + (n−1)².    Drawn for n = 5."""

    def construct(self):
        n, s, rad = 5, 0.56, 0.12
        m = n - 1
        Cd = np.array([-3.15, 0.3, 0.0])
        D = np.array([6.0, 0.0, 0.0])

        def P(x, y):
            return Cd + s * np.array([float(x), float(y), 0.0])

        pts = [(x, y) for y in range(m, -m - 1, -1)
               for x in range(-(m - abs(y)), m - abs(y) + 1)]
        blue = [p for p in pts if (p[0] + p[1] + m) % 2 == 0]
        orng = [p for p in pts if (p[0] + p[1] + m) % 2 == 1]
        check(len(pts) == n * n + (n - 1) ** 2 and len(blue) == n * n
              and len(orng) == (n - 1) ** 2, "41 = 25 + 16")
        check([sum(1 for p in pts if p[1] == y) for y in range(m, -m - 1, -1)]
              == list(range(1, 2 * n, 2)) + list(range(2 * n - 3, 0, -2)),
              "rows 1, 3, …, 2n−1, …, 3, 1")
        check({(x + y, x - y) for x, y in blue}
              == {(U, V) for U in range(-m, m + 1, 2) for V in range(-m, m + 1, 2)},
              "one colour: an n × n grid turned 45°")
        check({(x + y, x - y) for x, y in orng}
              == {(U, V) for U in range(-m + 1, m, 2) for V in range(-m + 1, m, 2)},
              "the other: an (n−1) × (n−1) grid turned 45°, inside it")

        dots = {p: Dot(P(*p), radius=rad, color=GREY_A) for p in pts}
        counts = VGroup()
        for y in range(m, -m - 1, -1):
            k = m - abs(y)
            t = tag(str(2 * k + 1), 24, GREY_A)
            t.move_to(P(-k, y) + LEFT * (0.3 + t.width / 2))
            counts.add(t)
        _safe(counts, *dots.values())
        for y in range(m, -m - 1, -1):
            row = [dots[(x, y)] for x in range(-(m - abs(y)), m - abs(y) + 1)]
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in row], lag_ratio=0.1),
                      FadeIn(counts[m - y]), run_time=0.32)
        self.hold(0.4)

        def grid_lines(lo, hi, color):
            g = VGroup()
            for V in range(lo, hi + 1, 2):
                g.add(Line(P((lo + V) / 2, (lo - V) / 2), P((hi + V) / 2, (hi - V) / 2),
                           color=color, stroke_width=2.5))
            for U in range(lo, hi + 1, 2):
                g.add(Line(P((U + lo) / 2, (U - lo) / 2), P((U + hi) / 2, (U - hi) / 2),
                           color=color, stroke_width=2.5))
            return g

        gb = grid_lines(-m, m, BLUE_D)
        go = grid_lines(-m + 1, m - 1, ORANGE)
        bdots = VGroup(*[dots[p] for p in blue])
        odots = VGroup(*[dots[p] for p in orng])
        self.play(bdots.animate.set_color(BLUE_C), odots.animate.set_color(ORANGE),
                  run_time=0.7)
        self.add(gb, go)
        self.bring_to_front(bdots, odots)
        self.play(Create(gb), Create(go), run_time=1.0)
        self.hold(0.5)

        # pull the inner grid out, then turn both back by 45°
        OB = VGroup(go, odots)
        self.play(FadeOut(counts), OB.animate.shift(D), run_time=1.3)
        self.play(Rotate(VGroup(gb, bdots), angle=-PI / 4, about_point=Cd),
                  Rotate(OB, angle=-PI / 4, about_point=Cd + D), run_time=1.5)
        q = s / np.sqrt(2)          # half the spacing of the turned grids

        def turned(p):              # (x, y) after the turn, in screen units
            x, y = p
            return np.array([(x + y) * q, (y - x) * q, 0.0])

        check(_same_spots(bdots, [Cd + turned(p) for p in blue]),
              "the blue dots, turned, stand on an axis-parallel n × n grid")
        check(_same_spots(odots, [Cd + D + turned(p) for p in orng]),
              "the orange dots, turned, stand on an axis-parallel (n−1) × (n−1) grid")
        xs_b = sorted({round(float(d.get_center()[0]), 6) for d in bdots})
        xs_o = sorted({round(float(d.get_center()[0]), 6) for d in odots})
        check(len(xs_b) == n and len(xs_o) == n - 1, "n and n − 1 columns of dots")

        pad = 0.36
        fb = Square(side_length=m * 2 * q + 2 * pad, stroke_color=BLUE_B, stroke_width=4)
        fb.move_to(Cd)
        fo = Square(side_length=(m - 1) * 2 * q + 2 * pad, stroke_color=ORANGE,
                    stroke_width=4).move_to(Cd + D)
        d_n = _dim(fb.get_corner(DL), fb.get_corner(DR), "n", DOWN * 0.2, BLUE_B, 30,
                   gap=0.08)
        d_m = _dim(fo.get_corner(DL), fo.get_corner(DR), "n−1", DOWN * 0.2, ORANGE, 30,
                   gap=0.08)
        sq_b = tag("n²", 36, BLUE_B).next_to(fb, UP, buff=0.2)
        sq_o = tag("(n−1)²", 36, ORANGE).next_to(fo, UP, buff=0.2)
        plus = tag("+", 44).move_to([(fb.get_right()[0] + fo.get_left()[0]) / 2,
                                     Cd[1], 0.0])
        nums = _line("1 + 3 + 5 + 7 + 9 + 7 + 5 + 3 + 1  =  41  =  25 + 16", 0, 3.3,
                     26, GREY_A)
        nums.set_x(0.0)
        _safe(fb, fo, d_n, d_m, sq_b, sq_o, plus, nums)
        _apart(d_n[1], d_m[1], sq_b, sq_o, plus, nums, gap=0.08)
        self.play(Create(fb), Create(fo), FadeIn(d_n), FadeIn(d_m), run_time=1.0)
        self.play(FadeIn(sq_b), FadeIn(sq_o), FadeIn(plus), run_time=0.7)
        self.play(FadeIn(nums), run_time=0.6)
        cap = caption("Cₙ  =  1 + 3 + … + (2n−1) + … + 3 + 1  =  n² + (n−1)²", 34)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B45

class B45_ConsecutiveSums(Board):
    """A run of k consecutive numbers m, m+1, …, m+k−1 (m ≥ 1, k ≥ 2) is a
    trapezoid of dots, one row per number. If k is odd, the rows pair off
    around the middle row, the longer as far over it as the shorter falls
    short: move the overhangs across and every row is the middle one,
    (2m+k−1)/2 long — k rows. If k is even, put each row end to end with
    its partner from the other end: k/2 rows of 2m+k−1. Either way
        m + (m+1) + … + (m+k−1) = k·(2m+k−1)/2,
    a rectangle of dots with a side that is odd and at least 3 — k or
    2m+k−1, whose sum 2(m+k) − 1 is odd. A power of 2 has no odd factor
    above 1, so it is never such a sum. Drawn for 2+3+4+5+6 = 5·4 and
    3+4+5+6 = 2·9."""

    def construct(self):
        s, rad = 0.46, 0.12

        def build(m, k, X0, Y0):
            pos = {(i, j): np.array([X0 + j * s, Y0 - i * s, 0.0])
                   for i in range(k) for j in range(m + i)}
            dots = {key: Dot(p, radius=rad, color=BLUE_C) for key, p in pos.items()}
            return pos, dots

        # ---- k odd: 2 + 3 + 4 + 5 + 6, levelled to 5 rows of 4
        m1, k1 = 2, 5
        XL, YT = -5.1, 2.75
        posL, dL = build(m1, k1, XL, YT)
        avg = m1 + (k1 - 1) // 2
        movesL = []                      # (dot key, target row, target column)
        for i in range((k1 - 1) // 2):
            long_i = k1 - 1 - i
            over = list(range(avg, m1 + long_i))
            check(len(over) == avg - (m1 + i), f"row {long_i}: overhang = shortfall")
            for t, j in enumerate(over):
                movesL.append(((long_i, j), i, m1 + i + t))
        endL = {key: posL[key] for key in posL}
        for key, ti, tj in movesL:
            endL[key] = np.array([XL + tj * s, YT - ti * s, 0.0])
        rectL = {(i, j) for i in range(k1) for j in range(avg)}
        check({(round((YT - p[1]) / s), round((p[0] - XL) / s)) for p in endL.values()}
              == rectL, "levelled: k rows of (2m+k−1)/2")
        check(sum(range(m1, m1 + k1)) == k1 * avg == 20, "2+3+4+5+6 = 5·4")

        # ---- k even: 3 + 4 + 5 + 6, paired into 2 rows of 9
        m2, k2 = 3, 4
        XR = 0.55
        posR, dR = build(m2, k2, XR, YT)
        L2 = 2 * m2 + k2 - 1
        movesR = []
        for i in range(k2 // 2):
            partner = k2 - 1 - i
            for j in range(m2 + i):
                movesR.append(((i, j), partner, m2 + partner + j))
        endR = {key: posR[key] for key in posR}
        for key, ti, tj in movesR:
            endR[key] = np.array([XR + tj * s, YT - ti * s, 0.0])
        rectR = {(i, j) for i in range(k2 // 2, k2) for j in range(L2)}
        check({(round((YT - p[1]) / s), round((p[0] - XR) / s)) for p in endR.values()}
              == rectR, "paired: k/2 rows of 2m+k−1")
        check(sum(range(m2, m2 + k2)) == (k2 // 2) * L2 == 18, "3+4+5+6 = 2·9")
        check(all((k * (2 * m + k - 1)) % 2 == 0 and
                  (k % 2 == 1 or (2 * m + k - 1) % 2 == 1)
                  for m in range(1, 30) for k in range(2, 30)),
              "one of k, 2m+k−1 is odd, and k(2m+k−1) is even")
        check(all((k if k % 2 else 2 * m + k - 1) >= 3
                  for m in range(1, 30) for k in range(2, 30)),
              "the odd one is at least 3")
        powers = {2 ** e for e in range(12)}
        check(not any(sum(range(m, m + k)) in powers
                      for m in range(1, 200) for k in range(2, 60)),
              "no power of 2 is such a sum (checked up to 2^11)")

        cntL = VGroup(*[tag(str(m1 + i), 24, GREY_A).move_to(posL[(i, 0)] + LEFT * 0.42)
                        for i in range(k1)])
        cntR = VGroup(*[tag(str(m2 + i), 24, GREY_A).move_to(posR[(i, 0)] + LEFT * 0.42)
                        for i in range(k2)])
        _safe(cntL, cntR, *dL.values(), *dR.values())
        for i in range(k1):
            self.play(LaggedStart(*[FadeIn(dL[(i, j)], scale=0.5) for j in range(m1 + i)],
                                  lag_ratio=0.08), FadeIn(cntL[i]), run_time=0.35)
        for i in range(k2):
            self.play(LaggedStart(*[FadeIn(dR[(i, j)], scale=0.5) for j in range(m2 + i)],
                                  lag_ratio=0.08), FadeIn(cntR[i]), run_time=0.35)
        self.hold(0.4)

        # ---- level the odd run
        mvL = VGroup(*[dL[key] for key, _, _ in movesL])
        x_mid = XL + (avg - 0.5) * s
        mid = DashedLine([x_mid, YT + 0.3, 0.0], [x_mid, YT - (k1 - 1) * s - 0.3, 0.0],
                         color=YELLOW_B, stroke_width=3, dash_length=0.08)
        self.play(Create(mid), mvL.animate.set_color(YELLOW_C), run_time=0.6)
        self.play(*[dL[key].animate.move_to(endL[key]) for key, _, _ in movesL],
                  run_time=1.3)
        check(_same_spots(dL.values(), endL.values()), "the overhangs fill the shortfalls")
        self.play(FadeOut(mid), mvL.animate.set_color(BLUE_C), FadeOut(cntL), run_time=0.5)

        # ---- pair the even run
        mvR = VGroup(*[dR[key] for key, _, _ in movesR])
        self.play(mvR.animate.set_color(YELLOW_C), run_time=0.4)
        self.play(*[dR[key].animate.move_to(endR[key]) for key, _, _ in movesR],
                  run_time=1.4)
        check(_same_spots(dR.values(), endR.values()), "each row joins its partner")
        self.play(mvR.animate.set_color(BLUE_C), FadeOut(cntR), run_time=0.5)

        # ---- the sides; the odd one in yellow
        ptsL = [endL[key] for key in endL]
        ptsR = [endR[key] for key in endR]

        def ext(pts):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return min(xs) - rad, max(xs) + rad, min(ys) - rad, max(ys) + rad

        x0L, x1L, y0L, y1L = ext(ptsL)
        x0R, x1R, y0R, y1R = ext(ptsR)
        dkL = _dim([x0L, y0L, 0], [x0L, y1L, 0], "k = 5", LEFT * 0.2, YELLOW_B, 26,
                   gap=0.1)
        dwL = _dim([x0L, y0L, 0], [x1L, y0L, 0], "(2m+k−1)/2 = 4", DOWN * 0.2, GREY_A, 24,
                   gap=0.17)   # clear of the end ticks (0.09), which lie over the label
        dkR = _dim([x0R, y0R, 0], [x0R, y1R, 0], "k/2 = 2", LEFT * 0.2, GREY_A, 24,
                   gap=0.1)
        dwR = _dim([x0R, y1R, 0], [x1R, y1R, 0], "2m+k−1 = 9", UP * 0.2, YELLOW_B, 26,
                   gap=0.08)
        eqL = _line("2 + 3 + 4 + 5 + 6  =  5 · 4", 0, -0.62, 28, WHITE)
        eqL.set_x(-3.85)
        eqR = _line("3 + 4 + 5 + 6  =  2 · 9", 0, -0.62, 28, WHITE)
        eqR.set_x(2.45)
        gen = _eq_line("m + (m+1) + … + (m+k−1)", "k · (2m+k−1)/2", 0.55, -1.45, 30,
                       YELLOW_B)
        par = _eq_line("k  +  (2m+k−1)", "2(m+k) − 1", 0.55, -2.2, 28, GREY_A)
        items = [dkL, dwL, dkR, dwR, eqL, eqR, gen, par]
        _safe(*items)
        _apart(dkL[1], dwL[1], dkR[1], dwR[1], eqL, eqR, gen, par, gap=0.08)
        for t in (dkL[1], dwL[1], dkR[1], dwR[1], eqL, eqR):
            _clear_of(t, list(dL.values()) + list(dR.values()), 0.06, "side label")
        self.play(FadeIn(dkL), FadeIn(dwL), FadeIn(dkR), FadeIn(dwR), run_time=0.9)
        self.play(FadeIn(eqL), FadeIn(eqR), run_time=0.7)
        self.hold(0.4)
        self.play(FadeIn(gen), run_time=0.8)
        self.play(FadeIn(par), run_time=0.8)
        cap = caption("m + (m+1) + … + (m+k−1)  =  k(2m+k−1)/2   ≠   2ⁿ"
                      "      (m ≥ 1, k ≥ 2)", 30)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)
