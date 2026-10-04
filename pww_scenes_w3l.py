# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3l.py — proofs without words, 2D (manim):
#     K30 conjugate partitions              K21 the meeting problem
#     K28 a random triangle and the centre  K32 distinct parts and odd parts
#     K7  Fibonacci addition by tilings     K24 Bertrand's paradox (caution)
#     K25 committees with a chair           K26 alternating sum of a row
#     K31 self-conjugate partitions
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# letter exponents or subscripts are set with _supline / _subline.
# Counting scenes enumerate a small case completely and check the counts and
# the bijections with check(...); the geometric scenes check areas, landings
# and regions, so a wrong construction fails the render.  A reflection is
# shown as what it is in space: a half-turn about the mirror line.

from itertools import combinations, product
from math import comb


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


def _supline(parts, size=30, color=YELLOW_B, sup_scale=0.62, rise=0.48):
    """A line of text with true superscripts, without TeX.  parts: list of
    (string, is_superscript) or (string, is_superscript, colour)."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for part in parts:
        s, sup = part[0], part[1]
        col = part[2] if len(part) > 2 else color
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        x += lead * sp
        t = _on_base(s.strip(" "), size * (sup_scale if sup else 1.0), col)
        t.shift(np.array([x - t.get_left()[0], rise * xh if sup else 0.0, 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sup else 0.02)
        g.add(t)
    return g


def _subline(parts, size=26, color=WHITE, sub_scale=0.64, drop=0.32):
    """A line of text with true subscripts, without relying on subscript
    letter glyphs (absent from some fonts).  parts: list of (string,
    is_subscript) or (string, is_subscript, colour)."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for part in parts:
        s, sub = part[0], part[1]
        col = part[2] if len(part) > 2 else color
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        x += lead * sp
        t = _on_base(s.strip(" "), size * (sub_scale if sub else 1.0), col)
        t.shift(np.array([x - t.get_left()[0], -drop * xh if sub else 0.0, 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sub else 0.02)
        g.add(t)
    return g


def _crow(parts, size=30, gap=0.16):
    """Coloured pieces [(text, colour), ...] side by side on one baseline."""
    g = VGroup()
    x = 0.0
    for s, col in parts:
        t = _on_base(s, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + gap
        g.add(t)
    return g


def _cap(group):
    """Place a composite formula where caption() puts its Text."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.set_x(0.0)
    group.to_edge(DOWN, buff=0.3)
    return group


# ------------------------------------------------------------ geometry helpers

def _dir(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _pip(p, poly):
    """Point strictly inside polygon (ray casting)."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def _tiles_exactly(polys, region, n=110):
    """True if the polygons tile `region` with no gap and no overlap: on a
    jittered grid over the region's bounding box every point inside the
    region lies in exactly one polygon and every point outside in none."""
    xs = [float(p[0]) for p in region]
    ys = [float(p[1]) for p in region]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for i in range(n):
        for j in range(n):
            x = x0 + (x1 - x0) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = y0 + (y1 - y0) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            k = sum(_pip((x, y), P) for P in polys)
            if k != (1 if _pip((x, y), region) else 0):
                return False
    return True


def _same_poly(P, Q, tol=1e-9):
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


def _boxes_of(m):
    """Sorted bounding boxes of every polygon (cell, tile) in a mobject."""
    return sorted(tuple(np.round(np.concatenate([q.get_corner(DL)[:2],
                                                 q.get_corner(UR)[:2]]), 6))
                  for q in m.get_family() if isinstance(q, Polygon))


# ------------------------------------------------------------ label hygiene

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


def _hit(m, segs, pad=0.05):
    """Index of the first screen segment passing through m's box (grown by
    pad), or None."""
    x0, x1 = m.get_left()[0] - pad, m.get_right()[0] + pad
    y0, y1 = m.get_bottom()[1] - pad, m.get_top()[1] + pad
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.02) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.05, gap=0.04):
    """Every label inside the safe area, clear of every segment in segs
    and of every other label."""
    for m in labels:
        check(_inside(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{_name(m)}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(_apart(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _clear_of(labels, mobs, what, gap=0.05):
    """Every label's box clear of every mobject's box."""
    for t in labels:
        for m in mobs:
            check(_apart(t, m, gap), f"{what}: '{_name(t)}' clear of {_name(m)}")


def _dot_segs(dots, grow=0.0):
    """The bounding squares of dots, as segments (so label checks see them)."""
    out = []
    for d in dots:
        c, r = d.get_center(), d.width / 2 + grow
        q = [c + np.array([sx * r, sy * r, 0.0])
             for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        out += [(q[i], q[(i + 1) % 4]) for i in range(4)]
    return out


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption inside the caption band, and no two visible text labels
    overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if not m.has_points() and not m.submobjects:
            continue
        check(_inside(m), f"{type(m).__name__} inside the safe area")
    texts = [t for m in shown for t in m.get_family()
             if isinstance(t, Text) and t.get_fill_opacity() > 0.01]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_apart(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


# ------------------------------------------------------------ partitions

def _partitions(n, maxp=None):
    """All partitions of n (parts in decreasing order), largest first."""
    if maxp is None:
        maxp = n
    if n == 0:
        return [()]
    out = []
    for p in range(min(n, maxp), 0, -1):
        for rest in _partitions(n - p, p):
            out.append((p,) + rest)
    return out


def _conj(lam):
    """The conjugate partition: its parts are the column lengths of lam."""
    return tuple(sum(1 for x in lam if x >= j) for j in range(1, (lam[0] if lam else 0) + 1))


def _cells(lam):
    """The cells (row, column) of the dot diagram of lam, row by row."""
    return [(i, j) for i, r in enumerate(lam) for j in range(r)]


def _centres(g):
    return sorted((round(float(d.get_center()[0]), 5), round(float(d.get_center()[1]), 5))
                  for d in g)


# ===================================================================== K30

class K30_ConjugatePartitions(Board):
    """Draw a partition as a dot diagram, one row per part, longest on top.
    Turning the diagram over its diagonal (a half-turn in space about that
    line) makes every row a column: the new rows are the old columns, a
    partition of the same n. Twice turned is the original, so this pairs
    the partitions of n off one to one. A diagram with at most k rows lies
    above a fence under row k; turned over, the fence stands right of
    column k: at most k parts become parts of size at most k. Shown for
    5 + 3 + 3 + 1 and then all seven partitions of 6 into ≤ 3 parts."""

    def construct(self):
        # ---------------- the mathematics, checked
        for n in range(1, 13):
            P = _partitions(n)
            check(all(_conj(_conj(l)) == l for l in P), "turning twice gives the partition back")
            check(all(sum(_conj(l)) == n for l in P), "the conjugate is a partition of n")
            for k in range(1, n + 1):
                A = [l for l in P if len(l) <= k]
                B = [l for l in P if l[0] <= k]
                check(sorted(_conj(l) for l in A) == sorted(B),
                      f"n = {n}: at most {k} parts <-> parts at most {k}")
        lam = (5, 3, 3, 1)
        lamc = _conj(lam)
        check(lamc == (4, 3, 3, 1, 1) and sum(lamc) == 12, "5+3+3+1 turns into 4+3+3+1+1")
        check(sorted(_cells(lamc)) == sorted((j, i) for i, j in _cells(lam)),
              "the conjugate's dots are the mirror images of the dots")
        six = [l for l in _partitions(6) if len(l) <= 3]
        six_c = [_conj(l) for l in six]
        check(len(six) == 7 and sorted(six_c) == sorted(l for l in _partitions(6) if l[0] <= 3),
              "the 7 partitions of 6 into at most 3 parts <-> the 7 with parts at most 3")

        # ---------------- one diagram turned over its diagonal
        s, rd = 0.56, 0.135
        P0 = np.array([-5.75, 3.3, 0.0])

        def at(i, j):
            return P0 + np.array([j * s, -i * s, 0.0])

        rowcol = [BLUE_B, TEAL_C, ORANGE, YELLOW_C]
        dots = VGroup(*[Dot(at(i, j), radius=rd, color=rowcol[i]) for i, j in _cells(lam)])
        rows = [VGroup(*[d for d, (i, j) in zip(dots, _cells(lam)) if i == r])
                for r in range(len(lam))]
        xR = P0[0] + 4 * s + 0.62               # right margin: row lengths
        yB = P0[1] - 4 * s - 0.55               # bottom margin: column lengths
        rl = [tag(str(v), 28, rowcol[i]).move_to([xR, at(i, 0)[1], 0]) for i, v in enumerate(lam)]
        cl = [tag(str(v), 28, GREY_A).move_to([at(0, j)[0], yB, 0]) for j, v in enumerate(lamc)]
        sumA = _crow([("5", rowcol[0]), ("+", WHITE), ("3", rowcol[1]), ("+", WHITE),
                      ("3", rowcol[2]), ("+", WHITE), ("1", rowcol[3]), ("=  12", WHITE)], 32)
        sumB = _crow([("4", GREY_A), ("+", GREY_A), ("3", GREY_A), ("+", GREY_A), ("3", GREY_A),
                      ("+", GREY_A), ("1", GREY_A), ("+", GREY_A), ("1", GREY_A),
                      ("=  12", GREY_A)], 32)
        sumA.shift(np.array([-1.2 - sumA.get_left()[0], 2.75, 0.0]))
        sumB.shift(np.array([-1.2 - sumB.get_left()[0], 1.45, 0.0]))
        swap = tag("rows  ↔  columns", 26, YELLOW_B)
        swap.move_to([-1.2 + swap.width / 2, 2.1, 0])

        for r in range(len(lam)):
            self.play(FadeIn(rows[r], lag_ratio=0.15), FadeIn(rl[r]), run_time=0.45)
        self.play(FadeIn(sumA), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in cl], lag_ratio=0.2),
                  run_time=0.9)
        diag = guide(at(-0.6, -0.6), at(4.25, 4.25), color=GREY_B)
        self.play(Create(diag), run_time=0.6)
        self.hold(0.3)

        # the half-turn about the diagonal; the margin numbers trade places
        moves = [rl[i].animate.move_to([at(0, i)[0], yB, 0]) for i in range(len(lam))]
        moves += [cl[j].animate.move_to([xR, at(j, 0)[1], 0]) for j in range(len(lamc))]
        self.play(_flip(dots, P0, np.array([1.0, -1.0, 0.0])), *moves, run_time=2.0)
        check(_centres(dots) == sorted((round(float(at(i, j)[0]), 5), round(float(at(i, j)[1]), 5))
                                       for i, j in _cells(lamc)),
              "the turned dots are the diagram of 4+3+3+1+1")
        self.play(FadeIn(swap), FadeIn(sumB), run_time=0.8)
        figA = VGroup(dots, diag, *rl, *cl)
        _labels_ok(rl + cl + [sumA, sumB, swap], [(diag.get_start(), diag.get_end())]
                   + _dot_segs(dots, 0.02), "K30 diagram")
        check(sumA.get_left()[0] > xR + 0.5, "sums clear of the diagram")
        self.hold(0.8)

        # ---------------- all partitions of 6 into at most 3 parts
        s2, r2, pitch = 0.22, 0.07, 1.62
        X0, yT, yD = -4.4, -0.05, -1.2

        def P2(slot, i, j, top):
            return np.array([X0 + slot * pitch + j * s2, top - i * s2, 0.0])

        groups, fences = [], []
        for q, l in enumerate(six):
            g = VGroup(*[Dot(P2(q, i, j, yT), radius=r2, color=BLUE_B) for i, j in _cells(l)])
            f = DashedLine(P2(q, 2.5, -0.5, yT), P2(q, 2.5, 5.5, yT), color=YELLOW_B,
                           stroke_width=3, dash_length=0.06)
            groups.append(g)
            fences.append(f)
        lab_top = tag("≤ 3 parts", 26, YELLOW_B).move_to([-5.65, yT - 2.5 * s2 + 0.3, 0])
        lab_bot = tag("parts ≤ 3", 26, YELLOW_B).move_to([-5.65, yD - 2.5 * s2, 0])
        self.play(LaggedStart(*[FadeIn(VGroup(g, f)) for g, f in zip(groups, fences)],
                              lag_ratio=0.12), FadeIn(lab_top), run_time=1.6)
        turned = [VGroup(g.copy(), f.copy()) for g, f in zip(groups, fences)]
        self.play(LaggedStart(*[_flip(t, P2(q, 0, 0, yT), np.array([1.0, -1.0, 0.0]),
                                      target=P2(q, 0, 0, yD))
                                for q, t in enumerate(turned)], lag_ratio=0.1),
                  run_time=2.6)
        for q, t in enumerate(turned):
            check(_centres(t[0]) == sorted((round(float(P2(q, i, j, yD)[0]), 5),
                                            round(float(P2(q, i, j, yD)[1]), 5))
                                           for i, j in _cells(six_c[q])),
                  f"{six[q]} turns into {six_c[q]}")
            a, b = t[1].get_start(), t[1].get_end()
            check(abs(a[0] - b[0]) < 1e-6 and abs(a[0] - P2(q, 0, 2.5, yD)[0]) < 1e-6,
                  "the fence under row 3 stands right of column 3")
            check(max(d.get_center()[0] for d in t[0]) < a[0], "every part at most 3")
        self.play(FadeIn(lab_bot), run_time=0.6)

        gal = VGroup(*groups, *fences, *turned)
        check(_inside(gal) and _inside(figA), "figures inside the safe area")
        check(gal.get_top()[1] < min(c.get_bottom()[1] for c in cl + rl) - 0.25,
              "gallery clear of the diagram above")
        _labels_ok([lab_top, lab_bot], [], "K30 gallery labels")
        _clear_of([lab_top, lab_bot], [gal], "K30 gallery labels")
        cap = caption("partitions of n into ≤ k parts   =   partitions of n into parts ≤ k", 30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K21

class K21_MeetingProblem(Board):
    """Two people arrive at independent uniformly random times x, y in the
    same hour and each waits a quarter of an hour: the pair (x, y) is a
    uniform point of the unit square, and they meet when |x − y| ≤ ¼, the
    band along the diagonal. What is left are two right isosceles corner
    triangles with legs ¾; slid together (two translations) they make a
    ¾ × ¾ square. So P(meet) = 1 − (¾)² = 7/16. (Waiting time w in
    general: 1 − (1 − w)².)"""

    def construct(self):
        w, d = 0.25, 1.1
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        band = [(0, 0), (w, 0), (1, 1 - w), (1, 1), (1 - w, 1), (0, w)]
        lo = [(w, 0), (1, 0), (1, 1 - w)]               # x − y > ¼
        up = [(0, w), (0, 1), (1 - w, 1)]               # y − x > ¼
        lo2 = [(x + d, y) for x, y in lo]                # slid right
        up2 = [(x + w + d, y - w) for x, y in up]        # slid right and down
        sq2 = [(w + d, 0), (1 + d, 0), (1 + d, 1 - w), (w + d, 1 - w)]
        check(_tiles_exactly([band, lo, up], sq), "band + two corner triangles = unit square")
        check(_tiles_exactly([lo2, up2], sq2), "the two triangles make a ¾ × ¾ square")
        check(close(abs(area(band)), 7 / 16) and close(abs(area(lo)), abs(area(up)))
              and close(abs(area(lo)) + abs(area(up)), (1 - w) ** 2), "areas 7/16 and (¾)²")
        for i in range(60):
            for j in range(60):
                x, y = (i + 0.37) / 60, (j + 0.61) / 60
                if abs(abs(x - y) - w) > 1e-3:
                    check(_pip((x, y), band) == (abs(x - y) <= w),
                          "the band is exactly |x − y| ≤ ¼")
        check(_pip((0.8 - 1e-9, 0.2), lo) and _pip((0.2, 0.8), up), "corners: x − y > ¼, y − x > ¼")

        k, O = 5.2, np.array([-5.9, -2.4, 0.0])

        def M(p):
            return O + k * np.array([p[0], p[1], 0.0])

        CM, CX = GREEN_D, RED_D
        # ---------------- the square of arrival times
        axx = Arrow(M((0, 0)), M((1.1, 0)), buff=0, color=GREY_A, stroke_width=3,
                    tip_length=0.18, max_tip_length_to_length_ratio=0.5)
        axy = Arrow(M((0, 0)), M((0, 1.1)), buff=0, color=GREY_A, stroke_width=3,
                    tip_length=0.18, max_tip_length_to_length_ratio=0.5)
        frame = Polygon(*[M(p) for p in sq], stroke_color=WHITE, stroke_width=3)
        lx = tag("x", 30, GREY_A).next_to(axx.get_end(), RIGHT, buff=0.12)
        ly = tag("y", 30, GREY_A).next_to(axy.get_end(), RIGHT, buff=0.14)
        ticks = VGroup(*[Line(M((t, 0)) + DOWN * 0.08, M((t, 0)) + UP * 0.08, color=GREY_A,
                              stroke_width=2) for t in (w, 0.5, 1 - w)],
                       *[Line(M((0, t)) + LEFT * 0.08, M((0, t)) + RIGHT * 0.08, color=GREY_A,
                              stroke_width=2) for t in (w, 0.5, 1 - w)])
        tl = [tag("0", 26, GREY_A).move_to(M((0, 0)) + np.array([-0.24, -0.3, 0])),
              tag("¼", 28, GREY_A).move_to(M((w, 0)) + DOWN * 0.34),
              tag("1", 26, GREY_A).move_to(M((1, 0)) + DOWN * 0.32),
              tag("¼", 28, GREY_A).move_to(M((0, w)) + LEFT * 0.3),
              tag("1", 26, GREY_A).move_to(M((0, 1)) + LEFT * 0.3)]
        self.play(Create(axx), Create(axy), Create(frame), FadeIn(lx), FadeIn(ly),
                  FadeIn(ticks), *[FadeIn(t) for t in tl], run_time=1.3)

        # two sample pairs of arrival times
        samples = []
        for (x, y), col in (((0.56, 0.4), CM), ((0.82, 0.2), CX)):
            g = VGroup(DashedLine(M((x, y)), M((x, 0)), color=col, stroke_width=2.5,
                                  dash_length=0.08),
                       DashedLine(M((x, y)), M((0, y)), color=col, stroke_width=2.5,
                                  dash_length=0.08),
                       Dot(M((x, y)), radius=0.1, color=col))
            samples.append(g)
        check(abs(0.56 - 0.4) <= w < abs(0.82 - 0.2), "one pair meets, one does not")
        self.play(FadeIn(samples[0]), run_time=0.7)
        self.play(FadeIn(samples[1]), run_time=0.7)
        self.hold(0.3)

        # ---------------- they meet in the band |x − y| ≤ ¼
        bnd = mk([M(p) for p in band], CM, FILL, stroke_width=0)
        e1 = Line(M((0, w)), M((1 - w, 1)), color=WHITE, stroke_width=3)
        e2 = Line(M((w, 0)), M((1, 1 - w)), color=WHITE, stroke_width=3)
        lab_b = tag("|x − y| ≤ ¼", 26).move_to(M((0.5, 0.5)))
        self.play(Create(e1), Create(e2), run_time=0.8)
        self.add(bnd)
        self.bring_to_back(bnd)
        bnd.set_fill(opacity=0)
        self.play(bnd.animate.set_fill(opacity=FILL), FadeIn(lab_b), run_time=1.0)
        self.hold(0.4)
        check(all(_apart(lab_b, m, 0.05) for m in samples[0]), "the sample clear of the label")
        poly_b = [M(p) for p in band]
        for c in (lab_b.get_corner(UL), lab_b.get_corner(UR), lab_b.get_corner(DL),
                  lab_b.get_corner(DR)):
            check(_pip(c, poly_b), "the band's label lies inside the band")
        check(_hit(lab_b, [(e1.get_start(), e1.get_end()), (e2.get_start(), e2.get_end())],
                   0.06) is None, "the band's label clear of its edges")

        # ---------------- the two corner triangles, legs ¾
        tlo = mk([M(p) for p in lo], CX, FILL, stroke_width=2)
        tup = mk([M(p) for p in up], CX, FILL, stroke_width=2)
        leg_up = tag("¾", 30, RED_B).move_to(M((0.375, 1)) + UP * 0.3)
        leg_lo = tag("¾", 30, RED_B).move_to(M((1, 0.375)) + RIGHT * 0.3)
        self.play(FadeIn(tup), FadeIn(tlo), FadeIn(leg_up), FadeIn(leg_lo),
                  FadeOut(samples[0]), FadeOut(samples[1]), run_time=1.0)
        self.hold(0.5)

        # slide them together: two translations
        self.play(tlo.animate.shift(M((d, 0)) - M((0, 0))),
                  tup.animate.shift(M((w + d, -w)) - M((0, 0))),
                  FadeOut(leg_up), FadeOut(leg_lo), run_time=2.0)
        check(_same_poly(tlo.get_vertices(), [M(p) for p in lo2], 1e-6)
              and _same_poly(tup.get_vertices(), [M(p) for p in up2], 1e-6),
              "the triangles land on the ¾ × ¾ square")
        out2 = Polygon(*[M(p) for p in sq2], stroke_color=YELLOW_B, stroke_width=5)
        s_top = tag("¾", 30, YELLOW_B).move_to(M(((w + d + 1 + d) / 2, 1 - w)) + UP * 0.3)
        s_rgt = tag("¾", 30, YELLOW_B).move_to(M((1 + d, (1 - w) / 2)) + RIGHT * 0.3)
        self.play(Create(out2), FadeIn(s_top), FadeIn(s_rgt), run_time=0.9)

        t1 = _crow([("(¾)²", RED_B), ("=", WHITE), ("9/16", RED_B)], 34)
        t2 = _crow([("1", WHITE), ("−", WHITE), ("9/16", RED_B), ("=", WHITE),
                    ("7/16", GREEN_B)], 34)
        xc = float(M(((w + d + 1 + d) / 2, 0))[0])
        t1.shift(np.array([xc - t1.get_center()[0], 3.25, 0.0]))
        t2.shift(np.array([xc - t2.get_center()[0], 2.5, 0.0]))
        self.play(FadeIn(t1), run_time=0.8)
        self.play(FadeIn(t2), Indicate(bnd, color=GREEN_B, scale_factor=1.0), run_time=1.0)

        segs = ([(M(sq[i]), M(sq[(i + 1) % 4])) for i in range(4)]
                + [(M(sq2[i]), M(sq2[(i + 1) % 4])) for i in range(4)]
                + [(e1.get_start(), e1.get_end()), (e2.get_start(), e2.get_end()),
                   (axx.get_start(), axx.get_end()), (axy.get_start(), axy.get_end())]
                + [(m.get_start(), m.get_end()) for m in ticks])
        _labels_ok([lx, ly, s_top, s_rgt, t1, t2] + tl[1:], segs, "K21 labels")
        _labels_ok([lab_b], [], "K21 band label")
        check(t2.get_bottom()[1] > s_top.get_top()[1] + 0.2, "texts above the ¾ square")
        cap = caption("they meet with probability   1 − (¾)²  =  7/16", 32)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K28

def _holds_centre(P):
    """True if the triangle P (three 2D points) has the origin strictly inside."""
    s = [np.sign(P[i][0] * P[(i + 1) % 3][1] - P[i][1] * P[(i + 1) % 3][0]) for i in range(3)]
    return abs(sum(s)) == 3


class K28_RandomTriangleCentre(Board):
    """Three independent uniform points on a circle are three uniform
    diameters and, for each, a fair choice of one of its two ends. Fix the
    diameters: the 8 choices of ends are equally likely, and they come in
    4 pairs, a triangle and its half-turn about the centre O. The six ends
    alternate around the circle; three ends next to each other lie within
    a half circle and miss O, so a triangle holds O exactly when its
    corners are every other end: one pair, 2 of the 8, whatever the
    diameters. So P(O inside) = 2/8 = ¼."""

    def construct(self):
        # ---------------- the mathematics, checked
        rng = np.random.default_rng(28)
        for _ in range(400):
            th = rng.uniform(0, PI, 3)
            tris = {e: [e[i] * np.array([np.cos(th[i]), np.sin(th[i])]) for i in range(3)]
                    for e in product((1, -1), repeat=3)}
            good = [e for e, P in tris.items() if _holds_centre(P)]
            check(len(good) == 2 and good[0] == tuple(-x for x in good[1]),
                  "always exactly two of the eight hold O: a triangle and its half-turn")
            ends = sorted([(th[i] + (0 if e > 0 else PI)) % TAU, i, e]
                          for i in range(3) for e in (1, -1))
            for e in tris:
                pos = sorted(q for q, (_a, i, s) in enumerate(ends) if s == e[i])
                alt = pos in ([0, 2, 4], [1, 3, 5])
                check(alt == _holds_centre(tris[e]), "O inside <=> the corners are every other end")
        deg = {"A": 60, "B": 200, "C": 135}                # the three points, in degrees
        names = ("A", "B", "C")
        th = [deg[n] * DEGREES for n in names]
        pats = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]    # top row; bottom = half-turns
        tri0 = [np.array([np.cos(t), np.sin(t)]) for t in th]
        check(not _holds_centre(tri0), "the first triangle misses O")
        winners = [e for e in product((1, -1), repeat=3)
                   if _holds_centre([e[i] * tri0[i] for i in range(3)])]
        check(sorted(winners) == sorted([(1, 1, -1), (-1, -1, 1)]), "the winning pair")
        for e in product((1, -1), repeat=3):
            P = [e[i] * tri0[i] for i in range(3)]
            dmin = min(abs(P[i][0] * P[(i + 1) % 3][1] - P[i][1] * P[(i + 1) % 3][0])
                       / np.linalg.norm(P[(i + 1) % 3] - P[i]) for i in range(3))
            check(dmin > 0.3, "O is clearly inside or outside every triangle")

        cols = [BLUE_C, ORANGE, TEAL_C]

        def figure(ctr, r, e, sw=2.0, dr=0.05, big=0.08, op=0.7):
            """Circle, its three diameters, the six ends, and the triangle
            on the ends chosen by the signs e (None: no triangle)."""
            O = to3(ctr)
            circ = Circle(radius=r, color=GREY_B, stroke_width=sw).move_to(O)
            dias = VGroup(*[Line(O + r * _dir(t), O - r * _dir(t), color=c, stroke_width=sw)
                            for t, c in zip(th, cols)])
            ends = VGroup(*[Dot(O + s * r * _dir(t), radius=dr, color=c)
                            for t, c in zip(th, cols) for s in (1, -1)])
            g = VGroup(circ, dias, ends)
            tri = None
            if e is not None:
                pts = [O + e[i] * r * _dir(th[i]) for i in range(3)]
                inside = _holds_centre([p[:2] - O[:2] for p in pts])
                tri = Polygon(*pts, fill_color=GREEN_D if inside else RED_E,
                              fill_opacity=op, stroke_color=WHITE, stroke_width=sw)
                corners = VGroup(*[Dot(pts[i], radius=big, color=cols[i]) for i in range(3)])
                g.add(tri, corners)
            g.add(Dot(O, radius=dr * 1.2, color=WHITE))
            return g, tri

        # ---------------- three random points
        C0, R0 = np.array([-4.2, 0.35, 0.0]), 2.15
        circ = Circle(radius=R0, color=GREY_B, stroke_width=3).move_to(C0)
        cen = Dot(C0, radius=0.08, color=WHITE)
        pts = [C0 + R0 * _dir(t) for t in th]
        pdots = VGroup(*[Dot(p, radius=0.11, color=c) for p, c in zip(pts, cols)])
        tri = Polygon(*pts, fill_color=RED_E, fill_opacity=0.7, stroke_color=WHITE,
                      stroke_width=3)
        self.play(Create(circ), FadeIn(cen), run_time=0.8)
        self.play(FadeIn(pdots, lag_ratio=0.3), run_time=0.8)
        self.play(Create(tri), run_time=0.7)
        self.bring_to_front(pdots, cen)
        self.hold(0.3)

        # each point is an end of a diameter
        dias = VGroup(*[Line(p, 2 * C0 - p, color=c, stroke_width=3) for p, c in zip(pts, cols)])
        odots = VGroup(*[Dot(2 * C0 - p, radius=0.11, color=c) for p, c in zip(pts, cols)])
        self.play(*[Create(d) for d in dias], FadeIn(odots, lag_ratio=0.3), run_time=1.2)
        self.bring_to_front(tri, pdots, odots, cen)
        self.hold(0.3)

        # its corners are three neighbouring ends: all in one half circle
        phi = 40 * DEGREES
        check(all(0.25 < (t - phi) % TAU < PI - 0.25 for t in th),
              "the three corners lie inside the half circle from 40° to 220°")
        half = Arc(radius=R0 + 0.14, start_angle=phi, angle=PI, arc_center=C0,
                   color=YELLOW_B, stroke_width=6)
        sep = DashedLine(C0 + (R0 + 0.3) * _dir(phi), C0 - (R0 + 0.3) * _dir(phi),
                         color=YELLOW_B, stroke_width=3, dash_length=0.1)
        check(_inside(half) and _inside(sep), "the half circle inside the frame")
        self.play(Create(half), Create(sep), run_time=1.0)
        self.hold(0.7)
        self.play(FadeOut(half), FadeOut(sep), run_time=0.5)

        # ---------------- the eight choices of ends, in half-turn pairs
        gx = [-0.55, 1.3, 3.15, 5.0]
        gy = [2.25, 0.1]
        rp = 0.8
        panels, ptri = {}, {}
        for c_, e in enumerate(pats):
            for r_, sgn in enumerate((1, -1)):
                ee = tuple(sgn * x for x in e)
                g, t_ = figure((gx[c_], gy[r_]), rp, ee)
                panels[ee], ptri[ee] = g, t_
        top8 = tag("2³ = 8", 30).move_to([(gx[0] + gx[-1]) / 2, 3.45, 0])
        self.play(LaggedStart(*[FadeIn(panels[tuple(s * x for x in e)])
                                for e in pats for s in (1, -1)], lag_ratio=0.18),
                  FadeIn(top8), run_time=3.2)
        self.hold(0.5)

        # exactly one pair holds the centre
        rings = VGroup(*[Circle(radius=rp + 0.12, color=YELLOW_B, stroke_width=5)
                         .move_to([gx[1], gy[r_], 0]) for r_ in (0, 1)])
        res = tag("2 / 8  =  ¼", 34, YELLOW_B).move_to([(gx[0] + gx[-1]) / 2, -1.55, 0])
        self.play(Create(rings), run_time=0.8)
        self.play(FadeIn(res), run_time=0.7)

        # in the big circle: the winner and its half-turn about O
        wpts = [C0 + w_ * R0 * _dir(t) for w_, t in zip((1, 1, -1), th)]
        win = Polygon(*wpts, fill_color=GREEN_D, fill_opacity=0.55, stroke_color=WHITE,
                      stroke_width=3)
        self.play(FadeOut(tri), FadeIn(win), run_time=0.8)
        self.bring_to_front(pdots, odots, cen)
        win2 = win.copy()
        self.add(win2)
        self.play(Rotate(win2, angle=PI, about_point=C0), run_time=1.8)
        self.bring_to_front(pdots, odots, cen)
        check(_same_poly(win2.get_vertices(),
                         [C0 + w_ * R0 * _dir(t) for w_, t in zip((-1, -1, 1), th)], 1e-6),
              "the half-turn of the winner is the other winner")
        check(_pip(C0, wpts) and _pip(C0, win2.get_vertices()), "both hold O")

        gal = VGroup(*panels.values(), rings)
        check(_inside(gal) and _inside(VGroup(circ, dias, pdots, odots)), "inside the frame")
        check(VGroup(circ, pdots, odots).get_right()[0] < gal.get_left()[0] - 0.3,
              "big circle clear of the gallery")
        check(_apart(top8, gal, 0.1) and _apart(res, gal, 0.1), "texts clear of the gallery")
        cap = caption("three random points:  P(centre inside)  =  2/8  =  ¼", 32)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K32

def _glaisher(lam):
    """Distinct parts -> odd parts: each part 2^a·m (m odd) becomes 2^a parts m."""
    out = []
    for p in lam:
        a = 1
        while p % 2 == 0:
            p //= 2
            a *= 2
        out += [p] * a
    return tuple(sorted(out, reverse=True))


def _merge(mu):
    """Odd parts -> distinct parts: c equal parts m merge into the parts
    2^a·m, one for each binary digit 2^a of c."""
    out = []
    for m in sorted(set(mu)):
        c, a = mu.count(m), 1
        while c:
            if c & 1:
                out.append(a * m)
            c >>= 1
            a *= 2
    return tuple(sorted(out, reverse=True))


_ODDCOL = {1: TEAL_D, 3: ORANGE, 5: BLUE_D, 7: GREEN_D, 9: PURPLE_B}


def _odd_part(p):
    while p % 2 == 0:
        p //= 2
    return p


class K32_DistinctOddParts(Board):
    """A part 2^a·m (m odd) of a partition into distinct parts is halved a
    times: 2^a parts m. Every part is now odd. Back: c equal odd parts m
    are merged by the binary digits of c, one part 2^a·m for each digit
    2^a. Parts with the same odd m came from distinct powers 2^a, so this
    is exactly the grouping they came from, and different digits give
    different parts: the two maps undo each other. Shown on
    10 + 5 + 4 + 2 = 21, then for all six partitions of 8 each way."""

    def construct(self):
        # ---------------- the mathematics, checked
        for n in range(1, 31):
            P = _partitions(n)
            D = [l for l in P if len(set(l)) == len(l)]
            Od = [l for l in P if all(x % 2 for x in l)]
            check(sorted(_glaisher(l) for l in D) == sorted(Od),
                  f"n = {n}: halving maps distinct onto odd")
            check(all(_merge(_glaisher(l)) == l for l in D)
                  and all(_glaisher(_merge(m)) == m for m in Od), "the maps undo each other")
        lam = (10, 5, 4, 2)
        mu = _glaisher(lam)
        check(mu == (5, 5, 5, 1, 1, 1, 1, 1, 1) and sum(lam) == sum(mu) == 21,
              "10 + 5 + 4 + 2 -> 5+5+5+1+1+1+1+1+1")
        check(mu.count(5) == 3 == 2 + 1 and mu.count(1) == 6 == 4 + 2, "3 = 2 + 1, 6 = 4 + 2")

        u, pch = 0.34, 0.44
        xL, y0 = -6.4, 2.95

        def bar(x, y, L, col, sw=1.5):
            return VGroup(*[Square(side_length=u, fill_color=col, fill_opacity=0.9,
                                   stroke_color=WHITE, stroke_width=sw)
                            .move_to([x + (i + 0.5) * u, y, 0]) for i in range(L)])

        ys = [y0 - r * pch for r in range(6)]
        left = [bar(xL, ys[r], L, _ODDCOL[_odd_part(L)]) for r, L in enumerate(lam)]
        headL = _crow([("10", BLUE_B), ("+", WHITE), ("5", BLUE_B), ("+", WHITE),
                       ("4", TEAL_B), ("+", WHITE), ("2", TEAL_B), ("=  21", WHITE)], 28, 0.13)
        headL.shift(np.array([xL - headL.get_left()[0], 3.42, 0.0]))
        self.play(LaggedStart(*[FadeIn(b, lag_ratio=0.1) for b in left], lag_ratio=0.3),
                  FadeIn(headL), run_time=1.6)
        self.hold(0.3)

        # ---------------- halve every even part until all parts are odd
        def cut(r, at):
            x = xL + at * u
            return Line([x, ys[r] - u / 2 - 0.09, 0], [x, ys[r] + u / 2 + 0.09, 0],
                        color=YELLOW_B, stroke_width=6)

        cuts1 = [cut(0, 5), cut(2, 2), cut(3, 1)]
        self.play(*[Create(c) for c in cuts1], run_time=0.8)
        cuts2 = [cut(2, 1), cut(2, 3)]
        self.play(*[Create(c) for c in cuts2], run_time=0.7)
        self.hold(0.3)

        # the pieces, sorted by their odd part
        x5, x1 = -1.7, 2.75
        pieces = [(0, 0, 5, x5, 0), (0, 5, 5, x5, 1), (1, 0, 5, x5, 2),
                  (2, 0, 1, x1, 0), (2, 1, 1, x1, 1), (2, 2, 1, x1, 2), (2, 3, 1, x1, 3),
                  (3, 0, 1, x1, 4), (3, 1, 1, x1, 5)]      # (row, first cell, length, x, row)
        check(sorted(L for _, _, L, _, _ in pieces) == sorted(mu), "the pieces are the odd parts")
        for r, L in enumerate(lam):
            ps = sorted((c0, ln) for rr, c0, ln, _, _ in pieces if rr == r)
            check(ps[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in zip(ps, ps[1:]))
                  and ps[-1][0] + ps[-1][1] == L and len({ln for _, ln in ps}) == 1
                  and len(ps) & (len(ps) - 1) == 0, f"{L} is cut into 2^a equal odd pieces")
        movers, targets = [], []
        for r, c0, ln, xt, rt in pieces:
            src = VGroup(*[left[r][i].copy() for i in range(c0, c0 + ln)])
            movers.append(src)
            targets.append(np.array([xt + ln * u / 2, ys[rt], 0.0]))
        self.play(LaggedStart(*[m.animate.move_to(t) for m, t in zip(movers, targets)],
                              lag_ratio=0.1), run_time=2.0)
        for m, (r, c0, ln, xt, rt) in zip(movers, pieces):
            check(close(m.get_left()[0], xt) and close(m.get_center()[1], ys[rt]),
                  "a piece lands in its column")
        headR = _crow([("5", BLUE_B), ("+", WHITE), ("5", BLUE_B), ("+", WHITE), ("5", BLUE_B),
                       ("+", WHITE)] + sum([[("1", TEAL_B), ("+", WHITE)] for _ in range(5)], [])
                      + [("1", TEAL_B), ("=  21", WHITE)], 28, 0.1)
        headR.shift(np.array([x5 - headR.get_left()[0], 3.42, 0.0]))
        self.play(FadeIn(headR), *[FadeOut(c) for c in cuts1 + cuts2], run_time=0.8)
        self.hold(0.3)

        # ---------------- back: merge by the binary digits of the counts
        def brk(x, ya, yb, col=YELLOW_B):
            return VGroup(Line([x, ya, 0], [x, yb, 0], color=col, stroke_width=3),
                          Line([x - 0.1, ya, 0], [x, ya, 0], color=col, stroke_width=3),
                          Line([x - 0.1, yb, 0], [x, yb, 0], color=col, stroke_width=3))

        bx5, bx1 = x5 + 5 * u + 0.18, x1 + u + 0.18
        groups = [([0, 1], bx5, "2"), ([2], bx5, "1"), ([3, 4, 5, 6], bx1, "4"), ([7, 8], bx1, "2")]
        brs, blabs = [], []
        for idx, bx, s in groups:
            ya = ys[pieces[idx[0]][4]] + u / 2
            yb = ys[pieces[idx[-1]][4]] - u / 2
            b = brk(bx, ya, yb)
            brs.append(b)
            blabs.append(tag(s, 28, YELLOW_B).move_to([bx + 0.28, (ya + yb) / 2, 0]))
        cnt5 = tag("3 = 2 + 1", 26, GREY_A).move_to([x5 + 5 * u / 2, ys[3] - 0.08, 0])
        cnt1 = tag("6 = 4 + 2", 26, GREY_A).move_to([x1 + 2.1, ys[3] - 0.08, 0])
        self.play(*[Create(b) for b in brs], *[FadeIn(t) for t in blabs], run_time=0.9)
        self.play(FadeIn(cnt5), FadeIn(cnt1), run_time=0.7)
        self.hold(0.4)
        backs, lands = [], []
        for idx, _, _ in groups:
            for q in idx:
                r, c0, ln, xt, rt = pieces[q]
                g = movers[q].copy()
                backs.append(g)
                lands.append(np.array([xL + (c0 + ln / 2) * u, ys[r], 0.0]))
        self.play(LaggedStart(*[g.animate.move_to(t) for g, t in zip(backs, lands)],
                              lag_ratio=0.08), run_time=1.8)
        for g, (r, c0, ln, xt, rt) in zip(backs, pieces):
            check(_boxes_of(g) == _boxes_of(VGroup(*[left[r][i] for i in range(c0, c0 + ln)])),
                  "merged pieces land exactly on the original part")
        self.play(*[Indicate(b, color=YELLOW_B, scale_factor=1.0) for b in left],
                  FadeOut(VGroup(*backs)), run_time=0.8)

        # ---------------- all partitions of 8, each way
        D8 = [l for l in _partitions(8) if len(set(l)) == len(l)]
        O8 = [_glaisher(l) for l in D8]
        check(len(D8) == 6 and sorted(O8) == sorted(l for l in _partitions(8)
                                                     if all(x % 2 for x in l)), "6 <-> 6")
        v, vp = 0.16, 0.2
        gx = [-5.45 + 2.15 * q for q in range(6)]
        yT, yO = -0.3, -1.25

        def stack(l, xc, ytop):
            w = max(l) * v
            return VGroup(*[Square(side_length=v, fill_color=_ODDCOL[_odd_part(p)],
                                   fill_opacity=0.9, stroke_color=WHITE, stroke_width=1)
                            .move_to([xc - w / 2 + (i + 0.5) * v, ytop - r * vp, 0])
                            for r, p in enumerate(l) for i in range(p)])

        tops = [stack(l, x, yT) for l, x in zip(D8, gx)]
        bots = [stack(l, x, yO) for l, x in zip(O8, gx)]
        arrows = [Arrow([x, yT - (len(l) - 1) * vp - 0.12, 0], [x, yO + 0.12, 0], buff=0,
                        color=GREY_B, stroke_width=2.5, tip_length=0.12,
                        max_tip_length_to_length_ratio=0.5) for l, x in zip(D8, gx)]
        self.play(LaggedStart(*[FadeIn(VGroup(t, a, b)) for t, a, b in zip(tops, arrows, bots)],
                              lag_ratio=0.2), run_time=2.6)

        top = VGroup(*left, *movers, *brs, *blabs, cnt5, cnt1, headL, headR)
        gal = VGroup(*tops, *bots, *arrows)
        check(_inside(top) and _inside(gal), "inside the frame")
        check(gal.get_top()[1] < min(cnt5.get_bottom()[1], cnt1.get_bottom()[1],
                                      VGroup(*movers).get_bottom()[1]) - 0.15,
              "the table clear of the example")
        _labels_ok([headL, headR, cnt5, cnt1] + blabs,
                   [s for b in brs for s in ((b[0].get_start(), b[0].get_end()),)],
                   "K32 labels")
        _clear_of([headL, headR, cnt5, cnt1] + blabs, list(left) + movers, "K32 labels")
        cap = caption("partitions of n into distinct parts  =  partitions of n into odd parts",
                      30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K7

def _seqs(n):
    """Tilings of the 2 × n strip as sequences of 'V' (one upright domino)
    and 'H' (a flat pair: two flat dominoes, one above the other)."""
    if n < 0:
        return []
    if n == 0:
        return [()]
    return [("V",) + t for t in _seqs(n - 1)] + [("H",) + t for t in _seqs(n - 2)]


def _dominoes(seq, x0=0):
    """The dominoes of a tiling, as frozensets of cells (column, row)."""
    out, x = [], x0
    for p in seq:
        if p == "V":
            out.append(frozenset({(x, 0), (x, 1)}))
            x += 1
        else:
            out += [frozenset({(x, 0), (x + 1, 0)}), frozenset({(x, 1), (x + 1, 1)})]
            x += 2
    return out


def _brute_tilings(n):
    """Every domino tiling of the 2 × n strip by exhaustive search."""
    out = []

    def go(covered, pieces):
        free = [(x, y) for x in range(n) for y in (0, 1) if (x, y) not in covered]
        if not free:
            out.append(frozenset(pieces))
            return
        x, y = free[0]
        if y == 0 and (x, 1) not in covered:
            go(covered | {(x, 0), (x, 1)}, pieces + [frozenset({(x, 0), (x, 1)})])
        if x + 1 < n and (x + 1, y) not in covered:
            go(covered | {(x, y), (x + 1, y)}, pieces + [frozenset({(x, y), (x + 1, y)})])

    go(frozenset(), [])
    return out


def _rect(x, y, w, h, col, sw=1.2):
    return mk([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], col, 0.9,
              stroke_width=sw)


def _tiling_mob(seq, org, u, x0=0, sw=1.2):
    """The dominoes of a tiling drawn with cell u, its column 0 at org
    (bottom-left), starting at column x0: upright teal, flat orange."""
    o = to3(org)
    g, x = VGroup(), x0
    for p in seq:
        if p == "V":
            g.add(_rect(o[0] + x * u, o[1], u, 2 * u, TEAL_D, sw))
            x += 1
        else:
            g.add(_rect(o[0] + x * u, o[1], 2 * u, u, ORANGE, sw),
                  _rect(o[0] + x * u, o[1] + u, 2 * u, u, ORANGE, sw))
            x += 2
    return g


class K7_FibonacciAddition(Board):
    """A domino tiling of the 2 × 7 strip either breaks along the line
    after column 3 — a tiling of 2 × 3 beside one of 2 × 4, t₃·t₄ ways —
    or a domino lies across that line. It is flat, and flat dominoes come
    in pairs one above the other, so a flat pair covers columns 3 and 4,
    with a tiling of 2 × 2 on its left and one of 2 × 3 on its right:
    t₂·t₃ ways. All 21 tilings are listed as the two products. The same
    split of a 2 × (m + n − 1) strip after column m − 1 gives, with
    tₖ = Fₖ₊₁, Fₘ₊ₙ = Fₘ·Fₙ₊₁ + Fₘ₋₁·Fₙ (here m = n = 4)."""

    def construct(self):
        # ---------------- the mathematics, checked
        F = [0, 1]
        for _ in range(30):
            F.append(F[-1] + F[-2])
        for k in range(0, 11):
            check(len(_seqs(k)) == F[k + 1], f"t{k} = F{k + 1}")
            check(sorted(map(sorted, map(lambda s: [tuple(sorted(d)) for d in s],
                                         [_dominoes(s) for s in _seqs(k)])))
                  == sorted(map(sorted, map(lambda s: [tuple(sorted(d)) for d in s],
                                            _brute_tilings(k)))),
                  f"V / flat-pair sequences give every tiling of 2 × {k}")
        for a in range(1, 8):
            for b in range(1, 8):
                check(len(_seqs(a + b)) == len(_seqs(a)) * len(_seqs(b))
                      + len(_seqs(a - 1)) * len(_seqs(b - 1)), "t(a+b) = ta·tb + t(a-1)·t(b-1)")
        for m in range(2, 15):
            for n in range(1, 15):
                check(F[m + n] == F[m] * F[n + 1] + F[m - 1] * F[n], "the Fibonacci identity")
        LA, RA = _seqs(3), _seqs(4)
        LB, RB = _seqs(2), _seqs(3)
        cellsA = [frozenset(_dominoes(l) + _dominoes(r, 3)) for l in LA for r in RA]
        cellsB = [frozenset(_dominoes(l) + _dominoes(("H",), 2) + _dominoes(r, 4))
                  for l in LB for r in RB]
        every = set(_brute_tilings(7))
        check(len(cellsA) == 15 and len(cellsB) == 6 and len(every) == 21 == F[8],
              "15 + 6 = 21 = F8")
        check(set(cellsA) | set(cellsB) == every and not set(cellsA) & set(cellsB)
              and len(set(cellsA)) == 15 and len(set(cellsB)) == 6,
              "the two tables list every tiling of 2 × 7 exactly once")
        crosses = lambda t: any(min(c[0] for c in d) < 3 <= max(c[0] for c in d) for d in t)
        check(not any(crosses(t) for t in cellsA) and all(crosses(t) for t in cellsB),
              "table 1 breaks after column 3, table 2 has a flat pair across")

        # ---------------- the two cases
        u0, ys0 = 0.32, 2.98
        xs0 = [-4.5, -1.15, 2.2]                  # left ends of the three strips

        def outline(x, w, color=GREY_B, sw=2.5, dashed=False):
            r = Polygon([x, ys0, 0], [x + w, ys0, 0], [x + w, ys0 + 2 * u0, 0],
                        [x, ys0 + 2 * u0, 0], stroke_color=color, stroke_width=sw)
            return DashedVMobject(r, num_dashes=int(14 + 10 * w)) if dashed else r

        def grid(x):
            g = VGroup(outline(x, 7 * u0))
            for c in range(1, 7):
                g.add(Line([x + c * u0, ys0, 0], [x + c * u0, ys0 + 2 * u0, 0],
                           stroke_color=GREY_D, stroke_width=1))
            g.add(Line([x, ys0 + u0, 0], [x + 7 * u0, ys0 + u0, 0], stroke_color=GREY_D,
                       stroke_width=1))
            return g

        def brk(x):
            return DashedLine([x + 3 * u0, ys0 - 0.12, 0], [x + 3 * u0, ys0 + 2 * u0 + 0.12, 0],
                              color=RED_B, stroke_width=4, dash_length=0.07)

        s0 = VGroup(grid(xs0[0]), brk(xs0[0]))
        l7 = tag("t₇", 30, YELLOW_B).next_to(s0, LEFT, buff=0.25)
        sA, sB = VGroup(grid(xs0[1]), brk(xs0[1])), VGroup(grid(xs0[2]), brk(xs0[2]))
        eq = tag("=", 34).move_to([(xs0[0] + 7 * u0 + xs0[1]) / 2, ys0 + u0, 0])
        pl = tag("+", 34).move_to([(xs0[1] + 7 * u0 + xs0[2]) / 2, ys0 + u0, 0])
        rA1, rA2 = outline(xs0[1], 3 * u0, YELLOW_B, 3, True), \
            outline(xs0[1] + 3 * u0, 4 * u0, YELLOW_B, 3, True)
        tA1 = tag("t₃", 26, YELLOW_B).move_to([xs0[1] + 1.5 * u0, ys0 + u0, 0])
        tA2 = tag("t₄", 26, YELLOW_B).move_to([xs0[1] + 5 * u0, ys0 + u0, 0])
        top = _rect(xs0[2] + 2 * u0, ys0 + u0, 2 * u0, u0, ORANGE, 2.5)
        bot = _rect(xs0[2] + 2 * u0, ys0, 2 * u0, u0, ORANGE, 2.5)
        forced = Square(side_length=u0, stroke_color=RED_B, stroke_width=5).move_to(
            [xs0[2] + 2.5 * u0, ys0 + 0.5 * u0, 0])
        rB1, rB2 = outline(xs0[2], 2 * u0, YELLOW_B, 3, True), \
            outline(xs0[2] + 4 * u0, 3 * u0, YELLOW_B, 3, True)
        tB1 = tag("t₂", 26, YELLOW_B).move_to([xs0[2] + u0, ys0 + u0, 0])
        tB2 = tag("t₃", 26, YELLOW_B).move_to([xs0[2] + 5.5 * u0, ys0 + u0, 0])
        intro = VGroup(s0, l7, sA, sB, eq, pl, rA1, rA2, tA1, tA2, top, bot, forced,
                       rB1, rB2, tB1, tB2)
        # the case analysis is shown enlarged first, then set in the top row
        home = intro.get_center()
        big, mid = 1.36, np.array([-0.15, 1.0, 0.0])
        intro.scale(big, about_point=home).shift(mid - home)
        check(_inside(intro), "the enlarged case analysis inside the frame")

        self.play(Create(s0[0]), FadeIn(l7), run_time=0.8)
        self.play(Create(s0[1]), run_time=0.5)
        self.play(TransformFromCopy(s0, sA), FadeIn(eq), run_time=0.8)
        self.play(Create(rA1), Create(rA2), FadeIn(tA1), FadeIn(tA2), run_time=0.9)
        self.play(TransformFromCopy(s0, sB), FadeIn(pl), run_time=0.8)
        self.play(FadeIn(top), run_time=0.5)
        self.play(Create(forced), run_time=0.4)
        self.play(FadeIn(bot), FadeOut(forced), run_time=0.5)
        self.bring_to_front(sB[1])
        self.play(Create(rB1), Create(rB2), FadeIn(tB1), FadeIn(tB2), run_time=0.9)
        self.hold(0.6)
        intro.remove(forced)
        self.play(intro.animate.shift(home - mid).scale(1 / big, about_point=home), run_time=1.0)
        check(close(s0[0][0].get_corner(DL), [xs0[0], ys0, 0], 1e-6)
              and close(sB[0][0].get_corner(UR), [xs0[2] + 7 * u0, ys0 + 2 * u0, 0], 1e-6),
              "the case analysis is back in the top row")

        # ---------------- every tiling: the two products
        u, px, py = 0.2, 1.75, 0.58
        XA, YA = -6.3, 1.95                        # table 1, cell (0, 0) bottom-left
        XB, YB = -6.3, -0.15                       # table 2

        def orgA(i, j):
            return np.array([XA + j * px, YA - i * py, 0.0])

        def orgB(i, j):
            return np.array([XB + j * px, YB - i * py, 0.0])

        Lm = [[_tiling_mob(LA[i], orgA(i, j), u) for j in range(5)] for i in range(3)]
        Rm = [[_tiling_mob(RA[j], orgA(i, j), u, 3) for j in range(5)] for i in range(3)]
        self.play(LaggedStart(*[FadeIn(Lm[i][0]) for i in range(3)], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(Rm[0][j]) for j in range(5)], lag_ratio=0.2),
                  run_time=1.3)
        self.play(*[TransformFromCopy(Lm[i][0], Lm[i][j]) for i in range(3) for j in range(1, 5)],
                  *[TransformFromCopy(Rm[0][j], Rm[i][j]) for i in range(1, 3) for j in range(5)],
                  run_time=1.5)
        linesA = VGroup(*[DashedLine(orgA(i, j) + np.array([3 * u, -0.07, 0]),
                                     orgA(i, j) + np.array([3 * u, 2 * u + 0.07, 0]),
                                     color=RED_B, stroke_width=3, dash_length=0.05)
                          for i in range(3) for j in range(5)])
        labA = tag("t₃ · t₄  =  3 · 5  =  15", 28).move_to([4.45, YA - py + u, 0])
        self.play(Create(linesA), FadeIn(labA), run_time=0.8)
        for i in range(3):
            for j in range(5):
                want = _tiling_mob(LA[i] + RA[j], orgA(i, j), u)
                check(_boxes_of(VGroup(Lm[i][j], Rm[i][j])) == _boxes_of(want),
                      "row tiling + column tiling = the listed tiling")

        Lb = [[_tiling_mob(LB[i], orgB(i, j), u) for j in range(3)] for i in range(2)]
        Hb = [[_tiling_mob(("H",), orgB(i, j), u, 2) for j in range(3)] for i in range(2)]
        Rb = [[_tiling_mob(RB[j], orgB(i, j), u, 4) for j in range(3)] for i in range(2)]
        for i in range(2):
            for j in range(3):
                Hb[i][j].set_stroke(YELLOW_B, 2.5)
        self.play(*[FadeIn(Hb[i][j]) for i in range(2) for j in range(3)], run_time=0.6)
        self.play(LaggedStart(*[FadeIn(Lb[i][0]) for i in range(2)], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(Rb[0][j]) for j in range(3)], lag_ratio=0.25),
                  run_time=1.0)
        self.play(*[TransformFromCopy(Lb[i][0], Lb[i][j]) for i in range(2) for j in range(1, 3)],
                  *[TransformFromCopy(Rb[0][j], Rb[1][j]) for j in range(3)], run_time=1.2)
        linesB = VGroup(*[DashedLine(orgB(i, j) + np.array([3 * u, -0.07, 0]),
                                     orgB(i, j) + np.array([3 * u, 2 * u + 0.07, 0]),
                                     color=RED_B, stroke_width=3, dash_length=0.05)
                          for i in range(2) for j in range(3)])
        labB = tag("t₂ · t₃  =  2 · 3  =  6", 28).move_to([0.95, YB - py / 2 + u, 0])
        self.play(Create(linesB), FadeIn(labB), run_time=0.8)
        for i in range(2):
            for j in range(3):
                want = _tiling_mob(LB[i] + ("H",) + RB[j], orgB(i, j), u)
                check(_boxes_of(VGroup(Lb[i][j], Hb[i][j], Rb[i][j])) == _boxes_of(want),
                      "left tiling + flat pair + right tiling = the listed tiling")

        sum1 = tag("t₇  =  t₃ · t₄  +  t₂ · t₃  =  15 + 6  =  21", 30, WHITE)
        sum1.move_to([0, -1.62, 0])
        sum2 = tag("F₈  =  F₄ · F₅  +  F₃ · F₄", 30, WHITE).move_to([0, -2.32, 0])
        self.play(FadeIn(sum1), run_time=0.8)
        self.play(FadeIn(sum2), run_time=0.7)

        tabA = VGroup(*[m for row in Lm + Rm for m in row], linesA)
        tabB = VGroup(*[m for row in Lb + Hb + Rb for m in row], linesB)
        intro = VGroup(s0, sA, sB, rA1, rA2, rB1, rB2, top, bot, l7, eq, pl)
        check(_inside(tabA) and _inside(tabB) and _inside(intro), "inside the frame")
        check(intro.get_bottom()[1] > tabA.get_top()[1] + 0.25
              and tabA.get_bottom()[1] > tabB.get_top()[1] + 0.25
              and tabB.get_bottom()[1] > sum1.get_top()[1] + 0.25, "rows clear of each other")
        check(labA.get_left()[0] > tabA.get_right()[0] + 0.3
              and labB.get_left()[0] > tabB.get_right()[0] + 0.3, "table labels clear")
        _labels_ok([l7, eq, pl, labA, labB, sum1, sum2], [], "K7 labels")
        for t_, reg in ((tA1, rA1), (tA2, rA2), (tB1, rB1), (tB2, rB2)):
            r0, r1 = reg.get_corner(DL), reg.get_corner(UR)
            check(r0[0] + 0.03 < t_.get_left()[0] and t_.get_right()[0] < r1[0] - 0.03
                  and r0[1] + 0.03 < t_.get_bottom()[1] and t_.get_top()[1] < r1[1] - 0.03,
                  f"'{t_.text}' inside its region")
        cap = _cap(_subline([("F", False), ("m+n", True), ("  =  F", False), ("m", True),
                             (" · F", False), ("n+1", True), ("  +  F", False), ("m−1", True),
                             (" · F", False), ("n", True), ("        tilings:  t", False),
                             ("k", True), (" = F", False), ("k+1", True)], 34, YELLOW_B))
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K24

class K24_BertrandParadox(Board):
    """Caution: 'a random chord' is not yet a definition. The inscribed
    equilateral triangle's side has length √3·R, and a chord is longer
    exactly when its midpoint lies within R/2 of the centre (the triangle's
    incircle). Three natural ways to choose the chord make three different
    quantities uniform:
      1. random endpoints: fix one end at a vertex (by symmetry); the chord
         is longer when the other end lies on the opposite third of the
         circle, an arc of 120°: P = ⅓.
      2. a random point on a random radius, chord perpendicular there:
         longer when the point lies on the inner half of the radius: P = ½.
      3. a random midpoint in the disc: longer when it lies in the disc of
         radius R/2, of area (½)² of the whole: P = ¼.
    The event is the same each time; the answers differ because 'uniform'
    refers to arc, to distance, or to area. Evenly spread midpoints of each
    kind show the three different distributions."""

    def construct(self):
        R = 1.5
        CEN = [np.array([-4.45, 0.62, 0.0]), np.array([0.0, 0.62, 0.0]),
               np.array([4.45, 0.62, 0.0])]
        side = np.sqrt(3) * R
        vtx = [270, 30, 150]                       # the triangle's vertices (degrees)
        LONG, SHORT = GREEN_C, RED_C

        # ---------------- the mathematics, checked
        for c in CEN[:1]:
            pts = [R * _dir(a * DEGREES) for a in vtx]
            check(all(close(np.linalg.norm(pts[i] - pts[(i + 1) % 3]), side) for i in range(3)),
                  "the inscribed triangle is equilateral with side √3·R")
            for i in range(3):
                a, b = pts[i], pts[(i + 1) % 3]
                dist = abs(a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(b - a)
                check(close(dist, R / 2), "its sides are R/2 from the centre (the incircle)")
        for k in range(1, 720):
            phi = k * 0.5
            if abs(phi - 30) > 0.3 and abs(phi - 150) > 0.3:
                ang = abs(((phi - 270) + 180) % 360 - 180)
                L = 2 * R * np.sin(np.radians(ang) / 2)
                check((L > side) == (30 < phi < 150), "1: longer <=> the far third of the circle")
        for k in range(1, 200):
            d = k / 200 * R
            if abs(d - R / 2) > 1e-3:
                check((2 * np.sqrt(R * R - d * d) > side) == (d < R / 2),
                      "2, 3: longer <=> within R/2 of the centre")
        rng = np.random.default_rng(24)
        M = 400000
        a1, a2 = rng.uniform(0, TAU, M), rng.uniform(0, TAU, M)
        p1 = np.mean(2 * R * np.abs(np.sin((a1 - a2) / 2)) > side)
        p2 = np.mean(2 * np.sqrt(R * R - (rng.uniform(0, R, M)) ** 2) > side)
        xy = rng.uniform(-R, R, (2, 2 * M))
        rr = np.hypot(*xy)
        rr = rr[rr < R][:M]
        p3 = np.mean(2 * np.sqrt(R * R - rr * rr) > side)
        check(abs(p1 - 1 / 3) < 0.004 and abs(p2 - 1 / 2) < 0.004 and abs(p3 - 1 / 4) < 0.004,
              "simulation agrees: 1/3, 1/2, 1/4")

        # ---------------- three circles, one triangle
        head = tag("Is a random chord longer than the side of the inscribed triangle?", 26)
        head.move_to([0, 3.45, 0])
        titles = [tag(s, 22, GREY_A).move_to(c + UP * (R + 0.45)) for s, c in
                  zip(("random endpoints", "random point on a radius", "random midpoint"), CEN)]
        circs = [Circle(radius=R, color=GREY_B, stroke_width=3).move_to(c) for c in CEN]
        tris = [Polygon(*[c + R * _dir(a * DEGREES) for a in vtx], stroke_color=WHITE,
                        stroke_width=2.5) for c in CEN]
        cdots = [Dot(c, radius=0.05, color=WHITE) for c in CEN]
        self.play(FadeIn(head), *[Create(ci) for ci in circs], *[FadeIn(d) for d in cdots],
                  run_time=1.0)
        self.play(*[Create(t) for t in tris], run_time=0.8)

        # ---------------- 1. random endpoints: one end fixed at a vertex
        c = CEN[0]
        P = c + R * _dir(270 * DEGREES)
        th = ValueTracker(272.0)

        def q_of(t):
            return c + R * _dir(t * DEGREES)

        def chord1():
            ln = np.linalg.norm(q_of(th.get_value()) - P)
            return Line(P, q_of(th.get_value()), color=LONG if ln > side else SHORT,
                        stroke_width=5)

        ch1 = always_redraw(chord1)
        qd = always_redraw(lambda: Dot(q_of(th.get_value()), radius=0.09, color=WHITE))
        pd = Dot(P, radius=0.11, color=YELLOW_B)
        self.play(FadeIn(titles[0]), FadeIn(pd), run_time=0.6)
        self.add(ch1, qd)
        self.play(th.animate.set_value(268.0 + 360.0), run_time=3.0, rate_func=linear)
        self.remove(ch1, qd, th)
        far = Arc(radius=R, start_angle=30 * DEGREES, angle=120 * DEGREES, arc_center=c,
                  color=LONG, stroke_width=9)
        near = VGroup(Arc(radius=R, start_angle=150 * DEGREES, angle=120 * DEGREES,
                          arc_center=c, color=SHORT, stroke_width=9),
                      Arc(radius=R, start_angle=270 * DEGREES, angle=120 * DEGREES,
                          arc_center=c, color=SHORT, stroke_width=9))
        ex1 = VGroup(Line(P, q_of(75), color=LONG, stroke_width=4),
                     Line(P, q_of(345), color=SHORT, stroke_width=4))
        pr1 = tag("120° / 360°  =  ⅓", 30, YELLOW_B).move_to(c + DOWN * (R + 0.55))
        self.play(Create(far), Create(near), FadeIn(ex1), run_time=1.0)
        self.bring_to_front(pd, tris[0])
        self.play(FadeIn(pr1), run_time=0.6)

        # ---------------- 2. a random point on a radius, chord across it there
        c = CEN[1]
        hh = ValueTracker(0.03 * R)

        def chord2():
            h = hh.get_value()
            w = np.sqrt(R * R - h * h)
            return Line(c + np.array([-w, h, 0]), c + np.array([w, h, 0]),
                        color=LONG if h < R / 2 else SHORT, stroke_width=5)

        rad = Line(c, c + UP * R, color=GREY_A, stroke_width=3)
        ch2 = always_redraw(chord2)
        md = always_redraw(lambda: Dot(c + UP * hh.get_value(), radius=0.09, color=WHITE))
        self.play(FadeIn(titles[1]), Create(rad), run_time=0.6)
        self.add(ch2, md)
        self.play(hh.animate.set_value(0.97 * R), run_time=2.6, rate_func=linear)
        self.remove(ch2, md, hh)
        inner = Line(c, c + UP * R / 2, color=LONG, stroke_width=9)
        outer = Line(c + UP * R / 2, c + UP * R, color=SHORT, stroke_width=9)
        h_in, h_out = 0.3 * R, 0.78 * R
        ex2 = VGroup(*[Line(c + np.array([-np.sqrt(R * R - h * h), h, 0]),
                            c + np.array([np.sqrt(R * R - h * h), h, 0]),
                            color=col, stroke_width=4)
                       for h, col in ((h_in, LONG), (h_out, SHORT))])
        pr2 = tag("(R/2) / R  =  ½", 30, YELLOW_B).move_to(c + DOWN * (R + 0.55))
        self.play(Create(inner), Create(outer), FadeIn(ex2), run_time=1.0)
        self.bring_to_front(tris[1], cdots[1])
        self.play(FadeIn(pr2), run_time=0.6)

        # ---------------- 3. a random midpoint in the disc
        c = CEN[2]
        dd = ValueTracker(0.06 * R)
        ua = _dir(215 * DEGREES)
        ut = np.array([-ua[1], ua[0], 0.0])

        def chord3():
            d = dd.get_value()
            w = np.sqrt(R * R - d * d)
            m = c + d * ua
            return Line(m - w * ut, m + w * ut, color=LONG if d < R / 2 else SHORT,
                        stroke_width=5)

        incirc = Circle(radius=R / 2, color=WHITE, stroke_width=2.5).move_to(c)
        ch3 = always_redraw(chord3)
        md3 = always_redraw(lambda: Dot(c + dd.get_value() * ua, radius=0.09, color=WHITE))
        self.play(FadeIn(titles[2]), Create(incirc), run_time=0.7)
        self.add(ch3, md3)
        self.play(dd.animate.set_value(0.96 * R), run_time=2.6, rate_func=linear)
        self.remove(ch3, md3, dd)
        disc_in = Circle(radius=R / 2, stroke_width=0, fill_color=LONG,
                         fill_opacity=0.55).move_to(c)
        ring = Annulus(inner_radius=R / 2, outer_radius=R, stroke_width=0, fill_color=SHORT,
                       fill_opacity=0.3).move_to(c)
        pr3 = tag("(½)²  =  ¼", 30, YELLOW_B).move_to(c + DOWN * (R + 0.55))
        self.add(disc_in, ring)
        self.bring_to_back(disc_in, ring)
        disc_in.set_fill(opacity=0)
        ring.set_fill(opacity=0)
        self.play(disc_in.animate.set_fill(opacity=0.55), ring.animate.set_fill(opacity=0.3),
                  run_time=0.9)
        self.play(FadeIn(pr3), run_time=0.6)
        self.hold(0.6)

        # ---------------- the same event, three different 'uniform's
        N, ga = 96, PI * (3 - np.sqrt(5))
        clouds, inside = [], []
        for q, c in enumerate(CEN):
            ds = []
            for k in range(N):
                if q == 0:
                    d = R * abs(np.cos(((k + 0.5) / N * TAU) / 2))   # ends 2θ apart
                elif q == 1:
                    d = (k + 0.5) / N * R                           # distance uniform
                else:
                    d = R * np.sqrt((k + 0.5) / N)                   # area uniform
                ds.append(d)
            inside.append(sum(d < R / 2 for d in ds))
            clouds.append(VGroup(*[Dot(c + d * _dir(k * ga), radius=0.03,
                                       color=YELLOW_B if d < R / 2 else GREY_A)
                                   for k, d in enumerate(ds)]))
        check(inside == [32, 48, 24], "evenly spread midpoints: 32, 48, 24 of 96 inside")
        dashed = [DashedVMobject(Circle(radius=R / 2, color=YELLOW_B, stroke_width=4),
                                 num_dashes=26).move_to(c) for c in CEN]
        same = tag("the same event: its midpoint lies within R/2 of the centre", 24, YELLOW_B)
        same.move_to([0, -2.2, 0])
        self.play(*[Create(d) for d in dashed], FadeOut(ex1), FadeOut(ex2), run_time=0.9)
        self.play(FadeIn(same), run_time=0.7)
        self.play(*[FadeIn(cl, lag_ratio=0.02) for cl in clouds], run_time=1.6)
        self.bring_to_front(inner, outer, pd, *dashed)

        figs = [VGroup(circs[q], tris[q], dashed[q], clouds[q]) for q in range(3)]
        for q in range(3):
            check(_inside(figs[q]), "panel inside the frame")
            check(titles[q].get_bottom()[1] > figs[q].get_top()[1] + 0.1, "title above its circle")
        _labels_ok([head] + titles + [pr1, pr2, pr3, same], [], "K24 labels")
        _clear_of(titles + [pr1, pr2, pr3, same, head], figs, "K24 labels")
        cap = caption("P(chord longer than the side)  =  ⅓,  ½  or  ¼:  "
                      "it depends on what is chosen uniformly", 28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K25

def _member_row(members, chair, centre, sp=0.24, r=0.085):
    """Four people in a row: members filled blue, the chair gold with a
    white ring, the others hollow."""
    c = to3(centre)
    g = VGroup()
    for p in range(4):
        q = c + np.array([(p - 1.5) * sp, 0.0, 0.0])
        if p == chair:
            g.add(Circle(radius=r * 1.12, fill_color=GOLD_C, fill_opacity=1.0,
                         stroke_color=WHITE, stroke_width=2.5).move_to(q))
        elif p in members:
            g.add(Circle(radius=r, fill_color=BLUE_C, fill_opacity=1.0,
                         stroke_width=0).move_to(q))
        else:
            g.add(Circle(radius=r * 0.8, stroke_color=GREY_B, stroke_width=1.5).move_to(q))
    return g


class K25_CommitteesWithChair(Board):
    """Count the committees with a chair drawn from n people. First the
    committee, then its chair: C(n, k) committees of k people, k chairs
    each, ∑ k·C(n, k) in all. First the chair, then the rest of the
    committee: n chairs, and any of the 2ⁿ⁻¹ subsets of the other people:
    n·2ⁿ⁻¹. The same pairs, counted twice. For n = 4 all 32 are drawn, in
    blocks of C(4, k) committees × k chairs, then copied into a table by
    chair: 4 rows of 2³."""

    def construct(self):
        n = 4
        for m in range(1, 12):
            check(sum(k * comb(m, k) for k in range(m + 1)) == m * 2 ** (m - 1),
                  "∑ k·C(n,k) = n·2^(n−1)")
        pairs = [(c, ch) for k in range(1, n + 1) for c in combinations(range(n), k) for ch in c]
        check(len(pairs) == 32 == n * 2 ** (n - 1) and len(set(pairs)) == 32,
              "32 committees with a chair")
        others = {r: sorted((s for k in range(n) for s in combinations(
            [p for p in range(n) if p != r], k)), key=lambda s: (len(s), s)) for r in range(n)}
        check(all(len(others[r]) == 8 for r in range(n)), "2³ subsets of the other three")
        by_chair = {(tuple(sorted(set(s) | {r})), r) for r in range(n) for s in others[r]}
        check(by_chair == set(pairs), "chair + any subset of the rest = the same 32 pairs")

        sp, rr = 0.24, 0.085
        half = 1.5 * sp + 1.12 * rr                # half-width of an item
        pw, ph, gap = 1.12, 0.44, 0.42
        top_y = 2.95

        # ---------------- blocks: C(4, k) committees × k chairs
        bx, x = {}, 0.0
        for k in range(1, n + 1):
            bx[k] = x
            x += k * pw + gap
        width = x - gap - (pw - 2 * half)               # first item's left edge to last's right
        x0 = -width / 2                                 # left edge of the first item
        items1, home1 = {}, {}
        for k in range(1, n + 1):
            for i, c in enumerate(combinations(range(n), k)):
                for j, ch in enumerate(c):
                    ctr = np.array([x0 + bx[k] + half + j * pw, top_y - i * ph, 0.0])
                    items1[(c, ch)] = _member_row(set(c), ch, ctr, sp, rr)
                    home1[(c, ch)] = ctr
        xl = x0
        hd1 = tag("the committee, then its chair", 24, GREY_A)
        hd1.move_to([xl + hd1.width / 2, 3.45, 0])
        self.play(FadeIn(hd1), run_time=0.6)
        terms = []
        yt = top_y - 5 * ph - 0.55
        for k in range(1, n + 1):
            grp = [items1[(c, ch)] for c in combinations(range(n), k) for ch in c]
            check(len(grp) == k * comb(n, k), f"block {k}: {k}·C(4,{k})")
            cx = x0 + bx[k] + half + (k - 1) * pw / 2
            t = tag(f"{k} · {comb(n, k)}", 28).move_to([cx, yt, 0])
            terms.append(t)
            self.play(LaggedStart(*[FadeIn(g) for g in grp], lag_ratio=0.08), FadeIn(t),
                      run_time=1.0 if k in (2, 3) else 0.7)
        plus = [tag("+", 28).move_to([(terms[q].get_right()[0] + terms[q + 1].get_left()[0]) / 2,
                                      yt, 0]) for q in range(3)]
        tot1 = tag("=  32", 28, YELLOW_B).next_to(terms[-1], RIGHT, buff=0.35)
        self.play(*[FadeIn(p_) for p_ in plus], FadeIn(tot1), run_time=0.7)
        self.hold(0.5)

        # ---------------- the same 32, sorted by chair
        hd2 = tag("the chair, then any subset of the rest", 24, GREY_A)
        y2 = yt - 1.2
        hd2.move_to([xl + hd2.width / 2, y2 + 0.5, 0])
        items2, moves = {}, []
        for r in range(n):
            for j, s in enumerate(others[r]):
                c = tuple(sorted(set(s) | {r}))
                ctr = np.array([xl + half + j * pw, y2 - r * ph, 0.0])
                g = items1[(c, r)].copy()
                items2[(c, r)] = g
                moves.append(g.animate.shift(ctr - home1[(c, r)]))
        self.play(FadeIn(hd2), run_time=0.6)
        self.play(LaggedStart(*moves, lag_ratio=0.04), run_time=3.0)
        for r in range(n):
            for j, s in enumerate(others[r]):
                c = tuple(sorted(set(s) | {r}))
                ctr = np.array([xl + half + j * pw, y2 - r * ph, 0.0])
                check(close(items2[(c, r)][r].get_center(),
                            ctr + np.array([(r - 1.5) * sp, 0, 0]), 1e-6),
                      "row r holds the chair r with each subset of the rest")
        tab2 = VGroup(*items2.values())
        tot2 = _supline([("4 · 2", False, WHITE), ("3", True, WHITE),
                         ("  =  32", False, YELLOW_B)], 30)
        tot2.move_to([tab2.get_right()[0] + 0.6 + tot2.width / 2, y2 - 1.5 * ph, 0])
        self.play(FadeIn(tot2), run_time=0.7)

        tab1 = VGroup(*items1.values())
        check(_inside(tab1) and _inside(tab2) and _inside(tot2) and _inside(tot1),
              "inside the frame")
        check(tab1.get_bottom()[1] > max(t.get_top()[1] for t in terms) + 0.15
              and hd2.get_top()[1] < min(t.get_bottom()[1] for t in terms) - 0.15
              and hd2.get_bottom()[1] > tab2.get_top()[1] + 0.08, "bands clear of each other")
        _labels_ok([hd1, hd2, tot1, tot2] + terms + plus, [], "K25 labels")
        _clear_of([hd1, hd2, tot1, tot2] + terms + plus, [tab1, tab2], "K25 labels")
        cap = _cap(_supline([("∑ k · C(n, k)  =  n · 2", False), ("n−1", True)], 34))
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K26

def _subset_row(members, centre, col, sp=0.3, r=0.1, ring=True):
    """A subset of {1, 2, 3, 4} as four places in a row, members filled in
    col; the first place (element 1) ringed in yellow."""
    c = to3(centre)
    g = VGroup()
    for p in range(4):
        q = c + np.array([(p - 1.5) * sp, 0.0, 0.0])
        filled = p in members
        st = YELLOW_B if (p == 0 and ring) else GREY_B
        sw = 3 if (p == 0 and ring) else 1.5
        g.add(Circle(radius=r if filled else r * 0.85, fill_color=col,
                     fill_opacity=1.0 if filled else 0.0, stroke_color=st,
                     stroke_width=sw if (p == 0 and ring) or not filled else 0).move_to(q))
    return g


class K26_AlternatingRowSum(Board):
    """Pair every subset of {1, …, n} (n ≥ 1) with the subset that differs
    from it only in the element 1: put 1 in if it is missing, take it out
    if it is there. Doing it twice gives the subset back, and no subset is
    paired with itself, so the subsets fall into pairs, and the two in a
    pair differ in size by one: one even, one odd. Hence there are as many
    subsets of even size as of odd size, C(n,0) − C(n,1) + C(n,2) − … = 0.
    Shown for n = 4: each row holds one pair, 8 rows."""

    def construct(self):
        n = 4
        for m in range(1, 12):
            check(sum((-1) ** k * comb(m, k) for k in range(m + 1)) == 0, "alternating sum 0")
        subsets = [frozenset(s) for k in range(n + 1) for s in combinations(range(n), k)]
        tog = {S: S ^ {0} for S in subsets}
        check(all(tog[tog[S]] == S and tog[S] != S and (len(tog[S]) - len(S)) % 2
                  for S in subsets), "toggling 1: an involution, no fixed points, parity changes")
        pairs = sorted([S for S in subsets if 0 not in S], key=lambda S: (len(S), sorted(S)))
        check(len(pairs) == 8 and {S for S in pairs} | {tog[S] for S in pairs} == set(subsets),
              "8 pairs cover all 16 subsets once")

        CE, CO = BLUE_C, ORANGE
        X = [-5.0 + 2.5 * k for k in range(n + 1)]
        Y0, dy = 2.55, 0.47

        def col_of(S):
            return CE if len(S) % 2 == 0 else CO

        pos = {}
        for p, S in enumerate(pairs):
            pos[S] = np.array([X[len(S)], Y0 - p * dy, 0.0])
            pos[tog[S]] = np.array([X[len(S) + 1], Y0 - p * dy, 0.0])
        for k in range(n + 1):
            check(sum(1 for S in subsets if len(S) == k) == comb(n, k), "C(4,k) per column")
        items = {S: _subset_row(S, pos[S], col_of(S)) for S in subsets}
        heads = [tag(("+" if k % 2 == 0 else "−") + str(comb(n, k)), 32,
                     BLUE_B if k % 2 == 0 else ORANGE).move_to([X[k], Y0 + 0.62, 0])
                 for k in range(n + 1)]

        # ---------------- the subsets by size
        for k in range(n + 1):
            grp = [items[S] for S in subsets if len(S) == k]
            self.play(LaggedStart(*[FadeIn(g) for g in grp], lag_ratio=0.1), FadeIn(heads[k]),
                      run_time=0.55 if k in (0, 4) else 0.8)
        self.hold(0.3)
        rings = [items[S][0] for S in subsets]
        self.play(*[Indicate(r_, color=YELLOW_B, scale_factor=1.5) for r_ in rings], run_time=0.9)

        # ---------------- toggle element 1: each subset slides onto its partner
        ghosts = [items[S].copy() for S in pairs]
        self.play(*[g.animate.shift(pos[tog[S]] - pos[S]) for g, S in zip(ghosts, pairs)],
                  run_time=1.3)
        for g, S in zip(ghosts, pairs):
            T = tog[S]
            check(close(g[0].get_center(), items[T][0].get_center(), 1e-6),
                  "the slid subset sits on its partner")
        self.play(*[g[0].animate.set_fill(col_of(tog[S]), opacity=1.0).scale(1 / 0.85)
                    for g, S in zip(ghosts, pairs)],
                  *[g[q].animate.set_fill(col_of(tog[S])) for g, S in zip(ghosts, pairs)
                    for q in range(1, 4) if q in S], run_time=0.8)
        for g, S in zip(ghosts, pairs):
            T = tog[S]
            check(all(close(g[q].get_center(), items[T][q].get_center(), 1e-6)
                      and (g[q].get_fill_opacity() > 0.5) == (q in T)
                      and close(g[q].width, items[T][q].width, 1e-6) for q in range(4)),
                  "putting 1 in turns the subset into its partner")
        self.play(*[FadeOut(g) for g in ghosts],
                  *[Indicate(items[tog[S]], color=YELLOW_B, scale_factor=1.12) for S in pairs],
                  run_time=0.7)

        # each row: one even, one odd
        links = VGroup(*[DoubleArrow(items[S].get_right() + RIGHT * 0.1,
                                     items[tog[S]].get_left() + LEFT * 0.1, buff=0,
                                     color=YELLOW_B, stroke_width=2.5, tip_length=0.13,
                                     max_tip_length_to_length_ratio=0.2) for S in pairs])
        self.play(Create(links), run_time=0.9)
        ev = tag("even:  1 + 6 + 1  =  8", 28, BLUE_B)
        od = tag("odd:  4 + 4  =  8", 28, ORANGE)
        ev.move_to([-3.4, -1.45, 0])
        od.move_to([3.0, -1.45, 0])
        alt = tag("1 − 4 + 6 − 4 + 1  =  0", 32, YELLOW_B).move_to([0, -2.22, 0])
        self.play(FadeIn(ev), FadeIn(od), run_time=0.8)
        self.play(FadeIn(alt), run_time=0.8)

        allit = VGroup(*items.values())
        check(_inside(allit) and _inside(links), "inside the frame")
        check(min(h.get_bottom()[1] for h in heads) > allit.get_top()[1] + 0.1,
              "headers above the columns")
        check(allit.get_bottom()[1] > max(ev.get_top()[1], od.get_top()[1]) + 0.2,
              "sums below the columns")
        for S in pairs:
            a, b = items[S], items[tog[S]]
            check(close(a.get_center()[1], b.get_center()[1]) and
                  abs(len(tog[S]) - len(S)) == 1, "a pair shares a row")
        _labels_ok(heads + [ev, od, alt], [], "K26 labels")
        _clear_of(heads + [ev, od, alt], [allit, links], "K26 labels")
        cap = _cap(_supline([("∑ (−1)", False), ("k", True), (" C(n, k)  =  0      for n ≥ 1", False)],
                            34))
        self.play(FadeIn(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K31

def _hooks(lam):
    """The diagonal hooks of lam: (corner, arm cells, leg cells) for each
    diagonal cell (i, i)."""
    lc = _conj(lam)
    out = []
    for i in range(len(lam)):
        if lam[i] < i + 1:
            break
        out.append(((i, i), [(i, j) for j in range(i + 1, lam[i])],
                    [(r, i) for r in range(i + 1, lc[i])]))
    return out


_HOOKCOL = [BLUE_C, TEAL_C, ORANGE, YELLOW_C]


class K31_SelfConjugatePartitions(Board):
    """A partition equal to its conjugate has a dot diagram symmetric in
    its diagonal: turned over, it lands on itself. Peel it into hooks
    around the diagonal: the i-th hook is the diagonal dot (i, i), the dots
    right of it in row i and the dots below it in column i. By the
    symmetry the arm and the leg are equally long, so a hook straightened
    out is a + 1 + a dots, odd; and the arms get strictly shorter inwards,
    so the hooks are all different. Conversely, distinct odd rows bent at
    their middles nest into a symmetric diagram. Shown for
    5 + 4 + 4 + 3 + 1 = 9 + 5 + 3, then all three symmetric diagrams of 12."""

    def construct(self):
        # ---------------- the mathematics, checked
        for n in range(1, 26):
            sc = [l for l in _partitions(n) if _conj(l) == l]
            do = sorted(tuple(sorted((len(a) + len(b) + 1 for _, a, b in _hooks(l)),
                                     reverse=True)) for l in sc)
            want = sorted(l for l in _partitions(n) if all(x % 2 for x in l)
                          and len(set(l)) == len(l))
            check(do == want, f"n = {n}: hooks of symmetric diagrams = distinct odd parts")
            for l in sc:
                hk = _hooks(l)
                check(all(len(a) == len(b) for _, a, b in hk), "arm = leg")
                check(sorted(c for h in hk for c in [h[0]] + h[1] + h[2]) == sorted(_cells(l)),
                      "the hooks cover the diagram once")
        lam = (5, 4, 4, 3, 1)
        check(_conj(lam) == lam and sum(lam) == 17, "5+4+4+3+1 is self-conjugate")
        hk = _hooks(lam)
        sizes = [1 + len(a) + len(b) for _, a, b in hk]
        check(sizes == [9, 5, 3], "hooks 9, 5, 3")

        s, rd = 0.56, 0.14
        P0 = np.array([-5.85, 3.2, 0.0])

        def at(i, j):
            return P0 + np.array([j * s, -i * s, 0.0])

        dots = {c: Dot(at(*c), radius=rd, color=GREY_A) for c in _cells(lam)}
        diag = guide(at(-0.7, -0.7), at(4.6, 4.6), color=GREY_B)
        allg = VGroup(*dots.values())
        self.play(LaggedStart(*[FadeIn(dots[c]) for c in _cells(lam)], lag_ratio=0.04),
                  run_time=1.3)
        self.play(Create(diag), run_time=0.6)
        # turned over its diagonal it lands on itself
        before = _centres(allg)
        self.play(_flip(allg, P0, np.array([1.0, -1.0, 0.0])), run_time=1.8)
        check(_centres(allg) == before, "the symmetric diagram lands on itself")

        # ---------------- its hooks
        hooks = []
        for q, (cn, arm, leg) in enumerate(hk):
            col = _HOOKCOL[q]
            corner = Dot(at(*cn), radius=rd, color=col).set_stroke(WHITE, 2.5)
            a = VGroup(*[Dot(at(*c), radius=rd, color=col) for c in arm])
            l_ = VGroup(*[Dot(at(*c), radius=rd, color=col) for c in leg])
            hooks.append(VGroup(corner, a, l_))
        self.play(FadeOut(allg), *[FadeIn(h) for h in hooks], run_time=1.0)
        self.hold(0.4)

        # ---------------- peel them off, outermost first, and straighten
        Xc, Ys = 1.6, [3.1, 2.4, 1.7]
        labs, rows = [], []
        for q, h0 in enumerate(hooks):
            T = np.array([Xc, Ys[q], 0.0])
            c0 = at(*hk[q][0])
            h = h0.copy()
            rows.append(h)
            self.add(h)
            self.play(h.animate.shift(T - c0), h0.animate.set_opacity(0.55), run_time=0.9)
            self.play(Rotate(h[2], angle=-PI / 2, about_point=T), run_time=0.8)
            a = len(hk[q][1])
            want = [T + np.array([k * s, 0, 0]) for k in range(-a, a + 1)]
            got = sorted((round(float(d.get_center()[0]), 5), round(float(d.get_center()[1]), 5))
                         for d in [h[0]] + list(h[1]) + list(h[2]))
            check(got == sorted((round(float(w[0]), 5), round(float(w[1]), 5)) for w in want),
                  f"hook {q + 1} straightens into a row of {2 * a + 1} centred on its corner")
            lb = tag(str(sizes[q]), 30, _HOOKCOL[q]).move_to(T + RIGHT * (4 * s + 0.75))
            labs.append(lb)
            self.play(FadeIn(lb), run_time=0.4)
        diag2 = guide([Xc, Ys[0] + 0.35, 0], [Xc, Ys[-1] - 0.35, 0], color=GREY_B)
        tot = _crow([("17", WHITE), ("=", WHITE), ("9", BLUE_B), ("+", WHITE), ("5", TEAL_B),
                     ("+", WHITE), ("3", ORANGE)], 32)
        tot.move_to([Xc, 0.95, 0])
        self.play(Create(diag2), FadeIn(tot), run_time=0.8)
        self.hold(0.5)

        # ---------------- every symmetric diagram of 12
        sc12 = [l for l in _partitions(12) if _conj(l) == l]
        check(sc12 == [(6, 2, 1, 1, 1, 1), (5, 3, 2, 1, 1), (4, 4, 2, 2)], "three of 12")
        s2, r2 = 0.25, 0.085
        gx = [-4.3, -0.4, 3.3]
        gy = -0.35
        gal, glabs = [], []
        for x, l in zip(gx, sc12):
            g = VGroup()
            for q, (cn, arm, leg) in enumerate(_hooks(l)):
                for c in [cn] + arm + leg:
                    d = Dot(np.array([x + c[1] * s2, gy - c[0] * s2, 0.0]), radius=r2,
                            color=_HOOKCOL[q])
                    if c == cn:
                        d.set_stroke(WHITE, 1.5)
                    g.add(d)
            gal.append(g)
            hs = [1 + len(a) + len(b) for _, a, b in _hooks(l)]
            t = _crow(sum([[(str(v), _HOOKCOL[q]), ("+", WHITE)] for q, v in enumerate(hs)],
                          [])[:-1], 28, 0.14)
            t.move_to([x + 2.5 * s2 + 1.3 + t.width / 2, gy - 2.5 * s2, 0])
            glabs.append(t)
        n12 = tag("12:", 30, GREY_A).move_to([-5.75, gy - 2.5 * s2, 0])
        self.play(FadeIn(n12), LaggedStart(*[FadeIn(VGroup(g, t)) for g, t in zip(gal, glabs)],
                                           lag_ratio=0.3), run_time=2.0)

        top = VGroup(*hooks, *rows, diag2)
        check(_inside(top) and _inside(VGroup(*gal)) and _inside(diag), "inside the frame")
        check(VGroup(*gal).get_top()[1] < tot.get_bottom()[1] - 0.25, "gallery below the sum")
        _labels_ok(labs + [tot, n12] + glabs, [(diag2.get_start(), diag2.get_end())],
                   "K31 labels")
        _clear_of(labs + [tot, n12] + glabs, list(hooks) + rows + gal, "K31 labels")
        check(max(h.get_right()[0] for h in rows) < min(l.get_left()[0] for l in labs) - 0.15,
              "row labels right of the rows")
        check(VGroup(*hooks, diag).get_right()[0] < VGroup(*rows).get_left()[0] - 0.4,
              "the diagram clear of the rows")
        cap = caption("symmetric dot diagrams of n   =   partitions of n into distinct odd parts",
                      30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)
