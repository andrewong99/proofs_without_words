# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2a.py — proofs without words, 2D (manim): series and sums.
#
#     B25 telescoping strips              B35 squares of reciprocals stay under 2
#     B21 thirds in a triangle            B22 halves from thirds
#     B24 the sum of k over 2^k           B46 the area of the Koch snowflake
#     B36 Cassini's identity              B19 alternating sum of squares
#     B26 cubes from consecutive odd numbers
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Infinite processes are drawn exactly for a few stages and then "…"; the
# caption states the limit. Every tiling, landing and count is checked with
# check(...) before or right after it is drawn, so a wrong construction fails
# the render instead of rendering quietly into a wrong picture.

from fractions import Fraction


# ---------------------------------------------------------------- helpers

def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _box(m):
    return m.get_critical_point(DL), m.get_critical_point(UR)


def _safe(*mobs, tol=0.02):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for i, m in enumerate(mobs):
        lo, hi = _box(m)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area (item {i}): [{lo[0]:.2f},{hi[0]:.2f}] x "
              f"[{lo[1]:.2f},{hi[1]:.2f}]")


def _sep(a, b, gap=0.04):
    a0, a1 = _box(a)
    b0, b1 = _box(b)
    return (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
            or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])


def _apart(*mobs, gap=0.04):
    """Fail the render if any two mobjects' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            check(_sep(mobs[i], mobs[j], gap), f"items {i} and {j} do not overlap")


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
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if m.has_points() or m.submobjects:
            _safe(m)
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_sep(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


def _dim(p, q, label, off, color=GREY_A, size=24, gap=0.14):
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


def _eq_row(left, right, x_eq, y, size=26, color=WHITE):
    """'left  =  right' with the '=' sign at x_eq (rows of a readout align)."""
    eq = tag("=", size, color).move_to([x_eq, y, 0.0])
    lt = tag(left, size, color)
    lt.move_to([x_eq - 0.3 - lt.width / 2, y, 0.0])
    rt = tag(right, size, color)
    rt.move_to([x_eq + 0.3 + rt.width / 2, y, 0.0])
    return VGroup(lt, eq, rt)


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


# ====================================================================== B25

class B25_TelescopingStrips(Board):
    """The unit square is cut at 1/2, 1/3, 1/4, …; the strip between the
    cuts 1/(k+1) and 1/k has width 1/k − 1/(k+1). Why that is 1/k · 1/(k+1):
    the triangle under the line from the top-left corner to the cut 1/k
    (height 1, base 1/k) is shrunk by a factor s towards its right end
    until its top touches the diagonal. Then its height s equals its
    distance from the left side, and its base s · 1/k reaches from there to
    1/k: s + s/k = 1/k, so s = 1/(k+1). The shrunk triangle stands exactly
    between the cuts 1/(k+1) and 1/k, and its base — the strip's width — is
    1/k · 1/(k+1). The cuts run down to 0, so the strips fill the square:
        1/(1·2) + 1/(2·3) + 1/(3·4) + … = 1.
    Six strips are built this way, a few more are added, then "…"."""

    def construct(self):
        S = 5.6
        O = np.array([-6.05, -2.45, 0.0])

        def P(x, y):
            return O + S * np.array([float(x), float(y), 0.0])

        K, KMORE = 6, 14
        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D, RED_D, PURPLE_B, GOLD_D,
                MAROON_B]
        tcol = [BLUE_B, TEAL_B, ORANGE, GREEN_B, RED_B, PURPLE_A]

        # exact arithmetic: the shrunk triangle lands between the cuts
        for k in range(1, KMORE + 1):
            a, s = Fraction(1, k), Fraction(1, k + 1)
            top = (a + s * (0 - a), s * 1)          # image of the corner (0, 1)
            foot = (a + s * (0 - a), 0)             # image of the corner (0, 0)
            check(top == (s, s), f"k={k}: the shrunk top lands on the diagonal")
            check(foot == (s, 0), f"k={k}: its left end is the cut 1/(k+1)")
            check(a - s == a * s, f"k={k}: 1/k - 1/(k+1) = 1/k * 1/(k+1)")
        check(sum(Fraction(1, k * (k + 1)) for k in range(1, KMORE + 1))
              == 1 - Fraction(1, KMORE + 1), "the strips and the rest make 1")

        square = Polygon(P(0, 0), P(1, 0), P(1, 1), P(0, 1),
                         stroke_color=WHITE, stroke_width=3)
        diag = DashedLine(P(0, 0), P(1, 1), color=GREY_A, stroke_width=2.5,
                          dash_length=0.1)
        side = tag("1", 26).move_to(P(0, 0.5) + LEFT * 0.3)
        ylab = O[1] - 0.3
        blabs = {1: tag("1", 24), 2: tag("½", 24), 3: tag("⅓", 24),
                 4: tag("¼", 24)}
        zero = tag("0", 24).move_to([O[0], ylab, 0])
        for k, t in blabs.items():
            t.move_to([P(Fraction(1, k), 0)[0], ylab, 0])
        _apart(zero, *blabs.values(), gap=0.05)
        _safe(side, zero, *blabs.values())

        self.play(Create(square), FadeIn(side), FadeIn(zero), FadeIn(blabs[1]),
                  run_time=1.0)
        self.play(Create(diag), run_time=0.6)

        # readout on the right: one row per strip, '=' signs aligned
        x_eq, y0, dy = 3.05, 3.0, 0.62
        rows = []
        names = ["1", "½", "⅓", "¼", "1/5", "1/6", "1/7"]
        for k in range(1, K + 1):
            rows.append(_eq_row(f"{names[k - 1]} − {names[k]}",
                                f"{names[k - 1]} · {names[k]}",
                                x_eq, y0 - (k - 1) * dy, 28, tcol[k - 1]))
        _safe(*rows)
        check(rows[0][0].get_left()[0] > P(1, 0)[0] + 0.6,
              "readout clear of the square")

        for k in range(1, K + 1):
            a, s = 1.0 / k, 1.0 / (k + 1)
            fast = k >= 4
            tri = mk([P(0, 1), P(0, 0), P(a, 0)], GREY_B, 0.42,
                     stroke_color=WHITE, stroke_width=2.5)
            self.play(FadeIn(tri), run_time=0.35 if fast else 0.5)
            self.play(tri.animate.scale(s, about_point=P(a, 0)),
                      run_time=0.8 if fast else 1.3)
            check(_same_poly(_verts(tri), [P(s, s), P(s, 0), P(a, 0)]),
                  f"k={k}: the shrunk triangle has its top on the diagonal "
                  f"and its base between the cuts")
            # on the diagonal: height = distance from the left side
            level = DashedLine(P(0, s), P(s, s), color=GREY_A, stroke_width=2,
                               dash_length=0.07)
            self.play(Create(level), run_time=0.25 if fast else 0.4)
            strip = mk([P(*q) for q in _rect(s, 0, a - s, 1)], cols[k - 1],
                       FILL, stroke_width=1.5)
            cut = Line(P(s, 0), P(s, 1), color=WHITE, stroke_width=2)
            self.add(strip)
            self.bring_to_back(strip)
            self.bring_to_front(square, diag, tri)
            anims = [FadeIn(strip), Create(cut), FadeIn(rows[k - 1])]
            if k + 1 in blabs:
                anims.append(FadeIn(blabs[k + 1]))
            self.play(*anims, run_time=0.45 if fast else 0.7)
            self.play(FadeOut(tri), FadeOut(level), run_time=0.25 if fast else 0.35)
            if k <= 2:
                self.hold(0.3)

        # more strips the same way, then the rest of the square is "…"
        extra = VGroup()
        for k in range(K + 1, KMORE + 1):
            a, s = 1.0 / k, 1.0 / (k + 1)
            extra.add(mk([P(*q) for q in _rect(s, 0, a - s, 1)],
                         cols[(k - 1) % len(cols)], FILL, stroke_width=0.8))
        dots_v = tag("⋮", 30).move_to(rows[-1].get_center() + DOWN * dy * 1.1)
        rest = 1.0 / (KMORE + 1)
        dots = tag("…", 22).move_to(P(rest / 2, 0.62))
        check(dots.width < rest * S - 0.04, "the dots fit in what is left")
        _clear_of_segment(dots, P(0, 0), P(1, 1), what="the dots")
        self.add(extra)
        self.bring_to_back(extra)
        self.bring_to_front(square, diag)
        extra.set_opacity(0)
        self.play(LaggedStart(*[m.animate.set_fill(opacity=FILL)
                                .set_stroke(opacity=1) for m in extra],
                              lag_ratio=0.3),
                  FadeIn(dots_v), run_time=1.3)
        self.play(FadeIn(dots), run_time=0.4)
        self.hold(0.4)

        # the strips fill the square
        self.play(square.animate.set_stroke(YELLOW_B, width=6), run_time=0.7)
        cap = caption("1/(1·2) + 1/(2·3) + 1/(3·4) + …   =   1", 34)
        self.play(Write(cap), run_time=1.3)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B35

class B35_ReciprocalSquares(Board):
    """The squares of side 1, 1/2, 1/3, … grouped by powers of two: group j
    holds the 2^j squares of side 1/k for k = 2^j, …, 2^(j+1) − 1. Each is
    no larger than a cell of side 1/2^j, and 2^j such cells stack into a
    strip 1/2^j wide and 1 high. So group j fits in that strip, with room
    to spare, and the strips 1, ½, ¼, … — each half of what is left — fill
    a 2 × 1 rectangle:
        1 + 1/2² + 1/3² + … < 1 + ½ + ¼ + … = 2.
    Drawn exactly for the groups j = 0 … 4 (k up to 31), then "…"."""

    def construct(self):
        u = 5.4
        O = np.array([-5.4, -2.3, 0.0])

        def P(x, y):
            return O + u * np.array([float(x), float(y), 0.0])

        J = 5
        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D, RED_D]

        # exact bookkeeping: every square inside its cell, cells fill strips
        for j in range(J):
            g = Fraction(1, 2 ** j)
            ks = list(range(2 ** j, 2 ** (j + 1)))
            check(len(ks) * g == 1, f"group {j}: 2^j cells of side 1/2^j stack to 1")
            check(all(Fraction(1, k) <= g for k in ks),
                  f"group {j}: every square fits in its cell")
            cover = sum(Fraction(1, k * k) for k in ks)
            check(cover == g if j == 0 else cover < g,
                  f"group {j}: the squares cover their strip (j = 0) or "
                  f"less than it (j >= 1)")
            check(2 - Fraction(2, 2 ** j) + g == 2 - g,
                  f"strip {j} is half of what is left of the 2 x 1 rectangle")

        def x_of(j):                       # left edge of strip j
            return 2.0 - 2.0 ** (1 - j)

        frame = Polygon(P(0, 0), P(2, 0), P(2, 1), P(0, 1),
                        stroke_color=YELLOW_B, stroke_width=5)
        ghost_frame = DashedVMobject(
            Polygon(P(0, 0), P(2, 0), P(2, 1), P(0, 1), stroke_color=GREY_B,
                    stroke_width=2), num_dashes=90)
        self.play(Create(ghost_frame), run_time=0.9)

        names = {1: "1", 2: "1/2²", 3: "1/3²", 4: "1/4²", 5: "1/5²",
                 6: "1/6²", 7: "1/7²"}
        sizes = {1: 44, 2: 32, 3: 26, 4: 22, 5: 20, 6: 18, 7: 17}
        wnames = {0: "1", 1: "½", 2: "¼", 3: "⅛"}
        dims = []
        for j in range(J):
            g = 2.0 ** (-j)
            x0 = x_of(j)
            cells = VGroup(*[Polygon(*[P(*q) for q in _rect(x0, i * g, g, g)],
                                     stroke_color=GREY_B, stroke_width=1.6
                                     if j < 4 else 1.0)
                             for i in range(2 ** j)])
            squares = VGroup()
            for i in range(2 ** j):
                k = 2 ** j + i
                sq = mk([P(*q) for q in _rect(x0, i * g, 1.0 / k, 1.0 / k)],
                        cols[j], FILL, stroke_width=1.5 if j < 3 else 1.0)
                check(close(sq.get_corner(DL), P(x0, i * g))
                      and sq.get_corner(UR)[0] <= P(x0 + g, 0)[0] + 1e-9
                      and sq.get_corner(UR)[1] <= P(0, (i + 1) * g)[1] + 1e-9,
                      f"square 1/{k} drawn inside its cell")
                squares.add(sq)
            labs = VGroup()
            for i in range(2 ** j):
                k = 2 ** j + i
                if k in names:
                    t = tag(names[k], sizes[k]).move_to(
                        P(x0 + 0.5 / k, i * g + 0.5 / k))
                    check(t.width < u / k - 0.08 and t.height < u / k - 0.08,
                          f"label of 1/{k}² inside its square")
                    labs.add(t)
            if j in wnames:
                d = _dim(P(x0, 0), P(x0 + g, 0), wnames[j], DOWN * 0.13,
                         GREY_A, 24, gap=0.08)
                dims.append(d)
            else:
                d = VGroup()
            self.play(Create(cells), run_time=0.7 if j < 3 else 0.5)
            self.play(LaggedStart(*[FadeIn(q) for q in squares],
                                  lag_ratio=0.15 if j < 3 else 0.05),
                      FadeIn(labs), FadeIn(d),
                      run_time=1.0 if j < 3 else 0.8)
            if j <= 2:
                self.hold(0.4)

        rest = 2.0 ** (-(J - 1))               # what is left: [2 - rest, 2]
        dots = tag("…", 22).move_to(P(2 - rest / 2, 17 / 32))   # mid-cell
        check(dots.width < rest * u - 0.04, "the dots fit in what is left")
        self.play(FadeIn(dots), run_time=0.4)
        self.hold(0.4)

        d_w = _dim(P(0, 1), P(2, 1), "2", UP * 0.16, YELLOW_B, 30, gap=0.1)
        d_h = _dim(P(0, 0), P(0, 1), "1", LEFT * 0.16, YELLOW_B, 30, gap=0.1)
        _safe(d_w, d_h, *dims)
        _apart(*[d[1] for d in dims])
        self.play(ReplacementTransform(ghost_frame, frame), FadeIn(d_w),
                  FadeIn(d_h), run_time=1.0)
        cap = caption("1 + 1/2² + 1/3² + 1/4² + …   <   1 + ½ + ¼ + ⅛ + …   =   2",
                      32)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B21

def _mid(p, q):
    return (np.asarray(p, float) + np.asarray(q, float)) / 2


def _inside_tri(p, tri, margin=0.0):
    """p inside the triangle, at least `margin` from every side."""
    A, B, C = [np.asarray(q, float)[:2] for q in tri]
    p = np.asarray(p, float)[:2]
    s = np.sign((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
    for U, V in ((A, B), (B, C), (C, A)):
        e = V - U
        cr = s * (e[0] * (p[1] - U[1]) - e[1] * (p[0] - U[0]))
        if cr / np.linalg.norm(e) < margin:
            return False
    return True


def _label_in_tri(t, tri, margin=0.03):
    return all(_inside_tri(t.get_corner(c), tri, margin) for c in (UL, UR, DL, DR))


def _rot_pt(p, c, ang):
    p, c = to3(p), to3(c)
    d = p - c
    ca, sa = np.cos(ang), np.sin(ang)
    return c + np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1], 0.0])


def _matches(polys, targets, tol=1e-6):
    """A one-to-one matching of the polygons onto the targets, each landing
    exactly (same vertices, same cyclic order)."""
    if len(polys) != len(targets):
        return False
    left = list(range(len(targets)))
    for P_ in polys:
        hit = next((i for i in left if _same_poly(P_, targets[i], tol)), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


class B21_TriangleThirds(Board):
    """Cut an equilateral triangle into four by its midlines; give the three
    corner quarters one colour each and repeat inside the middle quarter,
    and so on. The middle quarters all have the same centre, so a third of
    a turn about it carries the picture onto itself, every colour onto the
    next: the three colours are congruent. Together they fill the triangle,
    the shrinking middle aside, so each colour is a third of it:
        ¼ + 1/16 + 1/64 + … = ⅓.
    Six stages drawn, then "…"."""

    def construct(self):
        side = 7.0
        h = side * np.sqrt(3) / 2
        A = np.array([-side / 2, -2.78, 0.0])
        B = np.array([side / 2, -2.78, 0.0])
        C = np.array([0.0, -2.78 + h, 0.0])
        G = (A + B + C) / 3
        total = abs(area([A, B, C]))
        cols = [BLUE_D, ORANGE, GREEN_D]
        STAGES = 6

        def colour_of(v):                  # by the corner's direction from G
            ang = np.degrees(np.arctan2(v[1] - G[1], v[0] - G[0]))
            return int(((ang - 60.0) % 360.0) // 120.0)

        outline = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        _safe(outline)
        self.play(Create(outline), run_time=1.0)

        T = [A, B, C]
        pieces = {0: [], 1: [], 2: []}            # colour -> list of triangles
        mobs = {0: VGroup(), 1: VGroup(), 2: VGroup()}
        lab_txt = {1: ("¼", 44, 1.0), 2: ("1/16", 26, 1.0), 3: ("1/64", 24, 0.52)}
        for st in range(1, STAGES + 1):
            M = [_mid(T[0], T[1]), _mid(T[1], T[2]), _mid(T[2], T[0])]
            corners = [[T[0], M[0], M[2]], [T[1], M[1], M[0]], [T[2], M[2], M[1]]]
            centre = [M[0], M[1], M[2]]
            check(close(sum(centre) / 3, G), f"stage {st}: the middle keeps the centre G")
            got = sorted(colour_of(c[0]) for c in corners)
            check(got == [0, 1, 2], f"stage {st}: one corner of each colour")
            for c in corners:
                check(abs(abs(area(c)) - total / 4 ** st) < 1e-9,
                      f"stage {st}: each corner is 1/4^{st} of the triangle")
            cuts = Polygon(*centre, stroke_color=WHITE,
                           stroke_width=2.5 if st < 4 else 1.5)
            fast = st >= 4
            self.play(Create(cuts), run_time=0.7 if st == 1 else (0.5 if not fast else 0.3))
            new = []
            for c in corners:
                ci = colour_of(c[0])
                m = mk(c, cols[ci], FILL, stroke_width=2 if st < 4 else 1)
                pieces[ci].append(c)
                mobs[ci].add(m)
                new.append(m)
            lab_anim = []
            if st in lab_txt:
                txt, sz, sc = lab_txt[st]
                c0 = [c for c in corners if colour_of(c[0]) == 0][0]
                lab = tag(txt, sz).scale(sc).move_to((c0[0] + c0[1] + c0[2]) / 3)
                check(_label_in_tri(lab, c0), f"label {txt} inside its quarter")
                lab_anim.append(FadeIn(lab))
            self.add(*new)
            self.bring_to_front(cuts)
            for m in new:
                m.set_opacity(0)
            self.play(*[m.animate.set_fill(opacity=FILL).set_stroke(opacity=1)
                        for m in new], *lab_anim,
                      run_time=0.8 if st == 1 else (0.6 if not fast else 0.35))
            if st <= 2:
                self.hold(0.4)
            T = centre
        self.bring_to_front(outline)
        self.hold(0.5)

        # a third of a turn about G carries each colour onto the next
        for ci in (0, 1):
            src = mobs[ci].copy()
            src.set_fill(opacity=0.0).set_stroke(YELLOW_B, width=3.5)
            self.add(src)
            self.play(Rotate(src, angle=TAU / 3, about_point=G), run_time=1.6)
            dst = (ci + 1) % 3
            turned = [[_rot_pt(v, G, TAU / 3) for v in c] for c in pieces[ci]]
            check(_matches(turned, pieces[dst]),
                  f"colour {ci} turned a third lands on colour {dst}")
            check(_matches([_verts(m) for m in src], pieces[dst]),
                  f"the turned outline sits exactly on colour {dst}")
            self.play(Indicate(mobs[dst], color=YELLOW_B, scale_factor=1.0),
                      run_time=0.6)
            self.play(FadeOut(src), run_time=0.3)

        cap = caption("¼ + 1/16 + 1/64 + …   =   ⅓", 36)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B22

class B22_StripHalves(Board):
    """A strip, three times as long as it is high, is cut into thirds: shade
    one, leave one, and repeat inside the middle one — a square, which is
    cut into thirds across, whose middle third is a strip like the first,
    and so on. Each shaded piece is a third of the region it came from, so
    the shaded pieces are ⅓, 1/9, 1/27, … of the strip. Every middle piece
    has the same centre, and a half-turn about it carries the shaded
    pieces exactly onto the unshaded ones — shown as its two halves, a turn
    over the strip's long midline and then over its short one, which keeps
    the moving copy on the strip. The two match at every stage and
    together fill the strip, the shrinking middle aside. So
        ⅓ + 1/9 + 1/27 + … = ½.
    Seven stages drawn, then "…"."""

    def construct(self):
        u = 4.0
        O = np.array([-6.0, -1.62, 0.0])

        def P(x, y):
            return O + u * np.array([float(x), float(y), 0.0])

        STAGES = 7
        ctr = P(1.5, 0.5)
        strip = Polygon(P(0, 0), P(3, 0), P(3, 1), P(0, 1),
                        stroke_color=WHITE, stroke_width=4)
        _safe(strip)
        self.play(Create(strip), run_time=1.0)

        x0, y0, w, hh = Fraction(0), Fraction(0), Fraction(3), Fraction(1)
        shaded, unshaded = [], []
        sh_mobs, un_mobs, labs = VGroup(), VGroup(), []
        lab_txt = {1: ("⅓", 56), 2: ("1/9", 36), 3: ("1/27", 26), 4: ("1/81", 17)}
        for st in range(1, STAGES + 1):
            if w > hh:            # long: cut across its length
                t = w / 3
                S_ = (x0, y0, t, hh)
                U_ = (x0 + 2 * t, y0, t, hh)
                nxt = (x0 + t, y0, t, hh)
                cut_pts = [((x0 + t, y0), (x0 + t, y0 + hh)),
                           ((x0 + 2 * t, y0), (x0 + 2 * t, y0 + hh))]
            else:                 # a square: cut into thirds the other way
                t = hh / 3
                S_ = (x0, y0 + 2 * t, w, t)
                U_ = (x0, y0, w, t)
                nxt = (x0, y0 + t, w, t)
                cut_pts = [((x0, y0 + t), (x0 + w, y0 + t)),
                           ((x0, y0 + 2 * t), (x0 + w, y0 + 2 * t))]
            check(S_[2] * S_[3] == Fraction(3, 3 ** st),
                  f"stage {st}: the shaded piece is 1/3^{st} of the strip")
            check(S_[2] * S_[3] == U_[2] * U_[3], f"stage {st}: shaded = unshaded")
            # the half-turn about the common centre swaps the two pieces
            hs = (3 - S_[0] - S_[2], 1 - S_[1] - S_[3], S_[2], S_[3])
            check(hs == U_, f"stage {st}: the half-turn takes shaded onto unshaded")
            check(nxt[0] + nxt[2] / 2 == Fraction(3, 2) and nxt[1] + nxt[3] / 2
                  == Fraction(1, 2), f"stage {st}: the middle keeps the centre")
            shaded.append(S_)
            unshaded.append(U_)
            fast = st >= 4
            sw = 2.5 if st < 4 else 1.5
            cuts = VGroup(*[Line(P(*a), P(*b), color=WHITE, stroke_width=sw)
                            for a, b in cut_pts])
            sm = mk([P(*q) for q in _rect(*S_)], BLUE_D, FILL,
                    stroke_width=sw)
            um = mk([P(*q) for q in _rect(*U_)], GREY_D, 0.35,
                    stroke_width=sw)
            sh_mobs.add(sm)
            un_mobs.add(um)
            self.play(Create(cuts), run_time=0.6 if not fast else 0.3)
            anims = [FadeIn(sm), FadeIn(um)]
            if st in lab_txt:
                txt, sz = lab_txt[st]
                lab = tag(txt, sz).move_to(sm)
                check(lab.width < S_[2] * u - 0.1 and lab.height < S_[3] * u - 0.08,
                      f"label {txt} inside its piece")
                anims.append(FadeIn(lab))
                labs.append(lab)
            self.add(sm, um)
            self.bring_to_front(cuts, strip)
            self.play(*anims, run_time=0.7 if not fast else 0.35)
            if st <= 3:
                self.hold(0.3)
            x0, y0, w, hh = nxt
        self.hold(0.4)

        # the half-turn about the centre, as two turns over the midlines:
        # the shaded pieces land exactly on the unshaded ones
        turned = sh_mobs.copy().set_fill(ORANGE, opacity=0.6)
        self.add(turned)
        self.bring_to_front(strip, *labs)
        self.play(FadeIn(turned), run_time=0.4)
        self.play(Rotate(turned, angle=PI, axis=RIGHT, about_point=ctr),
                  run_time=1.5)
        self.play(Rotate(turned, angle=PI, axis=UP, about_point=ctr),
                  run_time=1.5)
        for m, U_ in zip(turned, unshaded):
            check(_same_poly(_verts(m), [P(*q) for q in _rect(*U_)]),
                  "each shaded piece, turned, covers its unshaded partner")
        for m, S_ in zip(turned, shaded):     # = the half-turn about the centre
            check(_same_poly(_verts(m), [2 * ctr - P(*q) for q in _rect(*S_)]),
                  "the two turns make the half-turn about the centre")
        self.play(turned.animate.set_fill(opacity=0.85), run_time=0.4)
        self.remove(*un_mobs)
        self.bring_to_front(strip)
        self.hold(0.5)
        cap = caption("⅓ + 1/9 + 1/27 + 1/81 + …   =   ½", 36)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B24

class B24_SumKOverPowersOfTwo(Board):
    """Row k of a staircase holds k pieces, each 1 wide and 1/2^k high:
    k/2^k in all. Rows 1, 2, 3, … are stacked downwards, each half as high
    as the one above and one piece longer. Read by columns instead: column
    j holds one piece from every row k ≥ j, so it is 1/2^j + 1/2^(j+1) + …
    high, twice its top piece: 1/2^(j−1). The columns are 1, ½, ¼, …, and
    the columns from the second on, stacked, rebuild the first. So
        ½ + 2/4 + 3/8 + 4/16 + … = 1 + ½ + ¼ + … = 2.
    Rows 1–6 drawn; the rows below them (7, 8, …) are the thin band at the
    bottom, 1/64 high, which every column carries along. Width and height
    use different screen scales (areas compare all the same)."""

    def construct(self):
        w, H = 1.85, 5.4                      # screen size of a 1 x 1 unit
        x0, yf = -6.2, -2.42                  # left edge, floor
        N = 6

        def P(x, y):
            return np.array([x0 + w * float(x), yf + H * float(y), 0.0])

        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D, RED_D, PURPLE_B]
        # exact bookkeeping
        for j in range(1, N + 1):
            hj = sum(Fraction(1, 2 ** k) for k in range(j, N + 1)) + Fraction(1, 2 ** N)
            check(hj == Fraction(1, 2 ** (j - 1)),
                  f"column {j}: rows j..6 and the band make 1/2^(j-1)")
        # rows 1-6, the band (rows 7, 8, … under columns 1-6) and the later
        # columns 7, 8, … (1/64 + 1/128 + … = 1/32) make the whole sum, 2
        whole = (sum(Fraction(k, 2 ** k) for k in range(1, N + 1))
                 + N * Fraction(1, 2 ** N) + Fraction(1, 2 ** (N - 1)))
        check(whole == 2, "rows + band + later columns = 2")

        def piece(j, lo, hi, color, op=FILL, sw=1.5):
            return mk([P(*q) for q in _rect(j - 1, lo, 1, hi - lo)], color, op,
                      stroke_width=sw)

        # pieces[(k, j)]: row k, column j (k >= j); band[j]: rows 7, 8, …
        pieces = {}
        rows = []
        for k in range(1, N + 1):
            lo, hi = 1.0 / 2 ** k, 1.0 / 2 ** (k - 1)
            row = VGroup()
            for j in range(1, k + 1):
                pieces[(k, j)] = piece(j, lo, hi, cols[k - 1],
                                       sw=1.5 if k < 5 else 1.0)
                row.add(pieces[(k, j)])
            rows.append(row)
        band = {j: piece(j, 0.0, 1.0 / 2 ** N, GREY_C, 0.55, sw=0.6)
                for j in range(1, N + 1)}
        floor = Line(P(0, 0) + LEFT * 0.1, P(N, 0) + RIGHT * 0.75, color=GREY_B,
                     stroke_width=2)
        more = tag("…", 26).move_to(P(N, 0) + RIGHT * 0.42 + UP * 0.1)

        rlab_txt = {1: ("1 · ½", 30), 2: ("2 · ¼", 28), 3: ("3 · ⅛", 24),
                    4: ("4 · 1/16", 20)}
        rlabs = {}
        for k, (txt, sz) in rlab_txt.items():
            lo, hi = 1.0 / 2 ** k, 1.0 / 2 ** (k - 1)
            t = tag(txt, sz, WHITE)
            t.move_to(P(k, (lo + hi) / 2) + RIGHT * (0.18 + t.width / 2))
            check(t.height < (hi - lo) * H - 0.06,
                  f"row {k}: its label fits beside the row")
            rlabs[k] = t
        _safe(floor, more, *rlabs.values())
        for k, t in rlabs.items():
            for (kk, j), m in pieces.items():
                check(_sep(t, m, 0.03), f"row label {k} clear of the pieces")

        self.play(Create(floor), run_time=0.5)
        for k in range(1, N + 1):
            anims = [LaggedStart(*[FadeIn(m, shift=DOWN * 0.1) for m in rows[k - 1]],
                                 lag_ratio=0.15)]
            if k in rlabs:
                anims.append(FadeIn(rlabs[k]))
            if k == N:
                anims += [FadeIn(b) for b in band.values()] + [FadeIn(more)]
            self.play(*anims, run_time=0.9 if k <= 3 else 0.6)
            if k <= 2:
                self.hold(0.3)
        self.hold(0.6)

        # read by columns: column j is 1/2^(j-1) high
        clab_txt = ["1", "½", "¼", "⅛", "1/16", "1/32"]
        outlines, clabs = {}, {}
        for j in range(1, N + 1):
            hj = 1.0 / 2 ** (j - 1)
            outlines[j] = Polygon(*[P(*q) for q in _rect(j - 1, 0, 1, hj)],
                                  stroke_color=YELLOW_B, stroke_width=4)
            clabs[j] = tag(clab_txt[j - 1], 26 if j < 5 else 22, YELLOW_B).move_to(
                P(j - 0.5, 0) + DOWN * 0.3)
        _safe(*clabs.values())
        _apart(*clabs.values(), more)
        self.play(*[FadeOut(t) for t in rlabs.values()], run_time=0.4)
        for j in range(1, N + 1):
            self.play(Create(outlines[j]), FadeIn(clabs[j]),
                      run_time=0.55 if j <= 2 else 0.35)
        self.hold(0.6)

        # columns 3, 4, … stacked on column 2 rebuild column 1
        colgrp = {j: VGroup(*[pieces[(k, j)] for k in range(j, N + 1)],
                            band[j], outlines[j]) for j in range(1, N + 1)}
        self.play(FadeOut(VGroup(*[clabs[j] for j in range(2, N + 1)])),
                  FadeOut(more), run_time=0.4)
        for j in range(3, N + 1):
            d = P(-(j - 2), 1 - 1.0 / 2 ** (j - 2)) - P(0, 0)
            self.play(colgrp[j].animate.shift(d), run_time=0.8 if j == 3 else 0.55)
            lo = 1 - 1.0 / 2 ** (j - 2)
            check(close(colgrp[j].get_corner(DL), P(1, lo), 1e-6)
                  and close(colgrp[j].get_corner(UR),
                            P(2, lo + 1.0 / 2 ** (j - 1)), 1e-6),
                  f"column {j} lands on top of the stack")
        top = 1 - 1.0 / 2 ** (N - 1)
        check(abs(top + 1.0 / 2 ** (N - 1) - 1) < 1e-12,
              "the stack leaves exactly the later columns' share at the top")
        self.hold(0.4)

        # the two unit columns, moved to the middle and framed: 2 x 1
        block = VGroup(*[m for (k, j), m in pieces.items()], *band.values())
        dx = RIGHT * -(x0 + w)
        self.play(*[FadeOut(outlines[j]) for j in range(1, N + 1)],
                  FadeOut(clabs[1]), FadeOut(floor), run_time=0.5)
        self.play(block.animate.shift(dx), run_time=1.0)
        check(close(block.get_corner(DL), P(0, 0) + dx, 1e-6)
              and close(block.get_corner(UR), P(2, 1) + dx, 1e-6),
              "column 1 and the stack fill 2 x 1")
        frame = Polygon(*[P(*q) + dx for q in _rect(0, 0, 2, 1)],
                        stroke_color=YELLOW_B, stroke_width=5)
        d_w = _dim(P(0, 0) + dx, P(2, 0) + dx, "2", DOWN * 0.18, YELLOW_B, 30,
                   gap=0.1)
        d_h = _dim(P(0, 0) + dx, P(0, 1) + dx, "1", LEFT * 0.16, YELLOW_B, 30,
                   gap=0.1)
        _safe(frame, d_w, d_h)
        self.play(Create(frame), FadeIn(d_w), FadeIn(d_h), run_time=0.9)
        cap = caption("½ + 2/4 + 3/8 + 4/16 + …   =   1 + ½ + ¼ + …   =   2", 34)
        self.play(Write(cap), run_time=1.3)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B46

def _koch_step(pts):
    """One Koch step on a counter-clockwise polygon: every edge p→q becomes
    p, p+d/3, apex, p+2d/3 with the apex outside. Returns the new polygon
    and the added triangles (one per old edge, in edge order)."""
    out, tris = [], []
    n = len(pts)
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        d = q - p
        a, b = p + d / 3, p + 2 * d / 3
        apex = (p + q) / 2 + (np.sqrt(3) / 6) * np.array([d[1], -d[0], 0.0])
        out += [p, a, apex, b]
        tris.append([a, apex, b])
    return out, tris


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _glide(mob, pivot, target, angle, **kw):
    """A rigid motion on every frame: the piece turns by `angle` about its
    pivot while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _tri_dirs(tri):
    c = sum(to3(v) for v in tri) / 3
    return c, sorted(float(np.arctan2(*(to3(v) - c)[1::-1])) % (TAU / 3) for v in tri)


class B46_KochSnowflake(Board):
    """The Koch snowflake grows from a triangle of area A. Stage 1 puts on
    each side the triangle whose side is the middle third: it is the side's
    middle cell of the 9-cell grid of A, folded out — A/9 each. At every
    later stage each edge gets the same treatment, so every new triangle is
    a ninth of the triangle it grows from, and each triangle has exactly
    four children (on the four edges that replaced its edge): each stage
    adds 4/9 of what the stage before added. With 3 triangles of A/9 at
    stage 1,
        A + 3·A/9 + 12·A/81 + 48·A/729 + … = A + (A/3)(1 + 4/9 + (4/9)² + …)
          = A + (A/3) · 9/5 = 8A/5.
    Four stages drawn, then "…"; the geometric series is summed by its
    formula."""

    def construct(self):
        R = 3.25
        G = np.array([-3.05, 0.38, 0.0])
        V = [G + R * np.array([np.cos(t), np.sin(t), 0.0])
             for t in (np.radians(210), np.radians(330), np.radians(90))]
        A_area = abs(area(V))
        cols = [TEAL_D, ORANGE, GREEN_D, RED_D]
        tcols = [TEAL_B, ORANGE, GREEN_B, RED_B]
        STAGES = 4

        polys, stages = [list(V)], []
        for k in range(1, STAGES + 1):
            pts, tris = _koch_step(polys[-1])
            check(len(tris) == 3 * 4 ** (k - 1), f"stage {k}: 3·4^(k-1) triangles")
            for t in tris:
                check(abs(abs(area(t)) - A_area / 9 ** k) < 1e-9,
                      f"stage {k}: each triangle is A/9^k")
            check(abs(abs(area(pts)) - abs(area(polys[-1])) - len(tris) * A_area / 9 ** k)
                  < 1e-9, f"stage {k}: the new triangles stick out, no overlap")
            polys.append(pts)
            stages.append(tris)
        # every stage-k triangle has four children, each a ninth of it
        for k in range(1, STAGES):
            for i, t in enumerate(stages[k - 1]):
                kids = stages[k][4 * i: 4 * i + 4]
                check(all(abs(abs(area(c)) - abs(area(t)) / 9) < 1e-9 for c in kids),
                      "four children, each a ninth of the parent")
        s_inf = 1 + Fraction(1, 3) / (1 - Fraction(4, 9))
        check(s_inf == Fraction(8, 5), "1 + (1/3)/(1 - 4/9) = 8/5")
        check(abs(abs(area(polys[-1])) / A_area
                  - float(1 + Fraction(1, 3) * sum(Fraction(4, 9) ** j
                                                   for j in range(STAGES)))) < 1e-9,
              "the drawn snowflake is the partial sum")

        tri = mk(V, BLUE_D, FILL, stroke_width=2.5)
        lab_A = tag("A", 44).move_to(G)
        _safe(Polygon(*polys[-1]))
        self.play(FadeIn(tri), FadeIn(lab_A), run_time=1.0)

        # the 9-cell grid of A: the middle cell of each side folds out
        grid = VGroup()
        for i in range(3):
            P0, P1, P2 = V[i], V[(i + 1) % 3], V[(i + 2) % 3]
            for f in (1 / 3, 2 / 3):
                grid.add(Line(P0 + f * (P1 - P0), P0 + f * (P2 - P0),
                              color=GREY_A, stroke_width=1.8))
        self.play(FadeOut(lab_A), Create(grid), run_time=0.9)
        cells = []
        for i in range(3):
            p, q = V[i], V[(i + 1) % 3]
            d = q - p
            inner = (p + q) / 2 - (np.sqrt(3) / 6) * np.array([d[1], -d[0], 0.0])
            cells.append(mk([p + d / 3, inner, p + 2 * d / 3], cols[0], 0.9,
                            stroke_width=2))
        self.play(*[FadeIn(c) for c in cells], run_time=0.6)
        self.play(*[Rotate(c, angle=PI, axis=_unit(V[(i + 1) % 3] - V[i]),
                           about_point=V[i]) for i, c in enumerate(cells)],
                  run_time=1.5)
        for c, t in zip(cells, stages[0]):
            check(_same_poly(_verts(c), t), "the folded cell is the stage-1 triangle")
        lab_9 = tag("A/9", 24).move_to(sum(stages[0][2]) / 3)
        check(_label_in_tri(lab_9, stages[0][2], 0.04), "A/9 inside its triangle")

        # readout
        xl, ys = 0.55, [2.95, 2.2, 1.45, 0.7, -0.05]
        terms = ["A", "+  3 · A/9", "+ 12 · A/81", "+ 48 · A/729", "+ …"]
        tc = [BLUE_B] + tcols
        rows = [tag(t_, 30, c_) for t_, c_ in zip(terms, tc)]
        for r_, y_ in zip(rows, ys):
            r_.move_to([xl + r_.width / 2, y_, 0.0])
        self.play(FadeOut(grid), FadeIn(lab_A), FadeIn(lab_9), FadeIn(rows[0]),
                  FadeIn(rows[1]), run_time=0.8)
        self.hold(0.4)

        # stage 2, and why each stage is 4/9 of the one before
        def stage_mobs(k, sw):
            return VGroup(*[mk(t, cols[k - 1], FILL, stroke_width=sw)
                            for t in stages[k - 1]])

        s2 = stage_mobs(2, 1.5)
        self.play(LaggedStart(*[FadeIn(m) for m in s2], lag_ratio=0.05),
                  run_time=0.9)
        T1 = stages[0][0]                          # the bottom triangle
        a_, apex, b_ = [to3(v) for v in T1]
        tgrid = VGroup()
        for P0, P1, P2 in ((a_, apex, b_), (apex, b_, a_), (b_, a_, apex)):
            for f in (1 / 3, 2 / 3):
                tgrid.add(Line(P0 + f * (P1 - P0), P0 + f * (P2 - P0),
                               color=WHITE, stroke_width=1.5))
        hi = Polygon(*T1, stroke_color=YELLOW_B, stroke_width=4)
        tip = [apex, apex + (a_ - apex) / 3, apex + (b_ - apex) / 3]
        self.play(Create(hi), Create(tgrid), run_time=0.8)
        kids = stages[1][0:4]
        copies = [mk(tip, YELLOW_E, 0.95, stroke_color=YELLOW_B, stroke_width=2)
                  for _ in kids]
        self.add(*copies)
        moves = []
        c0, dirs0 = _tri_dirs(tip)
        for cp, kid in zip(copies, kids):
            c1, dirs1 = _tri_dirs(kid)
            turn = (dirs1[0] - dirs0[0]) % (TAU / 3)
            if turn > TAU / 6:
                turn -= TAU / 3
            moves.append((cp, c0, c1, turn, kid))
        self.play(*[_glide(cp, c0_, c1_, ang) for cp, c0_, c1_, ang, _ in moves],
                  run_time=1.6)
        for cp, _, _, _, kid in moves:
            check(_same_poly(_verts(cp), kid, 1e-6),
                  "a ninth of the parent lands exactly on each child")
        arrows = VGroup()
        ratio_labs = VGroup()
        xa = max(r_.get_right()[0] for r_ in rows[1:4]) + 0.42
        for r in (1, 2):
            ar = Arrow([xa, ys[r] - 0.12, 0], [xa, ys[r + 1] + 0.12, 0], buff=0,
                       color=YELLOW_B, stroke_width=4, tip_length=0.16,
                       max_tip_length_to_length_ratio=0.5)
            arrows.add(ar)
            ratio_labs.add(tag("× 4/9", 26, YELLOW_B).next_to(ar, RIGHT, buff=0.15))
        _safe(*rows, arrows, ratio_labs)
        for r_ in rows[1:4]:
            check(r_.get_right()[0] < xa - 0.25, "rows clear of the ratio arrows")
        self.play(FadeIn(rows[2]), GrowArrow(arrows[0]), FadeIn(ratio_labs[0]),
                  run_time=0.8)
        self.hold(0.6)
        self.play(*[FadeOut(c) for c in copies], FadeOut(tgrid), FadeOut(hi),
                  run_time=0.5)

        # stages 3 and 4
        s3 = stage_mobs(3, 1.0)
        self.play(LaggedStart(*[FadeIn(m) for m in s3], lag_ratio=0.01),
                  FadeIn(rows[3]), GrowArrow(arrows[1]), FadeIn(ratio_labs[1]),
                  run_time=1.0)
        s4 = stage_mobs(4, 0.5)
        self.play(LaggedStart(*[FadeIn(m) for m in s4], lag_ratio=0.002),
                  FadeIn(rows[4]), run_time=0.9)
        outline = Polygon(*polys[-1], stroke_color=YELLOW_B, stroke_width=2.5)
        self.play(Create(outline), run_time=0.9)
        self.hold(0.3)
        cap = caption("A + (A/3)·(1 + 4/9 + (4/9)² + …)   =   A + (A/3)·9/5"
                      "   =   8A/5", 32)
        self.play(Write(cap), run_time=1.5)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B36

def _cells_rect(cells):
    """(x0, y0, w, h) if the cell set is exactly a full rectangle, else None."""
    if not cells:
        return None
    xs = [i for i, _ in cells]
    ys = [j for _, j in cells]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    full = {(i, j) for i in range(x0, x1) for j in range(y0, y1)}
    return (x0, y0, x1 - x0, y1 - y0) if set(cells) == full else None


class B36_Cassini(Board):
    """Cassini's identity, cell by cell. Lay the rectangle F₅ × F₇ = 5 × 13
    (orange) and the square F₆² = 8 × 8 (blue) on a common corner and remove
    their common part from both. What is left is a square F₅² of orange and
    a rectangle F₄ × F₆ = 3 × 8 of blue: the same comparison one step down,
    with the roles — and so the sign — reversed. Slide the blue rest onto
    the orange rest and remove the common part again, and again: removing
    equal parts never changes orange − blue, and in the end one orange cell
    is left over. So
        5·13 − 8² = −(3·8 − 5²) = +(2·5 − 3²) = −(1·3 − 2²) = +(1·2 − 1²) = 1,
    and in general F₍ₙ₋₁₎·F₍ₙ₊₁₎ − Fₙ² = (−1)ⁿ. Everything is whole cells:
    no slivers, no hidden gaps. Drawn for n = 6."""

    def construct(self):
        u = 0.58
        O = np.array([-6.2, -2.3, 0.0])

        def C(i, j):                           # centre of cell (i, j)
            return O + u * np.array([i + 0.5, j + 0.5, 0.0])

        def Pt(x, y):
            return O + u * np.array([float(x), float(y), 0.0])

        F = [0, 1, 1, 2, 3, 5, 8, 13]
        n = 6
        check(F[n - 1] * F[n + 1] - F[n] ** 2 == (-1) ** n, "Cassini for n = 6")
        org = {(i, j) for i in range(13) for j in range(5)}       # 5 x 13
        blu = {(i, j) for i in range(8) for j in range(8)}        # 8 x 8
        moves = [None, (8, -5), (-5, 3), (3, -2), (-2, 1)]
        # the expected rests after each removal: (orange, blue) rectangles
        want = [((8, 0, 5, 5), (0, 5, 8, 3)), ((8, 3, 5, 2), (13, 0, 3, 3)),
                ((11, 3, 2, 2), (8, 5, 3, 1)), ((11, 4, 2, 1), (13, 3, 1, 1)),
                ((12, 4, 1, 1), None)]
        o_, b_ = set(org), set(blu)
        for st, mv in enumerate(moves):
            if mv:
                b_ = {(i + mv[0], j + mv[1]) for (i, j) in b_}
            com = o_ & b_
            check(_cells_rect(com) is not None, f"step {st}: the common part is a rectangle")
            check(len(o_) - len(b_) == 1, f"step {st}: orange has one cell more")
            o_, b_ = o_ - com, b_ - com
            check(_cells_rect(o_) == want[st][0] and _cells_rect(b_) == want[st][1],
                  f"step {st}: the rests are the next square and rectangle")
            check(all(0 <= i < 16 and 0 <= j < 8 for (i, j) in o_ | b_),
                  "everything stays in the 16 x 8 field")
        check(o_ == {(12, 4)} and not b_, "one orange cell is left")

        def sq(i, j, color, op):
            return Square(side_length=u, fill_color=color, fill_opacity=op,
                          stroke_color=WHITE, stroke_width=1.2).move_to(C(i, j))

        orange = {c: sq(*c, ORANGE, 0.9) for c in sorted(org)}
        blue = {c: sq(*c, BLUE_D, 0.72) for c in sorted(blu)}
        d13 = _dim(Pt(0, 0), Pt(13, 0), "13", DOWN * 0.16, ORANGE, 26, gap=0.08)
        d5 = _dim(Pt(13, 0), Pt(13, 5), "5", RIGHT * 0.3, ORANGE, 26, gap=0.08)
        d8 = _dim(Pt(0, 8), Pt(8, 8), "8", UP * 0.16, BLUE_B, 26, gap=0.08)
        _safe(d13, d5, d8, VGroup(*blue.values()), VGroup(*orange.values()))

        # the readout: each line is the next comparison, sign reversed
        xl, y0, dy = 3.5, 3.15, 0.66
        spec = [("5·13 − 8²", {"5·13": ORANGE, "8²": BLUE_B}),
                ("= −(3·8 − 5²)", {"3·8": BLUE_B, "5²": ORANGE}),
                ("= +(2·5 − 3²)", {"2·5": ORANGE, "3²": BLUE_B}),
                ("= −(1·3 − 2²)", {"1·3": BLUE_B, "2²": ORANGE}),
                ("= +(1·2 − 1²)", {"1·2": ORANGE, "1²": BLUE_B}),
                ("= +1", {})]
        lines = []
        for r, (txt, t2c) in enumerate(spec):
            t = Text(txt, font_size=30, color=WHITE, t2c=t2c)
            t.move_to([xl + t.width / 2, y0 - r * dy, 0.0])
            lines.append(t)
        _safe(*lines)
        check(lines[0].get_left()[0] > Pt(16, 0)[0] + 0.3, "readout clear of the cells")

        self.play(LaggedStart(*[FadeIn(m) for m in orange.values()], lag_ratio=0.01),
                  FadeIn(d13), FadeIn(d5), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(m) for m in blue.values()], lag_ratio=0.01),
                  FadeIn(d8), run_time=1.2)
        self.play(FadeIn(lines[0]), run_time=0.5)
        self.hold(0.5)

        def remove_common(step):
            com = sorted(set(orange) & set(blue))
            x0_, y0_, w_, h_ = _cells_rect(com)
            box = Polygon(*[Pt(*q) for q in _rect(x0_, y0_, w_, h_)],
                          stroke_color=YELLOW_B, stroke_width=5)
            self.play(Create(box), run_time=0.5 if step == 0 else 0.4)
            gone = [orange.pop(c) for c in com] + [blue.pop(c) for c in com]
            return box, gone

        for st, mv in enumerate(moves):
            if mv:
                grp = VGroup(*blue.values())
                d = u * np.array([mv[0], mv[1], 0.0])
                self.bring_to_front(grp)
                self.play(grp.animate.shift(d), run_time=1.0 if st == 1 else 0.85)
                blue = {(i + mv[0], j + mv[1]): m for (i, j), m in blue.items()}
                check(all(close(m.get_center(), C(*c), 1e-6) for c, m in blue.items()),
                      f"step {st}: the blue rest lands on whole cells")
            box, gone = remove_common(st)
            extra = [FadeOut(d13), FadeOut(d5), FadeOut(d8)] if st == 0 else []
            self.play(*[FadeOut(m) for m in gone], FadeOut(box), *extra,
                      FadeIn(lines[st + 1]), run_time=0.8 if st == 0 else 0.6)
            check(_cells_rect(set(orange)) == want[st][0]
                  and _cells_rect(set(blue)) == want[st][1],
                  f"step {st}: the picture shows the expected rests")
            self.hold(0.35 if st else 0.6)

        last = orange[(12, 4)]
        ring = Square(side_length=u * 1.4, stroke_color=YELLOW_B,
                      stroke_width=5).move_to(last)
        self.play(Create(ring), last.animate.set_fill(YELLOW_E, opacity=1.0),
                  run_time=0.8)
        # where it sits: the rectangle and the square, outlined again
        ghost_r = DashedVMobject(Polygon(*[Pt(*q) for q in _rect(0, 0, 13, 5)],
                                         stroke_color=ORANGE, stroke_width=3),
                                 num_dashes=72)
        ghost_s = DashedVMobject(Polygon(*[Pt(*q) for q in _rect(0, 0, 8, 8)],
                                         stroke_color=BLUE_B, stroke_width=3),
                                 num_dashes=64)
        check(_sep(ring, d5[0], 0.02), "the ring clears the dimension line")
        self.play(Create(ghost_r), Create(ghost_s), FadeIn(d13), FadeIn(d5),
                  FadeIn(d8), run_time=1.0)
        cap = caption("Fₙ₋₁ · Fₙ₊₁ − Fₙ²   =   (−1)ⁿ", 36)
        self.play(Write(cap), run_time=1.3)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B19

class B19_AlternatingSquares(Board):
    """n² − (n−1)² + (n−2)² − … ± 1², built literally: fill the n-square,
    empty the (n−1)-square in its corner, fill the (n−2)-square, and so on.
    What stays filled is every other L of width one: k² − (k−1)² is the L
    with arms of k and k − 1 cells. Swing each L's short arm a quarter
    turn into the empty row just under its long arm: the L becomes two
    rows, k and k − 1 long, and the rows n, n−1, …, 1 stack into the
    staircase 1 + 2 + … + n:
        n² − (n−1)² + (n−2)² − … ± 1² = n(n+1)/2.
    Drawn for n = 6."""

    def construct(self):
        n, u = 6, 0.78
        O = np.array([-6.0, -2.35, 0.0])

        def C(i, j):
            return O + u * np.array([i + 0.5, j + 0.5, 0.0])

        def Pt(x, y):
            return O + u * np.array([float(x), float(y), 0.0])

        colk = {6: BLUE_D, 4: TEAL_D, 2: ORANGE}
        tcol = {6: BLUE_B, 4: TEAL_B, 2: ORANGE}
        # what stays filled: the L's k = n, n-2, …
        state = {}
        for k in range(n, 0, -1):
            plus = (n - k) % 2 == 0
            for i in range(k):
                for j in range(k):
                    state[(i, j)] = k if plus else None
        kept = sorted(k for k in range(n, 0, -2))
        for (i, j), k in state.items():
            g = max(i, j) + 1                      # the L the cell lies in
            check((k is not None) == ((n - g) % 2 == 0) and (k is None or k == g),
                  "every other L stays, in its own colour")
        check(sum((-1) ** (n - k) * k * k for k in range(1, n + 1)) == n * (n + 1) // 2,
              "the alternating sum is T_n")
        # the quarter turns, cell by cell
        folds = {}
        for k in kept:
            pv = (k / 2, k / 2 - 1)
            arm = [(k - 1, j) for j in range(k - 1)]
            land = []
            for (i, j) in arm:
                x, y = i + 0.5 - pv[0], j + 0.5 - pv[1]
                land.append((round(pv[0] - y - 0.5), round(pv[1] + x - 0.5)))
            check(sorted(land) == [(i, k - 2) for i in range(k - 1)],
                  f"L {k}: its short arm turns into row {k - 2}")
            folds[k] = (pv, arm, land)
        stair = {(i, j) for j in range(n) for i in range(j + 1)}
        rows_after = {(i, k - 1) for k in kept for i in range(k)} | \
                     {c for k in kept for c in folds[k][2]}
        check(rows_after == stair, "the rows make the staircase 1 + 2 + … + n")

        cells = {(i, j): Square(side_length=u, fill_color=BLUE_D, fill_opacity=0.0,
                                stroke_color=GREY_B, stroke_width=1.2).move_to(C(i, j))
                 for i in range(n) for j in range(n)}
        grid = VGroup(*cells.values())
        _safe(grid)

        # the readout: 6² − 5² + 4² − 3² + 2² − 1², term by term
        terms = []
        for k in range(n, 0, -1):
            plus = (n - k) % 2 == 0
            s_ = ("" if k == n else ("+ " if plus else "− ")) + f"{k}²"
            terms.append(tag(s_, 32, tcol[k] if plus else GREY_B))
        row1 = VGroup(*terms).arrange(RIGHT, buff=0.22)
        row1.move_to([2.75, 3.05, 0])
        row2 = Text("= 11 + 7 + 3", font_size=32,
                    t2c={"11": BLUE_B, "7": TEAL_B, "3": ORANGE})
        row2.next_to(row1, DOWN, buff=0.38).align_to(row1, LEFT)
        row3 = tag("= 6 + 5 + 4 + 3 + 2 + 1", 32)
        row3.next_to(row2, DOWN, buff=0.38).align_to(row1, LEFT)
        _safe(row1, row2, row3)
        check(row1.get_left()[0] > Pt(n, 0)[0] + 0.5, "readout clear of the square")

        self.add(grid)
        self.play(FadeIn(grid), run_time=0.6)
        for idx, k in enumerate(range(n, 0, -1)):
            plus = (n - k) % 2 == 0
            box = Polygon(*[Pt(*q) for q in _rect(0, 0, k, k)],
                          stroke_color=YELLOW_B, stroke_width=5)
            anims = []
            for i in range(k):
                for j in range(k):
                    m = cells[(i, j)]
                    if plus:
                        anims.append(m.animate.set_fill(colk[k], opacity=FILL)
                                     .set_stroke(WHITE, width=1.5))
                    else:
                        anims.append(m.animate.set_fill(opacity=0.0)
                                     .set_stroke(GREY_B, width=1.2))
            self.play(Create(box), FadeIn(terms[idx]), run_time=0.45)
            self.play(*anims, run_time=0.6)
            self.play(FadeOut(box), run_time=0.25)
        self.hold(0.3)
        self.play(FadeIn(row2), run_time=0.6)
        self.hold(0.6)

        # each L: its short arm swings into the empty row under its long arm
        for k in sorted(kept, reverse=True):
            pv, arm, land = folds[k]
            grp = VGroup(*[cells[c] for c in arm])
            self.bring_to_front(grp)
            self.play(Rotate(grp, angle=PI / 2, about_point=Pt(*pv)),
                      run_time=1.2 if k == n else 0.9)
            check(_same_spots(grp, [C(*c) for c in land]),
                  f"L {k}: the arm lands exactly in its row")
        # the cells that never held a filled square after the turns
        moved = {id(cells[c]) for k in kept for c in folds[k][1]}
        empty = [m for c, m in cells.items()
                 if id(m) not in moved and (c not in stair or state[c] is None)]
        check(len(empty) == n * n - n * (n + 1) // 2,
              "what is left empty is the other half-staircase")
        self.play(*[FadeOut(m) for m in empty], run_time=0.6)

        outline_pts = [Pt(0, 0), Pt(1, 0)]
        for j in range(1, n):
            outline_pts += [Pt(j, j), Pt(j + 1, j)]
        outline_pts += [Pt(n, n), Pt(0, n)]
        outline = Polygon(*outline_pts, stroke_color=YELLOW_B, stroke_width=5)
        rlabs = VGroup(*[tag(str(j + 1), 26, YELLOW_B).move_to(
            Pt(j + 1, j + 0.5) + RIGHT * 0.3) for j in range(n)])
        _safe(rlabs)
        self.play(Create(outline), FadeIn(rlabs), FadeIn(row3), run_time=1.0)
        cap = caption("n² − (n−1)² + (n−2)² − … ± 1²   =   1 + 2 + … + n"
                      "   =   n(n+1)/2", 32)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)


# ====================================================================== B26

class B26_CubesFromOdds(Board):
    """Group the odd numbers 1 | 3 + 5 | 7 + 9 + 11 | 13 + 15 + 17 + 19 | …:
    group k holds k consecutive odd numbers, standing evenly around k²
    (k² − k + 1, …, k² + k − 1). As bars of cells, the bars of a group pair
    off from the outside in, the longer as far past k² as the shorter falls
    short of it: move each overhang onto its partner and every bar is k²
    long. The group is then k bars of k², a k × k² block, which cuts into
    k squares k × k: k³. So
        n³ = (n² − n + 1) + (n² − n + 3) + … + (n² + n − 1).
    Drawn for k = 1 … 4. (Adding the groups gives the first T_n odd numbers,
    so 1³ + … + n³ = T_n²; that is B4's picture.)"""

    def construct(self):
        u, g = 0.48, 0.42
        x0 = -4.3
        K = 4
        cols = [BLUE_D, TEAL_D, ORANGE, GREEN_D]
        tcols = [BLUE_B, TEAL_B, ORANGE, GREEN_B]
        # row geometry: groups from the top, rows touching inside a group
        ytop = 3.3
        rows = {}                           # (k, r) -> (y_centre, length)
        y = ytop
        for k in range(1, K + 1):
            for r in range(k):
                L = k * k - k + 1 + 2 * r
                rows[(k, r)] = (y - u / 2, L)
                y -= u
            y -= g
        for k in range(1, K + 1):
            Ls = [rows[(k, r)][1] for r in range(k)]
            check(sum(Ls) == k ** 3, f"group {k} sums to k³")
            check(all(Ls[r] + Ls[k - 1 - r] == 2 * k * k for r in range(k)),
                  f"group {k}: the bars pair off around k²")
            check(Ls[0] == k * k - k + 1 and Ls[-1] == k * k + k - 1,
                  f"group {k}: from k²−k+1 to k²+k−1")
        odd = [rows[(k, r)][1] for k in range(1, K + 1) for r in range(k)]
        check(odd == list(range(1, 2 * len(odd), 2)), "the odd numbers in order")

        def cell(x_i, yc, color):
            return Square(side_length=u, fill_color=color, fill_opacity=FILL,
                          stroke_color=WHITE, stroke_width=1.0).move_to(
                [x0 + (x_i + 0.5) * u, yc, 0.0])

        bars = {}
        nums = {}
        for (k, r), (yc, L) in rows.items():
            bars[(k, r)] = [cell(i, yc, cols[k - 1]) for i in range(L)]
            t = tag(str(L), 24, tcols[k - 1])
            t.move_to([x0 - 0.18 - t.width / 2, yc, 0.0])
            nums[(k, r)] = t
        allbars = VGroup(*[m for b in bars.values() for m in b])
        _safe(allbars, *nums.values())

        for k in range(1, K + 1):
            self.play(*[FadeIn(VGroup(*bars[(k, r)]), shift=RIGHT * 0.15)
                        for r in range(k)],
                      *[FadeIn(nums[(k, r)]) for r in range(k)],
                      run_time=0.8 if k < 4 else 0.9)
        self.hold(0.4)

        # each group stands evenly around k²
        centres, clabs = VGroup(), VGroup()
        for k in range(1, K + 1):
            ytop_k = rows[(k, 0)][0] + u / 2
            ybot_k = rows[(k, k - 1)][0] - u / 2
            xc = x0 + k * k * u
            line = DashedLine([xc, ytop_k + 0.04, 0], [xc, ybot_k - 0.12, 0],
                              color=YELLOW_B, stroke_width=3, dash_length=0.08)
            lab = tag(f"{k * k}", 20, YELLOW_B).move_to([xc, ytop_k + 0.25, 0])
            _clear_of_segment(lab, line.get_start(), line.get_end(), gap=0.05,
                              what=f"the mark {k * k}")
            centres.add(line)
            clabs.add(lab)
        _safe(centres, clabs)
        for lab in clabs:
            for b in bars.values():
                for m in b:
                    check(_sep(lab, m, 0.02), "the k² marks clear of the bars")
        self.play(Create(centres), FadeIn(clabs), run_time=0.9)
        self.hold(0.4)

        # level each group: the overhang of a long bar fills its partner
        for k in range(2, K + 1):
            moves = []
            for r in range(k // 2):
                long_r = k - 1 - r
                d = rows[(k, long_r)][1] - k * k          # overhang = shortfall
                blk = bars[(k, long_r)][k * k:]
                check(len(blk) == d and rows[(k, r)][1] + d == k * k,
                      f"group {k}: the overhang fits the shortfall")
                dy = rows[(k, r)][0] - rows[(k, long_r)][0]
                moves.append((VGroup(*blk), np.array([-d * u, dy, 0.0]),
                              r, long_r, d))
            self.play(*[grp.animate.shift(v) for grp, v, *_ in moves],
                      run_time=1.1)
            for grp, v, r, long_r, d in moves:
                yc = rows[(k, r)][0]
                want = [np.array([x0 + (k * k - d + i + 0.5) * u, yc, 0.0])
                        for i in range(d)]
                check(_same_spots(grp, want), f"group {k}: the overhang lands")
                bars[(k, r)] += list(grp)
                bars[(k, long_r)] = bars[(k, long_r)][:k * k]
            for r in range(k):
                check(len(bars[(k, r)]) == k * k, f"group {k}: every bar is k²")

        # k bars of k² = k squares of side k = k³
        cuts, cubes = VGroup(), []
        for k in range(1, K + 1):
            ytop_k = rows[(k, 0)][0] + u / 2
            ybot_k = rows[(k, k - 1)][0] - u / 2
            blk = Polygon([x0, ybot_k, 0], [x0 + k * k * u, ybot_k, 0],
                          [x0 + k * k * u, ytop_k, 0], [x0, ytop_k, 0],
                          stroke_color=YELLOW_B, stroke_width=4)
            cuts.add(blk)
            for m in range(1, k):
                xm = x0 + m * k * u
                cuts.add(Line([xm, ybot_k, 0], [xm, ytop_k, 0], color=YELLOW_B,
                              stroke_width=4))
            t = tag(f"= {k}³", 30, tcols[k - 1])
            t.move_to([x0 + k * k * u + 0.3 + t.width / 2, (ytop_k + ybot_k) / 2, 0])
            cubes.append(t)
        _safe(cuts, *cubes)
        self.play(FadeOut(centres), FadeOut(clabs), run_time=0.4)
        self.play(Create(cuts), *[FadeIn(t) for t in cubes], run_time=1.2)
        cap = caption("n³  =  (n² − n + 1) + (n² − n + 3) + … + (n² + n − 1)", 32)
        self.play(Write(cap), run_time=1.4)
        _final_check(self, cap)
        self.hold(2.2)
