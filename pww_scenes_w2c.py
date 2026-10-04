# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2c.py — proofs without words, 2D (manim):
#     D16 e to the π, π to the e          D18 harmonic numbers and ln
#     D13 Jensen's inequality             D10 crossed ladders
#     D15 ln x lies below its tangent     D12 Young's inequality
#     H16 the integral of sin²            H18 xⁿ and its inverse fill the square
#     H17 the integral of ln x
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# exponents made of letters (π^e, x^(1/n)) are set with true superscripts
# by _supline.  Curves are polylines through densely sampled points; there
# are no Axes with numbers, tick values are tag() labels.
#
# Every scene asserts its invariants with check(...) before drawing — the
# inequality, exact landings of folded or shifted pieces, gap-free tilings —
# and checks at the end that its labels sit inside the safe area, clear of
# each other and of the lines, so a wrong picture fails the render.


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
    """A line of text with true superscripts, without TeX.

    parts: list of (string, is_superscript) or (string, is_superscript,
    colour).  Normal pieces share one baseline; a superscript sits `rise`
    x-heights above it, smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for part in parts:
        s, sup = part[0], part[1]
        col = part[2] if len(part) > 2 else color
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        body = s.strip(" ")
        x += lead * sp
        t = _on_base(body, size * (sup_scale if sup else 1.0), col)
        y0 = rise * xh if sup else 0.0
        t.shift(np.array([x - t.get_left()[0], y0, 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sup else 0.02)
        g.add(t)
    return g


def _cap(group):
    """Place a composite formula where caption() puts its Text: centred,
    along the bottom edge."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.set_x(0.0)
    group.to_edge(DOWN, buff=0.3)
    return group


def _int(lo, hi, size=34, color=YELLOW_B):
    """An integral sign with small limits, built from Text pieces (no TeX,
    and no reliance on superscript letters the target fonts may lack)."""
    sgn = Text("∫", font_size=size * 1.45, color=color)
    up = Text(hi, font_size=size * 0.55, color=color)
    dn = Text(lo, font_size=size * 0.55, color=color)
    f = size / 34
    up.next_to(sgn, RIGHT, buff=0.03 * f).align_to(sgn, UP).shift(UP * 0.04 * f)
    dn.next_to(sgn, RIGHT, buff=0.03 * f).align_to(sgn, DOWN).shift(
        LEFT * 0.13 * f + DOWN * 0.04 * f)
    return VGroup(sgn, up, dn)


def _row(*parts, buff=0.14):
    """Parts set side by side, their centres on one horizontal line."""
    return VGroup(*parts).arrange(RIGHT, buff=buff)


def _trow(parts, size=30, buff=0.2):
    """A row of coloured Text pieces [(s, colour), ...] on one baseline."""
    g = VGroup()
    x = 0.0
    for s, col in parts:
        t = _on_base(s, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + buff
        g.add(t)
    return g


# ------------------------------------------------------------ geometry helpers

def _simpson(g, lo, hi, n=400):
    """Composite Simpson rule (n even)."""
    t = np.linspace(lo, hi, n + 1)
    y = g(t)
    return (hi - lo) / (3 * n) * (y[0] + y[-1] + 4 * y[1:-1:2].sum()
                                  + 2 * y[2:-1:2].sum())


def _curve(pts, color=YELLOW_B, width=5):
    """Open polyline through screen points (dense sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _mirror(p, a, b):
    """Mirror image of the point p in the line ab (2D or 3D points)."""
    p, a, b = to3(p), to3(a), to3(b)
    d = (b - a) / np.linalg.norm(b - a)
    f = a + np.dot(p - a, d) * d
    return 2 * f - p


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


def _verts(poly):
    return [to3(v) for v in poly.get_vertices()]


def _same_pts(P, Q, tol=1e-6):
    """Two point lists agree point by point (same order)."""
    return len(P) == len(Q) and all(close(to3(a), to3(b), tol)
                                    for a, b in zip(P, Q))


def _dim(p, q, color=WHITE, tick=0.1, width=2.5):
    """A dimension bar from p to q with end ticks across it."""
    p, q = to3(p), to3(q)
    d = q - p
    n = np.array([-d[1], d[0], 0.0]) / max(np.linalg.norm(d), 1e-9) * tick
    return VGroup(Line(p, q, color=color, stroke_width=width),
                  Line(p - n, p + n, color=color, stroke_width=width),
                  Line(q - n, q + n, color=color, stroke_width=width))


def _tick(p, d=UP, s=0.08, color=GREY_B):
    """A short tick mark across an axis at screen point p."""
    p = to3(p)
    return Line(p - d * s, p + d * s, color=color, stroke_width=2)


# ------------------------------------------------------------ label hygiene

def _segs(pts):
    """Consecutive segments of a polyline of screen points."""
    pts = [to3(p) for p in pts]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _circle_segs(c, r, n=72):
    c = to3(c)
    pts = [c + r * np.array([np.cos(a), np.sin(a), 0.0])
           for a in np.linspace(0, TAU, n + 1)]
    return _segs(pts)


def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by
    pad), or None."""
    x0, x1, y0, y1 = _box(m, pad)
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        # quick reject on the boxes
        if (max(p[0], q[0]) < x0 or min(p[0], q[0]) > x1
                or max(p[1], q[1]) < y0 or min(p[1], q[1]) > y1):
            continue
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.015) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _overlap(a, b, gap=0.05):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _safe(*mobs, tol=0.02):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for m in mobs:
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area: {_name(m)} "
              f"[{lo[0]:.2f},{hi[0]:.2f}]x[{lo[1]:.2f},{hi[1]:.2f}]")


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every line in segs (a
    list of screen segments) and of every other label."""
    _safe(*labels)
    for m in labels:
        i = _hit(m, segs, pad)
        bx = ", ".join(f"{v:.2f}" for v in _box(m))
        sg = "" if i is None else (f" {np.round(segs[i][0][:2], 2)}-"
                                   f"{np.round(segs[i][1][:2], 2)}")
        check(i is None, f"{what}: '{_name(m)}' [{bx}] clear of the lines "
              f"(hits #{i}{sg})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            bi = ", ".join(f"{v:.2f}" for v in _box(labels[i]))
            bj = ", ".join(f"{v:.2f}" for v in _box(labels[j]))
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' [{bi}] clear of "
                  f"'{_name(labels[j])}' [{bj}]")


def _below(cap, *mobs, gap=0.03):
    """Fail the render if the caption reaches up into any content above it
    (only content that overlaps it horizontally counts)."""
    c0, c1 = cap.get_critical_point(DL), cap.get_critical_point(UR)
    check(c1[1] < SAFE_BOTTOM + 0.02 and c0[1] > -4.0 and cap.width <= 13.45,
          "caption inside the caption band")
    for m in mobs:
        m0, m1 = m.get_critical_point(DL), m.get_critical_point(UR)
        if m1[0] < c0[0] or m0[0] > c1[0]:
            continue
        check(c1[1] + gap <= m0[1], f"caption clear of {_name(m)} "
              f"(caption top {c1[1]:.2f}, content bottom {m0[1]:.2f})")


# ========================================================== D16  e^π vs π^e

def _tangent_pts(c1, r1, c2, r2):
    """Touch points of the two outer common tangents of two circles
    (screen centres c1, c2; radii r1 < r2): [(T1, T2), (T1', T2')]."""
    c1, c2 = to3(c1), to3(c2)
    d = c2 - c1
    L = float(np.linalg.norm(d))
    u = d / L
    v = np.array([-u[1], u[0], 0.0])
    s = (r2 - r1) / L
    out = []
    for sg in (1, -1):
        n = s * u + sg * np.sqrt(1 - s * s) * v     # n·(c2 − c1) = r2 − r1
        out.append((c1 - r1 * n, c2 - r2 * n))
    return out


def _line_in_disc(a, b, c, r):
    """The chord that the line through screen points a, b cuts from the
    disc (c, r), as two screen points."""
    a, b, c = to3(a), to3(b), to3(c)
    d = (b - a) / np.linalg.norm(b - a)
    f = a - c
    bq = float(np.dot(f, d))
    cq = float(np.dot(f, f)) - r * r
    disc = bq * bq - cq
    check(disc > 0, "the line crosses the magnifier")
    s = np.sqrt(disc)
    return a + (-bq - s) * d, a + (-bq + s) * d


class D16_EPiPiE(Board):
    """y = ln x is concave, so it lies below each of its tangent lines.
    The tangent at a has slope 1/a and passes through the origin when
    ln a = a · (1/a) = 1, i.e. a = e: the line from O touching the curve
    touches at T = (e, 1), with rise 1 over run e (slope 1/e).  Every other
    point of the curve lies below that line, so the chord from O to
    (π, ln π) is lower: ln π / π < 1/e, i.e. e·ln π < π, i.e. π^e < e^π.
    π is close to e, so at true scale the gap at x = π (0.011) is far
    below one pixel; a magnifier (×20) shows it."""

    def construct(self):
        E_, P_ = float(np.e), float(np.pi)
        lnp = float(np.log(P_))
        kx, ky = 1.65, 2.1
        O = np.array([-6.1, -0.6, 0.0])

        def S(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        # ---- the claims
        check(abs(np.log(E_) - E_ * (1 / E_)) < 1e-12,
              "the tangent at e, slope 1/e, passes through O")
        xs_all = np.linspace(0.05, 12.0, 20001)
        check(bool(np.all(np.log(xs_all) <= xs_all / E_ + 1e-12)),
              "ln x ≤ x/e: the curve lies below the line OT")
        check(lnp / P_ < 1 / E_ and E_ * lnp < P_ and P_ ** E_ < E_ ** P_,
              "ln π / π < 1/e ⟺ e ln π < π ⟺ π^e < e^π")
        gap = P_ / E_ - lnp
        check(0.0109 < gap < 0.0111, "the gap at π is 0.011")

        x_lo, x_hi = 0.335, 4.62
        xs = np.concatenate([np.geomspace(x_lo, 1.0, 90)[:-1],
                             np.linspace(1.0, x_hi, 240)])
        cpts = [S(x, np.log(x)) for x in xs]
        axes = VGroup(Line(S(-0.12, 0), S(4.8, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -1.1), S(0, 1.86), color=GREY_B, stroke_width=2))
        curve = _curve(cpts, YELLOW_B, 5)
        # beside the steep lower branch of the curve, right of it
        l_curve = tag("y = ln x", 26, YELLOW_B)
        l_curve.next_to(S(np.exp(-0.62), -0.62), RIGHT, buff=0.3)
        l_O = tag("O", 24, GREY_A).next_to(S(0, 0), DL, buff=0.08)
        self.play(Create(axes), FadeIn(l_O), run_time=0.6)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.3)

        # ---- the line from O that touches the curve: at T = (e, 1)
        T = S(E_, 1.0)
        tline = Line(S(0, 0), S(x_hi, x_hi / E_), color=WHITE, stroke_width=3.5)
        l_tline = tag("y = x/e", 26).next_to(tline.get_end(), UP, buff=0.14)
        dotT = Dot(T, radius=0.065, color=WHITE)
        l_T = tag("T", 24).next_to(T, UL, buff=0.16)
        tri = mk([S(0, 0), S(E_, 0), T], BLUE_E, 0.45, stroke_width=0)
        leg = Line(S(E_, 0), T, color=WHITE, stroke_width=2.5)
        tk_e = _tick(S(E_, 0))
        l_e = tag("e", 28).next_to(S(E_, 0), DOWN, buff=0.16)
        l_1 = tag("1", 26).next_to(Line(S(E_, 0), T), LEFT, buff=0.12)
        self.play(Create(tline), FadeIn(dotT), FadeIn(l_T), FadeIn(l_tline),
                  run_time=1.0)
        self.add(tri)
        self.bring_to_back(tri)
        self.play(FadeIn(tri), Create(leg), Create(tk_e), FadeIn(l_e),
                  FadeIn(l_1), run_time=0.9)
        self.bring_to_front(curve, tline, dotT)
        self.hold(0.5)

        # ---- every chord from O to the curve lies under OT (touching at T)
        xv = ValueTracker(1.55)

        def chord():
            x = xv.get_value()
            return Line(S(0, 0), S(x, np.log(x)), color=ORANGE, stroke_width=3.5)

        def pdot():
            x = xv.get_value()
            return Dot(S(x, np.log(x)), radius=0.06, color=ORANGE)

        ch = always_redraw(chord)
        pd = always_redraw(pdot)
        self.play(FadeIn(ch), FadeIn(pd), run_time=0.4)
        self.play(xv.animate.set_value(4.45), run_time=2.4, rate_func=smooth)
        self.play(xv.animate.set_value(P_), run_time=1.0, rate_func=smooth)
        ch.clear_updaters()
        pd.clear_updaters()
        self.remove(ch, pd)
        Pi = S(P_, lnp)
        chord_pi = Line(S(0, 0), Pi, color=ORANGE, stroke_width=3.5)
        dotPi = Dot(Pi, radius=0.06, color=ORANGE)
        self.add(chord_pi, dotPi)

        # ---- at x = π: rise ln π over run π
        guide_pi = DashedLine(S(P_, 0), Pi, color=ORANGE, stroke_width=2.5,
                              dash_length=0.08)
        tk_pi = _tick(S(P_, 0))
        l_pi = tag("π", 28, ORANGE).next_to(S(P_, 0), DOWN, buff=0.16)
        l_pi.align_to(l_e, DOWN)
        l_lnpi = tag("ln π", 26, ORANGE).next_to(
            Line(S(P_, 0), Pi), RIGHT, buff=0.14)
        self.play(Create(guide_pi), Create(tk_pi), FadeIn(l_pi), FadeIn(l_lnpi),
                  run_time=0.9)
        self.bring_to_front(curve, tline, chord_pi, dotT, dotPi)

        # ---- the magnifier: the line OT passes just above (π, ln π)
        M = 20.0
        rs = 0.094
        Rb = M * rs
        Cb = np.array([4.55, -0.62, 0.0])

        def Z(p):
            return Cb + M * (to3(p) - Pi)

        lens = Circle(radius=rs, color=GREY_A, stroke_width=2).move_to(Pi)
        big = Circle(radius=Rb, color=GREY_A, stroke_width=3).move_to(Cb)
        big.set_fill(BLACK, opacity=1.0)
        (a1, b1), (a2, b2) = _tangent_pts(Pi, rs, Cb, Rb)
        conn = VGroup(Line(a1, b1, color=GREY_A, stroke_width=1.6),
                      Line(a2, b2, color=GREY_A, stroke_width=1.6))
        zx = np.linspace(P_ - 0.12, P_ + 0.12, 4001)
        zin = [Z(S(x, np.log(x))) for x in zx]
        zin = [p for p in zin if np.linalg.norm(p - Cb) <= Rb * 0.995]
        check(len(zin) > 100, "the curve crosses the magnifier")
        z_curve = _curve(zin, YELLOW_B, 5)
        z_t = _line_in_disc(Z(S(0, 0)), Z(S(x_hi, x_hi / E_)), Cb, Rb * 0.995)
        z_c = _line_in_disc(Z(S(0, 0)), Z(Pi), Cb, Rb * 0.995)
        z_tline = Line(*z_t, color=WHITE, stroke_width=3.5)
        z_chord = Line(*z_c, color=ORANGE, stroke_width=3.5)
        z_guide = DashedLine(Cb, Cb + DOWN * Rb * 0.995, color=ORANGE,
                             stroke_width=2.5, dash_length=0.1)
        z_dot = Dot(Cb, radius=0.075, color=ORANGE)
        top = Z(S(P_, P_ / E_))
        check(close(Z(Pi), Cb), "the magnifier is centred on (π, ln π)")
        check(abs((top - Cb)[1] - M * ky * gap) < 1e-9 and (top - Cb)[1] > 0.3,
              "magnified, the line OT passes visibly above (π, ln π)")
        z_gap = Line(Cb, top, color=RED_B, stroke_width=7)
        l_M = tag("× 20", 24, GREY_A).move_to([5.95, -2.47, 0.0])
        self.play(Create(lens), run_time=0.4)
        self.play(Create(conn), FadeIn(big), run_time=0.8)
        self.play(Create(z_curve), Create(z_tline), Create(z_chord),
                  Create(z_guide), FadeIn(z_dot), FadeIn(l_M), run_time=1.1)
        self.play(Create(z_gap), run_time=0.5)
        self.bring_to_front(z_dot)
        self.hold(0.5)

        # ---- slopes: ln π / π < 1/e, so e·ln π < π
        row1 = _trow([("ln π / π", ORANGE), ("<", WHITE), ("1/e", WHITE)], 30)
        row2 = _trow([("⟹", GREY_A), ("e · ln π", WHITE), ("<", WHITE),
                      ("π", WHITE)], 30)
        # the two '<' signs one above the other
        row1.shift(np.array([4.95 - row1[1].get_center()[0], 3.25, 0.0]))
        row2.shift(np.array([4.95 - row2[2].get_center()[0], 2.5, 0.0]))
        self.play(FadeIn(row1), run_time=0.8)
        self.play(FadeIn(row2), run_time=0.8)

        # ---- checks on the finished figure
        segs = (_segs(cpts) + _segs([S(-0.12, 0), S(4.8, 0)])
                + _segs([S(0, -1.1), S(0, 1.86)])
                + _segs([S(0, 0), S(x_hi, x_hi / E_)]) + _segs([S(0, 0), Pi])
                + _segs([S(E_, 0), T]) + _segs([S(P_, 0), Pi])
                + _segs([a1, b1]) + _segs([a2, b2]) + _circle_segs(Cb, Rb)
                + _circle_segs(Pi, rs, 24))
        labels = [l_curve, l_O, l_tline, l_T, l_e, l_1, l_pi, l_lnpi, l_M,
                  row1, row2]
        _labels_ok(labels, segs, "D16")
        content = [axes, curve, tline, tri, big, conn, *labels]
        _safe(*content)
        cap = _cap(_supline([("π", False), ("e", True), ("  <  ", False),
                             ("e", False), ("π", True)], 44, YELLOW_B))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D18  Hₙ and ln

class D18_HarmonicLog(Board):
    """Bars of width 1 and heights 1, 1/2, …, 1/n (here n = 5): total area
    1 + 1/2 + … + 1/n.  Standing over [k, k + 1] each bar touches y = 1/x
    at its top-left corner and rises above the curve (1/k ≥ 1/x there), so
    together they contain the region under the curve from 1 to n + 1, of
    area ln(n + 1).  Slid one unit left, onto [k − 1, k], bars 2 … n lie
    under the curve on [1, n] (1/k ≤ 1/x there) and bar 1 is the unit
    square: the sum is at most 1 + ln n.  (Both inequalities are strict:
    the orange overhangs and the blue gaps have positive area.)"""

    def construct(self):
        n = 5
        kx, ky = 1.6, 4.0
        O = np.array([-5.9, -2.45, 0.0])

        def S(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        def inv(x):
            return 1.0 / x

        # ---- the claims
        H = sum(1.0 / k for k in range(1, n + 1))
        for k in range(1, n + 1):
            xs = np.linspace(k, k + 1, 201)
            check(bool(np.all(1.0 / k >= inv(xs) - 1e-15)),
                  f"bar {k} over [k, k+1] stands above the curve")
            if k >= 2:
                xs = np.linspace(k - 1, k, 201)
                check(bool(np.all(1.0 / k <= inv(xs) + 1e-15)),
                      f"bar {k} over [k−1, k] lies under the curve")
        check(abs(_simpson(inv, 1, n + 1, 2000) - np.log(n + 1)) < 1e-9,
              "area under 1/x from 1 to n+1 = ln(n+1)")
        check(abs(_simpson(inv, 1, n, 2000) - np.log(n)) < 1e-9,
              "area under 1/x from 1 to n = ln n")
        over = sum(1.0 / k - _simpson(inv, k, k + 1, 400) for k in range(1, n + 1))
        check(abs(over - (H - np.log(n + 1))) < 1e-9 and over > 0,
              "orange overhangs = sum − ln(n+1) > 0")
        under = sum(_simpson(inv, k - 1, k, 400) - 1.0 / k for k in range(2, n + 1))
        check(abs(under - (1 + np.log(n) - H)) < 1e-9 and under > 0,
              "blue gaps = 1 + ln n − sum > 0")

        x_lo, x_hi = 0.8, n + 1.45
        cx = np.concatenate([np.geomspace(x_lo, 2.0, 120)[:-1],
                             np.linspace(2.0, x_hi, 200)])
        cpts = [S(x, inv(x)) for x in cx]
        axes = VGroup(Line(S(-0.15, 0), S(n + 1.6, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -0.04), S(0, 1.3), color=GREY_B, stroke_width=2))
        ticks = VGroup(*[_tick(S(k, 0)) for k in range(1, n + 2)])
        tlabs = VGroup(*[tag(str(k), 24, GREY_A).next_to(S(k, 0), DOWN, buff=0.14)
                         for k in range(1, n + 2)])
        curve = _curve(cpts, YELLOW_B, 5)
        l_curve = tag("y = 1/x", 26, YELLOW_B).next_to(cpts[0], RIGHT, buff=0.22)
        self.play(Create(axes), Create(ticks), FadeIn(tlabs), run_time=0.8)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.0)

        # ---- the bars over [k, k+1]: heights 1, 1/2, …, 1/n
        def bar_pts(k, s):
            return [S(k + s, 0), S(k + 1 + s, 0), S(k + 1 + s, 1.0 / k),
                    S(k + s, 1.0 / k)]

        bars = VGroup()
        for k in range(1, n + 1):
            b = mk(bar_pts(k, 0), TEAL_D, 0.7, stroke_width=2.5)
            t = tag("1" if k == 1 else f"1/{k}", 24).move_to(S(k + 0.5, 0.5 / k))
            bars.add(VGroup(b, t))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars],
                              lag_ratio=0.25), run_time=1.6)
        self.bring_to_front(curve)

        # ---- they contain the region under the curve from 1 to n+1
        def region(a, b):
            xs = np.linspace(a, b, 160)
            return [S(a, 0)] + [S(x, inv(x)) for x in xs] + [S(b, 0)]

        outA = Polygon(*region(1, n + 1), stroke_color=BLUE_B, stroke_width=6)
        overs = VGroup(*[mk([S(k, 1.0 / k)] + [S(x, inv(x)) for x in
                                                np.linspace(k, k + 1, 60)]
                            + [S(k + 1, 1.0 / k)], ORANGE, 0.95, stroke_width=0)
                         for k in range(1, n + 1)])
        self.play(Create(outA), run_time=1.0)
        self.play(FadeIn(overs), run_time=0.7)
        self.bring_to_front(curve, outA)

        def sum_text():
            return [("1 + 1/2 + 1/3 + 1/4 + 1/5", TEAL_B)]

        row1 = _trow(sum_text() + [(">", WHITE), ("ln 6", BLUE_B)], 30)
        row2 = _trow(sum_text() + [("<", WHITE), ("1", WHITE), ("+", WHITE),
                                   ("ln 5", BLUE_B)], 30)
        # row 2 is the longer: right-aligned; row 1's '>' above its '<'
        row2.shift(np.array([6.35 - row2.get_right()[0], 2.4, 0.0]))
        row1.shift(np.array([row2[1].get_center()[0] - row1[1].get_center()[0],
                             3.2, 0.0]))
        self.play(FadeIn(row1), run_time=0.8)
        self.hold(0.9)

        # labels and lines of the first picture
        segs1 = (_segs(cpts) + _segs([S(-0.15, 0), S(n + 1.6, 0)])
                 + _segs([S(0, -0.04), S(0, 1.3)])
                 + sum((_segs(bar_pts(k, 0) + [bar_pts(k, 0)[0]])
                        for k in range(1, n + 1)), []))
        labs1 = [l_curve, row1, *tlabs]
        _labels_ok(labs1, segs1, "D18 (bars right)")
        for k, b in enumerate(bars, start=1):
            check(_hit(b[1], segs1, 0.04) is None,
                  f"bar label 1/{k} clear of the lines")

        # ---- slide the bars one unit left: onto [k − 1, k]
        self.play(FadeOut(overs), FadeOut(outA), run_time=0.6)
        self.play(bars.animate.shift(LEFT * kx), run_time=1.6)
        for k, b in enumerate(bars, start=1):
            check(_same_pts(_verts(b[0]), bar_pts(k, -1)),
                  f"bar {k} lands on [k−1, k]")
        self.bring_to_front(curve)
        outB = Polygon(*region(1, n), stroke_color=BLUE_B, stroke_width=6)
        gaps = VGroup(*[mk([S(k - 1, 1.0 / k)] + [S(x, inv(x)) for x in
                                                   np.linspace(k - 1, k, 60)],
                           BLUE_D, 0.95, stroke_width=0)
                        for k in range(2, n + 1)])
        unit = Polygon(*bar_pts(1, -1), stroke_color=WHITE, stroke_width=5)
        self.play(Create(outB), run_time=1.0)
        self.play(FadeIn(gaps), run_time=0.7)
        self.bring_to_front(curve, outB)
        self.play(Create(unit), run_time=0.6)

        self.play(FadeIn(row2), run_time=0.8)

        segs2 = (_segs(cpts) + _segs([S(-0.15, 0), S(n + 1.6, 0)])
                 + _segs([S(0, -0.04), S(0, 1.3)])
                 + sum((_segs(bar_pts(k, -1) + [bar_pts(k, -1)[0]])
                        for k in range(1, n + 1)), []))
        labs2 = [l_curve, row1, row2, *tlabs]
        _labels_ok(labs2, segs2, "D18 (bars left)")
        for k, b in enumerate(bars, start=1):
            check(_hit(b[1], segs2, 0.04) is None,
                  f"bar label 1/{k} clear of the lines (shifted)")
        content = [axes, curve, bars, outB, *labs2]
        _safe(*content)
        cap = caption("ln(n + 1)  <  1 + 1/2 + … + 1/n  <  1 + ln n", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D13  Jensen

class D13_Jensen(Board):
    """Masses t and 1 − t (0 ≤ t ≤ 1) at the points X = (x, f(x)) and
    Y = (y, f(y)) of a convex curve balance at C = tX + (1 − t)Y, on the
    chord XY.  C's coordinates are the weighted means: abscissa
    tx + (1 − t)y, height t f(x) + (1 − t) f(y).  Convex means the chord
    lies above the arc, so the curve point F straight below C, of height
    f(tx + (1 − t)y), is not higher than C.  (Drawn for an increasing f, so
    the height guides to the axis stay clear of the curve; the argument
    does not use it.)"""

    def construct(self):
        def f(u):
            return 0.25 * np.exp(0.45 * u)

        kx, ky = 1.3, 1.35
        O = np.array([-2.85, -2.45, 0.0])

        def S(u, v):
            return O + np.array([kx * u, ky * v, 0.0])

        x0, y0, t0 = 0.9, 6.0, 0.4
        fx, fy = f(x0), f(y0)

        def m_of(t):
            return t * x0 + (1 - t) * y0

        def chord_h(t):
            return t * fx + (1 - t) * fy

        # ---- the claims
        us = np.linspace(-0.5, 7.0, 3001)
        d2 = np.diff(f(us), 2)
        check(bool(np.all(d2 > 0)), "f is convex")
        for t in np.linspace(0.0, 1.0, 401):
            check(chord_h(t) >= f(m_of(t)) - 1e-12,
                  "the chord lies above the arc: t f(x) + (1−t) f(y) ≥ f(tx + (1−t)y)")
            C = t * np.array([x0, fx]) + (1 - t) * np.array([y0, fy])
            check(close(C, (m_of(t), chord_h(t))), "C = tX + (1−t)Y has the weighted-mean coordinates")
            check(abs(area([(x0, fx), (y0, fy), C])) < 1e-12, "C lies on the chord XY")
        check(chord_h(t0) - f(m_of(t0)) > 0.8, "a visible gap at the chosen t")

        u_hi = 6.3
        cu = np.linspace(0.0, u_hi, 260)
        cpts = [S(u, f(u)) for u in cu]
        axes = VGroup(Line(S(-0.1, 0), S(6.6, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -0.05), S(0, 4.25), color=GREY_B, stroke_width=2))
        curve = _curve(cpts, YELLOW_B, 5)
        l_f = tag("f", 30, YELLOW_B).next_to(cpts[-1], RIGHT, buff=0.16)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_f), run_time=1.1)

        # ---- two points of the curve, their coordinates
        X, Y = S(x0, fx), S(y0, fy)
        gx = VGroup(DashedLine(S(x0, 0), X, color=GREY_B, stroke_width=2,
                               dash_length=0.08),
                    DashedLine(S(0, fx), X, color=GREY_B, stroke_width=2,
                               dash_length=0.08))
        gy = VGroup(DashedLine(S(y0, 0), Y, color=GREY_B, stroke_width=2,
                               dash_length=0.08),
                    DashedLine(S(0, fy), Y, color=GREY_B, stroke_width=2,
                               dash_length=0.08))
        l_x = tag("x", 28).next_to(S(x0, 0), DOWN, buff=0.16)
        l_y = tag("y", 28).next_to(S(y0, 0), DOWN, buff=0.16)
        l_y.align_to(l_x, UP)
        l_fx = tag("f(x)", 26).next_to(S(0, fx), LEFT, buff=0.16)
        l_fy = tag("f(y)", 26).next_to(S(0, fy), LEFT, buff=0.16)
        dX, dY = Dot(X, radius=0.06), Dot(Y, radius=0.06)
        self.play(Create(gx), Create(gy), FadeIn(dX), FadeIn(dY), FadeIn(l_x),
                  FadeIn(l_y), FadeIn(l_fx), FadeIn(l_fy), run_time=1.1)
        chord = Line(X, Y, color=WHITE, stroke_width=4)
        self.play(Create(chord), run_time=0.8)
        self.bring_to_front(dX, dY)

        # ---- masses t and 1 − t at X and Y balance at C on the chord
        tv = ValueTracker(t0)

        def disc(P, w):
            return Circle(radius=0.07 + 0.2 * np.sqrt(w), stroke_width=0
                          ).set_fill(TEAL_B, 0.6).move_to(P)

        wX = always_redraw(lambda: disc(X, tv.get_value()))
        wY = always_redraw(lambda: disc(Y, 1 - tv.get_value()))
        l_t = tag("t", 28, TEAL_B).move_to(X + np.array([-0.45, 0.4, 0.0]))
        l_1t = tag("1 − t", 28, TEAL_B).move_to(Y + np.array([0.82, -0.25, 0.0]))

        def Cpt():
            t = tv.get_value()
            return S(m_of(t), chord_h(t))

        def Fpt():
            t = tv.get_value()
            return S(m_of(t), f(m_of(t)))

        dC = always_redraw(lambda: Dot(Cpt(), radius=0.075, color=ORANGE))
        dF = always_redraw(lambda: Dot(Fpt(), radius=0.075, color=BLUE_B))
        gap = always_redraw(lambda: Line(Fpt(), Cpt(), color=RED_B, stroke_width=7))
        vert = always_redraw(lambda: DashedLine(
            S(m_of(tv.get_value()), 0), Fpt(), color=GREY_B, stroke_width=2,
            dash_length=0.08))
        self.play(FadeIn(wX), FadeIn(wY), FadeIn(l_t), FadeIn(l_1t), run_time=0.8)
        self.play(FadeIn(dC, scale=0.5), run_time=0.6)
        self.play(Create(vert), FadeIn(dF), Create(gap), run_time=0.8)
        self.bring_to_front(dC, dF)

        # ---- whatever the weights: C on the chord, never below the curve
        self.play(tv.animate.set_value(0.86), run_time=1.4, rate_func=smooth)
        self.play(tv.animate.set_value(0.14), run_time=2.0, rate_func=smooth)
        self.play(tv.animate.set_value(t0), run_time=1.2, rate_func=smooth)
        for m in (wX, wY, dC, dF, gap, vert):
            m.clear_updaters()

        # ---- read off the coordinates of C and F
        C, F = S(m_of(t0), chord_h(t0)), S(m_of(t0), f(m_of(t0)))
        check(close(dC.get_center(), C) and close(dF.get_center(), F),
              "parked at the chosen t")
        gC = DashedLine(C, S(0, chord_h(t0)), color=ORANGE, stroke_width=2.5,
                        dash_length=0.08)
        gF = DashedLine(F, S(0, f(m_of(t0))), color=BLUE_B, stroke_width=2.5,
                        dash_length=0.08)
        l_m = tag("tx + (1 − t)y", 26).next_to(S(m_of(t0), 0), DOWN, buff=0.16)
        l_m.align_to(l_x, UP)
        l_C = tag("t f(x) + (1 − t) f(y)", 26, ORANGE).next_to(
            S(0, chord_h(t0)), LEFT, buff=0.16)
        l_F = tag("f(tx + (1 − t)y)", 26, BLUE_B).next_to(
            S(0, f(m_of(t0))), LEFT, buff=0.16)
        tks = VGroup(_tick(S(x0, 0)), _tick(S(y0, 0)), _tick(S(m_of(t0), 0)),
                     _tick(S(0, fx), RIGHT), _tick(S(0, fy), RIGHT))
        tkC = _tick(S(0, chord_h(t0)), RIGHT)
        tkF = _tick(S(0, f(m_of(t0))), RIGHT)
        self.play(FadeIn(l_m), Create(tks), run_time=0.7)
        self.play(Create(gC), Create(tkC), FadeIn(l_C), run_time=0.9)
        self.play(Create(gF), Create(tkF), FadeIn(l_F), run_time=0.9)
        self.bring_to_front(dC, dF)
        self.hold(0.6)

        segs = (_segs(cpts) + _segs([S(-0.1, 0), S(6.6, 0)])
                + _segs([S(0, -0.05), S(0, 4.25)]) + _segs([X, Y])
                + _segs([S(x0, 0), X]) + _segs([S(0, fx), X])
                + _segs([S(y0, 0), Y]) + _segs([S(0, fy), Y])
                + _segs([S(m_of(t0), 0), F]) + _segs([F, C])
                + _segs([C, S(0, chord_h(t0))]) + _segs([F, S(0, f(m_of(t0)))])
                + _circle_segs(X, wX.width / 2, 24) + _circle_segs(Y, wY.width / 2, 24))
        labels = [l_f, l_x, l_y, l_fx, l_fy, l_t, l_1t, l_m, l_C, l_F]
        _labels_ok(labels, segs, "D13")
        content = [axes, curve, chord, wX, wY, *labels]
        _safe(*content)
        cap = caption("f(tx + (1 − t)y)  ≤  t f(x) + (1 − t) f(y),     0 ≤ t ≤ 1", 32)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D10  crossed ladders

def _homothety(mob, centre, factor, **kw):
    """Enlargement about `centre`, the factor growing linearly from 1 to
    `factor`: every frame is a similar copy, exact at the end."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().scale(1 + alpha * (factor - 1), about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


class D10_CrossedLadders(Board):
    """Poles a and b stand d apart; the ladders from the foot of each to the
    top of the other cross at X, at height h, above the point F that cuts
    the ground into p (on a's side) and q (on b's side).  Shrunk about the
    foot of pole a by p/d, the right triangle under the ladder that leans
    on pole b lands exactly on the triangle under X on a's side: h/b = p/d.
    Shrunk about the foot of pole b by q/d, the triangle under the other
    ladder lands on the triangle under X on b's side: h/a = q/d.  Adding,
    h/a + h/b = (p + q)/d = 1, so 1/h = 1/a + 1/b — whatever d is: slide
    the pole and X stays on the level h."""

    def construct(self):
        a, b, d0 = 3.4, 2.2, 4.6
        h = a * b / (a + b)
        k = 1.4
        O = np.array([-5.6, -1.75, 0.0])

        def S(x, y):
            return O + np.array([k * x, k * y, 0.0])

        def geo(d):
            p, q = d * h / b, d * h / a
            return p, q

        # ---- the claims, for many distances d
        for d in np.linspace(1.5, 9.0, 31):
            A0, A1 = np.array([0.0, a]), np.array([d, 0.0])     # top of a → foot of b
            B0, B1 = np.array([d, b]), np.array([0.0, 0.0])     # top of b → foot of a
            # the crossing: A0 + s(A1 − A0) = B1 + u(B0 − B1)
            s, _ = np.linalg.solve(np.column_stack([A1 - A0, -(B0 - B1)]), B1 - A0)
            Xc = A0 + s * (A1 - A0)
            p, q = geo(d)
            check(abs(Xc[1] - h) < 1e-12 and abs(Xc[0] - p) < 1e-12,
                  "the ladders cross at height h = ab/(a+b), above p = dh/b")
            check(abs(p + q - d) < 1e-12, "p + q = d")
            check(abs(h / b - p / d) < 1e-12 and abs(h / a - q / d) < 1e-12,
                  "h/b = p/d and h/a = q/d")
        check(abs(1 / h - (1 / a + 1 / b)) < 1e-12, "1/h = 1/a + 1/b")

        p0, q0 = geo(d0)
        dv = ValueTracker(d0)

        def D():
            return dv.get_value()

        # ---- ground and poles
        ground = Line(S(-0.45, 0), S(7.55, 0), color=GREY_B, stroke_width=3)
        poleA = Line(S(0, 0), S(0, a), color=GREY_A, stroke_width=9)
        l_a = tag("a", 30).next_to(Line(S(0, 0), S(0, a)), LEFT, buff=0.2)

        def pole_b():
            return Line(S(D(), 0), S(D(), b), color=GREY_A, stroke_width=9)

        def lab_b():
            return tag("b", 30).next_to(Line(S(D(), 0), S(D(), b)), RIGHT, buff=0.2)

        poleB, l_b = pole_b(), lab_b()
        self.play(Create(ground), run_time=0.5)
        self.play(Create(poleA), Create(poleB), FadeIn(l_a), FadeIn(l_b), run_time=0.9)

        # ---- the ladders cross at X, height h
        def ladA():                      # from the top of pole a to the foot of pole b
            return Line(S(0, a), S(D(), 0), color=TEAL_B, stroke_width=4)

        def ladB():                      # from the top of pole b to the foot of pole a
            return Line(S(D(), b), S(0, 0), color=ORANGE, stroke_width=4)

        def Xp():
            return S(geo(D())[0], h)

        def Fp():
            return S(geo(D())[0], 0)

        LA, LB = ladA(), ladB()
        self.play(Create(LA), Create(LB), run_time=1.0)

        def xdot():
            return Dot(Xp(), radius=0.07, color=YELLOW_B)

        def xf():
            return DashedLine(Xp(), Fp(), color=YELLOW_B, stroke_width=3,
                              dash_length=0.09)

        def lab_h():
            p = geo(D())[0]
            return tag("h", 30, YELLOW_B).next_to(
                Line(S(p, 0.2 * h), S(p, 0.62 * h)), RIGHT, buff=0.14)

        dX, XF, l_h = xdot(), xf(), lab_h()
        self.play(FadeIn(dX), Create(XF), FadeIn(l_h), run_time=0.8)

        # ground: p, q and d
        def ground_marks():
            p, q = geo(D())
            g = VGroup(_tick(S(0, 0), UP, 0.1, GREY_A), _tick(S(p, 0), UP, 0.1, GREY_A),
                       _tick(S(D(), 0), UP, 0.1, GREY_A))
            lp = tag("p", 28, ORANGE).move_to(S(p / 2, 0) + DOWN * 0.34)
            lq = tag("q", 28, TEAL_B).move_to(S(p + q / 2, 0) + DOWN * 0.34)
            bar = _dim(S(0, 0) + DOWN * 0.72, S(D(), 0) + DOWN * 0.72, GREY_A, 0.1, 2.5)
            ld = tag("d", 28).move_to(S(D() / 2, 0) + DOWN * 1.02)
            return VGroup(g, lp, lq, bar, ld)

        marks = ground_marks()
        self.play(FadeIn(marks), run_time=0.8)
        self.hold(0.4)

        # ---- shrink the triangle under ladder B about the foot of pole a
        OL, OR_ = S(0, 0), S(d0, 0)
        triB = mk([OL, OR_, S(d0, b)], ORANGE, 0.32, stroke_width=0)
        self.add(triB)
        self.bring_to_back(triB)
        self.play(FadeIn(triB), run_time=0.6)
        smB = triB.copy().set_fill(ORANGE, 0.75)
        self.add(smB)
        self.play(_homothety(smB, OL, p0 / d0), run_time=1.6)
        check(_same_pts(_verts(smB), [OL, S(p0, 0), S(p0, h)]),
              "the triangle under ladder B, shrunk by p/d about the foot of a, "
              "lands on (foot, F, X)")
        self.play(FadeOut(triB), run_time=0.4)
        self.bring_to_front(LA, LB, XF, dX, poleA)

        row1 = _trow([("h/b", ORANGE), ("=", WHITE), ("p/d", ORANGE)], 32)
        row2 = _trow([("h/a", TEAL_B), ("=", WHITE), ("q/d", TEAL_B)], 32)
        row3 = _trow([("h/a + h/b", WHITE), ("=", WHITE), ("(p + q)/d", WHITE),
                      ("=", WHITE), ("1", WHITE)], 32)
        xe = 3.0
        for r, y in ((row1, 3.25), (row2, 2.55), (row3, 1.85)):
            r.shift(np.array([xe - r[1].get_center()[0], y, 0.0]))
        self.play(FadeIn(row1), run_time=0.7)

        # ---- shrink the triangle under ladder A about the foot of pole b
        triA = mk([OR_, OL, S(0, a)], TEAL_D, 0.32, stroke_width=0)
        self.add(triA)
        self.bring_to_back(triA)
        self.play(FadeIn(triA), run_time=0.6)
        smA = triA.copy().set_fill(TEAL_D, 0.75)
        self.add(smA)
        self.play(_homothety(smA, OR_, q0 / d0), run_time=1.6)
        check(_same_pts(_verts(smA), [OR_, S(p0, 0), S(p0, h)]),
              "the triangle under ladder A, shrunk by q/d about the foot of b, "
              "lands on (foot, F, X)")
        self.play(FadeOut(triA), run_time=0.4)
        self.bring_to_front(LA, LB, XF, dX, poleA, poleB)
        self.play(FadeIn(row2), run_time=0.7)
        self.play(FadeIn(row3), run_time=0.8)
        self.hold(0.6)

        # ---- wherever the poles stand: X stays on the level h
        level = DashedLine(S(-0.3, h), S(7.4, h), color=YELLOW_B, stroke_width=2,
                           dash_length=0.12).set_opacity(0.7)
        self.add(level)
        self.bring_to_back(level)
        self.play(Create(level), run_time=0.7)

        def small(col, which):
            p = geo(D())[0]
            pts = ([S(0, 0), S(p, 0), S(p, h)] if which == 0
                   else [S(D(), 0), S(p, 0), S(p, h)])
            return mk(pts, col, 0.75, stroke_width=0)

        live = [always_redraw(pole_b), always_redraw(lab_b), always_redraw(ladA),
                always_redraw(ladB), always_redraw(xf), always_redraw(xdot),
                always_redraw(lab_h), always_redraw(ground_marks),
                always_redraw(lambda: small(ORANGE, 0)),
                always_redraw(lambda: small(TEAL_D, 1))]
        self.remove(poleB, l_b, LA, LB, XF, dX, l_h, marks, smB, smA)
        self.add(live[8], live[9], *live[:8])
        self.play(dv.animate.set_value(7.0), run_time=1.8, rate_func=smooth)
        self.play(dv.animate.set_value(3.1), run_time=2.0, rate_func=smooth)
        self.play(dv.animate.set_value(d0), run_time=1.2, rate_func=smooth)
        for m in live:
            m.clear_updaters()

        # ---- label hygiene at the extremes and at the end
        for d in (3.1, d0, 7.0):
            dv.set_value(d)
            p, q = geo(d)
            segs = (_segs([S(-0.45, 0), S(7.55, 0)]) + _segs([S(0, 0), S(0, a)])
                    + _segs([S(d, 0), S(d, b)]) + _segs([S(0, a), S(d, 0)])
                    + _segs([S(d, b), S(0, 0)]) + _segs([S(p, h), S(p, 0)])
                    + _segs([S(-0.3, h), S(7.4, h)]))
            labs = [l_a, lab_b(), lab_h(), *ground_marks()[1:3], ground_marks()[4],
                    row1, row2, row3]
            _labels_ok(labs, segs, f"D10 at d = {d}")
            check(_hit(ground_marks()[4], _segs([S(0, 0) + DOWN * 0.72,
                                                 S(d, 0) + DOWN * 0.72]), 0.04) is None,
                  "label d clear of its bar")
        dv.set_value(d0)
        content = [ground, poleA, *live, level, row1, row2, row3, l_a]
        _safe(*content)
        cap = caption("1/h  =  1/a  +  1/b", 38)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D15  ln x ≤ x − 1

class D15_LogBelowTangent(Board):
    """D14 mirrored in the line y = x: vertical strips become horizontal
    bands.  The band between the y-axis and the curve x = eʸ (that is,
    y = ln x), from height 0 to height ln x, has area ∫ eʸ dy = x − 1 (the
    mirror image of the region under eᵗ).  For x > 1 the curve stays right
    of x = 1 there (eʸ ≥ 1), so the band contains the rectangle of width 1
    and height ln x: ln x ≤ x − 1.  For 0 < x < 1 the band runs from ln x
    up to 0, has area 1 − x and lies inside the rectangle of width 1 and
    height −ln x (eʸ ≤ 1): 1 − x ≤ −ln x.  Either way ln x ≤ x − 1, so the
    tangent y = x − 1 at (1, 0) never dips below the curve.  (The orange
    excess is x − 1 − ln x in both cases.)"""

    def construct(self):
        kx, ky = 1.7, 1.3
        O = np.array([-5.75, -0.1, 0.0])

        def P(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        x1, x2 = 3.2, 0.25
        L1, L2 = float(np.log(x1)), float(np.log(x2))
        x_lo, x_hi = 0.115, 3.6

        # ---- the claims
        for x in (x1, x2, 0.6, 1.7):
            lo, hi = sorted((0.0, float(np.log(x))))
            band = _simpson(np.exp, lo, hi)
            check(abs(band - abs(x - 1)) < 1e-9,
                  "the band left of the curve between 0 and ln x has area |x − 1|")
            check(np.log(x) <= x - 1, "ln x ≤ x − 1")
        check(bool(np.all(np.exp(np.linspace(0, L1, 60)) >= 1)),
              "eʸ ≥ 1 on [0, ln x]: the width-1 rectangle lies in the band")
        check(bool(np.all(np.exp(np.linspace(L2, 0, 60)) <= 1)),
              "eʸ ≤ 1 on [ln x, 0]: the band lies in the width-1 rectangle")
        ex1 = _simpson(lambda y: np.exp(y) - 1, 0, L1)
        check(abs(ex1 - (x1 - 1 - L1)) < 1e-9, "excess = x − 1 − ln x (x > 1)")
        ex2 = _simpson(lambda y: 1 - np.exp(y), L2, 0)
        check(abs(ex2 - (x2 - 1 - L2)) < 1e-9, "excess = x − 1 − ln x (x < 1)")
        xs = np.linspace(0.01, 20, 20001)
        check(bool(np.all(np.log(xs) <= xs - 1 + 1e-15)), "the tangent stays above")

        axes = VGroup(Line(P(-0.12, 0), P(x_hi + 0.12, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, -2.22), P(0, 2.7), color=GREY_B, stroke_width=2))
        cys = np.linspace(np.log(x_lo), np.log(x_hi), 260)
        cpts = [P(np.exp(y), y) for y in cys]
        curve = _curve(cpts, YELLOW_B, 5)
        l_curve = tag("y = ln x", 28, YELLOW_B).next_to(cpts[-1], RIGHT, buff=0.18)
        t0, t1 = 0.0, x_hi
        tang = Line(P(t0, t0 - 1), P(t1, t1 - 1), color=WHITE, stroke_width=3.5)
        l_tang = tag("y = x − 1", 28).next_to(P(t1, t1 - 1), RIGHT, buff=0.18)
        dot = Dot(P(1, 0), radius=0.07)
        tk1 = _tick(P(1, 0))
        l_one = tag("1", 26).next_to(P(1, 0), DR, buff=0.12)

        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.2)
        self.play(Create(tang), FadeIn(dot), Create(tk1), FadeIn(l_one),
                  FadeIn(l_tang), run_time=1.0)
        self.hold(0.6)
        self.play(tang.animate.set_stroke(opacity=0.25),
                  l_tang.animate.set_opacity(0.35), run_time=0.5)

        def ys(a, b, m=80):
            return np.linspace(a, b, m)

        # ---- x > 1: the band (area x − 1) contains the rectangle (area ln x)
        bandA = Polygon(P(0, 0), *[P(np.exp(y), y) for y in ys(0, L1)], P(0, L1),
                        stroke_color=YELLOW_B, stroke_width=4)
        rectA = mk([P(0, 0), P(1, 0), P(1, L1), P(0, L1)], BLUE_D, stroke_width=1.5)
        excA = mk([P(1, 0)] + [P(np.exp(y), y) for y in ys(0, L1)] + [P(1, L1)],
                  ORANGE, 0.85, stroke_width=1.5)
        tkA = _tick(P(0, L1), RIGHT)
        l_A = tag("ln x", 26, BLUE_B).next_to(P(0, L1), LEFT, buff=0.16)
        self.play(Create(bandA), run_time=0.9)
        self.play(FadeIn(rectA), FadeIn(excA), Create(tkA), FadeIn(l_A), run_time=0.9)
        self.bring_to_front(bandA, curve, dot)

        LX = 1.75
        rowA = _trow([("x > 1 :", GREY_A), ("ln x", BLUE_B), ("≤", WHITE),
                      ("x − 1", YELLOW_B)], 30, 0.22)
        rowA.shift(np.array([LX - rowA.get_left()[0], 0.95, 0.0]))
        self.play(FadeIn(rowA), run_time=0.8)
        self.hold(0.8)

        # ---- x < 1: the band (area 1 − x) lies in the rectangle (area −ln x)
        rectB = Polygon(P(0, L2), P(1, L2), P(1, 0), P(0, 0), stroke_color=YELLOW_B,
                        stroke_width=4)
        bandB = mk([P(0, L2)] + [P(np.exp(y), y) for y in ys(L2, 0)] + [P(0, 0)],
                   BLUE_D, stroke_width=1.5)
        excB = mk([P(1, L2), P(1, 0)] + [P(np.exp(y), y) for y in ys(0, L2)],
                  ORANGE, 0.85, stroke_width=1.5)
        tkB = _tick(P(0, L2), RIGHT)
        l_B = tag("ln x", 26, YELLOW_B).next_to(P(0, L2), LEFT, buff=0.16)
        self.play(Create(rectB), Create(tkB), FadeIn(l_B), run_time=0.9)
        self.play(FadeIn(bandB), FadeIn(excB), run_time=0.9)
        self.bring_to_front(rectB, curve, dot)
        rowB = _trow([("x < 1 :", GREY_A), ("1 − x", BLUE_B), ("≤", WHITE),
                      ("−ln x", YELLOW_B)], 30, 0.22)
        rowB.shift(np.array([LX - rowB.get_left()[0], 0.15, 0.0]))
        self.play(FadeIn(rowB), run_time=0.8)
        self.hold(0.8)

        # ---- either way: the curve stays below its tangent
        rowC = _trow([("⟹", WHITE), ("ln x", YELLOW_B), ("≤", WHITE),
                      ("x − 1", WHITE)], 30, 0.22)
        rowC.shift(np.array([LX + 1.15 - rowC.get_left()[0], -0.75, 0.0]))
        self.play(tang.animate.set_stroke(opacity=1.0),
                  l_tang.animate.set_opacity(1.0), FadeIn(rowC), run_time=0.9)
        self.bring_to_front(tang, curve, dot)

        segs = (_segs(cpts) + _segs([P(-0.12, 0), P(x_hi + 0.12, 0)])
                + _segs([P(0, -2.22), P(0, 2.7)]) + _segs([P(t0, t0 - 1), P(t1, t1 - 1)])
                + _segs([P(0, 0), P(1, 0), P(1, L1), P(0, L1)])
                + _segs([P(0, L2), P(1, L2), P(1, 0)]))
        labels = [l_curve, l_tang, l_one, l_A, l_B, rowA, rowB, rowC]
        _labels_ok(labels, segs, "D15")
        content = [axes, curve, tang, rectA, excA, bandA, rectB, bandB, excB, *labels]
        _safe(*content)
        cap = caption("ln x  ≤  x − 1,     x > 0", 38)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ========================================================== D12  Young

class D12_Young(Board):
    """f increasing, f(0) = 0.  A: the region under f over [0, a]; B: the
    region left of f (between the y-axis and the curve) over heights
    [0, b] — the region under f⁻¹, turned.  A point of the a × b rectangle
    is either on or below the curve (then it is in A) or on or above it
    (then it is in B): A and B cover the rectangle, meeting only along the
    curve, so ab ≤ area A + area B.  What sticks out (orange) is the part
    of A above height b when b < f(a), the part of B right of x = a when
    b > f(a), and nothing when b = f(a): then they tile it exactly."""

    def construct(self):
        def f(u):
            return 0.42 * np.power(u, 1.8)

        def finv(v):
            return np.power(v / 0.42, 1 / 1.8)

        kx, ky = 1.9, 1.4
        O = np.array([-3.5, -2.45, 0.0])

        def S(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        a = 3.0
        fa = float(f(a))
        b0, b2 = 2.3, 3.75
        u_hi = 3.55

        def A_pts():
            return [S(0, 0)] + [S(x, f(x)) for x in np.linspace(0, a, 140)] + [S(a, 0)]

        def B_pts(b):
            return [S(0, 0)] + [S(finv(y), y) for y in np.linspace(0, b, 140)] + [S(0, b)]

        def rect_pts(b):
            return [S(0, 0), S(a, 0), S(a, b), S(0, b)]

        def excess_pts(b):
            if b < fa - 1e-9:
                x0 = finv(b)
                return ([S(x0, b)] + [S(x, f(x)) for x in np.linspace(x0, a, 60)]
                        + [S(a, b)])
            if b > fa + 1e-9:
                return ([S(a, fa)] + [S(finv(y), y) for y in np.linspace(fa, b, 60)]
                        + [S(a, b)])
            return None

        # ---- the claims: A and B cover the rectangle, meeting on the curve
        check(abs(f(0.0)) < 1e-15 and bool(np.all(np.diff(f(np.linspace(0, 4, 400))) > 0)),
              "f increasing with f(0) = 0")
        for b in (b0, fa, b2, 1.4, 3.3):
            # the parts of A and B inside the rectangle tile it exactly
            # (both pieces are built on one shared polyline of the curve)
            xe = min(a, float(finv(b)))
            C = [(x, float(f(x))) for x in np.linspace(0, xe, 200)]
            Ar = C + ([(a, b)] if xe < a else []) + [(a, 0)]
            Br = C + ([(a, b)] if xe >= a and b > fa else []) + [(0, b)]
            check(_tiles_exactly([Ar, Br], [(0, 0), (a, 0), (a, b), (0, b)], 90),
                  f"A and B tile the a × b rectangle (b = {b:.2f})")
            IA = 0.42 * a ** 2.8 / 2.8                      # exact antiderivatives
            IB = (1 / 0.42) ** (1 / 1.8) * b ** (1 + 1 / 1.8) / (1 + 1 / 1.8)
            check(abs(IA - _simpson(f, 0, a, 2000)) < 1e-6, "∫f checked")
            check(IA + IB >= a * b - 1e-12, "ab ≤ ∫f + ∫f⁻¹")
        IBa = (1 / 0.42) ** (1 / 1.8) * fa ** (1 + 1 / 1.8) / (1 + 1 / 1.8)
        check(abs(0.42 * a ** 2.8 / 2.8 + IBa - a * fa) < 1e-12, "equality when b = f(a)")

        bv = ValueTracker(b0)

        def B():
            return bv.get_value()

        axes = VGroup(Line(S(-0.1, 0), S(u_hi + 0.15, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -0.08), S(0, 4.12), color=GREY_B, stroke_width=2))
        cx = np.linspace(0, u_hi, 220)
        cpts = [S(x, f(x)) for x in cx]
        curve = _curve(cpts, YELLOW_B, 5)
        l_f = tag("f", 30, YELLOW_B).next_to(cpts[-1], RIGHT, buff=0.14)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_f), run_time=1.0)

        # ---- A: under f over [0, a]
        regA = mk(A_pts(), BLUE_D, stroke_width=0)
        tka = _tick(S(a, 0))
        l_a = tag("a", 30).next_to(S(a, 0), DOWN, buff=0.16)
        l_IA = _row(_int("0", "a", 24, WHITE), tag("f(x) dx", 24), buff=0.06)
        l_IA.move_to(S(2.2, 0.42))
        self.play(FadeIn(regA), Create(tka), FadeIn(l_a), run_time=0.9)
        self.bring_to_front(curve)
        self.play(FadeIn(l_IA), run_time=0.6)

        # ---- B: left of f over heights [0, b]
        def regB():
            return mk(B_pts(B()), TEAL_D, stroke_width=0)

        def tkb():
            return _tick(S(0, B()), RIGHT)

        def lab_b():
            return tag("b", 30).next_to(S(0, B()), LEFT, buff=0.16)

        rB, tB, lB = regB(), tkb(), lab_b()
        l_IB = _row(_int("0", "b", 24, WHITE),
                    _supline([("f", False), ("−1", True), ("(y) dy", False)], 24, WHITE),
                    buff=0.06)
        l_IB.move_to(S(0.72, 1.72))
        self.play(FadeIn(rB), Create(tB), FadeIn(lB), run_time=0.9)
        self.bring_to_front(curve)
        self.play(FadeIn(l_IB), run_time=0.6)
        self.hold(0.4)

        # ---- the a × b rectangle: covered; what sticks out is orange
        def rect():
            return Polygon(*rect_pts(B()), stroke_color=WHITE, stroke_width=5)

        def excess():
            pts = excess_pts(B())
            if pts is None:
                return VMobject()
            return mk(pts, ORANGE, 0.95, stroke_width=0)

        rc = rect()
        self.play(Create(rc), run_time=1.0)
        ex = excess()
        self.add(ex)
        self.play(FadeIn(ex), run_time=0.7)
        self.bring_to_front(curve, rc)
        self.hold(1.0)

        # ---- slide b: through b = f(a) (exact tiling) to b > f(a)
        live = [always_redraw(regB), always_redraw(excess), always_redraw(rect),
                always_redraw(tkb), always_redraw(lab_b)]
        self.remove(rB, ex, rc, tB, lB)
        # back to front: B, A, the overhang, the curve, the rectangle, labels
        self.add(live[0])
        self.bring_to_front(regA)
        self.add(live[1])
        self.bring_to_front(curve)
        self.add(*live[2:])
        self.bring_to_front(l_IA, l_IB)
        self.play(bv.animate.set_value(fa), run_time=1.5, rate_func=smooth)
        corner = Dot(S(a, fa), radius=0.09, color=YELLOW_B)
        l_eq = tag("b = f(a)", 26, YELLOW_B).next_to(S(a, fa), RIGHT, buff=0.3)
        self.play(FadeIn(corner, scale=0.5), FadeIn(l_eq), run_time=0.5)
        self.hold(0.6)
        self.play(FadeOut(corner), FadeOut(l_eq), run_time=0.4)
        self.play(bv.animate.set_value(b2), run_time=1.4, rate_func=smooth)
        self.hold(0.6)
        self.play(bv.animate.set_value(b0), run_time=1.6, rate_func=smooth)
        for m in live:
            m.clear_updaters()

        # ---- label hygiene at the visited values of b
        for b in (b0, fa, b2):
            bv.set_value(b)
            segs = (_segs(cpts) + _segs([S(-0.1, 0), S(u_hi + 0.15, 0)])
                    + _segs([S(0, -0.08), S(0, 4.12)]) + _segs(rect_pts(b) + [S(0, 0)]))
            ex_pts = excess_pts(b)
            if ex_pts is not None:
                segs += _segs(ex_pts + [ex_pts[0]])
            labs = [l_f, l_a, lab_b(), l_IA, l_IB]
            if abs(b - fa) < 1e-9:
                labs.append(l_eq)
            _labels_ok(labs, segs, f"D12 at b = {b:.2f}")
            # the integral labels sit inside their own regions
            for lab, poly in ((l_IA, A_pts()), (l_IB, B_pts(b))):
                for c in (lab.get_corner(UL), lab.get_corner(UR), lab.get_corner(DL),
                          lab.get_corner(DR)):
                    check(_pip(c, poly), "region label inside its region")
        bv.set_value(b0)
        order = [id(m) for m in self.mobjects]
        check(order.index(id(live[1])) > order.index(id(regA)) > order.index(id(live[0])),
              "the orange overhang is drawn above A and B")
        content = [axes, curve, regA, *live, l_f, l_a, l_IA, l_IB]
        _safe(*content)
        cap = _cap(_row(tag("ab  ≤", 30, YELLOW_B), _int("0", "a", 30),
                        tag("f(x) dx  +", 30, YELLOW_B), _int("0", "b", 30),
                        _supline([("f", False), ("−1", True), ("(y) dy", False)], 30),
                        buff=0.12))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# ========================================================== H16  ∫ sin²

def _same_set(P, Q, tol=1e-6):
    """The same points, in any order."""
    P = [to3(p) for p in P]
    Q = [to3(q) for q in Q]
    return (len(P) == len(Q)
            and all(any(np.linalg.norm(p - q) < tol for q in Q) for p in P)
            and all(any(np.linalg.norm(p - q) < tol for p in P) for q in Q))


class H16_SineSquaredIntegral(Board):
    """In the rectangle [0, π/2] × [0, 1]: the region under sin²x, folded
    over the line x = π/4, lands exactly on the region under cos²x
    (cos²x = sin²(π/2 − x)); folded again over the line y = ½, it lands on
    the part of the rectangle above sin²x (1 − cos²x = sin²x).  So two
    copies of the region tile the rectangle, and the region has half its
    area: π/4.  (The two folds together are the half-turn about the centre
    (π/4, ½): the curve sin² is symmetric about that point.)"""

    def construct(self):
        H = PI / 2
        kx, ky = 4.6, 5.0
        O = np.array([-5.4, -2.4, 0.0])

        def S(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        N = 181
        xs = np.linspace(0, H, N)
        R_pts = [S(0, 0)] + [S(x, np.sin(x) ** 2) for x in xs] + [S(H, 0)]
        C_pts = [S(0, 0)] + [S(x, np.cos(x) ** 2) for x in xs] + [S(H, 0)]
        U_pts = [S(0, 1)] + [S(x, np.sin(x) ** 2) for x in xs] + [S(H, 1)]
        rect_pts = [S(0, 0), S(H, 0), S(H, 1), S(0, 1)]

        # ---- the claims
        check(abs(_simpson(lambda x: np.sin(x) ** 2, 0, H) - PI / 4) < 1e-12,
              "∫ sin² over [0, π/2] = π/4")
        check(bool(np.allclose(np.sin(H - xs) ** 2, np.cos(xs) ** 2)),
              "cos²x = sin²(π/2 − x): mirror images in x = π/4")
        check(bool(np.allclose(1 - np.cos(xs) ** 2, np.sin(xs) ** 2)),
              "1 − cos²x = sin²x: turned over y = ½, cos² fits above sin²")
        check(_tiles_exactly([[p[:2] for p in R_pts], [p[:2] for p in U_pts]],
                             [p[:2] for p in rect_pts], 100),
              "the region under sin² and the region above it tile the rectangle")

        frame = Polygon(*rect_pts, stroke_color=GREY_A, stroke_width=2.5)
        axes = VGroup(Line(S(-0.06, 0), S(H + 0.08, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -0.03), S(0, 1.08), color=GREY_B, stroke_width=2))
        tks = VGroup(_tick(S(PI / 4, 0)), _tick(S(H, 0)), _tick(S(0, 0.5), RIGHT),
                     _tick(S(0, 1), RIGHT))
        l_p4 = tag("π/4", 26).next_to(S(PI / 4, 0), DOWN, buff=0.15)
        l_p2 = tag("π/2", 26).next_to(S(H, 0), DOWN, buff=0.15)
        l_h = tag("½", 26).next_to(S(0, 0.5), LEFT, buff=0.15)
        l_1 = tag("1", 26).next_to(S(0, 1), LEFT, buff=0.15)
        self.play(Create(axes), Create(frame), Create(tks), FadeIn(l_p4),
                  FadeIn(l_p2), FadeIn(l_h), FadeIn(l_1), run_time=1.0)

        # ---- sin² and the region under it
        sin_c = _curve([S(x, np.sin(x) ** 2) for x in xs], YELLOW_B, 5)
        l_sin = _supline([("sin", False), ("2", True), ("x", False)], 28, YELLOW_B)
        l_sin.next_to(S(H, 1), RIGHT, buff=0.2)
        reg = mk(R_pts, BLUE_D, stroke_width=0)
        self.play(Create(sin_c), FadeIn(l_sin), run_time=1.1)
        self.add(reg)
        self.bring_to_back(reg)
        self.play(FadeIn(reg), run_time=0.7)
        self.hold(0.4)

        # ---- fold a copy over x = π/4: the region under cos²
        axV = DashedLine(S(PI / 4, -0.02), S(PI / 4, 1.04), color=WHITE,
                         stroke_width=2.5, dash_length=0.1)
        cpy = mk(R_pts, TEAL_D, 0.72, stroke_width=0)
        self.play(Create(axV), run_time=0.5)
        self.add(cpy)
        self.play(Rotate(cpy, angle=PI, axis=UP, about_point=S(PI / 4, 0.5)),
                  run_time=1.6)
        check(_same_set(_verts(cpy), C_pts), "folded over x = π/4: the region under cos²")
        cos_c = _curve([S(x, np.cos(x) ** 2) for x in xs], TEAL_B, 4)
        l_cos = _supline([("cos", False), ("2", True), ("x", False)], 28, TEAL_B)
        l_cos.next_to(S(H, 0), RIGHT, buff=0.2).shift(UP * 0.3)
        self.play(Create(cos_c), FadeIn(l_cos), FadeOut(axV), run_time=0.9)
        self.bring_to_front(sin_c)
        self.hold(0.5)

        # ---- fold it over y = ½: it fills the rectangle above sin²
        axH = DashedLine(S(-0.02, 0.5), S(H + 0.03, 0.5), color=WHITE,
                         stroke_width=2.5, dash_length=0.1)
        self.play(Create(axH), cos_c.animate.set_stroke(opacity=0.45), run_time=0.5)
        self.play(Rotate(cpy, angle=PI, axis=RIGHT, about_point=S(PI / 4, 0.5)),
                  run_time=1.6)
        check(_same_set(_verts(cpy), U_pts), "folded over y = ½: the region above sin²")
        self.bring_to_front(sin_c)
        self.play(FadeOut(axH), run_time=0.4)

        # ---- two copies tile the 1 × π/2 rectangle
        row1 = _supline([("sin", False, YELLOW_B), ("2", True, YELLOW_B),
                         ("x", False, YELLOW_B), ("  +  ", False, WHITE),
                         ("cos", False, TEAL_B), ("2", True, TEAL_B),
                         ("x", False, TEAL_B), ("  =  1", False, WHITE)], 28)
        row2 = _row(tag("2", 28), _int("0", "π/2", 28, WHITE),
                    _supline([("sin", False), ("2", True), ("x dx  =  π/2", False)],
                             28, WHITE), buff=0.1)
        row1.move_to([4.3, 1.2, 0.0])
        row2.move_to([4.3, 0.05, 0.0])
        self.play(FadeIn(row1), run_time=0.8)
        self.play(FadeIn(row2), run_time=0.9)

        segs = (_segs([S(-0.06, 0), S(H + 0.08, 0)]) + _segs([S(0, -0.03), S(0, 1.08)])
                + _segs(rect_pts + [rect_pts[0]])
                + _segs([S(x, np.sin(x) ** 2) for x in xs])
                + _segs([S(x, np.cos(x) ** 2) for x in xs]))
        labels = [l_p4, l_p2, l_h, l_1, l_sin, l_cos, row1, row2]
        _labels_ok(labels, segs, "H16")
        content = [frame, axes, reg, cpy, *labels]
        _safe(*content)
        cap = _cap(_row(_int("0", "π/2", 32),
                        _supline([("sin", False), ("2", True), ("x dx  =  π/4", False)],
                                 32), buff=0.1))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== H18  xⁿ and x^(1/n)

class H18_PowerAndRoot(Board):
    """In the unit square the curve y = xⁿ cuts off A, the region under it
    (area ∫₀¹ xⁿ dx), and B, the region above it; together they fill the
    square.  Folded over the diagonal y = x, B becomes the region under
    the mirror-image curve y = x^(1/n) (area ∫₀¹ x^(1/n) dx), here slid
    onto a second unit square.  So the two integrals add up to 1.  Drawn
    for n = 3; the closing sweep of n shows the same tiling for any n > 0."""

    def construct(self):
        s = 4.8
        L0 = np.array([-6.0, -2.4, 0.0])
        R0 = np.array([1.35, -2.4, 0.0])

        def L(x, y):
            return L0 + s * np.array([x, y, 0.0])

        def R(x, y):
            return R0 + s * np.array([x, y, 0.0])

        n0 = 3.0
        nv = ValueTracker(n0)
        xs = np.linspace(0, 1, 241)

        def A_pts(n):
            return [L(x, x ** n) for x in xs] + [L(1, 0)]

        def B_pts(n):
            return [L(x, x ** n) for x in xs] + [L(0, 1)]

        def root_pts(n):            # region under x^(1/n) in the right square
            return [R(x ** n, x) for x in xs] + [R(1, 0)]

        sqL = [L(0, 0), L(1, 0), L(1, 1), L(0, 1)]
        sqR = [R(0, 0), R(1, 0), R(1, 1), R(0, 1)]

        # ---- the claims
        for n in (n0, 1.0, 2.0, 6.0, 0.5):
            check(_tiles_exactly([[p[:2] for p in A_pts(n)], [p[:2] for p in B_pts(n)]],
                                 [p[:2] for p in sqL], 90),
                  f"A and B tile the unit square (n = {n})")
            mir = [_mirror(p, L(0, 0), L(1, 1)) + (R0 - L0) for p in B_pts(n)]
            check(_same_pts(mir, root_pts(n)),
                  "B folded over y = x (and slid over) is the region under x^(1/n)")
            IA = 1 / (n + 1)
            IB = n / (n + 1)
            check(abs(_simpson(lambda x: x ** n, 0, 1, 2000) - IA) < 1e-6
                  and abs(IA + IB - 1) < 1e-12, "∫xⁿ + ∫x^(1/n) = 1")

        def square(pts):
            return Polygon(*pts, stroke_color=GREY_A, stroke_width=2.5)

        def ticks(F):
            return VGroup(tag("0", 24, GREY_A).next_to(F(0, 0), DL, buff=0.08),
                          tag("1", 24, GREY_A).next_to(F(1, 0), DOWN, buff=0.12),
                          tag("1", 24, GREY_A).next_to(F(0, 1), LEFT, buff=0.12))

        fL, tL = square(sqL), ticks(L)
        self.play(Create(fL), FadeIn(tL), run_time=0.8)

        def curve_L():
            n = nv.get_value()
            return _curve([L(x, x ** n) for x in xs], YELLOW_B, 5)

        cL = curve_L()
        l_cL = _supline([("y = x", False), ("n", True)], 28, YELLOW_B)
        l_cL.next_to(L(1, 1), UP, buff=0.18)
        self.play(Create(cL), FadeIn(l_cL), run_time=1.0)

        def regA():
            return mk(A_pts(nv.get_value()), BLUE_D, stroke_width=0)

        def regB():
            return mk(B_pts(nv.get_value()), ORANGE, 0.78, stroke_width=0)

        rA = regA()
        l_A = _row(_int("0", "1", 24, WHITE),
                   _supline([("x", False), ("n", True), (" dx", False)], 24, WHITE),
                   buff=0.06).move_to(L(0.775, 0.115))
        self.add(rA)
        self.bring_to_back(rA)
        self.play(FadeIn(rA), FadeIn(l_A), run_time=0.9)
        rB = regB()
        self.add(rB)
        self.bring_to_back(rB)
        self.play(FadeIn(rB), run_time=0.8)
        self.bring_to_front(cL)
        self.hold(0.5)

        # ---- the second square, with the mirror-image curve y = x^(1/n)
        fR, tR = square(sqR), ticks(R)

        def curve_R():
            n = nv.get_value()
            return _curve([R(x ** n, x) for x in xs], YELLOW_B, 5)

        cR = curve_R()
        l_cR = _supline([("y = x", False), ("1/n", True)], 28, YELLOW_B)
        l_cR.next_to(R(1, 1), UP, buff=0.18)
        l_cR.shift(LEFT * max(0.0, l_cR.get_right()[0] - 6.4))
        self.play(Create(fR), FadeIn(tR), Create(cR), FadeIn(l_cR), run_time=1.1)

        # ---- fold a copy of B over the diagonal y = x, slide it across
        diag = DashedLine(L(0, 0), L(1, 1), color=WHITE, stroke_width=2.5,
                          dash_length=0.1)
        self.play(Create(diag), run_time=0.5)
        cp = regB().set_stroke(WHITE, 2.5)
        self.add(cp)
        self.bring_to_front(l_A)
        self.play(Rotate(cp, angle=PI, axis=normalize(np.array([1.0, 1.0, 0.0])),
                         about_point=L(0, 0)), run_time=1.6)
        check(_same_pts(_verts(cp), [_mirror(p, L(0, 0), L(1, 1)) for p in B_pts(n0)]),
              "the fold is the mirror image in y = x")
        self.play(cp.animate.shift(R0 - L0), FadeOut(diag), run_time=1.4)
        check(_same_pts(_verts(cp), root_pts(n0)), "it lands on the region under x^(1/n)")
        self.bring_to_front(cR)
        l_R = _row(_int("0", "1", 24, WHITE),
                   _supline([("x", False), ("1/n", True), (" dx", False)], 24, WHITE),
                   buff=0.06).move_to(R(0.6, 0.3))
        self.play(FadeIn(l_R), run_time=0.7)
        self.hold(0.8)

        # label hygiene at n = 3 (labels inside their regions, clear of lines)
        segsL = _segs(sqL + [sqL[0]]) + _segs([L(x, x ** n0) for x in xs])
        segsR = _segs(sqR + [sqR[0]]) + _segs([R(x ** n0, x) for x in xs])
        labels = [*tL, *tR, l_cL, l_cR, l_A, l_R]
        _labels_ok(labels, segsL + segsR, "H18")
        for lab, poly in ((l_A, A_pts(n0)), (l_R, root_pts(n0))):
            for c in (lab.get_corner(UL), lab.get_corner(UR), lab.get_corner(DL),
                      lab.get_corner(DR)):
                check(_pip(c, poly), "region label inside its region")

        # ---- for every n: the same tiling and the same fold
        def regC():
            return mk(root_pts(nv.get_value()), ORANGE, 0.78, stroke_width=2.5)

        live = [always_redraw(regB), always_redraw(regA), always_redraw(regC),
                always_redraw(curve_L), always_redraw(curve_R)]
        self.play(FadeOut(l_A), FadeOut(l_R), run_time=0.4)
        self.remove(rA, rB, cp, cL, cR)
        self.add(*live)
        self.play(nv.animate.set_value(1.0), run_time=1.3, rate_func=smooth)
        self.play(nv.animate.set_value(6.0), run_time=1.6, rate_func=smooth)
        self.play(nv.animate.set_value(n0), run_time=1.3, rate_func=smooth)
        for m in live:
            m.clear_updaters()
        self.play(FadeIn(l_A), FadeIn(l_R), run_time=0.5)

        content = [fL, fR, *live, *labels]
        _safe(*content)
        cap = _cap(_row(_int("0", "1", 30),
                        _supline([("x", False), ("n", True), (" dx  +", False)], 30),
                        _int("0", "1", 30),
                        _supline([("x", False), ("1/n", True), (" dx  =  1", False)], 30),
                        buff=0.1))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== H17  ∫ ln x

class H17_LogIntegral(Board):
    """The rectangle [0, a] × [0, ln a] (a > 1) is cut by y = ln x into the
    region under the curve from 1 to a (area ∫₁ᵃ ln x dx) and the region
    left of it (the points with x ≤ eʸ).  Folded over the line y = x, the
    left region becomes the region under y = eˣ over [0, ln a], where eˣ
    rises from 1 to a: its area is a − 1.  Folded back, it fills the
    rectangle again: a·ln a = ∫₁ᵃ ln x dx + (a − 1).
    (Uses ∫₀ˣ eᵗ dt = eˣ − 1.)"""

    def construct(self):
        k = 1.76
        O = np.array([-5.6, -2.3, 0.0])

        def S(x, y):
            return O + k * np.array([x, y, 0.0])

        a = 3.2
        La = float(np.log(a))
        x_lo, x_hi = 0.75, a + 0.35
        ys = np.linspace(0, La, 200)
        diag_dir = normalize(np.array([1.0, 1.0, 0.0]))

        A1 = [S(1, 0)] + [S(np.exp(y), y) for y in ys] + [S(a, 0)]
        A2 = [S(np.exp(y), y) for y in ys] + [S(0, La), S(0, 0)]
        rect = [S(0, 0), S(a, 0), S(a, La), S(0, La)]
        EX = [S(y, np.exp(y)) for y in ys] + [S(La, 0), S(0, 0)]   # under eˣ on [0, ln a]

        # ---- the claims
        check(_tiles_exactly([[p[:2] for p in A1], [p[:2] for p in A2]],
                             [p[:2] for p in rect], 100),
              "the two regions tile the a × ln a rectangle")
        check(_same_pts([_mirror(p, S(0, 0), S(1, 1)) for p in A2], EX),
              "the left region, mirrored in y = x, is the region under eˣ")
        IL = _simpson(np.log, 1, a, 2000)
        check(abs(IL - (a * La - a + 1)) < 1e-9, "∫₁ᵃ ln x dx = a ln a − a + 1")
        check(abs(_simpson(np.exp, 0, La) - (a - 1)) < 1e-9,
              "the area under eˣ over [0, ln a] is a − 1")
        check(abs(abs(area([p[:2] for p in A2])) / k ** 2 - (a - 1)) < 1e-4,
              "the left region has area a − 1 (polygon)")

        axes = VGroup(Line(S(-0.12, 0), S(x_hi + 0.12, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -0.3), S(0, a + 0.1), color=GREY_B, stroke_width=2))
        cx = np.concatenate([np.geomspace(x_lo, 1.2, 40)[:-1], np.linspace(1.2, x_hi, 160)])
        cpts = [S(x, np.log(x)) for x in cx]
        curve = _curve(cpts, YELLOW_B, 5)
        l_ln = tag("y = ln x", 28, YELLOW_B).next_to(cpts[-1], RIGHT, buff=0.18)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_ln), run_time=1.1)

        tka, tk1 = _tick(S(a, 0)), _tick(S(1, 0))
        l_a = tag("a", 28).next_to(S(a, 0), DOWN, buff=0.16)
        l_1 = tag("1", 26).next_to(S(1, 0), DOWN, buff=0.16).shift(RIGHT * 0.16)
        l_1.align_to(l_a, UP)
        frame = Polygon(*rect, stroke_color=WHITE, stroke_width=4)
        # the rectangle's height, beside its right side
        l_La = tag("ln a", 26).next_to(Line(S(a, 0), S(a, La)), RIGHT, buff=0.16)
        self.play(Create(tka), Create(tk1), FadeIn(l_a), FadeIn(l_1), run_time=0.7)
        self.play(Create(frame), FadeIn(l_La), run_time=0.9)

        r1 = mk(A1, BLUE_D, stroke_width=0)
        l_I = _row(_int("1", "a", 26, WHITE), tag("ln x dx", 26), buff=0.06)
        l_I.move_to(S(2.45, 0.3))
        self.add(r1)
        self.bring_to_back(r1)
        self.play(FadeIn(r1), FadeIn(l_I), run_time=0.9)
        r2 = mk(A2, ORANGE, 0.85, stroke_width=0)
        self.add(r2)
        self.bring_to_back(r2)
        self.play(FadeIn(r2), run_time=0.8)
        self.bring_to_front(curve, frame)
        self.hold(0.5)

        # ---- the left region, turned over y = x: the region under eˣ
        diag = DashedLine(S(0, 0), S(2.2, 2.2), color=WHITE, stroke_width=2.5,
                          dash_length=0.1)
        self.play(Create(diag), run_time=0.5)
        self.bring_to_front(r2)
        self.play(Rotate(r2, angle=PI, axis=diag_dir, about_point=S(0, 0)), run_time=1.6)
        check(_same_pts(_verts(r2), EX), "the fold lands on the region under eˣ")
        tky1, tkya = _tick(S(0, 1), RIGHT), _tick(S(0, a), RIGHT)
        l_y1 = tag("1", 26).next_to(S(0, 1), LEFT, buff=0.16)
        l_ya = tag("a", 28).next_to(S(0, a), LEFT, buff=0.16)
        l_ex = _supline([("y = e", False), ("x", True)], 28, ORANGE)
        l_ex.next_to(S(La, a), RIGHT, buff=0.2)
        l_c = tag("a − 1", 26).move_to(S(0.8, 1.52))
        self.play(Create(tky1), Create(tkya), FadeIn(l_y1), FadeIn(l_ya),
                  FadeIn(l_ex), run_time=0.8)
        self.play(FadeIn(l_c), run_time=0.6)
        self.hold(1.0)

        segsF = (_segs(cpts) + _segs([S(-0.12, 0), S(x_hi + 0.12, 0)])
                 + _segs([S(0, -0.3), S(0, a + 0.1)]) + _segs(rect + [rect[0]])
                 + _segs(EX + [EX[0]]) + _segs([S(0, 0), S(2.2, 2.2)]))
        _labels_ok([l_ln, l_a, l_1, l_La, l_I, l_y1, l_ya, l_ex, l_c], segsF,
                   "H17 (folded)")
        for c in (l_c.get_corner(UL), l_c.get_corner(UR), l_c.get_corner(DL),
                  l_c.get_corner(DR)):
            check(_pip(c, EX), "a − 1 inside the region under eˣ")
            check(not _pip(c, rect), "a − 1 clear of the rectangle")

        # ---- turned back, it fills the rectangle again
        l_c2 = tag("a − 1", 28).move_to(S(0.6, 0.62))
        self.play(Rotate(r2, angle=-PI, axis=diag_dir, about_point=S(0, 0)),
                  Transform(l_c, l_c2), run_time=1.5)
        check(_same_pts(_verts(r2), A2), "folded back onto the left region")
        self.play(FadeOut(diag), FadeOut(l_ex), FadeOut(tky1), FadeOut(tkya),
                  FadeOut(l_y1), FadeOut(l_ya), run_time=0.6)
        self.bring_to_front(curve, frame, l_c)

        row = _row(_supline([("a · ln a  =", False)], 28, WHITE),
                   _int("1", "a", 28, BLUE_B), tag("ln x dx", 28, BLUE_B),
                   tag("+", 28), tag("(a − 1)", 28, ORANGE), buff=0.14)
        row.move_to([3.5, 1.7, 0.0])
        self.play(FadeIn(row), run_time=0.9)

        segs = (_segs(cpts) + _segs([S(-0.12, 0), S(x_hi + 0.12, 0)])
                + _segs([S(0, -0.3), S(0, a + 0.1)]) + _segs(rect + [rect[0]]))
        labels = [l_ln, l_a, l_1, l_La, l_I, l_c, row]
        _labels_ok(labels, segs, "H17")
        for lab, poly in ((l_I, A1), (l_c, A2)):
            for c in (lab.get_corner(UL), lab.get_corner(UR), lab.get_corner(DL),
                      lab.get_corner(DR)):
                check(_pip(c, poly), "region label inside its region")
        content = [axes, curve, frame, r1, r2, *labels]
        _safe(*content)
        cap = _cap(_row(_int("1", "a", 32), tag("ln x dx  =  a ln a − a + 1", 32,
                                                YELLOW_B), buff=0.1))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)
