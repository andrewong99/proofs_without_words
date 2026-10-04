# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w1c.py — calculus, circle and figurate scenes:
#     H13, H8, E11, H10, H14, H15, G19, H20, B42, D14.
#
# The "infinitesimal" scenes (H13, H8, H10, H20) draw the finite increment
# exactly — strips plus corner — and then let it shrink.  Beside the true
# figure a copy of the increment is magnified across its thin direction
# (÷ dx): that copy keeps a visible size, and its corner part visibly
# shrinks to nothing, which is the limit the caption states.
#
# Every scene asserts its key invariants with check(...), so a wrong
# construction fails the render instead of rendering quietly.

import re


# ------------------------------------------------------------ private helpers

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


def _cap(group):
    """Place a composite formula where caption() puts its Text: centred,
    along the bottom edge."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.set_x(0.0)
    group.to_edge(DOWN, buff=0.3)
    return group


def _mt(spec, size=30, color=WHITE):
    """Math text with raised exponents in ordinary layout: '^{...}' marks an
    exponent ('e^{x} − 1').  Every piece sits on one baseline; exponents are
    smaller and raised, so no superscript letters (missing on some fonts)
    are needed."""
    sp = (Text("M M", font_size=size).width - Text("MM", font_size=size).width)
    cap = Text("M", font_size=size).height
    out = VGroup()
    x = 0.0
    for p in re.split(r"(\^\{[^}]*\})", spec):
        if not p:
            continue
        if p.startswith("^{"):
            t = Text("M" + p[2:-1], font_size=size * 0.64, color=color)
            m = t.submobjects[0]
            base = m.get_bottom()[1]
            t.remove(m)
            t.shift(np.array([x + 0.025 - t.get_left()[0],
                              0.48 * cap - base, 0.0]))
            x = t.get_right()[0] + 0.02
            out.add(t)
            continue
        lead = len(p) - len(p.lstrip(" "))
        trail = len(p) - len(p.rstrip(" "))
        core = p.strip(" ")
        x += lead * sp
        if core:
            t = Text("M" + core, font_size=size, color=color)
            m = t.submobjects[0]
            base = m.get_bottom()[1]
            t.remove(m)
            t.shift(np.array([x - t.get_left()[0], -base, 0.0]))
            x = t.get_right()[0]
            out.add(t)
        x += trail * sp
    return out


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


def _dim(p, q, color=WHITE, tick=0.12, width=3):
    """A dimension bar from p to q with end ticks across it."""
    p, q = to3(p), to3(q)
    d = q - p
    n = np.array([-d[1], d[0], 0.0]) / max(np.linalg.norm(d), 1e-9) * tick
    return VGroup(Line(p, q, color=color, stroke_width=width),
                  Line(p - n, p + n, color=color, stroke_width=width),
                  Line(q - n, q + n, color=color, stroke_width=width))


def _below(cap, *mobs, gap=0.03):
    """Fail the render if the caption reaches up into any content above it
    (only content that overlaps it horizontally counts)."""
    c0, c1 = cap.get_critical_point(DL), cap.get_critical_point(UR)
    for m in mobs:
        m0, m1 = m.get_critical_point(DL), m.get_critical_point(UR)
        if m1[0] < c0[0] or m0[0] > c1[0]:
            continue
        check(c1[1] + gap <= m0[1], f"caption clear of {type(m).__name__} "
              f"(caption top {c1[1]:.2f}, content bottom {m0[1]:.2f})")


def _simpson(g, lo, hi, n=400):
    """Composite Simpson rule (n even)."""
    t = np.linspace(lo, hi, n + 1)
    y = g(t)
    return (hi - lo) / (3 * n) * (y[0] + y[-1] + 4 * y[1:-1:2].sum()
                                  + 2 * y[2:-1:2].sum())


# ========================================================== H. CALCULUS

class H13_FundamentalTheorem(Board):
    """A(x) = area under f from a to x.  Moving x to x + dx adds the strip
    under f over [x, x + dx]: a rectangle f(x)·dx plus a cap that fits in
    the box dx × (f(x + dx) − f(x)).  Magnified across by 1/dx, the strip
    becomes a column of height f(x) topped by the stretched cap; as dx → 0
    the cap's box collapses (f is continuous), so ΔA/dx → f(x).
    (Drawn with f increasing near x; in general the strip lies between the
    rectangles of the least and greatest values of f on [x, x + dx].)"""

    def construct(self):
        def f(t):
            return 1.6 + 0.9 * np.sin(0.75 * (t - 2.2)) + 0.15 * t

        def fp(t):
            return 0.675 * np.cos(0.75 * (t - 2.2)) + 0.15

        a, x = 0.8, 3.0
        h0, h1 = 1.0, 0.012
        kx, ky = 1.35, 1.62
        org = np.array([-6.1, -2.45, 0.0])
        X0, W = 2.2, 2.4                     # the magnified strip [X0, X0 + W]

        def P(t, y):
            return org + np.array([kx * t, ky * y, 0.0])

        def Q(s, y):                         # s in [0, 1] across the strip
            return np.array([X0 + W * s, org[1] + ky * y, 0.0])

        # ---- the claims
        check(bool(np.all(fp(np.linspace(x, x + h0, 401)) > 0)),
              "f increases on [x, x + dx]")
        for h in np.geomspace(h1, h0, 15):
            dA = _simpson(f, x, x + h)
            check(f(x) * h <= dA <= f(x + h) * h,
                  "f(x)·dx ≤ ΔA ≤ f(x + dx)·dx")
        check(abs(_simpson(f, x, x + h1) / h1 - f(x)) < 0.01 * f(x),
              "ΔA/dx has reached f(x)")
        t_end = 4.75
        check(bool(np.all(f(np.linspace(x + h0, t_end, 200)[1:]) > f(x + h0))),
              "the guides pass under the curve")

        N = 48

        def strip_pts(h):
            ts = np.linspace(x, x + h, N)
            rect = [(x, 0.0), (x + h, 0.0), (x + h, f(x)), (x, f(x))]
            cap = [(t, f(t)) for t in ts] + [(x + h, f(x))]
            return rect, cap

        def strip(h, magnified=False):
            """The strip over [x, x + h]: rectangle f(x)·h and the cap above
            it — true size, or magnified across onto [X0, X0 + W]."""
            rect, cap = strip_pts(h)
            if magnified:
                tr = [Q((p[0] - x) / h, p[1]) for p in rect]
                tc = [Q((p[0] - x) / h, p[1]) for p in cap]
            else:
                tr = [P(*p) for p in rect]
                tc = [P(*p) for p in cap]
            return VGroup(mk(tr, TEAL_D, stroke_width=1.5),
                          mk(tc, ORANGE, 0.9, stroke_width=1.5))

        # numeric: the magnified strip has area ΔA·W·ky/dx on screen
        r_, c_ = strip(h0, magnified=True)
        scr_area = abs(area([v[:2] for v in r_.get_vertices()])) + \
            abs(area([v[:2] for v in c_.get_vertices()]))
        check(abs(scr_area - _simpson(f, x, x + h0) * W * ky / h0) < 2e-3,
              "magnified strip = ΔA ÷ dx (scaled)")

        # ---- figure
        axes = VGroup(Line(P(0, 0), P(5.05, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 3.5), color=GREY_B, stroke_width=2))
        curve = ParametricFunction(lambda t: P(t, f(t)), t_range=[0.25, t_end],
                                   color=YELLOW_B, stroke_width=5)
        # above the curve's end: the guides to the magnified strip pass below
        l_f = tag("f", 28, YELLOW_B).next_to(P(t_end, f(t_end)), UR, buff=0.06)
        ta = Line(P(a, 0) + DOWN * 0.08, P(a, 0) + UP * 0.08, color=GREY_B)
        tx = Line(P(x, 0) + DOWN * 0.08, P(x, 0) + UP * 0.08, color=GREY_B)
        l_a = tag("a", 28).next_to(P(a, 0), DOWN, buff=0.16)
        l_x = tag("x", 28).next_to(P(x, 0), DOWN, buff=0.16)
        l_a.align_to(l_x, DOWN)

        self.play(Create(axes), run_time=0.7)
        self.play(Create(curve), FadeIn(l_f), run_time=1.2)
        self.play(Create(ta), Create(tx), FadeIn(l_a), FadeIn(l_x), run_time=0.6)

        # sweep the area from a to x
        sw = ValueTracker(a + 1e-3)

        def region(t1):
            ts = np.linspace(a, t1, 80)
            return mk([P(a, 0)] + [P(t, f(t)) for t in ts] + [P(t1, 0)],
                      BLUE_D, stroke_width=1.5)

        reg = always_redraw(lambda: region(sw.get_value()))
        self.add(reg)
        self.bring_to_front(curve)
        self.play(sw.animate.set_value(x), run_time=1.8, rate_func=linear)
        reg.clear_updaters()
        self.remove(reg)
        reg = region(x)
        self.add(reg)
        self.bring_to_front(curve)
        l_A = _row(_int("a", "x", 28, WHITE), tag("f(t) dt", 28),
                   buff=0.08).move_to(P(1.95, 0.8))
        self.play(FadeIn(l_A), run_time=0.6)

        # ---- the strip over [x, x + dx]: rectangle f(x)·dx plus a cap
        lt = ValueTracker(np.log(h0))

        def hh():
            return float(np.exp(lt.get_value()))

        st = strip(h0)
        self.play(FadeIn(st), run_time=0.9)
        self.bring_to_front(curve)
        ytop = f(x + h0) + 0.32

        def bracket():
            h = hh()
            return _dim(P(x, ytop), P(x + h, ytop), WHITE, 0.09, 2.5)

        def dx_label():
            return tag("dx", 26).next_to(bracket(), UP, buff=0.1)

        br = bracket()
        l_dx = dx_label()
        self.play(Create(br), FadeIn(l_dx), run_time=0.6)
        self.hold(0.4)

        # ---- magnified across by 1/dx: same heights, width W
        base_p = Line(Q(0, 0) + LEFT * 0.25, Q(1, 0) + RIGHT * 0.25,
                      color=GREY_B, stroke_width=2)

        def guides():
            h = hh()
            return VGroup(
                DashedLine(P(x + h, f(x)), Q(0, f(x)), color=GREY_B,
                           stroke_width=1.6, dash_length=0.08),
                DashedLine(P(x + h, f(x + h)), Q(0, f(x + h)), color=GREY_B,
                           stroke_width=1.6, dash_length=0.08))

        gd = guides()
        mag = st.copy()
        self.add(mag)
        self.play(Transform(mag, strip(h0, magnified=True)), Create(base_p),
                  Create(gd), run_time=1.6)
        l_div = tag("÷ dx", 28).next_to(Q(0.5, f(x + h0)), UP, buff=0.32)

        def box():
            h = hh()
            return DashedVMobject(Polygon(Q(0, f(x)), Q(1, f(x)), Q(1, f(x + h)),
                                          Q(0, f(x + h)), stroke_color=WHITE,
                                          stroke_width=2), num_dashes=28)

        bx = box()
        xd = X0 + W + 0.32
        dimf = _dim([xd, Q(0, 0)[1], 0], [xd, Q(0, f(x))[1], 0], TEAL_B, 0.1, 3)
        l_fx = tag("f(x)", 28, TEAL_B).next_to(dimf, RIGHT, buff=0.14)
        self.play(FadeIn(l_div), Create(bx), run_time=0.7)
        self.play(Create(dimf), FadeIn(l_fx), run_time=0.7)
        self.hold(0.6)

        # ---- dx → 0: the cap's box collapses onto the column's top
        st_r = always_redraw(lambda: strip(hh()))
        mag_r = always_redraw(lambda: strip(hh(), magnified=True))
        br_r = always_redraw(bracket)
        gd_r = always_redraw(guides)
        bx_r = always_redraw(box)
        l_dx_r = always_redraw(dx_label)
        self.remove(st, mag, br, gd, bx, l_dx)
        self.add(st_r, mag_r, br_r, gd_r, bx_r, l_dx_r)
        self.bring_to_front(curve)
        self.play(lt.animate.set_value(np.log(h1)), run_time=4.2, rate_func=smooth)
        for m in (st_r, mag_r, br_r, gd_r, bx_r, l_dx_r):
            m.clear_updaters()
        l_dx = l_dx_r
        cap_h = (f(x + h1) - f(x)) * ky
        check(cap_h < 0.02, "the magnified cap has collapsed")

        content = VGroup(axes, curve, l_f, l_a, l_x, reg, l_A, st_r, br_r, l_dx,
                         base_p, gd_r, mag_r, l_div, bx_r, dimf, l_fx)
        _safe(*content)
        _apart(l_f, l_a, l_x, l_A, l_dx, l_div, l_fx)
        cap = _cap(_row(tag("d/dx", 34, YELLOW_B), _int("a", "x", 34),
                        tag("f(t) dt   =   f(x)", 34, YELLOW_B), buff=0.16))
        _below(cap, l_a, l_x, axes)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


def _box(x0, y0, x1, y1, color, op=FILL, sw=2.0):
    """Axis-aligned filled rectangle between screen corners (x0,y0)-(x1,y1)."""
    return mk([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, op,
              stroke_width=sw)


class H8_SquareDerivative(Board):
    """Grow the x-square by dx: two strips x·dx and a corner dx·dx.
    Laid end to end (the right strip turned a quarter) they make one bar dx
    thick and x + x + dx long, so Δ(x²) = (2x + dx)·dx exactly.  Magnified
    across (÷ dx) the bar keeps the lengths x, x and dx; as dx → 0 the corner's
    piece shrinks to nothing and the bar's length tends to 2x."""

    def construct(self):
        x, d0, d1 = 2.5, 0.5, 0.012
        k = 1.32
        S0 = np.array([-6.0, -2.0, 0.0])
        XB, YB, W = -0.85, -0.95, 1.2           # bar: left end, bottom, thickness

        def P(a, b):
            return S0 + k * np.array([a, b, 0.0])

        def B(a, b):                            # a: math units along, b: screen
            return np.array([XB + k * a, YB + b, 0.0])

        # ---- the claims
        for d in (d0, 0.2, d1):
            parts = [x * d, x * d, d * d]
            check(abs(sum(parts) - ((x + d) ** 2 - x * x)) < 1e-12,
                  "strips + corner = (x + dx)² − x²")
            check(abs(sum(parts) - (2 * x + d) * d) < 1e-12, "= (2x + dx)·dx")
        check(d1 * k < 0.02, "the corner's piece has shrunk to nothing")

        def square_inc(d):
            R = _box(*P(x, 0)[:2], *P(x + d, x)[:2], TEAL_D)
            T = _box(*P(0, x)[:2], *P(x, x + d)[:2], TEAL_D)
            C = _box(*P(x, x)[:2], *P(x + d, x + d)[:2], ORANGE, 0.9)
            return R, T, C

        def bar(d, thick):
            return (_box(*B(0, 0)[:2], *B(x, thick)[:2], TEAL_D),
                    _box(*B(x, 0)[:2], *B(2 * x, thick)[:2], TEAL_D),
                    _box(*B(2 * x, 0)[:2], *B(2 * x + d, thick)[:2], ORANGE, 0.9))

        sq = _box(*P(0, 0)[:2], *P(x, x)[:2], BLUE_D)
        l_sq = tag("x²", 36).move_to(P(x / 2, x / 2))
        l_xb = tag("x", 28).next_to(P(x / 2, 0), DOWN, buff=0.16)
        l_xl = tag("x", 28).next_to(P(0, x / 2), LEFT, buff=0.16)
        self.play(FadeIn(sq), FadeIn(l_sq), FadeIn(l_xb), FadeIn(l_xl), run_time=1.1)
        self.hold(0.3)

        # ---- grow by dx: two strips and a corner
        R, T, C = square_inc(d0)
        self.play(FadeIn(R, shift=RIGHT * 0.25), FadeIn(T, shift=UP * 0.25),
                  run_time=0.9)
        self.play(FadeIn(C, scale=0.6), run_time=0.6)

        l_dxb = tag("dx", 26).next_to(P(x + d0 / 2, 0), DOWN, buff=0.16)
        l_dxb.align_to(l_xb, DOWN)
        l_dxl = tag("dx", 26).next_to(P(0, x + d0 / 2), LEFT, buff=0.16)
        l_sr = tag("x·dx", 26).next_to(P(x + d0, x / 2), RIGHT, buff=0.15)
        l_st = tag("x·dx", 26).next_to(P(x / 2, x + d0), UP, buff=0.14)
        l_c = tag("dx²", 26).next_to(P(x + d0, x + d0), UR, buff=0.08)
        self.play(FadeIn(l_dxb), FadeIn(l_dxl), run_time=0.5)
        self.play(FadeIn(l_sr), FadeIn(l_st), FadeIn(l_c), run_time=0.7)
        self.hold(0.6)

        # ---- end to end: the top strip slides, the right strip turns a
        # quarter, the corner slides — one bar dx thick
        cT, cR, cC = T.copy(), R.copy(), C.copy()
        self.add(cT, cR, cC)
        tgt = bar(d0, k * d0)
        self.play(_rigid(cT, shift=tgt[0].get_center() - cT.get_center()),
                  _rigid(cR, angle=-PI / 2,
                         shift=tgt[1].get_center() - cR.get_center()),
                  _rigid(cC, shift=tgt[2].get_center() - cC.get_center()),
                  run_time=1.9)
        for got, want in zip((cT, cR, cC), tgt):
            g = sorted(map(tuple, np.round(got.get_vertices()[:, :2], 7)))
            w_ = sorted(map(tuple, np.round(want.get_vertices()[:, :2], 7)))
            check(np.allclose(g, w_, atol=1e-6), "piece lands in the bar")
        self.remove(cT, cR, cC)
        pieces = VGroup(*tgt)
        self.add(pieces)
        self.hold(0.3)

        # ---- magnified across: ÷ dx
        l_div = tag("÷ dx", 28).next_to(B(0, W), UP, buff=0.22).align_to(
            B(0, 0), LEFT)
        self.play(pieces.animate.stretch(W / (k * d0), 1, about_edge=DOWN),
                  FadeIn(l_div), run_time=1.1)
        check(abs(pieces.height - W) < 1e-6 and abs(pieces.get_bottom()[1] - YB)
              < 1e-6, "the bar is magnified across, about its base")
        lx1 = tag("x", 28).move_to(B(x / 2, W / 2))
        lx2 = tag("x", 28).move_to(B(1.5 * x, W / 2))
        ldx = tag("dx", 24).move_to(B(2 * x + d0 / 2, W / 2))
        br2x = _dim(B(0, -0.28), B(2 * x, -0.28), YELLOW_B, 0.1, 3)
        l2x = tag("2x", 30, YELLOW_B).next_to(br2x, DOWN, buff=0.14)
        self.play(FadeIn(lx1), FadeIn(lx2), FadeIn(ldx), run_time=0.6)
        self.play(Create(br2x), FadeIn(l2x), run_time=0.7)
        self.hold(0.6)

        # ---- dx → 0: the strips thin, the corner and its piece vanish
        lg = ValueTracker(np.log(d0))
        sq_inc = always_redraw(lambda: VGroup(*square_inc(np.exp(lg.get_value()))))
        bar_r = always_redraw(lambda: VGroup(*bar(np.exp(lg.get_value()), W)))
        self.remove(R, T, C, pieces)
        self.add(sq_inc, bar_r)
        self.bring_to_front(lx1, lx2, ldx)

        def follow(lbl, fn):
            lbl.add_updater(lambda m: fn(m, float(np.exp(lg.get_value()))))

        follow(l_dxb, lambda m, d: m.set_x(P(x + d / 2, 0)[0]))
        follow(l_dxl, lambda m, d: m.set_y(P(0, x + d / 2)[1]))
        follow(l_sr, lambda m, d: m.next_to(P(x + d, x / 2), RIGHT, buff=0.15))
        follow(l_st, lambda m, d: m.next_to(P(x / 2, x + d), UP, buff=0.14))
        follow(l_c, lambda m, d: m.next_to(P(x + d, x + d), UR, buff=0.08))
        self.play(FadeOut(ldx), run_time=0.3)
        self.play(lg.animate.set_value(np.log(d1)), run_time=4.2, rate_func=smooth)
        for m in (sq_inc, bar_r, l_dxb, l_dxl, l_sr, l_st, l_c):
            m.clear_updaters()
        _apart(l_sq, l_xb, l_xl, l_dxb, l_dxl, l_sr, l_st, l_c, l_div, lx1, lx2,
               l2x)
        self.play(FadeOut(l_c), run_time=0.4)       # the corner's term is gone

        content = VGroup(sq, l_sq, l_xb, l_xl, sq_inc, l_dxb, l_dxl, l_sr, l_st,
                         bar_r, l_div, lx1, lx2, br2x, l2x)
        _safe(*content)
        cap = caption("d(x²)/dx  =  2x", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H10_ProductRule(Board):
    """A u × v rectangle grows to (u + du) × (v + dv): a strip v·du on the
    right, a strip u·dv on top, and the corner du·dv.  Group the corner with
    the right strip: Δ(uv) = (v + dv)·du + u·dv exactly.  Each group is
    magnified across its thin direction (÷ du, ÷ dv) so it stays visible;
    as du, dv → 0 together, the corner's share dv of the right column
    vanishes, leaving v·du + u·dv."""

    def construct(self):
        u, v = 3.0, 2.0
        du0, dv0, s1 = 0.55, 0.45, 0.03
        k = 1.45
        R0 = np.array([-4.2, -2.35, 0.0])
        XC, YR, W = 2.7, 1.85, 1.1        # magnified column x, magnified row y

        def P(a, b):
            return R0 + k * np.array([a, b, 0.0])

        # ---- the claims
        for s in (1.0, 0.3, s1):
            du, dv = du0 * s, dv0 * s
            check(abs((u + du) * (v + dv) - u * v
                      - ((v + dv) * du + u * dv)) < 1e-12,
                  "Δ(uv) = (v + dv)·du + u·dv")
        check(dv0 * s1 * k < 0.02, "the corner's share of the column vanishes")

        def inc(s):
            du, dv = du0 * s, dv0 * s
            Rs = _box(*P(u, 0)[:2], *P(u + du, v)[:2], TEAL_D)
            Ts = _box(*P(0, v)[:2], *P(u, v + dv)[:2], GREEN_D)
            Cs = _box(*P(u, v)[:2], *P(u + du, v + dv)[:2], ORANGE, 0.9)
            return Rs, Ts, Cs

        def column(s):                     # (v + dv)·du, magnified ÷ du
            dv = dv0 * s
            y0, y1, y2 = P(0, 0)[1], P(0, v)[1], P(0, v + dv)[1]
            return (_box(XC, y0, XC + W, y1, TEAL_D),
                    _box(XC, y1, XC + W, y2, ORANGE, 0.9))

        def row():                         # u·dv, magnified ÷ dv
            return _box(P(0, 0)[0], YR, P(u, 0)[0], YR + W, GREEN_D)

        def links(s):
            du, dv = du0 * s, dv0 * s
            kw = dict(color=GREY_B, stroke_width=1.6, dash_length=0.08)
            return VGroup(
                DashedLine(P(u + du, v + dv), [XC, P(0, v + dv)[1], 0], **kw),
                DashedLine(P(u + du, 0), [XC, P(0, 0)[1], 0], **kw),
                DashedLine(P(0, v + dv), [P(0, 0)[0], YR, 0], **kw),
                DashedLine(P(u, v + dv), [P(u, 0)[0], YR, 0], **kw))

        rect = _box(*P(0, 0)[:2], *P(u, v)[:2], BLUE_D)
        l_uv = tag("uv", 36).move_to(P(u / 2, v / 2))
        l_u = tag("u", 28).next_to(P(u / 2, 0), DOWN, buff=0.16)
        l_v = tag("v", 28).next_to(P(0, v / 2), LEFT, buff=0.16)
        self.play(FadeIn(rect), FadeIn(l_uv), FadeIn(l_u), FadeIn(l_v), run_time=1.1)
        self.hold(0.3)

        # ---- grow by du and dv
        Rs, Ts, Cs = inc(1.0)
        self.play(FadeIn(Rs, shift=RIGHT * 0.25), FadeIn(Ts, shift=UP * 0.25),
                  run_time=0.9)
        self.play(FadeIn(Cs, scale=0.6), run_time=0.5)
        l_du = tag("du", 26).next_to(P(u + du0 / 2, 0), DOWN, buff=0.16)
        l_du.align_to(l_u, DOWN)
        l_dv = tag("dv", 26).next_to(P(0, v + dv0 / 2), LEFT, buff=0.16)
        l_R = tag("v·du", 26, TEAL_B).next_to(P(u + du0, v / 2), RIGHT, buff=0.15)
        l_T = tag("u·dv", 26, GREEN_B).next_to(P(u / 2, v + dv0), UP, buff=0.14)
        l_C = tag("du·dv", 24, ORANGE).next_to(P(u + du0, v + dv0), UR, buff=0.14)
        self.play(FadeIn(l_du), FadeIn(l_dv), run_time=0.5)
        self.play(FadeIn(l_R), FadeIn(l_T), FadeIn(l_C), run_time=0.7)
        self.hold(0.6)

        # ---- magnified across: the right column ÷ du, the top strip ÷ dv
        cR, cC, cT = Rs.copy(), Cs.copy(), Ts.copy()
        self.add(cR, cC, cT)
        col_t, row_t = column(1.0), row()
        lk = links(1.0)
        self.play(Transform(cR, col_t[0]), Transform(cC, col_t[1]),
                  Transform(cT, row_t), Create(lk), run_time=1.7)
        for got, want in ((cR, col_t[0]), (cC, col_t[1]), (cT, row_t)):
            check(close(got.get_vertices(), want.get_vertices(), 1e-6),
                  "magnified pieces in place")
        dvc = _dim([XC + W + 0.25, P(0, 0)[1], 0], [XC + W + 0.25, P(0, v)[1], 0],
                   TEAL_B, 0.1, 3)
        l_cv = tag("v", 28, TEAL_B).next_to(dvc, RIGHT, buff=0.14)

        def dv_dim(s):
            dv = dv0 * s
            return _dim([XC + W + 0.25, P(0, v)[1], 0],
                        [XC + W + 0.25, P(0, v + dv)[1], 0], ORANGE, 0.1, 3)

        dvd = dv_dim(1.0)
        l_cdv = tag("dv", 26, ORANGE).next_to(dvd, RIGHT, buff=0.14)
        dru = _dim([P(0, 0)[0], YR + W + 0.25, 0], [P(u, 0)[0], YR + W + 0.25, 0],
                   GREEN_B, 0.1, 3)
        l_ru = tag("u", 28, GREEN_B).next_to(dru, UP, buff=0.12)
        l_ddu = tag("÷ du", 28).next_to(col_t[1], UP, buff=0.2)
        l_ddv = tag("÷ dv", 28).next_to(row_t, RIGHT, buff=0.25)
        self.play(FadeIn(l_ddu), FadeIn(l_ddv), run_time=0.5)
        self.play(Create(dvc), FadeIn(l_cv), Create(dvd), FadeIn(l_cdv),
                  Create(dru), FadeIn(l_ru), run_time=0.9)
        self.hold(0.7)

        # ---- du, dv → 0 together: the corner and its share dv vanish
        lg = ValueTracker(0.0)

        def s_now():
            return float(np.exp(lg.get_value()))

        inc_r = always_redraw(lambda: VGroup(*inc(s_now())))
        col_r = always_redraw(lambda: VGroup(*column(s_now())))
        lk_r = always_redraw(lambda: links(s_now()))
        dvd_r = always_redraw(lambda: dv_dim(s_now()))
        self.remove(Rs, Ts, Cs, cR, cC, lk, dvd)
        self.add(inc_r, col_r, lk_r, dvd_r)
        self.bring_to_front(l_ddu, l_cdv)

        def follow(lbl, fn):
            lbl.add_updater(lambda m: fn(m, s_now()))

        follow(l_du, lambda m, s: m.set_x(P(u + du0 * s / 2, 0)[0]))
        follow(l_dv, lambda m, s: m.set_y(P(0, v + dv0 * s / 2)[1]))
        follow(l_R, lambda m, s: m.next_to(P(u + du0 * s, v / 2), RIGHT, buff=0.15))
        follow(l_T, lambda m, s: m.next_to(P(u / 2, v + dv0 * s), UP, buff=0.14))
        follow(l_C, lambda m, s: m.next_to(P(u + du0 * s, v + dv0 * s), UR,
                                           buff=0.14))
        follow(l_ddu, lambda m, s: m.next_to(
            np.array([XC + W / 2, P(0, v + dv0 * s)[1], 0]), UP, buff=0.2))
        follow(l_cdv, lambda m, s: m.next_to(dv_dim(s), RIGHT, buff=0.14))
        self.play(lg.animate.set_value(np.log(s1)), run_time=4.4, rate_func=smooth)
        for m in (inc_r, col_r, lk_r, dvd_r, l_du, l_dv, l_R, l_T, l_C, l_ddu,
                  l_cdv):
            m.clear_updaters()
        _apart(l_uv, l_u, l_v, l_du, l_dv, l_R, l_T, l_C, l_cv, l_ru, l_ddu, l_ddv)
        # the corner is gone, and so is its term
        self.play(FadeOut(l_cdv), FadeOut(dvd_r), FadeOut(l_C), run_time=0.5)

        content = VGroup(rect, l_uv, l_u, l_v, inc_r, l_du, l_dv, l_R, l_T,
                         col_r, cT, lk_r, dvc, l_cv, dru, l_ru, l_ddu, l_ddv)
        _safe(*content)
        cap = caption("d(uv)  =  u·dv  +  v·du", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


def _unroll(rho, phi, mu, r):
    """Unrolling a disc of radius r cut along its upward radius (the map of
    the reviewed circle-area scene, copied).

    (rho, phi): radius, and angle along that circle measured from its
    lowest point (phi in [-pi, pi]; +-pi are the two lips of the cut).
    mu = 1 is the disc, mu = 0 the flat strip.  At stage mu every circle is
    an arc about the common centre (0, c), c = r(1-mu)/mu, of radius c + rho
    and length 2*pi*rho, its lowest point (0, -rho) fixed.  The map is
    area-preserving at every stage: (c + rho) * d(psi)/d(phi) = rho.
    At mu = 0 a circle of radius rho is the segment |x| <= pi*rho, y = -rho.
    """
    rho = np.asarray(rho, float)
    phi = np.asarray(phi, float)
    den = r * (1.0 - mu) + rho * mu
    safe = np.where(den > 0, den, 1.0)
    psi = np.where(den > 0, phi * rho * mu / safe, 0.0)
    x = phi * rho * np.sinc(psi / PI)
    y = -rho + 0.5 * phi * rho * psi * np.sinc(psi / TAU) ** 2
    return np.stack([x, y], -1)


class H20_DiscGrowth(Board):
    """The disc of radius r grows by a ring of width dr.  Cut along a radius
    and unrolled (every circle keeps its length, the map keeps area) the ring
    is a trapezoid between its two circumferences 2πr and 2π(r + dr), height
    dr: a rectangle 2πr·dr plus two corner triangles, together π·dr².
    Magnified across (÷ dr) the rectangle stays 2πr long while the corner
    triangles' bases π·dr shrink to nothing as dr → 0."""

    def construct(self):
        r, d0, d1 = 1.45, 0.42, 0.012
        O = np.array([0.0, 1.25, 0.0])     # low enough for the lips mid-unroll
        YM, W = -2.75, 0.95                 # magnified strip: bottom, height
        g = 0.12                            # the unrolled copy clears the ring
        NA, NS = 241, 9

        def ring_xy(r1, r2, mu):
            ph = np.linspace(-PI, PI, NA)
            rr = np.linspace(r2, r1, NS)
            parts = [_unroll(r2, ph, mu, r + d0),
                     _unroll(rr, np.full(NS, PI), mu, r + d0)[1:],
                     _unroll(r1, ph[::-1], mu, r + d0)[1:],
                     _unroll(rr[::-1], np.full(NS, -PI), mu, r + d0)[1:-1]]
            return np.vstack(parts)

        def scr(pts):
            return [O + np.array([p[0], p[1], 0.0]) for p in pts]

        # ---- the claims
        ann = PI * ((r + d0) ** 2 - r * r)
        for mu in (1.0, 0.7, 0.35, 0.1, 0.0):
            check(abs(abs(area(ring_xy(r, r + d0, mu))) - ann) < 3e-3 * ann,
                  f"the unrolling ring keeps its area (mu={mu})")
        for d in (d0, 0.1, d1):
            trap = [(-PI * (r + d), -(r + d)), (PI * (r + d), -(r + d)),
                    (PI * r, -r), (-PI * r, -r)]
            check(abs(abs(area(trap)) - PI * ((r + d) ** 2 - r * r)) < 1e-12,
                  "the strip is the ring's area")
            check(abs(abs(area(trap)) - (2 * PI * r * d + PI * d * d)) < 1e-12,
                  "strip = 2πr·dr + π·dr²")
        flat = ring_xy(r, r + d0, 0.0)
        check(abs(flat[:, 1].min() + r + d0) < 1e-9 and
              abs(flat[:, 1].max() + r) < 1e-9 and
              abs(np.abs(flat[:, 0]).max() - PI * (r + d0)) < 1e-9,
              "unrolled: between depths r and r + dr, as long as 2π(r + dr)")
        check(PI * d1 < 0.04, "the corner triangles have shrunk to nothing")
        for mu in np.linspace(0, 1, 21):
            pts = scr(ring_xy(r, r + d0, mu))
            check(max(abs(p[0]) for p in pts) < SAFE_X and
                  max(p[1] for p in pts) < SAFE_TOP, "the unroll stays on screen")

        def strip_parts(d, y_top, h):
            """Rectangle 2πr long and the two corner triangles (base π·d),
            top edge at y_top, height h (h = d: true size; h = W: ÷ dr)."""
            a, b = PI * r, PI * (r + d)
            yb = y_top - h
            rect = mk([(-a, yb), (a, yb), (a, y_top), (-a, y_top)], TEAL_D,
                      stroke_width=1.5)
            tl = mk([(-b, yb), (-a, yb), (-a, y_top)], ORANGE, 0.9, stroke_width=1.5)
            tr = mk([(a, yb), (b, yb), (a, y_top)], ORANGE, 0.9, stroke_width=1.5)
            return VGroup(rect, tl, tr)

        y_strip = O[1] - r - d0 - g         # the strip's top edge

        def ring(d):
            return AnnularSector(inner_radius=r, outer_radius=r + d, angle=TAU,
                                 start_angle=PI / 2, fill_color=TEAL_D,
                                 fill_opacity=FILL, stroke_color=WHITE,
                                 stroke_width=1.5).move_arc_center_to(O)

        disc = Circle(radius=r, fill_color=BLUE_D, fill_opacity=FILL,
                      stroke_color=WHITE, stroke_width=2).move_to(O)
        rad = Line(O, O + RIGHT * r, color=WHITE, stroke_width=3)
        l_r = tag("r", 28).next_to(O + RIGHT * r / 2, UP, buff=0.1)
        l_A = tag("πr²", 34).move_to(O + np.array([-0.35, -0.5, 0.0]))
        self.play(FadeIn(disc), Create(rad), FadeIn(Dot(O, radius=0.05)),
                  FadeIn(l_r), FadeIn(l_A), run_time=1.2)

        # ---- grow by dr: a ring
        rg = ring(d0)
        l_dr = tag("dr", 26).next_to(O + RIGHT * (r + d0), RIGHT, buff=0.14)
        self.play(GrowFromCenter(rg), run_time=0.9)
        self.bring_to_front(rad, l_r)
        self.play(FadeIn(l_dr), run_time=0.4)
        self.hold(0.5)

        # ---- cut a copy along the top radius and unroll it
        cut = Line(O + UP * r, O + UP * (r + d0), color=RED_B, stroke_width=6)
        self.play(Create(cut), run_time=0.5)
        m = ValueTracker(1.0)
        roll = always_redraw(lambda: Polygon(
            *scr(ring_xy(r, r + d0, m.get_value())), fill_color=TEAL_D,
            fill_opacity=FILL, stroke_color=WHITE, stroke_width=1.5).shift(
                DOWN * (1.0 - m.get_value()) * (d0 + g)))
        self.add(roll)
        self.bring_to_front(rad, l_r, l_dr)
        self.play(FadeOut(cut), m.animate.set_value(0.0), run_time=3.0,
                  rate_func=smooth)
        roll.clear_updaters()
        strip = strip_parts(d0, y_strip, d0)
        check(close(sorted(map(tuple, np.round(flat + O[:2] - [0, d0 + g], 9)))[0],
                    sorted(map(tuple, np.round(
                        [v[:2] for v in strip[1].get_vertices()], 9)))[0], 1e-9),
              "the flat ring is the trapezoid")
        self.remove(roll)
        self.add(strip)
        self.play(strip[1].animate.set_fill(ORANGE, 0.9),
                  strip[2].animate.set_fill(ORANGE, 0.9), run_time=0.01)
        l_in = tag("2πr", 26).next_to(np.array([3.2, y_strip, 0]), UP, buff=0.14)
        l_out = tag("2π(r + dr)", 26).next_to(
            np.array([3.4, y_strip - d0, 0]), DOWN, buff=0.14)
        self.play(FadeIn(l_in), FadeIn(l_out), run_time=0.7)
        self.hold(0.4)

        # ---- magnified across: ÷ dr
        mag = strip.copy()
        self.add(mag)
        guides = VGroup(*[DashedLine([sx * PI * r, y_strip, 0],
                                     [sx * PI * r, YM + W, 0], color=GREY_B,
                                     stroke_width=1.6, dash_length=0.08)
                          for sx in (-1, 1)])
        self.play(Transform(mag, strip_parts(d0, YM + W, W)), Create(guides),
                  run_time=1.3)
        l_div = tag("÷ dr", 28).move_to(np.array([-3.0, (y_strip - d0 + YM + W) / 2,
                                                  0.0]))
        l_2pr = tag("2πr", 30).move_to(np.array([0.0, YM + W / 2, 0.0]))
        self.play(FadeIn(l_div), FadeIn(l_2pr), run_time=0.6)
        self.hold(0.6)

        # ---- dr → 0: the ring and the strip thin, the corners vanish
        lg = ValueTracker(np.log(d0))

        def d_now():
            return float(np.exp(lg.get_value()))

        rg_r = always_redraw(lambda: ring(d_now()))
        st_r = always_redraw(lambda: strip_parts(d_now(), y_strip, d_now()))
        mg_r = always_redraw(lambda: strip_parts(d_now(), YM + W, W))
        self.remove(rg, strip, mag)
        self.add(rg_r, st_r, mg_r)
        self.bring_to_front(rad, l_r, l_2pr)
        l_dr.add_updater(lambda mm: mm.next_to(O + RIGHT * (r + d_now()), RIGHT,
                                               buff=0.14))
        l_out.add_updater(lambda mm: mm.next_to(
            np.array([3.4, y_strip - d_now(), 0]), DOWN, buff=0.14))
        self.play(lg.animate.set_value(np.log(d1)), run_time=4.2, rate_func=smooth)
        for mm in (rg_r, st_r, mg_r, l_dr, l_out):
            mm.clear_updaters()

        content = VGroup(disc, rad, l_r, l_A, rg_r, l_dr, st_r, l_in, l_out,
                         mg_r, guides, l_div, l_2pr)
        _safe(*content)
        _apart(l_r, l_A, l_dr, l_in, l_out, l_div, l_2pr)
        cap = caption("dA/dr  =  2πr        ( A = πr² )", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H14_MeanValue(Board):
    """Slide the chord AB parallel to itself, away from the arc.  The
    contact points close in on each other; the last position that still
    touches the curve meets it at a single point C without crossing: there
    the line is the tangent, so f′(c) equals the chord's slope.  (Drawn with
    the arc above its chord; otherwise slide the other way.)"""

    def construct(self):
        a, b = 0.7, 5.5

        def G(s):
            return s * (1 - s) * (1 + 0.9 * s)

        def f(x):
            return 0.5 + 0.45 * x + 2.3 * G((x - a) / (b - a))

        m = (f(b) - f(a)) / (b - a)

        def chord(x):
            return f(a) + m * (x - a)

        def g(x):                               # height of the arc over AB
            return f(x) - chord(x)

        # ---- c, found numerically: the highest point of the arc over AB
        lo, hi = a, b
        phi = (np.sqrt(5) - 1) / 2
        for _ in range(200):                    # golden-section search
            x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
            if g(x1) < g(x2):
                lo = x1
            else:
                hi = x2
        c = (lo + hi) / 2
        gmax = g(c)
        hstep = 1e-5
        fp_c = (f(c + hstep) - f(c - hstep)) / (2 * hstep)
        check(abs(fp_c - m) < 1e-7, "f′(c) = slope of the chord")
        check(a < c < b and gmax > 0, "c lies strictly between a and b")
        xs = np.linspace(a, b, 2001)
        check(bool(np.all(g(xs) <= gmax + 1e-12)),
              "the slid line at height gmax never crosses the curve")
        s_star = (-0.2 + np.sqrt(0.04 + 10.8)) / 5.4  # G′(s) = 1 − 0.2s − 2.7s²
        check(abs(c - (a + s_star * (b - a))) < 1e-6, "c agrees with G′ = 0")

        def roots(dl):
            """The two points of the arc at height dl above the chord."""
            out = []
            for l0, h0, up in ((a, c, True), (c, b, False)):
                lo_, hi_ = l0, h0
                for _ in range(60):
                    mid = (lo_ + hi_) / 2
                    if (g(mid) < dl) == up:
                        lo_ = mid
                    else:
                        hi_ = mid
                out.append((lo_ + hi_) / 2)
            return out

        kx, ky = 2.0, 1.35
        org = np.array([-6.2, -2.45, 0.0])

        def P(x, y):
            return org + np.array([kx * x, ky * y, 0.0])

        xl, xr = a - 0.45, b + 0.45
        over = 0.25
        check(P(xr, chord(xr) + gmax + over)[1] < SAFE_TOP - 0.05,
              "the overshooting line stays on screen")

        axes = VGroup(Line(P(0, 0), P(6.15, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 4.1), color=GREY_B, stroke_width=2))
        curve = ParametricFunction(lambda t: P(t, f(t)), t_range=[0.25, 5.95],
                                   color=YELLOW_B, stroke_width=5)
        l_f = tag("f", 28, YELLOW_B).next_to(P(5.95, f(5.95)), RIGHT, buff=0.12)
        A, B = P(a, f(a)), P(b, f(b))

        def drop(x, y, col=GREY_B):
            return DashedLine(P(x, 0), P(x, y), color=col, stroke_width=1.8,
                              dash_length=0.08)

        l_a = tag("a", 28).next_to(P(a, 0), DOWN, buff=0.16)
        l_b = tag("b", 28).next_to(P(b, 0), DOWN, buff=0.16)
        l_b.align_to(l_a, DOWN)
        ch = Line(A, B, color=WHITE, stroke_width=4)

        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_f), run_time=1.2)
        self.play(Create(drop(a, f(a))), Create(drop(b, f(b))), FadeIn(l_a),
                  FadeIn(l_b), FadeIn(Dot(A, radius=0.07)),
                  FadeIn(Dot(B, radius=0.07)), run_time=0.8)
        self.play(Create(ch), run_time=0.8)
        self.hold(0.4)

        # ---- slide a copy of the chord's line parallel to itself
        dl = ValueTracker(0.0)

        def sliding():
            d = dl.get_value()
            ln = Line(P(xl, chord(xl) + d), P(xr, chord(xr) + d),
                      color=TEAL_B, stroke_width=4)
            grp = VGroup(ln)
            if d < gmax - 1e-6:
                for x in roots(d):
                    grp.add(Dot(P(x, f(x)), radius=0.085, color=WHITE))
            elif d < gmax + 1e-6:
                grp.add(Dot(P(c, f(c)), radius=0.08, color=ORANGE))
            return grp

        sl = always_redraw(sliding)
        self.add(sl)
        self.bring_to_front(curve)
        self.play(dl.animate.set_value(gmax + over), run_time=3.2,
                  rate_func=linear)
        self.hold(0.3)
        self.play(dl.animate.set_value(gmax), run_time=1.0, rate_func=smooth)
        sl.clear_updaters()
        check(abs(dl.get_value() - gmax) < 1e-9, "stopped at the last contact")

        # ---- the last contact: a tangent at c, parallel to AB
        C = P(c, f(c))
        dC = drop(c, f(c), ORANGE)
        l_c = tag("c", 28, ORANGE).next_to(P(c, 0), DOWN, buff=0.16)
        l_c.align_to(l_a, DOWN)
        dotC = Dot(C, radius=0.09, color=ORANGE)
        self.play(Create(dC), FadeIn(l_c), FadeIn(dotC), run_time=0.8)
        self.play(Flash(C, color=ORANGE, line_length=0.25, flash_radius=0.25),
                  run_time=0.7)

        content = VGroup(axes, curve, l_f, l_a, l_b, l_c, ch, sl, dC, dotC)
        _safe(*content)
        _apart(l_f, l_a, l_b, l_c)
        cap = caption("f′(c)  =  (f(b) − f(a)) / (b − a)", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ========================================================== G. TRIGONOMETRY

def _unit_pieces(O, R, x):
    """On the circle of radius R about O, angle x at O from the radius OA
    along the x-axis: the inner triangle OAP, the sector OAP and the tangent
    triangle OAT (T on the tangent at A, on the line OP)."""
    O = to3(O)
    A = O + R * RIGHT
    P = O + R * np.array([np.cos(x), np.sin(x), 0.0])
    T = O + R * np.array([1.0, np.tan(x), 0.0])
    tri = mk([O, A, P], BLUE_D, 0.9, stroke_width=2)
    sec = Sector(radius=R, angle=x, start_angle=0.0, arc_center=O,
                 fill_color=TEAL_D, fill_opacity=0.9, stroke_color=WHITE,
                 stroke_width=2)
    tan_t = mk([O, A, T], ORANGE, 0.9, stroke_width=2)
    return tri, sec, tan_t, A, P, T


def _nesting_checks(xs):
    for x in xs:
        check(np.sin(x) < x < np.tan(x), f"sin x < x < tan x at x = {x:.3f}")
        t = np.linspace(0, x, 200)
        arc = np.stack([np.cos(t), np.sin(t)], 1)
        # the arc lies in the tangent triangle: 0 ≤ y, X ≤ 1, below line OT
        check(bool(np.all(arc[:, 0] <= 1 + 1e-12) and np.all(arc[:, 1] >= -1e-12)
                   and np.all(arc[:, 1] * np.cos(x) <= arc[:, 0] * np.sin(x)
                              + 1e-12)), "sector inside the tangent triangle")
        # the chord AP lies inside the disc (convexity): the triangle in the sector
        u = np.linspace(0, 1, 50)
        ch = np.stack([1 - u + u * np.cos(x), u * np.sin(x)], 1)
        check(bool(np.all(np.hypot(ch[:, 0], ch[:, 1]) <= 1 + 1e-12)),
              "inner triangle inside the sector")
        check(abs(abs(area([(0, 0), (1, 0), (np.cos(x), np.sin(x))]))
                  - np.sin(x) / 2) < 1e-12, "inner triangle = ½ sin x")
        check(abs(abs(area([(0, 0), (1, 0), (1, np.tan(x))])) - np.tan(x) / 2)
              < 1e-12, "tangent triangle = ½ tan x")
        poly = [(0.0, 0.0)] + [(np.cos(s), np.sin(s)) for s in np.linspace(0, x, 4001)]
        check(abs(abs(area(poly)) - x / 2) < 1e-6, "sector = ½ x")


def _ra_mark(V, d1, d2, s=0.18, color=GREY_A):
    """Right-angle mark at V between unit directions d1 and d2."""
    V, d1, d2 = to3(V), to3(d1), to3(d2)
    return VMobject(stroke_color=color, stroke_width=2).set_points_as_corners(
        [V + s * d1, V + s * (d1 + d2), V + s * d2])


class G19_SineXTangent(Board):
    """Unit circle, angle x at O, A = (1, 0), P on the circle, T on the
    tangent at A.  The triangle OAP lies inside the sector OAP (the chord is
    inside the disc), which lies inside the triangle OAT (the arc stays
    within the tangent line and the ray OT).  Their areas are ½·1·sin x,
    ½·1²·x and ½·1·tan x, so sin x < x < tan x for every 0 < x < π/2."""

    def construct(self):
        R = 4.2
        O = np.array([-5.9, -2.45, 0.0])
        x0 = 50 * DEGREES
        sweep = (24 * DEGREES, 55 * DEGREES)
        _nesting_checks([x0, *sweep, 10 * DEGREES, 80 * DEGREES])
        check(O[1] + R * np.tan(sweep[1]) < SAFE_TOP - 0.1, "T stays on screen")
        xv = ValueTracker(x0)

        def X():
            return xv.get_value()

        def dirn(t):
            return np.array([np.cos(t), np.sin(t), 0.0])

        arc = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O,
                  color=GREY_B, stroke_width=3)
        OA = Line(O, O + R * RIGHT, color=WHITE, stroke_width=3)
        l_1 = tag("1", 28).next_to((O + O + R * RIGHT) / 2, DOWN, buff=0.14)
        self.play(Create(arc), Create(OA), FadeIn(Dot(O, radius=0.06)),
                  FadeIn(l_1), run_time=1.0)

        ray = always_redraw(lambda: Line(O, O + R * dirn(X()), color=WHITE,
                                         stroke_width=3))
        ang = always_redraw(lambda: angle_arc(O, O + RIGHT, O + dirn(X()),
                                              radius=0.75))
        l_x = tag("x", 30, YELLOW_B)
        l_x.add_updater(lambda m: m.move_to(O + 1.12 * dirn(X() / 2)))
        self.play(Create(ray), Create(ang), FadeIn(l_x), run_time=0.8)

        def piece(i):
            return always_redraw(lambda: _unit_pieces(O, R, X())[i])

        tri, sec, tnt = piece(0), piece(1), piece(2)
        self.add(sec)
        self.bring_to_front(ray, ang, l_x)
        self.play(FadeIn(sec), run_time=0.6)
        self.add(tri)
        PF = always_redraw(lambda: DashedLine(
            O + R * np.array([np.cos(X()), np.sin(X()), 0]),
            O + R * np.array([np.cos(X()), 0, 0]), color=WHITE, stroke_width=2,
            dash_length=0.08))
        raF = always_redraw(lambda: _ra_mark(O + R * np.cos(X()) * RIGHT,
                                             LEFT, UP, 0.16))
        l_sin = tag("sin x", 26)
        l_sin.add_updater(lambda m: m.next_to(
            O + R * np.array([np.cos(X()), np.sin(X()) / 2, 0]), LEFT, buff=0.12))
        self.bring_to_front(ray, ang, l_x)
        self.play(FadeIn(tri), Create(PF), Create(raF), FadeIn(l_sin), run_time=0.9)

        AT = always_redraw(lambda: Line(O + R * RIGHT,
                                        O + R * np.array([1, np.tan(X()), 0]),
                                        color=WHITE, stroke_width=3))
        PT = always_redraw(lambda: Line(O + R * dirn(X()),
                                        O + R * np.array([1, np.tan(X()), 0]),
                                        color=WHITE, stroke_width=3))
        raA = _ra_mark(O + R * RIGHT, LEFT, UP, 0.2)
        l_tan = tag("tan x", 26)
        l_tan.add_updater(lambda m: m.next_to(
            O + R * np.array([1, np.tan(X()) / 2, 0]), RIGHT, buff=0.14))
        self.add(tnt)
        self.bring_to_back(tnt)
        self.play(FadeIn(tnt), Create(AT), Create(PT), Create(raA), FadeIn(l_tan),
                  run_time=1.0)
        self.hold(0.5)

        # ---- the three regions, whole, side by side, with their areas
        sg = 0.36
        Ox = [0.15, 2.55, 4.95]
        Oy = -0.35

        def copy_piece(i):
            return always_redraw(lambda: _unit_pieces(
                np.array([Ox[i], Oy, 0.0]), sg * R, X())[i])

        cps = [copy_piece(i) for i in range(3)]
        labs = [tag("½ sin x", 28, BLUE_B), tag("½ x", 28, TEAL_B),
                tag("½ tan x", 28, ORANGE)]
        for i, lb in enumerate(labs):
            lb.move_to([Ox[i] + sg * R / 2, Oy - 0.42, 0.0])
        lts = [tag("<", 30).move_to([(labs[i].get_right()[0] +
                                      labs[i + 1].get_left()[0]) / 2,
                                     labs[0].get_center()[1], 0]) for i in range(2)]
        srcs = [_unit_pieces(O, R, x0)[i] for i in range(3)]
        for i in range(3):
            mv = srcs[i].copy()
            self.play(Transform(mv, cps[i].copy().clear_updaters()), run_time=0.7)
            self.remove(mv)                  # the live copy takes its place
            self.add(cps[i])
            self.play(FadeIn(labs[i]), run_time=0.35)
        self.play(FadeIn(lts[0]), FadeIn(lts[1]), run_time=0.5)
        self.hold(0.6)

        # ---- any x in (0, π/2): the nesting, hence the order, persists
        self.play(xv.animate.set_value(sweep[0]), run_time=1.8, rate_func=smooth)
        self.play(xv.animate.set_value(sweep[1]), run_time=2.0, rate_func=smooth)
        self.play(xv.animate.set_value(x0), run_time=1.0, rate_func=smooth)
        for mob in self.mobjects:
            mob.clear_updaters()

        content = VGroup(arc, OA, l_1, ray, ang, l_x, sec, tri, PF, l_sin, tnt, AT,
                         PT, l_tan, *cps, *labs, *lts)
        _safe(*content)
        _apart(l_1, l_x, l_sin, l_tan, *labs, *lts)
        cap = caption("sin x  <  x  <  tan x        (0 < x < π/2)", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ========================================================== H. (limits)

def _lim_row(var="x → 0", size=34, color=YELLOW_B):
    """'lim' with a small 'x → 0' under it."""
    l = Text("lim", font_size=size, color=color)
    n = Text(var, font_size=size * 0.55, color=color).next_to(l, DOWN, buff=0.05)
    return VGroup(l, n)


class H15_SinOverX(Board):
    """On the unit circle the inner triangle (½ sin x) lies in the sector
    (½ x), which lies in the tangent triangle (½ tan x).  Stretch the figure
    vertically by 1/x — an affine map, so it keeps the nesting and the
    ratios of areas: the sector becomes a region of area ½ for every x, the
    triangles get heights sin x / x and tan x / x.  The inner apex sits on
    the outer triangle's side above the point cos x of the base, so
    sin x / x = cos x · tan x / x > cos x.  As x → 0, cos x → 1: both
    triangles close onto the triangle of height 1, and sin x / x → 1."""

    def construct(self):
        R = 4.4
        O = np.array([-6.4, -2.45, 0.0])
        O2, L = np.array([1.6, -2.45, 0.0]), 3.8      # the stretched view
        x0, x1 = 40 * DEGREES, 3 * DEGREES
        NS = 60
        _nesting_checks([x0, x1, 15 * DEGREES])

        def unit_shapes(x):
            ts = np.linspace(0.0, x, NS)
            return ([(0.0, 0.0), (1.0, 0.0), (np.cos(x), np.sin(x))],
                    [(0.0, 0.0)] + [(np.cos(t), np.sin(t)) for t in ts],
                    [(0.0, 0.0), (1.0, 0.0), (1.0, np.tan(x))])

        def M(p):
            return O + R * np.array([p[0], p[1], 0.0])

        def S(p, x):
            return O2 + L * np.array([p[0], p[1] / x, 0.0])

        cols = [BLUE_D, TEAL_D, ORANGE]

        def shapes(x, mp):
            u = unit_shapes(x)
            return [mk([mp(p) for p in u[i]], cols[i], 0.9, stroke_width=2)
                    for i in (2, 1, 0)]         # back to front: tan, sector, tri

        # ---- the stretch keeps area ratios; the stretched sector is always ½
        for x in (x0, 0.3, x1):
            u = unit_shapes(x)
            a_main = [abs(area(s_)) for s_ in u]
            a_str = [abs(area([(p[0], p[1] / x) for p in s_])) for s_ in u]
            check(all(abs(a_str[i] - a_main[i] / x) < 1e-9 for i in range(3)),
                  "the vertical stretch by 1/x divides every area by x")
            check(abs(a_str[1] - 0.5) < 1e-3, "stretched sector has area ½")
            check(abs(a_str[0] - np.sin(x) / x / 2) < 1e-12 and
                  abs(a_str[2] - np.tan(x) / x / 2) < 1e-12,
                  "stretched triangles: ½·sin x / x and ½·tan x / x")
            # the inner apex lies on the outer side, above cos x of the base
            check(abs(np.sin(x) / x - np.cos(x) * np.tan(x) / x) < 1e-12,
                  "sin x / x = cos x · tan x / x")
        check(abs(np.tan(x1) / x1 - 1) * L < 0.01 and
              abs(np.sin(x1) / x1 - 1) * L < 0.01,
              "at the last x both apexes sit on the level-1 corner")
        check(O2[1] + L * np.tan(x0) / x0 < 2.3, "room for the ledger")

        xv = ValueTracker(x0)

        def X():
            return xv.get_value()

        # ---- the nested figure
        arc = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O,
                  color=GREY_B, stroke_width=3)
        l_1 = tag("1", 28).next_to(M((0.5, 0)), DOWN, buff=0.14)
        main = always_redraw(lambda: VGroup(*shapes(X(), M)))
        rays = always_redraw(lambda: VGroup(
            Line(M((1, 0)), M((1, np.tan(X())))).set_stroke(WHITE, 3),
            Line(O, M((1, np.tan(X())))).set_stroke(WHITE, 3)))
        ang = angle_arc(O, O + RIGHT, O + np.array([np.cos(x0), np.sin(x0), 0]),
                        radius=0.75)
        l_x = tag("x", 28, YELLOW_B).move_to(
            O + 1.08 * np.array([np.cos(x0 / 2), np.sin(x0 / 2), 0]))
        self.play(Create(arc), FadeIn(Dot(O, radius=0.06)), FadeIn(l_1),
                  run_time=0.8)
        self.play(FadeIn(main), Create(rays), Create(ang), FadeIn(l_x),
                  run_time=1.2)
        self.bring_to_front(ang, l_x)

        def ledger(parts, y, cx, size=27):
            row = VGroup(*[tag(s_, size, c) for s_, c in parts]).arrange(
                RIGHT, buff=0.28)
            return row.move_to([cx, y, 0.0])

        lineA = ledger([("½ sin x", BLUE_B), ("<", WHITE), ("½ x", TEAL_B),
                        ("<", WHITE), ("½ tan x", ORANGE)], 3.3, -4.0)
        self.play(FadeIn(lineA), run_time=0.9)
        self.hold(0.6)

        # ---- stretched vertically by 1/x
        tgt = shapes(x0, lambda p: S(p, x0))
        mv = VGroup(*shapes(x0, M))
        self.add(mv)
        l_div = tag("÷ x", 30).move_to([2.2, 1.3, 0.0])
        self.play(Transform(mv, VGroup(*tgt)), FadeIn(l_div), run_time=1.8)
        self.remove(mv)
        strd = always_redraw(lambda: VGroup(*shapes(X(), lambda p: S(p, X()))))
        self.add(strd)
        lim_tri = DashedVMobject(Polygon(O2, O2 + L * RIGHT, O2 + L * (RIGHT + UP),
                                         stroke_color=WHITE, stroke_width=2.5),
                                 num_dashes=45)
        lvl = DashedLine(O2 + L * UP + RIGHT * L * 0.55, O2 + L * (UP + RIGHT)
                         + RIGHT * 0.45, color=GREY_B, stroke_width=2,
                         dash_length=0.08)
        l_lvl = tag("1", 28).next_to(lvl, RIGHT, buff=0.12)
        l_b = tag("1", 28).next_to(O2 + L * RIGHT / 2, DOWN, buff=0.14)
        dots = always_redraw(lambda: VGroup(
            Dot(S((np.cos(X()), np.sin(X())), X()), radius=0.08, color=BLUE_B),
            Dot(S((1.0, np.tan(X())), X()), radius=0.08, color=ORANGE)))
        lineB = ledger([("sin x / x", BLUE_B), ("<", WHITE), ("1", WHITE),
                        ("<", WHITE), ("tan x / x", ORANGE)], 3.3, 3.55)
        self.play(Create(lim_tri), Create(lvl), FadeIn(l_lvl), FadeIn(l_b),
                  FadeIn(dots), run_time=1.0)
        self.play(FadeIn(lineB), run_time=0.9)
        self.hold(0.5)

        # ---- the inner apex stands above cos x of the base
        drop = always_redraw(lambda: DashedLine(
            S((np.cos(X()), np.sin(X())), X()), S((np.cos(X()), 0.0), X()),
            color=BLUE_A, stroke_width=2.5, dash_length=0.08))
        cosbar = always_redraw(lambda: Line(
            O2 + 0.06 * UP, S((np.cos(X()), 0.0), X()) + 0.06 * UP,
            color=YELLOW_B, stroke_width=6))
        l_cos = tag("cos x", 26, YELLOW_B)
        l_cos.add_updater(lambda m: m.next_to(
            O2 + 0.5 * L * np.cos(X()) * RIGHT, UP, buff=0.16))
        lineC = ledger([("cos x", YELLOW_B), ("<", WHITE), ("sin x / x", BLUE_B),
                        ("<", WHITE), ("1", WHITE)], 2.72, 3.55)
        self.play(Create(drop), Create(cosbar), FadeIn(l_cos), run_time=0.9)
        self.play(FadeIn(lineC), run_time=0.9)
        self.hold(0.8)

        # ---- x → 0: cos x → 1, both triangles close onto the height-1 one
        self.play(FadeOut(ang), FadeOut(l_x), run_time=0.4)
        self.play(xv.animate.set_value(x1), run_time=4.6, rate_func=smooth)
        for mob in self.mobjects:
            mob.clear_updaters()

        content = VGroup(arc, l_1, main, rays, lineA, l_div, strd, lim_tri, lvl,
                         l_lvl, l_b, dots, lineB, drop, cosbar, l_cos, lineC)
        _safe(*content)
        _apart(l_1, lineA, l_div, l_lvl, l_b, lineB, l_cos, lineC)
        cap = _cap(_row(_lim_row(), tag("sin x / x   =   1", 34, YELLOW_B),
                        buff=0.2))
        cap[0].shift(UP * (cap[1].get_center()[1] - cap[0][0].get_center()[1]))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ========================================================== E. CIRCLE

class E11_SectorArea(Board):
    """Cut the sector (radius r, angle θ) into n equal wedges and turn every
    other one over.  Interleaved exactly — neighbours share a whole straight
    edge — they fill a parallelogram whose top and bottom are each made of
    n/2 arcs, half the arc in all, and whose height is r·cos(θ/2n).  As n
    grows the arcs flatten and the slant vanishes: a rectangle r by ½rθ.
    (Drawn with n = 6, then n = 36.)"""

    def construct(self):
        r = 3.2
        th = 150 * DEGREES
        a_lo = PI / 2 - th / 2                     # sector from 15° to 165°
        apex = np.array([-3.3, -1.25, 0.0])
        Y_top, x_c = 1.95, 3.4                     # the strip's top line, centre

        def dirn(t):
            return np.array([np.cos(t), np.sin(t), 0.0])

        def wedge(ap, a0, d, color, sw):
            N = max(4, int(np.ceil(d / (2 * DEGREES))) + 1)
            ts = np.linspace(a0, a0 + d, N)
            return mk([ap] + [ap + r * dirn(t) for t in ts], color, FILL,
                      stroke_width=sw)

        def layout(n):
            """Sector wedges (left to right) and their interleaved targets."""
            d = th / n
            c, h = 2 * r * np.sin(d / 2), r * np.cos(d / 2)
            X0 = x_c - (n / 2) * c / 2 + c / 4
            moves = []
            for i in range(n):
                a0 = a_lo + th - (i + 1) * d
                k = i // 2
                if i % 2 == 0:                     # arc down: apex on the top line
                    ap1, b0 = np.array([X0 + k * c, Y_top, 0.0]), -PI / 2 - d / 2
                else:                              # arc up: apex on the lower line
                    ap1 = np.array([X0 + k * c + c / 2, Y_top - h, 0.0])
                    b0 = PI / 2 - d / 2
                turn = (b0 - a0 + PI) % TAU - PI
                moves.append((a0, ap1, b0, turn))
            return d, c, h, X0, moves

        cols = [BLUE_D, TEAL_D]

        # ---- the claims: exact interleaving, area, and the limit shape
        for n in (6, 36):
            d, c, h, X0, moves = layout(n)
            ends = []
            for i, (a0, ap1, b0, turn) in enumerate(moves):
                e1, e2 = ap1 + r * dirn(b0), ap1 + r * dirn(b0 + d)
                ends.append((ap1, e1, e2))
                check(abs((a0 + turn - b0 + PI) % TAU - PI) < 1e-12,
                      "the turn takes the wedge's arc where it belongs")
            for i in range(n - 1):
                A_, B_ = ends[i], ends[i + 1]
                # wedge i's right edge = wedge i+1's left edge (as segments).
                # Arc down (even): left edge ends at e1, right at e2;
                # arc up (odd): left edge ends at e2, right at e1.
                right_i = {tuple(np.round(A_[0][:2], 9)), tuple(np.round(
                    (A_[1] if i % 2 else A_[2])[:2], 9))}
                left_j = {tuple(np.round(B_[0][:2], 9)), tuple(np.round(
                    (B_[2] if (i + 1) % 2 else B_[1])[:2], 9))}
                check(right_i == left_j, f"wedges {i}, {i + 1} share an edge")
            check(abs(n * 0.5 * r * r * d - 0.5 * r * r * th) < 1e-9,
                  "the wedges are the sector")
            check(abs((n / 2) * c * h - n * 0.5 * r * r * np.sin(d)) < 1e-9,
                  "parallelogram = the n triangles")
        _, c36, h36, X036, _ = layout(36)
        check(abs((18 * c36) - r * th / 2) < 0.002 and abs(h36 - r) < 0.003,
              "n = 36: top ≈ ½rθ, height ≈ r")
        check(X036 + 18 * c36 + 1.0 < SAFE_X, "room for the r label")

        # ---- the sector
        whole = Sector(radius=r, angle=th, start_angle=a_lo, arc_center=apex,
                       fill_color=BLUE_D, fill_opacity=FILL, stroke_color=WHITE,
                       stroke_width=2)
        ang = angle_arc(apex, apex + dirn(a_lo), apex + dirn(a_lo + th),
                        radius=0.6)
        l_th = tag("θ", 30, YELLOW_B).move_to(apex + 0.95 * UP)
        mid_r = apex + 0.5 * r * dirn(a_lo)
        l_r = tag("r", 28).move_to(mid_r + 0.32 * dirn(a_lo - PI / 2))
        l_arc = tag("rθ", 30).next_to(apex + r * UP, UP, buff=0.14)
        self.play(FadeIn(whole), run_time=1.0)
        self.play(Create(ang), FadeIn(l_th), FadeIn(l_r), FadeIn(l_arc),
                  run_time=0.8)
        self.hold(0.6)

        def cut(n):
            d = th / n
            return [wedge(apex, a_lo + th - (i + 1) * d, d, cols[i % 2],
                          1.5 if n < 12 else 0.8) for i in range(n)]

        def fly(n, wedges, run):
            d, _, _, _, moves = layout(n)
            anims = []
            for w, (_, ap1, b0, turn) in zip(wedges, moves):
                # spin about the wedge's own centre Q while Q travels to its
                # place: the same rigid end pose as turning about the apex
                Q = np.mean(w.get_vertices(), axis=0)
                cs, sn = np.cos(turn), np.sin(turn)
                v = Q - apex
                Q1 = ap1 + np.array([cs * v[0] - sn * v[1],
                                     sn * v[0] + cs * v[1], 0.0])
                anims.append(_rigid(w, angle=turn, about=Q, shift=Q1 - Q))
            self.play(LaggedStart(*anims, lag_ratio=0.12 if n < 12 else 0.05),
                      run_time=run)
            for w, (_, ap1, b0, turn) in zip(wedges, moves):
                want = wedge(ap1, b0, d, BLUE_D, 1)
                check(close(w.get_vertices(), want.get_vertices(), 1e-6),
                      "wedge landed exactly")

        ghost = DashedVMobject(Sector(radius=r, angle=th, start_angle=a_lo,
                                      arc_center=apex, stroke_color=GREY_B,
                                      stroke_width=2, fill_opacity=0),
                               num_dashes=60)

        # ---- n = 6
        w6 = cut(6)
        self.play(FadeOut(ang), FadeOut(l_th), run_time=0.4)
        self.remove(whole)
        self.add(ghost, *w6)
        self.play(*[w.animate.set_stroke(width=2.5) for w in w6], run_time=0.5)
        self.play(*[w.animate.set_stroke(width=1.5) for w in w6], run_time=0.3)
        fly(6, w6, 2.6)
        self.hold(0.8)

        # ---- n = 36
        w36 = cut(36)
        self.play(FadeOut(VGroup(*w6)), FadeIn(VGroup(*w36)), run_time=0.8)
        fly(36, w36, 3.4)
        self.hold(0.4)

        # ---- the limit: a rectangle r by ½rθ
        W, Hh = r * th / 2, r
        cx = X036 + 9 * c36 - c36 / 4
        cy = Y_top - h36 / 2
        box = Rectangle(width=W, height=Hh, stroke_color=YELLOW_B,
                        stroke_width=4).move_to([cx, cy, 0.0])
        b_w = _dim([cx - W / 2, cy - Hh / 2 - 0.28, 0],
                   [cx + W / 2, cy - Hh / 2 - 0.28, 0], YELLOW_B, 0.1, 3)
        l_w = tag("½ rθ", 30, YELLOW_B).next_to(b_w, DOWN, buff=0.12)
        b_h = _dim([cx + W / 2 + 0.3, cy - Hh / 2, 0],
                   [cx + W / 2 + 0.3, cy + Hh / 2, 0], YELLOW_B, 0.1, 3)
        l_h = tag("r", 30, YELLOW_B).next_to(b_h, RIGHT, buff=0.14)
        self.play(Create(box), run_time=0.9)
        self.play(Create(b_w), FadeIn(l_w), Create(b_h), FadeIn(l_h), run_time=0.8)

        content = VGroup(ghost, l_r, l_arc, *w36, box, b_w, l_w, b_h, l_h)
        _safe(*content)
        _apart(l_r, l_arc, l_w, l_h)
        cap = caption("A  =  ½ rθ · r  =  ½ r²θ", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ================================================== B. SUMS & FIGURATE NUMBERS

class B42_HalfSquareHalfDiagonal(Board):
    """The staircase 1 + 2 + … + n inside the n-square.  The diagonal cuts
    each of the n cells on it in half: the staircase is the triangle below
    the diagonal plus the n upper half-cells (teeth).  The triangle is half
    the square (a half-turn about the centre swaps the two halves), n²/2;
    the teeth pair up into n/2 cells.  (Drawn with n = 6.)"""

    def construct(self):
        n, u = 6, 0.78
        G = np.array([-5.45, -2.0, 0.0])
        # the half-turn sweeps a disc of radius (n/√2)·u about the centre
        ctr, reach = G + u * np.array([n / 2, n / 2, 0.0]), u * n / np.sqrt(2)
        check(ctr[0] - reach > -SAFE_X and ctr[1] + reach < SAFE_TOP and
              ctr[1] - reach > SAFE_BOTTOM, "the half-turn stays on screen")

        def P(a, b):
            return G + u * np.array([a, b, 0.0])

        # ---- the claims
        T = n * (n + 1) // 2
        check(abs(T - (n * n / 2 + n / 2)) < 1e-12, "Tₙ = n²/2 + n/2")
        low = [(0, 0), (n, 0), (n, n)]
        check(abs(abs(area(low)) - n * n / 2) < 1e-12, "the lower triangle is n²/2")
        check(abs(n * 0.5 - n / 2) < 1e-12 and n % 2 == 0, "n teeth make n/2 cells")
        stair_cells = [(i, j) for i in range(n) for j in range(i + 1)]
        below = [c for c in stair_cells if c[1] < c[0]]
        check(abs(len(below) + n * 0.5 - n * n / 2) < 1e-12,
              "cells under the diagonal + n lower halves = the triangle")

        cols = []
        for i in range(n):
            col = VGroup(*[cell(i, j, u, G, BLUE_D) for j in range(i)])
            cols.append(col)
        diag_cells = [cell(i, i, u, G, BLUE_D) for i in range(n)]
        for i in range(n):
            cols[i].add(diag_cells[i])
        l_cols = VGroup(*[tag(s, 26).next_to(P(i + 0.5, 0), DOWN, buff=0.16)
                          for i, s in ((0, "1"), (1, "2"), (2, "3"), (5, "n"))])
        l_dots = tag("…", 26).next_to(P(4.0, 0), DOWN, buff=0.16).align_to(
            l_cols[0], DOWN)
        for lb in l_cols:
            lb.align_to(l_cols[0], DOWN)

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cols],
                              lag_ratio=0.25), run_time=2.0)
        self.play(FadeIn(l_cols), FadeIn(l_dots), run_time=0.6)
        self.hold(0.3)

        # ---- the n-square and its diagonal
        sq = DashedVMobject(Polygon(P(0, 0), P(n, 0), P(n, n), P(0, n),
                                    stroke_color=GREY_B, stroke_width=2.5),
                            num_dashes=60)
        diag = Line(P(0, 0), P(n, n), color=YELLOW_B, stroke_width=4)
        b_n = _dim(P(n, 0) + RIGHT * 0.3, P(n, n) + RIGHT * 0.3, GREY_A, 0.1, 3)
        l_n = tag("n", 28).next_to(b_n, RIGHT, buff=0.14)
        self.play(Create(sq), Create(b_n), FadeIn(l_n), run_time=0.9)
        self.play(Create(diag), run_time=0.8)

        # ---- each diagonal cell is cut in two; the upper halves are teeth
        lows = [mk([P(i, i), P(i + 1, i), P(i + 1, i + 1)], BLUE_D,
                   stroke_width=1.5) for i in range(n)]
        teeth = [mk([P(i, i), P(i, i + 1), P(i + 1, i + 1)], BLUE_D,
                    stroke_width=1.5) for i in range(n)]
        for i in range(n):
            cols[i].remove(diag_cells[i])
        self.add(*lows, *teeth)
        self.remove(*diag_cells)
        self.bring_to_front(diag)
        self.play(*[t.animate.set_fill(ORANGE, 0.9) for t in teeth], run_time=0.8)
        self.hold(0.4)

        # ---- the teeth pair up into n/2 whole cells; dashed outlines keep
        # the places they came from
        holes = [DashedVMobject(Polygon(P(i, i), P(i, i + 1), P(i + 1, i + 1),
                                        stroke_color=ORANGE, stroke_width=2.5),
                                num_dashes=9) for i in range(n)]
        self.add(*holes)
        self.bring_to_front(*teeth)
        TX, TY = 1.4, -0.1
        anims = []
        for j in range(n // 2):
            S = np.array([TX + j * u, TY, 0.0])
            a, b = teeth[2 * j], teeth[2 * j + 1]
            anims.append(_rigid(a, shift=S - P(2 * j, 2 * j)))
            M = P(2 * j + 1.5, 2 * j + 1.5)
            anims.append(_rigid(b, angle=PI, about=M,
                                shift=S - P(2 * j + 1, 2 * j + 1)))
        self.play(LaggedStart(*anims, lag_ratio=0.12), run_time=2.4)
        for j in range(n // 2):
            S = np.array([TX + j * u, TY, 0.0])
            sq_pts = {tuple(np.round((S + u * np.array(v))[:2], 7))
                      for v in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))}
            got = {tuple(np.round(v[:2], 7)) for t in teeth[2 * j:2 * j + 2]
                   for v in t.get_vertices()}
            check(got == sq_pts, "two teeth make one whole cell")
        b_h = _dim([TX, TY - 0.28, 0], [TX + (n // 2) * u, TY - 0.28, 0],
                   ORANGE, 0.1, 3)
        l_h = tag("n/2", 30, ORANGE).next_to(b_h, DOWN, buff=0.14)
        self.play(Create(b_h), FadeIn(l_h), run_time=0.6)
        self.hold(0.4)

        # ---- what is left is the triangle below the diagonal: half the square
        tri = Polygon(P(0, 0), P(n, 0), P(n, n), stroke_color=YELLOW_B,
                      stroke_width=5)
        self.play(Create(tri), run_time=0.8)
        ghost = mk([P(0, 0), P(n, 0), P(n, n)], BLUE_D, 0.4, stroke_color=YELLOW_B,
                   stroke_width=3)
        self.add(ghost)
        self.play(Rotate(ghost, angle=PI, about_point=P(n / 2, n / 2)),
                  run_time=1.8)
        got = {tuple(np.round(v[:2], 7)) for v in ghost.get_vertices()}
        want = {tuple(np.round(P(*v)[:2], 7)) for v in ((0, 0), (0, n), (n, n))}
        check(got == want, "the half-turn lands on the other half")
        self.bring_to_front(*holes, diag, tri)
        lab = tag("n²/2", 34)
        lab_bg = BackgroundRectangle(lab, color=BLACK, fill_opacity=0.65, buff=0.08)
        lab_low = VGroup(lab_bg, lab).move_to(P(2 * n / 3 + 0.35, n / 3 - 0.35))
        lab2 = tag("n²/2", 34)
        lab_up = VGroup(lab2).move_to(P(n / 3 - 0.35, 2 * n / 3 + 0.35))
        self.play(FadeIn(lab_low), FadeIn(lab_up), run_time=0.7)

        content = VGroup(*cols, *lows, *teeth, *holes, l_cols, l_dots, sq, diag,
                         b_n, l_n, b_h, l_h, tri, ghost, lab_low, lab_up)
        _safe(*content)
        _apart(*l_cols, l_dots, l_n, l_h, lab_low, lab_up)
        cap = caption("1 + 2 + … + n  =  n²/2 + n/2", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ========================================================== D. INEQUALITIES

def _mrow(parts, size=30, buff=0.16):
    """A row of math pieces [(spec, colour), ...] built with _mt, all on one
    baseline, spaced by `buff`."""
    row = VGroup()
    x = 0.0
    for spec, col in parts:
        g = _mt(spec, size, col)
        g.shift(RIGHT * (x - g.get_left()[0]))   # baseline stays at y = 0
        x = g.get_right()[0] + buff
        row.add(g)
    return row


class D14_ExpAboveTangent(Board):
    """y = eᵗ and its tangent y = 1 + t at (0, 1).  The area under eᵗ from
    0 to x is eˣ − 1.  For x > 0 it contains the unit-height rectangle on
    [0, x] (eᵗ ≥ 1 there), whose area is x: eˣ − 1 ≥ x.  For x < 0 it lies
    inside the unit-height rectangle on [x, 0] (eᵗ ≤ 1 there), whose area is
    −x: 1 − eˣ ≤ −x.  Either way eˣ ≥ 1 + x: the tangent never rises above
    the curve.  (The orange excess is eˣ − 1 − x in both cases.)"""

    def construct(self):
        kx, ky = 1.8, 1.12
        O = np.array([-2.25, -1.3, 0.0])          # the point t = 0, y = 0
        t_lo, t_hi = -2.2, 1.45
        x1, x2 = 1.2, -1.6

        def P(t, y):
            return O + np.array([kx * t, ky * y, 0.0])

        # ---- the claims
        for x in (x1, x2, 0.3, -0.7):
            ar = _simpson(np.exp, min(0, x), max(0, x))
            check(abs(ar - abs(np.exp(x) - 1)) < 1e-9,
                  "area under eᵗ between 0 and x = |eˣ − 1|")
            check(np.exp(x) >= 1 + x, "eˣ ≥ 1 + x")
        check(bool(np.all(np.exp(np.linspace(0, x1, 50)) >= 1)),
              "eᵗ ≥ 1 on [0, x]: the rectangle lies under the curve")
        check(bool(np.all(np.exp(np.linspace(x2, 0, 50)) <= 1)),
              "eᵗ ≤ 1 on [x, 0]: the curve lies in the rectangle")
        ex = _simpson(lambda t: np.exp(t) - 1, 0, x1)
        check(abs(ex - (np.exp(x1) - 1 - x1)) < 1e-9, "excess = eˣ − 1 − x (x > 0)")
        ex2 = _simpson(lambda t: 1 - np.exp(t), x2, 0)
        check(abs(ex2 - (np.exp(x2) - 1 - x2)) < 1e-9, "excess = eˣ − 1 − x (x < 0)")
        check(P(t_hi, np.exp(t_hi))[1] < SAFE_TOP - 0.1, "the curve fits")

        axes = VGroup(Line(P(t_lo, 0), P(t_hi + 0.15, 0), color=GREY_B,
                           stroke_width=2),
                      Line(P(0, -1.3), P(0, 4.38), color=GREY_B, stroke_width=2))
        curve = ParametricFunction(lambda t: P(t, np.exp(t)),
                                   t_range=[t_lo, t_hi], color=YELLOW_B,
                                   stroke_width=5)
        tang = Line(P(-1.95, -0.95), P(t_hi, 1 + t_hi), color=WHITE,
                    stroke_width=3.5)
        l_curve = _mt("y = e^{t}", 28, YELLOW_B).next_to(
            P(t_hi, np.exp(t_hi)), RIGHT, buff=0.18)
        l_tang = _mt("y = 1 + t", 28).next_to(P(t_hi, 1 + t_hi), RIGHT, buff=0.18)
        dot = Dot(P(0, 1), radius=0.07)
        tick1 = Line(P(0, 1) + LEFT * 0.08, P(0, 1) + RIGHT * 0.08, color=GREY_B)
        l_one = tag("1", 26).next_to(P(0, 1), LEFT, buff=0.16).shift(UP * 0.14)

        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.2)
        self.play(Create(tang), FadeIn(dot), Create(tick1), FadeIn(l_one),
                  FadeIn(l_tang), run_time=1.0)
        self.hold(0.6)
        self.play(tang.animate.set_stroke(opacity=0.25),
                  l_tang.animate.set_opacity(0.35), run_time=0.5)

        def ts(a, b, m=80):
            return np.linspace(a, b, m)

        # ---- x > 0: the rectangle of area x lies under the curve
        rectA = mk([P(0, 0), P(x1, 0), P(x1, 1), P(0, 1)], BLUE_D, stroke_width=1.5)
        excA = mk([P(0, 1)] + [P(t, np.exp(t)) for t in ts(0, x1)] + [P(x1, 1)],
                  ORANGE, 0.85, stroke_width=1.5)
        outA = Polygon(P(0, 0), *[P(t, np.exp(t)) for t in ts(0, x1)], P(x1, 0),
                       stroke_color=YELLOW_B, stroke_width=4)
        tkA = Line(P(x1, 0) + DOWN * 0.08, P(x1, 0) + UP * 0.08, color=GREY_B)
        l_xA = tag("x", 28, BLUE_B).next_to(P(x1, 0), DOWN, buff=0.16)
        self.play(Create(outA), run_time=0.9)
        self.play(FadeIn(rectA), FadeIn(excA), Create(tkA), FadeIn(l_xA),
                  run_time=0.9)
        self.bring_to_front(outA, curve, dot)

        LX = 1.6
        rowA = _mrow([("x > 0 :", GREY_A), ("x", BLUE_B), ("≤", WHITE),
                      ("e^{x} − 1", YELLOW_B)], 30, 0.22).move_to(
                          [LX, 0.25, 0.0], aligned_edge=LEFT)
        self.play(FadeIn(rowA), run_time=0.8)
        self.hold(0.8)

        # ---- x < 0: the curve's region lies in the rectangle of area −x
        inB = mk([P(x2, 0)] + [P(t, np.exp(t)) for t in ts(x2, 0)] + [P(0, 0)],
                 BLUE_D, stroke_width=1.5)
        excB = mk([P(x2, 1), P(0, 1)] + [P(t, np.exp(t)) for t in ts(0, x2)],
                  ORANGE, 0.85, stroke_width=1.5)
        outB = Polygon(P(x2, 0), P(0, 0), P(0, 1), P(x2, 1), stroke_color=YELLOW_B,
                       stroke_width=4)
        tkB = Line(P(x2, 0) + DOWN * 0.08, P(x2, 0) + UP * 0.08, color=GREY_B)
        l_xB = tag("x", 28, YELLOW_B).next_to(P(x2, 0), DOWN, buff=0.16)
        l_xB.align_to(l_xA, DOWN)
        self.play(Create(outB), Create(tkB), FadeIn(l_xB), run_time=0.9)
        self.play(FadeIn(inB), FadeIn(excB), run_time=0.9)
        self.bring_to_front(outB, curve, dot)
        rowB = _mrow([("x < 0 :", GREY_A), ("1 − e^{x}", BLUE_B), ("≤", WHITE),
                      ("−x", YELLOW_B)], 30, 0.22).move_to(
                          [LX, -0.6, 0.0], aligned_edge=LEFT)
        self.play(FadeIn(rowB), run_time=0.8)
        self.hold(0.8)

        # ---- either way: the curve stays above its tangent
        rowC = _mrow([("⟹", WHITE), ("e^{x}", YELLOW_B), ("≥", WHITE),
                      ("1 + x", WHITE)], 30, 0.22).move_to(
                          [LX + 1.15, -1.45, 0.0], aligned_edge=LEFT)
        self.play(tang.animate.set_stroke(opacity=1.0),
                  l_tang.animate.set_opacity(1.0), FadeIn(rowC), run_time=0.9)
        self.bring_to_front(tang, curve, dot)

        content = VGroup(axes, curve, tang, l_curve, l_tang, dot, l_one, rectA,
                         excA, outA, l_xA, inB, excB, outB, l_xB, rowA, rowB, rowC)
        _safe(*content)
        _apart(l_curve, l_tang, l_one, l_xA, l_xB, rowA, rowB, rowC)
        cap = _cap(_mt("e^{x}  ≥  1 + x", 38, YELLOW_B))
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)
