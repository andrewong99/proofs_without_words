# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2d.py — H7 derivative of cosine, H19 Cavalieri in the plane,
# H25 derivative of the inverse, H23 mean value theorem for integrals,
# H22 odd functions over symmetric intervals, L2 dot product as a
# projection, L4 the shoelace formula, L8 distance from a point to a line,
# G13 sum to product.
#
# Every move of a piece is rigid (a shift, a Rotate about a point, or a fold
# about an in-plane axis, which is a half-turn in space), and every landing,
# area and length claim is checked numerically with check(...) before it is
# drawn, so a wrong construction fails the render instead of rendering
# quietly into a wrong picture.  Curves are polylines through finely
# sampled points of the function.


# ------------------------------------------------------------ private helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _dir(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _foot(p, a, b):
    """Foot of the perpendicular from p onto the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = b - a
    return a + np.dot(p - a, d) / np.dot(d, d) * d


def _curve(pts, color=WHITE, width=4):
    """Open polyline through screen points (fine sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on
    the side the vector `side` points to, its box clear of the line by
    `gap` whatever the slope."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


def _safe(*mobs, tol=0.03):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for i, m in enumerate(mobs):
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area (item {i}, {type(m).__name__}): "
              f"[{lo[0]:.2f},{hi[0]:.2f}] x [{lo[1]:.2f},{hi[1]:.2f}]")


def _apart(*mobs, gap=0.04):
    """Fail the render if any two labels' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            a0, a1 = mobs[i].get_critical_point(DL), mobs[i].get_critical_point(UR)
            b0, b1 = mobs[j].get_critical_point(DL), mobs[j].get_critical_point(UR)
            sep = (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
                   or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])
            check(sep, f"labels {i} and {j} do not overlap")


def _clear(m, p, q, gap=0.04, what="a label"):
    """Fail the render if the screen segment pq runs through m's box."""
    lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
    p, q = to3(p), to3(q)
    n = max(2, int(np.linalg.norm(q - p) / 0.01) + 2)
    for f in np.linspace(0.0, 1.0, n):
        x = p + (q - p) * f
        if lo[0] - gap < x[0] < hi[0] + gap and lo[1] - gap < x[1] < hi[1] + gap:
            check(False, f"a line runs through {what}")


def _clear_pts(m, pts, gap=0.04, what="a label"):
    """Fail the render if a sampled curve (screen points) enters m's box."""
    pts = [to3(p) for p in pts]
    for p, q in zip(pts[:-1], pts[1:]):
        _clear(m, p, q, gap, what)


def _below(cap, *mobs, gap=0.03):
    """Fail the render if the caption reaches up into content above it."""
    c0, c1 = cap.get_critical_point(DL), cap.get_critical_point(UR)
    for m in mobs:
        m0, m1 = m.get_critical_point(DL), m.get_critical_point(UR)
        if m1[0] < c0[0] or m0[0] > c1[0]:
            continue
        check(c1[1] + gap <= m0[1], "caption clear of the content")


def _int(lo, hi, size=34, color=YELLOW_B):
    """An integral sign with small limits, built from Text pieces (no TeX)."""
    sgn = Text("∫", font_size=size * 1.45, color=color)
    up = Text(hi, font_size=size * 0.55, color=color)
    dn = Text(lo, font_size=size * 0.55, color=color)
    f = size / 34
    up.next_to(sgn, RIGHT, buff=0.03 * f).align_to(sgn, UP).shift(UP * 0.04 * f)
    dn.next_to(sgn, RIGHT, buff=0.03 * f).align_to(sgn, DOWN).shift(
        LEFT * 0.13 * f + DOWN * 0.04 * f)
    return VGroup(sgn, up, dn)


def _row(*parts, buff=0.14):
    """Parts side by side, their centres on one horizontal line."""
    return VGroup(*parts).arrange(RIGHT, buff=buff)


def _cap(group):
    """Place a composite formula where caption() puts its Text."""
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    group.set_x(0.0)
    group.to_edge(DOWN, buff=0.3)
    return group


_BASE_CHARS = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789+=")


def _baseline(t):
    """Baseline of a Text: the median bottom of its non-descending glyphs."""
    ys = [g.get_bottom()[1] for ch, g in zip(t.text, t.submobjects)
          if ch in _BASE_CHARS]
    if not ys:
        ys = [g.get_bottom()[1] for g in t.submobjects]
    return float(np.median(ys))


def _supline(parts, size=30, color=YELLOW_B, sup_scale=0.62, rise=0.48,
             drop=0.32):
    """A line of text with true superscripts and subscripts, without TeX
    and without the superscript/subscript letters some fonts lack.

    parts: list of (string, kind) with kind False/0 (normal), True/1
    (superscript) or -1 (subscript), or (string, kind, colour).  Normal
    pieces share one baseline; a superscript sits `rise` x-heights above
    it, a subscript `drop` x-heights below it, both smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for part in parts:
        s, kind = part[0], int(part[1])
        col = part[2] if len(part) > 2 else color
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        body = s.strip(" ")
        x += lead * sp
        if body:
            # the baseline is read off a reference capital that is then
            # dropped, so a piece made only of '=' or '+' sits right too
            t = Text("M" + body, font_size=size * (sup_scale if kind else 1.0),
                     color=col)
            ref = t.submobjects[0]
            base = ref.get_bottom()[1]
            t.remove(ref)
            y0 = rise * xh if kind > 0 else (-drop * xh if kind < 0 else 0.0)
            t.shift(np.array([x - t.get_left()[0], y0 - base, 0.0]))
            x = t.get_right()[0] + (0.01 if kind else 0.02)
            g.add(t)
        x += trail * sp
    return g


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


def _texts(*mobs):
    """Every Text inside the given mobjects."""
    return [t for m in mobs for t in m.get_family() if isinstance(t, Text)]


# ========================================================= H. CALCULUS & LIMITS

class H7_CosineDerivative(Board):
    """The small triangle of the sine derivative, read across.  On the unit
    circle an arc dθ at P cuts a small right triangle; its horizontal leg
    runs from the vertical through P to the vertical through the new point,
    so it is the drop of cos θ (the foot on the axis moves back by exactly
    that much).  The big triangle (1, cos θ, sin θ), shrunk by dθ and turned
    a quarter, is its limit shape: the tangent at P is perpendicular to OP.
    A window rescaled by 1/dθ shows the arc straightening and the drop
    converging to sin θ · dθ, leftwards: d(cos θ) = −sin θ · dθ."""

    def construct(self):
        th = 35 * DEGREES
        R = 5.4
        O = np.array([-6.2, -2.45, 0.0])

        def C(a):                                   # point of the circle
            return O + R * np.array([np.cos(a), np.sin(a), 0.0])

        P = C(th)
        F = np.array([P[0], O[1], 0.0])
        d0, d1 = 0.45, 0.01
        lt = ValueTracker(np.log(d0))

        def dth():
            return float(np.exp(lt.get_value()))

        # ---- the unit circle and the big triangle
        axes = VGroup(Line(O, O + RIGHT * (R + 0.45), color=GREY_B, stroke_width=2),
                      Line(O, O + UP * (R + 0.45), color=GREY_B, stroke_width=2))
        quarter = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O,
                      color=GREY_B, stroke_width=3)
        OP = Line(O, P, color=WHITE, stroke_width=4)
        OF = Line(O, F, color=TEAL_D, stroke_width=6)
        FP = Line(F, P, color=BLUE_D, stroke_width=6)
        ang = angle_arc(O, F, P, radius=0.75)
        l_th = tag("θ", 28, YELLOW_B).move_to(O + 1.05 * angle_mid_dir(O, F, P))
        l_1 = tag("1", 28).move_to((O + P) / 2 + 0.3 * np.array([-np.sin(th),
                                                                np.cos(th), 0]))
        l_cos = tag("cos θ", 26, TEAL_B).next_to(OF, DOWN, buff=0.12)
        l_sin = tag("sin θ", 26, BLUE_B).next_to(FP, LEFT, buff=0.14)
        dotP = Dot(P, radius=0.06)

        self.play(Create(axes), Create(quarter), run_time=1.0)
        self.play(Create(OP), FadeIn(dotP), Create(ang), FadeIn(l_th), run_time=1.0)
        self.play(Create(OF), Create(FP), FadeIn(l_cos), FadeIn(l_sin), FadeIn(l_1),
                  run_time=1.0)
        self.hold(0.4)

        # ---- an arc dθ beyond P and the small triangle it cuts
        def small():
            d = dth()
            P2 = C(th + d)
            Cn = np.array([P[0], P2[1], 0.0])
            return VGroup(
                Arc(radius=R, start_angle=th, angle=d, arc_center=O,
                    color=YELLOW_B, stroke_width=6),
                Line(P, Cn, color=BLUE_D, stroke_width=5),
                Line(Cn, P2, color=TEAL_D, stroke_width=5),
                Dot(P2, radius=0.05, color=YELLOW_B))

        sm = always_redraw(small)
        l_d = tag("dθ", 26, YELLOW_B).move_to(C(th + d0 / 2) - 0.42 * np.array(
            [np.cos(th + d0 / 2), np.sin(th + d0 / 2), 0.0]))
        self.play(FadeIn(sm), FadeIn(l_d), run_time=1.0)

        # ---- its horizontal leg is the drop of cos θ: the foot moves back
        P2_0 = C(th + d0)
        F2 = np.array([P2_0[0], O[1], 0.0])
        top = np.array([P[0], P2_0[1], 0.0])
        check(abs((top[0] - P2_0[0]) - (F[0] - F2[0])) < 1e-12
              and abs((F[0] - F2[0]) - R * (np.cos(th) - np.cos(th + d0))) < 1e-12,
              "horizontal leg = OF − OF₂ = cos θ − cos(θ + dθ)")
        drop = DashedLine(P2_0, F2, color=GREY_B, stroke_width=2, dash_length=0.09)
        fall = VGroup(Line(F2, F, color=TEAL_B, stroke_width=10),
                      Line(top, P2_0, color=TEAL_B, stroke_width=9))
        for lab in (l_1, l_cos, l_sin, l_d, l_th):
            _clear(lab, P2_0, F2, gap=0.05, what="the dashed drop")
        self.play(Create(drop), run_time=0.7)
        self.play(Create(fall), run_time=1.0)
        self.hold(0.6)

        # ---- the big triangle, shrunk by dθ and turned a quarter, at P
        big = mk([O, F, P], YELLOW_E, 0.35, stroke_width=2)
        cp = big.copy()
        self.add(cp)
        self.play(cp.animate.scale(d0, about_point=O).shift(P - O), run_time=1.4)
        self.play(Rotate(cp, angle=PI / 2, about_point=P), run_time=1.2)
        ref_main = [P, P + d0 * R * UP * np.cos(th),
                    P + d0 * R * np.array([-np.sin(th), np.cos(th), 0.0])]
        got = sorted(map(tuple, np.round(cp.get_vertices()[:, :2], 8)))
        want = sorted(map(tuple, np.round([v[:2] for v in ref_main], 8)))
        check(np.allclose(got, want, atol=1e-6),
              "quarter-turned copy has legs cos θ·dθ (up), sin θ·dθ (across)")

        def ref_copy():
            d = dth()
            return mk([P, P + d * R * UP * np.cos(th),
                       P + d * R * np.array([-np.sin(th), np.cos(th), 0.0])],
                      YELLOW_E, 0.35, stroke_width=2)

        self.remove(cp)
        rc = always_redraw(ref_copy)
        self.add(rc)
        self.bring_to_front(sm)
        self.hold(0.3)

        # ---- the window: everything near P magnified by L/dθ
        L = 4.4
        anc = np.array([3.79, -1.95, 0.0])           # P in the window
        sx, cy = np.sin(th), np.cos(th)
        # in the window the legs are the fall of cos and the rise of sin per
        # unit dθ; the fall is widest at the largest dθ shown
        fall0 = (np.cos(th) - np.cos(th + d0)) / d0
        wl, wr = anc[0] - L * fall0 - 0.3, anc[0] + 0.3
        wb, wt = anc[1] - 0.3, anc[1] + L * cy + 0.3
        for d in np.geomspace(d1, d0, 25):
            fl = (np.cos(th) - np.cos(th + d)) / d
            rs = (np.sin(th + d) - np.sin(th)) / d
            check(sx <= fl <= fall0 + 1e-12 and rs <= cy,
                  "the magnified triangle stays inside the window, and its "
                  "horizontal leg is never shorter than sin θ · dθ")
        window = Polygon([wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0],
                         stroke_color=GREY_B, stroke_width=2)

        def to_win(p):                               # screen near P -> window
            return anc + (p - P) * (L / (R * dth()))

        def zoom():
            d = dth()
            s = R * d / L
            lo = P + (np.array([wl, wb, 0]) - anc) * s
            hi = P + (np.array([wr, wt, 0]) - anc) * s
            box = Polygon(lo, [hi[0], lo[1], 0], hi, [lo[0], hi[1], 0],
                          stroke_color=GREY_B, stroke_width=1.5)
            return VGroup(box,
                          DashedLine([hi[0], hi[1], 0], [wl, wt, 0],
                                     color=GREY_D, stroke_width=1.2),
                          DashedLine([hi[0], lo[1], 0], [wl, wb, 0],
                                     color=GREY_D, stroke_width=1.2))

        ref_w = mk([anc, anc + L * cy * UP, anc + L * np.array([-sx, cy, 0])],
                   YELLOW_E, 0.30, stroke_width=0)
        ref_edge = DashedVMobject(Polygon(anc, anc + L * cy * UP,
                                          anc + L * np.array([-sx, cy, 0]),
                                          stroke_color=WHITE, stroke_width=2),
                                  num_dashes=40)

        def inside():
            d = dth()
            P2 = C(th + d)
            Cn = np.array([P[0], P2[1], 0.0])
            arc = _curve([to_win(C(th + t * d)) for t in np.linspace(0, 1, 60)],
                         YELLOW_B, 5)
            return VGroup(arc,
                          Line(to_win(P), to_win(Cn), color=BLUE_D, stroke_width=6),
                          Line(to_win(Cn), to_win(P2), color=TEAL_D, stroke_width=7),
                          Dot(anc, radius=0.06))

        zm = always_redraw(zoom)
        ins = always_redraw(inside)
        w_ang = angle_arc(anc, anc + UP, anc + np.array([-sx, cy, 0]), radius=0.85)
        w_th = tag("θ", 28, YELLOW_B).move_to(
            anc + 1.15 * angle_mid_dir(anc, anc + UP, anc + np.array([-sx, cy, 0])))
        mid_h = anc + 0.5 * L * np.array([-sx, cy, 0])
        w_d = tag("dθ", 28, YELLOW_B).move_to(mid_h + 0.40 * np.array([cy, sx, 0]))
        w_cos = tag("cos θ · dθ", 24).next_to(
            np.array([wr, anc[1] + L * cy * 0.55, 0]), RIGHT, buff=0.15)
        # over the window: the true drop (teal) and the limit leg (white)
        w_top = _row(tag("−Δ(cos θ)", 24, TEAL_B), tag("sin θ · dθ", 24), buff=0.45)
        w_top.next_to(np.array([(wl + wr) / 2, wt, 0]), UP, buff=0.14)
        w_drop, w_sin = w_top

        self.play(Create(window), FadeIn(zm), run_time=1.0)
        self.play(FadeIn(ref_w), Create(ref_edge), FadeIn(ins), Create(w_ang),
                  FadeIn(w_th), FadeIn(w_d), FadeIn(w_cos), FadeIn(w_drop),
                  FadeIn(w_sin), run_time=1.0)
        self.hold(0.5)

        # ---- dθ -> 0: the arc straightens, the drop becomes sin θ · dθ
        self.play(FadeOut(l_d), FadeOut(drop), FadeOut(fall), run_time=0.4)
        self.play(lt.animate.set_value(np.log(d1)), run_time=4.5, rate_func=smooth)
        fl = (np.cos(th) - np.cos(th + d1)) / d1
        check(abs(fl - np.sin(th)) * L < 0.03,
              "in the window the drop has converged to sin θ · dθ")
        for m in (sm, rc, zm, ins):
            m.clear_updaters()

        final = _row(tag("d(cos θ)", 26, TEAL_B), tag("=  −sin θ · dθ", 26),
                     buff=0.18)
        final.next_to(np.array([(wl + wr) / 2, wt, 0]), UP, buff=0.14)
        self.play(FadeOut(w_drop), FadeOut(w_sin), FadeIn(final), run_time=0.9)

        content = VGroup(axes, quarter, OP, OF, FP, ang, l_th, l_1, l_cos, l_sin,
                         sm, rc, window, zm, ref_w, ref_edge, ins, w_ang, w_th,
                         w_d, w_cos, final)
        _safe(*content, w_top)
        _apart(l_th, l_1, l_cos, l_sin, w_th, w_d, w_cos, w_drop, w_sin)
        _apart(l_th, l_1, l_cos, l_sin, w_th, w_d, w_cos, final)
        for lab in (w_drop, w_sin, final):
            _clear(lab, [wl, wt, 0], [wr, wt, 0], gap=0.05, what="the window edge")
        cap = caption("d(cos θ) / dθ  =  −sin θ", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H19_CavalieriPlane(Board):
    """Two regions with equal chords at every height: R₂ is R₁ with each
    horizontal chord slid sideways by s(y).  Cut R₁ into horizontal strips
    and move a copy of every strip sideways — a translation, so each strip
    keeps its area and the strips (in disjoint bands) never overlap: the
    stack has area A₁ exactly.  Cut every strip in two and slide the halves
    a little further: still A₁, and the stack hugs R₂ ever more closely
    (its edge is never more than max|s′| · ½(strip height) from R₂'s).  In
    the limit the stack is R₂, so A₂ = A₁."""

    def construct(self):
        r, a = 2.4, 0.9

        def s(y):                                  # the sideways slide
            return a * np.sin(PI * y / r)

        def w(y):                                  # chord length at height y
            return 2.0 * np.sqrt(max(r * r - y * y, 0.0))

        NS = 192                                   # samples up the height
        ys = np.linspace(-r, r, NS + 1)
        right2 = max(s(y) + w(y) / 2 for y in ys)
        left2 = min(s(y) - w(y) / 2 for y in ys)
        yc = (SAFE_TOP + SAFE_BOTTOM) / 2
        c1 = np.array([-SAFE_X + 0.2 + r, yc, 0.0])
        c2 = np.array([SAFE_X - 0.2 - right2, yc, 0.0])
        D = c2[0] - c1[0]
        check(c1[0] + r + 1.0 < c2[0] + left2, "the two regions stand apart")

        def P1(x, y):
            return c1 + np.array([x, y, 0.0])

        def outline(shift_of):
            """Sampled outline: left edge up, right edge down."""
            left = [(shift_of(y) - w(y) / 2, y) for y in ys]
            right = [(shift_of(y) + w(y) / 2, y) for y in ys[::-1]]
            pts, out = left + right, []
            for p in pts:
                if not out or abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) > 1e-12:
                    out.append(p)
            return out

        R1pts = outline(lambda y: 0.0)
        R2pts = outline(s)
        A1, A2 = abs(area(R1pts)), abs(area(R2pts))
        check(abs(A1 - A2) < 1e-9, "equal chords at every sampled height: "
              "the two outlines have the same area")

        def strip_pts(n, k):
            """Strip k of n of R1 (local coordinates), on the shared samples."""
            j0, j1 = k * NS // n, (k + 1) * NS // n
            seg = ys[j0:j1 + 1]
            pts = [(-w(y) / 2, y) for y in seg] + [(w(y) / 2, y) for y in seg[::-1]]
            out = []
            for p in pts:
                if not out or abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) > 1e-12:
                    out.append(p)
            return out, 0.5 * (seg[0] + seg[-1])

        cols = [BLUE_D, TEAL_D]
        sw = {6: 1.6, 12: 1.0, 24: 0.6, 48: 0.0}

        def strips(n, dx_of=None):
            """The n strips of R1, each moved sideways by dx_of(mid-height)."""
            g = VGroup()
            for k in range(n):
                pts, ym = strip_pts(n, k)
                dx = 0.0 if dx_of is None else dx_of(ym)
                g.add(mk([P1(x + dx, y) for x, y in pts], cols[k % 2], FILL,
                         stroke_width=sw[n], stroke_color=WHITE))
            return g

        for n in (6, 12, 24, 48):
            parts = [strip_pts(n, k) for k in range(n)]
            check(abs(sum(abs(area(p)) for p, _ in parts) - A1) < 1e-9,
                  f"the {n} strips fill R1 exactly")
            dev = max(abs(s(y) - s(ym)) for k, (p, ym) in enumerate(parts)
                      for y in np.linspace(-r + k * 2 * r / n, -r + (k + 1) * 2 * r / n, 9))
            check(dev <= a * PI / r * (r / n) + 1e-12,
                  "a slid strip is never further than max|s′|·½h from R2")

        # ---- the two regions: R1, and the outline of R2
        reg1 = Polygon(*[P1(x, y) for x, y in R1pts], fill_color=BLUE_D,
                       fill_opacity=0.55, stroke_color=WHITE, stroke_width=3)
        out2 = DashedVMobject(Polygon(*[P1(x + D, y) for x, y in R2pts],
                                      stroke_color=WHITE, stroke_width=2.5),
                              num_dashes=110)
        self.play(FadeIn(reg1), Create(out2), run_time=1.3)

        # ---- a horizontal line sweeps up: the two chords are equal
        yt = ValueTracker(-r + 0.03)
        xl, xr = c1[0] - r - 0.15, c2[0] + right2 + 0.15
        check(xl >= -SAFE_X and xr <= SAFE_X, "the sweeping line stays on screen")

        def sweep():
            y = yt.get_value()
            hw = w(y) / 2
            return VGroup(
                DashedLine([xl, yc + y, 0], [xr, yc + y, 0], color=GREY_B,
                           stroke_width=1.5, dash_length=0.08),
                Line(P1(-hw, y), P1(hw, y), color=ORANGE, stroke_width=7),
                Line(P1(D + s(y) - hw, y), P1(D + s(y) + hw, y), color=ORANGE,
                     stroke_width=7))

        sw_m = always_redraw(sweep)
        self.add(sw_m)
        self.play(yt.animate.set_value(r - 0.03), run_time=2.6, rate_func=linear)
        sw_m.clear_updaters()
        self.play(FadeOut(sw_m), run_time=0.4)

        # ---- cut R1 into strips; a copy of each strip moves across
        n = 6
        S1 = strips(n)
        cuts = VGroup(*[DashedLine([xl, yc - r + k * 2 * r / n, 0],
                                   [xr, yc - r + k * 2 * r / n, 0], color=GREY_B,
                                   stroke_width=1.5, dash_length=0.08)
                        for k in range(1, n)])
        self.play(Create(cuts), run_time=0.8)
        self.play(FadeIn(S1), FadeOut(reg1), run_time=0.7)
        S2 = S1.copy()
        self.add(S2)
        self.bring_to_front(out2)
        self.play(S2.animate.shift(D * RIGHT), run_time=1.4)
        # each strip slides sideways by its own amount
        mids = [strip_pts(n, k)[1] for k in range(n)]
        self.play(*[m.animate.shift(s(ym) * RIGHT) for m, ym in zip(S2, mids)],
                  run_time=1.6)
        want = strips(n, lambda y: D + s(y))
        check(all(close(m.get_vertices(), t.get_vertices(), 1e-6)
                  for m, t in zip(S2, want)), "each strip landed, slid by s(mid)")
        check(abs(sum(abs(area(m.get_vertices())) for m in S2)
                  - sum(abs(area(m.get_vertices())) for m in S1)) < 1e-9,
              "the slid stack has exactly the area of R1")
        self.hold(0.5)
        self.play(FadeOut(cuts), run_time=0.4)

        # ---- cut every strip in two and slide the halves a little further
        for n2, t_slide in ((12, 1.1), (24, 0.9), (48, 0.8)):
            parent = [strip_pts(n, k)[1] for k in range(n)]
            T1 = strips(n2)
            # halves at their parent's place, then each slides on by its own
            # correction s(own mid) − s(parent mid)
            T2 = strips(n2, lambda y, _p=parent, _n=n2: D + s(
                _p[int((y + r) // (2 * r / n))]))
            self.remove(S1, S2)
            self.add(T1, T2)
            self.bring_to_front(out2)
            self.hold(0.3)
            mids2 = [strip_pts(n2, k)[1] for k in range(n2)]
            self.play(*[m.animate.shift((s(ym) - s(parent[k // 2])) * RIGHT)
                        for k, (m, ym) in enumerate(zip(T2, mids2))],
                      run_time=t_slide)
            want = strips(n2, lambda y: D + s(y))
            check(all(close(m.get_vertices(), t.get_vertices(), 1e-6)
                      for m, t in zip(T2, want)), f"n = {n2}: every half landed")
            check(abs(sum(abs(area(m.get_vertices())) for m in T2) - A1) < 1e-6,
                  f"n = {n2}: still exactly the area of R1")
            S1, S2, n = T1, T2, n2

        # ---- in the limit the stack is R2
        reg1 = Polygon(*[P1(x, y) for x, y in R1pts], fill_color=BLUE_D,
                       fill_opacity=FILL, stroke_color=WHITE, stroke_width=3)
        reg2 = Polygon(*[P1(x + D, y) for x, y in R2pts], fill_color=BLUE_D,
                       fill_opacity=FILL, stroke_color=WHITE, stroke_width=3)
        l1 = tag("A₁", 40).move_to(c1)
        l2 = tag("A₂", 40).move_to(c2)
        self.play(FadeOut(S1), FadeOut(S2), FadeIn(reg1), FadeIn(reg2),
                  FadeOut(out2), run_time=1.0)
        self.play(FadeIn(l1), FadeIn(l2), run_time=0.6)
        check(_pip(l1.get_corner(DL), [P1(x, y) for x, y in R1pts])
              and _pip(l1.get_corner(UR), [P1(x, y) for x, y in R1pts])
              and _pip(l2.get_corner(DL), [P1(x + D, y) for x, y in R2pts])
              and _pip(l2.get_corner(UR), [P1(x + D, y) for x, y in R2pts]),
              "each label inside its region")
        _safe(reg1, reg2, l1, l2)
        cap = caption("equal chords at every height   ⟹   A₁ = A₂", 34)
        _below(cap, reg1, reg2)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


def _frac(num, den, gap=0.08, pad=0.08, color=WHITE, width=2.5):
    """A stacked fraction from two mobjects: num over a bar over den."""
    w = max(num.width, den.width) + 2 * pad
    bar = Line(LEFT * w / 2, RIGHT * w / 2, color=color, stroke_width=width)
    num.next_to(bar, UP, buff=gap)
    den.next_to(bar, DOWN, buff=gap)
    return VGroup(num, bar, den)


def _eq_row(left, frac, ref, size=30, buff=0.2, color=WHITE):
    """left  =  frac, with '=' and the fraction bar on the centre line of
    the glyph `ref` inside `left` (superscripts do not shift the row)."""
    eq = Text("=", font_size=size, color=color)
    yc = ref.get_center()[1]
    eq.next_to(left, RIGHT, buff=buff).set_y(yc)
    frac.next_to(eq, RIGHT, buff=buff)
    frac.shift(UP * (yc - frac[1].get_center()[1]))
    return VGroup(left, eq, frac)


class H25_InverseDerivative(Board):
    """Reflect the graph of f, its tangent at (x, y) and the slope triangle
    of that tangent (run 1, rise f′(x)) in the line y = x — a fold, so a
    rigid motion.  The graph lands on the graph of the inverse, the tangent
    on its tangent at (y, x), and the triangle's horizontal and vertical
    legs trade places: the run is now f′(x) and the rise 1, so the slope of
    the inverse there is 1 / f′(x).  (Drawn with an increasing f.)"""

    def construct(self):
        def f(x):
            return x * x / 5.0

        def fp(x):
            return 2.0 * x / 5.0

        def g(y):                                   # the inverse of f
            return np.sqrt(5.0 * y)

        x0 = 3.6
        y0, m = f(x0), fp(x0)
        F = Frame(-0.35, 4.95, -0.35, 4.95, max_w=6.9,
                  centre=(-3.05, (SAFE_TOP + SAFE_BOTTOM) / 2))
        P = F.P

        def refl(p):                                # mirror in y = x (math)
            return (p[1], p[0])

        Q, Rc = (x0 - 1.0, y0 - m), (x0, y0 - m)    # run 1, rise f′(x)
        t0, t1 = 2.3, 4.4                           # the drawn tangent piece
        T0, T1 = (t0, y0 + m * (t0 - x0)), (t1, y0 + m * (t1 - x0))
        xs = np.linspace(0.0, 4.75, 240)
        # ---- the claims
        check(abs(g(f(x0)) - x0) < 1e-12 and all(abs(g(f(x)) - x) < 1e-12 for x in xs),
              "g undoes f")
        hstep = 1e-6
        gp = (g(y0 + hstep) - g(y0 - hstep)) / (2 * hstep)
        check(abs(gp - 1.0 / m) < 1e-6, "slope of the inverse at y is 1 / f′(x)")
        check(abs((refl(T1)[1] - refl(T0)[1]) / (refl(T1)[0] - refl(T0)[0]) - 1 / m)
              < 1e-12, "the mirrored tangent has slope 1 / f′(x)")
        check(all(f(x) >= y0 + m * (x - x0) - 1e-12 for x in xs),
              "f lies above its tangent (the triangle sits under the curve)")

        # ---- figure
        axes = VGroup(Line(P((-0.3, 0)), P((4.9, 0)), color=GREY_B, stroke_width=2),
                      Line(P((0, -0.3)), P((0, 4.9)), color=GREY_B, stroke_width=2))
        diag = DashedLine(P((-0.3, -0.3)), P((4.85, 4.85)), color=GREY_A,
                          stroke_width=2.5, dash_length=0.1)
        l_diag = tag("y = x", 26, GREY_A).move_to(P((1.95, 1.38)))
        c_f, c_g, c_run, c_rise = BLUE_C, PURPLE_B, GREEN_C, ORANGE
        curve = _curve([P((x, f(x))) for x in xs], c_f, 5)
        l_f = tag("f", 32, c_f).move_to(P((4.66, 3.38)))
        self.play(Create(axes), Create(diag), FadeIn(l_diag), run_time=1.0)
        self.play(Create(curve), FadeIn(l_f), run_time=1.2)

        tan = Line(P(T0), P(T1), color=WHITE, stroke_width=3.5)
        dotP = Dot(P((x0, y0)), radius=0.07, color=WHITE)
        l_P = tag("(x, y)", 26).move_to(P((4.25, 2.38)))
        run = Line(P(Q), P(Rc), color=c_run, stroke_width=7)
        rise = Line(P(Rc), P((x0, y0)), color=c_rise, stroke_width=7)
        l_run = tag("1", 28, c_run).next_to(run, DOWN, buff=0.12)
        l_rise = tag("f′(x)", 28, c_rise).next_to(rise, RIGHT, buff=0.14).shift(
            DOWN * 0.3)
        ra = _ra(P(Rc), P(Q), P((x0, y0)), 0.17)
        self.play(Create(tan), FadeIn(dotP), FadeIn(l_P), run_time=1.0)
        self.play(Create(run), Create(rise), Create(ra), FadeIn(l_run),
                  FadeIn(l_rise), run_time=1.1)

        # ---- the slope at (x, y): rise over run
        px = 3.75
        r1_left = tag("f′(x)", 34)
        row1 = _eq_row(r1_left, _frac(tag("f′(x)", 32, c_rise), tag("1", 32, c_run)),
                       r1_left[0], 34)
        row1.move_to([px, 1.55, 0.0])
        self.play(FadeIn(row1, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.5)

        # ---- fold everything over y = x
        mob = VGroup(curve.copy(), tan.copy(), run.copy(), rise.copy(), ra.copy(),
                     dotP.copy())
        self.add(mob)
        axis = _unit(np.array([1.0, 1.0, 0.0]))
        self.play(Rotate(mob, angle=PI, axis=axis, about_point=P((0, 0))),
                  run_time=2.2)
        # the fold lands every point on its mirror image
        c_m, tan_m, run_m, rise_m, ra_m, dot_m = mob
        check(close(tan_m.get_start(), P(refl(T0)), 1e-6)
              and close(tan_m.get_end(), P(refl(T1)), 1e-6), "tangent landed")
        check(close(run_m.get_start(), P(refl(Q)), 1e-6)
              and close(run_m.get_end(), P(refl(Rc)), 1e-6)
              and close(rise_m.get_end(), P((y0, x0)), 1e-6), "triangle landed")
        check(abs(run_m.get_start()[0] - run_m.get_end()[0]) < 1e-6
              and abs(rise_m.get_start()[1] - rise_m.get_end()[1]) < 1e-6,
              "the run (1) is now vertical, the rise (f′(x)) horizontal")
        pts = c_m.get_anchors()
        o = P((0, 0))
        # a point (X, Y) of the folded curve has X = f(Y): Y = f⁻¹(X)
        check(all(abs((p[0] - o[0]) / F.k - f((p[1] - o[1]) / F.k)) < 1e-6
                  and abs(p[2]) < 1e-6 for p in pts),
              "the folded curve is the graph of the inverse")
        self.play(c_m.animate.set_color(c_g), run_time=0.5)
        l_g = _supline([("f", 0), ("−1", 1)], 32, c_g).move_to(P((3.38, 4.66)))
        l_P2 = tag("(y, x)", 26).move_to(P((2.38, 4.25)))
        l_run2 = tag("1", 28, c_run).next_to(run_m, LEFT, buff=0.14)
        l_rise2 = tag("f′(x)", 28, c_rise).next_to(rise_m, UP, buff=0.12).shift(
            LEFT * 0.3)
        self.play(FadeIn(l_g), FadeIn(l_P2), FadeIn(l_run2), FadeIn(l_rise2),
                  run_time=0.9)
        # rise and run have traded places
        self.play(Indicate(VGroup(run, run_m), color=c_run, scale_factor=1.0),
                  run_time=0.8)
        self.play(Indicate(VGroup(rise, rise_m), color=c_rise, scale_factor=1.0),
                  run_time=0.8)

        r2_left = _supline([("(f", 0), ("−1", 1), (")′(y)", 0)], 34, WHITE)
        row2 = _eq_row(r2_left, _frac(tag("1", 32, c_run), tag("f′(x)", 32, c_rise)),
                       r2_left[0], 34)
        row2.move_to([px, -0.45, 0.0])
        self.play(FadeIn(row2, shift=RIGHT * 0.2), run_time=0.8)

        labels = [l_diag, l_f, l_P, l_run, l_rise, l_g, l_P2, l_run2, l_rise2]
        lines = [(P((-0.3, 0)), P((4.9, 0))), (P((0, -0.3)), P((0, 4.9))),
                 (P((-0.3, -0.3)), P((4.85, 4.85))), (P(T0), P(T1)),
                 (P(refl(T0)), P(refl(T1))), (P(Q), P(Rc)), (P(Rc), P((x0, y0))),
                 (P(refl(Q)), P(refl(Rc))), (P(refl(Rc)), P((y0, x0)))]
        for lab in labels:
            for p, q in lines:
                _clear(lab, p, q, gap=0.04, what=f"label {_texts(lab)[0].text!r}")
            _clear_pts(lab, [P((x, f(x))) for x in xs], 0.04, "a label (curve f)")
            _clear_pts(lab, [P((f(x), x)) for x in xs], 0.04, "a label (inverse)")
        _apart(*labels, gap=0.06)
        _safe(axes, diag, curve, c_m, row1, row2, *labels)
        check(row1.get_left()[0] > P((4.95, 0))[0] + 0.3, "panel clear of the figure")
        cap = _cap(_supline([("(f", 0), ("−1", 1), (")′(y)   =   1 / f′(x)", 0)], 36))
        _below(cap, axes, diag, row2, *labels)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)



def _pieces(xs, ys, t):
    """The parts of the region under the sampled graph (xs, ys) that lie
    above the level t ('hills') and the parts of the rectangle under t that
    lie above the graph ('valleys'), as polygons in (x, y): on the
    piecewise-linear graph through the samples, crossings interpolated."""
    X, Y = [float(xs[0])], [float(ys[0])]
    for k in range(len(xs) - 1):
        y0, y1 = ys[k] - t, ys[k + 1] - t
        if y0 * y1 < 0:
            X.append(float(xs[k] + (xs[k + 1] - xs[k]) * y0 / (y0 - y1)))
            Y.append(float(t))
        X.append(float(xs[k + 1]))
        Y.append(float(ys[k + 1]))
    hills, valleys = [], []
    run = [0]
    for i in range(1, len(X)):
        run.append(i)
        if abs(Y[i] - t) < 1e-15 or i == len(X) - 1:
            j = max(run, key=lambda q: abs(Y[q] - t))
            sgn = np.sign(Y[j] - t)
            pts = ([(X[run[0]], t)] + [(X[q], Y[q]) for q in run]
                   + [(X[run[-1]], t)])
            clean = []
            for p in pts:
                if not clean or abs(p[0] - clean[-1][0]) + abs(p[1] - clean[-1][1]) > 1e-13:
                    clean.append(p)
            if sgn != 0 and len(clean) >= 3 and abs(area(clean)) > 1e-12:
                (hills if sgn > 0 else valleys).append(clean)
            run = [i]
    return hills, valleys


def _under_level(xs, ys, t):
    """The region under min(graph, t), as one polygon (crossings
    interpolated, as in _pieces)."""
    X, Y = [float(xs[0])], [float(ys[0])]
    for k in range(len(xs) - 1):
        y0, y1 = ys[k] - t, ys[k + 1] - t
        if y0 * y1 < 0:
            X.append(float(xs[k] + (xs[k + 1] - xs[k]) * y0 / (y0 - y1)))
            Y.append(float(t))
        X.append(float(xs[k + 1]))
        Y.append(float(ys[k + 1]))
    return [(X[0], 0.0)] + [(x, min(y, t)) for x, y in zip(X, Y)] + [(X[-1], 0.0)]


class H23_IntegralMeanValue(Board):
    """A = area under f on [a, b].  Raise a level line t from the lowest
    value m of f to the highest, M.  The region's parts above the line
    (hills) shrink to nothing while the gaps between the curve and the line
    (valleys) grow from nothing; hills − valleys = A − t·(b − a) changes
    steadily, so at one level h they balance: there the region levels into
    the rectangle (b − a) × h of the same area, and m ≤ h ≤ M.  The arc of
    the curve from its lowest point to its highest runs from under that
    line to over it, so it crosses it at some c: h = f(c)."""

    def construct(self):
        a, b = 0.6, 5.6

        def f(x):
            return 2.0 - np.cos(PI * (x - a - 1.3) / 2.5) + 0.12 * (x - a - 2.5)

        kx, ky = 1.33, 1.55
        O = np.array([-5.55, -2.45, 0.0])

        def P(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        xs = np.linspace(a, b, 241)
        ys = f(xs)
        region = [(a, 0.0)] + list(zip(xs, ys)) + [(b, 0.0)]
        A = abs(area(region))
        h = A / (b - a)
        i_m, i_M = int(np.argmin(ys)), int(np.argmax(ys))
        m, M = float(ys[i_m]), float(ys[i_M])
        x_m, x_M = float(xs[i_m]), float(xs[i_M])

        def balance(t):
            H, V = _pieces(xs, ys, t)
            return (sum(abs(area(p)) for p in H), sum(abs(area(p)) for p in V))

        # ---- the claims
        check(a < x_m < x_M < b, "lowest and highest points inside [a, b]")
        hm, vm = balance(m)
        hM, vM = balance(M)
        check(vm < 1e-12 and abs(hm - (A - m * (b - a))) < 1e-9,
              "at t = m: no valleys (the m-rectangle lies under the graph)")
        check(hM < 1e-12 and abs(vM - (M * (b - a) - A)) < 1e-9,
              "at t = M: no hills (the region lies in the M-rectangle)")
        for t in np.linspace(m, M, 41):
            hh, vv = balance(t)
            check(abs((hh - vv) - (A - t * (b - a))) < 1e-9,
                  "hills − valleys = A − t·(b − a)")
        hh, vv = balance(h)
        check(abs(hh - vv) < 1e-9 and m < h < M, "balance at h, between m and M")
        lo, hi = x_m, x_M                          # f(x_m) < h < f(x_M)
        g = lambda x: np.interp(x, xs, ys) - h
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
        c = (lo + hi) / 2
        check(abs(np.interp(c, xs, ys) - h) < 1e-9 and x_m < c < x_M,
              "the arc from the lowest to the highest point meets the level at c")
        check(all((np.interp(x, xs, ys) - h) * (x - c) > 0
                  for x in np.linspace(a, b, 2001) if abs(x - c) > 1e-6),
              "a single crossing (one c) for this f")

        # ---- figure
        axes = VGroup(Line(P(0, 0), P(6.05, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 3.55), color=GREY_B, stroke_width=2))
        x_lo, x_hi = 0.3, 5.95
        cx = np.linspace(x_lo, x_hi, 300)
        curve = _curve([P(x, f(x)) for x in cx], YELLOW_B, 5)
        l_f = tag("f", 30, YELLOW_B).next_to(P(x_hi, f(x_hi)), RIGHT, buff=0.12)
        reg = mk([P(x, y) for x, y in region], BLUE_D, 0.55, stroke_width=0)
        ea = DashedLine(P(a, 0), P(a, f(a)), color=GREY_B, stroke_width=2,
                        dash_length=0.08)
        eb = DashedLine(P(b, 0), P(b, f(b)), color=GREY_B, stroke_width=2,
                        dash_length=0.08)
        cap_h = Text("M", font_size=28).height

        def under(s, x, col=WHITE):
            t = tag(s, 28, col)
            t.move_to(P(x, 0) + DOWN * (0.16 + cap_h / 2))
            return t

        l_a, l_b = under("a", a), under("b", b)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_f), run_time=1.1)
        self.add(reg)
        self.bring_to_back(reg)
        self.play(FadeIn(reg), Create(ea), Create(eb), FadeIn(l_a), FadeIn(l_b),
                  run_time=0.9)

        # ---- the lowest and the highest value of f on [a, b]
        dm, dM = Dot(P(x_m, m), radius=0.08), Dot(P(x_M, M), radius=0.08)
        gm = DashedLine(P(0, m), P(x_m, m), color=GREY_B, stroke_width=1.8,
                        dash_length=0.08)
        gM = DashedLine(P(0, M), P(x_M, M), color=GREY_B, stroke_width=1.8,
                        dash_length=0.08)
        l_m = tag("m", 28).next_to(P(0, m), LEFT, buff=0.14)
        l_M = tag("M", 28).next_to(P(0, M), LEFT, buff=0.14)
        self.play(FadeIn(dm), FadeIn(dM), Create(gm), Create(gM), FadeIn(l_m),
                  FadeIn(l_M), run_time=1.0)

        # ---- a level line rises from m to M: hills shrink, valleys grow
        lv = ValueTracker(m)
        c_h, c_v = ORANGE, TEAL_D
        sc = 0.48                                    # bar length per unit area
        bx, by_h, by_v = 3.55, 2.55, 1.95

        def level():
            t = lv.get_value()
            H, V = _pieces(xs, ys, t)
            g_ = VGroup(*[mk([P(x, y) for x, y in p], c_h, 0.85, stroke_width=0)
                          for p in H],
                        *[mk([P(x, y) for x, y in p], c_v, 0.85, stroke_width=0)
                          for p in V])
            g_.add(Line(P(a, t), P(b, t), color=WHITE, stroke_width=4))
            sh = sum(abs(area(p)) for p in H)
            sv = sum(abs(area(p)) for p in V)
            for y_, s_, col in ((by_h, sh, c_h), (by_v, sv, c_v)):
                if s_ * sc > 1e-3:
                    g_.add(Rectangle(width=s_ * sc, height=0.36, fill_color=col,
                                     fill_opacity=0.9, stroke_width=0)
                           .move_to([bx + s_ * sc / 2, y_, 0.0]))
            return g_

        base = Line([bx, by_v - 0.3, 0], [bx, by_h + 0.3, 0], color=GREY_B,
                    stroke_width=2)
        lev = always_redraw(level)
        self.add(lev, base)
        self.bring_to_front(curve, dm, dM)
        self.play(lv.animate.set_value(M), run_time=2.8, rate_func=linear)
        self.hold(0.3)
        self.play(lv.animate.set_value(h), run_time=1.6, rate_func=smooth)
        lev.clear_updaters()
        check(abs(lv.get_value() - h) < 1e-12, "stopped at the balance level")
        bal = DashedLine([bx + hh * sc, by_v - 0.3, 0], [bx + hh * sc, by_h + 0.3, 0],
                         color=WHITE, stroke_width=2, dash_length=0.07)
        self.play(Create(bal), run_time=0.5)

        # ---- the arc from the lowest to the highest point crosses the level
        arc = _curve([P(x, f(x)) for x in np.linspace(x_m, x_M, 120)], WHITE, 8)
        self.play(Create(arc), run_time=0.9)
        dc = Dot(P(c, h), radius=0.09, color=WHITE)
        drop_c = DashedLine(P(c, h), P(c, 0), color=WHITE, stroke_width=2,
                            dash_length=0.08)
        l_c = under("c", c)
        g_h = DashedLine(P(0, h), P(a, h), color=GREY_B, stroke_width=1.8,
                         dash_length=0.08)
        l_fc = tag("f(c)", 28).next_to(P(0, h), LEFT, buff=0.14)
        self.play(FadeIn(dc), Create(drop_c), FadeIn(l_c), Create(g_h), FadeIn(l_fc),
                  FadeOut(arc), run_time=1.0)

        # ---- hills and valleys balance: the region levels into the rectangle
        H, V = _pieces(xs, ys, h)
        hills = VGroup(*[mk([P(x, y) for x, y in p], c_h, 0.85, stroke_width=0)
                         for p in H])
        vals = VGroup(*[mk([P(x, y) for x, y in p], c_v, 0.85, stroke_width=0)
                        for p in V])
        self.remove(lev)
        bars = VGroup(Rectangle(width=hh * sc, height=0.36, fill_color=c_h,
                                fill_opacity=0.9, stroke_width=0)
                      .move_to([bx + hh * sc / 2, by_h, 0.0]),
                      Rectangle(width=vv * sc, height=0.36, fill_color=c_v,
                                fill_opacity=0.9, stroke_width=0)
                      .move_to([bx + vv * sc / 2, by_v, 0.0]))
        topline = Line(P(a, h), P(b, h), color=WHITE, stroke_width=4)
        # the region = its part under the level + the hills (same look)
        low = _under_level(xs, ys, h)
        check(abs(abs(area(low)) + hh - A) < 1e-9, "region = part under h + hills")
        reg_low = mk([P(x, y) for x, y in low], BLUE_D, 0.55, stroke_width=0)
        reg_high = VGroup(*[mk([P(x, y) for x, y in p], BLUE_D, 0.55, stroke_width=0)
                            for p in H])
        self.remove(reg)
        self.add(reg_low, reg_high, hills, vals, bars, topline)
        self.bring_to_back(reg_low, reg_high)
        self.bring_to_front(curve, dm, dM, dc, base, bal)
        rect = Polygon(P(a, 0), P(b, 0), P(b, h), P(a, h), stroke_color=YELLOW_B,
                       stroke_width=5)
        check(abs(abs(area(low)) + vv - h * (b - a)) < 1e-9,
              "part under h + valleys = the rectangle (b − a) × h")
        self.play(hills.animate.set_fill(opacity=0.0),
                  reg_high.animate.set_fill(opacity=0.0),
                  vals.animate.set_fill(BLUE_D, opacity=0.55), run_time=1.6)
        self.play(Create(rect), run_time=0.8)
        reg = reg_low

        labels = [l_f, l_a, l_b, l_c, l_m, l_M, l_fc]
        _apart(*labels, gap=0.06)
        _safe(axes, curve, reg, rect, bars, base, bal, *labels)
        for lab in labels:
            _clear_pts(lab, [P(x, f(x)) for x in cx], 0.04, "a label (curve)")
            for p, q in ((P(a, h), P(b, h)), (P(c, h), P(c, 0)), (P(a, 0), P(a, f(a))),
                         (P(b, 0), P(b, f(b)))):
                _clear(lab, p, q, 0.04, "a label")
        check(bars.get_left()[0] > P(6.05, 0)[0] + 0.3, "bars clear of the figure")
        cap = _cap(_row(_int("a", "b", 34), tag("f(x) dx   =   f(c) · (b − a)", 34,
                                                  YELLOW_B), buff=0.12))
        _below(cap, axes, *labels)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


class H22_OddSymmetric(Board):
    """f odd: f(−x) = −f(x), i.e. the graph is its own image under the
    half-turn about the origin.  So the half-turn carries each piece of
    the region between the graph and the axis on [0, a] onto the piece at
    the mirrored place on [−a, 0], above the axis where it was below and
    below where it was above: congruent pieces of opposite sign.  The
    signed areas cancel in pairs, A − A + B − B = 0."""

    def construct(self):
        r0, s0, a = 1.6, 3.0, 2.8

        def f(x):
            return 0.9 * x * (x * x - r0 * r0) * np.exp(-x * x / s0)

        kx, ky = 2.15, 2.0
        O = np.array([0.0, 0.15, 0.0])

        def P(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        xr = np.linspace(0.0, a, 281)
        check(max(abs(f(-x) + f(x)) for x in xr) < 1e-12, "f is odd")
        check(abs(f(r0)) < 1e-12 and all(f(x) < 0 for x in xr if 0 < x < r0)
              and all(f(x) > 0 for x in xr if r0 < x <= a),
              "on [0, a]: below the axis on (0, r), above on (r, a]")
        inner = [(0.0, 0.0)] + [(x, f(x)) for x in xr if x <= r0] + [(r0, 0.0)]
        outer = [(r0, 0.0)] + [(x, f(x)) for x in xr if x >= r0] + [(a, 0.0)]

        def turned(poly):
            return [(-x, -y) for x, y in poly]

        inner_l, outer_l = turned(inner), turned(outer)
        A_, B_ = abs(area(inner)), abs(area(outer))
        check(abs(abs(area(inner_l)) - A_) < 1e-12 and abs(abs(area(outer_l)) - B_)
              < 1e-12, "the half-turned pieces have the same areas")
        # the left pieces ARE the region of the graph on [−a, 0]
        check(all(abs(y - f(x)) < 1e-12 for x, y in inner_l[1:-1] + outer_l[1:-1]),
              "the half-turned pieces lie on the graph of f over [−a, 0]")
        check(all(y > 0 for x, y in inner_l[2:-2]) and all(y < 0 for x, y in outer_l[2:-2]),
              "signs swap: −A becomes +A, +B becomes −B")
        signed = [area_signed for area_signed in
                  (-A_, +B_, +A_, -B_)]
        check(abs(sum(signed)) < 1e-12, "the signed areas cancel")

        # ---- figure
        axes = VGroup(Line(P(-a - 0.25, 0), P(a + 0.25, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, -1.35), P(0, 1.35), color=GREY_B, stroke_width=2))
        dotO = Dot(O, radius=0.06)
        ticks = VGroup(*[Line(P(x, 0) + DOWN * 0.08, P(x, 0) + UP * 0.08,
                              color=GREY_B, stroke_width=2) for x in (-a, a)])
        l_a = tag("a", 28).next_to(P(a, 0), DOWN, buff=0.16)
        l_ma = tag("−a", 28).next_to(P(-a, 0), UP, buff=0.16)
        xs = np.linspace(-a, a, 561)
        curve = _curve([P(x, f(x)) for x in xs], YELLOW_B, 5)
        l_f = tag("f", 30, YELLOW_B).next_to(P(a, f(a)), RIGHT, buff=0.12)
        self.play(Create(axes), FadeIn(dotO), Create(ticks), FadeIn(l_a), FadeIn(l_ma),
                  run_time=1.0)
        self.play(Create(curve), FadeIn(l_f), run_time=1.3)

        c_pos, c_neg = BLUE_D, ORANGE
        p_in = mk([P(*q) for q in inner], c_neg, 0.7, stroke_width=0)
        p_out = mk([P(*q) for q in outer], c_pos, 0.7, stroke_width=0)
        p_in_l = mk([P(*q) for q in inner_l], c_pos, 0.7, stroke_width=0)
        p_out_l = mk([P(*q) for q in outer_l], c_neg, 0.7, stroke_width=0)
        pieces = VGroup(p_in, p_out, p_in_l, p_out_l)
        self.add(pieces)
        self.bring_to_back(pieces)
        self.play(FadeIn(pieces), run_time=0.9)

        def lab(s, poly, col):
            pts = np.array(poly, float)
            cx_, cy_ = np.mean(pts[1:-1, 0]), np.mean(pts[1:-1, 1]) * 0.75
            return tag(s, 30, col).move_to(P(cx_, cy_))

        l_in = lab("−A", inner, WHITE)
        l_out = lab("+B", outer, WHITE)
        self.play(FadeIn(l_in), FadeIn(l_out), run_time=0.6)

        # ---- the half-turn about O, done as its two folds (it never leaves
        # the frame): x → −x, the graph of f(−x); then y → −y
        cp = VGroup(mk([P(*q) for q in inner], c_neg, 0.55, stroke_color=WHITE,
                       stroke_width=4),
                    mk([P(*q) for q in outer], c_pos, 0.55, stroke_color=WHITE,
                       stroke_width=4),
                    _curve([P(x, f(x)) for x in xr], YELLOW_B, 7))
        # the copy appears in place, under the labels of the right half
        cp[0].set_fill(opacity=0.0).set_stroke(opacity=0.0)
        cp[1].set_fill(opacity=0.0).set_stroke(opacity=0.0)
        cp[2].set_stroke(opacity=0.0)
        self.add(cp)
        self.bring_to_front(l_in, l_out)
        self.play(cp[0].animate.set_fill(opacity=0.55).set_stroke(opacity=1.0),
                  cp[1].animate.set_fill(opacity=0.55).set_stroke(opacity=1.0),
                  cp[2].animate.set_stroke(opacity=1.0), run_time=0.4)
        # (the mirrored copy passes over the −a label: it steps aside)
        self.play(Rotate(cp, angle=PI, axis=UP, about_point=O), FadeOut(l_ma),
                  run_time=1.6)
        check(np.allclose(cp[0].get_vertices(), [P(-x, y) for x, y in inner], atol=1e-6)
              and np.allclose(cp[1].get_vertices(), [P(-x, y) for x, y in outer],
                              atol=1e-6), "first fold: the mirror image in the y-axis")
        l_fm = tag("f(−x)", 28, YELLOW_B).next_to(
            P(-0.5 * (r0 + a), max(f(x) for x in xr)), UP, buff=0.25)
        self.play(FadeIn(l_fm), run_time=0.5)
        self.hold(0.3)
        self.play(Rotate(cp, angle=PI, axis=RIGHT, about_point=O), FadeOut(l_fm),
                  run_time=1.6)
        check(np.allclose(cp[0].get_vertices(), [P(*q) for q in inner_l], atol=1e-6)
              and np.allclose(cp[1].get_vertices(), [P(*q) for q in outer_l], atol=1e-6),
              "the two folds (= the half-turn) land each piece on its partner")
        check(all(abs(p[2]) < 1e-6 for p in cp[2].get_anchors())
              and np.allclose(np.array(cp[2].get_anchors())[[0, -1]],
                              [P(0, 0), P(-a, f(-a))], atol=1e-6),
              "the right half of the graph lands on the left half")
        self.play(Indicate(cp, color=WHITE, scale_factor=1.0), FadeIn(l_ma),
                  run_time=0.6)
        l_in_l = lab("+A", inner_l, WHITE)
        l_out_l = lab("−B", outer_l, WHITE)
        self.play(FadeOut(cp), FadeIn(l_in_l), FadeIn(l_out_l), run_time=0.7)

        # ---- the pairs cancel
        labels = [l_a, l_ma, l_f, l_in, l_out, l_in_l, l_out_l]
        row = _supline([("A", 0, c_pos), (" − A", 0, c_neg), ("  + B", 0, c_pos),
                        (" − B", 0, c_neg), ("   =   0", 0, WHITE)], 34)
        row.move_to([0.0, 3.2, 0.0])
        def rims(*polys):
            return VGroup(*[Polygon(*[P(*q) for q in poly], stroke_color=WHITE,
                                    stroke_width=6, fill_opacity=0.0)
                            for poly in polys])

        rA, rB = rims(inner, inner_l), rims(outer, outer_l)
        self.play(Create(rA), FadeIn(VGroup(*row[:2])), run_time=1.0)
        self.play(FadeOut(rA), run_time=0.3)
        self.play(Create(rB), FadeIn(VGroup(*row[2:4])), run_time=1.0)
        self.play(FadeOut(rB), run_time=0.3)
        self.play(FadeIn(row[4]), run_time=0.6)
        _apart(l_fm, l_ma, l_f, gap=0.06)
        _clear_pts(l_fm, [P(-x, f(x)) for x in xr], 0.04, "the label f(−x)")

        _apart(*labels, row, gap=0.06)
        for t_, poly in ((l_in, inner), (l_out, outer), (l_in_l, inner_l),
                         (l_out_l, outer_l)):
            scr = [P(*q) for q in poly]
            check(all(_pip(t_.get_corner(v), scr) for v in (UL, UR, DL, DR)),
                  f"label {t_.text} inside its piece")
        for t_ in (l_a, l_ma, l_f):
            _clear_pts(t_, [P(x, f(x)) for x in xs], 0.04, "a label (curve)")
        _safe(axes, curve, pieces, row, *labels)
        cap = _cap(_row(_int("−a", "a", 34), tag("f(x) dx   =   0        "
                                                   "( f(−x) = −f(x) )", 34, YELLOW_B),
                        buff=0.12))
        _below(cap, axes, pieces, *labels)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# ========================================================= L. VECTORS

def _arrow(p, q, color, width=6):
    return Arrow(to3(p), to3(q), buff=0.0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.1, tip_length=0.26)


def _arc_pts(c, r, a0, a1, n=24):
    c = to3(c)
    return [c + r * _dir(t) for t in np.linspace(a0, a1, n)]


class L2_DotProjection(Board):
    """u at angle α to the x-axis, v at angle θ beyond it.  The signed
    shadow of v on the line of u is |v| cos θ.  Walk to the tip of v along
    the axes instead — up v₂, across v₁ — and the shadows of the two legs
    fill the same shadow: v₂ sin α (right triangle with hypotenuse v₂ and
    angle α at its top) plus v₁ cos α (right triangle with hypotenuse v₁
    and angle α, moved along the rectangle of the two perpendiculars).
    u's own right triangle gives |u| cos α = u₁, |u| sin α = u₂, so
    |u| · |v| cos θ = u₁v₁ + u₂v₂ = u · v.  (Drawn with θ acute and all
    components positive; with signed shadows the same sum holds always.)"""

    def construct(self):
        al, th = 25 * DEGREES, 42 * DEGREES
        nu, nv = 5.4, 4.9
        O = np.array([-5.75, -2.4, 0.0])
        uh = _dir(al)
        U, V = O + nu * uh, O + nv * _dir(al + th)
        v1, v2 = V[0] - O[0], V[1] - O[1]
        u1, u2 = U[0] - O[0], U[1] - O[1]
        Y = np.array([O[0], V[1], 0.0])            # up the y-axis by v₂
        U1 = np.array([U[0], O[1], 0.0])
        G, F = _foot(Y, O, U), _foot(V, O, U)
        K = Y + np.dot(V - Y, uh) * uh             # foot of V on the parallel
        # ---- the claims
        check(abs(np.linalg.norm(F - O) - nv * np.cos(th)) < 1e-9,
              "the shadow of v on u is |v| cos θ")
        check(abs(np.linalg.norm(G - O) - v2 * np.sin(al)) < 1e-9
              and abs(_ang(Y, O, G) - al) < 1e-9, "OG = v₂ sin α (angle α at Y)")
        check(abs(np.linalg.norm(K - Y) - v1 * np.cos(al)) < 1e-9
              and abs(_ang(Y, V, K) - al) < 1e-9, "YK = v₁ cos α (angle α at Y)")
        check(close(K - Y, F - G) and abs(np.dot(K - F, uh)) < 1e-9
              and abs(np.dot(Y - G, uh)) < 1e-9, "YGFK is a rectangle: GF = YK")
        check(abs(_cross(K - V, F - V)) < 1e-9 and np.dot(K - V, F - V) < 0,
              "V lies on KF")
        check(abs(np.linalg.norm(G - O) + np.linalg.norm(F - G)
                  - np.linalg.norm(F - O)) < 1e-9 and 0 < np.dot(G - O, uh)
              < np.dot(F - O, uh) < nu, "OG + GF = OF, all on the segment OU")
        check(abs(u1 - nu * np.cos(al)) < 1e-9 and abs(u2 - nu * np.sin(al)) < 1e-9,
              "u₁ = |u| cos α, u₂ = |u| sin α")
        check(abs(nu * nv * np.cos(th) - (u1 * v1 + u2 * v2)) < 1e-9,
              "|u| |v| cos θ = u₁v₁ + u₂v₂")

        c_u, c_v, c_2, c_1, c_s = BLUE_C, ORANGE, GREEN_C, PURPLE_B, YELLOW_B
        axes = VGroup(Line(O + LEFT * 0.3, U1 + RIGHT * 0.5, color=GREY_D,
                           stroke_width=2),
                      Line(O + DOWN * 0.3, Y + UP * 0.55, color=GREY_D, stroke_width=2))
        au, av = _arrow(O, U, c_u), _arrow(O, V, c_v)
        l_u = _beside(tag("u", 32, c_u), O, U, DOWN + RIGHT, 0.12, at=0.86)
        l_v = _beside(tag("v", 32, c_v), O, V, RIGHT, 0.14, at=0.8)
        a_th = angle_arc(O, U, V, radius=0.95, color=WHITE, width=3)
        l_th = tag("θ", 28).move_to(O + 1.3 * angle_mid_dir(O, U, V))
        self.play(Create(axes), FadeIn(Dot(O, radius=0.06)), run_time=0.8)
        self.play(GrowArrow(au), GrowArrow(av), FadeIn(l_u), FadeIn(l_v), run_time=1.1)
        self.play(Create(a_th), FadeIn(l_th), run_time=0.6)

        # ---- the shadow of v on the line of u
        ell = DashedLine(O - 0.35 * uh, U + 0.45 * uh, color=GREY_B, stroke_width=2,
                         dash_length=0.1)
        VF = DashedLine(V, F, color=GREY_A, stroke_width=2, dash_length=0.08)
        raF = _ra(F, O, V, 0.17)
        sh = Line(O, F, color=c_s, stroke_width=10)
        self.play(Create(ell), Create(VF), Create(raF), run_time=0.9)
        self.play(Create(sh), run_time=0.7)

        # ---- walk to v's tip along the axes: up v₂, then across v₁
        leg2 = Line(O, Y, color=c_2, stroke_width=6)
        leg1 = Line(Y, V, color=c_1, stroke_width=6)
        l_v2 = tag("v₂", 30, c_2).next_to(leg2, LEFT, buff=0.14).shift(DOWN * 0.6)
        l_v1 = tag("v₁", 30, c_1).next_to(leg1, DOWN, buff=0.12).shift(RIGHT * 0.25)
        self.play(Create(leg2), FadeIn(l_v2), run_time=0.8)
        self.play(Create(leg1), FadeIn(l_v1), run_time=0.7)

        # ---- the two legs cast the two parts of the shadow
        YG = DashedLine(Y, G, color=GREY_A, stroke_width=2, dash_length=0.08)
        raG = _ra(G, O, Y, 0.17)
        sh2 = Line(O, G, color=c_2, stroke_width=10)
        sh1 = Line(G, F, color=c_1, stroke_width=10)
        self.play(Create(YG), Create(raG), run_time=0.7)
        self.play(Create(sh2), Create(sh1), run_time=0.8)

        # ---- the angles α: at O, and at the top of each leg's triangle
        a_O = angle_arc(O, U1, U, radius=1.45, color=c_u, width=3)
        l_aO = tag("α", 26, c_u).move_to(O + 1.75 * angle_mid_dir(O, U1, U))
        a_Y1 = angle_arc(Y, O, G, radius=0.75, color=c_2, width=3)
        l_aY1 = tag("α", 26, c_2).move_to(Y + 1.03 * angle_mid_dir(Y, O, G))
        YK = DashedLine(Y, K, color=GREY_A, stroke_width=2, dash_length=0.08)
        VK = DashedLine(V, K, color=GREY_A, stroke_width=2, dash_length=0.08)
        raK = _ra(K, Y, V, 0.15)
        a_Y2 = angle_arc(Y, V, K, radius=0.85, color=c_1, width=3)
        l_aY2 = tag("α", 26, c_1).move_to(Y + 1.13 * angle_mid_dir(Y, V, K))
        self.play(Create(a_O), FadeIn(l_aO), run_time=0.6)
        self.play(Create(a_Y1), FadeIn(l_aY1), run_time=0.6)
        self.play(Create(YK), Create(VK), Create(raK), Create(a_Y2), FadeIn(l_aY2),
                  run_time=0.9)

        # ---- the shadows, carried out flat: v₂ sin α + v₁ cos α = |v| cos θ
        bx, by = 1.0, 2.55
        L2_, L1_ = np.linalg.norm(G - O), np.linalg.norm(F - G)
        b2, b1 = sh2.copy(), sh1.copy()
        self.add(b2, b1)
        self.play(_rigid(b2, -al, O, np.array([bx, by, 0]) - O),
                  _rigid(b1, -al, G, np.array([bx + L2_, by, 0]) - G), run_time=1.5)
        check(close(b2.get_start(), [bx, by, 0], 1e-6)
              and close(b2.get_end(), [bx + L2_, by, 0], 1e-6)
              and close(b1.get_start(), [bx + L2_, by, 0], 1e-6)
              and close(b1.get_end(), [bx + L2_ + L1_, by, 0], 1e-6),
              "the two shadows, laid flat end to end")
        r_2 = tag("v₂ sin α", 26, c_2).next_to(b2, DOWN, buff=0.2)
        r_1 = tag("v₁ cos α", 26, c_1).next_to(b1, DOWN, buff=0.2)
        r_1.align_to(r_2, DOWN)
        tot = VGroup(Line([bx, by + 0.3, 0], [bx + L2_ + L1_, by + 0.3, 0], color=c_s,
                          stroke_width=3),
                     Line([bx, by + 0.18, 0], [bx, by + 0.42, 0], color=c_s, stroke_width=3),
                     Line([bx + L2_ + L1_, by + 0.18, 0], [bx + L2_ + L1_, by + 0.42, 0],
                          color=c_s, stroke_width=3))
        r_s = tag("|v| cos θ", 28, c_s).next_to(tot, UP, buff=0.12)
        self.play(FadeIn(r_2), FadeIn(r_1), Create(tot), FadeIn(r_s), run_time=0.9)
        self.hold(0.4)

        # ---- u's own right triangle: u₁ = |u| cos α, u₂ = |u| sin α
        UU1 = DashedLine(U, U1, color=c_u, stroke_width=2.5, dash_length=0.08)
        base_u = Line(O, U1, color=c_u, stroke_width=5)
        raU = _ra(U1, O, U, 0.17)
        l_u1 = tag("u₁", 30, c_u).next_to(base_u, DOWN, buff=0.12).shift(RIGHT * 0.9)
        l_u2 = tag("u₂", 30, c_u).next_to(UU1, RIGHT, buff=0.14)
        self.play(Create(UU1), Create(base_u), Create(raU), FadeIn(l_u1), FadeIn(l_u2),
                  run_time=0.9)
        row2 = tag("u₂ = |u| sin α      u₁ = |u| cos α", 28, c_u)
        row2.move_to([bx + 2.45, 1.0, 0.0])
        self.play(FadeIn(row2, shift=RIGHT * 0.2), run_time=0.8)
        # × |u|, term by term: the green part gives v₂u₂, the purple v₁u₁
        row3 = _supline([("|u| |v| cos θ", 0, c_s), ("  =  ", 0, WHITE),
                         ("v₂u₂", 0, c_2), ("  +  ", 0, WHITE), ("v₁u₁", 0, c_1)], 30)
        row3.move_to([bx + 2.45, -0.1, 0.0])
        self.play(FadeIn(row3, shift=RIGHT * 0.2), run_time=0.9)

        # ---- label hygiene
        segs = [(O + LEFT * 0.3, U1 + RIGHT * 0.5), (O + DOWN * 0.3, Y + UP * 0.55),
                (O, U), (O, V), (O, Y), (Y, V), (Y, G), (V, F), (Y, K), (V, K),
                (U, U1), (O - 0.35 * uh, U + 0.45 * uh)]
        arcs = (_arc_pts(O, 0.95, al, al + th) + [None]
                + _arc_pts(O, 1.45, 0.0, al) + [None]
                + _arc_pts(Y, 0.75, -PI / 2, -PI / 2 + al) + [None]
                + _arc_pts(Y, 0.85, 0.0, al))
        labels = [l_u, l_v, l_th, l_v2, l_v1, l_aO, l_aY1, l_aY2, l_u1, l_u2]
        for lab in labels:
            for p, q in segs:
                _clear(lab, p, q, 0.04, f"label {lab.text!r}")
            run = []
            for pnt in arcs + [None]:
                if pnt is None:
                    if len(run) > 1:
                        _clear_pts(lab, run, 0.04, f"label {lab.text!r} (an arc)")
                    run = []
                else:
                    run.append(pnt)
        _apart(*labels, gap=0.06)
        _apart(r_2, r_1, r_s, row2, row3, gap=0.08)
        _safe(axes, au, av, *labels, tot, r_s, r_1, r_2, row2, row3, VK, YK)
        check(min(m.get_left()[0] for m in (row2, row3, r_2, tot))
              > max(U1[0] + 0.5, l_u2.get_right()[0]) + 0.3, "panel clear of the figure")
        cap = caption("u · v  =  u₁v₁ + u₂v₂  =  |u| |v| cos θ", 36)
        _below(cap, axes, l_u1, row3)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


class L4_Shoelace(Board):
    """Join the origin O to the ends of every edge PᵢPᵢ₊₁ of the polygon
    (taken counter-clockwise).  Each triangle O Pᵢ Pᵢ₊₁ is half of the
    parallelogram on OPᵢ and OPᵢ₊₁ (a half-turn about the middle of the edge
    completes it), whose area is the cross product xᵢyᵢ₊₁ − xᵢ₊₁yᵢ (L3):
    positive when the edge turns counter-clockwise about O, negative when
    it turns back.  The positive triangles fan out over the polygon and the
    region between it and O; the negative ones cover exactly that region
    again: it cancels, and the polygon is left.  (Drawn with a convex
    pentagon and O outside it; the signed count — every point of the
    polygon covered once more positively than negatively, every point
    outside equally often — holds for any simple polygon and any O.)"""

    def construct(self):
        Pm = [np.array(p) for p in ((2.15, 0.9), (4.6, 0.4), (5.6, 2.8), (3.4, 4.4),
                                    (1.2, 3.0))]
        n = len(Pm)
        crs = [Pm[i][0] * Pm[(i + 1) % n][1] - Pm[(i + 1) % n][0] * Pm[i][1]
               for i in range(n)]
        Am = area(Pm)
        check(Am > 0 and abs(sum(crs) / 2 - Am) < 1e-12, "½ Σ cross = area (CCW)")
        for i in range(n):
            check(abs(area([(0, 0), Pm[i], Pm[(i + 1) % n]]) - crs[i] / 2) < 1e-12,
                  "each signed triangle is half its cross product")
        pos = [i for i in range(n) if crs[i] > 0]
        neg = [i for i in range(n) if crs[i] < 0]
        check(pos == [1, 2, 3] and neg == [0, 4], "three edges turn forward, two back")
        for i in range(n):
            e0, e1, e2 = Pm[i - 1], Pm[i], Pm[(i + 1) % n]
            check(_cross(e1 - e0, e2 - e1) > 0, "convex, counter-clockwise")
            for j in range(n):
                if j != i:
                    check(abs(_cross(Pm[i], Pm[j])) > 0.5,
                          "no fan line runs through another vertex")
        for x in np.linspace(-0.5, 6.0, 131):
            for y in np.linspace(-0.5, 4.8, 107):
                q = (x + 1e-7, y + 1.3e-7)
                w = sum(np.sign(crs[i]) * _pip(q, [(0, 0), Pm[i], Pm[(i + 1) % n]])
                        for i in range(n))
                check(w == (1 if _pip(q, Pm) else 0),
                      "signed cover = 1 inside the polygon, 0 outside")
        shadow = [(0.0, 0.0), Pm[1], Pm[0], Pm[4]]      # covered + and −
        check(abs(abs(area(shadow)) - sum(abs(crs[i]) / 2 for i in neg)) < 1e-12,
              "the negative triangles fill the region between O and the polygon")

        F = Frame(-0.75, 6.15, -0.55, 4.95, max_w=7.3,
                  centre=(-2.95, (SAFE_TOP + SAFE_BOTTOM) / 2))
        Pt = F.P
        O = Pt((0, 0))
        V = [Pt(p) for p in Pm]
        c_p, c_n = BLUE_D, ORANGE

        # ---- the polygon, its edges taken counter-clockwise
        axes = VGroup(Line(Pt((-0.6, 0)), Pt((6.1, 0)), color=GREY_D, stroke_width=2),
                      Line(Pt((0, -0.45)), Pt((0, 4.9)), color=GREY_D, stroke_width=2))
        dotO = Dot(O, radius=0.07)
        l_O = tag("O", 28).next_to(O, DL, buff=0.08)
        poly = Polygon(*V, stroke_color=WHITE, stroke_width=4)
        ctr = np.mean(V, axis=0)

        def chevron(p, q):
            d = _unit(q - p)
            nrm = np.array([-d[1], d[0], 0.0])
            tip = (p + q) / 2 + 0.11 * d
            return VMobject(stroke_color=WHITE, stroke_width=4).set_points_as_corners(
                [tip - 0.2 * d + 0.13 * nrm, tip, tip - 0.2 * d - 0.13 * nrm])

        chev = VGroup(*[chevron(V[i], V[(i + 1) % n]) for i in range(n)])
        # each name in the widest free gap between the edges and fan lines
        # that meet at its vertex (P₁ sits inside the polygon)
        names = []
        for i, deg in enumerate((276, 0, 15, 88, 140)):
            names.append(tag(f"P{'₁₂₃₄₅'[i]}", 28).move_to(V[i] + 0.42 * _dir(deg * DEGREES)))
        self.play(Create(axes), FadeIn(dotO), FadeIn(l_O), run_time=0.8)
        self.play(Create(poly), *[FadeIn(t) for t in names], run_time=1.2)
        self.play(FadeIn(chev), run_time=0.5)

        # ---- one triangle: half the parallelogram on its two sides
        k_ic = 0.18

        def icon(pts, col, at, op=0.75):
            g = mk([Pt(p) for p in pts], col, op, stroke_width=2,
                   stroke_color=WHITE).scale(k_ic, about_point=O)
            return g.shift(at - O)

        i3 = 2                                              # edge P₃P₄
        T3 = [(0.0, 0.0), Pm[i3], Pm[i3 + 1]]
        t3 = mk([Pt(p) for p in T3], c_p, 0.6, stroke_color=WHITE, stroke_width=3)
        self.play(FadeIn(t3), run_time=0.6)
        at1 = np.array([1.2, 1.75, 0.0])
        ic3 = icon(T3, c_p, at1)
        self.play(ReplacementTransform(t3, ic3), run_time=1.0)
        mid = at1 + k_ic * F.k * to3((Pm[i3] + Pm[i3 + 1]) / 2)
        twin = ic3.copy().set_fill(c_p, 0.3)
        self.add(twin)
        self.play(Rotate(twin, angle=PI, about_point=mid), run_time=1.1)
        par = [at1, at1 + k_ic * F.k * to3(Pm[i3]),
               at1 + k_ic * F.k * to3(Pm[i3] + Pm[i3 + 1]), at1 + k_ic * F.k * to3(Pm[i3 + 1])]
        check(_same_poly(twin.get_vertices(), [par[2], par[3], par[1]], 1e-6)
              and abs(abs(area(par)) - 2 * abs(area(ic3.get_vertices()))) < 1e-9,
              "the half-turned twin completes the parallelogram (twice the triangle)")
        row1 = _supline([("=  ½ (x", 0), ("3", -1), ("y", 0), ("4", -1), (" − x", 0),
                         ("4", -1), ("y", 0), ("3", -1), (")", 0)], 28, WHITE)
        row1.next_to(VGroup(ic3, twin), RIGHT, buff=0.25)
        ip = [i for i in neg][0]                            # edge P₁P₂
        T1 = [(0.0, 0.0), Pm[ip], Pm[ip + 1]]
        at2 = np.array([1.2, 0.2, 0.0])
        ic1 = icon(T1, c_n, at2, 0.85)
        row2 = _supline([("=  ½ (x", 0), ("1", -1), ("y", 0), ("2", -1), (" − x", 0),
                         ("2", -1), ("y", 0), ("1", -1), (")  < 0", 0)], 28, WHITE)
        row2.next_to(ic1, RIGHT, buff=0.25)
        self.play(FadeIn(row1), run_time=0.7)
        t1 = mk([Pt(p) for p in T1], c_n, 0.7, stroke_color=WHITE, stroke_width=3)
        self.play(FadeIn(t1), run_time=0.5)
        self.play(ReplacementTransform(t1, ic1), FadeIn(row2), run_time=1.0)
        self.hold(0.4)

        # ---- every edge's triangle: forward ones over the polygon…
        tri = {i: mk([O, V[i], V[(i + 1) % n]], c_p if crs[i] > 0 else c_n,
                     0.45 if crs[i] > 0 else 0.55, stroke_width=2,
                     stroke_color=BLUE_B if crs[i] > 0 else ORANGE) for i in range(n)}
        spots = [(1.3, 0.33), (3.4, 1.07), (3.0, 2.4), (1.85, 3.0), (1.12, 1.3)]
        sg = {i: tag("+" if crs[i] > 0 else "−", 34, WHITE).move_to(Pt(spots[i]))
              for i in range(n)}
        op_of = {i: (0.45 if crs[i] > 0 else 0.55) for i in range(n)}
        # z-order fixed once: forward triangles, backward ones over them, then
        # the outline, its arrows and the names; only opacities animate
        for i in range(n):
            tri[i].set_fill(opacity=0.0).set_stroke(opacity=0.0)
        self.add(*[tri[i] for i in pos], *[tri[i] for i in neg])
        self.bring_to_front(poly, chev, dotO, l_O, *names)

        def show(i):
            return tri[i].animate.set_fill(opacity=op_of[i]).set_stroke(opacity=1.0)

        for i in pos:
            self.play(show(i), FadeIn(sg[i]), run_time=0.75)
        # …and the backward ones over the strip between O and the polygon
        for i in neg:
            self.play(show(i), FadeIn(sg[i]), run_time=0.75)
        self.hold(0.5)

        # ---- the strip is covered once each way: it cancels
        inner = mk(V, c_p, 0.45, stroke_width=0).set_fill(opacity=0.0)
        self.add(inner)
        self.bring_to_back(inner)
        self.play(*[tri[i].animate.set_fill(opacity=0.0).set_stroke(opacity=0.0)
                    for i in range(n)],
                  inner.animate.set_fill(opacity=0.45),
                  *[FadeOut(sg[i]) for i in neg], run_time=1.4)
        self.remove(*tri.values())
        lA = tag("A", 40).move_to(Pt(np.mean(Pm, axis=0)))
        self.play(inner.animate.set_fill(opacity=FILL),
                  *[FadeOut(sg[i]) for i in pos], FadeIn(lA), run_time=0.9)

        labels = [l_O] + names
        _apart(*labels, gap=0.05)
        _apart(*labels, *sg.values(), gap=0.05)
        _apart(*sg.values(), *chev, gap=0.04)
        _apart(*labels, *chev, gap=0.04)
        for i in range(n):
            tri_s = [O, V[i], V[(i + 1) % n]]
            check(all(_pip(sg[i].get_corner(c_), tri_s) for c_ in (UL, UR, DL, DR)),
                  f"sign of triangle {i + 1} inside it")
        for t_ in labels:
            for i in range(n):
                _clear(t_, V[i], V[(i + 1) % n], 0.04, f"label {t_.text}")
                _clear(t_, O, V[i], 0.04, f"label {t_.text} (fan line)")
            _clear(t_, Pt((-0.6, 0)), Pt((6.1, 0)), 0.03, f"label {t_.text}")
            _clear(t_, Pt((0, -0.45)), Pt((0, 4.9)), 0.03, f"label {t_.text}")
        _apart(row1, row2, ic3, ic1, gap=0.1)
        check(min(m.get_left()[0] for m in (ic3, twin, ic1))
              > max(Pt((6.1, 0))[0], max(t.get_right()[0] for t in names)) + 0.3,
              "panel clear of the figure")
        _safe(axes, poly, ic3, twin, ic1, row1, row2, lA, *labels)
        cap = _cap(_supline([("A  =  ½ ∑ (x", 0), ("i", -1), ("y", 0), ("i+1", -1),
                             (" − x", 0), ("i+1", -1), ("y", 0), ("i", -1), (")", 0)],
                            36))
        _below(cap, axes, l_O)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


class L8_PointLineDistance(Board):
    """The line ℓ: ax + by + c = 0, scaled so that c = −1, cuts the axes at
    A (1/a, 0) and B (0, 1/b); P = (x₀, y₀) lies beyond it.  The
    quadrilateral OAPB splits along OP into triangles with bases 1/a, 1/b
    on the axes and heights y₀, x₀; along AB into the right triangle OAB and
    the triangle ABP of height d over AB = √(a² + b²)/ab.  Twice the area,
    both ways: y₀/2a + x₀/2b = 1/2ab + d·AB/2; times 2ab: a x₀ + b y₀ − 1
    = d √(a² + b²).  (Drawn with a, b > 0 and P beyond ℓ from O; with
    |…| the same holds on either side, and every line not through O can be
    scaled to c = −1.)"""

    def construct(self):
        ia, ib = 3.2, 2.4                       # the intercepts 1/a, 1/b
        a, b = 1.0 / ia, 1.0 / ib
        x0, y0 = 2.3, 3.1
        Am, Bm, Pm = np.array([ia, 0.0]), np.array([0.0, ib]), np.array([x0, y0])
        nrm = np.array([a, b]) / np.hypot(a, b)
        dd = (a * x0 + b * y0 - 1.0) / np.hypot(a, b)
        Hm = Pm - dd * nrm
        AB = np.hypot(ia, ib)
        quad = [(0, 0), Am, Pm, Bm]
        OAP, OBP = [(0, 0), Am, Pm], [(0, 0), Pm, Bm]
        OAB, ABP = [(0, 0), Am, Bm], [Am, Pm, Bm]
        # ---- the claims
        check(abs(a * Hm[0] + b * Hm[1] - 1.0) < 1e-12 and dd > 0
              and 0 < np.dot(Hm - Am, Bm - Am) < AB ** 2, "H on ℓ, between A and B")
        check(abs(abs(area(OAP)) - ia * y0 / 2) < 1e-12
              and abs(abs(area(OBP)) - ib * x0 / 2) < 1e-12, "bases on the axes, heights y₀, x₀")
        check(abs(abs(area(OAB)) - ia * ib / 2) < 1e-12
              and abs(abs(area(ABP)) - AB * dd / 2) < 1e-12, "OAB and ABP (height d)")
        check(abs(abs(area(OAP)) + abs(area(OBP)) - abs(area(quad))) < 1e-12
              and abs(abs(area(OAB)) + abs(area(ABP)) - abs(area(quad))) < 1e-12,
              "both pairs fill OAPB")
        for i in range(4):
            check(_cross(np.subtract(quad[(i + 1) % 4], quad[i]),
                         np.subtract(quad[(i + 2) % 4], quad[(i + 1) % 4])) > 0,
                  "OAPB is convex")
        check(abs(AB - np.hypot(a, b) / (a * b)) < 1e-12, "AB = √(a² + b²)/ab")
        check(abs((y0 / (2 * a) + x0 / (2 * b)) - (1 / (2 * a * b) + dd * AB / 2)) < 1e-12
              and abs((a * x0 + b * y0 - 1) - dd * np.hypot(a, b)) < 1e-12,
              "y₀/2a + x₀/2b = 1/2ab + d·AB/2  ⟹  a x₀ + b y₀ − 1 = d √(a² + b²)")

        F = Frame(-0.55, 3.8, -0.65, 3.55, max_w=6.8,
                  centre=(-3.1, (SAFE_TOP + SAFE_BOTTOM) / 2))
        Pt = F.P
        O, A, B, P, H = Pt((0, 0)), Pt(Am), Pt(Bm), Pt(Pm), Pt(Hm)
        X0, Y0 = Pt((x0, 0)), Pt((0, y0))

        axes = VGroup(Line(Pt((-0.4, 0)), Pt((3.7, 0)), color=GREY_B, stroke_width=2),
                      Line(Pt((0, -0.5)), Pt((0, 3.5)), color=GREY_B, stroke_width=2))
        dotO = Dot(O, radius=0.06)
        l_O = tag("O", 28).next_to(O, DL, buff=0.08)
        t0, t1 = -0.11, 1.06
        L0, L1 = Pt(Am + t0 * (Bm - Am)), Pt(Am + t1 * (Bm - Am))
        line = Line(L0, L1, color=WHITE, stroke_width=4)
        l_ell = tag("ℓ", 32).next_to(L0, RIGHT, buff=0.1)
        tickA = Line(A + DOWN * 0.09, A + UP * 0.09, color=WHITE, stroke_width=3)
        tickB = Line(B + LEFT * 0.09, B + RIGHT * 0.09, color=WHITE, stroke_width=3)
        cap_h = Text("M", font_size=28).height
        l_ia = tag("1/a", 28).move_to(A + DOWN * (0.16 + cap_h / 2) + LEFT * 0.22)
        l_ib = tag("1/b", 28).next_to(B, LEFT, buff=0.16).shift(DOWN * 0.3)
        l_A = tag("A", 28).move_to(A + 0.36 * _dir(40 * DEGREES))
        l_B = tag("B", 28).move_to(B + 0.4 * _dir(55 * DEGREES))
        self.play(Create(axes), FadeIn(dotO), FadeIn(l_O), run_time=0.8)
        self.play(Create(line), Create(tickA), Create(tickB), FadeIn(l_ell), FadeIn(l_ia),
                  FadeIn(l_ib), FadeIn(l_A), FadeIn(l_B), run_time=1.2)

        dotP = Dot(P, radius=0.07)
        l_P = tag("P", 30).move_to(P + 0.36 * _dir(60 * DEGREES))
        dx = DashedLine(P, X0, color=GREY_B, stroke_width=2, dash_length=0.08)
        dy = DashedLine(P, Y0, color=GREY_B, stroke_width=2, dash_length=0.08)
        l_x0 = tag("x₀", 28).move_to(X0 + DOWN * (0.16 + cap_h / 2))
        l_y0 = tag("y₀", 28).next_to(Y0, LEFT, buff=0.16)
        self.play(FadeIn(dotP), FadeIn(l_P), Create(dx), Create(dy), FadeIn(l_x0),
                  FadeIn(l_y0), run_time=1.0)
        PH = Line(P, H, color=YELLOW_B, stroke_width=5)
        raH = _ra(H, A, P, 0.17)
        l_d = _beside(tag("d", 32, YELLOW_B), P, H, A - B, 0.14, at=0.45)
        self.play(Create(PH), Create(raH), FadeIn(l_d), run_time=0.9)
        self.hold(0.5)

        # ---- twice the area of OAPB, two ways
        c1, c2, c3, c4 = BLUE_D, TEAL_D, PURPLE_B, ORANGE
        tOAP, tOBP = F.poly(OAP, c1, 0.55, stroke_width=0), F.poly(OBP, c2, 0.55,
                                                                   stroke_width=0)
        tOAB, tABP = F.poly(OAB, c3, 0.5, stroke_width=0), F.poly(ABP, c4, 0.5,
                                                                  stroke_width=0)
        k_ic = 0.2

        def icon(pts, col):
            return mk([Pt(p) for p in pts], col, 0.85, stroke_width=1.5).scale(
                k_ic, about_point=O)

        ics = [icon(OAP, c1), icon(OBP, c2), icon(OAB, c3), icon(ABP, c4)]
        sym = [tag("+", 30), tag("=", 30), tag("+", 30)]
        row = VGroup(ics[0], sym[0], ics[1], sym[1], ics[2], sym[2], ics[3])
        row.arrange(RIGHT, buff=0.22, aligned_edge=DOWN)
        for s_ in sym:
            s_.set_y(row.get_bottom()[1] + 0.35)
        row.move_to([3.45, 2.0, 0.0])
        under = [tag(s_, 26, c_) for s_, c_ in (("y₀/2a", BLUE_B), ("x₀/2b", TEAL_B),
                                                 ("1/2ab", c3), ("d · AB/2", c4))]
        for u_, ic_ in zip(under, ics):
            u_.next_to(ic_, DOWN, buff=0.18)
        base_y = min(u_.get_bottom()[1] for u_ in under)
        for u_ in under:
            u_.shift(UP * (base_y - u_.get_bottom()[1]))
        head = tag("ℓ :  ax + by + c = 0,    c = −1", 28)
        head.move_to([3.45, 3.3, 0.0])

        self.add(tOAP, tOBP)
        self.bring_to_back(tOAP, tOBP)
        self.play(FadeIn(tOAP), FadeIn(tOBP), FadeIn(head), run_time=0.9)
        self.bring_to_back(tOAP, tOBP)
        self.play(FadeIn(VGroup(ics[0], sym[0], ics[1])), FadeIn(under[0]),
                  FadeIn(under[1]), run_time=0.9)
        self.hold(0.4)
        self.add(tOAB, tABP)
        self.bring_to_back(tOAB, tABP)
        self.play(FadeOut(tOAP), FadeOut(tOBP), FadeIn(tOAB), FadeIn(tABP), run_time=1.0)
        self.bring_to_back(tOAB, tABP)
        self.play(FadeIn(VGroup(sym[1], ics[2], sym[2], ics[3])), FadeIn(under[2]),
                  FadeIn(under[3]), run_time=0.9)
        self.hold(0.4)
        r2a = tag("AB  =  √(1/a² + 1/b²)", 28, c4)
        r2b = tag("=  √(a² + b²) / ab", 28, c4)
        r2a.move_to([3.2, 0.5, 0.0])
        r2b.next_to(r2a, DOWN, buff=0.2)
        r2b.shift(RIGHT * (r2a[2].get_x() - r2b[0].get_x()))   # '=' under '='
        r2 = VGroup(r2a, r2b)
        r3 = tag("a x₀ + b y₀ − 1  =  d · √(a² + b²)", 26, YELLOW_B)
        r3.move_to([3.45, -1.15, 0.0])
        x2ab = tag("× 2ab", 24, GREY_A).next_to(r3, UP, buff=0.14).align_to(r3, LEFT)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.5)
        self.play(FadeIn(x2ab), FadeIn(r3, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.4)

        labels = [l_O, l_ell, l_ia, l_ib, l_A, l_B, l_P, l_x0, l_y0, l_d]
        segs = [(Pt((-0.4, 0)), Pt((3.7, 0))), (Pt((0, -0.5)), Pt((0, 3.5))), (L0, L1),
                (P, X0), (P, Y0), (P, H), (O, P), (A, P), (B, P)]
        for lab in labels:
            for p, q in segs:
                _clear(lab, p, q, 0.04, f"label {lab.text!r}")
        _apart(*labels, gap=0.06)
        _apart(head, row, *under, r2, r3, gap=0.08)
        _apart(x2ab, r2, r3, *under, gap=0.06)
        _safe(axes, line, head, row, r2, r3, x2ab, *under, *labels)
        check(min(m.get_left()[0] for m in (head, row, r2, r3, *under))
              > max(Pt((3.7, 0))[0], l_ell.get_right()[0]) + 0.25, "panel clear of the figure")
        cap = caption("d  =  |a x₀ + b y₀ + c| / √(a² + b²)", 36)
        _below(cap, axes, l_x0, l_ia, l_O)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# ========================================================= G. TRIGONOMETRY

class G13_SumToProduct(Board):
    """Points at angles A and B on the unit circle; M the middle of their
    chord.  The trapezoid under the chord, turned half a turn about M,
    stacks on itself into a rectangle of height sin A + sin B with M at its
    centre: M stands ½(sin A + sin B) high.  Folding the isosceles triangle
    of the two radii over OM swaps the ends of the chord, so OM bisects the
    angle A − B (β = ½(A − B) on each side, OM at α = B + β = ½(A + B)) and
    meets the chord at a right angle: OM = cos β, and M stands
    cos β · sin α high.  So sin A + sin B = 2 sin α cos β."""

    def construct(self):
        A_, B_ = 125 * DEGREES, 25 * DEGREES
        al, be = (A_ + B_) / 2, (A_ - B_) / 2
        R = 3.6
        O = np.array([-2.7, -2.45, 0.0])

        def S(p):
            return O + R * to3(p)

        PA, PB = S(_dir(A_)[:2]), S(_dir(B_)[:2])
        M = (PA + PB) / 2
        FA, FB = np.array([PA[0], O[1], 0.0]), np.array([PB[0], O[1], 0.0])
        N = np.array([M[0], O[1], 0.0])
        trap = [FB, PB, PA, FA]
        copy_pts = [2 * M - p for p in trap]
        # ---- the claims
        check(close(copy_pts[1], PA) and close(copy_pts[2], PB) and all(
            abs(p[1] - (O[1] + R * (np.sin(A_) + np.sin(B_)))) < 1e-9
            for p in (copy_pts[0], copy_pts[3])),
            "the half-turned copy shares the chord and tops out at sin A + sin B")
        rect = [FA, FB, copy_pts[3], copy_pts[0]]
        check(abs(abs(area(trap)) * 2 - abs(area(rect))) < 1e-9
              and abs(abs(area(rect)) - (FB[0] - FA[0]) * R * (np.sin(A_) + np.sin(B_)))
              < 1e-9, "trapezoid + its half-turn = the rectangle")
        check(abs((M[1] - O[1]) - R * (np.sin(A_) + np.sin(B_)) / 2) < 1e-9,
              "M stands ½(sin A + sin B) high")
        u = _unit(M - O)
        check(close(O + 2 * np.dot(PA - O, u) * u - (PA - O), PB),
              "the fold over OM carries one end of the chord to the other")
        check(abs(np.dot(M - O, PA - PB)) < 1e-9, "OM ⊥ the chord")
        check(abs(_ang(O, PB, M) - be) < 1e-9 and abs(_ang(O, M, PA) - be) < 1e-9
              and abs(np.arctan2(*(M - O)[1::-1]) - al) < 1e-9, "β each side, OM at α")
        check(abs(np.linalg.norm(M - O) - R * np.cos(be)) < 1e-9
              and abs((M[1] - O[1]) - R * np.cos(be) * np.sin(al)) < 1e-9,
              "OM = cos β, M stands cos β · sin α high")

        # ---- the circle and the two points
        axis = Line(O + LEFT * (R + 0.3), O + RIGHT * (R + 0.3), color=GREY_B,
                    stroke_width=2)
        arc = Arc(radius=R, start_angle=0, angle=PI, arc_center=O, color=GREY_B,
                  stroke_width=3)
        dotO = Dot(O, radius=0.06)
        l_O = tag("O", 28).next_to(O, DOWN, buff=0.12)
        rA, rB = Line(O, PA, color=WHITE, stroke_width=3), Line(O, PB, color=WHITE,
                                                               stroke_width=3)
        dA, dB = Dot(PA, radius=0.07), Dot(PB, radius=0.07)
        aA = angle_arc(O, O + RIGHT, PA, radius=0.6, color=YELLOW_B, width=3)
        l_A = tag("A", 26, YELLOW_B).move_to(O + 0.85 * _dir(A_ / 2))
        aB = angle_arc(O, O + RIGHT, PB, radius=1.0, color=GREEN_B, width=3)
        l_B = tag("B", 26, GREEN_B).move_to(O + 1.45 * _dir(B_ / 2))
        self.play(Create(axis), Create(arc), FadeIn(dotO), FadeIn(l_O), run_time=1.0)
        self.play(Create(rA), Create(rB), FadeIn(dA), FadeIn(dB), Create(aA),
                  FadeIn(l_A), Create(aB), FadeIn(l_B), run_time=1.2)

        # ---- the heights sin A, sin B and the middle M of the chord
        c_a, c_b = BLUE_C, TEAL_C
        vA, vB = Line(FA, PA, color=c_a, stroke_width=6), Line(FB, PB, color=c_b,
                                                               stroke_width=6)
        l_sA = tag("sin A", 26, c_a).next_to(vA, LEFT, buff=0.14)
        l_sB = tag("sin B", 26, c_b).move_to([PB[0] - 0.62, O[1] + 0.5 * (PB[1] - O[1]), 0])
        chord = Line(PA, PB, color=WHITE, stroke_width=3)
        dM = Dot(M, radius=0.08, color=YELLOW_B)
        l_M = tag("M", 28, YELLOW_B).move_to(M + 0.36 * _dir(al))   # out along OM
        self.play(Create(vA), Create(vB), FadeIn(l_sA), FadeIn(l_sB), run_time=0.9)
        self.play(Create(chord), FadeIn(dM), FadeIn(l_M), run_time=0.8)

        # ---- the trapezoid, half-turned about M, stacks into a rectangle
        tz = mk(trap, BLUE_E, 0.45, stroke_width=0)
        self.add(tz)
        self.bring_to_back(tz)
        self.play(FadeIn(tz), run_time=0.5)
        cp = VGroup(mk(trap, BLUE_E, 0.45, stroke_width=0),
                    Line(FA, PA, color=c_a, stroke_width=6),
                    Line(FB, PB, color=c_b, stroke_width=6))
        self.add(cp)
        self.play(Rotate(cp, angle=PI, about_point=M), run_time=1.8)
        check(_same_poly(cp[0].get_vertices(), copy_pts, 1e-6), "the copy landed")
        # the left side of the rectangle: sin A, and on it the turned sin B
        l_sB2 = tag("sin B", 26, c_b).next_to(cp[2], LEFT, buff=0.14)
        outline = Polygon(*rect, stroke_color=WHITE, stroke_width=2.5)
        self.play(FadeIn(l_sB2), Create(outline), run_time=0.8)
        MN = DashedLine(M, N, color=YELLOW_B, stroke_width=3, dash_length=0.09)
        l_N = tag("N", 28).next_to(N, DOWN, buff=0.12)
        px = 3.85
        r1 = tag("MN  =  ½ (sin A + sin B)", 28)
        r1.move_to([px, 2.75, 0.0]).set_x(1.75 + r1.width / 2)
        self.play(Create(MN), FadeIn(l_N), FadeIn(r1, shift=RIGHT * 0.2), run_time=1.0)
        self.hold(0.6)
        self.play(FadeOut(cp), FadeOut(l_sB2), FadeOut(outline), FadeOut(tz),
                  run_time=0.7)

        # ---- fold O M P_A over OM: it lands on O M P_B
        OM = Line(O, M, color=YELLOW_B, stroke_width=4)
        self.play(Create(OM), FadeOut(aA), FadeOut(l_A), run_time=0.6)
        half = mk([O, M, PA], PURPLE_B, 0.5, stroke_color=PURPLE_A, stroke_width=2)
        self.add(half)
        self.play(Rotate(half, angle=PI, axis=u, about_point=O), run_time=1.6)
        check(all(close(v, w, 1e-6) for v, w in zip(half.get_vertices(), [O, M, PB])),
              "the fold landed on O M P_B")
        raM = _ra(M, O, PB, 0.17)
        b1 = Arc(radius=1.0, start_angle=B_, angle=be, arc_center=O, color=PURPLE_A,
                 stroke_width=3)
        b2 = Arc(radius=1.0, start_angle=al, angle=be, arc_center=O, color=PURPLE_A,
                 stroke_width=3)
        l_b1 = tag("β", 26, PURPLE_A).move_to(O + 1.3 * _dir(45 * DEGREES))
        l_b2 = tag("β", 26, PURPLE_A).move_to(O + 1.3 * _dir(al + be / 2))
        self.play(FadeOut(half), Create(raM), Create(b1), Create(b2), FadeIn(l_b1),
                  FadeIn(l_b2), run_time=0.9)

        # ---- OM = cos β at α = B + β; its height is cos β · sin α
        aal = Arc(radius=2.0, start_angle=0, angle=al, arc_center=O, color=YELLOW_B,
                  stroke_width=3)
        l_al = tag("α", 28, YELLOW_B).move_to(O + 2.25 * _dir(45 * DEGREES))
        l_1 = _beside(tag("1", 28), O, PA, LEFT + DOWN, 0.12, at=0.62)
        l_cb = _beside(tag("cos β", 26, YELLOW_B), O, M, LEFT, 0.12, at=0.84)
        raN = _ra(N, O, M, 0.17)
        self.play(Create(aal), FadeIn(l_al), FadeIn(l_1), FadeIn(l_cb), Create(raN),
                  run_time=1.0)
        r2a = tag("MN  =  OM · sin α", 28)
        r2b = tag("=  cos β · sin α", 28)
        r2a.next_to(r1, DOWN, buff=0.45).align_to(r1, LEFT)
        r2b.next_to(r2a, DOWN, buff=0.2)
        r2b.shift(RIGHT * (r2a[2].get_x() - r2b[0].get_x()))   # '=' under '='
        r2 = VGroup(r2a, r2b)
        r3 = VGroup(tag("α  =  ½(A + B)", 28, YELLOW_B),
                    tag("β  =  ½(A − B)", 28, PURPLE_A)).arrange(DOWN, buff=0.2,
                                                                aligned_edge=LEFT)
        r3.next_to(r2, DOWN, buff=0.5).align_to(r1, LEFT)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(r3, shift=RIGHT * 0.2), run_time=0.8)

        labels = [l_O, l_B, l_sA, l_sB, l_M, l_N, l_b1, l_b2, l_al, l_1, l_cb]
        segs = [(O + LEFT * (R + 0.3), O + RIGHT * (R + 0.3)), (O, PA), (O, PB),
                (FA, PA), (FB, PB), (PA, PB), (O, M), (M, N)]
        arcs = [_arc_pts(O, R, 0.0, PI, 90), _arc_pts(O, 1.0, 0.0, A_, 40),
                _arc_pts(O, 2.0, 0.0, al, 30)]
        for lab in labels:
            for p, q in segs:
                _clear(lab, p, q, 0.04, f"label {lab.text!r}")
            for pts in arcs:
                _clear_pts(lab, pts, 0.04, f"label {lab.text!r} (an arc)")
        _apart(*labels, gap=0.05)
        _apart(l_sB2, l_M, l_sA, gap=0.05)
        for lab in (l_sB2,):
            for p, q in segs + [(copy_pts[0], copy_pts[1]), (copy_pts[2], copy_pts[3])]:
                _clear(lab, p, q, 0.04, f"label {lab.text!r}")
            _clear_pts(lab, arcs[0], 0.04, "a rectangle label (circle)")
        _apart(r1, r2, r3, gap=0.1)
        _safe(axis, arc, outline, r1, r2, r3, l_sB2, *labels)
        check(min(r.get_left()[0] for r in (r1, r2, r3))
              > max(O[0] + R + 0.3, outline.get_right()[0]) + 0.2,
              "rows clear of the figure")
        cap = caption("sin A + sin B  =  2 sin((A+B)/2) · cos((A−B)/2)", 34)
        _below(cap, axis, l_O, l_N)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)
