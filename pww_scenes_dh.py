# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_dh.py — means & inequalities (D3–D6) and calculus (H1–H6).
#
# Each scene animates the construction of the visual argument; the key
# invariants are asserted with check(...) so a wrong construction fails the
# render instead of rendering quietly into a wrong picture.


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


def _ctag(s, size=26, color=WHITE, t2c=None):
    """A label in one Text (one font run, one baseline) whose substrings are
    recoloured glyph by glyph afterwards.  (Text's own t2c splits the text
    into runs, and after some glyphs have needed a fallback font the
    uncoloured runs can come out in a different face.)"""
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


def _on_base(s, x, base_y, size=28, color=WHITE):
    """A label centred on x whose baseline sits at base_y, so a row of
    labels shares one baseline whatever their ascenders and descenders.
    (The baseline is read off a reference capital that is then dropped.)"""
    t = Text("M" + s, font_size=size, color=color)
    m = t.submobjects[0]
    base = m.get_bottom()[1]
    t.remove(m)
    t.shift(np.array([x - t.get_center()[0], base_y - base, 0.0]))
    return t


def _under(s, p, size=28, color=WHITE, gap=0.14):
    """Label under the point p (a tick on a horizontal line), on the baseline
    shared by every _under label of that size and gap."""
    cap = Text("M", font_size=size).height
    return _on_base(s, p[0], p[1] - gap - cap, size, color)


def _over(s, p, size=28, color=WHITE, gap=0.12):
    """Label over the point p, descenders kept clear of the line."""
    probe = Text("Mg", font_size=size)
    depth = probe.submobjects[0].get_bottom()[1] - probe.submobjects[1].get_bottom()[1]
    return _on_base(s, p[0], p[1] + gap + depth, size, color)


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


def _lim(size=34, color=YELLOW_B):
    """'lim' with a small 'n → ∞' under it."""
    l = Text("lim", font_size=size, color=color)
    n = Text("n → ∞", font_size=size * 0.55, color=color).next_to(l, DOWN, buff=0.05)
    return VGroup(l, n)


def _row(*parts, buff=0.14):
    """Parts set side by side on one line.  A part built by _lim is lifted
    back so that its 'lim', not the group with its subscript, sits on the
    line."""
    row = VGroup(*parts).arrange(RIGHT, buff=buff)
    ref = next((p for p in parts if isinstance(p, Text)), None)
    if ref is not None:
        for p in parts:
            if isinstance(p, VGroup) and len(p) == 2 and isinstance(p[0], Text) \
                    and p[0].text == "lim":
                p.shift(UP * (ref.get_center()[1] - p[0].get_center()[1]))
    return row


def _cap(group):
    """Place a composite formula where caption() puts its Text."""
    group.to_edge(DOWN, buff=0.3)
    if group.width > 13.4:
        group.scale_to_fit_width(13.4)
    return group


def _rot(v, ang):
    """Rotate a screen vector counter-clockwise by ang (radians)."""
    v = to3(v)
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])


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


# ===================================================== D. MEANS & INEQUALITIES

class D3_CauchySchwarz(Board):
    """u = (a, b), v = (x, y) with a, b, x, y > 0.

    In the (a+y) x (b+x) rectangle, four corner right triangles (legs a, b
    twice and x, y twice) leave a parallelogram P with sides |u| and |v|.
    Translating the same four triangles into an a x b block and an x x y
    block leaves an a x x block and a b x y block, so area(P) = ax + by.
    Turned onto its |u| side and sheared, P becomes a |u| by h rectangle;
    h is the projection of v on u (a leg), the slanted side is |v| (the
    hypotenuse), so h <= |v| and ax + by <= |u||v|.  Signs: for any
    vectors |ax + by| <= |a||x| + |b||y|, which is this case.
    """

    def construct(self):
        a, b, x, y = 2.2, 0.9, 1.2, 1.9
        W, H = a + y, b + x
        ul, vl = float(np.hypot(a, b)), float(np.hypot(x, y))
        h = (a * x + b * y) / ul                 # height of P over its |u| side
        k = 1.25
        o_top = np.array([-6.2, 0.77, 0.0])      # math (0,0) of each panel
        o_bot = np.array([-6.2, -2.53, 0.0])
        drop = o_bot - o_top

        def T(p):
            return o_top + k * to3(p)

        V1, V2, V3, V4 = (a, 0), (a + y, x), (y, H), (0, b)
        tri_pts = [[(0, 0), (a, 0), (0, b)],              # ab, bottom-left
                   [(a, 0), (W, 0), (W, x)],              # xy, bottom-right
                   [(W, x), (W, H), (y, H)],              # ab, top-right
                   [(y, H), (0, H), (0, b)]]              # xy, top-left
        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D]
        moves = [(0, 0), (0, b), (-y, -x), (a, 0)]       # into the two blocks

        # ---- exact-geometry checks
        check(abs(area([V1, V2, V3, V4]) - (a * x + b * y)) < 1e-12,
              "area(P) = ax + by")
        check(abs(sum(area(t) for t in tri_pts) + area([V1, V2, V3, V4])
                  - W * H) < 1e-12, "four triangles + P tile the rectangle")
        moved = [[(p[0] + d[0], p[1] + d[1]) for p in t]
                 for t, d in zip(tri_pts, moves)]
        check(abs(area(moved[0]) + area(moved[2]) - a * b) < 1e-12
              and all(0 - 1e-12 <= p[0] <= a + 1e-12 and 0 - 1e-12 <= p[1] <= b + 1e-12
                      for p in moved[0] + moved[2]), "ab triangles fill [0,a]x[0,b]")
        check(abs(area(moved[1]) + area(moved[3]) - x * y) < 1e-12
              and all(a - 1e-12 <= p[0] <= W + 1e-12 and b - 1e-12 <= p[1] <= H + 1e-12
                      for p in moved[1] + moved[3]), "xy triangles fill [a,W]x[b,H]")
        check(h <= vl, "projection h is no longer than |v|")

        # ---- top panel: the rectangle, its four corners and P
        frame = Polygon(T((0, 0)), T((W, 0)), T((W, H)), T((0, H)),
                        stroke_color=WHITE, stroke_width=4)
        tris = [mk([T(p) for p in t], c) for t, c in zip(tri_pts, cols)]
        P = mk([T(V1), T(V2), T(V3), T(V4)], YELLOW_E, 0.85)

        def side_lab(s, at, d):
            if d is DOWN:
                return _under(s, T(at), 24, gap=0.12)
            if d is UP:
                return _over(s, T(at), 24, gap=0.10)
            return tag(s, 24).next_to(T(at), d, buff=0.12)

        labs_top = VGroup(
            side_lab("a", (a / 2, 0), DOWN), side_lab("y", (a + y / 2, 0), DOWN),
            side_lab("y", (y / 2, H), UP), side_lab("a", (y + a / 2, H), UP),
            side_lab("b", (0, b / 2), LEFT), side_lab("x", (0, b + x / 2), LEFT),
            side_lab("x", (W, x / 2), RIGHT), side_lab("b", (W, x + b / 2), RIGHT))
        cP = T((W / 2, H / 2))

        def inner(p, q, s, col, d=0.30):
            m = (T(p) + T(q)) / 2
            n = (cP - m) / np.linalg.norm(cP - m)
            return tag(s, 24, col).move_to(m + d * n)

        lab_u = inner(V4, V1, "|u|", BLACK)
        lab_v = inner(V1, V2, "|v|", BLACK)
        legend = VGroup(tag("u = (a, b)", 26, BLUE_B),
                        tag("v = (x, y)", 26, TEAL_B)).arrange(DOWN, buff=0.16,
                                                               aligned_edge=LEFT)
        legend.move_to(np.array([3.6, 2.95, 0.0]))

        self.play(Create(frame), FadeIn(labs_top), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(t) for t in tris], lag_ratio=0.2),
                  run_time=1.2)
        self.play(FadeIn(P), FadeIn(legend), run_time=0.8)
        self.play(FadeIn(lab_u), FadeIn(lab_v), run_time=0.6)
        self.hold(0.6)

        # ---- bottom panel: the same frame and the same four triangles ...
        frame2 = frame.copy()
        tris2 = [t.copy() for t in tris]
        labs_bot = VGroup(*[l.copy() for l in labs_top[0:2]],
                          *[l.copy() for l in labs_top[4:6]])
        self.play(frame2.animate.shift(drop), labs_bot.animate.shift(drop),
                  *[t.animate.shift(drop) for t in tris2], run_time=1.3)
        # ... translated into an a x b block and an x x y block
        self.play(*[t.animate.shift(k * to3(d)) for t, d in zip(tris2, moves)],
                  run_time=2.0)

        def Bm(p):
            return T(p) + drop

        blk_ax = mk([Bm((0, b)), Bm((a, b)), Bm((a, H)), Bm((0, H))], YELLOW_E, 0.85)
        blk_by = mk([Bm((a, 0)), Bm((W, 0)), Bm((W, b)), Bm((a, b))], YELLOW_E, 0.85)
        t_ax = tag("ax", 26, BLACK).move_to(blk_ax)
        t_by = tag("by", 26, BLACK).move_to(blk_by)
        self.play(FadeIn(blk_ax), FadeIn(blk_by), run_time=0.8)
        self.play(FadeIn(t_ax), FadeIn(t_by), run_time=0.5)
        self.hold(0.6)

        # ---- right panel: P turned onto its |u| side (a rigid motion)
        ang = float(np.arctan2(b, a))
        R0 = np.array([1.0, -1.75, 0.0])          # where V4 lands
        shift = R0 - (cP + _rot(T(V4) - cP, ang))
        Pm = P.copy()
        self.add(Pm)
        self.play(_rigid(Pm, ang, about=cP, shift=shift), run_time=2.2)
        W1, W2, W3, W4 = [cP + _rot(T(p) - cP, ang) + shift for p in (V1, V2, V3, V4)]
        check(close(W4, R0) and close(W1, R0 + RIGHT * k * ul),
              "P's |u| side lands horizontal")
        check(abs(W2[1] - R0[1] - k * h) < 1e-9 and abs(W3[1] - R0[1] - k * h) < 1e-9,
              "P's opposite side is at height h")
        lab_sum = tag("ax + by", 26, BLACK)

        # shear: base W4-W1 fixed (bold), top slides along its own line
        base = Line(W4, W1, color=YELLOW_B, stroke_width=8)
        g1 = guide(np.array([W4[0] - 0.5, R0[1], 0.0]),
                   np.array([W2[0] + 0.5, R0[1], 0.0]), YELLOW_B)
        g2 = guide(np.array([W4[0] - 0.5, W2[1], 0.0]),
                   np.array([W2[0] + 0.5, W2[1], 0.0]), YELLOW_B)
        slant = Line(W1, W2, color=TEAL_B, stroke_width=5)
        self.play(Create(base), Create(g1), Create(g2), run_time=0.6)
        self.add(slant)
        top = k * h
        rect_pts = [W1, W1 + UP * top, W4 + UP * top, W4]   # same vertex order
        self.play(Transform(Pm, mk(rect_pts, YELLOW_E, 0.85)), run_time=1.8)
        lab_sum.move_to((W4 + W1) / 2 + UP * top / 2)
        self.play(FadeOut(g1), FadeOut(g2), FadeIn(lab_sum), run_time=0.5)

        # the slanted side |v| swings upright: it overtops the height h
        phi = float(np.arctan2(*(W2 - W1)[1::-1]))
        ghost = DashedLine(W1, W2, color=TEAL_B, stroke_width=2, dash_length=0.08)
        swing = DashedVMobject(Arc(radius=k * vl, start_angle=phi,
                                   angle=PI / 2 - phi, arc_center=W1,
                                   color=TEAL_B, stroke_width=2), num_dashes=14)
        self.add(ghost)
        self.play(Rotate(slant, angle=PI / 2 - phi, about_point=W1),
                  Create(swing), run_time=1.6)
        check(close(slant.get_end(), W1 + UP * k * vl, 1e-6)
              or close(slant.get_start(), W1 + UP * k * vl, 1e-6),
              "swung side stands at height |v|")
        outline = DashedLine(W1 + UP * k * vl, W4 + UP * k * vl, color=TEAL_B,
                             stroke_width=4, dash_length=0.1)
        side_l = DashedLine(W4, W4 + UP * k * vl, color=TEAL_B, stroke_width=4,
                            dash_length=0.1)
        lab_ub = tag("|u|", 26, BLUE_B).next_to(base, DOWN, buff=0.14)
        lab_vr = tag("|v|", 26, TEAL_B).next_to(slant, RIGHT, buff=0.16)
        strip = mk([W4 + UP * top, W1 + UP * top, W1 + UP * k * vl,
                    W4 + UP * k * vl], TEAL_D, 0.28, stroke_width=0)
        self.play(Create(outline), Create(side_l), FadeIn(strip),
                  FadeIn(lab_ub), FadeIn(lab_vr), run_time=0.9)

        content = VGroup(frame, *tris, P, labs_top, lab_u, lab_v, legend, frame2,
                         *tris2, labs_bot, blk_ax, blk_by, Pm, base, slant,
                         outline, side_l, lab_ub, lab_vr, lab_sum, ghost, swing,
                         strip)
        _safe(*content)
        _apart(*labs_top, lab_u, lab_v, legend)
        self.play(Write(caption("u·v  =  ax + by   ≤   |u| |v|", 34)), run_time=1.2)
        self.hold(2.4)


class D4_TriangleInequality(Board):
    """Lay u + v along a line and drop the perpendicular from the bend.

    Each of u, v is the hypotenuse of a right triangle whose leg lies on
    the straight path, so swung down onto the line about its far end it
    overshoots the foot of the perpendicular.  The two swung segments
    therefore overlap and together cover the straight path:
    |u| + |v| = |u + v| + overlap.  (If the foot falls outside the segment,
    one side alone is already longer than u + v.)
    """

    def construct(self):
        Om, Am, Bm = np.array([0.0, 0.0]), np.array([3.0, 2.8]), np.array([9.0, 0.0])
        U = float(np.linalg.norm(Am - Om))
        V = float(np.linalg.norm(Bm - Am))
        S = float(np.linalg.norm(Bm - Om))
        k = 1.19
        org = np.array([-6.35, 0.3, 0.0])

        def P(p):
            return org + k * to3(p)

        O, A, B = P(Om), P(Am), P(Bm)
        F = P((Am[0], 0.0))                        # foot of the perpendicular
        A1 = P((U, 0.0))                           # u swung about O
        A2 = P((Bm[0] - V, 0.0))                   # v swung about B
        check(A2[0] <= F[0] <= A1[0], "swung segments overshoot the foot")
        check(abs((A1[0] - A2[0]) / k - (U + V - S)) < 1e-9,
              "overlap = |u| + |v| - |u + v|")

        def arrow(p, q, col, w=6):
            return Arrow(p, q, buff=0, color=col, stroke_width=w,
                         max_tip_length_to_length_ratio=0.07,
                         max_stroke_width_to_length_ratio=20)

        au, av = arrow(O, A, BLUE_D), arrow(A, B, TEAL_D)
        aw = arrow(O, B, YELLOW_B, 5)

        def off(p, q, s, col, d=0.32, size=30):
            m = (p + q) / 2
            t = q - p
            n = np.array([-t[1], t[0], 0.0]) / np.linalg.norm(t)
            return tag(s, size, col).move_to(m + d * n)

        lu = off(O, A, "u", BLUE_B)
        lv = off(A, B, "v", TEAL_B)
        lw = tag("u + v", 28, YELLOW_B).move_to(P((6.7, 0.5)))

        self.play(GrowArrow(au), run_time=0.9)
        self.play(GrowArrow(av), FadeIn(lu), run_time=0.9)
        self.play(FadeIn(lv), run_time=0.3)
        self.play(GrowArrow(aw), FadeIn(lw), run_time=1.0)
        self.hold(0.5)

        # the perpendicular from the bend
        perp = DashedLine(A, F, color=GREY_B, stroke_width=3, dash_length=0.1)
        s = 0.22
        ra = VMobject(stroke_color=GREY_B, stroke_width=3).set_points_as_corners(
            [F + UP * s, F + UP * s + RIGHT * s, F + RIGHT * s])
        self.play(Create(perp), Create(ra), FadeIn(Dot(F, radius=0.06,
                                                         color=GREY_B)),
                  run_time=0.9)

        # swing u down about O, and v down about B
        su = Line(O, A, color=BLUE_D, stroke_width=8)
        angA = float(np.arctan2(A[1] - O[1], A[0] - O[0]))
        arc_u = DashedVMobject(Arc(radius=k * U, start_angle=angA, angle=-angA,
                                   arc_center=O, color=BLUE_B, stroke_width=2),
                               num_dashes=16)
        self.add(su)
        self.play(Rotate(su, angle=-angA, about_point=O), Create(arc_u),
                  run_time=1.8)
        check(close(su.get_end(), A1, 1e-6), "u lands on the line at |u|")
        sv = Line(B, A, color=TEAL_D, stroke_width=8)
        angB = float(np.arctan2(A[1] - B[1], A[0] - B[0]))
        arc_v = DashedVMobject(Arc(radius=k * V, start_angle=angB, angle=PI - angB,
                                   arc_center=B, color=TEAL_B, stroke_width=2),
                               num_dashes=22)
        self.add(sv)
        self.play(Rotate(sv, angle=PI - angB, about_point=B), Create(arc_v),
                  run_time=1.8)
        check(close(sv.get_end(), A2, 1e-6), "v lands on the line at |v| from B")
        # lift them apart a hair so the overlap shows
        self.play(su.animate.shift(UP * 0.13), sv.animate.shift(DOWN * 0.13),
                  run_time=0.4)
        self.hold(0.5)

        # straighten the bent path into one bar, under the straight one
        y1, y2 = -1.1, -2.2
        bw = Line(O, B, color=YELLOW_B, stroke_width=10)
        bu = su.copy()
        bv = sv.copy()
        self.play(bw.animate.shift(UP * (y1 - O[1])), run_time=0.9)
        self.play(bu.animate.shift(UP * (y2 - bu.get_center()[1])), run_time=0.9)
        self.play(bv.animate.shift(UP * (y2 - bv.get_center()[1])
                                   + RIGHT * (A1[0] - A2[0])), run_time=1.2)
        check(abs(bv.get_start()[0] - A1[0]) < 1e-6
              or abs(bv.get_end()[0] - A1[0]) < 1e-6, "v's bar abuts u's bar")
        end_uv = max(bv.get_start()[0], bv.get_end()[0])
        check(abs(end_uv - (O[0] + k * (U + V))) < 1e-6, "bar = |u| + |v|")
        mark = DashedLine(B + DOWN * 0.15, np.array([B[0], y2 - 0.3, 0.0]),
                          color=GREY_B, stroke_width=2, dash_length=0.08)
        lw2 = tag("|u + v|", 26, YELLOW_B).next_to(bw, RIGHT, buff=0.25)
        lu2 = tag("|u|", 26, BLUE_B).next_to(bu, DOWN, buff=0.16)
        lv2 = tag("|v|", 26, TEAL_B).next_to(bv, DOWN, buff=0.16)
        self.play(Create(mark), FadeIn(lw2), FadeIn(lu2), FadeIn(lv2), run_time=0.8)

        content = VGroup(au, av, aw, lu, lv, lw, perp, ra, su, sv, arc_u, arc_v,
                         bw, bu, bv, mark, lw2, lu2, lv2)
        _safe(*content)
        _apart(lu, lv, lw, lw2, lu2, lv2)
        self.play(Write(caption("|u + v|  ≤  |u| + |v|", 36)), run_time=1.2)
        self.hold(2.4)


class D5_ReciprocalSum(Board):
    """Four x by 1/x rectangles (area 1 each), each a quarter-turn of the
    last about the centre, fit inside the square of side x + 1/x with a
    square hole of side |x - 1/x| left over.  So (x + 1/x)^2 >= 4 and
    x + 1/x >= 2, with equality exactly when the hole closes (x = 1).
    x + 1/x is also half the perimeter of the x by 1/x rectangle."""

    def construct(self):
        x0, k = 1.75, 2.2
        C = np.array([-2.55, 0.33, 0.0])          # centre of the big square
        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D]
        xt = ValueTracker(x0)

        def geom(xv):
            p, q = xv, 1.0 / xv
            s = p + q
            o = C - k * s / 2 * (RIGHT + UP)

            def S(pt):
                return o + k * to3(pt)

            R = [[(0, 0), (p, 0), (p, q), (0, q)],
                 [(p, 0), (s, 0), (s, p), (p, p)],
                 [(q, p), (s, p), (s, s), (q, s)],
                 [(0, q), (q, q), (q, s), (0, s)]]
            lo, hi = min(p, q), max(p, q)
            hole = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
            return p, q, s, S, R, hole

        def centre(r):
            return (sum(t[0] for t in r) / 4, sum(t[1] for t in r) / 4)

        def edge_labels(p, q, S):
            return VGroup(
                _under("x", S((p / 2, 0)), 28),
                _under("1/x", S((p + q / 2, 0)), 28),
                tag("1/x", 28).next_to(S((0, q / 2)), LEFT, buff=0.14),
                tag("x", 28).next_to(S((0, q + p / 2)), LEFT, buff=0.14))

        p, q, s, S, R, hole = geom(x0)
        for r in R:
            check(abs(area(r) - 1.0) < 1e-12, "each rectangle has area 1")
        check(abs(s * s - (4 + area(hole))) < 1e-12, "(x+1/x)² = 4 + hole")

        # one rectangle of area 1
        r1 = mk([S(t) for t in R[0]], cols[0])
        e = edge_labels(p, q, S)
        ones = [tag("1", 32).move_to(S(centre(r))) for r in R]
        self.play(FadeIn(r1), FadeIn(e[0]), FadeIn(e[2]), run_time=1.0)
        self.play(FadeIn(ones[0]), run_time=0.5)
        self.hold(0.4)

        # three quarter-turns about the centre of the square
        tiles = [r1]
        for i in range(1, 4):
            t = tiles[-1].copy()
            self.add(t)
            self.bring_to_front(*ones[:i])         # labels stay on top
            self.play(Rotate(t, angle=PI / 2, about_point=C), run_time=1.0)
            got = sorted(map(tuple, np.round(t.get_vertices()[:, :2], 9)))
            want = sorted(map(tuple, np.round([S(v)[:2] for v in R[i]], 9)))
            check(np.allclose(got, want, atol=1e-6),
                  f"quarter-turn {i} lands on rectangle {i + 1}")
            self.play(t.animate.set_fill(cols[i]), FadeIn(ones[i]), run_time=0.4)
            tiles.append(t)

        outline = Square(side_length=k * s, stroke_color=YELLOW_B,
                         stroke_width=5).move_to(C)
        self.play(Create(outline), FadeIn(e[1]), FadeIn(e[3]), run_time=1.0)
        hole_m = mk([S(t) for t in hole], RED_E, 0.45, stroke_color=RED_B,
                    stroke_width=3)
        hole_l = tag("(x − 1/x)²", 26, RED_B).move_to(S(centre(hole)))
        self.play(FadeIn(hole_m), FadeIn(hole_l), run_time=0.8)

        # the count, read off the figure
        eq1 = tag("(x + 1/x)²", 32, YELLOW_B)
        eq2 = _ctag("=  4 · 1  +  (x − 1/x)²", 32, WHITE,
                    {"4 · 1": BLUE_B, "(x − 1/x)²": RED_B})
        eq3 = _ctag("≥  4", 32, WHITE, {"4": BLUE_B})
        eqs = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        eqs.move_to(np.array([3.6, 0.6, 0.0]))
        self.play(FadeIn(eq1), run_time=0.6)
        self.play(FadeIn(eq2), run_time=0.8)
        self.play(FadeIn(eq3), run_time=0.6)
        self.hold(0.6)

        # the hole closes exactly when x = 1/x
        def build():
            pv, qv, sv, Sv, Rv, hv = geom(xt.get_value())
            g = VGroup(*[mk([Sv(t) for t in r], c) for r, c in zip(Rv, cols)])
            g.add(*[tag("1", 32).move_to(Sv(centre(r))) for r in Rv])
            g.add(mk([Sv(t) for t in hv], RED_E, 0.45, stroke_color=RED_B,
                     stroke_width=3))
            g.add(Square(side_length=k * sv, stroke_color=YELLOW_B,
                         stroke_width=5).move_to(C))
            g.add(edge_labels(pv, qv, Sv))
            return g

        live = always_redraw(build)
        self.play(FadeOut(hole_l), run_time=0.4)
        self.remove(*tiles, *ones, outline, hole_m, *e)
        self.add(live)
        self.play(xt.animate.set_value(1.0), run_time=2.4)
        x1 = tag("x = 1", 28, YELLOW_B).next_to(live, UP, buff=0.18)
        self.play(FadeIn(x1), run_time=0.4)
        self.hold(0.8)
        self.play(FadeOut(x1), run_time=0.3)
        self.play(xt.animate.set_value(x0), run_time=2.0)
        live.clear_updaters()
        self.play(FadeIn(hole_l), run_time=0.5)

        _safe(live, hole_l, eqs)
        _apart(*live[-1], hole_l, *eqs)
        self.play(Write(caption("x + 1/x  ≥  2,   x > 0", 36)), run_time=1.2)
        self.hold(2.4)


class D6_Rearrangement(Board):
    """a >= b, x >= y.  The frames a x x and b x y hold ax + by; the pieces
    a x y and b x x hold ay + bx.  The a x y piece fills the bottom of the
    first frame, the b x x piece cut at height y fills the second frame and
    the top-left of the first, and an (a-b) x (x-y) corner stays empty:
    ax + by = ay + bx + (a-b)(x-y).  The corner is a genuine rectangle
    exactly when the two pairs are ordered alike (both reversed: rename)."""

    def construct(self):
        a, b, x, y = 3.0, 1.5, 2.4, 1.0
        k = 1.7
        o1 = np.array([-6.1, -2.5, 0.0])          # frame a x x
        o2 = np.array([-0.45, -2.5, 0.0])         # frame b x y
        o_ay = np.array([-6.1, 1.95, 0.0])        # where the pieces start
        o_bx = np.array([3.75, -2.5, 0.0])
        check(a >= b and x >= y, "similarly ordered")
        check(abs((a * x + b * y) - (a * y + b * x) - (a - b) * (x - y)) < 1e-12,
              "ax + by = ay + bx + (a-b)(x-y)")

        def box(o, w, h):
            return [o, o + RIGHT * k * w, o + RIGHT * k * w + UP * k * h,
                    o + UP * k * h]

        def frame(o, w, h):
            return Polygon(*box(o, w, h), stroke_color=WHITE, stroke_width=4)

        F1, F2 = frame(o1, a, x), frame(o2, b, y)
        l_a = _under("a", o1 + RIGHT * k * a / 2, 28)
        l_x = tag("x", 28).next_to(o1 + UP * k * x / 2, LEFT, buff=0.14)
        l_b = _under("b", o2 + RIGHT * k * b / 2, 28)
        l_y = tag("y", 28).next_to(o2 + UP * k * y / 2, LEFT, buff=0.14)
        in1 = tag("ax", 32, GREY_B).move_to(F1)
        in2 = tag("by", 32, GREY_B).move_to(F2)
        self.play(Create(F1), Create(F2), run_time=1.0)
        self.play(FadeIn(VGroup(l_a, l_x, l_b, l_y, in1, in2)), run_time=0.7)
        self.hold(0.5)

        ay = mk(box(o_ay, a, y), ORANGE)
        bx = mk(box(o_bx, b, x), TEAL_D)
        t_ay = tag("ay", 30, BLACK).move_to(ay)
        t_bx = tag("bx", 30, BLACK).move_to(bx)
        self.play(FadeIn(ay), FadeIn(t_ay), FadeIn(bx), FadeIn(t_bx), run_time=0.9)
        self.hold(0.5)

        # a x y slides into the bottom of the a x x frame
        self.play(FadeOut(in1), FadeOut(in2), run_time=0.4)
        self.play(VGroup(ay, t_ay).animate.shift(o1 - o_ay), run_time=1.4)
        check(close(ay.get_vertices()[0], o1), "ay sits in the corner of frame 1")

        # b x x is cut at height y
        cut = DashedLine(o_bx + UP * k * y + LEFT * 0.25,
                         o_bx + UP * k * y + RIGHT * (k * b + 0.25),
                         color=WHITE, stroke_width=3, dash_length=0.1)
        lo = mk(box(o_bx, b, y), TEAL_D)
        hi = mk(box(o_bx + UP * k * y, b, x - y), TEAL_D)
        t_lo = tag("by", 28, BLACK).move_to(lo)
        t_hi = tag("b(x−y)", 26, BLACK).move_to(hi)
        self.play(Create(cut), run_time=0.6)
        self.remove(bx)
        self.add(lo, hi)
        self.play(FadeOut(cut), FadeOut(t_bx), FadeIn(t_lo), FadeIn(t_hi),
                  run_time=0.6)
        # ... its top part fills the top-left of frame 1, its foot fills frame 2
        self.play(VGroup(hi, t_hi).animate.shift(o1 - o_bx), run_time=1.3)
        self.play(VGroup(lo, t_lo).animate.shift(o2 - o_bx), run_time=1.3)
        check(close(hi.get_vertices()[0], o1 + UP * k * y), "b(x-y) on top of ay")
        check(close(lo.get_vertices()[0], o2), "by fills frame 2")
        check(abs(area([v[:2] for v in lo.get_vertices()]) - k * k * b * y) < 1e-9,
              "the foot is exactly b x y")

        # what is left empty is the (a-b) x (x-y) corner
        gap = mk(box(o1 + RIGHT * k * b + UP * k * y, a - b, x - y), RED_D, 0.85)
        g_w = tag("a − b", 26, RED_B).next_to(o1 + RIGHT * k * (a + b) / 2
                                               + UP * k * x, UP, buff=0.14)
        g_h = tag("x − y", 26, RED_B).next_to(o1 + RIGHT * k * a
                                               + UP * k * (x + y) / 2, RIGHT, buff=0.14)
        self.play(FadeIn(gap), FadeIn(g_w), FadeIn(g_h), run_time=0.9)
        check(abs(area([v[:2] for v in gap.get_vertices()])
                  - k * k * (a - b) * (x - y)) < 1e-9, "gap area (a-b)(x-y)")

        eq1 = tag("ax + by", 32)
        eq2 = _ctag("=  ay + bx", 32, WHITE, {"ay": ORANGE, "bx": TEAL_B})
        eq3 = _ctag("+  (a − b)(x − y)", 32, WHITE, {"(a − b)(x − y)": RED_B})
        eqs = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        eqs.move_to(np.array([4.45, 0.95, 0.0]))
        self.play(FadeIn(eq1), run_time=0.5)
        self.play(FadeIn(eq2), run_time=0.6)
        self.play(FadeIn(eq3), run_time=0.6)

        content = VGroup(F1, F2, l_a, l_x, l_b, l_y, ay, t_ay, lo, hi, t_lo,
                         t_hi, gap, g_w, g_h, eqs)
        _safe(*content)
        _apart(l_a, l_x, l_b, l_y, t_ay, t_lo, t_hi, g_w, g_h, *eqs)
        self.play(Write(caption("a ≥ b,  x ≥ y   ⟹   ax + by  ≥  ay + bx", 34)),
                  run_time=1.2)
        self.hold(2.4)


# ========================================================= H. CALCULUS & LIMITS

class H1_SineDerivative(Board):
    """On the unit circle an arc dθ at P cuts a small right triangle whose
    vertical leg is the rise of sin θ.  The big triangle (1, cos θ, sin θ),
    shrunk by dθ and turned a quarter, is its limit shape: the tangent at P
    is perpendicular to OP, so the angle between the vertical and the arc is
    θ.  A magnified window, rescaled by 1/dθ, shows the arc straightening
    and the rise converging to cos θ · dθ."""

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
        self.hold(0.4)

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
        self.hold(0.4)

        # ---- the window: everything near P magnified by L/dθ
        L = 4.4
        anc = np.array([3.79, -1.95, 0.0])           # P in the window
        sx, cy = np.sin(th), np.cos(th)
        # in the window the triangle's legs are the fall and rise of cos and
        # sin per unit dθ; the fall is widest at the largest dθ shown
        fall0 = (np.cos(th) - np.cos(th + d0)) / d0
        wl, wr = anc[0] - L * fall0 - 0.3, anc[0] + 0.3
        wb, wt = anc[1] - 0.3, anc[1] + L * cy + 0.3
        for d in np.geomspace(d1, d0, 25):
            fall = (np.cos(th) - np.cos(th + d)) / d
            rise = (np.sin(th + d) - np.sin(th)) / d
            check(fall <= fall0 + 1e-12 and rise <= cy,
                  "the magnified triangle stays inside the window")
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
            arc = ParametricFunction(lambda t: to_win(C(th + t * d)), t_range=[0, 1],
                                     color=YELLOW_B, stroke_width=5)
            return VGroup(arc,
                          Line(to_win(P), to_win(Cn), color=BLUE_D, stroke_width=7),
                          Line(to_win(Cn), to_win(P2), color=TEAL_D, stroke_width=6),
                          Dot(anc, radius=0.06))

        zm = always_redraw(zoom)
        ins = always_redraw(inside)
        w_ang = angle_arc(anc, anc + UP, anc + np.array([-sx, cy, 0]), radius=0.85)
        w_th = tag("θ", 28, YELLOW_B).move_to(
            anc + 1.15 * angle_mid_dir(anc, anc + UP, anc + np.array([-sx, cy, 0])))
        mid_h = anc + 0.5 * L * np.array([-sx, cy, 0])
        w_d = tag("dθ", 28, YELLOW_B).move_to(mid_h + 0.40 * np.array([cy, sx, 0]))
        w_sin = tag("sin θ · dθ", 24).next_to(
            np.array([anc[0] - L * sx / 2, wt, 0]), UP, buff=0.12)
        w_cos = tag("cos θ · dθ", 24).next_to(
            np.array([wr, anc[1] + L * cy * 0.62, 0]), RIGHT, buff=0.15)
        w_rise = tag("Δ(sin θ)", 24, BLUE_B).next_to(
            np.array([wr, anc[1] + L * cy * 0.38, 0]), RIGHT, buff=0.15)

        self.play(Create(window), FadeIn(zm), run_time=1.0)
        self.play(FadeIn(ref_w), Create(ref_edge), FadeIn(ins), Create(w_ang),
                  FadeIn(w_th), FadeIn(w_d), FadeIn(w_sin), FadeIn(w_cos),
                  FadeIn(w_rise), run_time=1.0)
        self.hold(0.5)

        # ---- dθ -> 0: the arc straightens, the rise becomes cos θ · dθ
        self.play(FadeOut(l_d), run_time=0.3)
        self.play(lt.animate.set_value(np.log(d1)), run_time=4.5, rate_func=smooth)
        rise = (np.sin(th + d1) - np.sin(th)) / d1
        check(abs(rise - np.cos(th)) * L < 0.03,
              "in the window the rise has converged to cos θ · dθ")
        for m in (sm, rc, zm, ins):
            m.clear_updaters()

        final = VGroup(tag("d(sin θ)", 26, BLUE_B),
                       tag("= cos θ · dθ", 26)).arrange(DOWN, aligned_edge=LEFT,
                                                         buff=0.16)
        final.next_to(np.array([wr, anc[1] + L * cy * 0.5, 0]), RIGHT, buff=0.15)
        self.play(FadeOut(w_cos), FadeOut(w_rise), FadeIn(final), run_time=0.9)

        content = VGroup(axes, quarter, OP, OF, FP, ang, l_th, l_1, l_cos, l_sin,
                         sm, rc, window, zm, ref_w, ref_edge, ins, w_ang, w_th,
                         w_d, w_sin, final)
        _safe(*content)
        _apart(l_th, l_1, l_cos, l_sin, w_th, w_d, w_sin, final)
        self.play(Write(caption("d(sin θ) / dθ  =  cos θ", 36)), run_time=1.2)
        self.hold(2.4)


class H2_RiemannSum(Board):
    """f increasing on [a, b].  The left-endpoint rectangles lie under the
    curve; the right-endpoint ones cover it.  Their difference is a stack of
    small boxes, one per strip, whose heights are the successive rises of f:
    slid sideways they pile into a single column Δx wide and f(b) - f(a)
    tall.  So both sums pinch the area to within (f(b) - f(a))·Δx, which
    halves every time n doubles."""

    def construct(self):
        def f(x):
            return 0.7 + 0.15 * x * x + 0.25 * np.sin(1.3 * x)

        a, b = 1.0, 5.0
        kx, ky = 1.5, 1.2
        org = np.array([-5.7, -2.35, 0.0])

        def P(x, y):
            return org + np.array([kx * x, ky * y, 0.0])

        xs = np.linspace(a, b, 401)
        check(bool(np.all(np.diff(f(xs)) > 0)), "f increases on [a, b]")

        axes = VGroup(Line(P(0, 0), P(7.3, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 4.9), color=GREY_B, stroke_width=2))
        curve = ParametricFunction(lambda t: P(t, f(t)), t_range=[0.35, 5.3],
                                   color=YELLOW_B, stroke_width=5)
        region = Polygon(P(a, 0), *[P(x, f(x)) for x in xs], P(b, 0),
                         fill_color=YELLOW_E, fill_opacity=0.32, stroke_width=0)
        la = _under("a", P(a, 0), 28)
        lb = _under("b", P(b, 0), 28)
        ea = DashedLine(P(a, 0), P(a, f(a)), color=GREY_B, stroke_width=2)
        eb = DashedLine(P(b, 0), P(b, f(b)), color=GREY_B, stroke_width=2)

        self.play(Create(axes), run_time=0.7)
        self.play(Create(curve), run_time=1.2)
        self.play(FadeIn(region), Create(ea), Create(eb), FadeIn(la), FadeIn(lb),
                  run_time=0.9)

        # the column's height never changes: f(a) up to f(b)
        xd = P(b + 1.0, 0)[0] + 0.3                 # right of the widest column
        dim = VGroup(Line([xd, P(0, f(a))[1], 0], [xd, P(0, f(b))[1], 0],
                          color=ORANGE, stroke_width=3),
                     Line([xd - 0.1, P(0, f(a))[1], 0], [xd + 0.1, P(0, f(a))[1], 0],
                          color=ORANGE, stroke_width=3),
                     Line([xd - 0.1, P(0, f(b))[1], 0], [xd + 0.1, P(0, f(b))[1], 0],
                          color=ORANGE, stroke_width=3))
        g_a = DashedLine(P(a, f(a)), [xd, P(0, f(a))[1], 0], color=GREY_B,
                         stroke_width=1.5, dash_length=0.08)
        g_b = DashedLine(P(b, f(b)), [xd, P(0, f(b))[1], 0], color=GREY_B,
                         stroke_width=1.5, dash_length=0.08)
        l_rise = tag("f(b) − f(a)", 26, ORANGE).next_to(dim, RIGHT, buff=0.15)

        def pieces(n):
            dx = (b - a) / n
            x = [a + i * dx for i in range(n + 1)]
            sw = {4: 2.0, 8: 1.6, 16: 1.0, 32: 0.6}[n]
            low = VGroup(*[mk([P(x[i], 0), P(x[i + 1], 0), P(x[i + 1], f(x[i])),
                               P(x[i], f(x[i]))], BLUE_D, stroke_width=sw)
                           for i in range(n)])
            gaps = VGroup(*[mk([P(x[i], f(x[i])), P(x[i + 1], f(x[i])),
                                P(x[i + 1], f(x[i + 1])), P(x[i], f(x[i + 1]))],
                               ORANGE, stroke_width=sw) for i in range(n)])
            shifts = [P(b, 0) - P(x[i], 0) for i in range(n)]
            return low, gaps, shifts, dx

        n_lab = None
        old = None
        for n, t_in, t_sl in [(4, 0.8, 1.6), (8, 1.0, 1.2), (16, 1.0, 1.2),
                              (32, 0.8, 1.0)]:
            low, gaps, shifts, dx = pieces(n)
            lab = tag(f"n = {n}", 30).move_to(np.array([-4.3, 2.9, 0.0]))
            if old is None:
                self.play(FadeIn(low), FadeIn(lab), run_time=t_in)
                self.play(FadeIn(gaps), run_time=t_in)
            else:
                self.play(*[FadeOut(m) for m in old], FadeIn(low), FadeIn(gaps),
                          ReplacementTransform(n_lab, lab), run_time=t_in)
            n_lab = lab
            self.bring_to_front(curve)
            self.play(*[g.animate.shift(sh) for g, sh in zip(gaps, shifts)],
                      run_time=t_sl)
            # the boxes now form one column [b, b + Δx] x [f(a), f(b)]
            xl, xr = P(b, 0)[0], P(b + dx, 0)[0]
            ys = []
            for g in gaps:
                lo_, hi_ = g.get_critical_point(DL), g.get_critical_point(UR)
                check(abs(lo_[0] - xl) < 1e-9 and abs(hi_[0] - xr) < 1e-9,
                      "every box lands in the column")
                ys.append((lo_[1], hi_[1]))
            ys.sort()
            check(all(abs(ys[i][1] - ys[i + 1][0]) < 1e-9 for i in range(n - 1))
                  and abs(ys[0][0] - P(0, f(a))[1]) < 1e-9
                  and abs(ys[-1][1] - P(0, f(b))[1]) < 1e-9,
                  "the boxes stack without gaps from f(a) to f(b)")
            l_dx = tag("Δx", 26).next_to(np.array([(xl + xr) / 2,
                                                   P(0, f(a))[1], 0]), DOWN, buff=0.12)
            if n == 4:
                self.play(Create(g_a), Create(g_b), Create(dim), FadeIn(l_rise),
                          FadeIn(l_dx), run_time=0.8)
                self.hold(0.6)
            else:
                self.play(FadeIn(l_dx), run_time=0.3)
                self.hold(0.3)
            old = [low, gaps, l_dx]

        content = VGroup(axes, curve, region, la, lb, ea, eb, dim, g_a, g_b,
                         l_rise, n_lab, *old)
        _safe(*content)
        _apart(la, lb, l_rise, n_lab, old[2])
        cap = _cap(_row(_int("a", "b", 32), tag("f(x) dx", 32, YELLOW_B),
                        tag("=", 32, YELLOW_B), _lim(32),
                        tag("∑ f(xₖ) Δx", 32, YELLOW_B), buff=0.22))
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H3_IntegralOfX(Board):
    """The region under y = x over [0, a] is the right triangle with legs
    a and a.  Turned half a revolution about the centre of the a by a
    square it lands exactly on the other half of the square, so twice the
    integral is a²."""

    def construct(self):
        a, k = 3.0, 1.75
        org = np.array([-4.4, -2.2, 0.0])

        def P(x, y):
            return org + k * np.array([x, y, 0.0])

        axes = VGroup(Line(P(0, 0), P(a + 0.55, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, a + 0.3), color=GREY_B, stroke_width=2))
        line = Line(P(0, 0), P(a + 0.25, a + 0.25), color=YELLOW_B, stroke_width=5)
        l_line = tag("y = x", 28, YELLOW_B).next_to(P(a + 0.25, a + 0.25), RIGHT,
                                                    buff=0.12)
        self.play(Create(axes), run_time=0.7)
        self.play(Create(line), FadeIn(l_line), run_time=1.0)

        # sweep the region under the line from 0 to a
        t = ValueTracker(0.001)
        sweep = always_redraw(lambda: mk([P(0, 0), P(t.get_value(), 0),
                                          P(t.get_value(), t.get_value())],
                                         BLUE_D, stroke_width=2))
        edge = always_redraw(lambda: Line(P(t.get_value(), 0),
                                          P(t.get_value(), t.get_value()),
                                          color=WHITE, stroke_width=4))
        self.add(sweep, edge)
        self.play(t.animate.set_value(a), run_time=2.0, rate_func=linear)
        sweep.clear_updaters()
        edge.clear_updaters()
        low = mk([P(0, 0), P(a, 0), P(a, a)], BLUE_D, stroke_width=2)
        self.remove(sweep, edge)
        self.add(low)
        self.bring_to_front(line)

        la = tag("a", 30).next_to(P(a, 0), DOWN, buff=0.14)
        lh = tag("a", 30).next_to(P(a, a / 2), RIGHT, buff=0.14)

        def int_label():
            return _row(_int("0", "a", 28, WHITE), tag("x dx", 28), buff=0.1)

        il = int_label().move_to(P(2 * a / 3, a / 3))
        self.play(FadeIn(la), FadeIn(lh), FadeIn(il), run_time=0.8)
        self.hold(0.4)

        # the square on [0, a], and the half-turn about its centre
        sq = Polygon(P(0, 0), P(a, 0), P(a, a), P(0, a), stroke_color=WHITE,
                     stroke_width=4)
        self.play(Create(sq), run_time=0.8)
        cp = low.copy().set_fill(TEAL_D)
        self.play(FadeIn(cp), run_time=0.3)
        self.play(Rotate(cp, angle=PI, about_point=P(a / 2, a / 2)), run_time=1.8)
        got = sorted(map(tuple, np.round(cp.get_vertices()[:, :2], 8)))
        want = sorted(map(tuple, np.round([P(0, 0)[:2], P(0, a)[:2], P(a, a)[:2]], 8)))
        check(np.allclose(got, want, atol=1e-6), "the half-turn fills the other half")
        check(abs(area([(0, 0), (a, 0), (a, a)]) * 2 - a * a) < 1e-12,
              "two triangles make the square")
        self.bring_to_front(line, l_line)
        il2 = int_label().move_to(P(a / 3, 2 * a / 3))
        self.play(FadeIn(il2), run_time=0.6)

        rhs = _row(tag("2 ·", 32), _int("0", "a", 32, WHITE), tag("x dx  =  a²", 32),
                   buff=0.12).move_to(np.array([3.9, 0.4, 0.0]))
        self.play(FadeIn(rhs), run_time=0.8)

        content = VGroup(axes, line, l_line, low, cp, la, lh, il, il2, sq, rhs)
        _safe(*content)
        _apart(l_line, la, lh, il, il2, rhs)
        cap = _cap(_row(_int("0", "a", 34), tag("x dx   =   a² / 2", 34, YELLOW_B),
                        buff=0.12))
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H4_IntegralOfXSquared(Board):
    """A = area under y = x² over [0, a].  Stretching by 2 across and 4 up
    maps the parabola onto itself, so the area over [0, 2a] is 8A.  Over
    [a, 2a] each slice has height (a+t)² = a² + 2at + t²: an a x a² block
    (a³), a triangle under the tangent at x = a (legs a and 2a²: a³), and a
    sliver whose slices are t² — the region A itself, sheared upward slice
    by slice (Cavalieri).  So 8A = A + a³ + a³ + A and A = a³/3."""

    def construct(self):
        a = 1.0
        kx, ky = 2.6, 1.42
        org = np.array([-5.6, -2.2, 0.0])

        def P(x, y):
            return org + np.array([kx * x, ky * y, 0.0])

        N = 120
        ts = np.linspace(0.0, a, N)

        axes = VGroup(Line(P(0, 0), P(2 * a + 0.3, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 4.1 * a * a), color=GREY_B, stroke_width=2))
        para = ParametricFunction(lambda s: P(s, s * s), t_range=[0, 2 * a],
                                  color=YELLOW_B, stroke_width=4)
        ax_l = VGroup(_under("a", P(a, 0), 28),
                      _under("2a", P(2 * a, 0), 28),
                      tag("a²", 28).next_to(P(0, a * a), LEFT, buff=0.14),
                      tag("4a²", 28).next_to(P(0, 4 * a * a), LEFT, buff=0.14))
        ticks = VGroup(*[Line(P(x, 0) + DOWN * 0.07, P(x, 0) + UP * 0.07,
                              color=GREY_B, stroke_width=2) for x in (a, 2 * a)],
                       *[Line(P(0, y) + LEFT * 0.07, P(0, y) + RIGHT * 0.07,
                              color=GREY_B, stroke_width=2) for y in (a * a, 4 * a * a)])
        self.play(Create(axes), run_time=0.7)
        self.play(Create(para), FadeIn(ax_l), FadeIn(ticks), run_time=1.1)

        A_pts = [P(t, t * t) for t in ts] + [P(a, 0)]
        regA = mk(A_pts, BLUE_D, stroke_width=2)
        lA = tag("A", 30).move_to(P(0.74 * a, 0.2 * a * a))
        self.play(FadeIn(regA), FadeIn(lA), run_time=0.8)
        self.hold(0.3)

        # ---- stretch by 2 across and 4 up: the parabola maps onto itself
        def stretch(p):
            return org + np.array([2 * (p[0] - org[0]), 4 * (p[1] - org[1]), 0.0])

        box = DashedVMobject(Polygon(P(0, 0), P(a, 0), P(a, a * a), P(0, a * a),
                                     stroke_color=GREY_A, stroke_width=2),
                             num_dashes=24)
        self.play(Create(box), run_time=0.5)
        big = regA.copy()
        bigbox = box.copy()
        x2 = _under("×2", P(1.5 * a, 0), 28, GREY_A)
        x4 = tag("×4", 26, GREY_A).next_to(P(0, 2.5 * a * a), LEFT, buff=0.16)
        self.play(big.animate.apply_function(stretch).set_fill(YELLOW_E, 0.2)
                  .set_stroke(YELLOW_B, 3),
                  bigbox.animate.apply_function(stretch),
                  FadeIn(x2), FadeIn(x4), run_time=2.2)
        want_big = [P(2 * t, 4 * t * t) for t in ts] + [P(2 * a, 0)]
        check(np.allclose(big.get_vertices(), np.array(want_big), atol=1e-6),
              "the stretched region is the region under y = x² over [0, 2a]")
        self.bring_to_back(big)
        self.bring_to_back(axes)
        grid = VGroup(DashedLine(P(a, 0), P(a, 4 * a * a), color=GREY_B,
                                 stroke_width=1.5, dash_length=0.08),
                      *[DashedLine(P(0, j * a * a), P(2 * a, j * a * a), color=GREY_B,
                                   stroke_width=1.5, dash_length=0.08)
                        for j in (1, 2, 3)])
        l8 = tag("8A", 30, YELLOW_B).next_to(P(2 * a, 2 * a * a), RIGHT, buff=0.18)
        self.play(Create(grid), FadeIn(l8), run_time=0.8)
        self.hold(0.4)

        # ---- over [a, 2a]: a block, a triangle under the tangent, a sliver
        block = mk([P(a, 0), P(2 * a, 0), P(2 * a, a * a), P(a, a * a)], TEAL_D)
        l_blk = tag("a³", 30).move_to(P(1.5 * a, 0.5 * a * a))
        self.play(FadeIn(block), FadeIn(l_blk), run_time=0.8)
        tri = mk([P(a, a * a), P(2 * a, a * a), P(2 * a, 3 * a * a)], GREEN_D)
        l_tri = tag("a³", 30).move_to(P(5 * a / 3, 5 * a * a / 3))
        check(abs(area([(a, a * a), (2 * a, a * a), (2 * a, 3 * a * a)]) - a ** 3) < 1e-12,
              "triangle under the tangent = a³")
        self.play(FadeIn(tri), FadeIn(l_tri), run_time=0.9)
        self.hold(0.3)

        sl = regA.copy()
        self.add(sl)
        self.play(sl.animate.shift(P(a, a * a) - P(0, 0)), run_time=1.2)
        sheared = mk([P(a + t, (a + t) ** 2) for t in ts] + [P(2 * a, 3 * a * a)],
                     BLUE_D, stroke_width=2)
        self.play(Transform(sl, sheared), run_time=1.6)
        A_math = [(t, t * t) for t in ts] + [(a, 0)]
        S_math = [(a + t, (a + t) ** 2) for t in ts] + [(2 * a, 3 * a * a)]
        check(abs(area(A_math) - area(S_math)) < 1e-12,
              "the shear keeps the area: sliver = A")
        check(abs(abs(area(A_math)) - a ** 3 / 3) < 1e-4, "A ≈ a³/3 (polygon check)")
        l_sl = tag("A", 30).move_to(P(1.8 * a, 1.8 ** 2 * a * a - 0.31 * a * a))
        self.play(FadeIn(l_sl), run_time=0.4)
        self.bring_to_front(para)

        # ---- the tally
        tally = VGroup(
            _ctag("8A  =  A + a³ + a³ + A", 32, WHITE, {"A": BLUE_B}),
            _ctag("6A  =  2a³", 32, WHITE, {"A": BLUE_B}),
            _ctag("A  =  a³ / 3", 32, WHITE, {"A": BLUE_B}),
        ).arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to(np.array([3.6, 0.8, 0.0]))
        for line in tally:
            self.play(FadeIn(line), run_time=0.6)

        content = VGroup(axes, para, ax_l, regA, lA, big, bigbox, box, grid, x2, x4,
                         l8, block, l_blk, tri, l_tri, sl, l_sl, tally)
        _safe(*content)
        _apart(*ax_l, lA, x2, x4, l8, l_blk, l_tri, l_sl, *tally)
        cap = _cap(_row(_int("0", "a", 34), tag("x² dx   =   a³ / 3", 34, YELLOW_B),
                        buff=0.12))
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H5_HyperbolaArea(Board):
    """The squeeze (x, y) -> (a·x, y/a) stretches across by a and squashes up
    by 1/a, so it keeps every area, and it maps y = 1/x onto itself.  It
    therefore carries the region under 1/x over [1, b] onto the region over
    [a, ab] with the same area; the region over [1, ab] is the one over
    [1, a] plus that one."""

    def construct(self):
        A, B = 2.0, 1.7                       # a and b, with b < a so the
        AB = A * B                            # strips [1,b] and [a,ab] are apart
        kx, ky = 2.9, 3.15
        org = np.array([-5.75, -2.35, 0.0])

        def P(x, y):
            return org + np.array([kx * x, ky * y, 0.0])

        N = 90
        axes = VGroup(Line(P(0, 0), P(AB + 0.5, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 1.85), color=GREY_B, stroke_width=2))
        curve = ParametricFunction(lambda t: P(t, 1 / t), t_range=[0.555, AB + 0.45],
                                   color=YELLOW_B, stroke_width=4)
        l_curve = tag("y = 1/x", 28, YELLOW_B).next_to(P(0.62, 1 / 0.62), RIGHT,
                                                       buff=0.2)
        self.play(Create(axes), run_time=0.7)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.1)
        ticks = VGroup()
        tlabs = VGroup()
        for v, s in [(1, "1"), (B, "b"), (A, "a"), (AB, "ab")]:
            ticks.add(Line(P(v, 0) + DOWN * 0.07, P(v, 0) + UP * 0.07,
                           color=GREY_B, stroke_width=2))
            tlabs.add(_under(s, P(v, 0), 28))
        self.play(FadeIn(ticks), FadeIn(tlabs), run_time=0.6)

        def strip_pts(lo, hi):
            xs = np.linspace(lo, hi, N)
            return [(x, 1 / x) for x in xs] + [(hi, 0.0), (lo, 0.0)]

        def region(pts, col, cuts):
            reg = mk([P(*p) for p in pts], col)
            lines = VGroup(*[Line(P(c[0], 0), P(c[0], c[1]), color=WHITE,
                                  stroke_width=1.2) for c in cuts])
            return VGroup(reg, lines)

        base = strip_pts(1.0, B)                       # math points of [1, b]
        cuts = [(1 + (B - 1) * j / 4, 1 / (1 + (B - 1) * j / 4)) for j in (1, 2, 3)]

        def squeezed(s):
            return region([(s * x, y / s) for x, y in base], TEAL_D,
                          [(s * x, y / s) for x, y in cuts])

        Rb = squeezed(1.0)
        l_b = _row(_int("1", "b", 27, WHITE), tag("dx/x", 27), buff=0.08)
        l_b.move_to(P((1 + B) / 2, 0.36))
        self.play(FadeIn(Rb), FadeIn(l_b), run_time=0.9)
        self.hold(0.4)

        legend = tag("(x, y)  →  (a·x,  y / a)", 28, GREY_A).move_to(
            np.array([3.2, 2.7, 0.0]))
        self.play(FadeIn(legend), run_time=0.6)

        # squeeze a copy along the hyperbola: s runs from 1 to a
        mv = squeezed(1.0)
        self.add(mv)

        def upd(m, alpha):
            m.become(squeezed(A ** alpha))

        self.play(UpdateFromAlphaFunc(mv, upd), run_time=3.0)
        img = [(A * x, y / A) for x, y in base]
        check(all(abs(y - 1 / x) < 1e-12 for x, y in img[:N]),
              "the squeezed top edge lies on y = 1/x")
        check(abs(img[0][0] - A) < 1e-12 and abs(img[N - 1][0] - AB) < 1e-12,
              "[1, b] lands on [a, ab]")
        check(abs(abs(area(img)) - abs(area(base))) < 1e-12,
              "the squeeze keeps the area")
        l_mv = _row(_int("1", "b", 27, WHITE), tag("dx/x", 27), buff=0.08)
        l_mv.move_to(P((A + AB) / 2, 0.165))
        self.play(FadeIn(l_mv), run_time=0.5)
        self.hold(0.4)

        # the region over [1, a] completes the region over [1, ab]
        Ra = mk([P(*p) for p in strip_pts(1.0, A)], BLUE_D)
        l_a = _row(_int("1", "a", 27, WHITE), tag("dx/x", 27), buff=0.08)
        l_a.move_to(P(1.5, 0.33))
        self.play(FadeOut(Rb), FadeOut(l_b), FadeIn(Ra), FadeIn(l_a), run_time=1.0)
        whole = Polygon(*[P(*p) for p in strip_pts(1.0, AB)], stroke_color=YELLOW_B,
                        stroke_width=5)
        self.bring_to_front(curve)
        self.play(Create(whole), run_time=1.0)
        self.hold(0.3)

        content = VGroup(axes, curve, l_curve, ticks, tlabs, Ra, l_a, mv, l_mv,
                         whole, legend)
        _safe(*content)
        _apart(l_curve, *tlabs, l_a, l_mv, legend)
        cap = _cap(_row(_int("1", "ab", 32), tag("dx/x   =", 32, YELLOW_B),
                        _int("1", "a", 32), tag("dx/x   +", 32, YELLOW_B),
                        _int("1", "b", 32), tag("dx/x", 32, YELLOW_B), buff=0.14))
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


class H6_IntegrationByParts(Board):
    """A curve rising from (u₁, v₁) to (u₂, v₂) cuts the rectangle
    [0, u₂] x [0, v₂], less the corner [0, u₁] x [0, v₁], into the region
    below it (∫ v du, vertical strips) and the region to its left (∫ u dv,
    horizontal strips).  Change the curve and the split changes, but the
    two always fill the same L-shape: u₂v₂ - u₁v₁."""

    def construct(self):
        P1m, P2m = np.array([1.2, 0.8]), np.array([4.6, 3.4])
        Ca = (np.array([2.2, 2.4]), np.array([3.0, 3.1]))     # rises early
        Cb = (np.array([3.4, 0.9]), np.array([4.4, 1.6]))     # rises late
        k = 1.48
        org = np.array([-6.05, -2.3, 0.0])

        def P(x, y):
            return org + k * np.array([x, y, 0.0])

        mt = ValueTracker(0.0)
        ts = np.linspace(0.0, 1.0, 120)

        def curve_pts(m):
            c1 = (1 - m) * Ca[0] + m * Cb[0]
            c2 = (1 - m) * Ca[1] + m * Cb[1]
            t = ts[:, None]
            return ((1 - t) ** 3 * P1m + 3 * (1 - t) ** 2 * t * c1
                    + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * P2m)

        for m in (0.0, 0.5, 1.0):
            cv = curve_pts(m)
            check(bool(np.all(np.diff(cv[:, 0]) > 0) and np.all(np.diff(cv[:, 1]) > 0)),
                  "the curve rises in both u and v")
            below = [tuple(p) for p in cv] + [(P2m[0], 0.0), (P1m[0], 0.0)]
            left = [tuple(p) for p in cv] + [(0.0, P2m[1]), (0.0, P1m[1])]
            check(abs(abs(area(below)) + abs(area(left)) + P1m[0] * P1m[1]
                      - P2m[0] * P2m[1]) < 1e-9,
                  "below + left + corner = the u₂ x v₂ rectangle")

        axes = VGroup(Line(P(0, 0), P(5.25, 0), color=GREY_B, stroke_width=2),
                      Line(P(0, 0), P(0, 3.95), color=GREY_B, stroke_width=2))
        l_u = tag("u", 30).next_to(P(5.25, 0), RIGHT, buff=0.12)
        l_v = tag("v", 30).next_to(P(0, 3.95), RIGHT, buff=0.14)
        self.play(Create(axes), FadeIn(l_u), FadeIn(l_v), run_time=0.8)

        def curve_mob():
            return VMobject(stroke_color=YELLOW_B, stroke_width=5).set_points_smoothly(
                [P(*p) for p in curve_pts(mt.get_value())])

        crv = curve_mob()
        d1, d2 = Dot(P(*P1m), color=YELLOW_B), Dot(P(*P2m), color=YELLOW_B)
        self.play(FadeIn(d1), run_time=0.3)
        self.play(Create(crv), run_time=1.4)
        self.play(FadeIn(d2), run_time=0.3)

        dash = VGroup(*[DashedLine(P(*p), P(*q), color=GREY_B, stroke_width=1.5,
                                   dash_length=0.08)
                        for p, q in [((P1m[0], 0), P1m), ((0, P1m[1]), P1m),
                                     ((P2m[0], 0), P2m), ((0, P2m[1]), P2m)]])
        ticks = VGroup(_under("u₁", P(P1m[0], 0), 28), _under("u₂", P(P2m[0], 0), 28),
                       tag("v₁", 28).next_to(P(0, P1m[1]), LEFT, buff=0.14),
                       tag("v₂", 28).next_to(P(0, P2m[1]), LEFT, buff=0.14))
        self.play(Create(dash), FadeIn(ticks), run_time=0.8)

        def below_mob():
            cv = curve_pts(mt.get_value())
            return mk([P(*p) for p in cv] + [P(P2m[0], 0), P(P1m[0], 0)], BLUE_D,
                      stroke_width=0)

        def left_mob():
            cv = curve_pts(mt.get_value())
            return mk([P(*p) for p in cv] + [P(0, P2m[1]), P(0, P1m[1])], ORANGE,
                      stroke_width=0)

        lab_b = _row(_int("u₁", "u₂", 30, WHITE), tag("v du", 30), buff=0.08)
        lab_b.move_to(P(3.93, 0.5))
        lab_l = _row(_int("v₁", "v₂", 30, WHITE), tag("u dv", 30), buff=0.08)
        lab_l.move_to(P(1.0, 2.95))

        def to_math(p):
            return (p - org)[:2] / k

        for m in np.linspace(0, 1, 11):           # labels stay inside their regions
            cv = curve_pts(m)
            (u0, v0), (u1, v1) = (to_math(lab_b.get_corner(DL)),
                                  to_math(lab_b.get_corner(UR)))
            check(P1m[0] < u0 and u1 < P2m[0] and v0 > 0
                  and v1 < np.interp(u0, cv[:, 0], cv[:, 1]),
                  "the ∫ v du label stays under the curve")
            (u0, v0), (u1, v1) = (to_math(lab_l.get_corner(DL)),
                                  to_math(lab_l.get_corner(UR)))
            check(P1m[1] < v0 and v1 < P2m[1] and u0 > 0
                  and u1 < np.interp(v0, cv[:, 1], cv[:, 0]),
                  "the ∫ u dv label stays left of the curve")
        reg_b, reg_l = below_mob(), left_mob()
        self.play(FadeIn(reg_b), FadeIn(lab_b), run_time=0.9)
        self.play(FadeIn(reg_l), FadeIn(lab_l), run_time=0.9)
        corner = mk([P(0, 0), P(P1m[0], 0), P(*P1m), P(0, P1m[1])], GREY_D, 0.9,
                    stroke_width=0)
        l_c = tag("u₁v₁", 24).move_to(P(P1m[0] / 2, P1m[1] / 2))
        self.play(FadeIn(corner), FadeIn(l_c), run_time=0.6)
        rect = Polygon(P(0, 0), P(P2m[0], 0), P(*P2m), P(0, P2m[1]),
                       stroke_color=YELLOW_B, stroke_width=4)
        self.bring_to_front(crv, d1, d2, lab_b, lab_l, l_c)
        self.play(Create(rect), run_time=0.8)

        eq = VGroup(_row(tag("+", 32, BLACK), _int("v₁", "v₂", 32, ORANGE),
                         tag("u dv", 32, ORANGE), buff=0.1),
                    _row(tag("+", 32), _int("u₁", "u₂", 32, BLUE_B),
                         tag("v du", 32, BLUE_B), buff=0.1),
                    tag("=  u₂v₂ − u₁v₁", 32)).arrange(DOWN, buff=0.3,
                                                       aligned_edge=LEFT)
        eq.move_to(np.array([4.35, 0.9, 0.0]))
        for line in eq:
            self.play(FadeIn(line), run_time=0.5)
        self.hold(0.4)

        # another curve through the same two points: a new split, same total
        live_b, live_l, live_c = (always_redraw(below_mob), always_redraw(left_mob),
                                  always_redraw(curve_mob))
        self.remove(reg_b, reg_l, crv)
        self.add(live_b, live_l, live_c)
        self.bring_to_front(rect, live_c, d1, d2, lab_b, lab_l, l_c)
        self.play(mt.animate.set_value(1.0), run_time=2.0)
        self.hold(0.3)
        self.play(mt.animate.set_value(0.0), run_time=1.8)
        for m in (live_b, live_l, live_c):
            m.clear_updaters()

        content = VGroup(axes, l_u, l_v, live_b, live_l, live_c, d1, d2, dash, ticks,
                         lab_b, lab_l, corner, l_c, rect, eq)
        _safe(*content)
        _apart(l_u, l_v, *ticks, lab_b, lab_l, l_c, eq)
        cap = _cap(_row(_int("v₁", "v₂", 32), tag("u dv  +", 32, YELLOW_B),
                        _int("u₁", "u₂", 32), tag("v du   =   u₂v₂ − u₁v₁", 32,
                                                    YELLOW_B), buff=0.12))
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)
