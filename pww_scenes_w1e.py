# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes_w1e.py — proofs without words: G22, E15, G11, E23, G14, G8, G21,
E28, G9, G10.

Built on pww_kit (Frame, mk, tag, caption, guide, angle_arc, check, ...).
Every scene states its geometric invariants with check(...) — lengths,
angles, exact landings of moved pieces, and that labels stay clear of the
lines and of each other — so a wrong construction fails the render instead
of drawing a wrong picture.
"""

from pww_kit import *


# ------------------------------------------------------------ private helpers

def _dir(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1 = (p - v) / np.linalg.norm(p - v)
    u2 = (q - v) / np.linalg.norm(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _wedge(vertex, p, q, radius, color, op=0.6):
    """Filled sector for the non-reflex angle p-vertex-q (screen points)."""
    v = to3(vertex)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return Sector(radius=radius, angle=span, start_angle=a1, arc_center=v,
                  fill_color=color, fill_opacity=op, stroke_width=0)


def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centroid while the
    centroid slides — carrying the screen points src onto dst (same order).
    Fails the render unless one rotation + translation lands every point.
    `turn` (radians) picks the direction when the angle is ±180°."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = np.arctan2(*(src[1] - src[0])[1::-1])
        a1 = np.arctan2(*(dst[1] - dst[0])[1::-1])
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    cs, sn = np.cos(turn), np.sin(turn)
    Rm = np.array([[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]])
    check(all(close(m1 + Rm @ (p - m0), q, 1e-6) for p, q in zip(src, dst)),
          "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _spiral(mob, centre, turn, factor, **kw):
    """Spiral similarity about `centre`: turn by `turn` while scaling by
    `factor` (geometrically in time), exact at the end."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=c)
                 .scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on
    the side the vector `side` points to, with its box clear of the line
    by `gap` whatever the slope."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


def _refl(p, c, u):
    """Mirror image of screen point p in the line through c along u."""
    p, c, u = to3(p), to3(c), _unit(u)
    d = p - c
    return c + 2 * np.dot(d, u) * u - d


def _ticks(p, q, n=1, color=WHITE, size=0.13, width=2.5, at=0.5):
    """n short tick marks across segment pq (equal-length marks)."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    g = VGroup()
    for k in range(n):
        o = c + d * 0.09 * (k - (n - 1) / 2)
        g.add(Line(o - nrm * size, o + nrm * size, color=color,
                   stroke_width=width))
    return g


def _pip(p, poly):
    """Point strictly inside polygon (ray casting); p, poly in any 2D/3D."""
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


def _tiles_exactly(polys, region, n=120):
    """True if the polygons tile `region` with no gap and no overlap
    (checked on a jittered grid over the region's bounding box)."""
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


# ---- label hygiene: every label is checked against the lines it must clear

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _arc_segs(c, r, a0, a1, n=48):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    pts = [c + r * _dir(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by
    pad), or None."""
    x0, x1, y0, y1 = _box(m, pad)
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.02) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _clear(m, segs, pad=0.06):
    return _hit(m, segs, pad) is None


def _overlap(a, b, gap=0.05):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _in_frame(*mobs):
    return all(m.get_left()[0] >= -SAFE_X and m.get_right()[0] <= SAFE_X
               and m.get_top()[1] <= SAFE_TOP and m.get_bottom()[1] >= SAFE_BOTTOM
               for m in mobs)


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every line in segs (a
    list of screen segments) and of every other label."""
    for m in labels:
        check(_in_frame(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{_name(m)}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


# ============================================ G8 / G9  — the difference rectangle

class _DiffRect(Board):
    """The rectangle of the sum formulas (G2, G3) with angle B turned back.

    OP = 1 at angle A − B. Its foot Q on the ray at angle A gives OQ = cos B
    and QP = sin B — QP now on the other side of the ray. The horizontal
    through Q and the vertical through P make a rectangle O W Y Z with Q on
    the top side ZY and P on the right side WY. The angle at P between PY
    and PQ is A (both sides of the angle at O turned through 90°), so the
    corner triangle Q Y P has PY = cosA·sinB and QY = sinA·sinB, while
    OZ = sinA·cosB, ZQ = cosA·cosB, WP = sin(A−B), OW = cos(A−B).
    """

    A_DEG, B_DEG = 64.0, 28.0
    R = 5.5
    O = np.array([-4.6, -2.15, 0.0])

    def common(self):
        Aa, Ba = self.A_DEG * DEGREES, self.B_DEG * DEGREES
        R, O = self.R, self.O
        tP = Aa - Ba

        def S(p):
            return O + R * to3(p)

        Qm = np.cos(Ba) * np.array([np.cos(Aa), np.sin(Aa)])
        Pm = np.array([np.cos(tP), np.sin(tP)])
        Ym, Zm, Wm = (np.array([Pm[0], Qm[1]]), np.array([0.0, Qm[1]]),
                      np.array([Pm[0], 0.0]))
        check(close(np.linalg.norm(Qm), np.cos(Ba))
              and close(np.linalg.norm(Pm - Qm), np.sin(Ba))
              and abs(np.dot(Pm - Qm, Qm)) < 1e-12,
              "OQ = cos B, QP = sin B, QP ⊥ OQ")
        check(0 < Qm[0] < Pm[0] and 0 < Pm[1] < Qm[1],
              "Q on the top side, P on the right side of the rectangle")
        check(close(Qm[1], np.sin(Aa) * np.cos(Ba))
              and close(Ym[1] - Pm[1], np.cos(Aa) * np.sin(Ba))
              and close(Pm[1], np.sin(Aa - Ba)), "OZ, PY, WP")
        check(close(Qm[0], np.cos(Aa) * np.cos(Ba))
              and close(Ym[0] - Qm[0], np.sin(Aa) * np.sin(Ba))
              and close(Pm[0], np.cos(Aa - Ba)), "ZQ, QY, OW")
        Q, Pt, Y, Z, W = S(Qm), S(Pm), S(Ym), S(Zm), S(Wm)
        check(abs(_ang(Pt, Y, Q) - Aa) < 1e-9, "angle YPQ = A")
        check(abs(_ang(O, Pt, Q) - Ba) < 1e-9, "angle POQ = B")

        xax = Line(O, O + 1.12 * R * RIGHT, color=GREY_D, stroke_width=2)
        yax = Line(O, O + 1.04 * R * UP, color=GREY_D, stroke_width=2)
        arc_lo = Arc(radius=R, start_angle=0, angle=tP, arc_center=O,
                     color=GREY_B, stroke_width=2.5)
        arc_hi = Arc(radius=R, start_angle=tP, angle=PI / 2 - tP, arc_center=O,
                     color=GREY_B, stroke_width=2.5)
        ray_end = O + 1.1 * R * _dir(Aa)
        ray = Line(O, ray_end, color=GREY_B, stroke_width=2)
        fillA = _wedge(O, O + RIGHT, Q, 0.75, GREEN_B, 0.22)
        angA = angle_arc(O, O + RIGHT, Q, radius=0.75, color=GREEN_B, width=4)
        labA = tag("A", 24, GREEN_B).move_to(O + 1.1 * _dir(0.4 * tP))
        OP = Line(O, Pt, color=YELLOW_B, stroke_width=4)
        angB = angle_arc(O, Pt, Q, radius=1.35, color=PURPLE_A, width=4)
        labB = tag("B", 24, PURPLE_A).move_to(O + 1.68 * _dir(tP + Ba / 2))
        lab1 = _beside(tag("1", 26, YELLOW_B), O, Pt, _dir(tP - PI / 2), 0.12)

        self.play(Create(xax), Create(yax), Create(arc_lo), Create(arc_hi),
                  FadeIn(Dot(O)), run_time=1.0)
        self.play(Create(ray), FadeIn(fillA), Create(angA), FadeIn(labA),
                  run_time=0.7)
        self.play(Create(OP), Create(angB), FadeIn(labB), FadeIn(lab1),
                  run_time=0.9)

        # drop P onto the ray: a right triangle with hypotenuse 1, angle B
        OQ = Line(O, Q, color=BLUE_B, stroke_width=6)
        QP = Line(Pt, Q, color=TEAL_B, stroke_width=6)
        mQ = _ra(Q, O, Pt, 0.2)
        lcB = _beside(tag("cos B", 22, BLUE_B), O, Q, _dir(Aa + PI / 2), 0.12)
        lsB = _beside(tag("sin B", 22, TEAL_B), Q, Pt, _dir(Aa + PI), 0.16)
        self.play(Create(QP), Create(mQ), run_time=0.8)
        self.play(Create(OQ), FadeIn(lcB), FadeIn(lsB), run_time=0.7)

        # the rectangle: the horizontal through Q, the vertical through P
        dZQ = DashedLine(Z, Q, color=GREY_B, stroke_width=2, dash_length=0.08)
        dQY = DashedLine(Q, Y, color=GREY_B, stroke_width=2, dash_length=0.08)
        dWY = DashedLine(W, Y, color=GREY_B, stroke_width=2, dash_length=0.08)
        mY = _ra(Y, Q, W, 0.18, GREY_B, 2)
        mZ = _ra(Z, O, Y, 0.18, GREY_B, 2)
        mW = _ra(W, O, Y, 0.18, GREY_B, 2)
        self.play(FadeOut(arc_hi), Create(dZQ), Create(dQY), Create(dWY),
                  Create(mY), Create(mZ), Create(mW), run_time=0.9)

        # the angle at P is A: both sides of the angle at O turned through 90°
        r = 0.52
        wedge = Sector(radius=r, angle=Aa, start_angle=0.0, arc_center=O,
                       fill_color=GREEN_B, fill_opacity=0.55, stroke_width=0)
        self.play(FadeIn(wedge), run_time=0.3)
        self.play(_rigid(wedge, [O, O + r * _dir(0.0), O + r * _dir(Aa)],
                         [Pt, Pt + r * _dir(PI / 2), Pt + r * _dir(PI / 2 + Aa)],
                         turn=PI / 2), run_time=1.5)
        angP = angle_arc(Pt, Y, Q, radius=r + 0.02, color=GREEN_B, width=4)
        labP = tag("A", 24, GREEN_B).move_to(Pt + 0.86 * _dir(PI / 2 + Aa / 2))
        self.play(Create(angP), FadeIn(labP), wedge.animate.set_fill(opacity=0.3),
                  run_time=0.5)
        self.bring_to_front(OQ, QP, Dot(O))

        segs = ([_seg(O, O + 1.12 * R * RIGHT), _seg(O, O + 1.04 * R * UP),
                 _seg(O, ray_end), _seg(O, Pt), _seg(Q, Pt), _seg(Z, Y),
                 _seg(W, Y)] + _arc_segs(O, R, 0.0, tP)
                + _arc_segs(O, 0.75, 0.0, Aa, 16) + _arc_segs(O, 1.35, tP, Aa, 16)
                + _arc_segs(Pt, r + 0.02, PI / 2, PI / 2 + Aa, 16))
        labels = [labA, labB, lab1, lcB, lsB, labP]
        return dict(S=S, Q=Q, P=Pt, Y=Y, Z=Z, W=W, Qm=Qm, Pm=Pm, segs=segs,
                    labels=labels)


class G8_SineDifference(_DiffRect):
    """sin(A − B) = sinA·cosB − cosA·sinB, read up the sides of the
    rectangle: the left side is the height of Q, sinA·cosB; the right side
    is the same height, made of WP = sin(A−B) and PY = cosA·sinB. Set side
    by side as in the sum formula, the drop from Q to P no longer stacks on
    the height of Q — it overlaps its top, and the overlap is taken away."""

    def construct(self):
        F = self.common()
        O, Q, Pt, Y, Z, W = self.O, F["Q"], F["P"], F["Y"], F["Z"], F["W"]

        OZ = Line(O, Z, color=BLUE_D, stroke_width=10)
        PY = Line(Pt, Y, color=TEAL_D, stroke_width=10)
        WP = Line(W, Pt, color=ORANGE, stroke_width=10)
        lOZ = tag("sinA·cosB", 22, BLUE_B).next_to(OZ, LEFT, buff=0.16)
        lPY = tag("cosA·sinB", 22, TEAL_B).next_to(PY, RIGHT, buff=0.16)
        lWP = tag("sin(A−B)", 22, ORANGE).next_to(WP, LEFT, buff=0.16)
        _labels_ok(F["labels"] + [lOZ, lPY, lWP], F["segs"], "G8 figure")
        self.play(Create(OZ), FadeIn(lOZ), run_time=0.7)
        self.play(Create(PY), FadeIn(lPY), run_time=0.6)
        self.play(Create(WP), FadeIn(lWP), run_time=0.6)
        self.bring_to_front(Dot(O))

        # the three heights side by side: the drop PY hangs from the top of
        # the height of Q, overlapping it; what is not overlapped is WP
        xb = 2.85
        cOZ, cPY, cWP = OZ.copy(), PY.copy(), WP.copy()
        self.play(cOZ.animate.shift(RIGHT * (xb - O[0])),
                  cPY.animate.shift(RIGHT * (xb + 0.22 - Pt[0])),
                  cWP.animate.shift(RIGHT * (xb + 1.15 - Pt[0])), run_time=1.6)
        check(close(cPY.get_end()[1], cOZ.get_end()[1], 1e-6)
              and close(cWP.get_end()[1], cPY.get_start()[1], 1e-6),
              "PY hangs from the top of OZ; WP reaches the bottom of PY")
        yP, yQ, y0 = Pt[1], Q[1], O[1]
        cut = DashedLine([xb - 0.3, yP, 0], [xb + 1.45, yP, 0], color=GREY_A,
                         stroke_width=2, dash_length=0.07)
        top = DashedLine([xb - 0.3, yQ, 0], [xb + 0.45, yQ, 0], color=GREY_B,
                         stroke_width=2, dash_length=0.07)
        base = DashedLine([xb - 0.3, y0, 0], [xb + 1.45, y0, 0], color=GREY_B,
                          stroke_width=2, dash_length=0.07)
        b1 = tag("sinA·cosB", 22, BLUE_B).next_to(cOZ, LEFT, buff=0.36)
        b2 = tag("cosA·sinB", 22, TEAL_B).next_to(cPY, RIGHT, buff=0.2)
        b3 = tag("sin(A−B)", 22, ORANGE).next_to(cWP, RIGHT, buff=0.2)
        bar_segs = ([_seg(cut.get_start(), cut.get_end()),
                     _seg(top.get_start(), top.get_end()),
                     _seg(base.get_start(), base.get_end())]
                    + [_seg(c.get_start(), c.get_end()) for c in (cOZ, cPY, cWP)])
        _labels_ok(F["labels"] + [lOZ, lPY, lWP, b1, b2, b3],
                   F["segs"] + bar_segs, "G8 bars")
        self.play(Create(cut), Create(top), Create(base), FadeIn(b1), FadeIn(b2),
                  FadeIn(b3), run_time=0.8)
        self.play(Write(caption("sin(A−B)  =  sinA·cosB − cosA·sinB", 34)))
        self.hold(2.2)


class G9_CosineDifference(_DiffRect):
    """cos(A − B) = cosA·cosB + sinA·sinB, read across the same rectangle:
    the top side is ZQ + QY = cosA·cosB + sinA·sinB — with B turned back
    the two pieces now add — and the bottom side is OW = cos(A − B)."""

    def construct(self):
        F = self.common()
        O, Q, Pt, Y, Z, W = self.O, F["Q"], F["P"], F["Y"], F["Z"], F["W"]

        ZQ = Line(Z, Q, color=BLUE_D, stroke_width=10)
        QY = Line(Q, Y, color=TEAL_D, stroke_width=10)
        OW = Line(O, W, color=ORANGE, stroke_width=10)
        lZQ = tag("cosA·cosB", 22, BLUE_B).next_to(ZQ, UP, buff=0.16)
        lQY = tag("sinA·sinB", 22, TEAL_B).next_to(QY, UP, buff=0.16)
        lQY.shift(RIGHT * 0.15)              # a little more room from the ray
        check(Q[0] < lQY.get_left()[0] and lQY.get_right()[0] < Y[0],
              "the sinA·sinB label stays over QY")
        lOW = tag("cos(A−B)", 22, ORANGE).next_to(OW, DOWN, buff=0.16)
        _labels_ok(F["labels"] + [lZQ, lQY, lOW], F["segs"], "G9 figure")
        check(_clear(lQY, F["segs"][2:3], 0.15), "sinA·sinB well clear of the ray")
        self.play(Create(ZQ), FadeIn(lZQ), run_time=0.7)
        self.play(Create(QY), FadeIn(lQY), run_time=0.6)
        self.play(Create(OW), FadeIn(lOW), run_time=0.6)
        self.bring_to_front(Dot(O))

        # top side = bottom side, set side by side
        x0 = 1.75
        top_shift = np.array([x0, 1.15, 0.0]) - Z
        bot_shift = np.array([x0, 0.1, 0.0]) - O
        cZQ, cQY, cOW = ZQ.copy(), QY.copy(), OW.copy()
        self.play(cZQ.animate.shift(top_shift), cQY.animate.shift(top_shift),
                  cOW.animate.shift(bot_shift), run_time=1.6)
        check(close(cQY.get_end()[0], cOW.get_end()[0], 1e-6),
              "top side = bottom side")
        xe = cOW.get_end()[0]
        g1 = DashedLine([x0, 1.3, 0], [x0, -0.15, 0], color=GREY_B,
                        stroke_width=2, dash_length=0.07)
        g2 = DashedLine([xe, 1.3, 0], [xe, -0.15, 0], color=GREY_B,
                        stroke_width=2, dash_length=0.07)
        b1 = tag("cosA·cosB", 22, BLUE_B).next_to(cZQ, UP, buff=0.22)
        b2 = tag("sinA·sinB", 22, TEAL_B).next_to(cQY, UP, buff=0.22)
        b3 = tag("cos(A−B)", 22, ORANGE).next_to(cOW, DOWN, buff=0.22)
        bar_segs = ([_seg(g.get_start(), g.get_end()) for g in (g1, g2)]
                    + [_seg(c.get_start(), c.get_end()) for c in (cZQ, cQY, cOW)])
        _labels_ok(F["labels"] + [lZQ, lQY, lOW, b1, b2, b3],
                   F["segs"] + bar_segs, "G9 bars")
        self.play(Create(g1), Create(g2), FadeIn(b1), FadeIn(b2), FadeIn(b3),
                  run_time=0.8)
        self.play(Write(caption("cos(A−B)  =  cosA·cosB + sinA·sinB", 34)))
        self.hold(2.2)


# ================================================== G10  — tangent of a sum

class G10_TangentSum(Board):
    """tan(A + B) from two stacked right triangles. On OD = 1 stands the
    right triangle ODE with angle A at O, so DE = tan A. On its hypotenuse
    stands the right triangle OEF with angle B at O, so EF = tan B · OE.
    The spiral similarity about E that turns EO onto EF — a quarter turn
    and a scale by tan B — carries ODE onto FGE: FG = tan B and
    GE = tanA·tanB. With H the foot of F, HDEG is a rectangle, so the right
    triangle OHF, whose angle at O is A + B, has HF = tan A + tan B and
    OH = 1 − tanA·tanB. (Drawn for A + B < 90°, where H lies on OD.)"""

    def construct(self):
        Aa, Ba = 24 * DEGREES, 35 * DEGREES
        tA, tB = float(np.tan(Aa)), float(np.tan(Ba))
        k = 4.8
        O = np.array([-6.1, -2.35, 0.0])

        def S(p):
            return O + k * to3(p)

        Om, Dm, Em = np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, tA])
        Fm = Em + tB * np.array([-tA, 1.0])        # EF ⊥ OE, EF = tanB·OE
        Gm, Hm = np.array([Fm[0], tA]), np.array([Fm[0], 0.0])

        def quarter(v):                            # clockwise quarter turn
            return np.array([v[1], -v[0]])

        check(close(Em + tB * quarter(Om - Em), Fm)
              and close(Em + tB * quarter(Dm - Em), Gm),
              "the spiral similarity about E carries O to F and D to G")
        check(abs(np.dot(Fm - Em, Em - Om)) < 1e-12, "right angle at E")
        check(close(np.linalg.norm(Fm - Em), tB * np.linalg.norm(Em - Om)),
              "EF = tanB·OE")
        check(close(np.linalg.norm(Fm - Gm), tB)
              and close(np.linalg.norm(Em - Gm), tA * tB),
              "FG = tanB, GE = tanA·tanB")
        check(close(Hm[0], 1 - tA * tB) and close(Fm[1], tA + tB)
              and close(Gm[1] - Hm[1], tA) and close(Dm[0] - Hm[0], tA * tB),
              "OH = 1 − tanA·tanB, HF = tanA + tanB, HG = DE, HD = GE")
        check(0 < Hm[0] < 1, "H lies on OD (A + B < 90°)")
        D, E, F, G, H = S(Dm), S(Em), S(Fm), S(Gm), S(Hm)
        check(abs(_ang(O, D, E) - Aa) < 1e-9 and abs(_ang(O, E, F) - Ba) < 1e-9
              and abs(_ang(O, H, F) - (Aa + Ba)) < 1e-9, "angles A, B, A + B at O")
        check(close(Fm[1] / Hm[0], np.tan(Aa + Ba)), "tan(A+B) = HF / OH")

        # ---- the first triangle: leg 1, angle A
        t1 = mk([O, D, E], BLUE_D, 0.5)
        raD = _ra(D, O, E, 0.2)
        aA = angle_arc(O, D, E, radius=1.0, color=BLUE_B, width=4)
        lA = tag("A", 24, BLUE_B).move_to(O + 1.42 * _dir(Aa / 2))
        l1 = tag("1", 26).move_to(O + 0.56 * (D - O) + 0.36 * UP)
        DEb = Line(D, E, color=BLUE_B, stroke_width=8)
        ltA = _beside(tag("tan A", 24, BLUE_B), D, E, RIGHT, 0.16)
        self.play(FadeIn(t1), Create(raD), run_time=0.8)
        self.play(Create(aA), FadeIn(lA), FadeIn(l1), Create(DEb), FadeIn(ltA),
                  run_time=0.9)

        # ---- the second triangle, on the hypotenuse: angle B
        t2 = mk([O, E, F], ORANGE, 0.4)
        raE = _ra(E, O, F, 0.2)
        aB = angle_arc(O, E, F, radius=1.65, color=ORANGE, width=4)
        lB = tag("B", 24, ORANGE).move_to(O + 2.0 * _dir(Aa + Ba / 2))
        self.play(FadeIn(t2), Create(raE), Create(aB), FadeIn(lB), run_time=1.0)
        self.bring_to_front(t1, DEb, raD)
        self.hold(0.3)

        # ---- turn the first triangle a quarter turn about E, scaled by tan B
        cp = mk([O, D, E], BLUE_D, 0.5)
        tagx = tag("× tan B", 24, ORANGE).move_to(E + np.array([0.95, 0.75, 0.0]))
        self.add(cp)
        self.bring_to_front(aA, lA, l1, DEb, ltA, raD)
        self.play(FadeIn(tagx), _spiral(cp, E, -PI / 2, tB), run_time=2.2)
        check(all(close(v, w, 1e-6) for v, w in zip(cp.get_vertices(), (F, G, E))),
              "the turned copy lands on F, G, E")
        raG = _ra(G, E, F, 0.18)
        FGb = Line(F, G, color=ORANGE, stroke_width=8)
        GEb = Line(G, E, color=RED_C, stroke_width=8)
        ltB = _beside(tag("tan B", 24, ORANGE), G, F, LEFT, 0.16, at=0.3)
        self.play(FadeOut(tagx), Create(raG), Create(FGb), Create(GEb), FadeIn(ltB),
                  run_time=0.9)
        self.bring_to_front(raE)

        # ---- the rectangle HDEG: HG = DE, HD = GE
        dGH = DashedLine(G, H, color=GREY_B, stroke_width=2, dash_length=0.08)
        raH = _ra(H, O, G, 0.18, GREY_B, 2)
        self.play(Create(dGH), Create(raH), run_time=0.6)
        HGb, HDb = DEb.copy(), GEb.copy()
        self.play(HGb.animate.shift(H - D), HDb.animate.shift(D - E), run_time=1.2)
        check(close(HGb.get_start(), H, 1e-6) and close(HGb.get_end(), G, 1e-6)
              and close(HDb.get_start(), H, 1e-6) and close(HDb.get_end(), D, 1e-6),
              "DE slides onto HG, GE onto HD")
        OHb = Line(O, H, color=WHITE, stroke_width=8)
        ltt = tag("tanA·tanB", 22, RED_B).next_to(HDb, DOWN, buff=0.16)
        l1m = tag("1 − tanA·tanB", 22).next_to(OHb, DOWN, buff=0.16)
        self.play(FadeIn(ltt), Create(OHb), FadeIn(l1m), run_time=0.8)
        self.bring_to_front(Dot(O, radius=0.05))

        segs = ([_seg(O, D), _seg(D, E), _seg(O, E), _seg(O, F), _seg(E, F),
                 _seg(F, H), _seg(G, E)]
                + _arc_segs(O, 1.0, 0.0, Aa, 16) + _arc_segs(O, 1.65, Aa, Aa + Ba, 16))
        main_labels = [lA, l1, ltA, lB, ltB, ltt, l1m]
        _labels_ok(main_labels, segs, "G10 figure")

        # ---- the right triangle OHF, angle A + B, set aside: read its tangent
        tri = Polygon(O, H, F, stroke_color=YELLOW_B, stroke_width=4)
        self.play(Create(tri), run_time=0.7)
        self.bring_to_front(OHb, HGb, FGb, raH)
        Op = np.array([0.55, O[1], 0.0])
        sh = Op - O
        big = VGroup(tri.copy(), OHb.copy(), HGb.copy(), FGb.copy(),
                     _ra(H, O, F, 0.2, GREY_B, 2))
        self.play(big.animate.shift(sh), run_time=1.5)
        Hp, Fp, Gp = H + sh, F + sh, G + sh
        aAB = angle_arc(Op, Hp, Fp, radius=1.0, color=YELLOW_B, width=4)
        lAB = tag("A+B", 24, YELLOW_B).move_to(Op + 1.5 * _dir((Aa + Ba) / 2))
        c1 = _beside(tag("tan A", 24, BLUE_B), Hp, Gp, RIGHT, 0.16)
        c2 = _beside(tag("tan B", 24, ORANGE), Gp, Fp, RIGHT, 0.16)
        c3 = tag("1 − tanA·tanB", 22).next_to(Line(Op, Hp), DOWN, buff=0.16)
        segs2 = segs + [_seg(Op, Hp), _seg(Hp, Fp), _seg(Op, Fp)] \
            + _arc_segs(Op, 1.0, 0.0, Aa + Ba, 16)
        _labels_ok(main_labels + [lAB, c1, c2, c3], segs2, "G10 copy")
        self.play(Create(aAB), FadeIn(lAB), FadeIn(c1), FadeIn(c2), FadeIn(c3),
                  run_time=0.9)
        self.play(Write(caption(
            "tan(A+B)  =  (tanA + tanB) / (1 − tanA·tanB)", 34)))
        self.hold(2.2)


# ============================================ G11  — cosine of a double angle

def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments (for label checks)."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


def _tick_segs(p, q, at=0.5, size=0.13):
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    return [(c - nrm * size, c + nrm * size)]


class G11_CosineDoubleAngle(Board):
    """cos 2θ = 2cos²θ − 1 = 1 − 2sin²θ in the semicircle on VE (radius 1).
    P at angle 2θ: the isosceles triangle VOP has equal base angles, which
    together fill the angle 2θ at O, so ∠PVE = θ. Folding VOP and EOP
    about their axes OM, ON halves the chords: VP = 2cos θ (from VMO, with
    hypotenuse 1 and angle θ at V) and EP = 2 sin θ (from ENO, angle θ at
    O). The foot F of P cuts the diameter into VF = 1 + cos 2θ and
    FE = 1 − cos 2θ. Flipped over the bisector at V and enlarged by 2cos θ,
    VMO lands exactly on VFP: VF = 2cos θ·cos θ; flipped over the bisector
    at E and enlarged by 2 sin θ, ENO lands on EFP: FE = 2 sin θ·sin θ."""

    def construct(self):
        th = 33 * DEGREES
        c, s = float(np.cos(th)), float(np.sin(th))
        R = 3.55
        O = np.array([0.0, -0.85, 0.0])

        def S(p):
            return O + R * to3(p)

        V, E = S((-1.0, 0.0)), S((1.0, 0.0))
        P = S((np.cos(2 * th), np.sin(2 * th)))
        F = S((np.cos(2 * th), 0.0))
        M, N, K = (V + P) / 2, (P + E) / 2, (O + P) / 2
        check(abs(_ang(P, V, O) - th) < 1e-9 and abs(_ang(V, P, O) - th) < 1e-9,
              "isosceles VOP: base angles θ, θ")
        check(abs(_ang(V, E, P) - th) < 1e-9, "∠PVE = θ (half of 2θ)")
        check(abs(np.dot(M - O, P - V)) < 1e-9 and abs(np.dot(N - O, P - E)) < 1e-9,
              "OM ⊥ VP and ON ⊥ PE at the midpoints")
        check(close(_refl(V, O, M - O), P) and close(_refl(E, O, N - O), P),
              "folding over OM (ON) carries V (E) onto P")
        check(close(np.linalg.norm(M - V), R * c) and close(np.linalg.norm(N - E), R * s)
              and abs(_ang(O, E, N) - th) < 1e-9, "VM = cos θ, EN = sin θ, ∠EON = θ")
        check(close(np.linalg.norm(F - V), R * (1 + np.cos(2 * th)))
              and close(np.linalg.norm(F - V), R * 2 * c * c)
              and close(np.linalg.norm(E - F), R * (1 - np.cos(2 * th)))
              and close(np.linalg.norm(E - F), R * 2 * s * s),
              "VF = 1 + cos 2θ = 2cos²θ, FE = 1 − cos 2θ = 2sin²θ")
        uV = _unit(_unit(E - V) + _unit(P - V))          # bisector at V
        uE = _unit(_unit(V - E) + _unit(P - E))          # bisector at E

        def land(X, C, u, f):
            return C + f * (_refl(X, C, u) - C)
        check(close(land(M, V, uV, 2 * c), F) and close(land(O, V, uV, 2 * c), P),
              "VMO, flipped at V and enlarged by 2cos θ, is VFP")
        check(close(land(N, E, uE, 2 * s), F) and close(land(O, E, uE, 2 * s), P),
              "ENO, flipped at E and enlarged by 2 sin θ, is EFP")

        # ---- the semicircle and P at angle 2θ
        arc = Arc(radius=R, start_angle=0, angle=PI, arc_center=O,
                  color=GREY_B, stroke_width=2.5)
        diam = Line(V, E, color=GREY_A, stroke_width=3)
        dotO = Dot(O, radius=0.06)
        self.play(Create(arc), Create(diam), FadeIn(dotO), run_time=1.0)
        OP = Line(O, P, color=WHITE, stroke_width=3)
        a2 = angle_arc(O, E, P, radius=0.6, color=YELLOW_B, width=4)
        l2 = tag("2θ", 24, YELLOW_B).move_to(O + 1.0 * _dir(th))
        self.play(Create(OP), Create(a2), FadeIn(l2), run_time=0.9)

        # ---- isosceles VOP: its equal base angles fill the angle 2θ at O
        VP = Line(V, P, color=WHITE, stroke_width=3)
        tVOP = mk([V, O, P], BLUE_E, 0.35, stroke_width=0)
        tk_r = VGroup(_ticks(V, O), _ticks(O, P))
        aV = angle_arc(V, E, P, radius=0.85, color=BLUE_B, width=4)
        aP = angle_arc(P, V, O, radius=0.85, color=BLUE_B, width=4)
        self.play(FadeIn(tVOP), Create(VP), FadeIn(tk_r), run_time=0.8)
        self.play(Create(aV), Create(aP), run_time=0.6)
        wV = _wedge(V, E, P, 0.5, BLUE_B, 0.65)
        wP = _wedge(P, V, O, 0.5, BLUE_B, 0.65)
        self.play(FadeIn(wV), FadeIn(wP), run_time=0.3)
        self.play(wV.animate.shift(O - V), Rotate(wP, angle=PI, about_point=K),
                  run_time=1.6)
        check(close(2 * K - P, O), "the half-turn about the midpoint of OP takes P to O")
        lV = tag("θ", 26, BLUE_B).move_to(V + 1.2 * _dir(th / 2))
        self.play(FadeIn(lV), FadeOut(wV), FadeOut(wP), FadeOut(aP), run_time=0.6)

        # ---- fold VOP about its axis OM: VM = MP = cos θ
        OM = DashedLine(O, M, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        self.play(Create(OM), run_time=0.5)
        half = mk([V, M, O], BLUE_D, 0.6)
        self.add(half)
        self.play(Rotate(half, angle=PI, axis=_unit(M - O), about_point=O),
                  run_time=1.3)
        raM = _ra(M, V, O, 0.18)
        tk_c = VGroup(_ticks(V, M, 2, BLUE_B), _ticks(M, P, 2, BLUE_B))
        lcos = _beside(tag("cos θ", 24, BLUE_B), V, M, _dir(th + PI / 2), 0.28)
        self.play(FadeOut(half), Create(raM), FadeIn(tk_c), FadeIn(lcos), run_time=0.7)

        # ---- isosceles EOP, folded about its axis ON: EN = NP = sin θ
        PE = Line(P, E, color=WHITE, stroke_width=3)
        tEOP = mk([E, O, P], TEAL_E, 0.35, stroke_width=0)
        tk_e = _ticks(O, E, at=0.72)
        self.play(FadeIn(tEOP), Create(PE), FadeIn(tk_e), run_time=0.7)
        ON = DashedLine(O, N, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        self.play(Create(ON), FadeOut(l2), run_time=0.5)
        halfE = mk([E, N, O], TEAL_D, 0.6)
        self.add(halfE)
        self.play(Rotate(halfE, angle=PI, axis=_unit(N - O), about_point=O),
                  run_time=1.3)
        raN = _ra(N, E, O, 0.18)
        tk_s = VGroup(_ticks(E, N, 2, TEAL_B), _ticks(N, P, 2, TEAL_B))
        lsin = _beside(tag("sin θ", 24, TEAL_B), E, N, O - N, 0.28)
        lt1 = tag("θ", 24, YELLOW_B).move_to(O + 1.0 * _dir(th / 2))
        lt2 = tag("θ", 24, YELLOW_B).move_to(O + 1.0 * _dir(1.5 * th))
        self.play(FadeOut(halfE), Create(raN), FadeIn(tk_s), FadeIn(lsin),
                  FadeIn(lt1), FadeIn(lt2), run_time=0.7)

        # ---- the foot F of P cuts the diameter into 1 + cos 2θ and 1 − cos 2θ
        PF = DashedLine(P, F, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        raF = _ra(F, E, P, 0.18)
        OF = Line(O, F, color=ORANGE, stroke_width=7)
        l1 = tag("1", 24).next_to(Line(V, O), DOWN, buff=0.24)
        lc2 = tag("cos 2θ", 24, ORANGE).next_to(OF, DOWN, buff=0.24)
        self.play(tVOP.animate.set_fill(opacity=0), tEOP.animate.set_fill(opacity=0),
                  Create(PF), Create(raF), run_time=0.7)
        self.play(Create(OF), FadeIn(l1), FadeIn(lc2), run_time=0.6)
        yb = -1.72
        bVF = Line([V[0], yb, 0], [F[0], yb, 0], color=BLUE_D, stroke_width=10)
        bFE = Line([F[0], yb, 0], [E[0], yb, 0], color=TEAL_D, stroke_width=10)
        guides = VGroup(*[DashedLine(X, [X[0], yb - 0.1, 0], color=GREY_B,
                                     stroke_width=1.5, dash_length=0.07)
                          for X in (V, F, E)])
        r2a = tag("1 + cos 2θ", 24, BLUE_B).next_to(bVF, DOWN, buff=0.16)
        r2b = tag("1 − cos 2θ", 24, TEAL_B).next_to(bFE, DOWN, buff=0.16)
        self.play(Create(guides), Create(bVF), Create(bFE), FadeIn(r2a),
                  FadeIn(r2b), run_time=0.9)
        self.bring_to_front(dotO)

        segs = (_arc_segs(O, R, 0.0, PI, 72)
                + [_seg(V, E), _seg(O, P), _seg(V, P), _seg(P, E), _seg(O, M),
                   _seg(O, N), _seg(P, F)]
                + _arc_segs(O, 0.6, 0.0, 2 * th, 16) + _arc_segs(V, 0.85, 0.0, th, 12)
                + _ra_segs(M, V, O, 0.18) + _ra_segs(N, E, O, 0.18)
                + _ra_segs(F, E, P, 0.18)
                + _tick_segs(V, O) + _tick_segs(O, P) + _tick_segs(O, E, 0.72)
                + _tick_segs(V, M) + _tick_segs(M, P) + _tick_segs(E, N)
                + _tick_segs(N, P)
                + [_seg(bVF.get_start(), bVF.get_end()),
                   _seg(bFE.get_start(), bFE.get_end())]
                + [_seg(g.get_start(), g.get_end()) for g in guides])
        labels = [lV, lcos, lsin, lt1, lt2, l1, lc2, r2a, r2b]
        _labels_ok(labels, segs, "G11 figure")

        # ---- VMO, flipped over the bisector at V and enlarged by 2cos θ, is VFP
        fixed = [aV, lV, lt1, lt2, lsin, a2, raF, PF, OF, dotO]
        cV = mk([V, M, O], BLUE_D, 0.55)
        self.play(FadeIn(cV), run_time=0.4)
        self.bring_to_front(*fixed)
        tagV = tag("× 2 cos θ", 24, BLUE_B).move_to(np.array([-4.75, -0.2, 0.0]))
        self.play(Rotate(cV, angle=PI, axis=uV, about_point=V), FadeIn(tagV),
                  run_time=1.2)
        self.play(cV.animate.scale(2 * c, about_point=V), run_time=1.3)
        check(all(close(a, b, 1e-6) for a, b in zip(cV.get_vertices(), (V, F, P))),
              "the enlarged copy lands on V, F, P")
        self.bring_to_front(*fixed)
        r3a = tag("= 2 cos θ · cos θ", 24, BLUE_B).next_to(r2a, DOWN, buff=0.14)
        self.play(FadeIn(r3a), FadeOut(tagV), run_time=0.6)

        # ---- ENO, flipped over the bisector at E and enlarged by 2 sin θ, is EFP
        cE = mk([E, N, O], TEAL_D, 0.55)
        self.play(FadeIn(cE), run_time=0.4)
        self.bring_to_front(*fixed)
        tagE = tag("× 2 sin θ", 24, TEAL_B).move_to(np.array([4.75, -0.2, 0.0]))
        self.play(Rotate(cE, angle=PI, axis=uE, about_point=E), FadeIn(tagE),
                  run_time=1.2)
        self.play(cE.animate.scale(2 * s, about_point=E), run_time=1.3)
        check(all(close(a, b, 1e-6) for a, b in zip(cE.get_vertices(), (E, F, P))),
              "the enlarged copy lands on E, F, P")
        self.bring_to_front(*fixed)
        r3b = tag("= 2 sin θ · sin θ", 24, TEAL_B).next_to(r2b, DOWN, buff=0.14)
        _labels_ok(labels + [r3a, r3b, tagV, tagE], segs, "G11 rows")
        self.play(FadeIn(r3b), FadeOut(tagE), run_time=0.6)
        self.play(Write(caption("cos 2θ  =  2cos²θ − 1  =  1 − 2sin²θ", 34)))
        self.hold(2.2)


# ============================================ G14  — the other Pythagorean identities

class G14_TangentSecant(Board):
    """1 + tan²θ = sec²θ and 1 + cot²θ = csc²θ. The unit circle's right
    triangle (legs cos θ, sin θ, hypotenuse 1) enlarged from O by 1/cos θ
    reaches the tangent at (1, 0): legs 1 and tan θ, hypotenuse sec θ.
    Enlarged by 1/sin θ instead, its far corner reaches the tangent at
    (0, 1): legs cot θ and 1, hypotenuse csc θ. Pythagoras in each."""

    def construct(self):
        th = 38 * DEGREES
        c, s = float(np.cos(th)), float(np.sin(th))
        k = 3.8
        O1, O2 = np.array([-6.1, -2.4, 0.0]), np.array([0.55, -2.4, 0.0])

        def tri_pts(O, f):
            return [O + k * f * to3((c, 0.0)), O, O + k * f * to3((c, s))]

        # [right-angle corner, O, far corner]
        U1, U2 = tri_pts(O1, 1.0), tri_pts(O2, 1.0)
        B1, B2 = tri_pts(O1, 1 / c), tri_pts(O2, 1 / s)
        check(close(B1[0], O1 + k * RIGHT) and close(B1[2], O1 + k * to3((1, np.tan(th)))),
              "÷ cos θ: the corner lands on (1, 0), the far end on the tangent x = 1")
        check(close(B2[2], O2 + k * to3((1 / np.tan(th), 1.0))),
              "÷ sin θ: the far end lands on the tangent y = 1 at (cot θ, 1)")
        check(close(np.linalg.norm(B1[2] - O1), k / c) and close(np.linalg.norm(B2[2] - O2), k / s),
              "hypotenuses sec θ and csc θ")
        check(close(1 + np.tan(th) ** 2, 1 / c ** 2)
              and close(1 + 1 / np.tan(th) ** 2, 1 / s ** 2), "the identities")

        def panel(O, top_tangent):
            xax = Line(O, O + k * 1.42 * RIGHT if top_tangent else O + k * 1.15 * RIGHT,
                       color=GREY_D, stroke_width=2)
            yax = Line(O, O + k * 1.1 * UP, color=GREY_D, stroke_width=2)
            arc = Arc(radius=k, start_angle=0, angle=PI / 2, arc_center=O,
                      color=GREY_B, stroke_width=2.5)
            return VGroup(xax, yax, arc)

        def triangle(pts):
            H, O, P = pts
            return VGroup(mk([H, O, P], BLUE_D, 0.5, stroke_width=0),
                          Line(O, H, color=ORANGE, stroke_width=7),
                          Line(H, P, color=PURPLE_B, stroke_width=7),
                          Line(O, P, color=YELLOW_B, stroke_width=5),
                          _ra(H, O, P, 0.2))

        def labels(pts, a, b, h):
            H, O, P = pts
            return VGroup(
                tag(a, 24, ORANGE).next_to(Line(O, H), DOWN, buff=0.18),
                _beside(tag(b, 24, PURPLE_A), H, P, (O - H), 0.16),
                _beside(tag(h, 24, YELLOW_B), O, P, _dir(th + PI / 2), 0.14))

        def labels_out(pts, a, b, h, at=0.5):
            H, O, P = pts
            return VGroup(
                tag(a, 24, ORANGE).next_to(Line(O, H), DOWN, buff=0.18),
                _beside(tag(b, 24, PURPLE_A), H, P, (H - O), 0.16),
                _beside(tag(h, 24, YELLOW_B), O, P, _dir(th + PI / 2), 0.14, at))

        pan1, pan2 = panel(O1, False), panel(O2, True)
        t1, t2 = triangle(U1), triangle(U2)
        L1, L2 = labels(U1, "cos θ", "sin θ", "1"), labels(U2, "cos θ", "sin θ", "1")
        arcs = VGroup(*[angle_arc(O, O + RIGHT, P, radius=0.75, color=WHITE, width=3)
                        for O, P in ((O1, U1[2]), (O2, U2[2]))])
        ths = VGroup(*[tag("θ", 24).move_to(O + 1.05 * _dir(th / 2)) for O in (O1, O2)])
        unit_id = tag("cos²θ + sin²θ = 1", 28).move_to([-0.3, 3.35, 0.0])
        self.play(Create(pan1), Create(pan2), run_time=1.0)
        self.play(FadeIn(t1), FadeIn(t2), Create(arcs), FadeIn(ths), run_time=0.9)
        self.play(FadeIn(L1), FadeIn(L2), FadeIn(unit_id), run_time=0.7)
        self.hold(0.4)

        # ---- panel 1: every side ÷ cos θ — the triangle grows onto the tangent x = 1
        tan1 = DashedLine(O1 + k * RIGHT, O1 + k * to3((1.0, 1.08)), color=GREY_B,
                          stroke_width=2, dash_length=0.08)
        f1 = tag("÷ cos θ", 26, WHITE).move_to(O1 + k * to3((0.55, 1.2)))
        ghost1 = DashedVMobject(Polygon(*U1, stroke_color=GREY_A, stroke_width=2),
                                num_dashes=40)
        self.play(Create(tan1), FadeIn(f1), run_time=0.6)
        self.add(ghost1)
        N1 = labels_out(B1, "1", "tan θ", "sec θ")
        self.play(FadeOut(L1), run_time=0.3)
        self.play(t1.animate.scale(1 / c, about_point=O1), run_time=1.6)
        check(close(t1[2].get_end(), B1[2], 1e-6), "the enlarged triangle reaches the tangent")
        self.bring_to_front(ghost1, arcs[0], ths[0])
        self.play(FadeIn(N1), run_time=0.5)
        id1 = tag("1 + tan²θ = sec²θ", 28).move_to([O1[0] + 2.1, 2.6, 0.0])
        self.play(FadeOut(f1), FadeIn(id1, shift=0.2 * DOWN), run_time=0.8)

        # ---- panel 2: every side ÷ sin θ — the far corner reaches the tangent y = 1
        tan2 = DashedLine(O2 + k * UP, O2 + k * to3((1.45, 1.0)), color=GREY_B,
                          stroke_width=2, dash_length=0.08)
        f2 = tag("÷ sin θ", 26, WHITE).move_to(O2 + k * to3((0.32, 1.22)))
        ghost2 = DashedVMobject(Polygon(*U2, stroke_color=GREY_A, stroke_width=2),
                                num_dashes=40)
        self.play(Create(tan2), FadeIn(f2), run_time=0.6)
        self.add(ghost2)
        N2 = labels_out(B2, "cot θ", "1", "csc θ", at=0.4)
        self.play(FadeOut(L2), run_time=0.3)
        self.play(t2.animate.scale(1 / s, about_point=O2), run_time=1.6)
        check(close(t2[2].get_end(), B2[2], 1e-6), "the enlarged triangle reaches the tangent")
        self.bring_to_front(ghost2, arcs[1], ths[1])
        self.play(FadeIn(N2), run_time=0.5)
        id2 = tag("cot²θ + 1 = csc²θ", 28).move_to([O2[0] + 2.7, 2.6, 0.0])
        segs = [_seg(O1, O1 + k * 1.15 * RIGHT), _seg(O1, O1 + k * 1.1 * UP),
                _seg(O2, O2 + k * 1.42 * RIGHT), _seg(O2, O2 + k * 1.1 * UP),
                _seg(tan1.get_start(), tan1.get_end()), _seg(tan2.get_start(), tan2.get_end())]
        for O, B in ((O1, B1), (O2, B2)):
            segs += [_seg(B[1], B[0]), _seg(B[0], B[2]), _seg(B[1], B[2])]
            segs += _arc_segs(O, k, 0.0, PI / 2, 40) + _arc_segs(O, 0.75, 0.0, th, 10)
        _labels_ok(list(N1) + list(N2) + list(ths) + [f2, unit_id, id1],
                   segs, "G14 while the tag shows")
        _labels_ok(list(N1) + list(N2) + list(ths) + [unit_id, id1, id2],
                   segs, "G14 final")
        self.play(FadeOut(f2), FadeIn(id2, shift=0.2 * DOWN), run_time=0.8)
        self.play(Write(caption("1 + tan²θ  =  sec²θ          1 + cot²θ  =  csc²θ", 34)))
        self.hold(2.2)


# ============================================ G21  — law of sines by the altitude

class G21_SinesByAltitude(Board):
    """a/sin A = b/sin B. The altitude from C splits the triangle into two
    right triangles that share it. The right triangle with hypotenuse 1 and
    angle A (height sin A), enlarged from A by b, is the left half: so the
    altitude is b·sin A. The same with angle B, enlarged from B by a, is the
    right half: the altitude is a·sin B. (Drawn with A and B acute, so the
    foot lies on AB; for an obtuse angle the foot falls outside and
    sin(180° − A) = sin A gives the same.)"""

    def construct(self):
        A = np.array([-4.6, -2.2, 0.0])
        B = np.array([3.6, -2.2, 0.0])
        C = np.array([-0.9, 2.7, 0.0])
        H = np.array([C[0], A[1], 0.0])
        a, b = float(np.linalg.norm(C - B)), float(np.linalg.norm(C - A))
        angA, angB = _ang(A, B, C), _ang(B, A, C)
        check(angA < PI / 2 and angB < PI / 2 and A[0] < H[0] < B[0],
              "A, B acute: the foot of the altitude lies on AB")
        check(close(C[1] - H[1], b * np.sin(angA)) and close(C[1] - H[1], a * np.sin(angB)),
              "the altitude is b·sin A and a·sin B")
        check(close(a / np.sin(angA), b / np.sin(angB)), "a/sin A = b/sin B")
        # unit right triangles at A and at B (unit u screen units; a, b in units)
        u = 1.5
        a, b = a / u, b / u
        UA = A + u * _dir(angA)
        UA_ = np.array([UA[0], A[1], 0.0])
        UB = B + u * _dir(PI - angB)
        UB_ = np.array([UB[0], B[1], 0.0])
        check(close(A + b * (UA - A), C) and close(A + b * (UA_ - A), H),
              "the unit triangle at A, enlarged by b from A, is AHC")
        check(close(B + a * (UB - B), C) and close(B + a * (UB_ - B), H),
              "the unit triangle at B, enlarged by a from B, is BHC")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3)
        arcA = angle_arc(A, B, C, radius=0.55, color=TEAL_B, width=4)
        arcB = angle_arc(B, A, C, radius=0.55, color=ORANGE, width=4)
        vA = tag("A", 28, TEAL_B).next_to(A, LEFT, buff=0.18)
        vB = tag("B", 28, ORANGE).next_to(B, RIGHT, buff=0.18)
        vC = tag("C", 28).next_to(C, UP, buff=0.16)
        lb = _beside(tag("b", 28), A, C, A - B, 0.16)
        la = _beside(tag("a", 28), B, C, B - A, 0.16)
        lc = tag("c", 28).next_to(Line(A, B), DOWN, buff=0.18).shift(1.2 * RIGHT)
        self.play(Create(tri), run_time=1.0)
        self.play(Create(arcA), Create(arcB), FadeIn(vA), FadeIn(vB), FadeIn(vC),
                  FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=0.8)
        alt = DashedLine(C, H, color=GREY_A, stroke_width=3, dash_length=0.1)
        raH = _ra(H, B, C, 0.22)
        self.play(Create(alt), Create(raH), run_time=0.8)
        self.hold(0.3)

        def unit_tri(V, U, U_, col, op):
            return VGroup(mk([V, U_, U], col, op, stroke_width=0),
                          Line(V, U, color=WHITE, stroke_width=4),
                          Line(U_, U, color=col, stroke_width=7))

        # ---- left: the unit triangle with angle A, enlarged by b
        tA = unit_tri(A, UA, UA_, TEAL_D, 0.55)
        l1A = _beside(tag("1", 24), A, UA, A - B, 0.12)
        lsA = _beside(tag("sin A", 24, TEAL_B), UA_, UA, B - A, 0.14)
        self.play(FadeIn(tA), FadeIn(l1A), FadeIn(lsA), run_time=0.8)
        self.hold(0.8)
        xb = tag("× b", 28, TEAL_B).move_to(np.array([-4.9, 1.0, 0.0]))
        self.play(FadeIn(xb), run_time=0.4)
        self.play(tA.animate.scale(b, about_point=A), FadeOut(l1A), FadeOut(lsA),
                  run_time=1.6)
        check(close(tA[2].get_start(), H, 1e-6) and close(tA[2].get_end(), C, 1e-6),
              "the enlarged height is the altitude")
        lbs = tag("b·sin A", 26, TEAL_B).next_to(Line(H, C), LEFT, buff=0.18)
        self.bring_to_front(arcA, raH)
        self.play(FadeIn(lbs), FadeOut(xb), run_time=0.6)

        # ---- right: the unit triangle with angle B, enlarged by a
        tB = unit_tri(B, UB, UB_, ORANGE, 0.45)
        l1B = _beside(tag("1", 24), B, UB, B - A, 0.12)
        lsB = _beside(tag("sin B", 24, ORANGE), UB_, UB, A - B, 0.14)
        self.play(FadeIn(tB), FadeIn(l1B), FadeIn(lsB), run_time=0.8)
        self.hold(0.8)
        xa = tag("× a", 28, ORANGE).move_to(np.array([3.4, 1.0, 0.0]))
        self.play(FadeIn(xa), run_time=0.4)
        self.play(tB.animate.scale(a, about_point=B), FadeOut(l1B), FadeOut(lsB),
                  run_time=1.6)
        check(close(tB[2].get_start(), H, 1e-6) and close(tB[2].get_end(), C, 1e-6),
              "the enlarged height is the altitude")
        las = tag("a·sin B", 26, ORANGE).next_to(Line(H, C), RIGHT, buff=0.18)
        self.bring_to_front(arcB, raH, lbs)
        self.play(FadeIn(las), FadeOut(xa), run_time=0.6)

        # one segment, two names
        hl = Line(H, C, color=YELLOW_B, stroke_width=6)
        self.play(Create(hl), run_time=0.6)
        self.bring_to_front(raH)
        segs = [_seg(A, B), _seg(B, C), _seg(C, A), _seg(C, H)] \
            + _arc_segs(A, 0.55, 0.0, angA, 12) + _arc_segs(B, 0.55, PI - angB, PI, 12) \
            + _ra_segs(H, B, C, 0.22)
        _labels_ok([vA, vB, vC, la, lb, lc, lbs, las], segs, "G21 final")
        _labels_ok([vA, vB, vC, la, lb, lc, l1A, lsA, xb],
                   segs + [_seg(UA, UA_)], "G21 unit triangle A")
        _labels_ok([vA, vB, vC, la, lb, lc, lbs, l1B, lsB, xa],
                   segs + [_seg(UB, UB_)], "G21 unit triangle B")
        self.play(Write(caption("b·sin A  =  a·sin B     ⟹     a / sin A  =  b / sin B", 32)))
        self.hold(2.2)


# ============================================ G22  — exact values at 30°, 45°, 60°

class G22_SpecialAngles(Board):
    """sin 30° = ½, sin 45° = √2/2, sin 60° = √3/2 from half an equilateral
    triangle and half a square, each with hypotenuse 1. Folding the
    equilateral triangle (side 1, angles 60°) on its axis shows the axis
    halves the base and the apex angle: legs ½ and √(1 − ¼) = √3/2 opposite
    30° and 60°. Folding the square on its diagonal 1 shows the diagonal
    halves the right angles: 45°, and legs x with x² + x² = 1, x = √2/2.
    With hypotenuse 1, each sine is the side opposite its angle."""

    def construct(self):
        s = 5.2                                    # the hypotenuse 1, on screen
        A = np.array([-6.2, -2.3, 0.0])
        B = A + s * RIGHT
        C = A + s * _dir(PI / 3)
        M = (A + B) / 2
        S0 = np.array([1.2, -2.3, 0.0])
        e = s / np.sqrt(2)
        S1, S2, S3 = S0 + e * RIGHT, S0 + e * (RIGHT + UP), S0 + e * UP
        check(close(np.linalg.norm(C - A), s) and close(np.linalg.norm(C - B), s),
              "equilateral, side 1")
        check(close(_refl(B, C, M - C), A), "folding on the axis CM carries B onto A")
        check(close(np.linalg.norm(M - A), s / 2) and abs(_ang(C, A, M) - PI / 6) < 1e-9
              and abs(_ang(A, M, C) - PI / 3) < 1e-9 and abs(_ang(M, A, C) - PI / 2) < 1e-9,
              "AM = ½, apex half-angle 30°, angle at A 60°, right angle at M")
        check(close(np.linalg.norm(C - M), s * np.sqrt(3) / 2)
              and close((s / 2) ** 2 + np.linalg.norm(C - M) ** 2, s ** 2),
              "CM = √(1 − ¼) = √3/2")
        check(close(np.linalg.norm(S2 - S0), s) and close(_refl(S3, S0, S2 - S0), S1),
              "diagonal 1; folding on it carries S3 onto S1")
        check(abs(_ang(S0, S1, S2) - PI / 4) < 1e-9 and abs(_ang(S2, S1, S0) - PI / 4) < 1e-9
              and close(2 * e ** 2, s ** 2) and close(e, s * np.sqrt(2) / 2),
              "45°, 45°; x² + x² = 1, x = √2/2")

        # ---- the equilateral triangle: sides 1, angles 60°
        tri = mk([A, B, C], BLUE_E, 0.4)
        sides = VGroup(_beside(tag("1", 26), A, C, A - B, 0.14),
                       _beside(tag("1", 26), B, C, B - A, 0.14),
                       tag("1", 26).next_to(Line(A, B), DOWN, buff=0.18))
        a60 = VGroup(*[angle_arc(V, P, Q, radius=0.55, color=TEAL_B, width=4)
                       for V, P, Q in ((A, B, C), (B, C, A), (C, A, B))])
        l60 = VGroup(*[tag("60°", 22, TEAL_B).move_to(V + 0.98 * angle_mid_dir(V, P, Q))
                       for V, P, Q in ((A, B, C), (B, C, A), (C, A, B))])
        self.play(FadeIn(tri), run_time=0.8)
        self.play(FadeIn(sides), Create(a60), FadeIn(l60), run_time=0.9)

        # ---- fold on the axis from C: the halves coincide
        axis = DashedLine(C, M, color=GREY_A, stroke_width=2.5, dash_length=0.09)
        self.play(Create(axis), run_time=0.6)
        rhalf = mk([M, B, C], BLUE_D, 0.6)
        self.add(rhalf)
        self.play(Rotate(rhalf, angle=PI, axis=UP, about_point=C), run_time=1.4)
        check(all(close(v, w, 1e-6) for v, w in zip(rhalf.get_vertices(), (M, A, C))),
              "the folded half lands on the other half")
        left = mk([A, M, C], BLUE_D, 0.55)
        raM = _ra(M, A, C, 0.2)
        h1 = tag("½", 28, ORANGE).next_to(Line(A, M), DOWN, buff=0.16)
        h2 = tag("½", 28, ORANGE).next_to(Line(M, B), DOWN, buff=0.16)
        a30 = angle_arc(C, A, M, radius=0.9, color=ORANGE, width=4)
        l30 = tag("30°", 22, ORANGE).move_to(C + 1.55 * angle_mid_dir(C, A, M))
        self.add(left)
        self.remove(rhalf)
        self.play(FadeOut(sides[2]), FadeOut(a60[2]), FadeOut(l60[2]), FadeOut(a60[1]),
                  FadeOut(l60[1]), tri.animate.set_fill(opacity=0.15),
                  FadeIn(h1), FadeIn(h2), Create(raM), Create(a30), FadeIn(l30),
                  run_time=0.9)
        self.bring_to_front(axis, a60[0], l60[0], sides[0])

        # ---- the long leg: √(1 − ¼) = √3/2
        CMb = Line(M, C, color=TEAL_B, stroke_width=7)
        AMb = Line(A, M, color=ORANGE, stroke_width=7)
        r1 = _beside(tag("√(1 − ¼)", 24, TEAL_B), M, C, RIGHT, 0.16, at=0.3)
        r2 = _beside(tag("√3/2", 26, TEAL_B), M, C, RIGHT, 0.16, at=0.3)
        tri_segs = [_seg(A, B), _seg(B, C), _seg(C, A), _seg(C, M)]
        _labels_ok([r1, sides[0], sides[1], l60[0], h1, h2, l30], tri_segs, "G22 root")
        self.play(Create(CMb), Create(AMb), FadeIn(r1), run_time=0.8)
        self.hold(0.4)
        self.play(ReplacementTransform(r1, r2), run_time=0.6)
        self.bring_to_front(raM)

        # ---- the square with diagonal 1, folded on the diagonal
        sq = mk([S0, S1, S2, S3], GREEN_E, 0.35)
        diag = Line(S0, S2, color=WHITE, stroke_width=3)
        l1d = _beside(tag("1", 26), S0, S2, S3 - S1, 0.14)
        self.play(FadeIn(sq), Create(diag), FadeIn(l1d), run_time=0.9)
        uhalf = mk([S0, S3, S2], GREEN_D, 0.6)
        self.add(uhalf)
        self.play(Rotate(uhalf, angle=PI, axis=_unit(S2 - S0), about_point=S0),
                  run_time=1.4)
        check(all(close(v, w, 1e-6) for v, w in zip(uhalf.get_vertices(), (S0, S1, S2))),
              "the folded half lands on the other half")
        low = mk([S0, S1, S2], GREEN_D, 0.55)
        self.add(low)
        self.remove(uhalf)
        self.bring_to_front(diag)
        raS = _ra(S1, S0, S2, 0.2)
        a45 = VGroup(angle_arc(S0, S1, S2, radius=0.75, color=YELLOW_B, width=4),
                     angle_arc(S2, S1, S0, radius=0.75, color=YELLOW_B, width=4))
        l45 = VGroup(tag("45°", 22, YELLOW_B).move_to(S0 + 1.25 * angle_mid_dir(S0, S1, S2)),
                     tag("45°", 22, YELLOW_B).move_to(S2 + 1.25 * angle_mid_dir(S2, S1, S0)))
        legs = VGroup(Line(S0, S1, color=YELLOW_B, stroke_width=7),
                      Line(S1, S2, color=YELLOW_B, stroke_width=7))
        q1 = VGroup(tag("√½", 26, YELLOW_B).next_to(legs[0], DOWN, buff=0.16),
                    tag("√½", 26, YELLOW_B).next_to(legs[1], RIGHT, buff=0.16))
        q2 = VGroup(tag("√2/2", 26, YELLOW_B).next_to(legs[0], DOWN, buff=0.16),
                    tag("√2/2", 26, YELLOW_B).next_to(legs[1], RIGHT, buff=0.16))
        sq_segs = [_seg(S0, S1), _seg(S1, S2), _seg(S2, S3), _seg(S3, S0), _seg(S0, S2)]
        _labels_ok([*q1, *l45, l1d], sq_segs, "G22 half-root")
        self.play(sq.animate.set_fill(opacity=0.12), Create(raS), Create(a45),
                  FadeIn(l45), Create(legs), FadeIn(q1), run_time=0.9)
        self.hold(0.4)
        self.play(ReplacementTransform(q1, q2), run_time=0.6)

        segs = [_seg(A, B), _seg(B, C), _seg(C, A), _seg(C, M), _seg(S0, S1),
                _seg(S1, S2), _seg(S2, S3), _seg(S3, S0), _seg(S0, S2)] \
            + _arc_segs(A, 0.55, 0.0, PI / 3, 12) + _ra_segs(M, A, C, 0.2) \
            + _arc_segs(C, 0.9, 4 * PI / 3, 3 * PI / 2, 10) \
            + _arc_segs(S0, 0.75, 0.0, PI / 4, 10) \
            + _arc_segs(S2, 0.75, 5 * PI / 4, 3 * PI / 2, 10) + _ra_segs(S1, S0, S2, 0.2)
        labels = [sides[0], sides[1], l60[0], h1, h2, l30, r2, l1d, *l45, *q2]
        _labels_ok(labels, segs, "G22")

        # ---- with hypotenuse 1, the sine is the side opposite the angle
        for arcs, side in ((VGroup(a30), AMb), (VGroup(a60[0]), CMb),
                           (VGroup(a45[0]), legs[1])):
            self.play(Indicate(arcs, scale_factor=1.25, color=WHITE),
                      Indicate(side, scale_factor=1.0, color=WHITE), run_time=0.7)
        self.play(Write(caption("sin 30° = ½      sin 45° = √2/2      sin 60° = √3/2", 34)))
        self.hold(2.2)


# ============================================ E15  — cyclic quadrilateral

def _wedge_span(vertex, p, q):
    """(start angle, span) of the non-reflex angle p-vertex-q, as _wedge
    and angle_arc draw it (counter-clockwise from start)."""
    v = to3(vertex)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return a1, span


class E15_CyclicQuadrilateral(Board):
    """Opposite angles of a cyclic quadrilateral add to 180°. The angle at
    A stands on the arc BCD, the angle at C on the arc DAB, and the two
    arcs make up the whole circle. Each inscribed angle is half the central
    angle on its arc: two copies of the angle at A fill the central angle
    on BCD, two copies of the angle at C fill the rest of the turn. So
    2∠A + 2∠C = 360°, and one of each makes a straight angle."""

    def construct(self):
        R = 3.0
        O = np.array([-2.4, 0.3, 0.0])
        deg = {"A": 110.0, "B": 205.0, "C": 300.0, "D": 360.0}
        P = {k: O + R * _dir(v * DEGREES) for k, v in deg.items()}
        A, B, C, D = P["A"], P["B"], P["C"], P["D"]
        angA, angC = _ang(A, B, D), _ang(C, B, D)
        arcBCD = (deg["D"] - deg["B"]) * DEGREES          # through C
        arcDAB = TAU - arcBCD                              # through A
        check(abs(angA - arcBCD / 2) < 1e-9 and abs(angC - arcDAB / 2) < 1e-9,
              "each inscribed angle is half its arc")
        check(abs(angA + angC - PI) < 1e-9, "∠A + ∠C = 180°")
        check(deg["A"] < deg["B"] < deg["C"] < deg["D"], "A, B, C, D in order round the circle")

        circ = Circle(radius=R, color=GREY_B, stroke_width=2.5).move_to(O)
        dotO = Dot(O, radius=0.06)
        quad = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=3)
        vl = VGroup(*[tag(k, 28).move_to(O + (R + 0.36) * _dir(deg[k] * DEGREES))
                      for k in "ABCD"])
        self.play(Create(circ), FadeIn(dotO), run_time=1.0)
        self.play(Create(quad), FadeIn(vl), run_time=1.0)

        # the two arcs the opposite angles stand on
        arc_b = Arc(radius=R, start_angle=deg["B"] * DEGREES, angle=arcBCD,
                    arc_center=O, color=BLUE_B, stroke_width=8)
        arc_o = Arc(radius=R, start_angle=0.0, angle=arcDAB, arc_center=O,
                    color=ORANGE, stroke_width=8)
        rw = 0.85
        wA = _wedge(A, B, D, rw, BLUE_D, 0.8)
        wC = _wedge(C, B, D, rw, ORANGE, 0.8)
        self.play(Create(arc_b), Create(arc_o), run_time=0.9)
        self.play(FadeIn(wA), FadeIn(wC), run_time=0.7)
        self.bring_to_front(quad)

        # the central angles: two radii
        rB = Line(O, B, color=GREY_A, stroke_width=2.5)
        rD = Line(O, D, color=GREY_A, stroke_width=2.5)
        self.play(Create(rB), Create(rD), run_time=0.6)

        # two copies of the angle at A fill the central angle on its arc
        a1, sA = _wedge_span(A, B, D)
        c1, sC = _wedge_span(C, B, D)
        startA = deg["B"] * DEGREES                      # from OB round to OD
        startC = 0.0                                     # from OD round to OB
        check(abs(2 * sA - arcBCD) < 1e-9 and abs(2 * sC - arcDAB) < 1e-9,
              "two copies of each angle fill its central angle")
        cps, moves = [], []
        for (V, a0, sp, st, col) in ((A, a1, sA, startA, BLUE_D),
                                     (C, c1, sC, startC, ORANGE)):
            for j in range(2):
                w = Sector(radius=rw, angle=sp, start_angle=a0, arc_center=V,
                           fill_color=col, fill_opacity=0.8, stroke_width=0)
                src = [V, V + rw * _dir(a0), V + rw * _dir(a0 + sp)]
                dst = [O, O + rw * _dir(st + j * sp), O + rw * _dir(st + (j + 1) * sp)]
                cps.append(w)
                moves.append(_rigid(w, src, dst))
        self.add(*cps[:2])
        self.play(LaggedStart(*moves[:2], lag_ratio=0.35), run_time=1.9)
        l2A = tag("2∠A", 26, BLUE_B).move_to(O + 1.55 * _dir(245 * DEGREES))
        self.play(FadeIn(l2A), run_time=0.4)
        self.add(*cps[2:])
        self.play(LaggedStart(*moves[2:], lag_ratio=0.35), run_time=1.9)
        l2C = tag("2∠C", 26, ORANGE).move_to(O + 1.55 * _dir(150 * DEGREES))
        ring = Circle(radius=rw, color=WHITE, stroke_width=2).move_to(O)
        self.bring_to_front(dotO)
        self.play(FadeIn(l2C), Create(ring), run_time=0.6)
        self.hold(0.4)

        # one of each: a straight angle
        Q = np.array([4.35, -0.75, 0.0])
        f = 2.0
        pair = VGroup(cps[1].copy(), cps[2].copy())      # A from 282.5°, C to 102.5°
        turn = PI - (startC + sC)                        # the pair's far edge → 180°
        a_lo = (startA + sA + turn) % TAU                # where the A copy then starts
        check(min(a_lo, TAU - a_lo) < 1e-9 and abs(sA + sC - PI) < 1e-9,
              "the pair turns onto the half-turn above the base line")
        self.add(pair)

        def upd(m, t, start=pair.copy()):
            m.become(start.copy().rotate(t * turn, about_point=O)
                     .shift(t * (Q - O)).scale(1 + t * (f - 1), about_point=O + t * (Q - O)))
        self.play(UpdateFromAlphaFunc(pair, upd), run_time=1.6)
        base = Line(Q + 2.35 * LEFT, Q + 2.35 * RIGHT, color=WHITE, stroke_width=3)
        lA = tag("∠A", 26, WHITE).move_to(Q + 1.15 * f * 0.5 * _dir(sA / 2))
        lC = tag("∠C", 26, WHITE).move_to(Q + 1.15 * f * 0.5 * _dir(sA + sC / 2))
        l180 = tag("180°", 28, YELLOW_B).next_to(base, DOWN, buff=0.2)
        segs = _arc_segs(O, R, 0.0, TAU, 96) + [
            _seg(A, B), _seg(B, C), _seg(C, D), _seg(D, A), _seg(O, B), _seg(O, D),
            _seg(base.get_start(), base.get_end())] + _arc_segs(O, rw, 0.0, TAU, 48) \
            + _arc_segs(Q, rw * f, 0.0, PI, 32)
        _labels_ok([*vl, l2A, l2C, l180], segs, "E15")
        _labels_ok([lA, lC], [_seg(base.get_start(), base.get_end()),
                              _seg(Q, Q + rw * f * _dir(sA))] + _arc_segs(Q, rw * f, 0.0, PI, 32),
                   "E15 half-turn", pad=0.04)
        self.play(Create(base), FadeIn(lA), FadeIn(lC), FadeIn(l180), run_time=0.8)
        self.play(Write(caption("2∠A + 2∠C = 360°     ⟹     ∠A + ∠C = 180°", 34)))
        self.hold(2.2)


# ============================================ E23  — squares and circles, in and around

class E23_InnerOuterSquares(Board):
    """The square inscribed in a circle is half the square around it, and
    the circle inscribed in a square is half the circle around it. Turned
    through 45°, the inner square has its corners at the midpoints of the
    outer square's sides; folded in over its sides, the outer square's four
    corners cover it exactly: outer = 2 × inner. One turn-and-shrink (45°,
    1/√2) carries the outer square onto the inner one and the circle in the
    outer square onto the circle in the inner square, so it halves every
    area alike: the inner circle is half the outer circle."""

    def construct(self):
        r = 3.0
        Z = np.array([0.0, 0.35, 0.0])

        def Pz(x, y):
            return Z + np.array([x, y, 0.0])

        outer = [Pz(r, r), Pz(-r, r), Pz(-r, -r), Pz(r, -r)]
        inner0 = [Z + r * _dir(PI / 4 + k * PI / 2) for k in range(4)]   # axis-aligned
        inner = [Z + r * _dir(k * PI / 2) for k in range(4)]             # turned 45°
        mids = [(outer[k] + outer[(k + 1) % 4]) / 2 for k in range(4)]
        check(all(close(Z + r * _dir(PI / 4 + PI / 4 + k * PI / 2), mids[(k + 0) % 4])
                  for k in range(4)), "turned 45°, the corners reach the side midpoints")
        corners = [[outer[k], inner[(k + 1) % 4], inner[k]] for k in range(4)]
        folded = [[_refl(outer[k], inner[k], inner[(k + 1) % 4] - inner[k]),
                   inner[(k + 1) % 4], inner[k]] for k in range(4)]
        check(all(close(f[0], Z) for f in folded), "each folded corner reaches the centre")
        check(_tiles_exactly(folded, inner), "the four folded corners tile the inner square")
        check(close(abs(area(outer)), 2 * abs(area(inner))), "outer = 2 × inner")
        q = 1 / np.sqrt(2)
        sim = [Z + q * (np.array([[np.cos(PI / 4), -np.sin(PI / 4), 0],
                                  [np.sin(PI / 4), np.cos(PI / 4), 0], [0, 0, 1]]) @ (p - Z))
               for p in outer]
        check(all(close(sim[k], inner[(k + 1) % 4]) for k in range(4)),
              "turn 45° and shrink by 1/√2: outer square → inner square")
        check(close(r * q, np.linalg.norm(mids[0] - Z) * q)
              and close(r * q, np.linalg.norm((inner[0] + inner[1]) / 2 - Z)),
              "… and the circle in the outer square → the circle in the inner square")

        circ = Circle(radius=r, color=GREY_A, stroke_width=3).move_to(Z)
        sqO = mk(outer, BLUE_E, 0.45)
        self.play(FadeIn(sqO), Create(circ), run_time=1.2)
        sqI = mk(inner0, TEAL_D, 0.75)
        dots0 = VGroup(*[Dot(p, radius=0.07, color=YELLOW_B) for p in inner0])
        self.play(FadeIn(sqI), FadeIn(dots0), run_time=0.9)
        self.hold(0.3)

        # turn the inner square 45°: its corners slide round to the midpoints
        self.play(Rotate(VGroup(sqI, dots0), angle=PI / 4, about_point=Z), run_time=1.6)
        check(all(close(d.get_center(), m, 1e-6) for d, m in
                  zip(dots0, [mids[(k + 0) % 4] for k in range(4)])),
              "the corners land on the midpoints")
        diags = VGroup(DashedLine(inner[0], inner[2], color=WHITE, stroke_width=2,
                                  dash_length=0.09),
                       DashedLine(inner[1], inner[3], color=WHITE, stroke_width=2,
                                  dash_length=0.09))
        cor = VGroup(*[mk(t, BLUE_D, 0.85) for t in corners])
        self.play(Create(diags), FadeIn(cor), run_time=0.8)
        self.bring_to_front(circ, dots0)

        # fold the four corners in over the inner square's sides
        self.play(*[Rotate(cor[k], angle=PI, axis=_unit(inner[(k + 1) % 4] - inner[k]),
                           about_point=inner[k]) for k in range(4)], run_time=2.0)
        check(all(close(cor[k].get_vertices()[0], Z, 1e-6) for k in range(4)),
              "every folded corner meets the centre")
        self.bring_to_front(circ, dots0)
        self.hold(0.9)
        self.play(*[Rotate(cor[k], angle=-PI, axis=_unit(inner[(k + 1) % 4] - inner[k]),
                           about_point=inner[k]) for k in range(4)], run_time=1.4)
        self.play(FadeOut(diags), FadeOut(dots0), run_time=0.4)
        self.bring_to_front(circ)

        # the circle in the inner square; one turn-and-shrink maps the outer
        # pair (square, circle in it) onto the inner pair
        cin = Circle(radius=r * q, color=YELLOW_B, stroke_width=4).move_to(Z)
        self.play(Create(cin), circ.animate.set_stroke(color=WHITE, width=4), run_time=0.9)
        ghost = VGroup(Polygon(*outer, stroke_color=WHITE, stroke_width=3),
                       Circle(radius=r, color=WHITE, stroke_width=3).move_to(Z))
        self.add(ghost)
        tagq = tag("45°,  × 1/√2", 26).move_to(np.array([5.0, 2.6, 0.0]))
        self.play(FadeIn(tagq), _spiral(ghost, Z, PI / 4, q), run_time=2.2)
        check(all(close(v, w, 1e-6) for v, w in
                  zip(ghost[0].get_vertices(), [inner[(k + 1) % 4] for k in range(4)]))
              and close(ghost[1].width, 2 * r * q, 1e-6),
              "the ghost lands on the inner square and the inner circle")
        _labels_ok([tagq], [_seg(outer[k], outer[(k + 1) % 4]) for k in range(4)]
                   + _arc_segs(Z, r, 0.0, TAU, 64), "E23")
        self.play(FadeOut(ghost), FadeOut(tagq), run_time=0.5)
        self.bring_to_front(circ, cin)
        self.play(Write(caption(
            "inner square = ½ outer square        inner circle = ½ outer circle", 30)))
        self.hold(2.2)


# ============================================ E28  — Thales by a rectangle

def _chevron(p, q, color=YELLOW_B, size=0.17, width=3, n=1):
    """Parallel-line mark: n small '>' at the middle of segment pq,
    pointing from p towards q."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = (p + q) / 2
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7, tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


class E28_ThalesRectangle(Board):
    """Thales' theorem by a rectangle. Turn the triangle ABC on the diameter
    AB half a turn about the centre O: A and B change places and C goes to
    D, the far end of the diameter through C. Triangle and copy make the
    parallelogram ACBD (a half-turn sends each side to a parallel one), and
    its diagonals AB and CD are both diameters — equal. Folding over the
    line through O square to CB maps the figure onto itself, so the angles
    at C and B are equal; slid along BC, the angle at B sits beside the
    angle at C on a straight line. Two equal angles making 180°: ∠ACB = 90°."""

    def construct(self):
        R = 2.95
        O = np.array([-0.9, 0.3, 0.0])
        A, B = O + R * LEFT, O + R * RIGHT
        C = O + R * _dir(118 * DEGREES)
        D = 2 * O - C
        u = _unit(np.array([-(B - C)[1], (B - C)[0], 0.0]))   # through O, ⊥ CB
        check(close(np.linalg.norm(D - O), R) and close(2 * O - A, B),
              "the half-turn about O: A ↔ B, C → D on the circle")
        check(abs(np.cross(C - A, B - D)[2]) < 1e-9 and abs(np.cross(B - C, D - A)[2]) < 1e-9,
              "ACBD has opposite sides parallel")
        check(close(np.linalg.norm(B - A), np.linalg.norm(D - C)), "equal diagonals")
        check(close(_refl(C, O, u), B) and close(_refl(A, O, u), D),
              "the fold over the line through O square to CB swaps C, B and A, D")
        check(abs(_ang(C, A, B) - _ang(B, C, D)) < 1e-9, "∠ACB = ∠CBD")
        check(abs(_ang(C, A, B) + _ang(B, C, D) - PI) < 1e-9, "∠ACB + ∠CBD = 180°")
        check(close(D - B, A - C), "BD, slid to C, runs along CA")

        circ = Circle(radius=R, color=GREY_B, stroke_width=2.5).move_to(O)
        dotO = Dot(O, radius=0.06)
        AB = Line(A, B, color=WHITE, stroke_width=3)
        lA = tag("A", 28).next_to(A, LEFT, buff=0.16)
        lB = tag("B", 28).next_to(B, RIGHT, buff=0.16)
        lC = tag("C", 28).move_to(C + 0.4 * _dir(80 * DEGREES))   # clear of BC produced
        lD = tag("D", 28).move_to(D + 0.36 * _unit(D - O))
        self.play(Create(circ), Create(AB), FadeIn(dotO), FadeIn(lA), FadeIn(lB),
                  run_time=1.0)
        t1 = mk([A, B, C], BLUE_D, 0.5)
        self.play(FadeIn(t1), FadeIn(lC), run_time=0.9)
        self.bring_to_front(AB, dotO)

        # half a turn about the centre
        t2 = mk([A, B, C], TEAL_D, 0.5)
        self.add(t2)
        self.play(Rotate(t2, angle=PI, about_point=O), run_time=1.8)
        check(all(close(v, w, 1e-6) for v, w in zip(t2.get_vertices(), (B, A, D))),
              "the turned copy is BAD")
        self.bring_to_front(AB, dotO)
        par = Polygon(A, C, B, D, stroke_color=WHITE, stroke_width=3)
        chev = VGroup(_chevron(A, C, YELLOW_B), _chevron(D, B, YELLOW_B),
                      _chevron(C, B, ORANGE, n=2), _chevron(A, D, ORANGE, n=2))
        self.play(FadeIn(lD), Create(par), FadeIn(chev), run_time=0.9)

        # its diagonals are two diameters
        dg = VGroup(Line(A, B, color=YELLOW_B, stroke_width=5),
                    Line(C, D, color=YELLOW_B, stroke_width=5))
        ticks = VGroup(*[_ticks(O, X, 1, YELLOW_B) for X in (A, B, C, D)])
        self.play(Create(dg), FadeIn(ticks), run_time=1.0)
        self.bring_to_front(dotO)
        self.hold(0.4)

        # fold over the line through O square to CB: the angle at C → at B
        axis = DashedLine(O - 3.25 * u, O + 3.25 * u, color=GREY_A, stroke_width=2,
                          dash_length=0.09)
        rw = 0.62
        wC = _wedge(C, A, B, rw, YELLOW_E, 0.85)
        self.play(Create(axis), FadeIn(wC), run_time=0.7)
        ghost = DashedVMobject(Polygon(A, C, B, D, stroke_color=WHITE, stroke_width=2),
                               num_dashes=60)
        wf = wC.copy()
        self.add(ghost, wf)
        self.play(Rotate(VGroup(ghost, wf), angle=PI, axis=u, about_point=O), run_time=1.8)
        aC0, sC0 = _wedge_span(C, A, B)
        aB0, sB0 = _wedge_span(B, C, D)
        check(abs(sC0 - sB0) < 1e-9, "the folded angle fits the angle at B")
        self.play(FadeOut(ghost), FadeOut(axis), run_time=0.4)

        # slide the angle at B along BC to C: beside the angle at C, a straight line
        ext = DashedLine(B, C + 1.25 * _unit(C - B), color=GREY_A, stroke_width=2,
                         dash_length=0.09)
        self.play(Create(ext), run_time=0.6)
        ws = wf.copy()
        self.play(ws.animate.shift(C - B), run_time=1.4)
        check(close(_refl(C, O, u) + (C - B), C) and close(_unit(D - B), _unit(A - C))
              and abs(_ang(C, C + (C - B), A) - sB0) < 1e-9 and abs(sC0 + sB0 - PI) < 1e-9,
              "slid to C, the angle at B fills the rest of the straight angle on BC")
        mark = _ra(C, A, B, 0.3, WHITE, 3)
        segs = _arc_segs(O, R, 0.0, TAU, 96) + [
            _seg(A, B), _seg(C, D), _seg(A, C), _seg(C, B), _seg(B, D), _seg(D, A),
            _seg(ext.get_start(), ext.get_end())]
        _labels_ok([lA, lB, lC, lD], segs, "E28")
        self.play(Create(mark), run_time=0.7)
        self.play(Write(caption(
            "equal diagonals  ⟹  ACBD is a rectangle  ⟹  ∠ACB = 90°", 32)))
        self.hold(2.2)
