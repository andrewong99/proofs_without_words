# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3c.py — proofs without words, 2D (manim), means & inequalities:
#     D20 the mediant                       D17 Napier's inequality
#     D19 bounds for n factorial            D24 bounds for ln(1 + x)
#     D23 AM–GM from a tangent line         D25 square roots add above
#     D22 chord shorter than arc            D26 Minkowski's inequality
#     D27 a sum times its reciprocals       D28 products of three numbers
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# exponents made of letters (e^(t − 1)) are set with true superscripts by
# _supline.  Curves are polylines through densely sampled points; axes carry
# no numbers (tick values, where needed, are tag() labels).
#
# Every scene asserts its invariants with check(...) before drawing — the
# inequality itself, exact landings of moved pieces, gap-free covers — and
# checks at the end that its labels sit inside the safe area, clear of each
# other and of the lines, so a wrong picture fails the render.


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


def _under(s, p, size=28, color=WHITE, gap=0.14):
    """Label centred under the screen point p, its baseline a fixed depth
    below p, so labels under one line share a baseline."""
    cap = Text("M", font_size=size).height
    t = _on_base(s, size, color)
    p = to3(p)
    t.shift(np.array([p[0] - t.get_center()[0], p[1] - gap - cap, 0.0]))
    return t


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


def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _perp(v):
    v = to3(v)
    return np.array([-v[1], v[0], 0.0])


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _foot(p, a, b):
    """Foot of the perpendicular from p onto the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = b - a
    return a + np.dot(p - a, d) / np.dot(d, d) * d


def _rot2(p, c, th):
    """Point p turned by th about c (in the plane)."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


def _rect(x0, y0, w, h):
    """Axis-aligned rectangle, counter-clockwise from the bottom-left."""
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


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


def _same_set(P, Q, tol=1e-6):
    """The same points, in any order."""
    P = [to3(p) for p in P]
    Q = [to3(q) for q in Q]
    return (len(P) == len(Q)
            and all(any(np.linalg.norm(p - q) < tol for q in Q) for p in P)
            and all(any(np.linalg.norm(p - q) < tol for p in P) for q in Q))


# ------------------------------------------------------------ drawing helpers

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


def _ra(v, p, q, s=0.18, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _dash(p, q, color=GREY_B, width=2, dash=0.08):
    return DashedLine(to3(p), to3(q), color=color, stroke_width=width,
                      dash_length=dash)


def _arrow(p, q, color, width=6, tip=0.24):
    """Arrow from p to q (screen) whose tip has a fixed size."""
    p, q = to3(p), to3(q)
    L = float(np.linalg.norm(q - p))
    return Arrow(p, q, buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=tip / L,
                 max_stroke_width_to_length_ratio=40)


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


# ------------------------------------------------------------ label hygiene

def _segs(pts):
    """Consecutive segments of a polyline of screen points."""
    pts = [to3(p) for p in pts]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _loop(pts):
    """Segments of a closed polygon of screen points."""
    pts = [to3(p) for p in pts]
    return _segs(pts + [pts[0]])


def _circle_segs(c, r, n=96, a0=0.0, a1=TAU):
    c = to3(c)
    pts = [c + r * np.array([np.cos(a), np.sin(a), 0.0])
           for a in np.linspace(a0, a1, n + 1)]
    return _segs(pts)


def _mob_segs(m):
    """Segments along the outline of a drawn mobject (its anchor points)."""
    pts = m.get_anchors()
    return [(to3(pts[i]), to3(pts[i + 1])) for i in range(len(pts) - 1)]


def _dot_segs(c, r=0.07):
    """A dot of radius r at screen point c, as segments (labels must clear it)."""
    return _circle_segs(c, r, 16)


def _marks(*mobs):
    """Segments of small marks (ticks, right-angle marks, dimension bars)."""
    out = []
    for m in mobs:
        subs = [m] if not m.submobjects else m.family_members_with_points()
        for x in subs:
            out += _mob_segs(x)
    return out


def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by
    pad), or None."""
    x0, x1, y0, y1 = _box(m, pad)
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
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


def _inside(m, poly, pad=0.03):
    """Fail the render unless label m's box (grown by pad) lies inside the
    screen polygon poly."""
    x0, x1, y0, y1 = _box(m, pad)
    for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        check(_pip(c, [to3(p) for p in poly]),
              f"label '{_name(m)}' inside its region")


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


# ========================================================== D20  the mediant

class D20_Mediant(Board):
    """The arrow u = (b, a) has run b and rise a, slope a/b; the arrow
    v = (d, c) has slope c/d, steeper (b, d > 0).  Placed head to tail they
    reach W = (b + d, a + c): the single arrow OW has run b + d and rise
    a + c, slope (a + c)/(b + d).  A copy of u from the head of v reaches the
    same W, so O, U, W, V is a parallelogram and OW its diagonal, inside the
    angle UOV: its slope lies between the slopes of u and v.
    (Drawn with a, b, c, d > 0; the argument needs only b, d > 0.)"""

    def construct(self):
        a, b, c, d = 1.2, 4.0, 2.6, 1.6
        k = 1.15
        O = np.array([-3.85, -1.55, 0.0])

        def S(x, y):
            return O + k * np.array([x, y, 0.0])

        U, V, W = S(b, a), S(d, c), S(b + d, a + c)

        # ---- the claims
        check(a / b < c / d, "a/b < c/d")
        check(a / b < (a + c) / (b + d) < c / d, "a/b < (a + c)/(b + d) < c/d")
        check(close(W - O, (U - O) + (V - O)), "W = U + V: head to tail")
        check(close(W - V, U - O) and close(W - U, V - O),
              "O, U, W, V is a parallelogram")
        check(_cross(U - O, W - O) > 0 and _cross(W - O, V - O) > 0
              and _cross(U - O, V - O) > 0,
              "OW lies inside the angle UOV (turning from OU towards OV)")
        for t in np.linspace(0.05, 0.95, 19):
            # every point of the far sides UW, VW is inside the angle too
            for P in (U + t * (W - U), V + t * (W - V)):
                check(_cross(U - O, P - O) >= -1e-12 and _cross(P - O, V - O) >= -1e-12,
                      "the parallelogram lies in the angle UOV")

        # ---- u = (b, a): run b, rise a
        Pu = S(b, 0)
        au = _arrow(O, U, BLUE_D)
        legs_u = VGroup(_dash(O, Pu, BLUE_B, 2.5), _dash(Pu, U, BLUE_B, 2.5),
                        _ra(Pu, O, U, 0.16, BLUE_B, 2))
        l_b = tag("b", 30, BLUE_B).next_to((O + Pu) / 2, DOWN, buff=0.15)
        l_a = tag("a", 30, BLUE_B).next_to((Pu + U) / 2, RIGHT, buff=0.15)
        dot0 = Dot(O, radius=0.06)
        self.play(FadeIn(dot0), GrowArrow(au), run_time=1.0)
        self.play(Create(legs_u), FadeIn(l_b), FadeIn(l_a), run_time=0.9)

        # ---- v = (d, c): rise c, run d (steeper)
        Pv = S(0, c)
        av = _arrow(O, V, TEAL_D)
        legs_v = VGroup(_dash(O, Pv, TEAL_B, 2.5), _dash(Pv, V, TEAL_B, 2.5),
                        _ra(Pv, O, V, 0.16, TEAL_B, 2))
        l_c = tag("c", 30, TEAL_B).next_to((O + Pv) / 2, LEFT, buff=0.15)
        l_d = tag("d", 30, TEAL_B).next_to((Pv + V) / 2, UP, buff=0.13)
        self.play(GrowArrow(av), run_time=1.0)
        self.play(Create(legs_v), FadeIn(l_c), FadeIn(l_d), run_time=0.9)
        self.bring_to_front(dot0)
        self.hold(0.4)

        # ---- head to tail: a copy of v (with its legs) slides to the head of u
        mv = VGroup(av.copy(), legs_v.copy())
        self.add(mv)
        self.play(mv.animate.shift(U - O), run_time=1.5)
        check(close(mv[0].get_start(), U) and close(mv[0].get_end(), W),
              "the copy of v runs from U to W")
        Pv2 = S(b, a + c)
        check(_same_pts([mv[1][0].get_start(), mv[1][0].get_end(),
                         mv[1][1].get_end()], [U, Pv2, W]),
              "its legs: up c from U, then across d to W")
        l_c2 = tag("c", 30, TEAL_B).next_to((U + Pv2) / 2, RIGHT, buff=0.15)
        l_d2 = tag("d", 30, TEAL_B).next_to((Pv2 + W) / 2, UP, buff=0.13)
        self.play(FadeIn(l_c2), FadeIn(l_d2), run_time=0.5)

        # ---- the single arrow OW: run b + d, rise a + c
        aw = _arrow(O, W, YELLOW_B, 6)
        off_h, off_v = 0.78, 0.55
        dimH = _dim(S(0, 0) + DOWN * off_h, S(b + d, 0) + DOWN * off_h, YELLOW_B)
        dimV = _dim(S(b + d, 0) + RIGHT * off_v, W + RIGHT * off_v, YELLOW_B)
        l_bd = tag("b + d", 30, YELLOW_B).next_to(dimH, DOWN, buff=0.12)
        l_ac = tag("a + c", 30, YELLOW_B).next_to(
            S(b + d, 0.7 * (a + c)) + RIGHT * off_v, RIGHT, buff=0.15)
        guides = VGroup(_dash(W, S(b + d, 0) + DOWN * (off_h + 0.1)),
                        _dash(Pu, S(b + d, 0) + RIGHT * (off_v + 0.1)),
                        _dash(W, W + RIGHT * (off_v + 0.1)))
        self.play(GrowArrow(aw), run_time=1.0)
        self.add(guides)
        self.bring_to_back(guides)
        self.play(Create(guides), Create(dimH), Create(dimV), FadeIn(l_bd),
                  FadeIn(l_ac), run_time=1.1)
        self.bring_to_front(au, mv, aw, dot0)
        self.hold(0.6)

        # ---- a copy of u from the head of v reaches the same W: a parallelogram
        mu = au.copy()
        self.add(mu)
        self.play(mu.animate.shift(V - O), run_time=1.4)
        check(close(mu.get_start(), V) and close(mu.get_end(), W),
              "the copy of u runs from V to W")
        para = mk([O, U, W, V], YELLOW_E, 0.16, stroke_width=0)
        self.add(para)
        self.bring_to_back(para)
        self.play(FadeIn(para), run_time=0.6)

        # ---- OW lies inside the angle UOV
        arc1 = angle_arc(O, U, W, 1.55, BLUE_B, 6)
        arc2 = angle_arc(O, W, V, 1.55, TEAL_B, 6)
        check(arc1.angle > 0.05 and arc2.angle > 0.05 and arc1.angle + arc2.angle < PI,
              "two genuine angles between u, u + v and v")
        self.play(Create(arc1), run_time=0.5)
        self.play(Create(arc2), run_time=0.5)
        self.bring_to_front(au, av, mv, mu, aw, dot0)
        self.hold(0.5)

        # ---- checks on the finished figure
        segs = (_segs([O, U]) + _segs([O, V]) + _segs([U, W]) + _segs([V, W])
                + _segs([O, W]) + _segs([O, Pu, U]) + _segs([O, Pv, V])
                + _segs([U, Pv2, W]) + _mob_segs(dimH[0]) + _mob_segs(dimV[0])
                + _segs([W, S(b + d, 0) + DOWN * (off_h + 0.1)])
                + _segs([Pu, S(b + d, 0) + RIGHT * (off_v + 0.1)])
                + _segs([W, W + RIGHT * (off_v + 0.1)])
                + _mob_segs(arc1) + _mob_segs(arc2)
                + _marks(legs_u[2], legs_v[2], mv[1][2], dimH, dimV) + _dot_segs(O, 0.06))
        labels = [l_b, l_a, l_c, l_d, l_c2, l_d2, l_bd, l_ac]
        _labels_ok(labels, segs, "D20")
        content = [au, av, aw, mv, mu, dimH, dimV, guides, *labels]
        _safe(*content)
        cap = caption("a/b  <  c/d   ⟹   a/b  <  (a + c)/(b + d)  <  c/d,"
                      "     b, d > 0", 32)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== D17  Napier

class D17_Napier(Board):
    """0 < a < b.  On [a, b] the curve y = 1/x falls from 1/a to 1/b, so
    the rectangle of height 1/b (area (b − a)/b) fits under it and the
    rectangle of height 1/a (area (b − a)/a) covers the region under it,
    whose area is ln b − ln a.  The three share the base b − a: divided by
    it, the mean height (ln b − ln a)/(b − a) of the region lies between
    1/b and 1/a.  (Uses ∫ₐᵇ dx/x = ln b − ln a.)"""

    def construct(self):
        a, b = 0.75, 2.2
        kx, ky = 2.9, 2.2
        O = np.array([-5.6, -1.64, 0.0])

        def S(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        def f(x):
            return 1.0 / x

        L = float(np.log(b) - np.log(a))
        m = L / (b - a)

        # ---- the claims
        xs = np.linspace(a, b, 2001)
        check(bool(np.all(f(xs) >= 1 / b - 1e-15)) and bool(np.all(f(xs) <= 1 / a + 1e-15)),
              "1/b ≤ 1/x ≤ 1/a on [a, b]: inner rectangle under, outer above")
        check(abs(_simpson(f, a, b, 2000) - L) < 1e-9,
              "area under 1/x from a to b = ln b − ln a")
        check((b - a) / b < L < (b - a) / a, "(b − a)/b < ln b − ln a < (b − a)/a")
        check(1 / b < m < 1 / a, "1/b < (ln b − ln a)/(b − a) < 1/a")
        check(abs(_simpson(lambda x: f(x) - 1 / b, a, b, 2000) - (L - (b - a) / b)) < 1e-9
              and abs(_simpson(lambda x: 1 / a - f(x), a, b, 2000)
                      - ((b - a) / a - L)) < 1e-9,
              "the coloured gaps are the two differences")

        x_lo, x_hi = 0.47, 3.2
        cx = np.concatenate([np.geomspace(x_lo, 1.2, 120)[:-1],
                             np.linspace(1.2, x_hi, 160)])
        cpts = [S(x, f(x)) for x in cx]
        ax_x = [S(-0.1, 0), S(x_hi + 0.15, 0)]
        ax_y = [S(0, -0.05), S(0, 2.2)]
        axes = VGroup(Line(*ax_x, color=GREY_B, stroke_width=2),
                      Line(*ax_y, color=GREY_B, stroke_width=2))
        curve = _curve(cpts, YELLOW_B, 5)
        l_curve = tag("y = 1/x", 28, YELLOW_B).next_to(cpts[0], RIGHT, buff=0.22)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.2)

        # ---- a and b, the region under the curve between them
        tka, tkb = _tick(S(a, 0)), _tick(S(b, 0))
        l_a = _under("a", S(a, 0), 30, WHITE, 0.2)
        l_b = _under("b", S(b, 0), 30, WHITE, 0.2)
        rx = np.linspace(a, b, 160)
        reg_pts = [S(a, 0)] + [S(x, f(x)) for x in rx] + [S(b, 0)]
        outline = Polygon(*reg_pts, stroke_color=YELLOW_B, stroke_width=5)
        self.play(Create(tka), Create(tkb), FadeIn(l_a), FadeIn(l_b), run_time=0.6)
        self.play(Create(outline), run_time=1.0)

        # ---- the inner rectangle, height 1/b, fits under the curve
        inner_pts = [S(a, 0), S(b, 0), S(b, 1 / b), S(a, 1 / b)]
        inner = mk(inner_pts, BLUE_D, 0.85, stroke_width=0)
        gap_in = mk([S(a, 1 / b)] + [S(x, f(x)) for x in rx], TEAL_D, 0.6,
                    stroke_width=0)
        g_b = _dash(S(0, 1 / b), S(a, 1 / b))
        tk_ib = _tick(S(0, 1 / b), RIGHT)
        l_ib = tag("1/b", 28, BLUE_B).next_to(S(0, 1 / b), LEFT, buff=0.16)
        self.add(inner)
        self.bring_to_back(inner)
        self.play(GrowFromEdge(inner, DOWN), run_time=1.0)
        self.play(Create(g_b), Create(tk_ib), FadeIn(l_ib), FadeIn(gap_in), run_time=0.8)
        self.bring_to_front(outline, curve)

        # ---- the outer rectangle, height 1/a, covers the region
        outer_pts = [S(a, 0), S(b, 0), S(b, 1 / a), S(a, 1 / a)]
        outer = Polygon(*outer_pts, stroke_color=WHITE, stroke_width=4)
        gap_out = mk([S(x, f(x)) for x in rx] + [S(b, 1 / a)], ORANGE, 0.85,
                     stroke_width=0)
        g_a = _dash(S(0, 1 / a), S(a, 1 / a))
        tk_ia = _tick(S(0, 1 / a), RIGHT)
        l_ia = tag("1/a", 28).next_to(S(0, 1 / a), LEFT, buff=0.16)
        self.play(Create(outer), Create(g_a), Create(tk_ia), FadeIn(l_ia), run_time=1.0)
        self.add(gap_out)
        self.bring_to_back(gap_out)
        self.play(FadeIn(gap_out), run_time=0.6)
        self.bring_to_front(outline, curve, outer)

        # ---- one base for all three: b − a
        yb = 0.78
        dimB = _dim(S(a, 0) + DOWN * yb, S(b, 0) + DOWN * yb, WHITE)
        l_ba = tag("b − a", 28).next_to(dimB, DOWN, buff=0.12)
        self.play(Create(dimB), FadeIn(l_ba), run_time=0.7)

        row = _trow([("(b − a)/b", BLUE_B), ("<", WHITE), ("ln b − ln a", YELLOW_B),
                     ("<", WHITE), ("(b − a)/a", WHITE)], 30)
        row.move_to([2.35, 2.45, 0.0])
        self.play(FadeIn(row), run_time=1.0)
        self.hold(0.6)

        # ---- divided by the base: the mean height lies between 1/b and 1/a
        mline = DashedLine(S(a, m), S(b, m), color=YELLOW_B, stroke_width=4,
                           dash_length=0.1)
        l_m = tag("(ln b − ln a)/(b − a)", 26, YELLOW_B).next_to(S(b, m), RIGHT, buff=0.2)
        self.play(Create(mline), FadeIn(l_m), run_time=1.0)
        self.hold(0.4)

        # ---- checks on the finished figure
        segs = (_segs(cpts) + _segs(ax_x) + _segs(ax_y) + _loop(outer_pts)
                + _loop(inner_pts) + _segs([S(0, 1 / b), S(a, 1 / b)])
                + _segs([S(0, 1 / a), S(a, 1 / a)]) + _segs([S(a, m), S(b, m)])
                + _segs(reg_pts) + _marks(tka, tkb, tk_ib, tk_ia, dimB))
        labels = [l_curve, l_a, l_b, l_ib, l_ia, l_ba, row, l_m]
        _labels_ok(labels, segs, "D17")
        content = [axes, curve, outer, inner, gap_out, dimB, *labels]
        _safe(*content)
        cap = caption("1/b  <  (ln b − ln a)/(b − a)  <  1/a,     0 < a < b", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== D19  bounds for n!

class D19_FactorialBounds(Board):
    """Bars of width 1 and heights ln 2, ln 3, …, ln n (here n = 5) have
    total area ln 2 + … + ln n = ln n!.  Standing over [k − 1, k], bar k
    rises above y = ln x (ln k ≥ ln x there; it meets the curve at its
    top-right corner), so the bars contain the region under the curve from
    1 to n, of area n ln n − n + 1.  Slid one unit right, onto [k, k + 1],
    each bar lies under the curve (ln k ≤ ln x there), inside the region
    under it from 1 to n + 1, of area (n + 1) ln(n + 1) − n.  Both
    inequalities are strict: the orange overhangs and the blue gaps have
    positive area.  (Uses ∫₁ᵗ ln x dx = t ln t − t + 1.)"""

    def construct(self):
        n = 5
        kx, ky = 1.85, 2.75
        O = np.array([-6.05, -2.5, 0.0])

        def S(x, y):
            return O + np.array([kx * x, ky * y, 0.0])

        def F(t):                      # ∫₁ᵗ ln x dx
            return t * np.log(t) - t + 1

        lnf = float(sum(np.log(k) for k in range(2, n + 1)))

        # ---- the claims
        check(abs(lnf - np.log(120.0)) < 1e-12, "ln 2 + ln 3 + ln 4 + ln 5 = ln 5!")
        for t in (n, n + 1, 3.7):
            check(abs(_simpson(np.log, 1, t, 2000) - F(t)) < 1e-9,
                  "∫₁ᵗ ln x dx = t ln t − t + 1")
        check(abs(F(n) - (n * np.log(n) - n + 1)) < 1e-12
              and abs(F(n + 1) - ((n + 1) * np.log(n + 1) - n)) < 1e-12,
              "the two areas are n ln n − n + 1 and (n + 1) ln(n + 1) − n")
        for k in range(2, n + 1):
            xs = np.linspace(k - 1, k, 201)
            check(bool(np.all(np.log(k) >= np.log(xs) - 1e-15)),
                  f"bar {k} over [k − 1, k] stands above the curve")
            xs = np.linspace(k, k + 1, 201)
            check(bool(np.all(np.log(k) <= np.log(xs) + 1e-15)),
                  f"bar {k} over [k, k + 1] lies under the curve")
        over = sum(np.log(k) - _simpson(np.log, k - 1, k) for k in range(2, n + 1))
        check(abs(over - (lnf - F(n))) < 1e-9 and over > 0,
              "orange overhangs = ln 5! − ∫₁⁵ ln x dx > 0")
        gaps = _simpson(np.log, 1, 2) + sum(_simpson(np.log, k, k + 1) - np.log(k)
                                            for k in range(2, n + 1))
        check(abs(gaps - (F(n + 1) - lnf)) < 1e-9 and gaps > 0,
              "blue gaps = ∫₁⁶ ln x dx − ln 5! > 0")
        check(F(n) < lnf < F(n + 1), "n ln n − n + 1 < ln n! < (n + 1) ln(n + 1) − n")

        x_hi = n + 1.4
        cx = np.linspace(1.0, x_hi, 260)
        cpts = [S(x, np.log(x)) for x in cx]
        ax_x = [S(-0.12, 0), S(x_hi + 0.15, 0)]
        ax_y = [S(0, -0.05), S(0, 0.95)]
        axes = VGroup(Line(*ax_x, color=GREY_B, stroke_width=2),
                      Line(*ax_y, color=GREY_B, stroke_width=2))
        ticks = VGroup(*[_tick(S(k, 0)) for k in range(1, n + 2)])
        tlabs = VGroup(*[_under(str(k), S(k, 0), 26, GREY_A, 0.2)
                         for k in range(1, n + 2)])
        curve = _curve(cpts, YELLOW_B, 5)
        l_curve = tag("y = ln x", 28, YELLOW_B).next_to(cpts[-1], UP, buff=0.22)
        l_curve.shift(LEFT * max(0.0, l_curve.get_right()[0] - 6.45))
        self.play(Create(axes), Create(ticks), FadeIn(tlabs), run_time=0.8)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.1)

        # ---- bars of heights ln 2 … ln 5 over [k − 1, k]
        def bar_pts(k, s):
            return [S(k - 1 + s, 0), S(k + s, 0), S(k + s, np.log(k)),
                    S(k - 1 + s, np.log(k))]

        bars = VGroup()
        for k in range(2, n + 1):
            b = mk(bar_pts(k, 0), TEAL_D, 0.7, stroke_width=2.5)
            t = tag(f"ln {k}", 24).move_to(S(k - 0.4, 0.26 * np.log(2)))
            bars.add(VGroup(b, t))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars],
                              lag_ratio=0.25), run_time=1.6)
        self.bring_to_front(curve)

        row0 = _trow([("ln 2 + ln 3 + ln 4 + ln 5", TEAL_B), ("=", WHITE),
                      ("ln 5!", TEAL_B)], 28)
        row0.shift(np.array([-5.65 - row0.get_left()[0], 3.25, 0.0]))
        self.play(FadeIn(row0), run_time=0.8)

        # ---- they cover the region under the curve from 1 to 5
        def region(lo, hi):
            xs = np.linspace(lo, hi, 160)
            return [S(lo, 0)] + [S(x, np.log(x)) for x in xs] + [S(hi, 0)]

        outA = Polygon(*region(1, n), stroke_color=BLUE_B, stroke_width=6)
        overs = VGroup(*[mk([S(k - 1, np.log(k))]
                            + [S(x, np.log(x)) for x in np.linspace(k - 1, k, 60)],
                            ORANGE, 0.95, stroke_width=0) for k in range(2, n + 1)])
        self.play(Create(outA), run_time=1.0)
        self.play(FadeIn(overs), run_time=0.7)
        self.bring_to_front(curve, outA)
        for b in bars:
            self.bring_to_front(b[1])

        row1 = _row(_int("1", "5", 28, BLUE_B), tag("ln x dx", 28, BLUE_B),
                    tag("<", 28), tag("ln 5!", 28, TEAL_B), buff=0.14)
        row1.shift(np.array([-5.65 - row1.get_left()[0], 2.5 - row1.get_center()[1], 0.0]))
        self.play(FadeIn(row1), run_time=0.8)
        self.hold(0.8)

        segs1 = (_segs(cpts) + _segs(ax_x) + _segs(ax_y) + _segs(region(1, n)) + _marks(ticks)
                 + sum((_loop(bar_pts(k, 0)) for k in range(2, n + 1)), []))
        labs1 = [l_curve, row0, row1, *tlabs]
        _labels_ok(labs1, segs1, "D19 (bars over [k − 1, k])")
        for k, b in zip(range(2, n + 1), bars):
            check(_hit(b[1], _segs(cpts), 0.05) is None,
                  f"bar label ln {k} clear of the curve")
            _inside(b[1], bar_pts(k, 0))

        # ---- slide the bars one unit right: onto [k, k + 1]
        self.play(FadeOut(overs), FadeOut(outA), run_time=0.6)
        self.play(bars.animate.shift(RIGHT * kx), run_time=1.6)
        for k, b in zip(range(2, n + 1), bars):
            check(_same_pts(_verts(b[0]), bar_pts(k, 1)), f"bar {k} lands on [k, k + 1]")
        self.bring_to_front(curve)
        outB = Polygon(*region(1, n + 1), stroke_color=BLUE_B, stroke_width=6)
        gapsB = VGroup(mk(region(1, 2), BLUE_D, 0.95, stroke_width=0),
                       *[mk([S(k, np.log(k))] + [S(x, np.log(x))
                                                  for x in np.linspace(k, k + 1, 60)]
                            + [S(k + 1, np.log(k))], BLUE_D, 0.95, stroke_width=0)
                         for k in range(2, n + 1)])
        self.play(Create(outB), run_time=1.0)
        self.play(FadeIn(gapsB), run_time=0.7)
        self.bring_to_front(curve, outB)
        for b in bars:
            self.bring_to_front(b[1])

        row2 = _row(tag("ln 5!", 28, TEAL_B), tag("<", 28), _int("1", "6", 28, BLUE_B),
                    tag("ln x dx", 28, BLUE_B), buff=0.14)
        row2.shift(np.array([-5.65 - row2.get_left()[0], 1.75 - row2.get_center()[1], 0.0]))
        self.play(FadeIn(row2), run_time=0.8)
        row3 = _row(_int("1", "t", 26, GREY_A), tag("ln x dx  =  t ln t − t + 1", 26, GREY_A),
                    buff=0.1)
        row3.shift(np.array([-5.65 - row3.get_left()[0], 1.0 - row3.get_center()[1], 0.0]))
        self.play(FadeIn(row3), run_time=0.8)

        segs2 = (_segs(cpts) + _segs(ax_x) + _segs(ax_y) + _segs(region(1, n + 1)) + _marks(ticks)
                 + sum((_loop(bar_pts(k, 1)) for k in range(2, n + 1)), []))
        labs2 = [l_curve, row0, row1, row2, row3, *tlabs]
        _labels_ok(labs2, segs2, "D19 (bars over [k, k + 1])")
        for k, b in zip(range(2, n + 1), bars):
            check(_hit(b[1], _segs(cpts), 0.05) is None,
                  f"bar label ln {k} clear of the curve (slid)")
            _inside(b[1], bar_pts(k, 1))
        content = [axes, curve, bars, outB, *labs2]
        _safe(*content)
        cap = caption("n ln n − n + 1  <  ln n!  <  (n + 1) ln(n + 1) − n", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== D24  ln(1 + x)

class D24_LogOnePlusX(Board):
    """Under y = 1/t the area from 1 to 1 + x is ln(1 + x).  For x > 0 the
    curve falls from 1 to 1/(1 + x) over [1, 1 + x]: the rectangle of
    height 1/(1 + x) (area x/(1 + x)) fits under it and the one of height 1
    (area x) covers the region, so x/(1 + x) < ln(1 + x) < x.  For
    −1 < x < 0 the region over [1 + x, 1] has area −ln(1 + x) and lies
    between the rectangles of heights 1 and 1/(1 + x) on a base of width
    −x: −x < −ln(1 + x) < −x/(1 + x), the same inequalities.
    (Uses ∫₁ᵘ dt/t = ln u.)"""

    def construct(self):
        x1, x2 = 1.3, -0.55
        kx, ky = 2.2, 2.15
        O = np.array([-5.0, -2.35, 0.0])

        def S(t, y):
            return O + np.array([kx * t, ky * y, 0.0])

        def f(t):
            return 1.0 / t

        u1, u2 = 1 + x1, 1 + x2
        C1, C2 = GREEN_B, PURPLE_A

        # ---- the claims
        for x in (x1, x2, 0.4, -0.3, 3.0, -0.9):
            lo, hi = sorted((1.0, 1.0 + x))
            check(abs(_simpson(f, lo, hi, 2000) - abs(np.log(1 + x))) < 1e-9,
                  "the area under 1/t between 1 and 1 + x is |ln(1 + x)|")
            check(x / (1 + x) < np.log(1 + x) < x, "x/(1 + x) < ln(1 + x) < x")
        ts = np.linspace(1, u1, 400)
        check(bool(np.all(f(ts) >= 1 / u1 - 1e-15)) and bool(np.all(f(ts) <= 1 + 1e-15)),
              "x > 0: 1/(1 + x) ≤ 1/t ≤ 1 on [1, 1 + x]")
        ts = np.linspace(u2, 1, 400)
        check(bool(np.all(f(ts) >= 1 - 1e-15)) and bool(np.all(f(ts) <= 1 / u2 + 1e-15)),
              "x < 0: 1 ≤ 1/t ≤ 1/(1 + x) on [1 + x, 1]")
        check(-x2 < -np.log(u2) < -x2 / u2, "−x < −ln(1 + x) < −x/(1 + x)")

        t_lo, t_hi = 0.4, 2.6
        cx = np.concatenate([np.geomspace(t_lo, 1.2, 120)[:-1], np.linspace(1.2, t_hi, 140)])
        cpts = [S(t, f(t)) for t in cx]
        ax_x = [S(-0.1, 0), S(2.75, 0)]
        ax_y = [S(0, -0.05), S(0, 2.65)]
        axes = VGroup(Line(*ax_x, color=GREY_B, stroke_width=2),
                      Line(*ax_y, color=GREY_B, stroke_width=2))
        curve = _curve(cpts, YELLOW_B, 5)
        l_curve = tag("y = 1/t", 28, YELLOW_B).next_to(cpts[0], RIGHT, buff=0.22)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.1)

        # the level 1, reached at t = 1
        tk1 = _tick(S(1, 0))
        l_1 = _under("1", S(1, 0), 28, WHITE, 0.2)
        g1 = _dash(S(0, 1), S(1, 1))
        tky1 = _tick(S(0, 1), RIGHT)
        l_y1 = tag("1", 28).next_to(S(0, 1), LEFT, buff=0.16)
        dot1 = Dot(S(1, 1), radius=0.06)
        self.play(Create(tk1), FadeIn(l_1), Create(g1), Create(tky1), FadeIn(l_y1),
                  FadeIn(dot1), run_time=0.8)

        def parts(lo, hi, h_in, h_out):
            """outline of the region, inner rectangle, region minus inner,
            outer rectangle outline, outer minus region (over [lo, hi])."""
            xs = np.linspace(lo, hi, 140)
            cp = [S(t, f(t)) for t in xs]
            reg = Polygon(S(lo, 0), *cp, S(hi, 0), stroke_color=YELLOW_B, stroke_width=5)
            inner = mk([S(lo, 0), S(hi, 0), S(hi, h_in), S(lo, h_in)], BLUE_D, 0.85,
                       stroke_width=0)
            if h_in == f(hi):          # x > 0: the curve is above the inner top on the left
                mid = mk([S(lo, h_in)] + cp, TEAL_D, 0.6, stroke_width=0)
                top = mk(cp + [S(hi, h_out)], ORANGE, 0.85, stroke_width=0)
            else:                      # x < 0: the curve falls from h_out to h_in
                mid = mk(cp + [S(lo, h_in)], TEAL_D, 0.6, stroke_width=0)
                top = mk(cp + [S(hi, h_out)], ORANGE, 0.85, stroke_width=0)
            outer_pts = [S(lo, 0), S(hi, 0), S(hi, h_out), S(lo, h_out)]
            outer = Polygon(*outer_pts, stroke_color=WHITE, stroke_width=4)
            return reg, inner, mid, top, outer, outer_pts

        X0 = 1.25
        # ---- x > 0: over [1, 1 + x]
        tku1 = _tick(S(u1, 0))
        l_u1 = _under("1 + x", S(u1, 0), 28, C1, 0.2)
        reg1, in1, mid1, top1, out1, out1_pts = parts(1.0, u1, 1 / u1, 1.0)
        l_h1 = tag("1/(1 + x)", 24, C1).next_to(S(u1, 0.42 / u1), RIGHT, buff=0.14)
        self.play(Create(tku1), FadeIn(l_u1), run_time=0.5)
        self.play(Create(reg1), run_time=0.9)
        self.add(in1)
        self.bring_to_back(in1)
        self.play(GrowFromEdge(in1, DOWN), run_time=0.9)
        self.add(mid1)
        self.bring_to_back(mid1)
        self.play(FadeIn(mid1), FadeIn(l_h1), run_time=0.6)
        self.add(top1)
        self.bring_to_back(top1)
        self.play(Create(out1), FadeIn(top1), run_time=0.9)
        self.bring_to_front(reg1, curve, dot1)
        h1 = tag("x > 0 :", 26, C1)
        r1 = _trow([("x/(1 + x)", BLUE_B), ("<", WHITE), ("ln(1 + x)", YELLOW_B),
                    ("<", WHITE), ("x", WHITE)], 26, 0.17)
        h1.move_to([X0, 3.2, 0.0], aligned_edge=LEFT)
        r1.shift(np.array([X0 + 0.05 - r1.get_left()[0], 2.45, 0.0]))
        self.play(FadeIn(h1), FadeIn(r1), run_time=0.9)
        self.hold(0.6)

        # ---- −1 < x < 0: over [1 + x, 1]
        tku2 = _tick(S(u2, 0))
        l_u2 = _under("1 + x", S(u2, 0), 28, C2, 0.2)
        reg2, in2, mid2, top2, out2, out2_pts = parts(u2, 1.0, 1.0, 1 / u2)
        g2 = _dash(S(0, 1 / u2), S(u2, 1 / u2))
        tky2 = _tick(S(0, 1 / u2), RIGHT)
        l_h2 = tag("1/(1 + x)", 24, C2).next_to(S(0, 1 / u2), LEFT, buff=0.16)
        self.play(Create(tku2), FadeIn(l_u2), run_time=0.5)
        self.play(Create(reg2), run_time=0.9)
        self.add(in2)
        self.bring_to_back(in2)
        self.play(GrowFromEdge(in2, DOWN), run_time=0.9)
        self.add(mid2)
        self.bring_to_back(mid2)
        self.play(FadeIn(mid2), run_time=0.5)
        self.add(top2)
        self.bring_to_back(top2)
        self.play(Create(out2), FadeIn(top2), Create(g2), Create(tky2), FadeIn(l_h2),
                  run_time=0.9)
        self.bring_to_front(reg2, curve, dot1)
        h2 = tag("−1 < x < 0 :", 26, C2)
        r2 = _trow([("−x", BLUE_B), ("<", WHITE), ("−ln(1 + x)", YELLOW_B),
                    ("<", WHITE), ("−x/(1 + x)", WHITE)], 26, 0.17)
        h2.move_to([X0, 1.35, 0.0], aligned_edge=LEFT)
        r2.shift(np.array([X0 + 0.05 - r2.get_left()[0], 0.6, 0.0]))
        self.play(FadeIn(h2), FadeIn(r2), run_time=0.9)
        self.hold(0.6)

        r3 = _trow([("⟹", WHITE), ("x/(1 + x)", WHITE), ("<", WHITE), ("ln(1 + x)", WHITE),
                    ("<", WHITE), ("x", WHITE)], 26, 0.17)
        r3.shift(np.array([X0 - r3.get_left()[0], -0.85, 0.0]))
        self.play(FadeIn(r3), run_time=0.8)

        # ---- checks on the finished figure
        segs = (_segs(cpts) + _segs(ax_x) + _segs(ax_y) + _loop(out1_pts) + _loop(out2_pts)
                + _segs([S(0, 1), S(1, 1)]) + _segs([S(0, 1 / u2), S(u2, 1 / u2)])
                + _loop([S(1, 0), S(u1, 0), S(u1, 1 / u1), S(1, 1 / u1)])
                + _marks(tk1, tky1, tku1, tku2, tky2) + _dot_segs(S(1, 1), 0.06))
        labels = [l_curve, l_1, l_y1, l_u1, l_h1, l_u2, l_h2, h1, r1, h2, r2, r3]
        _labels_ok(labels, segs, "D24")
        content = [axes, curve, out1, out2, *labels]
        _safe(*content)
        cap = caption("x/(1 + x)  <  ln(1 + x)  <  x,     x > −1,  x ≠ 0", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== D23  AM–GM from a tangent line

class D23_TangentLineAMGM(Board):
    """x₁, x₂, x₃ > 0 with arithmetic mean A; put tᵢ = xᵢ/A, so that
    t₁ + t₂ + t₃ = 3: folded over the point 1, the two deviations below 1
    exactly cover the one above.  The curve y = e^(t − 1) lies above its
    tangent y = t at (1, 1) (D14's eˢ ≥ 1 + s with s = t − 1), so each
    tᵢ ≤ e^(tᵢ − 1) (the red gaps).  Multiplying the three, t₁t₂t₃ is at
    most e^(t₁ + t₂ + t₃ − 3) = e⁰ = 1, i.e. x₁x₂x₃ ≤ A³.  Drawn for n = 3;
    the argument is the same for every n."""

    def construct(self):
        t1, t2, t3 = 0.3, 0.6, 2.1
        kx, ky = 2.55, 1.5
        O = np.array([-6.0, -1.6, 0.0])

        def S(t, y):
            return O + np.array([kx * t, ky * y, 0.0])

        def g(t):
            return np.exp(t - 1.0)

        ts = np.array([t1, t2, t3])
        # ---- the claims
        check(abs(ts.sum() - 3.0) < 1e-12, "t₁ + t₂ + t₃ = 3: their mean is 1")
        check(abs((1 - t1) + (1 - t2) - (t3 - 1)) < 1e-12,
              "the deviations below 1 balance the one above")
        tt = np.linspace(-1.0, 3.0, 4001)
        check(bool(np.all(g(tt) >= tt - 1e-15)), "e^(t − 1) ≥ t: the curve is above its tangent")
        check(abs(g(1.0) - 1.0) < 1e-15, "they touch at (1, 1) (slope e⁰ = 1 there)")
        check(bool(np.all(g(ts) > ts)), "each tᵢ < e^(tᵢ − 1)")
        check(abs(np.prod(g(ts)) - 1.0) < 1e-12, "e^(t₁−1) e^(t₂−1) e^(t₃−1) = e⁰ = 1")
        check(np.prod(ts) < 1.0, "t₁ t₂ t₃ < 1")
        for A in (0.7, 4.0, 13.0):
            xs = A * ts
            check(abs(xs.mean() - A) < 1e-12 and np.prod(xs) <= A ** 3,
                  "x₁x₂x₃ ≤ A³ for the numbers xᵢ = A tᵢ")

        t_hi = 2.2
        cxs = np.linspace(0.0, t_hi, 220)
        cpts = [S(t, g(t)) for t in cxs]
        ax_x = [S(-0.1, 0), S(2.45, 0)]
        ax_y = [S(0, -0.05), S(0, 3.45)]
        axes = VGroup(Line(*ax_x, color=GREY_B, stroke_width=2),
                      Line(*ax_y, color=GREY_B, stroke_width=2))
        curve = _curve(cpts, YELLOW_B, 5)
        l_curve = _supline([("y = e", False), ("t − 1", True)], 28, YELLOW_B)
        l_curve.next_to(cpts[-1], RIGHT, buff=0.15)
        tan_pts = [S(0, 0), S(2.2, 2.2)]
        tang = Line(*tan_pts, color=WHITE, stroke_width=3.5)
        l_tang = tag("y = t", 28).next_to(tan_pts[1], RIGHT, buff=0.15)
        dot1 = Dot(S(1, 1), radius=0.065)
        tk1 = _tick(S(1, 0))
        l_1 = _under("1", S(1, 0), 28, WHITE, 0.2)
        self.play(Create(axes), run_time=0.6)
        self.play(Create(curve), FadeIn(l_curve), run_time=1.1)
        self.play(Create(tang), FadeIn(l_tang), FadeIn(dot1), Create(tk1), FadeIn(l_1),
                  run_time=1.0)

        X0 = 0.8
        r0 = _supline([("e", False), ("t − 1", True), ("  ≥  t", False)], 28, GREY_A)
        r0.shift(np.array([X0 - r0.get_left()[0], 2.75, 0.0]))
        self.play(FadeIn(r0), run_time=0.7)

        # ---- the numbers, scaled by their mean A
        r1 = tag("A = (x₁ + x₂ + x₃)/3", 28)
        r1.move_to([X0, 2.05, 0.0], aligned_edge=LEFT)
        r2 = tag("t₁ = x₁/A,   t₂ = x₂/A,   t₃ = x₃/A", 26)
        r2.move_to([X0, 1.4, 0.0], aligned_edge=LEFT)
        self.play(FadeIn(r1), run_time=0.6)
        self.play(FadeIn(r2), run_time=0.6)
        cols = [ORANGE, ORANGE, TEAL_B]
        dots = VGroup(*[Dot(S(t, 0), radius=0.065, color=c) for t, c in zip(ts, cols)])
        tlabs = VGroup(*[_under(s_, S(t, 0), 28, c, 0.2)
                         for s_, t, c in zip(("t₁", "t₂", "t₃"), ts, cols)])
        self.play(FadeIn(dots), FadeIn(tlabs), run_time=0.7)

        # ---- their mean is 1: the deviations below 1, folded over 1, cover the one above
        yb1, yb2, yb3 = O[1] - 0.72, O[1] - 0.98, O[1] - 1.24

        def H(t, yb):
            return np.array([S(t, 0)[0], yb, 0.0])

        barT = Line(H(1, yb1), H(t3, yb1), color=TEAL_B, stroke_width=8)
        barA = Line(H(t1, yb2), H(1, yb2), color=ORANGE, stroke_width=8)
        barB = Line(H(t2, yb3), H(1, yb3), color=ORANGE, stroke_width=8)
        y_top = O[1] - 0.55               # below the tick labels
        drops = VGroup(*[_dash(H(t, y_top), H(t, yb), GREY_B, 1.5, 0.06)
                         for t, yb in ((t1, yb2), (t2, yb3), (1, yb3), (t3, yb1))])
        self.add(drops)
        self.bring_to_back(drops)
        self.play(Create(drops), Create(barT), Create(barA), Create(barB), run_time=1.0)
        # folded over the vertical line through 1: a half-turn in space,
        # rigid on every frame, landing as the mirror image
        self.play(Rotate(barA, PI, axis=UP, about_point=H(1, yb2)),
                  Rotate(barB, PI, axis=UP, about_point=H(1, yb3)), run_time=1.3)
        check(_same_set([barA.get_start(), barA.get_end()], [H(1, yb2), H(2 - t1, yb2)]),
              "1 − t₁, folded over 1, lands on [1, 2 − t₁]")
        self.play(barB.animate.shift(H(2 - t1, yb2) - H(1, yb3)), run_time=1.0)
        check(_same_set([barB.get_start(), barB.get_end()],
                        [H(2 - t1, yb2), H(t3, yb2)]),
              "1 − t₂ continues it exactly to t₃: (1 − t₁) + (1 − t₂) = t₃ − 1")
        drop1 = _dash(H(1, y_top), H(1, yb2), GREY_B, 1.5, 0.06)
        self.play(FadeOut(drops[0]), FadeOut(drops[1]), ReplacementTransform(drops[2], drop1),
                  run_time=0.5)
        ends = VGroup(_dash(H(1, yb1), H(1, yb2), WHITE, 2, 0.05),
                      _dash(H(t3, yb1), H(t3, yb2), WHITE, 2, 0.05))
        r3 = tag("t₁ + t₂ + t₃ = 3", 28)
        r3.move_to([X0, 0.7, 0.0], aligned_edge=LEFT)
        self.play(Create(ends), FadeIn(r3), run_time=0.8)
        self.hold(0.5)

        # ---- each tᵢ lies below e^(tᵢ − 1): the red gaps
        ups = VGroup(*[_dash(S(t, 0), S(t, t), GREY_B, 2) for t in ts])
        gaps = VGroup(*[Line(S(t, t), S(t, g(t)), color=RED_B, stroke_width=7) for t in ts])
        self.add(ups)
        self.bring_to_back(ups)
        self.play(Create(ups), run_time=0.7)
        self.play(LaggedStart(*[Create(m) for m in gaps], lag_ratio=0.3), run_time=1.1)
        self.bring_to_front(curve, tang, dot1, dots)

        # ---- multiply
        r4 = _supline([("t₁ t₂ t₃  ≤  e", False), ("t₁ + t₂ + t₃ − 3", True),
                       ("  =  e", False), ("0", True), ("  =  1", False)], 26, WHITE)
        r4.shift(np.array([X0 - r4.get_left()[0], -0.15, 0.0]))
        r5 = _supline([("⟹   x₁ x₂ x₃  ≤  A", False), ("3", True)], 28, YELLOW_B)
        r5.shift(np.array([X0 - r5.get_left()[0], -0.95, 0.0]))
        self.play(FadeIn(r4), run_time=0.9)
        self.play(FadeIn(r5), run_time=0.8)
        self.hold(0.4)

        # ---- checks on the finished figure
        segs = (_segs(cpts) + _segs(ax_x) + _segs(ax_y) + _segs(tan_pts)
                + sum((_segs([S(t, 0), S(t, g(t))]) for t in ts), [])
                + _segs([H(1, yb1), H(t3, yb1)]) + _segs([H(1, yb2), H(t3, yb2)])
                + _segs([H(1, y_top), H(1, yb2)]) + _segs([H(t3, y_top), H(t3, yb1)])
                + _marks(tk1) + _dot_segs(S(1, 1), 0.065)
                + sum((_dot_segs(S(t, 0), 0.065) for t in ts), []))
        labels = [l_curve, l_tang, l_1, *tlabs, r0, r1, r2, r3, r4, r5]
        _labels_ok(labels, segs, "D23")
        content = [axes, curve, tang, barT, barA, barB, drop1, drops[3], *labels]
        _safe(*content)
        cap = caption("ⁿ√(x₁ x₂ ⋯ xₙ)  ≤  (x₁ + x₂ + ⋯ + xₙ)/n", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== D25  √a + √b ≥ √(a + b)

def _turn(pt, c, n):
    """Math point turned n quarter-turns counter-clockwise about c."""
    x, y = pt[0] - c[0], pt[1] - c[1]
    for _ in range(n % 4):
        x, y = -y, x
    return (x + c[0], y + c[1])


class D25_SqrtSum(Board):
    """A right triangle with legs √a and √b and hypotenuse c.  Four copies,
    each a quarter-turn of the last about the centre of the square of side
    √a + √b, fill its corners around a tilted square of side c.  Slid
    (without turning) into two √a × √b rectangles, the same four triangles
    leave the squares a and b uncovered instead: c² = a + b, c = √(a + b).
    Slid back, the square on the hypotenuse sits inside the square of side
    √a + √b with the four triangles (area 2√(ab)) to spare, so
    √(a + b) < √a + √b; equality needs a or b to be 0."""

    def construct(self):
        p, q = 1.9, 1.15                 # √a and √b
        s = p + q
        k = 1.45
        X0 = np.array([-5.6, -1.71, 0.0])

        def P(pt):
            return X0 + k * np.array([pt[0], pt[1], 0.0])

        cc = (s / 2, s / 2)
        T = {"BL": [(0, 0), (p, 0), (0, q)], "BR": [(s, 0), (s, p), (p, 0)],
             "TR": [(s, s), (q, s), (s, p)], "TL": [(0, s), (0, q), (q, s)]}
        mv = {"TL": (p, -q), "BL": (0, p), "TR": (-q, 0)}

        def tr(pts, v):
            return [(x + v[0], y + v[1]) for x, y in pts]

        tilted = [(p, 0), (s, p), (q, s), (0, q)]
        sqa, sqb = _rect(0, 0, p, p), _rect(p, p, q, q)
        big = _rect(0, 0, s, s)
        c = float(np.hypot(p, q))

        # ---- the claims
        for i, key in enumerate(("BR", "TR", "TL"), start=1):
            check(_same_set([P(_turn(v, cc, i)) for v in T["BL"]], [P(v) for v in T[key]]),
                  f"a quarter-turn ×{i} about the centre carries the triangle to {key}")
        check(_tiles_exactly([T[x] for x in T] + [tilted], big, 100),
              "four triangles and the tilted square tile the big square")
        arr2 = [tr(T[x], mv[x]) for x in mv] + [T["BR"]]
        check(_tiles_exactly(arr2 + [sqa, sqb], big, 100),
              "the slid triangles and the squares a, b tile the big square")
        check(abs(abs(area(tilted)) - (p * p + q * q)) < 1e-12,
              "the tilted square has area c² = a + b")
        check(c < p + q and abs((p + q) ** 2 - c * c - 2 * p * q) < 1e-12,
              "(√a + √b)² = (a + b) + 2√(ab) > a + b")

        # ---- the right triangle
        tri = mk([P(v) for v in T["BL"]], BLUE_D, 0.85)
        ra = _ra(P((0, 0)), P((p, 0)), P((0, q)), 0.2, WHITE, 2.5)
        l_pa = _under("√a", P((p / 2, 0)), 28, WHITE, 0.14)
        l_qb = tag("√b", 28).next_to(P((0, q / 2)), LEFT, buff=0.14)
        l_c = tag("c", 30, YELLOW_B)
        _beside(l_c, P((p, 0)), P((0, q)), P((p, q)) - P((0, 0)), 0.12)
        self.play(FadeIn(tri), Create(ra), run_time=0.9)
        self.play(FadeIn(l_pa), FadeIn(l_qb), FadeIn(l_c), run_time=0.6)
        rider = VGroup(tri, ra, l_qb, l_c)       # what moves with the first triangle

        # ---- three quarter-turns about the centre of the (√a + √b)-square
        copies = {}
        prev = tri
        for i, key in enumerate(("BR", "TR", "TL"), start=1):
            cp = prev.copy()                 # each a quarter-turn of the last
            self.add(cp)
            self.bring_to_front(l_c)
            self.play(Rotate(cp, angle=PI / 2, about_point=P(cc)), run_time=0.9)
            check(_same_set(_verts(cp), [P(v) for v in T[key]]),
                  f"turned copy lands on the {key} corner")
            copies[key] = cp
            prev = cp
        # the sweep of every quarter-turn stays inside the safe area
        rad = max(np.linalg.norm(P(v) - P(cc)) for v in T["BL"])
        check(P(cc)[1] + rad <= SAFE_TOP and P(cc)[1] - rad >= SAFE_BOTTOM
              and P(cc)[0] + rad <= SAFE_X, "the turning copies stay on screen")
        outer = Polygon(*[P(v) for v in big], stroke_color=WHITE, stroke_width=3)
        l_qb2 = _under("√b", P((p + q / 2, 0)), 28, WHITE, 0.14)
        t_out = Polygon(*[P(v) for v in tilted], stroke_color=YELLOW_B, stroke_width=5)
        l_c2 = tag("c²", 34, YELLOW_B).move_to(P(cc))
        self.play(Create(outer), FadeIn(l_qb2), run_time=0.7)
        self.play(Create(t_out), FadeIn(l_c2), run_time=0.8)
        self.hold(0.6)

        # ---- slide three of them (no turning): two rectangles, squares a and b left
        self.play(FadeOut(t_out), FadeOut(l_c2), run_time=0.4)
        for key, grp in (("TL", copies["TL"]), ("BL", rider), ("TR", copies["TR"])):
            v = mv[key]
            self.play(grp.animate.shift(k * np.array([v[0], v[1], 0.0])), run_time=0.9)
            piece = tri if key == "BL" else grp
            check(_same_set(_verts(piece), [P(w) for w in tr(T[key], v)]),
                  f"{key} slides into place")
        oa = Polygon(*[P(v) for v in sqa], stroke_color=YELLOW_B, stroke_width=5)
        ob = Polygon(*[P(v) for v in sqb], stroke_color=YELLOW_B, stroke_width=5)
        l_sa = tag("a", 36, YELLOW_B).move_to(P((p / 2, p / 2)))
        l_sb = tag("b", 36, YELLOW_B).move_to(P((p + q / 2, p + q / 2)))
        self.play(Create(oa), Create(ob), FadeIn(l_sa), FadeIn(l_sb), run_time=0.9)
        R1 = _trow([("c²", YELLOW_B), ("=", WHITE), ("a + b", YELLOW_B)], 30)
        R1.shift(np.array([0.6 - R1.get_left()[0], 2.6, 0.0]))
        self.play(FadeIn(R1), run_time=0.7)
        self.hold(0.9)

        # labels of the slid arrangement, checked while it is on screen
        segs2 = sum((_loop([P(v) for v in tr(T[x], mv[x])]) for x in mv), []) + \
            _loop([P(v) for v in T["BR"]]) + _loop([P(v) for v in big])
        _labels_ok([l_pa, l_qb2, l_qb, l_c, l_sa, l_sb, R1], segs2, "D25 (squares a, b)")

        # ---- slide back: the square on the hypotenuse, area a + b, inside
        self.play(FadeOut(oa), FadeOut(ob), FadeOut(l_sa), FadeOut(l_sb), run_time=0.4)
        for key, grp in (("TR", copies["TR"]), ("BL", rider), ("TL", copies["TL"])):
            self.play(grp.animate.shift(-k * np.array([mv[key][0], mv[key][1], 0.0])),
                      run_time=0.7)
        check(_same_set(_verts(tri), [P(v) for v in T["BL"]])
              and _same_set(_verts(copies["TL"]), [P(v) for v in T["TL"]])
              and _same_set(_verts(copies["TR"]), [P(v) for v in T["TR"]]),
              "back in the corners")
        l_ab = tag("a + b", 34, YELLOW_B).move_to(P(cc))
        l_root = tag("√(a + b)", 28, YELLOW_B)
        _beside(l_root, P((p, 0)), P((0, q)), P((p, q)) - P((0, 0)), 0.12)
        self.play(Create(t_out), FadeIn(l_ab), run_time=0.8)
        self.play(ReplacementTransform(l_c, l_root), run_time=0.8)
        yb = 0.72
        dimS = _dim(P((0, 0)) + DOWN * yb, P((s, 0)) + DOWN * yb, WHITE)
        l_sum = tag("√a + √b", 28).next_to(dimS, DOWN, buff=0.12)
        self.play(Create(dimS), FadeIn(l_sum), run_time=0.7)

        R2 = _trow([("(√a + √b)²", WHITE), ("=", WHITE), ("a + b", YELLOW_B),
                    ("+", WHITE), ("2√(ab)", BLUE_B)], 30)
        R2.shift(np.array([0.6 - R2.get_left()[0], 1.55, 0.0]))
        R3 = _trow([("⟹", WHITE), ("√(a + b)", YELLOW_B), ("<", WHITE), ("√a + √b", WHITE)], 30)
        R3.shift(np.array([0.6 - R3.get_left()[0], 0.55, 0.0]))
        self.play(FadeIn(R2), run_time=0.9)
        self.play(FadeIn(R3), run_time=0.8)

        # ---- checks on the finished figure
        segs = (sum((_loop([P(v) for v in T[x]]) for x in T), [])
                + _loop([P(v) for v in big]) + _loop([P(v) for v in tilted])
                + _mob_segs(dimS[0]) + _mob_segs(ra))
        labels = [l_pa, l_qb, l_qb2, l_root, l_ab, l_sum, R1, R2, R3]
        _labels_ok(labels, segs, "D25")
        _inside(l_ab, [P(v) for v in tilted])
        _inside(l_root, [P(v) for v in tilted])
        content = [outer, dimS, *labels]
        _safe(*content)
        cap = caption("√a + √b  ≥  √(a + b)", 38)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D22  chord shorter than arc

class D22_ChordArc(Board):
    """P and Q on the unit circle at angles x < y (here 20° and 120°); the
    arc between them has length y − x.  The right triangle on the chord PQ
    with legs along the axes has the vertical leg h = sin y − sin x; turned
    about Q onto the chord it stops short of P, since a leg is shorter than
    the hypotenuse.  The arc, straightened like a wire with Q held fixed
    (every stage a circular arc of the same length), runs out along the
    chord's line past P: the chord is shorter than the arc.  So
    |sin x − sin y| ≤ PQ ≤ |x − y|."""

    def construct(self):
        xa, ya = np.radians(20.0), np.radians(120.0)
        R = 2.8
        C = np.array([-3.05, 0.2, 0.0])

        def U(px, py):
            return C + R * np.array([px, py, 0.0])

        P = U(np.cos(xa), np.sin(xa))
        Q = U(np.cos(ya), np.sin(ya))
        K = U(np.cos(ya), np.sin(xa))           # right angle: under Q, level with P
        h = float(np.sin(ya) - np.sin(xa))
        chord = float(2 * np.sin((ya - xa) / 2))
        arc = float(ya - xa)
        d = _unit(P - Q)
        nrm = _perp(d)
        if np.dot(nrm, (P + Q) / 2 - C) < 0:      # pointing away from the centre
            nrm = -nrm

        def wire_pts(kap, m=140):
            """A circular arc of length `arc` (unit-circle units) from Q,
            symmetric about the chord direction, curvature kap (1 = the
            circle itself, 0 = straight)."""
            ss = np.linspace(0.0, arc, m)
            if kap < 1e-6:
                return [Q + R * s_ * d for s_ in ss]
            phi = kap * arc / 2
            t0 = np.cos(phi) * d + np.sin(phi) * nrm
            m0 = np.sin(phi) * d - np.cos(phi) * nrm
            return [Q + R * ((np.sin(kap * s_) / kap) * t0
                             + ((1 - np.cos(kap * s_)) / kap) * m0) for s_ in ss]

        # ---- the claims
        check(abs(np.linalg.norm(Q - P) / R - chord) < 1e-12, "PQ = 2 sin((y − x)/2)")
        check(abs(np.linalg.norm(Q - K) / R - h) < 1e-12
              and abs(np.dot(Q - K, P - K)) < 1e-9, "QK = sin y − sin x, right angle at K")
        check(0 < h < chord < arc, "sin y − sin x < PQ < y − x")
        circ = [U(np.cos(a_), np.sin(a_)) for a_ in np.linspace(ya, xa, 140)]
        check(_same_pts(wire_pts(1.0), circ, 1e-9), "curvature 1: the wire is the arc QP")
        for kap in (1.0, 0.7, 0.4, 0.1, 0.0):
            W = wire_pts(kap, 2000)
            Lw = sum(np.linalg.norm(W[i + 1] - W[i]) for i in range(len(W) - 1)) / R
            check(abs(Lw - arc) < 1e-5, "every stage of the wire has length y − x")
            check(abs(_cross(d, W[-1] - Q)) < 1e-9, "its end stays on the chord's line")
        check(close(wire_pts(0.0)[-1], Q + R * arc * d), "straight: it reaches y − x along QP")
        for a_ in np.linspace(-3.0, 3.0, 25):
            for b_ in np.linspace(-3.0, 3.0, 25):
                check(abs(np.sin(a_) - np.sin(b_)) <= abs(a_ - b_) + 1e-15,
                      "|sin a − sin b| ≤ |a − b|")

        # ---- the unit circle, the angles x and y
        circle = Circle(radius=R, color=GREY_A, stroke_width=3).move_to(C)
        axis = Line(C + LEFT * (R + 0.35), C + RIGHT * (R + 0.35), color=GREY_B,
                    stroke_width=2)
        rP, rQ = Line(C, P, color=GREY_A, stroke_width=2.5), Line(C, Q, color=GREY_A,
                                                                    stroke_width=2.5)
        arc_x = angle_arc(C, C + RIGHT, P, 1.45, BLUE_B, 4)
        arc_y = angle_arc(C, C + RIGHT, Q, 0.45, TEAL_B, 4)
        l_x = tag("x", 28, BLUE_B).move_to(C + 1.85 * np.array([np.cos(xa / 2),
                                                                np.sin(xa / 2), 0.0]))
        l_y = tag("y", 28, TEAL_B).move_to(C + 0.74 * np.array([np.cos(ya / 2),
                                                                np.sin(ya / 2), 0.0]))
        l_one = tag("1", 26, GREY_A)
        _beside(l_one, C, Q, LEFT, 0.1, 0.3)
        dP, dQ = Dot(P, radius=0.07), Dot(Q, radius=0.07)
        l_P = tag("P", 28).next_to(P, UR, buff=0.15)
        l_Q = tag("Q", 28).next_to(Q, UL, buff=0.17)
        self.play(Create(circle), Create(axis), FadeIn(Dot(C, radius=0.05)), run_time=0.9)
        self.play(Create(rP), Create(rQ), FadeIn(dP), FadeIn(dQ), FadeIn(l_P), FadeIn(l_Q),
                  FadeIn(l_one), run_time=0.8)
        self.play(Create(arc_x), Create(arc_y), FadeIn(l_x), FadeIn(l_y), run_time=0.8)

        # ---- the arc QP, of length y − x
        arcm = _curve(wire_pts(1.0), ORANGE, 7)
        mid_a = (xa + ya) / 2
        l_arc = tag("y − x", 28, ORANGE).move_to(
            C + (R + 0.42) * np.array([np.cos(mid_a), np.sin(mid_a), 0.0]))
        self.play(Create(arcm), FadeIn(l_arc), run_time=1.0)
        self.bring_to_front(dP, dQ)

        # ---- the chord and its right triangle: vertical leg h = sin y − sin x
        chord_m = Line(Q, P, color=WHITE, stroke_width=3.5)
        legV = Line(Q, K, color=TEAL_B, stroke_width=6)
        legH = _dash(K, P, GREY_B, 2.5)
        raK = _ra(K, Q, P, 0.2, GREY_B, 2.5)
        l_h = tag("h", 30, TEAL_B).next_to((Q + K) / 2, LEFT, buff=0.14)
        self.play(Create(chord_m), run_time=0.7)
        self.play(Create(legV), Create(legH), Create(raK), FadeIn(l_h), run_time=0.9)
        self.bring_to_front(dP, dQ)
        X0 = 0.75
        r1 = _trow([("h", TEAL_B), ("=", WHITE), ("sin y − sin x", WHITE)], 30)
        r1.shift(np.array([X0 - r1.get_left()[0], -0.75, 0.0]))
        self.play(FadeIn(r1), run_time=0.7)

        # ---- turned about Q onto the chord, the leg stops short of P
        swing = Line(Q, K, color=TEAL_B, stroke_width=9)
        ang = float(np.arctan2(d[1], d[0]) - np.arctan2(*(K - Q)[1::-1]))
        ang = (ang + PI) % TAU - PI
        self.add(swing)
        self.play(Rotate(swing, angle=ang, about_point=Q), run_time=1.3)
        check(close(swing.get_start(), Q, 1e-6) and close(swing.get_end(), Q + R * h * d, 1e-6),
              "the leg h, turned about Q, lies along QP")
        check(np.linalg.norm(swing.get_end() - Q) < np.linalg.norm(P - Q),
              "and ends before P")
        tick_h = Line(Q + R * h * d - 0.16 * nrm, Q + R * h * d + 0.16 * nrm,
                      color=TEAL_B, stroke_width=4)
        self.play(Create(tick_h), run_time=0.4)
        r2 = _trow([("h", TEAL_B), ("≤", WHITE), ("PQ", WHITE)], 30)
        r2.shift(np.array([X0 - r2.get_left()[0], -1.55, 0.0]))
        self.play(FadeIn(r2), run_time=0.6)

        # ---- the arc, straightened with Q held, runs past P
        wire = _curve(wire_pts(1.0), ORANGE, 11)
        self.add(wire)
        self.bring_to_back(wire)
        self.bring_to_back(circle)

        def bend(m, a_):
            m.set_points_as_corners([to3(v) for v in wire_pts(1.0 - a_)])

        self.play(UpdateFromAlphaFunc(wire, bend), run_time=2.4, rate_func=smooth)
        Wend = Q + R * arc * d
        check(close(wire.get_end(), Wend, 1e-6) and close(wire.get_start(), Q, 1e-6),
              "straightened: from Q along QP to length y − x")
        check(np.linalg.norm(Wend - Q) > np.linalg.norm(P - Q), "it runs past P")
        tick_w = Line(Wend - 0.16 * nrm, Wend + 0.16 * nrm, color=ORANGE, stroke_width=4)
        self.play(Create(tick_w), run_time=0.4)
        self.bring_to_front(chord_m, swing, tick_h, dP, dQ)
        r3 = _trow([("h", TEAL_B), ("≤", WHITE), ("PQ", WHITE), ("≤", WHITE),
                    ("y − x", ORANGE)], 30)
        r3.shift(np.array([X0 - r3.get_left()[0], -1.55, 0.0]))
        self.play(ReplacementTransform(r2, r3), run_time=0.8)
        self.hold(0.5)

        # ---- checks on the finished figure
        segs = (_circle_segs(C, R) + _segs([C + LEFT * (R + 0.35), C + RIGHT * (R + 0.35)])
                + _segs([C, P]) + _segs([C, Q]) + _segs([Q, P]) + _segs([Q, K, P])
                + _segs([Q, Wend]) + _segs(wire_pts(1.0)) + _mob_segs(arc_x)
                + _mob_segs(arc_y) + _mob_segs(raK)
                + _segs([Q + R * h * d - 0.16 * nrm, Q + R * h * d + 0.16 * nrm])
                + _segs([Wend - 0.16 * nrm, Wend + 0.16 * nrm]))
        segs += _dot_segs(P, 0.07) + _dot_segs(Q, 0.07) + _dot_segs(C, 0.05)
        labels = [l_x, l_y, l_one, l_P, l_Q, l_arc, l_h, r1, r3]
        _labels_ok(labels, segs, "D22")
        content = [circle, axis, wire, *labels]
        _safe(*content)
        cap = caption("|sin x − sin y|  ≤  |x − y|", 38)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D26  Minkowski

class D26_Minkowski(Board):
    """u = (a, b) and v = (c, d) laid head to tail run from O through the
    bend U to W = (a + c, b + d).  By Pythagoras on the dashed legs,
    |u| = √(a² + b²), |v| = √(c² + d²), |u + v| = √((a + c)² + (b + d)²).
    Drop the perpendicular from the bend U to OW, foot F: in the right
    triangles OFU and WFU a leg is shorter than the hypotenuse, so u, swung
    about O onto OW, ends beyond F, and v, swung about W, ends short of F.
    Together they cover OW with a red overlap: |u| + |v| = |u + v| +
    overlap.  (Drawn with a, b, c, d > 0 and F between O and W; if F falls
    outside, one side alone is already longer than OW.)"""

    def construct(self):
        a, b, c, d = 3.2, 1.0, 1.4, 2.6
        k = 1.12
        O = np.array([-5.9, -1.75, 0.0])

        def S(x, y):
            return O + k * np.array([x, y, 0.0])

        U, W = S(a, b), S(a + c, b + d)
        Lu, Lv, Lw = float(np.hypot(a, b)), float(np.hypot(c, d)), float(np.hypot(a + c, b + d))
        e = _unit(W - O)
        F = _foot(U, O, W)
        U1 = O + e * np.linalg.norm(U - O)          # u swung about O onto OW
        U2 = W - e * np.linalg.norm(W - U)          # v swung about W onto OW

        def along(p):
            return float(np.dot(to3(p) - O, e))

        # ---- the claims
        check(abs(np.linalg.norm(U - O) / k - Lu) < 1e-12
              and abs(np.linalg.norm(W - U) / k - Lv) < 1e-12
              and abs(np.linalg.norm(W - O) / k - Lw) < 1e-12,
              "the three lengths are √(a² + b²), √(c² + d²), √((a + c)² + (b + d)²)")
        check(0 < along(F) < np.linalg.norm(W - O), "the foot F lies between O and W")
        check(abs(np.dot(U - F, e)) < 1e-12, "UF ⟂ OW")
        check(np.linalg.norm(F - O) < np.linalg.norm(U - O)
              and np.linalg.norm(W - F) < np.linalg.norm(W - U),
              "a leg is shorter than the hypotenuse")
        check(along(U2) < along(F) < along(U1), "swung, u ends beyond F and v short of F")
        check(abs(np.linalg.norm(U1 - U2) / k - (Lu + Lv - Lw)) < 1e-9 and Lu + Lv > Lw,
              "overlap = |u| + |v| − |u + v| > 0")
        for (p, q, r, s_) in ((1, 2, 3, 4), (-1, 2, 0.5, -3), (2, -1, 2, -1)):
            check(np.hypot(p, q) + np.hypot(r, s_) >= np.hypot(p + r, q + s_) - 1e-12,
                  "Minkowski for other signs too")

        # ---- u = (a, b), with its legs
        Pu, Pv = S(a, 0), S(a + c, b)
        au = _arrow(O, U, BLUE_D)
        legs_u = VGroup(_dash(O, Pu, BLUE_B, 2.5), _dash(Pu, U, BLUE_B, 2.5),
                        _ra(Pu, O, U, 0.16, BLUE_B, 2))
        l_a = tag("a", 30, BLUE_B).next_to((O + Pu) / 2, DOWN, buff=0.14)
        l_b = tag("b", 30, BLUE_B).next_to((Pu + U) / 2, RIGHT, buff=0.14)
        l_u = tag("u", 32, BLUE_B)
        _beside(l_u, O, U, DOWN, 0.12, 0.62)
        dO = Dot(O, radius=0.06)
        self.play(FadeIn(dO), GrowArrow(au), run_time=1.0)
        self.play(Create(legs_u), FadeIn(l_a), FadeIn(l_b), FadeIn(l_u), run_time=0.9)

        # ---- v = (c, d) from the head of u
        av = _arrow(U, W, TEAL_D)
        legs_v = VGroup(_dash(U, Pv, TEAL_B, 2.5), _dash(Pv, W, TEAL_B, 2.5),
                        _ra(Pv, U, W, 0.16, TEAL_B, 2))
        l_c = tag("c", 30, TEAL_B).next_to((U + Pv) / 2, DOWN, buff=0.14)
        l_d = tag("d", 30, TEAL_B).next_to((Pv + W) / 2, RIGHT, buff=0.14)
        l_v = tag("v", 32, TEAL_B)
        _beside(l_v, U, W, RIGHT, 0.14, 0.62)
        self.play(GrowArrow(av), run_time=1.0)
        self.play(Create(legs_v), FadeIn(l_c), FadeIn(l_d), FadeIn(l_v), run_time=0.9)

        # ---- the single arrow u + v: run a + c, rise b + d
        aw = _arrow(O, W, YELLOW_B, 6)
        l_w = tag("u + v", 30, YELLOW_B)
        _beside(l_w, O, W, UP, 0.16, 0.8)
        off_h, off_v = 0.78, 0.62
        dimH = _dim(S(0, 0) + DOWN * off_h, S(a + c, 0) + DOWN * off_h, YELLOW_B)
        dimV = _dim(S(a + c, 0) + RIGHT * off_v, W + RIGHT * off_v, YELLOW_B)
        l_ac = tag("a + c", 28, YELLOW_B).next_to(dimH, DOWN, buff=0.1)
        l_bd = tag("b + d", 28, YELLOW_B).next_to(
            S(a + c, 0.5 * (b + d)) + RIGHT * off_v, RIGHT, buff=0.14)
        guides = VGroup(_dash(W, S(a + c, 0) + DOWN * (off_h + 0.1)),
                        _dash(Pu, S(a + c, 0) + RIGHT * (off_v + 0.1)),
                        _dash(W, W + RIGHT * (off_v + 0.1)))
        self.play(GrowArrow(aw), FadeIn(l_w), run_time=1.0)
        self.add(guides)
        self.bring_to_back(guides)
        self.play(Create(guides), Create(dimH), Create(dimV), FadeIn(l_ac), FadeIn(l_bd),
                  run_time=1.0)
        self.bring_to_front(au, av, aw, dO)

        X0 = 1.0
        rows = [_trow([("|u|", BLUE_B), ("=", WHITE), ("√(a² + b²)", BLUE_B)], 26),
                _trow([("|v|", TEAL_B), ("=", WHITE), ("√(c² + d²)", TEAL_B)], 26),
                _trow([("|u + v|", YELLOW_B), ("=", WHITE),
                       ("√((a + c)² + (b + d)²)", YELLOW_B)], 26)]
        for r, y in zip(rows, (3.15, 2.5, 1.85)):
            r.shift(np.array([X0 - r.get_left()[0], y, 0.0]))
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.5), run_time=1.5)
        self.hold(0.5)

        # ---- the perpendicular from the bend; swing u about O and v about W
        perp = _dash(U, F, GREY_A, 2.5)
        raF = _ra(F, U, W, 0.16, GREY_A, 2)
        self.play(Create(perp), Create(raF), run_time=0.7)
        su = Line(O, U, color=BLUE_B, stroke_width=9)
        ang_u = float(np.arctan2(e[1], e[0]) - np.arctan2(*(U - O)[1::-1]))
        arc_u = DashedVMobject(Arc(radius=float(np.linalg.norm(U - O)),
                                   start_angle=float(np.arctan2(*(U - O)[1::-1])),
                                   angle=ang_u, arc_center=O, color=BLUE_B,
                                   stroke_width=2), num_dashes=10)
        self.add(su)
        self.play(Rotate(su, angle=ang_u, about_point=O), Create(arc_u), run_time=1.3)
        check(close(su.get_start(), O, 1e-6) and close(su.get_end(), U1, 1e-6),
              "u swung about O lies along OW and ends beyond F")
        sv = Line(W, U, color=TEAL_B, stroke_width=9)
        a0 = float(np.arctan2(*(U - W)[1::-1]))
        a1 = float(np.arctan2(*(O - W)[1::-1]))
        ang_v = (a1 - a0 + PI) % TAU - PI
        arc_v = DashedVMobject(Arc(radius=float(np.linalg.norm(U - W)), start_angle=a0,
                                   angle=ang_v, arc_center=W, color=TEAL_B,
                                   stroke_width=2), num_dashes=10)
        self.add(sv)
        self.play(Rotate(sv, angle=ang_v, about_point=W), Create(arc_v), run_time=1.3)
        check(close(sv.get_start(), W, 1e-6) and close(sv.get_end(), U2, 1e-6),
              "v swung about W lies along OW and ends short of F")
        n = _perp(e)
        self.play(su.animate.shift(n * 0.12), sv.animate.shift(-n * 0.12), run_time=0.5)
        ovl = Line(U2, U1, color=RED_B, stroke_width=11)
        self.play(Create(ovl), run_time=0.6)
        self.bring_to_front(perp, raF)

        bar = Line(ORIGIN, RIGHT * 0.55, color=RED_B, stroke_width=11)
        r4 = _trow([("|u| + |v|", WHITE), ("=", WHITE), ("|u + v|", YELLOW_B),
                    ("+", WHITE)], 26)
        bar.next_to(r4, RIGHT, buff=0.2)
        bar.set_y(r4[0].get_center()[1])
        row4 = VGroup(r4, bar)
        row4.shift(np.array([X0 - row4.get_left()[0], 0.9 - row4.get_center()[1], 0.0]))
        self.play(FadeIn(row4), run_time=0.8)
        self.hold(0.5)

        # ---- checks on the finished figure
        segs = (_segs([O, U]) + _segs([U, W]) + _segs([O, W]) + _segs([O, Pu, U])
                + _segs([U, Pv, W]) + _mob_segs(dimH[0]) + _mob_segs(dimV[0])
                + _segs([W, S(a + c, 0) + DOWN * (off_h + 0.1)])
                + _segs([Pu, S(a + c, 0) + RIGHT * (off_v + 0.1)])
                + _segs([W, W + RIGHT * (off_v + 0.1)]) + _segs([U, F])
                + _mob_segs(arc_u) + _mob_segs(arc_v)
                + _segs([O + n * 0.12, U1 + n * 0.12]) + _segs([W - n * 0.12, U2 - n * 0.12]))
        segs += (_marks(legs_u[2], legs_v[2], raF, dimH, dimV) + _dot_segs(O, 0.06)
                 + _segs([U2, U1]))
        labels = [l_a, l_b, l_u, l_c, l_d, l_v, l_w, l_ac, l_bd, *rows, row4]
        _labels_ok(labels, segs, "D26")
        content = [au, av, aw, dimH, dimV, guides, su, sv, *labels]
        _safe(*content)
        cap = caption("√(a² + b²) + √(c² + d²)  ≥  √((a + c)² + (b + d)²)", 34)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== D27  (a + b)(1/a + 1/b) ≥ 4

class D27_SumTimesReciprocals(Board):
    """The rectangle with sides a + b and 1/a + 1/b (a, b > 0; drawn with
    a > b) splits into a × 1/a and b × 1/b, of area 1 each, and a × 1/b,
    b × 1/a, of areas a/b and b/a.  Copies of the two unit pieces, the
    second cut in two, slide onto the other two pieces and fit inside them
    with an (a − b) × (1/b − 1/a) corner to spare:
    a/b + b/a = 1 + 1 + (a − b)(1/b − 1/a) ≥ 2.  So
    (a + b)(1/a + 1/b) = 2 + a/b + b/a ≥ 4, with equality only when a = b
    (then the corner closes)."""

    def construct(self):
        a, b = 2.75, 1.1
        ia, ib = 1 / a, 1 / b
        k = 2.55
        X0 = np.array([-5.05, 0.15, 0.0])          # bottom-left corner

        def P(pt):
            return X0 + k * np.array([pt[0], pt[1], 0.0])

        def PP(r):
            return [P(v) for v in r]

        big = _rect(0, 0, a + b, ia + ib)
        R1 = _rect(0, 0, a, ia)               # a × 1/a: area 1
        R4 = _rect(a, 0, b, ia)               # b × 1/a: area b/a
        R3 = _rect(0, ia, a, ib)              # a × 1/b: area a/b
        R2 = _rect(a, ia, b, ib)              # b × 1/b: area 1
        R1_to = _rect(0, ia, a, ia)           # R1 slid up by 1/a
        R2lo, R2hi = _rect(a, ia, b, ia), _rect(a, 2 * ia, b, ib - ia)
        R2lo_to, R2hi_to = _rect(a, 0, b, ia), _rect(0, 2 * ia, b, ib - ia)
        corner = _rect(b, 2 * ia, a - b, ib - ia)

        # ---- the claims
        check(abs(area(R1) - 1) < 1e-12 and abs(area(R2) - 1) < 1e-12
              and abs(area(R3) - a / b) < 1e-12 and abs(area(R4) - b / a) < 1e-12,
              "pieces of areas 1, 1, a/b, b/a")
        check(_tiles_exactly([R1, R2, R3, R4], big, 90), "they tile the big rectangle")
        check(abs(area(big) - (a + b) * (ia + ib)) < 1e-12
              and abs((a + b) * (ia + ib) - (2 + a / b + b / a)) < 1e-12,
              "(a + b)(1/a + 1/b) = 2 + a/b + b/a")
        check(_tiles_exactly([R2lo, R2hi], R2, 60), "the cut splits b × 1/b in two")
        check(_same_set(R2lo_to, R4), "the lower part lands exactly on b × 1/a")
        check(_tiles_exactly([R1_to, R2hi_to, corner], R3, 90),
              "the rest fills a × 1/b except the (a − b) × (1/b − 1/a) corner")
        check(ia < ib and b < a and abs(area(corner) - (a - b) * (ib - ia)) < 1e-12,
              "the corner is a genuine rectangle (a > b ⟺ 1/b > 1/a)")
        check(abs(a / b + b / a - 2 - (a - b) * (ib - ia)) < 1e-12,
              "a/b + b/a = 2 + (a − b)(1/b − 1/a)")

        # ---- the (a + b) × (1/a + 1/b) rectangle and its four pieces
        frame = Polygon(*PP(big), stroke_color=WHITE, stroke_width=4)
        cuts = VGroup(Line(P((a, 0)), P((a, ia + ib)), color=WHITE, stroke_width=2.5),
                      Line(P((0, ia)), P((a + b, ia)), color=WHITE, stroke_width=2.5))
        l_a = _under("a", P((a / 2, 0)), 30, WHITE, 0.15)
        l_b = _under("b", P((a + b / 2, 0)), 30, WHITE, 0.15)
        l_ia = tag("1/a", 28).next_to(P((0, ia / 2)), LEFT, buff=0.16)
        l_ib = tag("1/b", 28).next_to(P((0, ia + ib / 2)), LEFT, buff=0.16)
        self.play(Create(frame), Create(cuts), run_time=1.0)
        self.play(FadeIn(l_a), FadeIn(l_b), FadeIn(l_ia), FadeIn(l_ib), run_time=0.6)

        p1, p2 = mk(PP(R1), BLUE_D, 0.85, stroke_width=0), mk(PP(R2), BLUE_D, 0.85, stroke_width=0)
        p3, p4 = mk(PP(R3), ORANGE, 0.8, stroke_width=0), mk(PP(R4), ORANGE, 0.8, stroke_width=0)
        l1 = tag("1", 34).move_to(P((a / 2, ia / 2)))
        l2 = tag("1", 34).move_to(P((a + b / 2, (3 * ia + ib) / 2)))
        l3 = tag("a/b", 32).move_to(P((0.32 * a, ia + 0.5 * ia)))
        l4 = tag("b/a", 32).move_to(P((a + b / 2, ia / 2)))
        self.add(p1, p2)
        self.bring_to_back(p1, p2)
        self.play(FadeIn(p1), FadeIn(p2), FadeIn(l1), FadeIn(l2), run_time=0.8)
        self.add(p3, p4)
        self.bring_to_back(p3, p4)
        self.play(FadeIn(p3), FadeIn(p4), FadeIn(l3), FadeIn(l4), run_time=0.8)

        X1 = -5.75
        r1 = _trow([("(a + b)(1/a + 1/b)", WHITE), ("=", WHITE), ("1 + 1", BLUE_B),
                    ("+", WHITE), ("a/b + b/a", ORANGE)], 30)
        r1.shift(np.array([X1 - r1.get_left()[0], -0.95, 0.0]))
        self.play(FadeIn(r1), run_time=0.8)
        self.hold(0.5)

        # ---- the unit pieces, laid over the other two
        c1 = mk(PP(R1), BLUE_D, 0.6, stroke_color=WHITE, stroke_width=3.5)
        self.add(c1)
        self.bring_to_front(l3, l4)
        self.play(c1.animate.shift(P((0, ia)) - P((0, 0))), run_time=1.1)
        check(_same_set(_verts(c1), PP(R1_to)), "a × 1/a slides up onto the bottom of a × 1/b")
        cut = DashedLine(P((a, 2 * ia)), P((a + b, 2 * ia)), color=WHITE, stroke_width=3,
                         dash_length=0.1)
        self.play(Create(cut), run_time=0.5)
        c2lo = mk(PP(R2lo), BLUE_D, 0.6, stroke_color=WHITE, stroke_width=3.5)
        c2hi = mk(PP(R2hi), BLUE_D, 0.6, stroke_color=WHITE, stroke_width=3.5)
        self.add(c2lo, c2hi)
        self.bring_to_front(l2, l3, l4)
        self.play(c2lo.animate.shift(P((0, 0)) - P((0, ia))), run_time=1.0)
        check(_same_set(_verts(c2lo), PP(R2lo_to)), "the lower part covers b × 1/a exactly")
        self.play(c2hi.animate.shift(P((0, 0)) - P((a, 0))), run_time=1.1)
        check(_same_set(_verts(c2hi), PP(R2hi_to)), "the upper part slides into a × 1/b")
        self.bring_to_front(l1, l2, l3, l4)

        cfr = Polygon(*PP(corner), stroke_color=RED_B, stroke_width=5)
        l_c = tag("(a − b)(1/b − 1/a)", 24, WHITE).move_to(P((b + (a - b) / 2,
                                                            2 * ia + (ib - ia) / 2)))
        self.play(Create(cfr), FadeIn(l_c), run_time=0.8)
        r2 = _trow([("a/b + b/a", ORANGE), ("=", WHITE), ("1 + 1", BLUE_B), ("+", WHITE),
                    ("(a − b)(1/b − 1/a)", RED_B)], 30)
        r2.shift(np.array([X1 - r2.get_left()[0], -1.75, 0.0]))
        self.play(FadeIn(r2), run_time=0.8)
        r3 = _trow([("⟹", WHITE), ("(a + b)(1/a + 1/b)", WHITE), ("≥", WHITE),
                    ("1 + 1 + 1 + 1  =  4", WHITE)], 30)
        r3.shift(np.array([X1 - r3.get_left()[0], -2.55, 0.0]))
        self.play(FadeIn(r3), run_time=0.8)
        self.hold(0.4)

        # ---- checks on the finished figure
        segs = (_loop(PP(big)) + _segs([P((a, 0)), P((a, ia + ib))])
                + _segs([P((0, ia)), P((a + b, ia))]) + _loop(PP(R1_to)) + _loop(PP(R2hi_to))
                + _loop(PP(corner)) + _segs([P((a, 2 * ia)), P((a + b, 2 * ia))]))
        labels = [l_a, l_b, l_ia, l_ib, l1, l2, l3, l4, l_c, r1, r2, r3]
        _labels_ok(labels, segs, "D27")
        _inside(l1, PP(R1))
        _inside(l2, PP(R2))
        _inside(l3, PP(R1_to))
        _inside(l4, PP(R4))
        _inside(l_c, PP(corner))
        content = [frame, *labels]
        _safe(*content)
        cap = caption("(a + b)(1/a + 1/b)  ≥  4,     a, b > 0", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== D28  ab + bc + ca ≤ a² + b² + c²

class D28_ThreeProducts(Board):
    """D8 for each pair of a > b > c > 0.  On the larger square x², the two
    x × y rectangles laid along two of its sides overlap exactly in a
    y × y corner — the place of the square y² — and leave the opposite
    corner (x − y)² uncovered: x² + y² = 2xy + (x − y)².  For the pairs
    (a, b), (b, c), (c, a) the squares a², b², c² are each used twice and
    the rectangles ab, bc, ca twice:
    2(a² + b² + c²) = 2(ab + bc + ca) + (a − b)² + (b − c)² + (a − c)²,
    and halving gives ab + bc + ca ≤ a² + b² + c² (equality iff a = b = c)."""

    def construct(self):
        a, b, c = 2.55, 1.45, 0.6
        k = 1.6
        gap = 1.2
        yb = -0.82
        pairs = [("a", "b", a, b), ("b", "c", b, c), ("c", "a", a, c)]
        widths = [x * k for (_, _, x, _) in pairs]
        total = sum(widths) + 2 * gap
        lefts = [-total / 2, -total / 2 + widths[0] + gap,
                 -total / 2 + widths[0] + widths[1] + 2 * gap]

        # ---- the claims
        for (n1, n2, x, y) in pairs:
            check(x > y > 0, "the larger square comes first")
            A = _rect(0, 0, x, x)
            T = _rect(0, x - y, x, y)            # rectangle along the top
            Rr = _rect(x - y, 0, y, x)           # rectangle up the right side
            ov = _rect(x - y, x - y, y, y)
            cor = _rect(0, 0, x - y, x - y)
            check(_tiles_exactly([T, _rect(x - y, 0, y, x - y), cor], A, 80),
                  "the two rectangles (overlap counted once) and the corner tile x²")
            check(abs(area(T) + area(Rr) - (area(A) - area(cor) + area(ov))) < 1e-12
                  and abs(area(ov) - y * y) < 1e-12,
                  "they overlap in exactly a y × y square: 2xy = x² + y² − (x − y)²")
            check(abs(x * x + y * y - 2 * x * y - (x - y) ** 2) < 1e-12, "x² + y² = 2xy + (x − y)²")
        lhs = 2 * (a * a + b * b + c * c)
        rhs = 2 * (a * b + b * c + c * a) + (a - b) ** 2 + (b - c) ** 2 + (a - c) ** 2
        check(abs(lhs - rhs) < 1e-12, "the three panels add up")
        check(a * b + b * c + c * a < a * a + b * b + c * c, "ab + bc + ca < a² + b² + c²")

        panels = []
        for i, ((n1, n2, x, y), x0) in enumerate(zip(pairs, lefts)):
            org = np.array([x0, yb, 0.0])

            def P(pt, org=org):
                return org + k * np.array([pt[0], pt[1], 0.0])

            big_n, small_n = (n2, n1) if n1 == "c" else (n1, n2)
            prod = n1 + n2
            sq = mk([P(v) for v in _rect(0, 0, x, x)], BLUE_D, 0.85)
            sm = mk([P(v) for v in _rect(x - y, x - y, y, y)], ORANGE, 0.9)
            l_big = tag(f"{big_n}²", 34).move_to(P(((x - y) / 2, (x - y) / 2)))
            l_sm = tag(f"{small_n}²", 28).move_to(P((x - y / 2, x - y / 2)))
            rims = VGroup(Polygon(*[P(v) for v in _rect(0.03, x - y + 0.03, x - 0.06, y - 0.06)],
                                  stroke_color=TEAL_A, stroke_width=5),
                          Polygon(*[P(v) for v in _rect(x - y + 0.07, 0.07, y - 0.14, x - 0.14)],
                                  stroke_color=GREEN_A, stroke_width=5))
            l_r1 = tag(prod, 28).move_to(P(((x - y) / 2, x - y / 2)))
            l_r2 = tag(prod, 28).move_to(P((x - y / 2, (x - y) / 2)))
            cfr = Polygon(*[P(v) for v in _rect(0, 0, x - y, x - y)], stroke_color=RED_B,
                          stroke_width=5)
            l_cor = tag(f"({big_n}−{small_n})²", 26).move_to(P(((x - y) / 2, (x - y) / 2)))
            eq = tag(f"{n1}² + {n2}² = 2{prod} + ({big_n}−{small_n})²", 22)
            eq.move_to(np.array([x0 + x * k / 2, yb - 0.42, 0.0]))
            panels.append(dict(P=P, x=x, y=y, sq=sq, sm=sm, l_big=l_big, l_sm=l_sm,
                               rims=rims, l_r1=l_r1, l_r2=l_r2, cfr=cfr, l_cor=l_cor, eq=eq))

        # ---- each panel: D8 for one pair
        for i, pn in enumerate(panels):
            rt = 1.0 if i == 0 else 0.75
            self.play(FadeIn(pn["sq"]), FadeIn(pn["l_big"]), run_time=0.6 * rt)
            self.play(FadeIn(pn["sm"], shift=DOWN * 0.35), FadeIn(pn["l_sm"]),
                      run_time=0.7 * rt)
            x, y, P = pn["x"], pn["y"], pn["P"]
            check(_same_set(_verts(pn["sm"]), [P(v) for v in _rect(x - y, x - y, y, y)]),
                  "the small square sits exactly on the overlap of the two rectangles")
            self.play(FadeOut(pn["l_big"]), Create(pn["rims"]), FadeIn(pn["l_r1"]),
                      FadeIn(pn["l_r2"]), run_time=0.9 * rt)
            self.play(Create(pn["cfr"]), FadeIn(pn["l_cor"]), run_time=0.6 * rt)
            self.play(FadeIn(pn["eq"]), run_time=0.6 * rt)

        s1 = tag("2(a² + b² + c²) = 2(ab + bc + ca) + (a−b)² + (b−c)² + (a−c)²", 28)
        s1.move_to([0.0, -1.95, 0.0])
        s2 = tag("⟹  a² + b² + c² ≥ ab + bc + ca", 28, YELLOW_B)
        s2.move_to([0.0, -2.62, 0.0])
        self.play(FadeIn(s1), run_time=1.0)
        self.play(FadeIn(s2), run_time=0.8)
        self.hold(0.4)

        # ---- checks on the finished figure
        segs, labels = [], [s1, s2]
        for pn in panels:
            x, y, P = pn["x"], pn["y"], pn["P"]
            segs += (_loop([P(v) for v in _rect(0, 0, x, x)])
                     + _loop([P(v) for v in _rect(x - y, x - y, y, y)])
                     + sum((_mob_segs(r) for r in pn["rims"]), [])
                     + _loop([P(v) for v in _rect(0, 0, x - y, x - y)]))
            labels += [pn["l_sm"], pn["l_r1"], pn["l_r2"], pn["l_cor"], pn["eq"]]
            _inside(pn["l_cor"], [P(v) for v in _rect(0, 0, x - y, x - y)])
            _inside(pn["l_sm"], [P(v) for v in _rect(x - y, x - y, y, y)])
            _inside(pn["l_r1"], [P(v) for v in _rect(0, x - y, x - y, y)])
            _inside(pn["l_r2"], [P(v) for v in _rect(x - y, 0, y, x - y)])
        _labels_ok(labels, segs, "D28")
        for p1_, p2_ in zip(panels, panels[1:]):
            check(p2_["eq"].get_left()[0] - p1_["eq"].get_right()[0] > 0.5,
                  "the panel equations stand clearly apart")
        content = [pn["sq"] for pn in panels] + labels
        _safe(*content)
        cap = caption("ab + bc + ca  ≤  a² + b² + c²", 38)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)
