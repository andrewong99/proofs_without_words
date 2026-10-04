# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3f.py — proofs without words, 2D (manim):
#     G29 double angle in a semicircle      G31 sides, area and circumradius
#     G25 the tangent half-angle formulas   G30 sin x + cos x is at most √2
#     H24 the area under a cycloid          H11 derivative of the square root
#     H28 the shadow of a semicircle        H31 the arc-length element
#     H12 derivative of 1/x                 H30 derivative of tan
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Every move of a piece is rigid (a shift, a Rotate about a point, or a fold
# about an in-plane line, which is a half-turn in space) unless it is an
# announced enlargement (a homothety, drawn as a turn-and-scale about its
# centre).  Every landing, length and area claim is checked numerically with
# check(...) before or right after it is drawn, and at the end of each scene
# the labels are checked against the lines, marks and arcs they must clear,
# against each other, and against the safe area, so a wrong construction
# fails the render instead of rendering quietly into a wrong picture.
# The derivative scenes show an exact finite picture next to a window
# rescaled by the independent increment, in which the error terms visibly
# vanish as the increment shrinks.


# ------------------------------------------------------------ geometry helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _polar(a):
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


def _mirror(p, a, b):
    """Mirror image of p in the line ab."""
    return 2 * _foot(p, a, b) - to3(p)


def _rot2(p, c, th):
    """Point p turned by th about c (in the plane)."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments (label checks)."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


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


def _tick_segs(p, q, n=1, size=0.13, at=0.5):
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    return [(c + d * 0.09 * (k - (n - 1) / 2) - nrm * size,
             c + d * 0.09 * (k - (n - 1) / 2) + nrm * size) for k in range(n)]


def _wedge(vertex, p, q, r, color, op=0.6):
    """Filled sector for the NON-REFLEX angle p-vertex-q (cf. angle_arc)."""
    v = to3(vertex)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return Sector(radius=r, angle=span, start_angle=a1, arc_center=v,
                  fill_color=color, fill_opacity=op, stroke_width=0)


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on the
    side the vector `side` points to, its box clear of the line by `gap`."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


def _curve(pts, color=WHITE, width=4):
    """Open polyline through screen points (fine sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _region(pts, color, op=FILL, stroke=0.0, stroke_color=WHITE):
    """Filled polygon through densely sampled screen points (a curved region)."""
    return Polygon(*[to3(p) for p in pts], fill_color=color, fill_opacity=op,
                   stroke_color=stroke_color, stroke_width=stroke)


def _dedup(pts, eps=1e-12):
    out = []
    for p in pts:
        p = to3(p)
        if not out or np.linalg.norm(p - out[-1]) > eps:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps:
        out.pop()
    return out


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


def _homothety(mob, centre, factor, turn=0.0, **kw):
    """Enlargement about `centre`, drawn as a turn by `turn` together with a
    scaling that reaches `factor` (geometrically in time)."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=c)
                 .scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


def _landed(mob, pts, tol=1e-6):
    """The polygon mob's vertices are pts (same order)."""
    v = mob.get_vertices()
    return len(v) == len(pts) and all(close(a, to3(b), tol)
                                      for a, b in zip(v, pts))


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
    P = [to3(p)[:2] for p in P]
    Q = [to3(q)[:2] for q in Q]
    if len(P) != len(Q):
        return False
    n = len(P)
    for d in (1, -1):
        for s in range(n):
            if all(close(P[i], Q[(s + d * i) % n], tol) for i in range(n)):
                return True
    return False


# ------------------------------------------------------------ text helpers

def _on_base(s, x, base_y, size=28, color=WHITE):
    """A label centred on x whose baseline sits at base_y, so a row of labels
    shares one baseline whatever their ascenders and descenders.  (The
    baseline is read off a reference capital that is then dropped.)"""
    t = Text("M" + s, font_size=size, color=color)
    m = t.submobjects[0]
    base = m.get_bottom()[1]
    t.remove(m)
    t.shift(np.array([x - t.get_center()[0], base_y - base, 0.0]))
    t.label_text = s
    return t


def _supline(parts, size=30, color=YELLOW_B, sup_scale=0.62, rise=0.48):
    """A line of text with true superscripts, without TeX: parts is a list of
    (string, is_superscript) or (string, is_superscript, colour).  Normal
    pieces share one baseline; a superscript sits `rise` x-heights above."""
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
        t = _on_base(body, 0.0, 0.0, size * (sup_scale if sup else 1.0), col)
        y0 = rise * xh if sup else 0.0
        t.shift(np.array([x - t.get_left()[0], y0, 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sup else 0.02)
        g.add(t)
    return g


def _cap(group):
    """Place a composite formula where caption() puts its Text."""
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
        t = _on_base(s, 0.0, 0.0, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + buff
        g.add(t)
    return g


def _frac(num, den, gap=0.08, pad=0.08, color=WHITE, width=2.5):
    """A stacked fraction from two mobjects: num over a bar over den."""
    w = max(num.width, den.width) + 2 * pad
    bar = Line(LEFT * w / 2, RIGHT * w / 2, color=color, stroke_width=width)
    num.next_to(bar, UP, buff=gap)
    den.next_to(bar, DOWN, buff=gap)
    return VGroup(num, bar, den)


def _sfrac(num, den, size=26, color=WHITE, bar_color=None):
    """A small stacked fraction label from two strings."""
    return _frac(_on_base(num, 0.0, 0.0, size, color),
                 _on_base(den, 0.0, 0.0, size, color), gap=0.06, pad=0.06,
                 color=bar_color or color, width=2)


def _mrow(parts, size=30, color=WHITE, buff=0.16):
    """A formula row: plain strings (or (string, colour)) on one baseline, and
    ("/", num, den[, colour]) as stacked fractions whose bars sit on the math
    axis (the middle of '=')."""
    axis = _on_base("=", 0.0, 0.0, size).get_center()[1]
    g = VGroup()
    x = 0.0
    for p in parts:
        if isinstance(p, tuple) and p[0] == "gap":
            x += p[1]
            continue
        if isinstance(p, tuple) and p[0] == "/":
            col = p[3] if len(p) > 3 else color
            m = _frac(_on_base(p[1], 0.0, 0.0, size * 0.86, col),
                      _on_base(p[2], 0.0, 0.0, size * 0.86, col),
                      gap=0.07, pad=0.07, color=col, width=2.2)
            m.shift(np.array([x - m.get_left()[0], axis - m[1].get_center()[1], 0.0]))
        else:
            txt, col = (p, color) if isinstance(p, str) else p
            m = _on_base(txt, 0.0, 0.0, size, col)
            m.shift(np.array([x - m.get_left()[0], 0.0, 0.0]))
        x = m.get_right()[0] + buff
        g.add(m)
    return g


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


def _dim_segs(p, q, tick=0.1):
    p, q = to3(p), to3(q)
    d = q - p
    n = np.array([-d[1], d[0], 0.0]) / max(np.linalg.norm(d), 1e-9) * tick
    return [(p, q), (p - n, p + n), (q - n, q + n)]


# ------------------------------------------------------------ label hygiene

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _segs(pts, closed=False):
    """Consecutive segments of a polyline of screen points."""
    pts = [to3(p) for p in pts]
    out = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if closed:
        out.append((pts[-1], pts[0]))
    return out


def _arc_segs(c, r, a0, a1, n=48):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    return _segs([c + r * _polar(a) for a in np.linspace(a0, a1, n + 1)])


def _angle_segs(v, p, q, r, n=16):
    """Segments along the arc that angle_arc(v, p, q, r) draws."""
    v = to3(v)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return _arc_segs(v, r, a1, a1 + span, n)


def _circle_segs(c, r, n=120):
    return _arc_segs(c, r, 0.0, TAU, n)


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by pad),
    or None."""
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
    t = [x for x in m.get_family() if isinstance(x, Text)]
    if not t:
        return type(m).__name__
    return getattr(t[0], "label_text", None) or t[0].text


def _safe(*mobs, tol=0.02):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for m in mobs:
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area: {_name(m)} "
              f"[{lo[0]:.2f},{hi[0]:.2f}]x[{lo[1]:.2f},{hi[1]:.2f}]")


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every segment in segs and
    of every other label."""
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
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _below(cap, *mobs, gap=0.03):
    """Fail the render if the caption reaches up into content above it."""
    c0, c1 = cap.get_critical_point(DL), cap.get_critical_point(UR)
    check(c1[1] < SAFE_BOTTOM + 0.02 and cap.width <= 13.45,
          "caption inside the caption band")
    for m in mobs:
        m0, m1 = m.get_critical_point(DL), m.get_critical_point(UR)
        if m1[0] < c0[0] or m0[0] > c1[0]:
            continue
        check(c1[1] + gap <= m0[1], f"caption clear of {_name(m)}")


# ======================================================= G29  sin 2θ, semicircle

class G29_DoubleAngleSemicircle(Board):
    """The right triangle in a semicircle of radius 1 with angle θ at the
    end A of the diameter AB and third vertex P.  The isosceles triangle AOP
    (OA = OP = 1) has base angles θ, θ, which together fill the exterior
    angle at O: the central angle BOP is 2θ.  The height PF is read twice.
    In the right triangle OFP (hypotenuse 1, angle 2θ at O): PF = sin 2θ.
    Folding AOP about its axis OM halves AP: AM = MP = cos θ, and AMO has
    legs cos θ, sin θ.  Flipped over the bisector at A and enlarged by
    AP = 2 cos θ, AMO lands exactly on AFP, its leg MO on PF:
    PF = 2 cos θ · sin θ.  (Drawn with θ < 45°, so F lies between O and B.)"""

    def construct(self):
        th = 28 * DEGREES
        c, s = float(np.cos(th)), float(np.sin(th))
        R = 3.5
        O = np.array([-2.6, -1.75, 0.0])

        def S(p):
            return O + R * to3(p)

        A, B = S((-1.0, 0.0)), S((1.0, 0.0))
        P = S((np.cos(2 * th), np.sin(2 * th)))
        F = S((np.cos(2 * th), 0.0))
        M, K = (A + P) / 2, (O + P) / 2
        uA = _polar(th / 2)                          # bisector at A

        # ---- the claims
        check(abs(_ang(A, B, P) - th) < 1e-9, "∠PAB = θ")
        check(abs(_ang(P, A, O) - th) < 1e-9, "isosceles AOP: ∠APO = θ")
        check(abs(_ang(O, B, P) - 2 * th) < 1e-9, "central angle ∠BOP = 2θ")
        check(abs(_ang(P, A, B) - PI / 2) < 1e-9, "the angle in the semicircle is 90°")
        check(close(2 * K - P, O), "the half-turn about the middle of OP takes P to O")
        check(close(np.linalg.norm(P - F), R * np.sin(2 * th)),
              "OFP (hypotenuse 1, angle 2θ): PF = sin 2θ")
        check(abs(np.dot(M - O, P - A)) < 1e-9 and close(_mirror(A, O, M), P),
              "OM ⊥ AP, and folding over OM takes A to P")
        check(close(np.linalg.norm(M - A), R * c) and close(np.linalg.norm(M - O), R * s),
              "AMO: AM = cos θ, MO = sin θ")
        check(close(np.linalg.norm(P - A), 2 * R * c), "AP = 2 cos θ")

        def land(X):
            return A + 2 * c * (_mirror(X, A, A + uA) - A)
        check(close(land(M), F) and close(land(O), P),
              "AMO, flipped at A and enlarged by 2 cos θ, is AFP")
        check(close(np.linalg.norm(P - F), 2 * c * R * s), "PF = 2 cos θ · sin θ")
        check(F[0] > O[0], "θ < 45°: the foot F lies between O and B")

        # ---- the semicircle, the triangle, θ at A
        arc = Arc(radius=R, start_angle=0, angle=PI, arc_center=O,
                  color=GREY_B, stroke_width=2.5)
        diam = Line(A, B, color=GREY_A, stroke_width=3)
        dotO = Dot(O, radius=0.06)
        l1a = _on_base("1", (A[0] + O[0]) / 2, O[1] - 0.42, 26)
        self.play(Create(arc), Create(diam), FadeIn(dotO), FadeIn(l1a), run_time=1.0)
        AP = Line(A, P, color=WHITE, stroke_width=3.5)
        PB = Line(P, B, color=WHITE, stroke_width=2.5)
        aA = angle_arc(A, B, P, radius=1.0, color=YELLOW_B, width=4)
        lA = tag("θ", 28, YELLOW_B).move_to(A + 1.32 * _polar(th / 2))
        self.play(Create(AP), Create(PB), Create(aA), FadeIn(lA), run_time=1.0)

        # ---- isosceles AOP: its base angles θ, θ fill the angle at O
        OP = Line(O, P, color=WHITE, stroke_width=3)
        tk_r = VGroup(_ticks(A, O, at=0.72), _ticks(O, P, at=0.72))
        l1b = _beside(tag("1", 26), O, P, _polar(2 * th + PI / 2), 0.12, at=0.45)
        aP = angle_arc(P, A, O, radius=1.0, color=YELLOW_B, width=4)
        lP = tag("θ", 28, YELLOW_B).move_to(P + 1.3 * angle_mid_dir(P, A, O))
        self.play(Create(OP), FadeIn(tk_r), FadeIn(l1b), run_time=0.8)
        self.play(Create(aP), FadeIn(lP), run_time=0.6)
        wA = _wedge(A, B, P, 0.62, YELLOW_E, 0.75)
        wP = _wedge(P, A, O, 0.62, YELLOW_E, 0.75)
        self.play(FadeIn(wA), FadeIn(wP), run_time=0.4)
        self.play(wA.animate.shift(O - A), Rotate(wP, angle=PI, about_point=K),
                  run_time=1.6)
        a2 = angle_arc(O, B, P, radius=0.62, color=YELLOW_B, width=4)
        l2 = tag("2θ", 26, YELLOW_B).move_to(O + 0.98 * _polar(th))
        self.play(Create(a2), FadeIn(l2), FadeOut(wA), FadeOut(wP), FadeOut(aP),
                  FadeOut(lP), run_time=0.8)
        self.hold(0.3)

        # ---- first reading: OFP has hypotenuse 1 and angle 2θ
        PF = DashedLine(P, F, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        raF = _ra(F, B, P, 0.2)
        tOFP = mk([O, F, P], ORANGE, 0.42, stroke_width=0)
        hF1 = Line(F, P, color=ORANGE, stroke_width=8)
        self.play(Create(PF), Create(raF), run_time=0.7)
        self.play(FadeIn(tOFP), Create(hF1), run_time=0.8)
        self.bring_to_front(OP, l1b, a2, l2, dotO)
        # the row "sin 2θ  =  2 cos θ · sin θ" under the two copies of PF
        yb = O[1]
        base = yb - 0.48
        row = _trow([("sin 2θ", ORANGE), ("=", WHITE), ("2 cos θ · sin θ", BLUE_C)],
                    30, 0.38)
        row.shift(np.array([3.95 - row.get_center()[0], base, 0.0]))
        r1, eq, r2 = row
        X1, X2 = r1.get_center()[0], r2.get_center()[0]
        check(X1 - 0.6 > B[0], "the copies of PF stand clear of the semicircle")
        bar1 = hF1.copy()
        self.add(bar1)
        self.play(bar1.animate.shift((X1 - F[0]) * RIGHT), run_time=1.1)
        check(close(bar1.get_start(), [X1, F[1], 0]) and close(bar1.get_end(), [X1, P[1], 0]),
              "the first copy of PF stands on the baseline")
        self.play(FadeIn(r1), run_time=0.5)
        self.hold(0.3)

        # ---- fold AOP about its axis OM: AM = MP = cos θ, MO = sin θ
        self.play(tOFP.animate.set_fill(opacity=0.18), hF1.animate.set_stroke(width=5),
                  run_time=0.4)
        OM = DashedLine(O, M, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        self.play(Create(OM), run_time=0.5)
        half = mk([A, M, O], BLUE_D, 0.6)
        self.add(half)
        self.bring_to_front(lA)
        self.play(_fold(half, O, M), run_time=1.3)
        check(_same_poly(half.get_vertices(), [P, M, O], 1e-6),
              "the fold lands AMO on PMO")
        raM = _ra(M, A, O, 0.2)
        tk_c = VGroup(_ticks(A, M, 2, BLUE_B), _ticks(M, P, 2, BLUE_B))
        lcos = _beside(tag("cos θ", 26, BLUE_B), A, M, _polar(th + PI / 2), 0.14)
        lsin = _beside(tag("sin θ", 26, BLUE_B), M, O, A - M, 0.14, at=0.55)
        self.play(FadeOut(half), Create(raM), FadeIn(tk_c), FadeIn(lcos),
                  FadeIn(lsin), run_time=0.8)
        self.hold(0.4)

        # ---- AMO, flipped at A and enlarged by AP = 2 cos θ, is AFP
        cA = mk([A, M, O], BLUE_D, 0.4)
        self.play(FadeIn(cA), run_time=0.4)
        fixed = [OP, l1b, a2, l2, aA, lA, lcos, lsin, raF, PF, dotO, hF1]
        self.bring_to_front(*fixed)
        self.play(Rotate(cA, angle=PI, axis=uA, about_point=A), run_time=1.2)
        tagA = tag("× 2 cos θ", 26, BLUE_B).move_to(np.array([-5.45, -2.5, 0.0]))
        self.play(cA.animate.scale(2 * c, about_point=A), FadeIn(tagA), run_time=1.4)
        check(_landed(cA, [A, F, P]), "the enlarged copy lands on A, F, P")
        self.bring_to_front(*fixed)
        hF2 = Line(F, P, color=BLUE_C, stroke_width=8)
        self.play(Create(hF2), run_time=0.5)
        bar2 = hF2.copy()
        self.add(bar2)
        self.play(bar2.animate.shift((X2 - F[0]) * RIGHT), run_time=1.1)
        check(close(bar2.get_start(), [X2, F[1], 0]) and close(bar2.get_end(), [X2, P[1], 0]),
              "the second copy of PF stands on the baseline")
        self.play(FadeIn(r2), FadeOut(tagA), run_time=0.6)
        self.play(FadeIn(eq), run_time=0.4)

        # ---- label hygiene
        segs = (_arc_segs(O, R, 0.0, PI, 90)
                + [_seg(A, B), _seg(O, P), _seg(A, P), _seg(P, B), _seg(O, M),
                   _seg(P, F), _seg(bar1.get_start(), bar1.get_end()),
                   _seg(bar2.get_start(), bar2.get_end())]
                + _angle_segs(A, B, P, 1.0) + _angle_segs(O, B, P, 0.62)
                + _ra_segs(F, B, P, 0.2) + _ra_segs(M, A, O, 0.2)
                + _tick_segs(A, O, at=0.72) + _tick_segs(O, P, at=0.72)
                + _tick_segs(A, M, 2) + _tick_segs(M, P, 2))
        labels = [l1a, l1b, lA, l2, lcos, lsin, r1, r2, eq]
        _labels_ok(labels, segs, "G29 figure")
        _labels_ok([tagA], segs, "G29 enlargement tag")
        check(not _overlap(tagA, lcos), "the × 2 cos θ tag clears cos θ")
        content = VGroup(arc, diam, AP, PB, OP, bar1, bar2, cA, *labels)
        _safe(*content)
        cap = caption("sin 2θ  =  2 sin θ cos θ", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# ================================================= G31  abc = 4RA (circumradius)

class G31_CircumradiusArea(Board):
    """Triangle ABC with circumradius R, angle α at A, area A.  Slide A round
    its arc to B', the far end of the diameter through B: the inscribed
    angle on BC stays α all the way, and the angle at C becomes a right angle
    (it stands on a diameter).  In that right triangle a = 2R · sin α.  The
    height h from B onto AC cuts off the right triangle AGB with angle α and
    hypotenuse c: h = c · sin α, and A = ½ · b · h.  The two right triangles
    have the same shape: turned about B and shrunk by c/2R, B'CB lands
    exactly on AGB.  Eliminating sin α: A = ½ bc · a/2R, so abc = 4RA.
    (Drawn with an acute triangle, so the foot G lies on AC.)"""

    def construct(self):
        al, be = 70.0, 62.0
        ga = 180.0 - al - be
        Rr = 3.05
        O = np.array([-2.4, 0.35, 0.0])

        def on(t):
            return O + Rr * _polar(t * DEGREES)

        tB = 200.0
        tC = tB + 2 * al
        tA = (tC + 2 * be) % 360.0
        A, B, C = on(tA), on(tB), on(tC)
        tBp = (tB + 180.0) % 360.0
        Bp = on(tBp)
        G = _foot(B, A, C)
        a = float(np.linalg.norm(C - B))
        b = float(np.linalg.norm(A - C))
        c = float(np.linalg.norm(B - A))
        h = float(np.linalg.norm(G - B))
        sa = float(np.sin(al * DEGREES))
        Ar = 0.5 * abs(_cross(B - A, C - A))

        # ---- the claims
        check(abs(np.degrees(_ang(A, B, C)) - al) < 1e-9
              and abs(np.degrees(_ang(B, C, A)) - be) < 1e-9
              and abs(np.degrees(_ang(C, A, B)) - ga) < 1e-9, "triangle angles α, β, γ")
        check(max(al, be, ga) < 90, "acute: G lies on AC, B' on A's arc")
        tG = np.dot(G - A, C - A) / np.dot(C - A, C - A)
        check(0 < tG < 1, "the foot G of the height from B lies on AC")
        check(close(Bp, 2 * O - B), "B' is the far end of the diameter through B")
        check(abs(np.degrees(_ang(C, B, Bp)) - 90) < 1e-9, "∠BCB' = 90°")
        check(abs(a - 2 * Rr * sa) < 1e-9, "a = 2R sin α")
        check(abs(h - c * sa) < 1e-9, "h = c sin α")
        check(abs(Ar - 0.5 * b * h) < 1e-9, "A = ½ b h")
        check(abs(a * b * c - 4 * Rr * Ar) < 1e-9, "abc = 4RA")
        check(abs(al - ga) > 15, "the height and the diameter from B stand apart")
        k = c / (2 * Rr)
        turn = float(np.arctan2(*(A - B)[1::-1]) - np.arctan2(*(Bp - B)[1::-1]))
        check(close(B + k * (_rot2(Bp, B, turn) - B), A)
              and close(B + k * (_rot2(C, B, turn) - B), G),
              "turned about B and shrunk by c/2R, B'CB is AGB")
        # the slide: A runs clockwise to B' along its own arc
        d = (tBp - tA + 180.0) % 360.0 - 180.0
        check(d < 0, "A slides clockwise to B'")
        for f in np.linspace(0.0, 1.0, 41):
            X = on(tA + f * d)
            check(abs(np.degrees(_ang(X, B, C)) - al) < 1e-7,
                  "the inscribed angle on BC stays α along the arc")

        # ---- circle, triangle, a b c, α
        circ = Circle(radius=Rr, color=GREY_B, stroke_width=2.5).move_to(O)
        dotO = Dot(O, radius=0.06)
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        la = _beside(tag("a", 30, ORANGE), B, C, O - (B + C) / 2, 0.14, at=0.5)
        la.shift(2 * (_foot(la.get_center(), B, C) - la.get_center()))   # below BC
        lb = _beside(tag("b", 30), C, A, C - B, 0.14, at=0.5)
        lc = _beside(tag("c", 30), A, B, A - C, 0.14, at=0.5)
        aA = angle_arc(A, B, C, radius=0.62, color=YELLOW_B, width=4)
        lal = tag("α", 30, YELLOW_B).move_to(A + 0.95 * angle_mid_dir(A, B, C))
        OB = Line(O, B, color=GREY_A, stroke_width=2.5)
        lR = _beside(tag("R", 28, GREY_A), O, B, _polar((tB + 90) * DEGREES), 0.12,
                     at=0.3)
        self.play(Create(circ), FadeIn(dotO), run_time=0.9)
        self.play(Create(tri), FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=1.0)
        self.play(Create(aA), FadeIn(lal), Create(OB), FadeIn(lR), run_time=0.8)
        self.hold(0.3)

        # ---- slide A round its arc to B': the angle stays α
        tt = ValueTracker(tA)

        def X():
            return on(tt.get_value())

        mv = always_redraw(lambda: VGroup(
            Line(X(), B, color=YELLOW_E, stroke_width=3),
            Line(X(), C, color=YELLOW_E, stroke_width=3),
            angle_arc(X(), B, C, radius=0.62, color=YELLOW_B, width=4),
            tag("α", 30, YELLOW_B).move_to(X() + 0.95 * angle_mid_dir(X(), B, C)),
            Dot(X(), radius=0.07, color=YELLOW_B)))
        self.play(FadeIn(mv), run_time=0.4)
        self.play(tt.animate.set_value(tA + d), run_time=2.2, rate_func=smooth)
        mv.clear_updaters()
        self.remove(mv)
        diam = Line(B, Bp, color=YELLOW_B, stroke_width=5)
        BpC = Line(Bp, C, color=YELLOW_E, stroke_width=3)
        aBp = angle_arc(Bp, B, C, radius=0.62, color=YELLOW_B, width=4)
        lalp = tag("α", 30, YELLOW_B).move_to(Bp + 0.95 * angle_mid_dir(Bp, B, C))
        raC = _ra(C, B, Bp, 0.22, YELLOW_B, 3)
        hla = Line(B, C, color=ORANGE, stroke_width=7)
        l2R = _beside(tag("2R", 28, YELLOW_B), O, Bp, _polar((tBp + 90) * DEGREES),
                      0.12, at=0.75)
        self.add(diam, BpC, aBp, lalp)
        self.play(Create(raC), Create(hla), FadeIn(l2R),
                  OB.animate.set_stroke(opacity=0.0), run_time=0.9)
        self.bring_to_front(la, dotO)
        px = 3.9
        row1 = _trow([("a", ORANGE), ("=", WHITE), ("2R", YELLOW_B), ("· sin α", WHITE)],
                     32, 0.16)
        row1.move_to(np.array([px, 2.55, 0.0]))
        self.play(FadeIn(row1, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.4)

        # ---- the height from B onto AC: h = c · sin α, A = ½ b h
        BG = DashedLine(B, G, color=TEAL_B, stroke_width=3, dash_length=0.1)
        raG = _ra(G, A, B, 0.2, TEAL_B, 3)
        tAGB = mk([A, G, B], TEAL_D, 0.45, stroke_width=0)
        lh = _beside(tag("h", 30, TEAL_B), B, G, C - A, 0.12, at=0.62)
        self.play(Create(BG), Create(raG), run_time=0.8)
        self.play(FadeIn(tAGB), FadeIn(lh), run_time=0.7)
        self.bring_to_front(tri, aA, lal, lc, BG, raG)
        row2 = _trow([("h", TEAL_B), ("=", WHITE), ("c", WHITE), ("· sin α", WHITE)],
                     32, 0.16)
        row2.move_to(np.array([px, 1.55, 0.0]))
        row2.align_to(row1, LEFT)
        self.play(FadeIn(row2, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.3)

        # ---- same shape: B'CB turned about B and shrunk by c/2R is AGB
        cp = mk([Bp, C, B], YELLOW_E, 0.5, stroke_width=2.5, stroke_color=YELLOW_B)
        self.play(FadeIn(cp), run_time=0.4)
        ktag = tag("× c/2R", 28, YELLOW_B).move_to(np.array([-5.75, -1.6, 0.0]))
        self.play(_homothety(cp, B, k, turn), FadeIn(ktag), run_time=2.0)
        check(_landed(cp, [A, G, B]), "the turned, shrunk copy lands on A, G, B")
        self.play(cp.animate.set_fill(opacity=0.0).set_stroke(opacity=0.0),
                  FadeOut(ktag), run_time=0.6)
        self.remove(cp)

        # ---- the area
        fillA = mk([A, B, C], BLUE_D, 0.35, stroke_width=0)
        lA = tag("A", 32, BLUE_B).move_to(np.array([-0.95, -0.2, 0.0]))
        self.play(FadeIn(fillA), FadeIn(lA), run_time=0.7)
        self.bring_to_front(tri, aA, lal, BG, raG, hla, la, lb, lc, lh, lA)
        row3 = _trow([("A", BLUE_B), ("=", WHITE), ("½ · b · h", WHITE)], 32, 0.16)
        row3.move_to(np.array([px, 0.45, 0.0]))
        row3.align_to(row1, LEFT)
        row4 = _trow([("=", WHITE), ("½ bc · sin α", WHITE)], 32, 0.16)
        row4.next_to(row3, DOWN, buff=0.36)
        row4.align_to(row3[1], LEFT)
        self.play(FadeIn(row3, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(row4, shift=RIGHT * 0.2), run_time=0.8)

        # ---- label hygiene
        segs = (_arc_segs(O, Rr, 0.0, TAU, 160)
                + [_seg(A, B), _seg(B, C), _seg(C, A), _seg(B, Bp), _seg(Bp, C),
                   _seg(B, G)]
                + _angle_segs(A, B, C, 0.62) + _angle_segs(Bp, B, C, 0.62)
                + _ra_segs(C, B, Bp, 0.22) + _ra_segs(G, A, B, 0.2))
        labels = [la, lb, lc, lal, lalp, l2R, lh, lA]
        _labels_ok(labels, segs, "G31 figure")
        _labels_ok([lR], segs + [_seg(O, B)], "G31 radius label")
        check(_hit(lR, [_seg(B, G)]) is None and not any(_overlap(lR, m) for m in labels),
              "R clear of the height and the labels")
        _labels_ok([ktag], segs, "G31 tag")
        for m in (row1, row2, row3, row4):
            check(m.get_left()[0] > O[0] + Rr + 0.3, "the rows stand right of the circle")
        _safe(circ, row1, row2, row3, row4, *labels)
        cap = caption("abc  =  4RA", 38)
        _below(cap, circ, row4, *labels)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ============================================== G25  tangent half-angle formulas

class G25_TangentHalfAngle(Board):
    """The unit circle with S = (−1, 0), S' = (1, 0) and P at angle θ.  The
    isosceles triangle SOP has base angles that together fill the exterior
    angle θ at O, so the chord SP leaves S at θ/2: it meets the vertical axis
    at T with OT = tan(θ/2) = t.  Enlarged from S by some ratio k, SOT lands
    on SFP (F the foot of P): SF = k, FP = kt.  The angle SPS' is right (it
    stands on the diameter), so PFS' has the angle θ/2 at P: it is SFP turned
    a quarter about F and shrunk by t, and FS' = kt².  The diameter reads
    k + kt² = 2, so k = 2/(1 + t²); then cos θ = OF = k − 1 = (1 − t²)/(1 + t²)
    and sin θ = FP = kt = 2t/(1 + t²).  (Drawn with 0 < θ < 90°, so t < 1.)"""

    def construct(self):
        th = 64 * DEGREES
        t = float(np.tan(th / 2))
        k = 2.0 / (1.0 + t * t)
        Rs = 3.2
        O = np.array([-3.2, -0.75, 0.0])

        def U(p):
            return O + Rs * to3(p)

        S, Sp = U((-1.0, 0.0)), U((1.0, 0.0))
        P = U((np.cos(th), np.sin(th)))
        F = U((np.cos(th), 0.0))
        T = U((0.0, t))
        K = (O + P) / 2

        # ---- the claims
        check(abs(_cross(T - S, P - S)) < 1e-9, "T lies on the chord SP")
        check(abs(_ang(S, Sp, P) - th / 2) < 1e-9 and abs(_ang(P, S, O) - th / 2) < 1e-9,
              "isosceles SOP: base angles θ/2, θ/2")
        check(close(2 * K - P, O), "the half-turn about the middle of OP takes P to O")
        check(abs(np.linalg.norm(T - O) - Rs * t) < 1e-9, "OT = tan(θ/2) = t")
        check(close(S + k * (O - S), F) and close(S + k * (T - S), P),
              "SOT enlarged from S by k = 2/(1 + t²) is SFP")
        check(abs(_ang(P, S, Sp) - PI / 2) < 1e-9, "∠SPS' = 90°")

        def quarter(X):
            return F + t * (_rot2(X, F, -PI / 2) - F)
        check(close(quarter(S), P) and close(quarter(P), Sp),
              "SFP turned a quarter about F and shrunk by t is PFS'")
        check(abs(np.linalg.norm(F - S) - Rs * k) < 1e-9
              and abs(np.linalg.norm(P - F) - Rs * k * t) < 1e-9
              and abs(np.linalg.norm(Sp - F) - Rs * k * t * t) < 1e-9,
              "SF = k, FP = kt, FS' = kt²")
        check(abs(k + k * t * t - 2) < 1e-12, "k + kt² = 2")
        check(abs(np.cos(th) - (k - 1)) < 1e-12 and abs(np.sin(th) - k * t) < 1e-12,
              "cos θ = k − 1, sin θ = kt")
        check(0 < t < 1, "0 < θ < 90°: T inside the circle, F right of O")

        # ---- circle, diameter, P at angle θ
        arc = Arc(radius=Rs, start_angle=0, angle=PI, arc_center=O,
                  color=GREY_B, stroke_width=2.5)
        diam = Line(S, Sp, color=GREY_A, stroke_width=3)
        dotO = Dot(O, radius=0.06)
        b0 = O[1] - 0.45
        l1 = _on_base("1", (S[0] + O[0]) / 2, b0, 28)
        self.play(Create(arc), Create(diam), FadeIn(dotO), FadeIn(l1), run_time=1.0)
        OP = Line(O, P, color=WHITE, stroke_width=3)
        aO = angle_arc(O, Sp, P, radius=0.62, color=YELLOW_B, width=4)
        lth = tag("θ", 28, YELLOW_B).move_to(O + 0.92 * _polar(th / 2))
        dP = Dot(P, radius=0.06)
        self.play(Create(OP), Create(aO), FadeIn(lth), FadeIn(dP), run_time=1.0)

        # ---- the chord from S: base angles θ/2 + θ/2 fill θ
        SP = Line(S, P, color=WHITE, stroke_width=3.5)
        tk = VGroup(_ticks(S, O, at=0.75), _ticks(O, P, at=0.7))
        aS = angle_arc(S, Sp, P, radius=1.05, color=YELLOW_B, width=4)
        aP = angle_arc(P, S, O, radius=0.9, color=YELLOW_B, width=4)
        self.play(Create(SP), FadeIn(tk), run_time=0.8)
        self.play(Create(aS), Create(aP), run_time=0.5)
        wS = _wedge(S, Sp, P, 0.62, YELLOW_E, 0.75)
        wP = _wedge(P, S, O, 0.62, YELLOW_E, 0.75)
        self.play(FadeIn(wS), FadeIn(wP), run_time=0.3)
        self.play(wS.animate.shift(O - S), Rotate(wP, angle=PI, about_point=K),
                  run_time=1.5)
        lhalf = tag("θ/2", 26, YELLOW_B).move_to(S + 1.45 * _polar(th / 4))
        self.play(FadeOut(wS), FadeOut(wP), FadeOut(aP), FadeIn(lhalf), run_time=0.7)

        # ---- T on the vertical axis: OT = t
        OT = Line(O, T, color=TEAL_B, stroke_width=6)
        raO = _ra(O, S, T, 0.2)
        tSOT = mk([S, O, T], TEAL_D, 0.42, stroke_width=0)
        lt = _beside(tag("t", 30, TEAL_B), O, T, S - O, 0.14, at=0.55)
        dT = Dot(T, radius=0.06, color=TEAL_B)
        self.play(FadeIn(tSOT), Create(OT), Create(raO), FadeIn(dT), FadeIn(lt),
                  run_time=0.9)
        self.bring_to_front(SP, aS, lhalf)
        self.hold(0.4)

        # ---- SOT enlarged from S by k lands on SFP: SF = k, FP = kt
        cp = mk([S, O, T], ORANGE, 0.45, stroke_width=2.5, stroke_color=ORANGE)
        self.add(cp)
        self.bring_to_front(SP, OT, raO, lt, lhalf, l1, dT, dotO)
        tagk = tag("× k", 28, ORANGE).move_to(np.array([-6.05, 1.5, 0.0]))
        self.play(_homothety(cp, S, k), FadeIn(tagk), run_time=1.8)
        check(_landed(cp, [S, F, P]), "the enlarged copy lands on S, F, P")
        raF = _ra(F, Sp, P, 0.2)
        FP = Line(F, P, color=ORANGE, stroke_width=6)
        lkt = _beside(tag("kt", 28, ORANGE), F, P, S - F, 0.12, at=0.5)
        y1 = O[1] - 0.85
        dSF = _dim([S[0], y1, 0], [F[0], y1, 0], ORANGE, 0.1, 3)
        b1 = y1 - 0.45
        lk = _on_base("k", (S[0] + F[0]) / 2, b1, 30, ORANGE)
        self.play(Create(FP), Create(raF), FadeOut(tagk), run_time=0.7)
        self.play(Create(dSF), FadeIn(lk), FadeIn(lkt), run_time=0.7)
        self.bring_to_front(OP, aO, lth, dP)
        self.hold(0.3)

        # ---- the angle in the semicircle is right: PFS' is SFP turned a
        # ---- quarter about F and shrunk by t, so FS' = kt²
        PSp = Line(P, Sp, color=WHITE, stroke_width=3)
        raP = _ra(P, S, Sp, 0.22)
        self.play(Create(PSp), Create(raP), run_time=0.7)
        cq = mk([S, F, P], BLUE_D, 0.5, stroke_width=2.5, stroke_color=BLUE_B)
        self.add(cq)
        self.bring_to_front(FP, raF, lkt, dP)
        tagt = tag("× t", 28, BLUE_B).move_to(np.array([-0.55, -1.35, 0.0]))
        self.play(_homothety(cq, F, t, -PI / 2), FadeIn(tagt), run_time=2.0)
        check(_landed(cq, [P, F, Sp]), "the turned, shrunk copy lands on P, F, S'")
        dFS = _dim([F[0], y1, 0], [Sp[0], y1, 0], BLUE_B, 0.1, 3)
        lkt2 = _on_base("kt²", (F[0] + Sp[0]) / 2, b1, 30, BLUE_B)
        self.play(Create(dFS), FadeIn(lkt2), FadeOut(tagt), run_time=0.7)
        self.hold(0.3)

        # ---- the diameter: k + kt² = 2
        y2 = y1 - 0.75
        dSS = _dim([S[0], y2, 0], [Sp[0], y2, 0], WHITE, 0.1, 3)
        l2 = _on_base("2", (S[0] + Sp[0]) / 2, y2 - 0.45, 30)
        self.play(Create(dSS), FadeIn(l2), run_time=0.7)
        px = 0.75
        row0 = _mrow(["t  =  tan(θ/2)"], 30)
        row1 = _mrow([("k", ORANGE), "+", ("kt²", BLUE_B), "=", "2", ("gap", 0.3), "⟹",
                      ("gap", 0.3), ("k", ORANGE), "=", ("/", "2", "1 + t²")], 30)
        row2 = _mrow([("cos θ", YELLOW_B), "=", ("k", ORANGE), "−", "1", "=",
                      ("/", "1 − t²", "1 + t²")], 30)
        row3 = _mrow([("sin θ", YELLOW_B), "=", ("kt", ORANGE), "=",
                      ("/", "2t", "1 + t²")], 30)
        for r, y in ((row0, 2.85), (row1, 1.55), (row2, 0.0), (row3, -1.55)):
            r.shift(np.array([px - r.get_left()[0], y - r.get_center()[1], 0.0]))
        self.play(FadeIn(row0, shift=RIGHT * 0.2), run_time=0.6)
        self.play(FadeIn(row1, shift=RIGHT * 0.2), run_time=0.9)
        self.hold(0.3)

        # ---- read off cos θ = OF and sin θ = FP
        OF = Line(O, F, color=YELLOW_B, stroke_width=6)
        lOF = _on_base("cos θ", (O[0] + F[0]) / 2, b0, 26, YELLOW_B)
        lFP = _beside(tag("sin θ", 26, YELLOW_B), F, P, Sp - F, 0.14, at=0.22)
        self.play(Create(OF), FadeIn(lOF), run_time=0.6)
        self.bring_to_front(dotO)
        self.play(FadeIn(row2, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(lFP), run_time=0.4)
        self.play(FadeIn(row3, shift=RIGHT * 0.2), run_time=0.8)

        # ---- label hygiene
        segs = (_arc_segs(O, Rs, 0.0, PI, 90)
                + [_seg(S, Sp), _seg(O, P), _seg(S, P), _seg(O, T), _seg(F, P),
                   _seg(P, Sp)]
                + _dim_segs([S[0], y1, 0], [F[0], y1, 0])
                + _dim_segs([F[0], y1, 0], [Sp[0], y1, 0])
                + _dim_segs([S[0], y2, 0], [Sp[0], y2, 0])
                + _angle_segs(O, Sp, P, 0.62) + _angle_segs(S, Sp, P, 1.05)
                + _ra_segs(O, S, T, 0.2) + _ra_segs(F, Sp, P, 0.2)
                + _ra_segs(P, S, Sp, 0.22)
                + _tick_segs(S, O, at=0.75) + _tick_segs(O, P, at=0.7))
        labels = [l1, lth, lhalf, lt, lk, lkt, lkt2, l2, lOF, lFP]
        _labels_ok(labels, segs, "G25 figure")
        _labels_ok([tagk, tagt], segs, "G25 tags")
        check(not any(_overlap(tg, m) for tg in (tagk, tagt) for m in (l1, lth, lhalf, lt)),
              "the tags clear the labels shown with them")
        for r in (row0, row1, row2, row3):
            check(r.get_left()[0] > O[0] + Rs + 0.3, "the rows stand right of the figure")
        _safe(arc, row0, row1, row2, row3, *labels)
        cap = caption("t = tan(θ/2):    cos θ = (1 − t²)/(1 + t²),    sin θ = 2t/(1 + t²)", 30)
        _below(cap, arc, row3, *labels)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# ============================================== G30  sin x + cos x is at most √2

def _clip_line(p, d, box):
    """The part of the line p + s·d inside the axis-parallel box
    (x0, x1, y0, y1), as two points (Liang–Barsky)."""
    x0, x1, y0, y1 = box
    lo, hi = -1e9, 1e9
    for pi, di, a, b in ((p[0], d[0], x0, x1), (p[1], d[1], y0, y1)):
        if abs(di) < 1e-12:
            continue
        s1, s2 = (a - pi) / di, (b - pi) / di
        lo, hi = max(lo, min(s1, s2)), min(hi, max(s1, s2))
    return p + lo * d, p + hi * d


class G30_SinPlusCosBound(Board):
    """On the unit circle the point P = (cos x, sin x).  Turn its vertical leg
    sin x a quarter about its foot F down onto the axis: it ends at G with
    OG = cos x + sin x, and PG has slope −1, so the line through P parallel
    to the tangent X + Y = √2 meets the axis at G.  That tangent touches the
    circle at Q, at 45°: OQH is an isosceles right triangle with legs 1, 1,
    so it meets the axis at OH = √2.  The whole disc lies on O's side of its
    tangent, so every parallel through a point of the circle meets the axis
    before H: cos x + sin x ≤ √2, with equality only at x = 45°."""

    def construct(self):
        x0 = 70 * DEGREES
        Rs = 2.6
        O = np.array([-1.3, 0.85, 0.0])
        r2 = float(np.sqrt(2.0))

        def U(p):
            return O + Rs * to3(p)

        Q = U((1 / r2, 1 / r2))
        H = U((r2, 0.0))

        def Pt(x):
            return U((np.cos(x), np.sin(x)))

        def Ft(x):
            return U((np.cos(x), 0.0))

        def Gt(x):
            return U((np.cos(x) + np.sin(x), 0.0))

        box = (-1.25, 1.6, -1.0, 1.1)                     # math box for the lines
        dm = np.array([-1.0, 1.0]) / r2

        def line_pts(c):                                  # X + Y = c, clipped
            a, b = _clip_line(np.array([c, 0.0]), dm, box)
            return U(a), U(b)

        # ---- the claims
        check(abs(np.linalg.norm(Q - O) - Rs) < 1e-9 and abs(np.dot(H - Q, Q - O)) < 1e-9,
              "QH is the tangent at Q (⊥ OQ)")
        check(abs(np.linalg.norm(H - Q) - Rs) < 1e-9, "isosceles right triangle: QH = OQ = 1")
        check(abs(np.linalg.norm(H - O) - r2 * Rs) < 1e-9, "OH = √2")
        xs = np.linspace(0.0, TAU, 3601)
        vals = np.cos(xs) + np.sin(xs)
        check(np.all(vals <= r2 + 1e-12), "cos x + sin x ≤ √2 on the whole circle")
        check(np.all(np.abs(xs[vals > r2 - 1e-9] - PI / 4) < 1e-3),
              "equality only at x = 45°")
        for x in xs[::40]:
            p = Pt(x)
            check(np.dot(p - Q, Q - O) <= 1e-9, "the circle lies on O's side of the tangent")
            g, f = Gt(x), Ft(x)
            check(abs(np.dot(g - p, np.array([1.0, 1.0, 0.0]))) < 1e-9,
                  "PG has slope −1 (parallel to the tangent)")
            check(abs(abs(g[0] - f[0]) - abs(p[1] - f[1])) < 1e-9, "FG = FP")
        check(close(_rot2(Pt(x0), Ft(x0), -PI / 2), Gt(x0)),
              "the leg FP turned a quarter about F lands on FG")

        # ---- axes and circle
        yd1, yd2 = O[1] - Rs - 0.45, O[1] - Rs - 0.9
        xax = Line(U((-1.25, 0)), U((1.75, 0)), color=GREY_B, stroke_width=2)
        yax = Line(np.array([O[0], yd2 - 0.2, 0]), U((0, 1.1)), color=GREY_B, stroke_width=2)
        circ = Circle(radius=Rs, color=GREY_A, stroke_width=3).move_to(O)
        self.play(Create(xax), Create(yax), Create(circ), run_time=1.0)

        # ---- P = (cos x, sin x): its legs
        xt = ValueTracker(x0)

        def X():
            return xt.get_value()

        P0, F0, G0 = Pt(x0), Ft(x0), Gt(x0)
        OP = Line(O, P0, color=WHITE, stroke_width=3)
        ax = angle_arc(O, H, P0, radius=0.7, color=BLUE_B, width=4)
        lx = tag("x", 26, BLUE_B).move_to(O + 1.05 * _polar((x0 + PI / 4) / 2))
        OF = Line(O, F0, color=TEAL_B, stroke_width=6)
        FP = Line(F0, P0, color=ORANGE, stroke_width=6)
        dP = Dot(P0, radius=0.07)
        b0 = O[1] - 0.42
        lcos = _on_base("cos x", 0.0, b0, 26, TEAL_B)
        lcos.shift((O[0] + 0.11 - lcos.get_left()[0]) * RIGHT)      # clear of the y-axis
        check(abs(lcos.get_center()[0] - (O[0] + F0[0]) / 2) < 0.12, "cos x sits under OF")
        lsin = _beside(tag("sin x", 26, ORANGE), F0, P0, RIGHT, 0.12, at=0.5)
        self.play(Create(OP), Create(ax), FadeIn(lx), FadeIn(dP), run_time=0.8)
        self.play(Create(OF), Create(FP), FadeIn(lcos), FadeIn(lsin), run_time=0.8)
        self.hold(0.3)

        # ---- turn the leg sin x a quarter about F, down onto the axis
        FG = FP.copy()
        self.add(FG)
        self.play(Rotate(FG, angle=-PI / 2, about_point=F0), run_time=1.1)
        check(close(FG.get_start(), F0) and close(FG.get_end(), G0),
              "the turned leg ends at G = (cos x + sin x, 0)")
        lsin2 = _on_base("sin x", (F0[0] + G0[0]) / 2 - 0.2, b0, 26, ORANGE)
        check(lsin2.get_left()[0] > F0[0] + 0.1 and lsin2.get_right()[0] < G0[0] - 0.1,
              "sin x sits under FG")
        self.play(FadeOut(lsin), FadeIn(lsin2), run_time=0.5)
        la, lb = line_pts(np.cos(x0) + np.sin(x0))
        par = Line(la, lb, color=ORANGE, stroke_width=3)
        dG = Dot(G0, radius=0.07, color=ORANGE)
        self.play(Create(par), FadeIn(dG), run_time=0.8)
        colx = H[0] + 0.45
        gG = DashedLine(G0, [G0[0], yd2, 0], color=GREY_B, stroke_width=1.5,
                        dash_length=0.08)
        dOG = _dim([O[0], yd2, 0], [G0[0], yd2, 0], ORANGE, 0.1, 3)
        lsum = _on_base("cos x + sin x", 0.0, yd2 - 0.12, 30, ORANGE)
        lsum.shift((colx - lsum.get_left()[0]) * RIGHT)
        self.play(Create(gG), Create(dOG), FadeIn(lsum), run_time=0.8)
        self.hold(0.3)

        # ---- the tangent at 45°: OQH has legs 1, 1, so OH = √2; it is
        # ---- parallel to PG
        OQ = Line(O, Q, color=WHITE, stroke_width=3)
        a45 = angle_arc(O, H, Q, radius=0.9, color=GREY_A, width=3)
        l45 = tag("45°", 24, GREY_A).move_to(O + 1.5 * _polar(18 * DEGREES))
        l1a = _beside(tag("1", 26), O, Q, _polar(-PI / 4), 0.12, at=0.72)
        ta, tb = line_pts(r2)
        tang = Line(ta, tb, color=YELLOW_B, stroke_width=4)
        raQ = _ra(Q, O, H, 0.2)
        l1b = _beside(tag("1", 26), Q, H, _polar(PI / 4), 0.12, at=0.5)
        dQ = Dot(Q, radius=0.06, color=YELLOW_B)
        self.play(Create(OQ), Create(a45), FadeIn(l45), FadeIn(l1a), FadeIn(dQ),
                  run_time=0.9)
        self.play(Create(tang), Create(raQ), FadeIn(l1b), run_time=0.9)
        gH = DashedLine(H, [H[0], yd1, 0], color=GREY_B, stroke_width=1.5, dash_length=0.08)
        dOH = _dim([O[0], yd1, 0], [H[0], yd1, 0], YELLOW_B, 0.1, 3)
        lr2 = _on_base("√2", 0.0, yd1 - 0.12, 30, YELLOW_B)
        lr2.shift((colx - lr2.get_left()[0]) * RIGHT)
        self.play(Create(gH), Create(dOH), FadeIn(lr2), run_time=0.8)
        self.bring_to_front(dG, dP)
        self.hold(0.6)

        # ---- label hygiene (the still figure)
        segs = (_circle_segs(O, Rs, 160)
                + [_seg(xax.get_start(), xax.get_end()), _seg(yax.get_start(), yax.get_end()),
                   _seg(O, Q), _seg(ta, tb), _seg(O, P0), _seg(O, F0), _seg(F0, P0),
                   _seg(F0, G0), _seg(la, lb), _seg(H, [H[0], yd1, 0]),
                   _seg(G0, [G0[0], yd2, 0])]
                + _dim_segs([O[0], yd1, 0], [H[0], yd1, 0])
                + _dim_segs([O[0], yd2, 0], [G0[0], yd2, 0])
                + _angle_segs(O, H, Q, 0.9) + _angle_segs(O, H, P0, 0.7)
                + _ra_segs(Q, O, H, 0.2))
        labels = [l45, l1a, l1b, lr2, lx, lcos, lsin2, lsum]
        _labels_ok(labels, segs, "G30 figure")
        early = (_circle_segs(O, Rs, 160)
                 + [_seg(xax.get_start(), xax.get_end()),
                    _seg(yax.get_start(), yax.get_end()),
                    _seg(O, P0), _seg(O, F0), _seg(F0, P0)]
                 + _angle_segs(O, H, P0, 0.7))
        _labels_ok([lsin, lx, lcos], early, "G30 legs (before the turn)")

        # ---- P runs round the whole circle: the parallel never passes the tangent
        # (labels inside the disc step aside: every parallel crosses the disc)
        self.play(FadeOut(lsin2), FadeOut(lcos), FadeOut(lx), FadeOut(ax), FadeOut(FG),
                  FadeOut(l1a), FadeOut(l45), run_time=0.4)

        def moving():
            x = X()
            p, f, g = Pt(x), Ft(x), Gt(x)
            a, b = line_pts(np.cos(x) + np.sin(x))
            return VGroup(
                Line(a, b, color=ORANGE, stroke_width=3),
                Line(O, p, color=WHITE, stroke_width=3),
                Line(O, f, color=TEAL_B, stroke_width=6),
                Line(f, p, color=ORANGE, stroke_width=6),
                DashedLine(g, [g[0], yd2, 0], color=GREY_B, stroke_width=1.5,
                           dash_length=0.08),
                _dim([O[0], yd2, 0], [g[0], yd2, 0], ORANGE, 0.1, 3),
                Dot(g, radius=0.07, color=ORANGE), Dot(p, radius=0.07))

        mv = always_redraw(moving)
        self.remove(par, OP, OF, FP, gG, dOG, dG, dP)
        self.add(mv)
        self.bring_to_front(tang, dQ)
        for x in np.linspace(x0, x0 + TAU, 721):
            g = Gt(x)
            check(g[0] <= H[0] + 1e-9, "G never passes H during the sweep")
            a, b = line_pts(np.cos(x) + np.sin(x))
            check(min(a[1], b[1]) > yd1 + 0.15, "the moving parallel stays above the dims")
            for m in (lr2, lsum, l1b):
                check(_hit(m, [_seg(a, b)], 0.04) is None,
                      f"the moving parallel clears '{_name(m)}'")
        self.play(xt.animate.set_value(PI / 4), run_time=1.4, rate_func=smooth)
        self.hold(0.5)                                    # equality: the tangent itself
        self.play(xt.animate.set_value(x0 + TAU), run_time=5.0, rate_func=smooth)
        mv.clear_updaters()
        self.remove(mv)
        self.add(par, OP, OF, FP, gG, dOG, dG, dP)
        self.bring_to_front(tang, dQ)
        ax2 = angle_arc(O, H, P0, radius=0.7, color=BLUE_B, width=4)
        self.add(FG)
        self.bring_to_front(dG, dP)
        self.play(FadeIn(lcos), FadeIn(lsin2), FadeIn(lx), FadeIn(ax2), FadeIn(l1a),
                  FadeIn(l45), run_time=0.5)

        content = VGroup(xax, yax, circ, tang, par, dOH, dOG, *labels)
        _safe(*content)
        cap = caption("sin x + cos x  ≤  √2", 38)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ============================================== H24  area under a cycloid

class H24_CycloidArea(Board):
    """A disc of radius r rolls along the line; a point of its rim traces the
    cycloid, and the foot of that point on the disc's vertical diameter
    traces the companion curve.  At every height the two curves are joined
    by a horizontal half-chord of the rolling disc, so between them lie
    exactly the chords of a half-disc of radius r (the left half of the disc
    at the start): by Cavalieri the region between the curves has area ½πr².
    The companion curve is symmetric under the half-turn about the centre of
    the πr × 2r rectangle over the half-arch, so it halves the rectangle:
    the region under it has area πr².  The half-arch encloses
    πr² + ½πr²; the arch is symmetric, so A = 2 (πr² + ½πr²) = 3πr²."""

    def construct(self):
        r = 1.75
        xs, yb = -4.65, -1.45
        N = 240
        ss = np.linspace(0.0, PI, N + 1)

        def cyc(t):
            return np.array([xs + r * (t - np.sin(t)), yb + r * (1 - np.cos(t)), 0.0])

        def com(t):
            return np.array([xs + r * t, yb + r * (1 - np.cos(t)), 0.0])

        def disc_left(t):                       # left half of the start disc
            return np.array([xs - r * np.sin(t), yb + r * (1 - np.cos(t)), 0.0])

        Cc = np.array([xs + PI * r / 2, yb + r, 0.0])        # rectangle centre
        BR, TL = np.array([xs + PI * r, yb, 0.0]), np.array([xs, yb + 2 * r, 0.0])
        xm = xs + PI * r                                    # the arch's axis

        between = _dedup([cyc(t) for t in ss] + [com(t) for t in ss[::-1]])
        half_disc = _dedup([disc_left(t) for t in ss] + [TL])
        Lreg = _dedup([com(t) for t in ss] + [BR])
        Ureg = _dedup([com(t) for t in ss] + [TL])
        under = _dedup([cyc(t) for t in ss] + [BR])

        # ---- the claims
        for h in np.linspace(0.002, 2 * r - 0.002, 81):
            t = float(np.arccos(1 - h / r))
            w_arch = com(t)[0] - cyc(t)[0]
            w_disc = xs - disc_left(t)[0]
            check(abs(w_arch - w_disc) < 1e-12 and abs(w_disc - np.sqrt(r * r - (h - r) ** 2)) < 1e-9,
                  "at every height: chord between the curves = chord of the half-disc")
        A_b, A_d = abs(area(between)), abs(area(half_disc))
        check(abs(A_b - A_d) < 1e-3 and abs(A_d - PI * r * r / 2) < 1e-3,
              "between the curves: ½πr² (sampled outlines)")
        check(all(close(2 * Cc - com(t), com(PI - t), 1e-9) for t in ss),
              "the half-turn about the rectangle's centre maps the companion onto itself")
        check(_same_poly([2 * Cc - p for p in Ureg], Lreg, 1e-9),
              "the half-turn carries the part above the companion onto the part below")
        check(abs(abs(area(Lreg)) - PI * r * r) < 1e-3, "under the companion: πr²")
        check(abs(abs(area(under)) - 1.5 * PI * r * r) < 1e-3, "under the half-arch: (3/2)πr²")
        check(xs - r >= -SAFE_X and xs + 2 * PI * r <= SAFE_X, "the arch and the half-disc fit")
        Rmax = max(float(np.linalg.norm(p - Cc)) for p in Ureg)
        check(Cc[1] - Rmax >= SAFE_BOTTOM and Cc[1] + Rmax <= SAFE_TOP
              and Cc[0] - Rmax >= -SAFE_X and Cc[0] + Rmax <= SAFE_X,
              "the half-turning region stays inside the safe area all the way round")

        # ---- the disc rolls half a turn: rim point -> cycloid, its foot -> companion
        base = Line([xs - r - 0.15, yb, 0], [xs + 2 * PI * r + 0.15, yb, 0],
                    color=GREY_B, stroke_width=2)
        tt = ValueTracker(0.0)

        def roller():
            t = tt.get_value()
            c = np.array([xs + r * t, yb + r, 0.0])
            p, k = cyc(t), com(t)
            g = VGroup(Circle(radius=r, color=GREY_A, stroke_width=3).move_to(c),
                       DashedLine(c + r * DOWN, c + r * UP, color=GREY_B, stroke_width=1.5,
                                  dash_length=0.08),
                       Line(c, p, color=GREY_A, stroke_width=2))
            if t > 1e-3:
                g.add(Line(p, k, color=ORANGE, stroke_width=5),
                      _curve([cyc(u) for u in np.linspace(0, t, 120)], WHITE, 4),
                      _curve([com(u) for u in np.linspace(0, t, 120)], TEAL_B, 3.5),
                      Dot(k, radius=0.06, color=TEAL_B))
            g.add(Dot(p, radius=0.07, color=YELLOW_B))
            return g

        roll = always_redraw(roller)
        lr = tag("r", 28).move_to(np.array([xs + r / 2, yb + r + 0.24, 0.0]))
        rad0 = Line([xs, yb + r, 0], [xs + r, yb + r, 0], color=GREY_A, stroke_width=2)
        self.play(Create(base), FadeIn(roll), Create(rad0), FadeIn(lr), run_time=1.0)
        self.hold(0.3)
        self.play(FadeOut(rad0), FadeOut(lr), run_time=0.3)
        self.play(tt.animate.set_value(PI), run_time=3.4, rate_func=linear)
        roll.clear_updaters()
        cyc_m = _curve([cyc(t) for t in ss], WHITE, 4)
        com_m = _curve([com(t) for t in ss], TEAL_B, 3.5)
        self.add(cyc_m, com_m)
        self.play(FadeOut(roll), run_time=0.6)

        # ---- the other half of the arch is the mirror image
        cyc_r = _curve([np.array([2 * xm - p[0], p[1], 0.0]) for p in
                        (cyc(t) for t in ss)], WHITE, 4)
        self.play(Create(cyc_r), run_time=0.8)
        rect = DashedVMobject(Polygon([xs, yb, 0], BR, [xm, yb + 2 * r, 0], TL,
                                      stroke_color=GREY_B, stroke_width=2), num_dashes=70)
        ydim = yb - 0.32
        dpi = _dim([xs, ydim, 0], [xm, ydim, 0], WHITE, 0.09, 2.5)
        lpi = _on_base("πr", (xs + xm) / 2, ydim - 0.42, 28)
        d2r = _dim([xm + 0.32, yb, 0], [xm + 0.32, yb + 2 * r, 0], WHITE, 0.09, 2.5)
        l2r = tag("2r", 28).next_to(np.array([xm + 0.32, yb + r, 0.0]), RIGHT, buff=0.14)
        self.play(Create(rect), Create(dpi), FadeIn(lpi), Create(d2r), FadeIn(l2r),
                  run_time=1.0)
        self.hold(0.3)

        # ---- Cavalieri: at every height the two chords are equal
        hd_out = VGroup(Arc(radius=r, start_angle=PI / 2, angle=PI,
                            arc_center=[xs, yb + r, 0], color=GREY_A, stroke_width=3),
                        Line([xs, yb, 0], TL, color=GREY_A, stroke_width=2))
        self.play(Create(hd_out), run_time=0.8)
        hy = ValueTracker(0.0)

        def scan():
            h = hy.get_value()
            g = VGroup()
            if h < 1e-4:
                return g
            t = float(np.arccos(np.clip(1 - h / r, -1.0, 1.0)))
            us = np.linspace(0.0, t, 90)
            g.add(_region([cyc(u) for u in us] + [com(u) for u in us[::-1]], ORANGE, 0.6),
                  _region([disc_left(u) for u in us] + [[xs, yb + h, 0]], ORANGE, 0.6))
            y = yb + h
            g.add(DashedLine([xs - r - 0.1, y, 0], [com(t)[0] + 0.1, y, 0], color=GREY_B,
                             stroke_width=1.5, dash_length=0.08),
                  Line(disc_left(t), [xs, y, 0], color=YELLOW_B, stroke_width=6),
                  Line(cyc(t), com(t), color=YELLOW_B, stroke_width=6))
            return g

        sc = always_redraw(scan)
        self.add(sc)
        self.bring_to_front(cyc_m, com_m, hd_out)
        self.play(hy.animate.set_value(2 * r), run_time=3.6, rate_func=linear)
        sc.clear_updaters()
        self.remove(sc)
        reg_b = _region(between, ORANGE, 0.6)
        reg_d = _region(half_disc, ORANGE, 0.6)
        self.add(reg_b, reg_d)
        self.bring_to_front(cyc_m, com_m, hd_out)
        lb = tag("½πr²", 26).move_to(np.array([xs + 1.071 * r, yb + r, 0.0]))
        ld = tag("½πr²", 26).move_to(np.array([xs - 0.47 * r, yb + r, 0.0]))
        self.play(FadeIn(lb), FadeIn(ld), run_time=0.6)
        self.hold(0.4)

        # ---- the companion halves the rectangle: a half-turn about its centre
        regU = _region(Ureg, TEAL_D, 0.5)
        regL = _region(Lreg, BLUE_D, 0.6)
        self.play(FadeIn(regU), run_time=0.5)
        self.bring_to_front(cyc_m, com_m, lb)
        dC = Dot(Cc, radius=0.07, color=YELLOW_B)
        self.play(FadeIn(dC), run_time=0.3)
        self.play(Rotate(regU, angle=PI, about_point=Cc), run_time=1.8)
        check(_same_poly(regU.get_vertices(), Lreg, 1e-6),
              "the turned upper part lands exactly on the part under the companion")
        lL = tag("πr²", 30).move_to(np.array([xs + 0.72 * PI * r, yb + 0.45 * r, 0.0]))
        self.add(regL)
        self.remove(regU)
        self.bring_to_front(cyc_m, com_m, reg_b, lb)
        self.play(FadeOut(dC), FadeIn(lL), run_time=0.6)
        self.hold(0.4)

        # ---- the arch is symmetric: fold the half-arch over its axis
        half = VGroup(regL.copy(), reg_b.copy(), com_m.copy().set_stroke(opacity=0.6))
        self.add(half)
        self.bring_to_front(cyc_m, com_m, lb, lL)
        self.play(Rotate(half, angle=PI, axis=UP, about_point=[xm, yb, 0]), run_time=1.5)
        check(_same_poly(half[0].get_vertices(), [[2 * xm - p[0], p[1], 0] for p in Lreg], 1e-6),
              "the folded half lands on the mirror half-arch")
        lL2 = tag("πr²", 30).move_to(np.array([2 * xm - (xs + 0.72 * PI * r), yb + 0.45 * r, 0]))
        lb2 = tag("½πr²", 26).move_to(np.array([2 * xm - (xs + 1.071 * r), yb + r, 0.0]))
        self.bring_to_front(cyc_r, d2r, l2r)
        self.play(FadeIn(lL2), FadeIn(lb2), run_time=0.6)

        # ---- label hygiene
        segs = (_segs([cyc(t) for t in ss]) + _segs([com(t) for t in ss])
                + _segs([[2 * xm - cyc(t)[0], cyc(t)[1], 0] for t in ss])
                + _segs([[2 * xm - com(t)[0], com(t)[1], 0] for t in ss])
                + _arc_segs([xs, yb + r, 0], r, PI / 2, 3 * PI / 2, 60)
                + [_seg([xs, yb, 0], TL), _seg(base.get_start(), base.get_end()),
                   _seg([xs, yb, 0], BR), _seg(BR, [xm, yb + 2 * r, 0]),
                   _seg([xm, yb + 2 * r, 0], TL)]
                + _dim_segs([xs, ydim, 0], [xm, ydim, 0])
                + _dim_segs([xm + 0.32, yb, 0], [xm + 0.32, yb + 2 * r, 0]))
        labels = [lpi, l2r, lb, ld, lL, lL2, lb2]
        _labels_ok(labels, segs, "H24 figure")
        _labels_ok([lr], [_seg([xs, yb + r, 0], [xs + r, yb + r, 0])]
                   + _circle_segs([xs, yb + r, 0], r, 90), "H24 radius label")
        for m, poly in ((lb, between), (ld, half_disc), (lL, Lreg)):
            x0, x1, y0, y1 = _box(m, 0.03)
            check(all(_pip(q, poly) for q in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))),
                  f"'{_name(m)}' sits inside the region it names")
        row = _row(tag("A  =  2 (πr²  +  ½πr²)  =  3πr²", 34, YELLOW_B))
        row.move_to(np.array([0.0, 2.75, 0.0]))
        _safe(base, rect, dpi, d2r, row, *labels)
        cap = caption("A  =  3πr²", 38)
        _below(cap, base, dpi, lpi)
        self.play(FadeIn(row, shift=DOWN * 0.15), run_time=0.9)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ============================================== H11  derivative of √x

def _rect(x0, y0, x1, y1, color, op=FILL, stroke=1.5):
    """Axis-parallel filled rectangle between screen corners (x0, y0), (x1, y1)."""
    return Polygon([x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                   fill_color=color, fill_opacity=op, stroke_color=WHITE,
                   stroke_width=stroke)


class H11_SqrtDerivative(Board):
    """A square of area x (side √x) grows by dx: its side grows by Δ√x, and
    the gnomon of area dx is two strips √x · Δ√x and a corner (Δ√x)².  Laid
    end to end (the right strip turned a quarter) they make one bar Δ√x
    thick and 2√x + Δ√x long: dx = (2√x + Δ√x) · Δ√x exactly.  Rescaled
    across by ÷ dx the bar has area 1 and thickness Δ√x / dx.  As dx → 0 the
    corner's piece shrinks to nothing, the bar's length tends to 2√x and the
    bar settles onto the dashed 2√x × 1/(2√x) rectangle, so its thickness
    Δ√x / dx tends to 1/(2√x)."""

    def construct(self):
        x = 1.44
        s = float(np.sqrt(x))
        k = 2.3
        h0 = 0.2
        dx0 = (s + h0) ** 2 - x
        dx1 = 0.018

        def hof(d):
            return float(np.sqrt(x + d) - s)

        S0 = np.array([-5.7, -1.75, 0.0])

        def P(a, b):
            return S0 + k * np.array([a, b, 0.0])

        XB, YB = -1.75, -0.15

        def B(a, b):                     # a: math units along, b: screen units up
            return np.array([XB + k * a, YB + b, 0.0])

        # ---- the claims
        for d in (dx0, 0.25, 0.1, dx1):
            h = hof(d)
            check(abs(2 * s * h + h * h - d) < 1e-12, "two strips + corner = dx")
            check(abs((2 * s + h) * h - d) < 1e-12, "dx = (2√x + Δ√x) · Δ√x")
            check(abs((2 * s + h) * (h / d) - 1) < 1e-12, "÷ dx: the bar has area 1")
        h1 = hof(dx1)
        check(h1 * k < 0.02, "the corner's piece has shrunk to nothing")
        check((1 / (2 * s) - h1 / dx1) * k < 0.01,
              "the rescaled thickness has reached 1/(2√x)")
        check(abs(hof(dx0) - h0) < 1e-12, "the first step: Δ√x = h0")

        def square_inc(d):
            h = hof(d)
            Rr = _rect(*P(s, 0)[:2], *P(s + h, s)[:2], TEAL_D)
            Tt = _rect(*P(0, s)[:2], *P(s, s + h)[:2], TEAL_D)
            Cn = _rect(*P(s, s)[:2], *P(s + h, s + h)[:2], ORANGE, 0.9)
            return Rr, Tt, Cn

        def gnomon_outline(d):
            h = hof(d)
            m = VMobject(stroke_color=YELLOW_B, stroke_width=4)
            m.set_points_as_corners([P(s, 0), P(s + h, 0), P(s + h, s + h), P(0, s + h),
                                     P(0, s), P(s, s), P(s, 0)])
            return m

        def bar(d, thick):
            h = hof(d)
            return (_rect(*B(0, 0)[:2], *B(s, thick)[:2], TEAL_D),
                    _rect(*B(s, 0)[:2], *B(2 * s, thick)[:2], TEAL_D),
                    _rect(*B(2 * s, 0)[:2], *B(2 * s + h, thick)[:2], ORANGE, 0.9))

        def bar_outline(d, thick):
            h = hof(d)
            return Polygon(B(0, 0), B(2 * s + h, 0), B(2 * s + h, thick), B(0, thick),
                           stroke_color=YELLOW_B, stroke_width=4, fill_opacity=0)

        # ---- the square of area x
        sq = _rect(*P(0, 0)[:2], *P(s, s)[:2], BLUE_D)
        l_x = tag("x", 36).move_to(P(s / 2, s / 2))
        b_lo = S0[1] - 0.16 - Text("M", font_size=28).height
        l_sb = _on_base("√x", P(s / 2, 0)[0], b_lo, 28)
        l_sl = tag("√x", 28).next_to(P(0, s / 2), LEFT, buff=0.16)
        self.play(FadeIn(sq), FadeIn(l_x), FadeIn(l_sb), FadeIn(l_sl), run_time=1.1)
        self.hold(0.3)

        # ---- the area grows by dx: two strips and a corner, the side by Δ√x
        Rr, Tt, Cn = square_inc(dx0)
        self.play(FadeIn(Rr, shift=RIGHT * 0.2), FadeIn(Tt, shift=UP * 0.2), run_time=0.9)
        self.play(FadeIn(Cn, scale=0.6), run_time=0.5)
        l_hb = _on_base("Δ√x", P(s + h0 / 2, 0)[0] + 0.12, b_lo, 26)
        l_hl = tag("Δ√x", 26).next_to(P(0, s + h0 / 2), LEFT, buff=0.16)
        go = gnomon_outline(dx0)
        l_dx = tag("dx", 30, YELLOW_B).next_to(P(s + h0, s + h0), UR, buff=0.1)
        self.play(FadeIn(l_hb), FadeIn(l_hl), run_time=0.5)
        self.play(Create(go), FadeIn(l_dx), run_time=0.8)
        self.hold(0.5)

        # ---- end to end: the top strip slides, the right strip turns a
        # ---- quarter, the corner slides: one bar Δ√x thick, of area dx
        cT, cR, cC = Tt.copy(), Rr.copy(), Cn.copy()
        self.add(cT, cR, cC)
        tgt = bar(dx0, k * h0)
        self.play(_rigid(cT, shift=tgt[0].get_center() - cT.get_center()),
                  _rigid(cR, angle=-PI / 2, shift=tgt[1].get_center() - cR.get_center()),
                  _rigid(cC, shift=tgt[2].get_center() - cC.get_center()),
                  run_time=1.9)
        for got, want in zip((cT, cR, cC), tgt):
            g = sorted(map(tuple, np.round(got.get_vertices()[:, :2], 7)))
            w_ = sorted(map(tuple, np.round(want.get_vertices()[:, :2], 7)))
            check(np.allclose(g, w_, atol=1e-6), "each piece lands in the bar")
        self.remove(cT, cR, cC)
        pieces = VGroup(*tgt)
        self.add(pieces)
        bo = bar_outline(dx0, k * h0)
        # the bar's AREA (not a side): "▭ = dx" above it, as in the J-scenes
        l_bar = tag("▭ = dx", 28, YELLOW_B).next_to(B(0, k * h0), UP, buff=0.25)
        l_bar.align_to(B(0, 0), LEFT)
        check(not _overlap(l_bar, l_dx, 0.05), "the bar's area label is clear of dx")
        self.play(Create(bo), FadeIn(l_bar), run_time=0.7)
        self.hold(0.3)

        # ---- rescaled across: ÷ dx — the bar's area becomes 1
        th0 = k * h0 / dx0
        l_div = tag("÷ dx", 28).next_to(B(0, th0), UP, buff=0.25).align_to(B(0, 0), LEFT)
        bo2 = bar_outline(dx0, th0)
        l_one = tag("▭ = 1", 28, YELLOW_B).next_to(l_div, RIGHT, buff=0.45)
        self.play(pieces.animate.stretch(th0 / (k * h0), 1, about_edge=DOWN),
                  Transform(bo, bo2), FadeIn(l_div), FadeOut(l_bar), FadeIn(l_one),
                  run_time=1.2)
        check(abs(pieces.height - th0) < 1e-6 and abs(pieces.get_bottom()[1] - YB) < 1e-6,
              "the bar is rescaled across, about its base")

        # ---- its parts, its thickness, and the limit rectangle (dashed)
        thL = k / (2 * s)
        ref = DashedVMobject(Polygon(B(0, 0), B(2 * s, 0), B(2 * s, thL), B(0, thL),
                                     stroke_color=WHITE, stroke_width=2), num_dashes=60)
        l_s1 = tag("√x", 26).move_to(B(s / 2, th0 / 2))
        l_s2 = tag("√x", 26).move_to(B(1.5 * s, th0 / 2))
        l_hc = tag("Δ√x", 24, ORANGE).next_to(B(2 * s + h0 / 2, max(th0, thL)), UP, buff=0.16)
        d2s = _dim(B(0, -0.3), B(2 * s, -0.3), YELLOW_B, 0.1, 3)
        l_2s = tag("2√x", 30, YELLOW_B).next_to(d2s, DOWN, buff=0.14)
        xt = B(2 * s + h0, 0)[0] + 0.3

        def th_dim(d):
            return _dim([xt, YB, 0], [xt, YB + k * hof(d) / d, 0], WHITE, 0.09, 2.5)

        dth = th_dim(dx0)
        l_th = VGroup(tag("Δ√x", 26), Line(LEFT * 0.36, RIGHT * 0.36, stroke_width=2),
                      tag("dx", 26)).arrange(DOWN, buff=0.06)
        l_th.next_to(np.array([xt, YB + th0 / 2, 0.0]), RIGHT, buff=0.16)
        self.play(FadeIn(l_s1), FadeIn(l_s2), FadeIn(l_hc), run_time=0.6)
        self.play(Create(dth), FadeIn(l_th), run_time=0.6)
        self.play(Create(ref), Create(d2s), FadeIn(l_2s), run_time=0.9)
        segs0 = (_segs([P(0, 0), P(s + h0, 0), P(s + h0, s + h0), P(0, s + h0)], closed=True)
                 + [_seg(P(s, 0), P(s, s + h0)), _seg(P(0, s), P(s + h0, s))]
                 + _segs([B(0, 0), B(2 * s + h0, 0), B(2 * s + h0, th0), B(0, th0)],
                         closed=True)
                 + [_seg(B(s, 0), B(s, th0)), _seg(B(2 * s, 0), B(2 * s, th0))]
                 + _segs([B(0, 0), B(2 * s, 0), B(2 * s, thL), B(0, thL)], closed=True)
                 + _dim_segs(B(0, -0.3), B(2 * s, -0.3))
                 + _dim_segs([xt, YB, 0], [xt, YB + th0, 0]))
        _labels_ok([l_x, l_sb, l_sl, l_hb, l_hl, l_dx, l_div, l_one, l_s1, l_s2, l_hc,
                    l_th, l_2s], segs0, "H11 first step")
        # the "▭ = dx" label lived only while the bar was thin: check it there
        segs_thin = (_segs([P(0, 0), P(s + h0, 0), P(s + h0, s + h0), P(0, s + h0)], closed=True)
                     + [_seg(P(s, 0), P(s, s + h0)), _seg(P(0, s), P(s + h0, s))]
                     + _segs([B(0, 0), B(2 * s + h0, 0), B(2 * s + h0, k * h0),
                              B(0, k * h0)], closed=True))
        _labels_ok([l_bar, l_dx, l_x, l_sb, l_sl, l_hb, l_hl], segs_thin, "H11 bar label")
        self.hold(0.5)

        # ---- dx → 0: the strips thin, the corner's piece vanishes, the bar
        # ---- settles onto the 2√x × 1/(2√x) rectangle
        lg = ValueTracker(np.log(dx0))

        def dnow():
            return float(np.exp(lg.get_value()))

        sq_inc = always_redraw(lambda: VGroup(*square_inc(dnow())))
        go_r = always_redraw(lambda: gnomon_outline(dnow()))
        bar_r = always_redraw(lambda: VGroup(*bar(dnow(), k * hof(dnow()) / dnow())))
        bo_r = always_redraw(lambda: bar_outline(dnow(), k * hof(dnow()) / dnow()))
        dth_r = always_redraw(lambda: th_dim(dnow()))
        self.remove(Rr, Tt, Cn, go, pieces, bo, dth)
        self.add(sq_inc, go_r, bar_r, bo_r, dth_r)
        self.bring_to_front(ref, l_s1, l_s2, l_dx)

        def follow(lbl, fn):
            lbl.add_updater(lambda m: fn(m, dnow()))

        follow(l_hb, lambda m, d: m.set_x(P(s + hof(d) / 2, 0)[0] + 0.12))
        follow(l_hl, lambda m, d: m.set_y(P(0, s + hof(d) / 2)[1]))
        follow(l_dx, lambda m, d: m.next_to(P(s + hof(d), s + hof(d)), UR, buff=0.1))
        follow(l_s1, lambda m, d: m.set_y(YB + k * hof(d) / d / 2))
        follow(l_s2, lambda m, d: m.set_y(YB + k * hof(d) / d / 2))
        follow(l_th, lambda m, d: m.set_y(YB + k * hof(d) / d / 2))
        self.play(FadeOut(l_hc), run_time=0.3)
        self.play(lg.animate.set_value(np.log(dx1)), run_time=4.4, rate_func=smooth)
        for m in (sq_inc, go_r, bar_r, bo_r, dth_r, l_hb, l_hl, l_dx, l_s1, l_s2, l_one,
                  l_th):
            m.clear_updaters()
        l_lim = VGroup(tag("1", 26, YELLOW_B), Line(LEFT * 0.42, RIGHT * 0.42,
                                                    stroke_width=2, color=YELLOW_B),
                       tag("2√x", 26, YELLOW_B)).arrange(DOWN, buff=0.06)
        l_lim.move_to(l_th.get_center())
        self.play(FadeOut(l_th), FadeIn(l_lim), run_time=0.8)

        # ---- label hygiene
        hb = hof(dx1)
        thb = k * hb / dx1
        segs = (_segs([P(0, 0), P(s, 0), P(s, s), P(0, s)], closed=True)
                + _segs([P(s, 0), P(s + hb, 0), P(s + hb, s + hb), P(0, s + hb), P(0, s)])
                + _segs([B(0, 0), B(2 * s + hb, 0), B(2 * s + hb, thb), B(0, thb)], closed=True)
                + [_seg(B(s, 0), B(s, thb))]
                + _segs([B(0, 0), B(2 * s, 0), B(2 * s, thL), B(0, thL)], closed=True)
                + _dim_segs(B(0, -0.3), B(2 * s, -0.3))
                + _dim_segs([xt, YB, 0], [xt, YB + thb, 0]))
        labels = [l_x, l_sb, l_sl, l_hb, l_hl, l_dx, l_div, l_one, l_s1, l_s2, l_2s, l_lim]
        _labels_ok(labels, segs, "H11 final")
        content = VGroup(sq, sq_inc, bar_r, ref, d2s, *labels)
        _safe(*content)
        cap = caption("d(√x) / dx  =  1 / (2√x)", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ============================================== H28  the shadow of a semicircle

class H28_SemicircleShadow(Board):
    """Walk round the unit semicircle from (1, 0) to (−1, 0); plotted against
    the distance walked, the walker's height is the sine arch.  Cut the walk
    into N equal steps (5, then 10, 20, 40).  Each step's chord c = 2 sin(dθ/2) is perpendicular
    to the radius at its mid-angle θ, so the step's right triangle is that
    radius triangle (hypotenuse 1, legs cos θ, sin θ — the height, in
    orange) shrunk by c and turned a quarter: the step moves sideways by
    exactly c · sin θ.  The sideways
    moves tile the diameter: they add up to 2.  Unroll the chords onto the
    axis and stand a strip of height sin θ on each: the strips have total
    area 2 at every N.  As N grows the chords hug the arc (c → dθ) and the
    strips fill the region under the sine arch: its area is 2."""

    def construct(self):
        Rs = 2.3
        O = np.array([-4.1, -0.35, 0.0])
        G0 = np.array([-1.25, -0.35, 0.0])

        def C(t):
            return O + Rs * _polar(t)

        def Gr(t, y):
            return G0 + Rs * np.array([t, y, 0.0])

        def parts(N):
            d = PI / N
            c = 2 * np.sin(d / 2)                      # chord, in radii
            th = [k * d for k in range(N + 1)]
            tm = [(k + 0.5) * d for k in range(N)]
            return d, c, th, tm

        # ---- the claims
        for N in (5, 10, 20, 40):
            d, c, th, tm = parts(N)
            for kk in range(N):
                sh = np.cos(th[kk]) - np.cos(th[kk + 1])
                check(abs(sh - c * np.sin(tm[kk])) < 1e-12,
                      "a step moves sideways by c · sin(mid-angle)")
                q = C(th[kk]) + Rs * c * np.array([0.0, np.cos(tm[kk]), 0.0])
                check(abs(q[1] - C(th[kk + 1])[1]) < 1e-9,
                      "the step triangle's corner is level with the next point")
            check(abs(sum(np.cos(th[kk]) - np.cos(th[kk + 1]) for kk in range(N)) - 2) < 1e-12,
                  "the sideways moves tile the diameter: total 2")
            check(abs(sum(c * np.sin(t) for t in tm) - 2) < 1e-12, "the strips have area 2")
            check(N * c < PI, "the unrolled chords are a little shorter than the arc")
        check(PI - 40 * parts(40)[1] < 1e-3 and 1 - parts(40)[1] / parts(40)[0] < 1e-3,
              "at N = 40 the chords have closed onto the arc")
        check(G0[0] - (O[0] + Rs) > 0.4 and G0[0] + PI * Rs < SAFE_X - 0.3,
              "the semicircle and the graph stand apart, inside the frame")

        # ---- semicircle, diameter 2, and the axis for the graph
        arc = Arc(radius=Rs, start_angle=0, angle=PI, arc_center=O, color=GREY_A,
                  stroke_width=3)
        diam = Line(C(PI), C(0), color=GREY_B, stroke_width=2.5)
        dotO = Dot(O, radius=0.05)
        yd = O[1] - 0.45
        d2 = _dim([O[0] - Rs, yd, 0], [O[0] + Rs, yd, 0], WHITE, 0.1, 2.5)
        l2 = _on_base("2", O[0], yd - 0.45, 30)
        axis = Line(G0 + 0.1 * LEFT, Gr(PI, 0) + 0.25 * RIGHT, color=GREY_B, stroke_width=2)
        tk0, tkp = _tick(G0, UP, 0.08), _tick(Gr(PI, 0), UP, 0.08)
        l0 = _on_base("0", G0[0], G0[1] - 0.42, 26)
        lp = _on_base("π", Gr(PI, 0)[0], G0[1] - 0.42, 26)
        self.play(Create(arc), Create(diam), FadeIn(dotO), Create(axis), FadeIn(tk0),
                  FadeIn(tkp), FadeIn(l0), FadeIn(lp), run_time=1.0)
        self.play(Create(d2), FadeIn(l2), run_time=0.5)

        # ---- the walker's height, plotted against the distance walked
        wt = ValueTracker(0.0)

        def walker():
            t = wt.get_value()
            p, g = C(t), Gr(t, np.sin(t))
            grp = VGroup(DashedLine(p, g, color=GREY_B, stroke_width=1.5, dash_length=0.08),
                         Dot(p, radius=0.07, color=YELLOW_B), Dot(g, radius=0.07, color=YELLOW_B))
            if t > 1e-3:
                grp.add(_curve([Gr(u, np.sin(u)) for u in np.linspace(0, t, 100)], YELLOW_B, 4),
                        Arc(radius=Rs, start_angle=0, angle=t, arc_center=O,
                            color=YELLOW_B, stroke_width=4))
            return grp

        wk = always_redraw(walker)
        self.add(wk)
        self.play(wt.animate.set_value(PI), run_time=2.6, rate_func=linear)
        wk.clear_updaters()
        self.remove(wk)
        sine = _curve([Gr(u, np.sin(u)) for u in np.linspace(0, PI, 160)], YELLOW_B, 4)
        lsine = tag("sin θ", 28, YELLOW_B).move_to(Gr(PI / 2, 1.0) + 0.35 * UP)
        self.add(sine)
        self.play(FadeIn(lsine), run_time=0.5)

        # ---- five steps: each one's sideways move, dropped onto the diameter
        cols5 = [BLUE_D, TEAL_D, PURPLE_B, GREEN_D, MAROON_B]

        def colour(N, kk):
            return cols5[kk % 5] if N == 5 else (BLUE_D if kk % 2 == 0 else TEAL_D)

        def steps_fig(N, with_tri=True):
            d, c, th, tm = parts(N)
            g = VGroup()
            g.add(_curve([C(t) for t in th], WHITE, 2.5 if N <= 10 else 1.5))
            for kk in range(N):
                p, q = C(th[kk]), C(th[kk + 1])
                corner = np.array([p[0], q[1], 0.0])
                col = colour(N, kk)
                if with_tri:
                    g.add(Line(p, corner, color=GREY_B, stroke_width=1.5),
                          Line(corner, q, color=col, stroke_width=5))
                g.add(Line([q[0], O[1], 0], [p[0], O[1], 0], color=col,
                           stroke_width=9 if N <= 10 else 7))
            return g

        def drops(N):
            d, c, th, tm = parts(N)
            g = VGroup()
            for kk in range(N + 1):
                p = C(th[kk])
                top = p if kk == 0 else np.array([p[0], p[1], 0.0])
                g.add(DashedLine(top, [p[0], O[1], 0], color=GREY_D, stroke_width=1.2,
                                 dash_length=0.07))
            return g

        f5 = steps_fig(5)
        dr5 = drops(5)
        self.play(Create(f5[0]), run_time=0.8)
        self.play(*[Create(m) for m in f5[1:]], Create(dr5), run_time=1.4)
        self.hold(0.4)
        self.play(FadeOut(dr5), run_time=0.4)

        # ---- one step: the radius triangle at its mid-angle, shrunk by the
        # ---- chord c and turned a quarter, is the step's triangle exactly
        d5, c5, th5, tm5 = parts(5)
        kq = 1
        Pk, Pk1 = C(th5[kq]), C(th5[kq + 1])
        Pm, Fm = C(tm5[kq]), np.array([C(tm5[kq])[0], O[1], 0.0])
        corner = np.array([Pk[0], Pk1[1], 0.0])
        rad = VGroup(Line(O, Pm, color=WHITE, stroke_width=3),
                     Line(Fm, Pm, color=ORANGE, stroke_width=5))
        aO = angle_arc(O, C(0), Pm, radius=0.55, color=YELLOW_B, width=3)
        lth = tag("θ", 26, YELLOW_B).move_to(O + 0.8 * _polar(tm5[kq] / 2))
        l1 = _beside(tag("1", 26), O, Pm, LEFT, 0.12, at=0.55)
        tri = mk([O, Fm, Pm], ORANGE, 0.3, stroke_width=0)
        self.play(FadeIn(tri), Create(rad), Create(aO), FadeIn(lth), FadeIn(l1),
                  run_time=1.0)
        cp = mk([O, Fm, Pm], ORANGE, 0.55, stroke_width=2, stroke_color=ORANGE)
        self.add(cp)
        self.play(cp.animate.scale(c5, about_point=O).shift(Pk - O), run_time=1.3)
        self.play(Rotate(cp, angle=PI / 2, about_point=Pk), run_time=1.0)
        check(_landed(cp, [Pk, corner, Pk1]),
              "the shrunk, turned radius triangle lands on the step's triangle")
        lc = _beside(tag("c", 26), Pk, Pk1, O - Pk, 0.12, at=0.8)
        lcs = _beside(tag("c · sin θ", 26, ORANGE), corner, Pk1, UP, 0.12, at=0.5)
        check(abs(_cross(Pm - O, lc.get_center() - O)) / Rs > 0.35
              and abs(lc.get_center()[0] - Fm[0]) > 0.4,
              "the chord's label stands well away from the radius triangle's legs")
        self.play(FadeIn(lc), FadeIn(lcs), run_time=0.6)
        self.hold(0.8)

        # ---- unroll the chords onto the axis; a strip of height sin θ on each
        segs_fix = (_arc_segs(O, Rs, 0, PI, 90) + [_seg(C(PI), C(0))]
                    + _segs([C(t) for t in th5]))
        labs_demo = [lth, l1, lc, lcs]
        tri_segs = []
        for kk in range(5):
            p_, q_ = C(th5[kk]), C(th5[kk + 1])
            cr = np.array([p_[0], q_[1], 0.0])
            tri_segs += [_seg(p_, cr), _seg(cr, q_)]
        _labels_ok(labs_demo, segs_fix + [_seg(O, Pm), _seg(Fm, Pm)] + tri_segs
                   + _angle_segs(O, C(0), Pm, 0.55), "H28 one step")
        self.play(FadeOut(VGroup(tri, rad, aO, lth, l1, lc, cp)), run_time=0.5)
        chords = VGroup(*[Line(C(th5[kk]), C(th5[kk + 1]), color=colour(5, kk),
                               stroke_width=5) for kk in range(5)])
        self.add(chords)
        moves = []
        for kk, ch in enumerate(chords):
            start = Gr(kk * c5, 0)
            ang = -(tm5[kk] + PI / 2)
            moves.append(_rigid(ch, angle=ang, about=C(th5[kk]), shift=start - C(th5[kk])))
        self.play(*moves, FadeOut(lcs), run_time=1.6)
        for kk, ch in enumerate(chords):
            check(close(ch.get_start(), Gr(kk * c5, 0), 1e-6)
                  and close(ch.get_end(), Gr((kk + 1) * c5, 0), 1e-6),
                  "each chord lands end to end on the axis")

        def strips(N):
            d, c, th, tm = parts(N)
            return VGroup(*[_rect(Gr(kk * c, 0)[0], G0[1], Gr((kk + 1) * c, 0)[0],
                                  Gr(0, np.sin(tm[kk]))[1], colour(N, kk), 0.75,
                                  1.0 if N <= 10 else 0.0) for kk in range(N)])

        def links(N):
            d, c, th, tm = parts(N)
            return VGroup(*[DashedLine(C(tm[kk]), Gr(kk * c, np.sin(tm[kk])), color=GREY_D,
                                       stroke_width=1.2, dash_length=0.08)
                            for kk in range(N)])

        st5 = strips(5)
        lk5 = links(5)
        self.play(Create(lk5), run_time=0.7)
        self.play(*[GrowFromEdge(m, DOWN) for m in st5], run_time=1.0)
        self.bring_to_front(sine, chords)
        for kk in range(5):
            check(abs(abs(area(st5[kk].get_vertices())) / Rs ** 2
                      - (np.cos(th5[kk]) - np.cos(th5[kk + 1]))) < 1e-9,
                  "strip area = that step's sideways move")
        self.play(FadeOut(lk5), FadeOut(chords), run_time=0.5)
        self.hold(0.4)

        # ---- more steps: still exactly 2, and the strips fill the arch
        cur_f, cur_s = f5, st5
        for N, rt in ((10, 0.9), (20, 0.9), (40, 0.9)):
            nf, ns = steps_fig(N, with_tri=(N <= 20)), strips(N)
            self.play(FadeOut(cur_f), FadeOut(cur_s), FadeIn(nf), FadeIn(ns), run_time=rt)
            self.bring_to_front(sine)
            self.hold(0.5)
            cur_f, cur_s = nf, ns
        l2a = tag("2", 34).move_to(Gr(PI / 2, 0.42))

        # ---- label hygiene
        segs = (_arc_segs(O, Rs, 0, PI, 90) + [_seg(C(PI), C(0)), _seg(axis.get_start(),
                                                                       axis.get_end())]
                + _dim_segs([O[0] - Rs, yd, 0], [O[0] + Rs, yd, 0])
                + _segs([Gr(u, np.sin(u)) for u in np.linspace(0, PI, 160)]))
        labels = [l2, l0, lp, lsine]
        _labels_ok(labels, segs, "H28 figure")
        check(_hit(l2a, segs, 0.06) is None, "the area label clears the curve")
        under = [Gr(u, np.sin(u)) for u in np.linspace(0, PI, 160)]
        check(all(_pip(q, under) for q in (l2a.get_corner(UL), l2a.get_corner(UR),
                                            l2a.get_corner(DL), l2a.get_corner(DR))),
              "the area label sits under the arch")
        self.play(FadeIn(l2a), run_time=0.6)
        content = VGroup(arc, diam, d2, axis, sine, cur_s, *labels, l2a)
        _safe(*content)
        cap = _cap(_row(_int("0", "π", 31), tag("sin θ dθ   =   2", 33, YELLOW_B), buff=0.14))
        _below(cap, *content)
        self.play(FadeIn(cap), run_time=1.0)
        self.hold(2.2)


# ============================================== H31  the arc-length element

class H31_ArcLengthElement(Board):
    """A step dx along the curve y = f(x) rises by Δy; the chord of the step
    is the hypotenuse of the right triangle with legs dx and Δy, so its
    length is exactly √(dx² + Δy²).  The arc Δs is a little longer.  In a
    window rescaled by 1/dx the step keeps unit width: as dx → 0 the arc
    straightens onto the chord and the triangle settles onto the dashed
    tangent triangle (legs 1 and f′(x)), so Δs / √(dx² + Δy²) → 1 and
    ds = √(dx² + dy²).  (Drawn on a convex piece of curve, where the arc
    lies between the chord and the tangent.)"""

    def construct(self):
        def f(u):
            return 0.25 * u * u - 0.1 * u + 0.15

        def fp(u):
            return 0.5 * u - 0.1

        u0, d0, d1 = 1.0, 1.3, 0.012
        k = 1.55
        Z = np.array([-6.3, -2.2, 0.0])             # screen point of (0, 0)

        def M(u, y):
            return Z + k * np.array([u, y, 0.0])

        P = M(u0, f(u0))
        lt = ValueTracker(np.log(d0))

        def dnow():
            return float(np.exp(lt.get_value()))

        def arclen(a, b, n=2000):
            us = np.linspace(a, b, n + 1)
            ys = np.sqrt(1 + fp(us) ** 2)
            return float((b - a) / n * (ys[0] / 2 + ys[1:-1].sum() + ys[-1] / 2))

        # ---- the claims
        m0 = fp(u0)
        for d in np.geomspace(d1, d0, 30):
            dy = f(u0 + d) - f(u0)
            ch = float(np.hypot(d, dy))
            s_ = arclen(u0, u0 + d)
            check(ch <= s_ + 1e-12, "the arc is at least as long as the chord")
            us = np.linspace(u0, u0 + d, 50)
            check(all(f(u) <= f(u0) + dy / d * (u - u0) + 1e-12 for u in us)
                  and all(f(u) >= f(u0) + m0 * (u - u0) - 1e-12 for u in us),
                  "convex: the arc lies between the chord and the tangent")
        dy1 = f(u0 + d1) - f(u0)
        L = 4.4
        check((arclen(u0, u0 + d1) - np.hypot(d1, dy1)) / d1 * L < 0.005,
              "in the window the arc has merged with the chord")
        check(abs(dy1 / d1 - m0) * L < 0.03, "the triangle has reached the tangent's slope")

        # ---- the curve and a step dx
        us_all = np.linspace(0.0, 3.3, 200)
        curve = _curve([M(u, f(u)) for u in us_all], BLUE_C, 4)
        lf = tag("y = f(x)", 28, BLUE_C).move_to(M(3.05, f(3.3)) + np.array([0.15, 0.32, 0]))
        dotP = Dot(P, radius=0.06)
        self.play(Create(curve), FadeIn(lf), FadeIn(dotP), run_time=1.2)

        def step_fig():
            d = dnow()
            P1 = M(u0 + d, f(u0 + d))
            Cn = np.array([P1[0], P[1], 0.0])
            return VGroup(Line(P, Cn, color=TEAL_B, stroke_width=4),
                          Line(Cn, P1, color=ORANGE, stroke_width=4),
                          Line(P, P1, color=WHITE, stroke_width=2.5),
                          _curve([M(u, f(u)) for u in np.linspace(u0, u0 + d, 60)],
                                 YELLOW_B, 6),
                          Dot(P1, radius=0.06, color=YELLOW_B))

        sf = always_redraw(step_fig)
        P1_0 = M(u0 + d0, f(u0 + d0))
        Cn0 = np.array([P1_0[0], P[1], 0.0])
        l_dx = tag("dx", 26, TEAL_B).next_to(Line(P, Cn0), DOWN, buff=0.14)
        l_dy = tag("Δy", 26, ORANGE).next_to(Line(Cn0, P1_0), RIGHT, buff=0.14)
        self.play(FadeIn(sf), FadeIn(l_dx), FadeIn(l_dy), run_time=1.0)
        self.hold(0.4)

        # ---- the window: everything near P magnified by L/dx
        anc = np.array([0.45, -2.05, 0.0])           # P in the window
        rise0 = (f(u0 + d0) - f(u0)) / d0
        wl, wr = anc[0] - 0.35, anc[0] + L + 0.35
        wb, wt = anc[1] - 0.35, anc[1] + L * rise0 + 0.35
        for d in np.geomspace(d1, d0, 25):
            check((f(u0 + d) - f(u0)) / d <= rise0 + 1e-12, "the window holds the step")
        window = Polygon([wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0],
                         stroke_color=GREY_B, stroke_width=2)

        def to_win(p):
            return anc + (p - P) * (L / (k * dnow()))

        def zoom():
            d = dnow()
            sc = k * d / L
            lo = P + (np.array([wl, wb, 0]) - anc) * sc
            hi = P + (np.array([wr, wt, 0]) - anc) * sc
            box = Polygon(lo, [hi[0], lo[1], 0], hi, [lo[0], hi[1], 0],
                          stroke_color=GREY_B, stroke_width=1.5)
            return VGroup(box,
                          DashedLine([hi[0], hi[1], 0], [wl, wt, 0], color=GREY_D,
                                     stroke_width=1.2),
                          DashedLine([hi[0], lo[1], 0], [wl, wb, 0], color=GREY_D,
                                     stroke_width=1.2))

        ref = mk([anc, anc + L * RIGHT, anc + L * np.array([1.0, m0, 0.0])], YELLOW_E,
                 0.22, stroke_width=0)
        ref_edge = DashedVMobject(Polygon(anc, anc + L * RIGHT,
                                          anc + L * np.array([1.0, m0, 0.0]),
                                          stroke_color=WHITE, stroke_width=2),
                                  num_dashes=44)

        def inside():
            d = dnow()
            P1 = M(u0 + d, f(u0 + d))
            Cn = np.array([P1[0], P[1], 0.0])
            return VGroup(Line(to_win(P), to_win(Cn), color=TEAL_B, stroke_width=5),
                          Line(to_win(Cn), to_win(P1), color=ORANGE, stroke_width=5),
                          Line(to_win(P), to_win(P1), color=WHITE, stroke_width=3),
                          _curve([to_win(M(u, f(u))) for u in np.linspace(u0, u0 + d, 80)],
                                 YELLOW_B, 6),
                          Dot(anc, radius=0.06))

        zm = always_redraw(zoom)
        ins = always_redraw(inside)
        self.play(Create(window), FadeIn(zm), run_time=1.0)
        self.play(FadeIn(ref), Create(ref_edge), FadeIn(ins), run_time=0.9)
        # labels in and beside the window
        top0 = anc + L * np.array([1.0, rise0, 0.0])
        w_dx = tag("dx", 28, TEAL_B).next_to(np.array([anc[0] + L / 2, wb, 0.0]), DOWN,
                                             buff=0.12)
        w_dy = tag("Δy", 28, ORANGE).next_to(np.array([wr, anc[1] + L * rise0 / 2, 0]),
                                              RIGHT, buff=0.16)
        w_ch = _beside(tag("√(dx² + Δy²)", 26), anc, top0, UP + LEFT, 0.14, at=0.42)
        # between the arc and the dashed tangent, near the step's far end
        Xs = 0.88 * L
        Ys_arc = (f(u0 + Xs * d0 / L) - f(u0)) / d0 * L
        w_s = tag("Δs", 28, YELLOW_B).move_to(
            anc + np.array([Xs, m0 * Xs + 0.4 * (Ys_arc - m0 * Xs), 0]))
        self.play(FadeIn(w_dx), FadeIn(w_dy), FadeIn(w_ch), FadeIn(w_s), run_time=0.8)
        self.hold(0.6)

        # ---- dx → 0: the arc straightens onto the chord
        self.play(FadeOut(w_s), FadeOut(l_dx), FadeOut(l_dy), run_time=0.4)
        self.play(lt.animate.set_value(np.log(d1)), run_time=4.6, rate_func=smooth)
        for m in (sf, zm, ins):
            m.clear_updaters()
        top1 = anc + L * np.array([1.0, m0, 0.0])
        w_ds = _beside(tag("ds", 28, YELLOW_B), anc, top1, UP + LEFT, 0.14, at=0.45)
        w_dy2 = tag("dy", 28, ORANGE).next_to(np.array([wr, anc[1] + L * m0 / 2, 0]), RIGHT,
                                               buff=0.16)
        fin = _row(tag("ds", 28, YELLOW_B), tag("=", 28), tag("√(dx² + dy²)", 28), buff=0.14)
        fin.next_to(np.array([(wl + wr) / 2, wt, 0.0]), UP, buff=0.2)
        self.play(FadeOut(w_ch), FadeOut(w_dy), FadeIn(w_dy2), FadeIn(fin), run_time=0.9)

        # ---- label hygiene
        P1_1 = M(u0 + d1, f(u0 + d1))
        segs_main = (_segs([M(u, f(u)) for u in us_all])
                     + [_seg(P, Cn0), _seg(Cn0, P1_0), _seg(P, P1_0)])
        _labels_ok([lf, l_dx, l_dy], segs_main, "H31 main figure")
        segs_w0 = (_segs([[wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0]], closed=True)
                   + [_seg(anc, anc + L * RIGHT), _seg(anc + L * RIGHT, top0), _seg(anc, top0),
                      _seg(anc + L * RIGHT, top1), _seg(anc, top1)]
                   + _segs([anc + (M(u, f(u)) - P) * (L / (k * d0))
                            for u in np.linspace(u0, u0 + d0, 80)]))
        _labels_ok([w_dx, w_dy, w_ch, w_s], segs_w0, "H31 window, first step")
        segs_w1 = (_segs([[wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0]], closed=True)
                   + [_seg(anc, anc + L * RIGHT), _seg(anc + L * RIGHT, top1), _seg(anc, top1)])
        _labels_ok([w_dx, w_dy2, w_ds, fin], segs_w1, "H31 window, limit")
        check(not any(_overlap(m, n) for m in (lf, l_dx, l_dy) for n in (w_dx, w_dy, w_ch, fin)),
              "the main labels clear the window labels")
        for p_ in (P, P1_0):
            check(p_[0] < wl - 0.6, "the main figure stands left of the window")
        self.play(FadeIn(w_ds), run_time=0.5)
        content = VGroup(curve, window, ref_edge, lf, w_dx, w_dy2, w_ds, fin)
        _safe(*content)
        cap = caption("ds  =  √(dx² + dy²)", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ============================================== H12  derivative of 1/x

class H12_ReciprocalDerivative(Board):
    """Every rectangle from O to a point of y = 1/x has area 1.  Widen it
    from x to x + dx: to keep the area 1 its height drops by −Δ(1/x), and
    the top strip T it loses equals the right strip S it gains.  So the
    common rectangle x × 1/(x + dx), enlarged from O by (x + dx)/x in both
    directions, lands exactly on the bounding box (x + dx) × 1/x: the
    box's diagonal runs through the common corner, and the corner cell K
    (width dx, height −Δ(1/x)) lies on that diagonal — a small copy of the
    box, so −Δ(1/x)/dx = (1/x)/(x + dx).  In a window rescaled by 1/dx the
    cell has width 1; as dx → 0 the arc of the hyperbola across it
    straightens and the cell settles onto the dashed 1 × 1/x² cell, the
    shape of the x × 1/x rectangle: d(1/x) = −dx/x²."""

    def construct(self):
        x, d0, d1 = 1.2, 0.5, 0.008
        k = 2.2
        Z = np.array([-5.15, -2.3, 0.0])            # screen point of (0, 0)

        def M(u, y):
            return Z + k * np.array([u, y, 0.0])

        lt = ValueTracker(np.log(d0))

        def dnow():
            return float(np.exp(lt.get_value()))

        # ---- the claims
        for d in np.geomspace(d1, d0, 30):
            y1, y2 = 1 / x, 1 / (x + d)
            check(abs(x * y1 - 1) < 1e-12 and abs((x + d) * y2 - 1) < 1e-12,
                  "both rectangles have area 1")
            check(abs(x * (y1 - y2) - d * y2) < 1e-12, "lost strip T = gained strip S")
            fac = (x + d) / x
            check(abs(fac * x - (x + d)) < 1e-12 and abs(fac * y2 - y1) < 1e-12,
                  "the common rectangle, enlarged from O by (x + dx)/x, is the bounding box")
            check(abs((y1 - y2) / d - y1 / (x + d)) < 1e-12,
                  "the corner cell is a copy of the box: −Δ(1/x)/dx = (1/x)/(x + dx)")
        L = 3.5
        check(abs((1 / x - 1 / (x + d1)) / d1 - 1 / x ** 2) * L < 0.03,
              "in the window the cell has reached height 1/x²")

        # ---- axes, hyperbola, the rectangle x × 1/x of area 1
        us = np.linspace(0.45, 2.3, 220)
        axes = VGroup(Line(M(0, 0), M(2.4, 0), color=GREY_B, stroke_width=2),
                      Line(M(0, 0), M(0, 2.3), color=GREY_B, stroke_width=2))
        hyp = _curve([M(u, 1 / u) for u in us], YELLOW_B, 4)
        lhyp = tag("y = 1/x", 28, YELLOW_B).move_to(M(0.93, 2.0))
        self.play(Create(axes), Create(hyp), FadeIn(lhyp), run_time=1.0)
        R1 = _rect(*M(0, 0)[:2], *M(x, 1 / x)[:2], BLUE_D, 0.6, 2)
        bx = Z[1] - 0.16 - Text("M", font_size=28).height
        l_x = _on_base("x", M(x / 2, 0)[0], bx, 28)
        l_ix = tag("1/x", 28).next_to(M(0, 1 / x), LEFT, buff=0.16)
        tk_ix = _tick(M(0, 1 / x), RIGHT, 0.08)
        l_one = tag("1", 34).move_to(M(0.32, 0.43))
        self.play(FadeIn(R1), FadeIn(l_x), FadeIn(l_ix), FadeIn(tk_ix), FadeIn(l_one),
                  run_time=0.9)
        self.hold(0.3)

        # ---- widen by dx along the hyperbola: the area stays 1
        R1o = DashedVMobject(Polygon(M(0, 0), M(x, 0), M(x, 1 / x), M(0, 1 / x),
                                     stroke_color=WHITE, stroke_width=2), num_dashes=50)
        self.add(R1o)
        ut = ValueTracker(x)
        Rmov = always_redraw(lambda: _rect(*M(0, 0)[:2],
                                           *M(ut.get_value(), 1 / ut.get_value())[:2],
                                           BLUE_D, 0.6, 2))
        self.remove(R1)
        self.add(Rmov)
        self.bring_to_front(R1o, l_one, hyp)
        self.play(ut.animate.set_value(x + d0), run_time=1.6)
        Rmov.clear_updaters()
        l_dx = _on_base("dx", M(x + d0 / 2, 0)[0], bx, 26)
        self.play(FadeIn(l_dx), run_time=0.4)

        # ---- the strip lost (T) equals the strip gained (S)
        def strips(d):
            y1, y2 = 1 / x, 1 / (x + d)
            return (_rect(*M(0, y2)[:2], *M(x, y1)[:2], ORANGE, 0.75, 1.5),
                    _rect(*M(x, 0)[:2], *M(x + d, y2)[:2], ORANGE, 0.75, 1.5))

        T0, S0 = strips(d0)
        self.play(FadeIn(T0), FadeIn(S0), run_time=0.8)
        self.bring_to_front(R1o, l_one, hyp)
        self.hold(0.4)

        # ---- the common rectangle, enlarged from O by (x + dx)/x, is the box
        def box(d):
            return Polygon(M(0, 0), M(x + d, 0), M(x + d, 1 / x), M(0, 1 / x),
                           stroke_color=TEAL_B, stroke_width=3, fill_opacity=0)

        Bo = box(d0)
        cp = _rect(*M(0, 0)[:2], *M(x, 1 / (x + d0))[:2], TEAL_D, 0.5, 2)
        self.play(FadeIn(cp), run_time=0.4)
        self.play(cp.animate.scale((x + d0) / x, about_point=M(0, 0)), run_time=1.3)
        check(_same_poly(cp.get_vertices(), [M(0, 0), M(x + d0, 0), M(x + d0, 1 / x),
                                             M(0, 1 / x)], 1e-6),
              "the enlarged common rectangle lands exactly on the box")
        self.add(Bo)
        self.play(FadeOut(cp), run_time=0.4)

        def diag(d):
            return DashedLine(M(0, 0), M(x + d, 1 / x), color=TEAL_B, stroke_width=2.5,
                              dash_length=0.1)

        def cell(d):
            return _rect(*M(x, 1 / (x + d))[:2], *M(x + d, 1 / x)[:2], YELLOW_E, 0.85, 2)

        Dg, K0 = diag(d0), cell(d0)
        self.play(Create(Dg), run_time=0.8)
        self.play(FadeIn(K0), run_time=0.5)
        self.bring_to_front(hyp, Dg)
        self.hold(0.3)

        # ---- the window: the cell K magnified by L/dx
        Pw = np.array([0.95, 1.65, 0.0])              # the cell's top-left corner
        hL = L / x ** 2                               # limit height
        h0w = L * (1 / x - 1 / (x + d0)) / d0
        wl, wr = Pw[0] - 0.3, Pw[0] + L + 0.3
        wt, wb = Pw[1] + 0.3, Pw[1] - hL - 0.3
        window = Polygon([wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0],
                         stroke_color=GREY_B, stroke_width=2)
        Pm = M(x, 1 / x)

        def to_win(p):
            return Pw + (p - Pm) * (L / (k * dnow()))

        def zoom():
            d = dnow()
            sc = k * d / L
            lo = Pm + (np.array([wl, wb, 0]) - Pw) * sc
            hi = Pm + (np.array([wr, wt, 0]) - Pw) * sc
            return VGroup(Polygon(lo, [hi[0], lo[1], 0], hi, [lo[0], hi[1], 0],
                                  stroke_color=GREY_B, stroke_width=1.5),
                          DashedLine([lo[0], hi[1], 0], [wl, wt, 0], color=GREY_D,
                                     stroke_width=1.2),
                          DashedLine([lo[0], lo[1], 0], [wl, wb, 0], color=GREY_D,
                                     stroke_width=1.2))

        def inside():
            d = dnow()
            hh = L * (1 / x - 1 / (x + d)) / d
            g = VGroup(_rect(Pw[0], Pw[1] - hh, Pw[0] + L, Pw[1], YELLOW_E, 0.6, 2),
                       DashedLine([Pw[0], Pw[1] - hh, 0], [Pw[0] + L, Pw[1], 0],
                                  color=TEAL_B, stroke_width=2, dash_length=0.1),
                       _curve([to_win(M(u, 1 / u)) for u in np.linspace(x, x + d, 80)],
                              YELLOW_B, 5),
                       Dot(Pw, radius=0.06))
            return g

        ref = DashedVMobject(Polygon(Pw, Pw + L * RIGHT, Pw + L * RIGHT + hL * DOWN,
                                     Pw + hL * DOWN, stroke_color=WHITE, stroke_width=2),
                             num_dashes=60)
        zm = always_redraw(zoom)
        ins = always_redraw(inside)
        self.play(Create(window), FadeIn(zm), run_time=0.9)
        self.play(FadeIn(ins), Create(ref), run_time=0.8)
        w_dx = tag("dx", 28).next_to(np.array([Pw[0] + L / 2, wt, 0.0]), UP, buff=0.12)
        w_h = tag("−Δ(1/x)", 26, YELLOW_B).next_to(
            np.array([wr, Pw[1] - h0w / 2, 0.0]), RIGHT, buff=0.16)
        self.play(FadeIn(w_dx), FadeIn(w_h), run_time=0.6)
        self.hold(0.6)

        # ---- dx → 0: the cell becomes 1 × 1/x², the arc straightens
        main_r = always_redraw(lambda: VGroup(
            _rect(*M(0, 0)[:2], *M(x + dnow(), 1 / (x + dnow()))[:2], BLUE_D, 0.6, 2),
            *strips(dnow()), box(dnow()), cell(dnow()), diag(dnow())))
        self.remove(Rmov, T0, S0, Bo, Dg, K0)
        self.add(main_r)
        self.bring_to_front(R1o, l_one, hyp, l_x, l_ix)
        l_dx.add_updater(lambda m: m.set_x(M(x + dnow() / 2, 0)[0]))
        w_h.add_updater(lambda m: m.set_y(Pw[1] - L * (1 / x - 1 / (x + dnow())) / dnow() / 2))
        self.play(FadeOut(l_dx), run_time=0.3)
        l_dx.clear_updaters()
        self.play(lt.animate.set_value(np.log(d1)), run_time=4.6, rate_func=smooth)
        for m in (main_r, zm, ins, w_h):
            m.clear_updaters()
        w_lim = tag("dx/x²", 26, YELLOW_B).move_to(w_h.get_center())
        w_lim.align_to(w_h, LEFT)
        fin = _row(tag("d(1/x)", 28, YELLOW_B), tag("=  −dx/x²", 28), buff=0.14)
        fin.next_to(np.array([(wl + wr) / 2, wb, 0.0]), DOWN, buff=0.2)
        self.play(FadeOut(w_h), FadeIn(w_lim), FadeIn(fin), run_time=0.9)

        # ---- label hygiene
        segs_main = (_segs([M(u, 1 / u) for u in us])
                     + [_seg(M(0, 0), M(2.4, 0)), _seg(M(0, 0), M(0, 2.3))]
                     + _segs([M(0, 0), M(x, 0), M(x, 1 / x), M(0, 1 / x)], closed=True)
                     + _segs([M(0, 0), M(x + d0, 0), M(x + d0, 1 / x), M(0, 1 / x)],
                             closed=True)
                     + [_seg(M(x, 0), M(x, 1 / x)), _seg(M(0, 1 / (x + d0)), M(x + d0, 1 / (x + d0))),
                        _seg(M(0, 0), M(x + d0, 1 / x))])
        _labels_ok([lhyp, l_x, l_ix, l_one, l_dx], segs_main, "H12 main figure")
        segs_w = (_segs([[wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0]], closed=True)
                  + _segs([Pw, Pw + L * RIGHT, Pw + L * RIGHT + hL * DOWN, Pw + hL * DOWN],
                          closed=True))
        _labels_ok([w_dx, w_h], segs_w, "H12 window, first step")
        _labels_ok([w_dx, w_lim, fin], segs_w, "H12 window, limit")
        check(not any(_overlap(a_, b_) for a_ in (lhyp, l_x, l_ix, l_one)
                      for b_ in (w_dx, w_h, w_lim, fin)), "main labels clear the window's")
        check(M(2.4, 0)[0] + 0.25 < wl, "the main figure stands left of the window")
        content = VGroup(axes, hyp, window, ref, lhyp, l_x, l_ix, l_one, w_dx, w_lim, fin)
        _safe(*content)
        cap = caption("d(1/x) / dx  =  −1/x²", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)


# ============================================== H30  derivative of tan

class H30_TangentDerivative(Board):
    """On the tangent line x = 1 the ray at angle θ meets it at T, with
    XT = tan θ and OT = sec θ.  Turn the ray by dθ: T moves up the line by
    Δ(tan θ), while the arc of radius sec θ that it sweeps has length exactly
    sec θ · dθ and leaves T perpendicular to OT, at the angle θ to the line.
    The big triangle (1, tan θ, sec θ), shrunk by sec θ · dθ and folded into
    place at T, is the limit shape of the small one: its leg sec θ · dθ along
    the arc, its hypotenuse up the line.  A window rescaled by 1/dθ shows the
    arc straightening and the move up the line converging to
    sec θ · dθ / cos θ = sec²θ · dθ."""

    def construct(self):
        th = 32 * DEGREES
        c, s = float(np.cos(th)), float(np.sin(th))
        sec = 1 / c
        Rs = 3.6
        O = np.array([-6.25, -2.45, 0.0])

        def U(p):
            return O + Rs * to3(p)

        X = U((1.0, 0.0))
        T = U((1.0, np.tan(th)))
        d0, d1 = 0.25, 0.01
        lt = ValueTracker(np.log(d0))

        def dth():
            return float(np.exp(lt.get_value()))

        def Tp(d):
            return U((1.0, np.tan(th + d)))

        def Aend(d):                                  # end of the arc of radius sec θ
            return O + Rs * sec * _polar(th + d)

        # ---- the claims
        for d in np.geomspace(d1, d0, 25):
            dt = np.tan(th + d) - np.tan(th)
            check(abs(dt - sec * np.sin(d) / np.cos(th + d)) < 1e-12,
                  "Δ(tan θ) = sec θ · sin dθ · sec(θ + dθ)")
            check(abs(_cross(Aend(d) - O, Tp(d) - O)) < 1e-9
                  and np.linalg.norm(Aend(d) - O) < np.linalg.norm(Tp(d) - O),
                  "the arc ends on the new ray, inside OT'")
        check(abs(_ang(T, O, T + _polar(th + PI / 2)) - PI / 2) < 1e-12,
              "the arc leaves T perpendicular to OT")
        check(abs(_ang(T, T + UP, T + _polar(th + PI / 2)) - th) < 1e-12,
              "and at the angle θ to the line x = 1")
        phi = th / 2 + PI / 4
        sc = sec * d0

        def fold_img(Pt):
            q = T + sc * (to3(Pt) - O)
            return _mirror(q, T, T + _polar(phi))
        A_lim = T + Rs * sc * _polar(th + PI / 2)
        T_lim = T + Rs * sc * sec * UP
        check(close(fold_img(O), T) and close(fold_img(X), A_lim) and close(fold_img(T), T_lim),
              "shrunk by sec θ · dθ and folded at T, the big triangle is the limit shape")
        L = 2.4
        r1 = (np.tan(th + d1) - np.tan(th)) / d1
        check(abs(r1 - sec * sec) * L < 0.03, "in the window the move has become sec²θ · dθ")

        # ---- the quarter circle, the tangent line, the big triangle
        quarter = Arc(radius=Rs, start_angle=0, angle=PI / 2, arc_center=O, color=GREY_B,
                      stroke_width=2.5)
        axes = VGroup(Line(O, O + RIGHT * (Rs + 0.5), color=GREY_B, stroke_width=2),
                      Line(O, O + UP * (Rs + 0.3), color=GREY_B, stroke_width=2))
        tline = Line(X + 0.0 * DOWN, U((1.0, 1.25)), color=GREY_A, stroke_width=2.5)
        OX = Line(O, X, color=TEAL_B, stroke_width=5)
        XT = Line(X, T, color=BLUE_C, stroke_width=5)
        OT = Line(O, T, color=WHITE, stroke_width=3.5)
        aO = angle_arc(O, X, T, radius=0.8, color=YELLOW_B, width=4)
        lth = tag("θ", 28, YELLOW_B).move_to(O + 1.12 * _polar(th / 2))
        l1 = _on_base("1", (O[0] + X[0]) / 2, O[1] - 0.45, 28, TEAL_B)
        ltan = tag("tan θ", 26, BLUE_C).next_to(XT, RIGHT, buff=0.14)
        lsec = _beside(tag("sec θ", 26), O, T, DOWN + RIGHT, 0.12, at=0.55)
        dT = Dot(T, radius=0.06)
        self.play(Create(axes), Create(quarter), Create(tline), run_time=1.0)
        self.play(Create(OX), Create(XT), Create(OT), FadeIn(dT), run_time=1.0)
        self.play(Create(aO), FadeIn(lth), FadeIn(l1), FadeIn(ltan), FadeIn(lsec),
                  run_time=0.8)
        self.hold(0.3)

        # ---- turn the ray by dθ: T climbs the line; the arc sweeps sec θ · dθ
        def small():
            d = dth()
            return VGroup(Line(O, Tp(d), color=GREY_A, stroke_width=2.5),
                          Arc(radius=Rs * sec, start_angle=th, angle=d, arc_center=O,
                              color=YELLOW_B, stroke_width=5),
                          Line(T, Tp(d), color=ORANGE, stroke_width=6),
                          angle_arc(O, T, Tp(d), radius=2.1, color=YELLOW_B, width=3),
                          Dot(Tp(d), radius=0.06, color=ORANGE))

        sm = always_redraw(small)
        ldth = tag("dθ", 24, YELLOW_B).move_to(O + 3.05 * _polar(th + d0 / 2))
        self.play(FadeIn(sm), FadeIn(ldth), run_time=1.0)
        self.hold(0.4)

        # ---- the big triangle shrunk by sec θ · dθ and folded into place
        big = mk([O, X, T], YELLOW_E, 0.35, stroke_width=2)
        cp = big.copy()
        self.add(cp)
        self.play(cp.animate.scale(sc, about_point=O).shift(T - O), run_time=1.4)
        self.play(Rotate(cp, angle=PI, axis=_polar(phi), about_point=T), run_time=1.2)
        check(_landed(cp, [T, A_lim, T_lim]), "the folded copy is the limit triangle at T")

        def ref_copy():
            d = dth()
            k_ = Rs * sec * d
            return mk([T, T + k_ * _polar(th + PI / 2), T + k_ * sec * UP], YELLOW_E,
                      0.35, stroke_width=2)

        self.remove(cp)
        rc = always_redraw(ref_copy)
        self.add(rc)
        self.bring_to_front(sm, dT)
        self.hold(0.3)

        # ---- the window: everything near T magnified by L/dθ
        anc = np.array([3.95, -2.3, 0.0])            # T in the window
        rise0 = (np.tan(th + d0) - np.tan(th)) / d0
        aw0 = (Aend(d0) - T) * (L / (Rs * d0))
        wl = min(anc[0] + aw0[0], anc[0] - L * sec * s) - 0.3
        wr, wb, wt = anc[0] + 0.3, anc[1] - 0.3, anc[1] + L * rise0 + 0.3
        for d in np.geomspace(d1, d0, 25):
            aw = (Aend(d) - T) * (L / (Rs * d))
            check((np.tan(th + d) - np.tan(th)) / d <= rise0 + 1e-12
                  and anc[0] + aw[0] >= wl + 0.25, "the magnified figure stays in the window")
        window = Polygon([wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0],
                         stroke_color=GREY_B, stroke_width=2)

        def to_win(p):
            return anc + (p - T) * (L / (Rs * dth()))

        def zoom():
            d = dth()
            k_ = Rs * d / L
            lo = T + (np.array([wl, wb, 0]) - anc) * k_
            hi = T + (np.array([wr, wt, 0]) - anc) * k_
            return VGroup(Polygon(lo, [hi[0], lo[1], 0], hi, [lo[0], hi[1], 0],
                                  stroke_color=GREY_B, stroke_width=1.5),
                          DashedLine([hi[0], hi[1], 0], [wl, wt, 0], color=GREY_D,
                                     stroke_width=1.2),
                          DashedLine([hi[0], lo[1], 0], [wl, wb, 0], color=GREY_D,
                                     stroke_width=1.2))

        A_w = anc + L * sec * _polar(th + PI / 2)
        T_w = anc + L * sec * sec * UP
        ref_w = mk([anc, A_w, T_w], YELLOW_E, 0.28, stroke_width=0)
        ref_edge = DashedVMobject(Polygon(anc, A_w, T_w, stroke_color=WHITE, stroke_width=2),
                                  num_dashes=44)

        def inside():
            d = dth()
            arc_pts = [to_win(O + Rs * sec * _polar(th + u * d)) for u in np.linspace(0, 1, 60)]
            return VGroup(_curve(arc_pts, YELLOW_B, 5),
                          Line(to_win(Aend(d)), to_win(Tp(d)), color=GREY_A, stroke_width=3),
                          Line(anc, to_win(Tp(d)), color=ORANGE, stroke_width=6),
                          Dot(anc, radius=0.06))

        zm = always_redraw(zoom)
        ins = always_redraw(inside)
        w_ang = angle_arc(anc, anc + UP, A_w, radius=0.75)
        w_th = tag("θ", 28, YELLOW_B).move_to(anc + 1.05 * angle_mid_dir(anc, anc + UP, A_w))
        w_sec = tag("sec θ · dθ", 26).next_to(
            np.array([wl, anc[1] + 0.55 * L * sec * c, 0.0]), LEFT, buff=0.15)
        w_rise = tag("Δ(tan θ)", 26, ORANGE).next_to(
            np.array([wr, anc[1] + 0.5 * L * rise0, 0.0]), RIGHT, buff=0.15)
        self.play(Create(window), FadeIn(zm), run_time=1.0)
        self.play(FadeIn(ref_w), Create(ref_edge), FadeIn(ins), Create(w_ang), FadeIn(w_th),
                  FadeIn(w_sec), FadeIn(w_rise), run_time=1.0)
        self.hold(0.6)

        # ---- dθ → 0: the arc straightens, the move becomes sec²θ · dθ
        self.play(FadeOut(ldth), run_time=0.3)
        w_rise.add_updater(lambda m: m.set_y(
            anc[1] + 0.5 * L * (np.tan(th + dth()) - np.tan(th)) / dth()))
        self.play(lt.animate.set_value(np.log(d1)), run_time=4.6, rate_func=smooth)
        for m in (sm, rc, zm, ins, w_rise):
            m.clear_updaters()
        w_lim = tag("sec²θ · dθ", 26, ORANGE).move_to(w_rise.get_center())
        w_lim.align_to(w_rise, LEFT)
        fin = _row(tag("d(tan θ)", 28, ORANGE), tag("=  sec²θ · dθ", 28), buff=0.16)
        fin.next_to(np.array([(wl + wr) / 2, wt, 0.0]), UP, buff=0.18)
        self.play(FadeOut(w_rise), FadeIn(w_lim), FadeIn(fin), run_time=0.9)

        # ---- label hygiene
        T0p, A0 = Tp(d0), Aend(d0)
        segs_main = (_arc_segs(O, Rs, 0, PI / 2, 60)
                     + [_seg(O, O + RIGHT * (Rs + 0.5)), _seg(O, O + UP * (Rs + 0.3)),
                        _seg(X, U((1.0, 1.25))), _seg(O, T), _seg(O, T0p)]
                     + _arc_segs(O, Rs * sec, th, th + d0, 20)
                     + _angle_segs(O, X, T, 0.8) + _angle_segs(O, T, T0p, 2.1))
        _labels_ok([lth, l1, ltan, lsec, ldth], segs_main, "H30 main figure")
        segs_w0 = (_segs([[wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0]], closed=True)
                   + _segs([anc, A_w, T_w], closed=True)
                   + _segs([anc + (O + Rs * sec * _polar(th + u * d0) - T) * (L / (Rs * d0))
                            for u in np.linspace(0, 1, 40)])
                   + [_seg(anc, anc + L * rise0 * UP)]
                   + _angle_segs(anc, anc + UP, A_w, 0.75))
        _labels_ok([w_th, w_sec, w_rise], segs_w0, "H30 window, first step")
        _labels_ok([w_th, w_sec, w_lim, fin], _segs([[wl, wb, 0], [wr, wb, 0], [wr, wt, 0],
                                                     [wl, wt, 0]], closed=True)
                   + _segs([anc, A_w, T_w], closed=True) + _angle_segs(anc, anc + UP, A_w, 0.75),
                   "H30 window, limit")
        check(not any(_overlap(a_, b_) for a_ in (lth, l1, ltan, lsec)
                      for b_ in (w_th, w_sec, w_lim, fin)), "main labels clear the window's")
        content = VGroup(axes, quarter, tline, OX, XT, OT, window, ref_edge, lth, l1, ltan,
                         lsec, w_th, w_sec, w_lim, fin)
        _safe(*content)
        cap = caption("d(tan θ) / dθ  =  sec²θ", 36)
        _below(cap, *content)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.4)
