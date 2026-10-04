# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2j.py — proofs without words, 2D (manim):
#     J16 the pentagon's golden ratio        J19 the fractions are countable
#     J13 Fermat's little theorem (strings)  J18 the staircase paradox
#     J14 Zagier's windmills                 J11 root 3 is irrational
#     J22 Dudeney's hinged dissection        J27 64 = 65
#     K19 Fibonacci numbers in Pascal's triangle
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Every scene checks its own claims with check(...) before it animates —
# angles, landings of moved pieces, tilings, counts and bijections — so a
# wrong construction fails the render instead of drawing a wrong picture.
# J18 and J27 are cautions: they draw the false picture first and then show
# exactly where it fails.

from math import comb, gcd


# ------------------------------------------------------------ helpers

def _u(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _dir(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _rot(p, t, about=(0.0, 0.0)):
    """The 2D point p turned by the angle t about the point `about`."""
    p = np.asarray(p, float)[:2]
    o = np.asarray(about, float)[:2]
    cs, sn = np.cos(t), np.sin(t)
    d = p - o
    return o + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])


def _meet(p1, p2, q1, q2):
    """Intersection of the lines p1p2 and q1q2 (2D or 3D screen points)."""
    p1, p2, q1, q2 = (np.asarray(x, float)[:2] for x in (p1, p2, q1, q2))
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    return to3(p1 + t * d1)


def _foot(p, a, b):
    """Foot of the perpendicular from p to the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = _u(b - a)
    return a + np.dot(p - a, d) * d


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


def _clip(subj, clipp):
    """Sutherland–Hodgman: polygon subj ∩ convex polygon clipp (CCW)."""
    def inside(p, a1, a2):
        return ((a2[0] - a1[0]) * (p[1] - a1[1])
                - (a2[1] - a1[1]) * (p[0] - a1[0])) >= -1e-12

    def inter(p1, p2, a1, a2):
        d1, d2 = p2 - p1, a2 - a1
        den = d1[0] * d2[1] - d1[1] * d2[0]
        t = ((a1[0] - p1[0]) * d2[1] - (a1[1] - p1[1]) * d2[0]) / den
        return p1 + t * d1

    out = [np.asarray(p, float)[:2] for p in subj]
    n = len(clipp)
    for i in range(n):
        a1 = np.asarray(clipp[i], float)[:2]
        a2 = np.asarray(clipp[(i + 1) % n], float)[:2]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a1, a2):
                if not inside(s, a1, a2):
                    out.append(inter(s, e, a1, a2))
                out.append(e)
            elif inside(s, a1, a2):
                out.append(inter(s, e, a1, a2))
            s = e
    clean = []
    for p in out:
        if not clean or np.linalg.norm(p - clean[-1]) > 1e-9:
            clean.append(p)
    if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) < 1e-9:
        clean.pop()
    return clean


def _ccw(P):
    return list(P) if area(P) > 0 else list(P)[::-1]


def _overlap_area(P, Q):
    """Area of the intersection of two convex polygons."""
    R = _clip(_ccw(P), _ccw(Q))
    return abs(area(R)) if len(R) >= 3 else 0.0


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _u(p - v), _u(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ticks(p, q, n=1, color=WHITE, size=0.12, width=2.5, at=0.5):
    """n short tick marks across the screen segment pq (equal lengths)."""
    p, q = to3(p), to3(q)
    d = _u(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    c = p + (q - p) * at
    g = VGroup()
    for k in range(n):
        o = c + d * 0.09 * (k - (n - 1) / 2)
        g.add(Line(o - nrm * size, o + nrm * size, color=color,
                   stroke_width=width))
    return g


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on
    the side the vector `side` points to, its box clear of the line by
    `gap` whatever the slope."""
    p, q = to3(p), to3(q)
    d = _u(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame. In the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_u(q - p), about_point=p, **kw)


def _glide(mob, pivot, target, angle, **kw):
    """A rigid motion that stays rigid on every frame: the piece turns by
    `angle` about its pivot while the pivot travels straight to `target`."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .shift(a * (target - pivot)))

    return UpdateFromAlphaFunc(mob, upd, **kw)


def _spiral(mob, centre, turn, factor, **kw):
    """Spiral similarity about `centre`: turn by `turn` while scaling by
    `factor` (geometrically in time); every frame is a similar copy."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=c)
                 .scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


# ---- text hygiene

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


def _labels_ok(labels, segs, what, pad=0.05, gap=0.04):
    """Every label inside the safe area, clear of every segment in segs
    and of every other label."""
    for m in labels:
        nm = getattr(m, "text", type(m).__name__)
        check(_inside(m), f"{what}: '{nm}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{nm}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(_apart(labels[i], labels[j], gap),
                  f"{what}: '{getattr(labels[i], 'text', i)}' clear of "
                  f"'{getattr(labels[j], 'text', j)}'")


def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments (for label checks)."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _u(p - v), _u(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


def _poly_segs(pts):
    pts = [to3(p) for p in pts]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _arc_segs(c, r, a0, a1, n=24):
    c = to3(c)
    pts = [c + r * _dir(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _angle_segs(v, p, q, r, n=12):
    """The arc that angle_arc(v, p, q, r) draws, as segments."""
    v = to3(v)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return _arc_segs(v, r, a1, a1 + span, n)


_BASE_CHARS = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789+=")


def _baseline(t):
    """Baseline of a Text: the median bottom of its non-descending glyphs."""
    ys = [g.get_bottom()[1] for ch, g in zip(t.text, t.submobjects)
          if ch in _BASE_CHARS]
    if not ys:
        ys = [g.get_bottom()[1] for g in t.submobjects]
    return float(np.median(ys))


def _supline(parts, size=30, color=YELLOW_B, sup_scale=0.62, rise=0.48):
    """A line of text with true superscripts, without TeX.

    parts: list of (string, is_superscript). Normal pieces share one
    baseline; a superscript sits `rise` x-heights above it, smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for s, sup in parts:
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        body = s.strip(" ")
        x += lead * sp
        t = Text(body, font_size=size * (sup_scale if sup else 1.0),
                 color=color)
        y0 = rise * xh if sup else 0.0
        t.shift(np.array([x - t.get_left()[0], y0 - _baseline(t), 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sup else 0.02)
        g.add(t)
    return g


def _sup_caption(parts, size=30, color=YELLOW_B):
    g = _supline(parts, size, color)
    if g.width > 13.4:
        g.scale_to_fit_width(13.4)
    g.set_x(0.0)
    g.to_edge(DOWN, buff=0.3)
    return g


def _final_check(scene, cap):
    """Closing frame: everything but the caption inside the safe area, the
    caption inside the caption band, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        if not m.has_points() and not m.submobjects:
            continue
        check(_inside(m), f"{type(m).__name__} inside the safe area")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_apart(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


def _rows(lines, x0, y0, dy):
    """Left-align a list of mobjects at x0, the first centred at y0."""
    for i, m in enumerate(lines):
        m.move_to(np.array([x0 + m.width / 2, y0 - i * dy, 0.0]))
    return lines


PHI = (1 + np.sqrt(5)) / 2


# ===================================================================== J16

class J16_PentagonGoldenRatio(Board):
    """In the regular pentagon every side is seen at 36° from each of the
    other vertices (equal chords, equal inscribed angles: half of 72°).
    Two diagonals from the top and the bottom side make the triangle with
    angles 36°, 72°, 72°. The diagonal from a base corner cuts the 72° there
    into 36° + 36° and meets the opposite diagonal at P: the small triangle
    at the base has angles 36°, 72°, 72° too, so it is isosceles with legs
    s, and the triangle above it (36°, 36°) gives the rest of the diagonal
    the length s as well — the base of the small triangle is d − s. The
    small triangle, turned and enlarged by d/s, is the big one:
    d : s = s : (d − s), so d/s = φ."""

    def construct(self):
        R = 2.95
        O = np.array([-3.3, 0.42, 0.0])
        ang = {"A": -126, "B": -54, "C": 18, "D": 90, "E": 162}
        V = {k: O + R * _dir(np.radians(a)) for k, a in ang.items()}
        A, B, C, D, E = (V[k] for k in "ABCDE")
        s = float(np.linalg.norm(B - A))
        d = float(np.linalg.norm(D - A))
        P = _meet(A, C, B, D)

        # ---- the angle facts and lengths, checked
        d36, d72 = np.radians(36), np.radians(72)
        for k in "ABCDE":
            check(abs(np.linalg.norm(V[k] - O) - R) < 1e-9, "on one circle")
        check(abs(_ang(D, A, B) - d36) < 1e-9, "D sees the side AB at 36°")
        check(abs(_ang(A, B, C) - d36) < 1e-9 and abs(_ang(A, C, D) - d36) < 1e-9,
              "A sees BC and CD at 36° each")
        check(abs(_ang(A, B, D) - d72) < 1e-9 and abs(_ang(B, A, D) - d72) < 1e-9,
              "ABD: 72° at the base")
        check(abs(_ang(P, A, B) - d72) < 1e-9, "ABP: 72° at P")
        check(abs(np.linalg.norm(P - A) - s) < 1e-9, "AP = s")
        check(abs(np.linalg.norm(D - P) - s) < 1e-9, "PD = s")
        check(abs(np.linalg.norm(B - P) - (d - s)) < 1e-9, "BP = d − s")
        check(abs(d / s - s / (d - s)) < 1e-9 and abs(d / s - PHI) < 1e-12,
              "d/s = s/(d − s) = φ")
        # the spiral similarity A→D, B→A, P→B: turn −108°, factor d/s
        turn, fac = -np.radians(108), d / s
        Rm = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        M = np.eye(2) - fac * Rm
        rhs = D[:2] - fac * Rm @ A[:2]
        Z = to3(np.linalg.solve(M, rhs))              # its fixed centre

        def sim(X):
            return Z + to3(fac * Rm @ (X - Z)[:2])
        check(close(sim(A), D) and close(sim(B), A) and close(sim(P), B),
              "the small triangle, turned and enlarged, is the big one")

        # ---- the pentagon in its circle
        circ = Circle(radius=R, color=GREY_D, stroke_width=2).move_to(O)
        pent = Polygon(A, B, C, D, E, stroke_color=WHITE, stroke_width=3,
                       fill_color=GREY_E, fill_opacity=0.35)
        ls = tag("s", 28).next_to(Line(A, B), DOWN, buff=0.24)
        self.play(Create(circ), run_time=0.6)
        self.play(Create(pent), FadeIn(ls), run_time=1.0)

        # every side is seen at 36° from the circle: the centre sees 72°
        dotO = Dot(O, radius=0.045, color=GREY_B)
        cA = Line(O, A, color=GREY_B, stroke_width=1.5)
        cB = Line(O, B, color=GREY_B, stroke_width=1.5)
        a72 = angle_arc(O, A, B, radius=0.42, color=GREY_A, width=3)
        l72 = tag("72°", 20, GREY_A).move_to(O + 0.72 * _dir(-PI / 2))
        self.play(FadeIn(dotO), Create(cA), Create(cB), Create(a72),
                  FadeIn(l72), run_time=0.8)

        # two diagonals and a side: the triangle 36°, 72°, 72°
        big = Polygon(A, B, D, stroke_color=WHITE, stroke_width=3,
                      fill_color=BLUE_D, fill_opacity=0.55)
        aD = angle_arc(D, A, B, radius=0.95, color=YELLOW_B, width=4)
        lD = tag("36°", 22).move_to(D + 1.3 * angle_mid_dir(D, A, B))
        ld = _beside(tag("d", 28), D, A, LEFT, 0.14)
        self.play(FadeIn(big), Create(aD), FadeIn(lD), FadeIn(ld), run_time=1.0)
        self.play(FadeOut(cA), FadeOut(cB), FadeOut(a72), FadeOut(l72),
                  FadeOut(dotO), run_time=0.5)
        self.hold(0.3)

        # the diagonal from A: 72° = 36° + 36°; P on the other diagonal
        AC = Line(A, C, color=WHITE, stroke_width=3)
        aA1 = angle_arc(A, B, P, radius=0.62, color=YELLOW_B, width=4)
        aA2 = angle_arc(A, P, D, radius=0.75, color=YELLOW_B, width=4)
        lA1 = tag("36°", 20).move_to(A + 1.12 * angle_mid_dir(A, B, P))
        lA2 = tag("36°", 20).move_to(A + 1.22 * angle_mid_dir(A, P, D))
        dotP = Dot(P, radius=0.06, color=YELLOW_B)
        self.play(Create(AC), run_time=0.7)
        self.play(Create(aA1), Create(aA2), FadeIn(lA1), FadeIn(lA2), FadeIn(dotP),
                  run_time=0.8)

        # ABP: 36°, 72°, 72° — isosceles, AP = AB = s
        small = Polygon(A, B, P, stroke_color=WHITE, stroke_width=3,
                        fill_color=ORANGE, fill_opacity=0.62)
        aB = angle_arc(B, A, P, radius=0.5, color=YELLOW_B, width=4)
        lB = tag("72°", 20).move_to(B + 0.9 * angle_mid_dir(B, A, P))
        aP = angle_arc(P, A, B, radius=0.42, color=YELLOW_B, width=4)
        lP = tag("72°", 20).move_to(P + 0.78 * angle_mid_dir(P, A, B))
        tAB, tAP = _ticks(A, B, 1), _ticks(A, P, 1)
        self.play(FadeIn(small), Create(aB), FadeIn(lB), run_time=0.8)
        self.bring_to_front(aA1, lA1, dotP)
        self.play(Create(aP), FadeIn(lP), run_time=0.6)
        self.play(FadeIn(tAB), FadeIn(tAP), run_time=0.5)
        self.hold(0.3)

        # APD: 36° at A and at D — isosceles, PD = PA = s; so BP = d − s
        mid = Polygon(A, P, D, stroke_color=WHITE, stroke_width=3,
                      fill_color=GREEN_D, fill_opacity=0.6)
        tPD = _ticks(P, D, 1)
        self.play(FadeIn(mid), run_time=0.6)
        self.bring_to_front(aA2, lA2, aD, lD, dotP, tAP)
        self.play(Indicate(lA2, color=YELLOW_B), Indicate(lD, color=YELLOW_B),
                  FadeIn(tPD), run_time=0.9)
        # BP, marked in orange, labelled inside the triangle BPC
        bp = Line(B, P, color=ORANGE, stroke_width=8)
        la, lb_, lc = (np.linalg.norm(P - C), np.linalg.norm(B - C),
                       np.linalg.norm(B - P))
        inc = (la * B + lb_ * P + lc * C) / (la + lb_ + lc)
        lbp = tag("d − s", 22, ORANGE).move_to(inc + 0.12 * _u(B + P - 2 * C))
        self.play(Create(bp), FadeIn(lbp), run_time=0.7)
        self.hold(0.4)

        # the small triangle, turned and enlarged by d/s, is the big one
        segs = (_poly_segs([A, B, C, D, E]) + [(A, C), (B, D), (A, D)]
                + _arc_segs(O, R, 0, TAU, 72)
                + _angle_segs(D, A, B, 0.95) + _angle_segs(A, B, P, 0.62)
                + _angle_segs(A, P, D, 0.75) + _angle_segs(B, A, P, 0.5)
                + _angle_segs(P, A, B, 0.42))
        _labels_ok([ls, ld, lD, lA1, lA2, lB, lP, lbp], segs, "J16 figure")

        self.play(FadeOut(aA1), FadeOut(aA2), FadeOut(lA1), FadeOut(lA2),
                  FadeOut(aB), FadeOut(lB), FadeOut(aP), FadeOut(lP),
                  FadeOut(aD), FadeOut(lD), run_time=0.5)
        ghost = Polygon(A, B, P, stroke_color=YELLOW_B, stroke_width=4,
                        fill_color=ORANGE, fill_opacity=0.35)
        self.add(ghost)
        self.play(small.animate.set_stroke(YELLOW_B, 4), run_time=0.4)
        xf = tag("× d/s", 26, YELLOW_B).move_to(np.array([-0.45, 2.95, 0.0]))
        check(np.linalg.norm(xf.get_corner(DL) - O) > R + 0.2
              and xf.get_left()[0] > D[0] + 0.5, "× d/s clear of the circle")
        self.play(_spiral(ghost, Z, turn, fac), FadeIn(xf), run_time=2.2)
        check(all(close(a, b, 1e-6) for a, b in
                  zip(ghost.get_vertices(), (D, A, B))), "the copy lands on D, A, B")
        self.play(ghost.animate.set_fill(opacity=0.0), run_time=0.5)
        self.bring_to_front(small, tAB, tAP, tPD, bp, dotP)

        r1 = Text("d : s  =  s : (d − s)", font_size=32, color=WHITE,
                  t2c={"(d − s)": ORANGE})
        r2 = tag("x = d/s:   x = 1/(x − 1)", 28, GREY_A)
        r3 = tag("x² = x + 1", 28, GREY_A)
        r4 = tag("x = (1 + √5)/2 = φ ≈ 1.618", 28, YELLOW_B)
        _rows([r1, r2, r3, r4], 0.65, 1.9, 0.78)
        for r in (r1, r2, r3, r4):
            check(_inside(r) and r.get_left()[0] > pent.get_right()[0] + 0.3,
                  "panel text right of the figure, inside the frame")
        check(_apart(xf, r1, 0.1), "× d/s clear of the panel")
        self.play(FadeIn(r1), run_time=0.8)
        self.play(FadeIn(r2), run_time=0.7)
        self.play(FadeIn(r3), run_time=0.6)
        self.play(FadeIn(r4), run_time=0.7)
        cap = caption("diagonal : side  =  side : (diagonal − side)   ⟹   "
                      "d/s = φ = (1 + √5)/2", 30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J19

def _cantor_walk(N):
    """Cells (p, q) of the diagonals p + q = 2 … N + 1, in zigzag order:
    diagonal n runs down-left (p falling) when n is odd, up-right when n
    is even, so consecutive cells are always neighbours."""
    out = []
    for n in range(2, N + 2):
        cells = [(p, n - p) for p in range(n - 1, 0, -1)]      # down-left
        out += cells if n % 2 == 1 else cells[::-1]
    return out


class J19_FractionsCountable(Board):
    """Write p/q in column p, row q. The diagonal p + q = n holds only n − 1
    fractions, so a walk that runs through the diagonals one after another,
    zigzagging, reaches every p/q after finitely many steps. Skipping the
    repeats (2/2 = 1/1, …) numbers the positive fractions 1, 2, 3, …; with
    0 first and each fraction followed by its negative, all of ℚ."""

    def construct(self):
        N = 6
        walk = _cantor_walk(N)
        # the walk: every cell of the 6 diagonals once, neighbours in turn
        check(len(walk) == N * (N + 1) // 2 and len(set(walk)) == len(walk),
              "each cell of the diagonals once")
        check(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
                  for a, b in zip(walk, walk[1:])), "consecutive cells touch")
        reduced = [c for c in walk if gcd(*c) == 1]
        values = [c[0] / c[1] for c in reduced]
        check(len(set(values)) == len(values), "no value numbered twice")
        for cell_ in walk:
            g = gcd(*cell_)
            if g > 1:
                low = (cell_[0] // g, cell_[1] // g)
                check(low in reduced and walk.index(low) < walk.index(cell_),
                      "a repeat comes after its lowest terms")
        # every p/q with p + q ≤ 7 in lowest terms gets a number, and the walk
        # reaches p/q within ½(p+q)(p+q−1) steps
        check(set(reduced) == {(p, q) for p in range(1, 7) for q in range(1, 7)
                               if p + q <= 7 and gcd(p, q) == 1}, "all reached")
        for i, (p, q) in enumerate(walk):
            check(i + 1 <= (p + q) * (p + q - 1) // 2, "reached in time")

        c = 0.86
        X0, Y0 = -6.35, 3.25                 # top-left corner of the grid

        def ctr(p, q):
            return np.array([X0 + (p - 0.5) * c, Y0 - (q - 0.5) * c, 0.0])

        grid = VGroup()
        for i in range(N + 1):
            grid.add(Line([X0 + i * c, Y0, 0], [X0 + i * c, Y0 - N * c, 0],
                          color=GREY_D, stroke_width=1.5))
            grid.add(Line([X0, Y0 - i * c, 0], [X0 + N * c, Y0 - i * c, 0],
                          color=GREY_D, stroke_width=1.5))
        frac = {}
        for p in range(1, N + 1):
            for q in range(1, N + 1):
                frac[(p, q)] = tag(f"{p}/{q}", 24).move_to(ctr(p, q))
        more_r = tag("…", 30, GREY_B).move_to(ctr(N, 1) + RIGHT * 0.72)
        more_d = tag("…", 30, GREY_B).rotate(PI / 2).move_to(ctr(1, N) + DOWN * 0.68)
        self.play(FadeIn(grid), LaggedStart(*[FadeIn(frac[(p, q)])
                                              for q in range(1, N + 1)
                                              for p in range(1, N + 1)],
                                            lag_ratio=0.02),
                  FadeIn(more_r), FadeIn(more_d), run_time=1.6)

        # the table ℕ ↔ fractions, filled as the walk goes
        slots = 7
        tx0, dxs = 0.95, 0.74
        ty1, ty2 = 2.75, 2.05
        hN = tag("ℕ", 28, YELLOW_B).move_to([tx0 - 0.62, ty1, 0])
        hQ = tag("p/q", 24, GREY_A).move_to([tx0 - 0.62, ty2, 0])
        nums = [tag(str(i + 1), 24, YELLOW_B).move_to([tx0 + i * dxs, ty1, 0])
                for i in range(slots)]
        dots_t = VGroup(tag("…", 26, YELLOW_B).move_to([tx0 + slots * dxs, ty1, 0]),
                        tag("…", 26).move_to([tx0 + slots * dxs, ty2, 0]))
        bar = Line([tx0 - 0.95, (ty1 + ty2) / 2, 0],
                   [tx0 + slots * dxs + 0.3, (ty1 + ty2) / 2, 0],
                   color=GREY_C, stroke_width=1.5)
        self.play(FadeIn(hN), FadeIn(hQ), Create(bar), run_time=0.6)

        # diagonal labels p + q = n at the upper end of each diagonal
        dlab = {n: tag(str(n), 20, TEAL_B).move_to(
            np.array([X0 + (n - 1) * c + 0.04, Y0 + 0.2, 0.0])) for n in range(2, N + 2)}
        dhead = tag("p + q", 20, TEAL_B).next_to(dlab[2], LEFT, buff=0.22)

        def arrow(a, b):
            pa, pb = ctr(*a), ctr(*b)
            u = _u(pb - pa)

            def reach(m):
                hw, hh = m.width / 2 + 0.06, m.height / 2 + 0.06
                return min(hw / max(abs(u[0]), 1e-9), hh / max(abs(u[1]), 1e-9))
            return Arrow(pa + u * reach(frac[a]), pb - u * reach(frac[b]), buff=0,
                         color=YELLOW_B, stroke_width=3.5, tip_length=0.13,
                         max_tip_length_to_length_ratio=0.45)

        index_lab = {}
        k = 0
        prev = None
        arrows = []
        for n in range(2, N + 2):
            diag = [cc for cc in walk if sum(cc) == n]
            rt = 0.5 if n <= 4 else 0.3
            for j, cc in enumerate(diag):
                anims = []
                if j == 0:
                    anims.append(FadeIn(dlab[n]))
                    if n == 2:
                        anims.append(FadeIn(dhead))
                if prev is not None:
                    a = arrow(prev, cc)
                    arrows.append(a)
                    anims.append(GrowArrow(a))
                if gcd(*cc) == 1:
                    k += 1
                    # top-middle of the cell (the arrows pass through the
                    # corners); column 1 keeps its free top-left corner, as
                    # the vertical steps there cross the top-middle
                    off = (np.array([-c / 2 + 0.17, c / 2 - 0.14, 0.0]) if cc[0] == 1
                           else np.array([0.0, c / 2 - 0.15, 0.0]))
                    il = tag(str(k), 16, YELLOW_B).move_to(ctr(*cc) + off)
                    index_lab[cc] = il
                    anims += [FadeIn(il), frac[cc].animate.set_color(WHITE)]
                    if k <= slots:
                        tgt = tag(f"{cc[0]}/{cc[1]}", 24).move_to(
                            [tx0 + (k - 1) * dxs, ty2, 0])
                        anims += [FadeIn(nums[k - 1]),
                                  TransformFromCopy(frac[cc], tgt)]
                else:
                    fr = frac[cc]          # struck across the walk's direction
                    st = Line(fr.get_corner(UL) + LEFT * 0.04 + UP * 0.03,
                              fr.get_corner(DR) + RIGHT * 0.04 + DOWN * 0.03,
                              color=RED_B, stroke_width=3)
                    anims += [fr.animate.set_color(GREY_C), Create(st)]
                self.play(*anims, run_time=rt)
                prev = cc
        self.play(FadeIn(dots_t), run_time=0.4)
        self.hold(0.4)

        # all of ℚ: 0 first, then each fraction and its negative
        q_line = tag("ℚ:  0, ±1/1, ±2/1, ±1/2, ±1/3, ±3/1, …", 24)
        q_line.move_to([tx0 - 0.95 + q_line.width / 2, 1.05, 0])
        check(_inside(q_line), "the ℚ line fits")
        self.play(FadeIn(q_line), run_time=0.8)

        # hygiene: labels clear of each other, arrows clear of the labels
        segs = [(a.get_start(), a.get_end()) for a in arrows]
        for f in frac.values():
            check(_hit(f, segs, 0.02) is None, "the arrows stay off the fractions")
        for t in index_lab.values():
            check(_hit(t, segs, 0.06) is None, "the arrows stay off the numbers")
            for f in frac.values():
                check(_apart(t, f, 0.04), "index numbers clear of the fractions")
        for t in dlab.values():
            check(_hit(t, segs, 0.06) is None and _apart(t, dhead, 0.05),
                  "diagonal labels clear")
        cap = caption("every p/q is reached on its diagonal p + q   ⟹   "
                      "|ℚ| = |ℕ|", 30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J13

def _shift(s, r):
    """The string s turned r places: bead j takes the colour of bead j − r."""
    n = len(s)
    return tuple(s[(j - r) % n] for j in range(n))


class J13_FermatNecklaces(Board):
    """Fermat's little theorem, a = 2 colours and p = 5 beads. Of the 2⁵ = 32
    strings, 2 have one colour. Close every other string into a necklace:
    turning it gives 5 strings, and they are all different — if a turn by k
    places (0 < k < 5) changed nothing, bead 0 would match beads k, 2k, 3k,
    4k, which for prime 5 are all the beads: one colour. So the 30 strings
    fall into rings of exactly 5: 5 | 2⁵ − 2. The same holds for any number
    of colours a and any prime p. With 6 beads a turn by 2 visits only
    every other bead, and 2⁶ − 2 = 62 is not a multiple of 6."""

    def construct(self):
        p = 5
        strings = [tuple((i >> (p - 1 - j)) & 1 for j in range(p)) for i in range(2 ** p)]
        mono = [s for s in strings if len(set(s)) == 1]
        rest = [s for s in strings if len(set(s)) > 1]
        rings = []
        seen = set()
        for s in rest:
            if s not in seen:
                ring = [_shift(s, r) for r in range(p)]
                rings.append(ring)
                seen.update(ring)
        check(len(mono) == 2 and len(rest) == 30, "32 = 2 + 30")
        check(all(len(set(r)) == p for r in rings), "each ring holds 5 different strings")
        check(len(rings) == 6 and sorted(x for r in rings for x in r) == sorted(rest),
              "6 rings of 5 use every string once")
        for s in rest:
            for k in range(1, p):
                check(_shift(s, k) != s, "no turn fixes a two-coloured string")
        orbit = [(2 * j) % p for j in range(p)]
        check(sorted(orbit) == list(range(p)), "steps of 2 visit all 5 beads")
        orbit6 = sorted({(2 * j) % 6 for j in range(6)})
        check(orbit6 == [0, 2, 4] and (2 ** 6 - 2) % 6 == 2, "6 beads: steps of 2 miss half")
        for pp in (3, 5, 7, 11, 13):
            for a in (2, 3, 4, 5):
                check((a ** pp - a) % pp == 0, "p | a^p − a")

        C = {0: BLUE_C, 1: ORANGE}
        br, bs = 0.1, 0.26

        def string_mob(s, centre):
            centre = to3(centre)
            x0 = centre[0] - bs * (p - 1) / 2
            g = VGroup(Line([x0 - 0.12, centre[1], 0], [x0 + bs * (p - 1) + 0.12, centre[1], 0],
                            color=GREY_C, stroke_width=2))
            for j, b in enumerate(s):
                g.add(Circle(radius=br, fill_color=C[b], fill_opacity=1.0,
                             stroke_color=WHITE, stroke_width=1).move_to([x0 + j * bs, centre[1], 0]))
            return g

        def necklace(s, centre, R, bead=br):
            centre = to3(centre)
            n = len(s)
            g = VGroup(Circle(radius=R, color=GREY_C, stroke_width=2).move_to(centre))
            for j, b in enumerate(s):
                g.add(Circle(radius=bead, fill_color=C[b], fill_opacity=1.0, stroke_color=WHITE,
                             stroke_width=1.5).move_to(centre + R * _dir(PI / 2 - TAU * j / n)))
            return g

        # ---- all 32 strings
        grid_pos = {s: np.array([-5.25 + 1.5 * (i % 8), 3.2 - 0.45 * (i // 8), 0.0])
                    for i, s in enumerate(strings)}
        mobs = {s: string_mob(s, grid_pos[s]) for s in strings}
        TX = 2.0                                   # left edge of the counting text
        cnt = tag("2⁵ = 32", 28, YELLOW_B)
        cnt.move_to([TX + cnt.width / 2, -0.95, 0])
        self.play(LaggedStart(*[FadeIn(mobs[s]) for s in strings], lag_ratio=0.04),
                  FadeIn(cnt), run_time=1.8)
        self.hold(0.3)

        # ---- the two one-colour strings step aside
        mono_pos = [np.array([-5.75, -0.75, 0.0]), np.array([-5.75, -1.15, 0.0])]
        minus2 = tag("− 2", 28, GREY_A).move_to([-5.75, -1.65, 0])
        cnt2 = tag("32 − 2 = 30", 28, YELLOW_B)
        cnt2.move_to([TX + cnt2.width / 2, -0.95, 0])
        self.play(*[mobs[s].animate.move_to(q) for s, q in zip(mono, mono_pos)],
                  FadeIn(minus2), Transform(cnt, cnt2), run_time=1.2)

        # ---- the other 30 sort into the turns of 6 necklaces
        colx = [-4.75 + 1.9 * j for j in range(6)]
        rowy = [2.1 - 0.4 * r for r in range(p)]
        self.play(*[mobs[s].animate.move_to([colx[j], rowy[r], 0])
                    for j, ring in enumerate(rings) for r, s in enumerate(ring)],
                  run_time=2.2)
        for j, ring in enumerate(rings):
            for r, s in enumerate(ring):
                check(close(mobs[s].get_center(), [colx[j], rowy[r], 0], 1e-6),
                      "each string lands in its ring's column")
        necks = [necklace(ring[0], [colx[j], 3.05, 0], 0.38) for j, ring in enumerate(rings)]
        self.play(*[FadeIn(nk) for nk in necks], run_time=0.9)

        # turning a necklace one bead clockwise gives the next string below
        frames = [SurroundingRectangle(mobs[ring[0]], color=YELLOW_B, buff=0.06,
                                       stroke_width=3) for ring in rings]
        self.play(*[Create(f) for f in frames], run_time=0.5)
        for r in range(1, p + 1):
            anims = [Rotate(nk, angle=-TAU / p, about_point=nk[0].get_center())
                     for nk in necks]
            anims += [f.animate.move_to(mobs[ring[r % p]].get_center())
                      for f, ring in zip(frames, rings)]
            self.play(*anims, run_time=0.6)
            for nk, ring in zip(necks, rings):
                # the bead now on top, read clockwise, is the string ring[r]
                cen = nk[0].get_center()
                for jj in range(p):
                    pos = cen + 0.38 * _dir(PI / 2 - TAU * jj / p)
                    bead = min(nk[1:], key=lambda b: np.linalg.norm(b.get_center() - pos))
                    check(np.linalg.norm(bead.get_center() - pos) < 1e-6, "beads on their places")
                    want = C[ring[r % p][jj]]
                    check(bead.get_fill_color().to_hex().lower() == ManimColor(want).to_hex().lower(),
                          "turning the necklace reads the next string")
        self.play(*[FadeOut(f) for f in frames], run_time=0.4)
        cnt3 = tag("32 − 2 = 30 = 6 × 5", 28, YELLOW_B)
        cnt3.move_to([TX + cnt3.width / 2, -0.95, 0])
        self.play(Transform(cnt, cnt3), run_time=0.7)

        # ---- why the 5 turns differ: a turn by 2 that changed nothing
        S0 = (1, 0, 0, 1, 0)
        cS = np.array([-3.3, -1.6, 0.0])
        RS = 0.8
        big = necklace(S0, cS, RS, 0.15)
        lk = tag("turn by 2 = same?", 22, GREY_A).move_to(cS + UP * 1.2)
        self.play(FadeIn(big), FadeIn(lk), run_time=0.7)
        pts = [cS + RS * _dir(PI / 2 - TAU * j / p) for j in range(p)]
        star = VGroup()
        for i in range(p):
            a, b = orbit[i], orbit[(i + 1) % p]
            u = _u(pts[b] - pts[a])
            ch = Line(pts[a] + u * 0.16, pts[b] - u * 0.16, color=YELLOW_B, stroke_width=3)
            star.add(ch)
            self.play(Create(ch), big[1 + b].animate.set_fill(C[S0[0]]), run_time=0.42)
        one = tag("⟹ one colour", 22, YELLOW_B).move_to(cS + DOWN * 1.1)
        self.play(FadeIn(one), run_time=0.5)

        # ---- 6 beads: a turn by 2 visits every other bead only
        S6 = (0, 1, 0, 1, 0, 1)
        c6 = np.array([0.2, -1.6, 0.0])
        hexn = necklace(S6, c6, RS, 0.15)
        pts6 = [c6 + RS * _dir(PI / 2 - TAU * j / 6) for j in range(6)]
        tri = VGroup(*[Line(pts6[a] + _u(pts6[b] - pts6[a]) * 0.16,
                            pts6[b] - _u(pts6[b] - pts6[a]) * 0.16,
                            color=GREY_A, stroke_width=3)
                       for a, b in ((0, 2), (2, 4), (4, 0))])
        l6 = tag("6 beads", 22, GREY_A).move_to(c6 + UP * 1.2)
        l6b = tag("turn by 2 = same", 22, GREY_A).move_to(c6 + DOWN * 1.2)
        self.play(FadeIn(hexn), FadeIn(l6), run_time=0.6)
        self.play(Create(tri), run_time=0.8)
        self.play(Rotate(hexn, angle=-2 * TAU / 6, about_point=c6), run_time=0.8)
        self.play(FadeIn(l6b), run_time=0.5)
        for jj in range(6):
            pos = pts6[jj]
            bead = min(hexn[1:], key=lambda b: np.linalg.norm(b.get_center() - pos))
            check(bead.get_fill_color().to_hex().lower() == ManimColor(C[S6[jj]]).to_hex().lower(),
                  "the two-coloured 6-necklace is unchanged by a turn of 2")
        r6 = tag("2⁶ − 2 = 62 = 6 × 10 + 2", 24, GREY_A)
        r6.move_to([TX + r6.width / 2, -1.75, 0])
        self.play(FadeIn(r6), run_time=0.6)

        cap = _sup_caption([("p prime:   a", False), ("p", True),
                            (" − a strings fall into rings of p   ⟹   p | a", False),
                            ("p", True), (" − a", False)], 30)
        self.play(FadeIn(cap), run_time=1.0)
        texts = [cnt, minus2, lk, one, l6, l6b, r6]
        for i in range(len(texts)):
            check(_inside(texts[i]), "J13 text inside the frame")
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.08), "J13 texts apart")
        for t in (lk, one):
            check(_apart(t, big, 0.05), "labels clear of the big necklace")
        check(_apart(l6, hexn, 0.05) and _apart(l6b, hexn, 0.05) and _apart(r6, hexn, 0.05),
              "labels clear of the hexagon")
        check(_apart(lk, l6, 0.6) and _apart(one, l6b, 0.6), "the two necklaces' labels well apart")
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J18

class J18_StaircaseParadox(Board):
    """A caution. The staircase of n steps under the diagonal of the unit
    square has length 2 for every n: its treads slide down onto the bottom
    side, its risers across onto the right side. Doubling n folds every
    outer corner across to the inner side — a rigid fold, so the length
    cannot change — and the staircase closes in on the diagonal (no point
    farther than 1/(n√2)). The false conclusion would be 2 = √2. Where it
    fails: magnify any stretch and every tooth is the same right isosceles
    triangle, the staircase running at 45° to the diagonal, so it is always
    √2 times as long as the piece of diagonal it follows. Length is not
    carried to the limit of the shapes."""

    def construct(self):
        side = 5.2
        O = np.array([-6.3, -2.45, 0.0])

        def S(x, y):
            return O + side * np.array([x, y, 0.0])

        def treads(n):
            return [((k / n, k / n), ((k + 1) / n, k / n)) for k in range(n)]

        def risers(n):
            return [(((k + 1) / n, k / n), ((k + 1) / n, (k + 1) / n)) for k in range(n)]

        def seg_len(sg):
            return float(np.hypot(sg[1][0] - sg[0][0], sg[1][1] - sg[0][1]))

        for n in (1, 2, 4, 8, 16, 32, 1024):
            check(abs(sum(seg_len(s_) for s_ in treads(n) + risers(n)) - 2) < 1e-12,
                  "every staircase has length 2")
            corners = [((k + 1) / n, k / n) for k in range(n)]
            gap = max(abs(x - y) / np.sqrt(2) for x, y in corners)
            check(abs(gap - 1 / (n * np.sqrt(2))) < 1e-12, "farthest point 1/(n√2) off")
            # the treads cover the bottom side exactly once, the risers the right side
            check(close(sorted(t[0][0] for t in treads(n)), [k / n for k in range(n)])
                  and close(sorted(r[0][1] for r in risers(n)), [k / n for k in range(n)]),
                  "treads tile the bottom side, risers the right side")

        def stair(n, color=ORANGE, width=None):
            w = width or (5 if n <= 4 else (4 if n <= 8 else 3))
            g = VGroup()
            for a, b in treads(n):
                g.add(Line(S(*a), S(*b), color=color, stroke_width=w))
            for a, b in risers(n):
                g.add(Line(S(*a), S(*b), color=color, stroke_width=w))
            return g

        sq = Polygon(S(0, 0), S(1, 0), S(1, 1), S(0, 1), stroke_color=GREY_B,
                     stroke_width=2)
        diag = DashedLine(S(0, 0), S(1, 1), color=WHITE, stroke_width=3,
                          dash_length=0.1)
        l_r2 = _beside(tag("√2", 30), S(0, 0), S(1, 1), np.array([-1, 1, 0]), 0.14, 0.3)
        l_b = tag("1", 28, GREY_A).next_to(Line(S(0, 0), S(1, 0)), DOWN, buff=0.16)
        l_s = tag("1", 28, GREY_A).next_to(Line(S(1, 0), S(1, 1)), RIGHT, buff=0.16)
        self.play(Create(sq), FadeIn(l_b), FadeIn(l_s), run_time=0.9)
        self.play(Create(diag), FadeIn(l_r2), run_time=0.8)

        # panel
        PX = 0.3

        def at_left(m, y):
            return m.move_to([PX + m.width / 2, y, 0])

        n_lab = at_left(tag("n = 1", 32, WHITE), 3.15)
        len_lab = at_left(tag("length = 2", 32, ORANGE), 2.5)
        cur = stair(1)
        self.play(Create(cur), FadeIn(n_lab), run_time=1.0)
        self.play(FadeIn(len_lab), run_time=0.5)

        # doubling n: fold every outer corner across the small square's diagonal
        n = 1
        for rt in (1.1, 1.0, 0.8, 0.7):
            keep, Ls, axes = VGroup(), [], []
            w = 5 if n <= 2 else (4 if n <= 4 else 3)
            for k in range(n):
                x0, x1, y0 = k / n, (k + 1) / n, k / n
                h = 0.5 / n
                keep.add(Line(S(x0, y0), S(x0 + h, y0), color=ORANGE, stroke_width=w))
                keep.add(Line(S(x1, y0 + h), S(x1, y0 + 2 * h), color=ORANGE, stroke_width=w))
                L = VMobject(stroke_color=ORANGE, stroke_width=w)
                L.set_points_as_corners([S(x0 + h, y0), S(x1, y0), S(x1, y0 + h)])
                Ls.append(L)
                axes.append((S(x0 + h, y0), S(x1, y0 + h)))
            self.add(keep, *Ls)
            self.remove(cur)
            n_new = at_left(tag(f"n = {2 * n}", 32, WHITE), 3.15)
            self.play(*[Rotate(L, angle=PI, axis=_u(b - a), about_point=a)
                        for L, (a, b) in zip(Ls, axes)],
                      Transform(n_lab, n_new), run_time=rt)
            for L, (a, b), k in zip(Ls, axes, range(n)):
                inner = S((k + 0.5) / n, (k + 0.5) / n)
                check(close(L.get_points()[0], a, 1e-6) or close(L.get_points()[0], b, 1e-6),
                      "the fold keeps the corner's ends")
                check(min(np.linalg.norm(q - inner) for q in L.get_points()) < 1e-6,
                      "the outer corner lands on the diagonal")
            n *= 2
            cur = stair(n)
            self.add(cur)
            self.remove(keep, *Ls)
        check(n == 16, "four doublings: n = 16")

        # the length is 2: treads slide down, risers slide across
        tr = VGroup(*[Line(S(*a), S(*b), color=ORANGE, stroke_width=5) for a, b in treads(n)])
        rs = VGroup(*[Line(S(*a), S(*b), color=TEAL_B, stroke_width=5) for a, b in risers(n)])
        self.play(*[m.animate.set_color(TEAL_B) for m in cur[n:]], run_time=0.4)
        self.add(tr, rs)
        self.play(*[m.animate.shift(DOWN * side * k / n) for k, m in enumerate(tr)],
                  *[m.animate.shift(RIGHT * side * (1 - (k + 1) / n)) for k, m in enumerate(rs)],
                  run_time=1.6)
        for k, m in enumerate(tr):
            check(close(m.get_start(), S(k / n, 0), 1e-6), "treads land on the bottom side")
        for k, m in enumerate(rs):
            check(close(m.get_start(), S(1, k / n), 1e-6), "risers land on the right side")
        self.play(Indicate(l_b, color=ORANGE), Indicate(l_s, color=TEAL_B), run_time=0.8)
        self.play(FadeOut(tr), FadeOut(rs), cur.animate.set_color(ORANGE), run_time=0.6)

        gap_lab = at_left(tag("gap ≤ 1/(n√2) → 0", 28), 1.85)
        false_lab = at_left(Text("⟹  2 = √2 ?", font_size=32, color=RED_B), 1.2)
        self.play(FadeIn(gap_lab), run_time=0.6)
        self.play(FadeIn(false_lab), run_time=0.7)
        self.hold(0.5)

        # ---- where it fails: magnify two teeth around the middle
        a0, a1 = 7 / 16, 9 / 16
        mag = 4.0
        ic = np.array([4.35, -0.45, 0.0])
        wc = S((a0 + a1) / 2, (a0 + a1) / 2)

        def M(x, y):
            return ic + mag * (S(x, y) - wc)

        lens = Square(side_length=side * (a1 - a0), color=YELLOW_B, stroke_width=2.5).move_to(wc)
        frame = Square(side_length=mag * side * (a1 - a0), color=YELLOW_B,
                       stroke_width=2.5).move_to(ic)
        x4 = tag("× 4", 24, YELLOW_B).next_to(frame, UP, buff=0.1).align_to(frame, RIGHT)
        check(_inside(frame) and frame.get_left()[0] > PX + 0.1, "inset inside the frame")
        teeth = VGroup()
        for (ta, tb) in treads(n)[7:9]:
            teeth.add(Line(M(*ta), M(*tb), color=ORANGE, stroke_width=6))
        for (ra, rb) in risers(n)[7:9]:
            teeth.add(Line(M(*ra), M(*rb), color=ORANGE, stroke_width=6))
        dg = DashedLine(M(a0, a0), M(a1, a1), color=WHITE, stroke_width=3, dash_length=0.12)
        self.play(Create(lens), Create(frame), FadeIn(x4), run_time=0.8)
        self.play(Create(teeth), Create(dg), run_time=0.9)
        # label the upper tooth: its tread, its riser, its piece of diagonal
        t2 = Line(M(8 / 16, 8 / 16), M(9 / 16, 8 / 16))
        r2 = Line(M(9 / 16, 8 / 16), M(9 / 16, 9 / 16))
        lt = tag("1/n", 24, ORANGE).next_to(t2, DOWN, buff=0.14)
        lr = tag("1/n", 24, ORANGE).next_to(r2, RIGHT, buff=0.12)
        ld = _beside(tag("√2/n", 24), M(a0, a0), M(a1, a1), np.array([-1, 1, 0]), 0.12, 0.75)
        a45 = angle_arc(M(8 / 16, 8 / 16), M(9 / 16, 8 / 16), M(9 / 16, 9 / 16),
                        radius=0.55, color=YELLOW_B, width=4)
        l45 = tag("45°", 20, YELLOW_B).move_to(M(8 / 16, 8 / 16) + 1.0 * _dir(PI / 8))
        self.play(FadeIn(lt), FadeIn(lr), FadeIn(ld), Create(a45), FadeIn(l45), run_time=0.9)
        tooth = at_left(tag("1/n + 1/n = √2 · (√2/n)", 26), -2.2)
        always = at_left(tag("every n:  length = √2 · √2 = 2", 26, YELLOW_B), -2.75)
        self.play(FadeIn(tooth), run_time=0.8)
        self.play(FadeIn(always), run_time=0.8)

        segs = (_poly_segs([S(0, 0), S(1, 0), S(1, 1), S(0, 1)]) + [(S(0, 0), S(1, 1))]
                + [(m.get_start(), m.get_end()) for m in teeth]
                + [(M(a0, a0), M(a1, a1))] + _poly_segs(frame.get_vertices())
                + _poly_segs(lens.get_vertices())
                + _angle_segs(M(8 / 16, 8 / 16), M(9 / 16, 8 / 16), M(9 / 16, 9 / 16), 0.55))
        _labels_ok([l_r2, l_b, l_s, lt, lr, ld, l45, x4], segs, "J18 labels")
        cap = caption("the staircases tend to the diagonal, their lengths stay 2 ≠ √2", 30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J14

def _wm_solutions(p):
    """All (x, y, z) in positive integers with x² + 4yz = p."""
    out = []
    for x in range(1, int(np.sqrt(p)) + 1):
        r = p - x * x
        if r > 0 and r % 4 == 0:
            r //= 4
            out += [(x, y, r // y) for y in range(1, r + 1) if r % y == 0]
    return out


def _zeta(t):
    """Zagier's involution: the other windmill with the same outline."""
    x, y, z = t
    if x < y - z:
        return (x + 2 * z, z, y - x - z)
    if x < 2 * y:
        return (2 * y - x, y, x - y + z)
    return (x - 2 * y, x - y + z, y)


def _wm_rot(c, x):
    """Cell (i, j) turned a quarter about the centre (x/2, x/2) of [0, x]²."""
    return (x - 1 - c[1], c[0])


def _wm_cells(t):
    """The windmill (x, y, z): square cells [0, x)², and four y × z arms,
    the bottom one [0, y) × [−z, 0), the others turned by quarter turns.
    Returns (square cells, [arm cells × 4]) in cell coordinates whose centre
    of symmetry is (x/2, x/2)."""
    x, y, z = t
    sq = {(i, j) for i in range(x) for j in range(x)}
    arm = {(i, j) for i in range(y) for j in range(-z, 0)}
    arms = []
    for _ in range(4):
        arms.append(arm)
        arm = {_wm_rot(c, x) for c in arm}
    return sq, arms


def _wm_centred(t):
    """All cells of the windmill, centred: doubled coordinates of cell centres."""
    sq, arms = _wm_cells(t)
    x = t[0]
    return frozenset((2 * i + 1 - x, 2 * j + 1 - x) for i, j in sq.union(*arms))


def _cell_outline(cells):
    """Boundary of a union of unit cells (i, j) = [i, i+1] × [j, j+1], as the
    list of its corners (counter-clockwise, collinear points dropped)."""
    edges = set()
    for i, j in cells:
        for a, b in (((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)),
                     ((i + 1, j + 1), (i, j + 1)), ((i, j + 1), (i, j))):
            if (b, a) in edges:
                edges.remove((b, a))
            else:
                edges.add((a, b))
    nxt = dict(edges)
    check(len(nxt) == len(edges), "a simple outline")
    start = min(nxt)
    poly, cur = [start], nxt[start]
    while cur != start:
        poly.append(cur)
        cur = nxt[cur]
    check(len(poly) == len(edges), "one closed outline")
    out = []
    for k in range(len(poly)):
        p0, p1, p2 = poly[k - 1], poly[k], poly[(k + 1) % len(poly)]
        if (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]) != 0:
            out.append(p1)
    return out


class J14_ZagierWindmills(Board):
    """p = 13. Each solution of x² + 4yz = 13 is a windmill: an x-square
    with four y × z arms, turned a quarter each. Keeping the outline,
    rebuild the windmill around the other square that fits it: (1, 3, 1)
    and (3, 1, 1) trade places, and the cross (1, 1, 3) has only one
    square — so the windmills come in pairs plus one: an odd number. Now
    flip every arm over the diagonal through its corner (y ↔ z): (1, 1, 3)
    and (1, 3, 1) trade places, so the odd one out must be unchanged, its
    arms squares: (3, 1, 1), and 13 = 3² + 4·1² = 3² + 2²."""

    def construct(self):
        p = 13
        sols = _wm_solutions(p)
        check(sorted(sols) == [(1, 1, 3), (1, 3, 1), (3, 1, 1)], "three windmills for 13")
        for q in range(5, 400, 4):
            if all(q % d for d in range(2, int(q ** 0.5) + 1)):
                S_ = _wm_solutions(q)
                fz = [t for t in S_ if _zeta(t) == t]
                check(all(_zeta(t) in S_ and _zeta(_zeta(t)) == t for t in S_)
                      and fz == [(1, 1, (q - 1) // 4)], "ζ: an involution, one fixed point")
                check(len(S_) % 2 == 1 and sum(1 for t in S_ if t[1] == t[2]) == 1,
                      "odd count, exactly one windmill with y = z")
        for t in sols:
            check(len(_wm_centred(t)) == p, "13 cells")
        W1, W2, W3 = (1, 1, 3), (1, 3, 1), (3, 1, 1)
        check(_zeta(W2) == W3 and _zeta(W1) == W1, "ζ pairs (1,3,1) with (3,1,1)")
        check(_wm_centred(W2) == _wm_centred(W3), "the pair shares one outline")
        check(_wm_centred(W1) != _wm_centred(W2), "the cross has its own outline")

        u = 0.4
        BLU, ORG = BLUE_D, ORANGE
        ctr = {0: np.array([-4.25, 0.9, 0.0]), 1: np.array([-0.2, 0.9, 0.0]),
               2: np.array([3.6, 0.9, 0.0])}

        def P(t, c, i, j):
            x = t[0]
            return ctr[c] + u * np.array([i - x / 2, j - x / 2, 0.0])

        def cell_sq(t, c, cell, col):
            i, j = cell
            return mk([P(t, c, i, j), P(t, c, i + 1, j), P(t, c, i + 1, j + 1),
                       P(t, c, i, j + 1)], col, 0.85, stroke_width=1, stroke_color=GREY_B)

        def rect_outline(t, c, cells, width=3, color=WHITE):
            pts = _cell_outline(cells)
            return Polygon(*[P(t, c, *q) for q in pts], stroke_color=color,
                           stroke_width=width, fill_opacity=0)

        def windmill(t, c):
            sq, arms = _wm_cells(t)
            gs = VGroup(*[cell_sq(t, c, q, BLU) for q in sorted(sq)],
                        rect_outline(t, c, sq))
            ga = [VGroup(*[cell_sq(t, c, q, ORG) for q in sorted(a)],
                         rect_outline(t, c, a)) for a in arms]
            return VGroup(gs, *ga)

        hdr = Text("x² + 4yz = 13", font_size=34, t2c={"x²": BLUE_B, "4yz": ORANGE})
        hdr.move_to([0, 3.3, 0])
        wm = {0: windmill(W1, 0), 1: windmill(W2, 1), 2: windmill(W3, 2)}

        def trip(t, c):
            return tag(f"({t[0]}, {t[1]}, {t[2]})", 26).move_to(ctr[c] + DOWN * 1.85)
        tl = {0: trip(W1, 0), 1: trip(W2, 1), 2: trip(W3, 2)}
        xyz = tag("(x, y, z)", 22, GREY_A).move_to([-6.0, ctr[0][1] - 1.85, 0])
        self.play(FadeIn(hdr), run_time=0.7)
        self.play(*[FadeIn(wm[c]) for c in wm], *[FadeIn(tl[c]) for c in tl],
                  FadeIn(xyz), run_time=1.4)
        for c, t in ((0, W1), (1, W2), (2, W3)):
            check(close(wm[c].get_center(), ctr[c], 1e-6), "windmills centred")
        self.hold(0.5)

        # ---- ζ: keep the outline, rebuild around the other square
        outl = {c: Polygon(*[P(t, c, *q) for q in _cell_outline(
            set().union(*_wm_cells(t)[1], _wm_cells(t)[0]))],
            stroke_color=YELLOW_B, stroke_width=6, fill_opacity=0)
            for c, t in ((0, W1), (1, W2), (2, W3))}
        self.play(*[Create(outl[c]) for c in outl], run_time=1.0)
        # the other square in each outline (dashed)
        def dsq(t_new, c):
            x = t_new[0]
            t_old = (W1, W2, W3)[c]
            o = (t_old[0] - x) / 2
            pts = [P(t_old, c, o, o), P(t_old, c, o + x, o), P(t_old, c, o + x, o + x),
                   P(t_old, c, o, o + x)]
            return DashedVMobject(Polygon(*pts, stroke_color=WHITE, stroke_width=4),
                                  num_dashes=8 * x + 4)
        d1, d2 = dsq(W3, 1), dsq(W2, 2)
        self.play(Create(d1), Create(d2), run_time=0.8)
        # recolour: (1,3,1) → (3,1,1) at place 1, (3,1,1) → (1,3,1) at place 2
        new1, new2 = windmill(W3, 1), windmill(W2, 2)
        # the same cells, re-coloured into the other decomposition
        check(_wm_centred(W3) == _wm_centred(W2), "rebuilt windmill fills the same outline")
        self.play(FadeOut(wm[1]), FadeIn(new1), FadeOut(wm[2]), FadeIn(new2),
                  FadeOut(d1), FadeOut(d2),
                  Transform(tl[1], trip(W3, 1)), Transform(tl[2], trip(W2, 2)),
                  run_time=1.4)
        self.bring_to_front(outl[1], outl[2])
        wm[1], wm[2] = new1, new2
        # the cross: the only square that fits its outline is its own
        d0 = dsq(W1, 0)
        self.play(Create(d0), run_time=0.6)
        self.play(Indicate(outl[0], color=WHITE, scale_factor=1.04), FadeOut(d0), run_time=0.8)
        rowA = tag("same outline, other square:  (1, 3, 1) ↔ (3, 1, 1),  (1, 1, 3) alone"
                   "  ⟹  odd", 22)
        rowA.move_to([0, -1.55, 0])
        self.play(FadeIn(rowA), run_time=0.8)
        self.hold(0.5)
        self.play(*[FadeOut(outl[c]) for c in outl], run_time=0.5)

        # ---- σ: flip every arm over the diagonal through its corner (y ↔ z)
        def fold_arms(t, c, mob):
            x = t[0]
            corner, d = (0.0, 0.0), (1.0, -1.0)
            anims = []
            for k in range(4):
                a = P(t, c, *corner)
                b = P(t, c, corner[0] + d[0], corner[1] + d[1])
                anims.append(Rotate(mob[1 + k], angle=PI, axis=_u(b - a), about_point=a))
                # next arm: corner and direction turned a quarter about (x/2, x/2)
                corner = (x - corner[1], corner[0])
                d = (-d[1], d[0])
            return anims
        targets = {0: (W1, (1, 3, 1)), 1: (W3, W3), 2: (W2, (1, 1, 3))}
        anims = []
        for c, (t, _) in targets.items():
            anims += fold_arms(t, c, wm[c])
        self.play(*anims, Transform(tl[0], trip((1, 3, 1), 0)),
                  Transform(tl[2], trip((1, 1, 3), 2)), run_time=1.8)
        for c, (t, t_new) in targets.items():
            got = set()
            for arm in wm[c][1:]:
                for sqm in arm[:-1]:
                    v = sqm.get_center()
                    got.add(tuple(np.round((v - ctr[c])[:2] / u * 2).astype(int)))
            _, arms_new = _wm_cells(t_new)
            want = {(2 * i + 1 - t_new[0], 2 * j + 1 - t_new[0])
                    for a in arms_new for (i, j) in a}
            check(got == want, f"the flipped arms make the windmill {t_new}")
        rowB = tag("flip the arms, y ↔ z:  (1, 1, 3) ↔ (1, 3, 1),  (3, 1, 1) alone"
                   "  ⟹  y = z", 22)
        rowB.move_to([0, -2.15, 0])
        self.play(FadeIn(rowB), Indicate(wm[1], color=YELLOW_B, scale_factor=1.06),
                  run_time=1.0)

        # ---- the fixed windmill: four 1 × 1 arms make a 2 × 2 square
        t = W3
        blk0 = P(t, 1, 3, 0) + RIGHT * 0.45
        dests = [blk0 + u * np.array([i, j, 0.0]) for i, j in ((0, 0), (1, 0), (0, 1), (1, 1))]
        arms_m = [wm[1][1 + k] for k in range(4)]
        self.play(*[a.animate.shift(dst + u * np.array([0.5, 0.5, 0]) - a[0].get_center())
                    for a, dst in zip(arms_m, dests)], run_time=1.3)
        blk = VGroup(*arms_m)
        check(abs(blk.width - 2 * u) < 1e-6 and abs(blk.height - 2 * u) < 1e-6,
              "the four arms close up into a 2 × 2 square")
        check(blk.get_left()[0] > wm[1][0].get_right()[0] + 0.2
              and blk.get_right()[0] < wm[2].get_left()[0] - 0.2, "the block sits between")
        l3 = tag("3²", 26).next_to(wm[1][0], UP, buff=0.12)
        l2 = tag("2²", 26).next_to(blk, UP, buff=0.12)
        eq = Text("13 = 3² + 2²", font_size=34, t2c={"3²": BLUE_B, "2²": ORANGE})
        eq.move_to([0, 3.3, 0])
        self.play(FadeIn(l3), FadeIn(l2), Transform(hdr, eq), run_time=1.0)

        texts = [hdr, xyz, rowA, rowB, l3, l2] + list(tl.values())
        for i in range(len(texts)):
            check(_inside(texts[i]), "J14 text inside")
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.06), "J14 texts apart")
        for c in wm:
            check(_apart(tl[c], wm[c], 0.08) or c == 1, "triples below their windmills")
        cap = caption("p = 4k + 1 prime  ⟹  one windmill has y = z  ⟹  "
                      "p = x² + (2y)²", 28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J11

def _tri_level(V, ratio):
    """One level of the √3 descent in the arena triangle V (3 screen points,
    side a) with corner triangles of side b = ratio·a. Returns the corner
    triangles, the three overlaps (each with the index of the central
    triangle's vertex it touches) and the central triangle."""
    V = [to3(v) for v in V]
    t = 1 - ratio
    corners, overl, centre = [], [], [None, None, None]
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        corners.append([V[i], V[i] + ratio * (V[j] - V[i]), V[i] + ratio * (V[k] - V[i])])
    for k in range(3):                       # overlap on the side opposite V[k]
        i, j = (k + 1) % 3, (k + 2) % 3
        apex = t * (V[i] + V[j]) + (1 - 2 * t) * V[k]
        centre[k] = apex
        overl.append(([V[i] + ratio * (V[j] - V[i]), V[j] + ratio * (V[i] - V[j]), apex], k))
    return corners, overl, centre


def _dim2(p, q, label, off, color=GREY_A, size=24, gap=0.2):
    """Dimension line beside the screen segment pq, shifted by the vector
    `off`, with end ticks and the label beyond it."""
    p, q, off = to3(p), to3(q), to3(off)
    n = off / np.linalg.norm(off)
    a, b = p + off, q + off
    tick = 0.09 * n
    g = VGroup(Line(a, b, color=color, stroke_width=2),
               Line(a - tick, a + tick, color=color, stroke_width=2),
               Line(b - tick, b + tick, color=color, stroke_width=2))
    t = tag(label, size, color)
    t.move_to((a + b) / 2 + n * (gap + 0.5 * (abs(n[0]) * t.width + abs(n[1]) * t.height)))
    return VGroup(g, t)


class J11_RootThreeIrrational(Board):
    """√3 by descent, with triangles. If a² = 3b² with a least, the
    equilateral triangle of side a has the area of three of side b. Put
    those three in its corners: they overlap in three small triangles of
    side 2b − a and miss the middle one, of side 2a − 3b. Covered twice =
    not covered, so (2a − 3b)² = 3(2b − a)² — a smaller solution. Each
    overlap, turned half a turn about its tip, lands in a corner of the
    middle triangle, and the picture repeats, turned over and smaller by
    2 − √3, without end. Drawn at the only ratio that could work, a/b = √3."""

    def construct(self):
        r3 = np.sqrt(3.0)
        a0 = 6.0
        A = np.array([-6.3, -2.2, 0.0])
        B = A + RIGHT * a0
        C = A + a0 * _dir(PI / 3)
        G = (A + B + C) / 3
        ratio = 1 / r3

        sides = [(a0, a0 / r3)]
        for _ in range(4):
            a, b = sides[-1]
            sides.append((2 * a - 3 * b, 2 * b - a))
        for a, b in sides:
            check(close(a, r3 * b), "every level keeps a = √3·b")
            check(0 < 2 * a - 3 * b < a and 0 < 2 * b - a < b, "0 < a′ < a, 0 < b′ < b")
            check(close((2 * a - 3 * b) ** 2, 3 * (2 * b - a) ** 2),
                  "uncovered = three overlaps")
            check(3 * (1 - b / a) > 1, "no point is covered three times")
        check(close(sides[1][0] / sides[0][0], 2 - r3), "each level is 2 − √3 times the last")

        corners, overl, centre = _tri_level([A, B, C], ratio)
        # bookkeeping: the pieces' areas
        T = abs(area([A, B, C]))
        Tb = abs(area(corners[0]))
        To = abs(area(overl[0][0]))
        Tc = abs(area(centre))
        check(close(T, 3 * Tb), "△a = 3 △b")
        check(close(T, 3 * Tb - 3 * To + Tc), "△a = 3△b − 3 overlaps + uncovered")
        check(close(Tc, 3 * To), "uncovered = 3 overlaps")
        # the half-turn of each overlap about its tip lands in a corner of the middle
        for tri, k in overl:
            turned = [2 * centre[k] - v for v in tri]
            c_new = _tri_level(centre, ratio)[0]
            check(any(_same_poly(turned, cc, 1e-9) for cc in c_new),
                  "the half-turned overlap is a corner triangle of the next level")

        BLUE_OP = 0.4

        def tri(P, col, op=FILL, sw=2, stroke=WHITE):
            return mk(P, col, op, stroke_width=sw, stroke_color=stroke)

        arena = tri([A, B, C], GREY_E, 0.35, sw=4, stroke=YELLOW_B)
        dim_a = _dim2(A, C, "a", 0.24 * _dir(PI / 3 + PI / 2), YELLOW_B, 28)
        prem1 = tag("√3 = a/b   ⟹   a² = 3b²", 32, YELLOW_B)
        prem2 = tag("a, b ∈ ℕ,   a least", 24, GREY_A)
        _rows([prem1, prem2], 0.55, 3.2, 0.58)
        self.play(Create(arena), FadeIn(dim_a), run_time=1.1)
        self.play(FadeIn(prem1), FadeIn(prem2), run_time=0.8)

        # the three b-triangles slide into the corners
        cm = [tri(P_, BLUE_D, BLUE_OP) for P_ in corners]
        shifts = [_u(G - A), _u(G - B), _u(G - C)]
        dim_b = _dim2(A, A + RIGHT * sides[0][1], "b", DOWN * 0.2, BLUE_B, 26, gap=0.14)
        self.play(*[FadeIn(m, shift=0.7 * s) for m, s in zip(cm, shifts)], run_time=1.4)
        self.play(FadeIn(dim_b), run_time=0.5)
        self.hold(0.4)

        # covered twice (orange) and not covered (green)
        om = [tri(P_, ORANGE, 0.92, sw=1.5) for P_, _ in overl]
        gm = tri(centre, GREEN_D, 0.92, sw=1.5)
        o_bc = [P_ for P_, k in overl if k == 0][0]          # the overlap on BC
        dim_o = _dim2(o_bc[0], o_bc[1], "2b − a", 0.24 * _dir(PI / 6), ORANGE, 24)
        top = sorted(centre, key=lambda v: -v[1])[:2]
        dim_c = _dim2(min(top, key=lambda v: v[0]), max(top, key=lambda v: v[0]),
                      "2a − 3b", UP * 0.2, GREEN_B, 22, gap=0.12)
        eq1 = Text("(2a − 3b)² = 3(2b − a)²", font_size=30,
                   t2c={"(2a − 3b)²": GREEN_B, "3(2b − a)²": ORANGE})
        _rows([eq1], 0.55, 1.75, 0)
        self.play(*[FadeIn(m) for m in om], FadeIn(dim_o), run_time=0.9)
        self.play(FadeIn(gm), FadeIn(dim_c), run_time=0.8)
        self.play(Write(eq1), run_time=1.0)
        sub1 = tag("a′ = 2a − 3b,    b′ = 2b − a", 26)
        sub2 = Text("a′² = 3b′²,    0 < a′ < a  ↯", font_size=26, t2c={"↯": RED_B})
        _rows([sub1, sub2], 0.55, 0.95, 0.68)
        self.play(FadeIn(sub1), FadeIn(sub2), run_time=0.9)
        self.hold(0.5)

        labels0 = [dim_a, dim_b, dim_o, dim_c]
        segs = (_poly_segs([A, B, C]) + sum((_poly_segs(c_) for c_ in corners), [])
                + sum((_poly_segs(o_) for o_, _ in overl), []) + _poly_segs(centre))
        _labels_ok([d[1] for d in labels0], segs, "J11 labels")
        for t_ in (prem1, prem2, eq1, sub1, sub2):
            check(_inside(t_) and t_.get_left()[0] > B[0] + 0.5, "panel right of the figure")

        # ---- the descent: each overlap half-turns about its tip into a corner
        # of the middle triangle, which becomes the new arena
        anims = [Rotate(m, angle=PI, about_point=centre[k]) for m, (_, k) in zip(om, overl)]
        self.play(*[FadeOut(d) for d in labels0], run_time=0.5)
        self.play(*anims, gm.animate.set_fill(GREY_E, 0.55).set_stroke(ORANGE, 2.5),
                  run_time=1.6)
        corners1, overl1, centre1 = _tri_level(centre, ratio)
        for m in om:
            check(any(_same_poly(m.get_vertices(), cc, 1e-6) for cc in corners1),
                  "each overlap lands in a corner of the middle triangle")
        for m in om:
            m.set_fill(BLUE_D, BLUE_OP)
        om1 = [tri(P_, ORANGE, 0.92, sw=1) for P_, _ in overl1]
        gm1 = tri(centre1, GREEN_D, 0.92, sw=1)
        self.play(*[m.animate.set_fill(BLUE_D, 0.55) for m in om], run_time=0.4)
        self.play(*[FadeIn(m) for m in om1], FadeIn(gm1), run_time=0.8)
        self.hold(0.4)

        # ---- the same picture again: turn it over and enlarge by 2 + √3
        lvl1 = VGroup(gm, *om, *om1, gm1)
        self.play(*[FadeOut(m) for m in cm], arena.animate.set_fill(opacity=0)
                  .set_stroke(opacity=0.0), run_time=0.7)
        zf = 1 / (2 - r3)
        zl = tag("× (2 + √3)", 26, YELLOW_B).move_to(np.array([-1.0, 2.6, 0.0]))
        self.play(_spiral(lvl1, G, PI, zf), FadeIn(zl), run_time=2.0)
        check(_same_poly(gm.get_vertices(), [A, B, C], 1e-6),
              "the middle triangle, turned and enlarged, is the first arena")
        check(all(any(_same_poly(m.get_vertices(), cc, 1e-6) for cc in corners) for m in om),
              "its corner triangles are the first corner triangles")
        check(all(any(_same_poly(m.get_vertices(), P_, 1e-6) for P_, _ in overl) for m in om1)
              and _same_poly(gm1.get_vertices(), centre, 1e-6), "the same picture again")
        dim_a1 = _dim2(A, C, "a′", 0.24 * _dir(PI / 3 + PI / 2), YELLOW_B, 28)
        dim_b1 = _dim2(A, A + RIGHT * sides[0][1], "b′", DOWN * 0.2, BLUE_B, 26, gap=0.14)
        self.play(FadeIn(dim_a1), FadeIn(dim_b1), FadeOut(zl), run_time=0.7)

        # ---- and again
        corners2, overl2, centre2 = _tri_level(centre, ratio)
        self.play(*[Rotate(m, angle=PI, about_point=centre[k]) for m, (_, k) in
                    zip(om1, overl)],
                  gm1.animate.set_fill(GREY_E, 0.55).set_stroke(ORANGE, 2.5), run_time=1.3)
        check(all(any(_same_poly(m.get_vertices(), cc, 1e-6) for cc in corners2) for m in om1),
              "the second descent lands too")
        om2 = [tri(P_, ORANGE, 0.92, sw=1) for P_, _ in overl2]
        gm2 = tri(centre2, GREEN_D, 0.92, sw=1)
        tokens = ["a", ">", "a′", ">", "a″", ">", "a‴", ">", "…"]
        chain = VGroup(*[tag(t_, 30, YELLOW_B) for t_ in tokens])
        chain.arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        chain.move_to(np.array([0.55 + chain.width / 2, -0.75, 0.0]))
        never = Text("an endless descent in ℕ", font_size=24, color=RED_B)
        never.next_to(chain, DOWN, buff=0.3).align_to(chain, LEFT)
        self.play(*[m.animate.set_fill(BLUE_D, 0.55) for m in om1],
                  *[FadeIn(m) for m in om2], FadeIn(gm2), FadeIn(chain), run_time=0.9)
        self.play(FadeIn(never), run_time=0.6)
        for t_ in (chain, never):
            check(_inside(t_) and t_.get_left()[0] > B[0] + 0.5, "chain right of the figure")
        cap = caption("a² = 3b² has no least solution in ℕ   ⟹   √3 ∉ ℚ", 32)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J22

class J22_DudeneyDissection(Board):
    """Dudeney's hinged dissection. Equilateral triangle ABC of side s, D
    and E the midpoints of AB and BC. On AE extended by EF = EB = ½s draw
    the semicircle: its height over E is x with x² = AE·EF = h·½s = △, the
    triangle's area. The circle about E of radius x meets AC at J; take K
    on AC with JK = ½s, and drop perpendiculars DL, KM onto EJ. The four
    pieces, hinged at D, E and K, swing round — each a rigid turn about its
    hinge — into a square: the right angles at L and M become its corners
    and its sides are EJ = x and 2·DL = 2·KM = x."""

    def construct(self):
        r3 = np.sqrt(3.0)
        s = 2.0
        A0, B0, C0 = np.array([0.0, 0.0]), np.array([s, 0.0]), np.array([s / 2, s * r3 / 2])
        D0, E0 = (A0 + B0) / 2, (B0 + C0) / 2
        AE, EB = np.linalg.norm(E0 - A0), np.linalg.norm(B0 - E0)
        F0 = E0 + (E0 - A0) / AE * EB
        G0 = (A0 + F0) / 2
        rG = np.linalg.norm(F0 - A0) / 2
        ub = (B0 - E0) / EB
        w = E0 - G0
        tH = -np.dot(ub, w) + np.sqrt(np.dot(ub, w) ** 2 - (np.dot(w, w) - rG ** 2))
        H0 = E0 + tH * ub
        x = float(np.linalg.norm(H0 - E0))
        T = r3 / 4 * s * s
        v = (C0 - A0) / np.linalg.norm(C0 - A0)
        w = A0 - E0
        bq, cq = 2 * np.dot(v, w), np.dot(w, w) - x * x
        tJ = (-bq - np.sqrt(bq * bq - 4 * cq)) / 2
        J0 = A0 + tJ * v
        K0 = J0 + v * (s / 2)
        dEJ = (J0 - E0) / np.linalg.norm(J0 - E0)
        L0 = E0 + np.dot(D0 - E0, dEJ) * dEJ
        M0 = E0 + np.dot(K0 - E0, dEJ) * dEJ

        # ---- the construction and the dissection, checked
        check(abs(np.linalg.norm(H0 - G0) - rG) < 1e-12 and abs(np.dot(H0 - E0, F0 - A0)) < 1e-12,
              "H on the semicircle, EH ⊥ AF")
        check(close(x * x, AE * EB) and close(x * x, T), "x² = h·½s = △")
        check(0 < tJ < s and close(np.linalg.norm(J0 - E0), x), "J on AC with EJ = x")
        check(0 < tJ + s / 2 < s, "K on AC")
        check(close(2 * np.linalg.norm(D0 - L0), x) and close(2 * np.linalg.norm(K0 - M0), x),
              "2·DL = 2·KM = x")
        P1 = [A0, D0, L0, J0]
        P2 = [D0, B0, E0, L0]
        P3 = [E0, C0, K0, M0]
        P4 = [J0, M0, K0]
        check(_tiles_exactly([P1, P2, P3, P4], [A0, B0, C0]), "four pieces tile the triangle")

        def r2(p, c, t):
            return _rot(p, t, c)

        def pose(al):
            th = PI * al
            p2 = [r2(q, E0, th) for q in P2]
            Dt = r2(D0, E0, th)
            p1 = [r2(r2(q, E0, th), Dt, th) for q in P1]
            p4 = [r2(q, K0, -th) for q in P4]
            return p1, p2, P3, p4
        for al in np.linspace(0, 1, 121):
            ps = pose(al)
            worst = max(_overlap_area(ps[i], ps[j]) for i in range(4) for j in range(i + 1, 4))
            check(worst < 1e-9, "the pieces never overlap while they swing")
        f1, f2, f3, f4 = pose(1.0)
        Lp, Mp = 2 * E0 - L0, 2 * K0 - M0
        SQ = [M0, Lp, Lp + (Mp - M0), Mp]
        for i in range(4):
            p_, q_, r_ = SQ[i], SQ[(i + 1) % 4], SQ[(i + 2) % 4]
            check(close(np.linalg.norm(q_ - p_), x) and abs(np.dot(q_ - p_, r_ - q_)) < 1e-12,
                  "a square of side x")
        check(_tiles_exactly([f1, f2, f3, f4], SQ), "the swung pieces tile the square")
        check(close(abs(area(SQ)), T), "square = triangle")

        # ---- screen
        k = 1.82
        def S(p):
            return np.array([-6.5 + (p[0] + 0.35) * k, -2.95 + (p[1] + 1.2) * k, 0.0])
        A, B, C, D, E, F, G, H, J, K, L, M = (S(p_) for p_ in
                                             (A0, B0, C0, D0, E0, F0, G0, H0, J0, K0, L0, M0))

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3, fill_color=BLUE_E,
                      fill_opacity=0.35)
        ls = tag("s", 28).next_to(Line(A, B), DOWN, buff=0.16)
        self.play(Create(tri), FadeIn(ls), run_time=1.0)

        r1 = tag("△ = ½s · h", 30)
        r2_ = tag("x² = h · ½s = △", 30, YELLOW_B)
        _rows([r1, r2_], 1.3, 3.1, 0.7)

        # median AE = h, extended by EF = ½s; midpoints D, E
        dD, dE = Dot(D, radius=0.05), Dot(E, radius=0.05)
        ae = Line(A, E, color=GREY_A, stroke_width=3)
        ef = DashedLine(E, F, color=GREY_A, stroke_width=3, dash_length=0.08)
        lh = _beside(tag("h", 26), A, E, UP + LEFT * 0.3, 0.1, 0.55)
        lhalf = _beside(tag("½s", 24), E, F, DOWN + RIGHT, 0.1, 0.5)
        tEB, tEF = _ticks(E, B, 1), _ticks(E, F, 1)
        self.play(FadeIn(dD), FadeIn(dE), Create(ae), FadeIn(lh), FadeIn(r1), run_time=1.0)
        self.play(Create(ef), FadeIn(tEB), FadeIn(tEF), FadeIn(lhalf), run_time=0.9)

        # the semicircle on AF: its height over E is the mean x
        a0 = float(np.arctan2(*(F - G)[1::-1]))
        semi = Arc(radius=np.linalg.norm(F - G), start_angle=a0, angle=-PI, arc_center=G,
                   color=GREY_B, stroke_width=2.5)
        eh = Line(E, H, color=YELLOW_B, stroke_width=4)
        ext = DashedLine(B, H, color=GREY_B, stroke_width=2, dash_length=0.07)
        raE = _ra(E, H, F, 0.16)
        lx = _beside(tag("x", 28, YELLOW_B), E, H, RIGHT + UP * 0.6, 0.1, 0.62)
        self.play(Create(semi), run_time=1.0)
        self.play(Create(ext), Create(eh), Create(raE), FadeIn(lx), run_time=0.9)
        self.play(FadeIn(r2_), run_time=0.7)

        # swing x about E down to J on AC; K with JK = ½s; the cuts
        aH = float(np.arctan2(*(H - E)[1::-1]))
        aJ = float(np.arctan2(*(J - E)[1::-1]))
        span = (aJ - aH) % TAU - TAU                     # clockwise, under AB
        swing = Arc(radius=x * k, start_angle=aH, angle=span, arc_center=E,
                    color=YELLOW_B, stroke_width=2)
        check(close(E + x * k * _dir(aH + span), J, 1e-6), "the arc ends at J")
        dJ, dK = Dot(J, radius=0.05), Dot(K, radius=0.05)
        tJK = _ticks(J, K, 1)
        ljk = _beside(tag("½s", 24), J, K, LEFT + UP * 0.6, 0.12)
        self.play(Create(swing), run_time=1.0)
        self.play(FadeIn(dJ), FadeIn(dK), FadeIn(tJK), FadeIn(ljk), run_time=0.5)
        constr = [semi, ext, eh, raE, lx, lh, lhalf, tEB, tEF, ae, ef]
        segs_c = ([(A, E), (B, H), (E, H)] + _poly_segs([A, B, C])
                  + _arc_segs(G, np.linalg.norm(F - G), a0, a0 - PI, 48)
                  + _arc_segs(E, x * k, aH, aH + span, 48))
        _labels_ok([ls, lh, lhalf, lx], segs_c, "J22 construction labels")

        # the cuts: EJ (= x), and the perpendiculars DL, KM onto it
        ej = Line(E, J, color=WHITE, stroke_width=3)
        dl = Line(D, L, color=WHITE, stroke_width=3)
        km = Line(K, M, color=WHITE, stroke_width=3)
        raL = _ra(L, D, E, 0.13)
        raM = _ra(M, K, E, 0.13)
        lxj = _beside(tag("x", 26, YELLOW_B), E, J, D - L, 0.1, 0.84)
        self.play(*[FadeOut(m) for m in constr], run_time=0.6)
        self.play(Create(ej), FadeOut(swing), FadeIn(lxj), run_time=0.7)
        self.play(Create(dl), Create(km), Create(raL), Create(raM), run_time=0.8)
        _labels_ok([ls, lxj, ljk], _poly_segs([A, B, C]) + [(E, J), (D, L), (K, M)]
                   + _ra_segs(L, D, E, 0.13) + _ra_segs(M, K, E, 0.13), "J22 cut labels")
        self.hold(0.3)

        # ---- the pieces, hinged at D, E, K
        cols = [TEAL_D, BLUE_D, ORANGE, GOLD_D]
        pieces = VGroup(*[mk([S(q) for q in P_], c_, 0.85, stroke_width=2)
                          for P_, c_ in zip((P1, P2, P3, P4), cols)])
        aids = [tJK, ljk, lxj, ls, ej, dl, km, raL, raM, dJ, dK]
        hinges = VGroup(Dot(D, radius=0.07, color=WHITE), Dot(E, radius=0.07, color=WHITE),
                        Dot(K, radius=0.07, color=WHITE))
        self.play(FadeIn(pieces), *[FadeOut(m) for m in aids], FadeOut(dD), FadeOut(dE),
                  FadeOut(tri), FadeIn(hinges), run_time=1.0)
        self.hold(0.3)

        start = pieces.copy()

        def upd(mob, al):
            th = PI * al
            Dt = to3(_rot(D, th, E))                 # the hinge D, carried round E
            mob[0].become(start[0].copy().rotate(th, about_point=E).rotate(th, about_point=Dt))
            mob[1].become(start[1].copy().rotate(th, about_point=E))
            mob[2].become(start[2].copy())
            mob[3].become(start[3].copy().rotate(-th, about_point=K))
            hinges[0].move_to(Dt)

        self.play(UpdateFromAlphaFunc(pieces, upd), run_time=4.0, rate_func=smooth)
        want = [[S(q) for q in P_] for P_ in (f1, f2, f3, f4)]
        for m, wpts in zip(pieces, want):
            check(_same_poly(m.get_vertices(), wpts, 1e-6), "each piece lands in the square")
        sqm = Polygon(*[S(q) for q in SQ], stroke_color=YELLOW_B, stroke_width=5)
        ras = VGroup(*[_ra(S(SQ[i]), S(SQ[(i + 1) % 4]), S(SQ[i - 1]), 0.16, YELLOW_B)
                       for i in range(4)])
        lx1 = _beside(tag("x", 30, YELLOW_B), S(SQ[0]), S(SQ[1]), S(SQ[0]) - S(SQ[3]), 0.12)
        lx2 = _beside(tag("x", 30, YELLOW_B), S(SQ[1]), S(SQ[2]), S(SQ[1]) - S(SQ[0]), 0.12)
        self.play(Create(sqm), Create(ras), FadeIn(lx1), FadeIn(lx2), FadeOut(hinges),
                  run_time=1.0)
        r3_ = tag("□ = x² = △", 30, YELLOW_B)
        r4_ = tag("(√3/4) s² = x²", 34, YELLOW_B)
        _rows([r3_, r4_], 1.3, 1.6, 0.8)
        self.play(FadeIn(r3_), run_time=0.7)
        self.play(FadeIn(r4_), run_time=0.7)
        segs = _poly_segs([S(q) for q in SQ]) + sum((_poly_segs(w_) for w_ in want), [])
        _labels_ok([lx1, lx2], segs, "J22 square labels")
        for t_ in (r1, r2_, r3_, r4_):
            check(_inside(t_) and t_.get_left()[0] > sqm.get_right()[0] + 0.3,
                  "panel right of the figure")
        cap = caption("four hinged pieces turn the triangle into the square:   "
                      "(√3/4) s² = x²", 28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J27

class J27_SixtyFourSixtyFive(Board):
    """A caution. Cut the 8 × 8 square into two right triangles (legs 3, 8)
    and two right trapezoids (5, 3, 5): two slide and two make a quarter
    turn into a 5 × 13 rectangle, 65 cells. The 'diagonal' is not straight:
    the triangles' hypotenuses rise 3 in 8, the trapezoids' slanted sides 2
    in 5, and 3/8 < 5/13 < 2/5. Stretched across the diagonal, the four
    pieces leave a thin parallelogram on the sides (8, 3) and (5, 2) — area
    |8·2 − 3·5| = 1, which is Cassini's 5·13 − 8² = 1."""

    def construct(self):
        u = 0.42
        sq0 = np.array([-6.3, -0.15, 0.0])          # the square's corner (0, 0)
        re0 = np.array([0.8, 0.35, 0.0])            # the rectangle's corner (0, 0)

        def SqP(p):
            return sq0 + u * np.array([p[0], p[1], 0.0])

        def ReP(p):
            return re0 + u * np.array([p[0], p[1], 0.0])

        pieces = {"TriA": [(0, 5), (8, 5), (8, 8)], "TriB": [(0, 5), (8, 8), (0, 8)],
                  "TrapL": [(0, 0), (5, 0), (3, 5), (0, 5)],
                  "TrapR": [(5, 0), (8, 0), (8, 5), (3, 5)]}
        turn = {"TriA": 0.0, "TriB": 0.0, "TrapL": -PI / 2, "TrapR": -PI / 2}
        move = {"TriA": lambda p: (p[0], p[1] - 5), "TriB": lambda p: (p[0] + 5, p[1] - 3),
                "TrapL": lambda p: tuple(_rot(p, -PI / 2, (2.5, 2.5))),
                "TrapR": lambda p: tuple(_rot(p, -PI / 2, (8.0, 0.0)))}
        dest = {k: [move[k](q) for q in P_] for k, P_ in pieces.items()}
        sq_cells = [(0, 0), (8, 0), (8, 8), (0, 8)]
        rect = [(0, 0), (13, 0), (13, 5), (0, 5)]
        gap = [(0, 0), (8, 3), (13, 5), (5, 2)]
        check(_tiles_exactly(list(pieces.values()), sq_cells), "the four pieces tile 8 × 8")
        check(_tiles_exactly(list(dest.values()) + [gap], rect),
              "in the rectangle they leave exactly the thin parallelogram")
        check(close(sum(abs(area(P_)) for P_ in pieces.values()), 64)
              and close(abs(area(gap)), 1) and abs(8 * 2 - 3 * 5) == 1, "64 + 1 = 65")
        check(3 / 8 < 5 / 13 < 2 / 5 and 5 * 13 - 8 * 8 == 1, "3/8 < 5/13 < 2/5; Cassini")
        for k in pieces:
            for q in dest[k]:
                check(abs(q[0] - round(q[0])) < 1e-9 and abs(q[1] - round(q[1])) < 1e-9,
                      "the pieces land on the grid")

        COL = {"TriA": RED_D, "TriB": BLUE_D, "TrapL": GOLD_D, "TrapR": GREEN_D}

        def grid(P, W, Hh):
            g = VGroup()
            for i in range(W + 1):
                g.add(Line(P((i, 0)), P((i, Hh)), color=GREY_D, stroke_width=1))
            for j in range(Hh + 1):
                g.add(Line(P((0, j)), P((W, j)), color=GREY_D, stroke_width=1))
            return g

        gS, gR = grid(SqP, 8, 8), grid(ReP, 13, 5)
        mob = {k: mk([SqP(q) for q in P_], COL[k], FILL, stroke_width=2)
               for k, P_ in pieces.items()}
        l64 = tag("8 · 8 = 64", 30).next_to(gS, UP, buff=0.18)
        self.play(FadeIn(gS), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(mob[k]) for k in ("TrapL", "TrapR", "TriA", "TriB")],
                              lag_ratio=0.3), FadeIn(l64), run_time=1.6)
        self.hold(0.4)

        # two slide, two make a quarter turn
        l65 = tag("5 · 13 = 65", 30).next_to(gR, UP, buff=0.18)
        self.play(FadeIn(gR), FadeIn(l65), run_time=0.6)
        anims = []
        for k in ("TriA", "TriB", "TrapL", "TrapR"):
            c0 = np.mean([SqP(q) for q in pieces[k]], axis=0)
            c1 = np.mean([ReP(q) for q in dest[k]], axis=0)
            anims.append(_glide(mob[k], c0, c1, turn[k]))
        self.play(*anims, run_time=2.4)
        for k in pieces:
            check(_same_poly(mob[k].get_vertices(), [ReP(q) for q in dest[k]], 1e-6),
                  f"{k} lands in the rectangle")
        q65 = Text("64 = 65 ?", font_size=34, color=RED_B).move_to([-1.07, 1.75, 0])
        self.play(FadeIn(q65), run_time=0.7)
        self.hold(0.6)

        # ---- the resolution: lay the diagonal band flat and stretch it across
        L = float(np.hypot(13, 5))
        dd = np.array([13.0, 5.0]) / L
        nn = np.array([-dd[1], dd[0]])
        w = 0.14                                   # band half-width, in cells
        band = [tuple(-w * nn), tuple(L * dd - w * nn), tuple(L * dd + w * nn), tuple(w * nn)]
        band = [tuple(q) for q in _clip(_ccw(band), _ccw(rect))]   # within the rectangle
        diag = DashedLine(ReP((0, 0)), ReP((13, 5)), color=WHITE, stroke_width=2,
                          dash_length=0.07)
        bandm = Polygon(*[ReP(q) for q in band], stroke_color=YELLOW_B, stroke_width=2)
        self.play(Create(diag), run_time=0.7)
        self.play(Create(bandm), run_time=0.6)

        ks, mstr = 0.861, 8.0
        O_s = np.array([-6.0, -1.55, 0.0])

        def ST(p):
            p = np.asarray(p, float)
            return O_s + np.array([ks * np.dot(p, dd), ks * mstr * np.dot(p, nn), 0.0])

        clipped = {k: _clip(_ccw(dest[k]), _ccw(band)) for k in dest}
        for k, P_ in clipped.items():
            check(len(P_) >= 3, f"{k} reaches into the band")
        check(_tiles_exactly([clipped[k] for k in clipped] + [gap], band, 150),
              "inside the band: four pieces and the gap")
        # the band copy, first as it is, then turned flat and enlarged (a similarity)
        strip_now = VGroup(*[mk([ReP(q) for q in clipped[k]], COL[k], FILL, stroke_width=1.5)
                             for k in ("TriA", "TrapR", "TrapL", "TriB")])
        dg_now = DashedLine(ReP((0, 0)), ReP((13, 5)), color=WHITE, stroke_width=2,
                            dash_length=0.07)
        grp = VGroup(strip_now, dg_now)
        cb = ReP(tuple(L / 2 * dd))
        tgt_c = ST(tuple(L / 2 * dd))
        ang = -float(np.arctan2(5, 13))
        sc = ks / u
        start = grp.copy()

        def upd(m, a):
            m.become(start.copy().rotate(a * ang, about_point=cb).scale(sc ** a, about_point=cb)
                     .shift(a * (tgt_c - cb)))
        self.add(grp)
        self.play(UpdateFromAlphaFunc(grp, upd), run_time=1.6)
        self.play(grp.animate.stretch(mstr, 1, about_point=tgt_c), run_time=1.3)
        want = {k: [ST(q) for q in clipped[k]] for k in clipped}
        for m, k in zip(strip_now, ("TriA", "TrapR", "TrapL", "TriB")):
            check(_same_poly(m.get_vertices(), want[k], 1e-6), f"{k}: the band, laid flat")
        # "stretched 8 times across": a small double arrow and × 8
        x8a = DoubleArrow(ORIGIN, UP * 0.42, buff=0, color=YELLOW_B, stroke_width=3,
                          tip_length=0.1, max_tip_length_to_length_ratio=0.4)
        x8t = tag("× 8", 24, YELLOW_B)
        x8 = VGroup(x8a, x8t.next_to(x8a, RIGHT, buff=0.08))
        x8.move_to(np.array([5.55, -0.3, 0.0]))
        check(_apart(x8, gR, 0.1) and x8.get_bottom()[1] > O_s[1] + ks * mstr * w + 0.05,
              "× 8 clear of the rectangle and above the strip")
        gapm = mk([ST(q) for q in gap], PINK, 0.9, stroke_width=0)
        self.play(FadeIn(gapm), FadeIn(x8), run_time=0.8)
        one = tag("1", 28, BLACK).move_to(ST((6.5, 2.5)))
        check(_pip(one.get_center(), [ST(q) for q in gap]) and
              all(_pip(c_, [ST(q) for q in gap]) for c_ in (one.get_corner(UL), one.get_corner(UR),
                                                             one.get_corner(DL), one.get_corner(DR))),
              "the 1 sits inside the gap")
        self.play(FadeIn(one), run_time=0.5)

        # slopes of the four edges along the gap
        def slope_tag(s_, p, q, col, side):
            m = tag(s_, 22, col)
            mid = (ST(p) + ST(q)) / 2
            return m.move_to(np.array([mid[0], O_s[1] + side * (ks * mstr * w + 0.22), 0.0]))
        st = VGroup(slope_tag("3/8", (0, 0), (8, 3), RED_B, -1),
                    slope_tag("2/5", (8, 3), (13, 5), GREEN_B, -1),
                    slope_tag("2/5", (0, 0), (5, 2), GOLD_B, +1),
                    slope_tag("3/8", (5, 2), (13, 5), BLUE_B, +1))
        self.play(FadeIn(st), run_time=0.8)
        r_sl = tag("3/8  <  5/13  <  2/5", 24).move_to([-1.07, 1.0, 0])
        r_gap = tag("gap = 8 · 2 − 3 · 5 = 1", 21, PINK).move_to([-1.07, 0.35, 0])
        self.play(FadeIn(r_sl), run_time=0.7)
        self.play(FadeIn(r_gap), run_time=0.7)

        texts = [l64, l65, q65, x8, r_sl, r_gap] + list(st)
        for i in range(len(texts)):
            check(_inside(texts[i]), "J27 text inside")
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.06), "J27 texts apart")
        for t_ in (q65, r_sl, r_gap):
            check(_apart(t_, gS, 0.15) and _apart(t_, gR, 0.15) and _apart(t_, grp, 0.1),
                  "middle texts clear of the figures")
        for t_ in st:
            check(_apart(t_, grp, 0.05) and _apart(t_, gS, 0.05), "slope tags clear")
        cap = caption("8² = 5 · 13 − 1: the pieces leave a gap of area 1 along the diagonal",
                      28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K19

class K19_PascalFibonacci(Board):
    """Write Pascal's triangle flush left: row n, column k holds C(n, k), and
    each entry is the one above it plus the one above-left. The shallow
    diagonal Dₙ = {C(n−k, k)} then runs at 45°. An entry of Dₙ has its
    'above' on Dₙ₋₁ and its 'above-left' on Dₙ₋₂, and as k runs, these use
    every entry of Dₙ₋₁ and of Dₙ₋₂ exactly once. So each diagonal sum is
    the sum of the two before it, starting 1, 1: the Fibonacci numbers.
    Shown for D₆ = D₅ + D₄: 13 = 8 + 5."""

    def construct(self):
        N = 8
        fib = [1, 1]
        while len(fib) < N + 2:
            fib.append(fib[-1] + fib[-2])

        def C(n, k):
            return comb(n, k) if 0 <= k <= n else 0

        def diag(m):
            return [(m - k, k) for k in range(m // 2 + 1)]
        for m in range(N + 1):
            check(sum(C(*rc) for rc in diag(m)) == fib[m], f"D{m} sums to F{m + 1}")
        for m in range(2, 30):
            above = sorted((n - 1, k) for n, k in diag(m) if C(n - 1, k) > 0)
            left = sorted((n - 1, k - 1) for n, k in diag(m) if C(n - 1, k - 1) > 0)
            check(above == sorted(diag(m - 1)) and left == sorted(diag(m - 2)),
                  "the parents of Dₙ are Dₙ₋₁ and Dₙ₋₂, each entry once")
            check(all(C(n, k) == C(n - 1, k) + C(n - 1, k - 1) for n, k in diag(m)),
                  "Pascal's rule on the diagonal")

        c = 0.62
        X0, Y0 = -5.0, 3.25

        def pos(n, k):
            return np.array([X0 + k * c, Y0 - n * c, 0.0])

        nums = {}
        for n in range(N + 1):
            for k in range(n + 1):
                nums[(n, k)] = tag(str(C(n, k)), 22).move_to(pos(n, k))
        rows = [VGroup(*[nums[(n, k)] for k in range(n + 1)]) for n in range(N + 1)]
        rule = tag("C(n, k) = C(n−1, k) + C(n−1, k−1)", 24, GREY_A)
        rule.move_to([0.6 + rule.width / 2, 3.15, 0])
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.25), FadeIn(rule),
                  run_time=2.0)

        def arrow(a, b, col, width=3):
            pa, pb = pos(*a), pos(*b)
            u_ = _u(pb - pa)

            def reach(m):
                hw, hh = m.width / 2 + 0.06, m.height / 2 + 0.06
                return min(hw / max(abs(u_[0]), 1e-9), hh / max(abs(u_[1]), 1e-9))
            return Arrow(pa + u_ * reach(nums[a]), pb - u_ * reach(nums[b]), buff=0,
                         color=col, stroke_width=width, tip_length=0.12,
                         max_tip_length_to_length_ratio=0.5)

        # Pascal's rule once: 6 = 3 + 3
        ex = [arrow((3, 1), (4, 2), YELLOW_B), arrow((3, 2), (4, 2), YELLOW_B)]
        self.play(*[GrowArrow(a) for a in ex], nums[(4, 2)].animate.set_color(YELLOW_B),
                  run_time=0.8)
        self.hold(0.3)
        self.play(*[FadeOut(a) for a in ex], nums[(4, 2)].animate.set_color(WHITE), run_time=0.4)

        # the shallow diagonals, coloured in a cycle of three, and their sums
        cols = [GREEN_B, BLUE_B, ORANGE]
        sums, legs = {}, {}
        xs = X0 - 1.4 * c
        for m in range(N + 1):
            col = cols[m % 3]
            sp = np.array([xs, Y0 - (m + 1.4) * c, 0.0])
            sums[m] = tag(str(fib[m]), 24, col).move_to(sp)
            u_ = _u(pos(m, 0) - sp)
            legs[m] = Line(sp + u_ * 0.24, pos(m, 0) - u_ * 0.2, color=col, stroke_width=2)
            self.play(*[nums[rc].animate.set_color(col) for rc in diag(m)],
                      Create(legs[m]), FadeIn(sums[m]), run_time=0.42 if m < 6 else 0.5)
        self.hold(0.4)

        # D₆ = D₅ + D₄: every entry of D₆ is its 'above' (D₅) + its 'above-left' (D₄)
        m = 6
        others = [rc for rc in nums if sum(rc) not in (m, m - 1, m - 2)]
        self.play(*[nums[rc].animate.set_opacity(0.25) for rc in others],
                  *[sums[j].animate.set_opacity(0.3) for j in sums if j not in (4, 5, 6)],
                  *[legs[j].animate.set_stroke(opacity=0.3) for j in legs if j not in (4, 5, 6)],
                  run_time=0.6)
        arrs = []
        for n, k in diag(m):
            grp = []
            if C(n - 1, k) > 0:
                grp.append(arrow((n - 1, k), (n, k), cols[(m - 1) % 3], 3.5))
            if C(n - 1, k - 1) > 0:
                grp.append(arrow((n - 1, k - 1), (n, k), cols[(m - 2) % 3], 3.5))
            arrs += grp
            self.play(*[GrowArrow(a) for a in grp], Indicate(nums[(n, k)], color=cols[m % 3],
                                                             scale_factor=1.25), run_time=0.6)
        t1 = Text("13 = 1 + 5 + 6 + 1", font_size=28, color=GREEN_B)
        t2 = Text("= (1 + 4 + 3) + (1 + 3 + 1)", font_size=28,
                  t2c={"(1 + 4 + 3)": ORANGE, "(1 + 3 + 1)": BLUE_B})
        t3 = Text("= 8 + 5", font_size=28, t2c={"8": ORANGE, "5": BLUE_B})
        t1.move_to([0.6 + t1.width / 2, 1.9, 0])
        eqx = t1.get_left()[0] + Text("13 ", font_size=28).width
        t2.move_to([eqx + t2.width / 2, 1.25, 0])
        t3.move_to([eqx + t3.width / 2, 0.6, 0])
        self.play(FadeIn(t1), run_time=0.6)
        self.play(FadeIn(t2), run_time=0.7)
        self.play(FadeIn(t3), Indicate(sums[5], color=ORANGE), Indicate(sums[4], color=BLUE_B),
                  run_time=0.8)
        gen = tag("∑Dₙ = ∑Dₙ₋₁ + ∑Dₙ₋₂", 26, YELLOW_B)
        gen.move_to([0.6 + gen.width / 2, -0.4, 0])
        seq = tag("1, 1, 2, 3, 5, 8, 13, 21, 34, …", 26, YELLOW_B)
        seq.move_to([0.6 + seq.width / 2, -1.05, 0])
        self.play(FadeIn(gen), FadeIn(seq), *[nums[rc].animate.set_opacity(1.0) for rc in others],
                  *[sums[j].animate.set_opacity(1.0) for j in sums],
                  *[legs[j].animate.set_stroke(opacity=1.0) for j in legs], run_time=0.8)

        # hygiene: the arrows and legs stay off the numbers; texts apart
        segs = ([(a.get_start(), a.get_end()) for a in arrs]
                + [(l_.get_start(), l_.get_end()) for l_ in legs.values()])
        for t_ in list(nums.values()) + list(sums.values()):
            check(_hit(t_, segs, 0.03) is None, "arrows and legs clear of the numbers")
        for t_ in (rule, t1, t2, t3, gen, seq):
            check(_inside(t_) and t_.get_left()[0] > pos(N, N)[0] + 0.4, "panel right")
        allt = list(nums.values()) + list(sums.values())
        for i in range(len(allt)):
            for j in range(i + 1, len(allt)):
                check(_apart(allt[i], allt[j], 0.05), "numbers apart")
        cap = caption("∑ C(n − k, k) = Fₙ₊₁:  each shallow diagonal adds up the two "
                      "before it", 27)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)
