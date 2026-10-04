# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w2f.py — proofs without words, 2D (manim):
#     A14 similar figures on the sides      A12 Pythagoras by intersecting chords
#     A21 the parallelogram law             A25 squaring a rectangle
#     A13 Pappus's area theorem             A19 law of cosines by chords
#     A20 law of cosines, obtuse case       G27 sine of a sum by area
#     G15 cos 36° and the golden ratio
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Every piece moves rigidly (shift, Rotate, a fold in space), by an announced
# similarity, or by a shear that keeps its base and slides the opposite side
# along its own line. Every landing, area and length claim is checked with
# check(...) before or right after it is drawn, and labels are checked
# against the lines they must clear, so a wrong construction fails the render.


# ------------------------------------------------------------ private helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _dir(t):
    """Unit screen vector at angle t (radians)."""
    return np.array([np.cos(t), np.sin(t), 0.0])


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _cross2(u, v):
    return float(u[0] * v[1] - u[1] * v[0])


def _rot2(p, t, about=(0.0, 0.0)):
    """The 2D point p turned by the angle t about the point `about`."""
    p = np.asarray(p, float)[:2]
    o = np.asarray(about, float)[:2]
    cs, sn = np.cos(t), np.sin(t)
    d = p - o
    return o + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])


def _foot(p, u, v):
    """Foot of the perpendicular from p onto the line uv (2D or 3D)."""
    p, u, v = (np.asarray(x, float) for x in (p, u, v))
    d = v - u
    return u + np.dot(p - u, d) / np.dot(d, d) * d


def _refl(p, c, u):
    """Mirror image of p in the line through c along u (2D or 3D)."""
    p, c, u = (np.asarray(x, float) for x in (p, c, u))
    u = u / np.linalg.norm(u)
    d = p - c
    return c + 2 * np.dot(d, u) * u - d


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ticks(p, q, n=1, color=WHITE, size=0.12, width=2.5, at=0.5):
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


def _chevron(p, q, color=YELLOW_B, size=0.16, width=3, n=1, at=0.5):
    """Parallel-line mark: n small '>' on segment pq, pointing p -> q."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = p + (q - p) * at
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7,
                                          tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


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
    `factor` (geometrically in time) — a similar copy on every frame."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=c)
                 .scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


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


def _lbl(s, size=26, color=WHITE, bg=0.0):
    """tag() with an optional dark backing box for text sitting on fills."""
    t = tag(s, size, color)
    if bg > 0:
        box = BackgroundRectangle(t, color=BLACK, fill_opacity=bg, buff=0.05)
        return VGroup(box, t)
    return t


def _dim(p, q, side, color=GREY_A, off=0.3, tick=0.11, width=2.5):
    """Dimension bracket for the screen segment pq, drawn `off` away on the
    side of the screen vector `side`, with end ticks towards pq."""
    p, q, sd = to3(p), to3(q), to3(side)
    sd = sd / np.linalg.norm(sd)
    p1, q1 = p + sd * off, q + sd * off
    return VGroup(Line(p1, q1, color=color, stroke_width=width),
                  Line(p1, p1 - sd * tick, color=color, stroke_width=width),
                  Line(q1, q1 - sd * tick, color=color, stroke_width=width))


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


def _inside_or_on(p, poly, tol=1e-9):
    """p inside or on the boundary of the convex polygon poly (any sense)."""
    s = np.sign(area(poly))
    n = len(poly)
    for i in range(n):
        e = np.asarray(poly[(i + 1) % n], float) - np.asarray(poly[i], float)
        if s * _cross2(e, np.asarray(p, float) - np.asarray(poly[i], float)) < -tol:
            return False
    return True


def _landed(mob, pts, tol=1e-6):
    """The polygon mobject's vertices are exactly the screen points pts."""
    return _same_poly(mob.get_vertices(), [to3(p) for p in pts], tol)


# ---- label hygiene: every label is checked against the lines it must clear

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _poly_segs(pts):
    pts = [to3(p) for p in pts]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _arc_segs(c, r, a0, a1, n=48):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    pts = [c + r * _dir(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _angle_segs(v, p, q, r, n=12):
    """Segments of the non-reflex arc of radius r at v between p and q."""
    v = to3(v)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return _arc_segs(v, r, a1, a1 + span, n)


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


def _overlap(a, b, gap=0.05):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _in_frame(*mobs, tol=0.02):
    return all(m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
               and m.get_top()[1] <= SAFE_TOP + tol
               and m.get_bottom()[1] >= SAFE_BOTTOM - tol for m in mobs)


def _name(m):
    t = getattr(m, "text", None)
    if t:
        return t
    for s in getattr(m, "submobjects", []):
        t = getattr(s, "text", None)
        if t:
            return t
    return type(m).__name__


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


def _box_inside(m, poly, pad=0.04):
    """The label's box (grown by pad) lies inside the screen polygon."""
    x0, x1, y0, y1 = _box(m, pad)
    return all(_pip((x, y), [to3(p) for p in poly])
               for x in (x0, x1) for y in (y0, y1))


def _in_safe(mobs, tol=0.03):
    """Every mobject lies inside the content area (the band below
    SAFE_BOTTOM belongs to the caption)."""
    for m in mobs:
        if not m.has_points() and not m.submobjects:
            continue
        if (m.get_left()[0] < -SAFE_X - tol or m.get_right()[0] > SAFE_X + tol
                or m.get_bottom()[1] < SAFE_BOTTOM - tol
                or m.get_top()[1] > SAFE_TOP + tol):
            return False
    return True


def _cplx(p):
    return complex(float(p[0]), float(p[1]))


def _similarity(P0, Q0, P1, Q1):
    """The direct similarity z -> m z + t carrying P0 -> P1 and Q0 -> Q1
    (2D points): returns (m, t, fixed point) with complex m, t."""
    p0, q0, p1, q1 = map(_cplx, (P0, Q0, P1, Q1))
    m = (q1 - p1) / (q0 - p0)
    t = p1 - m * p0
    z = t / (1 - m)
    return m, t, np.array([z.real, z.imag])


def _apply(m, t, p):
    z = m * _cplx(p) + t
    return np.array([z.real, z.imag])


# ================================================ A14 SIMILAR FIGURES ON SIDES

class A14_SimilarFigures(Board):
    """Euclid VI.31. On each side of a right triangle stands the square on
    that side and, inside it, a figure; the three figures are similar and
    stand alike on their sides. The one similarity that carries the side c
    onto the side a carries the c-square onto the a-square and the c-figure
    onto the a-figure — so every figure is the same fraction k of its own
    square. With a² + b² = c², the figures add up the same way: semicircles
    (k = π/8), equilateral triangles (k = √3/4) or any shape at all."""

    def construct(self):
        a, b = 2.5, 3.6
        c = float(np.hypot(a, b))
        A, B = np.zeros(2), np.array([c, 0.0])
        C = np.array([b * b / c, a * b / c])          # the right angle
        check(abs(np.dot(A - C, B - C)) < 1e-9 and close(np.linalg.norm(B - C), a)
              and close(np.linalg.norm(C - A), b), "right triangle, legs a, b")
        check(_cross2(B - A, C - A) > 0, "A, B, C counter-clockwise")
        # each side as a directed edge of the counter-clockwise triangle:
        # its right-hand side is the outside
        side = {"c": (A, B), "a": (B, C), "b": (C, A)}

        def local(key, uv):
            """(u along the side, v outwards) -> the plane, in side units."""
            P, Q = side[key]
            d = Q - P
            n = np.array([d[1], -d[0]])
            return [P + u * d + v * n for (u, v) in uv]

        SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]
        ts = np.linspace(0.0, PI, 49)
        kinds = {
            "semi": [(0.5 - 0.5 * np.cos(t), 0.5 * np.sin(t)) for t in ts],
            "equi": [(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)],
            "any": [(0.0, 0.0), (1.0, 0.0), (0.9, 0.3), (0.52, 0.52),
                    (0.36, 0.86), (0.12, 0.44)],
        }
        # where each kind carries its k-label (u, v), well inside the figure
        lab_uv = {"semi": (0.5, 0.2), "equi": (0.5, 0.27), "any": (0.42, 0.33)}
        k_val = {"semi": PI / 8, "equi": np.sqrt(3) / 4}

        for key, s in (("a", a), ("b", b), ("c", c)):
            sq = local(key, SQ)
            check(all(close(np.linalg.norm(sq[(i + 1) % 4] - sq[i]), s)
                      for i in range(4)), "squares on the sides")
            check(not _pip(sq[0] + 0.25 * (sq[2] - sq[0]), [A, B, C]),
                  "each square stands outside the triangle")
            for kind, uv in kinds.items():
                fig = local(key, uv)
                check(all(_inside_or_on(p, sq) for p in fig),
                      "each figure lies in the square on its side")
                kf = abs(area(fig)) / s ** 2
                if kind in k_val:
                    check(abs(kf - k_val[kind]) < 2e-3 if kind == "semi"
                          else abs(kf - k_val[kind]) < 1e-12,
                          "semicircle π/8, equilateral √3/4 of the square")
        sims = {}
        for key in ("a", "b"):
            m, t, z = _similarity(A, B, *side[key])
            sims[key] = (m, t, z)
            for uv in [SQ] + list(kinds.values()):
                check(all(close(_apply(m, t, p), q) for p, q in
                          zip(local("c", uv), local(key, uv))),
                      "one similarity carries the c-square and the c-figure "
                      "onto the square and the figure on the leg")
            check(abs(abs(m) - {"a": a, "b": b}[key] / c) < 1e-12,
                  "the similarity scales by leg/c")
        check(abs(a * a + b * b - c * c) < 1e-9, "a² + b² = c²")

        # ---- screen
        phi = 82 * DEGREES
        allp = np.array([_rot2(p, phi) for key in "abc" for p in local(key, SQ)])
        F = Frame(allp[:, 0].min() - 0.12, allp[:, 0].max() + 0.12,
                  allp[:, 1].min() - 0.12, allp[:, 1].max() + 0.12,
                  max_w=9.6, centre=(-1.35, (SAFE_TOP + SAFE_BOTTOM) / 2))

        def S(p):
            return F.P(_rot2(p, phi))

        def poly(pts, col, op=FILL, **kw):
            return mk([S(p) for p in pts], col, op, **kw)

        COL = {"a": ORANGE, "b": TEAL_D, "c": BLUE_D}
        tri = Polygon(S(A), S(B), S(C), stroke_color=WHITE, stroke_width=3)
        raC = _ra(S(C), S(A), S(B), 0.2)
        sqs = {key: poly(local(key, SQ), GREY_D, 0.3, stroke_color=GREY_A,
                         stroke_width=2.5) for key in "abc"}
        sq_lab = {key: tag(key + "²", 30, GREY_A).move_to(
            S(local(key, [(0.5, 0.5)])[0])) for key in "abc"}

        self.play(Create(tri), Create(raC), run_time=0.8)
        self.play(*[FadeIn(sqs[k]) for k in "abc"],
                  *[FadeIn(sq_lab[k]) for k in "abc"], run_time=1.0)
        prem = tag("a² + b²  =  c²", 32).move_to([4.6, 2.9, 0.0])
        self.play(FadeIn(prem), run_time=0.6)
        self.bring_to_front(tri, raC)

        def figs(kind):
            return {key: poly(local(key, kinds[kind]), COL[key]) for key in "abc"}

        def klabs(kind):
            out = {}
            for key in "abc":
                p = S(local(key, [lab_uv[kind]])[0])
                out[key] = _lbl("k·" + key + "²", 24, WHITE, 0.0).move_to(p)
                check(_box_inside(out[key], [S(q) for q in local(key, kinds[kind])]),
                      f"label k·{key}² inside its figure")
            return out

        def flight(kind, keys, rt=2.0):
            """Copies of the c-square + c-figure spiral onto the legs."""
            cards = []
            for key in keys:
                card = VGroup(
                    poly(local("c", SQ), GREY_D, 0.0, stroke_color=YELLOW_B,
                         stroke_width=4),
                    poly(local("c", kinds[kind]), COL["c"], 0.85,
                         stroke_color=YELLOW_B, stroke_width=3))
                cards.append(card)
            self.play(*[FadeIn(cd) for cd in cards], run_time=0.35)
            anims = []
            for key, card in zip(keys, cards):
                m, t, z = sims[key]
                anims.append(_spiral(card, S(z), float(np.angle(m)), abs(m)))
            self.play(*anims, run_time=rt)
            for key, card in zip(keys, cards):
                check(_landed(card[0], [S(p) for p in local(key, SQ)])
                      and _landed(card[1], [S(p) for p in local(key, kinds[kind])]),
                      f"the copy lands on the {key}-square and its figure")
            self.play(*[c_.animate.set_fill(opacity=0).set_stroke(opacity=0)
                        for c_ in cards], run_time=0.5)
            self.remove(*cards)

        # ---- semicircles
        cur = figs("semi")
        self.play(*[FadeOut(sq_lab[k]) for k in "abc"],
                  *[FadeIn(cur[k]) for k in "abc"], run_time=0.9)
        self.bring_to_front(tri, raC)
        flight("semi", ["a"])
        flight("semi", ["b"])
        kl = klabs("semi")
        kv = tag("k = π/8", 30, YELLOW_B).next_to(prem, DOWN, buff=0.45)
        self.play(*[FadeIn(kl[k]) for k in "abc"], FadeIn(kv), run_time=0.7)
        self.hold(0.6)

        # ---- equilateral triangles
        nxt, nkl = figs("equi"), klabs("equi")
        nkv = tag("k = √3/4", 30, YELLOW_B).move_to(kv)
        self.play(*[FadeOut(cur[k]) for k in "abc"], *[FadeOut(kl[k]) for k in "abc"],
                  *[FadeIn(nxt[k]) for k in "abc"], *[FadeIn(nkl[k]) for k in "abc"],
                  FadeOut(kv), FadeIn(nkv), run_time=1.0)
        self.bring_to_front(tri, raC)
        cur, kl, kv = nxt, nkl, nkv
        self.hold(0.8)

        # ---- any figure: the same similarity still carries it across
        nxt, nkl = figs("any"), klabs("any")
        self.play(*[FadeOut(cur[k]) for k in "abc"], *[FadeOut(kl[k]) for k in "abc"],
                  *[FadeIn(nxt[k]) for k in "abc"], FadeOut(kv), run_time=1.0)
        self.bring_to_front(tri, raC)
        cur, kl = nxt, nkl
        flight("any", ["a", "b"], rt=2.2)
        self.play(*[FadeIn(kl[k]) for k in "abc"], run_time=0.6)

        segs = []
        for key in "abc":
            segs += _poly_segs([S(p) for p in local(key, SQ)])
        segs += _poly_segs([S(A), S(B), S(C)])
        _labels_ok([prem], segs, "A14 premise")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("k·a² + k·b²  =  k·c²", 36)))
        self.hold(2.2)


# ================================================ A12 PYTHAGORAS BY CHORDS

class A12_IntersectingChords(Board):
    """The circle about B through A has radius c. The leg BC lies on a
    diameter ED, which C cuts into CD = c − a and CE = c + a; the leg CA is
    perpendicular to it, so it is half the chord AA' (CA = CA' = b). The
    angle EAD is right (Thales), so the altitude AC cuts triangle EAD into
    two similar right triangles: a quarter-turn about C with enlargement
    b/(c − a) carries CDA onto CAE. Hence (c − a) : b = b : (c + a), the
    intersecting-chords relation b·b = (c − a)(c + a) = c² − a²."""

    def construct(self):
        a = 0.42                                   # the circle has radius c = 1
        b = float(np.sqrt(1 - a * a))
        c = 1.0
        Bm, Cm, Am = np.zeros(2), np.array([a, 0.0]), np.array([a, b])
        Apm = np.array([a, -b])
        Dm, Em = np.array([c, 0.0]), np.array([-c, 0.0])
        check(close(np.linalg.norm(Am - Bm), c) and close(np.linalg.norm(Apm - Bm), c)
              and close(np.linalg.norm(Dm - Bm), c) and close(np.linalg.norm(Em - Bm), c),
              "A, A', D, E on the circle of radius c about B")
        check(abs(np.dot(Am - Cm, Bm - Cm)) < 1e-12, "right angle at C")
        check(close(np.linalg.norm(Dm - Cm), c - a) and close(np.linalg.norm(Em - Cm), c + a),
              "CD = c − a, CE = c + a")
        check(abs(np.dot(Am - Em, Am - Dm)) < 1e-12, "angle EAD is right (Thales)")
        lam = b / (c - a)
        check(abs(lam - (c + a) / b) < 1e-12, "b/(c − a) = (c + a)/b")

        def q90(p):
            """Quarter-turn about C with enlargement lam (math coords)."""
            d = np.asarray(p, float) - Cm
            return Cm + lam * np.array([-d[1], d[0]])
        check(close(q90(Dm), Am) and close(q90(Am), Em),
              "the turned, enlarged CDA is CAE")

        R = 3.25
        O = np.array([-3.0, (SAFE_TOP + SAFE_BOTTOM) / 2, 0.0])

        def S(p):
            return O + R * to3(p)

        B, C, A, Ap, D, E = (S(p) for p in (Bm, Cm, Am, Apm, Dm, Em))

        tri = Polygon(B, C, A, stroke_color=WHITE, stroke_width=3)
        raC = _ra(C, B, A, 0.2)
        la = tag("a", 28).move_to((B + C) / 2 + 0.3 * UP)
        lb = _beside(tag("b", 28), C, A, RIGHT, 0.14, at=0.42)
        lc = _beside(tag("c", 28), B, A, _dir(PI * 0.75), 0.12, at=0.45)
        self.play(Create(tri), Create(raC), FadeIn(la), FadeIn(lb), FadeIn(lc),
                  run_time=1.2)

        # the circle about B through A: radius c
        dB = Dot(B, radius=0.06)
        t0 = float(np.arctan2(b, a))
        circ = Arc(radius=R, start_angle=t0, angle=TAU, arc_center=B,
                   color=GREY_B, stroke_width=3)
        rad = Line(B, A, color=YELLOW_B, stroke_width=4)
        self.play(FadeIn(dB), Create(rad), run_time=0.5)
        self.play(Create(circ),
                  Rotate(rad, angle=TAU, about_point=B), run_time=1.8)
        self.play(FadeOut(rad), run_time=0.3)

        # the diameter through C, and the chord through C perpendicular to it
        diam = Line(E, D, color=GREY_A, stroke_width=3)
        chord = DashedLine(A, Ap, color=WHITE, stroke_width=3, dash_length=0.09)
        raC2 = _ra(C, D, Ap, 0.2)
        dots = VGroup(*[Dot(p, radius=0.06) for p in (D, E, Ap)])
        self.play(Create(diam), FadeIn(dots), run_time=0.8)
        self.bring_to_front(tri, dB)
        dm1 = _dim(C, D, DOWN, off=0.42)
        dm2 = _dim(E, C, DOWN, off=0.42)
        l1 = tag("c − a", 26, ORANGE).next_to(dm1, DOWN, buff=0.12)
        l2 = tag("c + a", 26, TEAL_B).next_to(dm2, DOWN, buff=0.12)
        self.play(Create(dm1), Create(dm2), FadeIn(l1), FadeIn(l2), run_time=0.8)
        lb2 = _beside(tag("b", 28), C, Ap, RIGHT, 0.14, at=0.5)
        self.play(Create(chord), Create(raC2), FadeIn(lb2), run_time=0.8)
        self.hold(0.4)

        # Thales: the angle at A on the diameter ED is right
        EA = Line(E, A, color=GREY_B, stroke_width=2.5)
        AD = Line(A, D, color=GREY_B, stroke_width=2.5)
        raA = _ra(A, E, D, 0.27)
        self.play(Create(EA), Create(AD), Create(raA), run_time=0.9)

        # the two right triangles on either side of the altitude AC
        tS = mk([C, D, A], ORANGE, 0.75)
        tL = mk([C, A, E], TEAL_D, 0.45)
        aD = angle_arc(D, C, A, radius=0.55, color=YELLOW_B, width=4)
        aA = angle_arc(A, C, E, radius=0.62, color=YELLOW_B, width=4)
        self.add(tS, tL)
        self.bring_to_back(tS, tL)
        tS.set_opacity(0)
        tL.set_opacity(0)
        self.play(tS.animate.set_fill(opacity=0.75).set_stroke(opacity=1),
                  tL.animate.set_fill(opacity=0.45).set_stroke(opacity=1),
                  Create(aD), Create(aA), run_time=0.9)
        self.hold(0.4)

        # a quarter-turn about C with enlargement carries CDA onto CAE
        cp = mk([C, D, A], ORANGE, 0.75, stroke_color=YELLOW_B, stroke_width=3)
        arc90 = Arc(radius=0.5, start_angle=0.0, angle=PI / 2, arc_center=C,
                    color=YELLOW_B, stroke_width=5)
        self.play(FadeIn(cp), Create(arc90), run_time=0.5)
        self.play(_spiral(cp, C, PI / 2, lam), run_time=2.2)
        check(_landed(cp, [C, A, E]), "the copy lands on CAE")
        self.play(cp.animate.set_fill(opacity=0).set_stroke(opacity=0),
                  FadeOut(arc90), run_time=0.6)
        self.remove(cp)

        # read the similar triangles
        x0 = 3.65
        r1 = tag("(c − a) : b  =  b : (c + a)", 28).move_to([x0, 2.3, 0.0])
        r2 = tag("b · b  =  (c − a)(c + a)", 28).move_to([x0, 1.15, 0.0])
        r3 = tag("b²  =  c² − a²", 30, YELLOW_B).move_to([x0, 0.0, 0.0])
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(r3, shift=RIGHT * 0.2), run_time=0.8)

        segs = (_arc_segs(B, R, 0.0, TAU, 96)
                + [_seg(E, D), _seg(B, A), _seg(A, Ap), _seg(E, A), _seg(A, D)]
                + [_seg(*l.get_start_and_end()) for l in (dm1[0], dm2[0])]
                + _arc_segs(D, 0.55, PI - 1.0, PI, 8) + _arc_segs(A, 0.62, -PI / 2 - 1.0, -PI / 2, 8))
        _labels_ok([la, lb, lc, lb2, l1, l2, r1, r2, r3], segs, "A12")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("b²  =  (c − a)(c + a)  =  c² − a²    ⟹    a² + b²  =  c²", 30)))
        self.hold(2.2)


# ================================================ A21 PARALLELOGRAM LAW

class A21_ParallelogramLaw(Board):
    """Sides a (the base) and b; the top side sits over the base shifted by
    x, at height h, so b² = x² + h². The heights from the two top corners
    make right triangles on the base line: the long diagonal is the
    hypotenuse of legs a + x and h, the short one of legs a − x and h. In
    the (a + x)-square the two a·x rectangles are spare: moved round the
    (a − x)-square they make a² + x², and what stays is a² + x² as well."""

    def construct(self):
        a, x, h = 4.0, 1.4, 2.0
        b = float(np.hypot(x, h))
        Am, Bm = np.zeros(2), np.array([a, 0.0])
        Cm, Dm = np.array([a + x, h]), np.array([x, h])
        Dfm, Cfm = np.array([x, 0.0]), np.array([a + x, 0.0])     # feet
        d1, d2 = np.linalg.norm(Cm - Am), np.linalg.norm(Dm - Bm)
        check(close(np.linalg.norm(Dm - Am), b) and close(np.linalg.norm(Cm - Bm), b)
              and close(Cm - Dm, Bm - Am), "a parallelogram with sides a, b")
        check(abs(d1 * d1 - ((a + x) ** 2 + h * h)) < 1e-9
              and abs(d2 * d2 - ((a - x) ** 2 + h * h)) < 1e-9
              and abs(b * b - (x * x + h * h)) < 1e-9, "Pythagoras on the heights")
        check(abs(d1 * d1 + d2 * d2 - 2 * (a * a + b * b)) < 1e-9, "parallelogram law")

        F = Frame(-0.35, a + x + 0.35, -0.9, h + 0.5, max_w=6.8, max_h=3.55,
                  centre=(-3.05, 1.98))
        P = F.P
        A, B, C, D, Df, Cf = (P(p) for p in (Am, Bm, Cm, Dm, Dfm, Cfm))

        par = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=3.5)
        la = tag("a", 28).next_to(Line(D, C), UP, buff=0.12)
        lb = _beside(tag("b", 28), A, D, LEFT, 0.12)
        self.play(Create(par), FadeIn(la), FadeIn(lb), run_time=1.2)
        g1 = Line(A, C, color=YELLOW_B, stroke_width=4)
        g2 = Line(B, D, color=YELLOW_B, stroke_width=4)
        ld1 = _beside(tag("d₁", 28, YELLOW_B), A, C, DOWN + RIGHT * 0.2, 0.1, at=0.8)
        ld2 = _beside(tag("d₂", 28, YELLOW_B), B, D, RIGHT, 0.1, at=0.3)
        self.play(Create(g1), Create(g2), FadeIn(ld1), FadeIn(ld2), run_time=1.0)

        # the heights from the top corners, the base line extended
        ext = DashedLine(B, Cf, color=GREY_B, stroke_width=2.5, dash_length=0.08)
        hD = DashedLine(D, Df, color=GREY_B, stroke_width=2.5, dash_length=0.08)
        hC = DashedLine(C, Cf, color=GREY_B, stroke_width=2.5, dash_length=0.08)
        mD, mC = _ra(Df, D, B, 0.16, GREY_B, 2), _ra(Cf, C, A, 0.16, GREY_B, 2)
        lh = _beside(tag("h", 26), C, Cf, RIGHT, 0.12)
        lxD = tag("x", 26).next_to(Line(A, Df), DOWN, buff=0.12)
        lxC = tag("x", 26).next_to(Line(B, Cf), DOWN, buff=0.12)
        self.play(Create(ext), Create(hD), Create(hC), Create(mD), Create(mC),
                  FadeIn(lh), FadeIn(lxD), FadeIn(lxC), run_time=1.2)

        # three right triangles on the base line
        x0r = 3.75
        rows = [tag("d₁²  =  (a + x)² + h²", 28).move_to([x0r, 3.2, 0.0]),
                tag("d₂²  =  (a − x)² + h²", 28).move_to([x0r, 2.4, 0.0]),
                tag("b²  =  x² + h²", 28).move_to([x0r, 1.6, 0.0])]
        tris = [([A, Cf, C], TEAL_D, (A, Cf), "a + x"),
                ([Df, B, D], ORANGE, (Df, B), "a − x"),
                ([A, Df, D], PURPLE_B, (A, Df), "x")]
        for (pts, col, (p, q), s), row in zip(tris, rows):
            t = mk(pts, col, 0.5, stroke_color=col, stroke_width=3)
            dm = _dim(p, q, DOWN, color=col, off=0.62 if s != "x" else 0.62)
            ld = tag(s, 24, col).next_to(dm, DOWN, buff=0.08)
            self.add(t)
            self.bring_to_back(t)
            t.set_opacity(0)
            self.play(t.animate.set_fill(opacity=0.5).set_stroke(opacity=1),
                      Create(dm), FadeIn(ld), run_time=0.7)
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.6)
            self.hold(0.3)
            self.play(FadeOut(t), FadeOut(dm), FadeOut(ld), run_time=0.4)

        segs = (_poly_segs([A, B, C, D]) + [_seg(A, C), _seg(B, D), _seg(B, Cf),
                _seg(D, Df), _seg(C, Cf)])
        _labels_ok([la, lb, ld1, ld2, lh, lxD, lxC] + rows, segs, "A21 figure")

        # (a + x)² and (a − x)²: the two a·x rectangles are spare
        u = 0.47
        O1 = np.array([-6.4, -2.95, 0.0])
        O2 = np.array([-3.05, -2.95, 0.0])

        def R(o, x0, y0, w, hh, col, op=FILL):
            return mk([o + u * np.array([px, py, 0.0]) for px, py in
                       ((x0, y0), (x0 + w, y0), (x0 + w, y0 + hh), (x0, y0 + hh))],
                      col, op, stroke_width=2)

        sqa = R(O1, 0, 0, a, a, BLUE_D)
        sqx = R(O1, a, a, x, x, BLUE_D)
        R1 = R(O1, a, 0, x, a, ORANGE)
        R2 = R(O1, 0, a, a, x, ORANGE)
        sq2 = R(O2, x, x, a - x, a - x, GREEN_D)
        L1 = VGroup(tag("a²", 26).move_to(sqa), tag("x²", 20).move_to(sqx),
                    tag("ax", 20).move_to(R1), tag("ax", 20).move_to(R2))
        L2 = tag("(a − x)²", 22).move_to(sq2)
        cap1 = tag("(a + x)²", 24, GREY_A).next_to(VGroup(sqa, sqx, R1, R2), UP, buff=0.12)
        self.play(FadeIn(VGroup(sqa, sqx, R1, R2)), FadeIn(L1), FadeIn(cap1),
                  FadeIn(sq2), FadeIn(L2), run_time=1.0)
        self.hold(0.4)
        s1 = O2 - O1 - u * np.array([a, 0.0, 0.0])
        s2 = O2 + u * np.array([x, 0.0, 0.0]) - O1 - u * np.array([0.0, a, 0.0])
        self.play(R1.animate.shift(s1), L1[2].animate.shift(s1),
                  R2.animate.shift(s2), L1[3].animate.shift(s2), run_time=1.6)
        region = [O2 + u * np.array([px, py, 0.0]) for px, py in
                  ((0, 0), (a + x, 0), (a + x, x), (a, x), (a, a), (0, a))]
        check(_tiles_exactly([R1.get_vertices(), R2.get_vertices(),
                              sq2.get_vertices()], region),
              "(a − x)² and the two a·x rectangles make a² + x² exactly")
        check(_tiles_exactly([sqa.get_vertices(), sqx.get_vertices(), R1.get_vertices(),
                              R2.get_vertices(), sq2.get_vertices()],
                             [O1, O1 + u * (a + x) * RIGHT,
                              O1 + u * (a + x) * (RIGHT + UP), O1 + u * (a + x) * UP])
              is False, "the pieces have left the (a + x)-square")
        # read both as a² + x²: outlines of the a-square and the x-square
        da = DashedVMobject(Polygon(*[O2 + u * np.array([px, py, 0.0]) for px, py in
                                      ((0, 0), (a, 0), (a, a), (0, a))],
                                    stroke_color=WHITE, stroke_width=3), num_dashes=40)
        dx = DashedVMobject(Polygon(*[O2 + u * np.array([px, py, 0.0]) for px, py in
                                      ((a, 0), (a + x, 0), (a + x, x), (a, x))],
                                    stroke_color=WHITE, stroke_width=3), num_dashes=12)
        L2b = VGroup(_lbl("a²", 26, WHITE, 0.55).move_to(
                         O2 + u * np.array([(a + x) / 2, (a + x) / 2, 0])),
                     _lbl("x²", 20, WHITE, 0.55).move_to(O2 + u * np.array([a + x / 2, x / 2, 0])))
        ghost = DashedVMobject(Polygon(O1, O1 + u * (a + x) * RIGHT,
                                       O1 + u * (a + x) * (RIGHT + UP),
                                       O1 + u * (a + x) * UP, stroke_color=GREY_B,
                                       stroke_width=2), num_dashes=48)
        self.play(FadeOut(L2), FadeOut(L1[2]), FadeOut(L1[3]), Create(ghost),
                  Create(da), Create(dx), FadeIn(L2b), run_time=1.0)
        r4 = tag("(a + x)² + (a − x)²  =  2a² + 2x²", 26).move_to([3.3, -1.0, 0.0])
        r5 = tag("d₁² + d₂²  =  2a² + 2(x² + h²)", 26).move_to([3.3, -2.0, 0.0])
        self.play(FadeIn(r4, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(r5, shift=RIGHT * 0.2), run_time=0.8)
        check(_in_safe(self.mobjects), "everything inside the content area")
        _labels_ok([cap1, r4, r5] + rows, [], "A21 rows")
        self.play(Write(caption("d₁² + d₂²  =  2(a² + b²)", 36)))
        self.hold(2.2)


# ================================================ A25 SQUARING A RECTANGLE

class A25_SquaringRectangle(Board):
    """Euclid II.14. F cuts the diameter UV into p and q; the perpendicular
    at F meets the semicircle at P, FP = h, and the angle UPV is right.
    Turn triangle UFP a quarter-turn about F: U lands at U* on FP (FU* = p)
    and P at P* on FV (FP* = h), and its hypotenuse, turned through 90°
    from UP, is parallel to PV. In triangle FVP the half-square FP*P and
    the half-rectangle FVU* are what is left after removing P*VP and U*VP —
    triangles on the base PV with apexes on one parallel, equal by a shear.
    So ½h² = ½pq: the square on h has the area of the p × q rectangle."""

    def construct(self):
        p, q = 1.3, 3.1
        Um, Vm, Fm = np.array([-p, 0.0]), np.array([q, 0.0]), np.zeros(2)
        Om, r = (Um + Vm) / 2, (p + q) / 2
        h = float(np.sqrt(r * r - Om[0] ** 2))
        Pm = np.array([0.0, h])
        Usm, Psm = _rot2(Um, -PI / 2), _rot2(Pm, -PI / 2)     # the turned UFP
        check(close(np.linalg.norm(Pm - Om), r), "P on the semicircle")
        check(abs(np.dot(Pm - Um, Vm - Pm)) < 1e-12, "angle UPV is right")
        check(close(Usm, (0.0, p)) and close(Psm, (h, 0.0)) and 0 < p < h < q,
              "U* on FP at height p, P* on FV at distance h")
        check(abs(_cross2(Psm - Usm, Vm - Pm)) < 1e-12, "U*P* parallel to PV")
        check(close(abs(area([Fm, Psm, Pm])), h * h / 2)
              and close(abs(area([Fm, Vm, Usm])), p * q / 2), "the two halves")
        check(close(abs(area([Psm, Vm, Pm])), abs(area([Usm, Vm, Pm]))),
              "P*VP and U*VP: equal (same base PV, apexes on a parallel)")
        check(close(h * h, p * q), "h² = pq")

        Fr = Frame(-p - 0.4, q + 0.35, -0.95, r + 0.2, max_w=9.3,
                   centre=(-1.55, (SAFE_TOP + SAFE_BOTTOM) / 2))
        S = Fr.P
        U, V, F, P, O = S(Um), S(Vm), S(Fm), S(Pm), S(Om)
        Us, Ps = S(Usm), S(Psm)
        k = Fr.k

        # ---- the semicircle, the altitude, the right angle at P
        diam = Line(U, V, color=GREY_A, stroke_width=3)
        semi = Arc(radius=r * k, start_angle=0, angle=PI, arc_center=O,
                   color=GREY_B, stroke_width=3)
        FP = Line(F, P, color=WHITE, stroke_width=4)
        raF = _ra(F, V, P, 0.18)
        dp = _dim(U, F, DOWN, off=0.32)
        dq = _dim(F, V, DOWN, off=0.32)
        lp = tag("p", 28).next_to(dp, DOWN, buff=0.1)
        lq = tag("q", 28).next_to(dq, DOWN, buff=0.1)
        lh = _beside(tag("h", 28), F, P, LEFT, 0.14)
        self.play(Create(diam), Create(semi), run_time=1.0)
        self.play(Create(FP), Create(raF), Create(dp), Create(dq),
                  FadeIn(lp), FadeIn(lq), FadeIn(lh), run_time=1.0)
        UP_, PV_ = Line(U, P, color=WHITE, stroke_width=3), Line(P, V, color=WHITE,
                                                                 stroke_width=3)
        raP = _ra(P, U, V, 0.24)
        self.play(Create(UP_), Create(PV_), Create(raP), run_time=0.8)

        # ---- the left triangle, a quarter-turn about F
        tU = mk([F, U, P], BLUE_D, 0.55)
        self.add(tU)
        self.bring_to_back(tU)
        tU.set_opacity(0)
        self.play(tU.animate.set_fill(opacity=0.55).set_stroke(opacity=1), run_time=0.5)
        cp = mk([F, U, P], BLUE_D, 0.75, stroke_color=BLUE_A, stroke_width=3)
        arc = Arc(radius=0.55, start_angle=PI, angle=-PI / 2, arc_center=F,
                  color=YELLOW_B, stroke_width=5)
        deg = tag("90°", 22, YELLOW_B).move_to(F + 0.95 * _dir(0.75 * PI) + 0.05 * UP)
        self.play(FadeIn(cp), Create(arc), FadeIn(deg), run_time=0.5)
        self.play(Rotate(cp, angle=-PI / 2, about_point=F), run_time=1.8)
        check(_landed(cp, [F, Us, Ps]), "the turned triangle is F U* P*")
        self.play(FadeOut(arc), FadeOut(deg), run_time=0.3)
        par = VGroup(_chevron(Us, Ps, YELLOW_B), _chevron(P, V, YELLOW_B))
        lps = _beside(tag("p", 26, BLUE_A), F, Us, RIGHT, 0.14)
        self.play(FadeIn(par), FadeIn(lps), run_time=0.7)
        self.hold(0.4)

        # ---- the h-square on FP and the p × q rectangle on FV, FU*
        Sq = [F, Ps, Ps + (P - F), P]
        Rc = [F, V, V + (Us - F), Us]
        sq = Polygon(*Sq, stroke_color=TEAL_B, stroke_width=4)
        rc = Polygon(*Rc, stroke_color=YELLOW_B, stroke_width=4)
        lsq = tag("h²", 30, TEAL_B).move_to(S((0.5 * h, 0.5 * (p + h))) + 0.0 * UP)
        lrc = tag("pq", 30, YELLOW_B).move_to(S((0.5 * (h + q), 0.5 * p)))
        check(_box_inside(lsq, Sq) and not _box_inside(lsq, Rc, -0.05)
              and _box_inside(lrc, Rc) and not _pip(lrc.get_center(), Sq),
              "h² sits in the square only, pq in the rectangle only")
        UsPs = Line(Us, Ps, color=BLUE_A, stroke_width=2.5)
        self.add(UsPs)
        self.play(FadeOut(cp), tU.animate.set_fill(opacity=0.25), Create(sq),
                  Create(rc), FadeIn(lsq), FadeIn(lrc), run_time=1.1)
        self.bring_to_front(UsPs, par)
        self.hold(0.5)

        # ---- in triangle FVP: the half-square FP*P and the triangle P*VP
        half_sq = mk([F, Ps, P], TEAL_D, 0.7)
        rest = mk([Ps, V, P], ORANGE, 0.75)
        diag = DashedLine(Ps, P, color=TEAL_B, stroke_width=2.5, dash_length=0.08)
        self.add(half_sq, rest)
        self.bring_to_back(half_sq, rest, tU)
        half_sq.set_opacity(0)
        rest.set_opacity(0)
        self.play(half_sq.animate.set_fill(opacity=0.7).set_stroke(opacity=1),
                  rest.animate.set_fill(opacity=0.75).set_stroke(opacity=1),
                  run_time=0.8)
        self.hold(0.5)
        # the shear: base PV kept, apex P* slides along the parallel to U*
        base = Line(P, V, color=YELLOW_B, stroke_width=8)
        rail = DashedLine(Ps + 0.25 * (Ps - Us), Us + 0.25 * (Us - Ps), color=YELLOW_B,
                          stroke_width=2.5, dash_length=0.09)
        self.play(half_sq.animate.set_fill(opacity=0.0).set_stroke(opacity=0.0),
                  Create(diag), Create(base), Create(rail), run_time=0.6)
        self.bring_to_front(rest)
        self.play(Transform(rest, mk([Us, V, P], ORANGE, 0.75)), run_time=1.8)
        check(_landed(rest, [Us, V, P]), "the sheared triangle is U*VP")
        half_rc = mk([F, V, Us], YELLOW_E, 0.7)
        self.add(half_rc)
        self.bring_to_back(half_rc, tU)
        half_rc.set_opacity(0)
        self.play(half_rc.animate.set_fill(opacity=0.7).set_stroke(opacity=1),
                  FadeOut(base), FadeOut(rail), run_time=0.8)
        self.bring_to_front(sq, rc, UsPs, lsq, lrc, raF, par)

        r1 = tag("½ h²  =  ½ pq", 30).move_to([5.0, 2.2, 0.0])
        r2 = tag("h²  =  pq", 32, YELLOW_B).move_to([5.0, 1.2, 0.0])
        self.play(FadeIn(r1, shift=LEFT * 0.2), run_time=0.7)
        self.play(FadeIn(r2, shift=LEFT * 0.2), run_time=0.7)

        segs = ([_seg(U, V), _seg(F, P), _seg(U, P), _seg(P, V), _seg(Us, Ps)]
                + _arc_segs(O, r * k, 0.0, PI, 64) + _poly_segs(Sq) + _poly_segs(Rc)
                + [_seg(*d[0].get_start_and_end()) for d in (dp, dq)])
        _labels_ok([lp, lq, lh, lps, lsq, lrc, r1, r2], segs, "A25")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("h²  =  pq", 36)))
        self.hold(2.2)


# ================================================ A13 PAPPUS'S AREA THEOREM

class A13_PappusArea(Board):
    """Parallelograms P on CA and Q on CB, outside the triangle, any shape.
    Their outer sides meet at H. R on AB has its sides AL, BM parallel and
    equal to HC. Shear P keeping CA: its outer side slides along its own
    line until that side's end reaches H, so P gets the sides CA and CH;
    likewise Q. Slide both down by HC: they now hang from AB with H at C.
    Shear again, keeping AL (and BM): the common side slides along the line
    HC until C reaches K on AB. P and Q now fill R exactly: P + Q = R."""

    def construct(self):
        Am, Bm, Cm = np.array([0.0, 0.0]), np.array([5.2, 0.0]), np.array([1.6, 2.0])
        u1, u2 = np.array([-1.3, 0.9]), np.array([1.2, 0.8])
        Pm = [Cm, Am, Am + u1, Cm + u1]
        Qm = [Cm, Bm, Bm + u2, Cm + u2]
        n_ca = np.array([(Am - Cm)[1], -(Am - Cm)[0]])
        n_cb = np.array([(Bm - Cm)[1], -(Bm - Cm)[0]])
        check(np.dot(u1, n_ca) * np.dot(Bm - Cm, n_ca) < 0
              and np.dot(u2, n_cb) * np.dot(Am - Cm, n_cb) < 0,
              "P and Q stand outside the triangle")
        # H: where the outer sides' lines meet
        Mx = np.array([Am - Cm, -(Bm - Cm)]).T
        s, t = np.linalg.solve(Mx, (Cm + u2) - (Cm + u1))
        Hm = Cm + u1 + s * (Am - Cm)
        check(close(Hm, Cm + u2 + t * (Bm - Cm)), "H on both outer lines")
        w = Hm - Cm
        Km = Cm + (-Cm[1] / w[1]) * w                    # HC meets AB
        check(0.2 * Bm[0] < Km[0] < 0.8 * Bm[0], "K well inside AB")
        Lm, Mm = Am - w, Bm - w
        Rm = [Am, Bm, Mm, Lm]
        P1 = [Cm, Am, Am + w, Hm]                        # after the first shear
        Q1 = [Cm, Bm, Bm + w, Hm]
        P2 = [p - w for p in P1]                         # after the slide
        Q2 = [p - w for p in Q1]
        P3 = [Km - w, Am - w, Am, Km]                    # after the second shear
        Q3 = [Km - w, Bm - w, Bm, Km]
        # every shear keeps its base and slides the opposite side on its line
        check(abs(_cross2(Am + w - (Am + u1), Am - Cm)) < 1e-9
              and abs(_cross2(Hm - (Cm + u1), Am - Cm)) < 1e-9, "shear of P along DE")
        check(abs(_cross2(Bm + w - (Bm + u2), Bm - Cm)) < 1e-9
              and abs(_cross2(Hm - (Cm + u2), Bm - Cm)) < 1e-9, "shear of Q along FG")
        check(close(P2[3], Cm) and close(Q2[3], Cm), "the slide carries H to C")
        check(abs(_cross2(Km - Cm, w)) < 1e-9, "the second shear slides along HC")
        for X, Y in ((Pm, P1), (P1, P2), (P2, P3), (Qm, Q1), (Q1, Q2), (Q2, Q3)):
            check(close(abs(area(X)), abs(area(Y))), "shears and slides keep the area")
        check(_tiles_exactly([P3, Q3], Rm), "P and Q fill R exactly")
        check(close(abs(area(Pm)) + abs(area(Qm)), abs(area(Rm))), "P + Q = R")

        pts = np.array(Pm + Qm + Rm + [Hm])
        F = Frame(pts[:, 0].min() - 0.45, pts[:, 0].max() + 0.6,
                  pts[:, 1].min() - 0.15, pts[:, 1].max() + 0.3)
        S = F.P
        A, B, C, H, K, L, M = (S(p) for p in (Am, Bm, Cm, Hm, Km, Lm, Mm))
        W = S(w) - S((0.0, 0.0))

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        self.play(Create(tri), run_time=0.8)
        cP, cQ = TEAL_D, ORANGE
        mP = mk([S(p) for p in Pm], cP)
        mQ = mk([S(p) for p in Qm], cQ)
        lP = tag("P", 32).move_to(mP.get_center_of_mass())
        lQ = tag("Q", 32).move_to(mQ.get_center_of_mass())
        self.play(FadeIn(mP), FadeIn(mQ), FadeIn(lP), FadeIn(lQ), run_time=1.0)
        self.bring_to_front(tri)

        # H: the outer sides, extended, meet
        e1 = DashedLine(S(Cm + u1), H, color=GREY_B, stroke_width=2.5, dash_length=0.09)
        e2 = DashedLine(S(Cm + u2), H, color=GREY_B, stroke_width=2.5, dash_length=0.09)
        dH = Dot(H, radius=0.07, color=YELLOW_B)
        lH = tag("H", 28, YELLOW_B).move_to(H + 0.42 * _dir(PI * 5 / 6))
        self.play(Create(e1), Create(e2), run_time=0.8)
        self.play(FadeIn(dH), FadeIn(lH), run_time=0.4)
        # R on AB: sides parallel and equal to HC
        HC = Line(H, C, color=YELLOW_B, stroke_width=4)
        Rout = DashedVMobject(Polygon(A, B, M, L, stroke_color=YELLOW_B, stroke_width=3),
                              num_dashes=70)
        eq = VGroup(_ticks(H, C, 2, YELLOW_B), _ticks(A, L, 2, YELLOW_B),
                    _ticks(B, M, 2, YELLOW_B))
        self.play(Create(HC), run_time=0.6)
        self.play(TransformFromCopy(HC, Line(A, L, color=YELLOW_B, stroke_width=4)),
                  TransformFromCopy(HC, Line(B, M, color=YELLOW_B, stroke_width=4)),
                  run_time=1.0)
        self.play(Create(Rout), FadeIn(eq), run_time=0.8)
        lR = _beside(tag("R", 32, YELLOW_B), B, M, RIGHT, 0.22, at=0.25)
        self.play(FadeIn(lR), run_time=0.4)
        self.hold(0.3)

        def shear(mob, lab, target, base, rail, col, others, rt=1.6):
            """base = (fixed end at C, far end); rail = (outer corner, H)."""
            b0, b1 = base
            r0, r1 = rail
            edge = Line(b0, b1, color=col, stroke_width=8)
            g1 = DashedLine(b0 + 0.12 * (b0 - b1), b1 + 0.04 * (b1 - b0), color=col,
                            stroke_width=2, dash_length=0.09)
            g2 = DashedLine(r0 + 0.15 * (r0 - r1), r1, color=col, stroke_width=2,
                            dash_length=0.09)
            _labels_ok(others, [_seg(g1.get_start(), g1.get_end()),
                                _seg(g2.get_start(), g2.get_end())], "A13 guides")
            self.play(Create(edge), Create(g1), Create(g2), run_time=0.5)
            tgt = mk(target, col)
            self.play(Transform(mob, tgt), lab.animate.move_to(tgt.get_center_of_mass()),
                      run_time=rt)
            check(_landed(mob, target), "shear lands")
            self.play(FadeOut(edge), FadeOut(g1), FadeOut(g2), run_time=0.3)

        # first shears: keep CA (CB); the outer side slides to H
        ghosts = [DashedVMobject(Polygon(*[S(p) for p in X], stroke_color=col,
                                         stroke_width=2.5), num_dashes=44)
                  for X, col in ((Pm, TEAL_B), (Qm, ORANGE))]
        self.add(*ghosts)
        self.bring_to_front(mP, lP)
        shear(mP, lP, [S(p) for p in P1], (C, A), (S(Am + u1), H), cP,
              [lH, lR, lP, lQ])
        self.bring_to_front(mQ, lQ)
        shear(mQ, lQ, [S(p) for p in Q1], (C, B), (S(Bm + u2), H), cQ,
              [lH, lR, lP, lQ])
        self.play(FadeOut(e1), FadeOut(e2), run_time=0.3)

        # the slide: down along HC by its own length
        arr = Arrow(H + 0.5 * RIGHT, C + 0.5 * RIGHT, buff=0.0, color=YELLOW_B,
                    stroke_width=5, max_tip_length_to_length_ratio=0.15)
        self.play(GrowArrow(arr), run_time=0.5)
        self.play(VGroup(mP, lP).animate.shift(-W), VGroup(mQ, lQ).animate.shift(-W),
                  run_time=1.6)
        check(_landed(mP, [S(p) for p in P2]) and _landed(mQ, [S(p) for p in Q2]),
              "the slide lands")
        self.play(FadeOut(arr), run_time=0.3)
        self.bring_to_front(tri)

        # second shears: keep AL (BM); the common side slides along HC to K
        rail = guide(H, K - W, YELLOW_B, extend=0.0)
        bases = VGroup(Line(A, L, color=cP, stroke_width=8),
                       Line(B, M, color=cQ, stroke_width=8))
        self.play(Create(rail), Create(bases), run_time=0.5)
        tP, tQ = mk([S(p) for p in P3], cP), mk([S(p) for p in Q3], cQ)
        self.play(Transform(mP, tP), Transform(mQ, tQ),
                  lP.animate.move_to(tP.get_center_of_mass()),
                  lQ.animate.move_to(tQ.get_center_of_mass()), run_time=1.8)
        check(_landed(mP, [S(p) for p in P3]) and _landed(mQ, [S(p) for p in Q3]),
              "the second shears land")
        self.play(FadeOut(rail), FadeOut(bases), run_time=0.3)
        Rb = Polygon(A, B, M, L, stroke_color=YELLOW_B, stroke_width=6)
        self.play(FadeOut(Rout), Create(Rb), run_time=0.8)
        self.bring_to_front(lR)

        segs = (_poly_segs([A, B, C]) + _poly_segs([A, B, M, L]) + [_seg(H, C)]
                + _poly_segs([S(p) for p in P3]) + _poly_segs([S(p) for p in Q3]))
        _labels_ok([lH, lR], segs, "A13")
        check(_box_inside(lP, [S(p) for p in P3]) and _box_inside(lQ, [S(p) for p in Q3]),
              "P and Q labels inside their pieces")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("P + Q  =  R", 36)))
        self.hold(2.2)


# ================================================ A19 LAW OF COSINES BY CHORDS

class A19_CosinesChords(Board):
    """The circle about C through B has radius a. Read from A along the side
    b: the line through the centre meets it at D and E, AD = b − a and
    AE = b + a. Along the side c: the line meets it again at B', and the
    foot M of the perpendicular from C halves the chord B'B, each half
    a·cos B, so AB' = c − 2a·cos B. Triangle ADB', turned over about the
    bisector of the angle at A and enlarged by c/(b − a), lands on ABE (its
    angle at B' equals the angle at E: D, B', B, E lie on one circle). So
    (b − a)(b + a) = c(c − 2a·cos B)."""

    def construct(self):
        a, c0, beta = 1.6, 4.2, 52 * DEGREES
        b = float(np.sqrt(a * a + c0 * c0 - 2 * a * c0 * np.cos(beta)))
        gam = float(np.arccos((a * a + b * b - c0 * c0) / (2 * a * b)))
        Cm, Am = np.zeros(2), np.array([-b, 0.0])
        Bm = a * np.array([np.cos(PI - gam), np.sin(PI - gam)])
        c = float(np.linalg.norm(Bm - Am))
        cosB = (a * a + c * c - b * b) / (2 * a * c)
        check(abs(np.dot(Am - Bm, Cm - Bm) / (c * a) - cosB) < 1e-12, "angle B")
        check(b > a and 0 < cosB < 1 and abs(c - c0) < 1e-9
              and abs(cosB - np.cos(beta)) < 1e-12, "b > a, B acute")
        uAB = (Bm - Am) / c
        Bpm = Am + (c - 2 * a * cosB) * uAB
        Mm = Am + (c - a * cosB) * uAB
        Dm, Em = np.array([-a, 0.0]), np.array([a, 0.0])
        check(close(np.linalg.norm(Bpm - Cm), a) and c - 2 * a * cosB > 0,
              "B' on the circle, between A and B")
        check(abs(np.dot(Mm - Cm, uAB)) < 1e-12 and close((Bpm + Bm) / 2, Mm),
              "CM ⊥ AB and M halves B'B")
        check(close(np.linalg.norm(Bm - Mm), a * cosB), "MB = a·cos B")
        check(close((b - a) * (b + a), c * (c - 2 * a * cosB)), "power of A")
        uAE = np.array([1.0, 0.0])
        bis = (uAE + uAB) / np.linalg.norm(uAE + uAB)
        kk = c / (b - a)

        def flip_enlarge(p):
            return Am + kk * (_refl(p, Am, bis) - Am)
        check(close(flip_enlarge(Dm), Bm) and close(flip_enlarge(Bpm), Em),
              "ADB' turned over and enlarged by c/(b − a) is ABE")
        check(abs(_ang(Bpm, Am, Dm) - _ang(Em, Am, Bm)) < 1e-9,
              "angle AB'D = angle AEB")

        F = Frame(-b - 0.75, a + 0.3, -a - 0.62, Bm[1] + 1.5, max_w=8.2,
                  centre=(-2.15, (SAFE_TOP + SAFE_BOTTOM) / 2))
        S = F.P
        A, B, C, Bp, M, D, E = (S(p) for p in (Am, Bm, Cm, Bpm, Mm, Dm, Em))
        R = a * F.k
        nAB = to3(np.array([-uAB[1], uAB[0]]))          # left normal of AB

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        la = _beside(tag("a", 28), C, B, RIGHT, 0.12)
        dc = _dim(A, B, nAB, color=GREY_A, off=1.62)
        lc = _beside(tag("c", 28), A + 1.62 * nAB, B + 1.62 * nAB, nAB, 0.12)
        db = _dim(A, C, DOWN, color=GREY_A, off=0.95)
        lb = tag("b", 28).next_to(db, DOWN, buff=0.12)
        aB = angle_arc(B, A, C, radius=0.45, color=YELLOW_B, width=4)
        lB = tag("B", 24, YELLOW_B).move_to(B + 0.72 * angle_mid_dir(B, A, C))
        self.play(Create(tri), FadeIn(la), Create(dc), FadeIn(lc), Create(db),
                  FadeIn(lb), run_time=1.1)
        self.play(Create(aB), FadeIn(lB), run_time=0.5)

        # the circle about C through B
        dC = Dot(C, radius=0.06)
        t0 = float(np.arctan2(*(Bm - Cm)[::-1]))
        circ = Arc(radius=R, start_angle=t0, angle=TAU, arc_center=C,
                   color=GREY_B, stroke_width=3)
        rad = Line(C, B, color=YELLOW_B, stroke_width=4)
        self.play(FadeIn(dC), Create(rad), run_time=0.4)
        self.play(Create(circ), Rotate(rad, angle=TAU, about_point=C), run_time=1.6)
        self.play(FadeOut(rad), run_time=0.3)
        self.bring_to_front(tri, dC)

        # along b: the line through the centre, AD = b − a, AE = b + a
        CE = Line(C, E, color=WHITE, stroke_width=3)
        dots = VGroup(Dot(D, radius=0.06), Dot(E, radius=0.06))
        d1 = _dim(A, D, DOWN, color=ORANGE, off=0.45)
        d2 = _dim(A, E, DOWN, color=TEAL_B, off=R + 0.22)
        l1 = tag("b − a", 24, ORANGE).next_to(d1, DOWN, buff=0.12)
        l2 = tag("b + a", 24, TEAL_B).next_to(d2, DOWN, buff=0.12)
        self.play(Create(CE), FadeIn(dots), run_time=0.6)
        self.play(Create(d1), Create(d2), FadeIn(l1), FadeIn(l2), run_time=0.8)

        # along c: B' and the halves of the chord B'B
        dBp = Dot(Bp, radius=0.06)
        CM = DashedLine(C, M, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        raM = _ra(M, B, C, 0.16, GREY_A, 2)
        tk = VGroup(_ticks(Bp, M, 1, GREY_A), _ticks(M, B, 1, GREY_A))
        lcos = _beside(tag("a·cos B", 22), Bp, M, -nAB, 0.1, at=0.5)
        self.play(FadeIn(dBp), Create(CM), Create(raM), run_time=0.7)
        self.play(FadeIn(tk), FadeIn(lcos), run_time=0.6)
        d3 = _dim(A, Bp, nAB, color=ORANGE, off=0.45)
        l3 = _beside(tag("c − 2a·cos B", 22, ORANGE), A + 0.45 * nAB, Bp + 0.45 * nAB,
                     nAB, 0.12)
        self.play(Create(d3), FadeIn(l3), run_time=0.8)
        self.hold(0.4)

        # the two triangles: ADB' and ABE, with equal angles at B' and E
        DBp = Line(D, Bp, color=WHITE, stroke_width=2.5)
        EB = Line(E, B, color=WHITE, stroke_width=2.5)
        tS = mk([A, D, Bp], ORANGE, 0.8)
        tL = mk([A, B, E], TEAL_D, 0.4)
        angBp = angle_arc(Bp, A, D, radius=0.3, color=YELLOW_B, width=4)
        angE = angle_arc(E, A, B, radius=0.55, color=YELLOW_B, width=4)
        self.add(tS, tL)
        self.bring_to_back(tS, tL)
        tS.set_opacity(0)
        tL.set_opacity(0)
        self.play(Create(DBp), Create(EB),
                  tS.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                  tL.animate.set_fill(opacity=0.4).set_stroke(opacity=1), run_time=0.9)
        self.play(Create(angBp), Create(angE), run_time=0.6)
        self.hold(0.4)

        # turn ADB' over about the bisector at A, then enlarge it by c/(b − a)
        cp = mk([A, D, Bp], ORANGE, 0.85, stroke_color=YELLOW_B, stroke_width=3)
        ax = DashedLine(A, A + 2.4 * to3(bis), color=YELLOW_B, stroke_width=2,
                        dash_length=0.08)
        self.play(FadeIn(cp), Create(ax), run_time=0.5)
        self.play(Rotate(cp, angle=PI, axis=to3(bis), about_point=A), run_time=1.3)
        tagk = tag("× c/(b − a)", 26, YELLOW_B).move_to([4.1, 2.9, 0.0])
        self.play(cp.animate.scale(kk, about_point=A), FadeIn(tagk), run_time=1.5)
        check(_landed(cp, [A, B, E]), "the turned, enlarged copy is ABE")
        self.play(cp.animate.set_fill(opacity=0).set_stroke(opacity=0), FadeOut(ax),
                  FadeOut(tagk), run_time=0.6)
        self.remove(cp)

        x0 = 3.85
        r1 = tag("(b − a)(b + a) = c (c − 2a·cos B)", 22).move_to([x0, 1.9, 0.0])
        r2 = tag("b² − a²  =  c² − 2ac·cos B", 26, YELLOW_B).move_to([x0, 0.9, 0.0])
        self.play(FadeIn(r1, shift=LEFT * 0.2), run_time=0.7)
        self.play(FadeIn(r2, shift=LEFT * 0.2), run_time=0.7)

        segs = (_arc_segs(C, R, 0.0, TAU, 96)
                + [_seg(A, E), _seg(A, B), _seg(B, C), _seg(C, M), _seg(D, Bp), _seg(E, B)]
                + [_seg(*d[0].get_start_and_end()) for d in (d1, d2, d3, dc, db)]
                + _angle_segs(B, A, C, 0.45) + _angle_segs(E, A, B, 0.55)
                + _angle_segs(Bp, A, D, 0.3))
        _labels_ok([la, lb, lc, lB, l1, l2, lcos, l3, r1, r2], segs, "A19")
        _labels_ok([tagk], segs, "A19 factor")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("b²  =  a² + c² − 2ac·cos B", 36)))
        self.hold(2.2)


# ================================================ A20 LAW OF COSINES, OBTUSE

class A20_CosinesObtuse(Board):
    """Euclid II.12. The angle at C is obtuse, so the altitude from A falls
    outside, at D on BC extended: CD = x is the projection of b, AD = h.
    The right triangles ABD and ACD share the leg h: c² = (a + x)² + h² and
    b² = x² + h². The square on BD = a + x is a², x² and two rectangles
    a·x; the x² corner is the square on CD, which with h² makes b². So c²
    exceeds a² + b² by the two rectangles of a by the projection of b."""

    def construct(self):
        a, x, h = 3.0, 1.1, 2.0
        b, c = float(np.hypot(x, h)), float(np.hypot(a + x, h))
        Bm, Cm = np.zeros(2), np.array([a, 0.0])
        Dm, Am = np.array([a + x, 0.0]), np.array([a + x, h])
        cosC = np.dot(Am - Cm, Bm - Cm) / (b * a)
        check(cosC < 0 and close(x, -b * cosC), "C obtuse, CD = x = −b·cos C")
        check(abs(np.dot(Am - Dm, Dm - Bm)) < 1e-12, "AD ⊥ BD: the altitude falls outside")
        s = a + x
        Big = [Bm, Dm, Dm + (0, -s), Bm + (0, -s)]
        pA2 = [(0, -x), (a, -x), (a, -s), (0, -s)]
        pTop = [(0, 0), (a, 0), (a, -x), (0, -x)]
        pX2 = [(a, 0), (s, 0), (s, -x), (a, -x)]
        pRgt = [(a, -x), (s, -x), (s, -s), (a, -s)]
        pH2 = [Dm, Dm + (h, 0), Am + (h, 0), Am]
        check(_tiles_exactly([pA2, pTop, pX2, pRgt], Big), "(a + x)² = a² + 2a·x + x²")
        check(close(c * c, s * s + h * h) and close(b * b, x * x + h * h),
              "Pythagoras in ABD and ACD")
        check(close(c * c, a * a + b * b + 2 * a * x)
              and close(c * c, a * a + b * b - 2 * a * b * cosC), "law of cosines")

        F = Frame(-0.5, s + h + 0.35, -s - 0.15, h + 0.6, max_w=7.0,
                  centre=(-3.05, (SAFE_TOP + SAFE_BOTTOM) / 2))
        S = F.P
        A, B, C, D = (S(p) for p in (Am, Bm, Cm, Dm))

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        la = tag("a", 28).move_to(S((a / 2, 0.0)) + 0.32 * UP)
        lb = _beside(tag("b", 28), C, A, LEFT + UP * 0.3, 0.12, at=0.55)
        lc = _beside(tag("c", 28), B, A, LEFT + UP, 0.12)
        aC = angle_arc(C, B, A, radius=0.45, color=YELLOW_B, width=4)
        lC = tag("C", 24, YELLOW_B).move_to(C + 0.74 * angle_mid_dir(C, B, A))
        self.play(Create(tri), FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=1.1)
        self.play(Create(aC), FadeIn(lC), run_time=0.5)

        # the altitude from A falls outside, on BC extended beyond C
        ext = DashedLine(C, D, color=GREY_B, stroke_width=3, dash_length=0.09)
        alt = DashedLine(A, D, color=GREY_B, stroke_width=3, dash_length=0.09)
        raD = _ra(D, A, C, 0.18, GREY_B, 2)
        lx = tag("x", 26).move_to(S((a + x / 2, 0.0)) + 0.27 * UP)
        lh = _beside(tag("h", 26), D, A, LEFT, 0.12, at=0.4)
        self.play(Create(ext), Create(alt), Create(raD), FadeIn(lx), FadeIn(lh),
                  run_time=1.0)
        self.hold(0.3)

        # the right triangle ABD and the squares on its legs
        x0 = 3.55
        rows = [tag("c²  =  (a + x)² + h²", 28).move_to([x0, 3.0, 0.0]),
                tag("b²  =  x² + h²", 28).move_to([x0, 2.1, 0.0]),
                tag("(a + x)²  =  a² + 2a·x + x²", 28).move_to([x0, 1.2, 0.0]),
                tag("c²  =  a² + b² + 2a·x", 30, YELLOW_B).move_to([x0, 0.1, 0.0]),
                tag("x  =  −b·cos C", 28).move_to([x0, -0.8, 0.0])]
        hiABD = mk([A, B, D], BLUE_E, 0.35, stroke_color=BLUE_B, stroke_width=4)
        bigsq = Polygon(*[S(p) for p in Big], stroke_color=WHITE, stroke_width=3)
        hsq = Polygon(*[S(p) for p in pH2], stroke_color=WHITE, stroke_width=3)
        l_s = tag("(a + x)²", 30).move_to(S((s / 2, -s / 2)))
        l_h = tag("h²", 30).move_to(S((s + h / 2, h / 2)))
        self.add(hiABD)
        self.bring_to_back(hiABD)
        hiABD.set_opacity(0)
        self.play(hiABD.animate.set_fill(opacity=0.35).set_stroke(opacity=1),
                  Create(bigsq), Create(hsq), FadeIn(l_s), FadeIn(l_h), run_time=1.2)
        self.play(FadeIn(rows[0], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.4)

        # the right triangle ACD: the squares on its legs are x² and h²
        hiACD = mk([A, C, D], TEAL_E, 0.45, stroke_color=TEAL_B, stroke_width=4)
        mX2 = mk([S(p) for p in pX2], TEAL_D, 0.8)
        mH2 = mk([S(p) for p in pH2], TEAL_D, 0.8)
        lX2 = tag("x²", 24).move_to(mX2)
        self.add(hiACD, mX2, mH2)
        self.bring_to_back(hiACD, mX2, mH2)
        for m in (hiACD, mX2, mH2):
            m.set_opacity(0)
        self.play(hiABD.animate.set_fill(opacity=0).set_stroke(opacity=0),
                  hiACD.animate.set_fill(opacity=0.45).set_stroke(opacity=1),
                  mX2.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                  mH2.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                  FadeOut(l_s), FadeIn(lX2), run_time=1.0)
        self.play(FadeIn(rows[1], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.4)

        # the rest of the (a + x)-square: a² and two rectangles a·x
        mA2 = mk([S(p) for p in pA2], BLUE_D, 0.8)
        mT = mk([S(p) for p in pTop], ORANGE, 0.85)
        mR = mk([S(p) for p in pRgt], ORANGE, 0.85)
        lA2 = tag("a²", 32).move_to(mA2)
        lT = tag("a·x", 24).move_to(mT)
        lRr = tag("a·x", 24).move_to(mR)
        self.add(mA2, mT, mR)
        self.bring_to_back(mA2, mT, mR)
        for m in (mA2, mT, mR):
            m.set_opacity(0)
        self.play(hiACD.animate.set_fill(opacity=0).set_stroke(opacity=0),
                  mA2.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                  mT.animate.set_fill(opacity=0.85).set_stroke(opacity=1),
                  mR.animate.set_fill(opacity=0.85).set_stroke(opacity=1),
                  FadeIn(lA2), FadeIn(lT), FadeIn(lRr), run_time=1.0)
        self.play(FadeIn(rows[2], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.3)
        self.play(FadeIn(rows[3], shift=LEFT * 0.2), run_time=0.7)
        # x is the projection of b: the shadow of CA on the line BC
        sh = Line(C, A, color=YELLOW_B, stroke_width=7)
        self.play(Create(sh), run_time=0.4)
        self.play(sh.animate.put_start_and_end_on(C, D), run_time=0.9)
        self.play(FadeIn(rows[4], shift=LEFT * 0.2), FadeOut(sh), run_time=0.7)
        self.remove(hiABD, hiACD)
        self.bring_to_front(tri, ext, alt, raD, la, lb, lc, aC, lC, lx, lh)

        segs = (_poly_segs([A, B, C]) + [_seg(C, D), _seg(A, D)] + _poly_segs([S(p) for p in Big])
                + _poly_segs([S(p) for p in pH2]) + _poly_segs([S(p) for p in pTop])
                + _poly_segs([S(p) for p in pRgt]) + _angle_segs(C, B, A, 0.45))
        _labels_ok([la, lb, lc, lC, lx, lh] + rows, segs, "A20")
        for lab, P_ in ((lA2, pA2), (lT, pTop), (lRr, pRgt), (lX2, pX2), (l_h, pH2)):
            check(_box_inside(lab, [S(p) for p in P_]), "piece labels inside their pieces")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("c²  =  a² + b² − 2ab·cos C", 36)))
        self.hold(2.2)


# ================================================ G27 SINE OF A SUM BY AREA

class G27_SineSumArea(Board):
    """One triangle, two base–height readings. Its altitude 1 splits the
    apex angle into A and B, so the base is tan A + tan B and the sides are
    sec A and sec B. A rigid copy, turned until the side sec B lies flat,
    has height h = sec A·sin(A + B) over it (the side sec A makes the angle
    A + B with that base). Same triangle, same area:
    ½(tan A + tan B)·1 = ½ sec B · sec A · sin(A + B); times 2 cos A cos B,
    sin A cos B + cos A sin B = sin(A + B)."""

    def construct(self):
        Aa, Ba = 29 * DEGREES, 35 * DEGREES
        tA, tB = float(np.tan(Aa)), float(np.tan(Ba))
        sA, sB = 1 / float(np.cos(Aa)), 1 / float(np.cos(Ba))
        Pm, Hm = np.array([0.0, 1.0]), np.zeros(2)
        Lm, Rm = np.array([-tA, 0.0]), np.array([tB, 0.0])
        check(close(np.linalg.norm(Lm - Pm), sA) and close(np.linalg.norm(Rm - Pm), sB),
              "sides sec A, sec B")
        check(abs(_ang(Pm, Lm, Hm) - Aa) < 1e-12 and abs(_ang(Pm, Hm, Rm) - Ba) < 1e-12,
              "the altitude splits the apex angle into A and B")
        phi = float(np.arctan2(*(Rm - Pm)[::-1]))
        turn = (PI - phi + PI) % TAU - PI                      # P->R ends pointing left
        P2m, L2m, R2m = (_rot2(p, turn, Pm) for p in (Pm, Lm, Rm))
        K2m = np.array([L2m[0], P2m[1]])
        check(abs(R2m[1] - P2m[1]) < 1e-12 and R2m[0] < K2m[0] < P2m[0] and L2m[1] > P2m[1],
              "turned copy: base R'P' level, L' above it, the foot inside")
        hh = float(L2m[1] - P2m[1])
        check(close(hh, sA * np.sin(Aa + Ba)), "h = sec A·sin(A + B)")
        ar = abs(area([Lm, Rm, Pm]))
        check(close(ar, 0.5 * (tA + tB)) and close(ar, 0.5 * sB * hh), "one area, two readings")
        check(close(np.sin(Aa + Ba), (tA + tB) * np.cos(Aa) * np.cos(Ba)), "the identity")

        k = 3.0
        O1 = np.array([-3.85, 0.55, 0.0])                       # H on screen
        O2 = np.array([3.15, 0.25, 0.0]) - k * to3(K2m - Hm)    # copy placed right

        def S1(p):
            return O1 + k * to3(p)

        def S2(p):
            return O2 + k * to3(p)

        P, H, L, R = S1(Pm), S1(Hm), S1(Lm), S1(Rm)
        P2, L2, R2, K2 = S2(P2m), S2(L2m), S2(R2m), S2(K2m)

        # ---- the triangle, its altitude 1, the angles A and B
        tL = mk([P, L, H], BLUE_D, 0.6)
        tR = mk([P, H, R], TEAL_D, 0.6)
        out = Polygon(P, L, R, stroke_color=WHITE, stroke_width=3.5)
        alt = Line(P, H, color=WHITE, stroke_width=3)
        raH = _ra(H, R, P, 0.18)
        aA = angle_arc(P, L, H, radius=0.8, color=BLUE_B, width=4)
        aB = angle_arc(P, H, R, radius=0.72, color=TEAL_B, width=4)
        lA = tag("A", 26, BLUE_B).move_to(P + 1.1 * angle_mid_dir(P, L, H))
        lB = tag("B", 26, TEAL_B).move_to(P + 1.02 * angle_mid_dir(P, H, R))
        l1 = _beside(tag("1", 28), H, P, RIGHT, 0.12, at=0.42)
        self.play(FadeIn(tL), FadeIn(tR), Create(out), Create(alt), Create(raH),
                  run_time=1.2)
        self.play(Create(aA), Create(aB), FadeIn(lA), FadeIn(lB), FadeIn(l1), run_time=0.7)
        ltA = tag("tan A", 26, BLUE_B).next_to(Line(L, H), DOWN, buff=0.14)
        ltB = tag("tan B", 26, TEAL_B).next_to(Line(H, R), DOWN, buff=0.14)
        lsA = _beside(tag("sec A", 26, BLUE_B), L, P, LEFT + UP * 0.6, 0.12)
        lsB = _beside(tag("sec B", 26, TEAL_B), P, R, RIGHT + UP * 0.8, 0.12)
        self.play(FadeIn(ltA), FadeIn(ltB), FadeIn(lsA), FadeIn(lsB), run_time=0.8)

        # ---- reading 1: base tan A + tan B, height 1
        hb = Line(L, R, color=YELLOW_B, stroke_width=8)
        hh1 = Line(P, H, color=YELLOW_B, stroke_width=8)
        r1 = tag("½ (tan A + tan B) · 1", 28).move_to([O1[0], -0.75, 0.0])
        self.play(Create(hb), Create(hh1), run_time=0.7)
        self.play(FadeIn(r1, shift=UP * 0.15), run_time=0.6)
        self.play(FadeOut(hb), FadeOut(hh1), run_time=0.4)

        # ---- a rigid copy turned until the side sec B is the base
        cp = VGroup(mk([P, L, H], BLUE_D, 0.6, stroke_width=0),
                    mk([P, H, R], TEAL_D, 0.6, stroke_width=0),
                    Polygon(P, L, R, stroke_color=WHITE, stroke_width=3.5))
        self.add(cp)
        self.play(_rigid(cp, [P, L, R], [P2, L2, R2], turn=turn), run_time=2.0)
        check(_landed(cp[2], [P2, L2, R2]), "the copy is rigid and lands as P'L'R'")
        aAB = angle_arc(P2, R2, L2, radius=0.45, color=YELLOW_B, width=4)
        sides = [np.linalg.norm(L2 - P2), np.linalg.norm(K2 - L2), np.linalg.norm(P2 - K2)]
        inc = (sides[0] * K2 + sides[1] * P2 + sides[2] * L2) / sum(sides)
        lAB = _lbl("A + B", 22, YELLOW_B, 0.6).move_to(inc)   # on the bisector at P'
        check(abs(_ang(P2, R2, inc) - _ang(P2, inc, L2)) < 1e-9, "label on the bisector")
        lsB2 = tag("sec B", 26, TEAL_B).next_to(Line(R2, P2), DOWN, buff=0.14)
        lsA2 = _beside(tag("sec A", 26, BLUE_B), P2, L2, RIGHT + UP * 0.5, 0.12)
        self.play(Create(aAB), FadeIn(lAB), FadeIn(lsB2), FadeIn(lsA2), run_time=0.8)

        # ---- reading 2: base sec B, height h = sec A·sin(A + B)
        hgt = DashedLine(L2, K2, color=YELLOW_B, stroke_width=3.5, dash_length=0.09)
        raK = _ra(K2, P2, L2, 0.18, YELLOW_B, 2.5)
        lh = _beside(tag("h", 28, YELLOW_B), K2, L2, LEFT, 0.12, at=0.55)
        hb2 = Line(R2, P2, color=YELLOW_B, stroke_width=8)
        self.play(Create(hgt), Create(raK), FadeIn(lh), Create(hb2), run_time=0.9)
        r2 = tag("½ · sec B · h", 28).move_to([O2[0] + k * K2m[0] - 0.3, -0.75, 0.0])
        r2.move_to([S2(K2m)[0] - 0.4, -0.75, 0.0])
        rh = tag("h  =  sec A · sin(A + B)", 26, YELLOW_B).next_to(r2, DOWN, buff=0.3)
        self.play(FadeIn(r2, shift=UP * 0.15), run_time=0.6)
        self.play(FadeIn(rh, shift=UP * 0.15), FadeOut(hb2), run_time=0.7)
        eq = tag("=", 34).move_to([(r1.get_right()[0] + r2.get_left()[0]) / 2, -0.75, 0.0])
        self.play(FadeIn(eq), run_time=0.4)
        r3 = tag("sin(A + B)  =  (tan A + tan B) · cos A · cos B", 28).move_to([0.0, -2.35, 0.0])
        self.play(FadeIn(r3, shift=UP * 0.15), run_time=0.8)

        segs = (_poly_segs([P, L, R]) + [_seg(P, H)] + _poly_segs([P2, L2, R2])
                + [_seg(L2, K2)] + _angle_segs(P, L, H, 0.8) + _angle_segs(P, H, R, 0.72)
                + _angle_segs(P2, R2, L2, 0.45))
        _labels_ok([lA, lB, l1, ltA, ltB, lsA, lsB, lAB, lsB2, lsA2, lh, r1, r2, rh, eq, r3],
                   segs, "G27")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("sin(A + B)  =  sin A cos B + cos A sin B", 34)))
        self.hold(2.2)


# ================================================ G15 COS 36° AND THE GOLDEN RATIO

class G15_Cos36Golden(Board):
    """The 36°–72°–72° triangle with base 1 and legs φ. The bisector of the
    angle at B meets AC at D: triangle BCD has angles 36°, 72°, 72°, so
    BD = BC = 1, and triangle ABD has two 36° angles, so AD = BD = 1 and
    DC = φ − 1. Turned over about the bisector at C and enlarged by φ, CDB
    lands on CBA: (φ − 1) : 1 = 1 : φ, so φ² = φ + 1. In the isosceles
    triangle ABD the fold over DM halves AB: AM = φ/2, and cos 36° = φ/2."""

    def construct(self):
        phi = (1 + np.sqrt(5)) / 2
        Bm, Cm = np.array([-0.5, 0.0]), np.array([0.5, 0.0])
        Am = np.array([0.0, float(np.sqrt(phi * phi - 0.25))])
        Dm = Am + (Cm - Am) / phi
        Mm = (Am + Bm) / 2
        d36, d72 = 36 * DEGREES, 72 * DEGREES
        check(abs(_ang(Am, Bm, Cm) - d36) < 1e-12 and abs(_ang(Bm, Am, Cm) - d72) < 1e-12
              and abs(_ang(Cm, Am, Bm) - d72) < 1e-12, "36°, 72°, 72° with base 1, legs φ")
        check(abs(_ang(Bm, Am, Dm) - d36) < 1e-12 and abs(_ang(Bm, Dm, Cm) - d36) < 1e-12,
              "BD bisects the angle at B")
        check(abs(_ang(Dm, Bm, Cm) - d72) < 1e-12, "the angle BDC is 72°")
        check(close(np.linalg.norm(Dm - Bm), 1.0) and close(np.linalg.norm(Dm - Am), 1.0)
              and close(np.linalg.norm(Cm - Dm), phi - 1), "BD = AD = 1, DC = φ − 1")
        bis_m = np.array([np.cos(144 * DEGREES), np.sin(144 * DEGREES)])
        check(abs(_ang(Cm, Cm + bis_m, Bm) - d36) < 1e-12
              and abs(_ang(Cm, Cm + bis_m, Am) - d36) < 1e-12, "the bisector at C")

        def flip_enlarge(p):
            return Cm + phi * (_refl(p, Cm, bis_m) - Cm)
        check(close(flip_enlarge(Dm), Bm) and close(flip_enlarge(Bm), Am),
              "CDB turned over and enlarged by φ is CBA")
        check(abs(np.dot(Dm - Mm, Bm - Am)) < 1e-12 and close(np.linalg.norm(Mm - Am), phi / 2),
              "DM ⊥ AB, AM = φ/2")
        check(close(np.cos(d36), phi / 2) and close(phi * phi, phi + 1), "cos 36° = φ/2")

        F = Frame(-0.92, 1.25, -0.22, Am[1] + 0.2, max_w=7.0,
                  centre=(-3.0, (SAFE_TOP + SAFE_BOTTOM) / 2))
        S = F.P
        A, B, C, D, M = (S(p) for p in (Am, Bm, Cm, Dm, Mm))
        def out_n(P_, Q_, inner):
            """Unit normal of the line P_Q_ pointing away from the point inner."""
            n = _unit(np.array([-(Q_ - P_)[1], (Q_ - P_)[0]]))
            return -n if np.dot(n, to3(inner) - to3(P_)) > 0 else n
        nAC = out_n(A, C, B)
        nAB = out_n(A, B, C)

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        aA = angle_arc(A, B, C, radius=0.8, color=YELLOW_B, width=4)
        aB = angle_arc(B, A, C, radius=0.5, color=TEAL_B, width=4)
        aC = angle_arc(C, A, B, radius=0.5, color=TEAL_B, width=4)
        lA = tag("36°", 22, YELLOW_B).move_to(A + 1.14 * angle_mid_dir(A, B, C))
        lB = tag("72°", 22, TEAL_B).move_to(B + 0.9 * angle_mid_dir(B, A, C))
        lC = tag("72°", 22, TEAL_B).move_to(C + 0.9 * angle_mid_dir(C, A, B))
        l1 = tag("1", 28).next_to(Line(B, C), DOWN, buff=0.14)
        dAC = _dim(A, C, nAC, color=GREY_A, off=1.55)
        lphi = _beside(tag("φ", 30), A + 1.55 * nAC, C + 1.55 * nAC, nAC, 0.1)
        self.play(Create(tri), FadeIn(l1), Create(dAC), FadeIn(lphi), run_time=1.1)
        self.play(Create(aA), Create(aB), Create(aC), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                  run_time=0.8)
        self.hold(0.3)

        # bisect the angle at B
        BD = Line(B, D, color=WHITE, stroke_width=3)
        aB1 = angle_arc(B, A, D, radius=0.62, color=YELLOW_B, width=4)
        aB2 = angle_arc(B, D, C, radius=0.5, color=YELLOW_B, width=4)
        lB1 = tag("36°", 20, YELLOW_B).move_to(B + 1.3 * angle_mid_dir(B, A, D))
        lB2 = tag("36°", 20, YELLOW_B).move_to(B + 1.22 * angle_mid_dir(B, D, C))
        self.play(Create(BD), FadeOut(aB), FadeOut(lB), run_time=0.7)
        self.play(Create(aB1), Create(aB2), FadeIn(lB1), FadeIn(lB2), run_time=0.6)
        # angle sum in BCD: 72° at D, so BD = BC = 1
        aD = angle_arc(D, B, C, radius=0.42, color=TEAL_B, width=4)
        lD = tag("72°", 20, TEAL_B).move_to(D + 0.78 * angle_mid_dir(D, B, C))
        tBCD = mk([B, C, D], ORANGE, 0.45, stroke_width=0)
        lBD = _beside(tag("1", 28), B, D, UP + LEFT * 0.4, 0.12, at=0.6)
        self.add(tBCD)
        self.bring_to_back(tBCD)
        tBCD.set_opacity(0)
        self.play(tBCD.animate.set_fill(opacity=0.45), Create(aD), FadeIn(lD), run_time=0.7)
        self.play(FadeIn(lBD), run_time=0.5)
        # two 36° angles in ABD: AD = BD = 1, and DC = φ − 1
        tABD = mk([A, B, D], BLUE_D, 0.4, stroke_width=0)
        self.add(tABD)
        self.bring_to_back(tABD)
        tABD.set_opacity(0)
        dAD = _dim(A, D, nAC, color=BLUE_B, off=0.4)
        dDC = _dim(D, C, nAC, color=ORANGE, off=0.4)
        lAD = _beside(tag("1", 26, BLUE_B), A + 0.4 * nAC, D + 0.4 * nAC, nAC, 0.08)
        lDC = _beside(tag("φ − 1", 24, ORANGE), D + 0.4 * nAC, C + 0.4 * nAC, nAC, 0.08)
        self.play(tABD.animate.set_fill(opacity=0.4), Create(dAD), FadeIn(lAD), run_time=0.7)
        self.play(Create(dDC), FadeIn(lDC), run_time=0.6)
        self.hold(0.4)

        # CDB, turned over about the bisector at C and enlarged by φ, is CBA
        cp = mk([C, D, B], ORANGE, 0.75, stroke_color=YELLOW_B, stroke_width=3)
        bis = to3(bis_m)
        ax = DashedLine(C, C + 2.6 * bis, color=YELLOW_B, stroke_width=2, dash_length=0.08)
        self.play(FadeIn(cp), Create(ax), run_time=0.5)
        self.play(Rotate(cp, angle=PI, axis=bis, about_point=C), run_time=1.3)
        tagk = tag("× φ", 28, YELLOW_B).move_to([2.2, 3.05, 0.0])
        self.play(cp.animate.scale(phi, about_point=C), FadeIn(tagk), run_time=1.4)
        check(_landed(cp, [C, B, A]), "the turned, enlarged copy is CBA")
        self.play(cp.animate.set_fill(opacity=0).set_stroke(opacity=0), FadeOut(ax),
                  FadeOut(tagk), run_time=0.6)
        self.remove(cp)
        x0 = 3.5
        r1 = tag("(φ − 1) : 1  =  1 : φ", 30).move_to([x0, 2.6, 0.0])
        r2 = tag("φ²  =  φ + 1   ⟹   φ  =  (1 + √5)/2", 26).move_to([x0, 1.6, 0.0])
        self.play(FadeIn(r1, shift=LEFT * 0.2), run_time=0.7)
        self.play(FadeIn(r2, shift=LEFT * 0.2), run_time=0.7)

        # in the isosceles ABD, the fold over DM halves AB
        DMl = DashedLine(D, M, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        half = mk([A, D, M], BLUE_D, 0.6, stroke_color=BLUE_B, stroke_width=3)
        self.play(Create(DMl), run_time=0.5)
        self.play(FadeIn(half), run_time=0.3)
        self.play(Rotate(half, angle=PI, axis=_unit(M - D), about_point=D), run_time=1.3)
        check(_landed(half, [B, D, M]), "the fold carries ADM onto BDM")
        raM = _ra(M, A, D, 0.16)
        tkM = VGroup(_ticks(A, M, 2, YELLOW_B, at=0.62), _ticks(M, B, 2, YELLOW_B, at=0.38))
        lAM = _beside(tag("φ/2", 26, YELLOW_B), A, M, nAB, 0.26, at=0.45)
        self.play(FadeOut(half), Create(raM), FadeIn(tkM), FadeIn(lAM), run_time=0.7)
        r3 = tag("cos 36°  =  (φ/2) / 1", 30, YELLOW_B).move_to([x0, 0.4, 0.0])
        self.play(FadeIn(r3, shift=LEFT * 0.2), run_time=0.7)

        segs = (_poly_segs([A, B, C]) + [_seg(B, D), _seg(D, M)]
                + [_seg(*d[0].get_start_and_end()) for d in (dAC, dAD, dDC)]
                + [_seg(t.get_start(), t.get_end()) for g in tkM for t in g]
                + _angle_segs(A, B, C, 0.8) + _angle_segs(B, A, D, 0.62)
                + _angle_segs(B, D, C, 0.5) + _angle_segs(C, A, B, 0.5)
                + _angle_segs(D, B, C, 0.42))
        _labels_ok([lA, lB1, lB2, lC, lD, l1, lBD, lphi, lAD, lDC, lAM, r1, r2, r3],
                   segs, "G15")
        _labels_ok([tagk], segs, "G15 factor")
        check(_in_safe(self.mobjects), "everything inside the content area")
        self.play(Write(caption("cos 36°  =  φ/2  =  (1 + √5)/4", 36)))
        self.hold(2.2)
