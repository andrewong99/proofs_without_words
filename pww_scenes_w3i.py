# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3i.py — proofs without words, 2D (manim):
#     F41 flank triangles                  F35 Van Aubel's theorem
#     F36 squares on a parallelogram       F47 the square in a right triangle
#     F54 Stewart's theorem                F57 Viviani for regular polygons
#     F58 the triangle of medians          F61 the incentre-excentre lemma
#     F63 cutting corners at thirds        F66 exterior angle bisector theorem
#
# Every move of a piece is rigid -- a slide (shift), a turn about a named
# point (manim Rotate), or a fold (a half-turn in space about a line of the
# plane) -- unless it is an announced enlargement about a named centre or a
# shear that keeps its base and its pair of parallels. Every landing, area,
# angle and tiling claim is checked numerically with check(...) before or
# right after it is drawn, and every label is checked against everything
# drawn near it (lines, ticks, arcs, right-angle marks, dots), so a wrong
# construction fails the render instead of drawing a wrong picture.
# No LaTeX: every label is Text (tag / caption) with Unicode.


# ------------------------------------------------------------ point helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _angle(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _dir(p, q):
    """Direction angle of the ray p -> q."""
    d = to3(q) - to3(p)
    return float(np.arctan2(d[1], d[0]))


def _polar(a):
    return np.array([np.cos(a), np.sin(a), 0.0])


def _perp(v):
    v = to3(v)
    return np.array([-v[1], v[0], 0.0])


def _foot(p, a, b):
    """Foot of the perpendicular from p onto the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = b - a
    return a + np.dot(p - a, d) / np.dot(d, d) * d


def _meet(p1, p2, q1, q2):
    """Intersection of the lines p1p2 and q1q2."""
    p1, p2, q1, q2 = to3(p1), to3(p2), to3(q1), to3(q2)
    m = np.array([(p2 - p1)[:2], (q1 - q2)[:2]]).T
    t, _ = np.linalg.solve(m, (q1 - p1)[:2])
    return p1 + t * (p2 - p1)


def _rot2(p, c, th):
    """Point p turned by th about c (in the plane)."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


def _mirror(p, a, b):
    """Mirror image of p in the line ab."""
    f = _foot(p, a, b)
    return 2 * f - to3(p)


def _on_line(p, a, b, tol=1e-9):
    return abs(_cross(to3(b) - to3(a), to3(p) - to3(a))) < tol * max(
        1.0, float(np.linalg.norm(to3(b) - to3(a))))


def _between(p, a, b, tol=1e-9):
    """p lies on the segment ab, strictly inside it."""
    p, a, b = to3(p), to3(a), to3(b)
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    return _on_line(p, a, b, tol) and tol < t < 1 - tol


def _tri_from_angles(Bdeg, Cdeg, a=1.0):
    """Math triangle with B = (0, 0), C = (a, 0) and the given angles at B
    and C (A above BC, so A, B, C run counter-clockwise)."""
    Bn, Cn = Bdeg * DEGREES, Cdeg * DEGREES
    c = a * np.sin(Cn) / np.sin(PI - Bn - Cn)
    return (np.array([c * np.cos(Bn), c * np.sin(Bn)]), np.array([0.0, 0.0]),
            np.array([a, 0.0]))


def _scalene(A, B, C, gap=0.05):
    """Angles pairwise at least `gap` (radians) apart."""
    angs = sorted((_angle(A, B, C), _angle(B, C, A), _angle(C, A, B)))
    return angs[1] - angs[0] > gap and angs[2] - angs[1] > gap


def _outer_square(u, v):
    """Square on the directed side u -> v, on its right (outside a
    counter-clockwise polygon): [u, v, v + n, u + n]."""
    u, v = to3(u), to3(v)
    n = -_perp(v - u)
    return [u, v, v + n, u + n]


# ------------------------------------------------------------ marks

def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


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


def _arcs(v, p, q, r, n=1, color=YELLOW_B, width=4, gap=0.08):
    """n concentric arcs for the non-reflex angle p-v-q (equal-angle marks)."""
    return VGroup(*[angle_arc(v, p, q, r + k * gap, color, width)
                    for k in range(n)])


def _wedge(vertex, p, q, r, color, op=FILL, stroke=1.5, stroke_color=WHITE):
    """Filled sector for the NON-REFLEX angle p-vertex-q (cf. angle_arc)."""
    v, p, q = to3(vertex), to3(p), to3(q)
    a1 = float(np.arctan2(*(p - v)[1::-1]))
    a2 = float(np.arctan2(*(q - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    s = Sector(radius=r, start_angle=a1, angle=span, arc_center=v,
               color=color, fill_opacity=op)
    s.set_stroke(stroke_color, stroke)
    return s


def _turn_arrow(c, a0, a1, r=0.55, color=YELLOW_B, width=4):
    """A curved arrow about c from direction a0 to direction a1 (radians,
    the short way round): shows the sense of a turn."""
    span = (a1 - a0 + PI) % TAU - PI
    arc = Arc(radius=r, start_angle=a0, angle=span, arc_center=to3(c),
              color=color, stroke_width=width)
    arc.add_tip(tip_length=0.16, tip_width=0.16)
    return arc


def _bracket(p, q, nrm, text, gap=0.28, color=YELLOW_B, size=28, tick=0.11,
             buff=0.12, width=3):
    """Dimension bar beside segment pq, `gap` away along the unit normal
    nrm, with end ticks; the label sits beyond the bar. It spans the whole
    segment, so it names the union of whatever lies along it."""
    p, q, n = to3(p), to3(q), _unit(nrm)
    a, b = p + gap * n, q + gap * n
    bar = VGroup(Line(a, b, color=color, stroke_width=width),
                 Line(a - tick * n, a + tick * n, color=color, stroke_width=width),
                 Line(b - tick * n, b + tick * n, color=color, stroke_width=width))
    lab = tag(text, size, color)
    half = abs(n[0]) * lab.width / 2 + abs(n[1]) * lab.height / 2
    lab.move_to((a + b) / 2 + n * (buff + half + 0.02))
    return bar, lab


# ------------------------------------------------------------ moves

def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


def _homothety(mob, centre, factor, **kw):
    """Enlargement about `centre` by `factor` (geometric in time, so every
    frame is a true scaled copy)."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _shear_to(mob, base, apex0, apex1, **kw):
    """Shear of a triangle [base0, base1, apex]: the base stays, the apex
    slides straight from apex0 to apex1 along a parallel to the base."""
    b0, b1 = to3(base[0]), to3(base[1])
    a0, a1 = to3(apex0), to3(apex1)
    check(abs(_cross(b1 - b0, a1 - a0)) < 1e-9, "a shear moves the apex parallel to its base")

    def upd(m, alpha):
        m.become(mk([b0, b1, a0 + alpha * (a1 - a0)], m.get_fill_color(),
                    m.get_fill_opacity(), stroke_color=m.get_stroke_color(),
                    stroke_width=m.get_stroke_width()))
    return UpdateFromAlphaFunc(mob, upd, **kw)


# ------------------------------------------------------------ area / tiling

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
    P = [to3(p) for p in P]
    Q = [to3(q) for q in Q]
    if len(P) != len(Q):
        return False
    n = len(P)
    for d in (1, -1):
        for s in range(n):
            if all(close(P[i], Q[(s + d * i) % n], tol) for i in range(n)):
                return True
    return False


def _landed(mob, pts, tol=1e-6):
    """The polygon mob's vertices are pts (same order)."""
    v = mob.get_vertices()
    return len(v) == len(pts) and all(close(a, to3(b), tol)
                                      for a, b in zip(v, pts))


def _icon(poly, k, color, op=FILL):
    """A small copy of a piece, scaled by k (all icons share k, so their
    sizes compare as the pieces' areas do)."""
    return mk([to3(p) for p in poly], color, op, stroke_width=1.5).scale(k)


# ------------------------------------------------------------ text helpers

def _on_base(s, size, color):
    """Text s with its baseline at y = 0, as Pango sets it: a reference 'M'
    is laid out in front of it, measured and removed (so a lone '=' or '+'
    sits at its proper height)."""
    t = Text("M" + s, font_size=size, color=color)
    m = t.submobjects[0]
    base = m.get_bottom()[1]
    t.remove(m)
    t.shift(np.array([0.0, -base, 0.0]))
    return t


def _trow(parts, size=30, buff=0.16):
    """A row of coloured Text pieces [(s, colour), ...] on one baseline."""
    g = VGroup()
    x = 0.0
    for s, col in parts:
        t = _on_base(s, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + buff
        g.add(t)
    return g


def _row(*parts, buff=0.16):
    """Parts side by side, their centres on one horizontal line."""
    return VGroup(*parts).arrange(RIGHT, buff=buff)


# ------------------------------------------------------------ label hygiene

def _box(m, pad=0.0):
    return (m.get_left()[0] - pad, m.get_right()[0] + pad,
            m.get_bottom()[1] - pad, m.get_top()[1] + pad)


def _seg(p, q):
    return (to3(p), to3(q))


def _segs(*pts, closed=False):
    """Segments of the polyline through pts (closed: back to the start)."""
    pts = [to3(p) for p in pts]
    out = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if closed:
        out.append((pts[-1], pts[0]))
    return out


def _mob_segs(*mobs, n=8):
    """Every stroke actually drawn by the given mobjects -- lines, polygon
    edges, dashes, ticks, arcs, right-angle marks, dot outlines -- as short
    straight segments (each cubic piece sampled n times). Text is skipped:
    labels are checked against each other separately."""
    out = []

    def walk(m):
        if isinstance(m, Text):
            return
        pts = m.points if hasattr(m, "points") else np.zeros((0, 3))
        if len(pts) >= 4 and (m.get_stroke_width() > 0 or m.get_fill_opacity() > 0):
            ts = np.linspace(0.0, 1.0, n + 1)
            for i in range(0, len(pts) - 3, 4):
                a, b, c, d = pts[i:i + 4]
                if np.allclose(a, d) and np.allclose(a, b) and np.allclose(a, c):
                    continue
                cur = [((1 - t) ** 3) * a + 3 * ((1 - t) ** 2) * t * b
                       + 3 * (1 - t) * t * t * c + t ** 3 * d for t in ts]
                out.extend((cur[j], cur[j + 1]) for j in range(n))
        for s in m.submobjects:
            walk(s)

    for mob in mobs:
        walk(mob)
    return out


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
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.012) + 2)):
            x, y = (p + t * (q - p))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
    return None


def _overlap(a, b, gap=0.05):
    a, b = _box(a), _box(b)
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0]
                or a[3] + gap <= b[2] or b[3] + gap <= a[2])


def _in_frame(m, tol=0.02):
    return (m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
            and m.get_top()[1] <= SAFE_TOP + tol
            and m.get_bottom()[1] >= SAFE_BOTTOM - tol)


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.06, gap=0.06):
    """Every label inside the safe area, clear of every segment in segs and
    of every other label."""
    for m in labels:
        check(_in_frame(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        sg = "" if i is None else (f" {np.round(segs[i][0][:2], 2)}-"
                                   f"{np.round(segs[i][1][:2], 2)}")
        check(i is None, f"{what}: '{_name(m)}' at {np.round(m.get_center()[:2], 2)} "
              f"clear of the lines (hits #{i}{sg})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _park(m, X, segs, dirs=None, d0=0.24, d1=1.4, pad=0.07, avoid=(), sep=0.06,
          towards=None):
    """Move label m next to the point X, at the least distance in [d0, d1]
    where it clears every segment in segs and every mobject in `avoid`.
    Directions are tried in the order given (degrees), or else along the
    middles of the angular gaps between the directions to `towards`
    (widest gap first). Fails the render if no spot exists."""
    X = to3(X)

    def nearest(a):
        for d in np.arange(d0, d1 + 1e-9, 0.02):
            m.move_to(X + d * _polar(a))
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return d
        return None

    def why(a):                                # what blocks direction a (for the report)
        out = []
        for d in np.linspace(d0, d1, 4):
            m.move_to(X + d * _polar(a))
            i = _hit(m, segs, pad)
            out.append(f"d={d:.2f}:" + (f"seg{np.round(segs[i][0][:2], 2)}" if i is not None
                                        else "frame" if not _in_frame(m) else "label"))
        return ", ".join(out)

    if dirs is not None:                       # first direction that works
        for a in dirs:
            d = nearest(a * DEGREES)
            if d is not None:
                return m.move_to(X + d * _polar(a * DEGREES))
        check(False, f"a free spot for the label '{_name(m)}' at {np.round(X[:2], 2)}: "
              + " | ".join(f"{a}°: {why(a * DEGREES)}" for a in dirs[:4]))
    angs = sorted(_dir(X, q) % TAU for q in towards)
    found = []
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        mid = (a0 + a1) / 2
        d = nearest(mid)
        if d is not None:
            found.append((d, -(a1 - a0), mid))
    if not found:                              # no gap middle works: scan all round
        for k in range(36):
            d = nearest(k * TAU / 36)
            if d is not None:
                found.append((d, 0.0, k * TAU / 36))
    check(bool(found), f"a free spot for the label '{_name(m)}'")
    d, _, mid = min(found)                     # the nearest free spot
    return m.move_to(X + d * _polar(mid))


def _park_beside(m, p, q, segs, side=None, ats=(0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74),
                 gaps=(0.08, 0.12, 0.17, 0.23, 0.3), pad=0.06, avoid=(), sep=0.06):
    """Park label m beside the segment pq (several positions along it; on
    the side the vector `side` points to, or either side) where it clears
    every segment in segs and every mobject in `avoid`."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = _perp(d)
    sides = (nrm, -nrm)
    if side is not None:
        sides = (nrm,) if np.dot(nrm, to3(side)) > 0 else (-nrm,)
    for at in ats:
        for sd in sides:
            for g in gaps:
                hw, hh = m.width / 2, m.height / 2
                off = hw * abs(sd[0]) + hh * abs(sd[1]) + g
                m.move_to(p + at * (q - p) + off * sd)
                if (_hit(m, segs, pad) is None and _in_frame(m)
                        and not any(_overlap(m, o, sep) for o in avoid)):
                    return m
    check(False, f"a free spot beside a segment for the label '{_name(m)}'")


def _angle_label(m, V, P, Q, r0, segs=(), r1=3.0, pad=0.06, fracs=(0.5,),
                 avoid=(), sep=0.06):
    """Put label m inside the non-reflex angle P-V-Q -- on its bisector, or
    at the other fractions of the angle given in `fracs`, tried in order --
    at the least distance >= r0 from V where it clears both arms, segs and
    the mobjects in `avoid`."""
    V = to3(V)
    a1 = float(np.arctan2(*(to3(P) - V)[1::-1]))
    a2 = float(np.arctan2(*(to3(Q) - V)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    arms = [_seg(V, P), _seg(V, Q)] + list(segs)
    found = []                                   # (distance, order, direction)
    for k, f in enumerate(fracs):
        u = _polar(a1 + f * span)
        for d in np.arange(r0, r1, 0.02):
            m.move_to(V + d * u)
            if (_hit(m, arms, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)):
                found.append((round(d, 6), k, u))
                break
    check(bool(found), f"room for the angle label '{_name(m)}'")
    d, _, u = min(found, key=lambda x: (x[0], x[1]))     # the nearest spot
    return m.move_to(V + d * u)


def _all_in_frame(mobs, what):
    for i, m in enumerate(mobs):
        check(_in_frame(m), f"{what}: item {i} ({_name(m)}) inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")


def _final_check(scene, cap, what, pad=0.05, gap=0.05, skip=()):
    """The closing frame: every mobject inside the safe area, the caption in
    its band, every text label clear of every other label and of every
    stroke on screen (lines, ticks, arcs, right-angle marks, dots)."""
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM - 0.02, f"{what}: caption in its band")
    shown = [m for m in scene.mobjects if m is not cap]
    for m in shown:
        check(_in_frame(m), f"{what}: {_name(m)} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")
    texts = []

    def collect(m):
        if isinstance(m, Text):
            texts.append(m)
            return
        for s in m.submobjects:
            collect(s)
    for m in shown:
        collect(m)
    skip_ids = {id(x) for s in skip for x in s.get_family()}
    strokes = _mob_segs(*[m for m in shown if id(m) not in skip_ids])
    _labels_ok(texts, strokes, what, pad=pad, gap=gap)


# ============================================ F41 FLANK TRIANGLES

class F41_FlankTriangles(Board):
    """Squares stand outward on the three sides of ABC. The flank triangle
    at A lies between the two squares that meet at A; its sides from A are
    the squares' sides AQ (as long as AB, perpendicular to it) and AS (as
    long as AC, perpendicular to it). A quarter-turn about A carries S onto
    C and Q onto B′, the point beyond A with AB′ = AB on the line BA. So
    the turned flank is the triangle AB′C: its base AB′ equals AB and lies
    on the same line, and its apex is C, the apex of ABC over AB -- the
    same height. Equal bases, equal heights: equal areas. The same
    quarter-turn about B and about C settles the other two flanks. (Any
    triangle; drawn scalene and acute.)"""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(72, 47)
        qt = -PI / 2                                  # the quarter-turn (clockwise)
        names = ["A", "B", "C"]
        Vm = {"A": Am, "B": Bm, "C": Cm}

        def prev_next(v):
            i = names.index(v)
            return names[i - 1], names[(i + 1) % 3]

        def flank(X, U, W):          # vertex, previous, next (counter-clockwise)
            return [X, _rot2(W, X, -PI / 2), _rot2(U, X, PI / 2)]

        def math_pts():
            out = []
            for v in names:
                u, w = prev_next(v)
                f = flank(Vm[v], Vm[u], Vm[w])
                out += f + [_rot2(q, Vm[v], qt) for q in f]
            for u, w in (("A", "B"), ("B", "C"), ("C", "A")):
                out += _outer_square(Vm[u], Vm[w])
            return np.array([to3(p) for p in out])

        allp = math_pts()
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        Fr = Frame(lo[0] - 0.03, hi[0] + 0.03, lo[1] - 0.03, hi[1] + 0.03,
                   max_w=7.6, centre=(-2.45, (SAFE_TOP + SAFE_BOTTOM) / 2))
        V = {v: Fr.P(Vm[v]) for v in names}
        A, B, C = V["A"], V["B"], V["C"]
        T = abs(area([A, B, C]))
        sq = {"AB": _outer_square(A, B), "BC": _outer_square(B, C), "CA": _outer_square(C, A)}
        fl, img, other = {}, {}, {}
        for v in names:
            u, w = prev_next(v)
            fl[v] = flank(V[v], V[u], V[w])
            img[v] = [_rot2(q, V[v], qt) for q in fl[v]]
            other[v] = 2 * V[v] - V[w]               # the far end of the turned base

        # ---- checks: the squares, the flanks, the quarter-turn landings
        check(_scalene(A, B, C) and max(_angle(A, B, C), _angle(B, C, A),
                                        _angle(C, A, B)) < PI / 2, "scalene acute triangle")
        check(area([A, B, C]) > 0, "A, B, C run counter-clockwise")
        for key, S4 in sq.items():
            for i in range(4):
                e1, e2 = S4[(i + 1) % 4] - S4[i], S4[(i + 2) % 4] - S4[(i + 1) % 4]
                check(abs(np.linalg.norm(e1) - np.linalg.norm(e2)) < 1e-9
                      and abs(np.dot(e1, e2)) < 1e-9, f"the square on {key}")
            check(not _pip((S4[0] + S4[2]) / 2, [A, B, C]), f"the square on {key} is outside")
        for v in names:
            u, w = prev_next(v)
            X = V[v]
            sq_u = sq[u + v] if (u + v) in sq else sq[v + u]
            sq_w = sq[v + w] if (v + w) in sq else sq[w + v]
            check(any(close(fl[v][2], q) for q in sq_u)
                  and any(close(fl[v][1], q) for q in sq_w),
                  f"the flank at {v} joins corners of the two squares at {v}")
            check(abs(abs(area(fl[v])) - T) < 1e-9, f"the flank at {v} has the area of ABC")
            check(close(img[v][0], X) and close(img[v][1], other[v]) and close(img[v][2], V[u]),
                  f"the quarter-turn about {v} lands the flank on {v}, {w}′, {u}")
            check(_on_line(other[v], X, V[w]) and close((other[v] + V[w]) / 2, X),
                  f"{v} is the midpoint of {w}{w}′: equal bases on one line")

        # ---- the triangle, the squares, the flanks
        cT, cF = BLUE_D, {"A": ORANGE, "B": TEAL_D, "C": GREEN_D}
        tri = mk([A, B, C], cT, stroke_width=4)
        sqs = VGroup(*[mk(S4, GREY_D, 0.45, stroke_color=GREY_B, stroke_width=2)
                       for S4 in sq.values()])
        flm = {v: mk(fl[v], cF[v], stroke_width=2) for v in names}

        # every demo's pieces, built first, so the labels can avoid them all:
        # the copy (landed: the triangle v, w′, u), the turn arrow, the two
        # equal bases on one line with ticks, the common height and its foot
        parts = {}
        nt = {"A": 1, "B": 2, "C": 3}                 # tick counts: one per length
        for v in names:
            u, w = prev_next(v)
            X = V[v]
            H = _foot(V[u], X, V[w])
            check(_between(H, X, V[w]), "the foot of the height lies on the side")
            parts[v] = dict(
                cp=mk(fl[v], cF[v], 0.9, stroke_color=YELLOW_B, stroke_width=3),
                arrow=_turn_arrow(X, _dir(X, fl[v][2]), _dir(X, V[u]), r=0.5),
                marks=VGroup(Line(X, V[w], color=BLUE_B, stroke_width=7),
                             Line(X, other[v], color=YELLOW_B, stroke_width=7),
                             _ticks(X, V[w], nt[v], WHITE), _ticks(X, other[v], nt[v], WHITE),
                             DashedLine(V[u], H, color=WHITE, stroke_width=3,
                                        dash_length=0.08),
                             _ra(H, V[u], X, 0.16)))
        landed = VGroup(*[mk(img[v], cF[v], 0.9, stroke_width=3) for v in names])
        every = _mob_segs(tri, sqs, *flm.values(), landed,
                          *[parts[v]["arrow"] for v in names],
                          *[parts[v]["marks"] for v in names])
        labs = {}
        for v in names:
            labs[v] = _angle_label(tag(v, 30), V[v], fl[v][1], fl[v][2], 0.3,
                                   segs=every, pad=0.11, fracs=(0.5, 0.62, 0.38, 0.72, 0.28))
        self.play(FadeIn(tri), *[FadeIn(labs[v]) for v in names], run_time=1.0)
        self.play(FadeIn(sqs), run_time=0.9)
        self.add(*flm.values())
        self.bring_to_front(*labs.values())
        self.play(*[FadeIn(flm[v]) for v in names], run_time=0.8)
        self.hold(0.3)

        # ---- the panel: each flank = ABC, as icons at one scale, the
        # equals signs in one column
        k_ic = 0.36
        px = 4.3
        rows = {}
        for i, v in enumerate(names):
            y = 2.45 - 1.55 * i
            eq = tag("=", 34).move_to([px, y, 0.0])
            left = _icon(fl[v], k_ic, cF[v])
            left.move_to([px - 0.32 - left.width / 2, y, 0.0])
            right = _icon([A, B, C], k_ic, cT)
            right.move_to([px + 0.32 + right.width / 2, y, 0.0])
            rows[v] = VGroup(left, eq, right)
        check(min(r.get_left()[0] for r in rows.values()) > max(
            q[0] for v in names for q in fl[v] + img[v] + sq["CA"]) + 0.25,
            "the panel is clear of the figure")

        def demo(vs):
            cps = [parts[v]["cp"] for v in vs]
            arrows = [parts[v]["arrow"] for v in vs]
            marks = VGroup(*[parts[v]["marks"] for v in vs])
            self.add(*cps)
            self.bring_to_front(*labs.values())       # labels stay above the moving copies
            self.play(*[flm[v].animate.set_fill(opacity=0.3) for v in vs],
                      *[FadeIn(c) for c in cps], *[Create(a) for a in arrows], run_time=0.6)
            self.play(*[Rotate(c, angle=qt, about_point=V[v]) for c, v in zip(cps, vs)],
                      run_time=1.8)
            for c, v in zip(cps, vs):
                check(_landed(c, img[v]), f"the turned flank at {v} lands vertex by vertex")
            self.play(*[FadeOut(a) for a in arrows], Create(marks), run_time=1.0)
            self.bring_to_front(*labs.values())
            return cps, marks

        # ---- at A: the flank, turned a quarter-turn, stands on AB′ = AB
        cps, marks = demo(["A"])
        lBp = _park(tag("B′", 28), other["A"], every, d0=0.2, d1=0.7,
                    dirs=(175, 160, 190, 145, 205, 130, 220), avoid=list(labs.values()))
        check(np.linalg.norm(lBp.get_center() - other["A"]) < 0.55, "B′ sits at B′")
        self.play(FadeIn(lBp), run_time=0.4)
        _labels_ok([*labs.values(), lBp], _mob_segs(tri, sqs, *flm.values(), *cps, marks),
                   "F41 at A")
        self.play(FadeIn(rows["A"], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.5)
        self.play(FadeOut(cps[0]), FadeOut(marks), FadeOut(lBp),
                  flm["A"].animate.set_fill(opacity=FILL), run_time=0.6)

        # ---- at B and at C, the same quarter-turn
        cps, marks = demo(["B", "C"])
        _labels_ok(list(labs.values()), _mob_segs(tri, sqs, *flm.values(), *cps, marks),
                   "F41 at B and C")
        self.play(FadeIn(rows["B"], shift=LEFT * 0.2), FadeIn(rows["C"], shift=LEFT * 0.2),
                  run_time=0.8)
        self.hold(0.6)
        self.play(*[FadeOut(c) for c in cps], FadeOut(marks),
                  *[flm[v].animate.set_fill(opacity=FILL) for v in ("B", "C")], run_time=0.6)
        self.bring_to_front(*labs.values())

        cap = caption("each flank triangle has the area of ABC", 34)
        _final_check(self, cap, "F41")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F35 VAN AUBEL'S THEOREM

def _quad_from_angles(Ad, Bd, Cd, Dd, ab, bc):
    """Counter-clockwise quadrilateral ABCD with the given angles (degrees,
    summing to 360) and sides AB = ab, BC = bc; A = (0, 0), B on the
    x-axis. CD and DA follow from closing it up."""
    check(abs(Ad + Bd + Cd + Dd - 360) < 1e-9, "the angles of a quadrilateral")
    h1 = (180 - Bd) * DEGREES
    h2 = h1 + (180 - Cd) * DEGREES
    h3 = h2 + (180 - Dd) * DEGREES
    A = np.array([0.0, 0.0])
    B = A + ab * np.array([1.0, 0.0])
    C = B + bc * np.array([np.cos(h1), np.sin(h1)])
    u2, u3 = np.array([np.cos(h2), np.sin(h2)]), np.array([np.cos(h3), np.sin(h3)])
    l3, l4 = np.linalg.solve(np.array([u2, u3]).T, A - C)
    check(l3 > 0 and l4 > 0, "the quadrilateral closes up")
    return A, B, C, C + l3 * u2


class F35_VanAubel(Board):
    """Squares stand outward on the sides of a quadrilateral ABCD, with
    centres P, Q, R, S on AB, BC, CD, DA; M is the midpoint of the diagonal
    AC. A quarter-turn about B carries A to the far corner B₁ of the square
    on AB and the corner B₂ of the square on BC to C: so AB₂ and B₁C are
    equal and perpendicular. P is the midpoint of the square's diagonal AB₁
    and M of AC, so halving towards A takes B₁C onto PM; Q is the midpoint
    of CB₂, so halving towards C takes AB₂ onto MQ. Hence MP and MQ are
    equal and perpendicular. The same with D (quarter-turn about D, halving
    towards A and C) makes MS and MR equal and perpendicular, turning the
    same way. So one quarter-turn about M carries P to Q and R to S: it
    carries PR onto QS, which are therefore equal and perpendicular. (Any
    quadrilateral; drawn convex, with no two sides parallel.)"""

    def construct(self):
        Am, Bm, Cm, Dm = _quad_from_angles(100, 54, 146, 60, 3.0, 2.4)
        sq_m = [_outer_square(Am, Bm), _outer_square(Bm, Cm),
                _outer_square(Cm, Dm), _outer_square(Dm, Am)]
        allp = np.array([q for S4 in sq_m for q in S4])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        Fr = Frame(lo[0] - 0.05, hi[0] + 0.05, lo[1] - 0.05, hi[1] + 0.05,
                   max_w=8.0, centre=(-1.75, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D = (Fr.P(X) for X in (Am, Bm, Cm, Dm))
        SAB, SBC, SCD, SDA = (_outer_square(*e) for e in ((A, B), (B, C), (C, D), (D, A)))
        P, Q, R, S = (sum(s4) / 4 for s4 in (SAB, SBC, SCD, SDA))
        B1, B2 = SAB[2], SBC[3]            # corners at B of the squares on AB, BC
        D1, D2 = SCD[2], SDA[3]            # corners at D of the squares on CD, DA
        M = (A + C) / 2
        qt = PI / 2
        n = np.linalg.norm

        # ---- checks
        check(area([A, B, C, D]) > 0, "A, B, C, D run counter-clockwise")
        for X1, Y1 in (((A, B), (C, D)), ((B, C), (D, A))):
            check(abs(_cross(_unit(X1[1] - X1[0]), _unit(Y1[1] - Y1[0]))) > 0.2,
                  "no two sides parallel")
        sides = sorted(n(Y - X1) for X1, Y in ((A, B), (B, C), (C, D), (D, A)))
        check(min(np.diff(sides)) > 0.1, "four different side lengths")
        for S4, nm in ((SAB, "AB"), (SBC, "BC"), (SCD, "CD"), (SDA, "DA")):
            check(not _pip((S4[0] + S4[2]) / 2, [A, B, C, D]), f"the square on {nm} is outside")
        check(close(_rot2(A, B, qt), B1) and close(_rot2(B2, B, qt), C),
              "the quarter-turn about B: A -> B₁, B₂ -> C")
        check(close(_rot2(C, D, qt), D1) and close(_rot2(D2, D, qt), A),
              "the quarter-turn about D: C -> D₁, D₂ -> A")
        check(close((A + B1) / 2, P) and close((C + B2) / 2, Q)
              and close((C + D1) / 2, R) and close((A + D2) / 2, S),
              "each centre is the midpoint of a diagonal of its square")
        check(close(A + 0.5 * (B1 - A), P) and close(A + 0.5 * (C - A), M)
              and close(A + 0.5 * (D2 - A), S), "halving towards A: B₁ -> P, C -> M, D₂ -> S")
        check(close(C + 0.5 * (B2 - C), Q) and close(C + 0.5 * (A - C), M)
              and close(C + 0.5 * (D1 - C), R), "halving towards C: B₂ -> Q, A -> M, D₁ -> R")
        check(close(_rot2(P, M, qt), Q) and close(_rot2(R, M, qt), S),
              "the quarter-turn about M: P -> Q, R -> S")
        check(abs(n(R - P) - n(S - Q)) < 1e-9 and abs(np.dot(R - P, S - Q)) < 1e-9,
              "PR = QS and PR ⊥ QS")
        X = _meet(P, R, Q, S)
        check(_between(X, P, R) and _between(X, Q, S), "PR and QS cross")

        # ---- the pieces of every step (built first: labels avoid them all)
        cQ, cSq = BLUE_D, GREY_D
        cB, cD, cPR, cQS = GREEN_C, PURPLE_B, ORANGE, BLUE_B
        quad = mk([A, B, C, D], cQ, 0.55, stroke_width=4)
        sqs = VGroup(*[mk(s4, cSq, 0.45, stroke_color=GREY_B, stroke_width=2)
                       for s4 in (SAB, SBC, SCD, SDA)])
        diag_all = VGroup(*[DashedLine(s4[i], s4[i + 2], color=GREY_B, stroke_width=1.5,
                                       dash_length=0.07)
                            for s4 in (SAB, SBC, SCD, SDA) for i in (0, 1)])
        dots = VGroup(*[Dot(Y, radius=0.06, color=WHITE) for Y in (P, Q, R, S)])
        lPR = Line(P, R, color=cPR, stroke_width=6)
        lQS = Line(Q, S, color=cQS, stroke_width=6)
        ac = DashedLine(A, C, color=WHITE, stroke_width=2.5, dash_length=0.09)
        tkM = VGroup(_ticks(A, M, 1, YELLOW_B, at=0.3), _ticks(M, C, 1, YELLOW_B, at=0.7))
        dM = Dot(M, radius=0.07, color=YELLOW_B)
        wB = VGroup(mk([B, A, B2], cB, 0.3, stroke_width=0),
                    Line(B, A, color=WHITE, stroke_width=2.5),
                    Line(B, B2, color=WHITE, stroke_width=2.5))
        wD = VGroup(mk([D, C, D2], cD, 0.3, stroke_width=0),
                    Line(D, C, color=WHITE, stroke_width=2.5),
                    Line(D, D2, color=WHITE, stroke_width=2.5))
        chB0 = Line(A, B2, color=cB, stroke_width=6)      # stays, then halves onto MQ
        chB1 = Line(A, B2, color=cB, stroke_width=6)      # turns about B onto B₁C
        chD0 = Line(C, D2, color=cD, stroke_width=6)      # stays, then halves onto MS
        chD1 = Line(C, D2, color=cD, stroke_width=6)      # turns about D onto D₁A
        arB = _turn_arrow(B, _dir(B, A), _dir(B, B1), r=0.55)
        arD = _turn_arrow(D, _dir(D, C), _dir(D, D1), r=0.55)
        gA = VGroup(DashedLine(A, B1, color=YELLOW_B, stroke_width=2.5, dash_length=0.08),
                    DashedLine(A, D2, color=YELLOW_B, stroke_width=2.5, dash_length=0.08))
        gC = VGroup(DashedLine(C, B2, color=YELLOW_B, stroke_width=2.5, dash_length=0.08),
                    DashedLine(C, D1, color=YELLOW_B, stroke_width=2.5, dash_length=0.08))
        mk_M = VGroup(_ra(M, P, Q, 0.24, cB, 3.5), _ra(M, R, S, 0.24, cD, 3.5),
                      _ticks(M, P, 1, cB, at=0.7), _ticks(M, Q, 1, cB, at=0.7),
                      _ticks(M, R, 2, cD, at=0.7), _ticks(M, S, 2, cD, at=0.7))
        mk_X = VGroup(_ra(X, P, Q, 0.22, YELLOW_B),
                      _ticks(P, R, 3, WHITE, at=0.18), _ticks(Q, S, 3, WHITE, at=0.18))
        arM = _turn_arrow(M, _dir(M, P), _dir(M, Q), r=0.8, color=WHITE)
        landed_w = VGroup(mk([B, B1, C], cB, 0.3, stroke_width=2.5),
                          mk([D, D1, A], cD, 0.3, stroke_width=2.5),
                          Line(B1, C), Line(D1, A))
        spokes = VGroup(*[Line(M, Y) for Y in (P, Q, R, S)])
        every = _mob_segs(quad, sqs, diag_all, dots, lPR, lQS, ac, tkM, dM, wB, wD, chB0,
                          chD0, arB, arD, gA, gC, mk_M, mk_X, landed_w, spokes, arM)

        labs = {}
        gap_at = {"A": (SAB[3], SDA[2]), "B": (B1, B2), "C": (SBC[2], SCD[3]), "D": (D1, D2)}
        for nm, Y in (("A", A), ("B", B), ("C", C), ("D", D)):
            p1, p2 = gap_at[nm]                # the open corner between the two squares
            labs[nm] = _angle_label(tag(nm, 30), Y, p1, p2, 0.25, segs=every, pad=0.08,
                                    fracs=(0.5, 0.4, 0.6, 0.3, 0.7))
        for nm, Y, tw in (("P", P, [R, M, A, B1, B, SAB[3]]), ("Q", Q, [S, M, C, B2, B, SBC[2]]),
                          ("R", R, [P, M, C, D1, D, SCD[3]]), ("S", S, [Q, M, A, D2, D, SDA[2]])):
            labs[nm] = _park(tag(nm, 30), Y, every, towards=tw, d0=0.22, d1=0.9,
                             avoid=list(labs.values()))
        # M's label below M, on either side of AC: above M runs QS, which
        # passes between M and the crossing X of PR and QS
        late = _mob_segs(quad, sqs, dots, lPR, lQS, ac, tkM, dM, spokes, mk_M, mk_X, arM)
        labs["M"] = _park(tag("M", 24, YELLOW_B), M, late, d0=0.3, d1=0.8, pad=0.05,
                          dirs=(260, 256, 264, 252, 268, 248, 272),
                          avoid=list(labs.values()))
        check(n(labs["M"].get_center() - M) < n(labs["M"].get_center() - X) - 0.2,
              "M's label is nearer M than the crossing X")

        # the ledger, right of the figure
        lx = 2.45
        rows = [_trow([("MP = MQ,", cB), ("MP ⊥ MQ", cB)], 26, 0.22),
                _trow([("MR = MS,", cD), ("MR ⊥ MS", cD)], 26, 0.22),
                _trow([("P → Q,", WHITE), ("R → S", WHITE)], 26, 0.22)]
        for i, r in enumerate(rows):
            r.move_to([0.0, 1.9 - 0.75 * i, 0.0]).align_to(np.array([lx, 0.0, 0.0]), LEFT)
        check(rows[0].get_left()[0] > max(q[0] for s4 in (SAB, SBC, SCD, SDA) for q in s4) + 0.3,
              "the ledger is clear of the figure")
        _all_in_frame(rows, "F35 ledger")

        # ---- the quadrilateral, the squares, their centres; the claim
        self.play(FadeIn(quad), *[FadeIn(labs[k]) for k in "ABCD"], run_time=1.0)
        self.add(sqs)
        self.bring_to_back(sqs)
        self.play(FadeIn(sqs), run_time=0.9)
        self.play(Create(diag_all), run_time=0.7)
        self.play(FadeIn(dots), *[FadeIn(labs[k]) for k in "PQRS"], run_time=0.6)
        self.play(FadeOut(diag_all), Create(lPR), Create(lQS), run_time=0.9)
        self.bring_to_front(dots)
        self.hold(0.6)
        self.play(lPR.animate.set_stroke(opacity=0.25), lQS.animate.set_stroke(opacity=0.25),
                  run_time=0.5)

        # ---- M, the midpoint of the diagonal AC
        self.play(Create(ac), FadeIn(dM), FadeIn(tkM), run_time=0.8)

        # ---- quarter-turns about B and about D
        self.add(wB, wD)
        self.bring_to_back(wB, wD)
        self.bring_to_back(sqs)
        self.play(FadeIn(wB), FadeIn(wD), FadeIn(chB0), FadeIn(chD0), Create(arB), Create(arD),
                  run_time=0.7)
        mvB, mvD = VGroup(wB.copy(), chB1), VGroup(wD.copy(), chD1)
        self.add(mvB, mvD)
        self.bring_to_front(*[labs[k] for k in "ABCDPQRS"])
        self.play(Rotate(mvB, angle=qt, about_point=B), Rotate(mvD, angle=qt, about_point=D),
                  run_time=1.9)
        check(_landed(mvB[0][0], [B, B1, C]) and close(chB1.get_start(), B1, 1e-6)
              and close(chB1.get_end(), C, 1e-6), "about B: A lands on B₁, B₂ on C")
        check(_landed(mvD[0][0], [D, D1, A]) and close(chD1.get_start(), D1, 1e-6)
              and close(chD1.get_end(), A, 1e-6), "about D: C lands on D₁, D₂ on A")
        _labels_ok([labs[k] for k in "ABCDPQRS"], _mob_segs(quad, sqs, dots, ac, tkM, dM,
                                                            wB, wD, mvB, mvD, arB, arD),
                   "F35 turns")
        self.hold(0.3)
        self.play(FadeOut(arB), FadeOut(arD), FadeOut(wB), FadeOut(wD), FadeOut(mvB[0]),
                  FadeOut(mvD[0]), run_time=0.6)

        # ---- halving towards A, then towards C
        self.play(Create(gA), run_time=0.5)
        self.play(_homothety(chB1, A, 0.5), _homothety(chD0, A, 0.5), run_time=1.5)
        check(close(chB1.get_start(), P, 1e-6) and close(chB1.get_end(), M, 1e-6),
              "B₁C halved towards A lands on PM")
        check(close(chD0.get_start(), M, 1e-6) and close(chD0.get_end(), S, 1e-6),
              "CD₂ halved towards A lands on MS")
        self.play(FadeOut(gA), Create(gC), run_time=0.6)
        self.play(_homothety(chB0, C, 0.5), _homothety(chD1, C, 0.5), run_time=1.5)
        check(close(chB0.get_start(), M, 1e-6) and close(chB0.get_end(), Q, 1e-6),
              "AB₂ halved towards C lands on MQ")
        check(close(chD1.get_start(), R, 1e-6) and close(chD1.get_end(), M, 1e-6),
              "D₁A halved towards C lands on RM")
        self.play(FadeOut(gC), FadeIn(mk_M), run_time=0.6)
        self.bring_to_front(dots, dM)
        _labels_ok([labs[k] for k in "ABCDPQRS"], _mob_segs(quad, sqs, dots, ac, tkM, dM, chB0, chB1,
                                                   chD0, chD1, mk_M), "F35 spokes")
        self.play(FadeIn(labs["M"]), FadeIn(rows[0], shift=LEFT * 0.2),
                  FadeIn(rows[1], shift=LEFT * 0.2), run_time=0.8)
        self.hold(0.5)

        # ---- one quarter-turn about M carries MP onto MQ and MR onto MS,
        # so it carries P to Q, R to S, and PR onto QS
        turn = VGroup(Line(M, P, color=cB, stroke_width=7), Line(M, R, color=cD, stroke_width=7),
                      Line(P, R, color=cPR, stroke_width=7))
        self.play(FadeOut(mk_M), FadeIn(turn), Create(arM), run_time=0.6)
        self.bring_to_front(dots, dM, *labs.values())
        self.play(Rotate(turn, angle=qt, about_point=M), run_time=2.0)
        check(close(turn[0].get_end(), Q, 1e-6) and close(turn[1].get_end(), S, 1e-6)
              and close(turn[2].get_start(), Q, 1e-6) and close(turn[2].get_end(), S, 1e-6),
              "turned about M: P lands on Q, R on S, PR on QS")
        self.play(FadeIn(rows[2], shift=LEFT * 0.2), run_time=0.6)
        self.hold(0.3)
        faint = [chB0, chB1, chD0, chD1]
        self.play(FadeOut(turn), FadeOut(arM), FadeOut(tkM), FadeOut(ac),
                  *[m.animate.set_stroke(opacity=0.3) for m in faint],
                  lPR.animate.set_stroke(opacity=1.0), lQS.animate.set_stroke(opacity=1.0),
                  FadeIn(mk_X), run_time=0.8)
        self.bring_to_front(lPR, lQS, mk_X, dots, dM, *labs.values())

        cap = caption("PR = QS  and  PR ⊥ QS", 36)
        _final_check(self, cap, "F35")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F36 SQUARES ON A PARALLELOGRAM

class F36_ParallelogramSquares(Board):
    """Squares stand outward on the sides of a parallelogram ABCD, with
    centres P, Q, R, S on AB, BC, CD, DA. Each centre makes a triangle with
    the next centre and their shared corner: PBQ, QCR, RDS, SAP. A
    quarter-turn about P carries B to A (P is the centre of the square on
    AB) and carries PBQ exactly onto PAS: PB = PA, BQ = AS (half-diagonals
    of the squares on the opposite, equal sides BC and DA), and the angles
    at B and A are both 90° + ∠B. So PQ turns onto PS: PQ = PS and
    PQ ⊥ PS. The same quarter-turn about Q, R and S carries each corner
    triangle onto the one before it, so all four sides of PQRS are equal
    and all four angles are right: PQRS is a square. Its centre is the
    parallelogram's centre O (the half-turn about O swaps the opposite
    squares), and a quarter-turn about O permutes P, Q, R, S. (Any
    parallelogram; drawn with ∠B = 56°.)"""

    def construct(self):
        Bd, ab, bc = 56, 1.0, 1.55
        Bm = np.array([0.0, 0.0])
        Cm = np.array([bc, 0.0])
        Am = ab * np.array([np.cos(Bd * DEGREES), np.sin(Bd * DEGREES)])
        Dm = Am + Cm - Bm
        sq_m = [_outer_square(Am, Bm), _outer_square(Bm, Cm),
                _outer_square(Cm, Dm), _outer_square(Dm, Am)]
        allp = np.array([q for S4 in sq_m for q in S4])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        Fr = Frame(lo[0] - 0.04, hi[0] + 0.04, lo[1] - 0.04, hi[1] + 0.04,
                   max_w=8.0, centre=(-1.7, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D = (Fr.P(X) for X in (Am, Bm, Cm, Dm))
        SAB, SBC, SCD, SDA = (_outer_square(*e) for e in ((A, B), (B, C), (C, D), (D, A)))
        P, Q, R, S = (sum(s4) / 4 for s4 in (SAB, SBC, SCD, SDA))
        O = (A + C) / 2
        qt = PI / 2
        n = np.linalg.norm
        corner = {"P": [P, B, Q], "Q": [Q, C, R], "R": [R, D, S], "S": [S, A, P]}
        target = {"P": [P, A, S], "Q": [Q, B, P], "R": [R, C, Q], "S": [S, D, R]}
        ctr = {"P": P, "Q": Q, "R": R, "S": S}

        # ---- checks
        check(area([A, B, C, D]) > 0 and close(A + C, B + D), "ABCD is a parallelogram")
        check(abs(n(B - A) - n(C - B)) > 0.2 and abs(_angle(B, A, C) - PI / 2) > 0.3,
              "neither a rhombus nor a rectangle")
        for S4, nm in ((SAB, "AB"), (SBC, "BC"), (SCD, "CD"), (SDA, "DA")):
            check(not _pip((S4[0] + S4[2]) / 2, [A, B, C, D]), f"the square on {nm} is outside")
        for k in "PQRS":
            img = [_rot2(x, ctr[k], qt) for x in corner[k]]
            check(all(close(a, b) for a, b in zip(img, target[k])),
                  f"the quarter-turn about {k} lands the corner triangle on the next")
        check(abs(n(B - P) - n(A - P)) < 1e-9 and abs(np.dot(B - P, A - P)) < 1e-9,
              "P sees AB at a right angle, PA = PB")
        check(abs(n(Q - B) - n(S - A)) < 1e-9, "BQ = AS (opposite sides are equal)")
        for X, Y in ((P, Q), (Q, R), (R, S), (S, P)):
            check(close(_rot2(X, O, qt), Y), "the quarter-turn about O permutes P, Q, R, S")
        check(close((P + R) / 2, O) and close((Q + S) / 2, O), "O is the centre of PQRS")

        # ---- pieces (built first: labels avoid them all)
        cPar, cSq, cT, cM = BLUE_D, GREY_D, TEAL_D, ORANGE
        par = mk([A, B, C, D], cPar, 0.55, stroke_width=4)
        sqs = VGroup(*[mk(s4, cSq, 0.45, stroke_color=GREY_B, stroke_width=2)
                       for s4 in (SAB, SBC, SCD, SDA)])
        diag_all = VGroup(*[DashedLine(s4[i], s4[i + 2], color=GREY_B, stroke_width=1.5,
                                       dash_length=0.07)
                            for s4 in (SAB, SBC, SCD, SDA) for i in (0, 1)])
        dots = VGroup(*[Dot(Y, radius=0.06, color=WHITE) for Y in (P, Q, R, S)])
        claim = DashedVMobject(Polygon(P, Q, R, S, stroke_color=YELLOW_B, stroke_width=3),
                               num_dashes=60)
        tris = {k: mk(corner[k], cT, 0.45, stroke_color=WHITE, stroke_width=2) for k in "PQRS"}
        cps = {k: mk(corner[k], cM, 0.75, stroke_color=YELLOW_B, stroke_width=3) for k in "PQRS"}
        arrows = {k: _turn_arrow(ctr[k], _dir(ctr[k], corner[k][1]), _dir(ctr[k], target[k][1]),
                                 r=0.42) for k in "PQRS"}
        sqPQRS = Polygon(P, Q, R, S, stroke_color=YELLOW_B, stroke_width=5)
        tks = VGroup(*[_ticks(X, Y, 1, YELLOW_B, size=0.15, width=3.5)
                       for X, Y in ((P, Q), (Q, R), (R, S), (S, P))])
        ras = VGroup(_ra(P, Q, S, 0.2, YELLOW_B, 3), _ra(Q, R, P, 0.2, YELLOW_B, 3),
                     _ra(R, S, Q, 0.2, YELLOW_B, 3), _ra(S, P, R, 0.2, YELLOW_B, 3))
        dO = Dot(O, radius=0.06, color=YELLOW_B)
        arO = _turn_arrow(O, _dir(O, P), _dir(O, Q), r=0.5, color=WHITE)
        every = _mob_segs(par, sqs, diag_all, dots, claim, *tris.values(), *arrows.values(),
                          sqPQRS, tks, ras, dO, arO)

        labs = {}
        gap_at = {"A": (SAB[3], SDA[2]), "B": (SAB[2], SBC[3]), "C": (SBC[2], SCD[3]),
                  "D": (SCD[2], SDA[3])}
        for nm, Y in (("A", A), ("B", B), ("C", C), ("D", D)):
            p1, p2 = gap_at[nm]
            labs[nm] = _angle_label(tag(nm, 30), Y, p1, p2, 0.25, segs=every, pad=0.08,
                                    fracs=(0.5, 0.4, 0.6, 0.3, 0.7))
        far = {"P": (SAB[2], SAB[3]), "Q": (SBC[2], SBC[3]), "R": (SCD[2], SCD[3]),
               "S": (SDA[2], SDA[3])}
        for k in "PQRS":                          # in the outer half of each square
            f1, f2 = far[k]
            out = _unit((f1 + f2) / 2 - ctr[k])
            base = np.degrees(np.arctan2(out[1], out[0]))
            labs[k] = _park(tag(k, 30), ctr[k], every, d0=0.25, d1=0.9,
                            dirs=[base + t for t in (0, 25, -25, 50, -50)],
                            avoid=list(labs.values()))
        labs["O"] = _park(tag("O", 26, YELLOW_B), O, every, d0=0.22, d1=0.9,
                          dirs=(90, 70, 110, 50, 130), avoid=list(labs.values()))

        # ---- the parallelogram, its squares, their centres; the claim
        self.play(FadeIn(par), *[FadeIn(labs[k]) for k in "ABCD"], run_time=1.0)
        self.add(sqs)
        self.bring_to_back(sqs)
        self.play(FadeIn(sqs), run_time=0.9)
        self.play(Create(diag_all), run_time=0.7)
        self.play(FadeIn(dots), *[FadeIn(labs[k]) for k in "PQRS"], run_time=0.6)
        self.play(FadeOut(diag_all), Create(claim), run_time=0.9)
        self.hold(0.4)

        # ---- the four corner triangles
        self.add(*tris.values())
        self.bring_to_front(claim, dots, *[labs[k] for k in "ABCDPQRS"])
        self.play(*[FadeIn(t) for t in tris.values()], run_time=0.7)

        # ---- a quarter-turn about P: PBQ lands on PAS
        self.add(cps["P"])
        self.bring_to_front(dots, *[labs[k] for k in "ABCDPQRS"])
        self.play(FadeIn(cps["P"]), Create(arrows["P"]), run_time=0.5)
        self.play(Rotate(cps["P"], angle=qt, about_point=P), run_time=1.8)
        check(_landed(cps["P"], target["P"]), "PBQ, turned about P, lands vertex by vertex on PAS")
        self.play(FadeOut(arrows["P"]), FadeIn(tks[0]), FadeIn(tks[3]), Create(ras[0]),
                  run_time=0.7)
        _labels_ok([labs[k] for k in "ABCDPQRS"],
                   _mob_segs(par, sqs, dots, claim, *tris.values(), cps["P"], tks, ras),
                   "F36 at P")
        self.hold(0.4)
        self.play(FadeOut(cps["P"]), run_time=0.4)

        # ---- the same quarter-turn about Q, R and S
        rest = "QRS"
        for k in rest:
            self.add(cps[k])
        self.bring_to_front(dots, *[labs[k] for k in "ABCDPQRS"])
        self.play(*[FadeIn(cps[k]) for k in rest], *[Create(arrows[k]) for k in rest],
                  run_time=0.5)
        self.play(*[Rotate(cps[k], angle=qt, about_point=ctr[k]) for k in rest], run_time=1.8)
        for k in rest:
            check(_landed(cps[k], target[k]),
                  f"the corner triangle turned about {k} lands vertex by vertex")
        self.play(*[FadeOut(arrows[k]) for k in rest], FadeIn(tks[1]), FadeIn(tks[2]),
                  Create(ras[1]), Create(ras[2]), Create(ras[3]), run_time=0.7)
        self.hold(0.4)
        self.play(*[FadeOut(cps[k]) for k in rest], FadeOut(claim), Create(sqPQRS),
                  *[t.animate.set_fill(opacity=0.2) for t in tris.values()], run_time=0.8)
        self.bring_to_front(tks, ras, dots, *[labs[k] for k in "ABCDPQRS"])

        # ---- its centre is O, and a quarter-turn about O permutes P, Q, R, S
        spin = Polygon(P, Q, R, S, stroke_color=WHITE, stroke_width=3)
        self.play(FadeIn(dO), FadeIn(labs["O"]), Create(arO), FadeIn(spin), run_time=0.6)
        self.play(Rotate(spin, angle=qt, about_point=O), run_time=1.5)
        check(_landed(spin, [Q, R, S, P]), "turned about O, PQRS lands on itself: P -> Q -> R -> S")
        self.play(FadeOut(spin), FadeOut(arO), run_time=0.4)

        cap = caption("PQRS is a square", 36)
        _final_check(self, cap, "F36")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F47 THE SQUARE IN A RIGHT TRIANGLE

class F47_SquareInRightTriangle(Board):
    """The right triangle ABC has legs a = BC, b = CA and hypotenuse c; the
    square CEDF sits in the right angle with its far corner D on AB, side s.
    Its top side FD is parallel to CA, so the corner triangle FDB is ABC
    shrunk towards B: its leg along CA's direction is s where ABC's is b,
    so every length is multiplied by s/b and BD = (s/b)·c. Its right side
    ED is parallel to CB, so EAD is ABC shrunk towards A with factor s/a:
    DA = (s/a)·c. The two pieces make up the hypotenuse: (s/b)·c + (s/a)·c
    = c, so s/b + s/a = 1 and s = ab/(a + b). (Any right triangle; drawn
    with a ≠ b.)"""

    def construct(self):
        am, bm = 3.5, 4.5
        sm = am * bm / (am + bm)
        Fr = Frame(-0.55, bm + 0.75, -0.5, am + 0.62, max_w=8.0,
                   centre=(-2.55, (SAFE_TOP + SAFE_BOTTOM) / 2))
        C, B, A = Fr.P((0.0, 0.0)), Fr.P((0.0, am)), Fr.P((bm, 0.0))
        E, D, F = Fr.P((sm, 0.0)), Fr.P((sm, sm)), Fr.P((0.0, sm))
        n = np.linalg.norm
        a, b, c, s = n(B - C), n(A - C), n(B - A), n(E - C)
        k1, k2 = s / b, s / a

        # ---- checks
        check(abs(np.dot(B - C, A - C)) < 1e-9 and abs(a - b) > 0.5, "a right angle at C, a ≠ b")
        check(_on_line(D, A, B) and _between(D, A, B), "D lies on the hypotenuse")
        for X, Y, Z in ((C, E, D), (E, D, F), (D, F, C), (F, C, E)):
            check(abs(np.dot(X - Y, Z - Y)) < 1e-9, "CEDF has right angles")
        check(abs(n(E - C) - n(F - C)) < 1e-9 and abs(n(D - E) - s) < 1e-9, "CEDF is a square")
        check(abs(_cross(D - F, A - C)) < 1e-9 and abs(_cross(D - E, B - C)) < 1e-9,
              "FD ∥ CA and ED ∥ CB")
        check(close(B + k1 * (C - B), F) and close(B + k1 * (A - B), D),
              "ABC shrunk towards B by s/b is FDB")
        check(close(A + k2 * (C - A), E) and close(A + k2 * (B - A), D),
              "ABC shrunk towards A by s/a is EAD")
        check(abs(n(D - B) - k1 * c) < 1e-9 and abs(n(A - D) - k2 * c) < 1e-9,
              "BD = (s/b)c and DA = (s/a)c")
        check(abs(k1 + k2 - 1) < 1e-12 and abs(s - a * b / (a + b)) < 1e-9,
              "s/b + s/a = 1 and s = ab/(a + b)")

        # ---- pieces
        cT, cS, c1, c2 = BLUE_D, GOLD_D, TEAL_D, ORANGE
        tri = mk([A, B, C], cT, 0.55, stroke_width=4)
        raC = _ra(C, A, B, 0.22)
        sqr = mk([C, E, D, F], cS, 0.8, stroke_width=3)
        dD = Dot(D, radius=0.07, color=WHITE)
        t1 = mk([F, D, B], c1, 0.75, stroke_width=2)
        t2 = mk([E, A, D], c2, 0.75, stroke_width=2)
        # copies of ABC, vertex order starting at the centre of the shrinking
        cp1 = VGroup(mk([B, C, A], c1, 0.45, stroke_color=TEAL_B, stroke_width=3),
                     Line(C, A, color=TEAL_B, stroke_width=8))     # the leg that becomes FD
        cp2 = VGroup(mk([A, C, B], c2, 0.45, stroke_color=YELLOW_B, stroke_width=3),
                     Line(C, B, color=YELLOW_B, stroke_width=8))   # the leg that becomes ED
        hBD = Line(B, D, color=TEAL_B, stroke_width=8)
        hDA = Line(D, A, color=YELLOW_B, stroke_width=8)
        out = _unit(_perp(A - B))                   # away from C across AB
        if np.dot(out, C - B) > 0:
            out = -out
        # c names the whole hypotenuse BD + DA: a bracket spanning it
        cbar, lc2 = _bracket(B, A, out, "c", gap=1.12, color=WHITE, size=30)
        allm = [tri, raC, sqr, dD, t1, t2, hBD, hDA, cbar]
        strokes = _mob_segs(*allm, Line(F, D), Line(E, D))

        la = _park_beside(tag("a", 30), C, B, strokes, side=LEFT)
        lb = _park_beside(tag("b", 30), C, A, strokes, side=DOWN)
        lc = lc2
        ls1 = _park_beside(tag("s", 28), F, D, strokes, side=DOWN, ats=(0.5, 0.42, 0.58))
        ls2 = _park_beside(tag("s", 28), E, D, strokes, side=LEFT, ats=(0.5, 0.42, 0.58))
        lA = _park(tag("A", 30), A, strokes, dirs=(-30, -10, -50), d0=0.25)
        lB = _park(tag("B", 30), B, strokes, dirs=(180, 160, 200, 140), d0=0.25)
        lC = _park(tag("C", 30), C, strokes, dirs=(225, 205, 245), d0=0.25)
        dout = np.degrees(np.arctan2(out[1], out[0]))
        lD = _park(tag("D", 30), D, strokes, dirs=[dout + t for t in (0, 12, -12, 24, -24)],
                   d0=0.2, d1=0.5)
        l1 = _park_beside(tag("(s/b)·c", 24, TEAL_B), B, D, strokes, side=out,
                          ats=(0.5, 0.45, 0.55, 0.4, 0.6), gaps=(0.13, 0.15, 0.17),
                          avoid=[lB, lD, lc2], pad=0.07)
        l2 = _park_beside(tag("(s/a)·c", 24, YELLOW_B), D, A, strokes, side=out,
                          ats=(0.5, 0.45, 0.55, 0.4, 0.6), gaps=(0.13, 0.15, 0.17),
                          avoid=[lA, lD, lc2, l1], pad=0.07)
        for m in (l1, l2):
            check(n(m.get_center() - (hBD.get_center() if m is l1 else hDA.get_center()))
                  < 0.75, "each piece label sits at its piece")

        # the ledger
        lx = 1.95
        rows = [_trow([("BD = (s/b)·c", TEAL_B)], 30),
                _trow([("DA = (s/a)·c", YELLOW_B)], 30),
                _trow([("(s/b)·c + (s/a)·c = c", WHITE)], 30),
                _trow([("s/b + s/a = 1", WHITE)], 30)]
        for i, r in enumerate(rows):
            r.move_to([0.0, 2.6 - 0.95 * i, 0.0]).align_to(np.array([lx, 0.0, 0.0]), LEFT)
        check(min(r.get_left()[0] for r in rows) > max(
            m.get_right()[0] for m in (l2, lc2, lA, tri, cbar)) + 0.2
              or min(r.get_bottom()[1] for r in rows[:2]) > max(m.get_top()[1] for m in (l2, lc2, cbar)),
              "the ledger is clear of the figure")
        _all_in_frame(rows, "F47 ledger")
        for r in rows:
            check(_hit(r, strokes, 0.1) is None, "the ledger is clear of the figure's lines")

        # ---- the triangle and the square
        self.play(FadeIn(tri), Create(raC), FadeIn(cbar),
                  *[FadeIn(m) for m in (la, lb, lc, lA, lB, lC)], run_time=1.2)
        self.play(FadeIn(sqr), FadeIn(dD), FadeIn(lD), FadeIn(ls1), FadeIn(ls2), run_time=0.9)
        self.add(t1, t2)
        self.bring_to_back(t1, t2)
        self.bring_to_back(tri)
        self.play(FadeIn(t1), FadeIn(t2), run_time=0.6)
        self.bring_to_front(sqr, dD, raC, ls1, ls2)
        self.hold(0.3)

        # ---- ABC shrunk towards B: its leg b becomes FD = s
        self.add(cp1)
        self.play(FadeIn(cp1), run_time=0.5)
        self.play(_homothety(cp1, B, k1), run_time=2.0)
        check(_landed(cp1[0], [B, F, D]) and close(cp1[1].get_start(), F, 1e-6)
              and close(cp1[1].get_end(), D, 1e-6), "shrunk towards B, ABC lands on FDB")
        self.play(FadeOut(cp1), Create(hBD), FadeIn(l1), run_time=0.8)
        self.play(FadeIn(rows[0], shift=LEFT * 0.2), run_time=0.6)
        self.hold(0.3)

        # ---- ABC shrunk towards A: its leg a becomes ED = s
        self.add(cp2)
        self.play(FadeIn(cp2), run_time=0.5)
        self.play(_homothety(cp2, A, k2), run_time=2.0)
        check(_landed(cp2[0], [A, E, D]) and close(cp2[1].get_start(), E, 1e-6)
              and close(cp2[1].get_end(), D, 1e-6), "shrunk towards A, ABC lands on EAD")
        self.play(FadeOut(cp2), Create(hDA), FadeIn(l2), run_time=0.8)
        self.play(FadeIn(rows[1], shift=LEFT * 0.2), run_time=0.6)
        self.hold(0.3)

        # ---- the two pieces make up the hypotenuse
        self.play(Indicate(VGroup(cbar, lc), color=YELLOW_B, scale_factor=1.06), run_time=0.8)
        self.play(FadeIn(rows[2], shift=LEFT * 0.2), run_time=0.7)
        self.play(FadeIn(rows[3], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.3)

        _labels_ok([la, lb, lc, ls1, ls2, lA, lB, lC, lD, l1, l2], strokes, "F47")
        cap = caption("s/b + s/a = 1   ⟹   s = ab / (a + b)", 36)
        _final_check(self, cap, "F47")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F54 STEWART'S THEOREM

class F54_Stewart(Board):
    """The cevian AD = d cuts BC = a into BD = m and DC = n; AH = h is the
    altitude. Folding the right triangle ADH over AH carries D to D′ on BC:
    HD′ = HD and AD′ = AD = d. Pythagoras on both sides of the foot H, with
    the common height h: c² = h² + BH² and d² = h² + DH², so the height
    drops out, c² − d² = BH² − DH² = (BH − DH)(BH + DH) = BD · BD′ =
    m · BD′. On the other side, with AD′ = d, b² − d² = CH² − D′H² =
    (CH − HD′)(CH + HD) = D′C · n. Since BD′ + D′C = a, multiplying the
    first by n and the second by m and adding gives n(c² − d²) +
    m(b² − d²) = amn, which is b²m + c²n = a(d² + mn). (Drawn with H
    between D and C and D′ on DC; with signed lengths the same computation
    covers every position of H.)"""

    def construct(self):
        Am, Bm, Cm = np.array([2.6, 4.3]), np.array([0.0, 0.0]), np.array([7.0, 0.0])
        Dm = np.array([1.4, 0.0])
        Fr = Frame(-0.35, 7.35, -1.95, 4.75, max_w=7.6,
                   centre=(-2.55, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D = (Fr.P(X) for X in (Am, Bm, Cm, Dm))
        H = _foot(A, B, C)
        Dp = 2 * H - D                                   # D folded over AH
        n_ = np.linalg.norm
        a, b, c, d = n_(C - B), n_(C - A), n_(B - A), n_(D - A)
        m, n = n_(D - B), n_(C - D)
        h, x = n_(A - H), n_(H - D)

        # ---- checks
        check(_scalene(A, B, C) and max(_angle(A, B, C), _angle(B, C, A),
                                        _angle(C, A, B)) < PI / 2 - 0.15,
              "scalene acute triangle, no angle near 90°")
        check(_between(D, B, H) and _between(H, D, Dp) and _between(Dp, H, C),
              "B, D, H, D′, C lie in this order")
        check(close(_mirror(D, A, H), Dp) and abs(n_(Dp - A) - d) < 1e-9,
              "the fold over AH carries D to D′: AD′ = AD")
        check(abs(c * c - (h * h + n_(H - B) ** 2)) < 1e-9 and abs(d * d - (h * h + x * x)) < 1e-9
              and abs(b * b - (h * h + n_(C - H) ** 2)) < 1e-9, "Pythagoras about the foot H")
        check(abs((n_(H - B) - x) - m) < 1e-9 and abs((n_(H - B) + x) - n_(Dp - B)) < 1e-9,
              "BH − DH = BD = m, BH + DH = BD′")
        check(abs((n_(C - H) - x) - n_(C - Dp)) < 1e-9 and abs((n_(C - H) + x) - n) < 1e-9,
              "CH − D′H = D′C, CH + D′H = DC = n")
        check(abs(c * c - d * d - m * n_(Dp - B)) < 1e-9 and abs(b * b - d * d - n * n_(C - Dp)) < 1e-9,
              "c² − d² = m·BD′ and b² − d² = n·D′C")
        check(abs(n_(Dp - B) + n_(C - Dp) - a) < 1e-9, "BD′ + D′C = a")
        check(abs(b * b * m + c * c * n - a * (d * d + m * n)) < 1e-9, "Stewart")

        # ---- pieces
        cT, cB, cC, cD = BLUE_D, TEAL_B, ORANGE, YELLOW_B
        tri = mk([A, B, C], cT, 0.45, stroke_width=4)
        ad = Line(A, D, color=cD, stroke_width=4)
        adp = DashedLine(A, Dp, color=cD, stroke_width=3, dash_length=0.09)
        ah = DashedLine(A, H, color=WHITE, stroke_width=3, dash_length=0.09)
        raH = _ra(H, A, C, 0.18)
        dots = VGroup(*[Dot(X, radius=0.055, color=WHITE) for X in (D, H, Dp)])
        tkx = VGroup(_ticks(D, H, 1, WHITE), _ticks(H, Dp, 1, WHITE))
        down = np.array([0.0, -1.0, 0.0])
        mbar, lm = _bracket(B, D, down, "m", gap=0.62, color=WHITE, size=28)
        nbar, ln = _bracket(D, C, down, "n", gap=0.62, color=WHITE, size=28)
        abar, la = _bracket(B, C, down, "a", gap=1.22, color=WHITE, size=28)
        hl = {"ABH": mk([A, B, H], cB, 0.35, stroke_color=cB, stroke_width=4),
              "ADH": mk([A, D, H], cD, 0.35, stroke_color=cD, stroke_width=4),
              "ACH": mk([A, C, H], cC, 0.35, stroke_color=cC, stroke_width=4),
              "ADpH": mk([A, Dp, H], cD, 0.35, stroke_color=cD, stroke_width=4)}
        hbold = Line(A, H, color=WHITE, stroke_width=7)
        barBD = Line(B, D, color=cB, stroke_width=9)
        barBDp = Line(B, Dp, color=cB, stroke_width=9)
        barDpC = Line(Dp, C, color=cC, stroke_width=9)
        barDC = Line(D, C, color=cC, stroke_width=9)
        strokes = _mob_segs(tri, ad, adp, ah, raH, dots, tkx, mbar, nbar, abar)

        def away(P_, Q_, R_):
            """Unit normal of the line P_Q_ on the side away from R_."""
            nrm = _unit(_perp(Q_ - P_))
            return -nrm if np.dot(nrm, R_ - P_) > 0 else nrm

        lA = _park(tag("A", 30), A, strokes, dirs=(90, 70, 110), d0=0.25)
        lb = _park_beside(tag("b", 30), A, C, strokes, side=away(A, C, B))
        lc = _park_beside(tag("c", 30), B, A, strokes, side=away(B, A, C))
        lbase = {}
        avoid = [lA, lb, lc]
        for nm, X in (("B", B), ("D", D), ("H", H), ("D′", Dp), ("C", C)):
            lbase[nm] = _park(tag(nm, 28), X, strokes, dirs=(270, 255, 285, 240, 300), d0=0.24,
                              d1=0.5, avoid=avoid)
            avoid.append(lbase[nm])
        ld = _angle_label(tag("d", 30, cD), D, A, B, 0.6, segs=strokes, fracs=(0.5, 0.35, 0.65),
                          avoid=avoid)
        lh = _park_beside(tag("h", 30), A, H, strokes, side=RIGHT, ats=(0.62, 0.55, 0.7),
                          avoid=avoid + [ld])
        labels = [lA, lb, lc, ld, lh, lm, ln, la, *lbase.values()]
        _labels_ok(labels, strokes, "F54 figure")

        # the ledger: continuation lines put their "=" under the first one
        lx, fs, gap = 1.35, 26, 0.24
        r1 = _trow([("c² − d²", cB), ("=", cB), ("BH² − DH²", cB)], fs, gap)
        r2 = _trow([("=", cB), ("m · BD′", cB)], fs, gap)
        r3 = _trow([("b² − d²", cC), ("=", cC), ("CH² − D′H²", cC)], fs, gap)
        r4 = _trow([("=", cC), ("n · D′C", cC)], fs, gap)
        r5 = _trow([("BD′ + D′C", WHITE), ("=", WHITE), ("a", WHITE)], fs, gap)
        r6 = _trow([("n(c² − d²) + m(b² − d²)", YELLOW_B)], fs, gap)
        r7 = _trow([("=", YELLOW_B), ("amn", YELLOW_B)], fs, gap)
        rows = [r1, r2, r3, r4, r5, r6, r7]
        y = 3.25
        for i, r in enumerate(rows):
            r.shift(np.array([lx - r.get_left()[0], y - r[0].get_bottom()[1] * 0 - r.get_center()[1], 0.0]))
            y -= 0.6 if i not in (1, 3, 4) else 0.78
        for top, cont in ((r1, r2), (r3, r4), (r1, r7)):
            cont.shift(np.array([top[1].get_center()[0] - cont[0].get_center()[0], 0.0, 0.0]))
        _all_in_frame(rows, "F54 ledger")
        for r in rows:
            check(_hit(r, strokes + _segs(A, B, C, closed=True), 0.15) is None
                  and all(not _overlap(r, l, 0.1) for l in labels), "the ledger is clear of the figure")

        # ---- the triangle, the cevian, the altitude
        self.play(FadeIn(tri), FadeIn(lA), FadeIn(lb), FadeIn(lc), FadeIn(lbase["B"]),
                  FadeIn(lbase["C"]), run_time=1.1)
        self.play(Create(ad), FadeIn(dots[0]), FadeIn(lbase["D"]), FadeIn(ld),
                  FadeIn(mbar), FadeIn(nbar), FadeIn(lm), FadeIn(ln), FadeIn(abar), FadeIn(la),
                  run_time=1.2)
        self.play(Create(ah), Create(raH), FadeIn(dots[1]), FadeIn(lbase["H"]), FadeIn(lh),
                  run_time=0.9)

        # ---- fold ADH over the altitude: D lands on D′
        half = mk([A, D, H], cD, 0.5, stroke_color=cD, stroke_width=3)
        self.play(FadeIn(half), run_time=0.4)
        self.play(_fold(half, A, H), run_time=1.6)
        check(_landed(half, [A, Dp, H]), "the fold lands ADH on AD′H")
        self.play(FadeOut(half), Create(adp), FadeIn(dots[2]), FadeIn(lbase["D′"]),
                  FadeIn(tkx), run_time=0.8)
        self.bring_to_front(dots, *labels)

        # ---- the B side: ABH and ADH share the height h
        self.add(hl["ABH"], hl["ADH"])
        self.play(FadeIn(hl["ABH"]), FadeIn(hl["ADH"]), Create(hbold), run_time=0.8)
        self.bring_to_front(dots, *labels)
        self.play(FadeIn(r1, shift=LEFT * 0.15), run_time=0.7)
        self.play(FadeOut(hl["ABH"]), FadeOut(hl["ADH"]), FadeOut(hbold), Create(barBD),
                  run_time=0.6)
        self.play(ReplacementTransform(barBD, barBDp), run_time=0.8)
        self.play(FadeIn(r2, shift=LEFT * 0.15), run_time=0.6)
        self.hold(0.3)
        self.play(FadeOut(barBDp), run_time=0.3)

        # ---- the C side: ACH and AD′H share the height h
        self.add(hl["ACH"], hl["ADpH"])
        self.play(FadeIn(hl["ACH"]), FadeIn(hl["ADpH"]), Create(hbold), run_time=0.8)
        self.bring_to_front(dots, *labels)
        self.play(FadeIn(r3, shift=LEFT * 0.15), run_time=0.7)
        self.play(FadeOut(hl["ACH"]), FadeOut(hl["ADpH"]), FadeOut(hbold), Create(barDpC),
                  run_time=0.6)
        self.play(ReplacementTransform(barDpC.copy(), barDC), run_time=0.8)
        self.play(FadeIn(r4, shift=LEFT * 0.15), run_time=0.6)
        self.hold(0.3)
        self.play(FadeOut(barDC), run_time=0.3)

        # ---- BD′ and D′C make up BC
        self.play(Create(barBDp), run_time=0.6)
        self.play(FadeIn(r5, shift=LEFT * 0.15), run_time=0.6)
        self.play(FadeIn(r6, shift=LEFT * 0.15), FadeIn(r7, shift=LEFT * 0.15), run_time=0.8)
        self.hold(0.4)
        self.play(FadeOut(barBDp), FadeOut(barDpC), run_time=0.4)
        self.bring_to_front(dots, *labels)

        cap = caption("b²m + c²n  =  a(d² + mn)", 38)
        _final_check(self, cap, "F54")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F57 VIVIANI FOR REGULAR POLYGONS

class F57_VivianiPolygon(Board):
    """A regular polygon (here a pentagon) with side s and area A, and any
    point P inside it. The segments from P to the corners cut the polygon
    into one triangle on each side; the triangle on a side has that side s
    as its base and the distance dᵢ from P to the side as its height, so
    its area is ½ s dᵢ. Together the triangles are the polygon:
    A = ½ s (d₁ + … + d₅), so d₁ + … + d₅ = 2A/s, whatever P is. P then
    travels round a loop: the triangles keep filling the polygon, and the
    bar of the five distances (a supporting readout, drawn to scale) keeps
    the length 2A/s. (Every regular polygon; drawn for n = 5, with every
    foot on its side.)"""

    def construct(self):
        N, R = 5, 3.3
        O = np.array([-3.3, 0.25, 0.0])
        V = [O + R * _polar(PI / 2 + TAU * k / N) for k in range(N)]
        n_ = np.linalg.norm
        s, r = n_(V[1] - V[0]), R * np.cos(PI / N)
        Ar = abs(area(V))
        total = 2 * Ar / s

        def Pt(t):
            a = TAU * t
            return O + R * np.array([0.30 * np.cos(a + 0.9) + 0.06 * np.cos(2 * a),
                                     0.26 * np.sin(a + 0.9) - 0.05 * np.sin(3 * a), 0.0])

        def feet(P):
            return [_foot(P, V[i], V[(i + 1) % N]) for i in range(N)]

        # ---- checks: regular, the triangles tile it, the sum is 2A/s along the loop
        for i in range(N):
            check(abs(n_(V[(i + 1) % N] - V[i]) - s) < 1e-9, "a regular polygon: equal sides")
        check(abs(Ar - 0.5 * N * s * r) < 1e-9, "A = ½·n·s·r")
        for k, t in enumerate(np.linspace(0.0, 1.0, 241)):
            P = Pt(t)
            Fs = feet(P)
            check(all(_between(Fs[i], V[i], V[(i + 1) % N]) for i in range(N)),
                  "every foot lies on its side")
            check(abs(sum(n_(F - P) for F in Fs) - total) < 1e-9, "d₁ + … + d₅ = 2A/s")
            if k % 40 == 0:
                tris = [[P, V[i], V[(i + 1) % N]] for i in range(N)]
                check(_tiles_exactly(tris, V, n=70), "the five triangles tile the polygon")
                for i, T in enumerate(tris):
                    check(abs(abs(area(T)) - 0.5 * s * n_(Fs[i] - P)) < 1e-9,
                          "each triangle has area ½ s dᵢ")
        check(close(Pt(0.0), Pt(1.0)), "the loop closes")

        # ---- the pieces
        cols = [BLUE_D, TEAL_D, ORANGE, YELLOW_E, GREEN_D]
        brights = [BLUE_B, TEAL_B, ORANGE, YELLOW_B, GREEN_B]
        poly = mk(V, GREY_D, 0.5, stroke_width=4)
        t_ = ValueTracker(0.0)

        def tris_m():
            P = Pt(t_.get_value())
            return VGroup(*[mk([P, V[i], V[(i + 1) % N]], cols[i], 0.55, stroke_width=2)
                            for i in range(N)])

        def perps_m():
            P = Pt(t_.get_value())
            Fs = feet(P)
            g = VGroup()
            for i in range(N):
                g.add(Line(P, Fs[i], color=brights[i], stroke_width=5))
                g.add(_ra(Fs[i], P, V[i], 0.16, WHITE, 2.5))
            return g

        def dot_m():
            return Dot(Pt(t_.get_value()), radius=0.07, color=WHITE)

        # the readout: the five distances end to end, and the bar 2A/s, to scale
        kb = 5.1 / total
        bx, by = 0.95, -1.75

        def stack_m():
            P = Pt(t_.get_value())
            Fs = feet(P)
            g, x = VGroup(), bx
            for i in range(N):
                L = n_(Fs[i] - P) * kb
                g.add(Line([x, by, 0], [x + L, by, 0], color=brights[i], stroke_width=14))
                x += L
            return g

        ref = Line([bx, by - 0.45, 0], [bx + total * kb, by - 0.45, 0], color=WHITE,
                   stroke_width=14)
        parts = []
        for i in range(N):
            parts.append((f"d{'₁₂₃₄₅'[i]}", brights[i]))
            if i < N - 1:
                parts.append(("+", WHITE))
        l_sum = _trow(parts, 26, 0.14).next_to(
            Line([bx, by, 0], [bx + total * kb, by, 0]), UP, buff=0.22)
        l_ref = tag("2A / s", 26).next_to(ref, DOWN, buff=0.2)
        l_sum.align_to(np.array([bx, 0.0, 0.0]), LEFT)
        l_ref.align_to(np.array([bx, 0.0, 0.0]), LEFT)

        # the ledger
        lx = 0.95
        r1 = _trow([("A", WHITE), ("=", WHITE), ("½ s d₁ + … + ½ s d₅", WHITE)], 28, 0.2)
        r2 = _trow([("=", WHITE), ("½ s (d₁ + … + d₅)", WHITE)], 28, 0.2)
        r1.shift(np.array([lx - r1.get_left()[0], 2.7 - r1.get_center()[1], 0.0]))
        r2.shift(np.array([r1[1].get_center()[0] - r2[0].get_center()[0],
                           1.95 - r2.get_center()[1], 0.0]))
        _all_in_frame([r1, r2, l_sum, l_ref, ref], "F57 panel")
        check(bx + total * kb <= SAFE_X - 0.1, "the readout fits")
        check(min(m.get_left()[0] for m in (r1, r2, l_sum, ref)) > max(v[0] for v in V) + 0.4,
              "the panel is clear of the polygon")

        # static labels at the start (and end) position of P
        P0 = Pt(0.0)
        F0 = feet(P0)
        segs0 = (_segs(*V, closed=True) + [s_ for i in range(N) for s_ in _segs(P0, V[i])]
                 + _mob_segs(perps_m(), dot_m()))
        lP = _park(tag("P", 28), P0, segs0, towards=V + F0, d0=0.22, d1=0.8)
        lds = []
        for i in range(N):
            m = tag(f"d{'₁₂₃₄₅'[i]}", 28, brights[i])
            m.set_stroke(BLACK, width=5, background=True)      # legible on its fill
            lds.append(_park_beside(m, P0, F0[i], segs0, ats=(0.55, 0.45, 0.65, 0.35),
                                    gaps=(0.08, 0.12, 0.16), avoid=[lP] + lds))
        mid = (V[2] + V[3]) / 2                       # the bottom side
        ls = tag("s", 30).move_to(mid + 0.32 * _unit(mid - O))
        _labels_ok([lP, ls, *lds], segs0 + _mob_segs(tris_m()), "F57 at P")

        # ---- the polygon, P and its distances to the sides
        self.play(FadeIn(poly), FadeIn(ls), run_time=0.9)
        dP, perps = dot_m(), perps_m()
        self.play(FadeIn(dP), FadeIn(lP), Create(perps), *[FadeIn(m) for m in lds], run_time=1.3)

        # ---- the triangles on the sides fill the polygon
        tris = tris_m()
        self.add(tris)
        self.bring_to_front(perps, dP, lP, *lds)
        self.play(FadeIn(tris), run_time=1.0)
        self.play(FadeIn(r1, shift=LEFT * 0.2), run_time=0.8)
        self.play(FadeIn(r2, shift=LEFT * 0.2), run_time=0.8)
        stack = stack_m()
        self.play(FadeIn(stack), FadeIn(l_sum), run_time=0.7)
        self.play(FadeIn(ref), FadeIn(l_ref), run_time=0.6)
        self.hold(0.4)

        # ---- P anywhere: the triangles still fill it, the sum stays 2A/s
        self.remove(tris, perps, dP, stack)
        tris, perps = always_redraw(tris_m), always_redraw(perps_m)
        dP, stack = always_redraw(dot_m), always_redraw(stack_m)
        self.add(tris, perps, dP, stack)
        self.play(FadeOut(lP), *[FadeOut(m) for m in lds], run_time=0.4)
        self.play(t_.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        check(close(Pt(t_.get_value()), P0), "P is back where it started")
        for mob in (tris, perps, dP, stack):
            mob.clear_updaters()
        self.play(FadeIn(lP), *[FadeIn(m) for m in lds], run_time=0.5)

        cap = caption("d₁ + d₂ + d₃ + d₄ + d₅  =  2A / s   for every P inside", 32)
        _final_check(self, cap, "F57")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F58 THE TRIANGLE OF MEDIANS

def _slide_shear_check(src, mid, dst, what):
    """src -> mid is a shear (first two vertices fixed, apex moved parallel
    to them), mid -> dst a slide (a translation)."""
    check(close(src[0], mid[0]) and close(src[1], mid[1])
          and abs(_cross(src[1] - src[0], mid[2] - src[2])) < 1e-9, f"{what}: a shear")
    v = dst[0] - mid[0]
    check(all(close(m + v, d_) for m, d_ in zip(mid, dst)), f"{what}: a slide")


class F58_MedianTriangle(Board):
    """D, E, F are the midpoints of BC, CA, AB. Slide the median BE by half
    of BA (completing the parallelogram BEKF) and the median AD by half of
    BC (completing ADCK): with CF they now close up into the triangle CFK,
    whose sides are the three medians. K lies on the midline DE produced,
    with KE = ED, and E lies inside CFK, splitting it into ECF, EFK and
    EKC. Each of these has the same area as one of the four congruent
    quarters cut off by the midlines: ECF and EAF have equal bases EC = AE
    on CA and the apex F (shear F to D, then slide by CA/2); EFK and EDF
    have equal bases KE = ED on the line KD and the apex F; EKC and EDC
    have the same bases and the apex C. So the median triangle is three of
    the four quarters: ¾ of ABC. (Any triangle; drawn scalene and acute.)"""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(76, 48)
        Km = Am + (Cm - Bm) / 2
        P3m = Cm + (Am - Bm) / 2                      # piece 3 after its shear
        pts = np.array([Am, Bm, Cm, Km, P3m])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Fr = Frame(lo[0] - 0.07, hi[0] + 0.05, lo[1] - 0.09, hi[1] + 0.08, max_w=7.7,
                   centre=(-2.45, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        D, E, F = (B + C) / 2, (C + A) / 2, (A + B) / 2
        K = E + F - B
        T = abs(area([A, B, C]))

        # the pieces: [base, base, apex] -> sheared -> slid
        pieces = {
            "ECF": ([E, C, F], [E, C, D], [A, E, F]),
            "EFK": ([K, E, F], [K, E, A], [E, D, F]),
            "EKC": ([K, E, C], [K, E, C + (A - B) / 2], [E, D, C]),
        }
        quarters = [[A, F, E], [F, B, D], [E, D, C], [D, E, F]]

        # ---- checks
        check(_scalene(A, B, C) and max(_angle(A, B, C), _angle(B, C, A),
                                        _angle(C, A, B)) < PI / 2, "scalene acute triangle")
        check(close(K - F, E - B) and close(C - K, D - A), "FK = BE and KC = AD as slides")
        check(_on_line(K, D, E) and close((K + D) / 2, E), "K on DE produced, KE = ED")
        check(_pip(E, [C, F, K]), "E lies inside CFK")
        check(_tiles_exactly([p[0] for p in pieces.values()], [C, F, K]),
              "ECF, EFK, EKC tile the median triangle")
        check(_tiles_exactly(quarters, [A, B, C]), "the midlines cut ABC into four triangles")
        for Q in quarters:
            check(abs(abs(area(Q)) - T / 4) < 1e-9, "each quarter is ¼ of ABC")
        for nm, (src, mid, dst) in pieces.items():
            _slide_shear_check(src, mid, dst, nm)
            check(abs(abs(area(src)) - T / 4) < 1e-9, f"{nm} is ¼ of ABC")
        check(abs(abs(area([C, F, K])) - 0.75 * T) < 1e-9, "[CFK] = ¾ [ABC]")

        # ---- pieces
        cT, cMed, cPar = BLUE_E, YELLOW_B, GREY_B
        col = {"ECF": TEAL_D, "EFK": ORANGE, "EKC": PURPLE_B}
        tri = mk([A, B, C], cT, 0.5, stroke_width=4)
        dots = VGroup(*[Dot(X, radius=0.06, color=WHITE) for X in (D, E, F)])
        tks = VGroup(_ticks(A, F, 1), _ticks(F, B, 1), _ticks(B, D, 2), _ticks(D, C, 2),
                     _ticks(C, E, 3), _ticks(E, A, 3))
        mAD = Line(A, D, color=cMed, stroke_width=6)
        mBE = Line(B, E, color=cMed, stroke_width=6)
        mCF = Line(C, F, color=cMed, stroke_width=6)
        parBEKF = DashedVMobject(Polygon(B, E, K, F, stroke_color=cPar, stroke_width=2.5),
                                 num_dashes=48)
        parADCK = DashedVMobject(Polygon(A, D, C, K, stroke_color=cPar, stroke_width=2.5),
                                 num_dashes=48)
        medtri = mk([C, F, K], YELLOW_E, 0.25, stroke_width=0)
        splits = VGroup(Line(E, F, color=WHITE, stroke_width=2.5),
                        Line(E, K, color=WHITE, stroke_width=2.5))
        pm = {nm: mk(p[0], col[nm], 0.7, stroke_width=2) for nm, p in pieces.items()}
        midl = VGroup(Line(D, E, color=WHITE, stroke_width=2.5), Line(E, F, color=WHITE, stroke_width=2.5),
                      Line(F, D, color=WHITE, stroke_width=2.5))
        mtk = VGroup(_ticks(D, E, 1, at=0.5), _ticks(E, F, 2, at=0.5), _ticks(F, D, 3, at=0.5))
        dK = Dot(K, radius=0.06, color=WHITE)
        everything = _mob_segs(tri, dots, tks, mAD, mBE, mCF, parBEKF, parADCK, splits, midl,
                               mtk, dK, Line(F, K), Line(K, C), _ticks(K, E, 1, at=0.5),
                               *[mk(p[1], WHITE, 0.0) for p in pieces.values()])
        G = (A + B + C) / 3
        labs = {}
        for nm, X, away in (("A", A, A - G), ("B", B, B - G), ("C", C, C - G),
                            ("D", D, D - A), ("E", E, E - B), ("F", F, F - C), ("K", K, K - E)):
            base = np.degrees(np.arctan2(away[1], away[0]))
            labs[nm] = _park(tag(nm, 28), X, everything, d0=0.24, d1=0.8,
                             dirs=[base + t for t in (0, 20, -20, 40, -40, 60, -60, 90, -90)],
                             avoid=list(labs.values()))

        # the ledger: icons at one scale, the equals signs in one column
        px, room = 1.75, SAFE_X - 1.75 - 0.1

        def rows_at(k_ic):
            def icon(poly, c, op=0.8):
                return _icon(poly, k_ic, c, op)
            r0 = icon([C, F, K], YELLOW_E, 0.6)
            r1 = _row(tag("=", 30), icon(pieces["ECF"][0], col["ECF"]), tag("+", 30),
                      icon(pieces["EFK"][0], col["EFK"]), tag("+", 30),
                      icon(pieces["EKC"][0], col["EKC"]), buff=0.16)
            r2 = _row(tag("=", 30), icon(pieces["ECF"][2], col["ECF"]), tag("+", 30),
                      icon(pieces["EFK"][2], col["EFK"]), tag("+", 30),
                      icon(pieces["EKC"][2], col["EKC"]), buff=0.16)
            r3 = _row(tag("=", 30), tag("¾", 34), icon([A, B, C], cT, 0.8), buff=0.16)
            return [r0, r1, r2, r3]

        k_ic = 0.3
        while max(r.width for r in rows_at(k_ic)) > room or sum(
                r.height for r in rows_at(k_ic)) > 4.3:
            k_ic -= 0.01
        check(k_ic >= 0.18, "the ledger icons stay a readable size")
        r0, r1, r2, r3 = rows_at(k_ic)
        y = SAFE_TOP - 0.15
        for r in (r0, r1, r2, r3):
            r.move_to([0, y - r.height / 2, 0]).align_to(np.array([px, 0, 0]), LEFT)
            y -= r.height + 0.32
        r0.shift(np.array([r1[1].get_left()[0] - r0.get_left()[0], 0, 0]))
        _all_in_frame([r0, r1, r2, r3], "F58 ledger")
        check(min(r.get_left()[0] for r in (r0, r1, r2, r3)) > max(
            max(X[0] for X in (A, B, C, K, pieces["EKC"][1][2])),
            max(m.get_right()[0] for m in labs.values())) + 0.25, "the ledger is clear of the figure")

        # ---- the triangle, its midpoints, its medians
        self.play(FadeIn(tri), *[FadeIn(labs[k]) for k in "ABC"], run_time=1.0)
        self.play(FadeIn(dots), FadeIn(tks), *[FadeIn(labs[k]) for k in "DEF"], run_time=0.8)
        self.play(Create(mAD), Create(mBE), Create(mCF), run_time=1.0)
        self.bring_to_front(dots)
        self.hold(0.3)

        # ---- slide two medians: parallelograms BEKF and ADCK
        self.play(Create(parBEKF), run_time=0.6)
        self.play(mBE.animate.shift(F - B), run_time=1.3)
        check(close(mBE.get_start(), F, 1e-6) and close(mBE.get_end(), K, 1e-6), "BE slides onto FK")
        self.play(FadeIn(dK), FadeIn(labs["K"]), Create(parADCK), run_time=0.6)
        self.play(mAD.animate.shift(K - A), run_time=1.3)
        check(close(mAD.get_start(), K, 1e-6) and close(mAD.get_end(), C, 1e-6), "AD slides onto KC")
        self.add(medtri)
        self.bring_to_back(medtri)
        self.bring_to_back(tri)
        self.play(FadeIn(medtri), FadeOut(parBEKF), FadeOut(parADCK), run_time=0.7)

        # ---- E splits the median triangle into three pieces
        for m in pm.values():
            self.add(m)
        self.bring_to_front(mAD, mBE, mCF, splits, dots, dK, *labs.values())
        tkKE = _ticks(K, E, 1, at=0.5)
        self.play(Create(splits), *[FadeIn(m) for m in pm.values()], FadeIn(tkKE), run_time=0.9)
        self.play(FadeIn(r0, shift=LEFT * 0.2), FadeIn(r1, shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.3)

        # ---- the midlines cut ABC into four congruent quarters
        self.add(midl, mtk)
        self.bring_to_front(midl, mtk, tkKE, dots, dK, *labs.values())
        self.play(Create(midl), FadeIn(mtk), run_time=0.8)
        self.hold(0.3)

        # ---- each piece in turn: a shear (same base, apex along a parallel
        # to it), then a slide along the base line onto a quarter
        for nm in ("ECF", "EFK", "EKC"):
            src, mid, dst = pieces[nm]
            base_hl = Line(src[0], src[1], color=col[nm] if nm != "ECF" else TEAL_B, stroke_width=8)
            guide = DashedLine(src[2], mid[2], color=WHITE, stroke_width=2.5, dash_length=0.07)
            self.play(Create(base_hl), Create(guide), run_time=0.5)
            self.bring_to_front(pm[nm])
            self.play(_shear_to(pm[nm], src[:2], src[2], mid[2]), run_time=1.1)
            check(_same_poly(pm[nm].get_vertices(), mid, 1e-6), f"{nm} sheared")
            self.play(FadeOut(guide), pm[nm].animate.shift(dst[0] - mid[0]),
                      base_hl.animate.shift(dst[0] - mid[0]), run_time=1.1)
            check(_same_poly(pm[nm].get_vertices(), dst, 1e-6), f"{nm} lands on a quarter")
            self.play(FadeOut(base_hl), run_time=0.3)
            self.bring_to_front(mCF, mBE, mAD, midl, mtk, dots, dK, *labs.values())
        self.play(FadeIn(r2, shift=LEFT * 0.2), run_time=0.7)
        self.play(FadeIn(r3, shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.3)

        cap = caption("[triangle of the medians]  =  ¾ [ABC]", 34)
        _final_check(self, cap, "F58")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F61 THE INCENTRE-EXCENTRE LEMMA

def _tracked_wedge(V, p, q, r, color, op=0.75):
    """A filled wedge for the non-reflex angle p-V-q, with three invisible
    markers (vertex and the two arc ends) riding along, so a rigid move of
    the group can be checked point by point."""
    w = _wedge(V, p, q, r, color, op, stroke=0)
    V = to3(V)
    marks = VGroup(*[Dot(X, radius=0.01, fill_opacity=0.0, stroke_width=0)
                     for X in (V, V + r * _unit(to3(p) - V), V + r * _unit(to3(q) - V))])
    return VGroup(w, marks)


def _wedge_at(g, V, d1, d2, r):
    """The tracked wedge g sits at V with its arc ends along the unit
    directions d1, d2 (either order)."""
    v, e1, e2 = (m.get_center() for m in g[1])
    want = (to3(V) + r * _unit(d1), to3(V) + r * _unit(d2))
    return close(v, V, 1e-6) and ((close(e1, want[0], 1e-6) and close(e2, want[1], 1e-6))
                                  or (close(e1, want[1], 1e-6) and close(e2, want[0], 1e-6)))


class F61_IncentreExcentre(Board):
    """The bisector of the angle A (= 2α) meets the circumcircle again at
    M, the midpoint of the arc BC; the bisector of B (= 2β) meets it at the
    incentre I. ∠MBC and ∠MAC stand on the same arc MC, so ∠MBC = α, and
    ∠MBI = α + β. At I, the angle BIM is the exterior angle of ABI: the
    angle α at A, slid along AM to I, and the angle β at B, turned half a
    turn about the midpoint of BI, fill it exactly -- so ∠BIM = α + β too.
    The triangle MBI has equal angles at B and I: MB = MI. In the same way
    (C = 2γ) ∠MCI = ∠MIC = α + γ, so MC = MI. M is the centre of a circle
    through B, I and C. (Any triangle; drawn scalene and acute. The excentre
    opposite A, also on that circle, is not drawn.)"""

    def construct(self):
        Adeg, Bdeg, Cdeg = 56, 74, 50
        al, be, ga = (Adeg / 2) * DEGREES, (Bdeg / 2) * DEGREES, (Cdeg / 2) * DEGREES
        O = np.array([-2.75, 0.5, 0.0])
        R = 2.85
        M = O + R * _polar(-PI / 2)
        B = O + R * _polar(-PI / 2 - Adeg * DEGREES)
        C = O + R * _polar(-PI / 2 + Adeg * DEGREES)
        A = O + R * _polar(-PI / 2 + Adeg * DEGREES + 2 * Bdeg * DEGREES)
        n_ = np.linalg.norm
        a, b, c = n_(C - B), n_(A - C), n_(B - A)
        I = (a * A + b * B + c * C) / (a + b + c)
        NB, NC = (B + I) / 2, (C + I) / 2

        # ---- checks
        check(abs(_angle(A, B, C) - 2 * al) < 1e-9 and abs(_angle(B, C, A) - 2 * be) < 1e-9
              and abs(_angle(C, A, B) - 2 * ga) < 1e-9, "the angles are 2α, 2β, 2γ")
        check(_scalene(A, B, C), "scalene triangle")
        check(_on_line(I, A, M) and _between(I, A, M), "A, I, M in a row: AM bisects A")
        check(abs(_angle(A, B, I) - _angle(A, I, C)) < 1e-9, "AI bisects A")
        check(abs(_angle(B, A, I) - be) < 1e-9 and abs(_angle(C, A, I) - ga) < 1e-9,
              "BI and CI bisect B and C")
        check(abs(_angle(B, C, M) - al) < 1e-9 and abs(_angle(C, B, M) - al) < 1e-9,
              "inscribed angles on the arcs BM and MC are α")
        check(abs(_angle(B, I, M) - (al + be)) < 1e-9 and abs(_angle(B, M, I) - (al + be)) < 1e-9,
              "∠MBI = ∠MIB = α + β")
        check(abs(_angle(C, I, M) - (al + ga)) < 1e-9 and abs(_angle(C, M, I) - (al + ga)) < 1e-9,
              "∠MCI = ∠MIC = α + γ")
        check(abs(n_(M - B) - n_(M - I)) < 1e-9 and abs(n_(M - C) - n_(M - I)) < 1e-9,
              "MB = MI = MC")

        # ---- the pieces
        cA, cB, cC = YELLOW_B, TEAL_B, ORANGE
        circ = Circle(radius=R, color=GREY_B, stroke_width=2.5).move_to(O)
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        am = Line(A, M, color=WHITE, stroke_width=2.5)
        bi = Line(B, I, color=WHITE, stroke_width=2.5)
        ci = Line(C, I, color=WHITE, stroke_width=2.5)
        mb = Line(M, B, color=WHITE, stroke_width=3)
        mc = Line(M, C, color=WHITE, stroke_width=3)
        dI, dM = Dot(I, radius=0.065), Dot(M, radius=0.065)
        rA, rB, rC = 0.62, 0.55, 0.55
        arcA = VGroup(angle_arc(A, B, I, rA, cA, 4), angle_arc(A, I, C, rA + 0.08, cA, 4))
        arcB = VGroup(angle_arc(B, A, I, rB, cB, 4), angle_arc(B, I, C, rB + 0.08, cB, 4))
        arcBM = angle_arc(B, C, M, rB + 0.3, cA, 4)
        arcC = VGroup(angle_arc(C, A, I, rC, cC, 4), angle_arc(C, I, B, rC + 0.08, cC, 4))
        arcCM = angle_arc(C, B, M, rC + 0.3, cA, 4)
        chordMC = Line(M, C, color=cA, stroke_width=7)
        tkB, tkI, tkC = _ticks(M, B, 1, WHITE, at=0.45), _ticks(M, I, 1, WHITE, at=0.28), \
            _ticks(M, C, 1, WHITE, at=0.45)
        # the arc of the circle about M from C over I to B (the rest of
        # that circle runs below the frame)
        aC, aB, aI = _dir(M, C), _dir(M, B), _dir(M, I)
        span = (aB - aC) % TAU
        check(0 < (aI - aC) % TAU < span, "the arc from C to B passes I")
        check(abs((M + 0.28 * (I - M))[1] - B[1]) > 0.3, "the tick on MI is clear of BC")
        circM = DashedVMobject(Arc(radius=n_(M - I), start_angle=aC, angle=span, arc_center=M,
                                   color=YELLOW_B, stroke_width=2.5), num_dashes=40)
        rw = 0.5
        # the angle marks leave before the circle about M arrives: the Greek
        # labels avoid everything else, the point labels avoid that arc too
        strokes = _mob_segs(circ, tri, am, bi, ci, mb, mc, dI, dM, arcA, arcB, arcBM, arcC,
                            arcCM, tkB, tkI, tkC,
                            _wedge(I, B, M, rw + 0.02, WHITE), _wedge(I, C, M, rw + 0.02, WHITE))
        strokes_pts = strokes + _mob_segs(circM)

        labs = {}
        for nm, X, d in (("A", A, 90), ("B", B, 200), ("C", C, -20), ("M", M, 270)):
            labs[nm] = _park(tag(nm, 30), X, strokes_pts, dirs=(d, d + 20, d - 20, d + 40, d - 40),
                             d0=0.25, d1=0.8, avoid=list(labs.values()))
        labs["I"] = _park(tag("I", 30), I, strokes_pts, towards=[A, B, C, M], d0=0.22, d1=0.9,
                          avoid=list(labs.values()))
        avoid = list(labs.values())
        lgA = []
        for k, (P1, P2) in enumerate(((B, I), (I, C))):
            lgA.append(_angle_label(tag("α", 26, cA), A, P1, P2, rA + 0.25, segs=strokes,
                                    avoid=avoid + lgA))
        lgB = []
        for P1, P2 in ((A, I), (I, C)):
            lgB.append(_angle_label(tag("β", 26, cB), B, P1, P2, rB + 0.25, segs=strokes,
                                    avoid=avoid + lgA + lgB))
        lgBM = _angle_label(tag("α", 26, cA), B, C, M, rB + 0.45, segs=strokes,
                            avoid=avoid + lgA + lgB)
        lgC = []
        for P1, P2 in ((A, I), (I, B)):
            lgC.append(_angle_label(tag("γ", 26, cC), C, P1, P2, rC + 0.25, segs=strokes,
                                    avoid=avoid + lgA + lgB + [lgBM] + lgC))
        lgCM = _angle_label(tag("α", 26, cA), C, B, M, rC + 0.45, segs=strokes,
                            avoid=avoid + lgA + lgB + [lgBM] + lgC)
        labels = [*labs.values(), *lgA, *lgB, lgBM, *lgC, lgCM]
        _labels_ok(labels, strokes, "F61 figure")

        # the ledger
        lx = 1.15

        def ang_row(lhs, g, cg):
            return _trow([(lhs, WHITE), ("=", WHITE), ("α", cA), ("+", WHITE), (g, cg)], 28, 0.14)

        rows = [ang_row("∠MBI", "β", cB), ang_row("∠MIB", "β", cB),
                _trow([("MB = MI", WHITE)], 28),
                ang_row("∠MCI", "γ", cC), ang_row("∠MIC", "γ", cC),
                _trow([("MC = MI", WHITE)], 28)]
        y = 3.2
        for i, r in enumerate(rows):
            r.shift(np.array([lx - r.get_left()[0], y - r.get_center()[1], 0.0]))
            y -= 0.62 if i not in (2,) else 0.95
        _all_in_frame(rows, "F61 ledger")
        check(min(r.get_left()[0] for r in rows) > O[0] + R + 0.25, "the ledger is clear of the circle")

        # ---- the triangle in its circle; the bisectors
        self.play(Create(circ), Create(tri), *[FadeIn(labs[k]) for k in "ABC"], run_time=1.2)
        self.play(Create(am), FadeIn(dM), FadeIn(labs["M"]), Create(arcA), *[FadeIn(m) for m in lgA],
                  run_time=1.0)
        self.play(Create(bi), FadeIn(dI), FadeIn(labs["I"]), Create(arcB),
                  *[FadeIn(m) for m in lgB], run_time=1.0)
        self.play(Create(mb), Create(mc), run_time=0.7)
        self.bring_to_front(dI, dM)

        # ---- the same arc MC: ∠MBC = ∠MAC = α
        self.play(Create(chordMC), Indicate(arcA[1], color=WHITE, scale_factor=1.15), run_time=0.8)
        self.play(Create(arcBM), FadeIn(lgBM), run_time=0.6)
        self.play(FadeOut(chordMC), run_time=0.3)
        wedB = VGroup(_wedge(B, I, C, rB + 0.12, cB, 0.45, stroke=0),
                      _wedge(B, C, M, rB + 0.36, cA, 0.45, stroke=0))
        self.add(wedB)
        self.bring_to_back(wedB)
        self.play(FadeIn(wedB), FadeIn(rows[0], shift=LEFT * 0.2), run_time=0.7)

        # ---- at I: α slides along AM, β turns half a turn about the
        # midpoint of BI; together they fill the angle BIM
        wa = _tracked_wedge(A, B, I, rw, cA)
        wb = _tracked_wedge(B, A, I, rw, cB)
        dNB = Dot(NB, radius=0.05, color=cB)
        self.add(wa, wb)
        self.play(FadeIn(wa), FadeIn(wb), FadeIn(dNB), run_time=0.5)
        self.play(wa.animate.shift(I - A), Rotate(wb, angle=PI, about_point=NB), run_time=1.8)
        check(_wedge_at(wa, I, B - A, M - I, rw), "α slid along AM lands at I, between AB's "
                                                  "direction and IM")
        check(_wedge_at(wb, I, B - A, B - I, rw), "β turned about the midpoint of BI lands at I")
        self.play(FadeOut(dNB), FadeIn(rows[1], shift=LEFT * 0.2), run_time=0.6)
        self.play(FadeIn(tkB), FadeIn(tkI), FadeIn(rows[2], shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.4)
        self.play(FadeOut(wa), FadeOut(wb), FadeOut(wedB), run_time=0.4)

        # ---- the same at C
        wa2 = _tracked_wedge(A, I, C, rw, cA)
        wc = _tracked_wedge(C, A, I, rw, cC)
        dNC = Dot(NC, radius=0.05, color=cC)
        wedC = VGroup(_wedge(C, I, B, rC + 0.12, cC, 0.45, stroke=0),
                      _wedge(C, B, M, rC + 0.36, cA, 0.45, stroke=0))
        self.add(wedC)
        self.bring_to_back(wedC)
        self.play(Create(ci), Create(arcC), *[FadeIn(m) for m in lgC], Create(arcCM),
                  FadeIn(lgCM), FadeIn(wedC), run_time=1.0)
        self.play(FadeIn(rows[3], shift=LEFT * 0.2), run_time=0.5)
        self.add(wa2, wc)
        self.play(FadeIn(wa2), FadeIn(wc), FadeIn(dNC), run_time=0.4)
        self.play(wa2.animate.shift(I - A), Rotate(wc, angle=PI, about_point=NC), run_time=1.6)
        check(_wedge_at(wa2, I, C - A, M - I, rw), "the other α lands at I")
        check(_wedge_at(wc, I, C - A, C - I, rw), "γ turned about the midpoint of CI lands at I")
        self.play(FadeOut(dNC), FadeIn(rows[4], shift=LEFT * 0.2), run_time=0.5)
        self.play(FadeIn(tkC), FadeIn(rows[5], shift=LEFT * 0.2), run_time=0.6)
        self.hold(0.3)
        self.play(FadeOut(wa2), FadeOut(wc), FadeOut(wedC), run_time=0.4)

        # ---- the circle about M through B, I, C
        greek = [*lgA, *lgB, lgBM, *lgC, lgCM]
        self.play(*[FadeOut(m) for m in (arcA, arcB, arcBM, arcC, arcCM, *greek)], run_time=0.5)
        self.play(Create(circM), run_time=1.0)
        self.bring_to_front(dI, dM, *labs.values())
        _labels_ok(list(labs.values()), strokes_pts, "F61 end")

        cap = caption("MB = MI = MC", 38)
        _final_check(self, cap, "F61")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F63 CUTTING CORNERS AT THIRDS

class F63_CornersAtThirds(Board):
    """Each side of ABC is cut in three equal parts; the cut joining the
    two points next to a corner takes that corner off. The lines through
    the third-points parallel to the sides cut ABC into nine small
    triangles: the corner tile at A, slid along the grid, covers the five
    other upright cells, and turned half a turn about the right point it
    covers each of the three inverted ones. So the nine cells are congruent,
    each is 1/9 of ABC; each corner piece is one cell, and the hexagon left
    is the other six: 6/9 = 2/3 of ABC. (Any triangle; drawn scalene.)"""

    def construct(self):
        Am, Bm, Cm = _tri_from_angles(72, 44)
        pts = np.array([Am, Bm, Cm])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Fr = Frame(lo[0] - 0.08, hi[0] + 0.08, lo[1] - 0.09, hi[1] + 0.08, max_w=7.6,
                   centre=(-2.5, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        u, v = (B - A) / 3, (C - A) / 3

        def Pg(i, j):
            return A + i * u + j * v

        ups = {(i, j): [Pg(i, j), Pg(i + 1, j), Pg(i, j + 1)]
               for i in range(3) for j in range(3) if i + j <= 2}
        downs = {(i, j): [Pg(i + 1, j + 1), Pg(i, j + 1), Pg(i + 1, j)]
                 for i in range(2) for j in range(2) if i + j <= 1}
        corners = {"A": ups[(0, 0)], "B": ups[(2, 0)], "C": ups[(0, 2)]}
        X1, X2, Y1, Y2, Z1, Z2 = Pg(1, 0), Pg(2, 0), Pg(2, 1), Pg(1, 2), Pg(0, 2), Pg(0, 1)
        hexagon = [X1, X2, Y1, Y2, Z1, Z2]
        T = abs(area([A, B, C]))

        # ---- checks
        check(_scalene(A, B, C), "scalene triangle")
        check(close(X1, A + (B - A) / 3) and close(Y1, B + (C - B) / 3) and close(Z1, C + (A - C) / 3)
              and close(X2, A + 2 * (B - A) / 3) and close(Y2, B + 2 * (C - B) / 3)
              and close(Z2, C + 2 * (A - C) / 3), "the third-points")
        check(_same_poly(corners["A"], [A, X1, Z2]) and _same_poly(corners["B"], [B, Y1, X2])
              and _same_poly(corners["C"], [C, Z1, Y2]), "the corner pieces are cells of the grid")
        check(_tiles_exactly(list(ups.values()) + list(downs.values()), [A, B, C]),
              "nine cells tile ABC")
        check(_tiles_exactly(list(corners.values()) + [hexagon], [A, B, C]),
              "three corners and the hexagon tile ABC")
        for c in list(ups.values()) + list(downs.values()):
            check(abs(abs(area(c)) - T / 9) < 1e-9, "each cell is 1/9 of ABC")
        check(abs(abs(area(hexagon)) - 2 * T / 3) < 1e-9, "[hexagon] = 2/3 [ABC]")
        moves = []                             # (kind, parameter, target cell)
        for (i, j), cell in ups.items():
            if (i, j) != (0, 0):
                moves.append(("shift", i * u + j * v, cell))
        for (i, j), cell in downs.items():
            Zc = A + ((i + 1) * u + (j + 1) * v) / 2
            img = [2 * Zc - X for X in corners["A"]]
            check(all(close(a_, b_) for a_, b_ in zip(img, cell)),
                  "a half-turn takes the corner tile onto an inverted cell")
            moves.append(("turn", Zc, cell))

        # ---- the pieces
        cT, cCorner, cHex, cTile = BLUE_E, ORANGE, BLUE_D, YELLOW_B
        tri = mk([A, B, C], cT, 0.5, stroke_width=4)
        dots = VGroup(*[Dot(X, radius=0.055, color=WHITE) for X in hexagon])
        tks = VGroup()
        for P_, Q_, k in ((A, B, 1), (B, C, 2), (C, A, 3)):
            for t0 in (1 / 6, 1 / 2, 5 / 6):
                tks.add(_ticks(P_, Q_, k, WHITE, at=t0))
        cuts = VGroup(*[DashedLine(P_, Q_, color=WHITE, stroke_width=3, dash_length=0.09)
                        for P_, Q_ in ((X1, Z2), (X2, Y1), (Y2, Z1))])
        more = VGroup(*[DashedLine(P_, Q_, color=GREY_B, stroke_width=2.5, dash_length=0.09)
                        for P_, Q_ in ((X2, Z1), (Y2, X1), (Y1, Z2))])
        for P_, Q_ in ((X2, Z1), (Y2, X1), (Y1, Z2)):
            check(abs(_cross(Q_ - P_, (C - B) if P_ is X2 else (A - C) if P_ is Y2 else (B - A)))
                  < 1e-9, "the other three lines are parallels to the sides")
        hexm = mk(hexagon, cHex, 0.75, stroke_width=0)
        cornm = {k: mk(c, cCorner, 0.8, stroke_width=0) for k, c in corners.items()}
        G = (A + B + C) / 3
        segs = _mob_segs(tri, dots, tks, cuts, more)
        labs = {k: _park(tag(k, 30), X, segs, d0=0.25, d1=0.8,
                         dirs=[np.degrees(np.arctan2(*(X - G)[1::-1])) + t for t in (0, 20, -20, 40, -40)])
                for k, X in (("A", A), ("B", B), ("C", C))}

        # the ledger: icons at one scale, equals signs in one column
        k_ic = 0.2
        px = 1.9
        r1 = _row(_icon(corners["A"], k_ic, cCorner), tag("=", 32), tag("1/9", 32),
                  _icon([A, B, C], k_ic, cT, 0.8), buff=0.22)
        r2 = _row(_icon(hexagon, k_ic, cHex), tag("=", 32), tag("6/9", 32),
                  _icon([A, B, C], k_ic, cT, 0.8), buff=0.22)
        r3 = _row(tag("=", 32), tag("2/3", 32), _icon([A, B, C], k_ic, cT, 0.8), buff=0.22)
        r2.move_to([0, 0.75, 0]).align_to(np.array([px, 0, 0]), LEFT)
        r1.move_to([0, 2.45, 0])
        r3.move_to([0, -0.85, 0])
        for r in (r1, r3):
            k_eq = 1 if r is r1 else 0
            r.shift(np.array([r2[1].get_center()[0] - r[k_eq].get_center()[0], 0, 0]))
        _all_in_frame([r1, r2, r3], "F63 ledger")
        check(min(r.get_left()[0] for r in (r1, r2, r3)) > max(X[0] for X in (A, B, C)) + 0.3,
              "the ledger is clear of the figure")

        # ---- the triangle, its sides in thirds, the corners cut off
        self.play(FadeIn(tri), *[FadeIn(m) for m in labs.values()], run_time=1.0)
        self.play(FadeIn(dots), FadeIn(tks), run_time=0.9)
        self.play(Create(cuts), run_time=0.8)
        self.add(hexm, *cornm.values())
        self.bring_to_front(cuts, tks, dots)
        self.play(FadeIn(hexm), *[FadeIn(m) for m in cornm.values()], run_time=0.8)
        self.hold(0.4)

        # ---- the parallels through the third-points: nine cells
        self.play(Create(more), run_time=0.9)
        self.hold(0.3)

        # ---- the corner tile at A reaches every cell: slides and half-turns
        copies, anims = [], []
        for kind, par, cell in moves:
            cp = mk(corners["A"], cTile, 0.35, stroke_color=cTile, stroke_width=3)
            copies.append((cp, cell))
            if kind == "shift":
                anims.append(cp.animate.shift(par))
            else:
                anims.append(Rotate(cp, angle=PI, about_point=par))
        self.add(*[c for c, _ in copies])
        self.play(LaggedStart(*anims, lag_ratio=0.18), run_time=3.4)
        for cp, cell in copies:
            check(_same_poly(cp.get_vertices(), cell, 1e-6), "a copy of the corner tile lands on a cell")
        self.play(*[FadeOut(c) for c, _ in copies], run_time=0.5)
        self.bring_to_front(cuts, more, tks, dots, *labs.values())

        # ---- three corners of nine, the hexagon six of nine
        self.play(FadeIn(r1, shift=LEFT * 0.2), run_time=0.7)
        self.play(FadeIn(r2, shift=LEFT * 0.2), run_time=0.8)
        self.play(FadeIn(r3, shift=LEFT * 0.2), run_time=0.7)
        self.hold(0.3)

        cap = caption("[hexagon]  =  6/9 [ABC]  =  2/3 [ABC]", 36)
        _final_check(self, cap, "F63")
        self.play(Write(cap))
        self.hold(2.2)


# ============================================ F66 EXTERIOR ANGLE BISECTOR THEOREM

class F66_ExteriorBisector(Board):
    """AB > AC. The bisector of the exterior angle at A (between AC and BA
    produced) meets BC produced at E. Fold the right triangle APE (P the
    foot of E on the line AB) over AE: the two arms of the angle swap, so
    it lands on AQE (Q the foot on the line AC) and EP = EQ = h. So ABE and
    ACE have the same height h over AB and over AC: their areas are as
    AB : AC. They also have one apex A over the bases EB and EC on one line:
    their areas are as EB : EC. Hence EB : EC = AB : AC. (Any triangle with
    AB ≠ AC; drawn with AB > AC, so E lies beyond C; the feet P and Q fall
    on the sides produced.)"""

    def construct(self):
        a_, c_, b_ = 4.6, 4.4, 2.2
        x0 = (a_ ** 2 + c_ ** 2 - b_ ** 2) / (2 * a_)
        Am, Bm, Cm = to3((x0, np.sqrt(c_ ** 2 - x0 ** 2))), to3((0.0, 0.0)), to3((a_, 0.0))
        u1 = _unit(Am - Bm)                    # along BA produced beyond A
        u2 = _unit(Cm - Am)                    # along AC
        Em = _meet(Am, Am + (u1 + u2), Bm, Cm)  # the exterior bisector meets BC
        Pm, Qm = _foot(Em, Am, Bm), _foot(Em, Am, Cm)
        Fr = Frame(-0.35, Em[0] + 0.45, Qm[1] - 0.45, Pm[1] + 0.5, max_w=11.6,
                   centre=(-0.7, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, E, P, Q = (Fr.P(X) for X in (Am, Bm, Cm, Em, Pm, Qm))
        H = _foot(A, B, C)
        n_ = np.linalg.norm
        AB, AC, EB, EC = n_(B - A), n_(C - A), n_(B - E), n_(C - E)
        X1 = P + 0.35 * _unit(P - A)           # BA produced, a little past P
        Q1 = Q + 0.35 * _unit(Q - A)

        # ---- checks
        check(_scalene(A, B, C) and AB > AC + 0.5, "scalene, AB > AC")
        check(abs(_angle(A, P, E) - _angle(A, E, Q)) < 1e-9
              and abs(_angle(A, P, E) + _angle(A, E, Q) + _angle(A, B, C) - PI) < 1e-9,
              "AE bisects the exterior angle at A (between BA produced and AC)")
        check(_between(C, B, E), "E lies on BC produced beyond C")
        check(_on_line(P, A, B) and np.dot(P - A, A - B) > 0, "P lies on BA produced beyond A")
        check(_on_line(Q, A, C) and np.dot(Q - C, C - A) > 0, "Q lies on AC produced beyond C")
        check(close(_mirror(P, A, E), Q), "the fold over AE carries P to Q")
        h = n_(P - E)
        check(abs(n_(Q - E) - h) < 1e-9, "EP = EQ")
        sB, sC = abs(area([A, B, E])), abs(area([A, C, E]))
        check(abs(sB - AB * h / 2) < 1e-9 and abs(sC - AC * h / 2) < 1e-9,
              "[ABE] = ½ AB·h, [ACE] = ½ AC·h")
        check(abs(sB / sC - EB / EC) < 1e-9, "[ABE] : [ACE] = EB : EC")
        check(abs(EB / EC - AB / AC) < 1e-9, "EB : EC = AB : AC")
        check(_between(H, B, C), "the foot of A's height lies on BC")

        # ---- pieces
        cC, cY = ORANGE, YELLOW_B
        tri = mk([A, B, C], GREY_D, 0.5, stroke_width=4)
        extBA = DashedLine(A, X1, color=GREY_B, stroke_width=2.5, dash_length=0.09)
        extBC = DashedLine(C, E + 0.3 * _unit(E - C), color=GREY_B, stroke_width=2.5,
                           dash_length=0.09)
        extAC = DashedLine(C, Q1, color=GREY_B, stroke_width=2.5, dash_length=0.09)
        bis = Line(A, E, color=cY, stroke_width=4)
        rphi = 0.55
        arcs = VGroup(angle_arc(A, P, E, rphi, cY, 4), angle_arc(A, E, Q, rphi + 0.08, cY, 4))
        dE = Dot(E, radius=0.06)
        ep = Line(E, P, color=WHITE, stroke_width=3)
        eq = Line(E, Q, color=WHITE, stroke_width=3)
        raP, raQ = _ra(P, E, A, 0.17), _ra(Q, E, A, 0.17)
        ah = DashedLine(A, H, color=WHITE, stroke_width=2.5, dash_length=0.08)
        raH = _ra(H, A, C, 0.16)
        segs = _mob_segs(tri, extBA, extBC, extAC, bis, arcs, dE, ep, eq, raP, raQ, ah, raH,
                         Line(B, E))
        labs = {}
        for nm, X, d in (("A", A, 120), ("B", B, 200), ("C", C, 250), ("E", E, -30),
                         ("P", P, 60), ("Q", Q, 220), ("H", H, 270)):
            labs[nm] = _park(tag(nm, 30), X, segs, dirs=(d, d + 25, d - 25, d + 50, d - 50, d + 80,
                                                         d - 80), d0=0.24, d1=0.8,
                             avoid=list(labs.values()))
        lphi = [_angle_label(tag("φ", 26, cY), A, P, E, rphi + 0.25, segs=segs,
                             avoid=list(labs.values()))]
        lphi.append(_angle_label(tag("φ", 26, cY), A, E, Q, rphi + 0.3, segs=segs,
                                 avoid=list(labs.values()) + lphi))
        lh = [_park_beside(tag("h", 28, cY), E, P, segs, side=A - E, avoid=list(labs.values()) + lphi),
              None]
        lh[1] = _park_beside(tag("h", 28, cY), E, Q, segs, side=A - E,
                             avoid=list(labs.values()) + lphi + [lh[0]])
        labels = [*labs.values(), *lphi, *lh]
        _labels_ok(labels, segs, "F66 figure")

        # the ledger, top left (above the line BA)
        r1 = _trow([("[ABE]", BLUE_B), (":", WHITE), ("[ACE]", cC), ("=", WHITE), ("AB : AC", WHITE)],
                   28, 0.16)
        r2 = _trow([("[ABE]", BLUE_B), (":", WHITE), ("[ACE]", cC), ("=", WHITE), ("EB : EC", WHITE)],
                   28, 0.16)
        r1.move_to([0, 3.3, 0]).align_to(np.array([-SAFE_X + 0.1, 0, 0]), LEFT)
        r2.move_to([0, 2.65, 0]).align_to(np.array([-SAFE_X + 0.1, 0, 0]), LEFT)
        for r in (r1, r2):
            check(_hit(r, segs + _segs(A, B, C, closed=True), 0.15) is None
                  and all(not _overlap(r, l, 0.12) for l in labels) and _in_frame(r),
                  "the ledger is clear of the figure")

        # ---- the triangle; the exterior angle at A and its bisector
        self.play(FadeIn(tri), *[FadeIn(labs[k]) for k in "ABC"], run_time=1.0)
        self.play(Create(extBA), run_time=0.6)
        self.play(Create(bis), Create(arcs), *[FadeIn(m) for m in lphi], run_time=1.0)
        self.play(Create(extBC), FadeIn(dE), FadeIn(labs["E"]), run_time=0.8)

        # ---- the distances from E to the lines AB and AC; fold APE over AE
        self.play(Create(extAC), Create(ep), Create(eq), Create(raP), Create(raQ),
                  FadeIn(labs["P"]), FadeIn(labs["Q"]), run_time=1.0)
        half = mk([A, P, E], cY, 0.45, stroke_color=cY, stroke_width=3)
        self.play(FadeIn(half), run_time=0.4)
        self.play(_fold(half, A, E), run_time=1.8)
        check(_landed(half, [A, Q, E]), "the fold lands APE on AQE")
        self.play(FadeOut(half), FadeIn(lh[0]), FadeIn(lh[1]), run_time=0.6)

        # ---- the triangles ABE and ACE overlap (ACE lies inside ABE), so
        # each is shown in turn with the base and height in question
        def show(poly, colr, base, height, extra=()):
            fill = mk(poly, colr, 0.55, stroke_color=colr, stroke_width=4)
            b = Line(base[0], base[1], color=colr, stroke_width=10)
            hgt = height
            self.add(fill)
            self.bring_to_back(fill)
            self.bring_to_back(tri)
            self.play(FadeIn(fill), Create(b), Create(hgt), *extra, run_time=0.8)
            self.bring_to_front(*[m for m in labels if m in self.mobjects])
            self.hold(0.5)
            return VGroup(fill, b, hgt)

        def hide(g):
            self.play(FadeOut(g), run_time=0.35)

        # one height h over AB and over AC
        g = show([A, B, E], BLUE_D, (A, B), Line(E, P, color=cY, stroke_width=7))
        hide(g)
        g = show([A, C, E], cC, (A, C), Line(E, Q, color=cY, stroke_width=7))
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.7)
        hide(g)

        # one apex A over EB and over EC
        g = show([A, B, E], BLUE_D, (E, B), Line(A, H, color=cY, stroke_width=6),
                 extra=(Create(raH), FadeIn(labs["H"])))
        hide(g)
        g = show([A, C, E], cC, (E, C), Line(A, H, color=cY, stroke_width=6))
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.7)
        hide(g)

        # the finished figure: ABE filled, ACE outlined inside it
        fB = mk([A, B, E], BLUE_D, 0.45, stroke_width=0)
        oC = Polygon(A, C, E, stroke_color=cC, stroke_width=5)
        self.add(fB)
        self.bring_to_back(fB)
        self.bring_to_back(tri)
        self.play(FadeIn(fB), Create(oC), Create(ah), run_time=0.7)
        self.bring_to_front(*[m for m in labels if m in self.mobjects])
        check(all(m in self.mobjects for m in labels), "every label is on screen at the end")

        cap = caption("EB : EC  =  AB : AC", 38)
        _final_check(self, cap, "F66")
        self.play(Write(cap))
        self.hold(2.2)
