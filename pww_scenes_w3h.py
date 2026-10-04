# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3h.py — proofs without words, 2D (manim):
#     L11 order matters                      L15 determinants multiply
#     L12 projection is additive             F25 Menelaus's theorem
#     F48 the trapezoid's butterfly          F59 tangent lengths to the incircle
#     F55 the length of a median             F34 Pitot's theorem
#     F42 the area of a regular octagon      F51 Fagnano's problem
#
# Every move of a piece is rigid — a slide (shift), a turn about a named
# point (Rotate), a fold (a half-turn in space about a line of the plane,
# landing as the mirror image), or a turn about a moving centre that stays
# rigid on every frame — unless it is an announced shear (base kept, apex on
# its parallel), a homothety (turn-and-scale about its centre) or a linear
# map. Every landing, tiling, area, length and angle claim is checked with
# check(...) before or right after it is drawn, and every label is checked
# against the lines, marks and other labels it must clear, so a wrong
# construction fails the render instead of drawing a wrong picture.
# No LaTeX: every label is Text (tag / caption) with Unicode.


# ------------------------------------------------------------ private helpers

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
    return 2 * _foot(p, a, b) - to3(p)


def _on_line(p, a, b, tol=1e-9):
    return abs(_cross(to3(b) - to3(a), to3(p) - to3(a))) < tol * max(
        1.0, float(np.linalg.norm(to3(b) - to3(a))))


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _ra_segs(v, p, q, s=0.2):
    """The two strokes of a right-angle mark, as segments (label checks)."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


def _ticks(p, q, n=1, color=WHITE, size=0.12, width=2.5, at=0.5):
    """n short tick marks across segment pq (equal-length marks)."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = _perp(d)
    c = p + (q - p) * at
    g = VGroup()
    for k in range(n):
        o = c + d * 0.09 * (k - (n - 1) / 2)
        g.add(Line(o - nrm * size, o + nrm * size, color=color,
                   stroke_width=width))
    return g


def _tick_segs(p, q, n=1, size=0.12, at=0.5):
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = _perp(d)
    c = p + (q - p) * at
    return [(c + d * 0.09 * (k - (n - 1) / 2) - nrm * size,
             c + d * 0.09 * (k - (n - 1) / 2) + nrm * size) for k in range(n)]


def _chevron(p, q, color=YELLOW_B, size=0.16, width=3, n=1, at=0.5):
    """Parallel-line mark: n small '>' on segment pq, pointing p -> q."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = _perp(d)
    mid = p + (q - p) * at
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7,
                                          tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


def _chevron_segs(p, q, size=0.16, n=1, at=0.5):
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = _perp(d)
    mid = p + (q - p) * at
    out = []
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        out += [(tip - d * size + nrm * size * 0.7, tip),
                (tip, tip - d * size - nrm * size * 0.7)]
    return out


def _arrow(p, q, color, width=6, tip=0.26):
    return Arrow(to3(p), to3(q), buff=0.0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.2, tip_length=tip)


def _darrow(p, q, color=YELLOW_B, width=3):
    """Double arrow p <-> q (a height between two parallels)."""
    return DoubleArrow(to3(p), to3(q), buff=0, stroke_width=width,
                       color=color, tip_length=0.16,
                       max_tip_length_to_length_ratio=0.5)


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


def _landed(mob, pts, what, tol=1e-6):
    """After a move: the polygon's vertices are exactly pts, in order."""
    got = [to3(v) for v in mob.get_vertices()]
    want = [to3(p) for p in pts]
    check(len(got) == len(want)
          and all(close(g, w, tol) for g, w in zip(got, want)), what)


def _dashed_poly(pts, color=GREY_B, width=2, dashes=30):
    """Dashed outline of a polygon (a ghost of where a piece was)."""
    return DashedVMobject(Polygon(*[to3(p) for p in pts], stroke_color=color,
                                  stroke_width=width), num_dashes=dashes)


def _icon(poly, k, color, op=FILL):
    """A small copy of a piece, scaled by k (all icons share k, so their
    sizes compare as the pieces' areas do)."""
    return mk([to3(p) for p in poly], color, op, stroke_width=1.5).scale(k)


# ---- motions

def _rigid(mob, src, dst, turn=None, **kw):
    """Rigid motion of mob — a turn about its moving centre while that
    centre slides straight — carrying the screen points src onto dst (same
    order). Fails the render unless one rotation + translation lands every
    point. `turn` (radians) picks the direction for ±180°."""
    src = [to3(p) for p in src]
    dst = [to3(p) for p in dst]
    if turn is None:
        a0 = _dir(src[0], src[1])
        a1 = _dir(dst[0], dst[1])
        turn = (a1 - a0 + PI) % TAU - PI
    m0, m1 = sum(src) / len(src), sum(dst) / len(dst)
    check(all(close(m1 + _rot2(p, m0, turn) - m0, q, 1e-6)
              for p, q in zip(src, dst)), "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


def _homothety(mob, centre, factor, turn=0.0, **kw):
    """Enlargement about `centre`, drawn as a turn by `turn` together with
    a scaling that reaches `factor` (geometrically in time). With turn = π
    the end state is the homothety of ratio −factor."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=c)
                 .scale(factor ** alpha, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


# ---- text with true sub- and superscripts (no TeX, no letter glyphs that
# ---- some fonts lack)

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


def _scriptline(parts, size=30, color=WHITE, scale=0.64, sub_drop=0.32,
                sup_rise=0.48):
    """A line of text with true subscripts and superscripts. parts: list of
    (string, kind) or (string, kind, colour), kind 0 = normal, -1 =
    subscript, +1 = superscript. Normal pieces share one baseline."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for part in parts:
        s, kind = part[0], part[1]
        col = part[2] if len(part) > 2 else color
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" "))
        x += lead * sp
        t = _on_base(s.strip(" "), size * (scale if kind else 1.0), col)
        y0 = {0: 0.0, -1: -sub_drop * xh, 1: sup_rise * xh}[kind]
        t.shift(np.array([x - t.get_left()[0], y0, 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if kind else 0.02)
        g.add(t)
    return g


# ---- label hygiene: every label is checked against what it must clear

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


def _circle_segs(c, r, n=120):
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(0.0, TAU, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _arc_segs(c, r, a0, a1, n=40):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _angle_arc_segs(v, p, q, r, n=24):
    """The arc angle_arc(v, p, q, r) draws, as segments."""
    v = to3(v)
    a1, a2 = _dir(v, p), _dir(v, q)
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return _arc_segs(v, r, a1, a1 + span, n)


def _mob_segs(m):
    """Segments along every stroke of a mobject and its family (sampled
    from the Bezier points): for checking labels against curves, tips and
    dashed outlines."""
    out = []
    for sub in m.family_members_with_points():
        pts = sub.get_points()
        for i in range(0, len(pts) - 3, 4):
            a, b, c, d = pts[i:i + 4]
            prev = a
            for t in np.linspace(0.0, 1.0, 9)[1:]:
                x = ((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b
                     + 3 * (1 - t) * t * t * c + t ** 3 * d)
                out.append((prev, x))
                prev = x
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
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.015) + 2)):
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
    t = getattr(m, "text", None)
    if t:
        return t
    if isinstance(m, VGroup):
        s = "".join(getattr(x, "text", "") for x in m.submobjects)
        if s:
            return s
    return type(m).__name__


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every segment in segs
    (lines, ticks, arcs, marks) and of every other label."""
    for m in labels:
        check(_in_frame(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        sg = "" if i is None else (f" {np.round(segs[i][0][:2], 2)}-"
                                   f"{np.round(segs[i][1][:2], 2)}")
        check(i is None, f"{what}: '{_name(m)}' clear of the lines "
              f"(hits #{i}{sg})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _all_in_frame(mobs, what):
    for i, m in enumerate(mobs):
        check(_in_frame(m), f"{what}: item {i} ({_name(m)}) inside the safe "
              f"area [{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")


def _beside(m, p, q, side, gap=0.12, at=0.5):
    """Park label m beside the segment pq (at fraction `at` along it), on
    the side the vector `side` points to, with its box clear of the line
    by `gap` whatever the slope."""
    p, q = to3(p), to3(q)
    nrm = _perp(_unit(q - p))
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    hw, hh = m.width / 2, m.height / 2
    off = hw * abs(nrm[0]) + hh * abs(nrm[1]) + gap
    return m.move_to(p + at * (q - p) + off * nrm)


def _park(m, X, towards, segs, d0=0.3, d1=1.2, pad=0.07, avoid=(), sep=0.05):
    """Move label m next to the point X: along the middles of the angular
    gaps between the directions to `towards` (widest gap first), at the
    smallest distance where m clears every segment in segs and stays `sep`
    away from every mobject in `avoid`. Fails the render if no spot."""
    X = to3(X)
    angs = sorted(_dir(X, q) % TAU for q in towards)
    gaps = []
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        gaps.append((a1 - a0, (a0 + a1) / 2))
    gaps.sort(reverse=True)
    found = []
    for w, mid in gaps:
        for d in np.arange(d0, d1 + 1e-9, 0.03):
            m.move_to(X + d * _polar(mid))
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)):
                found.append((d, -w, mid))
                break
    check(bool(found), f"a free spot for the label '{_name(m)}'")
    d, _, mid = min(found)
    return m.move_to(X + d * _polar(mid))


def _park_beside(m, choices, segs, pad=0.06, avoid=(), prefer=None, only=False,
                 ats=(0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82)):
    """Park label m beside one of the segments in `choices` (tried in
    order, several positions along it; the side `prefer` points to first,
    or only that side if `only`) where it clears every segment in segs and
    every mobject in `avoid`."""
    for p, q in choices:
        p, q = to3(p), to3(q)
        nrm = _perp(_unit(q - p))
        if prefer is not None and np.dot(nrm, to3(prefer)) < 0:
            nrm = -nrm
        for at in ats:
            for sd in ((nrm,) if only else (nrm, -nrm)):
                for gap in (0.08, 0.13, 0.18, 0.24, 0.3):
                    _beside(m, p, q, sd, gap, at)
                    if (_hit(m, segs, pad) is None and _in_frame(m)
                            and not any(_overlap(m, o, 0.05) for o in avoid)):
                        return m
    check(False, f"a free spot for the label '{_name(m)}'")


def _bracket(p, q, nrm, text, gap=0.28, color=YELLOW_B, size=28, tick=0.11,
             buff=0.12, width=3, lab=None):
    """Dimension bar beside segment pq, `gap` away along the unit normal
    nrm, with end ticks; the label sits beyond the bar. It spans the whole
    segment, so it names the union of whatever lies along it."""
    p, q, n = to3(p), to3(q), _unit(nrm)
    a, b = p + gap * n, q + gap * n
    bar = VGroup(Line(a, b, color=color, stroke_width=width),
                 Line(a - tick * n, a + tick * n, color=color, stroke_width=width),
                 Line(b - tick * n, b + tick * n, color=color, stroke_width=width))
    if lab is None:
        lab = tag(text, size, color)
    half = abs(n[0]) * lab.width / 2 + abs(n[1]) * lab.height / 2
    lab.move_to((a + b) / 2 + n * (buff + half + 0.02))
    return bar, lab


def _bar_segs(bar):
    return [(ln.get_start(), ln.get_end()) for ln in bar]


def _final_check(scene, cap, extra_texts=()):
    """Closing frame: everything but the caption inside the safe area, the
    caption in its band, and no two text labels overlapping."""
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip]
    for m in shown:
        check(_in_frame(m, 0.03), f"{_name(m)} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")
    texts = []
    for m in shown:
        for t in m.get_family():
            if isinstance(t, Text) and t.width > 1e-6 and t.get_fill_opacity() > 0.05:
                texts.append(t)
    # pieces of one formula line (siblings of one VGroup) may touch
    parent = {}
    for m in shown:
        for g in m.get_family():
            if isinstance(g, VGroup) and not isinstance(g, Text):
                for t in g.submobjects:
                    if isinstance(t, Text):
                        parent.setdefault(id(t), id(g))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if parent.get(id(a)) is not None and parent.get(id(a)) == parent.get(id(b)):
                continue
            check(not _overlap(a, b, 0.0),
                  f"labels '{a.text}' and '{b.text}' apart")


# =========================================================== L11 ORDER MATTERS

def _flag_pieces():
    """A flag with no symmetry at all, in its own units: a pole (rectangle)
    and a pennant (triangle) flying from the top of the pole."""
    pole = [(0.30, 0.30), (0.52, 0.30), (0.52, 2.30), (0.30, 2.30)]
    pennant = [(0.52, 2.30), (0.52, 1.38), (1.82, 1.84)]
    return pole, pennant


class L11_OrderMatters(Board):
    """R is the quarter-turn about O (counter-clockwise), S the reflection
    in the vertical line through O. The same flag, which has no symmetry,
    goes through both orders side by side. Left: first R, then S — the
    product SR (the map applied first is written on the right). Right:
    first S, then R — the product RS. The two flags end in different places
    and face different ways (SR is the mirror in the line y = x, RS the
    mirror in y = −x), so RS ≠ SR."""

    def construct(self):
        k = 1.2
        OL = np.array([-3.3, 0.08, 0.0])
        OR = np.array([3.3, 0.08, 0.0])
        pole0, pen0 = _flag_pieces()
        Rm = np.array([[0.0, -1.0], [1.0, 0.0]])        # quarter-turn
        Sm = np.array([[-1.0, 0.0], [0.0, 1.0]])        # mirror x -> -x

        def scr(O, pts, M=np.eye(2)):
            return [O + k * to3(M @ np.asarray(p, float)) for p in pts]

        # ---- the claims
        SR, RS = Sm @ Rm, Rm @ Sm
        check(close(SR, [[0, 1], [1, 0]]) and close(RS, [[0, -1], [-1, 0]]),
              "SR is the mirror in y = x, RS the mirror in y = −x")
        far = max(np.linalg.norm(SR @ np.asarray(p) - RS @ np.asarray(p))
                  for p in pole0 + pen0)
        check(far > 2.0, "the two orders put the flag in different places")
        for M in (np.eye(2), Rm, Sm, SR, RS):
            for p in pole0 + pen0:
                q = M @ np.asarray(p)
                check(max(abs(q[0]), abs(q[1])) < 2.35, "the flag stays in its panel")

        # ---- two panels: the centre O, the mirror line, the turn arrow
        cR, cS = YELLOW_B, TEAL_B

        def panel(O):
            P = {}
            P["dot"] = Dot(O, radius=0.07, color=WHITE)
            P["mirror"] = DashedLine(O + 2.9 * DOWN, O + 2.95 * UP, color=cS,
                                     stroke_width=3, dash_length=0.12)
            arc = Arc(radius=0.8, start_angle=-84 * DEGREES, angle=PI / 2,
                      arc_center=O, color=cR, stroke_width=4)
            arc.add_tip(tip_length=0.2, tip_width=0.2)
            P["arc"] = arc
            P["R"] = tag("R", 32, cR).move_to(O + 1.22 * _polar(-42 * DEGREES))
            P["S"] = tag("S", 32, cS).move_to(O + np.array([-0.3, -2.62, 0.0]))
            P["ghost"] = VGroup(_dashed_poly(scr(O, pole0)),
                                _dashed_poly(scr(O, pen0)))
            return P

        def flag(O):
            return VGroup(mk(scr(O, pole0), GREY_A, 0.9, stroke_width=2),
                          mk(scr(O, pen0), ORANGE, 0.9, stroke_width=2))

        PL, PR = panel(OL), panel(OR)
        fL, fR = flag(OL), flag(OR)
        self.play(*[FadeIn(P[n]) for P in (PL, PR) for n in ("dot", "mirror")],
                  run_time=0.8)
        self.play(FadeIn(fL), FadeIn(fR), run_time=0.8)
        self.add(PL["ghost"], PR["ghost"])
        self.bring_to_back(PL["ghost"], PR["ghost"])
        self.hold(0.5)

        # heading slots: the product is written with the first map on the right
        yH = 3.42

        def slot(O, first):
            return np.array([O[0] + (0.2 if first else -0.2), yH, 0.0])

        def step_R(O, P, f, first):
            self.play(Create(P["arc"]), FadeIn(P["R"]), run_time=0.6)
            self.play(Rotate(f, angle=PI / 2, about_point=O), run_time=1.5)
            lr = P["R"].copy()
            self.play(lr.animate.scale(38 / 32).move_to(slot(O, first)),
                      run_time=0.6)
            return lr

        def step_S(O, P, f, first):
            self.play(P["mirror"].animate.set_stroke(width=6), FadeIn(P["S"]),
                      run_time=0.5)
            self.play(_fold(f, O, O + UP), run_time=1.6)
            self.play(P["mirror"].animate.set_stroke(width=3), run_time=0.3)
            ls = P["S"].copy()
            self.play(ls.animate.scale(38 / 32).move_to(slot(O, first)),
                      run_time=0.6)
            return ls

        # ---- left: first R, then S
        hL1 = step_R(OL, PL, fL, True)
        _landed(fL[0], scr(OL, pole0, Rm), "R turns the pole a quarter-turn")
        hL2 = step_S(OL, PL, fL, False)
        _landed(fL[0], scr(OL, pole0, SR), "then S mirrors it: SR")
        _landed(fL[1], scr(OL, pen0, SR), "the pennant lands at SR")
        self.hold(0.4)

        # ---- right: first S, then R
        hR1 = step_S(OR, PR, fR, True)
        _landed(fR[0], scr(OR, pole0, Sm), "S mirrors the pole")
        hR2 = step_R(OR, PR, fR, False)
        _landed(fR[0], scr(OR, pole0, RS), "then R turns it: RS")
        _landed(fR[1], scr(OR, pen0, RS), "the pennant lands at RS")

        neq = tag("≠", 40).move_to(np.array([0.0, yH, 0.0]))
        self.play(FadeIn(neq), run_time=0.6)

        # ---- label hygiene
        heads = [hL1, hL2, hR1, hR2, neq]
        for O, P, f in ((OL, PL, fL), (OR, PR, fR)):
            segs = (_mob_segs(P["mirror"]) + _mob_segs(P["arc"]) + _mob_segs(f)
                    + _mob_segs(P["ghost"]) + _mob_segs(P["dot"]))
            _labels_ok([P["R"], P["S"]], segs, "L11 panel")
        _labels_ok(heads + [PL["R"], PL["S"], PR["R"], PR["S"]], [],
                   "L11 headings")
        cap = caption("quarter-turn R, reflection S:   RS  ≠  SR", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)



# =========================================================== L15 DET(TS)

class L15_DeterminantsMultiply(Board):
    """Left, the plane before T; right, everything after T (the right-hand
    figure is always T of the left-hand one). S takes the unit square Q to
    the parallelogram P on its columns, of area det S; T takes Q to T(Q), of
    area det T, and P to TS(Q), of area det(TS). Shear P twice — first
    along its side s₁, then along its new vertical side — into a 2 × 2
    square R: shears keep areas. T maps lines to lines and keeps parallels,
    so the same two shears happen on the right, and keep the area det(TS)
    there too. R is det S = 4 translates of Q; T maps them to det S
    translates of T(Q) that fill T(R). So det(TS) = det S · det T.
    (Drawn with det S, det T > 0; orientations multiply the same way.)"""

    def construct(self):
        S = np.array([[2.0, 0.8], [0.6, 2.24]])
        T = np.array([[0.85, -0.4], [0.3, 0.75]])
        k = 1.85
        OL = np.array([-6.1, -2.75, 0.0])
        OR = np.array([2.2, -2.75, 0.0])

        def Lp(p):
            return OL + k * to3(p)

        def Rp(p):
            return OR + k * to3(T @ np.asarray(p, float))

        s1, s2 = S[:, 0], S[:, 1]
        s2p = s2 - (s2[0] / s1[0]) * s1              # shear 1: s₂ slides along s₁
        s1p = s1 - (s1[1] / s2p[1]) * s2p            # shear 2: s₁ slides along s₂'
        Z, e1, e2 = np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])
        Q = [Z, e1, e1 + e2, e2]
        P0 = [Z, s1, s1 + s2, s2]
        P1 = [Z, s1, s1 + s2p, s2p]
        R = [Z, s1p, s1p + s2p, s2p]
        offs = [Z, e1, e2, e1 + e2]
        cells = [[c + q for q in Q] for c in offs]
        dS, dT = float(np.linalg.det(S)), float(np.linalg.det(T))
        Tm = lambda P: [T @ p for p in P]

        # ---- the claims
        check(abs(area(P0) - dS) < 1e-12 and abs(area(Q) - 1) < 1e-12,
              "[Q] = 1, [S(Q)] = det S")
        check(abs(area(Tm(Q)) - dT) < 1e-12, "[T(Q)] = det T")
        check(abs(area(Tm(P0)) - np.linalg.det(T @ S)) < 1e-12,
              "[TS(Q)] = det(TS)")
        check(dS > 0 and dT > 0, "drawn with positive determinants")
        check(abs(_cross(s2p - s2, s1)) < 1e-12 and abs(area(P1) - area(P0)) < 1e-12,
              "shear 1 keeps the base s₁ and slides the top along it")
        check(abs(_cross(s1p - s1, s2p)) < 1e-12 and abs(area(R) - area(P1)) < 1e-12,
              "shear 2 keeps the side s₂' and slides the other along it")
        check(close(s1p, [2, 0]) and close(s2p, [0, 2]), "R is the 2 × 2 square")
        check(abs(_cross(T @ (s2p - s2), T @ s1)) < 1e-12
              and abs(_cross(T @ (s1p - s1), T @ s2p)) < 1e-12,
              "on the right the shears keep the images of the same lines")
        for P_ in (P0, P1, R):
            check(abs(area(Tm(P_)) - dT * area(P_)) < 1e-12,
                  "each shape on the right has det T times the area on the left")
        check(_tiles_exactly(cells, R), "four unit squares tile R")
        check(_tiles_exactly([Tm(c) for c in cells], Tm(R)),
              "their images, translates of T(Q), tile T(R)")
        for c in offs:
            check(_same_poly(Tm([c + q for q in Q]), [T @ c + p for p in Tm(Q)]),
                  "T of a translate of Q is a translate of T(Q)")
        check(abs(area(Tm(R)) - 4 * dT) < 1e-12 and abs(4 - dS) < 1e-12,
              "[T(R)] = (det S) · det T")

        cL, cR_, cY = BLUE_D, TEAL_D, YELLOW_B
        polyL = lambda P_, col=cL, op=0.8: mk([Lp(p) for p in P_], col, op, stroke_width=2.5)
        polyR = lambda P_, col=cR_, op=0.8: mk([Rp(p) for p in P_], col, op, stroke_width=2.5)
        cen = lambda pts: sum(to3(p) for p in pts) / len(pts)

        # ---- the unit square and T(Q); the map T between the two planes
        qL, qR = polyL(Q), polyR(Q)
        l1 = tag("1", 30).move_to(cen([Lp(p) for p in Q]))
        lT = tag("det T", 26).move_to(cen([Rp(p) for p in Q]))
        dotL, dotR = Dot(Lp(Z), radius=0.06), Dot(Rp(Z), radius=0.06)
        arrT = _arrow(np.array([-0.95, 1.6, 0.0]), np.array([0.55, 1.6, 0.0]), cY, 5)
        lTm = tag("T", 32, cY).next_to(arrT, UP, buff=0.1)
        self.play(FadeIn(qL), FadeIn(l1), FadeIn(dotL), run_time=0.8)
        self.play(GrowArrow(arrT), FadeIn(lTm), run_time=0.7)
        self.play(FadeIn(qR), FadeIn(lT), FadeIn(dotR), run_time=0.8)
        self.hold(0.4)

        # ---- S on the left (and so TS on the right)
        gL = _dashed_poly([Lp(p) for p in Q], GREY_B, 2, 24)
        gR = _dashed_poly([Rp(p) for p in Q], GREY_B, 2, 24)
        self.add(gL, gR)
        self.bring_to_back(gL, gR)
        arrS = _arrow(Lp(e1 + e2), Lp(s1 + s2), WHITE, 3, tip=0.2)
        lSm = tag("S", 30).move_to(Lp(s1 + s2) + 0.32 * _unit(Lp(s1 + s2) - Lp(e1 + e2)))
        lS = tag("det S", 28).move_to(cen([Lp(p) for p in P0]))
        lTS = tag("det(TS)", 26).move_to(cen([Rp(p) for p in P0]))
        self.play(GrowArrow(arrS), FadeIn(lSm), FadeOut(l1), FadeOut(lT), run_time=0.6)
        self.play(Transform(qL, polyL(P0)), Transform(qR, polyR(P0)), run_time=2.0)
        _landed(qL, [Lp(p) for p in P0], "Q lands on S(Q)")
        _landed(qR, [Rp(p) for p in P0], "T(Q) lands on TS(Q)")
        _labels_ok([lSm], _segs(*[Lp(p) for p in P0], closed=True) + _mob_segs(arrS)
                   + _mob_segs(arrT) + _mob_segs(lTm), "L15 S")
        self.play(FadeOut(arrS), FadeOut(lSm), FadeIn(lS), FadeIn(lTS), run_time=0.7)
        self.hold(0.5)
        self.play(FadeOut(gL), FadeOut(gR), run_time=0.3)

        # ---- two shears, the same on both sides
        def shear(Pa, Pb, base, slide, rt=1.8):
            """Pa -> Pb keeping the side `base` (two points) and sliding the
            opposite side along its own line `slide` (two points)."""
            gl = VGroup(guide(Lp(slide[0]), Lp(slide[1]), cY, extend=0.25),
                        guide(Rp(slide[0]), Rp(slide[1]), cY, extend=0.25))
            bl = VGroup(Line(Lp(base[0]), Lp(base[1]), color=cY, stroke_width=7),
                        Line(Rp(base[0]), Rp(base[1]), color=cY, stroke_width=7))
            self.play(Create(gl), Create(bl), run_time=0.6)
            self.play(Transform(qL, polyL(Pb)), Transform(qR, polyR(Pb)),
                      lS.animate.move_to(cen([Lp(p) for p in Pb])),
                      lTS.animate.move_to(cen([Rp(p) for p in Pb])), run_time=rt)
            _landed(qL, [Lp(p) for p in Pb], "a shear lands on the left")
            _landed(qR, [Rp(p) for p in Pb], "the same shear lands on the right")
            self.play(FadeOut(gl), FadeOut(bl), run_time=0.4)

        shear(P0, P1, (Z, s1), (s2, s1 + s2))
        shear(P1, R, (Z, s2p), (s1, s1 + s2p))
        self.hold(0.3)

        # ---- R is four unit squares; T(R) is four copies of T(Q)
        frL = Polygon(*[Lp(p) for p in R], stroke_color=cY, stroke_width=5)
        frR = Polygon(*[Rp(p) for p in R], stroke_color=cY, stroke_width=5)
        top = (Lp(R[3]) + Lp(R[2])) / 2
        self.play(lS.animate.move_to(top + 0.36 * UP).set_color(cY),
                  Create(frL), Create(frR), run_time=0.8)
        rsegs = _segs(*[Rp(p) for p in R], closed=True)
        _park_beside(lTS, [(Rp(R[1]), Rp(R[2])), (Rp(R[2]), Rp(R[3]))], rsegs,
                     prefer=Rp(R[1]) - Rp(R[3]), only=True)
        tgt_TS = lTS.get_center().copy()
        lTS.move_to(cen([Rp(p) for p in R]))
        self.play(lTS.animate.move_to(tgt_TS).set_color(cY), run_time=0.7)

        cL0 = mk([Lp(p) for p in cells[0]], BLUE_B, 0.85, stroke_width=2.5)
        cR0 = mk([Rp(p) for p in cells[0]], TEAL_B, 0.85, stroke_width=2.5)
        c1 = tag("1", 30, BLACK).move_to(cen([Lp(p) for p in cells[0]]))
        cT = tag("det T", 24, BLACK).move_to(cen([Rp(p) for p in cells[0]]))
        self.play(FadeIn(cL0), FadeIn(cR0), FadeIn(c1), FadeIn(cT), run_time=0.7)
        cps = []
        for c in offs[1:]:
            a = mk([Lp(p) for p in cells[0]], BLUE_B, 0.55, stroke_width=2.5)
            b = mk([Rp(p) for p in cells[0]], TEAL_B, 0.55, stroke_width=2.5)
            cps.append((a, b, c))
            self.add(a, b)
        self.bring_to_front(cL0, cR0, c1, cT, frL, frR)
        self.play(*[a.animate.shift(k * to3(c)) for a, b, c in cps],
                  *[b.animate.shift(k * to3(T @ c)) for a, b, c in cps], run_time=1.6)
        for (a, b, c), cl in zip(cps, cells[1:]):
            _landed(a, [Lp(p) for p in cl], "a unit square slides into its cell")
            _landed(b, [Rp(p) for p in cl], "its image slides into the image cell")
        self.bring_to_front(frL, frR, lS, lTS, dotL, dotR)
        self.hold(0.4)

        # ---- label hygiene
        segL = (_segs(*[Lp(p) for p in R], closed=True)
                + [(Lp(e1), Lp(e1 + 2 * e2)), (Lp(e2), Lp(2 * e1 + e2))])
        segR = (rsegs + [(Rp(e1), Rp(e1 + 2 * e2)), (Rp(e2), Rp(2 * e1 + e2))])
        _labels_ok([lS], segL, "L15 left", pad=0.06)
        _labels_ok([lTS], segR, "L15 right", pad=0.06)
        _labels_ok([c1], segL, "L15 left cell", pad=0.04)
        _labels_ok([cT], segR, "L15 right cell", pad=0.03)
        check(_pip(cT.get_corner(UL), [Rp(p) for p in cells[0]])
              and _pip(cT.get_corner(DR), [Rp(p) for p in cells[0]])
              and _pip(cT.get_corner(UR), [Rp(p) for p in cells[0]])
              and _pip(cT.get_corner(DL), [Rp(p) for p in cells[0]]),
              "det T sits inside T(Q)")
        _labels_ok([lTm, lS, lTS, c1, cT], _segs(arrT.get_start(), arrT.get_end()),
                   "L15 labels")
        for P_ in (Q, P0, P1, R):
            check(_hit(arrT, _segs(*[Lp(p) for p in P_], closed=True)
                       + _segs(*[Rp(p) for p in P_], closed=True), 0.1) is None,
                  "the arrow T stays clear of every stage of both figures")
            check(_hit(lTm, _segs(*[Lp(p) for p in P_], closed=True)
                       + _segs(*[Rp(p) for p in P_], closed=True), 0.1) is None,
                  "the label T stays clear of every stage of both figures")
        cap = caption("det(TS)  =  det T · det S", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== L12 PROJECTION

class L12_ProjectionAdditive(Board):
    """ℓ is the line of u. The shadow of an arrow on ℓ is cut out by the
    perpendiculars from its two ends, so an arrow slid anywhere keeps the
    same shadow, only slid along ℓ. Put w at the head of v: the shadows of v
    and of w lie end to end, and together they are the shadow of v + w,
    which runs from the tail of v to the head of w. Multiplying every
    shadow by |u| — a rectangle of height |u| on it — gives the dot products:
    u·(v + w) = u·v + u·w. (Drawn with all three shadows pointing along u;
    with signed shadows the same addition holds.)"""

    def construct(self):
        al = 10 * DEGREES
        uh, nh = _polar(al), _polar(al + PI / 2)
        O = np.array([-2.15, -1.15, 0.0])
        nu = 1.5
        v = 3.4 * _polar(60 * DEGREES)
        w = 4.2 * _polar(18 * DEGREES)
        V, W, VW = O + v, O + w, O + v + w
        sv, sw = float(np.dot(v, uh)), float(np.dot(w, uh))
        Fv, Fw, Fvw = O + sv * uh, O + sw * uh, O + (sv + sw) * uh
        Fw2 = V + np.dot(VW - V, uh) * uh - np.dot(V - O, nh) * nh   # foot of V+W

        # ---- the claims
        check(close(_foot(V, O, O + uh), Fv) and close(_foot(W, O, O + uh), Fw)
              and close(_foot(VW, O, O + uh), Fvw), "the feet of the perpendiculars")
        check(close(Fw2, Fvw), "the shadow of w moved to V ends at the foot of V + w")
        check(close(Fvw - Fv, Fw - O), "the slid shadow of w is the shadow of w, slid")
        check(sv > 0 and sw > 0 and abs(np.dot(v + w, uh) - (sv + sw)) < 1e-12,
              "shadows add: (v + w)·û = v·û + w·û (all positive here)")
        check(abs(nu * (sv + sw) - (nu * sv + nu * sw)) < 1e-12,
              "× |u|: u·(v + w) = u·v + u·w")
        Rv = [O, Fv, Fv - nu * nh, O - nu * nh]
        Rw = [Fv, Fvw, Fvw - nu * nh, Fv - nu * nh]
        Rvw = [O, Fvw, Fvw - nu * nh, O - nu * nh]
        check(_tiles_exactly([Rv, Rw], Rvw), "the two rectangles fill the third")
        check(abs(abs(area(Rv)) - nu * sv) < 1e-9 and abs(abs(area(Rw)) - nu * sw) < 1e-9,
              "the rectangles have areas u·v and u·w")

        cu, cv, cw, cs = BLUE_B, ORANGE, TEAL_B, YELLOW_B
        L0, L1 = O - 2.25 * uh, O + 8.25 * uh
        ell = DashedLine(L0, L1, color=GREY_B, stroke_width=2.5, dash_length=0.1)
        dotO = Dot(O, radius=0.06)
        au = _arrow(O - nu * uh, O, cu, 6)
        lu = _beside(tag("u", 32, cu), O - nu * uh, O, nh, 0.14)
        self.play(Create(ell), run_time=0.8)
        self.play(GrowArrow(au), FadeIn(lu), FadeIn(dotO), run_time=0.8)

        # ---- v and its shadow
        av = _arrow(O, V, cv, 6)
        lv = _beside(tag("v", 32, cv), O, V, -uh, 0.14, at=0.55)
        pv = DashedLine(V, Fv, color=GREY_A, stroke_width=2.5, dash_length=0.08)
        rv = _ra(Fv, V, O + 9 * uh, 0.17)
        shv = Line(O, Fv, color=cv, stroke_width=11)
        self.play(GrowArrow(av), FadeIn(lv), run_time=0.8)
        self.play(Create(pv), Create(rv), run_time=0.6)
        self.play(Create(shv), run_time=0.6)
        self.bring_to_front(dotO)

        # ---- w from O, with its shadow; then slid to the head of v
        aw = _arrow(O, W, cw, 6)
        shw = Line(O, Fw, color=cw, stroke_width=11)
        segs0 = (_mob_segs(ell) + _segs(O - nu * uh, O) + _segs(O, V) + _segs(V, Fv)
                 + _segs(O, W) + _segs(W, Fw) + _ra_segs(Fv, V, O + 9 * uh, 0.17)
                 + _ra_segs(Fw, W, O + 12 * uh, 0.17) + _mob_segs(av) + _mob_segs(au))
        lw = _park_beside(tag("w", 32, cw), [(O, W)], segs0, avoid=[lu, lv],
                          prefer=nh, only=True, ats=(0.8, 0.75, 0.85, 0.7))
        tip = ValueTracker(0.0)

        def wtip():
            return W + tip.get_value() * v

        pw = always_redraw(lambda: DashedLine(wtip(), _foot(wtip(), O, O + uh),
                                              color=GREY_A, stroke_width=2.5,
                                              dash_length=0.08))
        rw = always_redraw(lambda: _ra(_foot(wtip(), O, O + uh), wtip(),
                                       O + 12 * uh, 0.17))
        self.play(GrowArrow(aw), FadeIn(lw), run_time=0.8)
        self.add(pw, rw)
        self.play(FadeIn(pw), FadeIn(rw), run_time=0.4)
        self.play(Create(shw), run_time=0.6)
        self.hold(0.4)
        lw2 = _beside(tag("w", 32, cw), V, VW, nh, 0.12, at=0.55)
        self.play(aw.animate.shift(v), shw.animate.shift(Fv - O),
                  tip.animate.set_value(1.0), Transform(lw, lw2), run_time=2.0)
        check(close(aw.get_start(), V, 1e-6) and close(aw.get_end(), VW, 1e-6),
              "w now starts at the head of v")
        check(close(shw.get_start(), Fv, 1e-6) and close(shw.get_end(), Fvw, 1e-6),
              "its shadow has slid along ℓ, end to end with the shadow of v")
        pw.clear_updaters()
        rw.clear_updaters()

        # ---- v + w: its shadow is the two shadows together
        avw = _arrow(O, VW, cs, 6)
        lvw = tag("v + w", 30, cs)
        self.play(GrowArrow(avw), run_time=0.9)
        sh_all = Line(O + 0.17 * nh, Fvw + 0.17 * nh, color=cs, stroke_width=5)
        ends = VGroup(Line(O + 0.07 * nh, O + 0.27 * nh, color=cs, stroke_width=4),
                      Line(Fvw + 0.07 * nh, Fvw + 0.27 * nh, color=cs, stroke_width=4))
        self.play(Create(sh_all), FadeIn(ends), run_time=0.8)
        self.hold(0.5)

        # ---- times |u|: rectangles of height |u| on the shadows
        uc = Line(O - nu * uh, O, color=cu, stroke_width=6)
        self.add(uc)
        self.play(Rotate(uc, angle=PI / 2, about_point=O), run_time=1.0)
        check(close(uc.get_start(), O - nu * nh, 1e-6), "u stands up as the side |u|")
        lnu = _beside(tag("|u|", 28, cu), O - nu * nh, O, -uh, 0.12)
        rect_v = mk(Rv, cv, 0.75, stroke_width=2)
        rect_w = mk(Rw, cw, 0.75, stroke_width=2)
        self.play(FadeIn(lnu), run_time=0.4)
        self.add(rect_v, rect_w)
        self.bring_to_front(uc, shv, shw, dotO)
        self.play(FadeIn(rect_v), FadeIn(rect_w), run_time=0.9)
        luv = tag("u·v", 30, BLACK).move_to(sum(Rv) / 4)
        luw = tag("u·w", 30, BLACK).move_to(sum(Rw) / 4)
        self.play(FadeIn(luv), FadeIn(luw), run_time=0.6)
        bar, lsum = _bracket(Rvw[3], Rvw[2], -nh, "u·(v + w)", gap=0.22, color=cs,
                             size=30)
        self.play(FadeIn(bar), FadeIn(lsum), run_time=0.8)

        # ---- the label for v + w, beside its arrow
        segs = (_mob_segs(ell) + _segs(O - nu * uh, O) + _segs(O, V) + _segs(V, VW)
                + _segs(O, VW) + _segs(V, Fv) + _segs(VW, Fvw)
                + _ra_segs(Fv, V, O + 9 * uh, 0.17) + _ra_segs(Fvw, VW, O + 12 * uh, 0.17)
                + _segs(*Rv, closed=True) + _segs(*Rw, closed=True)
                + _segs(O + 0.17 * nh, Fvw + 0.17 * nh) + _bar_segs(bar)
                + _segs(O + 0.07 * nh, O + 0.27 * nh) + _segs(Fvw + 0.07 * nh, Fvw + 0.27 * nh)
                + _mob_segs(au) + _mob_segs(av) + _mob_segs(aw) + _mob_segs(avw))
        _park_beside(lvw, [(O, VW)], segs, avoid=[lu, lv, lw, lnu],
                     prefer=-nh, only=True, ats=(0.6, 0.66, 0.54, 0.72, 0.48))
        self.play(FadeIn(lvw), run_time=0.5)
        self.hold(0.3)

        _labels_ok([lu, lv, lw, lvw, lnu, lsum], segs + _circle_segs(O, 0.06, 16),
                   "L12 labels")
        rsegs = _segs(*Rv, closed=True) + _segs(*Rw, closed=True)
        _labels_ok([luv, luw], rsegs + _segs(O, Fvw), "L12 rectangle labels")
        check(_pip(luv.get_left(), Rv) and _pip(luv.get_right(), Rv)
              and _pip(luw.get_left(), Rw) and _pip(luw.get_right(), Rw),
              "the dot products sit in their rectangles")
        cap = caption("u · (v + w)  =  u · v  +  u · w", 38)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F25 MENELAUS

class F25_Menelaus(Board):
    """A line ℓ cuts AB at F, CA at E and BC produced at D. Drop the
    perpendiculars hA, hB, hC from A, B, C to ℓ; they are parallel. At F the
    right triangles AFA' and BFB' have the same angle at F, so a half-turn
    about F followed by an enlargement by hB/hA carries one exactly onto the
    other: AF/FB = hA/hB. In the same way at D (enlargement only, B and C
    on one side of ℓ): BD/DC = hB/hC, and at E: CE/EA = hC/hA. The product
    of the three ratios is hA/hB · hB/hC · hC/hA = 1. (Lengths unsigned; the
    same holds when ℓ cuts all three sides produced.)"""

    def construct(self):
        Am, Bm, Cm = np.array([0.6, 3.5]), np.array([0.0, 0.0]), np.array([5.0, 0.0])
        um = np.array([np.cos(28 * DEGREES), np.sin(28 * DEGREES)])
        Pm = np.array([2.5, 2.65])
        Fm = _meet(Am, Bm, Pm, Pm + um)[:2]
        Dm = _meet(Bm, Cm, Pm, Pm + um)[:2]
        Em = _meet(Cm, Am, Pm, Pm + um)[:2]
        Lm0, Lm1 = Dm - 0.45 * um, Pm + 1.75 * um
        pts = np.array([Am, Bm, Cm, Dm, Lm0, Lm1])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        Fr = Frame(lo[0] - 0.1, hi[0] + 0.1, lo[1] - 0.25, hi[1] + 0.2, max_w=8.7,
                   centre=(-2.05, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D, E, F = (Fr.P(p) for p in (Am, Bm, Cm, Dm, Em, Fm))
        L0, L1 = Fr.P(Lm0), Fr.P(Lm1)
        A1, B1, C1 = (_foot(X, L0, L1) for X in (A, B, C))
        n = np.linalg.norm
        hA, hB, hC = n(A - A1), n(B - B1), n(C - C1)

        # ---- the claims
        check(0 < np.dot(F - A, B - A) < n(B - A) ** 2 and 0 < np.dot(E - C, A - C) < n(A - C) ** 2,
              "ℓ cuts AB and CA inside")
        check(np.dot(D - B, C - B) < 0, "and BC produced beyond B")
        check(_cross(L1 - L0, A - L0) * _cross(L1 - L0, B - L0) < 0
              and _cross(L1 - L0, B - L0) * _cross(L1 - L0, C - L0) > 0,
              "A on one side of ℓ, B and C on the other")
        for X1, X in ((A1, A), (B1, B), (C1, C)):
            check(abs(np.dot(X - X1, L1 - L0)) < 1e-9, "the perpendiculars")
            check(0 < np.dot(X1 - L0, L1 - L0) < n(L1 - L0) ** 2, "feet on the drawn line")
        check(abs(n(A - F) / n(F - B) - hA / hB) < 1e-9, "AF/FB = hA/hB")
        check(abs(n(B - D) / n(D - C) - hB / hC) < 1e-9, "BD/DC = hB/hC")
        check(abs(n(C - E) / n(E - A) - hC / hA) < 1e-9, "CE/EA = hC/hA")
        check(abs(n(A - F) / n(F - B) * n(B - D) / n(D - C) * n(C - E) / n(E - A) - 1) < 1e-9,
              "the product is 1")

        def hom(src, ctr, f, turn):
            return [ctr + f * (_rot2(p, ctr, turn) - ctr) for p in src]

        check(_same_poly(hom([F, A, A1], F, hB / hA, PI), [F, B, B1], 1e-9),
              "at F: half-turn and ×hB/hA carry AFA' onto BFB'")
        check(_same_poly(hom([D, B, B1], D, hC / hB, 0.0), [D, C, C1], 1e-9),
              "at D: ×hC/hB carries DBB' onto DCC'")
        check(_same_poly(hom([E, C, C1], E, hA / hC, PI), [E, A, A1], 1e-9),
              "at E: half-turn and ×hA/hC carry ECC' onto EAA'")

        cA, cB, cC, cL = BLUE_B, TEAL_B, ORANGE, YELLOW_B
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        ext = DashedLine(B, D, color=GREY_B, stroke_width=2.5, dash_length=0.09)
        ell = Line(L0, L1, color=cL, stroke_width=3.5)
        dots = VGroup(*[Dot(X, radius=0.06, color=cL) for X in (D, E, F)])
        perp = {"A": Line(A, A1, color=cA, stroke_width=4),
                "B": Line(B, B1, color=cB, stroke_width=4),
                "C": Line(C, C1, color=cC, stroke_width=4)}
        ras = VGroup(_ra(A1, A, L1, 0.15), _ra(B1, B, L1, 0.15), _ra(C1, C, L1, 0.15))

        segs = (_segs(A, B, C, closed=True) + _segs(B, D) + _segs(L0, L1)
                + _segs(A, A1) + _segs(B, B1) + _segs(C, C1)
                + _ra_segs(A1, A, L1, 0.15) + _ra_segs(B1, B, L1, 0.15)
                + _ra_segs(C1, C, L1, 0.15)
                + [sg for X in (D, E, F) for sg in _circle_segs(X, 0.06, 16)])
        lA = _park(tag("A", 30), A, [B, C, A1], segs, d0=0.3)
        lB = _park(tag("B", 30), B, [A, C, D, B1], segs, d0=0.3, avoid=[lA])
        lC = _park(tag("C", 30), C, [A, B, C1], segs, d0=0.3, avoid=[lA, lB])
        lD = _park(tag("D", 30), D, [L1, C, L0], segs, d0=0.3, avoid=[lA, lB, lC])
        lE = _park(tag("E", 30), E, [A, C, L0, L1], segs, d0=0.3, avoid=[lA, lB, lC, lD])
        lF = _park(tag("F", 30), F, [A, B, L0, L1], segs, d0=0.3,
                   avoid=[lA, lB, lC, lD, lE])
        names = [lA, lB, lC, lD, lE, lF]

        def hlab(s, col):
            return _scriptline([("h", 0), (s, -1)], 30, col)

        lhA = _park_beside(hlab("A", cA), [(A, A1)], segs, avoid=names)
        lhB = _park_beside(hlab("B", cB), [(B, B1)], segs, avoid=names + [lhA])
        lhC = _park_beside(hlab("C", cC), [(C, C1)], segs, avoid=names + [lhA, lhB])
        hl = [lhA, lhB, lhC]

        self.play(Create(tri), *[FadeIn(m) for m in (lA, lB, lC)], run_time=1.0)
        self.play(Create(ell), Create(ext), FadeIn(dots), *[FadeIn(m) for m in (lD, lE, lF)],
                  run_time=1.2)
        self.play(*[Create(m) for m in perp.values()], Create(ras), *[FadeIn(m) for m in hl],
                  run_time=1.2)
        self.hold(0.4)

        # ---- the ledger
        x0 = 3.05

        def row(parts, y, size=28):
            g = _scriptline(parts, size)
            g.move_to(np.array([0.0, y, 0.0])).align_to(np.array([x0, 0.0, 0.0]), LEFT)
            return g

        W_ = WHITE
        r1 = row([("AF / FB  =  ", 0, W_), ("h", 0, cA), ("A", -1, cA), (" / ", 0, W_),
                  ("h", 0, cB), ("B", -1, cB)], 2.7)
        r2 = row([("BD / DC  =  ", 0, W_), ("h", 0, cB), ("B", -1, cB), (" / ", 0, W_),
                  ("h", 0, cC), ("C", -1, cC)], 1.85)
        r3 = row([("CE / EA  =  ", 0, W_), ("h", 0, cC), ("C", -1, cC), (" / ", 0, W_),
                  ("h", 0, cA), ("A", -1, cA)], 1.0)

        def pair(P1, col1, P2, col2, ctr, f, turn, rw):
            t1 = mk(P1, col1, 0.45, stroke_width=0)
            t2 = mk(P2, col2, 0.45, stroke_width=0)
            self.add(t1, t2)
            self.bring_to_back(t1, t2)
            self.play(FadeIn(t1), FadeIn(t2), run_time=0.5)
            cp = mk(P1, col1, 0.7, stroke_color=WHITE, stroke_width=2)
            self.add(cp)
            self.play(_homothety(cp, ctr, f, turn), run_time=1.6)
            _landed(cp, hom(P1, ctr, f, turn), "the copy lands")
            check(_same_poly([to3(v) for v in cp.get_vertices()], P2, 1e-6),
                  "the turned and enlarged copy is exactly the other triangle")
            self.play(FadeIn(rw, shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeOut(cp), FadeOut(t1), FadeOut(t2), run_time=0.5)

        pair([F, A, A1], cA, [F, B, B1], cB, F, hB / hA, PI, r1)
        pair([D, B, B1], cB, [D, C, C1], cC, D, hC / hB, 0.0, r2)
        pair([E, C, C1], cC, [E, A, A1], cA, E, hA / hC, PI, r3)

        # ---- the product: each height once above the bar, once below
        num = _scriptline([("h", 0, cA), ("A", -1, cA), (" · ", 0, W_), ("h", 0, cB),
                           ("B", -1, cB), (" · ", 0, W_), ("h", 0, cC), ("C", -1, cC)], 28)
        den = _scriptline([("h", 0, cB), ("B", -1, cB), (" · ", 0, W_), ("h", 0, cC),
                           ("C", -1, cC), (" · ", 0, W_), ("h", 0, cA), ("A", -1, cA)], 28)
        yb = -0.55
        lhs = tag("×", 30, GREY_A).move_to(np.array([x0 + 0.12, yb, 0.0]))
        xb = x0 + 0.55
        wb = max(num.width, den.width) + 0.16
        bar = Line(np.array([xb, yb, 0.0]), np.array([xb + wb, yb, 0.0]),
                   color=W_, stroke_width=2.5)
        num.move_to(np.array([xb + wb / 2, yb + 0.36, 0.0]))
        den.move_to(np.array([xb + wb / 2, yb - 0.4, 0.0]))
        one = tag("=  1", 28).move_to(np.array([0.0, yb, 0.0]))
        one.align_to(np.array([xb + wb + 0.2, 0.0, 0.0]), LEFT)
        self.play(FadeIn(lhs), Create(bar), FadeIn(num), FadeIn(den), run_time=0.9)
        # pieces of num/den: 0 h 1 sub | 2 · | 3 h 4 sub | 5 · | 6 h 7 sub
        for i, j in ((3, 0), (6, 3), (0, 6)):
            grp = VGroup(num[i], num[i + 1], den[j], den[j + 1])
            self.play(Indicate(grp, scale_factor=1.25), run_time=0.45)
            self.play(grp.animate.set_opacity(0.38), run_time=0.3)
        self.play(FadeIn(one, shift=RIGHT * 0.2), run_time=0.6)

        _labels_ok(names + hl, segs, "F25 figure")
        _labels_ok([r1, r2, r3, lhs, num, den, one], _segs(bar.get_start(), bar.get_end()),
                   "F25 ledger")
        check(min(m.get_left()[0] for m in (r1, r2, r3, lhs)) >
              max(tri.get_right()[0], ell.get_right()[0], lC.get_right()[0]) + 0.3,
              "the ledger sits right of the figure")
        cap = caption("(AF / FB) · (BD / DC) · (CE / EA)  =  1", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F48 BUTTERFLY

class F48_TrapezoidButterfly(Board):
    """ABCD is a trapezoid with AB ∥ DC; its diagonals meet at O. The
    triangles ABD and ABC stand on the same base AB with their apexes D, C
    on the parallel DC, so they have the same height: slide the apex D
    along DC to C (a shear) and ABD becomes ABC with the same area. Each
    of them is the bottom triangle ABO together with one side triangle
    (AOD, resp. BOC); taking the same piece ABO away from equal areas
    leaves equal areas: [AOD] = [BOC]."""

    def construct(self):
        Am, Bm, Cm, Dm = (np.array([0.0, 0.0]), np.array([6.2, 0.0]),
                          np.array([4.3, 3.1]), np.array([1.1, 3.1]))
        Fr = Frame(-0.45, 6.65, -0.75, 3.55, max_w=7.8,
                   centre=(-2.45, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D = (Fr.P(p) for p in (Am, Bm, Cm, Dm))
        O = _meet(A, C, B, D)
        n = np.linalg.norm
        HD, HC = _foot(D, A, B), _foot(C, A, B)

        # ---- the claims
        check(abs(_cross(B - A, C - D)) < 1e-9, "AB ∥ DC")
        check(abs(n(D - HD) - n(C - HC)) < 1e-9, "D and C at the same height over AB")
        check(abs(abs(area([A, B, D])) - abs(area([A, B, C]))) < 1e-9, "[ABD] = [ABC]")
        check(_tiles_exactly([[A, O, D], [A, B, O]], [A, B, D]), "ABD = AOD + ABO")
        check(_tiles_exactly([[B, C, O], [A, B, O]], [A, B, C]), "ABC = BOC + ABO")
        check(abs(abs(area([A, O, D])) - abs(area([B, C, O]))) < 1e-9, "[AOD] = [BOC]")
        check(abs(abs(area([A, O, D])) - abs(area([D, O, C]))) > 0.3
              and abs(abs(area([A, B, O])) - abs(area([A, O, D]))) > 0.3,
              "the other two triangles are visibly different")

        cAOD, cBOC, cABO = BLUE_D, ORANGE, GREY_B
        trap = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=4)
        marks = VGroup(_chevron(A, B), _chevron(D, C))
        segs = (_segs(A, B, C, D, closed=True) + _segs(A, C) + _segs(B, D)
                + _chevron_segs(A, B) + _chevron_segs(D, C))
        lA = _park(tag("A", 30), A, [B, C, D], segs)
        lB = _park(tag("B", 30), B, [A, C, D], segs, avoid=[lA])
        lC = _park(tag("C", 30), C, [A, B, D], segs, avoid=[lA, lB])
        lD = _park(tag("D", 30), D, [A, B, C], segs, avoid=[lA, lB, lC])
        self.play(Create(trap), FadeIn(marks), *[FadeIn(m) for m in (lA, lB, lC, lD)],
                  run_time=1.2)
        diags = VGroup(Line(A, C, color=WHITE, stroke_width=3),
                       Line(B, D, color=WHITE, stroke_width=3))
        lO = _park(tag("O", 28), O, [A, B, C, D], segs, d0=0.25,
                   avoid=[lA, lB, lC, lD])
        # O's label goes in the top triangle DOC, the one not in the story
        check(_pip(lO.get_center(), [D, O, C]), "O is labelled inside DOC")
        self.play(Create(diags), FadeIn(Dot(O, radius=0.06)), FadeIn(lO), run_time=0.9)

        # ---- ABD: the side triangle AOD and the bottom triangle ABO
        pAOD = mk([A, O, D], cAOD, 0.8, stroke_width=0)
        pABO = mk([A, B, O], cABO, 0.55, stroke_width=0)
        pBOC = mk([B, C, O], cBOC, 0.8, stroke_width=0)
        outD = Polygon(A, B, D, stroke_color=BLUE_B, stroke_width=6)
        self.add(pAOD, pABO)
        self.bring_to_back(pAOD, pABO)
        self.play(FadeIn(pAOD), FadeIn(pABO), Create(outD), run_time=0.9)
        self.hold(0.3)

        # ---- same base AB, apex on the parallel: equal heights
        hD = _darrow(D, HD, YELLOW_B)
        hC = _darrow(C, HC, YELLOW_B)
        gl = guide(D, C, YELLOW_B, extend=0.35)
        _labels_ok([lA, lB, lC, lD, lO], segs + _segs(gl.get_start(), gl.get_end())
                   + _segs(D, HD) + _segs(C, HC), "F48 labels with the heights")
        self.play(FadeIn(hD), FadeIn(hC), Create(gl), run_time=0.8)

        # ---- the shear: D slides along DC to C; ABD becomes ABC
        sh = mk([A, B, D], BLUE_B, 0.25, stroke_color=BLUE_B, stroke_width=6)
        self.remove(outD)
        self.add(sh)
        self.play(Transform(sh, mk([A, B, C], BLUE_B, 0.25, stroke_color=BLUE_B,
                                   stroke_width=6)), run_time=2.0)
        _landed(sh, [A, B, C], "the apex lands on C")
        self.play(FadeOut(hD), FadeOut(hC), FadeOut(gl), run_time=0.4)
        self.add(pBOC)
        self.bring_to_back(pBOC)
        self.play(FadeIn(pBOC), sh.animate.set_fill(opacity=0).set_stroke(ORANGE),
                  run_time=0.8)

        # ---- the ledger: equal triangles, less the same piece
        k_ic = 0.2
        x0 = 2.6

        def comp(side, scol):
            return VGroup(_icon(side, 1.0, scol), _icon([A, B, O], 1.0, cABO, 0.55))

        def icon(g):
            return g.scale(k_ic)

        r1 = VGroup(icon(comp([A, O, D], cAOD)), tag("=", 34),
                    icon(comp([B, C, O], cBOC))).arrange(RIGHT, buff=0.3)
        r1.move_to(np.array([x0 + r1.width / 2, 2.2, 0.0]))
        r2 = VGroup(icon(VGroup(_icon([A, O, D], 1.0, cAOD))), tag("=", 34),
                    icon(VGroup(_icon([B, C, O], 1.0, cBOC)))).arrange(RIGHT, buff=0.3)
        r2.move_to(np.array([x0 + r2.width / 2, 0.2, 0.0]))
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.4)

        # take the common piece away, in the ledger and in the figure
        g1, g2 = r1[0][1], r1[2][1]
        self.play(Indicate(pABO, color=WHITE, scale_factor=1.0), Indicate(g1, scale_factor=1.15),
                  Indicate(g2, scale_factor=1.15), run_time=0.8)
        self.play(pABO.animate.set_fill(opacity=0.12), g1.animate.set_opacity(0.15),
                  g2.animate.set_opacity(0.15), FadeOut(sh), run_time=0.7)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.8)
        self.hold(0.3)

        _labels_ok([lA, lB, lC, lD, lO], segs + _circle_segs(O, 0.06, 16), "F48 labels")
        check(min(m.get_left()[0] for m in (r1, r2)) > max(B[0], lB.get_right()[0]) + 0.3,
              "the ledger sits right of the figure")
        _all_in_frame([r1, r2, trap], "F48")
        cap = caption("[AOD]  =  [BOC]", 38)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F59 TANGENT LENGTHS

class F59_TangentLengths(Board):
    """The incircle touches BC, CA, AB at D, E, F. The radius to a touch
    point is perpendicular to the side, so folding the triangle AFI over
    AI (the bisector) lands it on AEI: AF = AE = x; likewise BD = BF = y and
    CD = CE = z. Cut the perimeter at the touch points and lay the six
    pieces in two rows, each x + y + z: together they are a + b + c = 2s,
    so each row is s. The second row is x followed by the whole side BC,
    which is y + z = a: so x = s − a, and in the same way y = s − b,
    z = s − c."""

    def construct(self):
        am, bm, cm = 5.0, 4.2, 3.4
        Ax = (cm * cm - bm * bm + am * am) / (2 * am)
        Am, Bm, Cm = np.array([Ax, np.sqrt(cm * cm - Ax * Ax)]), np.zeros(2), np.array([am, 0.0])
        k = 1.15
        org = np.array([-3.05, 0.02, 0.0])
        P = lambda p: org + k * to3(p)
        A, B, C = P(Am), P(Bm), P(Cm)
        n = np.linalg.norm
        a, b, c = n(C - B), n(A - C), n(B - A)
        s = (a + b + c) / 2
        x, y, z = s - a, s - b, s - c
        I = (a * A + b * B + c * C) / (a + b + c)
        r = abs(area([A, B, C])) / s
        D, E, F = B + y * _unit(C - B), C + z * _unit(A - C), A + x * _unit(B - A)

        # ---- the claims
        for Q in (D, E, F):
            check(abs(n(Q - I) - r) < 1e-9, "the incircle touches the sides")
        check(abs(np.dot(D - I, C - B)) < 1e-9 and abs(np.dot(E - I, A - C)) < 1e-9
              and abs(np.dot(F - I, B - A)) < 1e-9, "radius ⟂ side at the touch points")
        check(close(_mirror(F, A, I), E) and close(_mirror(D, B, I), F)
              and close(_mirror(E, C, I), D), "the folds over AI, BI, CI swap the touch points")
        check(abs(n(F - A) - n(E - A)) < 1e-9 and abs(n(D - B) - n(F - B)) < 1e-9
              and abs(n(E - C) - n(D - C)) < 1e-9, "equal tangents")
        check(abs(x + y - c) < 1e-9 and abs(y + z - a) < 1e-9 and abs(z + x - b) < 1e-9,
              "c = x + y, a = y + z, b = z + x")
        check(abs(x + y + z - s) < 1e-9, "x + y + z = s")
        check(min(x, y, z) > 0.9 and max(x, y, z) - min(x, y, z) > 1.0, "three distinct lengths")

        cx, cy, cz = BLUE_B, TEAL_B, ORANGE
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=3.5)
        circ = Circle(radius=r, color=GREY_A, stroke_width=3).move_to(I)
        dI = Dot(I, radius=0.06)
        segs = (_segs(A, B, C, closed=True) + _circle_segs(I, r))
        lA = _park(tag("A", 30), A, [B, C], segs, d0=0.3)
        lB = _park(tag("B", 30), B, [A, C], segs, d0=0.3)
        lC = _park(tag("C", 30), C, [A, B], segs, d0=0.3)
        self.play(Create(tri), *[FadeIn(m) for m in (lA, lB, lC)], run_time=1.0)
        self.play(Create(circ), FadeIn(dI), run_time=0.9)

        # ---- radii ⟂ sides, and the bisectors from the vertices
        rads = VGroup(*[Line(I, Q, color=GREY_A, stroke_width=2.5) for Q in (D, E, F)])
        ras = VGroup(_ra(D, I, C, 0.14), _ra(E, I, A, 0.14), _ra(F, I, B, 0.14))
        bis = VGroup(*[DashedLine(V, I, color=GREY_B, stroke_width=2, dash_length=0.08)
                       for V in (A, B, C)])
        self.play(Create(rads), Create(ras), Create(bis), run_time=1.0)

        # ---- fold each kite half over its bisector: equal tangents
        halves = [(mk([A, F, I], cx, 0.6, stroke_width=1.5), A, [A, E, I]),
                  (mk([B, D, I], cy, 0.6, stroke_width=1.5), B, [B, F, I]),
                  (mk([C, E, I], cz, 0.6, stroke_width=1.5), C, [C, D, I])]
        self.play(*[FadeIn(h) for h, _, _ in halves], run_time=0.5)
        self.play(*[_fold(h, V, I) for h, V, _ in halves], run_time=1.6)
        for h, V, dst in halves:
            _landed(h, dst, "a folded half lands on the other half")

        # ---- the six tangent pieces, coloured by vertex
        def piece(p, q, col):
            return Line(p, q, color=col, stroke_width=9)

        AF, FB = piece(A, F, cx), piece(F, B, cy)
        BD, DC = piece(B, D, cy), piece(D, C, cz)
        CE, EA = piece(C, E, cz), piece(E, A, cx)
        G = (A + B + C) / 3
        segs1 = segs + _segs(I, D) + _segs(I, E) + _segs(I, F)
        tl = []
        for p, q, t_, col in ((A, F, "x", cx), (E, A, "x", cx), (F, B, "y", cy),
                              (B, D, "y", cy), (D, C, "z", cz), (C, E, "z", cz)):
            m = tag(t_, 30, col)
            _park_beside(m, [(p, q)], segs1, avoid=[lA, lB, lC] + tl,
                         prefer=(p + q) / 2 - G, only=True)
            tl.append(m)
        self.play(*[FadeOut(h) for h, _, _ in halves], FadeOut(bis),
                  *[Create(m) for m in (AF, FB, BD, DC, CE, EA)],
                  *[FadeIn(m) for m in tl], run_time=1.0)
        self.hold(0.4)

        # ---- lay the perimeter out in two rows: x y z and x y z
        X0, y1, y2 = -4.1, -1.05, -2.1
        R = lambda t, yy: np.array([X0 + t, yy, 0.0])
        moves = [(VGroup(AF.copy(), FB.copy()), [A, B], [R(0, y1), R(c, y1)]),
                 (CE.copy(), [C, E], [R(c, y1), R(c + z, y1)]),
                 (EA.copy(), [A, E], [R(0, y2), R(x, y2)]),
                 (VGroup(BD.copy(), DC.copy()), [B, C], [R(x, y2), R(x + a, y2)])]
        for mob, src, dst in moves:
            self.add(mob)
        self.play(LaggedStart(*[_rigid(mob, src, dst, run_time=1.6)
                                for mob, src, dst in moves], lag_ratio=0.15))
        row1 = [(R(0, y1), R(x, y1)), (R(x, y1), R(c, y1)), (R(c, y1), R(s, y1))]
        row2 = [(R(0, y2), R(x, y2)), (R(x, y2), R(x + y, y2)), (R(x + y, y2), R(s, y2))]
        got1 = [(m.get_start(), m.get_end()) for m in (moves[0][0][0], moves[0][0][1])]
        check(close(got1[0][0], row1[0][0], 1e-6) and close(got1[0][1], row1[0][1], 1e-6)
              and close(got1[1][1], row1[1][1], 1e-6), "AB lies as x then y")
        e1 = moves[1][0]
        check(close(e1.get_start(), row1[2][0], 1e-6) and close(e1.get_end(), row1[2][1], 1e-6),
              "CE lies as z after them")
        e2 = moves[2][0]
        check(close(e2.get_end(), row2[0][0], 1e-6) and close(e2.get_start(), row2[0][1], 1e-6),
              "EA lies as x")
        bc = moves[3][0]
        check(close(bc[0].get_start(), row2[1][0], 1e-6) and close(bc[0].get_end(), row2[1][1], 1e-6)
              and close(bc[1].get_end(), row2[2][1], 1e-6), "BC lies as y then z")
        check(abs(c + z - s) < 1e-9 and abs(x + a - s) < 1e-9, "both rows have length s")

        # ---- the two rows are the whole perimeter, 2s: each row is s
        endl = DashedLine(R(s, y1 + 0.2), R(s, y2 - 0.2), color=GREY_B, stroke_width=2,
                          dash_length=0.07)
        brace = VGroup(Line(R(s + 0.35, y1), R(s + 0.35, y2), color=YELLOW_B, stroke_width=3),
                       Line(R(s + 0.24, y1), R(s + 0.35, y1), color=YELLOW_B, stroke_width=3),
                       Line(R(s + 0.24, y2), R(s + 0.35, y2), color=YELLOW_B, stroke_width=3))
        l2s = tag("a + b + c  =  2s", 28, YELLOW_B)
        l2s.next_to(brace, RIGHT, buff=0.15)
        self.play(Create(endl), Create(brace), FadeIn(l2s), run_time=0.9)
        bar_s, ls = _bracket(R(0, y1), R(s, y1), UP, "s", gap=0.25, size=30)
        self.play(FadeIn(bar_s), FadeIn(ls), run_time=0.7)

        # ---- the second row: x, then the whole side a
        bar_a, la = _bracket(R(x, y2), R(s, y2), UP, "a", gap=0.25, size=30)
        bar_x, lx = _bracket(R(0, y2), R(x, y2), UP, "s − a", gap=0.25, size=28)
        self.play(Indicate(VGroup(BD, DC), color=WHITE, scale_factor=1.0),
                  Indicate(bc, color=WHITE, scale_factor=1.0), run_time=0.8)
        self.play(FadeIn(bar_a), FadeIn(la), run_time=0.6)
        self.play(FadeIn(bar_x), FadeIn(lx), run_time=0.7)
        self.hold(0.3)

        rsegs = (row1 + row2 + _bar_segs(bar_s) + _bar_segs(bar_a) + _bar_segs(bar_x)
                 + _bar_segs(brace) + _segs(R(s, y1 + 0.2), R(s, y2 - 0.2)))
        _labels_ok([lA, lB, lC] + tl, segs1 + _ra_segs(D, I, C, 0.14)
                   + _ra_segs(E, I, A, 0.14) + _ra_segs(F, I, B, 0.14), "F59 triangle")
        _labels_ok([lA, lB, lC] + tl + [ls, la, lx, l2s], rsegs, "F59 rows")
        cap = caption("x = s − a,     y = s − b,     z = s − c", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F55 MEDIAN LENGTH

class F55_MedianLength(Board):
    """AM is the median to BC (M its midpoint). A half-turn about M carries
    the triangle ABC onto A'CB: ABA'C is a parallelogram with sides c, b,
    c, b and diagonals AA' = 2m and BC = a. Drop the heights h from C and A'
    to the line AB; with e the offset of C, the diagonals and the side b are
    hypotenuses of right triangles with legs c + e, h; c − e, h; e, h.
    So (2m)² + a² = (c + e)² + (c − e)² + 2h². In the square of side c + e
    the tilted square on the cut points has side² = c² + e²; folding the
    four corner triangles in over its sides leaves a hole of side c − e, so
    (c + e)² − (c² + e²) = (c² + e²) − (c − e)²: (c + e)² + (c − e)² =
    2c² + 2e². Hence (2m)² + a² = 2c² + 2(e² + h²) = 2b² + 2c², the
    parallelogram law, and m² = (2b² + 2c² − a²)/4."""

    def construct(self):
        cm, em, hm = 4.0, 1.3, 2.4
        Am, Bm, Cm = np.array([0.0, 0.0]), np.array([cm, 0.0]), np.array([em, hm])
        Fr = Frame(-0.45, cm + em + 0.45, -1.55, hm + 0.42, max_w=6.95,
                   centre=(-3.1, 0.98))
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        Ap = B + C - A
        M = (B + C) / 2
        H1, H2 = _foot(C, A, B), _foot(Ap, A, B)
        n = np.linalg.norm
        a, b, c = n(C - B), n(C - A), n(B - A)
        m, e, h = n(M - A), n(H1 - A), n(C - H1)

        # ---- the claims
        check(close(_rot2(A, M, PI), Ap) and close(_rot2(B, M, PI), C),
              "the half-turn about M: A -> A', B <-> C")
        check(abs(n(Ap - A) - 2 * m) < 1e-9, "AA' = 2m")
        check(abs(n(Ap - B) - b) < 1e-9 and abs(n(Ap - C) - c) < 1e-9, "sides c, b, c, b")
        check(0 < np.dot(H1 - A, B - A) < c * c and np.dot(H2 - B, B - A) > 0,
              "C's foot on AB, A''s foot beyond B")
        check(abs(n(H2 - A) - (c + e)) < 1e-9 and abs(n(B - H1) - (c - e)) < 1e-9
              and abs(n(Ap - H2) - h) < 1e-9, "legs c + e, c − e, e and h")
        check(abs((2 * m) ** 2 - ((c + e) ** 2 + h * h)) < 1e-9, "(2m)² = (c + e)² + h²")
        check(abs(a * a - ((c - e) ** 2 + h * h)) < 1e-9, "a² = (c − e)² + h²")
        check(abs(b * b - (e * e + h * h)) < 1e-9, "b² = e² + h²")
        check(abs((2 * m) ** 2 + a * a - 2 * b * b - 2 * c * c) < 1e-9, "parallelogram law")
        check(abs(m * m - (2 * b * b + 2 * c * c - a * a) / 4) < 1e-9, "the median formula")

        c2m, ca, cb, cc = YELLOW_B, ORANGE, TEAL_B, BLUE_B
        # ---- everything the labels must clear, over the whole scene
        dn = -_perp(_unit(B - A))
        if dn[1] > 0:
            dn = -dn
        bar_e, le = _bracket(A, H1, dn, "e", gap=0.62, size=26, color=GREY_A)
        bar_ce, lce = _bracket(H1, B, dn, "c − e", gap=0.62, size=26, color=ca)
        bar_cpe, lcpe = _bracket(A, H2, dn, "c + e", gap=1.25, size=26, color=c2m)
        segsP = (_segs(A, B, Ap, C, closed=True) + _segs(A, Ap) + _segs(B, C)
                 + _tick_segs(B, M) + _tick_segs(M, C))
        segsH = (segsP + _segs(B, H2 + 0.35 * _unit(B - A)) + _segs(C, H1) + _segs(Ap, H2)
                 + _ra_segs(H1, C, B, 0.16) + _ra_segs(H2, Ap, A, 0.16)
                 + _bar_segs(bar_e) + _bar_segs(bar_ce) + _bar_segs(bar_cpe))

        # ---- the triangle and its median
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        med = Line(A, M, color=c2m, stroke_width=5)
        tk = VGroup(_ticks(B, M, 1, at=0.5), _ticks(M, C, 1, at=0.5))
        base_segs = segsH
        lA = _park(tag("A", 30), A, [B, C, Ap, A + dn], base_segs, d0=0.3)
        lB = _park(tag("B", 30), B, [A, C, H2, Ap, B + dn], base_segs, d0=0.3, avoid=[lA])
        lC = _park(tag("C", 30), C, [A, B, Ap, H1], base_segs, d0=0.3, avoid=[lA, lB])
        lM = _park(tag("M", 26), M, [A, B, C, Ap], base_segs, d0=0.24, avoid=[lA, lB, lC])
        lc = _park_beside(tag("c", 30, cc), [(A, B)], base_segs, prefer=UP, only=True,
                          ats=(0.62, 0.7, 0.55, 0.75, 0.8), avoid=[lM])
        lb = _park_beside(tag("b", 30, cb), [(A, C)], base_segs, prefer=LEFT, only=True)
        la = _park_beside(tag("a", 30, ca), [(B, C)], base_segs, prefer=UP + RIGHT, only=True,
                          ats=(0.3, 0.25, 0.35, 0.2), avoid=[lB, lC, lM])
        lm = _park_beside(tag("m", 30, c2m), [(A, M)], base_segs, prefer=DOWN + RIGHT,
                          only=True, ats=(0.55, 0.6, 0.5), avoid=[lM, lc])
        self.play(Create(tri), *[FadeIn(t) for t in (lA, lB, lC, lc, lb, la)], run_time=1.1)
        self.play(Create(med), FadeIn(tk), FadeIn(lM), FadeIn(lm), run_time=0.8)
        self.hold(0.3)

        # ---- double the median: a half-turn about M
        half = VGroup(mk([A, B, C], GREY_B, 0.25, stroke_width=0),
                      Line(A, B, color=cc, stroke_width=4), Line(A, C, color=cb, stroke_width=4))
        self.add(half)
        dM = Dot(M, radius=0.06)
        self.play(FadeIn(dM), run_time=0.3)
        self.play(Rotate(half, angle=PI, about_point=M), run_time=1.8)
        check(close(half[1].get_start(), Ap, 1e-6) and close(half[1].get_end(), C, 1e-6)
              and close(half[2].get_end(), B, 1e-6), "the copy lands as A'CB")
        para = Polygon(A, B, Ap, C, stroke_color=WHITE, stroke_width=4)
        lAp = _park(tag("A'", 30), Ap, [A, B, C, H2], segsH, d0=0.3, avoid=[lA, lB, lC])
        lc2 = _park_beside(tag("c", 30, cc), [(C, Ap)], segsH, prefer=UP, only=True,
                           ats=(0.38, 0.45, 0.3), avoid=[lC, lAp])
        lb2 = _park_beside(tag("b", 30, cb), [(B, Ap)], segsH, prefer=RIGHT, only=True,
                           avoid=[lB, lAp])
        diag = Line(M, Ap, color=c2m, stroke_width=5)
        l2m = tag("2m", 30, c2m)
        self.play(Create(diag), FadeIn(lAp), FadeIn(lc2), FadeIn(lb2), FadeOut(half),
                  Create(para), run_time=0.9)
        _park_beside(l2m, [(M, Ap)], segsH, prefer=DOWN + RIGHT, only=True,
                     ats=(0.5, 0.42, 0.58, 0.66, 0.34), avoid=[lAp, lb2, lM, la])
        self.play(FadeOut(lm), FadeIn(l2m), run_time=0.6)
        self.hold(0.3)

        # ---- heights h from C and A' to the line AB; the offsets e, c ± e
        base = DashedLine(B, H2 + 0.35 * _unit(B - A), color=GREY_B, stroke_width=2.5,
                          dash_length=0.08)
        hC = DashedLine(C, H1, color=WHITE, stroke_width=3, dash_length=0.08)
        hA = DashedLine(Ap, H2, color=WHITE, stroke_width=3, dash_length=0.08)
        r1, r2 = _ra(H1, C, B, 0.16), _ra(H2, Ap, A, 0.16)
        self.play(Create(base), Create(hC), Create(hA), Create(r1), Create(r2), run_time=0.9)
        lh1 = _park_beside(tag("h", 28), [(C, H1)], segsH, prefer=LEFT, only=True,
                           ats=(0.55, 0.62, 0.45), avoid=[lb, lc])
        lh2 = _park_beside(tag("h", 28), [(Ap, H2)], segsH, prefer=RIGHT, only=True,
                           ats=(0.5, 0.6, 0.4), avoid=[lb2])
        self.play(FadeIn(lh1), FadeIn(lh2), FadeIn(bar_e), FadeIn(le), run_time=0.7)

        # ---- the ledger: three right triangles
        x0 = 0.8

        def row(parts, y, size=24):
            g = _scriptline([(t_ if i == 0 else " " + t_, 0, col)
                             for i, (t_, col) in enumerate(parts)], size)
            g.move_to(np.array([0.0, y, 0.0])).align_to(np.array([x0, 0.0, 0.0]), LEFT)
            return g

        rows = [row([("(2m)²", c2m), ("=", WHITE), ("(c + e)² + h²", WHITE)], 3.35),
                row([("a²", ca), ("=", WHITE), ("(c − e)² + h²", WHITE)], 2.8),
                row([("b²", cb), ("=", WHITE), ("e² + h²", WHITE)], 2.25)]
        tri_big = mk([A, H2, Ap], c2m, 0.5, stroke_width=0)
        tri_a = mk([H1, B, C], ca, 0.5, stroke_width=0)
        tri_b = mk([A, H1, C], cb, 0.5, stroke_width=0)
        for t_, extra, rw in ((tri_big, [bar_cpe, lcpe], rows[0]),
                              (tri_a, [bar_ce, lce], rows[1]),
                              (tri_b, [], rows[2])):
            self.add(t_)
            self.bring_to_back(t_)
            self.play(FadeIn(t_), *[FadeIn(x_) for x_ in extra], run_time=0.6)
            self.play(FadeIn(rw, shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeOut(t_), run_time=0.35)

        # ---- (c + e)² + (c − e)² = 2(c² + e²): fold four corners into a square
        side = 2.8
        q = side / (Fr.k * (cm + em))
        S0 = np.array([SAFE_X - 0.1 - side, -2.92, 0.0])
        Sq = lambda p: S0 + q * Fr.k * to3(p)
        Lm = cm + em
        P1, P2, P3, P4 = (np.array([cm, 0.0]), np.array([Lm, cm]), np.array([em, Lm]),
                          np.array([0.0, em]))
        Cn = [np.array([0.0, 0.0]), np.array([Lm, 0.0]), np.array([Lm, Lm]), np.array([0.0, Lm])]
        corners = [[Cn[0], P1, P4], [Cn[1], P2, P1], [Cn[2], P3, P2], [Cn[3], P4, P3]]
        folded = [[_mirror(to3(t_[0]), to3(t_[1]), to3(t_[2]))[:2], t_[1], t_[2]]
                  for t_ in corners]
        hole = [f[0] for f in folded]
        tilt = [P1, P2, P3, P4]
        check(_tiles_exactly([[Cn[0], Cn[1], Cn[2], Cn[3]][0:4]], Cn), "the big square")
        check(_tiles_exactly(corners + [tilt], Cn), "big square = tilted square + 4 corners")
        check(_tiles_exactly(folded + [hole], tilt), "tilted square = 4 folded corners + hole")
        check(all(abs(np.linalg.norm(hole[i] - hole[(i + 1) % 4]) - (cm - em)) < 1e-9
                  for i in range(4)), "the hole is a square of side c − e")
        check(abs(area(tilt) - (cm * cm + em * em)) < 1e-9, "tilted square = c² + e²")
        sq = Polygon(*[Sq(p) for p in Cn], stroke_color=WHITE, stroke_width=3)
        tl = Polygon(*[Sq(p) for p in tilt], stroke_color=YELLOW_B, stroke_width=3)
        cps = [mk([Sq(p) for p in t_], BLUE_D, 0.75, stroke_width=1.5) for t_ in corners]
        sq_segs = _segs(*[Sq(p) for p in Cn], closed=True) + _segs(*[Sq(p) for p in tilt],
                                                                   closed=True)
        les = _beside(tag("e", 24, GREY_A), Sq(Cn[0]), Sq(P4), LEFT, 0.1)
        lcs = _beside(tag("c", 24, cc), Sq(P4), Sq(Cn[3]), LEFT, 0.1)
        lsq = tag("(c + e)²", 24).move_to(Sq((Cn[0] + Cn[3]) / 2) + 1.05 * LEFT)
        self.play(Create(sq), FadeIn(lcs), FadeIn(les), FadeIn(lsq), run_time=0.8)
        self.play(Create(tl), *[FadeIn(p_) for p_ in cps], run_time=0.8)
        ltl = tag("c² + e²", 22, YELLOW_B).move_to(Sq(sum(tilt) / 4))
        self.play(FadeIn(ltl), run_time=0.5)
        self.play(FadeOut(ltl), run_time=0.3)
        self.play(*[_fold(p_, Sq(t_[1]), Sq(t_[2])) for p_, t_ in zip(cps, corners)],
                  run_time=1.6)
        for p_, f in zip(cps, folded):
            _landed(p_, [Sq(x_) for x_ in f], "a corner folds into the tilted square")
        holeM = Polygon(*[Sq(p) for p in hole], stroke_color=RED_B, stroke_width=3)
        lhole = tag("(c − e)²", 20, RED_B).move_to(Sq(sum(hole) / 4))
        self.play(Create(holeM), FadeIn(lhole), run_time=0.7)
        rows.append(row([("(c + e)² + (c − e)²", WHITE), ("=", WHITE), ("2c² + 2e²", WHITE)],
                        1.55))
        self.play(FadeIn(rows[3], shift=RIGHT * 0.2), run_time=0.7)

        # ---- add: the parallelogram law, and the median
        rows.append(row([("(2m)² + a²", WHITE), ("=", WHITE), ("2c² + 2(e² + h²)", WHITE)],
                        0.9))
        rows.append(row([("=", WHITE), ("2c² + 2b²", WHITE)], 0.35))
        rows[5].shift(RIGHT * (rows[4][1].get_left()[0] - rows[5][0].get_left()[0]))
        self.play(FadeIn(rows[4], shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(rows[5], shift=RIGHT * 0.2), run_time=0.6)
        self.hold(0.4)

        figlabels = [lA, lB, lC, lM, lAp, lc, lb, la, lc2, lb2, l2m, lh1, lh2, le, lce, lcpe]
        _labels_ok(figlabels, segsH + _segs(A, M) + _circle_segs(M, 0.06, 16), "F55 figure")
        _labels_ok([lcs, les, lsq], sq_segs, "F55 square")
        check(_pip(ltl.get_corner(UL), [Sq(p) for p in hole])
              and _pip(ltl.get_corner(DR), [Sq(p) for p in hole]),
              "c² + e² is written inside the tilted square, clear of the corners")
        check(_pip(lhole.get_corner(UL), [Sq(p) for p in hole])
              and _pip(lhole.get_corner(DR), [Sq(p) for p in hole]), "(c − e)² inside the hole")
        _labels_ok(rows + [lsq, lcs, les, lhole], [], "F55 ledger")
        check(rows[5].get_bottom()[1] > sq.get_top()[1] + 0.2, "the square sits below the rows")
        check(min(r_.get_left()[0] for r_ in rows) > max(Ap[0], H2[0], lb2.get_right()[0],
                                                         lh2.get_right()[0]) + 0.2,
              "the ledger sits right of the figure")
        cap = caption("m²  =  (2b² + 2c² − a²) / 4", 36)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F34 PITOT

class F34_Pitot(Board):
    """ABCD has an incircle (centre O) touching AB, BC, CD, DA. The two
    tangents from a vertex are equal: the radii to the touch points are
    perpendicular to the sides, and folding over the line from the vertex
    to O lands one right triangle on the other. Call the four tangent
    lengths w, x, y, z (from A, B, C, D). Then a = AB = w + x, c = CD =
    y + z, b = BC = x + y, d = DA = z + w: laid end to end, each pair of
    opposite sides uses each of the four lengths once, so a + c = b + d."""

    def construct(self):
        th = [150.0, 250.0, 355.0, 420.0]          # touch points on DA, AB, BC, CD
        k, r = 1.4, 1.0
        O = np.array([-3.55, 0.55, 0.0])
        T = [O + k * r * _polar(t * DEGREES) for t in th]

        def vert(i):                               # between touch points i and i+1
            t0, t1 = th[i], th[(i + 1) % 4] + (360.0 if i == 3 else 0.0)
            return O + k * r / np.cos((t1 - t0) / 2 * DEGREES) * _polar((t0 + t1) / 2 * DEGREES)

        A, B, C, D = vert(0), vert(1), vert(2), vert(3)
        Tda, Tab, Tbc, Tcd = T
        n = np.linalg.norm
        w, x, y, z = n(A - Tab), n(B - Tbc), n(C - Tcd), n(D - Tda)
        a, b, c, d = n(B - A), n(C - B), n(D - C), n(A - D)

        # ---- the claims
        for P, Q, U in ((Tda, D, A), (Tab, A, B), (Tbc, B, C), (Tcd, C, D)):
            check(_on_line(P, Q, U) and 0 < np.dot(P - Q, U - Q) < n(U - Q) ** 2,
                  "each touch point lies on its side")
            check(abs(np.dot(P - O, U - Q)) < 1e-9, "radius ⟂ side")
        check(abs(n(A - Tda) - w) < 1e-9 and abs(n(B - Tab) - x) < 1e-9
              and abs(n(C - Tbc) - y) < 1e-9 and abs(n(D - Tcd) - z) < 1e-9,
              "equal tangents from each vertex")
        check(close(_mirror(Tab, A, O), Tda) and close(_mirror(Tbc, B, O), Tab)
              and close(_mirror(Tcd, C, O), Tbc) and close(_mirror(Tda, D, O), Tcd),
              "folding over the line to O swaps the two touch points")
        check(abs(a - (w + x)) < 1e-9 and abs(b - (x + y)) < 1e-9
              and abs(c - (y + z)) < 1e-9 and abs(d - (z + w)) < 1e-9, "the sides")
        check(abs((a + c) - (b + d)) < 1e-9, "a + c = b + d")
        check(min(w, x, y, z) > 0.8 and max(abs(a - c), abs(b - d)) > 0.3,
              "an irregular tangential quadrilateral")

        cw, cx, cy, cz = BLUE_B, TEAL_B, ORANGE, PURPLE_B
        quad = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=3.5)
        circ = Circle(radius=k * r, color=GREY_A, stroke_width=3).move_to(O)
        dO = Dot(O, radius=0.06)
        segs = _segs(A, B, C, D, closed=True) + _circle_segs(O, k * r)
        Gq = (A + B + C + D) / 4
        names = []
        for s_, V in (("A", A), ("B", B), ("C", C), ("D", D)):
            names.append(_park(tag(s_, 30), V, [A, B, C, D], segs, d0=0.3, avoid=names))
        self.play(Create(quad), *[FadeIn(m) for m in names], run_time=1.0)
        self.play(Create(circ), FadeIn(dO), run_time=0.8)

        rads = VGroup(*[Line(O, P, color=GREY_A, stroke_width=2.5) for P in T])
        ras = VGroup(_ra(Tda, O, D, 0.14), _ra(Tab, O, A, 0.14), _ra(Tbc, O, B, 0.14),
                     _ra(Tcd, O, C, 0.14))
        spokes = VGroup(*[DashedLine(V, O, color=GREY_B, stroke_width=2, dash_length=0.08)
                          for V in (A, B, C, D)])
        self.play(Create(rads), Create(ras), Create(spokes), run_time=1.0)

        halves = [(mk([A, Tab, O], cw, 0.6, stroke_width=1.5), A, [A, Tda, O]),
                  (mk([B, Tbc, O], cx, 0.6, stroke_width=1.5), B, [B, Tab, O]),
                  (mk([C, Tcd, O], cy, 0.6, stroke_width=1.5), C, [C, Tbc, O]),
                  (mk([D, Tda, O], cz, 0.6, stroke_width=1.5), D, [D, Tcd, O])]
        self.play(*[FadeIn(h) for h, _, _ in halves], run_time=0.5)
        self.play(*[_fold(h, V, O) for h, V, _ in halves], run_time=1.6)
        for h, V, dst in halves:
            _landed(h, dst, "a folded half lands on the other half")

        def piece(p, q, col):
            return Line(p, q, color=col, stroke_width=9)

        pc = {"Aab": piece(A, Tab, cw), "Bab": piece(Tab, B, cx),
              "Bbc": piece(B, Tbc, cx), "Cbc": piece(Tbc, C, cy),
              "Ccd": piece(C, Tcd, cy), "Dcd": piece(Tcd, D, cz),
              "Dda": piece(D, Tda, cz), "Ada": piece(Tda, A, cw)}
        segs1 = segs + [(O, P) for P in T]
        tl = []
        for (p, q), t_, col in (((A, Tab), "w", cw), ((Tda, A), "w", cw),
                                ((Tab, B), "x", cx), ((B, Tbc), "x", cx),
                                ((Tbc, C), "y", cy), ((C, Tcd), "y", cy),
                                ((Tcd, D), "z", cz), ((D, Tda), "z", cz)):
            m = tag(t_, 28, col)
            _park_beside(m, [(p, q)], segs1, avoid=names + tl, prefer=(p + q) / 2 - Gq,
                         only=True)
            tl.append(m)
        self.play(*[FadeOut(h) for h, _, _ in halves], FadeOut(spokes),
                  *[Create(m) for m in pc.values()], *[FadeIn(m) for m in tl], run_time=1.0)
        self.hold(0.4)

        # ---- opposite sides end to end: a + c and b + d
        X0, y1, y2 = 0.45, 1.25, -0.75
        R = lambda t, yy: np.array([X0 + t, yy, 0.0])
        moves = [(VGroup(pc["Aab"].copy(), pc["Bab"].copy()), [A, B], [R(0, y1), R(a, y1)]),
                 (VGroup(pc["Ccd"].copy(), pc["Dcd"].copy()), [C, D], [R(a, y1), R(a + c, y1)]),
                 (VGroup(pc["Bbc"].copy(), pc["Cbc"].copy()), [B, C], [R(0, y2), R(b, y2)]),
                 (VGroup(pc["Dda"].copy(), pc["Ada"].copy()), [D, A], [R(b, y2), R(b + d, y2)])]
        for mob, src, dst in moves:
            self.add(mob)
        self.play(LaggedStart(*[_rigid(mob, src, dst, run_time=1.6)
                                for mob, src, dst in moves], lag_ratio=0.15))
        cuts1 = [0, w, a, a + y, a + c]                 # w x | y z
        cuts2 = [0, x, b, b + z, b + d]                 # x y | z w
        for (mob, src, dst), cuts, yy in ((moves[0], cuts1[0:3], y1), (moves[1], cuts1[2:5], y1),
                                          (moves[2], cuts2[0:3], y2), (moves[3], cuts2[2:5], y2)):
            check(close(mob[0].get_start(), R(cuts[0], yy), 1e-6)
                  and close(mob[0].get_end(), R(cuts[1], yy), 1e-6)
                  and close(mob[1].get_end(), R(cuts[2], yy), 1e-6),
                  "a side lies flat, its two tangent pieces in place")

        # letters over the pieces, brackets for the sides
        lets = VGroup()
        for cuts, yy, sgn, cols, nm in ((cuts1, y1, 1, (cw, cx, cy, cz), "wxyz"),
                                        (cuts2, y2, -1, (cx, cy, cz, cw), "xyzw")):
            for i in range(4):
                lets.add(tag(nm[i], 26, cols[i]).move_to(
                    R((cuts[i] + cuts[i + 1]) / 2, yy + sgn * 0.3)))
        bars = []
        for (p0, p1, yy, sgn, t_) in ((0, a, y1, 1, "a"), (a, a + c, y1, 1, "c"),
                                      (0, b, y2, -1, "b"), (b, b + d, y2, -1, "d")):
            bars.append(_bracket(R(p0, yy), R(p1, yy), sgn * UP, t_, gap=0.62, size=30))
        self.play(FadeIn(lets), run_time=0.6)
        self.play(*[FadeIn(bb) for bb, _ in bars], *[FadeIn(lb) for _, lb in bars],
                  run_time=0.8)
        endl = DashedLine(R(a + c, y1 + 0.2), R(b + d, y2 - 0.2), color=YELLOW_B,
                          stroke_width=2.5, dash_length=0.08)
        self.play(Create(endl), run_time=0.6)
        self.hold(0.4)

        rsegs = ([(R(0, y1), R(a + c, y1)), (R(0, y2), R(b + d, y2)),
                  (R(a + c, y1 + 0.2), R(b + d, y2 - 0.2))]
                 + [sg for bb, _ in bars for sg in _bar_segs(bb)])
        _labels_ok(names + tl, segs1 + _ra_segs(Tda, O, D, 0.14) + _ra_segs(Tab, O, A, 0.14)
                   + _ra_segs(Tbc, O, B, 0.14) + _ra_segs(Tcd, O, C, 0.14), "F34 figure")
        _labels_ok(list(lets) + [lb for _, lb in bars], rsegs, "F34 rows")
        check(min(m.get_left()[0] for m in lets) > max(m.get_right()[0] for m in names + tl)
              + 0.2, "the rows sit right of the figure")
        cap = caption("a + c  =  b + d", 38)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F42 OCTAGON

class F42_RegularOctagonArea(Board):
    """Extend four alternate sides of the regular octagon of side s: they
    make a square, and the four corners cut off are right isosceles
    triangles with hypotenuse s (the octagon's angles are 135°), so with
    legs s/√2 — each is half of a square of side s/√2. The square's side is
    s/√2 + s + s/√2 = (1 + √2)s. The four corners, each tipped over by 45°,
    fit together with their right angles at one point into a square of side
    s. So A = (1 + √2)²s² − s² = 2(1 + √2)s²."""

    def construct(self):
        W = 5.0
        s = W / (1 + np.sqrt(2))
        g = s / np.sqrt(2)
        C0 = np.array([-3.05, 0.3, 0.0])
        h, H = s / 2, W / 2
        V = [C0 + np.array(p + (0.0,)) for p in
             ((h, -H), (H, -h), (H, h), (h, H), (-h, H), (-H, h), (-H, -h), (-h, -H))]
        K = {"BR": C0 + np.array([H, -H, 0]), "TR": C0 + np.array([H, H, 0]),
             "TL": C0 + np.array([-H, H, 0]), "BL": C0 + np.array([-H, -H, 0])}
        # corners: right-angle vertex, then the two octagon vertices (counter-clockwise)
        cor = {"BR": [K["BR"], V[1], V[0]], "TR": [K["TR"], V[3], V[2]],
               "TL": [K["TL"], V[5], V[4]], "BL": [K["BL"], V[7], V[6]]}
        big = [K["BL"], K["BR"], K["TR"], K["TL"]]
        n = np.linalg.norm
        Z = np.array([4.05, 0.3, 0.0])                 # centre of the s × s square
        # each corner tips over by +45° into one quarter of the s × s square
        slot = {"TR": "B", "TL": "R", "BL": "T", "BR": "L"}
        sq = [Z + np.array([-h, -h, 0]), Z + np.array([h, -h, 0]),
              Z + np.array([h, h, 0]), Z + np.array([-h, h, 0])]
        quarter = {"B": [Z, sq[0], sq[1]], "R": [Z, sq[1], sq[2]],
                   "T": [Z, sq[2], sq[3]], "L": [Z, sq[3], sq[0]]}

        def moved(key):
            P = cor[key]
            m0 = sum(P) / 3
            dst_c = sum(quarter[slot[key]]) / 3
            return [dst_c + _rot2(p, m0, PI / 4) - m0 for p in P]

        # ---- the claims: every piece
        for i in range(8):
            check(abs(n(V[(i + 1) % 8] - V[i]) - s) < 1e-9, "all eight sides are s")
            check(abs(_angle(V[i], V[i - 1], V[(i + 1) % 8]) - 3 * PI / 4) < 1e-9,
                  "all eight angles are 135°")
        for key, P in cor.items():
            check(abs(_angle(P[0], P[1], P[2]) - PI / 2) < 1e-9, f"corner {key}: right angle")
            check(abs(n(P[1] - P[0]) - g) < 1e-9 and abs(n(P[2] - P[0]) - g) < 1e-9,
                  f"corner {key}: legs s/√2")
            check(abs(n(P[2] - P[1]) - s) < 1e-9, f"corner {key}: hypotenuse s")
            check(abs(abs(area(P)) - g * g / 2) < 1e-9, f"corner {key}: half a square of side s/√2")
            Q = moved(key)
            check(_same_poly(Q, quarter[slot[key]], 1e-9),
                  f"corner {key}, tipped by 45°, fills its quarter of the s × s square")
            check(close(Q[0], Z), f"corner {key}: its right angle lands at the centre")
        check(_tiles_exactly([V] + list(cor.values()), big),
              "octagon + four corners tile the square of side (1 + √2)s")
        check(_tiles_exactly([moved(kk) for kk in cor], sq), "the four corners tile the s × s square")
        check(abs(W - (g + s + g)) < 1e-9 and abs(W - (1 + np.sqrt(2)) * s) < 1e-9,
              "W = s/√2 + s + s/√2 = (1 + √2)s")
        check(abs(abs(area(V)) - (W * W - s * s)) < 1e-9
              and abs(abs(area(V)) - 2 * (1 + np.sqrt(2)) * s * s) < 1e-9,
              "A = W² − s² = 2(1 + √2)s²")

        cO, cK = BLUE_D, ORANGE
        octo = mk(V, cO, 0.85, stroke_width=3)
        tks = VGroup(*[_ticks(V[i], V[(i + 1) % 8], 1, size=0.11) for i in range(8)])
        tk_segs = [sg for i in range(8) for sg in _tick_segs(V[i], V[(i + 1) % 8], 1, 0.11)]
        ls = tag("s", 30).move_to((V[2] + V[3]) / 2 + 0.36 * _unit(C0 - (V[2] + V[3]) / 2))
        self.play(FadeIn(octo), FadeIn(tks), run_time=1.0)
        self.play(FadeIn(ls), run_time=0.5)

        # ---- extend four alternate sides: a square, four corners cut off
        ext = VGroup(*[DashedLine(a_, b_, color=GREY_A, stroke_width=2.5, dash_length=0.09)
                       for a_, b_ in ((V[0], K["BR"]), (V[1], K["BR"]), (V[2], K["TR"]),
                                      (V[3], K["TR"]), (V[4], K["TL"]), (V[5], K["TL"]),
                                      (V[6], K["BL"]), (V[7], K["BL"]))])
        self.play(Create(ext), run_time=1.0)
        pcs = {kk: mk(P, cK, 0.85, stroke_width=2) for kk, P in cor.items()}
        frame = Polygon(*big, stroke_color=WHITE, stroke_width=3)
        ras = VGroup(*[_ra(P[0], P[1], P[2], 0.18) for P in cor.values()])
        self.add(*pcs.values())
        self.bring_to_back(*pcs.values())
        self.play(*[FadeIn(p_) for p_ in pcs.values()], FadeOut(ext), Create(frame),
                  Create(ras), run_time=0.9)
        self.bring_to_front(octo, tks, ls)

        # ---- the side of the square: s/√2 + s + s/√2
        top = [K["TL"], V[4], V[3], K["TR"]]
        cuts = VGroup(*[Line(p_ + 0.12 * UP, p_ - 0.12 * UP, color=YELLOW_B, stroke_width=3)
                        for p_ in (V[4], V[3])])
        lg1 = tag("s/√2", 26, YELLOW_B).move_to((top[0] + top[1]) / 2 + 0.34 * UP)
        lmid = tag("s", 26, YELLOW_B).move_to((top[1] + top[2]) / 2 + 0.34 * UP)
        lg2 = tag("s/√2", 26, YELLOW_B).move_to((top[2] + top[3]) / 2 + 0.34 * UP)
        bar, lW = _bracket(K["BL"], K["BR"], DOWN, "(1 + √2)s", gap=0.25, size=28)
        self.play(Create(cuts), FadeIn(lg1), FadeIn(lmid), FadeIn(lg2), run_time=0.8)
        self.play(FadeIn(bar), FadeIn(lW), run_time=0.7)
        self.hold(0.4)

        # ---- the four corners tip over and meet in a square of side s
        ghost = VGroup(*[_dashed_poly(P, GREY_B, 2, 12) for P in cor.values()])
        self.add(ghost)
        movers = []
        for kk in ("TR", "BR", "TL", "BL"):
            P = cor[kk]
            movers.append(_rigid(pcs[kk], P, moved(kk), turn=PI / 4, run_time=1.8))
        self.play(LaggedStart(*movers, lag_ratio=0.18))
        for kk in cor:
            _landed(pcs[kk], moved(kk), f"corner {kk} lands in the s × s square")
        rim = Polygon(*sq, stroke_color=YELLOW_B, stroke_width=4)
        lsq = tag("s", 30, YELLOW_B).move_to((sq[0] + sq[1]) / 2 + 0.34 * DOWN)
        lsq2 = tag("s", 30, YELLOW_B).move_to((sq[1] + sq[2]) / 2 + 0.34 * RIGHT)
        self.play(Create(rim), FadeIn(lsq), FadeIn(lsq2), run_time=0.8)
        self.hold(0.4)

        segs = (_segs(*V, closed=True) + _segs(*big, closed=True) + tk_segs
                + [sg for P in cor.values() for sg in _ra_segs(P[0], P[1], P[2], 0.18)]
                + _bar_segs(bar) + [(c_.get_start(), c_.get_end()) for c_ in cuts]
                + _segs(*sq, closed=True))
        _labels_ok([ls, lg1, lmid, lg2, lW, lsq, lsq2], segs, "F42 labels")
        cap = caption("A  =  (1 + √2)²s² − s²  =  2(1 + √2)s²", 34)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)


# =========================================================== F51 FAGNANO

class F51_Fagnano(Board):
    """Acute triangle ABC; D, E, F on BC, CA, AB. Reflect the triangle in
    AB and in AC: D goes to D' and D'', and FD = FD', ED = ED''. So the
    perimeter DE + EF + FD is the length of the path D' F E D'', at least
    the straight segment D'D'' (equal when E, F lie on it). AD' = AD =
    AD'' and the reflections double the angle at A: D'AD'' is isosceles
    with apex angle 2A, so D'D'' = 2·AD·sin A, least when AD is least, at
    the foot of the altitude (the circle about A through it touches BC).
    The straight path then runs through the feet of the other two
    altitudes: the triangle of the feet is the shortest. (Acute triangles
    only: in an obtuse triangle two feet of altitudes fall outside the
    sides, and no inscribed triangle is shortest.)"""

    def construct(self):
        Bd, Cd = 70.0, 56.0
        Ad = 180.0 - Bd - Cd
        am = 4.0
        cm = am * np.sin(Cd * DEGREES) / np.sin(Ad * DEGREES)
        Am = cm * np.array([np.cos(Bd * DEGREES), np.sin(Bd * DEGREES)])
        Bm, Cm = np.array([0.0, 0.0]), np.array([am, 0.0])
        Fr = Frame(-3.6, 6.05, -0.5, 4.3)
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        C1, B2 = _mirror(C, A, B), _mirror(B, A, C)
        H, Hb, Hc = _foot(A, B, C), _foot(B, C, A), _foot(C, A, B)
        n = np.linalg.norm
        uH = float(np.dot(H - B, C - B) / n(C - B) ** 2)
        u0 = 0.7
        E0, F0 = C + 0.55 * (A - C), B + 0.62 * (A - B)

        def Dp(u):
            return B + u * (C - B)

        def unfold(u):
            D = Dp(u)
            D1, D2 = _mirror(D, A, B), _mirror(D, A, C)
            return D, D1, D2, _meet(D1, D2, A, C), _meet(D1, D2, A, B)

        # ---- the claims
        angs = (_angle(A, B, C), _angle(B, C, A), _angle(C, A, B))
        check(max(angs) < PI / 2 - 0.1, "an acute triangle")
        check(abs(_angle(A, B, C) - Ad * DEGREES) < 1e-9, "the angle A")
        for u in np.linspace(0.05, 0.95, 37):
            D, D1, D2, Es, Fs = unfold(u)
            check(abs(n(D1 - A) - n(D - A)) < 1e-9 and abs(n(D2 - A) - n(D - A)) < 1e-9,
                  "AD' = AD = AD''")
            check(abs(_angle(A, D1, D2) - 2 * Ad * DEGREES) < 1e-9, "the apex angle is 2A")
            check(abs(n(D2 - D1) - 2 * n(D - A) * np.sin(Ad * DEGREES)) < 1e-9,
                  "D'D'' = 2·AD·sin A")
            check(abs(n(Es - D) + n(Fs - Es) + n(D - Fs) - n(D2 - D1)) < 1e-9,
                  "with E, F on D'D'' the perimeter is D'D''")
        D, D1, D2, Es, Fs = unfold(u0)
        check(_on_line(D1, B, C1) and _on_line(D2, B2, C), "D' on BC₁, D'' on B₂C")
        per0 = n(E0 - D) + n(F0 - E0) + n(D - F0)
        check(per0 > n(D2 - D1) + 0.25, "the first triangle is visibly longer than D'D''")
        DH, D1H, D2H, EH, FH = unfold(uH)
        check(close(DH, H) and close(EH, Hb) and close(FH, Hc),
              "at the foot of the A-altitude the straight path meets the other feet")
        Ls = [n(unfold(u)[2] - unfold(u)[1]) for u in np.linspace(0.02, 0.98, 97)]
        check(min(Ls) >= n(D2H - D1H) - 1e-9, "D'D'' is least at the foot")
        for u in np.linspace(min(u0, uH), max(u0, uH), 9):
            D, D1, D2, Es, Fs = unfold(u)
            check(0 < np.dot(Es - C, A - C) < n(A - C) ** 2
                  and 0 < np.dot(Fs - B, A - B) < n(A - B) ** 2,
                  "the straight path crosses CA and AB inside the sides")

        u = ValueTracker(u0)
        sg = ValueTracker(0.0)

        def cur():
            D, D1, D2, Es, Fs = unfold(u.get_value())
            t = sg.get_value()
            return D, D1, D2, E0 + t * (Es - E0), F0 + t * (Fs - F0)

        cT, cP, cY = WHITE, BLUE_B, YELLOW_B
        tri = Polygon(A, B, C, stroke_color=cT, stroke_width=4)
        segs0 = _segs(A, B, C, closed=True) + _segs(A, C1, B) + _segs(A, B2, C)
        lA = tag("A", 30).move_to(A + 0.36 * UP)
        lB = _park(tag("B", 30), B, [A, C, C1], segs0, d0=0.3)
        lC = _park(tag("C", 30), C, [A, B, B2], segs0, d0=0.3, avoid=[lB])
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        inner = always_redraw(lambda: Polygon(*cur()[0:1], cur()[3], cur()[4],
                                              stroke_color=cP, stroke_width=5))
        dots = always_redraw(lambda: VGroup(*[Dot(p_, radius=0.06, color=cP)
                                              for p_ in (cur()[0], cur()[3], cur()[4])]))
        D, D1, D2, Ec, Fc = cur()
        segs1 = (segs0 + _segs(D, Ec, Fc, closed=True)
                 + [sg for X in (D, Ec, Fc) for sg in _circle_segs(X, 0.06, 16)])
        lD = tag("D", 28).move_to(D + 0.36 * DOWN)
        lE = _park(tag("E", 28), Ec, [A, C, D, Fc], segs1, d0=0.26, avoid=[lA, lB, lC, lD])
        lF = _park(tag("F", 28), Fc, [A, B, D, Ec], segs1, d0=0.26,
                   avoid=[lA, lB, lC, lD, lE])
        _labels_ok([lA, lB, lC, lD, lE, lF], segs1, "F51 first labels")
        self.play(Create(inner), FadeIn(dots), FadeIn(lD), FadeIn(lE), FadeIn(lF),
                  run_time=1.0)
        self.hold(0.4)

        # ---- reflect the triangle in AB and in AC: D -> D', D''
        cpL = VGroup(mk([A, B, C], GREY_B, 0.18, stroke_color=GREY_B, stroke_width=2.5),
                     Line(D, Fc, color=cP, stroke_width=5))
        cpR = VGroup(mk([A, B, C], GREY_B, 0.18, stroke_color=GREY_B, stroke_width=2.5),
                     Line(D, Ec, color=cP, stroke_width=5))
        self.add(cpL)
        self.play(_fold(cpL, A, B), run_time=1.6)
        _landed(cpL[0], [A, B, C1], "the fold over AB lands on ABC₁")
        check(close(cpL[1].get_start(), D1, 1e-6), "D lands on D'")
        self.add(cpR)
        self.play(_fold(cpR, A, C), run_time=1.6)
        _landed(cpR[0], [A, B2, C], "the fold over AC lands on AB₂C")
        check(close(cpR[1].get_start(), D2, 1e-6), "D lands on D''")
        wings = VGroup(mk([A, B, C1], GREY_B, 0.14, stroke_color=GREY_B, stroke_width=2.5),
                       mk([A, B2, C], GREY_B, 0.14, stroke_color=GREY_B, stroke_width=2.5))
        refl = always_redraw(lambda: VGroup(
            Line(cur()[1], cur()[4], color=cP, stroke_width=5),
            Line(cur()[3], cur()[2], color=cP, stroke_width=5)))
        ticks = always_redraw(lambda: VGroup(
            _ticks(cur()[0], cur()[4], 1, cP), _ticks(cur()[1], cur()[4], 1, cP),
            _ticks(cur()[0], cur()[3], 2, cP), _ticks(cur()[2], cur()[3], 2, cP)))
        nL = -_perp(_unit(C1 - B))
        if np.dot(nL, A - B) > 0:
            nL = -nL
        nR = _perp(_unit(C - B2))
        if np.dot(nR, A - C) > 0:
            nR = -nR
        lD1 = always_redraw(lambda: tag("D'", 28).move_to(cur()[1] + 0.4 * nL))
        lD2 = always_redraw(lambda: tag("D''", 28).move_to(cur()[2] + 0.42 * nR))
        dd = always_redraw(lambda: VGroup(Dot(cur()[1], radius=0.06, color=cP),
                                          Dot(cur()[2], radius=0.06, color=cP)))
        self.remove(cpL, cpR)
        self.add(wings, refl)
        self.bring_to_back(wings)
        self.play(FadeIn(ticks), FadeIn(dd), FadeIn(lD1), FadeIn(lD2), run_time=0.7)
        self.hold(0.4)

        # ---- the path D' F E D'' against the straight segment D'D''
        chord = always_redraw(lambda: DashedLine(cur()[1], cur()[2], color=cY,
                                                 stroke_width=3, dash_length=0.1))
        self.play(Create(chord), run_time=0.8)
        self.play(FadeOut(lE), FadeOut(lF), run_time=0.3)
        self.play(sg.animate.set_value(1.0), run_time=1.8)
        self.hold(0.3)

        # ---- AD' = AD = AD'', apex angle 2A
        legs = always_redraw(lambda: VGroup(*[
            DashedLine(A, p_, color=GREY_A, stroke_width=2, dash_length=0.08)
            for p_ in (cur()[1], cur()[0], cur()[2])]))
        legt = always_redraw(lambda: VGroup(*[
            _ticks(A, p_, 3, GREY_A, size=0.1, at=0.55) for p_ in (cur()[1], cur()[0], cur()[2])]))
        arcs = always_redraw(lambda: VGroup(
            angle_arc(A, cur()[1], B, 0.62, TEAL_B, 4), angle_arc(A, B, cur()[0], 0.62, TEAL_B, 4),
            angle_arc(A, cur()[0], C, 0.76, ORANGE, 4), angle_arc(A, C, cur()[2], 0.76, ORANGE, 4)))
        self.play(Create(legs), FadeIn(legt), run_time=0.8)
        self.play(Create(arcs), run_time=0.8)
        self.hold(0.4)

        # ---- slide D to the foot of the altitude: AD, so D'D'', is least
        self.play(FadeOut(lD), run_time=0.3)
        self.play(u.animate.set_value(uH), run_time=2.6)
        D, D1, D2, Ec, Fc = cur()
        check(close(D, H, 1e-9) and close(Ec, Hb, 1e-6) and close(Fc, Hc, 1e-6),
              "D, E, F are the feet of the altitudes")
        rH = _ra(H, A, C, 0.17)
        ring = DashedVMobject(Arc(radius=n(A - H), start_angle=-PI / 2 - 0.42, angle=0.84,
                                  arc_center=A, color=GREY_A, stroke_width=2.5),
                              num_dashes=14)
        self.play(Create(rH), Create(ring), run_time=0.8)
        self.hold(0.3)

        # ---- the other two altitudes run through E and F
        alts = VGroup(DashedLine(B, Hb, color=cY, stroke_width=2.5, dash_length=0.08),
                      DashedLine(C, Hc, color=cY, stroke_width=2.5, dash_length=0.08),
                      DashedLine(A, H, color=cY, stroke_width=2.5, dash_length=0.08))
        rb, rc = _ra(Hb, B, C, 0.15), _ra(Hc, C, B, 0.15)
        for m in (inner, dots, refl, ticks, lD1, lD2, dd, chord, legs, legt, arcs):
            m.clear_updaters()
        self.play(FadeOut(legs), FadeOut(legt), FadeOut(arcs), run_time=0.5)
        orth = Polygon(H, Hb, Hc, stroke_color=cY, stroke_width=6)
        self.play(Create(alts), Create(rb), Create(rc), run_time=1.0)
        segs2 = (segs0 + _segs(H, Hb, Hc, closed=True) + _segs(B, Hb) + _segs(C, Hc)
                 + _segs(A, H) + _segs(D1, Hc) + _segs(Hb, D2) + _segs(D1, D2)
                 + _ra_segs(H, A, C, 0.17) + _ra_segs(Hb, B, C, 0.15) + _ra_segs(Hc, C, B, 0.15)
                 + _mob_segs(ring) + _mob_segs(ticks)
                 + [sg for X in (H, Hb, Hc, D1, D2) for sg in _circle_segs(X, 0.06, 16)])
        lD = _park(tag("D", 28), H, [B, C, A, Hb, Hc], segs2, d0=0.28, avoid=[lA, lB, lC])
        lE = _park(tag("E", 28), Hb, [A, C, B, H, Hc, D2], segs2, d0=0.26,
                   avoid=[lA, lB, lC, lD])
        lF = _park(tag("F", 28), Hc, [A, B, C, H, Hb, D1], segs2, d0=0.26,
                   avoid=[lA, lB, lC, lD, lE])
        self.play(Create(orth), FadeIn(lD), FadeIn(lE), FadeIn(lF), run_time=0.9)
        self.hold(0.3)

        _labels_ok([lA, lB, lC, lD, lE, lF, lD1, lD2], segs2, "F51 labels")
        cap = caption("acute triangle:  DE + EF + FD is least for the feet of the altitudes", 30)
        _final_check(self, cap)
        self.play(Write(cap))
        self.hold(2.2)
