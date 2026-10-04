# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3j.py — proofs without words, 2D (manim):
#     F52 Pompeiu's theorem                 F46 bisecting the right angle
#     F50 Brahmagupta's theorem             F60 two squares sharing a corner
#     F62 reflections of the orthocentre    J20 triples from rational points
#     J26 every triangle is isosceles?      J12 the golden ratio is irrational
#     J17 squaring numbers ending in 5      J23 primes beyond 3
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode;
# letter subscripts (H with a, b, c) are set by _subline.
# Every move is rigid (shift, Rotate, a glide that turns while it travels,
# or a fold, i.e. a half-turn in space about a line of the plane) unless it
# is an announced similarity. Every landing, angle, length and tiling claim
# is checked numerically with check(...) before or right after it is drawn,
# and the labels are checked against the lines, marks and other labels they
# must clear, so a wrong construction fails the render instead of drawing a
# wrong picture. J26 is a caution: it draws the false figure first, then
# the true one, and shows exactly where the argument fails.


# ------------------------------------------------------------ geometry helpers

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


def _circumcentre(A, B, C):
    A, B, C = to3(A), to3(B), to3(C)
    return _meet((A + B) / 2, (A + B) / 2 + _perp(B - A),
                 (B + C) / 2, (B + C) / 2 + _perp(C - B))


def _orthocentre(A, B, C):
    A, B, C = to3(A), to3(B), to3(C)
    return _meet(A, A + _perp(C - B), B, B + _perp(A - C))


def _param(p, a, b):
    """Position of p along a -> b (0 at a, 1 at b), for a point on that line."""
    a, b, p = to3(a), to3(b), to3(p)
    d = b - a
    return float(np.dot(p - a, d) / np.dot(d, d))


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


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


def _glide(mob, pivot, target, angle, **kw):
    """Rigid on every frame: the piece turns by `angle` about its pivot
    while the pivot travels straight to `target`."""
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


# ------------------------------------------------------------ area / polygons

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


def _disjoint(polys, n=140):
    """True if no sample point lies in two of the polygons."""
    pts = np.array([to3(p) for P in polys for p in P])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for i in range(n):
        for j in range(n):
            x = lo[0] + (hi[0] - lo[0]) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = lo[1] + (hi[1] - lo[1]) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            if sum(_pip((x, y), P) for P in polys) > 1:
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


# ------------------------------------------------------------ text helpers

_BASE_CHARS = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789+=")


def _baseline(t):
    """Baseline of a Text: the median bottom of its non-descending glyphs
    (Text drops spaces from both .text and its glyphs, so they align)."""
    ys = [g.get_bottom()[1] for ch, g in zip(t.text, t.submobjects)
          if ch in _BASE_CHARS]
    if not ys:
        ys = [g.get_bottom()[1] for g in t.submobjects]
    return float(np.median(ys))


def _subline(parts, size=26, color=WHITE, sub_scale=0.66, drop=0.3):
    """A line of text with true subscripts, without TeX and without relying
    on subscript-letter glyphs (absent from some fonts). parts: list of
    (string, is_subscript); a subscript sits `drop` x-heights below the
    common baseline, smaller."""
    xh = Text("x", font_size=size).height
    sp = Text("x x", font_size=size).width - Text("xx", font_size=size).width
    g = VGroup()
    x = 0.0
    for s_, sub in parts:
        lead = len(s_) - len(s_.lstrip(" "))
        trail = len(s_) - len(s_.rstrip(" "))
        x += lead * sp
        t = Text(s_.strip(" "), font_size=size * (sub_scale if sub else 1.0),
                 color=color)
        y0 = -drop * xh if sub else 0.0
        t.shift(np.array([x - t.get_left()[0], y0 - _baseline(t), 0.0]))
        x = t.get_right()[0] + trail * sp + (0.01 if sub else 0.02)
        g.add(t)
    return g


def _hsub(letter, size=28, color=WHITE):
    """H with a letter subscript (Hₐ, H_b, H_c), set by _subline."""
    return _subline([("H", False), (letter, True)], size, color)


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


def _trow(parts, size=28, buff=0.16):
    """A row of coloured Text pieces [(s, colour), ...] on one baseline."""
    g = VGroup()
    x = 0.0
    for s, col in parts:
        t = _on_base(s, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + buff
        g.add(t)
    return g


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


def _circle_segs(c, r, n=160):
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(0.0, TAU, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _arc_segs(c, r, a0, a1, n=24):
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _angle_segs(v, p, q, r, n=14):
    """The arc that angle_arc(v, p, q, r) draws, as segments."""
    v = to3(v)
    a1 = float(np.arctan2(*(to3(p) - v)[1::-1]))
    a2 = float(np.arctan2(*(to3(q) - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return _arc_segs(v, r, a1, a1 + span, n)


def _ra_segs(v, p, q, s):
    """The two strokes of a right-angle mark, as segments."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    return [(v + s * u1, v + s * (u1 + u2)), (v + s * (u1 + u2), v + s * u2)]


def _mob_segs(m):
    """Straight pieces of every stroke in a mobject family (ticks, marks,
    lines), sampled from their anchors."""
    out = []
    for sm in m.get_family():
        pts = sm.get_anchors() if sm.has_points() else []
        for i in range(len(pts) - 1):
            if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-6:
                out.append((pts[i], pts[i + 1]))
    return out


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


def _in_frame(m, tol=0.02):
    return (m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
            and m.get_top()[1] <= SAFE_TOP + tol
            and m.get_bottom()[1] >= SAFE_BOTTOM - tol)


def _name(m):
    t = getattr(m, "text", None)
    if t:
        return t
    subs = [getattr(s, "text", "") for s in m.submobjects]
    return "".join(subs) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.06, gap=0.05):
    """Every label inside the safe area, clear of every line in segs and of
    every other label."""
    for m in labels:
        check(_in_frame(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{_name(m)}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(not _overlap(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


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


def _park_dirs(m, X, dirs_deg, segs, d0=0.25, d1=0.9, pad=0.06, avoid=(), sep=0.08):
    """Put label m next to X along the first of the given directions
    (degrees) where, at the least distance in [d0, d1], it clears every
    segment in segs and stays `sep` away from the mobjects in `avoid`."""
    X = to3(X)
    for a in dirs_deg:
        for d in np.arange(d0, d1 + 1e-9, 0.02):
            m.move_to(X + d * _polar(a * DEGREES))
            if (_hit(m, segs, pad) is None and _in_frame(m)
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"a free spot for the label '{_name(m)}'")


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


def _angle_label(m, V, P, Q, r0, pad=0.06, r1=3.0, segs=(),
                 fracs=(0.5,), avoid=(), sep=0.05):
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
    for f in fracs:
        u = _polar(a1 + f * span)
        for d in np.arange(r0, r1, 0.03):
            m.move_to(V + d * u)
            if (_hit(m, arms, pad) is None
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"room for the angle label '{_name(m)}'")


def _frac_in_angle(V, P, Q, X):
    """Where the ray V -> X lies inside the non-reflex angle P-V-Q, as the
    fraction _angle_label uses (0 at its start arm, 1 at its end arm)."""
    V = to3(V)
    a1 = float(np.arctan2(*(to3(P) - V)[1::-1]))
    a2 = float(np.arctan2(*(to3(Q) - V)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    ax = float(np.arctan2(*(to3(X) - V)[1::-1]))
    return ((ax - a1) % TAU) / span


def _along(m, p, q, side, gap=0.1, at=0.5):
    """Turn label m to run along the segment pq (reading left to right) and
    park it beside pq, on the side `side` points to. Returns the angle."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    ang = float(np.arctan2(d[1], d[0]))
    if ang > PI / 2:
        ang -= PI
    elif ang < -PI / 2:
        ang += PI
    nrm = np.array([-d[1], d[0], 0.0])
    if np.dot(nrm, to3(side)) < 0:
        nrm = -nrm
    h = m.height
    m.rotate(ang)
    m.move_to(p + at * (q - p) + (h / 2 + gap) * nrm)
    return ang


def _clear_rot(m, ang, segs, pad=0.06):
    """A label turned by ang: its own (unturned) box, grown by pad, meets
    none of the segments (tested in the label's frame)."""
    c = m.get_center()
    flat = m.copy().rotate(-ang, about_point=c)
    x0, x1, y0, y1 = _box(flat, pad)
    for p_, q_ in segs:
        a, b = _rot2(p_, c, -ang), _rot2(q_, c, -ang)
        L = float(np.linalg.norm(b - a))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.02) + 2)):
            x, y = (a + t * (b - a))[:2]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return False
    return True


def _box_segs(m, pad=0.0):
    """The four edges of m's box, as segments."""
    x0, x1, y0, y1 = _box(m, pad)
    c = [np.array([x0, y0, 0.0]), np.array([x1, y0, 0.0]),
         np.array([x1, y1, 0.0]), np.array([x0, y1, 0.0])]
    return [(c[i], c[(i + 1) % 4]) for i in range(4)]


def _all_in_frame(mobs, what):
    for i, m in enumerate(mobs):
        check(_in_frame(m), f"{what}: item {i} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")


def _final_check(scene, cap, rotated=()):
    """Closing frame: everything but the caption inside the safe area, the
    caption inside the caption band, and no two text labels overlapping.
    `rotated` lists (label, angle) for labels turned along a line: those
    are compared in their own frame."""
    rot = {id(t): a for t, a in rotated}
    skip = {id(x) for x in cap.get_family()}
    check(cap.width <= 13.45 and cap.get_bottom()[1] > -4.0
          and cap.get_top()[1] < SAFE_BOTTOM, "caption in the caption band")
    shown = [m for m in scene.mobjects if id(m) not in skip
             and not isinstance(m, ValueTracker)]
    for m in shown:
        if not m.has_points() and not m.submobjects:
            continue
        check(_in_frame(m), f"{type(m).__name__} inside the safe area "
              f"[{m.get_left()[0]:.2f}, {m.get_right()[0]:.2f}] x "
              f"[{m.get_bottom()[1]:.2f}, {m.get_top()[1]:.2f}]")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if id(a) in rot or id(b) in rot:
                if id(b) in rot:
                    a, b = b, a
                ok = _clear_rot(a, rot[id(a)], _box_segs(b), 0.0)
            else:
                ok = not _overlap(a, b, 0.0)
            check(ok, f"labels '{a.text}' and '{b.text}' apart")


PHI = (1 + np.sqrt(5)) / 2


# ===================================================================== F52

class F52_Pompeiu(Board):
    """ABC equilateral, P any point. Turn the triangle PBC by 60° about C
    (B goes to A, P to P'): P'A = PB, and CP = CP' with a 60° angle between
    them, so CPP' is equilateral and PP' = PC. The triangle APP' therefore
    has the sides PA, PB, PC. (It is flat exactly when P is on the
    circumcircle; the P shown is not.)"""

    def construct(self):
        Am = np.array([0.5, np.sqrt(3) / 2])
        Bm, Cm = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        Pm = np.array([0.36, 0.25])
        F = Frame(-0.08, 1.08, -0.1, 0.95, max_w=7.2,
                  centre=(-2.75, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, P = F.P(Am), F.P(Bm), F.P(Cm), F.P(Pm)
        th = -PI / 3                                   # B -> A about C
        Q = _rot2(P, C, th)                            # P'
        n = np.linalg.norm

        check(close(_rot2(B, C, th), A), "the turn about C takes B to A")
        check(abs(n(A - B) - n(B - C)) < 1e-9 and abs(n(B - C) - n(C - A)) < 1e-9,
              "ABC is equilateral")
        check(_pip(P, [A, B, C]), "P inside ABC")
        check(abs(n(Q - A) - n(P - B)) < 1e-9, "P'A = PB")
        check(abs(n(Q - P) - n(P - C)) < 1e-9 and abs(_angle(C, P, Q) - PI / 3) < 1e-9,
              "CPP' equilateral: PP' = PC")
        la, lb, lc = n(P - A), n(P - B), n(P - C)
        check(min(la, lb, lc) > 0.25 * F.k and abs(la - lb) > 0.05 * F.k
              and abs(lb - lc) > 0.05 * F.k and abs(la - lc) > 0.03 * F.k,
              "PA, PB, PC visibly different")
        check(la < lb + lc and lb < la + lc and lc < la + lb,
              "PA, PB, PC satisfy the strict triangle inequality")

        cA, cB, cC = BLUE_B, TEAL_B, ORANGE
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        tk = VGroup(*[_ticks(p, q, 1, GREY_A) for p, q in ((A, B), (B, C), (C, A))])
        lA = tag("A", 30).move_to(A + 0.36 * UP)
        lB = tag("B", 30).move_to(B + 0.36 * _unit(np.array([-1.0, -0.6, 0])))
        lC = tag("C", 30).move_to(C + 0.36 * _unit(np.array([1.0, -0.6, 0])))
        self.play(Create(tri), FadeIn(tk), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                  run_time=1.2)

        sPA = Line(P, A, color=cA, stroke_width=5)
        sPB = Line(P, B, color=cB, stroke_width=5)
        sPC = Line(P, C, color=cC, stroke_width=5)
        dP = Dot(P, radius=0.07, color=WHITE)
        lP = tag("P", 30)
        _park(lP, P, [A, B, C, Q], _segs(P, A) + _segs(P, B) + _segs(P, C)
              + _segs(P, Q), d0=0.3, avoid=[lA, lB, lC])
        self.play(Create(sPA), Create(sPB), Create(sPC), FadeIn(dP), FadeIn(lP),
                  run_time=1.2)
        self.hold(0.4)

        # turn the triangle PBC by 60° about C: B lands on A, P on P'
        piece = VGroup(mk([P, B, C], TEAL_D, op=0.45, stroke_width=0),
                       Line(P, B, color=cB, stroke_width=5),
                       Line(P, C, color=cC, stroke_width=5))
        self.play(FadeIn(piece[0]), run_time=0.5)
        self.add(piece)
        rot_arc = angle_arc(C, B, A, 0.62, YELLOW_B, 4)
        l60a = tag("60°", 22, YELLOW_B)           # (PC runs inside the angle BCA)
        _angle_label(l60a, C, P, A, 0.62 + 0.32, pad=0.12,
                     segs=_segs(P, C) + _segs(A, B) + _mob_segs(tk)
                     + _angle_segs(C, B, A, 0.62))
        self.play(Create(rot_arc), FadeIn(l60a), run_time=0.6)
        self.play(Rotate(piece, angle=th, about_point=C), run_time=2.0)
        check(close(piece[1].get_start(), Q, 1e-6) and close(piece[1].get_end(), A, 1e-6),
              "PB lands on P'A")
        check(close(piece[2].get_start(), Q, 1e-6) and close(piece[2].get_end(), C, 1e-6),
              "PC lands on P'C")
        check(_same_poly(piece[0].get_vertices(), [Q, A, C], 1e-6),
              "the turned triangle is P'AC")
        dQ = Dot(Q, radius=0.07, color=WHITE)
        lQ = tag("P'", 30)
        _park(lQ, Q, [A, C, P], _segs(Q, A) + _segs(Q, C) + _segs(Q, P)
              + _segs(A, C), d0=0.3, avoid=[lA, lC])
        self.play(FadeIn(dQ), FadeIn(lQ), FadeOut(rot_arc), FadeOut(l60a),
                  FadeOut(piece[0]), run_time=0.7)

        # CP = CP' with 60° between them: CPP' is equilateral, PP' = PC.
        # (Side CA runs inside the angle PCP', so its label sits in the
        # middle of the wider part, between CP and CA, just outside the arc.)
        sPQ = Line(P, Q, color=cC, stroke_width=5)
        r_arc = 0.95
        arc = angle_arc(C, P, Q, r_arc, YELLOW_B, 4)
        tk2 = VGroup(_ticks(C, P, 2, cC, at=0.42), _ticks(C, Q, 2, cC, at=0.42))
        segs_fig = (_segs(A, B, C, closed=True) + _segs(P, A) + _segs(P, B)
                    + _segs(P, C) + _segs(Q, A) + _segs(Q, C) + _segs(P, Q)
                    + _mob_segs(tk) + _mob_segs(tk2))
        check(_angle(C, P, A) > _angle(C, A, Q) and _angle(C, P, A) > 30 * DEGREES,
              "the part PCA of the angle PCP' is the wider one, room for the label")
        l60 = tag("60°", 22, YELLOW_B)
        _angle_label(l60, C, P, A, r_arc + 0.32, pad=0.12,
                     segs=segs_fig + _angle_segs(C, P, Q, r_arc),
                     avoid=[lA, lB, lC, lP, lQ], sep=0.06)
        self.play(Create(arc), FadeIn(l60), FadeIn(tk2), run_time=0.8)
        tk3 = _ticks(P, Q, 2, cC, at=0.5)
        eq = mk([C, P, Q], ORANGE, op=0.3, stroke_width=0)
        self.add(eq)
        self.bring_to_back(eq)
        self.play(Create(sPQ), FadeIn(tk3), FadeIn(eq), run_time=0.9)
        self.hold(0.5)

        # the triangle APP': sides PA, P'A = PB, PP' = PC
        tri_apq = mk([A, P, Q], YELLOW_E, op=0.5, stroke_width=0)
        self.add(tri_apq)
        self.bring_to_back(tri_apq)
        self.play(FadeOut(eq), FadeIn(tri_apq), run_time=0.9)
        self.bring_to_front(sPA, piece[1], sPQ, dP, dQ)

        # a copy of APP' slides out to the right, its sides named
        shift = np.array([3.85, -0.55, 0.0]) - (A + P + Q) / 3
        copy = VGroup(mk([A, P, Q], YELLOW_E, op=0.5, stroke_width=0),
                      Line(A, P, color=cA, stroke_width=6),
                      Line(A, Q, color=cB, stroke_width=6),
                      Line(P, Q, color=cC, stroke_width=6))
        self.add(copy)
        self.play(copy.animate.shift(shift), run_time=1.6)
        A2, P2, Q2 = A + shift, P + shift, Q + shift
        check(_same_poly(copy[0].get_vertices(), [A2, P2, Q2], 1e-6),
              "the copy is APP', moved without turning")
        G2 = (A2 + P2 + Q2) / 3
        tPA = _beside(tag("PA", 28, cA), A2, P2, (A2 + P2) / 2 - G2, 0.14)
        tPB = _beside(tag("PB", 28, cB), A2, Q2, (A2 + Q2) / 2 - G2, 0.14)
        tPC = _beside(tag("PC", 28, cC), P2, Q2, (P2 + Q2) / 2 - G2, 0.14)
        self.play(FadeIn(tPA), FadeIn(tPB), FadeIn(tPC), run_time=0.8)

        _labels_ok([lA, lB, lC, lP, lQ, l60], segs_fig + _angle_segs(C, P, Q, 0.95),
                   "F52 figure")
        _labels_ok([tPA, tPB, tPC], _segs(A2, P2, Q2, closed=True), "F52 copy")
        check(copy.get_left()[0] > tri.get_right()[0] + 0.4, "the copy clear of the figure")
        cap = caption("PA, PB, PC are the sides of a triangle", 32)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== F46

class F46_BisectRightAngle(Board):
    """ABC has its right angle at C; O is the centre of the square drawn
    outward on the hypotenuse AB. The diagonals of the square meet at right
    angles, so O sees AB at 90°, as C does: both lie on the circle with
    diameter AB. OA = OB (half-diagonals) are equal chords, so they are
    seen from C under equal angles: CO bisects the right angle, 45° each.
    Shown for every right triangle on the same hypotenuse: C rides the
    semicircle. (AB is drawn upright, the square to its left.)"""

    def construct(self):
        F = Frame(-2.12, 1.3, -1.16, 1.16)
        A, B = F.P((0.0, 1.0)), F.P((0.0, -1.0))
        M = (A + B) / 2
        R = np.linalg.norm(B - A) / 2
        S1, S2 = F.P((-2.0, -1.0)), F.P((-2.0, 1.0))       # far side of the square
        O = (A + B + S1 + S2) / 4
        phi = ValueTracker(30.0)

        def Cp(deg=None):
            d = phi.get_value() if deg is None else deg
            return M + R * _polar(d * DEGREES)

        # the claims, checked over the whole ride of C
        check(abs(np.linalg.norm(O - M) - R) < 1e-9, "O lies on the circle on AB")
        check(abs(_angle(O, A, B) - PI / 2) < 1e-9, "the diagonals meet at 90°")
        check(abs(np.linalg.norm(O - A) - np.linalg.norm(O - B)) < 1e-9, "OA = OB")
        for d in np.linspace(-70, 70, 29):
            C = Cp(d)
            check(abs(_angle(C, A, B) - PI / 2) < 1e-9, "C sees AB at 90°")
            check(abs(_angle(C, A, O) - PI / 4) < 1e-9 and
                  abs(_angle(C, O, B) - PI / 4) < 1e-9, "∠ACO = ∠OCB = 45°")

        cT, cO = TEAL_B, ORANGE
        sq = mk([A, B, S1, S2], BLUE_D, op=0.35, stroke_color=BLUE_B, stroke_width=3)
        diag = VGroup(DashedLine(A, S1, color=GREY_B, stroke_width=2, dash_length=0.09),
                      DashedLine(B, S2, color=GREY_B, stroke_width=2, dash_length=0.09))
        raO = _ra(O, A, B, 0.24, WHITE)
        lO = _park_dirs(tag("O", 30), O, [180], _segs(A, S1) + _segs(B, S2)
                        + _ra_segs(O, A, B, 0.24), d0=0.35, d1=1.2, pad=0.1)
        lA = tag("A", 30).move_to(A + np.array([-0.34, 0.27, 0.0]))
        lB = tag("B", 30).move_to(B + np.array([-0.34, -0.27, 0.0]))

        def tri():
            return Polygon(A, B, Cp(), stroke_color=WHITE, stroke_width=4)

        def raC():
            return _ra(Cp(), A, B, 0.24, WHITE)

        def lC():
            return tag("C", 30).move_to(Cp() + 0.38 * _unit(Cp() - M))

        tri_m, raC_m, lC_m = always_redraw(tri), always_redraw(raC), always_redraw(lC)
        self.play(Create(tri_m), FadeIn(lA), FadeIn(lB), FadeIn(lC_m), run_time=1.0)
        self.play(Create(raC_m), run_time=0.5)

        # the square on AB, its centre O: the diagonals meet at 90°
        self.add(sq)
        self.bring_to_back(sq)
        self.play(FadeIn(sq), run_time=0.8)
        self.play(Create(diag), run_time=0.7)
        dO = Dot(O, radius=0.06)
        self.play(Create(raO), FadeIn(lO), FadeIn(dO), run_time=0.6)
        self.hold(0.3)

        # both see AB at a right angle: one circle on the diameter AB
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(M)
        dM = Dot(M, radius=0.05, color=GREY_B)
        self.play(Create(circ), FadeIn(dM), run_time=1.3)
        self.play(Indicate(raC_m, color=YELLOW_B), Indicate(raO, color=YELLOW_B),
                  run_time=0.9)

        # equal chords OA = OB, equal arcs
        aA, aO, aB = (_dir(M, X) for X in (A, O, B))
        arcAO = Arc(radius=R, start_angle=aA, angle=(aO - aA) % TAU, arc_center=M,
                    color=cT, stroke_width=7)
        arcOB = Arc(radius=R, start_angle=aO, angle=(aB - aO) % TAU, arc_center=M,
                    color=cO, stroke_width=7)
        check(abs((aO - aA) % TAU - PI / 2) < 1e-9 and abs((aB - aO) % TAU - PI / 2) < 1e-9,
              "arcs AO and OB are quarter circles on the far side of AB")
        chOA = Line(O, A, color=cT, stroke_width=6)
        chOB = Line(O, B, color=cO, stroke_width=6)
        tks = VGroup(_ticks(O, A, 1, cT, at=0.5), _ticks(O, B, 1, cO, at=0.5))
        self.play(Create(chOA), Create(chOB), FadeIn(tks), run_time=0.8)
        self.play(Create(arcAO), Create(arcOB), run_time=0.8)

        # seen from C: equal angles, 45° each
        r_arc = 0.62

        def co():
            return Line(Cp(), O, color=YELLOW_B, stroke_width=5)

        def arcs():
            return VGroup(angle_arc(Cp(), A, O, r_arc, cT, 5),
                          angle_arc(Cp(), O, B, r_arc, cO, 5))

        def lab_pos(C, X, Y):
            return C + 1.08 * angle_mid_dir(C, X, Y)

        def labs():
            C = Cp()
            return VGroup(tag("45°", 24, cT).move_to(lab_pos(C, A, O)),
                          tag("45°", 24, cO).move_to(lab_pos(C, O, B)))

        # the labels clear the lines and marks at every stage of the ride
        for d in np.linspace(-52, 36, 23):
            C = Cp(d)
            ls = [tag("45°", 24).move_to(lab_pos(C, A, O)),
                  tag("45°", 24).move_to(lab_pos(C, O, B)),
                  tag("C", 30).move_to(C + 0.38 * _unit(C - M))]
            segs = (_segs(A, B, C, closed=True) + _segs(C, O)
                    + _angle_segs(C, A, O, r_arc) + _angle_segs(C, O, B, r_arc)
                    + _ra_segs(C, A, B, 0.24) + _circle_segs(M, R))
            _labels_ok(ls, segs, f"F46 labels at {d:.0f}°", pad=0.05)

        co_m, arcs_m, labs_m = always_redraw(co), always_redraw(arcs), always_redraw(labs)
        self.play(Create(co_m), run_time=0.7)
        self.play(Create(arcs_m), FadeIn(labs_m), run_time=0.9)
        self.hold(0.6)

        # every right triangle on AB: C rides the semicircle
        self.play(phi.animate.set_value(-50.0), run_time=2.4, rate_func=smooth)
        self.play(phi.animate.set_value(30.0), run_time=2.2, rate_func=smooth)
        for m in (tri_m, raC_m, lC_m, co_m, arcs_m, labs_m):
            m.clear_updaters()

        C = Cp()
        segs = (_segs(A, B, C, closed=True) + _segs(C, O) + _segs(A, S1) + _segs(B, S2)
                + _segs(A, B, S1, S2, closed=True) + _circle_segs(M, R)
                + _angle_segs(C, A, O, r_arc) + _angle_segs(C, O, B, r_arc)
                + _ra_segs(C, A, B, 0.24) + _ra_segs(O, A, B, 0.24)
                + _mob_segs(tks))
        _labels_ok([lA, lB, lC_m, lO, *labs_m], segs, "F46 final")
        cap = caption("CO bisects the right angle:  ∠ACO = ∠OCB = 45°", 32)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== F50

def _icon(poly, k, color, op=FILL):
    """A small copy of a polygon, scaled by k."""
    return mk([to3(p) for p in poly], color, op, stroke_width=1.5).scale(k)


def _wedge_poly(V, P, Q, r, n=24):
    """Outline of the sector _wedge(V, P, Q, r) as a polygon."""
    V = to3(V)
    a1 = float(np.arctan2(*(to3(P) - V)[1::-1]))
    a2 = float(np.arctan2(*(to3(Q) - V)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return [V] + [V + r * _polar(a1 + span * k / n) for k in range(n + 1)]


def _clear_of_poly(m, poly, pad=0.04):
    """m's box (grown by pad) neither crosses the polygon's outline nor
    has a corner, edge midpoint or centre inside it."""
    x0, x1, y0, y1 = _box(m, pad)
    probes = [(x, y) for x in (x0, (x0 + x1) / 2, x1) for y in (y0, (y0 + y1) / 2, y1)]
    return (_hit(m, _segs(*poly, closed=True), pad) is None
            and not any(_pip(q, poly) for q in probes))


def _sample_close(m1, m2, tol=1e-6):
    """Two mobjects drawn by the same points (any order of the samples)."""
    P, Q = m1.get_all_points(), m2.get_all_points()
    if len(P) == 0 or len(Q) == 0:
        return False
    d1 = max(min(np.linalg.norm(Q - p, axis=1)) for p in P)
    d2 = max(min(np.linalg.norm(P - q, axis=1)) for q in Q)
    return max(d1, d2) < tol


class F50_Brahmagupta(Board):
    """ABCD is inscribed in a circle and its diagonals cross at right angles
    at E. Draw the line through E perpendicular to BC; it meets AD at M.
    The angle α = ∠DBC is seen again at A (∠DAC: the same arc DC), and a
    quarter-turn carries it onto ∠AEM (its arms turn perpendicular to BE
    and BC, i.e. along EA and EM). So the triangle AEM has two angles α:
    MA = ME. In the same way β = ∠ACB is ∠ADB and ∠MED: MD = ME. So M is
    the midpoint of AD."""

    def construct(self):
        p_, q_ = 0.25, 0.15                       # E, where the chords cross
        Am = np.array([-np.sqrt(1 - q_ ** 2), q_])
        Cm = np.array([np.sqrt(1 - q_ ** 2), q_])
        Bm = np.array([p_, -np.sqrt(1 - p_ ** 2)])
        Dm = np.array([p_, np.sqrt(1 - p_ ** 2)])
        F = Frame(-1.2, 1.2, -1.15, 1.15, max_w=7.0,
                  centre=(-2.55, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D = F.P(Am), F.P(Bm), F.P(Cm), F.P(Dm)
        Oc, R = F.P((0.0, 0.0)), F.k
        E = _meet(A, C, B, D)
        Ff = _foot(E, B, C)                        # foot on BC
        M = _meet(E, Ff, A, D)
        n = np.linalg.norm

        check(all(abs(n(X - Oc) - R) < 1e-9 for X in (A, B, C, D)), "ABCD is cyclic")
        check(abs(_angle(E, A, B) - PI / 2) < 1e-9, "the diagonals are perpendicular")
        check(0 < _param(Ff, B, C) < 1 and 0 < _param(M, A, D) < 1,
              "the perpendicular meets BC and AD inside the sides")
        check(close(M, (A + D) / 2), "M is the midpoint of AD")
        al, be = _angle(B, D, C), _angle(C, A, B)
        check(abs(_angle(A, D, C) - al) < 1e-9 and abs(_angle(E, A, M) - al) < 1e-9,
              "α at B, at A and at E")
        check(abs(_angle(D, A, B) - be) < 1e-9 and abs(_angle(E, M, D) - be) < 1e-9,
              "β at C, at D and at E")
        check(abs(al + be - PI / 2) < 1e-9 and 25 * DEGREES < al < be - 10 * DEGREES,
              "α + β = 90°, both visible and distinct")
        # a quarter-turn takes the arms of α at B onto EA, EM (and β at C onto ED, EM)
        check(close(_rot2(_unit(E - B), ORIGIN, PI / 2), _unit(A - E))
              and close(_rot2(_unit(C - B), ORIGIN, PI / 2), _unit(M - E)),
              "α: the arms BE, BC turn a quarter onto EA, EM")
        check(close(_rot2(_unit(E - C), ORIGIN, -PI / 2), _unit(D - E))
              and close(_rot2(_unit(B - C), ORIGIN, -PI / 2), _unit(M - E)),
              "β: the arms CE, CB turn a quarter onto ED, EM")

        cAl, cBe = TEAL_D, ORANGE
        circ = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(Oc)
        quad = Polygon(A, B, C, D, stroke_color=WHITE, stroke_width=4)
        diags = VGroup(Line(A, C, color=WHITE, stroke_width=3),
                       Line(B, D, color=WHITE, stroke_width=3))
        raE = _ra(E, A, B, 0.22)
        dots = VGroup(*[Dot(X, radius=0.06) for X in (A, B, C, D, E)])
        lA = tag("A", 30).move_to(A + 0.38 * LEFT)
        lB = tag("B", 30).move_to(B + 0.38 * DOWN)
        lC = tag("C", 30).move_to(C + 0.38 * RIGHT)
        lD = tag("D", 30).move_to(D + 0.38 * UP)
        lE = tag("E", 28).move_to(E + 0.4 * _polar(45 * DEGREES))
        self.play(Create(circ), run_time=0.9)
        self.play(Create(quad), FadeIn(dots), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                  FadeIn(lD), run_time=1.1)
        self.play(Create(diags), Create(raE), FadeIn(lE), run_time=0.9)

        # the line through E perpendicular to BC, on to AD
        perp = Line(Ff, M, color=YELLOW_B, stroke_width=4)
        raF = _ra(Ff, E, C, 0.2, YELLOW_B)
        dM = Dot(M, radius=0.06, color=YELLOW_B)
        lM = tag("M", 30, YELLOW_B).move_to(M + 0.4 * _unit(M - E))
        self.play(Create(perp), Create(raF), run_time=1.0)
        self.play(FadeIn(dM), FadeIn(lM), run_time=0.5)
        self.hold(0.3)

        rw = 0.56                                  # wedge radius
        th_A, th_B = _dir(Oc, A), _dir(Oc, B)
        th_C, th_D = _dir(Oc, C), _dir(Oc, D)

        def greek(s, col, V, P, Q):
            return tag(s, 28, col).move_to(to3(V) + (rw + 0.31) * angle_mid_dir(V, P, Q))

        # ---- α: at B, along the circle to A, and a quarter-turn onto ∠AEM
        wB = _wedge(B, D, C, rw, cAl, op=0.85, stroke=0)
        gB = greek("α", TEAL_B, B, D, C)
        self.play(FadeIn(wB), FadeIn(gB), run_time=0.6)
        t = ValueTracker(th_B)

        def slide_al():
            X = Oc + R * _polar(t.get_value())
            return _wedge(X, D, C, rw, cAl, op=0.85, stroke=0)
        sl = always_redraw(slide_al)
        self.add(sl)
        # B -> A the short way round, away from C and D
        tgt = th_A - TAU if th_A > th_B else th_A
        check(tgt < th_B and all(not (tgt < a < th_B) and not (tgt < a - TAU < th_B)
                                 for a in (th_C, th_D)), "the arc B -> A misses C and D")
        self.play(t.animate.set_value(tgt), run_time=2.0, rate_func=smooth)
        sl.clear_updaters()
        check(_sample_close(sl, _wedge(A, D, C, rw, cAl, op=0.85, stroke=0), 1e-6),
              "the wedge arrives as ∠DAC")
        gA = greek("α", TEAL_B, A, D, C)
        self.play(FadeIn(gA), run_time=0.4)
        wq = wB.copy()
        self.add(wq)
        self.play(_glide(wq, B, E, PI / 2), run_time=1.8)
        check(_sample_close(wq, _wedge(E, A, M, rw, cAl, op=0.85, stroke=0), 1e-6),
              "the quarter-turn lands on ∠AEM")
        gE1 = greek("α", TEAL_B, E, A, M)
        tAEM = mk([A, E, M], cAl, op=0.25, stroke_width=0)
        tk1 = VGroup(_ticks(M, A, 1, WHITE), _ticks(M, E, 1, WHITE))
        self.add(tAEM)
        self.bring_to_back(tAEM)
        self.play(FadeIn(gE1), FadeIn(tAEM), FadeIn(tk1), run_time=0.9)
        self.hold(0.4)

        # ---- β: at C, along the circle to D, and a quarter-turn onto ∠MED
        wC = _wedge(C, A, B, rw, cBe, op=0.85, stroke=0)
        gC = greek("β", ORANGE, C, A, B)
        self.play(FadeIn(wC), FadeIn(gC), run_time=0.6)
        u = ValueTracker(th_C)

        def slide_be():
            X = Oc + R * _polar(u.get_value())
            return _wedge(X, A, B, rw, cBe, op=0.85, stroke=0)
        sl2 = always_redraw(slide_be)
        self.add(sl2)
        tgt2 = th_D if th_D > th_C else th_D + TAU
        check(tgt2 > th_C and all(not (th_C < a < tgt2) and not (th_C < a + TAU < tgt2)
                                  for a in (th_A, th_B)), "the arc C -> D misses A and B")
        self.play(u.animate.set_value(tgt2), run_time=2.0, rate_func=smooth)
        sl2.clear_updaters()
        check(_sample_close(sl2, _wedge(D, A, B, rw, cBe, op=0.85, stroke=0), 1e-6),
              "the wedge arrives as ∠ADB")
        gD = greek("β", ORANGE, D, A, B)
        self.play(FadeIn(gD), run_time=0.4)
        wq2 = wC.copy()
        self.add(wq2)
        self.play(_glide(wq2, C, E, -PI / 2), run_time=1.8)
        check(_sample_close(wq2, _wedge(E, M, D, rw, cBe, op=0.85, stroke=0), 1e-6),
              "the quarter-turn lands on ∠MED")
        gE2 = greek("β", ORANGE, E, M, D)
        tMED = mk([M, E, D], cBe, op=0.25, stroke_width=0)
        tk2 = _ticks(M, D, 1, WHITE)
        self.add(tMED)
        self.bring_to_back(tMED)
        self.play(FadeIn(gE2), FadeIn(tMED), FadeIn(tk2), run_time=0.9)

        # readout: two isosceles triangles
        k_ic = 0.42
        r1 = VGroup(_icon([A, E, M], k_ic, cAl, 0.8), tag("MA = ME", 32)).arrange(RIGHT, buff=0.3)
        r2 = VGroup(_icon([M, E, D], k_ic, cBe, 0.8), tag("MD = ME", 32)).arrange(RIGHT, buff=0.3)
        r3 = tag("AM = MD", 36, YELLOW_B)
        for r_, y in ((r1, 1.6), (r2, 0.35), (r3, -1.0)):
            r_.move_to(np.array([4.15, y, 0.0]))
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.6)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.6)
        self.play(FadeIn(r3), run_time=0.6)

        greeks = [gB, gA, gE1, gC, gD, gE2]
        segs = (_segs(A, B, C, D, closed=True) + _segs(A, C) + _segs(B, D) + _segs(Ff, M)
                + _circle_segs(Oc, R) + _ra_segs(E, A, B, 0.22) + _ra_segs(Ff, E, C, 0.2)
                + _mob_segs(tk1) + _mob_segs(tk2))
        _labels_ok([lA, lB, lC, lD, lE, lM] + greeks, segs, "F50 labels")
        wpolys = [_wedge_poly(B, D, C, rw), _wedge_poly(A, D, C, rw), _wedge_poly(E, A, M, rw),
                  _wedge_poly(C, A, B, rw), _wedge_poly(D, A, B, rw), _wedge_poly(E, M, D, rw)]
        for l_ in greeks + [lA, lB, lC, lD, lE, lM]:
            for iw, wp in enumerate(wpolys):
                check(_clear_of_poly(l_, wp), f"F50: '{_name(l_)}' at "
                      f"{l_.get_center()[:2].round(2)} clear of wedge #{iw}")
        _labels_ok([r1, r2, r3], [], "F50 readout")
        check(r1.get_left()[0] > circ.get_right()[0] + 0.5, "readout clear of the circle")
        cap = caption("EM ⊥ BC  ⟹  AM = MD", 34)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== F60

class F60_TwoSquares(Board):
    """Squares ABCD and AEFG share the corner A and leave two triangles
    between them, ABE and ADG (their angles at A add up to 180°). Turn ADG
    a quarter-turn about A: D lands on B and G on G', the point opposite E
    across A. Now ABE and ABG' have equal bases AE = AG' on one line and
    the same apex B, so the same height h: equal areas."""

    def construct(self):
        s1, s2, ph = 2.0, 1.6, 150 * DEGREES
        Am, Bm, Cm, Dm = (np.array([0.0, 0.0]), np.array([s1, 0.0]),
                          np.array([s1, s1]), np.array([0.0, s1]))
        Gm = s2 * np.array([np.cos(ph), np.sin(ph)])
        Em = s2 * np.array([np.cos(ph + PI / 2), np.sin(ph + PI / 2)])
        Fm = Em + Gm
        Fr = Frame(-2.62, 2.42, -1.78, 2.42, max_w=8.6,
                   centre=(-2.05, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D, E, Fp, G = (Fr.P(X) for X in (Am, Bm, Cm, Dm, Em, Fm, Gm))
        th = -PI / 2
        Gq = _rot2(G, A, th)                       # G'
        H = _foot(B, E, Gq)                        # foot of the height from B
        n = np.linalg.norm

        check(area([A, B, C, D]) > 0 and area([A, E, Fp, G]) < 0,
              "the squares run opposite ways round A")
        check(close(_rot2(D, A, th), B), "the quarter-turn takes D to B")
        check(close(Gq, 2 * A - E), "and G to G', opposite E across A: E, A, G' in a row")
        check(abs(_angle(A, D, G) + _angle(A, B, E) - PI) < 1e-9,
              "the angles at A add up to 180°")
        check(abs(abs(area([A, B, E])) - abs(area([A, D, G]))) < 1e-9, "[ABE] = [ADG]")
        hgt = n(B - H)
        check(abs(abs(area([A, B, E])) - n(E - A) * hgt / 2) < 1e-9
              and abs(abs(area([A, B, Gq])) - n(Gq - A) * hgt / 2) < 1e-9,
              "each is ½ · base · h")
        check(0 < _param(H, A, Gq) < 1, "the foot of h lies on AG'")
        check(_disjoint([[A, B, C, D], [A, D, G], [A, G, Fp, E], [A, E, B]]),
              "the squares and the two triangles between them do not overlap")

        cS1, cS2 = BLUE_D, GREEN_D
        cT, cG = TEAL_D, GOLD_D
        sq1 = mk([A, B, C, D], cS1, op=0.28, stroke_color=BLUE_B, stroke_width=3)
        sq2 = mk([A, E, Fp, G], cS2, op=0.28, stroke_color=GREEN_B, stroke_width=3)
        tABE = mk([A, B, E], cT, op=0.85, stroke_width=3)
        tADG = mk([A, D, G], cG, op=0.85, stroke_width=3)

        segs0 = (_segs(A, B, C, D, closed=True) + _segs(A, E, Fp, G, closed=True)
                 + _segs(B, E) + _segs(D, G))
        lA = _park_dirs(tag("A", 28), A, [195, 205, 185], segs0, d0=0.3, pad=0.07)
        lB = tag("B", 30).move_to(B + 0.36 * _unit(np.array([1.0, -0.5, 0])))
        lC = tag("C", 30).move_to(C + 0.34 * _unit(np.array([1.0, 1.0, 0])))
        lD = tag("D", 30).move_to(D + 0.34 * _unit(np.array([0.3, 1.0, 0])))
        lE = tag("E", 30).move_to(E + 0.36 * _unit(E - (A + Fp) / 2))
        lF = tag("F", 30).move_to(Fp + 0.36 * _unit(Fp - (A + Fp) / 2 + (Fp - A) * 0.0))
        lG = tag("G", 30).move_to(G + 0.36 * _unit(G - (A + Fp) / 2))

        self.play(FadeIn(sq1), FadeIn(lA), FadeIn(lB), FadeIn(lC), FadeIn(lD), run_time=1.0)
        self.play(FadeIn(sq2), FadeIn(lE), FadeIn(lF), FadeIn(lG), run_time=1.0)
        self.play(FadeIn(tABE), FadeIn(tADG), run_time=0.8)
        self.bring_to_front(lA)
        self.hold(0.5)

        # a quarter-turn about A: D -> B, G -> G' (their paths dashed)
        pathD = DashedVMobject(Arc(radius=n(D - A), start_angle=_dir(A, D), angle=th,
                                   arc_center=A, color=GREY_B, stroke_width=2),
                               num_dashes=22)
        pathG = DashedVMobject(Arc(radius=n(G - A), start_angle=_dir(A, G), angle=th,
                                   arc_center=A, color=GREY_B, stroke_width=2),
                               num_dashes=18)
        cp = tADG.copy()
        self.add(cp)
        self.play(Rotate(cp, angle=th, about_point=A), Create(pathD), Create(pathG),
                  run_time=2.2)
        check(_same_poly(cp.get_vertices(), [A, B, Gq], 1e-6), "the turned copy is ABG'")
        lGq = tag("G'", 30)
        segs1 = segs0 + _segs(A, B, Gq, closed=True)
        _park(lGq, Gq, [A, B, D, C], segs1, d0=0.3, pad=0.08, avoid=[lC, lD])
        self.play(FadeIn(lGq), FadeOut(pathD), FadeOut(pathG), run_time=0.7)

        # equal bases on one line, one apex B
        base = VGroup(Line(A, E, color=YELLOW_B, stroke_width=7),
                      Line(A, Gq, color=YELLOW_B, stroke_width=7))
        tks = VGroup(_ticks(A, E, 2, YELLOW_B, size=0.17, width=4),
                     _ticks(A, Gq, 2, YELLOW_B, size=0.17, width=4, at=0.32))
        check(abs(n(E - A) - n(Gq - A)) < 1e-9 and _on_line(A, E, Gq),
              "AE = AG' on one line")
        self.play(Create(base[0]), Create(base[1]), run_time=0.8)
        self.play(FadeIn(tks), run_time=0.4)
        alt = DashedLine(B, H, color=WHITE, stroke_width=3, dash_length=0.09)
        raH = _ra(H, B, Gq, 0.18)
        lh = tag("h", 30)
        segs2 = (segs1 + _segs(E, Gq) + _segs(B, H) + _ra_segs(H, B, Gq, 0.18)
                 + _mob_segs(tks))
        _park(lh, (B + H) / 2, [B, H], segs2, d0=0.2, pad=0.07,
              avoid=[lA, lB, lC, lD, lGq])
        self.play(Create(alt), Create(raH), FadeIn(lh), run_time=0.9)
        self.hold(0.4)

        # readout: ½ · base · h, twice
        k_ic = 0.3
        r1 = VGroup(_icon([A, B, E], k_ic, cT, 0.9), tag("= ½ · AE · h", 30))
        r2 = VGroup(_icon([A, D, G], k_ic, cG, 0.9), tag("= ½ · AG' · h", 30))
        for r_ in (r1, r2):
            r_.arrange(RIGHT, buff=0.25)
        r1.move_to(np.array([2.3 + r1.width / 2, 1.45, 0.0]))
        r2.move_to(np.array([2.3 + r2.width / 2, 0.1, 0.0]))
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.6)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.6)

        labels = [lA, lB, lC, lD, lE, lF, lG, lGq, lh]
        _labels_ok(labels, segs2, "F60 labels")
        _labels_ok([r1, r2], [], "F60 readout")
        check(r1.get_left()[0] > max(B[0], C[0]) + 0.6, "readout clear of the figure")
        cap = caption("[ABE] = [ADG]", 34)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== F62

class F62_OrthocentreReflections(Board):
    """H is the orthocentre of the acute triangle ABC; the altitudes from B
    and C have their feet E on CA and F on AB. The quadrilateral AFHE has
    right angles at E and F, so its angle at H is 180° − α; the vertical
    angle BHC (a half-turn about H) is 180° − α too. Fold the triangle BHC
    over BC: H lands on Hₐ, and ∠BHₐC = 180° − α is supplementary to the
    opposite angle α at A, so ABHₐC is cyclic: Hₐ is on the circumcircle.
    The same holds for the other two sides. (Shown for an acute triangle.)"""

    def construct(self):
        # angles 50°, 72°, 58° at A, B, C (a small angle at A leaves room
        # inside AFHE, whose diagonal AH = 2R cos A)
        a_, Bd, Cd, Ad = 4.0, 72 * DEGREES, 58 * DEGREES, 50 * DEGREES
        c_ = a_ * np.sin(Cd) / np.sin(Ad)
        Am = c_ * np.array([np.cos(Bd), np.sin(Bd)])
        Bm, Cm = np.array([0.0, 0.0]), np.array([a_, 0.0])
        Om = _circumcentre(Am, Bm, Cm)
        Rm = float(np.linalg.norm(to3(Am) - Om))
        mg = 0.36
        Fr = Frame(Om[0] - Rm - mg, Om[0] + Rm + mg, Om[1] - Rm - mg, Om[1] + Rm + mg,
                   max_w=8.0, centre=(-2.3, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        H = _orthocentre(A, B, C)
        E, Ff = _foot(B, C, A), _foot(C, A, B)     # feet on CA and AB
        Ha, Hb, Hc = _mirror(H, B, C), _mirror(H, C, A), _mirror(H, A, B)
        Oc = _circumcentre(A, B, C)
        R = np.linalg.norm(A - Oc)
        al = _angle(A, B, C)
        n = np.linalg.norm

        angs = (_angle(A, B, C), _angle(B, C, A), _angle(C, A, B))
        check(max(angs) < PI / 2 - 0.1 and min(abs(angs[i] - angs[j]) for i in range(3)
                                               for j in range(i + 1, 3)) > 0.05,
              "an acute scalene triangle")
        check(_pip(H, [A, B, C]), "H inside")
        check(close(_meet(B, E, C, Ff), H) and _on_line(H, A, _foot(A, B, C)),
              "the three altitudes meet at H")
        check(abs(_angle(H, Ff, E) - (PI - al)) < 1e-9, "∠FHE = 180° − α")
        check(abs(_angle(H, B, C) - (PI - al)) < 1e-9, "∠BHC = ∠FHE (vertical angles)")
        check(abs(_angle(Ha, B, C) - (PI - al)) < 1e-9, "∠BHₐC = 180° − α")
        for X in (Ha, Hb, Hc):
            check(abs(n(X - Oc) - R) < 1e-9, "each reflection lies on the circumcircle")
        check(_cross(C - B, A - B) * _cross(C - B, Ha - B) < 0, "A and Hₐ on opposite sides of BC")

        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        lA = tag("A", 30).move_to(A + 0.36 * UP)
        lB = tag("B", 30).move_to(B + 0.4 * _unit(np.array([-1.0, -0.15, 0])))
        lC = tag("C", 30).move_to(C + 0.4 * _unit(np.array([1.0, -0.15, 0])))
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        # two altitudes; they meet at H
        cAlt = GREY_A
        altB = Line(B, E, color=cAlt, stroke_width=3)
        altC = Line(C, Ff, color=cAlt, stroke_width=3)
        raE, raF = _ra(E, B, A, 0.17), _ra(Ff, C, A, 0.17)
        dH = Dot(H, radius=0.06)
        lH = tag("H", 28)
        segs0 = _segs(A, B, C, closed=True) + _segs(B, E) + _segs(C, Ff)
        # (the two wide angles at H will carry labels: H goes in a narrow one)
        mid = angle_mid_dir(H, Ff, B)
        _park_dirs(lH, H, [np.degrees(np.arctan2(mid[1], mid[0]))], segs0,
                   d0=0.25, d1=0.9, pad=0.07)
        self.play(Create(altB), Create(altC), Create(raE), Create(raF), run_time=1.1)
        self.play(FadeIn(dH), FadeIn(lH), run_time=0.4)

        # the quadrilateral AFHE: right angles at E, F, so 180° − α at H
        cQ, cW = PURPLE_B, YELLOW_E
        quad = mk([A, Ff, H, E], cQ, op=0.3, stroke_width=0)
        arcA = angle_arc(A, B, C, 0.5, YELLOW_B, 4)
        lal = tag("α", 28, YELLOW_B)
        # HA (drawn later) splits the angle at A: α goes in the wider part
        fH = _frac_in_angle(A, B, C, H)
        wide = fH / 2 if fH > 0.5 else (1 + fH) / 2
        _angle_label(lal, A, B, C, 0.62, segs=segs0 + _segs(H, A), pad=0.06,
                     fracs=(wide, wide + 0.04, wide - 0.04))
        rw = 0.45
        wH = _wedge(H, Ff, E, rw, cW, op=0.85, stroke=0)
        lq = tag("180° − α", 22, YELLOW_B)
        _angle_label(lq, H, Ff, E, rw + 0.2, segs=segs0 + _arc_segs(H, rw, 0, TAU),
                     pad=0.05, avoid=[lal, lA])
        self.add(quad)
        self.bring_to_back(quad)
        self.play(FadeIn(quad), Create(arcA), FadeIn(lal), run_time=0.8)
        self.play(FadeIn(wH), FadeIn(lq), run_time=0.7)
        self.hold(0.5)

        # half a turn about H: the vertical angle BHC is 180° − α too
        wH2 = wH.copy()
        self.add(wH2)
        self.play(Rotate(wH2, angle=PI, about_point=H), FadeOut(quad), run_time=1.3)
        check(_sample_close(wH2, _wedge(H, B, C, rw, cW, op=0.85, stroke=0), 1e-6),
              "the half-turn lands on ∠BHC")
        lq2 = tag("180° − α", 22, YELLOW_B)
        _angle_label(lq2, H, B, C, rw + 0.2, segs=segs0, pad=0.05, avoid=[lH])
        self.play(FadeIn(lq2), run_time=0.5)
        self.hold(0.3)

        # fold the triangle BHC over BC: H lands on Hₐ
        tBHC = VGroup(mk([B, H, C], TEAL_D, op=0.55, stroke_color=TEAL_B, stroke_width=3),
                      wH2.copy())
        self.play(FadeIn(tBHC[0]), FadeOut(lq2), run_time=0.5)
        self.add(tBHC)
        self.remove(wH2)
        self.play(_fold(tBHC, B, C), run_time=1.8)
        check(_same_poly(tBHC[0].get_vertices(), [B, Ha, C], 1e-6), "BHC folds onto BHₐC")
        check(_sample_close(tBHC[1], _wedge(Ha, B, C, rw, cW, op=0.85, stroke=0), 1e-6),
              "and its angle at H onto ∠BHₐC")
        dHa = Dot(Ha, radius=0.06)
        lHa = _hsub("a", 28)
        segs1 = segs0 + _segs(B, Ha, C)
        _park(lHa, Ha, [B, C], segs1 + _circle_segs(Oc, R), d0=0.25, pad=0.07)
        lq3 = tag("180° − α", 22, YELLOW_B)
        _angle_label(lq3, Ha, B, C, rw + 0.2, segs=segs1, pad=0.05, avoid=[lHa])
        self.play(FadeIn(dHa), FadeIn(lHa), FadeIn(lq3), run_time=0.6)

        # opposite angles α and 180° − α: A, B, Hₐ, C on one circle
        circ = Circle(radius=R, color=YELLOW_B, stroke_width=3).move_to(Oc)
        rd = _trow([("α", YELLOW_B), ("+", WHITE), ("(180° − α)", YELLOW_B),
                    ("=", WHITE), ("180°", WHITE)], 30, 0.16)
        rd.move_to(np.array([4.15, 1.6, 0.0]))
        self.play(FadeIn(rd), run_time=0.7)
        self.play(Create(circ), run_time=1.4)
        self.hold(0.4)

        # the same for the other two sides
        sHA = Line(H, A, color=cAlt, stroke_width=3)
        self.play(Create(sHA), FadeOut(lq), FadeOut(wH), run_time=0.6)
        tCHA = mk([C, H, A], GREEN_D, op=0.55, stroke_color=GREEN_B, stroke_width=3)
        tAHB = mk([A, H, B], BLUE_D, op=0.55, stroke_color=BLUE_B, stroke_width=3)
        self.play(FadeIn(tCHA), FadeIn(tAHB), run_time=0.5)
        self.play(_fold(tCHA, C, A), _fold(tAHB, A, B), run_time=1.8)
        check(_same_poly(tCHA.get_vertices(), [C, Hb, A], 1e-6), "CHA folds onto CH_bA")
        check(_same_poly(tAHB.get_vertices(), [A, Hc, B], 1e-6), "AHB folds onto AH_cB")
        dHb, dHc = Dot(Hb, radius=0.06), Dot(Hc, radius=0.06)
        lHb, lHc = _hsub("b", 28), _hsub("c", 28)
        segs2 = (segs1 + _segs(C, Hb, A) + _segs(A, Hc, B) + _segs(H, A)
                 + _circle_segs(Oc, R) + _ra_segs(E, B, A, 0.17) + _ra_segs(Ff, C, A, 0.17)
                 + _angle_segs(A, B, C, 0.5))
        _park(lHb, Hb, [C, A, Oc], segs2, d0=0.25, pad=0.07, avoid=[lA, lC])
        _park(lHc, Hc, [A, B, Oc], segs2, d0=0.25, pad=0.07, avoid=[lA, lB])
        self.play(FadeIn(dHb), FadeIn(dHc), FadeIn(lHb), FadeIn(lHc), run_time=0.6)

        labels = [lA, lB, lC, lH, lal, lHa, lq3, lHb, lHc]
        _labels_ok(labels, segs2, "F62 labels")
        for l_ in labels:
            check(_clear_of_poly(l_, _wedge_poly(Ha, B, C, rw)),
                  f"F62: '{_name(l_)}' clear of the wedge at Hₐ")
        for l_ in (lA, lB, lC, lH, lal, lq):
            check(_clear_of_poly(l_, _wedge_poly(H, Ff, E, rw))
                  and _clear_of_poly(l_, _wedge_poly(H, B, C, rw)),
                  f"F62: '{_name(l_)}' clear of the wedges at H")
        check(rd.get_left()[0] > circ.get_right()[0] + 0.4 and _in_frame(rd),
              "the readout clear of the circle")
        cap = caption("Reflected in the sides, H lands on the circumcircle", 32)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J20

def _dim_line(p, q, off, color=GREY_A, tick=0.09, width=2):
    """A dimension line parallel to pq, shifted by the vector off, with end
    ticks."""
    p, q, off = to3(p), to3(q), to3(off)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    a, b = p + off, q + off
    return VGroup(Line(a, b, color=color, stroke_width=width),
                  Line(a - tick * nrm, a + tick * nrm, color=color, stroke_width=width),
                  Line(b - tick * nrm, b + tick * nrm, color=color, stroke_width=width))


class J20_RationalPoints(Board):
    """A line through Q = (−1, 0) with rational slope n/m meets the unit
    circle again at P. QR is a diameter, so the angle at P is right; drop
    the height PH. The small step (m across, n up) fits along QP, and the
    same step turned a quarter fits at P along PR: QH : HP = HP : HR = m : n,
    so QH : HP : HR = m² : mn : n². In units where HP = 2mn: QH = 2m²,
    HR = 2n², the radius is m² + n² and OH = m² − n². Divided by the radius,
    P = ((m² − n²)/(m² + n²), 2mn/(m² + n²)) is a rational point; as it is,
    the right triangle OHP is the triple."""

    def construct(self):
        F = Frame(-1.3, 1.3, -1.16, 1.16, max_w=7.4,
                  centre=(-2.55, (SAFE_TOP + SAFE_BOTTOM) / 2))
        O, Q, Rr = F.P((0, 0)), F.P((-1, 0)), F.P((1, 0))
        r = F.k
        n_ = np.linalg.norm

        def Pof(t):
            return F.P(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))

        # rational slopes give rational points: the examples, checked exactly
        from fractions import Fraction as Fq
        ex = [(2, 1), (3, 2), (4, 1)]                     # (m, n): slope n/m
        for m, k in ex:
            t = Fq(k, m)
            x, y = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
            check(x * x + y * y == 1 and y == t * (x + 1), "rational point on the circle and line")
            check((x, y) == (Fq(m * m - k * k, m * m + k * k), Fq(2 * m * k, m * m + k * k)),
                  "P = ((m² − n²)/(m² + n²), 2mn/(m² + n²))")
            check((m * m - k * k) ** 2 + (2 * m * k) ** 2 == (m * m + k * k) ** 2, "the triple")

        axis = Line(F.P((-1.24, 0)), F.P((1.24, 0)), color=GREY_B, stroke_width=2)
        circ = Circle(radius=r, color=WHITE, stroke_width=3).move_to(O)
        dQ, dO = Dot(Q, radius=0.07), Dot(O, radius=0.05, color=GREY_B)
        lQ = _park_dirs(tag("(−1, 0)", 24), Q, [214, 222, 206, 230],
                        _circle_segs(O, r) + _segs(F.P((-1.24, 0)), F.P((1.24, 0))),
                        d0=0.3, d1=1.2, pad=0.1)
        self.play(Create(circ), Create(axis), FadeIn(dO), run_time=1.1)
        self.play(FadeIn(dQ), FadeIn(lQ), run_time=0.5)

        th = ValueTracker(float(np.arctan(0.5)))

        def Pt():
            return Pof(np.tan(th.get_value()))

        def ray():
            P = Pt()
            d = _unit(P - Q)
            return Line(Q, P + 0.55 * d, color=YELLOW_B, stroke_width=4)

        ray_m = always_redraw(ray)
        dP = always_redraw(lambda: Dot(Pt(), radius=0.08, color=YELLOW_B))
        self.play(Create(ray_m), run_time=0.8)
        self.add(dP)

        # the readout: slope -> point
        rows = [tag("1/2  →  (3/5, 4/5)", 25), tag("2/3  →  (5/13, 12/13)", 25),
                tag("1/4  →  (15/17, 8/17)", 25)]
        x0 = 0.95
        for i, row in enumerate(rows):
            row.move_to(np.array([x0 + row.width / 2, 2.55 - 0.72 * i, 0.0]))
        self.play(FadeIn(rows[0]), Flash(Pt(), color=YELLOW_B, line_length=0.18), run_time=0.7)
        for (m, k), row in zip(ex[1:], rows[1:]):
            self.play(th.animate.set_value(float(np.arctan(k / m))), run_time=1.1)
            self.play(FadeIn(row), Flash(Pt(), color=YELLOW_B, line_length=0.18), run_time=0.6)
        self.play(th.animate.set_value(float(np.arctan(0.5))), run_time=1.0)
        ray_m.clear_updaters()
        dP.clear_updaters()
        P = Pof(0.5)
        check(close(Pt(), P), "back at slope 1/2")
        self.hold(0.3)

        # ---- why: Thales and the step m across, n up
        H = F.P((0.6, 0.0))
        check(close(_foot(P, Q, Rr), H) and abs(_angle(P, Q, Rr) - PI / 2) < 1e-9,
              "P sees the diameter QR at a right angle; H is the foot")
        sPR = Line(P, Rr, color=WHITE, stroke_width=3)
        raP = _ra(P, Q, Rr, 0.2)
        sPH = DashedLine(P, H, color=WHITE, stroke_width=3, dash_length=0.09)
        raH = _ra(H, P, Rr, 0.17)
        self.play(Create(sPR), Create(raP), FadeOut(lQ), run_time=0.8)
        self.play(Create(sPH), Create(raH), run_time=0.7)

        m0, n0 = 2, 1                      # the slope drawn is n/m = 1/2
        u = 0.6                            # screen length of one step unit
        S = Q + 0.33 * (P - Q)             # where the step sits on QP
        S2, S3 = S + np.array([m0 * u, 0, 0]), S + np.array([m0 * u, n0 * u, 0])
        check(_on_line(S3, Q, P), "the step m across, n up runs along QP")
        stepA = mk([S, S2, S3], TEAL_D, op=0.85, stroke_color=TEAL_B, stroke_width=2)
        lm1 = tag("m", 26, TEAL_B).next_to(Line(S, S2), DOWN, buff=0.1)
        ln1 = tag("n", 26, TEAL_B).next_to(Line(S2, S3), RIGHT, buff=0.1)
        self.play(FadeIn(stepA), FadeIn(lm1), FadeIn(ln1), run_time=0.7)
        stepB = stepA.copy()
        self.add(stepB)
        self.play(_glide(stepB, S, P, -PI / 2), run_time=1.6)
        T2, T3 = P + np.array([0, -m0 * u, 0]), P + np.array([n0 * u, -m0 * u, 0])
        check(_same_poly(stepB.get_vertices(), [P, T2, T3], 1e-6), "the turned step sits at P")
        check(_on_line(T2, P, H) and _on_line(T3, P, Rr), "along PH and PR")
        lm2 = tag("m", 26, TEAL_B)
        lm2.move_to(P + 0.7 * (T2 - P) + np.array([-0.1 - lm2.width / 2, 0.0, 0.0]))
        ln2 = tag("n", 26, TEAL_B).next_to(Line(T2, T3), DOWN, buff=0.1)
        self.play(FadeIn(lm2), FadeIn(ln2), run_time=0.5)
        self.hold(0.4)

        # ---- lengths in units where HP = 2mn
        check(abs(n_(H - Q) / n_(P - H) - m0 / n0) < 1e-9
              and abs(n_(P - H) / n_(Rr - H) - m0 / n0) < 1e-9, "QH : HP = HP : HR = m : n")
        qh, hp, hr = 2 * m0 * m0, 2 * m0 * n0, 2 * n0 * n0
        check(abs(n_(H - Q) / n_(P - H) - qh / hp) < 1e-9 and
              abs(n_(Rr - H) / n_(P - H) - hr / hp) < 1e-9, "2m² : 2mn : 2n²")
        check(abs(n_(O - Q) / n_(P - H) - (m0 * m0 + n0 * n0) / hp) < 1e-9 and
              abs(n_(H - O) / n_(P - H) - (m0 * m0 - n0 * n0) / hp) < 1e-9,
              "radius m² + n², OH = m² − n²")
        dy1, dy2 = np.array([0.0, -0.36, 0.0]), np.array([0.0, -1.06, 0.0])
        dimQH = _dim_line(Q, H, dy2)                 # QH, under everything
        dimHR = _dim_line(H, Rr, dy1)
        dimOH = _dim_line(O, H, dy1)                 # OH, right under the triangle
        t2mn = tag("2mn", 28, YELLOW_B)
        t2m2 = tag("2m²", 28, YELLOW_B)
        t2n2 = tag("2n²", 28, YELLOW_B)
        t2m2.next_to(dimQH[0], DOWN, buff=0.1)
        t2n2.next_to(dimHR[0], DOWN, buff=0.1)
        _beside(t2mn, H, P, LEFT, 0.12, at=0.3)
        self.play(FadeIn(t2mn), run_time=0.5)
        self.play(Create(dimQH), Create(dimHR), FadeIn(t2m2), FadeIn(t2n2), run_time=0.9)

        # the radius m² + n², and OH = 2m² − (m² + n²) = m² − n²
        sOP = Line(O, P, color=ORANGE, stroke_width=5)
        tri = mk([O, H, P], ORANGE, op=0.35, stroke_width=0)
        tR = tag("m² + n²", 28, ORANGE)
        angR = _along(tR, O, P, Q - O, gap=0.1, at=0.35)
        tOH = tag("m² − n²", 28, ORANGE).next_to(dimOH[0], DOWN, buff=0.1)
        self.add(tri)
        self.bring_to_back(tri)
        self.play(Create(sOP), FadeIn(tri), FadeIn(tR), run_time=0.9)
        self.play(Create(dimOH), FadeIn(tOH), run_time=0.7)
        self.hold(0.4)

        # the examples become triples
        trip = [tag("3, 4, 5", 25, ORANGE), tag("5, 12, 13", 25, ORANGE),
                tag("15, 8, 17", 25, ORANGE)]
        arr = []
        xa = max(rw_.get_right()[0] for rw_ in rows) + 0.3
        for row, tr in zip(rows, trip):
            a = tag("→", 25)
            a.move_to(np.array([xa + a.width / 2, row.get_y(), 0.0]))
            tr.move_to(np.array([a.get_right()[0] + 0.3 + tr.width / 2, row.get_y(), 0.0]))
            arr.append(a)
        self.play(*[FadeIn(VGroup(a, tr), shift=RIGHT * 0.15) for a, tr in zip(arr, trip)],
                  run_time=0.9)

        segs = (_segs(Q, P) + _segs(P, Rr) + _segs(P, H) + _segs(O, P)
                + _segs(F.P((-1.24, 0)), F.P((1.24, 0))) + _circle_segs(O, r)
                + _ra_segs(P, Q, Rr, 0.2) + _ra_segs(H, P, Rr, 0.17)
                + _mob_segs(dimQH) + _mob_segs(dimHR) + _mob_segs(dimOH)
                + _segs(S, S2, S3, closed=True)
                + _segs(P, T2, T3, closed=True)
                + _segs(P, P + 0.55 * _unit(P - Q)))
        _labels_ok([lm1, ln1, lm2, ln2, t2mn, t2m2, t2n2, tOH], segs, "J20 figure")
        check(_clear_rot(tR, angR, segs) and _in_frame(tR), "J20: m² + n² along OP, clear")
        for l_ in (lm1, ln1, lm2, ln2, t2mn, t2m2, t2n2, tOH):
            check(_clear_rot(tR, angR, _box_segs(l_), 0.05),
                  f"J20: m² + n² clear of '{_name(l_)}'")
        _labels_ok(rows + arr + trip, [], "J20 readout")
        check(min(rw_.get_left()[0] for rw_ in rows) > circ.get_right()[0] + 0.5,
              "readout clear of the circle")
        cap = caption("(m² − n²)² + (2mn)² = (m² + n²)²", 34)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap, rotated=[(tR, angR)])
        self.hold(2.2)


# ===================================================================== J26

def _free_dir(X, towards):
    """Unit vector at X along the middle of the widest angular gap between
    the directions to the points `towards` (to park a label)."""
    X = to3(X)
    angs = sorted(_dir(X, q) % TAU for q in towards)
    best, mid = -1.0, 0.0
    for i, a0 in enumerate(angs):
        a1 = angs[(i + 1) % len(angs)] + (TAU if i == len(angs) - 1 else 0.0)
        if a1 - a0 > best:
            best, mid = a1 - a0, (a0 + a1) / 2
    return _polar(mid)


class J26_EveryTriangleIsosceles(Board):
    """A caution. The false proof: let the bisector of A and the
    perpendicular bisector of BC meet at O inside; drop OF ⊥ AB, OE ⊥ AC.
    Then AF = AE and OF = OE (the bisector), OB = OC (the perpendicular
    bisector), so FB = EC, and AB = AF + FB = AE + EC = AC. Every step but
    the figure is right: the two lines meet at the midpoint of the arc BC,
    on the circumcircle, outside the triangle. There the foot on the
    shorter side falls beyond its end: AB = AF − FB while AC = AE + EC,
    and AC − AB = 2·FB."""

    def construct(self):
        Bm, Cm, Am = np.array([-2.4, 0.0]), np.array([2.4, 0.0]), np.array([-0.9, 4.0])
        Fr = Frame(-3.45, 3.15, -2.12, 4.64, max_w=7.2,
                   centre=(-2.95, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = Fr.P(Am), Fr.P(Bm), Fr.P(Cm)
        D = (B + C) / 2
        n = np.linalg.norm
        ub = _unit(B - A) + _unit(C - A)                  # the true bisector
        Ot = _meet(A, A + ub, D, D + _perp(C - B))        # where it really meets
        Oc = _circumcentre(A, B, C)
        Rc = n(A - Oc)
        Of = D + 0.8 * Fr.k * _unit(_perp(C - B))         # the false point, drawn inside
        if np.dot(Of - D, A - D) < 0:
            Of = 2 * D - Of

        def Op(s_):
            return Of + (Ot - Of) * s_

        def feet(s_):
            O = Op(s_)
            return _foot(O, A, B), _foot(O, A, C)

        # the false figure is false; the true figure, checked
        Ff0, Ef0 = feet(0.0)
        check(_pip(Of, [A, B, C]) and 0 < _param(Ff0, A, B) < 1 and 0 < _param(Ef0, A, C) < 1,
              "the false figure: O inside, both feet on the sides")
        check(abs(_angle(A, B, Of) - _angle(A, Of, C)) > 5 * DEGREES,
              "(and its 'bisector' is not one)")
        Ft, Et = feet(1.0)
        check(abs(_angle(A, B, Ot) - _angle(A, Ot, C)) < 1e-9, "the true bisector")
        check(abs(n(Ot - Oc) - Rc) < 1e-9, "the true O lies on the circumcircle")
        check(_cross(C - B, A - B) * _cross(C - B, Ot - B) < 0, "below BC, outside the triangle")
        check(abs(n(Ft - A) - n(Et - A)) < 1e-9 and abs(n(Ft - Ot) - n(Et - Ot)) < 1e-9,
              "AF = AE, OF = OE")
        check(abs(n(Ot - B) - n(Ot - C)) < 1e-9 and abs(n(Ft - B) - n(Et - C)) < 1e-9,
              "OB = OC, FB = EC")
        check(_param(Ft, A, B) > 1 and 0 < _param(Et, A, C) < 1,
              "F falls beyond B, E stays on AC")
        check(abs(n(B - A) - (n(Ft - A) - n(Ft - B))) < 1e-9
              and abs(n(C - A) - (n(Et - A) + n(Et - C))) < 1e-9,
              "AB = AF − FB, AC = AE + EC")
        check(abs(n(C - A) - n(B - A) - 2 * n(Ft - B)) < 1e-9, "AC − AB = 2·FB")
        check(n(Ft - B) > 0.42, "FB visible")

        cBl, cRd = BLUE_B, RED_B
        s = ValueTracker(0.0)
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        lA = tag("A", 30).move_to(A + 0.36 * UP)
        lB = tag("B", 30).move_to(B + 0.42 * _polar(160 * DEGREES))
        lC = tag("C", 30).move_to(C + np.array([0.4, -0.1, 0.0]))
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        # ---- the false figure, as the argument wants it
        def pb():
            O = Op(s.get_value())
            return VGroup(Line(D, O, color=GREEN_B, stroke_width=3),
                          _ra(D, C, O, 0.18, GREEN_B))

        def bis():
            O = Op(s.get_value())
            return Line(A, O, color=YELLOW_B, stroke_width=3)

        def arcsA():
            O = Op(s.get_value())
            return VGroup(angle_arc(A, B, O, 0.62, YELLOW_B, 4),
                          angle_arc(A, O, C, 0.7, YELLOW_B, 4))

        def perps():
            O = Op(s.get_value())
            Fx, Ex = feet(s.get_value())
            g = VGroup(Line(O, Fx, color=WHITE, stroke_width=3),
                       Line(O, Ex, color=WHITE, stroke_width=3),
                       _ra(Fx, O, A, 0.16), _ra(Ex, O, A, 0.16),
                       DashedLine(O, B, color=GREY_B, stroke_width=2.5, dash_length=0.08),
                       DashedLine(O, C, color=GREY_B, stroke_width=2.5, dash_length=0.08))
            if _param(Fx, A, B) > 1:                     # AB, extended past B
                g.add(DashedLine(B, Fx, color=WHITE, stroke_width=2.5, dash_length=0.07))
            return g

        def dotO():
            return Dot(Op(s.get_value()), radius=0.07, color=YELLOW_B)

        def hi():
            Fx, Ex = feet(s.get_value())
            return VGroup(Line(A, Fx, color=cBl, stroke_width=8).set_opacity(0.85),
                          Line(A, Ex, color=cBl, stroke_width=8).set_opacity(0.85),
                          Line(Fx, B, color=cRd, stroke_width=8),
                          Line(Ex, C, color=cRd, stroke_width=8))

        def labels_at(s_):
            O = Op(s_)
            Fx, Ex = feet(s_)
            lO_ = _park(tag("O", 28), O, [A, B, C, D, Fx, Ex], segs_at(s_)
                        + _circle_segs(Oc, Rc), d0=0.3, d1=1.0, pad=0.07,
                        avoid=[lA, lB, lC])
            vF = _unit(_perp(B - A))
            if np.dot(vF, C - A) > 0:
                vF = -vF
            dF = 0.38 if s_ < 0.5 else 0.62              # (outside the ring at the end)
            lF_ = tag("F", 28).move_to(Fx + dF * _unit(vF + 0.35 * _unit(B - A)))
            vE = _unit(_perp(C - A))
            if np.dot(vE, B - A) > 0:
                vE = -vE
            lE_ = tag("E", 28).move_to(Ex + 0.36 * vE)
            return lO_, lF_, lE_

        def segs_at(s_):
            O = Op(s_)
            Fx, Ex = feet(s_)
            sg = (_segs(A, B, C, closed=True) + _segs(D, O) + _segs(A, O) + _segs(O, Fx)
                  + _segs(O, Ex) + _segs(O, B) + _segs(O, C) + _ra_segs(D, C, O, 0.18)
                  + _ra_segs(Fx, O, A, 0.16) + _ra_segs(Ex, O, A, 0.16)
                  + _angle_segs(A, B, O, 0.62) + _angle_segs(A, O, C, 0.7))
            if _param(Fx, A, B) > 1:
                sg += _segs(B, Fx)
            return sg

        pb_m, bis_m, arcs_m = always_redraw(pb), always_redraw(bis), always_redraw(arcsA)
        tkD = VGroup(_ticks(B, D, 1, GREEN_B), _ticks(D, C, 1, GREEN_B))
        self.play(Create(pb_m), FadeIn(tkD), run_time=0.9)
        self.play(Create(bis_m), Create(arcs_m), run_time=0.9)
        lO, lF, lE = labels_at(0.0)
        _labels_ok([lA, lB, lC, lO, lF, lE], segs_at(0.0) + _mob_segs(tkD),
                   "J26 false figure")
        dO = always_redraw(dotO)
        self.play(FadeIn(dO), FadeIn(lO), run_time=0.4)
        perps_m = always_redraw(perps)
        self.play(Create(perps_m), FadeIn(lF), FadeIn(lE), run_time=1.0)
        hi_m = always_redraw(hi)
        self.add(hi_m)
        self.bring_to_back(hi_m)
        self.play(FadeIn(hi_m), run_time=0.7)
        self.bring_to_front(dO)

        # the claimed chain
        def row(parts, y, size=30):
            g = _trow(parts, size, 0.14)
            return g.move_to(np.array([1.25 + g.width / 2, y, 0.0]))

        W = WHITE
        r1 = row([("AF", cBl), ("=", W), ("AE", cBl)], 2.7)
        r2 = row([("FB", cRd), ("=", W), ("EC", cRd)], 1.95)
        r3 = row([("AB", W), ("=", W), ("AF", cBl), ("+", W), ("FB", cRd)], 1.2)
        r4 = row([("AC", W), ("=", W), ("AE", cBl), ("+", W), ("EC", cRd)], 0.45)
        r5 = row([("AB = AC ?", RED_B)], -0.55, 34)
        self.play(FadeIn(r1), FadeIn(r2), run_time=0.8)
        self.play(FadeIn(r3), FadeIn(r4), run_time=0.8)
        self.play(FadeIn(r5), run_time=0.6)
        self.hold(0.8)

        # ---- the true figure: O slides to where the lines really meet
        self.play(FadeOut(lO), FadeOut(lF), FadeOut(lE), run_time=0.4)
        self.play(s.animate.set_value(1.0), run_time=3.4, rate_func=smooth)
        for m in (pb_m, bis_m, arcs_m, dO, perps_m, hi_m):
            m.clear_updaters()
        lO, lF, lE = labels_at(1.0)
        circ = Circle(radius=Rc, color=GREY_B, stroke_width=3).move_to(Oc)
        self.add(circ)
        self.bring_to_back(circ)
        self.play(FadeIn(lO), FadeIn(lF), FadeIn(lE), Create(circ), run_time=1.3)
        # where it fails: F is off the side AB
        ring = Circle(radius=0.3, color=RED_B, stroke_width=3).move_to(Ft)
        self.play(Create(ring), run_time=0.5)
        r3b = row([("AB", W), ("=", W), ("AF", cBl), ("−", YELLOW_B), ("FB", cRd)], 1.2)
        r5b = row([("AC − AB = 2·FB", YELLOW_B)], -0.55, 34)
        self.play(FadeOut(r3), FadeIn(r3b), run_time=0.8)
        self.play(FadeOut(r5), FadeIn(r5b), run_time=0.8)

        # checks on the closing frame
        labels = [lA, lB, lC, lO, lF, lE]
        segs = (segs_at(1.0) + _circle_segs(Oc, Rc) + _mob_segs(tkD)
                + _circle_segs(Ft, 0.3, 48))
        _labels_ok(labels, segs, "J26 true figure")
        _labels_ok([r1, r2, r3b, r4, r5b], [], "J26 readout")
        check(r1.get_left()[0] > circ.get_right()[0] + 0.3, "readout clear of the figure")
        cap = caption("The lines meet outside, on the circumcircle: AB = AF − FB, "
                      "but AC = AE + EC", 28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J12

class J12_GoldenIrrational(Board):
    """Suppose φ = a/b with whole numbers a > b. Cut the b × b square off
    the a × b rectangle: the rest, b × (a − b), is the same shape again (a
    quarter-turn and a shrink by 1/φ about one point carries the whole onto
    it), and its sides are whole numbers, smaller. Repeat: b > a − b >
    2b − a > 2a − 3b > … would be whole numbers decreasing for ever, which
    is impossible. So φ is irrational."""

    def construct(self):
        h = 4.0                                   # the short side, b, on screen
        o = np.array([-6.05, -1.7, 0.0])           # bottom-left corner
        lam = 1 / PHI

        # the spiral similarity S(z) = −i z/φ + (1 + i), in units of b
        def S(p):
            x, y = float(p[0]), float(p[1])
            return np.array([lam * y + 1.0, -lam * x + 1.0])

        c = np.array([(1 + lam) / (1 + lam * lam), (1 - lam) / (1 + lam * lam)])
        check(close(S(c), c), "the centre of the spiral is fixed")

        def P(p):
            return o + h * np.array([p[0], p[1], 0.0])

        R0 = [np.array(v, float) for v in ((0, 0), (PHI, 0), (PHI, 1), (0, 1))]
        Q0 = [np.array(v, float) for v in ((0, 0), (1, 0), (1, 1), (0, 1))]
        K = 11
        Rs, Qs = [R0], [Q0]
        for _ in range(K):
            Rs.append([S(v) for v in Rs[-1]])
            Qs.append([S(v) for v in Qs[-1]])
        for k in range(K):
            check(_tiles_exactly([Qs[k], Rs[k + 1]], Rs[k], 60),
                  f"rectangle {k} = square + the next rectangle")
            sq = Qs[k]
            sides = [np.linalg.norm(sq[(i + 1) % 4] - sq[i]) for i in range(4)]
            check(max(sides) - min(sides) < 1e-9 and abs(sides[0] - lam ** k) < 1e-9,
                  "a square of side 1/φ^k")
        # the whole-number sides a, b → b, a − b → …: coefficients (p, q) of a, b
        pairs = [((1, 0), (0, 1))]
        for _ in range(4):
            (p1, q1), (p2, q2) = pairs[-1]
            pairs.append(((p2, q2), (p1 - p2, q1 - q2)))
        for k, ((p1, q1), (p2, q2)) in enumerate(pairs):
            check(abs((p1 * PHI + q1) - lam ** (k - 1)) < 1e-9
                  and abs((p2 * PHI + q2) - lam ** k) < 1e-9,
                  "the sides of rectangle k are those combinations of a = φ, b = 1")

        cols = [BLUE_D, TEAL_D, GOLD_D, GREEN_D, PURPLE_B, MAROON_B, ORANGE, BLUE_D,
                TEAL_D, GOLD_D, GREEN_D]
        rect = mk([P(v) for v in R0], GREY_D, op=0.35, stroke_color=WHITE, stroke_width=3)
        la = tag("a", 32).next_to(Line(P(R0[0]), P(R0[1])), DOWN, buff=0.14)
        lb = tag("b", 32).next_to(Line(P(R0[0]), P(R0[3])), LEFT, buff=0.14)
        self.play(FadeIn(rect), FadeIn(la), FadeIn(lb), run_time=1.0)

        # readout: the supposition and the sides, step by step
        sup = tag("φ = a/b ?", 34, RED_B)
        sup.move_to(np.array([4.4, 3.2, 0.0]))
        names = ["a, b", "b, a − b", "a − b, 2b − a", "2b − a, 2a − 3b"]
        rows = [tag(t_, 30) for t_ in names]
        for i, r_ in enumerate(rows):
            r_.move_to(np.array([2.05 + r_.width / 2, 2.25 - 0.66 * i, 0.0]))
        self.play(FadeIn(sup), FadeIn(rows[0]), run_time=0.8)

        sq_m, in_lab = [], []
        inner = ["b", "a − b", "2b − a"]
        for k in range(3):
            sq = mk([P(v) for v in Qs[k]], cols[k], op=0.8, stroke_color=WHITE, stroke_width=3)
            self.play(FadeIn(sq), run_time=0.6)
            sq_m.append(sq)
            # the rest has the same shape: the outline turns a quarter and shrinks onto it
            ghost = Polygon(*[P(v) for v in Rs[k]], stroke_color=YELLOW_B, stroke_width=4)
            self.add(ghost)
            self.play(_spiral(ghost, P(c), -PI / 2, lam), run_time=1.5 if k == 0 else 1.1)
            check(_same_poly(ghost.get_vertices(), [P(v) for v in Rs[k + 1]], 1e-6),
                  f"the outline lands on rectangle {k + 1}")
            lab = tag(inner[k], 32 if k < 2 else 26).move_to(P(np.mean(Qs[k], axis=0)))
            in_lab.append(lab)
            self.play(FadeOut(ghost), FadeIn(lab), FadeIn(rows[k + 1]), run_time=0.6)
        # and so on, for ever
        more = []
        for k in range(3, K):
            more.append(mk([P(v) for v in Qs[k]], cols[k], op=0.8, stroke_color=WHITE,
                           stroke_width=max(0.6, 3 * lam ** (k - 3))))
        self.play(LaggedStart(*[FadeIn(m) for m in more], lag_ratio=0.45), run_time=2.4)
        dots = tag("…", 34).move_to(np.array([2.05 + 0.3, 2.25 - 0.66 * 4, 0.0]))
        chain = tag("a > b > a − b > 2b − a > … > 0", 26, YELLOW_B)
        chain.move_to(np.array([0.95 + chain.width / 2, -1.55, 0.0]))
        self.play(FadeIn(dots), run_time=0.4)
        self.play(FadeIn(chain), run_time=0.8)

        for l_, k in zip(in_lab, range(3)):
            check(all(_pip(q, [P(v) for v in Qs[k]]) for q in
                      (l_.get_corner(UL), l_.get_corner(UR), l_.get_corner(DL),
                       l_.get_corner(DR))), "each side label sits inside its square")
        _labels_ok([la, lb], _segs(*[P(v) for v in R0], closed=True), "J12 sides")
        _labels_ok([sup, *rows, dots, chain], [], "J12 readout")
        check(min(r_.get_left()[0] for r_ in [sup, *rows, chain]) > P(R0[1])[0] + 0.4,
              "readout clear of the rectangle")
        cap = caption("Whole sides would shrink for ever:  φ ≠ a/b,  φ is irrational", 30)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J17

def _pieces_row(parts, size=28, gaps=None):
    """Coloured Text pieces on one baseline; gaps[i] is the space after
    piece i (default 0.14)."""
    g = VGroup()
    x = 0.0
    for i, (s_, col) in enumerate(parts):
        t = _on_base(s_, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + (0.14 if gaps is None else gaps[i])
        g.add(t)
    return g


class J17_SquaresEndingInFive(Board):
    """Cut the square of side 10a + 5 into a 10a-square, two 10a × 5 strips
    and a 5 × 5 corner. A quarter-turn moves the top strip to the far side
    of the other strip: a 10a × 10(a + 1) rectangle, made of a(a + 1)
    blocks of 10 × 10, and the corner 25. Drawn for a = 2 (25²)."""

    def construct(self):
        a = 2
        L, s5 = 10 * a, 5                          # 20 and 5 units
        u = 0.215
        o = np.array([-5.6, -2.12, 0.0])

        def P(x, y):
            return o + u * np.array([x, y, 0.0])

        def box(x0, y0, x1, y1):
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        G = box(0, 0, L, L)
        Rt = box(L, 0, L + s5, L)
        Tp = box(0, L, L, L + s5)
        Kc = box(L, L, L + s5, L + s5)
        Tq = box(L + s5, 0, L + 2 * s5, L)          # where the top strip goes
        check(_tiles_exactly([G, Rt, Tp, Kc], box(0, 0, L + s5, L + s5), 80),
              "the square = big square + two strips + corner")
        check(_tiles_exactly([G, Rt, Tq], box(0, 0, L + 2 * s5, L), 80),
              "big square + strips = the 10a × 10(a + 1) rectangle")
        check((L + s5) ** 2 == L * (L + 2 * s5) + s5 * s5 and
              L * (L + 2 * s5) == 100 * a * (a + 1), "(10a + 5)² = 100a(a + 1) + 25")
        for k in (2, 3, 8):
            check((10 * k + 5) ** 2 == 100 * k * (k + 1) + 25, "the examples")

        cG, cS, cK = BLUE_D, TEAL_D, ORANGE
        mG = mk([P(*v) for v in G], cG, op=0.8, stroke_width=2)
        mR = mk([P(*v) for v in Rt], cS, op=0.8, stroke_width=2)
        mT = mk([P(*v) for v in Tp], cS, op=0.8, stroke_width=2)
        mK = mk([P(*v) for v in Kc], cK, op=0.85, stroke_width=2)
        self.play(LaggedStart(FadeIn(mG), FadeIn(mR), FadeIn(mT), FadeIn(mK),
                              lag_ratio=0.25), run_time=1.3)

        # side lengths: 10a and 5 along two sides
        dy, dx = np.array([0, -0.22, 0]), np.array([-0.22, 0, 0])
        dB1, dB2 = _dim_line(P(0, 0), P(L, 0), dy), _dim_line(P(L, 0), P(L + s5, 0), dy)
        dL1, dL2 = _dim_line(P(0, 0), P(0, L), dx), _dim_line(P(0, L), P(0, L + s5), dx)
        tB1 = tag("10a", 28).next_to(dB1[0], DOWN, buff=0.1)
        tB2 = tag("5", 28).next_to(dB2[0], DOWN, buff=0.1)
        tL1 = tag("10a", 28).next_to(dL1[0], LEFT, buff=0.1)
        tL2 = tag("5", 28).next_to(dL2[0], LEFT, buff=0.1)
        self.play(*[Create(d) for d in (dB1, dB2, dL1, dL2)],
                  *[FadeIn(t_) for t_ in (tB1, tB2, tL1, tL2)], run_time=0.9)
        self.hold(0.5)

        # the top strip makes a quarter-turn to the far side
        self.play(mT.animate.set_stroke(YELLOW_B, 4), run_time=0.4)
        c0 = np.mean([P(*v) for v in Tp], axis=0)
        c1 = np.mean([P(*v) for v in Tq], axis=0)
        self.play(_glide(mT, c0, c1, -PI / 2), FadeOut(dL2), FadeOut(tL2),
                  run_time=1.8)
        check(_same_poly(mT.get_vertices(), [P(*v) for v in Tq], 1e-6),
              "the strip lands beside the other strip")
        # one rectangle now: its outline replaces the edges inside it
        outline = Polygon(*[P(*v) for v in box(0, 0, L + 2 * s5, L)],
                          stroke_color=WHITE, stroke_width=2.5)
        self.play(mT.animate.set_stroke(cS, 0), mR.animate.set_stroke(cS, 0),
                  mG.animate.set_stroke(cG, 0), Create(outline), run_time=0.6)
        # the two strips, side by side, as one 10 × 10a piece (no seam between them)
        check(_tiles_exactly([Rt, Tq], box(L, 0, L + 2 * s5, L), 60), "the two strips: 10 × 10a")
        strips = mk([P(*v) for v in box(L, 0, L + 2 * s5, L)], cS, op=0.8, stroke_width=0)
        self.add(strips)
        self.remove(mR, mT)
        self.bring_to_front(outline)
        dBw = _dim_line(P(0, 0), P(L + 2 * s5, 0), dy)
        tBw = tag("10(a + 1)", 28).next_to(dBw[0], DOWN, buff=0.1)
        self.play(FadeOut(dB1), FadeOut(dB2), FadeOut(tB1), FadeOut(tB2),
                  Create(dBw), FadeIn(tBw), run_time=0.8)

        # a(a + 1) blocks of 10 × 10, and the corner 25
        grid = VGroup(*[DashedLine(P(x, 0), P(x, L), color=WHITE, stroke_width=2,
                                   dash_length=0.08) for x in range(10, L + 2 * s5, 10)],
                      *[DashedLine(P(0, y), P(L + 2 * s5, y), color=WHITE, stroke_width=2,
                                   dash_length=0.08) for y in range(10, L, 10)])
        blocks = VGroup(*[tag("100", 26).move_to(P(x + 5, y + 5))
                          for x in range(0, L + 2 * s5, 10) for y in range(0, L, 10)])
        check(len(blocks) == a * (a + 1), "a(a + 1) blocks")
        t25 = tag("25", 26).move_to(P(L + 2.5, L + 2.5))
        self.play(Create(grid), run_time=0.8)
        self.play(FadeIn(blocks), FadeIn(t25), run_time=0.8)
        self.hold(0.4)

        # the rule, on examples: a(a + 1), then 25
        W = WHITE
        ex = [(2, "25", "6"), (3, "35", "12"), (8, "85", "72")]
        rows = VGroup()
        for k, sq, pre in ex:
            r_ = _pieces_row([(f"{k} · {k + 1} = {pre}", BLUE_B), ("→", W),
                              (f"{sq}² =", W), (pre, BLUE_B), ("25", ORANGE)], 30,
                             gaps=[0.3, 0.3, 0.16, 0.015, 0.0])
            check(int(pre + "25") == int(sq) ** 2, "the example is right")
            rows.add(r_)
        for i, r_ in enumerate(rows):
            r_.move_to(np.array([1.5 + r_.width / 2, 1.9 - 0.85 * i, 0.0]))
        self.play(LaggedStart(*[FadeIn(r_, shift=RIGHT * 0.15) for r_ in rows],
                              lag_ratio=0.4), run_time=1.6)

        figure = VGroup(mG, strips, mK)
        check(rows.get_left()[0] > figure.get_right()[0] + 0.5, "readout clear of the figure")
        _labels_ok([tL1, tBw, t25, *blocks], _mob_segs(dL1) + _mob_segs(dBw)
                   + _mob_segs(grid) + _segs(*[P(*v) for v in box(0, 0, L + 2 * s5, L)],
                                             closed=True)
                   + _segs(*[P(*v) for v in Kc], closed=True), "J17 labels")
        check(all(m_.get_stroke_width() == 0 for m_ in (mG, strips)),
              "no piece edges left inside the rectangle")
        _labels_ok(list(rows), [], "J17 readout")
        cap = _cap_parts([("(10a + 5)² = 100 · a(a + 1) + 25", YELLOW_B)])
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)


def _cap_parts(parts, size=34):
    """A caption made of coloured pieces, placed where caption() puts its
    Text."""
    g = _pieces_row(parts, size)
    if g.width > 13.4:
        g.scale_to_fit_width(13.4)
    g.set_x(0.0)
    g.to_edge(DOWN, buff=0.3)
    return g


# ===================================================================== J23

def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


class J23_PrimesBeyondThree(Board):
    """Write the numbers in rows of six, in the columns 6k − 1, 6k, 6k + 1,
    6k + 2, 6k + 3, 6k + 4. Four columns hold only multiples of 2 or 3:
    6k = 2·3k, 6k + 2 = 2(3k + 1), 6k + 3 = 3(2k + 1), 6k + 4 = 2(3k + 2).
    Their only primes are 2 and 3 themselves, so every prime above 3 sits
    in one of the two columns 6k ± 1 (which also hold composites: 25, 35,
    49)."""

    def construct(self):
        heads = ["6k − 1", "6k", "6k + 1", "6k + 2", "6k + 3", "6k + 4"]
        offs = [-1, 0, 1, 2, 3, 4]
        K = 8
        xs = [-5.0 + 2.0 * j for j in range(6)]
        y_head, y0, dy = 3.38, 2.76, 0.58

        def cell(k, j):
            return np.array([xs[j], y0 - dy * k, 0.0])

        nums = {}
        for k in range(K + 1):
            for j, o in enumerate(offs):
                v = 6 * k + o
                if v >= 1:
                    nums[(k, j)] = v
        primes = [v for v in nums.values() if _is_prime(v)]
        check(sorted(nums.values()) == list(range(1, 6 * K + 5)), "every number once, in order")
        for (k, j), v in nums.items():
            if offs[j] in (0, 2, 4):
                check(v % 2 == 0, "6k, 6k + 2, 6k + 4 are even")
            if offs[j] == 3:
                check(v % 3 == 0, "6k + 3 is a multiple of 3")
            if _is_prime(v) and v > 3:
                check(offs[j] in (-1, 1), "a prime above 3 sits in a column 6k ± 1")
        check(set(v for v in primes if v <= 3) == {2, 3}, "2 and 3: the exceptions")

        hd = VGroup(*[tag(h_, 26, GREY_A).move_to(np.array([xs[j], y_head, 0.0]))
                      for j, h_ in enumerate(heads)])
        rule = Line(np.array([-6.2, y_head - 0.32, 0]), np.array([6.2, y_head - 0.32, 0]),
                    color=GREY_B, stroke_width=2)
        tx = {kj: tag(str(v), 28).move_to(cell(*kj)) for kj, v in nums.items()}
        self.play(FadeIn(hd), Create(rule), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(VGroup(*[tx[(k, j)] for j in range(6) if (k, j) in tx]))
                                for k in range(K + 1)], lag_ratio=0.25), run_time=2.2)
        self.hold(0.3)

        # four columns hold only multiples of 2 or 3
        top, bot = y0 + 0.3, y0 - dy * K - 0.3
        y_foot = bot - 0.32

        def band(j, col):
            r = Rectangle(width=1.7, height=top - bot, stroke_color=col, stroke_width=2,
                          fill_color=col, fill_opacity=0.22)
            return r.move_to(np.array([xs[j], (top + bot) / 2, 0.0]))

        b2 = VGroup(band(1, BLUE_D), band(3, BLUE_D), band(5, BLUE_D))
        f2 = VGroup(tag("2 · 3k", 24, BLUE_B).move_to(np.array([xs[1], y_foot, 0])),
                    tag("2(3k + 1)", 24, BLUE_B).move_to(np.array([xs[3], y_foot, 0])),
                    tag("2(3k + 2)", 24, BLUE_B).move_to(np.array([xs[5], y_foot, 0])))
        b3 = band(4, ORANGE)
        f3 = tag("3(2k + 1)", 24, ORANGE).move_to(np.array([xs[4], y_foot, 0]))
        self.add(b2, b3)
        self.bring_to_back(b2, b3)
        self.remove(b3)
        self.play(FadeIn(b2), FadeIn(f2), run_time=0.9)
        self.add(b3)
        self.bring_to_back(b3)
        self.play(FadeIn(b3), FadeIn(f3), run_time=0.8)
        dim = [tx[kj] for kj in tx if offs[kj[1]] in (0, 2, 3, 4) and nums[kj] > 3]
        self.play(*[t_.animate.set_opacity(0.35) for t_ in dim], run_time=0.6)

        # the primes
        def ring(kj, col, w=3):
            return Ellipse(width=0.7, height=0.5, color=col, stroke_width=w).move_to(cell(*kj))

        check(0.5 < dy - 0.04, "rings in one column do not touch")
        rings = VGroup(*[ring(kj, YELLOW_B) for kj, v in nums.items()
                         if _is_prime(v) and v > 3])
        self.play(LaggedStart(*[Create(r) for r in rings], lag_ratio=0.12), run_time=1.8)
        two3 = VGroup(*[ring(kj, WHITE, 2.5) for kj, v in nums.items() if v in (2, 3)])
        for r_ in list(rings) + list(two3):
            inner = [t_ for t_ in tx.values() if close(t_.get_center(), r_.get_center(), 1e-6)]
            check(len(inner) == 1 and inner[0].width < 0.62 and inner[0].height < 0.42,
                  "each ring holds its number")
        self.play(Create(two3), run_time=0.6)
        self.play(hd[0].animate.set_color(YELLOW_B), hd[2].animate.set_color(YELLOW_B),
                  run_time=0.5)

        check(len(rings) == len([v for v in primes if v > 3]), "every prime above 3 is ringed")
        for t_ in list(tx.values()) + list(hd) + list(f2) + [f3]:
            check(_in_frame(t_), f"J23: '{t_.text}' inside the safe area")
        for i, a_ in enumerate(list(f2) + [f3]):
            check(a_.get_bottom()[1] > SAFE_BOTTOM + 0.02 and a_.get_top()[1] < bot - 0.02,
                  "footers below the bands, above the caption band")
        cap = caption("Every prime p > 3 is 6k − 1 or 6k + 1", 34)
        self.play(Write(cap), run_time=1.0)
        _final_check(self, cap)
        self.hold(2.2)
