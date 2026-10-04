# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3g.py — proofs without words, 2D (manim):
#     H32 derivative of arctan            H27 the exponential's subtangent
#     H26 substitution stretches strips   H29 cosine above a parabola
#     L7  roots of unity add to zero      L5  Cramer's rule
#     L10 a linear map scales areas       L9  two reflections, one rotation
#     L13 the rotation matrix             L14 composing rotations
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode, and
# exponents or subscripts made of letters are set with true superscripts /
# subscripts by _supline.  Curves are polylines through finely sampled
# points; there are no Axes with numbers.
#
# Every move of a piece is rigid (a shift, a Rotate about a point, or a fold
# about an in-plane axis, which is a half-turn in space) unless it is an
# announced similarity, stretch, shear or linear map, and every landing,
# area, angle and tiling claim is checked numerically with check(...) before
# or right after it is drawn.  Labels are checked against the lines, marks
# and each other, so a wrong picture fails the render instead of rendering.


# ------------------------------------------------------------ private helpers

def _unit(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _polar(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _ang_of(p, q):
    """Direction angle of the ray p -> q."""
    d = to3(q) - to3(p)
    return float(np.arctan2(d[1], d[0]))


def _cross(a, b):
    a, b = to3(a), to3(b)
    return float(a[0] * b[1] - a[1] * b[0])


def _ang(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _perp(v):
    v = to3(v)
    return np.array([-v[1], v[0], 0.0])


def _rot2(p, c, th):
    """Point p turned by th about c (in the plane)."""
    p, c = to3(p), to3(c)
    d = p - c
    return c + np.array([d[0] * np.cos(th) - d[1] * np.sin(th),
                         d[0] * np.sin(th) + d[1] * np.cos(th), 0.0])


def _mirror(p, a, b):
    """Mirror image of p in the line ab."""
    p, a, b = to3(p), to3(a), to3(b)
    d = (b - a) / np.linalg.norm(b - a)
    f = a + np.dot(p - a, d) * d
    return 2 * f - p


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
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


def _wedge(vertex, p, q, r, color, op=FILL, stroke=0.0, stroke_color=WHITE):
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


def _arrow(p, q, color, width=6, tip=0.26):
    return Arrow(to3(p), to3(q), buff=0.0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.3, tip_length=tip,
                 max_stroke_width_to_length_ratio=40)


def _curve(pts, color=YELLOW_B, width=5):
    """Open polyline through screen points (fine sampling = smooth curve)."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([to3(p) for p in pts])
    return m


def _fold(mob, p, q, **kw):
    """Fold mob over the screen line pq: a half-turn in space about that
    line, rigid on every frame; in the plane it lands as the mirror image."""
    p, q = to3(p), to3(q)
    return Rotate(mob, angle=PI, axis=_unit(q - p), about_point=p, **kw)


def _glide(mob, pivot, target, angle, scale=1.0, **kw):
    """The piece turns by `angle` about its pivot while the pivot travels
    straight to `target`; with scale != 1 it also scales (geometrically in
    time) about the pivot: an announced similarity, rigid when scale = 1."""
    start = mob.copy()
    pivot, target = to3(pivot), to3(target)

    def upd(m, a):
        m.become(start.copy().rotate(a * angle, about_point=pivot)
                 .scale(scale ** a, about_point=pivot)
                 .shift(a * (target - pivot)))
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


def _landed(mob, pts, tol=1e-6):
    """The polygon mob's vertices are pts (same order)."""
    v = mob.get_vertices()
    return len(v) == len(pts) and all(close(a, to3(b), tol)
                                      for a, b in zip(v, pts))


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


def _simpson(g, lo, hi, n=400):
    """Composite Simpson rule (n even)."""
    t = np.linspace(lo, hi, n + 1)
    y = g(t)
    return (hi - lo) / (3 * n) * (y[0] + y[-1] + 4 * y[1:-1:2].sum()
                                  + 2 * y[2:-1:2].sum())


# ------------------------------------------------------------ text helpers

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


def _frac(num, den, gap=0.08, pad=0.08, color=WHITE, width=2.5):
    """A stacked fraction from two mobjects: num over a bar over den."""
    w = max(num.width, den.width) + 2 * pad
    bar = Line(LEFT * w / 2, RIGHT * w / 2, color=color, stroke_width=width)
    num.next_to(bar, UP, buff=gap)
    den.next_to(bar, DOWN, buff=gap)
    return VGroup(num, bar, den)


def _ctag(s, size=26, color=WHITE, t2c=None):
    """One Text (one font run, one baseline) whose substrings are recoloured
    glyph by glyph afterwards (Text's own t2c splits it into runs)."""
    t = Text(s, font_size=size, color=color)
    flat = s.replace(" ", "")
    check(len(t.submobjects) == len(flat), f"one glyph per character in {s!r}")
    for sub, col in (t2c or {}).items():
        i = flat.find(sub.replace(" ", ""))
        while i != -1:
            for gl in t.submobjects[i:i + len(sub.replace(" ", ""))]:
                gl.set_color(col)
            i = flat.find(sub.replace(" ", ""), i + len(sub.replace(" ", "")))
    return t


# ---- label hygiene: every label is checked against the lines it must clear

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


def _arc_segs(c, r, a0, a1, n=48):
    """Polyline segments along the arc of radius r about c from a0 to a1."""
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(a0, a1, n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def _mob_segs(m):
    """Segments along the outline of a VMobject (its anchors, in order)."""
    out = []
    for sub in m.family_members_with_points():
        pts = sub.get_anchors()
        out += [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
                if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-9]
    return out


def _hit(m, segs, pad=0.06):
    """Index of the first segment that passes through m's box (grown by
    pad), or None.  A label made by _rtag (turned along a line) is tested
    against its own turned box instead of the screen-aligned one."""
    if hasattr(m, "_a"):
        return _rhit(m, segs, pad)
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


def _rtag(s, size, color, angle):
    """A label turned by `angle` (to run along a line); it remembers its
    unturned width and height so the checks can use its true box."""
    t = tag(s, size, color)
    t._w, t._h, t._a = t.width, t.height, float(angle)
    t.rotate(angle)
    return t


def _rcorners(m, pad=0.0):
    c = m.get_center()
    u, v = _polar(m._a), _polar(m._a + PI / 2)
    w, h = m._w / 2 + pad, m._h / 2 + pad
    return [c + sx * w * u + sy * h * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _rhit(m, segs, pad=0.06):
    c = m.get_center()
    ca, sa = np.cos(-m._a), np.sin(-m._a)
    w, h = m._w / 2 + pad, m._h / 2 + pad
    for i, (p, q) in enumerate(segs):
        p, q = to3(p), to3(q)
        L = float(np.linalg.norm(q - p))
        for t in np.linspace(0.0, 1.0, max(2, int(L / 0.015) + 2)):
            d = p + t * (q - p) - c
            u0 = d[0] * ca - d[1] * sa
            u1 = d[0] * sa + d[1] * ca
            if abs(u0) <= w and abs(u1) <= h:
                return i
    return None


def _overlap(a, b, gap=0.05):
    """Do the labels' boxes (grown by gap) meet?  Turned labels (_rtag) use
    their turned boxes."""
    if hasattr(a, "_a") or hasattr(b, "_a"):
        if hasattr(b, "_a") and not hasattr(a, "_a"):
            a, b = b, a
        ca = _rcorners(a, gap / 2)
        sides = _segs(*ca, closed=True)
        if hasattr(b, "_a"):
            cb = _rcorners(b, gap / 2)
            return (_rhit(b, sides, 0.0) is not None
                    or _pip(cb[0], ca) or _pip(ca[0], cb))
        return _hit(b, sides, gap / 2) is not None or _pip(b.get_center(), ca)
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
    txt = [s.text for s in m.get_family() if isinstance(s, Text)]
    return "".join(txt) if txt else type(m).__name__


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


def _park_dirs(m, X, dirs_deg, segs, d0=0.25, d1=1.2, pad=0.06, avoid=(), sep=0.08):
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
        for d in np.arange(r0, r1, 0.02):
            m.move_to(V + d * u)
            if (_hit(m, arms, pad) is None
                    and not any(_overlap(m, o, sep) for o in avoid)):
                return m
    check(False, f"room for the angle label '{_name(m)}'")


def _inside(m, poly, pad=0.0):
    """All four corners of m's box (shrunk by -pad) inside the polygon."""
    x0, x1, y0, y1 = _box(m, pad)
    return all(_pip((x, y), poly) for x in (x0, x1) for y in (y0, y1))


def _fit_inside(m, polys, pad=0.08, step=0.05):
    """Move label m to the point nearest the common centroid of the given
    polygons where its box (grown by pad) lies inside every one of them."""
    pts = np.array([to3(p) for P_ in polys for p in P_])
    c0 = pts.mean(axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    best = None
    for xx in np.arange(lo[0], hi[0], step):
        for yy in np.arange(lo[1], hi[1], step):
            q = np.array([xx, yy, 0.0])
            d = np.linalg.norm(q - c0)
            if best is not None and d >= best[0]:
                continue
            m.move_to(q)
            if all(_inside(m, P_, pad) for P_ in polys):
                best = (d, q)
    check(best is not None, f"room for the label '{_name(m)}' inside its region")
    return m.move_to(best[1])




def _bracket(h, side, color=WHITE, width=3, serif=0.12):
    """A square bracket of height h ('[' for side=-1, ']' for side=1)."""
    s = -side * serif
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([np.array([s, h / 2, 0]), np.array([0, h / 2, 0]),
                             np.array([0, -h / 2, 0]), np.array([s, -h / 2, 0])])
    return m


def _matrix(cols, colors, size=30, hgap=0.45, vgap=0.22):
    """A 2-row matrix from Text entries, column by column, each column
    centred, with drawn brackets (no TeX)."""
    entries = [[tag(e, size, c) for e in col] for col, c in zip(cols, colors)]
    rh = max(t.height for col in entries for t in col)
    x = 0.0
    g = VGroup()
    for col in entries:
        w = max(t.width for t in col)
        for r, t in enumerate(col):
            t.move_to(np.array([x + w / 2, (0.5 - r) * (rh + vgap), 0.0]))
            g.add(t)
        x += w + hgap
    h = 2 * rh + vgap + 0.2
    lb = _bracket(h, -1).move_to(np.array([-0.18, 0, 0]), aligned_edge=RIGHT)
    rb = _bracket(h, 1).move_to(np.array([x - hgap + 0.18, 0, 0]), aligned_edge=LEFT)
    return VGroup(lb, g, rb)


# ========================================================== H32  d/dx arctan

class H32_ArctanDerivative(Board):
    """P = (x, 1) runs along the line y = 1. With B = (0, 1), OB = 1 and
    BP = x, so the angle θ between OB and OP has tan θ = x: θ = arctan x,
    and OP = √(1 + x²).  A step dx along the line turns OP by dθ; the
    circle about O through P cuts off a small triangle at P whose arc side
    is √(1 + x²)·dθ (radius times angle).  Its angle at P is θ (the step is
    perpendicular to OB, the arc to OP), so it is OBP shrunk by
    dx/√(1 + x²) and turned over: the copy, slid to P and folded onto the
    step, fits it, the fit becoming exact as dx → 0 (the window magnifies
    by 1/dx).  The arc side is then 1·dx/√(1 + x²):
    √(1 + x²)·dθ = dx/√(1 + x²), so dθ/dx = 1/(1 + x²)."""

    def construct(self):
        x0, dx0, dx1 = 1.2, 0.42, 0.006
        th = float(np.arctan(x0))            # angle between OB and OP
        R0 = float(np.hypot(1.0, x0))        # OP
        k = 3.1
        O = np.array([-6.05, -2.25, 0.0])

        def S(p):                            # math -> screen
            return O + k * to3(p)

        B, P = S((0, 1)), S((x0, 1))
        aP = float(np.arctan2(1.0, x0))      # direction of OP (from the x-axis)
        lt = ValueTracker(np.log(dx0))

        def dx():
            return float(np.exp(lt.get_value()))

        def Pp(d=None):
            d = dx() if d is None else d
            return S((x0 + d, 1))

        def aQ(d=None):                      # direction of OP'
            d = dx() if d is None else d
            return float(np.arctan2(1.0, x0 + d))

        def Q(d=None):                       # where the circle through P meets OP'
            return O + k * R0 * _polar(aQ(d))

        tang = np.array([np.cos(th), -np.sin(th), 0.0])     # circle's tangent at P

        def Rstar(d=None):                   # right-angle corner of the limit triangle
            d = dx() if d is None else d
            return P + k * d * np.cos(th) * tang

        # ---- the claims
        check(abs(np.tan(th) - x0) < 1e-12 and abs(PI / 2 - aP - th) < 1e-12,
              "tan θ = BP / OB = x: θ = arctan x, measured from OB")
        check(abs(np.linalg.norm(P - O) / k - R0) < 1e-12, "OP = √(1 + x²)")
        check(abs(np.dot(tang, P - O)) < 1e-9
              and abs(_ang(P, P + RIGHT, P + tang) - th) < 1e-12,
              "at P the step makes the angle θ with the circle (which is ⊥ OP)")
        for d in (dx0, 0.1, 0.01):
            dth = aP - aQ(d)
            check(abs(np.sin(dth) - d / (R0 * np.hypot(1.0, x0 + d))) < 1e-12,
                  "exactly: sin dθ = dx / (OP·OP′) (twice the area of OPP′)")
        h = 1e-7
        check(abs((np.arctan(x0 + h) - np.arctan(x0 - h)) / (2 * h) - 1 / (1 + x0 * x0))
              < 1e-6, "d/dx arctan x = 1/(1 + x²)")

        # ---- the figure: axes, unit circle, the line y = 1, triangle OBP
        axes = VGroup(Line(S((-0.08, 0)), S((1.95, 0)), color=GREY_B, stroke_width=2),
                      Line(S((0, -0.08)), S((0, 1.28)), color=GREY_B, stroke_width=2))
        ucirc = Arc(radius=k, start_angle=0, angle=PI / 2, arc_center=O,
                    color=GREY_C, stroke_width=2.5)
        line1 = Line(S((-0.12, 1)), S((1.95, 1)), color=GREY_A, stroke_width=3)
        dotO = Dot(O, radius=0.06)
        self.play(Create(axes), Create(ucirc), FadeIn(dotO), run_time=0.9)
        self.play(Create(line1), run_time=0.6)

        cA, cB = BLUE_D, TEAL_B
        tri = mk([O, B, P], cA, 0.45, stroke_width=0)
        OB = Line(O, B, color=cB, stroke_width=6)
        BP = Line(B, P, color=WHITE, stroke_width=5)
        OP = Line(O, P, color=WHITE, stroke_width=4)
        dotP = Dot(P, radius=0.07)
        raB = _ra(B, O, P, 0.2)
        l_1 = tag("1", 28, cB).next_to(OB, LEFT, buff=0.16)
        l_x = tag("x", 30).next_to(Line(B, P), UP, buff=0.14)
        arc_th = angle_arc(O, B, P, radius=0.62, color=GREEN_B, width=4)
        l_th = tag("θ", 28, GREEN_B)
        l_R = _rtag("√(1 + x²)", 24, WHITE, aP)
        l_R.move_to(O + 0.68 * k * _polar(aP) + 0.26 * _polar(aP + PI / 2))
        check(np.linalg.norm(_rcorners(l_R)[2] - O) < k - 0.1
              and np.linalg.norm(_rcorners(l_R)[3] - O) < k - 0.1,
              "the √(1 + x²) label stays inside the unit circle")
        _angle_label(l_th, O, B, P, 0.8, segs=_arc_segs(O, 0.62, aP, PI / 2, 12),
                     avoid=[l_R])
        self.add(tri)
        self.bring_to_back(tri)
        self.play(FadeIn(tri), Create(OB), Create(BP), Create(raB), FadeIn(l_1),
                  FadeIn(l_x), FadeIn(dotP), run_time=1.0)
        self.play(Create(OP), Create(arc_th), FadeIn(l_th), FadeIn(l_R), run_time=0.9)
        r1 = _row(_ctag("tan θ = x", 30, t2c={"θ": GREEN_B}),
                  _ctag("θ = arctan x", 30, t2c={"θ": GREEN_B}), buff=0.8)
        r1.move_to([-3.3, 2.95, 0])
        self.play(FadeIn(r1), run_time=0.7)
        self.hold(0.3)

        # ---- the step dx: OP turns by dθ; the circle through P
        cS, cC, cT = ORANGE, YELLOW_B, GREEN_B

        def main_bits():
            d = dx()
            a1, a0 = aP, aQ(d)
            return VGroup(
                Line(O, Pp(d), color=GREY_B, stroke_width=2.5),
                Arc(radius=k, start_angle=a0, angle=a1 - a0, arc_center=O,
                    color=cT, stroke_width=7),
                Arc(radius=k * R0, start_angle=a0, angle=a1 - a0, arc_center=O,
                    color=cC, stroke_width=5),
                Line(P, Pp(d), color=cS, stroke_width=7),
                Dot(Pp(d), radius=0.06, color=cS))

        bits = main_bits()
        l_dx = tag("dx", 28, cS)
        l_dx.move_to(P + np.array([1.08, 0.36, 0]))
        l_dth = tag("dθ", 26, cT)
        self.play(Create(bits[3]), FadeIn(bits[4]), FadeIn(l_dx), run_time=0.8)
        self.play(Create(bits[0]), Create(bits[1]), run_time=0.8)
        segs_fig = (_segs(S((-0.08, 0)), S((1.95, 0))) + _segs(S((0, -0.08)), S((0, 1.28)))
                    + _arc_segs(O, k, 0, PI / 2, 60) + _segs(S((-0.12, 1)), S((1.95, 1)))
                    + _segs(O, P) + _arc_segs(O, 0.62, aP, PI / 2, 12)
                    + _ra_segs(B, O, P, 0.2))
        segs_step = (_segs(O, Pp(dx0)) + _arc_segs(O, k * R0, aQ(dx0), aP, 12)
                     + _segs(P, Pp(dx0)))
        _park_dirs(l_dth, O + k * _polar((aP + aQ(dx0)) / 2),
                   [-60, -75, -45, -90, -30], segs_fig + segs_step, d0=0.25, d1=1.0,
                   avoid=[l_R, l_x, l_dx])
        self.play(FadeIn(l_dth), run_time=0.5)
        self.play(Create(bits[2]), run_time=0.7)
        self.bring_to_front(dotP)
        self.hold(0.3)

        # ---- a copy of OBP, shrunk by dx/√(1 + x²), slid to P, folded onto the step
        s = dx0 / R0
        cp = mk([O, B, P], cA, 0.75, stroke_width=2)
        self.add(cp)
        tag_s = _supline([("× dx / √(1 + x", 0), ("2", 1), (")", 0)], 26, WHITE)
        tag_s.move_to(O + np.array([1.62, -0.42, 0]))
        small0 = [O, O + s * (B - O), O + s * (P - O)]
        slid = [p + (P - O) for p in small0]
        _labels_ok([tag_s], _segs(*small0, closed=True) + segs_fig + segs_step,
                   "H32 shrink tag")
        _labels_ok([l_dx], _segs(*slid, closed=True), "H32 dx beside the slid copy")
        self.play(cp.animate.scale(s, about_point=O), FadeIn(tag_s), run_time=1.3)
        check(_landed(cp, small0), "shrunk about O by dx/√(1 + x²)")
        self.play(cp.animate.shift(P - O), FadeOut(tag_s), run_time=1.2)
        phi = aP / 2                         # bisects the step and the copy's hypotenuse
        ax = DashedLine(P - 0.6 * _polar(phi), P + 0.85 * _polar(phi), color=GREY_A,
                        stroke_width=2.5, dash_length=0.08)
        self.play(Create(ax), run_time=0.4)
        self.play(Rotate(cp, angle=PI, axis=_polar(phi), about_point=P), run_time=1.4)
        check(_landed(cp, [P, Rstar(dx0), Pp(dx0)]),
              "the folded copy: O to P, B to the right-angle corner, P to P′")
        self.play(FadeOut(ax), run_time=0.3)
        self.bring_to_front(bits, dotP)

        # ---- the window: the neighbourhood of P magnified by L / dx
        L = 3.7
        anc = np.array([2.4, 2.72, 0.0])
        wl, wr, wb, wt = anc[0] - 1.6, anc[0] + L + 0.3, anc[1] - 2.45, anc[1] + 0.72
        Rw = anc + L * np.cos(th) * tang

        def to_win(p, d=None):
            d = dx() if d is None else d
            return anc + (to3(p) - P) * (L / (k * d))

        check(close(to_win(Pp(dx0), dx0), anc + L * RIGHT) and close(to_win(Rstar(dx0), dx0), Rw),
              "the window shows the step as L and the copy as the fixed limit triangle")
        window = Polygon([wl, wb, 0], [wr, wb, 0], [wr, wt, 0], [wl, wt, 0],
                         stroke_color=GREY_B, stroke_width=2)

        def zoom():
            m = k * dx() / L
            lo = P + (np.array([wl, wb, 0]) - anc) * m
            hi = P + (np.array([wr, wt, 0]) - anc) * m
            return VGroup(Polygon(lo, [hi[0], lo[1], 0], hi, [lo[0], hi[1], 0],
                                  stroke_color=GREY_B, stroke_width=1.5),
                          DashedLine([hi[0], hi[1], 0], [wl, wt, 0], color=GREY_D,
                                     stroke_width=1.3, dash_length=0.07),
                          DashedLine([hi[0], lo[1], 0], [wl, wb, 0], color=GREY_D,
                                     stroke_width=1.3, dash_length=0.07))

        lim = mk([anc, Rw, anc + L * RIGHT], cA, 0.55, stroke_width=2)

        def arc_w(d):
            return [to_win(O + k * R0 * _polar(a), d) for a in np.linspace(aP, aQ(d), 40)]

        def inside():
            d = dx()
            arc = arc_w(d)
            return VGroup(
                mk(arc + [to_win(Pp(d), d)], cC, 0.18, stroke_width=0),
                _curve(arc, cC, 6),
                Line(to_win(Q(d), d), to_win(Pp(d), d), color=GREY_B, stroke_width=3),
                Line(anc, to_win(Pp(d), d), color=cS, stroke_width=7))

        zm = always_redraw(zoom)
        ins = always_redraw(inside)
        w_th_arc = angle_arc(anc, anc + RIGHT, Rw, radius=0.62, color=cT, width=4)
        w_th = tag("θ", 28, cT)
        w_dx = tag("dx", 30, cS).move_to(anc + np.array([L / 2, 0.36, 0]))
        w_arc = _supline([("√(1 + x", 0), ("2", 1), (") · dθ", 0)], 26, cC)
        _beside(w_arc, anc, Rw, np.array([-1.0, -1.0, 0.0]), gap=0.34, at=0.5)
        win_segs = (_segs(anc, Rw, anc + L * RIGHT, closed=True)
                    + _arc_segs(anc, 0.62, -th, 0.0, 12)
                    + _segs([wl, wb], [wr, wb], [wr, wt], [wl, wt], closed=True))
        for d in (dx0, dx1):
            win_segs += _segs(*arc_w(d)) + _segs(to_win(Q(d), d), anc + L * RIGHT)
        _angle_label(w_th, anc, anc + RIGHT, Rw, 0.8, segs=win_segs)

        self.play(FadeOut(l_dx), FadeOut(l_dth), Create(window), FadeIn(zm), run_time=0.9)
        self.add(lim)
        self.play(FadeIn(lim), FadeIn(ins), FadeIn(w_dx), Create(w_th_arc), FadeIn(w_th),
                  FadeIn(w_arc), run_time=1.0)
        self.hold(0.6)

        # ---- dx -> 0: the small triangle at P becomes the shrunk copy exactly
        self.remove(bits, cp)
        live = always_redraw(main_bits)

        def small_copy():
            d = dx()
            return mk([P, Rstar(d), Pp(d)], cA, 0.75, stroke_width=2)

        live_cp = always_redraw(small_copy)
        self.add(live_cp, live, dotP)
        self.play(lt.animate.set_value(np.log(dx1)), run_time=4.0, rate_func=smooth)
        for mob in (live, live_cp, zm, ins):
            mob.clear_updaters()
        qw = to_win(Q(dx1), dx1)
        check(np.linalg.norm(qw - Rw) < 0.03,
              "in the window the arc's end has reached the copy's corner")
        arc_len = R0 * (aP - aQ(dx1)) * (L / dx1)
        check(abs(arc_len - L * np.cos(th)) < 0.03,
              "magnified arc √(1 + x²)·dθ → the copy's leg dx/√(1 + x²)")

        # ---- the two readings of that side
        r2 = _supline([("√(1 + x", 0, cC), ("2", 1, cC), (") · dθ", 0, cC),
                       ("  =  ", 0, WHITE), ("dx", 0, cS),
                       (" / √(1 + x", 0, WHITE), ("2", 1, WHITE), (")", 0, WHITE)],
                      28)
        r3 = _supline([("dθ / dx  =  1 / (1 + x", 0, WHITE), ("2", 1, WHITE),
                       (")", 0, WHITE)], 30)
        r2.move_to([3.5, -0.75, 0])
        r3.move_to([3.5, -1.85, 0])
        self.play(FadeIn(r2), Indicate(lim, color=cA, scale_factor=1.0), run_time=1.0)
        self.play(FadeIn(r3), run_time=0.8)

        labels_main = [l_1, l_x, l_th, l_R]
        _labels_ok(labels_main, segs_fig, "H32 figure")
        _labels_ok([l_dx, l_dth, l_th, l_R, l_x], segs_fig + segs_step, "H32 step labels")
        _labels_ok([w_dx, w_th, w_arc], win_segs, "H32 window")
        _labels_ok([r1, r2, r3], [], "H32 rows")
        for r in (r2, r3):
            check(r.get_left()[0] > 0.3 and r.get_top()[1] < wb - 0.1, "rows below the window")
        _safe(axes, ucirc, line1, window, *labels_main, r1, r2, r3)
        cap = _cap(_supline([("d/dx arctan x  =  1 / (1 + x", 0), ("2", 1), (")", 0)], 36))
        _below(cap, axes, r3, *labels_main)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== H27  (eˣ)′ = eˣ

class H27_ExpSubtangent(Board):
    """e is the base whose graph rises with slope 1 at x = 0: the tangent at
    (0, 1) meets the x-axis at −1, a slope triangle with legs 1 and 1.
    Stretch the whole picture vertically by eᵃ: eᵃ·eˣ = eˣ⁺ᵃ, so the
    stretched graph is the graph of eˣ moved left by a.  A vertical
    stretch keeps tangents tangent and keeps where they cross the x-axis,
    so the triangle becomes legs 1 and eᵃ under a tangent of the stretched
    graph.  Slide everything right by a: the stretched graph lands exactly
    on y = eˣ, and the triangle on the slope triangle at (a, eᵃ).  The
    tangent there meets the axis one unit to the left: slope eᵃ/1 = eᵃ."""

    def construct(self):
        a = 1.2
        E = float(np.exp(a))
        F = Frame(-3.45, 1.75, -0.3, 4.05, max_w=7.9, centre=(-2.55, 0.375))
        P = F.P
        k = F.k
        xs = np.linspace(-3.3, 1.35, 320)
        xc = np.linspace(-3.3, 1.35 - a, 260)          # the part that is stretched
        t0, t1 = -1.0, 0.1                              # tangent at 0, drawn piece
        check(abs(np.exp(1.35 - a) * E - np.exp(1.35)) < 1e-12,
              "the stretched copy rises no higher than the curve drawn")
        check(all(abs(E * np.exp(x) - np.exp(x + a)) < 1e-12 for x in xc),
              "eᵃ·eˣ = eˣ⁺ᵃ: the stretch is the shift by −a")
        h = 1e-7
        check(abs((np.exp(h) - np.exp(-h)) / (2 * h) - 1) < 1e-6,
              "slope of eˣ at 0 is 1 (what makes e special)")
        check(abs((np.exp(a + h) - np.exp(a - h)) / (2 * h) - E) < 1e-6,
              "slope of eˣ at a is eᵃ")

        cC, cK, cT = YELLOW_B, TEAL_B, BLUE_D
        axes = VGroup(Line(P((-3.45, 0)), P((1.75, 0)), color=GREY_B, stroke_width=2),
                      Line(P((0, -0.3)), P((0, 4.05)), color=GREY_B, stroke_width=2))
        curve = _curve([P((x, np.exp(x))) for x in xs], cC, 5)
        l_c = _supline([("y = e", 0), ("x", 1)], 28, cC)
        l_c.next_to(P((1.35, np.exp(1.35))), RIGHT, buff=0.15).shift(DOWN * 0.2)
        self.play(Create(axes), run_time=0.7)
        self.play(Create(curve), FadeIn(l_c), run_time=1.2)

        # ---- the tangent at (0, 1): slope 1, it meets the axis one unit left
        T0 = [P((-1, 0)), P((0, 0)), P((0, 1))]
        tri = mk(T0, cT, 0.6, stroke_width=0)
        tan0 = Line(P((t0, 1 + t0)), P((t1, 1 + t1)), color=WHITE, stroke_width=4)
        dot0 = Dot(P((0, 1)), radius=0.07)
        l_b = tag("1", 28).next_to(Line(P((-1, 0)), P((0, 0))), DOWN, buff=0.16)
        l_h = tag("1", 28).move_to(P((-0.22, 0.42)))
        r0 = _supline([("(e", 0), ("x", 1), (")′(0)  =  1", 0)], 30, WHITE)
        r0.move_to([4.25, 2.6, 0])
        self.add(tri)
        self.bring_to_back(tri)
        self.play(FadeIn(tri), Create(tan0), FadeIn(dot0), FadeIn(l_b), FadeIn(l_h),
                  run_time=1.0)
        self.play(FadeIn(r0), run_time=0.7)
        self.hold(0.4)

        # ---- stretch a copy of the picture vertically by eᵃ
        cp_curve = _curve([P((x, np.exp(x))) for x in xc], cK, 5)
        cp_tan = tan0.copy().set_color(cK)
        cp_tri = mk(T0, cK, 0.5, stroke_width=0)
        cp = VGroup(cp_tri, cp_curve, cp_tan)
        self.add(cp)
        r1 = _supline([("e", 0), ("a", 1), (" · e", 0), ("x", 1), ("  =  e", 0),
                       ("x + a", 1)], 30, WHITE)
        r1.move_to([4.25, 1.45, 0])
        tag_s = _supline([("× e", 0), ("a", 1)], 28, cK)
        tag_s.move_to(P((-2.35, 1.6)))
        self.play(FadeOut(l_h), run_time=0.3)
        self.play(cp.animate.stretch(E, 1, about_point=P((0, 0))), FadeIn(tag_s),
                  run_time=1.8)
        check(close(cp_tri.get_vertices()[2], P((0, E)), 1e-6)
              and close(cp_tan.get_start(), P((t0, 0)), 1e-6),
              "stretched: the tangent still meets the axis at −1, the triangle is 1 by eᵃ")
        self.play(FadeIn(r1), run_time=0.7)
        self.hold(0.3)

        # ---- slide it right by a: it lands on y = eˣ itself
        self.play(cp.animate.shift(RIGHT * a * k), FadeOut(tag_s), run_time=1.8)
        self.hold(0.4)
        o = P((0, 0))
        anchors = cp_curve.get_anchors()
        check(all(abs((p[1] - o[1]) / k - np.exp((p[0] - o[0]) / k)) < 1e-6 for p in anchors),
              "the stretched, slid curve lies on y = eˣ")
        T1 = [P((a - 1, 0)), P((a, 0)), P((a, E))]
        check(_landed(cp_tri, T1), "the triangle lands under the point (a, eᵃ)")
        tan_a = Line(cp_tan.get_start(), cp_tan.get_end(), color=WHITE, stroke_width=4)
        check(abs((cp_tan.get_end() - cp_tan.get_start())[1]
                  / (cp_tan.get_end() - cp_tan.get_start())[0] - E) < 1e-6,
              "the landed tangent has slope eᵃ")
        dot_a = Dot(P((a, E)), radius=0.07)
        l_a = tag("a", 28).next_to(P((a, 0)), DOWN, buff=0.18)
        l_b1 = tag("1", 28).next_to(Line(P((a - 1, 0)), P((a, 0))), DOWN, buff=0.16)
        l_ha = _supline([("e", 0), ("a", 1)], 28, WHITE)
        l_ha.next_to(Line(P((a, 0)), P((a, E))), RIGHT, buff=0.15)
        tri_a = mk(T1, cT, 0.6, stroke_width=0)
        self.add(tri_a)
        self.bring_to_back(tri_a)
        self.play(FadeIn(tri_a), FadeOut(cp_tri), FadeOut(cp_curve), Create(tan_a),
                  FadeOut(cp_tan), FadeIn(dot_a), FadeIn(l_a), FadeIn(l_ha), FadeIn(l_b1),
                  FadeIn(l_h), run_time=1.0)
        self.bring_to_front(curve, dot_a)
        r2 = _supline([("(e", 0), ("x", 1), (")′(a)  =  e", 0), ("a", 1), (" / 1  =  e", 0),
                       ("a", 1)], 30, WHITE)
        r2.move_to([4.25, 0.3, 0])
        self.play(FadeIn(r2), run_time=0.8)
        self.hold(0.5)

        # ---- every point: the tangent always meets the axis one unit to the left
        av = ValueTracker(a)

        def moving():
            x0 = av.get_value()
            y0 = float(np.exp(x0))
            return VGroup(mk([P((x0 - 1, 0)), P((x0, 0)), P((x0, y0))], cT, 0.6,
                             stroke_width=0),
                          Line(P((x0 - 1, 0)), P((x0 + 0.1, y0 * 1.1)), color=WHITE,
                               stroke_width=4),
                          Dot(P((x0, y0)), radius=0.07),
                          tag("1", 28).next_to(Line(P((x0 - 1, 0)), P((x0, 0))), DOWN,
                                               buff=0.16))

        mv = always_redraw(moving)
        self.play(FadeOut(l_a), FadeOut(l_ha), run_time=0.4)
        self.remove(tri_a, tan_a, dot_a, l_b1)
        self.add(mv)
        self.bring_to_front(curve)
        self.play(av.animate.set_value(-2.2), run_time=2.2, rate_func=smooth)
        self.play(av.animate.set_value(a), run_time=2.0, rate_func=smooth)
        mv.clear_updaters()
        self.remove(mv)
        self.add(tri_a, tan_a, dot_a, l_b1)
        self.bring_to_back(tri_a)
        self.bring_to_front(curve, dot_a)
        self.play(FadeIn(l_a), FadeIn(l_ha), run_time=0.5)

        segs = (_segs(P((-3.45, 0)), P((1.75, 0))) + _segs(P((0, -0.3)), P((0, 4.05)))
                + _segs(*[P((x, np.exp(x))) for x in xs])
                + _segs(*T1, closed=True) + _segs(tan_a.get_start(), tan_a.get_end())
                + _segs(*T0, closed=True) + _segs(tan0.get_start(), tan0.get_end()))
        _labels_ok([l_c, l_b, l_h, l_b1, l_a, l_ha], segs, "H27 final figure")
        segs0 = (_segs(P((-3.45, 0)), P((1.75, 0))) + _segs(P((0, -0.3)), P((0, 4.05)))
                 + _segs(*[P((x, np.exp(x))) for x in xs]) + _segs(*T0, closed=True)
                 + _segs(tan0.get_start(), tan0.get_end()))
        _labels_ok([l_c, l_h, l_b], segs0, "H27 first triangle")
        stretched = (_segs(*[P((x, E * np.exp(x))) for x in xc])
                     + _segs(P((t0, 0)), P((t1, E * (1 + t1)))))
        _labels_ok([tag_s, l_c], segs0 + stretched, "H27 stretch tag")
        _labels_ok([r0, r1, r2], [], "H27 rows")
        for r in (r0, r1, r2):
            check(r.get_left()[0] > P((1.75, 0))[0] + 0.15 or r.get_bottom()[1] > 3.0,
                  "rows clear of the figure")
        _labels_ok([l_c, r0, r1, r2], [], "H27 rows beside the curve label")
        _safe(axes, curve, l_c, r0, r1, r2)
        cap = _cap(_supline([("d/dx e", 0), ("x", 1), ("  =  e", 0), ("x", 1)], 38))
        _below(cap, axes, curve, l_a, l_b, l_b1)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== H26  substitution

class H26_SubstitutionStrips(Board):
    """Cut [a, b] into strips of width du under y = f(g(u)).  The map
    u ↦ x = g(u) moves each strip sideways onto [g(u), g(u + du)] without
    changing any height: the strips land exactly on the region under f
    over [g(a), g(b)], now of widths dx ≈ g′(u)·du.  Squeeze each one back
    to width du while stretching its height by dx/du (area kept): it stands
    under y = f(g(u))·g′(u).  The same strips, the same total area; as du
    → 0 the stairs fill the region under f(g(u))·g′(u):
    ∫ from g(a) to g(b) of f(x) dx = ∫ from a to b of f(g(u))·g′(u) du."""

    def construct(self):
        a, b = 0.0, 2.0

        def g(u):
            return 0.8 * u + 0.3 * u * u

        def gp(u):
            return 0.8 + 0.6 * u

        def f(x):
            return 1.1 + 0.5 * np.sin(2.2 * x + 0.3)

        k, base = 2.1, -2.45
        Lx0, Rx0 = -6.3, -0.5

        def Lp(u, y):
            return np.array([Lx0 + k * (u - a), base + k * y, 0.0])

        def Rp(x, y):
            return np.array([Rx0 + k * (x - g(a)), base + k * y, 0.0])

        ga, gb = g(a), g(b)
        lhs = _simpson(lambda x: f(x), ga, gb, 2000)
        rhs = _simpson(lambda u: f(g(u)) * gp(u), a, b, 2000)
        check(abs(lhs - rhs) < 1e-9, "∫ f(x) dx over [g(a), g(b)] = ∫ f(g(u)) g′(u) du")
        check(all(gp(u) > 0 for u in np.linspace(a, b, 50)), "g increasing")

        def strip_u(i, N, n=14):          # math (u, y) outline under f(g(u))
            us = np.linspace(a + (b - a) * i / N, a + (b - a) * (i + 1) / N, n)
            return [(us[0], 0.0)] + [(u, f(g(u))) for u in us] + [(us[-1], 0.0)]

        def strip_x(i, N, n=14):          # the same strip moved by u -> g(u)
            return [(g(u), y) for u, y in strip_u(i, N, n)]

        def squeezed(i, N, n=14):         # x-strip squeezed to width du, area kept
            du = (b - a) / N
            u0 = a + du * i
            x0, x1 = g(u0), g(u0 + du)
            r = (x1 - x0) / du
            return [(u0 + (x - x0) / r, y * r) for x, y in strip_x(i, N, n)]

        for N in (6, 24):
            xs_poly = [strip_x(i, N) for i in range(N)]
            region = ([(ga, 0.0)] + [(x, f(x)) for x in np.linspace(ga, gb, 400)]
                      + [(gb, 0.0)])
            check(all(abs(y - f(x)) < 1e-12 for P_ in xs_poly for x, y in P_[1:-1]),
                  "stretched strips keep their heights: their tops lie on y = f(x)")
            check(abs(sum(abs(area(P_)) for P_ in xs_poly) - lhs) < 3e-3
                  and abs(abs(area(region)) - lhs) < 3e-3,
                  "the stretched strips fill the region under f (area ∫ f dx)")
            sq = [squeezed(i, N) for i in range(N)]
            check(all(abs(area(s_) - area(x_)) < 1e-12 for s_, x_ in zip(sq, xs_poly)),
                  "each squeezed strip keeps its area")
        check(_tiles_exactly([strip_x(i, 6, 40) for i in range(6)],
                             [(ga, 0.0)] + [(x, f(x)) for x in np.linspace(ga, gb, 300)]
                             + [(gb, 0.0)], 90),
              "the six stretched strips tile the region under f")
        dev = max(abs(y - f(g(u)) * gp(u)) for i in range(24)
                  for u, y in squeezed(i, 24)[1:-1])
        check(dev < 0.08, f"24 squeezed strips hug y = f(g(u))·g′(u) (max gap {dev:.3f})")

        # ---- the two planes
        axL = Line(Lp(a - 0.05, 0), Lp(b + 0.12, 0), color=GREY_B, stroke_width=2)
        axR = Line(Rp(ga - 0.05, 0), Rp(gb + 0.12, 0), color=GREY_B, stroke_width=2)
        l_u = tag("u", 26, GREY_A).next_to(Lp(b + 0.12, 0), RIGHT, buff=0.1)
        l_xa = tag("x", 26, GREY_A).next_to(Rp(gb + 0.12, 0), RIGHT, buff=0.1)
        tick = lambda p: Line(p + 0.08 * DOWN, p + 0.08 * UP, color=GREY_B, stroke_width=2)
        tks = VGroup(tick(Lp(a, 0)), tick(Lp(b, 0)), tick(Rp(ga, 0)), tick(Rp(gb, 0)))
        l_a = tag("a", 26).next_to(Lp(a, 0), DOWN, buff=0.15)
        l_b = tag("b", 26).next_to(Lp(b, 0), DOWN, buff=0.15)
        l_ga = tag("g(a)", 26).next_to(Rp(ga, 0), DOWN, buff=0.15)
        l_gb = tag("g(b)", 26).next_to(Rp(gb, 0), DOWN, buff=0.15)
        xsR = np.linspace(ga, gb, 300)
        cf, cg, ci = YELLOW_B, GREY_A, ORANGE
        crv_f = _curve([Rp(x, f(x)) for x in xsR], cf, 5)
        l_f = tag("f(x)", 28, cf).next_to(Rp(0.58, 1.6), UP, buff=0.18)
        usL = np.linspace(a, b, 300)
        crv_fg = DashedVMobject(_curve([Lp(u, f(g(u))) for u in usL], cg, 3),
                                num_dashes=60)
        l_fg = tag("f(g(u))", 26, cg)
        l_fg.next_to(Lp(b, f(g(b))), RIGHT, buff=0.15)
        self.play(Create(axL), Create(axR), Create(tks), FadeIn(l_a), FadeIn(l_b),
                  FadeIn(l_ga), FadeIn(l_gb), FadeIn(l_u), FadeIn(l_xa), run_time=1.0)
        self.play(Create(crv_f), FadeIn(l_f), run_time=1.0)
        self.play(Create(crv_fg), FadeIn(l_fg), run_time=1.0)

        # ---- strips of width du under f(g(u))
        N = 6
        hi = 3                                      # the strip that carries the labels
        cols = [BLUE_D, TEAL_D]

        def col(i):
            return ORANGE if i == hi else cols[i % 2]

        def poly(pts, c, op=0.8):
            return mk(pts, c, op, stroke_width=1.5)

        stripsL = VGroup(*[poly([Lp(*p) for p in strip_u(i, N)], col(i)) for i in range(N)])
        self.play(LaggedStart(*[FadeIn(s_) for s_ in stripsL], lag_ratio=0.12), run_time=1.2)
        self.bring_to_front(crv_fg)
        du_lab = tag("du", 26, ORANGE).next_to(
            Line(Lp(a + (b - a) * hi / N, 0), Lp(a + (b - a) * (hi + 1) / N, 0)), DOWN,
            buff=0.15)
        self.play(FadeIn(du_lab), run_time=0.5)

        # ---- u -> g(u): every strip slides and stretches sideways, heights kept
        movers = VGroup(*[s_.copy() for s_ in stripsL])
        startL = [[Lp(*p) for p in strip_u(i, N)] for i in range(N)]
        endR = [[Rp(*p) for p in strip_x(i, N)] for i in range(N)]

        def stretch_upd(m, al):
            for j, s_ in enumerate(m):
                s_.set_points_as_corners(
                    [(1 - al) * p + al * q for p, q in zip(startL[j], endR[j])]
                    + [(1 - al) * startL[j][0] + al * endR[j][0]])

        self.add(movers)
        self.play(UpdateFromAlphaFunc(movers, stretch_upd), run_time=2.4)
        for j, s_ in enumerate(movers):
            got = s_.get_anchors()
            check(close(got[0], endR[j][0], 1e-6) and close(got[-1], endR[j][0], 1e-6)
                  and all(any(np.linalg.norm(q - p) < 1e-6 for q in got) for p in endR[j]),
                  "each strip lands on its piece of the region under f")
        self.bring_to_front(crv_f)
        i0 = a + (b - a) * hi / N
        x_lo, x_hi = g(i0), g(i0 + (b - a) / N)
        dx_lab = tag("dx", 26, ORANGE).next_to(Line(Rp(x_lo, 0), Rp(x_hi, 0)), DOWN,
                                              buff=0.15)
        lvl = DashedLine(Lp(i0, f(g(i0))), Rp(x_lo, f(x_lo)), color=ORANGE,
                         stroke_width=2.5, dash_length=0.09)
        check(abs(Lp(i0, f(g(i0)))[1] - Rp(x_lo, f(x_lo))[1]) < 1e-12,
              "the strip's height carries over: f(g(u)) at u, f(x) at x = g(u)")
        self.play(FadeIn(dx_lab), Create(lvl), run_time=0.8)
        self.hold(0.6)

        # ---- squeeze each x-strip back to width du, its height × dx/du
        cI = WHITE
        crv_i = _curve([Lp(u, f(g(u)) * gp(u)) for u in usL], cI, 4)
        l_i = _supline([("f(g(u)) g′(u)", 0)], 26, cI)
        u_pk = max(np.linspace(a, 1.0, 200), key=lambda u: f(g(u)) * gp(u))
        tops = _segs(*[Lp(u, f(g(u)) * gp(u)) for u in usL])
        for n_ in (N, 24):
            for i in range(n_):
                tops += _segs(*[Lp(*p) for p in squeezed(i, n_)])
        for bf in np.arange(0.22, 1.2, 0.03):
            l_i.next_to(Lp(u_pk, f(g(u_pk)) * gp(u_pk)), UP, buff=bf)
            l_i.shift(RIGHT * max(0.0, -SAFE_X + 0.05 - l_i.get_left()[0]))
            if _hit(l_i, tops, 0.08) is None:
                break
        self.play(FadeOut(stripsL), FadeOut(lvl), run_time=0.6)
        self.play(Create(crv_i), FadeIn(l_i), run_time=1.0)
        backs = VGroup(*[s_.copy() for s_ in movers])
        startR = [[Rp(*p) for p in strip_x(i, N)] for i in range(N)]
        endL = [[Lp(*p) for p in squeezed(i, N)] for i in range(N)]

        def squeeze_upd(m, al):
            du = (b - a) / N
            for j, s_ in enumerate(m):
                u0 = a + du * j
                x0, x1 = g(u0), g(u0 + du)
                r = (x1 - x0) / du
                w, hh = r ** (-al), r ** al            # width and height factors
                bl = (1 - al) * Rp(x0, 0) + al * Lp(u0, 0)
                pts = [bl + np.array([k * (x - x0) * w, k * y * hh, 0.0])
                       for x, y in strip_x(j, N)]
                s_.set_points_as_corners(pts + [pts[0]])

        self.add(backs)
        self.bring_to_front(crv_f)
        self.play(UpdateFromAlphaFunc(backs, squeeze_upd), run_time=2.4)
        for j, s_ in enumerate(backs):
            got = s_.get_anchors()
            check(all(any(np.linalg.norm(q - p) < 1e-6 for q in got) for p in endL[j]),
                  "each strip lands on its squeezed place over [u, u + du]")
        self.bring_to_front(crv_i, crv_fg)
        self.hold(0.5)

        # ---- thinner strips: the stairs close in on f(g(u))·g′(u)
        N2 = 24
        cols2 = lambda i: cols[i % 2]
        fineR = VGroup(*[mk([Rp(*p) for p in strip_x(i, N2)], cols2(i), 0.8,
                            stroke_width=0.8) for i in range(N2)])
        fineL = VGroup(*[mk([Rp(*p) for p in strip_x(i, N2)], cols2(i), 0.8,
                            stroke_width=0.8) for i in range(N2)])
        endL2 = [[Lp(*p) for p in squeezed(i, N2)] for i in range(N2)]
        self.play(FadeOut(movers), FadeOut(backs), FadeOut(du_lab), FadeOut(dx_lab),
                  FadeIn(fineR), FadeIn(fineL), run_time=0.8)
        self.bring_to_front(crv_f)
        startR2 = [[Rp(*p) for p in strip_x(i, N2)] for i in range(N2)]

        def squeeze2(m, al):
            du = (b - a) / N2
            for j, s_ in enumerate(m):
                u0 = a + du * j
                x0, x1 = g(u0), g(u0 + du)
                r = (x1 - x0) / du
                w, hh = r ** (-al), r ** al
                bl = (1 - al) * Rp(x0, 0) + al * Lp(u0, 0)
                pts = [bl + np.array([k * (x - x0) * w, k * y * hh, 0.0])
                       for x, y in strip_x(j, N2)]
                s_.set_points_as_corners(pts + [pts[0]])

        self.play(UpdateFromAlphaFunc(fineL, squeeze2), run_time=2.0)
        for j, s_ in enumerate(fineL):
            got = s_.get_anchors()
            check(all(any(np.linalg.norm(q - p) < 1e-6 for q in got) for p in endL2[j]),
                  "fine strips land on their squeezed places")
        self.bring_to_front(crv_i, crv_fg)

        labels = [l_a, l_b, l_ga, l_gb, l_u, l_xa, l_f, l_fg, l_i]
        segs = (_segs(Lp(a - 0.05, 0), Lp(b + 0.12, 0)) + _segs(Rp(ga - 0.05, 0), Rp(gb + 0.12, 0))
                + _segs(*[Rp(x, f(x)) for x in xsR]) + _segs(*[Lp(u, f(g(u))) for u in usL])
                + _segs(*[Lp(u, f(g(u)) * gp(u)) for u in usL])
                + [_seg(t_.get_start(), t_.get_end()) for t_ in tks])
        for i in range(N):
            segs += _segs(*[Lp(*p) for p in squeezed(i, N)], closed=True)
            segs += _segs(*[Rp(*p) for p in strip_x(i, N)], closed=True)
        _labels_ok(labels + [du_lab, dx_lab], segs, "H26")
        _safe(axL, axR, crv_f, crv_i, *labels)
        cap = _cap(_row(_row(_int("g(a)", "g(b)", 32), tag("f(x) dx", 32, YELLOW_B),
                             buff=0.1),
                        tag("=", 32, YELLOW_B),
                        _row(_int("a", "b", 32), tag("f(g(u)) g′(u) du", 32, YELLOW_B),
                             buff=0.1), buff=0.3))
        _below(cap, axL, axR, *labels)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)


# ========================================================== H29  1 − cos x ≤ x²/2

class H29_CosineParabola(Board):
    """On the unit circle the point at arc length t from A = (1, 0) stands
    sin t above the diameter: the drop to the diameter is the shortest way
    from the point to that line, and the arc from A is another way, so
    sin t ≤ t.  Graph both against t: on [0, x] the curve y = sin t stays
    under the line y = t, so the region under the sine arch lies inside the
    triangle under the line.  Their areas are ∫₀ˣ sin t dt = 1 − cos x and
    ∫₀ˣ t dt = x²/2: 1 − cos x ≤ x²/2, i.e. cos x ≥ 1 − x²/2.
    (Drawn for 0 < x ≤ π; cos is even, and for |x| ≥ 2 the bound is
    trivial because 1 − cos x ≤ 2 ≤ x²/2.)"""

    def construct(self):
        x0 = 2.2
        k = 2.3
        Og = np.array([-6.1, -2.55, 0.0])

        def G(t, y):
            return Og + k * np.array([t, y, 0.0])

        C, r = np.array([4.15, -1.35, 0.0]), k

        def Cp(a):
            return C + r * _polar(a)

        ts = np.linspace(0, x0, 400)
        check(all(np.sin(t) <= t + 1e-15 for t in ts), "sin t ≤ t on [0, x]")
        I_sin = _simpson(np.sin, 0, x0, 2000)
        check(abs(I_sin - (1 - np.cos(x0))) < 1e-12, "∫₀ˣ sin t dt = 1 − cos x")
        check(abs(_simpson(lambda t: t, 0, x0, 200) - x0 * x0 / 2) < 1e-12,
              "∫₀ˣ t dt = x²/2")
        check(1 - np.cos(x0) <= x0 * x0 / 2, "1 − cos x ≤ x²/2")
        check(0 < x0 <= PI, "drawn for 0 < x ≤ π (sin t ≥ 0 on [0, x])")
        reg_sin = [(0.0, 0.0)] + [(t, np.sin(t)) for t in ts] + [(x0, 0.0)]
        tri = [(0.0, 0.0), (x0, 0.0), (x0, x0)]
        check(all(_pip(p, [(0, -1e-9), (x0 + 1e-9, -1e-9), (x0 + 1e-9, x0 + 2e-9)])
                  for p in [(t, 0.5 * np.sin(t)) for t in ts[1:-1]]),
              "the region under the sine arch lies in the triangle")

        cS, cT, cR = BLUE_B, YELLOW_B, ORANGE
        # ---- the graph: y = t and y = sin t
        tmax = 2.45
        axes = VGroup(Line(G(-0.1, 0), G(tmax + 0.12, 0), color=GREY_B, stroke_width=2),
                      Line(G(0, -0.1), G(0, 2.5), color=GREY_B, stroke_width=2))
        l_tax = tag("t", 26, GREY_A).next_to(G(tmax + 0.12, 0), RIGHT, buff=0.1)
        lin = Line(G(0, 0), G(tmax, tmax), color=cT, stroke_width=4)
        sinc = _curve([G(t, np.sin(t)) for t in np.linspace(0, tmax, 300)], cS, 5)
        l_lin = tag("y = t", 26, cT).next_to(G(tmax, tmax), RIGHT, buff=0.12)
        l_sin = tag("y = sin t", 26, cS).next_to(G(tmax, np.sin(tmax)), RIGHT, buff=0.12)
        tick_x = Line(G(x0, 0) + 0.08 * DOWN, G(x0, 0) + 0.08 * UP, color=GREY_B,
                      stroke_width=2)
        l_x = tag("x", 28).next_to(G(x0, 0), DOWN, buff=0.15)
        self.play(Create(axes), FadeIn(l_tax), run_time=0.8)
        self.play(Create(lin), Create(sinc), FadeIn(l_lin), FadeIn(l_sin), Create(tick_x),
                  FadeIn(l_x), run_time=1.4)

        # ---- the unit circle: the drop sin t is shorter than the arc t
        diam = Line(C + r * LEFT, C + r * RIGHT, color=GREY_B, stroke_width=2)
        semi = Arc(radius=r, start_angle=0, angle=PI, arc_center=C, color=GREY_C,
                   stroke_width=2.5)
        dotC = Dot(C, radius=0.05, color=GREY_A)
        l_one = tag("1", 26, GREY_A).next_to(Line(C, C + r * RIGHT), DOWN, buff=0.12)
        self.play(Create(diam), Create(semi), FadeIn(dotC), FadeIn(l_one), run_time=1.0)

        tv = ValueTracker(0.0)

        def circ_bits():
            t = max(tv.get_value(), 1e-4)
            P = Cp(t)
            F = np.array([P[0], C[1], 0.0])
            return VGroup(Arc(radius=r, start_angle=0, angle=t, arc_center=C, color=cT,
                              stroke_width=7),
                          Line(P, F, color=cS, stroke_width=7),
                          Dot(P, radius=0.07))

        def graph_bits():
            t = max(tv.get_value(), 1e-4)
            tt = np.linspace(0, t, 120)
            return VGroup(mk([G(0, 0), G(t, 0), G(t, t)], cR, 0.55, stroke_width=0),
                          mk([G(0, 0)] + [G(s, np.sin(s)) for s in tt] + [G(t, 0)], BLUE_D,
                             0.85, stroke_width=0),
                          Line(G(t, 0), G(t, np.sin(t)), color=cS, stroke_width=6),
                          Line(G(t, np.sin(t)), G(t, t), color=cT, stroke_width=3),
                          Dot(G(t, np.sin(t)), radius=0.06, color=cS),
                          Dot(G(t, t), radius=0.06, color=cT))

        cb, gb = always_redraw(circ_bits), always_redraw(graph_bits)
        self.add(gb, cb)
        self.bring_to_front(lin, sinc)
        self.play(tv.animate.set_value(x0), run_time=4.0, rate_func=linear)
        for m in (cb, gb):
            m.clear_updaters()
        P = Cp(x0)
        F = np.array([P[0], C[1], 0.0])
        check(abs(np.linalg.norm(P - F) - r * np.sin(x0)) < 1e-9, "the drop is sin x")
        check(r * np.sin(x0) < r * x0, "the drop is shorter than the arc")
        l_arc = tag("x", 28, cT).move_to(Cp(x0 / 2) + 0.32 * _polar(x0 / 2))
        l_drop = tag("sin x", 26, cS).next_to(Line(P, F), RIGHT, buff=0.14)
        self.play(FadeIn(l_arc), FadeIn(l_drop), run_time=0.7)

        # ---- the two areas
        l_area = tag("1 − cos x", 26, WHITE).move_to(G(1.2, 0.36))
        outline = Polygon(G(0, 0), G(x0, 0), G(x0, x0), stroke_color=cR, stroke_width=4)
        self.play(FadeIn(l_area), Create(outline), run_time=0.9)
        ic = 0.14                            # both icons share one scale
        ic_s = mk([G(*p) for p in reg_sin], BLUE_D, 0.85, stroke_width=1.5).scale(ic)
        ic_t = VGroup(mk([G(*p) for p in tri], cR, 0.55, stroke_width=0),
                      Polygon(*[G(*p) for p in tri], stroke_color=cR, stroke_width=4)
                      ).scale(ic)
        rowA = _row(ic_s, _row(tag("=", 26), _int("0", "x", 26, WHITE),
                               tag("sin t dt  =  1 − cos x", 26), buff=0.1), buff=0.22)
        rowB = _row(ic_t, _row(tag("=", 26), _int("0", "x", 26, WHITE),
                               _supline([("t dt  =  x", 0), ("2", 1), (" / 2", 0)], 26,
                                        WHITE), buff=0.1), buff=0.22)
        rowA.move_to([3.6, 3.1, 0]).align_to(np.array([1.05, 0, 0]), LEFT)
        rowB.move_to([3.6, 2.05, 0]).align_to(np.array([1.05, 0, 0]), LEFT)
        rowB[1].align_to(rowA[1], LEFT)
        self.play(FadeIn(rowA), run_time=0.8)
        self.play(FadeIn(rowB), run_time=0.8)
        self.hold(0.6)

        segs_g = (_segs(G(-0.1, 0), G(tmax + 0.12, 0)) + _segs(G(0, -0.1), G(0, 2.5))
                  + _segs(G(0, 0), G(tmax, tmax))
                  + _segs(*[G(t, np.sin(t)) for t in np.linspace(0, tmax, 300)])
                  + _segs(G(0, 0), G(x0, 0), G(x0, x0), closed=True)
                  + [_seg(tick_x.get_start(), tick_x.get_end())])
        segs_c = (_segs(C + r * LEFT, C + r * RIGHT) + _arc_segs(C, r, 0, PI, 90)
                  + _segs(P, F))
        _labels_ok([l_tax, l_lin, l_sin, l_x, l_area], segs_g, "H29 graph")
        check(_inside(l_area, [G(*p) for p in reg_sin], 0.05), "area label inside the region")
        _labels_ok([l_one, l_arc, l_drop], segs_c, "H29 circle")
        _labels_ok([rowA, rowB, l_lin, l_sin, l_arc], segs_c + segs_g, "H29 rows")
        _safe(axes, lin, sinc, semi, diam, rowA, rowB)
        cap = _cap(_row(_supline([("1 − cos x  ≤  x", 0), ("2", 1), (" / 2", 0)], 34),
                        tag("⟹", 34, YELLOW_B),
                        _supline([("cos x  ≥  1 − x", 0), ("2", 1), (" / 2", 0)], 34),
                        buff=0.5))
        _below(cap, axes, diam, l_x, l_one)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== L7  roots of unity

class L7_RootsOfUnity(Board):
    """The seven arrows from the centre to the corners 1, ω, …, ω⁶ of a
    regular heptagon (ω = e^(2πi/7)) all have length 1, and each is the
    one before turned through 2π/7.  Slid without turning, head to tail in
    that order, they make a path of seven equal steps that turns through
    the same 2π/7 at every corner: a full turn after seven steps, a regular
    heptagon, so the path closes and the sum is 0.  (Drawn for n = 7; the
    same holds for every n ≥ 2.)"""

    def construct(self):
        n = 7
        R = 2.05
        CL = np.array([-3.75, 0.3, 0.0])
        w = np.exp(2j * PI / n)
        check(abs(sum(w ** j for j in range(n))) < 1e-12, "1 + ω + … + ω⁶ = 0")

        def cpt(z, base):
            return base + R * np.array([z.real, z.imag, 0.0])

        Z = complex(3.35, 0.3)
        S = Z + 1 / (w - 1)               # chain start (units of R), centre Z
        # chain corners S_k = S + Σ_{j<k} ω^j (in units of R), centred on Z
        Sc = [S + sum(w ** j for j in range(kk)) for kk in range(n + 1)]
        ZS = np.array([Z.real, Z.imag, 0.0])

        def ch(z):                                   # chain point -> screen
            return ZS + R * np.array([(z - Z).real, (z - Z).imag, 0.0])

        rad = 1 / abs(w - 1)
        check(abs(Sc[n] - Sc[0]) < 1e-12, "the head-to-tail path closes")
        check(all(abs(abs(s - Z) - rad) < 1e-12 for s in Sc),
              "its corners lie on one circle: a regular heptagon")

        cols = [BLUE_B, TEAL_B, GREEN_B, YELLOW_B, ORANGE, RED_B, PURPLE_B]
        circ = Circle(radius=R, color=GREY_C, stroke_width=2).move_to(CL)
        dotC = Dot(CL, radius=0.06)
        arrows = VGroup(*[_arrow(CL, cpt(w ** j, CL), cols[j], 5, 0.24) for j in range(n)])
        heads = [cpt(w ** j, CL) for j in range(n)]
        labs = VGroup()
        for j in range(n):
            if j == 0:
                t = tag("1", 28, cols[j])
            elif j == 1:
                t = tag("ω", 30, cols[j])
            else:
                t = _supline([("ω", 0), (str(j), 1)], 30, cols[j])
            t.move_to(CL + (R + 0.42) * _polar(2 * PI * j / n))
            labs.add(t)
        r_w = _supline([("ω = e", 0), ("2πi/7", 1)], 30, WHITE)
        r_w.move_to([-3.75, 3.3, 0])
        self.play(Create(circ), FadeIn(dotC), run_time=0.7)
        self.play(LaggedStart(*[GrowArrow(a_) for a_ in arrows], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(l_) for l_ in labs], lag_ratio=0.15), run_time=1.6)
        self.play(FadeIn(r_w), run_time=0.6)
        # equal turns of 2π/7 between neighbours
        a01 = angle_arc(CL, heads[0], heads[1], radius=0.62, color=WHITE, width=3)
        l01 = _angle_label(tag("2π/7", 22), CL, heads[0], heads[1], 0.8,
                           segs=_arc_segs(CL, 0.62, 0, 2 * PI / n, 8))
        self.play(Create(a01), FadeIn(l01), run_time=0.6)
        self.hold(0.3)

        # ---- head to tail: each arrow slides (no turn) to the tip of the last
        movers = VGroup()
        for j in range(n):
            m = arrows[j].copy()
            self.add(m)
            self.play(m.animate.shift(ch(Sc[j]) - CL), run_time=0.75 if j < 2 else 0.55)
            check(close(m.get_start(), ch(Sc[j]), 1e-6) and close(m.get_end(), ch(Sc[j + 1]),
                                                                  1e-6),
                  "each arrow, slid, runs from one corner of the path to the next")
            movers.add(m)
        start = Dot(ch(Sc[0]), radius=0.1, color=WHITE)
        self.play(GrowFromCenter(start), run_time=0.4)
        self.play(Indicate(start, scale_factor=1.8, color=WHITE), run_time=0.7)

        # ---- the same turn at every corner: 2π/7, a full turn in all
        ext = VGroup()
        for j in range(1, n + 1):
            P_ = ch(Sc[j % n])
            d_in = _unit(ch(Sc[j]) - ch(Sc[j - 1]))
            q_out = ch(Sc[(j + 1) % n]) if j < n else ch(Sc[1])
            ext.add(DashedLine(P_, P_ + 0.62 * d_in, color=GREY_B, stroke_width=2,
                               dash_length=0.06),
                    angle_arc(P_, P_ + d_in, q_out, radius=0.45, color=WHITE, width=3))
        check(all(abs(_ang(ch(Sc[j]), ch(Sc[j]) + (ch(Sc[j]) - ch(Sc[j - 1])),
                           ch(Sc[j + 1])) - 2 * PI / n) < 1e-9 for j in range(1, n)),
              "the path turns through 2π/7 at every corner")
        self.play(LaggedStart(*[Create(e_) for e_ in ext], lag_ratio=0.08), run_time=1.6)
        l_ext = tag("2π/7", 22)
        j0 = 2
        P0 = ch(Sc[j0])
        d0 = _unit(ch(Sc[j0]) - ch(Sc[j0 - 1]))
        _angle_label(l_ext, P0, P0 + d0, ch(Sc[j0 + 1]), 0.6,
                     segs=_mob_segs(ext[2 * (j0 - 1)]) + _mob_segs(ext[2 * (j0 - 1) + 1]))
        self.play(FadeIn(l_ext), run_time=0.5)

        segs = (_segs(*[ch(s) for s in Sc]) + _arc_segs(CL, R, 0, TAU, 120)
                + [(CL, h) for h in heads] + _arc_segs(CL, 0.62, 0, 2 * PI / n, 8))
        for e_ in ext:
            segs += _mob_segs(e_)
        _labels_ok(list(labs) + [l01, l_ext, r_w], segs, "L7")
        _safe(circ, *movers, *labs, r_w)
        cap = _cap(_supline([("1 + ω + ω", 0), ("2", 1), (" + ω", 0), ("3", 1),
                             (" + … + ω", 0), ("6", 1), ("  =  0", 0)], 36))
        _below(cap, circ, *movers, *labs)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== L5  Cramer's rule

class L5_CramersRule(Board):
    """c = x·a + y·b.  The parallelogram on c and b keeps its side b on one
    line and its opposite side on the parallel line through c; sliding that
    side along its line by −y·b (a shear) brings c to x·a without changing
    the area: det(c, b) = det(x·a, b).  The parallelogram on x·a and b is
    cut by lines parallel to b into copies of the parallelogram on a and b,
    slid along a: two whole ones and 0.4 of a third here, x of them in all.
    So det(c, b) = x·det(a, b), and x = det(c, b) / det(a, b).  (Drawn with
    det(a, b) > 0 and x = 2.4, y = 0.7; for other signs the areas are
    signed.)"""

    def construct(self):
        am, bm = np.array([1.5, 0.3]), np.array([0.5, 1.3])
        x, y = 2.4, 0.7
        cm = x * am + y * bm
        k = 1.5
        O = np.array([-6.0, -2.35, 0.0])

        def P(v):
            return O + k * to3(v)

        def det(u, v):
            return float(u[0] * v[1] - u[1] * v[0])

        check(det(am, bm) > 0, "det(a, b) > 0")
        check(abs(det(cm, bm) - x * det(am, bm)) < 1e-12, "det(c, b) = x·det(a, b)")
        check(abs(det(cm - x * am, bm)) < 1e-12, "x·a lies on the line through c parallel to b")
        Pab = [(0, 0), tuple(am), tuple(am + bm), tuple(bm)]
        Pcb = [(0, 0), tuple(cm), tuple(cm + bm), tuple(bm)]
        Pxb = [(0, 0), tuple(x * am), tuple(x * am + bm), tuple(bm)]
        check(abs(area(Pcb) - area(Pxb)) < 1e-12, "the shear keeps the area")
        sh = lambda poly, v: [tuple(np.add(p, v)) for p in poly]
        part = [tuple(2 * am), tuple(x * am), tuple(x * am + bm), tuple(2 * am + bm)]
        check(_tiles_exactly([Pab, sh(Pab, am), part], Pxb, 120),
              "two copies and a 0.4-slice tile the parallelogram on x·a, b")
        check(abs(area(part) - (x - 2) * area(Pab)) < 1e-12, "the slice is 0.4 of a copy")

        cA, cB, cC = BLUE_B, ORANGE, YELLOW_B
        dotO = Dot(O, radius=0.06)
        arA = _arrow(P((0, 0)), P(am), cA, 5)
        arB = _arrow(P((0, 0)), P(bm), cB, 5)
        l_a = tag("a", 30, cA)
        _beside(l_a, P((0, 0)), P(am), DOWN, gap=0.14, at=0.55)
        l_b = tag("b", 30, cB)
        _beside(l_b, P((0, 0)), P(bm), LEFT, gap=0.14, at=0.5)
        pab = mk([P(p) for p in Pab], BLUE_D, 0.6, stroke_width=0)
        l_dab = tag("det(a, b)", 22)
        l_dab.move_to(P(np.mean(np.array(Pab), axis=0)))
        check(_inside(l_dab, [P(p) for p in Pab], 0.06), "det(a, b) inside its parallelogram")
        self.play(FadeIn(dotO), GrowArrow(arA), GrowArrow(arB), FadeIn(l_a), FadeIn(l_b),
                  run_time=1.0)
        self.add(pab)
        self.bring_to_back(pab)
        self.play(FadeIn(pab), FadeIn(l_dab), run_time=0.8)
        self.hold(0.3)

        # ---- c = x·a + y·b
        seg_xa = Line(P((0, 0)), P(x * am), color=cA, stroke_width=7)
        seg_yb = Line(P(x * am), P(cm), color=cB, stroke_width=7)
        arC = _arrow(P((0, 0)), P(cm), cC, 6)
        l_xa = tag("x·a", 28, cA)
        _beside(l_xa, P(am), P(x * am), DOWN, gap=0.14, at=0.55)
        l_yb = tag("y·b", 28, cB)
        _beside(l_yb, P(x * am), P(cm), RIGHT, gap=0.14, at=0.5)
        l_c = tag("c", 30, cC).next_to(P(cm), RIGHT, buff=0.15)
        r1 = _ctag("c = x·a + y·b", 30, t2c={"c": cC, "x·a": cA, "y·b": cB})
        r1.move_to([4.1, 2.9, 0])
        self.play(Create(seg_xa), FadeIn(l_xa), run_time=0.9)
        self.play(Create(seg_yb), FadeIn(l_yb), run_time=0.7)
        self.add(arC)
        self.play(GrowArrow(arC), FadeIn(l_c), FadeIn(r1), run_time=0.8)
        self.hold(0.3)

        # ---- the parallelogram on c and b, between two lines parallel to b
        pab_ol = Polygon(*[P(p) for p in Pab], stroke_color=cA, stroke_width=2.5)
        pcb = mk([P(p) for p in Pcb], YELLOW_E, 0.55, stroke_width=2.5)
        l_dcb = tag("det(c, b)", 24)
        _fit_inside(l_dcb, [[P(p) for p in Pcb], [P(p) for p in Pxb]], 0.08)
        self.add(pcb)
        self.bring_to_back(pcb)
        self.play(FadeOut(pab), FadeOut(l_dab), FadeIn(pab_ol), FadeIn(pcb), FadeIn(l_dcb),
                  run_time=1.0)
        par1 = DashedLine(P(-0.25 * bm), P(1.45 * bm), color=GREY_A, stroke_width=2.5,
                          dash_length=0.09)
        par2 = DashedLine(P(x * am - 0.25 * bm), P(cm + 1.3 * bm), color=GREY_A,
                          stroke_width=2.5, dash_length=0.09)
        self.play(Create(par1), Create(par2), run_time=0.8)

        # ---- shear: the far side slides along its line by −y·b
        def shear(m, al):
            v = -al * y * bm
            m.set_points_as_corners([P((0, 0)), P(cm + v), P(cm + bm + v), P(bm), P((0, 0))])

        self.play(UpdateFromAlphaFunc(pcb, shear), FadeOut(seg_yb), FadeOut(l_yb),
                  run_time=2.0)
        check(_same_poly(pcb.get_vertices()[:4], [P(p) for p in Pxb], 1e-6),
              "sheared onto the parallelogram on x·a and b")
        r2 = _ctag("det(c, b) = det(x·a, b)", 26, t2c={"c": cC, "x·a": cA})
        r2.move_to([4.1, 1.75, 0])
        self.play(FadeOut(par1), FadeOut(par2), FadeIn(r2), run_time=0.8)
        self.hold(0.3)

        # ---- copies of the parallelogram on a, b, slid along a, fill it
        c1 = mk([P(p) for p in Pab], BLUE_D, 0.75, stroke_width=1.5)
        c2 = c1.copy().set_fill(TEAL_D, 0.75)
        c3 = c1.copy()
        self.play(FadeOut(l_dcb), pcb.animate.set_fill(opacity=0.12), FadeIn(c1),
                  FadeIn(l_dab), run_time=0.6)
        self.add(c2)
        self.play(c2.animate.shift(k * to3(am)), run_time=0.9)
        self.add(c3)
        self.play(c3.animate.shift(2 * k * to3(am)), run_time=0.9)
        check(close(c2.get_vertices()[0], P(am), 1e-6) and close(c3.get_vertices()[0],
                                                                 P(2 * am), 1e-6),
              "the copies slid by a and 2a")
        cut = Line(P(x * am), P(x * am + bm), color=WHITE, stroke_width=3)
        piece = mk([P(p) for p in part], BLUE_D, 0.75, stroke_width=1.5)
        c3_ol = DashedVMobject(Polygon(*[P(p) for p in sh(Pab, 2 * am)], stroke_color=GREY_B,
                                       stroke_width=2), num_dashes=40)
        self.play(Create(cut), run_time=0.4)
        self.add(piece)
        self.play(FadeOut(c3), FadeIn(c3_ol), FadeIn(piece), run_time=0.8)
        self.play(FadeOut(c3_ol), run_time=0.4)
        self.bring_to_front(arA, arB, arC, seg_xa, dotO, l_dab)
        r3 = _ctag("= x · det(a, b)", 26, t2c={"x": cA})
        r3.next_to(r2, DOWN, buff=0.35)
        r3.shift(RIGHT * (r2[8].get_left()[0] - r3[0].get_left()[0]))
        self.play(FadeIn(r3), run_time=0.7)
        fr = _frac(_ctag("det(c, b)", 28, t2c={"c": cC}), tag("det(a, b)", 28))
        r4 = _row(tag("x  =", 30, cA), fr, buff=0.25).move_to([4.1, -0.9, 0])
        self.play(FadeIn(r4), run_time=0.8)

        segs = (_segs(*[P(p) for p in Pxb], closed=True) + _segs(*[P(p) for p in Pcb], closed=True)
                + _segs(*[P(p) for p in Pab], closed=True) + _segs(P((0, 0)), P(cm))
                + _segs(P(x * am), P(cm)) + _segs(*[P(p) for p in sh(Pab, am)], closed=True)
                + _segs(*[P(p) for p in sh(Pab, 2 * am)], closed=True)
                + _segs(P(-0.25 * bm), P(1.45 * bm)) + _segs(P(x * am - 0.25 * bm), P(cm + 1.3 * bm)))
        _labels_ok([l_a, l_b, l_xa, l_c, l_dab], segs, "L5 figure")
        _labels_ok([l_yb], _segs(*[P(p) for p in Pcb], closed=True) + _segs(P(x * am), P(cm))
                   + _segs(P(x * am - 0.25 * bm), P(cm + 1.3 * bm)), "L5 y·b")
        _labels_ok([r1, r2, r3, r4], [], "L5 rows")
        for r in (r1, r2, r3, r4):
            check(r.get_left()[0] > P(cm + bm)[0] + 0.3, "rows right of the figure")
        _safe(arA, arB, arC, pcb, *[l_a, l_b, l_xa, l_c], r1, r2, r3, r4)
        cap = _cap(_row(tag("x  =  det(c, b) / det(a, b)", 36, YELLOW_B), buff=0.1))
        _below(cap, arA, pcb, l_a, l_xa)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== L10  areas scale by det T

class L10_LinearMapArea(Board):
    """Lay a grid of squares of side h over a region S: the squares inside
    S (dark) and those crossing its edge (light) trap its area between two
    counts of h².  A linear map T carries the grid to a grid of congruent
    parallelograms, each of area det T·h² (the unit square goes to the
    parallelogram on T's columns), and carries inside to inside and edge
    to edge: the same counts trap the area of T(S) between det T times the
    two bounds.  As h shrinks the light band vanishes on both sides, so
    area T(S) = |det T|·area S.  (Drawn with det T > 0; a map with det T < 0
    also flips the picture over, and |det T| is the factor.)"""

    def construct(self):
        T = np.array([[1.3, 0.5], [0.2, 1.1]])
        dT = float(np.linalg.det(T))
        s, rho, H = 0.95, 1.55, 2.0
        Lc, Rc = np.array([-4.6, 0.25, 0.0]), np.array([2.9, 0.25, 0.0])
        check(dT > 0 and all(np.linalg.det((1 - al) * np.eye(2) + al * T) > 0
                             for al in np.linspace(0, 1, 101)),
              "det T > 0 and the in-between maps never fold the plane")

        def Lp(p):
            return Lc + s * to3(p)

        def Rp(p):
            return Rc + s * to3(T @ np.asarray(p, float))

        def cells(h):
            n = int(round(2 * H / h))
            inner, edge = [], []
            for i in range(n):
                for j in range(n):
                    x0, y0 = -H + i * h, -H + j * h
                    sq = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
                    if all(np.hypot(*c) <= rho for c in sq):
                        inner.append(sq)
                    else:
                        cx, cy = min(max(0.0, x0), x0 + h), min(max(0.0, y0), y0 + h)
                        if np.hypot(cx, cy) < rho:
                            edge.append(sq)
            return inner, edge

        disc_area = PI * rho * rho
        bounds = []
        for h in (1.0, 0.5, 0.25, 0.125):
            inner, edge = cells(h)
            lo, hi = len(inner) * h * h, (len(inner) + len(edge)) * h * h
            check(lo <= disc_area <= hi, "inside cells ≤ area S ≤ inside + edge cells")
            img = [[T @ np.array(c) for c in sq] for sq in inner[:5] + edge[:5]]
            check(all(abs(abs(area(q)) - dT * h * h) < 1e-12 for q in img),
                  "every image cell has area det T·h²")
            bounds.append((lo, hi))
        check(bounds[-1][1] - bounds[-1][0] < bounds[0][1] - bounds[0][0],
              "the trap closes as h shrinks")
        ell = [T @ np.array([rho * np.cos(t), rho * np.sin(t)]) for t in np.linspace(0, TAU, 200)]
        ell_fine = [T @ np.array([rho * np.cos(t), rho * np.sin(t)])
                    for t in np.linspace(0, TAU, 4001)[:-1]]
        check(abs(abs(area(ell_fine)) - dT * disc_area) < 1e-4, "area T(S) = det T·area S")

        cIn, cEd = BLUE_D, BLUE_A

        def panel(h, f, lw=1.2):
            inner, edge = cells(h)
            g = VGroup()
            g.add(VGroup(*[mk([f(c) for c in sq], cEd, 0.28, stroke_width=lw,
                              stroke_color=GREY_B) for sq in edge]))
            g.add(VGroup(*[mk([f(c) for c in sq], cIn, 0.85, stroke_width=lw,
                              stroke_color=GREY_B) for sq in inner]))
            return g

        circ = Circle(radius=s * rho, color=YELLOW_B, stroke_width=4).move_to(Lc)
        ellip = _curve([Rc + s * to3(q) for q in ell], YELLOW_B, 4)
        lS = tag("S", 32, YELLOW_B).move_to(Lc + np.array([0.0, s * H + 0.45, 0.0]))
        lTS = tag("T(S)", 32, YELLOW_B).move_to([Rc[0] + 0.3, 3.35, 0.0])
        arr = Arrow([-2.35, 0.25, 0], [0.05, 0.25, 0], buff=0.0, color=WHITE, stroke_width=5,
                    max_tip_length_to_length_ratio=0.15)
        lT = tag("T", 32).next_to(arr, UP, buff=0.12)

        # ---- the grid on S: inside cells and edge cells trap its area
        h = 1.0
        left = panel(h, Lp)
        self.play(Create(circ), FadeIn(lS), run_time=0.8)
        self.play(FadeIn(left), run_time=1.0)
        self.bring_to_front(circ)
        one = mk([Lp(c) for c in [(0, 0), (1, 0), (1, 1), (0, 1)]], ORANGE, 0.85,
                 stroke_width=2)
        l_one = tag("1", 28).move_to(Lp((0.5, 0.5)))
        self.play(FadeIn(one), FadeIn(l_one), run_time=0.6)
        self.hold(0.3)

        # ---- T carries the grid, the cells and S across (a linear map)
        mover = VGroup(left.copy(), one.copy(), circ.copy())
        start = mover.copy()

        def upd(m, al):
            M = (1 - al) * np.eye(2) + al * T
            m.become(start.copy().apply_matrix(M, about_point=Lc).shift(al * (Rc - Lc)))

        self.play(GrowArrow(arr), FadeIn(lT), run_time=0.6)
        self.play(UpdateFromAlphaFunc(mover, upd), run_time=2.4)
        check(_landed(mover[1], [Rp(c) for c in [(0, 0), (1, 0), (1, 1), (0, 1)]], 1e-6),
              "the unit cell lands on the parallelogram on T's columns")
        right = panel(h, Rp)
        one_R = mk([Rp(c) for c in [(0, 0), (1, 0), (1, 1), (0, 1)]], ORANGE, 0.85,
                   stroke_width=2)
        self.remove(mover)
        self.add(right, one_R, ellip)
        l_det = tag("det T", 24).move_to(Rp((0.5, 0.5)))
        check(_inside(l_det, [Rp(c) for c in [(0, 0), (1, 0), (1, 1), (0, 1)]], 0.04),
              "det T inside the image of the unit cell")
        self.play(FadeIn(l_det), FadeIn(lTS), run_time=0.7)
        self.hold(0.8)

        # ---- finer grids: the light band thins on both sides alike
        cur_L, cur_R = left, right
        for h in (0.5, 0.25, 0.125):
            lw = 1.0 if h > 0.2 else 0.5
            nL, nR = panel(h, Lp, lw), panel(h, Rp, lw)
            extra = [FadeOut(one), FadeOut(l_one), FadeOut(one_R), FadeOut(l_det)] if h == 0.5 else []
            self.play(FadeOut(cur_L), FadeOut(cur_R), FadeIn(nL), FadeIn(nR), *extra,
                      run_time=0.9)
            self.bring_to_front(circ, ellip)
            self.hold(0.5)
            cur_L, cur_R = nL, nR

        _labels_ok([lS, lTS, lT], _segs(*[Lp(p) for p in [(-H, -H), (H, -H), (H, H), (-H, H)]],
                                       closed=True)
                   + _segs(*[Rp(p) for p in [(-H, -H), (H, -H), (H, H), (-H, H)]], closed=True)
                   + _segs(arr.get_start(), arr.get_end()), "L10")
        check(arr.get_start()[0] > Lp((H, 0))[0] + 0.1, "arrow clear of the left grid")
        yv = (0.0 - T[1, 0] * (-H)) / T[1, 1]          # the image's left edge at arrow height
        check(Rp((-H, yv))[0] > arr.get_end()[0] + 0.3, "arrow clear of the image grid")
        _safe(circ, ellip, cur_L, cur_R, lS, lTS, arr, lT)
        cap = caption("area of T(S)  =  |det T| × area of S", 34)
        _below(cap, cur_L, cur_R)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== L9  two reflections, one turn

class L9_TwoReflections(Board):
    """ℓ₁ and ℓ₂ meet at O at the angle θ.  Fold the figure F over ℓ₁: F′;
    fold F′ over ℓ₂: F″.  O lies on both mirrors, so every point keeps its
    distance from O; a point X at angle α below ℓ₁ goes to X′ at α above
    it, which is β = θ − α below ℓ₂, and then to X″ at β above ℓ₂.  So X
    turns about O through α + α + β + β = 2θ, and F turned rigidly through
    2θ about O lands exactly on F″ (the double fold also keeps F's
    handedness).  (Drawn with X′ between the mirrors; otherwise the same
    count with signed angles.)"""

    def construct(self):
        O = np.array([-5.4, -1.75, 0.0])
        L1, th = 13 * DEGREES, 40 * DEGREES
        L2 = L1 + th
        r0, sF = 3.75, 0.82
        al = 25 * DEGREES
        be = th - al
        u1, u2 = _polar(L1), _polar(L2)
        X = O + r0 * _polar(L1 - al)
        e1, e2 = _polar(L1 - al), _polar(L1 - al + PI / 2)
        Floc = [(0, 0), (0.25, 0), (0.25, 0.45), (0.6, 0.45), (0.6, 0.7), (0.25, 0.7),
                (0.25, 1.0), (0.75, 1.0), (0.75, 1.25), (0, 1.25)]
        Fpts = [X + sF * (u * e1 + v * e2) for u, v in Floc]
        F1 = [_mirror(p, O, O + u1) for p in Fpts]
        F2 = [_mirror(p, O, O + u2) for p in F1]
        Frot = [_rot2(p, O, 2 * th) for p in Fpts]
        X1, X2 = F1[0], F2[0]
        check(all(close(a, b) for a, b in zip(F2, Frot)),
              "fold over ℓ₁ then ℓ₂ = turn through 2θ about O, point by point")
        check(abs(np.linalg.norm(X1 - O) - r0) < 1e-12 and abs(np.linalg.norm(X2 - O) - r0)
              < 1e-12, "the folds keep the distance from O")
        check(abs(_ang(O, X, O + u1) - al) < 1e-12 and abs(_ang(O, O + u1, X1) - al) < 1e-12
              and abs(_ang(O, X1, O + u2) - be) < 1e-12 and abs(_ang(O, O + u2, X2) - be)
              < 1e-12, "α, α, β, β with α + β = θ")
        check(abs(_ang(O, X, X2) - 2 * th) < 1e-12, "X to X″ is 2θ")
        check(np.sign(area(F2)) == np.sign(area(Fpts)) != np.sign(area(F1)),
              "one fold flips the figure, two folds restore its handedness")
        # F, F′, F″ each stay inside their own sector
        angs = lambda P_: [float(np.arctan2(*(p - O)[1::-1])) for p in P_]
        check(max(angs(Fpts)) < L1 - 0.05 and min(angs(F1)) > L1 + 0.05
              and max(angs(F1)) < L2 - 0.05 and min(angs(F2)) > L2 + 0.05,
              "F below ℓ₁, F′ between the mirrors, F″ beyond ℓ₂")

        # ---- the two mirrors through O, at the angle θ
        m1 = Line(O - 0.9 * u1, O + 6.6 * u1, color=GREY_A, stroke_width=3)
        m2 = Line(O - 0.9 * u2, O + 5.75 * u2, color=GREY_A, stroke_width=3)
        l_m1 = _supline([("ℓ", 0), ("1", -1)], 30, GREY_A)
        l_m1.next_to(O + 6.6 * u1, DOWN, buff=0.12)
        l_m2 = _supline([("ℓ", 0), ("2", -1)], 30, GREY_A)
        l_m2.next_to(O + 5.75 * u2, RIGHT, buff=0.14)
        dotO = Dot(O, radius=0.07)
        l_O = tag("O", 28)
        _park_dirs(l_O, O, [285, 300, 270, 120, 140], _segs(O - 0.9 * u1, O + 6.6 * u1)
                   + _segs(O - 0.9 * u2, O + 5.75 * u2), d0=0.3, d1=0.8)
        a_th = angle_arc(O, O + u1, O + u2, radius=0.85, color=WHITE, width=3)
        l_th = tag("θ", 28)
        rays = (_segs(O - 0.9 * u1, O + 6.6 * u1) + _segs(O - 0.9 * u2, O + 5.75 * u2)
                + _segs(O, X) + _segs(O, X1) + _segs(O, X2)
                + _arc_segs(O, 0.85, L1, L2, 10) + _arc_segs(O, 1.75, L1 - al, L2 + be, 40))
        _angle_label(l_th, O, O + u1, O + u2, 1.0, r1=1.5, segs=rays,
                     fracs=(0.3, 0.25, 0.35, 0.2, 0.5))
        self.play(Create(m1), Create(m2), FadeIn(dotO), FadeIn(l_O), FadeIn(l_m1),
                  FadeIn(l_m2), run_time=1.1)
        self.play(Create(a_th), FadeIn(l_th), run_time=0.6)

        # ---- the figure F and its point X
        cF, cF1, cF2 = BLUE_D, TEAL_D, ORANGE
        fig = mk(Fpts, cF, 0.85, stroke_width=2)
        rX = DashedLine(O, X, color=GREY_B, stroke_width=2, dash_length=0.08)
        dX = Dot(X, radius=0.06)
        l_F = tag("F", 30, BLUE_B)
        self.play(FadeIn(fig), Create(rX), FadeIn(dX), run_time=0.8)

        # ---- fold over ℓ₁, then over ℓ₂
        cp1 = fig.copy().set_fill(cF1, 0.85)
        self.add(cp1)
        self.play(Rotate(cp1, angle=PI, axis=u1, about_point=O), run_time=1.5)
        check(_landed(cp1, F1), "folded over ℓ₁: F′")
        rX1 = DashedLine(O, X1, color=GREY_B, stroke_width=2, dash_length=0.08)
        dX1 = Dot(X1, radius=0.06)
        R1 = 1.75
        a_a1 = angle_arc(O, X, O + u1, radius=R1, color=BLUE_B, width=4)
        a_a2 = angle_arc(O, O + u1, X1, radius=R1, color=BLUE_B, width=4)
        self.play(Create(rX1), FadeIn(dX1), Create(a_a1), Create(a_a2), run_time=0.8)
        cp2 = cp1.copy().set_fill(cF2, 0.85)
        self.add(cp2)
        self.play(Rotate(cp2, angle=PI, axis=u2, about_point=O), run_time=1.5)
        check(_landed(cp2, F2), "folded over ℓ₂: F″")
        rX2 = DashedLine(O, X2, color=GREY_B, stroke_width=2, dash_length=0.08)
        dX2 = Dot(X2, radius=0.06)
        a_b1 = angle_arc(O, X1, O + u2, radius=R1, color=GREEN_B, width=4)
        a_b2 = angle_arc(O, O + u2, X2, radius=R1, color=GREEN_B, width=4)
        self.play(Create(rX2), FadeIn(dX2), Create(a_b1), Create(a_b2), run_time=0.8)

        segs = (_segs(O - 0.9 * u1, O + 6.6 * u1) + _segs(O - 0.9 * u2, O + 5.75 * u2)
                + _segs(O, X) + _segs(O, X1) + _segs(O, X2)
                + _arc_segs(O, 0.85, L1, L2, 10) + _arc_segs(O, R1, L1 - al, L2 + be, 40))
        for P_ in (Fpts, F1, F2):
            segs += _segs(*P_, closed=True)
        lab_a1, lab_a2 = tag("α", 28, BLUE_B), tag("α", 28, BLUE_B)
        lab_b1, lab_b2 = tag("β", 28, GREEN_B), tag("β", 28, GREEN_B)
        placed = [l_th, l_O]
        for lab, (P_, Q_) in ((lab_a1, (X, O + u1)), (lab_a2, (O + u1, X1)),
                              (lab_b1, (X1, O + u2)), (lab_b2, (O + u2, X2))):
            _angle_label(lab, O, P_, Q_, R1 + 0.25, segs=segs, avoid=placed)
            placed.append(lab)
        self.play(FadeIn(lab_a1), FadeIn(lab_a2), FadeIn(lab_b1), FadeIn(lab_b2),
                  run_time=0.6)
        self.hold(0.4)

        # ---- F turned through 2θ about O lands on F″
        cp3 = fig.copy().set_fill(cF, 0.6)
        self.add(cp3)
        self.play(Rotate(cp3, angle=2 * th, about_point=O), run_time=2.0)
        check(_landed(cp3, F2, 1e-6), "F turned through 2θ about O is F″")
        self.play(Indicate(VGroup(cp2, cp3), color=YELLOW_B, scale_factor=1.0), run_time=0.8)
        self.remove(cp3)
        R2 = 2.55
        a_2t = angle_arc(O, X, X2, radius=R2, color=YELLOW_B, width=4)
        l_2t = tag("2θ", 30, YELLOW_B)
        segs2 = segs + _arc_segs(O, R2, L1 - al, L2 + be, 40)
        _angle_label(l_2t, O, X, X2, R2 + 0.25, segs=segs2, avoid=placed, fracs=(0.5, 0.6, 0.4))
        tks = VGroup(*[_ticks(O, P_, 1, at=0.85) for P_ in (X, X1, X2)])
        check(all(abs(0.85 * r0 - R) > 0.35 for R in (R1, R2)), "ticks clear of the arcs")
        self.play(Create(a_2t), FadeIn(l_2t), Create(tks), run_time=0.8)
        l_F.next_to(fig, DOWN, buff=0.12)
        l_F1 = _supline([("F′", 0)], 30, TEAL_B)
        l_F2 = _supline([("F″", 0)], 30, ORANGE)
        for P_, lab in ((F1, l_F1), (F2, l_F2)):
            c_ = np.mean(np.array(P_), axis=0)
            lab.move_to(c_ + 0.95 * _unit(c_ - O))
        self.play(FadeIn(l_F), FadeIn(l_F1), FadeIn(l_F2), run_time=0.6)

        row0 = _ctag("α + β = θ", 30, t2c={"α": BLUE_B, "β": GREEN_B})
        row = _ctag("α + α + β + β = 2θ", 30, t2c={"α": BLUE_B, "β": GREEN_B, "2θ": YELLOW_B})
        row0.move_to([3.3, 2.95, 0])
        row.move_to([3.3, 2.05, 0])
        self.play(FadeIn(row0), run_time=0.6)
        self.play(FadeIn(row), run_time=0.8)

        all_segs = segs2 + [s_ for t_ in tks for s_ in _mob_segs(t_)]
        _labels_ok([l_m1, l_m2, l_O, l_th, lab_a1, lab_a2, lab_b1, lab_b2, l_2t, l_F, l_F1, l_F2],
                   all_segs, "L9")
        _labels_ok([row0, row, l_m1, l_m2], [], "L9 row")
        _safe(m1, m2, fig, cp1, cp2, row0, row)
        cap = _cap(_supline([("reflection in ℓ", 0), ("1", -1), (", then in ℓ", 0), ("2", -1),
                             ("    =    rotation through 2θ", 0)], 32))
        _below(cap, m1, m2, fig, l_O)
        self.play(Write(cap), run_time=1.1)
        self.hold(2.2)


# ========================================================== L13  the rotation matrix

class L13_RotationMatrix(Board):
    """Turn the unit square through θ about O (a rigid turn).  Its first side
    e₁ goes to the hypotenuse of a right triangle with legs cos θ along the
    x-axis and sin θ up: R_θe₁ = (cos θ, sin θ).  Its second side e₂ is the
    first turned a quarter, so R_θe₂ is the hypotenuse of the same triangle
    turned a quarter: legs cos θ up the y-axis and sin θ to the left,
    R_θe₂ = (−sin θ, cos θ).  These are the columns of the matrix R_θ."""

    def construct(self):
        th = 32 * DEGREES
        c, s = float(np.cos(th)), float(np.sin(th))
        k = 3.4
        O = np.array([-3.95, -1.76, 0.0])

        def S(x, y):
            return O + k * np.array([x, y, 0.0])

        sq = [S(0, 0), S(1, 0), S(1, 1), S(0, 1)]
        rsq = [S(0, 0), S(c, s), S(c - s, s + c), S(-s, c)]
        T1 = [S(0, 0), S(c, 0), S(c, s)]
        T2 = [S(0, 0), S(0, c), S(-s, c)]
        check(all(close(_rot2(p, O, th), q) for p, q in zip(sq, rsq)), "the square turned by θ")
        check(all(close(_rot2(p, O, PI / 2), q) for p, q in zip(T1, T2)),
              "the e₁-triangle turned a quarter is the e₂-triangle")
        check(close(rsq[3], _rot2(rsq[1], O, PI / 2)), "R e₂ is R e₁ turned a quarter")

        cA, cB = BLUE_B, ORANGE
        axes = VGroup(Line(S(-0.75, 0), S(1.25, 0), color=GREY_B, stroke_width=2),
                      Line(S(0, -0.3), S(0, 1.5), color=GREY_B, stroke_width=2))
        l_xax = tag("x", 26, GREY_A).next_to(S(1.25, 0), RIGHT, buff=0.1)
        l_yax = tag("y", 26, GREY_A).next_to(S(0, 1.5), UP, buff=0.1)
        sqm = mk(sq, GREY_D, 0.5, stroke_width=2)
        e1 = _arrow(S(0, 0), S(1, 0), cA, 6)
        e2 = _arrow(S(0, 0), S(0, 1), cB, 6)
        l_e1 = _supline([("e", 0), ("1", -1)], 30, cA).next_to(S(1, 0), DOWN, buff=0.14)
        l_e2 = _supline([("e", 0), ("2", -1)], 30, cB).next_to(S(0, 1), LEFT, buff=0.14)
        dotO = Dot(O, radius=0.06)
        self.play(Create(axes), FadeIn(l_xax), FadeIn(l_yax), FadeIn(dotO), run_time=0.8)
        self.add(sqm)
        self.bring_to_back(sqm)
        self.play(FadeIn(sqm), GrowArrow(e1), GrowArrow(e2), FadeIn(l_e1), FadeIn(l_e2),
                  run_time=1.0)
        self.hold(0.3)

        # ---- turn the square through θ (rigid)
        turned = VGroup(mk(sq, GREY_C, 0.35, stroke_width=2), _arrow(S(0, 0), S(1, 0), cA, 6),
                        _arrow(S(0, 0), S(0, 1), cB, 6))
        ghost = DashedVMobject(Polygon(*sq, stroke_color=GREY_B, stroke_width=2), num_dashes=40)
        self.add(turned)
        self.play(FadeOut(sqm), FadeIn(ghost), Rotate(turned, angle=th, about_point=O),
                  run_time=2.0)
        check(_landed(turned[0], rsq, 1e-6), "the turned square")
        a1 = angle_arc(O, S(1, 0), S(c, s), radius=0.95, color=WHITE, width=3)
        a2 = angle_arc(O, S(0, 1), S(-s, c), radius=0.95, color=WHITE, width=3)
        l_a1, l_a2 = tag("θ", 26), tag("θ", 26)
        segs = (_segs(S(-0.75, 0), S(1.25, 0)) + _segs(S(0, -0.3), S(0, 1.5))
                + _segs(*sq, closed=True) + _segs(*rsq, closed=True)
                + _segs(*T1, closed=True) + _segs(*T2, closed=True)
                + _arc_segs(O, 0.95, 0, th, 10) + _arc_segs(O, 0.95, PI / 2, PI / 2 + th, 10))
        _angle_label(l_a1, O, S(1, 0), S(c, s), 1.12, segs=segs)
        _angle_label(l_a2, O, S(0, 1), S(-s, c), 1.12, segs=segs)
        self.play(Create(a1), Create(a2), FadeIn(l_a1), FadeIn(l_a2), run_time=0.7)

        # ---- R e₁: legs cos θ (along x) and sin θ (up)
        t1 = mk(T1, BLUE_D, 0.7, stroke_width=0)
        drop1 = DashedLine(S(c, s), S(c, 0), color=GREY_A, stroke_width=2.5, dash_length=0.08)
        leg1h = Line(S(0, 0), S(c, 0), color=cA, stroke_width=7)
        leg1v = Line(S(c, 0), S(c, s), color=cA, stroke_width=4)
        self.add(t1)
        self.bring_to_back(t1)
        self.bring_to_back(ghost)
        self.play(FadeIn(t1), Create(drop1), run_time=0.8)
        l_c1 = tag("cos θ", 26, cA)
        _beside(l_c1, S(0, 0), S(c, 0), DOWN, gap=0.14, at=0.5)
        l_s1 = tag("sin θ", 26, cA)
        _fit_inside(l_s1, [[S(c * 0.45, 0), S(c, 0), S(c, s), S(c * 0.45, s * 0.45)]], 0.07)
        self.play(FadeIn(l_c1), FadeIn(l_s1), Create(leg1h), run_time=0.8)
        self.hold(0.3)

        # ---- turned a quarter, the same triangle sits under R e₂
        t2 = t1.copy()
        self.add(t2)
        self.play(Rotate(t2, angle=PI / 2, about_point=O), run_time=1.6)
        check(_landed(t2, T2, 1e-6), "the quarter-turned triangle lands under R e₂")
        leg2v = Line(S(0, 0), S(0, c), color=cB, stroke_width=7)
        l_c2 = tag("cos θ", 26, cB)
        _beside(l_c2, S(0, 0), S(0, c), RIGHT, gap=0.14, at=0.42)
        l_s2 = tag("sin θ", 26, cB)
        _fit_inside(l_s2, [[S(0, c), S(-s, c), S(-s * 0.45, c * 0.45), S(0, c * 0.45)]], 0.07)
        self.play(FadeIn(l_c2), FadeIn(l_s2), Create(leg2v),
                  t2.animate.set_fill(ORANGE, 0.55), run_time=0.8)
        self.bring_to_front(turned[1], turned[2], dotO)

        # ---- the columns
        r1 = _supline([("R", 0, WHITE), ("θ", -1, WHITE), (" e", 0, cA), ("1", -1, cA),
                       ("  =  (cos θ,  sin θ)", 0, cA)], 30)
        r2 = _supline([("R", 0, WHITE), ("θ", -1, WHITE), (" e", 0, cB), ("2", -1, cB),
                       ("  =  (−sin θ,  cos θ)", 0, cB)], 30)
        r1.move_to([3.7, 2.7, 0])
        r2.move_to([3.7, 1.6, 0]).align_to(r1, LEFT)
        self.play(FadeIn(r1), run_time=0.7)
        self.play(FadeIn(r2), run_time=0.7)
        mat = _matrix([("cos θ", "sin θ"), ("−sin θ", "cos θ")], [cA, cB], 30)
        lhs = _supline([("R", 0), ("θ", -1), ("  =", 0)], 32, WHITE)
        row_m = _row(lhs, mat, buff=0.3).move_to([3.7, -0.6, 0])
        self.play(FadeIn(row_m), run_time=0.9)

        labels = [l_xax, l_yax, l_e1, l_e2, l_c1, l_s1, l_c2, l_s2, l_a1, l_a2]
        _labels_ok(labels, segs, "L13")
        _labels_ok([r1, r2, row_m], [], "L13 rows")
        for r in (r1, r2, row_m):
            check(r.get_left()[0] > S(1.25, 0)[0] + 0.3, "panel right of the figure")
        _safe(axes, ghost, turned, r1, r2, row_m, *labels)
        cap = _cap(_supline([("R", 0), ("θ", -1), (" e", 0), ("1", -1),
                             ("  =  (cos θ, sin θ),     R", 0), ("θ", -1), (" e", 0), ("2", -1),
                             ("  =  (−sin θ, cos θ)", 0)], 32))
        _below(cap, axes, ghost, *labels)
        self.play(Write(cap), run_time=1.0)
        self.hold(2.2)


# ========================================================== L14  composing rotations

def _colvec(top, bot, color=WHITE, size=26):
    """A one-column vector [top; bot] with drawn brackets."""
    return _matrix([(top, bot)], [color], size, hgap=0.0, vgap=0.16)


class L14_ComposingRotations(Board):
    """Turn e₁ through A, then through B: it lands where one turn through
    A + B takes it (rigid turns about O add their angles), so
    R_B R_A = R_{A+B}.  Read that landing point two ways.  Turned through A,
    e₁ is u = cos A·e₁ + sin A·e₂, the diagonal of the cos A by sin A
    rectangle; turning the whole frame through B turns that rectangle
    rigidly, so R_B u = cos A·R_Be₁ + sin A·R_Be₂, where R_Be₁ = (cos B, sin B)
    and R_Be₂ = (−sin B, cos B) are the columns of R_B.  Read directly it
    is (cos(A + B), sin(A + B)): hence cos(A + B) = cos A cos B − sin A sin B
    and sin(A + B) = sin A cos B + cos A sin B."""

    def construct(self):
        A, B = 25 * DEGREES, 35 * DEGREES
        R = 3.75
        O = np.array([-3.75, -2.5, 0.0])

        def Pp(x, y):
            return O + R * np.array([x, y, 0.0])

        ca, sa, cb, sb = (float(np.cos(A)), float(np.sin(A)), float(np.cos(B)),
                          float(np.sin(B)))
        E1, E2 = Pp(1, 0), Pp(0, 1)
        U, W = Pp(ca, sa), Pp(np.cos(A + B), np.sin(A + B))
        RE1, RE2 = Pp(cb, sb), Pp(-sb, cb)
        K, L_ = Pp(ca * cb, ca * sb), Pp(-sa * sb, sa * cb)
        check(close(_rot2(_rot2(E1, O, A), O, B), _rot2(E1, O, A + B)),
              "turn by A then by B = turn by A + B")
        check(close(O + ca * (RE1 - O) + sa * (RE2 - O), W),
              "R_B u = cos A·R_Be₁ + sin A·R_Be₂ lands on w")
        check(abs(np.cos(A + B) - (ca * cb - sa * sb)) < 1e-12
              and abs(np.sin(A + B) - (sa * cb + ca * sb)) < 1e-12, "the addition formulas")
        check(close(_rot2(Pp(ca, 0), O, B), K) and close(_rot2(Pp(0, sa), O, B), L_),
              "the u-rectangle turned by B has corners cos A·R_Be₁, sin A·R_Be₂")

        cA, cB, cU, cW = BLUE_B, ORANGE, WHITE, YELLOW_B
        segs = (_segs(Pp(-0.62, 0), Pp(1.12, 0)) + _segs(Pp(0, -0.06), Pp(0, 1.12))
                + _arc_segs(O, R, 0.0, PI / 2 + B + 0.05, 80)
                + _segs(O, E1) + _segs(O, E2) + _segs(O, U) + _segs(O, W)
                + _segs(O, RE1) + _segs(O, RE2) + _segs(O, K, W)
                + _segs(O, Pp(ca, 0), U) + _arc_segs(O, 0.7, 0, A, 8)
                + _arc_segs(O, 1.1, A, A + B, 10))
        axes = VGroup(Line(Pp(-0.62, 0), Pp(1.12, 0), color=GREY_B, stroke_width=2),
                      Line(Pp(0, -0.06), Pp(0, 1.12), color=GREY_B, stroke_width=2))
        arc = Arc(radius=R, start_angle=0.0, angle=PI / 2 + B + 0.05, arc_center=O,
                  color=GREY_D, stroke_width=2)
        e1 = _arrow(O, E1, cA, 5)
        e2 = _arrow(O, E2, cB, 5)
        l_e1 = _supline([("e", 0), ("1", -1)], 28, cA).next_to(E1, DOWN, buff=0.14)
        l_e2 = _supline([("e", 0), ("2", -1)], 28, cB)
        _park_dirs(l_e2, E2, [60, 30, 120, 150], segs, d0=0.25, d1=0.9)
        dotO = Dot(O, radius=0.06)
        self.play(Create(axes), Create(arc), FadeIn(dotO), GrowArrow(e1), GrowArrow(e2),
                  FadeIn(l_e1), FadeIn(l_e2), run_time=1.1)

        # ---- turn through A, then through B; one turn through A + B lands there too
        u = _arrow(O, E1, cU, 5)
        self.add(u)
        self.play(Rotate(u, angle=A, about_point=O), run_time=1.2)
        check(close(u.get_end(), U, 1e-6), "e₁ turned by A is u")
        aA = angle_arc(O, E1, U, radius=0.7, color=cU, width=3)
        lA = tag("A", 26, cU)
        _angle_label(lA, O, E1, U, 0.85, segs=segs)
        l_u = tag("u", 28, cU)
        _park_dirs(l_u, U, [25, 5, 45, -15], segs, d0=0.25, d1=0.9)
        self.play(Create(aA), FadeIn(lA), FadeIn(l_u), run_time=0.5)
        w = u.copy().set_color(cW)
        self.add(w)
        self.play(Rotate(w, angle=B, about_point=O), run_time=1.2)
        check(close(w.get_end(), W, 1e-6), "u turned by B is w")
        aB = angle_arc(O, U, W, radius=1.1, color=cW, width=3)
        lB = tag("B", 26, cW)
        _angle_label(lB, O, U, W, 1.25, segs=segs, fracs=(0.7, 0.75, 0.65, 0.8, 0.6))
        self.play(Create(aB), FadeIn(lB), run_time=0.5)
        gh = _arrow(O, E1, GREY_A, 4)
        self.add(gh)
        self.play(Rotate(gh, angle=A + B, about_point=O), run_time=1.5)
        check(close(gh.get_end(), W, 1e-6), "e₁ turned by A + B is w")
        self.play(Indicate(VGroup(w, gh), color=cW, scale_factor=1.0), run_time=0.7)
        self.remove(gh)
        r0 = _supline([("R", 0), ("B", -1), ("R", 0), ("A", -1), (" e", 0), ("1", -1),
                       ("  =  R", 0), ("A+B", -1), (" e", 0), ("1", -1)], 30, WHITE)
        r0.move_to([3.2, 3.2, 0]).align_to(np.array([0.3, 0, 0]), LEFT)
        self.play(FadeIn(r0), run_time=0.6)

        # ---- u = cos A·e₁ + sin A·e₂: its rectangle, turned rigidly through B
        rect = mk([O, Pp(ca, 0), U, Pp(0, sa)], GREY_D, 0.55, stroke_width=0)
        s1 = Line(O, Pp(ca, 0), color=cA, stroke_width=7)
        s2 = Line(Pp(ca, 0), U, color=cB, stroke_width=7)
        l_ca = tag("cos A", 24, cA)
        _beside(l_ca, O, Pp(ca, 0), DOWN, gap=0.14, at=0.5)
        l_sa = tag("sin A", 24, cB)
        _beside(l_sa, Pp(ca, 0), U, LEFT, gap=0.12, at=0.42)
        self.add(rect)
        self.bring_to_back(rect)
        self.play(FadeIn(rect), Create(s1), Create(s2), FadeIn(l_ca), FadeIn(l_sa),
                  run_time=1.0)
        self.hold(0.3)
        frame = VGroup(rect.copy(), Line(O, Pp(ca, 0), color=cA, stroke_width=7),
                       Line(Pp(ca, 0), U, color=cB, stroke_width=7),
                       _arrow(O, E1, cA, 4), _arrow(O, E2, cB, 4))
        self.add(frame)
        self.bring_to_front(w, u)
        self.play(FadeOut(l_ca), FadeOut(l_sa), rect.animate.set_fill(opacity=0.2),
                  Rotate(frame, angle=B, about_point=O), run_time=2.0)
        check(_landed(frame[0], [O, K, W, L_], 1e-6), "the u-rectangle turned by B")
        check(close(frame[1].get_end(), K, 1e-6) and close(frame[2].get_end(), W, 1e-6)
              and close(frame[3].get_end(), RE1, 1e-6) and close(frame[4].get_end(), RE2, 1e-6),
              "the turned frame: legs cos A·R_Be₁ and sin A·R_Be₂ end at w")
        self.bring_to_front(w)
        l_ca2 = tag("cos A", 24, cA)
        _beside(l_ca2, O, K, np.array([-np.sin(B), np.cos(B), 0]), gap=0.13, at=0.7)
        l_sa2 = tag("sin A", 24, cB)
        _beside(l_sa2, K, W, np.array([-np.cos(B), -np.sin(B), 0]), gap=0.13, at=0.42)
        check(_inside(l_ca2, [O, K, W], 0.0) and _inside(l_sa2, [O, K, W], 0.0),
              "the turned side labels sit inside the turned rectangle, on their sides")
        l_re1 = _supline([("R", 0), ("B", -1), (" e", 0), ("1", -1)], 26, cA)
        _park_dirs(l_re1, RE1, [35, 15, 55, 0, -20], segs, d0=0.3, d1=1.0)
        l_re2 = _supline([("R", 0), ("B", -1), (" e", 0), ("2", -1)], 26, cB)
        _park_dirs(l_re2, RE2, [125, 110, 140, 95, 155, 170], segs, d0=0.3, d1=0.9)
        self.play(FadeIn(l_ca2), FadeIn(l_sa2), FadeIn(l_re1), FadeIn(l_re2), run_time=0.8)

        # ---- read the landing point two ways
        r1 = _supline([("R", 0), ("B", -1), (" u  =  cos A · R", 0), ("B", -1), (" e", 0),
                       ("1", -1), ("  +  sin A · R", 0), ("B", -1), (" e", 0), ("2", -1)], 24,
                      WHITE)
        r1.next_to(r0, DOWN, buff=0.42).align_to(r0, LEFT)
        eq2 = tag("=", 24)
        r2 = _row(eq2, tag("cos A", 23, cA), _colvec("cos B", "sin B", cA, 21),
                  tag("+  sin A", 23, cB), _colvec("−sin B", "cos B", cB, 21), buff=0.12)
        r2.next_to(r1, DOWN, buff=0.3).align_to(r1, LEFT).shift(RIGHT * 0.45)
        r3 = _row(tag("=", 24),
                  _colvec("cos A cos B − sin A sin B", "sin A cos B + cos A sin B", WHITE, 21),
                  buff=0.14)
        r3.next_to(r2, DOWN, buff=0.28).align_to(r2, LEFT)
        r4 = _row(_supline([("R", 0), ("A+B", -1), (" e", 0), ("1", -1), ("  =", 0)], 24, WHITE),
                  _colvec("cos(A + B)", "sin(A + B)", cW, 22), buff=0.14)
        r4.next_to(r3, DOWN, buff=0.35).align_to(r0, LEFT)
        self.play(FadeIn(r1), run_time=0.8)
        self.play(FadeIn(r2), run_time=0.8)
        self.play(FadeIn(r3), run_time=0.8)
        self.play(FadeIn(r4), Indicate(w, color=cW, scale_factor=1.0), run_time=0.8)

        _labels_ok([l_e1, l_e2, lA, lB, l_u, l_ca2, l_sa2, l_re1, l_re2], segs, "L14 figure")
        _labels_ok([l_ca, l_sa], segs, "L14 first rectangle")
        rows = [r0, r1, r2, r3, r4]
        _labels_ok(rows, [], "L14 rows")
        _labels_ok(rows + [l_e1, l_e2, lA, lB, l_u, l_ca2, l_sa2, l_re1, l_re2], segs,
                   "L14 rows clear of the figure")
        _safe(axes, arc, *rows)
        cap = caption("cos(A + B) = cos A cos B − sin A sin B,    "
                      "sin(A + B) = sin A cos B + cos A sin B", 28)
        _below(cap, axes, r4, l_e1)
        self.play(Write(cap), run_time=1.2)
        self.hold(2.2)
