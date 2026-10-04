# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w3k.py — proofs without words, 2D (manim):
#     J24 only squares have an odd number of divisors
#     J28 the Chinese remainder theorem on a grid
#     J33 casting out nines             J25 the silver rectangle
#     J21 lattice points on a segment   J29 totients add up to n
#     J30 Farey neighbours              K22 five points in a square
#     K17 choosing two of n + 1         K27 the broken stick
#
# No LaTeX anywhere: every label is Text (tag / caption) with Unicode.
# Every scene checks its own claims with check(...) before it animates —
# counts, bijections, exact landings of folded or turned copies, tilings and
# areas of regions — and checks at the end that its labels sit inside the
# safe area, clear of each other and of the lines, so a wrong construction
# fails the render instead of drawing a wrong picture.

from math import gcd


# ------------------------------------------------------------ helpers

def _u(v):
    v = to3(v)
    return v / np.linalg.norm(v)


def _polar(a):
    """Unit screen vector at angle a (radians)."""
    return np.array([np.cos(a), np.sin(a), 0.0])


def _ang_at(v, p, q):
    """Non-reflex angle p-v-q in radians."""
    a, b = to3(p) - to3(v), to3(q) - to3(v)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


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


def _flip(mob, p, q, **kw):
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


def _verts(m):
    """Corner points of a Polygon / Rectangle mobject (screen)."""
    return [np.asarray(v, float) for v in m.get_vertices()]


# ---- text hygiene

def _inside(m, tol=1e-6):
    """True if the mobject lies inside the safe content area."""
    return (m.get_left()[0] >= -SAFE_X - tol and m.get_right()[0] <= SAFE_X + tol
            and m.get_bottom()[1] >= SAFE_BOTTOM - tol
            and m.get_top()[1] <= SAFE_TOP + tol)


def _apart(a, b, pad=0.04):
    """True if the bounding boxes of two mobjects do not meet (pad apart)."""
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


def _name(m):
    return getattr(m, "text", None) or type(m).__name__


def _labels_ok(labels, segs, what, pad=0.05, gap=0.04):
    """Every label inside the safe area, clear of every segment in segs
    and of every other label."""
    for m in labels:
        check(_inside(m), f"{what}: '{_name(m)}' inside the safe area")
        i = _hit(m, segs, pad)
        check(i is None, f"{what}: '{_name(m)}' clear of the lines (hits #{i})")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            check(_apart(labels[i], labels[j], gap),
                  f"{what}: '{_name(labels[i])}' clear of '{_name(labels[j])}'")


def _poly_segs(pts, closed=True):
    pts = [to3(p) for p in pts]
    n = len(pts)
    return [(pts[i], pts[(i + 1) % n]) for i in range(n if closed else n - 1)]


def _arc_segs(c, r, a0, a1, n=24):
    c = to3(c)
    pts = [c + r * _polar(a) for a in np.linspace(a0, a1, n + 1)]
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


def _dot_segs(p, r):
    """A dot of radius r as a small square of segments (for label checks)."""
    p = to3(p)
    c = [p + r * np.array([sx, sy, 0.0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return _poly_segs(c)


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
        check(_inside(m, 0.02), f"{type(m).__name__} inside the safe area")
    texts = [t for m in shown for t in m.get_family() if isinstance(t, Text)]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            check(_apart(texts[i], texts[j], 0.0),
                  f"labels '{texts[i].text}' and '{texts[j].text}' apart")


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


def _trow(parts, size=26, buff=0.12):
    """A row of coloured Text pieces [(s, colour), ...] on one baseline; a
    piece may carry its own gap to the next one as a third element."""
    g = VGroup()
    x = 0.0
    for part in parts:
        s, col = part[0], part[1]
        t = _on_base(s, size, col)
        t.shift(np.array([x - t.get_left()[0], 0.0, 0.0]))
        x = t.get_right()[0] + (part[2] if len(part) > 2 else buff)
        g.add(t)
    return g


def _left_at(m, x, y):
    """Move m so its left edge is at x and its centre at height y."""
    return m.move_to(np.array([x + m.width / 2, y, 0.0]))


# ===================================================================== J24

def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


class J24_OddDivisors(Board):
    """A divisor d of n is the width of a rectangle d × (n/d) of area n.
    Stand all of them in one corner: reflection in the diagonal turns the
    d × (n/d) rectangle into the (n/d) × d one, so the divisors fall into
    pairs d ↔ n/d. A rectangle is its own mirror image only if it is a
    square, d = n/d, which happens exactly when n is a square. So d(n) is
    even unless n is a square: 12 has 3 pairs, 16 has 2 pairs and 4 × 4."""

    def construct(self):
        # the claim, for many n: the pairing d ↔ n/d is an involution on the
        # divisors whose only possible fixed point is √n
        for n in range(1, 401):
            ds = _divisors(n)
            check(all(n // d in ds and n // (n // d) == d for d in ds), "d ↔ n/d")
            fixed = [d for d in ds if d * d == n]
            sq = int(round(np.sqrt(n))) ** 2 == n
            check(len(fixed) == (1 if sq else 0), "a fixed point only for squares")
            check((len(ds) % 2 == 1) == sq, "d(n) odd ⟺ n square")

        u = 0.335
        panels = [(12, np.array([-5.75, -2.3, 0.0])), (16, np.array([0.55, -2.3, 0.0]))]
        pair_cols = [BLUE_C, TEAL_C, ORANGE]
        sq_col = YELLOW
        y_text = 3.42                       # baseline row of each panel's line
        all_labels, all_segs = [], []
        for idx, (n, O) in enumerate(panels):
            def P(x, y, O=O):
                return O + u * np.array([x, y, 0.0])
            ds = _divisors(n)
            grid = VGroup()
            for i in range(n + 1):
                grid.add(Line(P(i, 0), P(i, n), color=GREY_D, stroke_width=1))
                grid.add(Line(P(0, i), P(n, i), color=GREY_D, stroke_width=1))
            axes = VGroup(Line(P(0, 0), P(n + 0.6, 0), color=GREY_B, stroke_width=2),
                          Line(P(0, 0), P(0, n + 0.3), color=GREY_B, stroke_width=2))
            title = _on_base(f"n = {n}", 30, YELLOW_B)
            title.shift(np.array([P(0, 0)[0] - title.get_left()[0], y_text, 0.0]))
            rects, dots, xl, yl = {}, {}, {}, {}
            for d in ds:
                e = n // d
                rects[d] = Polygon(P(0, 0), P(d, 0), P(d, e), P(0, e),
                                   stroke_color=WHITE, stroke_width=2.5, fill_opacity=0)
                dots[d] = Dot(P(d, e), radius=0.06, color=WHITE)
                xl[d] = tag(str(d), 18, GREY_A).move_to(P(d, 0) + DOWN * 0.22)
                yl[d] = tag(str(e), 18, GREY_A).move_to(P(0, e) + LEFT * 0.25)
                check(close(abs(area(_verts(rects[d]))), u * u * n),
                      f"the {d} × {e} rectangle has area {n}")

            self.play(FadeIn(grid), Create(axes), FadeIn(title), run_time=0.8)
            for d in ds:
                self.play(Create(rects[d]), FadeIn(dots[d]), FadeIn(xl[d]), FadeIn(yl[d]),
                          run_time=0.42)
            diag = DashedLine(P(0, 0), P(n + 0.3, n + 0.3), color=GREY_A,
                              stroke_width=2.5, dash_length=0.1)
            self.play(Create(diag), run_time=0.5)

            # the mirror in the diagonal: a copy of the whole set flips over
            ghost = VGroup(*[rects[d].copy().set_stroke(YELLOW_B, 3) for d in ds])
            self.add(ghost)
            self.play(_flip(ghost, P(0, 0), P(1, 1)), run_time=1.5)
            for k, d in enumerate(ds):
                e = n // d
                check(_same_poly(_verts(ghost[k]), [P(0, 0), P(e, 0), P(e, d), P(0, d)], 1e-6),
                      f"the {d} × {e} rectangle lands on the {e} × {d} one")
            # colour the pairs; the square (if any) pairs with itself
            anims = []
            npairs = 0
            for d in ds:
                e = n // d
                if d < e:
                    col = pair_cols[npairs % len(pair_cols)]
                    npairs += 1
                    for x in (d, e):
                        anims += [rects[x].animate.set_stroke(col, 4),
                                  dots[x].animate.set_color(col),
                                  xl[x].animate.set_color(col), yl[x].animate.set_color(col)]
                elif d == e:
                    anims += [rects[d].animate.set_stroke(sq_col, 4.5)
                              .set_fill(sq_col, 0.35),
                              dots[d].animate.set_color(sq_col).scale(1.4),
                              xl[d].animate.set_color(sq_col), yl[d].animate.set_color(sq_col)]
            self.play(*anims, FadeOut(ghost), run_time=0.8)
            if idx == 1:
                ring = Circle(radius=0.17, color=sq_col, stroke_width=3).move_to(P(4, 4))
                check(close(P(4, 4)[0] - O[0], P(4, 4)[1] - O[1]), "4 × 4 corner on the diagonal")
                self.play(Create(ring), run_time=0.5)
            single = 1 if any(d * d == n for d in ds) else 0
            check(2 * npairs + single == len(ds), "pairs + the square = all divisors")
            words = (f":  {npairs} pairs  →  {len(ds)} divisors" if not single else
                     f":  {npairs} pairs + 1  →  {len(ds)} divisors")
            count = _on_base(words, 24, WHITE)
            count.shift(np.array([title.get_right()[0] + 0.12 - count.get_left()[0],
                                  y_text, 0.0]))
            check(count.get_right()[0] < (panels[1][1][0] - 0.5 if idx == 0 else SAFE_X),
                  "the count line stays over its own panel")
            self.play(FadeIn(count, shift=LEFT * 0.1), run_time=0.6)
            self.hold(0.5)

            # hygiene for this panel
            segs = ([s for d in ds for s in _poly_segs(_verts(rects[d]))]
                    + [(diag.get_start(), diag.get_end())]
                    + [(a.get_start(), a.get_end()) for a in axes])
            labs = [title, count]
            for d in ds:
                segs += _dot_segs(dots[d].get_center(), 0.06 * (1.4 if d * d == n else 1))
            check(all(_hit(t, segs, 0.04) is None for t in (title, count)),
                  "title and count clear of the figure")
            ticks = [xl[d] for d in ds] + [yl[d] for d in ds]
            for t in ticks:
                check(_inside(t), "tick label inside")
                check(_hit(t, [(a.get_start(), a.get_end()) for a in axes]
                           + [s for d in ds for s in _poly_segs(_verts(rects[d]))], 0.02)
                      is None, f"tick label {t.text} clear of the lines")
            for i in range(len(ticks)):
                for j in range(i + 1, len(ticks)):
                    if ticks[i] is not ticks[j]:
                        check(_apart(ticks[i], ticks[j], 0.03), "tick labels apart")
            all_labels += labs
            all_segs += segs
        _labels_ok(all_labels, all_segs, "J24")
        cap = caption("divisors pair up as d ↔ n/d :   d(n) is odd   ⟺   n is a square", 30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J28

class J28_CRTGrid(Board):
    """Put k in row k mod m, column k mod n. From k to k + 1 the walk steps
    one row down and one column right, wrapping round at the edges. It is
    back in row 0 every m steps and in column 0 every n steps, so it first
    returns to its start after lcm(m, n) steps — and before that it never
    repeats a cell (a repeat at steps i < j would mean a return after j − i
    steps). With gcd(3, 5) = 1 that is 15 = 3 · 5 steps: all 15 cells, each
    once, so k ↔ (k mod 3, k mod 5) is a bijection. With 4 × 6 the walk
    closes after lcm(4, 6) = 12 steps and half the cells are never reached."""

    def construct(self):
        cs = 0.9
        grids = [(3, 5, np.array([-5.3, 2.5, 0.0])), (4, 6, np.array([0.95, 2.5, 0.0]))]
        for m, n, _ in grids:
            cells = [(k % m, k % n) for k in range(m * n)]
            L = m * n // gcd(m, n)
            check(len(set(cells[:L])) == L, "no repeat before the first return")
            check(cells[L % (m * n)] == (0, 0) if L < m * n else True, "back at the start")
            check(all(cells[k] != (0, 0) for k in range(1, L)), "first return at lcm")
            check((L == m * n) == (gcd(m, n) == 1), "all cells ⟺ coprime")
        for m in range(1, 13):                       # the general claim
            for n in range(1, 13):
                hit = {(k % m, k % n) for k in range(m * n)}
                check((len(hit) == m * n) == (gcd(m, n) == 1), "CRT ⟺ gcd = 1")

        pcol = TEAL_B
        all_text, all_segs = [], []
        for gi, (m, n, TL) in enumerate(grids):
            X0, Y0 = TL[0], TL[1]

            def ctr(r, c):
                return np.array([X0 + (c + 0.5) * cs, Y0 - (r + 0.5) * cs, 0.0])

            def corner(xc, yr):
                return np.array([X0 + xc * cs, Y0 - yr * cs, 0.0])

            box = {}
            for r in range(m):
                for c in range(n):
                    box[(r, c)] = Square(side_length=cs, stroke_color=GREY_C, stroke_width=1.5,
                                         fill_color=GREY_E, fill_opacity=0.0).move_to(ctr(r, c))
            frame = Rectangle(width=n * cs, height=m * cs, stroke_color=GREY_A,
                              stroke_width=2.5).move_to(
                np.array([X0 + n * cs / 2, Y0 - m * cs / 2, 0.0]))
            rh = [tag(str(r), 20, BLUE_B).move_to(ctr(r, 0) + LEFT * (cs / 2 + 0.24))
                  for r in range(m)]
            ch = [tag(str(c), 20, ORANGE).move_to(ctr(0, c) + UP * (cs / 2 + 0.22))
                  for c in range(n)]
            top_lab = tag(f"k mod {n}", 21, ORANGE).move_to(
                np.array([X0 + n * cs / 2, Y0 + cs / 2 + 0.22 + 0.36, 0.0]))
            side_lab = tag(f"k mod {m}", 21, BLUE_B).rotate(PI / 2).move_to(
                np.array([X0 - cs / 2 - 0.24 - 0.36, Y0 - m * cs / 2, 0.0]))
            title = VGroup(top_lab, side_lab)
            self.play(FadeIn(VGroup(*box.values())), Create(frame),
                      *[FadeIn(t) for t in rh + ch], FadeIn(title), run_time=0.8)

            nums = {}
            L = m * n // gcd(m, n)
            steps = []

            def reach(lab, u):
                hw, hh = lab.width / 2 + 0.07, lab.height / 2 + 0.07
                return min(hw / max(abs(u[0]), 1e-9), hh / max(abs(u[1]), 1e-9))

            for k in range(L + 1):
                r, c = k % m, k % n
                anims = []
                if k < L:
                    nums[k] = tag(str(k), 28).move_to(ctr(r, c))
                    anims.append(FadeIn(nums[k], scale=1.3))
                    anims.append(box[(r, c)].animate.set_fill(GREY_E, 0.9))
                if k > 0:
                    pr, pc = (k - 1) % m, (k - 1) % n
                    x1, y1 = pc + 1, pr + 1                # corner the step passes
                    u = _u(np.array([1.0, -1.0, 0.0]))
                    a = ctr(pr, pc) + u * reach(nums[k - 1], u)
                    c1 = corner(x1, y1)
                    c2 = corner(0 if x1 == n else x1, 0 if y1 == m else y1)
                    dest = nums[k] if k < L else nums[0]
                    b = ctr(r, c) - u * reach(dest, u)
                    col = pcol if k < L else YELLOW
                    if close(c1, c2):
                        seg = VGroup(Arrow(a, b, buff=0, color=col, stroke_width=3,
                                           tip_length=0.11,
                                           max_tip_length_to_length_ratio=0.35))
                        steps.append((a, b))
                    else:
                        seg = VGroup(Line(a, c1, color=col, stroke_width=3),
                                     Arrow(c2, b, buff=0, color=col, stroke_width=3,
                                           tip_length=0.11,
                                           max_tip_length_to_length_ratio=0.6))
                        steps += [(a, c1), (c2, b)]
                    anims.append(Create(seg))
                self.play(*anims, run_time=0.3 if k < L else 0.6)
            ring = Circle(radius=0.27, color=YELLOW, stroke_width=4).move_to(ctr(0, 0))
            self.play(Create(ring), run_time=0.5)
            self.play(FadeOut(ring), run_time=0.3)

            # readouts under the grid
            base_y = Y0 - m * cs - 0.5
            if gi == 0:
                lines = [tag("back at 0 after lcm(3, 5) = 15 steps", 21),
                         tag("15 = 3 · 5 :  every cell once", 21, YELLOW_B)]
            else:
                lines = [tag("back at 0 after lcm(4, 6) = 12 steps", 21),
                         tag("12 < 24 :  half the cells missed", 21, RED_B)]
            xl0 = X0 - cs / 2 - 0.24 - 0.5
            for i, t in enumerate(lines):
                _left_at(t, xl0, base_y - 0.5 * i)
            self.play(FadeIn(lines[0]), run_time=0.6)
            if gi == 1:
                empty = [box[(r, c)] for r in range(m) for c in range(n)
                         if (r, c) not in {(k % m, k % n) for k in range(L)}]
                check(len(empty) == m * n - L, "the unreached cells")
                self.play(*[e.animate.set_fill(RED_E, 0.55) for e in empty],
                          FadeIn(lines[1]), run_time=0.8)
            else:
                self.play(FadeIn(lines[1]), run_time=0.6)
                # one cell read both ways: 7 sits in row 1, column 2
                k7 = 7
                hi = Square(side_length=cs, stroke_color=YELLOW, stroke_width=4).move_to(
                    ctr(k7 % 3, k7 % 5))
                iso = _trow([("7 → (7 mod 3, 7 mod 5) = (", WHITE), ("1", BLUE_B),
                             (",", WHITE, 0.12), ("2", ORANGE), (")", WHITE)], size=21,
                            buff=0.04)
                iso.shift(np.array([xl0 - iso.get_left()[0], base_y - 1.02, 0.0]))
                zz = tag("ℤ₁₅  ≅  ℤ₃ × ℤ₅", 28, YELLOW_B)
                _left_at(zz, xl0, base_y - 1.62)
                self.play(Create(hi), Indicate(rh[k7 % 3], color=YELLOW, scale_factor=1.5),
                          Indicate(ch[k7 % 5], color=YELLOW, scale_factor=1.5),
                          FadeIn(iso), run_time=1.0)
                self.play(FadeIn(zz), run_time=0.6)
                lines += [iso, zz]
            self.hold(0.4)
            all_text += [top_lab, side_lab] + lines
            all_segs += steps + [(frame.get_corner(UL), frame.get_corner(UR)),
                                 (frame.get_corner(UR), frame.get_corner(DR)),
                                 (frame.get_corner(DR), frame.get_corner(DL)),
                                 (frame.get_corner(DL), frame.get_corner(UL))]
            for t in nums.values():
                check(_hit(t, steps, 0.02) is None, f"the walk stays off the number {t.text}")
            for t in rh + ch:
                check(_hit(t, all_segs, 0.03) is None, "headers clear of the grid")

        _labels_ok(all_text, all_segs, "J28", pad=0.05, gap=0.05)
        cap = caption("gcd(m, n) = 1 :   k → (k mod m, k mod n) reaches every cell once", 30)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J33

class J33_CastingOutNines(Board):
    """238 as blocks: two hundreds, three tens, eight ones. A ten is a strip
    of 9 and one spare cell. A hundred is ten tens: each row gives a strip
    of 9 and a spare, and the ten spares make a column that is itself a ten,
    9 and one spare — so 100 = 11 · 9 + 1 (and 1000, ten hundreds, is
    111 · 9 + 1 the same way). Strip every block of its nines: what is left
    is one cell per hundred, one per ten and the ones, 2 + 3 + 8 = 13, the
    digit sum. So 238 and 13 differ by a multiple of 9: both leave 4."""

    def construct(self):
        def dsum(n):
            return sum(int(ch) for ch in str(n))
        for n in range(1, 20001):
            check(n % 9 == dsum(n) % 9, "n ≡ digit sum (mod 9)")
        for k in range(1, 7):
            check((10 ** k - 1) % 9 == 0, "10^k − 1 is a multiple of 9")
        check(100 == 11 * 9 + 1 and 1000 == 10 * 100 == 111 * 9 + 1 and 10 == 9 + 1,
              "the blocks")

        N = 238
        h, t, o = N // 100, N // 10 % 10, N % 10
        cu = 0.27
        yb = -0.15                                  # bottom of every block
        GREY_CELL, NINE, SPARE = GREY_D, BLUE_D, GOLD_D

        def cellsq(x, y, col=GREY_CELL, op=0.9):
            return Square(side_length=cu, fill_color=col, fill_opacity=op,
                          stroke_color=GREY_B, stroke_width=1).move_to(
                np.array([x + cu / 2, y + cu / 2, 0.0]))

        # hundreds: 10 × 10; tens: vertical rods of 10; ones: a rod of 8
        hx = [-6.3, -3.4]
        tx = [-0.4, 0.0, 0.4]
        ox = 0.95
        H = [{(i, j): cellsq(x0 + i * cu, yb + j * cu) for i in range(10) for j in range(10)}
             for x0 in hx[:h]]
        T = [{j: cellsq(x0, yb + j * cu) for j in range(10)} for x0 in tx[:t]]
        O = {j: cellsq(ox, yb + j * cu) for j in range(o)}
        total = sum(len(b) for b in H) + sum(len(b) for b in T) + len(O)
        check(total == N, "the blocks hold 238 cells")

        top = _trow([("238", YELLOW_B, 0.25), ("=", WHITE, 0.25), ("2 · 100", WHITE, 0.25),
                     ("+", WHITE, 0.25), ("3 · 10", WHITE, 0.25), ("+", WHITE, 0.25),
                     ("8", WHITE)], size=34)
        top.move_to(np.array([hx[0] + top.width / 2, 3.38, 0.0]))
        ylab = yb + 10 * cu + 0.24
        hl = [tag("100", 22, GREY_A).move_to(np.array([x0 + 5 * cu, ylab, 0])) for x0 in hx[:h]]
        tl = tag("10", 22, GREY_A).move_to(np.array([(tx[0] + tx[t - 1] + cu) / 2, ylab, 0]))
        ol = tag("1", 22, GREY_A).move_to(np.array([ox + cu / 2, ylab, 0]))
        self.play(FadeIn(top), run_time=0.7)
        self.play(*[FadeIn(VGroup(*b.values())) for b in H + T], FadeIn(VGroup(*O.values())),
                  *[FadeIn(x) for x in hl + [tl, ol]], run_time=1.0)

        def strip(cells):
            """Outline round a run of nine neighbouring cells."""
            return SurroundingRectangle(VGroup(*cells), buff=0.0, color=WHITE,
                                        stroke_width=2.2)

        def eq_row(lhs, rhs, size=27, col=WHITE):
            return _trow([(lhs, col, 0.2), ("=", col, 0.2), (rhs, col)], size=size)

        # a ten = a strip of 9 and one spare
        rx = 1.85
        r10 = eq_row("10", "9 + 1")
        r10.shift(np.array([rx - r10.get_left()[0], 2.25, 0.0]))
        ten_strips = [strip([b[j] for j in range(9)]) for b in T]
        self.play(*[b[j].animate.set_fill(NINE, 0.9) for b in T for j in range(9)],
                  *[b[9].animate.set_fill(SPARE, 1.0) for b in T],
                  *[Create(s) for s in ten_strips], FadeIn(r10), run_time=1.0)

        # a hundred = ten rows, each a ten: 9 + 1; the ten spares make a ten
        row_strips = [strip([b[(i, j)] for i in range(9)]) for b in H for j in range(10)]
        self.play(*[b[(i, j)].animate.set_fill(NINE, 0.9) for b in H for i in range(9)
                    for j in range(10)],
                  *[b[(9, j)].animate.set_fill(SPARE, 1.0) for b in H for j in range(10)],
                  *[Create(s) for s in row_strips], run_time=1.2)
        col_strips = [strip([b[(9, j)] for j in range(9)]) for b in H]
        r100 = eq_row("100", "11 · 9 + 1")
        r100.shift(np.array([rx - r100.get_left()[0], 1.6, 0.0]))
        self.play(*[b[(9, j)].animate.set_fill(NINE, 0.9) for b in H for j in range(9)],
                  *[Create(s) for s in col_strips], FadeIn(r100), run_time=1.0)
        # a thousand is ten hundreds: the same picture one level up
        r1000 = eq_row("1000", "10 · 100")
        r1000.shift(np.array([rx - r1000.get_left()[0], 0.95, 0.0]))
        r1000b = _trow([("=", WHITE, 0.2), ("111 · 9 + 1", WHITE)], size=27)
        r1000b.shift(np.array([r1000[1].get_left()[0] - r1000b.get_left()[0], 0.4, 0.0]))
        self.play(FadeIn(r1000), run_time=0.6)
        self.play(FadeIn(r1000b), run_time=0.6)
        for b in H:                  # rows 0–9 (9 each) + right column 0–8 (9)
            in_strips = ([(i, j) for j in range(10) for i in range(9)]
                         + [(9, j) for j in range(9)])
            check(len(in_strips) == 99 and len(set(in_strips)) == 99
                  and set(b) - set(in_strips) == {(9, 9)}, "a hundred: 11 nines and one spare")
        for b in T:
            check(set(b) - set(range(9)) == {9}, "a ten: one nine and one spare")
        self.hold(0.4)

        # the spares — one per hundred, one per ten, and the ones — line up
        spares = [b[(9, 9)] for b in H] + [b[9] for b in T] + [O[j] for j in range(o)]
        check(len(spares) == dsum(N) == h + t + o, "13 spares = the digit sum")
        px, py = hx[0], -1.6
        targets = [np.array([px + (k + 0.5) * cu, py + cu / 2, 0.0]) for k in range(len(spares))]
        sum_lab = _trow([("2", WHITE, 0.14), ("+", WHITE, 0.14), ("3", WHITE, 0.14),
                         ("+", WHITE, 0.14), ("8", WHITE, 0.14), ("=", WHITE, 0.14),
                         ("13", YELLOW_B)], size=28)
        sum_lab.move_to(np.array([px + sum_lab.width / 2, py + cu + 0.4, 0.0]))
        self.play(*[O[j].animate.set_fill(SPARE, 1.0) for j in range(o)], run_time=0.5)
        self.play(*[s.animate.move_to(p) for s, p in zip(spares, targets)],
                  FadeIn(sum_lab), FadeOut(ol), run_time=1.5)
        for s, p in zip(spares, targets):
            check(close(s.get_center(), p, 1e-6), "spare in the pile")

        # everything left behind is nines: 2 · 11 + 3 = 25 of them
        n_nines = h * 11 + t
        check(9 * n_nines + len(spares) == N, "238 = 25 · 9 + 13")
        lx = -1.2
        r_n = _trow([("238", WHITE, 0.2), ("=", WHITE, 0.2), ("25 · 9", WHITE, 0.2),
                     ("+", WHITE, 0.2), ("(2 + 3 + 8)", WHITE)], size=28)
        r_n.shift(np.array([lx - r_n.get_left()[0], -0.95, 0.0]))
        self.play(FadeIn(r_n), *[Indicate(s, color=WHITE, scale_factor=1.03)
                                  for s in ten_strips + col_strips], run_time=1.0)

        # the pile: 13 = 9 + 4
        pile9 = strip(spares[:9])
        self.play(*[s.animate.set_fill(NINE, 0.9) for s in spares[:9]], Create(pile9),
                  run_time=0.8)
        four = tag("4", 30, YELLOW_B).next_to(VGroup(*spares[9:]), DOWN, buff=0.14)
        r13 = _trow([("13", WHITE, 0.2), ("=", WHITE, 0.2), ("9 + 4", WHITE)], size=28)
        r13.shift(np.array([r_n[1].get_left()[0] - r13[1].get_left()[0], -1.6, 0.0]))
        r238 = _trow([("238", YELLOW_B, 0.2), ("=", YELLOW_B, 0.2), ("26 · 9", YELLOW_B, 0.2),
                      ("+", YELLOW_B, 0.2), ("4", YELLOW_B)], size=30)
        r238.shift(np.array([r_n[1].get_left()[0] - r238[1].get_left()[0], -2.35, 0.0]))
        check(N % 9 == 4 and dsum(N) % 9 == 4 and N == 26 * 9 + 4, "both leave 4")
        self.play(FadeIn(four), FadeIn(r13), run_time=0.7)
        self.play(FadeIn(r238), run_time=0.7)

        texts = ([top, r10, r100, r1000, r1000b, r_n, r13, r238, sum_lab, four]
                 + hl + [tl])
        sp_ids = {id(s) for s in spares}
        blocks = VGroup(*[c for b in H + T for c in b.values() if id(c) not in sp_ids])
        pile = VGroup(*spares)
        for x in texts:
            check(_inside(x), f"J33: '{_name(x)}' inside")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.05), "J33 texts apart")
        for x in [r10, r100, r1000, r1000b, r_n, r13, r238, top, sum_lab]:
            check(_apart(x, blocks, 0.1), f"J33: '{_name(x)}' clear of the blocks")
        for x in [r_n, r13, r238, sum_lab, four]:
            check(_apart(x, pile, 0.05), f"J33: '{_name(x)}' clear of the pile")
        cap = caption("10, 100, 1000, …  ≡  1     ⟹     n  ≡  its digit sum    (mod 9)", 30)
        check(cap.width < 13.0, "J33 caption at full size")
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J25

def _wedge_label(m, V, P, Q, r0, segs=(), avoid=(), r1=4.0, pad=0.06, frac=0.5):
    """Put label m inside the non-reflex angle P-V-Q, on the ray at fraction
    `frac` of the angle, at the least distance ≥ r0 from V where it clears
    both arms (as long segments), every segment in segs and every mobject in
    avoid. Fails the render if there is no room."""
    V = to3(V)
    a1 = float(np.arctan2(*(to3(P) - V)[1::-1]))
    a2 = float(np.arctan2(*(to3(Q) - V)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    arms = [(V, V + 9 * _u(to3(P) - V)), (V, V + 9 * _u(to3(Q) - V))] + list(segs)
    u = _polar(a1 + frac * span)
    for d in np.arange(r0, r1, 0.02):
        m.move_to(V + d * u)
        if _hit(m, arms, pad) is None and all(_apart(m, o, 0.05) for o in avoid):
            return m
    check(False, f"room for the angle label '{_name(m)}'")


def _shrink_turn(mob, centre, factor, turn, **kw):
    """A similarity about `centre` in two halves: first shrink by `factor`
    towards it, then turn by `turn` about it. Every frame is a similar
    copy, and the small copy turns, so the motion stays near the centre."""
    start, c = mob.copy(), to3(centre)

    def upd(m, alpha):
        a1 = min(1.0, 2.0 * alpha)
        a2 = max(0.0, 2.0 * alpha - 1.0)
        m.become(start.copy().scale(factor ** a1, about_point=c)
                 .rotate(a2 * turn, about_point=c))
    return UpdateFromAlphaFunc(mob, upd, **kw)


class J25_SilverRectangle(Board):
    """Swing the diagonal √2 of a unit square down onto the base: a
    1 × (1 + √2) rectangle. Its diagonal makes 22.5° with the long side —
    the swing makes an isosceles triangle (two sides √2) whose exterior
    angle is 45°. Cut two unit squares off: the rest is (√2 − 1) × 1, and
    its diagonal also makes 22.5° with its long side — the second square's
    diagonal, swung up about its top corner, lands on the far corner (both
    √2): an isosceles triangle with a 45° apex, base angles 67.5°. Same
    angle, same shape: the rest is the rectangle shrunk by √2 − 1 = 1/(1 + √2)
    and turned a quarter turn. So 1 + √2 = 2 + 1/(1 + √2), and repeating,
    √2 = 1 + 1/(2 + 1/(2 + …))."""

    def construct(self):
        dl = 1.0 + np.sqrt(2.0)
        s = 1.0 / dl
        check(close(s, np.sqrt(2.0) - 1.0, 1e-12), "1/(1 + √2) = √2 − 1")
        k = 4.2
        G0 = np.array([-5.6, -2.45, 0.0])

        def M(x, y):
            return G0 + k * np.array([x, y, 0.0])

        G, Z, T, Y = M(0, 0), M(dl, 0), M(dl, 1), M(0, 1)
        O, P, F, D2 = M(1, 0), M(1, 1), M(2, 0), M(2, 1)
        R = [G, Z, T, Y]
        sq1 = [G, O, P, Y]
        sq2 = [O, F, D2, P]
        rem = [F, Z, T, D2]
        check(_tiles_exactly([sq1, sq2, rem], R), "two unit squares and the rest tile R")

        # the two isosceles triangles and their angles
        check(close(np.linalg.norm(Y - O), np.linalg.norm(Z - O)), "OY = OZ = √2")
        check(close(_ang_at(O, Y, G), PI / 4) and close(_ang_at(Z, Y, O), PI / 8)
              and close(_ang_at(Y, O, Z), PI / 8), "45° outside O, 22.5° at Y and Z")
        check(close(np.linalg.norm(F - P), np.linalg.norm(T - P)), "PF = PT = √2")
        check(close(_ang_at(P, F, T), PI / 4) and close(_ang_at(T, P, F), 3 * PI / 8)
              and close(_ang_at(F, P, T), 3 * PI / 8)
              and close(_ang_at(T, F, Z), PI / 8), "45° at P, 67.5° at F and T, 22.5° to TZ")

        # the similarity S: shrink by s and a quarter turn, with Z ↦ T
        R90 = np.array([[0.0, -1.0], [1.0, 0.0]])

        def S(p):
            q = T[:2] + s * R90 @ (to3(p)[:2] - Z[:2])
            return np.array([q[0], q[1], 0.0])
        c2 = np.linalg.solve(np.eye(2) - s * R90, T[:2] - s * R90 @ Z[:2])
        C = np.array([c2[0], c2[1], 0.0])
        check(close(S(C), C), "the centre is fixed")
        check(_same_poly([S(p) for p in R], rem, 1e-9), "S maps R onto the rest")
        check(close(S(Y), F) and close(S(Z), T), "S maps the diagonal ZY onto TF")
        check(_pip(C, rem), "the centre lies in the rest")

        # ---- 1. the unit square and the swing of its diagonal
        c_sq1, c_sq2, c_rem = BLUE_D, TEAL_D, ORANGE
        s1 = mk(sq1, c_sq1, 0.75)
        l_side = tag("1", 30).next_to(Line(G, Y), LEFT, buff=0.16)
        l_b1 = tag("1", 30).move_to((G + O) / 2 + DOWN * 0.32)
        self.play(FadeIn(s1), FadeIn(l_side), FadeIn(l_b1), run_time=0.8)
        diag = Line(O, Y, color=YELLOW, stroke_width=5)
        self.play(Create(diag), run_time=0.6)
        swing_arc = Arc(radius=np.sqrt(2) * k, start_angle=3 * PI / 4, angle=-3 * PI / 4,
                        arc_center=O, color=YELLOW, stroke_width=2).set_stroke(opacity=0.6)
        check(O[1] + np.sqrt(2) * k < SAFE_TOP, "the swing stays in the frame")
        swung = diag.copy()
        self.add(swung)
        self.play(Rotate(swung, angle=-3 * PI / 4, about_point=O), Create(swing_arc),
                  run_time=1.6)
        check(close(swung.get_start(), Z) or close(swung.get_end(), Z), "the √2 lands at Z")
        frameR = Polygon(*R, stroke_color=WHITE, stroke_width=3, fill_opacity=0)
        l_r2 = tag("√2", 30, YELLOW).move_to((O + Z) / 2 + DOWN * 0.34)
        self.play(Create(frameR), FadeIn(l_r2), FadeOut(swing_arc), run_time=0.8)

        # ---- 2. the isosceles triangle of the swing: 22.5° at the far corner
        diagR = Line(Z, Y, color=WHITE, stroke_width=3)
        tk1 = _ticks(O, Y, 1, YELLOW, 0.13)
        tk2 = _ticks(O, Z, 1, YELLOW, 0.13, at=0.62)
        a45 = angle_arc(O, Y, G, radius=0.55, color=YELLOW_B, width=4)
        aZ = angle_arc(Z, Y, O, radius=2.1, color=YELLOW_B, width=4)
        aY = angle_arc(Y, O, Z, radius=0.8, color=YELLOW_B, width=4)
        tick_segs = [(m_.get_start(), m_.get_end()) for m_ in list(tk1) + list(tk2)]
        segsA = ([(Z, Y), (O, Y), (G, Z), (Z, T), (T, Y), (Y, G), (F, D2), (O, P)]
                 + _angle_segs(O, Y, G, 0.55) + _angle_segs(Z, Y, O, 2.1)
                 + _angle_segs(Y, O, Z, 0.8) + tick_segs)
        l45 = _wedge_label(tag("45°", 26, YELLOW_B), O, Y, G, 0.75, segsA)
        lZ = _wedge_label(tag("22.5°", 24, YELLOW_B), Z, Y, O, 2.2, segsA)
        self.play(Create(diagR), run_time=0.6)
        self.play(Create(tk1), Create(tk2), Create(a45), FadeIn(l45), run_time=0.7)
        rowA = tag("two sides √2 :   ½ · 45°  =  22.5°", 26)
        _left_at(rowA, G0[0], 3.35)
        self.play(Create(aZ), Create(aY), FadeIn(lZ), FadeIn(rowA), run_time=0.9)
        self.hold(0.5)

        # ---- 3. cut off two unit squares; the rest is (√2 − 1) × 1
        s2 = mk(sq2, c_sq2, 0.75)
        rm = mk(rem, c_rem, 0.75)
        l_b2 = tag("1", 30).move_to((O + F) / 2 + DOWN * 0.32)
        l_b3 = tag("√2 − 1", 28, ORANGE).move_to((F + Z) / 2 + DOWN * 0.34)
        self.play(FadeOut(tk1), FadeOut(tk2), FadeOut(a45), FadeOut(l45), FadeOut(aY),
                  FadeOut(diag), FadeOut(swung), FadeOut(aZ), FadeOut(lZ), run_time=0.5)
        self.add(s2, rm)
        self.bring_to_back(s2, rm)
        self.play(FadeIn(s2), FadeIn(rm), FadeOut(l_r2), FadeIn(l_b2), FadeIn(l_b3),
                  run_time=0.9)
        self.bring_to_front(frameR, diagR)

        # ---- 4. the second square's diagonal swings up onto the far corner
        dPF = Line(P, F, color=YELLOW, stroke_width=5)
        self.play(Create(dPF), run_time=0.5)
        swung2 = dPF.copy()
        arc2 = Arc(radius=np.sqrt(2) * k, start_angle=-PI / 4, angle=PI / 4, arc_center=P,
                   color=YELLOW, stroke_width=2).set_stroke(opacity=0.6)
        self.add(swung2)
        self.play(Rotate(swung2, angle=PI / 4, about_point=P), Create(arc2), run_time=1.1)
        check(close(swung2.get_end(), T) or close(swung2.get_start(), T), "√2 lands at T")
        diagF = Line(T, F, color=WHITE, stroke_width=3)
        tk3 = _ticks(P, F, 2, YELLOW, 0.13)
        tk4 = _ticks(P, T, 2, YELLOW, 0.13, at=0.62)
        aP = angle_arc(P, F, T, radius=0.55, color=YELLOW_B, width=4)
        aF1 = angle_arc(F, P, T, radius=0.42, color=GREY_A, width=3)
        aF2 = angle_arc(F, P, T, radius=0.52, color=GREY_A, width=3)
        aT1 = angle_arc(T, P, F, radius=0.42, color=GREY_A, width=3)
        aT2 = angle_arc(T, P, F, radius=0.52, color=GREY_A, width=3)
        aT = angle_arc(T, F, Z, radius=1.85, color=YELLOW_B, width=4)
        tick_segs2 = [(m_.get_start(), m_.get_end()) for m_ in list(tk3) + list(tk4)]
        segsB = ([(T, F), (P, F), (P, T), (F, Z), (Z, T), (O, P), (F, D2), (Z, Y)]
                 + _angle_segs(P, F, T, 0.55) + _angle_segs(T, F, Z, 1.85)
                 + _angle_segs(T, P, F, 0.52) + _angle_segs(F, P, T, 0.52) + tick_segs2)
        lP = _wedge_label(tag("45°", 26, YELLOW_B), P, F, T, 0.75, segsB)
        lT = _wedge_label(tag("22.5°", 24, YELLOW_B), T, F, Z, 1.95, segsB)
        self.play(Create(diagF), FadeOut(arc2), run_time=0.6)
        self.play(Create(tk3), Create(tk4), Create(aP), FadeIn(lP), run_time=0.7)
        rowB = tag("two sides √2 :   ½ (180° − 45°)  =  67.5°  =  90° − 22.5°", 26)
        _left_at(rowB, G0[0], 2.72)
        self.play(Create(aF1), Create(aF2), Create(aT1), Create(aT2), FadeIn(rowB),
                  run_time=0.8)
        self.play(Create(aT), FadeIn(lT), run_time=0.7)
        self.hold(0.6)

        # ---- 5. same angle, same shape: R shrunk by √2 − 1 and turned
        self.play(*[FadeOut(x) for x in (tk3, tk4, aP, lP, aF1, aF2, aT1, aT2, dPF, swung2)],
                  FadeIn(aZ), FadeIn(lZ), run_time=0.6)
        self.hold(0.5)
        ghost = VGroup(Polygon(*R, stroke_color=YELLOW, stroke_width=4, fill_opacity=0),
                       Line(Z, Y, color=YELLOW, stroke_width=3),
                       angle_arc(Z, Y, O, radius=2.1, color=YELLOW, width=4))
        self.add(ghost)
        self.play(_shrink_turn(ghost, C, s, PI / 2), run_time=2.0)
        check(_same_poly(_verts(ghost[0]), rem, 1e-6), "the shrunk, turned R is the rest")
        check(close(ghost[1].get_start(), T, 1e-6) and close(ghost[1].get_end(), F, 1e-6),
              "its diagonal lands on TF")
        rowC = tag("√2 − 1  =  1 / (1 + √2)", 30, YELLOW_B)
        rowD = tag("1 + √2  =  2 + 1 / (1 + √2)", 30, YELLOW_B)
        _left_at(rowC, G0[0], 3.35)
        _left_at(rowD, G0[0], 2.72)
        self.play(FadeOut(rowA), FadeOut(rowB), run_time=0.4)
        self.play(FadeIn(rowC), run_time=0.6)
        self.play(FadeIn(rowD), FadeOut(ghost), run_time=0.7)
        self.hold(0.4)

        # ---- 6. and again, inside the rest: two squares and a smaller rest
        self.play(FadeOut(aZ), FadeOut(lZ), FadeOut(aT), FadeOut(lT), run_time=0.4)
        level = [sq1, sq2, rem]
        prev = VGroup(mk(sq1, c_sq1, 0.75), mk(sq2, c_sq2, 0.75), mk(rem, c_rem, 0.75))
        rests = [rm]
        for j in range(4):
            nxt = [[S(p) for p in P_] for P_ in level]
            check(_tiles_exactly(nxt, level[2], 90), f"level {j + 1} tiles the rest")
            cp = prev.copy()
            self.add(cp)
            self.play(_shrink_turn(cp, C, s, PI / 2), run_time=1.2 if j == 0 else 0.7)
            for m_, P_ in zip(cp, nxt):
                check(_same_poly(_verts(m_), P_, 1e-6), "the copy lands exactly")
            self.remove(rests[-1])
            rests.append(cp[2])
            prev = cp
            level = nxt
        self.bring_to_front(frameR, diagR, diagF)
        rowE = tag("=  2 + 1 / (2 + 1 / (2 + …))", 30, YELLOW_B)
        rowE.move_to(np.array([rowD.get_right()[0] + 0.25 + rowE.width / 2,
                               rowD.get_center()[1], 0.0]))
        check(rowE.get_right()[0] < SAFE_X, "the continued fraction fits")
        self.play(FadeIn(rowE), run_time=0.8)

        labels = [rowC, rowD, rowE, l_side, l_b1, l_b2, l_b3]
        segs = _poly_segs(R) + [(O, P), (F, D2), (Z, Y), (T, F)]
        _labels_ok(labels, segs, "J25", pad=0.05, gap=0.06)
        for lab in (lZ, lT, l45, lP):
            check(_inside(lab), "angle labels inside")
        cap = caption("√2  =  1 + 1 / (2 + 1 / (2 + 1 / (2 + …)))", 34)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J21

class J21_LatticeSegment(Board):
    """(12, 8) = 4 · (3, 2) with gcd(3, 2) = 1. The segment from (0, 0) to
    (12, 8) is four copies of the step from (0, 0) to (3, 2), each moved on
    by the lattice vector (3, 2), so the lattice points on it are equally
    spaced. Inside one step there is none: a lattice point there would have
    x = 1 or 2, where the step is at heights ⅔ and 1⅓ — y = 2x/3 is whole
    only when 3 divides x.
    So the lattice points strictly between the ends are the 4 − 1 = 3 joins
    of the steps; in general (a, b) = g · (a/g, b/g) with g = gcd(a, b)
    gives g − 1 of them."""

    def construct(self):
        a, b = 12, 8
        g = gcd(a, b)
        sa, sb = a // g, b // g
        check((g, sa, sb) == (4, 3, 2) and gcd(sa, sb) == 1, "(12, 8) = 4 · (3, 2)")
        on_seg = [(x, y) for x in range(a + 1) for y in range(b + 1)
                  if x * b == y * a]
        check(on_seg == [(k * sa, k * sb) for k in range(g + 1)],
              "the lattice points on the segment are the step ends")
        for A_ in range(1, 25):                     # the general claim
            for B_ in range(1, 25):
                n = sum(1 for x in range(1, A_) for y in range(1, B_) if x * B_ == y * A_)
                check(n == gcd(A_, B_) - 1, "gcd − 1 points between the ends")

        u = 0.5
        O = np.array([-6.2, -2.45, 0.0])

        def L(x, y):
            return O + u * np.array([x, y, 0.0])

        dots = VGroup(*[Dot(L(x, y), radius=0.04, color=GREY_B)
                        for x in range(a + 1) for y in range(b + 1)])
        seg = Line(L(0, 0), L(a, b), color=WHITE, stroke_width=3.5)
        lO = tag("(0, 0)", 22, GREY_A).next_to(L(0, 0), DOWN, buff=0.14)
        lA = tag("(12, 8)", 22, GREY_A).next_to(L(a, b), RIGHT, buff=0.14)
        self.play(FadeIn(dots), run_time=0.7)
        self.play(Create(seg), FadeIn(lO), FadeIn(lA), run_time=1.0)

        # the step (3, 2) and its three copies along the segment
        def step_tri(k):
            p0 = (k * sa, k * sb)
            return [L(*p0), L(p0[0] + sa, p0[1]), L(p0[0] + sa, p0[1] + sb)]
        cols = [BLUE_D, TEAL_D, BLUE_D, TEAL_D]
        tri0 = mk(step_tri(0), cols[0], 0.55, stroke_width=2)
        l3 = tag("3", 24).next_to(Line(step_tri(0)[0], step_tri(0)[1]), DOWN, buff=0.1)
        l2 = tag("2", 24).next_to(Line(step_tri(0)[1], step_tri(0)[2]), RIGHT, buff=0.1)
        self.play(FadeIn(tri0), FadeIn(l3), FadeIn(l2), run_time=0.8)
        tris = [tri0]
        for k in range(1, g):
            t_ = tri0.copy().set_fill(cols[k], 0.55)
            self.add(t_)
            self.play(t_.animate.shift(L(k * sa, k * sb) - L(0, 0)), run_time=0.55)
            check(_same_poly(_verts(t_), step_tri(k), 1e-9), f"copy {k} lands on its step")
            tris.append(t_)
        self.bring_to_front(seg)
        joins = VGroup(*[Dot(L(k * sa, k * sb), radius=0.1, color=YELLOW)
                         for k in range(1, g)])
        ends = VGroup(Dot(L(0, 0), radius=0.08, color=WHITE), Dot(L(a, b), radius=0.08,
                                                                    color=WHITE))
        ticks = VGroup(*[_ticks(L(k * sa, k * sb), L((k + 1) * sa, (k + 1) * sb), 1,
                                YELLOW_B, 0.12, at=0.5) for k in range(g)])
        row1 = tag("(12, 8)  =  4 · (3, 2)", 28)
        row2 = tag("gcd(12, 8)  =  4", 28, YELLOW_B)
        _left_at(row1, -6.2, 3.35)
        _left_at(row2, -6.2, 2.7)
        self.play(FadeIn(ends), FadeIn(joins), Create(ticks), FadeIn(row1), run_time=0.9)
        self.play(FadeIn(row2), run_time=0.6)
        self.hold(0.4)

        # one step, enlarged: it meets the grid lines between lattice points
        v = 1.5
        I0 = np.array([1.55, -1.3, 0.0])

        def J(x, y):
            return I0 + v * np.array([x, y, 0.0])
        box_small = Rectangle(width=sa * u, height=sb * u, stroke_color=YELLOW_B,
                              stroke_width=2.5).move_to(L(sa / 2, sb / 2))
        box_big = Rectangle(width=sa * v, height=sb * v, stroke_color=YELLOW_B,
                            stroke_width=2.5).move_to(J(sa / 2, sb / 2))
        grid = VGroup(*[Line(J(x, 0), J(x, sb), color=GREY_D, stroke_width=1.5)
                        for x in range(sa + 1)],
                      *[Line(J(0, y), J(sa, y), color=GREY_D, stroke_width=1.5)
                        for y in range(sb + 1)])
        idots = VGroup(*[Dot(J(x, y), radius=0.07, color=GREY_B)
                         for x in range(sa + 1) for y in range(sb + 1)])
        iseg = Line(J(0, 0), J(sa, sb), color=WHITE, stroke_width=4)
        iends = VGroup(Dot(J(0, 0), radius=0.1, color=YELLOW), Dot(J(sa, sb), radius=0.1,
                                                                     color=YELLOW))
        crosses = [(1, 2 / 3), (2, 4 / 3)]           # where x = 1, 2 meet the step
        for x, y in crosses:
            check(close(y, x * sb / sa) and (x != round(x) or y != round(y)),
                  "a crossing off the lattice")
        check(all(not (x * sb % sa == 0) for x in range(1, sa)), "y = 2x/3 never whole")
        rings = VGroup(*[Circle(radius=0.11, color=ORANGE, stroke_width=3).move_to(J(x, y))
                         for x, y in crosses])
        self.play(Create(box_small), run_time=0.4)
        self.play(TransformFromCopy(box_small, box_big), run_time=0.9)
        self.play(FadeIn(grid), FadeIn(idots), Create(iseg), FadeIn(iends), run_time=0.8)
        self.play(LaggedStart(*[Create(r) for r in rings], lag_ratio=0.3), run_time=0.9)
        cl = [tag("y = ⅔", 22, ORANGE), tag("y = 1⅓", 22, ORANGE)]
        cl[0].next_to(rings[0], RIGHT, buff=0.1).shift(DOWN * 0.16)
        cl[1].next_to(rings[1], RIGHT, buff=0.1).shift(DOWN * 0.16)
        xt = [tag(str(x), 20, GREY_A).next_to(J(x, 0), DOWN, buff=0.14) for x in (1, 2)]
        xlab = tag("x", 20, GREY_A).next_to(J(sa, 0), DOWN, buff=0.14).shift(RIGHT * 0.3)
        row3 = tag("y = 2x/3 :  whole only if 3 | x", 24)
        _left_at(row3, I0[0], -2.2)
        self.play(*[FadeIn(c) for c in cl + xt + [xlab]], FadeIn(row3), run_time=0.8)
        self.hold(0.4)

        row4 = tag("4 steps  ⟹  4 − 1 = 3 points between the ends", 26, YELLOW_B)
        _left_at(row4, -6.2, 2.12)
        self.play(Indicate(joins, color=YELLOW, scale_factor=1.6), FadeIn(row4), run_time=1.0)

        # hygiene
        segs_main = ([(L(0, 0), L(a, b))] + [s_ for t_ in tris for s_ in _poly_segs(_verts(t_))]
                     + [(m_.get_start(), m_.get_end()) for t_ in ticks for m_ in t_])
        for lab in (l3, l2):
            check(_hit(lab, [(L(0, 0), L(a, b))] + _poly_segs(_verts(tris[1])), 0.03) is None,
                  "leg labels clear")
        isegs = ([(J(0, 0), J(sa, sb))] + [(m_.get_start(), m_.get_end()) for m_ in grid])
        for c_ in cl:
            check(_hit(c_, isegs, 0.03) is None and all(_apart(c_, r, 0.02) for r in rings),
                  f"crossing label {c_.text} clear")
            check(all(_apart(c_, d_, 0.03) for d_ in idots), "crossing labels clear of dots")
        texts = [row1, row2, row3, row4, lO, lA] + cl + xt + [xlab]
        _labels_ok([row1, row2, row3, row4, lO, lA], segs_main + isegs, "J21", pad=0.05)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.05), "J21 texts apart")
        check(_apart(box_big, VGroup(dots, lA), 0.2), "the inset clear of the lattice")
        cap = caption("gcd(a, b) − 1 lattice points between (0, 0) and (a, b), equally spaced",
                      28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J29

def _phi(d):
    return sum(1 for a in range(1, d + 1) if gcd(a, d) == 1)


class J29_TotientSum(Board):
    """Write k/12 for k = 1, …, 12 and put each in lowest terms a/d. Its
    denominator d divides 12, and the fractions with denominator d are the
    a/d with a ≤ d and a prime to d — every such a/d is some k/12, since
    d | 12. Draw the twelve columns k/12 and a row for each divisor d: the
    row has a slot at every a/d (all on the columns), and k/12 drops into
    the row of its own denominator. Each column ends with exactly one dot,
    and row d gets exactly φ(d) of them: 1 + 1 + 2 + 2 + 2 + 4 = 12."""

    def construct(self):
        n = 12
        divs = [d for d in range(1, n + 1) if n % d == 0]
        red = {k: (k // gcd(k, n), n // gcd(k, n)) for k in range(1, n + 1)}
        for N_ in range(1, 200):                     # the identity, for many n
            check(sum(_phi(d) for d in range(1, N_ + 1) if N_ % d == 0) == N_, "∑ φ(d) = n")
        for d in divs:
            got = sorted(a for (a, dd) in red.values() if dd == d)
            check(got == [a for a in range(1, d + 1) if gcd(a, d) == 1],
                  f"row {d} holds exactly the a/{d} in lowest terms")
            check(len(got) == _phi(d), f"φ({d}) of them")

        X0, X1 = -4.85, 4.2
        ytop = 2.75

        def X(t):
            return X0 + (X1 - X0) * t

        rows_y = {d: 1.85 - 0.78 * i for i, d in enumerate(divs)}
        dcol = {1: GREY_A, 2: RED_B, 3: GREEN_B, 4: BLUE_B, 6: ORANGE, 12: PURPLE_A}

        axis = Line(np.array([X(0) - 0.15, ytop, 0]), np.array([X(1) + 0.15, ytop, 0]),
                    color=GREY_B, stroke_width=2)
        zero = tag("0", 20, GREY_A).next_to(np.array([X(0), ytop, 0]), DOWN, buff=0.12)
        tdots = {k: Dot(np.array([X(k / n), ytop, 0]), radius=0.07, color=WHITE)
                 for k in range(1, n + 1)}
        tlabs = {k: tag(f"{k}/{n}", 16).move_to(np.array([X(k / n), ytop + 0.3, 0]))
                 for k in range(1, n + 1)}
        for k in range(1, n):
            check(_apart(tlabs[k], tlabs[k + 1], 0.12), "the k/12 labels stand apart")
        # where each k/12 will land (its row, its lowest-terms label)
        land = {}
        for k in range(1, n + 1):
            a_, d_ = red[k]
            p_ = np.array([X(k / n), rows_y[d_], 0])
            land[k] = (p_, tag(f"{a_}/{d_}", 19, dcol[d_]).move_to(p_ + UP * 0.29))
        # the column guides, broken where a label will sit
        cols = VGroup()
        guide_segs = []
        for k in range(1, n + 1):
            p_, lb_ = land[k]
            x_ = X(k / n)
            y_bot = rows_y[divs[-1]] - 0.22
            pieces = [(ytop - 0.12, lb_.get_top()[1] + 0.06), (p_[1] - 0.1, y_bot)]
            for y0_, y1_ in pieces:
                if y0_ - y1_ > 0.08:
                    seg_ = DashedLine(np.array([x_, y0_, 0]), np.array([x_, y1_, 0]),
                                      color=GREY_D, stroke_width=1.5, dash_length=0.06)
                    cols.add(seg_)
                    guide_segs.append((seg_.get_start(), seg_.get_end()))
        self.play(Create(axis), FadeIn(zero), *[FadeIn(tdots[k]) for k in tdots],
                  *[FadeIn(tlabs[k]) for k in tlabs], run_time=1.0)

        # one row per divisor d, with a slot at every a/d
        rlab, slots = {}, {}
        for d in divs:
            y = rows_y[d]
            rlab[d] = tag(f"d = {d}", 24, dcol[d])
            _left_at(rlab[d], -6.45, y)
            slots[d] = VGroup(*[Circle(radius=0.09, color=dcol[d], stroke_width=2)
                                .set_stroke(opacity=0.55).move_to(np.array([X(a / d), y, 0]))
                                for a in range(1, d + 1)])
            for a in range(1, d + 1):         # every slot sits on a column
                check(abs(a * n / d - round(a * n / d)) < 1e-12, "a/d is some k/12")
        self.play(Create(cols), *[FadeIn(rlab[d]) for d in divs],
                  *[FadeIn(slots[d]) for d in divs], run_time=1.1)

        # each k/12 drops into the row of its denominator, in lowest terms
        phil = {}
        fdots, flabs = [], []
        for d in divs:
            ks = [k for k in range(1, n + 1) if red[k][1] == d]
            anims = []
            for k in ks:
                a = red[k][0]
                p = np.array([X(k / n), rows_y[d], 0])
                dt = tdots[k].copy()
                lb = tlabs[k].copy()
                tgt_dot = Dot(p, radius=0.09, color=dcol[d])
                tgt_lab = land[k][1].copy()
                check(close(land[k][0], p), "lands where the guide expects it")
                anims += [Transform(dt, tgt_dot), Transform(lb, tgt_lab)]
                fdots.append(dt)
                flabs.append(lb)
            phil[d] = tag(f"φ({d}) = {len(ks)}", 24, dcol[d])
            _left_at(phil[d], 4.62, rows_y[d])
            self.play(*anims, run_time=0.75)
            self.play(FadeIn(phil[d], shift=LEFT * 0.1), run_time=0.35)
        check(len(fdots) == n, "every k/12 placed once")
        # every column now holds exactly one dot
        xs = sorted(round(float(m_.get_center()[0]), 6) for m_ in fdots)
        check(xs == sorted(round(float(X(k / n)), 6) for k in range(1, n + 1)),
              "one dot per column")
        self.hold(0.3)

        total = tag("1 + 1 + 2 + 2 + 2 + 4  =  12", 28, YELLOW_B)
        total.move_to(np.array([(X0 + X1) / 2, -2.62, 0.0]))
        check(sum(_phi(d) for d in divs) == n, "1 + 1 + 2 + 2 + 2 + 4 = 12")
        brace = Line(np.array([4.5, rows_y[divs[0]] + 0.25, 0]),
                     np.array([4.5, rows_y[divs[-1]] - 0.25, 0]), color=GREY_B,
                     stroke_width=2)
        self.play(FadeIn(total), Create(brace), run_time=0.8)

        # hygiene: labels apart; slots and dots on their columns
        labs = list(tlabs.values()) + flabs + list(rlab.values()) + list(phil.values()) + [
            total, zero]
        for i in range(len(labs)):
            check(_inside(labs[i]), "J29 label inside")
            for j in range(i + 1, len(labs)):
                check(_apart(labs[i], labs[j], 0.03), "J29 labels apart")
        dot_like = [m_ for d in divs for m_ in slots[d]] + fdots
        for lb in flabs:
            check(all(_apart(lb, m_, 0.02) for m_ in dot_like), "row labels clear of the dots")
        for lb in labs:
            check(_hit(lb, guide_segs, 0.03) is None, f"guides clear of the label {_name(lb)}")
        cap = caption("∑ φ(d) over the divisors d of n   =   n", 32)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== J30

def _farey(N):
    fr = sorted({(a // gcd(a, b), b // gcd(a, b)) for b in range(1, N + 1)
                 for a in range(0, b + 1)}, key=lambda t: t[0] / t[1])
    return fr                                         # list of (a, b), a/b ascending


def _lattice_in_triangle(P, Q, R):
    """Lattice points in the closed triangle PQR (integer vertices):
    (interior, boundary) counts, exactly."""
    xs = [P[0], Q[0], R[0]]
    ys = [P[1], Q[1], R[1]]

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    sgn = 1 if cr(P, Q, R) > 0 else -1
    I = B = 0
    for x in range(min(xs), max(xs) + 1):
        for y in range(min(ys), max(ys) + 1):
            c = [sgn * cr(P, Q, (x, y)), sgn * cr(Q, R, (x, y)), sgn * cr(R, P, (x, y))]
            if min(c) < 0:
                continue
            if min(c) == 0:
                B += 1
            else:
                I += 1
    return I, B


class J30_FareyNeighbours(Board):
    """A fraction a/b is the point (b, a): its slope from O. Draw the ray of
    every fraction of F₄. Each lattice point with 1 ≤ b ≤ 4 lies on the ray
    of its own fraction in lowest terms, so between two neighbouring rays
    (inside b ≤ 4) there is none. The neighbours 1/3 < 1/2 span the triangle
    O, (3, 1), (2, 1): no lattice point inside (I = 0), none on its sides but
    the corners (B = 3; on O(3, 1) and O(2, 1) because 1/3 and 1/2 are in
    lowest terms). Pick's theorem gives A = I + B/2 − 1 = ½. The triangle is
    half the parallelogram on (3, 1) and (2, 1), whose area is the
    determinant 3·1 − 1·2 — so bc − ad = 2A = 1. The same holds for every
    pair of neighbours, in every Farey sequence."""

    def construct(self):
        N = 4
        F4 = _farey(N)
        check([f"{a}/{b}" for a, b in F4] == ["0/1", "1/4", "1/3", "1/2", "2/3", "3/4", "1/1"],
              "F₄")
        # every lattice point with 1 ≤ x ≤ 4, 0 ≤ y ≤ x lies on a ray of F₄
        for x in range(1, N + 1):
            for y in range(0, x + 1):
                g = gcd(x, y)
                check((y // g, x // g) in F4, "on the ray of its lowest terms")
        # every pair of neighbours: an empty triangle, area ½, bc − ad = 1
        for (a, b), (c, d) in zip(F4, F4[1:]):
            I_, B_ = _lattice_in_triangle((0, 0), (b, a), (d, c))
            check((I_, B_) == (0, 3), f"empty triangle for {a}/{b}, {c}/{d}")
            check(close(abs(area([(0, 0), (b, a), (d, c)])), 0.5), "area ½")
            check(b * c - a * d == 1 and close(I_ + B_ / 2 - 1, 0.5), "bc − ad = 1, Pick")
        for N_ in range(1, 12):                       # the claim for other Farey sequences
            FN = _farey(N_)
            check(all(b * c - a * d == 1 for (a, b), (c, d) in zip(FN, FN[1:])),
                  "neighbours in Fₙ: bc − ad = 1")

        u = 1.25
        O = np.array([-6.1, -2.55, 0.0])

        def L(x, y):
            return O + u * np.array([x, y, 0.0])

        fan = [(x, y) for x in range(0, N + 1) for y in range(0, x + 1)]
        dots = VGroup(*[Dot(L(x, y), radius=0.035 if (x, y) not in fan or (x, y) == (0, 0)
                            else 0.05, color=GREY_C if (x, y) not in fan else GREY_A)
                        for x in range(0, 6) for y in range(0, 5)])
        region = Polygon(L(0, 0), L(N, 0), L(N, N), stroke_color=GREY_B, stroke_width=1.5,
                         fill_opacity=0).set_stroke(opacity=0.6)
        xnum = [tag(str(x), 20, GREY_B).move_to(L(x, 0) + DOWN * 0.26) for x in range(1, 6)]
        ynum = [tag(str(y), 20, GREY_B).move_to(L(0, y) + LEFT * 0.28) for y in range(1, 5)]
        bl = tag("b", 22, GREY_A).move_to(L(5, 0) + DOWN * 0.26 + RIGHT * 0.45)
        al = tag("a", 22, GREY_A).move_to(L(0, 4) + LEFT * 0.28 + UP * 0.45)
        self.play(FadeIn(dots), Create(region), *[FadeIn(t_) for t_ in xnum + ynum + [bl, al]],
                  run_time=0.8)

        # the rays of F₄, in order, with the list along the top
        parts = [("F₄ :", GREY_A, 0.3)]
        for i, (a, b) in enumerate(F4):
            parts.append((f"{a}/{b}", WHITE, 0.16))
            if i < len(F4) - 1:
                parts.append(("<", GREY_B, 0.16))
        flist = _trow(parts, size=26)
        flist.move_to(np.array([-6.2 + flist.width / 2, 3.42, 0.0]))
        check(_inside(flist), "the Farey list fits")
        fr_mobs = [flist[1 + 2 * i] for i in range(len(F4))]
        rays, fdots = {}, {}
        self.play(FadeIn(flist[0]), run_time=0.3)
        for i, (a, b) in enumerate(F4):
            end = L(N, N * a / b)
            rays[(a, b)] = Line(L(0, 0), end, color=GREY_B, stroke_width=2)
            fdots[(a, b)] = Dot(L(b, a), radius=0.075, color=WHITE)
            an = [Create(rays[(a, b)]), FadeIn(fdots[(a, b)]), FadeIn(fr_mobs[i])]
            if i > 0:
                an.append(FadeIn(flist[2 * i]))
            self.play(*an, run_time=0.32)
        r_pt = tag("a/b  ↔  the point (b, a)", 26)
        _left_at(r_pt, 1.2, 2.2)
        self.play(FadeIn(r_pt), run_time=0.6)
        self.hold(0.3)

        # the neighbours 1/3 < 1/2 and their triangle
        A_, B_ = (1, 3), (1, 2)
        Bp, Dp = L(3, 1), L(2, 1)
        i_lo, i_hi = F4.index(A_), F4.index(B_)
        check(i_hi == i_lo + 1, "1/3 and 1/2 are neighbours in F₄")
        tri = mk([L(0, 0), Bp, Dp], ORANGE, 0.6, stroke_width=3)
        r_nb = tag("1/3 < 1/2 :   (3, 1),  (2, 1)", 26, YELLOW_B)
        _left_at(r_nb, 1.2, 1.5)
        self.play(rays[A_].animate.set_color(YELLOW), rays[B_].animate.set_color(YELLOW),
                  fr_mobs[i_lo].animate.set_color(YELLOW), fr_mobs[i_hi].animate.set_color(YELLOW),
                  FadeIn(r_nb), run_time=0.7)
        self.add(tri)
        self.bring_to_back(tri)
        self.play(FadeIn(tri), run_time=0.8)
        rings = VGroup(*[Circle(radius=0.14, color=YELLOW, stroke_width=3).move_to(p)
                         for p in (L(0, 0), Bp, Dp)])
        I_, Bn = _lattice_in_triangle((0, 0), (3, 1), (2, 1))
        check((I_, Bn) == (0, 3), "I = 0, B = 3")
        r_ib = tag("I = 0,    B = 3", 26)
        _left_at(r_ib, 1.2, 0.8)
        self.play(Create(rings), FadeIn(r_ib), run_time=0.8)
        r_pick = tag("A  =  I + B/2 − 1  =  ½", 26)
        _left_at(r_pick, 1.2, 0.1)
        tri_pts = [L(0, 0), Bp, Dp]
        self.play(FadeIn(r_pick), run_time=0.7)
        self.hold(0.4)

        # half of the parallelogram on (5, 2) and (2, 1)
        Mid = (Bp + Dp) / 2
        twin = tri.copy().set_fill(GOLD_E, 0.6)
        self.add(twin)
        for th in np.linspace(0, -PI, 25):           # the turning copy stays in view
            for v_ in tri_pts:
                q_ = Mid + np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                                     [0, 0, 1]]) @ (v_ - Mid)
                check(abs(q_[0]) <= SAFE_X and SAFE_BOTTOM <= q_[1] <= SAFE_TOP,
                      "the half-turn stays inside the frame")
        self.play(Rotate(twin, angle=-PI, about_point=Mid), run_time=1.2)
        check(_same_poly(_verts(twin), [L(5, 2), Dp, Bp], 1e-6), "the half-turn completes it")
        para = Polygon(L(0, 0), Bp, L(5, 2), Dp, stroke_color=YELLOW_B, stroke_width=3,
                       fill_opacity=0)
        check(close(abs(area([(0, 0), (3, 1), (5, 2), (2, 1)])), 3 * 1 - 1 * 2), "area = det")
        r_det = tag("2A  =  3 · 1 − 1 · 2  =  1", 26)
        _left_at(r_det, 1.2, -0.6)
        self.play(Create(para), FadeIn(r_det), run_time=0.8)
        r_gen = tag("b · c − a · d  =  1", 30, YELLOW_B)
        _left_at(r_gen, 1.2, -1.4)
        self.play(FadeIn(r_gen), run_time=0.7)
        self.hold(0.4)

        # every pair of neighbours: the fan splits into such triangles
        fills = VGroup()
        for i, ((a, b), (c, d)) in enumerate(zip(F4, F4[1:])):
            if (a, b) == A_:
                continue
            fills.add(mk([L(0, 0), L(b, a), L(d, c)], [BLUE_D, TEAL_D][i % 2], 0.45,
                         stroke_width=1.5))
        self.bring_to_back(fills)
        self.bring_to_back(dots)
        self.play(FadeIn(fills), FadeOut(rings), run_time=1.0)
        self.bring_to_front(tri, twin, para, *fdots.values())

        # hygiene
        texts = [flist, r_pt, r_nb, r_ib, r_pick, r_det, r_gen]
        for t_ in texts:
            check(_inside(t_), "J30 text inside")
            check(t_ is flist or t_.get_left()[0] > L(5, 0)[0] + 0.4, "panel right of the lattice")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.05), "J30 texts apart")
        segs = ([(r.get_start(), r.get_end()) for r in rays.values()]
                + _poly_segs([L(0, 0), Bp, L(5, 2), Dp]) + _poly_segs([L(0, 0), Bp, Dp])
                + _poly_segs([L(0, 0), L(N, 0), L(N, N)]))
        for lab in xnum + ynum + [bl, al]:
            check(_hit(lab, segs, 0.04) is None, f"axis number {lab.text} clear of the lines")
            check(_inside(lab) and all(_apart(lab, d_, 0.03) for d_ in dots),
                  "axis numbers clear of the dots")
        cap = caption("Farey neighbours a/b < c/d :   area ½   ⟹   bc − ad = 1", 32)
        check(cap.width < 13.0, "J30 caption at full size")
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K22

class K22_FivePointsSquare(Board):
    """Cut the unit square into four squares of side ½. Five points, four
    squares: two points lie in the same small square (a point on a cut
    belongs to both — either will do). Their horizontal and vertical
    distances are at most ½ each, so their distance is at most
    √(½² + ½²) = √2/2: stretch the legs of their right triangle to ½ and
    its hypotenuse grows into the small square's diagonal. The four
    corners and the centre show that √2/2 cannot be improved."""

    def construct(self):
        pts = [(0.15, 0.78), (0.82, 0.88), (0.32, 0.18), (0.6, 0.42), (0.9, 0.17)]
        boxes = [(0, 0), (1, 0), (0, 1), (1, 1)]           # (i, j): x ∈ [i/2, (i+1)/2], …

        def box_of(p):
            return (min(int(p[0] * 2), 1), min(int(p[1] * 2), 1))
        members = {b: [p for p in pts if box_of(p) == b] for b in boxes}
        full = [b for b in boxes if len(members[b]) >= 2]
        check(len(full) == 1 and full[0] == (1, 0), "exactly one square holds two points here")
        p, q = members[(1, 0)]
        d = float(np.hypot(p[0] - q[0], p[1] - q[1]))
        check(abs(p[0] - q[0]) <= 0.5 and abs(p[1] - q[1]) <= 0.5 and d <= np.sqrt(2) / 2,
              "legs ≤ ½, d ≤ √2/2")
        rng = np.random.default_rng(7)
        for _ in range(4000):                       # the claim, on random five-point sets
            S_ = rng.random((5, 2))
            dmin = min(np.hypot(*(S_[i] - S_[j])) for i in range(5) for j in range(i + 1, 5))
            check(dmin <= np.sqrt(2) / 2 + 1e-12, "some two of five are within √2/2")
        ext = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
        dext = min(np.hypot(ext[i][0] - ext[j][0], ext[i][1] - ext[j][1])
                   for i in range(5) for j in range(i + 1, 5))
        check(close(dext, np.sqrt(2) / 2), "corners + centre: the least distance is √2/2")

        s = 5.0
        S0 = np.array([-6.15, -2.7, 0.0])

        def Q(x, y):
            return S0 + s * np.array([x, y, 0.0])

        sq = Polygon(Q(0, 0), Q(1, 0), Q(1, 1), Q(0, 1), stroke_color=WHITE, stroke_width=3,
                     fill_opacity=0)
        l1a = tag("1", 28).next_to(Line(Q(0, 0), Q(0, 1)), LEFT, buff=0.15)
        l1b = tag("1", 28).next_to(Line(Q(0, 1), Q(1, 1)), UP, buff=0.12)
        dots = [Dot(Q(*pp), radius=0.09, color=WHITE) for pp in pts]
        self.play(Create(sq), FadeIn(l1a), FadeIn(l1b), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(dd, scale=1.6) for dd in dots], lag_ratio=0.2),
                  run_time=1.0)

        # four half-size squares
        cuts = VGroup(DashedLine(Q(0.5, 0), Q(0.5, 1), color=GREY_A, stroke_width=2.5,
                                 dash_length=0.12),
                      DashedLine(Q(0, 0.5), Q(1, 0.5), color=GREY_A, stroke_width=2.5,
                                 dash_length=0.12))
        bcol = {(0, 0): BLUE_E, (1, 0): GOLD_E, (0, 1): TEAL_E, (1, 1): PURPLE_E}
        tiles = {b: Polygon(Q(b[0] / 2, b[1] / 2), Q(b[0] / 2 + 0.5, b[1] / 2),
                            Q(b[0] / 2 + 0.5, b[1] / 2 + 0.5), Q(b[0] / 2, b[1] / 2 + 0.5),
                            stroke_width=0, fill_color=bcol[b], fill_opacity=0.45)
                 for b in boxes}
        check(_tiles_exactly([_verts(t_) for t_ in tiles.values()],
                             [Q(0, 0), Q(1, 0), Q(1, 1), Q(0, 1)]), "four squares tile the square")
        self.add(*tiles.values())
        self.bring_to_back(*tiles.values())
        lh = tag("½", 26, GREY_A).next_to(Line(Q(0.5, 1), Q(1, 1)), UP, buff=0.12)
        lh0 = tag("½", 26, GREY_A).next_to(Line(Q(0, 1), Q(0.5, 1)), UP, buff=0.12)
        self.play(Create(cuts), *[FadeIn(t_) for t_ in tiles.values()], FadeIn(lh),
                  FadeIn(lh0), FadeOut(l1b), run_time=1.0)
        r1 = tag("5 points, 4 squares  ⟹  2 in one", 26)
        _left_at(r1, -0.45, 3.3)
        hi = Polygon(Q(0.5, 0), Q(1, 0), Q(1, 0.5), Q(0.5, 0.5), stroke_color=YELLOW,
                     stroke_width=5, fill_opacity=0)
        self.play(FadeIn(r1), Create(hi), *[dots[pts.index(pp)].animate.set_color(YELLOW)
                                            for pp in (p, q)], run_time=1.0)
        self.hold(0.4)

        # their right triangle: legs at most ½ — stretch them to ½
        P_, Q_ = Q(*p), Q(*q)
        Cn = Q(q[0], p[1])                                   # the right-angle corner
        seg = Line(P_, Q_, color=YELLOW, stroke_width=5)
        tri = Polygon(P_, Cn, Q_, stroke_color=WHITE, stroke_width=2.5,
                      fill_color=YELLOW, fill_opacity=0.25)
        legs_lab = [tag("Δx", 22).next_to(Line(P_, Cn), UP, buff=0.08),
                    tag("Δy", 22).next_to(Line(Cn, Q_), RIGHT, buff=0.08)]
        self.play(Create(seg), FadeIn(tri), *[FadeIn(t_) for t_ in legs_lab], run_time=0.9)
        r2 = tag("Δx ≤ ½,   Δy ≤ ½", 26)
        _left_at(r2, -0.45, 2.6)
        self.play(FadeIn(r2), run_time=0.6)
        big = Polygon(Q(0.5, 0.5), Q(1, 0.5), Q(1, 0), stroke_color=WHITE, stroke_width=2.5,
                      fill_color=YELLOW, fill_opacity=0.25)
        check(_same_poly(_verts(big), [Q(0.5, 0.5), Q(1, 0.5), Q(1, 0)]), "the box's half")
        grow = tri.copy()
        self.add(grow)
        self.play(Transform(grow, big), FadeOut(legs_lab[0]), FadeOut(legs_lab[1]),
                  run_time=1.3)
        diag = Line(Q(0.5, 0.5), Q(1, 0), color=YELLOW_B, stroke_width=4)
        ld = tag("√2/2", 26, YELLOW_B)
        # beside the diagonal, inside the empty upper half of the small square
        _beside(ld, Q(0.5, 0.5), Q(1, 0), Q(0.5, 0), gap=0.14, at=0.55)
        r3 = tag("d  ≤  √(½² + ½²)  =  √2/2", 28, YELLOW_B)
        _left_at(r3, -0.45, 1.9)
        self.play(Create(diag), FadeIn(ld), FadeIn(r3), run_time=0.9)
        self.hold(0.5)

        # the bound is attained: four corners and the centre
        t0 = np.array([0.15, -2.75, 0.0])
        e = 2.45

        def E(x, y):
            return t0 + e * np.array([x, y, 0.0])
        esq = Polygon(E(0, 0), E(1, 0), E(1, 1), E(0, 1), stroke_color=WHITE, stroke_width=3,
                      fill_opacity=0)
        ecuts = VGroup(DashedLine(E(0.5, 0), E(0.5, 1), color=GREY_A, stroke_width=2,
                                  dash_length=0.1),
                       DashedLine(E(0, 0.5), E(1, 0.5), color=GREY_A, stroke_width=2,
                                  dash_length=0.1))
        edots = [Dot(E(*pp), radius=0.09, color=YELLOW) for pp in ext]
        espokes = VGroup(*[Line(E(0.5, 0.5), E(*c_), color=YELLOW_B, stroke_width=3)
                           for c_ in ext[:4]])
        r4 = tag("corners + centre:", 26)
        r5 = tag("each pair  ≥  √2/2", 26, YELLOW_B)
        _left_at(r4, 3.05, -1.15)
        _left_at(r5, 3.05, -1.8)
        self.play(Create(esq), Create(ecuts), *[FadeIn(x) for x in edots], run_time=0.9)
        self.play(Create(espokes), FadeIn(r4), FadeIn(r5), run_time=0.9)

        # hygiene
        segs = (_poly_segs([Q(0, 0), Q(1, 0), Q(1, 1), Q(0, 1)])
                + [(Q(0.5, 0), Q(0.5, 1)), (Q(0, 0.5), Q(1, 0.5)), (P_, Q_),
                   (Q(0.5, 0.5), Q(1, 0)), (Q(0.5, 0.5), Q(1, 0.5)), (Q(1, 0.5), Q(1, 0))]
                + _poly_segs([E(0, 0), E(1, 0), E(1, 1), E(0, 1)])
                + [(E(0.5, 0.5), E(*c_)) for c_ in ext[:4]]
                + [(E(0.5, 0), E(0.5, 1)), (E(0, 0.5), E(1, 0.5))])
        for dd in dots:
            segs += _dot_segs(dd.get_center(), 0.09)
        labels = [l1a, lh0, lh, ld, r1, r2, r3, r4, r5]
        _labels_ok(labels, segs, "K22", pad=0.05, gap=0.06)
        check(_apart(VGroup(esq, *edots), VGroup(r1, r2, r3), 0.2), "the second square clear")
        cap = caption("five points in a unit square:  two of them are at most √2/2 apart", 28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K17

class K17_ChooseTwo(Board):
    """Put the n + 1 = 6 things in a row. A pair {i, j}, i < j, is the apex
    of a tent with its feet on i and j — one apex for each pair, and each
    apex gives back its pair. Sort the pairs by their larger member j: their
    apexes lie on the line rising to the left from j, one for each partner
    i = 1, …, j − 1. So the lines hold 1, 2, 3, 4, 5 apexes and
    C(6, 2) = 1 + 2 + 3 + 4 + 5 = 15; in general C(n + 1, 2) = 1 + 2 + … + n."""

    def construct(self):
        from math import comb
        m = 6                                          # n + 1 things
        pairs = [(i, j) for j in range(2, m + 1) for i in range(1, j)]
        check(len(pairs) == comb(m, 2) == sum(range(1, m)), "C(6, 2) = 1 + … + 5 = 15")
        for N_ in range(2, 40):
            check(comb(N_ + 1, 2) == N_ * (N_ + 1) // 2 == sum(range(1, N_ + 1)),
                  "C(n + 1, 2) = 1 + 2 + … + n")

        sp, y0 = 1.7, -2.3

        def base(i):
            return np.array([-4.25 + (i - 1) * sp, y0, 0.0])

        def apex(i, j):
            return np.array([(base(i)[0] + base(j)[0]) / 2,
                             y0 + (base(j)[0] - base(i)[0]) / 2, 0.0])
        # one apex per pair, all different; group j on the line rising left from j
        apx = {pr: apex(*pr) for pr in pairs}
        check(len({tuple(np.round(v, 9)) for v in apx.values()}) == len(pairs),
              "different pairs, different apexes")
        for (i, j), v in apx.items():
            w = v - base(j)
            check(close(w[0], -w[1]) and w[0] < 0, "the apex lies on the line rising left from j")
            w2 = v - base(i)
            check(close(w2[0], w2[1]) and w2[0] > 0, "and on the line rising right from i")

        gcol = {2: BLUE_B, 3: TEAL_B, 4: GREEN_B, 5: GOLD_B, 6: RED_B}
        bdots = [Dot(base(i), radius=0.1, color=WHITE) for i in range(1, m + 1)]
        blabs = [tag(str(i), 28).next_to(base(i), DOWN, buff=0.2) for i in range(1, m + 1)]
        self.play(LaggedStart(*[FadeIn(VGroup(d_, l_)) for d_, l_ in zip(bdots, blabs)],
                              lag_ratio=0.15), run_time=1.0)

        # one pair, one tent
        ex = (2, 5)
        tent = VGroup(Line(base(ex[0]), apx[ex], color=WHITE, stroke_width=3),
                      Line(apx[ex], base(ex[1]), color=WHITE, stroke_width=3))
        exd = Dot(apx[ex], radius=0.1, color=WHITE)
        exl = tag("{2, 5}", 26).next_to(apx[ex], UP, buff=0.18)
        r1 = tag("pair {i, j}  ↔  the apex over i and j", 26)
        _left_at(r1, -6.4, 3.35)
        self.play(Create(tent), FadeIn(exd), FadeIn(exl), FadeIn(r1), run_time=1.2)
        self.hold(0.5)
        self.play(FadeOut(tent), FadeOut(exd), FadeOut(exl), run_time=0.5)

        # sorted by the larger member j: j − 1 apexes on the line from j
        r2 = tag("larger member j :   j − 1 partners", 26)
        _left_at(r2, -6.4, 2.75)
        self.play(FadeIn(r2), run_time=0.5)
        lines, legs, adots, counts = {}, [], [], {}
        for j in range(2, m + 1):
            top = apx[(1, j)]
            lines[j] = Line(base(j), top, color=gcol[j], stroke_width=4)
            lg = VGroup(*[Line(apx[(i, j)], base(i), color=GREY_B, stroke_width=1.5)
                          for i in range(1, j)])
            ds = VGroup(*[Dot(apx[(i, j)], radius=0.09, color=gcol[j]) for i in range(1, j)])
            counts[j] = tag(str(j - 1), 28, gcol[j]).move_to(top + np.array([-0.32, 0.32, 0]))
            self.play(Create(lines[j]), Create(lg), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(d_, scale=1.5) for d_ in ds], lag_ratio=0.15),
                      FadeIn(counts[j]), run_time=0.5)
            legs.append(lg)
            adots.append(ds)
        self.bring_to_front(*adots, *bdots)
        self.hold(0.3)

        total = _trow([("1 + 2 + 3 + 4 + 5", WHITE, 0.22), ("=", WHITE, 0.22),
                       ("15", YELLOW_B)], size=28)
        total.shift(np.array([1.0 - total.get_left()[0], 1.6, 0.0]))
        total2 = _trow([("=", WHITE, 0.22), ("C(6, 2)", YELLOW_B)], size=28)
        total2.shift(np.array([total[1].get_left()[0] - total2.get_left()[0], 0.95, 0.0]))
        self.play(FadeIn(total), *[Indicate(counts[j], color=gcol[j], scale_factor=1.4)
                                   for j in counts], run_time=1.0)
        self.play(FadeIn(total2), run_time=0.6)

        # hygiene: the counts sit just outside the triangle, clear of everything
        segs = ([(l_.get_start(), l_.get_end()) for l_ in lines.values()]
                + [(l_.get_start(), l_.get_end()) for lg in legs for l_ in lg])
        for v in list(apx.values()) + [base(i) for i in range(1, m + 1)]:
            segs += _dot_segs(v, 0.1)
        labels = [r1, r2, total, total2] + list(counts.values()) + blabs
        _labels_ok(labels, segs, "K17", pad=0.04, gap=0.06)
        cap = caption("C(n + 1, 2)  =  1 + 2 + … + n", 36)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)


# ===================================================================== K27

class K27_BrokenStick(Board):
    """Two independent uniform cuts X, Y of a stick of length 1 are a point
    spread evenly over the unit square. Swapping the cuts gives the same
    pieces, so fold the square along its diagonal: the point lies evenly in
    the half X < Y, and the pieces are a = X, b = Y − X, c = 1 − Y. They
    form a triangle exactly when each is shorter than the other two
    together, i.e. shorter than ½ (since a + b + c = 1): left of X = ½,
    above Y = ½, below Y − X = ½. These three lines are the midlines of the
    half-square, which they cut into four congruent triangles: the pieces
    form a triangle in the middle one, probability ¼. An affine map keeps
    area ratios; it carries the half-square onto the equilateral triangle of
    height 1, where a, b, c are the distances to the three sides (Viviani)
    and the good region is the middle quarter."""

    def construct(self):
        X_, Y_ = 0.2, 0.55
        a_, b_, c_ = X_, Y_ - X_, 1 - Y_
        check(max(a_, b_, c_) < 0.5, "this example makes a triangle")
        for p in [(0.2, 0.3), (0.6, 0.9), (0.45, 0.55), (0.1, 0.95)]:   # spot checks
            aa, bb, cc = p[0], p[1] - p[0], 1 - p[1]
            tri_ok = aa + bb > cc and bb + cc > aa and aa + cc > bb
            check(tri_ok == (max(aa, bb, cc) < 0.5), "triangle ⟺ every piece < ½")
        half = [(0, 0), (0, 1), (1, 1)]
        mids = {"L": (0, 0.5), "T": (0.5, 1), "D": (0.5, 0.5)}
        quarters = [[mids["L"], mids["D"], mids["T"]],            # the middle one
                    [(0, 0), mids["D"], mids["L"]], [mids["L"], mids["T"], (0, 1)],
                    [mids["D"], (1, 1), mids["T"]]]
        check(_tiles_exactly(quarters, half), "the midlines cut the half-square in four")
        check(all(close(abs(area(q)), 0.125) for q in quarters), "four congruent quarters")
        mid = quarters[0]
        for pt in [(0.2, 0.6), (0.4, 0.8), (0.45, 0.55)]:
            inside = _pip(pt, mid)
            aa, bb, cc = pt[0], pt[1] - pt[0], 1 - pt[1]
            check(inside == (max(aa, bb, cc) < 0.5), "the middle quarter is a, b, c < ½")
        check(close(abs(area(mid)) / abs(area(half)), 0.25), "P = ¼")

        col = {"a": BLUE_C, "b": GREEN_C, "c": ORANGE}

        # ---- 1. the stick and two cuts
        sl = 5.0
        s0 = np.array([-6.1, 3.15, 0.0])

        def ST(t):
            return s0 + np.array([sl * t, 0.0, 0.0])
        pa = Line(ST(0), ST(X_), color=col["a"], stroke_width=9)
        pb = Line(ST(X_), ST(Y_), color=col["b"], stroke_width=9)
        pc = Line(ST(Y_), ST(1), color=col["c"], stroke_width=9)
        cutX = Line(ST(X_) + UP * 0.2, ST(X_) + DOWN * 0.2, color=WHITE, stroke_width=3)
        cutY = Line(ST(Y_) + UP * 0.2, ST(Y_) + DOWN * 0.2, color=WHITE, stroke_width=3)
        lX = tag("X", 24).next_to(cutX, UP, buff=0.06)
        lY = tag("Y", 24).next_to(cutY, UP, buff=0.06)
        la = tag("a", 26, col["a"]).next_to(pa, DOWN, buff=0.14)
        lb = tag("b", 26, col["b"]).next_to(pb, DOWN, buff=0.14)
        lc = tag("c", 26, col["c"]).next_to(pc, DOWN, buff=0.14)
        self.play(Create(pa), Create(pb), Create(pc), run_time=0.8)
        self.play(Create(cutX), Create(cutY), FadeIn(lX), FadeIn(lY), FadeIn(la), FadeIn(lb),
                  FadeIn(lc), run_time=0.7)

        # the pieces close up into a triangle: a and c swing down about the cuts
        t_ = (a_ ** 2 - c_ ** 2 + b_ ** 2) / (2 * b_)
        h_ = np.sqrt(a_ ** 2 - t_ ** 2)
        apex = ST(X_) + sl * np.array([t_, -h_, 0.0])
        ang_a = float(np.arctan2(-h_, t_)) - PI          # from pointing left
        ang_c = float(np.arctan2(-h_, t_ - b_))          # from pointing right
        if ang_a < 0:
            ang_a += TAU
        if ang_c > 0:
            ang_c -= TAU
        sa, sc = pa.copy(), pc.copy()
        self.add(sa, sc)
        self.play(FadeOut(la), FadeOut(lb), FadeOut(lc),
                  Rotate(sa, angle=ang_a, about_point=ST(X_)),
                  Rotate(sc, angle=ang_c, about_point=ST(Y_)),
                  pa.animate.set_stroke(opacity=0.25), pc.animate.set_stroke(opacity=0.25),
                  run_time=1.3)
        check(close(sa.get_start(), apex, 1e-6) or close(sa.get_end(), apex, 1e-6),
              "a reaches the apex")
        check(close(sc.get_start(), apex, 1e-6) or close(sc.get_end(), apex, 1e-6),
              "c reaches the apex")
        R1 = tag("a + b + c = 1 :   a + b > c   ⟺   c < ½", 26)
        _left_at(R1, -0.3, 3.3)
        R1b = tag("triangle   ⟺   a, b, c  <  ½", 26, YELLOW_B)
        _left_at(R1b, -0.3, 2.7)
        self.play(FadeIn(R1), run_time=0.7)
        self.play(FadeIn(R1b), run_time=0.6)
        self.hold(0.3)
        self.play(Rotate(sa, angle=-ang_a, about_point=ST(X_)),
                  Rotate(sc, angle=-ang_c, about_point=ST(Y_)), run_time=0.8)
        self.remove(sa, sc)
        self.play(pa.animate.set_stroke(opacity=1.0), pc.animate.set_stroke(opacity=1.0),
                  FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=0.3)

        # ---- 2. the square of the two cuts
        k = 3.5
        q0 = np.array([-6.0, -2.45, 0.0])

        def SQ(x, y):
            return q0 + k * np.array([x, y, 0.0])
        sqr = Polygon(SQ(0, 0), SQ(1, 0), SQ(1, 1), SQ(0, 1), stroke_color=WHITE,
                      stroke_width=3, fill_opacity=0)
        diag = DashedLine(SQ(0, 0), SQ(1, 1), color=GREY_A, stroke_width=2, dash_length=0.1)
        upper = mk([SQ(*p) for p in half], GREY_D, 0.55, stroke_width=0)
        lower = mk([SQ(0, 0), SQ(1, 0), SQ(1, 1)], GREY_D, 0.55, stroke_width=0)
        t0 = tag("0", 20, GREY_A).next_to(SQ(0, 0), DL, buff=0.06)
        t1x = tag("1", 20, GREY_A).next_to(SQ(1, 0), DOWN, buff=0.1)
        t1y = tag("1", 20, GREY_A).next_to(SQ(0, 1), LEFT, buff=0.1)
        pt = Dot(SQ(X_, Y_), radius=0.09, color=YELLOW)
        pt2 = Dot(SQ(Y_, X_), radius=0.09, color=YELLOW)
        gx = DashedLine(SQ(X_, Y_), SQ(X_, 0), color=GREY_B, stroke_width=1.5, dash_length=0.07)
        gy = DashedLine(SQ(X_, Y_), SQ(0, Y_), color=GREY_B, stroke_width=1.5, dash_length=0.07)
        tX = tag("X", 22).next_to(SQ(X_, 0), DOWN, buff=0.1)
        tY = tag("Y", 22).next_to(SQ(0, Y_), LEFT, buff=0.1)
        self.add(upper, lower)
        self.bring_to_back(upper, lower)
        self.play(Create(sqr), FadeIn(upper), FadeIn(lower), FadeIn(t0), FadeIn(t1x),
                  FadeIn(t1y), run_time=0.9)
        self.play(FadeIn(pt, scale=1.6), Create(gx), Create(gy), FadeIn(tX), FadeIn(tY),
                  run_time=0.9)

        # ---- 3. the order of the cuts does not matter: fold along the diagonal
        self.play(Create(diag), FadeIn(pt2), run_time=0.6)
        lower2 = VGroup(lower, pt2)
        self.play(_flip(lower2, SQ(0, 0), SQ(1, 1)), run_time=1.2)
        check(_same_poly(_verts(lower), [SQ(*p) for p in half], 1e-6),
              "the lower half lands on the upper")
        check(close(pt2.get_center(), pt.get_center(), 1e-6), "(Y, X) lands on (X, Y)")
        self.remove(lower, pt2)
        upper.set_fill(GREY_D, 0.8)
        R2 = tag("X < Y :   a = X,   b = Y − X,   c = 1 − Y", 26)
        _left_at(R2, -0.3, 2.1)
        self.play(FadeIn(R2), run_time=0.6)

        # ---- 4. the three conditions are the midlines of the half-square
        ml = {"a": Line(SQ(0.5, 0.5), SQ(0.5, 1), color=col["a"], stroke_width=4),
              "c": Line(SQ(0, 0.5), SQ(0.5, 0.5), color=col["c"], stroke_width=4),
              "b": Line(SQ(0, 0.5), SQ(0.5, 1), color=col["b"], stroke_width=4)}
        mlab = {"a": tag("a = ½", 20, col["a"]), "c": tag("c = ½", 20, col["c"]),
                "b": tag("b = ½", 20, col["b"])}
        _beside(mlab["a"], SQ(0.5, 0.5), SQ(0.5, 1), RIGHT, gap=0.1, at=0.84)
        _beside(mlab["c"], SQ(0, 0.5), SQ(0.5, 0.5), DOWN, gap=0.1, at=0.42)
        r_in = (0.5 + 0.5 - 0.5 * np.sqrt(2)) / 2       # incircle of the corner quarter
        mlab["b"].move_to(SQ(r_in, 1 - r_in))
        self.play(FadeOut(gx), FadeOut(gy), run_time=0.3)
        for key in ("a", "c", "b"):
            self.play(Create(ml[key]), FadeIn(mlab[key]), run_time=0.6)
        good = mk([SQ(*p) for p in mid], YELLOW, 0.55, stroke_width=0)
        self.add(good)
        self.bring_to_back(good)
        self.bring_to_back(upper)
        q14 = tag("¼", 30, BLACK).move_to(SQ(0.42, 0.74))
        check(all(_pip(c_, [SQ(*p) for p in mid]) for c_ in (
            q14.get_corner(UL), q14.get_corner(UR), q14.get_corner(DL), q14.get_corner(DR))),
            "¼ inside the middle quarter")
        self.play(FadeIn(good), run_time=0.6)
        R3 = tag("P  =  ¼", 32, YELLOW_B)
        _left_at(R3, -0.3, 1.4)
        self.bring_to_front(pt)
        self.play(FadeIn(q14), FadeIn(R3), run_time=0.6)
        self.hold(0.4)

        # ---- 5. the same picture as the triangle of lengths (Viviani)
        H = 3.4
        side = 2 * H / np.sqrt(3)
        Vb = np.array([1.05, -2.45, 0.0])
        Vc = Vb + np.array([side, 0.0, 0.0])
        Va = Vb + np.array([side / 2, H, 0.0])
        img = {(0, 0): Vc, (0, 1): Vb, (1, 1): Va}

        def AFF(p):                                   # barycentric (a, b, c) = (x, y − x, 1 − y)
            return p[0] * Va + (p[1] - p[0]) * Vb + (1 - p[1]) * Vc
        for key_, v_ in img.items():
            check(close(AFF(key_), v_), "the affine map on the corners")
        eq_q = [[AFF(p) for p in q] for q in quarters]
        check(_tiles_exactly(eq_q, [Vb, Vc, Va]), "four quarters of the triangle")
        check(all(close(abs(area(q)), abs(area([Vb, Vc, Va])) / 4) for q in eq_q),
              "equal quarters")
        P_ = AFF((X_, Y_))
        # distances to the sides are a, b, c times the height
        def dist(p, u, v):
            u, v = to3(u), to3(v)
            d = v - u
            return abs(d[0] * (p - u)[1] - d[1] * (p - u)[0]) / np.linalg.norm(d)
        check(close(dist(P_, Vb, Vc), a_ * H) and close(dist(P_, Vc, Va), b_ * H)
              and close(dist(P_, Va, Vb), c_ * H), "distances = a, b, c")
        src = VGroup(upper.copy(), good.copy(), *[ml[k_].copy() for k_ in ("a", "c", "b")],
                     pt.copy())
        tgt = VGroup(mk([Vc, Vb, Va], GREY_D, 0.8, stroke_width=0),
                     mk([AFF(p) for p in mid], YELLOW, 0.55, stroke_width=0),
                     Line(AFF((0.5, 0.5)), AFF((0.5, 1)), color=col["a"], stroke_width=4),
                     Line(AFF((0, 0.5)), AFF((0.5, 0.5)), color=col["c"], stroke_width=4),
                     Line(AFF((0, 0.5)), AFF((0.5, 1)), color=col["b"], stroke_width=4),
                     Dot(P_, radius=0.09, color=YELLOW))
        # match vertex order of the triangle fills to the map
        src[0].set_points_as_corners([SQ(0, 0), SQ(0, 1), SQ(1, 1), SQ(0, 0)])
        tgt[0].set_points_as_corners([AFF((0, 0)), AFF((0, 1)), AFF((1, 1)), AFF((0, 0))])
        src[1].set_points_as_corners([SQ(*mid[0]), SQ(*mid[1]), SQ(*mid[2]), SQ(*mid[0])])
        tgt[1].set_points_as_corners([AFF(mid[0]), AFF(mid[1]), AFF(mid[2]), AFF(mid[0])])
        self.add(src)
        self.play(Transform(src, tgt), run_time=1.8)
        eq_out = Polygon(Vb, Vc, Va, stroke_color=WHITE, stroke_width=3, fill_opacity=0)
        feet = [Vb + np.dot(P_ - Vb, _u(Vc - Vb)) * _u(Vc - Vb),
                Vc + np.dot(P_ - Vc, _u(Va - Vc)) * _u(Va - Vc),
                Va + np.dot(P_ - Va, _u(Vb - Va)) * _u(Vb - Va)]
        perp = VGroup(*[Line(P_, f_, color=col[k_], stroke_width=4)
                        for f_, k_ in zip(feet, ("a", "b", "c"))])
        q14b = tag("¼", 30, BLACK).move_to(np.array([Va[0], Vb[1] + 0.4 * H, 0.0]))
        med = [AFF(p) for p in mid]
        check(all(_pip(c_, med) for c_ in (q14b.get_corner(UL), q14b.get_corner(UR),
                                            q14b.get_corner(DL), q14b.get_corner(DR))),
              "¼ inside the middle quarter")
        self.play(Create(eq_out), Create(perp), FadeIn(q14b), run_time=1.0)
        R4 = tag("a + b + c  =  1  =  height", 24)
        _left_at(R4, 2.1, 1.45)
        self.play(FadeIn(R4), run_time=0.6)

        # hygiene
        texts = [R1, R1b, R2, R3, R4, lX, lY, la, lb, lc, t0, t1x, t1y, tX, tY] + list(
            mlab.values())
        for i in range(len(texts)):
            check(_inside(texts[i]), f"K27: '{_name(texts[i])}' inside")
            for j in range(i + 1, len(texts)):
                check(_apart(texts[i], texts[j], 0.05),
                      f"K27: '{_name(texts[i])}' clear of '{_name(texts[j])}'")
        sq_segs = (_poly_segs([SQ(0, 0), SQ(1, 0), SQ(1, 1), SQ(0, 1)])
                   + [(SQ(0, 0), SQ(1, 1))] + [(m_.get_start(), m_.get_end())
                                               for m_ in ml.values()]
                   + _dot_segs(SQ(X_, Y_), 0.09))
        for key in mlab:
            check(_hit(mlab[key], sq_segs, 0.04) is None, f"label {key} = ½ clear")
        eq_segs = (_poly_segs([Vb, Vc, Va]) + [(P_, f_) for f_ in feet]
                   + [(AFF((0.5, 0.5)), AFF((0.5, 1))), (AFF((0, 0.5)), AFF((0.5, 0.5))),
                      (AFF((0, 0.5)), AFF((0.5, 1)))])
        check(_hit(q14, sq_segs, 0.03) is None and _hit(q14b, eq_segs, 0.04) is None,
              "the ¼ labels clear")
        check(_apart(q14b, Dot(P_, radius=0.09), 0.1), "¼ clear of the point")
        check(_apart(R4, VGroup(eq_out), 0.1) and _apart(R3, VGroup(eq_out), 0.1),
              "readouts clear of the triangle")
        check(_apart(VGroup(pa, pb, pc, la, lb, lc), sqr, 0.3), "stick clear of the square")
        cap = caption("two random cuts:  the pieces form a triangle with probability ¼", 28)
        self.play(Write(cap), run_time=1.2)
        _final_check(self, cap)
        self.hold(2.2)
