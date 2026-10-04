# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes_ag.py — proofs without words: A4, A7, A8, G1, G3, G4, G5, G6, G7.

Built on pww_kit (Frame, mk, tag, caption, guide, angle_arc, check, ...).
Every scene states its geometric invariants with check(...), so a wrong
construction fails the render instead of drawing a wrong picture.
"""

from pww_kit import *


# ------------------------------------------------------------ private helpers

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
    """True if the polygons tile `region` with no gap and no overlap:
    on a jittered grid over the region's bounding box, every point inside
    the region lies in exactly one polygon and every point outside it in
    none."""
    xs = [float(p[0]) for p in region]
    ys = [float(p[1]) for p in region]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for i in range(n):
        for j in range(n):
            # irrational offsets keep the samples off every edge
            x = x0 + (x1 - x0) * (i + 0.5 + 0.0731 * np.sqrt(2)) / (n + 0.2)
            y = y0 + (y1 - y0) * (j + 0.5 + 0.0597 * np.sqrt(3)) / (n + 0.2)
            k = sum(_pip((x, y), P) for P in polys)
            if k != (1 if _pip((x, y), region) else 0):
                return False
    return True


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at screen point v between the directions to p, q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1 = (p - v) / np.linalg.norm(p - v)
    u2 = (q - v) / np.linalg.norm(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _chord(M, d, sq):
    """Where the line M + t d leaves the polygon sq: [(edge index, point)]."""
    out = []
    n = len(sq)
    for i in range(n):
        u, v = np.asarray(sq[i], float), np.asarray(sq[(i + 1) % n], float)
        e = v - u
        mat = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(mat)) < 1e-12:
            continue
        t, s = np.linalg.solve(mat, u - M)
        if -1e-12 <= s <= 1 + 1e-12:
            out.append((i, M + t * d))
    return out


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ============================================================== A. PYTHAGORAS

class A4_Perigal(Board):
    """Perigal's dissection. Two lines through the centre of the larger leg
    square, one parallel and one perpendicular to the hypotenuse, cut it
    into four congruent quadrilaterals (a quarter-turn about the centre
    carries each onto the next). Each cut is exactly c long, so the four
    half-cuts of length c/2 meeting at right angles fit the four corners of
    the hypotenuse square; the pieces get there by translation alone, and
    the smaller leg square fills the hole in the middle."""

    def construct(self):
        a, b = 2.0, 4.0                      # legs, a < b
        c = float(np.hypot(a, b))
        A = np.array([0.0, 0.0])             # hypotenuse AB, vertical
        B = np.array([0.0, c])
        C = np.array([-a * b / c, a * a / c])        # the right angle
        na = np.array([-a * a / c, -a * b / c])      # outward edge, square on AC
        nb = np.array([-b * b / c, a * b / c])       # outward edge, square on CB
        check(close(np.linalg.norm(C - A), a) and close(np.linalg.norm(B - C), b)
              and abs(np.dot(C - A, B - C)) < 1e-9, "right triangle a, b, c")
        check(abs(np.dot(na, C - A)) < 1e-9 and abs(np.dot(nb, B - C)) < 1e-9
              and np.dot(na, B - A) < 0 and np.dot(nb, A - C) < 0,
              "leg squares are erected outwards")
        Sa = [A, C, C + na, A + na]                  # square on the short leg
        Sb = [C, B, B + nb, C + nb]                  # square on the long leg
        E = np.array([c, 0.0])
        Sc = [A, E, E + B, B]                        # square on the hypotenuse
        Mb = (Sb[0] + Sb[2]) / 2                     # centre of the b-square
        Mc = (Sc[0] + Sc[2]) / 2

        # ---- the two cuts through Mb
        par = _unit(B - A)                           # parallel to the hypotenuse
        per = np.array([par[1], -par[0]])            # perpendicular to it
        hits = _chord(Mb, par, Sb) + _chord(Mb, per, Sb)
        check(len(hits) == 4 and sorted(i for i, _ in hits) == [0, 1, 2, 3],
              "the two cuts meet each side of the b-square once")
        cut_pt = {i: p for i, p in hits}
        for p in cut_pt.values():
            check(close(np.linalg.norm(p - Mb), c / 2), "each half-cut is c/2")
        # piece k holds corner Sb[k]: Mb, cut on the edge before, corner,
        # cut on the edge after
        pieces = [[Mb, cut_pt[(k - 1) % 4], Sb[k], cut_pt[k]] for k in range(4)]
        check(close(sum(abs(area(p)) for p in pieces), b * b),
              "the four pieces make b²")
        sgn = 1.0 if area(Sb) > 0 else -1.0
        for k in range(4):
            rot = [Mb + np.array([[0, -sgn], [sgn, 0]]) @ (p - Mb)
                   for p in pieces[k]]
            nxt = pieces[(k + 1) % 4]
            check(all(any(close(r, q) for q in nxt) for r in rot),
                  "a quarter-turn about the centre maps piece k onto piece k+1")
        # where each piece goes: its two half-cuts must run along two sides
        # of the c-square from a corner, so the corner is Mc - (c/2)(u1+u2)
        shifts = []
        for p in pieces:
            u1, u2 = _unit(p[1] - Mb), _unit(p[3] - Mb)
            K = Mc - (c / 2) * (u1 + u2)
            check(any(close(K, q) for q in Sc), "piece lands on a corner")
            shifts.append(K - Mb)
        a_shift = Mc - (Sa[0] + Sa[2]) / 2
        final = [[q + s for q in p] for p, s in zip(pieces, shifts)]
        final.append([q + a_shift for q in Sa])
        check(close(sum(abs(area(p)) for p in final), c * c),
              "areas: four pieces + a² = c²")
        check(_tiles_exactly(final, Sc), "pieces + a² tile c² exactly")

        # ---- screen
        F = Frame(-b * (a + b) / c - 0.15, c + 0.15,
                  -a * b / c - 0.15, c + a * b / c + 0.15)
        P = F.P

        tri = Polygon(P(A), P(B), P(C), stroke_color=WHITE, stroke_width=3)
        ra = _ra(P(C), P(A), P(B), 0.2)
        c_out = Polygon(*[P(q) for q in Sc], stroke_color=YELLOW_B,
                        stroke_width=4)
        ghost_a = Polygon(*[P(q) for q in Sa], stroke_color=ORANGE,
                          stroke_width=1.5)
        ghost_b = Polygon(*[P(q) for q in Sb], stroke_color=TEAL_B,
                          stroke_width=1.5)
        sq_a = mk([P(q) for q in Sa], ORANGE)
        sq_b = mk([P(q) for q in Sb], TEAL_D)
        lab_a = tag("a²", 30).move_to(P((Sa[0] + Sa[2]) / 2))
        lab_b = tag("b²", 34).move_to(P(Mb))
        lab_c = tag("c²", 34, YELLOW_B).move_to(P(Mc))

        self.play(Create(tri), Create(ra), run_time=1.0)
        self.play(Create(c_out), FadeIn(ghost_a), FadeIn(ghost_b), run_time=1.0)
        self.play(FadeIn(sq_a), FadeIn(sq_b), FadeIn(lab_a), FadeIn(lab_b),
                  FadeIn(lab_c), run_time=1.0)
        self.hold(0.6)

        # the cut parallel to the hypotenuse, then the perpendicular one
        hyp = Line(P(A), P(B), color=YELLOW_B, stroke_width=8)
        cut1 = Line(P(hits[0][1]), P(hits[1][1]), color=YELLOW_B,
                    stroke_width=5)                  # parallel to AB
        cut2 = Line(P(hits[2][1]), P(hits[3][1]), color=YELLOW_B,
                    stroke_width=5)                  # perpendicular to AB
        dot = Dot(P(Mb), color=YELLOW_B, radius=0.07)
        self.play(FadeIn(dot), FadeOut(lab_b), run_time=0.5)
        self.play(Create(hyp), run_time=0.6)
        self.play(TransformFromCopy(hyp, cut1), run_time=1.1)
        self.play(FadeOut(hyp), run_time=0.3)
        mark = _ra(P(Mb), P(Mb) + to3(par), P(Mb) + to3(per), 0.24, YELLOW_B, 3)
        self.play(Create(cut2), Create(mark), run_time=0.9)

        cols = [TEAL_D, TEAL_E, TEAL_D, TEAL_E]
        mobs = [mk([P(q) for q in p], col) for p, col in zip(pieces, cols)]
        self.play(FadeOut(sq_b), *[FadeIn(m) for m in mobs], run_time=0.6)
        self.play(FadeOut(cut1), FadeOut(cut2), FadeOut(mark), FadeOut(dot),
                  run_time=0.4)

        # congruent: one piece, turned about the centre, covers each other
        ghost = mk([P(q) for q in pieces[0]], YELLOW_B, 0.45,
                   stroke_color=YELLOW_B, stroke_width=3)
        self.play(FadeIn(ghost), run_time=0.3)
        for _ in range(3):
            self.play(Rotate(ghost, angle=sgn * PI / 2, about_point=P(Mb)),
                      run_time=0.6)
            self.wait(0.2)
        self.play(FadeOut(ghost), run_time=0.3)

        # translate the four pieces into the corners of the c-square
        self.play(FadeOut(lab_c), run_time=0.3)
        self.play(LaggedStart(*[m.animate.shift(to3(s) * F.k)
                                for m, s in zip(mobs, shifts)],
                              lag_ratio=0.35), run_time=3.2)
        self.bring_to_front(lab_a)
        self.play(sq_a.animate.shift(to3(a_shift) * F.k), run_time=1.4)
        self.play(FadeIn(lab_b), run_time=0.5)
        self.play(Write(caption("a² + b²  =  c²", 36)))
        self.hold(2.2)


def _foot(p, u, v):
    """Foot of the perpendicular from p onto the line uv."""
    d = _unit(np.asarray(v, float) - np.asarray(u, float))
    return u + np.dot(p - u, d) * d


class A7_LawOfCosines(Board):
    """Law of cosines for an acute triangle (Euclid II.13 in the windmill
    figure of I.47). The three altitudes, extended, cut each side square
    into two rectangles; the two rectangles at a vertex are equal, because
    shear -> quarter-turn -> shear carries one onto the other. The pairs at
    A and B make up c²; what a² and b² keep over is the pair at C, each
    rectangle a side times the other side's shadow: ab·cos C."""

    def construct(self):
        Aa, Ba = 62 * DEGREES, 54 * DEGREES
        Ca = PI - Aa - Ba
        c = 4.0
        a, b = c * np.sin(Aa) / np.sin(Ca), c * np.sin(Ba) / np.sin(Ca)
        A, B = np.array([0.0, 0.0]), np.array([c, 0.0])
        C = b * np.array([np.cos(Aa), np.sin(Aa)])
        nc = np.array([0.0, -c])                           # c-square, below AB
        nb = b * np.array([-np.sin(Aa), np.cos(Aa)])       # b-square, off CA
        na = np.array([b * np.sin(Aa), c - b * np.cos(Aa)])  # a-square, off BC
        check(close(np.linalg.norm(B - C), a) and close(np.linalg.norm(na), a)
              and abs(np.dot(na, C - B)) < 1e-9 and abs(np.dot(nb, C - A)) < 1e-9
              and np.dot(na, A - B) < 0 and np.dot(nb, B - A) < 0,
              "triangle and outward squares")
        check(max(Aa, Ba, Ca) < PI / 2, "acute triangle")
        Fa, Fb, Fc = _foot(A, B, C), _foot(B, C, A), _foot(C, A, B)
        for f, u, v in ((Fa, B, C), (Fb, C, A), (Fc, A, B)):
            t = np.dot(f - u, v - u) / np.dot(v - u, v - u)
            check(0 < t < 1, "each altitude meets the opposite side inside it")

        rA_b = [A, Fb, Fb + nb, A + nb]        # in b², at A
        rC_b = [Fb, C, C + nb, Fb + nb]        # in b², at C
        rB_a = [B, Fa, Fa + na, B + na]        # in a², at B
        rC_a = [Fa, C, C + na, Fa + na]        # in a², at C
        rA_c = [A, A + nc, Fc + nc, Fc]        # in c², at A
        rB_c = [B, B + nc, Fc + nc, Fc]        # in c², at B
        cosA, cosB, cosC = np.cos(Aa), np.cos(Ba), np.cos(Ca)
        check(close(abs(area(rA_b)), b * c * cosA)
              and close(abs(area(rA_c)), b * c * cosA), "pair at A: bc·cosA")
        check(close(abs(area(rB_a)), a * c * cosB)
              and close(abs(area(rB_c)), a * c * cosB), "pair at B: ac·cosB")
        check(close(abs(area(rC_a)), a * b * cosC)
              and close(abs(area(rC_b)), a * b * cosC), "pair at C: ab·cosC")
        check(close(c * c, a * a + b * b - 2 * a * b * cosC), "law of cosines")

        def turn(pts, pivot, sgn):
            R = np.array([[0.0, -sgn], [sgn, 0.0]])
            return [pivot + R @ (p - pivot) for p in pts]

        # pair A: shear along BFb, -90° about A, shear along CFc
        A1 = [A, B, B + nb, A + nb]
        A2 = turn(A1, A, -1)
        check(all(close(p, q) for p, q in zip(A2, [A, A + nc, C + nc, C])),
              "quarter-turn about A carries the sheared b-part onto c-side")
        # pair B: shear along AFa, +90° about B, shear along CFc
        B1 = [B, A, A + na, B + na]
        B2 = turn(B1, B, +1)
        check(all(close(p, q) for p, q in zip(B2, [B, B + nc, C + nc, C])),
              "quarter-turn about B carries the sheared a-part onto c-side")
        check(abs(np.cross(np.append(B - Fb, 0), np.append(nb, 0))[2]) < 1e-9
              and abs(np.cross(np.append(A - Fa, 0), np.append(na, 0))[2]) < 1e-9
              and abs((Fc - C)[0]) < 1e-9,
              "every shear slides a side along its own line")

        # ---- screen
        lab_pts = [A + nb + 0.0 * nb, C + nb, B + na, C + na, A + nc, B + nc]
        xs = [p[0] for p in lab_pts + [A, B, C]]
        ys = [p[1] for p in lab_pts + [A, B, C]]
        F = Frame(min(xs) - 0.55, max(xs) + 0.55, min(ys) - 0.12,
                  max(ys) + 0.12)
        P = F.P
        COL_A, COL_B, COL_C = TEAL_D, ORANGE, PURPLE_B

        tri = Polygon(P(A), P(B), P(C), stroke_color=WHITE, stroke_width=3)
        c_out = Polygon(*[P(q) for q in [A, B, B + nc, A + nc]],
                        stroke_color=YELLOW_B, stroke_width=4)
        b_out = Polygon(*[P(q) for q in [A, C, C + nb, A + nb]],
                        stroke_color=TEAL_B, stroke_width=1.5)
        a_out = Polygon(*[P(q) for q in [B, C, C + na, B + na]],
                        stroke_color=ORANGE, stroke_width=1.5)
        angC = angle_arc(P(C), P(A), P(B), radius=0.42, color=YELLOW_B)
        labC = tag("C", 24, YELLOW_B).move_to(
            P(C) + 0.74 * to3(_unit(0.62 * _unit(A - C) + 0.38 * _unit(Fc - C))))
        out = 0.42
        lab_b = tag("b²", 30, TEAL_B).move_to(
            P((A + C) / 2 + nb) + out * to3(_unit(nb)))
        lab_a = tag("a²", 30, ORANGE).move_to(
            P((B + C) / 2 + na) + out * to3(_unit(na)))
        lab_c = tag("c²", 30, YELLOW_B).move_to(P(A + nc / 2) + LEFT * 0.45)

        alts = [DashedLine(P(C), P(Fc + nc), color=GREY_B, stroke_width=2,
                           dash_length=0.08),
                DashedLine(P(B), P(Fb + nb), color=GREY_B, stroke_width=2,
                           dash_length=0.08),
                DashedLine(P(A), P(Fa + na), color=GREY_B, stroke_width=2,
                           dash_length=0.08)]
        marks = [_ra(P(Fc), P(C), P(B), 0.13, GREY_B, 2),
                 _ra(P(Fb), P(B), P(C), 0.13, GREY_B, 2),
                 _ra(P(Fa), P(A), P(C), 0.13, GREY_B, 2)]

        self.play(Create(tri), run_time=0.9)
        self.play(Create(angC), FadeIn(labC), run_time=0.5)
        self.play(Create(c_out), Create(b_out), Create(a_out),
                  FadeIn(lab_a), FadeIn(lab_b), FadeIn(lab_c), run_time=1.2)
        self.play(LaggedStart(*[Create(d) for d in alts], lag_ratio=0.3),
                  *[Create(m) for m in marks], run_time=1.5)

        pA = mk([P(q) for q in rA_b], COL_A)
        pB = mk([P(q) for q in rB_a], COL_B)
        pCb = mk([P(q) for q in rC_b], COL_C)
        pCa = mk([P(q) for q in rC_a], COL_C)
        self.play(FadeIn(pA), FadeIn(pB), FadeIn(pCb), FadeIn(pCa), run_time=0.8)
        self.bring_to_front(*alts, *marks, tri, angC, labC)
        self.hold(0.6)

        def shear(mob, target, color, base, slide, rt=1.3):
            edge = Line(P(base[0]), P(base[1]), color=color, stroke_width=8)
            g1 = guide(P(base[0]), P(base[1]), color, extend=0.2)
            g2 = guide(P(slide[0]), P(slide[1]), color, extend=0.08)
            self.play(Create(edge), Create(g1), Create(g2), run_time=0.4)
            self.play(Transform(mob, mk([P(q) for q in target], color)),
                      run_time=rt)
            self.play(FadeOut(edge), FadeOut(g1), FadeOut(g2), run_time=0.3)

        def quarter(mob, pivot, start_vec, sgn, color, rt=1.3):
            st = float(np.arctan2(start_vec[1], start_vec[0]))
            arc = Arc(radius=0.5, start_angle=st, angle=sgn * PI / 2,
                      arc_center=P(pivot), color=color, stroke_width=5)
            mid = st + sgn * PI / 4
            deg = tag("90°", 20, color).move_to(
                P(pivot) + 0.88 * np.array([np.cos(mid), np.sin(mid), 0.0]))
            self.play(Create(arc), FadeIn(deg), run_time=0.4)
            self.play(Rotate(mob, angle=sgn * PI / 2, about_point=P(pivot)),
                      run_time=rt)
            self.play(FadeOut(arc), FadeOut(deg), run_time=0.3)

        # ---- the pair at A: b-part -> c-part
        self.bring_to_front(pA)
        shear(pA, A1, COL_A, base=(A, A + nb), slide=(Fb + nb, B))
        quarter(pA, A, nb, -1, COL_A)
        shear(pA, [A, A + nc, Fc + nc, Fc], COL_A, base=(A, A + nc),
              slide=(C, Fc + nc))
        # ---- the pair at B: a-part -> c-part
        self.bring_to_front(pB)
        shear(pB, B1, COL_B, base=(B, B + na), slide=(Fa + na, A), rt=1.1)
        quarter(pB, B, na, +1, COL_B, rt=1.1)
        shear(pB, [B, B + nc, Fc + nc, Fc], COL_B, base=(B, B + nc),
              slide=(C, Fc + nc), rt=1.1)
        self.bring_to_front(*alts, *marks, tri, angC, labC)
        self.hold(0.4)

        # ---- what a² and b² keep: a side times the other side's shadow.
        # The far end of each side drops along its altitude onto the other.
        ang_b = float(np.arctan2(*(A - C)[::-1]))
        ang_a = float(np.arctan2(*(B - C)[::-1]))
        pr_a = tag("a·cosC", 19).rotate(ang_b + PI).move_to(
            P((C + Fb) / 2) + 0.27 * to3(_unit(nb)))
        pr_b = tag("b·cosC", 19).rotate(ang_a).move_to(
            P((C + Fa) / 2) + 0.27 * to3(_unit(na)))
        for end, ft, lab in ((B, Fb, pr_a), (A, Fa, pr_b)):
            sh = Line(P(C), P(end), color=COL_C, stroke_width=9)
            self.play(Create(sh), run_time=0.45)
            self.play(sh.animate.put_start_and_end_on(P(C), P(ft)),
                      run_time=1.0)
            self.play(FadeIn(lab), run_time=0.4)
        ar_b = tag("ab·cosC", 21).move_to(P((Fb + C) / 2 + 0.6 * nb))
        ar_a = tag("ab·cosC", 21).move_to(P((Fa + C) / 2 + 0.6 * na))
        self.play(FadeIn(ar_a), FadeIn(ar_b), run_time=0.6)
        self.play(Write(caption("c²  =  a² + b² − 2ab·cosC", 34)))
        self.hold(2.2)


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
    R = np.array([[cs, -sn, 0.0], [sn, cs, 0.0], [0.0, 0.0, 1.0]])
    check(all(close(m1 + R @ (p - m0), q, 1e-6) for p, q in zip(src, dst)),
          "a rigid motion lands every vertex")
    start = mob.copy()

    def upd(m, alpha):
        m.become(start.copy().rotate(alpha * turn, about_point=m0)
                 .shift(alpha * (m1 - m0)))
    return UpdateFromAlphaFunc(mob, upd, **kw)


def _carry(mob, p0, p1, q0, q1, **kw):
    """Rigid motion carrying the screen segment p0->p1 onto q0->q1."""
    return _rigid(mob, [p0, p1], [q0, q1], **kw)


class A8_RightInradius(Board):
    """Inradius of a right triangle. At the right angle the two radii and
    the two tangent segments make a square, so both tangents from C are r.
    The tangents from A are equal, and so are those from B (each kite is
    symmetric about its bisector: fold it). Laying the legs end to end over
    the hypotenuse: a + b = c + 2r."""

    def construct(self):
        a, b = 3.0, 4.4
        c = float(np.hypot(a, b))
        r = (a + b - c) / 2
        x, y = b - r, a - r                      # tangents from A, from B
        C, A, B = np.array([0.0, 0.0]), np.array([b, 0.0]), np.array([0.0, a])
        I = np.array([r, r])
        Tb, Ta = np.array([r, 0.0]), np.array([0.0, r])
        Tc = A + x * (B - A) / c
        check(close(np.linalg.norm(I - Tc), r) and abs(np.dot(I - Tc, B - A)) < 1e-9,
              "the incircle touches the hypotenuse at Tc")
        check(close(np.linalg.norm(A - Tb), np.linalg.norm(A - Tc))
              and close(np.linalg.norm(B - Ta), np.linalg.norm(B - Tc))
              and close(x + y, c), "equal tangents; x + y = c")
        refl = lambda p, o, d: o + 2 * np.dot(p - o, d) * d - (p - o)
        check(close(refl(Tb, A, _unit(I - A)), Tc)
              and close(refl(Ta, B, _unit(I - B)), Tc),
              "folding over AI / BI carries Tb, Ta onto Tc")

        # bars: legs end to end (top), hypotenuse below, flush under x and y
        xs = b / 2 - (a + b) / 2
        yt, yb = -1.3, -2.05
        top = [(xs, xs + r), (xs + r, xs + b), (xs + b, xs + b + y),
               (xs + b + y, xs + a + b)]
        bot = [(xs + r, xs + r + x), (xs + r + x, xs + r + c)]
        check(close(top[2][1], bot[1][1]) and close(top[1][0], bot[0][0])
              and close((xs + a + b) - bot[1][1], r),
              "a + b = c + 2r: overhang r at each end")

        F = Frame(xs - 0.35, xs + a + b + 0.35, yb - 0.55, a + 0.2)
        P = F.P
        COL_R, COL_X, COL_Y = GOLD_D, BLUE_D, RED_D

        tri = Polygon(P(C), P(A), P(B), stroke_color=WHITE, stroke_width=3)
        raC = _ra(P(C), P(A), P(B), 0.2)
        la = tag("a", 28).next_to(P((B + C) / 2), LEFT, buff=0.22)
        lb = tag("b", 28).next_to(P((A + C) / 2), DOWN, buff=0.18)
        lc = tag("c", 28).move_to(P((A + B) / 2) + 0.38 * to3(_unit([a, b])))
        self.play(Create(tri), Create(raC), FadeIn(la), FadeIn(lb), FadeIn(lc),
                  run_time=1.2)

        circ = Circle(radius=r * F.k, color=GREY_A, stroke_width=3).move_to(P(I))
        dotI = Dot(P(I), radius=0.06)
        radii = VGroup(*[Line(P(I), P(T), color=GREY_A, stroke_width=2.5)
                         for T in (Ta, Tb, Tc)])
        rmarks = VGroup(_ra(P(Ta), P(I), P(B), 0.15, GREY_A, 2),
                        _ra(P(Tb), P(I), P(A), 0.15, GREY_A, 2),
                        _ra(P(Tc), P(I), P(A), 0.15, GREY_A, 2))
        self.play(Create(circ), FadeIn(dotI), run_time=1.0)
        self.play(Create(radii), Create(rmarks), run_time=0.8)

        # the square at the right angle: both tangents from C equal r
        sq = mk([P(C), P(Tb), P(I), P(Ta)], COL_R, 0.55, stroke_width=0)
        segCb = Line(P(C), P(Tb), color=COL_R, stroke_width=9)
        segCa = Line(P(C), P(Ta), color=COL_R, stroke_width=9)
        lr1 = tag("r", 26, COL_R).next_to(segCb, DOWN, buff=0.16)
        lr2 = tag("r", 26, COL_R).next_to(segCa, LEFT, buff=0.18)
        self.play(FadeIn(sq), Create(segCb), Create(segCa), FadeIn(lr1),
                  FadeIn(lr2), run_time=0.9)
        self.bring_to_front(radii, dotI)

        # equal tangents from A and from B: fold each kite over its bisector
        bisA = DashedLine(P(A), P(I), color=COL_X, stroke_width=2,
                          dash_length=0.08)
        bisB = DashedLine(P(B), P(I), color=COL_Y, stroke_width=2,
                          dash_length=0.08)
        hA = VGroup(mk([P(A), P(Tb), P(I)], COL_X, 0.5, stroke_width=0),
                    Line(P(A), P(Tb), color=COL_X, stroke_width=9))
        hB = VGroup(mk([P(B), P(Ta), P(I)], COL_Y, 0.5, stroke_width=0),
                    Line(P(B), P(Ta), color=COL_Y, stroke_width=9))
        self.play(Create(bisA), Create(bisB), FadeIn(hA), FadeIn(hB),
                  run_time=0.8)
        fA, fB = hA.copy(), hB.copy()
        self.add(fA, fB)
        self.play(Rotate(fA, angle=PI, axis=to3(_unit(I - A)), about_point=P(A)),
                  Rotate(fB, angle=PI, axis=to3(_unit(I - B)), about_point=P(B)),
                  run_time=1.6)
        self.play(FadeOut(hA[0]), FadeOut(hB[0]), FadeOut(fA[0]), FadeOut(fB[0]),
                  FadeOut(bisA), FadeOut(bisB), run_time=0.5)

        # unroll: legs end to end on top, the hypotenuse beneath
        moves =[(segCb, C, Tb, (top[0][0], yt), (top[0][1], yt)),
                 (hA[1], Tb, A, (top[1][0], yt), (top[1][1], yt)),
                 (hB[1], B, Ta, (top[2][0], yt), (top[2][1], yt)),
                 (segCa, Ta, C, (top[3][0], yt), (top[3][1], yt)),
                 (fA[1], Tc, A, (bot[0][0], yb), (bot[0][1], yb)),
                 (fB[1], B, Tc, (bot[1][0], yb), (bot[1][1], yb))]
        anims = []
        for seg, s0, s1, t0, t1 in moves:
            cp = seg.copy()
            self.add(cp)
            anims.append(_carry(cp, P(s0), P(s1), P(t0), P(t1)))
        self.play(LaggedStart(*anims, lag_ratio=0.22), run_time=3.4)

        tick = lambda xv, yv: Line(P((xv, yv)) + 0.13 * UP, P((xv, yv)) + 0.13 * DOWN,
                                   color=WHITE, stroke_width=2)
        ticks = VGroup(*[tick(v, yt) for v in (xs, xs + b, xs + a + b)],
                       *[tick(v, yb) for v in (bot[0][0], bot[1][1])])
        tb = tag("b", 26).next_to(P(((top[0][0] + top[1][1]) / 2, yt)), UP, buff=0.2)
        ta = tag("a", 26).next_to(P(((top[2][0] + top[3][1]) / 2, yt)), UP, buff=0.2)
        tc = tag("c", 26).next_to(P(((bot[0][0] + bot[1][1]) / 2, yb)), DOWN, buff=0.2)
        gl = guide(P((top[0][1], yt)), P((top[0][1], yb)), COL_R)
        gr = guide(P((top[3][0], yt)), P((top[3][0], yb)), COL_R)
        o1 = tag("r", 26, COL_R).move_to(P(((top[0][0] + top[0][1]) / 2, yb)))
        o2 = tag("r", 26, COL_R).move_to(P(((top[3][0] + top[3][1]) / 2, yb)))
        self.play(Create(ticks), FadeIn(tb), FadeIn(ta), FadeIn(tc), run_time=0.7)
        self.play(Create(gl), Create(gr), FadeIn(o1), FadeIn(o2), run_time=0.7)
        self.play(Write(caption("a + b  =  c + 2r    ⟹    r  =  (a + b − c)/2", 32)))
        self.hold(2.2)


# ============================================================ G. TRIGONOMETRY

class G1_PythagoreanIdentity(Board):
    """sin²θ + cos²θ = 1 is Pythagoras on the unit circle's right triangle
    (legs cos θ, sin θ, hypotenuse 1), proved on the spot: four copies of
    that triangle, turned through 0°, 90°, 180°, 270°, fill a square of side
    cos θ + sin θ around a tilted unit square; three of them then slide
    (translation only) into two cos θ × sin θ rectangles, and the holes
    left over are cos²θ and sin²θ."""

    def construct(self):
        th = 33 * DEGREES
        ca, sn = float(np.cos(th)), float(np.sin(th))
        s = ca + sn
        R = 2.7                                     # screen units per unit
        O = np.array([-3.55, 0.3, 0.0])
        Q0 = np.array([1.35, 0.3 - s * R / 2, 0.0])  # big square, lower left

        def Cp(p):
            return O + R * to3(p)

        def Sq(p):
            return Q0 + R * to3(p)

        H, Pt = Cp((ca, 0.0)), Cp((ca, sn))
        T0 = [H, O, Pt]                             # right angle first
        ctr = np.array([s / 2, s / 2])

        def quarter(p, k):
            q = np.asarray(p, float) - ctr
            for _ in range(k):
                q = np.array([-q[1], q[0]])
            return q + ctr

        BR = [(s, 0.0), (sn, 0.0), (s, sn)]          # same placement as T0
        arr1 = [[quarter(p, k) for p in BR] for k in range(4)]  # BR TR TL BL
        inner = [(sn, 0.0), (s, sn), (ca, s), (0.0, ca)]
        big = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
        shifts = [(-sn, ca), (0.0, -sn), (0.0, 0.0), (ca, 0.0)]
        arr2 = [[np.asarray(p) + np.asarray(d) for p in t]
                for t, d in zip(arr1, shifts)]
        sq_c = [(0.0, 0.0), (ca, 0.0), (ca, ca), (0.0, ca)]
        sq_s = [(ca, ca), (s, ca), (s, s), (ca, s)]
        check(close(np.linalg.norm(np.subtract(inner[1], inner[0])), 1.0),
              "the hole is a unit square")
        check(_tiles_exactly(arr1 + [inner], big),
              "4 triangles + unit square tile the (cosθ+sinθ)-square")
        check(_tiles_exactly(arr2 + [sq_c, sq_s], big),
              "4 triangles + cos²θ + sin²θ tile the same square")

        # ---- the unit circle and its right triangle
        circ = Circle(radius=R, color=GREY_B, stroke_width=2.5).move_to(O)
        xax = Line(O + 1.1 * R * LEFT, O + 1.1 * R * RIGHT, color=GREY_D,
                   stroke_width=2)
        yax = Line(O + 1.1 * R * DOWN, O + 1.1 * R * UP, color=GREY_D,
                   stroke_width=2)
        rad = Line(O, Pt, color=YELLOW_B, stroke_width=5)
        arc = angle_arc(O, H, Pt, radius=0.48, color=YELLOW_B, width=4)
        lth = tag("θ", 26, YELLOW_B).move_to(
            O + 0.78 * np.array([np.cos(th / 2), np.sin(th / 2), 0.0]))
        tri = mk(T0, BLUE_E)
        leg_c = Line(O, H, color=ORANGE, stroke_width=8)
        leg_s = Line(H, Pt, color=PURPLE_B, stroke_width=8)
        mark = _ra(H, O, Pt, 0.2)
        l1 = tag("1", 28, YELLOW_B).move_to(
            (O + Pt) / 2 + 0.32 * np.array([-sn, ca, 0.0]))
        lc = tag("cos θ", 24, ORANGE).next_to(leg_c, DOWN, buff=0.16)
        ls = tag("sin θ", 24, PURPLE_A).next_to(leg_s, LEFT, buff=0.14)
        ls.shift(DOWN * 0.08)

        self.play(Create(circ), Create(xax), Create(yax), FadeIn(Dot(O)),
                  run_time=1.0)
        self.play(Create(rad), Create(arc), FadeIn(lth), run_time=0.9)
        self.play(FadeIn(tri), Create(leg_c), Create(leg_s), Create(mark),
                  run_time=0.9)
        self.bring_to_front(rad, arc, lth)
        self.play(FadeIn(l1), FadeIn(lc), FadeIn(ls), run_time=0.6)
        self.hold(0.5)

        # ---- four copies around a unit square
        frame = Polygon(*[Sq(p) for p in big], stroke_color=YELLOW_B,
                        stroke_width=4)
        self.play(Create(frame), run_time=0.6)
        copies, anims = [], []
        for k, t in enumerate(arr1):
            cp = mk(T0, BLUE_E)
            copies.append(cp)
            self.add(cp)
            anims.append(_rigid(cp, T0, [Sq(p) for p in t],
                                turn=[0.0, PI / 2, PI, -PI / 2][k]))
        self.play(LaggedStart(*anims, lag_ratio=0.3), run_time=3.4)
        hole = mk([Sq(p) for p in inner], YELLOW_E, 0.85)
        l_hole = tag("1", 34, BLACK).move_to(Sq(ctr))
        self.play(FadeIn(hole), FadeIn(l_hole), run_time=0.7)
        self.hold(0.9)

        # ---- slide three of them: the holes become cos²θ and sin²θ
        self.play(FadeOut(hole), FadeOut(l_hole), run_time=0.4)
        self.play(*[cp.animate.shift(R * to3(d)) for cp, d in zip(copies, shifts)
                    if d != (0.0, 0.0)], run_time=2.0)
        f_c = mk([Sq(p) for p in sq_c], ORANGE, 0.85)
        f_s = mk([Sq(p) for p in sq_s], PURPLE_B, 0.85)
        l_c = tag("cos²θ", 30, BLACK).move_to(Sq((ca / 2, ca / 2)))
        l_s = tag("sin²θ", 26, BLACK).move_to(Sq((ca + sn / 2, ca + sn / 2)))
        self.play(FadeIn(f_c), FadeIn(f_s), FadeIn(l_c), FadeIn(l_s),
                  run_time=0.8)
        self.play(Write(caption("sin²θ + cos²θ  =  1", 36)))
        self.hold(2.2)


def _free_spot(cands, obstacles):
    """The candidate screen point farthest from every obstacle point."""
    obs = np.array([to3(o) for o in obstacles])
    best = max(cands, key=lambda c: float(np.min(np.linalg.norm(obs - to3(c), axis=1))))
    return to3(best)


def _sample(p, q, n=24):
    """Points along the screen segment pq (obstacles for label placement)."""
    return [to3(p) + (to3(q) - to3(p)) * f for f in np.linspace(0, 1, n)]


def _dir(deg_or_rad):
    """Unit screen vector at an angle given in radians."""
    return np.array([np.cos(deg_or_rad), np.sin(deg_or_rad), 0.0])


class G3_CosineAddition(Board):
    """cos(A+B) read across a rectangle. OP = 1 at angle A+B; the foot Q of
    P on the ray at angle A gives OQ = cos B, QP = sin B. Q's foot X on the
    axis and P's level make a rectangle OXYZ with Q on XY and P on ZY. The
    angle at Q between QY and QP is A (each side turned through 90° from the
    angle at O), so PY = sinA·sinB, while OX = cosA·cosB and ZP = cos(A+B).
    Opposite sides: ZP + PY = OX."""

    def construct(self):
        Aa, Ba = 28 * DEGREES, 36 * DEGREES
        R = 5.5
        O = np.array([-4.6, -2.15, 0.0])

        def S(p):
            return O + R * to3(p)

        Qm = np.cos(Ba) * np.array([np.cos(Aa), np.sin(Aa)])
        Pm = np.array([np.cos(Aa + Ba), np.sin(Aa + Ba)])
        Xm, Ym, Zm = (np.array([Qm[0], 0.0]), np.array([Qm[0], Pm[1]]),
                      np.array([0.0, Pm[1]]))
        check(close(np.linalg.norm(Qm), np.cos(Ba))
              and close(np.linalg.norm(Pm - Qm), np.sin(Ba))
              and abs(np.dot(Pm - Qm, Qm)) < 1e-12, "OQ = cos B, QP = sin B, QP ⊥ OQ")
        check(close(Xm[0], np.cos(Aa) * np.cos(Ba))
              and close(Ym[0] - Pm[0], np.sin(Aa) * np.sin(Ba))
              and close(Pm[0], np.cos(Aa + Ba)), "OX, PY, ZP")
        check(close(np.arccos(_unit(Pm - Qm)[1]), Aa), "angle YQP = A")
        Q, Pt, X, Y, Z = S(Qm), S(Pm), S(Xm), S(Ym), S(Zm)

        xax = Line(O, O + 1.12 * R * RIGHT, color=GREY_D, stroke_width=2)
        yax = Line(O, O + 1.04 * R * UP, color=GREY_D, stroke_width=2)
        arc = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O,
                  color=GREY_B, stroke_width=2.5)
        ray = Line(O, O + 1.1 * R * _dir(Aa), color=GREY_B, stroke_width=2)
        angA = angle_arc(O, X, Q, radius=0.75, color=GREEN_B, width=4)
        labA = tag("A", 24, GREEN_B).move_to(O + 1.05 * _dir(Aa / 2))
        OP = Line(O, Pt, color=YELLOW_B, stroke_width=4)
        angB = angle_arc(O, Q, Pt, radius=1.3, color=PURPLE_A, width=4)
        labB = tag("B", 24, PURPLE_A).move_to(O + 1.62 * _dir(Aa + Ba / 2))
        lab1 = tag("1", 26, YELLOW_B).move_to(
            (O + Pt) / 2 + 0.3 * _dir(Aa + Ba + PI / 2))

        self.play(Create(xax), Create(yax), Create(arc), FadeIn(Dot(O)),
                  run_time=1.0)
        self.play(Create(ray), Create(angA), FadeIn(labA), run_time=0.7)
        self.play(Create(OP), Create(angB), FadeIn(labB), FadeIn(lab1),
                  run_time=0.9)

        # drop P onto the ray: a right triangle with hypotenuse 1, angle B
        OQ = Line(O, Q, color=BLUE_B, stroke_width=6)
        QP = Line(Pt, Q, color=TEAL_B, stroke_width=6)
        mQ = _ra(Q, O, Pt, 0.2)
        lcB = tag("cos B", 22, BLUE_B).move_to(
            (O + Q) / 2 + 0.33 * _dir(Aa - PI / 2))
        lsB = tag("sin B", 22, TEAL_B).move_to(
            (Q + Pt) / 2 + 0.47 * _dir(Aa + PI))
        self.play(Create(QP), Create(mQ), run_time=0.8)
        self.play(Create(OQ), FadeIn(lcB), FadeIn(lsB), run_time=0.7)

        # drop Q onto the axis: OX = cos B · cos A
        dQX = DashedLine(Q, X, color=GREY_B, stroke_width=2, dash_length=0.08)
        mX = _ra(X, O, Q, 0.18, GREY_B, 2)
        OX = Line(O, X, color=BLUE_D, stroke_width=10)
        lOX = tag("cosA·cosB", 22, BLUE_B).next_to(OX, DOWN, buff=0.16)
        self.play(Create(dQX), Create(mX), run_time=0.5)
        self.play(Create(OX), FadeIn(lOX), run_time=0.6)
        self.bring_to_front(OQ, Dot(O))

        # the rectangle through Q and P
        dXY = DashedLine(Q, Y, color=GREY_B, stroke_width=2, dash_length=0.08)
        dZY = DashedLine(Z, Y, color=GREY_B, stroke_width=2, dash_length=0.08)
        dOZ = DashedLine(O, Z, color=GREY_B, stroke_width=2, dash_length=0.08)
        mY = _ra(Y, Q, Pt, 0.18, GREY_B, 2)
        mZ = _ra(Z, O, Y, 0.18, GREY_B, 2)
        self.play(Create(dXY), Create(dZY), Create(dOZ), Create(mY), Create(mZ),
                  run_time=0.8)

        # angle at Q = A: both sides of the angle at O turned through 90°
        wedge = Sector(radius=0.62, angle=Aa, start_angle=0.0, arc_center=O,
                       fill_color=GREEN_B, fill_opacity=0.55, stroke_width=0)
        self.play(FadeIn(wedge), run_time=0.3)
        r = 0.62
        self.play(_rigid(wedge, [O, O + r * _dir(0.0), O + r * _dir(Aa)],
                         [Q, Q + r * _dir(PI / 2), Q + r * _dir(PI / 2 + Aa)],
                         turn=PI / 2), run_time=1.5)
        angQ = angle_arc(Q, Y, Pt, radius=0.66, color=GREEN_B, width=4)
        labQ = tag("A", 24, GREEN_B).move_to(Q + 1.03 * _dir(PI / 2 + Aa / 2))
        self.play(Create(angQ), FadeIn(labQ), wedge.animate.set_fill(opacity=0.3),
                  run_time=0.5)

        # read across the top: ZP = cos(A+B), PY = sinA·sinB
        PY = Line(Pt, Y, color=TEAL_D, stroke_width=10)
        lPY = tag("sinA·sinB", 22, TEAL_B).next_to(PY, UP, buff=0.16)
        ZP = Line(Z, Pt, color=ORANGE, stroke_width=10)
        lZP = tag("cos(A+B)", 22, ORANGE).next_to(ZP, DOWN, buff=0.16)
        self.play(Create(PY), FadeIn(lPY), run_time=0.7)
        self.play(Create(ZP), FadeIn(lZP), run_time=0.7)

        # top side = bottom side, set side by side
        x0 = 1.45
        top_shift = np.array([x0, 1.05, 0.0]) - Z
        bot_shift = np.array([x0, 0.05, 0.0]) - O
        cZP, cPY, cOX = ZP.copy(), PY.copy(), OX.copy()
        self.play(cZP.animate.shift(top_shift), cPY.animate.shift(top_shift),
                  cOX.animate.shift(bot_shift), run_time=1.6)
        xe = x0 + R * Xm[0]
        # the end guides stop short of the labels above the top bar
        g1 = DashedLine([x0, 1.18, 0], [x0, -0.2, 0], color=GREY_B,
                        stroke_width=2, dash_length=0.07)
        g2 = DashedLine([xe, 1.18, 0], [xe, -0.2, 0], color=GREY_B,
                        stroke_width=2, dash_length=0.07)
        b1 = tag("cos(A+B)", 22, ORANGE).next_to(cZP, UP, buff=0.2)
        b2 = tag("sinA·sinB", 22, TEAL_B).next_to(cPY, UP, buff=0.2)
        b3 = tag("cosA·cosB", 22, BLUE_B).next_to(cOX, DOWN, buff=0.2)
        check(min(b1.get_bottom()[1], b2.get_bottom()[1]) > g2.get_top()[1],
              "guides clear of the bar labels")
        self.play(Create(g1), Create(g2), FadeIn(b1), FadeIn(b2), FadeIn(b3),
                  run_time=0.8)
        self.play(Write(caption("cos(A+B)  =  cosA·cosB − sinA·sinB", 34)))
        self.hold(2.2)


class G4_DoubleAngle(Board):
    """sin 2θ = 2 sinθ cosθ: one triangle, two bases. Two radii of the unit
    circle with angle 2θ between them make a triangle T. On base OP₁ = 1 its
    height is sin 2θ; on the chord its base is 2 sinθ and its height cosθ.
    Each way, cut T at half its height and turn the two top pieces through
    180° about the midpoints of the sides: a rectangle of base × ½ height.
    Same T, same area: 1·½ sin 2θ = 2 sinθ·½ cosθ."""

    def construct(self):
        th = 32 * DEGREES
        c, s, h = float(np.cos(th)), float(np.sin(th)), float(np.sin(2 * th))
        R = 4.5
        O = np.array([-5.25, -1.8, 0.0])

        def S(p):
            return O + R * to3(p)

        # ---- T in the unit circle: base OP1, apex P2 at height sin 2θ
        P1, P2 = S((1.0, 0.0)), S((np.cos(2 * th), h))
        L, M = (O + P2) / 2, (P1 + P2) / 2               # side midpoints
        F = S((np.cos(2 * th), h / 2))
        left = [[O, P1, M, L], [P2, L, F], [P2, F, M]]
        left2 = [left[0], [2 * L - p for p in left[1]],
                 [2 * M - p for p in left[2]]]
        rectL = [O, P1, S((1.0, h / 2)), S((0.0, h / 2))]
        check(_tiles_exactly(left, [O, P1, P2]), "three pieces make T")
        check(_tiles_exactly(left2, rectL), "T -> rectangle 1 × ½ sin 2θ")

        # ---- an upright copy T': base = the chord (2 sinθ), height cosθ
        Mr = np.array([2.65, -1.8, 0.0])
        Ap, B1, B2 = Mr + R * c * UP, Mr + R * s * LEFT, Mr + R * s * RIGHT
        U1, U2, G = (Ap + B1) / 2, (Ap + B2) / 2, Mr + R * c / 2 * UP
        right = [[B1, B2, U2, U1], [Ap, U1, G], [Ap, G, U2]]
        right2 = [right[0], [2 * U1 - p for p in right[1]],
                  [2 * U2 - p for p in right[2]]]
        rectR = [B1, B2, B2 + R * c / 2 * UP, B1 + R * c / 2 * UP]
        check(_tiles_exactly(right, [Ap, B1, B2]), "three pieces make T'")
        check(_tiles_exactly(right2, rectR), "T' -> rectangle 2 sinθ × ½ cosθ")
        check(close(abs(area(rectL)), abs(area(rectR)))
              and close(abs(area(rectL)), abs(area([O, P1, P2]))),
              "both rectangles have the area of T")

        xax = Line(O, O + 1.03 * R * RIGHT, color=GREY_D, stroke_width=2)
        yax = Line(O, O + 1.03 * R * UP, color=GREY_D, stroke_width=2)
        arc = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O,
                  color=GREY_B, stroke_width=2.5)
        T = mk([O, P1, P2], BLUE_D)
        a2 = angle_arc(O, P1, P2, radius=0.62, color=YELLOW_B, width=4)
        l2 = tag("2θ", 26, YELLOW_B).move_to(O + 0.98 * _dir(th))
        l1a = tag("1", 26).next_to(Line(O, P1), DOWN, buff=0.16)
        l1b = tag("1", 26).move_to((O + P2) / 2 + 0.3 * _dir(2 * th + PI / 2))
        self.play(Create(xax), Create(yax), Create(arc), FadeIn(Dot(O)),
                  run_time=1.0)
        self.play(FadeIn(T), Create(a2), FadeIn(l2), FadeIn(l1a), FadeIn(l1b),
                  run_time=1.2)
        self.hold(0.4)

        # the copy turns upright and moves right (rigid)
        T2 = mk([O, P1, P2], BLUE_D)
        self.add(T2)
        self.play(_rigid(T2, [O, P1, P2], [Ap, B1, B2], turn=-(PI / 2 + th)),
                  run_time=1.6)

        # left: height sin 2θ on base 1
        top = S((0.0, h))
        proj = DashedLine(P2, top, color=ORANGE, stroke_width=2, dash_length=0.08)
        hseg = Line(O, top, color=ORANGE, stroke_width=8)
        lsin = tag("sin 2θ", 24, ORANGE).next_to(S((0.0, 0.74 * h)), LEFT,
                                                 buff=0.14)
        self.play(Create(proj), Create(hseg), FadeIn(lsin), run_time=0.8)
        # right: base 2 sinθ, height cosθ
        axis = DashedLine(Ap, Mr, color=GREY_B, stroke_width=2, dash_length=0.08)
        mR = _ra(Mr, Ap, B2, 0.2)
        aR = angle_arc(Ap, Mr, B2, radius=0.75, color=YELLOW_B, width=4)
        lth = tag("θ", 24, YELLOW_B).move_to(Ap + 1.05 * angle_mid_dir(Ap, Mr, B2))
        l1c = tag("1", 26).move_to((Ap + B1) / 2 + 0.3 * _dir(PI / 2 + th))
        lcos = tag("cos θ", 24).move_to(Mr + 0.36 * R * c * UP + 0.66 * RIGHT)
        ls1 = tag("sin θ", 24).next_to(Line(B1, Mr), DOWN, buff=0.16)
        ls2 = tag("sin θ", 24).next_to(Line(Mr, B2), DOWN, buff=0.16)
        self.play(Create(axis), Create(mR), Create(aR), FadeIn(lth), FadeIn(l1c),
                  FadeIn(lcos), FadeIn(ls1), FadeIn(ls2), run_time=1.0)
        self.hold(0.4)

        # cut both at half height; the top pieces turn about side midpoints
        midL = DashedLine(S((0.0, h / 2)), M, color=WHITE, stroke_width=2,
                          dash_length=0.08)
        midR = DashedLine(U1, U2, color=WHITE, stroke_width=2, dash_length=0.08)
        pcs = [mk(p, BLUE_D) for p in (left[0], right[0])]
        tops = [mk(p, TEAL_D) for p in (left[1], left[2], right[1], right[2])]
        pivots = [L, M, U1, U2]
        ghosts = [DashedVMobject(Polygon(*q, stroke_color=GREY_B,
                                         stroke_width=1.5), num_dashes=60)
                  for q in ([O, P1, P2], [Ap, B1, B2])]
        self.play(FadeOut(a2), FadeOut(l2), FadeOut(l1b), FadeOut(l1c),
                  FadeOut(aR), FadeOut(lth), FadeOut(mR), FadeOut(axis),
                  FadeOut(lcos), arc.animate.set_stroke(opacity=0.35),
                  run_time=0.5)
        self.add(*ghosts)
        self.play(Create(midL), Create(midR), FadeOut(T), FadeOut(T2),
                  *[FadeIn(m) for m in pcs + tops], run_time=0.8)
        dots = [Dot(p, radius=0.06, color=YELLOW_B) for p in pivots]
        self.play(*[FadeIn(d) for d in dots], run_time=0.3)
        self.play(*[Rotate(m, angle=PI, about_point=p) for m, p in zip(tops, pivots)],
                  run_time=1.8)
        self.play(*[FadeOut(d) for d in dots], FadeOut(midL), FadeOut(midR),
                  run_time=0.4)
        arL = tag("1 · ½ sin 2θ", 26).move_to(S((0.5, h / 4)))
        arR = tag("2 sin θ · ½ cos θ", 26).move_to(Mr + R * c / 4 * UP)
        tick = Line(S((0.0, h / 2)) + 0.12 * LEFT, S((0.0, h / 2)) + 0.12 * RIGHT,
                    color=ORANGE, stroke_width=3)
        self.play(FadeIn(arL), FadeIn(arR), Create(tick), run_time=0.6)
        self.play(Write(caption(
            "½ sin 2θ  =  2 sin θ · ½ cos θ     ⟹     sin 2θ  =  2 sin θ cos θ", 30)))
        self.hold(2.2)


class G5_LawOfSines(Board):
    """Law of sines through the circumcircle. For side a = BC, slide A
    round its arc to B', the far end of the diameter through B: the
    inscribed angle on BC never changes, so it is still A at B'; the angle
    at C is now a right angle (it stands on a diameter), and in that right
    triangle a = 2R·sin A. The same for b and c, with the same 2R."""

    def construct(self):
        ang = {"A": 58.0, "B": 72.0, "C": 50.0}
        pos = {"A": 100.0, "B": 200.0, "C": 316.0}        # degrees on circle
        Rr = 3.1
        O = np.array([-2.3, 0.375, 0.0])

        def on(t):
            return O + Rr * _dir(t * DEGREES)

        V = {k: on(t) for k, t in pos.items()}
        opp = {"A": ("B", "C"), "B": ("C", "A"), "C": ("A", "B")}

        def angle_at(v, p, q):
            u1, u2 = _unit((p - v)[:2]), _unit((q - v)[:2])
            return float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1))))

        for k, (p, q) in opp.items():
            check(abs(angle_at(V[k], V[p], V[q]) - ang[k]) < 1e-9,
                  "inscribed triangle has the chosen angles")
        check(max(ang.values()) < 90, "acute: each antipode lies on its arc")

        col = {"A": ORANGE, "B": TEAL_D, "C": BLUE_D}
        lcol = {"A": ORANGE, "B": TEAL_B, "C": BLUE_B}
        side = {"A": "a", "B": "b", "C": "c"}
        circ = Circle(radius=Rr, color=GREY_B, stroke_width=2.5).move_to(O)
        tri = Polygon(*[V[k] for k in "ABC"], stroke_color=WHITE, stroke_width=3)
        arcs, labs, slabs = VGroup(), VGroup(), VGroup()
        for k, (p, q) in opp.items():
            arcs.add(angle_arc(V[k], V[p], V[q], radius=0.45, color=lcol[k], width=4))
            labs.add(tag(k, 24, lcol[k]).move_to(
                V[k] + 0.78 * angle_mid_dir(V[k], V[p], V[q])))
            mid = (V[p] + V[q]) / 2
            slabs.add(tag(side[k], 26, lcol[k]).move_to(
                mid + 0.3 * to3(_unit((mid - V[k])[:2]))))
        self.play(Create(circ), FadeIn(Dot(O, radius=0.06)), run_time=0.9)
        self.play(Create(tri), run_time=0.8)
        self.play(Create(arcs), FadeIn(labs), FadeIn(slabs), run_time=0.7)

        panel_y = [1.7, 0.75, -0.2]
        done = VGroup()
        diams = []
        for i, (k, rt) in enumerate((("A", 1.9), ("B", 1.3), ("C", 1.3))):
            p, q = opp[k]
            P, Q = V[p], V[q]
            Pp = 2 * O - P                                  # far end of diameter
            t0, t1 = pos[k], (pos[p] + 180.0) % 360.0
            d = (t1 - t0 + 180.0) % 360.0 - 180.0
            for f in np.linspace(0, 1, 41):                 # same angle all the way
                check(abs(angle_at(on(t0 + f * d), P, Q) - ang[k]) < 1e-7,
                      "inscribed angle constant along the arc")
            check(abs(angle_at(Q, P, Pp) - 90.0) < 1e-7, "angle on a diameter is 90°")
            check(close(np.linalg.norm(P - Q), 2 * Rr * np.sin(ang[k] * DEGREES)),
                  "side = 2R sin(opposite angle)")
            tt = ValueTracker(t0)

            def S_(tt=tt):
                return on(tt.get_value())

            hl = Line(P, Q, color=col[k], stroke_width=8)
            l1 = always_redraw(lambda S_=S_, P=P, k=k: Line(
                S_(), P, color=lcol[k], stroke_width=3))
            l2 = always_redraw(lambda S_=S_, Q=Q, k=k: Line(
                S_(), Q, color=lcol[k], stroke_width=3))
            mv_arc = always_redraw(lambda S_=S_, P=P, Q=Q, k=k: angle_arc(
                S_(), P, Q, radius=0.5, color=lcol[k], width=5))
            mv_lab = always_redraw(lambda S_=S_, P=P, Q=Q, k=k: tag(
                k, 24, lcol[k]).move_to(S_() + 0.82 * angle_mid_dir(S_(), P, Q)))
            mv_dot = always_redraw(lambda S_=S_, k=k: Dot(S_(), radius=0.07,
                                                          color=lcol[k]))
            self.play(Create(hl), run_time=0.4)
            self.play(FadeIn(l1), FadeIn(l2), FadeIn(mv_arc), FadeIn(mv_lab),
                      FadeIn(mv_dot), run_time=0.4)
            self.play(tt.animate.set_value(t0 + d), run_time=rt)
            diam = Line(P, Pp, color=YELLOW_B, stroke_width=5)
            nrm = to3(_unit(np.array([-(Pp - O)[1], (Pp - O)[0]])))
            cands = [O + f * (X - O) + sg * 0.34 * nrm for X in (P, Pp)
                     for f in (0.35, 0.5, 0.65) for sg in (1, -1)]
            obst = (_sample(V["A"], V["B"]) + _sample(V["B"], V["C"])
                    + _sample(V["C"], V["A"]) + _sample(Pp, P, 40)
                    + _sample(Pp, Q) + [on(t) for t in range(0, 360, 6)]
                    + [m.get_center() for m in (*labs, *slabs)]
                    + [mv_lab.get_center()] + sum(
                        (_sample(dd.get_start(), dd.get_end(), 40) for dd in diams), []))
            l2R = tag("2R", 26, YELLOW_B).move_to(_free_spot(cands, obst))
            mark = _ra(Q, P, Pp, 0.22, YELLOW_B, 3)
            self.play(Create(diam), FadeIn(l2R), Create(mark), run_time=0.8)
            res = tag(f"{side[k]}  =  2R · sin {k}", 30, lcol[k]).move_to(
                np.array([4.1, panel_y[i], 0.0]))
            self.play(TransformFromCopy(hl, res), run_time=0.7)
            done.add(res)
            for m in (l1, l2, mv_arc, mv_lab, mv_dot):
                m.clear_updaters()
            self.play(FadeOut(VGroup(l1, l2, mv_arc, mv_lab, mv_dot, l2R, mark)),
                      diam.animate.set_stroke(color=lcol[k], width=2.5, opacity=0.8),
                      run_time=0.4)
            diams.append(diam)
        self.play(Write(caption("a/sin A  =  b/sin B  =  c/sin C  =  2R", 34)))
        self.hold(2.2)


class G6_TriangleAreaSine(Board):
    """Area = ½ab·sin C. The height on side a is b·sin C (the right
    triangle with hypotenuse b and angle C). The height splits the triangle
    into two right triangles; each, turned through 180° about the midpoint
    of its hypotenuse, fills a corner of the a × b·sin C rectangle, so the
    rectangle is exactly two triangles."""

    def construct(self):
        a, b, Cang = 7.2, 5.1, 58 * DEGREES
        C = np.array([-3.9, -2.0, 0.0])
        B = C + a * RIGHT
        A = C + b * _dir(Cang)
        H = np.array([A[0], C[1], 0.0])               # foot of the height
        h = A[1] - C[1]
        check(close(h, b * np.sin(Cang)), "height on a is b·sin C")
        check(C[0] < H[0] < B[0], "foot of the height inside CB")
        left, right = [C, H, A], [H, B, A]             # the two right halves
        mL, mR = (C + A) / 2, (A + B) / 2              # hypotenuse midpoints
        left2 = [2 * mL - p for p in left]
        right2 = [2 * mR - p for p in right]
        rect = [C, B, B + h * UP, C + h * UP]
        check(_tiles_exactly([[C, B, A], left2, right2], rect),
              "triangle + its two turned halves tile the a × b·sinC rectangle")
        check(close(abs(area(rect)), 2 * abs(area([C, B, A])))
              and close(abs(area(rect)), a * b * np.sin(Cang)),
              "rectangle = 2 triangles = a·b·sin C")
        # every turning vertex stays on screen (radius b/2 and c/2)
        for piece, m in ((left, mL), (right, mR)):
            for p in piece:
                r = np.linalg.norm(p - m)
                check(m[1] + r < SAFE_TOP and m[1] - r > SAFE_BOTTOM
                      and abs(m[0]) + r < SAFE_X, "the swing stays in frame")

        tri = mk([C, B, A], BLUE_D)
        arcC = angle_arc(C, B, A, radius=0.6, color=YELLOW_B, width=4)
        lC = tag("C", 26, YELLOW_B).move_to(C + 0.9 * angle_mid_dir(C, B, A))
        la = tag("a", 28).next_to(Line(C, B), DOWN, buff=0.18)
        lb = tag("b", 28).move_to((C + A) / 2 + 0.32 * _dir(Cang + PI / 2))
        self.play(FadeIn(tri), Create(arcC), FadeIn(lC), FadeIn(la), FadeIn(lb),
                  run_time=1.3)

        # the height on a: opposite C in the right triangle with hypotenuse b
        alt = DashedLine(A, H, color=YELLOW_B, stroke_width=3, dash_length=0.1)
        mH = _ra(H, A, B, 0.2, YELLOW_B, 2.5)
        hyp = Line(C, A, color=YELLOW_B, stroke_width=6)
        lh = tag("b·sin C", 26, YELLOW_B).next_to(alt, RIGHT, buff=0.14)
        self.play(Create(hyp), run_time=0.5)
        self.play(Create(alt), Create(mH), FadeIn(lh), run_time=0.9)
        self.play(FadeOut(hyp), run_time=0.3)
        self.hold(0.4)

        # the rectangle on a with that height
        box = DashedVMobject(Polygon(*rect, stroke_color=GREY_B, stroke_width=2),
                             num_dashes=80)
        self.play(Create(box), run_time=0.8)

        # each half, turned about the midpoint of its hypotenuse
        cL, cR = mk(left, TEAL_D), mk(right, TEAL_D)
        pv = [Dot(m, radius=0.07, color=WHITE) for m in (mL, mR)]
        self.add(cL, cR, *pv)
        self.bring_to_front(tri, alt, mH, lh, arcC, lC)
        self.play(Rotate(cL, angle=PI, about_point=mL), run_time=1.4)
        self.play(Rotate(cR, angle=PI, about_point=mR), run_time=1.4)
        self.play(FadeOut(VGroup(*pv)), run_time=0.3)
        self.bring_to_front(lb)
        frame = Polygon(*rect, stroke_color=YELLOW_B, stroke_width=4)
        side = Line(C + h * UP, C, color=YELLOW_B, stroke_width=6)
        lh2 = tag("b·sin C", 26, YELLOW_B).next_to(side, LEFT, buff=0.15)
        self.play(Create(frame), Create(side), FadeIn(lh2), run_time=0.8)
        self.play(Write(caption(
            "2 · Area  =  a · b·sin C     ⟹     Area  =  ½ ab·sin C", 32)))
        self.hold(2.2)


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


class G7_HalfAngleTangent(Board):
    """tan(θ/2) = sinθ/(1 + cosθ). On the unit circle, P at angle θ seen
    from V = (−1, 0): the triangle OVP has two radii for sides, so its base
    angles are equal, and moved into the angle at O (one slid along the
    diameter, one turned about the midpoint of OP) they fill θ exactly —
    each is θ/2. The right triangle VMP then has legs sinθ and 1 + cosθ;
    shrunk from V by 1 + cosθ it is VOT, whose leg on the axis is tan(θ/2)."""

    def construct(self):
        th = 70 * DEGREES
        R = 3.1
        O = np.array([-0.3, 0.3, 0.0])

        def S(p):
            return O + R * to3(p)

        V, E = S((-1.0, 0.0)), S((1.0, 0.0))
        P = S((np.cos(th), np.sin(th)))
        M = S((np.cos(th), 0.0))
        T = S((0.0, np.tan(th / 2)))
        k = 1 + np.cos(th)
        check(abs(np.cross(P - V, T - V)[2]) < 1e-9, "T lies on VP")
        check(close(V + k * (O - V), M) and close(V + k * (T - V), P),
              "the homothety from V by 1 + cosθ carries VOT onto VMP")
        check(close(np.tan(th / 2), np.sin(th) / (1 + np.cos(th))), "identity")

        circ = Circle(radius=R, color=GREY_B, stroke_width=2.5).move_to(O)
        diam = Line(V, E, color=GREY_A, stroke_width=2)
        yax = DashedLine(O, S((0.0, 1.0)), color=GREY_D, stroke_width=2,
                         dash_length=0.08)
        rad = Line(O, P, color=YELLOW_B, stroke_width=4)
        aO = angle_arc(O, E, P, radius=0.5, color=YELLOW_B, width=4)
        lO = tag("θ", 26, YELLOW_B).move_to(O + 0.8 * _dir(th / 2) + 0.05 * UP)
        dots = VGroup(*[Dot(x, radius=0.06) for x in (O, V)])
        lV = tag("V", 24, GREY_A).next_to(V, LEFT, buff=0.12)
        self.play(Create(circ), Create(diam), FadeIn(dots), FadeIn(lV),
                  run_time=1.0)
        self.play(Create(rad), Create(aO), FadeIn(lO), run_time=0.8)

        # the isosceles triangle OVP: two radii, two equal base angles
        chord = Line(V, P, color=WHITE, stroke_width=3)
        ovr = Line(V, O, color=YELLOW_B, stroke_width=4)

        def tick(a, b):
            m, d = (a + b) / 2, _unit((b - a)[:2])
            n = np.array([-d[1], d[0], 0.0])
            return Line(m - 0.13 * n, m + 0.13 * n, color=YELLOW_B, stroke_width=3)

        ticks = VGroup(tick(V, O), tick(O, P))
        aV = angle_arc(V, E, P, radius=0.75, color=TEAL_B, width=4)
        aP = angle_arc(P, O, V, radius=0.75, color=TEAL_B, width=4)
        self.play(Create(chord), Create(ovr), Create(ticks), run_time=0.8)
        self.play(Create(aV), Create(aP), run_time=0.6)

        # move both base angles into the angle at O: they fill θ
        wV = _wedge(V, E, P, 0.75, TEAL_D)
        wP = _wedge(P, O, V, 0.75, TEAL_D)
        mOP = (O + P) / 2
        piv = Dot(mOP, radius=0.06, color=WHITE)
        self.play(FadeIn(wV), FadeIn(wP), FadeIn(piv), run_time=0.4)
        self.play(wV.animate.shift(O - V), Rotate(wP, angle=PI, about_point=mOP),
                  run_time=1.8)
        check(close(np.arctan2(*(P - V)[1::-1]), th / 2),
              "each wedge is θ/2 and they meet along the line through O parallel to VP")
        self.hold(0.4)
        lV2 = tag("θ/2", 24, TEAL_B).move_to(V + 1.08 * _dir(th / 4))
        lP2 = tag("θ/2", 22, TEAL_B).move_to(P + 1.08 * angle_mid_dir(P, O, V))
        self.play(FadeOut(wV), FadeOut(wP), FadeOut(piv), FadeIn(lV2), FadeIn(lP2),
                  run_time=0.6)

        # the right triangle VMP: legs sinθ and 1 + cosθ
        pm = Line(P, M, color=ORANGE, stroke_width=6)
        mM = _ra(M, P, E, 0.18)
        l1 = tag("1", 24).next_to(Line(V, O), DOWN, buff=0.14)
        lc = tag("cos θ", 22).next_to(Line(O, M), DOWN, buff=0.14)
        ls = tag("sin θ", 24, ORANGE).next_to(pm, RIGHT, buff=0.14)
        yb = O[1] - 0.82
        bar = Line([V[0], yb, 0], [M[0], yb, 0], color=BLUE_B, stroke_width=7)
        ends = VGroup(*[Line([x, yb - 0.12, 0], [x, yb + 0.12, 0], color=BLUE_B,
                             stroke_width=3) for x in (V[0], M[0])])
        lb = tag("1 + cos θ", 24, BLUE_B).next_to(bar, DOWN, buff=0.12)
        self.play(Create(pm), Create(mM), FadeIn(ls), run_time=0.7)
        self.play(FadeIn(l1), FadeIn(lc), Create(bar), Create(ends), FadeIn(lb),
                  run_time=0.8)
        self.hold(0.4)

        # VOT: the same angle at V over a unit leg; its other leg is tan(θ/2)
        self.play(Create(yax), run_time=0.4)
        vot = mk([V, O, T], GREEN_D, 0.4, stroke_width=0)
        ot = Line(O, T, color=GREEN_B, stroke_width=7)
        lt = tag("tan(θ/2)", 24, GREEN_A).move_to(
            O + 0.85 * UP + 0.12 * LEFT + 0.68 * LEFT)
        mO = _ra(O, T, V, 0.18, GREEN_B, 2.5)
        self.play(FadeIn(vot), Create(ot), Create(mO), FadeIn(lt), run_time=0.8)
        big = Polygon(V, O, T, stroke_color=GREEN_B, stroke_width=4)
        self.add(big)
        self.play(big.animate.scale(k, about_point=V), run_time=1.6)
        self.bring_to_front(pm, ls)
        self.hold(0.3)
        self.play(Write(caption("tan(θ/2)  /  1   =   sin θ  /  (1 + cos θ)", 34)))
        self.hold(2.2)
