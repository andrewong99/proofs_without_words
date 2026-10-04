# SPDX-License-Identifier: GPL-3.0-or-later
from pww_kit import *

# pww_scenes_w1h.py — F45 intercept theorem, F19 angle bisector theorem,
# F37 Morley (Conway's assembly), L3 determinant as area, L1 vector addition,
# F20 altitudes, L6 complex multiplication, D9 means in a trapezoid,
# F27 five-pointed star, K9 diagonals of a polygon.
#
# Every move of a piece is rigid (shift / Rotate / half-turn) unless it is a
# shear or an announced similarity, and every landing, area and angle claim
# is checked numerically with check(...) before anything is drawn, so a
# wrong construction fails the render instead of rendering quietly.


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


def _out(P, U, W, G, k):
    """Point k away from P along the normal of line UW pointing away from G."""
    d = _unit(to3(W) - to3(U))
    n = np.array([-d[1], d[0], 0.0])
    if np.dot(n, to3(P) - to3(G)) < 0:
        n = -n
    return to3(P) + k * n


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


def _chevron(p, q, color=YELLOW_B, size=0.17, width=3, n=1):
    """Parallel-line mark: n small '>' at the middle of segment pq."""
    p, q = to3(p), to3(q)
    d = _unit(q - p)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = (p + q) / 2
    marks = VGroup()
    for k in range(n):
        tip = mid + d * (size * 0.6 * (k - (n - 1) / 2) + size / 2)
        marks.add(VMobject(stroke_color=color, stroke_width=width)
                  .set_points_as_corners([tip - d * size + nrm * size * 0.7,
                                          tip,
                                          tip - d * size - nrm * size * 0.7]))
    return marks


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


def _ra(v, p, q, s=0.2, color=WHITE, width=2.5):
    """Right-angle mark at v between the directions to p and q."""
    v, p, q = to3(v), to3(p), to3(q)
    u1, u2 = _unit(p - v), _unit(q - v)
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([v + s * u1, v + s * u1 + s * u2, v + s * u2])
    return m


def _safe(*mobs, tol=0.02):
    """Fail the render if content leaves the safe area (caption band excluded)."""
    for i, m in enumerate(mobs):
        lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
        check(lo[0] >= -SAFE_X - tol and hi[0] <= SAFE_X + tol
              and lo[1] >= SAFE_BOTTOM - tol and hi[1] <= SAFE_TOP + tol,
              f"inside the safe area (item {i}): [{lo[0]:.2f},{hi[0]:.2f}] x "
              f"[{lo[1]:.2f},{hi[1]:.2f}]")


def _apart(*mobs, gap=0.04):
    """Fail the render if any two mobjects' bounding boxes overlap."""
    for i in range(len(mobs)):
        for j in range(i + 1, len(mobs)):
            a0, a1 = mobs[i].get_critical_point(DL), mobs[i].get_critical_point(UR)
            b0, b1 = mobs[j].get_critical_point(DL), mobs[j].get_critical_point(UR)
            sep = (a1[0] + gap <= b0[0] or b1[0] + gap <= a0[0]
                   or a1[1] + gap <= b0[1] or b1[1] + gap <= a0[1])
            check(sep, f"labels {i} and {j} do not overlap")


def _clear_of_segment(m, p, q, gap=0.06, what="a label"):
    """Fail the render if the segment pq passes through m's bounding box."""
    lo, hi = m.get_critical_point(DL), m.get_critical_point(UR)
    for f in np.linspace(0, 1, 200):
        x = to3(p) + (to3(q) - to3(p)) * f
        if (lo[0] - gap < x[0] < hi[0] + gap
                and lo[1] - gap < x[1] < hi[1] + gap):
            check(False, f"a line runs through {what}")


def _icon(poly, k, color, op=FILL):
    """A small copy of a piece, scaled by k (all icons share k, so their
    sizes compare as the pieces' areas do)."""
    return mk([to3(p) for p in poly], color, op, stroke_width=1.5).scale(k)


# ===================================================== F45 INTERCEPT THEOREM

class F45_InterceptTheorem(Board):
    """Euclid VI.2. DE ∥ BC. Triangles DBE and DCE stand on the same base DE
    between the same parallels, so they have equal areas: slide the apex B
    along BC to C (a shear). ADE and DBE have the same apex E and their
    bases AD, DB on one line, hence one height: their areas are as AD : DB.
    Likewise ADE and DCE (apex D, bases AE, EC): as AE : EC. Equal second
    terms give AD : DB = AE : EC."""

    def construct(self):
        Am, Bm, Cm = np.array([1.6, 5.0]), np.array([0.0, 0.0]), np.array([6.0, 0.0])
        t = 0.41                                    # AD : AB
        F = Frame(-0.45, 6.45, -0.55, 5.55, max_w=7.6,
                  centre=(-2.8, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        D, E = A + t * (B - A), A + t * (C - A)
        H = _foot(E, A, B)                          # height of E over AB
        K = _foot(D, A, C)                          # height of D over AC
        n = np.linalg.norm

        check(abs(_cross(E - D, C - B)) < 1e-9, "DE parallel to BC")
        sY, sB, sO = area([A, D, E]), area([D, B, E]), area([D, C, E])
        check(abs(abs(sB) - abs(sO)) < 1e-9, "DBE and DCE: equal areas")
        check(abs(abs(sY) / abs(sB) - n(D - A) / n(B - D)) < 1e-9,
              "ADE : DBE = AD : DB")
        check(abs(abs(sY) / abs(sO) - n(E - A) / n(C - E)) < 1e-9,
              "ADE : DCE = AE : EC")
        check(0 < np.dot(H - A, B - A) < np.dot(D - A, B - A),
              "the foot of E's height lies on AD")
        check(0 < np.dot(K - A, C - A) < np.dot(E - A, C - A),
              "the foot of D's height lies on AE")

        cY, cB, cO = YELLOW_E, BLUE_D, ORANGE
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        lA = tag("A", 30).move_to(A + 0.38 * UP)
        lB = tag("B", 30).move_to(B + 0.38 * _unit(np.array([-1.0, -0.8, 0])))
        lC = tag("C", 30).move_to(C + 0.38 * _unit(np.array([1.0, -0.8, 0])))
        # D and E sit above the line DE, clear of its dashed extension
        lD = tag("D", 30).move_to(D + 0.42 * _unit(np.array([-0.85, 0.6, 0])))
        lE = tag("E", 30).move_to(E + 0.42 * _unit(np.array([0.85, 0.6, 0])))
        de = Line(D, E, color=WHITE, stroke_width=4)
        par = VGroup(_chevron(D, E), _chevron(B, C))

        pY = mk([A, D, E], cY)
        pB = mk([D, E, B], cB)                       # apex last: it will slide

        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                  run_time=1.0)
        self.play(Create(de), FadeIn(lD), FadeIn(lE), FadeIn(par),
                  run_time=1.0)
        self.add(pY, pB)
        self.bring_to_back(pY, pB)
        self.play(FadeIn(pY), FadeIn(pB), run_time=0.8)
        self.hold(0.3)

        # ----- the panel: icons of the pieces, all at one scale
        k_ic = 0.3
        px = 3.95

        def row(parts, y):
            g = VGroup(*parts).arrange(RIGHT, buff=0.18)
            return g.move_to([px, y, 0.0])

        r1 = row([tag("AD : DB", 32), tag("=", 32),
                  _icon([A, D, E], k_ic, cY), tag(":", 32),
                  _icon([D, E, B], k_ic, cB)], 2.5)
        r2 = row([_icon([D, E, B], k_ic, cB), tag("=", 32),
                  _icon([D, E, C], k_ic, cO)], 1.0)
        r3 = row([tag("AE : EC", 32), tag("=", 32),
                  _icon([A, D, E], k_ic, cY), tag(":", 32),
                  _icon([D, E, C], k_ic, cO)], -0.5)

        # ----- one apex E, bases AD and DB on one line: one height
        bAD = Line(A, D, color=cY, stroke_width=9)
        bDB = Line(D, B, color=BLUE_B, stroke_width=9)
        hE = DashedLine(E, H, color=WHITE, stroke_width=3, dash_length=0.08)
        mH = _ra(H, E, A, 0.17)
        self.play(Create(bAD), Create(bDB), run_time=0.7)
        self.play(Create(hE), Create(mH), run_time=0.7)
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.7)
        self.hold(0.5)
        self.play(FadeOut(hE), FadeOut(mH), FadeOut(bAD), FadeOut(bDB),
                  run_time=0.4)

        # ----- shear: base DE kept, apex slides along BC, its parallel
        gDE = guide(D, E, YELLOW_B, extend=0.3)
        gBC = guide(B, C, YELLOW_B, extend=0.06)
        for lab in (lA, lB, lC, lD, lE):
            _clear_of_segment(lab, gDE.get_start(), gDE.get_end())
            _clear_of_segment(lab, gBC.get_start(), gBC.get_end())
        base = Line(D, E, color=YELLOW_B, stroke_width=8)
        self.play(Create(gDE), Create(gBC), Create(base), run_time=0.7)
        self.play(Transform(pB, mk([D, E, C], cB)), run_time=2.0)
        self.play(pB.animate.set_fill(cO), run_time=0.4)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.6)
        self.play(FadeOut(gDE), FadeOut(gBC), FadeOut(base), run_time=0.4)

        # ----- one apex D, bases AE and EC on one line: one height
        bAE = Line(A, E, color=cY, stroke_width=9)
        bEC = Line(E, C, color=cO, stroke_width=9)
        hD = DashedLine(D, K, color=WHITE, stroke_width=3, dash_length=0.08)
        mK = _ra(K, D, A, 0.17)
        self.play(Create(bAE), Create(bEC), run_time=0.7)
        self.play(Create(hD), Create(mK), run_time=0.7)
        self.play(FadeIn(r3, shift=RIGHT * 0.2), run_time=0.7)
        self.hold(0.5)
        self.play(FadeOut(hD), FadeOut(mK), FadeOut(bAE), FadeOut(bEC),
                  run_time=0.4)

        _safe(tri, lA, lB, lC, lD, lE, r1, r2, r3)
        _apart(lA, lB, lC, lD, lE)
        cap = caption("DE ∥ BC   ⟹   AD : DB  =  AE : EC", 34)
        self.play(Write(cap))
        self.hold(2.2)


# ================================================ F19 ANGLE BISECTOR THEOREM

class F19_AngleBisector(Board):
    """AD bisects the angle at A. Fold the right triangle APD (P the foot of
    D on AB) over AD: the two sides of the angle swap, so it lands on AQD
    (Q the foot on AC) and DP = DQ = h. Triangles ABD and ACD then have the
    same height h over AB and AC: areas as AB : AC. They also have one apex
    A over the bases BD, DC on one line: areas as BD : DC."""

    def construct(self):
        Am, Bm, Cm = np.array([1.8, 4.8]), np.array([0.0, 0.0]), np.array([6.5, 0.0])
        F = Frame(-0.5, 7.0, -0.6, 5.4, max_w=7.7,
                  centre=(-2.75, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C = F.P(Am), F.P(Bm), F.P(Cm)
        n = np.linalg.norm
        cb, cc = n(A - B), n(A - C)
        D = B + (C - B) * cb / (cb + cc)            # the bisector's foot
        P, Q = _foot(D, A, B), _foot(D, A, C)
        H = _foot(A, B, C)
        uad = _unit(D - A)
        Pm = A + 2 * np.dot(P - A, uad) * uad - (P - A)   # P mirrored in AD

        check(abs(_angle(A, B, D) - _angle(A, D, C)) < 1e-9, "AD bisects A")
        check(close(Pm, Q), "the fold over AD carries P to Q")
        check(abs(n(D - P) - n(D - Q)) < 1e-9, "DP = DQ")
        sb, sc = abs(area([A, B, D])), abs(area([A, D, C]))
        h = n(D - P)
        check(abs(sb - cb * h / 2) < 1e-9 and abs(sc - cc * h / 2) < 1e-9,
              "areas are half base times h")
        check(abs(sb / sc - n(D - B) / n(C - D)) < 1e-9, "areas as BD : DC")
        check(0 < np.dot(P - B, A - B) < cb ** 2
              and 0 < np.dot(Q - C, A - C) < cc ** 2, "feet on the sides")
        check(0 < np.dot(H - B, C - B) < n(C - B) ** 2, "altitude foot on BC")

        cB, cC = BLUE_D, ORANGE
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=4)
        lA = tag("A", 30).move_to(A + 0.38 * UP)
        lB = tag("B", 30).move_to(B + 0.38 * _unit(np.array([-1.0, -0.8, 0])))
        lC = tag("C", 30).move_to(C + 0.38 * _unit(np.array([1.0, -0.8, 0])))
        lD = tag("D", 30).move_to(D + 0.4 * DOWN)
        ad = Line(A, D, color=YELLOW_B, stroke_width=4)
        r_arc = 0.62
        arc1 = angle_arc(A, B, D, r_arc, YELLOW_B, 4)
        arc2 = angle_arc(A, D, C, r_arc + 0.08, YELLOW_B, 4)
        la1 = tag("α", 28, YELLOW_B).move_to(A + 1.0 * angle_mid_dir(A, B, D))
        la2 = tag("α", 28, YELLOW_B).move_to(A + 1.0 * angle_mid_dir(A, D, C))

        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC),
                  run_time=1.0)
        self.play(Create(ad), FadeIn(lD), Create(arc1), Create(arc2),
                  FadeIn(la1), FadeIn(la2), run_time=1.2)

        # perpendiculars from D to the two sides
        dp = Line(D, P, color=WHITE, stroke_width=3)
        dq = Line(D, Q, color=WHITE, stroke_width=3)
        mP, mQ = _ra(P, D, A, 0.18), _ra(Q, D, A, 0.18)
        self.play(Create(dp), Create(dq), Create(mP), Create(mQ),
                  run_time=0.9)

        # fold APD over AD: it lands on AQD
        half = mk([A, P, D], TEAL_D, op=0.55, stroke_color=TEAL_B,
                  stroke_width=3)
        self.play(FadeIn(half), run_time=0.4)
        self.play(Rotate(half, angle=PI, axis=uad, about_point=A),
                  run_time=1.8)
        check(all(close(v, w, 1e-6) for v, w in
                  zip(half.get_vertices(), [A, Pm, D])), "fold landed")
        tk = VGroup(_ticks(D, P, 1, YELLOW_B), _ticks(D, Q, 1, YELLOW_B))
        nP = _unit(np.array([-(P - D)[1], (P - D)[0], 0.0]))
        nQ = _unit(np.array([-(Q - D)[1], (Q - D)[0], 0.0]))
        if np.dot(nP, A - D) < 0:
            nP = -nP
        if np.dot(nQ, C - D) < 0:
            nQ = -nQ
        lhP = tag("h", 28).move_to((D + P) / 2 + 0.3 * nP)
        lhQ = tag("h", 28).move_to((D + Q) / 2 + 0.3 * nQ)
        self.play(FadeOut(half), FadeIn(tk), FadeIn(lhP), FadeIn(lhQ),
                  run_time=0.7)

        # the two triangles
        tB = mk([A, B, D], cB)
        tC = mk([A, D, C], cC)
        self.add(tB, tC)
        self.bring_to_back(tB, tC)
        self.play(FadeIn(tB), FadeIn(tC), run_time=0.8)

        k_ic = 0.27
        px = 3.95

        def row(parts, y):
            g = VGroup(*parts).arrange(RIGHT, buff=0.18)
            return g.move_to([px, y, 0.0])

        r1 = row([_icon([A, B, D], k_ic, cB), tag(":", 32),
                  _icon([A, D, C], k_ic, cC), tag("=", 32),
                  tag("AB : AC", 32)], 2.3)
        r2 = row([_icon([A, B, D], k_ic, cB), tag(":", 32),
                  _icon([A, D, C], k_ic, cC), tag("=", 32),
                  tag("BD : DC", 32)], 0.5)

        # one height h over the bases AB and AC
        bAB = Line(A, B, color=BLUE_B, stroke_width=9)
        bAC = Line(A, C, color=cC, stroke_width=9)
        hl = VGroup(Line(D, P, color=YELLOW_B, stroke_width=6),
                    Line(D, Q, color=YELLOW_B, stroke_width=6))
        self.play(Create(bAB), Create(bAC), Create(hl), run_time=0.8)
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.7)
        self.hold(0.6)
        self.play(FadeOut(bAB), FadeOut(bAC), FadeOut(hl), run_time=0.4)

        # one apex A over the bases BD and DC on one line
        bBD = Line(B, D, color=BLUE_B, stroke_width=9)
        bDC = Line(D, C, color=cC, stroke_width=9)
        alt = DashedLine(A, H, color=WHITE, stroke_width=3, dash_length=0.08)
        mH = _ra(H, A, C, 0.18)
        # the altitude crosses the left half-angle and DP: their labels
        # step aside while it is drawn; it must miss every other label
        for lab in (lA, lB, lC, lD, la2, lhQ):
            _clear_of_segment(lab, A, H, gap=0.03)
        self.play(FadeOut(la1), FadeOut(lhP), run_time=0.3)
        self.play(Create(bBD), Create(bDC), Create(alt), Create(mH),
                  run_time=0.9)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.7)
        self.hold(0.6)
        self.play(FadeOut(bBD), FadeOut(bDC), FadeOut(alt), FadeOut(mH),
                  FadeIn(la1), FadeIn(lhP), run_time=0.5)

        _safe(tri, lA, lB, lC, lD, r1, r2)
        _apart(lA, lB, lC, lD, la1, la2, lhP, lhQ)
        for lab in (la1, la2, lhP, lhQ, lD):
            for p, q in ((A, B), (A, C), (B, C), (A, D), (D, P), (D, Q)):
                _clear_of_segment(lab, p, q, gap=0.03)
        self.play(Write(caption("BD : DC  =  AB : AC", 36)))
        self.hold(2.2)


# ================================================== L1 VECTOR ADDITION COMMUTES

def _arrow(p, q, color, width=6):
    return Arrow(to3(p), to3(q), buff=0.0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.12, tip_length=0.28)


class L1_VectorAddition(Board):
    """u then v, and v then u. Each arrow, slid without turning, is the
    opposite side of the parallelogram on u and v: the two orders run along
    its two halves and end at the same corner, the far end of the diagonal."""

    def construct(self):
        um, vm = np.array([4.6, 1.15]), np.array([1.55, 3.35])
        sm = um + vm
        F = Frame(-1.0, 7.1, -0.75, 5.05)
        O, U, V, S = F.P((0, 0)), F.P(um), F.P(vm), F.P(sm)
        check(close(U + (V - O), S) and close(V + (U - O), S),
              "both orders end at the same corner")
        check(_cross(U - O, V - O) > 0, "v turns counter-clockwise from u")

        cu, cv, cs = BLUE_B, ORANGE, YELLOW_B
        au, av = _arrow(O, U, cu), _arrow(O, V, cv)

        def side_lab(s, p, q, col, away):
            m = (to3(p) + to3(q)) / 2
            return tag(s, 32, col).move_to(_out(m, p, q, away, 0.36))

        G = (O + S) / 2
        lu = side_lab("u", O, U, cu, G)
        lv = side_lab("v", O, V, cv, G)
        self.play(FadeIn(Dot(O, radius=0.06)), GrowArrow(au), GrowArrow(av),
                  FadeIn(lu), FadeIn(lv), run_time=1.3)
        self.hold(0.4)

        # u, then v: v slides (no turn) to the tip of u
        av2 = av.copy()
        lv2 = side_lab("v", U, S, cv, G)
        self.play(av2.animate.shift(U - O), run_time=1.3)
        check(close(av2.get_end(), S, 1e-6), "v from the tip of u ends at S")
        half1 = mk([O, U, S], BLUE_D, op=0.45, stroke_width=0)
        d1 = _arrow(O, S, cs)
        nd = _unit(np.array([-(S - O)[1], (S - O)[0], 0.0]))   # upper side
        l_uv = tag("u + v", 30, cs).move_to(G - 0.45 * nd).rotate(
            _dir(O, S), about_point=G - 0.45 * nd)
        self.add(half1)
        self.bring_to_back(half1)
        self.play(FadeIn(half1), FadeIn(lv2), GrowArrow(d1), FadeIn(l_uv),
                  run_time=1.0)
        self.hold(0.4)

        # v, then u: u slides to the tip of v
        au2 = au.copy()
        lu2 = side_lab("u", V, S, cu, G)
        self.play(au2.animate.shift(V - O), run_time=1.3)
        check(close(au2.get_end(), S, 1e-6), "u from the tip of v ends at S")
        half2 = mk([O, S, V], ORANGE, op=0.4, stroke_width=0)
        l_vu = tag("v + u", 30, cs).move_to(G + 0.45 * nd).rotate(
            _dir(O, S), about_point=G + 0.45 * nd)
        self.add(half2)
        self.bring_to_back(half2)
        self.play(FadeIn(half2), FadeIn(lu2), FadeIn(l_vu), run_time=0.9)
        self.bring_to_front(d1)
        corner = Dot(S, radius=0.1, color=cs)
        self.play(GrowFromCenter(corner), run_time=0.5)
        self.play(Indicate(corner, scale_factor=1.8, color=cs), run_time=0.7)
        self.remove(corner)

        # the two orders travel together and arrive together
        path1 = VMobject().set_points_as_corners([O, U, S])
        path2 = VMobject().set_points_as_corners([O, V, S])
        b1 = Dot(O, radius=0.11, color=cu)
        b2 = Dot(O, radius=0.11, color=cv)
        self.add(b1, b2)
        self.play(MoveAlongPath(b1, path1), MoveAlongPath(b2, path2),
                  run_time=2.0, rate_func=linear)
        check(close(b1.get_center(), S, 1e-6) and close(b2.get_center(), S, 1e-6),
              "both travellers arrive at S")
        self.play(FadeOut(b1), FadeOut(b2), run_time=0.4)

        _safe(au, av, au2, av2, lu, lv, lu2, lv2, l_uv, l_vu)
        # the two diagonal labels are turned along the diagonal, 0.45 either
        # side of it, so only their boxes (not their glyphs) meet
        _apart(lu, lv, lu2, lv2, l_uv)
        _apart(lu, lv, lu2, lv2, l_vu)
        for lab in (lu, lv, lu2, lv2):
            for p, q in ((O, U), (O, V), (U, S), (V, S), (O, S)):
                _clear_of_segment(lab, p, q, gap=0.03)
        self.play(Write(caption("u + v  =  v + u", 38)))
        self.hold(2.2)


# ================================================== L3 DETERMINANT AS AREA

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
    """True if the polygons tile `region` with no gap and no overlap, on a
    jittered sample grid over the region's bounding box."""
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


class L3_DeterminantArea(Board):
    """The parallelogram on (a, c) and (b, d) sits in the (a+b) × (c+d)
    rectangle with six corner pieces: two right triangles ½ac, two ½bd and
    two b × c rectangles. The same six pieces regroup in an equal rectangle:
    the triangles pair into an a × c and a b × d block, one b × c fills its
    own block and the other sits in the corner of the a × d block. Both
    times the uncovered part is the rectangle minus the same pieces, so the
    parallelogram has the area of the L: ad − bc."""

    def construct(self):
        a, b, c, d = 4.0, 1.5, 1.2, 3.5
        W, H, g = a + b, c + d, 1.1
        F = Frame(-0.6, 2 * W + g + 0.6, -0.62, H + 0.12)
        P = F.P
        k = F.k
        Dx = np.array([W + g, 0.0])                   # left -> right rectangle

        Om, Pm, Sm, Qm = (0, 0), (a, c), (a + b, c + d), (b, d)
        par = [Om, Pm, Sm, Qm]
        T1 = [(0, 0), (a, 0), (a, c)]
        T2 = [(b, d), (a + b, c + d), (b, c + d)]
        T3 = [(a, c), (a + b, c), (a + b, c + d)]
        T4 = [(0, 0), (b, d), (0, d)]
        R1 = [(a, 0), (a + b, 0), (a + b, c), (a, c)]
        R2 = [(0, d), (b, d), (b, c + d), (0, c + d)]
        rect = [(0, 0), (W, 0), (W, H), (0, H)]
        sh = lambda poly, v: [tuple(np.add(p, v)) for p in poly]
        T2n, T4n = sh(T2, (-b, -d)), sh(T4, (a, c))  # regrouped places
        Lm = [(0, c), (a, c), (a, c + d), (b, c + d), (b, d), (0, d)]

        check(abs(area(par) - (a * d - b * c)) < 1e-12, "parallelogram = ad − bc")
        check(_tiles_exactly([par, T1, T2, T3, T4, R1, R2], rect),
              "parallelogram + six pieces tile the rectangle")
        check(_tiles_exactly([Lm, T1, T2n, T3, T4n, R1, R2], rect),
              "L + the same six pieces tile the same rectangle")
        check(abs(area(Lm) - (a * d - b * c)) < 1e-12, "L = ad − bc")
        check(_tiles_exactly([T1, T2n], [(0, 0), (a, 0), (a, c), (0, c)])
              and _tiles_exactly([T3, T4n], [(a, c), (W, c), (W, H), (a, H)]),
              "the triangles pair into the a×c and b×d blocks")

        cA, cB, cR, cY = BLUE_D, TEAL_D, ORANGE, YELLOW_E
        Lp = lambda poly, col, op=FILL: F.poly(poly, col, op)
        Rp = lambda poly, col, op=FILL: F.poly(sh(poly, Dx), col, op)

        # ---- the two arrows and their parallelogram
        O, Pp, Ss, Qp = P(Om), P(Pm), P(Sm), P(Qm)
        arr1 = _arrow(O, Pp, WHITE, 5)
        arr2 = _arrow(O, Qp, WHITE, 5)
        lab1 = tag("(a, c)", 26).move_to(Pp + 0.78 * angle_mid_dir(Pp, O, Ss))
        lab2 = tag("(b, d)", 26).move_to(Qp + 0.78 * angle_mid_dir(Qp, O, Ss))
        Pi = Lp(par, cY)
        dash = VGroup(DashedLine(Pp, Ss, color=GREY_B, dash_length=0.09),
                      DashedLine(Qp, Ss, color=GREY_B, dash_length=0.09))
        self.play(FadeIn(Dot(O, radius=0.06)), GrowArrow(arr1), GrowArrow(arr2),
                  FadeIn(lab1), FadeIn(lab2), run_time=1.2)
        self.play(Create(dash), run_time=0.6)
        self.add(Pi)
        self.bring_to_back(Pi)
        self.play(FadeIn(Pi), FadeOut(dash), run_time=0.7)

        # ---- the rectangle and its six corner pieces
        box = Polygon(*[P(p) for p in rect], stroke_color=WHITE, stroke_width=4)
        pieces = [Lp(T1, cA), Lp(T2, cA), Lp(T3, cB), Lp(T4, cB),
                  Lp(R1, cR), Lp(R2, cR)]
        dimL = VGroup(tag("a", 28).move_to(P((a / 2, -0.34))),
                      tag("b", 28).move_to(P((a + b / 2, -0.34))),
                      tag("d", 28).move_to(P((-0.32, d / 2))),
                      tag("c", 28).move_to(P((-0.32, d + c / 2))))
        self.play(Create(box), FadeIn(dimL), run_time=0.9)
        self.add(*pieces)
        self.bring_to_back(*pieces)
        self.play(LaggedStart(*[FadeIn(p) for p in pieces], lag_ratio=0.15),
                  run_time=1.3)
        self.bring_to_front(arr1, arr2, lab1, lab2)
        self.hold(0.5)

        # ---- an equal rectangle, cut at a and at c
        boxR = Polygon(*[P(np.add(p, Dx)) for p in rect], stroke_color=WHITE,
                       stroke_width=4)
        grid = VGroup(guide(P(np.add((a, 0), Dx)), P(np.add((a, H), Dx))),
                      guide(P(np.add((0, c), Dx)), P(np.add((W, c), Dx))))
        dimR = VGroup(tag("a", 28).move_to(P(np.add((a / 2, -0.34), Dx))),
                      tag("b", 28).move_to(P(np.add((a + b / 2, -0.34), Dx))),
                      tag("c", 28).move_to(P(np.add((W + 0.32, c / 2), Dx))),
                      tag("d", 28).move_to(P(np.add((W + 0.32, c + d / 2), Dx))))
        self.play(Create(boxR), Create(grid), FadeIn(dimR), run_time=1.0)

        # ---- the same six pieces, regrouped
        stay = VGroup(*[p.copy() for p in (pieces[0], pieces[2], pieces[4],
                                           pieces[5])])
        self.add(stay)
        self.play(stay.animate.shift(to3(Dx) * k), run_time=1.6)
        m2, m4 = pieces[1].copy(), pieces[3].copy()
        self.add(m2)
        self.play(m2.animate.shift(to3(np.add(Dx, (-b, -d))) * k), run_time=1.3)
        self.add(m4)
        self.play(m4.animate.shift(to3(np.add(Dx, (a, c))) * k), run_time=1.3)
        check(close(m2.get_vertices()[0], P(np.add(T2n[0], Dx)), 1e-6)
              and close(m4.get_vertices()[1], P(np.add(T4n[1], Dx)), 1e-6),
              "the moved triangles landed")
        self.play(FadeOut(grid), run_time=0.4)

        # ---- what is left uncovered: the a×d block less a b×c corner
        hole = Rp(Lm, cY)
        blk = DashedVMobject(Polygon(*[P(np.add(p, Dx)) for p in
                                       [(0, c), (a, c), (a, H), (0, H)]],
                                     stroke_color=YELLOW_B, stroke_width=5),
                             num_dashes=40)
        lab_bc = tag("bc", 28).move_to(P(np.add((b / 2, d + c / 2), Dx)))
        lab_L = tag("ad − bc", 32, BLACK).move_to(P(np.add((2.55, 2.45), Dx)))
        self.add(hole)
        self.bring_to_back(hole)
        self.play(FadeIn(hole), run_time=0.7)
        self.play(Create(blk), FadeIn(lab_bc), run_time=0.9)
        self.play(FadeIn(lab_L), FadeOut(blk), run_time=0.7)
        lab_P = tag("ad − bc", 32, BLACK).move_to(P(((a + b) / 2, (c + d) / 2)))
        self.play(FadeIn(lab_P), Indicate(Pi, color=YELLOW_B, scale_factor=1.0),
                  run_time=0.9)

        check(_pip(np.array([lab_L.get_left()[0], lab_L.get_bottom()[1]]),
                   [P(np.add(p, Dx)) for p in Lm])
              and _pip(np.array([lab_L.get_right()[0], lab_L.get_top()[1]]),
                       [P(np.add(p, Dx)) for p in Lm]), "label inside the L")
        _safe(box, boxR, dimL, dimR)
        _apart(lab1, lab2, lab_P)
        self.play(Write(caption("area of the parallelogram on (a, c), (b, d)"
                                "  =  ad − bc", 30)))
        self.hold(2.2)


# ============================================ L6 MULTIPLYING COMPLEX NUMBERS

def _cx(p):
    p = to3(p)
    return complex(p[0], p[1])


def _ctag(s, size=26, color=WHITE, t2c=None):
    """One Text (one font run, one baseline) whose substrings are recoloured
    glyph by glyph afterwards (Text's own t2c splits it into runs)."""
    t = Text(s, font_size=size, color=color)
    flat = s.replace(" ", "")
    check(len(t.submobjects) == len(flat), f"one glyph per character in {s!r}")
    for sub, col in (t2c or {}).items():
        i = flat.find(sub)
        while i != -1:
            for gl in t.submobjects[i:i + len(sub)]:
                gl.set_color(col)
            i = flat.find(sub, i + len(sub))
    return t


class L6_ComplexMultiplication(Board):
    """z = x + iy is reached by x steps along 1 and y steps along i. Times w,
    1 becomes w and i becomes iw — w turned a right angle, same length — so
    zw = x·w + y·iw is reached by the same steps in the turned, scaled frame.
    Hence the triangle 0, 1, z turned by arg w and scaled by |w| lands on
    0, w, zw: lengths multiply, angles add."""

    def construct(self):
        x, y = 1.35, 0.72
        zc, wc = complex(x, y), complex(0.95, 0.78)
        zwc = zc * wc
        th, ph, lw = np.angle(zc), np.angle(wc), abs(wc)
        F = Frame(-1.12, 1.82, -0.36, 2.02, max_w=8.0,
                  centre=(-2.45, (SAFE_TOP + SAFE_BOTTOM) / 2))
        P = lambda q: F.P((q.real, q.imag)) if isinstance(q, complex) else F.P(q)
        O, E1, Ei = P(0j), P(1 + 0j), P(1j)
        Z, Wp, iW, ZW = P(zc), P(wc), P(1j * wc), P(zwc)
        X0, XW = P(complex(x, 0)), P(x * wc)

        check(abs(x * wc + y * (1j * wc) - zwc) < 1e-12, "zw = x·w + y·iw")
        check(abs(abs(zwc) - abs(zc) * lw) < 1e-12, "|zw| = |z||w|")
        check(abs(np.angle(zwc) - (th + ph)) < 1e-12, "arg zw = arg z + arg w")
        check(0 < th < ph < th + ph < PI / 2, "angles in order, first quadrant")

        def turn_scale(q, a=1.0):
            """Screen point q turned by a·arg w and scaled by |w|^a about 0."""
            v = (_cx(q) - _cx(O)) * np.exp(1j * a * ph) * lw ** a
            return to3(O) + np.array([v.real, v.imag, 0.0])

        check(close(turn_scale(E1), Wp) and close(turn_scale(Z), ZW)
              and close(turn_scale(X0), XW) and close(turn_scale(Ei), iW),
              "turn-and-scale sends 1, z, x, i to w, zw, xw, iw")

        axes = VGroup(Line(P(complex(-1.08, 0)), P(complex(1.8, 0)),
                           color=GREY_B, stroke_width=2),
                      Line(P(complex(0, -0.32)), P(complex(0, 2.0)),
                           color=GREY_B, stroke_width=2))
        dots0 = VGroup(*[Dot(q, radius=0.06, color=WHITE) for q in (O, E1, Ei)])
        l0 = tag("0", 26).move_to(O + np.array([-0.25, -0.25, 0]))
        l1 = tag("1", 26).move_to(E1 + 0.3 * DOWN)
        li = tag("i", 28).move_to(Ei + 0.28 * LEFT)
        self.play(Create(axes), FadeIn(dots0), FadeIn(l0), FadeIn(l1),
                  FadeIn(li), run_time=1.0)

        # z: x steps along 1 (blue), then y steps along i (teal); the panel
        # colours x and y the same way
        cX, cY = BLUE_B, TEAL_B
        px = 4.3
        rows = [_ctag("z = x + iy", 30, t2c={"x": cX, "y": cY}),
                tag("w·1 = w,   w·i = iw", 30),
                _ctag("zw = x·w + y·iw", 30, t2c={"x": cX, "y": cY})]
        for r, yy in zip(rows, (2.75, 1.75, 0.75)):
            r.move_to([px, yy, 0])
        legx = Line(O, X0, color=cX, stroke_width=7)
        legy = Line(X0, Z, color=cY, stroke_width=7)
        dz = Dot(Z, radius=0.07, color=WHITE)
        lz = tag("z", 30).move_to(Z + np.array([0.3, 0.12, 0]))
        tri1 = mk([O, E1, Z], BLUE_D, op=0.55)
        self.play(Create(legx), run_time=0.8)
        self.play(Create(legy), FadeIn(dz), FadeIn(lz), run_time=0.8)
        self.add(tri1)
        self.bring_to_back(tri1)
        self.play(FadeIn(tri1), FadeIn(rows[0]), run_time=0.7)
        self.hold(0.3)

        # w, and iw = w turned a right angle
        cW = ORANGE
        sw = Line(O, Wp, color=cW, stroke_width=5)
        dw = Dot(Wp, radius=0.07, color=cW)
        lwl = tag("w", 30, cW).move_to(Wp + np.array([0.33, -0.13, 0]))
        self.play(Create(sw), FadeIn(dw), FadeIn(lwl), run_time=0.8)
        siw = sw.copy()
        self.add(siw)
        self.play(Rotate(siw, angle=PI / 2, about_point=O), run_time=1.2)
        check(close(siw.get_end(), iW, 1e-6), "w turned a quarter is iw")
        diw = Dot(iW, radius=0.07, color=cW)
        liw = tag("iw", 30, cW).move_to(iW + np.array([-0.38, 0.1, 0]))
        rmk = _ra(O, Wp, iW, 0.2, cW, 2.5)
        self.play(FadeIn(diw), FadeIn(liw), Create(rmk), FadeIn(rows[1]),
                  run_time=0.8)
        self.hold(0.3)

        # zw: x steps along w, then y steps along iw
        legxw = Line(O, XW, color=cX, stroke_width=7)
        legyw = Line(XW, ZW, color=cY, stroke_width=7)
        dzw = Dot(ZW, radius=0.07, color=WHITE)
        lzw = tag("zw", 30).move_to(ZW + np.array([0.0, 0.33, 0]))
        self.play(Create(legxw), run_time=0.9)
        self.play(Create(legyw), FadeIn(dzw), FadeIn(lzw), run_time=0.9)
        self.play(FadeIn(rows[2]), run_time=0.6)
        self.hold(0.4)

        # the whole picture of z, turned by arg w and scaled by |w|
        ghost = VGroup(mk([O, E1, Z], BLUE_D, op=0.55),
                       Line(O, X0, color=cX, stroke_width=7),
                       Line(X0, Z, color=cY, stroke_width=7))
        start = ghost.copy()

        def upd(m, a):
            m.become(start.copy().rotate(a * ph, about_point=O)
                     .scale(lw ** a, about_point=O))

        self.add(ghost)
        self.play(UpdateFromAlphaFunc(ghost, upd), run_time=2.6)
        check(close(ghost[0].get_vertices()[1], Wp, 1e-6)
              and close(ghost[0].get_vertices()[2], ZW, 1e-6),
              "the turned, scaled triangle lands on 0, w, zw")
        tri2 = mk([O, Wp, ZW], cW, op=0.55)
        self.add(tri2)
        self.play(FadeOut(ghost), FadeIn(tri2),
                  FadeOut(VGroup(legx, legy, legxw, legyw,
                                 siw, rmk, diw, liw)), run_time=0.9)
        self.bring_to_front(sw, dw, dz, dzw)

        # angles add, lengths multiply
        r1, r2 = 0.27 * F.k, 0.5 * F.k
        a_th = angle_arc(O, E1, Z, r1, BLUE_B, 4)
        a_ph = angle_arc(O, E1, Wp, r2, GREEN_B, 4)
        a_th2 = angle_arc(O, Wp, ZW, r2, BLUE_B, 4)
        l_th = tag("θ", 26, BLUE_B).move_to(O + (r1 + 0.22) * angle_mid_dir(O, E1, Z))
        l_ph = tag("φ", 26, GREEN_B).move_to(O + (r2 + 0.24) * angle_mid_dir(O, E1, Wp))
        l_th2 = tag("θ", 26, BLUE_B).move_to(O + (r2 + 0.22) * angle_mid_dir(O, Wp, ZW))
        self.play(Create(a_th), FadeIn(l_th), run_time=0.6)
        self.play(Create(a_ph), FadeIn(l_ph), run_time=0.6)
        self.play(Create(a_th2), FadeIn(l_th2), run_time=0.6)

        def along(p, q, f, off, side_pt):
            m = to3(p) + f * (to3(q) - to3(p))
            return _out(m, p, q, side_pt, off)

        L_z = tag("|z|", 26).move_to(along(O, Z, 0.72, 0.33, ZW))
        L_w = tag("|w|", 26).move_to(along(O, Wp, 0.7, 0.33, Z))
        L_zw = tag("|z|·|w|", 26).move_to(along(O, ZW, 0.62, 0.62, Z))
        _clear_of_segment(L_zw, O, ZW, gap=0.05, what="|z|·|w|")
        _clear_of_segment(L_w, O, Wp, gap=0.03, what="|w|")
        _clear_of_segment(L_z, O, Z, gap=0.03, what="|z|")
        self.play(FadeIn(L_z), FadeIn(L_w), FadeIn(L_zw), run_time=0.8)

        row4 = VGroup(_ctag("θ = arg z", 30, t2c={"θ": BLUE_B}),
                      _ctag("φ = arg w", 30, t2c={"φ": GREEN_B}))
        row4.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([px, -0.75, 0])
        self.play(FadeIn(row4), run_time=0.6)

        _safe(axes, *rows, row4, lzw, liw)
        _apart(l0, l1, li, lz, lwl, lzw, l_th, l_ph, l_th2, L_z, L_w, L_zw)
        self.play(Write(caption("|zw| = |z|·|w|,    arg(zw) = arg z + arg w",
                                32)))
        self.hold(2.2)


# ================================================== F20 ALTITUDES ARE CONCURRENT

class F20_Altitudes(Board):
    """Turn ABC half a turn about the midpoint of each side: the three copies
    and ABC tile the doubled triangle A'B'C', whose sides pass through A, B,
    C parallel to the opposite sides (B'C' ∥ BC, ...), and AB' = BC = C'A,
    so A is the midpoint of B'C' (likewise B, C). The altitude from A is
    perpendicular to BC, hence to B'C', at its midpoint: it is the
    perpendicular bisector of B'C'. The three perpendicular bisectors of
    A'B'C' meet at its circumcentre, the point H with HA' = HB' = HC'."""

    def construct(self):
        aB, aC = 62 * DEGREES, 52 * DEGREES          # angle A = 66°
        ab = 4.6 * np.sin(aC) / np.sin(PI - aB - aC)
        Am = ab * np.array([np.cos(aB), np.sin(aB)])
        Bm, Cm = np.array([0.0, 0.0]), np.array([4.6, 0.0])
        Apm, Bpm, Cpm = Bm + Cm - Am, Cm + Am - Bm, Am + Bm - Cm
        pts = np.array([Apm, Bpm, Cpm])
        lo, hi = pts.min(axis=0) - 0.72, pts.max(axis=0) + 0.72
        F = Frame(lo[0], hi[0], lo[1], hi[1])
        A, B, C, Ap, Bp, Cp = [F.P(p) for p in (Am, Bm, Cm, Apm, Bpm, Cpm)]
        D, E, Fo = _foot(A, B, C), _foot(B, C, A), _foot(C, A, B)
        H = _meet(A, D, B, E)
        n = np.linalg.norm
        Ma, Mb, Mc = (B + C) / 2, (C + A) / 2, (A + B) / 2

        check(max(_angle(A, B, C), _angle(B, C, A), _angle(C, A, B)) < PI / 2,
              "acute triangle (H inside)")
        check(close(_rot2(A, Ma, PI), Ap) and close(_rot2(B, Mb, PI), Bp)
              and close(_rot2(C, Mc, PI), Cp), "half-turns give A', B', C'")
        check(abs(_cross(Cp - Bp, C - B)) < 1e-9 and abs(_cross(Ap - Cp, A - C)) < 1e-9
              and abs(_cross(Bp - Ap, B - A)) < 1e-9, "sides parallel")
        check(close((Bp + Cp) / 2, A) and close((Cp + Ap) / 2, B)
              and close((Ap + Bp) / 2, C), "A, B, C are the midpoints")
        check(abs(np.dot(Bp - Cp, A - D)) < 1e-9 and abs(np.dot(Cp - Ap, B - E)) < 1e-9
              and abs(np.dot(Ap - Bp, C - Fo)) < 1e-9,
              "each altitude is perpendicular to the outer side")
        check(abs(_cross(H - C, Fo - C)) < 1e-9, "the third altitude passes H")
        check(abs(n(H - Ap) - n(H - Bp)) < 1e-9 and abs(n(H - Bp) - n(H - Cp)) < 1e-9,
              "H is equidistant from A', B', C'")

        G = (A + B + C) / 3
        col = BLUE_D
        tri = mk([A, B, C], col, op=0.6, stroke_width=3)
        lA = tag("A", 30).move_to(_out(A, Bp, Cp, G, 0.36))
        lB = tag("B", 30).move_to(_out(B, Cp, Ap, G, 0.36))
        lC = tag("C", 30).move_to(_out(C, Ap, Bp, G, 0.36))
        self.play(FadeIn(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=1.0)

        # three half-turns about the side midpoints
        copies = []
        for M in (Ma, Mb, Mc):
            cp = mk([A, B, C], col, op=0.32, stroke_width=3)
            dm = Dot(M, radius=0.07, color=YELLOW_B)
            self.add(cp)
            self.play(FadeIn(dm), run_time=0.25)
            self.play(Rotate(cp, angle=PI, about_point=M), run_time=1.1)
            self.play(FadeOut(dm), run_time=0.2)
            copies.append(cp)
        self.bring_to_front(tri)
        lAp = tag("A'", 30).move_to(Ap + 0.38 * _unit(Ap - G))
        lBp = tag("B'", 30).move_to(Bp + 0.38 * _unit(Bp - G))
        lCp = tag("C'", 30).move_to(Cp + 0.38 * _unit(Cp - G))
        big = Polygon(Ap, Bp, Cp, stroke_color=WHITE, stroke_width=4)
        # parallel marks: inner side and the outer half-side beside it (the
        # outer sides' own midpoints are A, B, C)
        par = VGroup(_chevron(B, C, n=1), _chevron(A, Bp, n=1),
                     _chevron(C, A, n=2), _chevron(B, Cp, n=2),
                     _chevron(A, B, n=3), _chevron(C, Ap, n=3))
        self.play(Create(big), FadeIn(lAp), FadeIn(lBp), FadeIn(lCp),
                  FadeIn(par), run_time=1.0)
        self.hold(0.3)

        # A, B, C are the midpoints of the outer sides
        ticks = VGroup(_ticks(A, Bp, 1), _ticks(A, Cp, 1), _ticks(B, C, 1),
                       _ticks(B, Cp, 2), _ticks(B, Ap, 2), _ticks(C, A, 2),
                       _ticks(C, Ap, 3), _ticks(C, Bp, 3), _ticks(A, B, 3))
        self.play(FadeOut(par), FadeIn(ticks), run_time=0.9)
        self.hold(0.5)

        # each altitude: perpendicular to a side and to its parallel, through
        # the midpoint of the parallel
        alts = []
        for V, Ft, U, Wn, Q in ((A, D, C, Bp, B), (B, E, A, Cp, C),
                                (C, Fo, B, Ap, A)):
            ln = Line(V, Ft, color=YELLOW_B, stroke_width=5)
            m1 = _ra(Ft, V, U, 0.17, YELLOW_B, 2.5)
            m2 = _ra(V, Ft, Wn, 0.17, YELLOW_B, 2.5)
            self.play(Create(ln), Create(m1), Create(m2), run_time=0.9)
            alts.append(VGroup(ln, m1, m2))
        hdot = Dot(H, radius=0.08, color=YELLOW_B)
        # H's label goes in the widest gap between the nine rays from H
        rays = sorted(_dir(H, Q) % TAU for Q in (A, B, C, D, E, Fo, Ap, Bp, Cp))
        gaps = [((rays[(i + 1) % 9] - rays[i]) % TAU, rays[i]) for i in range(9)]
        gw, g0 = max(gaps)
        lH = tag("H", 28, YELLOW_B).move_to(
            H + 0.44 * np.array([np.cos(g0 + gw / 2), np.sin(g0 + gw / 2), 0.0]))
        self.play(FadeIn(hdot), FadeIn(lH), run_time=0.5)

        # H is the circumcentre of A'B'C': equally far from its corners
        rad = VGroup(*[DashedLine(H, Q, color=YELLOW_B, stroke_width=2.5,
                                  dash_length=0.1) for Q in (Ap, Bp, Cp)])
        self.play(FadeOut(ticks), Create(rad), run_time=1.0)
        rt = VGroup(*[_ticks(H, Q, 2, YELLOW_B, at=0.62) for Q in (Ap, Bp, Cp)])
        self.play(FadeIn(rt), run_time=0.5)

        labs = (lA, lB, lC, lAp, lBp, lCp, lH)
        _safe(big, *labs)
        _apart(*labs)
        for lab, nm in zip(labs, ("A", "B", "C", "A'", "B'", "C'", "H")):
            for p, q in ((H, Ap), (H, Bp), (H, Cp), (A, D), (B, E), (C, Fo)):
                _clear_of_segment(lab, p, q, gap=0.02, what=nm)
        self.play(Write(caption("the three altitudes of ABC meet in one point H",
                                32)))
        self.hold(2.2)


# ================================================== F27 THE FIVE-POINTED STAR

class F27_StarAngles(Board):
    """Tips A, B, C, D, E; the line BE cuts the point at A off as a triangle
    AXY (X on AC, Y on AD). The angle of AXY at X is an exterior angle of
    triangle XCE, so it holds the tips at C and E: C slides along CA to X
    (corresponding angles), E turns half a turn about the midpoint of EX
    (alternate angles). Likewise the angle at Y holds the tips at D and B.
    The three angles of AXY then line up at A along the parallel to XY: a
    straight angle."""

    def construct(self):
        pts = {"A": (0.4, 2.1), "B": (-2.2, 0.9), "C": (-1.0, -1.8),
               "D": (1.7, -1.3), "E": (2.2, 0.4)}
        F = Frame(-2.62, 2.62, -2.22, 2.5, max_w=8.0,
                  centre=(-2.45, (SAFE_TOP + SAFE_BOTTOM) / 2))
        A, B, C, D, E = [F.P(pts[k]) for k in "ABCDE"]
        X, Y = _meet(B, E, A, C), _meet(B, E, A, D)
        tips = {"A": (A, C, D), "B": (B, D, E), "C": (C, E, A),
                "D": (D, A, B), "E": (E, B, C)}
        al = {k: _angle(*v) for k, v in tips.items()}
        check(abs(sum(al.values()) - PI) < 1e-9, "the tips add up to 180°")
        tX = np.dot(X - B, E - B) / np.dot(E - B, E - B)
        tY = np.dot(Y - B, E - B) / np.dot(E - B, E - B)
        check(0 < tX < tY < 1, "B, X, Y, E in this order on BE")
        check(abs(_angle(X, A, Y) - (al["C"] + al["E"])) < 1e-9,
              "angle at X = C + E (exterior angle of XCE)")
        check(abs(_angle(Y, A, X) - (al["B"] + al["D"])) < 1e-9,
              "angle at Y = B + D (exterior angle of YDB)")

        r = 0.7
        check(2 * r < np.linalg.norm(Y - X) and r < np.linalg.norm(A - X)
              and r < np.linalg.norm(A - Y), "wedges fit at X and Y")
        cols = {"A": BLUE_D, "B": TEAL_D, "C": ORANGE, "D": GREEN_D, "E": RED_D}
        lcol = {"A": BLUE_B, "B": TEAL_B, "C": ORANGE, "D": GREEN_B, "E": RED_B}
        names = {"A": "α₁", "B": "α₂", "C": "α₃", "D": "α₄", "E": "α₅"}
        star = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners(
            [A, C, E, B, D, A])
        wed = {k: _wedge(v[0], v[1], v[2], r, cols[k], op=0.9)
               for k, v in tips.items()}
        labs = {k: tag(names[k], 30, lcol[k]).move_to(
            v[0] - 0.42 * angle_mid_dir(*v)) for k, v in tips.items()}
        self.play(Create(star), run_time=1.5)
        self.play(*[FadeIn(w) for w in wed.values()],
                  *[FadeIn(l) for l in labs.values()], run_time=1.0)
        self.bring_to_front(star)
        self.hold(0.4)

        # the point at A, cut off by BE
        axy = Polygon(A, X, Y, stroke_color=YELLOW_B, stroke_width=5)
        dX, dY = Dot(X, radius=0.06, color=YELLOW_B), Dot(Y, radius=0.06,
                                                           color=YELLOW_B)
        self.play(Create(axy), FadeIn(dX), FadeIn(dY), run_time=0.8)

        def half_turn(mob, M, run_time=1.5):
            md = Dot(M, radius=0.07, color=WHITE)
            self.play(FadeIn(md), run_time=0.25)
            self.play(Rotate(mob, angle=PI, about_point=M), run_time=run_time)
            self.play(FadeOut(md), run_time=0.2)

        # where each tip's wedge ends up: a translation keeps the directions
        # of its two sides, a half-turn reverses both
        def at_A(d1, d2, flips):
            sgn = (-1) ** flips
            return sorted([_dir(ORIGIN, sgn * d1) % TAU, _dir(ORIGIN, sgn * d2) % TAU])

        ivs = [at_A(C - A, D - A, 0),          # A stays
               at_A(A - C, E - C, 1),          # C: slide, half-turn
               at_A(B - E, C - E, 2),          # E: half-turn, half-turn
               at_A(A - D, B - D, 1),          # D: slide, half-turn
               at_A(D - B, E - B, 2)]          # B: half-turn, half-turn
        lo0 = _dir(ORIGIN, X - Y) % TAU        # the parallel, towards B
        ivs = sorted([((a - lo0) % TAU, (b - lo0) % TAU) if (b - a) < PI
                      else ((b - lo0) % TAU, (a - lo0 + TAU) % TAU)
                      for a, b in ivs])
        ivs = [(a, b if b > a else b + TAU) for a, b in ivs]
        check(abs(ivs[0][0]) < 1e-9 and abs(ivs[-1][1] - PI) < 1e-9
              and all(abs(ivs[i][1] - ivs[i + 1][0]) < 1e-9 for i in range(4)),
              "the five wedges tile the half-turn at A, edge to edge")

        # at X: C slides along CA, E turns about the midpoint of EX
        xce = Polygon(X, C, E, stroke_color=GREY_B, stroke_width=2)
        self.play(Create(xce), run_time=0.5)
        wc = wed["C"].copy()
        self.add(wc)
        self.play(wc.animate.shift(X - C), run_time=1.3)
        we = wed["E"].copy()
        self.add(we)
        half_turn(we, (E + X) / 2)
        check(abs(_angle(X, A, Y) - (al["C"] + al["E"])) < 1e-9, "X filled")
        self.play(FadeOut(xce), run_time=0.3)

        # at Y: D slides along DA, B turns about the midpoint of BY
        ydb = Polygon(Y, D, B, stroke_color=GREY_B, stroke_width=2)
        self.play(Create(ydb), run_time=0.5)
        wd = wed["D"].copy()
        self.add(wd)
        self.play(wd.animate.shift(Y - D), run_time=1.3)
        wb = wed["B"].copy()
        self.add(wb)
        half_turn(wb, (B + Y) / 2)
        self.play(FadeOut(ydb), run_time=0.3)
        self.bring_to_front(axy, dX, dY)
        self.hold(0.3)

        # the three angles of AXY line up at A along the parallel to XY
        u = _unit(Y - X)
        par = DashedLine(A - 1.9 * u, A + 1.9 * u, color=YELLOW_B,
                         stroke_width=3, dash_length=0.1)
        self.play(Create(par), run_time=0.6)
        gx, gy = VGroup(wc, we), VGroup(wd, wb)
        half_turn(gx, (A + X) / 2, 1.4)
        half_turn(gy, (A + Y) / 2, 1.4)
        # the five wedges at A now fill the half-disc below the parallel
        tot = (_angle(A, A - u, C) + al["A"] + _angle(A, D, A + u))
        check(abs(tot - PI) < 1e-9, "the five angles make a straight angle")
        check(abs(_angle(A, A - u, C) - (al["C"] + al["E"])) < 1e-9
              and abs(_angle(A, D, A + u) - (al["B"] + al["D"])) < 1e-9,
              "the half-turned groups fill the two sides")
        th0 = _dir(A, A + u)
        semi = Arc(radius=r + 0.06, start_angle=th0 + PI, angle=PI,
                   arc_center=A, color=YELLOW_B, stroke_width=5)
        self.play(Create(semi), run_time=0.6)

        # the same five angles, magnified and turned level beside the star
        mag, Mc = 2.55, np.array([4.3, -0.15, 0.0])
        five = VGroup(wed["A"].copy(), wc.copy(), we.copy(), wd.copy(),
                      wb.copy(), semi.copy())
        start5 = five.copy()

        def magnify(m, al):
            """Every frame an exact similarity: turn, enlarge, carry."""
            m.become(start5.copy().rotate(-al * th0, about_point=A)
                     .scale(mag ** al, about_point=A).shift(al * (Mc - A)))

        self.add(five)
        self.play(UpdateFromAlphaFunc(five, magnify), run_time=1.6)
        check(close(five[0].get_arc_center(), Mc, 1e-6), "magnified copy centred")
        R = (r + 0.06) * mag
        dia = Line(Mc + R * LEFT, Mc + R * RIGHT, color=YELLOW_B, stroke_width=5)

        def big_dir(d1, d2, flips):
            """Bisector direction of a wedge at A, in the turned copy."""
            sg = (-1) ** flips
            v = _unit(sg * d1) + _unit(sg * d2)
            return _rot2(_unit(v), ORIGIN, -th0)

        mids = {"A": big_dir(C - A, D - A, 0), "C": big_dir(A - C, E - C, 1),
                "E": big_dir(B - E, C - E, 2), "D": big_dir(A - D, B - D, 1),
                "B": big_dir(D - B, E - B, 2)}
        blabs = VGroup(*[tag(names[k], 30, WHITE).move_to(Mc + 0.68 * R * mids[k])
                         for k in "ACEDB"])
        l180 = tag("180°", 32, YELLOW_B).move_to(Mc + 0.42 * UP)
        self.play(Create(dia), FadeIn(blabs), FadeIn(l180), run_time=0.9)

        _safe(star, *labs.values(), l180, par, five, blabs)
        _apart(*labs.values(), l180)
        _apart(*blabs, l180)
        for lab in labs.values():
            for p, q in ((A, C), (C, E), (E, B), (B, D), (D, A),
                         (A - 1.9 * u, A + 1.9 * u)):
                _clear_of_segment(lab, p, q, gap=0.03, what="a tip label")
        self.play(Write(caption("α₁ + α₂ + α₃ + α₄ + α₅  =  180°", 36)))
        self.hold(2.2)


# ================================================== K9 DIAGONALS OF A POLYGON

class K9_PolygonDiagonals(Board):
    """Each corner sends a diagonal to every corner except itself and its
    two neighbours: n − 3 of them. Draw only the half of each diagonal that
    belongs to the corner it leaves from: every diagonal is then made of two
    halves of different colours, one from each end, so the n(n − 3) halves
    pair up into n(n − 3)/2 diagonals."""

    def construct(self):
        n = 7
        Rr, Oc = 2.85, np.array([-2.7, 0.3, 0.0])
        V = [Oc + Rr * np.array([np.cos(PI / 2 + TAU * k / n),
                                 np.sin(PI / 2 + TAU * k / n), 0.0])
             for k in range(n)]
        diags = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if (j - i) % n not in (1, n - 1)]
        check(len(diags) == n * (n - 3) // 2, "D = n(n−3)/2 for the heptagon")
        halves = {i: [j for j in range(n) if (j - i) % n not in (0, 1, n - 1)]
                  for i in range(n)}
        check(all(len(h) == n - 3 for h in halves.values()), "n − 3 per corner")
        check(sum(len(h) for h in halves.values()) == 2 * len(diags),
              "every diagonal counted from both ends")

        cols = [BLUE_B, TEAL_B, ORANGE, YELLOW_E, GREEN_B, RED_B, PURPLE_B]
        poly = Polygon(*V, stroke_color=WHITE, stroke_width=4)
        dots = VGroup(*[Dot(v, radius=0.09, color=cols[k])
                        for k, v in enumerate(V)])
        self.play(Create(poly), FadeIn(dots), run_time=1.0)

        def half(i, j):
            return Line(V[i], (V[i] + V[j]) / 2, color=cols[i], stroke_width=6)

        # the panel counts with icons: a half (one colour, from its corner to
        # the midpoint) and a diagonal (two halves of two colours)
        def half_icon(c, L=0.62):
            return VGroup(Line(ORIGIN, L * RIGHT, color=c, stroke_width=6),
                          Dot(ORIGIN, radius=0.08, color=c),
                          Dot(L * RIGHT, radius=0.05, color=WHITE))

        def diag_icon(c1, c2, L=0.62):
            return VGroup(Line(ORIGIN, L * RIGHT, color=c1, stroke_width=6),
                          Line(L * RIGHT, 2 * L * RIGHT, color=c2, stroke_width=6),
                          Dot(ORIGIN, radius=0.08, color=c1),
                          Dot(2 * L * RIGHT, radius=0.08, color=c2),
                          Dot(L * RIGHT, radius=0.05, color=WHITE))

        px = 3.95
        rows = [tag("n = 7", 32),
                VGroup(tag("n · (n − 3)  ×", 32), half_icon(cols[0])),
                VGroup(diag_icon(cols[0], cols[4]), tag("=  2  ×", 32),
                       half_icon(cols[0])),
                tag("7 · 4  =  28  =  2 · 14", 30, GREY_A)]
        for r in rows[1:3]:
            r.arrange(RIGHT, buff=0.22)
        for r, y in zip(rows, (2.75, 1.6, 0.45, -0.7)):
            r.move_to([px, y, 0])

        # corner 0: its two neighbours are joined by sides, not diagonals
        sides = VGroup(Line(V[0], V[1], color=GREY_B, stroke_width=9),
                       Line(V[0], V[-1], color=GREY_B, stroke_width=9))
        self.play(FadeIn(sides), run_time=0.5)
        self.play(FadeOut(sides), run_time=0.4)
        h0 = VGroup(*[half(0, j) for j in halves[0]])
        lab0 = tag("n − 3", 28, cols[0]).move_to(V[0] + 0.42 * UP)
        self.play(LaggedStart(*[Create(h) for h in h0], lag_ratio=0.2),
                  FadeIn(lab0), FadeIn(rows[0]), run_time=1.3)
        self.hold(0.4)

        # every corner does the same
        hs = VGroup(*[half(i, j) for i in range(1, n) for j in halves[i]])
        self.play(LaggedStart(*[Create(h) for h in hs], lag_ratio=0.06),
                  run_time=3.0)
        self.bring_to_front(dots)
        self.play(FadeIn(rows[1]), run_time=0.6)

        # each diagonal: two halves, one from each end
        mids = VGroup(*[Dot((V[i] + V[j]) / 2, radius=0.06, color=WHITE)
                        for i, j in diags])
        self.play(FadeIn(mids), run_time=0.6)
        i, j = diags[2]
        check((i, j) == (0, 4), "the panel's diagonal icon has this pair's colours")
        pair = VGroup(Line(V[i], (V[i] + V[j]) / 2, color=cols[i], stroke_width=12),
                      Line(V[j], (V[i] + V[j]) / 2, color=cols[j], stroke_width=12))
        self.play(FadeIn(pair), run_time=0.5)
        self.play(FadeIn(rows[2]), run_time=0.6)
        self.play(FadeOut(pair), run_time=0.4)
        self.play(FadeIn(rows[3]), run_time=0.6)

        _safe(poly, *rows, lab0)
        _apart(*rows)
        self.play(Write(caption("D  =  n(n − 3) / 2", 38)))
        self.hold(2.2)


# ================================================== D9 MEANS IN A TRAPEZOID

class D9_TrapezoidMeans(Board):
    """Parallel sides a (top) < b (bottom). Four cuts parallel to them, from
    the top: through the crossing of the diagonals (it is halved there; the
    harmonic mean); the cut that splits the trapezoid into two similar ones
    (the upper one, enlarged from the legs' meeting point, covers the lower
    one exactly, so a : g = g : b; the geometric mean); the midline, through
    the midpoints of the legs (the arithmetic mean); and the cut into two
    equal areas (the root mean square). A cut is longer the lower it lies,
    so the order on the page is the order of the means."""

    def construct(self):
        a, b, h, p = 2.0, 6.0, 4.6, 1.5
        F = Frame(-1.05, 7.25, -0.6, 5.12)
        P = F.P

        def xl(y):
            return p * y / h

        def xr(y):
            return b + (p + a - b) * y / h

        def yof(L):
            return (b - L) * h / (b - a)

        BLm, BRm, TRm, TLm = (0, 0), (b, 0), (p + a, h), (p, h)
        BL, BR, TR, TL = P(BLm), P(BRm), P(TRm), P(TLm)
        means = {"H": 2 * a * b / (a + b), "G": np.sqrt(a * b),
                 "A": (a + b) / 2, "R": np.sqrt((a * a + b * b) / 2)}
        ys = {k: yof(v) for k, v in means.items()}
        check(ys["H"] > ys["G"] > ys["A"] > ys["R"],
              "from the top: HM, GM, AM, RMS")
        for k, v in means.items():
            check(abs((xr(ys[k]) - xl(ys[k])) - v) < 1e-12, f"length of {k}")
        Om = np.array(_meet(BL, TR, TL, BR))
        Ohm = np.array([(Om[0] - F.centre[0]) / F.k + F.cx, (Om[1] - F.centre[1]) / F.k + F.cy])
        check(abs(Ohm[1] - ys["H"]) < 1e-9, "HM cut passes the diagonals' crossing")
        check(abs((Ohm[0] - xl(ys["H"])) - (xr(ys["H"]) - Ohm[0])) < 1e-9,
              "the crossing halves the HM cut")
        Pm = np.array([xl(h * b / (b - a)), h * b / (b - a)])   # legs meet
        check(abs(xr(Pm[1]) - Pm[0]) < 1e-12, "apex of the legs")
        g = means["G"]
        homo = lambda q, s: Pm + s * (np.array(q, float) - Pm)
        yg = ys["G"]
        check(close(homo(TLm, g / a), (xl(yg), yg)) and close(homo((xl(yg), yg), g / a), BLm)
              and close(homo((xr(yg), yg), g / a), BRm), "upper part enlarged by g/a = lower part")
        yr = ys["R"]
        up = (a + means["R"]) / 2 * (h - yr)
        dn = (means["R"] + b) / 2 * yr
        check(abs(up - dn) < 1e-9, "RMS cut halves the area")
        check(abs(xl(ys["A"]) - xl(h) / 2) < 1e-12, "midline through the leg midpoints")

        trap = Polygon(BL, BR, TR, TL, stroke_color=WHITE, stroke_width=4)
        la = tag("a", 30).move_to(P((p + a / 2, h + 0.32)))
        lb = tag("b", 30).move_to(P((b / 2, -0.34)))
        self.play(Create(trap), FadeIn(la), FadeIn(lb), run_time=1.0)

        cH, cG, cA, cR = BLUE_B, GREEN_B, YELLOW_B, RED_B

        def cut(k, col):
            y = ys[k]
            return Line(P((xl(y), y)), P((xr(y), y)), color=col, stroke_width=7)

        def lab(k, s, col):
            y = ys[k]
            t = tag(s, 26, col)
            return t.next_to(P((xr(y), y)), RIGHT, buff=0.3)

        # 1. through the crossing of the diagonals
        dg = VGroup(Line(BL, TR, color=GREY_B, stroke_width=2.5),
                    Line(TL, BR, color=GREY_B, stroke_width=2.5))
        O = to3(Om)
        self.play(Create(dg), run_time=0.8)
        self.play(FadeIn(Dot(O, radius=0.07, color=cH)), run_time=0.3)
        sH = cut("H", cH)
        LH, RH = P((xl(ys["H"]), ys["H"])), P((xr(ys["H"]), ys["H"]))
        tH = VGroup(_ticks(LH, O, 1, cH), _ticks(O, RH, 1, cH))
        lH = lab("H", "2ab/(a+b)", cH)
        self.play(Create(sH), FadeIn(tH), FadeIn(lH), run_time=0.9)
        self.hold(0.4)
        self.play(dg.animate.set_stroke(opacity=0.25), FadeOut(tH), run_time=0.4)

        # 2. into two similar trapezoids: the upper one, enlarged from the
        # legs' meeting point, covers the lower one exactly
        sG = cut("G", cG)
        lg = lab("G", "g", cG)
        self.play(Create(sG), FadeIn(lg), run_time=0.6)
        upper = F.poly([TLm, TRm, (xr(yg), yg), (xl(yg), yg)], GREEN_D, op=0.5)
        self.play(FadeIn(upper), run_time=0.4)
        cpy = upper.copy()
        start = cpy.copy()
        Pa = P(Pm)

        def grow(m, al):
            m.become(start.copy().scale((g / a) ** al, about_point=Pa))

        self.add(cpy)
        self.play(UpdateFromAlphaFunc(cpy, grow), run_time=1.9)
        check(close(cpy.get_vertices()[0], P(homo(TLm, g / a)), 1e-6),
              "the enlarged copy landed")
        rel = _ctag("a : g  =  g : b", 30, t2c={"g": cG}).move_to([5.05, 3.05, 0])
        _safe(rel)
        self.play(FadeIn(rel), run_time=0.6)
        self.hold(0.5)
        lG = lab("G", "√(ab)", cG)
        self.play(ReplacementTransform(lg, lG), run_time=0.6)
        self.play(FadeOut(upper), FadeOut(cpy), FadeOut(rel), run_time=0.5)

        # 3. through the midpoints of the legs
        yA = ys["A"]
        # equal-length marks on the two halves of each leg, placed clear of
        # the ends of the cuts already drawn
        tk = VGroup(_ticks(BL, P((xl(yA), yA)), 1), _ticks(P((xl(yA), yA)), TL, 1, at=0.75),
                    _ticks(BR, P((xr(yA), yA)), 2), _ticks(P((xr(yA), yA)), TR, 2, at=0.75))
        for yy in (yA * 0.5, yA + 0.75 * (h - yA)):
            check(min(abs(yy - ys[k]) for k in "HG") > 0.3, "ticks clear of the cuts")
        self.play(FadeIn(tk), run_time=0.6)
        sA = cut("A", cA)
        lA = lab("A", "(a+b)/2", cA)
        self.play(Create(sA), FadeIn(lA), run_time=0.7)
        self.hold(0.3)
        self.play(FadeOut(tk), run_time=0.3)

        # 4. into two equal areas
        sR = cut("R", cR)
        self.play(Create(sR), run_time=0.6)
        fu = F.poly([TLm, TRm, (xr(yr), yr), (xl(yr), yr)], PURPLE_B, op=0.38)
        fd = F.poly([(xl(yr), yr), (xr(yr), yr), BRm, BLm], MAROON_B, op=0.38)

        def bracket(y0, y1, col):
            """A bracket left of the trapezoid spanning heights y0..y1 (the
            whole region it names, across the cuts inside it)."""
            xb, tk = -0.3, 0.14
            return VGroup(Line(P((xb, y0)), P((xb, y1)), color=col, stroke_width=4),
                          Line(P((xb, y0)), P((xb + tk, y0)), color=col, stroke_width=4),
                          Line(P((xb, y1)), P((xb + tk, y1)), color=col, stroke_width=4))

        bu, bd = bracket(yr + 0.05, h, PURPLE_A), bracket(0.0, yr - 0.05, MAROON_A)
        Su = tag("S", 30, PURPLE_A).next_to(bu, LEFT, buff=0.14)
        Sd = tag("S", 30, MAROON_A).next_to(bd, LEFT, buff=0.14)
        self.add(fu, fd)
        self.bring_to_back(fu, fd)
        self.play(FadeIn(fu), FadeIn(fd), Create(bu), Create(bd), FadeIn(Su),
                  FadeIn(Sd), run_time=0.8)
        lR = lab("R", "√((a²+b²)/2)", cR)
        self.play(FadeIn(lR), run_time=0.5)
        self.bring_to_front(sH, sG, sA, sR)

        labs = (la, lb, lH, lG, lA, lR, Su, Sd)
        _safe(trap, bu, bd, *labs)
        _apart(*labs)
        check(bu.get_right()[0] < P((0.0, 0.0))[0] - 0.05, "brackets clear of the leg")
        self.play(Write(caption("2ab/(a+b)  ≤  √(ab)  ≤  (a+b)/2  ≤  √((a²+b²)/2)",
                                32)))
        self.hold(2.2)


# ================================================== F37 MORLEY (CONWAY'S ASSEMBLY)

def _plus(base, n, size=22, color=WHITE):
    """'x⁺' / 'x⁺⁺' built from two Text runs: the base and n small raised
    plus signs (no reliance on superscript glyphs in the target fonts)."""
    t = Text(base, font_size=size, color=color)
    if n == 0:
        return VGroup(t)
    s = Text("+" * n, font_size=size * 0.62, color=color)
    s.next_to(t, RIGHT, buff=0.02).align_to(t, UP).shift(UP * 0.05 * size / 22)
    return VGroup(t, s)


def _apex(U, V, angU, angV, away):
    """Apex of the triangle on base UV with angles angU at U, angV at V, on
    the side of the line UV away from the point `away`."""
    U, V, away = to3(U), to3(V), to3(away)
    side = -np.sign(_cross(V - U, away - U))
    LU = np.linalg.norm(V - U) * np.sin(angV) / np.sin(PI - angU - angV)
    return U + LU * _rot2(_unit(V - U), ORIGIN, side * angU)


def _conway(V, X, angV, angX, angW, away):
    """Conway's triangle with angles (angV, angX, angW), laid with its vertex
    V on V and its side VX on the given edge VX (whose length a neighbour
    fixed); returns its third vertex W, on the side away from `away`."""
    V, X, away = to3(V), to3(X), to3(away)
    LW = np.linalg.norm(X - V) * np.sin(angX) / np.sin(angW)
    side = -np.sign(_cross(X - V, away - V))
    return V + LW * _rot2(_unit(X - V), ORIGIN, side * angV)


class F37_MorleyConway(Board):
    """Built backwards (Conway). Fix a, b, c with a + b + c = 60° and write
    x⁺ = x + 60°, x⁺⁺ = x + 120°. Take an equilateral triangle PQR; on its
    sides put triangles with angles (a, b⁺, c⁺), (b, c⁺, a⁺), (c, a⁺, b⁺);
    then three triangles with angles (a, b, c⁺⁺), (b, c, a⁺⁺), (c, a, b⁺⁺),
    each scaled to the edge it meets first. They fit: around P, Q and R the
    angles make 60° + x⁺ + y⁺ + z⁺⁺ = 360°, and each corner piece's second
    edge comes out exactly the length of the edge it meets. The result is a
    triangle with angles 3a, 3b, 3c whose corners are split into three equal
    angles by the pieces' edges: its trisectors meet at P, Q, R. Any
    triangle is similar to one of these."""

    def construct(self):
        a, b, c = 25 * DEGREES, 19 * DEGREES, 16 * DEGREES
        p6, p12 = PI / 3, 2 * PI / 3
        check(abs(a + b + c - PI / 3) < 1e-12 and len({a, b, c}) == 3,
              "a + b + c = 60°, not all equal")

        # ---- the construction, in units of PQR's side (centre at 0)
        rr = 1 / np.sqrt(3)
        Pm, Qm, Rm = [rr * np.array([np.cos(t), np.sin(t), 0.0])
                      for t in (-PI / 2, PI / 6, 5 * PI / 6)]
        Am = _apex(Rm, Qm, b + p6, c + p6, Pm)      # (a, b+, c+) on RQ
        Bm = _apex(Pm, Rm, c + p6, a + p6, Qm)      # (b, c+, a+) on PR
        Cm = _apex(Qm, Pm, a + p6, b + p6, Rm)      # (c, a+, b+) on QP
        # the three corner pieces, each built on its own from its angles and
        # the one edge it shares with a piece already placed
        B2 = _conway(Rm, Am, c + p12, a, b, Qm)     # (a, b, c++) on RA
        C2 = _conway(Pm, Bm, a + p12, b, c, Rm)     # (b, c, a++) on PB
        A2 = _conway(Qm, Cm, b + p12, c, a, Pm)     # (c, a, b++) on QC
        check(close(B2, Bm, 1e-12) and close(C2, Cm, 1e-12) and close(A2, Am, 1e-12),
              "each corner piece's second edge meets its neighbour exactly")
        n = np.linalg.norm
        # the second edge of each corner piece has the length of the edge of
        # the side piece it meets (RB, PC, QA)
        check(abs(n(Bm - Rm) - n(B2 - Rm)) < 1e-12 and abs(n(Cm - Pm) - n(C2 - Pm)) < 1e-12
              and abs(n(Am - Qm) - n(A2 - Qm)) < 1e-12, "shared edges equal")
        for V, ring in ((Pm, (Qm, Rm, Bm, Cm)), (Qm, (Rm, Pm, Cm, Am)),
                        (Rm, (Pm, Qm, Am, Bm))):
            s = sum(_angle(V, ring[i], ring[(i + 1) % 4]) for i in range(4))
            check(abs(s - TAU) < 1e-12, "a full turn round P, Q, R")
        for V, U, W, x in ((Am, Bm, Cm, a), (Bm, Cm, Am, b), (Cm, Am, Bm, c)):
            check(abs(_angle(V, U, W) - 3 * x) < 1e-12, "corner angles 3a, 3b, 3c")
        for V, seq, x in ((Am, (Bm, Rm, Qm, Cm), a), (Bm, (Cm, Pm, Rm, Am), b),
                          (Cm, (Am, Qm, Pm, Bm), c)):
            check(all(abs(_angle(V, seq[i], seq[i + 1]) - x) < 1e-12
                      for i in range(3)), "corners cut in three equal angles")

        def trisector(V, U, W):
            t = _angle(V, U, W)
            sg = np.sign(_cross(U - V, W - V))
            return _rot2(_unit(U - V), ORIGIN, sg * t / 3)

        def meet_rays(p1, d1, p2, d2):
            m = np.array([d1[:2], -d2[:2]]).T
            t, _ = np.linalg.solve(m, (p2 - p1)[:2])
            return p1 + t * d1

        check(close(meet_rays(Bm, trisector(Bm, Cm, Am), Cm, trisector(Cm, Bm, Am)), Pm, 1e-12)
              and close(meet_rays(Cm, trisector(Cm, Am, Bm), Am, trisector(Am, Cm, Bm)), Qm, 1e-12)
              and close(meet_rays(Am, trisector(Am, Bm, Cm), Bm, trisector(Bm, Am, Cm)), Rm, 1e-12),
              "direct check: the trisectors of ABC meet at P, Q, R")

        # ---- to the screen, BC level
        tilt = -_dir(Bm, Cm)
        pts = [_rot2(v, ORIGIN, tilt) for v in (Pm, Qm, Rm, Am, Bm, Cm)]
        xs, ys_ = [v[0] for v in pts], [v[1] for v in pts]
        F = Frame(min(xs) - 0.5, max(xs) + 0.5, min(ys_) - 0.44, max(ys_) + 0.42)
        P, Q, R, A, B, C = [F.P(v) for v in pts]
        G = (A + B + C) / 3

        cE, cT, cU = YELLOW_E, BLUE_D, TEAL_D

        def lab_at(V, U, W, mob, d=None):
            """Put mob inside the angle U-V-W, far enough out to fit."""
            half = _angle(V, U, W) / 2
            if d is None:
                d = max(0.34, 0.17 / np.sin(half) + 0.06)
            return mob.move_to(V + d * angle_mid_dir(V, U, W))

        def piece(vs, col, labels):
            poly = mk(vs, col, op=0.8)
            labs = VGroup(*[lab_at(vs[i], vs[(i + 1) % 3], vs[(i + 2) % 3], m)
                            for i, m in enumerate(labels)])
            return VGroup(poly, labs)

        E = piece([P, Q, R], cE, [tag("60°", 17), tag("60°", 17), tag("60°", 17)])
        ticksE = VGroup(_ticks(P, Q), _ticks(Q, R), _ticks(R, P))
        TA = piece([A, R, Q], cT, [_plus("a", 0), _plus("b", 1), _plus("c", 1)])
        TB = piece([B, P, R], cT, [_plus("b", 0), _plus("c", 1), _plus("a", 1)])
        TC = piece([C, Q, P], cT, [_plus("c", 0), _plus("a", 1), _plus("b", 1)])
        UA = piece([P, B, C], cU, [_plus("a", 2), _plus("b", 0), _plus("c", 0)])
        UB = piece([Q, C, A], cU, [_plus("b", 2), _plus("c", 0), _plus("a", 0)])
        UC = piece([R, A, B], cU, [_plus("c", 2), _plus("a", 0), _plus("b", 0)])

        # legend
        leg = VGroup(tag("a + b + c = 60°", 26),
                     VGroup(_plus("x", 1, 26), tag("= x + 60°", 26)).arrange(RIGHT, buff=0.12),
                     VGroup(_plus("x", 2, 26), tag("= x + 120°", 26)).arrange(RIGHT, buff=0.12))
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_corner(UL, buff=0.56)
        self.play(FadeIn(leg), run_time=0.8)

        self.play(FadeIn(E), FadeIn(ticksE), run_time=0.9)

        def outward(U, W, away, d):
            """Shift of length d off the edge UW, away from the given point."""
            m = (U + W) / 2
            return _out(m, U, W, away, d) - m

        offT = [outward(R, Q, P, 0.45), outward(P, R, Q, 0.45),
                outward(Q, P, R, 0.45)]
        offU = [outward(B, C, A, 0.5), outward(C, A, B, 0.5),
                outward(A, B, C, 0.5)]
        for pc, off in zip((TA, TB, TC), offT):
            pc.shift(off)
        _safe(TA, TB, TC)
        self.play(*[FadeIn(pc) for pc in (TA, TB, TC)], run_time=0.8)
        self.play(*[pc.animate.shift(-off) for pc, off in zip((TA, TB, TC), offT)],
                  run_time=1.3)
        for pc, off in zip((UA, UB, UC), offU):
            pc.shift(off)
        _safe(UA, UB, UC)
        self.play(*[FadeIn(pc) for pc in (UA, UB, UC)], run_time=0.8)
        self.play(*[pc.animate.shift(-off) for pc, off in zip((UA, UB, UC), offU)],
                  run_time=1.4)
        self.bring_to_front(ticksE)

        # a full turn round each of P, Q, R
        ro = VGroup(_plus("60° +", 0, 26), _plus("a", 1, 26), tag("+", 26),
                    _plus("b", 1, 26), tag("+", 26), _plus("c", 2, 26),
                    tag("= 360°", 26)).arrange(RIGHT, buff=0.12)
        ro.to_corner(UR, buff=0.56)
        rings = VGroup(*[Circle(radius=0.6, color=YELLOW_B, stroke_width=4).move_to(V)
                         for V in (R, Q, P)])
        self.hold(0.4)
        self.play(FadeIn(ro), Create(rings[0]), run_time=0.9)
        self.play(Create(rings[1]), run_time=0.5)
        self.play(Create(rings[2]), run_time=0.5)
        self.hold(0.6)
        self.play(FadeOut(rings), run_time=0.4)

        # the outline: a triangle with angles 3a, 3b, 3c
        tri = Polygon(A, B, C, stroke_color=WHITE, stroke_width=5)
        lA = tag("A", 30).move_to(A + 0.36 * _unit(A - G))
        lB = tag("B", 30).move_to(B + 0.36 * _unit(B - G))
        lC = tag("C", 30).move_to(C + 0.36 * _unit(C - G))
        self.play(Create(tri), FadeIn(lA), FadeIn(lB), FadeIn(lC), run_time=0.9)

        # keep the corner letters, drop the inner ones: the corner angles are
        # cut in three equal parts by the pieces' edges
        inner = VGroup(E[1], TA[1][1], TA[1][2], TB[1][1], TB[1][2],
                       TC[1][1], TC[1][2], UA[1][0], UB[1][0], UC[1][0])
        tris = VGroup(*[Line(V, W, color=GREEN_B, stroke_width=6)
                        for V, W in ((A, R), (A, Q), (B, P), (B, R), (C, P), (C, Q))])
        Gm = (P + Q + R) / 3
        lP = tag("P", 24, BLACK).move_to(P + 0.3 * _unit(Gm - P))
        lQ = tag("Q", 24, BLACK).move_to(Q + 0.3 * _unit(Gm - Q))
        lR = tag("R", 24, BLACK).move_to(R + 0.3 * _unit(Gm - R))
        morl = Polygon(P, Q, R, stroke_color=YELLOW_B, stroke_width=6)
        self.play(FadeOut(inner), FadeOut(ticksE), run_time=0.6)
        arcs = VGroup()
        for V, seq in ((A, (B, R, Q, C)), (B, (C, P, R, A)), (C, (A, Q, P, B))):
            for i in range(3):
                arcs.add(angle_arc(V, seq[i], seq[i + 1], 0.42 + 0.07 * i,
                                   [YELLOW_B, ORANGE, YELLOW_B][i], 3))
        self.play(Create(tris), Create(arcs), run_time=1.2)
        self.play(Create(morl), FadeIn(lP), FadeIn(lQ), FadeIn(lR), run_time=0.8)

        corner_labs = [TA[1][0], TB[1][0], TC[1][0], UA[1][1], UA[1][2],
                       UB[1][1], UB[1][2], UC[1][1], UC[1][2]]
        _safe(tri, lA, lB, lC, leg, ro)
        _apart(leg, ro, lA, lB, lC)
        _apart(*corner_labs)
        self.play(Write(caption("angles 3a, 3b, 3c:  the trisectors meet in an "
                                "equilateral triangle", 30)))
        self.hold(2.2)
