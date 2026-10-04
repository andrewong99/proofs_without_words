# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d_w2.py — vpython scenes: Dandelin spheres, a ball as cones from its
centre, the torus, a cube as six pyramids, a ball growing by its surface,
a prism as three tetrahedra and a regular tetrahedron in a cube
(E20, I15, I16, I13, H21, I14, I17).

Each scene_<ID>() is found by name by pww_3d.py / the app. Every
construction shown here is rebuilt from its rule and checked before
anything is drawn: tangencies, tilings, congruences and volumes are
verified numerically, so a construction that does not close fails at
start-up instead of animating a wrong picture.
"""

from pww_3d_kit import *          # noqa: F401,F403  — must come first

import itertools

FPS = 40


# ================================================================ helpers

def _check(cond, what):
    """Refuse to start a scene whose construction is wrong."""
    if not cond:
        raise AssertionError(f"construction check failed: {what}")


def _smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _frames(seconds):
    """Yield t in [0, 1] once per frame for `seconds`."""
    n = max(1, int(round(seconds * FPS)))
    for i in range(n + 1):
        rate(FPS)
        yield i / n


def _wait(seconds):
    for _ in _frames(seconds):
        pass


def _ready(sc):
    """Start the clock once the browser has received the whole scene
    (vpython already blocks at the first canvas until a page connects)."""
    sc.waitfor("draw_complete")


def _pre(t):
    return f"<pre style='margin:0'>{t}</pre>"


def _rows(sc, n):
    """n readout lines under the canvas."""
    rows = []
    for _ in range(n):
        sc.append_to_caption("\n")
        rows.append(wtext(text=_pre(" ")))
    return rows


def _lab(pos, text, col=C_HL, height=16, visible=True, back=False):
    """A label; with back=True it gets a dark backing, so that it can sit
    on a coloured face and stay legible."""
    if back:
        return label(pos=pos, text=text, height=height, box=False, color=col,
                     background=V(0.03, 0.03, 0.05), opacity=0.62,
                     visible=visible)
    return label(pos=pos, text=text, height=height, box=False, color=col,
                 opacity=0, visible=visible)


def _show(objs, flag):
    for ob in objs:
        ob.visible = flag


def _sub(n):
    """An integer in subscript digits."""
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


class _AxisTurn:
    """The rotation by s·angle about a fixed axis through the origin
    (s = 1 is the whole turn): R(s)·v by Rodrigues' formula."""

    def __init__(self, axis, angle):
        self.axis = axis.norm()
        self.theta = angle

    def apply(self, s, v):
        a = s * self.theta
        if abs(a) < 1e-15:
            return V(v.x, v.y, v.z)
        k = self.axis
        c, sn = math.cos(a), math.sin(a)
        return v * c + k.cross(v) * sn + k * (k.dot(v) * (1 - c))


_IDENT = _AxisTurn(V(0, 1, 0), 0.0)


class _Rigid:
    """A rigid body made of several vpython objects (compounds), posed as a
    whole: world = P + R·offset for every part, all parts turned by R."""

    def __init__(self):
        self.parts = []          # (obj, offset from body centre, |axis|)

    def add(self, obj, offset):
        self.parts.append((obj, offset, obj.axis.mag))

    def pose(self, turn, s, P):
        for obj, off, L0 in self.parts:
            obj.pos = P + turn.apply(s, off)
            obj.axis = turn.apply(s, V(1, 0, 0)) * L0
            obj.up = turn.apply(s, V(0, 1, 0))


def _vt(v):
    return (v.x, v.y, v.z)


class _Dim:
    """A dimension line beside the edge p–q: a thin line set off by `off`,
    with a tick at each end and its label beyond the middle."""

    def __init__(self, p, q, off, text, col=C_HL, visible=True, lab_gap=2.4):
        self.off, self.lab_gap = off, lab_gap
        self.line = curve(pos=[p, q], radius=0.008, color=col, visible=visible)
        self.ticks = [curve(pos=[p, q], radius=0.008, color=col, visible=visible)
                      for _ in range(2)]
        self.lab = _lab(p, text, col, visible=visible)
        self.set(p, q)

    def set(self, p, q, trim=0.025):
        d = (q - p).norm()
        a, b = p + d * trim, q - d * trim
        o = self.off
        self.line.modify(0, pos=a + o)
        self.line.modify(1, pos=b + o)
        for tk, e in zip(self.ticks, (a, b)):
            tk.modify(0, pos=e + o * 0.45)
            tk.modify(1, pos=e + o * 1.55)
        self.lab.pos = (p + q) / 2 + o * self.lab_gap

    def show(self, flag):
        for c in [self.line, self.lab] + self.ticks:
            c.visible = flag


def _rot_about_line(p, a, k, ang):
    """The point p turned by `ang` about the line through a with unit
    direction k (Rodrigues)."""
    v = p - a
    c, s = math.cos(ang), math.sin(ang)
    return a + v * c + k.cross(v) * s + k * (k.dot(v) * (1 - c))


# The label offsets and the faces singled out in these scenes were placed by
# eye against this reference direction: what sc.forward reads in vpython 7.6
# before the user turns the view (see new_canvas in pww_3d_kit). It is pinned
# here so the layout does not depend on the vpython version.
_VIEW = V(0, 0, -1)


def _screen_basis(sc):
    """The 'right' and 'up' unit vectors used to lay out labels (the world x
    and y axes, from _VIEW; the canvas argument is kept for the callers)."""
    f = _VIEW.norm()
    right = f.cross(V(0, 1, 0)).norm()
    return right, right.cross(f).norm()


def _polar_angle(u, w, k):
    """Signed angle (about the unit axis k) that turns the part of u
    perpendicular to k onto the part of w perpendicular to k."""
    u = u - k * u.dot(k)
    w = w - k * w.dot(k)
    return math.atan2(k.dot(u.cross(w)), u.dot(w))


# ============================================================== E20

def scene_E20():
    """Dandelin: two balls in the cone, one on each side of the cutting
    plane, each touching the cone along a circle and the plane at a point
    F. For a point P of the section, the tangents from P to the upper ball
    form a cone of equal segments; PF₁ is one (F₁ is on the ball and on the
    plane) and so is PQ₁, Q₁ where the generator through P touches the
    ball: PF₁ turns along that cone onto PQ₁. Likewise PF₂ = PQ₂. So
    PF₁ + PF₂ = Q₁Q₂, the piece of a generator between the two circles of
    contact: the same on every generator. At the ends V₁, V₂ of the long
    axis this constant is V₁V₂ = 2a."""
    sc = new_canvas("PF₁ + PF₂ = 2a   (Dandelin spheres)",
                    "two balls in the cone touch the plane at F₁, F₂ and the cone along two "
                    "circles  ·  tangents from P to one ball are equal",
                    rng=2.45, forward=V(0.2349, -0.42, -0.8766), centre=V(0.0, -0.36, 0))
    sc.fov = 0.5
    s_right, s_up = _screen_basis(sc)
    AL, BE = math.radians(20), math.radians(26)
    sa, ca, sb, cb = math.sin(AL), math.cos(AL), math.sin(BE), math.cos(BE)
    yA, yp = 2.0, 0.0
    A = V(0, yA, 0)                              # apex; the axis points down
    n = V(-sb, cb, 0)                            # the plane's upward normal
    Pp = V(0, yp, 0)                             # where the plane meets the axis

    def gen(ph):
        """Unit vector from the apex down the generator at azimuth ph."""
        return V(sa * math.cos(ph), -ca, sa * math.sin(ph))

    def t_sec(ph):
        """Distance from the apex to the section along generator ph."""
        return (yA - yp) * cb / (ca * cb + sa * sb * math.cos(ph))

    def Q(ph, t):
        return A + gen(ph) * t

    # ---- the two balls: centre on the axis, tangent to the cone, tangent
    # to the plane, one above it and one below
    y1 = (yA * sa + yp * cb) / (cb + sa)
    y2 = (yp * cb - yA * sa) / (cb - sa)
    C1, C2 = V(0, y1, 0), V(0, y2, 0)
    R1, R2 = (yA - y1) * sa, (yA - y2) * sa
    F1 = C1 - n * (C1 - Pp).dot(n)
    F2 = C2 - n * (C2 - Pp).dot(n)
    t1, t2 = (yA - y1) * ca, (yA - y2) * ca      # apex to the contact circles

    # ---- checks: tangency to the cone along a circle, to the plane at F,
    # the balls on opposite sides, and the tangent lengths
    for C, R, t in ((C1, R1, t1), (C2, R2, t2)):
        for ph in [2 * math.pi * i / 24 for i in range(24)]:
            q = Q(ph, t)
            _check(abs((C - A).cross(gen(ph)).mag - R) < 1e-12,
                   "each ball is at distance r from every generator")
            _check(abs((q - C).dot(gen(ph))) < 1e-12 and abs((q - C).mag - R) < 1e-12,
                   "it touches every generator at its contact circle")
            _check(abs(q.y - (yA - t * ca)) < 1e-12, "the contact set is a level circle")
        _check(abs(abs((C - Pp).dot(n)) - R) < 1e-12, "each ball touches the plane")
    _check(abs((F1 - C1).mag - R1) < 1e-12 and abs((F2 - C2).mag - R2) < 1e-12 and
           abs((F1 - Pp).dot(n)) < 1e-12 and abs((F2 - Pp).dot(n)) < 1e-12,
           "F₁, F₂ are the points of contact with the plane")
    _check((C1 - Pp).dot(n) > 0 > (C2 - Pp).dot(n), "one ball above the plane, one below")
    _check(ca * cb > sa * sb, "the plane meets every generator: an ellipse")
    V1, V2 = Q(0, t_sec(0)), Q(math.pi, t_sec(math.pi))
    two_a = (V1 - V2).mag
    _check(abs(two_a - (t2 - t1)) < 1e-12, "Q₁Q₂ = V₁V₂ (= 2a)")
    _check(abs((V1 - F1).mag + (F1 - V2).mag - two_a) < 1e-12 and
           abs((V1 - F2).mag + (F2 - V2).mag - two_a) < 1e-12, "F₁, F₂ lie on V₁V₂")
    for ph in [2 * math.pi * i / 72 for i in range(72)]:
        P = Q(ph, t_sec(ph))
        _check(abs((P - Pp).dot(n)) < 1e-12, "P lies on the plane")
        d1, d2 = (P - F1).mag, (P - F2).mag
        _check(abs(d1 - (t_sec(ph) - t1)) < 1e-12 and abs(d2 - (t2 - t_sec(ph))) < 1e-12,
               "PF₁ = PQ₁ and PF₂ = PQ₂")
        _check(abs(d1 + d2 - (t2 - t1)) < 1e-12, "PF₁ + PF₂ = Q₁Q₂")

    # ---- the cone (apex down to below the lower ball), the plane, the section
    y_bot = y2 - R2 - 0.08
    t_bot = (yA - y_bot) / ca
    NC = 72
    for i in range(NC):
        p0, p1 = 2 * math.pi * i / NC, 2 * math.pi * (i + 1) / NC
        pm = (p0 + p1) / 2
        nm = V(ca * math.cos(pm), sa, ca * math.sin(pm))
        triangle(vs=[vertex(pos=A, normal=nm, color=C_A, opacity=0.16),
                     vertex(pos=Q(p0, t_bot), normal=V(ca * math.cos(p0), sa, ca * math.sin(p0)),
                            color=C_A, opacity=0.16),
                     vertex(pos=Q(p1, t_bot), normal=V(ca * math.cos(p1), sa, ca * math.sin(p1)),
                            color=C_A, opacity=0.16)])
    curve(pos=[Q(2 * math.pi * i / 96, t_bot) for i in range(97)], radius=0.008,
          color=C_A * 1.3)
    for i in range(8):
        ph = 2 * math.pi * (i + 0.5) / 8
        curve(pos=[A, Q(ph, t_bot)], radius=0.004, color=C_A * 1.1)

    centre_e = (V1 + V2) / 2
    m_ax = (V1 - V2).norm()                      # major axis direction
    z_ax = n.cross(m_ax).norm()                  # minor axis direction
    b_semi = math.sqrt((two_a / 2) ** 2 - ((F1 - F2).mag / 2) ** 2)
    ex, ez = two_a / 2 + 0.75, b_semi + 0.7
    pv = [vertex(pos=centre_e + m_ax * (sx * ex) + z_ax * (sz * ez), normal=n,
                 color=V(0.62, 0.47, 0.88), opacity=0.30)
          for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    quad(vs=pv)
    NE = 144
    curve(pos=[Q(2 * math.pi * i / NE, t_sec(2 * math.pi * i / NE)) for i in range(NE + 1)],
          radius=0.014, color=C_D)

    # screen directions along and across the long axis, for the labels
    ms = V(m_ax.dot(s_right), m_ax.dot(s_up), 0).norm()
    along = s_right * ms.x + s_up * ms.y
    across = s_right * (-ms.y) + s_up * ms.x     # "above" the long axis on screen

    # ---- the balls, their contact circles and points
    COL1, COL2 = C_C, C_B
    # (the balls are drawn a hair smaller than they are, so that the cone and
    # the plane, which touch them, do not flicker against their surfaces)
    SHR = 0.008
    ball1 = sphere(pos=C1, radius=R1 - SHR, color=COL1, opacity=0.30, visible=False)
    ball2 = sphere(pos=C2, radius=R2 - SHR, color=COL2, opacity=0.26, visible=False)
    circ1 = ring(pos=V(0, yA - t1 * ca, 0), axis=V(0, 1, 0), radius=t1 * sa,
                 thickness=0.013, color=COL1, visible=False)
    circ2 = ring(pos=V(0, yA - t2 * ca, 0), axis=V(0, 1, 0), radius=t2 * sa,
                 thickness=0.013, color=COL2, visible=False)
    dotF1 = sphere(pos=F1, radius=0.045, color=COL1 * 1.15, visible=False)
    dotF2 = sphere(pos=F2, radius=0.045, color=COL2 * 1.15, visible=False)
    labF1 = _lab(F1 - across * 0.13 + along * 0.15, "F₁", COL1 * 1.2, back=True,
                 visible=False)
    labF2 = _lab(F2 - across * 0.13 - along * 0.15, "F₂", COL2 * 1.2, back=True,
                 visible=False)

    # ---- P, its focal radii, its generator
    PH0 = math.radians(110)                      # P at the front of the section
    gline = curve(pos=[A, Q(PH0, t_bot)], radius=0.006, color=V(0.75, 0.75, 0.72),
                  visible=False)
    seg1 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.019, color=COL1, visible=False)
    seg2 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.019, color=COL2, visible=False)
    gseg1 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.026, color=COL1 * 1.15,
                  visible=False)
    gseg2 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.026, color=COL2 * 1.15,
                  visible=False)
    dotP = sphere(pos=V(0, 0, 0), radius=0.05, color=C_HL, visible=False)
    dotQ1 = sphere(pos=V(0, 0, 0), radius=0.04, color=COL1 * 1.15, visible=False)
    dotQ2 = sphere(pos=V(0, 0, 0), radius=0.04, color=COL2 * 1.15, visible=False)
    labP = _lab(V(0, 0, 0), "P", C_HL, back=True, visible=False)
    labQ1 = _lab(V(0, 0, 0), "Q₁", COL1 * 1.2, back=True, visible=False)
    labQ2 = _lab(V(0, 0, 0), "Q₂", COL2 * 1.2, back=True, visible=False)

    def frame_at(ph):
        return Q(ph, t_sec(ph)), Q(ph, t1), Q(ph, t2)

    def scr(v):
        return V(v.dot(s_right), v.dot(s_up), 0)

    def beside(p, ph):
        """A spot for P's label: outward from the section on the screen and
        off the generator through P."""
        o = scr(p - centre_e)
        o = o.norm() if o.mag > 1e-9 else V(1, 0, 0)
        g = scr(gen(ph)).norm()
        side = V(-g.y, g.x, 0)
        if side.dot(o) < 0:
            side = -side
        w = (o + side * 1.2).norm()
        return p + (s_right * w.x + s_up * w.y) * 0.2

    def place_P(ph, rest=False):
        P, q1, q2 = frame_at(ph)
        dotP.pos = P
        dotQ1.pos, dotQ2.pos = q1, q2
        gline.modify(0, pos=A)
        gline.modify(1, pos=Q(ph, t_bot))
        # at rest the free spot is below and to the right of P (the focal
        # radii rise to its left and right, the generator runs through it)
        labP.pos = P + s_right * 0.16 - s_up * 0.15 if rest else beside(P, ph)
        g = scr(gen(ph)).norm()
        side = s_right * (-g.y) + s_up * g.x  # across the generator on screen
        if side.dot(s_right) < 0:
            side = -side
        labQ1.pos = q1 + side * 0.21
        labQ2.pos = q2 + side * 0.21
        return P, q1, q2

    def set_seg(c, p, q):
        c.modify(0, pos=p)
        c.modify(1, pos=q)

    # ---- the cone of tangents from P to a ball: apex P, axis towards the
    # centre, touching the ball along a circle through F and through Q
    def tangent_cone(P, F, q, C, R):
        dPC = (C - P).mag
        k = (C - P) / dPC
        L = math.sqrt(dPC * dPC - R * R)         # every tangent from P
        M = P + k * (L * L / dPC)                # centre of the circle of contact
        rho = L * R / dPC
        e1 = (F - M).norm()
        e2 = k.cross(e1)
        ang = _polar_angle(F - P, q - P, k)      # the turn about P–centre: F to Q
        return k, L, M, rho, e1, e2, ang

    def on_circle(tc, th):
        k, L, M, rho, e1, e2, ang = tc
        return M + (e1 * math.cos(th) + e2 * math.sin(th)) * rho

    # checked at many positions of P: the circle of contact passes through F
    # and Q, every point of it is a tangent point at distance L, and the
    # turn about P–centre carries F exactly onto Q along it
    for ph in [2 * math.pi * i / 36 for i in range(36)]:
        P, q1, q2 = frame_at(ph)
        for F, q, C, R in ((F1, q1, C1, R1), (F2, q2, C2, R2)):
            tc = tangent_cone(P, F, q, C, R)
            k, L, M, rho, e1, e2, ang = tc
            _check(abs((F - P).mag - L) < 1e-9 and abs((q - P).mag - L) < 1e-9,
                   "PF and PQ are tangents of length L")
            _check((on_circle(tc, 0) - F).mag < 1e-9 and (on_circle(tc, ang) - q).mag < 1e-9,
                   "the circle of contact passes through F and Q")
            _check((_rot_about_line(F, P, k, ang) - q).mag < 1e-9,
                   "the turn about PC carries F onto Q")
            for th in (0.7, 2.1, 3.9, 5.3):
                X = on_circle(tc, th)
                _check(abs((X - C).mag - R) < 1e-9 and abs((X - P).dot(X - C)) < 1e-9 and
                       abs((X - P).mag - L) < 1e-9, "every generator of it is a tangent")

    P0, q10, q20 = frame_at(PH0)
    NT, KF = 60, 36
    # (drawn to just short of the ball, which the cone of tangents touches:
    # tangent surfaces would flicker against each other)
    REACH = 0.965
    tcones = []
    for F, q, C, R, col, colc in ((F1, q10, C1, R1, V(1.0, 0.70, 0.33), COL1),
                                  (F2, q20, C2, R2, V(0.45, 0.90, 0.80), COL2)):
        tc = tangent_cone(P0, F, q, C, R)
        surf = []
        for i in range(NT):
            a0, a1 = 2 * math.pi * i / NT, 2 * math.pi * (i + 1) / NT
            X0, X1 = on_circle(tc, a0), on_circle(tc, a1)
            nr = (X0 - P0).cross(X1 - P0).norm()
            X0, X1 = P0 + (X0 - P0) * REACH, P0 + (X1 - P0) * REACH
            surf.append(triangle(vs=[vertex(pos=P0, normal=nr, color=col, opacity=0.16),
                                     vertex(pos=X0, normal=nr, color=col, opacity=0.16),
                                     vertex(pos=X1, normal=nr, color=col, opacity=0.16)],
                                 visible=False))
        rim = curve(pos=[on_circle(tc, 2 * math.pi * i / 96) for i in range(97)],
                    radius=0.008, color=colc * 1.25, visible=False)
        # the part swept by the turning segment, brighter
        fv = [vertex(pos=P0, color=col, opacity=0.5, normal=V(0, 1, 0))
              for _ in range(KF + 2)]
        fan = [triangle(vs=[fv[0], fv[i], fv[i + 1]], visible=False)
               for i in range(1, KF + 1)]
        tcones.append(dict(tc=tc, surf=surf, rim=rim, fv=fv, fan=fan))

    def show_tcone(d, flag):
        _show(d["surf"] + [d["rim"]], flag)
        if not flag:
            _show(d["fan"], False)

    def set_fan(d, s):
        tc, fv = d["tc"], d["fv"]
        ang = tc[6]
        fv[0].pos = P0
        for j, vx in enumerate(fv[1:]):
            X = on_circle(tc, s * ang * j / KF)
            vx.pos = P0 + (X - P0) * REACH
            vx.normal = (X - P0).cross(tc[0]).norm()
        _show(d["fan"], abs(s * ang) > 1e-3)

    # the major axis, shown at the end
    axis_line = curve(pos=[V1, V2], radius=0.012, color=C_HL, visible=False)
    dotV = [sphere(pos=p, radius=0.04, color=C_HL, visible=False) for p in (V1, V2)]
    labV = [_lab(V1 + along * 0.24, "V₁", C_HL, back=True, visible=False),
            _lab(V2 - along * 0.24, "V₂", C_HL, back=True, visible=False)]
    lab2a = _lab(centre_e - along * 0.22 + across * 0.15, "2a", C_HL, back=True,
                 visible=False)

    row1, row2, row3 = _rows(sc, 3)

    def rows_for(ph):
        P, q1, q2 = frame_at(ph)
        d1, d2 = (P - F1).mag, (P - F2).mag
        row2.text = _pre(f"PF₁ = {d1:.4f} = PQ₁        PF₂ = {d2:.4f} = PQ₂")
        row3.text = _pre(f"PF₁ + PF₂  =  PQ₁ + PQ₂  =  Q₁Q₂  =  {d1 + d2:.4f}"
                         f"      (the same on every generator)")

    every = [ball1, ball2, circ1, circ2, dotF1, dotF2, labF1, labF2, gline, seg1, seg2,
             gseg1, gseg2, dotP, dotQ1, dotQ2, labP, labQ1, labQ2, axis_line, lab2a]
    every += dotV + labV
    _ready(sc)
    while True:
        # ---- the cone and the plane; the section is a closed curve
        _show(every, False)
        for d in tcones:
            show_tcone(d, False)
        row1.text = _pre("a plane cuts the cone in a closed curve")
        row2.text = row3.text = _pre(" ")
        _wait(1.6)
        # ---- a ball drops in from the apex until it meets the plane ...
        row1.text = _pre("a ball inside the cone touches it along a circle;  slid down "
                         "until it touches the plane: F₁")
        ball1.visible = True
        ys = yA - 0.25 / sa
        for t in _frames(1.8):
            y = ys + (y1 - ys) * _smooth(t)
            ball1.pos, ball1.radius = V(0, y, 0), (yA - y) * sa - SHR
        _show([circ1, dotF1, labF1], True)
        _wait(0.6)
        # ---- ... and one rises from below until it meets the plane
        row1.text = _pre("a second ball, below the plane, slid up until it touches "
                         "the plane: F₂")
        ball2.visible = True
        ys = y2 - 0.5
        for t in _frames(1.8):
            y = ys + (y2 - ys) * _smooth(t)
            ball2.pos, ball2.radius = V(0, y, 0), (yA - y) * sa - SHR
            ball2.opacity = 0.26 * min(1.0, 2 * t)
        _show([circ2, dotF2, labF2], True)
        _wait(0.8)
        # ---- a point P of the section, its distances to F₁ and F₂
        P, q1, q2 = place_P(PH0, rest=True)
        set_seg(seg1, P, F1)
        set_seg(seg2, P, F2)
        _show([dotP, labP, seg1, seg2], True)
        row1.text = _pre("P on the section:  PF₁ and PF₂")
        _wait(1.4)
        _show([gline, dotQ1, dotQ2, labQ1, labQ2], True)
        row1.text = _pre("the generator through P touches the balls at Q₁ and Q₂")
        _wait(1.4)
        # ---- the tangents from P to each ball form a cone; PF turns along it
        texts = ["the tangents from P to the upper ball form a cone: all of one length.  "
                 "PF₁ turns along it onto PQ₁",
                 "the tangents from P to the lower ball form a cone too:  "
                 "PF₂ turns along it onto PQ₂"]
        for d, F, q, seg, gseg, txt in ((tcones[0], F1, q1, seg1, gseg1, texts[0]),
                                        (tcones[1], F2, q2, seg2, gseg2, texts[1])):
            row1.text = _pre(txt)
            show_tcone(d, True)
            _wait(1.0)
            for t in _frames(2.4):
                s = _smooth(t)
                set_seg(seg, P, on_circle(d["tc"], s * d["tc"][6]))
                set_fan(d, s)
            set_seg(gseg, P, q)
            gseg.visible = True
            _wait(0.8)
            show_tcone(d, False)
        rows_for(PH0)
        row1.text = _pre("so PF₁ + PF₂ is the piece Q₁Q₂ of the generator between the "
                         "two circles")
        _wait(1.8)
        # ---- P runs round the section: Q₁Q₂ only turns about the axis
        set_seg(seg1, P, F1)
        set_seg(seg2, P, F2)
        _show([labQ1, labQ2], False)
        row1.text = _pre("every generator crosses the two circles the same distance apart")
        for t in _frames(8.0):
            ph = PH0 + 2 * math.pi * _smooth(t)
            P, q1, q2 = place_P(ph)
            set_seg(seg1, P, F1)
            set_seg(seg2, P, F2)
            set_seg(gseg1, P, q1)
            set_seg(gseg2, P, q2)
            if int(round(t * 320)) % 3 == 0:
                rows_for(ph)
        place_P(PH0, rest=True)
        rows_for(PH0)
        # ---- the constant is the major axis: P at either end of it
        _show([axis_line, lab2a, labQ1, labQ2] + dotV + labV, True)
        row1.text = _pre("with P at V₁ and then at V₂ the two sums add to 2·V₁V₂:  "
                         "the constant is V₁V₂ = 2a")
        row3.text = _pre(f"PF₁ + PF₂  =  Q₁Q₂  =  V₁V₂  =  2a  =  {two_a:.4f}")
        _wait(5.5)


# ============================================================== I15

_PHI = (1 + 5 ** 0.5) / 2


def _icosahedron():
    """Unit icosahedron: vertices and its 20 triangles (index triples)."""
    vs = []
    for a in (1, -1):
        for b in (_PHI, -_PHI):
            vs += [V(0, a, b), V(a, b, 0), V(b, 0, a)]
    vs = [v.norm() for v in vs]
    e = min((p - q).mag for p, q in itertools.combinations(vs, 2))
    tris = [t for t in itertools.combinations(range(12), 3)
            if all(abs((vs[i] - vs[j]).mag - e) < 1e-9 for i, j in itertools.combinations(t, 2))]
    return vs, tris


def _geodesic(nu, turn=None):
    """The icosahedron's faces cut into nu² triangles and pushed out onto
    the unit sphere: (points, triangles). `turn` turns the icosahedron
    first."""
    V0, T0 = _icosahedron()
    if turn is not None:
        V0 = [turn.apply(1, v) for v in V0]
    pts, key, tris = [], {}, []

    def idx(p):
        p = p.norm()
        k = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
        if k not in key:
            key[k] = len(pts)
            pts.append(p)
        return key[k]

    for a, b, c in T0:
        A_, B_, C_ = V0[a], V0[b], V0[c]
        g = {(i, j): idx(A_ + (B_ - A_) * (i / nu) + (C_ - A_) * (j / nu))
             for i in range(nu + 1) for j in range(nu + 1 - i)}
        for i in range(nu):
            for j in range(nu - i):
                tris.append((g[(i, j)], g[(i + 1, j)], g[(i, j + 1)]))
                if i + j < nu - 1:
                    tris.append((g[(i + 1, j)], g[(i + 1, j + 1)], g[(i, j + 1)]))
    return pts, tris


def _circumscribed(nu, r, turn=None):
    """The polyhedron around the ball of radius r whose faces touch the ball
    at the points r·u of a geodesic sphere: face i is the polygon cut from
    the tangent plane x·u_i = r by its neighbours (the polar of the geodesic
    sphere). Returns [(u_i, polygon)], every polygon in cyclic order."""
    U, T = _geodesic(nu, turn)
    corner = []
    for t in T:
        a, b, c = (U[i] for i in t)
        nr = (b - a).cross(c - a).norm()
        if nr.dot(a) < 0:
            nr = -nr
        d = nr.dot(a)
        # the geodesic triangles are the faces of the hull of the points
        _check(all(nr.dot(U[m]) < d - 1e-6 for m in range(len(U)) if m not in t),
               "the geodesic triangles bound a convex polyhedron")
        corner.append(nr * (r / d))              # on the three tangent planes of t
    around = {i: [] for i in range(len(U))}
    for k, t in enumerate(T):
        for i in t:
            around[i].append(k)
    out = []
    for i, u in enumerate(U):
        e1 = u.cross(V(0.31, 0.72, 0.18)).norm()
        e2 = u.cross(e1)
        ks = sorted(around[i], key=lambda k: math.atan2(corner[k].dot(e2), corner[k].dot(e1)))
        out.append((u, [corner[k] for k in ks]))
    return out


def scene_I15():
    """V = ⅓·r·S: around the ball put a polyhedron whose every face touches
    it. From the centre each face is the base of a pyramid, and its height
    is the radius to the point of contact, which stands at right angles on
    the face: r. So the polyhedron is ⅓·r·(sum of its faces) = ⅓·r·S,
    exactly, whatever the faces. With more and smaller faces the pyramids
    become thin cones on small patches and the polyhedron closes in on the
    ball: V(ball) = ⅓·r·4πr² = ⁴⁄₃πr³."""
    sc = new_canvas("V = ⅓·r·S = ⁴⁄₃πr³",
                    "a polyhedron whose faces all touch the ball is pyramids of height r "
                    "from its centre  ·  more and smaller faces close in on the ball",
                    rng=2.3, forward=V(-0.42, -0.40, -0.81), centre=V(0, 0, 0))
    sc.fov = 0.45
    s_right, s_up = _screen_basis(sc)
    r = 1.15
    NUS = [1, 2, 4]                              # polyhedra that come apart
    NU_LAST = 8                                  # the last, shown whole
    # the pyramid singled out is seen from the side, its base half towards
    # us: the icosahedron is turned so that one of its corners (a pentagonal
    # face of every polyhedron below) points that way
    want = (s_right * 0.85 + s_up * 0.22 - _VIEW.norm() * 0.40).norm()
    v0 = max(_icosahedron()[0], key=lambda v: v.dot(want))
    ax = v0.cross(want)
    TURN = _AxisTurn(ax, math.asin(min(1.0, ax.mag))) if ax.mag > 1e-12 else None
    if TURN is not None:
        _check((TURN.apply(1, v0) - want).mag < 1e-9, "the icosahedron turned into place")

    def area(poly, c):
        return sum((p - c).cross(q - c).mag / 2 for p, q in zip(poly, poly[1:] + poly[:1]))

    # ---- the polyhedra, checked: every face on its tangent plane, touching
    # the ball inside the face; the faces close up; the volume found from the
    # faces alone is ⅓·r·S
    polys = {}
    for nu in NUS + [NU_LAST]:
        faces = _circumscribed(nu, r, TURN)
        S = 0.0
        vol = 0.0
        for u, poly in faces:
            c = u * r
            _check(all(abs(p.dot(u) - r) < 1e-9 for p in poly), "every face lies on a tangent plane")
            for p, q in zip(poly, poly[1:] + poly[:1]):
                _check((p - c).cross(q - c).dot(u) > 0, "the point of contact is inside the face")
                vol += p.dot(q.cross(c)) / 6        # tetrahedron (centre, p, q, c)
            S += area(poly, c)
        # the divergence theorem, from an off-centre point: closes up exactly
        g = V(0.137, -0.219, 0.071)
        vol2 = 0.0
        for u, poly in faces:
            c = u * r
            for p, q in zip(poly, poly[1:] + poly[:1]):
                vol2 += (p - g).dot((q - g).cross(c - g)) / 6
        _check(abs(vol - vol2) < 1e-9, "the faces close up into a solid")
        _check(abs(vol - r * S / 3) < 1e-9, "V = ⅓·r·S")
        _check(S > 4 * math.pi * r * r and vol > 4 / 3 * math.pi * r ** 3, "it holds the ball")
        polys[nu] = (faces, S, vol)
    Ss = [polys[nu][1] for nu in NUS + [NU_LAST]]
    _check(all(a > b for a, b in zip(Ss, Ss[1:])), "more faces, closer to the ball")
    _check(polys[NU_LAST][1] / (4 * math.pi * r * r) < 1.006, "the last one hugs the ball")

    # ---- the pyramids as triangles on their own vertices, so that each can
    # be moved out along its axis. The twelve of the first polyhedron take
    # their colours by opposite pairs, so no two neighbours match; later the
    # pentagons are orange and the hexagons blue.
    PAL = [C_A, C_B, C_C, C_D, C_E, C_F]
    faces1 = polys[1][0]
    pair_of, npair = {}, 0
    for i, (u, _) in enumerate(faces1):
        if i in pair_of:
            continue
        j = min(range(len(faces1)), key=lambda m: (faces1[m][0] + u).mag)
        _check((faces1[j][0] + u).mag < 1e-9, "the faces come in opposite pairs")
        pair_of[i] = pair_of[j] = npair
        npair += 1

    def shade(nu, i, nv):
        if nu == 1:
            return PAL[pair_of[i]]
        base = C_C if nv == 5 else C_A
        return base * (0.82 + 0.10 * (i % 3))

    SIDE = 0.68                                  # side faces a little darker
    O = V(0, 0, 0)
    built = {}
    for nu in NUS:
        faces = polys[nu][0]
        pyr = []
        for i, (u, poly) in enumerate(faces):
            col = shade(nu, i, len(poly))
            c = u * r
            vx = []

            def mk(p, nr, cl):
                v = vertex(pos=p, normal=nr, color=cl)
                vx.append((v, V(p.x, p.y, p.z)))
                return v
            vc = mk(c, u, col)
            ring_ = [mk(p, u, col) for p in poly]
            tris = [triangle(vs=[vc, ring_[j], ring_[(j + 1) % len(poly)]], visible=False)
                    for j in range(len(poly))]
            for j in range(len(poly)):
                p, q = poly[j], poly[(j + 1) % len(poly)]
                nr = p.cross(q).norm()
                tris.append(triangle(vs=[mk(O, nr, col * SIDE), mk(p, nr, col * SIDE),
                                         mk(q, nr, col * SIDE)], visible=False))
            pyr.append(dict(u=u, poly=poly, c=c, vx=vx, tris=tris, col=col,
                            nbase=len(poly) + 1))
        built[nu] = pyr
    # the last polyhedron, whole: its faces only
    last_tris = []
    for i, (u, poly) in enumerate(polys[NU_LAST][0]):
        col = shade(NU_LAST, i, len(poly))
        c = u * r
        for j in range(len(poly)):
            p, q = poly[j], poly[(j + 1) % len(poly)]
            last_tris.append(triangle(vs=[vertex(pos=c, normal=u, color=col),
                                          vertex(pos=p, normal=u, color=col),
                                          vertex(pos=q, normal=u, color=col)],
                                      visible=False))

    def show_poly(nu, flag):
        for P_ in built[nu]:
            _show(P_["tris"], flag)

    def move(P_, d):
        sh = P_["u"] * d
        for v, home in P_["vx"]:
            v.pos = home + sh

    def explode(nu, d):
        for P_ in built[nu]:
            move(P_, d)

    def set_opacity(nu, op):
        for P_ in built[nu]:
            for v, _ in P_["vx"]:
                v.opacity = op

    def tint(nu, keep):
        """Dim every pyramid but `keep` (none: -1, all back to normal); the
        kept one keeps an opaque base on a see-through body."""
        for k, P_ in enumerate(built[nu]):
            for j, (v, _) in enumerate(P_["vx"]):
                is_base = j < P_["nbase"]
                cl = P_["col"] * (1.0 if is_base else SIDE)
                if keep >= 0 and k != keep:
                    cl = cl * 0.5
                v.color = cl
                v.opacity = 1.0 if (k != keep or is_base) else 0.3

    def pick(nu):
        return max(range(len(built[nu])), key=lambda k: built[nu][k]["u"].dot(want))

    ball = sphere(pos=O, radius=r - 0.004, color=C_B, opacity=0.9)
    rad_line = curve(pos=[O, O], radius=0.012, color=C_HL, visible=False)
    lab_r0 = _lab(O, "r", C_HL, back=True, visible=False)
    touch = [sphere(pos=P_["c"] * 1.002, radius=0.03, color=C_HL, visible=False)
             for P_ in built[1]]
    h_line = curve(pos=[O, O], radius=0.014, color=C_HL, visible=False)
    h_dot = [sphere(pos=O, radius=0.035, color=C_HL, visible=False) for _ in range(2)]
    sq_mark = curve(pos=[O, O, O], radius=0.008, color=C_HL, visible=False)
    e_base = curve(pos=[O] * 7, radius=0.01, color=C_HL * 0.95, visible=False)
    e_lat = [curve(pos=[O, O], radius=0.008, color=C_HL * 0.95, visible=False)
             for _ in range(6)]
    lab_h = _lab(O, "r", C_HL, back=True, visible=False)
    lab_A = _lab(O, "A", C_HL, back=True, visible=False)

    def single_out(nu, k, d, flag, name_base=True):
        """The pyramid k, pulled out to distance d: its edges, its height r
        from the apex to the point of contact, square to the base."""
        P_ = built[nu][k]
        u, poly = P_["u"], P_["poly"]
        sh = u * d
        a, b = sh, u * r + sh
        h_line.modify(0, pos=a)
        h_line.modify(1, pos=b)
        h_dot[0].pos, h_dot[1].pos = a, b
        w = (poly[0] - P_["c"]).norm()               # a direction in the face
        m = 0.09
        sq_mark.modify(0, pos=b + w * m)
        sq_mark.modify(1, pos=b + w * m - u * m)
        sq_mark.modify(2, pos=b - u * m)
        pts = [p + sh for p in poly]
        for j in range(7):
            e_base.modify(j, pos=pts[j % len(pts)])
        for j, cv in enumerate(e_lat):
            if j < len(pts):
                cv.modify(0, pos=a)
                cv.modify(1, pos=pts[j])
            cv.visible = flag and j < len(pts)
        hs = V((b - a).dot(s_right), (b - a).dot(s_up), 0).norm()
        perp = V(-hs.y, hs.x, 0)
        if perp.y < 0:
            perp = -perp
        lab_h.pos = (a + b) / 2 + (s_right * perp.x + s_up * perp.y) * 0.16
        lab_A.pos = b + u * 0.2
        _show([h_line, sq_mark, e_base, lab_h] + h_dot, flag)
        lab_A.visible = flag and name_base

    row1, row2, row3 = _rows(sc, 3)

    def numbers(nu):
        faces, S, vol = polys[nu]
        return (f"{len(faces)} faces:   S = {S / r ** 2:.4f}·r²    V = ⅓·r·S = {vol / r ** 3:.4f}·r³"
                f"        ball:  4π·r² = {4 * math.pi:.4f}·r²    ⁴⁄₃π·r³ = "
                f"{4 / 3 * math.pi:.4f}·r³")

    EXP = {1: 0.55, 2: 0.40, 4: 0.30}            # how far the pyramids part
    OUT = {1: 0.55, 2: 0.55, 4: 0.55}            # and the singled-out one, further
    _ready(sc)
    while True:
        # ---- the ball, then a polyhedron whose every face touches it
        for nu in NUS:
            show_poly(nu, False)
            explode(nu, 0)
            tint(nu, -1)
        _show(last_tris, False)
        ball.visible, ball.opacity = True, 0.9
        rdir = (s_right * 0.6 + s_up * 0.8).norm()
        rad_line.modify(1, pos=rdir * r)
        lab_r0.pos = rdir * (r / 2) + s_right * 0.14
        _show([rad_line, lab_r0], True)
        row1.text = _pre("a ball of radius r")
        row2.text = row3.text = _pre(" ")
        _wait(1.5)
        _show([rad_line, lab_r0], False)
        set_opacity(1, 0)
        show_poly(1, True)
        _show(touch, True)
        row1.text = _pre("around it a polyhedron whose every face touches the ball  ·  12 faces")
        for t in _frames(1.4):
            set_opacity(1, 0.5 * t)
        _wait(1.0)
        _show(touch, False)
        for t in _frames(0.6):
            set_opacity(1, 0.5 + 0.5 * t)
            ball.opacity = 0.9 * (1 - t)
        ball.visible = False
        # ---- it comes apart into pyramids, one on each face, apex at the centre
        row1.text = _pre("from the centre, every face is the base of a pyramid")
        for t in _frames(1.6):
            explode(1, EXP[1] * _smooth(t))
        k1 = pick(1)
        P1 = built[1][k1]
        tint(1, k1)
        for t in _frames(0.6):
            move(P1, EXP[1] + OUT[1] * _smooth(t))
        single_out(1, k1, EXP[1] + OUT[1], True)
        row1.text = _pre("its height is the radius to the point of contact, square to the "
                         "face:  r.   The pyramid is ⅓·r·A")
        row2.text = _pre("all twelve:  V = ⅓·r·A₁ + ⅓·r·A₂ + … + ⅓·r·A₁₂  =  "
                         "⅓·r·(A₁ + A₂ + … + A₁₂)  =  ⅓·r·S")
        row3.text = _pre(numbers(1))
        _wait(3.8)
        single_out(1, k1, 0, False)
        for t in _frames(0.5):
            move(P1, EXP[1] + OUT[1] * (1 - _smooth(t)))
        tint(1, -1)
        for t in _frames(1.0):
            explode(1, EXP[1] * (1 - _smooth(t)))
        _wait(0.4)
        # ---- more and smaller faces: thinner pyramids, closer to the ball
        prev = 1
        for nu in NUS[1:]:
            show_poly(prev, False)
            show_poly(nu, True)
            prev = nu
            n_f = len(polys[nu][0])
            row1.text = _pre(f"more, smaller faces, all touching the ball  ·  {n_f} faces:  "
                             "the pyramids get thinner, still of height r")
            row2.text = _pre(f"all {n_f}:  V = ⅓·r·(A₁ + A₂ + … + A{_sub(n_f)})  =  ⅓·r·S")
            row3.text = _pre(numbers(nu))
            _wait(0.6)
            for t in _frames(1.3):
                explode(nu, EXP[nu] * _smooth(t))
            k = pick(nu)
            Pk = built[nu][k]
            tint(nu, k)
            for t in _frames(0.5):
                move(Pk, EXP[nu] + OUT[nu] * _smooth(t))
            single_out(nu, k, EXP[nu] + OUT[nu], True, name_base=False)
            _wait(2.0)
            single_out(nu, k, 0, False)
            for t in _frames(0.4):
                move(Pk, EXP[nu] + OUT[nu] * (1 - _smooth(t)))
            tint(nu, -1)
            for t in _frames(1.0):
                explode(nu, EXP[nu] * (1 - _smooth(t)))
        # ---- in the limit: the ball itself
        show_poly(prev, False)
        _show(last_tris, True)
        n_f = len(polys[NU_LAST][0])
        row1.text = _pre(f"{n_f} faces: the polyhedron closes in on the ball;  V = ⅓·r·S for "
                         "every one of them, so for the ball too")
        row2.text = _pre("V(ball)  =  ⅓·r·S(ball)  =  ⅓·r·4πr²  =  ⁴⁄₃πr³")
        row3.text = _pre(numbers(NU_LAST))
        _wait(5.5)


# ============================================================== I16

def _rot_y(a, v):
    """v turned by a about the vertical axis (the turn that carries the
    direction (sin φ, 0, cos φ) to (sin(φ + a), 0, cos(φ + a)))."""
    c, s = math.cos(a), math.sin(a)
    return V(v.x * c + v.z * s, v.y, v.z * c - v.x * s)


def scene_I16():
    """V = 2πR·πr²: cut the torus into N equal wedges. Each wedge's outer
    side is longer than its middle by as much as its inner side is shorter:
    (R + x)·Δφ and (R − x)·Δφ. The front half straightens into a row with
    gaps, outer sides to the front; the back half straightens behind it,
    outer sides to the back, and slides forward into the gaps: the planar
    cut faces fit exactly, long and short sides alternate, and the wedges
    form a log of length 2πR along its middle. As the wedges get thinner the
    log straightens into the cylinder of base πr² and length 2πR."""
    import numpy as np
    sc = new_canvas("V = 2πR · πr²   (Pappus)",
                    "cut the torus into wedges and straighten it:  each wedge's long outer "
                    "side next to another's short inner side",
                    rng=2.75, forward=V(0, -0.55, -0.835), centre=V(0, -0.12, 0.55))
    sc.fov = 0.45
    R, r, N = 1.2, 0.5, 16
    H = math.pi / N                              # half the angle of a wedge
    ZA, D, GAP = 1.4, 1.2, 0.2                   # row line, wait behind, first cut
    c0 = V(0, 0, R)                              # middle of the wedge W₀ (towards us)
    cL = V(-R * math.sin(H), 0, R * math.cos(H))  # centres of its two cut faces
    cR = V(R * math.sin(H), 0, R * math.cos(H))
    nL = V(-math.cos(H), 0, -math.sin(H))        # and their outward normals
    nR = V(math.cos(H), 0, -math.sin(H))

    def dirn(ph):
        return V(math.sin(ph), 0, math.cos(ph))

    # a pose (a, P): the wedge W₀ turned by a about the vertical through its
    # middle c₀, which is then put at P: world = P + turn_a(p − c₀)
    def torus_pose(ph, gap=0.0):
        return ph, _rot_y(ph, c0) + dirn(ph) * gap

    def row_pose(j, back=0.0):
        """Place j of the row: even places hold W₀ itself, odd places W₀
        given a half-turn about the vertical through its right face centre;
        consecutive places share a cut face."""
        m, odd = divmod(j, 2)
        sh = V((4 * m - (N - 1)) * R * math.sin(H), 0, ZA - R * math.cos(H) - back)
        if not odd:
            return 0.0, c0 + sh
        return math.pi, _rot_y(math.pi, c0) + cR * 2 + sh

    def world(pose, p):
        return pose[1] + _rot_y(pose[0], p - c0)

    def faces(j):
        """(left centre, left normal, right centre, right normal) of place j."""
        P = row_pose(j)
        if j % 2 == 0:
            return world(P, cL), _rot_y(P[0], nL), world(P, cR), _rot_y(P[0], nR)
        return world(P, cR), _rot_y(P[0], nR), world(P, cL), _rot_y(P[0], nL)

    # ---- checks: neighbours in the row share a cut face exactly
    for j in range(N - 1):
        _, _, rc, rn = faces(j)
        lc, ln, _, _ = faces(j + 1)
        _check((rc - lc).mag < 1e-12 and (rn + ln).mag < 1e-12 and abs(rc.y) < 1e-12,
               "neighbouring wedges in the row share a cut face")
    _check(abs(faces(0)[0].x + faces(N - 1)[2].x) < 1e-12, "the row is centred")
    # each wedge, sliced along its axis: the slice at x outside the middle
    # is (R + x)·2H long; pairs ±x average R·2H, so the wedge holds πr²·R·2H
    nq = 300
    xs = (np.arange(nq) + 0.5) / nq * 2 * r - r
    X, Y = np.meshgrid(xs, xs)
    ins = X ** 2 + Y ** 2 <= r * r
    vol_w = float(((R + X) * 2 * H)[ins].sum() * (2 * r / nq) ** 2)
    _check(abs(vol_w / (math.pi * r * r * R * 2 * H) - 1) < 2e-3,
           "a wedge holds πr² times its middle arc R·Δφ")
    _check(abs(float((X * ins).sum())) < 1e-9, "the outer and inner halves balance about the middle")
    long_, short_ = (R + r) * 2 * H, (R - r) * 2 * H
    _check(abs((N // 2) * (long_ + short_) - 2 * math.pi * R) < 1e-12,
           "N/2 long and N/2 short sides make 2πR")

    # ---- which wedge goes where: the front half to the even places (outer
    # side to the front), the back half to the odd places (outer side back),
    # each half in the order of its wedges from left to right
    phis = [(k + 0.5) * 2 * H for k in range(N)]
    phis = [p if p <= math.pi else p - 2 * math.pi for p in phis]
    Fh = sorted([p for p in phis if abs(p) < math.pi / 2], key=math.sin)
    Bh = sorted([p for p in phis if abs(p) > math.pi / 2], key=math.sin)
    plan = [(p, 2 * i) for i, p in enumerate(Fh)] + [(p, 2 * i + 1) for i, p in enumerate(Bh)]
    _check(sorted(j for _, j in plan) == list(range(N)), "every place filled once")

    def nearest(a1, a0):
        return a1 + 2 * math.pi * round((a0 - a1) / (2 * math.pi))

    DA, DB, DS = 2.2, 2.2, 1.3                   # front half, back half, slide

    def pose_at(w, T):
        """Wedge w at time T after the cut: the front half straightens
        (0 … DA), the back half straightens behind it (DA … DA+DB), then
        slides forward into the gaps (… DA+DB+DS)."""
        ph, j = w
        a0, P0 = torus_pose(ph, GAP)
        if j % 2 == 0:
            a1, P1 = row_pose(j)
            u = _smooth(T / DA)
            return a0 + (nearest(a1, a0) - a0) * u, P0 + (P1 - P0) * u
        a1, P1 = row_pose(j, back=D)
        a1 = nearest(a1, a0)
        if T < DA + DB:
            u = _smooth((T - DA) / DB)
            return a0 + (a1 - a0) * u, P0 + (P1 - P0) * u
        P2 = row_pose(j)[1]
        return a1, P1 + (P2 - P1) * _smooth((T - DA - DB) / DS)

    # ---- no wedge passes through another: points spread through every
    # wedge are tested against every other wedge, all the way
    pts = []
    for ph in np.linspace(-H * 0.95, H * 0.95, 5):
        for rr in (0.0, 0.5 * r, 0.95 * r):
            for th in np.linspace(0, 2 * math.pi, 8 if rr > 0 else 1, endpoint=False):
                rho = R + rr * math.cos(th)
                pts.append([rho * math.sin(ph), rr * math.sin(th), rho * math.cos(ph)])
    pts = np.array(pts)
    c0n = np.array([0.0, 0.0, R])

    def rot_np(a, p):
        c, s = math.cos(a), math.sin(a)
        return np.stack([p[:, 0] * c + p[:, 2] * s, p[:, 1], p[:, 2] * c - p[:, 0] * s], -1)

    def in_W0(q, shrink=0.015):
        rho = np.hypot(q[:, 0], q[:, 2])
        ph = np.arctan2(q[:, 0], q[:, 2])
        return (np.abs(ph) < H * (1 - shrink)) & \
               ((rho - R) ** 2 + q[:, 1] ** 2 < (r * (1 - shrink)) ** 2)

    for T in np.linspace(0, DA + DB + DS, 150):
        poses = [pose_at(w, T) for w in plan]
        wp = [rot_np(a, pts - c0n) + np.array([P.x, P.y, P.z]) for a, P in poses]
        for i in range(N):
            for k in range(N):
                if k != i:
                    a, P = poses[k]
                    q = rot_np(-a, wp[i] - np.array([P.x, P.y, P.z])) + c0n
                    _check(not in_W0(q).any(), "no wedge passes through another")
    for w in plan:
        a, P = pose_at(w, DA + DB + DS)
        a1, P1 = row_pose(w[1])
        _check((P - P1).mag < 1e-12 and abs(math.sin((a - a1) / 2)) < 1e-12,
               "every wedge ends exactly in its place")

    # ---- the wedge, drawn once per colour and cloned: its curved side, its
    # two cut faces, and thin bands along its outer and inner equator
    NPH, NTH = 6, 40

    def tube(ph, th, rad=r):
        rho = R + rad * math.cos(th)
        return V(rho * math.sin(ph), rad * math.sin(th), rho * math.cos(ph))

    def tube_n(ph, th):
        return V(math.cos(th) * math.sin(ph), math.sin(th), math.cos(th) * math.cos(ph))

    def build(col):
        parts = []
        PH = [-H + 2 * H * i / NPH for i in range(NPH + 1)]
        TH = [2 * math.pi * k / NTH for k in range(NTH + 1)]
        for i in range(NPH):
            for k in range(NTH):
                vs = [vertex(pos=tube(PH[a], TH[b]), normal=tube_n(PH[a], TH[b]), color=col)
                      for a, b in ((i, k), (i + 1, k), (i + 1, k + 1), (i, k + 1))]
                parts.append(triangle(vs=[vs[0], vs[1], vs[2]]))
                parts.append(triangle(vs=[vs[0], vs[2], vs[3]]))
        for ph, nr in ((-H, nL), (H, nR)):
            cc = V(R * math.sin(ph), 0, R * math.cos(ph))
            for k in range(NTH):
                parts.append(triangle(vs=[vertex(pos=cc, normal=nr, color=col * 0.72),
                                          vertex(pos=tube(ph, TH[k]), normal=nr, color=col * 0.72),
                                          vertex(pos=tube(ph, TH[k + 1]), normal=nr,
                                                 color=col * 0.72)]))
        for th0, bc in ((0.0, C_D), (math.pi, C_HL)):  # outer, inner equator
            dth = 0.075
            for i in range(NPH):
                vs = [vertex(pos=tube(PH[a], th0 + e, r * 1.006), normal=tube_n(PH[a], th0),
                             color=bc)
                      for a, e in ((i, -dth), (i + 1, -dth), (i + 1, dth), (i, dth))]
                parts.append(triangle(vs=[vs[0], vs[1], vs[2]]))
                parts.append(triangle(vs=[vs[0], vs[2], vs[3]]))
        return compound(parts, origin=c0, visible=False)

    COL_F, COL_B = C_C, C_B
    protoF, protoB = build(COL_F), build(COL_B)
    bodies = []
    for ph, j in plan:
        if j % 2 == 0:
            cp = protoF if not any(b[1] is protoF for b in bodies) else protoF.clone()
        else:
            cp = protoB if not any(b[1] is protoB for b in bodies) else protoB.clone()
        cp.visible = False
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        bodies.append((body, cp, (ph, j)))

    def place(body, pose):
        body.pose(_AxisTurn(V(0, 1, 0), pose[0]), 1, pose[1])

    # ---- marks: the cuts, R and r on the whole torus; the cylinder at the end
    cuts = [ring(pos=dirn(k * 2 * H) * R, axis=V(math.cos(k * 2 * H), 0, -math.sin(k * 2 * H)),
                 radius=r * 1.004, thickness=0.009, color=V(0.08, 0.08, 0.1), visible=False)
            for k in range(N)]
    phR = math.radians(-118)
    y_top = r + 0.07
    dimR = _Dim(V(0, y_top, 0), dirn(phR) * R + V(0, y_top, 0), V(0, 0.07, 0), "R",
                lab_gap=2.6)
    axis_tick = curve(pos=[V(0, -r, 0), V(0, r + 0.3, 0)], radius=0.008, color=C_HL * 0.8)
    pr = V(-(R + r) - 0.14, 0, 0)                 # beside the left of the tube
    dimr = _Dim(pr, pr + V(0, r, 0), V(-0.07, 0, 0), "r", lab_gap=2.8)
    xa, xb = -math.pi * R, math.pi * R
    # (a little wider than the log, whose wedges bulge out by R(1 − cos H))
    rc_ = r + R * (1 - math.cos(H)) + 0.03
    cyl = cylinder(pos=V(xa, 0, ZA), axis=V(xb - xa, 0, 0), radius=rc_, color=C_HL,
                   opacity=0.16, visible=False)
    rims = [ring(pos=V(x, 0, ZA), axis=V(1, 0, 0), radius=rc_, thickness=0.012,
                 color=C_HL, visible=False) for x in (xa, xb)]
    dim2 = _Dim(V(xa, -r - 0.16, ZA + 0.25), V(xb, -r - 0.16, ZA + 0.25), V(0, -0.07, 0),
                "2πR", lab_gap=3.2)
    lab_pr = _lab(V(xb - 0.3, rc_ + 0.3, ZA), "πr²", C_HL, back=True, visible=False)
    dim2.show(False)

    row1, row2, row3 = _rows(sc, 3)
    _ready(sc)
    while True:
        # ---- the whole torus, cut into N equal wedges
        for body, cp, w in bodies:
            place(body, torus_pose(w[0]))
            cp.visible = True
        _show(cuts + [cyl, lab_pr] + rims, False)
        dim2.show(False)
        dimR.show(True)
        dimr.show(True)
        axis_tick.visible = True
        row1.text = _pre("a torus:  a disc of radius r carried round a circle of radius R")
        row2.text = _pre(f"outer equator 2π(R + r) = {2 * math.pi * (R + r):.4f}      "
                         f"middle circle 2πR = {2 * math.pi * R:.4f}      inner equator "
                         f"2π(R − r) = {2 * math.pi * (R - r):.4f}      (R = {R:g}, r = {r:g})")
        row3.text = _pre(" ")
        _wait(2.2)
        _show(cuts, True)
        row1.text = _pre(f"cut it into {N} equal wedges")
        _wait(1.0)
        _show(cuts, False)
        dimR.show(False)
        dimr.show(False)
        axis_tick.visible = False
        for t in _frames(0.8):
            for body, cp, w in bodies:
                place(body, torus_pose(w[0], GAP * _smooth(t)))
        row2.text = _pre(f"each wedge:  outer side (R + r)·Δφ = {long_:.4f},  middle R·Δφ = "
                         f"{R * 2 * H:.4f},  inner side (R − r)·Δφ = {short_:.4f}"
                         f"      (Δφ = 2π/{N})")
        # ---- the front half straightens out, the back half behind it
        row1.text = _pre("the front half straightens into a row, outer sides to the front, "
                         "a gap after each wedge")
        for t in _frames(DA):
            for body, cp, w in bodies:
                if w[1] % 2 == 0:
                    place(body, pose_at(w, DA * t))
        row1.text = _pre("the back half straightens behind it, outer sides to the back")
        for t in _frames(DB):
            for body, cp, w in bodies:
                if w[1] % 2 == 1:
                    place(body, pose_at(w, DA + DB * t))
        _wait(0.4)
        row1.text = _pre("and slides forward into the gaps:  the cut faces fit exactly")
        for t in _frames(DS):
            for body, cp, w in bodies:
                if w[1] % 2 == 1:
                    place(body, pose_at(w, DA + DB + DS * t))
        row1.text = _pre("along each side long and short sides alternate:  every pair is "
                         "(R + x)·Δφ + (R − x)·Δφ = 2R·Δφ")
        row3.text = _pre(f"front:  {N // 2}·(R + r)·Δφ + {N // 2}·(R − r)·Δφ  =  "
                         f"{N}·R·Δφ  =  2πR  =  {2 * math.pi * R:.4f}")
        _wait(3.0)
        # ---- thinner and thinner wedges: the cylinder πr² × 2πR
        _show([cyl, lab_pr] + rims, True)
        dim2.show(True)
        row1.text = _pre(f"the {N} wedges hold the torus's volume;  as they get thinner the "
                         "log straightens into the cylinder")
        row2.text = _pre(f"the middle line of the log:  {N} arcs R·Δφ = 2πR;  it strays from "
                         f"straight by R(1 − cos ½Δφ) = {R * (1 - math.cos(H)):.4f}, "
                         "which → 0 as Δφ → 0")
        row3.text = _pre(f"V  =  πr² · 2πR  =  2π²Rr²  =  {2 * math.pi ** 2 * R * r * r:.4f}")
        _wait(5.5)
        # ---- and back
        _show([cyl, lab_pr] + rims, False)
        dim2.show(False)
        row1.text = _pre(" ")
        row3.text = _pre(" ")
        for t in _frames(0.8):
            for body, cp, w in bodies:
                if w[1] % 2 == 1:
                    place(body, pose_at(w, DA + DB + DS * (1 - t)))
        for t in _frames(1.4):
            for body, cp, w in bodies:
                if w[1] % 2 == 1:
                    place(body, pose_at(w, DA + DB * (1 - t)))
        for t in _frames(1.4):
            for body, cp, w in bodies:
                if w[1] % 2 == 0:
                    place(body, pose_at(w, DA * (1 - t)))
        for t in _frames(0.6):
            for body, cp, w in bodies:
                place(body, torus_pose(w[0], GAP * (1 - _smooth(t))))
        _wait(0.6)


# ============================================================== I13

def _convex_apart(pa, ea, fa, pb, eb, fb, eps=1e-9):
    """Are two convex polyhedra disjoint (their interiors)? Given as vertex
    lists, edge direction lists and face normal lists: they are apart if
    and only if some face normal or cross product of edges separates them
    (the separating axis theorem)."""
    axes = list(fa) + list(fb)
    for d1 in ea:
        for d2 in eb:
            c = d1.cross(d2)
            if c.mag > 1e-9:
                axes.append(c.norm())
    for a in axes:
        a1 = [p.dot(a) for p in pa]
        b1 = [p.dot(a) for p in pb]
        if max(a1) <= min(b1) + eps or max(b1) <= min(a1) + eps:
            return True
    return False


def _poly_parts(faces):
    """Vertices, edge directions and face normals of a convex polyhedron
    given by its faces, for _convex_apart."""
    pts, edges, nrms = [], [], []
    for f in faces:
        for p in f:
            if all((p - q).mag > 1e-9 for q in pts):
                pts.append(p)
        for a, b in zip(f, f[1:] + f[:1]):
            edges.append((b - a).norm())
        nrms.append((f[1] - f[0]).cross(f[2] - f[0]).norm())
    return pts, edges, nrms


def _mesh(faces, col, opacity=1.0):
    """A closed polyhedron from planar faces (lists of points, either
    orientation): one compound whose reference point is the centroid of
    the vertices. Returns (compound, centroid, volume); the volume comes
    from the divergence theorem on the outward-oriented faces, so it
    checks that the faces really close up into the intended solid."""
    pts = [p for f in faces for p in f]
    g = sum(pts, V(0, 0, 0)) / len(pts)
    tris, vol = [], 0.0
    for f in faces:
        fc = sum(f, V(0, 0, 0)) / len(f)
        nrm = (f[1] - f[0]).cross(f[2] - f[0]).norm()
        if nrm.dot(fc - g) < 0:
            f = list(reversed(f))
            nrm = -nrm
        for i in range(1, len(f) - 1):
            p, q, r_ = f[0], f[i], f[i + 1]
            vol += (p - g).dot((q - g).cross(r_ - g)) / 6.0
            tris.append(triangle(vs=[vertex(pos=p, color=col, normal=nrm, opacity=opacity),
                                     vertex(pos=q, color=col, normal=nrm, opacity=opacity),
                                     vertex(pos=r_, color=col, normal=nrm, opacity=opacity)]))
    cp = compound(tris, origin=g)       # built in place: pos is g
    return cp, g, vol


def _tet_vol(a, b, c, d):
    return abs((b - a).dot((c - a).cross(d - a))) / 6.0


def scene_I13():
    """V = ⅓Bh: the planes through the centre of a cube and its edges cut
    it into six pyramids, one on each face, all meeting at the centre. A
    turn of the cube carries any face to any other, so the six are
    congruent: each is s³/6. Its base is the face, B = s², its height is
    half the edge, h = s/2: s³/6 = ⅓·s²·(s/2) = ⅓·B·h."""
    sc = new_canvas("V = ⅓·B·h:   a cube is six pyramids",
                    "from the centre, the six faces span six congruent pyramids of height s/2",
                    rng=2.35, forward=V(-0.48, -0.42, -0.77), centre=V(0.75, -0.1, 0))
    sc.fov = 0.45
    s_right, s_up = _screen_basis(sc)
    s = 1.9
    h = s / 2
    O = V(0, 0, 0)
    AX = [V(1, 0, 0), V(-1, 0, 0), V(0, 1, 0), V(0, -1, 0), V(0, 0, 1), V(0, 0, -1)]

    def face(nv):
        """The face of the cube with outward normal nv, as a cyclic square."""
        a = V(nv.y, nv.z, nv.x)                  # two directions in the face
        b = nv.cross(a)
        c = nv * h
        return [c + (a * sa_ + b * sb_) * h for sa_, sb_ in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

    pyr_faces = []
    for nv in AX:
        base = face(nv)
        pyr_faces.append([base] + [[O, base[i], base[(i + 1) % 4]] for i in range(4)])

    # ---- checks: each pyramid is ⅓·s²·(s/2); together they fill the cube
    # once; a turn of the cube carries the first to every other
    for fs in pyr_faces:
        vol = sum(_tet_vol(O, f[0], f[1], f[2]) + _tet_vol(O, f[0], f[2], f[3])
                  for f in fs[:1])
        _check(abs(vol - s ** 3 / 6) < 1e-12 and abs(vol - s * s * h / 3) < 1e-12,
               "each pyramid is s³/6 = ⅓·s²·(s/2)")
    n_grid = 12
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                p = V(-h + s * (i + 0.5) / n_grid, -h + s * (j + 0.37) / n_grid,
                      -h + s * (k + 0.61) / n_grid)
                inside = 0
                for nv in AX:
                    # in the pyramid on face nv: beyond the four planes through
                    # O and the edges of that face
                    t = p.dot(nv)
                    others = [e for e in AX if abs(e.dot(nv)) < 0.5]
                    inside += all(t > p.dot(e) + 1e-12 for e in others)
                _check(inside == 1, "the six pyramids fill the cube, each point once")
    for k, nv in enumerate(AX):
        turn = None
        for ax_, ang in ((V(0, 0, 1), 0.0), (V(0, 0, 1), math.pi), (V(0, 0, 1), math.pi / 2),
                         (V(0, 0, 1), -math.pi / 2), (V(0, 1, 0), math.pi / 2),
                         (V(0, 1, 0), -math.pi / 2)):
            T = _AxisTurn(ax_, ang)
            if (T.apply(1, V(1, 0, 0)) - nv).mag < 1e-9:
                turn = T
                break
        _check(turn is not None and
               {tuple(round(c, 9) for c in _vt(turn.apply(1, p))) for p in pyr_faces[0][0]} ==
               {tuple(round(c, 9) for c in _vt(p)) for p in pyr_faces[k][0]},
               "a turn of the cube carries one pyramid onto another")

    # ---- the six pyramids, coloured by axis (opposite ones alike), each with
    # its eight edges drawn as lines that move with it
    COLS = [C_A, C_A, C_C, C_C, C_B, C_B]
    bodies = []
    for fs, col, nv in zip(pyr_faces, COLS, AX):
        cp, g, vol = _mesh(fs, col)
        _check(abs(vol - s ** 3 / 6) < 1e-9, "the drawn pyramid closes up: s³/6")
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        base = fs[0]
        segs = [(base[i], base[(i + 1) % 4]) for i in range(4)] + [(O, b_) for b_ in base]
        edges = [curve(pos=[a_, b_], radius=0.011, color=C_HL * 0.9) for a_, b_ in segs]
        bodies.append(dict(body=body, cp=cp, g=g, nv=nv, segs=segs, edges=edges))
    corners = [V(x, y, z) * h for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a_, b_ in itertools.combinations(corners, 2):
        if abs((a_ - b_).mag - s) < 1e-9:
            curve(pos=[a_, b_], radius=0.009, color=C_HL * 0.55)
    dotO = sphere(pos=O, radius=0.05, color=C_HL, visible=False)
    labO = _lab(O + s_right * 0.16 + s_up * 0.12, "O", C_HL, back=True, visible=False)

    def pose(k, turn, s_, P):
        b = bodies[k]
        b["body"].pose(turn, s_, P)
        for cv, (a_, b_) in zip(b["edges"], b["segs"]):
            cv.modify(0, pos=P + turn.apply(s_, a_ - b["g"]))
            cv.modify(1, pos=P + turn.apply(s_, b_ - b["g"]))

    EXP = 0.75                                   # how far the pyramids part
    k1 = 0                                       # the one singled out: on the face +x
    g1, n1 = bodies[k1]["g"], bodies[k1]["nv"]
    # it leaves to the right and stands on its base: a quarter-turn about z
    # takes its axis +x to −y
    TURN1 = _AxisTurn(V(0, 0, 1), -math.pi / 2)
    _check((TURN1.apply(1, V(1, 0, 0)) - V(0, -1, 0)).mag < 1e-12, "it stands on its base")
    APEX_T = V(2.65, 0, 0)                       # where its apex goes: base on the floor

    def stand_pose(u):
        """Centroid position and turn fraction, u from 0 (parted) to 1."""
        P0 = g1 + n1 * EXP
        P1 = APEX_T - TURN1.apply(1, O - g1)
        return P0 + (P1 - P0) * u, u

    # checks on the standing pose: base horizontal at the cube's floor, apex
    # straight above its centre at height s/2
    Pst, _ = stand_pose(1.0)
    base_w = [Pst + TURN1.apply(1, p - g1) for p in pyr_faces[k1][0]]
    apex_w = Pst + TURN1.apply(1, O - g1)
    bc = sum(base_w, V(0, 0, 0)) / 4
    _check(all(abs(p.y + h) < 1e-12 for p in base_w) and abs((apex_w - bc).mag - h) < 1e-12 and
           (apex_w - bc).norm().dot(V(0, 1, 0)) > 1 - 1e-12,
           "standing: a square base on the floor, the apex s/2 above its centre")

    # it goes without touching the others, which wait in their places, and
    # the parted pyramids keep apart
    def parted(k):
        return _poly_parts([[p + bodies[k]["nv"] * EXP for p in f] for f in pyr_faces[k]])
    others = [parted(k) for k in range(6) if k != k1]
    for u in [i / 60 for i in range(1, 61)]:
        P, s_ = stand_pose(_smooth(u))
        me = _poly_parts([[P + TURN1.apply(s_, p - g1) for p in f] for f in pyr_faces[k1]])
        for ot in others:
            _check(_convex_apart(*me, *ot), "the singled-out pyramid leaves freely")
    for k in range(6):
        for k2 in range(k + 1, 6):
            _check(_convex_apart(*parted(k), *parted(k2)), "the parted pyramids are apart")

    # its dimensions: two edges of the base (front and right) and the height
    xL, xR_, zB, zF = bc.x - h, bc.x + h, bc.z - h, bc.z + h
    yb = -h
    dim_a = _Dim(V(xL, yb, zF), V(xR_, yb, zF), V(0, -0.06, 0.12), "s", visible=False,
                 lab_gap=2.6)
    dim_b = _Dim(V(xR_, yb, zF), V(xR_, yb, zB), V(0.12, -0.06, 0), "s", visible=False,
                 lab_gap=2.6)
    dim_h = _Dim(V(xR_ + 0.3, yb, zB), V(xR_ + 0.3, yb + h, zB), V(0.08, 0, 0), "s/2",
                 visible=False, lab_gap=3.6)
    h_line = curve(pos=[bc, apex_w], radius=0.012, color=C_HL, visible=False)
    h_dot = sphere(pos=bc, radius=0.035, color=C_HL, visible=False)

    def place_all(d):
        for k, b in enumerate(bodies):
            pose(k, _IDENT, 0, b["g"] + b["nv"] * d)

    def look(op, dim):
        for k, b in enumerate(bodies):
            b["cp"].opacity = op if k != k1 or not dim else 0.55
            b["cp"].color = V(1, 1, 1) if (not dim or k == k1) else V(0.45, 0.45, 0.45)
            for cv in b["edges"]:
                cv.color = C_HL * (0.9 if (not dim or k == k1) else 0.35)

    def show_dims(flag):
        for dm in (dim_a, dim_b, dim_h):
            dm.show(flag)
        h_line.visible = h_dot.visible = flag

    row1, row2, row3 = _rows(sc, 3)
    _ready(sc)
    while True:
        # ---- the cube; the lines from its centre to the corners
        place_all(0)
        look(0.45, False)
        _show([dotO, labO], True)
        show_dims(False)
        row1.text = _pre("a cube of edge s;  the lines from its centre O to the corners")
        row2.text = row3.text = _pre(" ")
        _wait(2.2)
        # ---- six pyramids, one on each face, apex at the centre
        row1.text = _pre("cut through O and the edges:  six pyramids, one on each face, "
                         "apex at O")
        _show([dotO, labO], False)
        for t in _frames(1.6):
            place_all(EXP * _smooth(t))
            look(0.45 + 0.25 * t, False)
        row2.text = _pre("a turn of the cube carries any face onto any other:  "
                         "the six pyramids are congruent,  6·V = s³")
        _wait(2.0)
        # ---- one of them stands on its base
        look(0.7, True)
        row1.text = _pre("one of them, stood on its base")
        for t in _frames(1.8):
            P, s_ = stand_pose(_smooth(t))
            pose(k1, TURN1, s_, P)
        show_dims(True)
        row1.text = _pre("its base is a face, B = s²;  its height is half the edge, h = s/2")
        row3.text = _pre(f"V  =  s³/6  =  ⅓ · s² · (s/2)  =  ⅓·B·h        "
                         f"(s = 1:  1/6 = {1 / 6:.4f})")
        _wait(4.5)
        show_dims(False)
        row1.text = _pre("back into the cube")
        for t in _frames(1.4):
            P, s_ = stand_pose(1 - _smooth(t))
            pose(k1, TURN1, s_, P)
        look(0.7, False)
        for t in _frames(1.2):
            place_all(EXP * (1 - _smooth(t)))
            look(0.7 - 0.25 * t, False)
        _wait(0.8)


# ============================================================== H21

def scene_H21():
    """dV/dr = 4πr²: grow the ball of radius r by dr. The new layer is a
    shell; cut along a grid it falls into thin tiles, each standing on a
    patch a of the ball and dr thick. A tile spreads outward, so it holds
    more than a·dr and less than a·(1 + dr/r)²·dr, its outer face; added
    up, 4πr²·dr < ΔV < 4π(r + dr)²·dr. Divide by dr and let dr shrink:
    ΔV/dr is pinched onto 4πr², the surface."""
    sc = new_canvas("dV/dr  =  4πr²",
                    "grow the ball by dr:  the new shell is thin tiles, each a patch of the "
                    "surface times dr",
                    rng=2.25, forward=V(-0.38, -0.36, -0.85), centre=V(0.25, 0.05, 0))
    sc.fov = 0.45
    s_right, s_up = _screen_basis(sc)
    r = 1.2
    DR = 0.36                                    # the first, thick shell
    NLAT, NLON = 6, 12

    def pt(ph, la, rho):
        return V(math.cos(la) * math.sin(ph), math.sin(la), math.cos(la) * math.cos(ph)) * rho

    def e_ph(ph):
        return V(math.cos(ph), 0, -math.sin(ph))

    def e_la(ph, la):
        return V(-math.sin(la) * math.sin(ph), math.cos(la), -math.sin(la) * math.cos(ph))

    LA = [-math.pi / 2 + math.pi * i / NLAT for i in range(NLAT + 1)]
    PH = [2 * math.pi * k / NLON for k in range(NLON + 1)]
    tiles = [(PH[k], PH[k + 1], LA[i], LA[i + 1]) for i in range(NLAT) for k in range(NLON)]

    def patch(t):
        """Area of the tile's inner face over r²: Δφ·(sin λ₂ − sin λ₁)."""
        return (t[1] - t[0]) * (math.sin(t[3]) - math.sin(t[2]))

    def tile_vol(t, rho0, rho1):
        return patch(t) * (rho1 ** 3 - rho0 ** 3) / 3

    # ---- checks: the patches cover the sphere; every tile is squeezed
    # between its inner face and its outer face times its thickness; the
    # tiles make up the shell
    _check(abs(sum(patch(t) for t in tiles) * r * r - 4 * math.pi * r * r) < 1e-12,
           "the patches cover the sphere: Σa = 4πr²")
    for dr in (DR, 0.1, 0.01):
        for t in tiles:
            a = patch(t) * r * r
            v = tile_vol(t, r, r + dr)
            _check(a * dr < v < a * (1 + dr / r) ** 2 * dr, "a·dr < tile < a·(1 + dr/r)²·dr")
        _check(abs(sum(tile_vol(t, r, r + dr) for t in tiles) -
                   4 / 3 * math.pi * ((r + dr) ** 3 - r ** 3)) < 1e-12, "the tiles make the shell")
    # a tile measured slice by slice (the layer at ρ is a·(ρ/r)²), no formula
    t0 = tiles[2 * NLON + 1]
    n_s = 2000
    v_sl = sum(patch(t0) * (r + DR * (i + 0.5) / n_s) ** 2 * DR / n_s for i in range(n_s))
    _check(abs(v_sl - tile_vol(t0, r, r + DR)) < 1e-7, "the tile, slice by slice")

    # ---- the ball, the shell (see-through) and the grid that cuts it
    sphere(pos=V(0, 0, 0), radius=r, color=C_A, opacity=0.8)          # the ball
    shell = sphere(pos=V(0, 0, 0), radius=r, color=C_C, opacity=0.35, visible=False)
    GRIDC = V(0.98, 0.72, 0.42)
    lat_rings = [ring(pos=V(0, 0, 0), axis=V(0, 1, 0), radius=r, thickness=0.007, color=GRIDC,
                      visible=False) for _ in LA[1:-1]]
    mer_rings = [ring(pos=V(0, 0, 0), axis=e_ph(PH[k]), radius=r, thickness=0.007, color=GRIDC,
                      visible=False) for k in range(NLON // 2)]

    def set_shell(dr):
        rho = r + dr
        shell.radius = rho + 0.002
        for rg, la in zip(lat_rings, LA[1:-1]):
            rg.pos = V(0, rho * math.sin(la), 0)
            rg.radius = rho * math.cos(la) + 0.004
        for rg in mer_rings:
            rg.radius = rho + 0.004

    # ---- three tiles lifted out: vertices we move
    # the first (named) tile near the right rim, seen from the side so that
    # both its face and its thickness show; two more upper and lower left
    want = [(s_right * 0.9 + s_up * 0.12 - _VIEW * 0.42).norm(),
            (-s_right * 0.55 + s_up * 0.62 - _VIEW * 0.56).norm(),
            (-s_right * 0.35 - s_up * 0.62 - _VIEW * 0.70).norm()]

    def centre_dir(t):
        return pt((t[0] + t[1]) / 2, (t[2] + t[3]) / 2, 1).norm()

    chosen = []
    for w_ in want:                              # (not the three-cornered tiles at the poles)
        best = max((t for t in tiles if -math.pi / 2 < t[2] - 1e-9 and t[3] + 1e-9 < math.pi / 2
                    and t not in chosen), key=lambda t: centre_dir(t).dot(w_))
        chosen.append(best)
    _check(len(set(chosen)) == 3, "three different tiles")
    NS = 3

    def tile_points(t, rho0, rho1, lift):
        """Positions (in a fixed order) of the tile's vertices, and normals."""
        ph0, ph1, la0, la1 = t
        sh = centre_dir(t) * lift
        G = [(ph0 + (ph1 - ph0) * i / NS, la0 + (la1 - la0) * j / NS)
             for j in range(NS + 1) for i in range(NS + 1)]
        out = []
        for (ph, la) in G:                       # outer face
            out.append((pt(ph, la, rho1) + sh, pt(ph, la, 1)))
        for (ph, la) in G:                       # inner face
            out.append((pt(ph, la, rho0) + sh, -pt(ph, la, 1)))
        for ph, sgn in ((ph0, -1), (ph1, 1)):    # the two meridian sides
            for j in range(NS + 1):
                la = la0 + (la1 - la0) * j / NS
                for rho in (rho0, rho1):
                    out.append((pt(ph, la, rho) + sh, e_ph(ph) * sgn))
        for la, sgn in ((la0, -1), (la1, 1)):    # the two cone sides
            for i in range(NS + 1):
                ph = ph0 + (ph1 - ph0) * i / NS
                for rho in (rho0, rho1):
                    out.append((pt(ph, la, rho) + sh, e_la(ph, la) * sgn))
        return out

    def tile_tris(nv):
        """Index triples, matching tile_points' order."""
        tr = []
        m = NS + 1
        for base in (0, m * m):
            for j in range(NS):
                for i in range(NS):
                    a, b, c, d = (base + j * m + i, base + j * m + i + 1,
                                  base + (j + 1) * m + i + 1, base + (j + 1) * m + i)
                    tr += [(a, b, c), (a, c, d)]
        off = 2 * m * m
        for side in range(4):
            for q in range(NS):
                a = off + side * 2 * m + 2 * q
                tr += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
        return tr

    TCOL = [C_C * 1.12, C_C, C_C]
    lifted = []
    for t, col in zip(chosen, TCOL):
        pts0 = tile_points(t, r, r + DR, 0)
        vs = [vertex(pos=p, normal=nr, color=col) for p, nr in pts0]
        trs = [triangle(vs=[vs[a], vs[b], vs[c]], visible=False) for a, b, c in tile_tris(len(vs))]
        lifted.append(dict(t=t, vs=vs, trs=trs))

    def set_tile(d, dr, lift):
        for v, (p, nr) in zip(d["vs"], tile_points(d["t"], r, r + dr, lift)):
            v.pos = p
            v.normal = nr

    # the tile's mesh closes up and holds what the formula says
    for d in lifted:
        P_ = [p for p, _ in tile_points(d["t"], r, r + DR, 0)]
        g = sum(P_, V(0, 0, 0)) / len(P_)
        vol = 0.0
        for a, b, c in tile_tris(len(P_)):
            pa, pb, pc = P_[a], P_[b], P_[c]
            nrm = (pb - pa).cross(pc - pa)
            # outward: away from the tile's centre
            sg = 1 if nrm.dot((pa + pb + pc) / 3 - g) > 0 else -1
            vol += sg * (pa - g).dot((pb - g).cross(pc - g)) / 6
        _check(abs(vol / tile_vol(d["t"], r, r + DR) - 1) < 0.02,
               "the drawn tile holds what its patch times dr says")

    # the patch under the first tile, outlined on the ball, and labels
    tA = chosen[0]
    bnd = ([(tA[0] + (tA[1] - tA[0]) * i / 12, tA[2]) for i in range(13)] +
           [(tA[1], tA[2] + (tA[3] - tA[2]) * j / 12) for j in range(1, 13)] +
           [(tA[1] - (tA[1] - tA[0]) * i / 12, tA[3]) for i in range(1, 13)] +
           [(tA[0], tA[3] - (tA[3] - tA[2]) * j / 12) for j in range(1, 13)])
    outline = curve(pos=[pt(ph, la, r * 1.004) for ph, la in bnd], radius=0.009,
                    color=C_HL, visible=False)
    lab_a = _lab(centre_dir(tA) * r * 1.01, "a", C_HL, back=True, visible=False)
    lab_t = _lab(V(0, 0, 0), "a·dr", C_HL, back=True, visible=False)
    LIFT = 0.6
    # dr, beside the shell on the right of the screen
    rdir = (s_right * 0.55 - s_up * 0.835).norm()      # to the lower right
    across = (s_right * 0.835 + s_up * 0.55).norm()
    dim_dr = _Dim(rdir * r, rdir * (r + DR), across * 0.09, "dr", visible=False, lab_gap=2.8)
    rad = curve(pos=[V(0, 0, 0), rdir * r], radius=0.012, color=C_HL, visible=False)
    lab_r = _lab(rdir * (r * 0.5) + across * 0.15, "r", C_HL, back=True, visible=False)

    def labels_for(dr, lift):
        lab_t.pos = centre_dir(tA) * (r + dr + lift + 0.02) + s_up * 0.02

    def set_dr(dr):
        """The dimension dr at the rim, its name outside the shell."""
        dim_dr.set(rdir * r, rdir * (r + dr))
        dim_dr.lab.pos = rdir * (r + dr + 0.2) + across * 0.05

    row1, row2, row3 = _rows(sc, 3)

    def readout(dr):
        dV = 4 / 3 * math.pi * ((1 + dr) ** 3 - 1)            # r = 1
        row3.text = _pre(f"dr = {dr:.3f}:    4πr² = {4 * math.pi:.4f}  <  ΔV/dr = {dV / dr:.4f}"
                         f"  <  4π(r + dr)² = {4 * math.pi * (1 + dr) ** 2:.4f}      (r = 1)")

    _ready(sc)
    while True:
        # ---- the ball; it grows by dr
        shell.visible = False
        _show(lat_rings + mer_rings + [outline, lab_a, lab_t], False)
        for d in lifted:
            _show(d["trs"], False)
        _show([rad, lab_r], True)
        dim_dr.show(False)
        row1.text = _pre("a ball of radius r")
        row2.text = row3.text = _pre(" ")
        _wait(1.5)
        row1.text = _pre("grow it by dr:  it gains a shell, ΔV = V(r + dr) − V(r)")
        shell.visible = True
        _show(lat_rings + mer_rings, True)
        for t in _frames(1.6):
            set_shell(DR * _smooth(t))
        set_dr(DR)
        dim_dr.show(True)
        _wait(1.0)
        # ---- cut along the grid: tiles, three of them lifted out
        row1.text = _pre("cut along the grid, the shell falls into thin tiles:  each stands "
                         "on a patch a of the ball and is dr thick")
        _show([rad, lab_r], False)
        for d in lifted:
            set_tile(d, DR, 0)
            _show(d["trs"], True)
        for t in _frames(1.4):
            for d in lifted:
                set_tile(d, DR, LIFT * _smooth(t))
        _show([outline, lab_a, lab_t], True)
        labels_for(DR, LIFT)
        row2.text = _pre("a tile spreads outwards:  a·dr  <  tile  <  a·(1 + dr/r)²·dr;   "
                         "all of them:  4πr²·dr  <  ΔV  <  4π(r + dr)²·dr")
        readout(DR / r)
        _wait(3.6)
        # ---- thinner and thinner: ΔV/dr is pinched onto 4πr²
        row1.text = _pre("divide by dr and let dr shrink:  ΔV/dr is pinched between 4πr² "
                         "and 4π(r + dr)²")
        dr_end = 0.012
        for t in _frames(5.0):
            dr = DR + (dr_end - DR) * _smooth(t)
            set_shell(dr)
            set_dr(dr)
            for d in lifted:
                set_tile(d, dr, LIFT)
            labels_for(dr, LIFT)
            lab_t.visible = dr > 0.05
            if int(round(t * 200)) % 3 == 0:
                readout(dr / r)
        readout(dr_end / r)
        row1.text = _pre("the ball grows by its surface:  dV/dr  =  4πr²")
        _wait(4.5)
        # ---- back to the start
        for t in _frames(0.8):
            for d in lifted:
                set_tile(d, dr_end, LIFT * (1 - _smooth(t)))


# ============================================================== I14

def _in_tet(p, t, eps=1e-12):
    """Is p strictly inside the tetrahedron t (four points)?"""
    a, b, c, d = t
    M = [b - a, c - a, d - a]
    det = M[0].dot(M[1].cross(M[2]))
    q = p - a
    l1 = q.dot(M[1].cross(M[2])) / det
    l2 = M[0].dot(q.cross(M[2])) / det
    l3 = M[0].dot(M[1].cross(q)) / det
    return min(l1, l2, l3, 1 - l1 - l2 - l3) > eps


def scene_I14():
    """V = ⅓Bh: the planes AEF and ABF cut the prism
    ABC–DEF into T₁ = ABCF, T₂ = ABEF, T₃ = ADEF. T₁ and T₂ together are
    the pyramid on the side BCFE with apex A, and the plane ABF halves it
    through its apex and the diagonal BF: equal halves of the base, the
    same apex, so T₁ = T₂. T₂ and T₃ together are the pyramid on the side
    ABED with apex F, halved through the diagonal AE: T₂ = T₃. So each is ⅓
    of the prism, and T₁, on the base ABC with its apex on the top, is
    ⅓·B·h."""
    sc = new_canvas("V = ⅓·B·h:   a prism is three equal tetrahedra",
                    "two cuts through a corner and the diagonals of two sides;  each pair "
                    "shares an apex and stands on the two halves of one side",
                    rng=2.25, forward=V(-0.22, -0.42, -0.88), centre=V(0.1, -0.05, 0))
    sc.fov = 0.45
    s_right, s_up = _screen_basis(sc)
    hh = 1.9
    A = V(-1.35, -hh / 2, -0.35)
    B = V(0.15, -hh / 2, 0.95)
    C = V(1.25, -hh / 2, -0.55)
    up = V(0, hh, 0)
    D, E, F = A + up, B + up, C + up
    PT = dict(A=A, B=B, C=C, D=D, E=E, F=F)
    TETS = [(A, B, C, F), (A, B, E, F), (A, D, E, F)]
    NAMES = ["T₁", "T₂", "T₃"]

    def tvol(t):
        return _tet_vol(*t)

    area_B = (B - A).cross(C - A).mag / 2
    prism = area_B * hh

    # ---- checks: the three tetrahedra fill the prism, each point once;
    # the pairs stand on the halves of a side, with a common apex; all equal
    _check(abs(sum(tvol(t) for t in TETS) - prism) < 1e-12, "the three add up to the prism")
    n_g = 14
    for i in range(n_g):
        for j in range(n_g):
            for k in range(n_g):
                u_, v_ = (i + 0.31) / n_g, (j + 0.57) / n_g
                if u_ + v_ >= 1:
                    continue
                p = A + (B - A) * u_ + (C - A) * v_ + up * ((k + 0.43) / n_g)
                _check(sum(_in_tet(p, t) for t in TETS) == 1, "each point of the prism in one piece")

    def plane_dist(p, a, b, c):
        nrm = (b - a).cross(c - a).norm()
        return abs((p - a).dot(nrm))

    # T₁ = A·BCF, T₂ = A·BEF: halves of the parallelogram BCFE, apex A
    _check((B + F - C - E).mag < 1e-12 or (C - B - (F - E)).mag < 1e-12,
           "BCFE is a parallelogram")
    _check(abs((C - B).cross(F - B).mag - (F - B).cross(E - B).mag) < 1e-12 and
           plane_dist(E, B, C, F) < 1e-12, "BF halves it")
    # T₂ = F·ABE, T₃ = F·ADE: halves of the parallelogram ABED, apex F
    _check(abs((B - A).cross(E - A).mag - (E - A).cross(D - A).mag) < 1e-12 and
           plane_dist(D, A, B, E) < 1e-12, "AE halves ABED")
    _check(abs(tvol(TETS[0]) - tvol(TETS[1])) < 1e-12 and
           abs(tvol(TETS[1]) - tvol(TETS[2])) < 1e-12, "T₁ = T₂ = T₃")
    _check(abs(tvol(TETS[0]) - area_B * hh / 3) < 1e-12, "T₁ = ⅓·B·h")

    # ---- the pieces, each with its edges
    COLS = [C_C, C_B, C_F]
    pieces = []
    for t, col in zip(TETS, COLS):
        fs = [[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]]
        cp, g, vol = _mesh(fs, col)
        _check(abs(vol - prism / 3) < 1e-9, "each drawn piece closes up: ⅓ of the prism")
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        segs = list(itertools.combinations(t, 2))
        edges = [curve(pos=[a_, b_], radius=0.011, color=C_HL * 0.9) for a_, b_ in segs]
        pieces.append(dict(body=body, cp=cp, g=g, segs=segs, edges=edges, faces=fs))

    def put(k, d):
        """Piece k moved by d from its place."""
        pc = pieces[k]
        pc["body"].pose(_IDENT, 0, pc["g"] + d)
        for cv, (a_, b_) in zip(pc["edges"], pc["segs"]):
            cv.modify(0, pos=a_ + d)
            cv.modify(1, pos=b_ + d)

    def away(a, b, c, toward):
        """Unit normal of the plane abc, on the side of `toward`."""
        nrm = (b - a).cross(c - a).norm()
        return nrm if nrm.dot(toward - a) > 0 else -nrm

    n1 = away(A, B, F, C)                        # T₁ leaves the cut ABF towards C
    n3 = away(A, E, F, D)                        # T₃ leaves the cut AEF towards D
    OUT1, OUT3 = 1.0, 1.0
    # the moves keep the pieces apart
    parts = [_poly_parts(pc["faces"]) for pc in pieces]

    def moved(k, d):
        return _poly_parts([[p + d for p in f] for f in pieces[k]["faces"]])

    for u in [i / 30 for i in range(1, 31)]:
        _check(_convex_apart(*moved(0, n1 * OUT1 * u), *parts[1]) and
               _convex_apart(*moved(2, n3 * OUT3 * u), *parts[1]) and
               _convex_apart(*moved(0, n1 * OUT1 * u), *moved(2, n3 * OUT3 * u)),
               "the pieces slide apart freely")

    # ---- the marks: the two sides with their diagonals, the common apexes,
    # and the vertex names, pushed away from the middle of the figure on the
    # screen
    centre = (A + B + C + D + E + F) / 6
    labs = {}
    for k, p in PT.items():
        o = p - centre
        o = V(o.dot(s_right), o.dot(s_up), 0).norm()
        labs[k] = _lab(p + (s_right * o.x + s_up * o.y) * 0.2, k, C_HL, back=True,
                       visible=False)

    def side_marks(p, q, r_, s_):
        """Outline of the side pqrs and its diagonal pr."""
        return [curve(pos=[p, q, r_, s_, p], radius=0.02, color=C_HL, visible=False),
                curve(pos=[p, r_], radius=0.016, color=C_HL, visible=False)]

    side1 = side_marks(B, C, F, E)               # diagonal B–F
    side2 = side_marks(A, B, E, D)               # diagonal A–E

    dotA = sphere(pos=A, radius=0.06, color=C_HL, visible=False)
    dotF = sphere(pos=F, radius=0.06, color=C_HL, visible=False)
    labT = [_lab(V(0, 0, 0), nm, C_HL, back=True, visible=False) for nm in NAMES]

    def name_pieces(d1, d3, flag):
        for k, (lb, d) in enumerate(zip(labT, (d1, V(0, 0, 0), d3))):
            lb.pos = pieces[k]["g"] + d
            lb.visible = flag

    def see(op):
        for pc in pieces:
            pc["cp"].opacity = op

    row1, row2, row3 = _rows(sc, 3)
    _ready(sc)
    while True:
        for k in range(3):
            put(k, V(0, 0, 0))
        see(0.55)
        _show(list(labs.values()), True)
        _show(side1 + side2 + [dotA, dotF], False)
        name_pieces(V(0, 0, 0), V(0, 0, 0), False)
        row1.text = _pre("a prism on the triangle ABC, its top DEF")
        row2.text = row3.text = _pre(" ")
        _wait(2.2)
        # ---- two cuts: through A, E, F and through A, B, F
        row1.text = _pre("cut it through A, E, F and through A, B, F:  three tetrahedra")
        row2.text = _pre("T₁ = ABCF  (orange),   T₂ = ABEF  (teal),   T₃ = ADEF  (purple)")
        see(0.8)
        _show(list(labs.values()), False)
        for t in _frames(1.4):
            put(0, n1 * (0.45 * _smooth(t)))
            put(2, n3 * (0.45 * _smooth(t)))
        name_pieces(n1 * 0.45, n3 * 0.45, True)
        _wait(1.6)
        # ---- T₁ and T₂: apex A, the halves of BCFE
        name_pieces(n1 * 0.45, n3 * 0.45, False)
        for t in _frames(1.0):
            put(0, n1 * (0.45 * (1 - _smooth(t))))
            put(2, n3 * (0.45 + (OUT3 - 0.45) * _smooth(t)))
        see(0.62)
        _show(side1 + [dotA, labs["A"], labs["B"], labs["C"], labs["E"], labs["F"]], True)
        row1.text = _pre("T₁ and T₂ have the same apex A, and their bases BCF and BEF are the "
                         "two halves of the side BCFE:  T₁ = T₂")
        _wait(3.4)
        _show(side1 + [dotA] + list(labs.values()), False)
        # ---- T₂ and T₃: apex F, the halves of ABED
        for t in _frames(1.2):
            put(0, n1 * (OUT1 * _smooth(t)))
            put(2, n3 * (OUT3 * (1 - _smooth(t))))
        _show(side2 + [dotF, labs["A"], labs["B"], labs["D"], labs["E"], labs["F"]], True)
        row1.text = _pre("T₂ and T₃ have the same apex F, and their bases ABE and ADE are the "
                         "two halves of the side ABED:  T₂ = T₃")
        _wait(3.4)
        _show(side2 + [dotF] + list(labs.values()), False)
        # ---- all three equal: each a third of the prism
        for t in _frames(1.0):
            put(2, n3 * (OUT3 * _smooth(t)))
        see(0.85)
        name_pieces(n1 * OUT1, n3 * OUT3, True)
        row1.text = _pre("so T₁ = T₂ = T₃:  each is a third of the prism;  T₁ stands on ABC "
                         "with its apex F on the top")
        row3.text = _pre(f"V(T₁)  =  ⅓ · B · h  =  ⅓ · {area_B:.4f} · {hh:g}  =  "
                         f"{area_B * hh / 3:.4f}        prism  B·h = {prism:.4f}")
        _wait(4.5)
        name_pieces(n1 * OUT1, n3 * OUT3, False)
        for t in _frames(1.3):
            put(0, n1 * (OUT1 * (1 - _smooth(t))))
            put(2, n3 * (OUT3 * (1 - _smooth(t))))
        _wait(0.6)


# ============================================================== I17

def scene_I17():
    """V = a³ − 4·a³/6 = ⅓a³: the diagonals of the faces through alternate
    corners of a cube are the edges of a regular tetrahedron. Cutting it
    out of the cube leaves the four other corners, each a tetrahedron with
    three right angles at the corner: its base is half of a face, ½a², and
    its height the edge a square to that face, so it is ⅓·½a²·a = a³/6."""
    sc = new_canvas("V = a³ − 4·a³/6 = ⅓·a³",
                    "the face diagonals through alternate corners of a cube make a regular "
                    "tetrahedron;  four corners, each ⅙ of the cube, are left",
                    rng=2.65, forward=V(-0.52, -0.40, -0.75), centre=V(0.55, -0.05, -0.38))
    sc.fov = 0.45
    s_right, s_up = _screen_basis(sc)
    a = 1.9
    h = a / 2
    signs = list(itertools.product((-1, 1), repeat=3))
    even = [V(*sg) * h for sg in signs if sg[0] * sg[1] * sg[2] > 0]
    odd = [V(*sg) * h for sg in signs if sg[0] * sg[1] * sg[2] < 0]

    def nbrs(p):
        return [q for q in even if abs((q - p).mag - a) < 1e-9]

    corners = [(p, nbrs(p)) for p in odd]          # (right-angle corner, its 3 neighbours)
    reg = even

    # ---- checks: the tetrahedron is regular (edge a√2); each corner has
    # three right angles and is a³/6; the five pieces fill the cube
    _check(all(abs((p - q).mag - a * math.sqrt(2)) < 1e-12
               for p, q in itertools.combinations(reg, 2)), "a regular tetrahedron of edge a√2")
    for c, nb in corners:
        _check(len(nb) == 3 and all(abs((x - c).dot(y - c)) < 1e-12
                                    for x, y in itertools.combinations(nb, 2)),
               "each corner: three edges a at right angles")
        _check(abs(_tet_vol(c, *nb) - a ** 3 / 6) < 1e-12, "each corner is a³/6")
    _check(abs(_tet_vol(*reg) - a ** 3 / 3) < 1e-12, "the regular tetrahedron is a³/3")
    pieces_t = [tuple(reg)] + [(c, *nb) for c, nb in corners]
    n_g = 12
    for i in range(n_g):
        for j in range(n_g):
            for k in range(n_g):
                p = V(-h + a * (i + 0.5) / n_g, -h + a * (j + 0.37) / n_g,
                      -h + a * (k + 0.61) / n_g)
                _check(sum(_in_tet(p, t) for t in pieces_t) == 1,
                       "the tetrahedron and the four corners fill the cube")

    # ---- the corner singled out later: the one on the floor nearest to us
    kc = 1 + max((i for i in range(4) if abs(corners[i][0].y + h) < 1e-12),
                 key=lambda i: -corners[i][0].dot(_VIEW))

    # ---- pieces: the regular tetrahedron and the four corners, with edges
    # (the singled-out corner orange, so the yellow marks stand out on it)
    rest = iter([C_B, C_E, C_D])
    COLC = [C_C if i == kc - 1 else next(rest) for i in range(4)]
    pieces = []
    for t, col in zip(pieces_t, [C_F] + COLC):
        fs = [[t[0], t[1], t[2]], [t[0], t[1], t[3]], [t[0], t[2], t[3]], [t[1], t[2], t[3]]]
        cp, g, vol = _mesh(fs, col)
        _check(abs(vol - _tet_vol(*t)) < 1e-9, "the drawn piece closes up")
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        segs = list(itertools.combinations(t, 2))
        edges = [curve(pos=[p, q], radius=0.011, color=C_HL * 0.9) for p, q in segs]
        pieces.append(dict(body=body, cp=cp, g=g, segs=segs, edges=edges, faces=fs, t=t))
    cube_c = [V(*sg) * h for sg in signs]
    for p, q in itertools.combinations(cube_c, 2):
        if abs((p - q).mag - a) < 1e-9:
            curve(pos=[p, q], radius=0.009, color=C_HL * 0.55)

    def put(k, d):
        pc = pieces[k]
        pc["body"].pose(_IDENT, 0, pc["g"] + d)
        for cv, (p, q) in zip(pc["edges"], pc["segs"]):
            cv.modify(0, pos=p + d)
            cv.modify(1, pos=q + d)

    def pose_kc(s_, P):
        """The singled-out corner turned by s_ of a half-turn about the
        vertical, its centroid at P."""
        pc = pieces[kc]
        pc["body"].pose(HALF, s_, P)
        for cv, (p, q) in zip(pc["edges"], pc["segs"]):
            cv.modify(0, pos=P + HALF.apply(s_, p - pc["g"]))
            cv.modify(1, pos=P + HALF.apply(s_, q - pc["g"]))

    OUT = 0.7                                    # the corners leave along the diagonals
    dirs = [V(0, 0, 0)] + [c.norm() for c, _ in corners]
    parts0 = _poly_parts(pieces[0]["faces"])
    for u in [i / 30 for i in range(1, 31)]:
        mv = [_poly_parts([[p + dirs[k] * OUT * u for p in f] for f in pieces[k]["faces"]])
              for k in range(1, 5)]
        for m_ in mv:
            _check(_convex_apart(*m_, *parts0), "a corner comes off freely")
        for m1, m2 in itertools.combinations(mv, 2):
            _check(_convex_apart(*m1, *m2), "the corners keep apart")

    # ---- the singled-out corner stands on half of the bottom face with its
    # right-angle edge upright; it leaves for a free spot in front, turning
    # half round on the way, so that its right angle is at the back and the
    # other half of the square, which it does not cover, lies in front
    c0, nb0 = corners[kc - 1]
    upright = [x for x in nb0 if abs((x - c0).norm().dot(V(0, 1, 0))) > 0.5]
    flat = [x for x in nb0 if x not in upright]
    _check(len(upright) == 1 and len(flat) == 2 and abs(c0.y + h) < 1e-12,
           "it stands on the floor with one edge upright")
    W = flat[0] + flat[1] - c0                   # the fourth corner of the bottom face
    _check(abs((W - c0).mag - a * math.sqrt(2)) < 1e-12 and
           abs((W - flat[0]).mag - a) < 1e-12 and abs((W - flat[1]).mag - a) < 1e-12,
           "its base is half of an a × a square")
    Mq = (c0 + W) / 2                            # centre of that square
    # where the square goes: right of the cube on the screen, a little nearer
    toward = V(-_VIEW.x, 0, -_VIEW.z).norm()
    SPOT = V(0, -h, 0) + s_right * 2.95 - toward * 0.35
    HALF = _AxisTurn(V(0, 1, 0), math.pi)
    gk = pieces[kc]["g"]

    def corner_pose(u):
        """(turn fraction, centroid) of the singled-out corner, u from 0
        (parted along its diagonal) to 1 (at the spot, turned half round):
        first straight out to the right, then to the spot while it turns."""
        P0 = gk + dirs[kc] * OUT
        Pw = P0 + s_right * 1.6
        P1 = SPOT + HALF.apply(1, gk - Mq)
        if u <= 0.5:
            return 0.0, P0 + (Pw - P0) * (2 * u)
        v = 2 * u - 1
        return v, Pw + (P1 - Pw) * v

    def at_spot(X):
        return SPOT + HALF.apply(1, X - Mq)

    # it travels without touching the others
    others = [_poly_parts([[p + dirs[k] * OUT for p in f] for f in pieces[k]["faces"]])
              for k in range(5) if k != kc]
    for u in [i / 80 for i in range(1, 81)]:
        s_, P = corner_pose(u)
        me = _poly_parts([[P + HALF.apply(s_, p - gk) for p in f] for f in pieces[kc]["faces"]])
        for ot in others:
            _check(_convex_apart(*me, *ot), "the corner goes to its spot freely")
    c0s, f0s, f1s, Ws, ups = (at_spot(X) for X in (c0, flat[0], flat[1], W, upright[0]))
    _check((ups - c0s).norm().dot(V(0, 1, 0)) > 1 - 1e-12 and
           (Ws - c0s).dot(_VIEW) < 0, "at the spot: upright, its right angle at the back")
    sq = curve(pos=[c0s, f0s, Ws, f1s, c0s], radius=0.011, color=C_HL, visible=False)
    diag = curve(pos=[f0s, f1s], radius=0.016, color=C_HL, visible=False)
    half_fill = triangle(vs=[vertex(pos=p + V(0, 0.004, 0), normal=V(0, 1, 0),
                                    color=V(0.55, 0.68, 0.95), opacity=0.55)
                             for p in (f0s, Ws, f1s)], visible=False)
    dim_1 = _Dim(Ws, f0s, V(0, -0.05, 0) + (Ws - f1s).norm() * 0.1, "a", visible=False,
                 lab_gap=2.6)
    dim_2 = _Dim(Ws, f1s, V(0, -0.05, 0) + (Ws - f0s).norm() * 0.1, "a", visible=False,
                 lab_gap=2.6)
    # its height: the apex is a above the floor, beside its rightmost corner
    rc = max((c0s, f0s, f1s, Ws), key=lambda X: X.dot(s_right)) + s_right * 0.32
    _check(abs(ups.y - c0s.y - a) < 1e-12, "the apex stands a above the floor")
    dim_h = _Dim(rc, rc + V(0, a, 0), s_right * 0.08, "a", visible=False, lab_gap=2.8)
    # one edge of the regular tetrahedron named
    e0, e1 = reg[0], reg[1]
    es = V((e1 - e0).dot(s_right), (e1 - e0).dot(s_up), 0).norm()
    off = V(-es.y, es.x, 0)                      # across the edge on the screen,
    mid = (e0 + e1) / 2                          # away from the middle of the cube
    if off.x * mid.dot(s_right) + off.y * mid.dot(s_up) < 0:
        off = -off
    lab_e = _lab(mid + (s_right * off.x + s_up * off.y) * 0.2, "a√2", C_HL, back=True,
                 visible=False)

    def see(op_reg, op_corner, dim_others=False):
        pieces[0]["cp"].opacity = op_reg
        for k in range(1, 5):
            pieces[k]["cp"].opacity = op_corner
            on = not dim_others or k == kc
            pieces[k]["cp"].color = V(1, 1, 1) if on else V(0.45, 0.45, 0.45)
            for cv in pieces[k]["edges"]:
                cv.color = C_HL * (0.9 if on else 0.35)

    def show_corner_marks(flag):
        _show([sq, diag, half_fill], flag)
        for dm in (dim_h, dim_1, dim_2):
            dm.show(flag)

    row1, row2, row3 = _rows(sc, 3)
    _ready(sc)
    while True:
        for k in range(5):
            put(k, V(0, 0, 0))
        see(0.5, 0.35)
        show_corner_marks(False)
        lab_e.visible = True
        row1.text = _pre("a cube of edge a:  the diagonals of its faces through alternate "
                         "corners make a regular tetrahedron, edge a√2")
        row2.text = row3.text = _pre(" ")
        _wait(2.6)
        # ---- cut off the four corners
        lab_e.visible = False
        row1.text = _pre("cut it out:  the four other corners of the cube come off")
        for t in _frames(1.8):
            for k in range(1, 5):
                put(k, dirs[k] * (OUT * _smooth(t)))
            see(0.5 + 0.5 * t, 0.35 + 0.5 * t)
        _wait(0.6)
        # ---- one corner: half a face times the edge
        see(1.0, 0.85, dim_others=True)
        row1.text = _pre("each corner stands on half of a face, ½·a²;  its third edge, square "
                         "to that face, is its height a")
        for t in _frames(1.6):
            s_, P = corner_pose(_smooth(t))
            pose_kc(s_, P)
        show_corner_marks(True)
        row2.text = _pre("a corner  =  ⅓ · ½a² · a  =  a³/6")
        _wait(4.2)
        show_corner_marks(False)
        for t in _frames(1.2):
            s_, P = corner_pose(1 - _smooth(t))
            pose_kc(s_, P)
        see(1.0, 0.85)
        row1.text = _pre("the regular tetrahedron is what is left of the cube")
        row3.text = _pre(f"V  =  a³ − 4·a³/6  =  ⅓·a³        (a = 1:  1 − 4/6 = {1 / 3:.4f})")
        _wait(4.0)
        for t in _frames(1.4):
            for k in range(1, 5):
                put(k, dirs[k] * (OUT * (1 - _smooth(t))))
            see(1.0 - 0.5 * t, 0.85 - 0.5 * t)
        _wait(0.6)
