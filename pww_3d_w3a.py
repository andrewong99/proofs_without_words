# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d_w3a.py — vpython scenes: Girard's theorem, Descartes' angle defect,
the twelve pentagons of a football, five points on a sphere and a slanted
slice of a cylinder (I19, I21, I25, K29, I24).

Each scene_<ID>() is found by name by pww_3d.py / the app. Every
construction shown here is rebuilt from its rule and checked before
anything is drawn: angles, areas, coverings, counts and tangencies are
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


def _rot_about_line(p, a, k, ang):
    """The point p turned by `ang` about the line through a with unit
    direction k (Rodrigues)."""
    v = p - a
    c, s = math.cos(ang), math.sin(ang)
    return a + v * c + k.cross(v) * s + k * (k.dot(v) * (1 - c))


def _screen_basis(sc):
    """Unit vectors to the right and up on the screen of canvas sc."""
    f = sc.forward.norm()
    right = f.cross(V(0, 1, 0)).norm()
    return right, right.cross(f).norm()


def _polar_angle(u, w, k):
    """Signed angle (about the unit axis k) that turns the part of u
    perpendicular to k onto the part of w perpendicular to k."""
    u = u - k * u.dot(k)
    w = w - k * w.dot(k)
    return math.atan2(k.dot(u.cross(w)), u.dot(w))


def _canvas(title, subtitle, rng, forward, centre, fov=0.45):
    """new_canvas, with the viewing direction set again once the canvas
    exists: vpython keeps a forward given to the constructor apart from the
    one it reports back (which stays (0, 0, −1)), and every screen direction
    below is worked out from sc.forward."""
    sc = new_canvas(title, subtitle, rng=rng, forward=forward, centre=centre)
    sc.up = V(0, 1, 0)
    sc.forward = forward
    sc.fov = fov
    _check((sc.forward.norm() - forward.norm()).mag < 1e-12, "the view direction is known")
    return sc


def _eye(sc):
    """Where the camera of canvas sc sits (range is the half-height of the
    view at the centre plane, seen under the angle fov)."""
    return sc.center - sc.forward.norm() * (sc.range / math.tan(sc.fov / 2))


def _frame_map(a1, b1, a2, b2):
    """The proper rotation taking the unit vector a1 to a2 and the
    half-plane through a1 towards b1 onto the one through a2 towards b2."""
    u1 = (b1 - a1 * a1.dot(b1)).norm()
    u2 = (b2 - a2 * a2.dot(b2)).norm()
    w1, w2 = a1.cross(u1), a2.cross(u2)

    def M(v):
        return a2 * v.dot(a1) + u2 * v.dot(u1) + w2 * v.dot(w1)
    return M


def _slerp(p, q, s):
    """The point a fraction s along the shorter great-circle arc from the
    unit vector p to the unit vector q."""
    om = math.acos(max(-1.0, min(1.0, p.dot(q))))
    if om < 1e-12:
        return V(p.x, p.y, p.z)
    return (p * math.sin((1 - s) * om) + q * math.sin(s * om)) / math.sin(om)


def _tangent(p, q):
    """Unit tangent at p of the great circle from p towards q."""
    return (q - p * p.dot(q)).norm()


def _fib_sphere(n):
    """n nearly evenly spread unit vectors (a Fibonacci lattice)."""
    ga = math.pi * (3 - math.sqrt(5))
    out = []
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        rr = math.sqrt(max(0.0, 1 - y * y))
        out.append(V(rr * math.cos(ga * i), y, rr * math.sin(ga * i)))
    return out


def _arc_sticks(pts, radius, col):
    """A polyline as thin cylinders (so that it can go into a compound)."""
    return [cylinder(pos=p, axis=q - p, radius=radius, color=col)
            for p, q in zip(pts, pts[1:]) if (q - p).mag > 1e-9]


# ============================================================== I19

def scene_I19():
    """Girard: a triangle of great-circle arcs with angles α, β, γ on a
    sphere of radius r. A great circle through a corner, turned about the
    diameter there, sweeps a lune and the antipodal lune; turned by π it
    sweeps the whole sphere, so turned by α it sweeps α/π of it: 4αr².
    The three pairs of lunes cover the sphere: every point once, except
    the triangle and its antipode, which lie in all three. So
    4αr² + 4βr² + 4γr² = 4πr² + 2·2A and A = (α + β + γ − π)·r². The
    sphere is shown twice: as it is, and turned over (its back, with the
    antipodal triangle in front), so that every lune is seen whole."""
    sc = _canvas("A = (α + β + γ − π)·r²   (Girard)",
                 "three lunes through the corners, with their antipodes, cover the sphere "
                 "once and the triangle and its antipode three times",
                 1.95, V(0, -math.sin(math.radians(14)), -math.cos(math.radians(14))),
                 V(0, 0, 0))
    s_right, s_up = _screen_basis(sc)
    RS = 1.18                                    # the sphere's radius (r)
    X0 = 1.62
    XL, XR = V(-X0, 0, 0), V(X0, 0, 0)
    FLIP = _AxisTurn(V(1, 0, 0), math.pi)        # turned over, about the screen's x-axis

    # ---- the triangle from its angles (the polar law of cosines)
    AL, BE, GA = math.radians(72), math.radians(84), math.radians(96)

    def side(x, y, z):
        return math.acos((math.cos(x) + math.cos(y) * math.cos(z)) / (math.sin(y) * math.sin(z)))
    sa_, sb_, sc_ = side(AL, BE, GA), side(BE, AL, GA), side(GA, AL, BE)
    A0 = V(0, 0, 1)
    B0 = V(math.sin(sc_), 0, math.cos(sc_))
    C0 = V(math.sin(sb_) * math.cos(AL), math.sin(sb_) * math.sin(AL), math.cos(sb_))
    # turned so that it faces the camera from the left copy, corner α at the top
    eye = _eye(sc)
    vL = (eye - XL).norm()                       # the middle of what the left copy shows
    g0 = (A0 + B0 + C0).norm()
    M = _frame_map(g0, A0, vL, vL + s_up)
    A, B, C = M(A0), M(B0), M(C0)
    VERT = (A, B, C)
    g = (A + B + C).norm()

    # ---- checks: the angles, the sides, the area (by the solid-angle
    # formula, which does not use Girard), and the views
    for X, Y, Z, ang in ((A, B, C, AL), (B, C, A, BE), (C, A, B, GA)):
        _check(abs(math.acos(_tangent(X, Y).dot(_tangent(X, Z))) - ang) < 1e-9,
               "the triangle has the angles α, β, γ")
    _check(abs((B - C).mag - 2 * math.sin(sa_ / 2)) < 1e-9, "its side opposite α")
    omega = 2 * math.atan2(abs(A.dot(B.cross(C))), 1 + A.dot(B) + B.dot(C) + C.dot(A))
    EXC = AL + BE + GA - math.pi
    _check(abs(omega - EXC) < 1e-9, "its area is (α + β + γ − π)·r²")
    _check(abs(g.dot(vL) - 1) < 1e-12, "the triangle faces the camera in the left copy")
    vR_local = FLIP.apply(-1, (eye - XR).norm())
    _check(abs((-g).dot(vR_local) - 1) < 1e-9,
           "the turned-over copy shows the antipodal triangle face on")
    _check(A.dot(s_up) > B.dot(s_up) and A.dot(s_up) > C.dot(s_up) and
           B.dot(s_right) < C.dot(s_right), "corner α at the top, β left, γ right")

    # the lune at each corner: between the two sides there, from the corner
    # to its antipode; e1 points along one side, e2 a quarter turn on
    lunes = []
    for X, Y, Z, ang in ((A, B, C, AL), (B, C, A, BE), (C, A, B, GA)):
        e1 = _tangent(X, Y)
        e2 = (_tangent(X, Z) - e1 * math.cos(ang)) / math.sin(ang)
        _check(abs(e2.mag - 1) < 1e-9 and abs(e2.dot(e1)) < 1e-9 and abs(e2.dot(X)) < 1e-9,
               "a right-angled frame at the corner")
        lunes.append(dict(X=X, e1=e1, e2=e2, ang=ang))

    def lune_pt(L, th, ps):
        """The point at arc th from the corner, on the half great circle
        leaving it at angle ps from the first side."""
        d = L["e1"] * math.cos(ps) + L["e2"] * math.sin(ps)
        return L["X"] * math.cos(th) + d * math.sin(th)

    def lune_normal(L, ps):
        return L["X"].cross(L["e1"] * math.cos(ps) + L["e2"] * math.sin(ps))

    circ = [A.cross(B).norm(), B.cross(C).norm(), C.cross(A).norm()]   # the three great circles
    t_side = [1 if g.dot(n) > 0 else -1 for n in circ]
    bound = [(0, 2), (0, 1), (1, 2)]             # the circles bounding the lune at A, B, C
    for L, Y, Z in zip(lunes, (B, C, A), (C, A, B)):
        n0, n1 = lune_normal(L, 0).norm(), lune_normal(L, L["ang"]).norm()
        _check(n0.cross(L["X"].cross(Y).norm()).mag < 1e-9 and
               n1.cross(L["X"].cross(Z).norm()).mag < 1e-9,
               "the turning circle starts on one side and ends on the other")

    # every point of the sphere, sorted by the six lunes: once, except the
    # triangle and its antipode, three times
    in_T = in_T2 = 0
    pts = _fib_sphere(6000)
    for x in pts:
        sd = [x.dot(n) * s for n, s in zip(circ, t_side)]
        if min(abs(v) for v in sd) < 1e-9:
            continue
        cnt = 0
        for i, j in bound:
            cnt += (sd[i] > 0 and sd[j] > 0) + (sd[i] < 0 and sd[j] < 0)
        tri = all(v > 0 for v in sd)
        anti = all(v < 0 for v in sd)
        in_T += tri
        in_T2 += anti
        _check(cnt == (3 if (tri or anti) else 1),
               "the six lunes cover the sphere once, the two triangles three times")
    _check(abs(in_T / len(pts) - EXC / (4 * math.pi)) < 0.01 and
           abs(in_T2 / len(pts) - EXC / (4 * math.pi)) < 0.01,
           "the share of the sphere in the triangle is (α + β + γ − π)/4π")
    # a lune's area: α/π of the sphere for the pair, measured on its mesh
    NTH = 48

    def lune_mesh(L, ps0, ps1, nps, rad):
        quads = []
        for i in range(NTH):
            for k in range(nps):
                th0, th1 = math.pi * i / NTH, math.pi * (i + 1) / NTH
                p0, p1 = ps0 + (ps1 - ps0) * k / nps, ps0 + (ps1 - ps0) * (k + 1) / nps
                quads.append([lune_pt(L, th0, p0) * rad, lune_pt(L, th1, p0) * rad,
                              lune_pt(L, th1, p1) * rad, lune_pt(L, th0, p1) * rad])
        return quads
    for L in lunes:
        nps = 24
        area = 0.0
        for ps0 in (0.0, math.pi):
            for q in lune_mesh(L, ps0, ps0 + L["ang"], nps, 1.0):
                area += (q[1] - q[0]).cross(q[2] - q[0]).mag / 2 + \
                    (q[2] - q[0]).cross(q[3] - q[0]).mag / 2
        _check(abs(area / (4 * L["ang"]) - 1) < 0.01, "a lune and its antipode: 4α·r²")
    _check(abs(4 * (AL + BE + GA) - (4 * math.pi + 4 * omega)) < 1e-9,
           "4αr² + 4βr² + 4γr² = 4πr² + 4A")

    # ---- drawing. Each copy of the sphere: the ball, both triangles (one
    # compound), three great circles, three lune pairs (fill and hatching)
    COL_BALL = V(0.20, 0.22, 0.30)
    COL_TRI = V(0.36, 0.36, 0.34)
    COL_LUNE = [C_C, V(0.20, 0.80, 0.70), V(0.72, 0.50, 1.0)]
    NAMES = ["α", "β", "γ"]
    NT = 24

    def tri_parts(sign):
        """Fill, outline and angle arcs of the triangle (sign 1) or of its
        antipode (sign −1), at the centred pose."""
        P = [X * sign for X in VERT]
        parts = []
        grid = {}
        for i in range(NT + 1):
            for j in range(NT + 1 - i):
                q = (P[0] * (NT - i - j) + P[1] * i + P[2] * j).norm()
                grid[(i, j)] = q
        for i in range(NT):
            for j in range(NT - i):
                tris = [((i, j), (i + 1, j), (i, j + 1))]
                if i + j < NT - 1:
                    tris.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
                for t in tris:
                    parts.append(triangle(vs=[vertex(pos=grid[k] * (RS * 1.004), normal=grid[k],
                                                     color=COL_TRI, shininess=0.1) for k in t]))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            arc = [_slerp(P[a], P[b], s / 30) * (RS * 1.032) for s in range(31)]
            parts += _arc_sticks(arc, 0.014, C_D)
        for k in range(3):
            X, Y, Z = P[k], P[(k + 1) % 3], P[(k + 2) % 3]
            e1, ez = _tangent(X, Y), _tangent(X, Z)
            e2 = (ez - e1 * e1.dot(ez)).norm()
            ang = math.acos(e1.dot(ez))
            arc = [(X * math.cos(0.17) + (e1 * math.cos(ang * s / 16) + e2 * math.sin(ang * s / 16))
                    * math.sin(0.17)).norm() * (RS * 1.033) for s in range(17)]
            parts += _arc_sticks(arc, 0.009, C_HL)
        return parts

    def angle_label_pos(sign, k):
        X, Y, Z = (V_ * sign for V_ in (VERT[k], VERT[(k + 1) % 3], VERT[(k + 2) % 3]))
        e1, ez = _tangent(X, Y), _tangent(X, Z)
        bis = (e1 + ez).norm()
        return (X * math.cos(0.36) + bis * math.sin(0.36)) * (RS * 1.04)

    class _Copy:
        pass

    copies = []
    for k in range(2):
        cp = _Copy()
        cp.ball = sphere(pos=V(0, 0, 0), radius=RS, color=COL_BALL, shininess=0.15)
        cp.tri = compound(tri_parts(1) + tri_parts(-1), origin=V(0, 0, 0))
        cp.tri.shininess = 0.0
        cp.body = _Rigid()
        cp.body.add(cp.tri, V(0, 0, 0))
        copies.append(cp)
    cL, cR = copies
    views = [(XL, _IDENT), (XR, FLIP)]

    def place_copy(cp, P, turn, s):
        cp.ball.pos = P
        cp.body.pose(turn, s, P)

    # labels: the angles and the area, on the triangle in the left copy and
    # on the antipodal one in the right copy
    def make_labels(sign):
        labs = [_lab(V(0, 0, 0), NAMES[k], C_HL, back=True, visible=False) for k in range(3)]
        labs.append(_lab(V(0, 0, 0), "A", C_D, back=True, visible=False))
        local = [angle_label_pos(sign, k) for k in range(3)] + [g * sign * (RS * 1.04)]
        return labs, local
    labsL, locL = make_labels(1)
    labsR, locR = make_labels(-1)

    def place_labels(labs, loc, P, turn, s):
        for lb, p in zip(labs, loc):
            lb.pos = P + turn.apply(s, p)

    # everything else is drawn in place, in both copies
    def in_view(v, p):
        P, turn = views[v]
        return P + turn.apply(1, p)

    def dir_view(v, d):
        return views[v][1].apply(1, d)

    gcirc = [[ring(pos=in_view(v, V(0, 0, 0)), axis=dir_view(v, n), radius=RS * 1.0305,
                   thickness=0.0055, color=V(0.78, 0.78, 0.74), opacity=0, visible=False)
              for n in circ] for v in range(2)]
    fills, hatch, steps = [], [], []
    for m, L in enumerate(lunes):
        K = max(6, int(round(math.degrees(L["ang"]) / 7)))
        ps_k = [L["ang"] * k / K for k in range(1, K)]
        steps.append(ps_k)
        rad = RS * (1.0235 + 0.002 * m)
        hatch.append([[ring(pos=in_view(v, V(0, 0, 0)), axis=dir_view(v, lune_normal(L, p)),
                            radius=rad, thickness=0.0055, color=COL_LUNE[m], visible=False)
                       for p in ps_k] for v in range(2)])
        nps_f = max(8, int(round(math.degrees(L["ang"]) / 6)))
        fl = []
        for v in range(2):
            tris = []
            for ps0 in (0.0, math.pi):
                for q in lune_mesh(L, ps0, ps0 + L["ang"], nps_f, RS * (1.012 + 0.003 * m)):
                    qw = [in_view(v, p) for p in q]
                    nr = [dir_view(v, p.norm()) for p in q]
                    vs = [vertex(pos=p, normal=n_, color=COL_LUNE[m], shininess=0.1)
                          for p, n_ in zip(qw, nr)]
                    tris.append(triangle(vs=[vs[0], vs[1], vs[2]]))
                    tris.append(triangle(vs=[vs[0], vs[2], vs[3]]))
            cpd = compound(tris, origin=views[v][0])
            cpd.opacity = 0.0
            cpd.shininess = 0.0
            cpd.visible = False
            fl.append(cpd)
        fills.append(fl)
    sweep = [ring(pos=in_view(v, V(0, 0, 0)), axis=V(1, 0, 0), radius=RS * 1.033,
                  thickness=0.016, color=C_HL, visible=False) for v in range(2)]
    stubs = [cylinder(pos=V(0, 0, 0), axis=V(0, 0.1, 0), radius=0.016, color=C_HL,
                      visible=False) for v in range(2)]

    def set_sweep(m, ps):
        """The turning circle at angle ps; the hatching it has passed."""
        L = lunes[m]
        for v in range(2):
            sweep[v].axis = dir_view(v, lune_normal(L, ps).norm())
        for v, ring_s in enumerate(hatch[m]):
            for p, rg in zip(steps[m], ring_s):
                if rg.visible != (p <= ps + 1e-9):
                    rg.visible = p <= ps + 1e-9

    def set_stubs(m):
        X = lunes[m]["X"]
        for v, sgn in ((0, 1), (1, -1)):
            stubs[v].pos = in_view(v, X * (sgn * RS * 0.98))
            stubs[v].axis = dir_view(v, X * (sgn * RS * 0.32))

    row1, row2, row3 = _rows(sc, 3)

    def reset():
        for cp in copies:
            place_copy(cp, V(0, 0, 0), _IDENT, 0)
        cR.ball.visible = cR.tri.visible = False
        cL.ball.visible = cL.tri.visible = True
        place_labels(labsL, locL, V(0, 0, 0), _IDENT, 0)
        _show(labsL, True)
        _show(labsR, False)
        for v in range(2):
            _show(gcirc[v], False)
            for rg in gcirc[v]:
                rg.opacity = 0
        for m in range(3):
            for v in range(2):
                _show(hatch[m][v], False)
                fills[m][v].visible = False
                fills[m][v].opacity = 0
        _show(sweep + stubs, False)

    def split(u):
        """u = 0: one sphere in the middle; u = 1: the left copy as it is,
        the right copy turned over."""
        place_copy(cL, XL * u, _IDENT, 0)
        place_labels(labsL, locL, XL * u, _IDENT, 0)
        place_copy(cR, XR * u, FLIP, _smooth((u - 0.2) / 0.8))

    reset()
    _ready(sc)
    while True:
        reset()
        row1.text = _pre("a triangle on a sphere of radius r:  its sides are arcs of great "
                         "circles, its angles α, β, γ, its area A")
        row2.text = row3.text = _pre(" ")
        _wait(2.4)
        # ---- a second copy, turned over: the back of the sphere
        row1.text = _pre("a copy of the sphere, turned over:  its back, with the antipodal "
                         "triangle in front  (its mirror image, area A too)")
        cR.ball.visible = cR.tri.visible = True
        for t in _frames(2.4):
            split(_smooth(t))
        place_labels(labsR, locR, XR, FLIP, 1)
        _show(labsR, True)
        _wait(0.8)
        # ---- the sides, extended to whole great circles
        row1.text = _pre("extend the sides:  three great circles")
        for v in range(2):
            _show(gcirc[v], True)
        for t in _frames(1.0):
            for v in range(2):
                for rg in gcirc[v]:
                    rg.opacity = t
        _wait(0.4)
        # ---- the three lune pairs, swept by a turning great circle
        acc = []
        for m, L in enumerate(lunes):
            nm = NAMES[m]
            row1.text = _pre(f"turn a great circle through the corner {nm} by {nm}, about the "
                             f"diameter there:  it sweeps a lune and, opposite, its antipode")
            if m == 0:
                row2.text = _pre("turned by π it would sweep the whole sphere, 4πr²:  so turned by "
                                 "α it sweeps α/π of it,  4αr²")
            set_stubs(m)
            for v in range(2):
                sweep[v].color = COL_LUNE[m] * 1.25
            set_sweep(m, 0.0)
            _show(sweep + stubs, True)
            for t in _frames(2.6):
                set_sweep(m, L["ang"] * _smooth(t))
            set_sweep(m, L["ang"])
            for v in range(2):
                fills[m][v].visible = True
            for t in _frames(0.5):
                for v in range(2):
                    fills[m][v].opacity = 0.26 * t
            _show(sweep + stubs, False)
            acc.append(f"4{nm}r²")
            row2.text = _pre("swept:   " + "  +  ".join(acc))
            _wait(0.7)
        # ---- the count
        row1.text = _pre("the six lunes cover the sphere:  every point once, but the triangle "
                         "and its antipode three times each")
        row2.text = _pre("4αr²  +  4βr²  +  4γr²   =   4πr²  +  2A  +  2A")
        row3.text = _pre(f"A  =  (α + β + γ − π)·r²        here:  α + β + γ = "
                         f"{math.degrees(AL + BE + GA):.0f}°,   A = {EXC:.4f}·r²  =  "
                         f"{EXC / (4 * math.pi):.2f} of the sphere")
        _wait(6.5)
        # ---- back to one sphere
        _show(labsR, False)
        for t in _frames(0.6):
            for m in range(3):
                for v in range(2):
                    fills[m][v].opacity = 0.26 * (1 - t)
        for m in range(3):
            for v in range(2):
                _show(hatch[m][v], False)
                fills[m][v].visible = False
            for v in range(2):
                _show(gcirc[v], False)
        for t in _frames(1.6):
            split(1 - _smooth(t))
        _wait(0.3)


# ============================================================== I21

def _quat(M):
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd)."""
    m = [[float(v) for v in row] for row in M]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s,
                (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s, 0.25 * s)


class _Turn:
    """The rotation path from the identity to Q: R(s) turns by s·θ about
    Q's own axis, so s = 1 lands exactly on Q."""

    def __init__(self, Q):
        w, x, y, z = _quat(Q)
        if w < 0:
            w, x, y, z = -w, -x, -y, -z
        self.theta = 2 * math.acos(min(1.0, w))
        n = math.sqrt(x * x + y * y + z * z)
        self.axis = V(x / n, y / n, z / n) if n > 1e-12 else V(0, 1, 0)

    def apply(self, s, v):
        """R(s)·v (Rodrigues)."""
        a = s * self.theta
        if abs(a) < 1e-12:
            return V(v.x, v.y, v.z)
        k = self.axis
        c, sn = math.cos(a), math.sin(a)
        return v * c + k.cross(v) * sn + k * (k.dot(v) * (1 - c))


def _frames_matrix(a1, b1, a2, b2):
    """The rotation matrix (rows) taking the orthonormal frame (a1, b1,
    a1×b1) onto (a2, b2, a2×b2)."""
    c1, c2 = a1.cross(b1), a2.cross(b2)
    F1, F2 = (a1, b1, c1), (a2, b2, c2)
    return [[sum(_vt(F2[k])[i] * _vt(F1[k])[j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _rot(v, k, ang):
    """v turned by ang about the unit axis k through the origin."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + k.cross(v) * s + k * (k.dot(v) * (1 - c))


def _hull_faces(verts):
    """Faces of a convex polyhedron (index lists, counter-clockwise seen
    from outside) from its vertices: the planes through three vertices with
    all the others behind."""
    found = {}
    for i, j, l in itertools.combinations(range(len(verts)), 3):
        nr = (verts[j] - verts[i]).cross(verts[l] - verts[i])
        if nr.mag < 1e-9:
            continue
        nr = nr.norm()
        side = [nr.dot(v - verts[i]) for v in verts]
        if max(side) > 1e-9:
            nr = -nr
            side = [-x for x in side]
            if max(side) > 1e-9:
                continue
        on = tuple(q for q, sv in enumerate(side) if sv > -1e-9)
        found[on] = nr
    faces = []
    for idx, nr in found.items():
        c = sum((verts[i] for i in idx), V(0, 0, 0)) / len(idx)
        e1 = (verts[idx[0]] - c).norm()
        e2 = nr.cross(e1)
        faces.append(sorted(idx, key=lambda i: math.atan2((verts[i] - c).dot(e2),
                                                          (verts[i] - c).dot(e1))))
    return faces


def scene_I21():
    """Descartes: lay the faces at a corner of a convex polyhedron flat and
    they leave a gap, the angle missing at that corner. Every corner's gap,
    put side by side, fills exactly two full turns. Why: the gaps are 360°
    at each of the V corners less all the face angles; a face with n sides
    is n − 2 triangles, so its angles add to (n − 2)·180°, and all the
    faces to (2E − 2F)·180°, every edge lying in two faces. So the gaps add
    to 360°·V − 180°·(2E − 2F) = 360°·(V − E + F) = 720° by Euler."""
    sc = _canvas("Σδ = 720°   (Descartes)",
                 "the angles missing at the corners of a convex polyhedron add up to "
                 "two full turns",
                 2.4, V(-0.42, -0.46, -0.78), V(0, 0, 0))
    s_right, s_up = _screen_basis(sc)
    ncam = -sc.forward.norm()                    # towards the viewer

    def scr(x, y):
        return sc.center + s_right * x + s_up * y

    # ---- the solid: a triangle at the bottom, a smaller one above it (the
    # sides are planar: one is the other shrunk towards a point above), and
    # an apex off the middle
    base = [V(-1.3, 0, 0.62), V(1.0, 0, 1.0), V(0.62, 0, -1.2)]
    cxz = V(0.08, 0, 0.12)
    top = [cxz + (b - cxz) * 0.70 + V(0, 1.15, 0) for b in base]
    model = base + top + [V(0.18, 1.95, 0.05)]
    KS = 0.95
    gm = sum(model, V(0, 0, 0)) / len(model)
    O = scr(-2.05, -0.1) - gm * KS
    P = [O + p * KS for p in model]
    NV = len(P)
    faces = _hull_faces(P)
    NF = len(faces)
    NE = sum(len(f) for f in faces) // 2
    fnorm = []
    for f in faces:
        nr = (P[f[1]] - P[f[0]]).cross(P[f[2]] - P[f[0]]).norm()
        fnorm.append(nr)
        _check(all(abs(nr.dot(P[i] - P[f[0]])) < 1e-9 for i in f), "every face is flat")
        _check(all(nr.dot(P[i] - P[f[0]]) < -1e-6 for i in range(NV) if i not in f),
               "the solid is convex: every other corner behind each face")
    _check((NV, NE, NF) == (7, 12, 7) and sorted(len(f) for f in faces) == [3, 3, 3, 3, 4, 4, 4],
           "seven corners, twelve edges, seven faces: four triangles and three quadrilaterals")
    _check(NV - NE + NF == 2, "V − E + F = 2")

    # ---- every corner: its edges in turn round it (seen from outside), the
    # faces between them, their angles, the gap; and the corner opened out:
    # cut along its first edge, the bends at the other edges go to zero
    RHO = 0.5                                    # the radius of the angle sectors
    corners = []
    for v in range(NV):
        fs = [k for k, f in enumerate(faces) if v in f]
        nxt = {}
        for k in fs:
            f = faces[k]
            i = f.index(v)
            nxt[f[(i + 1) % len(f)]] = (k, f[i - 1])  # face k runs from that edge to the next
        start = faces[fs[0]][(faces[fs[0]].index(v) + 1) % len(faces[fs[0]])]
        rays, fcs, w = [], [], start
        for _ in range(len(fs)):
            k, w2 = nxt[w]
            rays.append((P[w] - P[v]).norm())
            fcs.append(k)
            w = w2
        _check(w == start, "the faces close up round the corner")
        kk = len(rays)
        angs = [math.acos(max(-1.0, min(1.0, rays[i].dot(rays[(i + 1) % kk])))) for i in range(kk)]
        ms = [rays[i].cross(rays[(i + 1) % kk]).norm() for i in range(kk)]
        for i in range(kk):
            _check(ms[i].dot(fnorm[fcs[i]]) > 1 - 1e-9, "each sector lies in its face, outward")
        bends = [0.0] + [math.atan2(rays[i].dot(ms[i - 1].cross(ms[i])), ms[i - 1].dot(ms[i]))
                         for i in range(1, kk)]
        delta = 2 * math.pi - sum(angs)
        nv_out = sum((fnorm[k] for k in fcs), V(0, 0, 0)).norm()
        corners.append(dict(v=v, rays=rays, fcs=fcs, angs=angs, bends=bends, delta=delta,
                            k=kk, nout=nv_out))
    deltas = [c["delta"] for c in corners]
    _check(abs(sum(deltas) - 4 * math.pi) < 1e-9, "the gaps add up to 720°")
    _check(min(deltas) > math.radians(30), "every gap is clearly open")

    def chain(c, t):
        """Rays and sector normals of corner c, bends scaled by 1 − t."""
        r, m = [c["rays"][0]], [c["rays"][0].cross(c["rays"][1]).norm()]
        for i in range(c["k"]):
            r.append(_rot(r[i], m[i], c["angs"][i]))
            if i + 1 < c["k"]:
                m.append(_rot(m[i], r[i + 1], c["bends"][i + 1] * (1 - t)))
        return r, m

    for c in corners:
        r, m = chain(c, 0.0)
        _check(all((r[i] - c["rays"][i]).mag < 1e-9 for i in range(c["k"])) and
               (r[c["k"]] - c["rays"][0]).mag < 1e-9, "the chain rebuilds the corner, closed")
        r, m = chain(c, 1.0)
        _check(all((mm - m[0]).mag < 1e-12 for mm in m), "laid flat: all in one plane")
        _check(abs(_polar_angle(r[c["k"]], r[0], m[0]) % (2 * math.pi) - c["delta"]) < 1e-9,
               "the gap left is 360° less the face angles")
        # each sector, at the corner, lies inside its face
        for i, k in enumerate(c["fcs"]):
            f = faces[k]
            for j in range(1, 6):
                for rr in (0.3, 0.7, 1.0):
                    ms_i = c["rays"][i].cross(c["rays"][(i + 1) % c["k"]]).norm()
                    x = P[c["v"]] + _rot(c["rays"][i], ms_i, c["angs"][i] * j / 6) * (RHO * rr)
                    inside = all((P[f[(q + 1) % len(f)]] - P[f[q]]).cross(x - P[f[q]]).dot(fnorm[k])
                                 > -1e-9 for q in range(len(f)))
                    _check(inside, "each angle sector fits inside its face")
        # opening out, no sector passes through another (points of one
        # sector stay off the planes of the others, or outside their angle)
        for t in [i / 24 for i in range(25)]:
            r, m = chain(c, t)
            for i, j in itertools.combinations(range(c["k"]), 2):
                if j == i + 1:
                    continue
                for a_ in (0.25, 0.5, 0.75):
                    for rr in (0.4, 0.8):
                        x = _rot(r[i], m[i], c["angs"][i] * a_) * rr
                        d = x.dot(m[j])
                        px = x - m[j] * d
                        in_ang = (_polar_angle(r[j], px, m[j]) % (2 * math.pi) < c["angs"][j] and
                                  px.mag < 1)
                        _check(abs(d) > 1e-3 or not in_ang, "the sectors open without crossing")

    # ---- where things go: the corners are laid flat at a station on the
    # right; their gaps go into two discs below it. Angles on the screen are
    # counted anticlockwise from straight up.
    S = scr(1.3, 1.08)
    DISC = [scr(0.62, -1.42), scr(2.02, -1.42)]
    e_up, e_left = s_up, ncam.cross(s_up)
    _check((e_left + s_right).mag < 1e-9, "anticlockwise on the screen")

    def at_angle(c, th, rr):
        return c + (e_up * math.cos(th) - s_right * math.sin(th)) * rr

    def scr_angle(w):
        return math.atan2(-w.dot(s_right), w.dot(e_up))

    eye = _eye(sc)
    # the corner shown first: an upper corner (four faces) facing the viewer
    first = max((c for c in corners if c["k"] == 4),
                key=lambda c: c["nout"].dot((eye - P[c["v"]]).norm()))
    rest = sorted((c for c in corners if c is not first), key=lambda c: (-c["k"], c["v"]))
    order = [first] + rest
    # each corner is turned so that its first sector faces the viewer and
    # its gap, once flat, points straight down; its gap fills the next slot
    # of the discs (split at the end of the first disc)
    sigma = 0.0
    for c in order:
        w0 = _rot(-e_up, ncam, c["delta"] / 2)
        r0 = c["rays"][0]
        m0 = r0.cross(c["rays"][1]).norm()
        Mf = _frames_matrix(r0, m0.cross(r0), w0, ncam.cross(w0))
        c["turn"] = _Turn(Mf)
        _check((c["turn"].apply(1, r0) - w0).mag < 1e-9 and
               (c["turn"].apply(1, m0) - ncam).mag < 1e-9, "the corner turned to face the viewer")
        r, m = chain(c, 1.0)
        g_start = scr_angle(c["turn"].apply(1, r[c["k"]]))
        _check(abs((g_start - (math.pi - c["delta"] / 2) + math.pi) % (2 * math.pi) - math.pi) < 1e-9,
               "laid flat, its gap points straight down")
        pieces = []
        lo, hi = sigma, sigma + c["delta"]
        for d, (a0, a1) in enumerate(((0.0, 2 * math.pi), (2 * math.pi, 4 * math.pi))):
            p0, p1 = max(lo, a0), min(hi, a1)
            if p1 > p0 + 1e-12:
                pieces.append(dict(disc=d, slot=(p0 - a0, p1 - a0),
                                   at=(g_start + p0 - lo, g_start + p1 - lo)))
        c["pieces"] = pieces
        sigma = hi
    _check(abs(sigma - 4 * math.pi) < 1e-9, "the gaps fill the two discs exactly")
    # nothing leaves the view: the solid, the station and the discs
    hw, hh = sc.range * 980 / 600, sc.range

    def on_screen(p, m=0.15):
        q = p - sc.center
        return abs(q.dot(s_right)) < hw - m and abs(q.dot(s_up)) < hh - m

    _check(all(on_screen(p) for p in P), "the solid is in view")
    _check(all(on_screen(at_angle(c_, 2 * math.pi * i / 12, RHO)) for c_ in DISC + [S]
               for i in range(12)), "the station and the discs are in view")
    _check((DISC[1] - DISC[0]).mag > 2 * RHO + 0.25 and
           (S - DISC[0]).dot(s_up) > 2 * RHO + 0.35, "discs and station keep apart")
    _check(min((S - p).dot(s_right) for p in P) > RHO + 0.4, "the station is clear of the solid")
    # counting
    ntri = sum(len(f) - 2 for f in faces)
    _check(ntri == 2 * NE - 2 * NF, "the faces cut into 2E − 2F triangles")
    _check(abs(sum(sum(c["angs"]) for c in corners) - math.pi * ntri) < 1e-9,
           "all the face angles: (2E − 2F)·180°")
    _check(360 * NV - 180 * (2 * NE - 2 * NF) == 720, "360°·V − 180°·(2E − 2F) = 720°")

    # ---- drawing: the solid
    FCOL = [C_A, C_B, C_C, C_F, V(0.42, 0.65, 0.28), V(0.80, 0.45, 0.62), V(0.45, 0.50, 0.66)]
    for k, f in enumerate(faces):
        for i in range(1, len(f) - 1):
            triangle(vs=[vertex(pos=P[f[q]], normal=fnorm[k], color=FCOL[k], shininess=0.2)
                         for q in (0, i, i + 1)])
    for e in {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in faces for i in range(len(f))}:
        curve(pos=[P[e[0]], P[e[1]]], radius=0.012, color=C_HL * 0.85)
    diags = []
    for k, f in enumerate(faces):
        for i in range(2, len(f) - 1):
            lift = fnorm[k] * 0.008
            diags.append(curve(pos=[P[f[0]] + lift, P[f[i]] + lift], radius=0.013, color=C_D,
                               visible=False))
    dot = sphere(pos=P[0], radius=0.06, color=C_HL, visible=False)

    def bright(col, f=1.18):
        return V(min(1, col.x * f), min(1, col.y * f), min(1, col.z * f))

    # the sectors of every corner: fans of triangles on vertices we move
    NA = 10
    for c in corners:
        c["fans"] = []
        for i, k in enumerate(c["fcs"]):
            col = bright(FCOL[k])
            vs = [vertex(pos=P[c["v"]], color=col, normal=ncam) for _ in range(NA + 2)]
            tris = [triangle(vs=[vs[0], vs[j], vs[j + 1]], visible=False) for j in range(1, NA + 1)]
            c["fans"].append((vs, tris))

    def pose_corner(c, u, t):
        """Corner c taken off (u: 0 at its place, 1 at the station) and
        opened out by t."""
        r, m = chain(c, t)
        base_pt = P[c["v"]] + c["nout"] * 0.012
        Pc = base_pt + (S - base_pt) * _smooth(u)
        tr = c["turn"]
        for i, (vs, tris) in enumerate(c["fans"]):
            nr = tr.apply(u, m[i])
            vs[0].pos, vs[0].normal = Pc, nr
            for j in range(NA + 1):
                vs[j + 1].pos = Pc + tr.apply(u, _rot(r[i], m[i], c["angs"][i] * j / NA)) * RHO
                vs[j + 1].normal = nr

    def show_corner(c, flag):
        for vs, tris in c["fans"]:
            _show(tris, flag)

    # the gaps: one or two red pieces per corner, moved rigidly in the plane
    # of the screen from the station to their slots
    GCOL = [C_E, V(0.95, 0.42, 0.36)]
    NG = 14
    for n_, c in enumerate(order):
        for pc in c["pieces"]:
            col = GCOL[n_ % 2]
            vs = [vertex(pos=S, color=col, normal=ncam) for _ in range(NG + 2)]
            pc["vs"] = vs
            pc["tris"] = [triangle(vs=[vs[0], vs[j], vs[j + 1]], visible=False)
                          for j in range(1, NG + 1)]
            pc["edge"] = curve(pos=[S] * (NG + 3), radius=0.005, color=V(1, 0.85, 0.8),
                               visible=False)

    def pose_piece(pc, w):
        """w = 0 at the station, 1 in its slot."""
        a0, a1 = pc["at"]
        s0, s1 = pc["slot"]
        turn = (s0 - a0 + math.pi) % (2 * math.pi) - math.pi
        cen = S + (DISC[pc["disc"]] - S) * w
        pts = [at_angle(cen, a0 + turn * w + (a1 - a0) * j / NG, RHO) for j in range(NG + 1)]
        pc["vs"][0].pos = cen
        for j, p in enumerate(pts):
            pc["vs"][j + 1].pos = p
        for j, p in enumerate([cen] + pts + [cen]):
            pc["edge"].modify(j, pos=p)

    for c in order:
        for pc in c["pieces"]:
            a0, a1 = pc["at"]
            s0, s1 = pc["slot"]
            turn = (s0 - a0 + math.pi) % (2 * math.pi) - math.pi
            _check(abs(((a1 - a0) - (s1 - s0))) < 1e-12 and
                   abs(math.sin((a0 + turn - s0) / 2)) < 1e-12, "each piece lands in its slot")

    def show_piece(pc, flag):
        _show(pc["tris"] + [pc["edge"]], flag)

    rims = [ring(pos=d, axis=ncam, radius=RHO, thickness=0.006, color=C_HL * 0.8, visible=False)
            for d in DISC]
    lab360 = [_lab(d - e_up * (RHO + 0.24), "360°", C_HL, visible=False) for d in DISC]
    lab_d = _lab(S - e_up * (RHO * 0.62), "δ", C_HL, back=True, visible=False)
    every_piece = [pc for c in order for pc in c["pieces"]]

    row1, row2, row3 = _rows(sc, 3)

    def deg(x):
        return f"{math.degrees(x):.1f}°"

    _ready(sc)
    while True:
        for c in order:
            show_corner(c, False)
        for pc in every_piece:
            show_piece(pc, False)
        _show(diags + rims + lab360 + [dot, lab_d], False)
        row1.text = _pre(f"a convex polyhedron:  V = {NV} corners,  E = {NE} edges,  "
                         f"F = {NF} faces")
        row2.text = row3.text = _pre(" ")
        _wait(2.4)
        done = []
        for n_, c in enumerate(order):
            quick = n_ > 0
            T1, T2, T3, T4 = (0.5, 0.6, 0.15, 0.55) if quick else (1.2, 1.6, 0.5, 1.2)
            dot.pos = P[c["v"]]
            dot.visible = True
            pose_corner(c, 0, 0)
            show_corner(c, True)
            if not quick:
                row1.text = _pre("copy the faces at one corner, take them off and lay them flat")
                _wait(0.7)
            for t in _frames(T1):
                pose_corner(c, t, 0)
            for t in _frames(T2):
                pose_corner(c, 1, _smooth(t))
            for pc in c["pieces"]:
                pose_piece(pc, 0)
                show_piece(pc, True)
            if not quick:
                lab_d.visible = True
                parts = " + ".join(deg(a) for a in c["angs"])
                row1.text = _pre("they leave a gap:  δ, the angle missing at that corner")
                row2.text = _pre(f"δ  =  360° − ({parts})  =  {deg(c['delta'])}")
                _wait(2.0)
                lab_d.visible = False
                row1.text = _pre("do it at every corner, and put the gaps side by side")
                _show(rims + lab360, True)
            else:
                _wait(T3)
            for t in _frames(T4):
                for pc in c["pieces"]:
                    pose_piece(pc, _smooth(t))
            show_corner(c, False)
            dot.visible = False
            done.append(deg(c["delta"]))
            row2.text = _pre("gaps:  " + " + ".join(done) +
                             (f"  =  {math.degrees(sum(cc['delta'] for cc in order[:n_ + 1])):.1f}°"
                              if n_ else ""))
        row1.text = _pre(f"the {NV} gaps fill exactly two full turns:  720°")
        _wait(2.2)
        # ---- why: the count
        _show(diags, True)
        row1.text = _pre("why:  the gaps are 360° at each of the V corners, less all the face "
                         "angles;   a face with n sides is n − 2 triangles")
        row2.text = _pre(f"so its angles add to (n − 2)·180°;  every edge lies in two faces:  "
                         f"all faces  (2E − 2F)·180°  =  {ntri}·180°")
        row3.text = _pre(f"Σδ  =  360°·V − 180°·(2E − 2F)  =  360°·(V − E + F)  =  360°·2  =  720°"
                         f"        (Euler:  {NV} − {NE} + {NF} = 2)")
        _wait(7.0)


# ============================================================== I25

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


def _cyclic(pts, nrm):
    """The points in anticlockwise order seen from the side nrm points to."""
    c = sum(pts, V(0, 0, 0)) / len(pts)
    e1 = (pts[0] - c).norm()
    e2 = nrm.cross(e1)
    return sorted(pts, key=lambda p: math.atan2((p - c).dot(e2), (p - c).dot(e1)))


def _football():
    """The truncated icosahedron (unit circumradius): every edge of the
    icosahedron cut at its thirds; a pentagon round each old corner, a
    hexagon on each old face. Returns a list of faces (point lists)."""
    V0, T0 = _icosahedron()
    e = min((p - q).mag for p, q in itertools.combinations(V0, 2))
    third = {}
    for i, j in itertools.permutations(range(12), 2):
        if abs((V0[i] - V0[j]).mag - e) < 1e-9:
            third[(i, j)] = V0[i] + (V0[j] - V0[i]) / 3
    faces = []
    for i in range(12):
        faces.append(_cyclic([p for (a, b), p in third.items() if a == i], V0[i]))
    for a, b, c in T0:
        pts = [third[(a, b)], third[(b, a)], third[(b, c)], third[(c, b)], third[(c, a)],
               third[(a, c)]]
        faces.append(_cyclic(pts, (V0[a] + V0[b] + V0[c]).norm()))
    R = faces[0][0].mag
    return [[p / R for p in f] for f in faces]


def _geodesic(nu):
    """The icosahedron's faces cut into nu² triangles and pushed out onto
    the unit sphere: (points, triangles)."""
    V0, T0 = _icosahedron()
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


def _polar_ball(nu):
    """The polyhedron whose faces touch the unit ball at the points of a
    geodesic sphere (frequency nu): face i is cut from the tangent plane at
    u_i by its neighbours. nu = 1 is the dodecahedron; nu = 2 has twelve
    pentagons and thirty hexagons. Returns faces (point lists), scaled to
    unit circumradius."""
    U, T = _geodesic(nu)
    corner = []
    for t in T:
        a, b, c = (U[i] for i in t)
        nr = (b - a).cross(c - a).norm()
        if nr.dot(a) < 0:
            nr = -nr
        corner.append(nr / nr.dot(a))            # on the three tangent planes of t
    around = {i: [] for i in range(len(U))}
    for k, t in enumerate(T):
        for i in t:
            around[i].append(k)
    faces = [_cyclic([corner[k] for k in around[i]], U[i]) for i in range(len(U))]
    R = max(p.mag for f in faces for p in f)
    return [[p / R for p in f] for f in faces]


def _census(faces):
    """Corners, edges and faces of a polyhedron given by its faces (shared
    corners matched by position), and checks that it is a closed convex
    surface with three faces at every corner."""
    key, pts = {}, []

    def idx(p):
        k = (round(p.x, 7), round(p.y, 7), round(p.z, 7))
        if k not in key:
            key[k] = len(pts)
            pts.append(p)
        return key[k]
    fidx = [[idx(p) for p in f] for f in faces]
    edges = {}
    for fi, f in enumerate(fidx):
        for a, b in zip(f, f[1:] + f[:1]):
            edges.setdefault(tuple(sorted((a, b))), []).append(fi)
    at = {}
    for fi, f in enumerate(fidx):
        for a in f:
            at.setdefault(a, []).append(fi)
    _check(all(len(v) == 2 for v in edges.values()), "every edge lies in exactly two faces")
    _check(all(len(v) == 3 for v in at.values()), "three faces at every corner")
    for f in faces:
        c = sum(f, V(0, 0, 0)) / len(f)
        nr = (f[1] - f[0]).cross(f[2] - f[0]).norm()
        if nr.dot(c) < 0:
            nr = -nr
        _check(all(abs(nr.dot(p - f[0])) < 1e-9 for p in f), "every face is flat")
        _check(all(nr.dot(p - f[0]) < 1e-9 for p in pts), "the ball is convex")
    return pts, fidx, edges, at


def scene_I25():
    """Twelve pentagons: a ball of p pentagons and h hexagons, three at
    every corner. Taken apart, its faces have 5p + 6h sides and as many
    corners; each edge of the ball was two sides and each vertex three
    corners, so E = (5p + 6h)/2 and V = (5p + 6h)/3. Euler's V − E + F = 2
    becomes (5p + 6h)/3 − (5p + 6h)/2 + p + h = p/6 = 2: p = 12, whatever
    h is (a hexagon adds 6/3 − 6/2 + 1 = 0)."""
    sc = _canvas("V − E + F = 2   ⟹   p = 12",
                 "pentagons and hexagons, three at every corner:  count the edges and corners "
                 "from the faces, and Euler leaves exactly twelve pentagons",
                 2.15, V(0.0, -0.32, -0.95), V(0, 0, 0))
    s_right, s_up = _screen_basis(sc)
    eye = _eye(sc)
    RB = 1.3                                     # the football's radius

    # ---- the three balls, checked: flat faces, convex, every edge in two
    # faces, three faces at every corner; the counts from the faces; Euler
    balls = []
    for faces in (_polar_ball(1), _football(), _polar_ball(2)):
        pts, fidx, edges, at = _census(faces)
        p = sum(1 for f in faces if len(f) == 5)
        h = sum(1 for f in faces if len(f) == 6)
        _check(p + h == len(faces), "only pentagons and hexagons")
        nV, nE, nF = len(pts), len(edges), len(faces)
        _check(2 * nE == 5 * p + 6 * h and 3 * nV == 5 * p + 6 * h,
               "E = (5p + 6h)/2 and V = (5p + 6h)/3")
        _check(nV - nE + nF == 2, "V − E + F = 2")
        _check(p == 12, "twelve pentagons")
        balls.append(dict(faces=faces, pts=pts, fidx=fidx, edges=edges, at=at, p=p, h=h,
                          V=nV, E=nE, F=nF))
    ball = balls[1]
    _check((ball["V"], ball["E"], ball["F"], ball["h"]) == (60, 90, 32, 20), "the football")
    _check(all(abs((f[i] - f[i - 1]).mag - (ball["faces"][0][1] - ball["faces"][0][0]).mag)
               < 1e-9 for f in ball["faces"] for i in range(len(f))), "all its edges equal")
    _check(abs(5 / 3 - 5 / 2 + 1 - 1 / 6) < 1e-12 and abs(6 / 3 - 6 / 2 + 1) < 1e-12,
           "a pentagon adds 1/6 to V − E + F, a hexagon nothing")

    # ---- the football's faces as compounds, each posed on its own: turned
    # with the ball and pushed out along its normal
    COL_P, COL_H = C_C, V(0.74, 0.77, 0.83)
    EDGE = V(0.12, 0.12, 0.14)
    fbody = []
    for f in ball["faces"]:
        f = [q * RB for q in f]
        c = sum(f, V(0, 0, 0)) / len(f)
        nr = c.norm()
        col = COL_P if len(f) == 5 else COL_H
        parts = [triangle(vs=[vertex(pos=c, normal=nr, color=col, shininess=0.2),
                              vertex(pos=f[i], normal=nr, color=col, shininess=0.2),
                              vertex(pos=f[(i + 1) % len(f)], normal=nr, color=col,
                                     shininess=0.2)])
                 for i in range(len(f))]
        parts += [cylinder(pos=f[i], axis=f[(i + 1) % len(f)] - f[i], radius=0.016, color=EDGE)
                  for i in range(len(f))]
        cp = compound(parts, origin=c)
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        fbody.append(dict(cp=cp, body=body, c=c, n=nr, pts=f, k=len(f)))
    AXIS = V(0.25, 1, 0.12).norm()

    def pose_ball(th, d, pent_out=0.0):
        tr = _AxisTurn(AXIS, th)
        for fb in fbody:
            out = d + (pent_out if fb["k"] == 5 else 0.0)
            fb["body"].pose(tr, 1, tr.apply(1, fb["c"] + fb["n"] * out))
        return tr

    # ---- the marks: one edge (two sides once apart) and one corner (three)
    side_marks = [curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.03, color=C_D, visible=False)
                  for _ in range(2)]
    corner_marks = [sphere(pos=V(0, 0, 0), radius=0.06, color=C_HL, visible=False)
                    for _ in range(3)]

    def pick_marks(tr):
        """A pentagon–hexagon edge and its end facing the viewer most."""
        best = None
        for (a, b), fs in ball["edges"].items():
            if sorted(fbody[i]["k"] for i in fs) != [5, 6]:
                continue
            mid = tr.apply(1, (ball["pts"][a] + ball["pts"][b]) * (RB / 2))
            score = (eye - mid).norm().dot(mid.norm())
            if best is None or score > best[0]:
                best = (score, a, b, fs)
        _, a, b, fs = best
        va = tr.apply(1, ball["pts"][a]).dot(eye)
        vb = tr.apply(1, ball["pts"][b]).dot(eye)
        return (a, b), fs, (a if va > vb else b)

    def place_marks(tr, d, edge, efs, vtx):
        a, b = edge
        for cv, fi in zip(side_marks, efs):
            sh = fbody[fi]["n"] * d
            cv.modify(0, pos=tr.apply(1, ball["pts"][a] * RB + sh))
            cv.modify(1, pos=tr.apply(1, ball["pts"][b] * RB + sh))
        for sp, fi in zip(corner_marks, ball["at"][vtx]):
            sp.pos = tr.apply(1, ball["pts"][vtx] * RB + fbody[fi]["n"] * d)

    # ---- the three balls side by side at the end, each one compound
    SMALL = 0.85
    XS = [-2.3, 0.0, 2.3]
    trio = []
    for bl, x in zip(balls, XS):
        parts = []
        for f in bl["faces"]:
            f = [q * SMALL for q in f]
            c = sum(f, V(0, 0, 0)) / len(f)
            nr = c.norm()
            col = COL_P if len(f) == 5 else COL_H
            parts += [triangle(vs=[vertex(pos=c, normal=nr, color=col, shininess=0.2),
                                   vertex(pos=f[i], normal=nr, color=col, shininess=0.2),
                                   vertex(pos=f[(i + 1) % len(f)], normal=nr, color=col,
                                          shininess=0.2)])
                      for i in range(len(f))]
            parts += [cylinder(pos=f[i], axis=f[(i + 1) % len(f)] - f[i], radius=0.012,
                               color=EDGE) for i in range(len(f))]
        cp = compound(parts, origin=V(0, 0, 0))
        cp.visible = False
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        lab = _lab(V(x, 0, 0) - s_up * (SMALL + 0.32), f"p = 12,  h = {bl['h']}", C_HL,
                   visible=False)
        trio.append(dict(cp=cp, body=body, P=V(x, 0, 0), lab=lab))
    hw = sc.range * 980 / 600
    _check(all(abs(t_["P"].dot(s_right)) + SMALL < hw - 0.3 for t_ in trio), "the three balls in view")

    def pose_trio(th):
        tr = _AxisTurn(AXIS, th)
        for t_ in trio:
            t_["body"].pose(tr, 1, t_["P"])

    row1, row2, row3 = _rows(sc, 3)
    W = 0.32                                     # turning speed, radians a second
    _ready(sc)
    while True:
        th = 0.0
        for fb in fbody:
            fb["cp"].visible = True
        for t_ in trio:
            t_["cp"].visible = t_["lab"].visible = False
        _show(side_marks + corner_marks, False)
        row1.text = _pre("a ball of pentagons and hexagons, three at every corner:  p pentagons,"
                         "  h hexagons")
        row2.text = row3.text = _pre(" ")
        for t in _frames(2.6):
            pose_ball(th + W * 2.6 * t, 0)
        th += W * 2.6
        # ---- taken apart: each face with its own sides and corners
        row1.text = _pre("take it apart into its F = p + h faces:  they have 5p + 6h sides and "
                         "5p + 6h corners in all")
        DX = 0.42
        for t in _frames(2.0):
            pose_ball(th + W * 2.0 * t, DX * _smooth(t))
        th += W * 2.0
        tr = pose_ball(th, DX)
        edge, efs, vtx = pick_marks(tr)
        _wait(0.5)
        # ---- an edge was two sides, a vertex three corners
        row1.text = _pre("each edge of the ball has become two sides, each vertex three corners")
        place_marks(tr, DX, edge, efs, vtx)
        _show(side_marks, True)
        _wait(1.2)
        _show(corner_marks, True)
        row2.text = _pre("so   E = (5p + 6h)/2      V = (5p + 6h)/3      F = p + h")
        _wait(1.4)
        for t in _frames(1.2):
            d = DX * (1 - 0.75 * _smooth(t))
            pose_ball(th, d)
            place_marks(tr, d, edge, efs, vtx)
        _wait(0.6)
        for t in _frames(1.0):
            d = DX * 0.25 * (1 - _smooth(t))
            pose_ball(th, d)
            place_marks(tr, d, edge, efs, vtx)
        _show(side_marks + corner_marks, False)
        # ---- Euler
        row1.text = _pre("Euler:  V − E + F = 2")
        row3.text = _pre("(5p + 6h)/3  −  (5p + 6h)/2  +  p + h   =   p/6   =   2       ⟹   "
                         "p = 12")
        _wait(1.4)
        row2.text = _pre("a hexagon adds  6/3 − 6/2 + 1 = 0,   a pentagon  5/3 − 5/2 + 1 = 1/6:  "
                         "only the pentagons count")
        for t in _frames(2.4):
            pose_ball(th + W * 2.4 * t, 0, 0.16 * math.sin(math.pi * min(1, 1.25 * t)))
        th += W * 2.4
        pose_ball(th, 0)
        _wait(1.0)
        # ---- any number of hexagons: always twelve pentagons
        for fb in fbody:
            fb["cp"].visible = False
        pose_trio(th)
        for t_ in trio:
            t_["cp"].visible = t_["lab"].visible = True
        row1.text = _pre("so any such ball has exactly 12 pentagons, however many hexagons")
        row2.text = _pre("        ".join(f"h = {b['h']}:  {b['V']} − {b['E']} + {b['F']} = 2"
                                           for b in balls))
        for t in _frames(6.0):
            pose_trio(th + W * 6.0 * t)
        th += W * 6.0


# ============================================================== K29

def _turn_to(a, b):
    """The shortest rotation taking the unit vector a to the unit vector b."""
    ax = a.cross(b)
    if ax.mag < 1e-12:
        if a.dot(b) > 0:
            return _IDENT
        perp = a.cross(V(1, 0, 0)) if abs(a.x) < 0.9 else a.cross(V(0, 1, 0))
        return _AxisTurn(perp, math.pi)
    return _AxisTurn(ax, math.atan2(ax.mag, a.dot(b)))


def _sph(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return V(math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo))


def scene_K29():
    """Five points on a sphere: the great circle through two of them, P and
    Q, cuts the sphere into two hemispheres. Each of the other three points
    lies in one of them (on the circle it lies in both), so one closed
    hemisphere holds at least two of the three; it holds P and Q on its rim
    as well: four of the five (five if all three are on one side)."""
    sc = _canvas("some closed hemisphere holds 4 of any 5 points",
                 "the great circle through two of the points has at least two of the other "
                 "three on one side",
                 1.95, V(0, -0.36, -0.93), V(0, 0, 0))
    s_right, s_up = _screen_basis(sc)
    eye = _eye(sc)
    R = 1.32
    X0 = [_sph(-5, -70), _sph(-20, 10), _sph(50, 30), _sph(45, 140), _sph(-55, -20)]
    NAMES = ["P", "Q", "", "", ""]
    AXIS = V(0, 1, 0)
    SPIN = 2 * math.pi / 36                      # the whole sphere turns slowly

    # the five points wander, each turning about its own axis
    WANDER = [(V(0.22, 0.49, -0.69), -0.25), (V(-0.26, -0.19, 0.69), 0.36),
              (V(-0.48, -0.29, -0.61), 0.37), (V(-0.01, -0.22, 0.40), -0.23),
              (V(-0.28, -0.65, -0.72), -0.24)]

    def points_at(T):
        return [_AxisTurn(a, w * T).apply(1, x) for x, (a, w) in zip(X0, WANDER)]

    def split(X):
        """The circle's normal, turned towards the side holding at least two
        of the other three; how many points each closed side holds."""
        n = X[0].cross(X[1]).norm()
        sd = [n.dot(x) for x in X[2:]]
        pos = sum(1 for v in sd if v >= -1e-12)
        neg = sum(1 for v in sd if v <= 1e-12)
        if neg > pos:
            n, pos, neg = -n, neg, pos
        return n, 2 + pos, 2 + neg

    # ---- checks: P and Q on the circle; the side chosen holds ≥ 2 of the
    # other three, so ≥ 4 in all, at the start and all the way as they move;
    # P and Q never become antipodal, so their great circle is always one
    TW = 9.0
    n0, c0, d0 = split(X0)
    _check(c0 == 4 and d0 == 3, "at the start the other three split two and one")
    _check(min(abs(n0.dot(x)) for x in X0[2:]) > 0.25, "none of them near the circle at first")
    for k in range(361):
        X = points_at(TW * k / 360)
        n, c, d = split(X)
        _check(abs(n.dot(X[0])) < 1e-12 and abs(n.dot(X[1])) < 1e-12, "P and Q on the circle")
        _check(c >= 4 and c + d >= 7, "that closed hemisphere holds at least four")
        _check(X[0].cross(X[1]).mag > 0.3, "P and Q never antipodal")
        _check(max(a.dot(b) for a, b in itertools.combinations(X, 2)) < math.cos(math.radians(30)),
               "the points keep at least 30° apart")
    # a search over many hemispheres finds four too
    best = max(sum(1 for x in X0 if u.dot(x) >= 0) for u in _fib_sphere(2000))
    _check(best >= 4, "a sampled hemisphere also holds four")

    # ---- drawing
    ball = sphere(pos=V(0, 0, 0), radius=R, color=V(0.30, 0.40, 0.62), opacity=0.16,
                  shininess=0.1)

    def hemi(col, op):
        """Half a sphere, its pole along +y (radius a little over R)."""
        NL, NO = 10, 48
        tris = []
        for i in range(NL):
            for j in range(NO):
                q = []
                for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    la = math.pi / 2 * a / NL
                    lo = 2 * math.pi * b / NO
                    u = V(math.cos(la) * math.cos(lo), math.sin(la), math.cos(la) * math.sin(lo))
                    q.append(vertex(pos=u * R, normal=u, color=col, shininess=0.1))
                tris.append(triangle(vs=[q[0], q[1], q[2]]))
                tris.append(triangle(vs=[q[0], q[2], q[3]]))
        cp = compound(tris, origin=V(0, 0, 0))
        cp.opacity = op
        cp.visible = False
        body = _Rigid()
        body.add(cp, V(0, 0, 0))
        return cp, body
    shell_in, body_in = hemi(C_C, 0.30)
    shell_out, body_out = hemi(V(0.25, 0.55, 0.85), 0.20)
    gc = ring(pos=V(0, 0, 0), axis=V(0, 1, 0), radius=R * 1.008, thickness=0.014, color=C_D,
              visible=False)
    arc = curve(pos=[V(0, 0, 0)] * 97, radius=0.014, color=C_D, visible=False)
    dots = [sphere(pos=x * R, radius=0.075, color=C_D if k < 2 else C_HL) for k, x in enumerate(X0)]
    halos = [ring(pos=x * R, axis=V(0, 0, 1), radius=0.14, thickness=0.012, color=C_HL,
                  visible=False) for x in X0]
    labs = [_lab(V(0, 0, 0), nm, C_D, back=True) for nm in NAMES[:2]]

    def place(X, th, n=None, grow=None, show_split=False):
        """Everything for the points X (on the unit sphere), the sphere turned
        by th; n the chosen side's pole; grow: how much of the circle to draw."""
        tr = _AxisTurn(AXIS, th)
        W = [tr.apply(1, x) for x in X]
        for k, (dt, hl, w) in enumerate(zip(dots, halos, W)):
            dt.pos = w * R
            hl.pos, hl.axis = w * R, (eye - w * R).norm()
        for k, lb in enumerate(labs):
            w = W[k]
            # beside the point, away from the middle of the sphere on the screen
            o = V(w.dot(s_right), w.dot(s_up), 0)
            o = o.norm() if o.mag > 0.3 else V(0.71, 0.71, 0)
            lb.pos = w * R + (s_right * o.x + s_up * o.y) * 0.24
        if n is not None:
            nw = tr.apply(1, n)
            gc.axis = nw
            body_in.pose(_turn_to(V(0, 1, 0), nw), 1, V(0, 0, 0))
            body_out.pose(_turn_to(V(0, 1, 0), -nw), 1, V(0, 0, 0))
            if grow is not None:
                a0 = W[0]
                b0 = nw.cross(a0)
                for i in range(97):
                    ang = 2 * math.pi * grow * i / 96
                    arc.modify(i, pos=(a0 * math.cos(ang) + b0 * math.sin(ang)) * (R * 1.008))
        return W

    def colour_points(X, n):
        for k in range(2, 5):
            v = n.dot(X[k])
            dots[k].color = C_C * 1.15 if v >= -1e-12 else V(0.45, 0.70, 1.0)
            halos[k].visible = v >= -1e-12
        for k in range(2):
            halos[k].visible = True

    row1, row2, row3 = _rows(sc, 3)
    # the arc from P must reach Q going the right way round: Q lies at
    # angle atan2(Q·(n×P), Q·P) from P on the circle
    qa = math.atan2(X0[1].dot(n0.cross(X0[0])), X0[1].dot(X0[0])) % (2 * math.pi)
    _check(0 < qa < 2 * math.pi, "Q on the circle through P")
    _ready(sc)
    while True:
        th = 0.0
        X = X0
        for k, dt in enumerate(dots):
            dt.color = C_D if k < 2 else C_HL
        _show(halos + [gc, arc, shell_in, shell_out], False)
        ball.visible = True
        place(X, th)
        row1.text = _pre("five points on a sphere")
        row2.text = row3.text = _pre(" ")
        for t in _frames(2.2):
            place(X, th + SPIN * 2.2 * t)
        th += SPIN * 2.2
        # ---- the great circle through P and Q
        row1.text = _pre("two of them, P and Q:  the great circle through them")
        arc.visible = True
        for t in _frames(2.0):
            place(X, th + SPIN * 2.0 * t, n0, grow=_smooth(t))
        th += SPIN * 2.0
        arc.visible = False
        gc.visible = True
        # ---- two hemispheres
        row1.text = _pre("it cuts the sphere into two hemispheres;  each of the other three "
                         "points lies in one of them  (on the circle, in both)")
        shell_in.visible = shell_out.visible = True
        shell_out.opacity = 0.20
        ball.visible = False
        colour_points(X, n0)
        _show(halos, False)
        for t in _frames(2.6):
            place(X, th + SPIN * 2.6 * t, n0)
        th += SPIN * 2.6
        # ---- pigeonholes
        row1.text = _pre("three points, two sides:  one side holds at least two of them")
        row2.text = _pre(f"here:  {c0 - 2} on one side,  {d0 - 2} on the other")
        for t in _frames(2.0):
            place(X, th + SPIN * 2.0 * t, n0)
            shell_out.opacity = 0.20 * (1 - 0.6 * t)
        th += SPIN * 2.0
        row1.text = _pre("that closed hemisphere has P and Q on its rim and those two inside:  "
                         "four of the five")
        colour_points(X, n0)
        row3.text = _pre(f"in this closed hemisphere:  {c0} of the 5 points")
        for t in _frames(2.6):
            place(X, th + SPIN * 2.6 * t, n0)
        th += SPIN * 2.6
        # ---- wherever the points are
        row1.text = _pre("wherever the five points are:  the circle through P and Q, the side "
                         "with at least two of the other three")
        row2.text = _pre("its closed hemisphere holds P, Q and at least two more")
        for t in _frames(TW):
            X = points_at(TW * t)
            n, c, d = split(X)
            place(X, th + SPIN * TW * t, n)
            colour_points(X, n)
            if int(round(t * TW * FPS)) % 4 == 0:
                row3.text = _pre(f"in this closed hemisphere:  {c} of the 5 points")
        th += SPIN * TW
        _wait(1.5)


# ============================================================== I24

def scene_I24():
    """A slanted slice of a cylinder: two balls of the cylinder's radius,
    one above the cutting plane and one below, each touching the cylinder
    along a circle (its equator) and the plane at a point F. For a point P
    of the cut, PF₁ and PQ₁ are tangents from P to the upper ball (Q₁ where
    the vertical line through P meets the upper circle): the tangents from
    P to a ball form a cone of equal segments, and PF₁ turns along it onto
    PQ₁. Likewise PF₂ = PQ₂. So PF₁ + PF₂ = Q₁Q₂, the distance between the
    two circles along a vertical line: the same for every P. The cut is an
    ellipse with foci F₁, F₂, and the constant is its long axis V₁V₂ = 2a."""
    sc = _canvas("PF₁ + PF₂ = const:   a slanted slice of a cylinder is an ellipse",
                 "two balls in the cylinder touch the plane at F₁, F₂ and the cylinder along "
                 "two circles  ·  tangents from P to one ball are equal",
                 2.05, V(0.26, -0.40, -0.88), V(0.05, -0.05, 0))
    s_right, s_up = _screen_basis(sc)
    R = 0.72                                     # the cylinder's radius, and the balls'
    BE = math.radians(34)                        # the plane's slope
    sb, cb = math.sin(BE), math.cos(BE)
    n = V(-sb, cb, 0)                            # the plane's upward normal; it passes through O
    O = V(0, 0, 0)
    yc = R / cb                                  # the balls' centres, at ±yc on the axis
    C1, C2 = V(0, yc, 0), V(0, -yc, 0)
    F1, F2 = C1 - n * R, C2 + n * R

    def cut(ph):
        """The point of the cut above (R cos φ, ·, R sin φ)."""
        return V(R * math.cos(ph), R * math.cos(ph) * sb / cb, R * math.sin(ph))

    def foot(ph, y):
        return V(R * math.cos(ph), y, R * math.sin(ph))

    # ---- checks: each ball touches the cylinder along its equator and the
    # plane at F; one above, one below; the tangents; the constant
    for C in (C1, C2):
        _check(abs(abs((C - O).dot(n)) - R) < 1e-12, "each ball touches the plane")
        for ph in [2 * math.pi * i / 24 for i in range(24)]:
            q = foot(ph, C.y)
            _check(abs((q - C).mag - R) < 1e-12 and abs((q - C).dot(V(0, 1, 0))) < 1e-12,
                   "it touches every vertical line of the cylinder at its equator")
    _check((C1 - O).dot(n) > 0 > (C2 - O).dot(n), "one ball above the plane, one below")
    _check(abs((F1 - C1).mag - R) < 1e-12 and abs((F2 - C2).mag - R) < 1e-12 and
           abs((F1 - O).dot(n)) < 1e-12 and abs((F2 - O).dot(n)) < 1e-12,
           "F₁, F₂ are where the balls touch the plane")
    V1, V2 = cut(0), cut(math.pi)
    two_a = (V1 - V2).mag
    _check(abs(two_a - 2 * yc) < 1e-12, "V₁V₂ = Q₁Q₂ = 2a")
    a_s, b_s, c_s = two_a / 2, R, (F1 - F2).mag / 2
    _check(abs(a_s * a_s - b_s * b_s - c_s * c_s) < 1e-12, "a² = b² + c²: F₁, F₂ are the foci")
    _check(abs((V1 - F1).mag + (F1 - V2).mag - two_a) < 1e-12 and
           abs((V1 - F2).mag + (F2 - V2).mag - two_a) < 1e-12, "F₁, F₂ lie on V₁V₂")
    for ph in [2 * math.pi * i / 72 for i in range(72)]:
        P = cut(ph)
        _check(abs((P - O).dot(n)) < 1e-12 and abs(math.hypot(P.x, P.z) - R) < 1e-12,
               "P lies on the plane and on the cylinder")
        d1, d2 = (P - F1).mag, (P - F2).mag
        q1, q2 = foot(ph, yc), foot(ph, -yc)
        _check(abs(d1 - (q1 - P).mag) < 1e-12 and abs(d2 - (P - q2).mag) < 1e-12,
               "PF₁ = PQ₁ and PF₂ = PQ₂")
        _check(abs(d1 + d2 - two_a) < 1e-12, "PF₁ + PF₂ = Q₁Q₂")

    # the cone of tangents from P to a ball: apex P, axis towards the centre,
    # touching the ball along a circle through F and Q
    def tangent_cone(P, F, q, C):
        dPC = (C - P).mag
        k = (C - P) / dPC
        L = math.sqrt(dPC * dPC - R * R)
        M = P + k * (L * L / dPC)
        rho = L * R / dPC
        e1 = (F - M).norm()
        e2 = k.cross(e1)
        ang = _polar_angle(F - P, q - P, k)
        return k, L, M, rho, e1, e2, ang

    def on_circle(tc, th):
        k, L, M, rho, e1, e2, ang = tc
        return M + (e1 * math.cos(th) + e2 * math.sin(th)) * rho

    for ph in [2 * math.pi * i / 36 for i in range(36)]:
        P = cut(ph)
        for F, q, C in ((F1, foot(ph, yc), C1), (F2, foot(ph, -yc), C2)):
            tc = tangent_cone(P, F, q, C)
            k, L, M, rho, e1, e2, ang = tc
            _check(abs((F - P).mag - L) < 1e-9 and abs((q - P).mag - L) < 1e-9,
                   "PF and PQ are tangents of length L")
            _check((on_circle(tc, 0) - F).mag < 1e-9 and (on_circle(tc, ang) - q).mag < 1e-9,
                   "the circle of contact passes through F and Q")
            _check((_rot_about_line(F, P, k, ang) - q).mag < 1e-9, "the turn about PC carries F onto Q")
            for th in (0.7, 2.1, 3.9, 5.3):
                X = on_circle(tc, th)
                _check(abs((X - C).mag - R) < 1e-9 and abs((X - P).dot(X - C)) < 1e-9 and
                       abs((X - P).mag - L) < 1e-9, "every line of the cone is a tangent")

    # ---- the cylinder (open, see-through), its rims and a few vertical lines
    y_top, y_bot = yc + R + 0.16, -yc - R - 0.16
    NC = 72
    for i in range(NC):
        p0, p1 = 2 * math.pi * i / NC, 2 * math.pi * (i + 1) / NC
        vs = [vertex(pos=foot(p, y), normal=V(math.cos(p), 0, math.sin(p)), color=C_A,
                     opacity=0.15)
              for p, y in ((p0, y_bot), (p1, y_bot), (p1, y_top), (p0, y_top))]
        quad(vs=vs)
    for y in (y_top, y_bot):
        ring(pos=V(0, y, 0), axis=V(0, 1, 0), radius=R, thickness=0.006, color=C_A * 1.4)
    for i in range(8):
        ph = 2 * math.pi * (i + 0.5) / 8
        curve(pos=[foot(ph, y_bot), foot(ph, y_top)], radius=0.003, color=C_A * 1.2)
    # the plane, and the cut
    m_ax = (V1 - V2).norm()
    z_ax = n.cross(m_ax).norm()
    ex, ez = a_s + 0.55, R + 0.55
    quad(vs=[vertex(pos=O + m_ax * (sx * ex) + z_ax * (sz * ez), normal=n,
                    color=V(0.62, 0.47, 0.88), opacity=0.28)
             for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    curve(pos=[cut(2 * math.pi * i / 144) for i in range(145)], radius=0.013, color=C_D)

    def scr(v):
        return V(v.dot(s_right), v.dot(s_up), 0)

    along = (s_right * scr(m_ax).norm().x + s_up * scr(m_ax).norm().y)
    across = s_right * (-scr(m_ax).norm().y) + s_up * scr(m_ax).norm().x

    # ---- the balls, their circles of contact and points of contact
    COL1, COL2 = C_C, C_B
    SHR = 0.008
    ball1 = sphere(pos=C1, radius=R - SHR, color=COL1, opacity=0.30, visible=False)
    ball2 = sphere(pos=C2, radius=R - SHR, color=COL2, opacity=0.26, visible=False)
    circ1 = ring(pos=C1, axis=V(0, 1, 0), radius=R, thickness=0.012, color=COL1, visible=False)
    circ2 = ring(pos=C2, axis=V(0, 1, 0), radius=R, thickness=0.012, color=COL2, visible=False)
    dotF1 = sphere(pos=F1, radius=0.042, color=COL1 * 1.15, visible=False)
    dotF2 = sphere(pos=F2, radius=0.042, color=COL2 * 1.15, visible=False)
    # (each focus lies on the long axis; P's focal radius reaches it from
    # below on the screen, so its name goes below the axis, on the far side)
    labF1 = _lab(F1 - across * 0.17 + along * 0.11, "F₁", COL1 * 1.2, back=True, visible=False)
    labF2 = _lab(F2 - across * 0.15 - along * 0.15, "F₂", COL2 * 1.2, back=True, visible=False)

    # ---- P, its focal radii, its vertical line
    PH0 = math.radians(112)                      # P at the front of the cut
    vline = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.005, color=V(0.78, 0.78, 0.74),
                  visible=False)
    seg1 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.018, color=COL1, visible=False)
    seg2 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.018, color=COL2, visible=False)
    gseg1 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.025, color=COL1 * 1.15, visible=False)
    gseg2 = curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.025, color=COL2 * 1.15, visible=False)
    dotP = sphere(pos=V(0, 0, 0), radius=0.048, color=C_HL, visible=False)
    dotQ1 = sphere(pos=V(0, 0, 0), radius=0.038, color=COL1 * 1.15, visible=False)
    dotQ2 = sphere(pos=V(0, 0, 0), radius=0.038, color=COL2 * 1.15, visible=False)
    labP = _lab(V(0, 0, 0), "P", C_HL, back=True, visible=False)
    labQ1 = _lab(V(0, 0, 0), "Q₁", COL1 * 1.2, back=True, visible=False)
    labQ2 = _lab(V(0, 0, 0), "Q₂", COL2 * 1.2, back=True, visible=False)

    def set_seg(c, p, q):
        c.modify(0, pos=p)
        c.modify(1, pos=q)

    def place_P(ph):
        P, q1, q2 = cut(ph), foot(ph, yc), foot(ph, -yc)
        dotP.pos, dotQ1.pos, dotQ2.pos = P, q1, q2
        set_seg(vline, foot(ph, y_bot), foot(ph, y_top))
        # labels in the diagonal gaps: the vertical line runs up and down
        # the screen, the circles across it, the focal radii rise from P
        labP.pos = P + s_right * 0.17 - s_up * 0.13
        labQ1.pos = q1 - s_right * 0.17 + s_up * 0.12
        labQ2.pos = q2 - s_right * 0.17 - s_up * 0.12
        return P, q1, q2

    NT, KF = 60, 36
    REACH = 0.965
    P0, q10, q20 = cut(PH0), foot(PH0, yc), foot(PH0, -yc)
    tcones = []
    for F, q, C, col, colc in ((F1, q10, C1, V(1.0, 0.70, 0.33), COL1),
                               (F2, q20, C2, V(0.45, 0.90, 0.80), COL2)):
        tc = tangent_cone(P0, F, q, C)
        surf = []
        for i in range(NT):
            a0, a1 = 2 * math.pi * i / NT, 2 * math.pi * (i + 1) / NT
            X0_, X1_ = on_circle(tc, a0), on_circle(tc, a1)
            nr = (X0_ - P0).cross(X1_ - P0).norm()
            X0_, X1_ = P0 + (X0_ - P0) * REACH, P0 + (X1_ - P0) * REACH
            surf.append(triangle(vs=[vertex(pos=P0, normal=nr, color=col, opacity=0.16),
                                     vertex(pos=X0_, normal=nr, color=col, opacity=0.16),
                                     vertex(pos=X1_, normal=nr, color=col, opacity=0.16)],
                                 visible=False))
        rim = curve(pos=[on_circle(tc, 2 * math.pi * i / 96) for i in range(97)], radius=0.007,
                    color=colc * 1.25, visible=False)
        fv = [vertex(pos=P0, color=col, opacity=0.5, normal=V(0, 1, 0)) for _ in range(KF + 2)]
        fan = [triangle(vs=[fv[0], fv[i], fv[i + 1]], visible=False) for i in range(1, KF + 1)]
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

    # the long axis, at the end
    axis_line = curve(pos=[V1, V2], radius=0.011, color=C_HL, visible=False)
    dotV = [sphere(pos=p, radius=0.038, color=C_HL, visible=False) for p in (V1, V2)]
    labV = [_lab(V1 + along * 0.22, "V₁", C_HL, back=True, visible=False),
            _lab(V2 - along * 0.22, "V₂", C_HL, back=True, visible=False)]
    # (above the axis, below the cut's upper arc; P's focal radii run below it)
    lab2a = _lab(V2 + (V1 - V2) * 0.4 + across * 0.15, "2a", C_HL, back=True, visible=False)

    # everything inside the view
    hw, hh = sc.range * 980 / 600, sc.range
    for p in (foot(0, y_top), foot(math.pi, y_top), foot(0, y_bot), foot(math.pi, y_bot),
              foot(math.pi / 2, y_top), foot(-math.pi / 2, y_bot)):
        q = p - sc.center
        _check(abs(q.dot(s_right)) < hw - 0.2 and abs(q.dot(s_up)) < hh - 0.12,
               "the cylinder is in view")

    row1, row2, row3 = _rows(sc, 3)

    def rows_for(ph):
        P = cut(ph)
        d1, d2 = (P - F1).mag, (P - F2).mag
        row2.text = _pre(f"PF₁ = {d1:.4f} = PQ₁        PF₂ = {d2:.4f} = PQ₂")
        row3.text = _pre(f"PF₁ + PF₂  =  PQ₁ + PQ₂  =  Q₁Q₂  =  {d1 + d2:.4f}"
                         f"      (the same on every vertical line)")

    every = [ball1, ball2, circ1, circ2, dotF1, dotF2, labF1, labF2, vline, seg1, seg2,
             gseg1, gseg2, dotP, dotQ1, dotQ2, labP, labQ1, labQ2, axis_line, lab2a]
    every += dotV + labV
    _ready(sc)
    while True:
        _show(every, False)
        for d in tcones:
            show_tcone(d, False)
        row1.text = _pre("a slanted plane cuts the cylinder in a closed curve")
        row2.text = row3.text = _pre(" ")
        _wait(1.4)
        # ---- a ball slides down the cylinder until it meets the plane ...
        row1.text = _pre("a ball as wide as the cylinder touches it along a circle;  slid down "
                         "until it touches the plane: F₁")
        ball1.visible = circ1.visible = True
        ys = y_top - R
        for t in _frames(1.8):
            y = ys + (yc - ys) * _smooth(t)
            ball1.pos = V(0, y, 0)
            circ1.pos = V(0, y, 0)
        _show([dotF1, labF1], True)
        _wait(0.6)
        # ---- ... and one slides up until it meets it from below
        row1.text = _pre("a second ball, below the plane, slid up until it touches the plane: F₂")
        ball2.visible = circ2.visible = True
        ys = y_bot + R
        for t in _frames(1.8):
            y = ys + (-yc - ys) * _smooth(t)
            ball2.pos = V(0, y, 0)
            circ2.pos = V(0, y, 0)
        _show([dotF2, labF2], True)
        _wait(0.8)
        # ---- a point P of the cut, its distances to F₁ and F₂
        P, q1, q2 = place_P(PH0)
        set_seg(seg1, P, F1)
        set_seg(seg2, P, F2)
        _show([dotP, labP, seg1, seg2], True)
        row1.text = _pre("P on the cut:  PF₁ and PF₂")
        _wait(1.4)
        _show([vline, dotQ1, dotQ2, labQ1, labQ2], True)
        row1.text = _pre("the vertical line through P touches the balls on their circles, at "
                         "Q₁ and Q₂")
        _wait(1.2)
        # ---- the tangents from P to a ball form a cone; PF turns along it
        texts = ["the tangents from P to the upper ball form a cone: all of one length.  "
                 "PF₁ turns along it onto PQ₁",
                 "the tangents from P to the lower ball form a cone too:  "
                 "PF₂ turns along it onto PQ₂"]
        for d, F, q, seg, gseg, txt in ((tcones[0], F1, q1, seg1, gseg1, texts[0]),
                                        (tcones[1], F2, q2, seg2, gseg2, texts[1])):
            row1.text = _pre(txt)
            show_tcone(d, True)
            _wait(0.8)
            for t in _frames(2.4):
                s = _smooth(t)
                set_seg(seg, P, on_circle(d["tc"], s * d["tc"][6]))
                set_fan(d, s)
            set_seg(gseg, P, q)
            gseg.visible = True
            _wait(0.8)
            show_tcone(d, False)
        rows_for(PH0)
        row1.text = _pre("so PF₁ + PF₂ is Q₁Q₂, the piece of the vertical line between the two "
                         "circles")
        _wait(1.6)
        # ---- P runs round the cut: Q₁Q₂ only slides round the cylinder
        set_seg(seg1, P, F1)
        set_seg(seg2, P, F2)
        _show([labQ1, labQ2], False)               # (they would cross F₁, F₂ on the way)
        row1.text = _pre("every vertical line crosses the two circles the same distance apart")
        for t in _frames(6.5):
            ph = PH0 + 2 * math.pi * _smooth(t)
            P, q1, q2 = place_P(ph)
            set_seg(seg1, P, F1)
            set_seg(seg2, P, F2)
            set_seg(gseg1, P, q1)
            set_seg(gseg2, P, q2)
            if int(round(t * 260)) % 3 == 0:
                rows_for(ph)
        P, q1, q2 = place_P(PH0)
        rows_for(PH0)
        # ---- the constant is the long axis: P at either end of it
        _show([axis_line, lab2a, labQ1, labQ2] + dotV + labV, True)
        row1.text = _pre("with P at V₁ and then at V₂ the two sums add to 2·V₁V₂:  the constant "
                         "is V₁V₂ = 2a,  and the cut is an ellipse with foci F₁, F₂")
        row3.text = _pre(f"PF₁ + PF₂  =  Q₁Q₂  =  V₁V₂  =  2a  =  {two_a:.4f}")
        _wait(5.0)
