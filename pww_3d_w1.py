# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d_w1.py — vpython scenes: growing and dissecting cubes, nested cube
shells, unrolled surfaces, a stack of coins and the five regular solids
(B29, C11, C16, H9, I11, I12, I18, I23).

Each scene_<ID>() is found by name by pww_3d.py / the app. Every dissection,
packing, unrolling and folding shown here is rebuilt from its rule and
checked before anything is drawn: a construction that does not close fails
at start-up instead of animating a wrong picture.
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


def _set_box(bx, lo, hi, inset=0.0):
    """Place a box exactly on [lo, hi], drawn `inset` smaller so that the
    seams between touching pieces stay visible (never more than a third of
    its thinnest side)."""
    d = hi - lo
    e = min(inset, min(d.x, d.y, d.z) / 3)
    bx.pos = (lo + hi) / 2
    bx.size = V(max(d.x - e, 1e-4), max(d.y - e, 1e-4), max(d.z - e, 1e-4))


def _overlap(lo1, hi1, lo2, hi2, eps=1e-9):
    """Do two boxes (tuples or vectors) share interior points?"""
    return all(lo1[i] < hi2[i] - eps and lo2[i] < hi1[i] - eps for i in range(3))


def _tile_check(boxes, lo, hi, N, what):
    """Every cell centre of an N × N × N grid of the box [lo, hi] lies in
    exactly one of the boxes, and the volumes add up."""
    vol = sum((b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) * (b[1][2] - b[0][2])
              for b in boxes)
    _check(abs(vol - (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])) < 1e-9,
           what + " (volumes)")
    for i in range(N):
        for j in range(N):
            for k in range(N):
                p = [lo[q] + (hi[q] - lo[q]) * ((i, j, k)[q] + 0.5) / N
                     for q in range(3)]
                inside = sum(all(b[0][q] < p[q] < b[1][q] for q in range(3))
                             for b in boxes)
                _check(inside == 1, what)


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


def _cube_compound(cells, col, unit, gap, o, ref):
    """One compound of unit cubes (cell (i, j, k) is the cube with its corner
    at o + (i, j, k)·unit), built in place with its reference point `ref`."""
    cubes = [box(pos=o + (V(*c) + V(0.5, 0.5, 0.5)) * unit,
                 size=V(1, 1, 1) * (unit - gap), color=col, visible=False)
             for c in cells]
    return compound(cubes, origin=ref, visible=False)


# ============================================================== C16

def scene_C16():
    """(n+1)³ − n³ = 3n² + 3n + 1: the n-cube of unit cubes grows by one
    layer — three n × n slabs on three faces, three rods of n cubes along
    the three edges between them, and one unit cube in the corner."""
    n = 4
    N = n + 1
    sc = new_canvas("(n + 1)³ − n³  =  3n² + 3n + 1",
                    "grow the n-cube by one layer: three n × n slabs, three rods "
                    "of n cubes, one unit cube  ·  n = 4",
                    rng=3.6, forward=V(-0.60, -0.46, -0.66), centre=V(0.15, -0.1, 0.15))
    sc.fov = 0.5
    u, gap = 0.7, 0.08
    o = V(-N / 2, -N / 2, -N / 2) * u          # the (n+1)-cube is centred

    R = range(n)
    core = {(i, j, k) for i in R for j in R for k in R}
    slabs = [{(n, j, k) for j in R for k in R},
             {(i, n, k) for i in R for k in R},
             {(i, j, n) for i in R for j in R}]
    rods = [{(i, n, n) for i in R}, {(n, j, n) for j in R}, {(n, n, k) for k in R}]
    corner = {(n, n, n)}
    layer = slabs + rods + [corner]
    full = {(i, j, k) for i in range(N) for j in range(N) for k in range(N)}

    # ---- the layer, checked cube by cube
    _check(set().union(core, *layer) == full and
           len(core) + sum(len(p) for p in layer) == len(full),
           "core, slabs, rods and corner fill the (n+1)-cube without overlap")
    _check([len(p) for p in slabs] == [n * n] * 3 and
           [len(p) for p in rods] == [n] * 3, "three slabs of n², three rods of n")
    _check(N ** 3 - n ** 3 == 3 * n * n + 3 * n + 1 == sum(len(p) for p in layer),
           "(n+1)³ − n³ = 3n² + 3n + 1")
    # each piece comes straight in from outside, along the normal of the face,
    # edge or corner it fills, without passing through a cube already there
    dirs = [V(1, 0, 0), V(0, 1, 0), V(0, 0, 1),
            V(0, 1, 1).norm(), V(1, 0, 1).norm(), V(1, 1, 0).norm(),
            V(1, 1, 1).norm()]
    placed = set(core)
    for cells, d in zip(layer, dirs):
        for step in range(1, 25):
            sh = d * (2.5 * step / 24)
            for c in cells:
                lo = (c[0] + sh.x, c[1] + sh.y, c[2] + sh.z)
                for q in placed:
                    _check(not _overlap(lo, (lo[0] + 1, lo[1] + 1, lo[2] + 1),
                                        q, (q[0] + 1, q[1] + 1, q[2] + 1)),
                           "each piece slides in freely")
        placed |= cells

    def centre_of(cells):
        cs = list(cells)
        return o + (V(sum(c[0] for c in cs), sum(c[1] for c in cs),
                      sum(c[2] for c in cs)) / len(cs) + V(0.5, 0.5, 0.5)) * u

    COL = [C_C] * 3 + [C_B] * 3 + [C_E]
    g0 = centre_of(core)
    core_cp = _cube_compound(core, C_A, u, gap, o, g0)
    pieces = []
    D = 2.3                                      # start this far out (world)
    for cells, d, col in zip(layer, dirs, COL):
        g = centre_of(cells)
        cp = _cube_compound(cells, col, u, gap, o, g)
        pieces.append((cp, g, d))

    lab_n = _lab(o + V(n * u / 2, -0.38, n * u + 0.15), "n")
    lab_n1 = _lab(o + V(N * u / 2, -0.38, N * u + 0.15), "n + 1", visible=False)
    # every piece named on itself: n² on the slabs, n on the rods, 1 on the corner
    h = n / 2
    spots = [[(N, h, h), (h, N, h), (h, h, N)],
             [(h, N, N), (N, h, N), (N, N, h)],
             [(N, N, N)]]
    lab_kinds = [[_lab(o + V(*p) * u, t, visible=False, back=True) for p in grp]
                 for grp, t in zip(spots, ("n²", "n", "1"))]

    row1, row2 = _rows(sc, 2)
    core_cp.visible = True
    _ready(sc)
    while True:
        for cp, g, d in pieces:
            cp.visible = False
            cp.pos = g + d * D
        lab_n.pos = o + V(n * u / 2, -0.38, n * u + 0.15)
        lab_n.visible, lab_n1.visible = True, False
        for grp in lab_kinds:
            for lb in grp:
                lb.visible = False
        row1.text = _pre(f"n³  =  {n}³  =  {n ** 3}")
        row2.text = _pre(" ")
        _wait(1.8)
        added = []
        sym = ["n³ + 3n²", "n³ + 3n² + 3n", "n³ + 3n² + 3n + 1  =  (n + 1)³"]
        for idx, (cp, g, d) in enumerate(pieces):
            cp.visible = True
            for t in _frames(0.9):
                cp.pos = g + d * (D * (1 - _smooth(t)))
            added.append(len(layer[idx]))
            txt = f"n³ + layer  =  {n ** 3}  +  " + " + ".join(str(a) for a in added)
            if idx == 6:
                txt += f"  =  {N ** 3}  =  {N}³"
            row1.text = _pre(txt)
            if idx == 2:                         # the front slab's edge is n too
                lab_n.pos = o + V(n * u / 2, -0.38, N * u + 0.15)
            if idx in (2, 5, 6):
                for lb in lab_kinds[(2, 5, 6).index(idx)]:
                    lb.visible = True
                row2.text = _pre(sym[(2, 5, 6).index(idx)])
            _wait(0.25 if idx not in (2, 5) else 0.7)
        lab_n.visible, lab_n1.visible = False, True
        row1.text = _pre(f"n³ + 3n² + 3n + 1  =  {n ** 3} + 3·{n * n} + 3·{n} + 1  =  "
                         f"{n ** 3} + {3 * n * n} + {3 * n} + 1  =  {N ** 3}  =  {N}³")
        _wait(5.0)
        # ---- and back out, last in first out, staggered
        for grp in lab_kinds:
            for lb in grp:
                lb.visible = False
        total, dur, lag = 2.2, 1.0, 0.2
        m = len(pieces)
        for t in _frames(total):
            T = t * total
            for k, (cp, g, d) in enumerate(pieces):
                s = min(max((T - lag * (m - 1 - k)) / dur, 0.0), 1.0)
                cp.pos = g + d * (D * _smooth(s))
        for cp, _, _ in pieces:
            cp.visible = False
        _wait(0.6)


# ============================================================== H9

def scene_H9():
    """d/dx x³ = 3x²: the x-cube grows by dx into three slabs x²·dx, three
    rods x·dx² and a corner dx³. Pulled apart, the pieces are watched as dx
    shrinks: the slabs keep their faces x², the rods and the corner vanish,
    and ΔV/dx = 3x² + 3x·dx + dx² → 3x²."""
    x, DX = 2.0, 0.55
    sc = new_canvas("d/dx x³  =  3x²",
                    "grow the cube by dx: three slabs x²·dx, three rods x·dx², "
                    "one corner dx³  ·  as dx → 0 only the slabs count",
                    rng=3.0, forward=V(-0.60, -0.46, -0.66), centre=V(0.42, 0.3, 0.42))
    sc.fov = 0.5
    o = V(-x / 2, -x / 2, -x / 2)               # the x-cube is centred
    E = [V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)]
    PAIRS = [(1, 2), (0, 2), (0, 1)]

    def pieces(dx, gs=1.0, gr=1.0, gc=1.0, ex=0.0):
        """The eight boxes (lo, hi): the x-cube, the slabs grown gs·dx, the
        rods gr·dx, the corner gc·dx, all pushed out by ex along the
        direction of the face, edge or corner they cover."""
        out = [(o, o + V(x, x, x))]
        for i in range(3):
            lo, hi = [0, 0, 0], [x, x, x]
            lo[i], hi[i] = x, x + gs * dx
            out.append((o + V(*lo) + E[i] * ex, o + V(*hi) + E[i] * ex))
        for i, j in PAIRS:
            lo, hi = [0, 0, 0], [x, x, x]
            for q in (i, j):
                lo[q], hi[q] = x, x + gr * dx
            sh = (E[i] + E[j]) * ex
            out.append((o + V(*lo) + sh, o + V(*hi) + sh))
        c = V(1, 1, 1) * ex
        out.append((o + V(x, x, x) + c, o + V(x, x, x) + V(1, 1, 1) * (gc * dx) + c))
        return out

    # ---- exact bookkeeping: the pieces tile the (x + dx)-cube
    for dx in (DX, 0.3, 0.05):
        bs = [(_vt(lo - o), _vt(hi - o)) for lo, hi in pieces(dx)]
        _tile_check(bs, (0, 0, 0), (x + dx,) * 3, 12, "the 8 pieces tile the (x+dx)-cube")
        vols = [(b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) * (b[1][2] - b[0][2]) for b in bs]
        _check(abs(vols[0] - x ** 3) < 1e-9 and
               all(abs(v - x * x * dx) < 1e-9 for v in vols[1:4]) and
               all(abs(v - x * dx * dx) < 1e-9 for v in vols[4:7]) and
               abs(vols[7] - dx ** 3) < 1e-12,
               "x³, three x²·dx, three x·dx², dx³")
        _check(abs(sum(vols) - (x + dx) ** 3) < 1e-9, "(x + dx)³")

    COL = [C_A] + [C_C] * 3 + [C_B] * 3 + [C_E]
    boxes = [box(color=c, visible=False) for c in COL]
    INSET = 0.025

    def place(dx, gs=1.0, gr=1.0, gc=1.0, ex=0.0):
        for bx, (lo, hi), g in zip(boxes, pieces(dx, gs, gr, gc, ex),
                                   [1, gs, gs, gs, gr, gr, gr, gc]):
            bx.visible = g > 1e-3
            _set_box(bx, lo, hi, INSET)

    # x and dx along the front bottom edge: the slab's side x, the rod's dx
    OFF = V(0, -0.13, 0.05)

    def front_edge(gs, gr):
        """Ends of the two parts of the front bottom edge, (x then dx)."""
        z = x + gs * DX
        return ((o + V(0, 0, z), o + V(x, 0, z)),
                (o + V(x, 0, z), o + V(x + gr * DX, 0, z)))

    dim_x = _Dim(*front_edge(0, 0)[0], OFF, "x")
    dim_dx = _Dim(*front_edge(1, 1)[1], OFF, "dx", visible=False)
    EX = 0.6                                     # how far the pieces part

    def kind_pos(dx, ex):
        """On each piece: the outer face of a slab, the middle of a rod, the
        corner cube."""
        ps = pieces(dx, 1, 1, 1, ex)
        out = []
        for i in range(3):
            lo, hi = ps[1 + i]
            out.append((lo + hi) / 2 + E[i] * ((hi - lo).dot(E[i]) / 2))
        for q in range(3):
            lo, hi = ps[4 + q]
            out.append((lo + hi) / 2)
        lo, hi = ps[7]
        out.append((lo + hi) / 2)
        return out

    texts = ["x²·dx"] * 3 + ["x·dx²"] * 3 + ["dx³"]
    kind_labs = [_lab(p, t, visible=False, back=True)
                 for p, t in zip(kind_pos(DX, EX), texts)]

    def move_labels(dx, ex):
        for lb, p in zip(kind_labs, kind_pos(dx, ex)):
            lb.pos = p

    row1, row2, row3 = _rows(sc, 3)
    place(DX, 0, 0, 0)
    _ready(sc)
    while True:
        place(DX, 0, 0, 0)
        dim_x.set(*front_edge(0, 0)[0])
        dim_x.show(True)
        dim_dx.show(False)
        for lb in kind_labs:
            lb.visible = False
        row1.text = _pre(f"V = x³        (x = {x:g})")
        row2.text = row3.text = _pre(" ")
        _wait(1.6)
        # ---- the cube grows by dx: slabs, then rods, then the corner
        for t in _frames(1.4):
            place(DX, _smooth(t), 0, 0)
            dim_x.set(*front_edge(_smooth(t), 0)[0])
        row1.text = _pre("(x + dx)³  =  x³ + 3·x²·dx")
        _wait(0.3)
        for t in _frames(1.0):
            place(DX, 1, _smooth(t), 0)
        dim_dx.show(True)
        row1.text = _pre("(x + dx)³  =  x³ + 3·x²·dx + 3·x·dx²")
        _wait(0.2)
        for t in _frames(0.7):
            place(DX, 1, 1, _smooth(t))
        row1.text = _pre("(x + dx)³  =  x³ + 3·x²·dx + 3·x·dx² + dx³")
        _wait(0.8)
        # ---- pull the pieces apart and name them
        dim_x.show(False)
        dim_dx.show(False)
        for t in _frames(1.1):
            place(DX, 1, 1, 1, EX * _smooth(t))
        move_labels(DX, EX)
        for lb in kind_labs:
            lb.visible = True
        row2.text = _pre("ΔV = (x + dx)³ − x³ = 3·x²·dx + 3·x·dx² + dx³")
        row3.text = _pre("ΔV / dx  =  3x² + 3x·dx + dx²")
        _wait(2.4)
        # ---- now let dx shrink: slabs ~ dx, rods ~ dx², corner ~ dx³
        dx_end = 0.025
        for t in _frames(5.0):
            s = _smooth(t)
            dx = DX * (1 - s) + dx_end * s
            place(dx, ex=EX)
            move_labels(dx, EX)
            for lb in kind_labs[3:]:
                lb.visible = dx > 0.12
            if int(round(t * 200)) % 3 == 0 or t == 1:
                row2.text = _pre(
                    f"dx = {dx:.3f}      ΔV = 3·x²·dx + 3·x·dx² + dx³  =  "
                    f"{3 * x * x * dx:.4f} + {3 * x * dx * dx:.4f} + {dx ** 3:.5f}")
                row3.text = _pre(
                    f"ΔV / dx  =  3x² + 3x·dx + dx²  =  {3 * x * x:g} + "
                    f"{3 * x * dx:.3f} + {dx * dx:.4f}  =  "
                    f"{3 * x * x + 3 * x * dx + dx * dx:.4f}   →   3x² = {3 * x * x:g}")
        row1.text = _pre("d/dx x³  =  lim (dx → 0)  ΔV / dx  =  3x²:   three faces x²")
        _wait(4.0)
        # ---- back to the bare cube
        for lb in kind_labs:
            lb.visible = False
        place(DX, 0, 0, 0)
        _wait(0.8)


# ============================================================== C11

def _turned_aabb(parts, turn, s, P):
    """Axis-aligned bounds of boxes (offsets lo, hi from a body centre)
    after the body is turned by turn(s) and placed at P: a safe outer
    bound for collision checks while it moves."""
    pts = []
    for lo, hi in parts:
        for c in itertools.product(*zip(_vt(lo), _vt(hi))):
            pts.append(P + turn.apply(s, V(*c)))
    return ((min(q.x for q in pts), min(q.y for q in pts), min(q.z for q in pts)),
            (max(q.x for q in pts), max(q.y for q in pts), max(q.z for q in pts)))


def scene_C11():
    """a³ + b³ = (a + b)³ − 3ab(a + b): the (a+b)-cube is the cubes a³ and
    b³, three a²b blocks and three ab² blocks. Each a²b block and the ab²
    block beside it make an a × b × (a+b) slab; the pairs go round in a
    cycle (pairing two blocks with the same third direction leaves a pair
    that is no box). The three slabs slide out along their own directions,
    leaving a³ and b³ touching at a corner, and stand up side by side:
    three congruent slabs ab(a + b)."""
    a, b = 3, 2
    s = a + b
    k = 0.52                                    # world units per unit
    sc = new_canvas("a³ + b³  =  (a + b)³ − 3ab(a + b)",
                    "lift three a × b × (a+b) slabs, each an a²b block and an ab² "
                    "block, out of the (a+b)-cube: a³ and b³ remain  ·  a = 3, b = 2",
                    rng=4.2, forward=V(-0.55, -0.40, -0.73), centre=V(1.9, -0.35, -1.4))
    sc.fov = 0.5
    o = V(-s / 2, -s / 2, -s / 2) * k

    def block(S):
        """The block whose coordinates in S run over the b-part [a, a+b] and
        the others over [0, a]; x is mirrored (x = s − p) so that from the
        viewer a³ and b³ stand side by side, not one behind the other."""
        lo = [a if i in S else 0 for i in range(3)]
        hi = [s if i in S else a for i in range(3)]
        lo[0], hi[0] = s - hi[0], s - lo[0]
        return tuple(lo), tuple(hi)

    def vol(bk):
        return (bk[1][0] - bk[0][0]) * (bk[1][1] - bk[0][1]) * (bk[1][2] - bk[0][2])

    A3, B3 = block(()), block((0, 1, 2))
    # slab i: the a²b block {i} with the ab² block {i, i+1}: a cycle
    pairs = [((0,), (0, 1)), ((1,), (1, 2)), ((2,), (2, 0))]
    slabs = [(block(p), block(q)) for p, q in pairs]
    free = [V(-1, 0, 0), V(0, 1, 0), V(0, 0, 1)]   # +p is −x after the mirror

    # ---- the dissection, checked
    allb = [A3, B3] + [bk for sl in slabs for bk in sl]
    _tile_check(allb, (0, 0, 0), (s, s, s), 2 * s, "the eight blocks tile the (a+b)-cube")
    _check(vol(A3) == a ** 3 and vol(B3) == b ** 3, "the cubes a³ and b³")
    for blk2, blk1 in slabs:
        _check(vol(blk2) == a * a * b and vol(blk1) == a * b * b, "an a²b and an ab² block")
        lo = [min(blk2[0][i], blk1[0][i]) for i in range(3)]
        hi = [max(blk2[1][i], blk1[1][i]) for i in range(3)]
        _check(vol((lo, hi)) == vol(blk2) + vol(blk1) and
               sorted(hi[i] - lo[i] for i in range(3)) == sorted([a, b, s]),
               "each pair is one a × b × (a+b) slab")
    # the other pairing fails: {x}+{x,z} and {y}+{y,z} leave {z}+{x,y}
    bad = (block((2,)), block((0, 1)))
    lo = [min(bad[0][0][i], bad[1][0][i]) for i in range(3)]
    hi = [max(bad[0][1][i], bad[1][1][i]) for i in range(3)]
    _check(vol((lo, hi)) != vol(bad[0]) + vol(bad[1]), "a non-cyclic pairing is no box")
    _check(a ** 3 + b ** 3 == s ** 3 - 3 * a * b * s == s * (a * a - a * b + b * b),
           "a³ + b³ = (a+b)³ − 3ab(a+b) = (a+b)(a² − ab + b²)")

    def world(bk):
        return o + V(*bk[0]) * k, o + V(*bk[1]) * k

    # ---- the slabs as rigid bodies about their own centres
    centres, offs = [], []
    for sl in slabs:
        lo = V(*[min(sl[0][0][i], sl[1][0][i]) for i in range(3)])
        hi = V(*[max(sl[0][1][i], sl[1][1][i]) for i in range(3)])
        c = o + (lo + hi) / 2 * k
        centres.append(c)
        offs.append([(world(bk)[0] - c, world(bk)[1] - c) for bk in sl])
    # each one stands up the same way: a across, a + b up (a²b below), b deep
    TURN = [_AxisTurn(V(0, 1, 0), math.pi / 2), _AxisTurn(V(1, 0, 0), -math.pi / 2),
            _AxisTurn(V(0, 0, 1), -math.pi / 2)]
    for i in range(3):
        lo, hi = _turned_aabb(offs[i], TURN[i], 1, V(0, 0, 0))
        _check(all(abs((hi[q] - lo[q]) - d * k) < 1e-9 for q, d in enumerate((a, s, b))),
               "every slab stands a × (a+b) × b")
        lo2, hi2 = _turned_aabb(offs[i][:1], TURN[i], 1, V(0, 0, 0))
        lo1, hi1 = _turned_aabb(offs[i][1:], TURN[i], 1, V(0, 0, 0))
        _check(abs(hi2[1] - lo1[1]) < 1e-9, "the a²b block stands under the ab² block")
    order = [1, 2, 0]                            # up, towards the viewer, to the left
    rowdir = V(0.8, 0, -0.6)
    row0, step = V(2.95, 0, -1.7), 2.15
    slot = {i: row0 + rowdir * (step * n) for n, i in enumerate(order)}
    clear = b * k + 0.12                         # each slab is b thick along its way out
    # the way out, leg by leg: (from, to, turn from, turn to, seconds).
    # Straight out first; then through free space; the quarter-turn is made
    # in place where nothing is near; then into the row.
    paths = {}
    for i in order:
        c = centres[i]
        out_ = c + free[i] * clear
        if i == 1:                               # up, then over to the right
            w = out_ + V(5.0 * k, 0, 0)
            legs = [(c, out_, 0, 0, 0.9), (out_, w, 0, 0, 0.8), (w, w, 0, 1, 0.8),
                    (w, slot[i], 1, 1, 0.9)]
        elif i == 2:                             # forward, then along the front
            w = out_ + V(7.5 * k, 0, 0)
            f_ = slot[i] + V(0, 0, 4.7 * k)
            legs = [(c, out_, 0, 0, 0.9), (out_, w, 0, 0, 0.9), (w, w, 0, 1, 0.8),
                    (w, f_, 1, 1, 0.7), (f_, slot[i], 1, 1, 0.7)]
        else:                                    # left, round the front, along it
            w1 = V(out_.x, out_.y, 4.35 * k)
            w2 = V(7.5 * k, out_.y, 4.35 * k)
            f_ = slot[i] + V(0, 0, 4.7 * k)
            legs = [(c, out_, 0, 0, 0.9), (out_, w1, 0, 0, 0.7), (w1, w2, 0, 0, 1.0),
                    (w2, w2, 0, 1, 0.7), (w2, f_, 1, 1, 0.8), (f_, slot[i], 1, 1, 0.7)]
        paths[i] = legs

    def leg_pose(leg, u):
        p0, p1, s0, s1, _ = leg
        return s0 + (s1 - s0) * u, p0 * (1 - u) + p1 * u

    # ---- every leg checked against everything where it then stands
    def solid(bk):
        lo, hi = world(bk)
        return _vt(lo), _vt(hi)

    for i in order:
        others = [solid(A3), solid(B3)]
        for j in range(3):
            if j == i:
                continue
            if order.index(j) < order.index(i):
                others.append(_turned_aabb(offs[j], TURN[j], 1, slot[j]))
            else:
                others += [solid(bk) for bk in slabs[j]]
        for leg in paths[i]:
            for q in range(1, 49):
                sv, P = leg_pose(leg, q / 48)
                lo, hi = _turned_aabb(offs[i], TURN[i], sv, P)
                for ob in others:
                    _check(not _overlap(lo, hi, *ob), "each slab leaves without hitting a piece")
        lo, hi = _turned_aabb(offs[i], _IDENT, 0, paths[i][0][1])
        _check(not _overlap(lo, hi, _vt(o), _vt(o + V(s, s, s) * k)),
               "the straight slide takes the slab clear of the cube")
    for i, j in itertools.combinations(range(3), 2):
        _check(not _overlap(*_turned_aabb(offs[i], TURN[i], 1, slot[i]),
                            *_turned_aabb(offs[j], TURN[j], 1, slot[j])), "the row is clear")
    _check(abs(slot[1].y - (s * k / 2 + o.y)) < 1e-9, "the slabs stand on the floor")

    INSET = 0.03
    box(pos=sum(world(A3), V(0, 0, 0)) / 2, size=world(A3)[1] - world(A3)[0] - V(1, 1, 1) * INSET,
        color=C_A)
    box(pos=sum(world(B3), V(0, 0, 0)) / 2, size=world(B3)[1] - world(B3)[0] - V(1, 1, 1) * INSET,
        color=C_D)
    SCOL = [C_E, C_B, C_F]
    bodies = []
    for i in range(3):
        body = _Rigid()
        for (lo, hi), shade in zip(offs[i], (1.0, 0.62)):
            bx = box(pos=centres[i] + (lo + hi) / 2, size=hi - lo - V(1, 1, 1) * INSET,
                     color=SCOL[i] * shade)
            body.add(bx, (lo + hi) / 2)
        bodies.append(body)

    # ---- labels: the edges a, b; then the pieces
    E0 = o + V(0, 0, s) * k                      # front bottom left corner
    dims = [_Dim(E0, E0 + V(b, 0, 0) * k, V(0, -0.13, 0.05), "b"),
            _Dim(E0 + V(b, 0, 0) * k, E0 + V(s, 0, 0) * k, V(0, -0.13, 0.05), "a"),
            _Dim(E0 + V(s, 0, 0) * k, E0 + V(s, a, 0) * k, V(0.1, 0, 0.06), "a",
                 lab_gap=3.2),
            _Dim(E0 + V(s, a, 0) * k, E0 + V(s, s, 0) * k, V(0.1, 0, 0.06), "b",
                 lab_gap=3.2)]

    def mid(bk):
        lo, hi = world(bk)
        return (lo + hi) / 2

    # on the front faces: of the two cubes, and of each part in the row
    lab_a3 = _lab(mid(A3) + V(0, 0, a * k / 2), "a³", back=True, visible=False)
    lab_b3 = _lab(mid(B3) + V(0, 0, b * k / 2), "b³", back=True, visible=False)
    part_labs = []
    for i in range(3):
        part_labs.append([_lab(slot[i] + TURN[i].apply(1, (lo + hi) / 2) + V(0, 0, b * k / 2),
                               t, back=True, visible=False)
                          for (lo, hi), t in zip(offs[i], ("a²b", "ab²"))])

    row1, row2, row3 = _rows(sc, 3)
    cut = a * a * b + a * b * b
    _ready(sc)
    while True:
        for i in range(3):
            bodies[i].pose(TURN[i], 0, centres[i])
        for dm in dims:
            dm.show(True)
        for lb in [lab_a3, lab_b3] + [x for pl in part_labs for x in pl]:
            lb.visible = False
        row1.text = _pre(f"(a + b)³  =  {s}³  =  {s ** 3}")
        row2.text = row3.text = _pre(" ")
        _wait(2.2)
        for dm in dims:
            dm.show(False)
        taken = []
        for i in order:
            for leg in paths[i]:
                for t in _frames(leg[4]):
                    sv, P = leg_pose(leg, _smooth(t))
                    bodies[i].pose(TURN[i], sv, P)
            taken.append(i)
            row2.text = _pre(
                "slab  =  a × b × (a + b)  =  a²b + ab²  =  "
                f"{a * a * b} + {a * b * b}  =  {cut}:      {s ** 3}"
                + "".join(f" − {cut}" for _ in taken)
                + (f"  =  {s ** 3 - 3 * cut}" if len(taken) == 3 else ""))
            _wait(0.4)
        # names only once the row stands (labels are drawn over everything)
        for lb in [lab_a3, lab_b3] + [x for pl in part_labs for x in pl]:
            lb.visible = True
        row3.text = _pre(
            f"a³ + b³  =  {a ** 3} + {b ** 3}  =  {a ** 3 + b ** 3}  =  (a + b)³ − 3ab(a + b)"
            f"  =  (a + b)(a² − ab + b²)  =  {s} · {a * a - a * b + b * b}")
        _wait(5.0)
        # ---- and back, last out first in
        for lb in [lab_a3, lab_b3] + [x for pl in part_labs for x in pl]:
            lb.visible = False
        for i in reversed(order):
            for leg in reversed(paths[i]):
                for t in _frames(leg[4] * 0.45):
                    sv, P = leg_pose(leg, 1 - _smooth(t))
                    bodies[i].pose(TURN[i], sv, P)
        row2.text = row3.text = _pre(" ")
        _wait(0.8)


# ============================================================== B29

def scene_B29():
    """1 + 7 + 19 + … + (3n² − 3n + 1) = n³: the n-cube of unit cubes is
    n nested shells; shell k (the k-cube less the (k−1)-cube) is three
    faces of the k-cube, and seen corner-on its 3k(k−1)+1 cubes are a
    centred hexagon of dots (one dot on the outer corner of each cube)."""
    n = 4
    D = V(1, 1, 1).norm()                        # the cube's diagonal, towards the viewer
    sc = new_canvas("1 + 7 + 19 + … + (3n² − 3n + 1)  =  n³",
                    "a centred hexagon of dots is three faces of a cube seen corner-on; "
                    "the hexagons nest as shells into the n-cube  ·  n = 4",
                    rng=2.9, forward=-D, centre=V(0, 0, 0))
    sc.fov = 0.35
    u, gap = 0.62, 0.07
    o = V(-n / 2, -n / 2, -n / 2) * u            # the n-cube is centred

    shells = [[(i, j, l) for i in range(n) for j in range(n) for l in range(n)
               if max(i, j, l) == k - 1] for k in range(1, n + 1)]
    # ---- the shells, checked: they fill the cube, and each is a centred
    # hexagon when seen along the diagonal
    _check(sum(len(sh) for sh in shells) == n ** 3 and
           len(set(c for sh in shells for c in sh)) == n ** 3, "the shells fill the n-cube")
    for k, sh in enumerate(shells, 1):
        _check(len(sh) == k ** 3 - (k - 1) ** 3 == 3 * k * (k - 1) + 1,
               "shell k = k³ − (k−1)³ = 3k(k−1) + 1")
        dots = [(i + 1, j + 1, l + 1) for i, j, l in sh]
        # seen along (1,1,1) a dot is fixed by (x − z, y − z): all distinct,
        # and they fill the hexagon of side k − 1 about the corner (k, k, k)
        proj = {(x - z, y - z) for x, y, z in dots}
        _check(len(proj) == len(dots), "every dot is seen")
        hexagon = {(p, q) for p in range(-(k - 1), k) for q in range(-(k - 1), k)
                   if abs(p - q) <= k - 1}
        _check(proj == hexagon, "the dots are the centred hexagon of side k − 1")
    _check(sum(3 * k * (k - 1) + 1 for k in range(1, n + 1)) == n ** 3, "∑ = n³")

    # ---- screen axes of the corner-on view
    up_s = (V(0, 1, 0) - D * V(0, 1, 0).dot(D)).norm()   # vertical on the screen
    COLS = [C_E, C_D, C_B, C_A]
    bodies = []
    for k, sh in enumerate(shells, 1):
        ref = o + V(k, k, k) * (u / 2)           # centre of the k-cube
        parts = [box(pos=o + (V(*c) + V(0.5, 0.5, 0.5)) * u, size=V(1, 1, 1) * (u - gap),
                     color=COLS[k - 1], visible=False) for c in sh]
        parts += [sphere(pos=o + V(i + 1, j + 1, l + 1) * u, radius=0.055, color=C_HL,
                         visible=False) for i, j, l in sh]
        cp = compound(parts, origin=ref, visible=False)
        body = _Rigid()
        body.add(cp, ref)                        # turned about the cube's centre
        bodies.append((body, cp))

    R_n = n * u * math.sqrt(2 / 3)               # the outer hexagon's circumradius
    lab_count = _lab(-up_s * (R_n + 0.45), "", visible=False)
    row1, row2 = _rows(sc, 2)
    row1.text = _pre("shell k  =  k³ − (k − 1)³  =  3k(k − 1) + 1  cubes, seen corner-on: "
                     "a centred hexagon of 1, 7, 19, 37 dots")
    HEX = [3 * k * (k - 1) + 1 for k in range(1, n + 1)]
    GAM = math.radians(115)                      # turned to show the back face x = 0
    turn = _AxisTurn(up_s, GAM)
    EXP = 1.0                                    # how far the shells are pulled apart
    # each shell's count on the corner of its L on the face x = 0
    shell_labs = [_lab(turn.apply(1, o + V(0, k + 0.5, k + 0.5) * u + V(-0.3, 0, 0)
                                  + D * (EXP * k)), f"{3 * (k + 1) * k + 1}",
                       back=True, visible=False) for k in range(n)]
    _check(all((0, k, k) in shells[k] for k in range(n)),
           "each label sits on a cube of its own shell")
    FAR = 3.2                                    # where a shell comes from (towards the viewer)

    def pose(k, s_turn, disp):
        body, _ = bodies[k]
        body.pose(turn, s_turn, turn.apply(s_turn, D * disp))

    def running(m):
        return " + ".join(str(h) for h in HEX[:m]) + f"  =  {m ** 3}  =  {m}³"

    _ready(sc)
    while True:
        # ---- corner-on: the shells come in, one inside the next
        for k in range(n):
            _, cp = bodies[k]
            pose(k, 0, FAR)
            cp.opacity = 0
            cp.visible = True
            lab_count.visible = False
            for t in _frames(1.0):
                s_ = _smooth(t)
                pose(k, 0, FAR * (1 - s_))
                cp.opacity = min(1.0, 1.6 * t)
            lab_count.text = f"{HEX[k]}"
            lab_count.visible = True
            row2.text = _pre(running(k + 1))
            _wait(0.9)
        lab_count.visible = False
        # ---- turned away from the corner: a cube; its shells pulled apart
        for t in _frames(2.0):
            for k in range(n):
                pose(k, _smooth(t), 0)
        _wait(0.6)
        for t in _frames(1.4):
            for k in range(n):
                pose(k, 1, EXP * k * _smooth(t))
        for lb in shell_labs:
            lb.visible = True
        _wait(2.8)
        for lb in shell_labs:
            lb.visible = False
        for t in _frames(1.2):
            for k in range(n):
                pose(k, 1, EXP * k * (1 - _smooth(t)))
        for t in _frames(1.8):
            for k in range(n):
                pose(k, 1 - _smooth(t), 0)
        _wait(0.8)
        # ---- corner-on again: peel the shells off, the outer one first
        for k in reversed(range(n)):
            _, cp = bodies[k]
            for t in _frames(0.9):
                pose(k, 0, FAR * _smooth(t))
                cp.opacity = 1 - t
            cp.visible = False
            row2.text = _pre(running(k) if k else " ")
            if k:
                lab_count.text = f"{HEX[k - 1]}"
            lab_count.visible = k > 0
            _wait(0.6)
        lab_count.visible = False
        _wait(0.6)


# ============================================================== I12

def scene_I12():
    """V = Bh (Cavalieri): a stack of N equal coins (discs of area B,
    thickness h/N) pushed sideways. Every coin only slides, so every layer
    keeps its area B and the stack keeps its volume N·B·h/N = Bh. As the
    coins get thinner, the slanted stack becomes the oblique cylinder and
    the straight one the right cylinder: same base, same height, equal
    sections at every height, equal volumes."""
    k = 0.85                                     # world units per unit; r = 1
    r_u, h_u = 1.0, 3.5
    r, h = r_u * k, h_u * k
    sc = new_canvas("V = Bh   (Cavalieri)",
                    "push a stack of coins sideways: every layer keeps its area B, "
                    "so the slanted stack keeps the volume B·h",
                    rng=2.45, forward=V(0, -0.27, -0.963), centre=V(0.25, -0.1, 0))
    sc.fov = 0.22
    y0 = -h / 2
    xL, xR = -2.25, 0.55                         # bottom centres of the two stacks
    PUSH = 1.7                                   # how far the top of the right stack moves
    LEVELS = [8, 16, 32]

    def shift(N, j, p=1.0):
        """Sideways shift of coin j of N when the push has gone p of the way:
        proportional to the height of the coin's middle (a shear)."""
        return p * PUSH * (j + 0.5) / N

    # ---- checks: congruent coins, the push only slides them, every layer
    # is a disc of area B, and N coins hold N·B·h/N = Bh
    for N in LEVELS:
        t = h / N
        _check(abs(N * math.pi * r * r * t - math.pi * r * r * h) < 1e-12, "N·B·h/N = Bh")
        for j in range(N):
            # the coin's centre lies on the axis of the oblique cylinder
            yc = y0 + (j + 0.5) * t
            _check(abs(shift(N, j) - PUSH * (yc - y0) / h) < 1e-12,
                   "every coin is centred on the slanted axis")
        if N > LEVELS[0]:
            for j in range(N // 2):              # a coin splits into two halves
                _check(abs((shift(N, 2 * j) + shift(N, 2 * j + 1)) / 2 - shift(N // 2, j))
                       < 1e-12, "the halves of a coin straddle its position")

    COLS = [C_D, C_C]
    stacks = {}
    for N in LEVELS:
        t = h / N
        L = [cylinder(pos=V(xL, y0 + j * t, 0), axis=V(0, t, 0), radius=r,
                      color=COLS[j % 2], visible=False) for j in range(N)]
        Rr = [cylinder(pos=V(xR, y0 + j * t, 0), axis=V(0, t, 0), radius=r,
                       color=COLS[j % 2], visible=False) for j in range(N)]
        stacks[N] = (L, Rr)

    def show_level(N, flag):
        for c in stacks[N][0] + stacks[N][1]:
            c.visible = flag

    def place_right(N, xs):
        for c, x in zip(stacks[N][1], xs):
            c.pos = V(xR + x, c.pos.y, 0)

    # ---- the smooth solids: right and oblique cylinder, same base, same height
    # (drawn a hair wider than the coins, which they replace)
    m = PUSH / h
    rs = r * 1.006
    SOL = C_D * 0.92
    right_cyl = cylinder(pos=V(xL, y0, 0), axis=V(0, h, 0), radius=rs, color=SOL,
                         visible=False)
    NQ = 72
    parts = []
    for i in range(NQ):
        vs = []
        for ph in (2 * math.pi * i / NQ, 2 * math.pi * (i + 1) / NQ):
            nrm = V(math.cos(ph), -m * math.cos(ph), math.sin(ph)).norm()
            for y in (y0, y0 + h):
                vs.append(vertex(pos=V(xR + (y - y0) * m + rs * math.cos(ph), y,
                                       rs * math.sin(ph)), normal=nrm, color=SOL))
        parts.append(quad(vs=[vs[0], vs[2], vs[3], vs[1]], visible=False))
    parts += [cylinder(pos=V(xR + (yy - y0) * m, yy - 0.001, 0), axis=V(0, 0.002, 0),
                       radius=rs, color=SOL, visible=False) for yy in (y0, y0 + h)]
    oblique = compound(parts, origin=V(xR + PUSH / 2, 0, 0), visible=False)
    rims = [ring(pos=V(xL, yy, 0), axis=V(0, 1, 0), radius=rs, thickness=0.012,
                 color=C_HL, visible=False) for yy in (y0, y0 + h)]
    rims += [ring(pos=V(xR + (yy - y0) * m, yy, 0), axis=V(0, 1, 0), radius=rs,
                  thickness=0.012, color=C_HL, visible=False) for yy in (y0, y0 + h)]

    def show_smooth(flag, op=1.0):
        for o_ in [right_cyl, oblique]:
            o_.visible = flag
            o_.opacity = op
        for q in rims:
            q.visible = flag

    # ---- the moving level: a plane through both stacks, and its two sections
    plane = box(pos=V(0, 0, 0), size=V(xR + PUSH - xL + 2 * r + 0.5, 0.004, 2.6),
                color=C_HL, opacity=0.12, visible=False)
    secL = cylinder(pos=V(xL, 0, 0), axis=V(0, 0.016, 0), radius=r * 1.04, color=C_HL,
                    visible=False)
    secR = cylinder(pos=V(xR, 0, 0), axis=V(0, 0.016, 0), radius=r * 1.04, color=C_HL,
                    visible=False)

    def section(y, x_right, flag=True):
        plane.pos = V((xL + xR + PUSH) / 2, y, 0)
        secL.pos = V(xL, y - 0.008, 0)
        secR.pos = V(xR + x_right, y - 0.008, 0)
        for o_ in (plane, secL, secR):
            o_.visible = flag

    # ---- labels: B on both bases, h beside both stacks
    labs = [_lab(V(xL, y0 - 0.32, r * 0.7), "B"), _lab(V(xR, y0 - 0.32, r * 0.7), "B"),
            _Dim(V(xL - r - 0.18, y0, 0), V(xL - r - 0.18, y0 + h, 0), V(-0.1, 0, 0), "h",
                 lab_gap=2.6),
            _Dim(V(xR + PUSH + r + 0.18, y0, 0), V(xR + PUSH + r + 0.18, y0 + h, 0),
                 V(0.1, 0, 0), "h", lab_gap=2.6)]
    labs[-1].show(False)

    row1, row2, row3 = _rows(sc, 3)
    B = math.pi * r_u ** 2
    N0 = LEVELS[0]
    show_level(N0, True)
    _ready(sc)
    while True:
        show_smooth(False)
        section(0, 0, False)
        for N in LEVELS[1:]:
            show_level(N, False)
        show_level(N0, True)
        place_right(N0, [0] * N0)
        labs[1].pos = V(xR, y0 - 0.32, r * 0.7)
        labs[3].show(False)
        row1.text = _pre(f"{N0} equal coins:  each a disc of area B = πr² and thickness h/{N0}"
                         f"      (r = 1,  h = {h_u:g})")
        row2.text = row3.text = _pre(" ")
        _wait(1.8)
        # ---- push the right stack: each coin slides, by more the higher it is
        for t in _frames(2.2):
            p = _smooth(t)
            place_right(N0, [shift(N0, j, p) for j in range(N0)])
            labs[1].pos = V(xR + shift(N0, 0, p), y0 - 0.32, r * 0.7)
        labs[3].show(True)
        row1.text = _pre(f"the same {N0} coins, only slid sideways:   both stacks  "
                         f"{N0} · B · h/{N0}  =  B·h")
        _wait(0.6)
        # ---- a level moving up and down: the same area B on both sides
        T = h / N0
        for t in _frames(5.0):
            y = y0 + 0.012 + (h - 0.024) * (0.5 - 0.5 * math.cos(2 * math.pi * t))
            j = min(int((y - y0) / T), N0 - 1)
            section(y, shift(N0, j))
            if int(round(t * 200)) % 3 == 0:
                row2.text = _pre(f"height y = {(y - y0) / k:5.3f}:      left section B = πr² = "
                                 f"{B:.4f}      right section B = πr² = {B:.4f}")
        section(0, 0, False)
        row2.text = _pre(" ")
        # ---- thinner coins: each splits in two and the halves slide apart
        for N in LEVELS[1:]:
            half = N // 2
            parents = [shift(half, j) for j in range(half)]
            place_right(N, [parents[j // 2] for j in range(N)])
            show_level(N, True)
            show_level(half, False)
            for t in _frames(1.0):
                s_ = _smooth(t)
                place_right(N, [parents[j // 2] * (1 - s_) + shift(N, j) * s_
                                for j in range(N)])
            row1.text = _pre(f"{N} thinner coins, each B · h/{N}:   both stacks  "
                             f"{N} · B · h/{N}  =  B·h")
            _wait(0.5)
        # ---- in the limit: the right and the oblique cylinder
        for t in _frames(1.2):
            show_smooth(True, max(t, 0.02))
        show_level(LEVELS[-1], False)
        row1.text = _pre("thinner and thinner coins:  a right and an oblique cylinder, "
                         "same base B, same height h,  the same area B at every level")
        NL = LEVELS[-1]
        for t in _frames(4.0):
            y = y0 + 0.012 + (h - 0.024) * (0.5 - 0.5 * math.cos(2 * math.pi * t))
            section(y, PUSH * (y - y0) / h)
            if int(round(t * 160)) % 3 == 0:
                row2.text = _pre(f"height y = {(y - y0) / k:5.3f}:      left section B = "
                                 f"{B:.4f}      right section B = {B:.4f}")
        section(0, 0, False)
        row2.text = _pre(" ")
        row3.text = _pre(f"V(right)  =  V(oblique)  =  B·h  =  π · {h_u:g}  =  {B * h_u:.4f}")
        _wait(4.0)
        # ---- back to the first stack
        show_smooth(False)
        place_right(N0, [shift(N0, j) for j in range(N0)])
        show_level(N0, True)
        show_level(NL, False)
        labs[3].show(False)
        for t in _frames(1.5):
            p = 1 - _smooth(t)
            place_right(N0, [shift(N0, j, p) for j in range(N0)])
            labs[1].pos = V(xR + shift(N0, 0, p), y0 - 0.32, r * 0.7)
        _wait(0.6)


# ============================================================== I23

def scene_I23():
    """S = 2πrh + 2πr²: the lids swing open on a hinge at the front point
    of their rims; then the side unrolls about the front line through both
    hinges. Every intermediate side is part of a cylinder through that line,
    its cross-section an arc of length 2πr whose curvature goes from 1/r to
    0, so the side never stretches: it ends as a flat 2πr × h rectangle,
    the lids as two discs πr² touching its long sides."""
    k = 0.9                                      # world units per unit; r = 1
    r_u, h_u = 1.0, 2.4
    r, h = r_u * k, h_u * k
    sc = new_canvas("S = 2πrh + 2πr²",
                    "unroll the side into a 2πr × h rectangle and add the two discs",
                    rng=3.25, forward=V(-0.2, -0.27, -0.94), centre=V(0, 0, -0.25))
    sc.fov = 0.28
    y0 = -h / 2
    NS = 96
    S_ = [-math.pi * r + 2 * math.pi * r * i / NS for i in range(NS + 1)]

    def side_pt(s_, y, kap):
        """The point at arc length s_ from the front line on the side bent to
        curvature kap (a cylinder of radius 1/kap through the front line,
        bulging away from the viewer); kap = 0 is the flat sheet."""
        if kap < 1e-9:
            return V(s_, y, 0)
        return V(math.sin(kap * s_) / kap, y, -(1 - math.cos(kap * s_)) / kap)

    def side_nrm(s_, kap):
        return V(math.sin(kap * s_), 0, math.cos(kap * s_))

    # ---- checks: no stretching, the front line fixed, closed at the start,
    # flat at the end
    for kap in (1 / r, 0.6 / r, 0.2 / r, 0.0):
        for s_ in S_[::8]:
            ds = 1e-6
            d = (side_pt(s_ + ds, 0, kap) - side_pt(s_ - ds, 0, kap)) / (2 * ds)
            _check(abs(d.mag - 1) < 1e-6, "the side is bent, never stretched")
        _check((side_pt(0, y0, kap) - V(0, y0, 0)).mag < 1e-12, "the front line stays put")
    _check((side_pt(math.pi * r, 0, 1 / r) - side_pt(-math.pi * r, 0, 1 / r)).mag < 1e-9,
           "closed into the cylinder at the start")
    _check(abs(side_pt(math.pi * r, 0, 0).x - math.pi * r) < 1e-12, "flat: 2πr wide")

    # ---- the side: quads on shared vertices, one column per step of arc
    COL_SIDE, COL_LID = C_A, C_B
    bot = [vertex(pos=V(0, 0, 0), color=COL_SIDE) for _ in S_]
    top = [vertex(pos=V(0, 0, 0), color=COL_SIDE) for _ in S_]
    for i in range(NS):
        quad(vs=[bot[i], bot[i + 1], top[i + 1], top[i]])
    rim_b = curve(pos=[V(0, 0, 0)] * len(S_), radius=0.018, color=C_D)
    rim_t = curve(pos=[V(0, 0, 0)] * len(S_), radius=0.018, color=C_D)
    seams = [curve(pos=[V(0, 0, 0), V(0, 1, 0)], radius=0.012, color=C_HL) for _ in range(2)]

    def set_side(kap):
        for i, s_ in enumerate(S_):
            nb = side_nrm(s_, kap)
            pb, pt = side_pt(s_, y0, kap), side_pt(s_, y0 + h, kap)
            bot[i].pos, bot[i].normal = pb, nb
            top[i].pos, top[i].normal = pt, nb
            rim_b.modify(i, pos=pb)
            rim_t.modify(i, pos=pt)
        for sv, sm in zip((S_[0], S_[-1]), seams):
            sm.modify(0, pos=side_pt(sv, y0, kap))
            sm.modify(1, pos=side_pt(sv, y0 + h, kap))

    # ---- the lids: discs hinged at the front point of each rim
    TH = 0.03
    hinge = [V(0, y0 + h, 0), V(0, y0, 0)]
    lid_turn = [_AxisTurn(V(1, 0, 0), math.pi / 2), _AxisTurn(V(1, 0, 0), -math.pi / 2)]
    lid_n0 = [V(0, 1, 0), V(0, -1, 0)]
    lids = [cylinder(pos=V(0, 0, 0), axis=V(0, TH, 0), radius=r, color=COL_LID)
            for _ in range(2)]

    def set_lids(s_):
        for lid, H, tr, n0 in zip(lids, hinge, lid_turn, lid_n0):
            c = H + tr.apply(s_, V(0, 0, -r))
            nn = tr.apply(s_, n0)
            lid.pos = c - nn * (TH / 2)
            lid.axis = nn * TH
    for H, tr, n0 in zip(hinge, lid_turn, lid_n0):
        c1 = H + tr.apply(1, V(0, 0, -r))
        _check(abs(c1.z) < 1e-12 and abs(abs(c1.y - H.y) - r) < 1e-12 and
               abs(tr.apply(1, n0).z - 1) < 1e-12,
               "each lid lands flat, touching the rectangle at its hinge")

    # ---- labels
    lab_r = _lab(V(0.14, y0 + h + 0.1, -r / 2), "r", back=True)
    rad = curve(pos=[V(0, y0 + h + TH / 2 + 0.005, -r), V(0, y0 + h + TH / 2 + 0.005, 0)],
                radius=0.012, color=C_HL)
    dim_h0 = _Dim(V(-r, y0, -r), V(-r, y0 + h, -r), V(-0.13, 0, 0), "h", lab_gap=2.6)
    dim_h1 = _Dim(V(-math.pi * r, y0, 0), V(-math.pi * r, y0 + h, 0), V(-0.13, 0, 0), "h",
                  lab_gap=2.6)
    lab_2pr = _lab(V(-math.pi * r * 0.62, y0 - 0.28, 0), "2πr")
    lab_disc = [_lab(V(0, y0 + h + r, 0.05), "πr²", back=True),
                _lab(V(0, y0 - r, 0.05), "πr²", back=True)]
    final_labs = [lab_2pr] + lab_disc

    def show_start(flag):
        lab_r.visible = rad.visible = flag
        dim_h0.show(flag)

    def show_final(flag):
        for lb in final_labs:
            lb.visible = flag
        dim_h1.show(flag)

    row1, row2, row3 = _rows(sc, 3)
    set_side(1 / r)
    set_lids(0)
    show_final(False)
    _ready(sc)
    while True:
        set_side(1 / r)
        set_lids(0)
        show_start(True)
        row1.text = _pre(f"r = {r_u:g},  h = {h_u:g}:   a cylinder, its side and two lids")
        row2.text = row3.text = _pre(" ")
        _wait(2.0)
        show_start(False)
        # ---- the lids swing open on their hinges
        for t in _frames(1.8):
            set_lids(_smooth(t))
        _wait(0.3)
        # ---- the side unrolls: curvature 1/r → 0, arcs keep their length
        for t in _frames(4.0):
            u = _smooth(t)
            kap = (1 - u) / r
            set_side(kap)
            if int(round(t * 160)) % 3 == 0:
                rho = "∞" if kap < 1e-9 else f"{1 / kap / k:.3f}"
                row2.text = _pre(f"the side lies on a cylinder of radius ρ = {rho}:  its rims are "
                                 f"arcs of length 2πr = {2 * math.pi * r_u:.4f}, its height stays h")
        set_side(0.0)
        show_final(True)
        side, discs = 2 * math.pi * r_u * h_u, 2 * math.pi * r_u ** 2
        row2.text = _pre(f"rectangle 2πr × h = {side:.4f}      two discs 2·πr² = {discs:.4f}")
        row3.text = _pre(f"S  =  2πrh + 2πr²  =  2πr(h + r)  =  {side + discs:.4f}")
        _wait(5.5)
        # ---- and roll it up again
        show_final(False)
        row2.text = row3.text = _pre(" ")
        for t in _frames(2.4):
            set_side((_smooth(t)) / r)
        for t in _frames(1.4):
            set_lids(1 - _smooth(t))
        _wait(0.8)


# ============================================================== I11

def scene_I11():
    """S = πrl: cut the side of a cone along a slant line and open it. Every
    intermediate surface is part of a cone with the same apex and the same
    slant l, its rim an arc of length 2πr on a circle of radius ρ growing
    from r to l, so the side never stretches; at ρ = l it lies flat: a
    sector of radius l and arc 2πr, which is 2πr/2πl of the disc πl²."""
    k = 1.0
    r_u, l_u = 1.0, 2.5
    r, l = r_u * k, l_u * k
    TH = 2 * math.pi * r / l                     # the sector's angle
    sc = new_canvas("S = πrl",
                    "unroll the side of the cone: a sector of radius l and arc 2πr  ·  "
                    "r = 1,  l = 2.5",
                    rng=2.55, forward=V(0, -0.62, -0.785), centre=V(0, 0.05, 0.15))
    sc.fov = 0.32
    A = V(0, 0.95, 0)                            # the apex stays put
    PH0 = math.pi / 2                            # the front line (towards the viewer)
    NS = 120
    PSI = [-TH / 2 + TH * i / NS for i in range(NS + 1)]   # angle in the sector

    def gen_dir(psi, rho):
        """Unit vector along the slant line that sits at angle psi in the
        sector, on the cone of slant l whose rim has radius rho."""
        sb = rho / l
        cb = math.sqrt(max(1 - sb * sb, 0.0))
        ph = PH0 + psi * l / rho                 # arc rho·Δφ = l·Δψ
        return V(sb * math.cos(ph), -cb, sb * math.sin(ph))

    def nrm(psi, rho):
        sb = rho / l
        cb = math.sqrt(max(1 - sb * sb, 0.0))
        ph = PH0 + psi * l / rho
        return V(cb * math.cos(ph), sb, cb * math.sin(ph))

    # ---- checks: slant lines keep their length l and the rim its length
    # 2πr at every stage; closed at ρ = r, flat at ρ = l
    for rho in (r, 1.4 * r, 2.0 * r, l):
        rim = [A + gen_dir(p, rho) * l for p in PSI]
        arc = sum((q - p).mag for p, q in zip(rim, rim[1:]))
        _check(abs(arc - 2 * math.pi * r) < 2e-3 * l, "the rim keeps the length 2πr")
        _check(all(abs((q - A).mag - l) < 1e-12 for q in rim), "every slant line keeps length l")
        _check(abs(gen_dir(0, rho).dot(nrm(0, rho))) < 1e-12, "normals are normal")
    _check((gen_dir(TH / 2, r) - gen_dir(-TH / 2, r)).mag < 1e-12, "closed into the cone")
    _check(all(abs(gen_dir(p, l).y) < 1e-12 for p in PSI), "flat at the end")
    _check(abs(0.5 * l * 2 * math.pi * r - math.pi * l * l * (r / l)) < 1e-12,
           "½·l·2πr = πl²·(2πr / 2πl) = πrl")

    # ---- the side: a fan of triangles from the apex, on shared rim vertices
    COL = C_C
    rimv = [vertex(pos=A, color=COL) for _ in PSI]
    apexv = [vertex(pos=A, color=COL) for _ in range(NS)]
    for i in range(NS):
        triangle(vs=[apexv[i], rimv[i], rimv[i + 1]])
    rim_c = curve(pos=[A] * len(PSI), radius=0.016, color=C_D)
    NG = 6
    gens = [curve(pos=[A, A], radius=0.008, color=C_HL * 0.85) for _ in range(NG + 1)]

    def set_cone(rho):
        for i, p in enumerate(PSI):
            q = A + gen_dir(p, rho) * l
            rimv[i].pos, rimv[i].normal = q, nrm(p, rho)
            rim_c.modify(i, pos=q)
        for i in range(NS):
            apexv[i].normal = nrm((PSI[i] + PSI[i + 1]) / 2, rho)
        for j, g in enumerate(gens):
            g.modify(1, pos=A + gen_dir(-TH / 2 + TH * j / NG, rho) * l)

    # ---- labels: the rim 2πr and a slant line l, carried along as it opens;
    # at the end the disc of radius l around the sector, dashed
    PSI_L = -TH / 8                              # a slant line at the front right

    def lab_pos(rho):
        return (A + gen_dir(PSI_L, rho) * (l * 0.55) + V(0.13, 0.08, 0),
                A + gen_dir(0, rho) * l + V(0, -0.24, 0.12))

    # (italic letters: a lone upright l reads as a bar)
    lab_l = _lab(lab_pos(r)[0], "<i>l</i>", back=True)
    lab_rim = _lab(lab_pos(r)[1], "2π<i>r</i>", back=True)
    NDASH = 72
    dashes = []
    for i in range(NDASH):
        if i % 2:
            continue
        a0, a1 = 2 * math.pi * i / NDASH, 2 * math.pi * (i + 1) / NDASH
        dashes.append(curve(pos=[A + V(l * math.cos(a), 0, l * math.sin(a))
                                 for a in (a0, (a0 + a1) / 2, a1)],
                            radius=0.008, color=C_HL * 0.7, visible=False))
    lab_full = _lab(A + V(0, 0, -l - 0.3), "2π<i>l</i>", visible=False)
    lab_sec = _lab(A + V(-0.55, 0, l * 0.5), "π<i>rl</i>", back=True, visible=False)

    def move_labels(rho):
        lab_l.pos, lab_rim.pos = lab_pos(rho)

    def show_final(flag):
        for d in dashes:
            d.visible = flag
        for lb in (lab_full, lab_sec):
            lb.visible = flag

    row1, row2, row3 = _rows(sc, 3)
    set_cone(r)
    show_final(False)
    _ready(sc)
    while True:
        set_cone(r)
        move_labels(r)
        row1.text = _pre(f"a cone: base radius r = {r_u:g}, slant height l = {l_u:g};  "
                         "cut along a slant line at the back")
        row2.text = row3.text = _pre(" ")
        _wait(2.2)
        # ---- open it: cones of the same slant l, rim radius ρ from r to l
        b0, b1 = math.asin(r / l), math.pi / 2
        for t in _frames(4.2):
            beta = b0 + (b1 - b0) * _smooth(t)
            rho = l * math.sin(beta)
            set_cone(rho)
            move_labels(rho)
            if int(round(t * 168)) % 3 == 0:
                row2.text = _pre(f"slant lines stay l = {l_u:.3f};   the rim stays 2πr = "
                                 f"{2 * math.pi * r_u:.4f}, now an arc of a circle 2πρ, "
                                 f"ρ = {rho / k:.3f}")
        set_cone(l)
        move_labels(l)
        show_final(True)
        row2.text = _pre("a sector of radius l and arc 2πr:  the part 2πr / 2πl = r/l "
                         "of the disc πl²")
        row3.text = _pre(f"S  =  πl² · r/l  =  ½ · l · 2πr  =  πrl  =  {math.pi * r_u * l_u:.4f}")
        _wait(5.5)
        show_final(False)
        row2.text = row3.text = _pre(" ")
        for t in _frames(2.6):
            beta = b1 - (b1 - b0) * _smooth(t)
            set_cone(l * math.sin(beta))
            move_labels(l * math.sin(beta))
        set_cone(r)
        move_labels(r)
        _wait(0.8)


# ============================================================== I18

_PHI = (1 + 5 ** 0.5) / 2


def _platonic(n, m):
    """Vertices (centred) of the regular solid with m regular n-gons at
    every vertex, and its exact volume for edge 1."""
    p, q = _PHI, 1 / _PHI
    cyc = (lambda a, b, c: [(a, b, c), (b, c, a), (c, a, b)])
    if (n, m) == (3, 3):
        vs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        vol = 1 / (6 * math.sqrt(2))
    elif (n, m) == (3, 4):
        vs = [t for a in (1, -1) for t in cyc(a, 0, 0)]
        vol = math.sqrt(2) / 3
    elif (n, m) == (4, 3):
        vs = list(itertools.product((1, -1), repeat=3))
        vol = 1.0
    elif (n, m) == (3, 5):
        vs = [t for a in (1, -1) for b in (p, -p) for t in cyc(0, a, b)]
        vol = 5 * (3 + math.sqrt(5)) / 12
    else:
        vs = list(itertools.product((1, -1), repeat=3))
        vs += [t for a in (q, -q) for b in (p, -p) for t in cyc(0, a, b)]
        vol = (15 + 7 * math.sqrt(5)) / 4
    return [V(*v) for v in vs], vol


def _hull_faces(verts):
    """Faces of a convex polyhedron (index lists in cyclic order) from its
    vertices: the planes through three vertices with all others behind."""
    found = {}
    for i, j, l in itertools.combinations(range(len(verts)), 3):
        nr = (verts[j] - verts[i]).cross(verts[l] - verts[i])
        if nr.mag < 1e-9:
            continue
        nr = nr.norm()
        d = nr.dot(verts[i])
        if d < 0:
            nr, d = -nr, -d
        side = [nr.dot(v) - d for v in verts]
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


def _fan_beta(n, theta):
    """Azimuth step between neighbouring edges of n-gons whose edges lie on
    the cone of half-angle theta about −y and make the polygon's angle α:
    cos α = sin²θ·cos β + cos²θ."""
    alpha = math.pi * (n - 2) / n
    st, ct = math.sin(theta), math.cos(theta)
    if st > 1 - 1e-13:
        return alpha
    return math.acos(max(-1.0, min(1.0, (math.cos(alpha) - ct * ct) / (st * st))))


def _fan(n, m, theta, phi_c, s):
    """m regular n-gons (side s) round a vertex at the origin. Their edges
    from the vertex lie on the cone of half-angle theta about −y, at
    azimuths phi_c + (k − m/2)·β, neighbours making the polygon's angle α
    (so β = α when theta = 90°: flat). Returns the polygons (point lists
    starting at the vertex) and their upward normals."""
    st, ct = math.sin(theta), math.cos(theta)
    beta = _fan_beta(n, theta)
    rays = [V(st * math.cos(ph), -ct, st * math.sin(ph))
            for ph in [phi_c + (k - m / 2) * beta for k in range(m + 1)]]
    Rc = s / (2 * math.sin(math.pi / n))
    polys, nrms = [], []
    for k in range(m):
        u, w = rays[k], rays[k + 1]
        b = (u + w).norm()
        wp = (w - b * w.dot(b)).norm()
        c = b * Rc
        polys.append([c + (b * -math.cos(2 * math.pi * j / n)
                           + wp * math.sin(2 * math.pi * j / n)) * Rc for j in range(n)])
        nr = u.cross(w).norm()
        nrms.append(nr if nr.y >= 0 else -nr)
    return polys, nrms, rays


def _theta_closed(n, m):
    """Half-angle of the cone on which the edges of m n-gons meet up."""
    alpha = math.pi * (n - 2) / n
    return math.asin(min(1.0, math.sin(alpha / 2) / math.sin(math.pi / m)))


def scene_I18():
    """Only five regular solids: m ≥ 3 regular n-gons meet at a vertex, and
    they close up into a corner only if their angles add to less than
    360°: m·(180° − 360°/n) < 360°, i.e. (m − 2)(n − 2) < 4, i.e.
    1/m + 1/n > 1/2. That leaves three, four or five triangles, three
    squares and three pentagons; six triangles, four squares and three
    hexagons lie flat, anything more overlaps. Each corner that closes
    grows into its solid."""
    sc = new_canvas("only five regular solids:   1/m + 1/n > 1/2",
                    "m regular n-gons at a corner close up only if their angles add to "
                    "less than 360°",
                    rng=3.55, forward=V(0, -0.574, -0.819), centre=V(0, 0.15, 0))
    sc.fov = 0.4
    RC = 0.85                                    # circumradius of every solid
    XS = [-3.55, -0.77, 1.6, 3.93]               # columns, packed by their widths
    YT = [2.45, -1.15]                           # height of the corners, two rows
    PHC = -math.pi / 2                           # the gap opens towards the viewer
    COLN = {3: C_A, 4: C_B, 5: C_C, 6: C_F}
    # (n, m, column, row): triangles 3, 4, 5, 6 in the first row; squares 3,
    # 4, pentagons 3, hexagons 3 in the second
    FIGS = [(3, 3, 0, 0), (3, 4, 1, 0), (3, 5, 2, 0), (3, 6, 3, 0),
            (4, 3, 0, 1), (4, 4, 1, 1), (5, 3, 2, 1), (6, 3, 3, 1)]

    # ---- the arithmetic of the corner
    for n, m, _, _ in FIGS:
        tot = m * (180 - 360 / n)
        _check((tot < 360) == ((m - 2) * (n - 2) < 4) == (1 / m + 1 / n > 0.5 + 1e-12),
               "m·angle < 360° ⟺ (m−2)(n−2) < 4 ⟺ 1/m + 1/n > 1/2")
    closes = [(n, m) for n in range(3, 12) for m in range(3, 12) if (m - 2) * (n - 2) < 4]
    _check(sorted(closes) == [(3, 3), (3, 4), (3, 5), (4, 3), (5, 3)],
           "exactly five (n, m) close up")

    figs = []
    ext = {}
    for n, m, col, row in FIGS:
        P = V(XS[col], YT[row], 0)
        alpha = math.pi * (n - 2) / n
        flat = abs(m * alpha - 2 * math.pi) < 1e-9
        if flat:
            s = 1.15 / {3: 1.0, 4: math.sqrt(2), 6: 2.0}[n]
            thf = math.pi / 2
        else:
            verts, vol1 = _platonic(n, m)
            e0 = min((a - b).mag for a, b in itertools.combinations(verts, 2))
            s = RC / verts[0].mag * e0           # side of the solid of circumradius RC
            thf = _theta_closed(n, m)
        # the fan, checked flat and closed
        for th in (math.pi / 2, (math.pi / 2 + thf) / 2, thf):
            polys, _, _ = _fan(n, m, th, PHC, s)
            for k, pts in enumerate(polys):
                _check(pts[0].mag < 1e-12, "every polygon has its corner at the vertex")
                for j in range(n):
                    _check(abs((pts[(j + 1) % n] - pts[j]).mag - s) < 1e-9,
                           "regular polygons of side s")
                if k:
                    _check((polys[k - 1][1] - pts[n - 1]).mag < 1e-9,
                           "neighbours share their edge")
        flat_polys, _, _ = _fan(n, m, math.pi / 2, PHC, s)
        xs_ = [P.x + q.x for pts in flat_polys for q in pts]
        ext[(col, row)] = (min(xs_), max(xs_))
        polys, _, _ = _fan(n, m, thf, PHC, s)
        _check((polys[-1][1] - polys[0][n - 1]).mag < 1e-9,
               "the last edge meets the first: the corner is closed (or flat)")
        fig = dict(n=n, m=m, P=P, s=s, thf=thf, flat=flat, col=COLN[n])
        if not flat:
            # ---- the solid, from its vertices; its top corner is the folded fan
            vs = [v * (RC / verts[0].mag) for v in verts]
            faces = _hull_faces(vs)
            _check(len(faces) * n == len(vs) * m and all(len(f) == n for f in faces),
                   "the hull has the right faces")
            top = max(range(len(vs)), key=lambda i: vs[i].y + 1e-3 * vs[i].x + 1e-6 * vs[i].z)
            t1 = vs[top].norm()
            ax = t1.cross(V(0, 1, 0))
            R1 = (_AxisTurn(ax, math.atan2(ax.mag, t1.y)) if ax.mag > 1e-12
                  else _AxisTurn(V(1, 0, 0), 0 if t1.y > 0 else math.pi))
            vs = [R1.apply(1, v) for v in vs]
            nb = [i for i in range(len(vs)) if abs((vs[i] - vs[top]).mag - s) < 1e-9]
            _check(len(nb) == m, "m edges at the corner")
            w = vs[nb[0]] - vs[top]
            az = math.atan2(w.z, w.x)
            R2 = _AxisTurn(V(0, 1, 0), az - (PHC - math.pi))
            vs = [R2.apply(1, v - vs[top]) for v in vs]          # top vertex at 0
            top_faces = [f for f in faces if top in f]
            rest = [f for f in faces if top not in f]
            for pts in polys:
                _check(any(all(min((q - vs[i]).mag for i in f) < 1e-7 for q in pts)
                           for f in top_faces), "the closed fan is the solid's corner")
            fig.update(vs=vs, rest=rest, faces=faces, vol=vol1 * s ** 3)
        figs.append(fig)

    for row in (0, 1):
        for col in range(3):
            _check(ext[(col, row)][1] + 0.2 < ext[(col + 1, row)][0],
                   "neighbouring figures keep apart")
    _check(min(e[0] for e in ext.values()) > -5.3 and max(e[1] for e in ext.values()) < 5.3,
           "every figure inside the view")

    # ---- drawing: fans as triangles on vertices we move; solids as compounds
    def shade(col, k):
        return col * (1.0 if k % 2 == 0 else 0.8)

    for fig in figs:
        n, m, P = fig["n"], fig["m"], fig["P"]
        polys, nrms, _ = _fan(n, m, math.pi / 2, PHC, fig["s"])
        fig["vx"], fig["edges"] = [], []
        for k, (pts, nr) in enumerate(zip(polys, nrms)):
            vv = [vertex(pos=P + q, normal=nr, color=shade(fig["col"], k)) for q in pts]
            for j in range(1, n - 1):
                triangle(vs=[vv[0], vv[j], vv[j + 1]])
            fig["vx"].append(vv)
            fig["edges"].append(curve(pos=[P + q for q in pts + pts[:1]], radius=0.011,
                                      color=C_HL * 0.9))
        if fig["flat"]:
            continue
        # the gap wedge (moved by set_fan)
        KG = 24
        gv = [vertex(pos=P, color=C_E, opacity=0.55, normal=V(0, 1, 0))
              for _ in range(KG + 2)]
        fig["gap"] = gv
        fig["gap_tris"] = [triangle(vs=[gv[0], gv[i], gv[i + 1]]) for i in range(1, KG + 1)]
        vs = fig["vs"]
        g = sum(vs, V(0, 0, 0)) / len(vs)
        parts, vol, seen = [], 0.0, set()
        for k, f in enumerate(fig["faces"]):
            pts = [vs[i] for i in f]
            fc = sum(pts, V(0, 0, 0)) / n
            nr = (pts[1] - pts[0]).cross(pts[2] - pts[0]).norm()
            if nr.dot(fc - g) < 0:
                pts, nr = list(reversed(pts)), -nr
            for j in range(1, n - 1):
                vol += (pts[0] - g).dot((pts[j] - g).cross(pts[j + 1] - g)) / 6
            if f not in fig["rest"]:
                continue
            col = shade(fig["col"], k)
            for j in range(1, n - 1):
                parts.append(triangle(vs=[vertex(pos=P + pts[0], normal=nr, color=col),
                                          vertex(pos=P + pts[j], normal=nr, color=col),
                                          vertex(pos=P + pts[j + 1], normal=nr, color=col)]))
            for a_, b_ in zip(f, f[1:] + f[:1]):
                e = (min(a_, b_), max(a_, b_))
                if e in seen:
                    continue
                seen.add(e)
                parts.append(cylinder(pos=P + vs[a_], axis=vs[b_] - vs[a_], radius=0.011,
                                      color=C_HL * 0.9))
        _check(abs(vol - fig["vol"]) < 1e-9, "the solid's volume (its faces close up)")
        gp = sum([P + v for v in vs], V(0, 0, 0)) / len(vs)
        cp = compound(parts, origin=gp, visible=False)
        body = _Rigid()
        body.add(cp, gp - P)
        fig["body"], fig["cp"] = body, cp

    spin = _AxisTurn(V(0, 1, 0), 2 * math.pi)

    def set_fan(fig, u, s_spin=0.0):
        """Fold fig to u (0 flat, 1 closed), turned by s_spin of a full turn
        about the vertical through its corner."""
        th = math.pi / 2 + (fig["thf"] - math.pi / 2) * u
        polys, nrms, rays = _fan(fig["n"], fig["m"], th, PHC, fig["s"])
        P = fig["P"]
        # the gap: the part of the cone between the two free edges, round
        # the front; it vanishes as the corner closes
        gv = fig.get("gap")
        if gv is not None:
            m_ = fig["m"]
            beta = _fan_beta(fig["n"], th)
            start, span = PHC + m_ * beta / 2, 2 * math.pi - m_ * beta
            st, ct = math.sin(th), math.cos(th)
            for wi, wv in enumerate(gv[1:]):
                ph = start + span * wi / (len(gv) - 2)
                d = V(st * math.cos(ph), -ct, st * math.sin(ph))
                wv.pos = P + spin.apply(s_spin, d * (0.6 * fig["s"]))
            for tri in fig["gap_tris"]:
                tri.visible = span > 1e-3
        for vv, cv, pts, nr in zip(fig["vx"], fig["edges"], polys, nrms):
            nr = spin.apply(s_spin, nr)
            for vx_, q in zip(vv, pts):
                vx_.pos = P + spin.apply(s_spin, q)
                vx_.normal = nr
            for j, q in enumerate(pts + pts[:1]):
                cv.modify(j, pos=P + spin.apply(s_spin, q))

    # ---- labels: the angle sum at each corner
    labs = []
    for fig in figs:
        n, m = fig["n"], fig["m"]
        _check(360 % n == 0, "whole-degree angles")
        a_ = 180 - 360 // n
        txt = f"{m} × {a_}° = {m * a_}°"
        P = fig["P"]
        pos = P + (V(0, 0.55, -1.75) if P.y > 0 else V(0, -2.35, 1.0))
        labs.append(_lab(pos, txt, C_HL if not fig["flat"] else C_D))

    row1, row2, row3 = _rows(sc, 3)
    row1.text = _pre("the angle of a regular n-gon is 180° − 360°/n:   60°, 90°, 108°, 120°"
                     "   for n = 3, 4, 5, 6")
    row2.text = _pre("a corner needs m ≥ 3 faces and  m · angle < 360°   ⟺   "
                     "(m − 2)(n − 2) < 4   ⟺   1/m + 1/n > 1/2")
    folding = [f for f in figs if not f["flat"]]
    _ready(sc)
    while True:
        row3.text = _pre("less than 360°: the gap closes into a corner.     360°: the faces "
                         "lie flat.     more faces or more sides: over 360°, they overlap.")
        for fig in folding:
            set_fan(fig, 0)
        _wait(2.6)
        # ---- fold: every gap below 360° closes
        for t in _frames(3.0):
            u = _smooth(t)
            for fig in folding:
                set_fan(fig, u)
        _wait(0.8)
        # ---- each corner grows into its solid
        for fig in folding:
            fig["cp"].opacity = 0
            fig["cp"].visible = True
            fig["body"].pose(_IDENT, 0, fig["P"])
        for t in _frames(1.6):
            for fig in folding:
                fig["cp"].opacity = max(t, 0.02)
        row3.text = _pre("(n, m) = (3, 3), (3, 4), (3, 5), (4, 3), (5, 3):  four, eight and "
                         "twenty triangles, six squares, twelve pentagons")
        # ---- and turns once about the corner
        for t in _frames(8.0):
            u = _smooth(t)
            for fig in folding:
                set_fan(fig, 1, u)
                fig["body"].pose(spin, u, fig["P"])
        _wait(1.2)
        for t in _frames(1.0):
            for fig in folding:
                fig["cp"].opacity = max(1 - t, 0.02)
        for fig in folding:
            fig["cp"].visible = False
        for t in _frames(2.0):
            u = 1 - _smooth(t)
            for fig in folding:
                set_fan(fig, u)
        _wait(0.6)
