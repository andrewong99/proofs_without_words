# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d_solids.py — vpython scenes: cube packings and solids
(B3, B8, C5, I1, I2, I4, I8, I9).

Each scene_<ID>() is found by name by pww_3d.py / the app. Every packing
and dissection shown here is rebuilt from its rule and checked cube by
cube (or face by face) before anything is drawn: a construction that does
not close fails at start-up instead of animating a wrong picture.
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


def _bezier(a, c, b, s):
    """Quadratic Bézier from a to b with control point c."""
    return a * ((1 - s) ** 2) + c * (2 * s * (1 - s)) + b * (s * s)


# ---- integer rotations (cube packings) --------------------------------

def _rotations():
    """The 24 proper rotations of the cube as 3x3 integer matrices."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = [[0] * 3 for _ in range(3)]
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i][p] = s
            if _det(M) == 1:
                out.append(M)
    return out


def _det(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def _mul(M, c):
    return tuple(M[i][0] * c[0] + M[i][1] * c[1] + M[i][2] * c[2]
                 for i in range(3))


def _normal_form(cells):
    lo = [min(c[i] for c in cells) for i in range(3)]
    return frozenset((c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]) for c in cells), lo


def _pose_between(src, dst):
    """Proper rotation Q and translation t with Q·src + t = dst exactly,
    or None if dst is not a rotated copy of src."""
    dst_nf, dlo = _normal_form(dst)
    for Q in _rotations():
        nf, lo = _normal_form([_mul(Q, c) for c in src])
        if nf == dst_nf:
            return Q, tuple(dlo[i] - lo[i] for i in range(3))
    return None


# ---- continuous rotations ---------------------------------------------

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



def _cube_body(cells_by_layer, colours, unit=1.0, gap=0.1):
    """A _Rigid of unit cubes, one compound per layer (so the layers can be
    shown one at a time); cell (i, j, k) is the unit cube with its corner
    at (i, j, k). The body's reference point is the centroid of all its
    cube centres."""
    allc = [c for layer in cells_by_layer for c in layer]
    g = V(sum(c[0] for c in allc), sum(c[1] for c in allc),
          sum(c[2] for c in allc)) / len(allc) + V(0.5, 0.5, 0.5)
    body = _Rigid()
    s = unit - gap
    for layer, col in zip(cells_by_layer, colours):
        if not layer:
            continue
        gl = V(sum(c[0] for c in layer), sum(c[1] for c in layer),
               sum(c[2] for c in layer)) / len(layer) + V(0.5, 0.5, 0.5)
        cubes = [box(pos=(V(*c) + V(0.5, 0.5, 0.5) - gl) * unit,
                     size=V(s, s, s), color=col, visible=False)
                 for c in layer]
        cp = compound(cubes, origin=V(0, 0, 0))
        cp.visible = False
        body.add(cp, (gl - g) * unit)
    return body, g


def _wire_box(lo, hi, col=C_HL, radius=0.03):
    xs, ys, zs = (lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)
    segs = []
    for y in ys:
        for z in zs:
            segs.append(curve(pos=[V(xs[0], y, z), V(xs[1], y, z)],
                              radius=radius, color=col))
    for x in xs:
        for z in zs:
            segs.append(curve(pos=[V(x, ys[0], z), V(x, ys[1], z)],
                              radius=radius, color=col))
    for x in xs:
        for y in ys:
            segs.append(curve(pos=[V(x, y, zs[0]), V(x, y, zs[1])],
                              radius=radius, color=col))
    return segs


def _shade(col, k):
    """Alternate layer shading so the stacked layers stay readable."""
    return col * (1.0 if k % 2 == 0 else 0.68)


# ====================================================== cube packings

def _packing_show(sc, dims, targets, forms, order, cols, side, *,
                  col_x, depths, build_terms, line1_final, line3_text,
                  count_word, product_text, dim_labels, half=3):
    """Shared engine of B3 and B8: congruent piles of unit cubes stand in two
    columns, are built layer by layer, then fly one by one (a rigid turn,
    then straight down) into a box, which they fill exactly.

    dims          (W, H, D) of the box, in cubes (x, y = up, z)
    targets[p]    cell set of piece p in the box, cells (i, j, k)
    forms[p]      the layers (bottom first) of piece p as it stands in the
                  lineup; targets[p] must be a proper rotation of it
    order         assembly order (each piece is checked to drop in freely)
    side[p]       -1 / +1: which column piece p stands in
    """
    W, H, D = dims
    corner = V(-W / 2, -H / 2, -D / 2)              # box centred on the origin

    # ---- the packing, checked cube by cube
    box_cells = {(i, j, k) for i in range(W) for j in range(H) for k in range(D)}
    _check(sum(len(t) for t in targets) == len(box_cells),
           "the pieces hold exactly as many cubes as the box")
    _check(set().union(*targets) == box_cells, "no gaps: the box is filled")
    for a, b in itertools.combinations(targets, 2):
        _check(not (a & b), "no overlaps between pieces")
    sizes = {sum(len(L) for L in f) for f in forms}
    _check(len(sizes) == 1, "all pieces have the same number of cubes")
    occupied = set()
    for p in order:                    # each piece can be lowered into place
        for dy in range(1, 3 * H):
            moved = {(c[0], c[1] + dy, c[2]) for c in targets[p]}
            _check(not (moved & occupied), "piece drops in from above")
        occupied |= targets[p]

    ident = _Turn([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bodies, homes, slots, turns, hovers = [], [], [], [], []
    used = {-1: 0, 1: 0}
    for p in order:
        layers = forms[p]
        pose = _pose_between([c for L in layers for c in L], targets[p])
        _check(pose is not None, "each piece is a rigid copy of its lineup form")
        Q, t = pose
        body, g = _cube_body(layers, [_shade(cols[p], j) for j in range(len(layers))])
        # cube centres m map by m -> Q m + (t + h - Q h), h = (½, ½, ½)
        h = (0.5, 0.5, 0.5)
        Qh = _mul(Q, h)
        tc = V(t[0] + h[0] - Qh[0], t[1] + h[1] - Qh[1], t[2] + h[2] - Qh[2])
        slot = corner + V(*_mul(Q, (g.x, g.y, g.z))) + tc
        home = V(side[p] * col_x, -H / 2 + g.y, depths[used[side[p]]])
        used[side[p]] += 1
        body.pose(ident, 0, home)
        bodies.append(body)
        homes.append(home)
        slots.append(slot)
        turns.append(_Turn(Q))
        hovers.append(slot + V(0, H + 1.6, 0))

    _wire_box(corner, corner + V(W, H, D))
    for off, text in dim_labels:
        label(pos=corner + off, text=text, height=15, box=False, color=C_HL,
              opacity=0)

    sc.append_to_caption("\n")
    line1 = wtext(text="")
    sc.append_to_caption("\n")
    line2 = wtext(text="")
    sc.append_to_caption("\n")
    line3 = wtext(text="")

    def pre(txt):
        return f"<pre style='margin:0'>{txt}</pre>"

    _ready(sc)
    # ---- every copy is built layer by layer, bottom layer first
    shown = []
    for j, term in enumerate(build_terms):
        for b in bodies:
            b.parts[j][0].visible = True
        shown.append(term)
        line1.text = pre(f"one {count_word}:   " + " + ".join(reversed(shown)))
        _wait(0.6)
    line1.text = pre(f"one {count_word}:   {line1_final}")
    line3.text = pre(line3_text)
    _wait(1.2)
    size = sizes.pop()

    def fly_in(k):
        b, turn = bodies[k], turns[k]
        ctrl = (homes[k] + hovers[k]) / 2 + V(0, 3.5, 0)
        for t in _frames(2.0):                    # arc over, turning rigidly
            s = _smooth(t)
            b.pose(turn, s, _bezier(homes[k], ctrl, hovers[k], s))
        for t in _frames(0.8):                    # then straight down
            s = _smooth(t)
            b.pose(turn, 1, hovers[k] * (1 - s) + slots[k] * s)

    def fly_out(k, u):
        b, turn = bodies[k], turns[k]
        ctrl = (homes[k] + hovers[k]) / 2 + V(0, 3.5, 0)
        if u < 0.3:
            s = _smooth(u / 0.3)
            b.pose(turn, 1, slots[k] * (1 - s) + hovers[k] * s)
        else:
            s = _smooth((u - 0.3) / 0.7)
            b.pose(turn, 1 - s, _bezier(hovers[k], ctrl, homes[k], s))

    m = len(order)
    while True:
        for k in range(m):
            fly_in(k)
            line2.text = pre(f"in the box:    {k + 1} × {size}  =  "
                             f"{(k + 1) * size} cubes"
                             + ("" if k < m - 1 else f"  =  {product_text}"))
            _wait(1.4 if k == half - 1 else 0.35)
        _wait(5.0)
        # back to the lineup: last in, first out, staggered
        total, dur, lag = 2.0 + 0.5 * (m - 1), 2.0, 0.5
        done = set()
        for t in _frames(total):
            T = t * total
            for k in range(m):
                u = (T - lag * (m - 1 - k)) / dur
                if 0 < u < 1:
                    fly_out(k, u)
                elif u >= 1 and k not in done:
                    fly_out(k, 1.0)
                    done.add(k)
        line2.text = pre("")
        _wait(1.2)


def _to_box(cells, n):
    """Cells of a packing rule (X < n, Y, Z) -> scene cells (i, j, k):
    i = Z along x, j = n-1-X up, k = Y in depth."""
    return {(Z, n - 1 - X, Y) for X, Y, Z in cells}


# ============================================================== B3

def _b3_packing(n):
    """Six copies of the stepped pyramid in an n × (n+1) × (2n+1) box.

    In box coordinates (X < n, Y < n+1, Z < 2n+1) piece p is the pyramid
    {max(x,y) + z <= n-1}, turned by ROT[p] and shifted to OFF[p]. Pieces
    0-2 fill Z < n plus the staircase {Y <= X} of the layer Z = n; pieces
    3-5 fill the rest. Returned as cell sets."""
    ROT = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
           ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
           ((0, 1, 0), (0, 0, -1), (-1, 0, 0)),
           ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
           ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
           ((0, -1, 0), (-1, 0, 0), (0, 0, -1))]
    OFF = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, n), (0, 0, n + 1),
           (0, 1, n + 1)]
    base = [(x, y, z) for z in range(n) for x in range(n - z)
            for y in range(n - z)]
    out = []
    for M, off in zip(ROT, OFF):
        nf, _ = _normal_form([_mul(M, c) for c in base])
        out.append({(c[0] + off[0], c[1] + off[1], c[2] + off[2]) for c in nf})
    return out


def scene_B3():
    """1² + 2² + … + n² = n(n+1)(2n+1)/6: six stepped pyramids of unit
    cubes fill an n × (n+1) × (2n+1) box exactly."""
    n = 4
    sc = new_canvas("1² + 2² + … + n²  =  n(n+1)(2n+1) / 6",
                    "six copies of the stepped pyramid fill an "
                    "n × (n+1) × (2n+1) box  ·  n = 4",
                    rng=8.8, forward=V(0, -0.643, -0.766), centre=V(0, 0, 0))
    sc.fov = 0.5
    targets = [_to_box(p, n) for p in _b3_packing(n)]
    # the pyramid as it stands in the lineup: layer j is an (n-j)² square
    pyramid = [[(i, j, k) for i in range(n - j) for k in range(n - j)]
               for j in range(n)]
    _check(sum(len(L) for L in pyramid) == sum(k * k for k in range(1, n + 1)),
           "pyramid = 1² + … + n² cubes")
    W, H, D = 2 * n + 1, n, n + 1
    order = [1, 0, 2, 4, 5, 3]           # left half (pieces 0-2) first
    side = {0: -1, 1: -1, 2: -1, 3: 1, 4: 1, 5: 1}
    S = sum(k * k for k in range(1, n + 1))
    _packing_show(
        sc, (W, H, D), targets, [pyramid] * 6, order,
        [C_A, C_B, C_C, C_D, C_E, C_F], side,
        col_x=9.0, depths=(-6.0, 0.0, 6.0),
        build_terms=[f"{n - j}²" for j in range(n)],
        line1_final=(" + ".join(f"{k}²" for k in range(1, n + 1)) + "  =  "
                     + " + ".join(str(k * k) for k in range(1, n + 1))
                     + f"  =  {S}"),
        line3_text="six pyramids:  6 · (1² + 2² + … + n²)  =  n(n+1)(2n+1)",
        count_word="pyramid", product_text=f"{n} · {n + 1} · {2 * n + 1}",
        dim_labels=[(V(W / 2, -0.9, D + 0.2), "2n+1 = 9"),
                    (V(-0.9, H / 2, D), "n = 4"),
                    (V(W + 1.5, -0.3, D / 2), "n+1 = 5")])


# ============================================================== B8

def _b8_packing(n):
    """Six piles of triangular layers in an n × (n+1) × (n+2) box.

    The pile is {0 <= z <= y <= x <= n-1}: its layers z = 0, 1, … are the
    staircase triangles T_n, T_(n-1), …, T_1. It is chiral, and no packing
    of six rotated copies exists (exhaustive search, n = 2, 3, 4); this one
    uses three piles A, B, C and their point reflections through the
    centre of the box (mirror images). In box coordinates
    (X < n, Y < n+1, Z < n+2):
        A = {Z <= Y <= X},  B = {Y <= Z-1 <= X},  C = {Z <= X <= Y-1}."""
    A = {(x, y, z) for x in range(n) for y in range(n + 1)
         for z in range(n + 2) if z <= y <= x}
    B = {(x, y, z) for x in range(n) for y in range(n + 1)
         for z in range(n + 2) if y <= z - 1 <= x}
    C = {(x, y, z) for x in range(n) for y in range(n + 1)
         for z in range(n + 2) if z <= x <= y - 1}

    def inv(S):
        return {(n - 1 - x, n - y, n + 1 - z) for x, y, z in S}
    return [A, B, C, inv(B), inv(C), inv(A)]


def scene_B8():
    """T₁ + T₂ + … + Tₙ = n(n+1)(n+2)/6: six piles of triangular layers
    (three piles and their three mirror images) fill an n × (n+1) × (n+2)
    box exactly."""
    n = 4
    sc = new_canvas("T₁ + T₂ + … + Tₙ  =  n(n+1)(n+2) / 6",
                    "three piles of triangular layers and their three mirror "
                    "images fill an n × (n+1) × (n+2) box  ·  n = 4",
                    rng=8.0, forward=V(0, -0.643, -0.766), centre=V(0, 0, 0))
    sc.fov = 0.5
    targets = [_to_box(p, n) for p in _b8_packing(n)]
    # the pile as it stands in the lineup: layer j is the staircase T_(n-j)
    pile = [[(i, j, k) for i in range(n) for k in range(n) if j <= k <= i]
            for j in range(n)]
    mirror = [[(n - 1 - i, j, k) for i, j, k in L] for L in pile]
    T = [k * (k + 1) // 2 for k in range(n + 1)]
    _check([len(L) for L in pile] == [T[n - j] for j in range(n)],
           "layer j of the pile is the triangle T_(n-j)")
    flat = [c for L in pile for c in L]
    forms, side = [], {}
    for p, tgt in enumerate(targets):
        if _pose_between(flat, tgt) is not None:
            forms.append(pile)
            side[p] = -1                  # left column: the pile itself
        else:
            forms.append(mirror)
            side[p] = 1                   # right column: its mirror image
    _check(sorted(side.values()) == [-1, -1, -1, 1, 1, 1],
           "three piles and three mirror images")
    W, H, D = n + 2, n, n + 1
    total = sum(T[1:])
    sub = "₀₁₂₃₄₅₆₇₈₉"
    _packing_show(
        sc, (W, H, D), targets, forms, [0, 1, 2, 3, 4, 5],
        [C_A, C_B, C_C, C_D, C_E, C_F], side,
        col_x=7.4, depths=(-5.5, 0.0, 5.5),
        build_terms=[f"T{sub[n - j]}" for j in range(n)],
        line1_final=(" + ".join(f"T{sub[k]}" for k in range(1, n + 1)) + "  =  "
                     + " + ".join(str(T[k]) for k in range(1, n + 1))
                     + f"  =  {total}"),
        line3_text="six piles:     6 · (T₁ + T₂ + … + Tₙ)  =  n(n+1)(n+2)",
        count_word="pile", product_text=f"{n} · {n + 1} · {n + 2}",
        dim_labels=[(V(W / 2, -0.9, D + 0.2), "n+2 = 6"),
                    (V(-0.9, H / 2, D), "n = 4"),
                    (V(W + 1.5, -0.3, D / 2), "n+1 = 5")])


# ============================================================== C5

def scene_C5():
    """a³ − b³ = (a − b)(a² + ab + b²): take the b-cube out of a corner of
    the a-cube; the rest is three slabs of thickness a − b with faces a²,
    ab and b²; turned flat they make one slab (a − b) × (a² + ab + b²)."""
    sc = new_canvas("a³ − b³  =  (a − b)(a² + ab + b²)",
                    "take the b-cube out of the corner; the rest is three "
                    "slabs of thickness a − b",
                    rng=3.8, forward=V(-0.35, -0.42, -0.84), centre=V(0.4, 0.5, 0))
    sc.fov = 0.5
    a, b = 5, 3                         # the readout uses these numbers
    k = 0.6                             # world units per unit
    A, B, T = a * k, b * k, (a - b) * k
    o = V(-A / 2, -A / 2, -A / 2)       # the a-cube is centred on the origin

    def piece(lo, hi, col):
        lo, hi = V(*lo) * k + o, V(*hi) * k + o
        bx = box(pos=(lo + hi) / 2, size=hi - lo - V(0.03, 0.03, 0.03),
                 color=col, opacity=1.0)
        return bx, (lo + hi) / 2, hi - lo

    # the dissection of the a-cube (cube coordinates 0..a)
    cube_b, cb_home, _ = piece((a - b, a - b, a - b), (a, a, a), C_D)
    s1, s1_home, s1_dim = piece((0, 0, 0), (a, a - b, a), C_A)          # a·(a−b)·a
    s2, s2_home, s2_dim = piece((0, a - b, 0), (a - b, a, a), C_B)      # (a−b)·b·a
    s3, s3_home, s3_dim = piece((a - b, a - b, 0), (a, a, a - b), C_C)  # b·b·(a−b)

    # exact bookkeeping of the dissection
    vols = [round(d.x * d.y * d.z / k ** 3, 9) for d in (s1_dim, s2_dim, s3_dim)]
    _check(vols == [a * a * (a - b), a * b * (a - b), b * b * (a - b)],
           "the slabs are a²(a−b), ab(a−b), b²(a−b)")
    _check(sum(vols) + b ** 3 == a ** 3, "slabs + b-cube = a-cube")
    # the four pieces tile the a-cube: check on a fine grid of cell centres
    N = 2 * a
    def inside(p, lo, hi):
        return all(lo[i] < p[i] < hi[i] for i in range(3))
    boxes = [((a - b,) * 3, (a,) * 3), ((0, 0, 0), (a, a - b, a)),
             ((0, a - b, 0), (a - b, a, a)), ((a - b, a - b, 0), (a, a, a - b))]
    for i in range(N):
        for j in range(N):
            for m in range(N):
                p = ((i + 0.5) * a / N, (j + 0.5) * a / N, (m + 0.5) * a / N)
                _check(sum(inside(p, lo, hi) for lo, hi in boxes) == 1,
                       "the four pieces tile the a-cube")

    # final row on the floor, all with the thickness a−b vertical, flush:
    # a × a, then b × a (sharing the edge a), then b × b (sharing the edge b)
    floor = -A / 2
    x0, zf = -(A + 2 * B) / 2, A / 2      # left end of the row, front face
    s1_end = V(x0 + A / 2, floor + T / 2, zf - A / 2)
    s2_end = V(x0 + A + B / 2, floor + T / 2, zf - A / 2)
    s3_end = V(x0 + A + 1.5 * B, floor + T / 2, zf - B / 2)
    ident = _Turn([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    turn2 = _Turn([[0, -1, 0], [1, 0, 0], [0, 0, 1]])   # x → y: thickness up
    turn3 = _Turn([[1, 0, 0], [0, 0, 1], [0, -1, 0]])   # z → y: thickness up
    # after the turn the dimensions are (b, a−b, a) and (b, a−b, b)
    _check(abs(turn2.apply(1, V(s2_dim.x, 0, 0)).y - T) < 1e-9, "S2 turned flat")
    _check(abs(turn3.apply(1, V(0, 0, s3_dim.z)).y - T) < 1e-9, "S3 turned flat")

    bodies = []
    for obj in (s1, s2, s3):
        r = _Rigid()
        r.add(obj, V(0, 0, 0))
        bodies.append(r)
    # where each slab waits after the cube is opened up
    s1_out = s1_home + V(0, -0.15, 0.25)
    s2_out = s2_home + V(-0.45, 0.25, 0)
    s3_out = s3_home + V(0.25, 0.25, 0.55)

    cube_wire = _wire_box(o, o + V(A, A, A), col=V(0.5, 0.5, 0.42), radius=0.012)
    lab_a = label(pos=o + V(A / 2, -0.32, A), text="a", height=16, box=False,
                  color=C_HL, opacity=0)
    lab_b = label(pos=o + V(A - B / 2, A + 0.3, A), text="b", height=16,
                  box=False, color=C_HL, opacity=0)
    end_labels = [
        label(pos=s1_end + V(0, T / 2 + 0.35, 0), text="a²", height=17,
              box=False, color=C_HL, opacity=0),
        label(pos=s2_end + V(0, T / 2 + 0.35, 0), text="ab", height=17,
              box=False, color=C_HL, opacity=0),
        label(pos=s3_end + V(0, T / 2 + 0.35, 0), text="b²", height=17,
              box=False, color=C_HL, opacity=0),
        label(pos=V(x0 - 0.45, floor + T / 2, zf), text="a − b", height=16,
              box=False, color=C_HL, opacity=0),
    ]

    sc.append_to_caption("\n")
    line1 = wtext(text="")
    sc.append_to_caption("\n")
    line2 = wtext(text="")

    def pre(t):
        return f"<pre style='margin:0'>{t}</pre>"

    lift = cb_home + V(0, A / 2 + 0.5, 0)
    aside = lift + V(2.7, 0, -0.6)
    lab_b3 = label(pos=aside + V(0, B / 2 + 0.3, 0), text="b³", height=16,
                   box=False, color=C_D, opacity=0, visible=False)

    def show(flag, objs):
        for ob in objs:
            ob.visible = flag

    _ready(sc)
    while True:
        # ---- the whole cube
        show(True, [lab_a, lab_b])
        show(False, end_labels)
        cube_b.opacity = 1.0
        for bd, home in zip(bodies, (s1_home, s2_home, s3_home)):
            bd.pose(ident, 0, home)
        cube_b.pos = cb_home
        line1.text = pre(f"a³  =  {a}³  =  {a ** 3}")
        line2.text = pre(" ")
        _wait(2.0)
        # ---- take the b-cube out
        show(False, [lab_b])
        for t in _frames(1.4):
            cube_b.pos = cb_home * (1 - _smooth(t)) + lift * _smooth(t)
        for t in _frames(1.0):
            cube_b.pos = lift * (1 - _smooth(t)) + aside * _smooth(t)
        line1.text = pre(f"a³ − b³  =  {a ** 3} − {b ** 3}  =  {a ** 3 - b ** 3}")
        for t in _frames(0.8):
            cube_b.opacity = 1 - 0.9 * t
        show(False, [lab_a])
        show(True, [lab_b3])
        # ---- the three slabs come apart ...
        for t in _frames(1.0):
            s = _smooth(t)
            for bd, h, out in zip(bodies, (s1_home, s2_home, s3_home),
                                  (s1_out, s2_out, s3_out)):
                bd.pose(ident, 0, h * (1 - s) + out * s)
        # ---- ... and turn flat, one after another, into one slab
        plan = [(bodies[0], ident, s1_out, s1_end),
                (bodies[1], turn2, s2_out, s2_end),
                (bodies[2], turn3, s3_out, s3_end)]
        show(False, cube_wire)
        for bd, turn, start, end in plan:
            ctrl = (start + end) / 2 + V(0, 1.0, 0.4)
            for t in _frames(1.3):
                s = _smooth(t)
                bd.pose(turn, s, _bezier(start, ctrl, end, s))
        show(True, end_labels)
        line2.text = pre(
            f"(a−b)·a² + (a−b)·ab + (a−b)·b²  =  (a−b)(a² + ab + b²)  =  "
            f"{a - b} · ({a * a} + {a * b} + {b * b})  =  "
            f"{(a - b) * (a * a + a * b + b * b)}")
        _wait(5.5)
        # ---- and back
        show(False, end_labels)
        for bd, turn, start, end in reversed(plan):
            ctrl = (start + end) / 2 + V(0, 1.0, 0.4)
            for t in _frames(0.9):
                s = _smooth(t)
                bd.pose(turn, 1 - s, _bezier(end, ctrl, start, s))
        show(True, cube_wire)
        for t in _frames(0.8):
            s = _smooth(t)
            for bd, h, out in zip(bodies, (s1_home, s2_home, s3_home),
                                  (s1_out, s2_out, s3_out)):
                bd.pose(ident, 0, out * (1 - s) + h * s)
        show(False, [lab_b3])
        for t in _frames(0.6):
            cube_b.opacity = 0.1 + 0.9 * t
        for t in _frames(1.2):
            s = _smooth(t)
            cube_b.pos = _bezier(aside, lift, cb_home, s)
        _wait(0.6)


# ---- polyhedral pieces ----------------------------------------------

def _css(col):
    return f"rgb({int(255 * col.x)},{int(255 * col.y)},{int(255 * col.z)})"


def _mesh(faces, col, opacity=1.0):
    """A closed polyhedron from planar faces (lists of points, any order of
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
            tris.append(triangle(vs=[vertex(pos=p, color=col, normal=nrm,
                                            opacity=opacity),
                                     vertex(pos=q, color=col, normal=nrm,
                                            opacity=opacity),
                                     vertex(pos=r_, color=col, normal=nrm,
                                            opacity=opacity)]))
    cp = compound(tris, origin=g)       # built in place: pos is g
    return cp, g, vol


def _turn_y(p, quarter):
    """Rotate a point about the vertical axis by quarter·90°."""
    x, y, z = p.x, p.y, p.z
    for _ in range(quarter % 4):
        x, z = z, -x
    return V(x, y, z)


# ============================================================== I1

def scene_I1():
    """Frustum of a square pyramid: a central block, four side wedges and
    four corner pyramids (the dissection behind the rule of the 方亭).
    V = b²h + b(a−b)h + ⅓(a−b)²h = h(a² + ab + b²)/3."""
    sc = new_canvas("frustum:  V = h(a² + ab + b²) / 3",
                    "one block, four wedges, four corner pyramids  ·  "
                    "a = 6, b = 2, h = 3",
                    rng=3.4, forward=V(-0.38, -0.55, -0.74), centre=V(-0.3, -0.9, 0))
    sc.fov = 0.5
    a, b, h = 6, 2, 3
    k = 0.6                                     # world units per unit
    A, B, Hh = a * k, b * k, h * k
    y0 = -Hh / 2
    c = (A - B) / 2

    # ---- the nine pieces, +x / (+x,+z) versions turned about the axis
    def P(x, y, z):
        return V(x, y0 + y, z)

    block = [[P(-B / 2, 0, -B / 2), P(B / 2, 0, -B / 2), P(B / 2, 0, B / 2), P(-B / 2, 0, B / 2)],
             [P(-B / 2, Hh, -B / 2), P(B / 2, Hh, -B / 2), P(B / 2, Hh, B / 2), P(-B / 2, Hh, B / 2)]]
    block += [[block[0][i], block[0][(i + 1) % 4], block[1][(i + 1) % 4], block[1][i]]
              for i in range(4)]
    w0 = [P(B / 2, 0, -B / 2), P(A / 2, 0, -B / 2), P(B / 2, Hh, -B / 2)]
    w1 = [P(B / 2, 0, B / 2), P(A / 2, 0, B / 2), P(B / 2, Hh, B / 2)]
    wedge = [w0, w1, [w0[0], w0[1], w1[1], w1[0]], [w0[0], w0[2], w1[2], w1[0]],
             [w0[1], w0[2], w1[2], w1[1]]]
    base = [P(B / 2, 0, B / 2), P(A / 2, 0, B / 2), P(A / 2, 0, A / 2), P(B / 2, 0, A / 2)]
    apex = P(B / 2, Hh, B / 2)
    corner = [base] + [[base[i], base[(i + 1) % 4], apex] for i in range(4)]

    COL_BLOCK, COL_WEDGE, COL_CORNER = C_A, C_B, C_C
    blk, g_blk, v_blk = _mesh(block, COL_BLOCK)
    wedges, corners = [], []
    for q in range(4):
        wedges.append(_mesh([[_turn_y(p, q) for p in f] for f in wedge],
                            COL_WEDGE * (1.0 if q % 2 == 0 else 0.84)))
        corners.append(_mesh([[_turn_y(p, q) for p in f] for f in corner],
                             COL_CORNER * (1.0 if q % 2 == 0 else 0.84)))

    # ---- exact checks: the volumes of the meshes and of the formula
    unit3 = k ** 3
    _check(abs(v_blk / unit3 - b * b * h) < 1e-9, "block = b²h")
    for _, _, v in wedges:
        _check(abs(v / unit3 - 0.5 * b * (a - b) / 2 * h) < 1e-9,
               "wedge = ½·b·(a−b)/2·h")
    for _, _, v in corners:
        _check(abs(v / unit3 - ((a - b) / 2) ** 2 * h / 3) < 1e-9,
               "corner pyramid = ⅓((a−b)/2)²h")
    total = (v_blk + sum(v for _, _, v in wedges) + sum(v for _, _, v in corners)) / unit3
    _check(abs(total - h * (a * a + a * b + b * b) / 3) < 1e-9,
           "the nine pieces add up to h(a² + ab + b²)/3")
    # the slanted faces lie in the frustum's side planes x (or z) = A/2 - c·y/H
    for f in wedge[4:] + corner[2:3]:
        for p in f:
            _check(abs(p.x - (A / 2 - c * (p.y - y0) / Hh)) < 1e-9,
                   "slanted faces lie on the frustum's faces")
    for p in corner[3]:
        _check(abs(p.z - (A / 2 - c * (p.y - y0) / Hh)) < 1e-9,
               "slanted faces lie on the frustum's faces")

    # ---- outline of the frustum
    bot = [P(sx * A / 2, 0, sz * A / 2) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    top = [P(sx * B / 2, Hh, sz * B / 2) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    outline = [curve(pos=bot + [bot[0]], radius=0.014, color=C_HL),
               curve(pos=top + [top[0]], radius=0.014, color=C_HL)]
    outline += [curve(pos=[bot[i], top[i]], radius=0.014, color=C_HL) for i in range(4)]

    # explode directions: wedges straight out, corners out diagonally
    d = 1.1
    moves = [(blk, g_blk, V(0, 0, 0))]
    for q in range(4):
        moves.append((wedges[q][0], wedges[q][1], _turn_y(V(d, 0, 0), q)))
        moves.append((corners[q][0], corners[q][1], _turn_y(V(d, 0, d), q)))

    sc.append_to_caption("\n")
    rows = [wtext(text="")]

    def sw(col):
        return f"<span style='color:{_css(col)}'>■</span>"

    e = (a - b) // 2
    v_w = 4 * b * e * h / 2
    v_c = 4 * e * e * h / 3
    _check(v_w == int(v_w) and v_c == int(v_c), "integer readouts")
    rows_txt = [
        (COL_BLOCK, "block", "b²·h", f"{b}²·{h}", b * b * h),
        (COL_WEDGE, "4 wedges", "4 · ½·b·((a−b)/2)·h", f"4 · ½·{b}·{e}·{h}", int(v_w)),
        (COL_CORNER, "4 pyramids", "4 · ⅓·((a−b)/2)²·h", f"4 · ⅓·{e}²·{h}", int(v_c)),
        (None, "frustum", "h(a² + ab + b²)/3", f"{h}·({a * a}+{a * b}+{b * b})/3",
         round(total)),
    ]
    _check(sum(r[4] for r in rows_txt[:3]) == rows_txt[3][4], "the parts add up")
    lines = [(sw(col) if col else " ") + f" {name:<12s} {sym:<21s}=  {num:<16s}=  {val}"
             for col, name, sym, num, val in rows_txt]
    lines[3] += "  =  " + " + ".join(str(r[4]) for r in rows_txt[:3])

    def show_rows(m):
        txt = "\n".join(lines[:m]) if m else " "
        rows[0].text = f"<pre style='margin:0'>{txt}</pre>"

    groups = [[blk], [w[0] for w in wedges], [cc[0] for cc in corners]]

    def dim_all_but(gi):
        for i, grp in enumerate(groups):
            for ob in grp:      # a compound's colour tints its parts
                ob.color = V(1, 1, 1) if (gi is None or i == gi) else V(0.3, 0.3, 0.3)

    _ready(sc)
    while True:
        dim_all_but(None)
        for ob, g, _ in moves:
            ob.pos = g
        for o_ in outline:
            o_.visible = True
        show_rows(0)
        _wait(2.2)
        for o_ in outline:
            o_.visible = False
        for t in _frames(1.6):                      # explode
            s = _smooth(t)
            for ob, g, mv in moves:
                ob.pos = g + mv * s
        for gi in range(3):                         # one kind of piece at a time
            dim_all_but(gi)
            show_rows(gi + 1)
            _wait(1.7)
        dim_all_but(None)
        show_rows(4)
        _wait(2.0)
        for t in _frames(1.6):                      # and back together
            s = _smooth(t)
            for ob, g, mv in moves:
                ob.pos = g + mv * (1 - s)
        for o_ in outline:
            o_.visible = True
        _wait(3.0)


# ============================================================== I2

def scene_I2():
    """Cone = ⅓ cylinder, by Cavalieri against a square pyramid with the
    same base area (s² = πr²) and the same height h (= s, so its prism is
    a cube). At every height both sections shrink by (1 − y/h)², so they
    stay equal; the cube splits into three such pyramids (I7)."""
    sc = new_canvas("V(cone) = ⅓ πr²h   (Cavalieri)",
                    "same base area  πr² = s²,  same height h:  equal sections "
                    "at every height  ·  the cube is three such pyramids",
                    rng=2.75, forward=V(-0.12, -0.42, -0.90), centre=V(0, 0.0, 0))
    sc.fov = 0.4
    s_u = 3.0                                  # side of the square base (units)
    r_u = s_u / math.sqrt(math.pi)             # πr² = s²
    h_u = s_u                                  # height = s: the prism is a cube
    k = 0.8
    s, r, h = s_u * k, r_u * k, h_u * k
    y0 = -h / 2
    L = V(-2.3, 0, 0)                          # cone / cylinder axis (x, z)
    x0, z0 = 2.3 - s / 2, -s / 2               # back-left corner of the cube

    # ---- left: cylinder and cone
    cylinder(pos=L + V(0, y0, 0), axis=V(0, h, 0), radius=r, color=C_B,
             opacity=0.10)
    for yy in (y0, y0 + h):
        ring(pos=L + V(0, yy, 0), axis=V(0, 1, 0), radius=r, thickness=0.012,
             color=C_B)
    cone(pos=L + V(0, y0, 0), axis=V(0, h, 0), radius=r, color=C_A, opacity=0.38)

    # ---- right: the cube, its three pyramids (I7), ours on the bottom face
    apex = V(x0, y0 + h, z0)                  # top, back-left corner

    def Cn(i, j, m):
        return V(x0 + i * s, y0 + j * h, z0 + m * s)
    _wire_box(Cn(0, 0, 0), Cn(1, 1, 1), col=C_B, radius=0.012)
    box(pos=(Cn(0, 0, 0) + Cn(1, 1, 1)) / 2, size=V(s, h, s), color=C_B,
        opacity=0.07)
    for q in (Cn(1, 1, 1), Cn(1, 0, 0), Cn(0, 0, 1), Cn(1, 0, 1)):
        curve(pos=[apex, q], radius=0.008, color=V(0.55, 0.55, 0.5))
    base = [Cn(0, 0, 0), Cn(1, 0, 0), Cn(1, 0, 1), Cn(0, 0, 1)]
    _, _, v_pyr = _mesh([base] + [[base[i], base[(i + 1) % 4], apex]
                                    for i in range(4)], C_C, opacity=0.42)
    _check(abs(v_pyr - s * s * h / 3) < 1e-9, "pyramid = ⅓ · s² · h")
    # the other two pyramids of the cube (I7): bases on the right and front
    others = [[Cn(1, 0, 0), Cn(1, 1, 0), Cn(1, 1, 1), Cn(1, 0, 1)],
              [Cn(0, 0, 1), Cn(1, 0, 1), Cn(1, 1, 1), Cn(0, 1, 1)]]
    vols = [v_pyr]
    for bse in others:
        pts = bse + [apex]
        g = sum(pts, V(0, 0, 0)) / 5
        vol = 0.0
        for f in [bse] + [[bse[i], bse[(i + 1) % 4], apex] for i in range(4)]:
            fc = sum(f, V(0, 0, 0)) / len(f)
            n_ = (f[1] - f[0]).cross(f[2] - f[0])
            if n_.dot(fc - g) < 0:
                f = list(reversed(f))
            for i in range(1, len(f) - 1):
                vol += (f[0] - g).dot((f[i] - g).cross(f[i + 1] - g)) / 6
        vols.append(vol)
    _check(all(abs(v - s * s * h / 3) < 1e-9 for v in vols) and
           abs(sum(vols) - s * s * h) < 1e-9, "three equal pyramids fill the cube")

    # ---- the sections, at every height y: (1 − y/h)² · base, on both sides
    for frac in [i / 50 for i in range(51)]:
        rho, sig = r_u * (1 - frac), s_u * (1 - frac)
        _check(abs(math.pi * rho * rho - sig * sig) < 1e-9,
               "equal sections at every height")
    thick = 0.022
    disc = cylinder(pos=L + V(0, y0, 0), axis=V(0, thick, 0), radius=r, color=C_A)
    sq = box(pos=Cn(0.5, 0, 0.5), size=V(s, thick, s), color=C_C)
    cyl_ring = ring(pos=L + V(0, y0, 0), axis=V(0, 1, 0), radius=r,
                    thickness=0.016, color=C_HL)
    cube_sq = curve(pos=[Cn(0, 0, 0), Cn(1, 0, 0), Cn(1, 0, 1), Cn(0, 0, 1),
                         Cn(0, 0, 0)], radius=0.012, color=C_HL)
    # symbols only: equal bases, equal heights
    label(pos=L + V(0, y0 - 0.32, r + 0.25), text="πr²", height=16, box=False,
          color=C_A, opacity=0)
    label(pos=Cn(0.5, 0, 1) + V(0, -0.32, 0.25), text="s²", height=16, box=False,
          color=C_C, opacity=0)
    label(pos=L + V(-r - 0.3, y0 + h / 2, 0), text="h", height=16, box=False,
          color=C_HL, opacity=0)
    label(pos=Cn(1, 0.5, 1) + V(0.3, 0, 0), text="h", height=16, box=False,
          color=C_HL, opacity=0)

    sc.append_to_caption("\n")
    line1 = wtext(text="")
    sc.append_to_caption("\n")
    line2 = wtext(text="")

    def pre(t):
        return f"<pre style='margin:0'>{t}</pre>"

    line2.text = pre(
        f"s = {s_u:g},  r = s/√π = {r_u:.4f},  h = s:      "
        f"V(pyramid) = ⅓ V(cube) = ⅓ s²h   ⟹   V(cone) = ⅓ πr²h = ⅓ V(cylinder)")
    _ready(sc)
    i = 0
    while True:
        rate(FPS)
        i += 1
        frac = 0.5 - 0.5 * math.cos(i * math.pi / 160)      # 0 → 1 → 0
        frac = min(frac, 0.985)
        y = y0 + frac * h
        rho, sig = r * (1 - frac), s * (1 - frac)
        disc.pos = L + V(0, y - thick / 2, 0)
        disc.radius = max(rho, 1e-3)
        sq.pos = V(x0 + sig / 2, y, z0 + sig / 2)
        sq.size = V(max(sig, 1e-3), thick, max(sig, 1e-3))
        cyl_ring.pos = L + V(0, y, 0)
        cube_sq.origin = V(0, frac * h, 0)
        if i % 3 == 0:
            yu = frac * h_u
            line1.text = pre(
                f"height y = {yu:5.3f}     cone:  π·ρ² = π·({r_u * (1 - frac):.4f})² = "
                f"{math.pi * (r_u * (1 - frac)) ** 2:7.4f}      pyramid:  σ² = "
                f"({s_u * (1 - frac):.4f})² = {(s_u * (1 - frac)) ** 2:7.4f}")


# ============================================================== I4

class _Strip:
    """A band of a surface of revolution about the y axis, through several
    rims, drawn as quads that share their vertices (so it is cheap to build
    and to move). rims: list of (y, radius, (n_radial, n_up)); phi runs
    from phi0 to phi1 in n steps."""

    def __init__(self, n, col, rims, phi0=0.0, phi1=2 * math.pi):
        self.phis = [phi0 + (phi1 - phi0) * i / n for i in range(n + 1)]
        self.rows = []
        for y, rad, nr in rims:
            self.rows.append([vertex(pos=self._p(y, rad, ph),
                                     normal=self._n(nr, ph), color=col)
                              for ph in self.phis])
        self.quads = [quad(vs=[lo[i], lo[i + 1], hi[i + 1], hi[i]])
                      for lo, hi in zip(self.rows, self.rows[1:])
                      for i in range(n)]

    @staticmethod
    def _p(y, rad, ph):
        return V(rad * math.cos(ph), y, rad * math.sin(ph))

    @staticmethod
    def _n(nr, ph):
        return V(nr[0] * math.cos(ph), nr[1], nr[0] * math.sin(ph))

    def set(self, rims, normals=True):
        for row, (y, rad, nr) in zip(self.rows, rims):
            for vx, ph in zip(row, self.phis):
                vx.pos = self._p(y, rad, ph)
                if normals:
                    vx.normal = self._n(nr, ph)

    def show(self, flag):
        for q in self.quads:
            q.visible = flag


def _shell_of_revolution(profile, col, out=1.006, inn=0.994, np_=72):
    """A thin shell about the y axis on a sphere centred at the origin:
    profile = meridian points (radius, y); each point is pushed out and in
    along its own normal (the ray from the centre)."""
    Rp = 1.0                      # radius of the sweeping path (any > 0)
    outer = [(rr * out - Rp, y * out) for rr, y in profile]
    inner = [(rr * inn - Rp, y * inn) for rr, y in reversed(profile)]
    shp = [list(p) for p in outer + inner]
    shp.append(shp[0])
    return extrusion(path=paths.circle(pos=V(0, 0, 0), radius=Rp,
                                       up=V(0, 1, 0), np=np_),
                     shape=shp, color=col)


def _zone_area(r, y1, y2, n=400):
    """Area of the sphere between heights y1 < y2, measured on the surface
    itself (sum of thin conical bands), independent of any formula."""
    tot = 0.0
    for i in range(n):
        ya, yb = y1 + (y2 - y1) * i / n, y1 + (y2 - y1) * (i + 1) / n
        ra, rb = math.sqrt(max(r * r - ya * ya, 0)), math.sqrt(max(r * r - yb * yb, 0))
        tot += math.pi * (ra + rb) * math.hypot(rb - ra, yb - ya)
    return tot


def scene_I4():
    """Archimedes' hat-box: horizontal projection from the axis carries the
    sphere onto its cylinder without changing area. Around the axis a
    patch is stretched by r/ρ, along the slope it is shortened by ρ/r."""
    sc = new_canvas("S(sphere) = 4πr²   (Archimedes' hat-box)",
                    "project the sphere straight out onto its cylinder: "
                    "×r/ρ around the axis, ×ρ/r along the slope",
                    rng=2.2, forward=V(-0.22, -0.36, -0.91), centre=V(0, -0.15, 0))
    sc.fov = 0.4
    R = 1.5                                   # world radius; readouts use r = 1
    sphere(pos=V(0, 0, 0), radius=R, color=C_A, opacity=0.16)
    cylinder(pos=V(0, -R, 0), axis=V(0, 2 * R, 0), radius=R, color=C_B,
             opacity=0.08)
    for yy in (-R, R):
        ring(pos=V(0, yy, 0), axis=V(0, 1, 0), radius=R, thickness=0.01,
             color=C_B)

    dh = 0.2                                  # band height, in units of r
    N = 48
    e1, e2 = 1.004, 1.012                     # drawn just outside the surfaces

    def rims_sphere(y1, y2, e):
        r1, r2 = math.sqrt(max(1 - y1 * y1, 0)), math.sqrt(max(1 - y2 * y2, 0))
        return [(y1 * R, r1 * R * e, (r1, y1)), (y2 * R, r2 * R * e, (r2, y2))]

    def rims_cyl(y1, y2, e):
        return [(y1 * R, R * e, (1, 0)), (y2 * R, R * e, (1, 0))]

    y_start = -(1 - dh / 2 - 0.06)
    # the cylinder's bands are drawn as a cutaway (the far side and the right
    # side, up to the patch), so the sphere's zones stay in view in front
    cut0, cut1 = math.radians(150), math.radians(378)
    zone = _Strip(N, C_C, rims_sphere(y_start - dh / 2, y_start + dh / 2, e1))
    band = _Strip(N, C_B * 1.3, rims_cyl(y_start - dh / 2, y_start + dh / 2, e1),
                  cut0, cut1)
    # a patch and its image, turned to the side so neither hides the other
    ph0, ph1 = math.radians(18), math.radians(42)
    patch_s = _Strip(4, V(1, 1, 1), rims_sphere(y_start - dh / 2, y_start + dh / 2, e2),
                     ph0, ph1)
    patch_c = _Strip(4, V(1, 1, 1), rims_cyl(y_start - dh / 2, y_start + dh / 2, e2),
                     ph0, ph1)
    ray_phis = [ph0 + (ph1 - ph0) * j / 5 for j in range(6)]
    rays = [curve(pos=[V(0, 0, 0), V(0, 0, 0)], radius=0.008, color=V(1, 1, 1))
            for _ in ray_phis]

    def place(yc):
        """Band of height dh centred at yc (units of r)."""
        y1, y2 = yc - dh / 2, yc + dh / 2
        zone.set(rims_sphere(y1, y2, e1))
        band.set(rims_cyl(y1, y2, e1), normals=False)
        patch_s.set(rims_sphere(y1, y2, e2))
        patch_c.set(rims_cyl(y1, y2, e2), normals=False)
        rho = math.sqrt(1 - yc * yc)
        for cv, ph in zip(rays, ray_phis):
            c, s_ = math.cos(ph), math.sin(ph)
            cv.modify(0, pos=V(rho * R * c, yc * R, rho * R * s_))
            cv.modify(1, pos=V(R * 1.02 * c, yc * R, R * 1.02 * s_))
        return y1, y2, rho

    # the painted zones of phase 2: equal heights on both surfaces, each a
    # thin shell of revolution (an extrusion of its meridian profile),
    # made when first needed so the scene starts at once
    Z = 8
    painted = []
    for zi in range(Z):
        ya, yb = -1 + 2 * zi / Z, -1 + 2 * (zi + 1) / Z
        # Archimedes: equal heights, equal areas (measured on the surface)
        _check(abs(_zone_area(1.0, ya, yb) - 2 * math.pi * (yb - ya)) < 1e-4,
               "each zone of the sphere has the area of its cylinder band")

    def paint(zi):
        if zi < len(painted):
            for ob in painted[zi]:
                ob.visible = True
            return
        ya, yb = -1 + 2 * zi / Z, -1 + 2 * (zi + 1) / Z
        col = C_B if zi % 2 == 0 else C_C
        cyl = _Strip(32, col, rims_cyl(ya, yb, e1), cut0, cut1 - math.radians(16))
        painted.append([
            _shell_of_revolution([(math.sqrt(max(1 - y * y, 0)) * R, y * R)
                                  for y in [ya + (yb - ya) * j / 10
                                            for j in range(11)]], col)]
            + cyl.quads)

    sc.append_to_caption("\n")
    line1 = wtext(text="")
    sc.append_to_caption("\n")
    line2 = wtext(text="")
    sc.append_to_caption("\n")
    line3 = wtext(text="")

    def pre(t):
        return f"<pre style='margin:0'>{t}</pre>"

    moving = [zone, band, patch_s, patch_c]
    _ready(sc)
    while True:
        # ---- phase 1: one band, its projection and a patch, all heights
        for m in moving:
            m.show(True)
        for cv in rays:
            cv.visible = True
        line3.text = pre(f"r = 1,  Δh = {dh:g}")
        steps = 420
        for i in range(steps + 1):
            rate(FPS)
            yc = -(1 - dh / 2 - 0.06) * math.cos(2 * math.pi * i / steps)
            y1, y2, rho = place(yc)
            if i % 3 == 0:
                line1.text = pre(
                    f"height y = {yc:+.3f}    ρ = √(1 − y²) = {rho:.3f}      "
                    f"around:  2πρ → 2πr   ×{1 / rho:.3f}      "
                    f"along the slope:  ×{rho:.3f}      product = 1")
                line2.text = pre(
                    f"zone on the sphere (measured) = {_zone_area(1.0, y1, y2):.4f}"
                    f"      band on the cylinder  2πr·Δh = {2 * math.pi * dh:.4f}")
        # ---- phase 2: equal heights, equal areas, all the way up
        for m in moving:
            m.show(False)
        for cv in rays:
            cv.visible = False
        line1.text = pre(f"{Z} zones of equal height 2r/{Z}:  each = 2πr · 2r/{Z} "
                         f"= {2 * math.pi * 2 / Z:.4f}  on the sphere and on the cylinder")
        line2.text = pre(" ")
        line3.text = pre(" ")
        for zi in range(Z):
            paint(zi)
            _wait(0.55)
        line2.text = pre("sphere  =  4πr²  =  2πr · 2r  =  cylinder (without its lids)"
                         f"  =  {4 * math.pi:.4f}")
        _wait(4.0)
        for pair in painted:
            for ob in pair:
                ob.visible = False


# ============================================================== I8

def _grid_quads(fn, nu, nv, col, opacity=1.0, visible=True):
    """Quads of a parametrised surface; fn(i/nu, j/nv) -> (point, normal)."""
    rows = []
    for i in range(nu + 1):
        row = []
        for j in range(nv + 1):
            p, n_ = fn(i / nu, j / nv)
            row.append(vertex(pos=p, normal=n_, color=col, opacity=opacity))
        rows.append(row)
    return [quad(vs=[rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], rows[i][j + 1]],
                 visible=visible)
            for i in range(nu) for j in range(nv)]


def scene_I8():
    """Steinmetz bicylinder (牟合方盖): every horizontal section of the
    common part of two perpendicular cylinders is a square of side
    2√(r² − y²) around the inscribed sphere's circle of radius √(r² − y²).
    Square : circle = 4 : π at every height, so V = (4/π)·(4/3)πr³."""
    sc = new_canvas("Steinmetz bicylinder:  V = 16r³ / 3",
                    "every level: a square around the sphere's circle, "
                    "area ratio 4 : π",
                    rng=2.9, forward=V(-0.5, -0.45, -0.74), centre=V(0, -0.2, 0))
    sc.fov = 0.45
    R = 1.5
    Lc = 2.2                               # half-length of the drawn cylinders
    cyl_x = cylinder(pos=V(-Lc, 0, 0), axis=V(2 * Lc, 0, 0), radius=R,
                     color=C_B, opacity=0.13)
    cyl_z = cylinder(pos=V(0, 0, -Lc), axis=V(0, 0, 2 * Lc), radius=R,
                     color=C_F, opacity=0.13)

    # ---- the common part, face by face: x = ±w(y) and z = ±w(y)
    def w_of(y):
        return math.sqrt(max(R * R - y * y, 0.0))

    def face(sx, axis_x):
        def fn(u, v):
            y = R * math.sin(math.pi * (u - 0.5))        # denser near the poles
            w = w_of(y)
            t = (2 * v - 1) * w
            if axis_x:     # on the cylinder x² + y² = R²
                return V(sx * w, y, t), V(sx * w, y, 0) / R
            return V(t, y, sx * w), V(0, y, sx * w) / R  # on y² + z² = R²
        return fn
    quads = []
    for sx in (1, -1):
        for ax in (True, False):
            quads += _grid_quads(face(sx, ax), 36, 10, C_A, opacity=0.32,
                                 visible=False)
    bicyl = compound(quads, visible=False)       # shown after the cylinders
    ball = sphere(pos=V(0, 0, 0), radius=R, color=C_C, opacity=0.22, visible=False)

    # ---- the claims, checked: the common part of the two cylinders is
    # {|x| <= w(y), |z| <= w(y)} (its level y is a square of half-side w),
    # and the sphere's level y is the circle of radius w inscribed in it
    grid = [R * (2 * i + 1 - 24) / 24 for i in range(24)]
    for x in grid:
        for y in grid:
            for z in grid:
                in_both = x * x + y * y <= R * R and y * y + z * z <= R * R
                in_square = abs(x) <= w_of(y) and abs(z) <= w_of(y)
                _check(in_both == in_square, "every level is a square")
                if x * x + y * y + z * z <= R * R:
                    _check(x * x + z * z <= w_of(y) ** 2 + 1e-12,
                           "the sphere's level is the inscribed circle")
    V_sphere = 4 / 3 * math.pi * R ** 3
    _check(abs(4 / math.pi * V_sphere - 16 * R ** 3 / 3) < 1e-9,
           "(4/π)·(4/3)πr³ = 16r³/3")

    # ---- the moving level
    th = 0.025
    strip_x = box(pos=V(0, 0, 0), size=V(2 * Lc, th * 0.6, 2 * R), color=C_B,
                  opacity=0.35, visible=False)
    strip_z = box(pos=V(0, 0, 0), size=V(2 * R, th * 0.6, 2 * Lc), color=C_F,
                  opacity=0.35, visible=False)
    square = box(pos=V(0, 0, 0), size=V(2 * R, th, 2 * R), color=C_D, visible=False)
    disc = cylinder(pos=V(0, 0, 0), axis=V(0, th * 1.6, 0), radius=R, color=C_E,
                    visible=False)

    sc.append_to_caption("\n")
    line1 = wtext(text="")
    sc.append_to_caption("\n")
    line2 = wtext(text="")
    sc.append_to_caption("\n")
    line3 = wtext(text="")

    def pre(t):
        return f"<pre style='margin:0'>{t}</pre>"

    level = [strip_x, strip_z, square, disc]
    _ready(sc)
    # ---- the two cylinders, then their common part, then the sphere in it
    line3.text = pre("two cylinders of radius r, axes at right angles")
    _wait(2.0)
    bicyl.visible = True
    line3.text = pre("their common part, and the sphere of radius r inside it")
    for t in _frames(1.2):
        cyl_x.opacity = cyl_z.opacity = 0.13 - 0.08 * t
    _wait(0.6)
    ball.visible = True
    _wait(1.4)
    for ob in level:
        ob.visible = True
    line3.text = pre("V = (4/π) · V(sphere) = (4/π) · (4/3)πr³ = 16r³/3"
                     f"      (r = 1:  {16 / 3:.4f})")
    i = 0
    while True:
        rate(FPS)
        i += 1
        y = -0.97 * R * math.cos(i * math.pi / 200)
        w = w_of(y)
        strip_x.pos = V(0, y, 0)
        strip_x.size = V(2 * Lc, th * 0.6, 2 * w)
        strip_z.pos = V(0, y, 0)
        strip_z.size = V(2 * w, th * 0.6, 2 * Lc)
        square.pos = V(0, y, 0)
        square.size = V(2 * w, th, 2 * w)
        disc.pos = V(0, y - th * 0.3, 0)
        disc.radius = max(w, 1e-3)
        if i % 3 == 0:
            yu, wu = y / R, w / R
            line1.text = pre(f"height y = {yu:+.3f}      half-side w = √(1 − y²) = "
                             f"{wu:.4f}          (r = 1)")
            sq_a, ci_a = 4 * wu * wu, math.pi * wu * wu
            line2.text = pre(f"square (2w)² = {sq_a:.4f}      circle πw² = "
                             f"{ci_a:.4f}      ratio = {sq_a / ci_a:.4f} = 4/π")


# ============================================================== I9

def _solid_of_revolution(profile, col, centre, opacity=1.0, np_=128):
    """The solid swept by a closed meridian polygon (radius, y) about the
    vertical axis through `centre` (one extrusion)."""
    Rp = 1.0
    shp = [[rr - Rp, y] for rr, y in profile]
    if shp[0] != shp[-1]:
        shp.append(shp[0])
    return extrusion(path=paths.circle(pos=centre, radius=Rp, up=V(0, 1, 0), np=np_),
                     shape=shp, color=col, opacity=opacity)


def scene_I9():
    """The napkin ring: drill a sphere through its centre so that a band
    of height h is left. At height y the band's section is the annulus
    π(R² − y²) − π(R² − (h/2)²) = π((h/2)² − y²), whatever R is: the disc
    of a ball of diameter h. So every ring of height h has volume πh³/6."""
    sc = new_canvas("napkin ring:  V = πh³ / 6",
                    "two spheres drilled to the same band height h, and a ball "
                    "of diameter h: equal sections at every level",
                    rng=2.5, forward=V(0, -0.5, -0.87), centre=V(0, 0.1, 0))
    sc.fov = 0.4
    k = 0.7                                    # world units per unit
    h = 2.0
    Rs = [1.25, 2.2]                           # the two spheres (units)
    cx = [-3.0, 0.0, 3.0]                      # small ring, big ring, ball
    cols = [C_B, C_C, C_A]

    rings = []
    for i, Rr in enumerate(Rs):
        a = math.sqrt(Rr * Rr - (h / 2) ** 2)  # radius of the drilled hole
        c = V(cx[i], 0, 0)
        # meridian section: the sphere's arc from y = -h/2 to h/2, back down
        # the wall of the hole (they meet exactly at y = ±h/2)
        arc = [(math.sqrt(Rr * Rr - y * y) * k, y * k)
               for y in [-h / 2 + h * j / 40 for j in range(41)]]
        _check(abs(arc[0][0] - a * k) < 1e-12 and abs(arc[-1][0] - a * k) < 1e-12,
               "the hole's wall meets the sphere at y = ±h/2")
        _solid_of_revolution(arc + [(a * k, -h / 2 * k)], cols[i], c, opacity=0.4)
        # the sphere it was cut from, as two great circles
        for ax in (V(1, 0, 1), V(1, 0, -1)):
            ring(pos=c, axis=ax, radius=Rr * k, thickness=0.006, color=cols[i] * 0.8)
        rings.append((Rr, a, c))
    ball_c = V(cx[2], 0, 0)
    sphere(pos=ball_c, radius=h / 2 * k, color=cols[2], opacity=0.4)

    # ---- the claim, checked at many levels: annulus = disc
    for j in range(-20, 21):
        y = (h / 2) * j / 20
        disc_area = math.pi * ((h / 2) ** 2 - y * y)
        for Rr, a, _ in rings:
            ann = math.pi * (Rr * Rr - y * y) - math.pi * a * a
            _check(abs(ann - disc_area) < 1e-9, "annulus = disc at every level")
    _check(abs(4 / 3 * math.pi * (h / 2) ** 3 - math.pi * h ** 3 / 6) < 1e-12,
           "(4/3)π(h/2)³ = πh³/6")

    label(pos=V(cx[0], h / 2 * k + 0.75, 0), text=f"R = {Rs[0]:g}", height=14,
          box=False, color=cols[0], opacity=0)
    label(pos=V(cx[1], h / 2 * k + 1.25, 0), text=f"R = {Rs[1]:g}", height=14,
          box=False, color=cols[1], opacity=0)
    label(pos=V(cx[2], h / 2 * k + 0.75, 0), text="r = h/2", height=14,
          box=False, color=cols[2], opacity=0)

    # ---- the moving level: two annuli and a disc
    N = 48
    annuli = []
    for Rr, a, c in rings:
        st = _Strip(N, C_D, [(0, a * k, (0, 1)), (0, Rr * k, (0, 1))])
        annuli.append(st)
    disc = cylinder(pos=ball_c, axis=V(0, 0.02, 0), radius=h / 2 * k, color=C_D)

    def set_level(y):
        for st, (Rr, a, c) in zip(annuli, rings):
            rho = math.sqrt(max(Rr * Rr - y * y, a * a))
            for row, rad in zip(st.rows, (a * k, rho * k)):
                for vx, ph in zip(row, st.phis):
                    vx.pos = c + V(rad * math.cos(ph), y * k, rad * math.sin(ph))
        disc.pos = ball_c + V(0, y * k - 0.01, 0)
        disc.radius = max(math.sqrt(max((h / 2) ** 2 - y * y, 0)) * k, 1e-3)

    sc.append_to_caption("\n")
    line1 = wtext(text="")
    sc.append_to_caption("\n")
    line2 = wtext(text="")
    sc.append_to_caption("\n")
    line3 = wtext(text="")

    def pre(t):
        return f"<pre style='margin:0'>{t}</pre>"

    line3.text = pre(f"h = {h:g}:   every ring of height h has the volume of the ball "
                     f"of diameter h:  (4/3)π(h/2)³ = πh³/6 = {math.pi * h ** 3 / 6:.4f}")
    set_level(-(h / 2) * 0.985)
    _ready(sc)
    i = 0
    while True:
        rate(FPS)
        i += 1
        y = -(h / 2) * 0.985 * math.cos(i * math.pi / 180)
        set_level(y)
        if i % 3 == 0:
            parts = []
            for Rr, a, _ in rings:
                parts.append(f"π(R² − y²) − πa² = {math.pi * (Rr * Rr - y * y - a * a):.4f}")
            line1.text = pre(f"height y = {y:+.3f}")
            line2.text = pre(f"R = {Rs[0]:g}:  {parts[0]}      R = {Rs[1]:g}:  {parts[1]}"
                             f"      ball:  π((h/2)² − y²) = "
                             f"{math.pi * ((h / 2) ** 2 - y * y):.4f}")
