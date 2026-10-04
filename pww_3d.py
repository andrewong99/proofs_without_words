# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d.py — the vpython (3D) proofs without words.

Run standalone (any 3D id, including those in plugin modules pww_3d_*.py):
    python pww_3d.py C4              # embed-friendly: prints PWW_URL <url>
    python pww_3d.py C4 --external   # just open the default browser

vpython serves its canvas from a small local HTTP server and normally opens a
browser tab itself. When the GUI launches this script it wants the URL
instead, so `webbrowser.open` is captured before vpython is imported and the
address is printed on stdout as `PWW_URL <url>`. The GUI loads that in an
embedded web view; with --external the ordinary behaviour is restored.

One scene per process, by design: vpython's canvas and its server cannot be
torn down and rebuilt cleanly inside a live interpreter.

Mouse: right-drag (or Ctrl+drag) to rotate, Shift+drag to pan, scroll to zoom.
"""

from pww_3d_kit import *          # noqa: F401,F403  — must come first:
                                  # it captures webbrowser before vpython


# ------------------------------------------------------------------ C4

def scene_C4():
    """(a+b)^3 — eight blocks assemble a cube of edge a+b."""
    new_canvas("(a+b)³ = a³ + 3a²b + 3ab² + b³",
               "right-drag to rotate · the eight blocks separate and reassemble",
               rng=5.4)
    a, b = 2.0, 1.2
    s = a + b
    o = -s / 2.0        # so the cube is centred on the origin

    # (origin corner, size, colour) for each of the eight blocks
    blocks = [
        ((0, 0, 0), (a, a, a), C_A),        # a^3
        ((a, 0, 0), (b, a, a), C_B),        # a^2 b
        ((0, a, 0), (a, b, a), C_B),
        ((0, 0, a), (a, a, b), C_B),
        ((a, a, 0), (b, b, a), C_C),        # a b^2
        ((a, 0, a), (b, a, b), C_C),
        ((0, a, a), (a, b, b), C_C),
        ((a, a, a), (b, b, b), C_D),        # b^3
    ]
    objs, homes, aways = [], [], []
    for (px, py, pz), (w, h, d), col in blocks:
        centre = V(o + px + w / 2, o + py + h / 2, o + pz + d / 2)
        bx = box(pos=centre, size=V(w, h, d), color=col, opacity=0.94)
        objs.append(bx)
        homes.append(centre)
        n = centre.norm() if centre.mag > 1e-9 else V(1, 0, 0)
        aways.append(centre + n * 1.9)

    label(pos=V(0, s / 2 + 1.2, 0), text="1·a³   3·a²b   3·ab²   1·b³",
          height=17, box=False, color=C_HL, opacity=0)

    steps = 55
    while True:
        for direction in (1, -1):
            for i in range(steps):
                rate(45)
                f = i / (steps - 1.0)
                t = f if direction == 1 else 1 - f
                t = t * t * (3 - 2 * t)               # smoothstep
                for ob, h, aw in zip(objs, homes, aways):
                    ob.pos = h * (1 - t) + aw * t
            for _ in range(50):
                rate(45)


# ------------------------------------------------------------------ I7

def scene_I7():
    """A cube dissects into three congruent oblique square pyramids."""
    new_canvas("pyramid = ⅓ · prism",
               "one cube, three congruent pyramids · apex at a shared corner",
               rng=5.6)
    a = 2.4
    o = -a / 2.0
    apex = V(o, o, o)

    def corner(i, j, k):
        return V(o + i * a, o + j * a, o + k * a)

    # the three faces of the cube that do not contain the apex
    faces = [
        ([corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(1, 0, 1)], C_A),
        ([corner(0, 1, 0), corner(0, 1, 1), corner(1, 1, 1), corner(1, 1, 0)], C_B),
        ([corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)], C_C),
    ]

    # Each pyramid is built from its own vertex objects so the three can be
    # pulled apart independently; a vpython triangle/quad is moved by moving
    # the vertices it was built from.
    pyramids = []
    for base, col in faces:
        verts = []

        def vx(p, shade=1.0):
            v = vertex(pos=p, color=col * shade, opacity=0.95)
            verts.append((v, V(p.x, p.y, p.z)))
            return v

        # the base is drawn brighter than the four slanted sides, so an
        # exploded pyramid still reads as a solid rather than a flat patch
        vs = [vx(p, 1.0) for p in base]
        quad(v0=vs[0], v1=vs[1], v2=vs[2], v3=vs[3])
        for m in range(4):
            shade = 0.52 + 0.14 * m
            triangle(v0=vx(apex, shade), v1=vx(base[m], shade),
                     v2=vx(base[(m + 1) % 4], shade))
        centroid = (base[0] + base[1] + base[2] + base[3] + apex) / 5.0
        pyramids.append((verts, centroid.norm()))

    # wireframe of the whole cube
    for i in (0, 1):
        for j in (0, 1):
            curve(pos=[corner(i, j, 0), corner(i, j, 1)], radius=0.012, color=C_HL)
            curve(pos=[corner(i, 0, j), corner(i, 1, j)], radius=0.012, color=C_HL)
            curve(pos=[corner(0, i, j), corner(1, i, j)], radius=0.012, color=C_HL)

    label(pos=V(0, a / 2 + 3.0, 0), text="3 × (⅓ a³)  =  a³",
          height=17, box=False, color=C_HL, opacity=0)

    steps = 60
    while True:
        for direction in (1, -1):
            for i in range(steps):
                rate(45)
                f = i / (steps - 1.0)
                t = f if direction == 1 else 1 - f
                t = t * t * (3 - 2 * t)
                for verts, direction_v in pyramids:
                    shift = direction_v * (2.6 * t)
                    for v, home in verts:
                        v.pos = home + shift
            for _ in range(55):
                rate(45)


# ------------------------------------------------------------------ I3

def scene_I3():
    """Cavalieri / 祖暅: sphere = cylinder − double cone, slice by slice."""
    sc = new_canvas("V(sphere) = ⁴⁄₃πr³   (Cavalieri · 祖暅原理)",
                    "at every height the disc and the annulus have equal area",
                    rng=3.3, forward=V(-0.30, -0.20, -0.93))
    r = 1.6
    left, right = V(-2.3, 0, 0), V(2.3, 0, 0)

    sphere(pos=left, radius=r, color=C_A, opacity=0.14)
    cylinder(pos=right + V(0, -r, 0), axis=V(0, 2 * r, 0), radius=r,
             color=C_B, opacity=0.11)
    # the double cone removed from the cylinder: apexes meet at the centre
    cone(pos=right + V(0, -r, 0), axis=V(0, r, 0), radius=r,
         color=C_C, opacity=0.15)
    cone(pos=right + V(0, r, 0), axis=V(0, -r, 0), radius=r,
         color=C_C, opacity=0.15)

    thick = 0.045
    disc = cylinder(pos=left, axis=V(0, thick, 0), radius=r, color=C_A)
    ann_out = cylinder(pos=right, axis=V(0, thick, 0), radius=r, color=C_B)
    ann_in = cylinder(pos=right + V(0, -0.005, 0), axis=V(0, thick + 0.01, 0),
                      radius=0.001, color=V(0.06, 0.06, 0.08))

    label(pos=left + V(0, r + 0.55, 0), text="sphere", height=14,
          box=False, color=C_A, opacity=0)
    label(pos=right + V(0, r + 0.55, 0), text="cylinder − double cone",
          height=14, box=False, color=C_B, opacity=0)
    # the running readout lives under the canvas, where it cannot drift
    # out of view when the user rotates the scene
    sc.append_to_caption("\n")
    readout = wtext(text="")

    while True:
        for i in range(240):
            rate(40)
            y = r * math.sin(i * math.pi / 120.0)
            rho = math.sqrt(max(r * r - y * y, 1e-6))
            disc.pos = left + V(0, y, 0)
            disc.radius = rho
            ann_out.pos = right + V(0, y, 0)
            ann_in.pos = right + V(0, y - 0.005, 0)
            ann_in.radius = max(abs(y), 1e-3)
            if i % 3 == 0:
                readout.text = (
                    f"<pre>height y = {y:+.3f}        "
                    f"disc:    π·ρ² = {math.pi*rho*rho:7.4f}"
                    f"        annulus: π(r²−y²) = "
                    f"{math.pi*(r*r - y*y):7.4f}</pre>")


# ------------------------------------------------------------------ I5

def scene_I5():
    """d² = a² + b² + c²: Pythagoras twice, in perpendicular planes."""
    sc = new_canvas("d² = a² + b² + c²",
                    "the face diagonal, then the space diagonal", rng=2.6)
    a, b, c = 3.0, 2.0, 1.6
    o = V(-a / 2, -b / 2, -c / 2)

    box(pos=V(0, 0, 0), size=V(a, b, c), color=C_A, opacity=0.16)
    P000 = o
    P100 = o + V(a, 0, 0)
    P110 = o + V(a, b, 0)
    P111 = o + V(a, b, c)

    curve(pos=[P000, P100], radius=0.035, color=C_A)      # a
    curve(pos=[P100, P110], radius=0.035, color=C_B)      # b
    curve(pos=[P110, P111], radius=0.035, color=C_C)      # c
    curve(pos=[P000, P110], radius=0.030, color=C_D)      # face diagonal
    curve(pos=[P000, P111], radius=0.045, color=C_E)      # space diagonal

    label(pos=(P000 + P100) / 2 + V(0, -0.35, 0), text="a", height=16,
          box=False, color=C_A, opacity=0)
    label(pos=(P100 + P110) / 2 + V(0.35, 0, 0), text="b", height=16,
          box=False, color=C_B, opacity=0)
    label(pos=(P110 + P111) / 2 + V(0.35, 0, 0), text="c", height=16,
          box=False, color=C_C, opacity=0)
    label(pos=(P000 + P110) / 2 + V(0, -0.4, 0),
          text="√(a²+b²)", height=15, box=False, color=C_D, opacity=0)
    label(pos=(P000 + P111) / 2 + V(0, 0.45, 0),
          text="d = √(a²+b²+c²)", height=16, box=False, color=C_E, opacity=0)
    sc.append_to_caption(
        f"\n<pre>a = {a}   b = {b}   c = {c}\n"
        f"face diagonal² = a² + b²         = {a*a + b*b:.4f}\n"
        f"space diagonal² = (a²+b²) + c²   = {a*a + b*b + c*c:.4f}\n"
        f"d = {math.sqrt(a*a + b*b + c*c):.6f}</pre>")
    spin([])


# ------------------------------------------------------------------ I6

def scene_I6():
    """de Gua: the three-dimensional Pythagoras on a corner tetrahedron."""
    # look from the corner side, so the three right-angled faces are visible
    sc = new_canvas("de Gua's theorem:  A₁² + A₂² + A₃² = A₄²",
                    "three mutually perpendicular faces, and the slanted one",
                    rng=2.1, forward=V(0.58, 0.50, 0.64))
    a, b, c = 2.6, 2.0, 1.6
    O = V(-0.8, -0.9, -0.7)
    A = O + V(a, 0, 0)
    B = O + V(0, b, 0)
    C = O + V(0, 0, c)

    def face(p, q, r_, col, op=0.7):
        return triangle(v0=vertex(pos=p, color=col, opacity=op),
                        v1=vertex(pos=q, color=col, opacity=op),
                        v2=vertex(pos=r_, color=col, opacity=op))

    face(O, A, B, C_A)          # A3 = ab/2   (the xy face)
    face(O, B, C, C_B)          # A1 = bc/2
    face(O, C, A, C_C)          # A2 = ca/2
    face(A, B, C, C_E, 0.75)    # A4, the slanted face

    for p, q in [(O, A), (O, B), (O, C), (A, B), (B, C), (C, A)]:
        curve(pos=[p, q], radius=0.018, color=C_HL)

    A1, A2, A3 = b * c / 2, c * a / 2, a * b / 2
    # Heron on the slanted face
    ab, bc, ca = (A - B).mag, (B - C).mag, (C - A).mag
    s = (ab + bc + ca) / 2
    A4 = math.sqrt(s * (s - ab) * (s - bc) * (s - ca))

    sc.append_to_caption(
        f"\n<pre>A₁ = bc/2 = {A1:.4f}    A₂ = ca/2 = {A2:.4f}    "
        f"A₃ = ab/2 = {A3:.4f}    A₄ = {A4:.6f}\n"
        f"A₁² + A₂² + A₃² = {A1**2:.4f} + {A2**2:.4f} + {A3**2:.4f} "
        f"= {A1**2 + A2**2 + A3**2:.6f}\n"
        f"A₄²             = {A4**2:.6f}</pre>")
    spin([])


def main():
    """Built-in scenes are the scene_<ID> functions above; any other id
    is looked up in the plugin modules pww_3d_*.py."""
    return run_cli(globals())


if __name__ == "__main__":
    sys.exit(main())
