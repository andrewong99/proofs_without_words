# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_kit.py — shared building blocks for the 2D scene modules.

Every scene module (pww_scenes_*.py) does `from pww_kit import *`.
(The original pww_scenes.py keeps its own copies of the first helpers, so
its cached videos stay valid.)

Hard rules, because breaking them breaks rendering on a machine without TeX:
    never MathTex, Tex, Brace, BraceLabel, DecimalNumber, Integer, Variable,
    Matrix, or NumberLine / Axes with numbers or labels.
Use Text (via `tag` / `caption`) with Unicode: ² ³ ⁿ ₁ ₂ √ π θ φ ≥ ≤ ≠ · × ½ ⅓
¼ ∑ ∫ ∞ → ⟹. Keep scene text Latin/Greek/math only (no CJK glyphs): the
Chinese names live in the GUI, which renders them reliably.

A scene is any class named <ID>_<Name> (e.g. B3_SumOfSquares) in a file
named pww_scenes*.py. The app finds it by that name — no registry edit.
"""

from manim import *          # noqa: F401,F403  (re-exported to scene modules)
import numpy as np

# ---------------------------------------------------------------- style

FILL = 0.78                  # default fill opacity
EDGE = dict(stroke_color=WHITE, stroke_width=2)

# a categorical palette that reads on black; use in order for distinct parts
PALETTE = [BLUE_D, TEAL_D, ORANGE, YELLOW_E, GREEN_D, RED_D, PURPLE_B,
           MAROON_B, GOLD_D, GREY_B]

# frame: 14.22 x 8 units. Keep content inside x in [-6.6, 6.6] and
# y in [-3.0, 3.75]; the band below y = -3.0 belongs to the caption.
SAFE_X, SAFE_TOP, SAFE_BOTTOM = 6.6, 3.75, -3.0


def caption(s, size=30, color=YELLOW_B):
    """The closing statement along the bottom edge."""
    t = Text(s, font_size=size, color=color)
    t.to_edge(DOWN, buff=0.3)
    if t.width > 13.4:
        t.scale_to_fit_width(13.4)
    return t


def tag(s, size=26, color=WHITE):
    """A short label (a², √(ab), 90° …)."""
    return Text(s, font_size=size, color=color)


def to3(p):
    """(x, y) or (x, y, z) -> np.array([x, y, 0])."""
    p = np.asarray(p, dtype=float)
    return np.array([p[0], p[1], 0.0]) if p.shape[0] == 2 else p.copy()


def mk(pts, color, op=FILL, **kw):
    """Filled polygon from 2D/3D points (already in screen coordinates)."""
    o = dict(EDGE)
    o.update(kw)
    return Polygon(*[to3(p) for p in pts], fill_color=color,
                   fill_opacity=op, **o)


class Frame:
    """Map a mathematical bounding box onto the safe screen area.

        F = Frame(xmin, xmax, ymin, ymax)        # math coordinates
        F.P((x, y))  -> screen point (np.array, z = 0)
        F.k          -> screen units per math unit

    Uniform scale (no distortion), centred, fitted to the safe area.
    """

    def __init__(self, xmin, xmax, ymin, ymax, max_w=2 * SAFE_X,
                 max_h=SAFE_TOP - SAFE_BOTTOM, centre=None):
        w, h = xmax - xmin, ymax - ymin
        self.k = min(max_w / w, max_h / h)
        self.cx, self.cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.centre = (np.array([0.0, (SAFE_TOP + SAFE_BOTTOM) / 2])
                       if centre is None else np.asarray(centre, float))

    def P(self, p):
        return np.array([(p[0] - self.cx) * self.k + self.centre[0],
                         (p[1] - self.cy) * self.k + self.centre[1], 0.0])

    def poly(self, pts, color, op=FILL, **kw):
        return mk([self.P(p) for p in pts], color, op, **kw)


def guide(p, q, color=GREY_B, extend=0.0, width=2):
    """Dashed line through screen points p, q, extended by a fraction."""
    p, q = to3(p), to3(q)
    d = q - p
    return DashedLine(p - extend * d, q + extend * d, color=color,
                      stroke_width=width, dash_length=0.09)


def angle_arc(vertex, p, q, radius=0.5, color=YELLOW_B, width=5):
    """Arc for the NON-REFLEX angle p-vertex-q, whichever order you pass.

    manim's Angle measures counter-clockwise from the first line to the
    second and will happily draw the reflex angle; this never does.
    """
    v, p, q = to3(vertex), to3(p), to3(q)
    a1 = float(np.arctan2(*(p - v)[1::-1]))
    a2 = float(np.arctan2(*(q - v)[1::-1]))
    span = (a2 - a1) % TAU
    if span > PI:
        a1, span = a2, TAU - span
    return Arc(radius=radius, start_angle=a1, angle=span, arc_center=v,
               color=color, stroke_width=width)


def angle_mid_dir(vertex, p, q):
    """Unit vector along the bisector of the non-reflex angle p-vertex-q."""
    v = to3(vertex)
    u1 = to3(p) - v
    u2 = to3(q) - v
    b = u1 / np.linalg.norm(u1) + u2 / np.linalg.norm(u2)
    n = np.linalg.norm(b)
    if n < 1e-9:                     # straight angle: perpendicular to it
        b = np.array([-u1[1], u1[0], 0.0])
        n = np.linalg.norm(b)
    return b / n


def cell(i, j, u, origin, color, op=FILL, stroke=1.5):
    """Unit square (i, j) of a grid with side u whose (0,0) corner is origin."""
    sq = Square(side_length=u, fill_color=color, fill_opacity=op,
                stroke_color=WHITE, stroke_width=stroke)
    sq.move_to(to3(origin) + np.array([(i + 0.5) * u, (j + 0.5) * u, 0.0]))
    return sq


# ------------------------------------------------- exact-geometry checks

def area(pts):
    """Signed shoelace area of a polygon given as (x, y) points."""
    pts = [np.asarray(p, float)[:2] for p in pts]
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def check(cond, what):
    """Fail the render loudly if a geometric claim is false.

    Put the proof's invariants in the scene itself — equal areas, a point
    landing exactly on a line — so a wrong construction cannot render
    quietly into a wrong picture.
    """
    if not cond:
        raise AssertionError(f"geometry check failed: {what}")


def close(a, b, tol=1e-9):
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                            atol=tol))


class Board(Scene):
    """Base scene. End every construct() with self.hold(2.2) or longer:
    the app parks on the final frame, so the closing hold must show the
    finished figure and its formula for at least 1.2 s."""

    def hold(self, t=1.4):
        self.wait(t)
