# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_3d_kit.py — shared machinery for the vpython (3D) scenes.

Import it FIRST, before anything that imports vpython:

    from pww_3d_kit import *

because importing this module is what captures `webbrowser.open`, so that
vpython's attempt to open a browser tab is turned into a URL the GUI can
embed (printed as `PWW_URL <url>`). With --external on the command line the
ordinary browser behaviour is kept.

A 3D scene is any top-level function named scene_<ID> in a file named
pww_3d*.py (e.g. def scene_I2() in pww_3d_solids.py). Everything is launched
through pww_3d.py:

    python pww_3d.py I2              # embed-friendly
    python pww_3d.py I2 --external   # plain browser tab

One scene per process, by design: vpython's canvas and its server cannot be
torn down and rebuilt cleanly inside a live interpreter.
"""

import sys
import time

EXTERNAL = "--external" in sys.argv
_captured = []

if not EXTERNAL:
    import webbrowser

    def _capture(url, *a, **k):
        _captured.append(url)
        return True

    webbrowser.open = _capture
    webbrowser.open_new = _capture
    webbrowser.open_new_tab = _capture

from vpython import (canvas, box, sphere, cylinder, cone, pyramid, ring,  # noqa: E402,F401
                     curve, label, vertex, triangle, quad, vector, color,
                     arrow, wtext, button, slider, compound,
                     extrusion, shapes, paths)
from vpython import rate as _vp_rate                                   # noqa: E402
import math                                                             # noqa: E402,F401

V = vector

# palette roughly matching the 2D scenes
C_A = V(0.13, 0.53, 0.70)     # blue
C_B = V(0.18, 0.62, 0.55)     # teal
C_C = V(0.90, 0.49, 0.13)     # orange
C_D = V(0.85, 0.75, 0.16)     # yellow
C_E = V(0.78, 0.25, 0.22)     # red
C_F = V(0.55, 0.36, 0.75)     # purple
C_HL = V(0.95, 0.95, 0.60)    # highlight
BG = V(0.06, 0.06, 0.08)      # background (also the colour of "holes")


def new_canvas(title, subtitle="", rng=5.0,
               forward=V(-0.62, -0.42, -0.66), centre=V(0, 0, 0)):
    """A canvas with the view framed explicitly.

    vpython's autoscale is unreliable once objects move, so every scene sets
    its own `range` (half the view height in world units) and viewing
    direction rather than letting the first frame decide. The canvas is
    980 x 600 px; the visible half-width is range * 980/600.

    Note (vpython 7.6): the browser uses the `forward` given here, but the
    Python-side sc.forward keeps reading (0, 0, -1) until the user turns the
    view. A scene that needs the view direction should keep its own copy of
    `forward` (or assign sc.forward again after this call).
    """
    sc = canvas(title=f"<b>{title}</b><br><i>{subtitle}</i><br>",
                width=980, height=600, background=BG,
                forward=forward, range=rng, center=centre)
    sc.ambient = V(0.42, 0.42, 0.42)
    sc.autoscale = False
    _veil(sc, centre, rng, forward)
    return sc


# ---- a clean start ----------------------------------------------------------
# A scene builds its pieces before it starts to move: some are made and then
# hidden, faded or moved into place, and building can take several seconds.
# Without care the page shows all of that as it happens. So while a scene is
# being built the browser's view looks at an empty spot far to one side (with
# a short note), and it comes back to the scene at the scene's first
# sc.waitfor(...) or rate(...), when everything has been made. Python's own
# sc.center keeps its real value throughout, so scene code is unaffected.

_VEILED = []


def _veil(sc, centre, rng, forward):
    try:
        side = vector(forward).cross(V(0, 1, 0))
        side = side.norm() if side.mag > 1e-6 else V(1, 0, 0)
        far = vector(centre) + side * (1000.0 * max(float(rng), 1.0))
        note = label(pos=far, text="building the scene …", height=14,
                     box=False, line=False, opacity=0, color=V(0.62, 0.62, 0.66))
        sc.appendcmd({"center": far.value})    # the browser only
    except Exception:                           # a different vpython: no veil
        return
    _VEILED.append((sc, vector(centre), far, note))
    plain_waitfor = sc.waitfor

    def waitfor(eventtype, _plain=plain_waitfor):
        _unveil()
        return _plain(eventtype)

    sc.waitfor = waitfor


def _unveil():
    while _VEILED:
        sc, real, far, note = _VEILED.pop()
        try:
            c = sc.center
            if (c - far).mag < 1e-6 * far.mag:  # the page reported the far view back
                c = real
            sc.center = c                         # sends the real view to the browser
            note.visible = False
        except Exception:
            pass


def rate(n):
    """vpython's rate(); the first call also brings the view to the scene."""
    if _VEILED:
        _unveil()
    return _vp_rate(n)


def announce():
    """Tell the launcher where the canvas is being served."""
    if EXTERNAL:
        return
    # vpython opens its "browser" when the first canvas or object is made,
    # which comes after the scene's own set-up and checks; on a slow or busy
    # machine that can take several seconds. The GUI waits for this line, so
    # wait for the real URL rather than guessing one.
    deadline = time.time() + 120
    while not _captured and time.time() < deadline:
        time.sleep(0.05)
    url = _captured[0] if _captured else "http://localhost:7000"
    print(f"PWW_URL {url}", flush=True)


def spin(objs=None, seconds=None):
    """Idle loop; the canvas stays live for the user to drag."""
    while True:
        rate(30)


def _find_scene(eid, builtin_globals):
    """scene_<eid> from the launcher itself, else from the one plugin
    module that defines it (found by AST, so a broken plugin elsewhere
    cannot stop this scene from starting)."""
    fn = builtin_globals.get(f"scene_{eid}")
    if callable(fn):
        return fn
    import importlib.util
    from pathlib import Path
    from pww_discover import discover
    here = Path(__file__).resolve().parent
    found, _ = discover(here)
    ref = found.get(eid, {}).get("vpython")
    if ref is None:
        return None
    spec = importlib.util.spec_from_file_location(ref.file.stem, ref.file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, f"scene_{eid}", None)


def run_cli(builtin_globals):
    """Entry point used by pww_3d.py."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fn = _find_scene(args[0], builtin_globals) if args else None
    if fn is None:
        print("usage: python pww_3d.py <ID> [--external]   "
              "(no 3D scene found for that id)")
        return 2
    import threading
    threading.Thread(target=announce, daemon=True).start()
    fn()
    return 0
