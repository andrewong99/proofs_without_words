# SPDX-License-Identifier: GPL-3.0-or-later
"""pww_render.py -- how the app renders a 2D proof.

This is manim's own command line with one difference: a check(...) that is
not met on this machine is reported, and the render carries on.

    python pww_render.py render -ql --media_dir DIR pww_scenes_x.py SCENE

Why: many checks in the scenes are about layout -- a label clear of a line,
a caption inside the frame -- and text is measured with the fonts and the
text engine installed here, which differ between machines. Every check was
met when the scene was made; a few may miss by a hair elsewhere, and that
should not stop the animation from being shown.

To keep every check fatal (when writing or changing a scene), run manim
directly:

    python -m manim render -ql pww_scenes_x.py SCENE
"""
import sys

import pww_kit

_unmet = {}


def _report(cond, what):
    """pww_kit.check, but reporting instead of raising."""
    if cond:
        return
    _unmet[what] = _unmet.get(what, 0) + 1
    if _unmet[what] > 1:                 # say it once, not once per frame
        return
    msg = f"[pww] check not met on this machine, rendering anyway: {what}"
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:           # a console that cannot show it
        print(msg.encode("ascii", "backslashreplace").decode(), flush=True)


# Every scene module does "from pww_kit import *" when manim loads it, which
# happens after this line, so the scenes (and the kit's own helpers) pick up
# the reporting version.
pww_kit.check = _report

from manim.__main__ import main  # noqa: E402  (after the patch, on purpose)

if __name__ == "__main__":
    sys.exit(main())
