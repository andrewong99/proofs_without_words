# pww · proofs without words · 无字证明

A catalogue browser for proofs without words: **403 proofs in 12
categories, ranked by popularity**, each with its statement, its formula and,
where it matters, a note on what the picture does and does not show. The
animations play in 2D with **manim** or in 3D with **vpython** — every
entry is animated: 366 in 2D, 37 in 3D.

**Windows:** double-click **`pww.bat`**. The first time, it sets up its own
Python and packages inside the folder (see below), then opens the app.

**Anywhere else:** `python -m pip install -r requirements.txt`, then
`python pww_app.py` (see [Install](#install)).

---

## Launching with pww.bat

`pww.bat` finds a Python that has the app's packages and starts the app with
that Python's `pythonw.exe` — no console window stays open behind it.

It tries, in order, and takes the first interpreter that can import PyQt6
and matplotlib **and** can find manim and vpython:

1. `PWW_PYTHON` — set on the marked line at the top of `pww.bat`, or as an
   environment variable, to force a particular `python.exe`
2. a `.venv` folder beside the app
3. `.runtime\venv` — the one its own setup makes (below)
4. `python` on `PATH`
5. the `py` launcher — `py -3.12`, `py -3.13`, then `py -3`
6. `%LOCALAPPDATA%\Programs\Python\Python312`

### First run: it sets itself up

If none of them has the packages, `pww.bat` explains what it will do and,
after a 30-second countdown (press **N** to say no), puts everything inside
the folder, in **`.runtime`**:

1. **uv** 0.12.19, a Python installer, downloaded from PyPI and checked
   against the SHA-256 written in `pww.bat` before it is used;
2. **Python 3.12** — the one already on the computer if there is one,
   otherwise uv downloads it into `.runtime\python` (about 25 MB);
3. **`.runtime\venv`**, a virtual environment with the packages in
   `requirements.txt` (about 400 MB to download, 1.3 GB on disk).

Then it opens the app. It takes a few minutes, once. PATH, the registry and
any other Python on the computer are left alone; delete `.runtime` to undo it
all. If the setup stops (no internet, say), it says why and waits; running
`pww.bat` again carries on from where it stopped. When a later version of
pww changes `requirements.txt`, the next start installs what is new before
opening the app.

* It needs Windows 10 (version 1803 or newer) or 11, for their built-in
  `curl` and `tar`, and an internet connection the first time.
* Keep the folder's path short (under about 60 characters, like
  `C:\Users\you\Desktop\pww`): one JupyterLab file that vpython pulls in
  has a very long name, and Windows limits paths to 260 characters.
* Behind a company proxy that inspects HTTPS, set `UV_SYSTEM_CERTS=1` so uv
  uses the Windows certificate store.
* If you move the folder, the next start sets `.runtime` up again.
* To skip the setup and manage Python yourself, set `PWW_NO_SETUP=1`; then
  `pww.bat` behaves as before: if the only Python it finds can open the
  window but lacks manim or vpython, it names it, prints the `pip install`
  line for it, and opens the catalogue anyway.

Keep `pww.bat` beside `pww_app.py`. For a desktop icon, use a shortcut
(right-click `pww.bat` → *Send to* → *Desktop*) rather than a copy.

**When something fails with no console.** A `pythonw` app has nowhere to
print, and PyQt6's default for an exception escaping a slot is to abort the
whole process — the window would simply vanish. So the app appends every
uncaught traceback to **`pww_error.log`** beside it, shows it in a dialog, and
keeps running. The file only appears if something has actually gone wrong.

---

## Why it is built this way

Neither library can draw into a Qt widget:

* **manim** renders to an **mp4 file** (encoded through PyAV). Its Cairo
  renderer has no live widget at all.
* **vpython** serves its canvas from a **local HTTP server** and cannot be
  torn down and rebuilt inside a running interpreter — one scene per process.

So the GUI is a **catalogue and dispatcher**, not a canvas:

| | how it runs | shown in |
|---|---|---|
| 2D proof | `manim` subprocess → mp4, cached under `cache/` | in-panel video player |
| 3D proof | `pww_3d.py` subprocess → local server | embedded web view |

A 2D proof renders once: the mp4 is cached per scene and per quality, so
replays are instant. A 3D proof gets its own process, closed when you open
another one.

The split follows the material: manim for dissections, shears and
rearrangements, where the **sequence of moves is the proof**; vpython for
solids you have to **turn in your hand** to believe — right-drag (or
Ctrl+drag) to rotate, Shift+drag to pan, scroll to zoom.

### Playback

`Play 2D` loads the proof; the transport bar under the video controls it.
A proof shows only the buttons for the scenes it has: no `Open 3D` for a 2D
proof, and no `Play 2D`, `re-render` or quality box for a 3D one.

| | |
|---|---|
| `⏮` | back to the start |
| `▶` `❚❚` `↻` | play · pause · replay |
| slider | drag to scrub, or **click anywhere** on it to jump there |
| `0:04 / 0:09` | position / length |
| `space` | play / pause (ignored while typing in the filter box) |
| `←` `→` | step 2 s back / forward |

**It plays once and stops on the closing frame** — the finished figure with
its formula — and stays there. It never loops; it replays only when you press
`↻`. Scrubbing back off the final frame re-arms `▶` so playback resumes from
where you dropped it rather than restarting.

**No LaTeX required.** The scenes use `Text` with Unicode superscripts, never
`MathTex`; the formula panel uses matplotlib's *mathtext*. Nothing here needs a
TeX Live or MiKTeX install.

### The formula: LaTeX and GeoGebra

The formula above the video has two tabs, and `⧉ copy` puts the one you are
looking at on the clipboard:

* **Formula (LaTeX)** — the formula as shown, with its LaTeX source under it.
  Paste it inside `$ … $` or `\[ … \]`; it uses `amsmath` and `amssymb`.
  All 403 compile with pdfLaTeX.
* **GeoGebra** — the same statement in GeoGebra's input syntax, from
  `pww_geogebra.py`. Identities in letters go into GeoGebra's **CAS view**
  (`Sum(2k - 1, k, 1, n) = n^2`). Relations between points use GeoGebra's
  geometry commands and `==`
  (`Distance(P, A) Distance(P, B) == Distance(P, C) Distance(P, D)`); pasted
  into a construction with those points, they come out true or false. A
  statement GeoGebra has no notation for (√2 is irrational …) is a text
  object. The conventions are listed at the top of `pww_geogebra.py`.

Every GeoGebra entry was checked in GeoGebra itself (the 5.0 web app): all
403 parse; the 144 identities its CAS can simplify came out as claimed; and
the geometry relations were evaluated on constructed figures.

### Menus

* **Language · 语言** — *中文 + English* (the default) or *English*: the
  English interface drops every Chinese title and label. The choice is kept
  in `pww_settings.ini` beside the app.
* **Cache · 缓存**
  * *Clear cache…* — deletes the rendered videos: all of them, or only the
    old versions the app no longer uses (scenes that have changed since, or
    folders from an earlier version of pww). It shows how much each frees.
    The `clear cache…` button beside the quality box does the same.
  * *Clear temporary files…* — deletes manim's `partial_movie_files`, the
    clips it joins into each video; the finished videos stay.
  * *Delete temporary files after each render* — on by default, so those
    clips never pile up.
  * *Open cache folder*.

---

## Install

On Windows, `pww.bat` does this for you (see above). By hand, or on macOS and
Linux:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists manim (0.19 to 0.21: the scenes are tested with
0.21, and manim's minor versions change its API), vpython, PyQt6,
PyQt6-WebEngine, matplotlib and numpy, plus `setuptools<81`, because vpython
7.6 still imports `pkg_resources`, which setuptools 81 removed.

**Which Python.** 3.10 or newer; on Windows, **3.12 is the easy choice**:
every package has a ready-made wheel for it. As of October 2026, manim's
`moderngl` and `glcontext` have no Windows wheels for Python 3.14, so pip has
to compile them there, which needs Microsoft's C++ Build Tools; vpython's
wheels stop at 3.12 (on newer Pythons it builds from source, and without a
compiler leaves out its optional compiled speed-up). `pww.bat`'s own setup
always uses 3.12. To give pww a 3.12 of your own instead, in a `.venv` beside
the app — `pww.bat` looks there first — in the pww folder:

```bat
py install 3.12
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`py install` comes with Python's install manager; with the older `py`
launcher, install Python 3.12 from python.org instead of the first line.
Your main Python is left as it is.

**No separate ffmpeg is needed.** manim encodes and joins its video fragments
through **PyAV**, whose wheels bundle the FFmpeg libraries (manim's only
remaining subprocess calls are for LaTeX, which this app never uses). Qt's
player likewise ships its own FFmpeg backend.

Platform notes:

* **Windows / macOS** — the wheels bundle everything; nothing else to install.
* **Linux** — manim's pycairo/manimpango may build from source:
  `sudo apt install libcairo2-dev libpango1.0-dev pkg-config`
* Use a **virtualenv** if the system setuptools is old — manim's `srt`
  dependency fails to build against it.

Two optional pieces degrade gracefully rather than failing:

| missing | what happens instead |
|---|---|
| `PyQt6-WebEngine` | 3D proofs open in your default browser |
| PyQt6 multimedia | mp4s open in your system video player |

The app tells you in the status bar when it is running degraded.

---

## Files

```
pww_app.py            the PyQt6 GUI. This is the one you run.
pww.bat               Windows launcher: finds the right Python (or sets one
                      up in .runtime the first time), no console.
requirements.txt      pip install -r requirements.txt
pww_registry.py       all 403 entries: titles in English and Chinese,
                      statements, formulas, notes, popularity. Pure data.
pww_discover.py       finds the scenes by name (see "Adding a proof").
pww_kit.py            shared toolkit for the 2D scenes.
pww_render.py         how the app renders a 2D proof: manim's command line,
                      with a check that is not met logged instead of fatal.
pww_scenes.py         2D scenes, and pww_scenes_*.py: more 2D scenes.
pww_3d_kit.py         shared machinery for the 3D scenes.
pww_3d.py             3D launcher and scenes, and pww_3d_*.py: more 3D
                      scenes. Any 3D proof runs standalone too:
                          python pww_3d.py I2
pww_geogebra.py       every formula in GeoGebra syntax (the GeoGebra tab).
cache/                rendered mp4s. Safe to delete; they re-render on demand.
pww_settings.ini      your language and cache choices (made by the app).
```

The rendered videos are in **`cache\`** beside the app: one folder per proof
and quality, named `<Scene>-<quality>-<hash>`, e.g.
`cache\B2_OddSquares-qm-4428022b\videos\pww_scenes\720p30\B2_OddSquares.mp4`. The hash is
taken from the scene's code (and the shared kit it uses), so editing a scene
re-renders it automatically, and the old folder becomes an "old version" that
*Cache → Clear cache…* can remove. 3D proofs are never cached.

The scenes are kept out of the GUI module for a concrete reason: manim's CLI
**re-imports the scene file on every render**, so a render never drags PyQt6
in with it.

`.gitignore` keeps `cache/`, `.runtime/`, `.venv/`, `__pycache__/`,
`pww_error.log`, `pww_settings.ini`, `book/` and `_private/` out of any
repository: reference material you keep for yourself
belongs in `book/` or `_private/`. `.gitattributes` has git keep
`pww.bat` byte for byte, with the Windows line endings cmd.exe needs to
find the labels in a batch file reliably.

---

## Ranking

The list opens **ranked by popularity**, in three tiers — *Classics* (the
results and pictures nearly everyone meets at school), *Standard* (widely
taught or well known) and *Specialist* (for enthusiasts, olympiad geometry,
curios). Each entry has an editorial score from 0 to 100 (`pop` in
`pww_registry.py`); the rank is the order of those scores. Switch the box
above the list to *By category* for the topic view. The filter box works in
both.

## What is animated

All 403 entries are animated (● in the list; ○ would mark an entry still
waiting for its scene). These 37 are in 3D:

`B3` sum of squares (six pyramids fill a box) · `B8` sum of triangular
numbers · `B29` hexagons make cubes · `C4` (a+b)³ · `C5` a³ − b³ ·
`C11` a³ + b³ · `C16` growth of a cube · `E20` Dandelin spheres ·
`H9` derivative of x³ · `H21` a ball grows by its surface ·
`I1` frustum · `I2` cone = ⅓ cylinder · `I3` sphere (祖暅原理) ·
`I4` sphere's surface (hat-box) · `I5` space diagonal · `I6` de Gua ·
`I7` cube → three pyramids · `I8` bicylinder (牟合方盖) · `I9` napkin ring ·
`I11` cone's lateral surface · `I12` Cavalieri's stack of coins ·
`I13` cube as six pyramids · `I14` prism as three tetrahedra ·
`I15` ball as cones from its centre · `I16` torus · `I17` regular
tetrahedron in a cube · `I18` only five Platonic solids · `I19` Girard's
theorem · `I20` rhombic dodecahedron · `I21` Descartes' angle defect ·
`I22` octahedron in a cube · `I23` cylinder's surface · `I24` slanted slice
of a cylinder · `I25` twelve pentagons on a football · `I26` Menger sponge ·
`I27` stepped pyramid · `K29` five points on a sphere

The others are 2D. Run `python pww_registry.py` for the counts per
category.

A 3D scene shows "building the scene …" while it makes its pieces (a few
seconds for the larger ones), then starts.

---

## Adding a proof

The name is the registration. No list to edit:

* **2D** — a class `<ID>_<Name>(Board)` in any file named `pww_scenes*.py`,
  starting with `from pww_kit import *`.
* **3D** — a function `scene_<ID>()` in any file named `pww_3d*.py`, starting
  with `from pww_3d_kit import *`.

The ID is the catalogue entry's (`A1`, `B12`, …). The GUI finds the scene on
its next start and the entry's `○` becomes `●`. A broken file is reported
and skipped; it cannot stop the other scenes from loading.

Constraints worth keeping:

* **No LaTeX.** No `MathTex`, `Tex`, `Brace`, `DecimalNumber`, `Integer`,
  `Matrix`, or axes with number labels — every one of them pulls in LaTeX.
  Use `tag()` / `caption()` (manim `Text`) with Unicode: `a²`, `√`, `φ`, `≥`.
* **The animation is the proof.** Pieces must be congruent and tilings
  gap-free; a shear keeps its base and its parallels; a rotation is rigid.
  Check the geometry in the scene with `check(...)`, so a wrong construction
  fails the render instead of animating a wrong picture. (The app renders
  through `pww_render.py`, which logs a check that is not met and renders
  anyway: a label's clearance is measured with the fonts installed, which
  differ between machines. Run `python -m manim render -ql <file> <Scene>`
  to keep every check fatal.)
* **End on the result.** Finish with `caption(...)` on screen and
  `self.hold(2.2)`; the player parks on that frame.
* **Formulas in the registry are mathtext, not full LaTeX.** It has no `\ge`,
  `\le` or `\cfrac` — use `\geq`, `\leq`, `\frac`.
* **Your own drawing.** Design each figure yourself from the mathematical
  idea, and name only the theorem (Pythagoras, Ptolemy, Pick, …) — not the
  book, article or collection you met it in.

---

## A note on rigour

`J8` is Curry's missing-square paradox, and it is in the catalogue on purpose.
A picture is a proof only when its implicit claims are themselves checked; in
Curry's figure the "hypotenuse" is bent and the eye does not see it.

For the same reason `I2` (cone = ⅓ cylinder) is not the usual demonstration
— pouring three cones into a cylinder — which shows the result without
proving it. It is Cavalieri's: a cone and a pyramid of equal base area and
height have equal cross-sections at every height, and `I7` gives the
pyramid's ⅓. (Not via `I3`: that argument *uses* the cone's volume, so citing
it would be circular.)

Where a picture covers only part of a theorem, the entry's note says so —
`A7` draws the acute case, `F8` the convex quadrilateral, `J7` one point of
the disc, `J9` one triangle. The theorem is stated in full; the note marks
the scope of the drawing.

---

## Licence

Copyright (C) 2026 the pww authors.

pww is free software: you can redistribute it and/or modify it under the
terms of the **GNU General Public License, version 3 or (at your option) any
later version** — see [`LICENSE`](LICENSE). Every source file carries the tag
`SPDX-License-Identifier: GPL-3.0-or-later`.

It is built on PyQt6 and PyQt6-WebEngine (GPL v3), manim (MIT), vpython
(MIT-style), matplotlib and numpy (permissive). PyQt6's GPL v3 is the reason
the app is GPL v3 rather than v2.
