#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_app.py — Proofs Without Words / 无字证明

    python pww_app.py          (or double-click pww.bat on Windows)

A catalogue browser and dispatcher. It does not draw the proofs itself:
manim renders each 2D proof to an mp4 (cached, so a proof renders once ever)
and vpython serves each 3D proof from its own process. Neither library can
draw into a Qt widget, so the GUI owns the list, the formula panel and the
process handling, and hands the drawing to whichever engine suits the proof.

Requires: PyQt6, matplotlib, manim, vpython. No separate ffmpeg install:
manim (>= 0.18) encodes through PyAV, whose wheels bundle FFmpeg.
Optional: PyQt6-WebEngine (embeds the 3D canvas; without it, 3D proofs open
in your browser) and PyQt6 multimedia (plays the mp4 in-panel; without it,
the video opens in your system player).
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# ---- windowless launches (pythonw.exe, as used by pww.bat) ---------------
# pythonw has no console, so sys.stderr and sys.stdout are None and any
# traceback -- including a failed import below -- would simply vanish.
# Route stderr to pww_error.log instead. The file is created only if
# something is actually written, so a healthy run leaves nothing behind.
ERROR_LOG = Path(__file__).resolve().parent / "pww_error.log"


class _ErrorLogStream(io.TextIOBase):
    def writable(self):
        return True

    def write(self, s):
        if s:
            try:
                with open(ERROR_LOG, "a", encoding="utf-8") as f:
                    f.write(s)
            except OSError:
                pass
        return len(s)


if sys.stderr is None:
    sys.stderr = _ErrorLogStream()
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")

from PyQt6.QtCore import (Qt, QThread, QProcess, QUrl, pyqtSignal, QTimer,
                          QSettings)
from PyQt6.QtGui import (QPixmap, QColor, QPalette, QFont, QDesktopServices,
                         QAction, QActionGroup)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QPlainTextEdit, QStackedWidget, QMessageBox,
    QComboBox, QProgressBar, QHeaderView, QSlider, QTabWidget,
)

import pww_registry as reg
import pww_geogebra as ggb
from pww_discover import SCENE_CLASS, SceneRef, discover, imports_module

HERE = Path(__file__).resolve().parent
THREED_FILE = HERE / "pww_3d.py"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
# the user's choices (language, cache options), kept beside the app
SETTINGS_FILE = HERE / "pww_settings.ini"

GGB_HINT = ("Paste into GeoGebra. Identities in letters: the CAS view. "
            "Relations between points (A, B, P …): a construction with those "
            "points, where they come out true or false. GCD and Mod: whole "
            "numbers.")
GGB_TEXT_HINT = ("GeoGebra has no notation for this statement, so it is given "
                 "as a GeoGebra text object: it pastes as text.")

QUALITY = {"480p (fast)": "-ql", "720p": "-qm", "1080p (slow)": "-qh"}

# How far before the true end to park. Every scene closes on a static hold of
# at least 1.2 s showing the finished figure and its formula, so stopping a
# few frames early looks identical and avoids the black frame some backends
# show at EndOfMedia.
END_GUARD_MS = 150

# ---- optional Qt modules -------------------------------------------------
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    HAVE_MEDIA = True
except Exception:                                    # pragma: no cover
    HAVE_MEDIA = False

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAVE_WEB = True
except Exception:                                    # pragma: no cover
    HAVE_WEB = False

# ---- colours -------------------------------------------------------------
BG = "#101014"
PANEL = "#17171d"
FG = "#e8e8ee"
DIM = "#8b8b99"
ACCENT = "#f2c94c"
OK = "#5bc9a4"
STUB = "#6c6c7a"


# ==========================================================================
# formula rendering (matplotlib mathtext -> QPixmap; no LaTeX install needed)
# ==========================================================================

_formula_cache: dict[tuple, bytes] = {}


def formula_png(latex: str, dpi: int = 200, size: int = 17,
                color: str = ACCENT) -> bytes | None:
    """Render a mathtext string to transparent PNG bytes, or None if it
    does not parse. Results are memoised for the life of the process."""
    key = (latex, dpi, size, color)
    if key in _formula_cache:
        return _formula_cache[key]
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(0.1, 0.1))
        FigureCanvasAgg(fig)
        fig.patch.set_alpha(0.0)
        fig.text(0.0, 0.0, f"${latex}$", fontsize=size, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.08)
        data = buf.getvalue()
    except Exception:
        return None
    _formula_cache[key] = data
    return data


# ==========================================================================
# manim rendering, off the UI thread
# ==========================================================================

class RenderThread(QThread):
    finished_ok = pyqtSignal(str, str)      # scene, mp4 path ("" on failure)
    line = pyqtSignal(str)

    def __init__(self, ref: SceneRef, quality_flag: str):
        super().__init__()
        self.ref = ref
        self.scene = ref.name
        self.flag = quality_flag
        self._proc: subprocess.Popen | None = None
        # fixed here, on the GUI thread when Play is pressed: the render
        # lands where the source as it was at that moment says it should
        self._out = cache_dir(ref, quality_flag)

    def outdir(self) -> Path:
        return self._out

    def run(self):
        out = self.outdir()
        out.mkdir(parents=True, exist_ok=True)
        # pww_render.py is manim's command line, except that a layout check
        # missed on this machine's fonts is logged rather than fatal
        runner = HERE / "pww_render.py"
        via = [str(runner)] if runner.exists() else ["-m", "manim"]
        cmd = [sys.executable, *via, "render", self.flag,
               "--media_dir", str(out), str(self.ref.file), self.scene]
        self.line.emit("$ " + " ".join(cmd))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Fix the pipe encoding at both ends. Left to defaults, Windows uses
        # the ANSI code page (cp1252, cp936 ...) and a character outside it
        # in manim's progress output would raise mid-render.
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(HERE),
                # explicit: when this app runs under pythonw.exe there is no
                # stdin handle to inherit
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=env, bufsize=1)
        except Exception as exc:
            self.line.emit(f"could not start manim: {exc}")
            self.finished_ok.emit(self.scene, "")
            return
        result = ""
        try:
            for raw in self._proc.stdout:                  # type: ignore
                txt = raw.rstrip()
                if txt:
                    self.line.emit(txt)
            self._proc.wait()
            result = find_final_mp4(out, self.scene) or ""
        except Exception as exc:
            self.line.emit(f"render aborted: {type(exc).__name__}: {exc}")
        finally:
            # always release the UI, whatever happened above
            self.finished_ok.emit(self.scene, result)

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def find_final_mp4(out: Path, scene: str) -> str | None:
    """The assembled video for `scene`, never one of manim's fragments.

    manim leaves dozens of clips under partial_movie_files/ next to the
    finished video. Picking the newest .mp4 in the tree usually lands on the
    right one but not always, so the fragments are excluded outright and the
    file named after the scene is preferred.
    """
    if not out.is_dir():
        return None
    finals = [p for p in out.rglob("*.mp4")
              if "partial_movie_files" not in p.parts]
    if not finals:
        return None
    exact = [p for p in finals if p.stem == scene]
    pool = exact or finals
    return str(max(pool, key=lambda p: p.stat().st_mtime))


KIT_FILE = HERE / "pww_kit.py"


def _render_parts(path: Path, scene: str | None, memo: dict | None = None):
    """AST dumps of everything in `path` that can affect how `scene` renders:
    all top-level code except the module docstring and the other scene
    classes.

    `memo`, when given, keeps each file's parse and node dumps between calls,
    for fingerprinting many scenes at once (clearing the cache)."""
    if memo is not None and path in memo:
        tree, dumps = memo[path]
    else:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dumps = {}
        if memo is not None:
            memo[path] = (tree, dumps)
    parts = []
    for i, node in enumerate(tree.body):
        if (i == 0 and isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)):
            continue                                    # module docstring
        if (isinstance(node, ast.ClassDef) and node.name != scene
                and SCENE_CLASS.match(node.name)):
            continue                                    # another scene
        if i not in dumps:
            dumps[i] = ast.dump(node)
        parts.append(dumps[i])
    return parts, tree


def scene_fingerprint(ref: SceneRef, memo: dict | None = None) -> str:
    """8-hex digest of everything that decides how a scene renders.

    The cache key is scene name + quality + this hash, so an edited scene is
    rendered again instead of being served from its old video. The hash
    covers the scene's own class plus its module's shared code (imports,
    helpers, base classes, constants) and, for modules built on pww_kit, the
    kit. Other scene classes are left out, so editing or adding one scene
    invalidates only that scene. Hashed from the AST, not the text:
    comments, blank lines and shifted line numbers change nothing.
    """
    try:
        parts, tree = _render_parts(ref.file, ref.name, memo)
        if imports_module(tree, "pww_kit"):
            kit_parts, _ = _render_parts(KIT_FILE, None, memo)
            parts = kit_parts + parts
    except (OSError, SyntaxError, ValueError):
        return "00000000"            # unreadable source: manim will say why
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:8]


def cache_dir(ref: SceneRef, flag: str) -> Path:
    return CACHE / f"{ref.name}{flag}-{scene_fingerprint(ref)}"


def cached_mp4(ref: SceneRef, flag: str) -> str | None:
    return find_final_mp4(cache_dir(ref, flag), ref.name)


# ---- clearing the cache ----------------------------------------------------
_CACHE_NAME = re.compile(
    r"^(?P<scene>.+?)(?P<flag>-q[lmh])-(?P<hash>[0-9a-f]{8})$")


def survey_cache(found: dict) -> tuple[list[Path], list[Path]]:
    """Everything in cache/, split into (current, stale).

    Current: a folder named <Scene><flag>-<hash> whose hash is that scene's
    fingerprint now, i.e. a video the app would play. Stale: anything else --
    an older hash (the scene has been changed since), a scene that no longer
    exists, or a name left by an earlier version of the app.
    """
    scenes = {ref.name: ref for slot in found.values()
              for ref in slot.values() if ref.engine == "manim"}
    prints: dict[str, str] = {}
    memo: dict = {}                 # each module parsed once for the survey
    current: list[Path] = []
    stale: list[Path] = []
    if not CACHE.is_dir():
        return current, stale
    for p in sorted(CACHE.iterdir()):
        m = _CACHE_NAME.match(p.name) if p.is_dir() else None
        ref = scenes.get(m["scene"]) if m else None
        if ref is None:
            stale.append(p)
            continue
        if ref.name not in prints:
            prints[ref.name] = scene_fingerprint(ref, memo)
        (current if prints[ref.name] == m["hash"] else stale).append(p)
    return current, stale


def disk_usage(paths) -> tuple[int, int]:
    """(bytes, finished videos) in the given files and folders."""
    size = videos = 0
    for p in paths:
        if not p.is_dir():
            try:
                size += p.stat().st_size
            except OSError:
                pass
            continue
        for root, _dirs, files in os.walk(p):
            partial = "partial_movie_files" in Path(root).parts
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
                if f.endswith(".mp4") and not partial:
                    videos += 1
    return size, videos


def remove_path(p: Path) -> bool:
    """Delete a file or a folder tree; True if it is gone afterwards.

    A read-only file is made writable and tried again. A link is removed
    itself, never followed. A file another program has open is left, for
    the caller to report.
    """
    def again(func, path, _exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass

    try:
        if p.is_symlink() or not p.is_dir():
            p.unlink()
        elif sys.version_info >= (3, 12):
            shutil.rmtree(p, onexc=again)
        else:
            shutil.rmtree(p, onerror=again)
    except OSError:
        pass
    return not os.path.lexists(p)


def human_size(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


def fragment_dirs() -> list[Path]:
    """manim's partial_movie_files folders: the clips it joins into each
    finished video. Temporary: nothing plays them once the video exists."""
    if not CACHE.is_dir():
        return []
    return sorted(p for p in CACHE.glob("*/videos/*/*/partial_movie_files")
                  if p.is_dir() and not p.is_symlink())


# ==========================================================================
# widgets
# ==========================================================================

class SeekSlider(QSlider):
    """A slider that jumps to where you click, instead of paging towards it.

    QSlider's default is to step by pageStep when the groove is clicked,
    which makes scrubbing a video feel broken.
    """

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.LeftButton
                and self.maximum() > self.minimum()):
            frac = ev.position().x() / max(self.width(), 1)
            frac = min(max(frac, 0.0), 1.0)
            span = self.maximum() - self.minimum()
            self.setValue(self.minimum() + round(frac * span))
            self.sliderMoved.emit(self.value())
        super().mousePressEvent(ev)      # the handle is now under the cursor,
                                         # so this begins a normal drag


class FormulaLabel(QLabel):
    """Shows the identity, falling back to plain text if mathtext chokes."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(64)
        self.setStyleSheet(
            f"background:{PANEL}; border:1px solid #26262f; border-radius:6px;"
            f" color:{ACCENT}; padding:8px;")

    def show_formula(self, latex: str):
        data = formula_png(latex)
        if data:
            pm = QPixmap()
            pm.loadFromData(data)
            pm.setDevicePixelRatio(2.0)   # rendered at 200 dpi, shown at 100
            self.setPixmap(pm)
            self.setText("")
        else:
            self.setPixmap(QPixmap())
            f = QFont("monospace")
            f.setPointSize(11)
            self.setFont(f)
            self.setText(latex)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proofs Without Words · 无字证明")
        self.resize(1380, 880)
        self.current: reg.ProofEntry | None = None
        self.render_thread: RenderThread | None = None
        self.vp_proc: QProcess | None = None
        self._scrubbing = False       # user has the seek handle
        self._ended = False           # parked on the closing frame
        # Which entries have animations is decided by the scene files
        # themselves (pww_discover): a class named B3_... is B3's animation.
        self.found, self.problems = discover(HERE)
        self._check_registry_bindings()
        self.settings = QSettings(str(SETTINGS_FILE),
                                  QSettings.Format.IniFormat)
        self.lang = self.settings.value("ui/language", "both")
        if self.lang not in ("both", "en"):
            self.lang = "both"
        self.auto_clean = self.settings.value("cache/delete_fragments", True,
                                              type=bool)

        self._build()
        self._build_menus()
        self._retranslate()           # sets every label, fills the list
        if self.problems:
            self.log.appendPlainText("scene discovery:")
            for msg in self.problems:
                self.log.appendPlainText("  " + msg)

    # ------------------------------------------------------------------ UI
    def _build(self):
        split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)

        # ---------------- left: search + catalogue
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 4, 8)

        self.search = QLineEdit()
        self.search.textChanged.connect(self._filter)
        lv.addWidget(self.search)

        # the list can be ranked by popularity (default) or grouped by topic
        self.view = QComboBox()
        self.view.addItems(["", ""])            # texts: _retranslate
        self.view.currentIndexChanged.connect(self._repopulate)
        lv.addWidget(self.view)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        # let long bilingual labels run to their full width and scroll,
        # rather than being elided
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.currentItemChanged.connect(self._select)
        lv.addWidget(self.tree, 1)

        self.counts = QLabel()
        self.counts.setStyleSheet(f"color:{DIM}; font-size:11px;")
        lv.addWidget(self.counts)
        split.addWidget(left)

        # ---------------- right: detail
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 8, 10, 8)
        rv.setSpacing(8)

        self.title = QLabel("—")
        self.title.setStyleSheet(
            f"color:{FG}; font-size:19px; font-weight:600;")
        self.title.setWordWrap(True)
        rv.addWidget(self.title)

        self.statement = QLabel("")
        self.statement.setStyleSheet(f"color:{DIM}; font-size:13px;")
        self.statement.setWordWrap(True)
        rv.addWidget(self.statement)

        # ---------------- the formula: LaTeX and GeoGebra, one-click copy
        self.formula_tabs = QTabWidget()
        self.formula_tabs.setDocumentMode(True)
        tab_tex = QWidget()
        vt = QVBoxLayout(tab_tex)
        vt.setContentsMargins(0, 4, 0, 0)
        vt.setSpacing(3)
        self.formula = FormulaLabel()
        vt.addWidget(self.formula)
        self.latex_src = QLabel("")
        self.latex_src.setWordWrap(True)
        self.latex_src.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.latex_src.setStyleSheet(
            f"color:{DIM}; font-family:monospace; font-size:11px;")
        self.latex_src.setToolTip(
            "the LaTeX that copy puts on the clipboard: paste it inside "
            "$ … $ or \\[ … \\]  (uses amsmath and amssymb)")
        vt.addWidget(self.latex_src)
        tab_ggb = QWidget()
        vg = QVBoxLayout(tab_ggb)
        vg.setContentsMargins(0, 4, 0, 0)
        vg.setSpacing(3)
        self.ggb_text = QLabel("")
        self.ggb_text.setWordWrap(True)
        self.ggb_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ggb_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ggb_text.setMinimumHeight(64)
        self.ggb_text.setStyleSheet(
            f"background:{PANEL}; border:1px solid #26262f; border-radius:6px;"
            f" color:{ACCENT}; padding:8px; font-family:monospace;"
            " font-size:15px;")
        vg.addWidget(self.ggb_text)
        self.ggb_hint = QLabel(GGB_HINT)
        self.ggb_hint.setWordWrap(True)
        self.ggb_hint.setStyleSheet(f"color:{DIM}; font-size:11px;")
        vg.addWidget(self.ggb_hint)
        self.formula_tabs.addTab(tab_tex, "")
        self.formula_tabs.addTab(tab_ggb, "GeoGebra")
        self.btn_copy = QPushButton("")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self.copy_formula)
        self.formula_tabs.setCornerWidget(self.btn_copy,
                                          Qt.Corner.TopRightCorner)
        rv.addWidget(self.formula_tabs)

        # ---------------- controls
        bar = QHBoxLayout()
        self.btn2d = QPushButton("▶  Play 2D  (manim)")
        self.btn2d.clicked.connect(self.play_2d)
        self.btn3d = QPushButton("◆  Open 3D  (vpython)")
        self.btn3d.clicked.connect(self.play_3d)
        self.btn_re = QPushButton("↻ re-render")
        self.btn_re.clicked.connect(lambda: self.play_2d(force=True))
        self.quality = QComboBox()
        self.quality.addItems(list(QUALITY))
        self.quality.setCurrentIndex(1)
        self.lbl_quality = QLabel("quality")
        self.btn_clear = QPushButton("")
        self.btn_clear.setToolTip(
            f"delete rendered videos from {CACHE}  (they render again when "
            "played)")
        self.btn_clear.clicked.connect(self.clear_cache)
        # shown by _select, each only when the proof has that kind of scene
        for w in (self.btn2d, self.btn3d, self.btn_re, self.lbl_quality,
                  self.quality):
            w.setVisible(False)
        for w in (self.btn2d, self.btn3d, self.btn_re):
            w.setMinimumHeight(32)
        bar.addWidget(self.btn2d)
        bar.addWidget(self.btn3d)
        bar.addWidget(self.btn_re)
        bar.addStretch(1)
        bar.addWidget(self.lbl_quality)
        bar.addWidget(self.quality)
        bar.addWidget(self.btn_clear)
        rv.addLayout(bar)

        # ---------------- viewer
        self.stack = QStackedWidget()
        self.stack.setMinimumHeight(380)
        self.stack.setStyleSheet("background:#000; border-radius:6px;")

        self.placeholder = QLabel("select a proof")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet(f"color:{DIM}; font-size:14px;")
        self.stack.addWidget(self.placeholder)                     # index 0

        if HAVE_MEDIA:
            self.video = QVideoWidget()
            self.player = QMediaPlayer()
            self.audio = QAudioOutput()
            self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video)
            # play once and stop; the closing hold of every scene shows the
            # finished figure with its formula, and that is where we park
            if hasattr(self.player, "setLoops"):
                self.player.setLoops(1)
            self.player.positionChanged.connect(self._on_position)
            self.player.durationChanged.connect(self._on_duration)
            self.player.playbackStateChanged.connect(self._on_playstate)
            self.player.mediaStatusChanged.connect(self._on_status)
            self.stack.addWidget(self.video)                       # index 1
        else:
            self.video = None
            self.stack.addWidget(QLabel())

        if HAVE_WEB:
            self.web = QWebEngineView()
            self.stack.addWidget(self.web)                         # index 2
        else:
            self.web = None
            self.stack.addWidget(QLabel())

        rv.addWidget(self.stack, 1)

        # ---------------- transport: scrub, pause, replay
        self.transport = QWidget()
        tp = QHBoxLayout(self.transport)
        tp.setContentsMargins(2, 2, 2, 2)
        tp.setSpacing(8)

        self.btn_back = QPushButton("⏮")
        self.btn_back.setFixedWidth(38)
        self.btn_back.setToolTip("back to the start")
        self.btn_back.clicked.connect(self.to_start)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(46)
        self.btn_play.setToolTip("play / pause  (space)")
        self.btn_play.clicked.connect(self.toggle_play)

        self.seek = SeekSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.setToolTip("drag or click to scrub")
        self.seek.sliderPressed.connect(self._seek_begin)
        self.seek.sliderReleased.connect(self._seek_end)
        self.seek.sliderMoved.connect(self._seek_to)

        self.timelabel = QLabel("0:00 / 0:00")
        self.timelabel.setStyleSheet(
            f"color:{DIM}; font-family:monospace; font-size:12px;")
        self.timelabel.setMinimumWidth(96)
        self.timelabel.setAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)

        for w in (self.btn_back, self.btn_play):
            w.setMinimumHeight(28)
        tp.addWidget(self.btn_back)
        tp.addWidget(self.btn_play)
        tp.addWidget(self.seek, 1)
        tp.addWidget(self.timelabel)
        rv.addWidget(self.transport)
        self.transport.setVisible(HAVE_MEDIA)
        self._set_transport_enabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        rv.addWidget(self.progress)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{DIM}; font-size:12px;")
        rv.addWidget(self.note)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(96)
        self.log.setStyleSheet(
            f"background:#0c0c10; color:{DIM}; font-family:monospace;"
            " font-size:11px; border:1px solid #22222a; border-radius:5px;")
        rv.addWidget(self.log)

        split.addWidget(right)
        split.setSizes([440, 940])

        cap = []
        if not HAVE_MEDIA:
            cap.append("PyQt6 multimedia missing — mp4 opens in your player")
        if not HAVE_WEB:
            cap.append("PyQt6-WebEngine missing — 3D opens in your browser")
        self.statusBar().showMessage(
            " · ".join(cap) if cap else "ready", 0 if cap else 4000)

    def refs(self, e) -> dict:
        """{"manim": SceneRef, "vpython": SceneRef} for an entry (may be empty)."""
        return self.found.get(e.id, {})

    def animated(self, e) -> bool:
        return bool(self.refs(e))

    def _check_registry_bindings(self):
        """Explicit bindings left in the registry must agree with the files."""
        for e in reg.CATALOGUE:
            got = self.found.get(e.id, {})
            if e.manim_scene and getattr(got.get("manim"), "name", None) != e.manim_scene:
                self.problems.append(
                    f"{e.id}: registry names {e.manim_scene}, but no such "
                    "class was found in pww_scenes*.py")
            if e.vpython_id and "vpython" not in got:
                self.problems.append(
                    f"{e.id}: registry expects a 3D scene, but no "
                    f"scene_{e.id} was found in pww_3d*.py")

    def keyPressEvent(self, ev):
        """Space toggles playback — unless you are typing in the filter box."""
        if (ev.key() == Qt.Key.Key_Space
                and not self.search.hasFocus()
                and self.stack.currentIndex() == 1):
            self.toggle_play()
            ev.accept()
            return
        if ev.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) \
                and not self.search.hasFocus() \
                and self.stack.currentIndex() == 1 and HAVE_MEDIA:
            step = 2000 if ev.key() == Qt.Key.Key_Right else -2000
            self._seek_to(max(0, min(self.player.position() + step,
                                     self.player.duration())))
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _groups(self):
        """(heading, entries, expanded) for the current view."""
        if self.view.currentIndex() == 0:            # ranked
            out = []
            for i, (lo, en, zh) in enumerate(reg.TIERS):
                items = [e for e in reg.RANKED if reg.tier_of(e) == i]
                out.append((f"Tier {i + 1}.  {self.T(en, zh)}", items,
                            i < 2))
            return out
        return [(f"{letter}.  {self.T(en, zh)}", reg.in_category(letter),
                 letter in ("A", "B"))
                for letter, (en, zh) in reg.CATEGORIES.items()]

    def _populate(self):
        self.tree.clear()
        ranked = self.view.currentIndex() == 0
        for heading, items, expanded in self._groups():
            n = sum(1 for e in items if self.animated(e))
            top = QTreeWidgetItem([f"{heading}    ({n}/{len(items)})"])
            top.setForeground(0, QColor(FG))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for e in items:
                done = self.animated(e)
                mark = "●" if done else "○"
                rank = f"#{reg.RANK[e.id]:<4}" if ranked else ""
                kid = QTreeWidgetItem(
                    [f"  {mark} {rank}{e.id}  {self.T(e.en, e.zh, '  ·  ')}"])
                kid.setData(0, Qt.ItemDataRole.UserRole, e.id)
                kid.setForeground(0, QColor(OK if done else STUB))
                top.addChild(kid)
            self.tree.addTopLevelItem(top)
            top.setExpanded(expanded)
        done = sum(1 for e in reg.CATALOGUE if self.animated(e))
        total = len(reg.CATALOGUE)
        self.counts.setText(
            f"{total} proofs listed · all animated" if done == total else
            f"{total} proofs listed · {done} animated · "
            f"{total - done} awaiting a scene")

    def _repopulate(self, *_):
        keep = self.current.id if self.current else None
        self._populate()
        self._filter(self.search.text())
        if keep:
            for i in range(self.tree.topLevelItemCount()):
                top = self.tree.topLevelItem(i)
                for j in range(top.childCount()):
                    kid = top.child(j)
                    if kid.data(0, Qt.ItemDataRole.UserRole) == keep:
                        top.setExpanded(True)
                        self.tree.setCurrentItem(kid)
                        self.tree.scrollToItem(kid)
                        return

    def _filter(self, text: str):
        t = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                kid = top.child(j)
                eid = kid.data(0, Qt.ItemDataRole.UserRole)
                e = reg.BY_ID[eid]
                hay = " ".join([e.id, e.en, e.zh, e.statement,
                                e.formula, e.note]).lower()
                hit = (t in hay) if t else True
                kid.setHidden(not hit)
                shown += hit
            top.setHidden(shown == 0)
            if t:
                top.setExpanded(shown > 0)

    # ------------------------------------------------------------- actions
    def _select(self, item, _prev):
        if item is None:
            return
        eid = item.data(0, Qt.ItemDataRole.UserRole)
        if not eid:
            return
        e = reg.BY_ID[eid]
        self.current = e
        self._show_entry_text(e)

        # a button only for a kind of scene the proof has: no 3D button for
        # a 2D-only proof, no 2D controls for a 3D-only one
        has_2d = "manim" in self.refs(e)
        has_3d = "vpython" in self.refs(e)
        for w in (self.btn2d, self.btn_re, self.lbl_quality, self.quality):
            w.setVisible(has_2d)
        self.btn3d.setVisible(has_3d)
        self.btn2d.setEnabled(has_2d)
        self.btn_re.setEnabled(has_2d)
        self.btn3d.setEnabled(has_3d)

        self._stop_video()
        if not (has_2d or has_3d):
            self.placeholder.setText(
                f"{e.id} is listed but not animated yet.\n\n"
                f"{e.statement}\n\n"
                f"A class named {e.id}_<Name> in any pww_scenes*.py (2D), or "
                f"a function scene_{e.id} in any pww_3d*.py (3D), is picked "
                "up on the next start.")
        else:
            kinds = " and ".join(
                (["2D · manim"] if has_2d else []) +
                (["3D · vpython"] if has_3d else []))
            self.placeholder.setText(f"{e.id} — {kinds}\n\npress a button above")
        self.stack.setCurrentIndex(0)

    # --------------------------------------------------------------- 2D
    def play_2d(self, force: bool = False):
        e = self.current
        ref = self.refs(e).get("manim") if e else None
        if ref is None:
            return
        flag = QUALITY[self.quality.currentText()]
        if not force:
            hit = cached_mp4(ref, flag)
            if hit:
                self._show_video(hit)
                self.statusBar().showMessage("from cache", 3000)
                return
        if self.render_thread and self.render_thread.isRunning():
            self.statusBar().showMessage("a render is already running", 3000)
            return
        self.log.clear()
        self.progress.setVisible(True)
        self.btn2d.setEnabled(False)
        self.btn_re.setEnabled(False)
        self.placeholder.setText(
            f"rendering {ref.name} at {self.quality.currentText()}…\n\n"
            "first time only — the result is cached")
        self.stack.setCurrentIndex(0)
        self.render_thread = RenderThread(ref, flag)
        self.render_thread.line.connect(self._log)
        self.render_thread.finished_ok.connect(self._rendered)
        self.render_thread.start()

    def _log(self, s: str):
        self.log.appendPlainText(s)

    def _rendered(self, scene: str, path: str):
        self.progress.setVisible(False)
        self.btn2d.setEnabled(True)
        self.btn_re.setEnabled(True)
        if path and self.auto_clean:
            frag = Path(path).parent / "partial_movie_files"
            if frag.is_dir() and CACHE in frag.parents:
                remove_path(frag)
        if not path:
            self.placeholder.setText(
                f"{scene} failed to render.\n\n"
                "Check the log below. Most often: manim is not installed "
                "in the interpreter running this app.")
            self.stack.setCurrentIndex(0)
            return
        self._show_video(path)

    def _show_video(self, path: str):
        if not HAVE_MEDIA:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self.placeholder.setText(f"opened in your system player:\n{path}")
            self.stack.setCurrentIndex(0)
            return
        self._ended = False
        self._set_transport_enabled(True)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.stack.setCurrentIndex(1)
        self.player.play()
        self.statusBar().showMessage(Path(path).name, 4000)

    # ------------------------------------------------------- language, menus
    def T(self, en: str, zh: str, sep: str = " · ") -> str:
        """A label in the chosen language: bilingual (default) or English."""
        return en if self.lang == "en" else f"{en}{sep}{zh}"

    def _build_menus(self):
        mb = self.menuBar()
        self.menu_lang = mb.addMenu("")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.act_both = QAction("中文 + English", self, checkable=True)
        self.act_en = QAction("English", self, checkable=True)
        for act in (self.act_both, self.act_en):
            group.addAction(act)
            self.menu_lang.addAction(act)
        (self.act_en if self.lang == "en" else self.act_both).setChecked(True)
        self.act_both.triggered.connect(lambda: self._set_language("both"))
        self.act_en.triggered.connect(lambda: self._set_language("en"))

        self.menu_cache = mb.addMenu("")
        self.act_clear = QAction("", self)
        self.act_clear.triggered.connect(self.clear_cache)
        self.act_frag = QAction("", self)
        self.act_frag.triggered.connect(self.clear_fragments)
        self.act_auto = QAction("", self, checkable=True)
        self.act_auto.setChecked(self.auto_clean)
        self.act_auto.toggled.connect(self._set_auto_clean)
        self.act_open = QAction("", self)
        self.act_open.triggered.connect(self.open_cache_folder)
        self.menu_cache.addAction(self.act_clear)
        self.menu_cache.addAction(self.act_frag)
        self.menu_cache.addSeparator()
        self.menu_cache.addAction(self.act_auto)
        self.menu_cache.addSeparator()
        self.menu_cache.addAction(self.act_open)

    def _set_language(self, lang: str):
        if lang == self.lang:
            return
        self.lang = lang
        self.settings.setValue("ui/language", lang)
        self.settings.sync()
        self._retranslate()

    def _set_auto_clean(self, on: bool):
        self.auto_clean = bool(on)
        self.settings.setValue("cache/delete_fragments", self.auto_clean)
        self.settings.sync()

    def _retranslate(self):
        """Every label that depends on the language. The list is relabelled
        in place: the selection stays and a playing video keeps playing."""
        T = self.T
        self.setWindowTitle(T("Proofs Without Words", "无字证明"))
        self.search.setPlaceholderText(
            "filter…  (name, formula, id)" if self.lang == "en"
            else "filter…  (name, 中文, formula, id)")
        self.view.setItemText(0, T("Ranked by popularity", "按热门排序"))
        self.view.setItemText(1, T("By category", "按类别"))
        self.formula_tabs.setTabText(0, T("Formula (LaTeX)", "公式"))
        self.btn_copy.setText("⧉ " + T("copy", "复制"))
        self.btn_clear.setText(T("clear cache", "清除缓存") + "…")
        self.menu_lang.setTitle(T("Language", "语言"))
        self.menu_cache.setTitle(T("Cache", "缓存"))
        self.act_clear.setText(T("Clear cache", "清除缓存") + "…")
        self.act_frag.setText(T("Clear temporary files", "清除临时文件") + "…")
        self.act_auto.setText(T("Delete temporary files after each render",
                                "渲染后删除临时文件"))
        self.act_open.setText(T("Open cache folder", "打开缓存文件夹"))
        self.tree.blockSignals(True)
        try:
            self._repopulate()
        finally:
            self.tree.blockSignals(False)
        if self.current is not None:
            self._show_entry_text(self.current)

    def _show_entry_text(self, e):
        cat_en, cat_zh = reg.CATEGORIES[e.category]
        _, tier_en, tier_zh = reg.TIERS[reg.tier_of(e)]
        self.title.setText(f"{e.id}  ·  {self.T(e.en, e.zh, '  ·  ')}")
        self.statement.setText(
            f"{self.T(cat_en, cat_zh, ' / ')}  ·  #{reg.RANK[e.id]} of "
            f"{len(reg.CATALOGUE)}, {self.T(tier_en, tier_zh, ' / ')}"
            f"      —      {e.statement}")
        self.formula.show_formula(e.formula)
        self.latex_src.setText(e.formula)
        self.ggb_text.setText(ggb.GEOGEBRA.get(e.id, "") or "—")
        self.ggb_hint.setText(GGB_TEXT_HINT if e.id in ggb.TEXT_ONLY
                              else GGB_HINT)
        self.btn_copy.setEnabled(True)
        self.note.setText(e.note)

    def copy_formula(self):
        """Put the shown formula on the clipboard: LaTeX or GeoGebra."""
        e = self.current
        if e is None:
            return
        if self.formula_tabs.currentIndex() == 0:
            what, text = "LaTeX", e.formula
            tail = "   (paste inside $ … $; uses amsmath and amssymb)"
        else:
            what, text = "GeoGebra", ggb.GEOGEBRA.get(e.id, "")
            tail = ""
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"copied {what}:  {text}{tail}", 8000)

    def open_cache_folder(self):
        CACHE.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(CACHE)))

    def clear_fragments(self):
        """Delete manim's partial_movie_files; the finished videos stay."""
        title = self.T("Clear temporary files", "清除临时文件")
        if self.render_thread and self.render_thread.isRunning():
            QMessageBox.information(
                self, title,
                "A video is rendering. Clear the temporary files when it has "
                "finished.")
            return
        frags = fragment_dirs()
        if not frags:
            QMessageBox.information(
                self, title, "There are no temporary files in the cache.")
            return
        size, _ = disk_usage(frags)
        answer = QMessageBox.question(
            self, title,
            f"manim joins short clips into each video and leaves them behind "
            f"in partial_movie_files folders: {len(frags)} of them in\n"
            f"{CACHE}\nhold {human_size(size)}.\n\n"
            "Delete them? The finished videos stay.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            return
        left = [p for p in frags if not remove_path(p)]
        freed = size - disk_usage(left)[0]
        msg = (f"temporary files: deleted {len(frags) - len(left)} folders, "
               f"{human_size(freed)} freed")
        if left:
            msg += f"; {len(left)} could not be deleted"
        self.log.appendPlainText(msg)
        self.statusBar().showMessage(msg, 8000)

    # ------------------------------------------------------------ the cache
    def clear_cache(self):
        """Delete rendered videos: all of them, or only the old versions."""
        title = self.T("Clear cache", "清除缓存")
        if self.render_thread and self.render_thread.isRunning():
            QMessageBox.information(
                self, title,
                "A video is rendering. Clear the cache when it has finished.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            current, stale = survey_cache(self.found)
        finally:
            QApplication.restoreOverrideCursor()
        everything = current + stale
        if not everything:
            QMessageBox.information(
                self, title, f"The cache is already empty:\n{CACHE}")
            return
        all_bytes, all_videos = disk_usage(everything)
        old_bytes, _ = disk_usage(stale)

        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Question)
        n_dirs = sum(p.is_dir() for p in everything)
        box.setText(
            f"Rendered 2D videos are kept in\n{CACHE}\n\n"
            f"{all_videos} video{'s' if all_videos != 1 else ''} in "
            f"{n_dirs} folder{'s' if n_dirs != 1 else ''}"
            f" (one per proof and quality), {human_size(all_bytes)}.")
        info = ("A deleted video renders again the next time it is played. "
                "3D proofs are never cached.")
        if stale:
            info = (f"{len(stale)} of the folders, {human_size(old_bytes)}, "
                    "are old versions: the proof has been changed since, or "
                    "they were left by an earlier version of pww. The "
                    "app no longer uses them.\n\n" + info)
        box.setInformativeText(info)
        b_all = box.addButton(f"Delete all  ({human_size(all_bytes)})",
                              QMessageBox.ButtonRole.DestructiveRole)
        b_old = None
        if stale:
            b_old = box.addButton(
                f"Delete old versions only  ({human_size(old_bytes)})",
                QMessageBox.ButtonRole.AcceptRole)
        b_cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(b_all)
        box.setEscapeButton(b_cancel)
        box.exec()
        hit = box.clickedButton()
        if hit is b_all:
            self._delete_cached(everything)
        elif b_old is not None and hit is b_old:
            self._delete_cached(stale)

    def _delete_cached(self, targets: list[Path], attempt: int = 0,
                       freed: int = 0, gone: int = 0):
        targets = [p for p in targets if p.parent == CACHE]   # cache/ only
        if attempt == 0:
            self._release_video_in(targets)
        sizes = {p: disk_usage([p])[0] for p in targets}
        left = [p for p in targets if not remove_path(p)]
        freed += sum(n for p, n in sizes.items() if p not in left)
        gone += len(targets) - len(left)
        if left and attempt == 0:
            # the media backend can let go of its file a moment late
            QTimer.singleShot(
                600, lambda: self._delete_cached(left, 1, freed, gone))
            return
        msg = (f"cache: deleted {gone} folder{'s' if gone != 1 else ''}, "
               f"{human_size(freed)} freed")
        if left:
            msg += f"; {len(left)} could not be deleted"
            QMessageBox.warning(
                self, self.T("Clear cache", "清除缓存"),
                f"{len(left)} folder{'s' if len(left) != 1 else ''} could not "
                "be deleted, probably because another program has a file in "
                "them open:\n\n" + "\n".join(str(p) for p in left[:8]))
        self.log.appendPlainText(msg)
        self.statusBar().showMessage(msg, 8000)

    def _release_video_in(self, targets: list[Path]):
        """Close the player's file if it is in one of the targets: Windows
        will not delete a file that is open."""
        if not HAVE_MEDIA or self.player.source().isEmpty():
            return
        src = Path(self.player.source().toLocalFile())
        if not any(t == src or t in src.parents for t in targets):
            return
        self._stop_video()
        self.player.setSource(QUrl())
        QApplication.processEvents()
        self.placeholder.setText(
            "the cache was cleared\n\n▶ Play 2D renders this proof again")
        self.stack.setCurrentIndex(0)

    def _stop_video(self):
        if not HAVE_MEDIA:
            return
        self.player.stop()
        self._ended = False
        self._scrubbing = False
        self.seek.setRange(0, 0)
        self.seek.setValue(0)
        self.timelabel.setText("0:00 / 0:00")
        self.btn_play.setText("▶")
        self._set_transport_enabled(False)

    # ------------------------------------------------------------ transport
    def _set_transport_enabled(self, on: bool):
        for w in (self.btn_back, self.btn_play, self.seek):
            w.setEnabled(on)

    @staticmethod
    def _clock(ms: int) -> str:
        ms = max(int(ms), 0)
        return f"{ms // 60000}:{(ms // 1000) % 60:02d}"

    def _update_clock(self):
        self.timelabel.setText(
            f"{self._clock(self.player.position())} / "
            f"{self._clock(self.player.duration())}")

    def toggle_play(self):
        """Play, pause, or — when parked on the closing frame — replay."""
        if not HAVE_MEDIA or self.player.source().isEmpty():
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return
        dur = self.player.duration()
        if self._ended or (dur and self.player.position() >= dur - END_GUARD_MS):
            self._ended = False
            self.player.setPosition(0)
        self.player.play()

    def to_start(self):
        if not HAVE_MEDIA:
            return
        self._ended = False
        self.player.setPosition(0)
        self.seek.setValue(0)
        self._update_clock()
        self._on_playstate(self.player.playbackState())

    def _seek_begin(self):
        self._scrubbing = True

    def _seek_end(self):
        self._scrubbing = False
        self._seek_to(self.seek.value())

    def _seek_to(self, ms: int):
        if not HAVE_MEDIA:
            return
        self.player.setPosition(int(ms))
        dur = self.player.duration()
        # scrubbing back off the final frame re-arms play rather than replay
        if self._ended and dur and ms < dur - END_GUARD_MS:
            self._ended = False
            self._on_playstate(self.player.playbackState())
        self._update_clock()

    def _on_duration(self, dur: int):
        self.seek.setRange(0, max(int(dur), 0))
        self.seek.setPageStep(max(int(dur) // 20, 1000))
        self._update_clock()

    def _on_position(self, pos: int):
        if not self._scrubbing:
            self.seek.setValue(int(pos))
        self._update_clock()
        dur = self.player.duration()
        playing = (self.player.playbackState()
                   == QMediaPlayer.PlaybackState.PlayingState)
        if playing and not self._ended and dur and pos >= dur - END_GUARD_MS:
            self._park_at_end()

    def _on_status(self, status):
        # backstop: if positionChanged never lands inside the guard window,
        # EndOfMedia still parks us on the closing frame rather than looping
        if status == QMediaPlayer.MediaStatus.EndOfMedia and not self._ended:
            self._park_at_end()

    def _park_at_end(self):
        """Hold the last frame — the finished figure and its formula."""
        self._ended = True
        dur = self.player.duration()
        self.player.pause()
        if dur:
            self.player.setPosition(max(dur - END_GUARD_MS, 0))
            self.seek.setValue(self.player.position())
        self._update_clock()
        self.btn_play.setText("↻")
        self.statusBar().showMessage(
            "finished — press ↻ to play it again", 6000)

    def _on_playstate(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("❚❚")
        else:
            self.btn_play.setText("↻" if self._ended else "▶")

    # --------------------------------------------------------------- 3D
    def play_3d(self):
        e = self.current
        if not e or "vpython" not in self.refs(e):
            return
        self._kill_vpython()
        args = [str(THREED_FILE), e.id]
        if not HAVE_WEB:
            args.append("--external")
        self.vp_proc = QProcess(self)
        self.vp_proc.setWorkingDirectory(str(HERE))
        self.vp_proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self.vp_proc.readyReadStandardOutput.connect(self._vp_output)
        self.vp_proc.start(sys.executable, args)
        where = ("in the panel below" if HAVE_WEB
                 else "in your default browser (install PyQt6-WebEngine to "
                      "embed it here instead)")
        self.placeholder.setText(
            f"starting the 3D canvas for {e.id} {where}…\n\n"
            "right-drag (or Ctrl+drag) to rotate · Shift+drag to pan · "
            "scroll to zoom\n\n"
            "The scene keeps its own process; opening another 3D proof "
            "closes this one.")
        self.stack.setCurrentIndex(0)
        self.log.clear()
        self._log(f"$ {sys.executable} {' '.join(args)}")

    def _vp_output(self):
        if not self.vp_proc:
            return
        data = bytes(self.vp_proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        for line in data.splitlines():
            self._log(line)
            m = re.match(r"PWW_URL\s+(\S+)", line)
            if m and HAVE_WEB and self.web is not None:
                url = m.group(1)
                # give vpython's server a moment to finish binding
                QTimer.singleShot(
                    350, lambda u=url: (self.web.load(QUrl(u)),
                                        self.stack.setCurrentIndex(2)))

    def _kill_vpython(self):
        if self.vp_proc is not None:
            try:
                self.vp_proc.kill()
                self.vp_proc.waitForFinished(1500)
            except Exception:
                pass
            self.vp_proc = None
        if HAVE_WEB and self.web is not None:
            self.web.setHtml("")

    # ------------------------------------------------------------- closing
    def closeEvent(self, ev):
        self._stop_video()
        self._kill_vpython()
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.stop()
            self.render_thread.wait(2000)
        ev.accept()


def dark(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(BG))
    p.setColor(QPalette.ColorRole.WindowText, QColor(FG))
    p.setColor(QPalette.ColorRole.Base, QColor(PANEL))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(BG))
    p.setColor(QPalette.ColorRole.Text, QColor(FG))
    p.setColor(QPalette.ColorRole.Button, QColor("#22222b"))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(FG))
    p.setColor(QPalette.ColorRole.Highlight, QColor("#3a5a8c"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(p)


def install_excepthook():
    """Log and show uncaught exceptions instead of dying.

    PyQt6's default when an exception escapes a slot is to print it and call
    qFatal(): the whole process aborts. From a console you at least see the
    traceback; under pythonw.exe the window just vanishes. This hook keeps
    the app running, appends the traceback to pww_error.log, and shows it.
    """
    state = {"dialog_open": False}

    def show(title: str, text: str):
        if state["dialog_open"]:            # one dialog at a time; the log
            return                          # still records every failure
        state["dialog_open"] = True
        try:
            QMessageBox.critical(None, title, text)
        finally:
            state["dialog_open"] = False

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{text}")
        except OSError:
            pass
        if sys.__stderr__ is not None:      # a real console, when there is one
            try:
                sys.__stderr__.write(text)
            except Exception:
                pass
        # dialogs only from the GUI thread, and deferred to the event loop so
        # one never opens inside the handler that just failed
        if (QApplication.instance() is not None
                and threading.current_thread() is threading.main_thread()):
            msg = (f"{etype.__name__}: {value}\n\n"
                   "The app is still running. The full traceback was "
                   f"appended to\n{ERROR_LOG}")
            QTimer.singleShot(0, lambda: show("Proofs Without Words — error",
                                              msg))

    sys.excepthook = hook


def main():
    install_excepthook()
    app = QApplication(sys.argv)
    dark(app)
    w = Main()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
