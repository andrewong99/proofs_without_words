# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_discover.py — find the animations by naming convention.

    2D (manim):   a class named  <ID>_<Name>  in any  pww_scenes*.py
                  e.g. class B3_SumOfSquares(Board) -> entry B3
    3D (vpython): a function named  scene_<ID>  in any  pww_3d*.py
                  e.g. def scene_I2(): ...          -> entry I2

Files are parsed with `ast`; nothing is imported, so neither manim nor
vpython is loaded and the app starts fast. A module that fails to parse is
reported and skipped, never fatal: one broken scene must not take the
catalogue down with it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

ID_PATTERN = r"[A-Z]{1,3}\d{1,4}"
SCENE_CLASS = re.compile(rf"^({ID_PATTERN})_\w+$")
SCENE_FUNC = re.compile(rf"^scene_({ID_PATTERN})$")

KIT_FILES = {"pww_kit.py", "pww_3d_kit.py"}


@dataclass(frozen=True)
class SceneRef:
    engine: str          # "manim" or "vpython"
    file: Path           # module that defines it
    name: str            # class name (manim) or entry id (vpython)


def manim_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("pww_scenes*.py")
                  if p.name not in KIT_FILES)


def vpython_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("pww_3d*.py")
                  if p.name not in KIT_FILES)


def _parse(path: Path, problems: list[str]):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, ValueError) as exc:
        problems.append(f"{path.name}: cannot parse — {exc}")
        return None


def discover(folder: Path):
    """Return ({entry_id: {"manim": SceneRef, "vpython": SceneRef}}, problems).

    Only top-level definitions count. If two definitions claim the same id
    for the same engine, the first file (alphabetically) wins and the clash
    is reported.
    """
    found: dict[str, dict[str, SceneRef]] = {}
    problems: list[str] = []

    for path in manim_files(folder):
        tree = _parse(path, problems)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                m = SCENE_CLASS.match(node.name)
                if m:
                    _add(found, problems, m.group(1),
                         SceneRef("manim", path, node.name))

    for path in vpython_files(folder):
        tree = _parse(path, problems)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                m = SCENE_FUNC.match(node.name)
                if m:
                    _add(found, problems, m.group(1),
                         SceneRef("vpython", path, m.group(1)))

    return found, problems


def _add(found, problems, eid, ref):
    slot = found.setdefault(eid, {})
    if ref.engine in slot:
        prev = slot[ref.engine]
        problems.append(
            f"{eid}: two {ref.engine} scenes — {prev.file.name}:{prev.name} "
            f"and {ref.file.name}:{ref.name}; using the first")
        return
    slot[ref.engine] = ref


def imports_module(tree: ast.Module, module: str) -> bool:
    """True if the module does `import X` or `from X import ...`."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            return True
        if isinstance(node, ast.Import) and any(a.name == module
                                                for a in node.names):
            return True
    return False


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    found, problems = discover(here)
    n2 = sum(1 for v in found.values() if "manim" in v)
    n3 = sum(1 for v in found.values() if "vpython" in v)
    print(f"{len(found)} entries with scenes: {n2} manim, {n3} vpython")
    for p in problems:
        print("  problem:", p)
