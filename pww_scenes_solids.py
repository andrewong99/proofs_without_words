# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_scenes_solids.py — 2D (manim) scene for the solids family: I10.

Euler's polyhedron formula by flattening and collapsing. A cube turns until
one face is square to the eye; that face is lifted off, and the eye moves in
along its axis: the perspective picture is the cube flattened into the
plane (a Schlegel diagram), the missing face becoming the outside region.
Then edges that separate two different regions are deleted (E and F both
drop by 1) until no cycle is left, and leaves are pruned (V and E both drop
by 1) down to a single vertex: 1 − 0 + 1 = 2. Every move is checked on the
graph itself, so V − E + F is recomputed at each step, never assumed.
"""

from pww_kit import *


class I10_EulerFormula(Board):
    """V − E + F = 2: a cube, opened and flattened, then collapsed."""

    def construct(self):
        # ---------------------------------------------------------- the graph
        # vertices: A0..A3 = the face we take out (front), B0..B3 = back face
        names = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3"]
        cube = {}
        for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            cube[f"A{i}"] = np.array([sx, sy, 1.0])
            cube[f"B{i}"] = np.array([sx, sy, -1.0])
        edges = {}
        for i in range(4):
            j = (i + 1) % 4
            edges[f"a{i}{j}"] = (f"A{i}", f"A{j}")
            edges[f"b{i}{j}"] = (f"B{i}", f"B{j}")
            edges[f"s{i}"] = (f"A{i}", f"B{i}")
        faces = {"front": ["A0", "A1", "A2", "A3"],
                 "back": ["B0", "B1", "B2", "B3"]}
        for i in range(4):
            j = (i + 1) % 4
            faces[f"side{i}"] = [f"A{i}", f"A{j}", f"B{j}", f"B{i}"]
        sides = {}                       # the two faces on either side of an edge
        for e, (p, q) in edges.items():
            sides[e] = [f for f, vs in faces.items()
                        if p in vs and q in vs and
                        abs(vs.index(p) - vs.index(q)) in (1, 3)]
            check(len(sides[e]) == 2, f"edge {e} lies on two faces")
        check(len(cube) - len(edges) + len(faces) == 2, "cube: 8 − 12 + 6 = 2")

        # ------------------------------------------------------------ camera
        # the cube turns until the front face is square to the eye; the eye
        # sits on that face's axis at z = eye, and the picture is scaled so
        # the front face has half-size `size` on screen
        c = np.array([-2.85, 0.32, 0.0])
        yaw0, pitch0 = 30 * DEGREES, 22 * DEGREES
        turn = ValueTracker(0.0)          # 0: seen obliquely, 1: square on
        eye = ValueTracker(7.0)
        size = ValueTracker(1.45)
        lift = ValueTracker(0.0)          # the front face, taken off
        appear = ValueTracker(0.0)

        def rot(v, s):
            yaw, pitch = yaw0 * (1 - s), pitch0 * (1 - s)
            x, y, z = v
            x1, z1 = x * np.cos(yaw) + z * np.sin(yaw), -x * np.sin(yaw) + z * np.cos(yaw)
            return np.array([x1, y * np.cos(pitch) - z1 * np.sin(pitch),
                             y * np.sin(pitch) + z1 * np.cos(pitch)])

        def project(n, s, ze, sz, dz=0.0):
            p = rot(cube[n], s) + np.array([0.0, 0.0, dz])
            k = sz * (ze - 1.0)            # a corner of the front face lands at ±sz
            return c + np.array([k * p[0] / (ze - p[2]), k * p[1] / (ze - p[2]), 0.0])

        def P(n, dz=0.0):
            return project(n, turn.get_value(), eye.get_value(), size.get_value(), dz)

        out_h, in_h = 2.45, 0.92           # the flat picture: two squares
        ratio = in_h / out_h
        eye_end = 1.0 + 2.0 * ratio / (1.0 - ratio)    # back face at in_h
        PF = {n: project(n, 1.0, eye_end, out_h) for n in names}
        for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            check(close(PF[f"A{i}"], c + np.array([sx * out_h, sy * out_h, 0])) and
                  close(PF[f"B{i}"], c + np.array([sx * in_h, sy * in_h, 0])),
                  "the eye's last position gives the flat picture")
        # moving the eye in is a flattening, never a fold: the five faces
        # keep one orientation at every eye position on the way
        for t in np.linspace(0, 1, 41):
            ze = 7.0 + (eye_end - 7.0) * t
            sz = 1.45 + (out_h - 1.45) * t
            signs = [area([project(n, 1.0, ze, sz) for n in faces[f]])
                     for f in faces if f != "front"]
            check(all(a_ > 1e-6 for a_ in signs), "no face folds over")
        check(abs(sum(abs(area([PF[v] for v in faces[f]]))
                      for f in faces if f != "front") - (2 * out_h) ** 2) < 1e-9,
              "the flattened faces tile the outer square")

        col = {"front": PURPLE_B, "back": BLUE_D, "side0": TEAL_D,
               "side1": ORANGE, "side2": YELLOW_E, "side3": GREEN_D}

        def muted(c_, k=0.45):
            return interpolate_color(c_, BLACK, k)

        # ---- the cube as live mobjects, driven by the camera trackers
        def live_face(f):
            def build():
                a = appear.get_value()
                if f == "front":
                    lf = lift.get_value()
                    op = 0.32 * a * max(0.0, 1 - lf / 1.3)
                    return Polygon(*[P(n, lf) for n in faces[f]], fill_color=col[f],
                                   fill_opacity=op, stroke_color=YELLOW_B,
                                   stroke_width=4, stroke_opacity=min(1.0, lf * 3) * op / 0.32)
                return Polygon(*[P(n) for n in faces[f]], fill_color=col[f],
                               fill_opacity=0.32 * a, stroke_width=0)
            return always_redraw(build)

        def live_edge(e):
            p, q = edges[e]
            return always_redraw(lambda: Line(P(p), P(q), color=WHITE, stroke_width=3.5,
                                              stroke_opacity=appear.get_value()))

        def live_dot(n):
            return always_redraw(lambda: Dot(P(n), radius=0.075, color=WHITE,
                                             fill_opacity=appear.get_value()))

        order = ["back", "side3", "side0", "side1", "side2", "front"]
        live = [live_face(f) for f in order] + [live_edge(e) for e in edges] + \
               [live_dot(n) for n in names]
        self.add(*live)

        # ---------------------------------------------------------- counters
        def counter_group(V, E, F):
            rows = VGroup(
                tag(f"V = {V}", 34, WHITE),
                tag(f"E = {E}", 34, YELLOW_B),
                tag(f"F = {F}", 34, TEAL_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
            inv = VGroup(tag("V − E + F", 30, YELLOW_B),
                         tag(f"= {V} − {E} + {F} = {V - E + F}", 30, YELLOW_B)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            g = VGroup(rows, inv).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
            g.move_to(np.array([0.0, 0.55, 0.0])).align_to(np.array([2.0, 0, 0]), LEFT)
            check(g.get_right()[0] < SAFE_X, "counters inside the frame")
            return g

        # ------------------------------------------------- 1. the solid cube
        self.play(appear.animate.set_value(1.0), run_time=1.4)
        V, E, F = 8, 12, 6
        cnt = counter_group(V, E, F)
        self.play(FadeIn(cnt), run_time=0.8)
        self.hold(0.6)

        # ---------- 2. turn it square on, lift one face off, look straight in
        self.play(turn.animate.set_value(1.0), run_time=1.6)
        self.play(lift.animate.set_value(1.3), run_time=1.2)
        self.play(eye.animate.set_value(eye_end), size.animate.set_value(out_h),
                  run_time=2.2)

        # hand over to still mobjects in exactly the same places
        self.remove(*live)
        polys = {f: Polygon(*[PF[v] for v in faces[f]], fill_color=col[f],
                            fill_opacity=0.32, stroke_width=0)
                 for f in order if f != "front"}
        lines = {e: Line(PF[p], PF[q], color=WHITE, stroke_width=3.5)
                 for e, (p, q) in edges.items()}
        dots = {n: Dot(PF[n], radius=0.075, color=WHITE) for n in names}
        self.add(*polys.values(), *lines.values(), *dots.values())

        # the missing face is now everything outside; opaque fills with a
        # stroke of their own colour, so merged regions show no seams
        out_col = muted(col["front"], 0.7)
        outer_sq = Polygon(*[PF[v] for v in faces["front"]])
        canvas = Rectangle(width=7.6, height=6.7).move_to(np.array([-2.8, 0.37, 0]))
        outside = Difference(canvas, outer_sq, fill_color=out_col, fill_opacity=0.0,
                             stroke_color=out_col, stroke_width=2, stroke_opacity=0.0)
        self.bring_to_back(outside)
        self.play(outside.animate.set_fill(opacity=1.0).set_stroke(opacity=1.0),
                  *[polys[f].animate.set_fill(muted(col[f]), opacity=1.0)
                    .set_stroke(muted(col[f]), width=2, opacity=1.0) for f in polys],
                  run_time=1.0)
        self.hold(0.6)

        # --------------------- 3. delete edges that separate two regions
        parent = {f: f for f in faces}

        def find(f):
            while parent[f] != f:
                f = parent[f]
            return f

        region_col = {f: muted(col[f]) for f in faces}
        region_col["front"] = out_col
        members = {f: [f] for f in faces}
        alive_e = set(edges)
        alive_v = set(names)

        def connected():
            seen, stack = set(), [next(iter(alive_v))]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                for e in alive_e:
                    p, q = edges[e]
                    if u in (p, q):
                        stack.append(q if u == p else p)
            return seen == alive_v

        def regions():
            return len({find(f) for f in faces})

        # the outside floods the four side faces, then the back face
        for e in ["a01", "a12", "a23", "a30", "b01"]:
            r1, r2 = find(sides[e][0]), find(sides[e][1])
            check(r1 != r2, f"edge {e} separates two different regions")
            keep, gone = (r1, r2) if "front" in members[r1] else (r2, r1)
            line = lines[e]
            self.play(line.animate.set_color(RED).set_stroke(width=8), run_time=0.35)
            target = region_col[keep]
            anims = [polys[f].animate.set_fill(target, opacity=1.0).set_stroke(target)
                     for f in members[gone] if f != "front"]
            alive_e.discard(e)
            parent[gone] = keep
            members[keep] += members[gone]
            E, F = E - 1, F - 1
            check(connected(), "the graph stays connected")
            check(F == regions(), "F counts the regions")
            check(len(alive_v) - len(alive_e) + regions() == 2, "V − E + F = 2")
            new_cnt = counter_group(V, E, F)
            self.play(FadeOut(line), *anims, FadeOut(cnt), FadeIn(new_cnt),
                      run_time=0.75)
            cnt = new_cnt

        check(len(alive_e) == len(alive_v) - 1 and regions() == 1,
              "what is left is a tree, in one region")
        self.hold(0.5)

        # ------------------------------- 4. prune leaves down to one vertex
        for v in ["A0", "A1", "A2", "A3", "B0", "B1", "B3"]:
            inc = [e for e in alive_e if v in edges[e]]
            check(len(inc) == 1, f"{v} is a leaf")
            e = inc[0]
            self.play(lines[e].animate.set_color(RED).set_stroke(width=8),
                      dots[v].animate.set_color(RED).scale(1.4), run_time=0.3)
            alive_e.discard(e)
            alive_v.discard(v)
            V, E = V - 1, E - 1
            check(connected(), "the graph stays connected")
            check(len(alive_v) - len(alive_e) + regions() == 2, "V − E + F = 2")
            new_cnt = counter_group(V, E, F)
            self.play(FadeOut(lines[e]), FadeOut(dots[v]), FadeOut(cnt),
                      FadeIn(new_cnt), run_time=0.6)
            cnt = new_cnt

        check((V, E, F) == (1, 0, 1) and alive_v == {"B2"}, "one vertex is left")
        self.play(dots["B2"].animate.scale(1.6).set_color(YELLOW_B), run_time=0.5)
        self.play(Write(caption("V − E + F  =  2")))
        self.hold(2.4)
