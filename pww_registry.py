# SPDX-License-Identifier: GPL-3.0-or-later
"""
pww_registry.py — the catalogue of "proofs without words" (无字证明).

This module is pure data: no manim, no vpython, no Qt imports. It is safe to
import from anywhere (the GUI, a render subprocess, a test).

Every entry in the catalogue is listed here. Whether an entry is animated
is decided by pww_discover, by name: a manim class <ID>_<Name> in a
pww_scenes*.py file, or a vpython function scene_<ID> in a pww_3d*.py file.
The optional manim=/vpy= bindings below are only cross-checked by the GUI.

Formula strings are matplotlib *mathtext* (a LaTeX subset). They are rendered
by the GUI's formula panel without needing a TeX installation.
"""

from __future__ import annotations

from dataclasses import dataclass

MANIM = "manim"
VPYTHON = "vpython"

@dataclass(frozen=True)
class ProofEntry:
    id: str
    en: str
    zh: str
    category: str
    statement: str
    formula: str                      # mathtext, no surrounding $
    engines: tuple = ()               # (MANIM,) / (VPYTHON,) / both
    manim_scene: str | None = None    # optional explicit binding (checked)
    vpython_id: str | None = None     # optional explicit binding (checked)
    note: str = ""
    pop: int = 0                      # popularity, 0-100 (see TIERS)

    @property
    def status(self) -> str:
        """Returns "bound" when a scene is named here (discovery decides)."""
        return "bound" if (self.manim_scene or self.vpython_id) else "stub"

    @property
    def label(self) -> str:
        return f"{self.id}  {self.en}" + (f"  ·  {self.zh}" if self.zh else "")


CATEGORIES = {
    "A": ("Pythagoras family", "勾股定理族"),
    "B": ("Sums, series & figurate numbers", "级数与形数"),
    "C": ("Algebraic identities", "代数恒等式"),
    "D": ("Means & inequalities", "均值与不等式"),
    "E": ("Circle & conics", "圆与圆锥曲线"),
    "F": ("Triangle & polygon", "三角形与多边形"),
    "G": ("Trigonometry", "三角学"),
    "H": ("Calculus & limits", "微积分与极限"),
    "I": ("Solids in three dimensions", "立体几何"),
    "J": ("Number theory & curios", "数论与趣题"),
    "K": ("Combinatorics & probability", "组合与概率"),
    "L": ("Vectors & linear algebra", "向量与线性代数"),
}

# Popularity: how widely the result and its picture are known and taught.
# An editorial score, 0-100, grouped into three tiers; the ranked view sorts
# by it (ties by category and number).
TIERS = [
    (80, "Classics", "经典"),          # school-level, famous pictures
    (50, "Standard", "常见"),          # widely taught or well known
    (0, "Specialist", "专题"),         # enthusiasts, olympiad, curios
]


def _p(id, en, zh, statement, formula, manim=None, vpy=None, note="", pop=0):
    engines = []
    if manim:
        engines.append(MANIM)
    if vpy:
        engines.append(VPYTHON)
    return ProofEntry(
        id=id, en=en, zh=zh, category=id[0], statement=statement,
        formula=formula, engines=tuple(engines),
        manim_scene=manim, vpython_id=vpy, note=note, pop=pop,
    )


CATALOGUE: list[ProofEntry] = [
    # ---------------------------------------------------------------- A
    _p("A1", "Two arrangements of the (a+b)-square", "(a+b)²的两种拼法",
       "One square of side a+b, dissected two ways.",
       r"(a+b)^2 = c^2 + 4\cdot\frac{1}{2}ab \;\Rightarrow\; a^2+b^2=c^2",
       manim="A1_TwoArrangements",
       note="The same four triangles leave c² in one arrangement and a² + b² "
            "in the other. Not Zhao Shuang's figure — that is A2.", pop=97),
    _p("A2", "Zhao Shuang's hypotenuse diagram", "赵爽弦图",
       "Four right triangles around a central square make the square on the "
       "hypotenuse.",
       r"c^2 = (b-a)^2 + 4\cdot\frac{1}{2}ab = a^2 + b^2",
       manim="A2_ZhaoShuang",
       note="赵爽, commentary on the 周髀算经 (3rd century): "
            "「按弦图，又可以勾股相乘为朱实二，倍之为朱实四，以勾股之差自相乘"
            "为中黄实，加差实，亦成弦实。」 The four vermilion triangles (朱实) "
            "and the yellow centre (中黄实) make the hypotenuse square (弦实) — "
            "the animation uses his colours. Emblem of the 2002 ICM in "
            "Beijing, now of the Chinese Mathematical Society. Bhāskara II "
            "gave the same figure in the 12th century.", pop=95),
    _p("A3", "Euclid I.47 windmill (shear proof)", "欧几里得风车图",
       "Each leg-square becomes the matching rectangle of the hypotenuse "
       "square: shear, quarter-turn, shear.",
       r"a^2 = c\cdot p,\quad b^2 = c\cdot q,\quad p+q=c",
       manim="A3_EuclidWindmill",
       note="p and q are the parts of the hypotenuse cut off by the altitude "
            "from the right angle. Euclid halves each shape into a triangle "
            "(I.41) and matches the triangles by SAS (I.4); the animation "
            "keeps whole parallelograms instead. Each shear keeps its base "
            "and its parallels, so area is unchanged (I.35), and the "
            "quarter-turn is the SAS congruence made visible.", pop=90),
    _p("A4", "Perigal's dissection", "佩里加尔五片剖分",
       "The larger leg-square cut into four congruent quadrilaterals by two "
       "central lines parallel and perpendicular to the hypotenuse.",
       r"a^2 + b^2 = c^2", pop=80),
    _p("A5", "Garfield's trapezoid", "加菲尔德梯形",
       "A trapezoid of parallel sides a and b, counted two ways.",
       r"\frac{1}{2}(a+b)^2 = 2\cdot\frac{1}{2}ab + \frac{1}{2}c^2",
       manim="A5_Garfield",
       note="Published by James A. Garfield in 1876, five years before he "
            "became US president.", pop=85),
    _p("A6", "Altitude and similar triangles", "射影定理",
       "The altitude to the hypotenuse cuts the triangle into two copies of itself.",
       r"a^2 = c\,p,\quad b^2 = c\,q,\quad a^2+b^2 = c(p+q) = c^2",
       manim="A6_SimilarTriangles", pop=78),
    _p("A7", "Law of cosines", "余弦定理",
       "The altitudes cut each side square into two rectangles, equal in "
       "pairs at each vertex; c² is a² + b² less the two rectangles at C, "
       "each ab·cos C.",
       r"c^2 = a^2 + b^2 - 2ab\cos C",
       note="Shown for an acute triangle (Euclid II.13). When C is obtuse, "
            "cos C < 0 and the correction is added instead (II.12).", pop=82),
    _p("A8", "Inradius of a right triangle", "直角三角形内切圆半径",
       "Tangent lengths from each vertex, counted around the perimeter.",
       r"r = \frac{a+b-c}{2}", pop=45),
    _p("A9", "Thābit ibn Qurra's two cuts", "塔比·伊本·库拉的两刀剖分",
       "The squares a² and b² side by side on one base: two cuts from the "
       "point at distance b from the outer end of a² run to the far top "
       "corners and cut off two copies of the triangle, which turn a "
       "quarter-turn about those corners and close the square on c.",
       r"a^2 + b^2 = c^2",
       note="Thābit ibn Qurra (9th century).",
       pop=58),
    _p("A10", "Liu Hui's out-in dissection", "刘徽青朱出入图",
       "The parts of the leg squares that lie outside the hypotenuse square "
       "are moved, unchanged, into the gaps inside it (出入相补).",
       r"a^2 + b^2 = c^2",
       note="刘徽, commentary on the 九章算术 (263): the out-in complementary "
            "principle. His diagram is lost; the animation follows one "
            "reconstruction: 朱方 laid over the triangle, 青方 outside it, and "
            "the three pieces sticking out of the 弦方 (one 朱, two 青) slid "
            "without turning into its three gaps.",
       pop=62),
    _p("A11", "Leonardo's hexagons", "达·芬奇的六边形证明",
       "The leg squares with two copies of the triangle make a hexagon; the "
       "hypotenuse square with two copies makes another. The half of the first"
       " that holds the triangle, turned a quarter-turn about a vertex of the "
       "triangle, is exactly half of the second.",
       r"a^2 + b^2 + 2\cdot\frac{1}{2}ab = c^2 + 2\cdot\frac{1}{2}ab",
       note="Traditionally attributed to Leonardo da Vinci.",
       pop=60),
    _p("A12", "Pythagoras by intersecting chords", "相交弦证勾股定理",
       "The circle of radius c about the end of the leg a: the leg b is half "
       "a chord, and the diameter through the right angle is cut into c − a "
       "and c + a.",
       r"b^2 = (c-a)(c+a) = c^2 - a^2", pop=52),
    _p("A13", "Pappus's area theorem", "帕普斯面积定理",
       "Parallelograms on two sides of any triangle; their outer sides meet at"
       " H. Shear each, keeping its side on the triangle, until its other "
       "sides are parallel to HC; slide both along HC by its length; shear "
       "again along HC: together they fill the parallelogram on the third side"
       " whose sides are equal and parallel to HC.",
       r"[P] + [Q] = [R]",
       note="With squares on the legs of a right triangle, HC equals the "
            "hypotenuse and is perpendicular to it, so the third "
            "parallelogram is the square on the hypotenuse (Pythagoras).",
       pop=47),
    _p("A14", "Similar figures on the sides", "三边上的相似图形",
       "Semicircles, equilateral triangles or any three similar figures on "
       "the sides: area scales as the square of the side, so the figure on "
       "the hypotenuse is the sum of the other two.",
       r"k\,a^2 + k\,b^2 = k\,c^2",
       note="Euclid VI.31.", pop=55),
    _p("A15", "The lunes of Alhazen", "海什木月牙定理",
       "Semicircles on the legs, and the semicircle on the hypotenuse through "
       "the right angle: the two lunes together have exactly the triangle's "
       "area.",
       r"L_1 + L_2 = \frac{1}{2}ab",
       note="Ibn al-Haytham (c. 965–1040) proved this for any right triangle;"
            " Hippocrates of Chios (5th century BCE) had squared the single "
            "lune on an isosceles right triangle, the first curved region "
            "shown equal to a straight-sided one.",
       pop=66),
    _p("A16", "Reciprocal Pythagorean theorem", "倒数勾股定理",
       "Scale the triangle by 1/(ab): its legs become 1/b and 1/a and its "
       "hypotenuse c/(ab), which is 1/h for the altitude h on c, because ab = "
       "ch (both are twice the area).",
       r"\frac{1}{a^2} + \frac{1}{b^2} = \frac{1}{h^2}",
       pop=40),
    _p("A17", "Einstein's proof by similar areas", "爱因斯坦的相似面积证明",
       "The altitude splits the triangle into two copies similar to it, on "
       "hypotenuses a and b. Their areas add up to the whole, and similar "
       "areas scale as the squares of the hypotenuses.",
       r"k\,a^2 + k\,b^2 = k\,c^2",
       note="Often attributed to the young Einstein, who recalled proving "
            "the theorem from the similarity of triangles.", pop=57),
    _p("A18", "The Pythagorean tiling", "勾股镶嵌证明",
       "Tile the plane with the two leg squares; a tilted grid of hypotenuse "
       "squares lies over it, and each c-square is cut into pieces that make "
       "one a-square and one b-square.",
       r"a^2 + b^2 = c^2",
       note="Any placement of the grid works; centred on the big squares it "
            "gives Perigal's dissection (A4).",
       pop=59),
    _p("A19", "Law of cosines by intersecting chords", "相交弦证余弦定理",
       "The circle of radius a about C: the power of A, read along the side "
       "b and along the side c, gives b² − a² = c(c − 2a·cos B).",
       r"b^2 = a^2 + c^2 - 2ac\cos B",
       note="Drawn with b > a and an acute angle B.", pop=45),
    _p("A20", "Law of cosines, obtuse case", "余弦定理（钝角情形）",
       "When C is obtuse the altitude falls outside: the square on c is "
       "a² + b² plus two rectangles of a by the projection of b.",
       r"c^2 = a^2 + b^2 - 2ab\cos C,\quad C > 90^\circ",
       note="Euclid II.12.", pop=44),
    _p("A21", "Parallelogram law", "平行四边形恒等式",
       "Pythagoras on the two heights: the squares of the diagonals, (a + x)² "
       "+ h² and (a − x)² + h², add to twice the squares of the sides.",
       r"d_1^2 + d_2^2 = 2\left(a^2 + b^2\right)",
       note="Drawn with the foot of the height inside the base (0 < x < a).",
       pop=50),
    _p("A22", "Pythagoras from the incircle", "内切圆证勾股定理",
       "Equal tangents from the incircle give a + b − c = 2r and a + b + c = "
       "2(r + x + y). The square at the right angle and the kites at the other"
       " two corners, each halved along its axis, rearrange (two folds and a "
       "quarter-turn) into an r × (r + x + y) strip: ab/2 = (a + b − c)/2 · (a"
       " + b + c)/2, i.e. (a + b)² − c² = 2ab.",
       r"\frac{ab}{2} = \frac{a+b-c}{2}\cdot\frac{a+b+c}{2}",
       pop=38),
    _p("A23", "Bhaskara's chair", "婆什迦罗的椅子",
       "The four triangles and the central square of Zhao Shuang's figure: "
       "two triangles slide across, and the c-square becomes a chair-shaped "
       "a² and b² side by side.",
       r"c^2 = (b-a)^2 + 2ab = a^2 + b^2",
       note="Bhāskara II (12th century), whose figure carries the single "
            "word 'Behold!'.", pop=64),
    _p("A24", "Two half-squares", "两个等腰直角三角形证勾股",
       "Half-squares on the legs, both with their right angle at the "
       "triangle's right angle, make a quadrilateral (concave unless a = b) "
       "whose diagonals are the hypotenuse and its quarter-turned copy: equal "
       "and perpendicular. Cut along the hypotenuse and sheared twice, it "
       "becomes half the square on c.",
       r"\frac{a^2}{2} + \frac{b^2}{2} = \frac{c\cdot c}{2}",
       note="Drawn with a < b; for a = b the quadrilateral is a triangle.",
       pop=36),
    _p("A25", "Squaring a rectangle", "化矩形为正方形",
       "The altitude h on the diameter of a semicircle is the geometric mean "
       "of the parts p and q. Turned a quarter-turn about the foot of the "
       "altitude, the right triangle on p fits inside the right triangle on q "
       "with its hypotenuse parallel to the other chord (the angle in the "
       "semicircle is right); one shear along that parallel shows that half "
       "the h-square equals half the p × q rectangle.",
       r"h^2 = pq",
       note="Euclid II.14.",
       pop=50),
    _p("A26", "The 120° law of cosines", "120°三角形的余弦定理",
       "Three copies of the triangle with a 120° angle, turned by thirds of a "
       "turn, line the equilateral triangle on c around an equilateral "
       "triangle of side b − a (Zhao Shuang's figure on a triangular grid).",
       r"c^2 = 3ab + (b-a)^2 = a^2 + ab + b^2",
       note="The hexagonal-lattice analogue of Pythagoras (Eisenstein triples"
            " such as 3, 5, 7). Drawn for 3, 5, 7 on the triangular grid; "
            "areas are counted in unit equilateral triangles.",
       pop=25),
    _p("A27", "Pythagoras from Ptolemy", "托勒密定理导出勾股定理",
       "A rectangle in its circumcircle: its diagonals are diameters, and "
       "Ptolemy's theorem for it reads c·c = a·a + b·b.",
       r"c\cdot c = a\cdot a + b\cdot b",
       note="Ptolemy's theorem is cited, not re-proved; its proof is animated"
            " in E7.",
       pop=35),

    # ---------------------------------------------------------------- B
    _p("B1", "Triangular numbers", "三角形数",
       "Two staircases interlock into a rectangle.",
       r"1+2+\cdots+n = \frac{n(n+1)}{2}",
       manim="B1_Triangular", pop=96),
    _p("B2", "Odd numbers as gnomons", "奇数和为平方数",
       "Each odd number is an L-shaped shell of a growing square.",
       r"1+3+5+\cdots+(2n-1) = n^2",
       manim="B2_OddSquares", pop=98),
    _p("B3", "Sum of squares", "平方和",
       "Six stepped square pyramids interlock into an n × (n+1) × (2n+1) "
       "box.",
       r"\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6}", pop=84),
    _p("B4", "Nicomachus: sum of cubes", "尼科马库斯定理",
       "Each cube is a gnomon of the square on the triangular number.",
       r"\sum_{k=1}^{n} k^3 = \left(\frac{n(n+1)}{2}\right)^2",
       manim="B4_Nicomachus", pop=88),
    _p("B5", "Halving a square", "二分正方形",
       "Halve, halve again, forever: the square is exhausted.",
       r"\frac{1}{2}+\frac{1}{4}+\frac{1}{8}+\cdots = 1",
       manim="B5_GeometricHalves", pop=94),
    _p("B6", "The three L's", "三分之一",
       "At every stage three congruent L-shapes share the square equally.",
       r"\frac{1}{4}+\frac{1}{16}+\frac{1}{64}+\cdots = \frac{1}{3}",
       manim="B6_ThreeLs", pop=76),
    _p("B7", "Geometric series by similar triangles", "等比级数",
       "A staircase of nested similar triangles.",
       r"\sum_{k=0}^{\infty} r^k = \frac{1}{1-r},\quad |r|<1",
       note="The picture needs 0 < r < 1. For −1 < r < 0 split the terms by "
            "parity: Σ r^k = (1 + r)·Σ (r²)^k, and r² lies in (0, 1).", pop=70),
    _p("B8", "Sum of triangular numbers", "三角形数之和",
       "Six piles of stacked triangles (three of them mirror images) fill "
       "an n × (n+1) × (n+2) box.",
       r"\sum_{k=1}^{n} T_k = \frac{n(n+1)(n+2)}{6}", pop=62),
    _p("B9", "Fibonacci squares spiral", "斐波那契平方和",
       "Squares of Fibonacci side spiral into a rectangle.",
       r"\sum_{k=1}^{n} F_k^2 = F_n F_{n+1}",
       manim="B9_FibonacciSquares", pop=80),
    _p("B10", "Fibonacci partial sums", "斐波那契前n项和",
       "A telescoping staircase of Fibonacci bars.",
       r"\sum_{k=1}^{n} F_k = F_{n+2} - 1", pop=55),
    _p("B11", "Hockey-stick identity", "曲棍球棒恒等式",
       "A diagonal run of Pascal's triangle, starting at the edge, sums to "
       "the entry below its last term on the next diagonal.",
       r"\sum_{k=r}^{n}\binom{k}{r} = \binom{n+1}{r+1}", pop=60),
    _p("B12", "Pascal row sum", "帕斯卡行和",
       "Every subset of an n-set, counted by size.",
       r"\sum_{k=0}^{n}\binom{n}{k} = 2^n", pop=72),
    _p("B13", "Gauss's pairing", "高斯配对求和",
       "Stand 1, 2, …, n as columns of cells and drop the same row, turned "
       "over, into the gaps above: the top reads n, …, 1, and each of the n "
       "columns holds n + 1.",
       r"2\,(1+2+\cdots+n) = n(n+1)",
       pop=90),
    _p("B14", "Sum of even numbers", "偶数之和",
       "Each even number 2k is a column of height k, two wide. Side by side "
       "they make a staircase; cut it down the middle and turn the taller half"
       " over onto the shorter: the halves close into an n × (n+1) rectangle.",
       r"2+4+\cdots+2n = n(n+1)",
       note="Drawn for an even n; for an odd n the cut runs down the middle "
            "column.",
       pop=70),
    _p("B15", "Sum of squares in the plane", "平方和（平面拼法）",
       "Three copies of the squares 1², 2², …, n²: two stand in stacks at the "
       "ends of a rectangle (2n+1) wide and 1 + 2 + … + n high; the third, "
       "each square cut into its L-shaped strips of 1, 3, 5, … cells and "
       "straightened, fills the gap between the stacks.",
       r"3\sum_{k=1}^{n}k^2 = (2n+1)\cdot\frac{n(n+1)}{2}",
       pop=72),
    _p("B16", "Two triangular numbers make a square", "两个三角形数拼成平方数",
       "Cut the n × n square of dots along the staircase next to its "
       "diagonal: the two pieces are Tₙ and Tₙ₋₁.",
       r"T_{n-1} + T_n = n^2", pop=74),
    _p("B17", "Eight triangles around a dot", "8倍三角形数加1是平方数",
       "Eight copies of the staircase Tₙ, turned in pairs, surround a single "
       "unit square and fill the (2n+1)-square.",
       r"8\,T_n + 1 = (2n+1)^2",
       note="Known to Diophantus.", pop=63),
    _p("B18", "Up and down the staircase", "上下楼梯之和",
       "1 + 2 + … + n + … + 2 + 1 as a diamond of dots: turned 45°, it is "
       "the n × n square.",
       r"1+2+\cdots+(n-1)+n+(n-1)+\cdots+1 = n^2", pop=68),
    _p("B19", "Alternating sum of squares", "平方数的交错和",
       "Fill the n-square, empty the (n−1)-square in its corner, fill the "
       "(n−2)-square, and so on: every other L of width one stays, k² − (k−1)²"
       " with arms k and k − 1. A quarter turn of each L's short arm lays it "
       "under the long arm, and the rows n, n−1, …, 1 make the staircase Tₙ.",
       r"n^2-(n-1)^2+(n-2)^2-\cdots\pm1^2 = \frac{n(n+1)}{2}",
       note="Drawn for n = 6; for an odd n the last L is the single cell 1².",
       pop=45),
    _p("B20", "Powers of two", "2的幂之和",
       "Blocks of 1, 2, 4, … cells: each new block equals all the earlier ones"
       " plus the one missing cell, so the blocks always fill the next power "
       "of two but for that one cell.",
       r"1+2+4+\cdots+2^{n-1} = 2^n - 1",
       pop=69),
    _p("B21", "Thirds in a triangle", "三角形中的三分之一",
       "Cut an equilateral triangle into four by its midlines; colour the "
       "three corner quarters one colour each and repeat inside the middle "
       "quarter. A third of a turn about the centre carries each colour onto "
       "the next, so the three colours share alike: each gets ⅓.",
       r"\frac{1}{4}+\frac{1}{16}+\frac{1}{64}+\cdots = \frac{1}{3}",
       pop=52),
    _p("B22", "Halves from thirds", "三分之一的幂之和",
       "Cut a 3 × 1 strip into thirds: shade one end, leave the other, and "
       "repeat inside the middle square, cut into thirds the other way, and so"
       " on. A half-turn about the centre carries the shaded pieces onto the "
       "unshaded ones, so each is half.",
       r"\frac{1}{3}+\frac{1}{9}+\frac{1}{27}+\cdots = \frac{1}{2}",
       pop=50),
    _p("B23", "Arithmetic series", "等差数列求和",
       "Bars a, a + d, …, a + (n−1)d and the same bars turned over make an "
       "n × (2a + (n−1)d) rectangle.",
       r"\sum_{k=0}^{n-1}(a+kd) = \frac{n}{2}\left(2a+(n-1)d\right)", pop=71),
    _p("B24", "Sum of k over 2ᵏ", "k/2ᵏ之和",
       "Stack the halving rectangles as a staircase: row k holds k pieces of "
       "size 1/2ᵏ. Read by columns, the staircase is 1 + ½ + ¼ + …, and the "
       "columns after the first, stacked, rebuild the first: 2.",
       r"\sum_{k\geq 1}\frac{k}{2^k} = 2",
       note="Drawn with different horizontal and vertical scales: rows 1 to 6"
            " exactly, a thin band for the rows after them.",
       pop=46),
    _p("B25", "Telescoping strips", "裂项相消的条带",
       "Cut the unit square at 1/2, 1/3, 1/4, …: the strip between 1/(k+1) "
       "and 1/k has width 1/k − 1/(k+1) = 1/(k(k+1)), and the strips fill "
       "the square.",
       r"\sum_{k\geq 1}\frac{1}{k(k+1)} = 1", pop=55),
    _p("B26", "Cubes from consecutive odd numbers", "连续奇数之和为立方数",
       "Group the odd numbers 1 | 3 + 5 | 7 + 9 + 11 | …: the k-th group has k"
       " numbers centred on k². As bars of cells, each overhang past k² fills "
       "its partner's shortfall: the group levels into k bars of k², a k × k² "
       "block that cuts into k squares k × k, which is k³.",
       r"n^3 = (n^2-n+1) + (n^2-n+3) + \cdots + (n^2+n-1)",
       note="The first n groups are the first Tₙ odd numbers, which make Tₙ²;"
            " so 1³ + 2³ + … + n³ = Tₙ² (B4). Drawn for k = 1 to 4.",
       pop=54),
    _p("B27", "Pentagonal numbers", "五边形数",
       "The k-th pentagonal gnomon (3k − 2 dots) is two sides that make the "
       "k-th L-shaped shell of a square (2k − 1 dots) and one side that makes "
       "a row of a triangle (k − 1 dots): moved side by side, the nested "
       "pentagons become an n-square with Tₙ₋₁ on top.",
       r"P_n = \frac{n(3n-1)}{2} = n^2 + T_{n-1}",
       pop=48),
    _p("B28", "Hexagonal numbers are triangular", "六边形数是三角形数",
       "Each hexagonal gnomon (4k − 3 dots) is two consecutive rows of a "
       "triangle: its bottom V (2k − 1 dots) and its two vertical sides (2k − "
       "2). Re-stacked, the nested hexagons make the triangle T₂ₙ₋₁.",
       r"H_n = n(2n-1) = T_{2n-1}",
       pop=42),
    _p("B29", "Hexagons make cubes", "中心六边形数之和为立方数",
       "A centred hexagon of dots is three faces of a cube seen corner-on; "
       "the hexagons 1, 7, 19, … nest as shells into the n-cube.",
       r"\sum_{k=1}^{n}\left(3k(k-1)+1\right) = n^3", pop=58),
    _p("B30", "Adding triangular numbers", "三角形数的加法",
       "The staircase T_(a+b) is the staircases T_a and T_b with an a × b "
       "rectangle between them.",
       r"T_{a+b} = T_a + T_b + ab", pop=44),
    _p("B31", "Triangular number of a product", "乘积的三角形数",
       "Cut the staircase T_(ab) into b × b blocks: T_(a−1) full blocks and a "
       "small staircases, which regroup as T_a·T_b + T_(a−1)·T_(b−1).",
       r"T_{ab} = T_aT_b + T_{a-1}T_{b-1}",
       note="Drawn for a = 4, b = 3.",
       pop=30),
    _p("B32", "Doubling a triangular number", "T₂ₙ = 3Tₙ + Tₙ₋₁",
       "The staircase T₂ₙ cut into three copies of Tₙ and one of Tₙ₋₁.",
       r"T_{2n} = 3T_n + T_{n-1}",
       note="Drawn for n = 4.",
       pop=35),
    _p("B33", "Galileo's ratio", "伽利略的奇数比",
       "The first n odd numbers make the n-square; the next n make the L "
       "that grows it to the 2n-square, three times as large.",
       r"\frac{1+3+\cdots+(2n-1)}{(2n+1)+\cdots+(4n-1)} = \frac{1}{3}", pop=47),
    _p("B34", "The harmonic series diverges", "调和级数发散",
       "Group the bars ½ | ⅓ + ¼ | ⅕ + … + ⅛ | …: each group is at least as "
       "tall as ½, and there are endlessly many groups.",
       r"1+\frac{1}{2}+\frac{1}{3}+\frac{1}{4}+\cdots = \infty",
       note="Nicole Oresme's argument (14th century).", pop=75),
    _p("B35", "Squares of reciprocals stay under 2", "平方倒数和小于2",
       "Squares of side 1/k grouped by powers of two: group j holds 2ʲ "
       "squares, from side 1/2ʲ down, each inside a cell of side 1/2ʲ; the "
       "cells stack into a 1/2ʲ × 1 strip, and the strips 1, ½, ¼, … fill a 2 "
       "× 1 rectangle, with room to spare.",
       r"\sum_{k\geq 1}\frac{1}{k^2} < 2",
       pop=53),
    _p("B36", "Cassini's identity", "卡西尼恒等式",
       "The rectangle F_(n−1) × F_(n+1) and the square F_n² differ by one unit"
       " cell: remove their common part and the same comparison appears one "
       "step down, with the sign reversed; repeated, it ends with a single "
       "cell.",
       r"F_{n-1}F_{n+1} - F_n^2 = (-1)^n",
       note="Drawn for n = 6 (5 × 13 against 8²), in whole cells throughout.",
       pop=50),
    _p("B37", "Fibonacci sums of odd and even terms", "斐波那契奇数项与偶数项之和",
       "Bars of the odd-indexed Fibonacci numbers stack into F₂ₙ; the "
       "even-indexed ones into F₂ₙ₊₁ less one cell.",
       r"F_1+F_3+\cdots+F_{2n-1} = F_{2n},\quad F_2+F_4+\cdots+F_{2n} = F_{2n+1}-1",
       note="Drawn for n = 4.",
       pop=32),
    _p("B38", "Powers of three", "3的幂之和",
       "Around one middle cell lay copies of the figure on opposite sides, "
       "alternately left and right, below and above: with each pair the figure"
       " triples. A half-turn about the middle cell swaps the two sides, so "
       "each side, 1 + 3 + 9 + …, is half of 3ⁿ − 1.",
       r"1+3+9+\cdots+3^{n-1} = \frac{3^n-1}{2}",
       note="Drawn for n = 4 (a 9 × 9 board).",
       pop=34),
    _p("B39", "An alternating geometric series", "交错等比级数",
       "Pair the terms: (1 − ½) + (¼ − ⅛) + … = ½ + ⅛ + …. Done on a square "
       "(fill it, empty its left half, fill the lower-left quarter, empty its "
       "left half, …), the square falls into nested L's of three equal "
       "squares, and in every L two of the three are filled: twice the "
       "three-L's series ¼ + 1/16 + …, i.e. ⅔.",
       r"1-\frac{1}{2}+\frac{1}{4}-\frac{1}{8}+\cdots = \frac{2}{3}",
       pop=33),
    _p("B40", "Reciprocals of triangular numbers", "三角形数的倒数和",
       "1/Tₖ = 2/(k(k+1)): two copies of the telescoping strips fill two "
       "unit squares.",
       r"\sum_{k\geq 1}\frac{1}{T_k} = 2", pop=31),
    _p("B41", "Gabriel's staircase", "加百列阶梯",
       "Rectangles of width k and height rᵏ stack into a staircase S. Column "
       "1, c = r + r² + …, squashed to r of its height, fits under its own top"
       " block: c = r + rc = r/(1 − r). The whole staircase, squashed the same"
       " way and moved one column, is the staircase without column 1: S = c + "
       "rS, so S = c/(1 − r) = r/(1 − r)².",
       r"\sum_{k\geq 1}k\,r^k = \frac{r}{(1-r)^2},\quad 0<r<1",
       note="Drawn for r = 3/5, with different horizontal and vertical "
            "scales: rows 1 to 8 exactly, a thin band for the rows after "
            "them.",
       pop=36),
    _p("B42", "Half a square and half a diagonal", "半个正方形加半条对角线",
       "The staircase Tₙ is the triangle below the diagonal of the n-square "
       "plus the n half-cells it cuts.",
       r"1+2+\cdots+n = \frac{n^2}{2} + \frac{n}{2}",
       note="Drawn for an even n, so the teeth pair into whole cells.",
       pop=58),
    _p("B43", "Centred square numbers", "中心正方形数",
       "A diamond of dots is two interleaved squares, n² and (n−1)².",
       r"C_n = n^2 + (n-1)^2",
       note="Drawn for n = 5.",
       pop=30),
    _p("B44", "Odd numbers in a pyramid", "金字塔形的奇数和",
       "Rows of 1, 3, 5, … dots make a pyramid; cut it beside the middle "
       "column (each row of 2k − 1 splits as k + (k − 1)) and turn the smaller"
       " half over onto the slanted side of the larger: the n-square.",
       r"1+3+5+\cdots+(2n-1) = n^2",
       pop=55),
    _p("B45", "Sums of consecutive numbers", "连续整数之和",
       "A run of k ≥ 2 consecutive positive integers is a trapezoid of dots, k"
       " rows. For odd k it levels into k rows of the middle row (2m + k − "
       "1)/2; for even k its rows pair off into k/2 rows of 2m + k − 1. Either"
       " way it is a rectangle with an odd side at least 3 (k and 2m + k − 1 "
       "add up to an odd number), so a power of 2 is never such a sum.",
       r"m + (m+1) + \cdots + (m+k-1) = \frac{k(2m+k-1)}{2}",
       note="Drawn for 2 + 3 + 4 + 5 + 6 = 5·4 and 3 + 4 + 5 + 6 = 2·9.",
       pop=25),
    _p("B46", "Area of the Koch snowflake", "科赫雪花的面积",
       "Stage 1 folds out, on each side, the middle cell of the 9-cell grid of"
       " the triangle: three triangles of A/9. Every new triangle then has "
       "four children, each a ninth of it, so each stage adds 4/9 of the one "
       "before: 3·4ᵏ⁻¹ triangles of A/9ᵏ, a geometric series with ratio 4/9 on"
       " top of the first triangle.",
       r"A_\infty = A\left(1 + \frac{1}{3}\cdot\frac{1}{1-4/9}\right) = \frac{8}{5}A",
       note="Four stages are drawn; the series 1 + 4/9 + (4/9)² + … = 9/5 is "
            "summed by its formula.",
       pop=45),
    _p("B47", "Sierpiński's triangle has no area", "谢尔宾斯基三角形面积为零",
       "The removed triangles, ¼ + 3/16 + 9/64 + …, add up to the whole "
       "triangle.",
       r"\frac{1}{4}\sum_{k\geq 0}\left(\frac{3}{4}\right)^k = 1", pop=40),
    _p("B48", "The Cantor set has no length", "康托尔集长度为零",
       "The removed middle thirds, 1/3 + 2/9 + 4/27 + …, add up to the whole "
       "interval.",
       r"\frac{1}{3}\sum_{k\geq 0}\left(\frac{2}{3}\right)^k = 1", pop=38),
    _p("B49", "Halving a triangle", "二分三角形",
       "Halve a right isosceles triangle, halve the remaining half, and so "
       "on: the pieces ½, ¼, ⅛, … use up the whole triangle.",
       r"\frac{1}{2}+\frac{1}{4}+\frac{1}{8}+\cdots = 1", pop=60),

    # ---------------------------------------------------------------- C
    _p("C1", "Square of a sum", "和的平方",
       "A square of side a+b splits into four tiles.",
       r"(a+b)^2 = a^2 + 2ab + b^2",
       manim="C1_SquareOfSum", pop=93),
    _p("C2", "Square of a difference", "差的平方",
       "Remove two strips, add back the doubly-removed corner.",
       r"(a-b)^2 = a^2 - 2ab + b^2", pop=86),
    _p("C3", "Difference of squares", "平方差",
       "An L-shaped gnomon straightens into a rectangle.",
       r"a^2 - b^2 = (a+b)(a-b)",
       manim="C3_DifferenceOfSquares", pop=91),
    _p("C4", "Cube of a sum", "和的立方",
       "A cube of edge a+b splits into eight blocks.",
       r"(a+b)^3 = a^3 + 3a^2 b + 3ab^2 + b^3",
       vpy="C4", pop=83),
    _p("C5", "Difference of cubes", "立方差",
       "Take the b-cube from a corner of the a-cube: the rest is three slabs "
       "of thickness a − b, which lie flat as one.",
       r"a^3 - b^3 = (a-b)(a^2+ab+b^2)", pop=68),
    _p("C6", "Distributive law", "分配律",
       "One rectangle, two sub-rectangles.",
       r"a(b+c) = ab + ac", pop=75),
    _p("C7", "Completing the square", "配方法",
       "Split the bx strip in two and fill the missing corner.",
       r"x^2 + bx = \left(x+\frac{b}{2}\right)^2 - \frac{b^2}{4}", pop=81),
    _p("C8", "Four products in a square", "(a+b)² − (a−b)² = 4ab",
       "Four a × b rectangles, each a quarter-turn of the last about the "
       "centre, ring a square hole of side a − b; rectangles and hole together"
       " fill the (a + b)-square.",
       r"(a+b)^2 - (a-b)^2 = 4ab",
       pop=73),
    _p("C9", "Sum and difference squared", "(a+b)² + (a−b)² = 2(a²+b²)",
       "Lay two a-squares in opposite corners of the (a + b)-square: they "
       "overlap in an (a − b)-square and leave two b-square corners uncovered,"
       " so two a² and two b² cover (a + b)² once and (a − b)² twice.",
       r"(a+b)^2 + (a-b)^2 = 2\left(a^2+b^2\right)",
       pop=52),
    _p("C10", "Square of a trinomial", "三项和的平方",
       "An (a + b + c)-square cut into nine tiles: three squares and three "
       "pairs of equal rectangles.",
       r"(a+b+c)^2 = a^2+b^2+c^2+2ab+2bc+2ca", pop=66),
    _p("C11", "Sum of cubes", "立方和",
       "Lift three a × b × (a + b) slabs out of the (a + b)-cube: the cubes a³"
       " and b³ are what remain.",
       r"a^3+b^3 = (a+b)^3 - 3ab(a+b) = (a+b)\left(a^2-ab+b^2\right)",
       note="The slabs pair each a²b block with an ab² block, cycling round "
            "the cube; the factored form follows by taking out a + b.",
       pop=55),
    _p("C12", "Multiplying two sums", "两个和的乘积",
       "An (a + b) × (c + d) rectangle cut into four: ac, ad, bc, bd.",
       r"(a+b)(c+d) = ac+ad+bc+bd", pop=70),
    _p("C13", "Factoring with tiles", "十字相乘的图形",
       "The tiles x², an x × (p + q) strip cut into x × p and x × q, and a p ×"
       " q tile, laid edge to edge, are exactly an (x + p) × (x + q) "
       "rectangle.",
       r"x^2+(p+q)x+pq = (x+p)(x+q)",
       pop=62),
    _p("C14", "al-Khwārizmī's square", "花拉子米的配方",
       "x² + 10x = 39: four strips 2½ wide on the sides of the x-square, and "
       "four corner squares (2½)² complete the square (x + 5)² = 39 + 25.",
       r"x^2+10x=39\;\Rightarrow\;(x+5)^2=64,\;x=3",
       note="al-Khwārizmī (c. 820) solved this equation with this figure. The"
            " figure finds only the positive root, x = 3.",
       pop=58),
    _p("C15", "Babylonian multiplication", "巴比伦乘法",
       "Cut the excess off an a × b rectangle and move half of it: the "
       "square of the mean, short of the small square of half the "
       "difference.",
       r"ab = \left(\frac{a+b}{2}\right)^2 - \left(\frac{a-b}{2}\right)^2", pop=47),
    _p("C16", "Growth of a cube", "立方体的增量",
       "Grow the n-cube by one layer: three n × n slabs, three n-rods and "
       "one unit cube.",
       r"(n+1)^3 - n^3 = 3n^2+3n+1", pop=53),
    _p("C17", "Brahmagupta–Fibonacci identity", "婆罗摩笈多–斐波那契恒等式",
       "The right triangle with legs a, b (hypotenuse r) enlarged by c, and a "
       "copy enlarged by d and turned a quarter-turn, stand on one line with "
       "their hypotenuses cr and dr at a right angle. The segment joining "
       "their far ends is the hypotenuse of the right triangle with legs cr, "
       "dr and also of the one with legs ac − bd and ad + bc.",
       r"\left(a^2+b^2\right)\left(c^2+d^2\right) = (ac-bd)^2 + (ad+bc)^2",
       note="Drawn with ac > bd.",
       pop=33),
    _p("C18", "Sophie Germain's identity", "热尔曼恒等式",
       "Add 4a²b² to complete the square (a² + 2b²)², then take it out again "
       "as the square (2ab)²: a difference of two squares.",
       r"a^4+4b^4 = \left(a^2+2b^2+2ab\right)\left(a^2+2b^2-2ab\right)",
       note="The two a² × 2b² rectangles and the 2ab-square are matched by "
            "area (2·a²·2b² = (2ab)²), not by dissection.",
       pop=28),
    _p("C19", "Multiplication commutes", "乘法交换律",
       "An a × b array of dots, turned a quarter-turn, is a b × a array.",
       r"ab = ba", pop=60),
    _p("C20", "Difference of squares by trapezoids", "梯形证平方差",
       "Cut the L-shaped a² − b² along its diagonal into two trapezoids; turn "
       "one over and join: a parallelogram with base a + b and height a − b.",
       r"a^2 - b^2 = (a+b)(a-b)",
       pop=45),
    _p("C21", "One more than a rectangle", "n² 比 (n−1)(n+1) 多 1",
       "Turn the last column of the n-square a quarter-turn and lay it along "
       "the top as a new row: n − 1 of its cells complete an (n − 1) × (n + 1)"
       " rectangle, one cell is left over.",
       r"n^2 = (n-1)(n+1) + 1",
       note="Drawn for n = 5.",
       pop=40),
    _p("C22", "Egyptian unit fractions", "埃及分数",
       "Halve a rectangle, cut the other half in two thirds and one third: "
       "the pieces are ½, ⅓ and ⅙ of the whole.",
       r"\frac{1}{2}+\frac{1}{3}+\frac{1}{6} = 1", pop=30),
    _p("C23", "Difference of fourth powers", "四次方差",
       "The difference-of-squares rectangle twice: a⁴ − b⁴ is an "
       "(a² + b²) × (a² − b²) rectangle, and its short side is "
       "(a + b)(a − b) again.",
       r"a^4 - b^4 = \left(a^2+b^2\right)(a+b)(a-b)", pop=25),

    # ---------------------------------------------------------------- D
    _p("D1", "AM-GM in a semicircle", "半圆中的均值不等式",
       "A radius is never shorter than the altitude it stands over.",
       r"\frac{a+b}{2} \geq \sqrt{ab}",
       manim="D1_AMGM", pop=89),
    _p("D2", "The full chain of means", "均值链",
       "Four lengths in one semicircle, ordered by construction.",
       r"\sqrt{\frac{a^2+b^2}{2}} \geq \frac{a+b}{2} \geq \sqrt{ab} \geq \frac{2ab}{a+b}",
       manim="D2_MeansChain", pop=74),
    _p("D3", "Cauchy-Schwarz in the plane", "柯西-施瓦茨不等式",
       "A projection is never longer than the vector it came from.",
       r"|u\cdot v| \leq |u|\,|v|", pop=66),
    _p("D4", "Triangle inequality", "三角不等式",
       "A straight path beats a bent one.",
       r"|u+v| \leq |u| + |v|", pop=58),
    _p("D5", "x plus its reciprocal", "对勾不等式",
       "A rectangle of unit area has perimeter at least 4.",
       r"x + \frac{1}{x} \geq 2,\quad x>0", pop=64),
    _p("D6", "Rearrangement, two terms", "排序不等式",
       "The sign of a product of two differences.",
       r"a \geq b,\; x \geq y \;\Rightarrow\; ax+by-(ay+bx) = (a-b)(x-y) \geq 0", pop=42),
    _p("D7", "AM–GM from four rectangles", "四个矩形证均值不等式",
       "Four a × b rectangles fit inside the (a + b)-square around a square "
       "hole: 4ab ≤ (a + b)², with equality exactly when the hole closes.",
       r"\frac{a+b}{2} \geq \sqrt{ab}", pop=70),
    _p("D8", "Two squares cover two rectangles", "a² + b² ≥ 2ab",
       "Two a × b rectangles laid along two sides of the a-square overlap "
       "exactly in the b-square and leave the corner square (a − b)² "
       "uncovered: 2ab = a² + b² − (a − b)².",
       r"a^2 + b^2 \geq 2ab", pop=68),
    _p("D9", "Means in a trapezoid", "梯形中的四种平均",
       "In a trapezoid with parallel sides a < b: the segment through the "
       "crossing of the diagonals is the harmonic mean, the one cutting it "
       "into similar halves the geometric mean, the midline the arithmetic "
       "mean, the area bisector the root mean square — in that order.",
       r"\frac{2ab}{a+b} \leq \sqrt{ab} \leq \frac{a+b}{2} \leq \sqrt{\frac{a^2+b^2}{2}}",
       note="Only the geometric mean's length is derived on screen; the "
            "harmonic, arithmetic and root-mean-square segments are shown by "
            "their defining properties.",
       pop=56),
    _p("D10", "Crossed ladders", "交叉梯子",
       "Lines from the top of each of two poles to the foot of the other "
       "cross at a height h with 1/h = 1/a + 1/b, wherever the poles stand.",
       r"\frac{1}{h} = \frac{1}{a} + \frac{1}{b}", pop=45),
    _p("D11", "The square encloses most", "周长一定时正方形面积最大",
       "A rectangle a × b and the square of equal perimeter, side (a + b)/2, "
       "share a corner: the rectangle overhangs by (a − b)/2 and falls short "
       "of the square's top by as much; turned a quarter and slid in, the "
       "overhang fills that gap except a corner square ((a − b)/2)², always "
       "left uncovered.",
       r"ab \leq \left(\frac{a+b}{2}\right)^2",
       pop=60),
    _p("D12", "Young's inequality", "杨氏不等式",
       "For increasing f with f(0) = 0: the area under f up to a and the "
       "area to the left of it up to b together always cover the a × b "
       "rectangle.",
       r"ab \leq \int_0^a f(x)\,dx + \int_0^b f^{-1}(y)\,dy", pop=44),
    _p("D13", "Jensen's inequality", "詹森不等式",
       "On a convex curve the chord lies above the arc: the value at a "
       "weighted mean is at most the weighted mean of the values.",
       r"f\left(tx+(1-t)y\right) \leq t\,f(x)+(1-t)\,f(y)", pop=50),
    _p("D14", "eˣ lies above its tangent", "eˣ ≥ 1 + x",
       "The area under eᵗ from 0 to x is eˣ − 1: for x > 0 it contains the "
       "unit-height rectangle of area x, for x < 0 it lies inside the one of "
       "area −x. Either way eˣ − 1 ≥ x, so the tangent y = 1 + x never rises "
       "above the curve.",
       r"e^x \geq 1 + x",
       note="Uses ∫₀ˣ eᵗ dt = eˣ − 1.",
       pop=57),
    _p("D15", "ln x lies below its tangent", "ln x ≤ x − 1",
       "The mirror image of D14 in the line y = x: the band between the y-axis"
       " and y = ln x, from height 0 to ln x, has area x − 1 and contains (x >"
       " 1) or lies inside (x < 1) the width-1 rectangle of area |ln x|; so ln"
       " x ≤ x − 1 and the tangent at x = 1 stays above the curve.",
       r"\ln x \leq x - 1,\quad x > 0",
       note="Uses ∫₀ˢ eʸ dy = eˢ − 1, as D14 does.",
       pop=45),
    _p("D16", "e to the π, π to the e", "e^π 与 π^e 谁大",
       "The line from the origin touching y = ln x touches at x = e with slope"
       " 1/e; the line to (π, ln π) is lower, ln π/π < 1/e.",
       r"\pi^e < e^\pi",
       note="ln is concave, so it lies below this tangent; at x = π the gap "
            "π/e − ln π ≈ 0.011 is shown in a magnifier.",
       pop=55),
    _p("D17", "Napier's inequality", "纳皮尔不等式",
       "Under y = 1/x from a to b the area ln b − ln a lies between the inner "
       "rectangle (b − a)/b and the outer one (b − a)/a.",
       r"\frac{1}{b} < \frac{\ln b-\ln a}{b-a} < \frac{1}{a},\quad 0 < a < b",
       note="Uses ∫ dx/x = ln b − ln a over [a, b].",
       pop=40),
    _p("D18", "Harmonic numbers and the logarithm", "调和数与对数",
       "Bars of height 1/k (k = 1, …, n) over [k, k + 1] stand above y = 1/x "
       "and contain the region under it from 1 to n + 1; slid one unit left "
       "onto [k − 1, k], bars 2, …, n lie under the curve from 1 to n and bar "
       "1 is the unit square.",
       r"\ln(n+1) < 1+\frac{1}{2}+\cdots+\frac{1}{n} < 1+\ln n",
       note="Drawn for n = 5.",
       pop=52),
    _p("D19", "Bounds for n factorial", "n! 的上下界",
       "Bars of heights ln 2, …, ln n: standing over [k − 1, k] they cover the"
       " region under y = ln x from 1 to n; slid one unit right, onto [k, k + "
       "1], they fit under it from 1 to n + 1.",
       r"n\ln n - n + 1 < \ln n! < (n+1)\ln(n+1) - n,\quad n \geq 2",
       note="Drawn for n = 5. Uses ∫ ln x dx = t ln t − t + 1 over [1, t] "
            "(H17).",
       pop=38),
    _p("D20", "The mediant", "中位分数不等式",
       "Arrows (b, a) and (d, c) placed head to tail: the slope of their sum "
       "lies between their slopes.",
       r"\frac{a}{b} < \frac{c}{d} \;\Rightarrow\; \frac{a}{b} < \frac{a+c}{b+d} < \frac{c}{d},\quad b, d > 0",
       note="Drawn with a, b, c, d > 0; the argument needs only b, d > 0.",
       pop=41),
    _p("D21", "Heron's shortest path", "海伦最短路径",
       "From A to a line and on to B: reflect B in the line; the straight "
       "segment to the reflection is shortest, and where it crosses, the two "
       "angles with the line are equal.",
       r"AP + PB = AP + PB' \geq AB'",
       note="A and B lie on the same side of the line.",
       pop=62),
    _p("D22", "Chord shorter than arc", "弦短于弧",
       "Between two points of the unit circle the vertical drop is at most the"
       " chord, and the chord is shorter than the arc.",
       r"|\sin x - \sin y| \leq |x - y|",
       note="Drawn for x = 20°, y = 120°; the argument holds for any two "
            "points.",
       pop=30),
    _p("D23", "AM–GM from a tangent line", "切线证均值不等式",
       "The tangent y = x lies below y = eˣ⁻¹: each ratio xᵢ/A is at most "
       "e^(xᵢ/A − 1), and the product of the right sides is e⁰ = 1.",
       r"\frac{x_1+\cdots+x_n}{n} \geq \sqrt[n]{x_1\cdots x_n},\quad x_i > 0",
       note="Pólya's argument; A is the arithmetic mean. Takes eˣ ≥ 1 + x "
            "(D14) as known. Drawn for n = 3.",
       pop=35),
    _p("D24", "Bounds for ln(1 + x)", "ln(1+x) 的上下界",
       "Under y = 1/t the region between t = 1 and t = 1 + x lies between the "
       "rectangles on the same base of heights 1/(1 + x) and 1 — for x > 0, "
       "and with the roles swapped for −1 < x < 0.",
       r"\frac{x}{1+x} < \ln(1+x) < x,\quad x > -1,\; x \neq 0",
       note="Uses ∫ dt/t = ln u over [1, u].",
       pop=36),
    _p("D25", "Square roots add above", "√a + √b ≥ √(a+b)",
       "Four right triangles with legs √a and √b fill the corners of the (√a +"
       " √b)-square around the square on the hypotenuse; slid into two "
       "rectangles they leave the squares a and b, so the hypotenuse is √(a + "
       "b), and its square sits inside the bigger one with the four triangles "
       "to spare.",
       r"\sqrt{a} + \sqrt{b} \geq \sqrt{a+b}",
       pop=35),
    _p("D26", "Minkowski's inequality", "闵可夫斯基不等式",
       "Two arrows head to tail against the single arrow of their sum: a bent "
       "path is never shorter.",
       r"\sqrt{a^2+b^2} + \sqrt{c^2+d^2} \geq \sqrt{(a+c)^2+(b+d)^2}",
       note="Drawn with a, b, c, d > 0; the inequality holds for all real "
            "numbers.",
       pop=30),
    _p("D27", "A sum times its reciprocals", "(a+b)(1/a+1/b) ≥ 4",
       "The (a + b) × (1/a + 1/b) rectangle splits into two pieces of area 1 "
       "and two of areas a/b and b/a; laid over the last two, the unit pieces "
       "fit inside them with an (a − b) × (1/b − 1/a) corner to spare, so a/b "
       "+ b/a ≥ 2.",
       r"(a+b)\left(\frac{1}{a}+\frac{1}{b}\right) \geq 4,\quad a, b > 0",
       note="Drawn with a > b; equality when a = b.",
       pop=30),
    _p("D28", "Products of three numbers", "ab + bc + ca ≤ a² + b² + c²",
       "D8 three times: the three pairs of squares cover the three pairs of "
       "rectangles twice over, with three squares of differences to spare.",
       r"ab+bc+ca \leq a^2+b^2+c^2",
       note="Drawn for a > b > c > 0; the inequality holds for all real "
            "numbers.",
       pop=30),

    # ---------------------------------------------------------------- E
    _p("E1", "Circle area by wedges", "圆面积（扇形展开）",
       "Cut the disc into wedges and interleave them into a rectangle.",
       r"A = \frac{1}{2}(2\pi r)\cdot r = \pi r^2",
       manim="E1_CircleWedges", pop=95),
    _p("E2", "Circle area by unrolling annuli", "圆面积（圆环展开）",
       "Unroll the concentric rings into a triangle.",
       r"A = \frac{1}{2}\cdot(2\pi r)\cdot r = \pi r^2", pop=77),
    _p("E3", "Inscribed angle theorem", "圆周角定理",
       "An isosceles triangle on the radius doubles the angle.",
       r"\angle ABC = \frac{1}{2}\angle AOC",
       manim="E3_InscribedAngle", pop=84),
    _p("E4", "Thales' theorem", "泰勒斯定理",
       "The special case of the inscribed angle on a diameter.",
       r"\angle ACB = 90^\circ", pop=85),
    _p("E5", "Power of a point", "圆幂定理",
       "Two similar triangles cut by a pair of secants.",
       r"PA\cdot PB = PC\cdot PD", pop=63),
    _p("E6", "Secant-tangent", "切割线定理",
       "The tangent is the geometric mean of the whole secant and its "
       "external part.",
       r"PT^2 = PA\cdot PB",
       note="Uses the tangent–chord angle, the limiting case of the inscribed "
            "angle theorem (E3).", pop=52),
    _p("E7", "Ptolemy's theorem", "托勒密定理",
       "Triangles ABD, ACD, ABC, scaled by AC, AB, AD, tile a "
       "parallelogram.",
       r"AC\cdot BD = AB\cdot CD + BC\cdot AD", pop=61),
    _p("E8", "Area of an ellipse", "椭圆面积",
       "An affine stretch scales every area by the same factor.",
       r"A = \pi ab", pop=57),
    _p("E9", "Annulus from one chord", "圆环面积",
       "The area depends only on the chord tangent to the inner circle.",
       r"A = \pi\left(\frac{L}{2}\right)^2", pop=50),
    _p("E10", "Quadrature of the parabola", "抛物线求积",
       "Archimedes' exhaustion by inscribed triangles.",
       r"A = \frac{4}{3}\,A_{triangle}",
       note="The triangle is the inscribed one whose apex is where the "
            "tangent is parallel to the chord.", pop=65),
    _p("E11", "Area of a sector", "扇形面积",
       "Cut the sector into thin wedges and interleave them: a near-"
       "rectangle of height r and width half the arc, ½rθ.",
       r"A = \frac{1}{2}r^2\theta", pop=70),
    _p("E12", "Tangent–chord angle", "弦切角定理",
       "The angle between a tangent and a chord equals the inscribed angle on "
       "the other side of the chord: both are half the central angle.",
       r"\theta = \frac{1}{2}\angle AOB = \angle ACB",
       note="Drawn for an acute tangent–chord angle.",
       pop=55),
    _p("E13", "Angle between two chords", "圆内角定理",
       "Two chords crossing inside the circle: the angle between them is the "
       "exterior angle of a triangle of two inscribed angles, half the sum "
       "of the arcs they cut off.",
       r"\theta = \frac{\alpha+\beta}{2}", pop=45),
    _p("E14", "Angle between two secants", "圆外角定理",
       "Through the near point C draw the chord CE parallel to the other "
       "secant PAB. A fold over the diameter square to both chords swaps A, B "
       "and C, E, so the arc BE equals the near arc AC (β) and the rest of the"
       " far arc, ED, is α − β. Slid along the secant from P to C, θ fills "
       "∠ECD, an inscribed angle on ED: θ = (α − β)/2.",
       r"\theta = \frac{\alpha-\beta}{2}",
       note="Uses the inscribed angle theorem (E3).",
       pop=40),
    _p("E15", "Cyclic quadrilateral", "圆内接四边形对角互补",
       "Opposite angles stand on two arcs that make up the whole circle: half "
       "of 360°.",
       r"\angle A + \angle C = 180^\circ",
       note="Uses the inscribed angle theorem (E3).",
       pop=64),
    _p("E16", "Equal tangents", "切线长定理",
       "The two tangents from an outside point are mirror images in the "
       "line through the centre.",
       r"PA = PB", pop=58),
    _p("E17", "Reflection property of the parabola", "抛物线的光学性质",
       "PF = PD, so the perpendicular bisector of FD passes through P and "
       "halves ∠FPD; every other point of it is farther from F than from the "
       "directrix, so it is the tangent at P: rays parallel to the axis "
       "reflect through the focus.",
       r"PF = PD",
       pop=60),
    _p("E18", "Reflection property of the ellipse", "椭圆的光学性质",
       "Every other point Q of the tangent is outside the ellipse, so QF₁ + "
       "QF₂ > 2a = PF₁ + PF₂. Reflect F₂ in the tangent to F₂′: QF₂′ = QF₂, so"
       " the path F₁QF₂′ is shortest at P, hence straight — F₁, P, F₂′ are "
       "collinear and the two angles at P are equal.",
       r"PF_1 + PF_2 = 2a",
       pop=55),
    _p("E19", "Reflection property of the hyperbola", "双曲线的光学性质",
       "Fold PF₂ over the bisector ℓ of ∠F₁PF₂: it lies along PF₁, F₂ landing "
       "at F₂′ with F₂′F₁ = PF₁ − PF₂ = 2a. Any other point Q of ℓ has QF₂ = "
       "QF₂′ and QF₁ < QF₂′ + F₂′F₁, so QF₁ − QF₂ < 2a: Q lies between the "
       "branches. ℓ touches the branch at P without crossing it, so it is the "
       "tangent, and it bisects the angle between the focal radii.",
       r"|PF_1 - PF_2| = 2a",
       note="Drawn on the branch round F₂; the other branch is its mirror "
            "image.",
       pop=38),
    _p("E20", "Dandelin spheres", "丹德林双球",
       "Two balls in the cone touch the cutting plane at F₁, F₂ and the cone "
       "along two circles. PF₁ and PQ₁ (Q₁ where the generator through P meets"
       " the upper circle) are tangents from P to the same ball, so they are "
       "equal; likewise PF₂ = PQ₂. So PF₁ + PF₂ = Q₁Q₂, the same on every "
       "generator.",
       r"PF_1 + PF_2 = 2a",
       note="Shows the ellipse, where the plane cuts every generator.",
       pop=54),
    _p("E21", "The arbelos", "鞋匠刀形",
       "Semicircles are π/8 of the squares on their diameters, so the arbelos "
       "is π/8·(AB² − AC² − CB²) = π/8·2·AC·CB; and AC·CB = CD² because ADB is"
       " right-angled at D: the arbelos has the area of the circle on the "
       "half-chord CD.",
       r"A_{arbelos} = \pi\left(\frac{CD}{2}\right)^2",
       note="Due to Archimedes.",
       pop=48),
    _p("E22", "The salinon", "盐窖形",
       "Semicircles are π/8 of the squares on their diameters, so the salinon "
       "is π/8·(AB² + CD² − AC² − DB²). The axis EF, turned a quarter turn "
       "about the centre, lies on AD; two squares on AD in opposite corners of"
       " the square on AB overlap in the square on CD and leave the squares on"
       " AC and DB, so AB² + CD² = 2·AD² + AC² + DB² and the salinon is "
       "π/8·2·EF², the circle on EF.",
       r"A_{salinon} = \pi\left(\frac{d}{2}\right)^2",
       pop=30),
    _p("E23", "Inscribed and circumscribed squares", "内接正方形与外切正方形",
       "Turn the inner square 45°: its corners touch the midpoints of the "
       "outer square, and it is exactly half of it. The same holds for the "
       "circles in and around a square.",
       r"A_{inner} = \frac{1}{2}A_{outer}", pop=63),
    _p("E24", "π between hexagons", "正六边形夹逼π",
       "The inscribed hexagon has perimeter 6r, the circumscribed one 4√3·r, "
       "and the circle lies between them.",
       r"3 < \pi < 2\sqrt{3}",
       note="The upper bound uses that a convex arc is shorter than an "
            "enclosing path with the same ends.",
       pop=57),
    _p("E25", "Two secants from a point", "割线定理",
       "Triangles PAD and PCB (A, C the nearer points) share the angle at P "
       "and have equal angles at B and D (same arc): they are similar.",
       r"PA\cdot PB = PC\cdot PD",
       pop=50),
    _p("E26", "The Simson line", "西姆松线",
       "P on the arc BC, feet D on BC, E on CA, F on AB produced; γ = ∠PCA, δ "
       "= 180° − γ. ABPC is cyclic, so ∠ABP = δ and its neighbour ∠PBF = γ. D "
       "and F lie on the circle on diameter PB, so sliding the vertex from B "
       "to D keeps ∠PDF = γ; D and E lie on the circle on diameter PC, so in "
       "the cyclic PDCE ∠PDE = δ. At the middle foot D, γ + δ = 180°: D, E, F "
       "are collinear.",
       r"D,\ E,\ F\ \mathrm{collinear}",
       note="Drawn for P on the arc BC opposite A, where D is the middle foot"
            " and F lies on AB produced; uses E3 and E15.",
       pop=34),
    _p("E27", "Miquel's theorem", "密克定理",
       "Let the circles AEF and BFD meet again at M. Their exterior angles at "
       "M equal the angles at A and B; on the straight line through D and M "
       "the angle left over equals the angle at C, so CDME is cyclic too: all "
       "three circles pass through M.",
       r"(AEF)\cap(BFD)\cap(CDE) = \{M\}",
       note="Drawn with D, E, F inside the sides and M inside the triangle; "
            "other positions work the same with directed angles.",
       pop=26),
    _p("E28", "Thales by a rectangle", "矩形证泰勒斯定理",
       "Turn the triangle on a diameter half a turn about the centre: with "
       "its copy it makes a parallelogram with equal diagonals — a "
       "rectangle.",
       r"\angle ACB = 90^\circ", pop=60),
    _p("E29", "A radius bisects the chord it meets at right angles", "垂径定理",
       "Reflect in the line through the centre perpendicular to the chord: "
       "the circle and the chord map to themselves, swapping the halves.",
       r"OM\perp AB \;\Rightarrow\; AM = MB", pop=55),
    _p("E30", "Tangent at right angles to the radius", "切线垂直于半径",
       "Every other point of the tangent is outside the circle, farther "
       "from the centre than the point of contact: the radius is the "
       "shortest distance, so it is perpendicular.",
       r"OT\perp t", pop=45),
    _p("E31", "Tangent to a parabola", "抛物线的切线",
       "y = x² has focus F(0, ¼) and directrix y = −¼. For P(a, a²) with foot "
       "D on the directrix, PF = PD, so the perpendicular bisector of FD "
       "passes through P and every other point of it is farther from F than "
       "from the directrix: it is the tangent. F and D lie ¼ above and below "
       "the x-axis, so it crosses the axis at the midpoint of FD, (a/2, 0); a "
       "half-turn about that point carries P to (0, −a²) on the same line: the"
       " subtangent is half the abscissa.",
       r"y = 2ax - a^2",
       note="Uses the focus–directrix property, as in E17.",
       pop=30),
    _p("E32", "Tangent to a hyperbola", "双曲线的切线三角形",
       "At (1, 1) the mirror y = x maps xy = 1 to itself, so the tangent there"
       " is x + y = 2: P is the midpoint and the triangle has area 2. The "
       "squeeze (x, y) → (tx, y/t) slides the hyperbola along itself and keeps"
       " lines, midpoints and areas, so at P = (t, 1/t) the tangent cuts the "
       "axes at 2t and 2/t and the triangle still has area 2.",
       r"\frac{1}{2}\cdot 2t\cdot\frac{2}{t} = 2",
       note="Drawn on the branch in the first quadrant.",
       pop=25),
    _p("E33", "Archimedes' broken chord", "阿基米德折弦定理",
       "M is the midpoint of the arc ABC. The turn about M that takes C to A "
       "carries B to G on the longer chord AB, so AG = BC and MG = MB; the "
       "foot F of the perpendicular from M halves GB, so AF = FB + BC: F "
       "bisects the broken chord.",
       r"AF = FB + BC",
       pop=22),
    _p("E34", "Three circles on a line", "直线上相切的三个圆",
       "Circles touching each other and a line: the common tangent lengths "
       "2√(r₁r₂) add up, so 1/√r = 1/√r₁ + 1/√r₂ for the small one between.",
       r"\frac{1}{\sqrt{r}} = \frac{1}{\sqrt{r_1}} + \frac{1}{\sqrt{r_2}}",
       note="A favourite of the Japanese temple tablets (sangaku).", pop=30),
    _p("E35", "Common tangent of touching circles", "外切两圆的公切线",
       "Join the centres and drop the radii to the tangent: sliding the "
       "tangent segment up by r₂ gives a right triangle with hypotenuse r₁ + "
       "r₂ and leg r₁ − r₂; four r₁ × r₂ rectangles round an (r₁ − r₂)² hole "
       "show (r₁ + r₂)² − (r₁ − r₂)² = 4r₁r₂.",
       r"t = 2\sqrt{r_1r_2}",
       note="Drawn with r₁ > r₂.",
       pop=28),
    _p("E36", "Area of the Reuleaux triangle", "勒洛三角形的面积",
       "The triangle and one cap make a 60° sector of radius s. Add two more "
       "triangles: the sector at B slid along BA and the sector at C turned "
       "half a turn about the midpoint of AC join the sector at A in a "
       "half-disc of radius s, so A + 2·(√3/4)s² = ½πs².",
       r"A = \frac{1}{2}\left(\pi-\sqrt{3}\right)s^2",
       pop=35),
    _p("E37", "The circle of Apollonius", "阿波罗尼斯圆",
       "Points P with PA : PB = k: the inner and outer bisectors at P meet AB "
       "at fixed points and are perpendicular, so P sees that segment at a "
       "right angle — a circle.",
       r"\frac{PA}{PB} = k",
       note="Uses the angle bisector theorem (F19) for the inner and the "
            "outer bisector; for the outer one AD : DB = k is stated, not "
            "shown. Drawn for k = 2; for k = 1 the locus is the perpendicular"
            " bisector of AB.",
       pop=32),
    _p("E38", "The common chord", "两圆公共弦",
       "Reflect the figure in the line of centres: both circles map to "
       "themselves, so the two crossing points are mirror images and the "
       "chord is perpendicular to that line, bisected by it.",
       r"O_1O_2 \perp AB", pop=30),
    _p("E39", "Length of a chord", "弦长公式",
       "The radius bisecting the central angle θ cuts the isosceles triangle "
       "into two right triangles with hypotenuse r and angle θ/2 at the "
       "centre; the half-chord opposite that angle is r·sin(θ/2).",
       r"AB = 2r\sin\frac{\theta}{2}",
       note="Drawn for θ < 180°.",
       pop=45),
    _p("E40", "Equal angles on a segment lie on a circle", "定弦定角的轨迹是圆弧",
       "Points seeing AB at the same angle lie on one arc: for Q off the arc, "
       "let AQ meet the arc at X, which sees AB at φ; slid along that line to "
       "Q, φ fits strictly inside ∠AQB when Q is inside the circle and "
       "overshoots it when Q is outside, so only points of the arc see AB at "
       "φ.",
       r"\angle APB = \angle AQB \;\Rightarrow\; A,B,P,Q\ \mathrm{concyclic}",
       note="P and Q on the same side of AB. Uses the inscribed angle theorem"
            " (E3).",
       pop=40),

    # ---------------------------------------------------------------- F
    _p("F1", "Angle sum of a triangle", "三角形内角和",
       "Three angles rotate onto a straight line.",
       r"A + B + C = 180^\circ",
       manim="F1_AngleSum", pop=97),
    _p("F2", "Exterior angle theorem", "外角定理",
       "The two remote interior angles, carried to the vertex, exactly fill "
       "the exterior angle.",
       r"\gamma_{ext} = \alpha + \beta", pop=79),
    _p("F3", "Polygon angle sum", "多边形内角和",
       "Fan a convex polygon from one vertex into n-2 triangles.",
       r"S = (n-2)\cdot 180^\circ", pop=83),
    _p("F4", "Exterior angles of a convex polygon", "多边形外角和",
       "Shrink the polygon to a point: the exterior angles close a full turn.",
       r"\sum \gamma_{ext} = 360^\circ", pop=73),
    _p("F5", "Triangle area by shear", "三角形面积",
       "Shear the apex along a parallel: the area never moves.",
       r"A = \frac{1}{2}bh", pop=87),
    _p("F6", "Centroid divides the median 2:1", "重心分中线",
       "Two midlines and a parallelogram.",
       r"AG:GM = 2:1", pop=60),
    _p("F7", "Viviani's theorem", "维维亚尼定理",
       "Three perpendiculars from any interior point stack to the altitude.",
       r"d_1 + d_2 + d_3 = h",
       manim="F7_Viviani", pop=69),
    _p("F8", "Varignon's theorem", "瓦里尼翁定理",
       "For any simple quadrilateral, convex or not, the midpoint "
       "quadrilateral is a parallelogram of half the area.",
       r"[M_1M_2M_3M_4] = \frac{1}{2}[ABCD]",
       note="The animation shows the convex case. For a self-intersecting "
            "quadrilateral the midpoints still form a parallelogram, but its "
            "area is half the difference of the two lobes.", pop=59),
    _p("F9", "Napoleon's theorem", "拿破仑定理",
       "The centres of equilateral triangles erected outward on the three "
       "sides form an equilateral triangle.",
       r"\triangle N_1N_2N_3 \text{ is equilateral}", pop=56),
    _p("F10", "Ceva's theorem by areas", "塞瓦定理",
       "Three concurrent cevians, compared through the areas they cut.",
       r"\frac{BD}{DC}\cdot\frac{CE}{EA}\cdot\frac{AF}{FB} = 1", pop=48),
    _p("F11", "Pick's theorem", "皮克定理",
       "Lattice points count the area of a simple lattice polygon.",
       r"A = I + \frac{B}{2} - 1",
       note="In the polygon shown every triangle has a unit side on one "
            "lattice line and its apex on the next, so its area ½ can be "
            "seen. For a general lattice polygon that step needs the lemma "
            "that a lattice triangle with no other lattice points has area ½.", pop=71),
    _p("F12", "Area of a parallelogram", "平行四边形面积",
       "Cut a right triangle off one end and slide it to the other: a b × h "
       "rectangle.",
       r"A = bh",
       note="Drawn with the top overhanging the base by less than b, so one "
            "cut along an altitude suffices; a more slanted parallelogram "
            "needs several such cuts, or the shear of F5.",
       pop=88),
    _p("F13", "Area of a trapezoid", "梯形面积",
       "Two copies of the trapezoid, one turned half a turn about the midpoint"
       " of a leg, make a parallelogram of base a + b and the same height.",
       r"A = \frac{1}{2}(a+b)h",
       pop=85),
    _p("F14", "Triangle area by a half-turn", "三角形面积（旋转拼成矩形）",
       "Cut along the midline, cut the top triangle along its altitude, and "
       "turn each piece half a turn about the midpoint of its side: the "
       "triangle becomes a b × h/2 rectangle on the same base.",
       r"A = \frac{1}{2}bh",
       note="Take the longest side as the base, so that the altitude falls "
            "inside it.",
       pop=80),
    _p("F15", "Area of a rhombus or kite", "菱形与筝形的面积",
       "The diagonals cut it into four right triangles, which are exactly half"
       " of the d₁ × d₂ rectangle around it.",
       r"A = \frac{1}{2}d_1d_2",
       note="Nothing in the dissection uses the kite's symmetry: it works for"
            " a rhombus and for any convex quadrilateral with perpendicular "
            "diagonals.",
       pop=62),
    _p("F16", "Area from the inradius", "内切圆半径与面积",
       "Join the incentre to the vertices: three triangles of height r on the "
       "three sides; unrolled side by side, they make one triangle of height r"
       " on base a + b + c.",
       r"A = rs,\quad s = \frac{a+b+c}{2}",
       pop=58),
    _p("F17", "Midline theorem", "三角形中位线定理",
       "Turn the small top triangle half a turn about a side's midpoint: "
       "the midline and its copy make a segment as long as the base and "
       "parallel to it.",
       r"MN \parallel BC,\quad MN = \frac{1}{2}BC", pop=66),
    _p("F18", "Medians cut six equal areas", "三条中线分出六个等积三角形",
       "Each median halves the triangle; small triangles on equal bases "
       "with a common apex are equal in pairs, so all six are equal.",
       r"[\triangle_1] = \cdots = [\triangle_6] = \frac{1}{6}[ABC]", pop=46),
    _p("F19", "Angle bisector theorem", "角平分线定理",
       "The bisector's foot D is equally far from AB and AC, so triangles "
       "ABD and ACD have areas in the ratio AB : AC — and also BD : DC.",
       r"\frac{BD}{DC} = \frac{AB}{AC}", pop=60),
    _p("F20", "Altitudes are concurrent", "三条高交于一点",
       "Through each vertex draw the parallel to the opposite side: the "
       "altitudes become the perpendicular bisectors of the doubled "
       "triangle, which meet at its circumcentre.",
       r"AD\cap BE\cap CF = \{H\}", pop=57),
    _p("F21", "Perpendicular bisectors meet", "三边中垂线交于一点",
       "Points on the bisector of AB are as far from A as from B; where two "
       "bisectors cross, the point is as far from all three vertices, so it "
       "lies on the third.",
       r"OA = OB = OC",
       note="Drawn for an acute triangle; the argument is the same for any "
            "triangle.",
       pop=50),
    _p("F22", "Angle bisectors meet", "三条角平分线交于一点",
       "Points on an angle bisector are as far from its two sides; where two "
       "bisectors cross, the point is as far from all three sides.",
       r"d(I,AB) = d(I,BC) = d(I,CA)", pop=49),
    _p("F23", "The Euler line", "欧拉线",
       "The half-turn about the centroid with ratio ½ takes the triangle to "
       "its midpoint triangle and H to that triangle's orthocentre, which is "
       "O: H, G, O lie on a line with HG = 2GO.",
       r"HG = 2\,GO",
       note="Drawn for an acute, non-equilateral triangle.",
       pop=48),
    _p("F24", "The nine-point circle", "九点圆",
       "Halving towards H, and the half-turn-and-halving about G, carry the "
       "circumcircle onto one circle of half the radius, through the side "
       "midpoints and the midpoints of HA, HB, HC; the two images of a radius "
       "are opposite radii, so each foot of an altitude sees a diameter at a "
       "right angle and lies on the circle too.",
       r"R_9 = \frac{R}{2}",
       note="Drawn for an acute triangle; the theorem holds for every "
            "triangle.",
       pop=42),
    _p("F25", "Menelaus's theorem", "梅涅劳斯定理",
       "A line cuts the three sides (extended): drop perpendiculars from the "
       "vertices to it; each ratio becomes a ratio of two of the three "
       "heights, and they cancel.",
       r"\frac{AF}{FB}\cdot\frac{BD}{DC}\cdot\frac{CE}{EA} = 1",
       note="Unsigned lengths (signed ratios give −1); drawn with the line "
            "cutting AB and CA inside and BC produced.",
       pop=40),
    _p("F26", "Base angles of an isosceles triangle", "等腰三角形两底角相等",
       "Turn the triangle over about the bisector of the apex angle: AB falls "
       "along AC, and since AB = AC, B lands on C and C on B — the triangle "
       "lands on itself with the base angles exchanged.",
       r"AB = AC \;\Rightarrow\; \angle B = \angle C",
       pop=67),
    _p("F27", "The five-pointed star", "五角星五个角之和",
       "The line BE cuts the point at A off as a triangle AXY; its angles at X"
       " and Y are exterior angles of triangles XCE and YDB, so they hold the "
       "points C + E and B + D; the three angles of AXY then line up at A into"
       " a straight angle.",
       r"\alpha_1+\alpha_2+\alpha_3+\alpha_4+\alpha_5 = 180^\circ",
       pop=56),
    _p("F28", "Area of a regular polygon", "正多边形面积",
       "n triangles of height a (the apothem) on the n sides: laid in a row "
       "and interleaved, half of a p × a rectangle.",
       r"A = \frac{1}{2}pa",
       note="Drawn for n = 7.",
       pop=54),
    _p("F29", "Area of an equilateral triangle", "正三角形面积",
       "Cut along an altitude, turn one half over (the halves are mirror "
       "images) and re-pair them into an s/2 by (√3/2)s rectangle.",
       r"A = \frac{\sqrt{3}}{4}s^2",
       pop=59),
    _p("F30", "Area of a regular hexagon", "正六边形面积",
       "Six equilateral triangles of side s and height (√3/2)s; the two strips"
       " of three, set side by side, make a parallelogram of base 3s, and half"
       " a triangle moved from one end to the other squares it off: a 3s × "
       "(√3/2)s rectangle.",
       r"A = \frac{3\sqrt{3}}{2}s^2",
       pop=53),
    _p("F31", "The Fermat point", "费马点",
       "Turn triangle APB by 60° about A: PA + PB + PC becomes a path from C "
       "to a fixed point, shortest when straight — then each side subtends "
       "120°.",
       r"\angle APB = \angle BPC = \angle CPA = 120^\circ",
       note="Needs every angle of the triangle below 120°.",
       pop=47),
    _p("F32", "The one-seventh triangle", "七分之一三角形",
       "Cevians to the points one third along each side cut out a central "
       "triangle; three more cuts parallel to the cevians, and half-turns "
       "about the side midpoints, regroup the pieces into seven copies of it.",
       r"[DEF] = \frac{1}{7}[ABC]",
       pop=51),
    _p("F33", "Regular polygons that tile", "能铺满平面的正多边形",
       "Around a point the corners must make 360°. A regular polygon's corner "
       "is 60°, 90°, 108°, 120° or lies strictly between 120° and 180°; only "
       "60°, 90° and 120° go into 360° a whole number of times, so only "
       "triangles, squares and hexagons tile.",
       r"k\cdot\frac{(n-2)\,180^\circ}{n} = 360^\circ \;\Rightarrow\; n\in\{3,4,6\}",
       pop=61),
    _p("F34", "Pitot's theorem", "皮托定理",
       "The equal tangents from each vertex to the incircle: each pair of "
       "opposite sides uses each of the four tangent lengths once.",
       r"a + c = b + d", pop=37),
    _p("F35", "Van Aubel's theorem", "范·奥贝尔定理",
       "Squares on the sides of any quadrilateral: the segments PR and QS "
       "joining the centres of opposite squares are equal and perpendicular. "
       "With M the midpoint of a diagonal AC, a quarter-turn about B and "
       "halvings towards A and C show MP and MQ equal and perpendicular, "
       "likewise MR and MS (about D); so a quarter-turn about M carries P to Q"
       " and R to S, and PR onto QS.",
       r"PR = QS,\quad PR\perp QS",
       note="Drawn for a convex quadrilateral; the argument holds for any "
            "quadrilateral.",
       pop=32),
    _p("F36", "Squares on a parallelogram", "平行四边形外的四个正方形",
       "The centres of squares erected on the four sides of a parallelogram "
       "form a square: a quarter-turn about each centre carries the triangle "
       "it makes with the next centre and their shared corner onto the "
       "triangle at the neighbouring corner, so neighbouring sides are equal "
       "and perpendicular; a quarter-turn about the parallelogram's centre "
       "then permutes them.",
       r"PQRS\ \mathrm{is\ a\ square}",
       pop=30),
    _p("F37", "Morley's trisector theorem", "莫利三分角定理",
       "Built backwards: with a + b + c = 60°, an equilateral triangle and six"
       " triangles with prescribed angles fit exactly into a triangle with "
       "angles 3a, 3b, 3c, whose trisectors are the pieces' edges and meet at "
       "the equilateral triangle's corners.",
       r"\triangle PQR\ \mathrm{is\ equilateral}",
       note="The assembly argument is Conway's.",
       pop=58),
    _p("F38", "The midpoint triangle", "中点三角形",
       "The three midlines cut the triangle into four congruent copies at half"
       " scale.",
       r"[DEF] = \frac{1}{4}[ABC]",
       pop=55),
    _p("F39", "A triangle in a parallelogram", "平行四边形内的三角形",
       "A triangle on one side with its apex on the opposite side: shear the "
       "apex to a corner — half the parallelogram.",
       r"[T] = \frac{1}{2}[P]", pop=50),
    _p("F40", "Diagonals of a parallelogram bisect each other", "平行四边形对角线互相平分",
       "A half-turn about the midpoint O of AC swaps A and C and sends each "
       "side to the parallel line through the image point, so it maps the "
       "parallelogram onto itself and swaps B and D: O is also the midpoint of"
       " BD.",
       r"AO = OC,\quad BO = OD",
       pop=48),
    _p("F41", "Flank triangles", "侧翼三角形",
       "Squares on the three sides: each triangle between two neighbouring "
       "squares, turned a quarter-turn about their common corner, stands on a "
       "base equal to a side of the original and in line with it, under the "
       "same apex: equal base, equal height.",
       r"[\mathrm{flank}] = [ABC]",
       pop=33),
    _p("F42", "Area of a regular octagon", "正八边形面积",
       "A square of side s(1 + √2) less four corner half-squares of side s/√2,"
       " which, each tipped over by 45°, make one s × s square.",
       r"A = 2\left(1+\sqrt{2}\right)s^2",
       pop=36),
    _p("F43", "Area of a regular dodecagon", "正十二边形面积",
       "Cut the dodecagon of circumradius r and rearrange: three r × r "
       "squares.",
       r"A = 3r^2",
       note="Each third is a 60° rhombus of side r (two equilateral "
            "triangles) with two thin 15°–15°–150° caps; an altitude cut and "
            "a cut along one cap's axis turn it into an r × r square, 15 "
            "pieces in all. Kürschák's tile proves the same result "
            "differently.",
       pop=43),
    _p("F44", "Angle sum by a parallel line", "过顶点作平行线证内角和",
       "Through the apex draw the parallel to the base: the two base angles "
       "reappear there as alternate angles, beside the apex angle.",
       r"A + B + C = 180^\circ",
       note="Euclid I.32 extends a side and uses one alternate and one "
            "corresponding angle; the parallel through the apex with two "
            "alternate angles is the proof ancient sources credit to the "
            "Pythagoreans.",
       pop=75),
    _p("F45", "Intercept theorem", "平行线分线段成比例",
       "A line parallel to one side cuts the other two proportionally: "
       "triangles on the same base between the same parallels have equal "
       "areas.",
       r"DE\parallel BC \;\Rightarrow\; \frac{AD}{DB} = \frac{AE}{EC}",
       note="Euclid VI.2.", pop=65),
    _p("F46", "Bisecting the right angle", "直角平分线过斜边正方形中心",
       "The centre of the square on the hypotenuse sees the hypotenuse at a "
       "right angle, as the right-angle vertex does: both lie on one circle, "
       "and equal chords make the bisector.",
       r"\angle ACO = \angle OCB = 45^\circ",
       note="The square is drawn outward, on the far side of AB from C.",
       pop=25),
    _p("F47", "The square in a right triangle", "直角三角形的内接正方形",
       "The square in the right angle cuts off two triangles similar to the "
       "whole: s/b + s/a = 1.",
       r"s = \frac{ab}{a+b}", pop=30),
    _p("F48", "The trapezoid's butterfly", "梯形的蝴蝶三角形",
       "The diagonals of a trapezoid cut two side triangles of equal area: "
       "each is a triangle on a base with the same height, less the same "
       "piece.",
       r"[AOD] = [BOC]", pop=40),
    _p("F49", "A point inside a parallelogram", "平行四边形内一点",
       "Join any interior point to the four vertices: opposite triangles "
       "together make half the parallelogram.",
       r"[PAB] + [PCD] = \frac{1}{2}[ABCD]", pop=42),
    _p("F50", "Brahmagupta's theorem", "婆罗摩笈多定理",
       "In a cyclic quadrilateral ABCD with perpendicular diagonals crossing "
       "at E, the line through E perpendicular to BC meets AD at its midpoint "
       "M: inscribed angles make AEM and DEM isosceles.",
       r"EM\perp BC \;\Rightarrow\; AM = MD",
       pop=25),
    _p("F51", "Fagnano's problem", "法尼亚诺问题",
       "In an acute triangle, reflect the triangle in the two sides at A: an "
       "inscribed triangle DEF unfolds into a path from D′ to D″ (the images "
       "of D), never shorter than the straight segment D′D″. That segment is "
       "the base of an isosceles triangle with legs AD and apex angle 2A, so "
       "it is shortest when AD is the altitude; the straight path then runs "
       "through the feet of the other two altitudes.",
       r"DE+EF+FD\ \mathrm{minimal}",
       note="Acute triangles only; in an obtuse triangle two feet of the "
            "altitudes fall outside the sides and no inscribed triangle is "
            "shortest.",
       pop=35),
    _p("F52", "Pompeiu's theorem", "庞培定理",
       "For P in the plane of an equilateral triangle, turn by 60° about one "
       "vertex: PA, PB, PC become the sides of a triangle.",
       r"PA,\ PB,\ PC\ \mathrm{form\ a\ triangle}",
       note="The triangle is flat exactly when P lies on the circumcircle "
            "(then the longest of PA, PB, PC is the sum of the other two). "
            "Shown for P inside; the turn works for any P.",
       pop=28),
    _p("F53", "The British flag theorem", "英国国旗定理",
       "Drop perpendiculars from P to the sides of a rectangle, of lengths x₁,"
       " x₂, y₁, y₂: each segment from P to a corner is the diagonal of a "
       "corner box, so its square is the sum of two of x₁², x₂², y₁², y₂²; "
       "each side of the identity uses all four squares once, paired "
       "differently.",
       r"PA^2 + PC^2 = PB^2 + PD^2",
       pop=42),
    _p("F54", "Stewart's theorem", "斯图尔特定理",
       "Pythagoras on both sides of the altitude's foot H, the common height h"
       " eliminated: with D′ the mirror image of the cevian's foot D in H, c² "
       "− d² = BH² − DH² = m·BD′ and b² − d² = CH² − D′H² = n·D′C, and BD′ + "
       "D′C = a.",
       r"b^2m + c^2n = a\left(d^2 + mn\right)",
       note="Drawn with H between D and C; with signed lengths the same "
            "computation covers every position.",
       pop=30),
    _p("F55", "Length of a median", "中线长公式",
       "Double the median into a parallelogram by a half-turn about the "
       "midpoint: the parallelogram law (2m)² + a² = 2b² + 2c², from "
       "Pythagoras on the parallelogram's heights and (c + e)² + (c − e)² = "
       "2c² + 2e² (fold the corners of a square of side c + e inwards).",
       r"m_a^2 = \frac{2b^2+2c^2-a^2}{4}",
       pop=38),
    _p("F56", "Heron's formula", "海伦公式",
       "The incircle and an excircle: A = rs = rₐ(s − a), and similar right "
       "triangles at one vertex give r·rₐ = (s − b)(s − c).",
       r"A = \sqrt{s(s-a)(s-b)(s-c)}",
       pop=50),
    _p("F57", "Viviani for regular polygons", "正多边形的维维亚尼定理",
       "From any interior point, the triangles on the n sides have heights "
       "d₁, …, dₙ and together are the polygon: the sum of distances is "
       "fixed.",
       r"d_1+\cdots+d_n = \frac{2A}{s}", pop=30),
    _p("F58", "The triangle of medians", "中线三角形",
       "Slide the medians into a triangle by completing parallelograms; the "
       "midpoint of a side lies inside it and splits it into three triangles, "
       "each with an equal base on one line and the same apex as one of the "
       "four midline quarters of the original, so it has three quarters of the"
       " original area.",
       r"[\triangle_m] = \frac{3}{4}[ABC]",
       pop=30),
    _p("F59", "Tangent lengths to the incircle", "内切圆的切线长",
       "Equal tangents from each vertex: the three lengths x, y, z satisfy "
       "x + y = c, y + z = a, z + x = b, so x = s − a.",
       r"x = s - a,\quad y = s - b,\quad z = s - c", pop=40),
    _p("F60", "Two squares sharing a corner", "共顶点的两个正方形",
       "Squares ABCD and AEFG: the triangles ABE and ADG, turned a "
       "quarter-turn, have equal bases and heights.",
       r"[ABE] = [ADG]",
       note="ABE and ADG are the two triangles between the squares, whose "
            "angles at A add up to 180°.",
       pop=25),
    _p("F61", "The incentre–excentre lemma", "鸡爪定理",
       "The bisector from A meets the circumcircle at M, the midpoint of the "
       "arc BC: angle chasing makes MB = MI = MC.",
       r"MB = MI = MC",
       note="Uses the inscribed angle theorem (E3).",
       pop=30),
    _p("F62", "Reflections of the orthocentre", "垂心的对称点在外接圆上",
       "Reflect H in a side: the reflection sees that side at the angle 180° −"
       " A, so it lies on the circumcircle.",
       r"H_a \in (ABC)",
       note="Shown for an acute triangle; the result holds for every "
            "triangle.",
       pop=25),
    _p("F63", "Cutting corners at thirds", "切去三分之一角",
       "Cut off each corner of a triangle at the one-third points of the "
       "sides: each corner piece is a ninth, the hexagon left is two "
       "thirds.",
       r"[\mathrm{hexagon}] = \frac{2}{3}[ABC]", pop=30),
    _p("F64", "Every triangle and quadrilateral tiles", "任意三角形与四边形都能铺满平面",
       "Turn a quadrilateral half a turn about the midpoint of each side: the "
       "copies meet with all four angles (360°) around each vertex and repeat "
       "by translation. A triangle and its half-turned copy make a "
       "parallelogram, so triangles tile too.",
       r"\alpha+\beta+\gamma+\delta = 360^\circ",
       note="Drawn for a convex quadrilateral; the same construction tiles "
            "any simple quadrilateral.",
       pop=45),
    _p("F65", "Two triangles make a parallelogram", "两个三角形拼成平行四边形",
       "A copy of any triangle, turned half a turn about a side's midpoint, "
       "completes a parallelogram with the same base and height.",
       r"A = \frac{1}{2}bh", pop=78),
    _p("F66", "Exterior angle bisector theorem", "外角平分线定理",
       "The bisector of the exterior angle at A meets BC extended at E, with "
       "EB : EC = AB : AC (the triangles ABE and ACE have equal heights from E"
       " over AB and AC, and one height from A over EB and EC).",
       r"\frac{EB}{EC} = \frac{AB}{AC}",
       note="Needs AB ≠ AC (otherwise the exterior bisector is parallel to "
            "BC); drawn with AB = 2·AC.",
       pop=30),

    # ---------------------------------------------------------------- G
    _p("G1", "Pythagorean identity", "三角恒等式",
       "The unit circle's right triangle.",
       r"\sin^2\theta + \cos^2\theta = 1", pop=88),
    _p("G2", "Sine of a sum", "正弦和角公式",
       "One rectangle, two overlaid right triangles.",
       r"\sin(A+B) = \sin A\cos B + \cos A\sin B",
       manim="G2_SineAddition", pop=80),
    _p("G3", "Cosine of a sum", "余弦和角公式",
       "The same rectangle, read across instead of up.",
       r"\cos(A+B) = \cos A\cos B - \sin A\sin B",
       note="Drawn for A + B < 90°; the identity holds for all angles.", pop=76),
    _p("G4", "Double angle", "二倍角公式",
       "One triangle with two unit sides (radii) and apex angle 2θ, its area "
       "taken on two different bases.",
       r"\sin 2\theta = 2\sin\theta\cos\theta", pop=67),
    _p("G5", "Law of sines via the circumcircle", "正弦定理",
       "Every side subtends the same circle.",
       r"\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} = 2R", pop=70),
    _p("G6", "Area from two sides and the included angle", "三角形面积公式",
       "Height equals one side times the sine of the angle.",
       r"A = \frac{1}{2}ab\sin C", pop=72),
    _p("G7", "Half-angle tangent", "半角正切",
       "The inscribed angle halves the central angle on the unit circle.",
       r"\tan\frac{\theta}{2} = \frac{\sin\theta}{1+\cos\theta}", pop=47),
    _p("G8", "Sine of a difference", "正弦差角公式",
       "The rectangle of G2 with angle B turned back the other way: the pieces"
       " now overlap, and the overlap is subtracted.",
       r"\sin(A-B) = \sin A\cos B - \cos A\sin B",
       note="Drawn for 0 < B < A < 90°; the identity holds for all angles.",
       pop=62),
    _p("G9", "Cosine of a difference", "余弦差角公式",
       "The rectangle of G3 with angle B turned back: the top side is now the "
       "sum of the two pieces.",
       r"\cos(A-B) = \cos A\cos B + \sin A\sin B",
       note="Drawn for 0 < B < A < 90°; the identity holds for all angles.",
       pop=60),
    _p("G10", "Tangent of a sum", "正切和角公式",
       "A right triangle with leg 1 and angle A, a second one with angle B "
       "standing on its hypotenuse: similar triangles read off tan(A + B).",
       r"\tan(A+B) = \frac{\tan A+\tan B}{1-\tan A\tan B}",
       note="Drawn for A + B < 90°, where tan A·tan B < 1.",
       pop=58),
    _p("G11", "Cosine of a double angle", "余弦二倍角公式",
       "P at angle 2θ on the unit semicircle: the foot of the perpendicular "
       "from P cuts the diameter into 1 + cos 2θ and 1 − cos 2θ. Folding the "
       "isosceles triangles on the two chords gives chords 2cos θ and 2sin θ, "
       "and the right triangle in the semicircle shows the pieces are 2cos²θ "
       "and 2sin²θ.",
       r"\cos 2\theta = 2\cos^2\theta - 1 = 1 - 2\sin^2\theta",
       note="Drawn for 2θ < 90°; the identity holds for all θ.",
       pop=64),
    _p("G12", "Half-angle formulas", "半角公式",
       "On the unit circle with diameter VE and P at angle θ, the inscribed "
       "angle PVE is θ/2. In the right triangle VPE (hypotenuse 2), VP = 2 "
       "cos(θ/2) and PE = 2 sin(θ/2); the foot M of P gives VM = VP·cos(θ/2) ="
       " 1 + cos θ and ME = PE·sin(θ/2) = 1 − cos θ.",
       r"\sin^2\frac{\theta}{2} = \frac{1-\cos\theta}{2},\quad \cos^2\frac{\theta}{2} = \frac{1+\cos\theta}{2}",
       note="Drawn for θ = 70°; the identities hold for all θ.",
       pop=45),
    _p("G13", "Sum to product", "和差化积",
       "The midpoint of the chord between the points at angles A and B of the "
       "unit circle lies at angle (A + B)/2, at distance cos((A − B)/2) from "
       "the centre.",
       r"\sin A+\sin B = 2\sin\frac{A+B}{2}\cos\frac{A-B}{2}",
       note="Drawn with 0 < B < A < π.",
       pop=43),
    _p("G14", "The other Pythagorean identities", "1 + tan²θ = sec²θ",
       "Scale the unit-circle triangle by 1/cos θ: legs 1 and tan θ, "
       "hypotenuse sec θ; by 1/sin θ: legs cot θ and 1, hypotenuse csc θ.",
       r"1+\tan^2\theta = \sec^2\theta,\quad 1+\cot^2\theta = \csc^2\theta", pop=63),
    _p("G15", "cos 36° and the golden ratio", "cos36° 与黄金比",
       "In the 36°–72°–72° triangle, bisecting a base angle cuts off a "
       "similar triangle: the side ratio is φ, and half of it is cos 36°.",
       r"\cos 36^\circ = \frac{\varphi}{2} = \frac{1+\sqrt{5}}{4}", pop=47),
    _p("G16", "Three arctangents make π", "三个反正切之和为π",
       "On a square grid the angles arctan 1, arctan 2 and arctan 3 sit "
       "side by side at one lattice point and fill a straight angle.",
       r"\arctan 1+\arctan 2+\arctan 3 = \pi", pop=56),
    _p("G17", "Two arctangents make π/4", "两个反正切之和为π/4",
       "On a grid, the right triangle with legs 3 and 1 and the right triangle"
       " with legs of two and one grid diagonals share a side at O; with a "
       "half unit square they make a right isosceles triangle, so their angles"
       " at O add to 45°.",
       r"\arctan\frac{1}{2}+\arctan\frac{1}{3} = \frac{\pi}{4}",
       pop=52),
    _p("G18", "Projection formula", "三角形射影公式",
       "The altitude from A meets the line BC at H; the shadows of c and b on "
       "BC are BH = c cos B and HC = b cos C. With B and C acute they tile a; "
       "with B obtuse H falls beyond B and the shadow of c, of length −c cos "
       "B, is subtracted — the same formula.",
       r"a = b\cos C + c\cos B",
       note="Drawn for acute B and C and for obtuse B; an obtuse C is the "
            "mirror case.",
       pop=40),
    _p("G19", "sin x < x < tan x", "sin x < x < tan x",
       "On the unit circle: the triangle inside the sector inside the "
       "tangent triangle, with areas ½ sin x, ½x and ½ tan x.",
       r"\sin x < x < \tan x,\quad 0 < x < \frac{\pi}{2}", pop=66),
    _p("G20", "Jordan's inequality", "若尔当不等式",
       "On the unit circle the arc of angle 2x lies inside the semicircle "
       "drawn on its chord (radius sin x), so it is the shorter: 2x ≤ π sin x,"
       " with equality at x = π/2. On the graph: the sine arch lies above its "
       "chord y = 2x/π.",
       r"\sin x \geq \frac{2x}{\pi},\quad 0 \leq x \leq \frac{\pi}{2}",
       note="Uses the fact that a convex arc lying inside another with the "
            "same ends is the shorter.",
       pop=35),
    _p("G21", "Law of sines by the altitude", "正弦定理（高线证法）",
       "The altitude from C is b·sin A and also a·sin B.",
       r"\frac{a}{\sin A} = \frac{b}{\sin B}",
       note="Drawn with A and B acute; for an obtuse angle the foot falls "
            "outside and sin(180° − A) = sin A gives the same.",
       pop=61),
    _p("G22", "Exact values at 30°, 45° and 60°", "特殊角的三角函数值",
       "Half an equilateral triangle and half a square, with hypotenuse 1.",
       r"\sin 30^\circ = \frac{1}{2},\quad \sin 45^\circ = \frac{\sqrt{2}}{2},\quad \sin 60^\circ = \frac{\sqrt{3}}{2}",
       pop=72),
    _p("G23", "15° and 75°", "15°与75°的三角函数值",
       "Swing the hypotenuse 2 of the 30°–60°–90° triangle (legs 1, √3) about "
       "the 30° corner onto the longer leg's line. Past the corner it ends 2 +"
       " √3 from the right angle and makes an isosceles triangle with base "
       "angles of 15°: tan 75° = 2 + √3. The other way it ends 2 − √3 from the"
       " right angle, and the two swings trace a semicircle, so the angle "
       "opposite that piece is 15° again: tan 15° = 2 − √3.",
       r"\tan 15^\circ = 2-\sqrt{3},\quad \tan 75^\circ = 2+\sqrt{3}",
       pop=39),
    _p("G24", "Regular polygon area from the radius", "正多边形面积（外接圆半径）",
       "n isosceles triangles with sides r and apex angle 2π/n; each has "
       "height r sin(2π/n) on a side r, so area ½ r² sin(2π/n).",
       r"A = \frac{n}{2}r^2\sin\frac{2\pi}{n}",
       note="Drawn for n = 7. For n = 4 the height is the other side; for n ="
            " 3 its foot falls outside, and sin 120° = sin 60° gives the "
            "same.",
       pop=37),
    _p("G25", "The tangent half-angle parametrisation", "万能代换（半角正切参数化）",
       "The line through (−1, 0) with slope t = tan(θ/2) meets the unit circle"
       " again at the point of angle θ. Its foot cuts the diameter into k and "
       "kt², so k + kt² = 2 and k = 2/(1 + t²).",
       r"\cos\theta = \frac{1-t^2}{1+t^2},\quad \sin\theta = \frac{2t}{1+t^2}",
       note="Drawn for 0 < θ < 90° (t < 1).",
       pop=34),
    _p("G26", "arcsin + arccos", "反正弦与反余弦之和",
       "In a right triangle with hypotenuse 1 and a leg x, the two acute "
       "angles are arcsin x and arccos x.",
       r"\arcsin x + \arccos x = \frac{\pi}{2}",
       note="Drawn for 0 < x < 1; for negative x, arcsin(−x) = −arcsin x and "
            "arccos(−x) = π − arccos x give the same sum.",
       pop=41),
    _p("G27", "Sine of a sum by area", "面积证正弦和角公式",
       "A triangle whose altitude 1 splits the apex angle into A and B: its "
       "area is ½·sec A·sec B·sin(A + B), and also ½(tan A + tan B).",
       r"\sin(A+B) = \sin A\cos B + \cos A\sin B",
       note="Drawn with A + B < 90°, so the foot of the height lies on the "
            "base.",
       pop=50),
    _p("G28", "Quadrilateral area from its diagonals", "对角线求四边形面积",
       "Lines through the vertices parallel to the diagonals make a "
       "parallelogram with sides d₁, d₂ and angle θ. The diagonals cut it into"
       " four cells, each halved by a side of the quadrilateral, so the "
       "quadrilateral is half the parallelogram: ½ d₁ · d₂ sin θ.",
       r"A = \frac{1}{2}d_1d_2\sin\theta",
       note="Drawn for a convex quadrilateral; the formula also holds for a "
            "non-convex one, which the picture does not cover.",
       pop=35),
    _p("G29", "Double angle in a semicircle", "半圆中的二倍角",
       "The right triangle in a semicircle of radius 1 with angle θ at the end"
       " of the diameter: the central angle is 2θ, and the height from the "
       "third vertex is read two ways.",
       r"\sin 2\theta = 2\sin\theta\cos\theta",
       note="Drawn with θ < 45°; above 45° the foot lies between the end of "
            "the diameter and the centre and the angle at the centre is 180° "
            "− 2θ, with the same sine.",
       pop=35),
    _p("G30", "sin x + cos x is at most √2", "sin x + cos x ≤ √2",
       "The point (cos x, sin x) of the unit circle lies below the tangent "
       "line X + Y = √2 that touches the circle at 45°.",
       r"\sin x + \cos x \leq \sqrt{2}", pop=30),
    _p("G31", "Sides, area and circumradius", "abc = 4RA",
       "Slide the vertex round its arc to the far end of the diameter through "
       "B: a = 2R·sin α. The height from B is c·sin α, so the area is A = "
       "½·bc·sin α: eliminate sin α.",
       r"abc = 4RA",
       note="Drawn with an acute triangle; α is the angle at the vertex "
            "opposite a.",
       pop=35),

    # ---------------------------------------------------------------- H
    _p("H1", "Derivative of sine", "正弦的导数",
       "An infinitesimal arc on the unit circle, resolved into components.",
       r"\frac{d}{d\theta}\sin\theta = \cos\theta", pop=62),
    _p("H2", "Riemann sum", "黎曼和",
       "Rectangles closing on the region beneath the curve.",
       r"\int_a^b f = \lim_{n\to\infty}\sum_i f(x_i)\Delta x", pop=74),
    _p("H3", "Integral of x", "x 的积分",
       "The region is a triangle.",
       r"\int_0^a x\,dx = \frac{a^2}{2}", pop=66),
    _p("H4", "Integral of x squared", "x^2 的积分",
       "Doubling the interval makes the region eight times larger, yet it is "
       "two copies of itself plus two a³ pieces.",
       r"\int_0^a x^2\,dx = \frac{a^3}{3}", pop=63),
    _p("H5", "Area under 1/x", "1/x 下的面积",
       "Stretching across by a and squeezing up by 1/a keeps the curve "
       "y = 1/x and every area.",
       r"\int_1^{ab}\frac{dx}{x} = \int_1^{a}\frac{dx}{x} + \int_1^{b}\frac{dx}{x}", pop=54),
    _p("H6", "Integration by parts", "分部积分",
       "Two areas that together fill a rectangle less its corner.",
       r"\int_{v_1}^{v_2} u\,dv + \int_{u_1}^{u_2} v\,du = u_2v_2 - u_1v_1", pop=61),
    _p("H7", "Derivative of cosine", "余弦的导数",
       "The same small triangle as for sine, read across: its horizontal leg "
       "is the drop of cos θ, which tends to sin θ·dθ.",
       r"\frac{d}{d\theta}\cos\theta = -\sin\theta",
       pop=55),
    _p("H8", "Derivative of x²", "x² 的导数",
       "Grow the square by dx: two strips x·dx and a corner dx² that "
       "vanishes in the limit.",
       r"\frac{d}{dx}x^2 = 2x", pop=72),
    _p("H9", "Derivative of x³", "x³ 的导数",
       "Grow the cube by dx: three slabs x²·dx, three rods x·dx² and a corner "
       "dx³; divided by dx, the rods and the corner vanish in the limit.",
       r"\frac{d}{dx}x^3 = 3x^2",
       pop=60),
    _p("H10", "Product rule", "乘积法则",
       "A u × v rectangle grows by two strips, u·dv and v·du, and a corner "
       "du·dv that vanishes.",
       r"d(uv) = u\,dv + v\,du", pop=70),
    _p("H11", "Derivative of the square root", "√x 的导数",
       "A square of area x grows by dx: two strips √x·Δ√x and a corner (Δ√x)²,"
       " which vanishes in the limit, make up the dx.",
       r"\frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}}",
       pop=40),
    _p("H12", "Derivative of 1/x", "1/x 的导数",
       "Rectangles of area 1 under y = 1/x: widen by dx and, to keep the area,"
       " the height must drop by dx/(x(x + dx)), which tends to dx/x².",
       r"\frac{d}{dx}\frac{1}{x} = -\frac{1}{x^2}",
       pop=38),
    _p("H13", "Fundamental theorem of calculus", "微积分基本定理",
       "The area under f from a to x grows by a strip of width dx and height "
       "f(x).",
       r"\frac{d}{dx}\int_a^x f(t)\,dt = f(x)",
       note="Drawn with f increasing near x; in general the strip lies "
            "between the rectangles of the least and greatest values of f on "
            "[x, x + dx].",
       pop=78),
    _p("H14", "Mean value theorem", "拉格朗日中值定理",
       "Slide the chord parallel to itself: the last position still touching "
       "the curve is a tangent with the chord's slope.",
       r"f'(c) = \frac{f(b)-f(a)}{b-a}",
       note="Drawn with the arc above the chord; otherwise the line slides "
            "the other way.",
       pop=68),
    _p("H15", "sin x over x tends to 1", "重要极限 sin x / x → 1",
       "The sector lies between the inner triangle and the tangent "
       "triangle: cos x < sin x / x < 1.",
       r"\lim_{x\to 0}\frac{\sin x}{x} = 1", pop=67),
    _p("H16", "Integral of sin²", "sin² 的积分",
       "Over [0, π/2] the region under cos² is the mirror image of the region "
       "under sin² in x = π/4; turned over in y = ½ it fills exactly the rest "
       "of the 1 × π/2 rectangle above sin², since sin² + cos² = 1. Two copies"
       " fill the rectangle.",
       r"\int_0^{\pi/2}\sin^2x\,dx = \frac{\pi}{4}",
       pop=48),
    _p("H17", "Integral of ln x", "ln x 的积分",
       "The region under ln x from 1 to a and the region to the left of it "
       "(under eʸ, turned) fill an a × ln a rectangle.",
       r"\int_1^a\ln x\,dx = a\ln a - a + 1",
       note="Uses ∫₀ˢ eʸ dy = eˢ − 1: the turned region has area a − 1. Drawn"
            " for a > 1.",
       pop=42),
    _p("H18", "xⁿ and its inverse fill the square", "xⁿ 与 x^(1/n) 填满单位正方形",
       "The region under y = xⁿ on [0, 1] and the region to the left of it, "
       "which is the area under y = x^(1/n) turned, fill the unit square.",
       r"\int_0^1x^n\,dx + \int_0^1x^{1/n}\,dx = 1",
       note="Drawn for n = 3; a closing sweep varies n.",
       pop=45),
    _p("H19", "Cavalieri's principle in the plane", "平面上的卡瓦列里原理",
       "Slide every horizontal strip of a region sideways: the area does "
       "not change.",
       r"A_1 = A_2", pop=53),
    _p("H20", "A disc grows by its circumference", "圆面积的导数是周长",
       "Grow the radius by dr: the disc gains a ring; cut and unrolled, it is "
       "a trapezoid between 2πr and 2π(r + dr) — a rectangle 2πr·dr plus "
       "corners π·dr² that vanish.",
       r"\frac{dA}{dr} = 2\pi r",
       pop=61),
    _p("H21", "A ball grows by its surface", "球体积的导数是表面积",
       "Grow the radius by dr: the new shell splits into thin tiles, each "
       "standing on a patch of the surface; a tile's volume lies between its "
       "inner face times dr and its outer face times dr, so 4πr²·dr < ΔV < "
       "4π(r + dr)²·dr and dV/dr = 4πr².",
       r"\frac{dV}{dr} = 4\pi r^2",
       note="Takes S = 4πr² as known (I4).",
       pop=49),
    _p("H22", "Integrals over symmetric intervals", "对称区间上的积分",
       "An odd function's areas on either side of 0 are half-turns of each "
       "other and cancel.",
       r"\int_{-a}^{a}f(x)\,dx = 0\quad(f\ \mathrm{odd})",
       note="The half-turn is shown as its two folds, x → −x, then y → −y.",
       pop=44),
    _p("H23", "Mean value theorem for integrals", "积分中值定理",
       "The region under f levels into a rectangle of the same area; its "
       "height is a value f takes between its lowest and highest.",
       r"\int_a^b f(x)\,dx = f(c)\,(b-a)",
       note="Needs f continuous. The level line may meet the graph more than "
            "once; every meeting point is a valid c.",
       pop=46),
    _p("H24", "Area under a cycloid", "摆线下的面积",
       "The companion curve splits the half-arch: one part has half the "
       "rolling disc's area by Cavalieri, the other half of the πr × 2r "
       "rectangle.",
       r"A = 3\pi r^2",
       note="Roberval's argument (17th century).", pop=41),
    _p("H25", "Derivative of the inverse function", "反函数的导数",
       "Reflect the graph and its tangent in y = x: rise and run swap, so the "
       "slope becomes its reciprocal.",
       r"\left(f^{-1}\right)'(y) = \frac{1}{f'(x)}",
       note="Drawn with f increasing; the reflected slope 1/f′(x) needs f′(x)"
            " ≠ 0.",
       pop=47),
    _p("H26", "Substitution stretches the strips", "换元积分：拉伸细条",
       "Strips of width du under f(g(u)), moved sideways by u ↦ g(u), keep "
       "their heights and tile the region under f with widths dx ≈ g′(u)·du; "
       "squeezed back to width du with heights × dx/du (areas kept), they "
       "stand under f(g(u))·g′(u).",
       r"\int_{g(a)}^{g(b)}f(x)\,dx = \int_a^b f(g(u))\,g'(u)\,du",
       note="Drawn for increasing g; for decreasing g the limits swap and the"
            " areas are signed.",
       pop=33),
    _p("H27", "The exponential's constant subtangent", "指数曲线的次切距恒为1",
       "At x = 0 the graph of eˣ has slope 1: its tangent meets the x-axis one"
       " unit to the left. Stretched vertically by eᵃ the graph becomes itself"
       " moved left by a (eᵃ·eˣ = eˣ⁺ᵃ), and the stretch keeps the tangent's "
       "foot; slid back right by a, the tangent at (a, eᵃ) meets the axis one "
       "unit to the left: slope eᵃ/1 = eᵃ.",
       r"\frac{d}{dx}e^x = e^x",
       note="e is taken as the base whose graph has slope 1 at x = 0.",
       pop=35),
    _p("H28", "The shadow of a semicircle", "半圆的投影：∫sin = 2",
       "Walk round the unit semicircle: each step dθ moves sideways by "
       "sin θ·dθ, and the sideways moves add up to the diameter.",
       r"\int_0^{\pi}\sin\theta\,d\theta = 2", pop=40),
    _p("H29", "Cosine above a parabola", "1 − x²/2 ≤ cos x",
       "Integrate sin t ≤ t from 0 to x: the area under the sine arch is below"
       " the triangle under the line.",
       r"1 - \cos x \leq \frac{x^2}{2}",
       note="Drawn for 0 < x ≤ π; cos is even, and for |x| ≥ 2 the bound is "
            "trivial. Uses ∫ sin t dt = 1 − cos x over [0, x]; sin t ≤ t is "
            "shown on the unit circle (drop ≤ arc).",
       pop=30),
    _p("H30", "Derivative of tan", "正切的导数",
       "On the tangent line x = 1, T = (1, tan θ) is sec θ from O. Turning the"
       " ray by dθ sweeps an arc sec θ·dθ, perpendicular to the ray and at the"
       " angle θ to the line, so T moves up the line by sec θ·dθ / cos θ → "
       "sec²θ·dθ.",
       r"\frac{d}{d\theta}\tan\theta = \sec^2\theta",
       pop=38),
    _p("H31", "The arc-length element", "弧长微元",
       "A short piece of a curve is nearly the hypotenuse of a right triangle "
       "with legs dx and dy.",
       r"ds = \sqrt{dx^2 + dy^2}",
       note="Drawn on a convex piece of curve.",
       pop=40),
    _p("H32", "Derivative of arctan", "反正切的导数",
       "P = (x, 1) is √(1 + x²) from O and OP makes the angle θ = arctan x "
       "with the y-axis. A step dx along y = 1 slants at θ to the circle "
       "through P, so its part along the circle is dx/√(1 + x²) (the small "
       "triangle at P is OBP shrunk by dx/√(1 + x²) and turned over); at "
       "radius √(1 + x²) that arc subtends dθ = dx/(1 + x²).",
       r"\frac{d}{dx}\arctan x = \frac{1}{1+x^2}",
       note="Drawn for x > 0; the fit is exact only as dx → 0, shown in a "
            "window magnified by 1/dx.",
       pop=36),

    # ---------------------------------------------------------------- I
    _p("I1", "Frustum of a square pyramid", "方亭（四棱台）体积",
       "A frustum splits into a central block, four side wedges and four "
       "corner pyramids.",
       r"V = \frac{h}{3}\left(a^2 + ab + b^2\right)",
       note="Block b²h, wedges 4·½·b·(a−b)/2·h, corner pyramids "
            "4·((a−b)/2)²·h/3: together h(a² + ab + b²)/3. The rule is in "
            "the Moscow Mathematical Papyrus (c. 1850 BCE; its worked case "
            "h = 6, a = 4, b = 2 gives 56) and in the 九章算术 as 方亭术.", pop=58),
    _p("I2", "Cone is a third of its cylinder", "圆锥体积",
       "Cavalieri against a pyramid of equal base area and height.",
       r"V = \frac{1}{3}\pi r^2 h",
       note="'Pour three cones into the cylinder' shows the result without "
            "proving it. The rigorous route is Cavalieri: a cone and a pyramid "
            "of equal base area and height have equal cross-sections at every "
            "height, and I7 gives the pyramid's ⅓. Not via I3 — that argument "
            "uses the cone's volume, so citing it here would be circular. "
            "The animation takes h = s, so the pyramid is one of the three "
            "in I7's cube.", pop=79),
    _p("I3", "Sphere by Cavalieri", "球体积（祖暅原理）",
       "At every height, disc = annulus: sphere = cylinder minus double cone.",
       r"V = \pi r^2\cdot 2r - 2\cdot\frac{1}{3}\pi r^2 r = \frac{4}{3}\pi r^3",
       vpy="I3",
       note="The same principle Zu Geng (祖暅) stated in the 5th century: "
            "solids of equal cross-section at every height have equal volume.", pop=75),
    _p("I4", "Archimedes' hat-box", "球面积",
       "Projection straight out from the axis onto the enclosing cylinder "
       "preserves area.",
       r"S = 4\pi r^2", pop=64),
    _p("I5", "Space diagonal of a box", "长方体对角线",
       "Pythagoras applied twice, in perpendicular planes.",
       r"d^2 = a^2 + b^2 + c^2",
       vpy="I5", pop=66),
    _p("I6", "de Gua's theorem", "德古阿定理",
       "The three-dimensional Pythagoras, on a corner tetrahedron.",
       r"A_1^2 + A_2^2 + A_3^2 = A_4^2",
       vpy="I6", pop=40),
    _p("I7", "Pyramid is a third of its prism", "棱锥体积",
       "A cube dissects into three congruent oblique square pyramids.",
       r"V = \frac{1}{3}Bh",
       vpy="I7",
       note="The three pyramids share an apex at one corner of the cube; "
            "their bases are the three faces not containing that corner.", pop=77),
    _p("I8", "Steinmetz bicylinder", "牟合方盖",
       "Every level of the common part of two perpendicular cylinders is a "
       "square around the inscribed sphere's circle: 4 : π.",
       r"V = \frac{16}{3}r^3",
       note="Liu Hui's 牟合方盖, completed by Zu Chongzhi and Zu Geng.", pop=46),
    _p("I9", "The napkin ring", "餐巾环问题",
       "Volume depends only on the band height, not on the sphere's radius.",
       r"V = \frac{\pi h^3}{6}", pop=50),
    _p("I10", "Euler's polyhedron formula", "欧拉多面体公式",
       "Flatten the polyhedron and collapse the graph.",
       r"V - E + F = 2",
       note="Holds for convex polyhedra (any polyhedron whose surface is a "
            "topological sphere); a polyhedral torus gives V − E + F = 0.", pop=81),
    _p("I11", "Lateral area of a cone", "圆锥的侧面积",
       "Unroll the cone: a sector of radius l and arc 2πr, area ½·l·2πr.",
       r"S = \pi r l", pop=60),
    _p("I12", "Cavalieri's stack of coins", "卡瓦列里原理：一叠硬币",
       "Push a stack of coins sideways: every layer keeps its area, so the "
       "slanted stack keeps the volume Bh.",
       r"V = Bh", pop=62),
    _p("I13", "A cube as six pyramids", "正方体分成六个四棱锥",
       "From the centre, the six faces span six congruent pyramids of height "
       "s/2: each is s³/6 = ⅓·s²·(s/2).",
       r"V = \frac{1}{3}Bh",
       note="Shows ⅓Bh for this one square pyramid (apex over the centre of "
            "the base, height half its edge).",
       pop=50),
    _p("I14", "A prism as three tetrahedra", "三棱柱分成三个四面体",
       "Two cuts split a triangular prism into three tetrahedra; pairs share "
       "equal bases and heights, so all three are equal.",
       r"V = \frac{1}{3}Bh",
       note="Euclid XII.7.", pop=45),
    _p("I15", "A ball as cones from its centre", "球是以球心为顶点的锥体之和",
       "Wrap the ball in a polyhedron whose faces all touch it: from the "
       "centre each face is the base of a pyramid of height r, so the "
       "polyhedron is exactly ⅓·r·(its surface). With more and smaller faces "
       "it closes in on the ball: V = ⅓·r·S.",
       r"V = \frac{1}{3}rS = \frac{4}{3}\pi r^3",
       note="Takes S = 4πr² as known (I4).",
       pop=52),
    _p("I16", "Volume of a torus", "圆环体体积",
       "Cut the torus into thin wedges and lay them in a row, every other one "
       "turned round: each wedge's outer side is longer than its middle by as "
       "much as its inner side is shorter, so long and short sides alternate "
       "and the row straightens into a cylinder of base πr² and length 2πR.",
       r"V = 2\pi R\cdot\pi r^2",
       note="Pappus's centroid theorem.",
       pop=51),
    _p("I17", "A regular tetrahedron in a cube", "正方体中的正四面体",
       "Cut four corner tetrahedra, each ⅙ of the cube, off a cube: a "
       "regular tetrahedron of ⅓ of the cube remains.",
       r"V = a^3 - 4\cdot\frac{a^3}{6} = \frac{1}{3}a^3", pop=44),
    _p("I18", "Only five Platonic solids", "只有五种正多面体",
       "At a vertex at least three regular faces meet, with angles adding to "
       "less than 360°: three, four or five triangles, three squares, or three"
       " pentagons; six triangles, four squares or three hexagons already lie "
       "flat.",
       r"\frac{1}{m}+\frac{1}{n} > \frac{1}{2}",
       note="Shows that there are at most five kinds of corner and that each "
            "closes into a solid.",
       pop=63),
    _p("I19", "Girard's theorem", "吉拉尔定理（球面三角形）",
       "Three lunes through the triangle's angles, with their antipodes, "
       "cover the sphere once and the triangle and its antipode three times.",
       r"A = (\alpha+\beta+\gamma-\pi)\,r^2", pop=40),
    _p("I20", "The rhombic dodecahedron", "菱形十二面体",
       "The six pyramids from the centre of a cube, folded outward onto the "
       "faces of a second cube: twice the cube's volume.",
       r"V = 2a^3",
       note="The two triangles over each edge of the cube are coplanar (45° +"
            " 90° + 45° = 180°): the faces are 12 rhombi.",
       pop=30),
    _p("I21", "Descartes' angle defect", "笛卡尔角亏定理",
       "The angles missing at the vertices of a convex polyhedron add to 720°:"
       " 360°·V less the face angles 180°·(2E − 2F), which is 360°·2 by "
       "Euler's formula.",
       r"\sum_v\delta_v = 720^\circ",
       note="Shows one solid's gaps filling two full turns, then the count "
            "for any convex polyhedron.",
       pop=33),
    _p("I22", "An octahedron in a cube", "正方体中的正八面体",
       "The face centres of a cube span an octahedron: two square pyramids "
       "of height a/2 on a base of half the face, ⅙ of the cube.",
       r"V = \frac{1}{6}a^3", pop=32),
    _p("I23", "Surface of a cylinder", "圆柱的表面积",
       "Unroll the side into a 2πr × h rectangle and add the two discs.",
       r"S = 2\pi rh + 2\pi r^2", pop=55),
    _p("I24", "A slanted slice of a cylinder", "斜截圆柱得椭圆",
       "Two balls in the cylinder touch the slanting plane at F₁, F₂ and the "
       "cylinder along two circles. For a point P of the cut, PF₁ and PQ₁ (Q₁ "
       "where the vertical line through P meets the upper circle) are tangents"
       " from P to one ball, so they are equal; likewise PF₂ = PQ₂. So PF₁ + "
       "PF₂ = Q₁Q₂, the same for every P: an ellipse.",
       r"PF_1 + PF_2 = 2a",
       pop=40),
    _p("I25", "Why a football has twelve pentagons", "足球为何有十二个五边形",
       "Pentagons and hexagons with three at every vertex: count edges and "
       "vertices from the faces, and Euler's formula leaves exactly twelve "
       "pentagons.",
       r"V - E + F = 2 \;\Rightarrow\; p = 12",
       note="Also shows balls with 0 and 30 hexagons.",
       pop=40),
    _p("I26", "The Menger sponge has no volume", "门格海绵体积为零",
       "Each step keeps 20 of the 27 sub-cubes: the volume left is "
       "(20/27)ⁿ, which tends to 0.",
       r"\lim_{n\to\infty}\left(\frac{20}{27}\right)^n = 0", pop=30),
    _p("I27", "The stepped pyramid", "阶梯金字塔逼近棱锥",
       "A pyramid of n square slabs has volume (1² + 2² + … + n²)/n³ of its "
       "box; as the slabs thin out this tends to ⅓.",
       r"\frac{1^2+2^2+\cdots+n^2}{n^3} \to \frac{1}{3}",
       note="Squeezed: ⅓ ≤ (1² + … + n²)/n³ ≤ ⅓ + 1/n; the rims of the slabs "
            "make one layer. Takes the ⅓ from I7.",
       pop=40),

    # ---------------------------------------------------------------- J
    _p("J1", "Tennenbaum: root 2 is irrational", "根号2 无理性",
       "The smallest such square always contains a smaller one.",
       r"\sqrt{2} \notin \mathbb{Q}", pop=73),
    _p("J2", "The golden rectangle", "黄金矩形",
       "Remove a square; what remains is the same rectangle again.",
       r"\varphi^2 = \varphi + 1,\quad \varphi = \frac{1+\sqrt{5}}{2}",
       manim="J2_GoldenRectangle", pop=78),
    _p("J3", "Continued fraction for phi", "黄金比连分数",
       "The self-similarity of the golden rectangle, written out.",
       r"\varphi = 1 + \frac{1}{1 + \frac{1}{1 + \cdots}}", pop=49),
    _p("J4", "Spiral of Theodorus", "泰奥多罗斯螺线",
       "Chained right triangles with legs √n and 1 produce every square "
       "root.",
       r"h_n = \sqrt{n}", pop=60),
    _p("J5", "Handshakes", "握手问题",
       "An n by n grid, minus the diagonal, halved.",
       r"\binom{n}{2} = \frac{n(n-1)}{2}", pop=57),
    _p("J6", "Vandermonde's identity", "范德蒙德恒等式",
       "Choose r from two piles, split by how many came from the first.",
       r"\sum_k \binom{m}{k}\binom{n}{r-k} = \binom{m+n}{r}", pop=38),
    _p("J7", "The pizza theorem", "披萨定理",
       "Four cuts at 45 degrees through any interior point: alternate slices "
       "have equal total area.",
       r"A_{odd} = A_{even}",
       note="The animation cuts the pieces for the point drawn: its last "
            "step, splitting the central parallelogram into two copies of the "
            "leftover pieces, uses that point's proportions.", pop=44),
    _p("J8", "Curry's missing square", "消失的方格悖论",
       "A caution: the 'triangle' is not a triangle. Pictures need proofs.",
       r"\frac{3}{8} \ne \frac{2}{5}",
       note="Included deliberately. A proof without words is a proof only "
            "when the picture's implicit claims are themselves checked; "
            "here the hypotenuse is bent and the eye does not see it.", pop=68),
    _p("J9", "Wallace-Bolyai-Gerwien", "分割等价定理",
       "Any two polygons of equal area are cut-and-paste equivalent.",
       r"[P] = [Q] \Rightarrow P \sim Q",
       note="The animation cuts one triangle into a square of equal area. "
            "Every polygon is a union of triangles, and two squares combine "
            "into one by the Pythagorean dissection (A4), so each polygon "
            "can be cut into a single square; two polygons of equal area "
            "meet at the same square.", pop=43),

    _p("J10", "Endless Pythagorean triples", "无穷多组勾股数",
       "Every odd square is an L of width one between two consecutive "
       "squares: (2k+1)² turns the square of side 2k² + 2k into the next "
       "one.",
       r"(2k+1)^2 + \left(2k^2+2k\right)^2 = \left(2k^2+2k+1\right)^2", pop=52),
    _p("J11", "Root 3 is irrational", "√3 是无理数",
       "If an equilateral triangle of integer side had the area of three equal"
       " ones of integer side, putting those in its corners would leave one "
       "uncovered triangle equal in area to the three overlaps: a smaller "
       "example, without end.",
       r"\sqrt{3}\notin\mathbb{Q}",
       pop=46),
    _p("J12", "The golden ratio is irrational", "黄金比是无理数",
       "A golden rectangle with integer sides would, after cutting off a "
       "square, leave a smaller golden rectangle with integer sides — and "
       "so on for ever.",
       r"\varphi\notin\mathbb{Q}", pop=38),
    _p("J13", "Fermat's little theorem by necklaces", "项链证费马小定理",
       "Strings of p beads in a colours, less the a one-colour ones, fall into"
       " groups of p rotations each when p is prime.",
       r"p\mid a^p - a",
       note="Shown for a = 2, p = 5.",
       pop=50),
    _p("J14", "Zagier's windmills", "扎吉尔的风车",
       "Draw x² + 4yz = p as a square with four y × z arms. Rebuilding each "
       "windmill's outline around the other square it admits (turned over if "
       "need be) pairs them off but one, so their number is odd, and swapping "
       "the arms' sides must fix one: a square plus four square arms.",
       r"p = 4k+1 \;\Rightarrow\; p = x^2 + (2y)^2",
       note="Shown for p = 13. When y − z < x < 2y the rebuilt windmill turns"
            " the other way; read as (x, y, z) it is still a solution.",
       pop=49),
    _p("J15", "Euclid's algorithm", "辗转相除法",
       "Cut squares off an a × b rectangle until nothing is left. Each cut "
       "keeps the common measures of the two sides; the last square's side "
       "measures both sides — it is gcd(a, b).",
       r"\gcd(a,b) = \gcd(b,\,a-b)",
       pop=62),
    _p("J16", "The pentagon's golden ratio", "正五边形中的黄金比",
       "Two diagonals and a side make similar triangles: diagonal : side = "
       "side : (diagonal − side).",
       r"\frac{d}{s} = \varphi", pop=55),
    _p("J17", "Squaring numbers ending in 5", "尾数为5的数的平方",
       "Cut the (10a + 5)-square into a 10a-square, two 10a × 5 strips and a 5"
       " × 5 corner; turn one strip a quarter and lay it beside the other: a "
       "10a × 10(a + 1) rectangle of a(a + 1) blocks of 100, and the corner "
       "25.",
       r"(10a+5)^2 = 100\,a(a+1) + 25",
       note="Drawn for a = 2 (25²).",
       pop=37),
    _p("J18", "The staircase paradox", "阶梯悖论",
       "A staircase on the diagonal of the unit square always has length 2, "
       "however fine, though it closes in on a diagonal of length √2.",
       r"2 \neq \sqrt{2}",
       note="Included deliberately, like J8: length does not pass to the "
            "limit of shapes.", pop=50),
    _p("J19", "The fractions are countable", "有理数可数",
       "Lay the fractions p/q on a grid and walk it along its diagonals: "
       "every fraction is reached, so they can be numbered.",
       r"|\mathbb{Q}| = |\mathbb{N}|", pop=54),
    _p("J20", "Triples from rational points", "有理点与勾股数",
       "Rational slopes through (−1, 0) give rational points of the unit "
       "circle; clearing denominators gives the triples.",
       r"\left(m^2-n^2\right)^2 + (2mn)^2 = \left(m^2+n^2\right)^2",
       note="Drawn for slope 1/2 (3, 4, 5).",
       pop=40),
    _p("J21", "Lattice points on a segment", "线段上的格点",
       "The segment from (0, 0) to (a, b) passes, between its ends, through "
       "gcd(a, b) − 1 lattice points, equally spaced.",
       r"N = \gcd(a,b) - 1",
       note="Drawn for (12, 8).",
       pop=28),
    _p("J22", "Dudeney's hinged dissection", "杜德尼铰接剖分",
       "Four hinged pieces swing an equilateral triangle into a square of "
       "the same area.",
       r"\frac{\sqrt{3}}{4}s^2 = x^2", pop=45),
    _p("J23", "Primes beyond 3", "大于3的素数",
       "Write the numbers in six columns: four of the columns hold only "
       "multiples of 2 or 3, so every prime above 3 is 6k ± 1.",
       r"p > 3 \;\Rightarrow\; p = 6k\pm1",
       note="The columns 6k ± 1 also hold composites (25, 35, 49).",
       pop=36),
    _p("J24", "Only squares have an odd number of divisors", "只有平方数有奇数个约数",
       "Divisors pair up as the sides of rectangles of area n; only a square "
       "has a rectangle paired with itself.",
       r"d(n)\ \mathrm{odd} \;\Leftrightarrow\; n = m^2",
       note="Drawn for n = 12 and n = 16.",
       pop=36),
    _p("J25", "The silver rectangle", "白银矩形",
       "Cut two squares off a 1 × (1 + √2) rectangle: the rest is the same "
       "shape again, so 1 + √2 = 2 + 1/(1 + √2).",
       r"\sqrt{2} = 1+\frac{1}{2+\frac{1}{2+\cdots}}",
       note="Both diagonals make 22.5° with their long sides, so the rest is "
            "the rectangle shrunk by √2 − 1 and turned a quarter-turn.",
       pop=30),
    _p("J26", "Every triangle is isosceles?", "所有三角形都是等腰三角形？",
       "A famous false proof: the bisector of A and the perpendicular bisector"
       " of BC are drawn meeting inside the triangle — in truth they meet on "
       "the circumcircle, outside, where one foot of the perpendiculars falls "
       "beyond its side: AB = AF − FB while AC = AE + EC.",
       r"AB \neq AC",
       note="Included deliberately, like J8: the figure, not the logic, is "
            "wrong.",
       pop=40),
    _p("J27", "64 = 65", "64 = 65 悖论",
       "An 8 × 8 square cut into Fibonacci pieces re-forms a 5 × 13 "
       "rectangle — with a thin sliver of area one along its diagonal.",
       r"8^2 = 5\cdot13 - 1",
       note="Included deliberately; the sliver is Cassini's identity (B36).",
       pop=45),
    _p("J28", "Chinese remainder theorem on a grid", "中国剩余定理（网格）",
       "Number the cells of an m × n grid 0, 1, 2, … along a diagonal, "
       "wrapping round: when gcd(m, n) = 1 every cell is reached once.",
       r"\gcd(m,n) = 1 \;\Rightarrow\; \mathbb{Z}_{mn} \cong \mathbb{Z}_m\times\mathbb{Z}_n",
       note="Drawn for 3 × 5, with 4 × 6 as the counter-example.",
       pop=35),
    _p("J29", "Totients add up to n", "欧拉函数之和",
       "Write k/n for k = 1, …, n in lowest terms and sort by denominator: "
       "each divisor d of n takes exactly φ(d) of them.",
       r"\sum_{d\mid n}\varphi(d) = n",
       note="Drawn for n = 12.",
       pop=25),
    _p("J30", "Farey neighbours", "法里邻项",
       "Draw a/b as the lattice point (b, a). Neighbours a/b < c/d in a Farey "
       "sequence span, with O, a lattice triangle with no other lattice "
       "points, so by Pick its area is ½ — and it is half the parallelogram on"
       " (b, a) and (d, c), whose area is bc − ad.",
       r"bc - ad = 1",
       note="Uses Pick's theorem (F11) and the determinant as area (L3). "
            "Drawn for 1/3 < 1/2 in F₄; every pair of neighbours in F₄ gives "
            "such a triangle.",
       pop=22),
    _p("J31", "Parity by pairing dots", "奇偶性的点阵证明",
       "Odd numbers are pairs with one dot over: two odd numbers' spare dots "
       "pair up, an odd times an odd keeps one.",
       r"\mathrm{odd}+\mathrm{odd} = \mathrm{even},\quad \mathrm{odd}\times\mathrm{odd} = \mathrm{odd}",
       pop=45),
    _p("J32", "Doubling the square", "倍积正方形",
       "The square on the diagonal is four half-squares, the original two: "
       "double the area.",
       r"d^2 = 2s^2",
       note="Socrates' lesson in Plato's Meno.", pop=60),
    _p("J33", "Casting out nines", "弃九法",
       "Each block of 10, 100, 1000, … is a multiple of 9 plus one unit: a "
       "number and its digit sum leave the same remainder on division by 9.",
       r"n \equiv \mathrm{digit\ sum}(n)\ \ (\mathrm{mod}\ 9)", pop=35),

    # ---------------------------------------------------------------- K
    _p("K1", "Counting lattice paths", "格路计数",
       "A path of m steps east and n steps north is a choice of which m of "
       "its m + n steps go east.",
       r"N = \binom{m+n}{m}", pop=65),
    _p("K2", "Pascal's rule by paths", "帕斯卡法则（格路证法）",
       "Every path to a lattice point arrives either from the west or from "
       "the south.",
       r"\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}", pop=68),
    _p("K3", "Symmetry of binomial coefficients", "组合数的对称性",
       "Reflect a path in the diagonal: east and north steps swap.",
       r"\binom{n}{k} = \binom{n}{n-k}", pop=55),
    _p("K4", "Sum of squares of a row", "组合数的平方和",
       "Paths across an n × n grid, sorted by where they cross the middle "
       "anti-diagonal: C(n, k) ways in and C(n, k) ways out.",
       r"\sum_{k=0}^{n}\binom{n}{k}^2 = \binom{2n}{n}", pop=45),
    _p("K5", "Catalan numbers by reflection", "卡特兰数（反射法）",
       "A path that rises above the diagonal touches the line one step above "
       "it; reflect the rest of the path in that line from its first touch on:"
       " the bad paths match all paths to (n − 1, n + 1).",
       r"C_n = \binom{2n}{n} - \binom{2n}{n+1} = \frac{1}{n+1}\binom{2n}{n}",
       pop=50),
    _p("K6", "Domino tilings are Fibonacci numbers", "多米诺骨牌铺法与斐波那契数",
       "A 2 × n strip starts with either one upright domino or two flat "
       "ones: tₙ = tₙ₋₁ + tₙ₋₂.",
       r"t_n = t_{n-1} + t_{n-2} = F_{n+1}", pop=58),
    _p("K7", "Fibonacci addition by tilings", "斐波那契加法公式（铺砖证法）",
       "A tiling of the 2 × (m + n − 1) strip either breaks after column m − 1"
       " or has a flat pair lying across that line; a 2 × k strip has Fₖ₊₁ "
       "tilings.",
       r"F_{m+n} = F_mF_{n+1} + F_{m-1}F_n",
       note="Drawn for m = n = 4: a 2 × 7 strip, split after column 3.",
       pop=32),
    _p("K8", "Stars and bars", "隔板法",
       "k − 1 bars among n stars split them into k groups (empty groups "
       "allowed): choose the bars' places among the n + k − 1 symbols.",
       r"\binom{n+k-1}{k-1}",
       pop=60),
    _p("K9", "Diagonals of a polygon", "多边形的对角线条数",
       "Each of the n vertices sends n − 3 diagonals, and each diagonal is "
       "counted from both its ends.",
       r"D = \frac{n(n-3)}{2}", pop=57),
    _p("K10", "Regions of a circle: the pattern breaks", "圆内区域数：规律在31处中断",
       "Points on a circle joined in all ways give 1, 2, 4, 8, 16 regions, "
       "then 31 (no three chords through one point): every chord adds one "
       "region plus one for each crossing on it, and crossings are sets of "
       "four points.",
       r"R_n = 1 + \binom{n}{2} + \binom{n}{4}",
       note="Included as a caution, like J8: a pattern is not a proof.",
       pop=46),
    _p("K11", "The mutilated chessboard", "残缺棋盘",
       "Remove two opposite corners: 32 squares of one colour remain and 30 "
       "of the other, but every domino covers one of each.",
       r"30 \neq 32", pop=62),
    _p("K12", "Trominoes on a defective board", "L形三格骨牌铺缺角棋盘",
       "Quarter a 2ⁿ × 2ⁿ board with one cell missing and put one L-tromino at"
       " the centre, covering a cell of each quarter that lacks the missing "
       "cell: four smaller boards, each missing one cell; a 2 × 2 one is a "
       "single tromino.",
       r"3\mid 4^n - 1",
       pop=47),
    _p("K13", "Inclusion–exclusion", "容斥原理",
       "Add the three sets and the overlaps are counted twice, the common "
       "part three times: take the pairs away and put the triple back.",
       r"|A\cup B\cup C| = |A|+|B|+|C|-|A\cap B|-|B\cap C|-|C\cap A|+|A\cap B\cap C|",
       pop=59),
    _p("K14", "Squares on a grid", "方格中的正方形个数",
       "On an n × n grid a k × k square has (n − k + 1)² positions.",
       r"N = \sum_{k=1}^{n}(n-k+1)^2 = \sum_{k=1}^{n}k^2",
       note="Counts the squares with sides on the grid lines; tilted squares "
            "are not counted.",
       pop=54),
    _p("K15", "Rectangles on a grid", "方格中的矩形个数",
       "A rectangle is two of the n + 1 vertical lines and two of the n + 1 "
       "horizontal ones.",
       r"\binom{n+1}{2}^2", pop=48),
    _p("K16", "Lines cutting the plane", "直线分平面",
       "Lines in general position (no two parallel, no three through a point):"
       " the k-th line crosses the k − 1 before it in k − 1 points, which cut "
       "it into k pieces, each splitting a region in two.",
       r"R_n = 1 + n + \binom{n}{2}",
       pop=45),
    _p("K17", "Choosing two of n + 1", "从 n+1 个中选 2 个",
       "Sort the pairs by their larger member j: there are j − 1 partners "
       "below it.",
       r"\binom{n+1}{2} = 1+2+\cdots+n",
       note="Drawn for n + 1 = 6.",
       pop=40),
    _p("K18", "Six people and Ramsey", "拉姆齐数 R(3,3)=6",
       "Of the five edges from one vertex of K₆, three share a colour, say "
       "red. A red edge among their three ends closes a red triangle; if there"
       " is none, the ends form a blue one. K₅ coloured as a red pentagon and "
       "a blue pentagram has no one-colour triangle, so six is the least.",
       r"R(3,3) = 6",
       pop=44),
    _p("K19", "Fibonacci numbers in Pascal's triangle", "帕斯卡三角形中的斐波那契数",
       "The shallow diagonals of Pascal's triangle add up to Fibonacci "
       "numbers: each entry is the sum of two on the two diagonals before.",
       r"\sum_{k}\binom{n-k}{k} = F_{n+1}", pop=42),
    _p("K20", "The handshake lemma", "握手引理",
       "Each edge adds 1 to the degree at both its ends: the degrees add up "
       "to 2E, so odd degrees come in pairs.",
       r"\sum_v \deg v = 2E", pop=43),
    _p("K21", "The meeting problem", "约会问题",
       "Two arrival times are a point of the unit square; they meet when |x − "
       "y| ≤ ¼, the square less two corner triangles.",
       r"P = 1 - \left(\frac{3}{4}\right)^2 = \frac{7}{16}",
       note="The waiting time is ¼; in general P = 1 − (1 − w)².",
       pop=38),
    _p("K22", "Five points in a square", "抽屉原理：正方形中的五点",
       "Cut the unit square into four half-size squares: two of five points "
       "share one, within its diagonal √2/2.",
       r"d \leq \frac{\sqrt{2}}{2}",
       note="The four corners and the centre show that √2/2 cannot be "
            "improved.",
       pop=41),
    _p("K23", "The binomial theorem", "二项式定理",
       "Multiply out (a + b)ⁿ: one term for each way of picking a or b from "
       "every factor, and C(n, k) of them pick b exactly k times.",
       r"(a+b)^n = \sum_{k=0}^{n}\binom{n}{k}a^{n-k}b^k", pop=67),
    _p("K24", "Bertrand's paradox", "贝特朗悖论",
       "A random chord is longer than the inscribed triangle's side with "
       "probability ⅓, ½ or ¼ — depending on how it is chosen.",
       r"P \in \left\{\frac{1}{3},\,\frac{1}{2},\,\frac{1}{4}\right\}",
       note="Included as a caution: 'random' needs a definition.", pop=32),
    _p("K25", "Committees with a chair", "选委员会并选主席",
       "Count the committees with a chair two ways: choose the committee, "
       "then its chair, or the chair, then any subset of the rest.",
       r"\sum_{k}k\binom{n}{k} = n\,2^{n-1}", pop=30),
    _p("K26", "Alternating sum of a row", "组合数的交错和",
       "Pair each subset with the one that toggles the first element: even and"
       " odd sizes match one to one.",
       r"\sum_{k}(-1)^k\binom{n}{k} = 0\quad (n \geq 1)",
       pop=30),
    _p("K27", "The broken stick", "折断的木棍",
       "Two independent uniform cuts are a uniform point of the unit square; "
       "folded along the diagonal and mapped onto the triangle of possible "
       "lengths, the pieces form a triangle exactly when each is shorter than "
       "½ — the middle quarter.",
       r"P = \frac{1}{4}",
       note="'Random' means two independent uniform cuts; breaking the stick "
            "another way (e.g. cutting the longer piece again) changes the "
            "answer.",
       pop=40),
    _p("K28", "A random triangle and the centre", "随机三角形含圆心的概率",
       "Three random points are three random diameters and a fair choice of "
       "end on each. Of the eight triangles, exactly two contain the centre: "
       "one and its half-turn, the two whose corners are every other end "
       "around the circle.",
       r"P = \frac{1}{4}",
       pop=38),
    _p("K29", "Five points on a sphere", "球面上的五个点",
       "The great circle through two of the points has at least two of the "
       "other three on one side (a point on the circle lies on both sides): "
       "that closed hemisphere holds four.",
       r"\max_H\,|H\cap S| \geq 4",
       pop=25),
    _p("K30", "Conjugate partitions", "共轭分拆",
       "Turn a partition's dot diagram over its diagonal: rows become "
       "columns, so partitions into at most k parts match those with parts "
       "at most k.",
       r"p(n,\ \leq k\ \mathrm{parts}) = p(n,\ \mathrm{parts} \leq k)", pop=40),
    _p("K31", "Self-conjugate partitions", "自共轭分拆",
       "Peel a symmetric dot diagram into hooks around its diagonal: each hook"
       " is an odd number, all different; conversely, distinct odd hooks nest "
       "into a symmetric diagram.",
       r"\#\{\lambda = \lambda'\} = \#\{\mathrm{distinct\ odd\ parts}\}",
       pop=30),
    _p("K32", "Distinct parts and odd parts", "不同分拆与奇数分拆",
       "Split each even part in halves until all parts are odd; conversely "
       "merge equal odd parts by the binary digits of their count.",
       r"p(n\,|\,\mathrm{distinct}) = p(n\,|\,\mathrm{odd})",
       note="Euler's partition theorem, by Glaisher's correspondence.", pop=35),

    # ---------------------------------------------------------------- L
    _p("L1", "Vector addition commutes", "向量加法交换律",
       "Both orders of the two arrows run along the two halves of one "
       "parallelogram to the same corner.",
       r"\vec{u}+\vec{v} = \vec{v}+\vec{u}", pop=58),
    _p("L2", "Dot product as a projection", "数量积与投影",
       "u·v is the length of u times the signed shadow of v on u.",
       r"\vec{u}\cdot\vec{v} = u_1v_1 + u_2v_2 = |\vec{u}|\,|\vec{v}|\cos\theta",
       note="Drawn with θ acute and all components positive; with signed "
            "shadows the same sum holds in general.",
       pop=56),
    _p("L3", "Determinant as area", "行列式即面积",
       "The parallelogram on (a, c) and (b, d) sits in the (a + b) × (c + d) "
       "rectangle with six corner pieces; regrouped in a second copy of the "
       "rectangle, the same pieces cover all but an L-shape of area ad − bc.",
       r"\mathrm{area} = ad - bc",
       note="Drawn with b ≤ a and c ≤ d; the subtraction holds whenever ad > "
            "bc.",
       pop=60),
    _p("L4", "The shoelace formula", "鞋带公式",
       "A polygon's area as the signed sum of the triangles from the origin to"
       " each edge: half the cross products of consecutive vertices.",
       r"A = \frac{1}{2}\sum_i\left(x_iy_{i+1} - x_{i+1}y_i\right)",
       note="Drawn for a convex pentagon taken counter-clockwise, with O "
            "outside it; each triangle is half the parallelogram of L3. The "
            "signed count works for any simple polygon and any O.",
       pop=47),
    _p("L5", "Cramer's rule", "克莱姆法则",
       "Write c = x·a + y·b. Shearing along b removes the y·b part without "
       "changing area, so the parallelogram on c and b is x times the one on a"
       " and b.",
       r"x = \frac{\det(\vec{c},\vec{b})}{\det(\vec{a},\vec{b})}",
       note="Drawn with det(a, b) > 0, x = 2.4, y = 0.7; otherwise the areas "
            "are signed.",
       pop=40),
    _p("L6", "Multiplying complex numbers", "复数乘法：旋转与伸缩",
       "Multiplication by w turns by arg w and scales by |w|: the triangle "
       "0, 1, z goes to the similar triangle 0, w, zw.",
       r"|zw| = |z|\,|w|,\quad \arg(zw) = \arg z + \arg w", pop=57),
    _p("L7", "Roots of unity add to zero", "单位根之和为零",
       "The n arrows from the centre to the corners of a regular n-gon have "
       "length 1, each turned 2π/n from the last; placed head to tail they "
       "turn 2π/n at every corner and close up into a regular n-gon, so they "
       "add to 0.",
       r"\sum_{k=0}^{n-1}e^{2\pi ik/n} = 0",
       note="Drawn for n = 7.",
       pop=42),
    _p("L8", "Distance from a point to a line", "点到直线的距离",
       "The triangle with apex P on the segment of the line between its "
       "intercepts: its area two ways gives the height.",
       r"d = \frac{|ax_0+by_0+c|}{\sqrt{a^2+b^2}}",
       note="Drawn for a line not through O, scaled so that c = −1, with P "
            "beyond it; elsewhere the areas are signed, hence the absolute "
            "value.",
       pop=46),
    _p("L9", "Two reflections make a rotation", "两次反射等于一次旋转",
       "Reflect in a line and then in a second line through the same point at "
       "angle θ: every point turns through 2θ.",
       r"R_{\ell_2}\circ R_{\ell_1} = \mathrm{Rot}_{2\theta}",
       note="Drawn with the first image between the mirrors; elsewhere the "
            "same count holds with signed angles. The turn goes from ℓ₁ "
            "towards ℓ₂.",
       pop=38),
    _p("L10", "A linear map scales every area", "线性变换按行列式缩放面积",
       "A grid of unit squares maps to a grid of congruent parallelograms of "
       "area |det T|, so every region's area scales by it.",
       r"[T(S)] = |\det T|\,[S]",
       note="Drawn with det T > 0; for det T < 0 the picture also flips and "
            "|det T| is the factor.",
       pop=39),
    _p("L11", "Order matters", "矩阵乘法不可交换",
       "Turn a quarter-turn, then reflect — or reflect, then turn: a flag with"
       " no symmetry ends up in a different place, facing a different way (one"
       " order is the mirror in y = x, the other the mirror in y = −x).",
       r"RS \neq SR",
       pop=30),
    _p("L12", "Projection is additive", "投影的可加性（数量积分配律）",
       "Shadows of two arrows placed head to tail add along the line: the dot "
       "product distributes over a sum.",
       r"\vec{u}\cdot(\vec{v}+\vec{w}) = \vec{u}\cdot\vec{v} + \vec{u}\cdot\vec{w}",
       note="Drawn with all shadows pointing along u; signed shadows add the "
            "same way.",
       pop=28),
    _p("L13", "The rotation matrix", "旋转矩阵",
       "Turn the unit square through θ: its sides go to (cos θ, sin θ) and "
       "(−sin θ, cos θ), the columns of the matrix.",
       r"R_\theta e_1 = (\cos\theta,\,\sin\theta),\quad R_\theta e_2 = (-\sin\theta,\,\cos\theta)",
       pop=35),
    _p("L14", "Composing rotations", "旋转的合成",
       "Turning by A and then by B is turning by A + B: where the first "
       "unit vector lands, read two ways, gives the addition formulas.",
       r"R_BR_A = R_{A+B}", pop=35),
    _p("L15", "Determinants multiply", "行列式的乘法性",
       "Apply S and then T: every area is scaled by |det S| and then by |det "
       "T|.",
       r"\det(TS) = \det T\,\det S",
       note="Drawn with det S = 4 and det T > 0; orientation signs multiply "
            "the same way.",
       pop=30),
]

BY_ID = {e.id: e for e in CATALOGUE}


def _id_key(eid: str):
    """A1 < A2 < A10 < B1: category letters, then the number."""
    head = eid.rstrip("0123456789")
    return (head, int(eid[len(head):]))


def tier_of(e: ProofEntry) -> int:
    """0 = Classics, 1 = Standard, 2 = Specialist."""
    return next(i for i, (lo, _, _) in enumerate(TIERS) if e.pop >= lo)


RANKED: list[ProofEntry] = sorted(CATALOGUE, key=lambda e: (-e.pop, _id_key(e.id)))
RANK = {e.id: i + 1 for i, e in enumerate(RANKED)}


def in_category(letter: str) -> list[ProofEntry]:
    return [e for e in CATALOGUE if e.category == letter]


def implemented() -> list[ProofEntry]:
    return [e for e in CATALOGUE if e.status == "done"]


if __name__ == "__main__":
    from pathlib import Path
    from pww_discover import discover
    found, problems = discover(Path(__file__).resolve().parent)
    animated = [e for e in CATALOGUE if found.get(e.id)]
    print(f"{len(CATALOGUE)} entries, {len(animated)} animated")
    for letter, (en, zh) in CATEGORIES.items():
        items = in_category(letter)
        n = sum(1 for e in items if found.get(e.id))
        print(f"  {letter}. {en} / {zh}: {len(items)} entries, {n} animated")
    for msg in problems:
        print("  problem:", msg)
