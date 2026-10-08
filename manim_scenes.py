from manim import *
import numpy as np

class Pixels(Scene):
    """Image -> vector of numbers"""
    def construct(self):
        rng = np.random.default_rng(1)
        g = VGroup(*[Square(0.3, stroke_width=1, stroke_color=GREY, fill_color=WHITE,
                            fill_opacity=float(rng.random() ** 2)) for _ in range(64)]).arrange_in_grid(8, 8, buff=0)
        self.play(Create(g)); self.wait()
        vec = Matrix([["0.0"], ["0.8"], ["0.3"], [r"\vdots"], ["0.9"], ["0.1"]]).scale(0.9)
        self.play(g.animate.scale(0.6).to_edge(LEFT, buff=1.5))
        self.play(Write(vec)); self.wait()
        self.play(Write(MathTex(r"28\times 28 = 784").next_to(vec, DOWN)))
        self.wait(2)

class MatVec(Scene):
    """One layer = one matrix-vector product"""
    def construct(self):
        eq = MathTex(r"\mathbf{y}", "=", "W", r"\mathbf{x}", "+", r"\mathbf{b}").scale(1.6)
        self.play(Write(eq)); self.wait()
        dims = [MathTex(t, color=c).scale(0.7).next_to(eq[i], DOWN)
                for t, c, i in [(r"10\times1", YELLOW, 0), (r"10\times784", BLUE, 2),
                                (r"784\times1", GREEN, 3), (r"10\times1", RED, 5)]]
        self.play(*[FadeIn(d) for d in dims]); self.wait(3)

class Collapse(Scene):
    """Why hidden layers need ReLU"""
    def construct(self):
        a = MathTex(r"W_3\,W_2\,W_1\,\mathbf{x}", "=", r"(W_3 W_2 W_1)\,\mathbf{x}", "=", r"W\,\mathbf{x}").scale(1.3)
        self.play(Write(a)); self.wait(2)
        b = MathTex(r"W_3\,\mathrm{ReLU}\big(W_2\,\mathrm{ReLU}(W_1\mathbf{x})\big)", color=YELLOW).scale(1.3)
        self.play(FadeOut(a), Write(b)); self.wait(3)
