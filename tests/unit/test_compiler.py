import pytest
import numpy as np
from because.builder import FormulaParser, CausalGraph, NumPyroBuilder

@pytest.mark.unit
def test_builder_codegen():
    eqs = [
        "y ~ x + (1 | group)",
        "w ~ y"
    ]
    parser = FormulaParser(eqs)
    graph = CausalGraph(parser.parse())
    graph.build()

    builder = NumPyroBuilder(graph)
    code_str = builder.generate_model_code_string()

    assert "def numpyro_model" in code_str
    assert "alpha_y" in code_str
    assert "beta_y_x" in code_str
    assert "beta_w_y" in code_str
    assert "dist.Normal" in code_str

@pytest.mark.unit
def test_builder_generate_model_function():
    eqs = ["y ~ x"]
    parser = FormulaParser(eqs)
    graph = CausalGraph(parser.parse())
    graph.build()

    builder = NumPyroBuilder(graph)
    model_func = builder.generate_model_function()
    assert callable(model_func)
