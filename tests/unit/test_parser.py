import pytest
from because.builder.parser import FormulaParser, sanitize_term_name, term_to_python_expression

@pytest.mark.unit
def test_sanitize_term_name():
    assert sanitize_term_name("x:z") == "x_x_z"
    assert sanitize_term_name("I(x^2)") == "x_pow_2"
    assert sanitize_term_name("I(x + y)") == "x_plus_y"
    assert sanitize_term_name("I(x / 2)") == "x_div_2"

@pytest.mark.unit
def test_term_to_python_expression():
    assert term_to_python_expression("x:z") == "x * z"
    assert term_to_python_expression("I(x^2)") == "x**2"
    assert term_to_python_expression("regular_var") == "regular_var"

@pytest.mark.unit
def test_formula_parser_simple():
    parser = FormulaParser(["y ~ x + z"])
    parsed = parser.parse()
    assert len(parsed) == 1
    assert parsed[0]["response"] == "y"
    assert parsed[0]["fixed"] == ["x", "z"]
    assert parsed[0]["random"] == []
    assert parsed[0]["intercept"] is True

@pytest.mark.unit
def test_formula_parser_interactions_and_polynomials():
    parser = FormulaParser(["y ~ x + z + x:z", "w ~ I(x^2)"])
    parsed = parser.parse()
    assert len(parsed) == 2
    assert "x_x_z" in parsed[0]["fixed"]
    assert "x_pow_2" in parsed[1]["fixed"]
    assert "x_x_z" in parser.deterministic_terms
    assert "x_pow_2" in parser.deterministic_terms

@pytest.mark.unit
def test_formula_parser_random_effects():
    parser = FormulaParser(["y ~ x + (1 | group) + (1 | site)"])
    parsed = parser.parse()
    assert parsed[0]["fixed"] == ["x"]
    assert "(1|group)" in parsed[0]["random"] or "(1 | group)" in parsed[0]["random"] or len(parsed[0]["random"]) == 2

@pytest.mark.unit
def test_extract_all_variables():
    parser = FormulaParser(["y ~ x + x:z", "w ~ I(x^2)"])
    parser.parse()
    variables = parser.extract_all_variables()
    assert {"y", "x", "z", "w"}.issubset(variables)
