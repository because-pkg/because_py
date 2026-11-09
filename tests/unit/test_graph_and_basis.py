import pytest
from because.builder.parser import FormulaParser
from because.builder.graph import CausalGraph

@pytest.mark.unit
def test_causal_graph_topological_sort():
    equations = [
        "y ~ x + z",
        "z ~ x"
    ]
    parser = FormulaParser(equations)
    graph = CausalGraph(parser.parse())
    graph.build()
    topo = graph.get_topological_order()
    assert topo.index("x") < topo.index("z")
    assert topo.index("z") < topo.index("y")

@pytest.mark.unit
def test_causal_graph_cycle_detection():
    equations = [
        "y ~ x",
        "x ~ y"
    ]
    parser = FormulaParser(equations)
    graph = CausalGraph(parser.parse())
    with pytest.raises(ValueError, match="cyclic dependency"):
        graph.build()

@pytest.mark.unit
def test_causal_graph_basis_set_standard():
    equations = [
        "y ~ x",
        "z ~ y"
    ]
    parser = FormulaParser(equations)
    graph = CausalGraph(parser.parse())
    graph.build()
    basis = graph.get_basis_set()
    assert len(basis) == 1
    claim = basis[0]
    assert {claim["response"], claim["test_node"]} == {"z", "x"}
    assert claim["conditioning_set"] == ["y"]

@pytest.mark.unit
def test_causal_graph_latent_mag():
    equations = [
        "x ~ L",
        "y ~ L",
        "z ~ y"
    ]
    parser = FormulaParser(equations)
    graph = CausalGraph(parser.parse())
    graph.build()
    
    # x, y, and z are all descendants of L
    corrs = graph.get_induced_correlations(latent=["L"])
    assert len(corrs) == 2
    pairs = [{c["u"], c["v"]} for c in corrs]
    assert {"x", "y"} in pairs
    assert {"x", "z"} in pairs

    # Basis set for observed nodes: z _|_ x | y
    basis = graph.get_basis_set(latent=["L"])
    assert len(basis) == 1
    assert {basis[0]["response"], basis[0]["test_node"]} == {"z", "x"}
    assert basis[0]["conditioning_set"] == ["y"]

@pytest.mark.unit
def test_generate_dsep_equations():
    equations = [
        "Brain ~ Size",
        "Mass ~ Size",
        "Clutch ~ Brain",
        "Migration ~ Brain",
        "Lifespan ~ Clutch + Migration"
    ]
    parser = FormulaParser(equations)
    graph = CausalGraph(parser.parse())
    graph.build()
    dsep_eqs = graph.generate_dsep_equations(latent=["Size"])
    assert len(dsep_eqs) > 0
    # Every generated test should have an equation string
    for eq in dsep_eqs:
        assert "~" in eq["equation_string"]
