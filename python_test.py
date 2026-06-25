from because.builder.compiler import NumPyroBuilder
from because.builder.formula_parser import FormulaParser
from because.builder.causal_graph import CausalGraph

eqs = ["Abundance ~ Body_Mass_s + Thermal_Tol + (1 | Site) + (1 | Survey) + (1 | Species) + (1 | phylo) + (1 | spatial)"]
parsed = FormulaParser(eqs).parse()
graph = CausalGraph(parsed)

data = {
    "Abundance": [1]*4500,
    "Body_Mass_s": [1.0]*50,
    "Thermal_Tol": [1.0]*50,
    "Site": [1]*30,
    "Survey": [1]*90,
    "Species": [1]*50,
    "phylo": [1]*50,
    "spatial": [1]*30,
    "species_idx_obs": [1]*4500,
    "site_idx_obs": [1]*4500,
    "survey_idx_obs": [1]*4500
}

try:
    compiler = NumPyroBuilder(graph)
    compiler.generate_model_function(data_for_compilation=data)
    print("Success")
except Exception as e:
    print(repr(e))
