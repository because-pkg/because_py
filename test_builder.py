import jax
from because.builder import FormulaParser, CausalGraph, NumPyroBuilder

eqs = [
    "Abundance ~ Flower_Cover + Wind_Speed + I(Temperature_s - CTmax)"
]

parser = FormulaParser(eqs)
parsed = parser.parse()
deterministic_terms = parser.deterministic_terms

graph = CausalGraph(parsed, deterministic_terms=deterministic_terms)
graph.build()

print("Equations:")
for n in graph.nodes.values():
    print(f"{n.name}: parents = {n.parents}")
