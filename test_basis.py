from because.builder.parser import FormulaParser
from because.builder.graph import CausalGraph

equations = [
    "Brain ~ Size",
    "Mass ~ Size",
    "Clutch ~ Brain",
    "Migration ~ Brain",
    "Lifespan ~ Clutch + Migration"
]

parser = FormulaParser(equations)
parsed = parser.parse()
graph = CausalGraph(parsed)
graph.build()
basis = graph.get_basis_set(latent=["Size"])
for b in basis:
    print(f"{b['response']} _|_ {b['test_node']} | {b['conditioning_set']}")
