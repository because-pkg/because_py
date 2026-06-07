import jax
from because.builder import FormulaParser, CausalGraph, NumPyroBuilder

eqs = [
    "NDVI ~ Mean_Temp + U_Resource",
    "Flower_Cover ~ Mean_Temp + U_Resource"
]

parser = FormulaParser(eqs)
parsed = parser.parse()
det_terms = parser.deterministic_terms
graph = CausalGraph(parsed, deterministic_terms=det_terms)
graph.build()

builder = NumPyroBuilder(graph, fix_latent="loading")
code = builder.generate_model_function(data_for_compilation={"NDVI": jax.numpy.zeros(10), "Flower_Cover": jax.numpy.zeros(10), "Mean_Temp": jax.numpy.zeros(10)})
print("Done")
