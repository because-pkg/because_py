import sys
sys.path.insert(0, ".")
from because.builder.parser import FormulaParser
from because.builder.causal_graph import CausalGraph
from because.builder.compiler import NumPyroBuilder
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import pandas as pd
import json

df = pd.read_csv("because_research/data/primate_data.csv")
equations = ["Lifespan ~ Brain"]

parser = FormulaParser(equations)
graph = CausalGraph()
for src, dst in parser.edges:
    graph.add_edge(src, dst)

builder = NumPyroBuilder(None)
builder.generate_model_function(graph, {"Lifespan": df["Lifespan"].values, "Brain": df["Brain"].values})
print("SUCCESS")
