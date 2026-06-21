import networkx as nx
import itertools

def get_optimal_basis_set(dag, latents):
    observed_nodes = [n for n in dag.nodes if n not in latents]
    
    latent_sharers = {n: set() for n in observed_nodes}
    for l in latents:
        if l in dag.nodes:
            children = list(dag.successors(l))
            for c in children:
                if c in latent_sharers:
                    latent_sharers[c].update(x for x in children if x != c)
    
    pairs = list(itertools.combinations(observed_nodes, 2))
    
    basis_set = []
    topo_order = list(nx.topological_sort(dag))
    
    for vi, vj in pairs:
        if dag.has_edge(vi, vj) or dag.has_edge(vj, vi):
            continue
        if vj in latent_sharers[vi]:
            continue
            
        best_Z = None
        best_score = (float('inf'), float('inf'))
        
        ancestors = set(nx.ancestors(dag, vi)) | set(nx.ancestors(dag, vj))
        candidate_pool = [n for n in ancestors if n in observed_nodes and n != vi and n != vj]
        
        found_perfect = False
        for r in range(len(candidate_pool) + 1):
            for z_tuple in itertools.combinations(candidate_pool, r):
                Z = set(z_tuple)
                if nx.d_separated(dag, {vi}, {vj}, Z):
                    col_pen = sum(1 for z in Z if z in latent_sharers[vi] or z in latent_sharers[vj])
                    size_pen = len(Z)
                    
                    if (col_pen, size_pen) < best_score:
                        best_score = (col_pen, size_pen)
                        best_Z = list(Z)
                        
                    if best_score[0] == 0:
                        found_perfect = True
                        break
            if found_perfect:
                break
                
        if best_Z is not None:
            if topo_order.index(vi) > topo_order.index(vj):
                resp, test_node = vi, vj
            else:
                resp, test_node = vj, vi
                
            basis_set.append({
                "response": resp,
                "test_node": test_node,
                "conditioning_set": best_Z
            })
            
    return basis_set

dag = nx.DiGraph()
dag.add_edges_from([
    ("Size", "Brain"), ("Size", "Mass"),
    ("Brain", "Clutch"), ("Brain", "Migration"),
    ("Clutch", "Lifespan"), ("Migration", "Lifespan")
])

print(get_optimal_basis_set(dag, ["Size"]))
