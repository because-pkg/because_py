import networkx as nx
import re

class CausalGraph:
    """
    Constructs a Directed Acyclic Graph (DAG) from parsed formula equations.
    Provides methods for topological sorting and computing the d-separation basis set.
    """
    def __init__(self, parsed_equations, deterministic_terms=None):
        # Allow both dictionary or list for flexibility
        if isinstance(parsed_equations, list):
            self.parsed_equations = {eq["response"]: eq for eq in parsed_equations if eq["response"]}
        else:
            self.parsed_equations = parsed_equations
            
        self.deterministic_terms = deterministic_terms or {}
        self.dag = None
        
    def build(self):
        """
        Builds the directed acyclic graph from the parsed equations.
        Validates that it is a DAG.
        """
        G = nx.DiGraph()
        
        # Add edges for fixed predictors
        for eq in self.parsed_equations.values():
            response = eq["response"]
            G.add_node(response)
            for pred in eq["fixed"]:
                G.add_edge(pred, response)
                
        # Add explicit edges for deterministic component variables
        for det_name, det_info in self.deterministic_terms.items():
            expr = det_info["expression"]
            words = set(re.findall(r'[a-zA-Z_]\w*', expr))
            for w in words:
                if w not in ("jnp", "np", det_name):
                    G.add_edge(w, det_name)
                
        if not nx.is_directed_acyclic_graph(G):
            # Find the cycle to report to the user
            try:
                cycles = list(nx.simple_cycles(G))
                raise ValueError(f"The specified equations contain a cyclic dependency: {cycles}")
            except nx.NetworkXNoCycle:
                pass
            
        self.dag = G
        return G
        
    def get_topological_order(self):
        """
        Returns a list of nodes in topological order.
        """
        if self.dag is None:
            self.build()
        return list(nx.topological_sort(self.dag))
        
    def get_basis_set(self, latent=None):
        """
        Computes the minimal DAG basis set of implied conditional independencies.
        If latent variables are provided, projects to a MAG (Shipley & Douma 2021)
        and natively applies a collinearity penalty to select the optimal
        conditioning sets of purely observed variables.
        """
        if self.dag is None:
            self.build()
            
        latent = latent or []
        import itertools
        
        # We only test pairs of observed, non-deterministic nodes
        det_names = set(self.deterministic_terms.keys())
        observed_nodes = [n for n in self.dag.nodes if n not in latent and n not in det_names]
        
        # Track which nodes share a latent parent
        latent_sharers = {n: set() for n in observed_nodes}
        for l in latent:
            if l in self.dag.nodes:
                children = list(self.dag.successors(l))
                for c in children:
                    if c in latent_sharers:
                        latent_sharers[c].update(x for x in children if x != c)
        
        pairs = list(itertools.combinations(observed_nodes, 2))
        topo_order = list(nx.topological_sort(self.dag))
        
        # Helper for networkx deprecation
        def check_d_separated(G, u, v, z):
            if hasattr(nx, "is_d_separator"):
                return nx.is_d_separator(G, u, v, z)
            else:
                return nx.d_separated(G, u, v, z)
        
        basis_set = []
        for vi, vj in pairs:
            # Check if adjacent in DAG
            if self.dag.has_edge(vi, vj) or self.dag.has_edge(vj, vi):
                continue
            # Check if adjacent in MAG (share a latent parent)
            if vj in latent_sharers.get(vi, set()):
                continue
                
            best_Z = None
            best_score = (float('inf'), float('inf'))
            
            # The minimal separating set is guaranteed to be a subset of the ancestors
            ancestors = set(nx.ancestors(self.dag, vi)) | set(nx.ancestors(self.dag, vj))
            candidate_pool = [n for n in ancestors if n in observed_nodes and n != vi and n != vj and n not in det_names]
            
            found_perfect = False
            for r in range(len(candidate_pool) + 1):
                for z_tuple in itertools.combinations(candidate_pool, r):
                    Z = set(z_tuple)
                    if check_d_separated(self.dag, {vi}, {vj}, Z):
                        # Collinearity penalty
                        col_pen = sum(1 for z in Z if z in latent_sharers.get(vi, set()) or z in latent_sharers.get(vj, set()))
                        size_pen = len(Z)
                        
                        if (col_pen, size_pen) < best_score:
                            best_score = (col_pen, size_pen)
                            best_Z = list(Z)
                            
                        # If size is minimal (because we iterate r ascending) and col_pen is 0, this is globally optimal
                        if best_score[0] == 0:
                            found_perfect = True
                            break
                if found_perfect:
                    break
                    
            if best_Z is not None:
                # Collinearity Orientation Rule: Minimize RHS collinearity
                rhs1 = [vj] + best_Z # predictors if vi is response
                rhs2 = [vi] + best_Z # predictors if vj is response
                
                col_pen1 = sum(1 for z1, z2 in itertools.combinations(rhs1, 2) if z1 in latent_sharers.get(z2, set()))
                col_pen2 = sum(1 for z1, z2 in itertools.combinations(rhs2, 2) if z1 in latent_sharers.get(z2, set()))
                
                if col_pen2 < col_pen1:
                    resp, test_node = vj, vi
                elif col_pen1 < col_pen2:
                    resp, test_node = vi, vj
                else:
                    # Tie-breaker: Topo order dictates response vs predictor
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
        
    def get_induced_correlations(self, latent):
        """
        Identifies pairs of observed variables that share a latent common ancestor,
        implying a bidirected edge in the MAG (an induced correlation).
        """
        if self.dag is None:
            self.build()
            
        latent = latent or []
        if not latent:
            return []
            
        observed_nodes = [n for n in self.dag.nodes if n not in latent]
        correlations = []
        
        # Precompute descendants for latents
        latent_descendants = {l: set(nx.descendants(self.dag, l)) for l in latent}
        
        for i, u in enumerate(observed_nodes):
            for v in observed_nodes[i+1:]:
                # If they are already adjacent, skip
                if self.dag.has_edge(u, v) or self.dag.has_edge(v, u):
                    continue
                    
                # Check if they share a latent ancestor
                for l in latent:
                    if u in latent_descendants[l] and v in latent_descendants[l]:
                        correlations.append({"u": u, "v": v})
                        break # Only need to find one latent common ancestor
                        
        return correlations
        
    def generate_dsep_equations(self, latent=None):
        """
        Converts the basis set and induced correlations into string equations suitable for testing.
        """
        basis = self.get_basis_set(latent=latent)
        dsep_equations = []
        
        root_nodes = [n for n in self.dag.nodes if self.dag.in_degree(n) == 0]
        
        # Determine latent children if latents exist
        latent_children = set()
        if latent:
            for l in latent:
                latent_children.update(self.dag.successors(l))
                
        # 1. Standard / Filtered Basis Set Claims
        for claim in basis:
            resp = claim["response"]
            test_var = claim["test_node"]
            cond = claim["conditioning_set"]
            

            # Rule 2: Root node swap
            # Root nodes (no parents) should always be predictors, never responses.
            # If resp is a root node and test_var is not -> swap
            r_root = resp in root_nodes
            t_root = test_var in root_nodes
            if r_root and not t_root:
                resp, test_var = test_var, resp
                
            orig_eq = self.parsed_equations.get(resp)
            random_terms = orig_eq["random"] if orig_eq else []
            
            predictors = [test_var] + cond + random_terms
            
            if not predictors:
                eq_str = f"{resp} ~ 1"
            else:
                eq_str = f"{resp} ~ " + " + ".join(predictors)
                
            dsep_equations.append({
                "type": "dsep",
                "response": resp,
                "test_node": test_var,
                "conditioning_set": cond,
                "equation_string": eq_str,
                "random_terms": random_terms
            })
            
        # 2. MAG Induced Correlations (Bidirected edges)
        if latent:
            correlations = self.get_induced_correlations(latent)
            for corr in correlations:
                u, v = corr["u"], corr["v"]
                
                # Test correlation by regressing u on v (could be either direction)
                orig_eq = self.parsed_equations.get(u)
                random_terms = orig_eq["random"] if orig_eq else []
                
                predictors = [v] + random_terms
                eq_str = f"{u} ~ " + " + ".join(predictors)
                
                dsep_equations.append({
                    "type": "correlation",
                    "response": u,
                    "test_node": v,
                    "conditioning_set": [],
                    "equation_string": eq_str,
                    "random_terms": random_terms
                })
            
        return dsep_equations
