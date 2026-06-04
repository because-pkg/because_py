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
        If latent variables are provided, projects to a MAG by filtering untestable claims
        (Shipley & Douma 2021).
        """
        if self.dag is None:
            self.build()
            
        latent = latent or []
        topo_order = self.get_topological_order()
        basis_set = []
        
        for i, vi in enumerate(topo_order):
            parents_vi = set(self.dag.predecessors(vi))
            preceding_nodes = set(topo_order[:i])
            non_descendants = preceding_nodes - parents_vi
            
            for vj in non_descendants:
                basis_set.append({
                    "response": vi,
                    "test_node": vj,
                    "conditioning_set": list(parents_vi)
                })
                
        # Filtering
        filtered_basis = []
        det_names = set(self.deterministic_terms.keys())
        
        for claim in basis_set:
            vi = claim["response"]
            vj = claim["test_node"]
            cond = claim["conditioning_set"]
            
            # Deterministic nodes cannot be the focal response or test node
            # (they have 0 structural variance)
            if vi in det_names or vj in det_names:
                continue
                
            # MAG Projection: Remove untestable claims involving latents
            if latent:
                # Cannot condition on a latent
                if any(c in latent for c in cond):
                    continue
                # Cannot test independence involving a latent directly
                if vi in latent or vj in latent:
                    continue
                    
            filtered_basis.append(claim)
            
        return filtered_basis
        
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
        
        # 1. Standard / Filtered Basis Set Claims
        for claim in basis:
            resp = claim["response"]
            test_var = claim["test_node"]
            cond = claim["conditioning_set"]
            
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
