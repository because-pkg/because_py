import re
import hashlib

def sanitize_term_name(term):
    """
    Converts complex R terms into valid Python variable names.
    Matches R's because logic.
    """
    out = term
    out = out.replace(":", "_x_")
    
    if out.startswith("I(") and out.endswith(")"):
        out = out[2:-1]
        
    out = out.replace("^", "_pow_")
    out = out.replace("**", "_pow_")
    out = out.replace("+", "_plus_")
    out = out.replace("-", "_minus_")
    out = out.replace("*", "_times_")
    out = out.replace("/", "_div_")
    
    # Clean up non-alphanumeric chars
    out = re.sub(r'[^a-zA-Z0-9_]', '_', out)
    out = re.sub(r'_+', '_', out)
    out = out.strip('_')
    
    if len(out) > 32:
        checksum = hashlib.md5(term.encode()).hexdigest()[:6]
        out = f"det_{out[:20]}_{checksum}"
    else:
        out = f"det_{out}"
        
    return out

def term_to_python_expression(term):
    """
    Converts R deterministic term syntax to valid Python code for JAX.
    """
    if ":" in term and not "(" in term:
        # Interaction A:B -> A * B
        parts = term.split(":")
        return " * ".join(parts)
        
    if term.startswith("I(") and term.endswith(")"):
        inner = term[2:-1]
        # R uses ^ for power, Python uses **
        inner = inner.replace("^", "**")
        return inner
        
    return term

class FormulaParser:
    """
    Parses R-style formulas into structured components (response, fixed effects, random effects).
    Designed to mimic because's formula lists in R.
    Now supports deterministic nodes: I(A^2), A:B, A*B
    """
    def __init__(self, equations):
        self.equations = equations
        self.parsed_equations = []
        self.deterministic_terms = {}
        
    def parse(self):
        for eq in self.equations:
            # Basic cleanup, keep spaces inside I() if needed but usually better to strip
            eq_clean = eq.replace(" ", "")
            
            if "~" not in eq_clean:
                raise ValueError(f"Invalid formula (missing ~): {eq}")
                
            lhs, rhs = eq_clean.split("~", 1)
            response = lhs if lhs else None
            
            # Split RHS by '+' but be careful not to split inside parentheses like (1|group) or I(A+B)
            # Match + that is not followed by unbalanced parentheses
            # Better approach: parse carefully
            rhs_terms = []
            depth = 0
            current = []
            for char in rhs:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    
                if char == '+' and depth == 0:
                    rhs_terms.append("".join(current))
                    current = []
                else:
                    current.append(char)
            if current:
                rhs_terms.append("".join(current))
            
            # Expand A*B -> A + B + A:B
            expanded_terms = []
            for term in rhs_terms:
                if '*' in term and not ('(' in term or ')' in term):
                    parts = term.split('*')
                    expanded_terms.extend(parts)
                    expanded_terms.append(":".join(parts))
                else:
                    expanded_terms.append(term)
            
            fixed_effects = []
            random_effects = []
            intercept = True
            
            for term in expanded_terms:
                if not term:
                    continue
                if term == "0" or term == "-1":
                    intercept = False
                elif "|" in term and term.startswith("(") and term.endswith(")"):
                    random_effects.append(term)
                elif term == "1":
                    pass # Explicit intercept
                else:
                    # Check for deterministic logic
                    if ":" in term or (term.startswith("I(") and term.endswith(")")):
                        internal_name = sanitize_term_name(term)
                        python_expr = term_to_python_expression(term)
                        
                        self.deterministic_terms[internal_name] = {
                            "original": term,
                            "internal_name": internal_name,
                            "expression": python_expr
                        }
                        fixed_effects.append(internal_name)
                    else:
                        fixed_effects.append(term)
                    
            if response:
                # deduplicate if A*B created A + B but A was already there
                unique_fixed = []
                for fe in fixed_effects:
                    if fe not in unique_fixed:
                        unique_fixed.append(fe)
                        
                self.parsed_equations.append({
                    "raw": eq,
                    "response": response,
                    "fixed": unique_fixed,
                    "random": random_effects,
                    "intercept": intercept
                })
            
        return self.parsed_equations

    def extract_all_variables(self):
        """
        Returns a set of all base variable names mentioned across all equations.
        Requires extracting variables from deterministic expressions.
        """
        variables = set()
        for p in self.parsed_equations:
            if p["response"]:
                variables.add(p["response"])
            for fe in p["fixed"]:
                if fe in self.deterministic_terms:
                    # Extract variables from the python expression
                    expr = self.deterministic_terms[fe]["expression"]
                    # Find all words (variable names)
                    words = re.findall(r'[a-zA-Z_]\w*', expr)
                    for w in words:
                        if w not in ("jnp", "np"):
                            variables.add(w)
                else:
                    variables.add(fe)
        return variables
