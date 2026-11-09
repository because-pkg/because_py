import ast
import numpy as np

def _get_level_names(size, data):
    candidates = []
    for k, val in data.items():
        if k.startswith("N_") and int(val) == size:
            lname = k[2:]
            if not lname.isupper() and "ID" not in lname and "_id" not in lname:
                candidates.append(lname)
    return candidates


def evaluate_deterministic_node(var, det_terms, equations_dict, data, computed_vars, var_to_level, jnp):
    """
    Evaluates a deterministic transformation node (e.g. I(X^2) or X:Z) with cross-scale
    shape alignment across hierarchical levels.
    """
    expr = det_terms[var]["expression"]
    target_size = None
    parent_var = None

    if f"N_{var}" in data:
        target_size = int(data[f"N_{var}"])
        parent_var = var
    else:
        for p_var, eq in equations_dict.items():
            if var in eq.get("fixed", []) or var in eq.get("random", []):
                parent_var = p_var
                if p_var in data:
                    target_size = data[p_var].shape[0]
                elif f"N_{p_var}" in data:
                    target_size = int(data[f"N_{p_var}"])
                break

    if target_size is not None:
        used_vars = [node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)]
        used_vars = [v for v in used_vars if v in computed_vars]

        local_vars = dict(computed_vars)
        for v in used_vars:
            v_data = computed_vars[v]
            if hasattr(v_data, "shape") and len(v_data.shape) > 0 and v_data.shape[0] < target_size:
                v_size = v_data.shape[0]
                source_levels = [var_to_level[v]] if v in var_to_level else _get_level_names(v_size, data)
                target_levels = [var_to_level[parent_var]] if (parent_var and parent_var in var_to_level) else _get_level_names(target_size, data)

                idx_array = None
                for s in source_levels:
                    for t in target_levels:
                        idx_name = f"{s}_idx_{t}"
                        if idx_name in data:
                            idx_array = data[idx_name]
                            break
                    if idx_array is not None:
                        break

                if idx_array is None:
                    for k, val in data.items():
                        if "idx" in k and hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] == target_size:
                            if jnp.max(val) < v_size:
                                idx_array = val
                                break

                if idx_array is not None:
                    local_vars[v] = v_data[idx_array]
                else:
                    raise ValueError(
                        f"Shape mismatch in deterministic node {var}: {v} ({v_size}) vs target ({target_size}). "
                        f"No bridging index found."
                    )

        return eval(expr, {"jnp": jnp, "np": np}, local_vars)
    else:
        try:
            return eval(expr, {"jnp": jnp, "np": np}, computed_vars)
        except Exception:
            return None


def bridge_predictor(pred, var, pred_data, target_size, data, var_to_level, jnp, jax):
    """
    Aligns a predictor array to the target response size:
    - Coarse predictor -> fine response: broadcasts via index array
    - Fine predictor -> coarse response: aggregates via segment_sum (resolution locking)
    """
    if pred_data.shape[0] == target_size:
        return pred_data

    max_len = max(pred_data.shape[0], target_size)
    min_len = min(pred_data.shape[0], target_size)

    coarse_var = pred if pred_data.shape[0] < target_size else var
    fine_var = var if pred_data.shape[0] < target_size else pred
    source_levels = [var_to_level[coarse_var]] if coarse_var in var_to_level else _get_level_names(min_len, data)
    target_levels = [var_to_level[fine_var]] if fine_var in var_to_level else _get_level_names(max_len, data)

    idx_array = None
    for s in source_levels:
        for t in target_levels:
            idx_name = f"{s}_idx_{t}"
            if idx_name in data:
                idx_array = data[idx_name]
                break
        if idx_array is not None:
            break

    if idx_array is None:
        idx_name_1 = f"idx_{pred}"
        idx_array = data.get(idx_name_1)
        if idx_array is None or idx_array.shape[0] != max_len:
            possible_idx = [
                k for k, val in data.items()
                if k.startswith("idx_") and hasattr(val, "shape") and len(val.shape) > 0
                and val.shape[0] == max_len and jnp.max(val) < min_len
            ]
            if not possible_idx:
                possible_idx = [
                    k for k, val in data.items()
                    if "idx" in k and hasattr(val, "shape") and len(val.shape) > 0
                    and val.shape[0] == max_len and jnp.max(val) < min_len
                ]
                if not possible_idx:
                    raise ValueError(
                        f"Shape mismatch: {var} ({target_size}) vs {pred} ({pred_data.shape[0]}). "
                        f"No bridging index found of length {max_len}."
                    )
            idx_array = data[possible_idx[0]]

    if pred_data.shape[0] < target_size:
        return pred_data[idx_array]
    else:
        sum_pred = jax.ops.segment_sum(pred_data, idx_array, num_segments=target_size)
        count = jax.ops.segment_sum(jnp.ones_like(pred_data), idx_array, num_segments=target_size)
        mean_pred = sum_pred / jnp.where(count > 0, count, 1.0)
        return mean_pred
