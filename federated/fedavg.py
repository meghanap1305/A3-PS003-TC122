"""Weighted Federated Averaging (FedAvg) over PyTorch state dictionaries.

Public interface (agreed in FEDERATED_COMMON_CONTRACT.md, section 6):

    fedavg(client_states, client_num_examples) -> aggregated state_dict

    w_global = sum_k (n_k / sum_j n_j) * w_k
"""
from collections import OrderedDict
from numbers import Integral

import torch


def _validate_counts(client_states, client_num_examples):
    if client_states is None or len(client_states) == 0:
        raise ValueError("fedavg: client_states is empty; nothing to aggregate.")
    if client_num_examples is None or len(client_states) != len(client_num_examples):
        raise ValueError(
            "fedavg: client_states and client_num_examples must have the same "
            f"length (got {len(client_states)} and "
            f"{None if client_num_examples is None else len(client_num_examples)})."
        )

    counts = []
    for i, n in enumerate(client_num_examples):
        # bool is a subclass of int, but True/False is never a valid count.
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise ValueError(
                f"fedavg: example count for client {i} must be an integer, got {n!r}."
            )
        if n <= 0:
            raise ValueError(
                f"fedavg: example count for client {i} must be > 0, got {n}. "
                "Filter out empty clients before aggregating."
            )
        counts.append(int(n))
    return counts


def _validate_states(client_states):
    reference = client_states[0]
    ref_keys = list(reference.keys())
    ref_key_set = set(ref_keys)

    for i, state in enumerate(client_states):
        keys = set(state.keys())
        if keys != ref_key_set:
            raise ValueError(
                f"fedavg: client {i} state keys differ from client 0. "
                f"Missing: {sorted(ref_key_set - keys)}; "
                f"unexpected: {sorted(keys - ref_key_set)}."
            )
        for key in ref_keys:
            value, ref_value = state[key], reference[key]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"fedavg: client {i}, entry '{key}' is not a tensor "
                    f"({type(value).__name__})."
                )
            if value.shape != ref_value.shape:
                raise ValueError(
                    f"fedavg: shape mismatch for '{key}': client {i} has "
                    f"{tuple(value.shape)}, client 0 has {tuple(ref_value.shape)}."
                )
            if value.dtype != ref_value.dtype:
                raise ValueError(
                    f"fedavg: dtype mismatch for '{key}': client {i} has "
                    f"{value.dtype}, client 0 has {ref_value.dtype}."
                )
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(
                    f"fedavg: client {i}, entry '{key}' contains NaN/Inf."
                )
    return ref_keys


def fedavg(client_states, client_num_examples, non_float_policy="require_equal"):
    """Sample-count-weighted average of client state dicts.

    Args:
        client_states: list of state_dicts (one per client). Never mutated.
        client_num_examples: list of positive ints (unique local examples).
        non_float_policy: how to handle integer/bool entries (e.g. BatchNorm's
            ``num_batches_tracked``). Averaging them is meaningless, so:
              "require_equal" (default): all clients must hold identical values,
                                          otherwise raise ValueError.
              "first": copy the value from the first client.
            TrafficMLP has no such entries today; this is future-proofing.

    Returns:
        OrderedDict state_dict, accepted by ``model.load_state_dict``.
        Tensors are new objects (not views of any client's tensors).
    """
    if non_float_policy not in ("require_equal", "first"):
        raise ValueError("non_float_policy must be 'require_equal' or 'first'.")

    counts = _validate_counts(client_states, client_num_examples)
    keys = _validate_states(client_states)

    total = float(sum(counts))
    weights = [n / total for n in counts]

    aggregated = OrderedDict()
    for key in keys:
        reference = client_states[0][key]

        if reference.is_floating_point():
            # Accumulate in float64 for accuracy, then cast back.
            acc = torch.zeros_like(reference, dtype=torch.float64)
            for state, w in zip(client_states, weights):
                acc += state[key].detach().to(device=acc.device, dtype=torch.float64) * w
            aggregated[key] = acc.to(reference.dtype)
        else:
            if non_float_policy == "require_equal":
                for i, state in enumerate(client_states[1:], start=1):
                    if not torch.equal(state[key], reference):
                        raise ValueError(
                            f"fedavg: non-floating entry '{key}' differs between "
                            f"client 0 and client {i}; cannot average it. "
                            "Use non_float_policy='first' to copy client 0's value."
                        )
            aggregated[key] = reference.detach().clone()

    return aggregated
