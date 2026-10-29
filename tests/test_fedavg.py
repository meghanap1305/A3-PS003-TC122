import copy

import pytest
import torch

from federated.fedavg import fedavg
from src.model import TrafficMLP


def _state(value, extra=None):
    s = {"w": torch.full((2, 2), float(value)), "b": torch.full((2,), float(value))}
    if extra is not None:
        s.update(extra)
    return s


def test_weighted_arithmetic_exact_example():
    out = fedavg([_state(2), _state(4), _state(6)], [100, 200, 300])
    expected = (100 * 2 + 200 * 4 + 300 * 6) / 600  # 4.6667
    assert torch.allclose(out["w"], torch.full((2, 2), expected))
    assert torch.allclose(out["b"], torch.full((2,), expected))


def test_not_unweighted_mean():
    out = fedavg([_state(2), _state(4), _state(6)], [100, 200, 300])
    assert not torch.allclose(out["w"], torch.full((2, 2), 4.0))  # unweighted would be 4.0


def test_multi_parameter_different_values_per_tensor():
    a = {"x": torch.tensor([1.0, 10.0]), "y": torch.tensor([[0.0]])}
    b = {"x": torch.tensor([3.0, 20.0]), "y": torch.tensor([[8.0]])}
    out = fedavg([a, b], [1, 3])
    assert torch.allclose(out["x"], torch.tensor([2.5, 17.5]))
    assert torch.allclose(out["y"], torch.tensor([[6.0]]))


def test_empty_fails():
    with pytest.raises(ValueError):
        fedavg([], [])


def test_length_mismatch_fails():
    with pytest.raises(ValueError):
        fedavg([_state(1), _state(2)], [10])


@pytest.mark.parametrize("bad", [0, -5, 2.5, True, None])
def test_invalid_counts_fail(bad):
    with pytest.raises(ValueError):
        fedavg([_state(1), _state(2)], [10, bad])


def test_key_mismatch_fails():
    with pytest.raises(ValueError):
        fedavg([_state(1), {"w": torch.zeros(2, 2)}], [1, 1])


def test_shape_mismatch_fails():
    bad = {"w": torch.zeros(3, 3), "b": torch.zeros(2)}
    with pytest.raises(ValueError):
        fedavg([_state(1), bad], [1, 1])


def test_non_finite_fails():
    bad = _state(1)
    bad["w"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        fedavg([_state(1), bad], [1, 1])


def test_inputs_not_mutated_and_output_not_aliased():
    s1, s2 = _state(2), _state(4)
    before = copy.deepcopy([s1, s2])
    out = fedavg([s1, s2], [1, 1])
    for orig, snap in zip([s1, s2], before):
        for k in orig:
            assert torch.equal(orig[k], snap[k])
    out["w"].add_(100)                      # modifying output must not touch inputs
    assert torch.equal(s1["w"], before[0]["w"])


def test_non_float_entries():
    s1 = _state(1, {"steps": torch.tensor(5)})
    s2 = _state(3, {"steps": torch.tensor(5)})
    out = fedavg([s1, s2], [1, 1])
    assert out["steps"].item() == 5 and out["steps"].dtype == torch.int64

    s3 = _state(3, {"steps": torch.tensor(9)})
    with pytest.raises(ValueError):
        fedavg([s1, s3], [1, 1])
    assert fedavg([s1, s3], [1, 1], non_float_policy="first")["steps"].item() == 5


def test_result_loads_into_traffic_mlp():
    torch.manual_seed(0)
    models = [TrafficMLP() for _ in range(3)]
    states = [m.state_dict() for m in models]
    agg = fedavg(states, [10, 20, 30])
    target = TrafficMLP()
    target.load_state_dict(agg)             # raises if incompatible
    assert target(torch.randn(4, 12, 2)).shape == (4, 12, 2)
