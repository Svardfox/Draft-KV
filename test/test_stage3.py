import pytest
import torch

from draft_kv.train.draft_kv_mc_training_data import deranged_no_harm_loss
from script.draft_kv.train_stage3 import (
    _selection_eligibility,
    _selection_score,
    TRAIN_PROTOCOL,
)
from script.draft_kv.eval_stage3 import SUPPORTED_TRAIN_PROTOCOLS


def test_protection_only_penalizes_excess_harm_and_detaches_zero():
    zero = torch.tensor([1., 1., 1., 1.], requires_grad=True)
    deranged = torch.tensor([0.2, 1.05, 1.1, 1.4], requires_grad=True)
    losses = deranged_no_harm_loss(deranged, zero, tolerance=0.125)
    assert torch.allclose(losses, torch.tensor([0., 0., 0., 0.275]))
    losses.sum().backward()
    assert zero.grad is None
    assert torch.equal(deranged.grad, torch.tensor([0., 0., 0., 1.]))


def test_improvements_cannot_cancel_other_examples_harm():
    loss = deranged_no_harm_loss(
        torch.tensor([0., 2.]), torch.ones(2), tolerance=0.
    ).mean()
    assert loss.item() == 0.5


@pytest.mark.parametrize("tolerance", [-1., float("nan"), float("inf")])
def test_protection_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        deranged_no_harm_loss(torch.ones(2), torch.ones(2), tolerance=tolerance)


def test_protection_rejects_broadcasting():
    with pytest.raises(ValueError):
        deranged_no_harm_loss(torch.ones(2), torch.ones(1), tolerance=0.)


def test_selection_prioritizes_accuracy_without_zero_or_deranged_nll_filter():
    calibration = {
        "conditions": {"matched": {"accuracy": 0.7, "mean_nll": 2.}},
        "gaps": {
            "matched_minus_zero_gold_log_probability": -0.1,
            "matched_minus_zero_accuracy": 0.1,
        },
    }
    assert _selection_score(calibration) == 0.7
    assert _selection_eligibility(calibration, {"eligible": True})["eligible"]
    assert not _selection_eligibility(calibration, {"eligible": False})["eligible"]
    assert TRAIN_PROTOCOL in SUPPORTED_TRAIN_PROTOCOLS


def test_zero_weight_is_matched_ce_ablation():
    logits = torch.tensor([[0., 1.]], requires_grad=True)
    matched = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    deranged = torch.tensor([2.], requires_grad=True)
    loss = matched + 0. * deranged_no_harm_loss(
        deranged, torch.tensor([1.]), tolerance=0.1
    ).mean()
    expected = torch.autograd.grad(matched, logits, retain_graph=True)[0]
    loss.backward()
    assert torch.equal(logits.grad, expected)
    assert deranged.grad.item() == 0.
