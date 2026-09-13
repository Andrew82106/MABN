"""Tiny CPU-testable loss component proposed for the v5 pair manifest."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


PAIR_WEIGHT = 0.25


def weighted_ranknet_loss(positive_risk, negative_risk, pair_weights):
    """Return the group-balanced mean softplus(risk_safe-risk_error)."""
    assert positive_risk.shape == negative_risk.shape == pair_weights.shape
    assert positive_risk.ndim == 1 and torch.all(pair_weights > 0)
    losses = F.softplus(negative_risk - positive_risk)
    return (losses * pair_weights).sum() / pair_weights.sum()


def combined_epoch_objective(weighted_bce_mean, positive_risk, negative_risk, pair_weights):
    return weighted_bce_mean + PAIR_WEIGHT * weighted_ranknet_loss(
        positive_risk, negative_risk, pair_weights
    )


def cpu_check():
    positive = torch.tensor([-1.0, 2.0, 0.5], requires_grad=True)
    negative = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    weights = torch.tensor([1.0, 1.0, 2.0])
    loss = weighted_ranknet_loss(positive, negative, weights)
    loss.backward()
    assert torch.all(positive.grad < 0), positive.grad
    assert torch.all(negative.grad > 0), negative.grad
    ordered = weighted_ranknet_loss(torch.tensor([2.0]), torch.tensor([-2.0]), torch.ones(1))
    reversed_ = weighted_ranknet_loss(torch.tensor([-2.0]), torch.tensor([2.0]), torch.ones(1))
    assert ordered < reversed_
    zero_bce = combined_epoch_objective(torch.tensor(0.0), torch.tensor([0.0]),
                                        torch.tensor([0.0]), torch.ones(1))
    expected = PAIR_WEIGHT * torch.log(torch.tensor(2.0))
    assert torch.allclose(zero_bce, expected)
    return {
        "status": "CPU_tiny_loss_check_passed", "pair_weight": PAIR_WEIGHT,
        "gradient_positive_sign": "negative", "gradient_negative_sign": "positive",
        "ordered_loss": float(ordered), "reversed_loss": float(reversed_),
        "equal_score_combined_loss_with_zero_BCE": float(zero_bce),
        "GPU_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check",))
    parser.parse_args()
    print(json.dumps(cpu_check(), indent=2))


if __name__ == "__main__":
    main()
