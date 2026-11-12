from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from train_branch_publisher import choose, PROTOCOL


def candidate(identity, family, regularization, loss, validation, success=True):
    return {"id": identity, "family": family, "lambda": regularization,
        "optimizer": {"success": success, "loss": loss}, "metrics": {"validation": {"1": {"brier": validation}}}}


def test_initial_seed_chosen_on_training_not_validation():
    values = [candidate("a", "branch", .001, 1., .3), candidate("b", "branch", .001, 2., .01),
              candidate("c", "branch", .03, .9, .2)]
    assert choose(values)["branch"] == "c"


def test_unconverged_candidate_cannot_win_and_classifier_uses_same_primary_target():
    values = [candidate("a", "branch", .001, 1., .3), candidate("b", "branch", .03, .01, .01, False),
              candidate("c", "logistic", None, 100, .1), candidate("d", "logistic", None, 0., .2)]
    assert choose(values) == {"branch": "a", "logistic": "c"}
    assert PROTOCOL["threshold"] == .5 and PROTOCOL["bn_max_iter"] == 500
