from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from verify_typed_online_equivalence import project_mass


def variables():
    return [{"kind": "information_state", "id": "a"}, {"kind": "origin_hypothesis", "id": "doc"},
            {"kind": "authorization_state", "id": "a"}]


def test_reordering_preserves_distinct_channels_for_same_node():
    vs = variables()
    mass = np.arange(8, dtype=float) / 28
    actual = project_mass(mass, vs, [vs[2], vs[0], vs[1]])
    # Independent binary-string mapping, not the implementation's bit formula.
    expected = np.zeros(8)
    for i, p in enumerate(mass):
        old = list(reversed(f"{i:03b}"))
        new = [old[2], old[0], old[1]]
        expected[int("".join(reversed(new)), 2)] += p
    np.testing.assert_array_equal(actual, expected)


def test_unallocated_future_states_must_have_zero_mass():
    vs = variables()
    mass = np.array([.2, .3, .1, .4, 0, 0, 0, 0])
    np.testing.assert_array_equal(project_mass(mass, vs, vs[:2]), mass[:4])
    mass[4] = .01
    with pytest.raises(AssertionError):
        project_mass(mass, vs, vs[:2])


def test_identity_and_metadata_mismatches_cannot_be_hidden_by_projection():
    vs = variables()
    with pytest.raises(ValueError, match="duplicate"):
        project_mass(np.zeros(8), vs, [vs[0], vs[0]])
    with pytest.raises(ValueError, match="metadata"):
        project_mass(np.zeros(8), vs, [{**vs[0], "role": "executor"}, *vs[1:]])
