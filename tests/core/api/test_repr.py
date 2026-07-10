# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for test-function string representations."""

from surfaces.test_functions.algebraic import SphereFunction
from surfaces.test_functions.algebraic.multi_objective import ZDT1


def test_repr_shows_dimension_and_default_objective():
    func = SphereFunction(n_dim=2)

    assert repr(func) == "SphereFunction(n_dim=2, objective='minimize')"


def test_repr_shows_explicit_objective():
    func = SphereFunction(n_dim=2, objective="maximize")

    assert repr(func) == "SphereFunction(n_dim=2, objective='maximize')"


def test_repr_shows_per_objective_directions():
    func = ZDT1(n_dim=3, objective=["minimize", "maximize"])

    assert repr(func) == "ZDT1(n_dim=3, objective=['minimize', 'maximize'])"
