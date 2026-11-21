
from pathlib import Path

import numpy as np
import pytest

from python.plant_model.cstr_model import (
    load_parameters,
    reaction_rate_constant,
    cstr_dynamics,
    simulate_cstr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)


@pytest.fixture
def params():
    return load_parameters(CONFIG_PATH)


def test_parameter_loading(params):
    assert params.reactor_volume > 0
    assert params.jacket_volume > 0
    assert params.feed_flow_rate > 0
    assert params.heat_transfer_coefficient > 0


def test_reaction_rate_is_positive(params):
    rate = reaction_rate_constant(350.0, params)

    assert np.isfinite(rate)
    assert rate > 0


def test_reaction_rate_increases_with_temperature(params):
    low_rate = reaction_rate_constant(340.0, params)
    high_rate = reaction_rate_constant(360.0, params)

    assert high_rate > low_rate


def test_dynamics_returns_three_states(params):
    state = np.array([100.0, 350.0, 300.0])

    derivative = cstr_dynamics(
        0.0,
        state,
        params.nominal_coolant_flow,
        params
    )

    assert derivative.shape == (3,)
    assert np.all(np.isfinite(derivative))


def test_simulation_output(params):
    initial_state = np.array([100.0, 350.0, 300.0])

    solution = simulate_cstr(
        params,
        initial_state,
        params.nominal_coolant_flow,
        simulation_time=5.0,
        number_of_points=101
    )

    assert solution.success
    assert solution.y.shape == (3, 101)
    assert np.all(np.isfinite(solution.y))


def test_invalid_coolant_flow_is_rejected(params):
    initial_state = np.array([100.0, 350.0, 300.0])

    with pytest.raises(ValueError):
        simulate_cstr(
            params,
            initial_state,
            coolant_flow=1.0,
            simulation_time=5.0
        )


def test_invalid_initial_state_is_rejected(params):
    with pytest.raises(ValueError):
        simulate_cstr(
            params,
            np.array([-10.0, 350.0, 300.0]),
            params.nominal_coolant_flow
        )
