import math
from dataclasses import replace

import numpy as np
import pytest

from python.controllers.pi_controller import PIController
from python.controllers.fault_tolerant_controller import (
    FaultTolerantController,
    FTCMode,
)
from python.fault_detection.fault_detector import FaultDetector, FaultType
from python.plant_model.cstr_model import load_parameters
from python.experiments.integrated_simulation import (
    IntegratedScenario,
    run_integrated_simulation,
)


CONFIG_PATH = "configs/plant_parameters/cstr_nominal.json"


def get_params():
    return load_parameters(CONFIG_PATH)


def test_healthy_run_has_no_false_alarm_and_finite_output():
    scenario = IntegratedScenario(
        name="healthy",
        simulation_time=20.0,
        fault_time=20.0,
    )
    result = run_integrated_simulation(get_params(), scenario)

    assert not np.any(result.sensor_alarm)
    assert not np.any(result.actuator_alarm)
    assert np.all(np.isfinite(result.true_states))
    assert np.all(np.isfinite(result.estimated_states))
    assert len(result.time) == len(result.true_states)


def test_sensor_bias_is_detected_and_uses_estimated_temperature():
    scenario = IntegratedScenario(
        name="sensor",
        simulation_time=36.0,
        fault_time=30.0,
        sensor_fault_enabled=True,
        sensor_bias=2.0,
    )
    result = run_integrated_simulation(get_params(), scenario)

    detection_indices = np.flatnonzero(result.sensor_alarm)
    assert len(detection_indices) > 0
    detected_at = int(detection_indices[0])
    assert result.ftc_mode[detected_at] == "sensor_reconfiguration"
    assert result.control_temperature[detected_at] == pytest.approx(
        result.estimated_states[detected_at, 1]
    )


def test_actuator_stuck_is_detected_and_enters_safe_mode():
    scenario = IntegratedScenario(
        name="actuator",
        simulation_time=36.0,
        fault_time=30.0,
        actuator_stuck_enabled=True,
        stuck_flow=0.5,
    )
    result = run_integrated_simulation(get_params(), scenario)

    detection_indices = np.flatnonzero(result.actuator_alarm)
    assert len(detection_indices) > 0
    detected_at = int(detection_indices[0])
    assert result.actual_coolant_flow[detected_at] == pytest.approx(0.5)
    assert "safe_mode" in result.ftc_mode[detected_at:]
    np.testing.assert_allclose(
        result.actuator_residual,
        result.controller_command - result.actual_coolant_flow,
        rtol=1e-12,
        atol=1e-12,
    )


def test_detector_persistence_and_reset():
    detector = FaultDetector(persistence_samples=3)

    alarms = []
    for _ in range(3):
        alarms.append(detector.update(357.0, 356.0, 0.1, 0.1).sensor_alarm)
    assert not alarms[0]
    assert not alarms[1]
    assert alarms[2]

    detector.reset()
    assert detector.sensor_exceedances == 0
    assert detector.actuator_exceedances == 0
    assert detector.detected_fault == FaultType.NONE


def test_detector_clears_stale_fault_when_residuals_return():
    detector = FaultDetector(persistence_samples=1)
    detected = detector.update(358.0, 356.0, 0.1, 0.1)
    assert detected.fault_type == FaultType.SENSOR

    clear = detector.update(356.0, 356.0, 0.1, 0.1)
    assert clear.fault_type == FaultType.NONE
    assert not clear.fault_detected


def test_simultaneous_faults_are_not_silently_classified_as_one():
    detector = FaultDetector(persistence_samples=1)
    result = detector.update(358.0, 356.0, 0.1, 0.5)
    assert result.sensor_alarm
    assert result.actuator_alarm
    assert result.fault_type == FaultType.SENSOR_AND_ACTUATOR

    pi = PIController()
    ftc = FaultTolerantController(pi)
    decision = ftc.step(
        measured_temperature=358.0,
        estimated_temperature=356.0,
        sample_time=0.05,
        fault_detected=True,
        fault_type="sensor_and_actuator",
        control_authority_available=False,
    )
    assert decision.mode == FTCMode.SAFE_MODE
    assert decision.actuator_command is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_detector_rejects_non_finite_values(bad_value):
    detector = FaultDetector()
    with pytest.raises(ValueError):
        detector.update(bad_value, 356.0, 0.1, 0.1)


def test_pi_setpoint_and_limits_reject_non_finite_values():
    controller = PIController()
    with pytest.raises(ValueError):
        controller.update_setpoint(float("nan"))
    with pytest.raises(ValueError):
        controller.update_limits(float("-inf"), 0.5)


def test_ua_mismatch_keeps_nominal_estimator_parameters():
    nominal = get_params()
    actual = replace(
        nominal,
        heat_transfer_coefficient=nominal.heat_transfer_coefficient * 1.10,
    )
    scenario = IntegratedScenario(
        name="ua_mismatch",
        simulation_time=36.0,
        fault_time=30.0,
        random_seed=7040,
    )
    result = run_integrated_simulation(
        nominal,
        scenario,
        actual_params=actual,
    )

    assert np.all(np.isfinite(result.true_states))
    assert np.all(np.isfinite(result.estimated_states))
    assert nominal.heat_transfer_coefficient == pytest.approx(50000.0)
    assert actual.heat_transfer_coefficient == pytest.approx(55000.0)
