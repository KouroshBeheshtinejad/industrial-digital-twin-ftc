from dataclasses import dataclass
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from python.controllers.fault_tolerant_controller import FaultTolerantController
from python.controllers.pi_controller import PIController
from python.estimation.state_estimator import ModelBasedStateEstimator
from python.fault_detection.fault_detector import FaultDetector
from python.plant_model.cstr_model import cstr_dynamics


INITIAL_STATE = np.array([
    64.50615287,
    355.28373576,
    305.90638203,
])


@dataclass(frozen=True)
class IntegratedScenario:
    name: str
    simulation_time: float = 80.0
    sample_time: float = 0.05
    setpoint: float = 356.0
    fault_time: float = 30.0
    sensor_bias: float = 0.0
    sensor_fault_enabled: bool = False
    actuator_stuck_enabled: bool = False
    stuck_flow: float = 0.5
    measurement_noise_std: float = 0.05
    random_seed: int = 2026
    temperature_threshold: float = 0.75
    actuator_threshold: float = 0.01
    persistence_samples: int = 5

    def validate(self, params) -> None:
        values = (
            self.simulation_time,
            self.sample_time,
            self.setpoint,
            self.fault_time,
            self.sensor_bias,
            self.stuck_flow,
            self.measurement_noise_std,
            self.temperature_threshold,
            self.actuator_threshold,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Scenario numeric parameters must be finite.")
        if self.simulation_time <= 0:
            raise ValueError("Simulation time must be positive.")
        if self.sample_time <= 0:
            raise ValueError("Sample time must be positive.")
        if self.fault_time < 0 or self.fault_time > self.simulation_time:
            raise ValueError("Fault time must lie within the simulation.")
        if self.measurement_noise_std < 0:
            raise ValueError("Measurement noise standard deviation must be non-negative.")
        if self.persistence_samples < 1:
            raise ValueError("Persistence samples must be at least one.")
        if not params.minimum_coolant_flow <= self.stuck_flow <= params.maximum_coolant_flow:
            raise ValueError("Stuck flow must be within actuator limits.")


@dataclass
class SimulationResult:
    scenario: IntegratedScenario
    time: np.ndarray
    true_states: np.ndarray
    estimated_states: np.ndarray
    measured_temperature: np.ndarray
    sensor_bias: np.ndarray
    controller_command: np.ndarray
    actual_coolant_flow: np.ndarray
    temperature_residual: np.ndarray
    actuator_residual: np.ndarray
    sensor_alarm: np.ndarray
    actuator_alarm: np.ndarray
    diagnosed_fault: list[str]
    ftc_mode: list[str]
    control_temperature: np.ndarray


def default_scenarios() -> list[IntegratedScenario]:
    return [
        IntegratedScenario(name="healthy"),
        IntegratedScenario(
            name="sensor_bias",
            sensor_bias=2.0,
            sensor_fault_enabled=True,
        ),
        IntegratedScenario(
            name="actuator_stuck",
            actuator_stuck_enabled=True,
            stuck_flow=0.5,
        ),
        IntegratedScenario(
            name="simultaneous_faults",
            sensor_bias=2.0,
            sensor_fault_enabled=True,
            actuator_stuck_enabled=True,
            stuck_flow=0.5,
        ),
    ]


def run_integrated_simulation(
    params,
    scenario: IntegratedScenario,
    actual_params=None,
) -> SimulationResult:
    scenario.validate(params)
    if actual_params is None:
        actual_params = params
    number_of_steps = int(round(scenario.simulation_time / scenario.sample_time))
    time = np.linspace(0.0, scenario.simulation_time, number_of_steps + 1)
    rng = np.random.default_rng(scenario.random_seed)

    true_states = np.zeros((number_of_steps + 1, 3), dtype=float)
    estimated_states = np.zeros((number_of_steps + 1, 3), dtype=float)
    measured_temperature = np.zeros(number_of_steps + 1, dtype=float)
    sensor_bias = np.zeros(number_of_steps + 1, dtype=float)
    controller_command = np.zeros(number_of_steps + 1, dtype=float)
    actual_coolant_flow = np.zeros(number_of_steps + 1, dtype=float)
    temperature_residual = np.zeros(number_of_steps + 1, dtype=float)
    actuator_residual = np.zeros(number_of_steps + 1, dtype=float)
    sensor_alarm = np.zeros(number_of_steps + 1, dtype=bool)
    actuator_alarm = np.zeros(number_of_steps + 1, dtype=bool)
    diagnosed_fault = ["none"] * (number_of_steps + 1)
    ftc_mode = ["normal"] * (number_of_steps + 1)
    control_temperature = np.zeros(number_of_steps + 1, dtype=float)

    true_states[0] = INITIAL_STATE
    estimated_states[0] = INITIAL_STATE

    pi = PIController(
        setpoint=scenario.setpoint,
        minimum_output=params.minimum_coolant_flow,
        maximum_output=params.maximum_coolant_flow,
        coolant_bias=params.nominal_coolant_flow,
    )
    ftc = FaultTolerantController(pi)
    estimator = ModelBasedStateEstimator(params=params, initial_state=INITIAL_STATE)
    detector = FaultDetector(
        temperature_threshold=scenario.temperature_threshold,
        actuator_threshold=scenario.actuator_threshold,
        persistence_samples=scenario.persistence_samples,
    )

    initial_measurement = (
        INITIAL_STATE[1]
        + rng.normal(0.0, scenario.measurement_noise_std)
    )
    measured_temperature[0] = initial_measurement
    control_temperature[0] = initial_measurement

    initial_decision = ftc.step(
        measured_temperature=initial_measurement,
        estimated_temperature=estimator.estimated_temperature,
        sample_time=scenario.sample_time,
    )
    command = float(initial_decision.actuator_command)
    last_pi_command = command
    controller_command[0] = command
    actual_coolant_flow[0] = command

    for index in range(number_of_steps):
        interval_command = command

        if time[index] >= scenario.fault_time and scenario.actuator_stuck_enabled:
            actual_flow = scenario.stuck_flow
        else:
            actual_flow = interval_command

        actual_flow = float(np.clip(
            actual_flow,
            params.minimum_coolant_flow,
            params.maximum_coolant_flow,
        ))

        solution = solve_ivp(
            fun=lambda sample, state: cstr_dynamics(
                sample, state, actual_flow, actual_params
            ),
            t_span=(0.0, scenario.sample_time),
            y0=true_states[index],
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(f"CSTR integration failed: {solution.message}")
        if not np.all(np.isfinite(solution.y[:, -1])):
            raise RuntimeError("CSTR integration produced non-finite state values.")
        true_states[index + 1] = solution.y[:, -1]

        injected_bias = 0.0
        if time[index + 1] >= scenario.fault_time and scenario.sensor_fault_enabled:
            injected_bias = scenario.sensor_bias
        measurement = float(
            true_states[index + 1, 1]
            + injected_bias
            + rng.normal(0.0, scenario.measurement_noise_std)
        )
        measured_temperature[index + 1] = measurement
        sensor_bias[index + 1] = injected_bias

        estimator.predict(
            actual_coolant_flow=actual_flow,
            sample_time=scenario.sample_time,
        )
        estimated_states[index + 1] = estimator.estimated_state

        detection = detector.update(
            measured_temperature=measurement,
            estimated_temperature=estimator.estimated_temperature,
            commanded_flow=interval_command,
            actual_flow=actual_flow,
        )
        temperature_residual[index + 1] = detection.temperature_residual
        actuator_residual[index + 1] = detection.actuator_residual
        sensor_alarm[index + 1] = detection.sensor_alarm
        actuator_alarm[index + 1] = detection.actuator_alarm
        diagnosed_fault[index + 1] = detection.fault_type.value

        decision = ftc.step(
            measured_temperature=measurement,
            estimated_temperature=estimator.estimated_temperature,
            sample_time=scenario.sample_time,
            fault_detected=detection.fault_detected,
            fault_type=detection.fault_type.value,
            control_authority_available=not detection.actuator_alarm,
        )
        ftc_mode[index + 1] = decision.mode.value
        control_temperature[index + 1] = (
            float(measurement)
            if decision.control_temperature is None
            else float(decision.control_temperature)
        )

        if decision.actuator_command is not None:
            command = float(np.clip(
                decision.actuator_command,
                params.minimum_coolant_flow,
                params.maximum_coolant_flow,
            ))
            last_pi_command = command
        else:
            command = last_pi_command

        controller_command[index + 1] = interval_command
        actual_coolant_flow[index + 1] = actual_flow

    result = SimulationResult(
        scenario=scenario,
        time=time,
        true_states=true_states,
        estimated_states=estimated_states,
        measured_temperature=measured_temperature,
        sensor_bias=sensor_bias,
        controller_command=controller_command,
        actual_coolant_flow=actual_coolant_flow,
        temperature_residual=temperature_residual,
        actuator_residual=actuator_residual,
        sensor_alarm=sensor_alarm,
        actuator_alarm=actuator_alarm,
        diagnosed_fault=diagnosed_fault,
        ftc_mode=ftc_mode,
        control_temperature=control_temperature,
    )
    validate_result(result)
    return result


def validate_result(result: SimulationResult) -> None:
    arrays = (
        result.time,
        result.true_states,
        result.estimated_states,
        result.measured_temperature,
        result.sensor_bias,
        result.controller_command,
        result.actual_coolant_flow,
        result.temperature_residual,
        result.actuator_residual,
        result.control_temperature,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("Simulation produced a non-finite result.")
    expected = len(result.time)
    lengths = (
        len(result.measured_temperature),
        len(result.controller_command),
        len(result.actual_coolant_flow),
        len(result.temperature_residual),
        len(result.actuator_residual),
        len(result.diagnosed_fault),
        len(result.ftc_mode),
        len(result.control_temperature),
    )
    if not all(length == expected for length in lengths):
        raise RuntimeError("Simulation result arrays have inconsistent lengths.")


def calculate_metrics(result: SimulationResult, params) -> dict:
    time = result.time
    temperature = result.true_states[:, 1]
    error = result.scenario.setpoint - temperature
    actual_flows = result.actual_coolant_flow

    alarm_matches = np.zeros(len(time), dtype=bool)
    if result.scenario.sensor_fault_enabled and result.scenario.actuator_stuck_enabled:
        alarm_matches = result.sensor_alarm & result.actuator_alarm
        detected_fault = "sensor_and_actuator"
    elif result.scenario.sensor_fault_enabled:
        alarm_matches = result.sensor_alarm
        detected_fault = "sensor"
    elif result.scenario.actuator_stuck_enabled:
        alarm_matches = result.actuator_alarm
        detected_fault = "actuator"
    else:
        detected_fault = "none"

    detection_indices = np.flatnonzero(alarm_matches)
    if len(detection_indices):
        detection_time = float(time[detection_indices[0]])
        fault_detection_delay = float(detection_time - result.scenario.fault_time)
    else:
        detection_time = float("nan")
        fault_detection_delay = float("nan")

    false_alarm_mask = time < result.scenario.fault_time
    false_alarms = int(np.count_nonzero(
        (result.sensor_alarm | result.actuator_alarm) & false_alarm_mask
    ))

    mode_counts = {
        mode: float(np.count_nonzero(
            np.asarray(result.ftc_mode) == mode
        ) * result.scenario.sample_time)
        for mode in (
            "normal",
            "sensor_reconfiguration",
            "actuator_reconfiguration",
            "safe_mode",
        )
    }

    return {
        "scenario": result.scenario.name,
        "IAE": float(np.trapezoid(np.abs(error), time)),
        "ISE": float(np.trapezoid(error * error, time)),
        "final_temperature_error_K": float(error[-1]),
        "maximum_temperature_K": float(np.max(temperature)),
        "minimum_temperature_K": float(np.min(temperature)),
        "maximum_coolant_flow": float(np.max(actual_flows)),
        "minimum_coolant_flow": float(np.min(actual_flows)),
        "fault_detection_time_min": detection_time,
        "fault_detection_delay_min": fault_detection_delay,
        "false_alarms_before_fault": false_alarms,
        "mode_normal_min": mode_counts["normal"],
        "mode_sensor_reconfiguration_min": mode_counts["sensor_reconfiguration"],
        "mode_actuator_reconfiguration_min": mode_counts["actuator_reconfiguration"],
        "mode_safe_min": mode_counts["safe_mode"],
        "actuator_within_limits": bool(
            np.all(actual_flows >= params.minimum_coolant_flow - 1e-9)
            and np.all(actual_flows <= params.maximum_coolant_flow + 1e-9)
        ),
        "detected_fault": detected_fault,
    }
