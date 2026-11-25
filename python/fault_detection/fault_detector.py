
from dataclasses import dataclass
from enum import Enum
import math


class FaultType(Enum):
    NONE = "none"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    SENSOR_AND_ACTUATOR = "sensor_and_actuator"


@dataclass
class FaultDetectionResult:
    fault_detected: bool
    fault_type: FaultType
    temperature_residual: float
    actuator_residual: float
    sensor_alarm: bool
    actuator_alarm: bool


class FaultDetector:
    """
    Residual-based fault detector for the CSTR system.

    Sensor faults are detected using the difference between
    measured and estimated reactor temperature.

    Actuator faults are detected using the difference between
    commanded and actual coolant flow.
    """

    def __init__(
        self,
        temperature_threshold: float = 0.75,
        actuator_threshold: float = 0.01,
        persistence_samples: int = 5,
    ):
        if not math.isfinite(float(temperature_threshold)):
            raise ValueError("Temperature threshold must be finite.")
        if not math.isfinite(float(actuator_threshold)):
            raise ValueError("Actuator threshold must be finite.")
        if temperature_threshold <= 0:
            raise ValueError("Temperature threshold must be positive.")

        if actuator_threshold <= 0:
            raise ValueError("Actuator threshold must be positive.")

        if persistence_samples < 1:
            raise ValueError("Persistence samples must be at least one.")

        self.temperature_threshold = temperature_threshold
        self.actuator_threshold = actuator_threshold
        self.persistence_samples = persistence_samples

        self.sensor_exceedances = 0
        self.actuator_exceedances = 0

        self.detected_fault = FaultType.NONE

    def update(
        self,
        measured_temperature: float,
        estimated_temperature: float,
        commanded_flow: float,
        actual_flow: float,
    ) -> FaultDetectionResult:

        values = (
            measured_temperature,
            estimated_temperature,
            commanded_flow,
            actual_flow,
        )

        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("Detector inputs must be valid numbers.")

        if not all(map(lambda value: value == value and abs(value) != float("inf"), values)):
            raise ValueError("Detector inputs must be valid finite numbers.")

        temperature_residual = (
            measured_temperature - estimated_temperature
        )

        actuator_residual = commanded_flow - actual_flow

        if abs(actuator_residual) > self.actuator_threshold:
            self.actuator_exceedances += 1
        else:
            self.actuator_exceedances = 0

        if abs(temperature_residual) > self.temperature_threshold:
            self.sensor_exceedances += 1
        else:
            self.sensor_exceedances = 0

        actuator_alarm = (
            self.actuator_exceedances >= self.persistence_samples
        )

        sensor_alarm = (
            self.sensor_exceedances >= self.persistence_samples
        )

        if actuator_alarm and sensor_alarm:
            self.detected_fault = FaultType.SENSOR_AND_ACTUATOR
        elif actuator_alarm:
            self.detected_fault = FaultType.ACTUATOR
        elif sensor_alarm and not actuator_alarm:
            self.detected_fault = FaultType.SENSOR
        else:
            self.detected_fault = FaultType.NONE

        return FaultDetectionResult(
            fault_detected=self.detected_fault != FaultType.NONE,
            fault_type=self.detected_fault,
            temperature_residual=float(temperature_residual),
            actuator_residual=float(actuator_residual),
            sensor_alarm=sensor_alarm,
            actuator_alarm=actuator_alarm,
        )

    def reset(self):
        self.sensor_exceedances = 0
        self.actuator_exceedances = 0
        self.detected_fault = FaultType.NONE
