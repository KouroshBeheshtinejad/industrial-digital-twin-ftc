
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from python.controllers.pi_controller import PIController


class FTCMode(Enum):
    NORMAL = "normal"
    SENSOR_RECONFIGURATION = "sensor_reconfiguration"
    ACTUATOR_RECONFIGURATION = "actuator_reconfiguration"
    SAFE_MODE = "safe_mode"


@dataclass
class FTCDecision:
    mode: FTCMode
    actuator_command: Optional[float]
    control_temperature: Optional[float]
    fault_detected: bool
    control_authority_available: bool
    message: str


class FaultTolerantController:
    """
    Supervisory fault-tolerant controller.

    Selects the appropriate control mode based on fault diagnosis
    and actuator control authority.
    """

    def __init__(self, pi_controller: PIController):
        self.pi_controller = pi_controller
        self.mode = FTCMode.NORMAL
        self.last_command = None

    def reset(self):
        self.pi_controller.reset()
        self.mode = FTCMode.NORMAL
        self.last_command = None

    def step(
        self,
        measured_temperature: float,
        estimated_temperature: float,
        sample_time: float,
        fault_detected: bool = False,
        fault_type: Optional[str] = None,
        control_authority_available: bool = True,
    ) -> FTCDecision:

        if sample_time <= 0:
            raise ValueError("sample_time must be positive.")

        if not fault_detected:
            self.mode = FTCMode.NORMAL

            command = self.pi_controller.compute(
                temperature=measured_temperature,
                sample_time=sample_time,
            )

            self.last_command = command

            return FTCDecision(
                mode=self.mode,
                actuator_command=command,
                control_temperature=measured_temperature,
                fault_detected=False,
                control_authority_available=True,
                message="Normal closed-loop control.",
            )

        normalized_fault = (
            fault_type.lower().strip()
            if fault_type is not None
            else "unknown"
        )

        if normalized_fault == "sensor":

            self.mode = FTCMode.SENSOR_RECONFIGURATION

            command = self.pi_controller.compute(
                temperature=estimated_temperature,
                sample_time=sample_time,
            )

            self.last_command = command

            return FTCDecision(
                mode=self.mode,
                actuator_command=command,
                control_temperature=estimated_temperature,
                fault_detected=True,
                control_authority_available=True,
                message=(
                    "Sensor fault detected. "
                    "Controller uses estimated temperature."
                ),
            )

        if normalized_fault == "actuator":

            if control_authority_available:
                self.mode = FTCMode.ACTUATOR_RECONFIGURATION

                return FTCDecision(
                    mode=self.mode,
                    actuator_command=None,
                    control_temperature=estimated_temperature,
                    fault_detected=True,
                    control_authority_available=True,
                    message=(
                        "Actuator fault detected. "
                        "Alternative actuator reconfiguration is required."
                    ),
                )

            self.mode = FTCMode.SAFE_MODE

            return FTCDecision(
                mode=self.mode,
                actuator_command=None,
                control_temperature=estimated_temperature,
                fault_detected=True,
                control_authority_available=False,
                message=(
                    "Actuator authority is unavailable. "
                    "Normal temperature regulation cannot be guaranteed."
                ),
            )

        self.mode = FTCMode.SAFE_MODE

        return FTCDecision(
            mode=self.mode,
            actuator_command=None,
            control_temperature=estimated_temperature,
            fault_detected=True,
            control_authority_available=False,
            message=(
                f"Unsupported fault type: {normalized_fault}. "
                "System entered safe mode."
            ),
        )
