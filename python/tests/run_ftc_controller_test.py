
from python.controllers.pi_controller import PIController
from python.controllers.fault_tolerant_controller import (
    FaultTolerantController,
    FTCMode,
)


def main():

    pi = PIController()
    ftc = FaultTolerantController(pi)

    dt = 0.05

    print("\n========== FTC CONTROLLER TEST ==========")

    # Case 1: Healthy operation
    decision = ftc.step(
        measured_temperature=355.0,
        estimated_temperature=355.0,
        sample_time=dt,
    )

    print("\nHealthy operation:")
    print(decision)

    assert decision.mode == FTCMode.NORMAL
    assert decision.actuator_command is not None

    # Case 2: Sensor fault
    decision = ftc.step(
        measured_temperature=357.0,
        estimated_temperature=355.0,
        sample_time=dt,
        fault_detected=True,
        fault_type="sensor",
    )

    print("\nSensor fault:")
    print(decision)

    assert decision.mode == FTCMode.SENSOR_RECONFIGURATION
    assert decision.actuator_command is not None
    assert decision.control_temperature == 355.0

    # Case 3: Actuator fault with available authority
    decision = ftc.step(
        measured_temperature=355.0,
        estimated_temperature=355.0,
        sample_time=dt,
        fault_detected=True,
        fault_type="actuator",
        control_authority_available=True,
    )

    print("\nActuator fault with alternative authority:")
    print(decision)

    assert decision.mode == FTCMode.ACTUATOR_RECONFIGURATION
    assert decision.actuator_command is None

    # Case 4: Actuator fault without available authority
    decision = ftc.step(
        measured_temperature=355.0,
        estimated_temperature=355.0,
        sample_time=dt,
        fault_detected=True,
        fault_type="actuator",
        control_authority_available=False,
    )

    print("\nActuator fault without alternative authority:")
    print(decision)

    assert decision.mode == FTCMode.SAFE_MODE
    assert decision.actuator_command is None

    print("\nAll FTC controller tests passed.")


if __name__ == "__main__":
    main()
