
import numpy as np

from python.plant_model.cstr_model import (
    load_parameters,
    simulate_cstr
)
from python.estimation.state_estimator import ModelBasedStateEstimator


def main():
    params = load_parameters(
        "configs/plant_parameters/cstr_nominal.json"
    )

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203
    ])

    sample_time = 0.05
    simulation_time = 10.0
    coolant_flow = 0.1

    estimator = ModelBasedStateEstimator(
        params=params,
        initial_state=initial_state
    )

    true_state = initial_state.copy()

    maximum_errors = np.zeros(3)

    number_of_steps = int(simulation_time / sample_time)

    for _ in range(number_of_steps):
        plant_result = simulate_cstr(
            params=params,
            initial_state=true_state,
            coolant_flow=coolant_flow,
            simulation_time=sample_time
        )

        true_state = plant_result[-1] if isinstance(
            plant_result, np.ndarray
        ) else plant_result.y[:, -1]

        estimated_state = estimator.predict(
            actual_coolant_flow=coolant_flow,
            sample_time=sample_time
        )

        errors = np.abs(true_state - estimated_state)
        maximum_errors = np.maximum(maximum_errors, errors)

    print("\n========== STATE ESTIMATOR VALIDATION ==========")

    print(f"Simulation duration: {simulation_time:.2f} min")
    print(f"Sample time: {sample_time:.4f} min")

    print("\nFinal true state:")
    print(true_state)

    print("\nFinal estimated state:")
    print(estimator.estimated_state)

    print("\nMaximum absolute estimation errors:")
    print(f"Concentration: {maximum_errors[0]:.10f} mol/m^3")
    print(f"Reactor temperature: {maximum_errors[1]:.10f} K")
    print(f"Jacket temperature: {maximum_errors[2]:.10f} K")


if __name__ == "__main__":
    main()
