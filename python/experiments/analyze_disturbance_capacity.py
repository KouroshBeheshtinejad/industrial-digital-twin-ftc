from pathlib import Path

import numpy as np
from scipy.optimize import root

from python.plant_model.cstr_model import (
    load_parameters,
    cstr_dynamics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)


def find_equilibrium(params, coolant_flow, feed_temperature, initial_guess):
    from dataclasses import replace

    modified_params = replace(
        params,
        feed_temperature=feed_temperature,
    )

    result = root(
        lambda state: cstr_dynamics(
            0.0,
            state,
            coolant_flow,
            modified_params,
        ),
        initial_guess,
    )

    if not result.success:
        return None

    if np.linalg.norm(result.fun, ord=np.inf) > 1e-7:
        return None

    return result.x


def main():
    params = load_parameters(CONFIG_PATH)

    setpoint = 356.0

    initial_guess = np.array([
        64.5,
        355.3,
        305.9,
    ])

    print("\n========== DISTURBANCE CAPACITY ANALYSIS ==========")
    print(f"Target reactor temperature: {setpoint:.2f} K")

    print("\nFeed temperature | Required coolant flow | Feasible")

    for feed_temperature in np.arange(350.0, 356.0, 0.5):

        low = params.minimum_coolant_flow
        high = params.maximum_coolant_flow

        low_state = find_equilibrium(
            params, low, feed_temperature, initial_guess
        )

        high_state = find_equilibrium(
            params, high, feed_temperature, initial_guess
        )

        if low_state is None or high_state is None:
            print(f"{feed_temperature:.2f} K | Equilibrium not found")
            continue

        low_temp = low_state[1]
        high_temp = high_state[1]

        if not min(low_temp, high_temp) <= setpoint <= max(low_temp, high_temp):
            print(
                f"{feed_temperature:.2f} K | "
                f"Outside actuator capability | No"
            )
            continue

        for _ in range(60):
            middle = (low + high) / 2.0

            middle_state = find_equilibrium(
                params,
                middle,
                feed_temperature,
                initial_guess,
            )

            if middle_state is None:
                break

            if middle_state[1] > setpoint:
                low = middle
            else:
                high = middle

        required_flow = (low + high) / 2.0

        feasible = (
            params.minimum_coolant_flow
            <= required_flow
            <= params.maximum_coolant_flow
        )

        print(
            f"{feed_temperature:.2f} K | "
            f"{required_flow:.6f} m^3/min | "
            f"{feasible}"
        )


if __name__ == "__main__":
    main()
