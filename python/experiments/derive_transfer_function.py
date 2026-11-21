from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.signal import ss2tf

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


def numerical_jacobian_state(function, state):
    state = np.asarray(state, dtype=float)
    n = len(state)
    jacobian = np.zeros((n, n))

    for i in range(n):
        step = 1e-5 * max(1.0, abs(state[i]))

        x_plus = state.copy()
        x_minus = state.copy()

        x_plus[i] += step
        x_minus[i] -= step

        jacobian[:, i] = (
            function(x_plus) - function(x_minus)
        ) / (2.0 * step)

    return jacobian


def numerical_jacobian_input(function, input_value):
    step = 1e-6 * max(1.0, abs(input_value))

    return (
        function(input_value + step)
        - function(input_value - step)
    ) / (2.0 * step)


def main():
    params = load_parameters(CONFIG_PATH)
    u_star = params.nominal_coolant_flow

    def dynamics(state, input_value):
        return cstr_dynamics(
            0.0,
            state,
            input_value,
            params,
        )

    equilibrium = root(
        lambda state: dynamics(state, u_star),
        np.array([100.0, 350.0, 300.0]),
    )

    if not equilibrium.success:
        raise RuntimeError(equilibrium.message)

    x_star = equilibrium.x

    A = numerical_jacobian_state(
        lambda state: dynamics(state, u_star),
        x_star,
    )

    B = numerical_jacobian_input(
        lambda input_value: dynamics(x_star, input_value),
        u_star,
    ).reshape(3, 1)

    C = np.array([[0.0, 1.0, 0.0]])
    D = np.zeros((1, 1))

    numerator, denominator = ss2tf(A, B, C, D)

    numerator = numerator[0]

    # Remove numerically insignificant leading coefficients.
    threshold = 1e-10 * max(1.0, np.max(np.abs(numerator)))

    while len(numerator) > 1 and abs(numerator[0]) < threshold:
        numerator = numerator[1:]

    poles = np.linalg.eigvals(A)
    zeros = np.roots(numerator)

    dc_gain = (
        C @ (-np.linalg.solve(A, B)) + D
    )[0, 0]

    print("\n========== CSTR TRANSFER FUNCTION ==========")

    print("\nNumerator coefficients:")
    print(numerator)

    print("\nDenominator coefficients:")
    print(denominator)

    print("\nTransfer function:")
    print("G(s) = numerator(s) / denominator(s)")

    print("\nPoles:")
    for pole in poles:
        print(f"{pole:.8f}")

    print("\nZeros:")
    for zero in zeros:
        print(f"{zero:.8f}")

    print("\nDC gain:")
    print(f"{dc_gain:.8f} K/(m^3/min)")

    print("\n========== PROCESS CHARACTERISTICS ==========")

    print(
        "Stable:",
        bool(np.all(np.real(poles) < 0)),
    )

    print(
        "Minimum-phase:",
        bool(np.all(np.real(zeros) < 0)),
    )

    if dc_gain < 0:
        print(
            "Process direction: Increasing coolant flow "
            "decreases reactor temperature."
        )
    else:
        print(
            "Process direction: Increasing coolant flow "
            "increases reactor temperature."
        )


if __name__ == "__main__":
    main()
