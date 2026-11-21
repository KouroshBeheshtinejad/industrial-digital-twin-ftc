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

    nominal_input = params.nominal_coolant_flow

    initial_guess = np.array([
        100.0,
        350.0,
        300.0,
    ])

    def plant_dynamics(state, input_value):
        return cstr_dynamics(
            0.0,
            state,
            input_value,
            params,
        )

    equilibrium = root(
        lambda state: plant_dynamics(state, nominal_input),
        initial_guess,
    )

    if not equilibrium.success:
        raise RuntimeError(equilibrium.message)

    x_star = equilibrium.x
    u_star = nominal_input

    # A = partial f / partial x
    A = numerical_jacobian_state(
        lambda state: plant_dynamics(state, u_star),
        x_star,
    )

    # B = partial f / partial u
    B = numerical_jacobian_input(
        lambda input_value: plant_dynamics(x_star, input_value),
        u_star,
    ).reshape(3, 1)

    # Output: reactor temperature
    C = np.array([
        [0.0, 1.0, 0.0]
    ])

    D = np.zeros((1, 1))

    # Controllability matrix
    controllability = np.hstack([
        B,
        A @ B,
        A @ A @ B,
    ])

    controllability_rank = np.linalg.matrix_rank(
        controllability
    )

    # Observability matrix
    observability = np.vstack([
        C,
        C @ A,
        C @ A @ A,
    ])

    observability_rank = np.linalg.matrix_rank(
        observability
    )

    eigenvalues = np.linalg.eigvals(A)

    print("\n========== LINEARIZED CSTR MODEL ==========")

    print("\nOperating point:")
    print("x* =", x_star)
    print("u* =", u_star)

    print("\nA matrix:")
    print(A)

    print("\nB matrix:")
    print(B)

    print("\nC matrix:")
    print(C)

    print("\nD matrix:")
    print(D)

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\n========== SYSTEM PROPERTIES ==========")

    print(
        f"Controllability rank: "
        f"{controllability_rank}/3"
    )

    print(
        f"Observability rank: "
        f"{observability_rank}/3"
    )

    if controllability_rank == 3:
        print("The linearized system is controllable.")
    else:
        print("The linearized system is not fully controllable.")

    if observability_rank == 3:
        print("The linearized system is observable.")
    else:
        print("The linearized system is not fully observable.")

    print("\n========== STATE-SPACE EQUATIONS ==========")
    print("dx/dt = A dx + B du")
    print("dy/dt = C dx + D du")


if __name__ == "__main__":
    main()
