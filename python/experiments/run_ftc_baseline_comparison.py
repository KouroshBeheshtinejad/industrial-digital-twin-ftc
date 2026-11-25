from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python.plant_model.cstr_model import (
    load_parameters,
    cstr_dynamics,
)

from python.controllers.pi_controller import PIController


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def simulate_baseline(
    params,
    initial_state,
    simulation_time=80.0,
    sample_time=0.05,
):
    number_of_steps = int(simulation_time / sample_time) + 1

    time = np.linspace(0.0, simulation_time, number_of_steps)

    states = np.zeros((number_of_steps, 3))
    coolant_flows = np.zeros(number_of_steps)
    errors = np.zeros(number_of_steps)

    controller = PIController(
        kp=0.03,
        ki=0.003,
        setpoint=356.0,
        coolant_bias=params.nominal_coolant_flow,
        minimum_output=params.minimum_coolant_flow,
        maximum_output=params.maximum_coolant_flow,
    )

    states[0] = initial_state

    for i in range(number_of_steps - 1):
        current_state = states[i]

        temperature = current_state[1]

        coolant_flow = controller.compute(
            temperature=temperature,
            sample_time=sample_time,
        )

        from scipy.integrate import solve_ivp

        solution = solve_ivp(
            fun=lambda t, x: cstr_dynamics(
                t,
                x,
                coolant_flow,
                params,
            ),
            t_span=(time[i], time[i + 1]),
            y0=current_state,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        states[i + 1] = solution.y[:, -1]

        coolant_flows[i] = coolant_flow
        errors[i] = controller.setpoint - temperature

    coolant_flows[-1] = coolant_flows[-2]
    errors[-1] = controller.setpoint - states[-1, 1]

    return time, states, coolant_flows, errors


def calculate_metrics(time, states, coolant_flows, errors):
    temperature = states[:, 1]

    iae = np.trapezoid(np.abs(errors), time)
    ise = np.trapezoid(errors**2, time)

    return {
        "final_temperature_K": temperature[-1],
        "final_error_K": errors[-1],
        "maximum_temperature_K": np.max(temperature),
        "minimum_temperature_K": np.min(temperature),
        "maximum_coolant_flow": np.max(coolant_flows),
        "minimum_coolant_flow": np.min(coolant_flows),
        "IAE": iae,
        "ISE": ise,
        "actuator_within_limits": bool(
            np.all(coolant_flows >= 0.02)
            and np.all(coolant_flows <= 0.5)
        ),
    }


def main():
    params = load_parameters(CONFIG_PATH)

    initial_state = np.array([
        64.50615287,
        355.28373576,
        305.90638203,
    ])

    time, states, coolant_flows, errors = simulate_baseline(
        params=params,
        initial_state=initial_state,
    )

    metrics = calculate_metrics(
        time,
        states,
        coolant_flows,
        errors,
    )

    print("\n========== FTC BASELINE COMPARISON ==========\n")

    print("Controller: Modular PI Controller")
    print(f"Simulation duration: {time[-1]:.2f} min")
    print(f"Sample time: {time[1] - time[0]:.4f} min")

    print("\nPerformance metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        TABLE_DIR / "ftc_baseline_metrics.csv",
        index=False,
    )

    results_df = pd.DataFrame({
        "time_min": time,
        "concentration_mol_m3": states[:, 0],
        "reactor_temperature_K": states[:, 1],
        "jacket_temperature_K": states[:, 2],
        "coolant_flow_m3_min": coolant_flows,
        "tracking_error_K": errors,
    })

    results_df.to_csv(
        TABLE_DIR / "ftc_baseline_trajectory.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
    )

    axes[0].plot(
        time,
        states[:, 1],
        label="Reactor temperature",
    )
    axes[0].axhline(
        356.0,
        linestyle="--",
        label="Setpoint",
    )
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        coolant_flows,
        label="Coolant flow",
    )
    axes[1].axhline(0.02, linestyle="--", label="Minimum")
    axes[1].axhline(0.5, linestyle="--", label="Maximum")
    axes[1].set_ylabel("Flow (m^3/min)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        time,
        errors,
        label="Tracking error",
    )
    axes[2].set_xlabel("Time (min)")
    axes[2].set_ylabel("Error (K)")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ftc_baseline_comparison.png",
        dpi=200,
    )

    plt.close(fig)

    print("\nOutput files saved successfully.")


if __name__ == "__main__":
    main()
