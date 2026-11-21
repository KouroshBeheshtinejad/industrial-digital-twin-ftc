
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class CSTRParameters:
    reactor_volume: float
    reactor_density: float
    reactor_heat_capacity: float

    jacket_volume: float
    jacket_density: float
    jacket_heat_capacity: float
    jacket_inlet_temperature: float

    feed_flow_rate: float
    feed_concentration: float
    feed_temperature: float

    pre_exponential_factor: float
    activation_energy_over_gas_constant: float
    heat_of_reaction: float

    heat_transfer_coefficient: float

    nominal_coolant_flow: float
    minimum_coolant_flow: float
    maximum_coolant_flow: float


def load_parameters(config_path: str | Path) -> CSTRParameters:
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return CSTRParameters(
        reactor_volume=config["reactor"]["volume"],
        reactor_density=config["reactor"]["density"],
        reactor_heat_capacity=config["reactor"]["heat_capacity"],

        jacket_volume=config["jacket"]["volume"],
        jacket_density=config["jacket"]["density"],
        jacket_heat_capacity=config["jacket"]["heat_capacity"],
        jacket_inlet_temperature=config["jacket"]["inlet_temperature"],

        feed_flow_rate=config["feed"]["flow_rate"],
        feed_concentration=config["feed"]["concentration"],
        feed_temperature=config["feed"]["temperature"],

        pre_exponential_factor=config["reaction"]["pre_exponential_factor"],
        activation_energy_over_gas_constant=(
            config["reaction"]["activation_energy_over_gas_constant"]
        ),
        heat_of_reaction=config["reaction"]["heat_of_reaction"],

        heat_transfer_coefficient=config["heat_transfer"]["UA"],

        nominal_coolant_flow=config["actuator"]["nominal_coolant_flow"],
        minimum_coolant_flow=config["actuator"]["minimum_coolant_flow"],
        maximum_coolant_flow=config["actuator"]["maximum_coolant_flow"],
    )


def reaction_rate_constant(
    temperature: float,
    params: CSTRParameters
) -> float:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero Kelvin.")

    exponent = (
        -params.activation_energy_over_gas_constant / temperature
    )

    return params.pre_exponential_factor * np.exp(exponent)


def cstr_dynamics(
    time: float,
    state: np.ndarray,
    coolant_flow: float,
    params: CSTRParameters
) -> np.ndarray:

    concentration, reactor_temperature, jacket_temperature = state

    if concentration < 0:
        concentration = 0.0

    k = reaction_rate_constant(reactor_temperature, params)

    reaction_rate = k * concentration

    dilution_rate = (
        params.feed_flow_rate / params.reactor_volume
    )

    d_concentration = (
        dilution_rate * (params.feed_concentration - concentration)
        - reaction_rate
    )

    reactor_heat_capacity_total = (
        params.reactor_density
        * params.reactor_heat_capacity
        * params.reactor_volume
    )

    reaction_heat = (
        -params.heat_of_reaction
        * reaction_rate
        / (
            params.reactor_density
            * params.reactor_heat_capacity
        )
    )

    feed_heat = (
        dilution_rate
        * (params.feed_temperature - reactor_temperature)
    )

    jacket_heat = (
        params.heat_transfer_coefficient
        * (jacket_temperature - reactor_temperature)
        / reactor_heat_capacity_total
    )

    d_temperature = feed_heat + reaction_heat + jacket_heat

    jacket_flow_term = (
        coolant_flow / params.jacket_volume
    ) * (
        params.jacket_inlet_temperature - jacket_temperature
    )

    jacket_heat_transfer = (
        params.heat_transfer_coefficient
        * (reactor_temperature - jacket_temperature)
        / (
            params.jacket_density
            * params.jacket_heat_capacity
            * params.jacket_volume
        )
    )

    d_jacket_temperature = (
        jacket_flow_term + jacket_heat_transfer
    )

    return np.array([
        d_concentration,
        d_temperature,
        d_jacket_temperature
    ])


def simulate_cstr(
    params: CSTRParameters,
    initial_state: np.ndarray,
    coolant_flow: float,
    simulation_time: float = 20.0,
    number_of_points: int = 1001
):
    if not (
        params.minimum_coolant_flow
        <= coolant_flow
        <= params.maximum_coolant_flow
    ):
        raise ValueError("Coolant flow is outside actuator limits.")

    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (3,):
        raise ValueError("Initial state must contain exactly 3 values.")

    if initial_state[0] < 0 or initial_state[1] <= 0 or initial_state[2] <= 0:
        raise ValueError("Initial state contains physically invalid values.")

    time_points = np.linspace(
        0.0,
        simulation_time,
        number_of_points
    )

    solution = solve_ivp(
        fun=lambda t, x: cstr_dynamics(
            t, x, coolant_flow, params
        ),
        t_span=(0.0, simulation_time),
        y0=initial_state,
        t_eval=time_points,
        method="RK45",
        rtol=1e-7,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(
            f"CSTR simulation failed: {solution.message}"
        )

    return solution
