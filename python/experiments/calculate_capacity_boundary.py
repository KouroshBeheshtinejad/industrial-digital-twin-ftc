from pathlib import Path

import numpy as np

from python.plant_model.cstr_model import (
    load_parameters,
    reaction_rate_constant,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "plant_parameters"
    / "cstr_nominal.json"
)


def required_feed_temperature(params, setpoint, coolant_flow):

    k = reaction_rate_constant(setpoint, params)

    dilution_rate = (
        params.feed_flow_rate / params.reactor_volume
    )

    concentration = (
        dilution_rate * params.feed_concentration
        / (dilution_rate + k)
    )

    jacket_capacity_rate = (
        params.jacket_density
        * params.jacket_heat_capacity
    )

    jacket_temperature = (
        coolant_flow * params.jacket_inlet_temperature
        + (
            params.heat_transfer_coefficient
            / jacket_capacity_rate
        ) * setpoint
    ) / (
        coolant_flow
        + params.heat_transfer_coefficient
        / jacket_capacity_rate
    )

    reaction_heat = (
        -params.heat_of_reaction
        * k
        * concentration
        / (
            params.reactor_density
            * params.reactor_heat_capacity
        )
    )

    jacket_heat = (
        params.heat_transfer_coefficient
        * (jacket_temperature - setpoint)
        / (
            params.reactor_density
            * params.reactor_heat_capacity
            * params.reactor_volume
        )
    )

    feed_temperature = (
        setpoint
        - (reaction_heat + jacket_heat) / dilution_rate
    )

    return {
        "feed_temperature": feed_temperature,
        "concentration": concentration,
        "jacket_temperature": jacket_temperature,
    }


def main():

    params = load_parameters(CONFIG_PATH)

    setpoint = 356.0

    minimum = required_feed_temperature(
        params,
        setpoint,
        params.minimum_coolant_flow,
    )

    maximum = required_feed_temperature(
        params,
        setpoint,
        params.maximum_coolant_flow,
    )

    print("\n========== CSTR CAPACITY BOUNDARY ==========")

    print(f"Target temperature: {setpoint:.6f} K")

    print("\nAt minimum coolant flow:")
    print(
        f"Feed temperature: "
        f"{minimum['feed_temperature']:.6f} K"
    )
    print(
        f"Jacket temperature: "
        f"{minimum['jacket_temperature']:.6f} K"
    )

    print("\nAt maximum coolant flow:")
    print(
        f"Feed temperature: "
        f"{maximum['feed_temperature']:.6f} K"
    )
    print(
        f"Jacket temperature: "
        f"{maximum['jacket_temperature']:.6f} K"
    )

    print("\nFeasible feed-temperature interval:")
    print(
        f"{min(minimum['feed_temperature'], maximum['feed_temperature']):.6f}"
        f" K to "
        f"{max(minimum['feed_temperature'], maximum['feed_temperature']):.6f} K"
    )

    print("\nMaximum coolant flow:")
    print(f"{params.maximum_coolant_flow:.6f} m^3/min")


if __name__ == "__main__":
    main()
