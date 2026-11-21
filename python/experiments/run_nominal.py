from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from python.plant_model.cstr_model import (load_parameters, simulate_cstr)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (PROJECT_ROOT/ "configs"/ "plant_parameters"/ "cstr_nominal.json")

params = load_parameters(CONFIG_PATH)

initial_state = np.array([
    100.0,  # Initial reactant concentration (mol/m^3)
    350.0,  # Initial reactor temperature (K)
    300.0   # Initial jacket temperature (K)
])

solution = simulate_cstr(params=params, initial_state=initial_state, coolant_flow=params.nominal_coolant_flow, simulation_time=20.0)

time = solution.t
concentration = solution.y[0]
reactor_temperature = solution.y[1]
jacket_temperature = solution.y[2]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(time, concentration)
axes[0].set_ylabel("Concentration (mol/m^3)")
axes[0].grid(True)

axes[1].plot(time, reactor_temperature)
axes[1].set_ylabel("Reactor Temperature (K)")
axes[1].grid(True)

axes[2].plot(time, jacket_temperature)
axes[2].set_ylabel("Jacket Temperature (K)")
axes[2].set_xlabel("Time (min)")
axes[2].grid(True)

fig.suptitle("CSTR Nominal Open-Loop Simulation")
fig.tight_layout()

output_directory = PROJECT_ROOT / "results" / "figures"
output_directory.mkdir(parents=True, exist_ok=True)

fig.savefig(output_directory / "cstr_nominal_response.png",dpi=150)

print("Simulation completed successfully.")
print(f"Final concentration: {concentration[-1]:.4f} mol/m^3")
print(f"Final reactor temperature: {reactor_temperature[-1]:.4f} K")
print(f"Final jacket temperature: {jacket_temperature[-1]:.4f} K")

plt.show()