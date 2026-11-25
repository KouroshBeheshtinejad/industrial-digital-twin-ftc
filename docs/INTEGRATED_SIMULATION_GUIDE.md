# Integrated CSTR Simulation Guide

## Run

From the repository root:

```powershell
python -m python.experiments.run_integrated_simulation
```

Run one scenario:

```powershell
python -m python.experiments.run_integrated_simulation sensor_bias
```

Available scenarios are `healthy`, `sensor_bias`, `actuator_stuck`,
`simultaneous_faults`, and `all`.

## Scenario Defaults

- Simulation duration: 80 minutes
- Sample interval: 0.05 minutes
- Setpoint: 356 K
- Fault time: 30 minutes
- Sensor bias: +2 K
- Stuck coolant flow: 0.5 m^3/min
- Temperature threshold: 0.75 K
- Actuator threshold: 0.01 m^3/min
- Persistence: 5 samples
- Measurement noise standard deviation: 0.05 K

## Outputs

Each run creates a timestamped directory under
`results/integrated_simulation/`. Every scenario contains:

- `trajectory.csv`: complete sample history
- `integrated_response.png`: temperature, flow, residual, and mode plots

The run directory contains `metrics.csv`.

The CSV includes actual and estimated states, measured temperature, injected
bias, controller and actual flows, both residuals, alarms, diagnosis, FTC mode,
and control temperature.

## Metrics

- IAE and ISE use the configured sample interval and time grid.
- Fault detection time is the first persistent alarm.
- Detection delay is detection time minus configured fault time.
- False alarms count alarms that occur before fault injection.
- Mode time is estimated by multiplying sample count by the sample interval.
- Final temperature error is setpoint minus final actual temperature.

NaN detection values indicate that no matching fault alarm occurred.

## Assumptions and Interpretation

The digital twin uses nominal CSTR parameters and actual flow feedback. It is
not a corrected observer. Nominal noise does not guarantee that residual
thresholds reject model mismatch. Sensor and actuator fault injection are
simulation assumptions, not evidence of physical fault detectability.
