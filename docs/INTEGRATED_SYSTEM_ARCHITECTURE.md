# Integrated CSTR Digital Twin and FTC Architecture

## Scope

This system is a simulation verification environment. It does not validate a
physical CSTR and does not claim fault-tolerance capability on hardware.

## Components

- `python/plant_model/cstr_model.py`: nonlinear CSTR and jacket dynamics.
- `python/controllers/pi_controller.py`: negative-gain PI temperature control.
- `python/estimation/state_estimator.py`: open-loop model-based state predictor.
- `python/fault_detection/fault_detector.py`: residual thresholds and persistence.
- `python/controllers/fault_tolerant_controller.py`: supervisory mode selection.
- `python/experiments/integrated_simulation.py`: central time-stepped engine.
- `python/experiments/run_integrated_simulation.py`: reproducible command interface.

## Step Sequence

Each interval uses the controller decision selected at the preceding sample:

1. Use the controller command and apply actuator limits.
2. Apply the stuck-actuator fault, if configured.
3. Advance the true CSTR with the actual coolant flow.
4. Generate the measured reactor temperature, bias, and optional noise.
5. Predict the digital-twin state using actual coolant-flow feedback.
6. Calculate `r_T = measured_temperature - estimated_temperature`.
7. Calculate `r_u = controller_command - actual_coolant_flow`.
8. Apply fault persistence and diagnosis.
9. Pass the diagnosis to the FTC and select the mode for the next step.
10. Record states, residuals, alarms, commands, and FTC mode.

## Residual and FTC Rules

- A sensor alarm selects `SENSOR_RECONFIGURATION` and estimated temperature
  feedback.
- An actuator alarm with no alternative authority selects `SAFE_MODE`.
- The existing `ACTUATOR_RECONFIGURATION` enum is retained, but no alternative
  actuator is implemented. It is only a supervisory state.
- Simultaneous sensor and actuator alarms are diagnosed explicitly as
  `sensor_and_actuator`; the integrated engine enters `SAFE_MODE`.
- In safe mode, the engine holds the last controller request. It does not invent
  actuator authority or physical recovery.

## State Estimator Limitation

`ModelBasedStateEstimator` is an open-loop model predictor, not a corrected
observer. It depends on the actual actuator flow and is sensitive to model
uncertainty. Sensor residuals can therefore include model mismatch as well as
sensor bias. No mismatch compensation or physical observer validation is claimed.
