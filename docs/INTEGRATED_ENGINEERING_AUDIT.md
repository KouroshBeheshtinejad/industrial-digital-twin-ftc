# Integrated CSTR Engineering Audit

## Audit Basis

The audit inspected the actual integrated engine, fault detector, PI controller,
FTC controller, state estimator, generated CSV files, plots, and tests. It also
reran the complete default scenario and the Python test suite.

## Verification Summary

| Finding | Classification |
|---|---|
| Controller and plant are advanced with the same interval command | Verified correct |
| Sensor fault does not modify the true CSTR state | Verified correct |
| Estimator advances with actual coolant flow | Verified correct |
| Fault injection timestamps are exact | Verified correct |
| Sensor and actuator faults are detected independently | Verified correct |
| Simultaneous diagnosis is represented explicitly | Verified correct |
| FTC enters SAFE_MODE in both actuator-only and simultaneous faults | Verified correct |
| IAE and ISE use actual reactor temperature and actual time samples | Verified correct |
| CSV residual and flow-command relationship initially was inconsistent | Verified defect, corrected |
| CSV arrays are consistently sized and finite | Verified correct |
| Random seeds and reproducibility are implemented | Verified correct |
| Full physical emergency shutdown is not implemented | Verified limitation |

## 1. Timing Analysis

The simulation uses a 0.05-minute sample interval and 1601 samples for the
80-minute default run. The implementation uses a zero-order-hold convention:

1. A controller command exists for the current interval.
2. Actuator limits are applied.
3. An actuator stuck fault is applied at `time[index] >= fault_time`.
4. The nonlinear CSTR is advanced over the current 0.05-minute interval.
5. The measurement is generated at the next sample time. Sensor bias begins
   when `time[index + 1] >= fault_time`.
6. The estimator predicts the next state using the actual coolant flow and the
   0.05-minute interval.
7. Residuals are calculated and the detector updates persistence counters.
8. FTC selects a mode using the new measurement and estimated state.
9. The new FTC/PI command is held for the following interval.

This is a ZOH, discrete-sample convention. There is no future-state information
leakage: the controller and FTC receive the measurement and estimate for the
current sample, and never receive the following plant state. The estimator
predicts only over the completed interval.

Sensor bias activation occurs exactly at 30.00 minutes. Actuator stuck activation
occurs exactly at 30.00 minutes. The 30.20-minute sensor detection and
30.25-minute actuator detection are consistent with five persistence samples:

- Sensor: 30.00, 30.05, 30.10, 30.15, 30.20
- Actuator: 30.00, 30.05, 30.10, 30.15, 30.20, 30.25

The actuator residual threshold requires 0.01 m^3/min exceedance. It persists
for six samples after 30.00 before reporting the alarm at 30.25.

## Verified Defect and Correction

Before the correction, CSV `controller_command` could contain the command selected
after FTC/detection while `actuator_residual` used the command for the interval
that had just been applied. As a result, the logged equation
`actuator_residual = controller_command - actual_coolant_flow` was not always
satisfied.

The implementation now logs `interval_command` with the actual flow and detector
residual for the same interval. A regression assertion was added to verify:

```text
actuator_residual == controller_command - actual_coolant_flow
```

This correction does not alter CSTR dynamics or performance metrics.

## 2. Performance Metrics

For every sample:

```text
e(t) = 356.0 K - T_actual(t)
```

IAE and ISE are computed with `numpy.trapezoid` on the 0.05-minute time grid.
This is correct numerical integration of the actual reactor-temperature tracking
error. Final temperature error is the final setpoint-minus-actual value.

The regenerated metrics were independently recomputed from CSV temperature:

| Scenario | IAE | ISE | Final Error (K) |
|---|---:|---:|---:|
| Healthy | 18.991797 | 8.355625 | 0.006755 |
| Sensor bias | 19.463460 | 8.472839 | 0.012341 |
| Actuator stuck | 67.514384 | 66.448744 | 1.230496 |
| Simultaneous | 67.514384 | 66.448744 | 1.230496 |

Fault detection delay is
`first_matching_alarm_time - configured_fault_time`. A no-alarm result is
represented as NaN. False alarms are counted as pre-fault samples for which
either alarm is true. No pre-fault false alarms occurred in the rerun.

## 3. Simultaneous Fault Assessment

The simultaneous scenario independently enables:

- Sensor bias of +2 K at and after 30.00 minutes
- Actuator stuck at 0.5 m^3/min at and after 30.00 minutes

At the first persistent simultaneous alarm, both booleans are true and
`diagnosed_fault == "sensor_and_actuator"`. FTC sets SAFE_MODE.

Both residual columns remain populated. The sensor residual is approximately
+2 K and the actuator residual is approximately -0.451 m^3/min in the first
post-detection CSV rows.

IAE and ISE are exactly identical to actuator-only because the physical plant
uses the same stuck actuator trajectory in both scenarios and sensor bias is
not applied to physical state. The additional sensor bias changes only the
measurement residual and diagnosis; it does not change true reactor dynamics.

## 4. Digital Twin and State Estimator

The physical CSTR uses `actual_flow`, including 0.5 m^3/min during the stuck
fault. Sensor bias is added only to the measurement. It is not added to the
plant state or estimator state.

The estimator starts from the same initial state as the true plant and performs
one open-loop model integration per sample using actual coolant flow. It does
not consume measured temperature or correct against temperature residuals.

Consequently, this is a model-based prediction, not a measurement-corrected
observer. Sensor residuals include sensor bias and possible model mismatch.
The implementation does not claim a validated physical-state observer.

## 5. SAFE_MODE and Actuator Authority

Actuator-only and simultaneous faults enter SAFE_MODE after both required
persistence alarms become active.

In SAFE_MODE, the integrated engine:

- Does not obtain a new PI command.
- Holds `last_pi_command`.
- Continues applying actual actuator behavior.
- For a stuck actuator, continues setting actual flow to 0.5 m^3/min.
- Cannot restore actuator authority or cause physical recovery.

This is only software simulation behavior. It is not a verified emergency
shutdown, hardware interlock, or physical safety-system result.

## 6. Numerical and Software Validation

Verified finite-value checks cover scenario parameters, PI parameters,
controller temperatures and setpoints, FTC temperature inputs, detector inputs,
plant integration output, estimator state, and final result arrays.

The random seed is fixed per scenario and NumPy's default RNG is used for
reproducible measurement noise. CSV output uses 1601 finite rows per scenario,
and plots are saved as PNG files.

Potential concern: the state estimator's `predict()` validates flow finiteness
and positive sample time but does not explicitly validate infinite sample time.
The integrated engine rejects non-finite scenario sample time before this
interface is called.

Potential concern: `PIController.update_limits()` does not verify that
`coolant_bias` remains within new limits. It is only relevant if that method
is called independently; integrated initialization supplies limits containing
the nominal bias.

Potential concern: the full integrated simultaneous scenario is covered by the
actual rerun and CSV inspection, but not yet by an automated test dedicated to
full-engine simultaneous trajectories.

## 7. Test and Run Results

Test command:

```powershell
python -m pytest python/tests -q
```

Result: `16 passed`.

Integrated rerun command:

```powershell
python -m python.experiments.run_integrated_simulation
```

Fresh rerun timestamp:
`20261125_165536`

Generated location:
`results/integrated_simulation/20261125_165536/`

The rerun produced metrics plus CSV and PNG files for all four scenarios.
An independent CSV recomputation matched the reported IAE and ISE values.
The actuator residual invariant held to numerical rounding.

## 8. Remaining Engineering Limitations

- This is simulation verification, not physical CSTR validation.
- Residual thresholds and persistence are configurable, not experimentally
  established optimal values.
- The estimator cannot distinguish model mismatch from a sensor fault.
- The detector may issue alarms due to nominal model mismatch.
- Safe mode has no redundant actuator or physical recovery mechanism.
- No actuator dynamics, valve stiction, saturation lag, or sensor dynamics are
  explicitly modeled beyond the existing CSTR assumptions.
- No hardware fault-injection validation is claimed.

## 9. Exact Execution Commands

All default scenarios:

```powershell
python -m python.experiments.run_integrated_simulation
```

Healthy:

```powershell
python -m python.experiments.run_integrated_simulation healthy
```

Sensor bias:

```powershell
python -m python.experiments.run_integrated_simulation sensor_bias
```

Actuator stuck:

```powershell
python -m python.experiments.run_integrated_simulation actuator_stuck
```

Simultaneous faults:

```powershell
python -m python.experiments.run_integrated_simulation simultaneous_faults
```
