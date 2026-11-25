# CSTR Digital Twin Robustness and Reliability Report

## Scope and Reproducibility

This report evaluates the existing integrated CSTR engine with no threshold
optimization. The default robustness run used:

- 10 seeds per condition
- 60-minute simulation horizon
- 0.10-minute sample interval
- 270 total runs
- fixed seed ranges:
  - Healthy noise: 6000-6039
  - UA mismatch: 7000-7059
  - Sensor bias: 8000-8119
  - Actuator stuck: 9000-9049

The seed ranges above follow the implementation. Seed 8000 and above were
used for all 12 sensor-bias conditions. Seed 9000 and above were used for the
five actuator stuck conditions.

Structured outputs:

- `results/robustness_validation/raw_results.csv`
- `results/robustness_validation/aggregated_results.csv`
- `results/robustness_validation/sensor_bias_detection_rate.png`
- `results/robustness_validation/actuator_stuck_detection_rate.png`
- `results/robustness_validation/ua_mismatch_alarm_rate.png`

The 270-run sweep completed successfully. It is intentionally lighter than the
full 80-minute, 0.05-minute integrated run.

## Healthy Measurement Noise

| Noise std (K) | Runs | Sensor false alarms | Actuator false alarms | Mean IAE | IAE std | Mean ISE | ISE std | Mean final error (K) | Max temperature deviation (K) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 10 | 0/10 | 0/10 | 18.497 | 0.039 | 8.289 | 0.019 | 0.0455 | 0.7163 |
| 0.05 | 10 | 0/10 | 0/10 | 18.502 | 0.083 | 8.296 | 0.048 | 0.0457 | 0.7163 |
| 0.10 | 10 | 0/10 | 0/10 | 18.579 | 0.136 | 8.327 | 0.091 | 0.0467 | 0.7163 |
| 0.20 | 10 | 0/10 | 0/10 | 18.427 | 0.316 | 8.259 | 0.170 | 0.0481 | 0.7163 |

Temperature residual standard deviation increased from approximately 0.020 K
at 0.02-K noise to 0.198 K at 0.20-K noise.

**Assessment:** The existing 0.75-K temperature threshold and 5-sample
persistence produced no false sensor or actuator alarms in these 40 healthy
runs. This supports robustness to measurement noise up to 0.20 K in the tested
conditions, but does not establish industrial robustness.

## UA Parameter Uncertainty

The nominal digital twin retained UA = 50000 W/K. The actual CSTR plant used
the following controlled UA multipliers:

| UA mismatch | Runs | Alarm/detection rate | Mean IAE | IAE std | Mean ISE | ISE std | Mean temperature residual (K) | Residual std (K) | Mean final error (K) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10% | 10 | 0/10 | 9.036 | 0.057 | 2.936 | 0.010 | 0.408 | 0.119 | 0.0269 |
| -5% | 10 | 0/10 | 14.727 | 0.073 | 5.393 | 0.029 | 0.191 | 0.070 | 0.0535 |
| 0% | 10 | 0/10 | 18.531 | 0.083 | 8.297 | 0.048 | 0.001 | 0.050 | 0.0473 |
| +5% | 10 | 0/10 | 20.872 | 0.070 | 11.160 | 0.045 | -0.168 | 0.063 | 0.0286 |
| +10% | 10 | 0/10 | 22.478 | 0.081 | 13.865 | 0.064 | -0.313 | 0.087 | 0.0234 |
| +15% | 10 | 0/10 | 23.441 | 0.087 | 16.237 | 0.062 | -0.443 | 0.108 | 0.0171 |

No persistent temperature or actuator alarm occurred at any tested UA mismatch.
The residual mean did grow materially with mismatch, especially negative UA.
Thus, the tested mismatch remained below the configured 0.75-K temperature
threshold for sustained alarm behavior.

**Important limitation:** Absence of an alarm does not prove model correctness.
The detector has no explicit model-mismatch state. A positive temperature
sensor residual and an open-loop twin residual are represented through the same
temperature residual. Therefore, the architecture cannot reliably distinguish
model mismatch from sensor bias.

## Sensor-Bias Sensitivity

| Bias (K) | Detection rate | Mean delay (min) | Delay std (min) | False sensor alarm rate |
|---:|---:|---:|---:|---:|
| +0.25 | 0/10 | NaN | NaN | 0/10 |
| +0.50 | 0/10 | NaN | NaN | 0/10 |
| +0.75 | 10/10 | 9.49 | 7.25 | 0/10 |
| +1.00 | 10/10 | 0.40 | 0.00 | 0/10 |
| +1.50 | 10/10 | 0.40 | 0.00 | 0/10 |
| +2.00 | 10/10 | 0.40 | 0.00 | 0/10 |
| -0.25 | 0/10 | NaN | NaN | 0/10 |
| -0.50 | 0/10 | NaN | NaN | 0/10 |
| -0.75 | 10/10 | 3.81 | 4.72 | 0/10 |
| -1.00 | 10/10 | 0.40 | 0.00 | 0/10 |
| -2.00 | 10/10 | 0.40 | 0.00 | 0/10 |

Residual means followed bias magnitude closely:

- Positive residual means: +0.1255, +0.2503, +0.3756, +0.5014, +0.7510,
  +1.0012 K
- Negative residual means: -0.1249, -0.2508, -0.3756, -0.5013, -0.7496,
  -1.0016 K

The 0.75-K condition is not uniformly prompt. Positive bias showed a mean delay
of 9.49 minutes with 7.25-minute standard deviation; negative bias showed a
mean delay of 3.81 minutes with 4.72-minute standard deviation.

**Assessment:** Within this sweep, +/-1.00 K and above were detected in every
seed with a 0.40-minute delay. +/-0.25 and +/-0.50 were not detected.
The +/-0.75 boundary was seed-dependent and not a reliable threshold.
A minimum detectable bias is not claimed beyond the tested conditions.

## Actuator-Stuck Sensitivity

| Stuck flow (m^3/min) | Detection rate | Delay mean (min) | Delay std (min) | Mean IAE | IAE std | Mean ISE | ISE std | Mean final error (K) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 10/10 | 0.50 | 0.00 | 31.303 | 0.027 | 19.628 | 0.022 | -0.928 |
| 0.30 | 10/10 | 0.50 | 0.00 | 40.965 | 0.070 | 32.481 | 0.072 | 1.110 |
| 0.40 | 10/10 | 0.50 | 0.00 | 42.243 | 0.057 | 34.982 | 0.059 | 1.166 |
| 0.45 | 10/10 | 0.50 | 0.00 | 42.658 | 0.070 | 35.844 | 0.070 | 1.185 |
| 0.50 | 10/10 | 0.50 | 0.00 | 43.041 | 0.050 | 36.588 | 0.053 | 1.200 |

All stuck flows in the tested set were detected with 0.50-minute delay.
The residual mean magnitude increased from 0.0142 m^3/min at 0.02 to
0.2264 m^3/min at 0.50.

FTC entered SAFE_MODE after actuator detection in every tested actuator-fault
run. SAFE_MODE holds the last PI request in software and the stuck physical
flow remains unchanged. It does not restore authority or recover the actuator.

## Statistical Method

Raw results contain one row per run with:

- Experiment and condition
- Random seed
- Detection and alarm flags
- IAE and ISE
- Temperature/final error metrics
- Residual statistics
- Detection times and delays where applicable

Aggregated results contain run count, seed list, mean, standard deviation,
minimum, maximum, detection rate, and false-alarm rates. NaN values mean no
matching alarm/detection occurred.

## Research Questions

1. **Healthy robustness to noise:** No false alarms at 0.02, 0.05, 0.10,
   or 0.20 K in 10 seeds per condition.
2. **Sensitivity to UA mismatch:** No alarm occurred for -10% through +15%
   in this 60-minute 0.10-minute sweep.
3. **Reliable sensor biases:** +/-1.00 K and above were reliable in the
   tested set. +/-0.75 was detected but delayed. +/-0.25 and +/-0.50 were
   not detected.
4. **Reliable actuator faults:** 0.02, 0.30, 0.40, 0.45, and 0.50 m^3/min
   were all detected in 10/10 runs with 0.50-minute delay.
5. **Magnitude and delay:** Actuator delay was largely constant because
   persistence dominated. Sensor delay decreased sharply from seed-dependent
   behavior near +/-0.75 K to 0.40 minute at +/-1.00 K and above.
6. **Sensor versus mismatch distinction:** The current architecture cannot
   distinguish temperature residual caused by model mismatch from a sensor
   fault. It applies the same sensor residual path to both.
7. **Industrial validation limitations:** No actuator dynamics, hardware
   validation, model-mismatch observer, optimal threshold study, uncertainty
   set validation, sensor dynamics, or physical emergency shutdown validation
   is implemented.

## Verified Findings and Limitations

Verified experimentally:

- Noise robustness under the four tested healthy noise levels
- UA mismatch sensitivity through +15% and -10%
- Sensor sensitivity for the listed 12 magnitudes
- Actuator stuck sensitivity for 0.02 and 0.30-0.50 m^3/min
- FTC SAFE_MODE behavior in simulation

Unresolved:

- Industrial threshold adequacy
- General model mismatch
- Minimum detectable sensor bias
- Physical actuator failure recovery
- Performance with different operating points and concurrent disturbances

## Commands

Default 10-run sweep:

```powershell
python -m python.experiments.run_robustness_validation --runs 10
```

Custom run count and timing:

```powershell
python -m python.experiments.run_robustness_validation --runs 20 --simulation-time 60 --sample-time 0.10
```

Integrated simulation:

```powershell
python -m python.experiments.run_integrated_simulation
```

Tests:

```powershell
python -m pytest python/tests -q
```
