
## CSTR Disturbance Capacity Analysis

### Experiment Configuration

- Reactor temperature setpoint: 356 K
- Coolant flow limits: 0.02–0.50 m³/min
- Feed temperature: Variable
- Plant model: Nonlinear CSTR

### Results

| Parameter | Minimum Coolant Flow | Maximum Coolant Flow |
|---|---:|---:|
| Coolant flow (m³/min) | 0.02 | 0.50 |
| Feed temperature (K) | 348.966494 | 351.316933 |
| Jacket temperature (K) | 320.958084 | 301.308411 |

### Engineering Interpretation

For the selected reactor temperature setpoint of 356 K, the
feasible feed-temperature interval is approximately:

348.966494 K to 351.316933 K.

Below the lower boundary, the minimum coolant flow prevents
the controller from reducing cooling sufficiently.

Above the upper boundary, the maximum coolant flow is
insufficient to compensate for the increased thermal load.

These boundaries are specific to the selected benchmark
parameters and actuator constraints.

### Conclusion

The CSTR has a finite disturbance rejection capability.
Disturbances outside this operating envelope cannot be
compensated for by the existing actuator configuration.
