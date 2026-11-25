from dataclasses import dataclass


@dataclass
class PIController:
    """
    PI controller for the nonlinear CSTR process.

    The controller accounts for the negative process gain:
    increasing coolant flow decreases reactor temperature.
    """

    kp: float = 0.03
    ki: float = 0.003
    setpoint: float = 356.0
    coolant_bias: float = 0.1
    minimum_output: float = 0.02
    maximum_output: float = 0.5

    integral_error: float = 0.0

    def __post_init__(self):
        if self.kp < 0 or self.ki < 0:
            raise ValueError("Controller gains must be non-negative.")

        if self.minimum_output >= self.maximum_output:
            raise ValueError(
                "Minimum output must be smaller than maximum output."
            )

        if not self.minimum_output <= self.coolant_bias <= self.maximum_output:
            raise ValueError(
                "Coolant bias must be within actuator limits."
            )

    def compute(self, temperature, sample_time):
        """
        Calculate the saturated PI control output.

        Parameters
        ----------
        temperature : float
            Measured or estimated reactor temperature (K).

        sample_time : float
            Controller sampling interval (min).

        Returns
        -------
        float
            Coolant flow command (m^3/min).
        """

        if sample_time <= 0:
            raise ValueError("Sample time must be positive.")

        if temperature != temperature:
            raise ValueError("Temperature must be a valid number.")

        error = self.setpoint - temperature

        unconstrained_output = (
            self.coolant_bias
            - self.kp * error
            - self.ki * self.integral_error
        )

        # Conditional integration anti-windup.
        integration_allowed = (
            self.minimum_output < unconstrained_output < self.maximum_output
            or (
                unconstrained_output >= self.maximum_output
                and error > 0
            )
            or (
                unconstrained_output <= self.minimum_output
                and error < 0
            )
        )

        if integration_allowed:
            self.integral_error += error * sample_time

        # Recalculate output after integral update.
        unconstrained_output = (
            self.coolant_bias
            - self.kp * error
            - self.ki * self.integral_error
        )

        output = min(
            max(unconstrained_output, self.minimum_output),
            self.maximum_output,
        )

        return output

    def reset(self):
        """Reset the integral state."""
        self.integral_error = 0.0

    def update_setpoint(self, new_setpoint):
        """Update the reactor temperature setpoint."""
        self.setpoint = float(new_setpoint)

    def update_limits(self, minimum_output, maximum_output):
        """Update actuator output constraints."""

        if minimum_output >= maximum_output:
            raise ValueError(
                "Minimum output must be smaller than maximum output."
            )

        self.minimum_output = float(minimum_output)
        self.maximum_output = float(maximum_output)

        self.integral_error = 0.0
