from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from python.plant_model.cstr_model import cstr_dynamics


@dataclass
class ModelBasedStateEstimator:
    """
    Model-based state estimator for a nonlinear CSTR.

    The estimator predicts the internal plant state using the
    mathematical process model and the actual coolant flow.
    """

    params: object
    initial_state: np.ndarray
    integration_method: str = "RK45"
    relative_tolerance: float = 1e-7
    absolute_tolerance: float = 1e-9

    estimated_state: np.ndarray = field(init=False)
    elapsed_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.estimated_state = np.asarray(
            self.initial_state, dtype=float
        ).copy()

        if self.estimated_state.shape != (3,):
            raise ValueError(
                "The initial state must contain exactly three values."
            )

        if not np.all(np.isfinite(self.estimated_state)):
            raise ValueError("Initial state must contain finite values.")

        if self.relative_tolerance <= 0 or self.absolute_tolerance <= 0:
            raise ValueError("Integration tolerances must be positive.")

    def predict(self, actual_coolant_flow, sample_time):
        """
        Predict the next process state over one sampling interval.

        Parameters
        ----------
        actual_coolant_flow : float
            Actual coolant flow entering the reactor jacket.
        sample_time : float
            Prediction interval in minutes.

        Returns
        -------
        numpy.ndarray
            Estimated state [C_A, T, T_j].
        """

        if sample_time <= 0:
            raise ValueError("Sample time must be positive.")

        if not np.isfinite(actual_coolant_flow):
            raise ValueError("Coolant flow must be finite.")

        initial_state = self.estimated_state.copy()

        solution = solve_ivp(
            fun=lambda time, state: cstr_dynamics(
                time,
                state,
                actual_coolant_flow,
                self.params
            ),
            t_span=(0.0, sample_time),
            y0=initial_state,
            method=self.integration_method,
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance
        )

        if not solution.success:
            raise RuntimeError(
                f"State estimation failed: {solution.message}"
            )

        self.estimated_state = solution.y[:, -1].copy()
        self.elapsed_time += sample_time

        return self.estimated_state.copy()

    @property
    def estimated_concentration(self):
        return float(self.estimated_state[0])

    @property
    def estimated_temperature(self):
        return float(self.estimated_state[1])

    @property
    def estimated_jacket_temperature(self):
        return float(self.estimated_state[2])

    def reset(self, initial_state):
        """Reset the estimator to a specified initial state."""

        initial_state = np.asarray(initial_state, dtype=float)

        if initial_state.shape != (3,):
            raise ValueError(
                "The initial state must contain exactly three values."
            )

        if not np.all(np.isfinite(initial_state)):
            raise ValueError("Initial state must contain finite values.")

        self.estimated_state = initial_state.copy()
        self.elapsed_time = 0.0
