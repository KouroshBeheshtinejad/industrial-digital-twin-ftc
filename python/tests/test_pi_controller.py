import pytest

from python.controllers.pi_controller import PIController


def test_controller_output_within_limits():
    controller = PIController()

    for temperature in [350.0, 355.0, 356.0, 360.0]:
        output = controller.compute(
            temperature=temperature,
            sample_time=0.05,
        )

        assert 0.02 <= output <= 0.5


def test_controller_at_setpoint():
    controller = PIController()

    output = controller.compute(
        temperature=356.0,
        sample_time=0.05,
    )

    assert output == pytest.approx(0.1)


def test_higher_temperature_increases_cooling():
    controller = PIController()

    low_temperature_output = controller.compute(
        temperature=355.0,
        sample_time=0.05,
    )

    controller.reset()

    high_temperature_output = controller.compute(
        temperature=357.0,
        sample_time=0.05,
    )

    assert high_temperature_output > low_temperature_output


def test_controller_reset():
    controller = PIController()

    controller.compute(
        temperature=355.0,
        sample_time=0.05,
    )

    controller.reset()

    assert controller.integral_error == 0.0


def test_invalid_sample_time():
    controller = PIController()

    with pytest.raises(ValueError):
        controller.compute(
            temperature=356.0,
            sample_time=0.0,
        )


def test_invalid_actuator_limits():
    with pytest.raises(ValueError):
        PIController(
            minimum_output=0.5,
            maximum_output=0.2,
        )
