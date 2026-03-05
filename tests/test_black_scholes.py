import math

import pytest

from src.pricing.black_scholes import (
    PricingInputError,
    black_scholes_d1_d2,
    black_scholes_price,
)


def test_black_scholes_call_reference_value():
    price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        option_type="call",
    )

    assert price == pytest.approx(10.450583572185565, rel=1e-9)


def test_black_scholes_put_reference_value():
    price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        option_type="put",
    )

    assert price == pytest.approx(5.573526022256971, rel=1e-9)


def test_put_call_parity_holds():
    call_price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        option_type="call",
    )
    put_price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        option_type="put",
    )

    lhs = call_price - put_price
    rhs = 100.0 - 100.0 * math.exp(-0.05)

    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_option_value_equals_intrinsic_value_at_expiry():
    call_price = black_scholes_price(
        spot=120.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=0.0,
        volatility=0.2,
        option_type="call",
    )
    put_price = black_scholes_price(
        spot=80.0,
        strike=100.0,
        rate=0.05,
        time_to_maturity=0.0,
        volatility=0.2,
        option_type="put",
    )

    assert call_price == pytest.approx(20.0)
    assert put_price == pytest.approx(20.0)


def test_zero_volatility_matches_deterministic_limit_for_call():
    spot = 100.0
    strike = 105.0
    rate = 0.03
    time_to_maturity = 2.0

    price = black_scholes_price(
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_maturity=time_to_maturity,
        volatility=0.0,
        option_type="call",
    )

    expected = max(spot - strike * math.exp(-rate * time_to_maturity), 0.0)
    assert price == pytest.approx(expected, rel=1e-12)


def test_zero_volatility_matches_deterministic_limit_for_put():
    spot = 90.0
    strike = 100.0
    rate = 0.05
    time_to_maturity = 1.0

    price = black_scholes_price(
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_maturity=time_to_maturity,
        volatility=0.0,
        option_type="put",
    )

    expected = max(strike * math.exp(-rate * time_to_maturity) - spot, 0.0)
    assert price == pytest.approx(expected, rel=1e-12)


def test_invalid_spot_raises():
    with pytest.raises(PricingInputError):
        black_scholes_price(
            spot=0.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=1.0,
            volatility=0.2,
            option_type="call",
        )


def test_invalid_strike_raises():
    with pytest.raises(PricingInputError):
        black_scholes_price(
            spot=100.0,
            strike=0.0,
            rate=0.05,
            time_to_maturity=1.0,
            volatility=0.2,
            option_type="call",
        )


def test_negative_time_to_maturity_raises():
    with pytest.raises(PricingInputError):
        black_scholes_price(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=-1.0,
            volatility=0.2,
            option_type="call",
        )


def test_negative_volatility_raises():
    with pytest.raises(PricingInputError):
        black_scholes_price(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=1.0,
            volatility=-0.2,
            option_type="call",
        )


def test_invalid_option_type_raises():
    with pytest.raises(PricingInputError):
        black_scholes_price(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=1.0,
            volatility=0.2,
            option_type="invalid",
        )


def test_d1_d2_require_positive_time_to_maturity():
    with pytest.raises(PricingInputError):
        black_scholes_d1_d2(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=0.0,
            volatility=0.2,
        )


def test_d1_d2_require_positive_volatility():
    with pytest.raises(PricingInputError):
        black_scholes_d1_d2(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=1.0,
            volatility=0.0,
        )