from __future__ import annotations

from math import erf, exp, log, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


class PricingInputError(ValueError):
    """Raised when pricing inputs are invalid"""


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution"""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _validate_inputs(
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    option_type: str,
) -> str:
    """Validate pricing inputs and normalize the option type"""
    if spot <= 0.0:
        raise PricingInputError("spot must be greater than 0")
    if strike <= 0.0:
        raise PricingInputError("strike must be greater than 0")
    if time_to_maturity < 0.0:
        raise PricingInputError("time_to_maturity must be non-negative")
    if volatility < 0.0:
        raise PricingInputError("volatility must be non-negative")

    normalized_option_type = option_type.strip().lower()
    if normalized_option_type not in {"call", "put"}:
        raise PricingInputError("option_type must be 'call' or 'put'")

    return normalized_option_type


def black_scholes_d1_d2(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    volatility: float,
) -> tuple[float, float]:
    """Compute the Black-Scholes d1 and d2 terms"""
    if time_to_maturity <= 0.0:
        raise PricingInputError("time_to_maturity must be greater than 0")
    if volatility <= 0.0:
        raise PricingInputError("volatility must be greater than 0")

    sqrt_t = sqrt(time_to_maturity)
    numerator = log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity
    denominator = volatility * sqrt_t

    d1 = numerator / denominator
    d2 = d1 - volatility * sqrt_t
    return d1, d2


def _deterministic_option_value(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> float:
    """Price the option in the zero-volatility limit"""
    discounted_strike = strike * exp(-rate * time_to_maturity)

    if option_type == "call":
        return max(spot - discounted_strike, 0.0)

    return max(discounted_strike - spot, 0.0)


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    volatility: float,
    option_type: OptionType,
) -> float:
    """Price a European call or put under the Black-Scholes model"""
    normalized_option_type = _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        option_type=option_type,
    )

    if time_to_maturity == 0.0:
        if normalized_option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    if volatility == 0.0:
        return _deterministic_option_value(
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_maturity=time_to_maturity,
            option_type=normalized_option_type,
        )

    d1, d2 = black_scholes_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
    )

    discount_factor = exp(-rate * time_to_maturity)

    if normalized_option_type == "call":
        return spot * normal_cdf(d1) - strike * discount_factor * normal_cdf(d2)

    return strike * discount_factor * normal_cdf(-d2) - spot * normal_cdf(-d1)