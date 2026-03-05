from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal, Optional, Dict, Any

import numpy as np

from pricing.black_scholes import PricingInputError

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Result container for Monte Carlo pricing."""
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    antithetic: bool
    control_variate: bool
    diagnostics: Dict[str, Any]


def _validate_inputs(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    option_type: str,
) -> OptionType:
    if s0 <= 0:
        raise PricingInputError("s0 must be > 0")
    if k <= 0:
        raise PricingInputError("k must be > 0")
    if t <= 0:
        raise PricingInputError("t must be > 0")
    if sigma < 0:
        raise PricingInputError("sigma must be >= 0")
    if n_paths <= 0:
        raise PricingInputError("n_paths must be > 0")

    opt = option_type.lower().strip()
    if opt not in ("call", "put"):
        raise PricingInputError("option_type must be 'call' or 'put'")
    return opt


def _z_value(confidence_level: float) -> float:
    """
    Z values for common confidence levels (normal approximation).
    Keeping this dependency-free (no SciPy).
    """
    z_map = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    return z_map.get(round(confidence_level, 2), z_map[0.95])


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate GBM paths (for later visualization / path-dependent payoffs).

    Returns:
        paths: ndarray shape (n_paths, n_steps + 1), includes S0 at column 0.
    """
    if n_steps <= 0:
        raise PricingInputError("n_steps must be > 0")
    if n_paths <= 0:
        raise PricingInputError("n_paths must be > 0")
    if t <= 0:
        raise PricingInputError("t must be > 0")
    if s0 <= 0:
        raise PricingInputError("s0 must be > 0")
    if sigma < 0:
        raise PricingInputError("sigma must be >= 0")

    rng = np.random.default_rng(seed)
    dt = t / n_steps

    # Generate shocks
    z = rng.standard_normal(size=(n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * z

    # Log-paths for numerical stability
    log_s = np.empty((n_paths, n_steps + 1), dtype=float)
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = log_s[:, [0]] + np.cumsum(increments, axis=1)

    return np.exp(log_s)


def price_european_option_mc(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    option_type: str,
    *,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
    antithetic: bool = True,
    control_variate: bool = True,
    confidence_level: float = 0.95,
    batch_size: int = 250_000,
) -> MonteCarloResult:
    """
    Price a European option using Monte Carlo simulation under GBM.

    Variance reduction:
      - Antithetic variates (Z and -Z)
      - Control variate using discounted terminal stock price:
            Y = exp(-rT) * S_T,  E[Y] = S0  (risk-neutral)

    Returns:
        MonteCarloResult with price, std_error and CI.
    """
    opt = _validate_inputs(s0, k, r, sigma, t, n_paths, option_type)

    rng = np.random.default_rng(seed)
    disc = exp(-r * t)
    mu = (r - 0.5 * sigma**2) * t
    vol = sigma * sqrt(t)

    n_total = 0
    sum_x = 0.0
    sum_x2 = 0.0

    sum_y = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0

    remaining = n_paths

    while remaining > 0:
        n_batch = min(batch_size, remaining)

        # Antithetic variates: generate half, mirror to -Z, then slice to n_batch
        if antithetic:
            half = (n_batch + 1) // 2
            z_half = rng.standard_normal(size=half)
            z = np.concatenate([z_half, -z_half])[:n_batch]
        else:
            z = rng.standard_normal(size=n_batch)

        st = s0 * np.exp(mu + vol * z)

        if opt == "call":
            payoff = np.maximum(st - k, 0.0)
        else:
            payoff = np.maximum(k - st, 0.0)

        x = disc * payoff  # discounted payoff
        y = disc * st      # control variate (discounted terminal stock), E[y] = s0

        n_total += n_batch
        sum_x += float(np.sum(x))
        sum_x2 += float(np.sum(x * x))

        sum_y += float(np.sum(y))
        sum_y2 += float(np.sum(y * y))
        sum_xy += float(np.sum(x * y))

        remaining -= n_batch

    mean_x = sum_x / n_total

    if n_total > 1:
        var_x = (sum_x2 - n_total * mean_x * mean_x) / (n_total - 1)
        mean_y = sum_y / n_total
        var_y = (sum_y2 - n_total * mean_y * mean_y) / (n_total - 1)
        cov_xy = (sum_xy - n_total * mean_x * mean_y) / (n_total - 1)
    else:
        mean_y = sum_y / n_total
        var_x = 0.0
        var_y = 0.0
        cov_xy = 0.0

    used_cv = bool(control_variate and var_y > 0.0)

    if used_cv:
        b = cov_xy / var_y
        ey = s0
        price = mean_x - b * (mean_y - ey)

        var_adj = var_x + (b * b) * var_y - 2.0 * b * cov_xy
        var_adj = max(var_adj, 0.0)
        std_error = sqrt(var_adj / n_total)

        diagnostics = {
            "mean_x_plain": mean_x,
            "mean_y": mean_y,
            "E_y": ey,
            "b_control_variate": b,
            "var_plain": var_x,
            "var_control_variate": var_adj,
            "variance_reduction_factor": (var_x / var_adj) if var_adj > 0 else np.inf,
        }
    else:
        price = mean_x
        std_error = sqrt(var_x / n_total) if n_total > 1 else 0.0
        diagnostics = {
            "var_plain": var_x,
            "mean_x_plain": mean_x,
            "note": "control_variate disabled or var_y == 0",
        }

    z_val = _z_value(confidence_level)
    ci_low = price - z_val * std_error
    ci_high = price + z_val * std_error

    return MonteCarloResult(
        price=float(price),
        std_error=float(std_error),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_paths=int(n_total),
        antithetic=bool(antithetic),
        control_variate=bool(used_cv),
        diagnostics=diagnostics,
    )