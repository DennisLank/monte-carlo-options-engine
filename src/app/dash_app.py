from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, Input, Output, State, dcc, html

from pricing.black_scholes import black_scholes_price


OptionType = Literal["call", "put"]

@dataclass(frozen=True)
class ConvergenceResult:
    n: np.ndarray
    price: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    std_error: np.ndarray


def _simulate_terminal_prices_gbm(
    spot: float,
    rate: float,
    time_to_maturity: float,
    volatility: float,
    n_paths: int,
    seed: Optional[int],
    antithetic: bool,
) -> np.ndarray:
    """Simulate terminal prices S_T under GBM (risk-neutral)."""
    rng = np.random.default_rng(seed)

    mu = (rate - 0.5 * volatility**2) * time_to_maturity
    vol = volatility * sqrt(time_to_maturity)

    if antithetic:
        half = (n_paths + 1) // 2
        z_half = rng.standard_normal(size=half)
        z = np.concatenate([z_half, -z_half])[:n_paths]
    else:
        z = rng.standard_normal(size=n_paths)

    st = spot * np.exp(mu + vol * z)
    return st


def _discounted_payoff(
    st: np.ndarray,
    strike: float,
    rate: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> np.ndarray:
    disc = exp(-rate * time_to_maturity)
    if option_type == "call":
        payoff = np.maximum(st - strike, 0.0)
    else:
        payoff = np.maximum(strike - st, 0.0)
    return disc * payoff


def _z_value(confidence_level: float) -> float:
    # dependency-free z values for typical levels
    z_map = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    return z_map.get(round(confidence_level, 2), z_map[0.95])


def _mc_convergence(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    volatility: float,
    option_type: OptionType,
    n_paths: int,
    seed: Optional[int],
    antithetic: bool,
    control_variate: bool,
    confidence_level: float,
    n_checkpoints: int = 14,
) -> tuple[ConvergenceResult, np.ndarray, np.ndarray]:
    """
    Run ONE simulation with n_paths and compute convergence at checkpoints.

    Returns:
        convergence series, terminal prices S_T, discounted payoffs X (final sample)
    """
    st = _simulate_terminal_prices_gbm(
        spot=spot,
        rate=rate,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    x = _discounted_payoff(st, strike, rate, time_to_maturity, option_type)  # discounted payoff
    y = exp(-rate * time_to_maturity) * st  # control variate candidate, E[y] = spot

    # checkpoints (log-spaced, include n_paths)
    if n_paths < 2000:
        checkpoints = np.unique(np.linspace(50, n_paths, min(n_checkpoints, n_paths), dtype=int))
    else:
        checkpoints = np.unique(np.logspace(2, np.log10(n_paths), n_checkpoints, dtype=int))
    if checkpoints[-1] != n_paths:
        checkpoints = np.append(checkpoints, n_paths)

    # prefix sums for fast stats
    cx = np.cumsum(x)
    cx2 = np.cumsum(x * x)
    cy = np.cumsum(y)
    cy2 = np.cumsum(y * y)
    cxy = np.cumsum(x * y)

    prices = []
    se_list = []
    ci_low = []
    ci_high = []

    z = _z_value(confidence_level)

    for n in checkpoints:
        if n < 2:
            mean_x = cx[n - 1] / n
            price_n = float(mean_x)
            se_n = 0.0
        else:
            mean_x = cx[n - 1] / n
            var_x = (cx2[n - 1] - n * mean_x * mean_x) / (n - 1)

            if control_variate:
                mean_y = cy[n - 1] / n
                var_y = (cy2[n - 1] - n * mean_y * mean_y) / (n - 1)
                cov_xy = (cxy[n - 1] - n * mean_x * mean_y) / (n - 1)

                if var_y > 0:
                    b = cov_xy / var_y
                    price_n = float(mean_x - b * (mean_y - spot))
                    var_adj = var_x + (b * b) * var_y - 2.0 * b * cov_xy
                    var_adj = max(float(var_adj), 0.0)
                    se_n = sqrt(var_adj / n)
                else:
                    price_n = float(mean_x)
                    se_n = sqrt(max(float(var_x), 0.0) / n)
            else:
                price_n = float(mean_x)
                se_n = sqrt(max(float(var_x), 0.0) / n)

        prices.append(price_n)
        se_list.append(se_n)
        ci_low.append(price_n - z * se_n)
        ci_high.append(price_n + z * se_n)

    conv = ConvergenceResult(
        n=checkpoints.astype(int),
        price=np.array(prices, dtype=float),
        ci_low=np.array(ci_low, dtype=float),
        ci_high=np.array(ci_high, dtype=float),
        std_error=np.array(se_list, dtype=float),
    )
    return conv, st, x


# ---------------------------------------------------------------------
# Dash App UI
# ---------------------------------------------------------------------
app = Dash(__name__)
app.title = "Monte Carlo Options Engine - Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "20px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Monte Carlo Options Engine - Dashboard"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "14px"},
                    children=[
                        html.H4("Parameter"),
                        html.Label("Option Type"),
                        dcc.RadioItems(
                            id="option_type",
                            options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}],
                            value="call",
                            inline=True,
                        ),
                        html.Br(),
                        html.Label("Spot (S0)"),
                        dcc.Slider(id="spot", min=10, max=300, step=1, value=100, tooltip={"placement": "bottom"}),
                        html.Br(),
                        html.Label("Strike (K)"),
                        dcc.Slider(id="strike", min=10, max=300, step=1, value=100, tooltip={"placement": "bottom"}),
                        html.Br(),
                        html.Label("Volatilität (σ)"),
                        dcc.Slider(id="vol", min=0.0, max=1.0, step=0.01, value=0.20, tooltip={"placement": "bottom"}),
                        html.Br(),
                        html.Label("Laufzeit (T in Jahren)"),
                        dcc.Slider(id="ttm", min=0.01, max=5.0, step=0.01, value=1.0, tooltip={"placement": "bottom"}),
                        html.Br(),
                        html.Label("Zinssatz (r)"),
                        dcc.Slider(id="rate", min=-0.02, max=0.10, step=0.001, value=0.01, tooltip={"placement": "bottom"}),
                    ],
                ),
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "14px"},
                    children=[
                        html.H4("Simulation"),
                        html.Label("Anzahl Pfade (n_paths)"),
                        dcc.Slider(
                            id="n_paths",
                            min=5_000,
                            max=300_000,
                            step=5_000,
                            value=100_000,
                            tooltip={"placement": "bottom"},
                        ),
                        html.Br(),
                        html.Label("Seed (Reproduzierbarkeit)"),
                        dcc.Input(id="seed", type="number", value=42, style={"width": "120px"}),
                        html.Br(),
                        html.Br(),
                        dcc.Checklist(
                            id="vr_opts",
                            options=[
                                {"label": " Antithetic Variates", "value": "antithetic"},
                                {"label": " Control Variate (E[e^{-rT} S_T] = S0)", "value": "control"},
                            ],
                            value=["antithetic", "control"],
                        ),
                        html.Br(),
                        html.Button("Run Simulation", id="run_btn", n_clicks=0, style={"padding": "10px 14px"}),
                        html.Div(id="result_box", style={"marginTop": "12px", "fontSize": "14px"}),
                    ],
                ),
            ],
        ),
        html.Br(),
        dcc.Graph(id="convergence_graph"),
        html.Br(),
        dcc.Graph(id="dist_graph"),
        html.Div(
            style={"marginTop": "10px", "color": "#555", "fontSize": "12px"},
            children=[
                "Hinweis: Monte Carlo ist ein Schätzer. Je mehr Pfade, desto stabiler (kleiner Standard Error). "
                "Antithetic + Control Variate reduzieren Varianz deutlich."
            ],
        ),
    ],
)


@app.callback(
    Output("result_box", "children"),
    Output("convergence_graph", "figure"),
    Output("dist_graph", "figure"),
    Input("run_btn", "n_clicks"),
    State("option_type", "value"),
    State("spot", "value"),
    State("strike", "value"),
    State("rate", "value"),
    State("vol", "value"),
    State("ttm", "value"),
    State("n_paths", "value"),
    State("seed", "value"),
    State("vr_opts", "value"),
)
def run_simulation(
    n_clicks: int,
    option_type: str,
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    ttm: float,
    n_paths: int,
    seed: Optional[int],
    vr_opts: list[str],
):
    if n_clicks == 0:
        empty_fig = go.Figure().update_layout(template="plotly_white")
        return "Klicke auf „Run Simulation“.", empty_fig, empty_fig

    opt: OptionType = "call" if option_type == "call" else "put"
    antithetic = "antithetic" in vr_opts
    control = "control" in vr_opts

    # Black-Scholes benchmark
    bs = black_scholes_price(
        spot=float(spot),
        strike=float(strike),
        rate=float(rate),
        time_to_maturity=float(ttm),
        volatility=float(vol),
        option_type=opt,
    )

    conv, st, x = _mc_convergence(
        spot=float(spot),
        strike=float(strike),
        rate=float(rate),
        time_to_maturity=float(ttm),
        volatility=float(vol),
        option_type=opt,
        n_paths=int(n_paths),
        seed=int(seed) if seed is not None else None,
        antithetic=bool(antithetic),
        control_variate=bool(control),
        confidence_level=0.95,
        n_checkpoints=14,
    )

    mc_price = float(conv.price[-1])
    mc_se = float(conv.std_error[-1])
    mc_ci = (float(conv.ci_low[-1]), float(conv.ci_high[-1]))
    diff = mc_price - float(bs)

    # ---------------- Convergence Figure ----------------
    fig_conv = go.Figure()

    fig_conv.add_trace(
        go.Scatter(
            x=conv.n,
            y=conv.price,
            mode="lines+markers",
            name="Monte Carlo Preis",
        )
    )

    fig_conv.add_trace(
        go.Scatter(
            x=np.concatenate([conv.n, conv.n[::-1]]),
            y=np.concatenate([conv.ci_high, conv.ci_low[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,0,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="95% CI",
        )
    )

    fig_conv.add_trace(
        go.Scatter(
            x=conv.n,
            y=[bs] * len(conv.n),
            mode="lines",
            name="Black-Scholes (Benchmark)",
        )
    )

    fig_conv.update_layout(
        template="plotly_white",
        title="Konvergenz: Preis vs. Anzahl Pfade (log-Skala)",
        xaxis_title="n_paths",
        yaxis_title="Optionspreis",
        xaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---------------- Distribution Figure ----------------
    fig_dist = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Terminalpreise S_T", "Discounted Payoffs"),
    )

    fig_dist.add_trace(go.Histogram(x=st, nbinsx=50, name="S_T"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=x, nbinsx=50, name="disc payoff"), row=1, col=2)

    fig_dist.update_layout(
        template="plotly_white",
        title="Verteilungen (eine Simulation mit n_paths)",
        showlegend=False,
    )
    fig_dist.update_xaxes(title_text="S_T", row=1, col=1)
    fig_dist.update_xaxes(title_text="discounted payoff", row=1, col=2)

    # ---------------- Result box ----------------
    vr_text = []
    if antithetic:
        vr_text.append("Antithetic")
    if control:
        vr_text.append("Control Variate")
    vr_text = ", ".join(vr_text) if vr_text else "keine"

    result = html.Div(
        children=[
            html.Div(f"Black-Scholes: {bs:.6f}"),
            html.Div(f"Monte Carlo:  {mc_price:.6f}  (SE={mc_se:.6f}, 95% CI=[{mc_ci[0]:.6f}, {mc_ci[1]:.6f}])"),
            html.Div(f"Diff (MC - BS): {diff:+.6f}"),
            html.Div(f"Variance Reduction: {vr_text}"),
        ]
    )

    return result, fig_conv, fig_dist


if __name__ == "__main__":
    app.run(debug=True)