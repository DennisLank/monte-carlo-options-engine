# Monte Carlo Options Engine

A modular quantitative finance project for pricing European options with Monte Carlo simulation and Black-Scholes as an analytical benchmark.

This repository is being built as a clean, testable and extensible Python project focused on numerical option pricing, model transparency and interactive visualization.

---

## Project Status

This project is currently in early development.

The repository structure is in place and the initial pricing layer is being implemented step by step with a focus on:

- correctness
- clear architecture
- reproducibility
- maintainability

### Implemented so far

- project structure and initial setup
- Black-Scholes pricing module for European options
- first unit tests for pricing logic and edge cases

### Next steps

- Monte Carlo pricing under geometric Brownian motion
- pricing comparison between Monte Carlo and Black-Scholes
- confidence intervals and convergence analysis
- interactive frontend for visualization

---

## Project Goal

The purpose of this project is to build a structured options pricing engine that can be extended over time.

Rather than a one-off script or notebook, the repository is intended to demonstrate:

- analytical pricing with Black-Scholes
- Monte Carlo simulation for European options
- transparent comparison between analytical and simulated prices
- clean Python backend design
- testable quantitative logic
- interactive financial visualizations

---

## Planned Core Functionality

The initial scope includes:

- Black-Scholes pricing for European call and put options
- Monte Carlo pricing under geometric Brownian motion
- comparison of Monte Carlo results with Black-Scholes prices
- confidence intervals for simulated prices
- reproducible simulations using fixed random seeds

Planned future extensions include:

- Greeks calculation
- implied volatility solver
- variance reduction techniques
- sensitivity analysis
- additional payoff structures

---

## Frontend

A Dash-based frontend is planned to make the pricing logic and simulation behavior easy to explore and understand.

Planned inputs include:

- spot price
- strike price
- volatility
- risk-free rate
- time to maturity
- number of simulation paths

Planned visual outputs include:

- simulated price paths
- terminal price distribution
- payoff distribution
- convergence of Monte Carlo estimates
- parameter sensitivity plots

---

## Project Structure

The repository is organized with a modular layout to keep pricing logic, frontend code and testing clearly separated.

    monte-carlo-options-engine/
    ├─ README.md
    ├─ requirements.txt
    ├─ .gitignore
    ├─ src/
    │  ├─ pricing/
    │  ├─ utils/
    │  └─ app/
    ├─ tests/
    └─ assets/

This structure is intended to support clean separation of concerns from the beginning of the project.

---

## Technology Stack

### Current
- Python
- NumPy
- SciPy
- pytest

### Planned for the interactive layer
- Plotly
- Dash

---

## Development Principles

This project is being built with an engineering-first approach.

Key principles include:

- modular design
- testable logic
- explicit model assumptions
- reproducible results
- clean separation between backend and frontend
- extensibility for future quantitative features

---

## Roadmap

### Phase 1
- set up repository structure
- implement Black-Scholes pricing
- add first unit tests

### Phase 2
- implement Monte Carlo pricing for European options
- add convergence and distribution analysis
- improve validation and error handling

### Phase 3
- build the Dash frontend
- add Greeks
- add implied volatility
- implement variance reduction techniques
- expand documentation and usage examples

---

## Disclaimer

This project is intended as a quantitative finance software project for learning, engineering practice and portfolio development.

It is not financial advice and is not intended for live trading or investment decisions without further validation and review.

---

## Author

Dennis Marvin Lank