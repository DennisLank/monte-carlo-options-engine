# Monte Carlo Options Engine

A modular quantitative finance project for pricing European options with Monte Carlo simulation and Black-Scholes as an analytical benchmark.

The goal of this repository is to build a clean, testable and extensible Python-based pricing engine that combines numerical simulation, analytical pricing and interactive visualization.

---

## Project Status

This project is currently in its early development phase.

At this stage, the repository has been set up with a modular structure to separate pricing logic, utilities, application code and tests from the beginning.

The implementation is being developed step by step with a focus on:

- correctness
- clear architecture
- reproducibility
- maintainability

---

## Project Goal

The purpose of this project is to create a structured options pricing engine that can be extended over time.

Instead of building a one-off script or notebook, the project is intended to demonstrate:

- analytical pricing with Black-Scholes
- Monte Carlo simulation for European options
- transparent comparison between analytical and simulated prices
- clean Python backend design
- interactive financial visualizations
- test-driven quantitative development

---

## Planned Core Functionality

The initial development scope includes:

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

## Frontend Concept

A Dash-based frontend is planned to make the pricing logic and simulation behavior easy to explore and understand.

The interface is intended to provide interactive inputs for:

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

Current structure:

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

The project is planned around the following Python stack:

- Python
- NumPy
- SciPy
- Plotly
- Dash
- pytest

These tools were chosen to support numerical accuracy, interactive visualization and maintainable code design.

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
- implement Monte Carlo pricing for European options
- add first unit tests

### Phase 2
- build the Dash frontend
- add convergence and distribution visualizations
- improve validation and error handling

### Phase 3
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