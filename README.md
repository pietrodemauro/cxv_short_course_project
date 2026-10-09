# Train Eco-Driving Optimization 

Optimal train trajectory and eco-driving profiles using mathematical optimization and interactive [marimo](https://marimo.io) notebooks.

## Overview

This project computes optimal traction and braking profiles $u(t)$ for a single-mass point-train model traveling along a corridor of length $D$ within an allocated arrival time $T$. The objective balances energy/control effort with passenger ride comfort (penalizing jerk / abrupt control variations), subject to boundary conditions, actuator limits, and speed restrictions.

## Notebooks

- **`train-echo-driving-single-mass.py`**: Linear / convex formulation with viscous friction and speed limits. Solved directly with standard convex programming via [CVXPY](https://www.cvxpy.org/).
- **`train-echo-driving-single-mass_nonlinear_dccp.py`**: Non-convex formulation incorporating the full nonlinear Davis resistance equation ($R(v) = a_0 + a_1 v + a_2 v^2$). Solved using [DCCP](https://github.com/cvxgrp/dccp) (Disciplined Convex-Concave Programming).
- **`train-echo-driving-single-mass_nonlinear_dnlp.py`**: Advanced formulation comparing temporal vs. spatial domain constraints, including sharp position-dependent speed limits and track grade / elevation profile variations.

## Requirements & Setup

Install dependencies:

```bash
pip install marimo cvxpy dccp numpy matplotlib
```

Run any notebook interactively with `marimo`:

```bash
marimo run train-echo-driving-single-mass.py
# or to edit:
marimo edit train-echo-driving-single-mass.py
```

