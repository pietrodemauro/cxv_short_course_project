import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eco-Driving: Optimal Train Position, Speed and Command Profiles

    This notebook formulates and solves an **eco-driving trajectory optimization problem** for a train moving along a track of length $D$ within a scheduled arrival time $T$.

    The goal is to compute the optimal traction and braking profile $u(t)$ that satisfies:
    1. System kinematic and dynamic equations (with viscous friction).
    2. Boundary conditions (starting and arriving at rest at prescribed coordinates).
    3. State and control limits, including piecewise speed limits across track sections.
    4. A multi-objective cost balancing control effort and passenger ride comfort (penalizing jerk / abrupt control variations).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import dccp
    import matplotlib.pyplot as plt

    return cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Parameters & Scenario Configuration

    Adjust the sliders below to explore how the total travel distance $D$ and the allocated schedule time $T$ affect the optimal profile.
    """)
    return


@app.cell
def _(mo):
    slider_D = mo.ui.slider(
        start=1000.0,
        stop=5000.0,
        step=100.0,
        value=3000.0,
        label="Track Distance D [m]",
        show_value=True,
    )

    slider_T = mo.ui.slider(
        start=60.0,
        stop=300.0,
        step=5.0,
        value=180.0,
        label="Travel Time T [s]",
        show_value=True,
    )

    mo.vstack([
        mo.md("### Problem Parameters"),
        slider_D,
        slider_T,
    ])
    return slider_D, slider_T


@app.cell
def _(np, slider_D, slider_T):
    # Retrieve slider values
    D = float(slider_D.value)       # Track length [m]
    T = float(slider_T.value)       # Total travel time [s]

    # Vector of regularization terms gamma to compare
    gamma_values = [0.0, 1.0, 10.0, 100.0]

    # Discretization parameters
    N = 200                         # Number of discretization intervals
    dt = T / N                      # Sampling period [s]

    # Vehicle and physical limits
    c = 0.003                        # Normalized viscous drag coefficient [1/s]
    v_max = 25.0                    # Global maximum allowable speed [m/s] (~90 km/h)
    u_max = 1.0                     # Maximum acceleration (traction limit) [m/s^2]
    u_min = -1.2                    # Maximum deceleration (braking limit) [m/s^2]

    # Time grids for plotting and analysis
    tempo = np.linspace(0, T, N + 1)
    tempo_u = np.linspace(0, T - dt, N)
    return D, N, T, c, dt, gamma_values, tempo, tempo_u, u_max, u_min, v_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mathematical Formulation

    ### 1. Decision Variables
    The continuous time horizon $[0, T]$ is divided into $N$ equal stages of step size $\Delta t = T / N$.
    The optimization variables are:
    * $\mathbf{x} = [x_0, x_1, \dots, x_N]^T \in \mathbb{R}^{N+1}$: Position of the train at stage $k$ [m].
    * $\mathbf{v} = [v_0, v_1, \dots, v_N]^T \in \mathbb{R}^{N+1}$: Speed of the train at stage $k$ [m/s].
    * $\mathbf{u} = [u_0, u_1, \dots, u_{N-1}]^T \in \mathbb{R}^N$: Acceleration / deceleration control input during interval $k$ [$\text{m/s}^2$].

    ---

    ### 2. Constraints
    #### a. System Dynamics
    Using forward linear dynamics:
    $$\begin{aligned}
    x_{k+1} &= x_k + v_k \Delta t, \quad &k = 0, \dots, N-1 \\
    v_{k+1} &= v_k + (u_k - c v_k) \Delta t, \quad &k = 0, \dots, N-1
    \end{aligned}$$
    where $c$ is the normalized viscous resistance coefficient.

    #### b. Boundary & Operating Constraints
    * **Boundary conditions**: Train starts and finishes at a complete stop:
      $$x_0 = 0, \quad v_0 = 0, \quad x_N = D, \quad v_N = 0$$
    * **Non-negative speed and section speed limits**:
      $$0 \le v_k \le v_{\max, k}, \quad k = 0, \dots, N$$
    * **Actuation limits**:
      $$u_{\min} \le u_k \le u_{\max}, \quad k = 0, \dots, N-1$$

    ---

    ### 3. Objective Function
    The objective balances aggregate control energy with passenger comfort:
    $$\min_{\mathbf{x}, \mathbf{v}, \mathbf{u}} \; J(\mathbf{u}) = \sum_{k=0}^{N-1} u_k^2 + \gamma \sum_{k=0}^{N-2} (u_{k+1} - u_k)^2$$
    where:
    * $\sum_{k=0}^{N-1} u_k^2$ minimizes control effort and promotes smooth acceleration.
    * $\gamma \sum_{k=0}^{N-2} (u_{k+1} - u_k)^2$ **Regularization term:** penalizes control rate of change (discrete jerk), smoothing transitions between acceleration, cruising, and braking phases.
    """)
    return


@app.cell
def _(D, N, T, c, cp, dt, gamma_values, mo, np, u_max, u_min, v_max):
    # 1. Sectional Speed Limits (v_max_array)
    v_max_array = np.full(N + 1, v_max)

    # Section 1: [0, T/4] -> 10.0 m/s (e.g., station departure / slow zone)
    idx_s1_start = 0
    idx_s1_end = int((T / 4) / dt)
    v_max_array[idx_s1_start:idx_s1_end] = 10.0

    # Section 2: [T/4, 3*T/4] -> 25.0 m/s (main open track)
    idx_s2_start = idx_s1_end
    idx_s2_end = int((3 * T / 4) / dt)
    v_max_array[idx_s2_start:idx_s2_end] = 25.0

    # Section 3: [3*T/4, T] -> 15.0 m/s (station approach / speed restriction)
    idx_s3_start = idx_s2_end
    idx_s3_end = N + 1
    v_max_array[idx_s3_start:idx_s3_end] = 15.0

    results = {}
    statuses = []

    for _g_val in gamma_values:
        # Decision Variables
        _x_var = cp.Variable(N + 1, name=f"position_gamma_{_g_val}")
        _v_var = cp.Variable(N + 1, name=f"velocity_gamma_{_g_val}")
        _u_var = cp.Variable(N, name=f"control_gamma_{_g_val}")

        # Constraints Definition
        _constraints = [
            # Boundary conditions
            _x_var[0] == 0,
            _v_var[0] == 0,
            _x_var[N] == D,
            _v_var[N] == 0,

            # State transition dynamics
            _x_var[1:] == _x_var[:-1] + _v_var[:-1] * dt,
            _v_var[1:] == _v_var[:-1] + (_u_var - c * _v_var[:-1]) * dt,

            # Operating and physical limits
            _v_var >= 0,
            _v_var <= v_max_array,
            _u_var >= u_min,
            _u_var <= u_max,
        ]

        # Objective Function: Effort + Jerk (Comfort) Penalty
        _cost = cp.sum_squares(_u_var) + _g_val * cp.sum_squares(_u_var[1:] - _u_var[:-1])

        _prob = cp.Problem(cp.Minimize(_cost), _constraints)
        _prob.solve()

        if _prob.status in ["optimal", "optimal_inaccurate"] and _x_var.value is not None:
            results[_g_val] = {
                "x": np.array(_x_var.value),
                "v": np.array(_v_var.value),
                "u": np.array(_u_var.value),
                "cost": _prob.value,
                "status": _prob.status,
            }
            statuses.append(f"- **γ = {_g_val}**: `{_prob.status}` (Optimal Cost: `{_prob.value:.2f}`)")
        else:
            statuses.append(f"- **γ = {_g_val}**: `{_prob.status}` (Failed / Infeasible)")

    mo.md(r"### Solver Results per $\gamma$:" + "\n" + "\n".join(statuses))
    return results, v_max_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimization Results & Trajectory Profiles

    Below are the comparative profiles for position $x(t)$, speed $v(t)$ (with sectional speed limits), and control input $u(t)$ across all values of $\gamma$.
    """)
    return


@app.cell
def _(D, np, plt, results, tempo, tempo_u, u_max, u_min, v_max_array):
    if not results:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No optimal solutions found for the specified gamma values", ha="center", va="center")
        ax.axis("off")
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        cmap = plt.get_cmap("viridis")
        gamma_list = list(results.keys())
        colors = [cmap(i) for i in np.linspace(0.2, 0.85, max(1, len(gamma_list)))]

        # 1. Position Trajectory
        for _color, (_g_val, _sol) in zip(colors, results.items()):
            axs[0].plot(tempo, _sol["x"], label=f"γ = {_g_val}", color=_color, linewidth=2)
        axs[0].axhline(D, color="black", linestyle="--", alpha=0.7, label=f"Destination (D = {D:.0f} m)")
        axs[0].set_ylabel("Position [m]")
        axs[0].grid(True, linestyle=":", alpha=0.6)
        axs[0].legend(loc="best")
        axs[0].set_title(r"Optimal Trajectory Profiles for Different $\gamma$", fontsize=14, fontweight="bold")

        # 2. Speed Profile vs Sectional Limits
        for _color, (_g_val, _sol) in zip(colors, results.items()):
            axs[1].plot(tempo, _sol["v"], label=f"γ = {_g_val}", color=_color, linewidth=2)
        axs[1].step(tempo, v_max_array, color="red", linestyle="--", where="post", label="Speed Limit", alpha=0.7)
        axs[1].set_ylabel("Speed [m/s]")
        axs[1].grid(True, linestyle=":", alpha=0.6)
        axs[1].legend(loc="best")

        # 3. Control Acceleration / Deceleration Profile
        for _color, (_g_val, _sol) in zip(colors, results.items()):
            axs[2].plot(tempo_u, _sol["u"], label=f"γ = {_g_val}", color=_color, linewidth=1.8)
        axs[2].axhline(u_max, color="red", linestyle="--", alpha=0.7, label=f"Max Traction ({u_max} m/s²)")
        axs[2].axhline(u_min, color="purple", linestyle="--", alpha=0.7, label=f"Max Braking ({u_min} m/s²)")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Acceleration [m/s²]")
        axs[2].grid(True, linestyle=":", alpha=0.6)
        axs[2].legend(loc="best")

        plt.tight_layout()

    fig
    return


if __name__ == "__main__":
    app.run()
