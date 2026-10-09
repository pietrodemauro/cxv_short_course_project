import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eco-Driving: Optimal Train Position, Speed and Command Profiles (Nonlinear Davis Dynamics)

    This notebook formulates and solves an **eco-driving trajectory optimization problem** for a train moving along a flat track of length $D$ within a scheduled arrival time $T$.

    Key aspects of this formulation:
    1. **Full Nonlinear Davis Formula**: The motion resistance includes constant, linear, and quadratic aerodynamic drag:
       $$R(v) = a_0 + a_1 v + a_2 v^2$$
    2. **Non-Convex / DCCP Optimization**: Because the quadratic speed term enters the state transition equality constraint, standard DCP rules are violated. The problem is solved without any linear approximation using **DCCP** (Disciplined Convex-Concave Programming).
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

    Adjust the sliders below to explore how total travel distance $D$, allocated travel time $T$, and comfort weight $\gamma$ affect the optimal profile.
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

    slider_gamma = mo.ui.slider(
        start=0.0,
        stop=50.0,
        step=0.5,
        value=10.0,
        label="Comfort Weight γ (Jerk penalty)",
        show_value=True,
    )

    mo.vstack([
        mo.md("### Problem Parameters"),
        slider_D,
        slider_T,
        slider_gamma,
    ])
    return slider_D, slider_T, slider_gamma


@app.cell
def _(np, slider_D, slider_T, slider_gamma):
    # Retrieve slider values
    D = float(slider_D.value)       # Track length [m]
    T = float(slider_T.value)       # Total travel time [s]
    gamma = float(slider_gamma.value)  # Weight on control rate-of-change (comfort)

    # Discretization parameters
    N = 500                         # Number of discretization intervals
    dt = T / N                      # Sampling interval [s]

    # Full nonlinear Davis resistance coefficients per unit mass [m/s^2]:
    # R(v) = a0 + a1 * v + a2 * v^2
    a0 = 0.01                       # Mechanical rolling resistance [m/s^2]
    a1 = 0.001                      # Linear mechanical drag [1/s]
    a2 = 0.0003                     # Quadratic aerodynamic drag [1/m]

    # Vehicle operational and physical limits
    v_max = 25.0                    # Global maximum allowable speed [m/s] (~90 km/h)
    u_max = 1.0                     # Maximum traction acceleration [m/s^2]
    u_min = -1.2                    # Maximum braking deceleration [m/s^2]

    # Time grids for plotting and analysis
    tempo = np.linspace(0, T, N + 1)
    tempo_u = np.linspace(0, T - dt, N)
    return D, N, T, a0, a1, a2, dt, gamma, tempo, tempo_u, u_max, u_min, v_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mathematical Formulation & DCCP Structure

    ### 1. Decision Variables
    * $\mathbf{x} = [x_0, \dots, x_N]^T \in \mathbb{R}^{N+1}$: Position trajectory [$m$].
    * $\mathbf{v} = [v_0, \dots, v_N]^T \in \mathbb{R}^{N+1}$: Speed trajectory [$m/s$].
    * $\mathbf{u} = [u_0, \dots, u_{N-1}]^T \in \mathbb{R}^N$: Acceleration / deceleration command [$\text{m/s}^2$].

    ---

    ### 2. Nonlinear System Dynamics
    Using forward Euler discretization:
    $$\begin{aligned}
    x_{k+1} &= x_k + v_k \Delta t \\
    v_{k+1} &= v_k + \left(u_k - (a_0 + a_1 v_k + a_2 v_k^2)\right) \Delta t
    \end{aligned}$$

    ---

    ### 3. Understanding the Difference-of-Convex (DC) Decomposition
    In standard Disciplined Convex Programming (DCP), all equality constraints must be strictly **affine**.
    Rearranging the velocity dynamic equation yields:
    $$\underbrace{\big(v_{k+1} - v_k - u_k \Delta t + a_0 \Delta t + a_1 v_k \Delta t\big)}_{\text{Affine term } f_{\text{aff}}} + \underbrace{a_2 \Delta t \, v_k^2}_{\text{Convex term } f_{\text{cvx}}} = 0$$

    An equality $f_{\text{aff}} + f_{\text{cvx}} = 0$ is equivalent to two simultaneous inequalities:
    1. **Convex inequality**:
       $$g_1(v, u) \le 0 \implies \big(f_{\text{aff}} + a_2 \Delta t \, v_k^2\big) \le 0$$
       Here $g_1$ is convex, which is fully compliant with DCP.
    2. **Non-convex / Difference-of-Convex inequality**:
       $$g_2(v, u) - h_2(v, u) \le 0 \implies \underbrace{-f_{\text{aff}}}_{g_2(v, u)\text{ (affine/convex)}} - \underbrace{a_2 \Delta t \, v_k^2}_{h_2(v, u)\text{ (convex)}} \le 0$$

       Notice that:
       * $g_2(v, u) = -v_{k+1} + v_k + u_k \Delta t - a_0 \Delta t - a_1 v_k \Delta t$ is **convex** (affine).
       * $h_2(v, u) = a_2 \Delta t \, v_k^2$ is **convex** (since $a_2 > 0$ and $v_k^2$ is convex).
       * Subtracting $h_2$ makes this a canonical **Difference-of-Convex (DC)** constraint: $g_2 - h_2 \le 0$.

    ### 4. Boundary & Operating Constraints
    * **Boundary conditions**: Train starts and finishes at a complete stop:
      $$x_0 = 0, \quad v_0 = 0, \quad x_N = D, \quad v_N = 0$$
    * **Non-negative speed and section speed limits**:
      $$0 \le v_k \le v_{\max, k}, \quad k = 0, \dots, N$$
    * **Actuation limits**:
      $$u_{\min} \le u_k \le u_{\max}, \quad k = 0, \dots, N-1$$

    ---

    ### 5. Objective Function
    $$\min_{\mathbf{x}, \mathbf{v}, \mathbf{u}} \; \sum_{k=0}^{N-1} u_k^2 + \gamma \sum_{k=0}^{N-2} (u_{k+1} - u_k)^2$$
    """)
    return


@app.cell
def _(D, N, T, a0, a1, a2, cp, dt, gamma, mo, np, u_max, u_min, v_max):
    # 1. Decision Variables
    x = cp.Variable(N + 1, name="position")
    v = cp.Variable(N + 1, name="velocity")
    u = cp.Variable(N, name="control")

    # 2. Physics-Consistent Initial Guess for DCCP Warm-Start
    tempo_grid = np.linspace(0, T, N + 1)
    v_guess = (np.pi * D / (2 * T)) * np.sin(np.pi * tempo_grid / T)
    x_guess = (D / 2) * (1 - np.cos(np.pi * tempo_grid / T))
    u_guess = np.diff(v_guess) / dt

    v.value = v_guess
    x.value = x_guess
    u.value = u_guess

    # 3. Piecewise Sectional Speed Limits (v_max_array)
    v_max_array = np.full(N + 1, v_max)
    idx_s1_end = int((T / 4) / dt)
    idx_s2_end = int((3 * T / 4) / dt)
    v_max_array[:idx_s1_end] = 10.0
    v_max_array[idx_s1_end:idx_s2_end] = 25.0
    v_max_array[idx_s2_end:] = 15.0

    # 4. Full Nonlinear Davis Resistance: R(v) = a0 + a1 * v + a2 * v^2
    davis_friction = a0 + a1 * v[:-1] + a2 * cp.square(v[:-1])

    # 5. Constraints Definition
    constraints = [
        # Boundary conditions
        x[0] == 0,
        v[0] == 0,
        x[N] == D,
        v[N] == 0,


        # Kinematic and dynamic state transitions
        x[1:] == x[:-1] + v[:-1] * dt,
        v[1:] == v[:-1] + (u - davis_friction) * dt,

        # Operating and physical limits
        v >= 0,
        v <= v_max_array,
        u >= u_min,
        u <= u_max,
    ]

    # 6. Objective Function: Traction Energy + Comfort (Jerk) Penalty
    cost = cp.sum_squares(u) + gamma * cp.sum_squares(u[1:] - u[:-1])

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # 7. Non-Convex Solve via DCCP with Conic Interior-Point Solver (CLARABEL)
    prob.solve(method="dccp", solver=cp.CLARABEL, max_iter=50)
    #prob.solve(nlp=True, verbose=True)

    if prob.status in ["optimal", "optimal_inaccurate", "converged"]:
        status_badge = f"**Solver Status:** `{prob.status}` | **Optimal Cost:** `{prob.value:.2f}` (Solved via DCCP)"
    else:
        status_badge = f"**Solver Status:** `{prob.status}` (Problem did not converge or is infeasible for selected parameters)."

    mo.md(status_badge)
    return prob, u, v, v_max_array, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimization Results & Trajectory Profiles

    Below are the resulting trajectories for position $x(t)$, velocity $v(t)$ (compared against speed limits), and control input $u(t)$.
    """)
    return


@app.cell
def _(D, plt, prob, tempo, tempo_u, u, u_max, u_min, v, v_max_array, x):
    if prob.status not in ["optimal", "optimal_inaccurate", "converged"] or x.value is None:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Infeasible or unsolved problem for current parameters", ha="center", va="center")
        ax.axis("off")
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # 1. Position Trajectory
        axs[0].plot(tempo, x.value, label="Position x(t)", color="#1f77b4", linewidth=2)
        axs[0].axhline(D, color="gray", linestyle="--", label=f"Destination (D = {D:.0f} m)")
        axs[0].set_ylabel("Position [m]")
        axs[0].grid(True, linestyle=":", alpha=0.6)
        axs[0].legend(loc="best")
        axs[0].set_title("Optimal Trajectory Profiles (Nonlinear Davis Dynamics)", fontsize=13, fontweight="bold")

        # 2. Velocity Profile vs Sectional Limits
        axs[1].plot(tempo, v.value, label="Speed v(t)", color="#ff7f0e", linewidth=2)
        axs[1].step(tempo, v_max_array, color="#d62728", linestyle="--", where="post", label="Section Speed Limit")
        axs[1].set_ylabel("Speed [m/s]")
        axs[1].grid(True, linestyle=":", alpha=0.6)
        axs[1].legend(loc="best")

        # 3. Control Acceleration / Deceleration Profile
        axs[2].plot(tempo_u, u.value, label="Control u(t)", color="#2ca02c", linewidth=2)
        axs[2].axhline(u_max, color="#d62728", linestyle="--", label=f"Max Traction ({u_max} m/s²)")
        axs[2].axhline(u_min, color="#9467bd", linestyle="--", label=f"Max Braking ({u_min} m/s²)")
        axs[2].fill_between(tempo_u, 0, u.value, where=(u.value > 0.01), color="#2ca02c", alpha=0.25, label="Traction Phase")
        axs[2].fill_between(tempo_u, 0, u.value, where=(u.value < -0.01), color="#d62728", alpha=0.2, label="Braking Phase")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Acceleration [m/s²]")
        axs[2].grid(True, linestyle=":", alpha=0.6)
        axs[2].legend(loc="best")

        plt.tight_layout()

    fig
    return


if __name__ == "__main__":
    app.run()
