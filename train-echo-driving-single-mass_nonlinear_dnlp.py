import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eco-Driving: Optimal Train Position, Speed and Command Profiles (DCCP vs. CVXPY DNLP)

    This notebook investigates the **eco-driving trajectory optimization problem** for a train governed by nonlinear dynamics.

    ---

    ### Motivation & Equivalence between Time and Space Formulations
    1. **The Reference DCCP Model (Time Domain Constraints)**:speed limits are imposed at predetermined temporal switching times on a flat track:
       * $t \in [0, T/4) \implies v(t) \le 10.0\text{ m/s}$
       * $t \in [T/4, 3T/4) \implies v(t) \le 25.0\text{ m/s}$
       * $t \in [3T/4, T] \implies v(t) \le 15.0\text{ m/s}$

       Because limits are fixed across the time grid and the track is flat, this problem satisfies **DCCP (Disciplined Convex-Concave Programming)** rules and is solved via successive convexifications (`method="dccp"`).

    2. **The Equivalent Spatial Formulation with Track Elevation (CVXPY DNLP)**:
       In real-world railway corridors:
       * **Sharp Spatial Speed Limits**: Speed limits physically depend on track position $x$. To compare directly with the reference DCCP trajectory, we map the temporal switching instances $t_1 = T/4$ and $t_2 = 3T/4$ to their **exact spatial counterparts**:
         $$x_1 = x_{\text{DCCP}}(T/4), \qquad x_2 = x_{\text{DCCP}}(3T/4)$$
         The spatial speed limits switch sharply at these exact spatial coordinates:
         $$v(t) \le v_{\max}(x(t)) = \begin{cases}
         10.0\text{ m/s} & \text{if } x(t) < x_1 \\
         25.0\text{ m/s} & \text{if } x_1 \le x(t) < x_2 \\
         15.0\text{ m/s} & \text{if } x(t) \ge x_2
         \end{cases}$$
       * **Track Grade / Elevation Slope**: Real tracks feature elevation variations (hills and valleys). Modeling the track elevation profile $h(x)$ along the corridor of length $D$ as:
         $$h(x) = \frac{H_{\text{hill}}}{2} \left[1 - \cos\left(2\pi \frac{x}{D}\right)\right]$$
         the local slope (grade) is given by its spatial derivative:
         $$\text{slope}(x) = \frac{dh}{dx}(x) = \frac{\pi H_{\text{hill}}}{D} \sin\left(2\pi \frac{x}{D}\right)$$
         For small incline angles $\theta(x) \approx \frac{dh}{dx}$, the component of gravitational acceleration acting along the direction of motion is:
         $$a_{\text{grade}}(x) = g \sin\theta(x) \approx g \frac{dh}{dx}(x) = g \frac{\pi H_{\text{hill}}}{D} \sin\left(2\pi \frac{x}{D}\right)$$
         which opposes traction uphill and aids motion downhill, coupling nonlinearly into the discrete velocity dynamics:
         $$v_{k+1} = v_k + \Big(u_k - \big(a_0 + a_1 v_k + a_2 v_k^2\big) - a_{\text{grade}}(x_k)\Big) \Delta t$$

    3. **Why this requires DNLP**:
       Both the sharp spatial limit $v_{\max}(x_k)$ and the grade term $a_{\text{grade}}(x_k)$ depend nonlinearly on the position state variable $x_k$. This breaks DCP and DCCP rules.
       Using **CVXPY DNLP (`nlp=True`)**, we model the sharp step transitions smoothly via steep hyperbolic tangent steps (`cp.nlp.tanh`), and solve the fully coupled nonlinear program.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import dccp
    import matplotlib.pyplot as plt
    import time

    return cp, mo, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Scenario Configuration

    Adjust the sliders below to explore how track distance $D$, arrival schedule $T$, comfort weight $\gamma$, and hill height $H_{\text{hill}}$ affect both DCCP and DNLP formulations.
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

    slider_hill = mo.ui.slider(
        start=0.0,
        stop=30.0,
        step=2.0,
        value=15.0,
        label="Hill Peak Height H [m] (DNLP slope)",
        show_value=True,
    )

    mo.vstack([
        mo.md("### Scenario Parameters"),
        slider_D,
        slider_T,
        slider_gamma,
        slider_hill,
    ])
    return slider_D, slider_T, slider_gamma, slider_hill


@app.cell
def _(np, slider_D, slider_T, slider_gamma, slider_hill):
    # Scenario parameters
    D = float(slider_D.value)             # Track length [m]
    T = float(slider_T.value)             # Total travel time [s]
    gamma = float(slider_gamma.value)     # Jerk/comfort penalty weight
    H_hill = float(slider_hill.value)     # Peak hill height [m] for DNLP

    # Discretization parameters matching DCCP formulation
    N = 200                               # Number of temporal intervals
    dt = T / N                            # Sampling step [s]

    # Full nonlinear Davis resistance coefficients per unit mass [m/s^2]:
    # R(v) = a0 + a1 * v + a2 * v^2
    a0 = 0.01                             # Mechanical rolling friction [m/s^2]
    a1 = 0.001                            # Linear mechanical resistance [1/s]
    a2 = 0.0003                           # Quadratic aerodynamic drag [1/m]

    # Physical limits matching reference DCCP
    v_max = 25.0                          # Global maximum speed limit [m/s]
    u_max = 1.0                           # Maximum traction acceleration [m/s^2]
    u_min = -1.2                          # Maximum braking deceleration [m/s^2]
    g = 9.81                              # Gravitational acceleration [m/s^2]

    # Time grids
    tempo = np.linspace(0.0, T, N + 1)
    tempo_u = np.linspace(0.0, T - dt, N)
    return (
        D,
        H_hill,
        N,
        T,
        a0,
        a1,
        a2,
        dt,
        g,
        gamma,
        tempo,
        tempo_u,
        u_max,
        u_min,
        v_max,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Solving the Reference DCCP Model (Time-Domain Limits)

    Here we solve the benchmark formulation with temporal piecewise limits:
    $$\begin{cases}
    v(t) \le 10.0\text{ m/s} & \text{for } 0 \le t < T/4 \\
    v(t) \le 25.0\text{ m/s} & \text{for } T/4 \le t < 3T/4 \\
    v(t) \le 15.0\text{ m/s} & \text{for } 3T/4 \le t \le T
    \end{cases}$$
    """)
    return


@app.cell
def _(D, N, T, a0, a1, a2, cp, dt, gamma, mo, np, time, u_max, u_min, v_max):
    # 1. DCCP Decision Variables
    x_dccp = cp.Variable(N + 1, name="x_dccp")
    v_dccp = cp.Variable(N + 1, name="v_dccp")
    u_dccp = cp.Variable(N, name="u_dccp")

    # 2. Physics-Consistent Harmonic Initial Guess
    tempo_grid = np.linspace(0.0, T, N + 1)
    v_dccp.value = (np.pi * D / (2.0 * T)) * np.sin(np.pi * tempo_grid / T)
    x_dccp.value = (D / 2.0) * (1.0 - np.cos(np.pi * tempo_grid / T))
    u_dccp.value = np.diff(v_dccp.value) / dt

    # 3. Piecewise Temporal Limits
    idx_s1 = int((T / 4.0) / dt)
    idx_s2 = int((3.0 * T / 4.0) / dt)
    v_lim_time = np.full(N + 1, v_max)
    v_lim_time[:idx_s1] = 10.0
    v_lim_time[idx_s1:idx_s2] = 25.0
    v_lim_time[idx_s2:] = 15.0

    # 4. Davis Resistance
    davis_dccp = a0 + a1 * v_dccp[:-1] + a2 * cp.square(v_dccp[:-1])

    # 5. Constraints
    constrs_dccp = [
        x_dccp[0] == 0,
        v_dccp[0] == 0,
        x_dccp[N] == D,
        v_dccp[N] == 0,
        x_dccp[1:] == x_dccp[:-1] + v_dccp[:-1] * dt,
        v_dccp[1:] == v_dccp[:-1] + (u_dccp - davis_dccp) * dt,
        v_dccp >= 0,
        v_dccp <= v_lim_time,
        u_dccp >= u_min,
        u_dccp <= u_max,
    ]

    cost_dccp = cp.sum_squares(u_dccp) + gamma * cp.sum_squares(u_dccp[1:] - u_dccp[:-1])
    prob_dccp = cp.Problem(cp.Minimize(cost_dccp), constrs_dccp)

    t0_dccp = time.time()
    prob_dccp.solve(method="dccp", solver=cp.CLARABEL, max_iter=50)
    time_dccp = time.time() - t0_dccp

    # Extract equivalent spatial coordinates at switching instances t = T/4 and t = 3T/4
    if prob_dccp.status in ["optimal", "optimal_inaccurate", "converged"] and x_dccp.value is not None:
        x1_equiv = float(x_dccp.value[idx_s1])
        x2_equiv = float(x_dccp.value[idx_s2])
        status_dccp = (
            f"**DCCP Status:** `{prob_dccp.status}` | "
            f"**Optimal Cost:** `{prob_dccp.value:.2f}` | "
            f"**CPU Time:** `{time_dccp:.3f} s` | "
            f"**Equivalent Spatial Switching:** $x_1 = {x1_equiv:.1f}\\text{{ m}}$, $x_2 = {x2_equiv:.1f}\\text{{ m}}$"
        )
    else:
        x1_equiv = 0.15 * D
        x2_equiv = 0.85 * D
        status_dccp = f"**DCCP Status:** `{prob_dccp.status}` (Solver did not converge)"

    mo.md(status_dccp)
    return (
        prob_dccp,
        time_dccp,
        u_dccp,
        v_dccp,
        v_lim_time,
        x1_equiv,
        x2_equiv,
        x_dccp,
    )


@app.cell(hide_code=True)
def _(mo, x1_equiv, x2_equiv):
    mo.md(rf"""
    ## Step 2: Solving the Equivalent CVXPY DNLP Model (Spatial Domain Limits & Track Slope)

    Now we impose the **exact same speed limits directly in space**, with sharp switching at:
    * **Section 1:** $x < {x1_equiv:.1f}\text{{ m}} \implies v(t) \le 10.0\text{{ m/s}}$
    * **Section 2:** ${x1_equiv:.1f}\text{{ m}} \le x < {x2_equiv:.1f}\text{{ m}} \implies v(t) \le 25.0\text{{ m/s}}$
    * **Section 3:** $x \ge {x2_equiv:.1f}\text{{ m}} \implies v(t) \le 15.0\text{{ m/s}}$

    To model sharp spatial transitions smoothly in CVXPY DNLP without numerical instability, we use steep hyperbolic tangent steps ($\delta = 5.0\text{{ m}}$):
    ```python
    step1 = 0.5 * (1.0 + cp.nlp.tanh((x_dnlp - x1_equiv) / scale))
    step2 = 0.5 * (1.0 + cp.nlp.tanh((x_dnlp - x2_equiv) / scale))
    v_max_spatial = 10.0 + (25.0 - 10.0) * step1 + (15.0 - 25.0) * step2
    ```

    Additionally, the **track slope gradient** $a_{{\text{{grade}}}}(x_k)$ is coupled directly into the velocity transition equation:
    ```python
    grade_friction = g * (np.pi * H_hill / D) * cp.nlp.sin(2.0 * np.pi * x_dnlp[:-1] / D)
    v_dnlp[1:] == v_dnlp[:-1] + (u_dnlp - davis_dnlp - grade_friction) * dt
    ```
    """)
    return


@app.cell
def _(
    D,
    H_hill,
    N,
    a0,
    a1,
    a2,
    cp,
    dt,
    g,
    gamma,
    mo,
    np,
    prob_dccp,
    time,
    u_dccp,
    u_max,
    u_min,
    v_dccp,
    x1_equiv,
    x2_equiv,
    x_dccp,
):
    # 1. DNLP Decision Variables
    x_dnlp = cp.Variable(N + 1, name="x_dnlp")
    v_dnlp = cp.Variable(N + 1, name="v_dnlp")
    u_dnlp = cp.Variable(N, name="u_dnlp")

    # 2. Warm-start with the DCCP solution
    if prob_dccp.status in ["optimal", "optimal_inaccurate", "converged"] and x_dccp.value is not None:
        x_dnlp.value = x_dccp.value.copy()
        v_dnlp.value = v_dccp.value.copy()
        u_dnlp.value = u_dccp.value.copy()

    # 3. Sharp Spatial Speed Limit Transitions (scale = 5m for near-ideal sharp steps)
    sharp_scale = 5.0
    step1 = 0.5 * (1.0 + cp.nlp.tanh((x_dnlp - x1_equiv) / sharp_scale))
    step2 = 0.5 * (1.0 + cp.nlp.tanh((x_dnlp - x2_equiv) / sharp_scale))
    v_max_spatial = 10.0 + (25.0 - 10.0) * step1 + (15.0 - 25.0) * step2

    # 4. Davis Resistance
    davis_dnlp = a0 + a1 * v_dnlp[:-1] + a2 * cp.square(v_dnlp[:-1])

    # 5. Position-Dependent Gravitational Slope Force: g * sin(theta(x))
    grade_dnlp = g * (np.pi * H_hill / D) * cp.nlp.sin(2.0 * np.pi * x_dnlp[:-1] / D)

    # 6. Constraints with spatial limits and track grade
    constrs_dnlp = [
        x_dnlp[0] == 0,
        v_dnlp[0] == 0,
        x_dnlp[N] == D,
        v_dnlp[N] == 0,
        x_dnlp[1:] == x_dnlp[:-1] + v_dnlp[:-1] * dt,
        v_dnlp[1:] == v_dnlp[:-1] + (u_dnlp - davis_dnlp - grade_dnlp) * dt,
        v_dnlp >= 0,
        v_dnlp <= v_max_spatial,
        u_dnlp >= u_min,
        u_dnlp <= u_max,
    ]

    cost_dnlp = cp.sum_squares(u_dnlp) + gamma * cp.sum_squares(u_dnlp[1:] - u_dnlp[:-1])
    prob_dnlp = cp.Problem(cp.Minimize(cost_dnlp), constrs_dnlp)

    is_dcp = prob_dnlp.is_dcp()
    is_dnlp = prob_dnlp.is_dnlp()

    t0_dnlp = time.time()
    try:
        prob_dnlp.solve(nlp=True)
        time_dnlp = time.time() - t0_dnlp
        status_dnlp = (
            f"**Grammar:** `DCP = {is_dcp}` | `DNLP = {is_dnlp}` &nbsp;|&nbsp; "
            f"**DNLP Status:** `{prob_dnlp.status}` &nbsp;|&nbsp; "
            f"**Optimal Cost:** `{prob_dnlp.value:.2f}` &nbsp;|&nbsp; "
            f"**CPU Time:** `{time_dnlp:.3f} s`"
        )
    except Exception as e:
        time_dnlp = time.time() - t0_dnlp
        status_dnlp = f"**DNLP Solve Error:** `{e}`"

    mo.md(status_dnlp)
    return prob_dnlp, sharp_scale, time_dnlp, u_dnlp, v_dnlp, x_dnlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimization Results & Optimal Profiles (CVXPY DNLP)

    Below are the standalone optimal profiles for the train under position-dependent speed limits and track slope:
    1. **Position $x(t)$** reaching destination $D$ with track section markers $x_1$ and $x_2$.
    2. **Velocity $v(t)$** strictly constrained by the sharp spatial speed limit $v_{\max}(x(t))$.
    3. **Control input $u(t)$** showing traction and braking phases.
    4. **Track Elevation $h(x(t))$ & Gravitational Grade Acceleration $a_{\text{grade}}(x(t))$**.
    """)
    return


@app.cell
def _(
    D,
    H_hill,
    g,
    np,
    plt,
    prob_dnlp,
    sharp_scale,
    tempo,
    tempo_u,
    time_dnlp,
    u_dnlp,
    u_max,
    u_min,
    v_dnlp,
    x1_equiv,
    x2_equiv,
    x_dnlp,
):
    if prob_dnlp.status not in ["optimal", "optimal_inaccurate", "converged"] or x_dnlp.value is None:
        _fig_dnlp, _ax = plt.subplots(figsize=(8, 2))
        _ax.text(0.5, 0.5, "DNLP optimization did not converge.", ha="center", va="center")
        _ax.axis("off")
    else:
        _fig_dnlp, _axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

        # 1. Position Trajectory
        _axs[0].plot(tempo, x_dnlp.value, label="Position x(t)", color="#1f77b4", linewidth=2.2)
        _axs[0].axhline(x1_equiv, color="orange", linestyle=":", label=f"Section 1/2 boundary (x1 = {x1_equiv:.0f} m)")
        _axs[0].axhline(x2_equiv, color="purple", linestyle=":", label=f"Section 2/3 boundary (x2 = {x2_equiv:.0f} m)")
        _axs[0].axhline(D, color="gray", linestyle="--", label=f"Destination (D = {D:.0f} m)")
        _axs[0].set_ylabel("Position [m]")
        _axs[0].grid(True, linestyle=":", alpha=0.6)
        _axs[0].legend(loc="best")
        _axs[0].set_title(
            f"Optimal Train Trajectory via CVXPY DNLP (Cost: {prob_dnlp.value:.2f}, Time: {time_dnlp:.3f}s)",
            fontsize=13,
            fontweight="bold",
        )

        # 2. Velocity Profile vs Spatial Speed Limit
        _s1 = 0.5 * (1.0 + np.tanh((x_dnlp.value - x1_equiv) / sharp_scale))
        _s2 = 0.5 * (1.0 + np.tanh((x_dnlp.value - x2_equiv) / sharp_scale))
        _v_lim = 10.0 + (25.0 - 10.0) * _s1 + (15.0 - 25.0) * _s2

        _axs[1].plot(tempo, v_dnlp.value, label="Speed v(t)", color="#ff7f0e", linewidth=2.2)
        _axs[1].plot(tempo, _v_lim, color="#d62728", linestyle="--", label="Spatial Speed Limit v_max(x(t)) [10 → 25 → 15 m/s]")
        _axs[1].set_ylabel("Speed [m/s]")
        _axs[1].grid(True, linestyle=":", alpha=0.6)
        _axs[1].legend(loc="best")

        # 3. Control Acceleration / Deceleration Profile
        _axs[2].plot(tempo_u, u_dnlp.value, label="Control u(t)", color="#2ca02c", linewidth=2.0)
        _axs[2].axhline(u_max, color="#d62728", linestyle=":", label=f"Max Traction ({u_max} m/s²)")
        _axs[2].axhline(u_min, color="#9467bd", linestyle=":", label=f"Max Braking ({u_min} m/s²)")
        _axs[2].fill_between(tempo_u, 0, u_dnlp.value, where=(u_dnlp.value > 0.01), color="#2ca02c", alpha=0.25, label="Traction Phase")
        _axs[2].fill_between(tempo_u, 0, u_dnlp.value, where=(u_dnlp.value < -0.01), color="#d62728", alpha=0.2, label="Braking Phase")
        _axs[2].set_ylabel("Acceleration [m/s²]")
        _axs[2].grid(True, linestyle=":", alpha=0.6)
        _axs[2].legend(loc="best")

        # 4. Elevation Profile and Grade Force
        _elev = H_hill * (np.sin(np.pi * np.clip(x_dnlp.value, 0, D) / D) ** 2)
        _grade = g * (np.pi * H_hill / D) * np.sin(2.0 * np.pi * np.clip(x_dnlp.value, 0, D) / D)
        _ax_elev = _axs[3]
        _ax_grade = _ax_elev.twinx()

        _p1 = _ax_elev.plot(tempo, _elev, color="#8c564b", linewidth=2.0, label="Track Elevation h(x(t))")
        _p2 = _ax_grade.plot(tempo, _grade, color="#e377c2", linestyle="-.", label="Grade Accel g*sin(θ)")

        _ax_elev.set_xlabel("Time [s]")
        _ax_elev.set_ylabel("Elevation [m]", color="#8c564b")
        _ax_grade.set_ylabel("Grade Accel [m/s²]", color="#e377c2")
        _ax_elev.grid(True, linestyle=":", alpha=0.6)

        _plots = _p1 + _p2
        _labels = [p.get_label() for p in _plots]
        _ax_elev.legend(_plots, _labels, loc="best")

        plt.tight_layout()

    _fig_dnlp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparative Analysis: DCCP (Time Domain) vs. DNLP (Space Domain with Slope)

    Below is the **direct comparison** between:
    * **Solid lines (DNLP)**: Track slope $a_{\text{grade}}(x(t))$ and sharp spatial piecewise limits $v_{\max}(x(t))$.
    * **Dashed lines (DCCP)**: Reference flat track formulation with temporal limits $v_{\max}(t)$.
    """)
    return


@app.cell
def _(
    D,
    H_hill,
    g,
    np,
    plt,
    prob_dccp,
    prob_dnlp,
    sharp_scale,
    tempo,
    tempo_u,
    time_dccp,
    time_dnlp,
    u_dccp,
    u_dnlp,
    u_max,
    u_min,
    v_dccp,
    v_dnlp,
    v_lim_time,
    x1_equiv,
    x2_equiv,
    x_dccp,
    x_dnlp,
):
    if (
        prob_dccp.status not in ["optimal", "optimal_inaccurate", "converged"]
        or x_dccp.value is None
        or prob_dnlp.status not in ["optimal", "optimal_inaccurate", "converged"]
        or x_dnlp.value is None
    ):
        _fig_comp, _ax_comp = plt.subplots(figsize=(8, 2))
        _ax_comp.text(0.5, 0.5, "One of the optimization problems did not converge.", ha="center", va="center")
        _ax_comp.axis("off")
    else:
        _fig_comp, _axs_comp = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

        # 1. Position Profiles
        _axs_comp[0].plot(tempo, x_dnlp.value, label="DNLP Position (Spatial limits + Slope)", color="#1f77b4", linewidth=2.2)
        _axs_comp[0].plot(tempo, x_dccp.value, label="DCCP Position (Time limits, Flat)", color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.8)
        _axs_comp[0].axhline(x1_equiv, color="orange", linestyle=":", label=f"Boundary x1 = {x1_equiv:.0f} m")
        _axs_comp[0].axhline(x2_equiv, color="purple", linestyle=":", label=f"Boundary x2 = {x2_equiv:.0f} m")
        _axs_comp[0].axhline(D, color="gray", linestyle="-.", label=f"Destination D = {D:.0f} m")
        _axs_comp[0].set_ylabel("Position [m]")
        _axs_comp[0].grid(True, linestyle=":", alpha=0.6)
        _axs_comp[0].legend(loc="best")
        _axs_comp[0].set_title(
            f"Comparison: DCCP (Cost: {prob_dccp.value:.2f}, Time: {time_dccp:.2f}s) vs. "
            f"DNLP (Cost: {prob_dnlp.value:.2f}, Time: {time_dnlp:.2f}s)",
            fontsize=13,
            fontweight="bold",
        )

        # 2. Velocity Profiles & Limits
        _s1_c = 0.5 * (1.0 + np.tanh((x_dnlp.value - x1_equiv) / sharp_scale))
        _s2_c = 0.5 * (1.0 + np.tanh((x_dnlp.value - x2_equiv) / sharp_scale))
        _v_lim_c = 10.0 + (25.0 - 10.0) * _s1_c + (15.0 - 25.0) * _s2_c

        _axs_comp[1].plot(tempo, v_dnlp.value, label="DNLP Speed v_dnlp(t)", color="#ff7f0e", linewidth=2.2)
        _axs_comp[1].plot(tempo, v_dccp.value, label="DCCP Speed v_dccp(t)", color="#2ca02c", linestyle="--", linewidth=1.8)
        _axs_comp[1].plot(tempo, _v_lim_c, label="Spatial Limit v_max(x(t)) [DNLP]", color="#d62728", linewidth=1.6)
        _axs_comp[1].step(tempo, v_lim_time, where="post", label="Temporal Limit v_max(t) [DCCP]", color="black", linestyle=":", alpha=0.7)
        _axs_comp[1].set_ylabel("Speed [m/s]")
        _axs_comp[1].grid(True, linestyle=":", alpha=0.6)
        _axs_comp[1].legend(loc="best")

        # 3. Control Profiles
        _axs_comp[2].plot(tempo_u, u_dnlp.value, label="DNLP Control u_dnlp(t)", color="#ff7f0e", linewidth=2.0)
        _axs_comp[2].plot(tempo_u, u_dccp.value, label="DCCP Control u_dccp(t)", color="#2ca02c", linestyle="--", linewidth=1.8)
        _axs_comp[2].axhline(u_max, color="#d62728", linestyle=":", label=f"Max Traction ({u_max} m/s²)")
        _axs_comp[2].axhline(u_min, color="#9467bd", linestyle=":", label=f"Max Braking ({u_min} m/s²)")
        _axs_comp[2].set_ylabel("Acceleration [m/s²]")
        _axs_comp[2].grid(True, linestyle=":", alpha=0.6)
        _axs_comp[2].legend(loc="best")

        # 4. Elevation Profile and Grade Force along the DNLP Trajectory
        _elev_c = H_hill * (np.sin(np.pi * np.clip(x_dnlp.value, 0, D) / D) ** 2)
        _grade_c = g * (np.pi * H_hill / D) * np.sin(2.0 * np.pi * np.clip(x_dnlp.value, 0, D) / D)
        _ax_elev_c = _axs_comp[3]
        _ax_grade_c = _ax_elev_c.twinx()

        _p1_c = _ax_elev_c.plot(tempo, _elev_c, color="#8c564b", linewidth=2.0, label="Track Elevation h(x(t))")
        _p2_c = _ax_grade_c.plot(tempo, _grade_c, color="#e377c2", linestyle="-.", label="Grade Accel g*sin(θ)")

        _ax_elev_c.set_xlabel("Time [s]")
        _ax_elev_c.set_ylabel("Elevation [m]", color="#8c564b")
        _ax_grade_c.set_ylabel("Grade Accel [m/s²]", color="#e377c2")
        _ax_elev_c.grid(True, linestyle=":", alpha=0.6)

        _plots_c = _p1_c + _p2_c
        _labels_c = [p.get_label() for p in _plots_c]
        _ax_elev_c.legend(_plots_c, _labels_c, loc="best")

        plt.tight_layout()

    _fig_comp
    return


if __name__ == "__main__":
    app.run()
