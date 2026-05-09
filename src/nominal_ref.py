"""
Nominal reference trajectory solver for low-thrust orbit transfers.

Solves a minimum-energy open-loop trajectory optimization problem (NLP) for
point-to-point transfers under two-body gravity, then linearizes and
discretizes the dynamics about the nominal trajectory for use in the SDP-based
covariance steering formulation.

All quantities are nondimensionalized with:
    L = 1 AU,  T = 1 day,  V = AU / day.
"""

import numpy as np
import scipy as sp


class OpenLoop:
    """Minimum-energy open-loop transfer between two Keplerian states.

    Solves
        min  ∫ ||u||² dt
        s.t. ẋ = f(x) + B u,   x(0) = x₀,  x(tf) = xf,  ||u|| ≤ u_max

    via direct transcription (piece-wise constant controls) and SciPy SLSQP.

    After `solve()`, the following attributes are available:

    Attributes
    ----------
    x_ref : ndarray (n, N)
        Nominal reference state trajectory [r_x, r_y, v_x, v_y].
    u_ref : ndarray (N-1, m)
        Optimal open-loop control history.
    A_lin : list[ndarray (n, n)]
        Continuous-time Jacobians evaluated along the reference.
    B_lin : list[ndarray (n, m)]
        Continuous-time input matrices (constant, included for generality).
    A_d : list[ndarray (n, n)]
        Discrete-time state-transition matrices (via matrix exponential).
    B_d : list[ndarray (n, m)]
        Discrete-time input matrices (via matrix exponential).

    D_d : list[ndarray (n, n)]
        Discrete-time diffusion matrices (Euler–Maruyama, D_d = σ √Δt).
    DD : list[ndarray (n, n)]
        Discrete noise covariance D_d D_dᵀ.
    t_grid : ndarray (N,)
        Nondimensional time nodes.
    dt_grid : ndarray (N-1,)
        Nondimensional time steps.
    """

    #  Physical constants                                                  #
    MU_SUN: float = 1.32712440018e11   # km³ s⁻²
    AU: float = 149597870.7            # km
    DAY: float = 86400.0               # s

    def __init__(self, mu0, mu1, n_nodes):
        """
        Parameters
        ----------
        mu0 : array-like (4,)
            Initial nondimensional state [r_x, r_y, v_x, v_y].
        mu1 : array-like (4,)
            Final nondimensional state.
        n_nodes : int
            Number of time-grid nodes (≥ 2).
        """
        self.mu0 = np.asarray(mu0, dtype=np.float64)
        self.mu1 = np.asarray(mu1, dtype=np.float64)

        self.r0 = self.mu0[:2]
        self.v0 = self.mu0[2:]
        self.rf = self.mu1[:2]
        self.vf = self.mu1[2:]

        # State / control dimensions
        self.n = 4   # [r_x, r_y, v_x, v_y]
        self.m = 2   # [u_x, u_y]

        # Time of flight (from Benedikter et al. 2022)
        self.tf_days = 348.79
        self.tf_sec = self.tf_days * self.DAY

        # Nondimensionalization scales
        self._L = self.AU
        self._T = self.DAY
        self._V = self._L / self._T
        self._A = self._L / self._T**2

        self.mu_nd = self.MU_SUN * self._T**2 / self._L**3

        # Time grid (nondimensional)
        self.N = n_nodes
        t_dim = np.linspace(0.0, self.tf_sec, self.N)
        self.t_grid = t_dim / self._T
        self.dt_grid = np.diff(self.t_grid)

        # Thrust constraint
        self.u_max_km = 1e-3 / 1000.0   # 1 mm s⁻² → km s⁻²
        self.u_max_nd = self.u_max_km / self._A

        # Process noise intensity (velocity diffusion)
        self.gv_nd = (1e-4 / 1000.0) * np.sqrt(self._T) / self._V
        # Solved quantities (populated by solve())
        self.x_ref = None
        self.u_ref = None

    def __repr__(self):
        return (f"OpenLoop(mu0={self.mu0}, mu1={self.mu1}, "
                f"N={self.N}, tf={self.tf_days:.2f} days)")

    #  NLP components                                                      #
    def _initial_guess(self):
        """Two-impulse Hohmann-like initial guess for the optimizer."""
        r1 = np.linalg.norm(self.r0)
        r2 = np.linalg.norm(self.rf)

        v1_circ = np.sqrt(self.mu_nd / r1)
        v2_circ = np.sqrt(self.mu_nd / r2)
        v_peri = np.sqrt(2 * self.mu_nd / r1 - 2 * self.mu_nd / (r1 + r2))
        v_apo = np.sqrt(2 * self.mu_nd / r2 - 2 * self.mu_nd / (r1 + r2))

        dv1 = abs(v_peri - v1_circ)
        dv2 = abs(v2_circ - v_apo)

        u0 = np.zeros(self.m * (self.N - 1))
        burn_frac = 0.10
        t_dim = self.t_grid * self._T

        for k in range(self.N - 1):
            t_mid = 0.5 * (t_dim[k] + t_dim[k + 1])
            idx = self.m * k

            if t_mid < self.tf_sec * burn_frac:
                v_dir = self.v0 / np.linalg.norm(self.v0)
                u0[idx:idx + self.m] = v_dir * dv1 / (self.tf_sec * burn_frac)
            elif t_mid > self.tf_sec * (1.0 - burn_frac):
                v_dir = self.vf / np.linalg.norm(self.vf)
                u0[idx:idx + self.m] = v_dir * dv2 / (self.tf_sec * burn_frac)

        # Clip to thrust constraint
        for k in range(self.N - 1):
            idx = k * self.m
            u_norm = np.linalg.norm(u0[idx:idx + self.m])
            if u_norm > self.u_max_km:
                u0[idx:idx + self.m] *= self.u_max_km / u_norm

        return u0 / self._A   # nondimensionalize

    def _objective(self, u_flat):
        """Minimum-energy cost ∫ ||u||² dt."""
        J = 0.0
        controls = u_flat.reshape((self.N - 1, self.m))
        for k in range(self.N - 1):
            # uk = u_flat[self.m * k: self.m * (k + 1)]
            uk = controls[k]
            J += np.dot(uk, uk) * self.dt_grid[k]
        return J

    def _dynamics_nd(self, x, u):
        """Nondimensional two-body + thrust dynamics ẋ = f(x, u)."""
        r = x[:2]
        v = x[2:]
        r_norm = max(np.linalg.norm(r), 1e-12)
        dr = v
        dv = -self.mu_nd * r / r_norm**3 + u
        return np.hstack((dr, dv))

    def _rk4_step(self, x, u, dt):
        """Single RK4 integration step."""

        k1 = self._dynamics_nd(x, u)
        k2 = self._dynamics_nd(x + 0.5 * dt * k1, u)
        k3 = self._dynamics_nd(x + 0.5 * dt * k2, u)
        k4 = self._dynamics_nd(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _propagate(self, controls):
        """Forward-propagate state under piece-wise constant control."""
        state = np.zeros((self.n, self.N))
        state[:, 0] = np.hstack((self.r0, self.v0))
        for k in range(self.N - 1):
            state[:, k + 1] = self._rk4_step(state[:, k], controls[k], self.dt_grid[k])
        return state

    def _boundary_constraint(self, u_flat):
        """Terminal-state equality constraint xf = xf_target."""
        controls = u_flat.reshape((self.N - 1, self.m))
        state = self._propagate(controls)
        xf = state[:, -1]
        return np.hstack((xf[:2] - self.rf, xf[2:] - self.vf))

    def _thrust_constraint(self, u_flat):
        """Inequality constraint: u_max - ||u_k|| ≥ 0 for all k."""
        controls = u_flat.reshape((self.N - 1, self.m))
        return np.array([
            self.u_max_nd - np.linalg.norm(controls[k])
            for k in range(self.N - 1)
        ])

    def _linearize(self):
        """Compute continuous-time Jacobians A(t), B along the reference."

        A(t) = ∂f/∂x |_{x*(t)},   B = [0; I]  (constant).
        """
        I2 = np.eye(2)
        Z2 = np.zeros((2, 2))

        self.A_lin = []
        self.B_lin = [np.vstack((Z2, I2)) for _ in range(self.N - 1)]

        for k in range(self.N - 1):
            r = self.x_ref[:2, k]
            r_norm = max(np.linalg.norm(r), 1e-12)

            # Gravity gradient  ∂(−μr/||r||³)/∂r
            a11 = self.mu_nd * (3.0 * r[0]**2 - r_norm**2) / r_norm**5
            a22 = self.mu_nd * (3.0 * r[1]**2 - r_norm**2) / r_norm**5
            a12 = self.mu_nd * (3.0 * r[0] * r[1]) / r_norm**5

            Avr = np.array([[a11, a12], [a12, a22]])
            self.A_lin.append(np.vstack((
                np.hstack((Z2, I2)),
                np.hstack((Avr, Z2))
            )))

    def _discretize(self):
        self.A_d = []
        self.B_d = []

        for k in range(self.N - 1):
            M = np.zeros((self.n + self.m, self.n + self.m))
            M[:self.n, :self.n] = self.A_lin[k]
            M[:self.n, self.n:] = self.B_lin[k]
            eM = np.eye(self.n + self.m) + M * self.dt_grid[k]
            
            self.A_d.append(eM[:self.n, :self.n])
            self.B_d.append(eM[:self.n, self.n:])

        # Diffusion matrix σ (velocity-only noise):  D_d = σ √Δt
        Z24 = np.zeros((2, 4))
        Z22 = np.zeros((2, 2))
        I2 = np.eye(2)
        self.D_d = [
            np.vstack((Z24, np.hstack((Z22, self.gv_nd * I2)))) * np.sqrt(self.dt_grid[k])
            for k in range(self.N - 1)
        ]
        # Noise covariance D_d D_dᵀ
        self.DD = [Dk @ Dk.T + 1e-7 * np.diag([0.05, 0.05, 0.1, 0.1]) for Dk in self.D_d]

    def _ensure_contiguous(self):
        """Convert all matrices to contiguous float64 arrays (MOSEK requirement)."""
        for k in range(len(self.A_d)):
            self.A_d[k] = np.ascontiguousarray(self.A_d[k], dtype=np.float64)
            self.B_d[k] = np.ascontiguousarray(self.B_d[k], dtype=np.float64)
            self.DD[k] = np.ascontiguousarray(self.DD[k], dtype=np.float64)

    def solve(self):
        """Run the full solve pipeline: NLP → linearize → discretize."""

        # NLP
        result = sp.optimize.minimize(
            self._objective,
            self._initial_guess(),
            method='SLSQP',
            constraints=[
                {"type": "eq", "fun": self._boundary_constraint},
                {"type": "ineq", "fun": self._thrust_constraint},
            ],
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        self.u_ref = result.x.reshape((self.N - 1, self.m))
        self.x_ref = self._propagate(self.u_ref)

        # Linearize & discretize
        self._linearize()
        self._discretize()
        self._ensure_contiguous()
