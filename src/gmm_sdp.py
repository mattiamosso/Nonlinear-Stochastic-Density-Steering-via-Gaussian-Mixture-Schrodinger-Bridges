"""
GMM covariance steering via Semidefinite Programming (SDP).

Module overview
---------------
``CovarianceSteering``
    Solves the per-component optimal covariance steering problem using MOSEK.
``GMMflow_SDP``
    Builds the full Gaussian-mixture SDE (drift + diffusion) by combining
    optimal-transport coupling with per-component SDP solutions.
"""

import bisect

import numpy as np
import ot
import torch
from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, ProblemStatus
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal


class CovarianceSteering:
    """Per-component optimal covariance steering (OCS) via SDP.

    Solves, for each initial–final Guassian pair (i,j):

        min  Σ_k [ tr(Q S_k) + tr(R Y_k) ]
        s.t. S_{k+1} = A_k S_k A_kᵀ + B_k U_k A_kᵀ + A_k U_kᵀ B_kᵀ
                        + B_k Y_k B_kᵀ + D_k
             [S_k  U_kᵀ]
             [U_k  Y_k ] ∈ PSD
             S_0 = Σ₀
             Σ_f − S_N ∈ PSD   (terminal inequality)

    Parameters
    ----------
    mu0, mu1 : ndarray (N0, n) and (N1, n)
        Initial / final mean states.
    sigma0, sigma1 : ndarray (N0, n, n) and (N1, n, n)
        Initial / final covariance matrices.
    A_d, B_d, DD : list of list of ndarray
        Discrete-time system matrices for every (i, j) pair.
    x_ref, u_ref : list of ndarray
        Nominal reference trajectories.
    t_grid, dt_grid : list of ndarray
        Time grids.
    n_intervals : int
        Number of SDP intervals (= N_nodes − 1).
    """

    def __init__(self, mu0, mu1, sigma0, sigma1,
                 A_d, B_d, DD, x_ref, u_ref,
                 t_grid, dt_grid, n_intervals=100):

        self.A_d_all = A_d
        self.B_d_all = B_d
        self.DD_all = DD
        self.t_grid_all = t_grid
        self.dt_grid_all = dt_grid

        self.x_ref = x_ref
        self.u_ref = u_ref

        self.mu0 = np.ascontiguousarray(mu0, dtype=np.float64)
        self.mu1 = np.ascontiguousarray(mu1, dtype=np.float64)
        self.N0 = self.mu0.shape[0]
        self.N1 = self.mu1.shape[0]

        self.sigma0 = np.ascontiguousarray(sigma0, dtype=np.float64)
        self.sigma1 = np.ascontiguousarray(sigma1, dtype=np.float64)

        self.N = n_intervals   # number of intervals
        self.n = 4             # state dimension
        self.m = 2             # control dimension

    def solve(self):
        """Solve the SDP for every (i, j) pair and store results.

        Populates
        ---------
        self.transport_cost : ndarray (N0, N1)
        self.sol : list of list of dict
            Each dict has keys: 'cov', 'mu', 'U', 'v', 'K'.
        """
        self.transport_cost = np.empty((self.N0, self.N1))
        self.sol = [[None] * self.N1 for _ in range(self.N0)]

        # Cost weights — low Q to prioritise terminal feasibility
        Q_mat = Matrix.dense(1e-08 * np.diag([0.1, 0.1, 0.1, 0.1]))
        R_mat = Matrix.dense(np.eye(self.m))

        for i in range(self.N0):
            for j in range(self.N1):
                self._solve_pair(i, j, Q_mat, R_mat)

    def _solve_pair(self, i, j, Q_mat, R_mat):
        """Solve the SDP for a single (i, j) component pair."""
        idx = i * self.N1 + j
        Si = self.sigma0[i]
        Sf = self.sigma1[j]

        A_k = self.A_d_all[idx]
        B_k = self.B_d_all[idx]
        D_k = self.DD_all[idx]
        dt_k = self.dt_grid_all[idx]

        with Model("covariance_steering") as M:
            Si_mat = Matrix.dense(Si)
            Sf_mat = Matrix.dense(Sf)

            # Decision variables
            Y = [M.variable(f"Y{k}", Domain.inPSDCone(self.m)) for k in range(self.N)]
            U = [M.variable(f"U{k}", [self.m, self.n], Domain.unbounded()) for k in range(self.N)]
            S = [M.variable(f"S{k}", Domain.inPSDCone(self.n)) for k in range(self.N + 1)]

            cost = Expr.constTerm(0.0)

            for k in range(self.N):
                Ak = Matrix.dense(A_k[k])
                Bk = Matrix.dense(B_k[k])
                Dk = Matrix.dense(D_k[k])

                # S_{k+1} = A S Aᵀ + B U Aᵀ + A Uᵀ Bᵀ + B Y Bᵀ + D
                rhs = Expr.neg(S[k + 1])
                rhs = Expr.add(rhs, Expr.mul(Expr.mul(Ak, S[k]), Matrix.transpose(Ak)))
                rhs = Expr.add(rhs, Expr.mul(Expr.mul(Bk, U[k]), Matrix.transpose(Ak)))
                rhs = Expr.add(rhs, Expr.mul(Expr.mul(Ak, Expr.transpose(U[k])), Matrix.transpose(Bk)))
                rhs = Expr.add(rhs, Expr.mul(Expr.mul(Bk, Y[k]), Matrix.transpose(Bk)))
                rhs = Expr.add(rhs, Dk)
                M.constraint(rhs, Domain.equalsTo(0.0))

                # Schur complement LMI: [S  Uᵀ; U  Y] ≥ 0
                lmi = Expr.stack([[S[k], Expr.transpose(U[k])], [U[k], Y[k]]])
                M.constraint(lmi, Domain.inPSDCone(self.n + self.m))

                # Running cost
                cost = Expr.add(cost, Expr.dot(Q_mat, S[k]))
                cost = Expr.add(cost, Expr.dot(R_mat, Y[k]))

            # Boundary conditions
            M.constraint(Expr.sub(S[0], Si_mat), Domain.equalsTo(0.0))
            
            # Terminal: Σ_f − S_N ∈ PSD  (inequality — allows tighter covariance)
            M.constraint(Expr.sub(Sf_mat, S[self.N]), Domain.inPSDCone(self.n))

            M.objective(ObjectiveSense.Minimize, cost)

            try:
                M.solve()
                feasible = M.getProblemStatus() in [
                    ProblemStatus.PrimalFeasible,
                    ProblemStatus.PrimalAndDualFeasible,
                ]
                if not feasible:
                    raise RuntimeError("SDP not feasible")

                self.sol[i][j] = self._extract_solution(S, U, Y, idx)

                # Total cost = SDP cost + nominal effort
                v_sq = np.sum(self.u_ref[idx]**2, axis=1)
                nominal_effort = np.sum(v_sq * dt_k)
                self.transport_cost[i, j] = M.primalObjValue() + nominal_effort

            except Exception as e:
                print(f"Warning: SDP failed for pair ({i},{j}): {e}")
                self.sol[i][j] = self._fallback_solution(idx)
                self.transport_cost[i, j] = 1e6

    def _extract_solution(self, S, U, Y, idx):
        """Extract decision-variable levels into a solution dict."""
        n, m, N = self.n, self.m, self.N
        eye_n = torch.eye(n, dtype=torch.float64)
        return {
            "cov": [
                torch.tensor(S[k].level().reshape(n, n), dtype=torch.float64)
                for k in range(N + 1)
            ],
            "mu": [
                torch.tensor(mu_t, dtype=torch.float64)
                for mu_t in self.x_ref[idx].T
            ],
            "U": [
                torch.tensor(U[k].level().reshape(m, n), dtype=torch.float64)
                for k in range(N)
            ],
            "Y": [
                torch.tensor(Y[k].level().reshape(m, m), dtype=torch.float64)
                for k in range(N)
            ],
            "v": torch.tensor(self.u_ref[idx], dtype=torch.float64),
            "K": [
                torch.tensor(U[k].level().reshape(m, n), dtype=torch.float64)
                @ torch.linalg.inv(
                    torch.tensor(S[k].level().reshape(n, n), dtype=torch.float64)
                )
                for k in range(N)
            ],
        }

    def _fallback_solution(self, idx):
        """Return a zero-feedback fallback when the SDP fails."""
        n, m, N = self.n, self.m, self.N
        return {
            "cov": [torch.eye(n, dtype=torch.float64) * 1e-3 for _ in range(N + 1)],
            "mu": [torch.tensor(mu_t, dtype=torch.float64) for mu_t in self.x_ref[idx].T],
            "U": [torch.zeros(m, n, dtype=torch.float64) for _ in range(N)],
            "v": torch.tensor(self.u_ref[idx], dtype=torch.float64),
            "K": [torch.zeros(m, n, dtype=torch.float64) for _ in range(N)],
        }


class GMMflow_SDP(nn.Module):
    """Full GMM Schrödinger-bridge SDE with LTV dynamics.

    Combines per-component covariance-steering solutions with an
    optimal-transport coupling to define a mixture drift and diffusion
    compatible with ``torchsde``.

    Parameters
    ----------
    mu0, mu1 : Tensor (N0, n) / (N1, n)
        Component means.
    sigma0, sigma1 : Tensor (N0, n, n) / (N1, n, n)
        Component covariances.
    w0, w1 : Tensor (N0,) / (N1,)
        Mixture weights, or None for uniform.
    A_d, B_d, DD : list of list of ndarray
        Discrete-time matrices (passed to ``CovarianceSteering``).
    A_lin, B_lin, D_lin : list of list of ndarray
        Continuous-time matrices for the SDE.
    x_ref, u_ref : list of ndarray
        Nominal reference trajectories.
    t_grid, dt_grid : list of ndarray
        Time grids per (i, j) pair.
    n_nodes : int
        Number of time grid nodes.
    diffusion_scale : float
        Scaling factor ε for the diffusion (0 = deterministic).
    """

    noise_type = "general"
    sde_type = "ito"

    # Physical constants (nondimensional)
    _MU_SUN_ND = 2.959122082855911e-04   # AU³ day⁻²
    _U_CLAMP = 0.04                       # saturation for divergence prevention

    def __init__(self, mu0, mu1, sigma0, sigma1, w0, w1,
                 A_d, B_d, DD, A_lin, B_lin, D_lin,
                 x_ref, u_ref, t_grid, dt_grid,
                 n_nodes=101, diffusion_scale=1.0):
        super().__init__()

        self.n = 4   # state dimension
        self.m = 2   # control dimension

        # --- Store continuous-time matrices as tensors ---
        self.A_lin = self._stack_matrices(A_lin)
        self.B_lin = self._stack_matrices(B_lin)
        self.D_lin = self._stack_matrices(D_lin) * diffusion_scale

        # --- Means and weights ---
        self.mu0 = mu0
        self.mu1 = mu1
        self.N0 = mu0.shape[0]
        self.N1 = mu1.shape[0]

        self.w0 = w0 if w0 is not None else torch.ones(self.N0) / self.N0
        self.w1 = w1 if w1 is not None else torch.ones(self.N1) / self.N1

        # --- Time grids ---
        self.t_grid_all = t_grid
        self.dt_grid_all = dt_grid

        # --- Solve per-component SDP ---
        self.cs = CovarianceSteering(
            mu0, mu1, sigma0, sigma1,
            A_d, B_d, DD, x_ref, u_ref,
            t_grid, dt_grid, n_intervals=n_nodes - 1,
        )
        self.cs.solve()

        # --- Optimal transport coupling ---
        self.cost_matrix = torch.tensor(self.cs.transport_cost)
        self.coupling = ot.emd(self.w0, self.w1, self.cost_matrix)

    @staticmethod
    def _stack_matrices(mat_list):
        """Convert nested list of arrays → Tensor (n_pairs, n_steps, ...)."""
        return torch.stack([
            torch.stack([torch.tensor(m, dtype=torch.float64) for m in pair])
            for pair in mat_list
        ])

    def _time_index(self, idx, t):
        """Map continuous time *t* to the bracketing grid index and fraction."""
        T_ij = self.t_grid_all[idx]
        k = bisect.bisect_right(T_ij, t) - 1
        k = max(0, min(k, len(T_ij) - 2))
        alpha = float(t - T_ij[k]) / float(T_ij[k + 1] - T_ij[k])
        alpha = max(0.0, min(1.0, alpha))
        return k, alpha

    def _interpolate_ref(self, i, j, t):
        """Linearly interpolate reference mean and covariance at time *t*.

        Returns
        -------
        mu_t : Tensor (n,)
        sigma_t : Tensor (n, n)
        """
        idx = i * self.N1 + j
        k, alpha = self._time_index(idx, t)

        mu_k = self.cs.sol[i][j]["mu"][k]
        mu_next = self.cs.sol[i][j]["mu"][k + 1]

        sigma_k = self.cs.sol[i][j]["cov"][k]
        sigma_next = self.cs.sol[i][j]["cov"][k + 1]

        mu_t = (1.0 - alpha) * mu_k + alpha * mu_next
        sigma_t = (1.0 - alpha) * sigma_k + alpha * sigma_next

        return mu_t, sigma_t

    def _component_log_weights(self, x, i, j, mu_t, sigma_t):
        """Log-probability contribution of component (i, j) at states *x*.

        Returns log Λ_{ij} + log N(x; μ_t, Σ_t), or -1e20 if inactive.
        """
        if self.coupling[i, j] <= 0:
            return None

        try:
            dist = MultivariateNormal(
                loc=mu_t,
                covariance_matrix=sigma_t,
            )
            return torch.log(self.coupling[i, j]) + dist.log_prob(x)
        except ValueError:
            return None

    @staticmethod
    def _safe_softmax(log_w, dim=1):
        """Numerically stable softmax over *dim*."""
        max_log = torch.max(log_w, dim=dim, keepdim=True)[0]
        exp_w = torch.exp(log_w - max_log)
        return exp_w / (torch.sum(exp_w, dim=dim, keepdim=True))

    #  Mixture control
    def compute_control(self, x, t):
        """Evaluate the mixture feedback control u(x, t).

        Parameters
        ----------
        x : Tensor (B, n)
        t : float

        Returns
        -------
        u : Tensor (B, m)
        """
        B = x.shape[0]
        n_pairs = self.N0 * self.N1
        log_w = torch.full((B, n_pairs), -1e20, dtype=torch.float64)
        u_all = torch.zeros(B, n_pairs, self.m, dtype=torch.float64)

        for i in range(self.N0):
            for j in range(self.N1):
                idx = i * self.N1 + j
                k, _ = self._time_index(idx, t)

                v_k = self.cs.sol[i][j]["v"][k]          # feedforward
                K_k = self.cs.sol[i][j]["K"][k]          # feedback gain
                mu_t, sigma_t = self._interpolate_ref(i, j, t)

                u_ij = (K_k @ (x - mu_t).T).T + v_k     # u = K (x − μ) + v
                u_all[:, idx, :] = u_ij

                lw = self._component_log_weights(x, i, j, mu_t, sigma_t)
                if lw is not None:
                    log_w[:, idx] = lw

        weights = self._safe_softmax(log_w)
        u_final = torch.sum(u_all * weights.unsqueeze(2), dim=1)

        return u_final

    def f(self, t, y):
        """SDE drift: nonlinear gravity + mixture feedback control.

        ẏ = [v; −μ r/||r||³] + [0; u(y, t)]
        """
        r = y[:, :2]
        v = y[:, 2:]
        r_norm = torch.clamp(torch.linalg.vector_norm(r, dim=1, keepdim=True), min=1e-12)

        gravity = -self._MU_SUN_ND * r / r_norm**3
        drift_free = torch.cat([v, gravity], dim=1)

        u = self.compute_control(y, t)
        drift_ctrl = torch.cat([torch.zeros_like(u), u], dim=1)

        return drift_free + drift_ctrl

    def g(self, t, y):        
        """SDE diffusion: time-varying mixture covariance."""
        return self.D_lin[0][0].unsqueeze(0).repeat(y.shape[0], 1, 1)

    #  Cost functions
    def compute_ot_cost(self):
        """Optimal-transport cost Σ_{ij} Λ_{ij} C_{ij}."""
        return torch.sum(self.coupling * self.cost_matrix).item()

    def sample_gmm(self, t, n_samples):
        """Sample from the time-marginal GMM ρ(·, t)."""
        n_pairs = self.N0 * self.N1
        means = torch.zeros(n_pairs, self.n, dtype=torch.float64)
        covs = torch.zeros(n_pairs, self.n, self.n, dtype=torch.float64)

        count = 0
        for i in range(self.N0):
            for j in range(self.N1):
                mu_t, sigma_t = self._interpolate_ref(i, j, t)
                means[count] = mu_t
                covs[count] = sigma_t
                count += 1

        mix = Categorical(self.coupling.reshape(-1).float())
        components = MultivariateNormal(
            loc=means,
            covariance_matrix=covs + 1e-12 * torch.eye(self.n, dtype=torch.float64),
        )
        return MixtureSameFamily(mix, components).sample([n_samples])

    def estimate_control_cost(self, n_samples=5000):
        """Monte-Carlo estimate of J_ctrl = ∫₀ᵀ E[||u(x,t)||²] dt."""
        # Use the first pair's time grid (all pairs share the same grid)
        T = self.t_grid_all[0]
        dt = self.dt_grid_all[0]
        J = 0.0
        for k, t in enumerate(T[:-1]):
            samples = self.sample_gmm(t, n_samples)
            u_k = self.compute_control(samples, t)
            J += float(dt[k]) * torch.mean(
                torch.linalg.vector_norm(u_k, dim=1) ** 2
            ).item()
        return J

    def compute_density(self, x, t):
        """Evaluate ρ(x, t) at given states (for debugging / visualization).

        Parameters
        ----------
        x : Tensor (B, n)
        t : float

        Returns
        -------
        rho : Tensor (B,)
        """
        rho = torch.zeros(x.shape[0], dtype=torch.float64)
        for i in range(self.N0):
            for j in range(self.N1):
                mu_t, sigma_t = self._interpolate_ref(i, j, t)
                try:
                    dist = MultivariateNormal(
                        loc=mu_t,
                        covariance_matrix=sigma_t + 1e-12 * torch.eye(self.n),
                    )
                    rho += self.coupling[i, j] * torch.exp(dist.log_prob(x))
                except ValueError:
                    continue
        return rho