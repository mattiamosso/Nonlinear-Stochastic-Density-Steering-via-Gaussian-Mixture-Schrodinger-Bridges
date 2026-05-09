# Nonlinear Stochastic Density Steering via Gaussian Mixture Schrödinger Bridges and Multiple Linearizations

[![arXiv](https://img.shields.io/badge/arXiv-2604.15576-b31b1b.svg)](https://arxiv.org/abs/2604.15576)

This paper studies the optimal density steering problem for nonlinear continuous-time stochastic systems.
To accurately capture the nonlinear nature of the system in high-uncertainty areas that significantly deviate from a nominal linearization point, we introduce the concept of **Multiple Distribution-to-Distribution Linearization**.
Under the assumption that the boundary distributions are well-approximated by a Gaussian Mixture Model (GMM), the proposed approach consists of a mixture of elementary policies, each solving a Gaussian-to-Gaussian Optimal Covariance Steering (OCS) problem from the components of the initial mixture to the components of the terminal mixture.
Each OCS solution uses a local linearization around the mean trajectory connecting the initial and final GMM components.
We analyze our method both theoretically and numerically, using a case study problem of Earth-to-Mars orbit transfer calculation.

## Project Structure

```
GMM-SOC/
├── src/                             # Source package
│   ├── __init__.py                  # Public API exports
│   ├── gmm_sdp.py                   # OCS via SDP + GMM SDE (main paper method)
│   ├── nominal_ref.py               # Open-loop NLP + linearisation/discretisation
│   ├── gmm_flow.py                  # GSB with trivial dynamics, diagonal covariances
│   ├── gmm_flow_fast.py             # Sparse-lambda variant of gmm_flow
│   └── gmm_msb.py                   # Multi-marginal Schrödinger Bridge
├── notebooks/
│   └── earth_mars_transfer.ipynb    # Earth-to-Mars numerical experiment
└── README.md
```

## Installation

Requires Python ≥ 3.9 and the following dependencies:

```bash
pip install numpy scipy torch POT tqdm
```

A [MOSEK](https://www.mosek.com/) licence is also required (free for academic use):

```bash
pip install mosek
```

## Usage

See `notebooks/exp1.ipynb` for a complete walkthrough of the Earth-to-Mars orbit transfer experiment.

## Citation

```bibtex
@article{mosso2026nonlinear,
  title   = {Nonlinear Stochastic Density Steering via Gaussian Mixture
             Schrödinger Bridges and Multiple Linearizations},
  author  = {Mosso, Mattia and Rapakoulias, George and Guan, Yue
             and Tsiotras, Panagiotis},
  journal = {arXiv preprint arXiv:2604.15576},
  year    = {2026}
}
```
