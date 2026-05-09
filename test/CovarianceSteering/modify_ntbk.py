import json
import sys

try:
    with open('main.ipynb', 'r') as f:
        nb = json.load(f)

    # Define new source code for the Mean Control Problem (replacing original Cell 3)
    # Note: Am, Bm, etc are defined in previous cells (Cell 2), so they are available.
    source_mean = [
        "# --------------------------\n",
        "# 1. Mean Control Problem\n",
        "# --------------------------\n",
        "M_mean = mf.Model(\"mean_control\")\n",
        "\n",
        "# Variables\n",
        "v = [M_mean.variable(f\"v{k}\", [m], mf.Domain.unbounded()) for k in range(N)]\n",
        "V = [M_mean.variable(f\"V{_k}\", mf.Domain.inPSDCone(m)) for _k in range(N)]\n",
        "mu = [M_mean.variable(f\"mu{k}\", [n], mf.Domain.unbounded()) for k in range(N + 1)]\n",
        "Mmu = [M_mean.variable(f\"Mmu{_k}\", mf.Domain.inPSDCone(n)) for _k in range(N+1)]\n",
        "\n",
        "J_mean = mf.Expr.constTerm(0.0)\n",
        "\n",
        "for k in range(N):\n",
        "    # Mean propagation\n",
        "    M_mean.constraint(\n",
        "        mf.Expr.sub(mu[k+1], mf.Expr.add(mf.Expr.mul(Am[k], mu[k]), mf.Expr.mul(Bm[k], v[k]))),\n",
        "        mf.Domain.equalsTo(0.0)\n",
        "    )\n",
        "    \n",
        "    # Relaxations\n",
        "    PSD_block_v = mf.Expr.stack([[V[k], v[k]], [mf.Expr.transpose(v[k]), mf.Expr.constTerm(1.0)]])\n",
        "    M_mean.constraint(PSD_block_v, mf.Domain.inPSDCone(m+1))\n",
        "    \n",
        "    PSD_block_mu = mf.Expr.stack([[Mmu[k], mu[k]], [mf.Expr.transpose(mu[k]), mf.Expr.constTerm(1.0)]])\n",
        "    M_mean.constraint(PSD_block_mu, mf.Domain.inPSDCone(n+1))\n",
        "    \n",
        "    # Cost\n",
        "    J_mean = mf.Expr.add(J_mean, mf.Expr.dot(R_param, V[k]))\n",
        "    J_mean = mf.Expr.add(J_mean, mf.Expr.dot(Q_param, Mmu[k]))\n",
        "\n",
        "# Boundary conditions\n",
        "M_mean.constraint(mf.Expr.sub(mu[0], mf.Matrix.dense(mu_i[:, None])), mf.Domain.equalsTo(0.0))\n",
        "M_mean.constraint(mf.Expr.sub(mu[N], mf.Matrix.dense(mu_f[:, None])), mf.Domain.equalsTo(0.0))\n",
        "\n",
        "M_mean.objective(mf.ObjectiveSense.Minimize, J_mean)\n",
        "M_mean.setLogHandler(sys.stdout)\n",
        "M_mean.solve()\n",
        "print(\"Mean Problem status:\", M_mean.getProblemStatus())\n"
    ]

    # Define new source code for the Covariance Steering Problem (replacing original Cell 4)
    source_cov = [
        "# --------------------------\n",
        "# 2. Covariance Steering Problem\n",
        "# --------------------------\n",
        "M_cov = mf.Model(\"covariance_steering\")\n",
        "\n",
        "# Variables\n",
        "Y = [M_cov.variable(f\"Y{k}\", mf.Domain.inPSDCone(m)) for k in range(N)]\n",
        "U = [M_cov.variable(f\"U{k}\", [m, n], mf.Domain.unbounded()) for k in range(N)]\n",
        "S = [M_cov.variable(f\"S{k}\", mf.Domain.inPSDCone(n)) for k in range(N + 1)]\n",
        "\n",
        "J_cov = mf.Expr.constTerm(0.0)\n",
        "\n",
        "for k in range(N):\n",
        "    # Covariance propagation\n",
        "    constr = mf.Expr.neg(S[k + 1])\n",
        "    constr = mf.Expr.add(constr, mf.Expr.mul(mf.Expr.mul(Am[k], S[k]), mf.Matrix.transpose(Am[k])))\n",
        "    constr = mf.Expr.add(constr, mf.Expr.mul(mf.Expr.mul(Bm[k], U[k]), mf.Matrix.transpose(Am[k])))\n",
        "    constr = mf.Expr.add(constr, mf.Expr.mul(mf.Expr.mul(Am[k], mf.Matrix.transpose(U[k])), mf.Matrix.transpose(Bm[k])))\n",
        "    constr = mf.Expr.add(constr, mf.Expr.mul(mf.Expr.mul(Bm[k], Y[k]), mf.Matrix.transpose(Bm[k])))\n",
        "    constr = mf.Expr.add(constr, DmDmT[k])\n",
        "    M_cov.constraint(constr, mf.Domain.equalsTo(0.0))\n",
        "    \n",
        "    # LMI constraint\n",
        "    X = mf.Expr.stack([[S[k], mf.Expr.transpose(U[k])], [U[k], Y[k]]])\n",
        "    M_cov.constraint(X, mf.Domain.inPSDCone(n + m))\n",
        "    \n",
        "    # Cost\n",
        "    J_cov = mf.Expr.add(J_cov, mf.Expr.dot(Q_param, S[k]))\n",
        "    J_cov = mf.Expr.add(J_cov, mf.Expr.dot(R_param, Y[k]))\n",
        "\n",
        "# Boundary conditions\n",
        "M_cov.constraint(mf.Expr.sub(S[0], mf.Matrix.dense(Si)), mf.Domain.equalsTo(0.0))\n",
        "M_cov.constraint(mf.Expr.sub(S[N], mf.Matrix.dense(Sf)), mf.Domain.equalsTo(0.0))\n",
        "\n",
        "M_cov.objective(mf.ObjectiveSense.Minimize, J_cov)\n",
        "M_cov.setLogHandler(sys.stdout)\n",
        "M_cov.solve()\n",
        "print(\"Covariance Problem status:\", M_cov.getProblemStatus())\n"
    ]
    
    # Modify the cells
    # Index 3: "Create Fusion model" -> Mean Control
    nb['cells'][3]['source'] = source_mean
    
    # Index 4: Loop -> Covariance Control
    nb['cells'][4]['source'] = source_cov
    
    # Index 5: Boundary & Solve -> clear (as solved in previous cells)
    nb['cells'][5]['source'] = ["# Optimization solved in the previous two cells (split into Mean and Covariance problems)\n"]

    with open('main.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Successfully modified main.ipynb")

except Exception as e:
    print(f"Error: {e}")
