# -*- coding: utf-8 -*-
"""
solve1.py  — Corollary 1 / Theorem 1 的 LMI，可同时测试 D 的三种解读
依赖: pip install cvxpy numpy
可选求解器: MOSEK 或 CVXOPT；否则回落 SCS。
"""

import numpy as np
import cvxpy as cp


# ---------- 工具 ----------
def diag_mat(vec):
    return np.diag(np.array(vec).reshape(-1))


def sym(expr):
    return expr + expr.T


def build_Dpi(D_list, phi, psi):
    """
    Dπ = diag_i( sum_k [(π/(ψ_k-φ_k))^2 * D_{ik}] )
    D_list: [D_dim1, D_dim2, ...]，每个 shape (n,)
    """
    if isinstance(D_list, list):
        D_arr = np.stack([np.array(d).reshape(-1) for d in D_list], axis=1)
    else:
        D_arr = np.array(D_list).reshape(-1, 1)
    n, q = D_arr.shape
    phi = np.array(phi).reshape(-1)
    psi = np.array(psi).reshape(-1)
    if phi.size == 1:
        phi = np.repeat(phi, q)
    if psi.size == 1:
        psi = np.repeat(psi, q)
    coeff = (np.pi / (psi - phi)) ** 2  # (q,)
    d_sum = D_arr @ coeff  # (n,)
    return np.diag(d_sum)


# ---------- 定理1 / 推论1 的 LMI ----------
def theorem1_feasible(
    A_list,
    W1_list,
    W2_list,
    D_list,
    phi,
    psi,
    xi_bar,
    beta1,
    beta2,
    sigma1,
    sigma2,
    mu,
    p_min=1e-3,
    eps=1e-9,
    verbose=False,
):
    m = len(A_list)
    n = A_list[0].shape[0]
    # 常量
    T1 = diag_mat(beta1)
    T2 = diag_mat(beta2)
    Dpi = build_Dpi(D_list, phi, psi)
    I_n = np.eye(n)
    T1_c = cp.Constant(T1)
    T2_c = cp.Constant(T2)
    Dpi_c = cp.Constant(Dpi)
    I_nC = cp.Constant(I_n)
    Z = cp.Constant(np.zeros((n, n)))

    # 变量
    p_vars = [cp.Variable(n) for _ in range(m)]  # diag(P_i)
    rho = [cp.Variable(nonneg=True) for _ in range(m)]  # ρ_i >= 0
    Q1 = cp.Variable((n, n), PSD=True)
    Q2 = cp.Variable((n, n), PSD=True)
    r1 = cp.Variable(n)
    r2 = cp.Variable(n)  # diag 元
    R1 = cp.diag(r1)
    R2 = cp.diag(r2)

    cons = []
    cons += [Q1 >> eps * np.eye(n), Q2 >> eps * np.eye(n)]
    cons += [r1 >= eps, r2 >= eps]

    Pmats = []
    for i in range(m):
        P_i = cp.diag(p_vars[i])
        Pmats.append(P_i)
        # ——规范化：防退化但不过分保守——
        cons += [p_vars[i] >= p_min]  # 每个对角元 ≥ p_min
        cons += [cp.sum(p_vars[i]) == n]  # trace(P_i) = n
        cons += [P_i - rho[i] * I_nC << 0]  # P_i ≤ ρ_i I

    # 4×4 块矩阵（式(15)）
    for i in range(m):
        A_i = A_list[i]
        W1_i = W1_list[i]
        W2_i = W2_list[i]
        P_i = Pmats[i]
        xi_sum = 0
        for j in range(m):
            xi_sum = xi_sum + float(xi_bar[i, j]) * Pmats[j]

        Xi11 = (
            -2 * P_i @ Dpi_c
            - sym(P_i @ A_i)
            + xi_sum
            + Q1
            + rho[i] * sigma1 * I_nC
            - T1_c @ R1 @ T2_c
        )
        Xi13 = P_i @ W1_i + 0.5 * R1 @ (T1_c + T2_c)
        Xi22 = (mu - 1) * Q1 - T1_c @ R2 @ T2_c + rho[i] * sigma2 * I_nC
        Xi24 = 0.5 * R2 @ (T1_c + T2_c)
        Xi33 = Q2 - R1
        Xi44 = (mu - 1) * Q2 - R2

        block = cp.bmat(
            [
                [Xi11, Z, Xi13, P_i @ W2_i],
                [Z, Xi22, Z, Xi24],
                [Xi13.T, Z, Xi33, Z],
                [(P_i @ W2_i).T, Xi24.T, Z, Xi44],
            ]
        )
        cons += [block << -eps * np.eye(4 * n)]

    # 求解（优先 MOSEK→CVXOPT→SCS）
    prob = cp.Problem(cp.Minimize(0), cons)
    solved = False
    for solver, kws in [
        ("MOSEK", dict(verbose=verbose)),
        ("CVXOPT", dict(verbose=verbose)),
        ("SCS", dict(verbose=verbose, eps=1e-6, max_iters=200000)),
    ]:
        try:
            prob.solve(solver=solver, **kws)
            solved = True
            break
        except Exception:
            continue
    if not solved:
        prob.solve(verbose=verbose)

    feasible = prob.status in ["optimal", "optimal_inaccurate"]
    result = None
    if feasible:
        result = {
            "P_list": [np.diag(p.value) for p in p_vars],
            "Q1": Q1.value,
            "Q2": Q2.value,
            "R1": np.diag(r1.value),
            "R2": np.diag(r2.value),
            "rho": [ri.value for ri in rho],
        }
    return feasible, result


# ---------- 搜最大 σ2 ----------
def maximize_sigma2(
    A_list,
    W1_list,
    W2_list,
    D_list,
    phi,
    psi,
    xi_bar,
    beta1,
    beta2,
    sigma1,
    mu,
    s2_low=0.0,
    s2_high=1.0,
    tol=1e-4,
    verbose=False,
    expand_cap=10.0,
    max_iter=80,
):
    high = s2_high
    low = s2_low
    best_sol = None
    # 向上扩展到“不可行”或上限
    while high < expand_cap:
        feas, _ = theorem1_feasible(
            A_list,
            W1_list,
            W2_list,
            D_list,
            phi,
            psi,
            xi_bar,
            beta1,
            beta2,
            sigma1,
            high,
            mu,
            verbose=verbose,
        )
        if feas:
            low = high
            high *= 2
        else:
            break
    # 二分
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        feas, sol = theorem1_feasible(
            A_list,
            W1_list,
            W2_list,
            D_list,
            phi,
            psi,
            xi_bar,
            beta1,
            beta2,
            sigma1,
            mid,
            mu,
            verbose=verbose,
        )
        if feas:
            low = mid
            best_sol = sol
        else:
            high = mid
        if high - low < tol:
            break
    return low, best_sol


# ---------- Example 1 参数（论文） ----------
def example1_params():
    A1 = np.diag([0.5, 0.5])
    A2 = np.diag([0.4, 0.4])
    W11 = np.array([[0.5, 0.3], [0.2, 0.1]])
    W21 = np.array([[0.1, 0.2], [0.3, 0.4]])
    W12 = np.array([[0.5, 0.6], [0.4, 0.2]])
    W22 = np.array([[0.2, 0.3], [0.4, 0.5]])
    xi_bar = np.array([[-0.6, 0.6], [0.4, -0.4]])  # Markov 情形
    beta1 = np.array([0.0, 0.0])
    beta2 = np.array([1.0, 1.0])
    mu = 0.05
    return [A1, A2], [W11, W12], [W21, W22], xi_bar, beta1, beta2, mu


def run_case(label, D_list, phi, psi, sigma1_list):
    A_list, W1_list, W2_list, xi_bar, beta1, beta2, mu = example1_params()
    print(f"\n[{label}]  σ1  ->  max feasible σ2")
    for s1 in sigma1_list:
        s2_star, _ = maximize_sigma2(
            A_list,
            W1_list,
            W2_list,
            D_list,
            phi,
            psi,
            xi_bar,
            beta1,
            beta2,
            s1,
            mu,
            s2_low=0.0,
            s2_high=1.0,
            tol=1e-4,
            verbose=False,
        )
        print(f"{s1:<4} -> {s2_star:.6f}")


if __name__ == "__main__":
    sigma1_list = [0.1, 0.2, 0.4, 0.8, 1.6]

    # 三种常见解读 —— 全都测一遍
    # 1) q=1, 用 D 的第一列 [0.5, 0.3]
    run_case(
        "q=1, D[:,0]",
        D_list=[np.array([0.5, 0.3])],
        phi=[-1.0],
        psi=[1.0],
        sigma1_list=sigma1_list,
    )

    # 2) q=1, 用 D 的第二列 [0.5, 0.7]
    run_case(
        "q=1, D[:,1]",
        D_list=[np.array([0.5, 0.7])],
        phi=[-1.0],
        psi=[1.0],
        sigma1_list=sigma1_list,
    )

    # 3) q=2, 两列都用（按式(15)对 q 维求和）
    run_case(
        "q=2, use both cols",
        D_list=[np.array([0.5, 0.3]), np.array([0.5, 0.7])],
        phi=[-1.0, -1.0],
        psi=[1.0, 1.0],
        sigma1_list=sigma1_list,
    )

    # 单点可行性（论文常用的 σ1=0.8, σ2=0.5）
    from math import isfinite

    A_list, W1_list, W2_list, xi_bar, beta1, beta2, mu = example1_params()
    D_list = [np.array([0.5, 0.3])]  # 你也可以切到上面任一解读
    feas, _ = theorem1_feasible(
        A_list,
        W1_list,
        W2_list,
        D_list,
        [-1.0],
        [1.0],
        xi_bar,
        beta1,
        beta2,
        0.8,
        0.5,
        mu,
    )
    print(f"\nFeasible at (σ1=0.8, σ2=0.5) with q=1,D[:,0] :", feas)
