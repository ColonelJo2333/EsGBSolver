"""
一般静态球对称黑洞的标量场散射截面计算模块

线元假设：ds² = -A(r)dt² + B(r)dr² + r²dΩ²

主接口：
    compute_scattering(A, B, dA, dB, omega, lmax, r_horizon, ...)
    → ScatteringResult

依赖：numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn, spherical_yn, eval_legendre
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# 结果类
# ═══════════════════════════════════════════════════════════════════════════════

class ScatteringResult:
    """
    散射计算结果

    属性
    ----
    omega       : 频率
    lmax        : 最大角动量
    mu          : 场质量
    s_matrix    : S-matrix 数组 S_l, shape (lmax+1,)
    theta       : 角度数组（弧度）
    dsigma      : 微分散射截面 dσ/dΩ
    a_in        : 入射波振幅 A_in^l
    a_out       : 出射波振幅 A_out^l
    """
    def __init__(self, omega, lmax, mu, s_matrix, theta, dsigma, a_in, a_out):
        self.omega = omega
        self.lmax = lmax
        self.mu = mu
        self.s_matrix = s_matrix
        self.theta = theta
        self.dsigma = dsigma
        self.a_in = a_in
        self.a_out = a_out

    def unitarity_check(self):
        """返回各分波 |S_l|，理想值为 1"""
        return np.abs(self.s_matrix)

    def plot(self, ax=None, label=None, **kwargs):
        """绘制微分散射截面 dσ/dΩ vs θ（对数纵轴）"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        theta_deg = np.degrees(self.theta)
        ax.semilogy(theta_deg, self.dsigma, label=label, **kwargs)
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=13)
        ax.set_ylabel(r'$d\sigma / d\Omega$', fontsize=13)
        ax.set_xlim(theta_deg[0], theta_deg[-1])
        ax.set_title(rf'$\omega = {self.omega},\; l_{{\max}} = {self.lmax}$',
                      fontsize=14)
        if label:
            ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


# ═══════════════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _yennie_reduction(a):
    """
    Yennie-Ravenhall reduction 一步

    递推关系（0-based 索引）：
      l = 0:  a_new[0] = a[0] - 1/3 · a[1]
      l ≥ 1:  a_new[l] = a[l] - (l+1)/(2l+3)·a[l+1] - l/(2l-1)·a[l-1]

    输出数组长度比输入少 1
    """
    n = len(a)
    if n < 2:
        return a.copy()
    result = np.zeros(n - 1, dtype=complex)
    # l = 0 特殊处理
    result[0] = a[0] - a[1] / 3.0
    # l = 1, ..., n-2
    for l in range(1, n - 1):
        result[l] = (a[l]
                     - (l + 1) / (2*l + 3) * a[l + 1]
                     - l / (2*l - 1) * a[l - 1])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 主计算接口
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scattering(A, B, dA, dB, omega, lmax, r_horizon,
                       epsilon=1e-5, rmax=600, mu=0.0,
                       r_sample_range=(300, 550), r_sample_step=0.5,
                       n_reduction=3,
                       theta_min_deg=20, theta_max_deg=180, theta_step_deg=0.1,
                       rtol=1e-12, atol=1e-12,
                       verbose=True):
    """
    计算一般静态球对称黑洞的标量场散射截面

    参数
    ----
    A, B        : 度规函数 A(r), B(r)，Python callable
    dA, dB      : 度规函数导数 A'(r), B'(r)，Python callable
    omega       : 散射频率
    lmax        : 角动量截断
    r_horizon   : 视界半径
    epsilon     : 视界偏移量（积分起点 r0 = r_h + ε）
    rmax        : ODE 积分终点（默认 600）
    mu          : 场质量（默认 0，无质量标量场）
    r_sample_range : 远场采样区间 (r_min, r_max)，默认 (300, 550)
    r_sample_step  : 远场采样步长，默认 0.5
    n_reduction : Yennie-Ravenhall reduction 次数（默认 3）
    theta_min_deg, theta_max_deg, theta_step_deg : 散射角范围（度）
    rtol, atol  : ODE 求解精度
    verbose     : 是否打印进度

    返回
    ----
    ScatteringResult
    """
    r0 = r_horizon + epsilon
    k = np.sqrt(omega**2 - mu**2)  # 波数

    # ─── Step 1: 远场采样点 ─────────────────────────────────────────────────
    r_sample = np.arange(r_sample_range[0],
                         r_sample_range[1] + r_sample_step / 2,
                         r_sample_step)

    # ─── Step 2: 乌龟坐标 r*(r) = ε + ∫_{r0}^{r} √(B/A) dr' ──────────────
    if verbose:
        print("计算乌龟坐标...")

    integrand = lambda r: np.sqrt(B(r) / A(r))

    rstar = np.zeros(len(r_sample))
    # 第一个点：从 r0 到 r_sample[0] 的完整积分
    rstar[0] = epsilon + quad(integrand, r0, r_sample[0],
                              limit=500, epsabs=1e-12, epsrel=1e-12)[0]
    # 后续点：累积积分（每段仅 0.5 宽度，高效且精确）
    for i in range(1, len(r_sample)):
        rstar[i] = rstar[i-1] + quad(integrand, r_sample[i-1], r_sample[i])[0]

    # 插值函数，用于连续求值
    rstar_interp = CubicSpline(r_sample, rstar)

    # ─── Step 3 & 4: 对每个 l 求解 ODE 并提取 S-matrix ───────────────────
    S_arr = np.zeros(lmax + 1, dtype=complex)
    A_in_arr = np.zeros(lmax + 1, dtype=complex)
    A_out_arr = np.zeros(lmax + 1, dtype=complex)

    # 近视界入射波初始条件（对所有 l 相同）
    R0 = np.exp(-1j * omega * epsilon)
    Rp0 = -1j * omega * np.sqrt(B(r0) / A(r0)) * R0
    y0 = [R0.real, R0.imag, Rp0.real, Rp0.imag]

    max_step = (rmax - r0) / 10000  # 类似 Mathematica 的 MaxStepFraction -> 1/10000

    for l_val in range(lmax + 1):
        if verbose and l_val % 10 == 0:
            print(f"  求解 l = {l_val}/{lmax} ...")

        # ── 径向方程 ODE ──
        # A/B·R'' + (BA'-AB')/(2B²)·R' +
        #   [ω²-l(l+1)A/r²-μ²A-A'/(2rB)+AB'/(2rB²)]·R = 0
        #
        # 转化为一阶系统：y = [Re(R), Im(R), Re(R'), Im(R')]
        def ode_rhs(r, y, _l=l_val):
            R_c = y[0] + 1j * y[1]
            Rp_c = y[2] + 1j * y[3]

            Ar = A(r); Br = B(r)
            dAr = dA(r); dBr = dB(r)

            # R' 的系数
            p = (Br * dAr - Ar * dBr) / (2.0 * Br**2)
            # R 的系数
            q = (omega**2
                 - _l * (_l + 1) * Ar / r**2
                 - mu**2 * Ar
                 - dAr / (2.0 * r * Br)
                 + Ar * dBr / (2.0 * r * Br**2))

            # R'' = -(B/A)(p·R' + q·R)
            Rpp = -(Br / Ar) * (p * Rp_c + q * R_c)
            return [y[2], y[3], Rpp.real, Rpp.imag]

        sol = solve_ivp(ode_rhs, [r0, rmax], y0,
                        method='DOP853', t_eval=r_sample,
                        rtol=rtol, atol=atol,
                        max_step=max_step)

        if not sol.success:
            if verbose:
                print(f"  [!] l={l_val} ODE 求解失败: {sol.message}")
            continue

        R_data = sol.y[0] + 1j * sol.y[1]

        # ── 远场拟合：线性最小二乘提取 A_in, A_out ──
        # 渐近形式：
        # R ≈ ω·r* · [A_out·i^(l+1)·h_l^(1)(k·r*)
        #           + A_in·(-i)^(l+1)·conj(h_l^(1)(k·r*))]
        rs = rstar_interp(sol.t).real
        kr = k * rs
        h1 = spherical_jn(l_val, kr) + 1j * spherical_yn(l_val, kr)

        f_out = omega * rs * (1j)**(l_val + 1) * h1
        f_in = omega * rs * (-1j)**(l_val + 1) * np.conj(h1)

        # 复数线性最小二乘：R_data = A_out · f_out + A_in · f_in
        M_mat = np.column_stack([f_out, f_in])
        coeffs = np.linalg.lstsq(M_mat, R_data, rcond=None)[0]

        A_out_arr[l_val] = coeffs[0]
        A_in_arr[l_val] = coeffs[1]
        S_arr[l_val] = (-1)**(l_val + 1) * coeffs[0] / coeffs[1]

    if verbose:
        abs_S = np.abs(S_arr)
        print(f"S-matrix 幺正性: |S_l| in [{abs_S.min():.6f}, {abs_S.max():.6f}]")

    # ─── Step 5: 微分散射截面（Yennie-Ravenhall reduction）──────────────
    if verbose:
        print("计算微分散射截面...")

    # 初始系数 a0_l = (2l+1)(S_l - 1)
    l_arr = np.arange(lmax + 1, dtype=float)
    a_table = (2 * l_arr + 1) * (S_arr - 1)

    # n 次 reduction
    for _ in range(n_reduction):
        a_table = _yennie_reduction(a_table)

    # 角度网格
    theta = np.arange(theta_min_deg, theta_max_deg + theta_step_deg / 2,
                      theta_step_deg) * np.pi / 180
    cos_theta = np.cos(theta)

    # Legendre 多项式矩阵 P_l(cosθ)，形状 (n_l, n_theta)
    n_l = len(a_table)
    P_mat = np.zeros((n_l, len(theta)))
    for l_idx in range(n_l):
        P_mat[l_idx] = eval_legendre(l_idx, cos_theta)

    # 散射振幅 f(θ) = 1/(1-cosθ)^n · 1/(2ik) · Σ_l a_l · P_l(cosθ)
    f_sum = a_table @ P_mat   # shape (n_theta,)
    f_amp = f_sum / (2j * k) / (1.0 - cos_theta)**n_reduction
    dsigma = np.abs(f_amp)**2

    if verbose:
        print("完成！")

    return ScatteringResult(omega, lmax, mu, S_arr, theta, dsigma,
                            A_in_arr, A_out_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# Schwarzschild 便捷函数
# ═══════════════════════════════════════════════════════════════════════════════

def schwarzschild_scattering(omega=20, lmax=70, epsilon=1e-5, **kwargs):
    """
    Schwarzschild 黑洞散射截面（A = 1-1/r, B = 1/A, r_h = 1）

    用于验证代码正确性。
    """
    # Schwarzschild 度规 (r_h = 1, 即 2M = 1)
    A  = lambda r: 1.0 - 1.0 / r
    B  = lambda r: 1.0 / (1.0 - 1.0 / r)
    dA = lambda r: 1.0 / r**2
    dB = lambda r: -1.0 / (r - 1)**2

    return compute_scattering(A, B, dA, dB, omega, lmax,
                              r_horizon=1.0, epsilon=epsilon, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 总吸收截面 σ_abs 计算
# ═══════════════════════════════════════════════════════════════════════════════

class AbsorptionResult:
    """单个 ω 的吸收截面结果"""
    def __init__(self, omega, mu, sigma_abs, sigma_l_list, lmax_used):
        self.omega = omega
        self.mu = mu
        self.sigma_abs = sigma_abs
        self.sigma_l_list = sigma_l_list   # 各分波贡献列表
        self.lmax_used = lmax_used


def _compute_Sl_single(A, B, dA, dB, omega, l_val, mu, r0, rmax,
                       y0, max_step, r_sample, rstar_interp, k,
                       rtol=1e-12, atol=1e-12):
    """
    计算单个 l 的 S-matrix 元素 S_l

    复用 compute_scattering 中的 ODE + 远场拟合逻辑。
    返回 (S_l, A_in, A_out) 或 (None, None, None) 失败时。
    """
    def ode_rhs(r, y, _l=l_val):
        R_c = y[0] + 1j * y[1]
        Rp_c = y[2] + 1j * y[3]
        Ar = A(r); Br = B(r)
        dAr = dA(r); dBr = dB(r)
        p = (Br * dAr - Ar * dBr) / (2.0 * Br**2)
        q = (omega**2
             - _l * (_l + 1) * Ar / r**2
             - mu**2 * Ar
             - dAr / (2.0 * r * Br)
             + Ar * dBr / (2.0 * r * Br**2))
        Rpp = -(Br / Ar) * (p * Rp_c + q * R_c)
        return [y[2], y[3], Rpp.real, Rpp.imag]

    sol = solve_ivp(ode_rhs, [r0, rmax], y0,
                    method='DOP853', t_eval=r_sample,
                    rtol=rtol, atol=atol,
                    max_step=max_step)

    if not sol.success:
        return None, None, None

    R_data = sol.y[0] + 1j * sol.y[1]

    # 远场拟合
    rs = rstar_interp(sol.t).real
    kr = k * rs
    h1 = spherical_jn(l_val, kr) + 1j * spherical_yn(l_val, kr)

    f_out = omega * rs * (1j)**(l_val + 1) * h1
    f_in = omega * rs * (-1j)**(l_val + 1) * np.conj(h1)

    M_mat = np.column_stack([f_out, f_in])
    coeffs = np.linalg.lstsq(M_mat, R_data, rcond=None)[0]

    A_out = coeffs[0]
    A_in = coeffs[1]
    S_l = (-1)**(l_val + 1) * A_out / A_in

    return S_l, A_in, A_out


def compute_absorption(A, B, dA, dB, omega, mu, r_horizon,
                       epsilon=1e-5, rmax=600,
                       r_sample_range=(300, 550), r_sample_step=0.5,
                       rtol=1e-12, atol=1e-12,
                       lmax_cap=None, convergence_tol=1e-7,
                       verbose=True):
    """
    计算单个 ω 的总吸收截面 σ_abs（自适应 l_max 截断）

    参数
    ----
    convergence_tol : 连续 3 个 σ_l < convergence_tol * σ_abs 时停止
    lmax_cap        : l 安全上限，默认 max(50, ceil(3*ω))

    返回
    ----
    AbsorptionResult
    """
    if omega <= mu:
        raise ValueError(f"需要 ω > μ 才有散射 (ω={omega}, μ={mu})")

    k = np.sqrt(omega**2 - mu**2)
    r0 = r_horizon + epsilon

    if lmax_cap is None:
        lmax_cap = max(50, int(np.ceil(3 * omega)))

    # 远场采样点
    r_sample = np.arange(r_sample_range[0],
                         r_sample_range[1] + r_sample_step / 2,
                         r_sample_step)

    # 乌龟坐标
    integrand = lambda r: np.sqrt(B(r) / A(r))
    rstar = np.zeros(len(r_sample))
    rstar[0] = epsilon + quad(integrand, r0, r_sample[0],
                              limit=500, epsabs=1e-12, epsrel=1e-12)[0]
    for i in range(1, len(r_sample)):
        rstar[i] = rstar[i-1] + quad(integrand, r_sample[i-1], r_sample[i])[0]
    rstar_interp = CubicSpline(r_sample, rstar)

    # 近视界初始条件
    R0 = np.exp(-1j * omega * epsilon)
    Rp0 = -1j * omega * np.sqrt(B(r0) / A(r0)) * R0
    y0 = [R0.real, R0.imag, Rp0.real, Rp0.imag]
    max_step = (rmax - r0) / 10000

    sigma_abs = 0.0
    sigma_l_list = []
    consecutive_small = 0  # 连续小贡献计数

    for l_val in range(lmax_cap + 1):
        S_l, _, _ = _compute_Sl_single(
            A, B, dA, dB, omega, l_val, mu, r0, rmax,
            y0, max_step, r_sample, rstar_interp, k,
            rtol=rtol, atol=atol)

        if S_l is None:
            if verbose:
                print(f"  [!] l={l_val} ODE 求解失败，跳过")
            sigma_l_list.append(0.0)
            continue

        sigma_l = (np.pi / k**2) * (2 * l_val + 1) * (1.0 - np.abs(S_l)**2)
        sigma_l_list.append(sigma_l)
        sigma_abs += sigma_l

        # 自适应截断检查
        if sigma_abs > 0 and abs(sigma_l) < convergence_tol * abs(sigma_abs):
            consecutive_small += 1
        else:
            consecutive_small = 0

        if consecutive_small >= 3 and l_val >= 3:
            if verbose:
                print(f"  ω={omega:.4f}: 收敛于 l_max={l_val}, "
                      f"σ_abs={sigma_abs:.6e}")
            return AbsorptionResult(omega, mu, sigma_abs,
                                    sigma_l_list, l_val)

    if verbose:
        print(f"  ω={omega:.4f}: 达到 l_max_cap={lmax_cap}, "
              f"σ_abs={sigma_abs:.6e} (可能未完全收敛)")

    return AbsorptionResult(omega, mu, sigma_abs,
                            sigma_l_list, lmax_cap)


def scan_absorption(A, B, dA, dB, mu, r_horizon,
                    omega_range=None, omega_start=2.0, omega_end=20.0,
                    omega_step=0.5,
                    **kwargs):
    """
    扫描 ω 计算总吸收截面

    参数
    ----
    omega_range : 自定义 ω 数组（优先使用），若为 None 则用 start/end/step
    其余参数传递给 compute_absorption

    返回
    ----
    results : list of AbsorptionResult
    table   : np.ndarray, shape (N, 3)，列为 [ω, σ_abs, l_max_used]
    """
    if omega_range is None:
        omega_range = np.arange(omega_start,
                                omega_end + omega_step / 2,
                                omega_step)

    # 过滤掉 ω ≤ μ 的值
    omega_range = omega_range[omega_range > mu]

    verbose = kwargs.pop('verbose', True)
    results = []

    if verbose:
        print(f"吸收截面扫描: ω ∈ [{omega_range[0]:.2f}, {omega_range[-1]:.2f}], "
              f"共 {len(omega_range)} 个点, μ={mu}")
        print("=" * 60)

    for i, omega in enumerate(omega_range):
        if verbose:
            print(f"[{i+1}/{len(omega_range)}] ω = {omega:.4f} ...")

        res = compute_absorption(A, B, dA, dB, omega, mu, r_horizon,
                                 verbose=verbose, **kwargs)
        results.append(res)

    # 构建汇总表
    table = np.array([[r.omega, r.sigma_abs, r.lmax_used] for r in results])

    if verbose:
        print("=" * 60)
        print("扫描完成！")
        print(f"{'ω':>8s}  {'σ_abs':>14s}  {'l_max':>5s}")
        for row in table:
            print(f"{row[0]:8.4f}  {row[1]:14.6e}  {int(row[2]):5d}")

    return results, table


def schwarzschild_absorption_scan(mu=0.0, **kwargs):
    """
    Schwarzschild 极限下的吸收截面扫描（用于验证）

    r_h = 1, M = 1/2
    μ=0 时：
      低频极限：σ_abs → 4π r_h² = 4π ≈ 12.566（面积定理）
      高频极限：振荡收敛于 27πM² = 27π/4 ≈ 21.206（几何光学截面）
    μ≠0 时：
      阈值附近 ω → μ：σ_abs → ∞（σ_l ∝ 1/k∞²，k∞ → 0）
      高频极限：仍然收敛于 27π/4（质量效应高频可忽略）
    """
    A  = lambda r: 1.0 - 1.0 / r
    B  = lambda r: 1.0 / (1.0 - 1.0 / r)
    dA = lambda r: 1.0 / r**2
    dB = lambda r: -1.0 / (r - 1)**2

    return scan_absorption(A, B, dA, dB, mu=mu, r_horizon=1.0, **kwargs)
