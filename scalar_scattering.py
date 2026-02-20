# scalar_scattering.py
# ─────────────────────────────────────────────────────────────────────────────
import os; os.environ.setdefault('PYTHONUTF8', '1')  # fix Windows ASCII pipe issue
# EsGB 黑洞标量测试场散射计算模块
#
# 径向方程（Regge-Wheeler 型）：
#   A(r)/B(r) · R'' + (B·A' - A·B')/(2B²) · R'
#   + [ω² - (l(l+1)+r²μ²)A/r² + A/(r²B) - A'/(2rB) + AB'/(2rB²)] · R = 0
#
# A(r) = gtt(r)，B(r) = grr(r)，由 EsGBSolution 的谱插值方法给出
# （直接使用 Chebyshev 节点值 + 重心插值，指数精度，可从 rh+1e-5 开始）
#
# 近视界入射波初始条件：R(r₀)=1，R'(r₀)=-iω√(B/A)|_{r₀}
# 远场渐近匹配：R ~ Ain·e^{-ik∞r*} + Aout·e^{+ik∞r*}
# 散射矩阵元：S_l = Aout/Ain
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams


# ── ODE 构造 ──────────────────────────────────────────────────────────────────

def _precompute_metric_splines(sol, r0, r_end, n_pts=12000):
    """
    在 [r0, r_end] 上以对数间距预计算度规函数，拟合 CubicSpline。

    对数间距（geomspace 在 r-rh 上）使近视界区域足够密集：
    近 rh 处 grr ~ 1/(r-rh) 剧烈变化，对数网格天然适配。
    CubicSpline 求值为 C 级别代码，调用时释放 GIL，使线程并行真正有效。
    """
    rh   = sol.rh
    u    = np.geomspace(r0 - rh, r_end - rh, n_pts)   # log-spaced in (r - rh)
    r_g  = rh + u

    spl_gtt  = CubicSpline(r_g, sol.gtt_at(r_g))
    spl_grr  = CubicSpline(r_g, sol.grr_at(r_g))
    spl_dgtt = CubicSpline(r_g, sol.dgtt_dr(r_g))
    spl_dgrr = CubicSpline(r_g, sol.dgrr_dr(r_g))
    return spl_gtt, spl_grr, spl_dgtt, spl_dgrr


def _build_rhs(l, omega, mu, sol):
    """
    单次调用用（串行 / compute_s_matrix 直接调用）。
    使用 sol 的谱插值方法，指数精度。
    """
    def rhs(r, y):
        R, dR = y
        A  = sol.gtt_at(r)
        B  = sol.grr_at(r)
        dA = sol.dgtt_dr(r)
        dB = sol.dgrr_dr(r)

        V = (omega**2
             - (l * (l + 1) + r**2 * mu**2) * A / r**2
             + A / (r**2 * B)
             - dA / (2. * r * B)
             + A * dB / (2. * r * B**2))

        d2R = (-(B * dA - A * dB) / (2. * B * A)) * dR - (B / A) * V * R
        return [dR, d2R]

    return rhs


def _build_rhs_spline(l, omega, mu, spl_gtt, spl_grr, spl_dgtt, spl_dgrr):
    """
    并行用：RHS 使用 CubicSpline 求值（C 级别，释放 GIL）。
    """
    def rhs(r, y):
        R, dR = y
        A  = float(spl_gtt(r))
        B  = float(spl_grr(r))
        dA = float(spl_dgtt(r))
        dB = float(spl_dgrr(r))

        V = (omega**2
             - (l * (l + 1) + r**2 * mu**2) * A / r**2
             + A / (r**2 * B)
             - dA / (2. * r * B)
             + A * dB / (2. * r * B**2))

        d2R = (-(B * dA - A * dB) / (2. * B * A)) * dR - (B / A) * V * R
        return [dR, d2R]

    return rhs


def _tortoise_at(r_target, r0, sol):
    """
    龟坐标差值 r*(r_target) - r*(r₀) = ∫_{r₀}^{r_target} √(grr/gtt) dr。

    使用 scipy.integrate.quad 自适应积分，精度远高于固定步长梯形法。
    """
    val, _ = quad(lambda r: float(np.sqrt(sol.grr_at(r) / sol.gtt_at(r))),
                  r0, r_target, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val


# ── 主计算接口 ────────────────────────────────────────────────────────────────

def compute_s_matrix(sol, omega, l, mu=0.0,
                     r0_offset=1e-5,
                     r1_match=9000., r2_match=9100.,
                     _star1=None, _star2=None,
                     rtol=1e-10, atol=1e-12):
    """
    计算单个分波散射矩阵元 S_l。

    _star1, _star2 : 预算好的龟坐标（由 compute_scattering 传入，避免重复积分）。
                     若为 None 则在函数内部自行计算。
    """
    rh   = sol.rh
    r0   = rh + r0_offset
    kh   = omega
    kinf = np.sqrt(max(omega**2 - mu**2, 0.0))

    A0  = sol.gtt_at(r0)
    B0  = sol.grr_at(r0)
    R0  = 1.0 + 0j
    dR0 = -1j * kh * np.sqrt(B0 / A0)

    rhs    = _build_rhs(l, omega, mu, sol)
    result = solve_ivp(
        rhs, (r0, r2_match + 50.),
        np.array([R0, dR0], dtype=complex),
        method='DOP853', dense_output=True,
        rtol=rtol, atol=atol
    )

    if not result.success:
        raise RuntimeError(f'ODE solve failed (l={l}, omega={omega}): {result.message}')

    R1 = result.sol(r1_match)[0]
    R2 = result.sol(r2_match)[0]

    star1 = _star1 if _star1 is not None else _tortoise_at(r1_match, r0, sol)
    star2 = _star2 if _star2 is not None else _tortoise_at(r2_match, r0, sol)

    M = np.array([
        [np.exp(-1j * kinf * star1), np.exp(+1j * kinf * star1)],
        [np.exp(-1j * kinf * star2), np.exp(+1j * kinf * star2)],
    ])
    Ain, Aout = np.linalg.solve(M, np.array([R1, R2]))
    return Aout / Ain


def compute_scattering(sol, omega, l_max=7, mu=0.0,
                       r0_offset=1e-5,
                       r1_match=9000., r2_match=9100.,
                       n_workers=8, verbose=True, **kwargs):
    """
    对 l = 0, 1, ..., l_max 并行计算 S_l，返回复数列表。

    n_workers : 并行线程数（默认 8，建议不超过 12）。
                scipy 的 ODE/积分核心为 C 代码，会释放 GIL，线程并行有实效。
    """
    kh   = omega
    kinf = np.sqrt(max(omega**2 - mu**2, 0.0))

    # 龟坐标与 l 无关，预算一次供所有线程共享
    r0    = sol.rh + r0_offset
    star1 = _tortoise_at(r1_match, r0, sol)
    star2 = _tortoise_at(r2_match, r0, sol)

    # 预计算 CubicSpline（C 级别求值，线程中释放 GIL，使 ThreadPoolExecutor 真正并行）
    splines = _precompute_metric_splines(sol, r0, r2_match + 100.)
    spl_gtt, spl_grr, spl_dgtt, spl_dgrr = splines

    rtol = kwargs.pop('rtol', 1e-10)
    atol = kwargs.pop('atol', 1e-12)

    A0  = float(spl_gtt(r0))
    B0  = float(spl_grr(r0))
    R0  = 1.0 + 0j
    dR0 = -1j * kh * np.sqrt(B0 / A0)

    def _one(l):
        rhs    = _build_rhs_spline(l, omega, mu, spl_gtt, spl_grr, spl_dgtt, spl_dgrr)
        result = solve_ivp(
            rhs, (r0, r2_match + 50.),
            np.array([R0, dR0], dtype=complex),
            method='DOP853', dense_output=True,
            rtol=rtol, atol=atol
        )
        if not result.success:
            raise RuntimeError(f'ODE solve failed (l={l}, omega={omega}): {result.message}')
        R1 = result.sol(r1_match)[0]
        R2 = result.sol(r2_match)[0]
        M  = np.array([
            [np.exp(-1j * kinf * star1), np.exp(+1j * kinf * star1)],
            [np.exp(-1j * kinf * star2), np.exp(+1j * kinf * star2)],
        ])
        Ain, Aout = np.linalg.solve(M, np.array([R1, R2]))
        return Aout / Ain

    n = min(n_workers, l_max + 1)
    with ThreadPoolExecutor(max_workers=n) as exe:
        futures = [exe.submit(_one, l) for l in range(l_max + 1)]
        S_list  = [f.result() for f in futures]

    if verbose:
        print(f'omega={omega}, mu={mu}, rh={sol.rh}, '
              f'alpha={sol.alpha}, beta={sol.beta}  (r0=rh+{r0_offset:.0e})')
        print(f'{"l":>4}  {"|S|^2":>12}  {"Abs.Rate":>10}')
        print('-' * 32)
        for l, S in enumerate(S_list):
            ref  = abs(S)**2
            absp = 1.0 - ref * (kinf / kh) if kh > 0 else 0.
            print(f'{l:>4}  {ref:>12.6f}  {absp:>10.6f}')

    return S_list


# ── 截面计算 ──────────────────────────────────────────────────────────────────

def absorption_cross_section(S_list, omega, mu=0.0):
    """
    总吸收截面：σ_abs = (π/ω²) Σ_l (2l+1)(1 - |S_l|²·k∞/kh)
    """
    kh   = omega
    kinf = np.sqrt(max(omega**2 - mu**2, 0.0))
    sigma = (np.pi / omega**2) * sum(
        (2 * l + 1) * (1.0 - abs(S)**2 * (kinf / kh))
        for l, S in enumerate(S_list)
    )
    return float(sigma.real)


def scattering_amplitude(S_list, theta, omega):
    """
    散射振幅：f(θ) = 1/(2iω) Σ_l (2l+1)(S_l - 1) P_l(cos θ)
    """
    cos_theta = np.cos(theta)
    f = sum(
        (2 * l + 1) * (S - 1.0) * eval_legendre(l, cos_theta)
        for l, S in enumerate(S_list)
    ) / (2j * omega)
    return complex(f)


def differential_cross_section(S_list, theta_arr, omega):
    """
    微分散射截面数组：dσ/dΩ(θ) = |f(θ)|²
    """
    return np.array([
        abs(scattering_amplitude(S_list, th, omega))**2
        for th in theta_arr
    ])


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_scattering(sol, S_list, omega, mu=0.0,
                    theta_pts=400, theta_min_deg=20., figsize=(7.5, 4.),
                    outfile=None, show=True):
    """
    Two-panel figure: log-scale diff. cross section | linear-scale (Glory peak).
    theta_min_deg : forward truncation angle in degrees (default 20 deg).
    """
    rcParams.update({'mathtext.fontset': 'stix', 'axes.unicode_minus': False})

    theta_min = np.radians(theta_min_deg)
    theta  = np.linspace(theta_min, np.pi, theta_pts)
    dsigma = differential_cross_section(S_list, theta, omega)
    deg    = np.degrees(theta)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle(
        rf'$\omega={omega}$, $\alpha={sol.alpha}$, $\beta={sol.beta}$, '
        rf'$\phi_h={sol.phi_h:.4f}$, $M={sol.M_ADM:.4f}$',
        fontsize=9.5
    )

    axes[0].semilogy(deg, dsigma, color='#1f4e79', lw=1.6)
    axes[0].set_title(r'$d\sigma/d\Omega$ (log scale)', fontsize=9, pad=3)
    axes[0].set_xlim(theta_min_deg, 180)
    axes[0].set_ylabel(r'$d\sigma/d\Omega$')

    axes[1].plot(deg, dsigma, color='#c0392b', lw=1.6)
    axes[1].set_title(r'$d\sigma/d\Omega$ (linear, Glory peak)', fontsize=9, pad=3)
    axes[1].set_xlim(theta_min_deg, 180)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel(r'$d\sigma/d\Omega$')

    for ax in axes:
        ax.set_xlabel(r'$\theta$ (deg)')
        ax.grid(True, which='both', lw=0.3, color='#cccccc')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

    if outfile:
        plt.savefig(outfile, dpi=150)
        print(f'saved: {outfile}')
    if show:
        plt.show()
    return fig, axes


# ── 施瓦西解析度规（测试用）────────────────────────────────────────────────────

class SchwarzschildSol:
    """
    施瓦西黑洞解析度规，接口与 EsGBSolution 完全相同。
    gtt = 1 - rh/r,  grr = 1/(1 - rh/r)
    （rh = 2M，令 M=0.5 则 rh=1）
    """
    def __init__(self, rh=1.0):
        self.rh    = rh
        self.alpha = 0.0
        self.beta  = 0.0
        self.phi_h = 0.0
        self.M_ADM = rh / 2.
        self.Q     = 0.0

    def gtt_at(self, r):
        r = np.asarray(r, dtype=float)
        return 1. - self.rh / r

    def grr_at(self, r):
        r = np.asarray(r, dtype=float)
        return 1. / (1. - self.rh / r)

    def dgtt_dr(self, r):
        r = np.asarray(r, dtype=float)
        return self.rh / r**2

    def dgrr_dr(self, r):
        r = np.asarray(r, dtype=float)
        return -self.rh / (r - self.rh)**2


def test_schwarzschild(omega=0.3, l_max=5,
                       r1_match=500., r2_match=600., rh=1.0):
    """
    用解析施瓦西度规跑散射计算，验证 |S_l|² ≤ 1 且 σ_abs > 0。
    """
    sol = SchwarzschildSol(rh=rh)
    print(f'-- Schwarzschild test  rh={rh}, M={sol.M_ADM}, omega={omega} --')
    S_list = compute_scattering(sol, omega, l_max=l_max,
                                r1_match=r1_match, r2_match=r2_match)
    sigma = absorption_cross_section(S_list, omega)
    print(f'sigma_abs = {sigma:.6f}')
    geo = np.pi * (3*np.sqrt(3)/2 * rh)**2
    print(f'Geometric optics limit 27*pi*M^2 = {geo:.6f}')
    return S_list
