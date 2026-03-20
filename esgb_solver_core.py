# esgb_solver_core.py
# ─────────────────────────────────────────────────────────────────────────────
# EsGB (Einstein-scalar-Gauss-Bonnet) 标量化黑洞  Chebyshev 伪谱 BVP 求解器
# 支持多种耦合函数：多项式 f=φ²+βφ⁴ / 指数 f=(λ²/8)(1-e^{-3φ²})
# 公开接口：solve(rh, alpha, coupling, phi_h_init, N=13)  →  EsGBSolution
# ─────────────────────────────────────────────────────────────────────────────
import warnings
import numpy as np
from scipy import linalg
import re, os

_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# 耦合函数
# ══════════════════════════════════════════════════════════════════════════════

class Coupling:
    """EsGB 耦合函数 f(φ) 的基类。子类须实现 fdot, fddot。"""

    def f(self, phi):
        raise NotImplementedError

    def fdot(self, phi):
        """f'(φ)"""
        raise NotImplementedError

    def fddot(self, phi):
        """f''(φ)"""
        raise NotImplementedError

    def fdot_half(self, phi):
        """f'(φ)/2 — 场方程中出现的组合"""
        return self.fdot(phi) / 2.

    def fddot_half(self, phi):
        """f''(φ)/2 — 场方程中出现的组合"""
        return self.fddot(phi) / 2.

    def phi_h_allowed_range(self, rh, alpha):
        """视界正则性对 φ_h 的约束。返回 (phi_c1, phi_c2)，无禁戒返回 (inf, inf)。"""
        return np.inf, np.inf

    def label_str(self):
        """用于图例的 LaTeX 参数字符串"""
        raise NotImplementedError


class PolynomialCoupling(Coupling):
    r"""f(φ) = φ² + β·φ⁴"""

    def __init__(self, beta):
        self.beta = beta

    def f(self, phi):
        return phi ** 2 + self.beta * phi ** 4

    def fdot(self, phi):
        return 2. * phi + 4. * self.beta * phi ** 3

    def fddot(self, phi):
        return 2. + 12. * self.beta * phi ** 2

    def phi_h_allowed_range(self, rh, alpha):
        beta = self.beta
        limit = rh ** 2 / (np.sqrt(96.) * alpha)
        if beta >= 0:
            return np.inf, np.inf
        phi_star = np.sqrt(-1. / (6. * beta))
        fdot_max = (4. / 3.) * phi_star
        if fdot_max <= limit + 1e-12:
            return np.inf, np.inf
        coeffs = [4 * beta, 0, 2, -limit]
        roots = np.roots(coeffs)
        pos_real = sorted([r.real for r in roots
                           if abs(r.imag) < 1e-8 and r.real > 0])
        if len(pos_real) < 2:
            return np.inf, np.inf
        return pos_real[0], pos_real[1]

    def label_str(self):
        return rf'\beta={self.beta}'

    def __repr__(self):
        return f'PolynomialCoupling(beta={self.beta})'


class ExponentialCoupling(Coupling):
    r"""f(φ) = 1 - e^{-βφ²}"""

    def __init__(self, beta=3.0):
        self.beta = beta

    def f(self, phi):
        return 1. - np.exp(-self.beta * phi ** 2)

    def fdot(self, phi):
        b = self.beta
        return 2. * b * phi * np.exp(-b * phi ** 2)

    def fddot(self, phi):
        b = self.beta
        return 2. * b * np.exp(-b * phi ** 2) * (1. - 2. * b * phi ** 2)

    def phi_h_allowed_range(self, rh, alpha):
        limit = rh ** 2 / (np.sqrt(96.) * alpha)
        # f'(φ) 单峰，最大值在 φ* = 1/√(2β)
        phi_star = 1. / np.sqrt(2. * self.beta)
        fdot_max = abs(self.fdot(phi_star))
        if fdot_max <= limit + 1e-12:
            return np.inf, np.inf
        from scipy.optimize import brentq
        g = lambda phi: abs(self.fdot(phi)) - limit
        try:
            phi_c1 = brentq(g, 1e-10, phi_star)
            phi_c2 = brentq(g, phi_star, phi_star * 10.)
        except ValueError:
            return np.inf, np.inf
        return phi_c1, phi_c2

    def label_str(self):
        return rf'\beta={self.beta}'

    def __repr__(self):
        return f'ExponentialCoupling(beta={self.beta})'


# ── 方程加载（耦合函数无关）────────────────────────────────────────────────────
def _load_eq(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if '\u2192' in text:
        text = text.split('\u2192', 1)[1].strip()
    while 'pow(' in text:
        new = re.sub(r'pow\(([^()]+),([^()]+)\)', r'(\1)**(\2)', text)
        if new == text:
            break
        text = new
    # 将硬编码的耦合函数表达式替换为通用参数 fdot_half, fddot_half
    #   f'(φ)/2 = φ(1+2βφ²)  →  fdot_half
    #   f''(φ)/2 = 1+6βφ²    →  fddot_half
    text = text.replace('phival*(1 + 2*beta*(phival)**(2))', 'fdot_half')
    text = text.replace('(1 + 6*beta*(phival)**(2))', 'fddot_half')
    return text


def _make_func(expr, name):
    code = (
        f'def {name}(ht,dhtx,d2htx,Gt,dGtx,phival,dphix,d2phix,'
        f'x,rh,alpha,fdot_half,fddot_half):\n    return {expr}\n'
    )
    ns = {'np': np}
    exec(code, ns)
    return ns[name]

_r1 = _make_func(_load_eq(os.path.join(_DIR, 'eq1.txt')), '_r1')
_r2 = _make_func(_load_eq(os.path.join(_DIR, 'eq2.txt')), '_r2')
_r3 = _make_func(_load_eq(os.path.join(_DIR, 'eq3.txt')), '_r3')


def _res1(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
          x, rh, alpha, fdot_half, fddot_half):
    return _r1(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
               x, rh, alpha, fdot_half, fddot_half)

def _res2(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
          x, rh, alpha, fdot_half, fddot_half):
    return _r2(ht, dhtx, 0., Gt, dGtx, phi, dphix, 0.,
               x, rh, alpha, fdot_half, fddot_half)

def _res3(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
          x, rh, alpha, fdot_half, fddot_half):
    return _r3(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
               x, rh, alpha, fdot_half, fddot_half)


# ── Chebyshev 微分矩阵 ────────────────────────────────────────────────────────
def chebyshev_matrix(N):
    """Gauss-Lobatto 节点 x_j=cos(jπ/N)，返回 (x, D, D²)"""
    j  = np.arange(N + 1, dtype=float)
    x  = np.cos(j * np.pi / N)
    c  = np.ones(N + 1); c[0] = 2.; c[N] = 2.
    cs = c * (-1.) ** j
    X  = np.tile(x, (N + 1, 1))
    D  = np.outer(cs, 1. / cs) / (X.T - X + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return x, D, D @ D


# ── 视界正则性 ────────────────────────────────────────────────────────────────
def phi_h_prime(phi_h, rh, alpha, coupling):
    """视界处 dφ/dr（正则性唯一确定）"""
    fdot = coupling.fdot(phi_h)
    if abs(fdot) < 1e-30:
        return 0.
    disc = 1. - 96. * (alpha * fdot) ** 2 / rh ** 4
    if disc < 0:
        return 1e10
    return rh / (4. * alpha * fdot) * (-1. + np.sqrt(disc))


# ── 残差向量 ──────────────────────────────────────────────────────────────────
def build_residual(u, x_nodes, D, D2, rh, alpha, coupling):
    N = len(x_nodes) - 1;  M = N + 1
    ht = u[:M];  Gt = u[M:2*M];  phi = u[2*M:]
    dhtx  = D @ ht;   d2htx = D2 @ ht
    dGtx  = D @ Gt
    dphix = D @ phi;  d2phix = D2 @ phi
    x = x_nodes

    # 预计算耦合函数导数（向量化）
    fdot_half  = coupling.fdot_half(phi)
    fddot_half = coupling.fddot_half(phi)

    e1 = _res1(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
               x, rh, alpha, fdot_half, fddot_half)
    e2 = _res2(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
               x, rh, alpha, fdot_half, fddot_half)
    e3 = _res3(ht, dhtx, d2htx, Gt, dGtx, phi, dphix, d2phix,
               x, rh, alpha, fdot_half, fddot_half)

    # 视界处 ht 的 Robin BC
    _ph   = phi[N];   _Gt   = Gt[N];   _dGtx  = dGtx[N]
    _dp   = dphix[N]; _d2p  = d2phix[N]
    _fdot  = coupling.fdot(_ph)
    _fddot = coupling.fddot(_ph)
    _denom = 2. * _Gt * (rh ** 2 + 4. * _dp * _fdot * alpha)
    _numer = (_Gt * (-1. + 2. * _dGtx + _dp ** 2 + _Gt) * rh ** 2
              + 4. * (-2. * _dp ** 2 * _fddot * _Gt
                      - 2. * _d2p * _fdot * _Gt
                      + 3. * _dp * _fdot * (1. + _Gt)) * alpha)
    _dhtx_bc = ht[N] * _numer / _denom

    R = np.zeros(3 * M)
    R[0]           = ht[0]  - 1.
    R[1:N]         = e1[1:N]
    R[N]           = dhtx[N] - _dhtx_bc
    R[M]           = Gt[0]  - 1.
    R[M+1:M+N]     = e2[1:N]
    R[M+N]         = e2[N]
    R[2*M]         = phi[0]
    R[2*M+1:3*M-1] = e3[1:N]
    R[3*M-1]       = dphix[N] - phi_h_prime(phi[N], rh, alpha, coupling) * (rh / 2.)
    return R


# ── 数值 Jacobian ─────────────────────────────────────────────────────────────
def _jacobian(F, u, eps=1e-7):
    n = len(u);  J = np.zeros((n, n))
    for i in range(n):
        up = u.copy(); um = u.copy()
        up[i] += eps;  um[i] -= eps
        J[:, i] = (F(up) - F(um)) / (2. * eps)
    return J


# ── Newton 求解器 ─────────────────────────────────────────────────────────────
def newton_solve(F, u0, tol=1e-9, maxiter=50, verbose=False, quiet=True):
    """quiet=True 时抑制 Ill-conditioned matrix 等 LinAlgWarning。"""
    u = u0.copy()
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    if quiet:
        warnings.filterwarnings('ignore', category=linalg.LinAlgWarning)
    try:
        for _ in range(maxiter):
            R  = F(u);  nr = np.max(np.abs(R))
            if nr < tol:
                return u, True, nr
            J  = _jacobian(F, u)
            try:
                du = linalg.solve(J, -R)
            except Exception:
                du, *_ = linalg.lstsq(J, -R)
            lam = 1.
            for _ in range(12):
                if np.max(np.abs(F(u + lam * du))) < nr * (1. - 1e-4 * lam):
                    break
                lam *= .5
            else:
                lam = 1e-3
            u = u + lam * du
        R = F(u);  nr = np.max(np.abs(R))
        return u, False, nr
    finally:
        ctx.__exit__(None, None, None)


# ── 重心插值 ──────────────────────────────────────────────────────────────────
def bary_interp(x_nodes, f_nodes, x_eval):
    N = len(x_nodes) - 1
    w = (-1.) ** np.arange(N + 1)
    w[0] *= .5;  w[N] *= .5
    x_eval = np.atleast_1d(x_eval)
    out = np.zeros(len(x_eval))
    for k, xk in enumerate(x_eval):
        d  = xk - x_nodes
        je = np.where(np.abs(d) < 1e-14)[0]
        if len(je):
            out[k] = f_nodes[je[0]]
        else:
            wd = w / d
            out[k] = np.dot(wd, f_nodes) / wd.sum()
    return out


# ── 解对象 ────────────────────────────────────────────────────────────────────
class EsGBSolution:
    """
    封装单个 EsGB 黑洞的数值解及物理量。

    属性
    ----
    rh, alpha, N       : 输入参数
    coupling           : 耦合函数对象
    beta               : 多项式耦合的 β 值（向后兼容，其他耦合为 None）
    converged          : Newton 是否收敛
    residual           : ||R||_inf
    M_ADM              : ADM 质量
    Q                  : 标量荷（近端外推）
    phi_h              : 视界标量场值
    r, gtt, grr, phi   : 精细插值曲线（对数均匀 r 点）
    r_schw, gtt_schw, grr_schw : 施瓦西参考曲线（同 r 网格）
    label              : 自动生成的图例标签
    """
    def __init__(self, rh, alpha, coupling, phi_h, M_ADM, Q,
                 r, gtt, grr, phi,
                 converged, residual, N,
                 x_nodes, D_mat, ht_nodes, Gt_nodes):
        self.rh        = rh
        self.alpha     = alpha
        self.coupling  = coupling
        self.beta      = coupling.beta if hasattr(coupling, 'beta') else None
        self.N         = N
        self.converged = converged
        self.residual  = residual
        self.M_ADM     = M_ADM
        self.Q         = Q
        self.phi_h     = phi_h
        self.r         = r
        self.gtt       = gtt
        self.grr       = grr
        self.phi       = phi
        # 施瓦西参考（同 r 网格）
        self.r_schw   = r
        self.gtt_schw = 1. - rh / r
        self.grr_schw = 1. / (1. - rh / r)
        # 自动标签
        self.label = (
            f'$\\alpha={alpha},\\,{coupling.label_str()}$'
            f'  ($\\phi_h={phi_h:.3f}$)'
        )
        # ── 谱插值数据 ────────────────────────────────────────────
        self._x_nodes   = x_nodes
        self._ht_nodes  = ht_nodes
        self._Gt_nodes  = Gt_nodes
        self._dht_nodes = D_mat @ ht_nodes
        self._dGt_nodes = D_mat @ Gt_nodes

    def __repr__(self):
        status = 'converged' if self.converged else 'NOT converged'
        return (f'EsGBSolution({status}, '
                f'alpha={self.alpha}, {self.coupling}, '
                f'phi_h={self.phi_h:.4f}, '
                f'M={self.M_ADM:.4f}, Q={self.Q:.4f}, '
                f'||R||={self.residual:.2e})')

    def _spec_eval(self, r, f_nodes):
        r_a  = np.atleast_1d(np.asarray(r, dtype=float))
        x_a  = np.clip(1. - 2. * self.rh / r_a,
                        self._x_nodes[-1], self._x_nodes[0])
        return bary_interp(self._x_nodes, f_nodes, x_a)

    def gtt_at(self, r):
        scalar = np.isscalar(r)
        r_a    = np.atleast_1d(np.asarray(r, dtype=float))
        x_a    = np.clip(1. - 2. * self.rh / r_a,
                         self._x_nodes[-1], self._x_nodes[0])
        ht     = bary_interp(self._x_nodes, self._ht_nodes, x_a)
        out    = ht * (1. + x_a) / 2.
        return float(out[0]) if scalar else out

    def grr_at(self, r):
        scalar = np.isscalar(r)
        r_a    = np.atleast_1d(np.asarray(r, dtype=float))
        x_a    = np.clip(1. - 2. * self.rh / r_a,
                         self._x_nodes[-1], self._x_nodes[0])
        Gt     = bary_interp(self._x_nodes, self._Gt_nodes, x_a)
        out    = Gt * 2. / (1. + x_a)
        return float(out[0]) if scalar else out

    def dgtt_dr(self, r):
        scalar = np.isscalar(r)
        r_a    = np.atleast_1d(np.asarray(r, dtype=float))
        x_a    = np.clip(1. - 2. * self.rh / r_a,
                         self._x_nodes[-1], self._x_nodes[0])
        ht     = bary_interp(self._x_nodes, self._ht_nodes,  x_a)
        dht    = bary_interp(self._x_nodes, self._dht_nodes, x_a)
        dxdr   = 2. * self.rh / r_a ** 2
        out    = (dht * (1. + x_a) / 2. + ht / 2.) * dxdr
        return float(out[0]) if scalar else out

    def dgrr_dr(self, r):
        scalar = np.isscalar(r)
        r_a    = np.atleast_1d(np.asarray(r, dtype=float))
        x_a    = np.clip(1. - 2. * self.rh / r_a,
                         self._x_nodes[-1], self._x_nodes[0])
        Gt     = bary_interp(self._x_nodes, self._Gt_nodes,  x_a)
        dGt    = bary_interp(self._x_nodes, self._dGt_nodes, x_a)
        dxdr   = 2. * self.rh / r_a ** 2
        out    = (dGt * 2. / (1. + x_a) - Gt * 2. / (1. + x_a) ** 2) * dxdr
        return float(out[0]) if scalar else out


# ── 主求解接口 ────────────────────────────────────────────────────────────────
def solve(rh=1.0, alpha=0.125, beta=-0.5, phi_h_init=0.55, N=13,
          coupling=None, r_max=1e4, n_pts=800, verbose=False):
    """
    求解 EsGB 标量化黑洞，返回 EsGBSolution 对象。

    参数
    ----
    rh          : 视界半径
    alpha       : Gauss-Bonnet 耦合强度
    beta        : φ⁴ 耦合系数（仅当 coupling=None 时使用，向后兼容）
    phi_h_init  : 初始猜测
    N           : Chebyshev 配点数
    coupling    : 耦合函数对象（Coupling 子类实例）。
                  若为 None，自动创建 PolynomialCoupling(beta)。
    r_max       : 绘图/插值最大 r
    n_pts       : 对数均匀插值点数
    verbose     : 是否打印 Newton 迭代过程

    返回
    ----
    EsGBSolution
    """
    if coupling is None:
        coupling = PolynomialCoupling(beta)

    x_nd, D, D2 = chebyshev_matrix(N)
    M = N + 1

    # 初始猜测
    u0 = np.zeros(3 * M)
    u0[:M]   = 1.;  u0[M:2*M] = 1.
    u0[2*M:] = phi_h_init * (1. - x_nd) / 2.

    def F(u):
        return build_residual(u, x_nd, D, D2, rh, alpha, coupling)

    u_sol, converged, residual = newton_solve(F, u0, verbose=verbose)

    # 提取物理量
    ht  = u_sol[:M];  Gt  = u_sol[M:2*M];  phi_nd = u_sol[2*M:]
    dhtx = D @ ht
    M_ADM = rh / 2. * (1. + 2. * dhtx[0])
    c_phi = np.mean([phi_nd[j] / (1. - x_nd[j]) for j in range(1, 4)])
    Q     = 2. * rh * c_phi
    phi_h = phi_nd[N]

    # 精细插值
    r_min_interp = rh * 1.001
    r_fine = np.logspace(np.log10(r_min_interp), np.log10(r_max), n_pts)
    x_fine = np.clip(1. - 2. * rh / r_fine, x_nd[-1], x_nd[0])

    ht_f   = bary_interp(x_nd, u_sol[:M],        x_fine)
    Gt_f   = bary_interp(x_nd, u_sol[M:2*M],     x_fine)
    phi_f  = bary_interp(x_nd, u_sol[2*M:3*M],   x_fine)

    gtt = ht_f  * (1. + x_fine) / 2.
    grr = Gt_f  * 2. / (1. + x_fine)

    if verbose:
        status = '[OK]' if converged else '[FAIL]'
        print(f'{status}  alpha={alpha}, {coupling}, '
              f'phi_h={phi_h:.5f}, M={M_ADM:.5f}, '
              f'Q={Q:.5f}, ||R||={residual:.2e}')

    return EsGBSolution(rh, alpha, coupling, phi_h, M_ADM, Q,
                        r_fine, gtt, grr, phi_f,
                        converged, residual, N,
                        x_nd, D, u_sol[:M], u_sol[M:2*M])


# ── φ_h 允许范围分析 ──────────────────────────────────────────────────────────
def phi_h_allowed_range(rh, alpha, coupling, verbose=True):
    """
    解析/数值给出视界正则性对 phi_h 的约束。
    返回 (phi_c1, phi_c2)：无禁戒时返回 (inf, inf)。
    """
    phi_c1, phi_c2 = coupling.phi_h_allowed_range(rh, alpha)

    if verbose:
        print(f'rh={rh}, alpha={alpha}, {coupling}')
        limit = rh ** 2 / (np.sqrt(96.) * alpha)
        print(f'  |f_dot| limit = {limit:.6f}')
        if np.isfinite(phi_c1):
            print(f'  -> 禁戒区间: ({phi_c1:.6f}, {phi_c2:.6f})')
            print(f'  -> 允许区间: (0, {phi_c1:.6f}) ∪ ({phi_c2:.6f}, +∞)')
        else:
            print(f'  -> phi_h 无上界')

    return phi_c1, phi_c2


# ── 分支扫描 ──────────────────────────────────────────────────────────────────
def scan_branches(rh, alpha, coupling, N=13, phi_scan=None, tol=1e-9):
    """
    扫描 phi_h_init，返回所有收敛的唯一解分支。

    返回
    ----
    list of dict: {'phi_init', 'phi_h', 'M_ADM', 'Q', 'residual'}
    """
    phi_c1, phi_c2 = coupling.phi_h_allowed_range(rh, alpha)

    if phi_scan is None:
        if np.isfinite(phi_c1):
            phi_scan = np.linspace(0.02, phi_c1 * 0.90, 18)
        else:
            if hasattr(coupling, 'beta') and coupling.beta < 0:
                phi_star = np.sqrt(-1. / (6. * coupling.beta))
            else:
                phi_star = 1.0
            phi_scan = np.linspace(0.02, phi_star * 1.5, 25)

    x_nd, D, D2 = chebyshev_matrix(N)
    M = N + 1

    def F(u):
        return build_residual(u, x_nd, D, D2, rh, alpha, coupling)

    results = []
    for phi0 in phi_scan:
        u = np.zeros(3 * M)
        u[:M] = 1.;  u[M:2*M] = 1.
        u[2*M:] = phi0 * (1. - x_nd) / 2.
        u_s, conv, res = newton_solve(F, u, tol=tol)
        if res < tol * 10:
            dhtx = D @ u_s[:M]
            M_ADM = rh / 2. * (1. + 2. * dhtx[0])
            c_phi = np.mean([u_s[2*M+j] / (1. - x_nd[j]) for j in range(1, 4)])
            Q     = 2. * rh * c_phi
            phi_h = u_s[3*M-1]
            # 去重
            if not any(abs(phi_h - r['phi_h']) < 0.02 for r in results):
                results.append({'phi_init': phi0, 'phi_h': phi_h,
                                 'M_ADM': M_ADM, 'Q': Q, 'residual': res})

    return results
