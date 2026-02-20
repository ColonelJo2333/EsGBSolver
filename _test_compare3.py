import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import importlib, esgb_solver_core as core; importlib.reload(core)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset' : 'stix',
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 7.5,
    'axes.linewidth'   : 0.8,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction'  : 'in',
    'ytick.direction'  : 'in',
    'xtick.top'        : True,
    'ytick.right'      : True,
    'lines.linewidth'  : 1.2,
})

# ══════════════════════════════════════════════════════════════════
CASES = [
    dict(alpha=0.125, beta=-0.5, phi_h_init=0.55),
    dict(alpha=0.125, beta=-0.6, phi_h_init=0.55),
    dict(alpha=0.1,   beta=-0.5, phi_h_init=0.55),
]
rh_global, N_global = 1.0, 13
# ══════════════════════════════════════════════════════════════════

_SOL_STYLES = [
    dict(color='#0072B2', lw=2.8, linestyle='-'),
    dict(color='#D55E00', lw=2.8, linestyle=(0, (6, 2, 1.5, 2))),
    dict(color='#009E73', lw=2.8, linestyle=(0, (2, 1.3))),
]
_SCHW_STYLE = dict(color='#4D4D4D', lw=2.4, linestyle=(0, (7, 3)), alpha=0.9)
_LW_REF = _SCHW_STYLE['lw']

def _ls(i):
    return dict(_SOL_STYLES[i % len(_SOL_STYLES)])

def _ls_schw(i):
    return dict(_SCHW_STYLE)

# ── 求解 + Z₂ 翻转 ────────────────────────────────────────────────
solutions = []
for kw in CASES:
    sol = core.solve(rh=rh_global, N=N_global, verbose=True, **kw)
    if sol.phi_h < 0:           # Z₂: phi -> -phi 是同一物理解，统一取正
        sol.phi   = -sol.phi
        sol.Q     = -sol.Q
        sol.phi_h = -sol.phi_h
        sol.label = (f'$\\alpha={sol.alpha},\\,\\beta={sol.beta}$'
                     f'  ($\\phi_h={sol.phi_h:.3f}$)')
    solutions.append(sol)
print()

# ── 坐标轴基础格式（log x） ───────────────────────────────────────
def _setup_ax(ax, logy=False):
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: (f'{v:.3g}' if v < 10 else
                      f'{int(v)}' if v < 1000 else
                      f'$10^{{{int(np.log10(v))}}}$')))
    ax.set_xlabel(r'$r/r_h$', fontsize=10)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=4)
    ax.tick_params(which='minor', length=2)
    ax.grid(True, which='major', lw=0.3,  color='#bbbbbb', zorder=0)
    ax.grid(True, which='minor', lw=0.15, color='#dddddd', zorder=0)

# ── inset 放大图 ──────────────────────────────────────────────────
def _add_inset(ax, solutions, key, x1=3., x2=15.,
               bbox_loc=(0.02, 0.02), logy=False):
    """
    在 ax 中插入 [x1,x2] 区间放大图。
    bbox_loc: inset 框左下角在 axes fraction 坐标系中的位置
    logy    : 是否对 y 轴也取 log（用于 grr 双对数 inset）
    """
    axins = ax.inset_axes([bbox_loc[0], bbox_loc[1], 0.40, 0.38])

    schw_done = set()
    for i, sol in enumerate(solutions):
        mask = (sol.r >= x1) & (sol.r <= x2)
        r_m  = sol.r[mask]
        y    = {'gtt': sol.gtt, 'grr': sol.grr, 'phi': sol.phi}[key][mask]
        axins.plot(r_m, y, **_ls(i), zorder=3)
        rh_k = round(sol.rh, 6)
        if rh_k not in schw_done:
            if key == 'phi':
                axins.axhline(0., **_ls_schw(i), zorder=2)
            else:
                ys = {'gtt': sol.gtt_schw,
                      'grr': sol.grr_schw}[key][mask]
                axins.plot(r_m, ys, **_ls_schw(i), zorder=2)
            schw_done.add(rh_k)

    axins.set_xscale('log')
    if logy:
        axins.set_yscale('log')
    axins.set_xlim(x1, x2)
    axins.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f'{int(v)}' if v >= 1 else f'{v:.1g}'))
    axins.xaxis.set_major_locator(ticker.LogLocator(subs=[1, 2, 3, 5]))
    axins.xaxis.set_minor_locator(ticker.NullLocator())
    axins.tick_params(which='both', direction='in', labelsize=6.5,
                      length=2.5, top=True, right=True)
    axins.grid(True, which='major', lw=0.25, color='#cccccc', zorder=0)

    # y 范围：只取 EsGB 解（不含 Schwarzschild，避免 grr 发散污染）
    ys_esgb = []
    for sol in solutions:
        mask = (sol.r >= x1) & (sol.r <= x2)
        ys_esgb.extend(
            {'gtt': sol.gtt, 'grr': sol.grr, 'phi': sol.phi}[key][mask].tolist())
    ylo, yhi = min(ys_esgb), max(ys_esgb)
    if logy:
        # log y：用乘法 padding
        pad_lo = ylo / 1.15
        pad_hi = yhi * 1.15
        axins.set_ylim(max(pad_lo, 1e-6), pad_hi)
    else:
        pad = max((yhi - ylo) * 0.18, 1e-3)
        axins.set_ylim(ylo - pad, yhi + pad)

    # 标注放大区域（细实线矩形 + 连接线）
    ax.indicate_inset_zoom(axins, edgecolor='#777777', lw=0.7, alpha=0.8)

    # inset 边框
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor('#777777')

    return axins

# ── 主图 ──────────────────────────────────────────────────────────
rh_val = solutions[0].rh
rmin   = rh_val * 1.04      # 曲线起点（略高于视界）
rmax   = 5e3                # 横轴右端
xleft  = rh_val * 0.75      # 横轴左端（视界左侧留白）

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)

# ── PANEL 配置 ────────────────────────────────────────────────────
# gtt：图例在右下角（约占 axes 右侧 0-40% 高度）
#       → inset 紧贴图例正上方，放在右侧：x=0.56, y=0.42, w=0.40, h=0.38
# grr：图例在右上角（约占 axes 右侧 60-100% 高度）
#       → inset 紧贴图例正下方，放在右侧：x=0.56, y=0.04, w=0.40, h=0.38
#       → 双对数坐标
# phi：图例在右上角，inset 在左下区域（phi 在大 r 端趋 0，左下空白多）
PANEL_CFG = [
    dict(key='gtt',
         ylabel=r'$-g_{tt}(r)$', title=r'(a) $-g_{tt}$',
         legend_loc='lower right',
         inset_bbox=(0.56, 0.42),   # 图例正上方（图例在右下，inset 紧贴其上）
         inset_x1=3., inset_x2=15.,
         logy=False,
         ylim_bottom=0.0),          # gtt 从 0 开始
    dict(key='grr',
         ylabel=r'$g_{rr}(r)$',  title=r'(b) $g_{rr}$',
         legend_loc='upper right',
         inset_bbox=(0.57, 0.02),   # 图例在右上 → inset 放右侧底部（图例正下方）
         inset_x1=3., inset_x2=8.,
         logy=False,                # 线性 y 轴
         ylim_bottom=None),         # grr 不限制 bottom（log 轴自动）
    dict(key='phi',
         ylabel=r'$\phi(r)$',     title=r'(c) Scalar field $\phi$',
         legend_loc='upper right',
         inset_bbox=(0.57, 0.02),   # 图例在右上 → inset 放右侧底部（图例正下方）
         inset_x1=3., inset_x2=15.,
         logy=False,
         ylim_bottom=None),
]

for ax, cfg in zip(axes, PANEL_CFG):
    key  = cfg['key']
    logy = cfg['logy']
    schw_drawn = set()

    for i, sol in enumerate(solutions):
        mask  = sol.r >= rmin
        r_m   = sol.r[mask]
        y_sol = {'gtt': sol.gtt, 'grr': sol.grr, 'phi': sol.phi}[key][mask]
        ax.plot(r_m, y_sol, label=sol.label, **_ls(i), zorder=3)

        rh_key = round(sol.rh, 6)
        if rh_key not in schw_drawn:
            if key == 'phi':
                ax.axhline(0., **_ls_schw(i), zorder=2,
                           label='Schwarzschild' if i == 0 else '_')
            else:
                y_schw = {'gtt': sol.gtt_schw,
                          'grr': sol.grr_schw}[key][mask]
                ax.plot(r_m, y_schw,
                        label='Schwarzschild' if i == 0 else '_',
                        **_ls_schw(i), zorder=2)
            schw_drawn.add(rh_key)

    # 视界竖线
    ax.axvline(rh_val, lw=0.5, color='#bbbbbb', ls=':', zorder=1)

    _setup_ax(ax, logy=logy)
    ax.set_xlim(xleft, rmax)
    ax.set_ylabel(cfg['ylabel'], fontsize=10)
    ax.set_title(cfg['title'], fontsize=9, pad=3)

    # y 轴范围
    if key == 'grr':
        # grr 线性坐标：Schwarzschild 在近视界发散，用 EsGB 解范围限定 y 轴
        mask_all = solutions[0].r >= rmin
        grr_vals = np.concatenate([s.grr[s.r >= rmin] for s in solutions])
        yhi_grr  = grr_vals.max() * 1.08
        ax.set_ylim(bottom=1.0, top=yhi_grr)
    elif cfg['ylim_bottom'] is not None:
        ax.set_ylim(bottom=cfg['ylim_bottom'])

    # 图例
    leg = ax.legend(loc=cfg['legend_loc'], framealpha=0.92,
                    edgecolor='#cccccc', handlelength=2.5,
                    borderpad=0.7, labelspacing=0.38, handletextpad=0.5,
                    fontsize=7.2)
    leg.get_frame().set_linewidth(0.6)

    # 放大 inset
    _add_inset(ax, solutions, key,
               x1=cfg['inset_x1'], x2=cfg['inset_x2'],
               bbox_loc=cfg['inset_bbox'],
               logy=logy)

plt.savefig('esgb_comparison.pdf')
plt.savefig('esgb_comparison.png', dpi=150)
print('Saved OK.')
