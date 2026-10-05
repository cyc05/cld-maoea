# -*- coding: utf-8 -*-
"""用 matplotlib 程序化重绘 Figure 1（CLD-MaOEA 框架图），v2：消除重叠与交叉。
上带在线主循环（6 框 + 底部回流 + 虚线 archive），下带离线准则链（S1→S2→S3 + 三条竖直虚线配置覆盖）。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(13.4, 4.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 35)
ax.axis('off')

C_EDGE = '#222222'
C_LOOP = '#f4f7fb'
C_CHAIN = '#dce9f7'
C_BLUE = '#3a6ea5'


def band(x, y, w, h, label, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.55,rounding_size=1.2',
                                fc=fc, ec=C_EDGE, lw=1.1))
    ax.text(x + 1.2, y + h - 0.3, label, ha='left', va='top', fontsize=10.6,
            style='italic', color='#333333')


def box(x, y, w, h, title, rest=None, fc='white', dashed=False, fs=10.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=0.8',
                                fc=fc, ec=C_EDGE, lw=1.1,
                                linestyle=(0, (4, 2)) if dashed else 'solid'))
    if rest:
        n_title = title.count('\n') + 1
        ax.text(x + w / 2, y + h - 0.5, title, ha='center', va='top', fontsize=fs,
                fontweight='bold', linespacing=1.3)
        ax.text(x + w / 2, y + h - 0.5 - 2.5 * n_title, rest, ha='center', va='top',
                fontsize=fs - 0.7, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs,
                linespacing=1.3)


def arrow(x1, y1, x2, y2, dashed=False, color=C_EDGE, conn='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11,
                                 lw=1.1 if not dashed else 1.0,
                                 linestyle=(0, (4, 2)) if dashed else 'solid',
                                 color=color, connectionstyle=conn, shrinkA=1.2, shrinkB=1.2))


def seg(points, dashed=False, color=C_EDGE):
    """折线 + 末端箭头。"""
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]],
                color=color, lw=1.1 if not dashed else 1.0,
                linestyle=(0, (4, 2)) if dashed else 'solid')
    arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], dashed=dashed, color=color)


# ---------- 上带：在线主循环 ----------
band(2, 14.0, 96, 20.0, 'Online loop: CLD-MaOEA', C_LOOP)
BY, BH = 17.5, 9.0
boxes = [
    (3.5, 13.0, 'Initialize population $P_0$', 'normalize objectives;\nreference directions $W$'),
    (18.5, 16.0, 'M1: Closed-loop\ndominance control', 'rank-1 share $\\rho$ → PD controller\n→ dominance strength $\\phi$'),
    (36.5, 15.0, 'M2: Directional\ndifferentiation mating', 'per-direction strength $\\phi_x$\nfrom convergence lag'),
    (53.5, 13.0, 'Offspring: SBX\ncrossover + mutation', '(escape: larger step)'),
    (68.5, 14.5, 'M3: Environmental selection', 'Pareto fronts + R2-greedy\ntruncation, niche protection'),
]
centers = []
for x, w, t, r in boxes:
    box(x, BY, w, BH, t, r)
    centers.append((x, x + w, x + w / 2))

OUT_X, OUT_W, OUT_Y, OUT_H = 91.0, 6.2, BY + 1.8, 5.4
box(OUT_X, OUT_Y, OUT_W, OUT_H, 'Output\npopulation', fs=10.0)

for i in range(len(boxes) - 1):
    arrow(centers[i][1], BY + BH / 2, centers[i + 1][0], BY + BH / 2)
arrow(centers[-1][1], BY + BH / 2, OUT_X, BY + BH / 2)
# 回流：Output 底部 → 下方通道 → Initialize 底部
seg([(OUT_X + OUT_W / 2, OUT_Y), (OUT_X + OUT_W / 2, 15.6), (10.0, 15.6), (10.0, BY)])
ax.text(50, 15.0, 'next generation', ha='center', va='top', fontsize=9.0, style='italic',
        color='#333333', bbox=dict(fc=C_LOOP, ec='none', pad=0.5))

# ---------- corner archive（虚线，M3 上方） ----------
AR_X, AR_W, AR_Y, AR_H = 66.0, 24.0, 29.0, 4.0
box(AR_X, AR_Y, AR_W, AR_H, 'Corner archive (passive, read-only)', 'merge at output: union / replace',
    fc='#fdfdfd', dashed=True, fs=9.2)
arrow(75.7, BY + BH, 75.7, AR_Y, dashed=True)
ax.text(74.6, 27.9, 'observe', ha='right', va='center', fontsize=8.8, style='italic',
        color='#333333', bbox=dict(fc=C_LOOP, ec='none', pad=0.4))
seg([(AR_X + AR_W, AR_Y + AR_H / 2), (96.5, AR_Y + AR_H / 2), (96.5, OUT_Y + OUT_H)], dashed=True)

# ---------- 下带：离线准则链 ----------
band(2, 1.5, 96, 10.0, 'Offline criterion chain (Figure 3)', C_CHAIN)
s_y, s_h = 3.4, 4.8
box(6.0, s_y, 24.0, s_h, 'S1  Union probe', 'passive archive, 3 seeds;\nclassify gap: generative / selective', fs=9.4)
box(38.0, s_y, 24.0, s_h, 'S2  Three-condition validation', 'paired significance; ≥8/10 seeds;\nworst-seed guard', fs=9.4)
box(70.0, s_y, 24.0, s_h, 'S3  Regression gate', 'zero regression → commit;\nlog decision (audit trail)', fs=9.4)
arrow(30.0, s_y + s_h / 2, 38.0, s_y + s_h / 2)
arrow(62.0, s_y + s_h / 2, 70.0, s_y + s_h / 2)

# 三条竖直配置覆盖虚线：准则链顶 → M1 底 / M3 底 / archive 底（走 M3 与 Output 之间的空巷 x=87）
for tx, ty in [(26.5, BY), (75.7, BY), (87.0, AR_Y)]:
    arrow(tx, 12.3, tx, ty, dashed=True, color=C_BLUE)
ax.text(50.0, 13.2, 'per-instance on/off configuration overrides (auditable)',
        ha='center', va='center', fontsize=9.0, style='italic', color=C_BLUE,
        bbox=dict(fc='#eef4fa', ec='none', pad=1.0))

# 图例
leg = [Line2D([0], [0], color=C_EDGE, lw=1.1, label='evolutionary loop'),
       Line2D([0], [0], color=C_EDGE, lw=1.0, linestyle=(0, (4, 2)), label='archive observe / merge'),
       Line2D([0], [0], color=C_BLUE, lw=1.0, linestyle=(0, (4, 2)), label='configuration override')]
ax.legend(handles=leg, loc='lower left', bbox_to_anchor=(0.003, -0.03), fontsize=8.4,
          frameon=False, ncol=3, handlelength=2.2, columnspacing=1.4)

plt.subplots_adjust(left=0.004, right=0.996, top=0.997, bottom=0.04)
out = os.path.join(ROOT, 'paper', 'figures_spie', 'fig1_framework.png')
plt.savefig(out, dpi=300)
print('saved', out)