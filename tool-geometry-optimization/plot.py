import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib

# =========================================================
# 论文绘图字体与全局风格设置
# （严格按照示例图风格）
# =========================================================

# 中文：宋体
plt.rcParams['font.sans-serif'] = ['SimSun']
# 负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 全局风格
plt.rcParams.update({
    # 字体
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    # 坐标轴线宽
    'axes.linewidth': 1.0,

    # 曲线线宽
    'lines.linewidth': 2.0,

    # 图像分辨率
    'figure.dpi': 300,
    'savefig.dpi': 600,

    # 网格
    'grid.linestyle': '--',
    'grid.alpha': 0.35,

    # 图例
    'legend.frameon': True,
    'legend.edgecolor': 'black',

    # 导出边距
    'savefig.bbox': 'tight'
})

# =========================================================
# 加载模型
# =========================================================
bp_data = joblib.load('bp_f_model.pkl')
svr_data = joblib.load('svr_t_model.pkl')

bp_model = bp_data['model_F']
svr_model = svr_data['models']['T']

# =========================================================
# 参数范围与固定值
# =========================================================
gamma_range = np.linspace(5, 20, 100)
alpha_range = np.linspace(4, 16, 100)
r_range = np.linspace(0.01, 0.08, 100)

gamma_fixed = 10
alpha_fixed = 8
r_fixed = 0.04

# =========================================================
# 计算单因素响应
# =========================================================
F_gamma, T_gamma = [], []
F_alpha, T_alpha = [], []
F_r, T_r = [], []

# γ → F/T
for g in gamma_range:
    x = np.array([[g, alpha_fixed, r_fixed]])
    F_gamma.append(bp_model.predict(x)[0])
    T_gamma.append(svr_model.predict(x)[0])

# α → F/T
for a in alpha_range:
    x = np.array([[gamma_fixed, a, r_fixed]])
    F_alpha.append(bp_model.predict(x)[0])
    T_alpha.append(svr_model.predict(x)[0])

# r → F/T
for r in r_range:
    x = np.array([[gamma_fixed, alpha_fixed, r]])
    F_r.append(bp_model.predict(x)[0])
    T_r.append(svr_model.predict(x)[0])

# 固定坐标范围
F_ylim = [400, 500]
T_ylim = [850, 1050]

# 颜色
force_color = '#003CB3'  # 蓝色
temp_color = '#D60000'  # 红色


# =========================================================
# 双纵轴单因素响应曲线
# =========================================================
def draw_dual_curve(x, F, T, xlabel, subtitle, save_name):
    fig, ax1 = plt.subplots(figsize=(6.0, 4.8))

    # 【优化】为底部的 figtext 图题预留出空间，防止导出时被切字
    fig.subplots_adjust(bottom=0.18)

    # 左轴：切削力
    line1, = ax1.plot(x, F, color=force_color, linewidth=2.0, label='切削力')
    ax1.set_xlabel(xlabel, fontsize=13)
    ax1.set_ylabel('切削力 $F$ (N)', color=force_color, fontsize=13)
    ax1.tick_params(axis='y', colors=force_color)
    ax1.set_ylim(F_ylim)
    ax1.grid(True)

    # 右轴：切削温度
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, T, color=temp_color, linewidth=2.0, linestyle='--', dashes=(4, 2), label='切削温度')
    ax2.set_ylabel('切削温度 $T$ (°C)', color=temp_color, fontsize=13)
    ax2.tick_params(axis='y', colors=temp_color)
    ax2.set_ylim(T_ylim)

    # 图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11, framealpha=1.0, fancybox=False)

    # 【优化】y 从 -0.02 移到 0.03（确保在画布内），配合上面的 subplots_adjust 完美显示
    plt.figtext(0.5, 0.03, subtitle, ha='center', fontsize=12)

    plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.show()


# =========================================================
# 绘制三张单因素响应曲线
# =========================================================
draw_dual_curve(gamma_range, F_gamma, T_gamma, '前角 $\\gamma$ (°)', '图1 前角响应曲线', '前角响应曲线.tif')
draw_dual_curve(alpha_range, F_alpha, T_alpha, '后角 $\\alpha$ (°)', '图2 后角响应曲线', '后角响应曲线.tif')
draw_dual_curve(r_range, F_r, T_r, '刀尖钝圆半径 $r$ (mm)', '图3 刀尖钝圆半径响应曲线', '刀尖钝圆半径响应曲线.tif')


# =========================================================
# 3D响应面函数
# =========================================================
def draw_surface(X, Y, Z, xlabel, ylabel, zlabel, subtitle, zlim, save_name):
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection='3d')

    # 仅 XY 投影面为灰色
    gray = (0.45, 0.45, 0.45, 1.0)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 1.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1.0))
    ax.zaxis.pane.set_facecolor(gray)

    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    # 响应面
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k', linewidth=0.35, antialiased=True, alpha=0.97)

    # 底部等高线
    ax.contour(X, Y, Z, zdir='z', offset=zlim[0], cmap='jet', levels=8, linewidths=1.0)

    # 坐标轴与标签
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    # 彻底关闭内置机制，并应用旋转
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, fontsize=13, labelpad=8)
    ax.zaxis.label.set_rotation(90)
    ax.set_zlim(zlim)

    # 视角
    ax.view_init(elev=24, azim=-132)

    # colorbar
    cbar = fig.colorbar(surf, shrink=0.72, aspect=18, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # 子图布局与边距调整
    plt.subplots_adjust(
        left=0.15,  # 给左侧留出足够空间放置逆时针旋转 90° 后的 Z 轴标签
        right=0.88,
        bottom=0.15,  # 增大底部留白，给图题让路
        top=0.96
    )

    # 子图标题位置微调
    plt.figtext(0.5, 0.04, subtitle, ha='center', fontsize=12)

    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    plt.show()


# =========================================================
# γ-α-切削力响应面
# =========================================================
G, A = np.meshgrid(gamma_range, alpha_range)
X_pred = np.column_stack([G.ravel(), A.ravel(), np.full_like(G.ravel(), r_fixed)])

F_pred = bp_model.predict(X_pred).reshape(G.shape)

# 【优化】删除了未在函数体内使用的 '切削力响应面' 字符串参数，防止接口错位
draw_surface(
    G, A, F_pred,
    '前角 $\\gamma$ (°)', '后角 $\\alpha$ (°)', '切削力 $F$ (N)',
    '图4 切削力响应面（$\\gamma-\\alpha$ 平面，$r=0.05$ mm）',
    [400, 500], '切削力响应面.tif'
)

# =========================================================
# γ-α-切削温度响应面
# =========================================================
T_pred = svr_model.predict(X_pred).reshape(G.shape)

# 【优化】同上，保持传参和定义严格一致
draw_surface(
    G, A, T_pred,
    '前角 $\\gamma$ (°)', '后角 $\\alpha$ (°)', '切削温度 $T$ (°C)',
    '图5 切削温度响应面（$\\gamma-\\alpha$ 平面，$r=0.05$ mm）',
    [850, 1050], '切削温度响应面.tif'
)