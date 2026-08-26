import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

# =========================================================
# 全局论文绘图风格
# =========================================================
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({

    # 字体
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    # 坐标轴
    'axes.linewidth': 1.0,

    # 图像分辨率
    'figure.dpi': 300,
    'savefig.dpi': 600,

    # 网格
    'grid.linestyle': '--',
    'grid.alpha': 0.35,

    # 边距
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
# 构造分析样本
# =========================================================
np.random.seed(42)

n_samples = 120

gamma = np.random.uniform(5, 20, n_samples)
alpha = np.random.uniform(4, 16, n_samples)
r = np.random.uniform(0.01, 0.08, n_samples)

X = np.column_stack([gamma, alpha, r])

feature_names = [
    '前角',
    '后角',
    '刀尖钝圆半径'
]

X_df = pd.DataFrame(X, columns=feature_names)

# =========================================================
# SHAP背景数据
# =========================================================
background = shap.sample(X_df, 40)

# =========================================================
# BP模型（切削力）SHAP分析
# =========================================================
print('开始计算切削力 SHAP值...')

explainer_F = shap.KernelExplainer(
    bp_model.predict,
    background
)

shap_values_F = explainer_F.shap_values(X_df)

# =========================================================
# 切削力 SHAP条形图（论文风格）
# =========================================================

# 平均SHAP值
importance_F = np.mean(np.abs(shap_values_F), axis=0)

# 排序
idx_F = np.argsort(importance_F)

features_sorted_F = np.array(feature_names)[idx_F]
importance_sorted_F = importance_F[idx_F]

# 蓝色渐变
colors_F = [
    '#DCE8FF',
    '#7FA6FF',
    '#003CB3'
]

fig, ax = plt.subplots(figsize=(6.4, 4.8))

bars = ax.barh(
    features_sorted_F,
    importance_sorted_F,
    height=0.5,
    color=colors_F,
    edgecolor='black',
    linewidth=0.8
)

# 数值标注
for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.002,
        bar.get_y() + bar.get_height()/2,
        f'{width:.3f}',
        va='center',
        fontsize=10
    )

# 标签与标题
ax.set_xlabel('平均SHAP值（切削力 / N）')
# 网格
ax.grid(axis='x')

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()

plt.savefig(
    '切削力_SHAP条形图.tif',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.15
)

plt.show()

# =========================================================
# 切削力 SHAP分布图
# =========================================================
plt.figure(figsize=(6.4, 4.8))

shap.summary_plot(
    shap_values_F,
    X_df,
    show=False
)

plt.xlabel('SHAP值（切削力 / N）', fontsize=13)


plt.tight_layout()

plt.savefig(
    '切削力_SHAP分布图.tif',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.15
)

plt.show()

# =========================================================
# SVR模型（切削温度）SHAP分析
# =========================================================
print('开始计算切削温度 SHAP值...')

explainer_T = shap.KernelExplainer(
    svr_model.predict,
    background
)

shap_values_T = explainer_T.shap_values(X_df)

# =========================================================
# 切削温度 SHAP条形图（论文风格）
# =========================================================

importance_T = np.mean(np.abs(shap_values_T), axis=0)

idx_T = np.argsort(importance_T)

features_sorted_T = np.array(feature_names)[idx_T]
importance_sorted_T = importance_T[idx_T]


fig, ax = plt.subplots(figsize=(6.4, 4.8))
# 蓝色渐变
colors_T = [
    '#DCE8FF',
    '#7FA6FF',
    '#003CB3'
]
bars = ax.barh(
    features_sorted_T,
    importance_sorted_T,
    height=0.5,
    color=colors_T,
    edgecolor='black',
    linewidth=0.8
)

# 数值标注
for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.002,
        bar.get_y() + bar.get_height()/2,
        f'{width:.3f}',
        va='center',
        fontsize=10
    )

# 标签与标题
ax.set_xlabel('平均SHAP值（切削温度 / ℃）')

# 网格
ax.grid(axis='x')

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()

plt.savefig(
    '切削温度_SHAP条形图.tif',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.15
)

plt.show()

# =========================================================
# 切削温度 SHAP分布图
# =========================================================
plt.figure(figsize=(6.4, 4.8))

shap.summary_plot(
    shap_values_T,
    X_df,
    show=False
)

plt.xlabel('SHAP值（切削温度 / ℃）', fontsize=13)

plt.tight_layout()

plt.savefig(
    '切削温度_SHAP分布图.tif',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.15
)

plt.show()

print('SHAP分析完成！')
