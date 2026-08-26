"""
NSGA-II 多目标优化示例（优化版）
目标：主切削力↓、最大主应力↓、切削温度↓
变量：前角、后角、刀尖钝圆半径
"""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.core.callback import Callback
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

 # 1. 全局设置
SEED = 42
np.random.seed(SEED)
# 自动找一个支持中文的字体（Linux/Win/mac 都适用）
def get_chinese_font():
    for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if any(key in fm.FontProperties(fname=font).get_name().lower()
               for key in ['simhei',
                           'simsun',
                           'microsoft yahei',
                           'wenquanyi',
                           'heiti',
                           'songti',
                           'fangsong']):
            return font
    return None

ch_font = get_chinese_font()
if ch_font:
    plt.rcParams['font.family'] = fm.FontProperties(fname=ch_font).get_name()
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # fallback
plt.rcParams['axes.unicode_minus'] = False

 # 2. 加载模型
bp_data = joblib.load('bp_fx_model.pkl')
svr_data = joblib.load('svr_fy_t_model.pkl')
# BP主切削力模型
bp_fx_model = bp_data["model"]
# SVR模型
svr_fy_model = svr_data["models"]["σ"]
svr_t_model = svr_data["models"]["T"]

 # 3. 定义优化问题
class MultiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=3,#输入变量
            n_obj=3,#优化目标数
            xl=np.array([5, 4, 0.01]),
            xu=np.array([20, 12, 0.08])
        )
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = bp_fx_model.predict(X)          # 主切削力
        f2 = svr_fy_model.predict(X)         # 背向力
        f3 = svr_t_model.predict(X)           # 切削温度
        out["F"] = np.column_stack([f1, f2, f3])
problem = MultiObjectiveProblem()

 # 4. 自定义HV回调
class HVCallback(Callback):
    def __init__(self, ref_point):
        super().__init__()
        self.hv = []
        self.ref_point = ref_point

    def notify(self, algorithm):
        F = algorithm.opt.get("F")
        hv = Hypervolume(ref_point=self.ref_point, normalize=False).do(F)
        self.hv.append(hv)

 # 5. 运行NSGA-Ⅱ
def run_optimization():
     algorithm = NSGA2(pop_size=100,crossover=SBX(prob=0.9,eta=15), mutation=PM(prob=0.2,eta=20))
     # 先用一代跑一个 dummy 结果，估算参考点
     dummy_res = minimize(problem,
                          algorithm,
                          ('n_gen', 20),
                          seed=SEED,
                          verbose=False)
     ref_point = np.max(dummy_res.F, axis=0) * 1.1
     # 重新跑正式优化
     hv_cb = HVCallback(ref_point)
     algorithm = NSGA2(pop_size=100, crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=0.2, eta=20))
     result = minimize(problem,
                       algorithm,
                       ('n_gen', 150),#最大迭代轮次
                       seed=SEED,
                       callback=hv_cb,
                       verbose=True,
                       save_history=True)
     return result, hv_cb
result, hv_cb = run_optimization()

 # 6. 整理Pareto解
pareto_designs = result.X #提取最优设计变量组合
pareto_objectives = result.F #提取最优解对应的目标函数值

df_pareto = pd.DataFrame(
    np.hstack([pareto_designs, pareto_objectives]),
    columns=['前角(°)', '后角(°)', '刀尖钝圆半径(mm)', '主切削力(N)',
             '背向力(N)', '切削温度(℃)']
).round(3) #保留三位小数
print('\n===== 部分 Pareto 前沿解 =====')
print(df_pareto.head(10)) #输出前10行最优解
df_pareto.to_csv('ParetoSolutions.csv', index=False)

 # 7. 推荐折中解（先归一化，再×权重）
weights = np.array([0.4,0.3,0.3])
# 各目标最小值
obj_min = pareto_objectives.min(axis=0)
# 各目标最大值
obj_max = pareto_objectives.max(axis=0)
# 归一化
norm_obj = (pareto_objectives - obj_min) / (obj_max - obj_min)
scores = np.dot(norm_obj, weights)
best_idx = np.argmin(scores)
rec = df_pareto.iloc[best_idx]
print('\n===== 推荐折中解 =====')
print(rec)

# 8. 可视化
def plot_2d_with_path(x_col, y_col, title):
    """
    2D Pareto 前沿 + 搜寻路径（每代采样点连线）
    x_col, y_col：0主切削力，1-背向力，2切削温度
    """
    plt.figure(figsize=(6, 4))

    # 1) 画最终 Pareto 前沿（蓝色）
    plt.scatter(result.F[:, x_col], result.F[:, y_col],
                c='tab:blue', s=25, label='Pareto 前沿', zorder=3)

    # 2) 画每一代的采样点作为“路径”
    #    algorithm.pop 在每代都会被覆盖，因此需要提前在每代 callback 里保存
    #    这里简单演示：用 result.history 里的信息
    path_x, path_y = [], []
    for algo in result.history:
        pop_F = algo.pop.get("F")
        path_x.append(pop_F[:, x_col])
        path_y.append(pop_F[:, y_col])

    # 把所有采样点扁平化后画灰色细线
    all_x = np.concatenate(path_x)
    all_y = np.concatenate(path_y)
    plt.scatter(all_x, all_y, c='lightgray', s=1, alpha=0.3, label='搜索路径')

    plt.xlabel(['主切削力(N)', '背向力(N)', '切削温度(℃)'][x_col])
    plt.ylabel(['主切削力(N)', '背向力(N)', '切削温度(℃)'][y_col])
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_3d_with_path():
    """
    3D Pareto 前沿 + 搜寻路径
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Pareto 前沿
    ax.scatter(pareto_objectives[:, 0],
               pareto_objectives[:, 1],
               pareto_objectives[:, 2],
               c='tab:blue', s=25, label='Pareto 前沿')

    # 搜索路径
    for algo in result.history:
        pop_F = algo.pop.get("F")
        ax.scatter(pop_F[:, 0],
                   pop_F[:, 1],
                   pop_F[:, 2],
                   c='lightgray', s=1, alpha=0.2)

    ax.set_xlabel('主切削力(N)')
    ax.set_ylabel('背向力(N)')
    ax.set_zlabel('切削温度(℃)')
    ax.set_title('Pareto 前沿 + 搜索路径')
    plt.tight_layout()
    plt.show()

def plot_2d_scatter_with_color():
    """
    用圆点散点图替代原来的 histogram2d 热图
    x: 主切削力
    y: 背向力
    颜色: 切削温度
    """
    x = pareto_objectives[:, 0]
    y = pareto_objectives[:, 1]
    z = pareto_objectives[:, 2]

    plt.figure(figsize=(5.5, 4.5))
    sc = plt.scatter(x, y, c=z, cmap='viridis', s=25, edgecolors='k', linewidths=0.3)
    cb = plt.colorbar(sc, label='切削温度(℃)')
    plt.xlabel('主切削力(N)')
    plt.ylabel('背向力(N)')
    #plt.title('Pareto 前沿二维散点图')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# 执行绘图
plot_2d_with_path(0, 1, "Pareto 前沿 + 搜寻路径：主切削力 vs 背向力")
plot_2d_with_path(0, 2, "Pareto 前沿 + 搜寻路径：主切削力 vs 切削温度")
plot_3d_with_path()
#plot_hv_curve()
plot_2d_scatter_with_color()