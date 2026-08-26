import joblib
import numpy as np

# 1. 加载模型文件
bp_data = joblib.load('bp_f_model.pkl')
svr_data = joblib.load('svr_t_model.pkl')

# 2. 提取真正模型
# BP切削力模型
bp_model = bp_data["model_F"]
# SVR温度模型
svr_model = svr_data["models"]["T"]

# 3. 输入刀具参数
# 前角(°)
rake_angle = 19
# 后角(°)
clearance_angle = 4.6
# 刀尖钝圆半径(mm)
edge_radius = 0.02

# 4. 构造输入
X_input = np.array([
    [rake_angle, clearance_angle, edge_radius]
])

# 5. 模型预测
# BP预测切削力
force_pred = bp_model.predict(X_input)[0]
# SVR预测切削温度
temp_pred = svr_model.predict(X_input)[0]

# 6. 输出结果

print("========== 预测结果 ==========")

print(f"前角: {rake_angle} °")
print(f"后角: {clearance_angle} °")
print(f"刀尖钝圆半径: {edge_radius} mm")

print("--------------------------------")

print(f"BP预测切削力 F: {force_pred:.4f}")

print(f"SVR预测切削温度 T: {temp_pred:.4f}")