'''
1.基于XGBoost和optuna完成最优超参数确定
2.利用得到的最终超参数完成XGBoost模型建立
3.利用得到的XGBoost模型和SHAP完成特征重要性排序
'''

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
import joblib
from data_process import preprocess_data
from time_feature import create_delta_si
from SHAP_xgboost import shap_feature_importance

# 1.全局设置
SEED = 42
np.random.seed(SEED)

# 2.数据读取
data = pd.read_csv(
    r"C:\Users\liuhuiru\Desktop\Si_model\datas\traindata.csv",
     index_col='time',
     encoding="gbk")
# 转成真正时间
data.index = pd.to_datetime(data.index)
# 时间排序
data = data.sort_index()
# 删除重复时间
data = data[~data.index.duplicated(keep='first')]

# 划分训练集、验证集
train_size = int(len(data)*0.8)
#print(train_size)
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:]
# 训练集
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_train = preprocess_data(X_train,scaler_path=None,scaler_save_path="scaler_xgb.pkl")# 预处理

#print("X_train:",X_train.shape)
remove_time = pd.to_datetime([
    "2020-7-21 0:02:00",
    "2020-7-21 0:31:00",
    "2020-7-21 0:56:00"
])
X_train = X_train.drop(index=remove_time,errors="ignore")

y_train = create_delta_si(y_train)
y_train = y_train.loc[X_train.index]
#print("X_train:",X_train.shape)
#print("y_train:",y_train.shape)
print("index是否一致:",X_train.index.equals(y_train.index))
# 验证集
X_val = val_data.drop(columns=['label'])
y_val_true = val_data['label']
X_val = preprocess_data(X_val,scaler_path="scaler_xgb.pkl",scaler_save_path=None)
y_val_true = y_val_true.loc[X_val.index]
last_si_val = data['label'].shift(1)
last_si_val = last_si_val.loc[X_val.index]


# 3.定义optuna优化
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators",350, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 4, 8),
        "gamma": trial.suggest_float("gamma", 0.05, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    delta_pred = model.predict(X_val)
    y_pred = delta_pred + last_si_val.values
    mse =mean_squared_error(y_val_true, y_pred)
    return mse

# 4.开始optuna优化
study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50)
best_params = study.best_params
# 保存最佳超参数
joblib.dump( best_params,"best_xgboost_params.pkl")
print(best_params)

# 5.使用最优超参数重新训练模型
data_all = pd.read_csv(
    r"C:\Users\liuhuiru\Desktop\Si_model\datas\traindata.csv",
     index_col='time',
     encoding="gbk")
data_all.index = pd.to_datetime(data_all.index)
data_all = data_all.sort_index()
data_all = data_all[~data_all.index.duplicated(keep='first')]

x_all = data_all.drop(columns=['label'])
y_all = data_all['label']
x_all = preprocess_data(x_all,scaler_path=None,scaler_save_path="scaler_xgb_shap.pkl")
x_all = x_all.drop(index=remove_time,errors="ignore")

y_all = create_delta_si(y_all)
y_all = y_all.loc[x_all.index]
final_model = xgb.XGBRegressor(**best_params,objective="reg:squarederror",random_state=SEED,n_jobs=-1)
final_model.fit(x_all, y_all)
joblib.dump( final_model,"shap_xgboost.pkl")
joblib.dump( x_all.columns.tolist(),"xgb_features.pkl")
print("模型保存完成")
importance_df,top_features = shap_feature_importance(
    model=final_model,
    X=x_all,
    top_n=30,
    save_path="XGBoost_SHAP_top30.tif"
)
# 保存完整SHAP排序
joblib.dump(importance_df,"shap_feature_importance.pkl")
# 保存Top30特征名称
joblib.dump(top_features,"shap_top30_features.pkl")
print("Top30特征:")
for i, f in enumerate(top_features,1):
    print(i, f)
