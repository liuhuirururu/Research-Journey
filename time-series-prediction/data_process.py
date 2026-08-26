'''
特征处理包括;
1.缺失值、异常值、重复值等处理
2.特征融合
3.添加滞后特征、时间间隔特征等（针对时序数据的处理）
4.Z-score标准化
'''
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from time_feature import add_time_delta,add_lag_diff_feature

# 全局设置
SEED = 42
np.random.seed(SEED)

# IRQ异常值检测
def box_outlier_replace(data,colums):
    data_new = data.copy().astype(float)
    for col in colums:
        # 计算四分位数
        Q1 = data_new[col].quantile(0.25)
        Q3 = data_new[col].quantile(0.75)
        IQR = Q3 - Q1
        # 异常值上下界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 判断异常点
        lower_outlier = data_new[col] < lower_bound
        upper_outlier = data_new[col] > upper_bound
        outlier_num = lower_outlier.sum() + upper_outlier.sum()
        if outlier_num > 0:
            print(f"{col}:检测到{outlier_num}个异常值")
        # 边界值替换
        data_new.loc[lower_outlier,col] = lower_bound
        data_new.loc[upper_outlier,col] = upper_bound
    return data_new

"""
将数据按月份分开（避免中间7月份数据空缺的影响）
"""
def split_by_month(data):
    # 提取time里面的月份
    data.index = pd.to_datetime(data.index)
    month = data.index.month
    train_months = [5,6,7,8,9,10]
    val_months = [11]
    train_data = data[month.isin(train_months)]
    val_data = data[month.isin(val_months)]
    return train_data, val_data

"""
在月份内部创建时间窗口
data：X_train
label:y_train
"""
def create_month_window(data,label,window=6):
    X_window = []
    y_window = []
    months = data.index.month.unique() #当前所有月份
    for m in months:
        # 当前月份数据
        month_data = data[data.index.month == m]
        month_label = label.loc[month_data.index]
        X_month = month_data.values
        y_month = month_label.values
        # 生成历史窗口
        for i in range(window,len(X_month)):
            history_and_current = X_month[i-window:i+1] #历史信息 t-i,t-1 + t
            target = y_month[i]
            X_window.append(history_and_current)
            y_window.append(target)
    return np.array(X_window), np.array(y_window)

# 只对X数据进行预处理
def preprocess_data(X,scaler_path = None,scaler_save_path = "scaler_x.pkl"):
    # 异常值处理
    feature_cols = X.columns
    X_clean = box_outlier_replace(X,feature_cols)
    # 特征融合
    X_clean['顶压mean'] = (X_clean['顶压'] + X_clean['顶压2'] + X_clean['顶压3'] + X_clean['顶压4']) / 4
    X_clean['冷风压力mean'] = (X_clean['冷风压力'] + X_clean['冷风压力2']) / 2
    X_clean['热风压力mean'] = (X_clean['热风压力'] + X_clean['热风压力2']) / 2
    X_clean['顶温mean'] = (X_clean['顶温东北'] + X_clean['顶温西南'] + X_clean['顶温西北'] + X_clean[
        '顶温东南']) / 4
    drop_cols = ['顶压', '顶压2', '顶压3', '顶压4', '冷风压力', '冷风压力2', '热风压力', '热风压力2',
                 '顶温东北', '顶温西南', '顶温西北', '顶温东南']
    X_clean.drop(columns = drop_cols, inplace=True)
    X_clean = add_lag_diff_feature(X_clean)
    X_clean = add_time_delta(X_clean)
    # 标准化
    if scaler_path is not None:
        # 验证集、测试集
        scaler_x = joblib.load(scaler_path)
        X_scaled = scaler_x.transform(X_clean)
    else:
        # 训练集部分
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X_clean)
        # 保存标准化参数
        if scaler_save_path is not None:
            joblib.dump(scaler_x, scaler_save_path)
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X_clean.columns,
        index=X_clean.index
    )

    return X_scaled







