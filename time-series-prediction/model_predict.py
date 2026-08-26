'''
基于得到的最优XGBoost模型完成模型评估
'''

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import plotly.graph_objects as go
import plotly.express as px
from data_process import preprocess_data

# 1.全局设置
SEED = 42
np.random.seed(SEED)

# 2.首先定义下输出指标的函数
def evaluate_prediction(y_true, y_pred, save_path=None,plot_path=None,residual_path=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # 1.MSE
    mse = mean_squared_error(y_true, y_pred)
    # 2.Hit Rate
    error = np.abs(y_true - y_pred)
    hit_rate = np.mean(error <= 0.1) * 100
    # 3.Tendency
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    # 这里加上符号函数。sign:大于0返回1，等于0返回0，小于0返回-1
    tendency_flag = (np.sign(true_diff) == np.sign(pred_diff)).astype(int)
    tendency = np.mean(tendency_flag)*100
    # 4.绘制预测值-真实值对比图
    if plot_path is not None:
        x_axis = np.arange(len(y_true))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_true, mode="lines",name="True Values",
                                 line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=x_axis, y=y_pred, mode="lines",name="Predicted Values",
                                 line=dict(color="red", width=2)))
        fig.update_layout(title="True Values vs. Predicted Values",xaxis_title_text="time ",yaxis_title_text="label:Si%",
                          hovermode="x unified",template="plotly_white",width=1400,height=500)
        fig.write_html(plot_path)
    # 5.绘制残差图
    residual = y_true - y_pred
    if residual_path is not None:
        x_axis = np.arange(len(residual))
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=x_axis, y=residual, mode="lines",name="Residuals"))
        #添加0参考线
        fig_res.add_hline(y=0)
        fig_res.update_layout(xaxis_title="time",yaxis_title="residual",hovermode="x unified",template="plotly_white",
                              width=1400,height=500)
        fig_res.write_html(residual_path)

    # 保存结果
    metrics = {"MSE": mse,"Hit Rate(%)": hit_rate,"Tendency(%)": tendency}
    if save_path is not None:
        result = pd.DataFrame([metrics])
        result.to_csv(save_path, index=False,encoding="gbk")
    return metrics

# 3.加载模型
model = joblib.load("final_xgboost.pkl")
print("XGBoost模型加载完成")

# 拼接历史数据
def add_history_x(history_x,test_x,history_times):
    history = history_x.loc[history_times]
    # 拼接
    X = pd.concat([history,test_x],axis=0)
    return X.sort_index()

def load_test_data(x_path,y_path,history_y):
    X = pd.read_csv(x_path,index_col="time",encoding="gbk")
    y = pd.read_csv(y_path,index_col="time",encoding="gbk")
    X.index = pd.to_datetime(X.index)
    y.index = pd.to_datetime(y.index)
    y = y["label"]
    y = pd.concat([history_y,y])
    y = y.sort_index()
    return X,y

def predict_month(test_x_path,test_y_path,history_time,history_y,month_name):
    test_x, y = load_test_data(test_x_path, test_y_path,history_y)
    print("拼接历史信息前:", test_x.shape)
    test_x = add_history_x(history_x, test_x, history_time)
    print("拼接历史信息后:", test_x.shape)
    # 标准化处理
    test_x.index = pd.to_datetime(test_x.index)
    test_x = test_x.sort_index()
    X = preprocess_data(test_x,scaler_path="scaler_xgb_shap2.pkl",scaler_save_path=None)
    print("标准化处理后:", X.shape)
    final_features = joblib.load("final_xgb_features.pkl")
    X = X[final_features]
    y_true = y.iloc[1:]
    print(y_true.shape)
    delta_pred = model.predict(X)
    # history_y + test_y用于提供上一时刻Si
    last_si = y.shift(1)
    last_si = last_si[1:]
    # 输出检查
    print("预测输入维度:", X.shape)
    print("输入特征:", len(final_features))
    y_pred = delta_pred + last_si.values
    # 保存预测结果
    prediction_result = pd.DataFrame({
        "time": X.index,
        "label": y_pred
    })
    prediction_result.to_csv(
        f"{month_name}_output.csv",
        index=False,
        encoding="gbk"
    )
    metrics = evaluate_prediction(
        y_true=y_true,
        y_pred=y_pred,
        save_path=f"{month_name}_metrics.csv",
        plot_path=f"{month_name}_prediction.html",
        residual_path=f"{month_name}_residual.html"
    )
    return metrics
# 调用
history_path =  r"C:\Users\liuhuiru\Desktop\Si_model\datas\traindata.csv"
history_data = pd.read_csv(history_path,index_col="time",encoding="gbk")
history_data.index=pd.to_datetime(history_data.index)
history_x = history_data.drop(columns = "label")
# 7:,"2020/6/30  22:52:00",,
# 12:"2020/11/30  23:13:00",,
history7_times = pd.to_datetime(["2020/6/30  23:16:00","2020/6/30  23:38:00"])
history12_times = pd.to_datetime(["2020/11/30  23:32:00","2020/11/30  23:54:00"])

history_y7 = pd.Series([0.407],index=pd.to_datetime(["2020/6/30  23:38:00"]),name="label")
history_y12 = pd.Series([0.553],index=pd.to_datetime(["2020/11/30  23:54:00"]),name="label")

if __name__ == "__main__":
    # 输入测试集路径
    test7x_path = r"C:\Users\liuhuiru\Desktop\Si_model\datas\test7.csv"
    test7y_path = r"C:\Users\liuhuiru\Desktop\Si_model\datas\test7_y.csv"
    predict_month(test7x_path,test7y_path,history7_times,history_y7,"test7")
    test12x_path = r"C:\Users\liuhuiru\Desktop\Si_model\datas\test12.csv"
    test12y_path = r"C:\Users\liuhuiru\Desktop\Si_model\datas\test12_y.csv"
    predict_month(test12x_path, test12y_path, history12_times, history_y12,"test12")


