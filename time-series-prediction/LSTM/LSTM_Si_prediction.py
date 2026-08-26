'''
本项目代码对比了LSTM，CNN-LSTM，CNN-LSTM-Attention
CNN-LSTM表现更好，作为保留
'''

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM,Dense,Dropout,Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from data_process import preprocess_data,split_by_month,create_month_window
import tensorflow as tf
import joblib

# 1.全局设置
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# 2.数据读取
data = pd.read_csv(
    r"C:\Users\liuhuiru\Desktop\Si_prediction\datas\traindata.csv",
     index_col='time',
     encoding="gbk")
# 划分训练集、验证集
train_data,val_data = split_by_month(data)
# 训练集
feature_cols =['last','设定喷煤量','CO','上小时实际喷煤量','顶温mean','冷风温度','CO2',
               '实际风速','鼓风湿度','全压差','顶温下降管','透气性指数','阻力系数','富氧流量','顶压mean','炉腹煤气量']
X_train = train_data.drop(columns='label')
y_train = train_data['label']
X_train = preprocess_data(X_train,scaler_path=None,scaler_save_path="scaler_lstm.pkl")# 预处理
X_train = X_train[feature_cols]
# 1.X_train,y_train=create_month_window(X_train,y_train)#创建时间窗口
# 验证集
X_val = val_data.drop(columns='label')
y_val = val_data['label']
X_val = preprocess_data(X_val,scaler_path="scaler_lstm.pkl",scaler_save_path=None)
X_val = X_val[feature_cols]
# 2.X_val,y_val = create_month_window(X_val,y_val)

'''
# 加入时间注意力机制
@register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self,**kwargs):
        super(TemporalAttention,self).__init__(**kwargs)
    def build(self, input_shape):
        # input_shape:
        # (batch, time_steps, hidden_dim)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        score = tf.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(score,axis=1)
        context_vector = x * attention_weights
        context_vector = tf.reduce_sum(context_vector,axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[2]
        )
'''

# 3.建立LSTM模型（加入CNN）
def build_lstm_model(input_shape,
                     filters,kernel_size,
                     units,dropout,learning_rate):
    model = Sequential()
    #使用CNN提取局部时间特征
    model.add(Conv1D(filters=filters,kernel_size=kernel_size,padding = "same",activation='relu',input_shape=input_shape))
    #LSTM学习长期依赖
    model.add(LSTM(units,return_sequences=False))
    #model.add(TemporalAttention())
    model.add(Dropout(dropout))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
# 超参数优化
def objective(trial):
    # 3.
    window = trial.suggest_int("window", 4, 12)
    # 4. (增加窗口创建)
    X_train_win,y_train_win = create_month_window(X_train,y_train,window=window)
    X_val_win,y_val_win = create_month_window(X_val,y_val,window=window)
    # 搜索参数
    params = {
        "filters": trial.suggest_int("filters", 16, 48),
        "kernel_size": trial.suggest_int("kernel_size", 2, 3),
        "units": trial.suggest_int("units", 16, 64),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    }
    # 建立模型
    model = build_lstm_model(input_shape=(X_train_win.shape[1],X_train_win.shape[2]),
                             filters=params["filters"],
                             kernel_size=params["kernel_size"],
                             units=params["units"],
                             dropout=params["dropout"],
                             learning_rate=params["learning_rate"])
    # 训练模型
    model.fit(X_train_win,y_train_win,epochs=50,batch_size=params["batch_size"],
              validation_data=(X_val_win,y_val_win),verbose=0)
    # 预测
    y_pred = model.predict(X_val_win,verbose=0)
    y_pred = y_pred.reshape(-1)
    # 评价
    mse = mean_squared_error(y_val_win,y_pred)
    tf.keras.backend.clear_session()
    return mse

study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30)
best_params = study.best_params
print(best_params)
joblib.dump(best_params,"best_list_params.pkl")
