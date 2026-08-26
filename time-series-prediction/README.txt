# Si Prediction
基于SHAP特征选择与XGBoost的铁水硅含量预测

## 项目简介

针对高炉铁水硅含量预测问题，利用高炉生产过程中的多变量炉况数据，建立下一时刻铁水硅含量预测模型，实现 `t(Si) → t+1(Si)`。
项目对比XGBoost、LSTM和CNN-LSTM等模型，并采用Optuna进行XGBoost超参数优化，结合SHAP进行特征重要性分析与特征选择。实验结果表明，XGBoost在当前数据集上的综合预测效果较好。

## 主要方法
- 数据预处理与特征工程
- 时序特征构造
- XGBoost回归预测
- Optuna超参数优化
- SHAP特征分析与选择
- LSTM时序预测
- CNN-LSTM时序预测
- 模型对比与性能评价
