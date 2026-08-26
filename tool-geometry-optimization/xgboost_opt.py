# 这个最好加入早停法
# 为方便加入早停法，我把pipeline去掉了
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)
class XGBoost_Optimization:

    def __init__(self, file_path):
        self.file_path = file_path
        self.X = None
        self.y = None

        self.targets = ['Fx', 'σ', 'T']
        self.results = {}

    # ===================== 1. 数据 =====================
    def load_data(self):
        df = pd.read_excel(self.file_path)

        self.X = df.iloc[:, 1:4].values
        self.y = df.iloc[:, 4:7].values

    # ===================== 2. 模型 =====================
    def build_model(self, **params):

        model = xgb.XGBRegressor(
            **params,
            n_estimators=2000, #使用早停法，所以可以增大树的数量，让模型自行裁断
            objective="reg:squarederror",
            early_stopping_rounds=50, #如果50轮没咋变就终止
            tree_method="hist",
            random_state=42,
        )
        return model

    # ===================== 3. 内层优化 =====================
    def optimize(self, X_train, y_train, n_trials=30):

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):

            params = {
               # "n_estimators": trial.suggest_int("n_estimators", 200 ,3000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":1.0,
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 6),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            }

            scores = []

            for tr_idx, va_idx in inner_cv.split(X_train):

                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                # 标准化
                x_scaler = StandardScaler()
                X_tr_scaled = x_scaler.fit_transform(X_tr)
                X_va_scaled = x_scaler.transform(X_va)
                y_scaler = StandardScaler()
                y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
                y_va_scaled = y_scaler.transform(y_va.reshape(-1, 1)).ravel()

                model = self.build_model(**params)
                model.fit(
                    X_tr_scaled,y_tr_scaled,
                    eval_set=[(X_va_scaled, y_va_scaled)],
                    verbose=False
                )

                pred_scaled = model.predict(X_va_scaled)
                pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                scores.append(r2_score(y_va, pred))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    # ===================== 4. 外层CV =====================
    def run_cv(self, n_trials=30):

        self.load_data()

        outer_cv = KFold(n_splits=5, shuffle=False)

        for t_idx, target in enumerate(self.targets):

            y_all = self.y[:, t_idx]

            r2_list, rmse_list, mape_list, mae_list = [], [], [], []

            for train_idx, test_idx in outer_cv.split(self.X):

                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = y_all[train_idx], y_all[test_idx]
                # 加入外层标准化
                x_scaler = StandardScaler()
                X_train_scaled = x_scaler.fit_transform(X_train)
                X_test_scaled = x_scaler.transform(X_test)
                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

                # ===== 内层优化 =====
                best_params = self.optimize(X_train, y_train, n_trials)

                # ===== 训练最终模型 =====
                # 在训练集内部，再划分出一个验证集（因为外侧测试集做early stopping会泄露）
                model = self.build_model(**best_params)
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_train_scaled,y_train_scaled,
                    test_size=0.2,random_state=None
                )

                model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

                pred_scaled = model.predict(X_test_scaled)
                pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

                # ===== 指标 =====
                r2_list.append(r2_score(y_test, pred))
                rmse_list.append(np.sqrt(mean_squared_error(y_test, pred)))
                mape_list.append(mean_absolute_percentage_error(y_test, pred))
                mae_list.append(mean_absolute_error(y_test , pred))

            self.results[target] = {
                "R2_mean": np.mean(r2_list),
                "R2_std": np.std(r2_list),

                "RMSE_mean": np.mean(rmse_list),
                "RMSE_std": np.std(rmse_list),

                "MAPE_mean": np.mean(mape_list),
                "MAPE_std": np.std(mape_list),

                "MAE_mean": np.mean(mae_list),
                "MAE_std": np.std(mae_list),
            }

    # ===================== 5. 输出 =====================
    def print_results(self):

        print("\n========== XGBOOST NESTED CV RESULTS ==========")

        for k, v in self.results.items():
            print(f"\n[{k}]")
            print(f"R2   = {v['R2_mean']:.4f} ± {v['R2_std']:.4f}")
            print(f"RMSE = {v['RMSE_mean']:.3f} ± {v['RMSE_std']:.3f}")
            print(f"MAPE = {v['MAPE_mean']:.2%} ± {v['MAPE_std']:.2%}")
            print(f"MAE  = {v['MAE_mean']:.3f} ± {v['MAE_std']:.3f}")


# ===================== 主程序 =====================
if __name__ == "__main__":

    model = XGBoost_Optimization(
        r"C:\Users\liuhuiru\Desktop\tool_opt\final_data.xlsx"
    )

    model.run_cv(n_trials=100)
    model.print_results()