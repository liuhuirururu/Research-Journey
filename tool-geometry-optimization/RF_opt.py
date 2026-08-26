import pandas as pd
import numpy as np
import optuna
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

np.random.seed(42)
random.seed(42)
class RF_Optimization:

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
    def build_model(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),   # 保持与你SVR一致（统一实验）
            ("rf", rf)
        ])

        model = TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler()
        )

        return model

    # ===================== 3. 内层优化 =====================
    def optimize(self, X_train, y_train, n_trials=30):

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

            scores = []

            for tr_idx, va_idx in inner_cv.split(X_train):

                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                model = self.build_model(**params)
                model.fit(X_tr, y_tr)

                pred = model.predict(X_va)
                scores.append(r2_score(y_va, pred))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    # ===================== 4. 外层CV=====================
    def run_cv(self, n_trials=30):

        self.load_data()

        outer_cv = KFold(n_splits=5, shuffle=False)

        for t_idx, target in enumerate(self.targets):

            y_all = self.y[:, t_idx]

            r2_list, rmse_list, mape_list, mae_list = [], [], [], []

            for train_idx, test_idx in outer_cv.split(self.X):

                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = y_all[train_idx], y_all[test_idx]

                # ===== 内层调参 =====
                best_params = self.optimize(X_train, y_train, n_trials)

                # ===== 训练最终模型 =====
                model = self.build_model(**best_params)
                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                # ===== 指标 =====
                r2_list.append(r2_score(y_test, pred))
                rmse_list.append(np.sqrt(mean_squared_error(y_test, pred)))
                mape_list.append(mean_absolute_percentage_error(y_test, pred))
                mae_list.append(mean_absolute_error(y_test ,pred))

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

        print("\n========== RF NESTED CV RESULTS ==========")

        for k, v in self.results.items():
            print(f"\n[{k}]")
            print(f"R2   = {v['R2_mean']:.4f} ± {v['R2_std']:.4f}")
            print(f"RMSE = {v['RMSE_mean']:.3f} ± {v['RMSE_std']:.3f}")
            print(f"MAPE = {v['MAPE_mean']:.2%} ± {v['MAPE_std']:.2%}")
            print(f"MAE  = {v['MAE_mean']:.3f} ± {v['MAE_std']:.3f}")


# ===================== 主程序 =====================
if __name__ == "__main__":

    model = RF_Optimization(
        r"C:\Users\liuhuiru\Desktop\tool_opt\final_data.xlsx"
    )

    model.run_cv(n_trials=100)
    model.print_results()