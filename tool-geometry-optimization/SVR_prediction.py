"""
读取100组原始数据
    ↓
Outer 5Fold
    ↓
第1折：
    训练集(80)
        ↓
        仅在这80上做异常检测(IQR+IF)
        ↓
        Inner CV + Optuna找最优参数
        ↓
        用最优参数训练clean_train
        ↓
        预测outer_test(20)
        ↓
        保存R2/RMSE/MAPE/MAXERR

第2折 ... 第5折重复
    ↓
5折指标求 mean±std
（这是论文报告结果）
    ↓
然后在全100组上：
    ↓
全数据异常检测
    ↓
Optuna + InnerCV 再找一次全局最优参数
    ↓
fit(X_all,y_all)
    ↓
得到最终部署模型
    ↓
predict_new()
"""
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 绘图设置 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks")


class SVR_Optimization:

    def __init__(self, file_path):

        self.file_path = file_path
        self.targets = ['Fx', 'Fy', 'T']
        # 原始数据
        self.X = None
        self.y = None
        # 最终部署模型
        self.models = {}
        # 最终最优参数
        self.best_params = {}
        # 5折outer cv结果
        self.cv_results = {}

    def load_data(self):
        """只读取数据"""
        df = pd.read_excel(self.file_path)
        self.X = df.iloc[:, 1:4].values
        self.y = df.iloc[:, 4:7].copy()
        self.y.columns = self.targets

    def clean_train_data(self, X_train, y_train, target_name):
        """只清洗当前fold测试集。训练集不能动"""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        return X_train, y_train


    def build_model(self, C, epsilon, gamma):
        """ 构建严格无泄漏SVR pipeline"""
        x_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(
                kernel='rbf',
                C=C,
                epsilon=epsilon,
                gamma=gamma
            ))
        ])
        # ✔ 关键：y标准化也进入模型内部（无泄露核心）
        model = TransformedTargetRegressor(
            regressor=x_pipe,
            transformer=StandardScaler()
        )
        return model

    def optimize_params(self, X_train, y_train, n_trials=80):

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True)
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

    def optimize_and_train(self, n_trials=50):

        self.load_data()

        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        self.cv_results = {}

        final_params_list = {}  # ⭐用于最终模型参数稳定化

        for target in self.targets:

            y_all = self.y[target].values

            fold_metrics = {
                "r2": [],
                "rmse": [],
                "mape": [],
                "max_err": []
            }

            best_params_per_fold = []

            # ================= OUTER CV =================
            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(self.X), 1):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = y_all[train_idx], y_all[test_idx]

                # ===== Inner CV tuning =====
                best_param = self.optimize_params(X_train, y_train, n_trials)

                best_params_per_fold.append(best_param)

                # ===== train final model for this fold =====
                model = self.build_model(**best_param)
                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                # ===== metrics =====
                fold_metrics["r2"].append(r2_score(y_test, pred))
                fold_metrics["rmse"].append(np.sqrt(mean_squared_error(y_test, pred)))
                fold_metrics["mape"].append(mean_absolute_percentage_error(y_test, pred))
                fold_metrics["max_err"].append(np.max(np.abs(y_test - pred)))

            # ================= CV RESULT =================
            self.cv_results[target] = {
                "r2_mean": np.mean(fold_metrics["r2"]),
                "r2_std": np.std(fold_metrics["r2"]),
                "rmse_mean": np.mean(fold_metrics["rmse"]),
                "rmse_std": np.std(fold_metrics["rmse"]),
                "mape_mean": np.mean(fold_metrics["mape"]),
                "mape_std": np.std(fold_metrics["mape"]),
                "max_err_mean": np.mean(fold_metrics["max_err"]),
                "max_err_std": np.std(fold_metrics["max_err"])
            }

            # ================= FINAL PARAM (关键改动) =================
            # ⭐ 用CV结果“稳定化参数”而不是重新optimize全数据
            final_param = {
                k: np.mean([p[k] for p in best_params_per_fold])
                for k in best_params_per_fold[0]
            }

            final_model = self.build_model(**final_param)
            final_model.fit(self.X, y_all)

            self.models[target] = final_model

            # 保存最终参数
            self.best_params[target] = final_param
            print("\n========== FINAL MODEL EVALUATION ==========")

            for target in self.targets:
                y_true = self.y[target].values
                y_pred = self.models[target].predict(self.X)

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = mean_absolute_percentage_error(y_true, y_pred)
                max_err = np.max(np.abs(y_true - y_pred))

                print(f"\n[{target}] FINAL MODEL")
                print(f"R2   = {r2:.4f}")
                print(f"RMSE = {rmse:.4f}")
                print(f"MAPE = {mape:.2%}")
                print(f"MAX  = {max_err:.4f}")

        self.plot_results()

    def plot_results(self):
        fig,axes = plt.subplots(2,3,figsize=(18,10))
        for i,target in enumerate(self.targets):
            y_true = self.y[target].values
            y_pred = self.models[target].predict(self.X)
            residual = y_true - y_pred
            # 散点图
            axes[0,i].scatter(y_true,y_pred,c='orange',edgecolors='k')
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())
            axes[0,i].plot([mn,mx],[mn,mx],'r--')
            axes[0,i].set_title(f"{target}预测")
            # 残差图
            axes[1, i].scatter(
                y_pred,
                residual,
                c='green',
                edgecolors='k'
            )
            axes[1,i].set_title(f"{target}残差")
        plt.tight_layout()
        plt.show()

    def predict_new(self,x):
        x = np.array(x)
        return np.array([
            self.models[t].predict(x)[0]
            for t in self.targets
        ])


if __name__ == "__main__":
    model = SVR_Optimization(
        r"C:\Users\liuhuiru\Desktop\tool_opt\data100.xlsx"
    )

    model.optimize_and_train(
        n_trials=50
    )
    """
    pred = model.predict_new([
        [15, 8, 0.03]
    ])

    print("\n预测结果:")
    print(
        f"Fx={pred[0]:.4f}, "
        f"Fy={pred[1]:.4f}, "
        f"T={pred[2]:.4f}"
    )
    """
