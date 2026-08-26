import pandas as pd
import numpy as np
import optuna
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


np.random.seed(42)
random.seed(42)


class BP_FT_Model:

    def __init__(self, file_path):

        self.file_path = file_path

        self.X = None
        self.y = None

        self.model_F = None

        self.best_params_F = {}

        self.metrics = {}

    # ===================== 1. 数据 =====================
    def load_data(self):

        df = pd.read_excel(self.file_path)

        self.X = df.iloc[:, 1:4].values

        # F = 第5列
        self.y_F = df.iloc[:, 4].values

    # ===================== 2. 模型 =====================
    def build_model(self, hidden_dim, alpha):

        mlp = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation='tanh',
            solver='lbfgs',
            alpha=alpha,
            max_iter=5000,
            random_state=42
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", mlp)
        ])

        model = TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler()
        )

        return model

    # ===================== 3. Optuna + 5折CV =====================
    def optimize(self, X, y, n_trials=50):

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):

            params = {
                "hidden_dim": trial.suggest_int("hidden_dim", 4, 16),
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
            }

            scores = []

            for tr_idx, va_idx in cv.split(X):

                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                model = self.build_model(**params)
                model.fit(X_tr, y_tr)

                pred = model.predict(X_va)
                scores.append(r2_score(y_va, pred))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    # ===================== 4. 训练 =====================
    def train(self, n_trials=100):

        self.load_data()

        # ================= F =================
        best_params_F = self.optimize(self.X, self.y_F, n_trials)
        self.best_params_F = best_params_F

        print("\n===== Best Hyperparameters for F =====")
        print(self.best_params_F)

        model_F = self.build_model(**best_params_F)
        model_F.fit(self.X, self.y_F)
        pred_F = model_F.predict(self.X)

        self.model_F = model_F

        self.metrics["F"] = {
            "R2": r2_score(self.y_F, pred_F),
            "RMSE": np.sqrt(mean_squared_error(self.y_F, pred_F)),
            "MAPE": mean_absolute_percentage_error(self.y_F, pred_F)
        }

        self.plot(self.y_F, pred_F, "F")
        self.plot_residual(self.y_F, pred_F, "F")

    # ===================== 5. 可视化 =====================
    def plot(self, y_true, y_pred, target):

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)

        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())

        plt.plot([min_v, max_v], [min_v, max_v], '--')

        plt.xlabel("Actual cutting force / N")
        plt.ylabel("Predicted cutting force / N")
        plt.title(f"Parity Analysis of BP Model for {target}")

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        plt.text(
            0.05, 0.95,
            f"$R^2$={r2:.4f}\nRMSE = {rmse:.2f} N",
            transform=plt.gca().transAxes,
            verticalalignment='top'
        )

        plt.tight_layout()
        plt.show()

    def plot_residual(self, y_true, y_pred, target):

        residual = y_pred - y_true

        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residual, alpha=0.6)
        plt.axhline(0, linestyle='--')

        plt.xlabel("Predicted cutting force / N")
        plt.ylabel("Residual of cutting force / N")
        plt.title(f"Residual Analysis of BP Model for {target} Prediction")

        plt.tight_layout()
        plt.show()

    # ===================== 6. 预测接口 =====================
    def predict(self, x):

        x = np.array(x).reshape(1, -1)

        F = self.model_F.predict(x)[0]

        return {
            "F": F
        }

    # ===================== 7. 保存模型 =====================
    def save(self, path="bp_f_model.pkl"):

        joblib.dump({
            "model_F": self.model_F,
            "best_params_F": self.best_params_F,
            "metrics": self.metrics
        }, path)

        print("Saved:", path)

    # ===================== 8. 加载模型 =====================
    def load(self, path="bp_f_model.pkl"):

        data = joblib.load(path)

        self.model_F = data["model_F"]
        self.best_params_F = data["best_params_F"]
        self.metrics = data["metrics"]

        print("Loaded:", path)


if __name__ == '__main__':

    model = BP_FT_Model(r"C:\Users\liuhuiru\Desktop\tool_opt\final_data.xlsx")

    model.train(n_trials=100)

    model.save('bp_f_model.pkl')