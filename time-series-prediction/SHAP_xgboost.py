import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_feature_importance(model,X,top_n=30,save_path="SHAP_top30.tif"):
    plt.rcParams['font.sans-serif'] = [
        'SimSun',
        'Microsoft YaHei'
    ]

    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 600,
    })

    # 创建解释器
    explainer = shap.TreeExplainer(model)
    # SHAP值
    shap_values = explainer.shap_values(X)
    # 平均绝对SHAP
    importance = np.mean(np.abs(shap_values),axis=0)*10
    # 特征排序
    feature_names = X.columns
    importance_df = pd.DataFrame(
        {
            "feature":feature_names,
            "importance":importance
        }
    )
    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )
    # 保存全部排序
    importance_df.to_csv(
        "shap_feature_importance.csv",
        index=False,
        encoding="gbk"
    )
    # 取Top30
    top_features = (
        importance_df
        .head(top_n)
        .sort_values(
            "importance",
            ascending=True
        ))
    fig, ax = plt.subplots(figsize=(10, 8))
    # 蓝色渐变
    colors = plt.cm.Blues(np.linspace(0.35,0.9,len(top_features)))
    bars = ax.barh(
        top_features["feature"],
        top_features["importance"],
        height=0.55,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}',
            va='center',
            fontsize=9
        )
    ax.grid(
        axis='x',
        linestyle='--',
        alpha=0.3
    )

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.set_xlabel("Mean(|SHAP value|)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()
    # Top特征名称（按照SHAP重要性排序）
    top_features = (
        importance_df
        .head(top_n)
        ["feature"]
        .tolist()
    )
    return importance_df, top_features


