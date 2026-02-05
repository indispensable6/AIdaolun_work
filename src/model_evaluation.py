import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import platform
from typing import Dict, List

# 跨平台字体配置：解决Windows/macOS/Linux中文显示乱码问题
def setup_plt_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    elif system == "Linux":
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    elif system == "Darwin":  # macOS
        plt.rcParams["font.family"] = ["PingFang SC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.rcParams["figure.dpi"] = 100  # 画布分辨率
    plt.rcParams["savefig.dpi"] = 300  # 保存图片分辨率

setup_plt_font()  # 初始化字体

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
    """
    评估回归模型性能，返回核心指标
    参数：y_true-真实绩点，y_pred-预测绩点，model_name-模型名称
    返回：包含MAE/RMSE/R²的字典
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"❌ 真实值与预测值长度不匹配：{len(y_true)} vs {len(y_pred)}")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("❌ 真实值和预测值必须为一维数组")

    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # 打印结果
    print(f"{model_name}：")
    print(f"   - MAE（平均绩点误差）：{mae:.4f}")
    print(f"   - RMSE（均方根误差）：{rmse:.4f}")
    print(f"   - R²（决定系数）：{r2:.4f}\n")

    return {"模型": model_name, "MAE": mae, "RMSE": rmse, "R²（决定系数）": r2}

def plot_model_comparison(metrics_list: List[Dict], project_root: str):
    """绘制模型性能对比图（R²指标），保存至results/"""
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(metrics_list)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="模型", y="R²（决定系数）", hue="模型", data=df,
                palette=["#3498db", "#e74c3c", "#2ecc71"], legend=False)
    plt.title("学生绩点预测模型性能对比（R²越高越好）", fontsize=14, pad=20)
    plt.ylim(0, 1)  # R²范围0-1
    plt.xlabel("模型", fontsize=12)
    plt.ylabel("R²（决定系数）", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()  # 适配布局，防止标签截断

    save_path = os.path.join(results_dir, "model_r2_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📸 模型对比图已保存：{save_path}")

def plot_feature_importance(model, preprocessor, project_root: str):
    """绘制特征重要性Top10图，修复【数组长度不一致】报错，保存至results/"""
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 仅树模型有特征重要性属性
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️ 模型{model.__class__.__name__}无特征重要性属性，跳过绘图")
        return

    # 【核心修复】从流水线自动提取特征名称，匹配预处理后的列数
    num_features = preprocessor.transformers_[0][2]  # 数值特征列表
    cat_encoder = preprocessor.named_transformers_["cat"]  # 类别编码器
    cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])  # 独热编码后类别特征名
    all_features = list(num_features) + list(cat_features)  # 合并所有特征名

    # 二次校验：防止特征名与重要性数组长度不匹配，自动截断
    importances = model.feature_importances_
    if len(all_features) != len(importances):
        print(f"⚠️ 特征名长度({len(all_features)})与重要性长度({len(importances)})不匹配，自动截断")
        min_len = min(len(all_features), len(importances))
        all_features = all_features[:min_len]
        importances = importances[:min_len]

    # 构建特征重要性DataFrame，取Top10
    df = pd.DataFrame({"特征": all_features, "重要性": importances})
    df = df.sort_values(by="重要性", ascending=False).head(10)

    # 绘图
    plt.figure(figsize=(12, 6))
    sns.barplot(x="重要性", y="特征", hue="特征", data=df,
                palette="Greens_r", legend=False)
    plt.title("学生绩点预测Top10特征重要性", fontsize=14, pad=20)
    plt.xlabel("重要性得分", fontsize=12)
    plt.ylabel("特征", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📸 特征重要性图已保存：{save_path}")

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, project_root: str):
    """绘制真实绩点vs预测绩点散点图，添加长度校验，保存至results/"""
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 长度校验：自动截断至相同长度
    if len(y_true) != len(y_pred):
        print(f"⚠️ 真实值与预测值长度不匹配，自动截断")
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color="#2E86AB", s=50)  # 散点图
    # 绘制完美预测线（y=x）
    min_gpa, max_gpa = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_gpa, max_gpa], [min_gpa, max_gpa], "r--", lw=2, label="完美预测线")
    plt.xlabel("真实绩点（GPA）", fontsize=12)
    plt.ylabel("预测绩点（GPA）", fontsize=12)
    plt.title("学生真实绩点 vs 预测绩点", fontsize=14, pad=20)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, "pred_vs_true.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📸 绩点对比图已保存：{save_path}")