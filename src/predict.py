import joblib
import pandas as pd
import os
import sys
import numpy as np  # 用于绩点范围限制
from typing import Dict  # 仅导入需要的，移除错误的Float

def predict_gpa(new_student_data: Dict, project_root: str) -> float:
    """
    学生绩点预测核心函数：加载模型+特征流水线，处理输入，返回预测绩点
    参数：
        new_student_data: 字典，包含学生8个特征（与数据集中一致）
        project_root: 项目根目录路径（用于定位模型/流水线文件）
    返回：
        float：预测绩点（保留2位小数，强制限制在1.0-4.0）
    """
    # 基础参数校验
    if not isinstance(new_student_data, dict):
        raise TypeError("❌ new_student_data必须是字典类型")
    if not isinstance(project_root, str) or not os.path.isdir(project_root):
        raise NotADirectoryError(f"❌ 项目根目录无效：{project_root}")

    # 定位模型和特征流水线文件（使用线性回归，拟合效果最优）
    models_dir = os.path.join(project_root, "models")
    model_path = os.path.join(models_dir, "linear_regression.pkl")
    preprocessor_path = os.path.join(models_dir, "feature_preprocessor.pkl")

    # 校验目录/文件是否存在
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"❌ 模型目录不存在：{models_dir}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"❌ 特征流水线缺失：{preprocessor_path}\n请先运行main.py训练模型")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件缺失：{model_path}\n请先运行main.py训练模型")

    # 加载模型和特征预处理流水线
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        raise RuntimeError(f"❌ 加载模型/流水线失败：{str(e)}") from e

    # 处理输入数据：转换为DataFrame（适配sklearn流水线）
    input_df = pd.DataFrame([new_student_data])

    # 校验输入特征是否完整（8个特征，无缺失）
    required_features = [
        "major", "gender", "attendance", "homework_completion",
        "lib_borrow", "club_participation", "class_interaction", "exam_score"
    ]
    missing_features = [f for f in required_features if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"❌ 缺失特征：{missing_features}，请补充后重新预测")

    # 数值特征类型校验（防止非数字输入）
    numeric_features = ["attendance", "homework_completion", "lib_borrow",
                        "class_interaction", "exam_score"]
    for feat in numeric_features:
        if not pd.api.types.is_numeric_dtype(input_df[feat]):
            raise TypeError(f"❌ 特征{feat}必须是数值类型（整数/小数）")

    # 特征预处理：使用训练好的流水线转换输入
    input_processed = preprocessor.transform(input_df)

    # 预测绩点 + 【核心修复】强制限制绩点在1.0-4.0合理范围
    pred_gpa = model.predict(input_processed)[0]
    pred_gpa = np.clip(pred_gpa, 1.0, 4.0)  # 防止超范围
    pred_gpa = round(pred_gpa, 2)  # 保留2位小数，贴合实际

    return pred_gpa