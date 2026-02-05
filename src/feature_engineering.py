from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple

def build_feature_preprocessor() -> Tuple[ColumnTransformer, list, list]:
    """
    构建特征预处理流水线：数值特征标准化 + 类别特征独热编码
    适配：移除入学排名，新增class_interaction/exam_score两个数值特征
    返回：预处理流水线、数值特征列表、类别特征列表
    """
    # 数值特征（适配新特征，无entrance_rank）
    num_features = [
        "attendance", "homework_completion", "lib_borrow",
        "class_interaction", "exam_score"
    ]
    # 类别特征（无变化）
    cat_features = ["major", "gender", "club_participation"]

    # 创建预处理流水线：避免虚拟变量共线性，独热编码drop="first"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),  # 数值特征标准化
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features)  # 类别特征独热编码
        ],
        remainder="passthrough"  # 保留未指定的列（防止漏特征）
    )

    return preprocessor, num_features, cat_features