import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 路径配置：自动获取项目根目录，添加src到环境变量（导入核心模块）
def get_project_paths():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)
    return {
        "root": PROJECT_ROOT, "src": SRC_DIR,
        "models": os.path.join(PROJECT_ROOT, "models"),
        "results": os.path.join(PROJECT_ROOT, "results"),
        "data": os.path.join(PROJECT_ROOT, "data")
    }

# 初始化路径
PATHS = get_project_paths()

# 导入src模块，添加友好的错误提示
try:
    from src.feature_engineering import build_feature_preprocessor
    from src.model_evaluation import evaluate_model, plot_model_comparison, plot_feature_importance, plot_pred_vs_true
    from src.predict import predict_gpa
except ImportError as e:
    raise ImportError(
        f"❌ 导入核心模块失败：{e}\n请确认：\n"
        f"1. src目录存在且包含feature_engineering.py、model_evaluation.py、predict.py\n"
        f"2. 所有文件均为修复后的版本（无Float导入错误）"
    ) from e

# 全局变量：特征预处理流水线（供评估函数使用）
preprocessor = None

def init_directories():
    """初始化项目目录：自动创建models/results/data/docs等，防止文件不存在报错"""
    for dir_path in PATHS.values():
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 项目目录初始化完成：{list(PATHS.keys())}")

def load_and_clean_data():
    """数据加载+预处理：读取csv，缺失值/异常值处理，适配新特征"""
    data_path = os.path.join(PATHS["data"], "simulated_data.csv")
    # 校验数据文件
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 数据文件缺失：{data_path}\n请先运行python data/generate_simulated_data.py生成数据")

    # 读取数据（兼容utf-8-sig/gbk，解决中文乱码）
    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="gbk")

    print(f"✅ 成功读取数据：{data_path}")
    print(f"📊 原始数据规模：{df.shape[0]} 条 × {df.shape[1]} 列")
    print(f"📋 数据列名：{list(df.columns)}")

    # 校验核心列（8特征+1绩点）
    required_cols = [
        "gpa", "major", "gender", "attendance", "homework_completion",
        "lib_borrow", "club_participation", "class_interaction", "exam_score"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ 数据缺失核心列：{missing_cols}")

    # 缺失值处理（本模拟数据无缺失，保留逻辑适配真实数据）
    print("\n🔍 缺失值统计：")
    missing_stats = df.isnull().sum()
    print(missing_stats[missing_stats > 0] if missing_stats.sum() > 0 else "无缺失值")
    if missing_stats.sum() > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())  # 数值特征中位数填充
        cat_cols = df.select_dtypes(include=[object]).columns
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])  # 类别特征众数填充
        print("✅ 缺失值已完成填充")

    # 异常值处理：按业务规则限制范围
    print("\n🔧 异常值处理：")
    clip_rules = {
        "gpa": (1.0, 4.0), "attendance": (20, 32), "homework_completion": (0.6, 1.0),
        "lib_borrow": (0, 10), "class_interaction": (0, 20), "exam_score": (60, 100)
    }
    for col, (min_val, max_val) in clip_rules.items():
        error_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
        if error_count > 0:
            df[col] = df[col].clip(min_val, max_val)
            print(f"   - {col}：修正{error_count}条异常值（范围{min_val}-{max_val}）")
    print("✅ 异常值处理完成")

    # 数据类型转换（确保类别特征为整数）
    df["gender"] = df["gender"].astype(int)
    df["club_participation"] = df["club_participation"].astype(int)
    print("✅ 数据类型转换完成")

    print(f"\n✅ 数据预处理完成！最终规模：{df.shape[0]} 条 × {df.shape[1]} 列")
    return df

def load_data():
    """划分训练集/测试集：特征X + 标签y，测试集占比20%"""
    df = load_and_clean_data()
    X = df.drop("gpa", axis=1)  # 特征矩阵：移除标签列
    y = df["gpa"]               # 标签向量：绩点
    # 随机划分，固定种子保证可复现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("📝 数据集划分完成：")
    print(f"   - 训练集：{X_train.shape[0]} 条")
    print(f"   - 测试集：{X_test.shape[0]} 条")
    return X_train, X_test, y_train, y_test

def train_models(X_train_processed, y_train):
    """训练3个模型：线性回归（基线）、决策树（网格调优）、随机森林（集成）"""
    models = {}
    models_dir = PATHS["models"]

    # 1. 线性回归（基线模型，拟合效果最优）
    print("\n🚀 开始训练线性回归模型（基线）...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_processed, y_train)
    lr_path = os.path.join(models_dir, "linear_regression.pkl")
    joblib.dump(lr_model, lr_path)
    print(f"✅ 线性回归模型已保存：{lr_path}")
    models["线性回归"] = {"model": lr_model, "path": lr_path}

    # 2. 决策树（网格搜索调优核心参数）
    print("\n🚀 开始训练决策树模型（网格搜索调优）...")
    dt_params = {"max_depth": [5, 10, 15], "min_samples_split": [10, 20, 30]}
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_grid = GridSearchCV(dt_model, dt_params, cv=5, scoring="r2", n_jobs=-1)
    dt_grid.fit(X_train_processed, y_train)
    best_dt = dt_grid.best_estimator_
    print(f"🔧 决策树最优参数：{dt_grid.best_params_}")
    dt_path = os.path.join(models_dir, "decision_tree.pkl")
    joblib.dump(best_dt, dt_path)
    print(f"✅ 决策树模型已保存：{dt_path}")
    models["决策树"] = {"model": best_dt, "path": dt_path}

    # 3. 随机森林（集成学习，优化参数提升拟合）
    print("\n🚀 开始训练随机森林模型（集成学习）...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_processed, y_train)
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"✅ 随机森林模型已保存：{rf_path}")
    models["随机森林"] = {"model": rf_model, "path": rf_path}

    return models

def evaluate_models(models, X_test_processed, y_test):
    """评估模型性能，生成3张评估图表"""
    print("\n📊 步骤4：模型评估")
    print("-" * 30)
    metrics_list = []
    # 预测各模型结果
    y_pred_lr = models["线性回归"]["model"].predict(X_test_processed)
    y_pred_dt = models["决策树"]["model"].predict(X_test_processed)
    y_pred_rf = models["随机森林"]["model"].predict(X_test_processed)
    # 计算评估指标
    metrics_list.append(evaluate_model(y_test, y_pred_lr, "线性回归"))
    metrics_list.append(evaluate_model(y_test, y_pred_dt, "决策树"))
    metrics_list.append(evaluate_model(y_test, y_pred_rf, "随机森林"))
    # 生成可视化图表
    plot_model_comparison(metrics_list, PATHS["root"])
    plot_feature_importance(models["随机森林"]["model"], preprocessor, PATHS["root"])
    plot_pred_vs_true(y_test, y_pred_lr, PATHS["root"])  # 用线性回归结果绘图（拟合最好）
    return metrics_list

def run_prediction_example():
    """运行示例预测：2个学生案例，展示预测效果"""
    print("\n🎓 步骤5：学生绩点预测示例")
    print("-" * 30)
    # 示例1：优秀学生（高特征值）
    student1 = {
        "major": "人工智能学院", "gender": 1, "attendance": 30, "homework_completion": 0.98,
        "lib_borrow": 4, "club_participation": 1, "class_interaction": 18, "exam_score": 95
    }
    # 示例2：学业预警学生（低特征值）
    student2 = {
        "major": "文学院", "gender": 0, "attendance": 22, "homework_completion": 0.65,
        "lib_borrow": 6, "club_participation": 0, "class_interaction": 2, "exam_score": 62
    }
    # 预测并打印结果
    try:
        pred1 = predict_gpa(student1, PATHS["root"])
        pred2 = predict_gpa(student2, PATHS["root"])
        # 评级逻辑
        def get_rating(gpa):
            if gpa >= 3.5: return "优秀（建议申请奖学金）"
            elif gpa >= 2.5: return "良好（可参与学术竞赛）"
            elif gpa >= 1.5: return "合格（建议加强作业与出勤）"
            else: return "需预警（建议联系辅导员辅导）"
        # 打印示例1
        print("\n👨‍🎓 示例学生1预测：")
        print("-" * 40)
        for k, v in student1.items(): print(f"{k}: {v}")
        print(f"🎯 预测绩点：{pred1} | 🏆 评级：{get_rating(pred1)}")
        # 打印示例2
        print("\n👩‍🎓 示例学生2预测：")
        print("-" * 40)
        for k, v in student2.items(): print(f"{k}: {v}")
        print(f"🎯 预测绩点：{pred2} | 🏆 评级：{get_rating(pred2)}")
    except Exception as e:
        print(f"⚠️ 示例预测失败：{str(e)}")

def main():
    """项目主流程：初始化→加载数据→特征工程→训练模型→评估→示例预测"""
    global preprocessor
    print("=" * 50)
    print("🎯 学生绩点（GPA）预测项目 - 一键运行流程")
    print("=" * 50)
    # 1. 初始化目录
    init_directories()
    # 2. 加载并划分数据
    print("\n📥 步骤1：数据加载与预处理")
    print("-" * 30)
    X_train, X_test, y_train, y_test = load_data()
    # 3. 特征工程
    print("\n🔧 步骤2：特征工程")
    print("-" * 30)
    preprocessor, _, _ = build_feature_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)  # 训练集拟合+转换
    X_test_processed = preprocessor.transform(X_test)        # 测试集仅转换
    # 保存特征流水线
    preprocessor_path = os.path.join(PATHS["models"], "feature_preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✅ 特征工程流水线已保存：{preprocessor_path}")
    print(f"✅ 特征转换完成！处理后特征维度：{X_train_processed.shape[1]}")
    # 4. 训练模型
    print("\n🚀 步骤3：模型训练")
    print("-" * 30)
    models = train_models(X_train_processed, y_train)
    # 5. 评估模型
    evaluate_models(models, X_test_processed, y_test)
    # 6. 示例预测
    run_prediction_example()
    # 完成提示
    print("\n🎉 项目全流程运行完成！")
    print(f"📁 模型文件路径：{PATHS['models']}")
    print(f"📁 评估图表路径：{PATHS['results']}")
    print(f"🖥️  可视化GUI运行：python gui.py")

if __name__ == "__main__":
    # 强制使用项目根目录为工作目录，防止路径错误
    os.chdir(PATHS["root"])
    # 全局异常捕获，友好提示
    try:
        main()
    except Exception as e:
        print(f"\n❌ 项目运行失败：{str(e)}")
        sys.exit(1)