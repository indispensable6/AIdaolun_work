import pandas as pd
import numpy as np
import os

# 确保data目录存在
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA_DIR, exist_ok=True)

# 固定随机种子，结果可复现
np.random.seed(42)

# 生成模拟数据
n_samples = 10000
majors = ["人工智能学院", "文学院", "物理与天文学院", "法学院", "教育学部","心理学部","环境学院","体育与运动学院","哲学学院",
          "经济与工商管理学院","马克思主义学院","社会学院","外国语言文学学院","新闻传播学院","历史学院","数学科学学院","化学学院","地理科学学部",
          "统计学院","生命科学学院","政府管理学院","艺术与传媒学院"]
data = {
    "major": np.random.choice(majors, size=n_samples),
    "gender": np.random.randint(0, 2, size=n_samples),  # 0=女，1=男
    "attendance": np.random.randint(20, 32, size=n_samples),  # 出勤次数20-32
    "homework_completion": np.random.uniform(0.6, 1.0, size=n_samples),  # 作业完成率0.6-1.0
    "lib_borrow": np.random.randint(0, 10, size=n_samples),  # 周均借阅量0-10
    "club_participation": np.random.randint(0, 2, size=n_samples),  # 是否参加社团0/1
    # 新增特征1：课堂互动次数（0-20次）
    "class_interaction": np.random.randint(0, 20, size=n_samples),
    # 新增特征2：期中/期末测试成绩（60-100分）
    "exam_score": np.random.randint(60, 100, size=n_samples)
}

# ========== 全新绩点生成公式 ==========
# 步骤1：计算所有特征的实际均值（精准控制绩点均值=2.5）
mean_attendance = data["attendance"].mean()
mean_homework = data["homework_completion"].mean()
mean_lib = data["lib_borrow"].mean()
mean_club = data["club_participation"].mean()
mean_interaction = data["class_interaction"].mean()
mean_exam = data["exam_score"].mean()

# 步骤2：权重设计（作业占比绝对主导）
w_homework = 2.5       # 作业
w_exam = 0.037         # 测试成绩
w_attendance = 0.05     # 出勤
w_interaction = 0.05   # 课堂互动
w_lib = 0.03           # 借阅量
w_club = 0.05          # 社团

# 步骤3：计算当前线性组合均值，添加偏移量锁死绩点均值=2.5
current_mean = (
    w_attendance * mean_attendance +
    w_homework * mean_homework +
    w_lib * mean_lib +
    w_club * mean_club +
    w_interaction * mean_interaction +
    w_exam * mean_exam
)
bias = 2.5 - current_mean  # 精准偏移量

# 步骤4：生成绩点（低噪声保证拟合，clip限制1.0-4.0）
data["gpa"] = (
    w_attendance * data["attendance"] +
    w_homework * data["homework_completion"] +
    w_lib * data["lib_borrow"] +
    w_club * data["club_participation"] +
    w_interaction * data["class_interaction"] +
    w_exam * data["exam_score"] +
    bias +
    np.random.normal(0, 0.2, size=n_samples)  # 低噪声，保证模型拟合度
).clip(1.0, 4.0)  # 确保绩点覆盖1.0-4.0

# 保存数据
df = pd.DataFrame(data)
save_path = os.path.join(DATA_DIR, "simulated_data.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")

# 验证输出（直观看到绩点分布和新增特征）
print(f"✅ 数据生成完成！")
print(f"📁 保存路径：{save_path}")
print(f"📊 数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
print(f"📈 绩点分布验证：")
print(f"   - 均值：{df['gpa'].mean():.2f}")
print(f"   - 最小值：{df['gpa'].min():.2f}")
print(f"   - 最大值：{df['gpa'].max():.2f}")
print(f"   - 1.0-2.0区间：{len(df[(df['gpa']>=1.0) & (df['gpa']<2.0)])} 条")
print(f"   - 2.0-3.0区间：{len(df[(df['gpa']>=2.0) & (df['gpa']<3.0)])} 条")
print(f"   - 3.0-4.0区间：{len(df[(df['gpa']>=3.0) & (df['gpa']<=4.0)])} 条")
print(f"🔧 权重优先级：作业(2.5) > 测试成绩(0.15) > 出勤(0.1) > 课堂互动(0.08) > 借阅(0.03) > 社团(0.05)")