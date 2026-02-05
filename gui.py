import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import numpy as np
from PIL import Image, ImageTk

# 路径配置：自动获取项目根目录，添加src到环境变量
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# 导入预测函数，添加友好错误提示
try:
    from predict import predict_gpa
except ImportError as e:
    messagebox.showerror("导入错误", f"无法导入预测模块：{e}\n请确认src/predict.py文件存在且完整")
    sys.exit(1)

class GPAPredictGUI:
    """学生绩点预测GUI主类：分特征输入区、结果展示区、图表查看区"""
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 学生绩点（GPA）预测系统")
        self.root.geometry("1200x800")  # 窗口初始大小
        self.root.resizable(True, True)  # 支持窗口缩放

        # 初始化ttk样式（包含LabelFrame+所有Label的样式，解决所有参数报错）
        self._init_ttk_style()

        # 评估图表路径（与main.py生成的一致）
        self.chart_paths = {
            "模型性能对比": os.path.join(PROJECT_ROOT, "results", "model_r2_comparison.png"),
            "特征重要性Top10": os.path.join(PROJECT_ROOT, "results", "feature_importance.png"),
            "真实vs预测绩点": os.path.join(PROJECT_ROOT, "results", "pred_vs_true.png")
        }

        # 初始化所有界面组件
        self._init_widgets()

    def _init_ttk_style(self):
        """【最终修复】自定义所有ttk组件样式，解决LabelFrame/Label的参数报错问题"""
        self.style = ttk.Style(self.root)
        # 1. 配置LabelFrame样式：框架内边距
        self.style.configure("Custom.TLabelframe", padding=10)
        # 2. 配置LabelFrame标题样式：微软雅黑12号加粗
        self.style.configure(
            "Custom.TLabelframe.Label",
            font=("微软雅黑", 12, "bold"),
            foreground="#333333"
        )
        # 3. 配置【普通标签】样式：微软雅黑10号（特征名称、绩点说明等）
        self.style.configure(
            "Normal.TLabel",
            font=("微软雅黑", 10),
            foreground="#333333"
        )
        # 4. 配置【提示标签】样式：微软雅黑8号+灰色（输入框范围提示）
        self.style.configure(
            "Hint.TLabel",
            font=("微软雅黑", 8),
            foreground="#888888"  # 灰色，替代原fg="gray"
        )
        # 5. 配置【大标题标签】样式：微软雅黑14号（绩点标题、详情标题等）
        self.style.configure(
            "Title.TLabel",
            font=("微软雅黑", 14),
            foreground="#333333"
        )
        # 6. 配置【绩点数值标签】样式：微软雅黑40号加粗+红色
        self.style.configure(
            "Gpa.TLabel",
            font=("微软雅黑", 40, "bold"),
            foreground="#E74C3C"
        )
        # 7. 配置【评级标签】样式：微软雅黑16号+绿色
        self.style.configure(
            "Rating.TLabel",
            font=("微软雅黑", 16),
            foreground="#27AE60"
        )
        # 8. 配置预测按钮样式：微软雅黑12号+内边距
        self.style.configure(
            "Accent.TButton",
            font=("微软雅黑", 12),
            padding=8
        )
        # 9. 配置下拉框样式：微软雅黑10号
        self.style.configure(
            "Custom.TCombobox",
            font=("微软雅黑", 10)
        )

    def _init_widgets(self):
        """初始化界面：标题+输入区+结果区+图表区（所有ttk组件用style，无原生参数）"""
        # 顶部主标题（用tk.Label，支持自由设置字体/颜色，无兼容问题）
        tk.Label(
            self.root, text="学生绩点（GPA）预测系统",
            font=("微软雅黑", 20, "bold"), foreground="#2E86AB"
        ).pack(pady=10)

        # 主容器：左右分栏（输入区+结果区）
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 左侧：特征输入区（应用自定义LabelFrame样式）
        input_frame = ttk.LabelFrame(
            main_frame,
            text="📝 学生特征输入",
            style="Custom.TLabelframe"
        )
        main_frame.add(input_frame, weight=2)

        # 特征配置：匹配8个特征，分下拉/单选/数值输入，带范围提示
        self.feat_config = {
            "major": {"type": "combo", "label": "所属学院", "opts": ["人工智能学院", "文学院", "物理与天文学院", "法学院", "教育学部","心理学部","环境学院","体育与运动学院","哲学学院",
          "经济与工商管理学院","马克思主义学院","社会学院","外国语言文学学院","新闻传播学院","历史学院","数学科学学院","化学学院","地理科学学部",
          "统计学院","生命科学学院","政府管理学院","艺术与传媒学院"]},
            "gender": {"type": "radio", "label": "性别", "opts": [(0, "女"), (1, "男")]},
            "attendance": {"type": "entry", "label": "出勤次数", "hint": "20-32", "dtype": "int"},
            "homework_completion": {"type": "entry", "label": "作业完成率", "hint": "0.6-1.0", "dtype": "float"},
            "lib_borrow": {"type": "entry", "label": "周均借阅量", "hint": "0-10", "dtype": "int"},
            "club_participation": {"type": "radio", "label": "是否参加社团", "opts": [(0, "否"), (1, "是")]},
            "class_interaction": {"type": "entry", "label": "课堂互动次数", "hint": "0-20", "dtype": "int"},
            "exam_score": {"type": "entry", "label": "测试成绩", "hint": "60-100", "dtype": "int"}
        }

        # 动态生成输入组件，网格布局
        self.input_vars = {}
        row = 0
        for feat, cfg in self.feat_config.items():
            # 特征名称标签：应用【普通标签】样式（移除font参数）
            ttk.Label(input_frame, text=cfg["label"], style="Normal.TLabel").grid(
                row=row, column=0, padx=10, pady=8, sticky=tk.W
            )
            # 下拉框（学院）：应用自定义下拉框样式
            if cfg["type"] == "combo":
                var = tk.StringVar(value=cfg["opts"][0])
                combo = ttk.Combobox(
                    input_frame, textvariable=var, values=cfg["opts"],
                    state="readonly", width=20, style="Custom.TCombobox"
                )
                combo.grid(row=row, column=1, padx=5, pady=8)
                self.input_vars[feat] = var
            # 单选框（性别/社团）
            elif cfg["type"] == "radio":
                var = tk.IntVar(value=cfg["opts"][0][0])
                radio_frame = ttk.Frame(input_frame)
                radio_frame.grid(row=row, column=1, padx=5, pady=8, sticky=tk.W)
                for val, txt in cfg["opts"]:
                    ttk.Radiobutton(radio_frame, text=txt, variable=var, value=val).pack(side=tk.LEFT, padx=5)
                self.input_vars[feat] = var
            # 数值输入框+范围提示
            elif cfg["type"] == "entry":
                var = tk.StringVar()
                entry = ttk.Entry(input_frame, textvariable=var, width=22)
                entry.grid(row=row, column=1, padx=5, pady=8)
                # 范围提示标签：应用【提示标签】样式（移除font/fg参数）
                ttk.Label(input_frame, text=cfg["hint"], style="Hint.TLabel").grid(
                    row=row, column=2, padx=2, pady=8, sticky=tk.W
                )
                # 输入验证：仅允许数字
                vcmd = self.root.register(lambda s, f=feat: self._validate_input(s, self.feat_config[f]["dtype"]))
                entry.config(validate="key", validatecommand=(vcmd, "%P"))
                self.input_vars[feat] = var
            row += 1

        # 预测按钮（应用自定义按钮样式）
        predict_btn = ttk.Button(
            input_frame, text="🚀 一键预测绩点", command=self._predict,
            style="Accent.TButton", width=20
        )
        predict_btn.grid(row=row, column=0, columnspan=3, pady=20)

        # 右侧：结果展示区（应用自定义LabelFrame样式）
        result_frame = ttk.LabelFrame(
            main_frame,
            text="📊 预测结果展示",
            style="Custom.TLabelframe"
        )
        main_frame.add(result_frame, weight=3)

        # 预测绩点说明：应用【大标题标签】样式
        ttk.Label(result_frame, text="预测绩点（GPA）：", style="Title.TLabel").pack(pady=10)
        # 预测绩点数值：应用【绩点数值标签】样式（红色大字体）
        self.gpa_var = tk.StringVar(value="——")
        ttk.Label(result_frame, textvariable=self.gpa_var, style="Gpa.TLabel").pack(pady=5)
        # 学业评级：应用【评级标签】样式（绿色字体）
        self.rating_var = tk.StringVar(value="——")
        ttk.Label(result_frame, textvariable=self.rating_var, style="Rating.TLabel").pack(pady=10)

        # 输入特征详情标题：应用【大标题标签】样式
        ttk.Label(result_frame, text="📋 输入特征详情", style="Title.TLabel").pack(pady=5, anchor=tk.W, padx=10)
        # 特征详情文本框（tk.Text，支持自由设置字体）
        self.detail_text = tk.Text(result_frame, height=10, width=40, font=("微软雅黑", 10))
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.detail_text.config(state=tk.DISABLED)

        # 底部：图表查看区（应用自定义LabelFrame样式）
        chart_frame = ttk.LabelFrame(
            self.root,
            text="📸 模型评估图表查看",
            style="Custom.TLabelframe"
        )
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 图表选择下拉框：应用自定义下拉框样式
        self.chart_var = tk.StringVar(value=list(self.chart_paths.keys())[0])
        chart_combo = ttk.Combobox(
            chart_frame, textvariable=self.chart_var, values=list(self.chart_paths.keys()),
            state="readonly", style="Custom.TCombobox", width=25
        )
        chart_combo.pack(pady=5)
        chart_combo.bind("<<ComboboxSelected>>", self._show_chart)

        # 图表画布+滚动条（支持缩放/滚动）
        self.chart_canvas = tk.Canvas(chart_frame, bg="white", bd=1, relief=tk.SUNKEN)
        x_scroll = ttk.Scrollbar(chart_frame, orient=tk.HORIZONTAL, command=self.chart_canvas.xview)
        y_scroll = ttk.Scrollbar(chart_frame, orient=tk.VERTICAL, command=self.chart_canvas.yview)
        self.chart_canvas.config(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 初始化显示第一张图表
        self._show_chart()

    def _validate_input(self, value, dtype):
        """输入验证：仅允许整数/浮点数，空值暂时允许"""
        if not value:
            return True
        try:
            if dtype == "int":
                int(value)
            elif dtype == "float":
                float(value)
            return True
        except ValueError:
            return False

    def _get_input_data(self):
        """获取并校验输入数据：类型转换+范围校验，返回合规的特征字典"""
        input_data = {}
        try:
            # 遍历获取所有输入值
            for feat, cfg in self.feat_config.items():
                var = self.input_vars[feat]
                input_data[feat] = var.get()

            # 数值特征类型转换+范围校验（按业务规则）
            range_checks = {
                "attendance": (int, 20, 32),
                "homework_completion": (float, 0.6, 1.0),
                "lib_borrow": (int, 0, 10),
                "class_interaction": (int, 0, 20),
                "exam_score": (int, 60, 100)
            }
            for feat, (dtype, min_val, max_val) in range_checks.items():
                val = dtype(input_data[feat])
                if not (min_val <= val <= max_val):
                    raise ValueError(f"{self.feat_config[feat]['label']}必须在{min_val}-{max_val}之间")
                input_data[feat] = val

            return input_data
        except ValueError as e:
            messagebox.showerror("输入错误", f"特征输入不合法：{e}\n请检查后重新输入！")
            return None
        except Exception as e:
            messagebox.showerror("数据错误", f"读取输入失败：{e}")
            return None

    def _predict(self):
        """核心预测逻辑：调用predict.py→获取结果→展示"""
        # 1. 获取并校验输入
        input_data = self._get_input_data()
        if not input_data:
            return

        # 2. 调用预测函数
        try:
            pred_gpa = predict_gpa(input_data, PROJECT_ROOT)
        except FileNotFoundError as e:
            messagebox.showerror("模型缺失", f"未找到模型/流水线：{e}\n请先运行python main.py训练模型！")
            return
        except Exception as e:
            messagebox.showerror("预测失败", f"绩点预测出错：{str(e)}")
            return

        # 3. 生成学业评级
        if pred_gpa >= 3.5:
            rating = "优秀（建议申请奖学金）"
        elif pred_gpa >= 2.5:
            rating = "良好（可参与学术竞赛）"
        elif pred_gpa >= 1.5:
            rating = "合格（建议加强作业与出勤）"
        else:
            rating = "需预警（建议联系辅导员辅导）"

        # 4. 更新界面展示
        self.gpa_var.set(f"{pred_gpa}")
        self.rating_var.set(f"学业评级：{rating}")

        # 5. 展示输入特征详情
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        for feat, val in input_data.items():
            self.detail_text.insert(tk.END, f"{self.feat_config[feat]['label']}：{val}\n")
        self.detail_text.config(state=tk.DISABLED)

        # 预测成功提示
        messagebox.showinfo("预测成功", f"✅ 绩点预测完成！\n📌 预测绩点：{pred_gpa}\n🏆 学业评级：{rating}")

    def _show_chart(self, event=None):
        """展示选中的评估图表，自动缩放适配画布，带缺失提示"""
        chart_name = self.chart_var.get()
        chart_path = self.chart_paths[chart_name]

        # 校验图表文件是否存在
        if not os.path.exists(chart_path):
            self.chart_canvas.delete(tk.ALL)
            self.chart_canvas.create_text(
                200, 100, text=f"图表文件不存在！\n请先运行python main.py生成\n路径：{chart_path}",
                font=("微软雅黑", 10), fill="red", anchor=tk.CENTER
            )
            return

        # 加载并缩放图片（保持比例，适配画布）
        try:
            img = Image.open(chart_path)
            canvas_w = self.chart_canvas.winfo_width() - 20
            canvas_h = self.chart_canvas.winfo_height() - 20
            img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)  # 高质量缩放
            photo = ImageTk.PhotoImage(img)

            # 显示图片并保留引用（防止垃圾回收）
            self.chart_canvas.delete(tk.ALL)
            self.chart_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.chart_canvas.image = photo
            self.chart_canvas.config(scrollregion=self.chart_canvas.bbox(tk.ALL))
        except Exception as e:
            self.chart_canvas.delete(tk.ALL)
            self.chart_canvas.create_text(
                200, 100, text=f"加载图表失败：{e}",
                font=("微软雅黑", 10), fill="red", anchor=tk.CENTER
            )

# 程序入口：校验依赖+启动GUI
if __name__ == "__main__":
    # 校验pillow库（图片展示必需）
    try:
        from PIL import Image, ImageTk
    except ImportError:
        if messagebox.askyesno("依赖缺失", "未检测到pillow库（图表展示所需），是否立即安装？"):
            os.system("pip install pillow")
            messagebox.showinfo("安装完成", "pillow安装成功，请重新运行本程序！")
        sys.exit(1)

    # 启动GUI
    root = tk.Tk()
    app = GPAPredictGUI(root)
    root.mainloop()