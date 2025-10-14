# PRML-Assignments

模式识别与机器学习 (Pattern Recognition and Machine Learning) 课程作业集

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📚 内容简介

本仓库包含清华大学自动化系《模式识别与机器学习》课程的所有编程作业与实验报告。每个作业独立成文件夹，包含完整的代码实现、数据集、实验日志和 LaTeX 报告。

## 📂 项目结构

```
PRML-Assignments/
├── HW1-SVM/                    # 作业1：支持向量机
├── HW2-XXX/                    # 作业2：待更新
├── HW3-XXX/                    # 作业3：待更新
└── README.md                   # 本文件
```

每个作业文件夹的标准结构：
```
HW*-XXX/
├── code/                       # 源代码
├── data/                       # 数据集（或符号链接）
├── log/                        # 实验日志与可视化结果
├── report/                     # 实验报告（LaTeX + PDF）
└── requirements.pdf            # 作业要求文档
```

## 🚀 快速开始

### 环境配置

```bash
# 推荐使用 Python 3.8+
python --version

# 安装基础依赖
pip install numpy scikit-learn matplotlib pandas
```

### 运行作业

每个作业文件夹包含独立的代码注释，请参考具体作业的文件。

通用运行方式：
```bash
cd HW*-XXX/code
python main.py  # 或其他主程序
```

## 📝 作业列表

### ✅ HW1-SVM：支持向量机分类

**任务**：基于 CelebA 数据集的人脸分类（10 类），对比属性特征与原始像素特征的性能

**关键内容**：
- 线性 SVM 实现与训练
- 超参数 C 值优化（网格搜索）
- 特征标准化与性能对比

**主要结果**：
- 属性特征（40 维）：准确率 **92%**，最优 C = 1.83×10⁻²
- 原始像素（116,412 维）：准确率 64%，最优 C = 2.48×10⁻⁵

**快速运行**：
```bash
cd HW1-SVM/code
python svm.py                    # 训练分类器
python plot_c_curve.py attrs     # C 值优化曲线
```

**报告位置**：`HW1-SVM/report/SVM.pdf`

---

### 🔲 HW2-XXX：待更新

**任务**：TBD

**关键内容**：TBD

**快速运行**：
```bash
# 待补充
```

---

### 🔲 HW3-XXX：待更新

**任务**：TBD

---

## 📊 实验报告规范

所有实验报告统一使用 LaTeX 编写，包含以下标准章节：

1. **实验目的与要求**
2. **实验原理与方法**
3. **代码实现说明**
4. **实验结果与分析**
5. **总结与改进建议**

报告源文件位于 `report/` 目录，可使用以下命令重新编译：
```bash
cd HW*-XXX/report
pdflatex report.tex
```

## 🛠️ 开发环境

- **IDE**: Visual Studio Code
- **Python**: 3.8+ (Anaconda 推荐)
- **主要库**: NumPy, scikit-learn, Matplotlib, PyTorch (部分作业)
- **LaTeX**: TeX Live / MikTeX (中文支持需 ctex 宏包)
- **版本控制**: Git

## 📈 进度追踪

| 作业 | 状态 | 完成日期 | 准确率/得分 |
|-----|------|---------|-----------|
| HW1-SVM | ✅ 已完成 | 2025-10-15 | 92% (属性) / 64% (像素) |
| HW2-XXX | ⏳ 待完成 | - | - |
| HW3-XXX | ⏳ 待完成 | - | - |

## 📝 使用说明

1. **克隆仓库**
   ```bash
   git clone https://github.com/Gotham-Zolio/PRML-Assignments.git
   cd PRML-Assignments
   ```

2. **进入具体作业目录**
   ```bash
   cd HW1-SVM
   ```

3. **按照作业文档运行代码**
   - 查看 `requirements.pdf` 了解作业要求
   - 进入 `code/` 目录运行脚本
   - 查看 `report/*.pdf` 获取详细报告

4. **查看实验结果**
   - 终端输出显示准确率等指标
   - `log/` 目录保存可视化图表
   - 报告中包含完整的结果分析

## ⚠️ 注意事项

- **数据集**：部分作业使用的数据集较大，未上传至仓库，请根据 `README` 自行下载
- **路径问题**：代码中使用相对路径，请确保在正确目录下运行
- **依赖安装**：不同作业可能需要额外的库，请参考各作业的说明文档
- **LaTeX 编译**：中文报告需要安装 CTeX 套装或相应字体

## 📧 联系方式

- **Author**: Gotham-Zolio
- **Course**: Pattern Recognition and Machine Learning
- **Institution**: Tsinghua University, Department of Automation
- **Academic Year**: 2024-2025

## 📜 许可与声明

本项目仅用于学习和教学目的，请勿用于：
- 商业用途
- 学术不端（抄袭、代写等）
- 未经授权的传播

欢迎学习交流，严禁直接抄袭。

---

**Last Updated**: 2025-10-15
