# BCI-based Robotic Arm Control System
## Final Year Project (3YP) - 徐正 (Zheng XU)

本文件夹包含我在毕业设计中**自己完成的所有工作**，与原始CTNet GitHub项目区分开来。

## 中文版概览

该仓库整理了我在曼彻斯特大学毕业设计期间独立完成的主要工作，主题是基于脑机接口（BCI）的机械臂控制系统。系统将基于 CTNet 的运动想象 EEG 分类、面向目标到达任务的强化学习控制，以及 SO-101 机械臂的仿真到实物部署流程整合在一起。

仓库内容聚焦于我自己的代码、实验结果、训练模型和项目文档，并与原始 CTNet GitHub 项目分开维护。目前结果包括 PhysioNet 运动想象数据集上的 88.78% pooled accuracy、通道缩减与滤波消融实验，以及离线或仿真条件下的端到端评估。由于 NHS 伦理审批限制，真人被试硬件验证尚未完成。

中文快速链接：
- [项目索引](05_Documentation/PROJECT_INDEX.md)
- [最终报告草稿](01_Reports/final_report_draft_v10.pdf)
- [周报](01_Reports/week_2026_04_22_feedback.pdf)
- [代码目录](02_Code/)
- [实验结果目录](03_Experiments/)
- [参考资料](07_References/README.md)

## English Overview

This repository contains the work I completed for my final-year project at The University of Manchester. The project develops a brain-computer interface (BCI) pipeline for robotic arm control by combining CTNet-based motor imagery EEG classification, reinforcement learning for target-reaching control, and simulation-to-real deployment on an SO-101 robotic arm.

This repository is kept separate from the original CTNet GitHub project and focuses on my own code, experiments, trained models, and documentation. Current results include 88.78% pooled accuracy on the PhysioNet motor imagery dataset, channel-reduction and filtering ablation studies, and offline or simulation-based end-to-end evaluation. Due to NHS ethics approval constraints, hardware validation with human subjects has not yet been completed.

Quick links:
- [Project Index](05_Documentation/PROJECT_INDEX.md)
- [Final Report Draft](01_Reports/final_report_draft_v10.pdf)
- [Weekly Report](01_Reports/week_2026_04_22_feedback.pdf)
- [Code Directory](02_Code/)
- [Experiment Outputs](03_Experiments/)
- [References](07_References/README.md)

---

## 📁 文件夹结构

```
3YP/
├── 01_Reports/                    # 报告与文档
│   ├── final_report_draft_v7.*    # 最终报告 (最新版)
│   ├── week_2026_03_24.*          # 周报
│   ├── ZhengXu_CV_v5.*            # 简历 (英文/中文)
│   ├── architecture_diagram.*     # 系统架构图
│   └── Risk assessment form-*.docx
│
├── 02_Code/                       # 代码
│   ├── EEG_Classification/        # EEG分类相关
│   │   ├── CTNet_model.py         # EEGTransformer模型 (基于原CTNet修改)
│   │   ├── physionet_loader.py    # PhysioNet数据加载器
│   │   ├── train_physionet_ctnet.py
│   │   ├── finetune_physionet_ctnet.py  # 微调脚本
│   │   ├── channel_reduction_study.py   # 通道消融实验
│   │   └── finetune_filter_comparison.py # 滤波消融实验
│   │
│   ├── Reinforcement_Learning/    # 强化学习
│   │   ├── dqn_model.py           # DQN模型
│   │   ├── dqn_transformer.py     # Transformer-based DQN
│   │   ├── train_dqn_rl.py        # DQN训练脚本
│   │   ├── ctnet_dqn_e2e_eval.py  # 端到端评估
│   │   └── rl_control_test.py     # RL控制测试
│   │
│   ├── Physical_Control/          # 物理机械臂控制
│   │   ├── serial_arm_env*.py     # SO-101串口控制
│   │   ├── phy_control.py         # 物理控制主程序
│   │   ├── brainflow_physical_control.py  # BrainFlow+物理臂
│   │   ├── openbci_stream.py      # OpenBCI数据流
│   │   └── serial_*.json          # 机械臂位置配置
│   │
│   ├── Simulation/                # PyBullet仿真
│   │   ├── pybullet_arm_env.py    # PyBullet环境
│   │   ├── arm_gym_env.py         # Gymnasium包装
│   │   └── gym_control.py         # 仿真控制
│   │
│   ├── Utils/                     # 工具脚本
│   │   ├── requirements.txt
│   │   └── plot_*.py
│   │
│   └── lerobot/                   # LeRobot框架 (HuggingFace, 备用)
│
├── 03_Experiments/                # 实验结果
│   ├── Channel_Reduction/         # 64→8通道消融
│   ├── Filter_Ablation/           # 8-30Hz滤波消融 (+18.44%)
│   ├── DQN_Training/              # DQN训练曲线
│   └── E2E_Evaluation/            # CTNet+DQN端到端评估
│
├── 04_Trained_Models/             # 训练好的模型
│   ├── physionet_ctnet*/          # PhysioNet预训练CTNet
│   └── dqn_*.pth                  # DQN策略网络
│
├── 05_Documentation/              # 文档与演示
│   ├── phy_control_arm.mp4        # 物理机械臂演示视频
│   └── gym_arm.gif                # 仿真演示
│
└── 06_Data/                       # 数据集
    ├── BCICIV_2a_gdf/             # BCI Competition IV-2a (575MB)
    ├── BCICIV_2b_gdf/             # BCI Competition IV-2b (272MB)
    ├── physionet_raw/             # PhysioNet EEG (1.8GB)
    └── true_labels/               # 标签文件
```

---

## 🔬 主要贡献 (我自己的工作)

### 1. EEG分类 (基于原CTNet改进)
- **PhysioNet数据集适配**: 109人跨被试预训练，达到88.78%准确率
- **两阶段迁移学习**: 预训练+微调，提升32个百分点
- **通道消融研究**: 64→8通道，保持72.54%准确率，识别C3为关键通道
- **滤波消融实验**: 证明8-30Hz带通滤波提升+18.44%准确率

### 2. 强化学习控制
- **Transformer-based DQN**: 设计用于序列决策的DQN架构
- **EEG-aware RL**: 将EEG分类结果作为状态输入
- **闭环误差补偿**: 82%分类准确率下达到99%目标到达率

### 3. 物理机械臂集成
- **SO-101串口控制**: 实现多关节同步运动、速度规划
- **Sim2Real管线**: PyBullet仿真→物理臂零样本迁移
- **OpenBCI集成**: BrainFlow API实时脑电采集

### 4. 实验与评估
- 系统的消融实验设计
- 完整的端到端离线评估管线
- 详细的实验报告和可视化

---

## ⚠️ 注意事项

1. **原始CTNet项目文件**保留在 `CTNet/` 文件夹中，包括：
   - 原始Jupyter notebooks
   - BCI IV-2a/2b数据集和标签
   - 原始README和LICENSE

2. **数据集**已包含在 `06_Data/` 文件夹中：
   - PhysioNet: `06_Data/physionet_raw/` (1.8GB, 109人)
   - BCI IV-2a: `06_Data/BCICIV_2a_gdf/` (575MB, 9人, 4分类)
   - BCI IV-2b: `06_Data/BCICIV_2b_gdf/` (272MB, 9人, 2分类)

3. **硬件验证状态**：由于NHS伦理审批限制，所有评估均为仿真/离线，真人被试测试待完成。

---

## 🚀 快速开始

```bash
# 安装依赖
pip install -r 02_Code/Utils/requirements.txt

# 运行PhysioNet CTNet训练
python 02_Code/EEG_Classification/train_physionet_ctnet.py

# 运行通道消融实验
python 02_Code/EEG_Classification/channel_reduction_study.py

# 运行仿真控制
python 02_Code/Simulation/gym_control.py
```

---

**作者**: 徐正 (Zheng XU)  
**项目**: 曼彻斯特大学 Final Year Project  
**日期**: 2025-2026学年
