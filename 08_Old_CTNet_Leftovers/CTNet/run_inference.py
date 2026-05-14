#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速运行单次推理脚本
直接修改下方配置即可运行
"""

from inference import run_inference
import os

# ============================================================
# 配置参数 - 修改此处
# ============================================================

# 模型文件路径（选择一个已训练的模型）
MODEL_PATH = './A_heads_2_depth_6_full_a_1762043947/model_1.pth'  # 修改为您的模型路径

# 数据目录
DATA_DIR = './mymat_raw/'

# 数据集类型: 'A' (BCI IV-2a, 4类) 或 'B' (BCI IV-2b, 2类)
DATASET_TYPE = 'A'

# 受试者ID (1-9)
SUBJECT_ID = 1

# 结果保存目录
SAVE_DIR = './inference_results/'

# 是否显示图形
SHOW_PLOT = True

# ============================================================
# 运行推理
# ============================================================

if __name__ == "__main__":
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("\n可用的模型文件:")
        
        # 列出可能的模型文件
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.pth'):
                    print(f"  {os.path.join(root, f)}")
        
        print("\n请修改 MODEL_PATH 为正确的模型路径")
        exit(1)
    
    # 运行推理
    results = run_inference(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        dataset_type=DATASET_TYPE,
        subject_id=SUBJECT_ID,
        save_dir=SAVE_DIR,
        show_plot=SHOW_PLOT
    )
    
    print("\n推理完成!")
    print(f"准确率: {results['accuracy']:.2%}")

