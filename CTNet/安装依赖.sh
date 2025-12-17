#!/bin/bash
# CTNet 依赖安装脚本

echo "========================================="
echo "开始安装 CTNet 依赖包"
echo "========================================="

# 如果使用 GPU，需要先安装 PyTorch (CUDA版本)
# 请根据您的 CUDA 版本选择合适的安装命令
# 访问 https://pytorch.org/get-started/locally/ 查看最新的安装命令

# 安装基础依赖（如果有 GPU 版本 PyTorch，跳过 torch 相关）
echo "安装核心依赖..."
pip install einops>=0.6.0
pip install torchsummary>=1.5.1
pip install scikit-learn>=1.0.0
pip install opencv-python>=4.5.0
pip install mne>=1.5.1
pip install openpyxl>=3.0.0
pip install xlrd>=2.0.0

# 或者一次性安装所有依赖
# pip install -r requirements.txt

echo "========================================="
echo "安装完成！"
echo "========================================="

