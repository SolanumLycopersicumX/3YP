#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查 GPU 可用性和状态
"""

import torch

print("=" * 50)
print("GPU 检查工具")
print("=" * 50)

# 检查 CUDA 是否可用
print(f"\n1. CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"2. CUDA 版本: {torch.version.cuda}")
    print(f"3. cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"4. GPU 数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n5. GPU {i} 信息:")
        print(f"   名称: {torch.cuda.get_device_name(i)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 检查显存使用情况
        torch.cuda.set_device(i)
        print(f"   当前显存使用: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"   缓存显存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    print(f"\n6. 当前使用的 GPU: {torch.cuda.current_device()}")
    print(f"   设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # 测试 GPU 计算
    print("\n7. GPU 计算测试:")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✓ GPU 计算正常")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ GPU 计算失败: {e}")
else:
    print("\n⚠️  未检测到 CUDA，将使用 CPU 进行训练")
    print("   如果您有 NVIDIA GPU，请检查：")
    print("   1. 是否安装了 NVIDIA 驱动")
    print("   2. 是否安装了 CUDA")
    print("   3. PyTorch 是否支持 CUDA（pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118）")

print("\n" + "=" * 50)
print("检查完成")
print("=" * 50)

