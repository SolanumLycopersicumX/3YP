# CTNet_model.py 使用指南

## 概述

`CTNet_model.py` 是一个基于卷积 Transformer 网络的脑机接口(BCI) EEG 信号分类模型，用于运动想象(Motor Imagery)任务。该模型结合了 CNN (改进版 EEGNet) 和 Transformer 编码器，能够有效提取 EEG 信号的局部特征和全局依赖关系。

## 主要功能

1. **模型定义**：实现了 EEGTransformer 模型架构
2. **训练流程**：完整的训练、验证和测试流程
3. **数据增强**：支持分段混合数据增强方法
4. **评估模式**：支持 subject-dependent 和 LOSO 两种评估模式
5. **结果保存**：自动保存训练过程、预测结果和评估指标

## 数据要求

### 数据格式
- 数据文件应放在 `mymat_raw/` 目录下
- 文件命名格式：`{dataset_type}{subject_id:02d}{mode}.mat`
  - `dataset_type`: 'A' (BCI IV-2a) 或 'B' (BCI IV-2b)
  - `subject_id`: 受试者编号 (1-9)
  - `mode`: 'T' (训练) 或 'E' (测试)

### 示例文件名
- `A01T.mat` - 数据集 A，受试者 1，训练数据
- `A01E.mat` - 数据集 A，受试者 1，测试数据
- `B03T.mat` - 数据集 B，受试者 3，训练数据

### 数据内容
MAT 文件应包含：
- `data`: shape 为 (样本数, 通道数, 1000) 的 numpy 数组
- `label`: 对应的标签数组

## 使用方法

### 方法一：直接运行脚本（使用默认参数）

```bash
python CTNet_model.py
```

默认配置（在文件末尾的 `if __name__ == "__main__"` 部分）：
- 数据集类型：'B'
- 评估模式：'LOSO-No' (subject-dependent)
- 受试者数量：9
- 训练轮数：1000
- Transformer 头数：2
- 嵌入维度：16
- 深度：6

### 方法二：修改配置参数

编辑文件末尾的参数部分：

```python
# 数据路径和基本配置
DATA_DIR = r'./mymat_raw/'          # 数据目录
EVALUATE_MODE = 'LOSO-No'          # 'LOSO-No' 或 'LOSO'
N_SUBJECT = 9                       # 受试者数量
N_AUG = 3                           # 数据增强倍数
N_SEG = 8                           # 分段数量
EPOCHS = 1000                       # 训练轮数

# 模型架构参数
EMB_DIM = 16                        # Transformer 嵌入维度
HEADS = 2                           # 注意力头数
DEPTH = 6                           # Transformer 层数
TYPE = 'A'                          # 'A' (BCI IV-2a) 或 'B' (BCI IV-2b)

# 验证集比例
validate_ratio = 0.3                # 验证集占训练集的比例

# EEGNet 参数
EEGNet1_F1 = 8                      # 第一个卷积层的滤波器数量
EEGNet1_KERNEL_SIZE = 64            # 卷积核大小
EEGNet1_D = 2                       # 深度乘数
EEGNet1_POOL_SIZE1 = 8              # 第一个池化层大小
EEGNet1_POOL_SIZE2 = 8              # 第二个池化层大小
FLATTEN_EEGNet1 = 240               # 展平后的特征维度

# Dropout 率（根据评估模式自动调整）
if EVALUATE_MODE != 'LOSO':
    EEGNet1_DROPOUT_RATE = 0.5      # subject-dependent 模式
else:
    EEGNet1_DROPOUT_RATE = 0.25     # LOSO 模式
```

### 方法三：作为模块导入使用

```python
from CTNet_model import main, EEGTransformer

# 定义配置参数
result = main(
    dirs="experiment_results",          # 结果保存目录
    evaluate_mode='subject-dependent',  # 评估模式
    heads=2,                            # 注意力头数
    emb_size=16,                        # 嵌入维度
    depth=6,                            # Transformer 深度
    dataset_type='A',                   # 数据集类型
    eeg1_f1=8,                          # EEGNet F1 参数
    eeg1_kernel_size=64,                # 卷积核大小
    eeg1_D=2,                           # 深度乘数
    eeg1_pooling_size1=8,               # 池化大小1
    eeg1_pooling_size2=8,               # 池化大小2
    eeg1_dropout_rate=0.5,              # Dropout 率
    flatten_eeg1=240,                   # 展平特征维度
    validate_ratio=0.2                  # 验证集比例
)
```

## 评估模式说明

### 1. Subject-Dependent (受试者内)
- `EVALUATE_MODE = 'LOSO-No'` 或其他非 'LOSO' 的值
- 每个受试者的训练数据和测试数据来自同一受试者
- 适用于受试者特定的分类任务

### 2. Cross-Subject / LOSO (受试者间)
- `EVALUATE_MODE = 'LOSO'`
- Leave-One-Subject-Out 交叉验证
- 每次用 8 个受试者的数据训练，1 个受试者的数据测试
- 适用于跨受试者泛化评估

## 输出结果

训练完成后，会在结果目录下生成以下文件：

1. **model_{subject_id}.pth** - 每个受试者的最佳模型权重
2. **result_metric.xlsx** - 所有受试者的评估指标（准确率、精确率、召回率、F1、Kappa）
3. **process_train.xlsx** - 训练过程中的指标变化（每个受试者一个工作表）
4. **pred_true.xlsx** - 预测结果和真实标签（每个受试者一个工作表）

## 模型架构参数说明

### Transformer 参数
- **heads**: 多头注意力的头数（建议 2-8）
- **emb_size**: 嵌入维度（建议 16-48）
- **depth**: Transformer 编码器层数（建议 3-6）

### EEGNet 参数
- **f1**: 第一层卷积滤波器数量
- **kernel_size**: 时间卷积核大小
- **D**: 深度卷积的深度乘数
- **pooling_size1/2**: 平均池化大小
- **dropout_rate**: Dropout 概率

### 数据集特定参数
- **数据集 A (BCI IV-2a)**: 4 类，22 通道
- **数据集 B (BCI IV-2b)**: 2 类，3 通道

## 训练过程监控

训练过程中会输出：
- 每个 epoch 的训练损失和准确率
- 验证集的损失和准确率
- 最佳模型保存时的信息
- 测试集的最终准确率和 Kappa 值

示例输出：
```
[CTNet][Epoch 1/1000] Step 1/10 loss=1.2345 acc=0.5623
Subject 1 train_acc: 0.7234 train_loss: 0.654321	val_acc: 0.712345 val_loss: 0.7654321
epoch: 50	The test accuracy is: 0.8234
```

## 依赖要求

主要依赖库：
- Python 3.10
- PyTorch 1.13.1
- numpy, pandas
- scipy (用于读取 .mat 文件)
- einops (用于张量重排)
- torchsummary (可选，用于模型摘要)

## 注意事项

1. **GPU 使用**：代码会自动检测并使用 GPU（如果可用）
2. **随机种子**：每个受试者训练时会设置随机种子以确保可重复性
3. **内存管理**：训练过程中会自动清理 GPU 缓存
4. **数据归一化**：训练数据会自动进行 Z-score 归一化
5. **模型保存**：只保存验证损失最低的模型权重

## 常见问题

### Q: 如何修改批次大小和学习率？
A: 在 `ExP` 类的 `__init__` 方法中修改 `batch_size` 和 `learning_rate` 参数，或在调用 `main()` 函数时传递这些参数。

### Q: 如何只训练特定受试者？
A: 修改 `N_SUBJECT` 或在 `main()` 函数中修改循环范围。

### Q: 如何查看模型结构？
A: 设置环境变量 `ENABLE_MODEL_SUMMARY=1` 或直接修改代码启用模型摘要。

### Q: 训练中断后如何继续？
A: 当前代码不支持断点续训，需要重新开始训练。可以修改代码添加检查点保存和加载功能。

## 示例：完整训练流程

```python
# 1. 确保数据文件在正确位置
# ./mymat_raw/A01T.mat, A01E.mat, A02T.mat, ...

# 2. 修改配置参数（可选）
# 编辑 CTNet_model.py 文件末尾的参数部分

# 3. 运行训练
python CTNet_model.py

# 4. 查看结果
# 结果保存在类似 "CTNet_A_heads_2_depth_6_1765333879/" 的目录中
```

## 性能调优建议

1. **Subject-Dependent 模式**：
   - 可以使用较小的 dropout (0.25-0.3)
   - 可以增加训练轮数
   - 可以增加数据增强倍数

2. **LOSO 模式**：
   - 建议使用较大的 dropout (0.5)
   - 可以使用更多的训练轮数
   - 注意避免过拟合

3. **超参数搜索**：
   - 可以尝试不同的 heads (2, 4, 8)
   - 可以调整 emb_size (16, 32, 48)
   - 可以修改 depth (3, 4, 6)

## 引用

如果使用本代码，请引用：
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024).

