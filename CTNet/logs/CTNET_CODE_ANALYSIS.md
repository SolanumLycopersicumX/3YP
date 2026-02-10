# CTNet 代码分析报告

> 分析目的：理解 CTNet 的数据处理流程，验证数据划分是否正确
> 
> 分析时间：2026-02-10

---

## 目录

1. [整体架构](#1-整体架构)
2. [数据加载流程](#2-数据加载流程)
3. [数据划分策略](#3-数据划分策略)
4. [数据处理细节](#4-数据处理细节)
5. [潜在问题与建议](#5-潜在问题与建议)
6. [关键代码位置](#6-关键代码位置)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CTNet 训练流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  .mat 文件   │ → │  数据加载    │ → │  归一化     │             │
│  │  (A01T.mat) │    │  utils.py   │    │  z-score    │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         ↓                                     ↓                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  训练集     │ ← │  划分       │ ← │  打乱       │             │
│  │  (90%)      │    │  Train/Val  │    │  shuffle    │             │
│  │  验证集     │    │             │    │             │             │
│  │  (10%)      │    │             │    │             │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         ↓                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  数据增强   │ → │  模型训练    │ → │  评估       │             │
│  │  interaug() │    │  CTNet      │    │  Val + Test │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据加载流程

### 2.1 数据文件结构

```
mymat_raw/
├── A01T.mat    # Subject 1, Session 1 (Training)
├── A01E.mat    # Subject 1, Session 2 (Evaluation)
├── A02T.mat    # Subject 2, Session 1
├── A02E.mat    # Subject 2, Session 2
├── ...
├── A09T.mat    # Subject 9, Session 1
├── A09E.mat    # Subject 9, Session 2
├── B01T.mat    # Dataset B, Subject 1, Session 1
├── B01E.mat    # Dataset B, Subject 1, Session 2
└── ...
```

### 2.2 数据格式

| 数据集 | 数据形状 | 标签范围 | 采样率 |
|--------|----------|----------|--------|
| **IV-2a** | (288, 22, 1000) | 1-4 | 250 Hz |
| **IV-2b** | (?, 3, 1000) | 1-2 | 250 Hz |

- `288`: 每个 session 的 trial 数量
- `22`: EEG 通道数
- `1000`: 时间点 (4秒 × 250Hz)

### 2.3 加载函数

```python
# utils.py: load_data_evaluate()

def load_data_evaluate(dir_path, dataset_type, n_sub, mode_evaluate="LOSO"):
    if mode_evaluate == "LOSO":
        return load_data_LOSO(...)      # 跨被试
    else:
        return load_data_subject_dependent(...)  # 被试内
```

**两种评估模式：**

| 模式 | 训练数据 | 测试数据 | 用途 |
|------|----------|----------|------|
| **Subject-dependent** | 同一被试 Session 1 | 同一被试 Session 2 | 被试特定模型 |
| **LOSO** | 8个被试的所有数据 | 1个被试的所有数据 | 跨被试泛化 |

---

## 3. 数据划分策略

### 3.1 当前划分方式

```python
# CTNet_model.py: train() 方法

# 第一步：加载数据
train_data, train_label, test_data, test_label = load_data_evaluate(...)

# 第二步：从训练数据中划分验证集
dataset_size = len(img)
val_size = int(self.validate_ratio * dataset_size)  # 默认 10%
train_size = dataset_size - val_size

indices = torch.randperm(dataset_size)  # 随机打乱
train_indices = indices[:train_size]
val_indices = indices[train_size:]
```

### 3.2 数据划分图示

```
┌─────────────────────────────────────────────────────────────────┐
│                    Subject-dependent 模式                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Session 1 (A01T.mat)              Session 2 (A01E.mat)        │
│  ┌───────────────────────┐         ┌───────────────────────┐   │
│  │    288 trials         │         │    288 trials         │   │
│  │                       │         │                       │   │
│  │  ┌─────────────────┐  │         │     ┌─────────────┐   │   │
│  │  │   训练集        │  │         │     │  测试集     │   │   │
│  │  │   (90%)         │  │         │     │  (100%)     │   │   │
│  │  │   ~259 trials   │  │         │     │  288 trials │   │   │
│  │  └─────────────────┘  │         │     └─────────────┘   │   │
│  │  ┌─────────────────┐  │         │                       │   │
│  │  │   验证集        │  │         │     完全独立的        │   │
│  │  │   (10%)         │  │         │     Session!          │   │
│  │  │   ~29 trials    │  │         │                       │   │
│  │  └─────────────────┘  │         │                       │   │
│  └───────────────────────┘         └───────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 ✅ 正确之处

1. **测试集完全独立**
   - 测试数据来自不同的 Session（不同天采集）
   - 不存在数据泄露风险

2. **验证集从训练集划分**
   - 验证集用于模型选择和早停
   - 不参与训练

3. **归一化参数计算正确**
   ```python
   # 只用训练数据计算统计量
   target_mean = np.mean(self.allData)
   target_std = np.std(self.allData)
   
   # 测试数据用相同参数归一化
   self.testData = (self.testData - target_mean) / target_std
   ```

---

## 4. 数据处理细节

### 4.1 数据增强 (interaug)

```python
def interaug(self, timg, label):
    # 段内交换增强 (Segmentation-based Augmentation)
    # 将 1000 个时间点分成多个段
    # 从同类别的不同 trial 中随机选择段进行组合
    
    number_segmentation_points = 1000 // self.number_seg  # 默认 8 段
    
    for clsAug in range(self.number_class):
        cls_idx = np.where(label == clsAug + 1)  # 找到该类别的所有样本
        # ... 随机组合不同样本的段
```

**增强策略：**
- 只对训练数据应用
- 按类别分别增强
- 保持类别平衡

### 4.2 标签处理

```python
# 原始标签: 1, 2, 3, 4
# 转换为: 0, 1, 2, 3 (CrossEntropyLoss 要求)

label = torch.from_numpy(label - 1)  # 关键转换
```

⚠️ **注意**：这个 `-1` 操作分散在代码多处，需要注意一致性。

### 4.3 训练循环

```python
for e in range(self.n_epochs):
    # 训练阶段
    for img, label in train_dataloader:
        aug_data, aug_label = self.interaug(...)  # 数据增强
        img_aug = torch.cat((img, aug_data))       # 合并原始+增强
        outputs = self.model(img_aug)
        loss = criterion(outputs, label_aug)
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    for img, label in val_dataloader:
        outputs = self.model(img)
        val_loss = criterion(outputs, label)
    
    # 早停检查
    if val_loss < best_val_loss:
        save_model()
    elif epochs_without_improvement >= patience:
        break
```

---

## 5. 潜在问题与建议

### 5.1 ⚠️ 问题：验证集划分没有固定种子

```python
# 当前代码
indices = torch.randperm(dataset_size)  # 每次运行不同！

# 建议修改
indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42))
```

**影响**：每次运行验证集不同，结果不完全可复现

### 5.2 ⚠️ 问题：没有 K-Fold 选项

当前只支持单次 Train/Val 划分，没有 K-Fold 交叉验证。

**建议**：添加 K-Fold CV 选项（已在 `cross_validation.py` 中实现）

### 5.3 ⚠️ 问题：数据增强可能有轻微泄露

```python
# interaug() 使用 self.allData，包含了验证集数据
aug_data, aug_label = self.interaug(self.allData, self.allLabel)
```

**影响**：轻微，因为只用于生成增强数据，不直接用于训练
**建议**：严格来说应该只用训练集数据做增强

### 5.4 ✅ 正确之处

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 测试集独立 | ✅ | 来自不同 Session |
| 归一化无泄露 | ✅ | 只用训练集计算参数 |
| 验证集不参与训练 | ✅ | 只用于模型选择 |
| 早停机制 | ✅ | 避免过拟合 |
| 最佳模型保存 | ✅ | 选择验证损失最低的 |

---

## 6. 关键代码位置

| 功能 | 文件 | 函数/行号 |
|------|------|-----------|
| 数据加载 | `utils.py` | `load_data_evaluate()` L28 |
| Subject-dependent 划分 | `utils.py` | `load_data_subject_dependent()` L55 |
| LOSO 划分 | `utils.py` | `load_data_LOSO()` L74 |
| 数据归一化 | `CTNet_model.py` | `get_source_data()` L363-370 |
| Train/Val 划分 | `CTNet_model.py` | `train()` L384-397 |
| 数据增强 | `CTNet_model.py` | `interaug()` L315-342 |
| 训练循环 | `CTNet_model.py` | `train()` L485-615 |
| 评估 | `CTNet_model.py` | `train()` L630-656 |
| Channel 配置 | `utils.py` | `numberClassChannel()` L227 |

---

## 7. 与教授问题的对应

### Q1: 如何划分 Train/Val/Test？

**回答**：
- **训练集**：Session 1 的 90%（约 259 trials）
- **验证集**：Session 1 的 10%（约 29 trials）
- **测试集**：Session 2 的 100%（288 trials）

### Q2: 测试集是否独立？

**回答**：✅ 是的，测试集来自不同的 Session（不同天采集），完全独立。

### Q3: 有没有数据泄露风险？

**回答**：
- 归一化：✅ 无泄露（只用训练数据计算参数）
- 数据增强：⚠️ 轻微风险（使用了包含验证集的 allData）
- 测试集：✅ 无泄露（完全独立的 Session）

---

*分析完成。CTNet 的数据处理总体正确，有一些可改进的小问题。*

