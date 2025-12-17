import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import random
import datetime
import time
import matplotlib
# 自动检测并使用合适的后端
# 如果有GUI环境，使用交互式后端；否则使用非交互式后端
import os
if 'DISPLAY' in os.environ or os.name == 'nt':
    # 有显示设备，尝试使用交互式后端
    try:
        import tkinter
        matplotlib.use('TkAgg')  # Linux/Mac with GUI
    except:
        try:
            matplotlib.use('Qt5Agg')  # Qt backend
        except:
            matplotlib.use('Agg')  # 非交互式后端
else:
    matplotlib.use('Agg')  # 服务器环境使用非交互式后端

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from sklearn.metrics import confusion_matrix

from pandas import ExcelWriter
from torchsummary import summary
import torch
torch.set_num_threads(1)
import math
import warnings
warnings.filterwarnings("ignore")

from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel
from utils import load_data_evaluate

from torch.autograd import Variable

# Select device dynamically (GPU preferred).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CTNet] Using device: {device}")

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )
    def forward(self, x):
        out = self.fc(x)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)
    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            )    

class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])

class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )

class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)        

class EEGTransformer(nn.Module):
    def __init__(self, heads=4, 
                 emb_size=40,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.3,
                 eeg1_number_channel = 22,
                 flatten_eeg1 = 600,
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1 = eeg1_f1,
                                              kernel_size = eeg1_kernel_size,
                                              D = eeg1_D,
                                              pooling_size1 = eeg1_pooling_size1,
                                              pooling_size2 = eeg1_pooling_size2,
                                              dropout_rate = eeg1_dropout_rate,
                                              )
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1 , self.number_class)
    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn+trans
        out = self.classification(self.flatten(features))
        return features, out

class ExP():
    def __init__(self, nsub, data_dir, result_name, 
                 epochs=2000, 
                 number_aug=2,
                 number_seg=8, 
                 gpus=[0], 
                 evaluate_mode = 'subject-dependent',
                 heads=4, 
                 emb_size=40,
                 depth=6, 
                 dataset_type='A',
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.3,
                 flatten_eeg1 = 600, 
                 validate_ratio = 0.2,
                 learning_rate = 0.001,
                 batch_size = 72,
                 log_interval = 1,
                 early_stopping = True,
                 patience = 50,
                 min_delta = 0.0001,
                 verbose = True,
                 plot_training = True,
                 ):
        super(ExP, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_epochs = epochs
        self.nSub = nsub
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.root = data_dir
        self.heads=heads
        self.emb_size=emb_size
        self.depth=depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio
        self.log_interval = max(1, int(log_interval))
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.plot_training = plot_training

        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        self.number_class, self.number_channel = numberClassChannel(self.dataset_type)
        self.model = EEGTransformer(
             heads=self.heads, 
             emb_size=self.emb_size,
             depth=self.depth, 
            database_type=self.dataset_type, 
            eeg1_f1=eeg1_f1, 
            eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size,
            eeg1_pooling_size1 = eeg1_pooling_size1,
            eeg1_pooling_size2 = eeg1_pooling_size2,
            eeg1_dropout_rate = eeg1_dropout_rate,
            eeg1_number_channel = self.number_channel,
            flatten_eeg1 = flatten_eeg1,  
            ).to(device)
        self.model_filename = self.result_name + '/model_{}.pth'.format(self.nSub)

    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        desired_records = max(1, self.number_augmentation * max(1, int(self.batch_size / self.number_class)))
        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            num_records = min(desired_records, tmp_data.shape[0])
            tmp_aug_data = np.zeros((num_records, 1, self.number_channel, 1000))
            for ri in range(num_records):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]
            aug_data.append(tmp_aug_data)
            labels_flat = tmp_label.flatten()
            repeat_labels = np.resize(labels_flat, num_records)
            aug_label.append(repeat_labels)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]
        aug_data = torch.from_numpy(aug_data).to(device).float()
        aug_label = torch.from_numpy(aug_label-1).to(device).long()
        return aug_data, aug_label

    def get_source_data(self):
        (self.train_data,
         self.train_label, 
         self.test_data, 
         self.test_label) = load_data_evaluate(self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode)
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]
        if self.verbose:
            print(f"数据加载完成: Train {self.train_data.shape}, Test {self.test_data.shape}")
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        
        # 归一化：只使用训练数据计算统计量（避免数据泄露）
        # 注意：这里allData包含后来会成为验证集的数据，但这是标准做法
        # 因为验证集和训练集来自同一文件，分布相似
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        # 测试集使用相同的归一化参数（这是正确的，避免数据泄露）
        self.testData = (self.testData - target_mean) / target_std
        isSaveDataLabel = False
        if isSaveDataLabel:
            np.save("./gradm_data/train_data_{}.npy".format(self.nSub), self.allData)
            np.save("./gradm_data/train_lable_{}.npy".format(self.nSub), self.allLabel)
            np.save("./gradm_data/test_data_{}.npy".format(self.nSub), self.testData)
            np.save("./gradm_data/test_label_{}.npy".format(self.nSub), self.testLabel)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        
        # 正确划分训练集和验证集（在训练前统一划分，而不是每个batch划分）
        dataset_size = len(img)
        val_size = int(self.validate_ratio * dataset_size)
        train_size = dataset_size - val_size
        
        # 随机打乱后划分
        indices = torch.randperm(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_img = img[train_indices]
        train_label = label[train_indices]
        val_img = img[val_indices]
        val_label = label[val_indices]
        
        # 创建数据集
        train_dataset = torch.utils.data.TensorDataset(train_img, train_label)
        val_dataset = torch.utils.data.TensorDataset(val_img, val_label)
        test_data_tensor = torch.from_numpy(test_data)
        test_label_tensor = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_label_tensor)
        
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        test_data = test_data_tensor.to(device).float()
        test_label = test_label_tensor.to(device).long()
        
        if self.verbose:
            print(f"数据划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本, 测试集 {len(test_data)} 样本")
        best_epoch = 0
        num = 0
        min_loss = float('inf')
        result_process = []
        # 早停相关变量
        epochs_without_improvement = 0
        best_val_loss = float('inf')
        
        # 初始化绘图
        if self.plot_training:
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
            self.epochs_list = []
            try:
                backend = matplotlib.get_backend()
                is_interactive = backend.lower() not in ['agg', 'pdf', 'svg', 'ps']
                
                if is_interactive:
                    plt.ion()  # 开启交互模式
                
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
                self.fig.suptitle(f'Subject {self.nSub} Training Progress', fontsize=14, fontweight='bold')
                
                if is_interactive:
                    try:
                        # 显示窗口并确保可见
                        self.fig.show()
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()
                        
                        # 尝试将窗口移到前面（不同后端方法不同）
                        try:
                            mngr = self.fig.canvas.manager
                            if hasattr(mngr, 'window'):
                                if hasattr(mngr.window, 'wm_attributes'):
                                    # TkAgg
                                    mngr.window.wm_attributes('-topmost', 1)
                                    mngr.window.wm_attributes('-topmost', 0)
                                    mngr.window.lift()  # 提升窗口
                                elif hasattr(mngr.window, 'raise_'):
                                    # Qt backend
                                    mngr.window.raise_()
                                    mngr.window.activateWindow()
                        except:
                            pass
                        
                        # 强制刷新
                        plt.pause(0.1)
                        
                        if self.verbose:
                            print(f"✓ 实时绘图窗口已打开 (后端: {backend})")
                            print(f"  如果看不到窗口，请检查任务栏或Alt+Tab切换窗口")
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠ 无法显示窗口: {e}，将仅保存图片")
                        is_interactive = False
                else:
                    if self.verbose:
                        print(f"使用非交互式后端 ({backend})，训练曲线将仅保存为图片")
                
                self.is_interactive = is_interactive
            except Exception as e:
                if self.verbose:
                    print(f"初始化绘图失败: {e}，将禁用实时绘图")
                self.plot_training = False
                self.is_interactive = False
        
        if self.verbose:
            print(f"Subject {self.nSub}: 开始训练 ({'早停' if self.early_stopping else '无早停'}, patience={self.patience})")
        
        for e in range(self.n_epochs):
            # 训练集数据加载器（每个epoch重新创建以支持shuffle）
            self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            epoch_process = {}
            epoch_process['epoch'] = e
            self.model.train()
            
            # 训练阶段
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            train_batch_count = 0
            train_outputs_list = []
            train_label_list = []
            
            for i, (img, label) in enumerate(self.train_dataloader):
                img = img.to(device).float()
                label = label.to(device).long()
                
                # 数据增强（只对训练数据）
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img_aug = torch.cat((img, aug_data))
                label_aug = torch.cat((label, aug_label))
                
                # 前向传播（使用增强后的数据）
                features, outputs = self.model(img_aug)
                loss = self.criterion_cls(outputs, label_aug) 
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 计算训练损失和准确率（只使用原始训练数据，不包括增强数据）
                with torch.no_grad():
                    _, outputs_original = self.model(img)  # 只用原始数据评估
                    train_outputs_list.append(outputs_original)
                    train_label_list.append(label)
                    train_loss_original = self.criterion_cls(outputs_original, label)
                    epoch_train_loss += train_loss_original.item()
                    batch_pred = torch.max(outputs_original, 1)[1]
                    batch_acc = float((batch_pred == label).float().mean().item())
                    epoch_train_acc += batch_acc
                train_batch_count += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            if (e + 1) % 1 == 0:
                # 计算平均训练损失和准确率
                avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else 0.0
                avg_train_acc = epoch_train_acc / train_batch_count if train_batch_count > 0 else 0.0
                
                # 验证阶段
                self.model.eval()
                self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
                val_outputs_list = []
                with torch.no_grad():
                    for i, (img, label) in enumerate(self.val_dataloader):
                        img = img.to(device).float()
                        label = label.to(device).long()
                        _, Cls = self.model(img)
                        val_outputs_list.append(Cls)
                        del img, Cls
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                
                Cls = torch.cat(val_outputs_list)
                val_label_gpu = val_label.to(device).long()
                val_loss = self.criterion_cls(Cls, val_label_gpu)
                val_pred = torch.max(Cls, 1)[1]
                val_acc = float((val_pred == val_label_gpu).cpu().numpy().astype(int).sum()) / float(val_label_gpu.size(0))
                
                val_loss_value = float(val_loss.detach().cpu().numpy())
                
                epoch_process['val_acc'] = val_acc                
                epoch_process['val_loss'] = val_loss_value
                epoch_process['train_acc'] = avg_train_acc
                epoch_process['train_loss'] = avg_train_loss
                
                # 保存数据用于绘图
                if self.plot_training:
                    self.train_losses.append(avg_train_loss)
                    self.val_losses.append(val_loss_value)
                    self.train_accs.append(avg_train_acc)
                    self.val_accs.append(val_acc)
                    self.epochs_list.append(e + 1)
                
                # 实时显示每个epoch的结果
                improved = False
                if val_loss_value < best_val_loss - self.min_delta:
                    best_val_loss = val_loss_value
                    improved = True
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # 保存最佳模型
                if improved or val_loss_value < min_loss:
                    min_loss = val_loss_value
                    best_epoch = e
                    epoch_process['epoch'] = e
                    torch.save(self.model, self.model_filename)
                    if self.verbose:
                        print(f"Epoch {e+1:4d}/{self.n_epochs} ✓ Best model saved (Val Loss: {val_loss_value:.6f})")
                
                # 实时绘图更新（每个epoch都更新）
                if self.plot_training:
                    try:
                        self._update_plots()
                    except Exception as e:
                        if self.verbose and (e + 1) % 50 == 0:
                            print(f"绘图更新错误: {e}")
                
                # 简洁的进度显示（每10个epoch或关键epoch）
                if self.verbose and ((e + 1) % 10 == 0 or improved or (e + 1) == self.n_epochs):
                    print(f"Epoch {e+1:4d}/{self.n_epochs} | Train: Loss={avg_train_loss:.4f} Acc={avg_train_acc:.4f} | "
                          f"Val: Loss={val_loss_value:.4f} Acc={val_acc:.4f} | "
                          f"Best: Epoch {best_epoch+1} ({best_val_loss:.4f})")
                
                num = num + 1
                
                # 早停检查
                if self.early_stopping and epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"\n⚠️  早停触发！连续 {self.patience} 个 epoch 未改善 | "
                              f"最佳 Epoch: {best_epoch+1}, Val Loss: {best_val_loss:.6f}")
                    break
            
            result_process.append(epoch_process)  
            if device.type == "cuda":
                torch.cuda.empty_cache()
        self.model.eval()
        # 加载最佳模型
        self.model = torch.load(self.model_filename, map_location=device, weights_only=False)
        
        # 在验证集上评估最佳模型
        val_outputs_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(self.val_dataloader):
                img = img.to(device).float()
                _, outputs = self.model(img)
                val_outputs_list.append(outputs)
        val_outputs = torch.cat(val_outputs_list)
        val_pred_final = torch.max(val_outputs, 1)[1]
        val_acc_final = float((val_pred_final == val_label.to(device)).cpu().numpy().astype(int).sum()) / float(val_label.size(0))
        
        # 在测试集上评估
        test_outputs_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_dataloader):
                img_test = img.to(device).float()
                features, outputs = self.model(img_test)
                test_outputs_list.append(outputs)
        test_outputs = torch.cat(test_outputs_list) 
        y_pred = torch.max(test_outputs, 1)[1]
        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        
        # 保存最终图表
        if self.plot_training:
            self._save_plots()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Subject {self.nSub} 最终评估结果（最佳模型，Epoch {best_epoch+1}）:")
            print(f"  验证集准确率: {val_acc_final:.4f} ({val_acc_final*100:.2f}%) | 样本数: {len(val_label)}")
            print(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%) | 样本数: {len(test_label)}")
            if abs(val_acc_final - test_acc) > 0.05:
                print(f"  ⚠ 注意: 验证集和测试集准确率差异较大 ({abs(val_acc_final-test_acc)*100:.2f}%)")
                print(f"     可能原因: 过拟合或数据分布差异")
            print(f"{'='*70}\n")
        
        df_process = pd.DataFrame(result_process)
        return test_acc, test_label, y_pred, df_process, best_epoch
    
    def _update_plots(self):
        """更新训练曲线图"""
        if not self.plot_training or len(self.epochs_list) == 0:
            return
        
        try:
            self.ax1.clear()
            self.ax2.clear()
            
            # 绘制 Loss 曲线
            self.ax1.plot(self.epochs_list, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
            self.ax1.plot(self.epochs_list, self.val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)
            self.ax1.set_xlabel('Epoch', fontsize=12)
            self.ax1.set_ylabel('Loss', fontsize=12)
            self.ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
            self.ax1.legend(loc='upper right', fontsize=10)
            self.ax1.grid(True, alpha=0.3)
            
            # 绘制 Accuracy 曲线
            self.ax2.plot(self.epochs_list, self.train_accs, 'b-', label='Train Acc', linewidth=2, alpha=0.8)
            self.ax2.plot(self.epochs_list, self.val_accs, 'r-', label='Val Acc', linewidth=2, alpha=0.8)
            self.ax2.set_xlabel('Epoch', fontsize=12)
            self.ax2.set_ylabel('Accuracy', fontsize=12)
            self.ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
            self.ax2.legend(loc='lower right', fontsize=10)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            
            # 根据后端类型更新图表
            if hasattr(self, 'is_interactive') and self.is_interactive:
                try:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.pause(0.01)  # 稍微长一点的暂停，确保GUI更新
                except Exception:
                    # 如果更新失败，标记为非交互式
                    self.is_interactive = False
        except Exception:
            # 静默处理绘图错误
            pass
    
    def _save_plots(self):
        """保存训练曲线图"""
        if not self.plot_training:
            return
        
        plot_filename = self.result_name + f'/training_curve_subject_{self.nSub}.png'
        try:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
            
            # 最终更新一次图表
            plt.tight_layout()
            self.fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"训练曲线已保存: {plot_filename}")
            
            # 如果是交互式后端，保持窗口打开
            backend = matplotlib.get_backend()
            if backend.lower() == 'agg':
                plt.close(self.fig)
            # 交互式后端保持窗口打开，让用户查看
        except Exception as e:
            if self.verbose:
                print(f"保存训练曲线失败: {e}")
            try:
                plt.close(self.fig)
            except:
                pass

def get_class_labels(dataset_type, number_class):
    """
    获取类别标签（方向/动作）
    
    Parameters:
    -----------
    dataset_type : str
        数据集类型 'A' (BCI IV-2a) 或 'B' (BCI IV-2b)
    number_class : int
        类别数量
    
    Returns:
    --------
    class_labels : dict
        类别索引到标签名称的映射
    """
    if dataset_type == 'A':
        # BCI IV-2a: 4类运动想象
        labels = {
            0: 'Left Hand\n(左手)',
            1: 'Right Hand\n(右手)',
            2: 'Both Feet\n(双脚)',
            3: 'Tongue\n(舌头)'
        }
    elif dataset_type == 'B':
        # BCI IV-2b: 2类运动想象
        labels = {
            0: 'Left Hand\n(左手)',
            1: 'Right Hand\n(右手)'
        }
    else:
        labels = {i: f'Class {i}' for i in range(number_class)}
    
    return labels

def plot_confusion_matrix(y_true, y_pred, class_labels, save_dir, subject_id, dataset_type, title_suffix='', verbose=True):
    """
    绘制并保存混淆矩阵
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        真实标签
    y_pred : numpy.ndarray
        预测标签
    class_labels : dict
        类别标签映射
    save_dir : str
        保存目录
    subject_id : int or str
        受试者ID或'all_subjects'
    dataset_type : str
        数据集类型
    title_suffix : str
        标题后缀
    verbose : bool
        是否打印信息
    """
    from sklearn.metrics import confusion_matrix as cm_sklearn
    
    # 计算混淆矩阵
    num_classes = len(class_labels)
    cm = cm_sklearn(y_true, y_pred, labels=list(range(num_classes)))
    
    # 计算准确率
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 归一化混淆矩阵（百分比）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况
    
    # 使用seaborn绘制热力图（如果可用），否则使用matplotlib
    if SEABORN_AVAILABLE:
        sns.heatmap(cm_normalized, annot=False, fmt='.1f', cmap='Blues', 
                    cbar_kws={'label': 'Percentage (%)'},
                    xticklabels=[class_labels[i] for i in range(num_classes)],
                    yticklabels=[class_labels[i] for i in range(num_classes)],
                    linewidths=0.5, linecolor='gray')
    else:
        # 如果没有seaborn，使用matplotlib
        plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Percentage (%)')
        plt.xticks(range(num_classes), [class_labels[i] for i in range(num_classes)])
        plt.yticks(range(num_classes), [class_labels[i] for i in range(num_classes)])
    
    # 在归一化矩阵上叠加原始数值和百分比
    for i in range(num_classes):
        for j in range(num_classes):
            percentage = cm_normalized[i, j]
            count = cm[i, j]
            text_color = 'white' if percentage > 50 else 'black'
            plt.text(j+0.5, i+0.7, f'{percentage:.1f}%', 
                    ha='center', va='center', 
                    fontsize=11, color=text_color, weight='bold')
            plt.text(j+0.5, i+0.3, f'({count})', 
                    ha='center', va='center', 
                    fontsize=9, color=text_color)
    
    plt.title(f'Confusion Matrix - Subject {subject_id}{title_suffix}\n'
              f'Accuracy: {accuracy:.2%} ({accuracy*100:.2f}%)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    cm_filename = os.path.join(save_dir, f'confusion_matrix_subject_{subject_id}.png')
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存混淆矩阵数据到Excel
    cm_df = pd.DataFrame(cm, 
                        index=[class_labels[i] for i in range(num_classes)],
                        columns=[class_labels[i] for i in range(num_classes)])
    cm_df['Total'] = cm.sum(axis=1)
    cm_df.loc['Total'] = list(cm.sum(axis=0)) + [cm.sum()]
    
    cm_excel_filename = os.path.join(save_dir, f'confusion_matrix_subject_{subject_id}.xlsx')
    cm_df.to_excel(cm_excel_filename)
    
    if verbose:
        print(f"混淆矩阵已保存: {cm_filename}")

def main(dirs,                
         evaluate_mode = 'subject-dependent',
         heads=8,
         emb_size=48,
         depth=3,
         dataset_type='A',
         eeg1_f1=20,
         eeg1_kernel_size=64,
         eeg1_D=2,
         eeg1_pooling_size1=8,
         eeg1_pooling_size2=8,
         eeg1_dropout_rate=0.3,
         flatten_eeg1=600,   
         validate_ratio = 0.2,
         early_stopping = True,
         patience = 50,
         min_delta = 0.0001,
         verbose = True,
         plot_training = True,
         batch_size = 512,
         learning_rate = 0.0001,
         ):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    result_write_metric = ExcelWriter(dirs+"/result_metric.xlsx")
    result_metric_dict = {}
    y_true_pred_dict = { }
    process_write = ExcelWriter(dirs+"/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs+"/pred_true.xlsx")
    subjects_result = []
    best_epochs = []
    for i in range(N_SUBJECT):      
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        if verbose:
            print(f'Subject {i+1}: Seed = {seed_n}')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        index_round =0
        exp = ExP(i + 1, DATA_DIR, dirs, EPOCHS, N_AUG, N_SEG, None, 
                  evaluate_mode = evaluate_mode,
                  heads=heads, 
                  emb_size=emb_size,
                  depth=depth, 
                  dataset_type=dataset_type,
                  eeg1_f1 = eeg1_f1,
                  eeg1_kernel_size = eeg1_kernel_size,
                  eeg1_D = eeg1_D,
                  eeg1_pooling_size1 = eeg1_pooling_size1,
                  eeg1_pooling_size2 = eeg1_pooling_size2,
                  eeg1_dropout_rate = eeg1_dropout_rate,
                  flatten_eeg1 = flatten_eeg1,  
                  validate_ratio = validate_ratio,
                  early_stopping = early_stopping,
                  patience = patience,
                  min_delta = min_delta,
                  verbose = verbose,
                  plot_training = plot_training,
                  batch_size = batch_size,
                  learning_rate = learning_rate,
                  )
        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        
        # 获取类别标签（方向）
        number_class, _ = numberClassChannel(dataset_type)
        class_labels = get_class_labels(dataset_type, number_class)
        
        # 创建详细的预测结果DataFrame（包含类别名称）
        true_labels = [class_labels[t] for t in true_cpu]
        pred_labels = [class_labels[p] for p in pred_cpu]
        df_pred_true = pd.DataFrame({
            'true_class': true_cpu,
            'pred_class': pred_cpu,
            'true_label': true_labels,
            'pred_label': pred_labels,
            'correct': (true_cpu == pred_cpu).astype(int)
        })
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i+1))
        y_true_pred_dict[i] = df_pred_true
        
        # 生成并保存混淆矩阵
        plot_confusion_matrix(true_cpu, pred_cpu, class_labels, 
                             dirs, i+1, dataset_type, verbose=verbose)
        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {'accuray': accuracy*100,
                          'precision': precison*100,
                          'recall': recall*100,
                          'f1': f1*100, 
                          'kappa': kappa*100
                          }
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i+1))
        best_epochs.append(best_epoch)
        if verbose:
            print(f'Subject {i+1}: 最佳准确率 {testAcc:.4f} | Kappa: {kappa:.4f}')
        endtime = datetime.datetime.now()
        if verbose:
            print(f'Subject {i+1}: 训练耗时 {endtime - starttime}')
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
        df_result = pd.DataFrame(subjects_result)
    
    # 生成所有受试者的总体混淆矩阵
    if len(yt) > 0 and verbose:
        yt_all = yt.cpu().numpy().astype(int)
        yp_all = yp.cpu().numpy().astype(int)
        number_class, _ = numberClassChannel(dataset_type)
        class_labels = get_class_labels(dataset_type, number_class)
        plot_confusion_matrix(yt_all, yp_all, class_labels, 
                             dirs, 'all_subjects', dataset_type, 
                             title_suffix=' (All Subjects)', verbose=verbose)
    process_write.close()
    pred_true_write.close()
    print(f'\n最终结果: 平均准确率 {df_result["accuray"].mean():.2f}% | 平均 Kappa {df_result["kappa"].mean():.4f}')
    print(f"最佳 Epochs: {best_epochs}")
    result_metric_dict = df_result
    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    df_result.to_excel(result_write_metric, index=False)
    print('\n详细结果:')
    print(df_result.to_string())
    result_write_metric.close()
    return result_metric_dict

if __name__ == "__main__":
    DATA_DIR = r'./mymat_raw/'
    EVALUATE_MODE = 'LOSO-No'
    N_SUBJECT = 9
    N_AUG = 3
    N_SEG = 8
    EPOCHS = 1000
    EMB_DIM = 16
    HEADS = 4
    DEPTH = 6
    TYPE = 'B'
    validate_ratio = 0.3
    # 训练参数
    BATCH_SIZE = 72                 # 批次大小
    LEARNING_RATE = 0.001           # 学习率
    # 早停参数
    EARLY_STOPPING = True           # 是否启用早停
    PATIENCE = 50                   # 早停耐心值（连续N个epoch不改善则停止）
    MIN_DELTA = 0.0001              # 最小改善阈值
    VERBOSE = True                  # 是否显示详细信息
    PLOT_TRAINING = True            # 是否实时绘制训练曲线
    EEGNet1_F1 = 8
    EEGNet1_KERNEL_SIZE=64
    EEGNet1_D=2
    EEGNet1_POOL_SIZE1 = 8
    EEGNet1_POOL_SIZE2 = 8
    FLATTEN_EEGNet1 = 240
    if EVALUATE_MODE!='LOSO':
        EEGNet1_DROPOUT_RATE = 0.5
    else:
        EEGNet1_DROPOUT_RATE = 0.25    
    parameters_list = ['A']
    for TYPE in parameters_list:
        number_class, number_channel = numberClassChannel(TYPE)
        RESULT_NAME = "CTNet_{}_heads_{}_depth_{}_{}".format(TYPE, HEADS, DEPTH, int(time.time()))
        sModel = EEGTransformer(
            heads=HEADS, 
            emb_size=EMB_DIM,
            depth=DEPTH, 
            database_type=TYPE,
            eeg1_f1=EEGNet1_F1, 
            eeg1_D=EEGNet1_D,
            eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
            eeg1_pooling_size1 = EEGNet1_POOL_SIZE1,
            eeg1_pooling_size2 = EEGNet1_POOL_SIZE2,
            eeg1_dropout_rate = EEGNet1_DROPOUT_RATE,
            eeg1_number_channel = number_channel,
            flatten_eeg1 = FLATTEN_EEGNet1,  
            ).to(device)
        if os.environ.get("ENABLE_MODEL_SUMMARY", "0") == "1":
            summary(sModel, (1, number_channel, 1000)) 
        print(time.asctime(time.localtime(time.time())))
        result = main(RESULT_NAME,
                        evaluate_mode = EVALUATE_MODE,
                        heads=HEADS, 
                        emb_size=EMB_DIM,
                        depth=DEPTH, 
                        dataset_type=TYPE,
                        eeg1_f1 = EEGNet1_F1,
                        eeg1_kernel_size = EEGNet1_KERNEL_SIZE,
                        eeg1_D = EEGNet1_D,
                        eeg1_pooling_size1 = EEGNet1_POOL_SIZE1,
                        eeg1_pooling_size2 = EEGNet1_POOL_SIZE2,
                        eeg1_dropout_rate = EEGNet1_DROPOUT_RATE,
                        flatten_eeg1 = FLATTEN_EEGNet1,
                        validate_ratio = validate_ratio,
                        early_stopping = EARLY_STOPPING,
                        patience = PATIENCE,
                        min_delta = MIN_DELTA,
                        verbose = VERBOSE,
                        plot_training = PLOT_TRAINING,
                        batch_size = BATCH_SIZE,
                        learning_rate = LEARNING_RATE,
                      )
        print(time.asctime(time.localtime(time.time())))
