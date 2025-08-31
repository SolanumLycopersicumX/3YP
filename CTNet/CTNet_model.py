import os
import numpy as np
import pandas as pd
import random
import datetime
import time

from pandas import ExcelWriter
from torchsummary import summary
import torch
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

# Set device to CPU
device = torch.device("cpu")

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
        number_records_by_augmentation = self.number_augmentation * int(self.batch_size / self.number_class)
        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:number_records_by_augmentation])
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
        print('-'*20, "train size：", self.train_data.shape, "test size：", self.test_data.shape)
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
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
        dataset = torch.utils.data.TensorDataset(img, label)
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        test_data = test_data.to(device).float()
        test_label = test_label.to(device).long()
        best_epoch = 0
        num = 0
        min_loss = 100
        result_process = []
        for e in range(self.n_epochs):
            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
            epoch_process = {}
            epoch_process['epoch'] = e
            self.model.train()
            outputs_list = []
            label_list = []
            val_data_list = []
            val_label_list = []
            for i, (img, label) in enumerate(self.dataloader):
                number_sample = img.shape[0]
                number_validate = int(self.validate_ratio * number_sample)
                train_data = img[:-number_validate]
                train_label = label[:-number_validate]
                val_data_list.append(img[number_validate:])
                val_label_list.append(label[number_validate:])
                img = train_data.to(device).float()
                label = train_label.to(device).long()
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                features, outputs = self.model(img)
                outputs_list.append(outputs)
                label_list.append(label)
                loss = self.criterion_cls(outputs, label) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            del img
            torch.cuda.empty_cache()
            if (e + 1) % 1 == 0:
                self.model.eval()
                val_data = torch.cat(val_data_list).to(device).float()
                val_label = torch.cat(val_label_list).to(device).long()
                val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
                self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
                outputs_list = []
                with torch.no_grad():
                    for i, (img, _) in enumerate(self.val_dataloader):
                        img = img.to(device).float()
                        _, Cls = self.model(img)
                        outputs_list.append(Cls)
                        del img, Cls
                        torch.cuda.empty_cache()
                Cls = torch.cat(outputs_list)
                val_loss = self.criterion_cls(Cls, val_label)
                val_pred = torch.max(Cls, 1)[1]
                val_acc = float((val_pred == val_label).cpu().numpy().astype(int).sum()) / float(val_label.size(0))
                epoch_process['val_acc'] = val_acc                
                epoch_process['val_loss'] = val_loss.detach().cpu().numpy()  
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                epoch_process['train_acc'] = train_acc
                epoch_process['train_loss'] = loss.detach().cpu().numpy()
                num = num + 1
                if min_loss>val_loss:
                    min_loss = val_loss
                    best_epoch = e
                    epoch_process['epoch'] = e
                    torch.save(self.model, self.model_filename)
                    print("{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.7f}".format(self.nSub,
                                                                                           epoch_process['epoch'],
                                                                                           epoch_process['train_acc'],
                                                                                           epoch_process['train_loss'],
                                                                                           epoch_process['val_acc'],
                                                                                           epoch_process['val_loss'],
                                                                                        ))
            result_process.append(epoch_process)  
            del label, val_data, val_label
            torch.cuda.empty_cache()
        self.model.eval()
        self.model = torch.load(self.model_filename, map_location=device)
        outputs_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_dataloader):
                img_test = img.to(device).float()
                features, outputs = self.model(img_test)
                val_pred = torch.max(outputs, 1)[1]
                outputs_list.append(outputs)
        outputs = torch.cat(outputs_list) 
        y_pred = torch.max(outputs, 1)[1]
        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        print("epoch: ", best_epoch, '\tThe test accuracy is:', test_acc)
        df_process = pd.DataFrame(result_process)
        return test_acc, test_label, y_pred, df_process, best_epoch

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
         validate_ratio = 0.2
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
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        index_round =0
        print('Subject %d' % (i+1))
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
                  validate_ratio = validate_ratio
                  )
        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i+1))
        y_true_pred_dict[i] = df_pred_true
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
        print(' THE BEST ACCURACY IS ' + str(testAcc) + "\tkappa is " + str(kappa) )
        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
        df_result = pd.DataFrame(subjects_result)
    process_write.close()
    pred_true_write.close()
    print('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) + "kappa is: " + str(df_result['kappa'].mean()) + "\n" )
    print("best epochs: ", best_epochs)
    result_metric_dict = df_result
    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    df_result.to_excel(result_write_metric, index=False)
    print('-'*9, ' all result ', '-'*9)
    print(df_result)
    print("*"*40)
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
    HEADS = 2
    DEPTH = 6
    TYPE = 'B'
    validate_ratio = 0.3
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
                      )
        print(time.asctime(time.localtime(time.time())))
