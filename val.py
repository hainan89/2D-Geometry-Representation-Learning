import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# 检查 GPU 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device

from WorkerActNetV3 import WorkerActNet
from WorkerActDataset import WorkerActDataset

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

import pickle
import torch.nn.functional as F

output_dir = f"./output-1743951311.6536357"

## 测试集中与训练集中时间窗口保持一致
ws = 1

act_period = 5 # 单位分钟, 确保动作类别和参与者id的有足够多的样本

## 测试集与训练集中的动作类型划分保持一致
act_task = [1, 2, 3, 4, 8, 9, 10, 11, 12, 15] # 作业类动作
act_trans = [5, 6, 7, 16, 13, 14] # 过度类动作

## 测试集选择训练集中没有参与过的实验者数据
## 原始的动作数据中，实验者数据是按照数组的形式存储，id下标从 0 开始
## 训练接中的实验者编号
# participant_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 实验参与者序号
participant_id = [11, 12, 13]
participant_id = [int(one) for one in list(np.array(participant_id) - 1)]

val_dataset = WorkerActDataset(ws, act_period, act_task, act_trans, participant_id)
with open(os.path.join(output_dir, "val_dataset.pkl"), 'wb') as f:
    pickle.dump(val_dataset, f)
    
print("validation dataset size:", val_dataset.size)

val_loader = DataLoader(val_dataset, batch_size=1024)
 
# 加载训练好的模型参数
model = WorkerActNet()
model.load_state_dict(torch.load(os.path.join(output_dir, 'model_state_dict.pth')))
model.to(device)

# 设置模型为评估模式
model.eval()  

# 定义损失函数
criterion_class = nn.CrossEntropyLoss()

running_loss_class1 = 0.0

class_label = []
class_out = []

for i, (batch_x, batch_y) in enumerate(val_loader):
    # 将数据移到 GPU 上（如果可用）
    batch_x = batch_x.to(torch.float)
    batch_x = batch_x.to(device)
    
    class_label.append(batch_y.item())
    

    # 前向传播
    outputs = model(batch_x)
    
    # 获得模型输出
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).cpu().numpy()
    
    class_out.extend(predicted_class)
           


cm = confusion_matrix(class_label, class_out)
print(cm)
        
        
with open(os.path.join(output_dir, "val_result.pkl"), 'wb') as f:
    pickle.dump({"class_label":class_label, "class_out":class_out}, f)
    
    
    
    
    