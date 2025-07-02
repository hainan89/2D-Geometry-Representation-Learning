import torch  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import pickle


import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torch.nn.functional as F

from sklearn.metrics import classification_report

output_dir = f"./output-{time.time()}"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    

# 检查 GPU 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device

from WorkerActNetV2 import WorkerActNet
from WorkerActDataset import WorkerActDataset

ws = 1
act_period = 12 #60 * 24 ## 单位分钟
act_task = [1, 2, 3, 4, 8, 9, 10, 11, 12, 15] # 作业类动作
act_trans = [5, 6, 7, 16, 13, 14] # 过度类动作

## 原始的动作数据中，实验者数据是按照数组的形式存储，id下标从 0 开始
participant_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 实验参与者序号
participant_id = list(np.array(participant_id) - 1)

train_dataset = WorkerActDataset(ws, act_period, act_task, act_trans, participant_id)  
with open(os.path.join(output_dir, "train_dataset.pkl"), 'wb') as f:
    pickle.dump(train_dataset, f)
    
train_loader = DataLoader(train_dataset, batch_size=1024)
print("dataset size:", train_dataset.size)

    
# 实例化模型
model = WorkerActNet().to(device)

# 定义损失函数
criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 200  # 训练的轮数

total_loss = np.inf

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    
    running_loss_class1 = 0.0
    running_loss_reg1 = 0.0
    
    class_label = []
    reg_label = []

    class_out = []
    class_out_p = []
    reg_out = []
    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        class_label.extend(batch_y[:, 0].numpy())
        reg_label.extend(batch_y[:, 1].numpy())
        
        # 将数据移到 GPU 上（如果可用）
        batch_x = batch_x.to(torch.float)
        batch_x = batch_x.to(device)
        
        class1_labels = batch_y[:, 0].to(torch.long).to(device)
        reg1_labels = batch_y[:, 1].to(torch.float).to(device)

        # 前向传播
        class1_out, p1_out = model(batch_x)
        
        # 获得模型输出
        probabilities = F.softmax(class1_out, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).cpu().numpy()
        
        class_out.extend(predicted_class)
        class_out_p.extend(probabilities)
        reg_out.extend(p1_out.detach().cpu().numpy())
        
        
        # 计算损失
        loss_class1 = criterion_class(class1_out, class1_labels)
        loss_reg1 = criterion_reg(p1_out.squeeze(1), reg1_labels)

        # 总损失（可以根据需要调整权重）
        loss = loss_class1 + loss_reg1

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失
        running_loss_class1 += loss_class1.item()
        running_loss_reg1 += loss_reg1.item()

    # 打印每个 epoch 的平均损失
    epoch_status = f'Epoch [{epoch+1}/{num_epochs}], '
    epoch_status = epoch_status + f'Loss Class1: {running_loss_class1/len(train_loader):.4f}, '
    epoch_status = epoch_status + f'Loss Reg1: {running_loss_reg1/len(train_loader):.4f} \n' 
    
    if (running_loss_class1 + running_loss_reg1) < total_loss:
        ## 保存模型
        total_loss = running_loss_class1 + running_loss_reg1
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_state_dict.pth'))
        
        ## 保存最优模型的输出结果
        with open(os.path.join(output_dir, "train_result.pkl"), 'wb') as f:
            pickle.dump({"class_label":class_label, 
                         "reg_label":reg_label,
                         "class_out":class_out, 
                         "class_out_p":class_out_p,
                         "reg_out":reg_out}, f)
        ## 打印模型的分类性能    
        report = classification_report(class_label, class_out)
        print('====分类结果====\n',report)
    
    with open(os.path.join(output_dir, "train_epoch_status.txt"), 'a') as f:
        f.writelines(epoch_status)
    print(epoch_status)