import torch  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import json
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device

from WorkerActNetV3 import WorkerActNet
from WorkerActDataset import WorkerActDataset

ws = 1
act_period = 480 # 单位分钟, 确保动作类别和参与者id的有足够多的样本
act_task = [1, 2, 3, 4, 8, 9, 10, 11, 12, 15] # 作业类动作
act_trans = [5, 6, 7, 16, 13, 14] # 过度类动作

## 原始的动作数据中，实验者数据是按照数组的形式存储，id下标从 0 开始
participant_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 实验参与者序号
participant_id = [int(one) for one in list(np.array(participant_id) - 1)]

data_set_info = {"Note":"ws单位为秒，act_period单位为分钟，act类别总共16类编号1-16，participant总共13人编号1-13，便于训练从编号0开始",
                 "ws":ws, "act_period":act_period, "act_task":act_task, "act_trans":act_trans,
                 "participant_id":participant_id}

train_dataset = WorkerActDataset(ws, act_period, act_task, act_trans, participant_id)  
with open(os.path.join(output_dir, "train_dataset.pkl"), 'wb') as f:
    pickle.dump(train_dataset, f)
with open(os.path.join(output_dir, "data_set_info.json"), "w", encoding="utf-8") as f:
    json.dump(data_set_info, f, indent=4, ensure_ascii=False)
    
train_loader = DataLoader(train_dataset, batch_size=1024)
print("dataset size:", train_dataset.size)

    
# 实例化模型
model = WorkerActNet().to(device)

# 定义损失函数
criterion_class = nn.CrossEntropyLoss()
# criterion_reg = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 200  # 训练的轮数

total_loss = np.inf

print(f"Start training...")
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    
    running_loss_class1 = 0.0

    
    class_label = []

    class_out = []
    # class_out_p = []

    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        class_label.extend(batch_y.numpy())
        
        # 将数据移到 GPU 上（如果可用）
        batch_x = batch_x.to(torch.float)
        batch_x = batch_x.to(device)
        
        batch_labels = batch_y.to(torch.long).to(device)

        # 前向传播
        outputs = model(batch_x)
        
        # 获得模型输出
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).cpu().numpy()
        
        class_out.extend(predicted_class)
        # class_out_p.extend(probabilities)

        # print("predicted_class", predicted_class)
        # 总损失（可以根据需要调整权重）
        loss = criterion_class(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失
        running_loss_class1 += loss.item()
        
        print(f"Batch i:{i}/{train_dataset.size/1024}, Batch Loss: {loss.item()}")

    # 打印每个 epoch 的平均损失
    epoch_status = f'Epoch [{epoch+1}/{num_epochs}], '
    epoch_status = epoch_status + f'Loss Class1: {running_loss_class1/len(train_loader):.4f} \n'
    
    if (running_loss_class1) < total_loss:
        ## 保存模型
        total_loss = running_loss_class1
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_state_dict.pth'))
        
        ## 保存最优模型的输出结果
        with open(os.path.join(output_dir, "train_result.pkl"), 'wb') as f:
            pickle.dump({"class_label":class_label, 
                         "class_out":class_out}, f)
        ## 打印模型的分类性能    
        report = classification_report(class_label, class_out)
        print('====分类结果====\n',report)
    
    with open(os.path.join(output_dir, "train_epoch_status.txt"), 'a') as f:
        f.writelines(epoch_status)
    print(epoch_status)