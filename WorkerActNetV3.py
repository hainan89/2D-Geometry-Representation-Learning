# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的函数式接口
import torch.nn.functional as F
import numpy as np
'''

'''

# 定义工人行为识别网络模型类，继承自nn.Module
class WorkerActNet(nn.Module):
    def __init__(self, input_size=27, conv_channels=1, kernel_size=5, hidden_size=1024, num_classes=17): # 16个动作+1个标号为-1的无识别动作
        # 调用父类的初始化方法
        super(WorkerActNet, self).__init__()
        
        # 创建多个1D卷积层，每个输入特征对应一个卷积层
        self.conv_layers = nn.ModuleList([nn.Conv1d(conv_channels, 1, kernel_size=kernel_size, padding=kernel_size//2) for _ in range(input_size)])
        
        # 为每个卷积层创建对应的批归一化层
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(input_size)])

        # 定义ELU激活函数
        self.elu = nn.ELU()

        # 定义最大池化层，池化窗口大小为2
        self.max_pool = nn.MaxPool1d(2)

        # 定义LSTM层，用于处理序列数据
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 定义全连接层，用于分类
        self.fc_class1_ly1 = nn.Linear(hidden_size, 1024)  # 第一层全连接
        self.fc_class1_ly2 = nn.Linear(1024, 1024)         # 第二层全连接
        self.fc_class1 = nn.Linear(1024, num_classes)      # 输出层
 
    
    def forward(self, x):
        # 存储每个特征的归一化输出
        norm_outs = []
        
        # 对每个输入特征进行处理
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # 提取当前特征并进行卷积操作
            conv_out = conv(x[:, :, i].unsqueeze(1))
            
            # 对卷积输出进行批归一化
            norm_out = bn(conv_out)
            
            # 应用ELU激活函数
            elu_out = self.elu(norm_out)
            
            # 进行最大池化操作
            pool_out = self.max_pool(elu_out)
            
            # 将处理后的特征添加到列表中
            norm_outs.append(pool_out)
        
        # 将所有特征在通道维度上拼接
        norm_cat = torch.cat(norm_outs, dim=1)
        
        # 调整张量维度顺序，使其符合LSTM的输入要求
        norm_cat = norm_cat.permute(0, 2, 1)
        
        # 通过LSTM层处理序列数据
        lstm_out, _ = self.lstm(norm_cat)

        # 只取LSTM最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 通过全连接层进行分类
        ly1_out = self.fc_class1_ly1(lstm_out)    # 第一层全连接
        ly2_out = self.fc_class1_ly2(ly1_out)     # 第二层全连接
        logits = self.fc_class1(ly2_out)          # 输出层，得到logits
        
        return logits
        