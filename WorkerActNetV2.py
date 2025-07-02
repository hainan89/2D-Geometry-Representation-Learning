import torch
import torch.nn as nn

class WorkerActNet(nn.Module):
    def __init__(self, input_size=27, conv_channels=1, kernel_size=5, hidden_size=1024, num_classes=16):
        super(WorkerActNet, self).__init__()
        
        # 一维卷积层，每个维度一个卷积核
        self.conv_layers = nn.ModuleList([nn.Conv1d(conv_channels, 1, kernel_size=kernel_size, padding=kernel_size//2) for _ in range(input_size)])
        
        # 批归一化层，每个维度一个
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(input_size)])
        
        # 激活函数层
        self.elu = nn.ELU()
        
        # 最大池化层
        self.max_pool = nn.MaxPool1d(2)
        
        # LSTM 网络
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 分类任务的输出层
        self.fc_class1_ly1 = nn.Linear(hidden_size, 1024)
        self.fc_class1_ly2 = nn.Linear(1024, 1024)
        self.fc_class1 = nn.Linear(1024, num_classes)
        # 分类任务的输出层
        # self.fc_class2 = nn.Linear(hidden_size, num_classes)
        
        # 回归任务的输出层
        self.fc_p1_ly1 = nn.Linear(hidden_size, 1024)
        self.fc_p1_ly2 = nn.Linear(1024, 1024)
        self.fc_p1 = nn.Linear(1024, 1)
        # self.fc_p2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x.shape: (batch_size, seq_length, input_size)
        #print("input shape", x.shape)
        
        # 列表用于存储每个维度归一化后的输出
        norm_outs = []
        
        # 对每个维度应用一维卷积和批归一化
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            conv_out = conv(x[:, :, i].unsqueeze(1))  # 卷积输出
            # conv_out.shape: (batch_size, 1, seq_length)
            
            norm_out = bn(conv_out)  # 批归一化
            # norm_out.shape: (batch_size, 1, seq_length)
            
            elu_out = self.elu(norm_out)
            
            pool_out = self.max_pool(elu_out)
            
            
            norm_outs.append(pool_out)
        
        # 将归一化后的输出在维度上拼接
        norm_cat = torch.cat(norm_outs, dim=1)
        # norm_cat.shape: (batch_size, input_size, seq_length)
        # print("norm_cat", norm_cat.shape)
        
        # 激活函数
        #         norm_cat = self.elu(norm_cat)
        #         print("norm_cat = self.elu(norm_cat)", norm_cat.shape)
        
        # 最大池化
        norm_cat = norm_cat.permute(0, 2, 1)  # 转换为 (batch_size, seq_length, input_size) 以符合池化层的输入要求
        #         print("norm_cat = norm_cat.permute(0, 2, 1)", norm_cat.shape)
        
        
        
        # LSTM 处理
        lstm_out, _ = self.lstm(norm_cat)
        # lstm_out.shape: (batch_size, seq_length, hidden_size)
        
        # 通常我们只取 LSTM 的最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        # lstm_out.shape: (batch_size, hidden_size)
        
        # 分类任务
        ly1_out = self.fc_class1_ly1(lstm_out)
        ly2_out = self.fc_class1_ly2(ly1_out)
        class1_out = self.fc_class1(ly2_out)
        # class_out.shape: (batch_size, num_classes)
        
        # 回归任务
        p_ly1_out = self.fc_p1_ly1(lstm_out)
        p_ly2_out = self.fc_p1_ly2(p_ly1_out)
        p1_out = self.fc_p1(p_ly2_out)
        # offset_out.shape: (batch_size, 1)

        return class1_out, p1_out