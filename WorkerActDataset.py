import os  
import pickle  
import torch  
from torch.utils.data import Dataset, DataLoader  

from DataLoader import get_loc_data
import random

import numpy as np
import pandas as pd
import sys

'''
原始数据集为 VTT_ConIot_Dataset：
（1）包括13个实验者，每个实验者16个动作类别，每个实验者佩戴了3个位置的传感器（6号实验者没有执行11号动作，没有数据）
（2）每个动作，每个实验者有1分钟的IMU数据
（3）每个佩戴位置有4个传感器，加速度计、陀螺仪、磁力计、气压计

当前数据生成模型，在原始数据集VTT基础上，组合不同实验者的不同动作数据，生成新建筑工人行为动作序列数据。
（1）将原始的16类动作，分为： 执行动作 和 过度动作
    执行动作： 1, 2, 3, 4, 8, 9, 10, 11, 12, 15
    过度动作： 5, 6, 7, 13, 14, 16
（2）按照给定的时间要求，随机组合执行动作和过度动作生成一个动作序列
（3）对于每个动作序列，从13个实验者数据中，随机选择一个实验者的数据，归入到新生成的数据中
（4）动作序列生成以及挑选实验者数据的时候，通过记录动作和实验者被归入的次数，总体确保生成的动作类别和采用的实验者比例均衡。
'''

class WorkerActDataset(Dataset):
    def __init__(self, ws, act_period, act_task, act_trans, participant_id):
        '''
        ## ws: 时间窗口的长度单位为秒，一个 act 片为 1 分钟，对应 6000 条传感器器技术，所以 1 个 ws 单位对应 100 条记录
        ## act_period: 整个数据序列的时间长度 单位为分钟 m
        ## act_task: 任务操作类动作的索引
        ## act_trans: 多度动作类的索引
        ## participant_id: 参与者的缩影
        ## container: 原始实验者的传感器数据 dict 类型key 为动作类别，value 为当前 key 类动作的实验者数据
        '''
        self.act_u_size = 6000 ## 单个act片长度为 6000对应 1 分钟
        self.ws = ws #时间窗口的长度单位为秒，对应单个act片中的100个传感器数据
        self.sec_size = self.ws * 100
        
        self.act_period = self.act_u_size ##动作周期
        self.act_p_power = 8 ## 动作概率幂函数强度
        
        
        self.L = act_period
        # 作业类动作
        self.act_task = act_task # [1, 2, 3, 4, 8, 9, 10, 11, 12, 15]

        # 过度类动作
        self.act_trans = act_trans # [5, 6, 7, 16, 13, 14]

        # 实验参与者序号
        self.participant_id = participant_id #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 原始实验者的传感器数据
        self.container = self.load_raw_data()
        
        ## 按照给定的 L 要求，获得 act 和对应的实验者编号
        self.act_sec, self.p_sec = self.get_act_p_seq(self.L, self.act_task, self.act_trans, self.participant_id)
        
        self.size = self.__len__()
       
  
    def __len__(self):
        return len(self.act_sec) * self.act_u_size - self.sec_size + 1
       
  
    def __getitem__(self, idx):
        start_pos = idx
        end_pos = idx + self.sec_size - 1
        
        start_sec = int(np.floor(start_pos / self.act_u_size))
        end_sec = int(np.floor(end_pos / self.act_u_size))
        
        x_container = []
        y_container = []
        for sec_i in np.arange(start_sec, end_sec + 1):
            task_id = self.act_sec[sec_i]
            p_id = self.p_sec[sec_i]
            
            x_container.append(self.container[task_id][p_id])
            y_container.append(task_id - 1) # task_id 是从1开始编号的，误差损失函数计算要求从 0 开始
            
        x_stack = np.vstack(x_container)
        x_c_start_idx = start_sec * self.act_u_size
        x_c_end_idx = (end_sec + 1) * self.act_u_size - 1
        x_pd = pd.DataFrame(x_stack, index = np.arange(x_c_start_idx, x_c_end_idx + 1))
        x = np.array(x_pd.loc[start_pos:end_pos, :].values)
        
        if len(y_container) == 1:
            y = y_container[0]
        else:
            y = 16 ## （0-15对应16个动作，16对应无识别动作）
        
        ## 更具当前数据片的位置，计算对应动作类型的置信度
        # y_p = self.act_p(start_pos, end_pos, self.act_period, self.act_p_power)
        
        # ## 处理当前时间窗口跨越两种动作的情况
        # if len(y_container) == 1:
        #     y = np.array(([y_container[0], y_p]))
        # elif len(y_container) == 2:
        #     ## 对于横跨两个类别的时间窗口
        #     mid_pos = np.mean([start_pos, end_pos])
        #     NP = np.round(mid_pos / self.act_period)
        #     ref_val = NP * self.act_period
        #     if mid_pos < ref_val:
        #         y_index = 0
        #     else:
        #         y_index = 1
            
        #     y = np.array([y_container[y_index], y_p])
        # else:
        #     print("时间窗口太长，无法有效判别动作类型")
        #     sys.exit(1)  # 1 表示程序以非零状态退出，通常表示错误
        
        return x, y
    
    def act_p(self, start_pos, end_pos, p, a):
        ## 基于幂函数构建的在连续过程中动作类型的概率表征
        x = np.arange(start_pos, end_pos)
        y = 1-np.sin(np.pi * (x + 0.5 * p) / (p) )**a
        
        return np.mean(y)
        
    
    def load_raw_data(self):
        print("Loading raw data...\n")
        data_all_x, data_all_y = get_loc_data(loc="all")
        
        container = {}
        for i, (x, y) in enumerate(zip(data_all_x, data_all_y)):
            if y not in container.keys():
                container[y] = [x]
            else:
                container[y].append(x)
        print("Loading raw data done!\n")
        return container
    
    def get_equal_next(self, freq, keys):
        '''
        ## 根据给定的 freq 记录，以及候选记录，选择下一个目标，确保 freq 中的各个项目数均衡
        ## freq: pd.Series, index为项目索引，value 为项目频次
        ## keys: 确保选中的 task_id在给定的keys中
        '''   
#         print(freq, keys)
        
        flag = np.min(freq.values)
        flag_ids = freq[freq == flag].index
        
        ## 确保选一个在给定 keys 范围内的 id
        t_i = 1
        while(t_i):
            
            task_id = np.random.choice(flag_ids)
            if task_id in keys:
                freq[task_id] = freq[task_id] + 1
                break
                
            t_i = t_i + 1
            if t_i > 10:
                print("原始动作数据部分缺失, 无法生成有效序列")
                sys.exit(1)  # 1 表示程序以非零状态退出，通常表示错误
                
        return task_id, freq

    def get_act_p_seq(self, L, act_task, act_trans, participant_id):
        '''
        ## 生成行为类型均衡，以及数据来源均衡动作序列以及实验来源序列
        ## L:为生成数据的时间长度，单位为分钟
        ## act_tack: 候选作业任务类动作
        ## act_trans: 候选过度类动作
        ## participant_id: 候选传感器数据来源
        '''

        print("生成", L, "分钟的动作数据\n")

        task_freq = pd.Series(np.zeros(len(act_task)), index=act_task) #16类活动
        trans_freq = pd.Series(np.zeros(len(act_trans)), index=act_trans)
        particpant_freq = pd.Series(np.zeros(len(participant_id)), index=participant_id)

        act_sec = []
        participant_sec = []

        while(1):

            if len(act_sec) > L:
                break
                
            ######################################
            ## 获得对应的任务动作
            task_id, task_freq = self.get_equal_next(task_freq, self.container.keys())
            act_sec.append(task_id)
            
            ## 获得对应动作的实验者序号
            p_id_task, particpant_freq = self.get_equal_next(particpant_freq, np.arange(len(self.container[task_id])))
            participant_sec.append(p_id_task)
            ######################################
            
            ######################################
            ## 获得对应任务动作的延续动作
            while(1):
                
                if len(act_sec) > L:
                    break

                ## 确定task延续的概率
                task_c_p = random.uniform(0.7, 0.8)
                ## 从0-1的均匀分布中随机产生一个数，如果小于task_c_p，则动作连续
                task_c_d = random.random()
                if task_c_d <= task_c_p:

                    task_id = act_sec[-1]
                    task_freq[task_id] = task_freq[task_id] + 1
                    act_sec.append(task_id)

                    ## 获得对应动作的实验者序号
                    p_id_task, particpant_freq = self.get_equal_next(particpant_freq, np.arange(len(self.container[task_id])))
                    participant_sec.append(p_id_task)
                else:
                    break
            ######################################

            ######################################
            ## 获得对应的过度动作
            trans_id, trans_freq = self.get_equal_next(trans_freq, self.container.keys())
            act_sec.append(trans_id)
            
            p_id_trans, particpant_freq = self.get_equal_next(particpant_freq, np.arange(len(self.container[trans_id])))
            participant_sec.append(p_id_trans)
            ######################################
            
            
            ######################################
            ## 获得过度动作的延续动作
            while(1):
                if len(act_sec) > L:
                    break
                
                ## 确定task延续的概率
                task_c_p = random.uniform(0.5, 0.8)
                ## 从0-1的均匀分布中随机产生一个数，如果小于task_c_p，则动作连续
                task_c_d = random.random()
                if task_c_d <= task_c_p:

                    trans_id = act_sec[-1]
                    trans_freq[trans_id] = trans_freq[trans_id] + 1
                    act_sec.append(trans_id)

                    ## 获得对应动作的实验者序号
                    p_id_task, particpant_freq = self.get_equal_next(particpant_freq, np.arange(len(self.container[trans_id])))
                    participant_sec.append(p_id_task)
                else:
                    break
            ######################################


        return act_sec, participant_sec

        
        
if __name__ == "__main__":
    print("WorkerActDataSet")
        
