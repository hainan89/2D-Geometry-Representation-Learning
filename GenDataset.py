# %%
import os
import pickle
from WorkerActDataset import WorkerActDataset
from WorkerActDatasetTS import WorkerActDatasetTS

# %%
import numpy as np

# %%
import time

# %%
base_dir = './data_set_train_TS'
time_flag = str(time.time())
output_dir = os.path.join(base_dir, time_flag)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# %%
ws = 1 # 滑动窗口时间单位为 秒
act_period = 480 # 单位分钟, 确保动作类别和参与者id的有足够多的样本
act_task = [1, 2, 3, 4, 8, 9, 10, 11, 12, 15] # 作业类动作
act_trans = [5, 6, 7, 16, 13, 14] # 过度类动作

## 原始的动作数据中，实验者数据是按照数组的形式存储，id下标从 0 开始
participant_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 实验参与者序号
# participant_id = [11, 12, 13]
participant_id = [int(one) for one in list(np.array(participant_id) - 1)]

train_dataset = WorkerActDatasetTS(ws, act_period, act_task, act_trans, participant_id)
  


# %%
with open(os.path.join(output_dir, "train_dataset.pkl"), 'wb') as f:
    pickle.dump(train_dataset, f)

# %%
data_set_info = {"Note":"ws单位为秒，act_period单位为分钟，act类别总共16类编号1-16，participant总共13人编号1-13，便于训练从编号0开始。Y为每个时间点标签，不采用窗口标签",
                 "ws":ws, "act_period":act_period, "act_task":act_task, "act_trans":act_trans,
                 "participant_id":participant_id}

# %%
import json

# 写入文件（自动格式化）
with open(os.path.join(output_dir, "data_set_info.json"), "w", encoding="utf-8") as f:
    json.dump(data_set_info, f, indent=4, ensure_ascii=False)


