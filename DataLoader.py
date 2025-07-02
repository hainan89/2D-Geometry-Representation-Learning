'''
## 加载VTT_ConIOT数据集中的数据
'''
import pickle
import yaml
import os
import numpy as np
import pandas as pd

def load_config():
    with open('./config.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
        
    return data_config

def load_raw_data(fpath):
    data = pd.read_csv(fpath)
    
    return data

def load_raw_a_u_data(a_i, u_i):
    cfg = load_config()
    
    fpath = os.path.join(cfg['data_folder'], f"activity_{a_i}_user_{u_i}_combined.csv")
    
    if os.path.exists(fpath):
        data = load_raw_data(fpath)

        return data
    
def extend_matrix_to_6000_rows(matrix):
    # 获取矩阵的当前行数
    num_rows, num_cols = matrix.shape
    
    # 如果行数已经等于或超过100行，不需要扩展
    if num_rows == 6000:
        return matrix
    
    if num_rows > 6000:
        return matrix[0:6000, :]
    
    # 计算需要扩展的行数
    additional_rows = 6000 - num_rows
    
    # 扩展矩阵，复制最后一行 additional_rows 次
    extended_matrix = np.vstack([matrix, np.tile(matrix[-1], (additional_rows, 1))])
    
    return extended_matrix

from sklearn.preprocessing import MinMaxScaler

def get_a_u_loc(a_i, u_i, loc="trousers"):
    data = load_raw_a_u_data(a_i, u_i)
    
    if loc == "all":
        loc_list = ["trousers", "back", "hand"]
    else:
        loc_list = [loc]
    
    features = []
    for one_loc in loc_list:
        features_sec = [f"{one_loc}_Gx_dps",f"{one_loc}_Gy_dps",f"{one_loc}_Gz_dps",
                    f"{one_loc}_Ax_g",f"{one_loc}_Ay_g",f"{one_loc}_Az_g",
                    f"{one_loc}_tot_g", f"{one_loc}_tot_dps", f"{one_loc}_mbar"]

        features.extend(features_sec)
    
    r = data.loc[:,features].interpolate(method='linear', axis=0)
    r = extend_matrix_to_6000_rows(r.values)
    
    scaler = MinMaxScaler()
    r = scaler.fit_transform(r)
    
    return r

def get_loc_data(loc="trousers"):
    data_x = []
    data_y = []
    
    for u_i in np.arange(1, 14):
        for a_i in np.arange(1, 17):
            if a_i == 11 and u_i == 6:
                continue
            data_x.append(get_a_u_loc(a_i=a_i, u_i=u_i, loc=loc))
            data_y.append(a_i)
    return data_x, data_y




