import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
#from sklearn.preprocessing import StandardScaler
from utils.tools import StandardScaler

class KalmanAwareDataset(Dataset):
    def __init__(
        self,df_data,
        config,
        split,inverse=True):

        assert split in ['train', 'val', 'test'], "split must be one of ['train', 'val', 'test']"
        self.split = split
        self.scale = config["data"]["scale"]
        self.inverse = inverse
        self.input_cols = config["data"]["input_columns"]
        self.target_cols = config["data"]["target_columns"]
        self.pose_cols = config["data"]["pose_columns"]

        #Prepare the split
        self.split_ratios = config["data"]["split_indices"]
        self.dataset_length = len(df_data)
        self.split_indices = self.read_splits()
        border1, border2 = self.split_indices[self.split]
        df_split = df_data.iloc[border1:border2]

        #Raw values before scaling
        self.data_time = df_split[["INS_time_sec"]].values

        #Store scalars
        self.scaler_input = None
        self.scaler_target = None

        if self.scale:
            train_idx = self.split_indices["train"]
            df_train = df_data.iloc[train_idx[0]:train_idx[1]]

            self.scaler_input = StandardScaler()
            self.scaler_target = StandardScaler()

            self.scaler_input.fit(df_train[self.input_cols].values)
            self.scaler_target.fit(df_train[self.target_cols].values)

            df_input = pd.DataFrame(self.scaler_input.transform(df_split[self.input_cols]), columns=self.input_cols)
            df_target = pd.DataFrame(self.scaler_target.transform(df_split[self.target_cols]), columns=self.target_cols)

        else:
            df_input = df_split[self.input_cols]
            df_target = df_split[self.target_cols]

        #pose is for kalman and need not be inversed
        df_pose = df_split[self.pose_cols]
        self.data_input = df_input.values
        self.data_target = df_target.values
        self.data_pose = df_pose.values

        self.window = config["data"]["window_size"]
        self.history = config["data"]["history_size"]
        self.prediction_length = self.window - self.history

    def read_splits(self):
        ratios = [self.split_ratios["train"]/100,self.split_ratios["val"]/100,self.split_ratios["test"]/100]#make it ratios
        train = [0,int(self.dataset_length*ratios[0])]
        val = [int(self.dataset_length * ratios[0]),int(self.dataset_length*(ratios[0]+ratios[1]))]
        test = [int(self.dataset_length*(ratios[0]+ratios[1])), int(self.dataset_length)]
        return {"train":train, "val":val, "test":test}


    def __len__(self):
        return len(self.data_input) - self.window + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.window
        x_seq = self.data_input[s_begin:s_end]
        y_seq = self.data_target[s_begin:s_end]
        p_seq = self.data_pose[s_begin:s_end]
        return x_seq,y_seq,p_seq

    def inverse_input(self, data):
        if self.scaler_input:
            return self.scaler_input.inverse_transform(data)
        return data

    def inverse_target(self, data):
        if self.scaler_target:
            return self.scaler_target.inverse_transform(data)
        return data

    def inverse_pose(self, data):
        if self.scaler_pose:
            return self.scaler_pose.inverse_transform(data)
        return data
