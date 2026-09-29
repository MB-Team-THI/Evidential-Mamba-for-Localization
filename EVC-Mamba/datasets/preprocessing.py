import pandas as pd
import yaml
from typing import List
import os
from datasets.dataset_loader import KalmanAwareDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

def load_config(path:str) -> dict:
    with open(path,'r') as f:
        return yaml.safe_load(f) #Returns as a python dictionary

def load_and_concatenate(input_dir: str, file_pattern: str, file_indices: List[int], selected_columns: List[str]) -> pd.DataFrame:
    df_list = []
    for idx in file_indices:
        file_path = os.path.join(input_dir, file_pattern.format(idx))
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=selected_columns)
            df_list.append(df)
        else:
            print(f"Warning: File not found - {file_path}")

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=selected_columns)


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[columns]

def preprocess_dataset(config:dict)-> pd.DataFrame:
    config = config
    if config["preprocess_dataset"]["input_dir"] == 'data/Revsted_ICE':
        df = load_and_concatenate(
                input_dir=config["preprocess_dataset"]["input_dir"],
                file_pattern=config["preprocess_dataset"]["file_pattern"],
                file_indices=[i for i in range(1,2)],
                selected_columns=config["preprocess_dataset"]["selected_columns"]
               )

    else:
        df = load_and_concatenate(
                input_dir=config["preprocess_dataset"]["input_dir"],
                file_pattern=config["preprocess_dataset"]["file_pattern"],
                file_indices=[i for i in range(1,8)],
                selected_columns=config["preprocess_dataset"]["selected_columns"]
               )
    columns = config["preprocess_dataset"]["selected_columns"]
    df = df.loc[:, columns]
    df["acc_hor_x"] = df["acc_hor_x"] * 9.81
    df["acc_hor_y"] = df["acc_hor_y"] * -9.81
    df["rate_hor_z"] = df["rate_hor_z"] * -1*(np.pi / 180)
    df["ins_yaw"] = df["ins_yaw"] * -1* (np.pi / 180)

    print("Hi")

    # save_path = "preprocessed_dataset.csv"
    # df.to_csv(save_path, index=False)
    # print(f"Preprocessed dataset saved to {save_path}")



    # delta_x_list = []
    # delta_y_list = []
    # window_size=100
    # for start in range(0, len(df), 100):
    #     end = start + window_size
    #     window = df.iloc[start:end]
    #
    #     if len(window) < window_size:
    #         break  # skip incomplete window at the end
    #
    #     # Compute relative displacements
    #     delta_x = window['ins_pos_rel_x'].values - window['ins_pos_rel_x'].values[0]
    #     delta_y = window['ins_pos_rel_y'].values - window['ins_pos_rel_y'].values[0]
    #
    #     # Stack as 2 columns: shape (100, 2)
    #     delta_xy = np.stack([delta_x, delta_y], axis=1)
    #
    #     delta_x_list.append(delta_xy)
    #
    # # Combine all windows
    # delta_xy_all = np.concatenate(delta_x_list, axis=0)
    #
    # # Convert to DataFrame
    # delta_df = pd.DataFrame(delta_xy_all, columns=['delta_x', 'delta_y'])
    # delta_df['delta_x'] = np.where((delta_df['delta_x'] > 50) | (delta_df['delta_x'] < -50), 0, delta_df['delta_x'])
    # delta_df['delta_y'] = np.where((delta_df['delta_y'] > 50) | (delta_df['delta_y'] < -50), 0, delta_df['delta_y'])
    # import matplotlib.pyplot as plt
    #
    # # Assuming delta_df already exists
    # plt.figure(figsize=(8, 6))
    # plt.plot(delta_df['delta_x'], label='Delta X')
    # plt.plot(delta_df['delta_y'], label='Delta Y')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Displacement (m)')
    # plt.title('Relative Displacements over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return df.loc[:,columns] #Helps in ordering the columns

if __name__ == "__main__":
    data = preprocess_dataset("./configs/base.yaml")
    config = load_config("./configs/base.yaml")
    dataset = KalmanAwareDataset(
        df_data=data,
        config=config,
        split='train',  # or 'val', 'test'
        inverse=False
    )
    print(data.head())
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Test: iterate and print one batch
    for i, (x, y, p) in enumerate(loader):
        print("Input batch shape:", x.shape)
        print("Target batch shape:", y.shape)
        print("Pose batch shape:", p.shape)
        print("Input sample:", x[0])
        print("Target sample:", y[0])
        print("Pose sample:", p[0])
        break