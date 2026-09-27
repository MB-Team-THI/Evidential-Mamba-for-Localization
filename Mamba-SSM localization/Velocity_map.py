import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

folder_path = "data/rev-sted"
#Revsted consists of 7 datasets.
#dataset 6 max speed is 40Kmphr and all driving in low speed region.
#dataset 7 is high speed and more dynamic both are kept as test datset
train_csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(1,6,1)]
test_csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(6,8,1)]

train_df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in train_csv_files]
test_df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in test_csv_files]
train_df = pd.concat(train_df_list, ignore_index=True)
test_df = pd.concat(test_df_list, ignore_index=True)
#Interested columns for speed mapping learning
#INS_time_sec, ins_velCOG_hor_x, ins_velCOG_hor_y, ins_vel_hor_z, speedo_obd
interested_columns = ['INS_time_sec', "INS_velCOG_hor_x", "INS_velCOG_hor_y", "speedo_obd","LatAcc_obd",'SW_pos_obd','Yawrate_obd', 'VelFR_obd','VelFL_obd','VelRR_obd','VelRL_obd']
train_dataset = train_df[interested_columns].copy()
test_dataset = test_df[interested_columns].copy()
#Velocity magnitude at COG
train_dataset['Total_vel_COG'] = np.sqrt(train_dataset["INS_velCOG_hor_x"]**2+train_dataset["INS_velCOG_hor_y"]**2)*3.6 #Conversion to KMPH
test_dataset['Total_vel_COG'] = np.sqrt(test_dataset["INS_velCOG_hor_x"]**2+test_dataset["INS_velCOG_hor_y"]**2)*3.6

feature_cols = ["speedo_obd","LatAcc_obd",'SW_pos_obd','Yawrate_obd', 'VelFR_obd','VelFL_obd','VelRR_obd','VelRL_obd']
#Dataset splits
X_train = train_dataset[feature_cols].values
y_train = train_dataset['Total_vel_COG'].values

X_test = test_dataset[feature_cols].values
y_test = test_dataset['Total_vel_COG'].values


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=40)

#Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#Non-Linear-regression with multiple columns
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=20, random_state=42, verbose=True)
print('model-fit-nonlinear regression all obd')
model.fit(X_train_scaled, y_train)
y_pred_val = model.predict(X_val_scaled)
y_pred_test = model.predict(X_test_scaled)
val_mae = mean_absolute_error(y_val, y_pred_val)
test_mae = mean_absolute_error(y_test, y_pred_test)
print(f'Validation MAE: {val_mae:.3f}')
print(f'Test MAE: {test_mae:.3f}')
plt.scatter(X_test[:,0], y_test, color='blue', label='True')
plt.scatter(X_test[:,0], y_pred_test, color='red', label='Predicted')
plt.title('NLR Predictions vs Ground Truth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

results_df = pd.DataFrame({
    "True_Total_vel_COG": y_test,
    "OBD_based_prediction_Total_vel_COG": y_pred_test
})

results_df.to_csv("obd_velocity_Revsted_6_7.csv", index=False)
print("Saved predictions to test_predictions_vs_ground_truth.csv")

# #Non-Linear-regression with only speedometer
# model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=10, random_state=42, verbose=True)
# print('model-fit-nonlinear regression only speedometer')
# model.fit(X_train_scaled[:,0].reshape(-1,1), y_train) #0 is speedometer column
# y_pred_val = model.predict(X_val_scaled[:,0].reshape(-1,1))
# y_pred_test = model.predict(X_test_scaled[:,0].reshape(-1,1))
# val_mae = mean_absolute_error(y_val, y_pred_val)
# test_mae = mean_absolute_error(y_test, y_pred_test)
# print(f'Validation MAE: {val_mae:.3f}')
# print(f'Test MAE: {test_mae:.3f}')
# plt.scatter(X_test[:,0], y_test, color='blue', label='True')
# plt.scatter(X_test[:,0], y_pred_test, color='red', label='Predicted')
# plt.title('SVR Predictions vs Ground Truth')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
#
# #Based on the above two experiments nonlinear regression with multiple onboard sensors yield the best results
#
# #Linear Regression with multiple columns
# model = LinearRegression()
# print('model-fit-linear regression all obd')
# model.fit(X_train_scaled, y_train)
# y_pred_val = model.predict(X_val_scaled)
# y_pred_test = model.predict(X_test_scaled)
# val_mae = mean_absolute_error(y_val, y_pred_val)
# test_mae = mean_absolute_error(y_test, y_pred_test)
# print(f'Validation MAE: {val_mae:.3f}')
# print(f'Test MAE: {test_mae:.3f}')
# plt.scatter(X_test[:,0], y_test, color='blue', label='True')
# plt.scatter(X_test[:,0], y_pred_test, color='red', label='Predicted')
# plt.title('SVR Predictions vs Ground Truth')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
#
# #Linear Regression with single column
# model = LinearRegression()
# print('model-fit-linear regression only speedometer')
# model.fit(X_train_scaled[:,0].reshape(-1,1), y_train)
# y_pred_val = model.predict(X_val_scaled[:,0].reshape(-1,1))
# y_pred_test = model.predict(X_test_scaled[:,0].reshape(-1,1))
# val_mae = mean_absolute_error(y_val, y_pred_val)
# test_mae = mean_absolute_error(y_test, y_pred_test)
# print(f'Validation MAE: {val_mae:.3f}')
# print(f'Test MAE: {test_mae:.3f}')
# plt.scatter(X_test[:,0], y_test, color='blue', label='True')
# plt.scatter(X_test[:,0], y_pred_test, color='red', label='Predicted')
# plt.title('SVR Predictions vs Ground Truth')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
#
# #Speed map learning for every 5kmphr
# X = dataset[["speedo_obd","LatAcc_obd",'SW_pos_obd','Yawrate_obd', 'VelFR_obd','VelFL_obd','VelRR_obd','VelRL_obd']].values
# y = dataset['Total_vel_COG'].values
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=40)
# #speed map
# max_speed = np.ceil(dataset['speedo_obd'].max() / 5) * 5
# speed_bins = np.arange(0, max_speed + 5, 5)
# models = []
# for i in range(len(speed_bins) - 1):
#     lower = speed_bins[i]
#     upper = speed_bins[i + 1]
#
#     # Filter data within the current speed bin
#     mask = (X_train[:,0] >= lower) & (X_train[:,0]  < upper)
#     x_bin = X_train[mask,0].reshape(-1, 1)
#     y_bin = y_train[mask]
#
#     if len(x_bin) >= 2:  # Ensure enough data points to fit
#         model = LinearRegression()
#         model.fit(x_bin, y_bin)
#         models.append((lower, upper, model))
#
# y_pred_test = np.zeros_like(y_test)
# for i in range(len(X_test)):
#     speed = X_test[i, 0]
#     for lower, upper, model in models:
#         if lower <= speed < upper:
#             y_pred_test[i] = model.predict([[speed]])[0]
#             break
#     else:
#         # If speed is outside known range, default to mean or zero
#         print('outside')
#         y_pred_test[i] = 0
#
# # Compute evaluation metrics
# mae_test_speedmap = mean_absolute_error(y_test, y_pred_test)
# print('Mean absolute error in speed map', mae_test_speedmap)
# plt.scatter(X_test[:,0], y_test, color='blue', label='True')
# plt.scatter(X_test[:,0], y_pred_test, color='red', label='Predicted')
# plt.title('Speed map Predictions vs Ground Truth')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()