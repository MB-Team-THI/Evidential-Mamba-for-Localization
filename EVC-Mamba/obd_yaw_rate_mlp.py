from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

#data = pd.read_csv('/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/ICRA_2026/Kalman_Pose/data/Revsted2/ReVSTEDv2_5.csv')
df = pd.read_csv("preprocessed_dataset.csv")

n = len(df)
n_train = int(n * 0.5)
n_val   = int(n * 0.3)
n_test  = n - n_train - n_val

df['Yawrate_obd'] = df['Yawrate_obd'] * -1 * (np.pi / 180)
# Split
df_train = df.iloc[:n_train]
df_val   = df.iloc[n_train:n_train+n_val]
df_test  = df.iloc[n_train+n_val:]


plt.figure(figsize=(12,4))
plt.plot(df_val['rate_hor_z'], label='Good Sensor Yawrate')
plt.plot(df_val['Yawrate_obd'], label='OBD Yawrate')
plt.xlabel('Sample index')
plt.ylabel('Yaw rate [rad/s]')
plt.title('Comparison of OBD and Good Sensor Yaw Rate')
plt.legend()
plt.show()

features = ["Yawrate_obd", "LatAcc_obd", "brake_pressure_obd", "speedo_obd", "SW_pos_obd", "VelFR_obd", "VelFL_obd", "VelRR_obd", "VelRL_obd"]
features = ["Yawrate_obd"]
X_train = df_train[features].values
y_train = df_train["rate_hor_z"].values

X_val = df_val[features].values
y_val = df_val["rate_hor_z"].values

X_test = df_test[features].values
y_test = df_test["rate_hor_z"].values


poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),  # increase degree if needed
    ("linreg", LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)
joblib.dump(poly_model, "obd_yawrate_poly_model.pkl")
print("poly_model_done")
# MLP pipeline with scaling
mlp_model = Pipeline([
    ("scaler", StandardScaler()),  # scale input for better training
    ("mlp", MLPRegressor(hidden_layer_sizes=(128,32),
                         activation='relu',  tol=1e-7,
                         max_iter=100,verbose=True,
                         random_state=42))
])

y_baseline_pred = df_test["Yawrate_obd"].values

# Compute RMSE against the ground truth
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline_pred))
print("Baseline Test RMSE (OBD only):", baseline_rmse)

# Train
mlp_model.fit(X_train, y_train)

# Evaluate
y_val_pred = mlp_model.predict(X_val)
y_test_pred = mlp_model.predict(X_test)

joblib.dump(mlp_model, "obd_to_ins_yaw_mlp_model.pkl")
print('model_saved')
print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))