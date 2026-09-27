"""Current state is still experimenting with the Extended Kalman Filter
To be done:
1) Horizontation of accelerations from body
2) During GPS Outage mandatory noise in position-x,position-y,yaw angle(heading)
3) Eduardo implementation add noise to velocity and beta as well
4) Ours input velocity and beta from learnt mappings. OBD --> Mapping --> velocity , beta.
5) Horizontation of yaw rate
6) Noise matrices refinement. One without outage and one during outage
7) check state equations for low velocity division problem
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


folder_path = "data/rev-sted"
#Rev-sted 1 has GPS data outages do not consider it.
csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(3,4)]#Loading only one file
df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

computed_df = pd.read_csv('computed_inertial_nav_data.csv')

combined_df.loc[:29999, 'computed_a_hor_x'] = computed_df['computed_a_hor_x']
combined_df.loc[:29999, 'computed_a_hor_y'] = computed_df['computed_a_hor_y']
combined_df.loc[:29999, 'computed_yaw_angle'] = computed_df['computed_yaw_angle']


# plt.figure(figsize=(10, 5))
# plt.plot(combined_df['Correvit_vehiclesideslip_angle_COG'], label='Correvit Slip Angle (COG)')
# plt.plot(combined_df['INS_slip_angle_COG'], label='INS Slip Angle (COG)')
# plt.xlabel('Time Step')
# plt.ylabel('Slip Angle (degrees)')
# plt.title('Comparison of Slip Angles: Correvit vs INS fro dataset quality check')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# We need to implement Extended Kalman Filter for this purpose as state equations are Non-Linear.

# Define state equations and state vector.
x_s_1 = np.zeros(9)
#Format of state is [xs (0),ys (1),θyaw (2),θ'yaw (3),vs (4),βsv (5),βdot (6),ax (7),ay (8)]
def state_equations(x_s_1,delta_t):
    theta_yaw = x_s_1[2]
    theta_yaw_rate = x_s_1[3]
    vs = x_s_1[4]
    beta_sv = x_s_1[5]
    beta_dot = x_s_1[6]
    ax = x_s_1[7]
    ay = x_s_1[8]
    #New state vector
    x_s = np.zeros_like(x_s_1)
    #Position Update
    x_s[0] = x_s_1[0] + vs * np.cos(theta_yaw + beta_sv) * delta_t + ((ax * np.cos(theta_yaw) - ay * np.sin(theta_yaw)) * (delta_t**2) / 2)
    x_s[1] = x_s_1[1] + vs * np.sin(theta_yaw + beta_sv) * delta_t + ((ax * np.sin(theta_yaw) + ay * np.cos(theta_yaw)) * (delta_t ** 2) / 2)
    #print('X_state',x_s[0], 'Y_state', x_s[1])
    #Yaw angle and yaw rate
    x_s[2] = theta_yaw + theta_yaw_rate*delta_t
    x_s[3] = theta_yaw_rate
    #Velocity and sideslip angles
    x_s[4] = vs + (ax * np.cos(beta_sv) + ay * np.sin(beta_sv))*delta_t
    x_s[5] = beta_sv + beta_dot * delta_t #Beta_dot can be computed from single track model equations
    x_s[6] = (1/vs *  (ay * np.cos(beta_sv) - ax * np.sin(beta_sv))) - theta_yaw_rate
    #Constant acceleration assumption
    x_s[7] = ax
    x_s[8] = ay
    if x_s[4]<1:
        x_s[5] = 0
        x_s[6] = 0
    return x_s
#Format of measurement is [xh (0),yh (1),θyaw (2),θ'yaw (3),vh (4),βh (5), ax (6),ay (7)]
#Signs are verified
def measurement_vector(data,iter):
    z_h = np.zeros(8)
    z_h[0] = data['ins_pos_rel_x'].iloc[iter]
    z_h[1] = data['ins_pos_rel_y'].iloc[iter]
    z_h[2] = -1*np.deg2rad(data['ins_yaw'].iloc[iter])#-makes it converted to equations coordinate system and verified
    #z_h[2] = -1 * np.deg2rad(data['computed_yaw_angle'].iloc[iter])
    z_h[3] = -1*np.deg2rad(data['rate_hor_z'].iloc[iter])#-makes it converted to equations coordinate system and verified
    #z_h[3] = -1 * np.deg2rad(data['rate_body_hr_z'].iloc[iter])  # -makes it converted to equations coordinate system and verified
    z_h[4] = data[('INS_totalvelCOG_hor')].iloc[iter]#This should be from Speedometer, currently test dataset checked with speedometer
    z_h[5] = 1*np.deg2rad(data[('Correvit_vehiclesideslip_angle_COG')].iloc[iter])#This should be from UAHI learnt with correvit, mostly negative sign
    z_h[6] = data[('acc_hor_x')].iloc[iter]*9.81     #Positive and 9.81 verified
    z_h[7] = -1*data[('acc_hor_y')].iloc[iter]*9.81  #Negative and 9.81 verified
    #z_h[6] = data[('computed_a_hor_x')].iloc[iter]*9.81    #Positive and 9.81 verified
    #z_h[7] = -1*data[('computed_a_hor_y')].iloc[iter]*9.81 #Negative and 9.81 verified
    #z_h[6] = data[('acc_body_hr_x')].iloc[iter]*9.81   #Positive and 9.81 verified
    #z_h[7] = -1*data[('acc_body_hr_y')].iloc[iter]*9.81 #Negative for equations corrdinate system and 9.81 verified
    return z_h

#Return predicted measurement vector from state [xh (0),yh (1),θyaw (2),θ'yaw (3),vh (4),βh (5), ax (6),ay (7)]
def h(x):
    z_pred = np.zeros(8)
    z_pred[0] = x[0]   # x
    z_pred[1] = x[1]   # y
    z_pred[2] = x[2]   # yaw
    z_pred[3] = x[3]   # yaw_rate
    z_pred[4] = x[4]   # v_s
    z_pred[5] = x[5]   # beta
    z_pred[6] = x[7]   # ax
    z_pred[7] = x[8]   # ay
    return z_pred

# Noise matrix should be tailor made for working
n = 9 #State Dimension
m = 8 #Measurement Dimension
P = np.eye(n)*0.001  #Initial state covariance
Q = np.eye(n)*0.01 #Process noise covariance
std_dev_measurments = [.01,.01,.00026,.00005,.0083,.05, .03,.03] #Measurement noise std with satellites
R_s = np.diag([sd**2 for sd in std_dev_measurments]) #R with satellite
R_o = np.diag([sd**2 for sd in std_dev_measurments]) #R during satellite outage
R_o[0, 0] = 1e6 #Position x corruption
R_o[1, 1] = 1e6 #Position y corruption
R_o[2, 2] = 1e6 #Yaw angle orientation Corruption GPS
R_o[4, 4] = 1e6 #Velocity corruption
R_o[5, 5] = 1e6 #Beta corruption

def ekf_predict(x, P, delta_t):
    F = numerical_jacobian_f(x, delta_t)
    x_pred = state_equations(x, delta_t)#Predicted state estimate
    P_pred = F @ P @ F.T + Q#Predicted state covariance
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z_meas,use_satellite=True):
    H = numerical_jacobian_h(x_pred)
    z_pred = h(x_pred)
    y = z_meas - z_pred
    R_used = R_s if use_satellite else R_o #Decides the noise matrix
    S = H @ P_pred @ H.T + R_used
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_new, P_new

def numerical_jacobian_f(x, dt, eps=1e-5):
    n = len(x)
    F = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        fx_plus = state_equations(x_plus, dt)
        fx_minus = state_equations(x_minus, dt)
        F[:, i] = (fx_plus - fx_minus) / (2 * eps)
    return F

def numerical_jacobian_h(x, eps=1e-5):
    m = len(h(x))
    n = len(x)
    H = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        h_plus = h(x_plus)
        h_minus = h(x_minus)
        H[:, i] = (h_plus - h_minus) / (2 * eps)
    return H


# From one iter point calculate all and check code.
iter_start = 5000 #int(input('Iter point of the data'))
outage = 50*60*5
#Initialization
delta_t = 0.02  #0.02 Or get from timestamp differences if available

# Initialize state from measurement at iter_start
z_init = measurement_vector(combined_df, iter_start)
x_s_1 = np.zeros(9)
x_s_1[0] = z_init[0]
x_s_1[1] = z_init[1]
x_s_1[2] = z_init[2]
x_s_1[3] = z_init[3]
x_s_1[4] = z_init[4]
x_s_1[5] = z_init[5]
x_s_1[6] = 0
x_s_1[7] = z_init[6]
x_s_1[8] = z_init[7]

# Run state update over 250 iterations
trajectory = [x_s_1.copy()]
for i in range(iter_start + 1, iter_start + outage+1):
    dt = delta_t
    # Predict
    x_pred, P_pred = ekf_predict(x_s_1, P, dt)
    # Measurement
    z = measurement_vector(combined_df, i)
    # Update
    x, P = ekf_update(x_pred, P_pred, z, use_satellite=False)
    x_s_1 = x.copy() #Recursion
    trajectory.append(x.copy())

# Convert to DataFrame for easier analysis/plotting
trajectory_df = pd.DataFrame(trajectory, columns=['xs', 'ys', 'theta_yaw', 'theta_yaw_rate',
                                                  'vs', 'beta_sv', 'beta_dot', 'ax', 'ay'])
measured_x = combined_df['ins_pos_rel_x'].iloc[iter_start:iter_start+outage+1].values
measured_y = combined_df['ins_pos_rel_y'].iloc[iter_start:iter_start+outage+1].values

estimated_x = trajectory_df['xs'].values
estimated_y = trajectory_df['ys'].values
drift = np.sqrt((estimated_x[-1] -measured_x[-1])**2+(estimated_y[-1] -measured_y[-1])**2)
drift_per_step = np.sqrt((estimated_x - measured_x)**2 + (estimated_y - measured_y)**2)
max_drift = np.max(drift_per_step)
print('drift_final', drift)
print('drift_max_during_outage', max_drift)
plt.figure(figsize=(10, 6))
plt.plot(measured_x, measured_y, label='Measured INS Position', linestyle='--', color='blue')
plt.plot(estimated_x, estimated_y, label='Estimated Position during outage', color='orange')
drift_text = f'Max Drift Outage: {max_drift:.2f} m'
text_x = min(plt.xlim()) + 1
text_y = max(plt.ylim()) - 10
plt.text(text_x, text_y, drift_text, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Trajectory Comparison: Estimated vs Measured')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()