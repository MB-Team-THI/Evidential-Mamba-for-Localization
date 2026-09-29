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
from scipy.spatial.transform import Rotation as R_V
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_V
from scipy.linalg import expm, svd

#Outage induced for only dataset 6,7 in REV-STED
#Ensure the OBD data is rightly added

folder_path = "data/Revsted2"
#Rev-sted 1 has GPS data outages do not consider it.
csv_files = [f'ReVSTEDv2_{i}.csv' for i in range(6,8)]#Loading only one file
df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

#Aligning the combined df to the test dataset
start_idx = 48060 - 150 - 250
combined_df = combined_df.iloc[start_idx:-150-250].reset_index(drop=True)

#Combine Speedo measurements
vel = np.load('results/evidential-transformer-results/pred.npy')
vel_df = pd.DataFrame(vel, columns=['vx', 'vy'])

obd_data = np.load('results/evidential-transformer-results/gt_poses.npy')
gt_pose_columns = ['gt_x', 'gt_y', 'gt_yaw', 'gt_yawrate', 'gt_vx', 'gt_vy', 'gt_ax', 'gt_ay', "obd_yawrate","LatAcc_obd", "brake_pressure_obd", "speedo_obd", "SW_pos_obd","VelFR_obd","VelFL_obd","VelRR_obd","VelRL_obd","ins_pitch","ins_roll"]
obd_data = pd.DataFrame(obd_data, columns=gt_pose_columns)
obd_data['vx'] = ((obd_data['VelFR_obd'] + obd_data['VelFL_obd'] +
                  obd_data['VelRR_obd'] + obd_data['VelRL_obd']) / 4)*5/18

obd_data['vy'] = ((obd_data['VelFR_obd'] - obd_data['VelFL_obd'] +
                  obd_data['VelRR_obd'] - obd_data['VelRL_obd']) / 4)*-5/18


vel_var = np.load('results/evidential-transformer-results/variance.npy')
vel_var_obd_df = pd.DataFrame(vel_var, columns=['vx_delta', 'vy_delta'])
combined_df['vx'] = vel_df['vx']
combined_df['vy'] = vel_df['vy']
combined_df['vx_obd_vmm'] = obd_data['vx']
combined_df['vy_obd_vmm'] = obd_data['vy']
combined_df['TotalVelCOG_OBD'] = np.sqrt(vel_df['vx']**2 + vel_df['vy']**2)
beta = -np.degrees(np.arctan2(vel_df['vy'], vel_df['vx']))
combined_df['VSA_OBD'] = (beta + 180) % 360 - 180

#Uncertainty calculation for vel and beta
combined_df['vel_uncertainty'] = (vel_var_obd_df['vx_delta'] + vel_var_obd_df['vy_delta'])
vx = vel_df['vx']
vy = vel_df['vy']
vx_var = vel_var_obd_df['vx_delta']
vy_var = vel_var_obd_df['vy_delta']
denom = (vx**2 + vy**2)
beta_uncertainty = np.sqrt((vy**2 * vx_var + vx**2 * vy_var) / (denom**2))
combined_df['beta_uncertainty'] = beta_uncertainty
combined_df.loc[combined_df['TotalVelCOG_OBD'] < 1, 'beta_uncertainty'] = np.float32(0.1)
combined_df.loc[combined_df['TotalVelCOG_OBD'] < 1, 'VSA_OBD'] = np.float32(0)
combined_df['Correvit_total_vel_COG'] = np.sqrt(combined_df['Correvit_Vel_COG_x']**2+combined_df['Correvit_Vel_COG_y']**2)
combined_df['vel_abs_error'] = np.abs(combined_df['Correvit_total_vel_COG'] - combined_df['TotalVelCOG_OBD'])

# plt.figure()
# plt.plot(combined_df['vel_abs_error'], label='Absolute Velocity Error')
# plt.plot(np.sqrt(combined_df['ins_vel_hor_x']**2+combined_df['ins_vel_hor_y']**2), label='Absolute Velocity IMU+RTK')
# plt.plot(1*combined_df['vel_uncertainty'], label='Velocity Uncertainty')
# plt.xlabel('Time / Index')
# plt.ylabel('Velocity (m/s)')
# plt.title('Velocity Error vs Uncertainty')
# plt.legend()
# plt.grid()
# plt.show()

deg2rad = np.pi / 180
rad2deg = 180 / np.pi

#Finding out the rotation matrix between IMU and Vehicle

#If the vehicle is on a slightly tilted road then this would cause roll and pitch alignment to hor vehicle frame.
#But we might have the same correction to world frame as well due to INS roll pitch yaw.

######################### Bias + tilt Correction procedue 1#########################################
#Lets assume ADMA is pointing correctly similar to vehicle and get bias
A_imu = []
A_veh = []

W_imu = []
W_veh = []

for i in range(1000,5000): # vehicle standstill from plots
    omega_imu = np.array([
        combined_df["rate_body_hr_x"].iloc[i]*deg2rad ,
        -1*combined_df["rate_body_hr_y"].iloc[i]*deg2rad ,
        -1*combined_df["rate_body_hr_z"].iloc[i]*deg2rad
    ])

    acc_imu = np.array([
        combined_df['acc_body_hr_x'].iloc[i]*9.81,
        -1*combined_df['acc_body_hr_y'].iloc[i]*9.81,
        -1*combined_df['acc_body_hr_z'].iloc[i]*9.81
    ])

    omega_veh = np.array([
        combined_df["rate_hor_x"].iloc[i]*deg2rad ,
        combined_df["rate_hor_y"].iloc[i]*deg2rad ,
        combined_df["rate_hor_z"].iloc[i]*deg2rad
    ])

    acc_veh = np.array([
        combined_df['acc_hor_x'].iloc[i]*9.81,
        combined_df['acc_hor_y'].iloc[i]*9.81,
        combined_df['acc_hor_z'].iloc[i]*9.81
    ])

    A_imu.append(acc_imu)
    A_veh.append(acc_veh)

    W_imu.append(omega_imu)
    W_veh.append(omega_veh)

A_imu_raw = np.array(A_imu)
A_veh_raw = np.array(A_veh)
A_imu_norm = A_imu_raw / np.linalg.norm(A_imu, axis=1, keepdims=True)
A_veh_norm = A_veh_raw / np.linalg.norm(A_veh, axis=1, keepdims=True)

W_imu = np.array(W_imu)


def estimate_R(A, B):
    H = np.zeros((3,3))
    for a, b in zip(A, B):
        H += np.outer(b, a)

    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R

R_acc = estimate_R(A_imu_norm, A_veh_norm)
print(R_acc)
roll_est = np.arctan2(R_acc[2,1], R_acc[2,2])
pitch_est = np.arctan2(-R_acc[2,0], np.sqrt(R_acc[2,1]**2 + R_acc[2,2]**2))

R_body_hf = R_V.from_euler('ZYX',[0.0, pitch_est, roll_est],degrees=False).as_matrix()
R_vehicle_hf = R_V.from_euler('ZYX',[0.0, np.deg2rad(0.34), np.deg2rad(-1.04)],degrees=False).as_matrix()
R_v_b = R_body_hf.T @ R_vehicle_hf
r = R_V.from_matrix(R_v_b)

roll, pitch, yaw = r.as_euler('ZYX', degrees=False)

print("roll-bodyimu-vehicle:", roll)
print("pitch-bodyimu-vehicle:", pitch)
print("yaw-bodyimu-vehicle:", yaw)

##Bias Estimation###
A_imu_to_veh = (R_acc @ A_imu_raw.T).T

# Mean acceleration in vehicle frame
a_mean_vehicle = np.mean(A_imu_to_veh, axis=0)
a_mean_body = np.mean(A_imu_raw, axis=0)
# Define gravity in vehicle frame
g = np.array([0, 0, -9.81]) # adjust sign if needed

# Accelerometer bias vehicle
b_acc_veh = a_mean_vehicle - g
print("Accelerometer bias in vehicle frame:", b_acc_veh)

# Accelerator bias in body frame.
# Logic bring acceleration global to body and then find bias
roll  = combined_df["ins_roll"].iloc[1000:5000].to_numpy() * deg2rad
pitch = -1*combined_df["ins_pitch"].iloc[1000:5000].to_numpy() * deg2rad
yaw   = -1*combined_df["ins_yaw"].iloc[1000:5000].to_numpy() * deg2rad



R_list = []
for r, p, y in zip(roll, pitch, yaw):

    R_list.append(R_V.from_euler('ZYX', [y, p, r], degrees=False).as_matrix())

R_mean = np.mean(R_list, axis=0)
U, _, Vt = np.linalg.svd(R_mean)
R_vg = U @ Vt
R_gv = R_vg.T
#g_vehicle_expected = R_v_b @ R_gv @ g # Rotating to vehicle frame even considering 2 degreees adma misalignment
g_vehicle_expected = R_gv @ g
b_acc_body = a_mean_body - g_vehicle_expected
b_acc_body = b_acc_body - [-.003,+.000,0] #[-.003,-.004,0] #x = -.003
print("Accelerometer bias in body frame:", b_acc_body) # Correct bias


# Zero Bias Gyroscope
W_imu = np.array(W_imu)
bias_gyro_standstill = np.mean(W_imu, axis=0)
bias_gyro_standstill = bias_gyro_standstill
print("Gyro bias in body frame radians:", bias_gyro_standstill )




deg2rad = np.pi / 180
rad2deg = 180 / np.pi
#Error state EKF.
# Define state equations and state vector.
x_nom = {
    "p": np.zeros(3),
    "v": np.zeros(3),
    "R": np.eye(3),
    "b_g": np.zeros(3),
    "b_a": np.zeros(3)
}
R_enu_from_ned = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

def eskf_predict(x_nom, omega_m, acc_m, dt):
    R = x_nom["R"]
    v = x_nom["v"]
    p = x_nom["p"]
    bg = x_nom["b_g"]
    ba = x_nom["b_a"]

    # bias corrected IMU
    omega = omega_m - bg
    acc = acc_m - ba

    # rotation update
    R_new = update_rotation(R, omega, dt)
    # acceleration in world frame
    a_world = R @ acc + np.array([0, 0, +9.81])
    #print("acc_world",a_world,"acc_body",acc)
    # velocity update
    # midpoint velocity
    #v_mid = v + 0.5 * a_world * dt

    v_new = v + a_world * dt
    # position update
    p_new = p +  v * dt + 0.5 * a_world * dt**2
    return {
        "p": p_new,
        "v": v_new,
        "R": R_new,
        "b_g": bg,
        "b_a": ba
    }


def imu_input(data, i):
    omega_imu = np.array([
        data["rate_body_hr_x"].iloc[i]*deg2rad ,
        -1*data["rate_body_hr_y"].iloc[i]*deg2rad ,
        -1*data["rate_body_hr_z"].iloc[i]*deg2rad
    ])

    acc_imu = np.array([
         1*data['acc_body_hr_x'].iloc[i]*9.81,
        -1*data['acc_body_hr_y'].iloc[i]*9.81,
        -1*data['acc_body_hr_z'].iloc[i]*9.81])

    acc_imu_bias_corrected = acc_imu - b_acc_body
    return omega_imu - bias_gyro_standstill, acc_imu_bias_corrected

def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def delta_q(omega, dt):
    theta = np.linalg.norm(omega * dt)
    if theta < 1e-8:
        return np.array([1, 0, 0, 0])

    axis = omega / np.linalg.norm(omega)
    half = theta / 2.0

    return np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half)
    ])

def q_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def orthonormalize(R):
    # Project matrix back to SO(3)
    U, _, Vt = svd(R)
    R_ortho = U @ Vt

    # Ensure right-handedness (det = +1)
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt

    return R_ortho

def R_to_q(R):
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])

        if i == 0:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S

        elif i == 1:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S

        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S

    return np.array([w, x, y, z])

# def update_rotation(R, omega, dt):
#     omega_hat = skew(omega * dt)
#     R_new = R @ expm(omega_hat)
#     # enforce SO(3) constraint (numerical stability)
#     R_new = orthonormalize(R_new)
#     return R_new

def update_rotation(R, omega, dt):
    theta = omega * dt
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        # First-order approximation (avoids numerical issues)
        dR = np.eye(3) + skew(theta)
    else:
        k = theta / angle
        K = skew(k)
        dR = (
            np.eye(3)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K)
        )
    R_new = R @ dR
    # enforce SO(3)
    R_new = orthonormalize(R_new)
    return R_new

#def update_rotation(R, omega, dt):
    #return R_V.from_euler('ZYX', R_V.from_matrix(R).as_euler('ZYX', degrees=False) + omega * dt, degrees=False).as_matrix()

# def update_rotation(R, omega, dt):
#     q = R_to_q(R)
#     dq = delta_q(omega, dt)
#     q_new = q_mul(q, dq)
#     q_new = q_new / np.linalg.norm(q_new)
#     return q_to_R(q_new)

def q_to_R(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

print("check measurement vector")
def measurement_vector(data, i):
    #ins_yaw should be negative
    #I am not sure about vy_obd sign, currently keeping it negative based on two tests in 5 min outages
    z = np.zeros(5)
    z[0] = data['ins_pos_rel_x'].iloc[i]
    z[1] = data['ins_pos_rel_y'].iloc[i]
    z[2] = -1*np.deg2rad(data['ins_yaw'].iloc[i])
    z[3] = data['vx_obd_vmm'].iloc[i]
    z[4] = -1 * data['vy_obd_vmm'].iloc[i]
    #z[3] = data['Correvit_Vel_COG_x'].iloc[i]
    #z[4] = -1*data['Correvit_Vel_COG_y'].iloc[i] #'vy_obd'
    return z

dx = np.zeros(15) # 3 pos + 3 vel + 3 attitude + 3 gyro bias + 3 accel bias
P = np.eye(15) * 0.01

# This build_f version was 1st, 2nd version seems to work better but no clue why?
# def build_F(dt, R, acc, omega):
#     F = np.zeros((15, 15))
#     # position error
#     F[0:3, 3:6] = np.eye(3)
#     # velocity error
#     F[3:6, 6:9] = -R @ skew(acc)
#     # attitude error
#     F[6:9, 6:9] = -skew(omega)
#     return np.eye(15) + F * dt

def build_F(dt, R, acc, omega):
    F = np.zeros((15, 15))
    # position
    F[0:3, 3:6] = np.eye(3)
    # velocity
    F[3:6, 6:9] = -R @ skew(acc)
    F[3:6, 12:15] = -R   # accel bias effect
    # attitude
    F[6:9, 6:9] = -skew(omega)
    F[6:9, 9:12] = -np.eye(3)  # gyro bias effect
    return np.eye(15) + F * dt

#Q matrix initialization
Q = np.zeros((15,15))
Q[0:3, 0:3] = 0.01 * np.eye(3)
Q[3:6, 3:6] = 0.01 * np.eye(3)     # accel noise → velocity
Q[6:9, 6:9] = 0.01 * np.eye(3)     # gyro noise → attitude
Q[9:12, 9:12] = 1e-3 * np.eye(3)   # gyro bias RW
Q[12:15, 12:15] = 1e-3 * np.eye(3) # accel bias RW





#need to check angle wrapping for this yaw as measrument yaw is from 0 to 360 to np.radians.

def h(x_nom):
    p = x_nom["p"]
    v = x_nom["v"]
    yaw = np.arctan2(x_nom["R"][1,0], x_nom["R"][0,0])
    R = x_nom["R"]
    v_body = R.T @ v

    return np.array([
        p[0],
        p[1],
        yaw,
        v_body[0],
        v_body[1]
    ])

# def H_matrix(x_nom, z):
#     H = np.zeros((5, 15))
#     R = x_nom["R"]
#     v_body = np.array([z[3], z[4], 0.0])
#     #position
#     H[0:2, 0:2] = np.eye(2)
#     # velocity wrt velocity state
#     H[3:5, 3:5] = np.eye(2)
#     # velocity wrt attitude error  <-- IMPORTANT
#     H[3:5, 6:9] =  -(R @ skew(v_body))[0:2, :]
#     # yaw wrt attitude
#     H[2, 6:9] = np.array([0, 0, 1])
#     return H

def H_matrix(x_nom,z):
    H = np.zeros((5, 15))
    R = x_nom["R"]
    v = x_nom["v"]
    # position
    H[0:2, 0:2] = np.eye(2)
    # yaw
    H[2, 6:9] = np.array([0, 0, 1])
    # velocity wrt velocity (nav -> body projection)
    H[3:5, 3:6] = R.T[0:2, :]
    # velocity wrt attitude error
    H[3:5, 6:9] = -(R.T @ skew(v))[0:2, :]
    return H

def eskf_cov_predict(P, R, acc, omega, dt, Q):
    F = build_F(dt, R, acc, omega)
    P = F @ P @ F.T + Q
    return P

def eskf_update(x_nom, P, z, R_meas):
    H = H_matrix(x_nom, z)
    z_pred = h(x_nom)
    # innovation
    y = z - z_pred
    y[2] = wrap_angle(y[2])

    # Kalman gain
    S = H @ P @ H.T + R_meas
    K = P @ H.T @ np.linalg.inv(S)
    # error-state update
    dx = K @ y
    # --- INJECTION (MOST IMPORTANT PART) ---
    x_nom["p"] += dx[0:3]
    x_nom["v"] += dx[3:6]
    dtheta = dx[6:9]
    x_nom["R"] = x_nom["R"] @ expm(skew(dtheta))
    x_nom["b_g"] += dx[9:12]
    x_nom["b_a"] += dx[12:15]
    # --- Covariance update ---
    I = np.eye(15)
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R_meas @ K.T
    return x_nom, P


def eskf_predict_step(x_nom, P, omega_m, acc_m, dt, Q):
    # IMU correction
    omega = omega_m - x_nom["b_g"]
    acc = acc_m - x_nom["b_a"]
    # 1) nominal propagation (YOU already have this)
    x_nom = eskf_predict(x_nom, omega, acc, dt)
    # 2) covariance propagation
    P = eskf_cov_predict(P, x_nom["R"], acc, omega, dt, Q)
    return x_nom, P


from scipy.spatial.transform import Rotation as R_V

def initial_state(data,iter):
    #Position
    px = data['ins_pos_rel_x'].iloc[iter_start]
    py = data['ins_pos_rel_y'].iloc[iter_start]
    p = np.array([px, py, 0.0])

    #Ins yaw and pitch with negative sign works
    roll =  data['ins_roll'].iloc[iter_start]
    pitch = -1*data['ins_pitch'].iloc[iter_start]
    yaw = -1*data['ins_yaw'].iloc[iter_start]


    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)


    R_w_ned= R_V.from_euler('ZYX', [yaw, pitch, roll], degrees=False).as_matrix()

    R_init =R_w_ned

    c = np.cos(yaw)
    s = np.sin(yaw)

    R_vel = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    vx_body = data['ins_vel_hor_x'].iloc[iter_start]
    vy_body = -1*data['ins_vel_hor_y'].iloc[iter_start]
    #Negative sign for vy works checked with a small outage 6 secs with lateral manuever
    v_body = np.array([vx_body, vy_body, 0.0])
    v = R_vel @ v_body

    x_init = {
        "p": p,
        "v": v,
        "R": R_init,
        "b_g": np.zeros(3),
        "b_a": np.zeros(3)
    }
    return x_init




R_meas = np.diag([
    1e12, # x position e10 high noise
    1e12, # y position
    1e11, # yaw
    1e11, # vx e8
    1e11 # vy
]) #13 for 5 mins, 15 for 10 mins, 12 for 2 mins
#Plots no correctio 5 mins #position e13 and vel e25
#Transformer all e11

# From one iter point calculate all and check code.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define outage start points
outage_duration = int(50 * 60 * 5)  # 5 mins at 50Hz
#iter_starts = [25000,50000,75000,100000,150000]  # define your start points here
iter_starts = [25000]
all_max_drifts = []
all_final_drifts = []

for iter_start in iter_starts:
    # Re-initialize state for each outage
    x_nom = initial_state(combined_df, iter_start)
    x_nom_init = x_nom.copy()  # make sure this is your fresh initial state
    P_init = P.copy()

    trajectory = [x_nom_init.copy()]
    x_nom_cur = x_nom_init.copy()
    P_cur = P_init.copy()
    delta_t = 0.02
    for i in range(iter_start + 1, iter_start + outage_duration + 1):
        omega, acc = imu_input(combined_df, i)
        x_nom_cur, P_cur = eskf_predict_step(x_nom_cur, P_cur, omega, acc, delta_t, Q)
        z = measurement_vector(combined_df, i)
        x_nom_cur, P_cur = eskf_update(x_nom_cur, P_cur, z, R_meas)
        trajectory.append(x_nom_cur.copy())

    # Build trajectory df
    rows = []
    for state in trajectory:
        p = state["p"]
        v = state["v"]
        R = state["R"]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        rows.append({"xs": p[0], "ys": p[1], "vx": v[0], "vy": v[1], "yaw": yaw})
    trajectory_df = pd.DataFrame(rows)

    measured_x = combined_df['ins_pos_rel_x'].iloc[iter_start:iter_start + outage_duration + 1].values
    measured_y = combined_df['ins_pos_rel_y'].iloc[iter_start:iter_start + outage_duration + 1].values
    estimated_x = trajectory_df['xs'].values
    estimated_y = trajectory_df['ys'].values

    traj_data = pd.DataFrame({
        'measured_x': measured_x,
        'measured_y': measured_y,
        'estimated_x': estimated_x,
        'estimated_y': estimated_y,
    })

    drift_per_step = np.sqrt((estimated_x - measured_x)**2 + (estimated_y - measured_y)**2)
    final_drift = np.sqrt((estimated_x[-1] - measured_x[-1])**2 + (estimated_y[-1] - measured_y[-1])**2)
    max_drift = np.max(drift_per_step)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(measured_y, measured_x, 'k-', linewidth=1.5, label='Ground Truth (INS)')
    ax.plot(estimated_y, estimated_x, 'b--', linewidth=1.5, label='ES-EKF Estimated')

    # Mark start and end
    ax.plot(measured_y[0], measured_x[0], 'go', markersize=8, label='Start')
    ax.plot(measured_y[-1], measured_x[-1], 'rs', markersize=8, label='End')

    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'Trajectory — Iter {iter_start} | Max drift: {max_drift:.2f} m')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'trajectory_iter_{iter_start}.png', dpi=150)
    plt.show()

    traj_data.to_csv(f'trajectory_obd.csv', index=False)
    all_max_drifts.append(max_drift)
    all_final_drifts.append(final_drift)

    print(f"Iter {iter_start} | Final drift: {final_drift:.2f} m | Max drift: {max_drift:.2f} m")

# Summary statistics
all_max_drifts = np.array(all_max_drifts)
all_final_drifts = np.array(all_final_drifts)

print(f"\n--- Summary over {len(iter_starts)} outages ---")
print(f"Max drift   | Mean: {np.mean(all_max_drifts):.2f} m | Std: {np.std(all_max_drifts):.2f} m")
print(f"Final drift | Mean: {np.mean(all_final_drifts):.2f} m | Std: {np.std(all_final_drifts):.2f} m")

# Plot drift per outage
plt.figure(figsize=(10, 5))
x = np.arange(len(iter_starts))
plt.bar(x, all_max_drifts, yerr=all_max_drifts.std(), capsize=5, color='orange', label='Max drift per outage')
plt.xticks(x, [str(s) for s in iter_starts], rotation=45)
plt.xlabel('Outage start index')
plt.ylabel('Max position drift (m)')
plt.title('Max drift across multiple outages')
plt.legend()
plt.tight_layout()
plt.show()