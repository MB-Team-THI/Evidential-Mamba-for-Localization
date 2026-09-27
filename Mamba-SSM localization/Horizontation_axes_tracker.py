import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

folder_path = "data/rev-sted"
#Rev-sted 1 has GPS data outages do not consider it.
csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(3,4)] #Loading only one file
df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True )

def rot_matrix_to_quat(M):
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    return np.array([x, y, z, w])

def quat_to_rot_matrix(q):
    x, y, z, w = q
    xx = x*x
    yy = y*y
    zz = z*z
    ww = w*w

    xy = x*y
    xz = x*z
    xw = x*w
    yz = y*z
    yw = y*w
    zw = z*w

    rot = np.array([
        [ww + xx - yy - zz, 2*(xy - zw),       2*(xz + yw)],
        [2*(xy + zw),       ww - xx + yy - zz, 2*(yz - xw)],
        [2*(xz - yw),       2*(yz + xw),       ww - xx - yy + zz]
    ])

    return rot

def quat_normalize(q):
    return q / np.linalg.norm(q)

def orient_quaternion(q, ref_q):
    # Flip q if it points opposite of ref_q
    if np.dot(q, ref_q) < 0:
        return -q
    return q

#Now we want to convert body frame (ADMA) to vehicle's horizontal frame.
#This is achieved by rotating from ADMA to local tangent plane (fixed) and then again to vehicles horizontal frame using yaw angle
#Finally the ax, ay will be obtained in vehicle frame which will be used for state equations.

#First find Horizontation at standstill and go to local tangent plane

rotations = []
for i in range(0,100): #Standstill time of the car
        theta_roll = np.arcsin(combined_df["acc_body_hr_y"].iloc[i])
        theta_pitch = np.arcsin(combined_df["acc_body_hr_x"].iloc[i])
        theta_h = np.arccos(np.clip(combined_df["acc_body_hr_z"].iloc[i],-1,1)) #acceleration cannot be greater than 1
        r_ho = np.cross(np.array([combined_df["acc_body_hr_x"].iloc[i], combined_df["acc_body_hr_y"].iloc[i], combined_df["acc_body_hr_z"].iloc[i]]),np.array([0,0,1]))

        rx, ry, rz = r_ho #r̂ components
        rm = rx**2 + ry**2 + rz**2
        root_rm = np.sqrt(rm)
        c_theta = np.cos(theta_h)
        s_theta = np.sin(theta_h)

        #Explicit Rodrigues Rotation Matrix
        ip_R_ip1 = 1/rm * np.array([
        [(rx**2+(ry**2+rz**2)*c_theta), rx*ry*(1-c_theta) - rz*root_rm*s_theta,  rx*rz*(1-c_theta) + ry*root_rm*s_theta],
        [rx*ry*(1-c_theta)+rz*root_rm*s_theta, ry**2+(rx**2+rz**2)*c_theta,  ry*rz*(1-c_theta)-rx*root_rm*s_theta],
        [rx*rz*(1-c_theta)-ry*root_rm*s_theta, ry*rz*(1-c_theta)+rx*root_rm*s_theta, rz**2+(rx**2+ry**2)*c_theta]])

        if root_rm <= 1e-6:
                ip_R_ip1 = np.eye(3) #Identity matrix and no rotations are required

        theta_yaw = np.deg2rad(combined_df["ins_yaw"].iloc[i])
        R_yaw = np.array([
                [np.cos(theta_yaw), -np.sin(theta_yaw), 0 ],
                [np.sin(theta_yaw), np.cos(theta_yaw), 0 ],
                [0 , 0, 1]])
        tp_R_ip = R_yaw @ ip_R_ip1
        #Convert to quaternion
        quaternion = rot_matrix_to_quat(tp_R_ip)
        rotations.append(quaternion)

ref_q = rotations[0]
aligned_quats = [ref_q]

for q in rotations[1:]:
    q_aligned = orient_quaternion(q, ref_q)
    aligned_quats.append(q_aligned)

q_avg = np.mean(aligned_quats, axis=0)
q_avg = quat_normalize(q_avg)
tp_R_ip = quat_to_rot_matrix(q_avg)

print('standstill over')
def orthonormalize_rotation_matrix(rotation_matrix):
        r3 = rotation_matrix[:, 2]
        r2 = rotation_matrix[:, 1]
        r1_temp = np.cross(r2, r3)
        r2_temp = np.cross(r3, r1_temp)

        # Normalize all columns
        r1 = r1_temp / np.linalg.norm(r1_temp)
        r2 = r2_temp / np.linalg.norm(r2_temp)
        r3 = r3 / np.linalg.norm(r3)

        # Stack them as columns into an orthonormal matrix
        R_orthonormal = np.column_stack((r1, r2, r3))

        return R_orthonormal

delta_t = 0.02
tp_R_ip_t1 = tp_R_ip #Initialization

computed_a_hor_x_list = []
computed_a_hor_y_list = []
computed_yaw = [np.deg2rad(combined_df["ins_yaw"].iloc[0])]
computed_yaw_rate = []
deg2rad = np.deg2rad(1)


for i in range(0,30000):
        # skew_theta_dot = np.array([
        # [0, -deg2rad *combined_df['rate_body_hr_z'].iloc[i], deg2rad *combined_df['rate_body_hr_y'].iloc[i]],
        # [deg2rad *combined_df['rate_body_hr_z'].iloc[i], 0, -deg2rad *combined_df['rate_body_hr_x'].iloc[i] ],
        # [-deg2rad *combined_df['rate_body_hr_y'].iloc[i] , deg2rad *combined_df['rate_body_hr_x'].iloc[i] , 0]])
        #
        # tp_R_ip_t2 = (delta_t * skew_theta_dot + np.identity(3)) @ tp_R_ip_t1
        # tp_R_ip_t2 = orthonormalize_rotation_matrix(tp_R_ip_t2)

        #Better derivative for rotation matrix
        gyro_deg = np.array([
            combined_df["rate_body_hr_x"].iloc[i],
            combined_df["rate_body_hr_y"].iloc[i],
            combined_df["rate_body_hr_z"].iloc[i]
        ])
        gyro_corrected = deg2rad * (gyro_deg)

        omega = gyro_corrected
        theta = np.linalg.norm(omega * delta_t)

        if theta > 1e-8:
            k = omega / np.linalg.norm(omega)
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R_delta = (np.eye(3)
                    + np.sin(theta) * K
                    + (1 - np.cos(theta)) * (K @ K))
        else:
            R_delta = np.eye(3)

        tp_R_ip_t2 = tp_R_ip_t1 @ R_delta
        tp_R_ip_t2 = orthonormalize_rotation_matrix(tp_R_ip_t2)

        a_tp_t2 = tp_R_ip_t2 @ np.array([combined_df["acc_body_hr_x"].iloc[i], combined_df["acc_body_hr_y"].iloc[i], combined_df["acc_body_hr_z"].iloc[i]])
        rotrate_tp_t2 = tp_R_ip_t2 @ np.array([deg2rad *combined_df["rate_body_hr_x"].iloc[i], deg2rad *combined_df["rate_body_hr_y"].iloc[i], deg2rad *combined_df["rate_body_hr_z"].iloc[i]])

        new_yaw = computed_yaw[-1] + rotrate_tp_t2[2] * delta_t  # Integrate yaw rate
        computed_yaw.append(new_yaw)
        computed_yaw_rate.append(rotrate_tp_t2[2])
        a_OG = np.sqrt(a_tp_t2[0]**2+a_tp_t2[1]**2)
        theta_a_OG = np.arctan2(a_tp_t2[1],a_tp_t2[0])
        theta_r_OG = np.arctan2(rotrate_tp_t2[1],rotrate_tp_t2[0])

        a_hor_x = a_OG*np.cos(theta_a_OG - new_yaw) #Can use GT yaw as well
        a_hor_y = a_OG*np.sin(theta_a_OG - new_yaw)

        computed_a_hor_x_list.append(a_hor_x)
        computed_a_hor_y_list.append(a_hor_y)
        tp_R_ip_t1 = tp_R_ip_t2

plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)

plt.plot(computed_a_hor_x_list, label="Computed_acc_hor_x")
plt.plot(combined_df["acc_body_hr_x"].iloc[0:30000].values, label="Reference acc_body_x")
plt.plot(combined_df["acc_hor_x"].iloc[0:30000].values, label="Reference acc_hor_x")
plt.legend()
plt.title("Horizontal Acceleration X")

plt.subplot(2, 2, 2)
plt.plot(combined_df["acc_hor_y"].iloc[0:30000].values, label="Reference acc_hor_y")
plt.plot(computed_a_hor_y_list, label="Computed_acc_hor_y")
plt.plot(combined_df["acc_body_hr_y"].iloc[0:30000].values, label="Reference acc_body_y")
plt.legend()
plt.title("Horizontal Acceleration Y")
computed_yaw_deg = np.rad2deg(computed_yaw) % 360
ref_yaw_deg = combined_df["ins_yaw"].iloc[:len(computed_yaw_deg)].values
yaw_diff = (ref_yaw_deg - computed_yaw_deg + 180) % 360 - 180
plt.subplot(2, 2, 3)
plt.plot(combined_df["ins_yaw"].iloc[0:30000].values, label="Reference Yaw (deg)")
plt.plot(computed_yaw_deg, label="Computed Yaw (deg)")
plt.plot(yaw_diff , label="Yaw diff(deg)")
plt.legend()
plt.title("Yaw Comparison (Degrees)")
plt.subplot(2, 2, 4)
plt.plot(combined_df["rate_hor_z"].iloc[0:30000].values, label="Rate Hor Z(deg/s)")
plt.plot(combined_df["rate_body_hr_z"].iloc[0:30000].values, label="Rate Body Hr Z(deg/s)")
plt.plot(np.rad2deg(computed_yaw_rate) , label="Computed Yaw rate(deg/s)")
plt.legend()
plt.title("Yaw rate Comparison (Degrees/s)")
plt.tight_layout()
plt.show()
print('Max heading angle',np.max(yaw_diff))

#Fast checking
# new_data = {
#     "computed_a_hor_x": computed_a_hor_x_list[:30000],
#     "computed_a_hor_y": computed_a_hor_y_list[:30000],
#     "computed_yaw_angle": computed_yaw_deg[:30000]
# }
#
# # Create the DataFrame
# computed_df = pd.DataFrame(new_data)
#
# # Save to CSV
# computed_df.to_csv("computed_inertial_nav_data.csv", index=False)