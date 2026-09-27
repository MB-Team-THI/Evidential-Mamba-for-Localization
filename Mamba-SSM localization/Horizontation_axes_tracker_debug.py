import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R

folder_path = "data/rev-sted"
#Rev-sted 1 has GPS data outages do not consider it.
csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(3,4)] #Loading only one file
df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

#Now we want to convert body frame (ADMA) to vehicle's horizontal frame.
#This is achieved by rotating from ADMA to local tangent plane (fixed) and then again to vehicles horizontal frame using yaw angle
#Finally the ax, ay will be obtained in vehicle frame which will be used for state equations.

#First find Horizontation at standstill and go to local tangent plane

# rotations = []
# for i in range(0,2): #Standstill time of the car
#         theta_roll = np.arcsin(combined_df["acc_body_hr_y"].iloc[i])
#         theta_pitch = np.arcsin(combined_df["acc_body_hr_x"].iloc[i])
#         theta_h = np.arccos(np.clip(combined_df["acc_body_hr_z"].iloc[i],-1,1)) #acceleration cannot be greater than 1
#         r_ho = np.cross(np.array([combined_df["acc_body_hr_x"].iloc[i], combined_df["acc_body_hr_y"].iloc[i], combined_df["acc_body_hr_z"].iloc[i]]),np.array([0,0,1]))
#
#         rx, ry, rz = r_ho #r̂ components
#         rm = rx**2 + ry**2 + rz**2
#         root_rm = np.sqrt(rm)
#         c_theta = np.cos(theta_h)
#         s_theta = np.sin(theta_h)
#
#         #Explicit Rodrigues Rotation Matrix
#         ip_R_ip1 = 1/rm * np.array([
#         [(rx**2+(ry**2+rz**2)*c_theta), rx*ry*(1-c_theta) - rz*root_rm*s_theta,  rx*rz*(1-c_theta) + ry*root_rm*s_theta],
#         [rx*ry*(1-c_theta)+rz*root_rm*s_theta, ry**2+(rx**2+rz**2)*c_theta,  ry*rz*(1-c_theta)-rx*root_rm*s_theta],
#         [rx*rz*(1-c_theta)-ry*root_rm*s_theta, ry*rz*(1-c_theta)+rx*root_rm*s_theta, rz**2+(rx**2+ry**2)*c_theta]])
#
#         if root_rm <= 1e-6:
#                 ip_R_ip1 = np.eye(3) #Identity matrix and no rotations are required
#
#         theta_yaw = np.deg2rad(combined_df["ins_yaw"].iloc[i])
#         R_yaw = np.array([
#                 [np.cos(theta_yaw), -np.sin(theta_yaw), 0 ],
#                 [np.sin(theta_yaw), np.cos(theta_yaw), 0 ],
#                 [0 , 0, 1]])
#         tp_R_ip = R_yaw @ ip_R_ip1
#         #Convert to quaternion
#         quaternion = rot_matrix_to_quat(tp_R_ip)
#         rotations.append(quaternion)
#
# ref_q = rotations[0]
# aligned_quats = [ref_q]
#
# for q in rotations[1:]:
#     q_aligned = orient_quaternion(q, ref_q)
#     aligned_quats.append(q_aligned)
#
# q_avg = np.mean(aligned_quats, axis=0)
# q_avg = quat_normalize(q_avg)
# tp_R_ip = quat_to_rot_matrix(q_avg)
# print('standstill over')

roll = combined_df["ins_roll"].iloc[0]
pitch = combined_df["ins_pitch"].iloc[0]
yaw = combined_df["ins_yaw"].iloc[0]
global_R_body  = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
tp_R_ip = global_R_body

print('hi')


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
q_tp_ip_t1 = R.from_matrix(tp_R_ip).as_quat() #Initialization
computed_a_hor_x_list = []
computed_a_hor_y_list = []
computed_yaw = [np.deg2rad(combined_df["ins_yaw"].iloc[0])]
computed_yaw_rate = []
deg2rad = np.pi / 180
rad2deg = 180 / np.pi

frobenius_norms = []

for i in range(0,30000):
        # skew_theta_dot = np.array([
        # [0, -deg2rad *combined_df['rate_body_hr_z'].iloc[i], deg2rad *combined_df['rate_body_hr_y'].iloc[i]],
        # [deg2rad *combined_df['rate_body_hr_z'].iloc[i], 0, -deg2rad *combined_df['rate_body_hr_x'].iloc[i] ],
        # [-deg2rad *combined_df['rate_body_hr_y'].iloc[i] , deg2rad *combined_df['rate_body_hr_x'].iloc[i] , 0]])
        #
        # tp_R_ip_t2 = (delta_t * skew_theta_dot + np.identity(3)) @ tp_R_ip_t1
        # tp_R_ip_t2 = orthonormalize_rotation_matrix(tp_R_ip_t2)

        #Better derivative for rotation matrix
        if i>1:
            w1 = rotrate_tp_t2[2]
        gyro_deg = np.array([
            combined_df["rate_body_hr_x"].iloc[i],
            combined_df["rate_body_hr_y"].iloc[i],
            combined_df["rate_body_hr_z"].iloc[i]
        ])
        gyro_corrected = deg2rad * (gyro_deg)
        omega = gyro_corrected
        theta = np.linalg.norm(omega * delta_t)

        omega_mag = np.linalg.norm(omega)

        if omega_mag > 1e-8:
            axis = omega / omega_mag
            delta_theta = omega_mag * delta_t
            delta_q = R.from_rotvec(axis * delta_theta).as_quat()
        else:
            delta_q = np.array([0, 0, 0, 1])  # No rotation

        q_tp_ip_t2 = (R.from_quat(q_tp_ip_t1) * R.from_quat(delta_q)).as_quat()
        q_tp_ip_t2 = q_tp_ip_t2 / np.linalg.norm(q_tp_ip_t2)
        tp_R_ip_t2 = R.from_quat(q_tp_ip_t2).as_matrix()

        # if theta > 1e-8:
        #     k = omega / np.linalg.norm(omega)
        #     K = np.array([
        #         [0, -k[2], k[1]],
        #         [k[2], 0, -k[0]],
        #         [-k[1], k[0], 0]
        #     ])
        #     R_delta = (np.eye(3)
        #             + np.sin(theta) * K
        #             + (1 - np.cos(theta)) * (K @ K))
        # else:
        #     R_delta = np.eye(3)

        #tp_R_ip_t2 = tp_R_ip_t1 @ R_delta
        #tp_R_ip_t2 = orthonormalize_rotation_matrix(tp_R_ip_t2)
        #a_tp_t2 = R.from_euler('xyz', [combined_df["ins_roll"].iloc[i], combined_df["ins_pitch"].iloc[i], combined_df["ins_yaw"].iloc[i]], degrees=True).as_matrix()@ np.array([combined_df["acc_body_hr_x"].iloc[i], combined_df["acc_body_hr_y"].iloc[i], combined_df["acc_body_hr_z"].iloc[i]])


        GT_tp_R_ip_t2 = R.from_euler('xyz', [combined_df["ins_roll"].iloc[i], combined_df["ins_pitch"].iloc[i],
                             combined_df["ins_yaw"].iloc[i]], degrees=True).as_matrix()
        diff_matrix = GT_tp_R_ip_t2 - tp_R_ip_t2
        frobenius_norm = np.linalg.norm(diff_matrix, ord='fro')
        frobenius_norms.append(frobenius_norm)
        #print("Frobenius norm of difference:", i, frobenius_norm)
        a_tp_t2 = tp_R_ip_t2 @ np.array([combined_df["acc_body_hr_x"].iloc[i], combined_df["acc_body_hr_y"].iloc[i], combined_df["acc_body_hr_z"].iloc[i]])
        rotrate_tp_t2 = tp_R_ip_t2 @ np.array([deg2rad *combined_df["rate_body_hr_x"].iloc[i], deg2rad *combined_df["rate_body_hr_y"].iloc[i], deg2rad *combined_df["rate_body_hr_z"].iloc[i]])
        w4 = rotrate_tp_t2[2]
        if i<2:
                new_yaw = computed_yaw[-1] + rotrate_tp_t2[2] * delta_t# Integrate yaw rate
        else:
                w2 = 0.5 * (w1 + w4)
                w3 = w2
                yaw_delta = (delta_t / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
                new_yaw = computed_yaw[-1] + yaw_delta
        new_yaw = computed_yaw[-1] + deg2rad*combined_df["rate_hor_z"].iloc[i]*delta_t
        new_yaw = new_yaw  % (2 * np.pi)
        # print(new_yaw)
        computed_yaw.append(new_yaw)
        computed_yaw_rate.append(rotrate_tp_t2[2])

        a_OG = np.sqrt(a_tp_t2[0]**2+a_tp_t2[1]**2)
        theta_a_OG = np.arctan2(a_tp_t2[1],a_tp_t2[0])
        theta_r_OG = np.arctan2(rotrate_tp_t2[1],rotrate_tp_t2[0])

        #a_hor_x = a_OG*np.cos(theta_a_OG - np.deg2rad(combined_df["ins_yaw"].iloc[i])) #Can use GT yaw as well
        #a_hor_y = a_OG*np.sin(theta_a_OG - np.deg2rad(combined_df["ins_yaw"].iloc[i]))
        a_hor_x = a_OG*np.cos(theta_a_OG - new_yaw)  # Can use GT yaw as well
        a_hor_y = a_OG*np.sin(theta_a_OG - new_yaw)

        computed_a_hor_x_list.append(a_hor_x)
        computed_a_hor_y_list.append(a_hor_y)
        tp_R_ip_t1 = tp_R_ip_t2
        q_tp_ip_t1 = R.from_matrix(tp_R_ip_t1).as_quat()

plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
acc_error_x = np.array(computed_a_hor_x_list) - combined_df["acc_hor_x"].iloc[:len(computed_a_hor_x_list)].values
plt.plot(acc_error_x*9.81, label="Error: Computed - Reference acc_hor_x")
plt.legend()
plt.title("Horizontal Acceleration error X in m/s2")

plt.subplot(2, 2, 2)
acc_error_y = np.array(computed_a_hor_y_list) - combined_df["acc_hor_y"].iloc[:len(computed_a_hor_y_list)].values
plt.plot(acc_error_y*9.81, label="Error: Computed - Reference acc_hor_y")
plt.legend()
plt.title("Horizontal Acceleration Y")


computed_yaw_deg = np.rad2deg(computed_yaw) % 360
ref_yaw_deg = combined_df["ins_yaw"].iloc[:len(computed_yaw_deg)].values
yaw_diff = (ref_yaw_deg - computed_yaw_deg + 180) % 360 - 180
plt.subplot(2, 2, 3)
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
print('Max heading angle ES EKF',np.max(np.abs(yaw_diff)))

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
# Plot the Frobenius norm over time (or iterations)
plt.figure(figsize=(10, 6))
plt.plot(frobenius_norms, label='Frobenius Norm of Rotation Matrix Difference', color='green')
plt.xlabel('Time Step')
plt.ylabel('Frobenius Norm')
plt.title('Rotation Matrix Deviation Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()