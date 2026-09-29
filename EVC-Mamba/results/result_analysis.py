import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import tikzplotlib

# pred = np.load('final-ml_paper/pred_fine_tune.npy')
# true = np.load('final-ml_paper/true_fine_tune.npy')
# obd_data = np.load('final-ml_paper/gt_poses_pretrain.npy')


#Comparision model codes
pred = np.load('evidential-mamba-results/pred.npy')
true = np.load('evidential-mamba-results/true.npy')
obd_data = np.load('evidential-mamba-results/gt_poses.npy')
var = np.load('evidential-mamba-results/variance.npy')

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 300)

y1 = np.exp(-x**2 / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))
y2 = np.exp(-x**2 / (2 * 1.5**2)) / (1.5 * np.sqrt(2 * np.pi))

# ---- Concentrated (low uncertainty) — dark blue #004C99 ----
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x, y1, linewidth=2, color='#004C99')
ax.fill_between(x, y1, alpha=0.4, color='#004C99')
ax.axis('off')
plt.tight_layout()
plt.savefig('bell_concentrated.svg', format='svg', bbox_inches='tight', transparent=True)
plt.show()

# ---- Spread out (high uncertainty) — light blue #CCE5FF ----
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x, y2, linewidth=2, color='#CCE5FF')
ax.fill_between(x, y2, alpha=0.4, color='#CCE5FF')
ax.axis('off')
plt.tight_layout()
plt.savefig('bell_spread.svg', format='svg', bbox_inches='tight', transparent=True)
plt.show()

gt_pose_columns = ['gt_x', 'gt_y', 'gt_yaw', 'gt_yawrate', 'gt_vx', 'gt_vy', 'gt_ax', 'gt_ay', "obd_yawrate","LatAcc_obd", "brake_pressure_obd", "speedo_obd", "SW_pos_obd","VelFR_obd","VelFL_obd","VelRR_obd","VelRL_obd","ins_pitch","ins_roll"]

########################### Variance plots and correltation ##############################

abs_err_vx = np.abs(pred[:, 0] - true[:, 0])
abs_err_vy = np.abs(pred[:, 1] - true[:, 1])

var_vx = var[:, 0]
var_vy = var[:, 1]

n = len(abs_err_vx)

for name, err, v in [('Vx', abs_err_vx, var_vx), ('Vy', abs_err_vy, var_vy*.1)]:
    r, p = stats.pearsonr(err, v)
    spearman_r, p = stats.spearmanr(err, v)
    # t-statistic
    t_star = r * np.sqrt((n - 2) / (1 - r ** 2))

    # t critical value at 99% confidence (two-tailed)
    t_crit = stats.t.ppf(0.995, df=n - 2)

    reject = abs(t_star) > t_crit

    print(f"\n{name}")
    print(f"  Pearson r  : {r:.4f}")
    print(f"  t*         : {t_star:.1f}")
    print(f"  t_crit(99%): {t_crit:.3f}")
    print(f"  Spearman ρ : {spearman_r:.4f}")
    print(f"  Null hypothesis rejected at 99% confidence: {reject}")

std_vx = np.sqrt(var[:, 0])
std_vy = np.sqrt(var[:, 1])*0.26

confidence_levels = np.linspace(0.01, 0.99, 50)

def compute_coverage(pred_mean, pred_std, true_vals, confidence_levels):
    coverages = []
    for conf in confidence_levels:
        # z-score for this confidence level (two-tailed)
        z = stats.norm.ppf((1 + conf) / 2)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        inside = np.mean((true_vals >= lower) & (true_vals <= upper))
        coverages.append(inside)
    return np.array(coverages)

from scipy import stats

coverage_vx = compute_coverage(pred[:, 0], std_vx, true[:, 0], confidence_levels)
coverage_vy = compute_coverage(pred[:, 1], std_vy, true[:, 1], confidence_levels)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, coverage, name in zip(axes, [coverage_vx, coverage_vy], ['Vx', 'Vy']):
    ax.plot(confidence_levels, confidence_levels, 'k--', label='Perfect calibration')
    ax.plot(confidence_levels, coverage, label='EviMamba-INS')
    ax.fill_between(confidence_levels, confidence_levels, coverage,
                    alpha=0.2, label='Calibration gap')
    ax.set_xlabel('Expected confidence level')
    ax.set_ylabel('Empirical coverage')
    ax.set_title(f'Calibration Plot — {name}')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
np.savetxt('coverage_vx.csv', np.column_stack([confidence_levels, coverage_vx]),
           delimiter=',', fmt='%.4f')
np.savetxt('coverage_vy.csv', np.column_stack([confidence_levels, coverage_vy]),
           delimiter=',', fmt='%.4f')

















###########################Baseline from OBD data##########################################
obd_data = pd.DataFrame(obd_data, columns=gt_pose_columns)
obd_data['vx'] = ((obd_data['VelFR_obd'] + obd_data['VelFL_obd'] +
                  obd_data['VelRR_obd'] + obd_data['VelRL_obd']) / 4)*5/18

obd_data['vy'] = ((obd_data['VelFR_obd'] - obd_data['VelFL_obd'] +
                  obd_data['VelRR_obd'] - obd_data['VelRL_obd']) / 4)*-5/18

obd_vx = obd_data['vx'].to_numpy()
obd_vy = obd_data['vy'].to_numpy()
obd_vel = np.stack([obd_vx, obd_vy], axis=1)

abs_error = np.abs(obd_vel - true)
mae_per_column = np.mean(abs_error, axis=0)
rmse_per_column = np.sqrt(np.mean((obd_vel - true)**2, axis=0))
max_ae_per_column = np.max(abs_error, axis=0)

# Print results with labels
column_names = ['Vx', 'Vy']
print("Error Metrics (MAE and Max AE):\n")
for name, mae, rmse, maxae in zip(column_names, mae_per_column,rmse_per_column, max_ae_per_column):
    print(f"{name:>8} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | Max AE: {maxae:.6f}")

column_names = ['Vx', 'Vy']

print("Column-wise statistics for 'pred':\n")
for i, name in enumerate(column_names):
    col = pred[:, i]
    print(f"{name:>8}:")
    print(f"  Mean     : {np.mean(col):.6f}")
    print(f"  Std Dev  : {np.std(col):.6f}")
    print(f"  Min      : {np.min(col):.6f}")
    print(f"  Max      : {np.max(col):.6f}")
    print(f"  Median   : {np.median(col):.6f}")
    print(f"  5th pct  : {np.percentile(col, 5):.6f}")
    print(f"  95th pct : {np.percentile(col, 95):.6f}")
    print(f" ")