from scipy.spatial.transform import Rotation as R
import numpy as np

R1 = R.from_euler('z', 30, degrees=True)
R2 = R.from_euler('z', 60, degrees=True)

rotations = [R1, R2]

mean_rot = R.mean(rotations)
print(mean_rot.as_euler('zyx', degrees=True))