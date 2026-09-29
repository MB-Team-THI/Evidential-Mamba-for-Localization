
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import EKFConfig
from so3 import exp_so3, orthonormalize, skew

# error-state slices
P_, V_, TH, BA, BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
N_ERR = 15


@dataclass
class NominalState:
    p: np.ndarray
    v: np.ndarray
    R: np.ndarray
    ba: np.ndarray
    bg: np.ndarray

    def copy(self):
        return NominalState(self.p.copy(), self.v.copy(), self.R.copy(), self.ba.copy(), self.bg.copy())


class ESEKF:
    def __init__(self, cfg: EKFConfig, p0, v0, R0, ba0=None, bg0=None, P0: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.g = np.asarray(cfg.gravity, dtype=float)
        self.x = NominalState(np.asarray(p0, float).copy(), np.asarray(v0, float).copy(),
                              np.asarray(R0, float).copy(),
                              np.zeros(3) if ba0 is None else np.asarray(ba0, float).copy(),
                              np.zeros(3) if bg0 is None else np.asarray(bg0, float).copy())
        if P0 is None:
            s = np.repeat(np.asarray(cfg.init_std, float), 3)
            P0 = np.diag(s ** 2)
        self.P = P0.copy()
        self._n_prop = 0
        self.last_nis = np.nan


    def predict(self, acc: np.ndarray, gyr: np.ndarray, dt: float):
        x, c = self.x, self.cfg
        a_c = acc - x.ba
        w_c = gyr - x.bg
        R = x.R
        a_g = R @ a_c + self.g                          # acceleration in global frame

        # --- nominal kinematics (Eq. 10) ---
        x.p = x.p + x.v * dt + 0.5 * a_g * dt ** 2
        x.v = x.v + a_g * dt
        x.R = R @ exp_so3(w_c * dt)
        self._n_prop += 1
        if c.reorthonormalize_every and self._n_prop % c.reorthonormalize_every == 0:
            x.R = orthonormalize(x.R)

        # --- linearised error-state transition F_t ---
        F = np.eye(N_ERR)
        F[P_, V_] = np.eye(3) * dt
        F[V_, TH] = -R @ skew(a_c) * dt
        F[V_, BA] = -R * dt
        F[TH, TH] = exp_so3(w_c * dt).T
        F[TH, BG] = -np.eye(3) * dt

        # --- process noise Q_t = G Q_w G^T (continuous densities -> discrete) ---
        # G maps [n_a, n_g, n_ba, n_bg] into [dv, dtheta, db_a, db_g]; with
        # isotropic noise G Q_w G^T is block diagonal:
        Q = np.zeros((N_ERR, N_ERR))
        Q[V_, V_] = np.eye(3) * c.sigma_a ** 2 * dt
        Q[TH, TH] = np.eye(3) * c.sigma_g ** 2 * dt
        Q[BA, BA] = np.eye(3) * c.sigma_ba ** 2 * dt
        Q[BG, BG] = np.eye(3) * c.sigma_bg ** 2 * dt

        self.P = F @ self.P @ F.T + Q                   # Eq. (12)
        self.P = 0.5 * (self.P + self.P.T)

    def update_velocity(self, v_xy: np.ndarray, var_xy: np.ndarray) -> bool:
        """EVC-Mamba virtual velocity measurement (Eqs. 13-17).

        v_xy   : [v_x, v_y] vehicle-frame velocity estimate  (v~_t = gamma)
        var_xy : evidential total uncertainty delta_v (variance, (m/s)^2) per axis
        """
        c, x = self.cfg, self.x
        z = np.array([v_xy[0], v_xy[1], 0.0])                                  # Eq. (13)
        var = np.maximum(np.asarray(var_xy, float) * c.vel_var_scale, c.min_vel_var)
        M = np.diag([var[0], var[1], c.sigma_z2])

        Rvg = x.R.T                                                           # global -> vehicle
        v_b = Rvg @ x.v
        y = z - v_b                                                           # Eq. (14)
        H = np.zeros((3, N_ERR))                                              # Eq. (15)
        H[:, V_] = Rvg
        # local perturbation: h = exp(-[dth]x) R^T v ~= R^T v + [R^T v]_x dth
        # (equals the paper's -R^v_g[v]_x term under the opposite sign convention for dtheta)
        H[:, TH] = skew(v_b)
        return self._update(y, H, M, gate=c.nis_gate)

    def update_position(self, p_meas: np.ndarray, std: float) -> bool:
        """GNSS/RTK position update (used only before the outage)."""
        H = np.zeros((3, N_ERR))
        H[:, P_] = np.eye(3)
        return self._update(p_meas - self.x.p, H, np.eye(3) * std ** 2)

    def _update(self, y, H, M, gate: float = 0.0) -> bool:
        P = self.P
        S = H @ P @ H.T + M
        S_inv = np.linalg.inv(S)
        self.last_nis = float(y @ S_inv @ y)
        if gate > 0 and self.last_nis > gate:           # reject outlier
            return False
        K = P @ H.T @ S_inv                                                   # Eq. (16)
        dx = K @ y
        IKH = np.eye(N_ERR) - K @ H
        self.P = IKH @ P @ IKH.T + K @ M @ K.T          # Joseph form
        self._inject_and_reset(dx)
        return True

    def _inject_and_reset(self, dx: np.ndarray):
        """Eq. (17): additive for p, v, biases; multiplicative for R. Then reset dx = 0."""
        x = self.x
        x.p = x.p + dx[P_]
        x.v = x.v + dx[V_]
        x.R = x.R @ exp_so3(dx[TH])
        x.ba = x.ba + dx[BA]
        x.bg = x.bg + dx[BG]
        # covariance reset Jacobian G = diag(I, I, I - 0.5[dth]_x, I, I)
        G = np.eye(N_ERR)
        G[TH, TH] = np.eye(3) - 0.5 * skew(dx[TH])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def state(self) -> NominalState:
        return self.x
