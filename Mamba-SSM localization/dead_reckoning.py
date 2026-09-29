import argparse
import json
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from config import EKFConfig, SimConfig
from es_ekf import ESEKF
from metrics import max_position_drift
from so3 import exp_so3

GRAVITY = np.array([0.0, 0.0, -9.81])




class Recording:
    t: np.ndarray       
    acc: np.ndarray     
    gyr: np.ndarray     
    p: np.ndarray       
    v: np.ndarray       
    R: np.ndarray       

    @property
    def dt(self):
        return float(np.median(np.diff(self.t)))

    @property
    def v_body(self):
        return np.einsum("nji,nj->ni", self.R, self.v)       # R^T v


@dataclass
class IMUErrorModel:
    acc_noise: float = 5.0e-4      
    gyr_noise: float = 3.0e-6      
    acc_bias_rw: float = 5.0e-6    
    gyr_bias_rw: float = 1.0e-7    
    acc_bias0: float = 5.0e-4      
    gyr_bias0: float = 2.5e-6      
    def ekf_config(self, base: EKFConfig = EKFConfig()) -> EKFConfig:
        return replace(base, sigma_a=self.acc_noise, sigma_g=self.gyr_noise,
                       sigma_ba=self.acc_bias_rw, sigma_bg=self.gyr_bias_rw,
                       init_std=(0.05, 0.05, 0.005, self.acc_bias0, self.gyr_bias0))

    def corrupt(self, rec: Recording, rng) -> Recording:
        n, dt = len(rec.t), rec.dt
        ba = rng.normal(0, self.acc_bias0, 3) + np.cumsum(rng.normal(0, self.acc_bias_rw * np.sqrt(dt), (n, 3)), 0)
        bg = rng.normal(0, self.gyr_bias0, 3) + np.cumsum(rng.normal(0, self.gyr_bias_rw * np.sqrt(dt), (n, 3)), 0)
        acc = rec.acc + ba + rng.normal(0, self.acc_noise / np.sqrt(dt), (n, 3))
        gyr = rec.gyr + bg + rng.normal(0, self.gyr_noise / np.sqrt(dt), (n, 3))
        return replace(rec, acc=acc, gyr=gyr)



def generate_trajectory(duration: float, rate: float = 100.0, seed: int = 0) -> Recording:
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate
    n = int(duration * rate) + 1
    t = np.arange(n) * dt

    # --- speed profile: random targets (incl. stops), accel-limited first-order tracking ---
    speed = np.zeros(n)
    v, target, t_next = 0.0, 10.0, 0.0
    for k in range(n):
        if t[k] >= t_next:
            target = 0.0 if rng.random() < 0.08 else rng.uniform(4.0, 22.0)
            t_next = t[k] + rng.uniform(8.0, 35.0)
        acc_cmd = np.clip((target - v) / 3.0, -3.5, 2.5)
        v = max(v + acc_cmd * dt, 0.0)
        speed[k] = v

    # --- curvature profile: straights / curves, lateral acc. limited to 4 m/s^2 ---
    kappa = np.zeros(n)
    kc, ktarget, t_next = 0.0, 0.0, 0.0
    for k in range(n):
        if t[k] >= t_next:
            ktarget = 0.0 if rng.random() < 0.35 else rng.choice([-1, 1]) * rng.uniform(1 / 150, 1 / 15)
            t_next = t[k] + rng.uniform(3.0, 15.0)
        klim = 4.0 / max(speed[k] ** 2, 1.0)
        kc += (np.clip(ktarget, -klim, klim) - kc) * dt / 1.2
        kappa[k] = kc

    yaw_rate = kappa * speed
    yaw = np.cumsum(yaw_rate) * dt
    a_lat = speed * yaw_rate
    a_lon = np.gradient(speed, dt)

    # sideslip (kinematic + dynamic part), road grade, roll/pitch from load transfer
    beta = 1.3 * kappa - 0.004 * a_lat
    grade = 0.015 * np.sin(2 * np.pi * t / 90.0) + 0.01 * np.sin(2 * np.pi * t / 37.0 + 1.0)
    roll = 0.006 * a_lat
    pitch = grade - 0.004 * a_lon

    v_road = np.stack([speed * np.cos(beta), speed * np.sin(beta), np.zeros(n)], 1)
    R_road = Rotation.from_euler("ZYX", np.stack([yaw, grade, np.zeros(n)], 1))
    v_g = R_road.apply(v_road)
    R = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], 1)).as_matrix()

    # --- IMU consistent with the filter's discrete mechanisation (Eq. 10) ---
    a_g = np.zeros((n, 3))
    a_g[:-1] = np.diff(v_g, axis=0) / dt
    a_g[-1] = a_g[-2]
    rel = Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:]))   # R_k^T R_{k+1}
    gyr = np.zeros((n, 3))
    gyr[:-1] = rel.as_rotvec() / dt
    gyr[-1] = gyr[-2]
    acc = np.einsum("nji,nj->ni", R, a_g - GRAVITY)                          # f = R^T (a - g)
    p = np.zeros((n, 3))
    p[1:] = np.cumsum(v_g[:-1] * dt + 0.5 * a_g[:-1] * dt ** 2, axis=0)
    return Recording(t=t, acc=acc, gyr=gyr, p=p, v=v_g, R=R)



class VelocitySource:
    """Interface: get(k) -> (v_xy [2], var_xy [2]) at IMU index k, or None."""
    name = "base"

    def get(self, k: int):
        raise NotImplementedError


class NoVelocity(VelocitySource):
    name = "No Correction"

    def get(self, k):
        return None


class PrecomputedVelocity(VelocitySource):

    def __init__(self, t_imu, t_meas, v_xy, var_xy, name="EVC-Mamba"):
        self.name = name
        idx = np.searchsorted(t_imu, t_meas)
        idx = np.clip(idx, 0, len(t_imu) - 1)
        self.table: Dict[int, tuple] = {int(i): (v, s) for i, v, s in zip(idx, v_xy, var_xy)}

    def get(self, k):
        return self.table.get(k)


@dataclass
class SensorErrorSpec:
    rmse: tuple                  # target RMSE (v_x, v_y) [m/s], Table I
    corr_frac: float = 0.6       # share of error variance that is time-correlated
    tau: float = 2.0             # correlation time of the Gauss-Markov part [s]
    heteroscedastic: bool = True # error grows with dynamic manoeuvres
    evidential: bool = False     # reports per-sample variance (delta_v) vs. fixed variance
    unc_noise: float = 0.35      # log-normal scatter of reported vs. true std


SENSOR_PRESETS = {
    "OSD-Baseline": SensorErrorSpec(rmse=(0.21, 0.081), corr_frac=0.8, tau=5.0),
    "Transformer": SensorErrorSpec(rmse=(0.08, 0.017), corr_frac=0.6, tau=2.0),
    "EVC-Mamba": SensorErrorSpec(rmse=(0.07, 0.011), corr_frac=0.6, tau=2.0, evidential=True),
    "Correvit": SensorErrorSpec(rmse=(0.03, 0.010), corr_frac=0.5, tau=1.0, heteroscedastic=False),
}


class SimulatedVelocitySensor(VelocitySource):
    error = s(t) * [ white + first-order Gauss-Markov ]
    s(t)  = heteroscedastic scale (manoeuvre intensity), normalised to keep the target
            RMSE, and faded to zero at standstill (wheel speeds are exact at rest).
    Evidential sensors report var = (s * sigma)^2 * LogNormal(0, unc_noise^2),
    others report the constant var = RMSE^2.
    """

    def __init__(self, name: str, rec: Recording, meas_idx: np.ndarray, spec: SensorErrorSpec, rng):
        self.name = name
        t, vb = rec.t[meas_idx], rec.v_body[meas_idx, :2]
        n = len(meas_idx)
        rmse = np.asarray(spec.rmse)
        speed = np.linalg.norm(vb, axis=1)

        if spec.heteroscedastic:
            acc_h = np.linalg.norm(rec.acc[meas_idx, :2] - rec.acc[meas_idx, :2].mean(0), axis=1)
            s = 1.0 + np.abs(acc_h) / 1.5
        else:
            s = np.ones(n)
        s = s * np.clip(speed / 1.0, 0.0, 1.0)                      # exact at standstill
        s = s / np.sqrt(np.mean(s ** 2))                            # keep target RMSE

        dtm = np.median(np.diff(t))
        phi = np.exp(-dtm / spec.tau)
        sig_c = rmse * np.sqrt(spec.corr_frac)
        sig_w = rmse * np.sqrt(1.0 - spec.corr_frac)
        gm = np.zeros((n, 2))
        drive = rng.normal(0, 1, (n, 2)) * sig_c * np.sqrt(1 - phi ** 2)
        gm[0] = rng.normal(0, 1, 2) * sig_c
        for i in range(1, n):
            gm[i] = phi * gm[i - 1] + drive[i]
        err = s[:, None] * (gm + rng.normal(0, 1, (n, 2)) * sig_w)

        if spec.evidential:
            var = (s[:, None] * rmse) ** 2 * np.exp(rng.normal(0, spec.unc_noise, (n, 2)))
        else:
            var = np.tile(rmse ** 2, (n, 1))
        self.table = {int(k): (vb[i] + err[i], var[i]) for i, k in enumerate(meas_idx)}
        self.v_meas, self.v_true, self.var = vb + err, vb, var

    def get(self, k):
        return self.table.get(k)



def run_scenario(rec: Recording, source: VelocitySource, k0: int, align_time: float,
                 outage: float, ekf_cfg: EKFConfig, sim: SimConfig, rng, return_track=False):
    dt = rec.dt
    n_align = int(round(align_time / dt))
    n_out = int(round(outage / dt))
    k_gnss = max(int(round(1.0 / (sim.gnss_rate * dt))), 1)

    # initialise from reference with a small attitude error
    R0 = rec.R[k0] @ exp_so3(rng.normal(0, 0.002, 3))
    ekf = ESEKF(ekf_cfg, rec.p[k0], rec.v[k0], R0)

    track = np.zeros((n_out, 3))
    for j in range(n_align + n_out):
        k = k0 + j
        ekf.predict(rec.acc[k], rec.gyr[k], dt)
        kk = k + 1                                                  # filter now at t[kk]
        if j < n_align and (kk - k0) % k_gnss == 0:                 # GNSS available
            ekf.update_position(rec.p[kk] + rng.normal(0, sim.gnss_std, 3), sim.gnss_std)
        m = source.get(kk)
        if m is not None:
            ekf.update_velocity(*m)
        if j >= n_align:
            track[j - n_align] = ekf.x.p
    truth = rec.p[k0 + n_align + 1:k0 + n_align + n_out + 1]
    drift = max_position_drift(track, truth)
    return (drift, track, truth) if return_track else drift



def scenario_starts(n_samples, dt, sim: SimConfig):
    span = int(round((sim.align_time + max(sim.outage_durations)) / dt)) + 2
    starts = [i * span for i in range(sim.n_scenarios)]
    if starts[-1] + span >= n_samples:
        raise ValueError("Recording too short for the requested non-overlapping scenarios.")
    return starts


_SHARED = {}   # recording + sources, inherited by forked workers (avoids pickling large arrays)


def _job(args):
    s_idx, k0, dur, seed = args
    rec, sources, ekf_cfg, sim = (_SHARED[k] for k in ("rec", "sources", "ekf_cfg", "sim"))
    src = sources[s_idx]
    return src.name, dur, run_scenario(rec, src, k0, sim.align_time, dur, ekf_cfg, sim,
                                       np.random.default_rng(seed))


def evaluate_outages(rec: Recording, sources, ekf_cfg: EKFConfig, sim: SimConfig, workers=1):
    starts = scenario_starts(len(rec.t), rec.dt, sim)
    _SHARED.update(rec=rec, sources=list(sources), ekf_cfg=ekf_cfg, sim=sim)
    # same seed per scenario across methods -> identical GNSS noise / init errors
    jobs = [(si, k0, dur, sim.seed + 1000 * i)
            for si in range(len(sources)) for dur in sim.outage_durations for i, k0 in enumerate(starts)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_job, jobs)
    else:
        out = [_job(j) for j in jobs]
    res = {}
    for name, dur, d in out:
        res.setdefault(name, {}).setdefault(dur, []).append(d)
    return res


def print_table(res, durations):
    head = f"{'Method':<16}" + "".join(f"{int(d // 60)} min".rjust(16) for d in durations)
    print("\nMaximum horizontal position drift [m] (mean ± std over scenarios)")
    print(head + "\n" + "-" * len(head))
    for name, per in res.items():
        row = "".join(f"{np.mean(per[d]):7.2f} ± {np.std(per[d]):5.2f}".rjust(16) for d in durations)
        print(f"{name:<16}{row}")


def build_sources(rec: Recording, sim: SimConfig, methods, seed=0):
    rng = np.random.default_rng(seed + 7)
    dt = rec.dt
    t_meas = np.arange(0.0, rec.t[-1], 1.0 / sim.vel_rate)
    meas_idx = np.unique(np.clip(np.round(t_meas / dt).astype(int), 1, len(rec.t) - 1))
    out = []
    for m in methods:
        if m == "No Correction":
            out.append(NoVelocity())
        else:
            out.append(SimulatedVelocitySensor(m, rec, meas_idx, SENSOR_PRESETS[m], rng))
    return out


def plot_track(rec, sources, ekf_cfg, sim, k0, outage, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"No Correction": ("r:", 1.2), "EVC-Mamba": ("b--", 1.2), "Correvit": ("k:", 1.2),
              "OSD-Baseline": ("m-.", 1.0), "Transformer": ("c--", 1.0)}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    truth = None
    for src in sources:
        _, track, truth = run_scenario(rec, src, k0, sim.align_time, outage, ekf_cfg, sim,
                                       np.random.default_rng(sim.seed), return_track=True)
        st, lw = styles.get(src.name, ("-", 1.0))
        ax.plot(track[:, 0], track[:, 1], st, lw=lw, label=src.name)
    ax.plot(truth[:, 0], truth[:, 1], "g-", lw=2, label="Ground Truth", zorder=0)
    ax.plot(*truth[0, :2], "k^", label="Start")
    ax.plot(*truth[-1, :2], "ks", label="End")
    ax.set_xlabel("X position (m)"); ax.set_ylabel("Y position (m)")
    ax.set_title(f"{outage / 60:.0f}-min GNSS outage"); ax.axis("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description="Simulate GNSS outages with ES-EKF + virtual velocity correction")
    ap.add_argument("--durations", type=float, nargs="+", default=[120, 300, 600], help="outage lengths [s]")
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--imu-rate", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot", default="outage_trajectory.png")
    ap.add_argument("--json", default="outage_results.json")
    args = ap.parse_args()

    sim = SimConfig(imu_rate=args.imu_rate, outage_durations=tuple(args.durations),
                    n_scenarios=args.scenarios, seed=args.seed)
    span = sim.align_time + max(sim.outage_durations)
    rec_true = generate_trajectory(sim.n_scenarios * span + 30.0, sim.imu_rate, seed=args.seed)
    imu_model = IMUErrorModel()
    rec = imu_model.corrupt(rec_true, np.random.default_rng(args.seed + 1))
    ekf_cfg = imu_model.ekf_config()
    sources = build_sources(rec, sim, sim.methods, args.seed)

    for s in sources:  # sanity: emulated velocity accuracy (cf. Table I)
        if isinstance(s, SimulatedVelocitySensor):
            e = s.v_meas - s.v_true
            print(f"{s.name:<14} RMSE vx {np.sqrt(np.mean(e[:, 0]**2)):.3f}  vy {np.sqrt(np.mean(e[:, 1]**2)):.3f} m/s")

    res = evaluate_outages(rec, sources, ekf_cfg, sim, workers=args.workers)
    print_table(res, sim.outage_durations)
    with open(args.json, "w") as f:
        json.dump({m: {str(d): v for d, v in per.items()} for m, per in res.items()}, f, indent=2)

    if args.plot:
        dur = 300.0 if 300.0 in sim.outage_durations else sim.outage_durations[0]
        pick = [s for s in sources if s.name in ("No Correction", "EVC-Mamba", "Correvit")]
        plot_track(rec, pick, ekf_cfg, sim, scenario_starts(len(rec.t), rec.dt, sim)[0], dur, args.plot)


if __name__ == "__main__":
    main()
