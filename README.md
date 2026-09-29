# Evidential Mamba for Localization

**Uncertainty-Aware Velocity Correction for Proprioceptive Vehicle Localization using Evidential Mamba**

Abinav Kalyanasundaram, Karthikeyan Chandra Sekaran, Wolfgang Utschick, Michael Botsch  
AImotion Bavaria, Technische Hochschule Ingolstadt · Technische Universität München

[![arXiv](https://img.shields.io/badge/arXiv-2607.05669-b31b1b.svg)](https://arxiv.org/abs/2607.05669)
[![Dataset](https://img.shields.io/badge/Dataset-ReV--StED-blue.svg)](https://doi.org/10.5281/zenodo.15270060)

## Overview

Inertial navigation drifts without bound when GNSS is unavailable, for example in tunnels or parking garages. **EVC-Mamba** turns a vehicle's existing onboard sensors (wheel speeds, steering wheel angle, yaw rate, lateral acceleration, brake pressure) into a **virtual velocity sensor**, so no extra hardware is needed.

- **Mamba selective state space model:** models the temporal dynamics of the sensor data in linear time.
- **Evidential regression head:** predicts a Normal-Inverse-Gamma distribution, which gives the velocity and its uncertainty in a single forward pass.
- **Error-State EKF:** fuses the uncertainty-aware velocity as a virtual measurement alongside IMU mechanization to limit position drift.

## Results

Evaluated on the [ReV-StED dataset](https://doi.org/10.5281/zenodo.15270060).

**Velocity estimation (m/s)**

| Model | vx MAE | vx RMSE | vy MAE | vy RMSE |
|---|---|---|---|---|
| OSD-Baseline | 0.18 | 0.21 | 0.048 | 0.081 |
| Transformer | 0.03 | 0.08 | 0.008 | 0.017 |
| **EVC-Mamba** | **0.03** | **0.07** | **0.006** | **0.011** |

**Maximum position drift (m) during GNSS outage**

| Method | 2 min | 5 min | 10 min |
|---|---|---|---|
| No correction | 7.8 ± 2.3 | 31.8 ± 5.1 | 142 ± 29.9 |
| OSD-Baseline | 1.6 ± 0.4 | 4.8 ± 2.1 | 13.0 ± 3.4 |
| Transformer | 1.2 ± 0.5 | 2.9 ± 1.5 | 7.2 ± 2.6 |
| **EVC-Mamba** | **1.0 ± 0.5** | **2.4 ± 1.3** | **5.9 ± 2.0** |
| Correvit (external sensor) | 0.9 ± 0.5 | 2.2 ± 1.2 | 5.4 ± 1.7 |

EVC-Mamba comes within about 10% of a dedicated external velocity sensor and runs in real time on an NVIDIA Jetson Orin (20–24 ms per inference, 22.8 MFLOPs).

The Code is currently being reorganized for easier implementation and will be completed by Oct 4th. 

## Citation

```bibtex
@article{kalyanasundaram2026evcmamba,
  title   = {Uncertainty-Aware Velocity Correction for Proprioceptive Vehicle Localization using Evidential Mamba},
  author  = {Kalyanasundaram, Abinav and Chandra Sekaran, Karthikeyan and Utschick, Wolfgang and Botsch, Michael},
  journal = {arXiv preprint arXiv:2607.05669},
  year    = {2026}
}
```
