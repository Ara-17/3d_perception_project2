# Traditional Multi-Modal Data Fusion (Camera + 77 GHz Doppler Radar)

**Course:** 3D Perception & Learning-Based Data Fusion (A.Y. 2026/2027) — University of Parma  
**Student:** Arashpreet Singh  
**Dataset:** [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes) (`CAM_FRONT` + `RADAR_FRONT` + 3D LiDAR Ground Truth)

---

## 1. Project Overview
This repository implements a **Traditional (Non-Learning) Multi-Modal Sensor Fusion** pipeline for 3D multi-object tracking (MOT), metric depth estimation, and dynamic Time-to-Collision (TTC) safety estimation.

Building upon Project 1 (monocular `YOLOv8m` + `MiDaS DPT_Hybrid`), Project 2 integrates an active **77 GHz Automotive Doppler Radar** to resolve monocular long-range scale drift, reject environmental radar clutter, and measure instantaneous radial velocity ($v_r$) via the Doppler effect.

### Core Pipeline Stages:
1. **3D-to-2D Radar Projection:** 4-stage rigid coordinate transformation (`Radar -> Ego(t_r) -> Global -> Ego(t_c) -> Camera(t_c)`) using quaternion rotations and perspective projection ($\mathbf{K}$).
2. **Combined Visual Prior & Online Calibration:** Merges 50% inner-ROI MiDaS disparity with Pinhole perspective box-height geometry, continuously calibrated online by validated Radar locks.
3. **Inner-Box Clutter Filtering & Camera-Radar Consistency Gate ($\pm 22\%$):** Discards **89.3%** of background Radar clutter and blocks underbody multipath reflections.
4. **Global Hungarian Data Association & 4D Linear Kalman Filter:** Solves frame-to-frame assignment globally (`scipy.optimize.linear_sum_assignment`) and tracks kinematic states $[X, Z, v_{x,\text{rel}}, v_{z,\text{rel}}]^T$ with adaptive sensor noise covariance ($\mathbf{R}_{\text{radar}}$ vs. $\mathbf{R}_{\text{cam}}$).

---

## 2. Quantitative Benchmark Results (`nuScenes v1.0-mini`)
Evaluated across all **10 scenes (404 synchronized keyframes at 2 Hz, 2,309 visible Ground Truth road objects $\le 50\text{ m}$)**:

| Target Class | GT Objects | Precision | Recall | Avg 2D IoU | RMSE Camera (Proj. 1) | RMSE Kalman (All Tracks) | RMSE Radar Lock `[FUSED]` | Doppler Vel. MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Car** | 1,269 | 0.66 | 0.73 | 0.72 | 6.98 m | 4.20 m (-39.8%) | **1.78 m (-74.5%)** | **0.35 m/s** |
| **Truck/Bus** | 263 | 0.67 | 0.52 | 0.75 | 8.18 m | 3.90 m (-52.3%) | **3.57 m (-56.4%)** | **0.24 m/s** |
| **Person** | 612 | 0.77 | 0.46 | 0.63 | 6.19 m | 3.30 m (-46.7%) | **2.71 m (-56.2%)** | **0.34 m/s** |
| **Bike/Moto** | 165 | 0.59 | 0.12 | 0.63 | 5.23 m | 4.14 m (-20.8%) | **3.90 m (-25.4%)** | **0.09 m/s** |
| **TOTAL / MEAN** | **2,309** | — | — | **0.70** | **6.93 m** | **4.00 m (-42.3%)** | **2.21 m (-68.1%)** | **0.33 m/s** |

- **Overall 3D Ranging Error Reduction:** **-42.3%** across all objects (`4.00 m` RMSE) and **-68.1%** on active Radar locks (`2.21 m` RMSE; **-74.5% / 1.78 m on Cars**).
- **Environmental Radar Clutter Rejected:** **89.3%** (`9,832 / 11,011` points).

---

## 3. Repository Structure
```text
3d_perception_project2/
├── tracker.py                # 4D Linear Kalman Filter, 3D-to-2D Radar alignment & gating
├── demo_video.py             # Real-time Camera + Top-Down BEV visualization & MP4 export
├── benchmark_plots.py        # Full 10-scene quantitative evaluation, CSV & plots generator
├── requirements.txt          # Python dependencies
├── output/
│   └── plots_en/             # Generated benchmark plots (300 DPI) & metrics_per_class.csv
└── README.md
```

---

## 4. Setup & Execution

### 1. Install Dependencies
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dataset Placement
Download `v1.0-mini` from nuScenes and extract it inside `data/v1.0-mini/`:
```text
data/v1.0-mini/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

### 3. Run Real-Time Fusion & Top-Down BEV Demo
Change `SCENE_INDEX` (`0` to `9`) in `demo_video.py` to test different urban scenes:
```bash
python demo_video.py
```

### 4. Run Full Quantitative Benchmark & Generate Plots
```bash
python benchmark_plots.py
```