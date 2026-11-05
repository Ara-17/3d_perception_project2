# Traditional Multi-Modal Data Fusion (Camera + 77 GHz Doppler Radar)

**Course:** 3D Perception & Learning-Based Data Fusion (A.Y. 2026/2027) - University of Parma  
**Student:** Arashpreet Singh  
**Dataset:** [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes) (`CAM_FRONT` + `RADAR_FRONT` + 3D LiDAR Ground Truth)

---

## 1. Overview
This repository contains the Python/PyTorch implementation for **Project 2 (Traditional Multi-Modal Data Fusion)**.

Building upon Project 1 (monocular `YOLOv8m` + `MiDaS DPT_Hybrid`), this pipeline fuses the passive front RGB camera (`CAM_FRONT`) with an active **77 GHz Automotive Doppler Radar (`RADAR_FRONT`)** using deterministic, non-learning algorithms. By combining visual semantics with physical Radar range and Doppler radial velocity (`v_r`), the system resolves long-range monocular scale drift, filters out environmental Radar clutter, and performs continuous 3D Multi-Object Tracking (MOT) with dynamic Time-to-Collision (TTC) safety alerts.

### Core Pipeline Stages:
1. **3D-to-2D Radar Projection:** Transforms 3D Radar point clouds from the front bumper into the windshield camera frame through a 4-stage rigid coordinate chain (`Radar(t_r) -> Ego(t_r) -> Global -> Ego(t_c) -> Camera(t_c)`) using sensor calibration quaternions (`R, T`), GPS/IMU ego-pose compensation, and the camera intrinsic matrix (`K`).
2. **Visual Depth Estimation & Online Radar Calibration:** Combines 50% inner-ROI MiDaS disparity sampling with a Pinhole geometric box-height model (`Z_pinhole = (f_y * H_real) / h_box`), continuously calibrated online whenever a vehicle is locked by both Camera and Radar.
3. **Inner-Box Clutter Filtering & Camera-Radar Check (±22% Gate):** Discards **89.3%** of raw environmental Radar clutter (barriers, walls, curbs), compensates for the bumper-to-center geometric offset (`+1.30 m` for Cars, `+2.10 m` for Trucks/Buses), and blocks underbody multipath reflections using a `±22%` camera-radar consistency gate.
4. **Global Hungarian Data Association & 4D Linear Kalman Filter:** Solves frame-to-frame object assignment globally via the Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`) and tracks each target's 4D kinematic state `[X, Z, vx_rel, vz_rel]` using adaptive measurement noise matrices (`R_radar` for active `[FUSED]` locks vs. `R_cam` for optical `[CAM]` fallback).

---

## 2. Project Structure
* `tracker.py` — Core module implementing the 4D Linear Kalman Filter (`KalmanTrack`), 3D-to-2D sensor alignment, Ground Truth extraction, and Camera-Radar consistency gating.
* `demo_video.py` — Real-time visualization script rendering the split-screen Camera view + Top-Down Bird's-Eye View (BEV) radar map and exporting `.mp4` videos for any scene (`SCENE_INDEX` from `0` to `9`).
* `benchmark_plots.py` — Quantitative evaluation script running across all 10 scenes (404 keyframes) to compute per-class metrics, export `metrics_per_class.csv`, and generate the 7 benchmark charts in `output/plots_en/`.
* `requirements.txt` — List of required Python dependencies.
* `.gitignore` — Excludes virtual environments, heavy raw datasets, pre-trained weights, and rendered video files.

---

## 3. Dataset, Videos & Presentation Material (OneDrive)
Due to GitHub file size limits, the `nuScenes v1.0-mini` dataset, the rendered split-screen output videos (`video_scene_0.mp4`, etc.), the technical report (PDF), and the presentation slides (PPTX/PDF) are hosted on the cloud at the following link:

**[Download Dataset, Demo Videos, Report & Slides (OneDrive)](https://univpr-my.sharepoint.com/:f:/g/personal/arashpreet_singh_studenti_unipr_it/IgCL0NCJ1b8uSaIzFz0YgGMYAe_3fYTuF4HRvTK-8OqkuEI?e=QaItTG)**

### How to set up the data:
1. Download the `nuScenes v1.0-mini` archive from the OneDrive link above (or from the official [nuScenes download page](https://www.nuscenes.org/download)).
2. Create a folder named `data/v1.0-mini/` inside the project root directory and extract the dataset so that it follows this exact structure:

```text
3d_perception_project2/
└── data/
    └── v1.0-mini/
        ├── maps/
        ├── samples/
        ├── sweeps/
        └── v1.0-mini/
```

---

## 4. Quantitative Benchmark Results (`nuScenes v1.0-mini`)
The pipeline was evaluated across all **10 urban scenes (404 synchronized keyframes at 2 Hz, Δt = 0.5 s)** on **2,309 visible Ground Truth road objects (≤ 50 m in `CAM_FRONT`)**:

| Target Class | GT Objects | Precision | Recall | Avg 2D IoU | RMSE Camera (Proj. 1) | RMSE Kalman (All Tracks) | RMSE Radar Lock `[FUSED]` | Doppler Vel. MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Car** | 1,269 | 0.66 | 0.73 | 0.72 | 6.98 m | 4.20 m (-39.8%) | **1.78 m (-74.5%)** | **0.35 m/s** |
| **Truck/Bus** | 263 | 0.67 | 0.52 | 0.75 | 8.18 m | 3.90 m (-52.3%) | **3.57 m (-56.4%)** | **0.24 m/s** |
| **Person** | 612 | 0.77 | 0.46 | 0.63 | 6.19 m | 3.30 m (-46.7%) | **2.71 m (-56.2%)** | **0.34 m/s** |
| **Bike/Moto** | 165 | 0.59 | 0.12 | 0.63 | 5.23 m | 4.14 m (-20.8%) | **3.90 m (-25.4%)** | **0.09 m/s** |
| **TOTAL / MEAN** | **2,309** | — | — | **0.70** | **6.93 m** | **4.00 m (-42.3%)** | **2.21 m (-68.1%)** | **0.33 m/s** |

* **Overall 3D Distance RMSE Reduction:** **-42.3%** across all tracked objects (`4.00 m` vs. `6.93 m`) and **-68.1%** on targets with active Radar locks (`2.21 m` overall; **-74.5% / 1.78 m on Cars**).
* **Doppler Radial Velocity Accuracy:** Mean Absolute Error (MAE) of **0.33 m/s** (~1.2 km/h) compared to 3D Ground Truth velocity.
* **Environmental Radar Clutter Rejected:** **89.3%** (`9,832 / 11,011` raw Radar points filtered out).

---

## 5. Installation & Execution

1. **Create and activate a virtual environment (Windows PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install the required libraries:**
```powershell
pip install -r requirements.txt
```

3. **Run the real-time Camera + BEV fusion demo (and export `.mp4` video):**
*(You can change `SCENE_INDEX = 0..9` at the top of `demo_video.py` to test different daytime, nighttime, or rainy scenes)*
```powershell
python demo_video.py
```

4. **Run the full 10-scene quantitative benchmark (generates CSV table and PNG charts):**
```powershell
python benchmark_plots.py
```
*(Note: Pre-trained weights for YOLOv8m and MiDaS are loaded automatically from local cache or downloaded on the first run).*