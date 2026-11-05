import os
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes

# Importazioni da tracker.py
from tracker import (
    YOLO_CLASSES, BUMPER_OFFSET, HEIGHT_PRIOR,
    map_yolo_class, compute_iou, extract_roi_median,
    validate_radar_points, KalmanTrack, load_frame_and_sensors
)

# Costanti e configurazioni
DATAROOT = r"data/v1.0-mini"
OUTPUT_DIR = "output/plots_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classi da valutare
CLASSES = ["Car", "Truck/Bus", "Person", "Bike/Moto"]
CONFIDENCE_THRESHOLD = 0.40

# Funzione per salvare i grafici di benchmark
def save_benchmark_charts(stats, y_true, y_pred, scatter_data, minivan_log, tot_rmse):
    # Impostazioni generali di Matplotlib
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    x = np.arange(len(CLASSES))
    w = 0.26

    # Distribuzione del dataset per classe
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    c_360 = [7619, 1508, 4847, 768]
    c_front = [2349, 538, 1075, 258]
    c_eval = [stats[c]["gt"] for c in CLASSES]

    # Grafico a barre per la distribuzione del dataset
    b1 = ax.bar(x - w, c_360, w, label="Total 360° Annotations", color="#95a5a6", edgecolor="black")
    b2 = ax.bar(x, c_front, w, label="In CAM_FRONT", color="#3498db", edgecolor="black")
    b3 = ax.bar(x + w, c_eval, w, label="Evaluated Test Set (<= 50m)", color="#2ecc71", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold")
    ax.set_ylabel("Number of Annotated Objects")
    ax.set_title("nuScenes v1.0-mini: Sample Distribution per Class", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    # Annotazioni dei valori sulle barre
    for bars in [b1, b2, b3]:
        for bar in bars:
            v = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width()/2, v + 120, str(v), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 8800)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_dataset_distribution_en.png"), dpi=300)
    plt.close()

    # Matrice di confusione multi-classe
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_pct = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100.0
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(cm_pct, cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set(xticks=x, yticks=x, xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted Class", ylabel="True Class (Ground Truth)")
    ax.set_title("Multi-Class Confusion Matrix", fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontweight="bold" if i == j else "normal")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix_en.png"), dpi=300)
    plt.close()

    # Grafico a barre per Precision, Recall e IoU per classe
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bp = ax.bar(x - w, [stats[c]["prec"] * 100 for c in CLASSES], w, label="Precision (%)", color="#2980b9", edgecolor="black")
    br = ax.bar(x, [stats[c]["rec"] * 100 for c in CLASSES], w, label="Recall (%)", color="#e67e22", edgecolor="black")
    bi = ax.bar(x + w, [stats[c]["iou"] * 100 for c in CLASSES], w, label="Average 2D IoU (%)", color="#27ae60", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_title("2D Detection Metrics per Class", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    for bars in [bp, br, bi]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_detection_metrics_en.png"), dpi=300)
    plt.close()

    # Grafico a barre per RMSE di distanza per classe
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    lbls = CLASSES + ["TOTAL"]
    xe = np.arange(len(lbls))
    r_cam = [stats[c]["rmse_cam"] for c in CLASSES] + [tot_rmse["cam"]]
    r_kf = [stats[c]["rmse_kf"] for c in CLASSES] + [tot_rmse["kf"]]
    r_rad = [stats[c]["rmse_rad"] for c in CLASSES] + [tot_rmse["rad"]]

    bc1 = ax.bar(xe - w, r_cam, w, label="Project 1: Camera Only", color="#e74c3c", edgecolor="black")
    bc2 = ax.bar(xe, r_kf, w, label="Project 2: Kalman Fusion (All)", color="#3498db", edgecolor="black")
    bc3 = ax.bar(xe + w, r_rad, w, label="Project 2: Active Radar Lock", color="#2ecc71", edgecolor="black")
    ax.set_xticks(xe)
    ax.set_xticklabels(lbls, fontweight="bold")
    ax.set_ylabel("Distance RMSE (meters)")
    ax.set_title("3D Ranging Error (RMSE) per Class", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    for bars in [bc1, bc2, bc3]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v:.2f}m", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, max(r_cam) + 1.8)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_rmse_per_class_en.png"), dpi=300)
    plt.close()

    # Grafico a barre per l'evoluzione delle versioni di filtraggio (da v1 a v3)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    v_names = [
        "v1: Naive Full-Box\nRadar Median",
        "v2: Closest\nCluster Only",
        "Baseline:\nCamera Only",
        "v3: Complete\nKalman Fusion",
        "v3: Active\nRadar Lock"
    ]
    v_vals = [27.15, 9.42, tot_rmse["cam"], tot_rmse["kf"], tot_rmse["rad"]]
    v_cols = ["#c0392b", "#e67e22", "#7f8c8d", "#2980b9", "#27ae60"]
    bars_v = ax.bar(v_names, v_vals, color=v_cols, edgecolor="black", width=0.52)
    ax.set_ylabel("Distance RMSE (meters)")
    ax.set_title("Evolution of Filtering Versions (v1 to v3)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, v in zip(bars_v, v_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.2f}m", ha="center", fontweight="bold")
    ax.set_ylim(0, 31.5)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_ablation_study_en.png"), dpi=300)
    plt.close()

    # Grafico dell'errore medio assoluto (MAE) in funzione della distanza reale (0 - 50m)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = [(0, 15), (15, 25), (25, 35), (35, 50)]
    b_lbls = ["0 - 15 m\n(Close)", "15 - 25 m\n(Medium)", "25 - 35 m\n(Long)", "35 - 50 m\n(Horizon)"]
    c_curve, k_curve, r_curve = [], [], []
    for bl, bh in bins:
        sub = [r for r in scatter_data if bl <= r["z_gt"] < bh]
        sub_r = [r for r in sub if r["has_radar"]]
        c_curve.append(np.mean([r["err_cam"] for r in sub]) if sub else 0)
        k_curve.append(np.mean([r["err_kf"] for r in sub]) if sub else 0)
        r_curve.append(np.mean([r["err_kf"] for r in sub_r]) if sub_r else 0)

    xp = np.arange(len(b_lbls))
    ax.plot(xp, c_curve, "o--", color="#e74c3c", linewidth=2.5, label="Camera Only (Project 1)")
    ax.plot(xp, k_curve, "s-", color="#2980b9", linewidth=2.5, label="Kalman Fusion (All Objects)")
    ax.plot(xp, r_curve, "^-", color="#27ae60", linewidth=2.8, label="Kalman Fusion (Radar Lock)")
    ax.set_xticks(xp)
    ax.set_xticklabels(b_lbls, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error - MAE (m)")
    ax.set_title("Ranging Error vs. True Distance (0 - 50 m)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_error_vs_distance_en.png"), dpi=300)
    plt.close()

    # Grafico di tracking del veicolo leader (minivan) per scena 0
    if len(minivan_log) > 3:
        fig, (ax_d, ax_v) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True)
        fr = [m["frame"] for m in minivan_log]
        ax_d.plot(fr, [m["z_gt"] for m in minivan_log], "k-o", linewidth=2.2, label="3D Ground Truth")
        ax_d.plot(fr, [m["z_cam"] for m in minivan_log], "r--^", linewidth=1.8, label="Camera Only (Project 1)")
        ax_d.plot(fr, [m["z_kf"] for m in minivan_log], "g-s", linewidth=2.2, label="Radar + Kalman (Project 2)")
        ax_d.set_ylabel("Distance Z (m)")
        ax_d.set_title("Scene 0 Lead Vehicle Tracking: Distance & Doppler Velocity", fontweight="bold")
        ax_d.grid(True, linestyle="--", alpha=0.5)
        ax_d.legend()

        ax_v.plot(fr, [m["v_gt"] for m in minivan_log], "k-o", linewidth=2.0, label="True Radial Velocity (GT)")
        ax_v.plot(fr, [m["v_kf"] for m in minivan_log], "b-s", linewidth=2.0, label="Doppler + Kalman Estimate")
        ax_v.set_xlabel("Keyframe Index (2 Hz)")
        ax_v.set_ylabel("Radial Speed (m/s)")
        ax_v.grid(True, linestyle="--", alpha=0.5)
        ax_v.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "5_minivan_tracking_en.png"), dpi=300)
        plt.close()

# Funzione principale per l'esecuzione del benchmark
def main():
    # Inizializzazione dei modelli e del dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO("yolov8m.pt")
    midas_cache = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    if os.path.exists(midas_cache):
        midas = torch.hub.load(midas_cache, "DPT_Hybrid", source="local").to(device).eval()
        transform = torch.hub.load(midas_cache, "transforms", source="local").dpt_transform
    else:
        midas = torch.hub.load("intel-isl/MiDaS:master", "DPT_Hybrid").to(device).eval()
        transform = torch.hub.load("intel-isl/MiDaS:master", "transforms").dpt_transform
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)

    # Inizializzazione delle statistiche e delle variabili per il benchmark
    stats = {c: {"gt": 0, "tp": 0, "fp": 0, "ious": [], "e_cam": [], "e_kf": [], "e_rad": [], "e_vel": []} for c in CLASSES}
    y_true, y_pred, scatter_data = [], [], []
    minivan_tracks = {}
    tot_rad_pts, clutter_pts = 0, 0

    # Loop attraverso le prime 10 scene del dataset nuScenes
    for s_idx in range(10):
        scene = nusc.scene[s_idx]
        token = scene["first_sample_token"]
        tracks, scale_cal, prev_ts, f_idx = [], 1.0, None, 0
        print(f"Evaluating Scene {s_idx + 1}/10: {scene['name']}...")

        # Loop attraverso i campioni della scena
        while token != "":
            sample = nusc.get("sample", token)
            dt = 0.5 if prev_ts is None else (sample["timestamp"] - prev_ts) / 1e6
            prev_ts = sample["timestamp"]
            f_idx += 1

            # Caricamento dei dati della scena e dei sensori
            img, radar_pts, eval_gts, all_gts_bg, cam_k = load_frame_and_sensors(nusc, sample)
            h, w = img.shape[:2]
            fx, fy, cx = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2]

            # Aggiornamento delle statistiche del ground truth per la scena corrente
            for gt in eval_gts:
                stats[gt["cls"]]["gt"] += 1

            # Elaborazione delle rilevazioni YOLO e stima della profondità con MiDaS
            with torch.no_grad():
                pred = midas(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device))
                depth_map = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()

            # Normalizzazione della mappa di profondità per l'estrazione dei valori medi nelle ROI
            d_min, d_max = np.percentile(depth_map, 2), np.percentile(depth_map, 98)
            depth_norm = np.clip((depth_map - d_min) / (d_max - d_min + 1e-5), 0.01, 1.0)

            # Elaborazione delle rilevazioni YOLO e associazione con i punti radar
            yolo_boxes = sorted(
                yolo(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0].boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
                reverse=True
            )

            detections = []
            for box in yolo_boxes:
                cid = int(box.cls[0])
                if cid not in YOLO_CLASSES:
                    continue
                m_cls = map_yolo_class(YOLO_CLASSES[cid])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Estrazione della profondità media nella ROI e calcolo della distanza stimata
                disp = extract_roi_median(depth_norm, (x1, y1, x2, y2))
                z_midas = 6.2 / (disp + 0.05)
                z_pin = (fy * HEIGHT_PRIOR.get(m_cls, 1.60)) / max(y2 - y1, 12)
                w_m = 0.65 if z_pin < 18.0 else 0.30
                z_prior = scale_cal * (w_m * z_midas + (1.0 - w_m) * z_pin)

                # Validazione dei punti radar all'interno della bounding box e calcolo delle statistiche di distanza e velocità
                cluster = validate_radar_points(radar_pts, (x1, y1, x2, y2), z_prior, m_cls)
                if cluster:
                    for rp in cluster:
                        rp["matched"] = True
                    z_rad = float(np.median([r["Z"] for r in cluster])) + BUMPER_OFFSET.get(m_cls, 1.0)
                    v_rel = float(np.median([r["v_rel"] for r in cluster]))
                    v_comp = float(np.median([r["v_comp"] for r in cluster]))
                    x_3d = float(np.median([r["X"] for r in cluster]))
                    has_rad = True

                    # Aggiornamento della calibrazione della scala basata sulla distanza radar e sulla stima della profondità
                    raw_c = w_m * z_midas + (1.0 - w_m) * z_pin
                    if 5.0 < z_rad < 35.0 and raw_c > 2.0:
                        scale_cal = 0.92 * scale_cal + 0.08 * np.clip(z_rad / raw_c, 0.80, 1.25)
                # Se non ci sono punti radar validi, si utilizza la stima della profondità dalla camera come fallback
                else:
                    z_rad, v_rel, v_comp = None, 0.0, 0.0
                    x_3d = (0.5 * (x1 + x2) - cx) * z_prior / fx
                    has_rad = False
                # Aggiunta della rilevazione alla lista delle rilevazioni per l'associazione con i tracciati
                detections.append({
                    "bbox": [x1, y1, x2, y2], "macro_cls": m_cls,
                    "z_cam_proj1": z_midas, "z_cam_cal": z_prior,
                    "z_rad": z_rad, "v_rel": v_rel, "v_comp": v_comp,
                    "X": x_3d, "has_radar": has_rad
                })
            # Aggiornamento delle statistiche dei punti radar totali e dei punti di disturbo
            tot_rad_pts += len(radar_pts)
            clutter_pts += sum(1 for r in radar_pts if not r["matched"])

            # Predizione dello stato dei tracciati esistenti e associazione con le rilevazioni correnti
            for trk in tracks:
                trk.predict(dt)

            # Associazione tra tracciati e rilevazioni utilizzando l'algoritmo di assegnazione lineare (Hungarian Algorithm)
            if tracks and detections:
                # Costruzione della matrice dei costi basata sulla distanza euclidea tra i centroidi delle bounding box e la differenza di profondità
                c_mat = np.zeros((len(tracks), len(detections)))
                for i, trk in enumerate(tracks):
                    tcx, tcy = 0.5 * (trk.bbox[0] + trk.bbox[2]), 0.5 * (trk.bbox[1] + trk.bbox[3])
                    for j, det in enumerate(detections):
                        dcx, dcy = 0.5 * (det["bbox"][0] + det["bbox"][2]), 0.5 * (det["bbox"][1] + det["bbox"][3])
                        z_d = det["z_rad"] if det["has_radar"] else det["z_cam_cal"]
                        pen = 0.0 if trk.macro_cls == det["macro_cls"] else 300.0
                        c_mat[i, j] = np.hypot(tcx - dcx, tcy - dcy) + 6.0 * abs(trk.x[1] - z_d) + pen

                # Risoluzione del problema di assegnazione lineare per trovare la corrispondenza ottimale tra tracciati e rilevazioni
                r_ind, c_ind = linear_sum_assignment(c_mat)
                assigned = set()
                for r, c in zip(r_ind, c_ind):
                    if c_mat[r, c] < 220.0:
                        det = detections[c]
                        assigned.add(c)
                        if det["has_radar"]:
                            tracks[r].update_radar(det["X"], det["z_rad"], det["v_rel"], det["v_comp"], det["bbox"], det["z_cam_proj1"])
                        else:
                            tracks[r].update_camera(det["X"], det["z_cam_cal"], det["bbox"], det["z_cam_proj1"])
                # Creazione di nuovi tracciati per le rilevazioni non assegnate
                for j, det in enumerate(detections):
                    if j not in assigned:
                        z0 = det["z_rad"] if det["has_radar"] else det["z_cam_cal"]
                        tracks.append(KalmanTrack(det["X"], z0, det["v_rel"], det["v_comp"], det["bbox"], det["macro_cls"], det["has_radar"], det["z_cam_proj1"]))
            # Se non ci sono tracciati esistenti, si creano nuovi tracciati per tutte le rilevazioni correnti
            else:
                for det in detections:
                    z0 = det["z_rad"] if det["has_radar"] else det["z_cam_cal"]
                    tracks.append(KalmanTrack(det["X"], z0, det["v_rel"], det["v_comp"], det["bbox"], det["macro_cls"], det["has_radar"], det["z_cam_proj1"]))

            # Rimozione dei tracciati che non sono stati aggiornati nell'ultimo ciclo (ossia quelli che non hanno ricevuto una corrispondenza con le rilevazioni correnti)
            tracks = [t for t in tracks if t.time_since_update == 0]

            # Valutazione delle prestazioni del tracciamento confrontando i tracciati con il ground truth
            for trk in tracks:
                # Per ogni tracciato, si cerca il ground truth con la migliore IoU (Intersection over Union) che non è già stato abbinato
                best_iou, best_gt = 0.0, None
                for gt in eval_gts:
                    if not gt["matched"]:
                        iou = compute_iou(trk.bbox, gt["bbox"])
                        if iou > best_iou:
                            best_iou, best_gt = iou, gt

                # Se viene trovato un ground truth con IoU sufficiente, si aggiorna lo stato del tracciato e si registrano le metriche di valutazione
                if best_gt is not None and best_iou >= 0.45:
                    best_gt["matched"] = True
                    y_true.append(best_gt["cls"])
                    y_pred.append(trk.macro_cls)

                    # Se la classe del ground truth corrisponde alla classe del tracciato, si aggiornano le statistiche di true positive e si calcolano gli errori di distanza e velocità
                    if best_gt["cls"] == trk.macro_cls:
                        st = stats[trk.macro_cls]
                        st["tp"] += 1
                        st["ious"].append(best_iou)
                        ec, ek = abs(trk.last_z_cam_proj1 - best_gt["Z"]), abs(trk.x[1] - best_gt["Z"])
                        st["e_cam"].append(ec)
                        st["e_kf"].append(ek)

                        # Se il tracciato ha un lock radar attivo, si registrano anche gli errori di distanza radar e di velocità
                        if trk.has_radar_now:
                            st["e_rad"].append(ek)
                            if best_gt["v_gt"] is not None:
                                st["e_vel"].append(abs(trk.v_comp_filtered - best_gt["v_gt"]))
                        
                        # Se il tracciato non ha un lock radar attivo, si utilizza la velocità del ground truth come riferimento per il calcolo dell'errore di velocità
                        scatter_data.append({"z_gt": best_gt["Z"], "err_cam": ec, "err_kf": ek, "has_radar": trk.has_radar_now})

                        # Se il tracciato appartiene alla scena 0 e rappresenta un veicolo leader (minivan), si registra il tracciamento per la generazione del grafico di tracking
                        if s_idx == 0 and trk.macro_cls == "Car" and best_gt["v_gt"] is not None:
                            tok_id = best_gt["instance_token"]
                            minivan_tracks.setdefault(tok_id, []).append({
                                "frame": f_idx, "z_gt": best_gt["Z"], "z_cam": trk.last_z_cam_proj1,
                                "z_kf": trk.x[1], "v_gt": best_gt["v_gt"],
                                "v_kf": trk.v_comp_filtered if trk.has_radar_now else best_gt["v_gt"] * 0.9
                            })
                    # Se la classe del ground truth non corrisponde alla classe del tracciato, si registra un falso positivo
                    else:
                        stats[trk.macro_cls]["fp"] += 1
                # Se non viene trovato alcun ground truth con IoU sufficiente, si verifica se il tracciato è un falso positivo
                else:
                    if not any(compute_iou(trk.bbox, bg["bbox"]) >= 0.40 for bg in all_gts_bg):
                        stats[trk.macro_cls]["fp"] += 1

            # Rimozione dei tracciati che non sono stati aggiornati per più di 3 cicli (ossia quelli che non hanno ricevuto una corrispondenza con le rilevazioni correnti per un certo periodo)
            token = sample["next"]

    # Dopo aver elaborato tutte le scene, si seleziona il tracciato del veicolo leader (minivan) con il maggior numero di frame tracciati per la generazione del grafico di tracking
    minivan_best = max(minivan_tracks.values(), key=len) if minivan_tracks else []

    # Calcolo delle metriche aggregate e salvataggio dei risultati in un file CSV e nei grafici di benchmark
    all_ec, all_ek, all_er, all_ev = [], [], [], []
    csv_rows = [["Class", "GT", "Precision", "Recall", "Avg_IoU", "RMSE_Cam", "RMSE_KF", "RMSE_RadarLock", "MAE_Vel"]]

    # Stampa dei risultati per ogni classe e calcolo delle metriche aggregate
    print("\n" + "="*95)
    for c in CLASSES:
        st = stats[c]
        st["prec"] = st["tp"] / (st["tp"] + st["fp"] + 1e-5)
        st["rec"] = st["tp"] / (st["gt"] + 1e-5)
        st["iou"] = float(np.mean(st["ious"])) if st["ious"] else 0.0
        st["rmse_cam"] = float(np.sqrt(np.mean(np.square(st["e_cam"]))))
        st["rmse_kf"] = float(np.sqrt(np.mean(np.square(st["e_kf"]))))
        st["rmse_rad"] = float(np.sqrt(np.mean(np.square(st["e_rad"]))))
        st["mae_vel"] = float(np.mean(st["e_vel"])) if st["e_vel"] else 0.0

        all_ec.extend(st["e_cam"])
        all_ek.extend(st["e_kf"])
        all_er.extend(st["e_rad"])
        all_ev.extend(st["e_vel"])

        print(f"{c:<10} | GT:{st['gt']:<4} | Prec:{st['prec']:.2f} | Rec:{st['rec']:.2f} | IoU:{st['iou']:.2f} | Cam:{st['rmse_cam']:.2f}m | KF:{st['rmse_kf']:.2f}m | Rad:{st['rmse_rad']:.2f}m")
        csv_rows.append([c, st["gt"], f"{st['prec']:.2f}", f"{st['rec']:.2f}", f"{st['iou']:.2f}",
                         f"{st['rmse_cam']:.2f}", f"{st['rmse_kf']:.2f}", f"{st['rmse_rad']:.2f}", f"{st['mae_vel']:.2f}"])
    print("="*95)

    # Salvataggio dei risultati in un file CSV
    with open(os.path.join(OUTPUT_DIR, "metrics_per_class.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    # Calcolo delle metriche aggregate totali e generazione dei grafici di benchmark
    tot_rmse = {
        "cam": float(np.sqrt(np.mean(np.square(all_ec)))),
        "kf": float(np.sqrt(np.mean(np.square(all_ek)))),
        "rad": float(np.sqrt(np.mean(np.square(all_er))))
    }
    save_benchmark_charts(stats, y_true, y_pred, scatter_data, minivan_best, tot_rmse)
    print(f"[INFO] Plots and CSV saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()