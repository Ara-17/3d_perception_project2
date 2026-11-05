import os
import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes

# Importazioni di funzioni e costanti dal modulo tracker
from tracker import (
    YOLO_CLASSES, BUMPER_OFFSET, HEIGHT_PRIOR,
    map_yolo_class, extract_roi_median, validate_radar_points,
    KalmanTrack, load_frame_and_sensors
)

# Costanti di configurazione
DATAROOT = r"data/v1.0-mini"
SCENE_INDEX = 1
CONFIDENCE_THRESHOLD = 0.40

# Funzione per disegnare il pannello BEV (Bird's Eye View)
def draw_bev_panel(height, tracks, gts, radar_pts, clutter_pct):
    # Creazione di un'immagine BEV vuota con sfondo grigio
    bev_w = 500
    bev = np.full((height, bev_w, 3), 24, dtype=np.uint8)
    ego_u, ego_v = bev_w // 2, height - 45
    scale_z, scale_x = (height - 110) / 55.0, (bev_w - 60) / 40.0

    # Funzione interna per convertire le coordinate 3D in coordinate BEV
    def to_bev(X, Z):
        return int(ego_u + X * scale_x), int(ego_v - Z * scale_z)

    # Disegno delle linee di distanza sul pannello BEV
    for d in [10, 20, 30, 40, 50]:
        _, vy = to_bev(0, d)
        cv2.line(bev, (30, vy), (bev_w - 30, vy), (55, 55, 55), 1)
        cv2.putText(bev, f"{d}m", (5, vy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    # Disegno del veicolo ego sul pannello BEV
    cv2.rectangle(bev, (ego_u - 10, ego_v - 15), (ego_u + 10, ego_v + 15), (255, 200, 0), -1)
    cv2.putText(bev, "EGO", (ego_u - 16, ego_v + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    # Disegno dei punti radar non corrispondenti sul pannello BEV
    for rp in radar_pts:
        if not rp["matched"]:
            ru, rv = to_bev(rp["X"], rp["Z"])
            if 0 <= ru < bev_w and 0 <= rv < height:
                cv2.circle(bev, (ru, rv), 2, (85, 85, 85), -1)

    # Disegno dei ground truth sul pannello BEV
    for gt in gts:
        gu, gv = to_bev(gt["X"], gt["Z"])
        if 0 <= gu < bev_w and 0 <= gv < height:
            cv2.rectangle(bev, (gu - 7, gv - 7), (gu + 7, gv + 7), (255, 255, 255), 1)

    # Disegno dei tracciamenti sul pannello BEV
    for trk in tracks:
        tu, tv = to_bev(trk.x[0], trk.x[1])
        if 0 <= tu < bev_w and 0 <= tv < height:
            col = (0, 255, 0) if trk.has_radar_now else (0, 215, 255)
            cv2.circle(bev, (tu, tv), 6, col, -1)
            v_end = int(tv - trk.v_comp_filtered * scale_z * 1.2)
            cv2.arrowedLine(bev, (tu, tv), (tu, v_end), col, 2, tipLength=0.3)
            cv2.putText(bev, f"ID:{trk.id}", (tu + 8, tv + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # Aggiunta di testo informativo sul pannello BEV
    cv2.putText(bev, "TOP-DOWN BEV (RADAR + KALMAN)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bev, f"Clutter Rejected: {clutter_pct:.1f}%", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)
    cv2.putText(bev, "[White]: Ground Truth | [Green]: Radar+KF | [Yellow]: Cam+KF", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
    return bev

# Funzione principale per eseguire la demo del video
def main():
    # Configurazione del dispositivo e caricamento dei modelli YOLO e MiDaS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO("yolov8m.pt")
    midas_cache = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    if os.path.exists(midas_cache):
        midas = torch.hub.load(midas_cache, "DPT_Hybrid", source="local").to(device).eval()
        transform = torch.hub.load(midas_cache, "transforms", source="local").dpt_transform
    else:
        midas = torch.hub.load("intel-isl/MiDaS:master", "DPT_Hybrid").to(device).eval()
        transform = torch.hub.load("intel-isl/MiDaS:master", "transforms").dpt_transform

    # Inizializzazione del dataset NuScenes e selezione della scena
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    scene = nusc.scene[SCENE_INDEX]
    sample_token = scene["first_sample_token"]

    # Creazione della cartella di output e definizione del percorso del video
    os.makedirs("output", exist_ok=True)
    video_path = f"output/video_scene_{SCENE_INDEX}_{scene['name']}.mp4"
    out_video = None

    # Inizializzazione delle variabili per i tracciamenti, la scala di calibrazione e il timestamp precedente
    tracks = []
    scale_cal = 1.0
    prev_ts = None

    # Stampa delle informazioni sulla scena e sul percorso del video
    print(f"[INFO] Riproduzione Scena {SCENE_INDEX} ({scene['name']}): {scene['description']}")
    print(f"[INFO] Il video verrà salvato in: {video_path}")

    # Ciclo principale per elaborare i campioni della scena
    while sample_token != "":
        # Recupero del campione corrente e calcolo del delta time
        sample = nusc.get("sample", sample_token)
        dt = 0.5 if prev_ts is None else (sample["timestamp"] - prev_ts) / 1e6
        prev_ts = sample["timestamp"]

        # Caricamento dell'immagine, dei punti radar, dei ground truth e della matrice intrinseca della camera
        img, radar_pts, eval_gts, _, cam_k = load_frame_and_sensors(nusc, sample)
        h, w = img.shape[:2]
        fx, fy, cx = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2]

        # Calcolo della mappa di profondità utilizzando il modello MiDaS
        with torch.no_grad():
            pred = midas(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device))
            depth_map = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # Normalizzazione della mappa di profondità e clipping dei valori estremi
        d_min, d_max = np.percentile(depth_map, 2), np.percentile(depth_map, 98)
        depth_norm = np.clip((depth_map - d_min) / (d_max - d_min + 1e-5), 0.01, 1.0)

        # Esecuzione della rilevazione degli oggetti utilizzando il modello YOLO e ordinamento delle bounding box per area
        yolo_boxes = sorted(
            yolo(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0].boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
            reverse=True
        )

        # Elaborazione delle rilevazioni YOLO e associazione con i punti radar
        detections = []
        for box in yolo_boxes:
            # Filtraggio delle classi YOLO non rilevanti e mappatura della classe macro
            cls_id = int(box.cls[0])
            if cls_id not in YOLO_CLASSES:
                continue
            macro_cls = map_yolo_class(YOLO_CLASSES[cls_id])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Estrazione della profondità mediana dalla mappa di profondità normalizzata all'interno della ROI della bounding box
            disp_med = extract_roi_median(depth_norm, (x1, y1, x2, y2))
            z_midas = 6.2 / (disp_med + 0.05)
            z_pinhole = (fy * HEIGHT_PRIOR.get(macro_cls, 1.60)) / max(y2 - y1, 12)
            w_m = 0.65 if z_pinhole < 18.0 else 0.30
            z_prior = scale_cal * (w_m * z_midas + (1.0 - w_m) * z_pinhole)

            # Validazione dei punti radar all'interno della bounding box e calcolo delle statistiche medie
            cluster = validate_radar_points(radar_pts, (x1, y1, x2, y2), z_prior, macro_cls)
            if cluster:
                for rp in cluster:
                    rp["matched"] = True
                z_rad = float(np.median([r["Z"] for r in cluster])) + BUMPER_OFFSET.get(macro_cls, 1.0)
                v_rel = float(np.median([r["v_rel"] for r in cluster]))
                v_comp = float(np.median([r["v_comp"] for r in cluster]))
                x_3d = float(np.median([r["X"] for r in cluster]))
                has_radar = True
            else:
                z_rad, v_rel, v_comp = None, 0.0, 0.0
                x_3d = (0.5 * (x1 + x2) - cx) * z_prior / fx
                has_radar = False

            # Aggiunta della rilevazione alla lista delle rilevazioni con le informazioni calcolate
            detections.append({
                "bbox": [x1, y1, x2, y2], "macro_cls": macro_cls,
                "z_cam_proj1": z_midas, "z_cam_cal": z_prior,
                "z_rad": z_rad, "v_rel": v_rel, "v_comp": v_comp,
                "X": x_3d, "has_radar": has_radar
            })

        # Aggiornamento dei tracciamenti esistenti utilizzando il filtro di Kalman e calcolo della matrice dei costi per l'associazione
        for trk in tracks:
            trk.predict(dt)

        # Associazione delle rilevazioni ai tracciamenti esistenti utilizzando l'algoritmo di assegnazione lineare
        if tracks and detections:
            # Creazione della matrice dei costi basata sulla distanza euclidea, la differenza di profondità e la penalità per classi diverse
            cost_mat = np.zeros((len(tracks), len(detections)))
            for i, trk in enumerate(tracks):
                tcx, tcy = 0.5 * (trk.bbox[0] + trk.bbox[2]), 0.5 * (trk.bbox[1] + trk.bbox[3])
                for j, det in enumerate(detections):
                    dcx, dcy = 0.5 * (det["bbox"][0] + det["bbox"][2]), 0.5 * (det["bbox"][1] + det["bbox"][3])
                    z_d = det["z_rad"] if det["has_radar"] else det["z_cam_cal"]
                    pen = 0.0 if trk.macro_cls == det["macro_cls"] else 300.0
                    cost_mat[i, j] = np.hypot(tcx - dcx, tcy - dcy) + 6.0 * abs(trk.x[1] - z_d) + pen

            # Risoluzione del problema di assegnazione lineare per trovare le corrispondenze ottimali tra tracciamenti e rilevazioni
            r_ind, c_ind = linear_sum_assignment(cost_mat)
            assigned = set()
            for r, c in zip(r_ind, c_ind):
                if cost_mat[r, c] < 220.0:
                    det = detections[c]
                    assigned.add(c)
                    if det["has_radar"]:
                        tracks[r].update_radar(det["X"], det["z_rad"], det["v_rel"], det["v_comp"], det["bbox"], det["z_cam_proj1"])
                    else:
                        tracks[r].update_camera(det["X"], det["z_cam_cal"], det["bbox"], det["z_cam_proj1"])
            
            # Creazione di nuovi tracciamenti per le rilevazioni non assegnate
            for j, det in enumerate(detections):
                if j not in assigned:
                    z0 = det["z_rad"] if det["has_radar"] else det["z_cam_cal"]
                    tracks.append(KalmanTrack(det["X"], z0, det["v_rel"], det["v_comp"], det["bbox"], det["macro_cls"], det["has_radar"], det["z_cam_proj1"]))
        # Creazione di nuovi tracciamenti se non ci sono tracciamenti esistenti
        else:
            for det in detections:
                z0 = det["z_rad"] if det["has_radar"] else det["z_cam_cal"]
                tracks.append(KalmanTrack(det["X"], z0, det["v_rel"], det["v_comp"], det["bbox"], det["macro_cls"], det["has_radar"], det["z_cam_proj1"]))

        # Rimozione dei tracciamenti che non sono stati aggiornati
        tracks = [t for t in tracks if t.time_since_update == 0]

        # Disegna punti radar e bounding box
        for rp in radar_pts:
            col = (255, 255, 0) if rp["matched"] else (110, 110, 110)
            rad = 5 if rp["matched"] else 2
            cv2.circle(img, (int(rp["u"]), int(rp["v"])), rad, col, -1)

        # Disegna tracciamenti e calcola il tempo di collisione (TTC) per ciascun tracciamento
        for trk in tracks:
            # Calcolo del tempo di collisione (TTC) basato sulla profondità e sulla velocità relativa
            x1, y1, x2, y2 = trk.bbox
            z_f, v_r = trk.x[1], trk.x[3]
            ttc = (-z_f / v_r) if v_r < -0.5 else 99.9
            is_vru = trk.macro_cls in ["Person", "Bike/Moto"]

            # Determinazione del colore della bounding box in base al TTC e alla profondità
            if ttc < 4.0 or (is_vru and z_f < 10.0) or (not is_vru and z_f < 7.0):
                col = (0, 0, 255)
            elif ttc < 7.0 or is_vru:
                col = (0, 215, 255)
            else:
                col = (0, 230, 0)

            # Creazione della stringa di testo per la visualizzazione delle informazioni del tracciamento
            mode_str = "FUSED" if trk.has_radar_now else "CAM"
            ttc_str = f"TTC:{ttc:.1f}s" if ttc < 20.0 else "SAFE"
            lbl = f"ID:{trk.id} {trk.macro_cls} | {z_f:.1f}m | {trk.v_comp_filtered:+.1f}m/s | {ttc_str} [{mode_str}]"
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, lbl, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)

        # Calcolo della percentuale di clutter (punti radar non corrispondenti) e combinazione dell'immagine con il pannello BEV
        clutter_pct = sum(1 for r in radar_pts if not r["matched"]) / max(len(radar_pts), 1) * 100.0
        combined = np.hstack((img, draw_bev_panel(h, tracks, eval_gts, radar_pts, clutter_pct)))
        disp = cv2.resize(combined, (int(combined.shape[1] * 0.75), int(combined.shape[0] * 0.75)))

        # Salvataggio del frame nel video di output e visualizzazione della finestra
        if out_video is None:
            out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (disp.shape[1], disp.shape[0]))
        out_video.write(disp)

        # Visualizzazione della finestra con il frame combinato e gestione dell'uscita premendo 'q'
        cv2.imshow(f"Scene {SCENE_INDEX} Demo (Press 'q' to exit)", disp)
        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

        # Passaggio al campione successivo nella scena    
        sample_token = sample["next"]

    # Rilascio delle risorse e chiusura delle finestre OpenCV
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()