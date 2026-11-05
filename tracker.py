import os
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

# Classi COCO di YOLO usate nel progetto
YOLO_CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Offset tra paraurti (colpito dal radar) e centro 3D del veicolo (ground truth)
BUMPER_OFFSET = {"Car": 1.30, "Truck/Bus": 2.10, "Person": 0.15, "Bike/Moto": 0.35}

# Altezza media reale (metri) per la stima prospettica Pinhole sulle lunghe distanze
HEIGHT_PRIOR = {"Car": 1.58, "Truck/Bus": 2.85, "Person": 1.72, "Bike/Moto": 1.45}

# Funzioni di utilità per il tracciamento e la valutazione
def map_nusc_class(cat_name):
    if "car" in cat_name:
        return "Car"
    if any(k in cat_name for k in ["truck", "bus", "construction", "trailer"]):
        return "Truck/Bus"
    if "pedestrian" in cat_name:
        return "Person"
    if any(k in cat_name for k in ["bicycle", "motorcycle"]):
        return "Bike/Moto"
    return None

# Funzione per mappare le classi YOLO alle macro-classi utilizzate nel tracciamento
def map_yolo_class(label):
    if label == "Car":
        return "Car"
    if label in ["Truck", "Bus"]:
        return "Truck/Bus"
    if label == "Person":
        return "Person"
    if label in ["Bicycle", "Motorcycle"]:
        return "Bike/Moto"
    return None

# Funzione per calcolare l'Intersection over Union (IoU) tra due bounding box
def compute_iou(boxA, boxB):
    ix1, iy1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    ix2, iy2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-5)

# Funzione per estrarre la profondità mediana da una mappa di profondità all'interno di un bounding box
def extract_roi_median(depth_map, box, crop_ratio=0.50):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    cx1 = int(x1 + w * (1 - crop_ratio) / 2)
    cy1 = int(y1 + h * (1 - crop_ratio) / 2)
    cx2 = int(x2 - w * (1 - crop_ratio) / 2)
    cy2 = int(y2 - h * (1 - crop_ratio) / 2)
    roi = depth_map[max(0, cy1):max(1, cy2), max(0, cx1):max(1, cx2)]
    return float(np.median(roi)) if roi.size > 0 else 0.0

# Funzione per validare i punti radar all'interno di un bounding box e rispetto a una profondità stimata
def validate_radar_points(radar_pts, box, z_cam_prior, macro_cls):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    u_center = 0.5 * (x1 + x2)

    # Ritaglio interno per escludere asfalto sotto le ruote e sfondo ai bordi
    mx = 0.22 if macro_cls == "Person" else 0.15
    rx1, rx2 = x1 + int(mx * w), x2 - int(mx * w)
    ry1, ry2 = y1 + int(0.22 * h), y2 - int(0.06 * h)

    offset = BUMPER_OFFSET.get(macro_cls, 1.0)
    gate_tol = max(2.6, 0.22 * z_cam_prior)

    valid = []
    for rp in radar_pts:
        if not rp["matched"] and (rx1 <= rp["u"] <= rx2) and (ry1 <= rp["v"] <= ry2):
            if abs((rp["Z"] + offset) - z_cam_prior) <= gate_tol:
                valid.append(rp)

    if not valid:
        return []

    valid.sort(key=lambda p: p["Z"] + 0.015 * abs(p["u"] - u_center))
    z_front = valid[0]["Z"]
    return [p for p in valid if abs(p["Z"] - z_front) <= 1.2]

# Classe per rappresentare un tracciato Kalman con stato 4D e aggiornamenti basati su radar e camera
class KalmanTrack:
    _id_counter = 1

    # Inizializzazione del tracciato con posizione, velocità relativa, bounding box e altre informazioni
    def __init__(self, x_pos, z_pos, v_rel, v_comp, bbox, macro_cls, has_radar, z_cam_raw):
        self.id = KalmanTrack._id_counter
        KalmanTrack._id_counter += 1

        # Vettore di stato 4D: [X, Z, vx_rel, vz_rel]
        self.x = np.array([x_pos, z_pos, 0.0, v_rel], dtype=float)
        init_var = 0.25 if has_radar else 2.5
        self.P = np.diag([0.4, init_var, 1.5, 1.5])

        self.bbox = bbox
        self.macro_cls = macro_cls
        self.has_radar_now = has_radar
        self.v_comp_filtered = v_comp
        self.last_z_cam_proj1 = z_cam_raw
        self.time_since_update = 0
    
    # Funzione di predizione dello stato del tracciato usando il modello di transizione di stato
    def predict(self, dt=0.5):
        F = np.array([
            [1, 0, dt * 0.6, 0],
            [0, 1, 0, dt * 0.6],
            [0, 0, 0.80, 0],
            [0, 0, 0, 0.80]
        ], dtype=float)
        Q = np.diag([0.25, 0.35, 1.0, 1.0]) * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time_since_update += 1

    # Funzione di aggiornamento dello stato del tracciato usando le misurazioni radar
    def update_radar(self, x_meas, z_radar, v_rel, v_comp, bbox, z_cam_raw):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        R_radar = np.diag([0.12, 0.05, 0.15])

        z_vec = np.array([x_meas, z_radar, v_rel], dtype=float)
        y = z_vec - (H @ self.x)
        S = H @ self.P @ H.T + R_radar
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ H) @ self.P

        self.v_comp_filtered = 0.35 * self.v_comp_filtered + 0.65 * v_comp if self.has_radar_now else v_comp
        self.bbox = bbox
        self.has_radar_now = True
        self.last_z_cam_proj1 = z_cam_raw
        self.time_since_update = 0

    # Funzione di aggiornamento dello stato del tracciato usando le misurazioni camera
    def update_camera(self, x_meas, z_cam_cal, bbox, z_cam_raw):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R_cam = np.diag([0.5, 2.2])

        z_vec = np.array([x_meas, z_cam_cal], dtype=float)
        y = z_vec - (H @ self.x)
        S = H @ self.P @ H.T + R_cam
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ H) @ self.P

        self.bbox = bbox
        self.has_radar_now = False
        self.last_z_cam_proj1 = z_cam_raw
        self.time_since_update = 0

# Funzione per caricare i dati del frame e dei sensori (camera e radar) da un campione NuScenes
def load_frame_and_sensors(nusc, sample):
    cam_token = sample['data']['CAM_FRONT']
    radar_token = sample['data']['RADAR_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    radar_data = nusc.get('sample_data', radar_token)

    # Caricamento dei punti radar e calcolo delle velocità relative e componenti
    pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, radar_data['filename']))
    r_dist = np.sqrt(pc.points[0, :]**2 + pc.points[1, :]**2) + 1e-5
    v_rel = (pc.points[0, :] * pc.points[6, :] + pc.points[1, :] * pc.points[7, :]) / r_dist
    v_comp = (pc.points[0, :] * pc.points[8, :] + pc.points[1, :] * pc.points[9, :]) / r_dist

    # Allineamento 3D dal paraurti (Radar) al parabrezza (Camera)
    cs_rad = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_rad['rotation']).rotation_matrix)
    pc.translate(np.array(cs_rad['translation']))

    # Allineamento 3D dal radar al sistema di riferimento della camera
    pos_rad = nusc.get('ego_pose', radar_data['ego_pose_token'])
    pc.rotate(Quaternion(pos_rad['rotation']).rotation_matrix)
    pc.translate(np.array(pos_rad['translation']))

    # Allineamento 3D dal sistema di riferimento della camera al sistema di riferimento globale
    pos_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
    rot_ego = Quaternion(pos_cam['rotation']).rotation_matrix
    pc.translate(-np.array(pos_cam['translation']))
    pc.rotate(rot_ego.T)

    # Allineamento 3D dal sistema di riferimento globale al sistema di riferimento della camera
    cs_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    rot_cam = Quaternion(cs_cam['rotation']).rotation_matrix
    pc.translate(-np.array(cs_cam['translation']))
    pc.rotate(rot_cam.T)

    # Proiezione dei punti 3D radar nel piano dell'immagine della camera
    cam_k = np.array(cs_cam['camera_intrinsic'])
    pts_2d = view_points(pc.points[:3, :], cam_k, normalize=True)

    # Caricamento dell'immagine della camera e filtraggio dei punti radar proiettati all'interno dell'immagine
    img = cv2.imread(os.path.join(nusc.dataroot, cam_data['filename']))
    h, w = img.shape[:2]
    x_3d, z_3d = pc.points[0, :], pc.points[2, :]

    # Filtraggio dei punti radar proiettati all'interno dell'immagine e con profondità valida
    mask = (z_3d > 2.0) & (z_3d < 55.0) & (pts_2d[0, :] >= 0) & (pts_2d[0, :] < w) & (pts_2d[1, :] >= 0) & (pts_2d[1, :] < h)
    radar_pts = [{
        "u": float(pts_2d[0, i]), "v": float(pts_2d[1, i]),
        "X": float(x_3d[i]), "Z": float(z_3d[i]),
        "v_rel": float(v_rel[i]), "v_comp": float(v_comp[i]), "matched": False
    } for i in np.where(mask)[0]]

    # Caricamento delle annotazioni di ground truth per la camera e filtraggio in base alla visibilità e alla profondità
    _, boxes_gt, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    eval_gts, all_gts_bg = [], []

    # Iterazione sulle annotazioni di ground truth per la camera e filtraggio in base alla classe, alla profondità e alla dimensione del bounding box
    for box in boxes_gt:
        ann = nusc.get('sample_annotation', box.token)
        cls_name = map_nusc_class(box.name)
        z_gt, x_gt = float(box.center[2]), float(box.center[0])
        if cls_name is None or z_gt <= 1.5 or z_gt > 65.0:
            continue

        # Proiezione dei vertici del bounding box 3D nel piano dell'immagine della camera e calcolo del bounding box 2D
        c2d = view_points(box.corners(), cam_k, normalize=True)[:2, :]
        x1, y1 = int(np.clip(np.min(c2d[0]), 0, w - 1)), int(np.clip(np.min(c2d[1]), 0, h - 1))
        x2, y2 = int(np.clip(np.max(c2d[0]), 0, w - 1)), int(np.clip(np.max(c2d[1]), 0, h - 1))
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            continue

        # Calcolo della velocità del ground truth nel sistema di riferimento della camera
        vel_g = nusc.box_velocity(box.token)
        if not np.any(np.isnan(vel_g)):
            vel_c = rot_cam.T @ (rot_ego.T @ vel_g)
            v_gt = float((x_gt * vel_c[0] + z_gt * vel_c[2]) / (np.hypot(x_gt, z_gt) + 1e-5))
        else:
            v_gt = None

        # Creazione di un dizionario per rappresentare il ground truth e aggiunta alla lista di tutti i ground truth e a quelli da valutare
        item = {"instance_token": ann["instance_token"], "cls": cls_name, "bbox": [x1, y1, x2, y2],
                "X": x_gt, "Z": z_gt, "v_gt": v_gt, "matched": False}
        all_gts_bg.append(item)
        if int(ann['visibility_token']) >= 2 and z_gt <= 50.0 and (x2 - x1) >= 22 and (y2 - y1) >= 22:
            eval_gts.append(item)

    return img, radar_pts, eval_gts, all_gts_bg, cam_k