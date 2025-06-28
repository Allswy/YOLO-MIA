import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from torchvision.ops import box_iou

V8N_MODEL_PATH = '/root/runs/detect/train2/weights/best.pt'

V8L_MODEL_PATH = '/root/base_v8l/yolov8l_voc_mia/weights/best.pt'
MEMBER_PATH = '/root/autodl-tmp/datasets/train_files.txt'
NON_MEMBER_PATH = '/root/autodl-tmp/datasets/non_member_samples.txt'
CSV_OUTPUT_PATH = 'mia_features1.csv'
IOU_THRESHOLD = 0.5
MAX_MATCHED = 10 

model_n = YOLO(V8N_MODEL_PATH)
model_l = YOLO(V8L_MODEL_PATH)

def compute_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compute_area(box):
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))

def extract_image_features(img_path, label):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图像：{img_path}")
        return None
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width

    result_n = model_n(img_path, verbose=False)[0]
    result_l = model_l(img_path, verbose=False)[0]

    boxes_n = result_n.boxes.xyxy.cpu()
    conf_n = result_n.boxes.conf.cpu()
    cls_n = result_n.boxes.cls.cpu()

    boxes_l = result_l.boxes.xyxy.cpu()
    conf_l = result_l.boxes.conf.cpu()
    cls_l = result_l.boxes.cls.cpu()

    iou_matrix = box_iou(boxes_n, boxes_l)
    matched_n = set()
    matched_l = set()
    per_box_features = []

    for i in range(len(boxes_n)):
        for j in range(len(boxes_l)):
            iou = iou_matrix[i, j]
            if iou > IOU_THRESHOLD and cls_n[i] == cls_l[j]:
                matched_n.add(i)
                matched_l.add(j)

                center_n = compute_box_center(boxes_n[i])
                center_l = compute_box_center(boxes_l[j])
                center_offset = ((center_n[0] - center_l[0])**2 + (center_n[1] - center_l[1])**2)**0.5
                center_offset = center_offset.item()

                tx1, ty1, tx2, ty2 = boxes_n[i]
                px1, py1, px2, py2 = boxes_l[j]
                tmp_len = center_offset / min(((tx2 - tx1)**2 + (ty2 - ty1)**2)**0.5, ((px2 - px1)**2 + (py2 - py1)**2)**0.5)

                area_n = compute_area(boxes_n[i])
                area_l = compute_area(boxes_l[j])
                area_diff = abs(area_n - area_l).item()
                area_diff = area_diff / min(area_n.item(), area_l.item())

                conf_diff = abs(conf_n[i] - conf_l[j]).item()

                per_box_features.append((conf_diff, area_diff, center_offset))
                break

    while len(per_box_features) < MAX_MATCHED:
        per_box_features.append((0.0, 0.0, 0.0))
    per_box_features = per_box_features[:MAX_MATCHED]

    missing_idxs = [j for j in range(len(boxes_l)) if j not in matched_l]
    missing_count = len(missing_idxs)
    missing_area = sum([compute_area(boxes_l[i]).item() for i in missing_idxs])
    total_l_area = sum([compute_area(box).item() for box in boxes_l])
    missing_area_rate = missing_area / total_l_area if total_l_area > 0 else 0.0

    missing_confs = [conf_l[j].item() for j in missing_idxs]
    missing_conf_avg = sum(missing_confs) / missing_count if missing_count else 0.0


    false_idxs = [i for i in range(len(boxes_n)) if i not in matched_n]
    false_area = sum([compute_area(boxes_n[i]).item() for i in false_idxs])

    total_l_area = sum([compute_area(box).item() for box in boxes_l])
    false_area_rate = false_area / total_l_area if total_l_area > 0 else 0.0

    row = {
        "img": os.path.basename(img_path),
        "v8n_missing_count": missing_count,
        "v8l_missing_conf_avg": missing_conf_avg,
        "missing_area_rate": missing_area_rate,
        "false_area_rate": false_area_rate,
        "label": label
    }

    valid_feats = [feat for feat in per_box_features if any(feat)]
    if valid_feats:
        row["mean_conf_diff"] = sum(f[0] for f in valid_feats) / len(valid_feats)
        row["mean_area_diff"] = sum(f[1] for f in valid_feats) / len(valid_feats)
        row["mean_center_offset"] = sum(f[2] for f in valid_feats) / len(valid_feats)
    else:
        row["mean_conf_diff"] = 0.0
        row["mean_area_diff"] = 0.0
        row["mean_center_offset"] = 0.0

    return row

def process_txt_list(txt_path, label):
    features = []
    with open(txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]
    for path in image_paths:
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在，已跳过: {path}")
            continue
        row = extract_image_features(path, label)
        features.append(row)
    return features

member_features = process_txt_list(MEMBER_PATH, label=1)
non_member_features = process_txt_list(NON_MEMBER_PATH, label=0)

all_features = member_features + non_member_features
df = pd.DataFrame(all_features)
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"✅ 已完成，共提取 {len(df)} 张图像的特征，保存至 {CSV_OUTPUT_PATH}")
