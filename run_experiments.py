import os
import shutil
import subprocess
from time import sleep

REPEAT_TIMES = 5
BASE_EXP_DIR = "experiments"
FEATURE_OUTPUT_DIR = "all_features"
YOLO_RUNS_DIR = "/root/runs/detect"

os.makedirs(BASE_EXP_DIR, exist_ok=True)
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

for i in range(1, REPEAT_TIMES + 1):
    print(f"\n=== Running Experiment {i} ===")


    existing_trains = set(os.listdir(YOLO_RUNS_DIR)) if os.path.exists(YOLO_RUNS_DIR) else set()

    subprocess.run(['python', '/root/autodl-tmp/datasets/split.py'])
    subprocess.run(['python', '/root/train.py'])

    sleep(2)  

    new_trains = set(os.listdir(YOLO_RUNS_DIR)) - existing_trains
    if not new_trains:
        print("⚠️ 未检测到新训练目录，请检查训练是否完成。")
        continue

    new_train_dir = sorted(new_trains)[-1]  
    full_train_path = os.path.join(YOLO_RUNS_DIR, new_train_dir)

    subprocess.run(['python', 'get_data.py'])

    exp_dir = os.path.join(BASE_EXP_DIR, f'exp_{i:02d}')
    os.makedirs(exp_dir, exist_ok=True)

    shutil.copy('/root/autodl-tmp/datasets/train_files.txt', os.path.join(exp_dir, 'train_files.txt'))
    shutil.copy('/root/autodl-tmp/datasets/non_member_samples.txt', os.path.join(exp_dir, 'non_number_samples.txt'))
    shutil.move(full_train_path, os.path.join(exp_dir, 'train_result'))

    shutil.move('mia_features.csv', os.path.join(FEATURE_OUTPUT_DIR, f'mia_features_{i:02d}.csv'))

    for leftover in os.listdir(YOLO_RUNS_DIR):
        path = os.path.join(YOLO_RUNS_DIR, leftover)
        if os.path.isdir(path):
            shutil.rmtree(path)

print("\n✅ 所有实验完成！训练目录已清理干净。")
