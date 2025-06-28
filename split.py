import os
import random

def collect_all_train_image_paths(image_dirs, label_root="/root/autodl-tmp/datasets/datasets/VOC/labels"):

    img_paths = []
    for img_dir in image_dirs:
        if not os.path.exists(img_dir):
            print(f"目录不存在: {img_dir}")
            continue
        for fname in os.listdir(img_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(img_dir, fname)
                # 构造对应标签路径，替换 images -> labels，扩展名改为 txt
                rel_path = os.path.relpath(full_path, "/root/autodl-tmp/datasets/datasets/VOC/images")
                label_path = os.path.join(label_root, os.path.splitext(rel_path)[0] + ".txt")

                if os.path.exists(label_path):
                    img_paths.append(full_path)
                else:
                    print(f"跳过无标签图片: {full_path}")
    return img_paths

def split_member_nonmember(paths):
    random.shuffle(paths)
    half = len(paths) // 10
    member_samples = paths[:half]
    non_member_samples = paths[9 * half:]
    return member_samples, non_member_samples

def split_train_val(member_samples, train_ratio=0.8):
    random.shuffle(member_samples)
    split_index = int(len(member_samples) * train_ratio)
    train_samples = member_samples[:split_index]
    val_samples = member_samples[split_index:]
    return train_samples, val_samples

def save_list_to_file(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write(item + '\n')
    print(f"已保存 {len(lst)} 条路径到 {filename}")



def main():

    image_dirs = [
        "/root/autodl-tmp/datasets/datasets/VOC/images/train2007",
        "/root/autodl-tmp/datasets/datasets/VOC/images/train2012",
        "/root/autodl-tmp/datasets/datasets/VOC/images/val2007",
        "/root/autodl-tmp/datasets/datasets/VOC/images/val2012"
    ]

    all_paths = collect_all_train_image_paths(image_dirs)
    print(f"收集到总共 {len(all_paths)} 张带标签的训练图片")

    member_samples, non_member_samples = split_member_nonmember(all_paths)
    save_list_to_file(member_samples, "member_samples.txt")
    save_list_to_file(non_member_samples, "non_member_samples.txt") 

    train_samples, val_samples = split_train_val(member_samples, train_ratio=0.8)
    save_list_to_file(train_samples, "train_files.txt")
    #save_list_to_file(val_samples, "val_files.txt")
    #merge_image_paths('train_files.txt', 'non_member_samples.txt', 'merged_paths.txt')


if __name__ == "__main__":
    main()
