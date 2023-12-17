import os
import shutil

from tqdm import tqdm

k400_videos_path = "../Kinetics400/videos_train/"
# mkdir 'videos'
if not os.path.exists("videos"):
    os.makedirs("videos")

file_list = [
    open("kinetics-100-replaced/train.txt", "r"),
    open("kinetics-100-replaced/val.txt", "r"),
    open("kinetics-100-replaced/test.txt", "r"),
]

print("total 10k videos is being copied, please wait...")

for file in file_list:
    for i, line in tqdm(enumerate(file)):
        # split line
        filename = line.split()[0]
        # copy k400_videos_path+filename to 'videos'
        src_path = k400_videos_path + filename
        dst_path = "videos/" + filename
        if os.path.exists(dst_path):
            print("duplicate file: " + filename)
        elif os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print("no file: " + filename)
    file.close()
