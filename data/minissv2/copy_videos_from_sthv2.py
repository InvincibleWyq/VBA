import os
import shutil

from tqdm import tqdm

sthv2_videos_path = "../sthv2/videos/"
# mkdir 'videos'
if not os.path.exists("videos"):
    os.makedirs("videos")
if not os.path.exists("videos/full"):
    os.makedirs("videos/full")
if not os.path.exists("videos/small"):
    os.makedirs("videos/small")

file_list_small = [
    open("minissv2_small/train.txt", "r"),
    open("minissv2_small/val.txt", "r"),
    open("minissv2_small/test.txt", "r"),
]
file_list_full = [
    open("minissv2_full/train.txt", "r"),
    open("minissv2_full/val.txt", "r"),
    open("minissv2_full/test.txt", "r"),
]

print("copying, please wait...")

for file in file_list_small:
    for i, line in tqdm(enumerate(file)):
        # split line
        filename = line.split()[0]
        # copy videos
        src_path = sthv2_videos_path + filename
        dst_path = "videos/small/" + filename
        if os.path.exists(dst_path):
            print("duplicate file: " + filename)
        elif os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print("no file: " + filename)
    file.close()

for file in file_list_full:
    for i, line in tqdm(enumerate(file)):
        # split line
        filename = line.split()[0]
        # copy videos
        src_path = sthv2_videos_path + filename
        dst_path = "videos/full/" + filename
        if os.path.exists(dst_path):
            print("duplicate file: " + filename)
        elif os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print("no file: " + filename)
    file.close()
