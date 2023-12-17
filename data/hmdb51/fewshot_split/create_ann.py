"""
search all the files in the folder and create .txt annotation file
with the following format:
brush_hair/xxx.avi 0
brush_hair/xxy.avi 0
cartwheel/123.avi 1
...

numbers is alphabetically sorted
"""

import glob
import os

train_labels = [
    'brush_hair', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive',
    'draw_sword', 'dribble', 'drink', 'fall_floor', 'flic_flac', 'handstand',
    'hug', 'jump', 'kiss', 'pullup', 'punch', 'push', 'ride_bike',
    'ride_horse', 'shake_hands', 'shoot_bow', 'situp', 'stand', 'sword',
    'sword_exercise', 'throw', 'turn', 'walk', 'wave'
]
val_labels = [
    'cartwheel', 'eat', 'golf', 'hit', 'laugh', 'shoot_ball', 'shoot_gun',
    'smile', 'somersault', 'swing_baseball'
]
test_labels = [
    'fencing', 'kick', 'kick_ball', 'pick', 'pour', 'pushup', 'run', 'sit',
    'smoke', 'talk'
]

# use for loop to list out all directories in this folder alphabetically
for i, folder in enumerate(sorted(os.listdir('/home/wyq/Downloads/hmdb51/'))):
    # use glob to list out all files in the folder
    for file in glob.glob(os.path.join(folder, '*.avi')):
        # write the file name and label to the txt file
        with open('/home/wyq/Downloads/hmdb51/ann.txt', 'a') as f:
            f.write(file + ' ' + str(i) + '\n')
        if folder in train_labels:
            with open('/home/wyq/Downloads/hmdb51/train.txt', 'a') as f:
                f.write(file + ' ' + str(i) + '\n')
        elif folder in val_labels:
            with open('/home/wyq/Downloads/hmdb51/val.txt', 'a') as f:
                f.write(file + ' ' + str(i) + '\n')
        elif folder in test_labels:
            with open('/home/wyq/Downloads/hmdb51/test.txt', 'a') as f:
                f.write(file + ' ' + str(i) + '\n')
        else:
            print('error\n' + folder)
