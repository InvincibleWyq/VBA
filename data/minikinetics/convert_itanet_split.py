import re

my_ann_train = open('kinetics-100-replaced/train.txt', 'w')
my_ann_val = open('kinetics-100-replaced/val.txt', 'w')
my_ann_test = open('kinetics-100-replaced/test.txt', 'w')
ITA_train = open('kinetics-100/train.list', 'r')
ITA_val = open('kinetics-100/val.list', 'r')
ITA_test = open('kinetics-100/test.list', 'r')
OriginAnn = open('kinetics400_train_list_videos.txt', 'r')

found = False

for i, line in enumerate(ITA_train):
    found = False
    # split each line by '/'
    out = re.split(' |/', line)
    label = out[-2]
    filename = out[-1][:11]
    # search OriginAnn file, see if there is filename
    OriginAnn.seek(0)
    for ori_line in OriginAnn:
        if filename in ori_line:
            # write ori_line to my_ann_test
            my_ann_train.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + line)

for i, line in enumerate(ITA_val):
    # split each line by '/'
    found = False
    out = re.split(' |/', line)
    label = out[-2]
    filename = out[-1][:11]
    # search OriginAnn file, see if there is filename
    OriginAnn.seek(0)
    for ori_line in OriginAnn:
        if filename in ori_line:
            # write ori_line to my_ann_test
            my_ann_val.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + line)

for i, line in enumerate(ITA_test):
    # split each line by '/'
    found = False
    out = re.split(' |/', line)
    label = out[-2]
    filename = out[-1][:11]
    # search OriginAnn file, see if there is filename
    OriginAnn.seek(0)
    for ori_line in OriginAnn:
        if filename in ori_line:
            # write ori_line to my_ann_test
            my_ann_test.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + line)

OriginAnn.close()
ITA_train.close()
ITA_val.close()
ITA_test.close()
my_ann_train.close()
my_ann_val.close()
my_ann_test.close()
