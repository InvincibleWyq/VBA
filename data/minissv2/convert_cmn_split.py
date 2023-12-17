from tqdm import tqdm

my_ann_train = open('minissv2_small/train.txt', 'w')
my_ann_val = open('minissv2_small/val.txt', 'w')
my_ann_test = open('minissv2_small/test.txt', 'w')

CMN_train = open('cmn_split/smsm-100/train.list', 'r')
CMN_val = open('cmn_split/smsm-100/val.list', 'r')
CMN_test = open('cmn_split/smsm-100/test.list', 'r')

OriginTrain = open('sthv2_train_list_videos.txt', 'r')
OriginVal = open('sthv2_val_list_videos.txt', 'r')

found = False

print("total 10k ann is being converted, please wait...")

for i, line in tqdm(enumerate(CMN_train)):
    found = False
    # split line
    filename = line.split('/')[-1][:-1]
    # search OriginTrain file, see if there is filename
    OriginTrain.seek(0)
    for ori_line in OriginTrain:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_train.write(ori_line)
            found = True
            break
    if found:
        continue
    OriginVal.seek(0)
    for ori_line in OriginVal:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_train.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + filename)

for i, line in tqdm(enumerate(CMN_val)):
    found = False
    # split line
    filename = line.split('/')[-1][:-1]
    # search OriginTrain file, see if there is filename
    OriginTrain.seek(0)
    for ori_line in OriginTrain:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_val.write(ori_line)
            found = True
            break
    if found:
        continue
    OriginVal.seek(0)
    for ori_line in OriginVal:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_val.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + filename)

for i, line in tqdm(enumerate(CMN_test)):
    found = False
    # split line
    filename = line.split('/')[-1][:-1]
    # search OriginTrain file, see if there is filename
    OriginTrain.seek(0)
    for ori_line in OriginTrain:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_test.write(ori_line)
            found = True
            break
    if found:
        continue
    OriginVal.seek(0)
    for ori_line in OriginVal:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_test.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + filename)

OriginTrain.close()
OriginVal.close()
CMN_train.close()
CMN_val.close()
CMN_test.close()
my_ann_train.close()
my_ann_val.close()
my_ann_test.close()
