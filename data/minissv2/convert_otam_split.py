from tqdm import tqdm

my_ann_train = open('minissv2_full/train.txt', 'w')
my_ann_val = open('minissv2_full/val.txt', 'w')
my_ann_test = open('minissv2_full/test.txt', 'w')

OTAM_train_train = open('otam_split/somethingv2_meta_train_train.txt', 'r')
OTAM_train_val = open('otam_split/somethingv2_meta_train_val.txt', 'r')
OTAM_val = open('otam_split/somethingv2_meta_val.txt', 'r')
OTAM_test = open('otam_split/somethingv2_meta_test.txt', 'r')

OriginTrain = open('sthv2_train_list_videos.txt', 'r')
OriginVal = open('sthv2_val_list_videos.txt', 'r')

found = False

print('takes about 15min, please wait...')
print('total 67013 lines')
for i, line in tqdm(enumerate(OTAM_train_train)):
    found = False
    # split line
    filename = line.split()[0]
    # search OriginTrain file, see if there is filename
    OriginTrain.seek(0)
    for ori_line in OriginTrain:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_train.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + filename)

print('total 10487 lines')
for i, line in tqdm(enumerate(OTAM_train_val)):
    found = False
    # split line
    filename = line.split()[0]
    # search OriginVal file, see if there is filename
    OriginVal.seek(0)
    for ori_line in OriginVal:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_train.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + filename)

print('total 1926 lines')
for i, line in tqdm(enumerate(OTAM_val)):
    found = False
    # split line
    filename = line.split()[0]
    # search OriginVal file, see if there is filename
    OriginVal.seek(0)
    for ori_line in OriginVal:
        if filename == ori_line.split('.')[0]:
            # write ori_line to my_ann_test
            my_ann_val.write(ori_line)
            found = True
            break
    if not found:
        print('no file: ' + filename)

print('total 2857 lines')
for i, line in tqdm(enumerate(OTAM_test)):
    found = False
    # split line
    filename = line.split()[0]
    # search OriginVal file, see if there is filename
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
OTAM_train_train.close()
OTAM_train_val.close()
OTAM_val.close()
OTAM_test.close()
my_ann_train.close()
my_ann_val.close()
my_ann_test.close()
