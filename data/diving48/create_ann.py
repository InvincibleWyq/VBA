train_set = [
    0, 1, 2, 3, 7, 8, 10, 13, 15, 17, 20, 21, 22, 23, 24, 25, 26, 29, 32, 34,
    35, 37, 38, 41, 42, 45, 46
]
val_set = [4, 11, 14, 16, 28, 31, 33, 36, 39, 47]
test_set = [5, 6, 9, 12, 18, 19, 27, 40, 43, 44]

# for all elements in training_set, go through all lines in diving48_train_list_videos.txt
# if the last element in the line is equal to that element, write the line to train.txt
for i in train_set:
    for line in open('diving48_train_list_videos.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('train.txt', 'a') as f:
                f.write(line)
    for line in open('diving48_val_list_videos.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('train.txt', 'a') as f:
                f.write(line)
for i in val_set:
    for line in open('diving48_train_list_videos.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('val.txt', 'a') as f:
                f.write(line)
    for line in open('diving48_val_list_videos.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('val.txt', 'a') as f:
                f.write(line)
for i in test_set:
    for line in open('diving48_train_list_videos.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('test.txt', 'a') as f:
                f.write(line)
    for line in open('diving48_val_list_videos.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('test.txt', 'a') as f:
                f.write(line)