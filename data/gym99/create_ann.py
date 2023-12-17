train_set = [
    0, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 40, 41, 43, 44, 46, 47, 48, 49, 53,
    55, 58, 59, 68, 70, 71, 72, 73, 74, 75, 78, 80, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94
]
val_set = [10, 19, 32, 56, 57, 62, 66, 69, 79, 82, 95, 96]
test_set = [
    1, 5, 8, 18, 20, 31, 39, 42, 45, 50, 51, 52, 54, 60, 61, 63, 64, 65, 67,
    76, 77, 81, 97, 98
]

# for all elements in training_set, go through all lines in gym_train_list_rawframes.txt
# if the last element in the line is equal to that element, write the line to train.txt
for i in train_set:
    for line in open('gym_train_list_rawframes.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('train.txt', 'a') as f:
                f.write(line)
    for line in open('gym_val_list_rawframes.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('train.txt', 'a') as f:
                f.write(line)
for i in val_set:
    for line in open('gym_train_list_rawframes.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('val.txt', 'a') as f:
                f.write(line)
    for line in open('gym_val_list_rawframes.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('val.txt', 'a') as f:
                f.write(line)
for i in test_set:
    for line in open('gym_train_list_rawframes.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('test.txt', 'a') as f:
                f.write(line)
    for line in open('gym_val_list_rawframes.txt'):
        if line[-1] != '\n':
            line = line + '\n'
        if int(line.split()[-1]) == i:
            with open('test.txt', 'a') as f:
                f.write(line)