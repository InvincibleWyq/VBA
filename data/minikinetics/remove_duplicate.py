from tqdm import tqdm

duplicate_file = open("duplicate_list.txt", "r")
file_read_list = [
    open("kinetics-100-replaced/train.txt", "r"),
    open("kinetics-100-replaced/val.txt", "r"),
    open("kinetics-100-replaced/test.txt", "r"),
]
file_write_list = [
    open("kinetics-100-replaced/train_new.txt", "w"),
    open("kinetics-100-replaced/val_new.txt", "w"),
    open("kinetics-100-replaced/test_new.txt", "w"),
]
file_k400 = open("kinetics400_train_list_videos.txt", "r")

duplicate_list = []

for file_read in file_read_list:
    for i, line in enumerate(file_read):
        duplicate = False
        duplicate_file.seek(0)
        for j, duplicate_line in enumerate(duplicate_file):
            if duplicate_line[:-1] == line.split()[0] and line.split(
            )[0] not in duplicate_list:
                # first time to find duplicate, add to duplicate_list
                duplicate_list.append(line.split()[0])
            elif duplicate_line[:-1] == line.split()[0] and line.split(
            )[0] in duplicate_list:
                # second or more time to find duplicate, write to new file
                duplicate = True
                label = line.split()[1]
                file_k400.seek(0)
                for k400_line in file_k400:
                    already_in = False
                    if label == k400_line.split()[1]:
                        k400_filename = k400_line.split()[0]
                        file_read_list2 = [
                            open("kinetics-100-replaced/train.txt.bak", "r"),
                            open("kinetics-100-replaced/val.txt.bak", "r"),
                            open("kinetics-100-replaced/test.txt.bak", "r"),
                        ]
                        for file_read2 in file_read_list2:
                            # if k400_filename exists in file_read2, then continue
                            if already_in:
                                break
                            file_read2.seek(0)
                            for line2 in file_read2:
                                if k400_filename == line2.split()[0]:
                                    already_in = True
                                    break
                        for file_read2 in file_read_list2:
                            file_read2.close()
                        if already_in:
                            continue
                        file_write_list[file_read_list.index(file_read)].write(
                            k400_line)
                        break
                break
        if not duplicate:
            file_write_list[file_read_list.index(file_read)].write(line)

# close all files
duplicate_file.close()
for file_read in file_read_list:
    file_read.close()
for file_write in file_write_list:
    file_write.close()
