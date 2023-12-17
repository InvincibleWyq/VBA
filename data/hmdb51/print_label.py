file_list = [
    open('fewshot_split/train.txt', 'r'),
    open('fewshot_split/val.txt', 'r'),
    open('fewshot_split/test.txt', 'r')
]
name = ['train', 'val', 'test']
print('fewshot-hmdb51:')
for i, file in enumerate(file_list):
    print(name[i] + ' label: ')
    label_set = set()
    for line in file:
        label_set.add(line.split()[1])
    print(label_set)
    file.close()
'''
train label:
{'2', '50', '4', '11', '3', '32', '49', '10', '39', '29', '9', '15', '28',
'45', '51', '17', '0', '5', '48', '13', '46', '20', '34', '8', '6', '23',
'36', '43', '19', '27', '31'}

val label:
{'44', '18', '24', '37', '1', '12', '16', '40', '42', '35'}

test label:
{'14', '30', '25', '21', '26', '47', '22', '38', '41', '33'}
'''
