file_list = [
    open('minissv2_small/train.txt', 'r'),
    open('minissv2_small/val.txt', 'r'),
    open('minissv2_small/test.txt', 'r')
]
name = ['train', 'val', 'test']
print('minissv2_small:')
for i, file in enumerate(file_list):
    print(name[i] + ' label: ')
    label_set = set()
    for line in file:
        label_set.add(line.split()[1])
    print(label_set)
    file.close()
'''
train label: {'160', '67', '111', '92', '31', '54', '87', '104', '156', '144',
'76', '32', '88', '102', '19', '20', '137', '38', '100', '159', '98', '128',
'64', '21', '49', '34', '117', '105', '75', '121', '36', '55', '26', '3', '1',
'136', '69', '106', '141', '17', '166', '153', '56', '129', '33', '12', '59',
'125', '173', '74', '30', '99', '35', '84', '171', '4', '146', '22', '58',
'48', '23', '63', '0', '78'}

val label: {'89', '157', '60', '27', '113', '62', '155', '41', '120', '70',
'18', '162'}

test label: {'123', '53', '172', '119', '51', '10', '110', '95', '101', '109',
'140', '148', '24', '13', '46', '81', '44', '107', '52', '169', '143', '122',
'134', '170'}
'''

file_list = [
    open('minissv2_full/train.txt', 'r'),
    open('minissv2_full/val.txt', 'r'),
    open('minissv2_full/test.txt', 'r')
]
name = ['train', 'val', 'test']
print('\nminissv2_full:')
for i, file in enumerate(file_list):
    print(name[i] + ' label: ')
    label_set = set()
    for line in file:
        label_set.add(line.split()[1])
    print(label_set)
    file.close()
'''
train label: {'124', '16', '111', '25', '6', '149', '92', '9', '168', '29',
'37', '66', '156', '32', '103', '88', '102', '19', '157', '8', '38', '100',
'61', '137', '159', '10', '45', '49', '34', '95', '132', '145', '140', '5',
'57', '3', '43', '147', '141', '166', '82', '56', '59', '125', '52', '91',
'14', '169', '65', '60', '99', '114', '73', '171', '138', '143', '22', '93',
'68', '112', '170', '165', '23', '78'}

val label: {'50', '123', '28', '97', '20', '146', '128', '31', '153', '105',
'63', '70'}

test label: {'67', '71', '54', '7', '62', '86', '76', '47', '121', '148', '94',
'13', '89', '136', '126', '129', '72', '12', '79', '30', '11', '154', '158',
'0'}
'''
