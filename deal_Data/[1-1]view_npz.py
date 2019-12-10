import numpy as np

line_num = 20
dataName = "Yelp"

ui_data = np.load('../Data/' + dataName + '/' + dataName +'_rating.npz')
file = ui_data.files
print(file)

'''
train_data = open('../Data/' + dataName +'/train.txt', "w")
test_data = open('../Data/' + dataName +'/test.txt', "w")
for line in ui_data['train_data']:
    train_data.writelines(str(line) + "\n")
for line in ui_data['test_data'].tolist().values():
    test_data.writelines(str(line)+ "\n")
train_data.close()
test_data.close()

'''

i = 0
for line in ui_data['train_data']:
    if i <= line_num:
        print(line)
        i += 1
    else:
        break


i = 0
list_line = ui_data['test_data'].tolist()
for line in list_line.values():
    if i <= line_num:
        print(line)
        i += 1
    else:
        break

