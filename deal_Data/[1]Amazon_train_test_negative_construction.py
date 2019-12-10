#!/usr/bin/python3
# 2019.11.12
# Author Zhang Yihao @NUS

import json
import numpy as np
from collections import defaultdict

dataName = "Baby"
split_ratio = 4  # train_data与test_data的比例
iu_dict = defaultdict(list)
ui_dict = defaultdict(list)
ui_neg_dict = defaultdict(list)
reviews_data = []
reviews_train_data = []
reviews_test_data = []
train_list = []
test_list = []
global max_num_item  # 最大items编号

with open('../Data/' + dataName + '/' + dataName) as f:
    for line in f:
        line_dict = json.dumps(eval(line))
        reviews_data.append(json.loads(line_dict))
    f.close()

'''建立ID和Num的对应关系 '''
asin2itemNum = {}
reviewerID2userNum = {}
num = 1
for ui in reviews_data:
    if ui["asin"] not in asin2itemNum:
        asin2itemNum[ui["asin"]] = num
        num += 1
num = 1
for uu in reviews_data:
    if uu["reviewerID"] not in reviewerID2userNum:
        reviewerID2userNum[uu["reviewerID"]] = num
        num += 1

# loading_user_items
max_num_item = 1
for d in reviews_data:
    user_id = reviewerID2userNum.get(d["reviewerID"])
    item_id = asin2itemNum.get(d["asin"])
    if user_id not in ui_dict:
        ui_dict[user_id] = list()
    ui_dict[user_id].append(item_id)
    if item_id not in iu_dict:
        iu_dict[item_id] = list()
        if int(item_id) > max_num_item:
            max_num_item = int(item_id)
    iu_dict[item_id].append(user_id)

#划分reviews_train_data和reviews_test_data
if split_ratio != 0:
    num = 0
    for l_d in reviews_data:
        #u_id = reviewerID2userNum.get(l_d["reviewerID"])
        #len_items = len(ui_dict.get(int(u_id)))
        if num % (split_ratio + 1) < split_ratio:
            reviews_train_data.append(l_d)
            num += 1
        else:
            reviews_test_data.append(l_d)
            num += 1
else:
    num = 1
    for l_d in reviews_data:
        u_id = reviewerID2userNum.get(l_d["reviewerID"])
        len_items = len(ui_dict.get(int(u_id)))
        if num < len_items:
            # line_dict = json.dumps(eval(line))
            reviews_train_data.append(l_d)
        else:
            # line_dict = json.dumps(eval(line))
            reviews_test_data.append(l_d)
            num = 0
        num += 1

def split_train_test_data(train_neg_num):
    train_data = open('../Data/' + dataName + '/' + dataName + ".train.rating", "w")
    test_data = open('../Data/' + dataName + '/' + dataName + ".test.rating", "w")
    for l_train in reviews_train_data:
        user_id = reviewerID2userNum.get(l_train["reviewerID"])
        item_id = asin2itemNum.get(l_train["asin"])
        label_pos = "1"
        str_line = str(user_id) + "\t" + str(item_id) + "\t" + label_pos
        train_data.writelines(str_line + "\n")
        str_pos_list = ui_dict[int(user_id)]
        label_neg = "0"
        for i in range(train_neg_num):
            neg_item = np.random.randint(max_num_item)
            while str(neg_item) in str_pos_list:
                neg_item = np.random.randint(max_num_item)
            train_data.writelines(str(user_id) + "\t" + str(neg_item) + "\t" + label_neg + "\n")
    train_data.close()
    # 写入test_data
    for l_test in reviews_test_data:
        user_id = reviewerID2userNum.get(l_test["reviewerID"])
        item_id = asin2itemNum.get(l_test["asin"])
        label_pos = "1"
        str_line = str(user_id) + "\t" + str(item_id) + "\t" + label_pos
        test_data.writelines(str_line + "\n")
        test_list.append(str_line)
    test_data.close()


def get_train_data():
    for d_line in reviews_data:
        user_id = reviewerID2userNum.get(d_line["reviewerID"])
        item_id = asin2itemNum.get(d_line["asin"])
        # uiPair = "[" + str(user_id) + " " + str(item_id) + "]"
        ui_array = np.array([user_id, item_id])
        train_list.append(ui_array)
    return train_list


def get_test_data(num_negatives):
    test_data = {}
    t_num = 0
    u_id_old = 0
    for d_item in reviews_test_data:
        user_id = reviewerID2userNum.get(d_item["reviewerID"])
        if user_id != u_id_old:
            str_pos_list = ui_dict[int(user_id)]
            ui_neg_dict[user_id] = list()
            for i in range(num_negatives):
                neg_item = np.random.randint(max_num_item)
                while str(neg_item) in str_pos_list:
                    neg_item = np.random.randint(max_num_item)
                ui_neg_dict[user_id].append(neg_item)
            str_num_neg = str(user_id) + "," + str(ui_neg_dict[user_id])
            str_num_neg = tuple(eval(str_num_neg))
            if ui_neg_dict != '':
                test_data[t_num] = str_num_neg
                t_num += 1
        u_id_old = user_id
    return test_data


def get_negative_data(num_negative, max_num_item):
    test_neg = open('../Data/' + dataName + '/' + dataName + ".test.negative", "w")
    for t_line in test_list:
        user_id = t_line.split("\t")[0]
        item_id = t_line.split("\t")[1]
        str_pos_list = ui_dict[int(user_id)]
        str_pos_neg = "(" + user_id + "," + item_id + ")"
        for _ in range(num_negative):
            neg_item = np.random.randint(max_num_item)
            while str(neg_item) in str_pos_list:
                neg_item = np.random.randint(max_num_item)
            str_pos_neg = str_pos_neg + "\t" + str(neg_item)
        test_neg.writelines(str_pos_neg + "\n")
    test_neg.close()
    return "test_negative successful!"


if __name__ == '__main__':
    train_neg_num = 4
    num_negatives = 100  # 构建num_negatives的个数
    train_list = get_train_data()
    test_dict = get_test_data(num_negatives)
    split_train_test_data(train_neg_num)
    test_negative = get_negative_data(num_negatives, max_num_item)
    '''保存数据为npz格式'''
    np.savez('../Data/' + dataName + '/' + dataName +'_rating.npz', train_data=np.array(train_list), test_data=np.array(test_dict))
