#!/usr/bin/python3
# 2019.12.02
# Author Zhang Yihao

import json
import numpy as np
from collections import defaultdict

dataName = "ml-1m"
iu_dict = defaultdict(list)
ui_dict = defaultdict(list)
reviews_data = []
reviews_train_data = []
reviews_test_data = []

global max_num_item  # 最大items编号
global max_num_user  # 最大items编号
global line_num  # 最大items编号

line_num = 1
with open('../Data/' + dataName + '/' + dataName, encoding='UTF-8') as f:
    for line in f:
        reviews_data.append(line)
        line_num += 1
    f.close()

'''建立ID和Num的对应关系 '''
User2Num = {}
Item2Num = {}
num = 1
for uu in reviews_data:
    sp_str = uu.split("\t")
    user_id = sp_str[0]
    if user_id not in User2Num:
        User2Num[user_id] = num
        num += 1
num = 1
for ui in reviews_data:
    sp_str = ui.split("\t")
    item_id = sp_str[1]
    if item_id not in Item2Num:
        Item2Num[item_id] = num
        num += 1

# loading_user_items
max_num_item = 1
max_num_user = 1

for d in reviews_data:
    sp_str = d.split("\t")
    user_id = sp_str[0]
    item_id = sp_str[1]
    user_num = User2Num.get(user_id)
    item_num = Item2Num.get(item_id)
    if user_num not in ui_dict:
        ui_dict[user_num] = list()
        if int(user_num) > max_num_user:
            max_num_user = int(user_num)
    ui_dict[user_num].append(item_num)
    if item_id not in iu_dict:
        iu_dict[item_num] = list()
        if int(item_num) > max_num_item:
            max_num_item = int(item_num)
    iu_dict[item_num].append(user_num)

print("line_num=====", line_num)
print("max_num_user=====", max_num_user)
print("max_num_item=====", max_num_item)
sparse_value = 1-(line_num/(max_num_item*max_num_user))
print("sparse_value=====", sparse_value*100)
