#!/usr/bin/python3
# 2019.11.21
# Author Zhang Yihao

from collections import defaultdict
import json

dataName = "Beauty"
num_link = 20  # the number of each user link to items
# num_bought = 10

# user -> item
ui_dict = defaultdict(list)
iu_dict = defaultdict(list)

reviews_data = []

with open('G:/Datasets/Amazon datasets/' + dataName + '.json') as f:
    for line in f:
        reviews_data.append(json.loads(line))
    f.close()

for line_data in reviews_data:
    user_id = line_data["reviewerID"]
    item_id = line_data["asin"]
    ui_dict[user_id].append(item_id)
print("len(ui_dict)=====", len(ui_dict))

ex_data = open('../Data/' + dataName +'/' + dataName, "w")
for line_dict in reviews_data:
    user_id = line_dict["reviewerID"]
    item_id = line_dict["asin"]
    ui_num = len(ui_dict[user_id])
    if ui_num >= num_link:
        ex_data.writelines(str(line_dict) + "\n")
        iu_dict[item_id].append(user_id)
print("len(iu_dict)=====", len(iu_dict))
ex_data.close()
