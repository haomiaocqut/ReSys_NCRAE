#!/usr/bin/python3
# 2019.11.01
# Author Zhang Yihao @NUS

import numpy as np

dataName = "Yelp"
factor_num = "10"

data = np.load('../Data/' + dataName + '/' + dataName + '_rating_e' + factor_num +'.npz')
for f_name in data.files:
    file = open('../Data/' + dataName + '/' + dataName + '.' + factor_num + '.' + f_name, "w")
    num = 0  # user,item 开始编号
    for line in data[f_name]:
        line = line.tolist()
        line = str(line).replace(",", " ").strip('[').strip(']')
        line = str(num) + " " + line
        num += 1
        file.writelines(line + "\n")
    file.close()
