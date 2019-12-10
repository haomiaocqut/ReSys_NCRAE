import random
dataName = 'Baby_HR&NDCG@10_NeuCF_100_lr=0.0005'

hr = ""
ndcg = ""
loss = ""
str_num = ""
hrP = 0
ndcgP = 0
num = 0

with open('../Experiment result/' + dataName + '.txt') as f:
    for line in f:
        if line.startswith("Iteration"):
            str_spl = line.split(",")
            hr_i = round(float(str_spl[0][-6:]) + hrP, 4)
            hr = hr + " " + str(hr_i)
            ndcg_i = round(float(str_spl[1][-6:]) + ndcgP, 4)
            ndcg = ndcg + " " + str(ndcg_i)
            loss = loss + " " + str_spl[2][7:14]
            num += 1
            str_num = str_num + " " + str(num)

    print("[" + str_num.strip() + "]")
    print("=" * 100)
    print("[" + hr.strip() + "]")
    print("="*100)
    print("[" + ndcg.strip() + "]")
    print("="*100)
    print("[" + loss.strip() + "]")

    f.close()
