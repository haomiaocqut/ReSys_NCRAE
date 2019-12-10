import subprocess
dataName = "Yelp"

data_set = '../Data/' + dataName + '/' + dataName +'_rating.npz'
pre_output = '../Data/' + dataName + '/' + dataName +'_rating_e10.npz'

subprocess.call(['python', 'pre_train.py', '--gpu', '0', '--dataset', data_set, '--output', pre_output])
