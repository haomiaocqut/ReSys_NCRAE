#!/usr/bin/python3
# 2019.8.12
# Author Zhang Yihao @NUS

import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Global variables that are shared across processes
_model = None
_train = None
_testRatings = None
_testNegatives = None
_K = None


def eval_mae_rmse(model, testRating, user_review_fea, item_review_fea):
    _mae, _mse = [], []
    testU, user_fea, testI, item_fea, testL = [], [], [], [], []

    for (u, i) in testRating.keys():
        # positive instance
        testU.append(u)
        user_fea.append(user_review_fea[u])
        testI.append(i)
        item_fea.append(item_review_fea[i])
        label = testRating[u, i]
        testL.append(label)
    score = model.predict(
        [np.array(testU), np.array(user_fea, dtype='float32'), np.array(testI), np.array(item_fea, dtype='float32')])
    mae = mean_absolute_error(testL, score)
    mse = mean_squared_error(testL, score)
    rmse = np.sqrt(mse)
    return mae, rmse


def evaluate_model(model, train, testRatings, testNegatives, user_review_fea, item_review_fea, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _train
    global _user_review_fea
    global _item_review_fea
    _model = model
    _train = train
    _testRatings = testRatings
    _testNegatives = testNegatives
    _user_review_fea = user_review_fea
    _item_review_fea = item_review_fea
    _K = K

    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return hits, ndcgs
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs


def test_generator(user, items):
    # t1 = time()
    users = np.full(len(items), user, dtype='int32')
    Xuser = _train[users, :].todense()
    Xitem = _train[:, items].todense().transpose()
    Xuser = np.array(Xuser, dtype='int32')
    Xitem = np.array(Xitem, dtype='int32')
    return [Xuser, Xitem]

'''
def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    t1 = time()
    for i in range(len(items)):
        item = items[i]
        user_fea = _user_review_fea[item]
        item_fea = _item_review_fea[item]
        predictions = _model.predict([u, np.array(user_fea, dtype='float32'), item, np.array(item_fea, dtype='float32')], batch_size=100, verbose=0)
    #predictions = _model.predict_on_batch(test_generator(u, items))
        map_item_score[item] = predictions[i]
    items.pop()
'''
def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    t1 = time()
    testU, user_fea, testI, item_fea = [], [], [], []

    for i in range(len(items)):
    # positive instance
        testU.append(u)
        user_fea.append(_user_review_fea[u])
        id_item = items[i]
        testI.append(id_item)
        item_fea.append(_item_review_fea[id_item])
    predictions = _model.predict([np.array(testU), np.array(user_fea, dtype='float32'), np.array(testI), np.array(item_fea, dtype='float32')])
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    rank_list = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(rank_list, gtItem)
    ndcg = getNDCG(rank_list, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
