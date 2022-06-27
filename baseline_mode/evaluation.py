'''
Author: your name
Date: 2021-09-25 15:44:45
LastEditTime: 2021-10-08 21:45:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Sequential_Recommendation_Tensorflow-master\evaluation.py
'''
import imp
import numpy as np
import math
import random

def Metric_PrecN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(preds)

    return sum / count

def Metric_RecallN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(target)
    return sum / count

def cal_PR(target_list,predict_list,k=[1,5,10]):

    display_list = []

    for s in k:
        prec = Metric_PrecN(target_list,predict_list,s)
        recall = Metric_RecallN(target_list,predict_list,s)
        display = "Prec@{}:{:g} Recall@{}:{:g}".format(s,round(prec,4),s,round(recall,4))
        display_list.append(display)

    return ' '.join(display_list)


def Metric_HR(TopN, target_list, predict_list):
    sums = 0
    count = 0
    #print(len(target_list))
    #print(len(predict_list))
    for i in range(len(target_list)):
        preds = predict_list[i]
        top_preds = preds[:TopN]

        for target in target_list[i]:
            if target in top_preds:
                sums+=1
            count +=1
    print(sums, count)
    return float(sums) / count

# --------------------ndcg-------------------------------

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def Metric_NDCG(TopN, target_list, predict_list):

    sums = 0
    count = 0
    ndcg_sum = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        top_preds = preds[:TopN]

        ndcg = getNDCG(top_preds, target_list[i])

        # print(ndcg)

        ndcg_sum += ndcg



    #     for target in target_list[i]:
    #         if target in top_preds:
    #             sums+=1
    #         count +=1
    # print(sums, count)
    # print(target_list)
    # # print(len(target_list))
    # print('-------------')
    # print(top_preds)
    return ndcg_sum/len(target_list)

# --------------------ndcg-------------------------------

def Metric_MRR(target_list, predict_list):

    sums = 0
    count = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        for t in target_list[i]:
            rank = preds.index(t) + 1
            sums += 1 / rank
            count += 1

    return float(sums) / count

def SortItemsbyScore(next_list,  item_list, item_score_list, reverse=False,remove_hist=False, usr = None, usrclick = None):

    totals = len(item_score_list)
    result_items = []
    result_score = []
    for i in range(totals):
        u = usr[i]
        u_clicks = usrclick[u]
        item_score = item_score_list[i]
        tuple_list = sorted(list(zip(item_list,item_score)),key=lambda x:x[1],reverse=reverse)
        tuple_dict = dict(zip(item_list, item_score))
        next_item = next_list[i][0]
        next_score = tuple_dict[next_item]

        if remove_hist:
            tmp = []
            for t in tuple_list:
                if t[0] not in u_clicks:
                    tmp.append(t)
            tuple_list = tmp

        # ----------------随机采样100个-------------------------------
        tuple_list_100 = random.sample(tuple_list, 100)
        tuple_list_100.append((next_item, next_score))
        tuple_list_100 = sorted(tuple_list_100,
                                key=lambda x: x[1],
                                reverse=reverse)

        # ----------------------------------------------------------------------
        x, y = zip(*tuple_list_100)
        sorted_item = list(x)
        sorted_score = list(y)
        result_items.append(sorted_item)
        result_score.append(sorted_score)

    return result_items,result_score
