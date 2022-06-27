import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pandas as pd
import random
import pickle
from tqdm import tqdm
import os
import sys
sys.path.append('../')
sys.path.append('../../')
from make_datasets import make_datasets
from collections import defaultdict
from multiprocessing import Process, Queue
import numpy as np


# trainIterator = DataIterator('train',d_train, batch_size, neg_sample,
#                                  all_items, items_usr_clicked, shuffle=True)

class DataIterator:

    def __init__(self,
                 mode,
                 data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = self.datasize - self.idx

        cur = self.data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u in cur['user']:
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                # batch_neg.append(random.sample(user_item_set,self.neg_count))
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # print(set(self.all_items))
                # print(set(self.item_usr_clicked[u]))
                # print(user_item_set)
                # print(self.neg_count)
                batch_neg.append(random.sample(user_item_set,self.neg_count)) # 防止报错 Sample larger than population or is negative




        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)

# if __name__ == '__main__':
#     file_path = 'datasets/ml_sample1.txt'
#     names = ['user', 'item', 'rateing', 'timestamps']
#     data = pd.read_csv(file_path, header=None, sep='::', names=names)
#     # print(data)
#     d_train, d_test, d_info = make_datasets(data, 5, 3, 4)
#     print(d_train)
#     num_usr, num_item, items_usr_clicked, _, _ = d_info
#     all_items = [i for i in range(num_item)]

#     # Define DataIterator

#     trainIterator = DataIterator('train', d_train, 21, 5,
#                                  all_items, items_usr_clicked, shuffle=True)
#     for epoch in range(6):
#         for data in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
#             batch_usr, batch_seq, batch_pos, batch_neg = data
#             # print(batch_usr, batch_seq, batch_pos, batch_neg)


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen,
                    result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self,
                 User,
                 usernum,
                 itemnum,
                 batch_size=64,
                 maxlen=10,
                 n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(User, usernum, itemnum, batch_size, maxlen,
                              self.result_queue, np.random.randint(2e9))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()



# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('%s' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        if valid_user % 5000 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user