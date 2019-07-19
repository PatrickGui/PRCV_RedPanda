import json
import os

from torch.autograd import Variable
import torch
from data_load import *

'''
训练过程中保存loss和acc
'''
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return float(self.total_value)/self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)




'''
保存训练
'''
def snapshot(savepathPre,savePath,state):

    if not os.path.exists(savepathPre):
        os.makedirs(savepathPre)
    torch.save(state, os.path.join(savepathPre, savePath))


'''
将44,45类删除

'''
def deleteNosiseType():
    data_train = json.load(open(ANNOTATION_TRAIN, 'r'))
    data_val = json.load(open(ANNOTATION_VAL, 'r'))

    for e in data_train:
        if e['disease_class']==44:
            data_train.remove(e)
            continue
        if e['disease_class'] == 45:
            data_train.remove(e)
            continue
        if e['disease_class'] > 45:
            e['disease_class'] = e['disease_class']-2

    with open(ANNOTATION_TRAIN,
              'w') as f:
        json.dump(data_train, f, ensure_ascii=False)
    with open(ANNOTATION_VAL,
              'w') as f:
        json.dump(data_val, f, ensure_ascii=False)


if __name__=='__main__':
    deleteNosiseType()
