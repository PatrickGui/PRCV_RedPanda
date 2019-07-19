import random
import datetime
import models_load
from data_load import *
from data_preprocess import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
#
# SEED=10
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

ANNOTATION_TRAIN = '/media/phgui/2D6E11B1D89550E7/IDADP-PRCV2019-training/protocol/Train/'
IMAGE_PRE = '/media/phgui/2D6E11B1D89550E7/IDADP-PRCV2019-training/'
ANNOTATION_VAL = '/media/phgui/2D6E11B1D89550E7/IDADP-PRCV2019-training/protocol/Val/'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True

NB_CLASS = 59
IMAGE_SIZE = 224
BATCH_SIZE = 16  #2250/16
DATE = str(datetime.date.today())
def getModel():
    print('[+] loading model... ', end='', flush=True)
    model = models_load.densenet121_finetune(NB_CLASS)
    model.cuda()
    pth = os.listdir('./model/DesNet121') ##pth[0] --> 最新的.pth文件
    if len(pth) != 0:
        model.load_state_dict(torch.load('./model/DesNet121/' + pth[0])['state_dict'])
        print('[+] loading model :'+pth[0])
    print('Done')
    return model

def train(epochNum):
    writer = SummaryWriter('log/' + DATE + '/DesNet121/')  # 创建 /log/日期/InceptionResnet的组织形式  不同模型需要修改不同名称
    train_dataset = MyDataSet(
        root=ANNOTATION_TRAIN,
        transform=preprocess(normalize_torch, image_size=IMAGE_SIZE),
        root_pre=IMAGE_PRE
    )
    val_dataset = MyDataSet(
        root=ANNOTATION_VAL,
        transform=preprocess(normalize_torch, IMAGE_SIZE),
        root_pre=IMAGE_PRE
    )
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataLoader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # train_dataLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    # val_dataLoader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

    model = getModel()


    # weight = torch.Tensor(
    #     [1, 3, 3, 3, 3, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 3, 4, 2, 3, 1, 1, 3, 2, 2, 1, 3, 3, 1, 3, 2, 3,
    #      3, 3, 3, 2, 1, 3, 2, 3, 3, 3, 1, 3, 3, 4, 4, 3, 2, 2, 3, 1, 1, 3]).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    patience = 0  #防止局部最小
    lr = 0.0
    # momentum = 0.0
    min_loss = 3.1
    min_acc = 0.80
    print('min_loss is :%f' % (min_loss))
    for epoch in range(epochNum):
        print('Epoch {}/{}'.format(epoch, epochNum-1))
        if epoch==0 or epoch==1 or epoch==2:
            lr = 1e-3
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr, amsgrad=True, weight_decay=1e-4)
            # optimizer = torch.optim.SGD(params=model.fresh_params(), lr=lr, momentum=0.9)
        elif epoch==3:
            lr = 1e-4
            momentum = 0.9
            print('set lr=:%f,momentum=%f' % (lr, momentum))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-4)
#             optimizer=torch.optim.SGD(params=model.parameters(),lr=lr,momentum=momentum)

        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('./model/DesNet121/' + DATE + '_loss_best.pth')['state_dict'])
            lr = lr / 10
            print('loss has increased lr divide 10 lr now is :%f' % (lr))

        running_loss = RunningMean()
        running_corrects = RunningMean()
        for n_iter, (inputs, labels) in tqdm(enumerate(train_dataLoader)):

            model.train(True)
            n_batchsize = inputs.size(0)
            # 判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
            if torch.cuda.is_available():
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 找出分数最高的对应的channel，即为top-1类别
            _, preds = torch.max(outputs.data, dim=1)
            if isinstance(outputs, tuple):
                loss = sum((criterion(o, labels)) for o in outputs)
                # loss = sum(criterion(o, labels) for o in outputs)
            else:
                loss = criterion(outputs, labels)
            # print(loss.item())
            running_loss.update(loss.item(), 1)

            running_corrects.update(torch.sum(preds==labels.data).data, n_batchsize)
            # running_corrects.update(torch.sum(preds == labels.data).data, n_batchsize)

            loss.backward()
            optimizer.step()

            if n_iter%10 == 9:
                print('[epoch:%d, batch:%d]:Acc:%f,loss:%f'%
                      (epoch, n_iter, running_corrects.value, running_loss.value))
                if n_iter %30 == 29:
                    writer.add_scalar('Train/Acc', running_corrects.value,n_iter)
                    writer.add_scalar('Train/Loss', running_loss.value, n_iter)


                    lx, px = predict(model, val_dataLoader)
                    _, preds = torch.max(px, dim=1)
                    accuracy = torch.mean((preds == lx).float())
                    log_loss = criterion(Variable(px), Variable(lx))
                    log_loss = log_loss.item()

                    writer.add_scalar('Val/Acc', accuracy, n_iter)
                    writer.add_scalar('Val/Loss', log_loss, n_iter)

                    print('[epoch:%d,batch:%d]: val_loss:%f,val_acc:%f,val_total:%d' % (
                    epoch, n_iter, log_loss, accuracy, len(val_dataset)))

        print('[epoch:%d] :acc: %f,loss:%f,lr:%f,patience:%d' % (
        epoch, running_corrects.value, running_loss.value, lr, patience))


        lx, px = predict(model, val_dataLoader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.item()

        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        print('[epoch:%d]: val_loss:%f,val_acc:%f,' % (epoch, log_loss, accuracy))

        writer.add_scalar('Val/Acc', accuracy, n_iter)
        writer.add_scalar('Val/Loss', log_loss, n_iter)

        if log_loss < min_loss:
            snapshot(os.getcwd()+'/model/DesNet121', DATE + '_loss_best.pth', {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': log_loss,
                'val_correct': accuracy})
            patience = 0
            min_loss = log_loss
            print('save new model loss,now loss is ', min_loss)
        else:
            patience += 1
        if accuracy > min_acc:
            snapshot(os.getcwd()+'model/DesNet121', DATE + '_acc_best.pth', {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': log_loss,
                'val_correct': accuracy})
            min_acc = accuracy
            print('save new model acc,now acc is ', min_acc)

'''
预测data在model上的结果
'''
def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            all_labels.append(labels)
            inputs = Variable(inputs).cuda()
            outputs = model(inputs)
            all_outputs.append(outputs.data.cpu())
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()
    return all_labels, all_outputs
'''
测试data在model上的结果
'''
def test(model):
    test_dataset = MyDataSet(
        root=ANNOTATION_VAL,
        transform=preprocess(normalize_torch, IMAGE_SIZE),
        root_pre=IMAGE_PRE
    )
    test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_dataLoader):
            all_labels.append(labels)
            inputs = Variable(inputs).cuda()
            outputs = model(inputs)
            all_outputs.append(outputs.data.cpu())
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()
        _, preds = torch.max(all_outputs, dim=1)
        accuracy = torch.mean((preds == all_labels).float())
        print('Test Acc:%f, Test Total:%d' % (accuracy, len(test_dataset)))

if __name__ == '__main__':
    # train(20)
    test(getModel())




