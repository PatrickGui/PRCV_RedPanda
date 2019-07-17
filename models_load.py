from functools import partial

from torch import nn
import torchvision.models as M



class DenseNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, input):
        return self.net(input)

class ResNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.bn=nn.BatchNorm2d(3)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
         #   self.net.avgpool=nn.AdaptiveAvgPool2d(2)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
      #  x=self.bn(x)
        return self.net(x)
'''
    每类模型参数定义以及训练结果统计在
    https://docs.qq.com/sheet/DYVRDUGpOWlZWdlJI?tab=BB08J2&coord=B12A0A0
'''

resnet18_finetune = partial(ResNetFinetune, net_cls=M.resnet18)
resnet34_finetune = partial(ResNetFinetune, net_cls=M.resnet34)
resnet50_finetune = partial(ResNetFinetune, net_cls=M.resnet50)
resnet101_finetune = partial(ResNetFinetune, net_cls=M.resnet101)
resnet152_finetune = partial(ResNetFinetune, net_cls=M.resnet152)

densenet121_finetune = partial(DenseNetFinetune, net_cls=M.densenet121)

densenet161_finetune = partial(DenseNetFinetune, net_cls=M.densenet161)
densenet201_finetune = partial(DenseNetFinetune, net_cls=M.densenet201)