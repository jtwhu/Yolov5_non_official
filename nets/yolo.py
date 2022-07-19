import torch 
import torch.nn as nn

from CSPDarknet import *


class YoloBody(nn.Module):
    def __init__(self):
        super(YoloBody, self).__init__()

        # backbone输出三个有效特征层
        self.backbone = CSPDarknet()
        
    def forward(self, x):
        feature1, fearture2, feature3 = self.backbone(x)


if __name__ == "__main__":

    yolo = YoloBody()

    print("========testing=========")

    print(yolo)