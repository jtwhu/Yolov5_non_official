import torch 
import torch.nn as nn

from CSPDarknet import *


class YoloBody(nn.Module):
    def __init__(self, num_classes, base_channels=64, base_depth=3):
        super(YoloBody, self).__init__()

        # backbone输出三个有效特征层
        self.backbone = CSPDarknet(base_channels=base_channels, base_depth=base_depth)

        self.conv_for_feat3        = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.upsample1             = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3_for_upsample1   = C3_block(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2        = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.upsample2             = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3_for_upsample2   = C3_block(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.downsample1           = Conv(base_channels*4, base_channels*4, 3, 2)
        self.conv3_for_downsample1 = C3_block(base_channels*8, base_channels*8, base_depth, shortcut=False)

        self.downsample2           = Conv(base_channels*8, base_channels*8, 3, 2)
        self.conv3_for_downsample2 = C3_block(base_channels*16, base_channels*16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3*(5+num_classes) =>80, 80, 3 * (4+1+num_classes)
        self.yolo_head_p3 = nn.Conv2d(base_channels*4 , 3 *(5+num_classes), 1)
        self.yolo_head_p4 = nn.Conv2d(base_channels*8 , 3 *(5+num_classes), 1)
        self.yolo_head_p5 = nn.Conv2d(base_channels*16, 3 *(5+num_classes), 1)

        
    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)

        #======================================上采样并堆叠=============================================
        # feat1 = (80, 80, 256 ) ==> P3 = (80, 80, 256 )
        # feat2 = (40, 40, 512 ) ==> P4 = (40, 40, 512 )
        # feat3 = (20, 20, 1024) ==> P5 = (20, 20, 1024)

        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample1(P5)

        # 40, 40, 512 -> 40, 40, 1024
        P5 = torch.cat([P5_upsample, feat2], 1)
        # csp_layer
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P5)
        # 通道数调整：40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample2(P4)

        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)
        # csp_layer : 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)       

        #======================================下采样并堆叠=============================================
        P3_downsample = self.downsample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.downsample2(P4)
        P5 = torch.cat([P4_downsample, P5])
        P5 = self.conv3_for_downsample2(P5)


         #======================================检测结果输出=============================================
         #   y3=(batch_size,75,80,80)
         out2 = self.yolo_head_p3(P3)

         #   y2=(batch_size,75,40,40)
         out1 = self.yolo_head_p4(P4)

         #   y1=(batch_size,75,20,20)
         out0 = self.yolo_head_p5(P5)
         
         return out0, out1, out2

if __name__ == "__main__":

    yolo = YoloBody(num_classes=64, base_channels=64, base_depth=3)

    print("========testing=========")

    print(yolo)