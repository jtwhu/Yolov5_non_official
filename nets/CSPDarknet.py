from email.mime import base
from turtle import forward
import torch 
import torch.nn as nn 

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Focus(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act=True): # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(cin*4, cout, k, s, p, g, act)
    
    def forward(self, x):
        # 320, 320, 12 ->320, 320, 64
        temp = self.conv(
            # 640, 640, 3 => 320, 320, 12
            # 隔一个像素取出一个像素，取四种方式在通道维度上进行堆叠
            # 结果是尺寸压缩，通道数扩张4倍数
            # b, c, w, h,所以dim=1
            torch.cat(
                [
                    x[..., ::2, ::2]  ,
                    x[..., 1::2, ::2] ,
                    x[..., ::2, 1::2] ,
                    x[..., 1::2, 1::2]
                ], 1
            )
        )
        print(temp.size())
        return temp

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)

class BottleNeck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(BottleNeck, self).__init__()

        c_ = int(c2*e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_block, self).__init__()
        
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2*c_, c2, 1)

        # 叠加残差块,这块语法没太搞清楚
        self.m = nn.Sequential( *[BottleNeck(c_, c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)])
    
    def forward(self, x):
        return self.cv3(
            torch.cat(
                self.m(self.cv1(x)),
                self.cv2(x)     
            ),dim=1
        )

class SPP():
    # spatial pyramid pooling layer used in YoloV3-SPP
    # SPP使用了不同大小的池化核做最大池化进行特征提取，提高网络感受野
    # yolov4中间，SPP模块被镶嵌在FPN结构中，但是v5中被放到了backbone中
    def __init__(self, c1, c2, k=(5, 9, 13)) -> None:
        super(SPP, self).__init__()
        
        c_ = c1 // 2
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k)+1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding= x // 2) for x in k])

        def forward(self, x):
            x = self.cv1(x)
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth):
        super(CSPDarknet, self).__init__()

        # 640， 640， 3 -》320， 320， 12-》320， 320， 64
        self.stem = Focus(3, base_channels, k=3)

        # 320, 320, 64 --> 160, 160, 128
        self.dark2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            Conv(base_channels, base_channels*2, 3, 2),
            # 160, 160, 128 -> 160, 160, 128
            C3_block(base_channels*2, base_channels*2, base_depth)
        )

        self.dark3 = nn.Sequential(
            # 160， 160， 128 -> 80, 80, 256
            Conv(base_channels*2, base_channels*4, 3, 2),
            # 80, 80, 256 -> 80, 80 256
            C3_block(base_channels*2, base_channels*2, base_depth*3)
        )

        self.dark4 = nn.Sequential(
            # 80, 80, 256 -> 40 * 40 * 512
            Conv(base_channels*4, base_channels*8, 3, 2),
            # 80, 80, 256 -> 80, 80 256
            # base_depth代表残差快需要堆叠几层
            C3_block(base_channels*8, base_channels*8, base_depth*3)
        )

        self.dark5 = nn.Sequential(
            # 40 * 40 * 512 -> 20 * 20 * 1024
            Conv(base_channels*8, base_channels*16, 3, 2),
            # 20 * 20 * 1024
            SPP(base_channels*16, base_channels*16),
            # 20 * 20 * 1024
            C3_block(base_channels*16, base_channels*16, base_depth*3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)

        x = self.dark3(x)
        feat1 = x

        x = self.dark4(x)
        feat2 = x

        x = self.dark5(x)
        feat3 = x

        return feat1, feat2, feat3

if __name__ == "__main__":
    fs = Focus(3, 64)
    print(fs)
    x = torch.ones([1, 3, 640, 640])
    fs.forward(x)
    pass