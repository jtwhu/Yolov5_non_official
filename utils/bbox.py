import numpy as np
import torch
from torchvision.ops import nms

class DecodeBbox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBbox, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape

        #============================================================
        #  20x20的特征层对应的anchor为[161, 90], [156, 198], [373, 326]
        #  40x40的特征层对应的anchor为[30, 61], [62, 45], [59, 119]
        #=============================================================
        
        self.anchors_mask = anchors_mask


    def decode_bbox(self, inputs):
        outputs = []

        for i, inputs in enumerate(inputs):
            batch_size = inputs.size(0)
            input_height = input.size(2)
            input_width = input.size(3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        '''
        对20*20的特征图进行可视化
        '''
        # input: batch_size, 3 * (4 + 1 + num_classes), 20, 20
        batch_size = input.size(0)#        print(input.size()[3])
        input_height = input.size(2)
        input_width = input.size(3)

        # 待检测图像；640, 640
        # 待可视化特征图：20, 20
        # stride_h = stride_W = 640 / 20 = 32

        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width

        # 计算相对于特征层的先验框anchor大小
        # 计算了缩放倍数==> 输入图像大小 / 特征图大小
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[2]]]
        print(scaled_anchors)

        # 对输出进行reshape
        # batch_size, 3*(4+1+num_classes), 20, 20====>
        # batch_size, 3, 5+num_classes, 20, 20 =====>转置
        # batch_size, 3, 20, 20, 5+num_classes
        prediction = input.view(batch_size, 
                                3, 
                                num_classes+5,
                                input_height, 
                                input_width).permute(0, 1, 3, 4, 2).contiguous()
        
        # =====================以下参数都被固定到0-1之间=========================
        # 先验框中心位置调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框宽高调整参数
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # 获取置信度
        conf = torch.sigmoid(prediction[..., 4])
        # 获取种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor


        #########################################################################
        #                                                                       #
        #                           先验框生成                                   #
        #                                                                       #
        #########################################################################

        # ===========================S1:先验框中心生成===============================
        # 
        #   [
        #       [0, 1, 2, 3 ……, 19], 
        #       [0, 1, 2, 3 ……, 19], 
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ] * (batch_size * 3)
        # 生成列表==>重复增加tensor==>在channel维度重复扩增tensor
        # 最终输出：batch_size, 3, 20, 20

        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * 3, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size*3, 1, 1).view(y.shape).type(FloatTensor)

        print(grid_x.shape)
        print(grid_y.shape)

        # ===========================S2:先验框宽高生成===============================
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height*input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height*input_width).view(h.shape)


        # 使用预测结果对先验框进行调整
        # 输出调整后的先验框


    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30,61], [62,45], [59,119], [10,13], [16,30], [33,23]])
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors=anchors, anchors_mask=anchors_mask, num_classes=80)
