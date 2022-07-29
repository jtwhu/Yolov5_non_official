from turtle import forward
from unittest import result
import torch
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6,7,8], [3,4,5], [0,1,2]], label_smoothing=0):
        
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05
        self.obj_ratio      = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio      = 0.5 * (num_classes / 80)
        self.cuda = cuda

    
    def get_pred_boxes(self, l, x, y, w, h, targets, scaled_anchors, in_h, in_w):
        bs = len(targets)

        # 生成网格
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor[0]).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor[1]).type_as(x)
        
        # bs, 3, 20, 20, 5 + num_classes
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        pred_boxes_x = torch.unsqueeze(x * 2 - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2 - 0.5 + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)

        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_w, pred_boxes_h], dim=-1)
        return pred_boxes

    
    def box_iou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # xywh==>xyxy
        # 预测框左上角右下角
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        # 真实框左上角右下角
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half

        # 计算真实框与预测框所有的iou
        inertsect_mins = torch.max(b1_mins, b2_mins)# 相交区域的左下角
        intersect_maxes = torch.min(b1_maxes, b2_maxes)# 相交区域的右上角
        intersect_wh = torch.max(intersect_maxes - inertsect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area
        
        # GIOU的计算公式即可得到以下代码
        # 计算包围两个框的最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh    = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

        # 计算对角线距离
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou         = iou - (enclose_area - union_area) / enclose_area
        return giou

    
    # 待验证
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result


    def BCELoss(self, pred, target):
        epsilon = 1e-7 
        pred = self.clip_by_tensor(pred, epsilon, 1.0-epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output


    def forward(self, l, input, targets=None, y_ture=None):
        """
            l:有效特征层索引
            input:loss输入
            input: bs, 3*(5+num_classes), 20, 20
                   bs, 3*(5+num_classes), 40, 40
                   bs, 3*(5+num_classes), 80, 80

            targets:真实标签bs, num_gt, 5
        """

        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 将anchor匹配到特征图大小
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 调整输入大小 bs, 3 * (5+num_classes), 20, 20 => bs, 3, 5 + num_classes, 20, 20 => batch_size, 3, 20, 20, 5 + num_classes
        prediction = input.view(bs, 3, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 计算先验框中心调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 预测结果解码
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)

        if self.cuda:
            y_true = y_true.type_as(x)
        
        loss = 0
        n = torch.sum(y_true[..., 4] == 1)
        if n != 0:
            giou = self.box_iou(pred_boxes, y_true[..., :4].type_as(x))
            loss_reg = torch.mean((1-giou)[y_true[..., 4] == 1])# 对于包含物体的GT,计算loss,loss公式推导https://blog.csdn.net/weixin_51917840/article/details/119322638
            loss_cls = torch.mean(self.BCELoss())
            loss += loss_reg* self.box_ratio + loss_cls * self.cls_ratio
            tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            tobj        = torch.zeros_like(y_true[..., 4])

        loss_conf   = torch.mean(self.BCELoss(conf, tobj))

        loss        += loss_conf * self.balance[l] * self.obj_ratio
        return loss