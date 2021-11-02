"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import torch
import torch.nn as nn

class SoftIOULoss(nn.Module):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, weight=1.):
        super(SoftIOULoss, self).__init__()
        self.activation = nn.Softmax2d()
        self.weight = weight

    def setWeights(self, weights):
        self.weight = weights

    def forward(self, y_preds, y_truths, eps=1e-8):
        '''
        :param y_preds: [bs,num_classes,768,1024]
        :param y_truths: [bs,num_calsses,768,1024]
        :param eps:
        :return:
        '''
        bs = y_preds.size(0)
        num_classes = y_preds.size(1)
        ious_bs = torch.zeros(bs, num_classes).to(y_preds.device)
        for idx in range(bs):
            y_pred = y_preds[idx]  # [num_classes,768,1024]
            y_truth = y_truths[idx]  # [num_classes,768,1024]
            intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(1, 2)) + eps / 2
            union = torch.sum(torch.mul(y_pred, y_pred), dim=(1, 2)) + torch.sum(torch.mul(y_truth, y_truth),
                                                                                 dim=(1, 2)) + eps

            ious_sub = intersection / (union - intersection)
            ious_bs[idx] = ious_sub

        ious = torch.mean(ious_bs, dim=0)
        iou = torch.mean(ious)
        iou_loss = 1 - iou
        return iou_loss * self.weight