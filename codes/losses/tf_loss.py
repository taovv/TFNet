import torch
from torch import nn
from mmseg.models.losses import CrossEntropyLoss
from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class TFLoss(nn.Module):
    def __init__(self, kl_weight=0.1, bce_weight=1.0):
        super(TFLoss, self).__init__()
        self.ce_loss = CrossEntropyLoss(loss_weight=bce_weight, use_sigmoid=False)
        self.kl_loss = nn.KLDivLoss(reduction='sum')
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.kl_weight = kl_weight

    def forward(self, pred1, pred2, gt):
        n, c, h, w = pred1.shape
        gt_ = gt.squeeze(1)
        with torch.no_grad():
            mean_pred = self.sm(0.5 * pred1 + pred2)
        loss_kl = (self.kl_loss(self.log_sm(pred2), mean_pred) +
                   self.kl_loss(self.log_sm(pred1), mean_pred)) / (n * h * w)
        loss_bce = self.ce_loss(pred1, gt_, weight=None, ignore_index=255)
        return self.kl_weight * loss_kl + loss_bce
