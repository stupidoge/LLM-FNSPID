# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
class ContrastiveLoss(torch.nn.Module):
   def __init__(self, margin=0.5):
       super(ContrastiveLoss, self).__init__()
       self.margin = margin
       self.cos = nn.CosineSimilarity(dim=2)


   def forward(self, enc_out, positive_samples, negative_samples):
       # Compute cosine similarity between encoded output and positive samples
       positive_sim = self.cos(enc_out, positive_samples)


       # Compute cosine similarity between encoded output and negative samples
       negative_sim = self.cos(enc_out, negative_samples)


       # Compute contrastive loss
       loss = torch.mean(torch.clamp(self.margin - positive_sim + negative_sim, min=0))


       return loss

# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self):
#         super(ContrastiveLoss, self).__init__()
#         # create CrossEntropyLoss
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.criterion = nn.CrossEntropyLoss()
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07)).to(self.device)
#         self.criterion = self.criterion.to(self.device)
#
#     def forward(self, image_features, text_features):
#         batch_size = len(image_features)
#         # print(print("contrastive input", image_features.shape, text_features.shape))
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
#
#         logit_scale = self.logit_scale.exp()
#         # class
#         labels = np.arange(batch_size)
#         labels = torch.LongTensor(labels).to(self.device)
#         logits1 = logit_scale * image_features @ text_features.t()
#         logits2 = logit_scale * text_features @ image_features.t()
#
#         # labels size = logits size
#         loss_i = self.criterion(logits1, labels)
#         loss_t = self.criterion(logits2, labels)
#         loss = loss_i + loss_t
#
#         return loss
#
#
# class ContrastiveLoss3D(torch.nn.Module):
#     def __init__(self):
#         super(ContrastiveLoss, self).__init__()
#         # create CrossEntropyLoss
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.criterion = nn.CrossEntropyLoss()
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07)).to(self.device)
#         self.criterion = self.criterion.to(self.device)
#
#     def forward(self, image_features, text_features):
#         batch_size = len(image_features)
#         # print(print("contrastive input", image_features.shape, text_features.shape))
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
#
#         logit_scale = self.logit_scale.exp()
#         # class
#         labels = np.arange(batch_size)
#         labels = torch.LongTensor(labels).to(self.device)
#         logits1 = logit_scale * image_features @ text_features.t()
#         logits2 = logit_scale * text_features @ image_features.t()
#
#         # labels size = logits size
#         loss_i = self.criterion(logits1, labels)
#         loss_t = self.criterion(logits2, labels)
#         loss = loss_i + loss_t
#
#         return loss