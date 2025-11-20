from typing import List, Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.utils import add_prefix, SampleList, OptSampleList
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@MODELS.register_module()
class EncoderDecoderSeg(EncoderDecoder):

    def extract_feat(self, inputs: Tensor):

        backbone_out = self.backbone(inputs)

        # 假设格式为 (feats, loss1, loss2)
        if not isinstance(backbone_out, (list, tuple)) or len(backbone_out) != 3:
            raise TypeError(
                'Backbone is expected to return (feats, loss1, loss2), '
                f'but got type {type(backbone_out)} with value {backbone_out}'
            )

        feats, loss1, loss2 = backbone_out

        if isinstance(feats, Tensor):
            feats = [feats]

        return feats, loss1, loss2

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:

        feats, _, _ = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(feats, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:

        feats, _, _ = self.extract_feat(inputs)
        return self.decode_head.forward(feats)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        feats, loss1, loss2 = self.extract_feat(inputs)

        losses = dict()

        backbone_losses = {
            'loss_backbone_1': loss1,
            'loss_backbone_2': loss2,
        }

        losses.update(backbone_losses)

        loss_decode = self._decode_head_forward_train(feats, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(feats, data_samples)
            losses.update(loss_aux)

        return losses
