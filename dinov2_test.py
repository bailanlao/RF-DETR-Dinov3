from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge, RFDETRMediumV3, RFDETRNanoV3,RFDETRMediumV3Plus
import torch
from rfdetr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import torch
import torch.nn as nn
import sys
import numpy as np

# model=RFDETRMedium(pretrain_weights=None,force_no_pretrain=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# core_model=model.model.model.to(device)
# encoder=core_model.backbone[0].encoder
# print(encoder)
dinov2_vitl14 = torch.hub.load('D:/__easyHelper__/dinov2-main',  'dinov2_vitl14', source='local')
print(dinov2_vitl14)
