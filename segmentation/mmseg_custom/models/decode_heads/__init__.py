# Copyright (c) OpenMMLab. All rights reserved.
from .mask2former_head import Mask2FormerHead
from .mask2former_soft_head import Mask2FormerSoftHead
from .uper_head_custom import UPerHeadCustom

__all__ = [
    'Mask2FormerHead',
    'UPerHeadCustom',
    'Mask2FormerSoftHead',
]
