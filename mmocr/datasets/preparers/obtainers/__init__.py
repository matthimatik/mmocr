# Copyright (c) OpenMMLab. All rights reserved.
from .aws_s3_obtainer import AWSS3Obtainer
from .naive_data_obtainer import NaiveDataObtainer
from .trudi_obtainer import TrudiObtainer

__all__ = ['NaiveDataObtainer', 'AWSS3Obtainer', 'TrudiObtainer']
