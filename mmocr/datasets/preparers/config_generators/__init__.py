# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDatasetConfigGenerator
from .textdet_config_generator import TextDetConfigGenerator
from .textrecog_config_generator import TextRecogConfigGenerator
from .textspotting_config_generator import TextSpottingConfigGenerator
from .trudi_textdet_config_generator import TrudiTextDetConfigGenerator
from .trudi_textrecog_config_generator import TrudiTextRecogConfigGenerator
from .trudi_textspotting_config_generator import \
    TrudiTextSpottingConfigGenerator

__all__ = [
    'BaseDatasetConfigGenerator', 'TextDetConfigGenerator',
    'TextRecogConfigGenerator', 'TextSpottingConfigGenerator',
    'TrudiTextDetConfigGenerator', 'TrudiTextRecogConfigGenerator',
    'TrudiTextSpottingConfigGenerator'
]
