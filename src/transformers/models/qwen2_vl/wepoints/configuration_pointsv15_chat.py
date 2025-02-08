import copy
from typing import Any, Dict

from transformers import PretrainedConfig, Qwen2Config

try:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
except ImportError:
    print('Please upgrade transformers to version 4.46.3 or higher')


class POINTSV15ChatConfig(PretrainedConfig):
    model_type = "pointsv1.5_chat"
    is_composition = True
    """Configuration class for `POINTSV1.5`."""

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        vision_config = kwargs.pop("vision_config", None)
        llm_config = kwargs.pop("llm_config", None)
        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        if isinstance(llm_config, dict):
            self.llm_config = Qwen2Config(**llm_config)
        else:
            self.llm_config = llm_config

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        return output
