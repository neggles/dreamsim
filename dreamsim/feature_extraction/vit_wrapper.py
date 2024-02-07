from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel


class ViTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ViTModel(PreTrainedModel):
    config_class = ViTConfig

    def __init__(self, model: nn.Module, config: dict):
        super().__init__(config)
        self.model = model
        self.blocks = model.blocks

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
