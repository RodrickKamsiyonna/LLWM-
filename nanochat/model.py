"""Architecture selection for training and portable checkpoint loading."""
from nanochat.gpt import GPT, GPTConfig
from nanochat.llwm import LLWM, LLWMConfig


def config_from_dict(values):
    values = dict(values)
    architecture = values.get("architecture", "gpt")
    if architecture == "llwm":
        return LLWMConfig(**values)
    if architecture == "gpt":
        values.pop("architecture", None)
        return GPTConfig(**values)
    raise ValueError(f"Unknown checkpoint architecture: {architecture}")


def create_model(config):
    return LLWM(config) if isinstance(config, LLWMConfig) else GPT(config)
