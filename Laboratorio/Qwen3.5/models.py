"""Public entry point for loading a model."""

from config import ModelConfig
from qwen import load_model as load_qwen


def load_model(config: ModelConfig | None = None):
    return load_qwen(config or ModelConfig())
