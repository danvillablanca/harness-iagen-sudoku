from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Defaults for loading Qwen and generating answers.

    backend: auto, mlx (Apple Silicon), cuda, mps, or cpu.
    thinking: enable or disable Qwen's reasoning mode.
    Token limits constrain input prompts and generated output separately.
    """

    model_id: str = "Qwen/Qwen3.5-2B"
    thinking: bool = False
    max_output_tokens: int = 2048
    max_prompt_tokens: int = 2048
    backend: str = "auto"

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string.")
        if self.backend not in {"auto", "mlx", "cuda", "mps", "cpu"}:
            raise ValueError("backend must be auto, mlx, cuda, mps, or cpu.")
        if type(self.thinking) is not bool:
            raise ValueError("thinking must be True or False.")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer.")
        if type(self.max_prompt_tokens) is not int or self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be a positive integer.")
