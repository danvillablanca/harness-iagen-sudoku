# qwen.py

"""Qwen3.5-2B: MLX on Apple Silicon; PyTorch on CUDA, MPS, or CPU."""

import os
import platform
import shutil
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import psutil
from huggingface_hub import constants as hf

from config import ModelConfig

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "Qwen/Qwen3.5-2B"
GIB = 1024**3
RAM_GIB = {"mlx": 8, "mps": 8, "cuda": 6, "cpu": 12}

MAX_PROMPT_TOKENS = 2048


def _check_resources(backend):
    minimum_cpus = 4 if backend == "cpu" else 2
    if (psutil.cpu_count(logical=True) or 0) < minimum_cpus:
        raise RuntimeError(f"At least {minimum_cpus} logical CPUs are required for {backend}.")

    available = psutil.virtual_memory().available / GIB
    if available < RAM_GIB[backend]:
        raise RuntimeError(
            f"Insufficient RAM: {available:.1f} GiB available; "
            f"{RAM_GIB[backend]} GiB required for {backend}."
        )

    caches = [(hf.HF_HUB_CACHE, 10)]
    if not hf.HF_HUB_DISABLE_XET:
        caches.append((hf.HF_XET_CACHE, 1))

    for location, minimum in caches:
        path = Path(location).expanduser().resolve()
        while not path.exists():
            path = path.parent

        if shutil.disk_usage(path).free < minimum * GIB:
            raise RuntimeError(
                f"At least {minimum} GiB of free disk space "
                f"is required on the drive containing {location}."
            )


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str | None
    backend: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


@dataclass
class Qwen:
    model: object
    tokenizer: object
    backend: str
    config: ModelConfig

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        enable_thinking: bool | None = None,
        shots: tuple = (),
    ) -> str:
        """Return only the generated text, preserving the original interface."""
        return self.generate_with_stats(
            prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            shots=shots,
        ).text

    def generate_with_stats(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        enable_thinking: bool | None = None,
        shots: tuple = (),
    ) -> GenerationResult:
        """Return text and token counts measured during generation.

        `shots` is a sequence of (user, assistant) example turns placed
        before the prompt.
        """
        max_tokens = self.config.max_output_tokens if max_tokens is None else max_tokens
        enable_thinking = self.config.thinking if enable_thinking is None else enable_thinking

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty text prompt.")

        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError(
                "max_tokens must be a positive integer."
            )

        if type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be True or False.")

        messages = []
        for user_text, assistant_text in shots:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": prompt})

        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        prompt_tokens = len(tokens)

        if prompt_tokens > self.config.max_prompt_tokens:
            raise ValueError(
                f"The prompt contains {prompt_tokens} tokens; "
                f"the limit is {self.config.max_prompt_tokens}."
            )

        if self.backend == "mlx":
            import mlx.core as mx
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            parts = []
            last_chunk = None

            with mx.stream(mx.gpu):
                for chunk in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=tokens,
                    max_tokens=max_tokens,
                    sampler=make_sampler(temp=0),
                ):
                    parts.append(chunk.text)
                    last_chunk = chunk

            if last_chunk is None:
                raise RuntimeError("MLX returned no generation output.")

            # Use the runtime counter, including special tokens when counted.
            return GenerationResult(
                text="".join(parts),
                prompt_tokens=prompt_tokens,
                generated_tokens=int(last_chunk.generation_tokens),
                finish_reason=getattr(last_chunk, "finish_reason", None),
                backend=self.backend,
            )

        import torch

        inputs = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.backend,
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
                return_dict_in_generate=False,
            )

        # Count actual continuation IDs before decoding removes special tokens.
        continuation_ids = output[0, prompt_tokens:].tolist()
        generated_tokens = len(continuation_ids)

        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = set()
        elif isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids)

        ended_with_eos = (
            bool(continuation_ids)
            and continuation_ids[-1] in eos_ids
        )
        reached_limit = generated_tokens >= max_tokens

        # PyTorch finish reasons are inferred from the returned token IDs.
        if ended_with_eos and reached_limit:
            finish_reason = "eos_or_length"
        elif ended_with_eos:
            finish_reason = "stop"
        elif reached_limit:
            finish_reason = "length"
        else:
            finish_reason = None

        # Keep the closing thinking marker so the parser can separate reasoning
        # from the final answer, while removing other model control tokens.
        text = self.tokenizer.decode(continuation_ids, skip_special_tokens=False)
        for special_token in self.tokenizer.all_special_tokens:
            if special_token != "</think>":
                text = text.replace(special_token, "")

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
            backend=self.backend,
        )


def load_model(config: ModelConfig | None = None) -> Qwen:
    """Load the model after checking available resources."""
    config = config or ModelConfig()
    if (
        config.backend in ("auto", "mlx")
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and find_spec("mlx_lm")
    ):
        import mlx.core as mx
        from mlx_lm import load

        if mx.metal.is_available():
            _check_resources("mlx")

            if (
                mx.device_info(mx.gpu)["max_recommended_working_set_size"]
                < 6 * GIB
            ):
                raise RuntimeError(
                    "Metal recommends less than 6 GiB of working memory."
                )

            with mx.stream(mx.gpu):
                model, tokenizer = load(config.model_id, lazy=False)

            return Qwen(model, tokenizer, "mlx", config)

    if config.backend == "mlx":
        raise RuntimeError("MLX requires mlx_lm and an available Apple Silicon GPU.")

    import torch

    backend = config.backend if config.backend != "auto" else (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    _check_resources(backend)

    if backend == "cuda" and torch.cuda.mem_get_info()[0] < 6 * GIB:
        raise RuntimeError("At least 6 GiB of free VRAM is required.")

    if backend == "mps" and torch.mps.recommended_max_memory() < 6 * GIB:
        raise RuntimeError(
            "Metal recommends less than 6 GiB of working memory."
        )

    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    dtype = torch.float32 if backend == "cpu" else torch.float16
    if (backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            and torch.cuda.is_bf16_supported()):
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = (
        Qwen3_5ForCausalLM.from_pretrained(config.model_id, dtype=dtype)
        .to(backend)
        .eval()
    )

    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()

    return Qwen(model, tokenizer, backend, config)