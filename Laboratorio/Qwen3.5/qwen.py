"""Qwen3.5-2B: MLX en Apple Silicon; PyTorch en CUDA, MPS o CPU."""

import os
import platform
import shutil
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import psutil
from huggingface_hub import constants as hf

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "Qwen/Qwen3.5-2B"
GIB = 1024**3
RAM_GIB = {"mlx": 8, "mps": 8, "cuda": 6, "cpu": 12}


def _check_resources(backend):
    if (psutil.cpu_count(logical=True) or 0) < 4:
        raise RuntimeError("Se requieren al menos 4 CPU lógicas.")

    available = psutil.virtual_memory().available / GIB
    if available < RAM_GIB[backend]:
        raise RuntimeError(
            f"RAM insuficiente: {available:.1f} GiB disponibles; "
            f"se requieren {RAM_GIB[backend]} GiB para {backend}."
        )

    # Política simple: conservar margen libre incluso con el modelo descargado.
    caches = [(hf.HF_HUB_CACHE, 10)]
    if not hf.HF_HUB_DISABLE_XET:
        caches.append((hf.HF_XET_CACHE, 1))
    for location, minimum in caches:
        path = Path(location).expanduser().resolve()
        while not path.exists():
            path = path.parent
        if shutil.disk_usage(path).free < minimum * GIB:
            raise RuntimeError(f"Se requieren {minimum} GiB libres en la unidad de {location}.")


@dataclass
class Qwen:
    model: object
    tokenizer: object
    backend: str

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        if not prompt.strip() or not 1 <= max_tokens <= 512:
            raise ValueError("Usa un prompt no vacío y entre 1 y 512 tokens de salida.")

        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokens) > 2048:
            raise ValueError("El prompt supera los 2048 tokens permitidos.")

        if self.backend == "mlx":
            import mlx.core as mx
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            with mx.stream(mx.gpu):
                return generate(
                    self.model, self.tokenizer, prompt=tokens,
                    max_tokens=max_tokens, sampler=make_sampler(temp=0), verbose=False,
                )

        import torch

        inputs = torch.tensor([tokens], dtype=torch.long, device=self.backend)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=pad_id,
            )
        return self.tokenizer.decode(output[0, len(tokens):], skip_special_tokens=True)


def load_model() -> Qwen:
    """Devuelve el modelo listo para generar; detiene la carga si faltan recursos."""
    if platform.system() == "Darwin" and platform.machine() == "arm64" and find_spec("mlx_lm"):
        import mlx.core as mx
        from mlx_lm import load

        if mx.metal.is_available():
            _check_resources("mlx")
            if mx.device_info(mx.gpu)["max_recommended_working_set_size"] < 6 * GIB:
                raise RuntimeError("Metal recomienda menos de 6 GiB de memoria de trabajo.")
            with mx.stream(mx.gpu):
                model, tokenizer = load(MODEL_ID, lazy=False)
            return Qwen(model, tokenizer, "mlx")

    import torch

    backend = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _check_resources(backend)
    if backend == "cuda" and torch.cuda.mem_get_info()[0] < 6 * GIB:
        raise RuntimeError("Se requieren al menos 6 GiB de VRAM libre.")
    if backend == "mps" and torch.mps.recommended_max_memory() < 6 * GIB:
        raise RuntimeError("Metal recomienda menos de 6 GiB de memoria de trabajo.")

    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    dtype = torch.float32 if backend == "cpu" else torch.float16
    if backend == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(backend).eval()
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    return Qwen(model, tokenizer, backend)