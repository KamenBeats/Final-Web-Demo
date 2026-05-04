"""Task 3 — Model loading + GPU management."""

import os
import threading
from pathlib import Path

import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict

from model_manager import manager

# ── Config ────────────────────────────────────────────────────────────────────
_T3_DIR = Path(__file__).parent

CONTROLNET_CONFIG  = os.environ.get("T3_CONTROLNET_CONFIG",  str(_T3_DIR / "config_promax.json"))
CONTROLNET_WEIGHTS = os.environ.get("T3_CONTROLNET_WEIGHTS", str(_T3_DIR / "controlnet_weight_promax.safetensors"))
BASE_MODEL         = os.environ.get("T3_BASE_MODEL",         "SG161222/RealVisXL_V5.0_Lightning")
VAE_MODEL          = os.environ.get("T3_VAE_MODEL",          "madebyollin/sdxl-vae-fp16-fix")
LORA_PATH          = os.environ.get("T3_LORA_PATH",          str(_T3_DIR / "lora_best"))
QWEN_ID            = os.environ.get("T3_QWEN_ID",            "Qwen/Qwen3-4B")
T3_DEVICE          = os.environ.get("T3_DEVICE",             "")   # auto-detected below
T3_DTYPE           = os.environ.get("T3_DTYPE",              "")

# ── Globals ───────────────────────────────────────────────────────────────────
_pipe          = None   # StableDiffusionXLFillPipeline on CPU RAM after load
_loading       = False  # guard against concurrent loads
qwen_tokenizer = None   # AutoTokenizer
qwen_model     = None   # AutoModelForCausalLM
_lora_deltas: dict = {}       # precomputed (alpha/r) * B@A for each layer (CPU)
_lora_current_scale: float = 0.0  # currently applied LoRA scale


def _load_to_ram():
    """Load all weights from disk into CPU RAM (no VRAM used)."""
    global _pipe, _loading, T3_DEVICE, T3_DTYPE

    if _pipe is not None:
        return
    if _loading:
        return
    _loading = True

    try:
        if not T3_DEVICE:
            T3_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if not T3_DTYPE:
            T3_DTYPE = "fp16" if (T3_DEVICE == "cuda") else "fp32"

        missing = []
        if not Path(CONTROLNET_WEIGHTS).exists():
            missing.append(f"ControlNet weights: {CONTROLNET_WEIGHTS}")
        for mod_file in ("controlnet_union.py", "pipeline_fill_sd_xl.py"):
            if not (_T3_DIR / mod_file).exists():
                missing.append(mod_file)
        if missing:
            print("[Task3] Skipping model load — missing files:")
            for m in missing:
                print(f"  - {m}")
            return

        from .controlnet_union import ControlNetModel_Union
        from .pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

        dtype = (torch.float16 if T3_DTYPE == "fp16"
                 else torch.bfloat16 if T3_DTYPE == "bf16"
                 else torch.float32)

        print("[Task3] Loading ControlNet Union → CPU RAM…")
        config     = ControlNetModel_Union.load_config(CONTROLNET_CONFIG)
        controlnet = ControlNetModel_Union.from_config(config)
        sd         = load_state_dict(CONTROLNET_WEIGHTS)
        incompatible = controlnet.load_state_dict(sd, strict=False, assign=True)
        if incompatible.missing_keys:
            print(f"[Task3] ControlNet missing keys ({len(incompatible.missing_keys)}): "
                  f"{incompatible.missing_keys[:5]}")
        if incompatible.unexpected_keys:
            print(f"[Task3] ControlNet unexpected keys ({len(incompatible.unexpected_keys)}): "
                  f"{incompatible.unexpected_keys[:5]}")
        controlnet = controlnet.to(dtype=dtype)

        print("[Task3] Loading VAE → CPU RAM…")
        vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype)

        print("[Task3] Loading SDXL Fill pipeline → CPU RAM…")
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            vae=vae,
            controlnet=controlnet,
            variant="fp16" if (dtype == torch.float16) else None,
            low_cpu_mem_usage=False,
        )
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        if LORA_PATH and Path(LORA_PATH).exists():
            try:
                from safetensors.torch import load_file as _st_load
                print(f"[Task3] Loading LoRA from {LORA_PATH}…")

                adapter_safetensors = Path(LORA_PATH) / "adapter_model.safetensors"
                adapter_bin = Path(LORA_PATH) / "adapter_model.bin"
                if adapter_safetensors.exists():
                    adapter_weights = _st_load(str(adapter_safetensors), device="cpu")
                elif adapter_bin.exists():
                    adapter_weights = torch.load(str(adapter_bin), map_location="cpu")
                else:
                    raise FileNotFoundError(f"No adapter weights found in {LORA_PATH}")

                # Read lora_alpha from adapter_config.json
                import json as _json
                cfg_path = Path(LORA_PATH) / "adapter_config.json"
                lora_alpha = 32
                if cfg_path.exists():
                    with open(cfg_path) as _f:
                        lora_alpha = _json.load(_f).get("lora_alpha", 32)

                # Collect lora_A and lora_B weights
                # Key format: base_model.model.{unet_path}.lora_A.default.weight
                lora_a_w, lora_b_w = {}, {}
                for k, v in adapter_weights.items():
                    if ".lora_A.weight" in k:
                        unet_key = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
                        lora_a_w[unet_key] = v.to(dtype=dtype)
                    elif ".lora_B.weight" in k:
                        unet_key = k.replace("base_model.model.", "").replace(".lora_B.weight", ".weight")
                        lora_b_w[unet_key] = v.to(dtype=dtype)

                # Precompute deltas: (alpha/r) * B @ A — stored on CPU, applied in-place dynamically
                global _lora_deltas
                _lora_deltas = {}
                unet_param_names = {n for n, _ in pipe.unet.named_parameters()}
                for key in lora_a_w:
                    if key not in lora_b_w or key not in unet_param_names:
                        continue
                    A = lora_a_w[key]   # [r, in_features]
                    B = lora_b_w[key]   # [out_features, r]
                    if A.ndim != 2 or B.ndim != 2:
                        continue  # skip conv layers
                    r = A.shape[0]
                    _lora_deltas[key] = (B @ A) * (lora_alpha / r)
                print(f"[Task3] LoRA ready: {len(_lora_deltas)} layers precomputed "
                      f"(alpha={lora_alpha}). Default scale=0.0.")
            except Exception as e:
                print(f"[Task3] LoRA loading failed: {e}")
                import traceback; traceback.print_exc()

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Task3] xformers attention enabled.")
        except Exception:
            pass

        _pipe = pipe
        print("[Task3] Pipeline in CPU RAM — registering with ModelManager…")
        manager.register("task3", {"pipe": _pipe})
        print("[Task3] Registered with ModelManager.")
    except Exception as e:
        print(f"[Task3] ERROR loading pipeline: {e}")
        import traceback; traceback.print_exc()
    finally:
        _loading = False


def preload_to_cpu():
    """Start background preload into CPU RAM at startup."""
    t = threading.Thread(target=_load_to_ram, daemon=True, name="task3-preload")
    t.start()


def wait_until_loaded(timeout: float = 180.0):
    """Block until pipeline is in CPU RAM (or timeout)."""
    import time
    deadline = time.time() + timeout
    while _pipe is None and time.time() < deadline:
        time.sleep(0.5)
    return _pipe is not None


def set_lora_scale(scale: float):
    """Apply LoRA at the given scale by updating UNet weights in-place.
    scale=0.0 → base model only; scale=1.0 → full LoRA effect.
    """
    global _lora_current_scale
    if _pipe is None or not _lora_deltas:
        _lora_current_scale = scale
        return
    delta_scale = scale - _lora_current_scale
    if abs(delta_scale) < 1e-6:
        return
    try:
        param_dict = {n: p for n, p in _pipe.unet.named_parameters()}
        for key, delta in _lora_deltas.items():
            if key in param_dict:
                p = param_dict[key]
                with torch.no_grad():
                    p.data.add_(delta.to(device=p.device, dtype=p.dtype) * delta_scale)
        _lora_current_scale = scale
        print(f"[Task3] LoRA scale set to {scale}")
    except Exception as e:
        print(f"[Task3] Could not set LoRA scale: {e}")
