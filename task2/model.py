"""Task 2 — Model loading (SDXL Inpainting + ControlNet Depth + DPT + Qwen3-4B)."""

import os
import threading
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from transformers import DPTImageProcessor, DPTForDepthEstimation, AutoTokenizer, AutoModelForCausalLM

_T2_DIR = Path(__file__).parent

T2_DEVICE        = os.environ.get("T2_DEVICE",        "cuda:0")
SDXL_INPAINT_ID  = os.environ.get("T2_SDXL_ID",       "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_ID    = os.environ.get("T2_CONTROLNET_ID",  "diffusers/controlnet-depth-sdxl-1.0")
DPT_ID           = os.environ.get("T2_DPT_ID",         "Intel/dpt-hybrid-midas")
QWEN_ID          = os.environ.get("T2_QWEN_ID",         "Qwen/Qwen3-4B")
LORA_PATH        = os.environ.get("T2_LORA_PATH",      str(_T2_DIR / "ckp" / "pytorch_lora_weights.safetensors"))

# ── Globals ───────────────────────────────────────────────────────────────────
pipe           = None   # StableDiffusionXLControlNetInpaintPipeline
dp             = None   # DPTImageProcessor (CPU only, no .to() needed)
dm             = None   # DPTForDepthEstimation
qwen_tokenizer = None   # AutoTokenizer
qwen_model     = None   # AutoModelForCausalLM
_loading       = False
# ── Loading ───────────────────────────────────────────────────────────────────

def load_to_ram():
    """Load tất cả models vào CPU RAM rồi register với ModelManager."""
    global pipe, dp, dm, qwen_tokenizer, qwen_model, _loading

    if pipe is not None:
        return
    if _loading:
        return
    _loading = True

    try:
        print("[Task2] Loading ControlNet Depth → CPU RAM…")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_ID, torch_dtype=torch.float16, low_cpu_mem_usage=False
        )

        print("[Task2] Loading SDXL Inpainting pipeline → CPU RAM…")
        _pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            SDXL_INPAINT_ID,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=False,
        )
        # The pipeline loads on CPU by default; avoid redundant .to('cpu') calls that
        # can fail on meta tensors during initialization.
        _pipe.vae.enable_slicing()
        _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            _pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
        try:
            _pipe.enable_xformers_memory_efficient_attention()
            print("[Task2] xformers attention enabled.")
        except Exception:
            pass
        _pipe.set_progress_bar_config(disable=True)

        if LORA_PATH and Path(LORA_PATH).exists():
            print(f"[Task2] Loading LoRA from {LORA_PATH}…")
            _pipe.load_lora_weights(LORA_PATH)
            _pipe.set_adapters(["default_0"], adapter_weights=[1.0])
            print("[Task2] LoRA loaded with adapter weight=1.0")
        else:
            print(f"[Task2] No LoRA found at {LORA_PATH}, skipping.")

        print("[Task2] Loading DPT depth model → CPU RAM…")
        _dp = DPTImageProcessor.from_pretrained(DPT_ID)
        _dm = DPTForDepthEstimation.from_pretrained(
            DPT_ID, torch_dtype=torch.float16, device_map={"" : "cpu"}
        )
        from accelerate.hooks import remove_hook_from_module as _rmhook
        _rmhook(_dm, recurse=True)
        _dm.eval()

        print("[Task2] Loading Qwen3-4B → CPU RAM…")
        _qwen_tok = AutoTokenizer.from_pretrained(QWEN_ID)
        _qwen_mod = AutoModelForCausalLM.from_pretrained(
            QWEN_ID, torch_dtype=torch.float16, device_map={"" : "cpu"},
        )
        _rmhook(_qwen_mod, recurse=True)  # remove accelerate hooks → plain .to() works
        _qwen_mod.eval()

        pipe           = _pipe
        dp             = _dp
        dm             = _dm
        qwen_tokenizer = _qwen_tok
        qwen_model     = _qwen_mod
        print("[Task2] All weights in CPU RAM — ready for fast GPU transfer.")

        # Register chỉ pipe với ModelManager (dm/qwen sẽ lazy load on-demand trong inference)
        from model_manager import manager

        class _PipelineWrapper:
            """Wrapper để ModelManager có thể move Diffusers pipeline."""
            def __init__(self, pipeline):
                self._pipe = pipeline

            def parameters(self):
                """Expose parameters từ underlying pipeline."""
                try:
                    for param in self._pipe.parameters():
                        yield param
                except (TypeError, AttributeError):
                    # If parameters() doesn't work, at least yield something so we don't error
                    pass
            
            def to(self, device, **kwargs):
                # Diffusers pipeline dùng cú pháp khác
                device = torch.device(device)
                try:
                    self._pipe.to(device)
                except NotImplementedError as e:
                    # Handle meta tensors: DiffusionPipeline is NOT a nn.Module so
                    # it has no to_empty(). Must iterate components individually,
                    # or reload directly to the target device.
                    if "meta tensor" in str(e).lower() or "Cannot copy out of meta tensor" in str(e):
                        print(f"[Task2] Warning: meta tensor detected, reloading pipeline directly to {device}")
                        import gc
                        old_pipe = self._pipe
                        try:
                            _cn = ControlNetModel.from_pretrained(
                                CONTROLNET_ID, torch_dtype=torch.float16
                            )
                            _new = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                                SDXL_INPAINT_ID,
                                controlnet=_cn,
                                torch_dtype=torch.float16,
                                variant="fp16",
                            )
                            _new.vae.enable_slicing()
                            _new.scheduler = DPMSolverMultistepScheduler.from_config(
                                _new.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
                            )
                            try:
                                _new.enable_xformers_memory_efficient_attention()
                            except Exception:
                                pass
                            _new.set_progress_bar_config(disable=True)
                            if LORA_PATH and Path(LORA_PATH).exists():
                                _new.load_lora_weights(LORA_PATH)
                                _new.set_adapters(["default_0"], adapter_weights=[1.0])
                            _new.to(device)
                            self._pipe = _new
                            # Sync global M.pipe so inference code uses the correct pipeline
                            import task2.model as _M
                            _M.pipe = _new
                            del old_pipe
                            gc.collect()
                            torch.cuda.empty_cache()
                            print(f"[Task2] Pipeline reloaded successfully to {device}")
                        except Exception as reload_err:
                            print(f"[Task2] Reload failed: {reload_err}")
                            raise RuntimeError(
                                f"Cannot move pipeline to {device} (meta tensors present, reload also failed): {reload_err}"
                            ) from e
                    else:
                        raise
                return self

        manager.register("task2", {"pipe": _PipelineWrapper(pipe)})

    except Exception as e:
        print(f"[Task2] ERROR loading pipeline: {e}")
        import traceback; traceback.print_exc()
    finally:
        _loading = False


def preload_to_cpu():
    """Gọi lúc startup trong background thread."""
    t = threading.Thread(target=load_to_ram, daemon=True, name="task2-preload")
    t.start()


def wait_until_loaded(timeout: float = 180.0):
    """Block cho đến khi load xong (dùng bởi inference nếu cần)."""
    t0 = time.time()
    while _loading:
        if time.time() - t0 > timeout:
            raise RuntimeError("[Task2] Timeout waiting for model load.")
        time.sleep(0.5)


def ensure_qwen_loaded(timeout: float = 180.0):
    """Ensure Qwen model is available (wait for preload if still loading)."""
    wait_until_loaded(timeout)
    if qwen_model is None or qwen_tokenizer is None:
        # Try loading if not loaded yet
        load_to_ram()
        wait_until_loaded(timeout)
