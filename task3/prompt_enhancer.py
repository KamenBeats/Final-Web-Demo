"""
task3/prompt_enhancer.py — Outpainting-aware prompt enhancement using Qwen3-4B.

Unlike task2 (inpainting), outpainting prompts must convey:
  - Seamless visual continuation of the existing scene
  - Which directions are being expanded and by how much
  - Overlap context so the model blends, not cuts
  - The visual style / atmosphere inferred from the image caption

The output is a single CLIP-budget-aware positive prompt (no negative needed —
the SDXL fill pipeline doesn't accept a negative prompt with the same weight).
"""

import re
import torch

from . import model as M

# ── CLIP token budget ─────────────────────────────────────────────────────────
_CLIP_MAX = 75


def _clip_tokens(text: str) -> int:
    try:
        return len(M._pipe.tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        words = re.findall(r"[a-zA-Z0-9']+|[^a-zA-Z0-9'\s]", text)
        return int(len(words) * 1.35)


def _truncate_to_clip(text: str, max_tokens: int = _CLIP_MAX) -> str:
    if _clip_tokens(text) <= max_tokens:
        return text
    try:
        ids  = M._pipe.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        text = M._pipe.tokenizer.decode(ids, skip_special_tokens=True).strip().rstrip(",").strip()
        if "," in text:
            text = ",".join(text.split(",")[:-1]).strip()
        return text
    except Exception:
        # word-level fallback
        words = text.split()
        while words and _clip_tokens(" ".join(words)) > max_tokens:
            words.pop()
        return " ".join(words)


# ── Direction helpers ─────────────────────────────────────────────────────────
def _describe_expansion(alignment: str, resize_option: str, custom_resize_pct,
                         target_res_label: str) -> str:
    """
    Convert UI parameters into a human-readable expansion description for the LLM.
    Example: "expanding in all 4 directions, image covers ~67% of canvas (1:1 ratio)"
    """
    # Direction from alignment
    dir_map = {
        "Middle": "in all 4 directions",
        "Left":   "to the right",
        "Right":  "to the left",
        "Top":    "downward",
        "Bottom": "upward",
    }
    direction = dir_map.get(alignment, "in all directions")

    # Coverage percentage
    if resize_option == "Full":
        pct = 100
    elif resize_option == "50%":
        pct = 50
    elif resize_option == "33%":
        pct = 33
    elif resize_option == "25%":
        pct = 25
    elif resize_option == "Custom":
        try:
            pct = int(custom_resize_pct or 100)
        except Exception:
            pct = 100
    else:
        pct = 100

    expand_pct = round((100 - pct) / 2) if alignment == "Middle" else (100 - pct)

    res_info = ""
    if target_res_label not in ("Customize", ""):
        res_info = f", target canvas ratio {target_res_label}"

    return (
        f"expanding {direction}, original image covers ~{pct}% of canvas "
        f"(~{expand_pct}% expansion per expanded edge{res_info})"
    )


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are an expert AI outpainting prompt engineer for Stable Diffusion XL.

Your job: given a visual scene description + expansion parameters, write ONE seamless
outpainting prompt that tells the model to naturally extend the image.

STRICT OUTPUT RULES:
- Output ONLY the prompt text — no explanation, no JSON, no quotes, no markdown
- Write 2–3 flowing descriptive sentences (NOT comma-separated tags)
- Total: 60–75 words
- Do NOT mention "outpainting", "expanding", "canvas", "AI", or any technical terms
- Do NOT start with "I" or "The prompt is"

PROMPT STRUCTURE (follow this order):
1. "Seamless photorealistic continuation of the same [SCENE + ATMOSPHERE], preserving
   the exact existing design and atmosphere."
2. "Extend the [KEY VISUAL ELEMENTS] naturally beyond the current frame in the
   [DIRECTION] direction."
3. "Maintain the same [MATERIALS / LIGHTING / STYLE / PERSPECTIVE] with consistent
   shadows, textures, depth, and realistic proportions throughout the newly generated area."

DIRECTION GUIDANCE — use the expansion info provided:
- "all directions" → "in every direction, widening the scene on all sides"
- "to the right"   → "to the right side, extending the rightward view"
- "to the left"    → "to the left side, widening the leftward perspective"
- "downward"       → "below the existing scene, extending the lower area"
- "upward"         → "above the existing scene, extending skyward or toward the ceiling"
"""


# ── Qwen call ─────────────────────────────────────────────────────────────────
def _call_qwen(user_message: str) -> str:
    """Run Qwen3-4B inference. Returns raw decoded text."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_message},
    ]
    text = M.qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = M.qwen_tokenizer(text, return_tensors="pt").to(M.qwen_model.device)
    with torch.no_grad():
        out = M.qwen_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return M.qwen_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _clean_llm_output(text: str) -> str:
    """Strip leftover <think> tags, leading/trailing quotes."""
    text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\s*think\s*>.*",                   "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>",                            "", text)
    return text.strip().strip("\"'").strip()


# ── Rule-based fallback ───────────────────────────────────────────────────────
def _rule_fallback(raw_prompt: str, expansion_desc: str) -> str:
    """Minimal template if Qwen is unavailable."""
    scene = raw_prompt.strip().rstrip(".").strip() if raw_prompt.strip() else "the scene"
    return (
        f"Seamless photorealistic continuation of the same {scene}, "
        f"preserving the exact existing design and atmosphere. "
        f"Extend all key visual elements naturally beyond the current frame, "
        f"{expansion_desc.replace('expanding ', '')}. "
        f"Keep the same architecture, materials, lighting direction, perspective, "
        f"and aesthetic with consistent shadows, textures, and realistic depth."
    )


# ── Public API ────────────────────────────────────────────────────────────────
def enhance_prompt(
    raw_prompt: str,
    image_caption: str = "",
    alignment: str = "Middle",
    resize_option: str = "Full",
    custom_resize_pct = 100,
    target_res_label: str = "1:1",
    overlap_left: float = 10,
    overlap_right: float = 10,
    overlap_top: float = 10,
    overlap_bottom: float = 10,
) -> str:
    """
    Convert user raw prompt → outpainting-aware enhanced prompt.

    Args:
        raw_prompt:       User's freeform description (can be empty).
        image_caption:    Auto-generated BLIP/visual caption of uploaded image.
        alignment:        Canvas placement (Middle / Left / Right / Top / Bottom).
        resize_option:    Full / 50% / 33% / 25% / Custom.
        custom_resize_pct: Numeric % when resize_option == "Custom".
        target_res_label: "1:1" / "16:9" / "9:16" / "Customize".
        overlap_*:        Overlap band per side (%).

    Returns:
        Enhanced prompt string, CLIP-truncated to 75 tokens.
    """
    expansion_desc = _describe_expansion(
        alignment, resize_option, custom_resize_pct, target_res_label
    )

    # Build the scene description from caption + user text
    scene_parts = []
    if image_caption.strip():
        scene_parts.append(image_caption.strip())
    if raw_prompt.strip():
        scene_parts.append(raw_prompt.strip())
    scene_desc = ". ".join(scene_parts) if scene_parts else "the scene"

    # Overlap context hint
    overlap_sides = []
    if overlap_left  > 0: overlap_sides.append("left")
    if overlap_right > 0: overlap_sides.append("right")
    if overlap_top   > 0: overlap_sides.append("top")
    if overlap_bottom > 0: overlap_sides.append("bottom")
    overlap_hint = (
        f"Overlap bands on: {', '.join(overlap_sides)} edges "
        f"(model can see the existing content in the overlap zone to blend seamlessly)."
        if overlap_sides else ""
    )

    user_message = (
        f"Scene description: {scene_desc}\n\n"
        f"Expansion parameters: {expansion_desc}\n"
        + (f"{overlap_hint}\n" if overlap_hint else "") +
        f"\nGenerate the outpainting prompt."
    )

    print(f"[Task3 Enhance] {expansion_desc}")
    print(f"[Task3 Enhance] scene: {scene_desc[:100]}")

    # Ensure Qwen is available
    ensure_qwen_loaded()
    if M.qwen_tokenizer is None or M.qwen_model is None:
        print("[Task3 Enhance] Qwen not loaded → rule-based fallback")
        result = _rule_fallback(scene_desc, expansion_desc)
        return _truncate_to_clip(result)

    try:
        raw_out = _call_qwen(user_message)
        enhanced = _clean_llm_output(raw_out)
        if not enhanced:
            raise ValueError("Empty LLM output")
        print(f"[Task3 Enhance] → {enhanced[:120]}")
    except Exception as e:
        print(f"[Task3 Enhance] Qwen error: {e} → rule-based fallback")
        enhanced = _rule_fallback(scene_desc, expansion_desc)

    return _truncate_to_clip(enhanced)


# ── Qwen GPU / CPU management (mirrors task2 pattern) ────────────────────────
def ensure_qwen_loaded():
    """Load Qwen3-4B vào CPU RAM riêng cho Task 3 (không dùng chung với task2)."""
    if M.qwen_tokenizer is not None and M.qwen_model is not None:
        return  # already loaded

    # Load independently — Task 3 owns its own Qwen instance
    print("[Task3 Enhance] Loading Qwen3-4B → CPU RAM…")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        def _rmhook(m, recurse=False):
            for attr in ("_hf_hook", "_forward_hooks"):
                try:
                    delattr(m, attr)
                except AttributeError:
                    pass
            if recurse:
                for child in m.children():
                    _rmhook(child, recurse=True)

        tok = AutoTokenizer.from_pretrained(M.QWEN_ID, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(
            M.QWEN_ID,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _rmhook(mod, recurse=True)
        mod.eval()
        M.qwen_tokenizer = tok
        M.qwen_model     = mod
        print("[Task3 Enhance] Qwen3-4B loaded into CPU RAM")
    except Exception as e:
        print(f"[Task3 Enhance] Failed to load Qwen: {e}")


def move_qwen_to_gpu():
    """Move Qwen to GPU. Called before inference."""
    if M.qwen_model is None:
        return
    device = M.T3_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        M.qwen_model.to(device)
        print(f"[Task3 Enhance] Qwen → {device}")
    except Exception as e:
        print(f"[Task3 Enhance] move_qwen_to_gpu failed: {e}")


def move_qwen_to_cpu():
    """Move Qwen back to CPU. Called after Qwen inference, before diffusion."""
    if M.qwen_model is None:
        return
    try:
        M.qwen_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Task3 Enhance] Qwen → CPU")
    except Exception as e:
        print(f"[Task3 Enhance] move_qwen_to_cpu failed: {e}")
