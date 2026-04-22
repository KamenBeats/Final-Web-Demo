"""Task 2 — Inference helpers: depth, mask, inpaint, enhance prompt."""

import re
import time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from . import model as M
from .prompt_enhancer import enhance_prompt  # noqa: F401  (re-exported for ui.py)


MAX_T2_LONG = 3840


# ── Depth estimation ──────────────────────────────────────────────────────────

def estimate_depth(pil_image: Image.Image) -> Image.Image:
    W, H = pil_image.size
    M.dm.to("cuda:0")
    try:
        with torch.no_grad():
            inputs = M.dp(images=pil_image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to("cuda:0", torch.float16) for k, v in inputs.items()}
            pred = M.dm(**inputs).predicted_depth
        dt = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze()
        dn = dt.cpu().numpy()
    finally:
        M.dm.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dn = (dn - dn.min()) / (dn.max() - dn.min() + 1e-8) * 255.0
    return Image.fromarray(np.stack([dn.astype(np.uint8)] * 3, axis=-1))


# ── Mask helpers ──────────────────────────────────────────────────────────────

def regularize_mask(mask_image, target_size, roundness_threshold=0.75):
    mask_np = np.array(mask_image.convert("L").resize(target_size, Image.NEAREST))
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_image.convert("L").resize(target_size, Image.NEAREST)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0
    canvas = np.zeros_like(binary)
    if circularity >= roundness_threshold and len(contour) >= 5:
        (cx, cy), (ma, mb), angle = cv2.fitEllipse(contour)
        if min(ma, mb) / max(ma, mb) > 0.90:
            cv2.circle(canvas, (int(cx), int(cy)), int(max(ma, mb) / 2), 255, -1)
        else:
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(ma / 2), int(mb / 2)),
                        angle, 0, 360, 255, -1)
    else:
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
        cv2.fillPoly(canvas, [box], 255)
    return Image.fromarray(canvas)


def extract_mask(editor_val) -> Image.Image | None:
    if editor_val is None:
        return None
    layers = editor_val.get("layers") or []
    if not layers or layers[0] is None:
        return None
    layer = layers[0]
    arr = np.array(layer)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        mask_arr = np.where(alpha > 10, 255, 0).astype(np.uint8)
        return Image.fromarray(mask_arr, mode="L")
    return layer if isinstance(layer, Image.Image) else Image.fromarray(arr)


def cap_image(img: Image.Image, max_long: int = MAX_T2_LONG) -> Image.Image:
    if img is None:
        return img
    if max(img.width, img.height) > max_long:
        s = max_long / max(img.width, img.height)
        return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img


# ── Blending helpers ──────────────────────────────────────────────────────────

def _alpha_blend(src, dst, mask, feather=15):
    mask_f = cv2.GaussianBlur(mask.astype(float), (feather * 2 + 1, feather * 2 + 1), 0) / 255.0
    mask_f = mask_f[:, :, np.newaxis]
    return (src.astype(float) * mask_f + dst.astype(float) * (1 - mask_f)).astype(np.uint8)


def _safe_poisson_blend(src, dst, mask, margin=10):
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    H, W = mask.shape
    if (mask_eroded[0].max() > 0 or mask_eroded[-1].max() > 0 or
            mask_eroded[:, 0].max() > 0 or mask_eroded[:, -1].max() > 0):
        return _alpha_blend(src, dst, mask)
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _alpha_blend(src, dst, mask)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    center = (x + w // 2, y + h // 2)
    try:
        return cv2.seamlessClone(src, dst, mask_eroded, center, cv2.NORMAL_CLONE)
    except Exception:
        return _alpha_blend(src, dst, mask)


# ── Core inpaint ──────────────────────────────────────────────────────────────

def inpaint(init_image, mask_image, depth_map, prompt, negative_prompt, steps,
            strength=1.0, guidance_scale=12.0, cn_scale=0.3,
            inpaint_size=1024, crop_padding=128, seed=42):
    """Crop vùng mask → inpaint VỚI depth → paste lại. (Matches notebook exactly)"""
    W, H = init_image.size
    mask_np = np.array(mask_image.convert("L"))
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        raise ValueError("Mask trống")

    # Simple crop with padding (same as notebook)
    x1 = max(0, int(xs.min()) - crop_padding)
    y1 = max(0, int(ys.min()) - crop_padding)
    x2 = min(W, int(xs.max()) + crop_padding)
    y2 = min(H, int(ys.max()) + crop_padding)

    crop_img   = init_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_mask  = mask_image.convert("L").crop((x1, y1, x2, y2))
    crop_depth = depth_map.convert("RGB").crop((x1, y1, x2, y2))
    crop_w, crop_h = crop_img.size

    # Scale to inpaint_size (divisible by 64 for SDXL)
    scale = inpaint_size / max(crop_w, crop_h)
    inp_w = (int(crop_w * scale) // 64) * 64
    inp_h = (int(crop_h * scale) // 64) * 64

    inp_img   = crop_img.resize((inp_w, inp_h), Image.LANCZOS)
    inp_mask  = crop_mask.resize((inp_w, inp_h), Image.NEAREST)
    inp_depth = crop_depth.resize((inp_w, inp_h), Image.LANCZOS)

    print(f"  Crop: ({x1},{y1})→({x2},{y2}) [{crop_w}x{crop_h}] | Inpaint: {inp_w}x{inp_h}")

    generator = torch.Generator(device=M.T2_DEVICE).manual_seed(seed)
    result_small = M.pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=inp_img,
        mask_image=inp_mask,
        control_image=inp_depth,
        height=inp_h,
        width=inp_w,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(cn_scale),
        generator=generator,
    ).images[0]

    result_crop = result_small.resize((crop_w, crop_h), Image.LANCZOS)
    output = init_image.convert("RGB").copy()
    output.paste(result_crop, (x1, y1), mask=crop_mask.resize((crop_w, crop_h), Image.NEAREST))
    return output


def _poisson_blend(result_raw, init_image, mask_pil, label=""):
    dst = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
    src = cv2.cvtColor(np.array(result_raw), cv2.COLOR_RGB2BGR)
    mask_blend = np.array(mask_pil.convert("L"))
    contours, _ = cv2.findContours(mask_blend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        center = (x + w // 2, y + h // 2)
        try:
            blended_cv = cv2.seamlessClone(src, dst, mask_blend, center, cv2.NORMAL_CLONE)
            return Image.fromarray(cv2.cvtColor(blended_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[Task2] {label}Poisson blend failed: {e}")
    return result_raw


# ── Public inference entry point ─────────────────────────────────────────────

def run_inference(editor_val, prompt, step, task_type,
                  strength=1.0, guidance_scale=12.0, cn_scale=0.3, neg_prompt_override="", seed=42):
    if editor_val is None:
        raise gr.Error("Vui lòng upload ảnh trước.")
    bg = editor_val.get("background")
    if bg is None:
        raise gr.Error("Vui lòng upload ảnh trước.")
    input_img = cap_image(bg.convert("RGB"))
    if max(input_img.width, input_img.height) > MAX_T2_LONG:
        raise gr.Error(f"Ảnh quá lớn — tối đa {MAX_T2_LONG}px cạnh dài.")
    mask_pil = extract_mask(editor_val)
    if mask_pil is None or np.array(mask_pil).max() == 0:
        raise gr.Error("Vui lòng vẽ mask lên ảnh trước.")

    from model_manager import manager

    if M.pipe is None:
        gr.Info("Đang load model SDXL lên GPU…")
        M.load_to_ram()
        print("[Task2] Waiting for load_to_ram() to complete...")
        M.wait_until_loaded(timeout=180.0)
        print("[Task2] load_to_ram() completed!")

    # Ensure task2 is active
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"\n[Task2] Activation attempt {attempt+1}/{max_retries}")
            manager.activate("task2")
            
            # If activate() returns without exception, consider it success
            if manager.active_task == "task2":
                print(f"[Task2] ✅ Task2 activated successfully")
                break
            else:
                last_error = f"active_task not set (got {manager.active_task})"
                print(f"[Task2] ⚠️  {last_error}")
                
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"[Task2] ❌ Exception: {last_error}")
            if attempt < max_retries - 1:
                time.sleep(0.2)
    else:
        # All retries exhausted
        raise gr.Error(f"Failed to activate task2: {last_error}")

    print(f"[Task2] Starting inference...")
    
    # Track whether start_inference was successfully called
    inference_started = False
    try:
        manager.start_inference("task2")
        inference_started = True
        
        t0 = time.time()

        if task_type == "Delete":
            if not prompt.strip():
                prompt = (
                    "empty wall, bare floor, plain surface, seamless background, "
                    "continuation of surroundings, no objects, nothing here, "
                    "clean empty space, smooth texture, natural lighting"
                )
            _default_neg = (
                "additional objects, extra items, unwanted objects, "
                "multiple objects, cluttered, busy background, "
                "unrelated decoration, extra furniture, "
                "low quality, blurry, distorted, deformed, artifacts, "
                "3d render, CGI, plastic texture, flat shading, "
                "floating, wrong perspective, impossible geometry, "
                "cartoon, anime, painting, sketch, "
                "text, watermark, logo"
            )
        elif task_type == "Add":
            if not prompt.strip():
                prompt = "a beautiful interior design object, high quality"
            _default_neg = (
                "low quality, blurry, distorted, deformed, artifacts, "
                "3d render, CGI, plastic texture, flat shading, "
                "floating, wrong perspective, impossible geometry, "
                "cartoon, anime, painting, sketch, "
                "text, watermark, logo"
            )
        else:  # Replace
            _default_neg = (
                "additional objects, extra items, unwanted objects, "
                "multiple objects, cluttered, busy background, "
                "unrelated decoration, extra furniture, "
                "low quality, blurry, distorted, deformed, artifacts, "
                "3d render, CGI, plastic texture, flat shading, "
                "floating, wrong perspective, impossible geometry, "
                "cartoon, anime, painting, sketch, "
                "text, watermark, logo"
            )
        neg_prompt = neg_prompt_override.strip() if neg_prompt_override and neg_prompt_override.strip() else _default_neg

        W, H = input_img.size
        if (mask_pil.width, mask_pil.height) != (W, H):
            mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        mask_pil = mask_pil.convert("L").resize((W, H), Image.NEAREST)

        depth_map = estimate_depth(input_img)

        result_raw = inpaint(
            input_img, mask_pil, depth_map,
            prompt=prompt, negative_prompt=neg_prompt,
            steps=int(step), strength=strength,
            guidance_scale=guidance_scale, cn_scale=cn_scale, seed=int(seed),
        )
        del depth_map  # free depth map tensor / PIL image before blending
        result = _poisson_blend(result_raw, input_img, mask_pil)
        del result_raw

        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
    finally:
        # Only call end_inference if we successfully called start_inference
        if inference_started:
            manager.end_inference()

    info = (f"Task: {task_type} | Steps: {int(step)} | Strength: {strength} | "
            f"CFG: {guidance_scale} | CN: {cn_scale} | Time: {elapsed:.1f}s")
    return result, info
