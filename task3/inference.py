"""Task 3 — Inference helpers (canvas prep, blending, infer).

Must match training exactly (Long/outpainting/src/train.py →
prepare_image_and_mask_like_app).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

import gradio as gr

from . import model as M

# ── Constants ─────────────────────────────────────────────────────────────────
SDXL_BUCKETS = {
    "1:1":       (1024, 1024),
    "4:3":       (1360, 1024),   # must be multiple of 8 for SDXL VAE (8× downsampling)
    "3:4":       (1024, 1360),
    "16:9":      (1344, 768),
    "9:16":      (768,  1344),
    "Customize": (0,    0),
}

_MAX_OUTPUT_PX = 4096  # hard cap for output resolution (4K)

_RESIZE_OPTS = ["Full", "50%", "33%", "25%", "Custom"]
_ALIGN_OPTS  = ["Middle", "Left", "Right", "Top", "Bottom"]

_GUIDANCE_SCALE   = 2.5      # match CLI inference_preserve_blend_v2.py
_CONTROLNET_SCALE = 1.0
_SEED             = 1234
_BLEND_RADIUS     = 24
_COLOR_MATCH_STR  = 0.6


def _resolve_res(target_res_label, custom_w, custom_h):
    bw, bh = SDXL_BUCKETS.get(target_res_label, (0, 0))
    if bw == 0:  # Customize
        bw, bh = int(custom_w or 1024), int(custom_h or 1024)
    # Hard cap: neither dimension may exceed 2K
    bw = min(bw, _MAX_OUTPUT_PX)
    bh = min(bh, _MAX_OUTPUT_PX)
    return bw, bh


def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def _prepare_canvas(
    image: Image.Image,
    width: int,
    height: int,
    overlap_percentage: float,
    resize_option: str,
    custom_resize_percentage: int,
    alignment: str,
    overlap_left: float,
    overlap_right: float,
    overlap_top: float,
    overlap_bottom: float,
    use_padding_mode: bool = False,
    pad_left_px: int = 0,
    pad_top_px: int = 0,
):
    """Prepare canvas, mask, and cnet_image — matches training exactly.

    Args:
        use_padding_mode: when True (Customize mode), source is kept at original
            size and placed at (pad_left_px, pad_top_px). No scale-to-fit.
        overlap_left/right/top/bottom: individual overlap percentages (0-50).
    """
    target_size = (width, height)

    if use_padding_mode:
        # Customize mode: keep source at original size, place at pad offset
        source = image.copy()
        new_width  = source.width
        new_height = source.height
        margin_x   = max(0, int(pad_left_px))
        margin_y   = max(0, int(pad_top_px))
    else:
        # Standard mode: scale source to fit inside canvas
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width  = int(image.width  * scale_factor)
        new_height = int(image.height * scale_factor)
        source = image.resize((new_width, new_height), Image.LANCZOS)

        if resize_option == "Full":
            resize_percentage = 100
        elif resize_option == "50%":
            resize_percentage = 50
        elif resize_option == "33%":
            resize_percentage = 33
        elif resize_option == "25%":
            resize_percentage = 25
        else:
            resize_percentage = custom_resize_percentage

        resize_factor = resize_percentage / 100.0
        new_width  = max(int(source.width  * resize_factor), 64)
        new_height = max(int(source.height * resize_factor), 64)
        source = source.resize((new_width, new_height), Image.LANCZOS)

        if alignment == "Middle":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - new_width
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = 0
        else:  # Bottom
            margin_x = (target_size[0] - new_width) // 2
            margin_y = target_size[1] - new_height

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    white_gaps_patch = 2

    overlap_x_left   = max(int(new_width  * (overlap_left   / 100.0)), 1)
    overlap_x_right  = max(int(new_width  * (overlap_right  / 100.0)), 1)
    overlap_y_top    = max(int(new_height * (overlap_top    / 100.0)), 1)
    overlap_y_bottom = max(int(new_height * (overlap_bottom / 100.0)), 1)

    left_overlap   = margin_x + overlap_x_left   if overlap_left   > 0 else margin_x + white_gaps_patch
    right_overlap  = margin_x + new_width  - overlap_x_right  if overlap_right  > 0 else margin_x + new_width  - white_gaps_patch
    top_overlap    = margin_y + overlap_y_top    if overlap_top    > 0 else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y_bottom if overlap_bottom > 0 else margin_y + new_height - white_gaps_patch

    if not use_padding_mode:
        if alignment == "Left":
            left_overlap   = margin_x + overlap_x_left if overlap_left > 0 else margin_x
        elif alignment == "Right":
            right_overlap  = margin_x + new_width - overlap_x_right if overlap_right > 0 else margin_x + new_width
        elif alignment == "Top":
            top_overlap    = margin_y + overlap_y_top if overlap_top > 0 else margin_y
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - overlap_y_bottom if overlap_bottom > 0 else margin_y + new_height

    left_overlap   = max(0, min(left_overlap,  target_size[0] - 1))
    right_overlap  = max(left_overlap  + 1, min(right_overlap,  target_size[0]))
    top_overlap    = max(0, min(top_overlap,   target_size[1] - 1))
    bottom_overlap = max(top_overlap   + 1, min(bottom_overlap, target_size[1]))

    preserve_rect = (left_overlap, top_overlap, right_overlap, bottom_overlap)
    mask_draw.rectangle([(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0)

    # cnet_image: black out only the pure outer canvas (pixels with no source content).
    # The overlap band stays visible so the model has it as reference context and
    # does not have to regenerate from the inner preserve boundary, avoiding blur/seam.
    cnet_outer_mask = Image.new("L", target_size, 255)  # 255 = black out
    ImageDraw.Draw(cnet_outer_mask).rectangle(
        [(margin_x, margin_y), (margin_x + new_width - 1, margin_y + new_height - 1)],
        fill=0,  # 0 = keep source pixels visible (including overlap band)
    )
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), cnet_outer_mask)

    return background, mask, cnet_image, preserve_rect


def _make_blend_mask(size, preserve_rect, blend_radius: int) -> Image.Image:
    w, h    = size
    x1, y1, x2, y2 = preserve_rect
    core    = Image.new("L", (w, h), 0)
    ImageDraw.Draw(core).rectangle([(x1, y1), (x2, y2)], fill=255)
    if blend_radius > 0:
        return core.filter(ImageFilter.GaussianBlur(radius=blend_radius))
    return core


def _color_match(generated: Image.Image, background: Image.Image,
                 blend_mask: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return generated
    # Ensure blend_mask has same size as generated image
    if blend_mask.size != generated.size:
        blend_mask = blend_mask.resize(generated.size, Image.BILINEAR)
    a    = np.asarray(blend_mask).astype(np.float32) / 255.0
    band = (a > 0.05) & (a < 0.95)
    if band.sum() < 128:
        return generated
    gen = np.asarray(generated).astype(np.float32)
    bg  = np.asarray(background).astype(np.float32)
    out = gen.copy()
    for c in range(3):
        shift = (float(bg[..., c][band].mean()) - float(gen[..., c][band].mean())) * strength
        out[..., c] = np.clip(out[..., c] + shift, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _sharpen_generated(
    generated: Image.Image,
    blend_mask: Image.Image,
    strength: float = 1.0,
) -> Image.Image:
    """Sharpen only the generated area to compensate for VAE decode softness.

    blend_mask: 255 = preserve (no sharpen), 0 = generated (full sharpen).
    strength: 0.0 = off, 1.0 = normal, 2.0 = strong.
    """
    if strength <= 0:
        return generated
    # Ensure blend_mask has same size as generated image
    if blend_mask.size != generated.size:
        blend_mask = blend_mask.resize(generated.size, Image.BILINEAR)
    sharpened = generated.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=int(80 * strength), threshold=2)
    )
    gen_area_mask = Image.eval(blend_mask, lambda x: 255 - x)
    gen_area_mask = gen_area_mask.filter(ImageFilter.GaussianBlur(radius=3))
    return Image.composite(sharpened, generated, gen_area_mask)


def _canvas_from_pads(image, pad_left_px, pad_right_px, pad_top_px, pad_bottom_px):
    """Compute exact canvas size from image + pixel padding, snapped to mult-of-8 for VAE."""
    pl = int(pad_left_px or 0)
    pr = int(pad_right_px or 0)
    pt = int(pad_top_px or 0)
    pb = int(pad_bottom_px or 0)
    fw = image.width  + pl + pr
    fh = image.height + pt + pb
    # Snap to nearest multiple of 8 (SDXL VAE requirement)
    bw = max(((fw + 7) // 8) * 8, 64)
    bh = max(((fh + 7) // 8) * 8, 64)
    return bw, bh, fw, fh, pl, pr, pt, pb


def preview(image, target_res_label, custom_w, custom_h,
            alignment, resize_option, custom_resize_pct,
            overlap_percentage, overlap_left, overlap_right,
            overlap_top, overlap_bottom,
            pad_left_px=0, pad_right_px=0, pad_top_px=0, pad_bottom_px=0):
    if image is None:
        return None
    is_custom = (target_res_label == "Customize")
    if is_custom:
        bw, bh, fw, fh, *_ = _canvas_from_pads(
            image, pad_left_px, pad_right_px, pad_top_px, pad_bottom_px
        )
        if fw > _MAX_OUTPUT_PX or fh > _MAX_OUTPUT_PX:
            return None  # invalid — don't show preview
    else:
        bw, bh = _resolve_res(target_res_label, custom_w, custom_h)
    try:
        bg, mask, _cnet, _rect = _prepare_canvas(
            image, bw, bh, overlap_percentage,
            resize_option, custom_resize_pct, alignment,
            overlap_left, overlap_right, overlap_top, overlap_bottom,
            use_padding_mode=is_custom,
            pad_left_px=int(pad_left_px or 0),
            pad_top_px=int(pad_top_px or 0),
        )
    except Exception:
        return image
    vis       = bg.copy().convert("RGBA")
    red_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    overlay   = Image.new("RGBA", bg.size, (255, 0, 0, 80))
    red_layer.paste(overlay, (0, 0), mask)
    return Image.alpha_composite(vis, red_layer)


def infer(
    image,
    target_res_label, custom_w, custom_h,
    alignment, resize_option, custom_resize_pct,
    overlap_percentage, overlap_left, overlap_right,
    overlap_top, overlap_bottom,
    pad_left_px, pad_right_px, pad_top_px, pad_bottom_px,
    prompt, num_steps, sharpen_strength, lora_scale,
):
    if image is None:
        raise gr.Error("Vui lòng upload ảnh trước.")

    is_custom = (target_res_label == "Customize")

    if is_custom:
        # Canvas expansion mode: exact pixel arithmetic, image pasted at (left, top)
        bw, bh, fw_exact, fh_exact, pl, pr, pt, pb = _canvas_from_pads(
            image, pad_left_px, pad_right_px, pad_top_px, pad_bottom_px
        )
        if fw_exact > _MAX_OUTPUT_PX or fh_exact > _MAX_OUTPUT_PX:
            raise gr.Error(
                f"Kích thước canvas ({fw_exact}×{fh_exact}px) vượt giới hạn {_MAX_OUTPUT_PX}px. "
                f"Giảm padding để tiếp tục."
            )
    else:
        # Standard SDXL-bucket mode: downscale input if needed
        if image.width > _MAX_OUTPUT_PX or image.height > _MAX_OUTPUT_PX:
            scale = _MAX_OUTPUT_PX / max(image.width, image.height)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
            )
        bw, bh = _resolve_res(target_res_label, custom_w, custom_h)

    from model_manager import manager

    if M._pipe is None:
        if M._loading:
            import time
            print("[Task3] Waiting for background preload…")
            M.wait_until_loaded()
        else:
            M._load_to_ram()

    if M._pipe is None:
        raise gr.Error(
            "Pipeline Task 3 chưa được load. "
            "Kiểm tra log khởi động — có thể thiếu controlnet_union.py, "
            "pipeline_fill_sd_xl.py hoặc file weights."
        )

    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)

    if not is_custom and not can_expand(image.width, image.height, bw, bh, alignment):
        alignment = "Middle"

    background, _mask, cnet_image, preserve_rect = _prepare_canvas(
        image, bw, bh, overlap_percentage,
        resize_option, custom_resize_pct, alignment,
        overlap_left, overlap_right, overlap_top, overlap_bottom,
        use_padding_mode=is_custom,
        pad_left_px=int(pad_left_px or 0),
        pad_top_px=int(pad_top_px or 0),
    )
    blend_mask = _make_blend_mask(background.size, preserve_rect, _BLEND_RADIUS)

    prompt_text = f"{prompt}, high quality, 4k" if prompt else "high quality, 4k"

    device = M.T3_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

    manager.activate("task3")
    manager.start_inference("task3")
    try:
        M.set_lora_scale(float(lora_scale))

        with torch.no_grad():
            (prompt_embeds, neg_pe,
             pooled_pe, neg_pooled_pe) = M._pipe.encode_prompt(prompt_text, device, True)

        last_image = None
        try:
            for out in M._pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_pe,
                pooled_prompt_embeds=pooled_pe,
                negative_pooled_prompt_embeds=neg_pooled_pe,
                image=cnet_image,
                num_inference_steps=num_steps,
                guidance_scale=_GUIDANCE_SCALE,
                controlnet_conditioning_scale=_CONTROLNET_SCALE,
            ):
                last_image = out
                yield cnet_image, out
        finally:
            del prompt_embeds, neg_pe, pooled_pe, neg_pooled_pe

        clean_base = last_image.copy()                   
        pr = preserve_rect
        clean_base.paste(                                 
            background.crop(pr), (pr[0], pr[1])
        )

        generated_adj = _color_match(last_image, clean_base, blend_mask, _COLOR_MATCH_STR)
        generated_adj = _sharpen_generated(generated_adj, blend_mask, float(sharpen_strength))
        result = Image.composite(clean_base, generated_adj, blend_mask)
        del last_image, blend_mask, clean_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        yield cnet_image, result
    finally:
        manager.end_inference()
