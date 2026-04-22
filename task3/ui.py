"""Task 3 — Gradio UI."""

import time

import gradio as gr

from .inference import (
    preview,
    infer,
    _resolve_res,
    SDXL_BUCKETS,
    _RESIZE_OPTS,
    _ALIGN_OPTS,
)
from .model import preload_to_cpu

_IMG_H = 480  # fixed height — identical for both input and output images


def create_task3_tab():
    gr.Markdown(
        "**Mở rộng ảnh (Outpainting)** — Ảnh gốc được đặt lên canvas SDXL, "
        "vùng màu đỏ trong preview là nơi model tự vẽ thêm."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 1 — Top settings (target size + position/scale)
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Row(equal_height=True, elem_classes="t3-top-row"):
        with gr.Column(scale=1, min_width=180, elem_classes="t3-top-col"):
            target_res = gr.Radio(
                choices=list(SDXL_BUCKETS.keys()),
                value="1:1",
                label="📐 Kích thước đầu ra",
            )

        with gr.Column(scale=1, min_width=170):
            alignment = gr.Dropdown(
                choices=_ALIGN_OPTS,
                value="Middle",
                label="📍 Vị trí ảnh",
                info="Middle=giữa, Left/Right/Top/Bottom=sát cạnh",
            )

        with gr.Column(scale=1, min_width=180, elem_classes="t3-top-col"):
            resize_option = gr.Radio(
                choices=["Full", "50%", "33%", "25%"],
                value="Full",
                label="🔲 Kích thước ảnh gốc",
            )
            # kept as hidden fixed value so inference signature is unchanged
            custom_resize_pct = gr.Number(value=100, visible=False, label="_resize_pct")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 2 — Advanced settings (full-width accordion)
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
        with gr.Row():
            # Overlap — 2×2 grid
            with gr.Column(scale=2):
                gr.Markdown("**Overlap theo hướng (%)**")
                with gr.Row():
                    overlap_top    = gr.Slider(0, 50, step=1, value=10, label="Trên ↑")
                    overlap_bottom = gr.Slider(0, 50, step=1, value=10, label="Dưới ↓")
                with gr.Row():
                    overlap_left   = gr.Slider(0, 50, step=1, value=10, label="Trái ←")
                    overlap_right  = gr.Slider(0, 50, step=1, value=10, label="Phải →")

            # Inference params — vertical stack in narrower column
            with gr.Column(scale=1):
                gr.Markdown("**Tham số inference**")
                num_steps  = gr.Slider(10, 50, step=5, value=20, label="Số bước")
                sharpen    = gr.Slider(0.0, 2.0, step=0.1, value=1.0, label="Sắc nét")
                lora_scale = gr.Slider(0.0, 1.0, step=0.1, value=0.0, label="Base Model")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 3 — Customize canvas expansion (pixel-exact, image pasted at (left, top))
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Group(visible=False) as custom_pad_group:
        gr.Markdown(
            "**Mở rộng canvas tùy chỉnh** — nhập số pixel cần thêm vào mỗi cạnh  "
            "<small style='color:#888'>)Ảnh gốc giữ nguyên kích thước, được dán tại tọa độ (left, top) · Tối đa 4096px</small>"
        )
        size_info = gr.HTML(
            value="<span style='color:#888;font-size:0.9em'>Upload ảnh để xem kích thước</span>"
        )
        with gr.Row():
            pad_top    = gr.Number(value=0, minimum=0, maximum=4096, precision=0, label="Trên ↑ (px)")
            pad_bottom = gr.Number(value=0, minimum=0, maximum=4096, precision=0, label="Dưới ↓ (px)")
        with gr.Row():
            pad_left   = gr.Number(value=0, minimum=0, maximum=4096, precision=0, label="Trái ← (px)")
            pad_right  = gr.Number(value=0, minimum=0, maximum=4096, precision=0, label="Phải → (px)")
        # Hidden computed canvas size (snapped to mult of 8 for VAE)
        custom_w = gr.Number(value=1024, visible=False, label="_canvas_w")
        custom_h = gr.Number(value=1024, visible=False, label="_canvas_h")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 4 — Main image row (same height both sides)
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Row(elem_classes="t3-img-row"):
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", label="Ảnh đầu vào",
                sources=["upload", "clipboard"],
                height=_IMG_H,
                elem_id="t3_input_img",
            )

        with gr.Column(scale=1):
            with gr.Column(elem_classes="image-container", min_width=0):
                output_tabs = gr.Tabs(selected="tab_preview")
                with output_tabs:
                    with gr.Tab("Preview vùng mở rộng", id="tab_preview"):
                        preview_image = gr.Image(
                            interactive=False, container=False,
                            sources=[], show_label=False,
                            height=_IMG_H,
                        )
                    with gr.Tab("Kết quả", id="tab_result"):
                        result_image = gr.Image(
                            interactive=False, container=False,
                            sources=[], show_label=False,
                            height=_IMG_H,
                        )
                output_info = gr.HTML(value="", elem_classes="floating-info", visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 5 — Controls row (prompt left, generate right)
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt mô tả nội dung mở rộng (tuỳ chọn)",
                placeholder="Để trống để dùng prompt mặc định...",
                lines=2,
            )
        with gr.Column(scale=1):
            run_button = gr.Button("🚀 Generate", variant="primary")

    # ── History ───────────────────────────────────────────────────────────────
    with gr.Accordion("📋 Lịch sử xử lý", open=False):
        history_gallery = gr.Gallery(
            columns=4, object_fit="contain",
            interactive=False, allow_preview=False,
            show_label=False, label="",
        )
        clear_hist_btn = gr.Button("Xóa lịch sử", size="sm", variant="secondary")

    # ── overlap_percentage placeholder (kept for inference signature compat) ──
    overlap_percentage = gr.Number(value=10, visible=False, label="_overlap_pct")

    # ── Param lists ───────────────────────────────────────────────────────────
    _preview_inputs = [
        input_image, target_res, custom_w, custom_h,
        alignment, resize_option, custom_resize_pct,
        overlap_percentage, overlap_left, overlap_right,
        overlap_top, overlap_bottom,
        pad_left, pad_right, pad_top, pad_bottom,
    ]
    _all_inputs = _preview_inputs + [prompt_input, num_steps, sharpen, lora_scale]

    _INPUT_KEYS = [
        "input_image", "target_res", "custom_w", "custom_h",
        "alignment", "resize_option", "custom_resize_pct",
        "overlap_percentage", "overlap_left", "overlap_right",
        "overlap_top", "overlap_bottom",
        "pad_left", "pad_right", "pad_top", "pad_bottom",
        "prompt", "num_steps", "sharpen", "lora_scale",
    ]

    # ── Helper: compute exact canvas size from padding + realtime HTML status ──
    _EMPTY_INFO = "<span style='color:#888;font-size:0.9em'>Upload ảnh để xem kích thước</span>"

    def _calc_canvas(image, pl, pr, pt, pb):
        if image is None:
            return gr.update(), gr.update(), _EMPTY_INFO
        try:
            orig_w, orig_h = image.width, image.height
            pl, pr, pt, pb = int(pl or 0), int(pr or 0), int(pt or 0), int(pb or 0)
            fw = orig_w + pl + pr
            fh = orig_h + pt + pb
            # Snap to multiple of 8 for VAE compatibility
            fw8 = max(((fw + 7) // 8) * 8, 64)
            fh8 = max(((fh + 7) // 8) * 8, 64)
            valid = fw <= 4096 and fh <= 4096
            if valid:
                html = (
                    f"<div style='font-size:0.9em;line-height:1.8'>"
                    f"📷 Gốc&nbsp;<b>{orig_w}×{orig_h}</b>&nbsp;&nbsp;→&nbsp;&nbsp;"
                    f"🖼 Canvas&nbsp;<b style='color:#4CAF50'>{fw8}×{fh8}</b>&nbsp;✅"
                    f"</div>"
                )
                return gr.update(value=fw8), gr.update(value=fh8), html
            else:
                over = []
                if fw > 4096: over.append(f"W {fw} > 4096")
                if fh > 4096: over.append(f"H {fh} > 4096")
                html = (
                    f"<div style='font-size:0.9em;line-height:1.8'>"
                    f"📷 Gốc&nbsp;<b>{orig_w}×{orig_h}</b>&nbsp;&nbsp;→&nbsp;&nbsp;"
                    f"🖼 Canvas&nbsp;<b style='color:#f44336'>{fw}×{fh}</b>&nbsp;"
                    f"❌ <span style='color:#f44336'>Vượt giới hạn: {', '.join(over)}</span>"
                    f"</div>"
                )
                return gr.update(), gr.update(), html
        except Exception:
            return gr.update(), gr.update(), gr.update()

    # ── Show/hide customize padding section ───────────────────────────────────
    target_res.change(
        fn=lambda v: gr.update(visible=(v == "Customize")),
        inputs=target_res,
        outputs=custom_pad_group,
        queue=False,
    )

    # ── Update canvas size and size display on padding change ─────────────────
    for _pad in [pad_left, pad_right, pad_top, pad_bottom]:
        _pad.change(
            fn=_calc_canvas,
            inputs=[input_image, pad_left, pad_right, pad_top, pad_bottom],
            outputs=[custom_w, custom_h, size_info],
            queue=False,
        )

    # Recompute canvas when image is uploaded
    def _on_img_upload(img, pl, pr, pt, pb, res):
        if res == "Customize":
            return _calc_canvas(img, pl, pr, pt, pb)
        return gr.update(), gr.update(), gr.update()

    input_image.change(
        fn=_on_img_upload,
        inputs=[input_image, pad_left, pad_right, pad_top, pad_bottom, target_res],
        outputs=[custom_w, custom_h, size_info],
        queue=False,
    )

    # ── Auto-preview on canvas param change ───────────────────────────────────
    for _comp in _preview_inputs:
        _comp.change(
            fn=preview,
            inputs=_preview_inputs,
            outputs=preview_image,
            queue=False,
        )

    input_image.change(
        fn=lambda: gr.update(selected="tab_preview"),
        outputs=output_tabs,
        queue=False,
    )

    # ── History logic ─────────────────────────────────────────────────────────
    history_state = gr.State([])

    def _save_history(history, result_img, *all_inputs):
        if result_img is None:
            return history
        entry = {"output": result_img}
        for k, v in zip(_INPUT_KEYS, all_inputs):
            entry[k] = v
        return [entry] + history[:19]

    def _gallery_items(history):
        return [e["output"] for e in history]

    def _restore_history(history, evt: gr.SelectData):
        if not history or evt.index >= len(history):
            return tuple(gr.update() for _ in range(len(_all_inputs)))
        e = history[evt.index]
        return tuple(e.get(k, gr.update()) for k in _INPUT_KEYS)

    clear_hist_btn.click(fn=lambda: ([], []), outputs=[history_state, history_gallery])
    history_gallery.select(
        fn=_restore_history,
        inputs=history_state,
        outputs=_all_inputs,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    _NUM_STEPS_IDX = len(_preview_inputs) + 1  # prompt is at +0, num_steps at +1

    def _run(*args):
        t0   = time.time()
        last = None
        for _cnet_img, out_img in infer(*args):
            last = out_img
            yield last, gr.update(visible=False)
        dt     = round(time.time() - t0, 1)
        bw, bh = _resolve_res(args[1], args[2], args[3])
        steps  = args[_NUM_STEPS_IDX]
        info   = f"Hoàn thành trong {dt}s · {bw}×{bh}px · {steps} steps"
        yield last, gr.update(value=f"<span>{info}</span>", visible=True)

    for trigger in (run_button.click, prompt_input.submit):
        trigger(
            fn=lambda: (gr.update(value=None), gr.update(visible=False)),
            outputs=[result_image, output_info],
        ).then(
            fn=lambda: gr.update(selected="tab_result"),
            outputs=output_tabs,
        ).then(
            fn=_run,
            inputs=_all_inputs,
            outputs=[result_image, output_info],
        ).then(
            fn=_save_history,
            inputs=[history_state, result_image] + _all_inputs,
            outputs=history_state,
        ).then(
            fn=_gallery_items,
            inputs=history_state,
            outputs=history_gallery,
        )
