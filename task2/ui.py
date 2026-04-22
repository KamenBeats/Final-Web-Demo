"""Task 2 — Gradio UI for Inpainting & Editing."""

import gradio as gr
from .inference import run_inference, enhance_prompt as _qwen_enhance
from . import model as M


def _enhance_prompt_ui(prompt: str) -> str:
    """Đưa pipe + dm về CPU trước, chạy Qwen, rồi trả về enhanced prompt."""
    if not prompt.strip():
        raise gr.Error("Vui lòng nhập prompt trước khi enhance.")

    from model_manager import manager
    
    # Đưa tất cả model khác về CPU để nhường VRAM cho Qwen
    import torch
    if M.pipe is not None:
        try:
            M.pipe.to("cpu")
        except Exception:
            pass
    if M.dm is not None:
        try:
            M.dm.to("cpu")
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset active_task so next activate() call will move task2 back to GPU
    manager.active_task = None
    print("[Task2] Reset active_task to force GPU move on next inference")

    # Qwen tự move lên GPU, chạy, rồi tự move về CPU bên trong enhance_prompt()
    return _qwen_enhance(prompt)


def create_task2_tab():
    gr.Markdown(
        "**Chỉnh sửa ảnh (Inpainting)** — "
        "Vẽ mask trên vùng muốn sửa/điền, sau đó chọn chế độ Add, Delete hoặc Replace."
    )

    # ── Top row: Mode selector + 4 advanced params ───────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=160):
            task_type = gr.Radio(
                choices=["Add", "Delete", "Replace"],
                value="Add",
                label="Chế độ",
                info="Add=thêm · Delete=xóa · Replace=thay thế",
            )
        with gr.Column(scale=1):
            steps_slider = gr.Slider(
                10, 50, step=1, value=30,
                label="Số bước inference",
            )
        with gr.Column(scale=1):
            strength_slider = gr.Slider(
                0.1, 1.0, step=0.05, value=1.0,
                label="Strength (ảnh hưởng mask)",
            )
        with gr.Column(scale=1):
            guidance_scale_slider = gr.Slider(
                1.0, 20.0, step=0.5, value=12.0,
                label="Guidance Scale (CFG)",
            )
        with gr.Column(scale=1):
            cn_scale_slider = gr.Slider(
                0.0, 1.0, step=0.1, value=0.3,
                label="ControlNet Scale",
            )

    # ── Main row ─────────────────────────────────────────────────────────────
    with gr.Row():
        # ── Left column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            image_editor = gr.ImageEditor(
                label="Vẽ mask",
                type="pil",
                sources=["upload", "clipboard"],
                layers=True,
                height=480,
            )

            with gr.Group():
                prompt_input = gr.Textbox(
                    label="Prompt yêu cầu",
                    lines=3,
                    placeholder="Ví dụ: a modern lamp, warm lighting, ...",
                )
                enhance_button = gr.Button(
                    "✨ Enhance Prompt",
                    variant="secondary",
                    size="sm",
                )
            with gr.Accordion("⚙️ Show more", open=False):
                neg_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    value=(
                        "additional objects, extra items, unwanted objects, "
                        "multiple objects, cluttered, busy background, "
                        "unrelated decoration, extra furniture, "
                        "low quality, blurry, distorted, deformed, artifacts, "
                        "3d render, CGI, plastic texture, flat shading, "
                        "floating, wrong perspective, impossible geometry, "
                        "cartoon, anime, painting, sketch, "
                        "text, watermark, logo"
                    ),
                    placeholder="Những gì KHÔNG muốn xuất hiện trong ảnh…",
                )

        # ── Right column ─────────────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes="image-container", min_width=0):
                output_image = gr.Image(
                    label="Kết quả",
                    interactive=False,
                    sources=[],
                    height=480,
                )
                output_info = gr.HTML(
                    value="",
                    elem_classes="floating-info",
                    visible=False,
                )

            run_button = gr.Button("🚀 Xử lý", variant="primary")

    # ── Event handlers ───────────────────────────────────────────────────────
    # Khi đổi mode → tự cập nhật slider về default phù hợp
    _NEG_ADD = (
        "additional objects, extra items, unwanted objects, "
        "multiple objects, cluttered, busy background, "
        "unrelated decoration, extra furniture, "
        "low quality, blurry, distorted, deformed, artifacts, "
        "3d render, CGI, plastic texture, flat shading, "
        "floating, wrong perspective, impossible geometry, "
        "cartoon, anime, painting, sketch, "
        "text, watermark, logo"
    )
    _NEG_DELETE = (
        "object, furniture, item, person, blurry, smudge, artifacts, "
        "distorted texture, mismatched pattern, text, watermark, low quality, "
        "messy, floating debris, ghosting"
    )
    _DEFAULTS = {
        "Add":     {"strength": 1.0,  "cfg": 12.0, "neg": _NEG_ADD,    "steps": 50},
        "Replace": {"strength": 1.0,  "cfg": 12.0, "neg": _NEG_ADD,    "steps": 50},
        "Delete":  {"strength": 0.99, "cfg": 20.0, "neg": _NEG_DELETE, "steps": 50},
    }

    def _on_mode_change(mode):
        d = _DEFAULTS.get(mode, _DEFAULTS["Add"])
        return gr.update(value=d["strength"]), gr.update(value=d["cfg"]), gr.update(value=d["neg"]), gr.update(value=d["steps"])

    task_type.change(
        fn=_on_mode_change,
        inputs=task_type,
        outputs=[strength_slider, guidance_scale_slider, neg_prompt_input, steps_slider],
        queue=False,
    )

    enhance_button.click(
        fn=_enhance_prompt_ui,
        inputs=prompt_input,
        outputs=prompt_input,
    )

    def _clear_output():
        return None, gr.update(value="<div></div>", visible=False)

    def _run(editor, prompt, steps, task, strength, cfg, cn, neg):
        import time
        img, info_text = run_inference(editor, prompt, steps, task, strength, cfg, cn, neg_prompt_override=neg)
        html = f"<div>{info_text}<span style='display:none'>{time.time()}</span></div>"
        return img, gr.update(value=html, visible=True)

    run_button.click(
        fn=_clear_output,
        inputs=None,
        outputs=[output_image, output_info],
        queue=False,
    ).then(
        fn=_run,
        inputs=[
            image_editor,
            prompt_input,
            steps_slider,
            task_type,
            strength_slider,
            guidance_scale_slider,
            cn_scale_slider,
            neg_prompt_input,
        ],
        outputs=[output_image, output_info],
    )