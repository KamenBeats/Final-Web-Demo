"""Task 1 — Gradio UI for Multi-Exposure Fusion."""

import gradio as gr
from .inference import run


def create_task1_tab():
    """Create the UI tab for Task 1 (Multi-Exposure Fusion)."""
    gr.Markdown(
        "**Multi-Exposure Fusion (UFRetinex-MEF-ToneNet)** — "
        "Tải 2+ ảnh với các mức độ sáng khác nhau để ghép thành 1 ảnh cân bằng ánh sáng. Hoặc 1 ảnh và chọn tăng cường ánh sáng để tạo ra ảnh chất lượng hơn."
    )

    with gr.Row():
        # ── Left column (Preview + Upload) ───────────────────────────────────
        with gr.Column(scale=1):
            preview_gallery = gr.Gallery(
                label="Preview ảnh đầu vào",
                height=480,  # Fixed height to match output
                interactive=False,
                elem_classes="input-gallery",
                columns=3,   # Cố định lưới 2 cột
            )
            
            input_files = gr.File(
                label="Chọn ảnh (2+ ảnh có cùng kích thước)",
                file_count="multiple",
                file_types=["image"],
            )

        # ── Right column (Result + Controls) ─────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes="image-container", min_width=0):
                output_image = gr.Image(
                    label="Kết quả",
                    interactive=False,
                    height=480,  # Match height with preview_gallery
                )
                output_info = gr.HTML(
                    value="",
                    elem_classes="floating-info",
                    visible=False,
                )

            with gr.Row():
                apply_phase2_chk = gr.Checkbox(
                    label="Tăng cường màu sắc - ánh sáng",
                    value=True,
                )
                align_chk = gr.Checkbox(
                    label="Căn chỉnh ảnh",
                    value=True,
                )

            run_button = gr.Button("🚀 Ghép & Xử lý", variant="primary")

    # ── Event handlers ───────────────────────────────────────────────────────
    def _update_gallery(files):
        if not files: return None
        return [f.name for f in files]

    input_files.change(
        fn=_update_gallery,
        inputs=input_files,
        outputs=preview_gallery,
    )

    def _process_run(files, p2, align):
        if not files:
            raise gr.Error("Vui lòng chọn ảnh trước.")
        
        paths = [f.name for f in files]
        img, info_text = run(paths, p2, align)
        
        html_info = f"<div>{info_text}</div>"
        return img, gr.update(value=html_info, visible=True)
    
    def _clear_output():
        return None, gr.update(value="", visible=False)

    run_button.click(
        fn=_clear_output,
        inputs=None,
        outputs=[output_image, output_info],
        queue=False
    ).then(
        fn=_process_run,
        inputs=[input_files, apply_phase2_chk, align_chk],
        outputs=[output_image, output_info],
    )