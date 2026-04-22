"""WebDemo — Gradio entry point."""

import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)

import threading

import gradio as gr

from model_manager import manager
from task1 import create_task1_tab, preload_to_cpu as preload_task1_cpu
from task2 import create_task2_tab, preload_to_cpu as preload_task2_cpu
from task3 import create_task3_tab, preload_to_cpu as preload_task3_cpu

# ── Preload ALL models to CPU RAM at startup ──────────────────────────────────
print("[Startup] Preloading Task 1 weights to CPU RAM (background)…")
preload_task1_cpu()
print("[Startup] Preloading Task 2 pipeline to CPU RAM (background)…")
preload_task2_cpu()
print("[Startup] Preloading Task 3 pipeline to CPU RAM (background)…")
preload_task3_cpu()

import time as _time
_time.sleep(1)  # give background threads a moment to start

print("[Startup] Activating Task 1 on GPU (default tab)…")
manager.activate("task1")
print("[Startup] Ready.")

# ── Tab indices ───────────────────────────────────────────────────────────────
_TASK_NAMES = ["task1", "task2", "task3"]

# ── Build UI ──────────────────────────────────────────────────────────────────
_CSS = """
.image-container { 
    position: relative !important; 
    padding: 0 !important; 
    gap: 0 !important; 
}

.floating-info {
    position: absolute !important;
    bottom: 2px !important;
    right: 2px !important;
    background: rgba(0, 0, 0, 0.75) !important;
    color: #4ade80 !important;
    padding: 4px 10px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    pointer-events: none !important;
    z-index: 100 !important;
    width: max-content !important;
    border: none !important;
    min-height: 0 !important;
    margin: 0 !important;
}

div:has(> .floating-info) {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    box-shadow: none !important;
}

button[aria-label="Share"],
.share-button {
    display: none !important;
}


/* Surgically override Gradio's dynamic inline style that forces 1-column layout for single images. 
   This safely locks the default grid into 2 columns without breaking its native structure. */
.input-gallery div[style*="grid-template-columns"],
.input-gallery .grid-wrap > div {
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
}

/* Task 3 — align image bottoms: output has a tab-bar on top so align row to flex-end */
.t3-img-row {
    align-items: flex-end !important;
}

/* Task 3 — top settings row: equal height, radio options pinned to bottom */
.t3-top-row {
    align-items: stretch !important;
}
.t3-top-col {
    display: flex !important;
    flex-direction: column !important;
}
.t3-top-col > .block {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
.t3-top-col > .block > .form {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

/* Task 3 — input image: float source buttons (upload/clipboard) to top-right
   next to the clear/fullscreen icon buttons, hide the bottom toolbar border */
#t3_input_img .source-wrap {
    position: absolute !important;
    top: 4px !important;
    right: 72px !important;
    bottom: auto !important;
    left: auto !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    gap: 2px !important;
    z-index: 20 !important;
    display: flex !important;
    align-items: center !important;
}
"""

with gr.Blocks(title="Image Processing Demo", css=_CSS) as demo:
    gr.Markdown("# Image Processing Demo")
    _loading_banner = gr.Markdown(visible=False)

    with gr.Tabs() as tabs:
        with gr.Tab("Task 1 — Multi-Exposure Fusion", id="tab1"):
            create_task1_tab()

        with gr.Tab("Task 2 — Inpainting & Editing", id="tab2"):
            create_task2_tab()

        with gr.Tab("Task 3 — Outpainting", id="tab3"):
            create_task3_tab()

    # ── Tab-switch handler ────────────────────────────────────────────────────
    _TAB_MAP = {
        "tab1": "task1", 0: "task1",
        "tab2": "task2", 1: "task2",
        "tab3": "task3", 2: "task3",
        "Task 1 — Multi-Exposure Fusion": "task1",
        "Task 2 — Inpainting & Editing": "task2",
        "Task 3 — Outpainting": "task3",
    }

    def _on_tab_select(evt: gr.SelectData):
        task = _TAB_MAP.get(evt.index) or _TAB_MAP.get(evt.value)
        print(f"[TabSwitch] evt.index={evt.index!r}  evt.value={evt.value!r}  → {task}")
        if task:
            # Run activate in background thread so it never blocks the Gradio event loop.
            # The inference functions also call activate() themselves, so this is a
            # best-effort preemptive move — if it races with an ongoing inference it will
            # simply wait in the background without holding up the UI queue.
            def _do_activate(t=task):
                try:
                    manager.activate(t)
                except Exception as e:
                    print(f"[TabSwitch] activate({t!r}) error: {e}")
            threading.Thread(target=_do_activate, daemon=True).start()
        return gr.update(visible=False)

    tabs.select(
        fn=_on_tab_select,
        inputs=None,
        outputs=_loading_banner,
        queue=False,   # fire immediately, do not enter the Gradio job queue
    )

demo.queue(max_size=4).launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    theme=gr.themes.Soft(),
    enable_monitoring=False,
)
