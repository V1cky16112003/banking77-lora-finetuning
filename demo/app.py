"""Gradio demo UI, calling the FastAPI /predict endpoint over HTTP (not
importing the model directly) — one inference path, per the design doc's
Approach B architecture.

Design decisions applied (design review):
- Information hierarchy: input (primary, auto-focused) > submit+chips
  (secondary) > result area (tertiary) > title (least prominent).
- Empty/first-load state: calm placeholder, not a blank box.
- Success state: top-1 label only (reformatted for display), no confidence.
- Wait-state copy: reassures the user during CPU-only multi-second latency.
- Gradio's stock theme — deliberately no custom CSS (Pass 4).
"""
from __future__ import annotations

import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"

EMPTY_STATE_MESSAGE = "Try one of the examples below, or type your own message."
WAIT_STATE_MESSAGE = "Running inference on CPU — this can take a few seconds."

EXAMPLE_PROMPTS = [
    "I lost my card, what do I do?",
    "Why was my balance not updated after a transfer?",
    "The exchange rate on my cash withdrawal seems wrong.",
]


def classify(message: str):
    if not message or not message.strip():
        return EMPTY_STATE_MESSAGE

    try:
        response = requests.post(API_URL, json={"message": message}, timeout=60)
    except requests.RequestException as exc:
        return f"Request failed: {exc}. Is the FastAPI server running?"

    if response.status_code != 200:
        detail = response.json().get("detail", {})
        return f"Prediction failed: {detail}"

    return response.json()["predicted_label_display"]


with gr.Blocks(title="Banking Support Intent Classifier") as demo:
    gr.Markdown(
        "### Banking Support Intent Classifier\n"
        "LoRA fine-tuned Qwen2.5-1.5B-Instruct — 77-class Banking77 intent classification."
    )

    message_input = gr.Textbox(
        label="Customer support message",
        placeholder="Type a banking support message...",
        autofocus=True,
    )

    with gr.Row():
        example_buttons = [gr.Button(example, size="sm") for example in EXAMPLE_PROMPTS]

    submit_button = gr.Button("Classify", variant="primary")

    result_output = gr.Textbox(label="Predicted intent", value=EMPTY_STATE_MESSAGE, interactive=False)

    def _on_submit_start():
        # 4A: disable while in-flight; Pass 3: reassuring wait-state copy.
        return gr.update(interactive=False), WAIT_STATE_MESSAGE

    def _on_submit_end(message: str):
        result = classify(message)
        return gr.update(interactive=True), result

    submit_button.click(
        fn=_on_submit_start, inputs=None, outputs=[submit_button, result_output]
    ).then(fn=_on_submit_end, inputs=message_input, outputs=[submit_button, result_output])

    # Enter-to-submit (11B)
    message_input.submit(
        fn=_on_submit_start, inputs=None, outputs=[submit_button, result_output]
    ).then(fn=_on_submit_end, inputs=message_input, outputs=[submit_button, result_output])

    for button in example_buttons:
        button.click(fn=lambda ex=button.value: ex, inputs=None, outputs=message_input)


if __name__ == "__main__":
    demo.launch()
