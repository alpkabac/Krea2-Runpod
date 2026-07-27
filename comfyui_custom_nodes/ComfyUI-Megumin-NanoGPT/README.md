# ComfyUI-Megumin-NanoGPT

A small ComfyUI node that sends text to NanoGPT's OpenAI-compatible chat-completions API, returns the generated result as a plain `STRING`, and publishes that text in ComfyUI history so Megumin can store the prompt that actually rendered the image.

## Installation

Copy the `ComfyUI-Megumin-NanoGPT` directory into:

```text
ComfyUI/custom_nodes/
```

Restart ComfyUI. The node appears under:

```text
Megumin → LLM → Megumin NanoGPT Text
```

The node uses Python's standard library and requires no additional pip packages.

## Workflow

1. Put `%ai_text%` in the node's `ai_text` widget.
2. Put `%prompt%` in `fallback_text`. If NanoGPT fails and fallback is enabled, the deterministic prompt is used.
3. Enter the exact NanoGPT model identifier.
4. Enter the API key, or set the `NANOGPT_API_KEY` environment variable before starting ComfyUI.
5. Connect the node's `text` output to the positive `CLIP Text Encode` text input.

`%prompt%` remains available for ordinary workflows. `%ai_text%` is optional and only affects workflows that include it.

## identity_suffix (LoRA trigger words)

`identity_suffix` is an **optional** widget. If your workflow needs literal trigger
words (e.g. `a woman, a woman` for two stacked character LoRAs) to always be
present in the final prompt, put them here instead of writing them into
`ai_text`.

This text is **never sent to the LLM** — it is appended in Python, after the
API call succeeds *or* falls back, directly onto whatever text is about to be
returned. This guarantees the trigger phrase always lands in the prompt that
reaches `CLIP Text Encode`, without depending on the model to "understand"
and faithfully reproduce a meta-instruction. Smaller/faster chat models are
prone to echoing instruction-shaped text (e.g. "Use the exact trigger phrase
X for identity Y") verbatim into their output instead of transforming it,
which pollutes the actual image prompt with nonsense. Keeping trigger-word
injection out of the LLM's hands avoids that failure mode entirely.

The endpoint defaults to:

```text
https://nano-gpt.com/api/v1/chat/completions
```

It can be changed directly in the node if NanoGPT's endpoint differs for your account.
