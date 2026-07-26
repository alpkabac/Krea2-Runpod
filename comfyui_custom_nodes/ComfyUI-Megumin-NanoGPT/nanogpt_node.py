import json
import os
import urllib.error
import urllib.request


DEFAULT_ENDPOINT = "https://nano-gpt.com/api/v1/chat/completions"


class MeguminNanoGPTText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ai_text": (
                    "STRING",
                    {
                        "default": "%ai_text%",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "fallback_text": (
                    "STRING",
                    {
                        "default": "%prompt%",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                "endpoint": (
                    "STRING",
                    {
                        "default": DEFAULT_ENDPOINT,
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": (
                            "Convert the supplied roleplay scene into one finished image-generation prompt. "
                            "Infer the visible action, pose, anatomy/contact, clothing state, location, lighting, "
                            "expression, and camera composition directly from the roleplay scene. Preserve "
                            "configured character identities and LoRA triggers, but do not treat appearance or "
                            "style tags as evidence for the action. Only an explicitly labeled user-selected "
                            "action override may replace the scene action. Output only the final prompt."
                        ),
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 500, "min": 32, "max": 4096, "step": 16},
                ),
                "timeout_seconds": (
                    "INT",
                    {"default": 120, "min": 5, "max": 600, "step": 5},
                ),
                "fallback_on_error": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Megumin/LLM"
    OUTPUT_NODE = True

    @staticmethod
    def _result(text):
        clean = str(text or "").strip()
        return {
            "ui": {"text": [clean]},
            "result": (clean,),
        }

    def generate(
        self,
        ai_text,
        fallback_text,
        model,
        api_key,
        endpoint,
        system_prompt,
        temperature,
        max_tokens,
        timeout_seconds,
        fallback_on_error,
    ):
        resolved_key = (api_key or os.environ.get("NANOGPT_API_KEY", "")).strip()
        resolved_model = (model or os.environ.get("NANOGPT_MODEL", "")).strip()
        resolved_endpoint = (endpoint or DEFAULT_ENDPOINT).strip()

        if not resolved_key:
            raise ValueError(
                "NanoGPT API key is required. Enter it in the node or set NANOGPT_API_KEY."
            )
        if not resolved_model:
            raise ValueError(
                "NanoGPT model is required. Enter the exact NanoGPT model identifier."
            )
        if not resolved_endpoint.startswith(("http://", "https://")):
            raise ValueError("NanoGPT endpoint must start with http:// or https://.")

        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": str(system_prompt or "").strip()},
                {"role": "user", "content": str(ai_text or "").strip()},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        request = urllib.request.Request(
            resolved_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ComfyUI-Megumin-NanoGPT/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text)
            raise RuntimeError(
                f"NanoGPT returned HTTP {error.code}: {details[:1000]}"
            ) from error
        except urllib.error.URLError as error:
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text)
            raise RuntimeError(f"Could not reach NanoGPT: {error.reason}") from error

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as error:
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text)
            raise RuntimeError(
                f"NanoGPT returned invalid JSON: {response_text[:1000]}"
            ) from error

        text = self._extract_text(data)
        if not text:
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text)
            raise RuntimeError(
                f"NanoGPT response contained no generated text: {response_text[:1000]}"
            )
        return self._result(text)

    @staticmethod
    def _extract_text(data):
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            choice = choices[0] or {}
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            text = choice.get("text")
            if isinstance(text, str):
                return text

        if isinstance(data, dict):
            for key in ("response", "output", "text"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
        return ""


NODE_CLASS_MAPPINGS = {
    "MeguminNanoGPTText": MeguminNanoGPTText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeguminNanoGPTText": "Megumin NanoGPT Text",
}
