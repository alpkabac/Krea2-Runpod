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
                            "configured character identities, but do not treat appearance or "
                            "style tags as evidence for the action. Only an explicitly labeled user-selected "
                            "action override may replace the scene action. Output only the final prompt -- "
                            "never meta-commentary, labels, identity names, file names, or IDs. "
                            "Any LoRA trigger words are appended separately in code and must not be typed here."
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
            },
            "optional": {
                "identity_suffix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Megumin/LLM"
    OUTPUT_NODE = True

    @staticmethod
    def _combine(text, identity_suffix):
        # identity_suffix (e.g. repeated LoRA trigger words) is appended in
        # code, never sent to the LLM. This guarantees the literal trigger
        # phrase lands in the final CLIP-encoded prompt regardless of whether
        # the model succeeded, failed, or mangled the rewrite, and it removes
        # any temptation for a weak/fast model to "helpfully" echo back
        # meta-instruction wording instead of transforming it.
        clean_text = str(text or "").strip()
        clean_suffix = str(identity_suffix or "").strip()
        if not clean_suffix:
            return clean_text
        return f"{clean_text} {clean_suffix}".strip() if clean_text else clean_suffix

    @classmethod
    def _result(cls, text, identity_suffix=""):
        clean = cls._combine(text, identity_suffix)
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
        identity_suffix="",
    ):
        resolved_key = (api_key or os.environ.get("NANOGPT_API_KEY", "")).strip()
        resolved_model = (model or os.environ.get("NANOGPT_MODEL", "")).strip()
        resolved_endpoint = (endpoint or DEFAULT_ENDPOINT).strip()

        # An unsubstituted %placeholder% means the caller never provided a
        # value; treat it the same as empty instead of sending it upstream.
        if resolved_key.startswith("%") and resolved_key.endswith("%"):
            resolved_key = os.environ.get("NANOGPT_API_KEY", "").strip()
        if resolved_model.startswith("%") and resolved_model.endswith("%"):
            resolved_model = os.environ.get("NANOGPT_MODEL", "").strip()

        # Missing config must not kill a render job after a GPU cold start:
        # honor fallback_on_error and render the deterministic fallback text.
        config_error = ""
        if not resolved_key:
            config_error = "NanoGPT API key is required. Enter it in the node, the Megumin UI, or set NANOGPT_API_KEY."
        elif not resolved_model:
            config_error = "NanoGPT model is required. Enter the exact NanoGPT model identifier."
        elif not resolved_endpoint.startswith(("http://", "https://")):
            config_error = "NanoGPT endpoint must start with http:// or https://."
        if config_error:
            if fallback_on_error and str(fallback_text or "").strip():
                print(f"[Megumin NanoGPT] {config_error} Using fallback_text.")
                return self._result(fallback_text, identity_suffix)
            raise ValueError(config_error)

        try:
            resolved_temperature = max(0.0, min(2.0, float(temperature)))
        except (TypeError, ValueError):
            resolved_temperature = 0.2

        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": str(system_prompt or "").strip()},
                {"role": "user", "content": str(ai_text or "").strip()},
            ],
            "temperature": resolved_temperature,
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
                return self._result(fallback_text, identity_suffix)
            raise RuntimeError(
                f"NanoGPT returned HTTP {error.code}: {details[:1000]}"
            ) from error
        except urllib.error.URLError as error:
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text, identity_suffix)
            raise RuntimeError(f"Could not reach NanoGPT: {error.reason}") from error

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as error:
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text, identity_suffix)
            raise RuntimeError(
                f"NanoGPT returned invalid JSON: {response_text[:1000]}"
            ) from error

        text = self._extract_text(data)
        if not text:
            if fallback_on_error and str(fallback_text or "").strip():
                return self._result(fallback_text, identity_suffix)
            raise RuntimeError(
                f"NanoGPT response contained no generated text: {response_text[:1000]}"
            )
        return self._result(text, identity_suffix)

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
