"""Safe, on-demand Hugging Face LoRA loading for the Megumin Krea RunPod worker.

The normal ComfyUI LoRA widgets validate against a list built at process startup.
That makes them unsuitable for serverless jobs which first need to download a
selected file.  This node accepts explicit hf:// references, downloads only an
allowlisted repository, then applies the LoRA directly to the model and CLIP.
"""

import hashlib
import os
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import comfy.sd
import comfy.utils
import folder_paths


ALLOWED_REPOSITORIES = {
    repo.strip().lower()
    for repo in os.environ.get("MEGUMIN_RUNTIME_LORA_REPOS", "malcolmrey/krea2").split(",")
    if repo.strip()
}
MAX_LORA_BYTES = int(os.environ.get("MEGUMIN_RUNTIME_LORA_MAX_BYTES", str(1024 * 1024 * 1024)))
DOWNLOAD_LOCK = threading.Lock()


def _safe_hf_reference(reference):
    """Return (repository, filename) for an allowlisted hf:// reference."""
    value = str(reference or "").strip()
    if not value or value.lower() == "none":
        return None
    if not value.startswith("hf://"):
        return None

    parsed = urllib.parse.urlparse(value)
    # hf://malcolmrey/krea2/file.safetensors -> netloc=malcolmrey, path=/krea2/file
    pieces = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    if len(pieces) < 3:
        raise ValueError("Runtime LoRA references must use hf://owner/repository/filename.safetensors")
    repository = f"{pieces[0]}/{pieces[1]}".lower()
    filename = "/".join(pieces[2:])
    if repository not in ALLOWED_REPOSITORIES:
        raise ValueError(f"Runtime LoRA repository is not allowed: {repository}")
    if not filename.lower().endswith(".safetensors") or ".." in filename.split("/"):
        raise ValueError("Runtime LoRA filename must be a safe .safetensors path")
    return repository, filename


def _runtime_directory():
    base = Path(folder_paths.get_folder_paths("loras")[0])
    path = base / "megumin_runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_hf_lora(repository, filename):
    cache_key = hashlib.sha256(f"{repository}/{filename}".encode("utf-8")).hexdigest()[:16]
    target = _runtime_directory() / f"{cache_key}-{Path(filename).name}"
    if target.is_file() and target.stat().st_size > 0:
        return target

    url = f"https://huggingface.co/{repository}/resolve/main/{urllib.parse.quote(filename)}"
    with DOWNLOAD_LOCK:
        if target.is_file() and target.stat().st_size > 0:
            return target
        request = urllib.request.Request(url, headers={"User-Agent": "Megumin-RunPod-RuntimeLora/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            declared_size = int(response.headers.get("Content-Length", "0") or 0)
            if declared_size > MAX_LORA_BYTES:
                raise ValueError(f"Runtime LoRA exceeds the {MAX_LORA_BYTES} byte safety limit")
            fd, temp_path = tempfile.mkstemp(prefix=".megumin-", suffix=".part", dir=target.parent)
            written = 0
            try:
                with os.fdopen(fd, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_LORA_BYTES:
                            raise ValueError(f"Runtime LoRA exceeds the {MAX_LORA_BYTES} byte safety limit")
                        output.write(chunk)
                if written == 0:
                    raise ValueError("Runtime LoRA download was empty")
                os.replace(temp_path, target)
            except Exception:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                raise
    return target


def _resolve_lora(reference):
    remote = _safe_hf_reference(reference)
    if remote:
        return _download_hf_lora(*remote)

    # Local names are only used for the baked-in default LoRA.  folder_paths
    # confines the lookup to ComfyUI's LoRA directory.
    value = str(reference or "").strip()
    if not value or value.lower() == "none":
        return None
    path = folder_paths.get_full_path("loras", value)
    if not path:
        raise ValueError(f"Local LoRA was not found: {value}")
    return Path(path)


class MeguminRuntimeLoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_01": ("STRING", {"default": "None", "multiline": False}),
                "strength_01": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_02": ("STRING", {"default": "None", "multiline": False}),
                "strength_02": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_03": ("STRING", {"default": "None", "multiline": False}),
                "strength_03": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_04": ("STRING", {"default": "None", "multiline": False}),
                "strength_04": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "Megumin/RunPod"

    def apply(self, model, clip, lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, lora_04, strength_04):
        for reference, strength in (
            (lora_01, strength_01), (lora_02, strength_02), (lora_03, strength_03), (lora_04, strength_04),
        ):
            path = _resolve_lora(reference)
            if not path or float(strength) == 0:
                continue
            lora = comfy.utils.load_torch_file(str(path), safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, float(strength), float(strength))
        return (model, clip)
