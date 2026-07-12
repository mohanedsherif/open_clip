from __future__ import annotations
import json, os, platform, subprocess, sys
from pathlib import Path


def _version(module: str) -> str | None:
    try: return __import__(module).__version__
    except (ImportError, AttributeError): return None

def main() -> None:
    result = {"os": platform.platform(), "architecture": platform.machine(), "python": platform.python_version(), "python_executable": sys.executable, "venv": os.environ.get("VIRTUAL_ENV"), "cwd": str(Path.cwd()), "cpu_count": os.cpu_count()}
    for key, command in {"git_commit": ["git", "rev-parse", "HEAD"], "git_branch": ["git", "branch", "--show-current"], "pip": [sys.executable, "-m", "pip", "--version"]}.items():
        try: result[key] = subprocess.check_output(command, text=True).strip()
        except (OSError, subprocess.CalledProcessError): result[key] = None
    try:
        import torch
        result.update({"torch": torch.__version__, "torchvision": _version("torchvision"), "timm": _version("timm"), "transformers": _version("transformers"), "pillow": _version("PIL"), "numpy": _version("numpy"), "pandas": _version("pandas"), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(), "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())})
        result["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result.update({"gpu": props.name, "gpu_memory_gib": round(props.total_memory / 2**30, 2), "gpu_compute_capability": list(torch.cuda.get_device_capability(0)), "gpu_allocated_bytes": torch.cuda.memory_allocated(0), "gpu_reserved_bytes": torch.cuda.memory_reserved(0), "bf16_supported": torch.cuda.is_bf16_supported(), "fp16_supported": True})
    except ImportError: result["torch"] = None
    try:
        import open_clip, open_clip_train
        result.update({"open_clip": open_clip.__version__, "open_clip_path": open_clip.__file__, "open_clip_train_path": open_clip_train.__file__})
        if "open_clip\\src\\open_clip" not in str(Path(open_clip.__file__).resolve()).lower().replace("/", "\\"):
            raise RuntimeError("open_clip is not imported from this repository's editable source; run pip install -e '.[training]'.")
    except ImportError: result["open_clip"] = None
    output = Path("outputs/environment/environment.json"); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True)); print(f"Saved: {output}")

if __name__ == "__main__": main()
