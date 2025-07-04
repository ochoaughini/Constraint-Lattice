"""Helper to download Gemma-2B-IT and Phi-2 weights into hf-cache/.
Run once then set HF_HUB_OFFLINE=1 to force offline mode.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

CACHE = Path(__file__).resolve().parent.parent / "hf-cache"
CACHE.mkdir(exist_ok=True)

for model in ("google/gemma-2-2b-it", "microsoft/phi-2"):
    snapshot_download(model, local_dir=str(CACHE), local_dir_use_symlinks=False)
print("✓ Models cached in", CACHE)
