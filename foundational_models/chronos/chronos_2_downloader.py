from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPO_ID = "amazon/chronos-2"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "chronos_2"


def download_chronos_2(local_dir: str | Path | None = None) -> Path:
    target_dir = Path(local_dir) if local_dir is not None else LOCAL_MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_dir


if __name__ == "__main__":
    model_dir = download_chronos_2()
    print(f"Downloaded Chronos-2 model to: {model_dir}")
