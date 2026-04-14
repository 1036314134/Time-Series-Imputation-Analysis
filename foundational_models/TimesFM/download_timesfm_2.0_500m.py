from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPO_ID = "google/timesfm-2.0-500m-pytorch"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "timesfm_2.0_500m_pytorch"


def download_timesfm_2_0_500m(local_dir: str | Path | None = None) -> Path:
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
    model_dir = download_timesfm_2_0_500m()
    print(f"Downloaded TimesFM 2.0 500M PyTorch model to: {model_dir}")
