from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="thuml/sundial-base-128m",
    local_dir="./sundial_base_128m",
    proxies={
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }
)