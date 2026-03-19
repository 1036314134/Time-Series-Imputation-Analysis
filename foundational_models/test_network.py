import requests

proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

try:
    r = requests.get("https://huggingface.co", proxies=proxies, timeout=10)
    print("成功:", r.status_code)
except Exception as e:
    print("失败:", e)