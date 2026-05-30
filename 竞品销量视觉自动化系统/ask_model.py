"""截图 + 问minicpm-v"""
import base64, requests, sys

img_path = sys.argv[1] if len(sys.argv) > 1 else "captures/now.png"
question = sys.argv[2] if len(sys.argv) > 2 else "这是什么页面？"

with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:11434/api/generate", json={
    "model": "minicpm-v:latest", "prompt": question, "images": [b64],
    "stream": False, "options": {"temperature": 0, "num_predict": 64}
}, timeout=120)

print(resp.json()["response"].strip())
