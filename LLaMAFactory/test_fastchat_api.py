import requests
import threading
import time
from typing import Dict

def get_img_from_instruction(prompt: str, info: Dict[str, str], height=384, width=384, num_inference_steps=40, guidance_scale=7.5):
    port = '8001'
    img_path = requests.post(f"http://localhost:{port}/generate", json={"prompt": prompt, "h": height, "w": width, "save_dir": "./test", "info": info, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale}).json()["path"]
    return img_path


results = []
lock = threading.Lock()

def thread_task(prompt, info):
    global results
    try:
        img_path = get_img_from_instruction(prompt, info)
        with lock:
            results.append(img_path)
    except Exception as e:
        with lock:
            results.append(f"An error occurred: {e}")

prompts = [
    "A beautiful landscape",
    "A cute animal"
]
infos = [
    {"type": "landscape"},
    {"type": "animal"}
]

threads = []
for i in range(2):
    thread = threading.Thread(target=thread_task, args=(prompts[i], infos[i]))
    threads.append(thread)
    thread.start()

while any(thread.is_alive() for thread in threads) or results:
    with lock:
        while results:
            print("Received image path:", results.pop(0))
    time.sleep(0.1)

for thread in threads:
    thread.join()