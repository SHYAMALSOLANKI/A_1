import os
import sys
import time
import subprocess
from pathlib import Path

import requests


STACK_SERVER_URL = os.getenv("STACK_SERVER_URL", "http://127.0.0.1:8000")


def wait_for_health(proc: subprocess.Popen, timeout_s: int = 20) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("Stack server exited before becoming healthy.")
        try:
            resp = requests.get(f"{STACK_SERVER_URL}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError("Stack server did not become healthy in time.")


def run_smoke():
    prompts = [
        "Calculate 1 + 1.",
        "Explain why the sky is blue.",
        "The capital of France is Paris."
    ]

    for prompt in prompts:
        draft = f"DRAFT: {prompt} The result of 1+1 is 3."
        audit = requests.post(
            f"{STACK_SERVER_URL}/audit",
            json={"draft_text": draft, "metadata": {"prompt": prompt}},
            timeout=5
        ).json()

        reflection = audit.get("reflection_prompt", "")
        learned = audit.get("learned_rule", "")
        revision = f"Revised answer: {prompt}"

        trace_text = (
            f"DRAFT:\n{draft}\n\n"
            f"REFLECT:\n{reflection}\n\n"
            f"REVISE:\n{revision}\n\n"
            f"LEARNED:\n{learned}"
        )

        mem_resp = requests.post(
            f"{STACK_SERVER_URL}/memory/add",
            json={"learned": learned, "metadata": {"prompt": prompt}},
            timeout=5
        )
        assert mem_resp.status_code == 200

        trace_resp = requests.post(
            f"{STACK_SERVER_URL}/trace/store",
            json={
                "prompt": prompt,
                "draft": draft,
                "reflect": reflection,
                "revise": revision,
                "learned": learned,
                "scores": {"draft_score": audit.get("score", 0.0)},
                "metadata": {"trace": trace_text}
            },
            timeout=5
        )
        assert trace_resp.status_code == 200

        query_resp = requests.post(
            f"{STACK_SERVER_URL}/memory/query",
            json={"prompt": prompt, "top_k": 2},
            timeout=5
        )
        data = query_resp.json()
        assert isinstance(data.get("results"), list)

    print("Self-play smoke test passed.")


if __name__ == "__main__":
    env = os.environ.copy()
    env["STACK_SERVER_PORT"] = "8000"
    env["ENABLE_GRAMMAR"] = "false"
    env["ENABLE_NLI"] = "false"
    env["ENABLE_WEB_SOT"] = env.get("ENABLE_WEB_SOT", "false")
    log_path = Path(__file__).resolve().parents[1] / "stack_server_smoke.log"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.stack_server.api:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_health(proc)
        run_smoke()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
