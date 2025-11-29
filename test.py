# test.py — Terminal chat client for the FastAPI backend (SSE)

import os
import sys
import json
import signal
from dotenv import load_dotenv
import requests

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://3.27.241.187:8000")
CHAT_URL = f"{API_BASE}/api/chat"
HEALTH_URL = f"{API_BASE}/api/health"

history = []
use_docs = True


def print_banner():
    print("=" * 70)
    print(" SheBots CLI — FastAPI backend tester (SSE streaming) ".center(70, "="))
    print("=" * 70)
    print(f"Backend: {API_BASE}")
    print("Commands: /rag on | /rag off | /clear | /history | /quit")
    print("-" * 70)


def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        model = data.get("model", "unknown")
        rag_ready = data.get("rag_ready", False)
        print(f"Health OK → model: {model} | RAG: {rag_ready}")
    except Exception as e:
        print(f"Health check failed: {e}")


def sse_chat(message: str, history_list, use_docs_flag: bool) -> str:
    payload = {
        "message": message,
        "history": history_list,
        "useDocs": use_docs_flag,
    }
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(CHAT_URL, json=payload, headers=headers, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                print(f"\n[HTTP {resp.status_code}] {err}")
                return ""

            assistant_text = []
            current_event = None

            print("\nAssistant: ", end="", flush=True)
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()

                if line == "":
                    current_event = None
                    continue

                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue

                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "start":
                        pass
                    elif current_event == "error":
                        print(f"\n[error] {data.get('error','unknown error')}")
                    elif current_event == "end":
                        pass
                    else:
                        token = data.get("token", "")
                        if token:
                            assistant_text.append(token)
                            print(token, end="", flush=True)

            print()
            return "".join(assistant_text)

    except requests.exceptions.RequestException as e:
        print(f"\n[network error] {e}")
        return ""


def handle_command(cmd: str) -> bool:
    global use_docs, history

    if cmd == "/quit":
        return False
    if cmd == "/rag on":
        use_docs = True
        print("RAG enabled.")
        return True
    if cmd == "/rag off":
        use_docs = False
        print("RAG disabled.")
        return True
    if cmd == "/clear":
        history = []
        print("History cleared.")
        return True
    if cmd == "/history":
        if not history:
            print("[empty history]")
        else:
            for i, m in enumerate(history, 1):
                print(f"{i:02d}. {m['role']}: {m['content'][:120]}{'…' if len(m['content'])>120 else ''}")
        return True

    print("Unknown command. Try: /rag on | /rag off | /clear | /history | /quit")
    return True


def main():
    print_banner()
    check_health()

    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    while True:
        try:
            user = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user:
            continue

        if user.startswith("/"):
            if not handle_command(user):
                break
            else:
                continue

        history.append({"role": "user", "content": user})
        reply = sse_chat(user, history_list=history, use_docs_flag=use_docs)

        if reply:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()