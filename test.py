# test.py
# Terminal chat client for the FastAPI backend (SSE).
# Lets you test /api/chat without the React frontend.

import os
import sys
import json
import time
import signal
import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CHAT_URL = f"{API_BASE}/api/chat"
HEALTH_URL = f"{API_BASE}/api/health"

# Conversation history in the same shape the backend expects
history = []  # [{ "role": "user"|"assistant"|"system", "content": "..." }]

use_docs = False  # toggle RAG usage from the terminal


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
        device = data.get("device", "unknown")
        print(f"Health OK → model: {model} | device: {device}")
    except Exception as e:
        print(f"Health check failed: {e}")


def sse_chat(message: str, history_list, use_docs_flag: bool) -> str:
    """
    Sends a chat request and streams tokens from the backend.
    Returns the full assistant response text.
    """
    payload = {
        "message": message,
        "history": history_list,
        "useDocs": use_docs_flag,
    }
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(CHAT_URL, json=payload, headers=headers, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                # Non-stream (JSON error)
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                print(f"\n[HTTP {resp.status_code}] {err}")
                return ""

            # Parse SSE lines
            assistant_text = []
            current_event = None

            print("\nAssistant: ", end="", flush=True)
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()

                # Blank line = end of an SSE event block
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
                        # Keep-alives or non-JSON payloads can occur
                        continue

                    if current_event == "start":
                        # Optional: print meta info
                        # print(f"\n[stream started: {data}]")
                        pass
                    elif current_event == "error":
                        print(f"\n[error] {data.get('error','unknown error')}")
                    elif current_event == "end":
                        # Optional: print token count
                        # print(f"\n[stream ended: {data}]")
                        pass
                    else:
                        # Normal token chunk
                        token = data.get("token", "")
                        if token:
                            assistant_text.append(token)
                            print(token, end="", flush=True)

            print()  # newline after stream ends
            return "".join(assistant_text)

    except requests.exceptions.RequestException as e:
        print(f"\n[network error] {e}")
        return ""


def handle_command(cmd: str) -> bool:
    """Returns True if the caller should continue; False to quit."""
    global use_docs, history

    if cmd == "/quit":
        return False
    if cmd == "/rag on":
        use_docs = True
        print("RAG: ON (Use web snippets)")
        return True
    if cmd == "/rag off":
        use_docs = False
        print("RAG: OFF")
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

    # Graceful Ctrl+C
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

        # Append user message to history
        history.append({"role": "user", "content": user})

        # Call backend and stream
        reply = sse_chat(user, history_list=history, use_docs_flag=use_docs)

        if reply:
            # Append assistant response to history
            history.append({"role": "assistant", "content": reply})
        else:
            # Remove the last user message if request failed (optional)
            pass


if __name__ == "__main__":
    main()
