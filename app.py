from flask import Flask, render_template, request, Response, stream_with_context
import requests
import json
import configparser
import os
import time
import sys
import threading

# Load config
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

OLLAMA_HOST    = config.get("ollama", "host",                fallback="127.0.0.1")
OLLAMA_PORT    = config.get("ollama", "port",                fallback="11434")
OLLAMA_BASE    = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
RETRY_INTERVAL = config.getfloat("ollama", "retry_interval", fallback=3)
RETRY_TIMEOUT  = config.getfloat("ollama", "retry_timeout",  fallback=60)

FLASK_HOST  = config.get("flask", "host",           fallback="0.0.0.0")
FLASK_PORT  = config.getint("flask", "port",        fallback=5000)
FLASK_DEBUG = config.getboolean("flask", "debug",   fallback=False)

# ── Connection state ──
_ollama_up = False
_state_lock = threading.Lock()
_state_listeners: list[threading.Event] = []


def _set_state(up: bool):
    global _ollama_up
    with _state_lock:
        changed = _ollama_up != up
        _ollama_up = up
        if changed:
            for ev in _state_listeners:
                ev.set()


def _subscribe() -> threading.Event:
    ev = threading.Event()
    with _state_lock:
        _state_listeners.append(ev)
    return ev


def _unsubscribe(ev: threading.Event):
    with _state_lock:
        _state_listeners.discard(ev) if hasattr(_state_listeners, 'discard') else None
        try:
            _state_listeners.remove(ev)
        except ValueError:
            pass


def _monitor_loop():
    """Background thread: continuously probe Ollama and update _ollama_up."""
    global _ollama_up
    url = f"{OLLAMA_BASE}/api/tags"
    while True:
        try:
            r = requests.get(url, timeout=3)
            up = r.ok
        except requests.exceptions.RequestException:
            up = False

        prev = _ollama_up
        _set_state(up)
        if up != prev:
            print(f"[ollama] {'✓ Connected' if up else '✗ Disconnected'} ({OLLAMA_HOST}:{OLLAMA_PORT})")

        time.sleep(RETRY_INTERVAL)


def wait_for_ollama():
    """Block startup until Ollama is ready (or retry_timeout exceeded)."""
    url = f"{OLLAMA_BASE}/api/tags"
    deadline = time.time() + RETRY_TIMEOUT
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(url, timeout=3)
            if r.ok:
                _set_state(True)
                print(f"[ollama] ✓ Ready ({OLLAMA_HOST}:{OLLAMA_PORT})")
                return
        except requests.exceptions.RequestException:
            pass
        remaining = deadline - time.time()
        if remaining <= 0:
            print(
                f"[ollama] ✗ Host not reachable after {RETRY_TIMEOUT}s — starting anyway.",
                file=sys.stderr,
            )
            return
        print(
            f"[ollama] Waiting for {OLLAMA_HOST}:{OLLAMA_PORT} "
            f"(attempt {attempt}, {remaining:.0f}s left)…"
        )
        time.sleep(min(RETRY_INTERVAL, remaining))


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def list_models():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}, 502


@app.route("/api/status")
def status_stream():
    """SSE stream that emits the current connection state and every change."""
    def generate():
        # Send current state immediately
        with _state_lock:
            current = _ollama_up
        yield f"data: {json.dumps({'up': current})}\n\n"

        ev = _subscribe()
        try:
            while True:
                # Wait for a state change (with heartbeat every 15 s)
                triggered = ev.wait(timeout=15)
                if triggered:
                    ev.clear()
                    with _state_lock:
                        current = _ollama_up
                    yield f"data: {json.dumps({'up': current})}\n\n"
                else:
                    # Heartbeat to keep the connection alive
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            _unsubscribe(ev)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    model = data.get("model", "llama3")
    messages = data.get("messages", [])
    stream = data.get("stream", True)

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": True,   # enable native thinking field for qwen3/deepseek-r1 etc.
    }

    def generate():
        try:
            with requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
    )


@app.route("/api/pull", methods=["POST"])
def pull_model():
    data = request.json
    name = data.get("name", "")

    def generate():
        try:
            with requests.post(
                f"{OLLAMA_BASE}/api/pull",
                json={"name": name, "stream": True},
                stream=True,
                timeout=600,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


def find_free_port(start: int, max_tries: int = 100) -> int:
    """Return the first free TCP port >= start."""
    import socket
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + max_tries}")


def update_config_port(new_port: int):
    """Persist the chosen port back into config.ini."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.ini")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    cfg.set("flask", "port", str(new_port))
    with open(cfg_path, "w") as f:
        cfg.write(f)
    print(f"[flask] config.ini updated → port = {new_port}")


if __name__ == "__main__":
    # When Flask debug/reloader is on, it spawns a child process with this env
    # var set. Port selection must only happen in the parent (first) process so
    # the reloader child inherits the already-chosen port instead of bumping it
    # again each restart.
    in_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    if not in_reloader_child:
        port = find_free_port(FLASK_PORT)
        if port != FLASK_PORT:
            print(f"[flask] Port {FLASK_PORT} in use — using {port} instead.")
            update_config_port(port)
        else:
            print(f"[flask] Port {port} is free.")
        # Export so the reloader child picks up the same port
        os.environ["FLASK_RUN_PORT"] = str(port)
    else:
        port = int(os.environ.get("FLASK_RUN_PORT", FLASK_PORT))

    wait_for_ollama()
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()

    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=port)