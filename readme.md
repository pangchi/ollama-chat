# Ollama Chat UI

A local Open-WebUI-style chat interface for your Ollama instance.

## Setup

```bash
cd ollama-ui
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Features

- 🔴 Live streaming responses
- 🗂 Persistent chat history (localStorage)
- 🤖 Switch models from the sidebar
- ⬇️ Pull new models with progress log
- 📋 Code blocks with syntax highlighting + copy button
- ✍️ Markdown rendering (tables, lists, headers, inline code)
- ♻️ Clear / delete individual chats

## Config

Edit `app.py` to change the Ollama host:

```python
OLLAMA_BASE = "http://192.168.1.169:11434"
```