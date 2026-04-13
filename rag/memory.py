import json
from pathlib import Path
from collections import deque

CHAT_PATH = Path("data/chats")

class ChatMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_context(self):
        return "\n".join(
            [f"{m['role']}: {m['content']}" for m in self.history]
        )

def save_message(chat_id, role, content):
    CHAT_PATH.mkdir(parents=True, exist_ok=True)
    file = CHAT_PATH / f"{chat_id}.json"

    if file.exists():
        data = json.loads(file.read_text())
    else:
        data = []

    data.append({"role": role, "content": content})

    file.write_text(json.dumps(data, indent=2))

def load_chat(chat_id):
    file = CHAT_PATH / f"{chat_id}.json"
    if file.exists():
        return json.loads(file.read_text())
    return []