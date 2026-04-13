import os
import json

FILE = "insights.json"

def save_insight(file, content):
    data = []

    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            data = json.load(f)

    data.append({"file": file, "content": content})

    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_insights():
    if not os.path.exists(FILE):
        return ""

    with open(FILE, "r") as f:
        data = json.load(f)

    return "\n\n".join([d["content"] for d in data])