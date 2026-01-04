import json
import os


def fingerprint_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "fingerprint.json")


def load_fingerprint(persist_dir: str) -> dict | None:
    path = fingerprint_path(persist_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_fingerprint(persist_dir: str, fingerprint: dict):
    os.makedirs(persist_dir, exist_ok=True)
    with open(fingerprint_path(persist_dir), "w") as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)


def fingerprint_matches(persist_dir: str, expected: dict) -> bool:
    stored = load_fingerprint(persist_dir)
    return stored == expected
