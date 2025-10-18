import json
import os


def load_json(path, mode='r', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        data = json.load(f)
    return data

def save_json(data, path, mode='w', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

