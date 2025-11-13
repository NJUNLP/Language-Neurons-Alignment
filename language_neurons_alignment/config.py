import os

MODEL_NAME = "Mistral"

LANG_MAPPING = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "bn": "Bengali",
    "th": "Thai",
    "sw": "Swahili"
}

def set_model_name(name: str):
    global MODEL_NAME
    MODEL_NAME = name

def resolve_model_path(model_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(model_path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path not found: {path}")
    return path

def get_suffix_from_path(model_path: str) -> str:
    base = os.path.basename(os.path.normpath(model_path)) or "model"
    return f"-{base}"

def get_langs():
    return list(LANG_MAPPING.keys())
