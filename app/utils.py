import os
from pathlib import Path

def get_model_path():
    base_dir = Path(__file__).resolve().parent.parent
    return os.path.join(base_dir, "model", "model.pth")