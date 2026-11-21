import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LOCAL_TORCH_DIR = BASE_DIR / "models" / "torch_hub"
LOCAL_DEEPFACE_DIR = BASE_DIR / "models" / "deepface"
DOWNLOADS_DIR = BASE_DIR / "downloads"
DATA_DIR = BASE_DIR / "data"
SQLITE_DB_PATH = DATA_DIR / "vision_results.db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

for directory in (
    LOCAL_TORCH_DIR,
    LOCAL_DEEPFACE_DIR,
    DOWNLOADS_DIR,
    DATA_DIR,
    TEMPLATES_DIR,
    STATIC_DIR / "css",
    STATIC_DIR / "js",
):
    directory.mkdir(parents=True, exist_ok=True)


def configure_runtime_env() -> None:
    """torch, torchvision এবং deepface ইম্পোর্ট করার আগেই ক্যাশ ও GPU env সেট করে।"""
    os.environ["TORCH_HOME"] = str(LOCAL_TORCH_DIR)
    os.environ["DEEPFACE_HOME"] = str(LOCAL_DEEPFACE_DIR)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


ALLOWED_PIP_PACKAGES = {
    "opencv-python": "opencv-python",
    "deepface": "deepface",
    "tf-keras": "tf-keras",
    "scipy": "scipy",
    "yt-dlp": "yt-dlp",
}

CLASSIFICATION_MAP = {
    "person": "Person / Human",
    "bicycle": "Vehicle",
    "car": "Vehicle",
    "motorcycle": "Vehicle",
    "airplane": "Vehicle",
    "bus": "Vehicle",
    "train": "Vehicle",
    "truck": "Vehicle",
    "boat": "Vehicle",
    "traffic light": "Outdoor / Street",
    "fire hydrant": "Outdoor / Street",
    "stop sign": "Outdoor / Street",
    "parking meter": "Outdoor / Street",
    "bench": "Furniture & Home",
    "bird": "Animal",
    "cat": "Animal",
    "dog": "Animal",
    "horse": "Animal",
    "sheep": "Animal",
    "cow": "Animal",
    "elephant": "Animal",
    "bear": "Animal",
    "zebra": "Animal",
    "giraffe": "Animal",
    "backpack": "Accessories & Bag",
    "umbrella": "Accessories & Bag",
    "handbag": "Accessories & Bag",
    "tie": "Accessories & Bag",
    "suitcase": "Accessories & Bag",
    "frisbee": "Sports & Gear",
    "skis": "Sports & Gear",
    "snowboard": "Sports & Gear",
    "sports ball": "Sports & Gear",
    "kite": "Sports & Gear",
    "baseball bat": "Sports & Gear",
    "baseball glove": "Sports & Gear",
    "skateboard": "Sports & Gear",
    "surfboard": "Sports & Gear",
    "tennis racket": "Sports & Gear",
    "bottle": "Food & Drink",
    "wine glass": "Food & Drink",
    "cup": "Food & Drink",
    "fork": "Food & Drink",
    "knife": "Food & Drink",
    "spoon": "Food & Drink",
    "bowl": "Food & Drink",
    "banana": "Food & Drink",
    "apple": "Food & Drink",
    "sandwich": "Food & Drink",
    "orange": "Food & Drink",
    "broccoli": "Food & Drink",
    "carrot": "Food & Drink",
    "hot dog": "Food & Drink",
    "pizza": "Food & Drink",
    "donut": "Food & Drink",
    "cake": "Food & Drink",
    "chair": "Furniture & Home",
    "couch": "Furniture & Home",
    "potted plant": "Furniture & Home",
    "bed": "Furniture & Home",
    "dining table": "Furniture & Home",
    "toilet": "Furniture & Home",
    "tv": "Electronics & Appliances",
    "laptop": "Electronics & Appliances",
    "mouse": "Electronics & Appliances",
    "remote": "Electronics & Appliances",
    "keyboard": "Electronics & Appliances",
    "cell phone": "Electronics & Appliances",
    "microwave": "Electronics & Appliances",
    "oven": "Electronics & Appliances",
    "toaster": "Electronics & Appliances",
    "sink": "Electronics & Appliances",
    "refrigerator": "Electronics & Appliances",
    "book": "Stationery & Books",
    "clock": "Furniture & Home",
    "vase": "Furniture & Home",
    "scissors": "Stationery & Books",
    "teddy bear": "Animal",
    "hair drier": "Electronics & Appliances",
    "toothbrush": "Personal Care",
}

CATEGORY_BOX_COLORS = {
    "Person / Human": "#00E5FF",
    "Vehicle": "#FF9100",
    "Animal": "#69F0AE",
    "Electronics & Appliances": "#B388FF",
    "Furniture & Home": "#FFD740",
    "Food & Drink": "#FF5252",
    "Accessories & Bag": "#40C4FF",
    "Sports & Gear": "#FF80AB",
    "Outdoor / Street": "#A7FFEB",
    "Stationery & Books": "#CFD8DC",
    "Personal Care": "#80D8FF",
    "Other Object": "#E0E0E0",
}
