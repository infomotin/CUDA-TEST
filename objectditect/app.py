import base64
from collections import Counter
import colorsys
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# সব মডেল/ক্যাশ অ্যাপ্লিকেশন রানিং ফোল্ডারের ভেতরে রাখার জন্য ENV সেটআপ
# (torch, torchvision এবং deepface ইম্পোর্ট করার আগেই সেট করতে হবে)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOCAL_TORCH_DIR = BASE_DIR / "models" / "torch_hub"
LOCAL_DEEPFACE_DIR = BASE_DIR / "models" / "deepface"
LOCAL_TORCH_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DEEPFACE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(LOCAL_TORCH_DIR)
os.environ["DEEPFACE_HOME"] = str(LOCAL_DEEPFACE_DIR)

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)

# ঐচ্ছিক DeepFace মডিউল (Human Gender & Age অনুমানের জন্য)
try:
    from deepface import DeepFace  # type: ignore

    DEEPFACE_AVAILABLE = True
except Exception:
    DeepFace = None  # type: ignore
    DEEPFACE_AVAILABLE = False

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = FastAPI(title="GTX 1650 CUDA Object, Color, Classification & Human Age/Gender App")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA ডিভাইস পাওয়া যায়নি! GTX 1650 ও PyTorch CUDA সেটআপ যাচাই করুন।")

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(DEVICE)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(DEVICE).total_memory / (1024 ** 3)

MODEL_CACHE: Dict[str, Any] = {}

CLASSIFICATION_MAP: Dict[str, str] = {
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

CATEGORY_BOX_COLORS: Dict[str, str] = {
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


def classify_object(label: str) -> str:
    return CLASSIFICATION_MAP.get(label.lower().strip(), "Other Object")


def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue_deg = h * 360.0

    if v < 0.18:
        return "Black"
    if s < 0.16 and v > 0.82:
        return "White"
    if s < 0.18:
        return "Gray"

    if hue_deg < 15 or hue_deg >= 345:
        return "Red"
    if hue_deg < 40:
        return "Orange"
    if hue_deg < 68:
        return "Yellow"
    if hue_deg < 165:
        return "Green"
    if hue_deg < 200:
        return "Cyan"
    if hue_deg < 255:
        return "Blue"
    if hue_deg < 290:
        return "Purple"
    return "Pink / Magenta"


def _crop_box_safe(image: Image.Image, box: Tuple[float, float, float, float], pad_ratio: float = 0.0) -> Image.Image:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = image.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    if pad_ratio > 0:
        bw = x2 - x1
        bh = y2 - y1
        if bw > 20 and bh > 20:
            px = max(2, int(bw * pad_ratio))
            py = max(2, int(bh * pad_ratio))
            if (x2 - px) > (x1 + px) and (y2 - py) > (y1 + py):
                x1 += px
                x2 -= px
                y1 += py
                y2 -= py

    return image.crop((x1, y1, x2, y2))


def detect_dominant_color(image: Image.Image, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
    crop = _crop_box_safe(image, box, pad_ratio=0.15).resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(crop, dtype=np.uint8).reshape(-1, 3)

    quant = (arr // 16) * 16 + 8
    unique_colors, counts = np.unique(quant, axis=0, return_counts=True)
    best_idx = int(np.argmax(counts))
    dom_rgb = tuple(int(c) for c in unique_colors[best_idx])
    color_name = _rgb_to_color_name(dom_rgb)
    hex_color = "#{:02X}{:02X}{:02X}".format(*dom_rgb)

    return {
        "color_name": color_name,
        "hex": hex_color,
        "rgb": list(dom_rgb),
    }


def analyze_human_attributes(image: Image.Image, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """
    person ডিটেক্ট হলে ক্রপ করা অংশ থেকে সম্ভাব্য Gender (Male/Female) এবং Age অনুমান করে।
    """
    if not DEEPFACE_AVAILABLE or DeepFace is None:
        return {
            "gender": "Unknown (Install deepface)",
            "age": None,
            "age_label": "N/A",
            "estimated": False,
        }

    person_crop = _crop_box_safe(image, box, pad_ratio=0.05)
    # খুব ছোট ক্রপ হলে রিসাইজ করে বিশ্লেষণে সুবিধা দেওয়া
    if person_crop.width < 120 or person_crop.height < 120:
        person_crop = person_crop.resize((160, 160), Image.Resampling.BILINEAR)

    rgb_arr = np.asarray(person_crop, dtype=np.uint8)

    try:
        result = DeepFace.analyze(
            img_path=rgb_arr,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        item = result[0] if isinstance(result, list) and result else result
        if not isinstance(item, dict):
            return {"gender": "Unknown", "age": None, "age_label": "N/A", "estimated": False}

        raw_gender = str(item.get("dominant_gender", "")).strip()
        if raw_gender.lower() in ("man", "male"):
            gender_label = "Male (ছেলে/পুরুষ)"
        elif raw_gender.lower() in ("woman", "female"):
            gender_label = "Female (মেয়ে/মহিলা)"
        else:
            gender_label = raw_gender or "Unknown"

        age_val = item.get("age")
        age_int: Optional[int] = int(round(float(age_val))) if age_val is not None else None
        age_label = f"~{age_int} yrs" if age_int is not None else "N/A"

        return {
            "gender": gender_label,
            "age": age_int,
            "age_label": age_label,
            "estimated": True,
        }
    except Exception:
        return {
            "gender": "Unknown",
            "age": None,
            "age_label": "N/A",
            "estimated": False,
        }


def load_detection_model(model_name: str = "fasterrcnn"):
    model_name = model_name.lower().strip()
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    torch.cuda.empty_cache()

    if model_name == "ssdlite":
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights)
    else:
        model_name = "fasterrcnn"
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

    model.to(DEVICE)
    model.eval()

    dummy = [torch.zeros((3, 320, 320), device=DEVICE, dtype=torch.float32)]
    with torch.inference_mode():
        _ = model(dummy)
    torch.cuda.synchronize(DEVICE)

    categories = weights.meta.get("categories", [])
    preprocess = weights.transforms()

    bundle = {
        "name": model_name,
        "model": model,
        "categories": categories,
        "preprocess": preprocess,
    }
    MODEL_CACHE[model_name] = bundle
    return bundle


def download_all_models():
    for m_name in ("fasterrcnn", "ssdlite"):
        load_detection_model(m_name)


@app.on_event("startup")
def startup_event():
    download_all_models()


def draw_detections(image: Image.Image, boxes, labels, scores, categories: List[str], threshold: float):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    detections = []
    width, height = annotated.size

    for box, label_idx, score in zip(boxes, labels, scores):
        conf = float(score.item())
        if conf < threshold:
            continue

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1_c = max(0.0, min(x1, float(width)))
        y1_c = max(0.0, min(y1, float(height)))
        x2_c = max(0.0, min(x2, float(width)))
        y2_c = max(0.0, min(y2, float(height)))

        idx = int(label_idx.item())
        class_name = categories[idx] if 0 <= idx < len(categories) else f"Class_{idx}"
        category_group = classify_object(class_name)
        color_info = detect_dominant_color(image, (x1_c, y1_c, x2_c, y2_c))

        human_info: Optional[Dict[str, Any]] = None
        if class_name.lower() == "person":
            human_info = analyze_human_attributes(image, (x1_c, y1_c, x2_c, y2_c))

        box_color = CATEGORY_BOX_COLORS.get(category_group, "#00E5FF")
        draw.rectangle([x1_c, y1_c, x2_c, y2_c], outline=box_color, width=3)

        if human_info and human_info.get("estimated"):
            short_g = "Male" if "Male" in human_info["gender"] else ("Female" if "Female" in human_info["gender"] else human_info["gender"])
            text = f"{class_name} | {short_g}, {human_info['age_label']} | {color_info['color_name']} ({conf * 100:.1f}%)"
        else:
            text = f"{class_name} | {color_info['color_name']} ({conf * 100:.1f}%)"

        bbox = draw.textbbox((x1_c, y1_c), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_y1 = max(0.0, y1_c - text_h - 6)
        draw.rectangle(
            [x1_c, label_y1, x1_c + text_w + 8, label_y1 + text_h + 4],
            fill=box_color,
        )
        draw.text((x1_c + 4, label_y1 + 2), text, fill="#081018", font=font)

        detections.append(
            {
                "label": class_name,
                "classification": category_group,
                "color": color_info["color_name"],
                "color_hex": color_info["hex"],
                "human_attributes": human_info,
                "confidence": round(conf * 100, 2),
                "box": [round(x1_c, 1), round(y1_c, 1), round(x2_c, 1), round(y2_c, 1)],
            }
        )

    return annotated, detections


def summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter(item["label"] for item in detections)
    sorted_labels = dict(sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    class_counts = Counter(item["classification"] for item in detections)
    sorted_classes = dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    color_counts = Counter(item["color"] for item in detections)
    sorted_colors = dict(sorted(color_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    gender_counts = Counter()
    for item in detections:
        h_attr = item.get("human_attributes")
        if h_attr and h_attr.get("gender"):
            gender_counts[h_attr["gender"]] += 1
    sorted_genders = dict(sorted(gender_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "unique_count": len(sorted_labels),
        "unique_labels": list(sorted_labels.keys()),
        "counts_by_label": sorted_labels,
        "counts_by_classification": sorted_classes,
        "counts_by_color": sorted_colors,
        "counts_by_gender": sorted_genders,
    }


@app.get("/health")
def health_check():
    alloc_mb = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)
    return {
        "status": "ok",
        "gpu": GPU_NAME,
        "total_vram_gb": round(TOTAL_VRAM_GB, 2),
        "allocated_vram_mb": round(alloc_mb, 2),
        "reserved_vram_mb": round(reserved_mb, 2),
        "torch_model_dir": str(LOCAL_TORCH_DIR),
        "deepface_model_dir": str(LOCAL_DEEPFACE_DIR),
        "deepface_available": DEEPFACE_AVAILABLE,
        "loaded_models": list(MODEL_CACHE.keys()),
    }


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.95),
    model_name: str = Query("fasterrcnn"),
    max_side: int = Query(1024, ge=320, le=1920),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="অনুগ্রহ করে একটি বৈধ ইমেজ ফাইল (JPG, PNG, WEBP) আপলোড করুন।")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="ইমেজ ফাইলটি খালি।")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="ইমেজটি ডিকোড করা যাচ্ছে না।")

    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    bundle = load_detection_model(model_name)
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    categories = bundle["categories"]

    torch.cuda.reset_peak_memory_stats(DEVICE)

    t_h2d_start = time.perf_counter()
    input_tensor = preprocess(pil_img).to(DEVICE, non_blocking=True)
    torch.cuda.synchronize(DEVICE)
    h2d_ms = (time.perf_counter() - t_h2d_start) * 1000

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    try:
        start_evt.record()
        with torch.inference_mode():
            outputs = model([input_tensor])
        end_evt.record()
        torch.cuda.synchronize(DEVICE)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU VRAM পূর্ণ হয়ে গেছে! Max Image Side কমিয়ে (যেমন 640) আবার চেষ্টা করুন।",
        )

    gpu_ms = start_evt.elapsed_time(end_evt)
    peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    current_vram_mb = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)

    pred = outputs[0]
    boxes = pred["boxes"].detach().cpu()
    labels = pred["labels"].detach().cpu()
    scores = pred["scores"].detach().cpu()

    del input_tensor, outputs, pred

    annotated_img, detections = draw_detections(pil_img, boxes, labels, scores, categories, threshold)
    summary = summarize_detections(detections)

    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG", quality=90)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "model": bundle["name"],
        "gpu": GPU_NAME,
        "deepface_available": DEEPFACE_AVAILABLE,
        "image_size": {"width": pil_img.width, "height": pil_img.height},
        "h2d_time_ms": round(h2d_ms, 2),
        "gpu_inference_ms": round(gpu_ms, 2),
        "current_vram_mb": round(current_vram_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "total_objects": len(detections),
        "unique_count": summary["unique_count"],
        "unique_labels": summary["unique_labels"],
        "counts_by_label": summary["counts_by_label"],
        "counts_by_classification": summary["counts_by_classification"],
        "counts_by_color": summary["counts_by_color"],
        "counts_by_gender": summary["counts_by_gender"],
        "detections": detections,
        "annotated_image_base64": f"data:image/jpeg;base64,{b64_img}",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="bn">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GTX 1650 CUDA Object, Color, Classification & Human Age/Gender</title>
  <style>
    :root {
      --bg: #0b1020;
      --card: #131b33;
      --accent: #00e5ff;
      --text: #eef2ff;
      --muted: #9aa7c7;
      --border: #253259;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: linear-gradient(135deg, var(--bg), #0f172a);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      padding: 18px 28px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      background: rgba(19, 27, 51, 0.85);
      backdrop-filter: blur(8px);
    }
    .brand { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
    .gpu-badge {
      background: rgba(0, 229, 255, 0.12);
      border: 1px solid rgba(0, 229, 255, 0.4);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
    }
    .container {
      max-width: 1260px;
      margin: 24px auto;
      padding: 0 20px;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 22px;
    }
    @media (max-width: 920px) {
      .container { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .card h3 { margin-top: 0; margin-bottom: 14px; font-size: 1rem; color: var(--accent); }
    .field { margin-bottom: 14px; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
    select, input[type="file"], input[type="range"] {
      width: 100%;
      padding: 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0c1224;
      color: var(--text);
    }
    .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
    button {
      cursor: pointer;
      border: none;
      border-radius: 9px;
      padding: 10px 14px;
      font-weight: 600;
      font-size: 0.9rem;
      transition: 0.2s;
    }
    .btn-primary {
      background: linear-gradient(90deg, var(--accent), #00b8d4);
      color: #061018;
      flex: 1;
    }
    .btn-secondary {
      background: #1d294d;
      color: var(--text);
      border: 1px solid var(--border);
    }
    button:hover { opacity: 0.92; transform: translateY(-1px); }
    .preview-box {
      width: 100%;
      min-height: 350px;
      border-radius: 12px;
      border: 1px dashed var(--border);
      background: #090e1c;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .preview-box img, .preview-box video {
      max-width: 100%;
      max-height: 520px;
      object-fit: contain;
      display: block;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    @media (max-width: 900px) {
      .stats { grid-template-columns: repeat(2, 1fr); }
    }
    .stat-item {
      background: #0c1224;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }
    .stat-label { font-size: 0.75rem; color: var(--muted); }
    .stat-val { font-size: 1.05rem; font-weight: 700; margin-top: 4px; color: var(--accent); }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 14px;
    }
    @media (max-width: 1050px) {
      .summary-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 650px) {
      .summary-grid { grid-template-columns: 1fr; }
    }
    .summary-box {
      background: #0c1224;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
    }
    .summary-title { font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; font-weight: 600; }
    .badge-wrap { display: flex; flex-wrap: wrap; gap: 7px; }
    .obj-badge {
      background: rgba(0, 229, 255, 0.12);
      border: 1px solid rgba(0, 229, 255, 0.35);
      color: var(--text);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .obj-badge span.cnt { color: var(--accent); }
    .color-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid rgba(255,255,255,0.5);
      display: inline-block;
    }
    .det-list {
      margin-top: 14px;
      max-height: 270px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .det-item {
      background: #0c1224;
      border: 1px solid var(--border);
      padding: 9px 12px;
      border-radius: 8px;
      font-size: 0.84rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .det-sub { font-size: 0.76rem; color: var(--muted); margin-top: 3px; line-height: 1.4; }
    .status-msg { font-size: 0.85rem; color: var(--muted); margin-top: 8px; min-height: 20px; }
  </style>
</head>
<body>
  <header>
    <div class="brand">⚡ CUDA Object, Color, Classification & Human Age/Gender</div>
    <div class="gpu-badge" id="gpuInfo">GPU স্ট্যাটাস লোড হচ্ছে...</div>
  </header>

  <div class="container">
    <div class="card">
      <h3>১. ইনপুট ও কন্ট্রোল</h3>

      <div class="field">
        <label>মডেল নির্বাচন করুন (4 GB VRAM অপ্টিমাইজড)</label>
        <select id="modelSelect">
          <option value="fasterrcnn">Faster R-CNN MobileNetV3 (Balanced Accuracy)</option>
          <option value="ssdlite">SSDLite MobileNetV3 (Ultra Fast)</option>
        </select>
      </div>

      <div class="field">
        <label>Confidence Threshold: <span id="threshVal">0.50</span></label>
        <input type="range" id="threshInput" min="0.1" max="0.95" step="0.05" value="0.5" />
      </div>

      <div class="field">
        <label>Max Image Side (VRAM Guard): <span id="sideVal">1024</span>px</label>
        <input type="range" id="sideInput" min="320" max="1600" step="64" value="1024" />
      </div>

      <div class="field">
        <label>ছবি আপলোড করুন (JPG, PNG, WEBP)</label>
        <input type="file" id="fileInput" accept="image/*" />
      </div>

      <div class="btn-row">
        <button class="btn-primary" id="detectBtn">🔍 Analyze Objects</button>
        <button class="btn-secondary" id="camBtn">📷 Webcam</button>
        <button class="btn-secondary" id="snapBtn" disabled>📸 Capture</button>
      </div>

      <div class="status-msg" id="statusMsg"></div>

      <div class="det-list" id="detList"></div>
    </div>

    <div class="card">
      <h3>২. আউটপুট, ক্লাসিফিকেশন, রঙ ও Human বিশ্লেষণ</h3>
      <div class="preview-box" id="previewBox">
        <span style="color: var(--muted);">একটি ছবি আপলোড করুন অথবা Webcam থেকে ক্যাপচার করুন</span>
      </div>

      <div class="stats">
        <div class="stat-item">
          <div class="stat-label">GPU Inference</div>
          <div class="stat-val" id="gpuTime">-</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">H2D Transfer</div>
          <div class="stat-val" id="h2dTime">-</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Peak VRAM</div>
          <div class="stat-val" id="peakVram">-</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Total Objects</div>
          <div class="stat-val" id="objCount">-</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Unique Objects</div>
          <div class="stat-val" id="uniqueCount">-</div>
        </div>
      </div>

      <div class="summary-grid">
        <div class="summary-box">
          <div class="summary-title">Unique Objects & Count</div>
          <div class="badge-wrap" id="uniqueBadges">
            <span style="color: var(--muted); font-size: 0.82rem;">এখনো কোনো ডেটা নেই</span>
          </div>
        </div>

        <div class="summary-box">
          <div class="summary-title">Object Classification</div>
          <div class="badge-wrap" id="classBadges">
            <span style="color: var(--muted); font-size: 0.82rem;">এখনো কোনো ডেটা নেই</span>
          </div>
        </div>

        <div class="summary-box">
          <div class="summary-title">Detected Colors</div>
          <div class="badge-wrap" id="colorBadges">
            <span style="color: var(--muted); font-size: 0.82rem;">এখনো কোনো ডেটা নেই</span>
          </div>
        </div>

        <div class="summary-box">
          <div class="summary-title">Human Gender Summary</div>
          <div class="badge-wrap" id="genderBadges">
            <span style="color: var(--muted); font-size: 0.82rem;">কোনো Human ডেটা নেই</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const gpuInfo = document.getElementById("gpuInfo");
    const modelSelect = document.getElementById("modelSelect");
    const threshInput = document.getElementById("threshInput");
    const threshVal = document.getElementById("threshVal");
    const sideInput = document.getElementById("sideInput");
    const sideVal = document.getElementById("sideVal");
    const fileInput = document.getElementById("fileInput");
    const detectBtn = document.getElementById("detectBtn");
    const camBtn = document.getElementById("camBtn");
    const snapBtn = document.getElementById("snapBtn");
    const statusMsg = document.getElementById("statusMsg");
    const previewBox = document.getElementById("previewBox");
    const detList = document.getElementById("detList");
    const uniqueBadges = document.getElementById("uniqueBadges");
    const classBadges = document.getElementById("classBadges");
    const colorBadges = document.getElementById("colorBadges");
    const genderBadges = document.getElementById("genderBadges");

    let selectedFile = null;
    let videoEl = null;
    let mediaStream = null;

    threshInput.addEventListener("input", () => threshVal.textContent = Number(threshInput.value).toFixed(2));
    sideInput.addEventListener("input", () => sideVal.textContent = sideInput.value);

    async function loadHealth() {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        const dfStatus = data.deepface_available ? "Age/Gender: ON" : "Age/Gender: Off";
        gpuInfo.textContent = `${data.gpu} | VRAM: ${data.allocated_vram_mb} / ${(data.total_vram_gb * 1024).toFixed(0)} MB | ${dfStatus}`;
      } catch (e) {
        gpuInfo.textContent = "GPU তথ্য পাওয়া যায়নি";
      }
    }
    loadHealth();

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        selectedFile = e.target.files[0];
        stopCamera();
        const url = URL.createObjectURL(selectedFile);
        previewBox.innerHTML = `<img src="${url}" alt="Preview" />`;
        statusMsg.textContent = `ফাইল নির্বাচিত: ${selectedFile.name}`;
      }
    });

    function stopCamera() {
      if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
      }
      snapBtn.disabled = true;
      videoEl = null;
    }

    camBtn.addEventListener("click", async () => {
      try {
        stopCamera();
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoEl = document.createElement("video");
        videoEl.srcObject = mediaStream;
        videoEl.autoplay = true;
        videoEl.playsInline = true;
        previewBox.innerHTML = "";
        previewBox.appendChild(videoEl);
        snapBtn.disabled = false;
        statusMsg.textContent = "ওয়েবক্যাম চালু হয়েছে। Capture চাপুন।";
      } catch (err) {
        statusMsg.textContent = "ওয়েবক্যাম অ্যাক্সেস করা যায়নি: " + err.message;
      }
    });

    snapBtn.addEventListener("click", () => {
      if (!videoEl) return;
      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth || 640;
      canvas.height = videoEl.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          selectedFile = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
          stopCamera();
          previewBox.innerHTML = `<img src="${URL.createObjectURL(selectedFile)}" alt="Captured" />`;
          statusMsg.textContent = "ওয়েবক্যাম থেকে ছবি ক্যাপচার করা হয়েছে।";
        }
      }, "image/jpeg", 0.9);
    });

    function renderBadges(container, entries, withDot = false, colorLookup = {}, emptyText = "কোনো ডেটা নেই") {
      if (!entries || entries.length === 0) {
        container.innerHTML = `<span style="color: var(--muted); font-size: 0.82rem;">${emptyText}</span>`;
        return;
      }
      container.innerHTML = entries.map(([key, cnt]) => {
        const dot = withDot
          ? `<span class="color-dot" style="background:${colorLookup[key] || '#00e5ff'}"></span>`
          : "";
        return `<div class="obj-badge">${dot}${key}<span class="cnt">×${cnt}</span></div>`;
      }).join("");
    }

    detectBtn.addEventListener("click", async () => {
      if (!selectedFile) {
        statusMsg.textContent = "আগে একটি ছবি আপলোড বা ক্যাপচার করুন!";
        return;
      }
      detectBtn.disabled = true;
      statusMsg.textContent = "GTX 1650 জিপিইউতে অবজেক্ট, রঙ, ক্লাসিফিকেশন ও Human বিশ্লেষণ চলছে...";

      const formData = new FormData();
      formData.append("file", selectedFile);

      const params = new URLSearchParams({
        threshold: threshInput.value,
        model_name: modelSelect.value,
        max_side: sideInput.value,
      });

      try {
        const res = await fetch(`/detect?${params.toString()}`, {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || "ডিটেকশন ব্যর্থ হয়েছে");
        }

        previewBox.innerHTML = `<img src="${data.annotated_image_base64}" alt="Detected Result" />`;
        document.getElementById("gpuTime").textContent = `${data.gpu_inference_ms} ms`;
        document.getElementById("h2dTime").textContent = `${data.h2d_time_ms} ms`;
        document.getElementById("peakVram").textContent = `${data.peak_vram_mb} MB`;
        document.getElementById("objCount").textContent = data.total_objects;
        document.getElementById("uniqueCount").textContent = data.unique_count;

        renderBadges(uniqueBadges, Object.entries(data.counts_by_label || {}), false);
        renderBadges(classBadges, Object.entries(data.counts_by_classification || {}), false);

        const colorHexMap = {};
        (data.detections || []).forEach(d => {
          if (d.color && d.color_hex && !colorHexMap[d.color]) {
            colorHexMap[d.color] = d.color_hex;
          }
        });
        renderBadges(colorBadges, Object.entries(data.counts_by_color || {}), true, colorHexMap);
        renderBadges(genderBadges, Object.entries(data.counts_by_gender || {}), false, {}, "কোনো Human ডেটা নেই");

        detList.innerHTML = "";
        if (!data.detections || data.detections.length === 0) {
          detList.innerHTML = `<div class="det-item"><span>কোনো অবজেক্ট পাওয়া যায়নি (Threshold কমাতে পারেন)</span></div>`;
        } else {
          data.detections.forEach((d, idx) => {
            let humanLine = "";
            if (d.label.toLowerCase() === "person" && d.human_attributes) {
              humanLine = `<br/>Human: <strong>${d.human_attributes.gender}</strong> • Est. Age: <strong>${d.human_attributes.age_label}</strong>`;
            }
            const div = document.createElement("div");
            div.className = "det-item";
            div.innerHTML = `
              <div>
                <div><strong>#${idx + 1} ${d.label}</strong></div>
                <div class="det-sub">
                  Class: ${d.classification} • Color: <span class="color-dot" style="background:${d.color_hex}"></span> ${d.color}
                  ${humanLine}
                </div>
              </div>
              <strong>${d.confidence}%</strong>
            `;
            detList.appendChild(div);
          });
        }

        statusMsg.textContent = `সম্পন্ন (${data.model}) — ${data.total_objects} টি মোট ও ${data.unique_count} টি Unique অবজেক্ট শনাক্ত হয়েছে।`;
        loadHealth();
      } catch (err) {
        statusMsg.textContent = "❌ ত্রুটি: " + err.message;
      } finally {
        detectBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)