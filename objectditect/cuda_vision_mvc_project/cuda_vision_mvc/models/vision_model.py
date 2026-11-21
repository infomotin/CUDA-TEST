import colorsys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)

from config.settings import CATEGORY_BOX_COLORS, CLASSIFICATION_MAP

try:
    from deepface import DeepFace  # type: ignore

    DEEPFACE_AVAILABLE = True
except Exception:
    DeepFace = None  # type: ignore
    DEEPFACE_AVAILABLE = False


class CudaVisionEngine:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA ডিভাইস পাওয়া যায়নি! GTX 1650 ও PyTorch CUDA সেটআপ যাচাই করুন।")
        self.device = torch.device("cuda:0")
        self.gpu_name = torch.cuda.get_device_name(self.device)
        self.total_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        self.model_cache: Dict[str, Any] = {}

    def get_vram_stats(self) -> Dict[str, Any]:
        alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        return {
            "gpu": self.gpu_name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "allocated_vram_mb": round(alloc_mb, 2),
            "reserved_vram_mb": round(reserved_mb, 2),
            "loaded_models": list(self.model_cache.keys()),
        }

    def load_model(self, model_name: str = "fasterrcnn") -> Dict[str, Any]:
        key = model_name.lower().strip()
        if key in self.model_cache:
            return self.model_cache[key]

        torch.cuda.empty_cache()

        if key == "ssdlite":
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = ssdlite320_mobilenet_v3_large(weights=weights)
        else:
            key = "fasterrcnn"
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

        model.to(self.device)
        model.eval()

        dummy = [torch.zeros((3, 320, 320), device=self.device, dtype=torch.float32)]
        with torch.inference_mode():
            _ = model(dummy)
        torch.cuda.synchronize(self.device)

        bundle = {
            "name": key,
            "model": model,
            "categories": weights.meta.get("categories", []),
            "preprocess": weights.transforms(),
        }
        self.model_cache[key] = bundle
        return bundle

    @staticmethod
    def classify_label(label: str) -> str:
        return CLASSIFICATION_MAP.get(label.lower().strip(), "Other Object")

    @staticmethod
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

    @staticmethod
    def _crop_box_safe(
        image: Image.Image,
        box: Tuple[float, float, float, float],
        pad_ratio: float = 0.0,
    ) -> Image.Image:
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

    def detect_dominant_color(self, image: Image.Image, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        crop = self._crop_box_safe(image, box, pad_ratio=0.15).resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.asarray(crop, dtype=np.uint8).reshape(-1, 3)

        quant = (arr // 16) * 16 + 8
        unique_colors, counts = np.unique(quant, axis=0, return_counts=True)
        best_idx = int(np.argmax(counts))
        dom_rgb = tuple(int(c) for c in unique_colors[best_idx])
        return {
            "color_name": self._rgb_to_color_name(dom_rgb),
            "hex": "#{:02X}{:02X}{:02X}".format(*dom_rgb),
            "rgb": list(dom_rgb),
        }

    def analyze_human(self, image: Image.Image, box: Tuple[float, float, float, float]) -> Dict[str, Any]:
        if not DEEPFACE_AVAILABLE or DeepFace is None:
            return {
                "gender": "Not Installed (Use Library Manager)",
                "age": None,
                "age_label": "N/A",
                "estimated": False,
            }

        person_crop = self._crop_box_safe(image, box, pad_ratio=0.05)
        if person_crop.width < 120 or person_crop.height < 120:
            person_crop = person_crop.resize((160, 160), Image.Resampling.BILINEAR)

        try:
            res = DeepFace.analyze(
                img_path=np.asarray(person_crop, dtype=np.uint8),
                actions=["age", "gender"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            item = res[0] if isinstance(res, list) and res else res
            if not isinstance(item, dict):
                return {"gender": "Unknown", "age": None, "age_label": "N/A", "estimated": False}

            raw_g = str(item.get("dominant_gender", "")).strip().lower()
            if raw_g in ("man", "male"):
                gender = "Male (ছেলে/পুরুষ)"
            elif raw_g in ("woman", "female"):
                gender = "Female (মেয়ে/মহিলা)"
            else:
                gender = raw_g or "Unknown"

            age_raw = item.get("age")
            age_int = int(round(float(age_raw))) if age_raw is not None else None
            return {
                "gender": gender,
                "age": age_int,
                "age_label": f"~{age_int} yrs" if age_int is not None else "N/A",
                "estimated": True,
            }
        except Exception:
            return {"gender": "Unknown", "age": None, "age_label": "N/A", "estimated": False}

    @staticmethod
    def _matches_filter(
        class_name: str,
        class_group: str,
        allowed_classes: Optional[set],
        allowed_labels: Optional[set],
    ) -> bool:
        if allowed_classes is not None and class_group.lower() not in allowed_classes:
            return False
        if allowed_labels is not None and class_name.lower() not in allowed_labels:
            return False
        return True

    def run_image_inference(
        self,
        pil_img: Image.Image,
        bundle: Dict[str, Any],
        threshold: float,
        allowed_classes: Optional[set],
        allowed_labels: Optional[set],
        enable_human_attr: bool = True,
    ) -> Dict[str, Any]:
        model = bundle["model"]
        preprocess = bundle["preprocess"]
        categories = bundle["categories"]

        torch.cuda.reset_peak_memory_stats(self.device)

        t_h2d = time.perf_counter()
        input_tensor = preprocess(pil_img).to(self.device, non_blocking=True)
        torch.cuda.synchronize(self.device)
        h2d_ms = (time.perf_counter() - t_h2d) * 1000

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        with torch.inference_mode():
            outputs = model([input_tensor])
        end_evt.record()
        torch.cuda.synchronize(self.device)

        gpu_ms = start_evt.elapsed_time(end_evt)
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        current_vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

        pred = outputs[0]
        boxes = pred["boxes"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        scores = pred["scores"].detach().cpu()
        del input_tensor, outputs, pred

        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        width, height = annotated.size

        detections: List[Dict[str, Any]] = []

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
            class_group = self.classify_label(class_name)

            if not self._matches_filter(class_name, class_group, allowed_classes, allowed_labels):
                continue

            color_info = self.detect_dominant_color(pil_img, (x1_c, y1_c, x2_c, y2_c))
            human_info = (
                self.analyze_human(pil_img, (x1_c, y1_c, x2_c, y2_c))
                if (enable_human_attr and class_name.lower() == "person")
                else None
            )

            box_color = CATEGORY_BOX_COLORS.get(class_group, "#00E5FF")
            draw.rectangle([x1_c, y1_c, x2_c, y2_c], outline=box_color, width=3)

            if human_info and human_info.get("estimated"):
                short_g = (
                    "Male"
                    if "Male" in human_info["gender"]
                    else ("Female" if "Female" in human_info["gender"] else human_info["gender"])
                )
                label_txt = f"{class_name} | {short_g}, {human_info['age_label']} | {color_info['color_name']} ({conf * 100:.1f}%)"
            else:
                label_txt = f"{class_name} | {color_info['color_name']} ({conf * 100:.1f}%)"

            bbox = draw.textbbox((x1_c, y1_c), label_txt, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ly1 = max(0.0, y1_c - th - 6)
            draw.rectangle([x1_c, ly1, x1_c + tw + 8, ly1 + th + 4], fill=box_color)
            draw.text((x1_c + 4, ly1 + 2), label_txt, fill="#081018", font=font)

            detections.append(
                {
                    "label": class_name,
                    "classification": class_group,
                    "color": color_info["color_name"],
                    "color_hex": color_info["hex"],
                    "human_attributes": human_info,
                    "confidence": round(conf * 100, 2),
                    "box": [round(x1_c, 1), round(y1_c, 1), round(x2_c, 1), round(y2_c, 1)],
                }
            )

        return {
            "annotated_image": annotated,
            "detections": detections,
            "h2d_time_ms": round(h2d_ms, 2),
            "gpu_inference_ms": round(gpu_ms, 2),
            "current_vram_mb": round(current_vram_mb, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
