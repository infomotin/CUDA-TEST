import base64
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from config.settings import ALLOWED_PIP_PACKAGES, DOWNLOADS_DIR
from models.db_model import save_result_to_sqlite
from models.vision_model import CudaVisionEngine
from services.alert_service import evaluate_alerts, parse_filter_set, push_notification, summarize_detections

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False


def install_allowed_package(package_key: str) -> Dict[str, Any]:
    key = package_key.lower().strip()
    if key not in ALLOWED_PIP_PACKAGES:
        raise ValueError(f"অনুমোদিত প্যাকেজ: {list(ALLOWED_PIP_PACKAGES.keys())}")

    pip_target = ALLOWED_PIP_PACKAGES[key]
    push_notification("info", "Library Install", f"Installing '{pip_target}' via pip...")

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_target],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "")[-400:]
        push_notification("error", "Install Failed", f"{pip_target} failed: {err_tail}")
        raise RuntimeError(f"{pip_target} ইনস্টল করা যায়নি: {err_tail}")

    global CV2_AVAILABLE, cv2
    if key == "opencv-python":
        try:
            import cv2 as _cv2  # type: ignore

            cv2 = _cv2
            CV2_AVAILABLE = True
        except Exception:
            pass

    push_notification("success", "Library Installed", f"'{pip_target}' installed successfully!")
    return {"status": "ok", "package": pip_target, "cv2_available": CV2_AVAILABLE}


def list_download_files() -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    if DOWNLOADS_DIR.exists():
        for p in sorted(DOWNLOADS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file():
                files.append(
                    {
                        "filename": p.name,
                        "size_kb": round(p.stat().st_size / 1024, 2),
                        "download_url": f"/api/downloads/{p.name}",
                    }
                )
    return files


def save_json_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_name = f"detection_report_{ts}.json"
    out_path = DOWNLOADS_DIR / report_name
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    push_notification("success", "Saved to ./downloads", f"JSON report saved as {report_name}")
    return {"status": "ok", "filename": report_name, "download_url": f"/api/downloads/{report_name}"}


def process_image_detection(
    engine: CudaVisionEngine,
    raw_bytes: bytes,
    threshold: float,
    model_name: str,
    max_side: int,
    allowed_classifications: Optional[str],
    target_labels: Optional[str],
    enable_human_attr: bool,
    save_to_downloads: bool,
    save_to_db: bool,
) -> Dict[str, Any]:
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    bundle = engine.load_model(model_name)
    inf = engine.run_image_inference(
        pil_img=pil_img,
        bundle=bundle,
        threshold=threshold,
        allowed_classes=parse_filter_set(allowed_classifications),
        allowed_labels=parse_filter_set(target_labels),
        enable_human_attr=enable_human_attr,
    )

    detections = inf["detections"]
    annotated_img = inf["annotated_image"]
    summary = summarize_detections(detections)

    saved_filename: Optional[str] = None
    if save_to_downloads:
        ts = time.strftime("%Y%m%d_%H%M%S")
        saved_filename = f"detected_{ts}.jpg"
        annotated_img.save(DOWNLOADS_DIR / saved_filename, format="JPEG", quality=92)
        push_notification("success", "Image Saved", f"Saved to ./downloads/{saved_filename}")

    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG", quality=88)
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    record_id: Optional[int] = None
    if save_to_db:
        record_id = save_result_to_sqlite(
            source_type="image",
            model_name=bundle["name"],
            total_objects=len(detections),
            unique_count=summary["unique_count"],
            gpu_ms=inf["gpu_inference_ms"],
            peak_vram_mb=inf["peak_vram_mb"],
            saved_file=saved_filename,
            summary=summary,
            detections=detections,
            preview_base64=data_uri,
        )

    fired_alerts = evaluate_alerts("image", detections, no_detect_frames=0)

    return {
        "db_record_id": record_id,
        "model": bundle["name"],
        "gpu": engine.gpu_name,
        "saved_file": saved_filename,
        "alerts_fired": fired_alerts,
        "image_size": {"width": pil_img.width, "height": pil_img.height},
        "h2d_time_ms": inf["h2d_time_ms"],
        "gpu_inference_ms": inf["gpu_inference_ms"],
        "current_vram_mb": inf["current_vram_mb"],
        "peak_vram_mb": inf["peak_vram_mb"],
        "total_objects": len(detections),
        "unique_count": summary["unique_count"],
        "unique_labels": summary["unique_labels"],
        "counts_by_label": summary["counts_by_label"],
        "counts_by_classification": summary["counts_by_classification"],
        "counts_by_color": summary["counts_by_color"],
        "counts_by_gender": summary["counts_by_gender"],
        "detections": detections,
        "annotated_image_base64": data_uri,
    }


def process_video_detection(
    engine: CudaVisionEngine,
    raw_bytes: bytes,
    original_filename: Optional[str],
    threshold: float,
    model_name: str,
    max_side: int,
    frame_stride: int,
    max_frames: int,
    allowed_classifications: Optional[str],
    target_labels: Optional[str],
    enable_human_attr: bool,
    save_video_to_downloads: bool,
    save_to_db: bool,
) -> Dict[str, Any]:
    if not CV2_AVAILABLE or cv2 is None:
        raise RuntimeError("ভিডিও প্রসেসিংয়ের জন্য 'opencv-python' প্রয়োজন। Library Manager থেকে ইনস্টল করুন।")

    suffix = Path(original_filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(raw_bytes)
        in_path = Path(tmp_in.name)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        in_path.unlink(missing_ok=True)
        raise ValueError("ভিডিও ফাইলটি ওপেন করা যাচ্ছে না।")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0:
        fps = 15.0
    out_fps = max(5.0, round(fps / frame_stride, 2))

    bundle = engine.load_model(model_name)
    allowed_cls_set = parse_filter_set(allowed_classifications)
    allowed_lbl_set = parse_filter_set(target_labels)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_filename = f"video_detected_{ts}.avi"
    out_path = DOWNLOADS_DIR / out_filename

    writer = None
    all_detections: List[Dict[str, Any]] = []
    processed_frames = 0
    no_detect_frames = 0
    raw_frame_idx = 0
    total_gpu_ms = 0.0
    peak_vram_overall = 0.0
    sample_preview_uri: Optional[str] = None

    push_notification("info", "Video Processing Started", f"Processing '{original_filename or 'video'}'...")

    try:
        while processed_frames < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if raw_frame_idx % frame_stride != 0:
                raw_frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            w, h = pil_img.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

            inf = engine.run_image_inference(
                pil_img=pil_img,
                bundle=bundle,
                threshold=threshold,
                allowed_classes=allowed_cls_set,
                allowed_labels=allowed_lbl_set,
                enable_human_attr=enable_human_attr,
            )

            ann_rgb = np.asarray(inf["annotated_image"], dtype=np.uint8)
            ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (ann_bgr.shape[1], ann_bgr.shape[0]))

            if save_video_to_downloads and writer is not None:
                writer.write(ann_bgr)

            if sample_preview_uri is None:
                buf = io.BytesIO()
                inf["annotated_image"].save(buf, format="JPEG", quality=85)
                sample_preview_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            if len(inf["detections"]) == 0:
                no_detect_frames += 1

            for det in inf["detections"]:
                det_copy = dict(det)
                det_copy["frame"] = processed_frames + 1
                all_detections.append(det_copy)

            total_gpu_ms += inf["gpu_inference_ms"]
            peak_vram_overall = max(peak_vram_overall, inf["peak_vram_mb"])
            processed_frames += 1
            raw_frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        in_path.unlink(missing_ok=True)

    if not save_video_to_downloads and out_path.exists():
        out_path.unlink(missing_ok=True)
        out_filename = None

    summary = summarize_detections(all_detections)
    avg_gpu_ms = round(total_gpu_ms / processed_frames, 2) if processed_frames > 0 else 0.0

    record_id: Optional[int] = None
    if save_to_db:
        record_id = save_result_to_sqlite(
            source_type="video",
            model_name=bundle["name"],
            total_objects=len(all_detections),
            unique_count=summary["unique_count"],
            gpu_ms=avg_gpu_ms,
            peak_vram_mb=round(peak_vram_overall, 2),
            saved_file=out_filename,
            summary=summary,
            detections=all_detections,
            preview_base64=sample_preview_uri,
        )

    fired_alerts = evaluate_alerts("video", all_detections, no_detect_frames=no_detect_frames)
    push_notification(
        "success",
        "Video Detection Complete",
        f"Processed {processed_frames} frames ({no_detect_frames} with no detection).",
    )

    return {
        "db_record_id": record_id,
        "model": bundle["name"],
        "gpu": engine.gpu_name,
        "processed_frames": processed_frames,
        "no_detect_frames": no_detect_frames,
        "avg_gpu_inference_ms": avg_gpu_ms,
        "peak_vram_mb": round(peak_vram_overall, 2),
        "saved_video_file": out_filename,
        "download_url": f"/api/downloads/{out_filename}" if out_filename else None,
        "alerts_fired": fired_alerts,
        "total_detections_across_frames": len(all_detections),
        "unique_count": summary["unique_count"],
        "unique_labels": summary["unique_labels"],
        "counts_by_label": summary["counts_by_label"],
        "counts_by_classification": summary["counts_by_classification"],
        "counts_by_color": summary["counts_by_color"],
        "counts_by_gender": summary["counts_by_gender"],
        "sample_frame_base64": sample_preview_uri,
    }



def resolve_youtube_video_bytes(youtube_url: str) -> Tuple[bytes, str]:
    url = (youtube_url or "").strip()
    if not url:
        raise ValueError("অনুগ্রহ করে একটি বৈধ YouTube URL দিন।")

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_tmpl = str(Path(tmp_dir) / "yt_video.%(ext)s")
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "mp4/best[ext=mp4]/best",
            "--no-playlist",
            "-o",
            out_tmpl,
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            err_tail = (proc.stderr or proc.stdout or "")[-350:]
            raise RuntimeError(f"YouTube ভিডিও লোড করা যায়নি: {err_tail}")

        downloaded = list(Path(tmp_dir).glob("yt_video.*"))
        if not downloaded:
            raise RuntimeError("YouTube থেকে ভিডিও ফাইল পাওয়া যায়নি।")

        video_file = downloaded[0]
        return video_file.read_bytes(), video_file.name