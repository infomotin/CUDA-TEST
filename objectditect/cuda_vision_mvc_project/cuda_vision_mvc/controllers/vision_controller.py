from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config.settings import (
    ALLOWED_PIP_PACKAGES,
    CATEGORY_BOX_COLORS,
    CLASSIFICATION_MAP,
    DOWNLOADS_DIR,
    LOCAL_DEEPFACE_DIR,
    LOCAL_TORCH_DIR,
    SQLITE_DB_PATH,
    TEMPLATES_DIR,
)
from models.db_model import get_sqlite_result_detail, list_sqlite_results
from models.vision_model import DEEPFACE_AVAILABLE, CudaVisionEngine
from services.alert_service import ALERT_CONFIG, NOTIFICATIONS, push_notification, update_alert_config
from services.media_service import (
    CV2_AVAILABLE,
    install_allowed_package,
    list_download_files,
    process_image_detection,
    process_video_detection,
    save_json_report,
)

router = APIRouter()
engine = CudaVisionEngine()


class ModelDownloadRequest(BaseModel):
    model_name: str = "fasterrcnn"


class LibraryInstallRequest(BaseModel):
    package: str


class AlertConfigRequest(BaseModel):
    alert_on_detection: bool = True
    alert_on_no_detection_video: bool = True
    watch_labels_csv: str = ""
    webhook_url: str = ""


def initialize_controller() -> None:
    push_notification("info", "MVC Studio Ready", f"Initialized on {engine.gpu_name}.")
    engine.load_model("fasterrcnn")


@router.get("/", response_class=HTMLResponse)
def view_home():
    index_file = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@router.get("/health")
def health():
    stats = engine.get_vram_stats()
    stats.update(
        {
            "status": "ok",
            "torch_model_dir": str(LOCAL_TORCH_DIR),
            "deepface_model_dir": str(LOCAL_DEEPFACE_DIR),
            "downloads_dir": str(DOWNLOADS_DIR),
            "sqlite_db_path": str(SQLITE_DB_PATH),
            "cv2_available": CV2_AVAILABLE,
            "deepface_available": DEEPFACE_AVAILABLE,
            "available_classifications": sorted(list(set(CLASSIFICATION_MAP.values()) | {"Other Object"})),
            "installable_packages": list(ALLOWED_PIP_PACKAGES.keys()),
            "alert_config": ALERT_CONFIG,
        }
    )
    return stats


@router.get("/api/notifications")
def api_get_notifications():
    return {"notifications": NOTIFICATIONS}


@router.post("/api/notifications/clear")
def api_clear_notifications():
    NOTIFICATIONS.clear()
    return {"status": "cleared"}


@router.get("/api/alerts/config")
def api_get_alert_config():
    return {"alert_config": ALERT_CONFIG}


@router.post("/api/alerts/config")
def api_set_alert_config(req: AlertConfigRequest):
    cfg = update_alert_config(
        alert_on_detection=req.alert_on_detection,
        alert_on_no_detection_video=req.alert_on_no_detection_video,
        watch_labels_csv=req.watch_labels_csv,
        webhook_url=req.webhook_url,
    )
    return {"status": "ok", "alert_config": cfg}


@router.get("/api/db/results")
def api_db_results(limit: int = Query(25, ge=1, le=200)):
    return {"db_path": str(SQLITE_DB_PATH), "results": list_sqlite_results(limit=limit)}


@router.get("/api/db/results/{record_id}")
def api_db_result_detail(record_id: int):
    row = get_sqlite_result_detail(record_id)
    if not row:
        raise HTTPException(status_code=404, detail="রেকর্ডটি পাওয়া যায়নি।")
    return row


@router.post("/api/models/download")
def api_download_model(req: ModelDownloadRequest):
    m_name = req.model_name.lower().strip()
    if m_name not in ("fasterrcnn", "ssdlite", "all"):
        raise HTTPException(status_code=400, detail="সমর্থিত মডেল: fasterrcnn, ssdlite অথবা all")

    push_notification("info", "Model Download", f"Downloading/loading '{m_name}' into ./models/torch_hub...")
    if m_name == "all":
        engine.load_model("fasterrcnn")
        engine.load_model("ssdlite")
    else:
        engine.load_model(m_name)
    return {"status": "ok", "loaded_models": list(engine.model_cache.keys()), "model_dir": str(LOCAL_TORCH_DIR)}


@router.post("/api/library/install")
def api_install_library(req: LibraryInstallRequest):
    try:
        return install_allowed_package(req.package)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/downloads")
def api_list_downloads():
    return {"downloads_dir": str(DOWNLOADS_DIR), "files": list_download_files()}


@router.get("/api/downloads/{filename}")
def api_download_file(filename: str):
    safe_name = Path(filename).name
    file_path = DOWNLOADS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="ফাইলটি পাওয়া যায়নি।")
    return FileResponse(path=str(file_path), filename=safe_name)


@router.post("/api/save-report")
async def api_save_report(request: Request):
    payload: Dict[str, Any] = await request.json()
    return save_json_report(payload)


@router.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.95),
    model_name: str = Query("fasterrcnn"),
    max_side: int = Query(1024, ge=320, le=1920),
    allowed_classifications: Optional[str] = Query(None),
    target_labels: Optional[str] = Query(None),
    enable_human_attr: bool = Query(True),
    save_to_downloads: bool = Query(False),
    save_to_db: bool = Query(True),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="অনুগ্রহ করে একটি বৈধ ইমেজ ফাইল আপলোড করুন।")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="ইমেজ ফাইলটি খালি।")

    try:
        return process_image_detection(
            engine=engine,
            raw_bytes=raw_bytes,
            threshold=threshold,
            model_name=model_name,
            max_side=max_side,
            allowed_classifications=allowed_classifications,
            target_labels=target_labels,
            enable_human_attr=enable_human_attr,
            save_to_downloads=save_to_downloads,
            save_to_db=save_to_db,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.95),
    model_name: str = Query("fasterrcnn"),
    max_side: int = Query(640, ge=320, le=1280),
    frame_stride: int = Query(3, ge=1, le=20),
    max_frames: int = Query(150, ge=10, le=600),
    allowed_classifications: Optional[str] = Query(None),
    target_labels: Optional[str] = Query(None),
    enable_human_attr: bool = Query(False),
    save_video_to_downloads: bool = Query(True),
    save_to_db: bool = Query(True),
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="ভিডিও ফাইলটি খালি।")

    try:
        return process_video_detection(
            engine=engine,
            raw_bytes=raw_bytes,
            original_filename=file.filename,
            threshold=threshold,
            model_name=model_name,
            max_side=max_side,
            frame_stride=frame_stride,
            max_frames=max_frames,
            allowed_classifications=allowed_classifications,
            target_labels=target_labels,
            enable_human_attr=enable_human_attr,
            save_video_to_downloads=save_video_to_downloads,
            save_to_db=save_to_db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

from fastapi import Form

@router.post("/detect-youtube")
async def detect_youtube_video(
    youtube_url: str = Form(...),
    threshold: float = Query(0.5, ge=0.1, le=0.95),
    model_name: str = Query("fasterrcnn"),
    max_side: int = Query(640, ge=320, le=1280),
    frame_stride: int = Query(3, ge=1, le=20),
    max_frames: int = Query(120, ge=10, le=500),
    allowed_classifications: Optional[str] = Query(None),
    target_labels: Optional[str] = Query(None),
    enable_human_attr: bool = Query(False),
    save_video_to_downloads: bool = Query(True),
    save_to_db: bool = Query(True),
):
    try:
        raw_bytes, fname = resolve_youtube_video_bytes(youtube_url)
        return process_video_detection(
            engine=engine,
            raw_bytes=raw_bytes,
            original_filename=fname,
            threshold=threshold,
            model_name=model_name,
            max_side=max_side,
            frame_stride=frame_stride,
            max_frames=max_frames,
            allowed_classifications=allowed_classifications,
            target_labels=target_labels,
            enable_human_attr=enable_human_attr,
            save_video_to_downloads=save_video_to_downloads,
            save_to_db=save_to_db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))