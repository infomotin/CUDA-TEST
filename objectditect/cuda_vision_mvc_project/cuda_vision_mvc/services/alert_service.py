import json
import time
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional

NOTIFICATIONS: List[Dict[str, Any]] = []

ALERT_CONFIG: Dict[str, Any] = {
    "alert_on_detection": True,
    "alert_on_no_detection_video": True,
    "watch_labels": [],
    "webhook_url": "",
}


def summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter(item["label"] for item in detections)
    sorted_labels = dict(sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    class_counts = Counter(item["classification"] for item in detections)
    sorted_classes = dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    color_counts = Counter(item["color"] for item in detections)
    sorted_colors = dict(sorted(color_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    gender_counts: Counter = Counter()
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


def send_webhook_alert(payload: Dict[str, Any]) -> None:
    url = (ALERT_CONFIG.get("webhook_url") or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3):
            pass
    except Exception:
        pass


def push_notification(
    level: str,
    title: str,
    message: str,
    alert_triggered: bool = False,
) -> Dict[str, Any]:
    item = {
        "id": len(NOTIFICATIONS) + 1,
        "level": level,
        "title": title,
        "message": message,
        "alert": alert_triggered,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    NOTIFICATIONS.insert(0, item)
    if len(NOTIFICATIONS) > 80:
        del NOTIFICATIONS[80:]

    if alert_triggered:
        send_webhook_alert(item)
    return item


def evaluate_alerts(
    source_type: str,
    detections: List[Dict[str, Any]],
    no_detect_frames: int = 0,
) -> List[Dict[str, Any]]:
    fired: List[Dict[str, Any]] = []
    watch_set = {w.lower() for w in (ALERT_CONFIG.get("watch_labels") or [])}

    matched = [
        d for d in detections if not watch_set or str(d.get("label", "")).lower() in watch_set
    ]

    if ALERT_CONFIG.get("alert_on_detection", True) and len(matched) > 0:
        names = ", ".join(sorted({m["label"] for m in matched})[:5])
        fired.append(
            push_notification(
                "warning",
                f"OBJECT ALERT ({source_type.upper()})",
                f"Detected {len(matched)} matching object(s): {names}",
                alert_triggered=True,
            )
        )

    if (
        source_type == "video"
        and ALERT_CONFIG.get("alert_on_no_detection_video", True)
        and no_detect_frames > 0
    ):
        fired.append(
            push_notification(
                "error",
                "NO DETECTION ALERT (RUNNING VIDEO)",
                f"No matching object detected in {no_detect_frames} processed frame(s)!",
                alert_triggered=True,
            )
        )

    return fired


def update_alert_config(
    alert_on_detection: bool,
    alert_on_no_detection_video: bool,
    watch_labels_csv: str,
    webhook_url: str,
) -> Dict[str, Any]:
    ALERT_CONFIG["alert_on_detection"] = bool(alert_on_detection)
    ALERT_CONFIG["alert_on_no_detection_video"] = bool(alert_on_no_detection_video)
    ALERT_CONFIG["watch_labels"] = [x.strip() for x in (watch_labels_csv or "").split(",") if x.strip()]
    ALERT_CONFIG["webhook_url"] = (webhook_url or "").strip()
    push_notification("info", "Alert Rules Updated", "Alert & No-Detection Video settings updated.")
    return ALERT_CONFIG


def parse_filter_set(raw: Optional[str]) -> Optional[set]:
    if not raw:
        return None
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return set(items) if items else None
