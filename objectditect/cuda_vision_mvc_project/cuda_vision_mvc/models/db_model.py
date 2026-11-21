import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from config.settings import SQLITE_DB_PATH


def init_sqlite_db() -> None:
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                total_objects INTEGER NOT NULL,
                unique_count INTEGER NOT NULL,
                gpu_ms REAL NOT NULL,
                peak_vram_mb REAL NOT NULL,
                saved_file TEXT,
                summary_json TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                preview_base64 TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_result_to_sqlite(
    source_type: str,
    model_name: str,
    total_objects: int,
    unique_count: int,
    gpu_ms: float,
    peak_vram_mb: float,
    saved_file: Optional[str],
    summary: Dict[str, Any],
    detections: List[Dict[str, Any]],
    preview_base64: Optional[str],
) -> int:
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    try:
        cur = conn.execute(
            """
            INSERT INTO detection_results (
                created_at, source_type, model_name, total_objects, unique_count,
                gpu_ms, peak_vram_mb, saved_file, summary_json, detections_json, preview_base64
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                source_type,
                model_name,
                int(total_objects),
                int(unique_count),
                float(gpu_ms),
                float(peak_vram_mb),
                saved_file,
                json.dumps(summary, ensure_ascii=False),
                json.dumps(detections[:100], ensure_ascii=False),
                preview_base64,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_sqlite_results(limit: int = 25) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, source_type, model_name, total_objects, unique_count,
                   gpu_ms, peak_vram_mb, saved_file, summary_json
            FROM detection_results
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        items: List[Dict[str, Any]] = []
        for r in rows:
            items.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "source_type": r["source_type"],
                    "model_name": r["model_name"],
                    "total_objects": r["total_objects"],
                    "unique_count": r["unique_count"],
                    "gpu_ms": r["gpu_ms"],
                    "peak_vram_mb": r["peak_vram_mb"],
                    "saved_file": r["saved_file"],
                    "summary": json.loads(r["summary_json"] or "{}"),
                }
            )
        return items
    finally:
        conn.close()


def get_sqlite_result_detail(record_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        r = conn.execute("SELECT * FROM detection_results WHERE id = ?", (record_id,)).fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "created_at": r["created_at"],
            "source_type": r["source_type"],
            "model_name": r["model_name"],
            "total_objects": r["total_objects"],
            "unique_count": r["unique_count"],
            "gpu_ms": r["gpu_ms"],
            "peak_vram_mb": r["peak_vram_mb"],
            "saved_file": r["saved_file"],
            "summary": json.loads(r["summary_json"] or "{}"),
            "detections": json.loads(r["detections_json"] or "[]"),
            "preview_base64": r["preview_base64"],
        }
    finally:
        conn.close()
