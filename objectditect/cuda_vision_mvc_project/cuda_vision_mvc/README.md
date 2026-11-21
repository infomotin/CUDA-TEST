# GTX 1650 CUDA Vision MVC Studio

## Folder Structure (MVC Pattern)
- `main.py` — Entrypoint ও FastAPI App/Static mount
- `config/settings.py` — Path, local model cache (`./models`), `downloads`, `data` ও class mappings
- `models/db_model.py` — SQLite (`./data/vision_results.db`) CRUD
- `models/vision_model.py` — CUDA (`GTX 1650`) Object Detection, Color, Classification & Human Age/Gender
- `services/alert_service.py` — Notifications, Alert Rules, Video No-Detection Alert & Webhook
- `services/media_service.py` — Image/Video processing, pip installer & `./downloads` manager
- `controllers/vision_controller.py` — API & View routes
- `templates/index.html` — HTML View
- `static/css/style.css` & `static/js/app.js` — Frontend UI & logic

## Run
```bash
py main.py
```
