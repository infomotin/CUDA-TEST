import os
import sys

from config.settings import configure_runtime_env

# torch, torchvision এবং deepface ইম্পোর্ট করার আগেই local model cache ও GPU env সেট করা
configure_runtime_env()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import STATIC_DIR
from controllers.vision_controller import initialize_controller, router as vision_router
from models.db_model import init_sqlite_db

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = FastAPI(title="GTX 1650 CUDA Vision MVC Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(vision_router)


@app.on_event("startup")
def on_startup() -> None:
    init_sqlite_db()
    initialize_controller()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
