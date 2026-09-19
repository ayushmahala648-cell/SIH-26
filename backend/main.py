"""
FastAPI Main Application for परिधि (Paridhi) - Urban Parcel Mapping & Cadastral Feature Extraction.
Serves REST API endpoints and hosts the static GIS web dashboard.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api import router as cadastral_router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="परिधि (Paridhi) — AI Cadastral Parcel Mapping API",
    description="Automated Urban Land Parcel Mapping & Cadastral Feature Extraction from Drone Imagery (SIH 2026 - SIH26012)",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API
app.include_router(cadastral_router)

# Ensure directories exist
for folder in ["images", "outputs", "uploads", "static"]:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# Mount Static Directories for static assets
app.mount("/images", StaticFiles(directory=os.path.join(BASE_DIR, "images")), name="images")
app.mount("/outputs", StaticFiles(directory=os.path.join(BASE_DIR, "outputs")), name="outputs")
app.mount("/uploads", StaticFiles(directory=os.path.join(BASE_DIR, "uploads")), name="uploads")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/")
def serve_index():
    """Serves the main landing and upload page."""
    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/index.html")
def serve_index_html():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/new_analysis.html")
def serve_new_analysis():
    """Serves the interactive cadastral workspace."""
    analysis_path = os.path.join(BASE_DIR, "new_analysis.html")
    return FileResponse(analysis_path)


# Serve root static assets (CSS, JS)
@app.get("/style.css")
def serve_style():
    return FileResponse(os.path.join(BASE_DIR, "style.css"), media_type="text/css")


@app.get("/script.js")
def serve_script():
    return FileResponse(os.path.join(BASE_DIR, "script.js"), media_type="application/javascript")


@app.get("/new_analysis.js")
def serve_analysis_script():
    script_path = os.path.join(BASE_DIR, "new_analysis.js")
    if os.path.exists(script_path):
        return FileResponse(script_path, media_type="application/javascript")
    return {"error": "File not found"}


if __name__ == "__main__":
    print("Starting Paridhi Cadastral Server on http://localhost:8000 ...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Paridhi Cadastral Server on port {port} ...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)

