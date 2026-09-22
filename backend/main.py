"""
Main FastAPI Application for परिधि (Paridhi) - Cadastral AI Mapping System.
Combines YOLOv8 Pretrained Building Segmentation & Spectral GIS Terrain Extraction
with Full REST API, Cadastral Land Registry, and Interactive GIS Web Portal.
"""

import os
import io
import csv
import json
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from backend.cadastral_engine import engine, extract_cadastral_features

# Base directory resolution
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "Frontend" if (PROJECT_ROOT / "Frontend").exists() else PROJECT_ROOT
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
IMAGES_DIR = FRONTEND_DIR / "images"

for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Dense Urban & Rural Cadastral Mapping API",
    description="Automated AI Cadastral Parcel & Feature Mapping from Drone Aerial Imagery",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset folders
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# In-memory analysis cache
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}


# =========================================================================
# 1. DIRECT CADASRAL ENDPOINTS (Friend's API Specifications Enhanced)
# =========================================================================

@app.post("/segment/cadastral/visualize")
async def visualize_cadastral(
    file: UploadFile = File(...),
    alpha: float = Query(0.42, ge=0.1, le=0.9, description="Color fill transparency (0.1 to 0.9)"),
    draw_labels: bool = Query(True, description="Draw cadastral labels on features")
):
    """
    Draws and colors houses, trees, big land (farming, playground, vacant land), and roads
    directly over the uploaded drone image with translucent colored fills and crisp boundary vectors.
    """
    contents = await file.read()
    try:
        img, features = extract_cadastral_features(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Render translucent filled colors and sharp vector perimeters
    annotated = engine.render_colored_overlay(img, features, alpha=alpha, draw_labels=draw_labels)

    success, encoded_img = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode output image.")

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/segment/cadastral/wireframe")
async def wireframe_cadastral(file: UploadFile = File(...)):
    """Generates a clean black blueprint background with only the vectors and color fills visible."""
    contents = await file.read()
    try:
        img, features = extract_cadastral_features(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    wireframe = engine.render_wireframe(img, features)

    success, encoded_img = cv2.imencode(".jpg", wireframe, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode wireframe image.")

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/segment/cadastral/colored")
async def colored_mask_cadastral(file: UploadFile = File(...)):
    """Generates a categorical solid color segmentation map."""
    contents = await file.read()
    try:
        img, features = extract_cadastral_features(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    colored = engine.render_colored_mask(img, features)

    success, encoded_img = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode mask image.")

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/segment/cadastral/geojson")
async def get_cadastral_json(
    file: UploadFile = File(...),
    gsd_meters: float = Query(0.08, ge=0.01, le=2.0, description="Ground Sample Distance in meters/pixel")
):
    """Returns raw GeoJSON-ready coordinate data and cadastral attributes."""
    contents = await file.read()
    try:
        img, features = extract_cadastral_features(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    h, w = img.shape[:2]
    geojson_data = engine.to_geojson((w, h), features, gsd_meters=gsd_meters)
    return JSONResponse(content=geojson_data)


# =========================================================================
# 2. REST API FOR WEB DASHBOARD & UPLOAD FLOW
# =========================================================================

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

class AnalysisRequest(BaseModel):
    image_name: Optional[str] = None
    image_path: Optional[str] = None
    gsd_meters: float = 0.08
    min_building_area: int = 70
    min_field_area: int = 2500


class ParcelUpdateRequest(BaseModel):
    analysis_id: str
    parcel_id: str
    verification_status: Optional[str] = None
    land_use: Optional[str] = None


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Dense Urban & Rural Cadastral Mapping API",
        "model": "YOLOv8 Pretrained Building Segmentation + Morphological GIS Terrain",
        "classes": [
            "Building (Crimson)",
            "Tree / Forest (Emerald Green)",
            "Agricultural Land (Amber)",
            "Playground / Open Turf (Lime)",
            "Vacant / Open Land (Sand Tan)",
            "Road / Pathway (Yellow)"
        ]
    }


@app.get("/api/sample-images")
def list_sample_images():
    """Lists bundled drone images available in images folder."""
    samples = []
    if IMAGES_DIR.exists():
        for f in sorted(IMAGES_DIR.iterdir()):
            if f.suffix.lower() in ALLOWED_EXTENSIONS or ".jpg" in f.name.lower():
                samples.append({
                    "filename": f.name,
                    "url": f"/images/{f.name}",
                    "size_bytes": f.stat().st_size
                })
    return {"samples": samples}


@app.post("/api/upload")
async def upload_photo(file: UploadFile = File(...)):
    """Handles drone image upload from the home landing page."""
    try:
        file_ext = Path(file.filename or "").suffix.lower()
        if not file_ext or file_ext not in ALLOWED_EXTENSIONS:
            # Check if filename has double extension like .jpg.jpeg
            clean_ext = ".jpg"
            for ext in ALLOWED_EXTENSIONS:
                if ext in (file.filename or "").lower():
                    clean_ext = ext
                    break
            file_ext = clean_ext

        unique_filename = f"{uuid.uuid4().hex[:12]}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "message": "Image uploaded successfully",
            "filename": unique_filename,
            "original_name": file.filename,
            "image_url": f"/uploads/{unique_filename}",
            "file_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/api/analyze")
def run_analysis(req: AnalysisRequest):
    """
    Executes the full YOLO & GIS analysis pipeline on uploaded image or sample image.
    Generates annotated overlay, RFC 7946 GeoJSON, CSV land register, and summary metrics.
    """
    t0 = time.time()
    target_path = None

    # Resolve target image
    if req.image_path:
        cand = Path(req.image_path)
        if cand.exists():
            target_path = cand
        elif (UPLOAD_DIR / cand.name).exists():
            target_path = UPLOAD_DIR / cand.name
        elif (PROJECT_ROOT / req.image_path.lstrip("/\\")).exists():
            target_path = PROJECT_ROOT / req.image_path.lstrip("/\\")

    if not target_path and req.image_name:
        cand_upload = UPLOAD_DIR / req.image_name
        cand_sample = IMAGES_DIR / req.image_name
        if cand_upload.exists():
            target_path = cand_upload
        elif cand_sample.exists():
            target_path = cand_sample

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Target image not found.")

    img = cv2.imread(str(target_path))
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read or decode image.")

    h_orig, w_orig = img.shape[:2]

    # Extract features
    features = engine.extract_features(
        img,
        min_building_area=req.min_building_area,
        min_field_area=req.min_field_area
    )

    analysis_id = str(uuid.uuid4())[:8]
    out_dir = OUTPUT_DIR / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Colored Visual Overlay
    overlay = engine.render_colored_overlay(img, features, alpha=0.45, draw_labels=True)
    overlay_filename = "cadastral_overlay.jpg"
    overlay_path = out_dir / overlay_filename
    cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # 2. Save Wireframe Blueprint
    wireframe = engine.render_wireframe(img, features)
    wireframe_filename = "cadastral_wireframe.jpg"
    cv2.imwrite(str(out_dir / wireframe_filename), wireframe, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # 3. GeoJSON
    geojson_data = engine.to_geojson((w_orig, h_orig), features, gsd_meters=req.gsd_meters)
    geojson_path = out_dir / "cadastral_features.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    # 4. CSV Cadastral Land Register
    csv_path = out_dir / "cadastral_register.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Parcel ID", "Feature Type", "Land Use", "Category", "Area (sq m)",
            "Area (hectares)", "Area (acres)", "Area (gaj)", "Perimeter (m)",
            "Vertex Count", "Confidence", "Status"
        ])
        for feat in geojson_data["features"]:
            p = feat["properties"]
            writer.writerow([
                p["parcel_id"], p["feature_type"], p.get("land_use", "Vacant"), p["category"], p["area_sqm"],
                p["area_hectares"], p["area_acres"], p["area_gaj"], p["perimeter_m"],
                p["vertex_count"], f"{int(p['confidence'] * 100)}%", p["verification_status"]
            ])

    meta = geojson_data["metadata"]
    duration = round(time.time() - t0, 2)

    # Summary
    summary = {
        "analysis_id": analysis_id,
        "image_name": target_path.name,
        "image_dims": [w_orig, h_orig],
        "total_features": meta["total_features"],
        "total_buildings": meta["counts"].get("Building", 0),
        "total_trees": meta["counts"].get("Tree / Forest", 0),
        "total_farmlands": meta["counts"].get("Agricultural Land", 0),
        "total_playgrounds": meta["counts"].get("Playground / Open Turf", 0),
        "total_roads": meta["counts"].get("Road / Pathway", 0),
        "total_vacant": meta["counts"].get("Vacant / Open Land", 0),
        "total_surveyed_area_sqm": meta["total_surveyed_sqm"],
        "total_surveyed_hectares": round(meta["total_surveyed_sqm"] / 10000.0, 4),
        "total_surveyed_acres": round(meta["total_surveyed_sqm"] / 4046.86, 4),
        "total_surveyed_gaj": round(meta["total_surveyed_sqm"] * 1.196, 2),
        "total_builtup_sqm": meta["total_builtup_sqm"],
        "builtup_pct": meta["builtup_pct"],
        "gsd_meters": req.gsd_meters,
        "processing_time_sec": duration
    }

    parcels_list = [f["properties"] for f in geojson_data["features"]]

    result_payload = {
        "success": True,
        "summary": summary,
        "overlay_url": f"/outputs/{analysis_id}/{overlay_filename}",
        "wireframe_url": f"/outputs/{analysis_id}/{wireframe_filename}",
        "geojson_url": f"/outputs/{analysis_id}/cadastral_features.geojson",
        "csv_url": f"/outputs/{analysis_id}/cadastral_register.csv",
        "parcels": parcels_list,
        "geojson_data": geojson_data
    }

    # Save metadata
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    ANALYSIS_CACHE[analysis_id] = result_payload
    return result_payload


@app.get("/api/results/{analysis_id}")
def get_analysis_results(analysis_id: str):
    """Retrieves cached or persisted analysis results."""
    if analysis_id in ANALYSIS_CACHE:
        return ANALYSIS_CACHE[analysis_id]

    meta_file = OUTPUT_DIR / analysis_id / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            ANALYSIS_CACHE[analysis_id] = data
            return data

    raise HTTPException(status_code=404, detail="Analysis ID not found.")


@app.post("/api/parcels/update")
def update_parcel_verification(req: ParcelUpdateRequest):
    """Updates verification status (Verified, Pending Review, Disputed) or land use."""
    meta_file = OUTPUT_DIR / req.analysis_id / "metadata.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = None
    for p in data.get("parcels", []):
        if p.get("parcel_id") == req.parcel_id:
            if req.verification_status:
                p["verification_status"] = req.verification_status
            if req.land_use:
                p["land_use"] = req.land_use
            updated = p
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Parcel not found.")

    # Update in geojson_data
    if "geojson_data" in data and "features" in data["geojson_data"]:
        for feat in data["geojson_data"]["features"]:
            if feat.get("id") == req.parcel_id:
                if req.verification_status:
                    feat["properties"]["verification_status"] = req.verification_status
                if req.land_use:
                    feat["properties"]["land_use"] = req.land_use

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    ANALYSIS_CACHE[req.analysis_id] = data
    return {"success": True, "updated_parcel": updated}


@app.get("/api/export/{analysis_id}/{format}")
def export_cadastral_file(analysis_id: str, format: str):
    """Downloads GeoJSON, CSV, or Overlay image."""
    subfolder = OUTPUT_DIR / analysis_id
    if not subfolder.exists():
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    fmt = format.lower()
    if fmt == "geojson":
        return FileResponse(
            subfolder / "cadastral_features.geojson",
            media_type="application/geo+json",
            filename=f"cadastral_{analysis_id}.geojson"
        )
    elif fmt == "csv":
        return FileResponse(
            subfolder / "cadastral_register.csv",
            media_type="text/csv",
            filename=f"cadastral_register_{analysis_id}.csv"
        )
    elif fmt in ["overlay", "jpg", "jpeg", "image"]:
        return FileResponse(
            subfolder / "cadastral_overlay.jpg",
            media_type="image/jpeg",
            filename=f"cadastral_overlay_{analysis_id}.jpg"
        )
    elif fmt in ["wireframe", "blueprint"]:
        return FileResponse(
            subfolder / "cadastral_wireframe.jpg",
            media_type="image/jpeg",
            filename=f"cadastral_wireframe_{analysis_id}.jpg"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Allowed: geojson, csv, overlay, wireframe")


# =========================================================================
# 3. STATIC HTML & ASSET SERVING
# =========================================================================

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Dense Urban & Rural Cadastral Mapping API is running"}


@app.get("/index.html")
def serve_index_html():
    return serve_index()


@app.get("/new_analysis.html")
def serve_new_analysis():
    page = FRONTEND_DIR / "new_analysis.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="new_analysis.html not found.")


@app.get("/style.css")
def serve_css():
    css = FRONTEND_DIR / "style.css"
    if css.exists():
        return FileResponse(css, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found.")


@app.get("/script.js")
def serve_js():
    js = FRONTEND_DIR / "script.js"
    if js.exists():
        return FileResponse(js, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found.")


@app.get("/new_analysis.js")
def serve_new_analysis_js():
    js = FRONTEND_DIR / "new_analysis.js"
    if js.exists():
        return FileResponse(js, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="new_analysis.js not found.")


if __name__ == "__main__":
    import uvicorn
    print("Starting Dense Urban & Rural Cadastral Mapping Server on http://localhost:8000 ...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
