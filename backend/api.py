"""
REST API Routes for Cadastral Feature Extraction & Mapping.
"""

import os
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ParcelUpdateRequest,
    AnalysisSummary
)
from backend.analysis_manager import AnalysisManager

router = APIRouter(prefix="/api", tags=["cadastral"])
manager = AnalysisManager()


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Paridhi Cadastral Mapping API",
        "version": "1.0.0"
    }


@router.get("/sample-images")
def get_sample_images():
    """Returns metadata for all bundled sample drone aerial photos."""
    samples = manager.list_sample_images("images")
    return {"samples": samples}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Uploads a high-resolution drone image for cadastral mapping."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: .jpg, .jpeg, .png, .tif, .tiff"
        )

    file_id = str(uuid.uuid4())[:8]
    clean_name = f"{file_id}_{os.path.basename(file.filename)}"
    save_path = os.path.join(manager.uploads_dir, clean_name)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "filename": clean_name,
        "image_path": save_path,
        "url": f"/uploads/{clean_name}"
    }


@router.post("/analyze")
def analyze_cadastral(req: AnalysisRequest):
    """
    Executes the deep learning & geospatial analysis pipeline on a specified image.
    Supports either image_name from images/ or image_path from uploads/.
    """
    target_path = None
    if req.image_path and os.path.exists(req.image_path):
        target_path = req.image_path
    elif req.image_name:
        candidate = os.path.join("images", req.image_name)
        if os.path.exists(candidate):
            target_path = candidate
        else:
            # Check uploads
            candidate_up = os.path.join(manager.uploads_dir, req.image_name)
            if os.path.exists(candidate_up):
                target_path = candidate_up

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(
            status_code=404,
            detail=f"Target image could not be found. Provided: name='{req.image_name}', path='{req.image_path}'"
        )

    try:
        result = manager.run_pipeline(
            image_path=target_path,
            gsd_meters=req.gsd_meters,
            simplify_tolerance=req.simplify_tolerance,
            min_parcel_area_px=req.min_parcel_area_px,
            snap_orthogonal=req.snap_orthogonal
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")


@router.get("/results/{analysis_id}")
def get_analysis_results(analysis_id: str):
    """Retrieves full analysis results, parcel registers, and GeoJSON vectors."""
    result = manager.get_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")
    return result


@router.post("/parcels/update")
def update_parcel_verification(req: ParcelUpdateRequest):
    """
    Updates a parcel's verification status (Verified, Disputed, Pending) or Land-Use classification
    supporting human-in-the-loop validation.
    """
    updated = manager.update_parcel(
        analysis_id=req.analysis_id,
        parcel_id=req.parcel_id,
        status=req.verification_status,
        land_use=req.land_use
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Parcel '{req.parcel_id}' or Analysis not found.")
    return {"success": True, "updated_parcel": updated}


@router.get("/export/{analysis_id}/{format}")
def export_file(analysis_id: str, format: str):
    """
    Downloads analysis output in requested format:
    - 'geojson': RFC 7946 GeoJSON FeatureCollection
    - 'csv': Tabular Cadastral Land Register
    - 'overlay': Annotated aerial overlay image
    """
    subfolder = os.path.join(manager.outputs_dir, analysis_id)
    if not os.path.exists(subfolder):
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")

    fmt = format.lower()
    if fmt == "geojson":
        file_path = os.path.join(subfolder, "cadastral_features.geojson")
        filename = f"cadastral_parcels_{analysis_id}.geojson"
        media_type = "application/geo+json"
    elif fmt == "csv":
        file_path = os.path.join(subfolder, "cadastral_register.csv")
        filename = f"cadastral_register_{analysis_id}.csv"
        media_type = "text/csv"
    elif fmt in ["overlay", "jpg", "jpeg", "image"]:
        file_path = os.path.join(subfolder, "cadastral_overlay.jpg")
        filename = f"cadastral_overlay_{analysis_id}.jpg"
        media_type = "image/jpeg"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{format}'. Allowed: geojson, csv, overlay")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested export file was not generated.")

    return FileResponse(file_path, media_type=media_type, filename=filename)

