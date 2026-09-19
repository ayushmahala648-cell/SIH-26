"""
Pydantic schemas for Cadastral API requests, responses, and verification workflows.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AnalysisRequest(BaseModel):
    image_name: Optional[str] = Field(None, description="Name of sample image in images/ directory")
    image_path: Optional[str] = Field(None, description="Full or relative path to uploaded image")
    gsd_meters: float = Field(0.08, ge=0.01, le=2.0, description="Ground Sample Distance in meters per pixel")
    simplify_tolerance: float = Field(2.0, ge=0.5, le=10.0, description="Douglas-Peucker simplification tolerance")
    min_parcel_area_px: int = Field(100, ge=20, description="Minimum pixel area to consider as a parcel")
    snap_orthogonal: bool = Field(True, description="Snap rectangular buildings to orthogonal edges")


class ParcelUpdateRequest(BaseModel):
    analysis_id: str
    parcel_id: str
    verification_status: Optional[str] = Field(None, description="'Verified', 'Pending Review', or 'Disputed'")
    land_use: Optional[str] = Field(None, description="Updated land use classification")
    notes: Optional[str] = None


class ParcelProperty(BaseModel):
    parcel_id: str
    numeric_id: int
    area_sqm: float
    area_hectares: float
    area_acres: float
    area_gaj: float
    perimeter_m: float
    compactness: float
    builtup_area_sqm: float
    builtup_pct: float
    building_count: int
    land_use: str
    confidence: float
    verification_status: str
    centroid_px: List[float]
    centroid_geo: List[float]


class AnalysisSummary(BaseModel):
    analysis_id: str
    image_name: str
    image_dims: List[int]
    total_parcels: int
    total_buildings: int
    total_roads: int
    total_surveyed_area_sqm: float
    total_surveyed_hectares: float
    total_builtup_sqm: float
    overall_builtup_pct: float
    gsd_meters: float
    processing_time_sec: float
    created_at: str


class AnalysisResponse(BaseModel):
    success: bool
    summary: AnalysisSummary
    overlay_url: str
    geojson_url: str
    csv_url: str
    parcels: List[ParcelProperty]
    geojson_pixel: Dict[str, Any]
    geojson_wgs84: Dict[str, Any]

