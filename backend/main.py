import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="SIH Image Upload Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


IMAGES_DIR = FRONTEND_DIR / "images"
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}



@app.get("/")
def home():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok", "message": "Backend server is running"}


@app.get("/new_analysis.html")
def new_analysis_page():
    page = FRONTEND_DIR / "new_analysis.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/style.css")
def get_style():
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/script.js")
def get_script():
    js_file = FRONTEND_DIR / "script.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")



@app.post("/api/upload")
async def upload_photo(file: UploadFile = File(...)):
    
    try:
        
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{file_ext}'. Only {', '.join(ALLOWED_EXTENSIONS)} are allowed."
            )

        
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename

        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "message": "Image uploaded and stored successfully",
            "filename": unique_filename,
            "original_name": file.filename,
            "image_url": f"/uploads/{unique_filename}"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
