# filename: api.py

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from main import extract_doc_metadata

app = FastAPI(
    title="Medical Document Metadata API",
    description="Upload a prescription or lab report (PDF/JPG/PNG) and extract metadata fields.",
    version="1.0.0",
)


@app.post("/extract")
async def extract_metadata(file: UploadFile = File(...)):
    """Endpoint: Upload a file and get metadata JSON."""
    try:
        # Create a temporary file path safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        # Call your main extractor
        metadata = extract_doc_metadata(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        return JSONResponse(content=metadata.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
