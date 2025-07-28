from fastapi import FastAPI, File, UploadFile, Form
from detect_plot import process_plot
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload


@app.post("/detect-plot")
async def detect_plot_api(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    scale_pixels: float = Form(...),     # e.g., 100
    scale_feet: float = Form(...),       # e.g., 10
):
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    result = process_plot(temp_path, (x, y), scale_pixels, scale_feet)
    os.remove(temp_path)

    return JSONResponse(content=result)
