from fastapi import FastAPI, File, UploadFile, Form
from detect_plot import process_plot
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI()

@app.post("/detect-plot")
async def detect_plot_api(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    scale_pixels: float = Form(...),
    scale_feet: float = Form(...),
):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    result = process_plot(img, (x, y), scale_pixels, scale_feet)

    return JSONResponse(content=result)
