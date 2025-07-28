from fastapi import FastAPI, File, UploadFile, Form, Request
from detect_plot import process_plot
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()

# ✅ CORS Middleware Add Karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect-plot")
async def detect_plot_api(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    scale_pixels: float = Form(...),
    scale_feet: float = Form(...),
    request: Request = None,
):
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        result = process_plot(img, (x, y), scale_pixels, scale_feet)

        return JSONResponse(content=result)

    except Exception as e:
        print("🔥 Exception:", str(e))
        if request:
            print("🔥 Request headers:", request.headers)
        return JSONResponse(status_code=500, content={"error": str(e)})
