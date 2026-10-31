
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import predict_image

app = FastAPI(
    title="Deepfake AI Detection API",
    description="Backend phat hien Deepfake bang ConvNeXt-Tiny V2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {
        "message": "Deepfake AI Detection API",
        "status": "running"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model": "ConvNeXt-Tiny V2"
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Chi ho tro JPG, PNG va WEBP"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="File anh rong"
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Anh vuot qua 10 MB"
        )

    try:
        result = predict_image(image_bytes)
        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
