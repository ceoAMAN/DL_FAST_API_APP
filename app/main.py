from fastapi import FastAPI, HTTPException
from .schemas import InputData, Prediction
from .predict import inference

app = FastAPI(
    title="Deep Learning Inference API",
    description="A simple FastAPI application for PyTorch model inference",
    version="1.0.0"
)

@app.post("/predict", response_model=Prediction)
async def predict(data: InputData):
    try:
        result = inference.predict(data.features)
        return Prediction(prediction=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))