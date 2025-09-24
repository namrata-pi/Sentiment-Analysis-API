
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment using Logistic Regression",
    version="1.0.0"
)


try:
    model = joblib.load('sentiment1_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None



class PredictionRequest(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "Looking forward to the demo!"
            }
        }


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]

    class Config:
        schema_extra = {
            "example": {
                "label": "positive",
                "confidence": 0.87,
                "probabilities": {
                    "negative": 0.05,
                    "neutral": 0.08,
                    "positive": 0.87
                }
            }
        }



LABEL_MAP = {-1: "negative", 0: "neutral", 1: "positive"}


@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API is running!",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if len(text) > 1000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        predicted_label = LABEL_MAP[prediction]
        confidence = float(np.max(probabilities))

        prob_dict = {
            "negative": float(probabilities[0]),
            "neutral": float(probabilities[1]),
            "positive": float(probabilities[2])
        }

        logger.info(f"Prediction made for text: '{text[:50]}...' -> {predicted_label} ({confidence:.3f})")

        return PredictionResponse(
            label=predicted_label,
            confidence=confidence,
            probabilities=prob_dict
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.post("/predict_batch")
async def predict_batch(texts: list[str]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Too many texts (max 100)")

    results = []
    for text in texts:
        if text.strip():
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]

            results.append({
                "text": text,
                "label": LABEL_MAP[prediction],
                "confidence": float(np.max(probabilities)),
                "probabilities": {
                    "negative": float(probabilities[0]),
                    "neutral": float(probabilities[1]),
                    "positive": float(probabilities[2])
                }
            })

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)