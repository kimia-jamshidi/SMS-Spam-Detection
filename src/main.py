
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from text_preprocessing import preprocess_message


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"


# 2. Load trained model and related objects
model = joblib.load(MODEL_DIR / "spam_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")


# 3. Create the FastAPI application
app = FastAPI(
    title="Spam Detection API",
    version="1.0.0"
)


# 4. Define the request schema
class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="SMS message to classify"
    )


# 5. Define the response schema
class PredictionResponse(BaseModel):
    message: str
    prediction: str


# 6. Health check endpoint
@app.get("/")
def root():
    return {"message": "Spam Detection API is running"}



@app.post("/predict", response_model=PredictionResponse)
def predict_spam(request: MessageRequest):
    # 1. Preprocess the input message
    message_final = preprocess_message(request.message)

    # 2. Convert the message into numerical features
    message_vector = vectorizer.transform([message_final])

    # 3. Predict the encoded label
    prediction_encoded = model.predict(message_vector)

    # 4. Convert the encoded label back to its original text
    prediction_label = encoder.inverse_transform(prediction_encoded)[0]

    # 5. Return the API response
    return {
        "message": request.message,
        "prediction": prediction_label
    }
