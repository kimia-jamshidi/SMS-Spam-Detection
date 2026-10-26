import joblib
from pathlib import Path
from text_preprocessing import preprocess_message

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(MODEL_DIR / "spam_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

message = "Congratulations! You have won a free prize"
message_final = preprocess_message(message)
print("Processed message:", message_final)

message_vector = vectorizer.transform([message_final])

prediction = model.predict(message_vector)
label = encoder.inverse_transform(prediction)

print("Prediction:", label[0])