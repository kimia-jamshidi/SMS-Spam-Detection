# SMS Spam Classification

A machine learning project that classifies SMS messages as **spam** or **ham (normal)** using Natural Language Processing (NLP) techniques and machine learning models. The trained model is integrated with a **FastAPI REST API** to support real-time message classification.

## Dataset

The dataset contains SMS messages with two labels:

- `ham` → normal messages
- `spam` → unwanted messages

Labels were encoded as:

- `ham` → `0`
- `spam` → `1`

## NLP Pipeline

The text data was processed using the following steps:

- Lowercase conversion
- Removing punctuation
- Tokenization
- Stopword removal
- Lemmatization

The final cleaned text was used for feature extraction. The same preprocessing function is reused when the API receives a new message, ensuring consistency between training and inference.

## Feature Extraction

Text was converted into numerical features using the **Bag of Words** approach with `CountVectorizer`.

According to the original model development results:

- Messages: 5,567
- Features: 8,910

## Models

Two classification models were evaluated:

- Multinomial Naive Bayes
- Logistic Regression

## Results

The original model evaluation reported the following accuracy scores:

| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes | 97.48% |
| Logistic Regression | 98.47% |

The original evaluation also reported:

- Ham detection F1-score: **0.99**
- Spam detection F1-score: **0.94**

These results describe the original model evaluation. The model currently saved for API inference is the **Multinomial Naive Bayes** model.

## FastAPI Integration

The trained model is integrated into a REST API built with FastAPI.

### API Features

- Loads the trained model, vectorizer, and label encoder from saved `.joblib` files.
- Accepts SMS messages through a JSON request.
- Applies the same text preprocessing pipeline used during training.
- Converts the processed message into numerical features using the saved `CountVectorizer`.
- Predicts whether the message is `spam` or `ham`.
- Validates incoming requests using Pydantic.
- Returns predictions as JSON responses.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Checks whether the API is running |
| `POST` | `/predict` | Classifies an SMS message |

### Example Request

Send a `POST` request to `/predict` with the following JSON body:

```json
{
  "message": "Congratulations! You have won a free prize"
}
```

### Example Response

```json
{
  "message": "Congratulations! You have won a free prize",
  "prediction": "spam"
}
```

### Input Validation

The API validates incoming requests using Pydantic. Invalid requests, such as an empty message, a missing `message` field, or an invalid data type, return HTTP `422 Unprocessable Entity`.

### Run the API Locally

1. Clone or download the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Ensure the saved model artifacts are available in the `models/` directory.
5. Start the API from the `src/` directory:

```bash
python -m uvicorn main:app --reload
```

The API will be available at:

- API root: `http://127.0.0.1:8000/`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Use the Swagger UI to send test requests to the `/predict` endpoint.

## Testing

Automated API tests are implemented with `pytest` and FastAPI's `TestClient`.

The current test suite covers:

- Root endpoint response
- Successful spam prediction
- Empty message validation
- Missing message validation
- Invalid message type validation

Run the tests from the `src/` directory:

```bash
python -m pytest -v
```

**Current test result:**