
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Spam Detection API is running"
    }


def test_predict_spam_message():
    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You have won a free prize"
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"


def test_predict_empty_message():
    response = client.post(
        "/predict",
        json={"message": ""}
    )

    assert response.status_code == 422


def test_predict_missing_message():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_predict_invalid_message_type():
    response = client.post(
        "/predict",
        json={"message": 123}
    )

    assert response.status_code == 422
