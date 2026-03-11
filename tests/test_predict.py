from fastapi.testclient import TestClient
from app.main import app, N_FEATURES

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["n_features"] == N_FEATURES


def test_predict_endpoint_valid_input():
    features = [0.0] * N_FEATURES

    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200

    data = response.json()
    assert "fraud_probability" in data
    assert "threshold" in data
    assert "prediction" in data
    assert isinstance(data["fraud_probability"], float)
    assert data["prediction"] in [0, 1]


def test_predict_endpoint_invalid_input_length():
    features = [0.0] * (N_FEATURES - 1)

    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
