# Deep Learning Inference API

A production-grade FastAPI application for serving PyTorch model predictions.

## Project Structure

```
dl_fastapi_app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── schemas.py
│   └── utils.py
├── model/
│   └── model.pth
├── training_example.py
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python training_example.py
```

3. Start the FastAPI application:
```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

## API Usage

### Predict Endpoint

**Endpoint:** `POST /predict`

**Request Format:**
```json
{
  "features": [0.1, 0.2, 0.3, 0.4]
}
```

**Response Format:**
```json
{
  "prediction": 2
}
```

### Example Request

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, 0.3, 0.4]}'
```

Using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [0.1, 0.2, 0.3, 0.4]}
)
print(response.json())
```

## Documentation

Once the server is running, view the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc