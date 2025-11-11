# 🌐 NVDA Trading Signal API

FastAPI service for serving CNN-based trading signal predictions.

## 📋 Overview

This API serves the best-performing CNN model (CustomCNN) from Phase 3, providing real-time trading signal predictions for NVDA stock.

**Signals:**
- `-1` = **SHORT** (Sell/Short position)
- `0` = **HOLD** (No action)
- `1` = **LONG** (Buy/Long position)

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
# From project root directory
python api/main.py
```

The server will start on `http://localhost:8000`

### 2. Access API Documentation

Open your browser and go to:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### GET `/health`
Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/saved_models/CustomCNN.h5",
  "timestamp": "2024-11-11T12:30:00"
}
```

---

### POST `/predict`
**Main prediction endpoint** - Predicts trading signal from a sequence of features.

**Input:** 10 timesteps, each with 28 normalized features

**Request Body:**
```json
{
  "sequences": [
    [0.5, 0.3, -0.2, 0.1, ...],  // Timestep 1 (28 features)
    [0.4, 0.2, -0.1, 0.2, ...],  // Timestep 2
    [0.3, 0.1, 0.0, 0.3, ...],   // Timestep 3
    // ... 10 timesteps total
  ]
}
```

**Response:**
```json
{
  "signal": 1,
  "signal_name": "LONG",
  "confidence": 0.68,
  "probabilities": {
    "SHORT": 0.15,
    "HOLD": 0.17,
    "LONG": 0.68
  },
  "timestamp": "2024-11-11T12:30:00",
  "model_name": "CustomCNN"
}
```

---

### POST `/predict/single`
Simplified endpoint for single feature vector (repeats it to create a sequence).

**Input:** 28 normalized features

**Request Body:**
```json
{
  "features": [0.5, 0.3, -0.2, 0.1, ...]  // 28 features
}
```

**Response:** Same as `/predict`

**Note:** For better accuracy, use `/predict` with proper temporal sequences.

---

### GET `/model/info`
Get information about the loaded model.

**Response:**
```json
{
  "model_name": "CustomCNN",
  "model_path": "models/saved_models/CustomCNN.h5",
  "input_shape": "(None, 10, 28)",
  "output_shape": "(None, 3)",
  "num_parameters": 245123,
  "num_features": 28,
  "sequence_length": 10,
  "output_classes": {
    "0": "SHORT (-1)",
    "1": "HOLD (0)",
    "2": "LONG (1)"
  }
}
```

---

## 💻 Usage Examples

### Python (requests library)

```python
import requests
import json

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
payload = {
    "sequences": [
        [0.5] * 28,  # Timestep 1
        [0.4] * 28,  # Timestep 2
        # ... 10 timesteps total
    ]
}

response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)

result = response.json()
print(f"Signal: {result['signal_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Prediction (simplified example)
curl -X POST "http://localhost:8000/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.3, -0.2, 0.1, 0.0, 0.2, -0.1, 0.4, 
                 0.3, 0.1, 0.2, -0.3, 0.5, 0.0, 0.1, -0.2,
                 0.4, 0.2, 0.1, -0.1, 0.3, 0.0, 0.2, 0.1,
                 -0.2, 0.3, 0.4, 0.1]
  }'
```

### JavaScript (fetch)

```javascript
// Make prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sequences: Array(10).fill(Array(28).fill(0.5))
  })
})
.then(response => response.json())
.then(data => {
  console.log('Signal:', data.signal_name);
  console.log('Confidence:', data.confidence);
});
```

---

## 🧪 Testing the API

### Run the Test Suite

```bash
# Make sure API is running first!
python test_api.py
```

This will test:
- ✅ Health check
- ✅ Model info
- ✅ Predictions with real data
- ✅ Predictions with dummy data
- ✅ Error handling

### Manual Testing with Swagger UI

1. Start the API: `python api/main.py`
2. Open http://localhost:8000/docs
3. Click "Try it out" on any endpoint
4. Modify the request body
5. Click "Execute"

---

## 📊 Understanding the Response

### Signal Values
| Value | Name | Action |
|-------|------|--------|
| -1 | SHORT | Sell/short the stock |
| 0 | HOLD | Do nothing |
| 1 | LONG | Buy/long the stock |

### Confidence Score
- Range: 0.0 to 1.0
- Higher = More confident in prediction
- **Important**: High confidence doesn't guarantee correctness!

### Probabilities
Shows the model's probability distribution across all three classes:
```json
{
  "SHORT": 0.15,  // 15% chance
  "HOLD": 0.17,   // 17% chance
  "LONG": 0.68    // 68% chance (highest = prediction)
}
```

---

## ⚙️ Configuration

### Change Port

Edit `api/main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change to 8080
```

### Use Different Model

Edit `api/main.py`:
```python
MODEL_PATH = "models/saved_models/SimpleCNN.h5"  # Use SimpleCNN instead
```

---

## 🐛 Troubleshooting

### Error: "Model not loaded"
**Problem:** Model file not found

**Solution:**
1. Make sure you ran Phase 3: `python 03_train_models_standalone.py`
2. Check if `models/saved_models/CustomCNN.h5` exists
3. Verify MODEL_PATH in `api/main.py`

### Error: "Expected 28 features, got X"
**Problem:** Wrong number of input features

**Solution:** 
- Ensure you're sending exactly 28 normalized features per timestep
- Check that features are properly normalized (z-score)

### Error: "Connection refused"
**Problem:** API server not running

**Solution:** Start the server with `python api/main.py`

---

## 📁 File Structure

```
api/
├── main.py              # FastAPI application
├── __init__.py          # Package init (optional)
└── README.md            # This file

test_api.py              # Test client
models/saved_models/     # Trained models
data/                    # Feature data (for testing)
```

---

## 🚀 Production Deployment

### Using Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📈 Performance Notes

- **Inference time**: ~10-50ms per prediction (CPU)
- **Throughput**: ~20-100 requests/second (single worker)
- **Memory**: ~500MB (model + API)

For higher throughput:
- Use multiple workers: `uvicorn api.main:app --workers 4`
- Deploy on GPU-enabled server
- Implement request batching

---

## 🔒 Security Considerations

⚠️ **This is a demo API** - For production use:

1. **Add authentication** (API keys, OAuth)
2. **Implement rate limiting**
3. **Add input validation** (more strict)
4. **Use HTTPS**
5. **Monitor and log** requests
6. **Sanitize error messages**

---

## 📝 API Specifications

- **Framework**: FastAPI 0.110+
- **Server**: Uvicorn
- **ML Framework**: TensorFlow/Keras
- **Model**: CustomCNN (Multi-scale CNN)
- **Input**: 10 timesteps × 28 features
- **Output**: 3 classes (Short/Hold/Long)

---

## ✅ Next Steps

After Phase 4:
1. **Phase 5**: Data drift monitoring dashboard
2. **Phase 6**: Backtesting with this API
3. **Phase 7**: Final report and documentation

---

## 🆘 Support

Issues? Check:
1. Server logs in terminal
2. Swagger docs at `/docs`
3. Test with `test_api.py`
4. Review Phase 3 model training results

**Happy Trading! 📈🤖**
