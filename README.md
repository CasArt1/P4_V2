# P4_V2
Develop a systematic trading strategy using DL models trained on engineered time series features. Multiple neural network architectures are trained to predict trading signals. The project emphasizes practical ML engineering including feature engineering, model tracking with MLFlow, data drift monitoring, backtesting, trading using an API.

NVDA ML Trading Strategy 📈🤖
A deep learning-based systematic trading strategy for NVDA using CNN models, with MLFlow experiment tracking, data drift monitoring, and professional backtesting.
🎯 Project Overview
This project implements a complete ML trading pipeline:

Feature Engineering: 20+ technical indicators (momentum, volatility, volume)
Deep Learning: CNN architectures for signal prediction
MLFlow Tracking: Comprehensive experiment management
Production API: FastAPI endpoint for predictions
Drift Monitoring: Streamlit dashboard for data drift detection
Backtesting: Realistic trading simulation with transaction costs

Trading Signals: Long (1), Hold (0), Short (-1)
Target Definition: 5-day forward returns with ±2% thresholds

📁 Project Structure
project/
├── data/                      # Raw and processed data
│   └── NVDA_raw_data.csv
├── features/                  # Feature engineering scripts
│   └── feature_engineering.py
├── models/                    # Trained models and architectures
│   ├── cnn_simple.py
│   ├── cnn_deep.py
│   └── cnn_custom.py
├── api/                       # FastAPI prediction service
│   └── main.py
├── backtesting/               # Backtesting engine
│   └── backtest.py
├── drift_monitoring/          # Streamlit drift dashboard
│   └── app.py
├── notebooks/                 # Jupyter notebooks for EDA
│   └── eda.ipynb
├── mlruns/                    # MLFlow experiment logs
├── 01_data_collection.py      # Phase 1: Data download script
├── requirements.txt           # Python dependencies
└── README.md                  # This file

🚀 Setup Instructions
1. Create Virtual Environment
bash# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
2. Install Dependencies
bashpip install -r requirements.txt
3. Download Data
bashpython 01_data_collection.py
This will download 15 years of NVDA historical data and save it to data/NVDA_raw_data.csv.

📊 How to Run Each Phase
Phase 1: Data Collection ✅
bashpython 01_data_collection.py
Phase 2: Feature Engineering (Coming Soon)
bashpython features/feature_engineering.py
Phase 3: Model Training (Coming Soon)
bash# Start MLFlow UI (in separate terminal)
mlflow ui

# Train models
python models/train_cnn.py
Phase 4: API Service (Coming Soon)
bash# Start API server
uvicorn api.main:app --reload

# API will be available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
Phase 5: Drift Monitoring Dashboard (Coming Soon)
bashstreamlit run drift_monitoring/app.py
Phase 6: Backtesting (Coming Soon)
bashpython backtesting/backtest.py

🛠️ Technical Stack

Data: yfinance, pandas, numpy
ML: TensorFlow/Keras (or PyTorch)
Experiment Tracking: MLFlow
API: FastAPI, Uvicorn
Monitoring: Streamlit, scipy
Visualization: matplotlib, seaborn, plotly
Technical Indicators: ta, pandas-ta


📈 Trading Strategy Details
Target Variable

Long (1): 5-day forward return > +2%
Hold (0): 5-day forward return between -2% and +2%
Short (-1): 5-day forward return < -2%

Features (20+)
Momentum Indicators:

RSI (14, 21 periods)
MACD
Rate of Change
Stochastic Oscillator

Volatility Indicators:

Bollinger Bands
ATR
Historical Volatility
Standard Deviation

Volume Indicators:

OBV
Volume ROC
VWAP

Model Architecture

CNN Models: 3 different architectures
Class Weighting: Handle imbalanced data
Training: 60% train, 20% validation, 20% test

Backtesting Parameters

Commission: 0.125% per trade
Borrow Rate: 0.25% annualized (for shorts)
Position Sizing: Configurable
Stop Loss & Take Profit: Customizable


📊 Performance Metrics
Evaluation includes:

Sharpe Ratio
Sortino Ratio
Calmar Ratio
Maximum Drawdown
Win Rate
Total Trades
Model Accuracy vs. Strategy Profitability


🎓 Academic Requirements
Code (40%):

Clean, modular design
Comprehensive documentation
Git version control
Professional README

Report (60%):

Strategy overview and methodology
Feature engineering details
Model architecture and MLFlow tracking
Data drift analysis
Backtesting results
Conclusions and recommendations


📝 Development Workflow

Branch naming: feature/phase-X-description
Commits: Clear, descriptive messages
Code style: PEP 8 compliance
Documentation: Docstrings for all functions
Testing: Test each component before moving forward


🔗 API Endpoints (When Ready)
POST /predict
Predict trading signal for given features.
Request Body:
json{
  "features": [0.52, 0.31, -0.15, ...]
}
Response:
json{
  "signal": 1,
  "signal_name": "LONG",
  "confidence": 0.87,
  "timestamp": "2024-11-10T12:00:00"
}

🐛 Troubleshooting
Issue: yfinance download fails
Solution: Check internet connection, try again (API rate limits)
Issue: TensorFlow GPU not detected
Solution: Install tensorflow-gpu and CUDA toolkit
Issue: MLFlow UI won't start
Solution: Check port 5000 isn't in use, try mlflow ui --port 5001

📚 Resources

MLFlow Documentation
FastAPI Documentation
Streamlit Documentation
TensorFlow Tutorials


👨‍💻 Author
Hector Sebastian Castañeda Arteaga
Course: Microstructure and Trading
Date: November 2025

📄 License
This project is for educational purposes.


