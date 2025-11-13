# P4_V2
Develop a systematic trading strategy using DL models trained on engineered time series features. Multiple neural network architectures are trained to predict trading signals. The project emphasizes practical ML engineering including feature engineering, model tracking with MLFlow, data drift monitoring, backtesting, trading using an API.

NVDA ML Trading Strategy
A deep learning-based systematic trading strategy for NVDA using CNN models, with MLFlow experiment tracking, data drift monitoring, and professional backtesting.
Project Overview
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




