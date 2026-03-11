# Fraud Monitoring System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run-orange)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-purple)

A production-style Machine Learning system that detects fraudulent transactions and monitors model health automatically.

This project demonstrates a full **MLOps pipeline**, including model training, API deployment, monitoring, CI/CD automation, and data drift detection.

---

# Project Overview

This system simulates how real-world machine learning systems operate in production.

The system performs four major tasks:

### 1. Fraud Detection
An **XGBoost machine learning model** predicts whether a transaction is fraudulent.

### 2. API Service
The trained model is exposed through a **FastAPI REST API**.

### 3. Monitoring System
The system continuously monitors incoming data and detects **data drift**.

### 4. Alert System
If drift is detected, alerts and monitoring reports are automatically generated.

---

# System Architecture

Below is the high-level architecture of the system.

```
Training Data
     │
     ▼
Model Training (XGBoost)
     │
     ▼
Saved ML Model
     │
     ▼
FastAPI Application
     │
     ▼
Docker Container
     │
     ▼
Google Cloud Run Deployment
     │
     ▼
API Requests
     │
     ▼
Fraud Predictions
     │
     ▼
Monitoring System
     │
     ▼
Drift Detection (PSI + KL)
     │
     ▼
Alerts + Monitoring Reports
     │
     ▼
Streamlit Monitoring Dashboard
```

---

# Project Structure

```
fraud-monitoring-system/
│
├── app/                     # FastAPI application
│   └── main.py
│
├── training/                # Model training scripts
│
├── model/                   # Trained ML model
│   └── fraud_model.joblib
│
├── monitoring/              # Drift detection scripts
│   ├── drift_report.py
│   ├── generate_evidently_drift.py
│   └── background_monitor.py
│
├── frontend/                # Streamlit monitoring dashboard
│
├── data/                    # Sample datasets
│
├── reports/                 # Generated monitoring reports
│
├── tests/                   # Unit tests
│   ├── test_predict.py
│   └── test_drift.py
│
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── .flake8                  # Linting configuration
├── README.md
│
└── .github/
    └── workflows/
        ├── CI.yml
        ├── deploy-cloudrun.yml
        └── monitor-drift.yml
```

---

# Machine Learning Model

The system uses an **XGBoost classifier** trained to detect fraudulent transactions.

Example features used in the model:

- transaction amount  
- transaction time  
- anonymized financial behavior features  

The trained model is stored as:

```
model/fraud_model.joblib
```

Model output:

```
0 → legitimate transaction
1 → fraudulent transaction
```

---

# API Endpoint

The model is served using **FastAPI**.

Example request:

```
POST /predict
```

Example input:

```json
{
  "amount": 150.75,
  "time": 3600
}
```

Example response:

```json
{
  "prediction": 0
}
```

---

# API Documentation

FastAPI automatically generates interactive API documentation.

After running the server, open:

```
http://localhost:8000/docs
```

This provides an interactive API interface where you can test requests and view the API schema.

---

# Docker Deployment

The API is containerized using Docker.

Build the Docker image:

```
docker build -t fraud-monitoring-system .
```

Run the container locally:

```
docker run -p 8000:8000 fraud-monitoring-system
```

The API will be available at:

```
http://localhost:8000
```

---

# Cloud Deployment

The containerized API is deployed using **Google Cloud Run**.

Deployment is automated using **GitHub Actions CI/CD pipelines**.

Every push to the repository can automatically trigger:

- dependency installation  
- code validation  
- container build  
- deployment to Cloud Run  

---

# Data Drift Monitoring

Machine learning models can degrade when the distribution of incoming data changes.

This system automatically detects **data drift** by comparing:

- baseline dataset (training data distribution)  
- live dataset (recent prediction data)  

Monitoring runs automatically and generates drift reports.

---

# Drift Metrics

Two statistical metrics are used to detect drift.

## Population Stability Index (PSI)

Measures distribution changes between datasets.

| PSI Value | Meaning |
|------|------|
| < 0.1 | No change |
| 0.1 – 0.25 | Moderate shift |
| > 0.25 | Significant drift |

## KL Divergence

Measures how one probability distribution differs from another.

Higher values indicate greater divergence.

---

# Monitoring Output

Each monitoring run generates an **HTML drift report** containing:

- overall drift status  
- dataset comparison  
- feature-level drift metrics  
- statistical thresholds  

Example report fields include:

- baseline rows  
- live rows  
- PSI scores  
- KL divergence  
- drift status per feature  

Reports are stored in:

```
reports/drift_report.html
```

---

# Alert System

When drift exceeds predefined thresholds, the system automatically generates alerts.

Alert information includes:

- drift metric  
- threshold value  
- severity level  
- timestamp  

Alerts are stored in:

```
alerts.json
```

---

# Background Monitoring Job

The monitoring system runs automatically in the background.

A scheduler checks for drift **every 12 hours** and performs:

- drift detection  
- report generation  
- alert creation  

This simulates how real ML systems continuously monitor model performance.

---

# Monitoring Dashboard

The project includes a **Streamlit dashboard** for viewing monitoring results.

The dashboard displays:

- drift status  
- monitoring reports  
- alert details  
- system metrics  

Run the dashboard with:

```
streamlit run frontend/streamlit_monitor.py
```

---

# CI/CD Pipeline

The project uses **GitHub Actions** for automation.

Workflows included:

| Workflow | Purpose |
|--------|--------|
| CI | Run tests and linting |
| Deploy | Deploy API to Cloud Run |
| Monitor | Run drift detection workflow |

---

# Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Core programming language |
| XGBoost | Fraud detection model |
| FastAPI | API service |
| Docker | Containerization |
| Google Cloud Run | Deployment |
| GitHub Actions | CI/CD |
| Streamlit | Monitoring dashboard |
| Pandas | Data processing |
| NumPy | Numerical computation |

---

# How to Run Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the API:

```
uvicorn app.main:app --reload
```

Open the API at:

```
http://localhost:8000
```

Interactive documentation:

```
http://localhost:8000/docs
```

---

# Future Improvements

Possible enhancements include:

- automated model retraining  
- real-time streaming monitoring  
- advanced alerting system  
- experiment tracking  
- model versioning  
- real-time fraud scoring  

---