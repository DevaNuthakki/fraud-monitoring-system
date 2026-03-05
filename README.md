# Fraud Monitoring System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run-orange)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-purple)

A production-style Machine Learning system that detects fraudulent transactions and monitors data drift automatically.

This project demonstrates a full **MLOps pipeline**, including model training, API deployment, CI/CD automation, and drift monitoring.

---

# Project Overview

This system performs three major tasks:

### 1. Fraud Detection
A machine learning model predicts whether a transaction is fraudulent.

### 2. API Service
The model is exposed through a **FastAPI REST API**.

### 3. Automated Monitoring
A scheduled **GitHub Actions workflow** monitors data drift and generates reports.

---

# System Architecture

Below is the high-level architecture of the system.

```
Training Data
     │
     ▼
Model Training
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
Predictions
     │
     ▼
Monitoring Workflow (GitHub Actions)
     │
     ▼
Drift Detection Report
```

---

# Project Structure

```
fraud-monitoring-system/
│
├── app/                 # FastAPI application
├── model/               # Trained ML model
├── monitoring/          # Drift detection scripts
├── data/                # Sample datasets
├── reports/             # Generated monitoring reports
│
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
├── .gitignore
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

The model is trained to detect fraudulent transactions based on transaction features.

Example features:

- transaction amount
- transaction time
- account behavior patterns

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

FastAPI automatically generates API documentation.

After running the server, open:

```
http://localhost:8000/docs
```

This provides an interactive API interface.

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

---

# Cloud Deployment

The container is deployed using **Google Cloud Run**.

Deployment is automated using **GitHub Actions CI/CD pipelines**.

Every push to the repository can trigger:

- Docker image build
- Container deployment
- Service update

---

# Data Drift Monitoring

The system includes automated monitoring to detect changes in incoming data.

A scheduled **GitHub Actions workflow runs every 6 hours**.

The monitoring script compares:

- baseline dataset (training distribution)
- live dataset (incoming data)

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

The report is automatically uploaded as a **GitHub Actions artifact**.

---

# CI/CD Pipeline

The project uses **GitHub Actions** for automation.

Workflows included:

| Workflow | Purpose |
|--------|--------|
| CI | Install dependencies and validate code |
| Deploy | Deploy API to Cloud Run |
| Monitor | Run drift detection |

---

# Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Core programming language |
| FastAPI | API service |
| Scikit-learn | Machine learning model |
| Docker | Containerization |
| Google Cloud Run | Deployment |
| GitHub Actions | CI/CD |
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
- alerting system for drift detection
- experiment tracking
- model versioning
- real-time fraud scoring

---
