# Fraud Monitoring System

A production-style Machine Learning system that detects fraudulent transactions and monitors data drift automatically.

This project demonstrates a full **MLOps pipeline**, including model training, API deployment, CI/CD, and automated drift monitoring.

---

# Project Overview

This system performs three main tasks:

1. **Fraud Detection**
   - A trained machine learning model predicts whether a transaction is fraudulent.

2. **API Service**
   - A FastAPI application exposes the model as an API endpoint.

3. **Automated Monitoring**
   - A scheduled GitHub Actions workflow monitors data drift and generates reports.

---

# Architecture


Data → Model Training → FastAPI API → Docker → Cloud Run Deployment
↓
GitHub Actions Monitoring
↓
Drift Detection Report


---

# Project Structure


fraud-monitoring-system/
│
├── app/ # FastAPI application
├── model/ # Trained ML model
├── monitoring/ # Drift detection scripts
├── data/ # Sample datasets
├── reports/ # Generated monitoring reports
│
├── Dockerfile # Container configuration
├── requirements.txt # Python dependencies
├── .gitignore
├── README.md
│
└── .github/
└── workflows/
├── CI.yml
├── deploy-cloudrun.yml
└── monitor-drift.yml


---

# Machine Learning Model

The model is trained to detect fraudulent transactions based on transaction features.

Example features:

- transaction amount
- transaction time
- account behavior patterns

The model outputs:


0 → legitimate transaction
1 → fraudulent transaction


---

# API Endpoint

The model is served using **FastAPI**.

Example request:


POST /predict


Example input:

```json
{
  "amount": 150.75,
  "time": 3600
}

Example response:

{
  "prediction": 0
}
Docker Deployment

The API is containerized using Docker.

Build the image:


docker build -t fraud-monitoring-system .


Run locally:


docker run -p 8000:8000 fraud-monitoring-system

Cloud Deployment

The container is deployed to Google Cloud Run.

Deployment is automated using GitHub Actions CI/CD.

Every push to the repository can trigger:

Docker image build

Container deployment

Service update

Data Drift Monitoring

The system includes automated monitoring to detect changes in incoming data.

A scheduled GitHub Actions workflow runs every 6 hours.

The monitoring script compares:

baseline dataset (training distribution)

live dataset (incoming data)

Drift Metrics

Two statistical metrics are used:

Population Stability Index (PSI)

Measures distribution changes between datasets.

Typical thresholds:

PSI Value	Meaning
< 0.1	No change
0.1 - 0.25	Moderate shift
> 0.25	Significant drift
KL Divergence

Measures how one probability distribution differs from another.

Higher values indicate greater divergence.

Monitoring Output

Each monitoring run generates an HTML drift report containing:

overall drift status

dataset comparison

feature-level drift metrics

statistical thresholds

Example report fields:

baseline rows

live rows

PSI scores

KL divergence

drift status per feature

The report is automatically uploaded as a GitHub Actions artifact.

CI/CD Pipeline

The project uses GitHub Actions for automation.

Workflows include:

Workflow	Purpose
CI	Install dependencies and validate code
Deploy	Deploy API to Cloud Run
Monitor	Run drift detection
Technologies Used
Technology	Purpose
Python	Core programming language
FastAPI	API service
Scikit-learn	Machine learning model
Docker	Containerization
Google Cloud Run	Deployment
GitHub Actions	CI/CD
Pandas	Data processing
NumPy	Numerical computation
How to Run Locally

Install dependencies:


pip install -r requirements.txt


Run the API:


uvicorn app.main:app --reload


API will be available at:


http://localhost:8000

Future Improvements

Possible enhancements include:

automated model retraining

real-time streaming monitoring

alerting system for drift detection

experiment tracking

model versioning