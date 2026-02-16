# Workflow-CI: MLflow CI/CD Pipeline

**Machine Learning System - Kriteria 3**
**Author:** Amirullah
**Project:** House Prices Prediction with Automated Retraining

---

## 📋 Project Overview

This repository implements a complete **CI/CD pipeline** for machine learning model training using **MLflow Project** and **GitHub Actions**. The system automatically retrains the House Prices prediction model whenever code changes are pushed to the repository.

**Dataset:** [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 🎯 Features

- ✅ **MLflow Project** structure for reproducible ML workflows
- ✅ **GitHub Actions** for automated training on push/PR
- ✅ **Automated artifact storage** (models, plots, metrics)
- ✅ **Docker image** build and push to Docker Hub
- ✅ **Model versioning** with MLflow Model Registry
- ✅ **Comprehensive logging** and metrics tracking

---

## 📁 Repository Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLProject                   # MLflow project configuration
│   ├── conda.yaml                  # Environment dependencies
│   ├── modelling.py                # Training script
│   ├── dataset_preprocessing/      # Preprocessed datasets
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   ├── artifacts/                  # Generated artifacts (plots, metrics)
│   └── models/                     # Trained model files
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- MLflow 2.11.3
- Docker (for Docker image build)
- GitHub account
- Docker Hub account (for ADVANCE level)

### Local Development

1. **Clone the repository:**
   ```bash
   git clone git@github.com:amirullahh/Workflow-CI.git
   cd Workflow-CI/MLProject
   ```

2. **Install dependencies:**
   ```bash
   pip install -r conda.yaml  # Install from conda.yaml pip section
   ```

3. **Run training locally:**
   ```bash
   mlflow run . -P n_estimators=200 -P max_depth=15 -P random_state=42
   ```

   Or directly:
   ```bash
   python modelling.py 200 15 42
   ```

4. **View results:**
   ```bash
   mlflow ui
   # Open http://localhost:5000
   ```

---

## 🔄 CI/CD Workflow

### Automated Retraining Triggers

The GitHub Actions workflow automatically runs when:

1. **Push to main branch** with changes in:
   - `MLProject/**`
   - `.github/workflows/**`

2. **Pull Request** to main branch

3. **Manual trigger** via GitHub Actions UI (workflow_dispatch)

### Workflow Steps

1. **Setup Environment**
   - Install Python 3.12
   - Install MLflow and dependencies

2. **Train Model**
   - Run `modelling.py` with MLflow tracking
   - Generate artifacts (plots, metrics)

3. **Save Artifacts**
   - Upload to GitHub Actions artifacts
   - Commit model files to repository
   - Create backup archives

4. **Build Docker Image** (ADVANCE)
   - Build Docker image with `mlflow models build-docker`
   - Push to Docker Hub

5. **Generate Report**
   - Create comprehensive training summary
   - Display metrics and artifacts

---

## 🐳 Docker Deployment

### Build Docker Image Locally

```bash
cd MLProject
mlflow models build-docker \
  --model-uri mlruns/<experiment_id>/<run_id>/artifacts/model \
  --name your-dockerhub-username/house-prices-model:latest
```

### Push to Docker Hub

```bash
docker push your-dockerhub-username/house-prices-model:latest
```

### Run Model Container

```bash
docker run -p 5000:8080 your-dockerhub-username/house-prices-model:latest
```

### Make Predictions

```bash
curl -X POST http://localhost:5000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

---

## ⚙️ Configuration

### GitHub Secrets (Required for Docker)

Add these secrets in your GitHub repository settings:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

**How to add:**
1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`

### MLflow Parameters

Configurable in `MLProject` file:

- `n_estimators`: Number of trees (default: 200)
- `max_depth`: Maximum tree depth (default: 15)
- `random_state`: Random seed (default: 42)

---

## 📊 Artifacts Generated

Each training run produces:

1. **Model File**
   - `models/model.pkl` - Trained RandomForest model

2. **Visualizations**
   - `feature_importance.png` - Top 20 important features
   - `actual_vs_predicted.png` - Prediction accuracy plot

3. **Metrics**
   - `metrics_summary.json` - Complete performance metrics
   - MLflow tracking logs

4. **MLflow Model**
   - Logged to MLflow Model Registry
   - Includes model signature and environment

---

## 📈 Model Performance

**Latest Model Metrics:**

| Metric | Training | Test |
|--------|----------|------|
| **RMSE** | ~0.06 | ~0.13 |
| **MAE** | ~0.04 | ~0.09 |
| **R²** | ~0.97 | ~0.88 |
| **MAPE** | - | ~0.76% |

*Note: Actual values depend on training run*

---

## 🔧 Troubleshooting

### Workflow Fails with "No space left on device"

**Solution:** GitHub Actions runners have limited disk space. The workflow includes cleanup steps.

### Docker build fails

**Solution:**
1. Ensure GitHub Secrets are configured
2. Check Docker Hub credentials
3. Verify MLflow model is properly logged

### Model artifacts not uploaded

**Solution:**
1. Check that training completed successfully
2. Verify `models/` and `artifacts/` directories exist
3. Check GitHub Actions logs for errors

---

## 📝 Submission Information

**Kriteria 3 - CI Workflow with MLflow Project**

**Level:** ADVANCE (4 points)

**Requirements Met:**
- ✅ MLflow Project structure with `MLProject` and `conda.yaml`
- ✅ GitHub Actions workflow for auto-retrain
- ✅ Artifacts saved to GitHub Actions & repository
- ✅ Docker image build with `mlflow models build-docker`
- ✅ Docker push to Docker Hub

---

## 🤝 Contributing

This is an educational project for Dicoding Machine Learning System submission.

---

## 📄 License

This project is for educational purposes (Dicoding submission).

---

## 👤 Author

**Amirullah**
- GitHub: [@amirullahh](https://github.com/amirullahh)
- Dicoding: Membangun Sistem Machine Learning
- Date: February 2026

---

## 🔗 Links

- **GitHub Repository:** https://github.com/amirullahh/Workflow-CI
- **Preprocessing Repository:** https://github.com/amirullahh/Eksperimen_SML_Amirullah
- **DagsHub (Model Training):** https://dagshub.com/amirullahh/MSML-Amirullah
- **Docker Hub:** https://hub.docker.com/r/[your-username]/house-prices-model

---

**🎉 Generated with MLflow CI/CD Pipeline**
