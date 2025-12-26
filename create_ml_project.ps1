# Create ML Project Structure
# Based on production-ready data science project layout

# Create root project directory
$projectName = "customer_churn_prediction"
New-Item -ItemType Directory -Force -Path $projectName | Out-Null
Set-Location $projectName

# Data directories
New-Item -ItemType Directory -Force -Path "data/raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data/preprocessed" | Out-Null
New-Item -ItemType Directory -Force -Path "data/features" | Out-Null
New-Item -ItemType Directory -Force -Path "data/predictions" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed" | Out-Null

# Notebooks
New-Item -ItemType Directory -Force -Path "notebooks" | Out-Null
New-Item -ItemType File -Force -Path "notebooks/Baseline.ipynb" | Out-Null
New-Item -ItemType File -Force -Path "notebooks/EDA.ipynb" | Out-Null

# Source code with pipelines
New-Item -ItemType Directory -Force -Path "src/pipelines" | Out-Null
New-Item -ItemType File -Force -Path "src/pipelines/__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "src/pipelines/feature_eng_pipeline.py" | Out-Null
New-Item -ItemType File -Force -Path "src/pipelines/inference_pipeline.py" | Out-Null
New-Item -ItemType File -Force -Path "src/pipelines/training_pipeline.py" | Out-Null
New-Item -ItemType File -Force -Path "src/utils.py" | Out-Null

# Tests
New-Item -ItemType Directory -Force -Path "tests" | Out-Null
New-Item -ItemType File -Force -Path "tests/__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "tests/test_training.py" | Out-Null

# Root level configuration files
New-Item -ItemType File -Force -Path "docker-compose.yml" | Out-Null
New-Item -ItemType File -Force -Path "Dockerfile" | Out-Null
New-Item -ItemType File -Force -Path "env.sample" | Out-Null
New-Item -ItemType File -Force -Path "requirements.txt" | Out-Null

Write-Host "ML project structure created successfully!" -ForegroundColor Green
Write-Host "Project name: $projectName" -ForegroundColor Cyan
