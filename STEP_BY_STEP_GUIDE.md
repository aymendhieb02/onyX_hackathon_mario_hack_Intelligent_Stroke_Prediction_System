# 📚 Complete Step-by-Step Guide - StrokeCare AI

This guide explains every step of the StrokeCare AI project from setup to deployment.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation Steps](#installation-steps)
5. [Running the Applications](#running-the-applications)
6. [Understanding the Models](#understanding-the-models)
7. [How the Prediction Works](#how-the-prediction-works)
8. [Deployment Guide](#deployment-guide)
9. [Troubleshooting](#troubleshooting)
10. [File Explanations](#file-explanations)

---

## 🎯 Project Overview

**StrokeCare AI** is a machine learning-powered web application that predicts stroke risk based on patient health data. It provides:

- **Binary Prediction**: Whether a patient is at risk (0 = No Risk, 1 = Risk Detected)
- **Probability Prediction**: Risk percentage (0-100%)
- **Risk Factors**: Identified health risk factors
- **Personalized Recommendations**: AI-generated health insights

### Two Applications Available:

1. **Flask App** (`app.py`) - Full-featured web application with beautiful UI
2. **Streamlit App** (`streamlit_app.py`) - Simple, interactive interface

---

## 📁 Project Structure

```
stroke_prediction_app/
├── app.py                          # Flask web application
├── streamlit_app.py                # Streamlit web application
├── model_wrappers.py               # Model wrapper classes (REQUIRED for loading models)
├── stroke_binary_model.pkl         # Binary prediction model (0 or 1)
├── stroke_probability_model.pkl    # Probability prediction model (0-100%)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── templates/
│   ├── index.html                  # Flask frontend template
│   └── deploy.html                 # Deployment instructions page
├── static/
│   ├── style.css                   # CSS stylesheet
│   └── script.js                   # JavaScript for Flask app
├── model_regression.ipynb          # Jupyter notebook (model training)
├── notebookb202aa49c3-1 (2).ipynb # Additional notebook
├── README.md                       # Project documentation
├── DEPLOYMENT.md                   # Deployment instructions
└── STEP_BY_STEP_GUIDE.md          # This file
```

---

## 🔧 Prerequisites

Before starting, ensure you have:

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Model Files** (must be in project root):
   - `stroke_binary_model.pkl`
   - `stroke_probability_model.pkl`

3. **Git** (for version control and deployment)

4. **Internet Connection** (for installing packages and deployment)

---

## 📦 Installation Steps

### Step 1: Clone or Download the Project

If using Git:
```bash
git clone https://github.com/aymendhieb02/onyX_hackathon_mario_hack_Intelligent_Stroke_Prediction_System.git
cd onyX_hackathon_mario_hack_Intelligent_Stroke_Prediction_System
```

Or download the ZIP file and extract it.

### Step 2: Navigate to Project Directory

```bash
cd stroke_prediction_app
# or wherever you extracted the project
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- Streamlit (interactive app framework)
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- requests (API calls)
- joblib (model serialization)

### Step 5: Verify Model Files

Check that both model files exist:
```bash
# Windows
dir stroke_*.pkl

# Mac/Linux
ls stroke_*.pkl
```

You should see:
- `stroke_binary_model.pkl`
- `stroke_probability_model.pkl`

---

## 🚀 Running the Applications

### Option 1: Flask Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5000
   ```

3. **What you'll see:**
   - Beautiful multi-step form
   - Step 1: Personal information
   - Step 2: Medical history
   - Step 3: Lifestyle factors
   - Results with risk assessment

4. **Features:**
   - Animated risk gauge
   - Binary prediction indicator
   - Risk factors list
   - AI-generated insights
   - Downloadable report

### Option 2: Streamlit Application

1. **Start the Streamlit server:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:8501
   ```

3. **What you'll see:**
   - Sidebar with input form
   - Main area showing results
   - BMI calculator
   - Downloadable report

---

## 🧠 Understanding the Models

### Model Architecture

The project uses **two separate models**:

#### 1. Binary Model (`stroke_binary_model.pkl`)
- **Purpose**: Predicts if patient is at risk (0 or 1)
- **Output**: 
  - `0` = No stroke risk detected
  - `1` = Stroke risk detected
- **Use Case**: Quick yes/no assessment

#### 2. Probability Model (`stroke_probability_model.pkl`)
- **Purpose**: Predicts risk percentage
- **Output**: Probability between 0.0 and 1.0 (converted to 0-100%)
- **Use Case**: Detailed risk assessment

### Model Components

Each model contains:
- **Trained ML Model**: The actual prediction algorithm
- **Scaler**: Normalizes input features
- **Preprocessing Function**: Transforms raw data into model-ready format
- **Expected Features**: List of features the model expects

### Model Wrapper Classes

Located in `model_wrappers.py`:

1. **`StrokeBinaryPredictor`**: Wraps binary model
2. **`StrokeProbabilityPredictor`**: Wraps probability model
3. **`complete_preprocessing`**: Preprocesses input data

**Why needed?** The models were saved with these classes, so they must be available when loading.

---

## 🔄 How the Prediction Works

### Step-by-Step Prediction Process

#### 1. **User Input** (Frontend)
```
User fills form with:
- Age: 65
- Gender: Male
- Hypertension: Yes
- Heart Disease: No
- Glucose: 150
- BMI: 28
- Smoking: Former smoker
```

#### 2. **Data Preparation** (`prepare_input_dataframe`)
```python
Converts to DataFrame:
{
    'age': 65,
    'gender': 'Male',
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 150.0,
    'bmi': 28.0,
    'smoking_status': 'formerly smoked'
}
```

#### 3. **Preprocessing** (`complete_preprocessing`)
```python
Creates features:
- age_30_45, age_45_60, age_60_75, age_55_plus, etc.
- age_squared, age_cubed, age_log
- age_hypertension_interaction
- gender_encoded, Residence_type_encoded, etc.
- Removes original categorical columns
```

#### 4. **Scaling** (Using scaler from model)
```python
Normalizes features to same scale
```

#### 5. **Prediction**
```python
Binary Model → 1 (Risk Detected)
Probability Model → 0.45 (45% risk)
```

#### 6. **Post-Processing**
```python
- Convert probability to percentage: 45%
- Determine risk level: MODERATE (30-60%)
- Identify risk factors: Age 60-69, Hypertension, Elevated Glucose
```

#### 7. **Display Results**
- Risk percentage: 45%
- Risk level: MODERATE
- Binary: ⚠️ Risk Detected
- Risk factors: Listed
- Recommendations: Generated

---

## 🌐 Deployment Guide

### Deployment to Streamlit Cloud (Easiest)

#### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### Step 2: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"

#### Step 3: Configure
- **Repository**: Select your repository
- **Branch**: `main`
- **Main file**: `streamlit_app.py`
- Click "Deploy"

#### Step 4: Wait
- Streamlit Cloud will:
  - Clone your repository
  - Install dependencies
  - Load models
  - Start the app

#### Step 5: Access
- Your app will be live at:
  ```
  https://YOUR_APP_NAME.streamlit.app
  ```

### Deployment to Heroku (Flask App)

#### Step 1: Create Procfile
Create file named `Procfile` (no extension):
```
web: gunicorn app:app
```

#### Step 2: Install Gunicorn
```bash
pip install gunicorn
```

#### Step 3: Create runtime.txt
```
python-3.11.0
```

#### Step 4: Deploy
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Deployment to Railway

```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

### Deployment with Docker

#### Create Dockerfile:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
docker build -t stroke-app .
docker run -p 8501:8501 stroke-app
```

---

## 🔍 Troubleshooting

### Problem: Models won't load

**Error**: `Can't get attribute 'StrokeBinaryPredictor'`

**Solution**:
1. Ensure `model_wrappers.py` exists
2. Check imports in `app.py` and `streamlit_app.py`:
   ```python
   from model_wrappers import StrokeBinaryPredictor, StrokeProbabilityPredictor, complete_preprocessing
   ```

### Problem: `complete_preprocessing` not found

**Error**: `Can't get attribute 'complete_preprocessing'`

**Solution**:
- The function is in `model_wrappers.py`
- Make sure it's imported (see above)

### Problem: App hangs on loading

**Causes**:
- Models are too large
- Missing dependencies
- Model files corrupted

**Solutions**:
1. Check model files exist and are not corrupted
2. Verify all dependencies installed
3. Check Streamlit Cloud logs for errors

### Problem: Predictions always return same value

**Causes**:
- Models not loading (using rule-based fallback)
- Input data not properly formatted

**Solutions**:
1. Check console for model loading errors
2. Verify model files are correct
3. Check input data format matches expected format

### Problem: Feature mismatch error

**Error**: `Feature names should match`

**Solution**:
- The `complete_preprocessing` function handles this
- Ensure it's properly imported and used

---

## 📄 File Explanations

### Core Application Files

#### `app.py` (Flask Application)
- **Purpose**: Main Flask web server
- **Key Functions**:
  - `load_models()`: Loads ML models
  - `predict_with_model()`: Makes predictions
  - `prepare_input_dataframe()`: Formats input data
  - `calculate_risk_score()`: Rule-based fallback
  - `get_ai_insights()`: AI-generated recommendations
- **Routes**:
  - `/`: Home page
  - `/predict`: Prediction endpoint
  - `/download-report`: Download assessment report
  - `/deploy`: Deployment instructions
  - `/health`: Health check

#### `streamlit_app.py` (Streamlit Application)
- **Purpose**: Streamlit web interface
- **Key Functions**:
  - `load_models()`: Loads ML models (cached)
  - `predict_stroke_risk()`: Makes predictions
  - `prepare_input_dataframe()`: Formats input data
  - `calculate_risk_score()`: Rule-based fallback
- **Features**:
  - Sidebar input form
  - Real-time results
  - BMI calculator
  - Downloadable reports

#### `model_wrappers.py` (CRITICAL)
- **Purpose**: Contains classes and functions needed to load models
- **Contents**:
  - `StrokeBinaryPredictor`: Wrapper for binary model
  - `StrokeProbabilityPredictor`: Wrapper for probability model
  - `complete_preprocessing()`: Preprocessing function
- **Why Critical**: Models were saved with these classes, so they MUST be available when loading

### Model Files

#### `stroke_binary_model.pkl`
- **Type**: Pickled Python object
- **Contains**: Binary prediction model
- **Size**: ~1-5 MB (typically)
- **Usage**: Loaded at app startup

#### `stroke_probability_model.pkl`
- **Type**: Pickled Python object
- **Contains**: Probability prediction model
- **Size**: ~1-5 MB (typically)
- **Usage**: Loaded at app startup

### Configuration Files

#### `requirements.txt`
- Lists all Python packages needed
- Used by: `pip install -r requirements.txt`

#### `.streamlit/config.toml`
- Streamlit configuration
- Sets theme, server settings
- Only needed for Streamlit app

#### `.gitignore`
- Tells Git which files to ignore
- Excludes: `__pycache__`, `.env`, etc.

### Frontend Files (Flask)

#### `templates/index.html`
- Main HTML template for Flask app
- Contains: Form, results display, UI elements

#### `static/style.css`
- CSS stylesheet
- Defines: Colors, layouts, animations

#### `static/script.js`
- JavaScript for interactivity
- Handles: Form navigation, API calls, result display

### Documentation Files

#### `README.md`
- Project overview
- Quick start guide
- Basic usage instructions

#### `DEPLOYMENT.md`
- Detailed deployment instructions
- Multiple platform guides
- Troubleshooting tips

#### `STEP_BY_STEP_GUIDE.md` (This file)
- Complete step-by-step explanation
- Detailed troubleshooting
- File explanations

---

## 🎓 Understanding the Code Flow

### Flask App Flow

```
1. User visits http://localhost:5000
   ↓
2. Flask serves templates/index.html
   ↓
3. User fills form (3 steps)
   ↓
4. JavaScript collects data
   ↓
5. POST request to /predict
   ↓
6. app.py receives data
   ↓
7. prepare_input_dataframe() formats data
   ↓
8. predict_with_model() calls models
   ↓
9. Models return predictions
   ↓
10. Results formatted and sent back
   ↓
11. JavaScript displays results
```

### Streamlit App Flow

```
1. User visits http://localhost:8501
   ↓
2. Streamlit renders streamlit_app.py
   ↓
3. Models loaded (cached)
   ↓
4. User fills sidebar form
   ↓
5. User clicks "Assess Stroke Risk"
   ↓
6. predict_stroke_risk() called
   ↓
7. Models make predictions
   ↓
8. Results displayed in main area
```

---

## 🔐 Important Notes

### Model Loading

- Models are loaded **once at startup**
- If loading fails, app uses **rule-based prediction**
- Models are **large files** - may take time to load
- Models require **wrapper classes** to load properly

### Data Privacy

- **No data is stored** - all processing is in-memory
- **No database** - everything is temporary
- **Secure** - data is not logged or saved

### Model Accuracy

- Models are trained on **historical data**
- Results are **probabilistic**, not definitive
- Always **consult healthcare professionals** for medical decisions

---

## 📞 Getting Help

### Common Issues

1. **Models not loading?**
   - Check `model_wrappers.py` exists
   - Verify imports are correct
   - Check model files are not corrupted

2. **App won't start?**
   - Check Python version (3.8+)
   - Verify all dependencies installed
   - Check for syntax errors

3. **Predictions seem wrong?**
   - Models might not be loading (check console)
   - Using rule-based fallback
   - Verify input data format

### Debugging Tips

1. **Check Console Output**
   - Flask: Look at terminal where you ran `python app.py`
   - Streamlit: Look at terminal where you ran `streamlit run`

2. **Enable Debug Mode**
   - Flask: Already enabled (`debug=True`)
   - Streamlit: Add `st.write()` statements

3. **Test Models Directly**
   ```python
   import pickle
   from model_wrappers import StrokeBinaryPredictor, StrokeProbabilityPredictor
   
   with open('stroke_binary_model.pkl', 'rb') as f:
       model = pickle.load(f)
   print("Model loaded successfully!")
   ```

---

## ✅ Quick Checklist

Before deploying, ensure:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files present (`stroke_binary_model.pkl`, `stroke_probability_model.pkl`)
- [ ] `model_wrappers.py` exists and has all classes
- [ ] Imports are correct in `app.py` and `streamlit_app.py`
- [ ] Tested locally (both Flask and Streamlit work)
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud configured (if deploying)

---

## 🎉 Success Indicators

You'll know everything is working when:

1. **Flask App**:
   - Server starts without errors
   - Models load successfully (see console)
   - Can fill form and get predictions
   - Results display correctly

2. **Streamlit App**:
   - App loads without hanging
   - Models load (no errors in logs)
   - Can input data and get predictions
   - Results display correctly

3. **Deployed App**:
   - Accessible via URL
   - Loads without errors
   - Predictions work correctly

---

## 📚 Additional Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **scikit-learn Documentation**: https://scikit-learn.org/
- **GitHub Repository**: https://github.com/aymendhieb02/onyX_hackathon_mario_hack_Intelligent_Stroke_Prediction_System

---

## 🏁 Summary

This project provides a complete stroke risk prediction system with:

1. **Two web applications** (Flask and Streamlit)
2. **Dual model system** (binary and probability)
3. **Beautiful UI** with comprehensive results
4. **Easy deployment** to multiple platforms
5. **Complete documentation** for every step

Follow this guide step-by-step, and you'll have a fully functional stroke prediction system!

---

**Built with ❤️ for better health outcomes**

*Last Updated: 2024*

