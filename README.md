# ❤️ StrokeCare AI - Stroke Risk Prediction System

A comprehensive, AI-powered web application for stroke risk assessment with both Flask and Streamlit interfaces.

## 🌟 Features

- **Dual Model System**: 
  - Binary prediction (0/1) for stroke risk detection
  - Probability prediction (0-100%) for risk percentage
  
- **Beautiful UI**: 
  - Modern, responsive design
  - Interactive forms with step-by-step guidance
  - Real-time risk visualization with animated gauges
  
- **Comprehensive Assessment**:
  - Personal information (age, gender, residence, etc.)
  - Medical history (hypertension, heart disease, glucose, BMI)
  - Lifestyle factors (smoking status)
  
- **AI-Powered Insights**: 
  - Personalized health recommendations
  - Risk factor identification
  - Actionable lifestyle modifications

## 📁 Project Structure

```
stroke_prediction_app/
├── app.py                 # Flask application
├── streamlit_app.py       # Streamlit application
├── requirements.txt       # Python dependencies
├── stroke_binary_model.pkl      # Binary prediction model
├── stroke_probability_model.pkl # Probability prediction model
├── templates/
│   └── index.html         # Flask frontend template
├── static/
│   ├── style.css         # Stylesheet
│   └── script.js         # Frontend JavaScript
├── DEPLOYMENT.md          # Deployment guide
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Model files: `stroke_binary_model.pkl` and `stroke_probability_model.pkl`

### Installation

1. **Clone or download the project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**
   - `stroke_binary_model.pkl`
   - `stroke_probability_model.pkl`

### Running the Applications

#### Flask App

```bash
python app.py
```

Visit `http://localhost:5000`

#### Streamlit App

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501`

## 🎯 Usage

### Flask Version

1. Fill in the multi-step form:
   - **Step 1**: Personal information (age, gender, residence, marital status, work type)
   - **Step 2**: Medical history (hypertension, heart disease, glucose level, BMI)
   - **Step 3**: Lifestyle (smoking status)

2. Click "Get My Risk Assessment"

3. View results:
   - Risk percentage gauge
   - Risk level (LOW/MODERATE/HIGH)
   - Binary prediction (Risk Detected/No Risk)
   - Identified risk factors
   - AI-generated personalized insights

### Streamlit Version

1. Fill in patient information in the sidebar
2. Click "Assess Stroke Risk"
3. View comprehensive results with:
   - Risk metrics
   - Risk factors
   - Detailed recommendations
   - Downloadable report

## 🔧 Model Details

The application uses two exported models:

1. **Binary Model** (`stroke_binary_model.pkl`):
   - Returns 0 (no stroke risk) or 1 (stroke risk detected)
   - Uses optimized threshold for clinical balance

2. **Probability Model** (`stroke_probability_model.pkl`):
   - Returns risk probability (0.0 to 1.0)
   - Converted to percentage (0-100%) for display

Both models include:
- Complete preprocessing pipeline
- Feature engineering (age groups, interactions, etc.)
- Categorical encoding
- Scaling and normalization

## 📊 Features Used

The models analyze:
- **Age**: Multiple age group features (30-45, 45-60, 60-75, 55+, 65+, 75+, 80+)
- **Age transformations**: Squared, cubed, logarithmic
- **Age interactions**: With hypertension, heart disease, glucose
- **Clinical factors**: Hypertension, heart disease
- **Biometric**: Glucose level, BMI
- **Categorical**: Gender, marital status, work type, residence, smoking status
- **Risk scores**: Age-dominated risk, composite risk indicators

## 🌐 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:

- Streamlit Cloud
- Heroku
- Railway
- Docker
- AWS/GCP/Azure

## 🔒 Security & Privacy

- No data is stored
- All processing is done in-memory
- Models are loaded once at startup
- HTTPS recommended for production

## ⚠️ Disclaimer

**This application is for informational purposes only and does not constitute medical advice.**

- Not a substitute for professional medical consultation
- Results should be interpreted by healthcare professionals
- Always consult a doctor for proper diagnosis and treatment

## 🛠️ Development

### Adding New Features

1. **Update Models**: Retrain models in `model_regression.ipynb`
2. **Export Models**: Use the export section to generate new `.pkl` files
3. **Update Preprocessing**: Ensure `prepare_input_dataframe()` matches model expectations
4. **Test Locally**: Test both Flask and Streamlit versions

### Dependencies

Key dependencies:
- Flask: Web framework
- Streamlit: Interactive app framework
- scikit-learn: Machine learning
- pandas: Data manipulation
- numpy: Numerical computing

See `requirements.txt` for complete list.

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please:
1. Test thoroughly
2. Update documentation
3. Follow code style
4. Add comments for complex logic

## 📞 Support

For issues:
1. Check [DEPLOYMENT.md](DEPLOYMENT.md) troubleshooting section
2. Verify model files are present
3. Check application logs
4. Test locally first

## 🎉 Acknowledgments

- Built with advanced machine learning techniques
- Uses clinical domain knowledge for feature engineering
- Implements best practices for imbalanced data handling
- Optimized for both accuracy and clinical utility

---

**Built with ❤️ for better health outcomes**

StrokeCare AI © 2024 • Powered by Advanced Machine Learning

