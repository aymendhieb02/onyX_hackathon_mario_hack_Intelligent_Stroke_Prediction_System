"""
Stroke Risk Prediction Web Application
A compassionate, human-centered medical screening tool
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
import requests
import json
from datetime import datetime

app = Flask(__name__)

# =============================================================================
# LOAD MODELS
# =============================================================================

# Paths to the exported models
BINARY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stroke_binary_model.pkl')
PROBABILITY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stroke_probability_model.pkl')

# Try to load the exported models
binary_model = None
probability_model = None

try:
    with open(BINARY_MODEL_PATH, 'rb') as f:
        binary_model = pickle.load(f)
    print("✅ Binary model loaded successfully")
except FileNotFoundError:
    print(f"⚠️ Binary model not found at {BINARY_MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error loading binary model: {e}")

try:
    with open(PROBABILITY_MODEL_PATH, 'rb') as f:
        probability_model = pickle.load(f)
    print("✅ Probability model loaded successfully")
except FileNotFoundError:
    print(f"⚠️ Probability model not found at {PROBABILITY_MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error loading probability model: {e}")

if binary_model is None or probability_model is None:
    print("⚠️ Using rule-based prediction as fallback")

# =============================================================================
# OPENROUTER AI CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "tngtech/tng-r1t-chimera:free"

def get_ai_insights(patient_data, risk_level, risk_percentage, risk_factors):
    """Get personalized AI-generated health insights"""
    
    prompt = f"""You are a compassionate healthcare AI assistant. A patient has just received their stroke risk assessment. 
    
Patient Profile:
- Age: {patient_data['age']} years
- Gender: {patient_data['gender']}
- Hypertension: {'Yes' if patient_data['hypertension'] else 'No'}
- Heart Disease: {'Yes' if patient_data['heart_disease'] else 'No'}
- Average Glucose Level: {patient_data['avg_glucose_level']} mg/dL
- BMI: {patient_data['bmi']}
- Smoking Status: {patient_data['smoking_status']}
- Marital Status: {'Married' if patient_data['ever_married'] else 'Not Married'}
- Work Type: {patient_data['work_type']}
- Residence: {patient_data['residence_type']}

Assessment Results:
- Risk Level: {risk_level}
- Risk Score: {risk_percentage}%
- Key Risk Factors: {', '.join(risk_factors) if risk_factors else 'None identified'}

Please provide:
1. A warm, empathetic opening message (2-3 sentences)
2. Personalized health recommendations (3-4 bullet points)
3. Lifestyle modifications specific to their risk factors (2-3 suggestions)
4. An encouraging closing message

Keep the tone supportive, professional, and hopeful. Avoid medical jargon. Focus on actionable advice.
Response should be in a caring, human tone - remember this person may be worried about their health."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Stroke Risk Prediction App"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a caring, knowledgeable healthcare assistant providing personalized health guidance. Be warm, supportive, and focus on empowering patients with actionable advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"OpenRouter API error: {response.status_code}")
            return generate_fallback_insights(patient_data, risk_level, risk_factors)
            
    except Exception as e:
        print(f"AI insights error: {e}")
        return generate_fallback_insights(patient_data, risk_level, risk_factors)
    
    # If we got here without returning, use fallback
    return generate_fallback_insights(patient_data, risk_level, risk_factors)

def generate_fallback_insights(patient_data, risk_level, risk_factors):
    """Generate insights when AI is unavailable"""
    
    insights = []
    
    # Opening
    if risk_level == "LOW":
        insights.append("🌟 Great news! Your stroke risk assessment shows a low risk level. This is encouraging, but maintaining healthy habits is still important.")
    elif risk_level == "MODERATE":
        insights.append("💛 Your assessment shows a moderate risk level. While this isn't cause for alarm, it's an opportunity to make positive changes for your health.")
    else:
        insights.append("🏥 Your assessment indicates an elevated risk level. Please don't be discouraged - understanding your risk is the first step toward better health.")
    
    # Recommendations based on risk factors
    insights.append("\n\n📋 **Personalized Recommendations:**\n")
    
    if patient_data['age'] >= 60:
        insights.append("• Regular health check-ups are especially important at your age. Consider scheduling a comprehensive cardiovascular screening.\n")
    
    if patient_data['hypertension']:
        insights.append("• Monitor your blood pressure daily. Keep a log to share with your doctor.\n")
    
    if patient_data['heart_disease']:
        insights.append("• Stay consistent with any prescribed heart medications. Never skip doses without consulting your doctor.\n")
    
    if patient_data['avg_glucose_level'] > 140:
        insights.append("• Your glucose levels suggest monitoring is needed. Consider consulting an endocrinologist.\n")
    
    if patient_data['bmi'] > 30:
        insights.append("• Gradual weight management through balanced nutrition can significantly reduce your risk.\n")
    
    if patient_data['smoking_status'] in ['smokes', 'formerly smoked']:
        insights.append("• If you smoke, quitting is the single most impactful change you can make. Resources are available to help.\n")
    
    # Lifestyle suggestions
    insights.append("\n🌱 **Lifestyle Tips:**\n")
    insights.append("• Aim for 30 minutes of moderate activity most days - even walking counts!\n")
    insights.append("• Reduce sodium intake and increase fruits, vegetables, and whole grains.\n")
    insights.append("• Manage stress through meditation, deep breathing, or activities you enjoy.\n")
    
    # Closing
    insights.append("\n💪 Remember: Your health journey is unique. Small, consistent steps lead to meaningful improvements. You have the power to positively influence your health outcomes.")
    
    return "".join(insights)

# =============================================================================
# RISK CALCULATION
# =============================================================================

def calculate_risk_score(patient_data):
    """
    Calculate stroke risk using smart prediction with threshold adjustment
    Returns: risk_percentage, risk_level, risk_factors, explanation
    """
    
    risk_factors = []
    explanation = []
    base_score = 0
    
    # Age factor (most significant)
    age = patient_data['age']
    if age >= 70:
        base_score += 35
        risk_factors.append("Age 70+")
        explanation.append(f"Age ({age} years) is a significant factor - stroke risk increases substantially after 70.")
    elif age >= 60:
        base_score += 25
        risk_factors.append("Age 60-69")
        explanation.append(f"Age ({age} years) contributes to elevated risk - cardiovascular vigilance recommended.")
    elif age >= 50:
        base_score += 15
        risk_factors.append("Age 50-59")
        explanation.append(f"Age ({age} years) is entering a period where regular screening becomes important.")
    elif age >= 40:
        base_score += 5
    
    # Hypertension (major factor)
    if patient_data['hypertension']:
        base_score += 20
        risk_factors.append("Hypertension")
        explanation.append("Hypertension (high blood pressure) significantly increases stroke risk by damaging blood vessels over time.")
    
    # Heart disease (major factor)
    if patient_data['heart_disease']:
        base_score += 20
        risk_factors.append("Heart Disease")
        explanation.append("Heart disease is closely linked to stroke risk through shared cardiovascular mechanisms.")
    
    # Glucose level
    glucose = patient_data['avg_glucose_level']
    if glucose >= 200:
        base_score += 15
        risk_factors.append("Very High Glucose")
        explanation.append(f"Glucose level ({glucose} mg/dL) indicates potential diabetes, which damages blood vessels.")
    elif glucose >= 140:
        base_score += 10
        risk_factors.append("Elevated Glucose")
        explanation.append(f"Glucose level ({glucose} mg/dL) is elevated - monitoring recommended.")
    elif glucose >= 100:
        base_score += 5
    
    # BMI
    bmi = patient_data['bmi']
    if bmi >= 35:
        base_score += 10
        risk_factors.append("Severe Obesity")
        explanation.append(f"BMI ({bmi}) indicates severe obesity, which strains the cardiovascular system.")
    elif bmi >= 30:
        base_score += 7
        risk_factors.append("Obesity")
        explanation.append(f"BMI ({bmi}) indicates obesity, a modifiable risk factor for stroke.")
    elif bmi >= 25:
        base_score += 3
    
    # Smoking
    smoking = patient_data['smoking_status']
    if smoking == 'smokes':
        base_score += 15
        risk_factors.append("Current Smoker")
        explanation.append("Smoking damages blood vessels and significantly increases stroke risk. Quitting has immediate benefits.")
    elif smoking == 'formerly smoked':
        base_score += 5
        risk_factors.append("Former Smoker")
        explanation.append("Former smoking history contributes slightly to risk, but quitting was a positive step.")
    
    # Combination factors (synergistic effects)
    if age >= 60 and patient_data['hypertension']:
        base_score += 10
        if "Age + Hypertension Combination" not in risk_factors:
            explanation.append("The combination of advanced age and hypertension creates compounded risk.")
    
    if patient_data['hypertension'] and patient_data['heart_disease']:
        base_score += 10
        explanation.append("Having both hypertension and heart disease significantly elevates cardiovascular risk.")
    
    # Cap the score at 95%
    risk_percentage = min(95, max(5, base_score))
    
    # Determine risk level with smart thresholds
    if risk_percentage >= 60:
        risk_level = "HIGH"
    elif risk_percentage >= 30:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    return risk_percentage, risk_level, risk_factors, explanation

def predict_with_model(patient_data):
    """Use ML models for prediction - both binary and probability"""
    
    # Always get rule-based score for risk factors and explanation
    rule_percentage, rule_level, risk_factors, explanation = calculate_risk_score(patient_data)
    
    # If models are not loaded, use rule-based
    if binary_model is None or probability_model is None:
        return rule_percentage, rule_level, risk_factors, explanation
    
    # Prepare input DataFrame for the models
    # The models expect raw input and handle preprocessing internally
    try:
        input_df = prepare_input_dataframe(patient_data)
        
        # Get binary prediction (0 or 1)
        binary_prediction = binary_model.predict(input_df)[0]
        
        # Get probability prediction (0.0 to 1.0)
        risk_probability = probability_model.predict(input_df)[0]
        risk_percentage = int(risk_probability * 100)
        
        # Ensure percentage is in valid range
        risk_percentage = max(0, min(100, risk_percentage))
        
        # If binary prediction is 1 (stroke predicted), ensure minimum risk
        if binary_prediction == 1:
            risk_percentage = max(risk_percentage, 30)  # At least moderate if stroke predicted
        
        # Determine risk level
        if risk_percentage >= 60:
            risk_level = "HIGH"
        elif risk_percentage >= 30:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return risk_percentage, risk_level, risk_factors, explanation
        
    except Exception as e:
        print(f"ML prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to rule-based
        return rule_percentage, rule_level, risk_factors, explanation

def prepare_input_dataframe(patient_data):
    """Prepare patient data as DataFrame for the exported models"""
    # The exported models expect raw input data and handle preprocessing internally
    # Create a DataFrame with the same column names as the training data
    
    # Convert to DataFrame with proper column names
    input_data = pd.DataFrame([{
        'age': float(patient_data.get('age', 0)),
        'gender': patient_data.get('gender', 'Male'),
        'hypertension': int(patient_data.get('hypertension', 0)),
        'heart_disease': int(patient_data.get('heart_disease', 0)),
        'ever_married': patient_data.get('ever_married', 'No'),
        'work_type': patient_data.get('work_type', 'Private'),
        'Residence_type': patient_data.get('residence_type', 'Urban'),
        'avg_glucose_level': float(patient_data.get('avg_glucose_level', 100)),
        'bmi': float(patient_data.get('bmi', 25)),
        'smoking_status': patient_data.get('smoking_status', 'never smoked')
    }])
    
    return input_data

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    
    try:
        data = request.json
        print(f"📥 Received data: {data}")  # Debug log
        
        # Parse patient data - handle both boolean and integer values
        patient_data = {
            'age': float(data.get('age', 0)),
            'gender': data.get('gender', 'Male'),
            'hypertension': bool(data.get('hypertension', 0)),  # Convert 0/1 to bool
            'heart_disease': bool(data.get('heart_disease', 0)),  # Convert 0/1 to bool
            'ever_married': data.get('ever_married', 'No'),
            'work_type': data.get('work_type', 'Private'),
            'residence_type': data.get('residence_type', 'Urban'),
            'avg_glucose_level': float(data.get('avg_glucose_level', 100)),
            'bmi': float(data.get('bmi', 25)),
            'smoking_status': data.get('smoking_status', 'never smoked')
        }
        
        print(f"📊 Parsed patient data: {patient_data}")  # Debug log
        
        # Calculate risk
        risk_percentage, risk_level, risk_factors, explanation = predict_with_model(patient_data)
        
        print(f"🎯 Risk: {risk_percentage}% - {risk_level}")  # Debug log
        print(f"⚠️ Factors: {risk_factors}")  # Debug log
        
        # Get AI insights
        ai_insights = get_ai_insights(patient_data, risk_level, risk_percentage, risk_factors)
        
        # Get binary prediction if model is available
        binary_prediction = None
        if binary_model is not None:
            try:
                input_df = prepare_input_dataframe(patient_data)
                binary_prediction = int(binary_model.predict(input_df)[0])
            except:
                pass
        
        return jsonify({
            'success': True,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'explanation': explanation,
            'ai_insights': ai_insights,
            'binary_prediction': binary_prediction,  # 0 or 1
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'binary_model_loaded': binary_model is not None,
        'probability_model_loaded': probability_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/download-report', methods=['POST'])
def download_report():
    """Generate and download assessment report"""
    try:
        data = request.json
        patient_data = data.get('patient_data', {})
        result = data.get('result', {})
        
        # Generate report text
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           STROKE RISK ASSESSMENT REPORT                       ║
╚══════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Age: {patient_data.get('age', 'N/A')} years
Gender: {patient_data.get('gender', 'N/A')}
Residence Type: {patient_data.get('residence_type', 'N/A')}
Marital Status: {patient_data.get('ever_married', 'N/A')}
Work Type: {patient_data.get('work_type', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDICAL HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hypertension: {'Yes' if patient_data.get('hypertension') else 'No'}
Heart Disease: {'Yes' if patient_data.get('heart_disease') else 'No'}
Average Glucose Level: {patient_data.get('avg_glucose_level', 'N/A')} mg/dL
BMI (Body Mass Index): {patient_data.get('bmi', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIFESTYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Smoking Status: {patient_data.get('smoking_status', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSESSMENT RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk Percentage: {result.get('risk_percentage', 'N/A')}%
Risk Level: {result.get('risk_level', 'N/A')}
Binary Prediction: {'Risk Detected' if result.get('binary_prediction') == 1 else 'No Risk Detected' if result.get('binary_prediction') == 0 else 'N/A'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTIFIED RISK FACTORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join('• ' + factor for factor in result.get('risk_factors', [])) if result.get('risk_factors') else 'No significant risk factors identified'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPLANATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join('• ' + exp for exp in result.get('explanation', [])) if result.get('explanation') else 'No additional explanations'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI-GENERATED INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{result.get('ai_insights', 'No insights available')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ IMPORTANT DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This assessment is for informational purposes only and does not 
constitute medical advice. Please consult a healthcare professional 
for proper diagnosis and treatment.

This report was generated by StrokeCare AI - A Compassionate 
Healthcare Tool powered by Advanced Machine Learning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        # Create response with report
        from flask import Response
        response = Response(
            report,
            mimetype='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename=stroke_risk_assessment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            }
        )
        return response
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/deploy')
def deploy_page():
    """Deployment instructions page"""
    return render_template('deploy.html')

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🏥 STROKE RISK PREDICTION SYSTEM                          ║
    ║   A Compassionate Healthcare Tool                            ║
    ║                                                              ║
    ║   Starting server at http://localhost:5000                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=5000)
