/**
 * Stroke Care AI - Interactive Assessment System
 * Human-centered approach to stroke risk prediction
 */

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const AppState = {
    currentStep: 1,
    totalSteps: 3,
    formData: {},
    isSubmitting: false,
    lastResult: null,  // Store last prediction result
    lastPatientData: null  // Store last patient data
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const Elements = {
    form: document.getElementById('assessmentForm'),
    progressFill: document.getElementById('progressFill'),
    stepIndicators: document.querySelectorAll('.step'),
    formSteps: document.querySelectorAll('.form-step'),
    nextBtn: document.getElementById('nextBtn'),
    backBtn: document.getElementById('backBtn'),
    submitBtn: document.getElementById('submitBtn'),
    resultsSection: document.getElementById('resultsSection'),
    loadingState: document.getElementById('loadingState'),
    resultsContent: document.getElementById('resultsContent'),
    formSection: document.getElementById('formSection'),
    // BMI Modal
    bmiModal: document.getElementById('bmiModal'),
    bmiCalculatorBtn: document.getElementById('bmiCalculatorBtn'),
    bmiCloseBtn: document.getElementById('bmiCloseBtn'),
    calculateBmiBtn: document.getElementById('calculateBmiBtn'),
    bmiResult: document.getElementById('bmiResult'),
    // Results elements
    gaugePercentage: document.getElementById('gaugePercentage'),
    riskBadge: document.getElementById('riskBadge'),
    riskFactorsList: document.getElementById('riskFactorsList'),
    aiInsightsContent: document.getElementById('aiInsightsContent'),
    startOverBtn: document.getElementById('startOverBtn')
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    updateProgressBar();
    updateNavigationButtons();
    initializeFormDefaults();
}

function initializeFormDefaults() {
    // Set default values
    const genderMale = document.getElementById('genderMale');
    if (genderMale) genderMale.checked = true;
    
    // Initialize summary if needed
    updateSummaryPreview();
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Form submission - CRITICAL: prevent default form submit
    const form = document.getElementById('assessmentForm');
    if (form) {
        form.addEventListener('submit', handleSubmit);
    }
    
    // Navigation buttons
    Elements.nextBtn?.addEventListener('click', handleNext);
    Elements.backBtn?.addEventListener('click', handleBack);
    Elements.submitBtn?.addEventListener('click', handleSubmit);
    Elements.startOverBtn?.addEventListener('click', startOver);
    
    // BMI Calculator
    Elements.bmiCalculatorBtn?.addEventListener('click', openBmiModal);
    Elements.bmiCloseBtn?.addEventListener('click', closeBmiModal);
    Elements.calculateBmiBtn?.addEventListener('click', calculateBmi);
    
    // Close modal on outside click
    Elements.bmiModal?.addEventListener('click', (e) => {
        if (e.target === Elements.bmiModal) closeBmiModal();
    });
    
    // Form change listeners for summary
    const formInputs = Elements.form?.querySelectorAll('input, select');
    formInputs?.forEach(input => {
        input.addEventListener('change', updateSummaryPreview);
    });
    
    // Age input highlight
    const ageInput = document.getElementById('age');
    ageInput?.addEventListener('input', handleAgeInput);
    
    // BMI input highlight
    const bmiInput = document.getElementById('bmi');
    bmiInput?.addEventListener('input', handleBmiInput);
    
    // Glucose input highlight
    const glucoseInput = document.getElementById('glucose');
    glucoseInput?.addEventListener('input', handleGlucoseInput);
}

// ============================================================================
// NAVIGATION
// ============================================================================

// Global functions called by onclick handlers in HTML
function nextStep(stepNum) {
    if (!validateCurrentStep()) return;
    AppState.currentStep = stepNum;
    updateUI();
}

function prevStep(stepNum) {
    AppState.currentStep = stepNum;
    updateUI();
}

function handleNext() {
    if (!validateCurrentStep()) return;
    
    if (AppState.currentStep < AppState.totalSteps) {
        AppState.currentStep++;
        updateUI();
    }
}

function handleBack() {
    if (AppState.currentStep > 1) {
        AppState.currentStep--;
        updateUI();
    }
}

function updateUI() {
    updateFormSteps();
    updateProgressBar();
    updateNavigationButtons();
    updateStepIndicators();
    
    // Scroll to top of form
    Elements.formSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateFormSteps() {
    // Update form steps by ID
    for (let i = 1; i <= AppState.totalSteps; i++) {
        const stepEl = document.getElementById(`step${i}`);
        if (stepEl) {
            stepEl.classList.toggle('active', i === AppState.currentStep);
        }
    }
}

function updateProgressBar() {
    const progress = (AppState.currentStep / AppState.totalSteps) * 100;
    if (Elements.progressFill) {
        Elements.progressFill.style.width = `${progress}%`;
    }
}

function updateNavigationButtons() {
    // Back button
    if (Elements.backBtn) {
        Elements.backBtn.style.display = AppState.currentStep > 1 ? 'flex' : 'none';
    }
    
    // Next button
    if (Elements.nextBtn) {
        Elements.nextBtn.style.display = AppState.currentStep < AppState.totalSteps ? 'flex' : 'none';
    }
    
    // Submit button
    if (Elements.submitBtn) {
        Elements.submitBtn.style.display = AppState.currentStep === AppState.totalSteps ? 'flex' : 'none';
    }
}

function updateStepIndicators() {
    Elements.stepIndicators.forEach((indicator, index) => {
        indicator.classList.remove('active', 'completed');
        
        if (index + 1 < AppState.currentStep) {
            indicator.classList.add('completed');
        } else if (index + 1 === AppState.currentStep) {
            indicator.classList.add('active');
        }
    });
}

// ============================================================================
// VALIDATION
// ============================================================================

function validateCurrentStep() {
    const currentStepEl = document.getElementById(`step${AppState.currentStep}`);
    if (!currentStepEl) return true;
    
    const requiredInputs = currentStepEl.querySelectorAll('[required]');
    let isValid = true;
    
    requiredInputs.forEach(input => {
        if (!input.value || input.value.trim() === '') {
            isValid = false;
            highlightError(input);
        } else {
            clearError(input);
        }
    });
    
    // Special validation for radio groups
    if (AppState.currentStep === 1) {
        const genderSelected = document.querySelector('input[name="gender"]:checked');
        if (!genderSelected) {
            isValid = false;
            showNotification('Please select your gender', 'warning');
        }
    }
    
    if (AppState.currentStep === 3) {
        const smokingSelected = document.querySelector('input[name="smoking"]:checked');
        if (!smokingSelected) {
            isValid = false;
            showNotification('Please select your smoking status', 'warning');
        }
    }
    
    // Custom validations
    if (AppState.currentStep === 1) {
        const age = parseInt(document.getElementById('age')?.value);
        if (age && (age < 0 || age > 120)) {
            isValid = false;
            showNotification('Please enter a valid age (0-120)', 'warning');
        }
    }
    
    if (AppState.currentStep === 2) {
        const glucose = parseFloat(document.getElementById('glucose')?.value);
        if (glucose && (glucose < 50 || glucose > 500)) {
            isValid = false;
            showNotification('Please enter a valid glucose level (50-500 mg/dL)', 'warning');
        }
        
        const bmi = parseFloat(document.getElementById('bmi')?.value);
        if (bmi && (bmi < 10 || bmi > 60)) {
            isValid = false;
            showNotification('Please enter a valid BMI (10-60)', 'warning');
        }
    }
    
    if (!isValid && !document.querySelector('.notification')) {
        showNotification('Please fill in all required fields', 'warning');
    }
    
    return isValid;
}

function highlightError(input) {
    input.style.borderColor = '#ef4444';
    input.style.animation = 'shake 0.5s ease';
    setTimeout(() => {
        input.style.animation = '';
    }, 500);
}

function clearError(input) {
    input.style.borderColor = '';
}

// ============================================================================
// FORM SUBMISSION
// ============================================================================

async function handleSubmit(e) {
    e.preventDefault();
    
    if (!validateCurrentStep()) return;
    if (AppState.isSubmitting) return;
    
    AppState.isSubmitting = true;
    
    // Collect form data
    const formData = collectFormData();
    
    // Show loading state
    showLoadingState();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('An error occurred. Please try again.', 'error');
        hideLoadingState();
    } finally {
        AppState.isSubmitting = false;
    }
}

function collectFormData() {
    return {
        age: parseInt(document.getElementById('age')?.value) || 0,
        gender: document.querySelector('input[name="gender"]:checked')?.value || 'Male',
        hypertension: document.getElementById('hypertension')?.checked ? 1 : 0,
        heart_disease: document.getElementById('heartDisease')?.checked ? 1 : 0,
        ever_married: document.querySelector('input[name="married"]:checked')?.value === 'yes' ? 'Yes' : 'No',
        work_type: document.getElementById('workType')?.value || 'Private',
        residence_type: document.querySelector('input[name="residence"]:checked')?.value || 'Urban',
        avg_glucose_level: parseFloat(document.getElementById('glucose')?.value) || 100,
        bmi: parseFloat(document.getElementById('bmi')?.value) || 25,
        smoking_status: document.querySelector('input[name="smoking"]:checked')?.value || 'never smoked'
    };
}

// ============================================================================
// LOADING STATE
// ============================================================================

function showLoadingState() {
    Elements.formSection?.classList.add('hidden');
    Elements.resultsSection?.classList.remove('hidden');
    Elements.loadingState?.classList.remove('hidden');
    Elements.resultsContent?.classList.add('hidden');
    
    // Scroll to results
    Elements.resultsSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideLoadingState() {
    Elements.loadingState?.classList.add('hidden');
    Elements.formSection?.classList.remove('hidden');
    Elements.resultsSection?.classList.add('hidden');
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayResults(result) {
    Elements.loadingState?.classList.add('hidden');
    Elements.resultsContent?.classList.remove('hidden');
    
    // Store result and patient data for download
    AppState.lastResult = result;
    AppState.lastPatientData = collectFormData();
    
    // Animate risk gauge
    animateRiskGauge(result.risk_percentage);
    
    // Update risk badge
    updateRiskBadge(result.risk_level, result.risk_percentage);
    
    // Display binary prediction if available
    if (result.binary_prediction !== undefined && result.binary_prediction !== null) {
        displayBinaryPrediction(result.binary_prediction);
    }
    
    // Display risk factors
    displayRiskFactors(result.risk_factors);
    
    // Display AI insights
    displayAiInsights(result.ai_insights);
}

function animateRiskGauge(percentage) {
    const gaugePercentage = document.getElementById('riskPercentage');
    const gaugeFill = document.querySelector('.gauge-fill');
    const gaugeNeedle = document.querySelector('.gauge-needle');
    
    // Animate percentage counter
    let current = 0;
    const target = Math.round(percentage);
    const duration = 1500;
    const increment = target / (duration / 16);
    
    const counterAnimation = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(counterAnimation);
        }
        if (gaugePercentage) {
            gaugePercentage.textContent = Math.round(current);
        }
    }, 16);
    
    // Animate gauge fill
    if (gaugeFill) {
        const fillAmount = (percentage / 100) * 251; // 251 is the stroke-dasharray
        gaugeFill.style.strokeDashoffset = 251 - fillAmount;
        
        // Set color based on risk
        if (percentage < 30) {
            gaugeFill.style.stroke = '#22c55e';
        } else if (percentage < 60) {
            gaugeFill.style.stroke = '#eab308';
        } else {
            gaugeFill.style.stroke = '#ef4444';
        }
    }
    
    // Animate needle
    if (gaugeNeedle) {
        const rotation = -90 + (percentage / 100) * 180;
        gaugeNeedle.style.transform = `rotate(${rotation}deg)`;
        gaugeNeedle.setAttribute('transform', `rotate(${rotation} 100 100)`);
    }
}

function updateRiskBadge(riskLevel, percentage) {
    const badge = document.getElementById('riskBadge');
    if (!badge) return;
    
    badge.className = 'risk-level-badge';
    
    const badgeText = badge.querySelector('.badge-text') || badge;
    
    if (riskLevel === 'LOW') {
        badge.classList.add('low');
        badgeText.textContent = '✓ LOW RISK';
    } else if (riskLevel === 'MODERATE') {
        badge.classList.add('moderate');
        badgeText.textContent = '⚠ MODERATE RISK';
    } else {
        badge.classList.add('high');
        badgeText.textContent = '⚠ HIGH RISK';
    }
}

function displayRiskFactors(factors) {
    const container = document.getElementById('riskFactorsList');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (!factors || factors.length === 0) {
        const tag = document.createElement('span');
        tag.className = 'risk-factor-tag positive';
        tag.innerHTML = '<span>✓</span> No significant risk factors identified';
        container.appendChild(tag);
        return;
    }
    
    factors.forEach(factor => {
        const tag = document.createElement('span');
        tag.className = 'risk-factor-tag';
        tag.innerHTML = `<span>!</span> ${factor}`;
        container.appendChild(tag);
    });
}

function displayBinaryPrediction(binaryPred) {
    // Add binary prediction indicator to the risk gauge area
    const gaugeContainer = document.querySelector('.risk-gauge-container');
    if (!gaugeContainer) return;
    
    // Remove existing binary indicator if any
    const existing = gaugeContainer.querySelector('.binary-prediction');
    if (existing) existing.remove();
    
    const binaryDiv = document.createElement('div');
    binaryDiv.className = 'binary-prediction';
    binaryDiv.style.cssText = `
        margin-top: 1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    `;
    
    if (binaryPred === 1) {
        binaryDiv.style.background = 'rgba(239, 68, 68, 0.1)';
        binaryDiv.style.color = '#dc2626';
        binaryDiv.style.border = '2px solid #ef4444';
        binaryDiv.innerHTML = '⚠️ <strong>Stroke Risk Detected</strong> - Immediate medical consultation recommended';
    } else {
        binaryDiv.style.background = 'rgba(34, 197, 94, 0.1)';
        binaryDiv.style.color = '#16a34a';
        binaryDiv.style.border = '2px solid #22c55e';
        binaryDiv.innerHTML = '✓ <strong>No Stroke Risk Detected</strong> - Continue healthy lifestyle';
    }
    
    gaugeContainer.appendChild(binaryDiv);
}

function displayAiInsights(insights) {
    const container = document.getElementById('aiInsights');
    if (!container) return;
    
    if (!insights || insights.trim() === '') {
        container.textContent = 'AI insights are being generated... Please check back in a moment.';
        return;
    }
    
    // Format the insights with better styling
    let formattedInsights = insights
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    
    container.innerHTML = formattedInsights;
}

// ============================================================================
// BMI CALCULATOR
// ============================================================================

function openBmiModal() {
    Elements.bmiModal?.classList.remove('hidden');
}

function closeBmiModal() {
    Elements.bmiModal?.classList.add('hidden');
}

function calculateBmi() {
    const weight = parseFloat(document.getElementById('bmiWeight')?.value);
    const height = parseFloat(document.getElementById('bmiHeight')?.value);
    
    if (!weight || !height || height === 0) {
        showNotification('Please enter valid weight and height', 'warning');
        return;
    }
    
    const heightInMeters = height / 100;
    const bmi = weight / (heightInMeters * heightInMeters);
    const roundedBmi = Math.round(bmi * 10) / 10;
    
    // Determine category
    let category, categoryClass;
    if (bmi < 18.5) {
        category = 'Underweight';
        categoryClass = 'low';
    } else if (bmi < 25) {
        category = 'Normal weight';
        categoryClass = 'low';
    } else if (bmi < 30) {
        category = 'Overweight';
        categoryClass = 'medium';
    } else {
        category = 'Obese';
        categoryClass = 'high';
    }
    
    // Display result
    if (Elements.bmiResult) {
        Elements.bmiResult.innerHTML = `
            <div style="font-size: 2rem; font-weight: bold; color: var(--gray-800);">
                ${roundedBmi}
            </div>
            <div class="bmi-category range-label ${categoryClass}">
                ${category}
            </div>
            <button type="button" class="btn btn-primary" style="margin-top: 1rem;" 
                    onclick="useBmiValue(${roundedBmi})">
                Use this BMI
            </button>
        `;
    }
}

function useBmiValue(bmi) {
    const bmiInput = document.getElementById('bmi');
    if (bmiInput) {
        bmiInput.value = bmi;
        handleBmiInput({ target: bmiInput });
    }
    closeBmiModal();
    showNotification('BMI value applied!', 'success');
}

// ============================================================================
// INPUT HANDLERS
// ============================================================================

function handleAgeInput(e) {
    const age = parseInt(e.target.value);
    const container = e.target.closest('.form-group');
    
    // Remove existing indicator
    const existing = container?.querySelector('.age-indicator');
    existing?.remove();
    
    if (!age) return;
    
    let indicator = document.createElement('div');
    indicator.className = 'age-indicator input-hint';
    
    if (age >= 60) {
        indicator.innerHTML = '<span style="color: var(--danger);">⚠ Age is a significant risk factor</span>';
    } else if (age >= 45) {
        indicator.innerHTML = '<span style="color: var(--warning);">Moderate age-related risk</span>';
    } else {
        indicator.innerHTML = '<span style="color: var(--success);">✓ Lower age-related risk</span>';
    }
    
    container?.appendChild(indicator);
}

function handleBmiInput(e) {
    const bmi = parseFloat(e.target.value);
    const container = e.target.closest('.form-group');
    
    const existing = container?.querySelector('.bmi-indicator');
    existing?.remove();
    
    if (!bmi) return;
    
    let indicator = document.createElement('div');
    indicator.className = 'bmi-indicator input-hint';
    
    if (bmi >= 30) {
        indicator.innerHTML = '<span style="color: var(--danger);">⚠ Obese range - higher risk</span>';
    } else if (bmi >= 25) {
        indicator.innerHTML = '<span style="color: var(--warning);">Overweight range</span>';
    } else if (bmi >= 18.5) {
        indicator.innerHTML = '<span style="color: var(--success);">✓ Normal BMI range</span>';
    } else {
        indicator.innerHTML = '<span style="color: var(--warning);">Underweight range</span>';
    }
    
    container?.appendChild(indicator);
}

function handleGlucoseInput(e) {
    const glucose = parseFloat(e.target.value);
    const container = e.target.closest('.form-group');
    
    const existing = container?.querySelector('.glucose-indicator');
    existing?.remove();
    
    if (!glucose) return;
    
    let indicator = document.createElement('div');
    indicator.className = 'glucose-indicator input-hint';
    
    if (glucose >= 200) {
        indicator.innerHTML = '<span style="color: var(--danger);">⚠ High glucose - consult doctor</span>';
    } else if (glucose >= 140) {
        indicator.innerHTML = '<span style="color: var(--warning);">Elevated glucose level</span>';
    } else {
        indicator.innerHTML = '<span style="color: var(--success);">✓ Normal glucose range</span>';
    }
    
    container?.appendChild(indicator);
}

// ============================================================================
// SUMMARY PREVIEW
// ============================================================================

function updateSummaryPreview() {
    const age = document.getElementById('age')?.value || '--';
    const gender = document.querySelector('input[name="gender"]:checked')?.value || '--';
    const glucose = document.getElementById('glucose')?.value || '--';
    const bmi = document.getElementById('bmi')?.value || '--';
    const hypertension = document.getElementById('hypertension')?.checked ? 'Yes' : 'No';
    const heartDisease = document.getElementById('heartDisease')?.checked ? 'Yes' : 'No';
    
    // Update summary items
    const summaryAge = document.getElementById('summaryAge');
    const summaryGender = document.getElementById('summaryGender');
    const summaryGlucose = document.getElementById('summaryGlucose');
    const summaryBmi = document.getElementById('summaryBmi');
    const summaryHypertension = document.getElementById('summaryHypertension');
    const summaryHeartDisease = document.getElementById('summaryHeartDisease');
    
    if (summaryAge) summaryAge.textContent = age;
    if (summaryGender) summaryGender.textContent = gender;
    if (summaryGlucose) summaryGlucose.textContent = glucose + (glucose !== '--' ? ' mg/dL' : '');
    if (summaryBmi) summaryBmi.textContent = bmi;
    if (summaryHypertension) summaryHypertension.textContent = hypertension;
    if (summaryHeartDisease) summaryHeartDisease.textContent = heartDisease;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function startNewAssessment() {
    // Reset state
    AppState.currentStep = 1;
    AppState.formData = {};
    
    // Reset form
    Elements.form?.reset();
    
    // Update UI
    updateUI();
    
    // Hide results, show form
    Elements.resultsSection?.classList.add('hidden');
    Elements.formSection?.classList.remove('hidden');
    
    // Reset indicators
    document.querySelectorAll('.age-indicator, .bmi-indicator, .glucose-indicator, .binary-prediction').forEach(el => el.remove());
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Alias for compatibility
function startOver() {
    startNewAssessment();
}

function showNotification(message, type = 'info') {
    // Remove existing notification
    const existing = document.querySelector('.notification');
    existing?.remove();
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 400px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    `;
    
    const colors = {
        success: '#22c55e',
        warning: '#eab308',
        error: '#ef4444',
        info: '#3b82f6'
    };
    
    notification.style.backgroundColor = colors[type] || colors.info;
    if (type === 'warning') notification.style.color = '#1f2937';
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Auto remove
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Add CSS animation for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideOut {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100px); }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

// ============================================================================
// KEYBOARD NAVIGATION
// ============================================================================

document.addEventListener('keydown', (e) => {
    // Enter to proceed (when not in textarea)
    if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') {
        if (AppState.currentStep < AppState.totalSteps) {
            e.preventDefault();
            handleNext();
        }
    }
    
    // Escape to close modal
    if (e.key === 'Escape') {
        closeBmiModal();
    }
});

// ============================================================================
// DOWNLOAD REPORT
// ============================================================================

async function downloadReport() {
    // Check if we have stored result data
    if (!AppState.lastResult || !AppState.lastPatientData) {
        showNotification('No assessment data available. Please run an assessment first.', 'warning');
        return;
    }
    
    // Prepare data for report using stored data
    const reportData = {
        patient_data: AppState.lastPatientData,
        result: {
            risk_percentage: AppState.lastResult.risk_percentage || 0,
            risk_level: AppState.lastResult.risk_level || 'N/A',
            binary_prediction: AppState.lastResult.binary_prediction,
            risk_factors: AppState.lastResult.risk_factors || [],
            explanation: AppState.lastResult.explanation || [],
            ai_insights: AppState.lastResult.ai_insights || 'N/A'
        }
    };
    
    try {
        const response = await fetch('/download-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(reportData)
        });
        
        if (response.ok) {
            // Get the blob and create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `stroke_risk_assessment_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showNotification('Report downloaded successfully!', 'success');
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Download failed');
        }
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Failed to download report: ' + error.message, 'error');
    }
}

// ============================================================================
// PRINT RESULTS
// ============================================================================

function printResults() {
    window.print();
}

// ============================================================================
// EXPORT
// ============================================================================

// Make functions available globally for inline handlers
window.useBmiValue = useBmiValue;
window.printResults = printResults;
window.startOver = startOver;
window.startNewAssessment = startNewAssessment;
window.nextStep = nextStep;
window.prevStep = prevStep;
window.openBMICalculator = openBmiModal;
window.closeBMICalculator = closeBmiModal;
window.calculateBMI = calculateBmi;
window.downloadReport = downloadReport;
