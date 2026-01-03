# Presentation Images - Quick Reference

## ✅ Already Available

### 1. System Architecture Diagram
- **Location**: `d:\Santosh_minor\images\system_architecture.png`
- **Status**: ✅ READY
- **Slide**: "System Architecture" (Methodology section)

---

## 📝 Need to Generate from Notebooks

### 2. Confusion Matrix
- **Generate from**: `notebooks/04_model_optimization.ipynb`
- **Save as**: `images/confusion_matrix_rf.png`
- **Slide**: "Model Performance Comparison"
- **Quick Code**:
```python
# Add this at the end of 04_model_optimization.ipynb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=target_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Random Forest (Best Model)', fontsize=14, fontweight='bold')
plt.savefig('../images/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3. Model Comparison Chart
- **Generate from**: `notebooks/03_model_training.ipynb`
- **Save as**: `images/model_comparison.png`
- **Slide**: "Model Performance - Real World Generalization"
- **Quick Code**:
```python
# Add this after model evaluation section in 03_model_training.ipynb
import matplotlib.pyplot as plt

model_names = ['Logistic\nRegression', 'SVM', 'Random\nForest', 'XGBoost', 'Decision\nTree']
cv_scores = [cv_lr.mean(), cv_svm.mean(), cv_rf.mean(), cv_xgb.mean(), cv_dt.mean()]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, cv_scores, 
               color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
plt.ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
plt.ylim(0.65, 0.80)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4. SHAP Summary Plot
- **Generate from**: `notebooks/05_model_interpretability.ipynb`
- **Save as**: `images/shap_summary.png`
- **Slide**: "Explainability - SHAP Analysis (Global)"
- **Quick Code**:
```python
# Modify the SHAP summary plot section in 05_model_interpretability.ipynb
import matplotlib.pyplot as plt
import shap

sv_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(10, 8))
shap.summary_plot(sv_to_plot, X_test_sample, plot_type="dot", show=False)
plt.title('SHAP Feature Importance - Dropout Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5. SHAP Force Plot
- **Generate from**: `notebooks/05_model_interpretability.ipynb`
- **Save as**: `images/shap_force_plot.png`
- **Slide**: "Explainability - SHAP Analysis (Local)"
- **Quick Code** (SHAP v0.20+ compatible):
```python
# Fix for SHAP v0.20+ API changes
import matplotlib.pyplot as plt
import shap

# Select a student for local explanation
student_idx = 0
student_data = X_test_sample.iloc[student_idx]

# Get SHAP values for this student
sv_local = shap_values[0][student_idx] if isinstance(shap_values, list) else shap_values[student_idx]
base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value

# Generate matplotlib-based force plot
plt.figure(figsize=(16, 3))
shap.plots.force(base_value, sv_local, student_data, matplotlib=True, show=False)
plt.tight_layout()
plt.savefig('../images/shap_force_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Force plot saved successfully!")
```

### 6. Streamlit App Screenshot
- **Generate from**: Running application
- **Save as**: `images/streamlit_app.png`
- **Slide**: "Web Application - Streamlit Dashboard"
- **Steps**:
  1. Run `streamlit run app.py`
  2. Navigate to http://localhost:8501
  3. Fill in sample student data (e.g., Age=20, Gender=Male, etc.)
  4. Click "🔮 Predict Outcome"
  5. Take screenshot showing the prediction result with probability bars
  6. Save as `streamlit_app.png`

---

## 📊 Image Status Summary

| # | Image Name | Status | Source Notebook |
|---|------------|--------|-----------------|
| 1 | system_architecture.png | ✅ Ready | AI Generated |
| 2 | confusion_matrix_rf.png | ⏳ To Generate | 04_model_optimization.ipynb |
| 3 | model_comparison.png | ⏳ To Generate | 03_model_training.ipynb |
| 4 | shap_summary.png | ⏳ To Generate | 05_model_interpretability.ipynb |
| 5 | shap_force_plot.png | ⏳ To Generate | 05_model_interpretability.ipynb |
| 6 | streamlit_app.png | ⏳ To Generate | Screenshot of app.py |

---

## 🚀 Quick Generation Script

You can run this in each notebook to generate all visualizations at once:

### For Model Notebooks (03, 04):
```python
import os
os.makedirs('../images', exist_ok=True)
print("Images folder ready!")
```

### Testing Image Paths
Before compiling LaTeX, verify all images exist:
```powershell
Get-ChildItem d:\Santosh_minor\images\*.png | Select-Object Name
```

Expected output should show all 6 PNG files.
