## Quick Fix Instructions for 05_model_interpretability.ipynb

The notebook needs three main fixes based on our testing:

### 1. Load Target Encoder (Cell after loading model)

In the cell where you load the model, add this line:
```python
# Load Target Encoder ← ADD THIS LINE
target_encoder = joblib.load(config.MODEL_DIR / "target_encoder.pkl")
```

### 2. Fix SHAP Dependence Plot (Current Cell 7)

Replace the entire cell with:
```python
import numpy as np
import shap
import pandas as pd

# Get SHAP values for Dropout class (class 0)
if isinstance(shap_values, list):
    sv_for_ranking = shap_values[0]
else:
    sv_for_ranking = shap_values

print(f"SHAP values shape: {sv_for_ranking.shape}")
print(f"X_test_sample shape: {X_test_sample.shape}")

# Calculate mean absolute SHAP value for each feature
mean_abs_shap = np.abs(sv_for_ranking).mean(axis=0)

print(f"mean_abs_shap shape: {mean_abs_shap.shape}")
print(f"mean_abs_shap type: {type(mean_abs_shap)}")

# Create a Series for easier handling
feature_importance = pd.Series(mean_abs_shap, index=X_test_sample.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 5 features:")
print(feature_importance.head())

# Get the top feature name
top_feature_name = feature_importance.index[0]
print(f"\nTop feature for Dropout prediction: {top_feature_name}")

# Plot dependence
shap.dependence_plot(top_feature_name, sv_for_ranking, X_test_sample, 
                     interaction_index='auto')

# Save the plot
import matplotlib.pyplot as plt
plt.savefig('../images/shap_dependence.png', dpi=300, bbox_inches='tight')
print("✓ Dependence plot saved!")
```

### 3. Fix Force Plot (Current Cell after "Local Explanation")

Replace the force plot cell with:
```python
# Select a specific student
student_idx = 0
student_data = X_test_sample.iloc[student_idx]

print(f"Explaining prediction for student #{student_idx}")
print(f"Feature values:\n{student_data}")

# Get actual prediction
actual_prediction = model.predict(X_test_sample.iloc[[student_idx]])[0]
prediction_proba = model.predict_proba(X_test_sample.iloc[[student_idx]])[0]

print(f"\n🔮 Model Prediction: {target_encoder.classes_[actual_prediction]}")
print(f"Confidence Scores:")
for i, cls in enumerate(target_encoder.classes_):
    print(f"  {cls}: {prediction_proba[i]:.2%}")

# Generate force plot
import matplotlib.pyplot as plt

sv_local = shap_values[0][student_idx] if isinstance(shap_values, list) else shap_values[student_idx]
base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value

plt.figure(figsize=(16, 3))
shap.plots.force(base_value, sv_local, student_data, matplotlib=True, show=False)
plt.tight_layout()
plt.savefig('../images/shap_force_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Force plot saved to images/shap_force_plot.png")
```

## To apply these fixes:

1. Open the notebook in Jupyter
2. Add the target_encoder loading line to the "Load Data and Model" cell
3. Replace Cell 7 (dependence plot) with the code above
4. Replace the force plot cell with the code above
5. Re-run the notebook from the top

The main issues were:
- Missing target_encoder loading
- `np.argmax()` returning flattened index on multi-dimensional arrays
- Using old SHAP API (`shap.force_plot` instead of `shap.plots.force`)
