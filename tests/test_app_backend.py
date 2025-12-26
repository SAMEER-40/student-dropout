import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_model_and_preprocessor, get_feature_schema, get_categorical_options, make_prediction

def test_backend_connection():
    print("1. Loading resources...")
    try:
        model, preprocessor, target_encoder = load_model_and_preprocessor()
        print("   ✓ Model, Preprocessor, and TargetEncoder loaded.")
    except Exception as e:
        print(f"   ❌ Failed to load resources: {e}")
        return

    print("\n2. Getting feature schema...")
    try:
        num_cols, cat_cols, all_cols = get_feature_schema()
        print(f"   ✓ Schema loaded. {len(num_cols)} numerical, {len(cat_cols)} categorical features.")
    except Exception as e:
        print(f"   ❌ Failed to get schema: {e}")
        return

    print("\n3. Inspecting Preprocessor...")
    try:
        for name, transformer, cols in preprocessor.transformers_:
            print(f"   - Transformer '{name}': applied to {len(cols)} columns")
            print(f"     Columns: {cols}")
    except Exception as e:
        print(f"   ❌ Failed to inspect preprocessor: {e}")

    print("\n4. Creating dummy input data...")
    input_data = {}
    
    # Populate categorical with first available option
    for col in cat_cols:
        options = get_categorical_options(col)
        if options:
            input_data[col] = options[0]
        else:
            input_data[col] = "Unknown" # Fallback
            
    # Populate numerical with mean/dummy values
    for col in num_cols:
        input_data[col] = 0.0 # Simple dummy value
        
    input_df = pd.DataFrame([input_data])
    # Ensure column order
    input_df = input_df[all_cols]
    print("   ✓ Dummy input DataFrame created.")
    
    print("\n5. Testing Prediction...")
    try:
        pred_idx, probabilities = make_prediction(model, preprocessor, input_df)
        pred_label = target_encoder.inverse_transform([pred_idx])[0]
        print(f"   ✓ Prediction successful!")
        print(f"   - Predicted Class Index: {pred_idx}")
        print(f"   - Predicted Label: {pred_label}")
        print(f"   - Probabilities: {probabilities}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        # Print input dtypes for debugging
        print(f"   Input dtypes:\n{input_df.dtypes}")
        return

    print("\n✅ TEST PASSED: Backend is correctly connected.")

if __name__ == "__main__":
    test_backend_connection()
