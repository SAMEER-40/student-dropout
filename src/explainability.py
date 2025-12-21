"""
Model explainability utilities using SHAP and LIME
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from typing import Any, List


class ModelExplainer:
    """
    Class for explaining model predictions using SHAP and LIME.
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
    
    def initialize_shap(self, explainer_type: str = "tree"):
        """
        Initialize SHAP explainer.
        
        Args:
            explainer_type: Type of SHAP explainer ('tree', 'kernel', 'linear')
        """
        print(f"Initializing SHAP {explainer_type} explainer...")
        
        if explainer_type == "tree":
            # For tree-based models (Random Forest, XGBoost)
            self.shap_explainer = shap.TreeExplainer(self.model.model)
        elif explainer_type == "kernel":
            # For any model (slower but model-agnostic)
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X_train, 100)
            )
        elif explainer_type == "linear":
            # For linear models
            self.shap_explainer = shap.LinearExplainer(
                self.model.model,
                self.X_train
            )
        
        print("✓ SHAP explainer initialized")
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for given instances.
        
        Args:
            X: Instances to explain
            
        Returns:
            SHAP values
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap() first.")
        
        # Transform data if model uses scaler
        if hasattr(self.model, 'scaler'):
            X_scaled = self.model.scaler.transform(X)
        else:
            X_scaled = X
        
        shap_values = self.shap_explainer.shap_values(X_scaled)
        
        # For binary classification, some explainers return values for both classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use values for positive class
        
        return shap_values
    
    def plot_shap_summary(self, X: np.ndarray, max_display: int = 20):
        """
        Plot SHAP summary plot showing feature importance.
        
        Args:
            X: Data to explain
            max_display: Maximum number of features to display
        """
        shap_values = self.get_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, instance: np.ndarray, instance_index: int = 0):
        """
        Plot SHAP waterfall plot for a single prediction.
        
        Args:
            instance: Single instance to explain
            instance_index: Index for labeling
        """
        shap_values = self.get_shap_values(instance.reshape(1, -1))
        
        # Create explanation object
        if hasattr(self.model, 'scaler'):
            instance_scaled = self.model.scaler.transform(instance.reshape(1, -1))
        else:
            instance_scaled = instance.reshape(1, -1)
        
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.shap_explainer.expected_value if hasattr(self.shap_explainer, 'expected_value') else 0,
            data=instance_scaled[0],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"SHAP Explanation for Instance {instance_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_force(self, instance: np.ndarray, instance_index: int = 0):
        """
        Plot SHAP force plot for a single prediction.
        
        Args:
            instance: Single instance to explain
            instance_index: Index for labeling
        """
        shap_values = self.get_shap_values(instance.reshape(1, -1))
        
        base_value = self.shap_explainer.expected_value if hasattr(self.shap_explainer, 'expected_value') else 0
        
        # For binary classification, handle base value properly
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        shap.force_plot(
            base_value,
            shap_values[0],
            instance,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot for Instance {instance_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def initialize_lime(self, class_names: List[str] = None):
        """
        Initialize LIME explainer.
        
        Args:
            class_names: Names of the classes
        """
        print("Initializing LIME explainer...")
        
        if class_names is None:
            class_names = ['No Dropout', 'Dropout']
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        print("✓ LIME explainer initialized")
    
    def explain_with_lime(self, instance: np.ndarray, num_features: int = 10):
        """
        Explain a single prediction using LIME.
        
        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")
        
        # Transform instance if needed
        if hasattr(self.model, 'scaler'):
            instance_scaled = self.model.scaler.transform(instance.reshape(1, -1))[0]
        else:
            instance_scaled = instance
        
        # Create prediction function for LIME
        def predict_fn(X):
            return self.model.predict_proba(X)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            data_row=instance_scaled,
            predict_fn=predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def plot_lime_explanation(self, instance: np.ndarray, instance_index: int = 0, num_features: int = 10):
        """
        Plot LIME explanation for a single instance.
        
        Args:
            instance: Single instance to explain
            instance_index: Index for labeling
            num_features: Number of features to display
        """
        explanation = self.explain_with_lime(instance, num_features)
        
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for Instance {instance_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_feature_importance_comparison(models: dict, feature_names: List[str], top_n: int = 15):
    """
    Compare feature importance across multiple models.
    
    Args:
        models: Dictionary of {model_name: model} pairs
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    
    if len(models) == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        importance_df = model.get_feature_importance()
        top_features = importance_df.head(top_n)
        
        axes[idx].barh(range(top_n), top_features['Importance'].values, color='steelblue')
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels(top_features['Feature'].values)
        axes[idx].set_xlabel('Importance', fontsize=12)
        axes[idx].set_title(f'{model_name}\nFeature Importance', fontsize=14, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_prediction(model: Any, instance: np.ndarray, feature_names: List[str],
                      actual_label: int = None, instance_id: str = "Unknown"):
    """
    Provide comprehensive explanation for a single prediction.
    
    Args:
        model: Trained model
        instance: Single instance to explain
        feature_names: List of feature names
        actual_label: Actual label (if known)
        instance_id: Identifier for the instance
    """
    # Make prediction
    prediction = model.predict(instance.reshape(1, -1))[0]
    probability = model.predict_proba(instance.reshape(1, -1))[0]
    
    print(f"\n{'='*70}")
    print(f"PREDICTION EXPLANATION FOR INSTANCE: {instance_id}")
    print(f"{'='*70}")
    print(f"Predicted Class: {'Dropout' if prediction == 1 else 'No Dropout'}")
    print(f"Dropout Probability: {probability[1]:.2%}")
    print(f"Retention Probability: {probability[0]:.2%}")
    
    if actual_label is not None:
        actual = "Dropout" if actual_label == 1 else "No Dropout"
        correctness = "✓ Correct" if prediction == actual_label else "✗ Incorrect"
        print(f"Actual Class: {actual} {correctness}")
    
    print(f"{'='*70}\n")
